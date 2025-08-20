
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\cell_style.py ===
# Copyright (c) 2010-2024 openpyxl

from array import array

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Float,
    Bool,
    Integer,
    Sequence,
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.utils.indexed_list import IndexedList


from .alignment import Alignment
from .protection import Protection


class ArrayDescriptor:

    def __init__(self, key):
        self.key = key

    def __get__(self, instance, cls):
        return instance[self.key]

    def __set__(self, instance, value):
        instance[self.key] = value


class StyleArray(array):
    """
    Simplified named tuple with an array
    """

    __slots__ = ()
    tagname = 'xf'

    fontId = ArrayDescriptor(0)
    fillId = ArrayDescriptor(1)
    borderId = ArrayDescriptor(2)
    numFmtId = ArrayDescriptor(3)
    protectionId = ArrayDescriptor(4)
    alignmentId = ArrayDescriptor(5)
    pivotButton = ArrayDescriptor(6)
    quotePrefix = ArrayDescriptor(7)
    xfId = ArrayDescriptor(8)


    def __new__(cls, args=[0]*9):
        return array.__new__(cls, 'i', args)


    def __hash__(self):
        return hash(tuple(self))


    def __copy__(self):
        return StyleArray((self))


    def __deepcopy__(self, memo):
        return StyleArray((self))


class CellStyle(Serialisable):

    tagname = "xf"

    numFmtId = Integer()
    fontId = Integer()
    fillId = Integer()
    borderId = Integer()
    xfId = Integer(allow_none=True)
    quotePrefix = Bool(allow_none=True)
    pivotButton = Bool(allow_none=True)
    applyNumberFormat = Bool(allow_none=True)
    applyFont = Bool(allow_none=True)
    applyFill = Bool(allow_none=True)
    applyBorder = Bool(allow_none=True)
    applyAlignment = Bool(allow_none=True)
    applyProtection = Bool(allow_none=True)
    alignment = Typed(expected_type=Alignment, allow_none=True)
    protection = Typed(expected_type=Protection, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('alignment', 'protection')
    __attrs__ = ("numFmtId", "fontId", "fillId", "borderId",
                 "applyAlignment", "applyProtection", "pivotButton", "quotePrefix", "xfId")

    def __init__(self,
                 numFmtId=0,
                 fontId=0,
                 fillId=0,
                 borderId=0,
                 xfId=None,
                 quotePrefix=None,
                 pivotButton=None,
                 applyNumberFormat=None,
                 applyFont=None,
                 applyFill=None,
                 applyBorder=None,
                 applyAlignment=None,
                 applyProtection=None,
                 alignment=None,
                 protection=None,
                 extLst=None,
                ):
        self.numFmtId = numFmtId
        self.fontId = fontId
        self.fillId = fillId
        self.borderId = borderId
        self.xfId = xfId
        self.quotePrefix = quotePrefix
        self.pivotButton = pivotButton
        self.applyNumberFormat = applyNumberFormat
        self.applyFont = applyFont
        self.applyFill = applyFill
        self.applyBorder = applyBorder
        self.alignment = alignment
        self.protection = protection


    def to_array(self):
        """
        Convert to StyleArray
        """
        style = StyleArray()
        for k in ("fontId", "fillId", "borderId", "numFmtId", "pivotButton",
                  "quotePrefix", "xfId"):
            v = getattr(self, k, 0)
            if v is not None:
                setattr(style, k, v)
        return style


    @classmethod
    def from_array(cls, style):
        """
        Convert from StyleArray
        """
        return cls(numFmtId=style.numFmtId, fontId=style.fontId,
                   fillId=style.fillId, borderId=style.borderId, xfId=style.xfId,
                   quotePrefix=style.quotePrefix, pivotButton=style.pivotButton,)


    @property
    def applyProtection(self):
        return self.protection is not None or None


    @property
    def applyAlignment(self):
        return self.alignment is not None or None


class CellStyleList(Serialisable):

    tagname = "cellXfs"

    __attrs__ = ("count",)

    count = Integer(allow_none=True)
    xf = Sequence(expected_type=CellStyle)
    alignment = Sequence(expected_type=Alignment)
    protection = Sequence(expected_type=Protection)

    __elements__ = ('xf',)

    def __init__(self,
                 count=None,
                 xf=(),
                ):
        self.xf = xf


    @property
    def count(self):
        return len(self.xf)


    def __getitem__(self, idx):
        try:
            return self.xf[idx]
        except IndexError:
            print((f"{idx} is out of range"))
        return self.xf[idx]


    def _to_array(self):
        """
        Extract protection and alignments, convert to style array
        """
        self.prots = IndexedList([Protection()])
        self.alignments = IndexedList([Alignment()])
        styles = [] # allow duplicates
        for xf in self.xf:
            style = xf.to_array()
            if xf.alignment is not None:
                style.alignmentId = self.alignments.add(xf.alignment)
            if xf.protection is not None:
                style.protectionId = self.prots.add(xf.protection)
            styles.append(style)
        return IndexedList(styles)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\string.py ===
"""
Module for formatting output data in console (to string).
"""
from __future__ import annotations

from shutil import get_terminal_size
from typing import TYPE_CHECKING

import numpy as np

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas.io.formats.format import DataFrameFormatter


class StringFormatter:
    """Formatter for string representation of a dataframe."""

    def __init__(self, fmt: DataFrameFormatter, line_width: int | None = None) -> None:
        self.fmt = fmt
        self.adj = fmt.adj
        self.frame = fmt.frame
        self.line_width = line_width

    def to_string(self) -> str:
        text = self._get_string_representation()
        if self.fmt.should_show_dimensions:
            text = f"{text}{self.fmt.dimensions_info}"
        return text

    def _get_strcols(self) -> list[list[str]]:
        strcols = self.fmt.get_strcols()
        if self.fmt.is_truncated:
            strcols = self._insert_dot_separators(strcols)
        return strcols

    def _get_string_representation(self) -> str:
        if self.fmt.frame.empty:
            return self._empty_info_line

        strcols = self._get_strcols()

        if self.line_width is None:
            # no need to wrap around just print the whole frame
            return self.adj.adjoin(1, *strcols)

        if self._need_to_wrap_around:
            return self._join_multiline(strcols)

        return self._fit_strcols_to_terminal_width(strcols)

    @property
    def _empty_info_line(self) -> str:
        return (
            f"Empty {type(self.frame).__name__}\n"
            f"Columns: {pprint_thing(self.frame.columns)}\n"
            f"Index: {pprint_thing(self.frame.index)}"
        )

    @property
    def _need_to_wrap_around(self) -> bool:
        return bool(self.fmt.max_cols is None or self.fmt.max_cols > 0)

    def _insert_dot_separators(self, strcols: list[list[str]]) -> list[list[str]]:
        str_index = self.fmt._get_formatted_index(self.fmt.tr_frame)
        index_length = len(str_index)

        if self.fmt.is_truncated_horizontally:
            strcols = self._insert_dot_separator_horizontal(strcols, index_length)

        if self.fmt.is_truncated_vertically:
            strcols = self._insert_dot_separator_vertical(strcols, index_length)

        return strcols

    @property
    def _adjusted_tr_col_num(self) -> int:
        return self.fmt.tr_col_num + 1 if self.fmt.index else self.fmt.tr_col_num

    def _insert_dot_separator_horizontal(
        self, strcols: list[list[str]], index_length: int
    ) -> list[list[str]]:
        strcols.insert(self._adjusted_tr_col_num, [" ..."] * index_length)
        return strcols

    def _insert_dot_separator_vertical(
        self, strcols: list[list[str]], index_length: int
    ) -> list[list[str]]:
        n_header_rows = index_length - len(self.fmt.tr_frame)
        row_num = self.fmt.tr_row_num
        for ix, col in enumerate(strcols):
            cwidth = self.adj.len(col[row_num])

            if self.fmt.is_truncated_horizontally:
                is_dot_col = ix == self._adjusted_tr_col_num
            else:
                is_dot_col = False

            if cwidth > 3 or is_dot_col:
                dots = "..."
            else:
                dots = ".."

            if ix == 0 and self.fmt.index:
                dot_mode = "left"
            elif is_dot_col:
                cwidth = 4
                dot_mode = "right"
            else:
                dot_mode = "right"

            dot_str = self.adj.justify([dots], cwidth, mode=dot_mode)[0]
            col.insert(row_num + n_header_rows, dot_str)
        return strcols

    def _join_multiline(self, strcols_input: Iterable[list[str]]) -> str:
        lwidth = self.line_width
        adjoin_width = 1
        strcols = list(strcols_input)

        if self.fmt.index:
            idx = strcols.pop(0)
            lwidth -= np.array([self.adj.len(x) for x in idx]).max() + adjoin_width

        col_widths = [
            np.array([self.adj.len(x) for x in col]).max() if len(col) > 0 else 0
            for col in strcols
        ]

        assert lwidth is not None
        col_bins = _binify(col_widths, lwidth)
        nbins = len(col_bins)

        str_lst = []
        start = 0
        for i, end in enumerate(col_bins):
            row = strcols[start:end]
            if self.fmt.index:
                row.insert(0, idx)
            if nbins > 1:
                nrows = len(row[-1])
                if end <= len(strcols) and i < nbins - 1:
                    row.append([" \\"] + ["  "] * (nrows - 1))
                else:
                    row.append([" "] * nrows)
            str_lst.append(self.adj.adjoin(adjoin_width, *row))
            start = end
        return "\n\n".join(str_lst)

    def _fit_strcols_to_terminal_width(self, strcols: list[list[str]]) -> str:
        from pandas import Series

        lines = self.adj.adjoin(1, *strcols).split("\n")
        max_len = Series(lines).str.len().max()
        # plus truncate dot col
        width, _ = get_terminal_size()
        dif = max_len - width
        # '+ 1' to avoid too wide repr (GH PR #17023)
        adj_dif = dif + 1
        col_lens = Series([Series(ele).str.len().max() for ele in strcols])
        n_cols = len(col_lens)
        counter = 0
        while adj_dif > 0 and n_cols > 1:
            counter += 1
            mid = round(n_cols / 2)
            mid_ix = col_lens.index[mid]
            col_len = col_lens[mid_ix]
            # adjoin adds one
            adj_dif -= col_len + 1
            col_lens = col_lens.drop(mid_ix)
            n_cols = len(col_lens)

        # subtract index column
        max_cols_fitted = n_cols - self.fmt.index
        # GH-21180. Ensure that we print at least two.
        max_cols_fitted = max(max_cols_fitted, 2)
        self.fmt.max_cols_fitted = max_cols_fitted

        # Call again _truncate to cut frame appropriately
        # and then generate string representation
        self.fmt.truncate()
        strcols = self._get_strcols()
        return self.adj.adjoin(1, *strcols)


def _binify(cols: list[int], line_width: int) -> list[int]:
    adjoin_width = 1
    bins = []
    curr_width = 0
    i_last_column = len(cols) - 1
    for i, w in enumerate(cols):
        w_adjoined = w + adjoin_width
        curr_width += w_adjoined
        if i_last_column == i:
            wrap = curr_width + 1 > line_width and i > 0
        else:
            wrap = curr_width + 2 > line_width and i > 0
        if wrap:
            bins.append(i)
            curr_width = w_adjoined

    bins.append(len(cols))
    return bins

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\dense_ndim_array.py ===
import functools
from typing import List

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray, ArrayKind
from sympy.utilities.iterables import flatten


class DenseNDimArray(NDimArray):

    _array: List[Basic]

    def __new__(self, *args, **kwargs):
        return ImmutableDenseNDimArray(*args, **kwargs)

    @property
    def kind(self) -> ArrayKind:
        return ArrayKind._union(self._array)

    def __getitem__(self, index):
        """
        Allows to get items from N-dim array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([0, 1, 2, 3], (2, 2))
        >>> a
        [[0, 1], [2, 3]]
        >>> a[0, 0]
        0
        >>> a[1, 1]
        3
        >>> a[0]
        [0, 1]
        >>> a[1]
        [2, 3]


        Symbolic index:

        >>> from sympy.abc import i, j
        >>> a[i, j]
        [[0, 1], [2, 3]][i, j]

        Replace `i` and `j` to get element `(1, 1)`:

        >>> a[i, j].subs({i: 1, j: 1})
        3

        """
        syindex = self._check_symbolic_index(index)
        if syindex is not None:
            return syindex

        index = self._check_index_for_getitem(index)

        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            sl_factors, eindices = self._get_slice_data_for_array_access(index)
            array = [self._array[self._parse_index(i)] for i in eindices]
            nshape = [len(el) for i, el in enumerate(sl_factors) if isinstance(index[i], slice)]
            return type(self)(array, nshape)
        else:
            index = self._parse_index(index)
            return self._array[index]

    @classmethod
    def zeros(cls, *shape):
        list_length = functools.reduce(lambda x, y: x*y, shape, S.One)
        return cls._new(([0]*list_length,), shape)

    def tomatrix(self):
        """
        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1 for i in range(9)], (3, 3))
        >>> b = a.tomatrix()
        >>> b
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

        """
        from sympy.matrices import Matrix

        if self.rank() != 2:
            raise ValueError('Dimensions must be of size of 2')

        return Matrix(self.shape[0], self.shape[1], self._array)

    def reshape(self, *newshape):
        """
        Returns MutableDenseNDimArray instance with new shape. Elements number
        must be        suitable to new shape. The only argument of method sets
        new shape.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))
        >>> a.shape
        (2, 3)
        >>> a
        [[1, 2, 3], [4, 5, 6]]
        >>> b = a.reshape(3, 2)
        >>> b.shape
        (3, 2)
        >>> b
        [[1, 2], [3, 4], [5, 6]]

        """
        new_total_size = functools.reduce(lambda x,y: x*y, newshape)
        if new_total_size != self._loop_size:
            raise ValueError('Expecting reshape size to %d but got prod(%s) = %d' % (
                self._loop_size, str(newshape), new_total_size))

        # there is no `.func` as this class does not subtype `Basic`:
        return type(self)(self._array, newshape)


class ImmutableDenseNDimArray(DenseNDimArray, ImmutableNDimArray): # type: ignore
    def __new__(cls, iterable, shape=None, **kwargs):
        return cls._new(iterable, shape, **kwargs)

    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        shape = Tuple(*map(_sympify, shape))
        cls._check_special_bounds(flat_list, shape)
        flat_list = flatten(flat_list)
        flat_list = Tuple(*flat_list)
        self = Basic.__new__(cls, flat_list, shape, **kwargs)
        self._shape = shape
        self._array = list(flat_list)
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x,y: x*y, shape, 1)
        return self

    def __setitem__(self, index, value):
        raise TypeError('immutable N-dim array')

    def as_mutable(self):
        return MutableDenseNDimArray(self)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import simplify
        return self.applyfunc(simplify)

class MutableDenseNDimArray(DenseNDimArray, MutableNDimArray):

    def __new__(cls, iterable=None, shape=None, **kwargs):
        return cls._new(iterable, shape, **kwargs)

    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        flat_list = flatten(flat_list)
        self = object.__new__(cls)
        self._shape = shape
        self._array = list(flat_list)
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x,y: x*y, shape) if shape else len(flat_list)
        return self

    def __setitem__(self, index, value):
        """Allows to set items to MutableDenseNDimArray.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(2,  2)
        >>> a[0,0] = 1
        >>> a[1,1] = 1
        >>> a
        [[1, 0], [0, 1]]

        """
        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            value, eindices, slice_offsets = self._get_slice_data_for_array_assignment(index, value)
            for i in eindices:
                other_i = [ind - j for ind, j in zip(i, slice_offsets) if j is not None]
                self._array[self._parse_index(i)] = value[other_i]
        else:
            index = self._parse_index(index)
            self._setter_iterable_check(value)
            value = _sympify(value)
            self._array[index] = value

    def as_immutable(self):
        return ImmutableDenseNDimArray(self)

    @property
    def free_symbols(self):
        return {i for j in self._array for i in j.free_symbols}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\integrals.py ===
from sympy.core import Basic, diff
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.matrices import Matrix
from sympy.integrals import Integral, integrate
from sympy.geometry.entity import GeometryEntity
from sympy.simplify.simplify import simplify
from sympy.utilities.iterables import topological_sort
from sympy.vector import (CoordSys3D, Vector, ParametricRegion,
                        parametric_region_list, ImplicitRegion)
from sympy.vector.operators import _get_coord_systems


class ParametricIntegral(Basic):
    """
    Represents integral of a scalar or vector field
    over a Parametric Region

    Examples
    ========

    >>> from sympy import cos, sin, pi
    >>> from sympy.vector import CoordSys3D, ParametricRegion, ParametricIntegral
    >>> from sympy.abc import r, t, theta, phi

    >>> C = CoordSys3D('C')
    >>> curve = ParametricRegion((3*t - 2, t + 1), (t, 1, 2))
    >>> ParametricIntegral(C.x, curve)
    5*sqrt(10)/2
    >>> length = ParametricIntegral(1, curve)
    >>> length
    sqrt(10)
    >>> semisphere = ParametricRegion((2*sin(phi)*cos(theta), 2*sin(phi)*sin(theta), 2*cos(phi)),\
                            (theta, 0, 2*pi), (phi, 0, pi/2))
    >>> ParametricIntegral(C.z, semisphere)
    8*pi

    >>> ParametricIntegral(C.j + C.k, ParametricRegion((r*cos(theta), r*sin(theta)), r, theta))
    0

    """

    def __new__(cls, field, parametricregion):

        coord_set = _get_coord_systems(field)

        if len(coord_set) == 0:
            coord_sys = CoordSys3D('C')
        elif len(coord_set) > 1:
            raise ValueError
        else:
            coord_sys = next(iter(coord_set))

        if parametricregion.dimensions == 0:
            return S.Zero

        base_vectors = coord_sys.base_vectors()
        base_scalars = coord_sys.base_scalars()

        parametricfield = field

        r = Vector.zero
        for i in range(len(parametricregion.definition)):
            r += base_vectors[i]*parametricregion.definition[i]

        if len(coord_set) != 0:
            for i in range(len(parametricregion.definition)):
                parametricfield = parametricfield.subs(base_scalars[i], parametricregion.definition[i])

        if parametricregion.dimensions == 1:
            parameter = parametricregion.parameters[0]

            r_diff = diff(r, parameter)
            lower, upper = parametricregion.limits[parameter][0], parametricregion.limits[parameter][1]

            if isinstance(parametricfield, Vector):
                integrand = simplify(r_diff.dot(parametricfield))
            else:
                integrand = simplify(r_diff.magnitude()*parametricfield)

            result = integrate(integrand, (parameter, lower, upper))

        elif parametricregion.dimensions == 2:
            u, v = cls._bounds_case(parametricregion.parameters, parametricregion.limits)

            r_u = diff(r, u)
            r_v = diff(r, v)
            normal_vector = simplify(r_u.cross(r_v))

            if isinstance(parametricfield, Vector):
                integrand = parametricfield.dot(normal_vector)
            else:
                integrand = parametricfield*normal_vector.magnitude()

            integrand = simplify(integrand)

            lower_u, upper_u = parametricregion.limits[u][0], parametricregion.limits[u][1]
            lower_v, upper_v = parametricregion.limits[v][0], parametricregion.limits[v][1]

            result = integrate(integrand, (u, lower_u, upper_u), (v, lower_v, upper_v))

        else:
            variables = cls._bounds_case(parametricregion.parameters, parametricregion.limits)
            coeff = Matrix(parametricregion.definition).jacobian(variables).det()
            integrand = simplify(parametricfield*coeff)

            l = [(var, parametricregion.limits[var][0], parametricregion.limits[var][1]) for var in variables]
            result = integrate(integrand, *l)

        if not isinstance(result, Integral):
            return result
        else:
            return super().__new__(cls, field, parametricregion)

    @classmethod
    def _bounds_case(cls, parameters, limits):

        V = list(limits.keys())
        E = []

        for p in V:
            lower_p = limits[p][0]
            upper_p = limits[p][1]

            lower_p = lower_p.atoms()
            upper_p = upper_p.atoms()
            E.extend((p, q) for q in V if p != q and
                     (lower_p.issuperset({q}) or upper_p.issuperset({q})))

        if not E:
            return parameters
        else:
            return topological_sort((V, E), key=default_sort_key)

    @property
    def field(self):
        return self.args[0]

    @property
    def parametricregion(self):
        return self.args[1]


def vector_integrate(field, *region):
    """
    Compute the integral of a vector/scalar field
    over a a region or a set of parameters.

    Examples
    ========
    >>> from sympy.vector import CoordSys3D, ParametricRegion, vector_integrate
    >>> from sympy.abc import x, y, t
    >>> C = CoordSys3D('C')

    >>> region = ParametricRegion((t, t**2), (t, 1, 5))
    >>> vector_integrate(C.x*C.i, region)
    12

    Integrals over some objects of geometry module can also be calculated.

    >>> from sympy.geometry import Point, Circle, Triangle
    >>> c = Circle(Point(0, 2), 5)
    >>> vector_integrate(C.x**2 + C.y**2, c)
    290*pi
    >>> triangle = Triangle(Point(-2, 3), Point(2, 3), Point(0, 5))
    >>> vector_integrate(3*C.x**2*C.y*C.i + C.j, triangle)
    -8

    Integrals over some simple implicit regions can be computed. But in most cases,
    it takes too long to compute over them. This is due to the expressions of parametric
    representation becoming large.

    >>> from sympy.vector import ImplicitRegion
    >>> c2 = ImplicitRegion((x, y), (x - 2)**2 + (y - 1)**2 - 9)
    >>> vector_integrate(1, c2)
    6*pi

    Integral of fields with respect to base scalars:

    >>> vector_integrate(12*C.y**3, (C.y, 1, 3))
    240
    >>> vector_integrate(C.x**2*C.z, C.x)
    C.x**3*C.z/3
    >>> vector_integrate(C.x*C.i - C.y*C.k, C.x)
    (Integral(C.x, C.x))*C.i + (Integral(-C.y, C.x))*C.k
    >>> _.doit()
    C.x**2/2*C.i + (-C.x*C.y)*C.k

    """
    if len(region) == 1:
        if isinstance(region[0], ParametricRegion):
            return ParametricIntegral(field, region[0])

        if isinstance(region[0], ImplicitRegion):
            region = parametric_region_list(region[0])[0]
            return vector_integrate(field, region)

        if isinstance(region[0], GeometryEntity):
            regions_list = parametric_region_list(region[0])

            result = 0
            for reg in regions_list:
                result += vector_integrate(field, reg)
            return result

    return integrate(field, *region)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\convert_tf_models_to_pytorch.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import glob
import os

import requests

TFMODELS = {
    "bert-base-uncased": (
        "bert",
        "BertConfig",
        "",
        "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    ),
    "bert-base-cased": (
        "bert",
        "BertConfig",
        "",
        "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip",
    ),
    "bert-large-uncased": (
        "bert",
        "BertConfig",
        "",
        "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
    ),
    "albert-base": (
        "albert",
        "AlbertConfig",
        "",
        "https://storage.googleapis.com/albert_models/albert_base_v1.tar.gz",
    ),
    "albert-large": (
        "albert",
        "AlbertConfig",
        "",
        "https://storage.googleapis.com/albert_models/albert_large_v1.tar.gz",
    ),
    "gpt-2-117M": (
        "gpt2",
        "GPT2Config",
        "GPT2Model",
        "https://storage.googleapis.com/gpt-2/models/117M",
    ),
    "gpt-2-124M": (
        "gpt2",
        "GPT2Config",
        "GPT2Model",
        "https://storage.googleapis.com/gpt-2/models/124M",
    ),
}


def download_compressed_file(tf_ckpt_url, ckpt_dir):
    r = requests.get(tf_ckpt_url)
    compressed_file_name = tf_ckpt_url.split("/")[-1]
    compressed_file_dir = os.path.join(ckpt_dir, compressed_file_name)
    with open(compressed_file_dir, "wb") as f:
        f.write(r.content)
    return compressed_file_dir


def get_ckpt_prefix_path(ckpt_dir):
    # get prefix
    sub_folder_dir = None
    for o in os.listdir(ckpt_dir):
        sub_folder_dir = os.path.join(ckpt_dir, o)
        break
    if os.path.isfile(sub_folder_dir):
        sub_folder_dir = ckpt_dir
    unique_file_name = str(glob.glob(sub_folder_dir + "/*data-00000-of-00001"))
    prefix = (unique_file_name.rpartition(".")[0]).split("/")[-1]

    return os.path.join(sub_folder_dir, prefix)


def download_tf_checkpoint(model_name, tf_models_dir="tf_models"):
    import pathlib

    base_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), tf_models_dir)
    ckpt_dir = os.path.join(base_dir, model_name)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    tf_ckpt_url = TFMODELS[model_name][3]

    import re

    if re.search(".zip$", tf_ckpt_url) is not None:
        zip_dir = download_compressed_file(tf_ckpt_url, ckpt_dir)

        # unzip file
        import zipfile

        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(ckpt_dir)
            os.remove(zip_dir)

        return get_ckpt_prefix_path(ckpt_dir)

    elif re.search(".tar.gz$", tf_ckpt_url) is not None:
        tar_dir = download_compressed_file(tf_ckpt_url, ckpt_dir)

        # untar file
        import tarfile

        with tarfile.open(tar_dir, "r") as tar_ref:
            tar_ref.extractall(ckpt_dir)
            os.remove(tar_dir)

        return get_ckpt_prefix_path(ckpt_dir)

    else:
        for filename in [
            "checkpoint",
            "model.ckpt.data-00000-of-00001",
            "model.ckpt.index",
            "model.ckpt.meta",
        ]:
            r = requests.get(tf_ckpt_url + "/" + filename)
            with open(os.path.join(ckpt_dir, filename), "wb") as f:
                f.write(r.content)

        return get_ckpt_prefix_path(ckpt_dir)


def init_pytorch_model(model_name, tf_checkpoint_path):
    config_name = TFMODELS[model_name][1]
    config_module = __import__("transformers", fromlist=[config_name])
    model_config = getattr(config_module, config_name)

    parent_path = tf_checkpoint_path.rpartition("/")[0]
    config_path = glob.glob(parent_path + "/*config.json")
    config = model_config() if len(config_path) == 0 else model_config.from_json_file(str(config_path[0]))

    if not TFMODELS[model_name][2]:
        from transformers import AutoModelForPreTraining

        init_model = AutoModelForPreTraining.from_config(config)
    else:
        model_categroy_name = TFMODELS[model_name][2]
        module = __import__("transformers", fromlist=[model_categroy_name])
        model_categroy = getattr(module, model_categroy_name)
        init_model = model_categroy(config)
    return config, init_model


def convert_tf_checkpoint_to_pytorch(model_name, config, init_model, tf_checkpoint_path, is_tf2):
    load_tf_weight_func_name = "load_tf_weights_in_" + TFMODELS[model_name][0]

    module = __import__("transformers", fromlist=[load_tf_weight_func_name])

    if is_tf2 is False:
        load_tf_weight_func = getattr(module, load_tf_weight_func_name)
    else:
        if TFMODELS[model_name][0] != "bert":
            raise NotImplementedError("Only support tf2 ckeckpoint for Bert model")
        from transformers import convert_bert_original_tf2_checkpoint_to_pytorch

        load_tf_weight_func = convert_bert_original_tf2_checkpoint_to_pytorch.load_tf2_weights_in_bert

    # Expect transformers team will unify the order of signature in the future
    model = (
        load_tf_weight_func(init_model, config, tf_checkpoint_path)
        if is_tf2 is False
        else load_tf_weight_func(init_model, tf_checkpoint_path, config)
    )
    model.eval()
    return model


def tf2pt_pipeline(model_name, is_tf2=False):
    if model_name not in TFMODELS:
        raise NotImplementedError(model_name + " not implemented")
    tf_checkpoint_path = download_tf_checkpoint(model_name)
    config, init_model = init_pytorch_model(model_name, tf_checkpoint_path)
    model = convert_tf_checkpoint_to_pytorch(model_name, config, init_model, tf_checkpoint_path, is_tf2)
    # Could then use the model in Benchmark
    return config, model


def tf2pt_pipeline_test():
    # For test on linux only
    import logging

    import torch

    logger = logging.getLogger("")
    for model_name in TFMODELS:
        config, model = tf2pt_pipeline(model_name)
        assert config.model_type is TFMODELS[model_name][0]

        input = torch.randint(low=0, high=config.vocab_size - 1, size=(4, 128), dtype=torch.long)
        try:
            model(input)
        except RuntimeError as e:
            logger.exception(e)


if __name__ == "__main__":
    tf2pt_pipeline_test()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\dynamo_onnx_helper.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections.abc import Sequence
from logging import getLogger
from typing import Any

import numpy as np
import onnx
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class DynamoOnnxHelper:
    """
    Helper class for processing ONNX models exported by Torch Dynamo.
    """

    def __init__(self, model: onnx.ModelProto):
        self.model = OnnxModel(model)

    def update_edges(self, edge_mapping: dict) -> None:
        """
        Updates the edges in the model according to the given mapping.
        """
        for node in self.model.model.graph.node:
            for i in range(len(node.input)):
                if node.input[i] in edge_mapping:
                    node.input[i] = edge_mapping[node.input[i]]
            for i in range(len(node.output)):
                if node.output[i] in edge_mapping:
                    node.output[i] = edge_mapping[node.output[i]]

        for graph_input in self.model.model.graph.input:
            if graph_input.name in edge_mapping:
                graph_input.name = edge_mapping[graph_input.name]
        for graph_output in self.model.model.graph.output:
            if graph_output.name in edge_mapping:
                graph_output.name = edge_mapping[graph_output.name]

    def unroll_function(self, func_name: str) -> None:
        """
        Unrolls the function with the given name in the model.
        """
        logger.debug(f"Unrolling function {func_name}...")
        nodes_to_remove = []
        nodes_to_add = []
        edges_to_remove = []
        edges_to_add = []
        for node in self.model.model.graph.node:
            if node.op_type == func_name:
                nodes_to_remove.append(node)
                edges_to_remove.extend(list(node.input) + list(node.output))

        func_to_remove = None
        for f in self.model.model.functions:
            if f.name == func_name:
                nodes_to_add.extend(list(f.node))
                edges_to_add.extend(list(f.input) + list(f.output))
                func_to_remove = f

        assert len(edges_to_remove) == len(edges_to_add)

        for node in nodes_to_remove:
            self.model.model.graph.node.remove(node)
        for node in nodes_to_add:
            self.model.model.graph.node.append(node)
        if func_to_remove is not None:
            self.model.model.functions.remove(func_to_remove)

        edge_mapping = {}
        for i in range(len(edges_to_remove)):
            k = edges_to_remove[i]
            v = edges_to_add[i]
            if k != v:
                edge_mapping[k] = v

        return self.update_edges(edge_mapping)

    def remove_function(self, func_name: str, input_id: int, output_id: int) -> None:
        """
        Removes the function in the model.
        """
        edge_mapping = {}
        nodes_to_remove = []
        for node in self.model.model.graph.node:
            if node.op_type.find(func_name) != -1:
                edge_mapping[node.input[input_id]] = node.output[output_id]
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.model.model.graph.node.remove(node)

        self.update_edges(edge_mapping)

    def remove_dropout_layer(self) -> None:
        """
        Removes the dropout layer in the model.
        """
        logger.debug("Removing dropout layer...")
        self.remove_function("Dropout", 0, 0)

    def remove_lm_head_layer(self) -> None:
        """
        Removes the LM head layer in the model.
        """
        logger.debug("Removing LM head layer...")
        # bugbug: need to copy the right vi over
        self.remove_function("Linear_lm_head", 2, 0)

    def add_initializer(self, name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool = True):
        if raw:
            np_type = helper.tensor_dtype_to_np_dtype(data_type)
            if not isinstance(vals, np.ndarray):
                bytes = np.array(vals, dtype=np_type).tobytes()
            else:
                bytes = vals.astype(np_type).tobytes()
            tensor = helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=bytes,
                raw=True,
            )
        else:
            tensor = helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=vals,
                raw=False,
            )

        self.model.add_initializer(tensor)
        return tensor

    def convert_constants_to_initializers(self, min_size: int = 1) -> None:
        """
        Converts Constant ops of size [min_size] or higher to initializers
        """
        logger.debug(f"Converting constants greater than size {min_size} to initializers")

        constant_nodes = self.model.get_nodes_by_op_type("Constant")
        nodes_to_remove = []

        for node in constant_nodes:
            # Get info from Constant op
            np_data = self.model.get_constant_value(node.output[0])

            # Skip if there are less than [min_size] elements
            if np_data is None or np_data.size < min_size:
                continue

            # Add new initializer with same name as Constant op's output
            for att in node.attribute:
                if att.name == "value":
                    self.add_initializer(
                        name=node.output[0],
                        data_type=att.t.data_type,
                        dims=list(np_data.shape),
                        vals=np_data,
                    )
                    break

            nodes_to_remove.append(node)

        # Remove Constant ops from graph
        self.model.remove_nodes(nodes_to_remove)

    def clear_metadata(self) -> None:
        """
        Clear metadata fields in all nodes
        """
        for graph in self.model.graphs():
            graph.ClearField("metadata_props")
        for node in self.model.nodes():
            node.ClearField("metadata_props")

    @staticmethod
    def fold_transpose_initializers(model) -> None:
        """
        Constant fold Transpose initializers without changing the initializer names
        """
        from onnxscript import ir

        for name, initializer in model.graph.initializers.items():
            user_nodes = initializer.consumers()
            if len(user_nodes) == 1 and user_nodes[0].op_type == "Transpose":
                transpose_node = user_nodes[0]
                perm = transpose_node.attributes.get("perm")
                if perm is None:
                    transposed_tensor = ir.tensor(initializer.const_value.numpy().transpose())
                else:
                    transposed_tensor = ir.tensor(initializer.const_value.numpy().transpose(perm.as_ints()))
                new_initializer = ir.Value(
                    name=initializer.name,
                    shape=transposed_tensor.shape,
                    type=ir.TensorType(transposed_tensor.dtype),
                    const_value=transposed_tensor,
                )
                ir.convenience.replace_all_uses_with(transpose_node.outputs[0], new_initializer)
                model.graph.initializers[name] = new_initializer
                transpose_node.graph.remove(transpose_node, safe=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\actions\pointer_actions.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import Optional

from selenium.webdriver.remote.webelement import WebElement

from . import interaction
from .interaction import Interaction
from .mouse_button import MouseButton
from .pointer_input import PointerInput


class PointerActions(Interaction):
    def __init__(self, source: Optional[PointerInput] = None, duration: int = 250):
        """
        Args:
        - source: PointerInput instance
        - duration: override the default 250 msecs of DEFAULT_MOVE_DURATION in source
        """
        if source is None:
            source = PointerInput(interaction.POINTER_MOUSE, "mouse")
        self.source = source
        self._duration = duration
        super().__init__(source)

    def pointer_down(
        self,
        button=MouseButton.LEFT,
        width=None,
        height=None,
        pressure=None,
        tangential_pressure=None,
        tilt_x=None,
        tilt_y=None,
        twist=None,
        altitude_angle=None,
        azimuth_angle=None,
    ):
        self._button_action(
            "create_pointer_down",
            button=button,
            width=width,
            height=height,
            pressure=pressure,
            tangential_pressure=tangential_pressure,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            twist=twist,
            altitude_angle=altitude_angle,
            azimuth_angle=azimuth_angle,
        )
        return self

    def pointer_up(self, button=MouseButton.LEFT):
        self._button_action("create_pointer_up", button=button)
        return self

    def move_to(
        self,
        element,
        x=0,
        y=0,
        width=None,
        height=None,
        pressure=None,
        tangential_pressure=None,
        tilt_x=None,
        tilt_y=None,
        twist=None,
        altitude_angle=None,
        azimuth_angle=None,
    ):
        if not isinstance(element, WebElement):
            raise AttributeError("move_to requires a WebElement")

        self.source.create_pointer_move(
            origin=element,
            duration=self._duration,
            x=int(x),
            y=int(y),
            width=width,
            height=height,
            pressure=pressure,
            tangential_pressure=tangential_pressure,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            twist=twist,
            altitude_angle=altitude_angle,
            azimuth_angle=azimuth_angle,
        )
        return self

    def move_by(
        self,
        x,
        y,
        width=None,
        height=None,
        pressure=None,
        tangential_pressure=None,
        tilt_x=None,
        tilt_y=None,
        twist=None,
        altitude_angle=None,
        azimuth_angle=None,
    ):
        self.source.create_pointer_move(
            origin=interaction.POINTER,
            duration=self._duration,
            x=int(x),
            y=int(y),
            width=width,
            height=height,
            pressure=pressure,
            tangential_pressure=tangential_pressure,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            twist=twist,
            altitude_angle=altitude_angle,
            azimuth_angle=azimuth_angle,
        )
        return self

    def move_to_location(
        self,
        x,
        y,
        width=None,
        height=None,
        pressure=None,
        tangential_pressure=None,
        tilt_x=None,
        tilt_y=None,
        twist=None,
        altitude_angle=None,
        azimuth_angle=None,
    ):
        self.source.create_pointer_move(
            origin="viewport",
            duration=self._duration,
            x=int(x),
            y=int(y),
            width=width,
            height=height,
            pressure=pressure,
            tangential_pressure=tangential_pressure,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            twist=twist,
            altitude_angle=altitude_angle,
            azimuth_angle=azimuth_angle,
        )
        return self

    def click(self, element: Optional[WebElement] = None, button=MouseButton.LEFT):
        if element:
            self.move_to(element)
        self.pointer_down(button)
        self.pointer_up(button)
        return self

    def context_click(self, element: Optional[WebElement] = None):
        return self.click(element=element, button=MouseButton.RIGHT)

    def click_and_hold(self, element: Optional[WebElement] = None, button=MouseButton.LEFT):
        if element:
            self.move_to(element)
        self.pointer_down(button=button)
        return self

    def release(self, button=MouseButton.LEFT):
        self.pointer_up(button=button)
        return self

    def double_click(self, element: Optional[WebElement] = None):
        if element:
            self.move_to(element)
        self.pointer_down(MouseButton.LEFT)
        self.pointer_up(MouseButton.LEFT)
        self.pointer_down(MouseButton.LEFT)
        self.pointer_up(MouseButton.LEFT)
        return self

    def pause(self, duration: float = 0):
        self.source.create_pause(duration)
        return self

    def _button_action(self, action, **kwargs):
        meth = getattr(self.source, action)
        meth(**kwargs)
        return self

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\applyfunc.py ===
from sympy.core.expr import ExprBuilder
from sympy.core.function import (Function, FunctionClass, Lambda)
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.matrices.matrixbase import MatrixBase


class ElementwiseApplyFunction(MatrixExpr):
    r"""
    Apply function to a matrix elementwise without evaluating.

    Examples
    ========

    It can be created by calling ``.applyfunc(<function>)`` on a matrix
    expression:

    >>> from sympy import MatrixSymbol
    >>> from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    >>> from sympy import exp
    >>> X = MatrixSymbol("X", 3, 3)
    >>> X.applyfunc(exp)
    Lambda(_d, exp(_d)).(X)

    Otherwise using the class constructor:

    >>> from sympy import eye
    >>> expr = ElementwiseApplyFunction(exp, eye(3))
    >>> expr
    Lambda(_d, exp(_d)).(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> expr.doit()
    Matrix([
    [E, 1, 1],
    [1, E, 1],
    [1, 1, E]])

    Notice the difference with the real mathematical functions:

    >>> exp(eye(3))
    Matrix([
    [E, 0, 0],
    [0, E, 0],
    [0, 0, E]])
    """

    def __new__(cls, function, expr):
        expr = _sympify(expr)
        if not expr.is_Matrix:
            raise ValueError("{} must be a matrix instance.".format(expr))

        if expr.shape == (1, 1):
            # Check if the function returns a matrix, in that case, just apply
            # the function instead of creating an ElementwiseApplyFunc object:
            ret = function(expr)
            if isinstance(ret, MatrixExpr):
                return ret

        if not isinstance(function, (FunctionClass, Lambda)):
            d = Dummy('d')
            function = Lambda(d, function(d))

        function = sympify(function)
        if not isinstance(function, (FunctionClass, Lambda)):
            raise ValueError(
                "{} should be compatible with SymPy function classes."
                .format(function))

        if 1 not in function.nargs:
            raise ValueError(
                '{} should be able to accept 1 arguments.'.format(function))

        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))

        obj = MatrixExpr.__new__(cls, function, expr)
        return obj

    @property
    def function(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    @property
    def shape(self):
        return self.expr.shape

    def doit(self, **hints):
        deep = hints.get("deep", True)
        expr = self.expr
        if deep:
            expr = expr.doit(**hints)
        function = self.function
        if isinstance(function, Lambda) and function.is_identity:
            # This is a Lambda containing the identity function.
            return expr
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(self.function)
        elif isinstance(expr, ElementwiseApplyFunction):
            return ElementwiseApplyFunction(
                lambda x: self.function(expr.function(x)),
                expr.expr
            ).doit(**hints)
        else:
            return self

    def _entry(self, i, j, **kwargs):
        return self.function(self.expr._entry(i, j, **kwargs))

    def _get_function_fdiff(self):
        d = Dummy("d")
        function = self.function(d)
        fdiff = function.diff(d)
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        return fdiff

    def _eval_derivative(self, x):
        from sympy.matrices.expressions.hadamard import hadamard_product
        dexpr = self.expr.diff(x)
        fdiff = self._get_function_fdiff()
        return hadamard_product(
            dexpr,
            ElementwiseApplyFunction(fdiff, self.expr)
        )

    def _eval_derivative_matrix_lines(self, x):
        from sympy.matrices.expressions.special import Identity
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct

        fdiff = self._get_function_fdiff()
        lr = self.expr._eval_derivative_matrix_lines(x)
        ewdiff = ElementwiseApplyFunction(fdiff, self.expr)
        if 1 in x.shape:
            # Vector:
            iscolumn = self.shape[1] == 1
            for i in lr:
                if iscolumn:
                    ptr1 = i.first_pointer
                    ptr2 = Identity(self.shape[1])
                else:
                    ptr1 = Identity(self.shape[0])
                    ptr2 = i.second_pointer

                subexpr = ExprBuilder(
                    ArrayDiagonal,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                ewdiff,
                                ptr1,
                                ptr2,
                            ]
                        ),
                        (0, 2) if iscolumn else (1, 4)
                    ],
                    validator=ArrayDiagonal._validate
                )
                i._lines = [subexpr]
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 2
        else:
            # Matrix case:
            for i in lr:
                ptr1 = i.first_pointer
                ptr2 = i.second_pointer
                newptr1 = Identity(ptr1.shape[1])
                newptr2 = Identity(ptr2.shape[1])
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [ptr1, newptr1, ewdiff, ptr2, newptr2]
                        ),
                        (1, 2, 4),
                        (5, 7, 8),
                    ],
                    validator=ArrayContraction._validate
                )
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                i._lines = [subexpr]
        return lr

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import Transpose
        return self.func(self.function, Transpose(self.expr).doit())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\latex\__init__.py ===
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from re import compile as rcompile

from sympy.parsing.latex.lark import LarkLaTeXParser, TransformToSymPyExpr, parse_latex_lark # noqa

from .errors import LaTeXParsingError  # noqa


IGNORE_L = r"\s*[{]*\s*"
IGNORE_R = r"\s*[}]*\s*"
NO_LEFT = r"(?<!\\left)"
BEGIN_AMS_MAT = r"\\begin{matrix}"
END_AMS_MAT = r"\\end{matrix}"
BEGIN_ARR = r"\\begin{array}{.*?}"
END_ARR = r"\\end{array}"

# begin_delim_regex: end_delim_regex
MATRIX_DELIMS = {fr"\\left\({IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\)",
                 fr"{NO_LEFT}\({IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\)",
                 fr"\\left\[{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\]",
                 fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\]",
                 fr"\\left\|{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\|",
                 fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\|",
                 r"\\begin{pmatrix}": r"\\end{pmatrix}",
                 r"\\begin{bmatrix}": r"\\end{bmatrix}",
                 r"\\begin{vmatrix}": r"\\end{vmatrix}",
                 fr"\\left\({IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\)",
                 fr"{NO_LEFT}\({IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\)",
                 fr"\\left\[{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\]",
                 fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\]",
                 fr"\\left\|{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\|",
                 fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\|"
                 }

MATRIX_DELIMS_INV = {v: k for k, v in MATRIX_DELIMS.items()}

# begin_delim_regex: ideal_begin_delim_representative
BEGIN_DELIM_REPR = {fr"\\left\({IGNORE_L}{BEGIN_AMS_MAT}": "\\left(\\begin{matrix}",
                    fr"{NO_LEFT}\({IGNORE_L}{BEGIN_AMS_MAT}": "(\\begin{matrix}",
                    fr"\\left\[{IGNORE_L}{BEGIN_AMS_MAT}": "\\left[\\begin{matrix}",
                    fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_AMS_MAT}": "[\\begin{matrix}",
                    fr"\\left\|{IGNORE_L}{BEGIN_AMS_MAT}": "\\left|\\begin{matrix}",
                    fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_AMS_MAT}": "|\\begin{matrix}",
                    r"\\begin{pmatrix}": "\\begin{pmatrix}",
                    r"\\begin{bmatrix}": "\\begin{bmatrix}",
                    r"\\begin{vmatrix}": "\\begin{vmatrix}",
                    fr"\\left\({IGNORE_L}{BEGIN_ARR}": "\\left(\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\({IGNORE_L}{BEGIN_ARR}": "(\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"\\left\[{IGNORE_L}{BEGIN_ARR}": "\\left[\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_ARR}": "[\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"\\left\|{IGNORE_L}{BEGIN_ARR}": "\\left|\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_ARR}": "|\\begin{array}{COLUMN_SPECIFIERS}"
                    }

# end_delim_regex: ideal_end_delim_representative
END_DELIM_REPR = {fr"{END_AMS_MAT}{IGNORE_R}\\right\)": "\\end{matrix}\\right)",
                  fr"{END_AMS_MAT}{IGNORE_R}\)": "\\end{matrix})",
                  fr"{END_AMS_MAT}{IGNORE_R}\\right\]": "\\end{matrix}\\right]",
                  fr"{END_AMS_MAT}{IGNORE_R}\]": "\\end{matrix}]",
                  fr"{END_AMS_MAT}{IGNORE_R}\\right\|": "\\end{matrix}\\right|",
                  fr"{END_AMS_MAT}{IGNORE_R}\|": "\\end{matrix}|",
                  r"\\end{pmatrix}": "\\end{pmatrix}",
                  r"\\end{bmatrix}": "\\end{bmatrix}",
                  r"\\end{vmatrix}": "\\end{vmatrix}",
                  fr"{END_ARR}{IGNORE_R}\\right\)": "\\end{array}\\right)",
                  fr"{END_ARR}{IGNORE_R}\)": "\\end{array})",
                  fr"{END_ARR}{IGNORE_R}\\right\]": "\\end{array}\\right]",
                  fr"{END_ARR}{IGNORE_R}\]": "\\end{array}]",
                  fr"{END_ARR}{IGNORE_R}\\right\|": "\\end{array}\\right|",
                  fr"{END_ARR}{IGNORE_R}\|": "\\end{array}|"
                  }


def check_matrix_delimiters(latex_str):
    """Report mismatched, excess, or missing matrix delimiters."""
    spans = []
    for begin_delim in MATRIX_DELIMS:
        end_delim = MATRIX_DELIMS[begin_delim]

        p = rcompile(begin_delim)
        q = rcompile(end_delim)

        spans.extend([(*m.span(), m.group(),
                       begin_delim) for m in p.finditer(latex_str)])
        spans.extend([(*m.span(), m.group(),
                       end_delim) for m in q.finditer(latex_str)])

    spans.sort(key=(lambda x: x[0]))
    if len(spans) % 2 == 1:
        # Odd number of delimiters; therefore something
        # is wrong. We do not complain yet; let's see if
        # we can pinpoint the actual error.
        spans.append((None, None, None, None))

    spans = [(*x, *y) for (x, y) in zip(spans[::2], spans[1::2])]
    for x in spans:
        # x is supposed to be an 8-tuple of the following form:
        #
        # (begin_delim_span_start, begin_delim_span_end,
        # begin_delim_match, begin_delim_regex,
        # end_delim_span_start, end_delim_span_end,
        # end_delim_match, end_delim_regex)

        sellipsis = "..."
        s = x[0] - 10
        if s < 0:
            s = 0
            sellipsis = ""

        eellipsis = "..."
        e = x[1] + 10
        if e > len(latex_str):
            e = len(latex_str)
            eellipsis = ""

        if x[3] in END_DELIM_REPR:
            err = (f"Extra '{x[2]}' at index {x[0]} or "
                   "missing corresponding "
                   f"'{BEGIN_DELIM_REPR[MATRIX_DELIMS_INV[x[3]]]}' "
                   f"in LaTeX string: {sellipsis}{latex_str[s:e]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

        if x[7] is None:
            err = (f"Extra '{x[2]}' at index {x[0]} or "
                   "missing corresponding "
                   f"'{END_DELIM_REPR[MATRIX_DELIMS[x[3]]]}' "
                   f"in LaTeX string: {sellipsis}{latex_str[s:e]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

        correct_end_regex = MATRIX_DELIMS[x[3]]
        sellipsis = "..." if x[0] > 0 else ""
        eellipsis = "..." if x[5] < len(latex_str) else ""
        if x[7] != correct_end_regex:
            err = ("Expected "
                   f"'{END_DELIM_REPR[correct_end_regex]}' "
                   f"to close the '{x[2]}' at index {x[0]} but "
                   f"found '{x[6]}' at index {x[4]} of LaTeX "
                   f"string instead: {sellipsis}{latex_str[x[0]:x[5]]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

__doctest_requires__ = {('parse_latex',): ['antlr4', 'lark']}


@doctest_depends_on(modules=('antlr4', 'lark'))
def parse_latex(s, strict=False, backend="antlr"):
    r"""Converts the input LaTeX string ``s`` to a SymPy ``Expr``.

    Parameters
    ==========

    s : str
        The LaTeX string to parse. In Python source containing LaTeX,
        *raw strings* (denoted with ``r"``, like this one) are preferred,
        as LaTeX makes liberal use of the ``\`` character, which would
        trigger escaping in normal Python strings.
    backend : str, optional
        Currently, there are two backends supported: ANTLR, and Lark.
        The default setting is to use the ANTLR backend, which can be
        changed to Lark if preferred.

        Use ``backend="antlr"`` for the ANTLR-based parser, and
        ``backend="lark"`` for the Lark-based parser.

        The ``backend`` option is case-sensitive, and must be in
        all lowercase.
    strict : bool, optional
        This option is only available with the ANTLR backend.

        If True, raise an exception if the string cannot be parsed as
        valid LaTeX. If False, try to recover gracefully from common
        mistakes.

    Examples
    ========

    >>> from sympy.parsing.latex import parse_latex
    >>> expr = parse_latex(r"\frac {1 + \sqrt {\a}} {\b}")
    >>> expr
    (sqrt(a) + 1)/b
    >>> expr.evalf(4, subs=dict(a=5, b=2))
    1.618
    >>> func = parse_latex(r"\int_1^\alpha \dfrac{\mathrm{d}t}{t}", backend="lark")
    >>> func.evalf(subs={"alpha": 2})
    0.693147180559945
    """

    check_matrix_delimiters(s)

    if backend == "antlr":
        _latex = import_module(
            'sympy.parsing.latex._parse_latex_antlr',
            import_kwargs={'fromlist': ['X']})

        if _latex is not None:
            return _latex.parse_latex(s, strict)
    elif backend == "lark":
        return parse_latex_lark(s)
    else:
        raise NotImplementedError(f"Using the '{backend}' backend in the LaTeX" \
                                   " parser is not supported.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\unitsystem.py ===
"""
Unit system for physical quantities; include definition of constants.
"""
from __future__ import annotations

from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.quantities import Quantity

from .dimensions import Dimension


class UnitSystem(_QuantityMapper):
    """
    UnitSystem represents a coherent set of units.

    A unit system is basically a dimension system with notions of scales. Many
    of the methods are defined in the same way.

    It is much better if all base units have a symbol.
    """

    _unit_systems: dict[str, UnitSystem] = {}

    def __init__(self, base_units, units=(), name="", descr="", dimension_system=None, derived_units: dict[Dimension, Quantity]={}):

        UnitSystem._unit_systems[name] = self

        self.name = name
        self.descr = descr

        self._base_units = base_units
        self._dimension_system = dimension_system
        self._units = tuple(set(base_units) | set(units))
        self._base_units = tuple(base_units)
        self._derived_units = derived_units

        super().__init__()

    def __str__(self):
        """
        Return the name of the system.

        If it does not exist, then it makes a list of symbols (or names) of
        the base dimensions.
        """

        if self.name != "":
            return self.name
        else:
            return "UnitSystem((%s))" % ", ".join(
                str(d) for d in self._base_units)

    def __repr__(self):
        return '<UnitSystem: %s>' % repr(self._base_units)

    def extend(self, base, units=(), name="", description="", dimension_system=None, derived_units: dict[Dimension, Quantity]={}):
        """Extend the current system into a new one.

        Take the base and normal units of the current system to merge
        them to the base and normal units given in argument.
        If not provided, name and description are overridden by empty strings.
        """

        base = self._base_units + tuple(base)
        units = self._units + tuple(units)

        return UnitSystem(base, units, name, description, dimension_system, {**self._derived_units, **derived_units})

    def get_dimension_system(self):
        return self._dimension_system

    def get_quantity_dimension(self, unit):
        qdm = self.get_dimension_system()._quantity_dimension_map
        if unit in qdm:
            return qdm[unit]
        return super().get_quantity_dimension(unit)

    def get_quantity_scale_factor(self, unit):
        qsfm = self.get_dimension_system()._quantity_scale_factors
        if unit in qsfm:
            return qsfm[unit]
        return super().get_quantity_scale_factor(unit)

    @staticmethod
    def get_unit_system(unit_system):
        if isinstance(unit_system, UnitSystem):
            return unit_system

        if unit_system not in UnitSystem._unit_systems:
            raise ValueError(
                "Unit system is not supported. Currently"
                "supported unit systems are {}".format(
                    ", ".join(sorted(UnitSystem._unit_systems))
                )
            )

        return UnitSystem._unit_systems[unit_system]

    @staticmethod
    def get_default_unit_system():
        return UnitSystem._unit_systems["SI"]

    @property
    def dim(self):
        """
        Give the dimension of the system.

        That is return the number of units forming the basis.
        """
        return len(self._base_units)

    @property
    def is_consistent(self):
        """
        Check if the underlying dimension system is consistent.
        """
        # test is performed in DimensionSystem
        return self.get_dimension_system().is_consistent

    @property
    def derived_units(self) -> dict[Dimension, Quantity]:
        return self._derived_units

    def get_dimensional_expr(self, expr):
        from sympy.physics.units import Quantity
        if isinstance(expr, Mul):
            return Mul(*[self.get_dimensional_expr(i) for i in expr.args])
        elif isinstance(expr, Pow):
            return self.get_dimensional_expr(expr.base) ** expr.exp
        elif isinstance(expr, Add):
            return self.get_dimensional_expr(expr.args[0])
        elif isinstance(expr, Derivative):
            dim = self.get_dimensional_expr(expr.expr)
            for independent, count in expr.variable_count:
                dim /= self.get_dimensional_expr(independent)**count
            return dim
        elif isinstance(expr, Function):
            args = [self.get_dimensional_expr(arg) for arg in expr.args]
            if all(i == 1 for i in args):
                return S.One
            return expr.func(*args)
        elif isinstance(expr, Quantity):
            return self.get_quantity_dimension(expr).name
        return S.One

    def _collect_factor_and_dimension(self, expr):
        """
        Return tuple with scale factor expression and dimension expression.
        """
        from sympy.physics.units import Quantity
        if isinstance(expr, Quantity):
            return expr.scale_factor, expr.dimension
        elif isinstance(expr, Mul):
            factor = 1
            dimension = Dimension(1)
            for arg in expr.args:
                arg_factor, arg_dim = self._collect_factor_and_dimension(arg)
                factor *= arg_factor
                dimension *= arg_dim
            return factor, dimension
        elif isinstance(expr, Pow):
            factor, dim = self._collect_factor_and_dimension(expr.base)
            exp_factor, exp_dim = self._collect_factor_and_dimension(expr.exp)
            if self.get_dimension_system().is_dimensionless(exp_dim):
                exp_dim = 1
            return factor ** exp_factor, dim ** (exp_factor * exp_dim)
        elif isinstance(expr, Add):
            factor, dim = self._collect_factor_and_dimension(expr.args[0])
            for addend in expr.args[1:]:
                addend_factor, addend_dim = \
                    self._collect_factor_and_dimension(addend)
                if not self.get_dimension_system().equivalent_dims(dim, addend_dim):
                    raise ValueError(
                        'Dimension of "{}" is {}, '
                        'but it should be {}'.format(
                            addend, addend_dim, dim))
                factor += addend_factor
            return factor, dim
        elif isinstance(expr, Derivative):
            factor, dim = self._collect_factor_and_dimension(expr.args[0])
            for independent, count in expr.variable_count:
                ifactor, idim = self._collect_factor_and_dimension(independent)
                factor /= ifactor**count
                dim /= idim**count
            return factor, dim
        elif isinstance(expr, Function):
            fds = [self._collect_factor_and_dimension(arg) for arg in expr.args]
            dims = [Dimension(1) if self.get_dimension_system().is_dimensionless(d[1]) else d[1] for d in fds]
            return (expr.func(*(f[0] for f in fds)), *dims)
        elif isinstance(expr, Dimension):
            return S.One, expr
        else:
            return expr, Dimension(1)

    def get_units_non_prefixed(self) -> set[Quantity]:
        """
        Return the units of the system that do not have a prefix.
        """
        return set(filter(lambda u: not u.is_prefixed and not u.is_physical_constant, self._units))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_generated_io_windows.py ===
# ***********************************************************
# ******* WARNING: AUTOGENERATED! ALL EDITS WILL BE LOST ******
# *************************************************************
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ._ki import enable_ki_protection
from ._run import GLOBAL_RUN_CONTEXT

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from typing_extensions import Buffer

    from .._file_io import _HasFileNo
    from ._unbounded_queue import UnboundedQueue
    from ._windows_cffi import CData, Handle

assert not TYPE_CHECKING or sys.platform == "win32"


__all__ = [
    "current_iocp",
    "monitor_completion_key",
    "notify_closing",
    "readinto_overlapped",
    "register_with_iocp",
    "wait_overlapped",
    "wait_readable",
    "wait_writable",
    "write_overlapped",
]


@enable_ki_protection
async def wait_readable(sock: _HasFileNo | int) -> None:
    """Block until the kernel reports that the given object is readable.

    On Unix systems, ``sock`` must either be an integer file descriptor,
    or else an object with a ``.fileno()`` method which returns an
    integer file descriptor. Any kind of file descriptor can be passed,
    though the exact semantics will depend on your kernel. For example,
    this probably won't do anything useful for on-disk files.

    On Windows systems, ``sock`` must either be an integer ``SOCKET``
    handle, or else an object with a ``.fileno()`` method which returns
    an integer ``SOCKET`` handle. File descriptors aren't supported,
    and neither are handles that refer to anything besides a
    ``SOCKET``.

    :raises trio.BusyResourceError:
        if another task is already waiting for the given socket to
        become readable.
    :raises trio.ClosedResourceError:
        if another task calls :func:`notify_closing` while this
        function is still working.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_readable(sock)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def wait_writable(sock: _HasFileNo | int) -> None:
    """Block until the kernel reports that the given object is writable.

    See `wait_readable` for the definition of ``sock``.

    :raises trio.BusyResourceError:
        if another task is already waiting for the given socket to
        become writable.
    :raises trio.ClosedResourceError:
        if another task calls :func:`notify_closing` while this
        function is still working.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_writable(sock)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def notify_closing(handle: Handle | int | _HasFileNo) -> None:
    """Notify waiters of the given object that it will be closed.

    Call this before closing a file descriptor (on Unix) or socket (on
    Windows). This will cause any `wait_readable` or `wait_writable`
    calls on the given object to immediately wake up and raise
    `~trio.ClosedResourceError`.

    This doesn't actually close the object – you still have to do that
    yourself afterwards. Also, you want to be careful to make sure no
    new tasks start waiting on the object in between when you call this
    and when it's actually closed. So to close something properly, you
    usually want to do these steps in order:

    1. Explicitly mark the object as closed, so that any new attempts
       to use it will abort before they start.
    2. Call `notify_closing` to wake up any already-existing users.
    3. Actually close the object.

    It's also possible to do them in a different order if that's more
    convenient, *but only if* you make sure not to have any checkpoints in
    between the steps. This way they all happen in a single atomic
    step, so other tasks won't be able to tell what order they happened
    in anyway.
    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.notify_closing(handle)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def register_with_iocp(handle: int | CData) -> None:
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__ and `#52
    <https://github.com/python-trio/trio/issues/52>`__.
    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.register_with_iocp(handle)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def wait_overlapped(handle_: int | CData, lpOverlapped: CData | int) -> object:
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__ and `#52
    <https://github.com/python-trio/trio/issues/52>`__.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_overlapped(
            handle_, lpOverlapped
        )
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def write_overlapped(
    handle: int | CData, data: Buffer, file_offset: int = 0
) -> int:
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__ and `#52
    <https://github.com/python-trio/trio/issues/52>`__.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.write_overlapped(
            handle, data, file_offset
        )
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def readinto_overlapped(
    handle: int | CData, buffer: Buffer, file_offset: int = 0
) -> int:
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__ and `#52
    <https://github.com/python-trio/trio/issues/52>`__.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.readinto_overlapped(
            handle, buffer, file_offset
        )
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def current_iocp() -> int:
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__ and `#52
    <https://github.com/python-trio/trio/issues/52>`__.
    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.current_iocp()
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def monitor_completion_key() -> (
    AbstractContextManager[tuple[int, UnboundedQueue[object]]]
):
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__ and `#52
    <https://github.com/python-trio/trio/issues/52>`__.
    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.monitor_completion_key()
    except AttributeError:
        raise RuntimeError("must be called from async context") from None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\reduced_build_config_parser.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os

# Check if the flatbuffers module is available. If not we cannot handle type reduction information in the config.
try:
    import flatbuffers  # noqa: F401

    have_flatbuffers = True
    from .ort_format_model import GloballyAllowedTypesOpTypeImplFilter, OperatorTypeUsageManager
except ImportError:
    have_flatbuffers = False


def parse_config(config_file: str, enable_type_reduction: bool = False):
    """
    Parse the configuration file and return the required operators dictionary and an
    OpTypeImplFilterInterface instance.

    Configuration file lines can do the following:
    1. specify required operators
    2. specify globally allowed types for all operators
    3. specify what it means for no required operators to be specified

    1. Specifying required operators

    The basic format for specifying required operators is `domain;opset1,opset2;op1,op2...`
    e.g. `ai.onnx;11;Add,Cast,Clip,... for a single opset
         `ai.onnx;11,12;Add,Cast,Clip,... for multiple opsets

         note: Configuration information is accrued as the file is parsed. If an operator requires support from multiple
         opsets that can be done with one entry for each opset, or one entry with multiple opsets in it.

    If the configuration file is generated from ORT format models it may optionally contain JSON for per-operator
    type reduction. The required types are generally listed per input and/or output of the operator.
    The type information is in a map, with 'inputs' and 'outputs' keys. The value for 'inputs' or 'outputs' is a map
    between the index number of the input/output and the required list of types.

    For example, both the input and output types are relevant to ai.onnx:Cast.
    Type information for input 0 and output 0 could look like this:
        `{"inputs": {"0": ["float", "int32_t"]}, "outputs": {"0": ["float", "int64_t"]}}`

    which is added directly after the operator name in the configuration file.
    e.g.
        `ai.onnx;12;Add,Cast{"inputs": {"0": ["float", "int32_t"]}, "outputs": {"0": ["float", "int64_t"]}},Concat`

    If for example the types of inputs 0 and 1 were important, the entry may look like this (e.g. ai.onnx:Gather):
        `{"inputs": {"0": ["float", "int32_t"], "1": ["int32_t"]}}`

    Finally some operators do non-standard things and store their type information under a 'custom' key.
    ai.onnx.OneHot is an example of this, where the three input types are combined into a triple.
        `{"custom": [["float", "int64_t", "int64_t"], ["int64_t", "std::string", "int64_t"]]}`

    2. Specifying globally allowed types for all operators

    The format for specifying globally allowed types for all operators is:
        `!globally_allowed_types;T0,T1,...`

    Ti should be a C++ scalar type supported by ONNX and ORT.
    At most one globally allowed types specification is allowed.

    Specifying per-operator type information and specifying globally allowed types are mutually exclusive - it is an
    error to specify both.

    3. Specify what it means for no required operators to be specified

    By default, if no required operators are specified, NO operators are required.

    With the following line, if no required operators are specified, ALL operators are required:
        `!no_ops_specified_means_all_ops_are_required`

    :param config_file: Configuration file to parse
    :param enable_type_reduction: Set to True to use the type information in the config.
                                  If False the type information will be ignored.
                                  If the flatbuffers module is unavailable type information will be ignored as the
                                  type-based filtering has a dependency on the ORT flatbuffers schema.
    :return: required_ops: Dictionary of domain:opset:[ops] for required operators. If None, all operators are
                           required.
             op_type_impl_filter: OpTypeImplFilterInterface instance if type reduction is enabled, the flatbuffers
                                  module is available, and type reduction information is present. None otherwise.
    """

    if not os.path.isfile(config_file):
        raise ValueError(f"Configuration file {config_file} does not exist")

    # only enable type reduction when flatbuffers is available
    enable_type_reduction = enable_type_reduction and have_flatbuffers

    required_ops = {}
    no_ops_specified_means_all_ops_are_required = False
    op_type_usage_manager = OperatorTypeUsageManager() if enable_type_reduction else None
    has_op_type_reduction_info = False
    globally_allowed_types = None

    def process_non_op_line(line):
        if not line or line.startswith("#"):  # skip empty lines and comments
            return True

        if line.startswith("!globally_allowed_types;"):  # handle globally allowed types
            if enable_type_reduction:
                nonlocal globally_allowed_types
                if globally_allowed_types is not None:
                    raise RuntimeError("Globally allowed types were already specified.")
                globally_allowed_types = {segment.strip() for segment in line.split(";")[1].split(",")}
            return True

        if line == "!no_ops_specified_means_all_ops_are_required":  # handle all ops required line
            nonlocal no_ops_specified_means_all_ops_are_required
            no_ops_specified_means_all_ops_are_required = True
            return True

        return False

    with open(config_file) as config:
        for line in [orig_line.strip() for orig_line in config]:
            if process_non_op_line(line):
                continue

            domain, opset_str, operators_str = (segment.strip() for segment in line.split(";"))
            opsets = [int(s) for s in opset_str.split(",")]

            # any type reduction information is serialized json that starts/ends with { and }.
            # type info is optional for each operator.
            if "{" in operators_str:
                has_op_type_reduction_info = True

                # parse the entries in the json dictionary with type info
                operators = set()
                cur = 0
                end = len(operators_str)
                while cur < end:
                    next_comma = operators_str.find(",", cur)
                    next_open_brace = operators_str.find("{", cur)

                    if next_comma == -1:
                        next_comma = end

                    # the json string starts with '{', so if that is found (next_open_brace != -1)
                    # before the next comma (which would be the start of the next operator if there is no type info
                    # for the current operator), we have type info to parse.
                    # e.g. need to handle extracting the operator name and type info for OpB and OpD,
                    #      and just the operator names for OpA and OpC from this example string
                    #      OpA,OpB{"inputs": {"0": ["float", "int32_t"]}},OpC,OpD{"outputs": {"0": ["int32_t"]}}
                    if 0 < next_open_brace < next_comma:
                        operator = operators_str[cur:next_open_brace].strip()
                        operators.add(operator)

                        # parse out the json dictionary with the type info by finding the closing brace that matches
                        # the opening brace
                        i = next_open_brace + 1
                        num_open_braces = 1
                        while num_open_braces > 0 and i < end:
                            if operators_str[i] == "{":
                                num_open_braces += 1
                            elif operators_str[i] == "}":
                                num_open_braces -= 1
                            i += 1

                        if num_open_braces != 0:
                            raise RuntimeError("Mismatched { and } in type string: " + operators_str[next_open_brace:])

                        if op_type_usage_manager:
                            type_str = operators_str[next_open_brace:i]
                            op_type_usage_manager.restore_from_config_entry(domain, operator, type_str)

                        cur = i + 1
                    else:
                        # comma or end of line is next
                        end_str = next_comma if next_comma != -1 else end
                        operators.add(operators_str[cur:end_str].strip())
                        cur = end_str + 1

            else:
                operators = {op.strip() for op in operators_str.split(",")}

            for opset in opsets:
                if domain not in required_ops:
                    required_ops[domain] = {opset: operators}
                elif opset not in required_ops[domain]:
                    required_ops[domain][opset] = operators
                else:
                    required_ops[domain][opset].update(operators)

    if len(required_ops) == 0 and no_ops_specified_means_all_ops_are_required:
        required_ops = None

    op_type_impl_filter = None
    if enable_type_reduction:
        if not has_op_type_reduction_info:
            op_type_usage_manager = None
        if globally_allowed_types is not None and op_type_usage_manager is not None:
            raise RuntimeError(
                "Specifying globally allowed types and per-op type reduction info together is unsupported."
            )

        if globally_allowed_types is not None:
            op_type_impl_filter = GloballyAllowedTypesOpTypeImplFilter(globally_allowed_types)
        elif op_type_usage_manager is not None:
            op_type_impl_filter = op_type_usage_manager.make_op_type_impl_filter()

    return required_ops, op_type_impl_filter

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\Tensor.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Tensor(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Tensor()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensor(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def TensorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # Tensor
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Tensor
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Tensor
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Tensor
    def Dims(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # Tensor
    def DimsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    # Tensor
    def DimsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Tensor
    def DimsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Tensor
    def DataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Tensor
    def RawData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # Tensor
    def RawDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # Tensor
    def RawDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Tensor
    def RawDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # Tensor
    def StringData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Tensor
    def StringDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Tensor
    def StringDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # Tensor
    def ExternalDataOffset(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return -1

def TensorStart(builder):
    builder.StartObject(7)

def Start(builder):
    TensorStart(builder)

def TensorAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    TensorAddName(builder, name)

def TensorAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    TensorAddDocString(builder, docString)

def TensorAddDims(builder, dims):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(dims), 0)

def AddDims(builder, dims):
    TensorAddDims(builder, dims)

def TensorStartDimsVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)

def StartDimsVector(builder, numElems: int) -> int:
    return TensorStartDimsVector(builder, numElems)

def TensorAddDataType(builder, dataType):
    builder.PrependInt32Slot(3, dataType, 0)

def AddDataType(builder, dataType):
    TensorAddDataType(builder, dataType)

def TensorAddRawData(builder, rawData):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(rawData), 0)

def AddRawData(builder, rawData):
    TensorAddRawData(builder, rawData)

def TensorStartRawDataVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)

def StartRawDataVector(builder, numElems: int) -> int:
    return TensorStartRawDataVector(builder, numElems)

def TensorAddStringData(builder, stringData):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(stringData), 0)

def AddStringData(builder, stringData):
    TensorAddStringData(builder, stringData)

def TensorStartStringDataVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartStringDataVector(builder, numElems: int) -> int:
    return TensorStartStringDataVector(builder, numElems)

def TensorAddExternalDataOffset(builder, externalDataOffset):
    builder.PrependInt64Slot(6, externalDataOffset, -1)

def AddExternalDataOffset(builder, externalDataOffset):
    TensorAddExternalDataOffset(builder, externalDataOffset)

def TensorEnd(builder):
    return builder.EndObject()

def End(builder):
    return TensorEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\style.py ===
"""
    pygments.style
    ~~~~~~~~~~~~~~

    Basic style object.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from pip._vendor.pygments.token import Token, STANDARD_TYPES

# Default mapping of ansixxx to RGB colors.
_ansimap = {
    # dark
    'ansiblack': '000000',
    'ansired': '7f0000',
    'ansigreen': '007f00',
    'ansiyellow': '7f7fe0',
    'ansiblue': '00007f',
    'ansimagenta': '7f007f',
    'ansicyan': '007f7f',
    'ansigray': 'e5e5e5',
    # normal
    'ansibrightblack': '555555',
    'ansibrightred': 'ff0000',
    'ansibrightgreen': '00ff00',
    'ansibrightyellow': 'ffff00',
    'ansibrightblue': '0000ff',
    'ansibrightmagenta': 'ff00ff',
    'ansibrightcyan': '00ffff',
    'ansiwhite': 'ffffff',
}
# mapping of deprecated #ansixxx colors to new color names
_deprecated_ansicolors = {
    # dark
    '#ansiblack': 'ansiblack',
    '#ansidarkred': 'ansired',
    '#ansidarkgreen': 'ansigreen',
    '#ansibrown': 'ansiyellow',
    '#ansidarkblue': 'ansiblue',
    '#ansipurple': 'ansimagenta',
    '#ansiteal': 'ansicyan',
    '#ansilightgray': 'ansigray',
    # normal
    '#ansidarkgray': 'ansibrightblack',
    '#ansired': 'ansibrightred',
    '#ansigreen': 'ansibrightgreen',
    '#ansiyellow': 'ansibrightyellow',
    '#ansiblue': 'ansibrightblue',
    '#ansifuchsia': 'ansibrightmagenta',
    '#ansiturquoise': 'ansibrightcyan',
    '#ansiwhite': 'ansiwhite',
}
ansicolors = set(_ansimap)


class StyleMeta(type):

    def __new__(mcs, name, bases, dct):
        obj = type.__new__(mcs, name, bases, dct)
        for token in STANDARD_TYPES:
            if token not in obj.styles:
                obj.styles[token] = ''

        def colorformat(text):
            if text in ansicolors:
                return text
            if text[0:1] == '#':
                col = text[1:]
                if len(col) == 6:
                    return col
                elif len(col) == 3:
                    return col[0] * 2 + col[1] * 2 + col[2] * 2
            elif text == '':
                return ''
            elif text.startswith('var') or text.startswith('calc'):
                return text
            assert False, f"wrong color format {text!r}"

        _styles = obj._styles = {}

        for ttype in obj.styles:
            for token in ttype.split():
                if token in _styles:
                    continue
                ndef = _styles.get(token.parent, None)
                styledefs = obj.styles.get(token, '').split()
                if not ndef or token is None:
                    ndef = ['', 0, 0, 0, '', '', 0, 0, 0]
                elif 'noinherit' in styledefs and token is not Token:
                    ndef = _styles[Token][:]
                else:
                    ndef = ndef[:]
                _styles[token] = ndef
                for styledef in obj.styles.get(token, '').split():
                    if styledef == 'noinherit':
                        pass
                    elif styledef == 'bold':
                        ndef[1] = 1
                    elif styledef == 'nobold':
                        ndef[1] = 0
                    elif styledef == 'italic':
                        ndef[2] = 1
                    elif styledef == 'noitalic':
                        ndef[2] = 0
                    elif styledef == 'underline':
                        ndef[3] = 1
                    elif styledef == 'nounderline':
                        ndef[3] = 0
                    elif styledef[:3] == 'bg:':
                        ndef[4] = colorformat(styledef[3:])
                    elif styledef[:7] == 'border:':
                        ndef[5] = colorformat(styledef[7:])
                    elif styledef == 'roman':
                        ndef[6] = 1
                    elif styledef == 'sans':
                        ndef[7] = 1
                    elif styledef == 'mono':
                        ndef[8] = 1
                    else:
                        ndef[0] = colorformat(styledef)

        return obj

    def style_for_token(cls, token):
        t = cls._styles[token]
        ansicolor = bgansicolor = None
        color = t[0]
        if color in _deprecated_ansicolors:
            color = _deprecated_ansicolors[color]
        if color in ansicolors:
            ansicolor = color
            color = _ansimap[color]
        bgcolor = t[4]
        if bgcolor in _deprecated_ansicolors:
            bgcolor = _deprecated_ansicolors[bgcolor]
        if bgcolor in ansicolors:
            bgansicolor = bgcolor
            bgcolor = _ansimap[bgcolor]

        return {
            'color':        color or None,
            'bold':         bool(t[1]),
            'italic':       bool(t[2]),
            'underline':    bool(t[3]),
            'bgcolor':      bgcolor or None,
            'border':       t[5] or None,
            'roman':        bool(t[6]) or None,
            'sans':         bool(t[7]) or None,
            'mono':         bool(t[8]) or None,
            'ansicolor':    ansicolor,
            'bgansicolor':  bgansicolor,
        }

    def list_styles(cls):
        return list(cls)

    def styles_token(cls, ttype):
        return ttype in cls._styles

    def __iter__(cls):
        for token in cls._styles:
            yield token, cls.style_for_token(token)

    def __len__(cls):
        return len(cls._styles)


class Style(metaclass=StyleMeta):

    #: overall background color (``None`` means transparent)
    background_color = '#ffffff'

    #: highlight background color
    highlight_color = '#ffffcc'

    #: line number font color
    line_number_color = 'inherit'

    #: line number background color
    line_number_background_color = 'transparent'

    #: special line number font color
    line_number_special_color = '#000000'

    #: special line number background color
    line_number_special_background_color = '#ffffc0'

    #: Style definitions for individual token types.
    styles = {}

    #: user-friendly style name (used when selecting the style, so this
    # should be all-lowercase, no spaces, hyphens)
    name = 'unnamed'

    aliases = []

    # Attribute for lexers defined within Pygments. If set
    # to True, the style is not shown in the style gallery
    # on the website. This is intended for language-specific
    # styles.
    web_style_gallery_exclude = False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\polynomialring.py ===
"""Implementation of :class:`PolynomialRing` class. """


from sympy.polys.domains.ring import Ring
from sympy.polys.domains.compositedomain import CompositeDomain

from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.utilities import public

@public
class PolynomialRing(Ring, CompositeDomain):
    """A class for representing multivariate polynomial rings. """

    is_PolynomialRing = is_Poly = True

    has_assoc_Ring  = True
    has_assoc_Field = True

    def __init__(self, domain_or_ring, symbols=None, order=None):
        from sympy.polys.rings import PolyRing

        if isinstance(domain_or_ring, PolyRing) and symbols is None and order is None:
            ring = domain_or_ring
        else:
            ring = PolyRing(symbols, domain_or_ring, order)

        self.ring = ring
        self.dtype = ring.dtype

        self.gens = ring.gens
        self.ngens = ring.ngens
        self.symbols = ring.symbols
        self.domain = ring.domain


        if symbols:
            if ring.domain.is_Field and ring.domain.is_Exact and len(symbols)==1:
                self.is_PID = True

        # TODO: remove this
        self.dom = self.domain

    def new(self, element):
        return self.ring.ring_new(element)

    def of_type(self, element):
        """Check if ``a`` is of type ``dtype``. """
        return self.ring.is_element(element)

    @property
    def zero(self):
        return self.ring.zero

    @property
    def one(self):
        return self.ring.one

    @property
    def order(self):
        return self.ring.order

    def __str__(self):
        return str(self.domain) + '[' + ','.join(map(str, self.symbols)) + ']'

    def __hash__(self):
        return hash((self.__class__.__name__, self.ring, self.domain, self.symbols))

    def __eq__(self, other):
        """Returns `True` if two domains are equivalent. """
        if not isinstance(other, PolynomialRing):
            return NotImplemented
        return self.ring == other.ring

    def is_unit(self, a):
        """Returns ``True`` if ``a`` is a unit of ``self``"""
        if not a.is_ground:
            return False
        K = self.domain
        return K.is_unit(K.convert_from(a, self))

    def canonical_unit(self, a):
        u = self.domain.canonical_unit(a.LC)
        return self.ring.ground_new(u)

    def to_sympy(self, a):
        """Convert `a` to a SymPy object. """
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to `dtype`. """
        return self.ring.from_expr(a)

    def from_ZZ(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY `mpz` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY `mpq` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a `GaussianInteger` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a `GaussianRational` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert an algebraic number to ``dtype``. """
        if K1.domain != K0:
            a = K1.domain.convert_from(a, K0)
        if a is not None:
            return K1.new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        try:
            return a.set_ring(K1.ring)
        except (CoercionFailed, GeneratorsError):
            return None

    def from_FractionField(K1, a, K0):
        """Convert a rational function to ``dtype``. """
        if K1.domain == K0:
            return K1.ring.from_list([a])

        q, r = K0.numer(a).div(K0.denom(a))

        if r.is_zero:
            return K1.from_PolynomialRing(q, K0.field.ring.to_domain())
        else:
            return None

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert from old poly ring to ``dtype``. """
        if K1.symbols == K0.gens:
            ad = a.to_dict()
            if K1.domain != K0.domain:
                ad = {m: K1.domain.convert(c) for m, c in ad.items()}
            return K1(ad)
        elif a.is_ground and K0.domain == K1:
            return K1.convert_from(a.to_list()[0], K0.domain)

    def get_field(self):
        """Returns a field associated with `self`. """
        return self.ring.to_field().to_domain()

    def is_positive(self, a):
        """Returns True if `LC(a)` is positive. """
        return self.domain.is_positive(a.LC)

    def is_negative(self, a):
        """Returns True if `LC(a)` is negative. """
        return self.domain.is_negative(a.LC)

    def is_nonpositive(self, a):
        """Returns True if `LC(a)` is non-positive. """
        return self.domain.is_nonpositive(a.LC)

    def is_nonnegative(self, a):
        """Returns True if `LC(a)` is non-negative. """
        return self.domain.is_nonnegative(a.LC)

    def gcdex(self, a, b):
        """Extended GCD of `a` and `b`. """
        return a.gcdex(b)

    def gcd(self, a, b):
        """Returns GCD of `a` and `b`. """
        return a.gcd(b)

    def lcm(self, a, b):
        """Returns LCM of `a` and `b`. """
        return a.lcm(b)

    def factorial(self, a):
        """Returns factorial of `a`. """
        return self.dtype(self.domain.factorial(a))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\urls.py ===
from __future__ import annotations

import codecs
import re
import typing as t
import urllib.parse
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from .datastructures import iter_multi_items


def _codec_error_url_quote(e: UnicodeError) -> tuple[str, int]:
    """Used in :func:`uri_to_iri` after unquoting to re-quote any
    invalid bytes.
    """
    # the docs state that UnicodeError does have these attributes,
    # but mypy isn't picking them up
    out = quote(e.object[e.start : e.end], safe="")  # type: ignore
    return out, e.end  # type: ignore


codecs.register_error("werkzeug.url_quote", _codec_error_url_quote)


def _make_unquote_part(name: str, chars: str) -> t.Callable[[str], str]:
    """Create a function that unquotes all percent encoded characters except those
    given. This allows working with unquoted characters if possible while not changing
    the meaning of a given part of a URL.
    """
    choices = "|".join(f"{ord(c):02X}" for c in sorted(chars))
    pattern = re.compile(f"((?:%(?:{choices}))+)", re.I)

    def _unquote_partial(value: str) -> str:
        parts = iter(pattern.split(value))
        out = []

        for part in parts:
            out.append(unquote(part, "utf-8", "werkzeug.url_quote"))
            out.append(next(parts, ""))

        return "".join(out)

    _unquote_partial.__name__ = f"_unquote_{name}"
    return _unquote_partial


# characters that should remain quoted in URL parts
# based on https://url.spec.whatwg.org/#percent-encoded-bytes
# always keep all controls, space, and % quoted
_always_unsafe = bytes((*range(0x21), 0x25, 0x7F)).decode()
_unquote_fragment = _make_unquote_part("fragment", _always_unsafe)
_unquote_query = _make_unquote_part("query", _always_unsafe + "&=+#")
_unquote_path = _make_unquote_part("path", _always_unsafe + "/?#")
_unquote_user = _make_unquote_part("user", _always_unsafe + ":@/?#")


def uri_to_iri(uri: str) -> str:
    """Convert a URI to an IRI. All valid UTF-8 characters are unquoted,
    leaving all reserved and invalid characters quoted. If the URL has
    a domain, it is decoded from Punycode.

    >>> uri_to_iri("http://xn--n3h.net/p%C3%A5th?q=%C3%A8ry%DF")
    'http://\\u2603.net/p\\xe5th?q=\\xe8ry%DF'

    :param uri: The URI to convert.

    .. versionchanged:: 3.0
        Passing a tuple or bytes, and the ``charset`` and ``errors`` parameters,
        are removed.

    .. versionchanged:: 2.3
        Which characters remain quoted is specific to each part of the URL.

    .. versionchanged:: 0.15
        All reserved and invalid characters remain quoted. Previously,
        only some reserved characters were preserved, and invalid bytes
        were replaced instead of left quoted.

    .. versionadded:: 0.6
    """
    parts = urlsplit(uri)
    path = _unquote_path(parts.path)
    query = _unquote_query(parts.query)
    fragment = _unquote_fragment(parts.fragment)

    if parts.hostname:
        netloc = _decode_idna(parts.hostname)
    else:
        netloc = ""

    if ":" in netloc:
        netloc = f"[{netloc}]"

    if parts.port:
        netloc = f"{netloc}:{parts.port}"

    if parts.username:
        auth = _unquote_user(parts.username)

        if parts.password:
            password = _unquote_user(parts.password)
            auth = f"{auth}:{password}"

        netloc = f"{auth}@{netloc}"

    return urlunsplit((parts.scheme, netloc, path, query, fragment))


def iri_to_uri(iri: str) -> str:
    """Convert an IRI to a URI. All non-ASCII and unsafe characters are
    quoted. If the URL has a domain, it is encoded to Punycode.

    >>> iri_to_uri('http://\\u2603.net/p\\xe5th?q=\\xe8ry%DF')
    'http://xn--n3h.net/p%C3%A5th?q=%C3%A8ry%DF'

    :param iri: The IRI to convert.

    .. versionchanged:: 3.0
        Passing a tuple or bytes, the ``charset`` and ``errors`` parameters,
        and the ``safe_conversion`` parameter, are removed.

    .. versionchanged:: 2.3
        Which characters remain unquoted is specific to each part of the URL.

    .. versionchanged:: 0.15
        All reserved characters remain unquoted. Previously, only some reserved
        characters were left unquoted.

    .. versionchanged:: 0.9.6
       The ``safe_conversion`` parameter was added.

    .. versionadded:: 0.6
    """
    parts = urlsplit(iri)
    # safe = https://url.spec.whatwg.org/#url-path-segment-string
    # as well as percent for things that are already quoted
    path = quote(parts.path, safe="%!$&'()*+,/:;=@")
    query = quote(parts.query, safe="%!$&'()*+,/:;=?@")
    fragment = quote(parts.fragment, safe="%!#$&'()*+,/:;=?@")

    if parts.hostname:
        netloc = parts.hostname.encode("idna").decode("ascii")
    else:
        netloc = ""

    if ":" in netloc:
        netloc = f"[{netloc}]"

    if parts.port:
        netloc = f"{netloc}:{parts.port}"

    if parts.username:
        auth = quote(parts.username, safe="%!$&'()*+,;=")

        if parts.password:
            password = quote(parts.password, safe="%!$&'()*+,;=")
            auth = f"{auth}:{password}"

        netloc = f"{auth}@{netloc}"

    return urlunsplit((parts.scheme, netloc, path, query, fragment))


# Python < 3.12
# itms-services was worked around in previous iri_to_uri implementations, but
# we can tell Python directly that it needs to preserve the //.
if "itms-services" not in urllib.parse.uses_netloc:
    urllib.parse.uses_netloc.append("itms-services")


def _decode_idna(domain: str) -> str:
    try:
        data = domain.encode("ascii")
    except UnicodeEncodeError:
        # If the domain is not ASCII, it's decoded already.
        return domain

    try:
        # Try decoding in one shot.
        return data.decode("idna")
    except UnicodeDecodeError:
        pass

    # Decode each part separately, leaving invalid parts as punycode.
    parts = []

    for part in data.split(b"."):
        try:
            parts.append(part.decode("idna"))
        except UnicodeDecodeError:
            parts.append(part.decode("ascii"))

    return ".".join(parts)


def _urlencode(query: t.Mapping[str, str] | t.Iterable[tuple[str, str]]) -> str:
    items = [x for x in iter_multi_items(query) if x[1] is not None]
    # safe = https://url.spec.whatwg.org/#percent-encoded-bytes
    return urlencode(items, safe="!$'()*,/:;?@")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\cell\rich_text.py ===
# Copyright (c) 2010-2024 openpyxl

"""
RichText definition
"""
from copy import copy
from openpyxl.compat import NUMERIC_TYPES
from openpyxl.cell.text import InlineFont, Text
from openpyxl.descriptors import (
    Strict,
    String,
    Typed
)

from openpyxl.xml.functions import Element, whitespace

class TextBlock(Strict):
    """ Represents text string in a specific format

    This class is used as part of constructing a rich text strings.
    """
    font = Typed(expected_type=InlineFont)
    text = String()

    def __init__(self, font, text):
        self.font = font
        self.text = text


    def __eq__(self, other):
        return self.text == other.text and self.font == other.font


    def __str__(self):
        """Just retun the text"""
        return self.text


    def __repr__(self):
        font = self.font != InlineFont() and self.font or "default"
        return f"{self.__class__.__name__} text={self.text}, font={font}"


    def to_tree(self):
        el = Element("r")
        el.append(self.font.to_tree(tagname="rPr"))
        t = Element("t")
        t.text = self.text
        whitespace(t)
        el.append(t)
        return el

#
# Rich Text class.
# This class behaves just like a list whose members are either simple strings, or TextBlock() instances.
# In addition, it can be initialized in several ways:
# t = CellRFichText([...]) # initialize with a list.
# t = CellRFichText((...)) # initialize with a tuple.
# t = CellRichText(node) # where node is an Element() from either lxml or xml.etree (has a 'tag' element)
class CellRichText(list):
    """Represents a rich text string.

    Initialize with a list made of pure strings or :class:`TextBlock` elements
    Can index object to access or modify individual rich text elements
    it also supports the + and += operators between rich text strings
    There are no user methods for this class

    operations which modify the string will generally call an optimization pass afterwards,
    that merges text blocks with identical formats, consecutive pure text strings,
    and remove empty strings and empty text blocks
    """

    def __init__(self, *args):
        if len(args) == 1:
            args = args[0]
            if isinstance(args, (list, tuple)):
                CellRichText._check_rich_text(args)
            else:
                CellRichText._check_element(args)
                args = [args]
        else:
            CellRichText._check_rich_text(args)
        super().__init__(args)


    @classmethod
    def _check_element(cls, value):
        if not isinstance(value, (str, TextBlock, NUMERIC_TYPES)):
            raise TypeError(f"Illegal CellRichText element {value}")


    @classmethod
    def _check_rich_text(cls, rich_text):
        for t in rich_text:
            CellRichText._check_element(t)

    @classmethod
    def from_tree(cls, node):
        text = Text.from_tree(node)
        if text.t:
            return (text.t.replace('x005F_', ''),)
        s = []
        for r in text.r:
            t = ""
            if r.t:
                t = r.t.replace('x005F_', '')
            if r.rPr:
                s.append(TextBlock(r.rPr, t))
            else:
                s.append(t)
        return cls(s)

    # Merge TextBlocks with identical formatting
    # remove empty elements
    def _opt(self):
        last_t = None
        l = CellRichText(tuple())
        for t in self:
            if isinstance(t, str):
                if not t:
                    continue
            elif not t.text:
                continue
            if type(last_t) == type(t):
                if isinstance(t, str):
                    last_t += t
                    continue
                elif last_t.font == t.font:
                    last_t.text += t.text
                    continue
            if last_t:
                l.append(last_t)
            last_t = t
        if last_t:
            # Add remaining TextBlock at end of rich text
            l.append(last_t)
        super().__setitem__(slice(None), l)
        return self


    def __iadd__(self, arg):
        # copy used here to create new TextBlock() so we don't modify the right hand side in _opt()
        CellRichText._check_rich_text(arg)
        super().__iadd__([copy(e) for e in list(arg)])
        return self._opt()


    def __add__(self, arg):
        return CellRichText([copy(e) for e in list(self) + list(arg)])._opt()


    def __setitem__(self, indx, val):
        CellRichText._check_element(val)
        super().__setitem__(indx, val)
        self._opt()


    def append(self, arg):
        CellRichText._check_element(arg)
        super().append(arg)


    def extend(self, arg):
        CellRichText._check_rich_text(arg)
        super().extend(arg)


    def __repr__(self):
        return "CellRichText([{}])".format(', '.join((repr(s) for s in self)))


    def __str__(self):
        return ''.join([str(s) for s in self])


    def as_list(self):
        """
        Returns a list of the strings contained.
        The main reason for this is to make editing easier.
        """
        return [str(s) for s in self]


    def to_tree(self):
        """
        Return the full XML representation
        """
        container = Element("is")
        for obj in self:
            if isinstance(obj, TextBlock):
                container.append(obj.to_tree())

            else:
                el = Element("r")
                t = Element("t")
                t.text = obj
                whitespace(t)
                el.append(t)
                container.append(el)

        return container


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\datavalidation.py ===
# Copyright (c) 2010-2024 openpyxl

from collections import defaultdict
from itertools import chain
from operator import itemgetter

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Bool,
    NoneSet,
    String,
    Sequence,
    Alias,
    Integer,
    Convertible,
)
from openpyxl.descriptors.nested import NestedText

from openpyxl.utils import (
    rows_from_range,
    coordinate_to_tuple,
    get_column_letter,
)


def collapse_cell_addresses(cells, input_ranges=()):
    """ Collapse a collection of cell co-ordinates down into an optimal
        range or collection of ranges.

        E.g. Cells A1, A2, A3, B1, B2 and B3 should have the data-validation
        object applied, attempt to collapse down to a single range, A1:B3.

        Currently only collapsing contiguous vertical ranges (i.e. above
        example results in A1:A3 B1:B3).
    """

    ranges = list(input_ranges)

    # convert cell into row, col tuple
    raw_coords = (coordinate_to_tuple(cell) for cell in cells)

    # group by column in order
    grouped_coords = defaultdict(list)
    for row, col in sorted(raw_coords, key=itemgetter(1)):
        grouped_coords[col].append(row)

    # create range string from first and last row in column
    for col, cells in grouped_coords.items():
        col = get_column_letter(col)
        fmt = "{0}{1}:{2}{3}"
        if len(cells) == 1:
            fmt = "{0}{1}"
        r = fmt.format(col, min(cells), col, max(cells))
        ranges.append(r)

    return " ".join(ranges)


def expand_cell_ranges(range_string):
    """
    Expand cell ranges to a sequence of addresses.
    Reverse of collapse_cell_addresses
    Eg. converts "A1:A2 B1:B2" to (A1, A2, B1, B2)
    """
    # expand ranges to rows and then flatten
    rows = (rows_from_range(rs) for rs in range_string.split()) # list of rows
    cells = (chain(*row) for row in rows) # flatten rows
    return set(chain(*cells))


from .cell_range import MultiCellRange


class DataValidation(Serialisable):

    tagname = "dataValidation"

    sqref = Convertible(expected_type=MultiCellRange)
    cells = Alias("sqref")
    ranges = Alias("sqref")

    showDropDown = Bool(allow_none=True)
    hide_drop_down = Alias('showDropDown')
    showInputMessage = Bool(allow_none=True)
    showErrorMessage = Bool(allow_none=True)
    allowBlank = Bool(allow_none=True)
    allow_blank = Alias('allowBlank')

    errorTitle = String(allow_none = True)
    error = String(allow_none = True)
    promptTitle = String(allow_none = True)
    prompt = String(allow_none = True)
    formula1 = NestedText(allow_none=True, expected_type=str)
    formula2 = NestedText(allow_none=True, expected_type=str)

    type = NoneSet(values=("whole", "decimal", "list", "date", "time",
                           "textLength", "custom"))
    errorStyle = NoneSet(values=("stop", "warning", "information"))
    imeMode = NoneSet(values=("noControl", "off", "on", "disabled",
                              "hiragana", "fullKatakana", "halfKatakana", "fullAlpha","halfAlpha",
                              "fullHangul", "halfHangul"))
    operator = NoneSet(values=("between", "notBetween", "equal", "notEqual",
                               "lessThan", "lessThanOrEqual", "greaterThan", "greaterThanOrEqual"))
    validation_type = Alias('type')

    def __init__(self,
                 type=None,
                 formula1=None,
                 formula2=None,
                 showErrorMessage=False,
                 showInputMessage=False,
                 showDropDown=False,
                 allowBlank=False,
                 sqref=(),
                 promptTitle=None,
                 errorStyle=None,
                 error=None,
                 prompt=None,
                 errorTitle=None,
                 imeMode=None,
                 operator=None,
                 allow_blank=None,
                 ):
        self.sqref = sqref
        self.showDropDown = showDropDown
        self.imeMode = imeMode
        self.operator = operator
        self.formula1 = formula1
        self.formula2 = formula2
        if allow_blank is not None:
            allowBlank = allow_blank
        self.allowBlank = allowBlank
        self.showErrorMessage = showErrorMessage
        self.showInputMessage = showInputMessage
        self.type = type
        self.promptTitle = promptTitle
        self.errorStyle = errorStyle
        self.error = error
        self.prompt = prompt
        self.errorTitle = errorTitle


    def add(self, cell):
        """Adds a cell or cell coordinate to this validator"""
        if hasattr(cell, "coordinate"):
            cell = cell.coordinate
        self.sqref += cell


    def __contains__(self, cell):
        if hasattr(cell, "coordinate"):
            cell = cell.coordinate
        return cell in self.sqref


class DataValidationList(Serialisable):

    tagname = "dataValidations"

    disablePrompts = Bool(allow_none=True)
    xWindow = Integer(allow_none=True)
    yWindow = Integer(allow_none=True)
    dataValidation = Sequence(expected_type=DataValidation)

    __elements__ = ('dataValidation',)
    __attrs__ = ('disablePrompts', 'xWindow', 'yWindow', 'count')

    def __init__(self,
                 disablePrompts=None,
                 xWindow=None,
                 yWindow=None,
                 count=None,
                 dataValidation=(),
                ):
        self.disablePrompts = disablePrompts
        self.xWindow = xWindow
        self.yWindow = yWindow
        self.dataValidation = dataValidation


    @property
    def count(self):
        return len(self)


    def __len__(self):
        return len(self.dataValidation)


    def append(self, dv):
        self.dataValidation.append(dv)


    def to_tree(self, tagname=None):
        """
        Need to skip validations that have no cell ranges
        """
        ranges = self.dataValidation # copy
        self.dataValidation = [r for r in self.dataValidation if bool(r.sqref)]
        xml = super().to_tree(tagname)
        self.dataValidation = ranges
        return xml

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\util\_doctools.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable


class TablePlotter:
    """
    Layout some DataFrames in vertical/horizontal layout for explanation.
    Used in merging.rst
    """

    def __init__(
        self,
        cell_width: float = 0.37,
        cell_height: float = 0.25,
        font_size: float = 7.5,
    ) -> None:
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.font_size = font_size

    def _shape(self, df: pd.DataFrame) -> tuple[int, int]:
        """
        Calculate table shape considering index levels.
        """
        row, col = df.shape
        return row + df.columns.nlevels, col + df.index.nlevels

    def _get_cells(self, left, right, vertical) -> tuple[int, int]:
        """
        Calculate appropriate figure size based on left and right data.
        """
        if vertical:
            # calculate required number of cells
            vcells = max(sum(self._shape(df)[0] for df in left), self._shape(right)[0])
            hcells = max(self._shape(df)[1] for df in left) + self._shape(right)[1]
        else:
            vcells = max([self._shape(df)[0] for df in left] + [self._shape(right)[0]])
            hcells = sum([self._shape(df)[1] for df in left] + [self._shape(right)[1]])
        return hcells, vcells

    def plot(self, left, right, labels: Iterable[str] = (), vertical: bool = True):
        """
        Plot left / right DataFrames in specified layout.

        Parameters
        ----------
        left : list of DataFrames before operation is applied
        right : DataFrame of operation result
        labels : list of str to be drawn as titles of left DataFrames
        vertical : bool, default True
            If True, use vertical layout. If False, use horizontal layout.
        """
        from matplotlib import gridspec
        import matplotlib.pyplot as plt

        if not isinstance(left, list):
            left = [left]
        left = [self._conv(df) for df in left]
        right = self._conv(right)

        hcells, vcells = self._get_cells(left, right, vertical)

        if vertical:
            figsize = self.cell_width * hcells, self.cell_height * vcells
        else:
            # include margin for titles
            figsize = self.cell_width * hcells, self.cell_height * vcells
        fig = plt.figure(figsize=figsize)

        if vertical:
            gs = gridspec.GridSpec(len(left), hcells)
            # left
            max_left_cols = max(self._shape(df)[1] for df in left)
            max_left_rows = max(self._shape(df)[0] for df in left)
            for i, (_left, _label) in enumerate(zip(left, labels)):
                ax = fig.add_subplot(gs[i, 0:max_left_cols])
                self._make_table(ax, _left, title=_label, height=1.0 / max_left_rows)
            # right
            ax = plt.subplot(gs[:, max_left_cols:])
            self._make_table(ax, right, title="Result", height=1.05 / vcells)
            fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
        else:
            max_rows = max(self._shape(df)[0] for df in left + [right])
            height = 1.0 / np.max(max_rows)
            gs = gridspec.GridSpec(1, hcells)
            # left
            i = 0
            for df, _label in zip(left, labels):
                sp = self._shape(df)
                ax = fig.add_subplot(gs[0, i : i + sp[1]])
                self._make_table(ax, df, title=_label, height=height)
                i += sp[1]
            # right
            ax = plt.subplot(gs[0, i:])
            self._make_table(ax, right, title="Result", height=height)
            fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)

        return fig

    def _conv(self, data):
        """
        Convert each input to appropriate for table outplot.
        """
        if isinstance(data, pd.Series):
            if data.name is None:
                data = data.to_frame(name="")
            else:
                data = data.to_frame()
        data = data.fillna("NaN")
        return data

    def _insert_index(self, data):
        # insert is destructive
        data = data.copy()
        idx_nlevels = data.index.nlevels
        if idx_nlevels == 1:
            data.insert(0, "Index", data.index)
        else:
            for i in range(idx_nlevels):
                data.insert(i, f"Index{i}", data.index._get_level_values(i))

        col_nlevels = data.columns.nlevels
        if col_nlevels > 1:
            col = data.columns._get_level_values(0)
            values = [
                data.columns._get_level_values(i)._values for i in range(1, col_nlevels)
            ]
            col_df = pd.DataFrame(values)
            data.columns = col_df.columns
            data = pd.concat([col_df, data])
            data.columns = col
        return data

    def _make_table(self, ax, df, title: str, height: float | None = None) -> None:
        if df is None:
            ax.set_visible(False)
            return

        from pandas import plotting

        idx_nlevels = df.index.nlevels
        col_nlevels = df.columns.nlevels
        # must be convert here to get index levels for colorization
        df = self._insert_index(df)
        tb = plotting.table(ax, df, loc=9)
        tb.set_fontsize(self.font_size)

        if height is None:
            height = 1.0 / (len(df) + 1)

        props = tb.properties()
        for (r, c), cell in props["celld"].items():
            if c == -1:
                cell.set_visible(False)
            elif r < col_nlevels and c < idx_nlevels:
                cell.set_visible(False)
            elif r < col_nlevels or c < idx_nlevels:
                cell.set_facecolor("#AAAAAA")
            cell.set_height(height)

        ax.set_title(title, size=self.font_size)
        ax.axis("off")


def main() -> None:
    import matplotlib.pyplot as plt

    p = TablePlotter()

    df1 = pd.DataFrame({"A": [10, 11, 12], "B": [20, 21, 22], "C": [30, 31, 32]})
    df2 = pd.DataFrame({"A": [10, 12], "C": [30, 32]})

    p.plot([df1, df2], pd.concat([df1, df2]), labels=["df1", "df2"], vertical=True)
    plt.show()

    df3 = pd.DataFrame({"X": [10, 12], "Z": [30, 32]})

    p.plot(
        [df1, df3], pd.concat([df1, df3], axis=1), labels=["df1", "df2"], vertical=False
    )
    plt.show()

    idx = pd.MultiIndex.from_tuples(
        [(1, "A"), (1, "B"), (1, "C"), (2, "A"), (2, "B"), (2, "C")]
    )
    column = pd.MultiIndex.from_tuples([(1, "A"), (1, "B")])
    df3 = pd.DataFrame({"v1": [1, 2, 3, 4, 5, 6], "v2": [5, 6, 7, 8, 9, 10]}, index=idx)
    df3.columns = column
    p.plot(df3, df3, labels=["df3"])
    plt.show()


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\quotientring.py ===
"""Implementation of :class:`QuotientRing` class."""


from sympy.polys.agca.modules import FreeModuleQuotientRing
from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, CoercionFailed
from sympy.utilities import public

# TODO
# - successive quotients (when quotient ideals are implemented)
# - poly rings over quotients?
# - division by non-units in integral domains?

@public
class QuotientRingElement:
    """
    Class representing elements of (commutative) quotient rings.

    Attributes:

    - ring - containing ring
    - data - element of ring.ring (i.e. base ring) representing self
    """

    def __init__(self, ring, data):
        self.ring = ring
        self.data = data

    def __str__(self):
        from sympy.printing.str import sstr
        data = self.ring.ring.to_sympy(self.data)
        return sstr(data) + " + " + str(self.ring.base_ideal)

    __repr__ = __str__

    def __bool__(self):
        return not self.ring.is_zero(self)

    def __add__(self, om):
        if not isinstance(om, self.__class__) or om.ring != self.ring:
            try:
                om = self.ring.convert(om)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring(self.data + om.data)

    __radd__ = __add__

    def __neg__(self):
        return self.ring(self.data*self.ring.ring.convert(-1))

    def __sub__(self, om):
        return self.__add__(-om)

    def __rsub__(self, om):
        return (-self).__add__(om)

    def __mul__(self, o):
        if not isinstance(o, self.__class__):
            try:
                o = self.ring.convert(o)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring(self.data*o.data)

    __rmul__ = __mul__

    def __rtruediv__(self, o):
        return self.ring.revert(self)*o

    def __truediv__(self, o):
        if not isinstance(o, self.__class__):
            try:
                o = self.ring.convert(o)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring.revert(o)*self

    def __pow__(self, oth):
        if oth < 0:
            return self.ring.revert(self) ** -oth
        return self.ring(self.data ** oth)

    def __eq__(self, om):
        if not isinstance(om, self.__class__) or om.ring != self.ring:
            return False
        return self.ring.is_zero(self - om)

    def __ne__(self, om):
        return not self == om


class QuotientRing(Ring):
    """
    Class representing (commutative) quotient rings.

    You should not usually instantiate this by hand, instead use the constructor
    from the base ring in the construction.

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> I = QQ.old_poly_ring(x).ideal(x**3 + 1)
    >>> QQ.old_poly_ring(x).quotient_ring(I)
    QQ[x]/<x**3 + 1>

    Shorter versions are possible:

    >>> QQ.old_poly_ring(x)/I
    QQ[x]/<x**3 + 1>

    >>> QQ.old_poly_ring(x)/[x**3 + 1]
    QQ[x]/<x**3 + 1>

    Attributes:

    - ring - the base ring
    - base_ideal - the ideal used to form the quotient
    """

    has_assoc_Ring = True
    has_assoc_Field = False
    dtype = QuotientRingElement

    def __init__(self, ring, ideal):
        if not ideal.ring == ring:
            raise ValueError('Ideal must belong to %s, got %s' % (ring, ideal))
        self.ring = ring
        self.base_ideal = ideal
        self.zero = self(self.ring.zero)
        self.one = self(self.ring.one)

    def __str__(self):
        return str(self.ring) + "/" + str(self.base_ideal)

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.ring, self.base_ideal))

    def new(self, a):
        """Construct an element of ``self`` domain from ``a``. """
        if not isinstance(a, self.ring.dtype):
            a = self.ring(a)
        # TODO optionally disable reduction?
        return self.dtype(self, self.base_ideal.reduce_element(a))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, QuotientRing) and \
            self.ring == other.ring and self.base_ideal == other.base_ideal

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.ring.convert(a, K0))

    from_ZZ_python = from_ZZ
    from_QQ_python = from_ZZ_python
    from_ZZ_gmpy = from_ZZ_python
    from_QQ_gmpy = from_ZZ_python
    from_RealField = from_ZZ_python
    from_GlobalPolynomialRing = from_ZZ_python
    from_FractionField = from_ZZ_python

    def from_sympy(self, a):
        return self(self.ring.from_sympy(a))

    def to_sympy(self, a):
        return self.ring.to_sympy(a.data)

    def from_QuotientRing(self, a, K0):
        if K0 == self:
            return a

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. ``K[X]``. """
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. ``K(X)``. """
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        """
        Compute a**(-1), if possible.
        """
        I = self.ring.ideal(a.data) + self.base_ideal
        try:
            return self(I.in_terms_of_generators(1)[0])
        except ValueError:  # 1 not in I
            raise NotReversible('%s not a unit in %r' % (a, self))

    def is_zero(self, a):
        return self.base_ideal.contains(a.data)

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        (QQ[x]/<x**2 + 1>)**2
        """
        return FreeModuleQuotientRing(self, rank)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\__init__.py ===
"""
SymPy statistics module

Introduces a random variable type into the SymPy language.

Random variables may be declared using prebuilt functions such as
Normal, Exponential, Coin, Die, etc...  or built with functions like FiniteRV.

Queries on random expressions can be made using the functions

========================= =============================
    Expression                    Meaning
------------------------- -----------------------------
 ``P(condition)``          Probability
 ``E(expression)``         Expected value
 ``H(expression)``         Entropy
 ``variance(expression)``  Variance
 ``density(expression)``   Probability Density Function
 ``sample(expression)``    Produce a realization
 ``where(condition)``      Where the condition is true
========================= =============================

Examples
========

>>> from sympy.stats import P, E, variance, Die, Normal
>>> from sympy import simplify
>>> X, Y = Die('X', 6), Die('Y', 6) # Define two six sided dice
>>> Z = Normal('Z', 0, 1) # Declare a Normal random variable with mean 0, std 1
>>> P(X>3) # Probability X is greater than 3
1/2
>>> E(X+Y) # Expectation of the sum of two dice
7
>>> variance(X+Y) # Variance of the sum of two dice
35/6
>>> simplify(P(Z>1)) # Probability of Z being greater than 1
1/2 - erf(sqrt(2)/2)/2


One could also create custom distribution and define custom random variables
as follows:

1. If you want to create a Continuous Random Variable:

>>> from sympy.stats import ContinuousRV, P, E
>>> from sympy import exp, Symbol, Interval, oo
>>> x = Symbol('x')
>>> pdf = exp(-x) # pdf of the Continuous Distribution
>>> Z = ContinuousRV(x, pdf, set=Interval(0, oo))
>>> E(Z)
1
>>> P(Z > 5)
exp(-5)

1.1 To create an instance of Continuous Distribution:

>>> from sympy.stats import ContinuousDistributionHandmade
>>> from sympy import Lambda
>>> dist = ContinuousDistributionHandmade(Lambda(x, pdf), set=Interval(0, oo))
>>> dist.pdf(x)
exp(-x)

2. If you want to create a Discrete Random Variable:

>>> from sympy.stats import DiscreteRV, P, E
>>> from sympy import Symbol, S
>>> p = S(1)/2
>>> x = Symbol('x', integer=True, positive=True)
>>> pdf = p*(1 - p)**(x - 1)
>>> D = DiscreteRV(x, pdf, set=S.Naturals)
>>> E(D)
2
>>> P(D > 3)
1/8

2.1 To create an instance of Discrete Distribution:

>>> from sympy.stats import DiscreteDistributionHandmade
>>> from sympy import Lambda
>>> dist = DiscreteDistributionHandmade(Lambda(x, pdf), set=S.Naturals)
>>> dist.pdf(x)
2**(1 - x)/2

3. If you want to create a Finite Random Variable:

>>> from sympy.stats import FiniteRV, P, E
>>> from sympy import Rational, Eq
>>> pmf = {1: Rational(1, 3), 2: Rational(1, 6), 3: Rational(1, 4), 4: Rational(1, 4)}
>>> X = FiniteRV('X', pmf)
>>> E(X)
29/12
>>> P(X > 3)
1/4

3.1 To create an instance of Finite Distribution:

>>> from sympy.stats import FiniteDistributionHandmade
>>> dist = FiniteDistributionHandmade(pmf)
>>> dist.pmf(x)
Lambda(x, Piecewise((1/3, Eq(x, 1)), (1/6, Eq(x, 2)), (1/4, Eq(x, 3) | Eq(x, 4)), (0, True)))
"""

__all__ = [
    'P', 'E', 'H', 'density', 'where', 'given', 'sample', 'cdf','median',
    'characteristic_function', 'pspace', 'sample_iter', 'variance', 'std',
    'skewness', 'kurtosis', 'covariance', 'dependent', 'entropy', 'independent',
    'random_symbols', 'correlation', 'factorial_moment', 'moment', 'cmoment',
    'sampling_density', 'moment_generating_function', 'smoment', 'quantile',
    'coskewness', 'sample_stochastic_process',

    'FiniteRV', 'DiscreteUniform', 'Die', 'Bernoulli', 'Coin', 'Binomial',
    'BetaBinomial', 'Hypergeometric', 'Rademacher', 'IdealSoliton', 'RobustSoliton',
    'FiniteDistributionHandmade',

    'ContinuousRV', 'Arcsin', 'Benini', 'Beta', 'BetaNoncentral', 'BetaPrime',
    'BoundedPareto', 'Cauchy', 'Chi', 'ChiNoncentral', 'ChiSquared', 'Dagum', 'Davis', 'Erlang',
    'ExGaussian', 'Exponential', 'ExponentialPower', 'FDistribution',
    'FisherZ', 'Frechet', 'Gamma', 'GammaInverse', 'Gompertz', 'Gumbel',
    'Kumaraswamy', 'Laplace', 'Levy', 'Logistic','LogCauchy', 'LogLogistic', 'LogitNormal', 'LogNormal', 'Lomax',
    'Moyal', 'Maxwell', 'Nakagami', 'Normal', 'GaussianInverse', 'Pareto', 'PowerFunction',
    'QuadraticU', 'RaisedCosine', 'Rayleigh','Reciprocal', 'StudentT', 'ShiftedGompertz',
    'Trapezoidal', 'Triangular', 'Uniform', 'UniformSum', 'VonMises', 'Wald',
    'Weibull', 'WignerSemicircle', 'ContinuousDistributionHandmade',

    'FlorySchulz', 'Geometric','Hermite', 'Logarithmic', 'NegativeBinomial', 'Poisson', 'Skellam',
    'YuleSimon', 'Zeta', 'DiscreteRV', 'DiscreteDistributionHandmade',

    'JointRV', 'Dirichlet', 'GeneralizedMultivariateLogGamma',
    'GeneralizedMultivariateLogGammaOmega', 'Multinomial', 'MultivariateBeta',
    'MultivariateEwens', 'MultivariateT', 'NegativeMultinomial',
    'NormalGamma', 'MultivariateNormal', 'MultivariateLaplace', 'marginal_distribution',

    'StochasticProcess', 'DiscreteTimeStochasticProcess',
    'DiscreteMarkovChain', 'TransitionMatrixOf', 'StochasticStateSpaceOf',
    'GeneratorMatrixOf', 'ContinuousMarkovChain', 'BernoulliProcess',
    'PoissonProcess', 'WienerProcess', 'GammaProcess',

    'CircularEnsemble', 'CircularUnitaryEnsemble',
    'CircularOrthogonalEnsemble', 'CircularSymplecticEnsemble',
    'GaussianEnsemble', 'GaussianUnitaryEnsemble',
    'GaussianOrthogonalEnsemble', 'GaussianSymplecticEnsemble',
    'joint_eigen_distribution', 'JointEigenDistribution',
    'level_spacing_distribution',

    'MatrixGamma', 'Wishart', 'MatrixNormal', 'MatrixStudentT',

    'Probability', 'Expectation', 'Variance', 'Covariance', 'Moment',
    'CentralMoment',

    'ExpectationMatrix', 'VarianceMatrix', 'CrossCovarianceMatrix'

]
from .rv_interface import (P, E, H, density, where, given, sample, cdf, median,
        characteristic_function, pspace, sample_iter, variance, std, skewness,
        kurtosis, covariance, dependent, entropy, independent, random_symbols,
        correlation, factorial_moment, moment, cmoment, sampling_density,
        moment_generating_function, smoment, quantile, coskewness,
        sample_stochastic_process)

from .frv_types import (FiniteRV, DiscreteUniform, Die, Bernoulli, Coin,
        Binomial, BetaBinomial, Hypergeometric, Rademacher,
        FiniteDistributionHandmade, IdealSoliton, RobustSoliton)

from .crv_types import (ContinuousRV, Arcsin, Benini, Beta, BetaNoncentral,
        BetaPrime, BoundedPareto, Cauchy, Chi, ChiNoncentral, ChiSquared,
        Dagum, Davis, Erlang, ExGaussian, Exponential, ExponentialPower,
        FDistribution, FisherZ, Frechet, Gamma, GammaInverse, GaussianInverse,
        Gompertz, Gumbel, Kumaraswamy, Laplace, Levy, Logistic, LogCauchy,
        LogLogistic, LogitNormal, LogNormal, Lomax, Maxwell, Moyal, Nakagami,
        Normal, Pareto, QuadraticU, RaisedCosine, Rayleigh, Reciprocal,
        StudentT, PowerFunction, ShiftedGompertz, Trapezoidal, Triangular,
        Uniform, UniformSum, VonMises, Wald, Weibull, WignerSemicircle,
        ContinuousDistributionHandmade)

from .drv_types import (FlorySchulz, Geometric, Hermite, Logarithmic, NegativeBinomial, Poisson,
        Skellam, YuleSimon, Zeta, DiscreteRV, DiscreteDistributionHandmade)

from .joint_rv_types import (JointRV, Dirichlet,
        GeneralizedMultivariateLogGamma, GeneralizedMultivariateLogGammaOmega,
        Multinomial, MultivariateBeta, MultivariateEwens, MultivariateT,
        NegativeMultinomial, NormalGamma, MultivariateNormal, MultivariateLaplace,
        marginal_distribution)

from .stochastic_process_types import (StochasticProcess,
        DiscreteTimeStochasticProcess, DiscreteMarkovChain,
        TransitionMatrixOf, StochasticStateSpaceOf, GeneratorMatrixOf,
        ContinuousMarkovChain, BernoulliProcess, PoissonProcess, WienerProcess,
        GammaProcess)

from .random_matrix_models import (CircularEnsemble, CircularUnitaryEnsemble,
        CircularOrthogonalEnsemble, CircularSymplecticEnsemble,
        GaussianEnsemble, GaussianUnitaryEnsemble, GaussianOrthogonalEnsemble,
        GaussianSymplecticEnsemble, joint_eigen_distribution,
        JointEigenDistribution, level_spacing_distribution)

from .matrix_distributions import MatrixGamma, Wishart, MatrixNormal, MatrixStudentT

from .symbolic_probability import (Probability, Expectation, Variance,
        Covariance, Moment, CentralMoment)

from .symbolic_multivariate_probability import (ExpectationMatrix, VarianceMatrix,
        CrossCovarianceMatrix)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_unix_pipes.py ===
from __future__ import annotations

import errno
import os
import sys
from typing import TYPE_CHECKING

import trio

from ._abc import Stream
from ._util import ConflictDetector, final

if TYPE_CHECKING:
    from typing import Final as FinalType

assert not TYPE_CHECKING or sys.platform != "win32"

if os.name != "posix":
    # We raise an error here rather than gating the import in lowlevel.py
    # in order to keep jedi static analysis happy.
    raise ImportError

# XX TODO: is this a good number? who knows... it does match the default Linux
# pipe capacity though.
DEFAULT_RECEIVE_SIZE: FinalType = 65536


class _FdHolder:
    # This class holds onto a raw file descriptor, in non-blocking mode, and
    # is responsible for managing its lifecycle. In particular, it's
    # responsible for making sure it gets closed, and also for tracking
    # whether it's been closed.
    #
    # The way we track closure is to set the .fd field to -1, discarding the
    # original value. You might think that this is a strange idea, since it
    # overloads the same field to do two different things. Wouldn't it be more
    # natural to have a dedicated .closed field? But that would be more
    # error-prone. Fds are represented by small integers, and once an fd is
    # closed, its integer value may be reused immediately. If we accidentally
    # used the old fd after being closed, we might end up doing something to
    # another unrelated fd that happened to get assigned the same integer
    # value. By throwing away the integer value immediately, it becomes
    # impossible to make this mistake – we'll just get an EBADF.
    #
    # (This trick was copied from the stdlib socket module.)
    fd: int

    def __init__(self, fd: int) -> None:
        # make sure self.fd is always initialized to *something*, because even
        # if we error out here then __del__ will run and access it.
        self.fd = -1
        if not isinstance(fd, int):
            raise TypeError("file descriptor must be an int")
        self.fd = fd
        # Store original state, and ensure non-blocking mode is enabled
        self._original_is_blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)

    @property
    def closed(self) -> bool:
        return self.fd == -1

    def _raw_close(self) -> None:
        # This doesn't assume it's in a Trio context, so it can be called from
        # __del__. You should never call it from Trio context, because it
        # skips calling notify_fd_close. But from __del__, skipping that is
        # OK, because notify_fd_close just wakes up other tasks that are
        # waiting on this fd, and those tasks hold a reference to this object.
        # So if __del__ is being called, we know there aren't any tasks that
        # need to be woken.
        if self.closed:
            return
        fd = self.fd
        self.fd = -1
        os.set_blocking(fd, self._original_is_blocking)
        os.close(fd)

    def __del__(self) -> None:
        self._raw_close()

    def close(self) -> None:
        if not self.closed:
            trio.lowlevel.notify_closing(self.fd)
            self._raw_close()


@final
class FdStream(Stream):
    """
    Represents a stream given the file descriptor to a pipe, TTY, etc.

    *fd* must refer to a file that is open for reading and/or writing and
    supports non-blocking I/O (pipes and TTYs will work, on-disk files probably
    not).  The returned stream takes ownership of the fd, so closing the stream
    will close the fd too.  As with `os.fdopen`, you should not directly use
    an fd after you have wrapped it in a stream using this function.

    To be used as a Trio stream, an open file must be placed in non-blocking
    mode.  Unfortunately, this impacts all I/O that goes through the
    underlying open file, including I/O that uses a different
    file descriptor than the one that was passed to Trio. If other threads
    or processes are using file descriptors that are related through `os.dup`
    or inheritance across `os.fork` to the one that Trio is using, they are
    unlikely to be prepared to have non-blocking I/O semantics suddenly
    thrust upon them.  For example, you can use
    ``FdStream(os.dup(sys.stdin.fileno()))`` to obtain a stream for reading
    from standard input, but it is only safe to do so with heavy caveats: your
    stdin must not be shared by any other processes, and you must not make any
    calls to synchronous methods of `sys.stdin` until the stream returned by
    `FdStream` is closed. See `issue #174
    <https://github.com/python-trio/trio/issues/174>`__ for a discussion of the
    challenges involved in relaxing this restriction.

    Args:
      fd (int): The fd to be wrapped.

    Returns:
      A new `FdStream` object.
    """

    def __init__(self, fd: int) -> None:
        self._fd_holder = _FdHolder(fd)
        self._send_conflict_detector = ConflictDetector(
            "another task is using this stream for send",
        )
        self._receive_conflict_detector = ConflictDetector(
            "another task is using this stream for receive",
        )

    async def send_all(self, data: bytes) -> None:
        with self._send_conflict_detector:
            # have to check up front, because send_all(b"") on a closed pipe
            # should raise
            if self._fd_holder.closed:
                raise trio.ClosedResourceError("file was already closed")
            await trio.lowlevel.checkpoint()
            length = len(data)
            # adapted from the SocketStream code
            with memoryview(data) as view:
                sent = 0
                while sent < length:
                    with view[sent:] as remaining:
                        try:
                            sent += os.write(self._fd_holder.fd, remaining)
                        except BlockingIOError:
                            await trio.lowlevel.wait_writable(self._fd_holder.fd)
                        except OSError as e:
                            if e.errno == errno.EBADF:
                                raise trio.ClosedResourceError(
                                    "file was already closed",
                                ) from None
                            else:
                                raise trio.BrokenResourceError from e

    async def wait_send_all_might_not_block(self) -> None:
        with self._send_conflict_detector:
            if self._fd_holder.closed:
                raise trio.ClosedResourceError("file was already closed")
            try:
                await trio.lowlevel.wait_writable(self._fd_holder.fd)
            except BrokenPipeError as e:
                # kqueue: raises EPIPE on wait_writable instead
                # of sending, which is annoying
                raise trio.BrokenResourceError from e

    async def receive_some(self, max_bytes: int | None = None) -> bytes:
        with self._receive_conflict_detector:
            if max_bytes is None:
                max_bytes = DEFAULT_RECEIVE_SIZE
            else:
                if not isinstance(max_bytes, int):
                    raise TypeError("max_bytes must be integer >= 1")
                if max_bytes < 1:
                    raise ValueError("max_bytes must be integer >= 1")

            await trio.lowlevel.checkpoint()
            while True:
                try:
                    data = os.read(self._fd_holder.fd, max_bytes)
                except BlockingIOError:
                    await trio.lowlevel.wait_readable(self._fd_holder.fd)
                except OSError as exc:
                    if exc.errno == errno.EBADF:
                        raise trio.ClosedResourceError(
                            "file was already closed",
                        ) from None
                    else:
                        raise trio.BrokenResourceError from exc
                else:
                    break

            return data

    def close(self) -> None:
        self._fd_holder.close()

    async def aclose(self) -> None:
        self.close()
        await trio.lowlevel.checkpoint()

    def fileno(self) -> int:
        return self._fd_holder.fd

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_handshake.py ===
"""
_handshake.py
websocket - WebSocket client library for Python

Copyright 2024 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import hashlib
import hmac
import os
from base64 import encodebytes as base64encode
from http import HTTPStatus

from ._cookiejar import SimpleCookieJar
from ._exceptions import WebSocketException, WebSocketBadStatusException
from ._http import read_headers
from ._logging import dump, error
from ._socket import send

__all__ = ["handshake_response", "handshake", "SUPPORTED_REDIRECT_STATUSES"]

# websocket supported version.
VERSION = 13

SUPPORTED_REDIRECT_STATUSES = (
    HTTPStatus.MOVED_PERMANENTLY,
    HTTPStatus.FOUND,
    HTTPStatus.SEE_OTHER,
    HTTPStatus.TEMPORARY_REDIRECT,
    HTTPStatus.PERMANENT_REDIRECT,
)
SUCCESS_STATUSES = SUPPORTED_REDIRECT_STATUSES + (HTTPStatus.SWITCHING_PROTOCOLS,)

CookieJar = SimpleCookieJar()


class handshake_response:
    def __init__(self, status: int, headers: dict, subprotocol):
        self.status = status
        self.headers = headers
        self.subprotocol = subprotocol
        CookieJar.add(headers.get("set-cookie"))


def handshake(
    sock, url: str, hostname: str, port: int, resource: str, **options
) -> handshake_response:
    headers, key = _get_handshake_headers(resource, url, hostname, port, options)

    header_str = "\r\n".join(headers)
    send(sock, header_str)
    dump("request header", header_str)

    status, resp = _get_resp_headers(sock)
    if status in SUPPORTED_REDIRECT_STATUSES:
        return handshake_response(status, resp, None)
    success, subproto = _validate(resp, key, options.get("subprotocols"))
    if not success:
        raise WebSocketException("Invalid WebSocket Header")

    return handshake_response(status, resp, subproto)


def _pack_hostname(hostname: str) -> str:
    # IPv6 address
    if ":" in hostname:
        return f"[{hostname}]"
    return hostname


def _get_handshake_headers(
    resource: str, url: str, host: str, port: int, options: dict
) -> tuple:
    headers = [f"GET {resource} HTTP/1.1", "Upgrade: websocket"]
    if port in [80, 443]:
        hostport = _pack_hostname(host)
    else:
        hostport = f"{_pack_hostname(host)}:{port}"
    if options.get("host"):
        headers.append(f'Host: {options["host"]}')
    else:
        headers.append(f"Host: {hostport}")

    # scheme indicates whether http or https is used in Origin
    # The same approach is used in parse_url of _url.py to set default port
    scheme, url = url.split(":", 1)
    if not options.get("suppress_origin"):
        if "origin" in options and options["origin"] is not None:
            headers.append(f'Origin: {options["origin"]}')
        elif scheme == "wss":
            headers.append(f"Origin: https://{hostport}")
        else:
            headers.append(f"Origin: http://{hostport}")

    key = _create_sec_websocket_key()

    # Append Sec-WebSocket-Key & Sec-WebSocket-Version if not manually specified
    if not options.get("header") or "Sec-WebSocket-Key" not in options["header"]:
        headers.append(f"Sec-WebSocket-Key: {key}")
    else:
        key = options["header"]["Sec-WebSocket-Key"]

    if not options.get("header") or "Sec-WebSocket-Version" not in options["header"]:
        headers.append(f"Sec-WebSocket-Version: {VERSION}")

    if not options.get("connection"):
        headers.append("Connection: Upgrade")
    else:
        headers.append(options["connection"])

    if subprotocols := options.get("subprotocols"):
        headers.append(f'Sec-WebSocket-Protocol: {",".join(subprotocols)}')

    if header := options.get("header"):
        if isinstance(header, dict):
            header = [": ".join([k, v]) for k, v in header.items() if v is not None]
        headers.extend(header)

    server_cookie = CookieJar.get(host)
    client_cookie = options.get("cookie", None)

    if cookie := "; ".join(filter(None, [server_cookie, client_cookie])):
        headers.append(f"Cookie: {cookie}")

    headers.extend(("", ""))
    return headers, key


def _get_resp_headers(sock, success_statuses: tuple = SUCCESS_STATUSES) -> tuple:
    status, resp_headers, status_message = read_headers(sock)
    if status not in success_statuses:
        content_len = resp_headers.get("content-length")
        if content_len:
            response_body = sock.recv(
                int(content_len)
            )  # read the body of the HTTP error message response and include it in the exception
        else:
            response_body = None
        raise WebSocketBadStatusException(
            f"Handshake status {status} {status_message} -+-+- {resp_headers} -+-+- {response_body}",
            status,
            status_message,
            resp_headers,
            response_body,
        )
    return status, resp_headers


_HEADERS_TO_CHECK = {
    "upgrade": "websocket",
    "connection": "upgrade",
}


def _validate(headers, key: str, subprotocols) -> tuple:
    subproto = None
    for k, v in _HEADERS_TO_CHECK.items():
        r = headers.get(k, None)
        if not r:
            return False, None
        r = [x.strip().lower() for x in r.split(",")]
        if v not in r:
            return False, None

    if subprotocols:
        subproto = headers.get("sec-websocket-protocol", None)
        if not subproto or subproto.lower() not in [s.lower() for s in subprotocols]:
            error(f"Invalid subprotocol: {subprotocols}")
            return False, None
        subproto = subproto.lower()

    result = headers.get("sec-websocket-accept", None)
    if not result:
        return False, None
    result = result.lower()

    if isinstance(result, str):
        result = result.encode("utf-8")

    value = f"{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11".encode("utf-8")
    hashed = base64encode(hashlib.sha1(value).digest()).strip().lower()

    if hmac.compare_digest(hashed, result):
        return True, subproto
    else:
        return False, None


def _create_sec_websocket_key() -> str:
    randomness = os.urandom(16)
    return base64encode(randomness).decode("utf-8").strip()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\routing\matcher.py ===
from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass
from dataclasses import field

from .converters import ValidationError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .rules import Rule
from .rules import RulePart


class SlashRequired(Exception):
    pass


@dataclass
class State:
    """A representation of a rule state.

    This includes the *rules* that correspond to the state and the
    possible *static* and *dynamic* transitions to the next state.
    """

    dynamic: list[tuple[RulePart, State]] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    static: dict[str, State] = field(default_factory=dict)


class StateMachineMatcher:
    def __init__(self, merge_slashes: bool) -> None:
        self._root = State()
        self.merge_slashes = merge_slashes

    def add(self, rule: Rule) -> None:
        state = self._root
        for part in rule._parts:
            if part.static:
                state.static.setdefault(part.content, State())
                state = state.static[part.content]
            else:
                for test_part, new_state in state.dynamic:
                    if test_part == part:
                        state = new_state
                        break
                else:
                    new_state = State()
                    state.dynamic.append((part, new_state))
                    state = new_state
        state.rules.append(rule)

    def update(self) -> None:
        # For every state the dynamic transitions should be sorted by
        # the weight of the transition
        state = self._root

        def _update_state(state: State) -> None:
            state.dynamic.sort(key=lambda entry: entry[0].weight)
            for new_state in state.static.values():
                _update_state(new_state)
            for _, new_state in state.dynamic:
                _update_state(new_state)

        _update_state(state)

    def match(
        self, domain: str, path: str, method: str, websocket: bool
    ) -> tuple[Rule, t.MutableMapping[str, t.Any]]:
        # To match to a rule we need to start at the root state and
        # try to follow the transitions until we find a match, or find
        # there is no transition to follow.

        have_match_for = set()
        websocket_mismatch = False

        def _match(
            state: State, parts: list[str], values: list[str]
        ) -> tuple[Rule, list[str]] | None:
            # This function is meant to be called recursively, and will attempt
            # to match the head part to the state's transitions.
            nonlocal have_match_for, websocket_mismatch

            # The base case is when all parts have been matched via
            # transitions. Hence if there is a rule with methods &
            # websocket that work return it and the dynamic values
            # extracted.
            if parts == []:
                for rule in state.rules:
                    if rule.methods is not None and method not in rule.methods:
                        have_match_for.update(rule.methods)
                    elif rule.websocket != websocket:
                        websocket_mismatch = True
                    else:
                        return rule, values

                # Test if there is a match with this path with a
                # trailing slash, if so raise an exception to report
                # that matching is possible with an additional slash
                if "" in state.static:
                    for rule in state.static[""].rules:
                        if websocket == rule.websocket and (
                            rule.methods is None or method in rule.methods
                        ):
                            if rule.strict_slashes:
                                raise SlashRequired()
                            else:
                                return rule, values
                return None

            part = parts[0]
            # To match this part try the static transitions first
            if part in state.static:
                rv = _match(state.static[part], parts[1:], values)
                if rv is not None:
                    return rv
            # No match via the static transitions, so try the dynamic
            # ones.
            for test_part, new_state in state.dynamic:
                target = part
                remaining = parts[1:]
                # A final part indicates a transition that always
                # consumes the remaining parts i.e. transitions to a
                # final state.
                if test_part.final:
                    target = "/".join(parts)
                    remaining = []
                match = re.compile(test_part.content).match(target)
                if match is not None:
                    if test_part.suffixed:
                        # If a part_isolating=False part has a slash suffix, remove the
                        # suffix from the match and check for the slash redirect next.
                        suffix = match.groups()[-1]
                        if suffix == "/":
                            remaining = [""]

                    converter_groups = sorted(
                        match.groupdict().items(), key=lambda entry: entry[0]
                    )
                    groups = [
                        value
                        for key, value in converter_groups
                        if key[:11] == "__werkzeug_"
                    ]
                    rv = _match(new_state, remaining, values + groups)
                    if rv is not None:
                        return rv

            # If there is no match and the only part left is a
            # trailing slash ("") consider rules that aren't
            # strict-slashes as these should match if there is a final
            # slash part.
            if parts == [""]:
                for rule in state.rules:
                    if rule.strict_slashes:
                        continue
                    if rule.methods is not None and method not in rule.methods:
                        have_match_for.update(rule.methods)
                    elif rule.websocket != websocket:
                        websocket_mismatch = True
                    else:
                        return rule, values

            return None

        try:
            rv = _match(self._root, [domain, *path.split("/")], [])
        except SlashRequired:
            raise RequestPath(f"{path}/") from None

        if self.merge_slashes and rv is None:
            # Try to match again, but with slashes merged
            path = re.sub("/{2,}?", "/", path)
            try:
                rv = _match(self._root, [domain, *path.split("/")], [])
            except SlashRequired:
                raise RequestPath(f"{path}/") from None
            if rv is None or rv[0].merge_slashes is False:
                raise NoMatch(have_match_for, websocket_mismatch)
            else:
                raise RequestPath(f"{path}")
        elif rv is not None:
            rule, values = rv

            result = {}
            for name, value in zip(rule._converters.keys(), values):
                try:
                    value = rule._converters[name].to_python(value)
                except ValidationError:
                    raise NoMatch(have_match_for, websocket_mismatch) from None
                result[str(name)] = value
            if rule.defaults:
                result.update(rule.defaults)

            if rule.alias and rule.map.redirect_defaults:
                raise RequestAliasRedirect(result, rule.endpoint)

            return rule, result

        raise NoMatch(have_match_for, websocket_mismatch)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\fields\list.py ===
import itertools

from wtforms.utils import unset_value

from .. import widgets
from .core import Field
from .core import UnboundField

__all__ = ("FieldList",)


class FieldList(Field):
    """
    Encapsulate an ordered list of multiple instances of the same field type,
    keeping data as a list.

    >>> authors = FieldList(StringField('Name', [validators.DataRequired()]))

    :param unbound_field:
        A partially-instantiated field definition, just like that would be
        defined on a form directly.
    :param min_entries:
        if provided, always have at least this many entries on the field,
        creating blank ones if the provided input does not specify a sufficient
        amount.
    :param max_entries:
        accept no more than this many entries as input, even if more exist in
        formdata.
    :param separator:
        A string which will be suffixed to this field's name to create the
        prefix to enclosed list entries. The default is fine for most uses.
    """

    widget = widgets.ListWidget()

    def __init__(
        self,
        unbound_field,
        label=None,
        validators=None,
        min_entries=0,
        max_entries=None,
        separator="-",
        default=(),
        **kwargs,
    ):
        super().__init__(label, validators, default=default, **kwargs)
        if self.filters:
            raise TypeError(
                "FieldList does not accept any filters. Instead, define"
                " them on the enclosed field."
            )
        assert isinstance(
            unbound_field, UnboundField
        ), "Field must be unbound, not a field class"
        self.unbound_field = unbound_field
        self.min_entries = min_entries
        self.max_entries = max_entries
        self.last_index = -1
        self._prefix = kwargs.get("_prefix", "")
        self._separator = separator
        self._field_separator = unbound_field.kwargs.get("separator", "-")

    def process(self, formdata, data=unset_value, extra_filters=None):
        if extra_filters:
            raise TypeError(
                "FieldList does not accept any filters. Instead, define"
                " them on the enclosed field."
            )

        self.entries = []
        if data is unset_value or not data:
            try:
                data = self.default()
            except TypeError:
                data = self.default

        self.object_data = data

        if formdata:
            indices = sorted(set(self._extract_indices(self.name, formdata)))
            if self.max_entries:
                indices = indices[: self.max_entries]

            idata = iter(data)
            for index in indices:
                try:
                    obj_data = next(idata)
                except StopIteration:
                    obj_data = unset_value
                self._add_entry(formdata, obj_data, index=index)
        else:
            for obj_data in data:
                self._add_entry(formdata, obj_data)

        while len(self.entries) < self.min_entries:
            self._add_entry(formdata)

    def _extract_indices(self, prefix, formdata):
        """
        Yield indices of any keys with given prefix.

        formdata must be an object which will produce keys when iterated.  For
        example, if field 'foo' contains keys 'foo-0-bar', 'foo-1-baz', then
        the numbers 0 and 1 will be yielded, but not necessarily in order.
        """
        offset = len(prefix) + 1
        for k in formdata:
            if k.startswith(prefix):
                k = k[offset:].split(self._field_separator, 1)[0]
                if k.isdigit():
                    yield int(k)

    def validate(self, form, extra_validators=()):
        """
        Validate this FieldList.

        Note that FieldList validation differs from normal field validation in
        that FieldList validates all its enclosed fields first before running any
        of its own validators.
        """
        self.errors = []

        # Run validators on all entries within
        for subfield in self.entries:
            subfield.validate(form)
            self.errors.append(subfield.errors)

        if not any(x for x in self.errors):
            self.errors = []

        chain = itertools.chain(self.validators, extra_validators)
        self._run_validation_chain(form, chain)

        return len(self.errors) == 0

    def populate_obj(self, obj, name):
        values = getattr(obj, name, None)
        try:
            ivalues = iter(values)
        except TypeError:
            ivalues = iter([])

        candidates = itertools.chain(ivalues, itertools.repeat(None))
        _fake = type("_fake", (object,), {})
        output = []
        for field, data in zip(self.entries, candidates):
            fake_obj = _fake()
            fake_obj.data = data
            field.populate_obj(fake_obj, "data")
            output.append(fake_obj.data)

        setattr(obj, name, output)

    def _add_entry(self, formdata=None, data=unset_value, index=None):
        assert (
            not self.max_entries or len(self.entries) < self.max_entries
        ), "You cannot have more than max_entries entries in this FieldList"
        if index is None:
            index = self.last_index + 1
        self.last_index = index
        name = f"{self.short_name}{self._separator}{index}"
        id = f"{self.id}{self._separator}{index}"
        field = self.unbound_field.bind(
            form=None,
            name=name,
            prefix=self._prefix,
            id=id,
            _meta=self.meta,
            translations=self._translations,
        )
        field.process(formdata, data)
        self.entries.append(field)
        return field

    def append_entry(self, data=unset_value):
        """
        Create a new entry with optional default data.

        Entries added in this way will *not* receive formdata however, and can
        only receive object data.
        """
        return self._add_entry(data=data)

    def pop_entry(self):
        """Removes the last entry from the list and returns it."""
        entry = self.entries.pop()
        self.last_index -= 1
        return entry

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    @property
    def data(self):
        return [f.data for f in self.entries]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_pytesttester.py ===
"""
Pytest test running.

This module implements the ``test()`` function for NumPy modules. The usual
boiler plate for doing that is to put the following in the module
``__init__.py`` file::

    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester


Warnings filtering and other runtime settings should be dealt with in the
``pytest.ini`` file in the numpy repo root. The behavior of the test depends on
whether or not that file is found as follows:

* ``pytest.ini`` is present (develop mode)
    All warnings except those explicitly filtered out are raised as error.
* ``pytest.ini`` is absent (release mode)
    DeprecationWarnings and PendingDeprecationWarnings are ignored, other
    warnings are passed through.

In practice, tests run from the numpy repo are run in development mode with
``spin``, through the standard ``spin test`` invocation or from an inplace
build with ``pytest numpy``.

This module is imported by every numpy subpackage, so lies at the top level to
simplify circular import issues. For the same reason, it contains no numpy
imports at module scope, instead importing numpy within function calls.
"""
import os
import sys

__all__ = ['PytestTester']


def _show_numpy_info():
    import numpy as np

    print(f"NumPy version {np.__version__}")
    info = np.lib._utils_impl._opt_info()
    print("NumPy CPU features: ", (info or 'nothing enabled'))


class PytestTester:
    """
    Pytest test runner.

    A test function is typically added to a package's __init__.py like so::

      from numpy._pytesttester import PytestTester
      test = PytestTester(__name__).test
      del PytestTester

    Calling this test function finds and runs all tests associated with the
    module and all its sub-modules.

    Attributes
    ----------
    module_name : str
        Full path to the package to test.

    Parameters
    ----------
    module_name : module name
        The name of the module to test.

    Notes
    -----
    Unlike the previous ``nose``-based implementation, this class is not
    publicly exposed as it performs some ``numpy``-specific warning
    suppression.

    """
    def __init__(self, module_name):
        self.module_name = module_name
        self.__module__ = module_name

    def __call__(self, label='fast', verbose=1, extra_argv=None,
                 doctests=False, coverage=False, durations=-1, tests=None):
        """
        Run tests for module using pytest.

        Parameters
        ----------
        label : {'fast', 'full'}, optional
            Identifies the tests to run. When set to 'fast', tests decorated
            with `pytest.mark.slow` are skipped, when 'full', the slow marker
            is ignored.
        verbose : int, optional
            Verbosity value for test outputs, in the range 1-3. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to pytests.
        doctests : bool, optional
            .. note:: Not supported
        coverage : bool, optional
            If True, report coverage of NumPy code. Default is False.
            Requires installation of (pip) pytest-cov.
        durations : int, optional
            If < 0, do nothing, If 0, report time of all tests, if > 0,
            report the time of the slowest `timer` tests. Default is -1.
        tests : test or list of tests
            Tests to be executed with pytest '--pyargs'

        Returns
        -------
        result : bool
            Return True on success, false otherwise.

        Notes
        -----
        Each NumPy module exposes `test` in its namespace to run all tests for
        it. For example, to run all tests for numpy.lib:

        >>> np.lib.test() #doctest: +SKIP

        Examples
        --------
        >>> result = np.lib.test() #doctest: +SKIP
        ...
        1023 passed, 2 skipped, 6 deselected, 1 xfailed in 10.39 seconds
        >>> result
        True

        """
        import warnings

        import pytest

        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])

        # setup the pytest arguments
        pytest_args = ["-l"]

        # offset verbosity. The "-q" cancels a "-v".
        pytest_args += ["-q"]

        if sys.version_info < (3, 12):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                # Filter out distutils cpu warnings (could be localized to
                # distutils tests). ASV has problems with top level import,
                # so fetch module for suppression here.
                from numpy.distutils import cpuinfo  # noqa: F401

        # Filter out annoying import messages. Want these in both develop and
        # release mode.
        pytest_args += [
            "-W ignore:Not importing directory",
            "-W ignore:numpy.dtype size changed",
            "-W ignore:numpy.ufunc size changed",
            "-W ignore::UserWarning:cpuinfo",
            ]

        # When testing matrices, ignore their PendingDeprecationWarnings
        pytest_args += [
            "-W ignore:the matrix subclass is not",
            "-W ignore:Importing from numpy.matlib is",
            ]

        if doctests:
            pytest_args += ["--doctest-modules"]

        if extra_argv:
            pytest_args += list(extra_argv)

        if verbose > 1:
            pytest_args += ["-" + "v" * (verbose - 1)]

        if coverage:
            pytest_args += ["--cov=" + module_path]

        if label == "fast":
            # not importing at the top level to avoid circular import of module
            from numpy.testing import IS_PYPY
            if IS_PYPY:
                pytest_args += ["-m", "not slow and not slow_pypy"]
            else:
                pytest_args += ["-m", "not slow"]

        elif label != "full":
            pytest_args += ["-m", label]

        if durations >= 0:
            pytest_args += [f"--durations={durations}"]

        if tests is None:
            tests = [self.module_name]

        pytest_args += ["--pyargs"] + list(tests)

        # run tests.
        _show_numpy_info()

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return code == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\__init__.py ===
"""
============================
Typing (:mod:`numpy.typing`)
============================

.. versionadded:: 1.20

Large parts of the NumPy API have :pep:`484`-style type annotations. In
addition a number of type aliases are available to users, most prominently
the two below:

- `ArrayLike`: objects that can be converted to arrays
- `DTypeLike`: objects that can be converted to dtypes

.. _typing-extensions: https://pypi.org/project/typing-extensions/

Mypy plugin
-----------

.. versionadded:: 1.21

.. automodule:: numpy.typing.mypy_plugin

.. currentmodule:: numpy.typing

Differences from the runtime NumPy API
--------------------------------------

NumPy is very flexible. Trying to describe the full range of
possibilities statically would result in types that are not very
helpful. For that reason, the typed NumPy API is often stricter than
the runtime NumPy API. This section describes some notable
differences.

ArrayLike
~~~~~~~~~

The `ArrayLike` type tries to avoid creating object arrays. For
example,

.. code-block:: python

    >>> np.array(x**2 for x in range(10))
    array(<generator object <genexpr> at ...>, dtype=object)

is valid NumPy code which will create a 0-dimensional object
array. Type checkers will complain about the above example when using
the NumPy types however. If you really intended to do the above, then
you can either use a ``# type: ignore`` comment:

.. code-block:: python

    >>> np.array(x**2 for x in range(10))  # type: ignore

or explicitly type the array like object as `~typing.Any`:

.. code-block:: python

    >>> from typing import Any
    >>> array_like: Any = (x**2 for x in range(10))
    >>> np.array(array_like)
    array(<generator object <genexpr> at ...>, dtype=object)

ndarray
~~~~~~~

It's possible to mutate the dtype of an array at runtime. For example,
the following code is valid:

.. code-block:: python

    >>> x = np.array([1, 2])
    >>> x.dtype = np.bool

This sort of mutation is not allowed by the types. Users who want to
write statically typed code should instead use the `numpy.ndarray.view`
method to create a view of the array with a different dtype.

DTypeLike
~~~~~~~~~

The `DTypeLike` type tries to avoid creation of dtype objects using
dictionary of fields like below:

.. code-block:: python

    >>> x = np.dtype({"field1": (float, 1), "field2": (int, 3)})

Although this is valid NumPy code, the type checker will complain about it,
since its usage is discouraged.
Please see : :ref:`Data type objects <arrays.dtypes>`

Number precision
~~~~~~~~~~~~~~~~

The precision of `numpy.number` subclasses is treated as a invariant generic
parameter (see :class:`~NBitBase`), simplifying the annotating of processes
involving precision-based casting.

.. code-block:: python

    >>> from typing import TypeVar
    >>> import numpy as np
    >>> import numpy.typing as npt

    >>> T = TypeVar("T", bound=npt.NBitBase)
    >>> def func(a: "np.floating[T]", b: "np.floating[T]") -> "np.floating[T]":
    ...     ...

Consequently, the likes of `~numpy.float16`, `~numpy.float32` and
`~numpy.float64` are still sub-types of `~numpy.floating`, but, contrary to
runtime, they're not necessarily considered as sub-classes.

Timedelta64
~~~~~~~~~~~

The `~numpy.timedelta64` class is not considered a subclass of
`~numpy.signedinteger`, the former only inheriting from `~numpy.generic`
while static type checking.

0D arrays
~~~~~~~~~

During runtime numpy aggressively casts any passed 0D arrays into their
corresponding `~numpy.generic` instance. Until the introduction of shape
typing (see :pep:`646`) it is unfortunately not possible to make the
necessary distinction between 0D and >0D arrays. While thus not strictly
correct, all operations that can potentially perform a 0D-array -> scalar
cast are currently annotated as exclusively returning an `~numpy.ndarray`.

If it is known in advance that an operation *will* perform a
0D-array -> scalar cast, then one can consider manually remedying the
situation with either `typing.cast` or a ``# type: ignore`` comment.

Record array dtypes
~~~~~~~~~~~~~~~~~~~

The dtype of `numpy.recarray`, and the :ref:`routines.array-creation.rec`
functions in general, can be specified in one of two ways:

* Directly via the ``dtype`` argument.
* With up to five helper arguments that operate via `numpy.rec.format_parser`:
  ``formats``, ``names``, ``titles``, ``aligned`` and ``byteorder``.

These two approaches are currently typed as being mutually exclusive,
*i.e.* if ``dtype`` is specified than one may not specify ``formats``.
While this mutual exclusivity is not (strictly) enforced during runtime,
combining both dtype specifiers can lead to unexpected or even downright
buggy behavior.

API
---

"""
# NOTE: The API section will be appended with additional entries
# further down in this file

# pyright: reportDeprecated=false

from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray

__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]


__DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
__DIR_SET = frozenset(__DIR)


def __dir__() -> list[str]:
    return __DIR

def __getattr__(name: str):
    if name == "NBitBase":
        import warnings

        # Deprecated in NumPy 2.3, 2025-05-01
        warnings.warn(
            "`NBitBase` is deprecated and will be removed from numpy.typing in the "
            "future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper "
            "bound, instead. (deprecated in NumPy 2.3)",
            DeprecationWarning,
            stacklevel=2,
        )
        return NBitBase

    if name in __DIR_SET:
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __doc__ is not None:
    from numpy._typing._add_docstring import _docstrings
    __doc__ += _docstrings
    __doc__ += '\n.. autoclass:: numpy.typing.NBitBase\n'
    del _docstrings

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\array_algos\masked_reductions.py ===
"""
masked_reductions.py is for reduction algorithms using a mask-based approach
for missing values.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
)
import warnings

import numpy as np

from pandas._libs import missing as libmissing

from pandas.core.nanops import check_below_min_count

if TYPE_CHECKING:
    from pandas._typing import (
        AxisInt,
        npt,
    )


def _reductions(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
    **kwargs,
):
    """
    Sum, mean or product for 1D masked array.

    Parameters
    ----------
    func : np.sum or np.prod
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    min_count : int, default 0
        The required number of valid values to perform the operation. If fewer than
        ``min_count`` non-NA values are present the result will be NA.
    axis : int, optional, default None
    """
    if not skipna:
        if mask.any() or check_below_min_count(values.shape, None, min_count):
            return libmissing.NA
        else:
            return func(values, axis=axis, **kwargs)
    else:
        if check_below_min_count(values.shape, mask, min_count) and (
            axis is None or values.ndim == 1
        ):
            return libmissing.NA

        if values.dtype == np.dtype(object):
            # object dtype does not support `where` without passing an initial
            values = values[~mask]
            return func(values, axis=axis, **kwargs)
        return func(values, where=~mask, axis=axis, **kwargs)


def sum(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
):
    return _reductions(
        np.sum, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis
    )


def prod(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
):
    return _reductions(
        np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis
    )


def _minmax(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    """
    Reduction for 1D masked array.

    Parameters
    ----------
    func : np.min or np.max
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    axis : int, optional, default None
    """
    if not skipna:
        if mask.any() or not values.size:
            # min/max with empty array raise in numpy, pandas returns NA
            return libmissing.NA
        else:
            return func(values, axis=axis)
    else:
        subset = values[~mask]
        if subset.size:
            return func(subset, axis=axis)
        else:
            # min/max with empty array raise in numpy, pandas returns NA
            return libmissing.NA


def min(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    return _minmax(np.min, values=values, mask=mask, skipna=skipna, axis=axis)


def max(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    return _minmax(np.max, values=values, mask=mask, skipna=skipna, axis=axis)


def mean(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    if not values.size or mask.all():
        return libmissing.NA
    return _reductions(np.mean, values=values, mask=mask, skipna=skipna, axis=axis)


def var(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
    ddof: int = 1,
):
    if not values.size or mask.all():
        return libmissing.NA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _reductions(
            np.var, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof
        )


def std(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
    ddof: int = 1,
):
    if not values.size or mask.all():
        return libmissing.NA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _reductions(
            np.std, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\debug.py ===
import locale
import logging
import os
import sys
from optparse import Values
from types import ModuleType
from typing import Any, Dict, List, Optional

import pip._vendor
from pip._vendor.certifi import where
from pip._vendor.packaging.version import parse as parse_version

from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.configuration import Configuration
from pip._internal.metadata import get_environment
from pip._internal.utils.compat import open_text_resource
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_pip_version

logger = logging.getLogger(__name__)


def show_value(name: str, value: Any) -> None:
    logger.info("%s: %s", name, value)


def show_sys_implementation() -> None:
    logger.info("sys.implementation:")
    implementation_name = sys.implementation.name
    with indent_log():
        show_value("name", implementation_name)


def create_vendor_txt_map() -> Dict[str, str]:
    with open_text_resource("pip._vendor", "vendor.txt") as f:
        # Purge non version specifying lines.
        # Also, remove any space prefix or suffixes (including comments).
        lines = [
            line.strip().split(" ", 1)[0] for line in f.readlines() if "==" in line
        ]

    # Transform into "module" -> version dict.
    return dict(line.split("==", 1) for line in lines)


def get_module_from_module_name(module_name: str) -> Optional[ModuleType]:
    # Module name can be uppercase in vendor.txt for some reason...
    module_name = module_name.lower().replace("-", "_")
    # PATCH: setuptools is actually only pkg_resources.
    if module_name == "setuptools":
        module_name = "pkg_resources"

    try:
        __import__(f"pip._vendor.{module_name}", globals(), locals(), level=0)
        return getattr(pip._vendor, module_name)
    except ImportError:
        # We allow 'truststore' to fail to import due
        # to being unavailable on Python 3.9 and earlier.
        if module_name == "truststore" and sys.version_info < (3, 10):
            return None
        raise


def get_vendor_version_from_module(module_name: str) -> Optional[str]:
    module = get_module_from_module_name(module_name)
    version = getattr(module, "__version__", None)

    if module and not version:
        # Try to find version in debundled module info.
        assert module.__file__ is not None
        env = get_environment([os.path.dirname(module.__file__)])
        dist = env.get_distribution(module_name)
        if dist:
            version = str(dist.version)

    return version


def show_actual_vendor_versions(vendor_txt_versions: Dict[str, str]) -> None:
    """Log the actual version and print extra info if there is
    a conflict or if the actual version could not be imported.
    """
    for module_name, expected_version in vendor_txt_versions.items():
        extra_message = ""
        actual_version = get_vendor_version_from_module(module_name)
        if not actual_version:
            extra_message = (
                " (Unable to locate actual module version, using"
                " vendor.txt specified version)"
            )
            actual_version = expected_version
        elif parse_version(actual_version) != parse_version(expected_version):
            extra_message = (
                " (CONFLICT: vendor.txt suggests version should"
                f" be {expected_version})"
            )
        logger.info("%s==%s%s", module_name, actual_version, extra_message)


def show_vendor_versions() -> None:
    logger.info("vendored library versions:")

    vendor_txt_versions = create_vendor_txt_map()
    with indent_log():
        show_actual_vendor_versions(vendor_txt_versions)


def show_tags(options: Values) -> None:
    tag_limit = 10

    target_python = make_target_python(options)
    tags = target_python.get_sorted_tags()

    # Display the target options that were explicitly provided.
    formatted_target = target_python.format_given()
    suffix = ""
    if formatted_target:
        suffix = f" (target: {formatted_target})"

    msg = f"Compatible tags: {len(tags)}{suffix}"
    logger.info(msg)

    if options.verbose < 1 and len(tags) > tag_limit:
        tags_limited = True
        tags = tags[:tag_limit]
    else:
        tags_limited = False

    with indent_log():
        for tag in tags:
            logger.info(str(tag))

        if tags_limited:
            msg = f"...\n[First {tag_limit} tags shown. Pass --verbose to show all.]"
            logger.info(msg)


def ca_bundle_info(config: Configuration) -> str:
    levels = {key.split(".", 1)[0] for key, _ in config.items()}
    if not levels:
        return "Not specified"

    levels_that_override_global = ["install", "wheel", "download"]
    global_overriding_level = [
        level for level in levels if level in levels_that_override_global
    ]
    if not global_overriding_level:
        return "global"

    if "global" in levels:
        levels.remove("global")
    return ", ".join(levels)


class DebugCommand(Command):
    """
    Display debug information.
    """

    usage = """
      %prog <options>"""
    ignore_require_venv = True

    def add_options(self) -> None:
        cmdoptions.add_target_python_options(self.cmd_opts)
        self.parser.insert_option_group(0, self.cmd_opts)
        self.parser.config.load()

    def run(self, options: Values, args: List[str]) -> int:
        logger.warning(
            "This command is only meant for debugging. "
            "Do not use this with automation for parsing and getting these "
            "details, since the output and options of this command may "
            "change without notice."
        )
        show_value("pip version", get_pip_version())
        show_value("sys.version", sys.version)
        show_value("sys.executable", sys.executable)
        show_value("sys.getdefaultencoding", sys.getdefaultencoding())
        show_value("sys.getfilesystemencoding", sys.getfilesystemencoding())
        show_value(
            "locale.getpreferredencoding",
            locale.getpreferredencoding(),
        )
        show_value("sys.platform", sys.platform)
        show_sys_implementation()

        show_value("'cert' config value", ca_bundle_info(self.parser.config))
        show_value("REQUESTS_CA_BUNDLE", os.environ.get("REQUESTS_CA_BUNDLE"))
        show_value("CURL_CA_BUNDLE", os.environ.get("CURL_CA_BUNDLE"))
        show_value("pip._vendor.certifi.where()", where())
        show_value("pip._vendor.DEBUNDLED", pip._vendor.DEBUNDLED)

        show_vendor_versions()

        show_tags(options)

        return SUCCESS

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\tool_support.py ===
# util/tool_support.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls
"""support routines for the helpers in tools/.

These aren't imported by the enclosing util package as the are not
needed for normal library use.

"""
from __future__ import annotations

from argparse import ArgumentParser
from argparse import Namespace
import contextlib
import difflib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Union

from . import compat


class code_writer_cmd:
    parser: ArgumentParser
    args: Namespace
    suppress_output: bool
    diffs_detected: bool
    source_root: Path
    pyproject_toml_path: Path

    def __init__(self, tool_script: str):
        self.source_root = Path(tool_script).parent.parent
        self.pyproject_toml_path = self.source_root / Path("pyproject.toml")
        assert self.pyproject_toml_path.exists()

        self.parser = ArgumentParser()
        self.parser.add_argument(
            "--stdout",
            action="store_true",
            help="Write to stdout instead of saving to file",
        )
        self.parser.add_argument(
            "-c",
            "--check",
            help="Don't write the files back, just return the "
            "status. Return code 0 means nothing would change. "
            "Return code 1 means some files would be reformatted",
            action="store_true",
        )

    def run_zimports(self, tempfile: str) -> None:
        self._run_console_script(
            str(tempfile),
            {
                "entrypoint": "zimports",
                "options": f"--toml-config {self.pyproject_toml_path}",
            },
        )

    def run_black(self, tempfile: str) -> None:
        self._run_console_script(
            str(tempfile),
            {
                "entrypoint": "black",
                "options": f"--config {self.pyproject_toml_path}",
            },
        )

    def _run_console_script(self, path: str, options: Dict[str, Any]) -> None:
        """Run a Python console application from within the process.

        Used for black, zimports

        """

        is_posix = os.name == "posix"

        entrypoint_name = options["entrypoint"]

        for entry in compat.importlib_metadata_get("console_scripts"):
            if entry.name == entrypoint_name:
                impl = entry
                break
        else:
            raise Exception(
                f"Could not find entrypoint console_scripts.{entrypoint_name}"
            )
        cmdline_options_str = options.get("options", "")
        cmdline_options_list = shlex.split(
            cmdline_options_str, posix=is_posix
        ) + [path]

        kw: Dict[str, Any] = {}
        if self.suppress_output:
            kw["stdout"] = kw["stderr"] = subprocess.DEVNULL

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import %s; %s.%s()" % (impl.module, impl.module, impl.attr),
            ]
            + cmdline_options_list,
            cwd=str(self.source_root),
            **kw,
        )

    def write_status(self, *text: str) -> None:
        if not self.suppress_output:
            sys.stderr.write(" ".join(text))

    def write_output_file_from_text(
        self, text: str, destination_path: Union[str, Path]
    ) -> None:
        if self.args.check:
            self._run_diff(destination_path, source=text)
        elif self.args.stdout:
            print(text)
        else:
            self.write_status(f"Writing {destination_path}...")
            Path(destination_path).write_text(
                text, encoding="utf-8", newline="\n"
            )
            self.write_status("done\n")

    def write_output_file_from_tempfile(
        self, tempfile: str, destination_path: str
    ) -> None:
        if self.args.check:
            self._run_diff(destination_path, source_file=tempfile)
            os.unlink(tempfile)
        elif self.args.stdout:
            with open(tempfile) as tf:
                print(tf.read())
            os.unlink(tempfile)
        else:
            self.write_status(f"Writing {destination_path}...")
            shutil.move(tempfile, destination_path)
            self.write_status("done\n")

    def _run_diff(
        self,
        destination_path: Union[str, Path],
        *,
        source: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> None:
        if source_file:
            with open(source_file, encoding="utf-8") as tf:
                source_lines = list(tf)
        elif source is not None:
            source_lines = source.splitlines(keepends=True)
        else:
            assert False, "source or source_file is required"

        with open(destination_path, encoding="utf-8") as dp:
            d = difflib.unified_diff(
                list(dp),
                source_lines,
                fromfile=Path(destination_path).as_posix(),
                tofile="<proposed changes>",
                n=3,
                lineterm="\n",
            )
            d_as_list = list(d)
            if d_as_list:
                self.diffs_detected = True
                print("".join(d_as_list))

    @contextlib.contextmanager
    def add_arguments(self) -> Iterator[ArgumentParser]:
        yield self.parser

    @contextlib.contextmanager
    def run_program(self) -> Iterator[None]:
        self.args = self.parser.parse_args()
        if self.args.check:
            self.diffs_detected = False
            self.suppress_output = True
        elif self.args.stdout:
            self.suppress_output = True
        else:
            self.suppress_output = False
        yield

        if self.args.check and self.diffs_detected:
            sys.exit(1)
        else:
            sys.exit(0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\deltafunctions.py ===
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions import DiracDelta, Heaviside
from .integrals import Integral, integrate


def change_mul(node, x):
    """change_mul(node, x)

       Rearranges the operands of a product, bringing to front any simple
       DiracDelta expression.

       Explanation
       ===========

       If no simple DiracDelta expression was found, then all the DiracDelta
       expressions are simplified (using DiracDelta.expand(diracdelta=True, wrt=x)).

       Return: (dirac, new node)
       Where:
         o dirac is either a simple DiracDelta expression or None (if no simple
           expression was found);
         o new node is either a simplified DiracDelta expressions or None (if it
           could not be simplified).

       Examples
       ========

       >>> from sympy import DiracDelta, cos
       >>> from sympy.integrals.deltafunctions import change_mul
       >>> from sympy.abc import x, y
       >>> change_mul(x*y*DiracDelta(x)*cos(x), x)
       (DiracDelta(x), x*y*cos(x))
       >>> change_mul(x*y*DiracDelta(x**2 - 1)*cos(x), x)
       (None, x*y*cos(x)*DiracDelta(x - 1)/2 + x*y*cos(x)*DiracDelta(x + 1)/2)
       >>> change_mul(x*y*DiracDelta(cos(x))*cos(x), x)
       (None, None)

       See Also
       ========

       sympy.functions.special.delta_functions.DiracDelta
       deltaintegrate
    """

    new_args = []
    dirac = None

    #Sorting is needed so that we consistently collapse the same delta;
    #However, we must preserve the ordering of non-commutative terms
    c, nc = node.args_cnc()
    sorted_args = sorted(c, key=default_sort_key)
    sorted_args.extend(nc)

    for arg in sorted_args:
        if arg.is_Pow and isinstance(arg.base, DiracDelta):
            new_args.append(arg.func(arg.base, arg.exp - 1))
            arg = arg.base
        if dirac is None and (isinstance(arg, DiracDelta) and arg.is_simple(x)):
            dirac = arg
        else:
            new_args.append(arg)
    if not dirac:  # there was no simple dirac
        new_args = []
        for arg in sorted_args:
            if isinstance(arg, DiracDelta):
                new_args.append(arg.expand(diracdelta=True, wrt=x))
            elif arg.is_Pow and isinstance(arg.base, DiracDelta):
                new_args.append(arg.func(arg.base.expand(diracdelta=True, wrt=x), arg.exp))
            else:
                new_args.append(arg)
        if new_args != sorted_args:
            nnode = Mul(*new_args).expand()
        else:  # if the node didn't change there is nothing to do
            nnode = None
        return (None, nnode)
    return (dirac, Mul(*new_args))


def deltaintegrate(f, x):
    """
    deltaintegrate(f, x)

    Explanation
    ===========

    The idea for integration is the following:

    - If we are dealing with a DiracDelta expression, i.e. DiracDelta(g(x)),
      we try to simplify it.

      If we could simplify it, then we integrate the resulting expression.
      We already know we can integrate a simplified expression, because only
      simple DiracDelta expressions are involved.

      If we couldn't simplify it, there are two cases:

      1) The expression is a simple expression: we return the integral,
         taking care if we are dealing with a Derivative or with a proper
         DiracDelta.

      2) The expression is not simple (i.e. DiracDelta(cos(x))): we can do
         nothing at all.

    - If the node is a multiplication node having a DiracDelta term:

      First we expand it.

      If the expansion did work, then we try to integrate the expansion.

      If not, we try to extract a simple DiracDelta term, then we have two
      cases:

      1) We have a simple DiracDelta term, so we return the integral.

      2) We didn't have a simple term, but we do have an expression with
         simplified DiracDelta terms, so we integrate this expression.

    Examples
    ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.integrals.deltafunctions import deltaintegrate
        >>> from sympy import sin, cos, DiracDelta
        >>> deltaintegrate(x*sin(x)*cos(x)*DiracDelta(x - 1), x)
        sin(1)*cos(1)*Heaviside(x - 1)
        >>> deltaintegrate(y**2*DiracDelta(x - z)*DiracDelta(y - z), y)
        z**2*DiracDelta(x - z)*Heaviside(y - z)

    See Also
    ========

    sympy.functions.special.delta_functions.DiracDelta
    sympy.integrals.integrals.Integral
    """
    if not f.has(DiracDelta):
        return None

    # g(x) = DiracDelta(h(x))
    if f.func == DiracDelta:
        h = f.expand(diracdelta=True, wrt=x)
        if h == f:  # can't simplify the expression
            #FIXME: the second term tells whether is DeltaDirac or Derivative
            #For integrating derivatives of DiracDelta we need the chain rule
            if f.is_simple(x):
                if (len(f.args) <= 1 or f.args[1] == 0):
                    return Heaviside(f.args[0])
                else:
                    return (DiracDelta(f.args[0], f.args[1] - 1) /
                        f.args[0].as_poly().LC())
        else:  # let's try to integrate the simplified expression
            fh = integrate(h, x)
            return fh
    elif f.is_Mul or f.is_Pow:  # g(x) = a*b*c*f(DiracDelta(h(x)))*d*e
        g = f.expand()
        if f != g:  # the expansion worked
            fh = integrate(g, x)
            if fh is not None and not isinstance(fh, Integral):
                return fh
        else:
            # no expansion performed, try to extract a simple DiracDelta term
            deltaterm, rest_mult = change_mul(f, x)

            if not deltaterm:
                if rest_mult:
                    fh = integrate(rest_mult, x)
                    return fh
            else:
                from sympy.solvers import solve
                deltaterm = deltaterm.expand(diracdelta=True, wrt=x)
                if deltaterm.is_Mul:  # Take out any extracted factors
                    deltaterm, rest_mult_2 = change_mul(deltaterm, x)
                    rest_mult = rest_mult*rest_mult_2
                point = solve(deltaterm.args[0], x)[0]

                # Return the largest hyperreal term left after
                # repeated integration by parts.  For example,
                #
                #   integrate(y*DiracDelta(x, 1),x) == y*DiracDelta(x,0),  not 0
                #
                # This is so Integral(y*DiracDelta(x).diff(x),x).doit()
                # will return y*DiracDelta(x) instead of 0 or DiracDelta(x),
                # both of which are correct everywhere the value is defined
                # but give wrong answers for nested integration.
                n = (0 if len(deltaterm.args)==1 else deltaterm.args[1])
                m = 0
                while n >= 0:
                    r = S.NegativeOne**n*rest_mult.diff(x, n).subs(x, point)
                    if r.is_zero:
                        n -= 1
                        m += 1
                    else:
                        if m == 0:
                            return r*Heaviside(x - point)
                        else:
                            return r*DiracDelta(x,m-1)
                # In some very weak sense, x=0 is still a singularity,
                # but we hope will not be of any practical consequence.
                return S.Zero
    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\numbers.py ===
# Copyright (c) 2010-2024 openpyxl

import re

from openpyxl.descriptors import (
    String,
    Sequence,
    Integer,
)
from openpyxl.descriptors.serialisable import Serialisable


BUILTIN_FORMATS = {
    0: 'General',
    1: '0',
    2: '0.00',
    3: '#,##0',
    4: '#,##0.00',
    5: '"$"#,##0_);("$"#,##0)',
    6: '"$"#,##0_);[Red]("$"#,##0)',
    7: '"$"#,##0.00_);("$"#,##0.00)',
    8: '"$"#,##0.00_);[Red]("$"#,##0.00)',
    9: '0%',
    10: '0.00%',
    11: '0.00E+00',
    12: '# ?/?',
    13: '# ??/??',
    14: 'mm-dd-yy',
    15: 'd-mmm-yy',
    16: 'd-mmm',
    17: 'mmm-yy',
    18: 'h:mm AM/PM',
    19: 'h:mm:ss AM/PM',
    20: 'h:mm',
    21: 'h:mm:ss',
    22: 'm/d/yy h:mm',

    37: '#,##0_);(#,##0)',
    38: '#,##0_);[Red](#,##0)',
    39: '#,##0.00_);(#,##0.00)',
    40: '#,##0.00_);[Red](#,##0.00)',

    41: r'_(* #,##0_);_(* \(#,##0\);_(* "-"_);_(@_)',
    42: r'_("$"* #,##0_);_("$"* \(#,##0\);_("$"* "-"_);_(@_)',
    43: r'_(* #,##0.00_);_(* \(#,##0.00\);_(* "-"??_);_(@_)',

    44: r'_("$"* #,##0.00_)_("$"* \(#,##0.00\)_("$"* "-"??_)_(@_)',
    45: 'mm:ss',
    46: '[h]:mm:ss',
    47: 'mmss.0',
    48: '##0.0E+0',
    49: '@', }

BUILTIN_FORMATS_MAX_SIZE = 164
BUILTIN_FORMATS_REVERSE = dict(
        [(value, key) for key, value in BUILTIN_FORMATS.items()])

FORMAT_GENERAL = BUILTIN_FORMATS[0]
FORMAT_TEXT = BUILTIN_FORMATS[49]
FORMAT_NUMBER = BUILTIN_FORMATS[1]
FORMAT_NUMBER_00 = BUILTIN_FORMATS[2]
FORMAT_NUMBER_COMMA_SEPARATED1 = BUILTIN_FORMATS[4]
FORMAT_NUMBER_COMMA_SEPARATED2 = '#,##0.00_-'
FORMAT_PERCENTAGE = BUILTIN_FORMATS[9]
FORMAT_PERCENTAGE_00 = BUILTIN_FORMATS[10]
FORMAT_DATE_YYYYMMDD2 = 'yyyy-mm-dd'
FORMAT_DATE_YYMMDD = 'yy-mm-dd'
FORMAT_DATE_DDMMYY = 'dd/mm/yy'
FORMAT_DATE_DMYSLASH = 'd/m/y'
FORMAT_DATE_DMYMINUS = 'd-m-y'
FORMAT_DATE_DMMINUS = 'd-m'
FORMAT_DATE_MYMINUS = 'm-y'
FORMAT_DATE_XLSX14 = BUILTIN_FORMATS[14]
FORMAT_DATE_XLSX15 = BUILTIN_FORMATS[15]
FORMAT_DATE_XLSX16 = BUILTIN_FORMATS[16]
FORMAT_DATE_XLSX17 = BUILTIN_FORMATS[17]
FORMAT_DATE_XLSX22 = BUILTIN_FORMATS[22]
FORMAT_DATE_DATETIME = 'yyyy-mm-dd h:mm:ss'
FORMAT_DATE_TIME1 = BUILTIN_FORMATS[18]
FORMAT_DATE_TIME2 = BUILTIN_FORMATS[19]
FORMAT_DATE_TIME3 = BUILTIN_FORMATS[20]
FORMAT_DATE_TIME4 = BUILTIN_FORMATS[21]
FORMAT_DATE_TIME5 = BUILTIN_FORMATS[45]
FORMAT_DATE_TIME6 = BUILTIN_FORMATS[21]
FORMAT_DATE_TIME7 = 'i:s.S'
FORMAT_DATE_TIME8 = 'h:mm:ss@'
FORMAT_DATE_TIMEDELTA = '[hh]:mm:ss'
FORMAT_DATE_YYMMDDSLASH = 'yy/mm/dd@'
FORMAT_CURRENCY_USD_SIMPLE = '"$"#,##0.00_-'
FORMAT_CURRENCY_USD = '$#,##0_-'
FORMAT_CURRENCY_EUR_SIMPLE = '[$EUR ]#,##0.00_-'


COLORS = r"\[(BLACK|BLUE|CYAN|GREEN|MAGENTA|RED|WHITE|YELLOW)\]"
LITERAL_GROUP = r'".*?"' # anything in quotes
LOCALE_GROUP = r'\[(?!hh?\]|mm?\]|ss?\])[^\]]*\]' # anything in square brackets, except hours or minutes or seconds
STRIP_RE = re.compile(f"{LITERAL_GROUP}|{LOCALE_GROUP}")
TIMEDELTA_RE = re.compile(r'\[hh?\](:mm(:ss(\.0*)?)?)?|\[mm?\](:ss(\.0*)?)?|\[ss?\](\.0*)?', re.I)


# Spec 18.8.31 numFmts
# +ve;-ve;zero;text

def is_date_format(fmt):
    if fmt is None:
        return False
    fmt = fmt.split(";")[0] # only look at the first format
    fmt = STRIP_RE.sub("", fmt) # ignore some formats
    return re.search(r"(?<![_\\])[dmhysDMHYS]", fmt) is not None


def is_timedelta_format(fmt):
    if fmt is None:
        return False
    fmt = fmt.split(";")[0] # only look at the first format
    return TIMEDELTA_RE.search(fmt) is not None


def is_datetime(fmt):
    """
    Return date, time or datetime
    """
    if not is_date_format(fmt):
        return

    DATE = TIME = False

    if any((x in fmt for x in 'dy')):
        DATE = True
    if any((x in fmt for x in 'hs')):
        TIME = True

    if DATE and TIME:
        return "datetime"
    if DATE:
        return "date"
    return "time"


def is_builtin(fmt):
    return fmt in BUILTIN_FORMATS.values()


def builtin_format_code(index):
    """Return one of the standard format codes by index."""
    try:
        fmt = BUILTIN_FORMATS[index]
    except KeyError:
        fmt = None
    return fmt


def builtin_format_id(fmt):
    """Return the id of a standard style."""
    return BUILTIN_FORMATS_REVERSE.get(fmt)


class NumberFormatDescriptor(String):

    def __set__(self, instance, value):
        if value is None:
            value = FORMAT_GENERAL
        super().__set__(instance, value)


class NumberFormat(Serialisable):

    numFmtId = Integer()
    formatCode = String()

    def __init__(self,
                 numFmtId=None,
                 formatCode=None,
                ):
        self.numFmtId = numFmtId
        self.formatCode = formatCode


class NumberFormatList(Serialisable):

    count = Integer(allow_none=True)
    numFmt = Sequence(expected_type=NumberFormat)

    __elements__ = ('numFmt',)
    __attrs__ = ("count",)

    def __init__(self,
                 count=None,
                 numFmt=(),
                ):
        self.numFmt = numFmt


    @property
    def count(self):
        return len(self.numFmt)


    def __getitem__(self, idx):
        return self.numFmt[idx]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\compatibility_tags.py ===
"""Generate and work with PEP 425 Compatibility Tags."""

import re
from typing import List, Optional, Tuple

from pip._vendor.packaging.tags import (
    PythonVersion,
    Tag,
    android_platforms,
    compatible_tags,
    cpython_tags,
    generic_tags,
    interpreter_name,
    interpreter_version,
    ios_platforms,
    mac_platforms,
)

_apple_arch_pat = re.compile(r"(.+)_(\d+)_(\d+)_(.+)")


def version_info_to_nodot(version_info: Tuple[int, ...]) -> str:
    # Only use up to the first two numbers.
    return "".join(map(str, version_info[:2]))


def _mac_platforms(arch: str) -> List[str]:
    match = _apple_arch_pat.match(arch)
    if match:
        name, major, minor, actual_arch = match.groups()
        mac_version = (int(major), int(minor))
        arches = [
            # Since we have always only checked that the platform starts
            # with "macosx", for backwards-compatibility we extract the
            # actual prefix provided by the user in case they provided
            # something like "macosxcustom_". It may be good to remove
            # this as undocumented or deprecate it in the future.
            "{}_{}".format(name, arch[len("macosx_") :])
            for arch in mac_platforms(mac_version, actual_arch)
        ]
    else:
        # arch pattern didn't match (?!)
        arches = [arch]
    return arches


def _ios_platforms(arch: str) -> List[str]:
    match = _apple_arch_pat.match(arch)
    if match:
        name, major, minor, actual_multiarch = match.groups()
        ios_version = (int(major), int(minor))
        arches = [
            # Since we have always only checked that the platform starts
            # with "ios", for backwards-compatibility we extract the
            # actual prefix provided by the user in case they provided
            # something like "ioscustom_". It may be good to remove
            # this as undocumented or deprecate it in the future.
            "{}_{}".format(name, arch[len("ios_") :])
            for arch in ios_platforms(ios_version, actual_multiarch)
        ]
    else:
        # arch pattern didn't match (?!)
        arches = [arch]
    return arches


def _android_platforms(arch: str) -> List[str]:
    match = re.fullmatch(r"android_(\d+)_(.+)", arch)
    if match:
        api_level, abi = match.groups()
        return list(android_platforms(int(api_level), abi))
    else:
        # arch pattern didn't match (?!)
        return [arch]


def _custom_manylinux_platforms(arch: str) -> List[str]:
    arches = [arch]
    arch_prefix, arch_sep, arch_suffix = arch.partition("_")
    if arch_prefix == "manylinux2014":
        # manylinux1/manylinux2010 wheels run on most manylinux2014 systems
        # with the exception of wheels depending on ncurses. PEP 599 states
        # manylinux1/manylinux2010 wheels should be considered
        # manylinux2014 wheels:
        # https://www.python.org/dev/peps/pep-0599/#backwards-compatibility-with-manylinux2010-wheels
        if arch_suffix in {"i686", "x86_64"}:
            arches.append("manylinux2010" + arch_sep + arch_suffix)
            arches.append("manylinux1" + arch_sep + arch_suffix)
    elif arch_prefix == "manylinux2010":
        # manylinux1 wheels run on most manylinux2010 systems with the
        # exception of wheels depending on ncurses. PEP 571 states
        # manylinux1 wheels should be considered manylinux2010 wheels:
        # https://www.python.org/dev/peps/pep-0571/#backwards-compatibility-with-manylinux1-wheels
        arches.append("manylinux1" + arch_sep + arch_suffix)
    return arches


def _get_custom_platforms(arch: str) -> List[str]:
    arch_prefix, arch_sep, arch_suffix = arch.partition("_")
    if arch.startswith("macosx"):
        arches = _mac_platforms(arch)
    elif arch.startswith("ios"):
        arches = _ios_platforms(arch)
    elif arch_prefix == "android":
        arches = _android_platforms(arch)
    elif arch_prefix in ["manylinux2014", "manylinux2010"]:
        arches = _custom_manylinux_platforms(arch)
    else:
        arches = [arch]
    return arches


def _expand_allowed_platforms(platforms: Optional[List[str]]) -> Optional[List[str]]:
    if not platforms:
        return None

    seen = set()
    result = []

    for p in platforms:
        if p in seen:
            continue
        additions = [c for c in _get_custom_platforms(p) if c not in seen]
        seen.update(additions)
        result.extend(additions)

    return result


def _get_python_version(version: str) -> PythonVersion:
    if len(version) > 1:
        return int(version[0]), int(version[1:])
    else:
        return (int(version[0]),)


def _get_custom_interpreter(
    implementation: Optional[str] = None, version: Optional[str] = None
) -> str:
    if implementation is None:
        implementation = interpreter_name()
    if version is None:
        version = interpreter_version()
    return f"{implementation}{version}"


def get_supported(
    version: Optional[str] = None,
    platforms: Optional[List[str]] = None,
    impl: Optional[str] = None,
    abis: Optional[List[str]] = None,
) -> List[Tag]:
    """Return a list of supported tags for each version specified in
    `versions`.

    :param version: a string version, of the form "33" or "32",
        or None. The version will be assumed to support our ABI.
    :param platform: specify a list of platforms you want valid
        tags for, or None. If None, use the local system platform.
    :param impl: specify the exact implementation you want valid
        tags for, or None. If None, use the local interpreter impl.
    :param abis: specify a list of abis you want valid
        tags for, or None. If None, use the local interpreter abi.
    """
    supported: List[Tag] = []

    python_version: Optional[PythonVersion] = None
    if version is not None:
        python_version = _get_python_version(version)

    interpreter = _get_custom_interpreter(impl, version)

    platforms = _expand_allowed_platforms(platforms)

    is_cpython = (impl or interpreter_name()) == "cp"
    if is_cpython:
        supported.extend(
            cpython_tags(
                python_version=python_version,
                abis=abis,
                platforms=platforms,
            )
        )
    else:
        supported.extend(
            generic_tags(
                interpreter=interpreter,
                abis=abis,
                platforms=platforms,
            )
        )
    supported.extend(
        compatible_tags(
            python_version=python_version,
            interpreter=interpreter,
            platforms=platforms,
        )
    )

    return supported

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\rationalfield.py ===
"""Implementation of :class:`RationalField` class. """


from sympy.external.gmpy import MPQ

from sympy.polys.domains.groundtypes import SymPyRational, is_square, sqrtrem

from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class RationalField(Field, CharacteristicZero, SimpleDomain):
    r"""Abstract base class for the domain :ref:`QQ`.

    The :py:class:`RationalField` class represents the field of rational
    numbers $\mathbb{Q}$ as a :py:class:`~.Domain` in the domain system.
    :py:class:`RationalField` is a superclass of
    :py:class:`PythonRationalField` and :py:class:`GMPYRationalField` one of
    which will be the implementation for :ref:`QQ` depending on whether either
    of ``gmpy`` or ``gmpy2`` is installed or not.

    See also
    ========

    Domain
    """

    rep = 'QQ'
    alias = 'QQ'

    is_RationalField = is_QQ = True
    is_Numerical = True

    has_assoc_Ring = True
    has_assoc_Field = True

    dtype = MPQ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)

    def __init__(self):
        pass

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, RationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Returns hash code of ``self``. """
        return hash('QQ')

    def get_ring(self):
        """Returns ring associated with ``self``. """
        from sympy.polys.domains import ZZ
        return ZZ

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyRational(int(a.numerator), int(a.denominator))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Rational:
            return MPQ(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            return MPQ(*map(int, RR.to_rational(a)))
        else:
            raise CoercionFailed("expected `Rational` object, got %s" % a)

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`
            Generators of the extension. These should be expressions that are
            algebraic over `\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the returned :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicField`
            A :py:class:`~.Domain` representing the algebraic field extension.

        Examples
        ========

        >>> from sympy import QQ, sqrt
        >>> QQ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
        from sympy.polys.domains import AlgebraicField
        return AlgebraicField(self, *extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        """Convert a :py:class:`~.ANP` object to :ref:`QQ`.

        See :py:meth:`~.Domain.convert`
        """
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return MPQ(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return MPQ(a)

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return MPQ(a.numerator, a.denominator)

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return MPQ(a.numerator, a.denominator)

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return MPQ(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianElement`` object to ``dtype``. """
        if a.y == 0:
            return MPQ(a.x)

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return MPQ(*map(int, K0.to_rational(a)))

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  """
        return MPQ(a) / MPQ(b)

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
        return MPQ(a) / MPQ(b)

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies nothing.  """
        return self.zero

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__truediv__``. """
        return MPQ(a) / MPQ(b), self.zero

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numerator

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denominator

    def is_square(self, a):
        """Return ``True`` if ``a`` is a square.

        Explanation
        ===========
        A rational number is a square if and only if there exists
        a rational number ``b`` such that ``b * b == a``.
        """
        return is_square(a.numerator) and is_square(a.denominator)

    def exsqrt(self, a):
        """Non-negative square root of ``a`` if ``a`` is a square.

        See also
        ========
        is_square
        """
        if a.numerator < 0:  # denominator is always positive
            return None
        p_sqrt, p_rem = sqrtrem(a.numerator)
        if p_rem != 0:
            return None
        q_sqrt, q_rem = sqrtrem(a.denominator)
        if q_rem != 0:
            return None
        return MPQ(p_sqrt, q_sqrt)

QQ = RationalField()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\_chart.py ===
# Copyright (c) 2010-2024 openpyxl

from collections import OrderedDict
from operator import attrgetter

from openpyxl.descriptors import (
    Typed,
    Integer,
    Alias,
    MinMax,
    Bool,
    Set,
)
from openpyxl.descriptors.sequence import ValueSequence
from openpyxl.descriptors.serialisable import Serialisable

from ._3d import _3DBase
from .data_source import AxDataSource, NumRef
from .layout import Layout
from .legend import Legend
from .reference import Reference
from .series_factory import SeriesFactory
from .series import attribute_mapping
from .shapes import GraphicalProperties
from .title import TitleDescriptor

class AxId(Serialisable):

    val = Integer()

    def __init__(self, val):
        self.val = val


def PlotArea():
    from .chartspace import PlotArea
    return PlotArea()


class ChartBase(Serialisable):

    """
    Base class for all charts
    """

    legend = Typed(expected_type=Legend, allow_none=True)
    layout = Typed(expected_type=Layout, allow_none=True)
    roundedCorners = Bool(allow_none=True)
    axId = ValueSequence(expected_type=int)
    visible_cells_only = Bool(allow_none=True)
    display_blanks = Set(values=['span', 'gap', 'zero'])
    graphical_properties = Typed(expected_type=GraphicalProperties, allow_none=True)

    _series_type = ""
    ser = ()
    series = Alias('ser')
    title = TitleDescriptor()
    anchor = "E15" # default anchor position
    width = 15 # in cm, approx 5 rows
    height = 7.5 # in cm, approx 14 rows
    _id = 1
    _path = "/xl/charts/chart{0}.xml"
    style = MinMax(allow_none=True, min=1, max=48)
    mime_type = "application/vnd.openxmlformats-officedocument.drawingml.chart+xml"
    graphical_properties = Typed(expected_type=GraphicalProperties, allow_none=True) # mapped to chartspace

    __elements__ = ()


    def __init__(self, axId=(), **kw):
        self._charts = [self]
        self.title = None
        self.layout = None
        self.roundedCorners = None
        self.legend = Legend()
        self.graphical_properties = None
        self.style = None
        self.plot_area = PlotArea()
        self.axId = axId
        self.display_blanks = 'gap'
        self.pivotSource = None
        self.pivotFormats = ()
        self.visible_cells_only = True
        self.idx_base = 0
        self.graphical_properties = None
        super().__init__()


    def __hash__(self):
        """
        Just need to check for identity
        """
        return id(self)

    def __iadd__(self, other):
        """
        Combine the chart with another one
        """
        if not isinstance(other, ChartBase):
            raise TypeError("Only other charts can be added")
        self._charts.append(other)
        return self


    def to_tree(self, namespace=None, tagname=None, idx=None):
        self.axId = [id for id in self._axes]
        if self.ser is not None:
            for s in self.ser:
                s.__elements__ = attribute_mapping[self._series_type]
        return super().to_tree(tagname, idx)


    def _reindex(self):
        """
        Normalise and rebase series: sort by order and then rebase order

        """
        # sort data series in order and rebase
        ds = sorted(self.series, key=attrgetter("order"))
        for idx, s in enumerate(ds):
            s.order = idx
        self.series = ds


    def _write(self):
        from .chartspace import ChartSpace, ChartContainer
        self.plot_area.layout = self.layout

        idx_base = self.idx_base
        for chart in self._charts:
            if chart not in self.plot_area._charts:
                chart.idx_base = idx_base
                idx_base += len(chart.series)
        self.plot_area._charts = self._charts

        container = ChartContainer(plotArea=self.plot_area, legend=self.legend, title=self.title)
        if isinstance(chart, _3DBase):
            container.view3D = chart.view3D
            container.floor = chart.floor
            container.sideWall = chart.sideWall
            container.backWall = chart.backWall
        container.plotVisOnly = self.visible_cells_only
        container.dispBlanksAs = self.display_blanks
        container.pivotFmts = self.pivotFormats
        cs = ChartSpace(chart=container)
        cs.style = self.style
        cs.roundedCorners = self.roundedCorners
        cs.pivotSource = self.pivotSource
        cs.spPr = self.graphical_properties
        return cs.to_tree()


    @property
    def _axes(self):
        x = getattr(self, "x_axis", None)
        y = getattr(self, "y_axis", None)
        z = getattr(self, "z_axis", None)
        return OrderedDict([(axis.axId, axis) for axis in (x, y, z) if axis])


    def set_categories(self, labels):
        """
        Set the categories / x-axis values
        """
        if not isinstance(labels, Reference):
            labels = Reference(range_string=labels)
        for s in self.ser:
            s.cat = AxDataSource(numRef=NumRef(f=labels))


    def add_data(self, data, from_rows=False, titles_from_data=False):
        """
        Add a range of data in a single pass.
        The default is to treat each column as a data series.
        """
        if not isinstance(data, Reference):
            data = Reference(range_string=data)

        if from_rows:
            values = data.rows

        else:
            values = data.cols

        for ref in values:
            series = SeriesFactory(ref, title_from_data=titles_from_data)
            self.series.append(series)


    def append(self, value):
        """Append a data series to the chart"""
        l = self.series[:]
        l.append(value)
        self.series = l


    @property
    def path(self):
        return self._path.format(self._id)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\singleton.py ===
"""Singleton mechanism"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import Registry
from .sympify import sympify


if TYPE_CHECKING:
    from sympy.core.numbers import (
        Zero as _Zero,
        One as _One,
        NegativeOne as _NegativeOne,
        Half as _Half,
        Infinity as _Infinity,
        NegativeInfinity as _NegativeInfinity,
        ComplexInfinity as _ComplexInfinity,
        NaN as _NaN,
    )


class SingletonRegistry(Registry):
    """
    The registry for the singleton classes (accessible as ``S``).

    Explanation
    ===========

    This class serves as two separate things.

    The first thing it is is the ``SingletonRegistry``. Several classes in
    SymPy appear so often that they are singletonized, that is, using some
    metaprogramming they are made so that they can only be instantiated once
    (see the :class:`sympy.core.singleton.Singleton` class for details). For
    instance, every time you create ``Integer(0)``, this will return the same
    instance, :class:`sympy.core.numbers.Zero`. All singleton instances are
    attributes of the ``S`` object, so ``Integer(0)`` can also be accessed as
    ``S.Zero``.

    Singletonization offers two advantages: it saves memory, and it allows
    fast comparison. It saves memory because no matter how many times the
    singletonized objects appear in expressions in memory, they all point to
    the same single instance in memory. The fast comparison comes from the
    fact that you can use ``is`` to compare exact instances in Python
    (usually, you need to use ``==`` to compare things). ``is`` compares
    objects by memory address, and is very fast.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> a = Integer(0)
    >>> a is S.Zero
    True

    For the most part, the fact that certain objects are singletonized is an
    implementation detail that users should not need to worry about. In SymPy
    library code, ``is`` comparison is often used for performance purposes
    The primary advantage of ``S`` for end users is the convenient access to
    certain instances that are otherwise difficult to type, like ``S.Half``
    (instead of ``Rational(1, 2)``).

    When using ``is`` comparison, make sure the argument is sympified. For
    instance,

    >>> x = 0
    >>> x is S.Zero
    False

    This problem is not an issue when using ``==``, which is recommended for
    most use-cases:

    >>> 0 == S.Zero
    True

    The second thing ``S`` is is a shortcut for
    :func:`sympy.core.sympify.sympify`. :func:`sympy.core.sympify.sympify` is
    the function that converts Python objects such as ``int(1)`` into SymPy
    objects such as ``Integer(1)``. It also converts the string form of an
    expression into a SymPy expression, like ``sympify("x**2")`` ->
    ``Symbol("x")**2``. ``S(1)`` is the same thing as ``sympify(1)``
    (basically, ``S.__call__`` has been defined to call ``sympify``).

    This is for convenience, since ``S`` is a single letter. It's mostly
    useful for defining rational numbers. Consider an expression like ``x +
    1/2``. If you enter this directly in Python, it will evaluate the ``1/2``
    and give ``0.5``, because both arguments are ints (see also
    :ref:`tutorial-gotchas-final-notes`). However, in SymPy, you usually want
    the quotient of two integers to give an exact rational number. The way
    Python's evaluation works, at least one side of an operator needs to be a
    SymPy object for the SymPy evaluation to take over. You could write this
    as ``x + Rational(1, 2)``, but this is a lot more typing. A shorter
    version is ``x + S(1)/2``. Since ``S(1)`` returns ``Integer(1)``, the
    division will return a ``Rational`` type, since it will call
    ``Integer.__truediv__``, which knows how to return a ``Rational``.

    """
    __slots__ = ()

    Zero: _Zero
    One: _One
    NegativeOne: _NegativeOne
    Half: _Half
    Infinity: _Infinity
    NegativeInfinity: _NegativeInfinity
    ComplexInfinity: _ComplexInfinity
    NaN: _NaN

    # Also allow things like S(5)
    __call__ = staticmethod(sympify)

    def __init__(self):
        self._classes_to_install = {}
        # Dict of classes that have been registered, but that have not have been
        # installed as an attribute of this SingletonRegistry.
        # Installation automatically happens at the first attempt to access the
        # attribute.
        # The purpose of this is to allow registration during class
        # initialization during import, but not trigger object creation until
        # actual use (which should not happen until after all imports are
        # finished).

    def register(self, cls):
        # Make sure a duplicate class overwrites the old one
        if hasattr(self, cls.__name__):
            delattr(self, cls.__name__)
        self._classes_to_install[cls.__name__] = cls

    def __getattr__(self, name):
        """Python calls __getattr__ if no attribute of that name was installed
        yet.

        Explanation
        ===========

        This __getattr__ checks whether a class with the requested name was
        already registered but not installed; if no, raises an AttributeError.
        Otherwise, retrieves the class, calculates its singleton value, installs
        it as an attribute of the given name, and unregisters the class."""
        if name not in self._classes_to_install:
            raise AttributeError(
                "Attribute '%s' was not installed on SymPy registry %s" % (
                name, self))
        class_to_install = self._classes_to_install[name]
        value_to_install = class_to_install()
        self.__setattr__(name, value_to_install)
        del self._classes_to_install[name]
        return value_to_install

    def __repr__(self):
        return "S"

S = SingletonRegistry()


class Singleton(type):
    """
    Metaclass for singleton classes.

    Explanation
    ===========

    A singleton class has only one instance which is returned every time the
    class is instantiated. Additionally, this instance can be accessed through
    the global registry object ``S`` as ``S.<class_name>``.

    Examples
    ========

        >>> from sympy import S, Basic
        >>> from sympy.core.singleton import Singleton
        >>> class MySingleton(Basic, metaclass=Singleton):
        ...     pass
        >>> Basic() is Basic()
        False
        >>> MySingleton() is MySingleton()
        True
        >>> S.MySingleton is MySingleton()
        True

    Notes
    =====

    Instance creation is delayed until the first time the value is accessed.
    (SymPy versions before 1.0 would create the instance during class
    creation time, which would be prone to import cycles.)
    """
    def __init__(cls, *args, **kwargs):
        cls._instance = obj = Basic.__new__(cls)
        cls.__new__ = lambda cls: obj
        cls.__getnewargs__ = lambda obj: ()
        cls.__getstate__ = lambda obj: None
        S.register(cls)


# Delayed to avoid cyclic import
from .basic import Basic

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\inertia.py ===
from sympy import sympify
from sympy.physics.vector import Point, Dyadic, ReferenceFrame, outer
from collections import namedtuple

__all__ = ['inertia', 'inertia_of_point_mass', 'Inertia']


def inertia(frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0):
    """Simple way to create inertia Dyadic object.

    Explanation
    ===========

    Creates an inertia Dyadic based on the given tensor values and a body-fixed
    reference frame.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame the inertia is defined in.
    ixx : Sympifyable
        The xx element in the inertia dyadic.
    iyy : Sympifyable
        The yy element in the inertia dyadic.
    izz : Sympifyable
        The zz element in the inertia dyadic.
    ixy : Sympifyable
        The xy element in the inertia dyadic.
    iyz : Sympifyable
        The yz element in the inertia dyadic.
    izx : Sympifyable
        The zx element in the inertia dyadic.

    Examples
    ========

    >>> from sympy.physics.mechanics import ReferenceFrame, inertia
    >>> N = ReferenceFrame('N')
    >>> inertia(N, 1, 2, 3)
    (N.x|N.x) + 2*(N.y|N.y) + 3*(N.z|N.z)

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Need to define the inertia in a frame')
    ixx, iyy, izz = sympify(ixx), sympify(iyy), sympify(izz)
    ixy, iyz, izx = sympify(ixy), sympify(iyz), sympify(izx)
    return (ixx*outer(frame.x, frame.x) + ixy*outer(frame.x, frame.y) +
            izx*outer(frame.x, frame.z) + ixy*outer(frame.y, frame.x) +
            iyy*outer(frame.y, frame.y) + iyz*outer(frame.y, frame.z) +
            izx*outer(frame.z, frame.x) + iyz*outer(frame.z, frame.y) +
            izz*outer(frame.z, frame.z))


def inertia_of_point_mass(mass, pos_vec, frame):
    """Inertia dyadic of a point mass relative to point O.

    Parameters
    ==========

    mass : Sympifyable
        Mass of the point mass
    pos_vec : Vector
        Position from point O to point mass
    frame : ReferenceFrame
        Reference frame to express the dyadic in

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import ReferenceFrame, inertia_of_point_mass
    >>> N = ReferenceFrame('N')
    >>> r, m = symbols('r m')
    >>> px = r * N.x
    >>> inertia_of_point_mass(m, px, N)
    m*r**2*(N.y|N.y) + m*r**2*(N.z|N.z)

    """

    return mass*(
        (outer(frame.x, frame.x) +
         outer(frame.y, frame.y) +
         outer(frame.z, frame.z)) *
        (pos_vec.dot(pos_vec)) - outer(pos_vec, pos_vec))


class Inertia(namedtuple('Inertia', ['dyadic', 'point'])):
    """Inertia object consisting of a Dyadic and a Point of reference.

    Explanation
    ===========

    This is a simple class to store the Point and Dyadic, belonging to an
    inertia.

    Attributes
    ==========

    dyadic : Dyadic
        The dyadic of the inertia.
    point : Point
        The reference point of the inertia.

    Examples
    ========

    >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Inertia(N.x.outer(N.x) + N.y.outer(N.y) + N.z.outer(N.z), Po)
    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    In the example above the Dyadic was created manually, one can however also
    use the ``inertia`` function for this or the class method ``from_tensor`` as
    shown below.

    >>> Inertia.from_inertia_scalars(Po, N, 1, 1, 1)
    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    """
    __slots__ = ()

    def __new__(cls, dyadic, point):
        # Switch order if given in the wrong order
        if isinstance(dyadic, Point) and isinstance(point, Dyadic):
            point, dyadic = dyadic, point
        if not isinstance(point, Point):
            raise TypeError('Reference point should be of type Point')
        if not isinstance(dyadic, Dyadic):
            raise TypeError('Inertia value should be expressed as a Dyadic')
        return super().__new__(cls, dyadic, point)

    @classmethod
    def from_inertia_scalars(cls, point, frame, ixx, iyy, izz, ixy=0, iyz=0,
                             izx=0):
        """Simple way to create an Inertia object based on the tensor values.

        Explanation
        ===========

        This class method uses the :func`~.inertia` to create the Dyadic based
        on the tensor values.

        Parameters
        ==========

        point : Point
            The reference point of the inertia.
        frame : ReferenceFrame
            The frame the inertia is defined in.
        ixx : Sympifyable
            The xx element in the inertia dyadic.
        iyy : Sympifyable
            The yy element in the inertia dyadic.
        izz : Sympifyable
            The zz element in the inertia dyadic.
        ixy : Sympifyable
            The xy element in the inertia dyadic.
        iyz : Sympifyable
            The yz element in the inertia dyadic.
        izx : Sympifyable
            The zx element in the inertia dyadic.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
        >>> ixx, iyy, izz, ixy, iyz, izx = symbols('ixx iyy izz ixy iyz izx')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> I = Inertia.from_inertia_scalars(P, N, ixx, iyy, izz, ixy, iyz, izx)

        The tensor values can easily be seen when converting the dyadic to a
        matrix.

        >>> I.dyadic.to_matrix(N)
        Matrix([
        [ixx, ixy, izx],
        [ixy, iyy, iyz],
        [izx, iyz, izz]])

        """
        return cls(inertia(frame, ixx, iyy, izz, ixy, iyz, izx), point)

    def __add__(self, other):
        raise TypeError(f"unsupported operand type(s) for +: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    def __mul__(self, other):
        raise TypeError(f"unsupported operand type(s) for *: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    __radd__ = __add__
    __rmul__ = __mul__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\parsing.py ===
"""
:func:`~pandas.eval` source string parsing functions
"""
from __future__ import annotations

from io import StringIO
from keyword import iskeyword
import token
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

# A token value Python's tokenizer probably will never use.
BACKTICK_QUOTED_STRING = 100


def create_valid_python_identifier(name: str) -> str:
    """
    Create valid Python identifiers from any string.

    Check if name contains any special characters. If it contains any
    special characters, the special characters will be replaced by
    a special string and a prefix is added.

    Raises
    ------
    SyntaxError
        If the returned name is not a Python valid identifier, raise an exception.
        This can happen if there is a hashtag in the name, as the tokenizer will
        than terminate and not find the backtick.
        But also for characters that fall out of the range of (U+0001..U+007F).
    """
    if name.isidentifier() and not iskeyword(name):
        return name

    # Create a dict with the special characters and their replacement string.
    # EXACT_TOKEN_TYPES contains these special characters
    # token.tok_name contains a readable description of the replacement string.
    special_characters_replacements = {
        char: f"_{token.tok_name[tokval]}_"
        for char, tokval in (tokenize.EXACT_TOKEN_TYPES.items())
    }
    special_characters_replacements.update(
        {
            " ": "_",
            "?": "_QUESTIONMARK_",
            "!": "_EXCLAMATIONMARK_",
            "$": "_DOLLARSIGN_",
            "€": "_EUROSIGN_",
            "°": "_DEGREESIGN_",
            # Including quotes works, but there are exceptions.
            "'": "_SINGLEQUOTE_",
            '"': "_DOUBLEQUOTE_",
            # Currently not possible. Terminates parser and won't find backtick.
            # "#": "_HASH_",
        }
    )

    name = "".join([special_characters_replacements.get(char, char) for char in name])
    name = f"BACKTICK_QUOTED_STRING_{name}"

    if not name.isidentifier():
        raise SyntaxError(f"Could not convert '{name}' to a valid Python identifier.")

    return name


def clean_backtick_quoted_toks(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Clean up a column name if surrounded by backticks.

    Backtick quoted string are indicated by a certain tokval value. If a string
    is a backtick quoted token it will processed by
    :func:`_create_valid_python_identifier` so that the parser can find this
    string when the query is executed.
    In this case the tok will get the NAME tokval.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tok : Tuple[int, str]
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    if toknum == BACKTICK_QUOTED_STRING:
        return tokenize.NAME, create_valid_python_identifier(tokval)
    return toknum, tokval


def clean_column_name(name: Hashable) -> Hashable:
    """
    Function to emulate the cleaning of a backtick quoted name.

    The purpose for this function is to see what happens to the name of
    identifier if it goes to the process of being parsed a Python code
    inside a backtick quoted string and than being cleaned
    (removed of any special characters).

    Parameters
    ----------
    name : hashable
        Name to be cleaned.

    Returns
    -------
    name : hashable
        Returns the name after tokenizing and cleaning.

    Notes
    -----
        For some cases, a name cannot be converted to a valid Python identifier.
        In that case :func:`tokenize_string` raises a SyntaxError.
        In that case, we just return the name unmodified.

        If this name was used in the query string (this makes the query call impossible)
        an error will be raised by :func:`tokenize_backtick_quoted_string` instead,
        which is not caught and propagates to the user level.
    """
    try:
        tokenized = tokenize_string(f"`{name}`")
        tokval = next(tokenized)[1]
        return create_valid_python_identifier(tokval)
    except SyntaxError:
        return name


def tokenize_backtick_quoted_string(
    token_generator: Iterator[tokenize.TokenInfo], source: str, string_start: int
) -> tuple[int, str]:
    """
    Creates a token from a backtick quoted string.

    Moves the token_generator forwards till right after the next backtick.

    Parameters
    ----------
    token_generator : Iterator[tokenize.TokenInfo]
        The generator that yields the tokens of the source string (Tuple[int, str]).
        The generator is at the first token after the backtick (`)

    source : str
        The Python source code string.

    string_start : int
        This is the start of backtick quoted string inside the source string.

    Returns
    -------
    tok: Tuple[int, str]
        The token that represents the backtick quoted string.
        The integer is equal to BACKTICK_QUOTED_STRING (100).
    """
    for _, tokval, start, _, _ in token_generator:
        if tokval == "`":
            string_end = start[1]
            break

    return BACKTICK_QUOTED_STRING, source[string_start:string_end]


def tokenize_string(source: str) -> Iterator[tuple[int, str]]:
    """
    Tokenize a Python source code string.

    Parameters
    ----------
    source : str
        The Python source code string.

    Returns
    -------
    tok_generator : Iterator[Tuple[int, str]]
        An iterator yielding all tokens with only toknum and tokval (Tuple[ing, str]).
    """
    line_reader = StringIO(source).readline
    token_generator = tokenize.generate_tokens(line_reader)

    # Loop over all tokens till a backtick (`) is found.
    # Then, take all tokens till the next backtick to form a backtick quoted string
    for toknum, tokval, start, _, _ in token_generator:
        if tokval == "`":
            try:
                yield tokenize_backtick_quoted_string(
                    token_generator, source, string_start=start[1] + 1
                )
            except Exception as err:
                raise SyntaxError(f"Failed to parse backticks in '{source}'.") from err
        else:
            yield toknum, tokval

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\keysyms\common.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
# table for translating virtual keys to X windows key symbols


try:
    set
except NameError:
    from sets import Set as set

from pyreadline3.unicode_helper import ensure_unicode

validkey = set(
    [
        "cancel",
        "backspace",
        "tab",
        "clear",
        "return",
        "shift_l",
        "control_l",
        "alt_l",
        "pause",
        "caps_lock",
        "escape",
        "space",
        "prior",
        "next",
        "end",
        "home",
        "left",
        "up",
        "right",
        "down",
        "select",
        "print",
        "execute",
        "snapshot",
        "insert",
        "delete",
        "help",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        "f13",
        "f14",
        "f15",
        "f16",
        "f17",
        "f18",
        "f19",
        "f20",
        "f21",
        "f22",
        "f23",
        "f24",
        "num_lock",
        "scroll_lock",
        "vk_apps",
        "vk_processkey",
        "vk_attn",
        "vk_crsel",
        "vk_exsel",
        "vk_ereof",
        "vk_play",
        "vk_zoom",
        "vk_noname",
        "vk_pa1",
        "vk_oem_clear",
        "numpad0",
        "numpad1",
        "numpad2",
        "numpad3",
        "numpad4",
        "numpad5",
        "numpad6",
        "numpad7",
        "numpad8",
        "numpad9",
        "divide",
        "multiply",
        "add",
        "subtract",
        "vk_decimal",
    ]
)

escape_sequence_to_special_key = {
    "\\e[a": "up",
    "\\e[b": "down",
    "del": "delete",
}


class KeyPress(object):
    def __init__(self, char="", shift=False, control=False, meta=False, keyname=""):
        if control or meta or shift:
            char = char.upper()
        self.info = dict(
            char=char, shift=shift, control=control, meta=meta, keyname=keyname
        )

    def create(name):
        def get(self):
            return self.info[name]

        def set(self, value):
            self.info[name] = value

        return property(get, set)

    char = create("char")
    shift = create("shift")
    control = create("control")
    meta = create("meta")
    keyname = create("keyname")

    def __repr__(self):
        return "(%s,%s,%s,%s)" % tuple(map(ensure_unicode, self.tuple()))

    def tuple(self):
        if self.keyname:
            return (self.control, self.meta, self.shift, self.keyname)
        else:
            if self.control or self.meta or self.shift:
                return (self.control, self.meta, self.shift, self.char.upper())
            else:
                return (self.control, self.meta, self.shift, self.char)

    def __eq__(self, other):
        if isinstance(other, KeyPress):
            s = self.tuple()
            o = other.tuple()
            return s == o
        else:
            return False


def make_KeyPress_from_keydescr(keydescr):
    keyinfo = KeyPress()
    if len(keydescr) > 2 and keydescr[:1] == '"' and keydescr[-1:] == '"':
        keydescr = keydescr[1:-1]

    while True:
        lkeyname = keydescr.lower()
        if lkeyname.startswith("control-"):
            keyinfo.control = True
            keydescr = keydescr[8:]
        elif lkeyname.startswith("ctrl-"):
            keyinfo.control = True
            keydescr = keydescr[5:]
        elif keydescr.lower().startswith("\\c-"):
            keyinfo.control = True
            keydescr = keydescr[3:]
        elif keydescr.lower().startswith("\\m-"):
            keyinfo.meta = True
            keydescr = keydescr[3:]
        elif keydescr in escape_sequence_to_special_key:
            keydescr = escape_sequence_to_special_key[keydescr]
        elif lkeyname.startswith("meta-"):
            keyinfo.meta = True
            keydescr = keydescr[5:]
        elif lkeyname.startswith("alt-"):
            keyinfo.meta = True
            keydescr = keydescr[4:]
        elif lkeyname.startswith("shift-"):
            keyinfo.shift = True
            keydescr = keydescr[6:]
        else:
            if len(keydescr) > 1:
                if keydescr.strip().lower() in validkey:
                    keyinfo.keyname = keydescr.strip().lower()
                    keyinfo.char = ""
                else:
                    raise IndexError("Not a valid key: '%s'" % keydescr)
            else:
                keyinfo.char = keydescr
            return keyinfo


if __name__ == "__main__":
    import startup

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\performance_timeline.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: PerformanceTimeline (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page


@dataclass
class LargestContentfulPaint:
    '''
    See https://github.com/WICG/LargestContentfulPaint and largest_contentful_paint.idl
    '''
    render_time: network.TimeSinceEpoch

    load_time: network.TimeSinceEpoch

    #: The number of pixels being painted.
    size: float

    #: The id attribute of the element, if available.
    element_id: typing.Optional[str] = None

    #: The URL of the image (may be trimmed).
    url: typing.Optional[str] = None

    node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['renderTime'] = self.render_time.to_json()
        json['loadTime'] = self.load_time.to_json()
        json['size'] = self.size
        if self.element_id is not None:
            json['elementId'] = self.element_id
        if self.url is not None:
            json['url'] = self.url
        if self.node_id is not None:
            json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            render_time=network.TimeSinceEpoch.from_json(json['renderTime']),
            load_time=network.TimeSinceEpoch.from_json(json['loadTime']),
            size=float(json['size']),
            element_id=str(json['elementId']) if 'elementId' in json else None,
            url=str(json['url']) if 'url' in json else None,
            node_id=dom.BackendNodeId.from_json(json['nodeId']) if 'nodeId' in json else None,
        )


@dataclass
class LayoutShiftAttribution:
    previous_rect: dom.Rect

    current_rect: dom.Rect

    node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['previousRect'] = self.previous_rect.to_json()
        json['currentRect'] = self.current_rect.to_json()
        if self.node_id is not None:
            json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            previous_rect=dom.Rect.from_json(json['previousRect']),
            current_rect=dom.Rect.from_json(json['currentRect']),
            node_id=dom.BackendNodeId.from_json(json['nodeId']) if 'nodeId' in json else None,
        )


@dataclass
class LayoutShift:
    '''
    See https://wicg.github.io/layout-instability/#sec-layout-shift and layout_shift.idl
    '''
    #: Score increment produced by this event.
    value: float

    had_recent_input: bool

    last_input_time: network.TimeSinceEpoch

    sources: typing.List[LayoutShiftAttribution]

    def to_json(self):
        json = dict()
        json['value'] = self.value
        json['hadRecentInput'] = self.had_recent_input
        json['lastInputTime'] = self.last_input_time.to_json()
        json['sources'] = [i.to_json() for i in self.sources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=float(json['value']),
            had_recent_input=bool(json['hadRecentInput']),
            last_input_time=network.TimeSinceEpoch.from_json(json['lastInputTime']),
            sources=[LayoutShiftAttribution.from_json(i) for i in json['sources']],
        )


@dataclass
class TimelineEvent:
    #: Identifies the frame that this event is related to. Empty for non-frame targets.
    frame_id: page.FrameId

    #: The event type, as specified in https://w3c.github.io/performance-timeline/#dom-performanceentry-entrytype
    #: This determines which of the optional "details" fields is present.
    type_: str

    #: Name may be empty depending on the type.
    name: str

    #: Time in seconds since Epoch, monotonically increasing within document lifetime.
    time: network.TimeSinceEpoch

    #: Event duration, if applicable.
    duration: typing.Optional[float] = None

    lcp_details: typing.Optional[LargestContentfulPaint] = None

    layout_shift_details: typing.Optional[LayoutShift] = None

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        json['type'] = self.type_
        json['name'] = self.name
        json['time'] = self.time.to_json()
        if self.duration is not None:
            json['duration'] = self.duration
        if self.lcp_details is not None:
            json['lcpDetails'] = self.lcp_details.to_json()
        if self.layout_shift_details is not None:
            json['layoutShiftDetails'] = self.layout_shift_details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            frame_id=page.FrameId.from_json(json['frameId']),
            type_=str(json['type']),
            name=str(json['name']),
            time=network.TimeSinceEpoch.from_json(json['time']),
            duration=float(json['duration']) if 'duration' in json else None,
            lcp_details=LargestContentfulPaint.from_json(json['lcpDetails']) if 'lcpDetails' in json else None,
            layout_shift_details=LayoutShift.from_json(json['layoutShiftDetails']) if 'layoutShiftDetails' in json else None,
        )


def enable(
        event_types: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Previously buffered events would be reported before method returns.
    See also: timelineEventAdded

    :param event_types: The types of event to report, as specified in https://w3c.github.io/performance-timeline/#dom-performanceentry-entrytype The specified filter overrides any previous filters, passing empty filter disables recording. Note that not all types exposed to the web platform are currently supported.
    '''
    params: T_JSON_DICT = dict()
    params['eventTypes'] = [i for i in event_types]
    cmd_dict: T_JSON_DICT = {
        'method': 'PerformanceTimeline.enable',
        'params': params,
    }
    json = yield cmd_dict


@event_class('PerformanceTimeline.timelineEventAdded')
@dataclass
class TimelineEventAdded:
    '''
    Sent when a performance timeline event is added. See reportPerformanceTimeline method.
    '''
    event: TimelineEvent

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TimelineEventAdded:
        return cls(
            event=TimelineEvent.from_json(json['event'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\performance_timeline.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: PerformanceTimeline (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page


@dataclass
class LargestContentfulPaint:
    '''
    See https://github.com/WICG/LargestContentfulPaint and largest_contentful_paint.idl
    '''
    render_time: network.TimeSinceEpoch

    load_time: network.TimeSinceEpoch

    #: The number of pixels being painted.
    size: float

    #: The id attribute of the element, if available.
    element_id: typing.Optional[str] = None

    #: The URL of the image (may be trimmed).
    url: typing.Optional[str] = None

    node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['renderTime'] = self.render_time.to_json()
        json['loadTime'] = self.load_time.to_json()
        json['size'] = self.size
        if self.element_id is not None:
            json['elementId'] = self.element_id
        if self.url is not None:
            json['url'] = self.url
        if self.node_id is not None:
            json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            render_time=network.TimeSinceEpoch.from_json(json['renderTime']),
            load_time=network.TimeSinceEpoch.from_json(json['loadTime']),
            size=float(json['size']),
            element_id=str(json['elementId']) if 'elementId' in json else None,
            url=str(json['url']) if 'url' in json else None,
            node_id=dom.BackendNodeId.from_json(json['nodeId']) if 'nodeId' in json else None,
        )


@dataclass
class LayoutShiftAttribution:
    previous_rect: dom.Rect

    current_rect: dom.Rect

    node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['previousRect'] = self.previous_rect.to_json()
        json['currentRect'] = self.current_rect.to_json()
        if self.node_id is not None:
            json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            previous_rect=dom.Rect.from_json(json['previousRect']),
            current_rect=dom.Rect.from_json(json['currentRect']),
            node_id=dom.BackendNodeId.from_json(json['nodeId']) if 'nodeId' in json else None,
        )


@dataclass
class LayoutShift:
    '''
    See https://wicg.github.io/layout-instability/#sec-layout-shift and layout_shift.idl
    '''
    #: Score increment produced by this event.
    value: float

    had_recent_input: bool

    last_input_time: network.TimeSinceEpoch

    sources: typing.List[LayoutShiftAttribution]

    def to_json(self):
        json = dict()
        json['value'] = self.value
        json['hadRecentInput'] = self.had_recent_input
        json['lastInputTime'] = self.last_input_time.to_json()
        json['sources'] = [i.to_json() for i in self.sources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=float(json['value']),
            had_recent_input=bool(json['hadRecentInput']),
            last_input_time=network.TimeSinceEpoch.from_json(json['lastInputTime']),
            sources=[LayoutShiftAttribution.from_json(i) for i in json['sources']],
        )


@dataclass
class TimelineEvent:
    #: Identifies the frame that this event is related to. Empty for non-frame targets.
    frame_id: page.FrameId

    #: The event type, as specified in https://w3c.github.io/performance-timeline/#dom-performanceentry-entrytype
    #: This determines which of the optional "details" fields is present.
    type_: str

    #: Name may be empty depending on the type.
    name: str

    #: Time in seconds since Epoch, monotonically increasing within document lifetime.
    time: network.TimeSinceEpoch

    #: Event duration, if applicable.
    duration: typing.Optional[float] = None

    lcp_details: typing.Optional[LargestContentfulPaint] = None

    layout_shift_details: typing.Optional[LayoutShift] = None

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        json['type'] = self.type_
        json['name'] = self.name
        json['time'] = self.time.to_json()
        if self.duration is not None:
            json['duration'] = self.duration
        if self.lcp_details is not None:
            json['lcpDetails'] = self.lcp_details.to_json()
        if self.layout_shift_details is not None:
            json['layoutShiftDetails'] = self.layout_shift_details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            frame_id=page.FrameId.from_json(json['frameId']),
            type_=str(json['type']),
            name=str(json['name']),
            time=network.TimeSinceEpoch.from_json(json['time']),
            duration=float(json['duration']) if 'duration' in json else None,
            lcp_details=LargestContentfulPaint.from_json(json['lcpDetails']) if 'lcpDetails' in json else None,
            layout_shift_details=LayoutShift.from_json(json['layoutShiftDetails']) if 'layoutShiftDetails' in json else None,
        )


def enable(
        event_types: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Previously buffered events would be reported before method returns.
    See also: timelineEventAdded

    :param event_types: The types of event to report, as specified in https://w3c.github.io/performance-timeline/#dom-performanceentry-entrytype The specified filter overrides any previous filters, passing empty filter disables recording. Note that not all types exposed to the web platform are currently supported.
    '''
    params: T_JSON_DICT = dict()
    params['eventTypes'] = [i for i in event_types]
    cmd_dict: T_JSON_DICT = {
        'method': 'PerformanceTimeline.enable',
        'params': params,
    }
    json = yield cmd_dict


@event_class('PerformanceTimeline.timelineEventAdded')
@dataclass
class TimelineEventAdded:
    '''
    Sent when a performance timeline event is added. See reportPerformanceTimeline method.
    '''
    event: TimelineEvent

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TimelineEventAdded:
        return cls(
            event=TimelineEvent.from_json(json['event'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\performance_timeline.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: PerformanceTimeline (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page


@dataclass
class LargestContentfulPaint:
    '''
    See https://github.com/WICG/LargestContentfulPaint and largest_contentful_paint.idl
    '''
    render_time: network.TimeSinceEpoch

    load_time: network.TimeSinceEpoch

    #: The number of pixels being painted.
    size: float

    #: The id attribute of the element, if available.
    element_id: typing.Optional[str] = None

    #: The URL of the image (may be trimmed).
    url: typing.Optional[str] = None

    node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['renderTime'] = self.render_time.to_json()
        json['loadTime'] = self.load_time.to_json()
        json['size'] = self.size
        if self.element_id is not None:
            json['elementId'] = self.element_id
        if self.url is not None:
            json['url'] = self.url
        if self.node_id is not None:
            json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            render_time=network.TimeSinceEpoch.from_json(json['renderTime']),
            load_time=network.TimeSinceEpoch.from_json(json['loadTime']),
            size=float(json['size']),
            element_id=str(json['elementId']) if 'elementId' in json else None,
            url=str(json['url']) if 'url' in json else None,
            node_id=dom.BackendNodeId.from_json(json['nodeId']) if 'nodeId' in json else None,
        )


@dataclass
class LayoutShiftAttribution:
    previous_rect: dom.Rect

    current_rect: dom.Rect

    node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['previousRect'] = self.previous_rect.to_json()
        json['currentRect'] = self.current_rect.to_json()
        if self.node_id is not None:
            json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            previous_rect=dom.Rect.from_json(json['previousRect']),
            current_rect=dom.Rect.from_json(json['currentRect']),
            node_id=dom.BackendNodeId.from_json(json['nodeId']) if 'nodeId' in json else None,
        )


@dataclass
class LayoutShift:
    '''
    See https://wicg.github.io/layout-instability/#sec-layout-shift and layout_shift.idl
    '''
    #: Score increment produced by this event.
    value: float

    had_recent_input: bool

    last_input_time: network.TimeSinceEpoch

    sources: typing.List[LayoutShiftAttribution]

    def to_json(self):
        json = dict()
        json['value'] = self.value
        json['hadRecentInput'] = self.had_recent_input
        json['lastInputTime'] = self.last_input_time.to_json()
        json['sources'] = [i.to_json() for i in self.sources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=float(json['value']),
            had_recent_input=bool(json['hadRecentInput']),
            last_input_time=network.TimeSinceEpoch.from_json(json['lastInputTime']),
            sources=[LayoutShiftAttribution.from_json(i) for i in json['sources']],
        )


@dataclass
class TimelineEvent:
    #: Identifies the frame that this event is related to. Empty for non-frame targets.
    frame_id: page.FrameId

    #: The event type, as specified in https://w3c.github.io/performance-timeline/#dom-performanceentry-entrytype
    #: This determines which of the optional "details" fields is present.
    type_: str

    #: Name may be empty depending on the type.
    name: str

    #: Time in seconds since Epoch, monotonically increasing within document lifetime.
    time: network.TimeSinceEpoch

    #: Event duration, if applicable.
    duration: typing.Optional[float] = None

    lcp_details: typing.Optional[LargestContentfulPaint] = None

    layout_shift_details: typing.Optional[LayoutShift] = None

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        json['type'] = self.type_
        json['name'] = self.name
        json['time'] = self.time.to_json()
        if self.duration is not None:
            json['duration'] = self.duration
        if self.lcp_details is not None:
            json['lcpDetails'] = self.lcp_details.to_json()
        if self.layout_shift_details is not None:
            json['layoutShiftDetails'] = self.layout_shift_details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            frame_id=page.FrameId.from_json(json['frameId']),
            type_=str(json['type']),
            name=str(json['name']),
            time=network.TimeSinceEpoch.from_json(json['time']),
            duration=float(json['duration']) if 'duration' in json else None,
            lcp_details=LargestContentfulPaint.from_json(json['lcpDetails']) if 'lcpDetails' in json else None,
            layout_shift_details=LayoutShift.from_json(json['layoutShiftDetails']) if 'layoutShiftDetails' in json else None,
        )


def enable(
        event_types: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Previously buffered events would be reported before method returns.
    See also: timelineEventAdded

    :param event_types: The types of event to report, as specified in https://w3c.github.io/performance-timeline/#dom-performanceentry-entrytype The specified filter overrides any previous filters, passing empty filter disables recording. Note that not all types exposed to the web platform are currently supported.
    '''
    params: T_JSON_DICT = dict()
    params['eventTypes'] = [i for i in event_types]
    cmd_dict: T_JSON_DICT = {
        'method': 'PerformanceTimeline.enable',
        'params': params,
    }
    json = yield cmd_dict


@event_class('PerformanceTimeline.timelineEventAdded')
@dataclass
class TimelineEventAdded:
    '''
    Sent when a performance timeline event is added. See reportPerformanceTimeline method.
    '''
    event: TimelineEvent

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TimelineEventAdded:
        return cls(
            event=TimelineEvent.from_json(json['event'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\state_changes.py ===
# orm/state_changes.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""State tracking utilities used by :class:`_orm.Session`.

"""

from __future__ import annotations

import contextlib
from enum import Enum
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

from .. import exc as sa_exc
from .. import util
from ..util.typing import Literal

_F = TypeVar("_F", bound=Callable[..., Any])


class _StateChangeState(Enum):
    pass


class _StateChangeStates(_StateChangeState):
    ANY = 1
    NO_CHANGE = 2
    CHANGE_IN_PROGRESS = 3


class _StateChange:
    """Supplies state assertion decorators.

    The current use case is for the :class:`_orm.SessionTransaction` class. The
    :class:`_StateChange` class itself is agnostic of the
    :class:`_orm.SessionTransaction` class so could in theory be generalized
    for other systems as well.

    """

    _next_state: _StateChangeState = _StateChangeStates.ANY
    _state: _StateChangeState = _StateChangeStates.NO_CHANGE
    _current_fn: Optional[Callable[..., Any]] = None

    def _raise_for_prerequisite_state(
        self, operation_name: str, state: _StateChangeState
    ) -> NoReturn:
        raise sa_exc.IllegalStateChangeError(
            f"Can't run operation '{operation_name}()' when Session "
            f"is in state {state!r}",
            code="isce",
        )

    @classmethod
    def declare_states(
        cls,
        prerequisite_states: Union[
            Literal[_StateChangeStates.ANY], Tuple[_StateChangeState, ...]
        ],
        moves_to: _StateChangeState,
    ) -> Callable[[_F], _F]:
        """Method decorator declaring valid states.

        :param prerequisite_states: sequence of acceptable prerequisite
         states.   Can be the single constant _State.ANY to indicate no
         prerequisite state

        :param moves_to: the expected state at the end of the method, assuming
         no exceptions raised.   Can be the constant _State.NO_CHANGE to
         indicate state should not change at the end of the method.

        """
        assert prerequisite_states, "no prequisite states sent"
        has_prerequisite_states = (
            prerequisite_states is not _StateChangeStates.ANY
        )

        prerequisite_state_collection = cast(
            "Tuple[_StateChangeState, ...]", prerequisite_states
        )
        expect_state_change = moves_to is not _StateChangeStates.NO_CHANGE

        @util.decorator
        def _go(fn: _F, self: Any, *arg: Any, **kw: Any) -> Any:
            current_state = self._state

            if (
                has_prerequisite_states
                and current_state not in prerequisite_state_collection
            ):
                self._raise_for_prerequisite_state(fn.__name__, current_state)

            next_state = self._next_state
            existing_fn = self._current_fn
            expect_state = moves_to if expect_state_change else current_state

            if (
                # destination states are restricted
                next_state is not _StateChangeStates.ANY
                # method seeks to change state
                and expect_state_change
                # destination state incorrect
                and next_state is not expect_state
            ):
                if existing_fn and next_state in (
                    _StateChangeStates.NO_CHANGE,
                    _StateChangeStates.CHANGE_IN_PROGRESS,
                ):
                    raise sa_exc.IllegalStateChangeError(
                        f"Method '{fn.__name__}()' can't be called here; "
                        f"method '{existing_fn.__name__}()' is already "
                        f"in progress and this would cause an unexpected "
                        f"state change to {moves_to!r}",
                        code="isce",
                    )
                else:
                    raise sa_exc.IllegalStateChangeError(
                        f"Cant run operation '{fn.__name__}()' here; "
                        f"will move to state {moves_to!r} where we are "
                        f"expecting {next_state!r}",
                        code="isce",
                    )

            self._current_fn = fn
            self._next_state = _StateChangeStates.CHANGE_IN_PROGRESS
            try:
                ret_value = fn(self, *arg, **kw)
            except:
                raise
            else:
                if self._state is expect_state:
                    return ret_value

                if self._state is current_state:
                    raise sa_exc.IllegalStateChangeError(
                        f"Method '{fn.__name__}()' failed to "
                        "change state "
                        f"to {moves_to!r} as expected",
                        code="isce",
                    )
                elif existing_fn:
                    raise sa_exc.IllegalStateChangeError(
                        f"While method '{existing_fn.__name__}()' was "
                        "running, "
                        f"method '{fn.__name__}()' caused an "
                        "unexpected "
                        f"state change to {self._state!r}",
                        code="isce",
                    )
                else:
                    raise sa_exc.IllegalStateChangeError(
                        f"Method '{fn.__name__}()' caused an unexpected "
                        f"state change to {self._state!r}",
                        code="isce",
                    )

            finally:
                self._next_state = next_state
                self._current_fn = existing_fn

        return _go

    @contextlib.contextmanager
    def _expect_state(self, expected: _StateChangeState) -> Iterator[Any]:
        """called within a method that changes states.

        method must also use the ``@declare_states()`` decorator.

        """
        assert self._next_state is _StateChangeStates.CHANGE_IN_PROGRESS, (
            "Unexpected call to _expect_state outside of "
            "state-changing method"
        )

        self._next_state = expected
        try:
            yield
        except:
            raise
        else:
            if self._state is not expected:
                raise sa_exc.IllegalStateChangeError(
                    f"Unexpected state change to {self._state!r}", code="isce"
                )
        finally:
            self._next_state = _StateChangeStates.CHANGE_IN_PROGRESS

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\complexfield.py ===
"""Implementation of :class:`ComplexField` class. """


from sympy.external.gmpy import SYMPY_INTS
from sympy.core.numbers import Float, I
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.gaussiandomains import QQ_I
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import DomainError, CoercionFailed
from sympy.utilities import public

from mpmath import MPContext


@public
class ComplexField(Field, CharacteristicZero, SimpleDomain):
    """Complex numbers up to the given precision. """

    rep = 'CC'

    is_ComplexField = is_CC = True

    is_Exact = False
    is_Numerical = True

    has_assoc_Ring = False
    has_assoc_Field = True

    _default_precision = 53

    @property
    def has_default_precision(self):
        return self.precision == self._default_precision

    @property
    def precision(self):
        return self._context.prec

    @property
    def dps(self):
        return self._context.dps

    @property
    def tolerance(self):
        return self._tolerance

    def __init__(self, prec=None, dps=None, tol=None):
        # XXX: The tolerance parameter is ignored but is kept for backward
        # compatibility for now.

        context = MPContext()

        if prec is None and dps is None:
            context.prec = self._default_precision
        elif dps is None:
            context.prec = prec
        elif prec is None:
            context.dps = dps
        else:
            raise TypeError("Cannot set both prec and dps")

        self._context = context

        self._dtype = context.mpc
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

        # XXX: Neither of these is actually used anywhere.
        self._max_denom = max(2**context.prec // 200, 99)
        self._tolerance = self.one / self._max_denom

    @property
    def tp(self):
        # XXX: Domain treats tp as an alias of dtype. Here we need two separate
        # things: dtype is a callable to make/convert instances. We use tp with
        # isinstance to check if an object is an instance of the domain
        # already.
        return self._dtype

    def dtype(self, x, y=0):
        # XXX: This is needed because mpmath does not recognise fmpz.
        # It might be better to add conversion routines to mpmath and if that
        # happens then this can be removed.
        if isinstance(x, SYMPY_INTS):
            x = int(x)
        if isinstance(y, SYMPY_INTS):
            y = int(y)
        return self._dtype(x, y)

    def __eq__(self, other):
        return isinstance(other, ComplexField) and self.precision == other.precision

    def __hash__(self):
        return hash((self.__class__.__name__, self._dtype, self.precision))

    def to_sympy(self, element):
        """Convert ``element`` to SymPy number. """
        return Float(element.real, self.dps) + I*Float(element.imag, self.dps)

    def from_sympy(self, expr):
        """Convert SymPy's number to ``dtype``. """
        number = expr.evalf(n=self.dps)
        real, imag = number.as_real_imag()

        if real.is_Number and imag.is_Number:
            return self.dtype(real, imag)
        else:
            raise CoercionFailed("expected complex number, got %s" % expr)

    def from_ZZ(self, element, base):
        return self.dtype(element)

    def from_ZZ_gmpy(self, element, base):
        return self.dtype(int(element))

    def from_ZZ_python(self, element, base):
        return self.dtype(element)

    def from_QQ(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_QQ_python(self, element, base):
        return self.dtype(element.numerator) / element.denominator

    def from_QQ_gmpy(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_GaussianIntegerRing(self, element, base):
        return self.dtype(int(element.x), int(element.y))

    def from_GaussianRationalField(self, element, base):
        x = element.x
        y = element.y
        return (self.dtype(int(x.numerator)) / int(x.denominator) +
                self.dtype(0, int(y.numerator)) / int(y.denominator))

    def from_AlgebraicField(self, element, base):
        return self.from_sympy(base.to_sympy(element).evalf(self.dps))

    def from_RealField(self, element, base):
        return self.dtype(element)

    def from_ComplexField(self, element, base):
        return self.dtype(element)

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError("there is no ring associated with %s" % self)

    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
        return QQ_I

    def is_negative(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False

    def is_positive(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False

    def is_nonnegative(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False

    def is_nonpositive(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return self.one

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a*b

    def almosteq(self, a, b, tolerance=None):
        """Check if ``a`` and ``b`` are almost equal. """
        return self._context.almosteq(a, b, tolerance)

    def is_square(self, a):
        """Returns ``True``. Every complex number has a complex square root."""
        return True

    def exsqrt(self, a):
        r"""Returns the principal complex square root of ``a``.

        Explanation
        ===========
        The argument of the principal square root is always within
        $(-\frac{\pi}{2}, \frac{\pi}{2}]$. The square root may be
        slightly inaccurate due to floating point rounding error.
        """
        return a ** 0.5

CC = ComplexField()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\series.py ===
# Copyright (c) 2010-2024 openpyxl


from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    String,
    Integer,
    Bool,
    Alias,
    Sequence,
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.nested import (
    NestedInteger,
    NestedBool,
    NestedNoneSet,
    NestedText,
)

from .shapes import GraphicalProperties
from .data_source import (
    AxDataSource,
    NumDataSource,
    NumRef,
    StrRef,
)
from .error_bar import ErrorBars
from .label import DataLabelList
from .marker import DataPoint, PictureOptions, Marker
from .trendline import Trendline

attribute_mapping = {
    'area': ('idx', 'order', 'tx', 'spPr', 'pictureOptions', 'dPt', 'dLbls', 'errBars',
             'trendline', 'cat', 'val',),
    'bar':('idx', 'order','tx', 'spPr', 'invertIfNegative', 'pictureOptions', 'dPt',
           'dLbls', 'trendline', 'errBars', 'cat', 'val', 'shape'),
    'bubble':('idx','order', 'tx', 'spPr', 'invertIfNegative', 'dPt', 'dLbls',
              'trendline', 'errBars', 'xVal', 'yVal', 'bubbleSize', 'bubble3D'),
    'line':('idx', 'order', 'tx', 'spPr', 'marker', 'dPt', 'dLbls', 'trendline',
            'errBars', 'cat', 'val', 'smooth'),
    'pie':('idx', 'order', 'tx', 'spPr', 'explosion', 'dPt', 'dLbls', 'cat', 'val'),
    'radar':('idx', 'order', 'tx', 'spPr', 'marker', 'dPt', 'dLbls', 'cat', 'val'),
    'scatter':('idx', 'order', 'tx', 'spPr', 'marker', 'dPt', 'dLbls', 'trendline',
               'errBars', 'xVal', 'yVal', 'smooth'),
    'surface':('idx', 'order', 'tx', 'spPr', 'cat', 'val'),
                     }


class SeriesLabel(Serialisable):

    tagname = "tx"

    strRef = Typed(expected_type=StrRef, allow_none=True)
    v = NestedText(expected_type=str, allow_none=True)
    value = Alias('v')

    __elements__ = ('strRef', 'v')

    def __init__(self,
                 strRef=None,
                 v=None):
        self.strRef = strRef
        self.v = v


class Series(Serialisable):

    """
    Generic series object. Should not be instantiated directly.
    User the chart.Series factory instead.
    """

    tagname = "ser"

    idx = NestedInteger()
    order = NestedInteger()
    tx = Typed(expected_type=SeriesLabel, allow_none=True)
    title = Alias('tx')
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias('spPr')

    # area chart
    pictureOptions = Typed(expected_type=PictureOptions, allow_none=True)
    dPt = Sequence(expected_type=DataPoint, allow_none=True)
    data_points = Alias("dPt")
    dLbls = Typed(expected_type=DataLabelList, allow_none=True)
    labels = Alias("dLbls")
    trendline = Typed(expected_type=Trendline, allow_none=True)
    errBars = Typed(expected_type=ErrorBars, allow_none=True)
    cat = Typed(expected_type=AxDataSource, allow_none=True)
    identifiers = Alias("cat")
    val = Typed(expected_type=NumDataSource, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    #bar chart
    invertIfNegative = NestedBool(allow_none=True)
    shape = NestedNoneSet(values=(['cone', 'coneToMax', 'box', 'cylinder', 'pyramid', 'pyramidToMax']))

    #bubble chart
    xVal = Typed(expected_type=AxDataSource, allow_none=True)
    yVal = Typed(expected_type=NumDataSource, allow_none=True)
    bubbleSize = Typed(expected_type=NumDataSource, allow_none=True)
    zVal = Alias("bubbleSize")
    bubble3D = NestedBool(allow_none=True)

    #line chart
    marker = Typed(expected_type=Marker, allow_none=True)
    smooth = NestedBool(allow_none=True)

    #pie chart
    explosion = NestedInteger(allow_none=True)

    __elements__ = ()


    def __init__(self,
                 idx=0,
                 order=0,
                 tx=None,
                 spPr=None,
                 pictureOptions=None,
                 dPt=(),
                 dLbls=None,
                 trendline=None,
                 errBars=None,
                 cat=None,
                 val=None,
                 invertIfNegative=None,
                 shape=None,
                 xVal=None,
                 yVal=None,
                 bubbleSize=None,
                 bubble3D=None,
                 marker=None,
                 smooth=None,
                 explosion=None,
                 extLst=None,
                ):
        self.idx = idx
        self.order = order
        self.tx = tx
        if spPr is None:
            spPr = GraphicalProperties()
        self.spPr = spPr
        self.pictureOptions = pictureOptions
        self.dPt = dPt
        self.dLbls = dLbls
        self.trendline = trendline
        self.errBars = errBars
        self.cat = cat
        self.val = val
        self.invertIfNegative = invertIfNegative
        self.shape = shape
        self.xVal = xVal
        self.yVal = yVal
        self.bubbleSize = bubbleSize
        self.bubble3D = bubble3D
        if marker is None:
            marker = Marker()
        self.marker = marker
        self.smooth = smooth
        self.explosion = explosion


    def to_tree(self, tagname=None, idx=None):
        """The index can need rebasing"""
        if idx is not None:
            if self.order == self.idx:
                self.order = idx # rebase the order if the index has been rebased
            self.idx = idx
        return super().to_tree(tagname)


class XYSeries(Series):

    """Dedicated series for charts that have x and y series"""

    idx = Series.idx
    order = Series.order
    tx = Series.tx
    spPr = Series.spPr

    dPt = Series.dPt
    dLbls = Series.dLbls
    trendline = Series.trendline
    errBars = Series.errBars
    xVal = Series.xVal
    yVal = Series.yVal

    invertIfNegative = Series.invertIfNegative

    bubbleSize = Series.bubbleSize
    bubble3D = Series.bubble3D

    marker = Series.marker
    smooth = Series.smooth

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\_writer.py ===
# Copyright (c) 2010-2024 openpyxl

"""Write the workbook global settings to the archive."""

from openpyxl.utils import quote_sheetname
from openpyxl.xml.constants import (
    ARC_APP,
    ARC_CORE,
    ARC_CUSTOM,
    ARC_WORKBOOK,
    PKG_REL_NS,
    CUSTOMUI_NS,
    ARC_ROOT_RELS,
)
from openpyxl.xml.functions import tostring, fromstring

from openpyxl.packaging.relationship import Relationship, RelationshipList
from openpyxl.workbook.defined_name import (
    DefinedName,
    DefinedNameList,
)
from openpyxl.workbook.external_reference import ExternalReference
from openpyxl.packaging.workbook import ChildSheet, WorkbookPackage, PivotCache
from openpyxl.workbook.properties import WorkbookProperties
from openpyxl.utils.datetime import CALENDAR_MAC_1904


def get_active_sheet(wb):
    """
    Return the index of the active sheet.
    If the sheet set to active is hidden return the next visible sheet or None
    """
    visible_sheets = [idx for idx, sheet in enumerate(wb._sheets) if sheet.sheet_state == "visible"]
    if not visible_sheets:
        raise IndexError("At least one sheet must be visible")

    idx = wb._active_sheet_index
    sheet = wb.active
    if sheet and sheet.sheet_state == "visible":
        return idx

    for idx in visible_sheets[idx:]:
        wb.active = idx
        return idx

    return None


class WorkbookWriter:

    def __init__(self, wb):
        self.wb = wb
        self.rels = RelationshipList()
        self.package = WorkbookPackage()
        self.package.workbookProtection = wb.security
        self.package.calcPr = wb.calculation


    def write_properties(self):

        props = WorkbookProperties() # needs a mapping to the workbook for preservation
        if self.wb.code_name is not None:
            props.codeName = self.wb.code_name
        if self.wb.excel_base_date == CALENDAR_MAC_1904:
            props.date1904 = True
        self.package.workbookPr = props


    def write_worksheets(self):
        for idx, sheet in enumerate(self.wb._sheets, 1):
            sheet_node = ChildSheet(name=sheet.title, sheetId=idx, id="rId{0}".format(idx))
            rel = Relationship(type=sheet._rel_type, Target=sheet.path)
            self.rels.append(rel)

            if not sheet.sheet_state == 'visible':
                if len(self.wb._sheets) == 1:
                    raise ValueError("The only worksheet of a workbook cannot be hidden")
                sheet_node.state = sheet.sheet_state
            self.package.sheets.append(sheet_node)


    def write_refs(self):
        for link in self.wb._external_links:
            # need to match a counter with a workbook's relations
            rId = len(self.wb.rels) + 1
            rel = Relationship(type=link._rel_type, Target=link.path)
            self.rels.append(rel)
            ext = ExternalReference(id=rel.id)
            self.package.externalReferences.append(ext)


    def write_names(self):
        defined_names = list(self.wb.defined_names.values())

        for idx, sheet in enumerate(self.wb.worksheets):
            quoted = quote_sheetname(sheet.title)

            # local names
            if sheet.defined_names:
                names = sheet.defined_names.values()
                for n in names:
                    n.localSheetId = idx
                defined_names.extend(names)

            if sheet.auto_filter:
                name = DefinedName(name='_FilterDatabase', localSheetId=idx, hidden=True)
                name.value = f"{quoted}!{sheet.auto_filter}"
                defined_names.append(name)

            if sheet.print_titles:
                name = DefinedName(name="Print_Titles", localSheetId=idx)
                name.value = sheet.print_titles
                defined_names.append(name)

            if sheet.print_area:
                name = DefinedName(name="Print_Area", localSheetId=idx)
                name.value = sheet.print_area
                defined_names.append(name)

        self.package.definedNames = DefinedNameList(definedName=defined_names)


    def write_pivots(self):
        pivot_caches = set()
        for pivot in self.wb._pivots:
            if pivot.cache not in pivot_caches:
                pivot_caches.add(pivot.cache)
                c = PivotCache(cacheId=pivot.cacheId)
                self.package.pivotCaches.append(c)
                rel = Relationship(Type=pivot.cache.rel_type, Target=pivot.cache.path)
                self.rels.append(rel)
                c.id = rel.id
        #self.wb._pivots = [] # reset


    def write_views(self):
        active = get_active_sheet(self.wb)
        if self.wb.views:
            self.wb.views[0].activeTab = active
        self.package.bookViews = self.wb.views


    def write(self):
        """Write the core workbook xml."""

        self.write_properties()
        self.write_worksheets()
        self.write_names()
        self.write_pivots()
        self.write_views()
        self.write_refs()

        return tostring(self.package.to_tree())


    def write_rels(self):
        """Write the workbook relationships xml."""

        styles =  Relationship(type='styles', Target='styles.xml')
        self.rels.append(styles)

        theme =  Relationship(type='theme', Target='theme/theme1.xml')
        self.rels.append(theme)

        if self.wb.vba_archive:
            vba =  Relationship(type='', Target='vbaProject.bin')
            vba.Type ='http://schemas.microsoft.com/office/2006/relationships/vbaProject'
            self.rels.append(vba)

        return tostring(self.rels.to_tree())


    def write_root_rels(self):
        """Write the package relationships"""

        rels = RelationshipList()

        rel = Relationship(type="officeDocument", Target=ARC_WORKBOOK)
        rels.append(rel)
        rel = Relationship(Type=f"{PKG_REL_NS}/metadata/core-properties", Target=ARC_CORE)
        rels.append(rel)

        rel = Relationship(type="extended-properties", Target=ARC_APP)
        rels.append(rel)

        if len(self.wb.custom_doc_props) >= 1:
            rel = Relationship(type="custom-properties", Target=ARC_CUSTOM)
            rels.append(rel)

        if self.wb.vba_archive is not None:
            # See if there was a customUI relation and reuse it
            xml = fromstring(self.wb.vba_archive.read(ARC_ROOT_RELS))
            root_rels = RelationshipList.from_tree(xml)
            for rel in root_rels.find(CUSTOMUI_NS):
                rels.append(rel)

        return tostring(rels.to_tree())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\clipboards.py ===
""" io on the clipboard """
from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING
import warnings

from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.generic import ABCDataFrame

from pandas import (
    get_option,
    option_context,
)

if TYPE_CHECKING:
    from pandas._typing import DtypeBackend


def read_clipboard(
    sep: str = r"\s+",
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    **kwargs,
):  # pragma: no cover
    r"""
    Read text from clipboard and pass to :func:`~pandas.read_csv`.

    Parses clipboard contents similar to how CSV files are parsed
    using :func:`~pandas.read_csv`.

    Parameters
    ----------
    sep : str, default '\\s+'
        A string or regex delimiter. The default of ``'\\s+'`` denotes
        one or more whitespace characters.

    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    **kwargs
        See :func:`~pandas.read_csv` for the full argument list.

    Returns
    -------
    DataFrame
        A parsed :class:`~pandas.DataFrame` object.

    See Also
    --------
    DataFrame.to_clipboard : Copy object to the system clipboard.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
    >>> df.to_clipboard()  # doctest: +SKIP
    >>> pd.read_clipboard()  # doctest: +SKIP
         A  B  C
    0    1  2  3
    1    4  5  6
    """
    encoding = kwargs.pop("encoding", "utf-8")

    # only utf-8 is valid for passed value because that's what clipboard
    # supports
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise NotImplementedError("reading from clipboard only supports utf-8 encoding")

    check_dtype_backend(dtype_backend)

    from pandas.io.clipboard import clipboard_get
    from pandas.io.parsers import read_csv

    text = clipboard_get()

    # Try to decode (if needed, as "text" might already be a string here).
    try:
        text = text.decode(kwargs.get("encoding") or get_option("display.encoding"))
    except AttributeError:
        pass

    # Excel copies into clipboard with \t separation
    # inspect no more then the 10 first lines, if they
    # all contain an equal number (>0) of tabs, infer
    # that this came from excel and set 'sep' accordingly
    lines = text[:10000].split("\n")[:-1][:10]

    # Need to remove leading white space, since read_csv
    # accepts:
    #    a  b
    # 0  1  2
    # 1  3  4

    counts = {x.lstrip(" ").count("\t") for x in lines}
    if len(lines) > 1 and len(counts) == 1 and counts.pop() != 0:
        sep = "\t"
        # check the number of leading tabs in the first line
        # to account for index columns
        index_length = len(lines[0]) - len(lines[0].lstrip(" \t"))
        if index_length != 0:
            kwargs.setdefault("index_col", list(range(index_length)))

    # Edge case where sep is specified to be None, return to default
    if sep is None and kwargs.get("delim_whitespace") is None:
        sep = r"\s+"

    # Regex separator currently only works with python engine.
    # Default to python if separator is multi-character (regex)
    if len(sep) > 1 and kwargs.get("engine") is None:
        kwargs["engine"] = "python"
    elif len(sep) > 1 and kwargs.get("engine") == "c":
        warnings.warn(
            "read_clipboard with regex separator does not work properly with c engine.",
            stacklevel=find_stack_level(),
        )

    return read_csv(StringIO(text), sep=sep, dtype_backend=dtype_backend, **kwargs)


def to_clipboard(
    obj, excel: bool | None = True, sep: str | None = None, **kwargs
) -> None:  # pragma: no cover
    """
    Attempt to write text representation of object to the system clipboard
    The clipboard can be then pasted into Excel for example.

    Parameters
    ----------
    obj : the object to write to the clipboard
    excel : bool, defaults to True
            if True, use the provided separator, writing in a csv
            format for allowing easy pasting into excel.
            if False, write a string representation of the object
            to the clipboard
    sep : optional, defaults to tab
    other keywords are passed to to_csv

    Notes
    -----
    Requirements for your platform
      - Linux: xclip, or xsel (with PyQt4 modules)
      - Windows:
      - OS X:
    """
    encoding = kwargs.pop("encoding", "utf-8")

    # testing if an invalid encoding is passed to clipboard
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise ValueError("clipboard only supports utf-8 encoding")

    from pandas.io.clipboard import clipboard_set

    if excel is None:
        excel = True

    if excel:
        try:
            if sep is None:
                sep = "\t"
            buf = StringIO()

            # clipboard_set (pyperclip) expects unicode
            obj.to_csv(buf, sep=sep, encoding="utf-8", **kwargs)
            text = buf.getvalue()

            clipboard_set(text)
            return
        except TypeError:
            warnings.warn(
                "to_clipboard in excel mode requires a single character separator.",
                stacklevel=find_stack_level(),
            )
    elif sep is not None:
        warnings.warn(
            "to_clipboard with excel=False ignores the sep argument.",
            stacklevel=find_stack_level(),
        )

    if isinstance(obj, ABCDataFrame):
        # str(df) has various unhelpful defaults, like truncation
        with option_context("display.max_colwidth", None):
            objstr = obj.to_string(**kwargs)
    else:
        objstr = str(obj)
    clipboard_set(objstr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\soupsieve\__meta__.py ===
"""Meta related things."""
from __future__ import annotations
from collections import namedtuple
import re

RE_VER = re.compile(
    r'''(?x)
    (?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<micro>\d+))?
    (?:(?P<type>a|b|rc)(?P<pre>\d+))?
    (?:\.post(?P<post>\d+))?
    (?:\.dev(?P<dev>\d+))?
    '''
)

REL_MAP = {
    ".dev": "",
    ".dev-alpha": "a",
    ".dev-beta": "b",
    ".dev-candidate": "rc",
    "alpha": "a",
    "beta": "b",
    "candidate": "rc",
    "final": ""
}

DEV_STATUS = {
    ".dev": "2 - Pre-Alpha",
    ".dev-alpha": "2 - Pre-Alpha",
    ".dev-beta": "2 - Pre-Alpha",
    ".dev-candidate": "2 - Pre-Alpha",
    "alpha": "3 - Alpha",
    "beta": "4 - Beta",
    "candidate": "4 - Beta",
    "final": "5 - Production/Stable"
}

PRE_REL_MAP = {"a": 'alpha', "b": 'beta', "rc": 'candidate'}


class Version(namedtuple("Version", ["major", "minor", "micro", "release", "pre", "post", "dev"])):
    """
    Get the version (PEP 440).

    A biased approach to the PEP 440 semantic version.

    Provides a tuple structure which is sorted for comparisons `v1 > v2` etc.
      (major, minor, micro, release type, pre-release build, post-release build, development release build)
    Release types are named in is such a way they are comparable with ease.
    Accessors to check if a development, pre-release, or post-release build. Also provides accessor to get
    development status for setup files.

    How it works (currently):

    - You must specify a release type as either `final`, `alpha`, `beta`, or `candidate`.
    - To define a development release, you can use either `.dev`, `.dev-alpha`, `.dev-beta`, or `.dev-candidate`.
      The dot is used to ensure all development specifiers are sorted before `alpha`.
      You can specify a `dev` number for development builds, but do not have to as implicit development releases
      are allowed.
    - You must specify a `pre` value greater than zero if using a prerelease as this project (not PEP 440) does not
      allow implicit prereleases.
    - You can optionally set `post` to a value greater than zero to make the build a post release. While post releases
      are technically allowed in prereleases, it is strongly discouraged, so we are rejecting them. It should be
      noted that we do not allow `post0` even though PEP 440 does not restrict this. This project specifically
      does not allow implicit post releases.
    - It should be noted that we do not support epochs `1!` or local versions `+some-custom.version-1`.

    Acceptable version releases:

    ```
    Version(1, 0, 0, "final")                    1.0
    Version(1, 2, 0, "final")                    1.2
    Version(1, 2, 3, "final")                    1.2.3
    Version(1, 2, 0, ".dev-alpha", pre=4)        1.2a4
    Version(1, 2, 0, ".dev-beta", pre=4)         1.2b4
    Version(1, 2, 0, ".dev-candidate", pre=4)    1.2rc4
    Version(1, 2, 0, "final", post=1)            1.2.post1
    Version(1, 2, 3, ".dev")                     1.2.3.dev0
    Version(1, 2, 3, ".dev", dev=1)              1.2.3.dev1
    ```

    """

    def __new__(
        cls,
        major: int, minor: int, micro: int, release: str = "final",
        pre: int = 0, post: int = 0, dev: int = 0
    ) -> Version:
        """Validate version info."""

        # Ensure all parts are positive integers.
        for value in (major, minor, micro, pre, post):
            if not (isinstance(value, int) and value >= 0):
                raise ValueError("All version parts except 'release' should be integers.")

        if release not in REL_MAP:
            raise ValueError(f"'{release}' is not a valid release type.")

        # Ensure valid pre-release (we do not allow implicit pre-releases).
        if ".dev-candidate" < release < "final":
            if pre == 0:
                raise ValueError("Implicit pre-releases not allowed.")
            elif dev:
                raise ValueError("Version is not a development release.")
            elif post:
                raise ValueError("Post-releases are not allowed with pre-releases.")

        # Ensure valid development or development/pre release
        elif release < "alpha":
            if release > ".dev" and pre == 0:
                raise ValueError("Implicit pre-release not allowed.")
            elif post:
                raise ValueError("Post-releases are not allowed with pre-releases.")

        # Ensure a valid normal release
        else:
            if pre:
                raise ValueError("Version is not a pre-release.")
            elif dev:
                raise ValueError("Version is not a development release.")

        return super().__new__(cls, major, minor, micro, release, pre, post, dev)

    def _is_pre(self) -> bool:
        """Is prerelease."""

        return bool(self.pre > 0)

    def _is_dev(self) -> bool:
        """Is development."""

        return bool(self.release < "alpha")

    def _is_post(self) -> bool:
        """Is post."""

        return bool(self.post > 0)

    def _get_dev_status(self) -> str:  # pragma: no cover
        """Get development status string."""

        return DEV_STATUS[self.release]

    def _get_canonical(self) -> str:
        """Get the canonical output string."""

        # Assemble major, minor, micro version and append `pre`, `post`, or `dev` if needed..
        if self.micro == 0:
            ver = f"{self.major}.{self.minor}"
        else:
            ver = f"{self.major}.{self.minor}.{self.micro}"
        if self._is_pre():
            ver += f'{REL_MAP[self.release]}{self.pre}'
        if self._is_post():
            ver += f".post{self.post}"
        if self._is_dev():
            ver += f".dev{self.dev}"

        return ver


def parse_version(ver: str) -> Version:
    """Parse version into a comparable Version tuple."""

    m = RE_VER.match(ver)

    if m is None:
        raise ValueError(f"'{ver}' is not a valid version")

    # Handle major, minor, micro
    major = int(m.group('major'))
    minor = int(m.group('minor')) if m.group('minor') else 0
    micro = int(m.group('micro')) if m.group('micro') else 0

    # Handle pre releases
    if m.group('type'):
        release = PRE_REL_MAP[m.group('type')]
        pre = int(m.group('pre'))
    else:
        release = "final"
        pre = 0

    # Handle development releases
    dev = m.group('dev') if m.group('dev') else 0
    if m.group('dev'):
        dev = int(m.group('dev'))
        release = '.dev-' + release if pre else '.dev'
    else:
        dev = 0

    # Handle post
    post = int(m.group('post')) if m.group('post') else 0

    return Version(major, minor, micro, release, pre, post, dev)


__version_info__ = Version(2, 7, 0, "final")
__version__ = __version_info__._get_canonical()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_timeouts.py ===
from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, NoReturn

import trio

if TYPE_CHECKING:
    from collections.abc import Generator


def move_on_at(deadline: float, *, shield: bool = False) -> trio.CancelScope:
    """Use as a context manager to create a cancel scope with the given
    absolute deadline.

    Args:
      deadline (float): The deadline.
      shield (bool): Initial value for the `~trio.CancelScope.shield` attribute
          of the newly created cancel scope.

    Raises:
      ValueError: if deadline is NaN.

    """
    # CancelScope validates that deadline isn't math.nan
    return trio.CancelScope(deadline=deadline, shield=shield)


def move_on_after(
    seconds: float,
    *,
    shield: bool = False,
) -> trio.CancelScope:
    """Use as a context manager to create a cancel scope whose deadline is
    set to now + *seconds*.

    The deadline of the cancel scope is calculated upon entering.

    Args:
      seconds (float): The timeout.
      shield (bool): Initial value for the `~trio.CancelScope.shield` attribute
          of the newly created cancel scope.

    Raises:
      ValueError: if ``seconds`` is less than zero or NaN.

    """
    # duplicate validation logic to have the correct parameter name
    if seconds < 0:
        raise ValueError("`seconds` must be non-negative")
    if math.isnan(seconds):
        raise ValueError("`seconds` must not be NaN")
    return trio.CancelScope(
        shield=shield,
        relative_deadline=seconds,
    )


async def sleep_forever() -> NoReturn:
    """Pause execution of the current task forever (or until cancelled).

    Equivalent to calling ``await sleep(math.inf)``, except that if manually
    rescheduled this will raise a `RuntimeError`.

    Raises:
      RuntimeError: if rescheduled

    """
    await trio.lowlevel.wait_task_rescheduled(lambda _: trio.lowlevel.Abort.SUCCEEDED)
    raise RuntimeError("Should never have been rescheduled!")


async def sleep_until(deadline: float) -> None:
    """Pause execution of the current task until the given time.

    The difference between :func:`sleep` and :func:`sleep_until` is that the
    former takes a relative time and the latter takes an absolute time
    according to Trio's internal clock (as returned by :func:`current_time`).

    Args:
        deadline (float): The time at which we should wake up again. May be in
            the past, in which case this function executes a checkpoint but
            does not block.

    Raises:
      ValueError: if deadline is NaN.

    """
    with move_on_at(deadline):
        await sleep_forever()


async def sleep(seconds: float) -> None:
    """Pause execution of the current task for the given number of seconds.

    Args:
        seconds (float): The number of seconds to sleep. May be zero to
            insert a checkpoint without actually blocking.

    Raises:
        ValueError: if *seconds* is negative or NaN.

    """
    if seconds < 0:
        raise ValueError("`seconds` must be non-negative")
    if seconds == 0:
        await trio.lowlevel.checkpoint()
    else:
        await sleep_until(trio.current_time() + seconds)


class TooSlowError(Exception):
    """Raised by :func:`fail_after` and :func:`fail_at` if the timeout
    expires.

    """


@contextmanager
def fail_at(
    deadline: float,
    *,
    shield: bool = False,
) -> Generator[trio.CancelScope, None, None]:
    """Creates a cancel scope with the given deadline, and raises an error if it
    is actually cancelled.

    This function and :func:`move_on_at` are similar in that both create a
    cancel scope with a given absolute deadline, and if the deadline expires
    then both will cause :exc:`Cancelled` to be raised within the scope. The
    difference is that when the :exc:`Cancelled` exception reaches
    :func:`move_on_at`, it's caught and discarded. When it reaches
    :func:`fail_at`, then it's caught and :exc:`TooSlowError` is raised in its
    place.

    Args:
      deadline (float): The deadline.
      shield (bool): Initial value for the `~trio.CancelScope.shield` attribute
          of the newly created cancel scope.

    Raises:
      TooSlowError: if a :exc:`Cancelled` exception is raised in this scope
        and caught by the context manager.
      ValueError: if deadline is NaN.

    """
    with move_on_at(deadline, shield=shield) as scope:
        yield scope
    if scope.cancelled_caught:
        raise TooSlowError


@contextmanager
def fail_after(
    seconds: float,
    *,
    shield: bool = False,
) -> Generator[trio.CancelScope, None, None]:
    """Creates a cancel scope with the given timeout, and raises an error if
    it is actually cancelled.

    This function and :func:`move_on_after` are similar in that both create a
    cancel scope with a given timeout, and if the timeout expires then both
    will cause :exc:`Cancelled` to be raised within the scope. The difference
    is that when the :exc:`Cancelled` exception reaches :func:`move_on_after`,
    it's caught and discarded. When it reaches :func:`fail_after`, then it's
    caught and :exc:`TooSlowError` is raised in its place.

    The deadline of the cancel scope is calculated upon entering.

    Args:
      seconds (float): The timeout.
      shield (bool): Initial value for the `~trio.CancelScope.shield` attribute
          of the newly created cancel scope.

    Raises:
      TooSlowError: if a :exc:`Cancelled` exception is raised in this scope
        and caught by the context manager.
      ValueError: if *seconds* is less than zero or NaN.

    """
    with move_on_after(seconds, shield=shield) as scope:
        yield scope
    if scope.cancelled_caught:
        raise TooSlowError


# Users don't need to know that fail_at & fail_after wraps move_on_at and move_on_after
# and there is no functional difference. So we replace the return value when generating
# documentation.
if "sphinx" in sys.modules:  # pragma: no cover
    import inspect

    for c in (fail_at, fail_after):
        c.__signature__ = inspect.Signature.from_callable(c).replace(return_annotation=trio.CancelScope)  # type: ignore[union-attr]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\_typing.py ===
# Custom type aliases used throughout Beautiful Soup to improve readability.

# Notes on improvements to the type system in newer versions of Python
# that can be used once Beautiful Soup drops support for older
# versions:
#
# * ClassVar can be put on class variables now.
# * In 3.10, x|y is an accepted shorthand for Union[x,y].
# * In 3.10, TypeAlias gains capabilities that can be used to
#   improve the tree matching types (I don't remember what, exactly).
# * In 3.9 it's possible to specialize the re.Match type,
#   e.g. re.Match[str]. In 3.8 there's a typing.re namespace for this,
#   but it's removed in 3.12, so to support the widest possible set of
#   versions I'm not using it.

from typing_extensions import (
    runtime_checkable,
    Protocol,
    TypeAlias,
)
from typing import (
    Any,
    Callable,
    Dict,
    IO,
    Iterable,
    Mapping,
    Optional,
    Pattern,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from bs4.element import (
        AttributeValueList,
        NamespacedAttribute,
        NavigableString,
        PageElement,
        ResultSet,
        Tag,
    )


@runtime_checkable
class _RegularExpressionProtocol(Protocol):
    """A protocol object which can accept either Python's built-in
    `re.Pattern` objects, or the similar ``Regex`` objects defined by the
    third-party ``regex`` package.
    """

    def search(
        self, string: str, pos: int = ..., endpos: int = ...
    ) -> Optional[Any]: ...

    @property
    def pattern(self) -> str: ...


# Aliases for markup in various stages of processing.
#
#: The rawest form of markup: either a string, bytestring, or an open filehandle.
_IncomingMarkup: TypeAlias = Union[str, bytes, IO[str], IO[bytes]]

#: Markup that is in memory but has (potentially) yet to be converted
#: to Unicode.
_RawMarkup: TypeAlias = Union[str, bytes]

# Aliases for character encodings
#

#: A data encoding.
_Encoding: TypeAlias = str

#: One or more data encodings.
_Encodings: TypeAlias = Iterable[_Encoding]

# Aliases for XML namespaces
#

#: The prefix for an XML namespace.
_NamespacePrefix: TypeAlias = str

#: The URL of an XML namespace
_NamespaceURL: TypeAlias = str

#: A mapping of prefixes to namespace URLs.
_NamespaceMapping: TypeAlias = Dict[_NamespacePrefix, _NamespaceURL]

#: A mapping of namespace URLs to prefixes
_InvertedNamespaceMapping: TypeAlias = Dict[_NamespaceURL, _NamespacePrefix]

# Aliases for the attribute values associated with HTML/XML tags.
#

#: The value associated with an HTML or XML attribute. This is the
#: relatively unprocessed value Beautiful Soup expects to come from a
#: `TreeBuilder`.
_RawAttributeValue: TypeAlias = str

#: A dictionary of names to `_RawAttributeValue` objects. This is how
#: Beautiful Soup expects a `TreeBuilder` to represent a tag's
#: attribute values.
_RawAttributeValues: TypeAlias = (
    "Mapping[Union[str, NamespacedAttribute], _RawAttributeValue]"
)

#: An attribute value in its final form, as stored in the
# `Tag` class, after it has been processed and (in some cases)
# split into a list of strings.
_AttributeValue: TypeAlias = Union[str, "AttributeValueList"]

#: A dictionary of names to :py:data:`_AttributeValue` objects. This is what
#: a tag's attributes look like after processing.
_AttributeValues: TypeAlias = Dict[str, _AttributeValue]

#: The methods that deal with turning :py:data:`_RawAttributeValue` into
#: :py:data:`_AttributeValue` may be called several times, even after the values
#: are already processed (e.g. when cloning a tag), so they need to
#: be able to acommodate both possibilities.
_RawOrProcessedAttributeValues: TypeAlias = Union[_RawAttributeValues, _AttributeValues]

#: A number of tree manipulation methods can take either a `PageElement` or a
#: normal Python string (which will be converted to a `NavigableString`).
_InsertableElement: TypeAlias = Union["PageElement", str]

# Aliases to represent the many possibilities for matching bits of a
# parse tree.
#
# This is very complicated because we're applying a formal type system
# to some very DWIM code. The types we end up with will be the types
# of the arguments to the SoupStrainer constructor and (more
# familiarly to Beautiful Soup users) the find* methods.

#: A function that takes a PageElement and returns a yes-or-no answer.
_PageElementMatchFunction: TypeAlias = Callable[["PageElement"], bool]

#: A function that takes the raw parsed ingredients of a markup tag
#: and returns a yes-or-no answer.
#  Not necessary at the moment.
# _AllowTagCreationFunction:TypeAlias = Callable[[Optional[str], str, Optional[_RawAttributeValues]], bool]

#: A function that takes the raw parsed ingredients of a markup string node
#: and returns a yes-or-no answer.
#  Not necessary at the moment.
# _AllowStringCreationFunction:TypeAlias = Callable[[Optional[str]], bool]

#: A function that takes a `Tag` and returns a yes-or-no answer.
#: A `TagNameMatchRule` expects this kind of function, if you're
#: going to pass it a function.
_TagMatchFunction: TypeAlias = Callable[["Tag"], bool]

#: A function that takes a single string and returns a yes-or-no
#: answer. An `AttributeValueMatchRule` expects this kind of function, if
#: you're going to pass it a function. So does a `StringMatchRule`.
_StringMatchFunction: TypeAlias = Callable[[str], bool]

#: Either a tag name, an attribute value or a string can be matched
#: against a string, bytestring, regular expression, or a boolean.
_BaseStrainable: TypeAlias = Union[str, bytes, Pattern[str], bool]

#: A tag can be matched either with the `_BaseStrainable` options, or
#: using a function that takes the `Tag` as its sole argument.
_BaseStrainableElement: TypeAlias = Union[_BaseStrainable, _TagMatchFunction]

#: A tag's attribute vgalue can be matched either with the
#: `_BaseStrainable` options, or using a function that takes that
#: value as its sole argument.
_BaseStrainableAttribute: TypeAlias = Union[_BaseStrainable, _StringMatchFunction]

#: A tag can be matched using either a single criterion or a list of
#: criteria.
_StrainableElement: TypeAlias = Union[
    _BaseStrainableElement, Iterable[_BaseStrainableElement]
]

#: An attribute value can be matched using either a single criterion
#: or a list of criteria.
_StrainableAttribute: TypeAlias = Union[
    _BaseStrainableAttribute, Iterable[_BaseStrainableAttribute]
]

#: An string can be matched using the same techniques as
#: an attribute value.
_StrainableString: TypeAlias = _StrainableAttribute

#: A dictionary may be used to match against multiple attribute vlaues at once.
_StrainableAttributes: TypeAlias = Dict[str, _StrainableAttribute]

#: Many Beautiful soup methods return a PageElement or an ResultSet of
#: PageElements. A PageElement is either a Tag or a NavigableString.
#: These convenience aliases make it easier for IDE users to see which methods
#: are available on the objects they're dealing with.
_OneElement: TypeAlias = Union["PageElement", "Tag", "NavigableString"]
_AtMostOneElement: TypeAlias = Optional[_OneElement]
_QueryResults: TypeAlias = "ResultSet[_OneElement]"