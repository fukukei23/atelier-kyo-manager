
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\domainmatrix.py ===
"""

Module for the DomainMatrix class.

A DomainMatrix represents a matrix with elements that are in a particular
Domain. Each DomainMatrix internally wraps a DDM which is used for the
lower-level operations. The idea is that the DomainMatrix class provides the
convenience routines for converting between Expr and the poly domains as well
as unifying matrices with different domains.

"""
from __future__ import annotations
from collections import Counter
from functools import reduce

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from sympy.core.sympify import _sympify

from ..domains import Domain

from ..constructor import construct_domain

from .exceptions import (
    DMFormatError,
    DMBadInputError,
    DMShapeError,
    DMDomainError,
    DMNotAField,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError
)

from .domainscalar import DomainScalar

from sympy.polys.domains import ZZ, EXRAW, QQ

from sympy.polys.densearith import dup_mul
from sympy.polys.densebasic import dup_convert
from sympy.polys.densetools import (
    dup_mul_ground,
    dup_quo_ground,
    dup_content,
    dup_clear_denoms,
    dup_primitive,
    dup_transform,
)
from sympy.polys.factortools import dup_factor_list
from sympy.polys.polyutils import _sort_factors

from .ddm import DDM

from .sdm import SDM

from .dfm import DFM

from .rref import _dm_rref, _dm_rref_den


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['DomainMatrix.to_dfm', 'DomainMatrix.to_dfm_or_ddm']
else:
    __doctest_skip__ = ['DomainMatrix.from_list']


def DM(rows, domain):
    """Convenient alias for DomainMatrix.from_list

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DM
    >>> DM([[1, 2], [3, 4]], ZZ)
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DomainMatrix.from_list
    """
    return DomainMatrix.from_list(rows, domain)


class DomainMatrix:
    r"""
    Associate Matrix with :py:class:`~.Domain`

    Explanation
    ===========

    DomainMatrix uses :py:class:`~.Domain` for its internal representation
    which makes it faster than the SymPy Matrix class (currently) for many
    common operations, but this advantage makes it not entirely compatible
    with Matrix. DomainMatrix are analogous to numpy arrays with "dtype".
    In the DomainMatrix, each element has a domain such as :ref:`ZZ`
    or  :ref:`QQ(a)`.


    Examples
    ========

    Creating a DomainMatrix from the existing Matrix class:

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices import DomainMatrix
    >>> Matrix1 = Matrix([
    ...    [1, 2],
    ...    [3, 4]])
    >>> A = DomainMatrix.from_Matrix(Matrix1)
    >>> A
    DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    Directly forming a DomainMatrix:

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> A = DomainMatrix([
    ...    [ZZ(1), ZZ(2)],
    ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> A
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DDM
    SDM
    Domain
    Poly

    """
    rep: SDM | DDM | DFM
    shape: tuple[int, int]
    domain: Domain

    def __new__(cls, rows, shape, domain, *, fmt=None):
        """
        Creates a :py:class:`~.DomainMatrix`.

        Parameters
        ==========

        rows : Represents elements of DomainMatrix as list of lists
        shape : Represents dimension of DomainMatrix
        domain : Represents :py:class:`~.Domain` of DomainMatrix

        Raises
        ======

        TypeError
            If any of rows, shape and domain are not provided

        """
        if isinstance(rows, (DDM, SDM, DFM)):
            raise TypeError("Use from_rep to initialise from SDM/DDM")
        elif isinstance(rows, list):
            rep = DDM(rows, shape, domain)
        elif isinstance(rows, dict):
            rep = SDM(rows, shape, domain)
        else:
            msg = "Input should be list-of-lists or dict-of-dicts"
            raise TypeError(msg)

        if fmt is not None:
            if fmt == 'sparse':
                rep = rep.to_sdm()
            elif fmt == 'dense':
                rep = rep.to_ddm()
            else:
                raise ValueError("fmt should be 'sparse' or 'dense'")

        # Use python-flint for dense matrices if possible
        if rep.fmt == 'dense' and DFM._supports_domain(domain):
            rep = rep.to_dfm()

        return cls.from_rep(rep)

    def __reduce__(self):
        rep = self.rep
        if rep.fmt == 'dense':
            arg = self.to_list()
        elif rep.fmt == 'sparse':
            arg = dict(rep)
        else:
            raise RuntimeError # pragma: no cover
        args = (arg, rep.shape, rep.domain)
        return (self.__class__, args)

    def __getitem__(self, key):
        i, j = key
        m, n = self.shape
        if not (isinstance(i, slice) or isinstance(j, slice)):
            return DomainScalar(self.rep.getitem(i, j), self.domain)

        if not isinstance(i, slice):
            if not -m <= i < m:
                raise IndexError("Row index out of range")
            i = i % m
            i = slice(i, i+1)
        if not isinstance(j, slice):
            if not -n <= j < n:
                raise IndexError("Column index out of range")
            j = j % n
            j = slice(j, j+1)

        return self.from_rep(self.rep.extract_slice(i, j))

    def getitem_sympy(self, i, j):
        return self.domain.to_sympy(self.rep.getitem(i, j))

    def extract(self, rowslist, colslist):
        return self.from_rep(self.rep.extract(rowslist, colslist))

    def __setitem__(self, key, value):
        i, j = key
        if not self.domain.of_type(value):
            raise TypeError
        if isinstance(i, int) and isinstance(j, int):
            self.rep.setitem(i, j, value)
        else:
            raise NotImplementedError

    @classmethod
    def from_rep(cls, rep):
        """Create a new DomainMatrix efficiently from DDM/SDM.

        Examples
        ========

        Create a :py:class:`~.DomainMatrix` with an dense internal
        representation as :py:class:`~.DDM`:

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.matrices.ddm import DDM
        >>> drep = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> dM = DomainMatrix.from_rep(drep)
        >>> dM
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

        Create a :py:class:`~.DomainMatrix` with a sparse internal
        representation as :py:class:`~.SDM`:

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import ZZ
        >>> drep = SDM({0:{1:ZZ(1)},1:{0:ZZ(2)}}, (2, 2), ZZ)
        >>> dM = DomainMatrix.from_rep(drep)
        >>> dM
        DomainMatrix({0: {1: 1}, 1: {0: 2}}, (2, 2), ZZ)

        Parameters
        ==========

        rep: SDM or DDM
            The internal sparse or dense representation of the matrix.

        Returns
        =======

        DomainMatrix
            A :py:class:`~.DomainMatrix` wrapping *rep*.

        Notes
        =====

        This takes ownership of rep as its internal representation. If rep is
        being mutated elsewhere then a copy should be provided to
        ``from_rep``. Only minimal verification or checking is done on *rep*
        as this is supposed to be an efficient internal routine.

        """
        if not (isinstance(rep, (DDM, SDM)) or (DFM is not None and isinstance(rep, DFM))):
            raise TypeError("rep should be of type DDM or SDM")
        self = super().__new__(cls)
        self.rep = rep
        self.shape = rep.shape
        self.domain = rep.domain
        return self

    @classmethod
    @doctest_depends_on(ground_types=['python', 'gmpy'])
    def from_list(cls, rows, domain):
        r"""
        Convert a list of lists into a DomainMatrix

        Parameters
        ==========

        rows: list of lists
            Each element of the inner lists should be either the single arg,
            or tuple of args, that would be passed to the domain constructor
            in order to form an element of the domain. See examples.

        Returns
        =======

        DomainMatrix containing elements defined in rows

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import FF, QQ, ZZ
        >>> A = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], ZZ)
        >>> A
        DomainMatrix([[1, 0, 1], [0, 0, 1]], (2, 3), ZZ)
        >>> B = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], FF(7))
        >>> B
        DomainMatrix([[1 mod 7, 0 mod 7, 1 mod 7], [0 mod 7, 0 mod 7, 1 mod 7]], (2, 3), GF(7))
        >>> C = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)
        >>> C
        DomainMatrix([[1/2, 3], [1/4, 5]], (2, 2), QQ)

        See Also
        ========

        from_list_sympy

        """
        nrows = len(rows)
        ncols = 0 if not nrows else len(rows[0])
        conv = lambda e: domain(*e) if isinstance(e, tuple) else domain(e)
        domain_rows = [[conv(e) for e in row] for row in rows]
        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_list_sympy(cls, nrows, ncols, rows, **kwargs):
        r"""
        Convert a list of lists of Expr into a DomainMatrix using construct_domain

        Parameters
        ==========

        nrows: number of rows
        ncols: number of columns
        rows: list of lists

        Returns
        =======

        DomainMatrix containing elements of rows

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x, y, z
        >>> A = DomainMatrix.from_list_sympy(1, 3, [[x, y, z]])
        >>> A
        DomainMatrix([[x, y, z]], (1, 3), ZZ[x,y,z])

        See Also
        ========

        sympy.polys.constructor.construct_domain, from_dict_sympy

        """
        assert len(rows) == nrows
        assert all(len(row) == ncols for row in rows)

        items_sympy = [_sympify(item) for row in rows for item in row]

        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        domain_rows = [[items_domain[ncols*r + c] for c in range(ncols)] for r in range(nrows)]

        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_dict_sympy(cls, nrows, ncols, elemsdict, **kwargs):
        """

        Parameters
        ==========

        nrows: number of rows
        ncols: number of cols
        elemsdict: dict of dicts containing non-zero elements of the DomainMatrix

        Returns
        =======

        DomainMatrix containing elements of elemsdict

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x,y,z
        >>> elemsdict = {0: {0:x}, 1:{1: y}, 2: {2: z}}
        >>> A = DomainMatrix.from_dict_sympy(3, 3, elemsdict)
        >>> A
        DomainMatrix({0: {0: x}, 1: {1: y}, 2: {2: z}}, (3, 3), ZZ[x,y,z])

        See Also
        ========

        from_list_sympy

        """
        if not all(0 <= r < nrows for r in elemsdict):
            raise DMBadInputError("Row out of range")
        if not all(0 <= c < ncols for row in elemsdict.values() for c in row):
            raise DMBadInputError("Column out of range")

        items_sympy = [_sympify(item) for row in elemsdict.values() for item in row.values()]
        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        idx = 0
        items_dict = {}
        for i, row in elemsdict.items():
            items_dict[i] = {}
            for j in row:
                items_dict[i][j] = items_domain[idx]
                idx += 1

        return DomainMatrix(items_dict, (nrows, ncols), domain)

    @classmethod
    def from_Matrix(cls, M, fmt='sparse',**kwargs):
        r"""
        Convert Matrix to DomainMatrix

        Parameters
        ==========

        M: Matrix

        Returns
        =======

        Returns DomainMatrix with identical elements as M

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = Matrix([
        ...    [1.0, 3.4],
        ...    [2.4, 1]])
        >>> A = DomainMatrix.from_Matrix(M)
        >>> A
        DomainMatrix({0: {0: 1.0, 1: 3.4}, 1: {0: 2.4, 1: 1.0}}, (2, 2), RR)

        We can keep internal representation as ddm using fmt='dense'
        >>> from sympy import Matrix, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_Matrix(Matrix([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]]), fmt='dense')
        >>> A.rep
        [[1/2, 3/4], [0, 0]]

        See Also
        ========

        Matrix

        """
        if fmt == 'dense':
            return cls.from_list_sympy(*M.shape, M.tolist(), **kwargs)

        return cls.from_dict_sympy(*M.shape, M.todod(), **kwargs)

    @classmethod
    def get_domain(cls, items_sympy, **kwargs):
        K, items_K = construct_domain(items_sympy, **kwargs)
        return K, items_K

    def choose_domain(self, **opts):
        """Convert to a domain found by :func:`~.construct_domain`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[1, 2], [3, 4]], ZZ)
        >>> M
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> M.choose_domain(field=True)
        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)

        >>> from sympy.abc import x
        >>> M = DM([[1, x], [x**2, x**3]], ZZ[x])
        >>> M.choose_domain(field=True).domain
        ZZ(x)

        Keyword arguments are passed to :func:`~.construct_domain`.

        See Also
        ========

        construct_domain
        convert_to
        """
        elements, data = self.to_sympy().to_flat_nz()
        dom, elements_dom = construct_domain(elements, **opts)
        return self.from_flat_nz(elements_dom, data, dom)

    def copy(self):
        return self.from_rep(self.rep.copy())

    def convert_to(self, K):
        r"""
        Change the domain of DomainMatrix to desired domain or field

        Parameters
        ==========

        K : Represents the desired domain or field.
            Alternatively, ``None`` may be passed, in which case this method
            just returns a copy of this DomainMatrix.

        Returns
        =======

        DomainMatrix
            DomainMatrix with the desired domain or field

        Examples
        ========

        >>> from sympy import ZZ, ZZ_I
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.convert_to(ZZ_I)
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ_I)

        """
        if K == self.domain:
            return self.copy()

        rep = self.rep

        # The DFM, DDM and SDM types do not do any implicit conversions so we
        # manage switching between DDM and DFM here.
        if rep.is_DFM and not DFM._supports_domain(K):
            rep_K = rep.to_ddm().convert_to(K)
        elif rep.is_DDM and DFM._supports_domain(K):
            rep_K = rep.convert_to(K).to_dfm()
        else:
            rep_K = rep.convert_to(K)

        return self.from_rep(rep_K)

    def to_sympy(self):
        return self.convert_to(EXRAW)

    def to_field(self):
        r"""
        Returns a DomainMatrix with the appropriate field

        Returns
        =======

        DomainMatrix
            DomainMatrix with the appropriate field

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.to_field()
        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)

        """
        K = self.domain.get_field()
        return self.convert_to(K)

    def to_sparse(self):
        """
        Return a sparse DomainMatrix representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> A.rep
        [[1, 0], [0, 2]]
        >>> B = A.to_sparse()
        >>> B.rep
        {0: {0: 1}, 1: {1: 2}}
        """
        if self.rep.fmt == 'sparse':
            return self

        return self.from_rep(self.rep.to_sdm())

    def to_dense(self):
        """
        Return a dense DomainMatrix representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)
        >>> A.rep
        {0: {0: 1}, 1: {1: 2}}
        >>> B = A.to_dense()
        >>> B.rep
        [[1, 0], [0, 2]]

        """
        rep = self.rep

        if rep.fmt == 'dense':
            return self

        return self.from_rep(rep.to_dfm_or_ddm())

    def to_ddm(self):
        """
        Return a :class:`~.DDM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)
        >>> ddm = A.to_ddm()
        >>> ddm
        [[1, 0], [0, 2]]
        >>> type(ddm)
        <class 'sympy.polys.matrices.ddm.DDM'>

        See Also
        ========

        to_sdm
        to_dense
        sympy.polys.matrices.ddm.DDM.to_sdm
        """
        return self.rep.to_ddm()

    def to_sdm(self):
        """
        Return a :class:`~.SDM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> sdm = A.to_sdm()
        >>> sdm
        {0: {0: 1}, 1: {1: 2}}
        >>> type(sdm)
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        to_ddm
        to_sparse
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
        return self.rep.to_sdm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        """
        Return a :class:`~.DFM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> dfm = A.to_dfm()
        >>> dfm
        [[1, 0], [0, 2]]
        >>> type(dfm)
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm
        to_dense
        DFM
        """
        return self.rep.to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        """
        Return a :class:`~.DFM` or :class:`~.DDM` representation of *self*.

        Explanation
        ===========

        The :class:`~.DFM` representation can only be used if the ground types
        are ``flint`` and the ground domain is supported by ``python-flint``.
        This method will return a :class:`~.DFM` representation if possible,
        but will return a :class:`~.DDM` representation otherwise.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> dfm = A.to_dfm_or_ddm()
        >>> dfm
        [[1, 0], [0, 2]]
        >>> type(dfm)  # Depends on the ground domain and ground types
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm: Always return a :class:`~.DDM` representation.
        to_dfm: Returns a :class:`~.DFM` representation or raise an error.
        to_dense: Convert internally to a :class:`~.DFM` or :class:`~.DDM`
        DFM: The :class:`~.DFM` dense FLINT matrix representation.
        DDM: The Python :class:`~.DDM` dense domain matrix representation.
        """
        return self.rep.to_dfm_or_ddm()

    @classmethod
    def _unify_domain(cls, *matrices):
        """Convert matrices to a common domain"""
        domains = {matrix.domain for matrix in matrices}
        if len(domains) == 1:
            return matrices
        domain = reduce(lambda x, y: x.unify(y), domains)
        return tuple(matrix.convert_to(domain) for matrix in matrices)

    @classmethod
    def _unify_fmt(cls, *matrices, fmt=None):
        """Convert matrices to the same format.

        If all matrices have the same format, then return unmodified.
        Otherwise convert both to the preferred format given as *fmt* which
        should be 'dense' or 'sparse'.
        """
        formats = {matrix.rep.fmt for matrix in matrices}
        if len(formats) == 1:
            return matrices
        if fmt == 'sparse':
            return tuple(matrix.to_sparse() for matrix in matrices)
        elif fmt == 'dense':
            return tuple(matrix.to_dense() for matrix in matrices)
        else:
            raise ValueError("fmt should be 'sparse' or 'dense'")

    def unify(self, *others, fmt=None):
        """
        Unifies the domains and the format of self and other
        matrices.

        Parameters
        ==========

        others : DomainMatrix

        fmt: string 'dense', 'sparse' or `None` (default)
            The preferred format to convert to if self and other are not
            already in the same format. If `None` or not specified then no
            conversion if performed.

        Returns
        =======

        Tuple[DomainMatrix]
            Matrices with unified domain and format

        Examples
        ========

        Unify the domain of DomainMatrix that have different domains:

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
        >>> B = DomainMatrix([[QQ(1, 2), QQ(2)]], (1, 2), QQ)
        >>> Aq, Bq = A.unify(B)
        >>> Aq
        DomainMatrix([[1, 2]], (1, 2), QQ)
        >>> Bq
        DomainMatrix([[1/2, 2]], (1, 2), QQ)

        Unify the format (dense or sparse):

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
        >>> B = DomainMatrix({0:{0: ZZ(1)}}, (2, 2), ZZ)
        >>> B.rep
        {0: {0: 1}}

        >>> A2, B2 = A.unify(B, fmt='dense')
        >>> B2.rep
        [[1, 0], [0, 0]]

        See Also
        ========

        convert_to, to_dense, to_sparse

        """
        matrices = (self,) + others
        matrices = DomainMatrix._unify_domain(*matrices)
        if fmt is not None:
            matrices = DomainMatrix._unify_fmt(*matrices, fmt=fmt)
        return matrices

    def to_Matrix(self):
        r"""
        Convert DomainMatrix to Matrix

        Returns
        =======

        Matrix
            MutableDenseMatrix for the DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.to_Matrix()
        Matrix([
            [1, 2],
            [3, 4]])

        See Also
        ========

        from_Matrix

        """
        from sympy.matrices.dense import MutableDenseMatrix

        # XXX: If the internal representation of RepMatrix changes then this
        # might need to be changed also.
        if self.domain in (ZZ, QQ, EXRAW):
            if self.rep.fmt == "sparse":
                rep = self.copy()
            else:
                rep = self.to_sparse()
        else:
            rep = self.convert_to(EXRAW).to_sparse()

        return MutableDenseMatrix._fromrep(rep)

    def to_list(self):
        """
        Convert :class:`DomainMatrix` to list of lists.

        See Also
        ========

        from_list
        to_list_flat
        to_flat_nz
        to_dok
        """
        return self.rep.to_list()

    def to_list_flat(self):
        """
        Convert :class:`DomainMatrix` to flat list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]

        See Also
        ========

        from_list_flat
        to_list
        to_flat_nz
        to_dok
        """
        return self.rep.to_list_flat()

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Create :class:`DomainMatrix` from flat list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> element_list = [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
        >>> A = DomainMatrix.from_list_flat(element_list, (2, 2), ZZ)
        >>> A
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        """
        ddm = DDM.from_list_flat(elements, shape, domain)
        return cls.from_rep(ddm.to_dfm_or_ddm())

    def to_flat_nz(self):
        """
        Convert :class:`DomainMatrix` to list of nonzero elements and data.

        Explanation
        ===========

        Returns a tuple ``(elements, data)`` where ``elements`` is a list of
        elements of the matrix with zeros possibly excluded. The matrix can be
        reconstructed by passing these to :meth:`from_flat_nz`. The idea is to
        be able to modify a flat list of the elements and then create a new
        matrix of the same shape with the modified elements in the same
        positions.

        The format of ``data`` differs depending on whether the underlying
        representation is dense or sparse but either way it represents the
        positions of the elements in the list in a way that
        :meth:`from_flat_nz` can use to reconstruct the matrix. The
        :meth:`from_flat_nz` method should be called on the same
        :class:`DomainMatrix` that was used to call :meth:`to_flat_nz`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        Create a matrix with the elements doubled:

        >>> elements_doubled = [2*x for x in elements]
        >>> A2 = A.from_flat_nz(elements_doubled, data, A.domain)
        >>> A2 == 2*A
        True

        See Also
        ========

        from_flat_nz
        """
        return self.rep.to_flat_nz()

    def from_flat_nz(self, elements, data, domain):
        """
        Reconstruct :class:`DomainMatrix` after calling :meth:`to_flat_nz`.

        See :meth:`to_flat_nz` for explanation.

        See Also
        ========

        to_flat_nz
        """
        rep = self.rep.from_flat_nz(elements, data, domain)
        return self.from_rep(rep)

    def to_dod(self):
        """
        Convert :class:`DomainMatrix` to dictionary of dictionaries (dod) format.

        Explanation
        ===========

        Returns a dictionary of dictionaries representing the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2), ZZ(0)], [ZZ(3), ZZ(0), ZZ(4)]], ZZ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 2: 4}}
        >>> A.to_sparse() == A.from_dod(A.to_dod(), A.shape, A.domain)
        True
        >>> A == A.from_dod_like(A.to_dod())
        True

        See Also
        ========

        from_dod
        from_dod_like
        to_dok
        to_list
        to_list_flat
        to_flat_nz
        sympy.matrices.matrixbase.MatrixBase.todod
        """
        return self.rep.to_dod()

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create sparse :class:`DomainMatrix` from dict of dict (dod) format.

        See :meth:`to_dod` for explanation.

        See Also
        ========

        to_dod
        from_dod_like
        """
        return cls.from_rep(SDM.from_dod(dod, shape, domain))

    def from_dod_like(self, dod, domain=None):
        """
        Create :class:`DomainMatrix` like ``self`` from dict of dict (dod) format.

        See :meth:`to_dod` for explanation.

        See Also
        ========

        to_dod
        from_dod
        """
        if domain is None:
            domain = self.domain
        return self.from_rep(self.rep.from_dod(dod, self.shape, domain))

    def to_dok(self):
        """
        Convert :class:`DomainMatrix` to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(0)],
        ...    [ZZ(0), ZZ(4)]], (2, 2), ZZ)
        >>> A.to_dok()
        {(0, 0): 1, (1, 1): 4}

        The matrix can be reconstructed by calling :meth:`from_dok` although
        the reconstructed matrix will always be in sparse format:

        >>> A.to_sparse() == A.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_flat_nz
        """
        return self.rep.to_dok()

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create :class:`DomainMatrix` from dictionary of keys (dok) format.

        See :meth:`to_dok` for explanation.

        See Also
        ========

        to_dok
        """
        return cls.from_rep(SDM.from_dok(dok, shape, domain))

    def iter_values(self):
        """
        Iterate over nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.matrices.matrixbase.MatrixBase.iter_values
        """
        return self.rep.iter_values()

    def iter_items(self):
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.matrices.matrixbase.MatrixBase.iter_items
        """
        return self.rep.iter_items()

    def nnz(self):
        """
        Number of nonzero elements in the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[1, 0], [0, 4]], ZZ)
        >>> A.nnz()
        2
        """
        return self.rep.nnz()

    def __repr__(self):
        return 'DomainMatrix(%s, %r, %r)' % (str(self.rep), self.shape, self.domain)

    def transpose(self):
        """Matrix transpose of ``self``"""
        return self.from_rep(self.rep.transpose())

    def flat(self):
        rows, cols = self.shape
        return [self[i,j].element for i in range(rows) for j in range(cols)]

    @property
    def is_zero_matrix(self):
        return self.rep.is_zero_matrix()

    @property
    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        return self.rep.is_upper()

    @property
    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        return self.rep.is_lower()

    @property
    def is_diagonal(self):
        """
        True if the matrix is diagonal.

        Can return true for non-square matrices. A matrix is diagonal if
        ``M[i,j] == 0`` whenever ``i != j``.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], ZZ)
        >>> M.is_diagonal
        True

        See Also
        ========

        is_upper
        is_lower
        is_square
        diagonal
        """
        return self.rep.is_diagonal()

    def diagonal(self):
        """
        Get the diagonal entries of the matrix as a list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> M.diagonal()
        [1, 4]

        See Also
        ========

        is_diagonal
        diag
        """
        return self.rep.diagonal()

    @property
    def is_square(self):
        """
        True if the matrix is square.
        """
        return self.shape[0] == self.shape[1]

    def rank(self):
        rref, pivots = self.rref()
        return len(pivots)

    def hstack(A, *B):
        r"""Horizontally stack the given matrices.

        Parameters
        ==========

        B: DomainMatrix
            Matrices to stack horizontally.

        Returns
        =======

        DomainMatrix
            DomainMatrix by stacking horizontally.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        DomainMatrix([[1, 2, 5, 6], [3, 4, 7, 8]], (2, 4), ZZ)

        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        DomainMatrix([[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]], (2, 6), ZZ)

        See Also
        ========

        unify
        """
        A, *B = A.unify(*B, fmt=A.rep.fmt)
        return DomainMatrix.from_rep(A.rep.hstack(*(Bk.rep for Bk in B)))

    def vstack(A, *B):
        r"""Vertically stack the given matrices.

        Parameters
        ==========

        B: DomainMatrix
            Matrices to stack vertically.

        Returns
        =======

        DomainMatrix
            DomainMatrix by stacking vertically.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.vstack(B)
        DomainMatrix([[1, 2], [3, 4], [5, 6], [7, 8]], (4, 2), ZZ)

        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.vstack(B, C)
        DomainMatrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], (6, 2), ZZ)

        See Also
        ========

        unify
        """
        A, *B = A.unify(*B, fmt='dense')
        return DomainMatrix.from_rep(A.rep.vstack(*(Bk.rep for Bk in B)))

    def applyfunc(self, func, domain=None):
        if domain is None:
            domain = self.domain
        return self.from_rep(self.rep.applyfunc(func, domain))

    def __add__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        A, B = A.unify(B, fmt='dense')
        return A.add(B)

    def __sub__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        A, B = A.unify(B, fmt='dense')
        return A.sub(B)

    def __neg__(A):
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        if isinstance(B, DomainMatrix):
            A, B = A.unify(B, fmt='dense')
            return A.matmul(B)
        elif B in A.domain:
            return A.scalarmul(B)
        elif isinstance(B, DomainScalar):
            A, B = A.unify(B)
            return A.scalarmul(B.element)
        else:
            return NotImplemented

    def __rmul__(A, B):
        if B in A.domain:
            return A.rscalarmul(B)
        elif isinstance(B, DomainScalar):
            A, B = A.unify(B)
            return A.rscalarmul(B.element)
        else:
            return NotImplemented

    def __pow__(A, n):
        """A ** n"""
        if not isinstance(n, int):
            return NotImplemented
        return A.pow(n)

    def _check(a, op, b, ashape, bshape):
        if a.domain != b.domain:
            msg = "Domain mismatch: %s %s %s" % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        if ashape != bshape:
            msg = "Shape mismatch: %s %s %s" % (a.shape, op, b.shape)
            raise DMShapeError(msg)
        if a.rep.fmt != b.rep.fmt:
            msg = "Format mismatch: %s %s %s" % (a.rep.fmt, op, b.rep.fmt)
            raise DMFormatError(msg)
        if type(a.rep) != type(b.rep):
            msg = "Type mismatch: %s %s %s" % (type(a.rep), op, type(b.rep))
            raise DMFormatError(msg)

    def add(A, B):
        r"""
        Adds two DomainMatrix matrices of the same Domain

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to add

        Returns
        =======

        DomainMatrix
            DomainMatrix after Addition

        Raises
        ======

        DMShapeError
            If the dimensions of the two DomainMatrix are not equal

        ValueError
            If the domain of the two DomainMatrix are not same

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(4), ZZ(3)],
        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)

        >>> A.add(B)
        DomainMatrix([[5, 5], [5, 5]], (2, 2), ZZ)

        See Also
        ========

        sub, matmul

        """
        A._check('+', B, A.shape, B.shape)
        return A.from_rep(A.rep.add(B.rep))


    def sub(A, B):
        r"""
        Subtracts two DomainMatrix matrices of the same Domain

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to subtract

        Returns
        =======

        DomainMatrix
            DomainMatrix after Subtraction

        Raises
        ======

        DMShapeError
            If the dimensions of the two DomainMatrix are not equal

        ValueError
            If the domain of the two DomainMatrix are not same

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(4), ZZ(3)],
        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)

        >>> A.sub(B)
        DomainMatrix([[-3, -1], [1, 3]], (2, 2), ZZ)

        See Also
        ========

        add, matmul

        """
        A._check('-', B, A.shape, B.shape)
        return A.from_rep(A.rep.sub(B.rep))

    def neg(A):
        r"""
        Returns the negative of DomainMatrix

        Parameters
        ==========

        A : Represents a DomainMatrix

        Returns
        =======

        DomainMatrix
            DomainMatrix after Negation

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.neg()
        DomainMatrix([[-1, -2], [-3, -4]], (2, 2), ZZ)

        """
        return A.from_rep(A.rep.neg())

    def mul(A, b):
        r"""
        Performs term by term multiplication for the second DomainMatrix
        w.r.t first DomainMatrix. Returns a DomainMatrix whose rows are
        list of DomainMatrix matrices created after term by term multiplication.

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to multiply term-wise

        Returns
        =======

        DomainMatrix
            DomainMatrix after term by term multiplication

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> b = ZZ(2)

        >>> A.mul(b)
        DomainMatrix([[2, 4], [6, 8]], (2, 2), ZZ)

        See Also
        ========

        matmul

        """
        return A.from_rep(A.rep.mul(b))

    def rmul(A, b):
        return A.from_rep(A.rep.rmul(b))

    def matmul(A, B):
        r"""
        Performs matrix multiplication of two DomainMatrix matrices

        Parameters
        ==========

        A, B: DomainMatrix
            to multiply

        Returns
        =======

        DomainMatrix
            DomainMatrix after multiplication

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)

        >>> A.matmul(B)
        DomainMatrix([[1, 3], [3, 7]], (2, 2), ZZ)

        See Also
        ========

        mul, pow, add, sub

        """

        A._check('*', B, A.shape[1], B.shape[0])
        return A.from_rep(A.rep.matmul(B.rep))

    def _scalarmul(A, lamda, reverse):
        if lamda == A.domain.zero:
            return DomainMatrix.zeros(A.shape, A.domain)
        elif lamda == A.domain.one:
            return A.copy()
        elif reverse:
            return A.rmul(lamda)
        else:
            return A.mul(lamda)

    def scalarmul(A, lamda):
        return A._scalarmul(lamda, reverse=False)

    def rscalarmul(A, lamda):
        return A._scalarmul(lamda, reverse=True)

    def mul_elementwise(A, B):
        assert A.domain == B.domain
        return A.from_rep(A.rep.mul_elementwise(B.rep))

    def __truediv__(A, lamda):
        """ Method for Scalar Division"""
        if isinstance(lamda, int) or ZZ.of_type(lamda):
            lamda = DomainScalar(ZZ(lamda), ZZ)
        elif A.domain.is_Field and lamda in A.domain:
            K = A.domain
            lamda = DomainScalar(K.convert(lamda), K)

        if not isinstance(lamda, DomainScalar):
            return NotImplemented

        A, lamda = A.to_field().unify(lamda)
        if lamda.element == lamda.domain.zero:
            raise ZeroDivisionError
        if lamda.element == lamda.domain.one:
            return A

        return A.mul(1 / lamda.element)

    def pow(A, n):
        r"""
        Computes A**n

        Parameters
        ==========

        A : DomainMatrix

        n : exponent for A

        Returns
        =======

        DomainMatrix
            DomainMatrix on computing A**n

        Raises
        ======

        NotImplementedError
            if n is negative.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)

        >>> A.pow(2)
        DomainMatrix([[1, 2], [0, 1]], (2, 2), ZZ)

        See Also
        ========

        matmul

        """
        nrows, ncols = A.shape
        if nrows != ncols:
            raise DMNonSquareMatrixError('Power of a nonsquare matrix')
        if n < 0:
            raise NotImplementedError('Negative powers')
        elif n == 0:
            return A.eye(nrows, A.domain)
        elif n == 1:
            return A
        elif n % 2 == 1:
            return A * A**(n - 1)
        else:
            sqrtAn = A ** (n // 2)
            return sqrtAn * sqrtAn

    def scc(self):
        """Compute the strongly connected components of a DomainMatrix

        Explanation
        ===========

        A square matrix can be considered as the adjacency matrix for a
        directed graph where the row and column indices are the vertices. In
        this graph if there is an edge from vertex ``i`` to vertex ``j`` if
        ``M[i, j]`` is nonzero. This routine computes the strongly connected
        components of that graph which are subsets of the rows and columns that
        are connected by some nonzero element of the matrix. The strongly
        connected components are useful because many operations such as the
        determinant can be computed by working with the submatrices
        corresponding to each component.

        Examples
        ========

        Find the strongly connected components of a matrix:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = DomainMatrix([[ZZ(1), ZZ(0), ZZ(2)],
        ...                   [ZZ(0), ZZ(3), ZZ(0)],
        ...                   [ZZ(4), ZZ(6), ZZ(5)]], (3, 3), ZZ)
        >>> M.scc()
        [[1], [0, 2]]

        Compute the determinant from the components:

        >>> MM = M.to_Matrix()
        >>> MM
        Matrix([
        [1, 0, 2],
        [0, 3, 0],
        [4, 6, 5]])
        >>> MM[[1], [1]]
        Matrix([[3]])
        >>> MM[[0, 2], [0, 2]]
        Matrix([
        [1, 2],
        [4, 5]])
        >>> MM.det()
        -9
        >>> MM[[1], [1]].det() * MM[[0, 2], [0, 2]].det()
        -9

        The components are given in reverse topological order and represent a
        permutation of the rows and columns that will bring the matrix into
        block lower-triangular form:

        >>> MM[[1, 0, 2], [1, 0, 2]]
        Matrix([
        [3, 0, 0],
        [0, 1, 2],
        [6, 4, 5]])

        Returns
        =======

        List of lists of integers
            Each list represents a strongly connected component.

        See also
        ========

        sympy.matrices.matrixbase.MatrixBase.strongly_connected_components
        sympy.utilities.iterables.strongly_connected_components

        """
        if not self.is_square:
            raise DMNonSquareMatrixError('Matrix must be square for scc')

        return self.rep.scc()

    def clear_denoms(self, convert=False):
        """
        Clear denominators, but keep the domain unchanged.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[(1,2), (1,3)], [(1,4), (1,5)]], QQ)
        >>> den, Anum = A.clear_denoms()
        >>> den.to_sympy()
        60
        >>> Anum.to_Matrix()
        Matrix([
        [30, 20],
        [15, 12]])
        >>> den * A == Anum
        True

        The numerator matrix will be in the same domain as the original matrix
        unless ``convert`` is set to ``True``:

        >>> A.clear_denoms()[1].domain
        QQ
        >>> A.clear_denoms(convert=True)[1].domain
        ZZ

        The denominator is always in the associated ring:

        >>> A.clear_denoms()[0].domain
        ZZ
        >>> A.domain.get_ring()
        ZZ

        See Also
        ========

        sympy.polys.polytools.Poly.clear_denoms
        clear_denoms_rowwise
        """
        elems0, data = self.to_flat_nz()

        K0 = self.domain
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0

        den, elems1 = dup_clear_denoms(elems0, K0, K1, convert=convert)

        if convert:
            Kden, Knum = K1, K1
        else:
            Kden, Knum = K1, K0

        den = DomainScalar(den, Kden)
        num = self.from_flat_nz(elems1, data, Knum)

        return den, num

    def clear_denoms_rowwise(self, convert=False):
        """
        Clear denominators from each row of the matrix.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[(1,2), (1,3), (1,4)], [(1,5), (1,6), (1,7)]], QQ)
        >>> den, Anum = A.clear_denoms_rowwise()
        >>> den.to_Matrix()
        Matrix([
        [12,   0],
        [ 0, 210]])
        >>> Anum.to_Matrix()
        Matrix([
        [ 6,  4,  3],
        [42, 35, 30]])

        The denominator matrix is a diagonal matrix with the denominators of
        each row on the diagonal. The invariants are:

        >>> den * A == Anum
        True
        >>> A == den.to_field().inv() * Anum
        True

        The numerator matrix will be in the same domain as the original matrix
        unless ``convert`` is set to ``True``:

        >>> A.clear_denoms_rowwise()[1].domain
        QQ
        >>> A.clear_denoms_rowwise(convert=True)[1].domain
        ZZ

        The domain of the denominator matrix is the associated ring:

        >>> A.clear_denoms_rowwise()[0].domain
        ZZ

        See Also
        ========

        sympy.polys.polytools.Poly.clear_denoms
        clear_denoms
        """
        dod = self.to_dod()

        K0 = self.domain
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0

        diagonals = [K0.one] * self.shape[0]
        dod_num = {}
        for i, rowi in dod.items():
            indices, elems = zip(*rowi.items())
            den, elems_num = dup_clear_denoms(elems, K0, K1, convert=convert)
            rowi_num = dict(zip(indices, elems_num))
            diagonals[i] = den
            dod_num[i] = rowi_num

        if convert:
            Kden, Knum = K1, K1
        else:
            Kden, Knum = K1, K0

        den = self.diag(diagonals, Kden)
        num = self.from_dod_like(dod_num, Knum)

        return den, num

    def cancel_denom(self, denom):
        """
        Cancel factors between a matrix and a denominator.

        Returns a matrix and denominator on lowest terms.

        Requires ``gcd`` in the ground domain.

        Methods like :meth:`solve_den`, :meth:`inv_den` and :meth:`rref_den`
        return a matrix and denominator but not necessarily on lowest terms.
        Reduction to lowest terms without fractions can be performed with
        :meth:`cancel_denom`.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 2, 0],
        ...         [0, 2, 2],
        ...         [0, 0, 2]], ZZ)
        >>> Minv, den = M.inv_den()
        >>> Minv.to_Matrix()
        Matrix([
        [1, -1,  1],
        [0,  1, -1],
        [0,  0,  1]])
        >>> den
        2
        >>> Minv_reduced, den_reduced = Minv.cancel_denom(den)
        >>> Minv_reduced.to_Matrix()
        Matrix([
        [1, -1,  1],
        [0,  1, -1],
        [0,  0,  1]])
        >>> den_reduced
        2
        >>> Minv_reduced.to_field() / den_reduced == Minv.to_field() / den
        True

        The denominator is made canonical with respect to units (e.g. a
        negative denominator is made positive):

        >>> M = DM([[2, 2, 0]], ZZ)
        >>> den = ZZ(-4)
        >>> M.cancel_denom(den)
        (DomainMatrix([[-1, -1, 0]], (1, 3), ZZ), 2)

        Any factor common to _all_ elements will be cancelled but there can
        still be factors in common between _some_ elements of the matrix and
        the denominator. To cancel factors between each element and the
        denominator, use :meth:`cancel_denom_elementwise` or otherwise convert
        to a field and use division:

        >>> M = DM([[4, 6]], ZZ)
        >>> den = ZZ(12)
        >>> M.cancel_denom(den)
        (DomainMatrix([[2, 3]], (1, 2), ZZ), 6)
        >>> numers, denoms = M.cancel_denom_elementwise(den)
        >>> numers
        DomainMatrix([[1, 1]], (1, 2), ZZ)
        >>> denoms
        DomainMatrix([[3, 2]], (1, 2), ZZ)
        >>> M.to_field() / den
        DomainMatrix([[1/3, 1/2]], (1, 2), QQ)

        See Also
        ========

        solve_den
        inv_den
        rref_den
        cancel_denom_elementwise
        """
        M = self
        K = self.domain

        if K.is_zero(denom):
            raise ZeroDivisionError('denominator is zero')
        elif K.is_one(denom):
            return (M.copy(), denom)

        elements, data = M.to_flat_nz()

        # First canonicalize the denominator (e.g. multiply by -1).
        if K.is_negative(denom):
            u = -K.one
        else:
            u = K.canonical_unit(denom)

        # Often after e.g. solve_den the denominator will be much more
        # complicated than the elements of the numerator. Hopefully it will be
        # quicker to find the gcd of the numerator and if there is no content
        # then we do not need to look at the denominator at all.
        content = dup_content(elements, K)
        common = K.gcd(content, denom)

        if not K.is_one(content):

            common = K.gcd(content, denom)

            if not K.is_one(common):
                elements = dup_quo_ground(elements, common, K)
                denom = K.quo(denom, common)

        if not K.is_one(u):
            elements = dup_mul_ground(elements, u, K)
            denom = u * denom
        elif K.is_one(common):
            return (M.copy(), denom)

        M_cancelled = M.from_flat_nz(elements, data, K)

        return M_cancelled, denom

    def cancel_denom_elementwise(self, denom):
        """
        Cancel factors between the elements of a matrix and a denominator.

        Returns a matrix of numerators and matrix of denominators.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 3], [4, 12]], ZZ)
        >>> denom = ZZ(6)
        >>> numers, denoms = M.cancel_denom_elementwise(denom)
        >>> numers.to_Matrix()
        Matrix([
        [1, 1],
        [2, 2]])
        >>> denoms.to_Matrix()
        Matrix([
        [3, 2],
        [3, 1]])
        >>> M_frac = (M.to_field() / denom).to_Matrix()
        >>> M_frac
        Matrix([
        [1/3, 1/2],
        [2/3,   2]])
        >>> denoms_inverted = denoms.to_Matrix().applyfunc(lambda e: 1/e)
        >>> numers.to_Matrix().multiply_elementwise(denoms_inverted) == M_frac
        True

        Use :meth:`cancel_denom` to cancel factors between the matrix and the
        denominator while preserving the form of a matrix with a scalar
        denominator.

        See Also
        ========

        cancel_denom
        """
        K = self.domain
        M = self

        if K.is_zero(denom):
            raise ZeroDivisionError('denominator is zero')
        elif K.is_one(denom):
            M_numers = M.copy()
            M_denoms = M.ones(M.shape, M.domain)
            return (M_numers, M_denoms)

        elements, data = M.to_flat_nz()

        cofactors = [K.cofactors(numer, denom) for numer in elements]
        gcds, numers, denoms = zip(*cofactors)

        M_numers = M.from_flat_nz(list(numers), data, K)
        M_denoms = M.from_flat_nz(list(denoms), data, K)

        return (M_numers, M_denoms)

    def content(self):
        """
        Return the gcd of the elements of the matrix.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 4], [4, 12]], ZZ)
        >>> M.content()
        2

        See Also
        ========

        primitive
        cancel_denom
        """
        K = self.domain
        elements, _ = self.to_flat_nz()
        return dup_content(elements, K)

    def primitive(self):
        """
        Factor out gcd of the elements of a matrix.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 4], [4, 12]], ZZ)
        >>> content, M_primitive = M.primitive()
        >>> content
        2
        >>> M_primitive
        DomainMatrix([[1, 2], [2, 6]], (2, 2), ZZ)
        >>> content * M_primitive == M
        True
        >>> M_primitive.content() == ZZ(1)
        True

        See Also
        ========

        content
        cancel_denom
        """
        K = self.domain
        elements, data = self.to_flat_nz()
        content, prims = dup_primitive(elements, K)
        M_primitive = self.from_flat_nz(prims, data, K)
        return content, M_primitive

    def rref(self, *, method='auto'):
        r"""
        Returns reduced-row echelon form (RREF) and list of pivots.

        If the domain is not a field then it will be converted to a field. See
        :meth:`rref_den` for the fraction-free version of this routine that
        returns RREF with denominator instead.

        The domain must either be a field or have an associated fraction field
        (see :meth:`to_field`).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)

        >>> rref_matrix, rref_pivots = A.rref()
        >>> rref_matrix
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> rref_pivots
        (0, 1, 2)

        Parameters
        ==========

        method : str, optional (default: 'auto')
            The method to use to compute the RREF. The default is ``'auto'``,
            which will attempt to choose the fastest method. The other options
            are:

            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with
              division. If the domain is not a field then it will be converted
              to a field with :meth:`to_field` first and RREF will be computed
              by inverting the pivot elements in each row. This is most
              efficient for very sparse matrices or for matrices whose elements
              have complex denominators.

            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan
              elimination. Elimination is performed using exact division
              (``exquo``) to control the growth of the coefficients. In this
              case the current domain is always used for elimination but if
              the domain is not a field then it will be converted to a field
              at the end and divided by the denominator. This is most efficient
              for dense matrices or for matrices with simple denominators.

            - ``A.rref(method='CD')`` clears the denominators before using
              fraction-free Gauss-Jordan elimination in the associated ring.
              This is most efficient for dense matrices with very simple
              denominators.

            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and
              ``A.rref(method='CD_dense')`` are the same as the above methods
              except that the dense implementations of the algorithms are used.
              By default ``A.rref(method='auto')`` will usually choose the
              sparse implementations for RREF.

            Regardless of which algorithm is used the returned matrix will
            always have the same format (sparse or dense) as the input and its
            domain will always be the field of fractions of the input domain.

        Returns
        =======

        (DomainMatrix, list)
            reduced-row echelon form and list of pivots for the DomainMatrix

        See Also
        ========

        rref_den
            RREF with denominator
        sympy.polys.matrices.sdm.sdm_irref
            Sparse implementation of ``method='GJ'``.
        sympy.polys.matrices.sdm.sdm_rref_den
            Sparse implementation of ``method='FF'`` and ``method='CD'``.
        sympy.polys.matrices.dense.ddm_irref
            Dense implementation of ``method='GJ'``.
        sympy.polys.matrices.dense.ddm_irref_den
            Dense implementation of ``method='FF'`` and ``method='CD'``.
        clear_denoms
            Clear denominators from a matrix, used by ``method='CD'`` and
            by ``method='GJ'`` when the original domain is not a field.

        """
        return _dm_rref(self, method=method)

    def rref_den(self, *, method='auto', keep_domain=True):
        r"""
        Returns reduced-row echelon form with denominator and list of pivots.

        Requires exact division in the ground domain (``exquo``).

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [ZZ(2), ZZ(-1), ZZ(0)],
        ...     [ZZ(-1), ZZ(2), ZZ(-1)],
        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)

        >>> A_rref, denom, pivots = A.rref_den()
        >>> A_rref
        DomainMatrix([[6, 0, 0], [0, 6, 0], [0, 0, 6]], (3, 3), ZZ)
        >>> denom
        6
        >>> pivots
        (0, 1, 2)
        >>> A_rref.to_field() / denom
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> A_rref.to_field() / denom == A.convert_to(QQ).rref()[0]
        True

        Parameters
        ==========

        method : str, optional (default: 'auto')
            The method to use to compute the RREF. The default is ``'auto'``,
            which will attempt to choose the fastest method. The other options
            are:

            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan
              elimination. Elimination is performed using exact division
              (``exquo``) to control the growth of the coefficients. In this
              case the current domain is always used for elimination and the
              result is always returned as a matrix over the current domain.
              This is most efficient for dense matrices or for matrices with
              simple denominators.

            - ``A.rref(method='CD')`` clears denominators before using
              fraction-free Gauss-Jordan elimination in the associated ring.
              The result will be converted back to the original domain unless
              ``keep_domain=False`` is passed in which case the result will be
              over the ring used for elimination. This is most efficient for
              dense matrices with very simple denominators.

            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with
              division. If the domain is not a field then it will be converted
              to a field with :meth:`to_field` first and RREF will be computed
              by inverting the pivot elements in each row. The result is
              converted back to the original domain by clearing denominators
              unless ``keep_domain=False`` is passed in which case the result
              will be over the field used for elimination. This is most
              efficient for very sparse matrices or for matrices whose elements
              have complex denominators.

            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and
              ``A.rref(method='CD_dense')`` are the same as the above methods
              except that the dense implementations of the algorithms are used.
              By default ``A.rref(method='auto')`` will usually choose the
              sparse implementations for RREF.

            Regardless of which algorithm is used the returned matrix will
            always have the same format (sparse or dense) as the input and if
            ``keep_domain=True`` its domain will always be the same as the
            input.

        keep_domain : bool, optional
            If True (the default), the domain of the returned matrix and
            denominator are the same as the domain of the input matrix. If
            False, the domain of the returned matrix might be changed to an
            associated ring or field if the algorithm used a different domain.
            This is useful for efficiency if the caller does not need the
            result to be in the original domain e.g. it avoids clearing
            denominators in the case of ``A.rref(method='GJ')``.

        Returns
        =======

        (DomainMatrix, scalar, list)
            Reduced-row echelon form, denominator and list of pivot indices.

        See Also
        ========

        rref
            RREF without denominator for field domains.
        sympy.polys.matrices.sdm.sdm_irref
            Sparse implementation of ``method='GJ'``.
        sympy.polys.matrices.sdm.sdm_rref_den
            Sparse implementation of ``method='FF'`` and ``method='CD'``.
        sympy.polys.matrices.dense.ddm_irref
            Dense implementation of ``method='GJ'``.
        sympy.polys.matrices.dense.ddm_irref_den
            Dense implementation of ``method='FF'`` and ``method='CD'``.
        clear_denoms
            Clear denominators from a matrix, used by ``method='CD'``.

        """
        return _dm_rref_den(self, method=method, keep_domain=keep_domain)

    def columnspace(self):
        r"""
        Returns the columnspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The columns of this matrix form a basis for the columnspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.columnspace()
        DomainMatrix([[1], [2]], (2, 1), QQ)

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        rref, pivots = self.rref()
        rows, cols = self.shape
        return self.extract(range(rows), pivots)

    def rowspace(self):
        r"""
        Returns the rowspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The rows of this matrix form a basis for the rowspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.rowspace()
        DomainMatrix([[1, -1]], (1, 2), QQ)

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        rref, pivots = self.rref()
        rows, cols = self.shape
        return self.extract(range(len(pivots)), range(cols))

    def nullspace(self, divide_last=False):
        r"""
        Returns the nullspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The rows of this matrix form a basis for the nullspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([
        ...    [QQ(2), QQ(-2)],
        ...    [QQ(4), QQ(-4)]], QQ)
        >>> A.nullspace()
        DomainMatrix([[1, 1]], (1, 2), QQ)

        The returned matrix is a basis for the nullspace:

        >>> A_null = A.nullspace().transpose()
        >>> A * A_null
        DomainMatrix([[0], [0]], (2, 1), QQ)
        >>> rows, cols = A.shape
        >>> nullity = rows - A.rank()
        >>> A_null.shape == (cols, nullity)
        True

        Nullspace can also be computed for non-field rings. If the ring is not
        a field then division is not used. Setting ``divide_last`` to True will
        raise an error in this case:

        >>> from sympy import ZZ
        >>> B = DM([[6, -3],
        ...         [4, -2]], ZZ)
        >>> B.nullspace()
        DomainMatrix([[3, 6]], (1, 2), ZZ)
        >>> B.nullspace(divide_last=True)
        Traceback (most recent call last):
        ...
        DMNotAField: Cannot normalize vectors over a non-field

        Over a ring with ``gcd`` defined the nullspace can potentially be
        reduced with :meth:`primitive`:

        >>> B.nullspace().primitive()
        (3, DomainMatrix([[1, 2]], (1, 2), ZZ))

        A matrix over a ring can often be normalized by converting it to a
        field but it is often a bad idea to do so:

        >>> from sympy.abc import a, b, c
        >>> from sympy import Matrix
        >>> M = Matrix([[        a*b,       b + c,        c],
        ...             [      a - b,         b*c,     c**2],
        ...             [a*b + a - b, b*c + b + c, c**2 + c]])
        >>> M.to_DM().domain
        ZZ[a,b,c]
        >>> M.to_DM().nullspace().to_Matrix().transpose()
        Matrix([
        [                             c**3],
        [            -a*b*c**2 + a*c - b*c],
        [a*b**2*c - a*b - a*c + b**2 + b*c]])

        The unnormalized form here is nicer than the normalized form that
        spreads a large denominator throughout the matrix:

        >>> M.to_DM().to_field().nullspace(divide_last=True).to_Matrix().transpose()
        Matrix([
        [                   c**3/(a*b**2*c - a*b - a*c + b**2 + b*c)],
        [(-a*b*c**2 + a*c - b*c)/(a*b**2*c - a*b - a*c + b**2 + b*c)],
        [                                                          1]])

        Parameters
        ==========

        divide_last : bool, optional
            If False (the default), the vectors are not normalized and the RREF
            is computed using :meth:`rref_den` and the denominator is
            discarded. If True, then each row is divided by its final element;
            the domain must be a field in this case.

        See Also
        ========

        nullspace_from_rref
        rref
        rref_den
        rowspace
        """
        A = self
        K = A.domain

        if divide_last and not K.is_Field:
            raise DMNotAField("Cannot normalize vectors over a non-field")

        if divide_last:
            A_rref, pivots = A.rref()
        else:
            A_rref, den, pivots = A.rref_den()

            # Ensure that the sign is canonical before discarding the
            # denominator. Then M.nullspace().primitive() is canonical.
            u = K.canonical_unit(den)
            if u != K.one:
                A_rref *= u

        A_null = A_rref.nullspace_from_rref(pivots)

        return A_null

    def nullspace_from_rref(self, pivots=None):
        """
        Compute nullspace from rref and pivots.

        The domain of the matrix can be any domain.

        The matrix must be in reduced row echelon form already. Otherwise the
        result will be incorrect. Use :meth:`rref` or :meth:`rref_den` first
        to get the reduced row echelon form or use :meth:`nullspace` instead.

        See Also
        ========

        nullspace
        rref
        rref_den
        sympy.polys.matrices.sdm.SDM.nullspace_from_rref
        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
        """
        null_rep, nonpivots = self.rep.nullspace_from_rref(pivots)
        return self.from_rep(null_rep)

    def inv(self):
        r"""
        Finds the inverse of the DomainMatrix if exists

        Returns
        =======

        DomainMatrix
            DomainMatrix after inverse

        Raises
        ======

        ValueError
            If the domain of DomainMatrix not a Field

        DMNonSquareMatrixError
            If the DomainMatrix is not a not Square DomainMatrix

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> A.inv()
        DomainMatrix([[2/3, 1/3, 1/6], [1/3, 2/3, 1/3], [0, 0, 1/2]], (3, 3), QQ)

        See Also
        ========

        neg

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        m, n = self.shape
        if m != n:
            raise DMNonSquareMatrixError
        inv = self.rep.inv()
        return self.from_rep(inv)

    def det(self):
        r"""
        Returns the determinant of a square :class:`DomainMatrix`.

        Returns
        =======

        determinant: DomainElement
            Determinant of the matrix.

        Raises
        ======

        ValueError
            If the domain of DomainMatrix is not a Field

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.det()
        -2

        """
        m, n = self.shape
        if m != n:
            raise DMNonSquareMatrixError
        return self.rep.det()

    def adj_det(self):
        """
        Adjugate and determinant of a square :class:`DomainMatrix`.

        Returns
        =======

        (adjugate, determinant) : (DomainMatrix, DomainScalar)
            The adjugate matrix and determinant of this matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([
        ...     [ZZ(1), ZZ(2)],
        ...     [ZZ(3), ZZ(4)]], ZZ)
        >>> adjA, detA = A.adj_det()
        >>> adjA
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)
        >>> detA
        -2

        See Also
        ========

        adjugate
            Returns only the adjugate matrix.
        det
            Returns only the determinant.
        inv_den
            Returns a matrix/denominator pair representing the inverse matrix
            but perhaps differing from the adjugate and determinant by a common
            factor.
        """
        m, n = self.shape
        I_m = self.eye((m, m), self.domain)
        adjA, detA = self.solve_den_charpoly(I_m, check=False)
        if self.rep.fmt == "dense":
            adjA = adjA.to_dense()
        return adjA, detA

    def adjugate(self):
        """
        Adjugate of a square :class:`DomainMatrix`.

        The adjugate matrix is the transpose of the cofactor matrix and is
        related to the inverse by::

            adj(A) = det(A) * A.inv()

        Unlike the inverse matrix the adjugate matrix can be computed and
        expressed without division or fractions in the ground domain.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> A.adjugate()
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)

        Returns
        =======

        DomainMatrix
            The adjugate matrix of this matrix with the same domain.

        See Also
        ========

        adj_det
        """
        adjA, detA = self.adj_det()
        return adjA

    def inv_den(self, method=None):
        """
        Return the inverse as a :class:`DomainMatrix` with denominator.

        Returns
        =======

        (inv, den) : (:class:`DomainMatrix`, :class:`~.DomainElement`)
            The inverse matrix and its denominator.

        This is more or less equivalent to :meth:`adj_det` except that ``inv``
        and ``den`` are not guaranteed to be the adjugate and inverse. The
        ratio ``inv/den`` is equivalent to ``adj/det`` but some factors
        might be cancelled between ``inv`` and ``den``. In simple cases this
        might just be a minus sign so that ``(inv, den) == (-adj, -det)`` but
        factors more complicated than ``-1`` can also be cancelled.
        Cancellation is not guaranteed to be complete so ``inv`` and ``den``
        may not be on lowest terms. The denominator ``den`` will be zero if and
        only if the determinant is zero.

        If the actual adjugate and determinant are needed, use :meth:`adj_det`
        instead. If the intention is to compute the inverse matrix or solve a
        system of equations then :meth:`inv_den` is more efficient.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [ZZ(2), ZZ(-1), ZZ(0)],
        ...     [ZZ(-1), ZZ(2), ZZ(-1)],
        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)
        >>> Ainv, den = A.inv_den()
        >>> den
        6
        >>> Ainv
        DomainMatrix([[4, 2, 1], [2, 4, 2], [0, 0, 3]], (3, 3), ZZ)
        >>> A * Ainv == den * A.eye(A.shape, A.domain).to_dense()
        True

        Parameters
        ==========

        method : str, optional
            The method to use to compute the inverse. Can be one of ``None``,
            ``'rref'`` or ``'charpoly'``. If ``None`` then the method is
            chosen automatically (see :meth:`solve_den` for details).

        See Also
        ========

        inv
        det
        adj_det
        solve_den
        """
        I = self.eye(self.shape, self.domain)
        return self.solve_den(I, method=method)

    def solve_den(self, b, method=None):
        """
        Solve matrix equation $Ax = b$ without fractions in the ground domain.

        Examples
        ========

        Solve a matrix equation over the integers:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, xden = A.solve_den(b)
        >>> xden
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == xden * b
        True

        Solve a matrix equation over a polynomial ring:

        >>> from sympy import ZZ
        >>> from sympy.abc import x, y, z, a, b
        >>> R = ZZ[x, y, z, a, b]
        >>> M = DM([[x*y, x*z], [y*z, x*z]], R)
        >>> b = DM([[a], [b]], R)
        >>> M.to_Matrix()
        Matrix([
        [x*y, x*z],
        [y*z, x*z]])
        >>> b.to_Matrix()
        Matrix([
        [a],
        [b]])
        >>> xnum, xden = M.solve_den(b)
        >>> xden
        x**2*y*z - x*y*z**2
        >>> xnum.to_Matrix()
        Matrix([
        [ a*x*z - b*x*z],
        [-a*y*z + b*x*y]])
        >>> M * xnum == xden * b
        True

        The solution can be expressed over a fraction field which will cancel
        gcds between the denominator and the elements of the numerator:

        >>> xsol = xnum.to_field() / xden
        >>> xsol.to_Matrix()
        Matrix([
        [           (a - b)/(x*y - y*z)],
        [(-a*z + b*x)/(x**2*z - x*z**2)]])
        >>> (M * xsol).to_Matrix() == b.to_Matrix()
        True

        When solving a large system of equations this cancellation step might
        be a lot slower than :func:`solve_den` itself. The solution can also be
        expressed as a ``Matrix`` without attempting any polynomial
        cancellation between the numerator and denominator giving a less
        simplified result more quickly:

        >>> xsol_uncancelled = xnum.to_Matrix() / xnum.domain.to_sympy(xden)
        >>> xsol_uncancelled
        Matrix([
        [ (a*x*z - b*x*z)/(x**2*y*z - x*y*z**2)],
        [(-a*y*z + b*x*y)/(x**2*y*z - x*y*z**2)]])
        >>> from sympy import cancel
        >>> cancel(xsol_uncancelled) == xsol.to_Matrix()
        True

        Parameters
        ==========

        self : :class:`DomainMatrix`
            The ``m x n`` matrix $A$ in the equation $Ax = b$. Underdetermined
            systems are not supported so ``m >= n``: $A$ should be square or
            have more rows than columns.
        b : :class:`DomainMatrix`
            The ``n x m`` matrix $b$ for the rhs.
        cp : list of :class:`~.DomainElement`, optional
            The characteristic polynomial of the matrix $A$. If not given, it
            will be computed using :meth:`charpoly`.
        method: str, optional
            The method to use for solving the system. Can be one of ``None``,
            ``'charpoly'`` or ``'rref'``. If ``None`` (the default) then the
            method will be chosen automatically.

            The ``charpoly`` method uses :meth:`solve_den_charpoly` and can
            only be used if the matrix is square. This method is division free
            and can be used with any domain.

            The ``rref`` method is fraction free but requires exact division
            in the ground domain (``exquo``). This is also suitable for most
            domains. This method can be used with overdetermined systems (more
            equations than unknowns) but not underdetermined systems as a
            unique solution is sought.

        Returns
        =======

        (xnum, xden) : (DomainMatrix, DomainElement)
            The solution of the equation $Ax = b$ as a pair consisting of an
            ``n x m`` matrix numerator ``xnum`` and a scalar denominator
            ``xden``.

        The solution $x$ is given by ``x = xnum / xden``. The division free
        invariant is ``A * xnum == xden * b``. If $A$ is square then the
        denominator ``xden`` will be a divisor of the determinant $det(A)$.

        Raises
        ======

        DMNonInvertibleMatrixError
            If the system $Ax = b$ does not have a unique solution.

        See Also
        ========

        solve_den_charpoly
        solve_den_rref
        inv_den
        """
        m, n = self.shape
        bm, bn = b.shape

        if m != bm:
            raise DMShapeError("Matrix equation shape mismatch.")

        if method is None:
            method = 'rref'
        elif method == 'charpoly' and m != n:
            raise DMNonSquareMatrixError("method='charpoly' requires a square matrix.")

        if method == 'charpoly':
            xnum, xden = self.solve_den_charpoly(b)
        elif method == 'rref':
            xnum, xden = self.solve_den_rref(b)
        else:
            raise DMBadInputError("method should be 'rref' or 'charpoly'")

        return xnum, xden

    def solve_den_rref(self, b):
        """
        Solve matrix equation $Ax = b$ using fraction-free RREF

        Solves the matrix equation $Ax = b$ for $x$ and returns the solution
        as a numerator/denominator pair.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, xden = A.solve_den_rref(b)
        >>> xden
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == xden * b
        True

        See Also
        ========

        solve_den
        solve_den_charpoly
        """
        A = self
        m, n = A.shape
        bm, bn = b.shape

        if m != bm:
            raise DMShapeError("Matrix equation shape mismatch.")

        if m < n:
            raise DMShapeError("Underdetermined matrix equation.")

        Aaug = A.hstack(b)
        Aaug_rref, denom, pivots = Aaug.rref_den()

        # XXX: We check here if there are pivots after the last column. If
        # there were than it possibly means that rref_den performed some
        # unnecessary elimination. It would be better if rref methods had a
        # parameter indicating how many columns should be used for elimination.
        if len(pivots) != n or pivots and pivots[-1] >= n:
            raise DMNonInvertibleMatrixError("Non-unique solution.")

        xnum = Aaug_rref[:n, n:]
        xden = denom

        return xnum, xden

    def solve_den_charpoly(self, b, cp=None, check=True):
        """
        Solve matrix equation $Ax = b$ using the characteristic polynomial.

        This method solves the square matrix equation $Ax = b$ for $x$ using
        the characteristic polynomial without any division or fractions in the
        ground domain.

        Examples
        ========

        Solve a matrix equation over the integers:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, detA = A.solve_den_charpoly(b)
        >>> detA
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == detA * b
        True

        Parameters
        ==========

        self : DomainMatrix
            The ``n x n`` matrix `A` in the equation `Ax = b`. Must be square
            and invertible.
        b : DomainMatrix
            The ``n x m`` matrix `b` for the rhs.
        cp : list, optional
            The characteristic polynomial of the matrix `A` if known. If not
            given, it will be computed using :meth:`charpoly`.
        check : bool, optional
            If ``True`` (the default) check that the determinant is not zero
            and raise an error if it is. If ``False`` then if the determinant
            is zero the return value will be equal to ``(A.adjugate()*b, 0)``.

        Returns
        =======

        (xnum, detA) : (DomainMatrix, DomainElement)
            The solution of the equation `Ax = b` as a matrix numerator and
            scalar denominator pair. The denominator is equal to the
            determinant of `A` and the numerator is ``adj(A)*b``.

        The solution $x$ is given by ``x = xnum / detA``. The division free
        invariant is ``A * xnum == detA * b``.

        If ``b`` is the identity matrix, then ``xnum`` is the adjugate matrix
        and we have ``A * adj(A) == detA * I``.

        See Also
        ========

        solve_den
            Main frontend for solving matrix equations with denominator.
        solve_den_rref
            Solve matrix equations using fraction-free RREF.
        inv_den
            Invert a matrix using the characteristic polynomial.
        """
        A, b = self.unify(b)
        m, n = self.shape
        mb, nb = b.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if mb != m:
            raise DMShapeError("Matrix and vector must have the same number of rows")

        f, detA = self.adj_poly_det(cp=cp)

        if check and not detA:
            raise DMNonInvertibleMatrixError("Matrix is not invertible")

        # Compute adj(A)*b = det(A)*inv(A)*b using Horner's method without
        # constructing inv(A) explicitly.
        adjA_b = self.eval_poly_mul(f, b)

        return (adjA_b, detA)

    def adj_poly_det(self, cp=None):
        """
        Return the polynomial $p$ such that $p(A) = adj(A)$ and also the
        determinant of $A$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> p, detA = A.adj_poly_det()
        >>> p
        [-1, 5]
        >>> p_A = A.eval_poly(p)
        >>> p_A
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), QQ)
        >>> p[0]*A**1 + p[1]*A**0 == p_A
        True
        >>> p_A == A.adjugate()
        True
        >>> A * A.adjugate() == detA * A.eye(A.shape, A.domain).to_dense()
        True

        See Also
        ========

        adjugate
        eval_poly
        adj_det
        """

        # Cayley-Hamilton says that a matrix satisfies its own minimal
        # polynomial
        #
        #   p[0]*A^n + p[1]*A^(n-1) + ... + p[n]*I = 0
        #
        # with p[0]=1 and p[n]=(-1)^n*det(A) or
        #
        #   det(A)*I = -(-1)^n*(p[0]*A^(n-1) + p[1]*A^(n-2) + ... + p[n-1]*A).
        #
        # Define a new polynomial f with f[i] = -(-1)^n*p[i] for i=0..n-1. Then
        #
        #   det(A)*I = f[0]*A^n + f[1]*A^(n-1) + ... + f[n-1]*A.
        #
        # Multiplying on the right by inv(A) gives
        #
        #   det(A)*inv(A) = f[0]*A^(n-1) + f[1]*A^(n-2) + ... + f[n-1].
        #
        # So adj(A) = det(A)*inv(A) = f(A)

        A = self
        m, n = self.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if cp is None:
            cp = A.charpoly()

        if len(cp) % 2:
            # n is even
            detA = cp[-1]
            f = [-cpi for cpi in cp[:-1]]
        else:
            # n is odd
            detA = -cp[-1]
            f = cp[:-1]

        return f, detA

    def eval_poly(self, p):
        """
        Evaluate polynomial function of a matrix $p(A)$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> p = [QQ(1), QQ(2), QQ(3)]
        >>> p_A = A.eval_poly(p)
        >>> p_A
        DomainMatrix([[12, 14], [21, 33]], (2, 2), QQ)
        >>> p_A == p[0]*A**2 + p[1]*A + p[2]*A**0
        True

        See Also
        ========

        eval_poly_mul
        """
        A = self
        m, n = A.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if not p:
            return self.zeros(self.shape, self.domain)
        elif len(p) == 1:
            return p[0] * self.eye(self.shape, self.domain)

        # Evaluate p(A) using Horner's method:
        # XXX: Use Paterson-Stockmeyer method?
        I = A.eye(A.shape, A.domain)
        p_A = p[0] * I
        for pi in p[1:]:
            p_A = A*p_A + pi*I

        return p_A

    def eval_poly_mul(self, p, B):
        r"""
        Evaluate polynomial matrix product $p(A) \times B$.

        Evaluate the polynomial matrix product $p(A) \times B$ using Horner's
        method without creating the matrix $p(A)$ explicitly. If $B$ is a
        column matrix then this method will only use matrix-vector multiplies
        and no matrix-matrix multiplies are needed.

        If $B$ is square or wide or if $A$ can be represented in a simpler
        domain than $B$ then it might be faster to evaluate $p(A)$ explicitly
        (see :func:`eval_poly`) and then multiply with $B$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> b = DM([[QQ(5)], [QQ(6)]], QQ)
        >>> p = [QQ(1), QQ(2), QQ(3)]
        >>> p_A_b = A.eval_poly_mul(p, b)
        >>> p_A_b
        DomainMatrix([[144], [303]], (2, 1), QQ)
        >>> p_A_b == p[0]*A**2*b + p[1]*A*b + p[2]*b
        True
        >>> A.eval_poly_mul(p, b) == A.eval_poly(p)*b
        True

        See Also
        ========

        eval_poly
        solve_den_charpoly
        """
        A = self
        m, n = A.shape
        mb, nb = B.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if mb != n:
            raise DMShapeError("Matrices are not aligned")

        if A.domain != B.domain:
            raise DMDomainError("Matrices must have the same domain")

        # Given a polynomial p(x) = p[0]*x^n + p[1]*x^(n-1) + ... + p[n-1]
        # and matrices A and B we want to find
        #
        #   p(A)*B = p[0]*A^n*B + p[1]*A^(n-1)*B + ... + p[n-1]*B
        #
        # Factoring out A term by term we get
        #
        #   p(A)*B = A*(...A*(A*(A*(p[0]*B) + p[1]*B) + p[2]*B) + ...) + p[n-1]*B
        #
        # where each pair of brackets represents one iteration of the loop
        # below starting from the innermost p[0]*B. If B is a column matrix
        # then products like A*(...) are matrix-vector multiplies and products
        # like p[i]*B are scalar-vector multiplies so there are no
        # matrix-matrix multiplies.

        if not p:
            return B.zeros(B.shape, B.domain, fmt=B.rep.fmt)

        p_A_B = p[0]*B

        for p_i in p[1:]:
            p_A_B = A*p_A_B + p_i*B

        return p_A_B

    def lu(self):
        r"""
        Returns Lower and Upper decomposition of the DomainMatrix

        Returns
        =======

        (L, U, exchange)
            L, U are Lower and Upper decomposition of the DomainMatrix,
            exchange is the list of indices of rows exchanged in the
            decomposition.

        Raises
        ======

        ValueError
            If the domain of DomainMatrix not a Field

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> L, U, exchange = A.lu()
        >>> L
        DomainMatrix([[1, 0], [2, 1]], (2, 2), QQ)
        >>> U
        DomainMatrix([[1, -1], [0, 0]], (2, 2), QQ)
        >>> exchange
        []

        See Also
        ========

        lu_solve

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        L, U, swaps = self.rep.lu()
        return self.from_rep(L), self.from_rep(U), swaps

    def qr(self):
        r"""
        QR decomposition of the DomainMatrix.

        Explanation
        ===========

        The QR decomposition expresses a matrix as the product of an orthogonal
        matrix (Q) and an upper triangular matrix (R). In this implementation,
        Q is not orthonormal: its columns are orthogonal but not normalized to
        unit vectors. This avoids unnecessary divisions and is particularly
        suited for exact arithmetic domains.

        Note
        ====

        This implementation is valid only for matrices over real domains. For
        matrices over complex domains, a proper QR decomposition would require
        handling conjugation to ensure orthogonality.

        Returns
        =======

        (Q, R)
            Q is the orthogonal matrix, and R is the upper triangular matrix
            resulting from the QR decomposition of the DomainMatrix.

        Raises
        ======

        DMDomainError
            If the domain of the DomainMatrix is not a field (e.g., QQ).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[1, 2], [3, 4], [5, 6]], (3, 2), QQ)
        >>> Q, R = A.qr()
        >>> Q
        DomainMatrix([[1, 26/35], [3, 8/35], [5, -2/7]], (3, 2), QQ)
        >>> R
        DomainMatrix([[1, 44/35], [0, 1]], (2, 2), QQ)
        >>> Q * R == A
        True
        >>> (Q.transpose() * Q).is_diagonal
        True
        >>> R.is_upper
        True

        See Also
        ========

        lu

        """
        ddm_q, ddm_r = self.rep.qr()
        Q = self.from_rep(ddm_q)
        R = self.from_rep(ddm_r)
        return Q, R

    def lu_solve(self, rhs):
        r"""
        Solver for DomainMatrix x in the A*x = B

        Parameters
        ==========

        rhs : DomainMatrix B

        Returns
        =======

        DomainMatrix
            x in A*x = B

        Raises
        ======

        DMShapeError
            If the DomainMatrix A and rhs have different number of rows

        ValueError
            If the domain of DomainMatrix A not a Field

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(2)],
        ...    [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> B = DomainMatrix([
        ...    [QQ(1), QQ(1)],
        ...    [QQ(0), QQ(1)]], (2, 2), QQ)

        >>> A.lu_solve(B)
        DomainMatrix([[-2, -1], [3/2, 1]], (2, 2), QQ)

        See Also
        ========

        lu

        """
        if self.shape[0] != rhs.shape[0]:
            raise DMShapeError("Shape")
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        sol = self.rep.lu_solve(rhs.rep)
        return self.from_rep(sol)

    def fflu(self):
        """
        Fraction-free LU decomposition of DomainMatrix.

        Explanation
        ===========

        This method computes the PLDU decomposition
        using Gauss-Bareiss elimination in a fraction-free manner,
        it ensures that all intermediate results remain in
        the domain of the input matrix. Unlike standard
        LU decomposition, which introduces division, this approach
        avoids fractions, making it particularly suitable
        for exact arithmetic over integers or polynomials.

        This method satisfies the invariant:

        P * A = L * inv(D) * U

        Returns
        =======

        (P, L, D, U)
            - P (Permutation matrix)
            - L (Lower triangular matrix)
            - D (Diagonal matrix)
            - U (Upper triangular matrix)

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> P, L, D, U = A.fflu()
        >>> P
        DomainMatrix([[1, 0], [0, 1]], (2, 2), ZZ)
        >>> L
        DomainMatrix([[1, 0], [3, -2]], (2, 2), ZZ)
        >>> D
        DomainMatrix([[1, 0], [0, -2]], (2, 2), ZZ)
        >>> U
        DomainMatrix([[1, 2], [0, -2]], (2, 2), ZZ)
        >>> L.is_lower and U.is_upper and D.is_diagonal
        True
        >>> L * D.to_field().inv() * U == P * A.to_field()
        True
        >>> I, d = D.inv_den()
        >>> L * I * U == d * P * A
        True

        See Also
        ========

        sympy.polys.matrices.ddm.DDM.fflu

        References
        ==========

        .. [1]  Nakos, G. C., Turner, P. R., & Williams, R. M. (1997). Fraction-free
                algorithms for linear and polynomial equations. ACM SIGSAM Bulletin,
                31(3), 11-19. https://doi.org/10.1145/271130.271133
        .. [2]  Middeke, J.; Jeffrey, D.J.; Koutschan, C. (2020), "Common Factors
                in Fraction-Free Matrix Decompositions", Mathematics in Computer Science,
                15 (4): 589–608, arXiv:2005.12380, doi:10.1007/s11786-020-00495-9
        .. [3]  https://en.wikipedia.org/wiki/Bareiss_algorithm
        """
        from_rep = self.from_rep
        P, L, D, U = self.rep.fflu()
        return from_rep(P), from_rep(L), from_rep(D), from_rep(U)

    def _solve(A, b):
        # XXX: Not sure about this method or its signature. It is just created
        # because it is needed by the holonomic module.
        if A.shape[0] != b.shape[0]:
            raise DMShapeError("Shape")
        if A.domain != b.domain or not A.domain.is_Field:
            raise DMNotAField('Not a field')
        Aaug = A.hstack(b)
        Arref, pivots = Aaug.rref()
        particular = Arref.from_rep(Arref.rep.particular())
        nullspace_rep, nonpivots = Arref[:,:-1].rep.nullspace()
        nullspace = Arref.from_rep(nullspace_rep)
        return particular, nullspace

    def charpoly(self):
        r"""
        Characteristic polynomial of a square matrix.

        Computes the characteristic polynomial in a fully expanded form using
        division free arithmetic. If a factorization of the characteristic
        polynomial is needed then it is more efficient to call
        :meth:`charpoly_factor_list` than calling :meth:`charpoly` and then
        factorizing the result.

        Returns
        =======

        list: list of DomainElement
            coefficients of the characteristic polynomial

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.charpoly()
        [1, -5, -2]

        See Also
        ========

        charpoly_factor_list
            Compute the factorisation of the characteristic polynomial.
        charpoly_factor_blocks
            A partial factorisation of the characteristic polynomial that can
            be computed more efficiently than either the full factorisation or
            the fully expanded polynomial.
        """
        M = self
        K = M.domain

        factors = M.charpoly_factor_blocks()

        cp = [K.one]

        for f, mult in factors:
            for _ in range(mult):
                cp = dup_mul(cp, f, K)

        return cp

    def charpoly_factor_list(self):
        """
        Full factorization of the characteristic polynomial.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], ZZ)

        Compute the factorization of the characteristic polynomial:

        >>> M.charpoly_factor_list()
        [([1, -9], 2), ([1, -7, -4], 1)]

        Use :meth:`charpoly` to get the unfactorized characteristic polynomial:

        >>> M.charpoly()
        [1, -25, 203, -495, -324]

        The same calculations with ``Matrix``:

        >>> M.to_Matrix().charpoly().as_expr()
        lambda**4 - 25*lambda**3 + 203*lambda**2 - 495*lambda - 324
        >>> M.to_Matrix().charpoly().as_expr().factor()
        (lambda - 9)**2*(lambda**2 - 7*lambda - 4)

        Returns
        =======

        list: list of pairs (factor, multiplicity)
            A full factorization of the characteristic polynomial.

        See Also
        ========

        charpoly
            Expanded form of the characteristic polynomial.
        charpoly_factor_blocks
            A partial factorisation of the characteristic polynomial that can
            be computed more efficiently.
        """
        M = self
        K = M.domain

        # It is more efficient to start from the partial factorization provided
        # for free by M.charpoly_factor_blocks than the expanded M.charpoly.
        factors = M.charpoly_factor_blocks()

        factors_irreducible = []

        for factor_i, mult_i in factors:

            _, factors_list = dup_factor_list(factor_i, K)

            for factor_j, mult_j in factors_list:
                factors_irreducible.append((factor_j, mult_i * mult_j))

        return _collect_factors(factors_irreducible)

    def charpoly_factor_blocks(self):
        """
        Partial factorisation of the characteristic polynomial.

        This factorisation arises from a block structure of the matrix (if any)
        and so the factors are not guaranteed to be irreducible. The
        :meth:`charpoly_factor_blocks` method is the most efficient way to get
        a representation of the characteristic polynomial but the result is
        neither fully expanded nor fully factored.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], ZZ)

        This computes a partial factorization using only the block structure of
        the matrix to reveal factors:

        >>> M.charpoly_factor_blocks()
        [([1, -18, 81], 1), ([1, -7, -4], 1)]

        These factors correspond to the two diagonal blocks in the matrix:

        >>> DM([[6, -1], [9, 12]], ZZ).charpoly()
        [1, -18, 81]
        >>> DM([[1, 2], [5, 6]], ZZ).charpoly()
        [1, -7, -4]

        Use :meth:`charpoly_factor_list` to get a complete factorization into
        irreducibles:

        >>> M.charpoly_factor_list()
        [([1, -9], 2), ([1, -7, -4], 1)]

        Use :meth:`charpoly` to get the expanded characteristic polynomial:

        >>> M.charpoly()
        [1, -25, 203, -495, -324]

        Returns
        =======

        list: list of pairs (factor, multiplicity)
            A partial factorization of the characteristic polynomial.

        See Also
        ========

        charpoly
            Compute the fully expanded characteristic polynomial.
        charpoly_factor_list
            Compute a full factorization of the characteristic polynomial.
        """
        M = self

        if not M.is_square:
            raise DMNonSquareMatrixError("not square")

        # scc returns indices that permute the matrix into block triangular
        # form and can extract the diagonal blocks. M.charpoly() is equal to
        # the product of the diagonal block charpolys.
        components = M.scc()

        block_factors = []

        for indices in components:
            block = M.extract(indices, indices)
            block_factors.append((block.charpoly_base(), 1))

        return _collect_factors(block_factors)

    def charpoly_base(self):
        """
        Base case for :meth:`charpoly_factor_blocks` after block decomposition.

        This method is used internally by :meth:`charpoly_factor_blocks` as the
        base case for computing the characteristic polynomial of a block. It is
        more efficient to call :meth:`charpoly_factor_blocks`, :meth:`charpoly`
        or :meth:`charpoly_factor_list` rather than call this method directly.

        This will use either the dense or the sparse implementation depending
        on the sparsity of the matrix and will clear denominators if possible
        before calling :meth:`charpoly_berk` to compute the characteristic
        polynomial using the Berkowitz algorithm.

        See Also
        ========

        charpoly
        charpoly_factor_list
        charpoly_factor_blocks
        charpoly_berk
        """
        M = self
        K = M.domain

        # It seems that the sparse implementation is always faster for random
        # matrices with fewer than 50% non-zero entries. This does not seem to
        # depend on domain, size, bit count etc.
        density = self.nnz() / self.shape[0]**2
        if density < 0.5:
            M = M.to_sparse()
        else:
            M = M.to_dense()

        # Clearing denominators is always more efficient if it can be done.
        # Doing it here after block decomposition is good because each block
        # might have a smaller denominator. However it might be better for
        # charpoly and charpoly_factor_list to restore the denominators only at
        # the very end so that they can call e.g. dup_factor_list before
        # restoring the denominators. The methods would need to be changed to
        # return (poly, denom) pairs to make that work though.
        clear_denoms = K.is_Field and K.has_assoc_Ring

        if clear_denoms:
            clear_denoms = True
            d, M = M.clear_denoms(convert=True)
            d = d.element
            K_f = K
            K_r = M.domain

        # Berkowitz algorithm over K_r.
        cp = M.charpoly_berk()

        if clear_denoms:
            # Restore the denominator in the charpoly over K_f.
            #
            # If M = N/d then p_M(x) = p_N(x*d)/d^n.
            cp = dup_convert(cp, K_r, K_f)
            p = [K_f.one, K_f.zero]
            q = [K_f.one/d]
            cp = dup_transform(cp, p, q, K_f)

        return cp

    def charpoly_berk(self):
        """Compute the characteristic polynomial using the Berkowitz algorithm.

        This method directly calls the underlying implementation of the
        Berkowitz algorithm (:meth:`sympy.polys.matrices.dense.ddm_berk` or
        :meth:`sympy.polys.matrices.sdm.sdm_berk`).

        This is used by :meth:`charpoly` and other methods as the base case for
        for computing the characteristic polynomial. However those methods will
        apply other optimizations such as block decomposition, clearing
        denominators and converting between dense and sparse representations
        before calling this method. It is more efficient to call those methods
        instead of this one but this method is provided for direct access to
        the Berkowitz algorithm.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import QQ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], QQ)
        >>> M.charpoly_berk()
        [1, -25, 203, -495, -324]

        See Also
        ========

        charpoly
        charpoly_base
        charpoly_factor_list
        charpoly_factor_blocks
        sympy.polys.matrices.dense.ddm_berk
        sympy.polys.matrices.sdm.sdm_berk
        """
        return self.rep.charpoly()

    @classmethod
    def eye(cls, shape, domain):
        r"""
        Return identity matrix of size n or shape (m, n).

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.eye(3, QQ)
        DomainMatrix({0: {0: 1}, 1: {1: 1}, 2: {2: 1}}, (3, 3), QQ)

        """
        if isinstance(shape, int):
            shape = (shape, shape)
        return cls.from_rep(SDM.eye(shape, domain))

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        r"""
        Return diagonal matrix with entries from ``diagonal``.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> DomainMatrix.diag([ZZ(5), ZZ(6)], ZZ)
        DomainMatrix({0: {0: 5}, 1: {1: 6}}, (2, 2), ZZ)

        """
        if shape is None:
            N = len(diagonal)
            shape = (N, N)
        return cls.from_rep(SDM.diag(diagonal, domain, shape))

    @classmethod
    def zeros(cls, shape, domain, *, fmt='sparse'):
        """Returns a zero DomainMatrix of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.zeros((2, 3), QQ)
        DomainMatrix({}, (2, 3), QQ)

        """
        return cls.from_rep(SDM.zeros(shape, domain))

    @classmethod
    def ones(cls, shape, domain):
        """Returns a DomainMatrix of 1s, of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.ones((2,3), QQ)
        DomainMatrix([[1, 1, 1], [1, 1, 1]], (2, 3), QQ)

        """
        return cls.from_rep(DDM.ones(shape, domain).to_dfm_or_ddm())

    def __eq__(A, B):
        r"""
        Checks for two DomainMatrix matrices to be equal or not

        Parameters
        ==========

        A, B: DomainMatrix
            to check equality

        Returns
        =======

        Boolean
            True for equal, else False

        Raises
        ======

        NotImplementedError
            If B is not a DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.__eq__(A)
        True
        >>> A.__eq__(B)
        False

        """
        if not isinstance(A, type(B)):
            return NotImplemented
        return A.domain == B.domain and A.rep == B.rep

    def unify_eq(A, B):
        if A.shape != B.shape:
            return False
        if A.domain != B.domain:
            A, B = A.unify(B)
        return A == B

    def lll(A, delta=QQ(3, 4)):
        """
        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.
        See [1]_ and [2]_.

        Parameters
        ==========

        delta : QQ, optional
            The Lovász parameter. Must be in the interval (0.25, 1), with larger
            values producing a more reduced basis. The default is 0.75 for
            historical reasons.

        Returns
        =======

        The reduced basis as a DomainMatrix over ZZ.

        Throws
        ======

        DMValueError: if delta is not in the range (0.25, 1)
        DMShapeError: if the matrix is not of shape (m, n) with m <= n
        DMDomainError: if the matrix domain is not ZZ
        DMRankError: if the matrix contains linearly dependent rows

        Examples
        ========

        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.polys.matrices import DM
        >>> x = DM([[1, 0, 0, 0, -20160],
        ...         [0, 1, 0, 0, 33768],
        ...         [0, 0, 1, 0, 39578],
        ...         [0, 0, 0, 1, 47757]], ZZ)
        >>> y = DM([[10, -3, -2, 8, -4],
        ...         [3, -9, 8, 1, -11],
        ...         [-3, 13, -9, -3, -9],
        ...         [-12, -7, -11, 9, -1]], ZZ)
        >>> assert x.lll(delta=QQ(5, 6)) == y

        Notes
        =====

        The implementation is derived from the Maple code given in Figures 4.3
        and 4.4 of [3]_ (pp.68-69). It uses the efficient method of only calculating
        state updates as they are required.

        See also
        ========

        lll_transform

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm
        .. [2] https://web.archive.org/web/20221029115428/https://web.cs.elte.hu/~lovasz/scans/lll.pdf
        .. [3] Murray R. Bremner, "Lattice Basis Reduction: An Introduction to the LLL Algorithm and Its Applications"

        """
        return DomainMatrix.from_rep(A.rep.lll(delta=delta))

    def lll_transform(A, delta=QQ(3, 4)):
        """
        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm
        and returns the reduced basis and transformation matrix.

        Explanation
        ===========

        Parameters, algorithm and basis are the same as for :meth:`lll` except that
        the return value is a tuple `(B, T)` with `B` the reduced basis and
        `T` a transformation matrix. The original basis `A` is transformed to
        `B` with `T*A == B`. If only `B` is needed then :meth:`lll` should be
        used as it is a little faster.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.polys.matrices import DM
        >>> X = DM([[1, 0, 0, 0, -20160],
        ...         [0, 1, 0, 0, 33768],
        ...         [0, 0, 1, 0, 39578],
        ...         [0, 0, 0, 1, 47757]], ZZ)
        >>> B, T = X.lll_transform(delta=QQ(5, 6))
        >>> T * X == B
        True

        See also
        ========

        lll

        """
        reduced, transform = A.rep.lll_transform(delta=delta)
        return DomainMatrix.from_rep(reduced), DomainMatrix.from_rep(transform)


def _collect_factors(factors_list):
    """
    Collect repeating factors and sort.

    >>> from sympy.polys.matrices.domainmatrix import _collect_factors
    >>> _collect_factors([([1, 2], 2), ([1, 4], 3), ([1, 2], 5)])
    [([1, 4], 3), ([1, 2], 7)]
    """
    factors = Counter()
    for factor, exponent in factors_list:
        factors[tuple(factor)] += exponent

    factors_list = [(list(f), e) for f, e in factors.items()]

    return _sort_factors(factors_list)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\diophantine\diophantine.py ===
from __future__ import annotations

from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, int_valued
from sympy.core.intfunc import igcdex, ilcm, igcd, integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.external.gmpy import jacobi, remove, invert, iroot
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import divisors, factorint, perfect_power
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.modular import symmetric_residue
from sympy.ntheory.residue_ntheory import sqrt_mod, sqrt_mod_iter
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
                                       signed_permutations, ordered_partitions)


# these are imported with 'from sympy.solvers.diophantine import *
__all__ = ['diophantine', 'classify_diop']


class DiophantineSolutionSet(set):
    """
    Container for a set of solutions to a particular diophantine equation.

    The base representation is a set of tuples representing each of the solutions.

    Parameters
    ==========

    symbols : list
        List of free symbols in the original equation.
    parameters: list
        List of parameters to be used in the solution.

    Examples
    ========

    Adding solutions:

        >>> from sympy.solvers.diophantine.diophantine import DiophantineSolutionSet
        >>> from sympy.abc import x, y, t, u
        >>> s1 = DiophantineSolutionSet([x, y], [t, u])
        >>> s1
        set()
        >>> s1.add((2, 3))
        >>> s1.add((-1, u))
        >>> s1
        {(-1, u), (2, 3)}
        >>> s2 = DiophantineSolutionSet([x, y], [t, u])
        >>> s2.add((3, 4))
        >>> s1.update(*s2)
        >>> s1
        {(-1, u), (2, 3), (3, 4)}

    Conversion of solutions into dicts:

        >>> list(s1.dict_iterator())
        [{x: -1, y: u}, {x: 2, y: 3}, {x: 3, y: 4}]

    Substituting values:

        >>> s3 = DiophantineSolutionSet([x, y], [t, u])
        >>> s3.add((t**2, t + u))
        >>> s3
        {(t**2, t + u)}
        >>> s3.subs({t: 2, u: 3})
        {(4, 5)}
        >>> s3.subs(t, -1)
        {(1, u - 1)}
        >>> s3.subs(t, 3)
        {(9, u + 3)}

    Evaluation at specific values. Positional arguments are given in the same order as the parameters:

        >>> s3(-2, 3)
        {(4, 1)}
        >>> s3(5)
        {(25, u + 5)}
        >>> s3(None, 2)
        {(t**2, t + 2)}
    """

    def __init__(self, symbols_seq, parameters):
        super().__init__()

        if not is_sequence(symbols_seq):
            raise ValueError("Symbols must be given as a sequence.")

        if not is_sequence(parameters):
            raise ValueError("Parameters must be given as a sequence.")

        self.symbols = tuple(symbols_seq)
        self.parameters = tuple(parameters)

    def add(self, solution):
        if len(solution) != len(self.symbols):
            raise ValueError("Solution should have a length of %s, not %s" % (len(self.symbols), len(solution)))
        # make solution canonical wrt sign (i.e. no -x unless x is also present as an arg)
        args = set(solution)
        for i in range(len(solution)):
            x = solution[i]
            if not type(x) is int and (-x).is_Symbol and -x not in args:
                solution = [_.subs(-x, x) for _ in solution]
        super().add(Tuple(*solution))

    def update(self, *solutions):
        for solution in solutions:
            self.add(solution)

    def dict_iterator(self):
        for solution in ordered(self):
            yield dict(zip(self.symbols, solution))

    def subs(self, *args, **kwargs):
        result = DiophantineSolutionSet(self.symbols, self.parameters)
        for solution in self:
            result.add(solution.subs(*args, **kwargs))
        return result

    def __call__(self, *args):
        if len(args) > len(self.parameters):
            raise ValueError("Evaluation should have at most %s values, not %s" % (len(self.parameters), len(args)))
        rep = {p: v for p, v in zip(self.parameters, args) if v is not None}
        return self.subs(rep)


class DiophantineEquationType:
    """
    Internal representation of a particular diophantine equation type.

    Parameters
    ==========

    equation :
        The diophantine equation that is being solved.
    free_symbols : list (optional)
        The symbols being solved for.

    Attributes
    ==========

    total_degree :
        The maximum of the degrees of all terms in the equation
    homogeneous :
        Does the equation contain a term of degree 0
    homogeneous_order :
        Does the equation contain any coefficient that is in the symbols being solved for
    dimension :
        The number of symbols being solved for
    """
    name: str

    def __init__(self, equation, free_symbols=None):
        self.equation = _sympify(equation).expand(force=True)

        if free_symbols is not None:
            self.free_symbols = free_symbols
        else:
            self.free_symbols = list(self.equation.free_symbols)
            self.free_symbols.sort(key=default_sort_key)

        if not self.free_symbols:
            raise ValueError('equation should have 1 or more free symbols')

        self.coeff = self.equation.as_coefficients_dict()
        if not all(int_valued(c) for c in self.coeff.values()):
            raise TypeError("Coefficients should be Integers")

        self.total_degree = Poly(self.equation).total_degree()
        self.homogeneous = 1 not in self.coeff
        self.homogeneous_order = not (set(self.coeff) & set(self.free_symbols))
        self.dimension = len(self.free_symbols)
        self._parameters = None

    def matches(self):
        """
        Determine whether the given equation can be matched to the particular equation type.
        """
        return False

    @property
    def n_parameters(self):
        return self.dimension

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = symbols('t_:%i' % (self.n_parameters,), integer=True)
        return self._parameters

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        raise NotImplementedError('No solver has been written for %s.' % self.name)

    def pre_solve(self, parameters=None):
        if not self.matches():
            raise ValueError("This equation does not match the %s equation type." % self.name)

        if parameters is not None:
            if len(parameters) != self.n_parameters:
                raise ValueError("Expected %s parameter(s) but got %s" % (self.n_parameters, len(parameters)))

        self._parameters = parameters


class Univariate(DiophantineEquationType):
    """
    Representation of a univariate diophantine equation.

    A univariate diophantine equation is an equation of the form
    `a_{0} + a_{1}x + a_{2}x^2 + .. + a_{n}x^n = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x` is an integer variable.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import Univariate
    >>> from sympy.abc import x
    >>> Univariate((x - 2)*(x - 3)**2).solve() # solves equation (x - 2)*(x - 3)**2 == 0
    {(2,), (3,)}

    """

    name = 'univariate'

    def matches(self):
        return self.dimension == 1

    def solve(self, parameters=None, limit=None):
        self.pre_solve(parameters)

        result = DiophantineSolutionSet(self.free_symbols, parameters=self.parameters)
        for i in solveset_real(self.equation, self.free_symbols[0]).intersect(S.Integers):
            result.add((i,))
        return result


class Linear(DiophantineEquationType):
    """
    Representation of a linear diophantine equation.

    A linear diophantine equation is an equation of the form `a_{1}x_{1} +
    a_{2}x_{2} + .. + a_{n}x_{n} = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x_{1}, x_{2}, ..x_{n}` are integer variables.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import Linear
    >>> from sympy.abc import x, y, z
    >>> l1 = Linear(2*x - 3*y - 5)
    >>> l1.matches() # is this equation linear
    True
    >>> l1.solve() # solves equation 2*x - 3*y - 5 == 0
    {(3*t_0 - 5, 2*t_0 - 5)}

    Here x = -3*t_0 - 5 and y = -2*t_0 - 5

    >>> Linear(2*x - 3*y - 4*z -3).solve()
    {(t_0, 2*t_0 + 4*t_1 + 3, -t_0 - 3*t_1 - 3)}

    """

    name = 'linear'

    def matches(self):
        return self.total_degree == 1

    def solve(self, parameters=None, limit=None):
        self.pre_solve(parameters)

        coeff = self.coeff
        var = self.free_symbols

        if 1 in coeff:
            # negate coeff[] because input is of the form: ax + by + c ==  0
            #                              but is used as: ax + by     == -c
            c = -coeff[1]
        else:
            c = 0

        result = DiophantineSolutionSet(var, parameters=self.parameters)
        params = result.parameters

        if len(var) == 1:
            q, r = divmod(c, coeff[var[0]])
            if not r:
                result.add((q,))
            return result

        '''
        base_solution_linear() can solve diophantine equations of the form:

        a*x + b*y == c

        We break down multivariate linear diophantine equations into a
        series of bivariate linear diophantine equations which can then
        be solved individually by base_solution_linear().

        Consider the following:

        a_0*x_0 + a_1*x_1 + a_2*x_2 == c

        which can be re-written as:

        a_0*x_0 + g_0*y_0 == c

        where

        g_0 == gcd(a_1, a_2)

        and

        y == (a_1*x_1)/g_0 + (a_2*x_2)/g_0

        This leaves us with two binary linear diophantine equations.
        For the first equation:

        a == a_0
        b == g_0
        c == c

        For the second:

        a == a_1/g_0
        b == a_2/g_0
        c == the solution we find for y_0 in the first equation.

        The arrays A and B are the arrays of integers used for
        'a' and 'b' in each of the n-1 bivariate equations we solve.
        '''

        A = [coeff[v] for v in var]
        B = []
        if len(var) > 2:
            B.append(igcd(A[-2], A[-1]))
            A[-2] = A[-2] // B[0]
            A[-1] = A[-1] // B[0]
            for i in range(len(A) - 3, 0, -1):
                gcd = igcd(B[0], A[i])
                B[0] = B[0] // gcd
                A[i] = A[i] // gcd
                B.insert(0, gcd)
        B.append(A[-1])

        '''
        Consider the trivariate linear equation:

        4*x_0 + 6*x_1 + 3*x_2 == 2

        This can be re-written as:

        4*x_0 + 3*y_0 == 2

        where

        y_0 == 2*x_1 + x_2
        (Note that gcd(3, 6) == 3)

        The complete integral solution to this equation is:

        x_0 ==  2 + 3*t_0
        y_0 == -2 - 4*t_0

        where 't_0' is any integer.

        Now that we have a solution for 'x_0', find 'x_1' and 'x_2':

        2*x_1 + x_2 == -2 - 4*t_0

        We can then solve for '-2' and '-4' independently,
        and combine the results:

        2*x_1a + x_2a == -2
        x_1a == 0 + t_0
        x_2a == -2 - 2*t_0

        2*x_1b + x_2b == -4*t_0
        x_1b == 0*t_0 + t_1
        x_2b == -4*t_0 - 2*t_1

        ==>

        x_1 == t_0 + t_1
        x_2 == -2 - 6*t_0 - 2*t_1

        where 't_0' and 't_1' are any integers.

        Note that:

        4*(2 + 3*t_0) + 6*(t_0 + t_1) + 3*(-2 - 6*t_0 - 2*t_1) == 2

        for any integral values of 't_0', 't_1'; as required.

        This method is generalised for many variables, below.

        '''
        solutions = []
        for Ai, Bi in zip(A, B):
            tot_x, tot_y = [], []

            for arg in Add.make_args(c):
                if arg.is_Integer:
                    # example: 5 -> k = 5
                    k, p = arg, S.One
                    pnew = params[0]
                else:  # arg is a Mul or Symbol
                    # example: 3*t_1 -> k = 3
                    # example: t_0 -> k = 1
                    k, p = arg.as_coeff_Mul()
                    pnew = params[params.index(p) + 1]

                sol = sol_x, sol_y = base_solution_linear(k, Ai, Bi, pnew)

                if p is S.One:
                    if None in sol:
                        return result
                else:
                    # convert a + b*pnew -> a*p + b*pnew
                    if isinstance(sol_x, Add):
                        sol_x = sol_x.args[0]*p + sol_x.args[1]
                    if isinstance(sol_y, Add):
                        sol_y = sol_y.args[0]*p + sol_y.args[1]

                tot_x.append(sol_x)
                tot_y.append(sol_y)

            solutions.append(Add(*tot_x))
            c = Add(*tot_y)

        solutions.append(c)
        result.add(solutions)
        return result


class BinaryQuadratic(DiophantineEquationType):
    """
    Representation of a binary quadratic diophantine equation.

    A binary quadratic diophantine equation is an equation of the
    form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`, where `A, B, C, D, E,
    F` are integer constants and `x` and `y` are integer variables.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import BinaryQuadratic
    >>> b1 = BinaryQuadratic(x**3 + y**2 + 1)
    >>> b1.matches()
    False
    >>> b2 = BinaryQuadratic(x**2 + y**2 + 2*x + 2*y + 2)
    >>> b2.matches()
    True
    >>> b2.solve()
    {(-1, -1)}

    References
    ==========

    .. [1] Methods to solve Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0, [online],
          Available: https://www.alpertron.com.ar/METHODS.HTM
    .. [2] Solving the equation ax^2+ bxy + cy^2 + dx + ey + f= 0, [online],
          Available: https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

    """

    name = 'binary_quadratic'

    def matches(self):
        return self.total_degree == 2 and self.dimension == 2

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        self.pre_solve(parameters)

        var = self.free_symbols
        coeff = self.coeff

        x, y = var

        A = coeff[x**2]
        B = coeff[x*y]
        C = coeff[y**2]
        D = coeff[x]
        E = coeff[y]
        F = coeff[S.One]

        A, B, C, D, E, F = [as_int(i) for i in _remove_gcd(A, B, C, D, E, F)]

        # (1) Simple-Hyperbolic case: A = C = 0, B != 0
        # In this case equation can be converted to (Bx + E)(By + D) = DE - BF
        # We consider two cases; DE - BF = 0 and DE - BF != 0
        # More details, https://www.alpertron.com.ar/METHODS.HTM#SHyperb

        result = DiophantineSolutionSet(var, self.parameters)
        t, u = result.parameters

        discr = B**2 - 4*A*C
        if A == 0 and C == 0 and B != 0:

            if D*E - B*F == 0:
                q, r = divmod(E, B)
                if not r:
                    result.add((-q, t))
                q, r = divmod(D, B)
                if not r:
                    result.add((t, -q))
            else:
                div = divisors(D*E - B*F)
                div = div + [-term for term in div]
                for d in div:
                    x0, r = divmod(d - E, B)
                    if not r:
                        q, r = divmod(D*E - B*F, d)
                        if not r:
                            y0, r = divmod(q - D, B)
                            if not r:
                                result.add((x0, y0))

        # (2) Parabolic case: B**2 - 4*A*C = 0
        # There are two subcases to be considered in this case.
        # sqrt(c)D - sqrt(a)E = 0 and sqrt(c)D - sqrt(a)E != 0
        # More Details, https://www.alpertron.com.ar/METHODS.HTM#Parabol

        elif discr == 0:

            if A == 0:
                s = BinaryQuadratic(self.equation, free_symbols=[y, x]).solve(parameters=[t, u])
                for soln in s:
                    result.add((soln[1], soln[0]))

            else:
                g = sign(A)*igcd(A, C)
                a = A // g
                c = C // g
                e = sign(B / A)

                sqa = isqrt(a)
                sqc = isqrt(c)
                _c = e*sqc*D - sqa*E
                if not _c:
                    z = Symbol("z", real=True)
                    eq = sqa*g*z**2 + D*z + sqa*F
                    roots = solveset_real(eq, z).intersect(S.Integers)
                    for root in roots:
                        ans = diop_solve(sqa*x + e*sqc*y - root)
                        result.add((ans[0], ans[1]))

                elif int_valued(c):
                    solve_x = lambda u: -e*sqc*g*_c*t**2 - (E + 2*e*sqc*g*u)*t \
                                        - (e*sqc*g*u**2 + E*u + e*sqc*F) // _c

                    solve_y = lambda u: sqa*g*_c*t**2 + (D + 2*sqa*g*u)*t \
                                        + (sqa*g*u**2 + D*u + sqa*F) // _c

                    for z0 in range(0, abs(_c)):
                        # Check if the coefficients of y and x obtained are integers or not
                        if (divisible(sqa*g*z0**2 + D*z0 + sqa*F, _c) and
                            divisible(e*sqc*g*z0**2 + E*z0 + e*sqc*F, _c)):
                            result.add((solve_x(z0), solve_y(z0)))

        # (3) Method used when B**2 - 4*A*C is a square, is described in p. 6 of the below paper
        # by John P. Robertson.
        # https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

        elif is_square(discr):
            if A != 0:
                r = sqrt(discr)
                u, v = symbols("u, v", integer=True)
                eq = _mexpand(
                    4*A*r*u*v + 4*A*D*(B*v + r*u + r*v - B*u) +
                    2*A*4*A*E*(u - v) + 4*A*r*4*A*F)

                solution = diop_solve(eq, t)

                for s0, t0 in solution:

                    num = B*t0 + r*s0 + r*t0 - B*s0
                    x_0 = S(num) / (4*A*r)
                    y_0 = S(s0 - t0) / (2*r)
                    if isinstance(s0, Symbol) or isinstance(t0, Symbol):
                        if len(check_param(x_0, y_0, 4*A*r, parameters)) > 0:
                            ans = check_param(x_0, y_0, 4*A*r, parameters)
                            result.update(*ans)
                    elif x_0.is_Integer and y_0.is_Integer:
                        if is_solution_quad(var, coeff, x_0, y_0):
                            result.add((x_0, y_0))

            else:
                s = BinaryQuadratic(self.equation, free_symbols=var[::-1]).solve(parameters=[t, u])  # Interchange x and y
                while s:
                    result.add(s.pop()[::-1])  # and solution <--------+

        # (4) B**2 - 4*A*C > 0 and B**2 - 4*A*C not a square or B**2 - 4*A*C < 0

        else:

            P, Q = _transformation_to_DN(var, coeff)
            D, N = _find_DN(var, coeff)
            solns_pell = diop_DN(D, N)

            if D < 0:
                for x0, y0 in solns_pell:
                    for x in [-x0, x0]:
                        for y in [-y0, y0]:
                            s = P*Matrix([x, y]) + Q
                            try:
                                result.add([as_int(_) for _ in s])
                            except ValueError:
                                pass
            else:
                # In this case equation can be transformed into a Pell equation

                solns_pell = set(solns_pell)
                solns_pell.update((-X, -Y) for X, Y in list(solns_pell))

                a = diop_DN(D, 1)
                T = a[0][0]
                U = a[0][1]

                if all(int_valued(_) for _ in P[:4] + Q[:2]):
                    for r, s in solns_pell:
                        _a = (r + s*sqrt(D))*(T + U*sqrt(D))**t
                        _b = (r - s*sqrt(D))*(T - U*sqrt(D))**t
                        x_n = _mexpand(S(_a + _b) / 2)
                        y_n = _mexpand(S(_a - _b) / (2*sqrt(D)))
                        s = P*Matrix([x_n, y_n]) + Q
                        result.add(s)

                else:
                    L = ilcm(*[_.q for _ in P[:4] + Q[:2]])

                    k = 1

                    T_k = T
                    U_k = U

                    while (T_k - 1) % L != 0 or U_k % L != 0:
                        T_k, U_k = T_k*T + D*U_k*U, T_k*U + U_k*T
                        k += 1

                    for X, Y in solns_pell:

                        for i in range(k):
                            if all(int_valued(_) for _ in P*Matrix([X, Y]) + Q):
                                _a = (X + sqrt(D)*Y)*(T_k + sqrt(D)*U_k)**t
                                _b = (X - sqrt(D)*Y)*(T_k - sqrt(D)*U_k)**t
                                Xt = S(_a + _b) / 2
                                Yt = S(_a - _b) / (2*sqrt(D))
                                s = P*Matrix([Xt, Yt]) + Q
                                result.add(s)

                            X, Y = X*T + D*U*Y, X*U + Y*T

        return result


class InhomogeneousTernaryQuadratic(DiophantineEquationType):
    """

    Representation of an inhomogeneous ternary quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'inhomogeneous_ternary_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        return not self.homogeneous_order


class HomogeneousTernaryQuadraticNormal(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic normal diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadraticNormal
    >>> HomogeneousTernaryQuadraticNormal(4*x**2 - 5*y**2 + z**2).solve()
    {(1, 2, 4)}

    """

    name = 'homogeneous_ternary_quadratic_normal'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        if not self.homogeneous_order:
            return False

        nonzero = [k for k in self.coeff if self.coeff[k]]
        return len(nonzero) == 3 and all(i**2 in nonzero for i in self.free_symbols)

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        self.pre_solve(parameters)

        var = self.free_symbols
        coeff = self.coeff

        x, y, z = var

        a = coeff[x**2]
        b = coeff[y**2]
        c = coeff[z**2]

        (sqf_of_a, sqf_of_b, sqf_of_c), (a_1, b_1, c_1), (a_2, b_2, c_2) = \
            sqf_normal(a, b, c, steps=True)

        A = -a_2*c_2
        B = -b_2*c_2

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        # If following two conditions are satisfied then there are no solutions
        if A < 0 and B < 0:
            return result

        if (
            sqrt_mod(-b_2*c_2, a_2) is None or
            sqrt_mod(-c_2*a_2, b_2) is None or
            sqrt_mod(-a_2*b_2, c_2) is None):
            return result

        z_0, x_0, y_0 = descent(A, B)

        z_0, q = _rational_pq(z_0, abs(c_2))
        x_0 *= q
        y_0 *= q

        x_0, y_0, z_0 = _remove_gcd(x_0, y_0, z_0)

        # Holzer reduction
        if sign(a) == sign(b):
            x_0, y_0, z_0 = holzer(x_0, y_0, z_0, abs(a_2), abs(b_2), abs(c_2))
        elif sign(a) == sign(c):
            x_0, z_0, y_0 = holzer(x_0, z_0, y_0, abs(a_2), abs(c_2), abs(b_2))
        else:
            y_0, z_0, x_0 = holzer(y_0, z_0, x_0, abs(b_2), abs(c_2), abs(a_2))

        x_0 = reconstruct(b_1, c_1, x_0)
        y_0 = reconstruct(a_1, c_1, y_0)
        z_0 = reconstruct(a_1, b_1, z_0)

        sq_lcm = ilcm(sqf_of_a, sqf_of_b, sqf_of_c)

        x_0 = abs(x_0*sq_lcm // sqf_of_a)
        y_0 = abs(y_0*sq_lcm // sqf_of_b)
        z_0 = abs(z_0*sq_lcm // sqf_of_c)

        result.add(_remove_gcd(x_0, y_0, z_0))
        return result


class HomogeneousTernaryQuadratic(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadratic
    >>> HomogeneousTernaryQuadratic(x**2 + y**2 - 3*z**2 + x*y).solve()
    {(-1, 2, 1)}
    >>> HomogeneousTernaryQuadratic(3*x**2 + y**2 - 3*z**2 + 5*x*y + y*z).solve()
    {(3, 12, 13)}

    """

    name = 'homogeneous_ternary_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        if not self.homogeneous_order:
            return False

        nonzero = [k for k in self.coeff if self.coeff[k]]
        return not (len(nonzero) == 3 and all(i**2 in nonzero for i in self.free_symbols))

    def solve(self, parameters=None, limit=None):
        self.pre_solve(parameters)

        _var = self.free_symbols
        coeff = self.coeff

        x, y, z = _var
        var = [x, y, z]

        # Equations of the form B*x*y + C*z*x + E*y*z = 0 and At least two of the
        # coefficients A, B, C are non-zero.
        # There are infinitely many solutions for the equation.
        # Ex: (0, 0, t), (0, t, 0), (t, 0, 0)
        # Equation can be re-written as y*(B*x + E*z) = -C*x*z and we can find rather
        # unobvious solutions. Set y = -C and B*x + E*z = x*z. The latter can be solved by
        # using methods for binary quadratic diophantine equations. Let's select the
        # solution which minimizes |x| + |z|

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        def unpack_sol(sol):
            if len(sol) > 0:
                return list(sol)[0]
            return None, None, None

        if not any(coeff[i**2] for i in var):
            if coeff[x*z]:
                sols = diophantine(coeff[x*y]*x + coeff[y*z]*z - x*z)
                s = min(sols, key=lambda r: abs(r[0]) + abs(r[1]))
                result.add(_remove_gcd(s[0], -coeff[x*z], s[1]))
                return result

            var[0], var[1] = _var[1], _var[0]
            y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))
            if x_0 is not None:
                result.add((x_0, y_0, z_0))
            return result

        if coeff[x**2] == 0:
            # If the coefficient of x is zero change the variables
            if coeff[y**2] == 0:
                var[0], var[2] = _var[2], _var[0]
                z_0, y_0, x_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))

            else:
                var[0], var[1] = _var[1], _var[0]
                y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))

        else:
            if coeff[x*y] or coeff[x*z]:
                # Apply the transformation x --> X - (B*y + C*z)/(2*A)
                A = coeff[x**2]
                B = coeff[x*y]
                C = coeff[x*z]
                D = coeff[y**2]
                E = coeff[y*z]
                F = coeff[z**2]

                _coeff = {}

                _coeff[x**2] = 4*A**2
                _coeff[y**2] = 4*A*D - B**2
                _coeff[z**2] = 4*A*F - C**2
                _coeff[y*z] = 4*A*E - 2*B*C
                _coeff[x*y] = 0
                _coeff[x*z] = 0

                x_0, y_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, _coeff))

                if x_0 is None:
                    return result

                p, q = _rational_pq(B*y_0 + C*z_0, 2*A)
                x_0, y_0, z_0 = x_0*q - p, y_0*q, z_0*q

            elif coeff[z*y] != 0:
                if coeff[y**2] == 0:
                    if coeff[z**2] == 0:
                        # Equations of the form A*x**2 + E*yz = 0.
                        A = coeff[x**2]
                        E = coeff[y*z]

                        b, a = _rational_pq(-E, A)

                        x_0, y_0, z_0 = b, a, b

                    else:
                        # Ax**2 + E*y*z + F*z**2  = 0
                        var[0], var[2] = _var[2], _var[0]
                        z_0, y_0, x_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))

                else:
                    # A*x**2 + D*y**2 + E*y*z + F*z**2 = 0, C may be zero
                    var[0], var[1] = _var[1], _var[0]
                    y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))

            else:
                # Ax**2 + D*y**2 + F*z**2 = 0, C may be zero
                x_0, y_0, z_0 = unpack_sol(_diop_ternary_quadratic_normal(var, coeff))

        if x_0 is None:
            return result

        result.add(_remove_gcd(x_0, y_0, z_0))
        return result


class InhomogeneousGeneralQuadratic(DiophantineEquationType):
    """

    Representation of an inhomogeneous general quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'inhomogeneous_general_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return True
        # there may be Pow keys like x**2 or Mul keys like x*y
        return any(k.is_Mul for k in self.coeff) and not self.homogeneous


class HomogeneousGeneralQuadratic(DiophantineEquationType):
    """

    Representation of a homogeneous general quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'homogeneous_general_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        # there may be Pow keys like x**2 or Mul keys like x*y
        return any(k.is_Mul for k in self.coeff) and self.homogeneous


class GeneralSumOfSquares(DiophantineEquationType):
    r"""
    Representation of the diophantine equation

    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Details
    =======

    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \in Z` then there will be
    no solutions. Refer [1]_ for more details.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfSquares
    >>> from sympy.abc import a, b, c, d, e
    >>> GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve()
    {(15, 22, 22, 24, 24)}

    By default only 1 solution is returned. Use the `limit` keyword for more:

    >>> sorted(GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve(limit=3))
    [(15, 22, 22, 24, 24), (16, 19, 24, 24, 24), (16, 20, 22, 23, 26)]

    References
    ==========

    .. [1] Representing an integer as a sum of three squares, [online],
        Available:
        https://proofwiki.org/wiki/Integer_as_Sum_of_Three_Squares
    """

    name = 'general_sum_of_squares'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        if any(k.is_Mul for k in self.coeff):
            return False
        return all(self.coeff[k] == 1 for k in self.coeff if k != 1)

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)

        var = self.free_symbols
        k = -int(self.coeff[1])
        n = self.dimension

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        if k < 0 or limit < 1:
            return result

        signs = [-1 if x.is_nonpositive else 1 for x in var]
        negs = signs.count(-1) != 0

        took = 0
        for t in sum_of_squares(k, n, zeros=True):
            if negs:
                result.add([signs[i]*j for i, j in enumerate(t)])
            else:
                result.add(t)
            took += 1
            if took == limit:
                break
        return result


class GeneralPythagorean(DiophantineEquationType):
    """
    Representation of the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralPythagorean
    >>> from sympy.abc import a, b, c, d, e, x, y, z, t
    >>> GeneralPythagorean(a**2 + b**2 + c**2 - d**2).solve()
    {(t_0**2 + t_1**2 - t_2**2, 2*t_0*t_2, 2*t_1*t_2, t_0**2 + t_1**2 + t_2**2)}
    >>> GeneralPythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2).solve(parameters=[x, y, z, t])
    {(-10*t**2 + 10*x**2 + 10*y**2 + 10*z**2, 15*t**2 + 15*x**2 + 15*y**2 + 15*z**2, 15*t*x, 12*t*y, 60*t*z)}
    """

    name = 'general_pythagorean'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        if any(k.is_Mul for k in self.coeff):
            return False
        if all(self.coeff[k] == 1 for k in self.coeff if k != 1):
            return False
        if not all(is_square(abs(self.coeff[k])) for k in self.coeff):
            return False
        # all but one has the same sign
        # e.g. 4*x**2 + y**2 - 4*z**2
        return abs(sum(sign(self.coeff[k]) for k in self.coeff)) == self.dimension - 2

    @property
    def n_parameters(self):
        return self.dimension - 1

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)

        coeff = self.coeff
        var = self.free_symbols
        n = self.dimension

        if sign(coeff[var[0] ** 2]) + sign(coeff[var[1] ** 2]) + sign(coeff[var[2] ** 2]) < 0:
            for key in coeff.keys():
                coeff[key] = -coeff[key]

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        index = 0

        for i, v in enumerate(var):
            if sign(coeff[v ** 2]) == -1:
                index = i

        m = result.parameters

        ith = sum(m_i ** 2 for m_i in m)
        L = [ith - 2 * m[n - 2] ** 2]
        L.extend([2 * m[i] * m[n - 2] for i in range(n - 2)])
        sol = L[:index] + [ith] + L[index:]

        lcm = 1
        for i, v in enumerate(var):
            if i == index or (index > 0 and i == 0) or (index == 0 and i == 1):
                lcm = ilcm(lcm, sqrt(abs(coeff[v ** 2])))
            else:
                s = sqrt(coeff[v ** 2])
                lcm = ilcm(lcm, s if _odd(s) else s // 2)

        for i, v in enumerate(var):
            sol[i] = (lcm * sol[i]) / sqrt(abs(coeff[v ** 2]))

        result.add(sol)
        return result


class CubicThue(DiophantineEquationType):
    """
    Representation of a cubic Thue diophantine equation.

    A cubic Thue diophantine equation is a polynomial of the form
    `f(x, y) = r` of degree 3, where `x` and `y` are integers
    and `r` is a rational number.

    No solver is currently implemented for this equation type.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import CubicThue
    >>> c1 = CubicThue(x**3 + y**2 + 1)
    >>> c1.matches()
    True

    """

    name = 'cubic_thue'

    def matches(self):
        return self.total_degree == 3 and self.dimension == 2


class GeneralSumOfEvenPowers(DiophantineEquationType):
    """
    Representation of the diophantine equation

    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`

    where `e` is an even, integer power.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfEvenPowers
    >>> from sympy.abc import a, b
    >>> GeneralSumOfEvenPowers(a**4 + b**4 - (2**4 + 3**4)).solve()
    {(2, 3)}

    """

    name = 'general_sum_of_even_powers'

    def matches(self):
        if not self.total_degree > 3:
            return False
        if self.total_degree % 2 != 0:
            return False
        if not all(k.is_Pow and k.exp == self.total_degree for k in self.coeff if k != 1):
            return False
        return all(self.coeff[k] == 1 for k in self.coeff if k != 1)

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)

        var = self.free_symbols
        coeff = self.coeff

        p = None
        for q in coeff.keys():
            if q.is_Pow and coeff[q]:
                p = q.exp

        k = len(var)
        n = -coeff[1]

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        if n < 0 or limit < 1:
            return result

        sign = [-1 if x.is_nonpositive else 1 for x in var]
        negs = sign.count(-1) != 0

        took = 0
        for t in power_representation(n, p, k):
            if negs:
                result.add([sign[i]*j for i, j in enumerate(t)])
            else:
                result.add(t)
            took += 1
            if took == limit:
                break
        return result

# these types are known (but not necessarily handled)
# note that order is important here (in the current solver state)
all_diop_classes = [
    Linear,
    Univariate,
    BinaryQuadratic,
    InhomogeneousTernaryQuadratic,
    HomogeneousTernaryQuadraticNormal,
    HomogeneousTernaryQuadratic,
    InhomogeneousGeneralQuadratic,
    HomogeneousGeneralQuadratic,
    GeneralSumOfSquares,
    GeneralPythagorean,
    CubicThue,
    GeneralSumOfEvenPowers,
]

diop_known = {diop_class.name for diop_class in all_diop_classes}


def _remove_gcd(*x):
    try:
        g = igcd(*x)
    except ValueError:
        fx = list(filter(None, x))
        if len(fx) < 2:
            return x
        g = igcd(*[i.as_content_primitive()[0] for i in fx])
    except TypeError:
        raise TypeError('_remove_gcd(a,b,c) or _remove_gcd(*container)')
    if g == 1:
        return x
    return tuple([i//g for i in x])


def _rational_pq(a, b):
    # return `(numer, denom)` for a/b; sign in numer and gcd removed
    return _remove_gcd(sign(b)*a, abs(b))


def _nint_or_floor(p, q):
    # return nearest int to p/q; in case of tie return floor(p/q)
    w, r = divmod(p, q)
    if abs(r) <= abs(q)//2:
        return w
    return w + 1


def _odd(i):
    return i % 2 != 0


def _even(i):
    return i % 2 == 0


def diophantine(eq, param=symbols("t", integer=True), syms=None,
                permute=False):
    """
    Simplify the solution procedure of diophantine equation ``eq`` by
    converting it into a product of terms which should equal zero.

    Explanation
    ===========

    For example, when solving, `x^2 - y^2 = 0` this is treated as
    `(x + y)(x - y) = 0` and `x + y = 0` and `x - y = 0` are solved
    independently and combined. Each term is solved by calling
    ``diop_solve()``. (Although it is possible to call ``diop_solve()``
    directly, one must be careful to pass an equation in the correct
    form and to interpret the output correctly; ``diophantine()`` is
    the public-facing function to use in general.)

    Output of ``diophantine()`` is a set of tuples. The elements of the
    tuple are the solutions for each variable in the equation and
    are arranged according to the alphabetic ordering of the variables.
    e.g. For an equation with two variables, `a` and `b`, the first
    element of the tuple is the solution for `a` and the second for `b`.

    Usage
    =====

    ``diophantine(eq, t, syms)``: Solve the diophantine
    equation ``eq``.
    ``t`` is the optional parameter to be used by ``diop_solve()``.
    ``syms`` is an optional list of symbols which determines the
    order of the elements in the returned tuple.

    By default, only the base solution is returned. If ``permute`` is set to
    True then permutations of the base solution and/or permutations of the
    signs of the values will be returned when applicable.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``t`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy import diophantine
    >>> from sympy.abc import a, b
    >>> eq = a**4 + b**4 - (2**4 + 3**4)
    >>> diophantine(eq)
    {(2, 3)}
    >>> diophantine(eq, permute=True)
    {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

    >>> from sympy.abc import x, y, z
    >>> diophantine(x**2 - y**2)
    {(t_0, -t_0), (t_0, t_0)}

    >>> diophantine(x*(2*x + 3*y - z))
    {(0, n1, n2), (t_0, t_1, 2*t_0 + 3*t_1)}
    >>> diophantine(x**2 + 3*x*y + 4*x)
    {(0, n1), (-3*t_0 - 4, t_0)}

    See Also
    ========

    diop_solve
    sympy.utilities.iterables.permute_signs
    sympy.utilities.iterables.signed_permutations
    """

    eq = _sympify(eq)

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs

    try:
        var = list(eq.expand(force=True).free_symbols)
        var.sort(key=default_sort_key)
        if syms:
            if not is_sequence(syms):
                raise TypeError(
                    'syms should be given as a sequence, e.g. a list')
            syms = [i for i in syms if i in var]
            if syms != var:
                dict_sym_index = dict(zip(syms, range(len(syms))))
                return {tuple([t[dict_sym_index[i]] for i in var])
                            for t in diophantine(eq, param, permute=permute)}
        n, d = eq.as_numer_denom()
        if n.is_number:
            return set()
        if not d.is_number:
            dsol = diophantine(d)
            good = diophantine(n) - dsol
            return {s for s in good if _mexpand(d.subs(zip(var, s)))}
        eq = factor_terms(n)
        assert not eq.is_number
        eq = eq.as_independent(*var, as_Add=False)[1]
        p = Poly(eq)
        assert not any(g.is_number for g in p.gens)
        eq = p.as_expr()
        assert eq.is_polynomial()
    except (GeneratorsNeeded, AssertionError):
        raise TypeError(filldedent('''
    Equation should be a polynomial with Rational coefficients.'''))

    # permute only sign
    do_permute_signs = False
    # permute sign and values
    do_permute_signs_var = False
    # permute few signs
    permute_few_signs = False
    try:
        # if we know that factoring should not be attempted, skip
        # the factoring step
        v, c, t = classify_diop(eq)

        # check for permute sign
        if permute:
            len_var = len(v)
            permute_signs_for = [
                GeneralSumOfSquares.name,
                GeneralSumOfEvenPowers.name]
            permute_signs_check = [
                HomogeneousTernaryQuadratic.name,
                HomogeneousTernaryQuadraticNormal.name,
                BinaryQuadratic.name]
            if t in permute_signs_for:
                do_permute_signs_var = True
            elif t in permute_signs_check:
                # if all the variables in eq have even powers
                # then do_permute_sign = True
                if len_var == 3:
                    var_mul = list(subsets(v, 2))
                    # here var_mul is like [(x, y), (x, z), (y, z)]
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = (a[0]*a[1] for a in var_mul)
                    # if coeff(y*z), coeff(y*x), coeff(x*z) is not 0 then
                    # `xy_coeff` => True and do_permute_sign => False.
                    # Means no permuted solution.
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    # here var_mul is like [(x,), (y, )]
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any((xy_coeff, x_coeff)):
                        # means only x**2, y**2, z**2, const is present
                        do_permute_signs = True
                    elif not x_coeff:
                        permute_few_signs = True
                elif len_var == 2:
                    var_mul = list(subsets(v, 2))
                    # here var_mul is like [(x, y)]
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = (x[0]*x[1] for x in var_mul)
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    # here var_mul is like [(x,), (y, )]
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any((xy_coeff, x_coeff)):
                        # means only x**2, y**2 and const is present
                        # so we can get more soln by permuting this soln.
                        do_permute_signs = True
                    elif not x_coeff:
                        # when coeff(x), coeff(y) is not present then signs of
                        #  x, y can be permuted such that their sign are same
                        # as sign of x*y.
                        # e.g 1. (x_val,y_val)=> (x_val,y_val), (-x_val,-y_val)
                        # 2. (-x_vall, y_val)=> (-x_val,y_val), (x_val,-y_val)
                        permute_few_signs = True
        if t == 'general_sum_of_squares':
            # trying to factor such expressions will sometimes hang
            terms = [(eq, 1)]
        else:
            raise TypeError
    except (TypeError, NotImplementedError):
        fl = factor_list(eq)
        if fl[0].is_Rational and fl[0] != 1:
            return diophantine(eq/fl[0], param=param, syms=syms, permute=permute)
        terms = fl[1]

    sols = set()

    for term in terms:

        base, _ = term
        var_t, _, eq_type = classify_diop(base, _dict=False)
        _, base = signsimp(base, evaluate=False).as_coeff_Mul()
        solution = diop_solve(base, param)

        if eq_type in [
                Linear.name,
                HomogeneousTernaryQuadratic.name,
                HomogeneousTernaryQuadraticNormal.name,
                GeneralPythagorean.name]:
            sols.add(merge_solution(var, var_t, solution))

        elif eq_type in [
                BinaryQuadratic.name,
                GeneralSumOfSquares.name,
                GeneralSumOfEvenPowers.name,
                Univariate.name]:
            sols.update(merge_solution(var, var_t, sol) for sol in solution)

        else:
            raise NotImplementedError('unhandled type: %s' % eq_type)

    sols.discard(())
    null = tuple([0]*len(var))
    # if there is no solution, return trivial solution
    if not sols and eq.subs(zip(var, null)).is_zero:
        if all(check_assumptions(val, **s.assumptions0) is not False for val, s in zip(null, var)):
            sols.add(null)

    final_soln = set()
    for sol in sols:
        if all(int_valued(s) for s in sol):
            if do_permute_signs:
                permuted_sign = set(permute_signs(sol))
                final_soln.update(permuted_sign)
            elif permute_few_signs:
                lst = list(permute_signs(sol))
                lst = list(filter(lambda x: x[0]*x[1] == sol[1]*sol[0], lst))
                permuted_sign = set(lst)
                final_soln.update(permuted_sign)
            elif do_permute_signs_var:
                permuted_sign_var = set(signed_permutations(sol))
                final_soln.update(permuted_sign_var)
            else:
                final_soln.add(sol)
        else:
                final_soln.add(sol)
    return final_soln


def merge_solution(var, var_t, solution):
    """
    This is used to construct the full solution from the solutions of sub
    equations.

    Explanation
    ===========

    For example when solving the equation `(x - y)(x^2 + y^2 - z^2) = 0`,
    solutions for each of the equations `x - y = 0` and `x^2 + y^2 - z^2` are
    found independently. Solutions for `x - y = 0` are `(x, y) = (t, t)`. But
    we should introduce a value for z when we output the solution for the
    original equation. This function converts `(t, t)` into `(t, t, n_{1})`
    where `n_{1}` is an integer parameter.
    """
    sol = []

    if None in solution:
        return ()

    solution = iter(solution)
    params = numbered_symbols("n", integer=True, start=1)
    for v in var:
        if v in var_t:
            sol.append(next(solution))
        else:
            sol.append(next(params))

    for val, symb in zip(sol, var):
        if check_assumptions(val, **symb.assumptions0) is False:
            return ()

    return tuple(sol)


def _diop_solve(eq, params=None):
    for diop_type in all_diop_classes:
        if diop_type(eq).matches():
            return diop_type(eq).solve(parameters=params)


def diop_solve(eq, param=symbols("t", integer=True)):
    """
    Solves the diophantine equation ``eq``.

    Explanation
    ===========

    Unlike ``diophantine()``, factoring of ``eq`` is not attempted. Uses
    ``classify_diop()`` to determine the type of the equation and calls
    the appropriate solver function.

    Use of ``diophantine()`` is recommended over other helper functions.
    ``diop_solve()`` can return either a set or a tuple depending on the
    nature of the equation. All non-trivial solutions are returned: assumptions
    on symbols are ignored.

    Usage
    =====

    ``diop_solve(eq, t)``: Solve diophantine equation, ``eq`` using ``t``
    as a parameter if needed.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``t`` is a parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine import diop_solve
    >>> from sympy.abc import x, y, z, w
    >>> diop_solve(2*x + 3*y - 5)
    (3*t_0 - 5, 5 - 2*t_0)
    >>> diop_solve(4*x + 3*y - 4*z + 5)
    (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)
    >>> diop_solve(x + 3*y - 4*z + w - 6)
    (t_0, t_0 + t_1, 6*t_0 + 5*t_1 + 4*t_2 - 6, 5*t_0 + 4*t_1 + 3*t_2 - 6)
    >>> diop_solve(x**2 + y**2 - 5)
    {(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)}


    See Also
    ========

    diophantine()
    """
    var, coeff, eq_type = classify_diop(eq, _dict=False)

    if eq_type == Linear.name:
        return diop_linear(eq, param)

    elif eq_type == BinaryQuadratic.name:
        return diop_quadratic(eq, param)

    elif eq_type == HomogeneousTernaryQuadratic.name:
        return diop_ternary_quadratic(eq, parameterize=True)

    elif eq_type == HomogeneousTernaryQuadraticNormal.name:
        return diop_ternary_quadratic_normal(eq, parameterize=True)

    elif eq_type == GeneralPythagorean.name:
        return diop_general_pythagorean(eq, param)

    elif eq_type == Univariate.name:
        return diop_univariate(eq)

    elif eq_type == GeneralSumOfSquares.name:
        return diop_general_sum_of_squares(eq, limit=S.Infinity)

    elif eq_type == GeneralSumOfEvenPowers.name:
        return diop_general_sum_of_even_powers(eq, limit=S.Infinity)

    if eq_type is not None and eq_type not in diop_known:
            raise ValueError(filldedent('''
    Although this type of equation was identified, it is not yet
    handled. It should, however, be listed in `diop_known` at the
    top of this file. Developers should see comments at the end of
    `classify_diop`.
            '''))  # pragma: no cover
    else:
        raise NotImplementedError(
            'No solver has been written for %s.' % eq_type)


def classify_diop(eq, _dict=True):
    # docstring supplied externally

    matched = False
    diop_type = None
    for diop_class in all_diop_classes:
        diop_type = diop_class(eq)
        if diop_type.matches():
            matched = True
            break

    if matched:
        return diop_type.free_symbols, dict(diop_type.coeff) if _dict else diop_type.coeff, diop_type.name

    # new diop type instructions
    # --------------------------
    # if this error raises and the equation *can* be classified,
    #  * it should be identified in the if-block above
    #  * the type should be added to the diop_known
    # if a solver can be written for it,
    #  * a dedicated handler should be written (e.g. diop_linear)
    #  * it should be passed to that handler in diop_solve
    raise NotImplementedError(filldedent('''
        This equation is not yet recognized or else has not been
        simplified sufficiently to put it in a form recognized by
        diop_classify().'''))


classify_diop.func_doc = (  # type: ignore
    '''
    Helper routine used by diop_solve() to find information about ``eq``.

    Explanation
    ===========

    Returns a tuple containing the type of the diophantine equation
    along with the variables (free symbols) and their coefficients.
    Variables are returned as a list and coefficients are returned
    as a dict with the key being the respective term and the constant
    term is keyed to 1. The type is one of the following:

    * %s

    Usage
    =====

    ``classify_diop(eq)``: Return variables, coefficients and type of the
    ``eq``.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``_dict`` is for internal use: when True (default) a dict is returned,
    otherwise a defaultdict which supplies 0 for missing keys is returned.

    Examples
    ========

    >>> from sympy.solvers.diophantine import classify_diop
    >>> from sympy.abc import x, y, z, w, t
    >>> classify_diop(4*x + 6*y - 4)
    ([x, y], {1: -4, x: 4, y: 6}, 'linear')
    >>> classify_diop(x + 3*y -4*z + 5)
    ([x, y, z], {1: 5, x: 1, y: 3, z: -4}, 'linear')
    >>> classify_diop(x**2 + y**2 - x*y + x + 5)
    ([x, y], {1: 5, x: 1, x**2: 1, y**2: 1, x*y: -1}, 'binary_quadratic')
    ''' % ('\n    * '.join(sorted(diop_known))))


def diop_linear(eq, param=symbols("t", integer=True)):
    """
    Solves linear diophantine equations.

    A linear diophantine equation is an equation of the form `a_{1}x_{1} +
    a_{2}x_{2} + .. + a_{n}x_{n} = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x_{1}, x_{2}, ..x_{n}` are integer variables.

    Usage
    =====

    ``diop_linear(eq)``: Returns a tuple containing solutions to the
    diophantine equation ``eq``. Values in the tuple is arranged in the same
    order as the sorted variables.

    Details
    =======

    ``eq`` is a linear diophantine equation which is assumed to be zero.
    ``param`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_linear
    >>> from sympy.abc import x, y, z
    >>> diop_linear(2*x - 3*y - 5) # solves equation 2*x - 3*y - 5 == 0
    (3*t_0 - 5, 2*t_0 - 5)

    Here x = -3*t_0 - 5 and y = -2*t_0 - 5

    >>> diop_linear(2*x - 3*y - 4*z -3)
    (t_0, 2*t_0 + 4*t_1 + 3, -t_0 - 3*t_1 - 3)

    See Also
    ========

    diop_quadratic(), diop_ternary_quadratic(), diop_general_pythagorean(),
    diop_general_sum_of_squares()
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == Linear.name:
        parameters = None
        if param is not None:
            parameters = symbols('%s_0:%i' % (param, len(var)), integer=True)

        result = Linear(eq).solve(parameters=parameters)

        if param is None:
            result = result(*[0]*len(result.parameters))

        if len(result) > 0:
            return list(result)[0]
        else:
            return tuple([None]*len(result.parameters))


def base_solution_linear(c, a, b, t=None):
    """
    Return the base solution for the linear equation, `ax + by = c`.

    Explanation
    ===========

    Used by ``diop_linear()`` to find the base solution of a linear
    Diophantine equation. If ``t`` is given then the parametrized solution is
    returned.

    Usage
    =====

    ``base_solution_linear(c, a, b, t)``: ``a``, ``b``, ``c`` are coefficients
    in `ax + by = c` and ``t`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import base_solution_linear
    >>> from sympy.abc import t
    >>> base_solution_linear(5, 2, 3) # equation 2*x + 3*y = 5
    (-5, 5)
    >>> base_solution_linear(0, 5, 7) # equation 5*x + 7*y = 0
    (0, 0)
    >>> base_solution_linear(5, 2, 3, t) # equation 2*x + 3*y = 5
    (3*t - 5, 5 - 2*t)
    >>> base_solution_linear(0, 5, 7, t) # equation 5*x + 7*y = 0
    (7*t, -5*t)
    """
    a, b, c = _remove_gcd(a, b, c)

    if c == 0:
        if t is None:
            return (0, 0)
        if b < 0:
            t = -t
        return (b*t, -a*t)

    x0, y0, d = igcdex(abs(a), abs(b))
    x0 *= sign(a)
    y0 *= sign(b)
    if c % d:
        return (None, None)
    if t is None:
        return (c*x0, c*y0)
    if b < 0:
        t = -t
    return (c*x0 + b*t, c*y0 - a*t)


def diop_univariate(eq):
    """
    Solves a univariate diophantine equations.

    Explanation
    ===========

    A univariate diophantine equation is an equation of the form
    `a_{0} + a_{1}x + a_{2}x^2 + .. + a_{n}x^n = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x` is an integer variable.

    Usage
    =====

    ``diop_univariate(eq)``: Returns a set containing solutions to the
    diophantine equation ``eq``.

    Details
    =======

    ``eq`` is a univariate diophantine equation which is assumed to be zero.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_univariate
    >>> from sympy.abc import x
    >>> diop_univariate((x - 2)*(x - 3)**2) # solves equation (x - 2)*(x - 3)**2 == 0
    {(2,), (3,)}

    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == Univariate.name:
        return {(int(i),) for i in solveset_real(
            eq, var[0]).intersect(S.Integers)}


def divisible(a, b):
    """
    Returns `True` if ``a`` is divisible by ``b`` and `False` otherwise.
    """
    return not a % b


def diop_quadratic(eq, param=symbols("t", integer=True)):
    """
    Solves quadratic diophantine equations.

    i.e. equations of the form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`. Returns a
    set containing the tuples `(x, y)` which contains the solutions. If there
    are no solutions then `(None, None)` is returned.

    Usage
    =====

    ``diop_quadratic(eq, param)``: ``eq`` is a quadratic binary diophantine
    equation. ``param`` is used to indicate the parameter to be used in the
    solution.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``param`` is a parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.abc import x, y, t
    >>> from sympy.solvers.diophantine.diophantine import diop_quadratic
    >>> diop_quadratic(x**2 + y**2 + 2*x + 2*y + 2, t)
    {(-1, -1)}

    References
    ==========

    .. [1] Methods to solve Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0, [online],
          Available: https://www.alpertron.com.ar/METHODS.HTM
    .. [2] Solving the equation ax^2+ bxy + cy^2 + dx + ey + f= 0, [online],
          Available: https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

    See Also
    ========

    diop_linear(), diop_ternary_quadratic(), diop_general_sum_of_squares(),
    diop_general_pythagorean()
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == BinaryQuadratic.name:
        if param is not None:
            parameters = [param, Symbol("u", integer=True)]
        else:
            parameters = None
        return set(BinaryQuadratic(eq).solve(parameters=parameters))


def is_solution_quad(var, coeff, u, v):
    """
    Check whether `(u, v)` is solution to the quadratic binary diophantine
    equation with the variable list ``var`` and coefficient dictionary
    ``coeff``.

    Not intended for use by normal users.
    """
    reps = dict(zip(var, (u, v)))
    eq = Add(*[j*i.xreplace(reps) for i, j in coeff.items()])
    return _mexpand(eq) == 0


def diop_DN(D, N, t=symbols("t", integer=True)):
    """
    Solves the equation `x^2 - Dy^2 = N`.

    Explanation
    ===========

    Mainly concerned with the case `D > 0, D` is not a perfect square,
    which is the same as the generalized Pell equation. The LMM
    algorithm [1]_ is used to solve this equation.

    Returns one solution tuple, (`x, y)` for each class of the solutions.
    Other solutions of the class can be constructed according to the
    values of ``D`` and ``N``.

    Usage
    =====

    ``diop_DN(D, N, t)``: D and N are integers as in `x^2 - Dy^2 = N` and
    ``t`` is the parameter to be used in the solutions.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.
    ``t`` is the parameter to be used in the solutions.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_DN
    >>> diop_DN(13, -4) # Solves equation x**2 - 13*y**2 = -4
    [(3, 1), (393, 109), (36, 10)]

    The output can be interpreted as follows: There are three fundamental
    solutions to the equation `x^2 - 13y^2 = -4` given by (3, 1), (393, 109)
    and (36, 10). Each tuple is in the form (x, y), i.e. solution (3, 1) means
    that `x = 3` and `y = 1`.

    >>> diop_DN(986, 1) # Solves equation x**2 - 986*y**2 = 1
    [(49299, 1570)]

    See Also
    ========

    find_DN(), diop_bf_DN()

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Pages 16 - 17. [online], Available:
        https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    if D < 0:
        if N == 0:
            return [(0, 0)]
        if N < 0:
            return []
        # N > 0:
        sol = []
        for d in divisors(square_factor(N), generator=True):
            for x, y in cornacchia(1, int(-D), int(N // d**2)):
                sol.append((d*x, d*y))
                if D == -1:
                    sol.append((d*y, d*x))
        return sol

    if D == 0:
        if N < 0:
            return []
        if N == 0:
            return [(0, t)]
        sN, _exact = integer_nthroot(N, 2)
        if _exact:
            return [(sN, t)]
        return []

    # D > 0
    sD, _exact = integer_nthroot(D, 2)
    if _exact:
        if N == 0:
            return [(sD*t, t)]

        sol = []
        for y in range(floor(sign(N)*(N - 1)/(2*sD)) + 1):
            try:
                sq, _exact = integer_nthroot(D*y**2 + N, 2)
            except ValueError:
                _exact = False
            if _exact:
                sol.append((sq, y))
        return sol

    if 1 < N**2 < D:
        # It is much faster to call `_special_diop_DN`.
        return _special_diop_DN(D, N)

    if N == 0:
        return [(0, 0)]

    sol = []
    if abs(N) == 1:
        pqa = PQa(0, 1, D)
        *_, prev_B, prev_G = next(pqa)
        for j, (*_, a, _, _B, _G) in enumerate(pqa):
            if a == 2*sD:
                break
            prev_B, prev_G = _B, _G
        if j % 2:
            if N == 1:
                sol.append((prev_G, prev_B))
            return sol
        if N == -1:
            return [(prev_G, prev_B)]
        for _ in range(j):
            *_, _B, _G = next(pqa)
        return [(_G, _B)]

    for f in divisors(square_factor(N), generator=True):
        m = N // f**2
        am = abs(m)
        for sqm in sqrt_mod(D, am, all_roots=True):
            z = symmetric_residue(sqm, am)
            pqa = PQa(z, am, D)
            *_, prev_B, prev_G = next(pqa)
            for _ in range(length(z, am, D) - 1):
                _, q, *_, _B, _G = next(pqa)
                if abs(q) == 1:
                    if prev_G**2 - D*prev_B**2 == m:
                        sol.append((f*prev_G, f*prev_B))
                    elif a := diop_DN(D, -1):
                        sol.append((f*(prev_G*a[0][0] + prev_B*D*a[0][1]),
                                    f*(prev_G*a[0][1] + prev_B*a[0][0])))
                    break
                prev_B, prev_G = _B, _G
    return sol


def _special_diop_DN(D, N):
    """
    Solves the equation `x^2 - Dy^2 = N` for the special case where
    `1 < N**2 < D` and `D` is not a perfect square.
    It is better to call `diop_DN` rather than this function, as
    the former checks the condition `1 < N**2 < D`, and calls the latter only
    if appropriate.

    Usage
    =====

    WARNING: Internal method. Do not call directly!

    ``_special_diop_DN(D, N)``: D and N are integers as in `x^2 - Dy^2 = N`.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import _special_diop_DN
    >>> _special_diop_DN(13, -3) # Solves equation x**2 - 13*y**2 = -3
    [(7, 2), (137, 38)]

    The output can be interpreted as follows: There are two fundamental
    solutions to the equation `x^2 - 13y^2 = -3` given by (7, 2) and
    (137, 38). Each tuple is in the form (x, y), i.e. solution (7, 2) means
    that `x = 7` and `y = 2`.

    >>> _special_diop_DN(2445, -20) # Solves equation x**2 - 2445*y**2 = -20
    [(445, 9), (17625560, 356454), (698095554475, 14118073569)]

    See Also
    ========

    diop_DN()

    References
    ==========

    .. [1] Section 4.4.4 of the following book:
        Quadratic Diophantine Equations, T. Andreescu and D. Andrica,
        Springer, 2015.
    """

    # The following assertion was removed for efficiency, with the understanding
    #     that this method is not called directly. The parent method, `diop_DN`
    #     is responsible for performing the appropriate checks.
    #
    # assert (1 < N**2 < D) and (not integer_nthroot(D, 2)[1])

    sqrt_D = isqrt(D)
    F = {N // f**2: f for f in divisors(square_factor(abs(N)), generator=True)}
    P = 0
    Q = 1
    G0, G1 = 0, 1
    B0, B1 = 1, 0

    solutions = []
    while True:
        for _ in range(2):
            a = (P + sqrt_D) // Q
            P = a*Q - P
            Q = (D - P**2) // Q
            G0, G1 = G1, a*G1 + G0
            B0, B1 = B1, a*B1 + B0
            if (s := G1**2 - D*B1**2) in F:
                f = F[s]
                solutions.append((f*G1, f*B1))
        if Q == 1:
            break
    return solutions


def cornacchia(a:int, b:int, m:int) -> set[tuple[int, int]]:
    r"""
    Solves `ax^2 + by^2 = m` where `\gcd(a, b) = 1 = gcd(a, m)` and `a, b > 0`.

    Explanation
    ===========

    Uses the algorithm due to Cornacchia. The method only finds primitive
    solutions, i.e. ones with `\gcd(x, y) = 1`. So this method cannot be used to
    find the solutions of `x^2 + y^2 = 20` since the only solution to former is
    `(x, y) = (4, 2)` and it is not primitive. When `a = b`, only the
    solutions with `x \leq y` are found. For more details, see the References.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import cornacchia
    >>> cornacchia(2, 3, 35) # equation 2x**2 + 3y**2 = 35
    {(2, 3), (4, 1)}
    >>> cornacchia(1, 1, 25) # equation x**2 + y**2 = 25
    {(4, 3)}

    References
    ===========

    .. [1] A. Nitaj, "L'algorithme de Cornacchia"
    .. [2] Solving the diophantine equation ax**2 + by**2 = m by Cornacchia's
        method, [online], Available:
        http://www.numbertheory.org/php/cornacchia.html

    See Also
    ========

    sympy.utilities.iterables.signed_permutations
    """
    # Assume gcd(a, b) = gcd(a, m) = 1 and a, b > 0 but no error checking
    sols = set()

    if a + b > m:
        # xy = 0 must hold if there exists a solution
        if a == 1:
            # y = 0
            s, _exact = iroot(m // a, 2)
            if _exact:
                sols.add((int(s), 0))
            if a == b:
                # only keep one solution
                return sols
        if m % b == 0:
            # x = 0
            s, _exact = iroot(m // b, 2)
            if _exact:
                sols.add((0, int(s)))
        return sols

    # the original cornacchia
    for t in sqrt_mod_iter(-b*invert(a, m), m):
        if t < m // 2:
            continue
        u, r = m, t
        while (m1 := m - a*r**2) <= 0:
            u, r = r, u % r
        m1, _r = divmod(m1, b)
        if _r:
            continue
        s, _exact = iroot(m1, 2)
        if _exact:
            if a == b and r < s:
                r, s = s, r
            sols.add((int(r), int(s)))
    return sols


def PQa(P_0, Q_0, D):
    r"""
    Returns useful information needed to solve the Pell equation.

    Explanation
    ===========

    There are six sequences of integers defined related to the continued
    fraction representation of `\\frac{P + \sqrt{D}}{Q}`, namely {`P_{i}`},
    {`Q_{i}`}, {`a_{i}`},{`A_{i}`}, {`B_{i}`}, {`G_{i}`}. ``PQa()`` Returns
    these values as a 6-tuple in the same order as mentioned above. Refer [1]_
    for more detailed information.

    Usage
    =====

    ``PQa(P_0, Q_0, D)``: ``P_0``, ``Q_0`` and ``D`` are integers corresponding
    to `P_{0}`, `Q_{0}` and `D` in the continued fraction
    `\\frac{P_{0} + \sqrt{D}}{Q_{0}}`.
    Also it's assumed that `P_{0}^2 == D mod(|Q_{0}|)` and `D` is square free.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import PQa
    >>> pqa = PQa(13, 4, 5) # (13 + sqrt(5))/4
    >>> next(pqa) # (P_0, Q_0, a_0, A_0, B_0, G_0)
    (13, 4, 3, 3, 1, -1)
    >>> next(pqa) # (P_1, Q_1, a_1, A_1, B_1, G_1)
    (-1, 1, 1, 4, 1, 3)

    References
    ==========

    .. [1] Solving the generalized Pell equation x^2 - Dy^2 = N, John P.
        Robertson, July 31, 2004, Pages 4 - 8. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    sqD = isqrt(D)
    A2 = B1 = 0
    A1 = B2 = 1
    G1 = Q_0
    G2 = -P_0
    P_i = P_0
    Q_i = Q_0

    while True:
        a_i = (P_i + sqD) // Q_i
        A1, A2 = a_i*A1 + A2, A1
        B1, B2 = a_i*B1 + B2, B1
        G1, G2 = a_i*G1 + G2, G1
        yield P_i, Q_i, a_i, A1, B1, G1

        P_i = a_i*Q_i - P_i
        Q_i = (D - P_i**2) // Q_i


def diop_bf_DN(D, N, t=symbols("t", integer=True)):
    r"""
    Uses brute force to solve the equation, `x^2 - Dy^2 = N`.

    Explanation
    ===========

    Mainly concerned with the generalized Pell equation which is the case when
    `D > 0, D` is not a perfect square. For more information on the case refer
    [1]_. Let `(t, u)` be the minimal positive solution of the equation
    `x^2 - Dy^2 = 1`. Then this method requires
    `\sqrt{\\frac{\mid N \mid (t \pm 1)}{2D}}` to be small.

    Usage
    =====

    ``diop_bf_DN(D, N, t)``: ``D`` and ``N`` are coefficients in
    `x^2 - Dy^2 = N` and ``t`` is the parameter to be used in the solutions.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.
    ``t`` is the parameter to be used in the solutions.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_bf_DN
    >>> diop_bf_DN(13, -4)
    [(3, 1), (-3, 1), (36, 10)]
    >>> diop_bf_DN(986, 1)
    [(49299, 1570)]

    See Also
    ========

    diop_DN()

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Page 15. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    D = as_int(D)
    N = as_int(N)

    sol = []
    a = diop_DN(D, 1)
    u = a[0][0]

    if N == 0:
        if D < 0:
            return [(0, 0)]
        if D == 0:
            return [(0, t)]
        sD, _exact = integer_nthroot(D, 2)
        if _exact:
            return [(sD*t, t), (-sD*t, t)]
        return [(0, 0)]

    if abs(N) == 1:
        return diop_DN(D, N)

    if N > 1:
        L1 = 0
        L2 = integer_nthroot(int(N*(u - 1)/(2*D)), 2)[0] + 1
    else: # N < -1
        L1, _exact = integer_nthroot(-int(N/D), 2)
        if not _exact:
            L1 += 1
        L2 = integer_nthroot(-int(N*(u + 1)/(2*D)), 2)[0] + 1

    for y in range(L1, L2):
        try:
            x, _exact = integer_nthroot(N + D*y**2, 2)
        except ValueError:
            _exact = False
        if _exact:
            sol.append((x, y))
            if not equivalent(x, y, -x, y, D, N):
                sol.append((-x, y))

    return sol


def equivalent(u, v, r, s, D, N):
    """
    Returns True if two solutions `(u, v)` and `(r, s)` of `x^2 - Dy^2 = N`
    belongs to the same equivalence class and False otherwise.

    Explanation
    ===========

    Two solutions `(u, v)` and `(r, s)` to the above equation fall to the same
    equivalence class iff both `(ur - Dvs)` and `(us - vr)` are divisible by
    `N`. See reference [1]_. No test is performed to test whether `(u, v)` and
    `(r, s)` are actually solutions to the equation. User should take care of
    this.

    Usage
    =====

    ``equivalent(u, v, r, s, D, N)``: `(u, v)` and `(r, s)` are two solutions
    of the equation `x^2 - Dy^2 = N` and all parameters involved are integers.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import equivalent
    >>> equivalent(18, 5, -18, -5, 13, -1)
    True
    >>> equivalent(3, 1, -18, 393, 109, -4)
    False

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Page 12. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf

    """
    return divisible(u*r - D*v*s, N) and divisible(u*s - v*r, N)


def length(P, Q, D):
    r"""
    Returns the (length of aperiodic part + length of periodic part) of
    continued fraction representation of `\\frac{P + \sqrt{D}}{Q}`.

    It is important to remember that this does NOT return the length of the
    periodic part but the sum of the lengths of the two parts as mentioned
    above.

    Usage
    =====

    ``length(P, Q, D)``: ``P``, ``Q`` and ``D`` are integers corresponding to
    the continued fraction `\\frac{P + \sqrt{D}}{Q}`.

    Details
    =======

    ``P``, ``D`` and ``Q`` corresponds to P, D and Q in the continued fraction,
    `\\frac{P + \sqrt{D}}{Q}`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import length
    >>> length(-2, 4, 5) # (-2 + sqrt(5))/4
    3
    >>> length(-5, 4, 17) # (-5 + sqrt(17))/4
    4

    See Also
    ========
    sympy.ntheory.continued_fraction.continued_fraction_periodic
    """
    from sympy.ntheory.continued_fraction import continued_fraction_periodic
    v = continued_fraction_periodic(P, Q, D)
    if isinstance(v[-1], list):
        rpt = len(v[-1])
        nonrpt = len(v) - 1
    else:
        rpt = 0
        nonrpt = len(v)
    return rpt + nonrpt


def transformation_to_DN(eq):
    """
    This function transforms general quadratic,
    `ax^2 + bxy + cy^2 + dx + ey + f = 0`
    to more easy to deal with `X^2 - DY^2 = N` form.

    Explanation
    ===========

    This is used to solve the general quadratic equation by transforming it to
    the latter form. Refer to [1]_ for more detailed information on the
    transformation. This function returns a tuple (A, B) where A is a 2 X 2
    matrix and B is a 2 X 1 matrix such that,

    Transpose([x y]) =  A * Transpose([X Y]) + B

    Usage
    =====

    ``transformation_to_DN(eq)``: where ``eq`` is the quadratic to be
    transformed.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import transformation_to_DN
    >>> A, B = transformation_to_DN(x**2 - 3*x*y - y**2 - 2*y + 1)
    >>> A
    Matrix([
    [1/26, 3/26],
    [   0, 1/13]])
    >>> B
    Matrix([
    [-6/13],
    [-4/13]])

    A, B  returned are such that Transpose((x y)) =  A * Transpose((X Y)) + B.
    Substituting these values for `x` and `y` and a bit of simplifying work
    will give an equation of the form `x^2 - Dy^2 = N`.

    >>> from sympy.abc import X, Y
    >>> from sympy import Matrix, simplify
    >>> u = (A*Matrix([X, Y]) + B)[0] # Transformation for x
    >>> u
    X/26 + 3*Y/26 - 6/13
    >>> v = (A*Matrix([X, Y]) + B)[1] # Transformation for y
    >>> v
    Y/13 - 4/13

    Next we will substitute these formulas for `x` and `y` and do
    ``simplify()``.

    >>> eq = simplify((x**2 - 3*x*y - y**2 - 2*y + 1).subs(zip((x, y), (u, v))))
    >>> eq
    X**2/676 - Y**2/52 + 17/13

    By multiplying the denominator appropriately, we can get a Pell equation
    in the standard form.

    >>> eq * 676
    X**2 - 13*Y**2 + 884

    If only the final equation is needed, ``find_DN()`` can be used.

    See Also
    ========

    find_DN()

    References
    ==========

    .. [1] Solving the equation ax^2 + bxy + cy^2 + dx + ey + f = 0,
           John P.Robertson, May 8, 2003, Page 7 - 11.
           https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf
    """

    var, coeff, diop_type = classify_diop(eq, _dict=False)
    if diop_type == BinaryQuadratic.name:
        return _transformation_to_DN(var, coeff)


def _transformation_to_DN(var, coeff):

    x, y = var

    a = coeff[x**2]
    b = coeff[x*y]
    c = coeff[y**2]
    d = coeff[x]
    e = coeff[y]
    f = coeff[1]

    a, b, c, d, e, f = [as_int(i) for i in _remove_gcd(a, b, c, d, e, f)]

    X, Y = symbols("X, Y", integer=True)

    if b:
        B, C = _rational_pq(2*a, b)
        A, T = _rational_pq(a, B**2)

        # eq_1 = A*B*X**2 + B*(c*T - A*C**2)*Y**2 + d*T*X + (B*e*T - d*T*C)*Y + f*T*B
        coeff = {X**2: A*B, X*Y: 0, Y**2: B*(c*T - A*C**2), X: d*T, Y: B*e*T - d*T*C, 1: f*T*B}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return Matrix(2, 2, [S.One/B, -S(C)/B, 0, 1])*A_0, Matrix(2, 2, [S.One/B, -S(C)/B, 0, 1])*B_0

    if d:
        B, C = _rational_pq(2*a, d)
        A, T = _rational_pq(a, B**2)

        # eq_2 = A*X**2 + c*T*Y**2 + e*T*Y + f*T - A*C**2
        coeff = {X**2: A, X*Y: 0, Y**2: c*T, X: 0, Y: e*T, 1: f*T - A*C**2}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return Matrix(2, 2, [S.One/B, 0, 0, 1])*A_0, Matrix(2, 2, [S.One/B, 0, 0, 1])*B_0 + Matrix([-S(C)/B, 0])

    if e:
        B, C = _rational_pq(2*c, e)
        A, T = _rational_pq(c, B**2)

        # eq_3 = a*T*X**2 + A*Y**2 + f*T - A*C**2
        coeff = {X**2: a*T, X*Y: 0, Y**2: A, X: 0, Y: 0, 1: f*T - A*C**2}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return Matrix(2, 2, [1, 0, 0, S.One/B])*A_0, Matrix(2, 2, [1, 0, 0, S.One/B])*B_0 + Matrix([0, -S(C)/B])

    # TODO: pre-simplification: Not necessary but may simplify
    # the equation.
    return Matrix(2, 2, [S.One/a, 0, 0, 1]), Matrix([0, 0])


def find_DN(eq):
    """
    This function returns a tuple, `(D, N)` of the simplified form,
    `x^2 - Dy^2 = N`, corresponding to the general quadratic,
    `ax^2 + bxy + cy^2 + dx + ey + f = 0`.

    Solving the general quadratic is then equivalent to solving the equation
    `X^2 - DY^2 = N` and transforming the solutions by using the transformation
    matrices returned by ``transformation_to_DN()``.

    Usage
    =====

    ``find_DN(eq)``: where ``eq`` is the quadratic to be transformed.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import find_DN
    >>> find_DN(x**2 - 3*x*y - y**2 - 2*y + 1)
    (13, -884)

    Interpretation of the output is that we get `X^2 -13Y^2 = -884` after
    transforming `x^2 - 3xy - y^2 - 2y + 1` using the transformation returned
    by ``transformation_to_DN()``.

    See Also
    ========

    transformation_to_DN()

    References
    ==========

    .. [1] Solving the equation ax^2 + bxy + cy^2 + dx + ey + f = 0,
           John P.Robertson, May 8, 2003, Page 7 - 11.
           https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)
    if diop_type == BinaryQuadratic.name:
        return _find_DN(var, coeff)


def _find_DN(var, coeff):

    x, y = var
    X, Y = symbols("X, Y", integer=True)
    A, B = _transformation_to_DN(var, coeff)

    u = (A*Matrix([X, Y]) + B)[0]
    v = (A*Matrix([X, Y]) + B)[1]
    eq = x**2*coeff[x**2] + x*y*coeff[x*y] + y**2*coeff[y**2] + x*coeff[x] + y*coeff[y] + coeff[1]

    simplified = _mexpand(eq.subs(zip((x, y), (u, v))))

    coeff = simplified.as_coefficients_dict()

    return -coeff[Y**2]/coeff[X**2], -coeff[1]/coeff[X**2]


def check_param(x, y, a, params):
    """
    If there is a number modulo ``a`` such that ``x`` and ``y`` are both
    integers, then return a parametric representation for ``x`` and ``y``
    else return (None, None).

    Here ``x`` and ``y`` are functions of ``t``.
    """
    from sympy.simplify.simplify import clear_coefficients

    if x.is_number and not x.is_Integer:
        return DiophantineSolutionSet([x, y], parameters=params)

    if y.is_number and not y.is_Integer:
        return DiophantineSolutionSet([x, y], parameters=params)

    m, n = symbols("m, n", integer=True)
    c, p = (m*x + n*y).as_content_primitive()
    if a % c.q:
        return DiophantineSolutionSet([x, y], parameters=params)

    # clear_coefficients(mx + b, R)[1] -> (R - b)/m
    eq = clear_coefficients(x, m)[1] - clear_coefficients(y, n)[1]
    junk, eq = eq.as_content_primitive()

    return _diop_solve(eq, params=params)


def diop_ternary_quadratic(eq, parameterize=False):
    """
    Solves the general quadratic ternary form,
    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.

    Returns a tuple `(x, y, z)` which is a base solution for the above
    equation. If there are no solutions, `(None, None, None)` is returned.

    Usage
    =====

    ``diop_ternary_quadratic(eq)``: Return a tuple containing a basic solution
    to ``eq``.

    Details
    =======

    ``eq`` should be an homogeneous expression of degree two in three variables
    and it is assumed to be zero.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic
    >>> diop_ternary_quadratic(x**2 + 3*y**2 - z**2)
    (1, 0, 1)
    >>> diop_ternary_quadratic(4*x**2 + 5*y**2 - z**2)
    (1, 0, 2)
    >>> diop_ternary_quadratic(45*x**2 - 7*y**2 - 8*x*y - z**2)
    (28, 45, 105)
    >>> diop_ternary_quadratic(x**2 - 49*y**2 - z**2 + 13*z*y -8*x*y)
    (9, 1, 5)
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            HomogeneousTernaryQuadratic.name,
            HomogeneousTernaryQuadraticNormal.name):
        sol = _diop_ternary_quadratic(var, coeff)
        if len(sol) > 0:
            x_0, y_0, z_0 = list(sol)[0]
        else:
            x_0, y_0, z_0 = None, None, None

        if parameterize:
            return _parametrize_ternary_quadratic(
                (x_0, y_0, z_0), var, coeff)
        return x_0, y_0, z_0


def _diop_ternary_quadratic(_var, coeff):
    eq = sum(i*coeff[i] for i in coeff)
    if HomogeneousTernaryQuadratic(eq).matches():
        return HomogeneousTernaryQuadratic(eq, free_symbols=_var).solve()
    elif HomogeneousTernaryQuadraticNormal(eq).matches():
        return HomogeneousTernaryQuadraticNormal(eq, free_symbols=_var).solve()


def transformation_to_normal(eq):
    """
    Returns the transformation Matrix that converts a general ternary
    quadratic equation ``eq`` (`ax^2 + by^2 + cz^2 + dxy + eyz + fxz`)
    to a form without cross terms: `ax^2 + by^2 + cz^2 = 0`. This is
    not used in solving ternary quadratics; it is only implemented for
    the sake of completeness.
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            "homogeneous_ternary_quadratic",
            "homogeneous_ternary_quadratic_normal"):
        return _transformation_to_normal(var, coeff)


def _transformation_to_normal(var, coeff):

    _var = list(var)  # copy
    x, y, z = var

    if not any(coeff[i**2] for i in var):
        # https://math.stackexchange.com/questions/448051/transform-quadratic-ternary-form-to-normal-form/448065#448065
        a = coeff[x*y]
        b = coeff[y*z]
        c = coeff[x*z]
        swap = False
        if not a:  # b can't be 0 or else there aren't 3 vars
            swap = True
            a, b = b, a
        T = Matrix(((1, 1, -b/a), (1, -1, -c/a), (0, 0, 1)))
        if swap:
            T.row_swap(0, 1)
            T.col_swap(0, 1)
        return T

    if coeff[x**2] == 0:
        # If the coefficient of x is zero change the variables
        if coeff[y**2] == 0:
            _var[0], _var[2] = var[2], var[0]
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T

        _var[0], _var[1] = var[1], var[0]
        T = _transformation_to_normal(_var, coeff)
        T.row_swap(0, 1)
        T.col_swap(0, 1)
        return T

    # Apply the transformation x --> X - (B*Y + C*Z)/(2*A)
    if coeff[x*y] != 0 or coeff[x*z] != 0:
        A = coeff[x**2]
        B = coeff[x*y]
        C = coeff[x*z]
        D = coeff[y**2]
        E = coeff[y*z]
        F = coeff[z**2]

        _coeff = {}

        _coeff[x**2] = 4*A**2
        _coeff[y**2] = 4*A*D - B**2
        _coeff[z**2] = 4*A*F - C**2
        _coeff[y*z] = 4*A*E - 2*B*C
        _coeff[x*y] = 0
        _coeff[x*z] = 0

        T_0 = _transformation_to_normal(_var, _coeff)
        return Matrix(3, 3, [1, S(-B)/(2*A), S(-C)/(2*A), 0, 1, 0, 0, 0, 1])*T_0

    elif coeff[y*z] != 0:
        if coeff[y**2] == 0:
            if coeff[z**2] == 0:
                # Equations of the form A*x**2 + E*yz = 0.
                # Apply transformation y -> Y + Z ans z -> Y - Z
                return Matrix(3, 3, [1, 0, 0, 0, 1, 1, 0, 1, -1])

            # Ax**2 + E*y*z + F*z**2  = 0
            _var[0], _var[2] = var[2], var[0]
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T

        # A*x**2 + D*y**2 + E*y*z + F*z**2 = 0, F may be zero
        _var[0], _var[1] = var[1], var[0]
        T = _transformation_to_normal(_var, coeff)
        T.row_swap(0, 1)
        T.col_swap(0, 1)
        return T

    return Matrix.eye(3)


def parametrize_ternary_quadratic(eq):
    """
    Returns the parametrized general solution for the ternary quadratic
    equation ``eq`` which has the form
    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.

    Examples
    ========

    >>> from sympy import Tuple, ordered
    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import parametrize_ternary_quadratic

    The parametrized solution may be returned with three parameters:

    >>> parametrize_ternary_quadratic(2*x**2 + y**2 - 2*z**2)
    (p**2 - 2*q**2, -2*p**2 + 4*p*q - 4*p*r - 4*q**2, p**2 - 4*p*q + 2*q**2 - 4*q*r)

    There might also be only two parameters:

    >>> parametrize_ternary_quadratic(4*x**2 + 2*y**2 - 3*z**2)
    (2*p**2 - 3*q**2, -4*p**2 + 12*p*q - 6*q**2, 4*p**2 - 8*p*q + 6*q**2)

    Notes
    =====

    Consider ``p`` and ``q`` in the previous 2-parameter
    solution and observe that more than one solution can be represented
    by a given pair of parameters. If `p` and ``q`` are not coprime, this is
    trivially true since the common factor will also be a common factor of the
    solution values. But it may also be true even when ``p`` and
    ``q`` are coprime:

    >>> sol = Tuple(*_)
    >>> p, q = ordered(sol.free_symbols)
    >>> sol.subs([(p, 3), (q, 2)])
    (6, 12, 12)
    >>> sol.subs([(q, 1), (p, 1)])
    (-1, 2, 2)
    >>> sol.subs([(q, 0), (p, 1)])
    (2, -4, 4)
    >>> sol.subs([(q, 1), (p, 0)])
    (-3, -6, 6)

    Except for sign and a common factor, these are equivalent to
    the solution of (1, 2, 2).

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.

    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            "homogeneous_ternary_quadratic",
            "homogeneous_ternary_quadratic_normal"):
        x_0, y_0, z_0 = list(_diop_ternary_quadratic(var, coeff))[0]
        return _parametrize_ternary_quadratic(
            (x_0, y_0, z_0), var, coeff)


def _parametrize_ternary_quadratic(solution, _var, coeff):
    # called for a*x**2 + b*y**2 + c*z**2 + d*x*y + e*y*z + f*x*z = 0
    assert 1 not in coeff

    x_0, y_0, z_0 = solution

    v = list(_var)  # copy

    if x_0 is None:
        return (None, None, None)

    if solution.count(0) >= 2:
        # if there are 2 zeros the equation reduces
        # to k*X**2 == 0 where X is x, y, or z so X must
        # be zero, too. So there is only the trivial
        # solution.
        return (None, None, None)

    if x_0 == 0:
        v[0], v[1] = v[1], v[0]
        y_p, x_p, z_p = _parametrize_ternary_quadratic(
            (y_0, x_0, z_0), v, coeff)
        return x_p, y_p, z_p

    x, y, z = v
    r, p, q = symbols("r, p, q", integer=True)

    eq = sum(k*v for k, v in coeff.items())
    eq_1 = _mexpand(eq.subs(zip(
        (x, y, z), (r*x_0, r*y_0 + p, r*z_0 + q))))
    A, B = eq_1.as_independent(r, as_Add=True)


    x = A*x_0
    y = (A*y_0 - _mexpand(B/r*p))
    z = (A*z_0 - _mexpand(B/r*q))

    return _remove_gcd(x, y, z)


def diop_ternary_quadratic_normal(eq, parameterize=False):
    """
    Solves the quadratic ternary diophantine equation,
    `ax^2 + by^2 + cz^2 = 0`.

    Explanation
    ===========

    Here the coefficients `a`, `b`, and `c` should be non zero. Otherwise the
    equation will be a quadratic binary or univariate equation. If solvable,
    returns a tuple `(x, y, z)` that satisfies the given equation. If the
    equation does not have integer solutions, `(None, None, None)` is returned.

    Usage
    =====

    ``diop_ternary_quadratic_normal(eq)``: where ``eq`` is an equation of the form
    `ax^2 + by^2 + cz^2 = 0`.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic_normal
    >>> diop_ternary_quadratic_normal(x**2 + 3*y**2 - z**2)
    (1, 0, 1)
    >>> diop_ternary_quadratic_normal(4*x**2 + 5*y**2 - z**2)
    (1, 0, 2)
    >>> diop_ternary_quadratic_normal(34*x**2 - 3*y**2 - 301*z**2)
    (4, 9, 1)
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)
    if diop_type == HomogeneousTernaryQuadraticNormal.name:
        sol = _diop_ternary_quadratic_normal(var, coeff)
        if len(sol) > 0:
            x_0, y_0, z_0 = list(sol)[0]
        else:
            x_0, y_0, z_0 = None, None, None
        if parameterize:
            return _parametrize_ternary_quadratic(
                (x_0, y_0, z_0), var, coeff)
        return x_0, y_0, z_0


def _diop_ternary_quadratic_normal(var, coeff):
    eq = sum(i * coeff[i] for i in coeff)
    return HomogeneousTernaryQuadraticNormal(eq, free_symbols=var).solve()


def sqf_normal(a, b, c, steps=False):
    """
    Return `a', b', c'`, the coefficients of the square-free normal
    form of `ax^2 + by^2 + cz^2 = 0`, where `a', b', c'` are pairwise
    prime.  If `steps` is True then also return three tuples:
    `sq`, `sqf`, and `(a', b', c')` where `sq` contains the square
    factors of `a`, `b` and `c` after removing the `gcd(a, b, c)`;
    `sqf` contains the values of `a`, `b` and `c` after removing
    both the `gcd(a, b, c)` and the square factors.

    The solutions for `ax^2 + by^2 + cz^2 = 0` can be
    recovered from the solutions of `a'x^2 + b'y^2 + c'z^2 = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sqf_normal
    >>> sqf_normal(2 * 3**2 * 5, 2 * 5 * 11, 2 * 7**2 * 11)
    (11, 1, 5)
    >>> sqf_normal(2 * 3**2 * 5, 2 * 5 * 11, 2 * 7**2 * 11, True)
    ((3, 1, 7), (5, 55, 11), (11, 1, 5))

    References
    ==========

    .. [1] Legendre's Theorem, Legrange's Descent,
           https://public.csusm.edu/aitken_html/notes/legendre.pdf


    See Also
    ========

    reconstruct()
    """
    ABC = _remove_gcd(a, b, c)
    sq = tuple(square_factor(i) for i in ABC)
    sqf = A, B, C = tuple([i//j**2 for i,j in zip(ABC, sq)])
    pc = igcd(A, B)
    A /= pc
    B /= pc
    pa = igcd(B, C)
    B /= pa
    C /= pa
    pb = igcd(A, C)
    A /= pb
    B /= pb

    A *= pa
    B *= pb
    C *= pc

    if steps:
        return (sq, sqf, (A, B, C))
    else:
        return A, B, C


def square_factor(a):
    r"""
    Returns an integer `c` s.t. `a = c^2k, \ c,k \in Z`. Here `k` is square
    free. `a` can be given as an integer or a dictionary of factors.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import square_factor
    >>> square_factor(24)
    2
    >>> square_factor(-36*3)
    6
    >>> square_factor(1)
    1
    >>> square_factor({3: 2, 2: 1, -1: 1})  # -18
    3

    See Also
    ========
    sympy.ntheory.factor_.core
    """
    f = a if isinstance(a, dict) else factorint(a)
    return Mul(*[p**(e//2) for p, e in f.items()])


def reconstruct(A, B, z):
    """
    Reconstruct the `z` value of an equivalent solution of `ax^2 + by^2 + cz^2`
    from the `z` value of a solution of the square-free normal form of the
    equation, `a'*x^2 + b'*y^2 + c'*z^2`, where `a'`, `b'` and `c'` are square
    free and `gcd(a', b', c') == 1`.
    """
    f = factorint(igcd(A, B))
    for p, e in f.items():
        if e != 1:
            raise ValueError('a and b should be square-free')
        z *= p
    return z


def ldescent(A, B):
    """
    Return a non-trivial solution to `w^2 = Ax^2 + By^2` using
    Lagrange's method; return None if there is no such solution.

    Parameters
    ==========

    A : Integer
    B : Integer
        non-zero integer

    Returns
    =======

    (int, int, int) | None : a tuple `(w_0, x_0, y_0)` which is a solution to the above equation.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import ldescent
    >>> ldescent(1, 1) # w^2 = x^2 + y^2
    (1, 1, 0)
    >>> ldescent(4, -7) # w^2 = 4x^2 - 7y^2
    (2, -1, 0)

    This means that `x = -1, y = 0` and `w = 2` is a solution to the equation
    `w^2 = 4x^2 - 7y^2`

    >>> ldescent(5, -1) # w^2 = 5x^2 - y^2
    (2, 1, -1)

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.
    .. [2] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    if A == 0 or B == 0:
        raise ValueError("A and B must be non-zero integers")
    if abs(A) > abs(B):
        w, y, x = ldescent(B, A)
        return w, x, y
    if A == 1:
        return (1, 1, 0)
    if B == 1:
        return (1, 0, 1)
    if B == -1:  # and A == -1
        return

    r = sqrt_mod(A, B)
    if r is None:
        return
    Q = (r**2 - A) // B
    if Q == 0:
        return r, -1, 0
    for i in divisors(Q):
        d, _exact = integer_nthroot(abs(Q) // i, 2)
        if _exact:
            B_0 = sign(Q)*i
            W, X, Y = ldescent(A, B_0)
            return _remove_gcd(-A*X + r*W, r*X - W, Y*B_0*d)


def descent(A, B):
    """
    Returns a non-trivial solution, (x, y, z), to `x^2 = Ay^2 + Bz^2`
    using Lagrange's descent method with lattice-reduction. `A` and `B`
    are assumed to be valid for such a solution to exist.

    This is faster than the normal Lagrange's descent algorithm because
    the Gaussian reduction is used.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import descent
    >>> descent(3, 1) # x**2 = 3*y**2 + z**2
    (1, 0, 1)

    `(x, y, z) = (1, 0, 1)` is a solution to the above equation.

    >>> descent(41, -113)
    (-16, -3, 1)

    References
    ==========

    .. [1] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    if abs(A) > abs(B):
        x, y, z = descent(B, A)
        return x, z, y

    if B == 1:
        return (1, 0, 1)
    if A == 1:
        return (1, 1, 0)
    if B == -A:
        return (0, 1, 1)
    if B == A:
        x, z, y = descent(-1, A)
        return (A*y, z, x)

    w = sqrt_mod(A, B)
    x_0, z_0 = gaussian_reduce(w, A, B)

    t = (x_0**2 - A*z_0**2) // B
    t_2 = square_factor(t)
    t_1 = t // t_2**2

    x_1, z_1, y_1 = descent(A, t_1)

    return _remove_gcd(x_0*x_1 + A*z_0*z_1, z_0*x_1 + x_0*z_1, t_1*t_2*y_1)


def gaussian_reduce(w:int, a:int, b:int) -> tuple[int, int]:
    r"""
    Returns a reduced solution `(x, z)` to the congruence
    `X^2 - aZ^2 \equiv 0 \pmod{b}` so that `x^2 + |a|z^2` is as small as possible.
    Here ``w`` is a solution of the congruence `x^2 \equiv a \pmod{b}`.

    This function is intended to be used only for ``descent()``.

    Explanation
    ===========

    The Gaussian reduction can find the shortest vector for any norm.
    So we define the special norm for the vectors `u = (u_1, u_2)` and `v = (v_1, v_2)` as follows.

    .. math ::
        u \cdot v := (wu_1 + bu_2)(wv_1 + bv_2) + |a|u_1v_1

    Note that, given the mapping `f: (u_1, u_2) \to (wu_1 + bu_2, u_1)`,
    `f((u_1,u_2))` is the solution to `X^2 - aZ^2 \equiv 0 \pmod{b}`.
    In other words, finding the shortest vector in this norm will yield a solution with smaller `X^2 + |a|Z^2`.
    The algorithm starts from basis vectors `(0, 1)` and `(1, 0)`
    (corresponding to solutions `(b, 0)` and `(w, 1)`, respectively) and finds the shortest vector.
    The shortest vector does not necessarily correspond to the smallest solution,
    but since ``descent()`` only wants the smallest possible solution, it is sufficient.

    Parameters
    ==========

    w : int
        ``w`` s.t. `w^2 \equiv a \pmod{b}`
    a : int
        square-free nonzero integer
    b : int
        square-free nonzero integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import gaussian_reduce
    >>> from sympy.ntheory.residue_ntheory import sqrt_mod
    >>> a, b = 19, 101
    >>> gaussian_reduce(sqrt_mod(a, b), a, b) # 1**2 - 19*(-4)**2 = -303
    (1, -4)
    >>> a, b = 11, 14
    >>> x, z = gaussian_reduce(sqrt_mod(a, b), a, b)
    >>> (x**2 - a*z**2) % b == 0
    True

    It does not always return the smallest solution.

    >>> a, b = 6, 95
    >>> min_x, min_z = 1, 4
    >>> x, z = gaussian_reduce(sqrt_mod(a, b), a, b)
    >>> (x**2 - a*z**2) % b == 0 and (min_x**2 - a*min_z**2) % b == 0
    True
    >>> min_x**2 + abs(a)*min_z**2 < x**2 + abs(a)*z**2
    True

    References
    ==========

    .. [1] Gaussian lattice Reduction [online]. Available:
           https://web.archive.org/web/20201021115213/http://home.ie.cuhk.edu.hk/~wkshum/wordpress/?p=404
    .. [2] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    a = abs(a)
    def _dot(u, v):
        return u[0]*v[0] + a*u[1]*v[1]

    u = (b, 0)
    v = (w, 1) if b*w >= 0 else (-w, -1)
    # i.e., _dot(u, v) >= 0

    if b**2 < w**2 + a:
        u, v = v, u
    # i.e., norm(u) >= norm(v), where norm(u) := sqrt(_dot(u, u))

    while _dot(u, u) > (dv := _dot(v, v)):
        k = _dot(u, v) // dv
        u, v = v, (u[0] - k*v[0], u[1] - k*v[1])
    c = (v[0] - u[0], v[1] - u[1])
    if _dot(c, c) <= _dot(u, u) <= 2*_dot(u, v):
        return c
    return u


def holzer(x, y, z, a, b, c):
    r"""
    Simplify the solution `(x, y, z)` of the equation
    `ax^2 + by^2 = cz^2` with `a, b, c > 0` and `z^2 \geq \mid ab \mid` to
    a new reduced solution `(x', y', z')` such that `z'^2 \leq \mid ab \mid`.

    The algorithm is an interpretation of Mordell's reduction as described
    on page 8 of Cremona and Rusin's paper [1]_ and the work of Mordell in
    reference [2]_.

    References
    ==========

    .. [1] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    .. [2] Diophantine Equations, L. J. Mordell, page 48.

    """

    if _odd(c):
        k = 2*c
    else:
        k = c//2

    small = a*b*c
    step = 0
    while True:
        t1, t2, t3 = a*x**2, b*y**2, c*z**2
        # check that it's a solution
        if t1 + t2 != t3:
            if step == 0:
                raise ValueError('bad starting solution')
            break
        x_0, y_0, z_0 = x, y, z
        if max(t1, t2, t3) <= small:
            # Holzer condition
            break

        uv = u, v = base_solution_linear(k, y_0, -x_0)
        if None in uv:
            break

        p, q = -(a*u*x_0 + b*v*y_0), c*z_0
        r = Rational(p, q)
        if _even(c):
            w = _nint_or_floor(p, q)
            assert abs(w - r) <= S.Half
        else:
            w = p//q  # floor
            if _odd(a*u + b*v + c*w):
                w += 1
            assert abs(w - r) <= S.One

        A = (a*u**2 + b*v**2 + c*w**2)
        B = (a*u*x_0 + b*v*y_0 + c*w*z_0)
        x = Rational(x_0*A - 2*u*B, k)
        y = Rational(y_0*A - 2*v*B, k)
        z = Rational(z_0*A - 2*w*B, k)
        assert all(i.is_Integer for i in (x, y, z))
        step += 1

    return tuple([int(i) for i in (x_0, y_0, z_0)])


def diop_general_pythagorean(eq, param=symbols("m", integer=True)):
    """
    Solves the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Returns a tuple which contains a parametrized solution to the equation,
    sorted in the same order as the input variables.

    Usage
    =====

    ``diop_general_pythagorean(eq, param)``: where ``eq`` is a general
    pythagorean equation which is assumed to be zero and ``param`` is the base
    parameter used to construct other parameters by subscripting.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_pythagorean
    >>> from sympy.abc import a, b, c, d, e
    >>> diop_general_pythagorean(a**2 + b**2 + c**2 - d**2)
    (m1**2 + m2**2 - m3**2, 2*m1*m3, 2*m2*m3, m1**2 + m2**2 + m3**2)
    >>> diop_general_pythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2)
    (10*m1**2  + 10*m2**2  + 10*m3**2 - 10*m4**2, 15*m1**2  + 15*m2**2  + 15*m3**2  + 15*m4**2, 15*m1*m4, 12*m2*m4, 60*m3*m4)
    """
    var, coeff, diop_type  = classify_diop(eq, _dict=False)

    if diop_type == GeneralPythagorean.name:
        if param is None:
            params = None
        else:
            params = symbols('%s1:%i' % (param, len(var)), integer=True)
        return list(GeneralPythagorean(eq).solve(parameters=params))[0]


def diop_general_sum_of_squares(eq, limit=1):
    r"""
    Solves the equation `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Returns at most ``limit`` number of solutions.

    Usage
    =====

    ``general_sum_of_squares(eq, limit)`` : Here ``eq`` is an expression which
    is assumed to be zero. Also, ``eq`` should be in the form,
    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Details
    =======

    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \in Z` then there will be
    no solutions. Refer to [1]_ for more details.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_sum_of_squares
    >>> from sympy.abc import a, b, c, d, e
    >>> diop_general_sum_of_squares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345)
    {(15, 22, 22, 24, 24)}

    Reference
    =========

    .. [1] Representing an integer as a sum of three squares, [online],
        Available:
        https://proofwiki.org/wiki/Integer_as_Sum_of_Three_Squares
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == GeneralSumOfSquares.name:
        return set(GeneralSumOfSquares(eq).solve(limit=limit))


def diop_general_sum_of_even_powers(eq, limit=1):
    """
    Solves the equation `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`
    where `e` is an even, integer power.

    Returns at most ``limit`` number of solutions.

    Usage
    =====

    ``general_sum_of_even_powers(eq, limit)`` : Here ``eq`` is an expression which
    is assumed to be zero. Also, ``eq`` should be in the form,
    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_sum_of_even_powers
    >>> from sympy.abc import a, b
    >>> diop_general_sum_of_even_powers(a**4 + b**4 - (2**4 + 3**4))
    {(2, 3)}

    See Also
    ========

    power_representation
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == GeneralSumOfEvenPowers.name:
        return set(GeneralSumOfEvenPowers(eq).solve(limit=limit))


## Functions below this comment can be more suitably grouped under
## an Additive number theory module rather than the Diophantine
## equation module.


def partition(n, k=None, zeros=False):
    """
    Returns a generator that can be used to generate partitions of an integer
    `n`.

    Explanation
    ===========

    A partition of `n` is a set of positive integers which add up to `n`. For
    example, partitions of 3 are 3, 1 + 2, 1 + 1 + 1. A partition is returned
    as a tuple. If ``k`` equals None, then all possible partitions are returned
    irrespective of their size, otherwise only the partitions of size ``k`` are
    returned. If the ``zero`` parameter is set to True then a suitable
    number of zeros are added at the end of every partition of size less than
    ``k``.

    ``zero`` parameter is considered only if ``k`` is not None. When the
    partitions are over, the last `next()` call throws the ``StopIteration``
    exception, so this function should always be used inside a try - except
    block.

    Details
    =======

    ``partition(n, k)``: Here ``n`` is a positive integer and ``k`` is the size
    of the partition which is also positive integer.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import partition
    >>> f = partition(5)
    >>> next(f)
    (1, 1, 1, 1, 1)
    >>> next(f)
    (1, 1, 1, 2)
    >>> g = partition(5, 3)
    >>> next(g)
    (1, 1, 3)
    >>> next(g)
    (1, 2, 2)
    >>> g = partition(5, 3, zeros=True)
    >>> next(g)
    (0, 0, 5)

    """
    if not zeros or k is None:
        for i in ordered_partitions(n, k):
            yield tuple(i)
    else:
        for m in range(1, k + 1):
            for i in ordered_partitions(n, m):
                i = tuple(i)
                yield (0,)*(k - len(i)) + i


def prime_as_sum_of_two_squares(p):
    """
    Represent a prime `p` as a unique sum of two squares; this can
    only be done if the prime is congruent to 1 mod 4.

    Parameters
    ==========

    p : Integer
        A prime that is congruent to 1 mod 4

    Returns
    =======

    (int, int) | None : Pair of positive integers ``(x, y)`` satisfying ``x**2 + y**2 = p``.
                        None if ``p`` is not congruent to 1 mod 4.

    Raises
    ======

    ValueError
        If ``p`` is not prime number

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares
    >>> prime_as_sum_of_two_squares(7)  # can't be done
    >>> prime_as_sum_of_two_squares(5)
    (1, 2)

    Reference
    =========

    .. [1] Representing a number as a sum of four squares, [online],
           Available: https://schorn.ch/lagrange.html

    See Also
    ========

    sum_of_squares

    """
    p = as_int(p)
    if p % 4 != 1:
        return
    if not isprime(p):
        raise ValueError("p should be a prime number")

    if p % 8 == 5:
        # Legendre symbol (2/p) == -1 if p % 8 in [3, 5]
        b = 2
    elif p % 12 == 5:
        # Legendre symbol (3/p) == -1 if p % 12 in [5, 7]
        b = 3
    elif p % 5 in [2, 3]:
        # Legendre symbol (5/p) == -1 if p % 5 in [2, 3]
        b = 5
    else:
        b = 7
        while jacobi(b, p) == 1:
            b = nextprime(b)

    b = pow(b, p >> 2, p)
    a = p
    while b**2 > p:
        a, b = b, a % b
    return (int(a % b), int(b))  # convert from long


def sum_of_three_squares(n):
    r"""
    Returns a 3-tuple $(a, b, c)$ such that $a^2 + b^2 + c^2 = n$ and
    $a, b, c \geq 0$.

    Returns None if $n = 4^a(8m + 7)$ for some `a, m \in \mathbb{Z}`. See
    [1]_ for more details.

    Parameters
    ==========

    n : Integer
        non-negative integer

    Returns
    =======

    (int, int, int) | None : 3-tuple non-negative integers ``(a, b, c)`` satisfying ``a**2 + b**2 + c**2 = n``.
                             a,b,c are sorted in ascending order. ``None`` if no such ``(a,b,c)``.

    Raises
    ======

    ValueError
        If ``n`` is a negative integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_three_squares
    >>> sum_of_three_squares(44542)
    (18, 37, 207)

    References
    ==========

    .. [1] Representing a number as a sum of three squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    power_representation :
        ``sum_of_three_squares(n)`` is one of the solutions output by ``power_representation(n, 2, 3, zeros=True)``

    """
    # https://math.stackexchange.com/questions/483101/rabin-and-shallit-algorithm/651425#651425
    # discusses these numbers (except for 1, 2, 3) as the exceptions of H&L's conjecture that
    # Every sufficiently large number n is either a square or the sum of a prime and a square.
    special = {1: (0, 0, 1), 2: (0, 1, 1), 3: (1, 1, 1), 10: (0, 1, 3), 34: (3, 3, 4),
               58: (0, 3, 7), 85: (0, 6, 7), 130: (0, 3, 11), 214: (3, 6, 13), 226: (8, 9, 9),
               370: (8, 9, 15), 526: (6, 7, 21), 706: (15, 15, 16), 730: (0, 1, 27),
               1414: (6, 17, 33), 1906: (13, 21, 36), 2986: (21, 32, 39), 9634: (56, 57, 57)}
    n = as_int(n)
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    if n == 0:
        return (0, 0, 0)
    n, v = remove(n, 4)
    v = 1 << v
    if n % 8 == 7:
        return
    if n in special:
        return tuple([v*i for i in special[n]])

    s, _exact = integer_nthroot(n, 2)
    if _exact:
        return (0, 0, v*s)
    if n % 8 == 3:
        if not s % 2:
            s -= 1
        for x in range(s, -1, -2):
            N = (n - x**2) // 2
            if isprime(N):
                # n % 8 == 3 and x % 2 == 1 => N % 4 == 1
                y, z = prime_as_sum_of_two_squares(N)
                return tuple(sorted([v*x, v*(y + z), v*abs(y - z)]))
        # We will never reach this point because there must be a solution.
        assert False

    # assert n % 4 in [1, 2]
    if not((n % 2) ^ (s % 2)):
        s -= 1
    for x in range(s, -1, -2):
        N = n - x**2
        if isprime(N):
            # assert N % 4 == 1
            y, z = prime_as_sum_of_two_squares(N)
            return tuple(sorted([v*x, v*y, v*z]))
    # We will never reach this point because there must be a solution.
    assert False


def sum_of_four_squares(n):
    r"""
    Returns a 4-tuple `(a, b, c, d)` such that `a^2 + b^2 + c^2 + d^2 = n`.
    Here `a, b, c, d \geq 0`.

    Parameters
    ==========

    n : Integer
        non-negative integer

    Returns
    =======

    (int, int, int, int) : 4-tuple non-negative integers ``(a, b, c, d)`` satisfying ``a**2 + b**2 + c**2 + d**2 = n``.
                           a,b,c,d are sorted in ascending order.

    Raises
    ======

    ValueError
        If ``n`` is a negative integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_four_squares
    >>> sum_of_four_squares(3456)
    (8, 8, 32, 48)
    >>> sum_of_four_squares(1294585930293)
    (0, 1234, 2161, 1137796)

    References
    ==========

    .. [1] Representing a number as a sum of four squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    power_representation :
        ``sum_of_four_squares(n)`` is one of the solutions output by ``power_representation(n, 2, 4, zeros=True)``

    """
    n = as_int(n)
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    if n == 0:
        return (0, 0, 0, 0)
    # remove factors of 4 since a solution in terms of 3 squares is
    # going to be returned; this is also done in sum_of_three_squares,
    # but it needs to be done here to select d
    n, v = remove(n, 4)
    v = 1 << v
    if n % 8 == 7:
        d = 2
        n = n - 4
    elif n % 8 in (2, 6):
        d = 1
        n = n - 1
    else:
        d = 0
    x, y, z = sum_of_three_squares(n)  # sorted
    return tuple(sorted([v*d, v*x, v*y, v*z]))


def power_representation(n, p, k, zeros=False):
    r"""
    Returns a generator for finding k-tuples of integers,
    `(n_{1}, n_{2}, . . . n_{k})`, such that
    `n = n_{1}^p + n_{2}^p + . . . n_{k}^p`.

    Usage
    =====

    ``power_representation(n, p, k, zeros)``: Represent non-negative number
    ``n`` as a sum of ``k`` ``p``\ th powers. If ``zeros`` is true, then the
    solutions is allowed to contain zeros.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import power_representation

    Represent 1729 as a sum of two cubes:

    >>> f = power_representation(1729, 3, 2)
    >>> next(f)
    (9, 10)
    >>> next(f)
    (1, 12)

    If the flag `zeros` is True, the solution may contain tuples with
    zeros; any such solutions will be generated after the solutions
    without zeros:

    >>> list(power_representation(125, 2, 3, zeros=True))
    [(5, 6, 8), (3, 4, 10), (0, 5, 10), (0, 2, 11)]

    For even `p` the `permute_sign` function can be used to get all
    signed values:

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12)]

    All possible signed permutations can also be obtained:

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12), (12, 1), (-12, 1), (12, -1), (-12, -1)]
    """
    n, p, k = [as_int(i) for i in (n, p, k)]

    if n < 0:
        if p % 2:
            for t in power_representation(-n, p, k, zeros):
                yield tuple(-i for i in t)
        return

    if p < 1 or k < 1:
        raise ValueError(filldedent('''
    Expecting positive integers for `(p, k)`, but got `(%s, %s)`'''
    % (p, k)))

    if n == 0:
        if zeros:
            yield (0,)*k
        return

    if k == 1:
        if p == 1:
            yield (n,)
        elif n == 1:
            yield (1,)
        else:
            be = perfect_power(n)
            if be:
                b, e = be
                d, r = divmod(e, p)
                if not r:
                    yield (b**d,)
        return

    if p == 1:
        yield from partition(n, k, zeros=zeros)
        return

    if p == 2:
        if k == 3:
            n, v = remove(n, 4)
            if v:
                v = 1 << v
                for t in power_representation(n, p, k, zeros):
                    yield tuple(i*v for i in t)
                return
        feasible = _can_do_sum_of_squares(n, k)
        if not feasible:
            return
        if not zeros:
            if n > 33 and k >= 5 and k <= n and n - k in (
                13, 10, 7, 5, 4, 2, 1):
                '''Todd G. Will, "When Is n^2 a Sum of k Squares?", [online].
                Available: https://www.maa.org/sites/default/files/Will-MMz-201037918.pdf'''
                return
            # quick tests since feasibility includes the possibility of 0
            if k == 4 and (n in (1, 3, 5, 9, 11, 17, 29, 41) or remove(n, 4)[0] in (2, 6, 14)):
                # A000534
                return
            if k == 3 and n in (1, 2, 5, 10, 13, 25, 37, 58, 85, 130):  # or n = some number >= 5*10**10
                # A051952
                return
        if feasible is not True:  # it's prime and k == 2
            yield prime_as_sum_of_two_squares(n)
            return

    if k == 2 and p > 2:
        be = perfect_power(n)
        if be and be[1] % p == 0:
            return  # Fermat: a**n + b**n = c**n has no solution for n > 2

    if n >= k:
        a = integer_nthroot(n - (k - 1), p)[0]
        for t in pow_rep_recursive(a, k, n, [], p):
            yield tuple(reversed(t))

    if zeros:
        a = integer_nthroot(n, p)[0]
        for i in range(1, k):
            for t in pow_rep_recursive(a, i, n, [], p):
                yield tuple(reversed(t + (0,)*(k - i)))


sum_of_powers = power_representation


def pow_rep_recursive(n_i, k, n_remaining, terms, p):
    # Invalid arguments
    if n_i <= 0 or k <= 0:
        return

    # No solutions may exist
    if n_remaining < k:
        return
    if k * pow(n_i, p) < n_remaining:
        return

    if k == 0 and n_remaining == 0:
        yield tuple(terms)

    elif k == 1:
        # next_term^p must equal to n_remaining
        next_term, exact = integer_nthroot(n_remaining, p)
        if exact and next_term <= n_i:
            yield tuple(terms + [next_term])
        return

    else:
        # TODO: Fall back to diop_DN when k = 2
        if n_i >= 1 and k > 0:
            for next_term in range(1, n_i + 1):
                residual = n_remaining - pow(next_term, p)
                if residual < 0:
                    break
                yield from pow_rep_recursive(next_term, k - 1, residual, terms + [next_term], p)


def sum_of_squares(n, k, zeros=False):
    """Return a generator that yields the k-tuples of nonnegative
    values, the squares of which sum to n. If zeros is False (default)
    then the solution will not contain zeros. The nonnegative
    elements of a tuple are sorted.

    * If k == 1 and n is square, (n,) is returned.

    * If k == 2 then n can only be written as a sum of squares if
      every prime in the factorization of n that has the form
      4*k + 3 has an even multiplicity. If n is prime then
      it can only be written as a sum of two squares if it is
      in the form 4*k + 1.

    * if k == 3 then n can be written as a sum of squares if it does
      not have the form 4**m*(8*k + 7).

    * all integers can be written as the sum of 4 squares.

    * if k > 4 then n can be partitioned and each partition can
      be written as a sum of 4 squares; if n is not evenly divisible
      by 4 then n can be written as a sum of squares only if the
      an additional partition can be written as sum of squares.
      For example, if k = 6 then n is partitioned into two parts,
      the first being written as a sum of 4 squares and the second
      being written as a sum of 2 squares -- which can only be
      done if the condition above for k = 2 can be met, so this will
      automatically reject certain partitions of n.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_squares
    >>> list(sum_of_squares(25, 2))
    [(3, 4)]
    >>> list(sum_of_squares(25, 2, True))
    [(3, 4), (0, 5)]
    >>> list(sum_of_squares(25, 4))
    [(1, 2, 2, 4)]

    See Also
    ========

    sympy.utilities.iterables.signed_permutations
    """
    yield from power_representation(n, 2, k, zeros)


def _can_do_sum_of_squares(n, k):
    """Return True if n can be written as the sum of k squares,
    False if it cannot, or 1 if ``k == 2`` and ``n`` is prime (in which
    case it *can* be written as a sum of two squares). A False
    is returned only if it cannot be written as ``k``-squares, even
    if 0s are allowed.
    """
    if k < 1:
        return False
    if n < 0:
        return False
    if n == 0:
        return True
    if k == 1:
        return is_square(n)
    if k == 2:
        if n in (1, 2):
            return True
        if isprime(n):
            if n % 4 == 1:
                return 1  # signal that it was prime
            return False
        # n is a composite number
        # we can proceed iff no prime factor in the form 4*k + 3
        # has an odd multiplicity
        return all(p % 4 !=3 or m % 2 == 0 for p, m in factorint(n).items())
    if k == 3:
        return remove(n, 4)[0] % 8 != 7
    # every number can be written as a sum of 4 squares; for k > 4 partitions
    # can be 0
    return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\page.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Page
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime


class FrameId(str):
    '''
    Unique frame identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> FrameId:
        return cls(json)

    def __repr__(self):
        return 'FrameId({})'.format(super().__repr__())


class AdFrameType(enum.Enum):
    '''
    Indicates whether a frame has been identified as an ad.
    '''
    NONE = "none"
    CHILD = "child"
    ROOT = "root"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AdFrameExplanation(enum.Enum):
    PARENT_IS_AD = "ParentIsAd"
    CREATED_BY_AD_SCRIPT = "CreatedByAdScript"
    MATCHED_BLOCKING_RULE = "MatchedBlockingRule"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AdFrameStatus:
    '''
    Indicates whether a frame has been identified as an ad and why.
    '''
    ad_frame_type: AdFrameType

    explanations: typing.Optional[typing.List[AdFrameExplanation]] = None

    def to_json(self):
        json = dict()
        json['adFrameType'] = self.ad_frame_type.to_json()
        if self.explanations is not None:
            json['explanations'] = [i.to_json() for i in self.explanations]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            ad_frame_type=AdFrameType.from_json(json['adFrameType']),
            explanations=[AdFrameExplanation.from_json(i) for i in json['explanations']] if 'explanations' in json else None,
        )


@dataclass
class AdScriptId:
    '''
    Identifies the bottom-most script which caused the frame to be labelled
    as an ad.
    '''
    #: Script Id of the bottom-most script which caused the frame to be labelled
    #: as an ad.
    script_id: runtime.ScriptId

    #: Id of adScriptId's debugger.
    debugger_id: runtime.UniqueDebuggerId

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['debuggerId'] = self.debugger_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            debugger_id=runtime.UniqueDebuggerId.from_json(json['debuggerId']),
        )


class SecureContextType(enum.Enum):
    '''
    Indicates whether the frame is a secure context and why it is the case.
    '''
    SECURE = "Secure"
    SECURE_LOCALHOST = "SecureLocalhost"
    INSECURE_SCHEME = "InsecureScheme"
    INSECURE_ANCESTOR = "InsecureAncestor"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CrossOriginIsolatedContextType(enum.Enum):
    '''
    Indicates whether the frame is cross-origin isolated and why it is the case.
    '''
    ISOLATED = "Isolated"
    NOT_ISOLATED = "NotIsolated"
    NOT_ISOLATED_FEATURE_DISABLED = "NotIsolatedFeatureDisabled"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class GatedAPIFeatures(enum.Enum):
    SHARED_ARRAY_BUFFERS = "SharedArrayBuffers"
    SHARED_ARRAY_BUFFERS_TRANSFER_ALLOWED = "SharedArrayBuffersTransferAllowed"
    PERFORMANCE_MEASURE_MEMORY = "PerformanceMeasureMemory"
    PERFORMANCE_PROFILE = "PerformanceProfile"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PermissionsPolicyFeature(enum.Enum):
    '''
    All Permissions Policy features. This enum should match the one defined
    in services/network/public/cpp/permissions_policy/permissions_policy_features.json5.
    LINT.IfChange(PermissionsPolicyFeature)
    '''
    ACCELEROMETER = "accelerometer"
    ALL_SCREENS_CAPTURE = "all-screens-capture"
    AMBIENT_LIGHT_SENSOR = "ambient-light-sensor"
    ATTRIBUTION_REPORTING = "attribution-reporting"
    AUTOPLAY = "autoplay"
    BLUETOOTH = "bluetooth"
    BROWSING_TOPICS = "browsing-topics"
    CAMERA = "camera"
    CAPTURED_SURFACE_CONTROL = "captured-surface-control"
    CH_DPR = "ch-dpr"
    CH_DEVICE_MEMORY = "ch-device-memory"
    CH_DOWNLINK = "ch-downlink"
    CH_ECT = "ch-ect"
    CH_PREFERS_COLOR_SCHEME = "ch-prefers-color-scheme"
    CH_PREFERS_REDUCED_MOTION = "ch-prefers-reduced-motion"
    CH_PREFERS_REDUCED_TRANSPARENCY = "ch-prefers-reduced-transparency"
    CH_RTT = "ch-rtt"
    CH_SAVE_DATA = "ch-save-data"
    CH_UA = "ch-ua"
    CH_UA_ARCH = "ch-ua-arch"
    CH_UA_BITNESS = "ch-ua-bitness"
    CH_UA_HIGH_ENTROPY_VALUES = "ch-ua-high-entropy-values"
    CH_UA_PLATFORM = "ch-ua-platform"
    CH_UA_MODEL = "ch-ua-model"
    CH_UA_MOBILE = "ch-ua-mobile"
    CH_UA_FORM_FACTORS = "ch-ua-form-factors"
    CH_UA_FULL_VERSION = "ch-ua-full-version"
    CH_UA_FULL_VERSION_LIST = "ch-ua-full-version-list"
    CH_UA_PLATFORM_VERSION = "ch-ua-platform-version"
    CH_UA_WOW64 = "ch-ua-wow64"
    CH_VIEWPORT_HEIGHT = "ch-viewport-height"
    CH_VIEWPORT_WIDTH = "ch-viewport-width"
    CH_WIDTH = "ch-width"
    CLIPBOARD_READ = "clipboard-read"
    CLIPBOARD_WRITE = "clipboard-write"
    COMPUTE_PRESSURE = "compute-pressure"
    CONTROLLED_FRAME = "controlled-frame"
    CROSS_ORIGIN_ISOLATED = "cross-origin-isolated"
    DEFERRED_FETCH = "deferred-fetch"
    DEFERRED_FETCH_MINIMAL = "deferred-fetch-minimal"
    DEVICE_ATTRIBUTES = "device-attributes"
    DIGITAL_CREDENTIALS_GET = "digital-credentials-get"
    DIRECT_SOCKETS = "direct-sockets"
    DIRECT_SOCKETS_PRIVATE = "direct-sockets-private"
    DISPLAY_CAPTURE = "display-capture"
    DOCUMENT_DOMAIN = "document-domain"
    ENCRYPTED_MEDIA = "encrypted-media"
    EXECUTION_WHILE_OUT_OF_VIEWPORT = "execution-while-out-of-viewport"
    EXECUTION_WHILE_NOT_RENDERED = "execution-while-not-rendered"
    FENCED_UNPARTITIONED_STORAGE_READ = "fenced-unpartitioned-storage-read"
    FOCUS_WITHOUT_USER_ACTIVATION = "focus-without-user-activation"
    FULLSCREEN = "fullscreen"
    FROBULATE = "frobulate"
    GAMEPAD = "gamepad"
    GEOLOCATION = "geolocation"
    GYROSCOPE = "gyroscope"
    HID = "hid"
    IDENTITY_CREDENTIALS_GET = "identity-credentials-get"
    IDLE_DETECTION = "idle-detection"
    INTEREST_COHORT = "interest-cohort"
    JOIN_AD_INTEREST_GROUP = "join-ad-interest-group"
    KEYBOARD_MAP = "keyboard-map"
    LANGUAGE_DETECTOR = "language-detector"
    LOCAL_FONTS = "local-fonts"
    LOCAL_NETWORK_ACCESS = "local-network-access"
    MAGNETOMETER = "magnetometer"
    MEDIA_PLAYBACK_WHILE_NOT_VISIBLE = "media-playback-while-not-visible"
    MICROPHONE = "microphone"
    MIDI = "midi"
    OTP_CREDENTIALS = "otp-credentials"
    PAYMENT = "payment"
    PICTURE_IN_PICTURE = "picture-in-picture"
    POPINS = "popins"
    PRIVATE_AGGREGATION = "private-aggregation"
    PRIVATE_STATE_TOKEN_ISSUANCE = "private-state-token-issuance"
    PRIVATE_STATE_TOKEN_REDEMPTION = "private-state-token-redemption"
    PUBLICKEY_CREDENTIALS_CREATE = "publickey-credentials-create"
    PUBLICKEY_CREDENTIALS_GET = "publickey-credentials-get"
    RECORD_AD_AUCTION_EVENTS = "record-ad-auction-events"
    REWRITER = "rewriter"
    RUN_AD_AUCTION = "run-ad-auction"
    SCREEN_WAKE_LOCK = "screen-wake-lock"
    SERIAL = "serial"
    SHARED_AUTOFILL = "shared-autofill"
    SHARED_STORAGE = "shared-storage"
    SHARED_STORAGE_SELECT_URL = "shared-storage-select-url"
    SMART_CARD = "smart-card"
    SPEAKER_SELECTION = "speaker-selection"
    STORAGE_ACCESS = "storage-access"
    SUB_APPS = "sub-apps"
    SUMMARIZER = "summarizer"
    SYNC_XHR = "sync-xhr"
    TRANSLATOR = "translator"
    UNLOAD = "unload"
    USB = "usb"
    USB_UNRESTRICTED = "usb-unrestricted"
    VERTICAL_SCROLL = "vertical-scroll"
    WEB_APP_INSTALLATION = "web-app-installation"
    WEB_PRINTING = "web-printing"
    WEB_SHARE = "web-share"
    WINDOW_MANAGEMENT = "window-management"
    WRITER = "writer"
    XR_SPATIAL_TRACKING = "xr-spatial-tracking"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PermissionsPolicyBlockReason(enum.Enum):
    '''
    Reason for a permissions policy feature to be disabled.
    '''
    HEADER = "Header"
    IFRAME_ATTRIBUTE = "IframeAttribute"
    IN_FENCED_FRAME_TREE = "InFencedFrameTree"
    IN_ISOLATED_APP = "InIsolatedApp"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PermissionsPolicyBlockLocator:
    frame_id: FrameId

    block_reason: PermissionsPolicyBlockReason

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        json['blockReason'] = self.block_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            block_reason=PermissionsPolicyBlockReason.from_json(json['blockReason']),
        )


@dataclass
class PermissionsPolicyFeatureState:
    feature: PermissionsPolicyFeature

    allowed: bool

    locator: typing.Optional[PermissionsPolicyBlockLocator] = None

    def to_json(self):
        json = dict()
        json['feature'] = self.feature.to_json()
        json['allowed'] = self.allowed
        if self.locator is not None:
            json['locator'] = self.locator.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            feature=PermissionsPolicyFeature.from_json(json['feature']),
            allowed=bool(json['allowed']),
            locator=PermissionsPolicyBlockLocator.from_json(json['locator']) if 'locator' in json else None,
        )


class OriginTrialTokenStatus(enum.Enum):
    '''
    Origin Trial(https://www.chromium.org/blink/origin-trials) support.
    Status for an Origin Trial token.
    '''
    SUCCESS = "Success"
    NOT_SUPPORTED = "NotSupported"
    INSECURE = "Insecure"
    EXPIRED = "Expired"
    WRONG_ORIGIN = "WrongOrigin"
    INVALID_SIGNATURE = "InvalidSignature"
    MALFORMED = "Malformed"
    WRONG_VERSION = "WrongVersion"
    FEATURE_DISABLED = "FeatureDisabled"
    TOKEN_DISABLED = "TokenDisabled"
    FEATURE_DISABLED_FOR_USER = "FeatureDisabledForUser"
    UNKNOWN_TRIAL = "UnknownTrial"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class OriginTrialStatus(enum.Enum):
    '''
    Status for an Origin Trial.
    '''
    ENABLED = "Enabled"
    VALID_TOKEN_NOT_PROVIDED = "ValidTokenNotProvided"
    OS_NOT_SUPPORTED = "OSNotSupported"
    TRIAL_NOT_ALLOWED = "TrialNotAllowed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class OriginTrialUsageRestriction(enum.Enum):
    NONE = "None"
    SUBSET = "Subset"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class OriginTrialToken:
    origin: str

    match_sub_domains: bool

    trial_name: str

    expiry_time: network.TimeSinceEpoch

    is_third_party: bool

    usage_restriction: OriginTrialUsageRestriction

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['matchSubDomains'] = self.match_sub_domains
        json['trialName'] = self.trial_name
        json['expiryTime'] = self.expiry_time.to_json()
        json['isThirdParty'] = self.is_third_party
        json['usageRestriction'] = self.usage_restriction.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            origin=str(json['origin']),
            match_sub_domains=bool(json['matchSubDomains']),
            trial_name=str(json['trialName']),
            expiry_time=network.TimeSinceEpoch.from_json(json['expiryTime']),
            is_third_party=bool(json['isThirdParty']),
            usage_restriction=OriginTrialUsageRestriction.from_json(json['usageRestriction']),
        )


@dataclass
class OriginTrialTokenWithStatus:
    raw_token_text: str

    status: OriginTrialTokenStatus

    #: ``parsedToken`` is present only when the token is extractable and
    #: parsable.
    parsed_token: typing.Optional[OriginTrialToken] = None

    def to_json(self):
        json = dict()
        json['rawTokenText'] = self.raw_token_text
        json['status'] = self.status.to_json()
        if self.parsed_token is not None:
            json['parsedToken'] = self.parsed_token.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            raw_token_text=str(json['rawTokenText']),
            status=OriginTrialTokenStatus.from_json(json['status']),
            parsed_token=OriginTrialToken.from_json(json['parsedToken']) if 'parsedToken' in json else None,
        )


@dataclass
class OriginTrial:
    trial_name: str

    status: OriginTrialStatus

    tokens_with_status: typing.List[OriginTrialTokenWithStatus]

    def to_json(self):
        json = dict()
        json['trialName'] = self.trial_name
        json['status'] = self.status.to_json()
        json['tokensWithStatus'] = [i.to_json() for i in self.tokens_with_status]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            trial_name=str(json['trialName']),
            status=OriginTrialStatus.from_json(json['status']),
            tokens_with_status=[OriginTrialTokenWithStatus.from_json(i) for i in json['tokensWithStatus']],
        )


@dataclass
class SecurityOriginDetails:
    '''
    Additional information about the frame document's security origin.
    '''
    #: Indicates whether the frame document's security origin is one
    #: of the local hostnames (e.g. "localhost") or IP addresses (IPv4
    #: 127.0.0.0/8 or IPv6 ::1).
    is_localhost: bool

    def to_json(self):
        json = dict()
        json['isLocalhost'] = self.is_localhost
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            is_localhost=bool(json['isLocalhost']),
        )


@dataclass
class Frame:
    '''
    Information about the Frame on the page.
    '''
    #: Frame unique identifier.
    id_: FrameId

    #: Identifier of the loader associated with this frame.
    loader_id: network.LoaderId

    #: Frame document's URL without fragment.
    url: str

    #: Frame document's registered domain, taking the public suffixes list into account.
    #: Extracted from the Frame's url.
    #: Example URLs: http://www.google.com/file.html -> "google.com"
    #:               http://a.b.co.uk/file.html      -> "b.co.uk"
    domain_and_registry: str

    #: Frame document's security origin.
    security_origin: str

    #: Frame document's mimeType as determined by the browser.
    mime_type: str

    #: Indicates whether the main document is a secure context and explains why that is the case.
    secure_context_type: SecureContextType

    #: Indicates whether this is a cross origin isolated context.
    cross_origin_isolated_context_type: CrossOriginIsolatedContextType

    #: Indicated which gated APIs / features are available.
    gated_api_features: typing.List[GatedAPIFeatures]

    #: Parent frame identifier.
    parent_id: typing.Optional[FrameId] = None

    #: Frame's name as specified in the tag.
    name: typing.Optional[str] = None

    #: Frame document's URL fragment including the '#'.
    url_fragment: typing.Optional[str] = None

    #: Additional details about the frame document's security origin.
    security_origin_details: typing.Optional[SecurityOriginDetails] = None

    #: If the frame failed to load, this contains the URL that could not be loaded. Note that unlike url above, this URL may contain a fragment.
    unreachable_url: typing.Optional[str] = None

    #: Indicates whether this frame was tagged as an ad and why.
    ad_frame_status: typing.Optional[AdFrameStatus] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['loaderId'] = self.loader_id.to_json()
        json['url'] = self.url
        json['domainAndRegistry'] = self.domain_and_registry
        json['securityOrigin'] = self.security_origin
        json['mimeType'] = self.mime_type
        json['secureContextType'] = self.secure_context_type.to_json()
        json['crossOriginIsolatedContextType'] = self.cross_origin_isolated_context_type.to_json()
        json['gatedAPIFeatures'] = [i.to_json() for i in self.gated_api_features]
        if self.parent_id is not None:
            json['parentId'] = self.parent_id.to_json()
        if self.name is not None:
            json['name'] = self.name
        if self.url_fragment is not None:
            json['urlFragment'] = self.url_fragment
        if self.security_origin_details is not None:
            json['securityOriginDetails'] = self.security_origin_details.to_json()
        if self.unreachable_url is not None:
            json['unreachableUrl'] = self.unreachable_url
        if self.ad_frame_status is not None:
            json['adFrameStatus'] = self.ad_frame_status.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=FrameId.from_json(json['id']),
            loader_id=network.LoaderId.from_json(json['loaderId']),
            url=str(json['url']),
            domain_and_registry=str(json['domainAndRegistry']),
            security_origin=str(json['securityOrigin']),
            mime_type=str(json['mimeType']),
            secure_context_type=SecureContextType.from_json(json['secureContextType']),
            cross_origin_isolated_context_type=CrossOriginIsolatedContextType.from_json(json['crossOriginIsolatedContextType']),
            gated_api_features=[GatedAPIFeatures.from_json(i) for i in json['gatedAPIFeatures']],
            parent_id=FrameId.from_json(json['parentId']) if 'parentId' in json else None,
            name=str(json['name']) if 'name' in json else None,
            url_fragment=str(json['urlFragment']) if 'urlFragment' in json else None,
            security_origin_details=SecurityOriginDetails.from_json(json['securityOriginDetails']) if 'securityOriginDetails' in json else None,
            unreachable_url=str(json['unreachableUrl']) if 'unreachableUrl' in json else None,
            ad_frame_status=AdFrameStatus.from_json(json['adFrameStatus']) if 'adFrameStatus' in json else None,
        )


@dataclass
class FrameResource:
    '''
    Information about the Resource on the page.
    '''
    #: Resource URL.
    url: str

    #: Type of this resource.
    type_: network.ResourceType

    #: Resource mimeType as determined by the browser.
    mime_type: str

    #: last-modified timestamp as reported by server.
    last_modified: typing.Optional[network.TimeSinceEpoch] = None

    #: Resource content size.
    content_size: typing.Optional[float] = None

    #: True if the resource failed to load.
    failed: typing.Optional[bool] = None

    #: True if the resource was canceled during loading.
    canceled: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['type'] = self.type_.to_json()
        json['mimeType'] = self.mime_type
        if self.last_modified is not None:
            json['lastModified'] = self.last_modified.to_json()
        if self.content_size is not None:
            json['contentSize'] = self.content_size
        if self.failed is not None:
            json['failed'] = self.failed
        if self.canceled is not None:
            json['canceled'] = self.canceled
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            type_=network.ResourceType.from_json(json['type']),
            mime_type=str(json['mimeType']),
            last_modified=network.TimeSinceEpoch.from_json(json['lastModified']) if 'lastModified' in json else None,
            content_size=float(json['contentSize']) if 'contentSize' in json else None,
            failed=bool(json['failed']) if 'failed' in json else None,
            canceled=bool(json['canceled']) if 'canceled' in json else None,
        )


@dataclass
class FrameResourceTree:
    '''
    Information about the Frame hierarchy along with their cached resources.
    '''
    #: Frame information for this tree item.
    frame: Frame

    #: Information about frame resources.
    resources: typing.List[FrameResource]

    #: Child frames.
    child_frames: typing.Optional[typing.List[FrameResourceTree]] = None

    def to_json(self):
        json = dict()
        json['frame'] = self.frame.to_json()
        json['resources'] = [i.to_json() for i in self.resources]
        if self.child_frames is not None:
            json['childFrames'] = [i.to_json() for i in self.child_frames]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            frame=Frame.from_json(json['frame']),
            resources=[FrameResource.from_json(i) for i in json['resources']],
            child_frames=[FrameResourceTree.from_json(i) for i in json['childFrames']] if 'childFrames' in json else None,
        )


@dataclass
class FrameTree:
    '''
    Information about the Frame hierarchy.
    '''
    #: Frame information for this tree item.
    frame: Frame

    #: Child frames.
    child_frames: typing.Optional[typing.List[FrameTree]] = None

    def to_json(self):
        json = dict()
        json['frame'] = self.frame.to_json()
        if self.child_frames is not None:
            json['childFrames'] = [i.to_json() for i in self.child_frames]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            frame=Frame.from_json(json['frame']),
            child_frames=[FrameTree.from_json(i) for i in json['childFrames']] if 'childFrames' in json else None,
        )


class ScriptIdentifier(str):
    '''
    Unique script identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ScriptIdentifier:
        return cls(json)

    def __repr__(self):
        return 'ScriptIdentifier({})'.format(super().__repr__())


class TransitionType(enum.Enum):
    '''
    Transition type.
    '''
    LINK = "link"
    TYPED = "typed"
    ADDRESS_BAR = "address_bar"
    AUTO_BOOKMARK = "auto_bookmark"
    AUTO_SUBFRAME = "auto_subframe"
    MANUAL_SUBFRAME = "manual_subframe"
    GENERATED = "generated"
    AUTO_TOPLEVEL = "auto_toplevel"
    FORM_SUBMIT = "form_submit"
    RELOAD = "reload"
    KEYWORD = "keyword"
    KEYWORD_GENERATED = "keyword_generated"
    OTHER = "other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class NavigationEntry:
    '''
    Navigation history entry.
    '''
    #: Unique id of the navigation history entry.
    id_: int

    #: URL of the navigation history entry.
    url: str

    #: URL that the user typed in the url bar.
    user_typed_url: str

    #: Title of the navigation history entry.
    title: str

    #: Transition type.
    transition_type: TransitionType

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        json['url'] = self.url
        json['userTypedURL'] = self.user_typed_url
        json['title'] = self.title
        json['transitionType'] = self.transition_type.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=int(json['id']),
            url=str(json['url']),
            user_typed_url=str(json['userTypedURL']),
            title=str(json['title']),
            transition_type=TransitionType.from_json(json['transitionType']),
        )


@dataclass
class ScreencastFrameMetadata:
    '''
    Screencast frame metadata.
    '''
    #: Top offset in DIP.
    offset_top: float

    #: Page scale factor.
    page_scale_factor: float

    #: Device screen width in DIP.
    device_width: float

    #: Device screen height in DIP.
    device_height: float

    #: Position of horizontal scroll in CSS pixels.
    scroll_offset_x: float

    #: Position of vertical scroll in CSS pixels.
    scroll_offset_y: float

    #: Frame swap timestamp.
    timestamp: typing.Optional[network.TimeSinceEpoch] = None

    def to_json(self):
        json = dict()
        json['offsetTop'] = self.offset_top
        json['pageScaleFactor'] = self.page_scale_factor
        json['deviceWidth'] = self.device_width
        json['deviceHeight'] = self.device_height
        json['scrollOffsetX'] = self.scroll_offset_x
        json['scrollOffsetY'] = self.scroll_offset_y
        if self.timestamp is not None:
            json['timestamp'] = self.timestamp.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            offset_top=float(json['offsetTop']),
            page_scale_factor=float(json['pageScaleFactor']),
            device_width=float(json['deviceWidth']),
            device_height=float(json['deviceHeight']),
            scroll_offset_x=float(json['scrollOffsetX']),
            scroll_offset_y=float(json['scrollOffsetY']),
            timestamp=network.TimeSinceEpoch.from_json(json['timestamp']) if 'timestamp' in json else None,
        )


class DialogType(enum.Enum):
    '''
    Javascript dialog type.
    '''
    ALERT = "alert"
    CONFIRM = "confirm"
    PROMPT = "prompt"
    BEFOREUNLOAD = "beforeunload"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AppManifestError:
    '''
    Error while paring app manifest.
    '''
    #: Error message.
    message: str

    #: If critical, this is a non-recoverable parse error.
    critical: int

    #: Error line.
    line: int

    #: Error column.
    column: int

    def to_json(self):
        json = dict()
        json['message'] = self.message
        json['critical'] = self.critical
        json['line'] = self.line
        json['column'] = self.column
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            message=str(json['message']),
            critical=int(json['critical']),
            line=int(json['line']),
            column=int(json['column']),
        )


@dataclass
class AppManifestParsedProperties:
    '''
    Parsed app manifest properties.
    '''
    #: Computed scope value
    scope: str

    def to_json(self):
        json = dict()
        json['scope'] = self.scope
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            scope=str(json['scope']),
        )


@dataclass
class LayoutViewport:
    '''
    Layout viewport position and dimensions.
    '''
    #: Horizontal offset relative to the document (CSS pixels).
    page_x: int

    #: Vertical offset relative to the document (CSS pixels).
    page_y: int

    #: Width (CSS pixels), excludes scrollbar if present.
    client_width: int

    #: Height (CSS pixels), excludes scrollbar if present.
    client_height: int

    def to_json(self):
        json = dict()
        json['pageX'] = self.page_x
        json['pageY'] = self.page_y
        json['clientWidth'] = self.client_width
        json['clientHeight'] = self.client_height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            page_x=int(json['pageX']),
            page_y=int(json['pageY']),
            client_width=int(json['clientWidth']),
            client_height=int(json['clientHeight']),
        )


@dataclass
class VisualViewport:
    '''
    Visual viewport position, dimensions, and scale.
    '''
    #: Horizontal offset relative to the layout viewport (CSS pixels).
    offset_x: float

    #: Vertical offset relative to the layout viewport (CSS pixels).
    offset_y: float

    #: Horizontal offset relative to the document (CSS pixels).
    page_x: float

    #: Vertical offset relative to the document (CSS pixels).
    page_y: float

    #: Width (CSS pixels), excludes scrollbar if present.
    client_width: float

    #: Height (CSS pixels), excludes scrollbar if present.
    client_height: float

    #: Scale relative to the ideal viewport (size at width=device-width).
    scale: float

    #: Page zoom factor (CSS to device independent pixels ratio).
    zoom: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['offsetX'] = self.offset_x
        json['offsetY'] = self.offset_y
        json['pageX'] = self.page_x
        json['pageY'] = self.page_y
        json['clientWidth'] = self.client_width
        json['clientHeight'] = self.client_height
        json['scale'] = self.scale
        if self.zoom is not None:
            json['zoom'] = self.zoom
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            offset_x=float(json['offsetX']),
            offset_y=float(json['offsetY']),
            page_x=float(json['pageX']),
            page_y=float(json['pageY']),
            client_width=float(json['clientWidth']),
            client_height=float(json['clientHeight']),
            scale=float(json['scale']),
            zoom=float(json['zoom']) if 'zoom' in json else None,
        )


@dataclass
class Viewport:
    '''
    Viewport for capturing screenshot.
    '''
    #: X offset in device independent pixels (dip).
    x: float

    #: Y offset in device independent pixels (dip).
    y: float

    #: Rectangle width in device independent pixels (dip).
    width: float

    #: Rectangle height in device independent pixels (dip).
    height: float

    #: Page scale factor.
    scale: float

    def to_json(self):
        json = dict()
        json['x'] = self.x
        json['y'] = self.y
        json['width'] = self.width
        json['height'] = self.height
        json['scale'] = self.scale
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            x=float(json['x']),
            y=float(json['y']),
            width=float(json['width']),
            height=float(json['height']),
            scale=float(json['scale']),
        )


@dataclass
class FontFamilies:
    '''
    Generic font families collection.
    '''
    #: The standard font-family.
    standard: typing.Optional[str] = None

    #: The fixed font-family.
    fixed: typing.Optional[str] = None

    #: The serif font-family.
    serif: typing.Optional[str] = None

    #: The sansSerif font-family.
    sans_serif: typing.Optional[str] = None

    #: The cursive font-family.
    cursive: typing.Optional[str] = None

    #: The fantasy font-family.
    fantasy: typing.Optional[str] = None

    #: The math font-family.
    math: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.standard is not None:
            json['standard'] = self.standard
        if self.fixed is not None:
            json['fixed'] = self.fixed
        if self.serif is not None:
            json['serif'] = self.serif
        if self.sans_serif is not None:
            json['sansSerif'] = self.sans_serif
        if self.cursive is not None:
            json['cursive'] = self.cursive
        if self.fantasy is not None:
            json['fantasy'] = self.fantasy
        if self.math is not None:
            json['math'] = self.math
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            standard=str(json['standard']) if 'standard' in json else None,
            fixed=str(json['fixed']) if 'fixed' in json else None,
            serif=str(json['serif']) if 'serif' in json else None,
            sans_serif=str(json['sansSerif']) if 'sansSerif' in json else None,
            cursive=str(json['cursive']) if 'cursive' in json else None,
            fantasy=str(json['fantasy']) if 'fantasy' in json else None,
            math=str(json['math']) if 'math' in json else None,
        )


@dataclass
class ScriptFontFamilies:
    '''
    Font families collection for a script.
    '''
    #: Name of the script which these font families are defined for.
    script: str

    #: Generic font families collection for the script.
    font_families: FontFamilies

    def to_json(self):
        json = dict()
        json['script'] = self.script
        json['fontFamilies'] = self.font_families.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script=str(json['script']),
            font_families=FontFamilies.from_json(json['fontFamilies']),
        )


@dataclass
class FontSizes:
    '''
    Default font sizes.
    '''
    #: Default standard font size.
    standard: typing.Optional[int] = None

    #: Default fixed font size.
    fixed: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        if self.standard is not None:
            json['standard'] = self.standard
        if self.fixed is not None:
            json['fixed'] = self.fixed
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            standard=int(json['standard']) if 'standard' in json else None,
            fixed=int(json['fixed']) if 'fixed' in json else None,
        )


class ClientNavigationReason(enum.Enum):
    ANCHOR_CLICK = "anchorClick"
    FORM_SUBMISSION_GET = "formSubmissionGet"
    FORM_SUBMISSION_POST = "formSubmissionPost"
    HTTP_HEADER_REFRESH = "httpHeaderRefresh"
    INITIAL_FRAME_NAVIGATION = "initialFrameNavigation"
    META_TAG_REFRESH = "metaTagRefresh"
    OTHER = "other"
    PAGE_BLOCK_INTERSTITIAL = "pageBlockInterstitial"
    RELOAD = "reload"
    SCRIPT_INITIATED = "scriptInitiated"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ClientNavigationDisposition(enum.Enum):
    CURRENT_TAB = "currentTab"
    NEW_TAB = "newTab"
    NEW_WINDOW = "newWindow"
    DOWNLOAD = "download"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class InstallabilityErrorArgument:
    #: Argument name (e.g. name:'minimum-icon-size-in-pixels').
    name: str

    #: Argument value (e.g. value:'64').
    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class InstallabilityError:
    '''
    The installability error
    '''
    #: The error id (e.g. 'manifest-missing-suitable-icon').
    error_id: str

    #: The list of error arguments (e.g. {name:'minimum-icon-size-in-pixels', value:'64'}).
    error_arguments: typing.List[InstallabilityErrorArgument]

    def to_json(self):
        json = dict()
        json['errorId'] = self.error_id
        json['errorArguments'] = [i.to_json() for i in self.error_arguments]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_id=str(json['errorId']),
            error_arguments=[InstallabilityErrorArgument.from_json(i) for i in json['errorArguments']],
        )


class ReferrerPolicy(enum.Enum):
    '''
    The referring-policy used for the navigation.
    '''
    NO_REFERRER = "noReferrer"
    NO_REFERRER_WHEN_DOWNGRADE = "noReferrerWhenDowngrade"
    ORIGIN = "origin"
    ORIGIN_WHEN_CROSS_ORIGIN = "originWhenCrossOrigin"
    SAME_ORIGIN = "sameOrigin"
    STRICT_ORIGIN = "strictOrigin"
    STRICT_ORIGIN_WHEN_CROSS_ORIGIN = "strictOriginWhenCrossOrigin"
    UNSAFE_URL = "unsafeUrl"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CompilationCacheParams:
    '''
    Per-script compilation cache parameters for ``Page.produceCompilationCache``
    '''
    #: The URL of the script to produce a compilation cache entry for.
    url: str

    #: A hint to the backend whether eager compilation is recommended.
    #: (the actual compilation mode used is upon backend discretion).
    eager: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.eager is not None:
            json['eager'] = self.eager
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            eager=bool(json['eager']) if 'eager' in json else None,
        )


@dataclass
class FileFilter:
    name: typing.Optional[str] = None

    accepts: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        if self.name is not None:
            json['name'] = self.name
        if self.accepts is not None:
            json['accepts'] = [i for i in self.accepts]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']) if 'name' in json else None,
            accepts=[str(i) for i in json['accepts']] if 'accepts' in json else None,
        )


@dataclass
class FileHandler:
    action: str

    name: str

    #: Won't repeat the enums, using string for easy comparison. Same as the
    #: other enums below.
    launch_type: str

    icons: typing.Optional[typing.List[ImageResource]] = None

    #: Mimic a map, name is the key, accepts is the value.
    accepts: typing.Optional[typing.List[FileFilter]] = None

    def to_json(self):
        json = dict()
        json['action'] = self.action
        json['name'] = self.name
        json['launchType'] = self.launch_type
        if self.icons is not None:
            json['icons'] = [i.to_json() for i in self.icons]
        if self.accepts is not None:
            json['accepts'] = [i.to_json() for i in self.accepts]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            action=str(json['action']),
            name=str(json['name']),
            launch_type=str(json['launchType']),
            icons=[ImageResource.from_json(i) for i in json['icons']] if 'icons' in json else None,
            accepts=[FileFilter.from_json(i) for i in json['accepts']] if 'accepts' in json else None,
        )


@dataclass
class ImageResource:
    '''
    The image definition used in both icon and screenshot.
    '''
    #: The src field in the definition, but changing to url in favor of
    #: consistency.
    url: str

    sizes: typing.Optional[str] = None

    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.sizes is not None:
            json['sizes'] = self.sizes
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            sizes=str(json['sizes']) if 'sizes' in json else None,
            type_=str(json['type']) if 'type' in json else None,
        )


@dataclass
class LaunchHandler:
    client_mode: str

    def to_json(self):
        json = dict()
        json['clientMode'] = self.client_mode
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            client_mode=str(json['clientMode']),
        )


@dataclass
class ProtocolHandler:
    protocol: str

    url: str

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol
        json['url'] = self.url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=str(json['protocol']),
            url=str(json['url']),
        )


@dataclass
class RelatedApplication:
    url: str

    id_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.id_ is not None:
            json['id'] = self.id_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            id_=str(json['id']) if 'id' in json else None,
        )


@dataclass
class ScopeExtension:
    #: Instead of using tuple, this field always returns the serialized string
    #: for easy understanding and comparison.
    origin: str

    has_origin_wildcard: bool

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['hasOriginWildcard'] = self.has_origin_wildcard
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            origin=str(json['origin']),
            has_origin_wildcard=bool(json['hasOriginWildcard']),
        )


@dataclass
class Screenshot:
    image: ImageResource

    form_factor: str

    label: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['image'] = self.image.to_json()
        json['formFactor'] = self.form_factor
        if self.label is not None:
            json['label'] = self.label
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            image=ImageResource.from_json(json['image']),
            form_factor=str(json['formFactor']),
            label=str(json['label']) if 'label' in json else None,
        )


@dataclass
class ShareTarget:
    action: str

    method: str

    enctype: str

    #: Embed the ShareTargetParams
    title: typing.Optional[str] = None

    text: typing.Optional[str] = None

    url: typing.Optional[str] = None

    files: typing.Optional[typing.List[FileFilter]] = None

    def to_json(self):
        json = dict()
        json['action'] = self.action
        json['method'] = self.method
        json['enctype'] = self.enctype
        if self.title is not None:
            json['title'] = self.title
        if self.text is not None:
            json['text'] = self.text
        if self.url is not None:
            json['url'] = self.url
        if self.files is not None:
            json['files'] = [i.to_json() for i in self.files]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            action=str(json['action']),
            method=str(json['method']),
            enctype=str(json['enctype']),
            title=str(json['title']) if 'title' in json else None,
            text=str(json['text']) if 'text' in json else None,
            url=str(json['url']) if 'url' in json else None,
            files=[FileFilter.from_json(i) for i in json['files']] if 'files' in json else None,
        )


@dataclass
class Shortcut:
    name: str

    url: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['url'] = self.url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            url=str(json['url']),
        )


@dataclass
class WebAppManifest:
    background_color: typing.Optional[str] = None

    #: The extra description provided by the manifest.
    description: typing.Optional[str] = None

    dir_: typing.Optional[str] = None

    display: typing.Optional[str] = None

    #: The overrided display mode controlled by the user.
    display_overrides: typing.Optional[typing.List[str]] = None

    #: The handlers to open files.
    file_handlers: typing.Optional[typing.List[FileHandler]] = None

    icons: typing.Optional[typing.List[ImageResource]] = None

    id_: typing.Optional[str] = None

    lang: typing.Optional[str] = None

    #: TODO(crbug.com/1231886): This field is non-standard and part of a Chrome
    #: experiment. See:
    #: https://github.com/WICG/web-app-launch/blob/main/launch_handler.md
    launch_handler: typing.Optional[LaunchHandler] = None

    name: typing.Optional[str] = None

    orientation: typing.Optional[str] = None

    prefer_related_applications: typing.Optional[bool] = None

    #: The handlers to open protocols.
    protocol_handlers: typing.Optional[typing.List[ProtocolHandler]] = None

    related_applications: typing.Optional[typing.List[RelatedApplication]] = None

    scope: typing.Optional[str] = None

    #: Non-standard, see
    #: https://github.com/WICG/manifest-incubations/blob/gh-pages/scope_extensions-explainer.md
    scope_extensions: typing.Optional[typing.List[ScopeExtension]] = None

    #: The screenshots used by chromium.
    screenshots: typing.Optional[typing.List[Screenshot]] = None

    share_target: typing.Optional[ShareTarget] = None

    short_name: typing.Optional[str] = None

    shortcuts: typing.Optional[typing.List[Shortcut]] = None

    start_url: typing.Optional[str] = None

    theme_color: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.background_color is not None:
            json['backgroundColor'] = self.background_color
        if self.description is not None:
            json['description'] = self.description
        if self.dir_ is not None:
            json['dir'] = self.dir_
        if self.display is not None:
            json['display'] = self.display
        if self.display_overrides is not None:
            json['displayOverrides'] = [i for i in self.display_overrides]
        if self.file_handlers is not None:
            json['fileHandlers'] = [i.to_json() for i in self.file_handlers]
        if self.icons is not None:
            json['icons'] = [i.to_json() for i in self.icons]
        if self.id_ is not None:
            json['id'] = self.id_
        if self.lang is not None:
            json['lang'] = self.lang
        if self.launch_handler is not None:
            json['launchHandler'] = self.launch_handler.to_json()
        if self.name is not None:
            json['name'] = self.name
        if self.orientation is not None:
            json['orientation'] = self.orientation
        if self.prefer_related_applications is not None:
            json['preferRelatedApplications'] = self.prefer_related_applications
        if self.protocol_handlers is not None:
            json['protocolHandlers'] = [i.to_json() for i in self.protocol_handlers]
        if self.related_applications is not None:
            json['relatedApplications'] = [i.to_json() for i in self.related_applications]
        if self.scope is not None:
            json['scope'] = self.scope
        if self.scope_extensions is not None:
            json['scopeExtensions'] = [i.to_json() for i in self.scope_extensions]
        if self.screenshots is not None:
            json['screenshots'] = [i.to_json() for i in self.screenshots]
        if self.share_target is not None:
            json['shareTarget'] = self.share_target.to_json()
        if self.short_name is not None:
            json['shortName'] = self.short_name
        if self.shortcuts is not None:
            json['shortcuts'] = [i.to_json() for i in self.shortcuts]
        if self.start_url is not None:
            json['startUrl'] = self.start_url
        if self.theme_color is not None:
            json['themeColor'] = self.theme_color
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            background_color=str(json['backgroundColor']) if 'backgroundColor' in json else None,
            description=str(json['description']) if 'description' in json else None,
            dir_=str(json['dir']) if 'dir' in json else None,
            display=str(json['display']) if 'display' in json else None,
            display_overrides=[str(i) for i in json['displayOverrides']] if 'displayOverrides' in json else None,
            file_handlers=[FileHandler.from_json(i) for i in json['fileHandlers']] if 'fileHandlers' in json else None,
            icons=[ImageResource.from_json(i) for i in json['icons']] if 'icons' in json else None,
            id_=str(json['id']) if 'id' in json else None,
            lang=str(json['lang']) if 'lang' in json else None,
            launch_handler=LaunchHandler.from_json(json['launchHandler']) if 'launchHandler' in json else None,
            name=str(json['name']) if 'name' in json else None,
            orientation=str(json['orientation']) if 'orientation' in json else None,
            prefer_related_applications=bool(json['preferRelatedApplications']) if 'preferRelatedApplications' in json else None,
            protocol_handlers=[ProtocolHandler.from_json(i) for i in json['protocolHandlers']] if 'protocolHandlers' in json else None,
            related_applications=[RelatedApplication.from_json(i) for i in json['relatedApplications']] if 'relatedApplications' in json else None,
            scope=str(json['scope']) if 'scope' in json else None,
            scope_extensions=[ScopeExtension.from_json(i) for i in json['scopeExtensions']] if 'scopeExtensions' in json else None,
            screenshots=[Screenshot.from_json(i) for i in json['screenshots']] if 'screenshots' in json else None,
            share_target=ShareTarget.from_json(json['shareTarget']) if 'shareTarget' in json else None,
            short_name=str(json['shortName']) if 'shortName' in json else None,
            shortcuts=[Shortcut.from_json(i) for i in json['shortcuts']] if 'shortcuts' in json else None,
            start_url=str(json['startUrl']) if 'startUrl' in json else None,
            theme_color=str(json['themeColor']) if 'themeColor' in json else None,
        )


class AutoResponseMode(enum.Enum):
    '''
    Enum of possible auto-response for permission / prompt dialogs.
    '''
    NONE = "none"
    AUTO_ACCEPT = "autoAccept"
    AUTO_REJECT = "autoReject"
    AUTO_OPT_OUT = "autoOptOut"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class NavigationType(enum.Enum):
    '''
    The type of a frameNavigated event.
    '''
    NAVIGATION = "Navigation"
    BACK_FORWARD_CACHE_RESTORE = "BackForwardCacheRestore"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class BackForwardCacheNotRestoredReason(enum.Enum):
    '''
    List of not restored reasons for back-forward cache.
    '''
    NOT_PRIMARY_MAIN_FRAME = "NotPrimaryMainFrame"
    BACK_FORWARD_CACHE_DISABLED = "BackForwardCacheDisabled"
    RELATED_ACTIVE_CONTENTS_EXIST = "RelatedActiveContentsExist"
    HTTP_STATUS_NOT_OK = "HTTPStatusNotOK"
    SCHEME_NOT_HTTP_OR_HTTPS = "SchemeNotHTTPOrHTTPS"
    LOADING = "Loading"
    WAS_GRANTED_MEDIA_ACCESS = "WasGrantedMediaAccess"
    DISABLE_FOR_RENDER_FRAME_HOST_CALLED = "DisableForRenderFrameHostCalled"
    DOMAIN_NOT_ALLOWED = "DomainNotAllowed"
    HTTP_METHOD_NOT_GET = "HTTPMethodNotGET"
    SUBFRAME_IS_NAVIGATING = "SubframeIsNavigating"
    TIMEOUT = "Timeout"
    CACHE_LIMIT = "CacheLimit"
    JAVA_SCRIPT_EXECUTION = "JavaScriptExecution"
    RENDERER_PROCESS_KILLED = "RendererProcessKilled"
    RENDERER_PROCESS_CRASHED = "RendererProcessCrashed"
    SCHEDULER_TRACKED_FEATURE_USED = "SchedulerTrackedFeatureUsed"
    CONFLICTING_BROWSING_INSTANCE = "ConflictingBrowsingInstance"
    CACHE_FLUSHED = "CacheFlushed"
    SERVICE_WORKER_VERSION_ACTIVATION = "ServiceWorkerVersionActivation"
    SESSION_RESTORED = "SessionRestored"
    SERVICE_WORKER_POST_MESSAGE = "ServiceWorkerPostMessage"
    ENTERED_BACK_FORWARD_CACHE_BEFORE_SERVICE_WORKER_HOST_ADDED = "EnteredBackForwardCacheBeforeServiceWorkerHostAdded"
    RENDER_FRAME_HOST_REUSED_SAME_SITE = "RenderFrameHostReused_SameSite"
    RENDER_FRAME_HOST_REUSED_CROSS_SITE = "RenderFrameHostReused_CrossSite"
    SERVICE_WORKER_CLAIM = "ServiceWorkerClaim"
    IGNORE_EVENT_AND_EVICT = "IgnoreEventAndEvict"
    HAVE_INNER_CONTENTS = "HaveInnerContents"
    TIMEOUT_PUTTING_IN_CACHE = "TimeoutPuttingInCache"
    BACK_FORWARD_CACHE_DISABLED_BY_LOW_MEMORY = "BackForwardCacheDisabledByLowMemory"
    BACK_FORWARD_CACHE_DISABLED_BY_COMMAND_LINE = "BackForwardCacheDisabledByCommandLine"
    NETWORK_REQUEST_DATAPIPE_DRAINED_AS_BYTES_CONSUMER = "NetworkRequestDatapipeDrainedAsBytesConsumer"
    NETWORK_REQUEST_REDIRECTED = "NetworkRequestRedirected"
    NETWORK_REQUEST_TIMEOUT = "NetworkRequestTimeout"
    NETWORK_EXCEEDS_BUFFER_LIMIT = "NetworkExceedsBufferLimit"
    NAVIGATION_CANCELLED_WHILE_RESTORING = "NavigationCancelledWhileRestoring"
    NOT_MOST_RECENT_NAVIGATION_ENTRY = "NotMostRecentNavigationEntry"
    BACK_FORWARD_CACHE_DISABLED_FOR_PRERENDER = "BackForwardCacheDisabledForPrerender"
    USER_AGENT_OVERRIDE_DIFFERS = "UserAgentOverrideDiffers"
    FOREGROUND_CACHE_LIMIT = "ForegroundCacheLimit"
    BROWSING_INSTANCE_NOT_SWAPPED = "BrowsingInstanceNotSwapped"
    BACK_FORWARD_CACHE_DISABLED_FOR_DELEGATE = "BackForwardCacheDisabledForDelegate"
    UNLOAD_HANDLER_EXISTS_IN_MAIN_FRAME = "UnloadHandlerExistsInMainFrame"
    UNLOAD_HANDLER_EXISTS_IN_SUB_FRAME = "UnloadHandlerExistsInSubFrame"
    SERVICE_WORKER_UNREGISTRATION = "ServiceWorkerUnregistration"
    CACHE_CONTROL_NO_STORE = "CacheControlNoStore"
    CACHE_CONTROL_NO_STORE_COOKIE_MODIFIED = "CacheControlNoStoreCookieModified"
    CACHE_CONTROL_NO_STORE_HTTP_ONLY_COOKIE_MODIFIED = "CacheControlNoStoreHTTPOnlyCookieModified"
    NO_RESPONSE_HEAD = "NoResponseHead"
    UNKNOWN = "Unknown"
    ACTIVATION_NAVIGATIONS_DISALLOWED_FOR_BUG1234857 = "ActivationNavigationsDisallowedForBug1234857"
    ERROR_DOCUMENT = "ErrorDocument"
    FENCED_FRAMES_EMBEDDER = "FencedFramesEmbedder"
    COOKIE_DISABLED = "CookieDisabled"
    HTTP_AUTH_REQUIRED = "HTTPAuthRequired"
    COOKIE_FLUSHED = "CookieFlushed"
    BROADCAST_CHANNEL_ON_MESSAGE = "BroadcastChannelOnMessage"
    WEB_VIEW_SETTINGS_CHANGED = "WebViewSettingsChanged"
    WEB_VIEW_JAVA_SCRIPT_OBJECT_CHANGED = "WebViewJavaScriptObjectChanged"
    WEB_VIEW_MESSAGE_LISTENER_INJECTED = "WebViewMessageListenerInjected"
    WEB_VIEW_SAFE_BROWSING_ALLOWLIST_CHANGED = "WebViewSafeBrowsingAllowlistChanged"
    WEB_VIEW_DOCUMENT_START_JAVASCRIPT_CHANGED = "WebViewDocumentStartJavascriptChanged"
    WEB_SOCKET = "WebSocket"
    WEB_TRANSPORT = "WebTransport"
    WEB_RTC = "WebRTC"
    MAIN_RESOURCE_HAS_CACHE_CONTROL_NO_STORE = "MainResourceHasCacheControlNoStore"
    MAIN_RESOURCE_HAS_CACHE_CONTROL_NO_CACHE = "MainResourceHasCacheControlNoCache"
    SUBRESOURCE_HAS_CACHE_CONTROL_NO_STORE = "SubresourceHasCacheControlNoStore"
    SUBRESOURCE_HAS_CACHE_CONTROL_NO_CACHE = "SubresourceHasCacheControlNoCache"
    CONTAINS_PLUGINS = "ContainsPlugins"
    DOCUMENT_LOADED = "DocumentLoaded"
    OUTSTANDING_NETWORK_REQUEST_OTHERS = "OutstandingNetworkRequestOthers"
    REQUESTED_MIDI_PERMISSION = "RequestedMIDIPermission"
    REQUESTED_AUDIO_CAPTURE_PERMISSION = "RequestedAudioCapturePermission"
    REQUESTED_VIDEO_CAPTURE_PERMISSION = "RequestedVideoCapturePermission"
    REQUESTED_BACK_FORWARD_CACHE_BLOCKED_SENSORS = "RequestedBackForwardCacheBlockedSensors"
    REQUESTED_BACKGROUND_WORK_PERMISSION = "RequestedBackgroundWorkPermission"
    BROADCAST_CHANNEL = "BroadcastChannel"
    WEB_XR = "WebXR"
    SHARED_WORKER = "SharedWorker"
    WEB_LOCKS = "WebLocks"
    WEB_HID = "WebHID"
    WEB_SHARE = "WebShare"
    REQUESTED_STORAGE_ACCESS_GRANT = "RequestedStorageAccessGrant"
    WEB_NFC = "WebNfc"
    OUTSTANDING_NETWORK_REQUEST_FETCH = "OutstandingNetworkRequestFetch"
    OUTSTANDING_NETWORK_REQUEST_XHR = "OutstandingNetworkRequestXHR"
    APP_BANNER = "AppBanner"
    PRINTING = "Printing"
    WEB_DATABASE = "WebDatabase"
    PICTURE_IN_PICTURE = "PictureInPicture"
    SPEECH_RECOGNIZER = "SpeechRecognizer"
    IDLE_MANAGER = "IdleManager"
    PAYMENT_MANAGER = "PaymentManager"
    SPEECH_SYNTHESIS = "SpeechSynthesis"
    KEYBOARD_LOCK = "KeyboardLock"
    WEB_OTP_SERVICE = "WebOTPService"
    OUTSTANDING_NETWORK_REQUEST_DIRECT_SOCKET = "OutstandingNetworkRequestDirectSocket"
    INJECTED_JAVASCRIPT = "InjectedJavascript"
    INJECTED_STYLE_SHEET = "InjectedStyleSheet"
    KEEPALIVE_REQUEST = "KeepaliveRequest"
    INDEXED_DB_EVENT = "IndexedDBEvent"
    DUMMY = "Dummy"
    JS_NETWORK_REQUEST_RECEIVED_CACHE_CONTROL_NO_STORE_RESOURCE = "JsNetworkRequestReceivedCacheControlNoStoreResource"
    WEB_RTC_STICKY = "WebRTCSticky"
    WEB_TRANSPORT_STICKY = "WebTransportSticky"
    WEB_SOCKET_STICKY = "WebSocketSticky"
    SMART_CARD = "SmartCard"
    LIVE_MEDIA_STREAM_TRACK = "LiveMediaStreamTrack"
    UNLOAD_HANDLER = "UnloadHandler"
    PARSER_ABORTED = "ParserAborted"
    CONTENT_SECURITY_HANDLER = "ContentSecurityHandler"
    CONTENT_WEB_AUTHENTICATION_API = "ContentWebAuthenticationAPI"
    CONTENT_FILE_CHOOSER = "ContentFileChooser"
    CONTENT_SERIAL = "ContentSerial"
    CONTENT_FILE_SYSTEM_ACCESS = "ContentFileSystemAccess"
    CONTENT_MEDIA_DEVICES_DISPATCHER_HOST = "ContentMediaDevicesDispatcherHost"
    CONTENT_WEB_BLUETOOTH = "ContentWebBluetooth"
    CONTENT_WEB_USB = "ContentWebUSB"
    CONTENT_MEDIA_SESSION_SERVICE = "ContentMediaSessionService"
    CONTENT_SCREEN_READER = "ContentScreenReader"
    CONTENT_DISCARDED = "ContentDiscarded"
    EMBEDDER_POPUP_BLOCKER_TAB_HELPER = "EmbedderPopupBlockerTabHelper"
    EMBEDDER_SAFE_BROWSING_TRIGGERED_POPUP_BLOCKER = "EmbedderSafeBrowsingTriggeredPopupBlocker"
    EMBEDDER_SAFE_BROWSING_THREAT_DETAILS = "EmbedderSafeBrowsingThreatDetails"
    EMBEDDER_APP_BANNER_MANAGER = "EmbedderAppBannerManager"
    EMBEDDER_DOM_DISTILLER_VIEWER_SOURCE = "EmbedderDomDistillerViewerSource"
    EMBEDDER_DOM_DISTILLER_SELF_DELETING_REQUEST_DELEGATE = "EmbedderDomDistillerSelfDeletingRequestDelegate"
    EMBEDDER_OOM_INTERVENTION_TAB_HELPER = "EmbedderOomInterventionTabHelper"
    EMBEDDER_OFFLINE_PAGE = "EmbedderOfflinePage"
    EMBEDDER_CHROME_PASSWORD_MANAGER_CLIENT_BIND_CREDENTIAL_MANAGER = "EmbedderChromePasswordManagerClientBindCredentialManager"
    EMBEDDER_PERMISSION_REQUEST_MANAGER = "EmbedderPermissionRequestManager"
    EMBEDDER_MODAL_DIALOG = "EmbedderModalDialog"
    EMBEDDER_EXTENSIONS = "EmbedderExtensions"
    EMBEDDER_EXTENSION_MESSAGING = "EmbedderExtensionMessaging"
    EMBEDDER_EXTENSION_MESSAGING_FOR_OPEN_PORT = "EmbedderExtensionMessagingForOpenPort"
    EMBEDDER_EXTENSION_SENT_MESSAGE_TO_CACHED_FRAME = "EmbedderExtensionSentMessageToCachedFrame"
    REQUESTED_BY_WEB_VIEW_CLIENT = "RequestedByWebViewClient"
    POST_MESSAGE_BY_WEB_VIEW_CLIENT = "PostMessageByWebViewClient"
    CACHE_CONTROL_NO_STORE_DEVICE_BOUND_SESSION_TERMINATED = "CacheControlNoStoreDeviceBoundSessionTerminated"
    CACHE_LIMIT_PRUNED_ON_MODERATE_MEMORY_PRESSURE = "CacheLimitPrunedOnModerateMemoryPressure"
    CACHE_LIMIT_PRUNED_ON_CRITICAL_MEMORY_PRESSURE = "CacheLimitPrunedOnCriticalMemoryPressure"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class BackForwardCacheNotRestoredReasonType(enum.Enum):
    '''
    Types of not restored reasons for back-forward cache.
    '''
    SUPPORT_PENDING = "SupportPending"
    PAGE_SUPPORT_NEEDED = "PageSupportNeeded"
    CIRCUMSTANTIAL = "Circumstantial"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class BackForwardCacheBlockingDetails:
    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: int

    #: Url of the file where blockage happened. Optional because of tests.
    url: typing.Optional[str] = None

    #: Function name where blockage happened. Optional because of anonymous functions and tests.
    function: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.url is not None:
            json['url'] = self.url
        if self.function is not None:
            json['function'] = self.function
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
            url=str(json['url']) if 'url' in json else None,
            function=str(json['function']) if 'function' in json else None,
        )


@dataclass
class BackForwardCacheNotRestoredExplanation:
    #: Type of the reason
    type_: BackForwardCacheNotRestoredReasonType

    #: Not restored reason
    reason: BackForwardCacheNotRestoredReason

    #: Context associated with the reason. The meaning of this context is
    #: dependent on the reason:
    #: - EmbedderExtensionSentMessageToCachedFrame: the extension ID.
    context: typing.Optional[str] = None

    details: typing.Optional[typing.List[BackForwardCacheBlockingDetails]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        json['reason'] = self.reason.to_json()
        if self.context is not None:
            json['context'] = self.context
        if self.details is not None:
            json['details'] = [i.to_json() for i in self.details]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=BackForwardCacheNotRestoredReasonType.from_json(json['type']),
            reason=BackForwardCacheNotRestoredReason.from_json(json['reason']),
            context=str(json['context']) if 'context' in json else None,
            details=[BackForwardCacheBlockingDetails.from_json(i) for i in json['details']] if 'details' in json else None,
        )


@dataclass
class BackForwardCacheNotRestoredExplanationTree:
    #: URL of each frame
    url: str

    #: Not restored reasons of each frame
    explanations: typing.List[BackForwardCacheNotRestoredExplanation]

    #: Array of children frame
    children: typing.List[BackForwardCacheNotRestoredExplanationTree]

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['explanations'] = [i.to_json() for i in self.explanations]
        json['children'] = [i.to_json() for i in self.children]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            explanations=[BackForwardCacheNotRestoredExplanation.from_json(i) for i in json['explanations']],
            children=[BackForwardCacheNotRestoredExplanationTree.from_json(i) for i in json['children']],
        )


def add_script_to_evaluate_on_load(
        script_source: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,ScriptIdentifier]:
    '''
    Deprecated, please use addScriptToEvaluateOnNewDocument instead.

    **EXPERIMENTAL**

    :param script_source:
    :returns: Identifier of the added script.
    '''
    params: T_JSON_DICT = dict()
    params['scriptSource'] = script_source
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.addScriptToEvaluateOnLoad',
        'params': params,
    }
    json = yield cmd_dict
    return ScriptIdentifier.from_json(json['identifier'])


def add_script_to_evaluate_on_new_document(
        source: str,
        world_name: typing.Optional[str] = None,
        include_command_line_api: typing.Optional[bool] = None,
        run_immediately: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,ScriptIdentifier]:
    '''
    Evaluates given script in every frame upon creation (before loading frame's scripts).

    :param source:
    :param world_name: **(EXPERIMENTAL)** *(Optional)* If specified, creates an isolated world with the given name and evaluates given script in it. This world name will be used as the ExecutionContextDescription::name when the corresponding event is emitted.
    :param include_command_line_api: **(EXPERIMENTAL)** *(Optional)* Specifies whether command line API should be available to the script, defaults to false.
    :param run_immediately: **(EXPERIMENTAL)** *(Optional)* If true, runs the script immediately on existing execution contexts or worlds. Default: false.
    :returns: Identifier of the added script.
    '''
    params: T_JSON_DICT = dict()
    params['source'] = source
    if world_name is not None:
        params['worldName'] = world_name
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if run_immediately is not None:
        params['runImmediately'] = run_immediately
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.addScriptToEvaluateOnNewDocument',
        'params': params,
    }
    json = yield cmd_dict
    return ScriptIdentifier.from_json(json['identifier'])


def bring_to_front() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Brings page to front (activates tab).
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.bringToFront',
    }
    json = yield cmd_dict


def capture_screenshot(
        format_: typing.Optional[str] = None,
        quality: typing.Optional[int] = None,
        clip: typing.Optional[Viewport] = None,
        from_surface: typing.Optional[bool] = None,
        capture_beyond_viewport: typing.Optional[bool] = None,
        optimize_for_speed: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Capture page screenshot.

    :param format_: *(Optional)* Image compression format (defaults to png).
    :param quality: *(Optional)* Compression quality from range [0..100] (jpeg only).
    :param clip: *(Optional)* Capture the screenshot of a given region only.
    :param from_surface: **(EXPERIMENTAL)** *(Optional)* Capture the screenshot from the surface, rather than the view. Defaults to true.
    :param capture_beyond_viewport: **(EXPERIMENTAL)** *(Optional)* Capture the screenshot beyond the viewport. Defaults to false.
    :param optimize_for_speed: **(EXPERIMENTAL)** *(Optional)* Optimize image encoding for speed, not for resulting size (defaults to false)
    :returns: Base64-encoded image data.
    '''
    params: T_JSON_DICT = dict()
    if format_ is not None:
        params['format'] = format_
    if quality is not None:
        params['quality'] = quality
    if clip is not None:
        params['clip'] = clip.to_json()
    if from_surface is not None:
        params['fromSurface'] = from_surface
    if capture_beyond_viewport is not None:
        params['captureBeyondViewport'] = capture_beyond_viewport
    if optimize_for_speed is not None:
        params['optimizeForSpeed'] = optimize_for_speed
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.captureScreenshot',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['data'])


def capture_snapshot(
        format_: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Returns a snapshot of the page as a string. For MHTML format, the serialization includes
    iframes, shadow DOM, external resources, and element-inline styles.

    **EXPERIMENTAL**

    :param format_: *(Optional)* Format (defaults to mhtml).
    :returns: Serialized page data.
    '''
    params: T_JSON_DICT = dict()
    if format_ is not None:
        params['format'] = format_
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.captureSnapshot',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['data'])


def clear_device_metrics_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the overridden device metrics.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.clearDeviceMetricsOverride',
    }
    json = yield cmd_dict


def clear_device_orientation_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the overridden Device Orientation.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.clearDeviceOrientationOverride',
    }
    json = yield cmd_dict


def clear_geolocation_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the overridden Geolocation Position and Error.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.clearGeolocationOverride',
    }
    json = yield cmd_dict


def create_isolated_world(
        frame_id: FrameId,
        world_name: typing.Optional[str] = None,
        grant_univeral_access: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.ExecutionContextId]:
    '''
    Creates an isolated world for the given frame.

    :param frame_id: Id of the frame in which the isolated world should be created.
    :param world_name: *(Optional)* An optional name which is reported in the Execution Context.
    :param grant_univeral_access: *(Optional)* Whether or not universal access should be granted to the isolated world. This is a powerful option, use with caution.
    :returns: Execution context of the isolated world.
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    if world_name is not None:
        params['worldName'] = world_name
    if grant_univeral_access is not None:
        params['grantUniveralAccess'] = grant_univeral_access
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.createIsolatedWorld',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.ExecutionContextId.from_json(json['executionContextId'])


def delete_cookie(
        cookie_name: str,
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes browser cookie with given name, domain and path.

    **EXPERIMENTAL**

    :param cookie_name: Name of the cookie to remove.
    :param url: URL to match cooke domain and path.
    '''
    params: T_JSON_DICT = dict()
    params['cookieName'] = cookie_name
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.deleteCookie',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables page domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.disable',
    }
    json = yield cmd_dict


def enable(
        enable_file_chooser_opened_event: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables page domain notifications.

    :param enable_file_chooser_opened_event: **(EXPERIMENTAL)** *(Optional)* If true, the ```Page.fileChooserOpened```` event will be emitted regardless of the state set by ````Page.setInterceptFileChooserDialog``` command (default: false).
    '''
    params: T_JSON_DICT = dict()
    if enable_file_chooser_opened_event is not None:
        params['enableFileChooserOpenedEvent'] = enable_file_chooser_opened_event
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.enable',
        'params': params,
    }
    json = yield cmd_dict


def get_app_manifest(
        manifest_id: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, typing.List[AppManifestError], typing.Optional[str], typing.Optional[AppManifestParsedProperties], WebAppManifest]]:
    '''
    Gets the processed manifest for this current document.
      This API always waits for the manifest to be loaded.
      If manifestId is provided, and it does not match the manifest of the
        current document, this API errors out.
      If there is not a loaded page, this API errors out immediately.

    :param manifest_id: *(Optional)*
    :returns: A tuple with the following items:

        0. **url** - Manifest location.
        1. **errors** - 
        2. **data** - *(Optional)* Manifest content.
        3. **parsed** - *(Optional)* Parsed manifest properties. Deprecated, use manifest instead.
        4. **manifest** - 
    '''
    params: T_JSON_DICT = dict()
    if manifest_id is not None:
        params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getAppManifest',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['url']),
        [AppManifestError.from_json(i) for i in json['errors']],
        str(json['data']) if 'data' in json else None,
        AppManifestParsedProperties.from_json(json['parsed']) if 'parsed' in json else None,
        WebAppManifest.from_json(json['manifest'])
    )


def get_installability_errors() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[InstallabilityError]]:
    '''


    **EXPERIMENTAL**

    :returns: 
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getInstallabilityErrors',
    }
    json = yield cmd_dict
    return [InstallabilityError.from_json(i) for i in json['installabilityErrors']]


def get_manifest_icons() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Optional[str]]:
    '''
    Deprecated because it's not guaranteed that the returned icon is in fact the one used for PWA installation.

    **EXPERIMENTAL**

    :returns: 
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getManifestIcons',
    }
    json = yield cmd_dict
    return str(json['primaryIcon']) if 'primaryIcon' in json else None


def get_app_id() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[str], typing.Optional[str]]]:
    '''
    Returns the unique (PWA) app id.
    Only returns values if the feature flag 'WebAppEnableManifestId' is enabled

    **EXPERIMENTAL**

    :returns: A tuple with the following items:

        0. **appId** - *(Optional)* App id, either from manifest's id attribute or computed from start_url
        1. **recommendedId** - *(Optional)* Recommendation for manifest's id attribute to match current id computed from start_url
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getAppId',
    }
    json = yield cmd_dict
    return (
        str(json['appId']) if 'appId' in json else None,
        str(json['recommendedId']) if 'recommendedId' in json else None
    )


def get_ad_script_ancestry_ids(
        frame_id: FrameId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AdScriptId]]:
    '''


    **EXPERIMENTAL**

    :param frame_id:
    :returns: The ancestry chain of ad script identifiers leading to this frame's creation, ordered from the most immediate script (in the frame creation stack) to more distant ancestors (that created the immediately preceding script). Only sent if frame is labelled as an ad and ids are available.
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getAdScriptAncestryIds',
        'params': params,
    }
    json = yield cmd_dict
    return [AdScriptId.from_json(i) for i in json['adScriptAncestryIds']]


def get_frame_tree() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,FrameTree]:
    '''
    Returns present frame tree structure.

    :returns: Present frame tree structure.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getFrameTree',
    }
    json = yield cmd_dict
    return FrameTree.from_json(json['frameTree'])


def get_layout_metrics() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[LayoutViewport, VisualViewport, dom.Rect, LayoutViewport, VisualViewport, dom.Rect]]:
    '''
    Returns metrics relating to the layouting of the page, such as viewport bounds/scale.

    :returns: A tuple with the following items:

        0. **layoutViewport** - Deprecated metrics relating to the layout viewport. Is in device pixels. Use ``cssLayoutViewport`` instead.
        1. **visualViewport** - Deprecated metrics relating to the visual viewport. Is in device pixels. Use ``cssVisualViewport`` instead.
        2. **contentSize** - Deprecated size of scrollable area. Is in DP. Use ``cssContentSize`` instead.
        3. **cssLayoutViewport** - Metrics relating to the layout viewport in CSS pixels.
        4. **cssVisualViewport** - Metrics relating to the visual viewport in CSS pixels.
        5. **cssContentSize** - Size of scrollable area in CSS pixels.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getLayoutMetrics',
    }
    json = yield cmd_dict
    return (
        LayoutViewport.from_json(json['layoutViewport']),
        VisualViewport.from_json(json['visualViewport']),
        dom.Rect.from_json(json['contentSize']),
        LayoutViewport.from_json(json['cssLayoutViewport']),
        VisualViewport.from_json(json['cssVisualViewport']),
        dom.Rect.from_json(json['cssContentSize'])
    )


def get_navigation_history() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[int, typing.List[NavigationEntry]]]:
    '''
    Returns navigation history for the current page.

    :returns: A tuple with the following items:

        0. **currentIndex** - Index of the current navigation history entry.
        1. **entries** - Array of navigation history entries.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getNavigationHistory',
    }
    json = yield cmd_dict
    return (
        int(json['currentIndex']),
        [NavigationEntry.from_json(i) for i in json['entries']]
    )


def reset_navigation_history() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets navigation history for the current page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.resetNavigationHistory',
    }
    json = yield cmd_dict


def get_resource_content(
        frame_id: FrameId,
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Returns content of the given resource.

    **EXPERIMENTAL**

    :param frame_id: Frame id to get resource for.
    :param url: URL of the resource to get content for.
    :returns: A tuple with the following items:

        0. **content** - Resource content.
        1. **base64Encoded** - True, if content was served as base64.
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getResourceContent',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['content']),
        bool(json['base64Encoded'])
    )


def get_resource_tree() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,FrameResourceTree]:
    '''
    Returns present frame / resource tree structure.

    **EXPERIMENTAL**

    :returns: Present frame / resource tree structure.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getResourceTree',
    }
    json = yield cmd_dict
    return FrameResourceTree.from_json(json['frameTree'])


def handle_java_script_dialog(
        accept: bool,
        prompt_text: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Accepts or dismisses a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload).

    :param accept: Whether to accept or dismiss the dialog.
    :param prompt_text: *(Optional)* The text to enter into the dialog prompt before accepting. Used only if this is a prompt dialog.
    '''
    params: T_JSON_DICT = dict()
    params['accept'] = accept
    if prompt_text is not None:
        params['promptText'] = prompt_text
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.handleJavaScriptDialog',
        'params': params,
    }
    json = yield cmd_dict


def navigate(
        url: str,
        referrer: typing.Optional[str] = None,
        transition_type: typing.Optional[TransitionType] = None,
        frame_id: typing.Optional[FrameId] = None,
        referrer_policy: typing.Optional[ReferrerPolicy] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[FrameId, typing.Optional[network.LoaderId], typing.Optional[str]]]:
    '''
    Navigates current page to the given URL.

    :param url: URL to navigate the page to.
    :param referrer: *(Optional)* Referrer URL.
    :param transition_type: *(Optional)* Intended transition type.
    :param frame_id: *(Optional)* Frame id to navigate, if not specified navigates the top frame.
    :param referrer_policy: **(EXPERIMENTAL)** *(Optional)* Referrer-policy used for the navigation.
    :returns: A tuple with the following items:

        0. **frameId** - Frame id that has navigated (or failed to navigate)
        1. **loaderId** - *(Optional)* Loader identifier. This is omitted in case of same-document navigation, as the previously committed loaderId would not change.
        2. **errorText** - *(Optional)* User friendly error message, present if and only if navigation has failed.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    if referrer is not None:
        params['referrer'] = referrer
    if transition_type is not None:
        params['transitionType'] = transition_type.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    if referrer_policy is not None:
        params['referrerPolicy'] = referrer_policy.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.navigate',
        'params': params,
    }
    json = yield cmd_dict
    return (
        FrameId.from_json(json['frameId']),
        network.LoaderId.from_json(json['loaderId']) if 'loaderId' in json else None,
        str(json['errorText']) if 'errorText' in json else None
    )


def navigate_to_history_entry(
        entry_id: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Navigates current page to the given history entry.

    :param entry_id: Unique id of the entry to navigate to.
    '''
    params: T_JSON_DICT = dict()
    params['entryId'] = entry_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.navigateToHistoryEntry',
        'params': params,
    }
    json = yield cmd_dict


def print_to_pdf(
        landscape: typing.Optional[bool] = None,
        display_header_footer: typing.Optional[bool] = None,
        print_background: typing.Optional[bool] = None,
        scale: typing.Optional[float] = None,
        paper_width: typing.Optional[float] = None,
        paper_height: typing.Optional[float] = None,
        margin_top: typing.Optional[float] = None,
        margin_bottom: typing.Optional[float] = None,
        margin_left: typing.Optional[float] = None,
        margin_right: typing.Optional[float] = None,
        page_ranges: typing.Optional[str] = None,
        header_template: typing.Optional[str] = None,
        footer_template: typing.Optional[str] = None,
        prefer_css_page_size: typing.Optional[bool] = None,
        transfer_mode: typing.Optional[str] = None,
        generate_tagged_pdf: typing.Optional[bool] = None,
        generate_document_outline: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, typing.Optional[io.StreamHandle]]]:
    '''
    Print page as PDF.

    :param landscape: *(Optional)* Paper orientation. Defaults to false.
    :param display_header_footer: *(Optional)* Display header and footer. Defaults to false.
    :param print_background: *(Optional)* Print background graphics. Defaults to false.
    :param scale: *(Optional)* Scale of the webpage rendering. Defaults to 1.
    :param paper_width: *(Optional)* Paper width in inches. Defaults to 8.5 inches.
    :param paper_height: *(Optional)* Paper height in inches. Defaults to 11 inches.
    :param margin_top: *(Optional)* Top margin in inches. Defaults to 1cm (~0.4 inches).
    :param margin_bottom: *(Optional)* Bottom margin in inches. Defaults to 1cm (~0.4 inches).
    :param margin_left: *(Optional)* Left margin in inches. Defaults to 1cm (~0.4 inches).
    :param margin_right: *(Optional)* Right margin in inches. Defaults to 1cm (~0.4 inches).
    :param page_ranges: *(Optional)* Paper ranges to print, one based, e.g., '1-5, 8, 11-13'. Pages are printed in the document order, not in the order specified, and no more than once. Defaults to empty string, which implies the entire document is printed. The page numbers are quietly capped to actual page count of the document, and ranges beyond the end of the document are ignored. If this results in no pages to print, an error is reported. It is an error to specify a range with start greater than end.
    :param header_template: *(Optional)* HTML template for the print header. Should be valid HTML markup with following classes used to inject printing values into them: - ```date````: formatted print date - ````title````: document title - ````url````: document location - ````pageNumber````: current page number - ````totalPages````: total pages in the document  For example, ````<span class=title></span>```` would generate span containing the title.
    :param footer_template: *(Optional)* HTML template for the print footer. Should use the same format as the ````headerTemplate````.
    :param prefer_css_page_size: *(Optional)* Whether or not to prefer page size as defined by css. Defaults to false, in which case the content will be scaled to fit the paper size.
    :param transfer_mode: **(EXPERIMENTAL)** *(Optional)* return as stream
    :param generate_tagged_pdf: **(EXPERIMENTAL)** *(Optional)* Whether or not to generate tagged (accessible) PDF. Defaults to embedder choice.
    :param generate_document_outline: **(EXPERIMENTAL)** *(Optional)* Whether or not to embed the document outline into the PDF.
    :returns: A tuple with the following items:

        0. **data** - Base64-encoded pdf data. Empty if `` returnAsStream` is specified.
        1. **stream** - *(Optional)* A handle of the stream that holds resulting PDF data.
    '''
    params: T_JSON_DICT = dict()
    if landscape is not None:
        params['landscape'] = landscape
    if display_header_footer is not None:
        params['displayHeaderFooter'] = display_header_footer
    if print_background is not None:
        params['printBackground'] = print_background
    if scale is not None:
        params['scale'] = scale
    if paper_width is not None:
        params['paperWidth'] = paper_width
    if paper_height is not None:
        params['paperHeight'] = paper_height
    if margin_top is not None:
        params['marginTop'] = margin_top
    if margin_bottom is not None:
        params['marginBottom'] = margin_bottom
    if margin_left is not None:
        params['marginLeft'] = margin_left
    if margin_right is not None:
        params['marginRight'] = margin_right
    if page_ranges is not None:
        params['pageRanges'] = page_ranges
    if header_template is not None:
        params['headerTemplate'] = header_template
    if footer_template is not None:
        params['footerTemplate'] = footer_template
    if prefer_css_page_size is not None:
        params['preferCSSPageSize'] = prefer_css_page_size
    if transfer_mode is not None:
        params['transferMode'] = transfer_mode
    if generate_tagged_pdf is not None:
        params['generateTaggedPDF'] = generate_tagged_pdf
    if generate_document_outline is not None:
        params['generateDocumentOutline'] = generate_document_outline
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.printToPDF',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['data']),
        io.StreamHandle.from_json(json['stream']) if 'stream' in json else None
    )


def reload(
        ignore_cache: typing.Optional[bool] = None,
        script_to_evaluate_on_load: typing.Optional[str] = None,
        loader_id: typing.Optional[network.LoaderId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Reloads given page optionally ignoring the cache.

    :param ignore_cache: *(Optional)* If true, browser cache is ignored (as if the user pressed Shift+refresh).
    :param script_to_evaluate_on_load: *(Optional)* If set, the script will be injected into all frames of the inspected page after reload. Argument will be ignored if reloading dataURL origin.
    :param loader_id: **(EXPERIMENTAL)** *(Optional)* If set, an error will be thrown if the target page's main frame's loader id does not match the provided id. This prevents accidentally reloading an unintended target in case there's a racing navigation.
    '''
    params: T_JSON_DICT = dict()
    if ignore_cache is not None:
        params['ignoreCache'] = ignore_cache
    if script_to_evaluate_on_load is not None:
        params['scriptToEvaluateOnLoad'] = script_to_evaluate_on_load
    if loader_id is not None:
        params['loaderId'] = loader_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.reload',
        'params': params,
    }
    json = yield cmd_dict


def remove_script_to_evaluate_on_load(
        identifier: ScriptIdentifier
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deprecated, please use removeScriptToEvaluateOnNewDocument instead.

    **EXPERIMENTAL**

    :param identifier:
    '''
    params: T_JSON_DICT = dict()
    params['identifier'] = identifier.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.removeScriptToEvaluateOnLoad',
        'params': params,
    }
    json = yield cmd_dict


def remove_script_to_evaluate_on_new_document(
        identifier: ScriptIdentifier
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes given script from the list.

    :param identifier:
    '''
    params: T_JSON_DICT = dict()
    params['identifier'] = identifier.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.removeScriptToEvaluateOnNewDocument',
        'params': params,
    }
    json = yield cmd_dict


def screencast_frame_ack(
        session_id: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Acknowledges that a screencast frame has been received by the frontend.

    **EXPERIMENTAL**

    :param session_id: Frame number.
    '''
    params: T_JSON_DICT = dict()
    params['sessionId'] = session_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.screencastFrameAck',
        'params': params,
    }
    json = yield cmd_dict


def search_in_resource(
        frame_id: FrameId,
        url: str,
        query: str,
        case_sensitive: typing.Optional[bool] = None,
        is_regex: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[debugger.SearchMatch]]:
    '''
    Searches for given string in resource content.

    **EXPERIMENTAL**

    :param frame_id: Frame id for resource to search in.
    :param url: URL of the resource to search in.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    params['url'] = url
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.searchInResource',
        'params': params,
    }
    json = yield cmd_dict
    return [debugger.SearchMatch.from_json(i) for i in json['result']]


def set_ad_blocking_enabled(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable Chrome's experimental ad filter on all sites.

    **EXPERIMENTAL**

    :param enabled: Whether to block ads.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setAdBlockingEnabled',
        'params': params,
    }
    json = yield cmd_dict


def set_bypass_csp(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable page Content Security Policy by-passing.

    :param enabled: Whether to bypass page CSP.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setBypassCSP',
        'params': params,
    }
    json = yield cmd_dict


def get_permissions_policy_state(
        frame_id: FrameId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[PermissionsPolicyFeatureState]]:
    '''
    Get Permissions Policy state on given frame.

    **EXPERIMENTAL**

    :param frame_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getPermissionsPolicyState',
        'params': params,
    }
    json = yield cmd_dict
    return [PermissionsPolicyFeatureState.from_json(i) for i in json['states']]


def get_origin_trials(
        frame_id: FrameId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[OriginTrial]]:
    '''
    Get Origin Trials on given frame.

    **EXPERIMENTAL**

    :param frame_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.getOriginTrials',
        'params': params,
    }
    json = yield cmd_dict
    return [OriginTrial.from_json(i) for i in json['originTrials']]


def set_device_metrics_override(
        width: int,
        height: int,
        device_scale_factor: float,
        mobile: bool,
        scale: typing.Optional[float] = None,
        screen_width: typing.Optional[int] = None,
        screen_height: typing.Optional[int] = None,
        position_x: typing.Optional[int] = None,
        position_y: typing.Optional[int] = None,
        dont_set_visible_size: typing.Optional[bool] = None,
        screen_orientation: typing.Optional[emulation.ScreenOrientation] = None,
        viewport: typing.Optional[Viewport] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the values of device screen dimensions (window.screen.width, window.screen.height,
    window.innerWidth, window.innerHeight, and "device-width"/"device-height"-related CSS media
    query results).

    **EXPERIMENTAL**

    :param width: Overriding width value in pixels (minimum 0, maximum 10000000). 0 disables the override.
    :param height: Overriding height value in pixels (minimum 0, maximum 10000000). 0 disables the override.
    :param device_scale_factor: Overriding device scale factor value. 0 disables the override.
    :param mobile: Whether to emulate mobile device. This includes viewport meta tag, overlay scrollbars, text autosizing and more.
    :param scale: *(Optional)* Scale to apply to resulting view image.
    :param screen_width: *(Optional)* Overriding screen width value in pixels (minimum 0, maximum 10000000).
    :param screen_height: *(Optional)* Overriding screen height value in pixels (minimum 0, maximum 10000000).
    :param position_x: *(Optional)* Overriding view X position on screen in pixels (minimum 0, maximum 10000000).
    :param position_y: *(Optional)* Overriding view Y position on screen in pixels (minimum 0, maximum 10000000).
    :param dont_set_visible_size: *(Optional)* Do not set visible view size, rely upon explicit setVisibleSize call.
    :param screen_orientation: *(Optional)* Screen orientation override.
    :param viewport: *(Optional)* The viewport dimensions and scale. If not set, the override is cleared.
    '''
    params: T_JSON_DICT = dict()
    params['width'] = width
    params['height'] = height
    params['deviceScaleFactor'] = device_scale_factor
    params['mobile'] = mobile
    if scale is not None:
        params['scale'] = scale
    if screen_width is not None:
        params['screenWidth'] = screen_width
    if screen_height is not None:
        params['screenHeight'] = screen_height
    if position_x is not None:
        params['positionX'] = position_x
    if position_y is not None:
        params['positionY'] = position_y
    if dont_set_visible_size is not None:
        params['dontSetVisibleSize'] = dont_set_visible_size
    if screen_orientation is not None:
        params['screenOrientation'] = screen_orientation.to_json()
    if viewport is not None:
        params['viewport'] = viewport.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setDeviceMetricsOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_device_orientation_override(
        alpha: float,
        beta: float,
        gamma: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the Device Orientation.

    **EXPERIMENTAL**

    :param alpha: Mock alpha
    :param beta: Mock beta
    :param gamma: Mock gamma
    '''
    params: T_JSON_DICT = dict()
    params['alpha'] = alpha
    params['beta'] = beta
    params['gamma'] = gamma
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setDeviceOrientationOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_font_families(
        font_families: FontFamilies,
        for_scripts: typing.Optional[typing.List[ScriptFontFamilies]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set generic font families.

    **EXPERIMENTAL**

    :param font_families: Specifies font families to set. If a font family is not specified, it won't be changed.
    :param for_scripts: *(Optional)* Specifies font families to set for individual scripts.
    '''
    params: T_JSON_DICT = dict()
    params['fontFamilies'] = font_families.to_json()
    if for_scripts is not None:
        params['forScripts'] = [i.to_json() for i in for_scripts]
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setFontFamilies',
        'params': params,
    }
    json = yield cmd_dict


def set_font_sizes(
        font_sizes: FontSizes
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set default font sizes.

    **EXPERIMENTAL**

    :param font_sizes: Specifies font sizes to set. If a font size is not specified, it won't be changed.
    '''
    params: T_JSON_DICT = dict()
    params['fontSizes'] = font_sizes.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setFontSizes',
        'params': params,
    }
    json = yield cmd_dict


def set_document_content(
        frame_id: FrameId,
        html: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets given markup as the document's HTML.

    :param frame_id: Frame id to set HTML for.
    :param html: HTML content to set.
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    params['html'] = html
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setDocumentContent',
        'params': params,
    }
    json = yield cmd_dict


def set_download_behavior(
        behavior: str,
        download_path: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set the behavior when downloading a file.

    **EXPERIMENTAL**

    :param behavior: Whether to allow all or deny all download requests, or use default Chrome behavior if available (otherwise deny).
    :param download_path: *(Optional)* The default path to save downloaded files to. This is required if behavior is set to 'allow'
    '''
    params: T_JSON_DICT = dict()
    params['behavior'] = behavior
    if download_path is not None:
        params['downloadPath'] = download_path
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setDownloadBehavior',
        'params': params,
    }
    json = yield cmd_dict


def set_geolocation_override(
        latitude: typing.Optional[float] = None,
        longitude: typing.Optional[float] = None,
        accuracy: typing.Optional[float] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the Geolocation Position or Error. Omitting any of the parameters emulates position
    unavailable.

    :param latitude: *(Optional)* Mock latitude
    :param longitude: *(Optional)* Mock longitude
    :param accuracy: *(Optional)* Mock accuracy
    '''
    params: T_JSON_DICT = dict()
    if latitude is not None:
        params['latitude'] = latitude
    if longitude is not None:
        params['longitude'] = longitude
    if accuracy is not None:
        params['accuracy'] = accuracy
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setGeolocationOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_lifecycle_events_enabled(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Controls whether page will emit lifecycle events.

    :param enabled: If true, starts emitting lifecycle events.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setLifecycleEventsEnabled',
        'params': params,
    }
    json = yield cmd_dict


def set_touch_emulation_enabled(
        enabled: bool,
        configuration: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Toggles mouse event-based touch event emulation.

    **EXPERIMENTAL**

    :param enabled: Whether the touch event emulation should be enabled.
    :param configuration: *(Optional)* Touch/gesture events configuration. Default: current platform.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    if configuration is not None:
        params['configuration'] = configuration
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setTouchEmulationEnabled',
        'params': params,
    }
    json = yield cmd_dict


def start_screencast(
        format_: typing.Optional[str] = None,
        quality: typing.Optional[int] = None,
        max_width: typing.Optional[int] = None,
        max_height: typing.Optional[int] = None,
        every_nth_frame: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Starts sending each frame using the ``screencastFrame`` event.

    **EXPERIMENTAL**

    :param format_: *(Optional)* Image compression format.
    :param quality: *(Optional)* Compression quality from range [0..100].
    :param max_width: *(Optional)* Maximum screenshot width.
    :param max_height: *(Optional)* Maximum screenshot height.
    :param every_nth_frame: *(Optional)* Send every n-th frame.
    '''
    params: T_JSON_DICT = dict()
    if format_ is not None:
        params['format'] = format_
    if quality is not None:
        params['quality'] = quality
    if max_width is not None:
        params['maxWidth'] = max_width
    if max_height is not None:
        params['maxHeight'] = max_height
    if every_nth_frame is not None:
        params['everyNthFrame'] = every_nth_frame
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.startScreencast',
        'params': params,
    }
    json = yield cmd_dict


def stop_loading() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Force the page stop all navigations and pending resource fetches.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.stopLoading',
    }
    json = yield cmd_dict


def crash() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Crashes renderer on the IO thread, generates minidumps.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.crash',
    }
    json = yield cmd_dict


def close() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Tries to close page, running its beforeunload hooks, if any.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.close',
    }
    json = yield cmd_dict


def set_web_lifecycle_state(
        state: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Tries to update the web lifecycle state of the page.
    It will transition the page to the given state according to:
    https://github.com/WICG/web-lifecycle/

    **EXPERIMENTAL**

    :param state: Target lifecycle state
    '''
    params: T_JSON_DICT = dict()
    params['state'] = state
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setWebLifecycleState',
        'params': params,
    }
    json = yield cmd_dict


def stop_screencast() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Stops sending each frame in the ``screencastFrame``.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.stopScreencast',
    }
    json = yield cmd_dict


def produce_compilation_cache(
        scripts: typing.List[CompilationCacheParams]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests backend to produce compilation cache for the specified scripts.
    ``scripts`` are appended to the list of scripts for which the cache
    would be produced. The list may be reset during page navigation.
    When script with a matching URL is encountered, the cache is optionally
    produced upon backend discretion, based on internal heuristics.
    See also: ``Page.compilationCacheProduced``.

    **EXPERIMENTAL**

    :param scripts:
    '''
    params: T_JSON_DICT = dict()
    params['scripts'] = [i.to_json() for i in scripts]
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.produceCompilationCache',
        'params': params,
    }
    json = yield cmd_dict


def add_compilation_cache(
        url: str,
        data: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Seeds compilation cache for given url. Compilation cache does not survive
    cross-process navigation.

    **EXPERIMENTAL**

    :param url:
    :param data: Base64-encoded data
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    params['data'] = data
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.addCompilationCache',
        'params': params,
    }
    json = yield cmd_dict


def clear_compilation_cache() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears seeded compilation cache.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.clearCompilationCache',
    }
    json = yield cmd_dict


def set_spc_transaction_mode(
        mode: AutoResponseMode
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the Secure Payment Confirmation transaction mode.
    https://w3c.github.io/secure-payment-confirmation/#sctn-automation-set-spc-transaction-mode

    **EXPERIMENTAL**

    :param mode:
    '''
    params: T_JSON_DICT = dict()
    params['mode'] = mode.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setSPCTransactionMode',
        'params': params,
    }
    json = yield cmd_dict


def set_rph_registration_mode(
        mode: AutoResponseMode
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Extensions for Custom Handlers API:
    https://html.spec.whatwg.org/multipage/system-state.html#rph-automation

    **EXPERIMENTAL**

    :param mode:
    '''
    params: T_JSON_DICT = dict()
    params['mode'] = mode.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setRPHRegistrationMode',
        'params': params,
    }
    json = yield cmd_dict


def generate_test_report(
        message: str,
        group: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Generates a report for testing.

    **EXPERIMENTAL**

    :param message: Message to be displayed in the report.
    :param group: *(Optional)* Specifies the endpoint group to deliver the report to.
    '''
    params: T_JSON_DICT = dict()
    params['message'] = message
    if group is not None:
        params['group'] = group
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.generateTestReport',
        'params': params,
    }
    json = yield cmd_dict


def wait_for_debugger() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Pauses page execution. Can be resumed using generic Runtime.runIfWaitingForDebugger.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.waitForDebugger',
    }
    json = yield cmd_dict


def set_intercept_file_chooser_dialog(
        enabled: bool,
        cancel: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Intercept file chooser requests and transfer control to protocol clients.
    When file chooser interception is enabled, native file chooser dialog is not shown.
    Instead, a protocol event ``Page.fileChooserOpened`` is emitted.

    :param enabled:
    :param cancel: **(EXPERIMENTAL)** *(Optional)* If true, cancels the dialog by emitting relevant events (if any) in addition to not showing it if the interception is enabled (default: false).
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    if cancel is not None:
        params['cancel'] = cancel
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setInterceptFileChooserDialog',
        'params': params,
    }
    json = yield cmd_dict


def set_prerendering_allowed(
        is_allowed: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable/disable prerendering manually.

    This command is a short-term solution for https://crbug.com/1440085.
    See https://docs.google.com/document/d/12HVmFxYj5Jc-eJr5OmWsa2bqTJsbgGLKI6ZIyx0_wpA
    for more details.

    TODO(https://crbug.com/1440085): Remove this once Puppeteer supports tab targets.

    **EXPERIMENTAL**

    :param is_allowed:
    '''
    params: T_JSON_DICT = dict()
    params['isAllowed'] = is_allowed
    cmd_dict: T_JSON_DICT = {
        'method': 'Page.setPrerenderingAllowed',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Page.domContentEventFired')
@dataclass
class DomContentEventFired:
    timestamp: network.MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DomContentEventFired:
        return cls(
            timestamp=network.MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Page.fileChooserOpened')
@dataclass
class FileChooserOpened:
    '''
    Emitted only when ``page.interceptFileChooser`` is enabled.
    '''
    #: Id of the frame containing input node.
    frame_id: FrameId
    #: Input mode.
    mode: str
    #: Input node id. Only present for file choosers opened via an ``<input type="file">`` element.
    backend_node_id: typing.Optional[dom.BackendNodeId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FileChooserOpened:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            mode=str(json['mode']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None
        )


@event_class('Page.frameAttached')
@dataclass
class FrameAttached:
    '''
    Fired when frame has been attached to its parent.
    '''
    #: Id of the frame that has been attached.
    frame_id: FrameId
    #: Parent frame identifier.
    parent_frame_id: FrameId
    #: JavaScript stack trace of when frame was attached, only set if frame initiated from script.
    stack: typing.Optional[runtime.StackTrace]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameAttached:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            parent_frame_id=FrameId.from_json(json['parentFrameId']),
            stack=runtime.StackTrace.from_json(json['stack']) if 'stack' in json else None
        )


@event_class('Page.frameClearedScheduledNavigation')
@dataclass
class FrameClearedScheduledNavigation:
    '''
    Fired when frame no longer has a scheduled navigation.
    '''
    #: Id of the frame that has cleared its scheduled navigation.
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameClearedScheduledNavigation:
        return cls(
            frame_id=FrameId.from_json(json['frameId'])
        )


@event_class('Page.frameDetached')
@dataclass
class FrameDetached:
    '''
    Fired when frame has been detached from its parent.
    '''
    #: Id of the frame that has been detached.
    frame_id: FrameId
    reason: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameDetached:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            reason=str(json['reason'])
        )


@event_class('Page.frameSubtreeWillBeDetached')
@dataclass
class FrameSubtreeWillBeDetached:
    '''
    **EXPERIMENTAL**

    Fired before frame subtree is detached. Emitted before any frame of the
    subtree is actually detached.
    '''
    #: Id of the frame that is the root of the subtree that will be detached.
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameSubtreeWillBeDetached:
        return cls(
            frame_id=FrameId.from_json(json['frameId'])
        )


@event_class('Page.frameNavigated')
@dataclass
class FrameNavigated:
    '''
    Fired once navigation of the frame has completed. Frame is now associated with the new loader.
    '''
    #: Frame object.
    frame: Frame
    type_: NavigationType

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameNavigated:
        return cls(
            frame=Frame.from_json(json['frame']),
            type_=NavigationType.from_json(json['type'])
        )


@event_class('Page.documentOpened')
@dataclass
class DocumentOpened:
    '''
    **EXPERIMENTAL**

    Fired when opening document to write to.
    '''
    #: Frame object.
    frame: Frame

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DocumentOpened:
        return cls(
            frame=Frame.from_json(json['frame'])
        )


@event_class('Page.frameResized')
@dataclass
class FrameResized:
    '''
    **EXPERIMENTAL**


    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameResized:
        return cls(

        )


@event_class('Page.frameStartedNavigating')
@dataclass
class FrameStartedNavigating:
    '''
    **EXPERIMENTAL**

    Fired when a navigation starts. This event is fired for both
    renderer-initiated and browser-initiated navigations. For renderer-initiated
    navigations, the event is fired after ``frameRequestedNavigation``.
    Navigation may still be cancelled after the event is issued. Multiple events
    can be fired for a single navigation, for example, when a same-document
    navigation becomes a cross-document navigation (such as in the case of a
    frameset).
    '''
    #: ID of the frame that is being navigated.
    frame_id: FrameId
    #: The URL the navigation started with. The final URL can be different.
    url: str
    #: Loader identifier. Even though it is present in case of same-document
    #: navigation, the previously committed loaderId would not change unless
    #: the navigation changes from a same-document to a cross-document
    #: navigation.
    loader_id: network.LoaderId
    navigation_type: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameStartedNavigating:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            url=str(json['url']),
            loader_id=network.LoaderId.from_json(json['loaderId']),
            navigation_type=str(json['navigationType'])
        )


@event_class('Page.frameRequestedNavigation')
@dataclass
class FrameRequestedNavigation:
    '''
    **EXPERIMENTAL**

    Fired when a renderer-initiated navigation is requested.
    Navigation may still be cancelled after the event is issued.
    '''
    #: Id of the frame that is being navigated.
    frame_id: FrameId
    #: The reason for the navigation.
    reason: ClientNavigationReason
    #: The destination URL for the requested navigation.
    url: str
    #: The disposition for the navigation.
    disposition: ClientNavigationDisposition

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameRequestedNavigation:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            reason=ClientNavigationReason.from_json(json['reason']),
            url=str(json['url']),
            disposition=ClientNavigationDisposition.from_json(json['disposition'])
        )


@event_class('Page.frameScheduledNavigation')
@dataclass
class FrameScheduledNavigation:
    '''
    Fired when frame schedules a potential navigation.
    '''
    #: Id of the frame that has scheduled a navigation.
    frame_id: FrameId
    #: Delay (in seconds) until the navigation is scheduled to begin. The navigation is not
    #: guaranteed to start.
    delay: float
    #: The reason for the navigation.
    reason: ClientNavigationReason
    #: The destination URL for the scheduled navigation.
    url: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameScheduledNavigation:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            delay=float(json['delay']),
            reason=ClientNavigationReason.from_json(json['reason']),
            url=str(json['url'])
        )


@event_class('Page.frameStartedLoading')
@dataclass
class FrameStartedLoading:
    '''
    **EXPERIMENTAL**

    Fired when frame has started loading.
    '''
    #: Id of the frame that has started loading.
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameStartedLoading:
        return cls(
            frame_id=FrameId.from_json(json['frameId'])
        )


@event_class('Page.frameStoppedLoading')
@dataclass
class FrameStoppedLoading:
    '''
    **EXPERIMENTAL**

    Fired when frame has stopped loading.
    '''
    #: Id of the frame that has stopped loading.
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameStoppedLoading:
        return cls(
            frame_id=FrameId.from_json(json['frameId'])
        )


@event_class('Page.downloadWillBegin')
@dataclass
class DownloadWillBegin:
    '''
    **EXPERIMENTAL**

    Fired when page is about to start a download.
    Deprecated. Use Browser.downloadWillBegin instead.
    '''
    #: Id of the frame that caused download to begin.
    frame_id: FrameId
    #: Global unique identifier of the download.
    guid: str
    #: URL of the resource being downloaded.
    url: str
    #: Suggested file name of the resource (the actual name of the file saved on disk may differ).
    suggested_filename: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DownloadWillBegin:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            guid=str(json['guid']),
            url=str(json['url']),
            suggested_filename=str(json['suggestedFilename'])
        )


@event_class('Page.downloadProgress')
@dataclass
class DownloadProgress:
    '''
    **EXPERIMENTAL**

    Fired when download makes progress. Last call has ``done`` == true.
    Deprecated. Use Browser.downloadProgress instead.
    '''
    #: Global unique identifier of the download.
    guid: str
    #: Total expected bytes to download.
    total_bytes: float
    #: Total bytes received.
    received_bytes: float
    #: Download status.
    state: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DownloadProgress:
        return cls(
            guid=str(json['guid']),
            total_bytes=float(json['totalBytes']),
            received_bytes=float(json['receivedBytes']),
            state=str(json['state'])
        )


@event_class('Page.interstitialHidden')
@dataclass
class InterstitialHidden:
    '''
    Fired when interstitial page was hidden
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterstitialHidden:
        return cls(

        )


@event_class('Page.interstitialShown')
@dataclass
class InterstitialShown:
    '''
    Fired when interstitial page was shown
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterstitialShown:
        return cls(

        )


@event_class('Page.javascriptDialogClosed')
@dataclass
class JavascriptDialogClosed:
    '''
    Fired when a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload) has been
    closed.
    '''
    #: Frame id.
    frame_id: FrameId
    #: Whether dialog was confirmed.
    result: bool
    #: User input in case of prompt.
    user_input: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> JavascriptDialogClosed:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            result=bool(json['result']),
            user_input=str(json['userInput'])
        )


@event_class('Page.javascriptDialogOpening')
@dataclass
class JavascriptDialogOpening:
    '''
    Fired when a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload) is about to
    open.
    '''
    #: Frame url.
    url: str
    #: Frame id.
    frame_id: FrameId
    #: Message that will be displayed by the dialog.
    message: str
    #: Dialog type.
    type_: DialogType
    #: True iff browser is capable showing or acting on the given dialog. When browser has no
    #: dialog handler for given target, calling alert while Page domain is engaged will stall
    #: the page execution. Execution can be resumed via calling Page.handleJavaScriptDialog.
    has_browser_handler: bool
    #: Default dialog prompt.
    default_prompt: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> JavascriptDialogOpening:
        return cls(
            url=str(json['url']),
            frame_id=FrameId.from_json(json['frameId']),
            message=str(json['message']),
            type_=DialogType.from_json(json['type']),
            has_browser_handler=bool(json['hasBrowserHandler']),
            default_prompt=str(json['defaultPrompt']) if 'defaultPrompt' in json else None
        )


@event_class('Page.lifecycleEvent')
@dataclass
class LifecycleEvent:
    '''
    Fired for lifecycle events (navigation, load, paint, etc) in the current
    target (including local frames).
    '''
    #: Id of the frame.
    frame_id: FrameId
    #: Loader identifier. Empty string if the request is fetched from worker.
    loader_id: network.LoaderId
    name: str
    timestamp: network.MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LifecycleEvent:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            loader_id=network.LoaderId.from_json(json['loaderId']),
            name=str(json['name']),
            timestamp=network.MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Page.backForwardCacheNotUsed')
@dataclass
class BackForwardCacheNotUsed:
    '''
    **EXPERIMENTAL**

    Fired for failed bfcache history navigations if BackForwardCache feature is enabled. Do
    not assume any ordering with the Page.frameNavigated event. This event is fired only for
    main-frame history navigation where the document changes (non-same-document navigations),
    when bfcache navigation fails.
    '''
    #: The loader id for the associated navigation.
    loader_id: network.LoaderId
    #: The frame id of the associated frame.
    frame_id: FrameId
    #: Array of reasons why the page could not be cached. This must not be empty.
    not_restored_explanations: typing.List[BackForwardCacheNotRestoredExplanation]
    #: Tree structure of reasons why the page could not be cached for each frame.
    not_restored_explanations_tree: typing.Optional[BackForwardCacheNotRestoredExplanationTree]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackForwardCacheNotUsed:
        return cls(
            loader_id=network.LoaderId.from_json(json['loaderId']),
            frame_id=FrameId.from_json(json['frameId']),
            not_restored_explanations=[BackForwardCacheNotRestoredExplanation.from_json(i) for i in json['notRestoredExplanations']],
            not_restored_explanations_tree=BackForwardCacheNotRestoredExplanationTree.from_json(json['notRestoredExplanationsTree']) if 'notRestoredExplanationsTree' in json else None
        )


@event_class('Page.loadEventFired')
@dataclass
class LoadEventFired:
    timestamp: network.MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadEventFired:
        return cls(
            timestamp=network.MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Page.navigatedWithinDocument')
@dataclass
class NavigatedWithinDocument:
    '''
    **EXPERIMENTAL**

    Fired when same-document navigation happens, e.g. due to history API usage or anchor navigation.
    '''
    #: Id of the frame.
    frame_id: FrameId
    #: Frame's new url.
    url: str
    #: Navigation type
    navigation_type: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NavigatedWithinDocument:
        return cls(
            frame_id=FrameId.from_json(json['frameId']),
            url=str(json['url']),
            navigation_type=str(json['navigationType'])
        )


@event_class('Page.screencastFrame')
@dataclass
class ScreencastFrame:
    '''
    **EXPERIMENTAL**

    Compressed image data requested by the ``startScreencast``.
    '''
    #: Base64-encoded compressed image.
    data: str
    #: Screencast frame metadata.
    metadata: ScreencastFrameMetadata
    #: Frame number.
    session_id: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreencastFrame:
        return cls(
            data=str(json['data']),
            metadata=ScreencastFrameMetadata.from_json(json['metadata']),
            session_id=int(json['sessionId'])
        )


@event_class('Page.screencastVisibilityChanged')
@dataclass
class ScreencastVisibilityChanged:
    '''
    **EXPERIMENTAL**

    Fired when the page with currently enabled screencast was shown or hidden .
    '''
    #: True if the page is visible.
    visible: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreencastVisibilityChanged:
        return cls(
            visible=bool(json['visible'])
        )


@event_class('Page.windowOpen')
@dataclass
class WindowOpen:
    '''
    Fired when a new window is going to be opened, via window.open(), link click, form submission,
    etc.
    '''
    #: The URL for the new window.
    url: str
    #: Window name.
    window_name: str
    #: An array of enabled window features.
    window_features: typing.List[str]
    #: Whether or not it was triggered by user gesture.
    user_gesture: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WindowOpen:
        return cls(
            url=str(json['url']),
            window_name=str(json['windowName']),
            window_features=[str(i) for i in json['windowFeatures']],
            user_gesture=bool(json['userGesture'])
        )


@event_class('Page.compilationCacheProduced')
@dataclass
class CompilationCacheProduced:
    '''
    **EXPERIMENTAL**

    Issued for every compilation cache generated. Is only available
    if Page.setGenerateCompilationCache is enabled.
    '''
    url: str
    #: Base64-encoded data
    data: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CompilationCacheProduced:
        return cls(
            url=str(json['url']),
            data=str(json['data'])
        )