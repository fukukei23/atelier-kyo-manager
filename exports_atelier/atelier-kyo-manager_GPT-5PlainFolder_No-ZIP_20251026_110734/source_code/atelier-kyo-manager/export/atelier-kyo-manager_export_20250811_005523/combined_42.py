
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\holonomic\holonomic.py ===
"""
This module implements Holonomic Functions and
various operations on them.
"""

from sympy.core import Add, Mul, Pow
from sympy.core.numbers import (NaN, Infinity, NegativeInfinity, Float, I, pi,
        equal_valued, int_valued)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Ci, Shi, Si, erf, erfc, erfi)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals import meijerint
from sympy.matrices import Matrix
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains import QQ, RR
from sympy.polys.polyclasses import DMF
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.printing import sstr
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import nsimplify
from sympy.solvers.solvers import solve

from .recurrence import HolonomicSequence, RecurrenceOperator, RecurrenceOperators
from .holonomicerrors import (NotPowerSeriesError, NotHyperSeriesError,
    SingularityError, NotHolonomicError)


def _find_nonzero_solution(r, homosys):
    ones = lambda shape: DomainMatrix.ones(shape, r.domain)
    particular, nullspace = r._solve(homosys)
    nullity = nullspace.shape[0]
    nullpart = ones((1, nullity)) * nullspace
    sol = (particular + nullpart).transpose()
    return sol



def DifferentialOperators(base, generator):
    r"""
    This function is used to create annihilators using ``Dx``.

    Explanation
    ===========

    Returns an Algebra of Differential Operators also called Weyl Algebra
    and the operator for differentiation i.e. the ``Dx`` operator.

    Parameters
    ==========

    base:
        Base polynomial ring for the algebra.
        The base polynomial ring is the ring of polynomials in :math:`x` that
        will appear as coefficients in the operators.
    generator:
        Generator of the algebra which can
        be either a noncommutative ``Symbol`` or a string. e.g. "Dx" or "D".

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.abc import x
    >>> from sympy.holonomic.holonomic import DifferentialOperators
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    >>> R
    Univariate Differential Operator Algebra in intermediate Dx over the base ring ZZ[x]
    >>> Dx*x
    (1) + (x)*Dx
    """

    ring = DifferentialOperatorAlgebra(base, generator)
    return (ring, ring.derivative_operator)


class DifferentialOperatorAlgebra:
    r"""
    An Ore Algebra is a set of noncommutative polynomials in the
    intermediate ``Dx`` and coefficients in a base polynomial ring :math:`A`.
    It follows the commutation rule:

    .. math ::
       Dxa = \sigma(a)Dx + \delta(a)

    for :math:`a \subset A`.

    Where :math:`\sigma: A \Rightarrow A` is an endomorphism and :math:`\delta: A \rightarrow A`
    is a skew-derivation i.e. :math:`\delta(ab) = \delta(a) b + \sigma(a) \delta(b)`.

    If one takes the sigma as identity map and delta as the standard derivation
    then it becomes the algebra of Differential Operators also called
    a Weyl Algebra i.e. an algebra whose elements are Differential Operators.

    This class represents a Weyl Algebra and serves as the parent ring for
    Differential Operators.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.holonomic import DifferentialOperators
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    >>> R
    Univariate Differential Operator Algebra in intermediate Dx over the base ring
    ZZ[x]

    See Also
    ========

    DifferentialOperator
    """

    def __init__(self, base, generator):
        # the base polynomial ring for the algebra
        self.base = base
        # the operator representing differentiation i.e. `Dx`
        self.derivative_operator = DifferentialOperator(
            [base.zero, base.one], self)

        if generator is None:
            self.gen_symbol = Symbol('Dx', commutative=False)
        else:
            if isinstance(generator, str):
                self.gen_symbol = Symbol(generator, commutative=False)
            elif isinstance(generator, Symbol):
                self.gen_symbol = generator

    def __str__(self):
        string = 'Univariate Differential Operator Algebra in intermediate '\
            + sstr(self.gen_symbol) + ' over the base ring ' + \
            (self.base).__str__()

        return string

    __repr__ = __str__

    def __eq__(self, other):
        return self.base == other.base and \
               self.gen_symbol == other.gen_symbol


class DifferentialOperator:
    """
    Differential Operators are elements of Weyl Algebra. The Operators
    are defined by a list of polynomials in the base ring and the
    parent ring of the Operator i.e. the algebra it belongs to.

    Explanation
    ===========

    Takes a list of polynomials for each power of ``Dx`` and the
    parent ring which must be an instance of DifferentialOperatorAlgebra.

    A Differential Operator can be created easily using
    the operator ``Dx``. See examples below.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import DifferentialOperator, DifferentialOperators
    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')

    >>> DifferentialOperator([0, 1, x**2], R)
    (1)*Dx + (x**2)*Dx**2

    >>> (x*Dx*x + 1 - Dx**2)**2
    (2*x**2 + 2*x + 1) + (4*x**3 + 2*x**2 - 4)*Dx + (x**4 - 6*x - 2)*Dx**2 + (-2*x**2)*Dx**3 + (1)*Dx**4

    See Also
    ========

    DifferentialOperatorAlgebra
    """

    _op_priority = 20

    def __init__(self, list_of_poly, parent):
        """
        Parameters
        ==========

        list_of_poly:
            List of polynomials belonging to the base ring of the algebra.
        parent:
            Parent algebra of the operator.
        """

        # the parent ring for this operator
        # must be an DifferentialOperatorAlgebra object
        self.parent = parent
        base = self.parent.base
        self.x = base.gens[0] if isinstance(base.gens[0], Symbol) else base.gens[0][0]
        # sequence of polynomials in x for each power of Dx
        # the list should not have trailing zeroes
        # represents the operator
        # convert the expressions into ring elements using from_sympy
        for i, j in enumerate(list_of_poly):
            if not isinstance(j, base.dtype):
                list_of_poly[i] = base.from_sympy(sympify(j))
            else:
                list_of_poly[i] = base.from_sympy(base.to_sympy(j))

        self.listofpoly = list_of_poly
        # highest power of `Dx`
        self.order = len(self.listofpoly) - 1

    def __mul__(self, other):
        """
        Multiplies two DifferentialOperator and returns another
        DifferentialOperator instance using the commutation rule
        Dx*a = a*Dx + a'
        """

        listofself = self.listofpoly
        if isinstance(other, DifferentialOperator):
            listofother = other.listofpoly
        elif isinstance(other, self.parent.base.dtype):
            listofother = [other]
        else:
            listofother = [self.parent.base.from_sympy(sympify(other))]

        # multiplies a polynomial `b` with a list of polynomials
        def _mul_dmp_diffop(b, listofother):
            if isinstance(listofother, list):
                return [i * b for i in listofother]
            return [b * listofother]

        sol = _mul_dmp_diffop(listofself[0], listofother)

        # compute Dx^i * b
        def _mul_Dxi_b(b):
            sol1 = [self.parent.base.zero]
            sol2 = []

            if isinstance(b, list):
                for i in b:
                    sol1.append(i)
                    sol2.append(i.diff())
            else:
                sol1.append(self.parent.base.from_sympy(b))
                sol2.append(self.parent.base.from_sympy(b).diff())

            return _add_lists(sol1, sol2)

        for i in range(1, len(listofself)):
            # find Dx^i * b in ith iteration
            listofother = _mul_Dxi_b(listofother)
            # solution = solution + listofself[i] * (Dx^i * b)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))

        return DifferentialOperator(sol, self.parent)

    def __rmul__(self, other):
        if not isinstance(other, DifferentialOperator):

            if not isinstance(other, self.parent.base.dtype):
                other = (self.parent.base).from_sympy(sympify(other))

            sol = [other * j for j in self.listofpoly]
            return DifferentialOperator(sol, self.parent)

    def __add__(self, other):
        if isinstance(other, DifferentialOperator):

            sol = _add_lists(self.listofpoly, other.listofpoly)
            return DifferentialOperator(sol, self.parent)

        list_self = self.listofpoly
        if not isinstance(other, self.parent.base.dtype):
            list_other = [((self.parent).base).from_sympy(sympify(other))]
        else:
            list_other = [other]
        sol = [list_self[0] + list_other[0]] + list_self[1:]
        return DifferentialOperator(sol, self.parent)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        return self * (S.One / other)

    def __pow__(self, n):
        if n == 1:
            return self
        result = DifferentialOperator([self.parent.base.one], self.parent)
        if n == 0:
            return result
        # if self is `Dx`
        if self.listofpoly == self.parent.derivative_operator.listofpoly:
            sol = [self.parent.base.zero]*n + [self.parent.base.one]
            return DifferentialOperator(sol, self.parent)
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def __str__(self):
        listofpoly = self.listofpoly
        print_str = ''

        for i, j in enumerate(listofpoly):
            if j == self.parent.base.zero:
                continue

            j = self.parent.base.to_sympy(j)

            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue

            if print_str:
                print_str += ' + '

            if i == 1:
                print_str += '(' + sstr(j) + ')*%s' %(self.parent.gen_symbol)
                continue

            print_str += '(' + sstr(j) + ')' + '*%s**' %(self.parent.gen_symbol) + sstr(i)

        return print_str

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, DifferentialOperator):
            return self.listofpoly == other.listofpoly and \
                   self.parent == other.parent
        return self.listofpoly[0] == other and \
            all(i is self.parent.base.zero for i in self.listofpoly[1:])

    def is_singular(self, x0):
        """
        Checks if the differential equation is singular at x0.
        """

        base = self.parent.base
        return x0 in roots(base.to_sympy(self.listofpoly[-1]), self.x)


class HolonomicFunction:
    r"""
    A Holonomic Function is a solution to a linear homogeneous ordinary
    differential equation with polynomial coefficients. This differential
    equation can also be represented by an annihilator i.e. a Differential
    Operator ``L`` such that :math:`L.f = 0`. For uniqueness of these functions,
    initial conditions can also be provided along with the annihilator.

    Explanation
    ===========

    Holonomic functions have closure properties and thus forms a ring.
    Given two Holonomic Functions f and g, their sum, product,
    integral and derivative is also a Holonomic Function.

    For ordinary points initial condition should be a vector of values of
    the derivatives i.e. :math:`[y(x_0), y'(x_0), y''(x_0) ... ]`.

    For regular singular points initial conditions can also be provided in this
    format:
    :math:`{s0: [C_0, C_1, ...], s1: [C^1_0, C^1_1, ...], ...}`
    where s0, s1, ... are the roots of indicial equation and vectors
    :math:`[C_0, C_1, ...], [C^0_0, C^0_1, ...], ...` are the corresponding initial
    terms of the associated power series. See Examples below.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
    >>> from sympy import QQ
    >>> from sympy import symbols, S
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')

    >>> p = HolonomicFunction(Dx - 1, x, 0, [1])  # e^x
    >>> q = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])  # sin(x)

    >>> p + q  # annihilator of e^x + sin(x)
    HolonomicFunction((-1) + (1)*Dx + (-1)*Dx**2 + (1)*Dx**3, x, 0, [1, 2, 1])

    >>> p * q  # annihilator of e^x * sin(x)
    HolonomicFunction((2) + (-2)*Dx + (1)*Dx**2, x, 0, [0, 1])

    An example of initial conditions for regular singular points,
    the indicial equation has only one root `1/2`.

    >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]})
    HolonomicFunction((-1/2) + (x)*Dx, x, 0, {1/2: [1]})

    >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]}).to_expr()
    sqrt(x)

    To plot a Holonomic Function, one can use `.evalf()` for numerical
    computation. Here's an example on `sin(x)**2/x` using numpy and matplotlib.

    >>> import sympy.holonomic # doctest: +SKIP
    >>> from sympy import var, sin # doctest: +SKIP
    >>> import matplotlib.pyplot as plt # doctest: +SKIP
    >>> import numpy as np # doctest: +SKIP
    >>> var("x") # doctest: +SKIP
    >>> r = np.linspace(1, 5, 100) # doctest: +SKIP
    >>> y = sympy.holonomic.expr_to_holonomic(sin(x)**2/x, x0=1).evalf(r) # doctest: +SKIP
    >>> plt.plot(r, y, label="holonomic function") # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

    """

    _op_priority = 20

    def __init__(self, annihilator, x, x0=0, y0=None):
        """

        Parameters
        ==========

        annihilator:
            Annihilator of the Holonomic Function, represented by a
            `DifferentialOperator` object.
        x:
            Variable of the function.
        x0:
            The point at which initial conditions are stored.
            Generally an integer.
        y0:
            The initial condition. The proper format for the initial condition
            is described in class docstring. To make the function unique,
            length of the vector `y0` should be equal to or greater than the
            order of differential equation.
        """

        # initial condition
        self.y0 = y0
        # the point for initial conditions, default is zero.
        self.x0 = x0
        # differential operator L such that L.f = 0
        self.annihilator = annihilator
        self.x = x

    def __str__(self):
        if self._have_init_cond():
            str_sol = 'HolonomicFunction(%s, %s, %s, %s)' % (str(self.annihilator),\
                sstr(self.x), sstr(self.x0), sstr(self.y0))
        else:
            str_sol = 'HolonomicFunction(%s, %s)' % (str(self.annihilator),\
                sstr(self.x))

        return str_sol

    __repr__ = __str__

    def unify(self, other):
        """
        Unifies the base polynomial ring of a given two Holonomic
        Functions.
        """

        R1 = self.annihilator.parent.base
        R2 = other.annihilator.parent.base

        dom1 = R1.dom
        dom2 = R2.dom

        if R1 == R2:
            return (self, other)

        R = (dom1.unify(dom2)).old_poly_ring(self.x)

        newparent, _ = DifferentialOperators(R, str(self.annihilator.parent.gen_symbol))

        sol1 = [R1.to_sympy(i) for i in self.annihilator.listofpoly]
        sol2 = [R2.to_sympy(i) for i in other.annihilator.listofpoly]

        sol1 = DifferentialOperator(sol1, newparent)
        sol2 = DifferentialOperator(sol2, newparent)

        sol1 = HolonomicFunction(sol1, self.x, self.x0, self.y0)
        sol2 = HolonomicFunction(sol2, other.x, other.x0, other.y0)

        return (sol1, sol2)

    def is_singularics(self):
        """
        Returns True if the function have singular initial condition
        in the dictionary format.

        Returns False if the function have ordinary initial condition
        in the list format.

        Returns None for all other cases.
        """

        if isinstance(self.y0, dict):
            return True
        elif isinstance(self.y0, list):
            return False

    def _have_init_cond(self):
        """
        Checks if the function have initial condition.
        """
        return bool(self.y0)

    def _singularics_to_ord(self):
        """
        Converts a singular initial condition to ordinary if possible.
        """
        a = list(self.y0)[0]
        b = self.y0[a]

        if len(self.y0) == 1 and a == int(a) and a > 0:
            a = int(a)
            y0 = [S.Zero] * a
            y0 += [j * factorial(a + i) for i, j in enumerate(b)]

            return HolonomicFunction(self.annihilator, self.x, self.x0, y0)

    def __add__(self, other):
        # if the ground domains are different
        if self.annihilator.parent.base != other.annihilator.parent.base:
            a, b = self.unify(other)
            return a + b

        deg1 = self.annihilator.order
        deg2 = other.annihilator.order
        dim = max(deg1, deg2)
        R = self.annihilator.parent.base
        K = R.get_field()

        rowsself = [self.annihilator]
        rowsother = [other.annihilator]
        gen = self.annihilator.parent.derivative_operator

        # constructing annihilators up to order dim
        for i in range(dim - deg1):
            diff1 = (gen * rowsself[-1])
            rowsself.append(diff1)

        for i in range(dim - deg2):
            diff2 = (gen * rowsother[-1])
            rowsother.append(diff2)

        row = rowsself + rowsother

        # constructing the matrix of the ansatz
        r = []

        for expr in row:
            p = []
            for i in range(dim + 1):
                if i >= len(expr.listofpoly):
                    p.append(K.zero)
                else:
                    p.append(K.new(expr.listofpoly[i].to_list()))
            r.append(p)

        # solving the linear system using gauss jordan solver
        r = DomainMatrix(r, (len(row), dim+1), K).transpose()
        homosys = DomainMatrix.zeros((dim+1, 1), K)
        sol = _find_nonzero_solution(r, homosys)

        # if a solution is not obtained then increasing the order by 1 in each
        # iteration
        while sol.is_zero_matrix:
            dim += 1

            diff1 = (gen * rowsself[-1])
            rowsself.append(diff1)

            diff2 = (gen * rowsother[-1])
            rowsother.append(diff2)

            row = rowsself + rowsother
            r = []

            for expr in row:
                p = []
                for i in range(dim + 1):
                    if i >= len(expr.listofpoly):
                        p.append(K.zero)
                    else:
                        p.append(K.new(expr.listofpoly[i].to_list()))
                r.append(p)

            # solving the linear system using gauss jordan solver
            r = DomainMatrix(r, (len(row), dim+1), K).transpose()
            homosys = DomainMatrix.zeros((dim+1, 1), K)
            sol = _find_nonzero_solution(r, homosys)

        # taking only the coefficients needed to multiply with `self`
        # can be also be done the other way by taking R.H.S and multiplying with
        # `other`
        sol = sol.flat()[:dim + 1 - deg1]
        sol1 = _normalize(sol, self.annihilator.parent)
        # annihilator of the solution
        sol = sol1 * (self.annihilator)
        sol = _normalize(sol.listofpoly, self.annihilator.parent, negative=False)

        if not (self._have_init_cond() and other._have_init_cond()):
            return HolonomicFunction(sol, self.x)

        # both the functions have ordinary initial conditions
        if self.is_singularics() == False and other.is_singularics() == False:

            # directly add the corresponding value
            if self.x0 == other.x0:
                # try to extended the initial conditions
                # using the annihilator
                y1 = _extend_y0(self, sol.order)
                y2 = _extend_y0(other, sol.order)
                y0 = [a + b for a, b in zip(y1, y2)]
                return HolonomicFunction(sol, self.x, self.x0, y0)

            # change the initial conditions to a same point
            selfat0 = self.annihilator.is_singular(0)
            otherat0 = other.annihilator.is_singular(0)
            if self.x0 == 0 and not selfat0 and not otherat0:
                return self + other.change_ics(0)
            if other.x0 == 0 and not selfat0 and not otherat0:
                return self.change_ics(0) + other

            selfatx0 = self.annihilator.is_singular(self.x0)
            otheratx0 = other.annihilator.is_singular(self.x0)
            if not selfatx0 and not otheratx0:
                return self + other.change_ics(self.x0)
            return self.change_ics(other.x0) + other

        if self.x0 != other.x0:
            return HolonomicFunction(sol, self.x)

        # if the functions have singular_ics
        y1 = None
        y2 = None

        if self.is_singularics() == False and other.is_singularics() == True:
            # convert the ordinary initial condition to singular.
            _y0 = [j / factorial(i) for i, j in enumerate(self.y0)]
            y1 = {S.Zero: _y0}
            y2 = other.y0
        elif self.is_singularics() == True and other.is_singularics() == False:
            _y0 = [j / factorial(i) for i, j in enumerate(other.y0)]
            y1 = self.y0
            y2 = {S.Zero: _y0}
        elif self.is_singularics() == True and other.is_singularics() == True:
            y1 = self.y0
            y2 = other.y0

        # computing singular initial condition for the result
        # taking union of the series terms of both functions
        y0 = {}
        for i in y1:
            # add corresponding initial terms if the power
            # on `x` is same
            if i in y2:
                y0[i] = [a + b for a, b in zip(y1[i], y2[i])]
            else:
                y0[i] = y1[i]
        for i in y2:
            if i not in y1:
                y0[i] = y2[i]
        return HolonomicFunction(sol, self.x, self.x0, y0)

    def integrate(self, limits, initcond=False):
        """
        Integrates the given holonomic function.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).integrate((x, 0, x))  # e^x - 1
        HolonomicFunction((-1)*Dx + (1)*Dx**2, x, 0, [0, 1])
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [1, 0]).integrate((x, 0, x))
        HolonomicFunction((1)*Dx + (1)*Dx**3, x, 0, [0, 1, 0])
        """

        # to get the annihilator, just multiply by Dx from right
        D = self.annihilator.parent.derivative_operator

        # if the function have initial conditions of the series format
        if self.is_singularics() == True:

            r = self._singularics_to_ord()
            if r:
                return r.integrate(limits, initcond=initcond)

            # computing singular initial condition for the function
            # produced after integration.
            y0 = {}
            for i in self.y0:
                c = self.y0[i]
                c2 = []
                for j, cj in enumerate(c):
                    if cj == 0:
                        c2.append(S.Zero)

                    # if power on `x` is -1, the integration becomes log(x)
                    # TODO: Implement this case
                    elif i + j + 1 == 0:
                        raise NotImplementedError("logarithmic terms in the series are not supported")
                    else:
                        c2.append(cj / S(i + j + 1))
                y0[i + 1] = c2

            if hasattr(limits, "__iter__"):
                raise NotImplementedError("Definite integration for singular initial conditions")

            return HolonomicFunction(self.annihilator * D, self.x, self.x0, y0)

        # if no initial conditions are available for the function
        if not self._have_init_cond():
            if initcond:
                return HolonomicFunction(self.annihilator * D, self.x, self.x0, [S.Zero])
            return HolonomicFunction(self.annihilator * D, self.x)

        # definite integral
        # initial conditions for the answer will be stored at point `a`,
        # where `a` is the lower limit of the integrand
        if hasattr(limits, "__iter__"):

            if len(limits) == 3 and limits[0] == self.x:
                x0 = self.x0
                a = limits[1]
                b = limits[2]
                definite = True

        else:
            definite = False

        y0 = [S.Zero]
        y0 += self.y0

        indefinite_integral = HolonomicFunction(self.annihilator * D, self.x, self.x0, y0)

        if not definite:
            return indefinite_integral

        # use evalf to get the values at `a`
        if x0 != a:
            try:
                indefinite_expr = indefinite_integral.to_expr()
            except (NotHyperSeriesError, NotPowerSeriesError):
                indefinite_expr = None

            if indefinite_expr:
                lower = indefinite_expr.subs(self.x, a)
                if isinstance(lower, NaN):
                    lower = indefinite_expr.limit(self.x, a)
            else:
                lower = indefinite_integral.evalf(a)

            if b == self.x:
                y0[0] = y0[0] - lower
                return HolonomicFunction(self.annihilator * D, self.x, x0, y0)

            elif S(b).is_Number:
                if indefinite_expr:
                    upper = indefinite_expr.subs(self.x, b)
                    if isinstance(upper, NaN):
                        upper = indefinite_expr.limit(self.x, b)
                else:
                    upper = indefinite_integral.evalf(b)

                return upper - lower


        # if the upper limit is `x`, the answer will be a function
        if b == self.x:
            return HolonomicFunction(self.annihilator * D, self.x, a, y0)

        # if the upper limits is a Number, a numerical value will be returned
        elif S(b).is_Number:
            try:
                s = HolonomicFunction(self.annihilator * D, self.x, a,\
                    y0).to_expr()
                indefinite = s.subs(self.x, b)
                if not isinstance(indefinite, NaN):
                    return indefinite
                else:
                    return s.limit(self.x, b)
            except (NotHyperSeriesError, NotPowerSeriesError):
                return HolonomicFunction(self.annihilator * D, self.x, a, y0).evalf(b)

        return HolonomicFunction(self.annihilator * D, self.x)

    def diff(self, *args, **kwargs):
        r"""
        Differentiation of the given Holonomic function.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).diff().to_expr()
        cos(x)
        >>> HolonomicFunction(Dx - 2, x, 0, [1]).diff().to_expr()
        2*exp(2*x)

        See Also
        ========

        integrate
        """
        kwargs.setdefault('evaluate', True)
        if args:
            if args[0] != self.x:
                return S.Zero
            elif len(args) == 2:
                sol = self
                for i in range(args[1]):
                    sol = sol.diff(args[0])
                return sol

        ann = self.annihilator

        # if the function is constant.
        if ann.listofpoly[0] == ann.parent.base.zero and ann.order == 1:
            return S.Zero

        # if the coefficient of y in the differential equation is zero.
        # a shifting is done to compute the answer in this case.
        elif ann.listofpoly[0] == ann.parent.base.zero:

            sol = DifferentialOperator(ann.listofpoly[1:], ann.parent)

            if self._have_init_cond():
                # if ordinary initial condition
                if self.is_singularics() == False:
                    return HolonomicFunction(sol, self.x, self.x0, self.y0[1:])
                # TODO: support for singular initial condition
                return HolonomicFunction(sol, self.x)
            else:
                return HolonomicFunction(sol, self.x)

        # the general algorithm
        R = ann.parent.base
        K = R.get_field()

        seq_dmf = [K.new(i.to_list()) for i in ann.listofpoly]

        # -y = a1*y'/a0 + a2*y''/a0 ... + an*y^n/a0
        rhs = [i / seq_dmf[0] for i in seq_dmf[1:]]
        rhs.insert(0, K.zero)

        # differentiate both lhs and rhs
        sol = _derivate_diff_eq(rhs, K)

        # add the term y' in lhs to rhs
        sol = _add_lists(sol, [K.zero, K.one])

        sol = _normalize(sol[1:], self.annihilator.parent, negative=False)

        if not self._have_init_cond() or self.is_singularics() == True:
            return HolonomicFunction(sol, self.x)

        y0 = _extend_y0(self, sol.order + 1)[1:]
        return HolonomicFunction(sol, self.x, self.x0, y0)

    def __eq__(self, other):
        if self.annihilator != other.annihilator or self.x != other.x:
            return False
        if self._have_init_cond() and other._have_init_cond():
            return self.x0 == other.x0 and self.y0 == other.y0
        return True

    def __mul__(self, other):
        ann_self = self.annihilator

        if not isinstance(other, HolonomicFunction):
            other = sympify(other)

            if other.has(self.x):
                raise NotImplementedError(" Can't multiply a HolonomicFunction and expressions/functions.")

            if not self._have_init_cond():
                return self
            y0 = _extend_y0(self, ann_self.order)
            y1 = [(Poly.new(j, self.x) * other).rep for j in y0]
            return HolonomicFunction(ann_self, self.x, self.x0, y1)

        if self.annihilator.parent.base != other.annihilator.parent.base:
            a, b = self.unify(other)
            return a * b

        ann_other = other.annihilator

        a = ann_self.order
        b = ann_other.order

        R = ann_self.parent.base
        K = R.get_field()

        list_self = [K.new(j.to_list()) for j in ann_self.listofpoly]
        list_other = [K.new(j.to_list()) for j in ann_other.listofpoly]

        # will be used to reduce the degree
        self_red = [-list_self[i] / list_self[a] for i in range(a)]

        other_red = [-list_other[i] / list_other[b] for i in range(b)]

        # coeff_mull[i][j] is the coefficient of Dx^i(f).Dx^j(g)
        coeff_mul = [[K.zero for i in range(b + 1)] for j in range(a + 1)]
        coeff_mul[0][0] = K.one

        # making the ansatz
        lin_sys_elements = [[coeff_mul[i][j] for i in range(a) for j in range(b)]]
        lin_sys = DomainMatrix(lin_sys_elements, (1, a*b), K).transpose()

        homo_sys = DomainMatrix.zeros((a*b, 1), K)

        sol = _find_nonzero_solution(lin_sys, homo_sys)

        # until a non trivial solution is found
        while sol.is_zero_matrix:

            # updating the coefficients Dx^i(f).Dx^j(g) for next degree
            for i in range(a - 1, -1, -1):
                for j in range(b - 1, -1, -1):
                    coeff_mul[i][j + 1] += coeff_mul[i][j]
                    coeff_mul[i + 1][j] += coeff_mul[i][j]
                    if isinstance(coeff_mul[i][j], K.dtype):
                        coeff_mul[i][j] = DMFdiff(coeff_mul[i][j], K)
                    else:
                        coeff_mul[i][j] = coeff_mul[i][j].diff(self.x)

            # reduce the terms to lower power using annihilators of f, g
            for i in range(a + 1):
                if coeff_mul[i][b].is_zero:
                    continue
                for j in range(b):
                    coeff_mul[i][j] += other_red[j] * coeff_mul[i][b]
                coeff_mul[i][b] = K.zero

            # not d2 + 1, as that is already covered in previous loop
            for j in range(b):
                if coeff_mul[a][j] == 0:
                    continue
                for i in range(a):
                    coeff_mul[i][j] += self_red[i] * coeff_mul[a][j]
                coeff_mul[a][j] = K.zero

            lin_sys_elements.append([coeff_mul[i][j] for i in range(a) for j in range(b)])
            lin_sys = DomainMatrix(lin_sys_elements, (len(lin_sys_elements), a*b), K).transpose()

            sol = _find_nonzero_solution(lin_sys, homo_sys)

        sol_ann = _normalize(sol.flat(), self.annihilator.parent, negative=False)

        if not (self._have_init_cond() and other._have_init_cond()):
            return HolonomicFunction(sol_ann, self.x)

        if self.is_singularics() == False and other.is_singularics() == False:

            # if both the conditions are at same point
            if self.x0 == other.x0:

                # try to find more initial conditions
                y0_self = _extend_y0(self, sol_ann.order)
                y0_other = _extend_y0(other, sol_ann.order)
                # h(x0) = f(x0) * g(x0)
                y0 = [y0_self[0] * y0_other[0]]

                # coefficient of Dx^j(f)*Dx^i(g) in Dx^i(fg)
                for i in range(1, min(len(y0_self), len(y0_other))):
                    coeff = [[0 for i in range(i + 1)] for j in range(i + 1)]
                    for j in range(i + 1):
                        for k in range(i + 1):
                            if j + k == i:
                                coeff[j][k] = binomial(i, j)

                    sol = 0
                    for j in range(i + 1):
                        for k in range(i + 1):
                            sol += coeff[j][k]* y0_self[j] * y0_other[k]

                    y0.append(sol)

                return HolonomicFunction(sol_ann, self.x, self.x0, y0)

            # if the points are different, consider one
            selfat0 = self.annihilator.is_singular(0)
            otherat0 = other.annihilator.is_singular(0)

            if self.x0 == 0 and not selfat0 and not otherat0:
                return self * other.change_ics(0)
            if other.x0 == 0 and not selfat0 and not otherat0:
                return self.change_ics(0) * other

            selfatx0 = self.annihilator.is_singular(self.x0)
            otheratx0 = other.annihilator.is_singular(self.x0)
            if not selfatx0 and not otheratx0:
                return self * other.change_ics(self.x0)
            return self.change_ics(other.x0) * other

        if self.x0 != other.x0:
            return HolonomicFunction(sol_ann, self.x)

        # if the functions have singular_ics
        y1 = None
        y2 = None

        if self.is_singularics() == False and other.is_singularics() == True:
            _y0 = [j / factorial(i) for i, j in enumerate(self.y0)]
            y1 = {S.Zero: _y0}
            y2 = other.y0
        elif self.is_singularics() == True and other.is_singularics() == False:
            _y0 = [j / factorial(i) for i, j in enumerate(other.y0)]
            y1 = self.y0
            y2 = {S.Zero: _y0}
        elif self.is_singularics() == True and other.is_singularics() == True:
            y1 = self.y0
            y2 = other.y0

        y0 = {}
        # multiply every possible pair of the series terms
        for i in y1:
            for j in y2:
                k = min(len(y1[i]), len(y2[j]))
                c = [sum((y1[i][b] * y2[j][a - b] for b in range(a + 1)),
                         start=S.Zero) for a in range(k)]
                if not i + j in y0:
                    y0[i + j] = c
                else:
                    y0[i + j] = [a + b for a, b in zip(c, y0[i + j])]
        return HolonomicFunction(sol_ann, self.x, self.x0, y0)

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return self * -1 + other

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        return self * (S.One / other)

    def __pow__(self, n):
        if self.annihilator.order <= 1:
            ann = self.annihilator
            parent = ann.parent

            if self.y0 is None:
                y0 = None
            else:
                y0 = [list(self.y0)[0] ** n]

            p0 = ann.listofpoly[0]
            p1 = ann.listofpoly[1]

            p0 = (Poly.new(p0, self.x) * n).rep

            sol = [parent.base.to_sympy(i) for i in [p0, p1]]
            dd = DifferentialOperator(sol, parent)
            return HolonomicFunction(dd, self.x, self.x0, y0)
        if n < 0:
            raise NotHolonomicError("Negative Power on a Holonomic Function")
        Dx = self.annihilator.parent.derivative_operator
        result = HolonomicFunction(Dx, self.x, S.Zero, [S.One])
        if n == 0:
            return result
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def degree(self):
        """
        Returns the highest power of `x` in the annihilator.
        """
        return max(i.degree() for i in self.annihilator.listofpoly)

    def composition(self, expr, *args, **kwargs):
        """
        Returns function after composition of a holonomic
        function with an algebraic function. The method cannot compute
        initial conditions for the result by itself, so they can be also be
        provided.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x).composition(x**2, 0, [1])  # e^(x**2)
        HolonomicFunction((-2*x) + (1)*Dx, x, 0, [1])
        >>> HolonomicFunction(Dx**2 + 1, x).composition(x**2 - 1, 1, [1, 0])
        HolonomicFunction((4*x**3) + (-1)*Dx + (x)*Dx**2, x, 1, [1, 0])

        See Also
        ========

        from_hyper
        """

        R = self.annihilator.parent
        a = self.annihilator.order
        diff = expr.diff(self.x)
        listofpoly = self.annihilator.listofpoly

        for i, j in enumerate(listofpoly):
            if isinstance(j, self.annihilator.parent.base.dtype):
                listofpoly[i] = self.annihilator.parent.base.to_sympy(j)

        r = listofpoly[a].subs({self.x:expr})
        subs = [-listofpoly[i].subs({self.x:expr}) / r for i in range (a)]
        coeffs = [S.Zero for i in range(a)]  # coeffs[i] == coeff of (D^i f)(a) in D^k (f(a))
        coeffs[0] = S.One
        system = [coeffs]
        homogeneous = Matrix([[S.Zero for i in range(a)]]).transpose()
        while True:
            coeffs_next = [p.diff(self.x) for p in coeffs]
            for i in range(a - 1):
                coeffs_next[i + 1] += (coeffs[i] * diff)
            for i in range(a):
                coeffs_next[i] += (coeffs[-1] * subs[i] * diff)
            coeffs = coeffs_next
            # check for linear relations
            system.append(coeffs)
            sol, taus = (Matrix(system).transpose()
                ).gauss_jordan_solve(homogeneous)
            if sol.is_zero_matrix is not True:
                break

        tau = list(taus)[0]
        sol = sol.subs(tau, 1)
        sol = _normalize(sol[0:], R, negative=False)

        # if initial conditions are given for the resulting function
        if args:
            return HolonomicFunction(sol, self.x, args[0], args[1])
        return HolonomicFunction(sol, self.x)

    def to_sequence(self, lb=True):
        r"""
        Finds recurrence relation for the coefficients in the series expansion
        of the function about :math:`x_0`, where :math:`x_0` is the point at
        which the initial condition is stored.

        Explanation
        ===========

        If the point :math:`x_0` is ordinary, solution of the form :math:`[(R, n_0)]`
        is returned. Where :math:`R` is the recurrence relation and :math:`n_0` is the
        smallest ``n`` for which the recurrence holds true.

        If the point :math:`x_0` is regular singular, a list of solutions in
        the format :math:`(R, p, n_0)` is returned, i.e. `[(R, p, n_0), ... ]`.
        Each tuple in this vector represents a recurrence relation :math:`R`
        associated with a root of the indicial equation ``p``. Conditions of
        a different format can also be provided in this case, see the
        docstring of HolonomicFunction class.

        If it's not possible to numerically compute a initial condition,
        it is returned as a symbol :math:`C_j`, denoting the coefficient of
        :math:`(x - x_0)^j` in the power series about :math:`x_0`.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols, S
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).to_sequence()
        [(HolonomicSequence((-1) + (n + 1)Sn, n), u(0) = 1, 0)]
        >>> HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1]).to_sequence()
        [(HolonomicSequence((n**2) + (n**2 + n)Sn, n), u(0) = 0, u(1) = 1, u(2) = -1/2, 2)]
        >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]}).to_sequence()
        [(HolonomicSequence((n), n), u(0) = 1, 1/2, 1)]

        See Also
        ========

        HolonomicFunction.series

        References
        ==========

        .. [1] https://hal.inria.fr/inria-00070025/document
        .. [2] https://www3.risc.jku.at/publications/download/risc_2244/DIPLFORM.pdf

        """

        if self.x0 != 0:
            return self.shift_x(self.x0).to_sequence()

        # check whether a power series exists if the point is singular
        if self.annihilator.is_singular(self.x0):
            return self._frobenius(lb=lb)

        dict1 = {}
        n = Symbol('n', integer=True)
        dom = self.annihilator.parent.base.dom
        R, _ = RecurrenceOperators(dom.old_poly_ring(n), 'Sn')

        # substituting each term of the form `x^k Dx^j` in the
        # annihilator, according to the formula below:
        # x^k Dx^j = Sum(rf(n + 1 - k, j) * a(n + j - k) * x^n, (n, k, oo))
        # for explanation see [2].
        for i, j in enumerate(self.annihilator.listofpoly):

            listofdmp = j.all_coeffs()
            degree = len(listofdmp) - 1

            for k in range(degree + 1):
                coeff = listofdmp[degree - k]

                if coeff == 0:
                    continue

                if (i - k, k) in dict1:
                    dict1[(i - k, k)] += (dom.to_sympy(coeff) * rf(n - k + 1, i))
                else:
                    dict1[(i - k, k)] = (dom.to_sympy(coeff) * rf(n - k + 1, i))


        sol = []
        keylist = [i[0] for i in dict1]
        lower = min(keylist)
        upper = max(keylist)
        degree = self.degree()

        # the recurrence relation holds for all values of
        # n greater than smallest_n, i.e. n >= smallest_n
        smallest_n = lower + degree
        dummys = {}
        eqs = []
        unknowns = []

        # an appropriate shift of the recurrence
        for j in range(lower, upper + 1):
            if j in keylist:
                temp = sum((v.subs(n, n - lower)
                           for k, v in dict1.items() if k[0] == j),
                           start=S.Zero)
                sol.append(temp)
            else:
                sol.append(S.Zero)

        # the recurrence relation
        sol = RecurrenceOperator(sol, R)

        # computing the initial conditions for recurrence
        order = sol.order
        all_roots = roots(R.base.to_sympy(sol.listofpoly[-1]), n, filter='Z')
        all_roots = all_roots.keys()

        if all_roots:
            max_root = max(all_roots) + 1
            smallest_n = max(max_root, smallest_n)
        order += smallest_n

        y0 = _extend_y0(self, order)
        # u(n) = y^n(0)/factorial(n)
        u0 = [j / factorial(i) for i, j in enumerate(y0)]

        # if sufficient conditions can't be computed then
        # try to use the series method i.e.
        # equate the coefficients of x^k in the equation formed by
        # substituting the series in differential equation, to zero.
        if len(u0) < order:

            for i in range(degree):
                eq = S.Zero

                for j in dict1:

                    if i + j[0] < 0:
                        dummys[i + j[0]] = S.Zero

                    elif i + j[0] < len(u0):
                        dummys[i + j[0]] = u0[i + j[0]]

                    elif not i + j[0] in dummys:
                        dummys[i + j[0]] = Symbol('C_%s' %(i + j[0]))
                        unknowns.append(dummys[i + j[0]])

                    if j[1] <= i:
                        eq += dict1[j].subs(n, i) * dummys[i + j[0]]

                eqs.append(eq)

            # solve the system of equations formed
            soleqs = solve(eqs, *unknowns)

            if isinstance(soleqs, dict):

                for i in range(len(u0), order):

                    if i not in dummys:
                        dummys[i] = Symbol('C_%s' %i)

                    if dummys[i] in soleqs:
                        u0.append(soleqs[dummys[i]])

                    else:
                        u0.append(dummys[i])

                if lb:
                    return [(HolonomicSequence(sol, u0), smallest_n)]
                return [HolonomicSequence(sol, u0)]

            for i in range(len(u0), order):

                if i not in dummys:
                    dummys[i] = Symbol('C_%s' %i)

                s = False
                for j in soleqs:
                    if dummys[i] in j:
                        u0.append(j[dummys[i]])
                        s = True
                if not s:
                    u0.append(dummys[i])

        if lb:
            return [(HolonomicSequence(sol, u0), smallest_n)]

        return [HolonomicSequence(sol, u0)]

    def _frobenius(self, lb=True):
        # compute the roots of indicial equation
        indicialroots = self._indicial()

        reals = []
        compl = []
        for i in ordered(indicialroots.keys()):
            if i.is_real:
                reals.extend([i] * indicialroots[i])
            else:
                a, b = i.as_real_imag()
                compl.extend([(i, a, b)] * indicialroots[i])

        # sort the roots for a fixed ordering of solution
        compl.sort(key=lambda x : x[1])
        compl.sort(key=lambda x : x[2])
        reals.sort()

        # grouping the roots, roots differ by an integer are put in the same group.
        grp = []

        for i in reals:
            if len(grp) == 0:
                grp.append([i])
                continue
            for j in grp:
                if int_valued(j[0] - i):
                    j.append(i)
                    break
            else:
                grp.append([i])

        # True if none of the roots differ by an integer i.e.
        # each element in group have only one member
        independent = all(len(i) == 1 for i in grp)

        allpos = all(i >= 0 for i in reals)
        allint = all(int_valued(i) for i in reals)

        # if initial conditions are provided
        # then use them.
        if self.is_singularics() == True:
            rootstoconsider = []
            for i in ordered(self.y0.keys()):
                for j in ordered(indicialroots.keys()):
                    if equal_valued(j, i):
                        rootstoconsider.append(i)

        elif allpos and allint:
            rootstoconsider = [min(reals)]

        elif independent:
            rootstoconsider = [i[0] for i in grp] + [j[0] for j in compl]

        elif not allint:
            rootstoconsider = [i for i in reals if not int(i) == i]

        elif not allpos:

            if not self._have_init_cond() or S(self.y0[0]).is_finite ==  False:
                rootstoconsider = [min(reals)]

            else:
                posroots = [i for i in reals if i >= 0]
                rootstoconsider = [min(posroots)]

        n = Symbol('n', integer=True)
        dom = self.annihilator.parent.base.dom
        R, _ = RecurrenceOperators(dom.old_poly_ring(n), 'Sn')

        finalsol = []
        char = ord('C')

        for p in rootstoconsider:
            dict1 = {}

            for i, j in enumerate(self.annihilator.listofpoly):

                listofdmp = j.all_coeffs()
                degree = len(listofdmp) - 1

                for k in range(degree + 1):
                    coeff = listofdmp[degree - k]

                    if coeff == 0:
                        continue

                    if (i - k, k - i) in dict1:
                        dict1[(i - k, k - i)] += (dom.to_sympy(coeff) * rf(n - k + 1 + p, i))
                    else:
                        dict1[(i - k, k - i)] = (dom.to_sympy(coeff) * rf(n - k + 1 + p, i))

            sol = []
            keylist = [i[0] for i in dict1]
            lower = min(keylist)
            upper = max(keylist)
            degree = max(i[1] for i in dict1)
            degree2 = min(i[1] for i in dict1)

            smallest_n = lower + degree
            dummys = {}
            eqs = []
            unknowns = []

            for j in range(lower, upper + 1):
                if j in keylist:
                    temp = sum((v.subs(n, n - lower)
                               for k, v in dict1.items() if k[0] == j),
                               start=S.Zero)
                    sol.append(temp)
                else:
                    sol.append(S.Zero)

            # the recurrence relation
            sol = RecurrenceOperator(sol, R)

            # computing the initial conditions for recurrence
            order = sol.order
            all_roots = roots(R.base.to_sympy(sol.listofpoly[-1]), n, filter='Z')
            all_roots = all_roots.keys()

            if all_roots:
                max_root = max(all_roots) + 1
                smallest_n = max(max_root, smallest_n)
            order += smallest_n

            u0 = []

            if self.is_singularics() == True:
                u0 = self.y0[p]

            elif self.is_singularics() == False and p >= 0 and int(p) == p and len(rootstoconsider) == 1:
                y0 = _extend_y0(self, order + int(p))
                # u(n) = y^n(0)/factorial(n)
                if len(y0) > int(p):
                    u0 = [y0[i] / factorial(i) for i in range(int(p), len(y0))]

            if len(u0) < order:

                for i in range(degree2, degree):
                    eq = S.Zero

                    for j in dict1:
                        if i + j[0] < 0:
                            dummys[i + j[0]] = S.Zero

                        elif i + j[0] < len(u0):
                            dummys[i + j[0]] = u0[i + j[0]]

                        elif not i + j[0] in dummys:
                            letter = chr(char) + '_%s' %(i + j[0])
                            dummys[i + j[0]] = Symbol(letter)
                            unknowns.append(dummys[i + j[0]])

                        if j[1] <= i:
                            eq += dict1[j].subs(n, i) * dummys[i + j[0]]

                    eqs.append(eq)

                # solve the system of equations formed
                soleqs = solve(eqs, *unknowns)

                if isinstance(soleqs, dict):

                    for i in range(len(u0), order):

                        if i not in dummys:
                            letter = chr(char) + '_%s' %i
                            dummys[i] = Symbol(letter)

                        if dummys[i] in soleqs:
                            u0.append(soleqs[dummys[i]])

                        else:
                            u0.append(dummys[i])

                    if lb:
                        finalsol.append((HolonomicSequence(sol, u0), p, smallest_n))
                        continue
                    else:
                        finalsol.append((HolonomicSequence(sol, u0), p))
                        continue

                for i in range(len(u0), order):

                    if i not in dummys:
                        letter = chr(char) + '_%s' %i
                        dummys[i] = Symbol(letter)

                    s = False
                    for j in soleqs:
                        if dummys[i] in j:
                            u0.append(j[dummys[i]])
                            s = True
                    if not s:
                        u0.append(dummys[i])
            if lb:
                finalsol.append((HolonomicSequence(sol, u0), p, smallest_n))

            else:
                finalsol.append((HolonomicSequence(sol, u0), p))
            char += 1
        return finalsol

    def series(self, n=6, coefficient=False, order=True, _recur=None):
        r"""
        Finds the power series expansion of given holonomic function about :math:`x_0`.

        Explanation
        ===========

        A list of series might be returned if :math:`x_0` is a regular point with
        multiple roots of the indicial equation.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).series()  # e^x
        1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).series(n=8)  # sin(x)
        x - x**3/6 + x**5/120 - x**7/5040 + O(x**8)

        See Also
        ========

        HolonomicFunction.to_sequence
        """

        if _recur is None:
            recurrence = self.to_sequence()
        else:
            recurrence = _recur

        if isinstance(recurrence, tuple) and len(recurrence) == 2:
            recurrence = recurrence[0]
            constantpower = 0
        elif isinstance(recurrence, tuple) and len(recurrence) == 3:
            constantpower = recurrence[1]
            recurrence = recurrence[0]

        elif len(recurrence) == 1 and len(recurrence[0]) == 2:
            recurrence = recurrence[0][0]
            constantpower = 0
        elif len(recurrence) == 1 and len(recurrence[0]) == 3:
            constantpower = recurrence[0][1]
            recurrence = recurrence[0][0]
        else:
            return [self.series(_recur=i) for i in recurrence]

        n = n - int(constantpower)
        l = len(recurrence.u0) - 1
        k = recurrence.recurrence.order
        x = self.x
        x0 = self.x0
        seq_dmp = recurrence.recurrence.listofpoly
        R = recurrence.recurrence.parent.base
        K = R.get_field()
        seq = [K.new(j.to_list()) for j in seq_dmp]
        sub = [-seq[i] / seq[k] for i in range(k)]
        sol = list(recurrence.u0)

        if l + 1 < n:
            # use the initial conditions to find the next term
            for i in range(l + 1 - k, n - k):
                coeff = sum((DMFsubs(sub[j], i) * sol[i + j]
                            for j in range(k) if i + j >= 0), start=S.Zero)
                sol.append(coeff)

        if coefficient:
            return sol

        ser = sum((x**(i + constantpower) * j for i, j in enumerate(sol)),
                  start=S.Zero)
        if order:
            ser += Order(x**(n + int(constantpower)), x)
        if x0 != 0:
            return ser.subs(x, x - x0)
        return ser

    def _indicial(self):
        """
        Computes roots of the Indicial equation.
        """

        if self.x0 != 0:
            return self.shift_x(self.x0)._indicial()

        list_coeff = self.annihilator.listofpoly
        R = self.annihilator.parent.base
        x = self.x
        s = R.zero
        y = R.one

        def _pole_degree(poly):
            root_all = roots(R.to_sympy(poly), x, filter='Z')
            if 0 in root_all.keys():
                return root_all[0]
            else:
                return 0

        degree = max(j.degree() for j in list_coeff)
        inf = 10 * (max(1, degree) + max(1, self.annihilator.order))

        deg = lambda q: inf if q.is_zero else _pole_degree(q)
        b = min(deg(q) - j for j, q in enumerate(list_coeff))

        for i, j in enumerate(list_coeff):
            listofdmp = j.all_coeffs()
            degree = len(listofdmp) - 1
            if 0 <= i + b <= degree:
                s = s + listofdmp[degree - i - b] * y
            y *= R.from_sympy(x - i)

        return roots(R.to_sympy(s), x)

    def evalf(self, points, method='RK4', h=0.05, derivatives=False):
        r"""
        Finds numerical value of a holonomic function using numerical methods.
        (RK4 by default). A set of points (real or complex) must be provided
        which will be the path for the numerical integration.

        Explanation
        ===========

        The path should be given as a list :math:`[x_1, x_2, \dots x_n]`. The numerical
        values will be computed at each point in this order
        :math:`x_1 \rightarrow x_2 \rightarrow x_3 \dots \rightarrow x_n`.

        Returns values of the function at :math:`x_1, x_2, \dots x_n` in a list.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')

        A straight line on the real axis from (0 to 1)

        >>> r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        Runge-Kutta 4th order on e^x from 0.1 to 1.
        Exact solution at 1 is 2.71828182845905

        >>> HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r)
        [1.10517083333333, 1.22140257085069, 1.34985849706254, 1.49182424008069,
        1.64872063859684, 1.82211796209193, 2.01375162659678, 2.22553956329232,
        2.45960141378007, 2.71827974413517]

        Euler's method for the same

        >>> HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r, method='Euler')
        [1.1, 1.21, 1.331, 1.4641, 1.61051, 1.771561, 1.9487171, 2.14358881,
        2.357947691, 2.5937424601]

        One can also observe that the value obtained using Runge-Kutta 4th order
        is much more accurate than Euler's method.
        """

        from sympy.holonomic.numerical import _evalf
        lp = False

        # if a point `b` is given instead of a mesh
        if not hasattr(points, "__iter__"):
            lp = True
            b = S(points)
            if self.x0 == b:
                return _evalf(self, [b], method=method, derivatives=derivatives)[-1]

            if not b.is_Number:
                raise NotImplementedError

            a = self.x0
            if a > b:
                h = -h
            n = int((b - a) / h)
            points = [a + h]
            for i in range(n - 1):
                points.append(points[-1] + h)

        for i in roots(self.annihilator.parent.base.to_sympy(self.annihilator.listofpoly[-1]), self.x):
            if i == self.x0 or i in points:
                raise SingularityError(self, i)

        if lp:
            return _evalf(self, points, method=method, derivatives=derivatives)[-1]
        return _evalf(self, points, method=method, derivatives=derivatives)

    def change_x(self, z):
        """
        Changes only the variable of Holonomic Function, for internal
        purposes. For composition use HolonomicFunction.composition()
        """

        dom = self.annihilator.parent.base.dom
        R = dom.old_poly_ring(z)
        parent, _ = DifferentialOperators(R, 'Dx')
        sol = [R(j.to_list()) for j in self.annihilator.listofpoly]
        sol =  DifferentialOperator(sol, parent)
        return HolonomicFunction(sol, z, self.x0, self.y0)

    def shift_x(self, a):
        """
        Substitute `x + a` for `x`.
        """

        x = self.x
        listaftershift = self.annihilator.listofpoly
        base = self.annihilator.parent.base

        sol = [base.from_sympy(base.to_sympy(i).subs(x, x + a)) for i in listaftershift]
        sol = DifferentialOperator(sol, self.annihilator.parent)
        x0 = self.x0 - a
        if not self._have_init_cond():
            return HolonomicFunction(sol, x)
        return HolonomicFunction(sol, x, x0, self.y0)

    def to_hyper(self, as_list=False, _recur=None):
        r"""
        Returns a hypergeometric function (or linear combination of them)
        representing the given holonomic function.

        Explanation
        ===========

        Returns an answer of the form:
        `a_1 \cdot x^{b_1} \cdot{hyper()} + a_2 \cdot x^{b_2} \cdot{hyper()} \dots`

        This is very useful as one can now use ``hyperexpand`` to find the
        symbolic expressions/functions.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> # sin(x)
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).to_hyper()
        x*hyper((), (3/2,), -x**2/4)
        >>> # exp(x)
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).to_hyper()
        hyper((), (), x)

        See Also
        ========

        from_hyper, from_meijerg
        """

        if _recur is None:
            recurrence = self.to_sequence()
        else:
            recurrence = _recur

        if isinstance(recurrence, tuple) and len(recurrence) == 2:
            smallest_n = recurrence[1]
            recurrence = recurrence[0]
            constantpower = 0
        elif isinstance(recurrence, tuple) and len(recurrence) == 3:
            smallest_n = recurrence[2]
            constantpower = recurrence[1]
            recurrence = recurrence[0]
        elif len(recurrence) == 1 and len(recurrence[0]) == 2:
            smallest_n = recurrence[0][1]
            recurrence = recurrence[0][0]
            constantpower = 0
        elif len(recurrence) == 1 and len(recurrence[0]) == 3:
            smallest_n = recurrence[0][2]
            constantpower = recurrence[0][1]
            recurrence = recurrence[0][0]
        else:
            sol = self.to_hyper(as_list=as_list, _recur=recurrence[0])
            for i in recurrence[1:]:
                sol += self.to_hyper(as_list=as_list, _recur=i)
            return sol

        u0 = recurrence.u0
        r = recurrence.recurrence
        x = self.x
        x0 = self.x0

        # order of the recurrence relation
        m = r.order

        # when no recurrence exists, and the power series have finite terms
        if m == 0:
            nonzeroterms = roots(r.parent.base.to_sympy(r.listofpoly[0]), recurrence.n, filter='R')

            sol = S.Zero
            for j, i in enumerate(nonzeroterms):

                if i < 0 or not int_valued(i):
                    continue

                i = int(i)
                if i < len(u0):
                    if isinstance(u0[i], (PolyElement, FracElement)):
                        u0[i] = u0[i].as_expr()
                    sol += u0[i] * x**i

                else:
                    sol += Symbol('C_%s' %j) * x**i

            if isinstance(sol, (PolyElement, FracElement)):
                sol = sol.as_expr() * x**constantpower
            else:
                sol = sol * x**constantpower
            if as_list:
                if x0 != 0:
                    return [(sol.subs(x, x - x0), )]
                return [(sol, )]
            if x0 != 0:
                return sol.subs(x, x - x0)
            return sol

        if smallest_n + m > len(u0):
            raise NotImplementedError("Can't compute sufficient Initial Conditions")

        # check if the recurrence represents a hypergeometric series
        if any(i != r.parent.base.zero for i in r.listofpoly[1:-1]):
            raise NotHyperSeriesError(self, self.x0)

        a = r.listofpoly[0]
        b = r.listofpoly[-1]

        # the constant multiple of argument of hypergeometric function
        if isinstance(a.LC(), (PolyElement, FracElement)):
            c = - (S(a.LC().as_expr()) * m**(a.degree())) / (S(b.LC().as_expr()) * m**(b.degree()))
        else:
            c = - (S(a.LC()) * m**(a.degree())) / (S(b.LC()) * m**(b.degree()))

        sol = 0

        arg1 = roots(r.parent.base.to_sympy(a), recurrence.n)
        arg2 = roots(r.parent.base.to_sympy(b), recurrence.n)

        # iterate through the initial conditions to find
        # the hypergeometric representation of the given
        # function.
        # The answer will be a linear combination
        # of different hypergeometric series which satisfies
        # the recurrence.
        if as_list:
            listofsol = []
        for i in range(smallest_n + m):

            # if the recurrence relation doesn't hold for `n = i`,
            # then a Hypergeometric representation doesn't exist.
            # add the algebraic term a * x**i to the solution,
            # where a is u0[i]
            if i < smallest_n:
                if as_list:
                    listofsol.append(((S(u0[i]) * x**(i+constantpower)).subs(x, x-x0), ))
                else:
                    sol += S(u0[i]) * x**i
                continue

            # if the coefficient u0[i] is zero, then the
            # independent hypergeomtric series starting with
            # x**i is not a part of the answer.
            if S(u0[i]) == 0:
                continue

            ap = []
            bq = []

            # substitute m * n + i for n
            for k in ordered(arg1.keys()):
                ap.extend([nsimplify((i - k) / m)] * arg1[k])

            for k in ordered(arg2.keys()):
                bq.extend([nsimplify((i - k) / m)] * arg2[k])

            # convention of (k + 1) in the denominator
            if 1 in bq:
                bq.remove(1)
            else:
                ap.append(1)
            if as_list:
                listofsol.append(((S(u0[i])*x**(i+constantpower)).subs(x, x-x0), (hyper(ap, bq, c*x**m)).subs(x, x-x0)))
            else:
                sol += S(u0[i]) * hyper(ap, bq, c * x**m) * x**i
        if as_list:
            return listofsol
        sol = sol * x**constantpower
        if x0 != 0:
            return sol.subs(x, x - x0)

        return sol

    def to_expr(self):
        """
        Converts a Holonomic Function back to elementary functions.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols, S
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(x**2*Dx**2 + x*Dx + (x**2 - 1), x, 0, [0, S(1)/2]).to_expr()
        besselj(1, x)
        >>> HolonomicFunction((1 + x)*Dx**3 + Dx**2, x, 0, [1, 1, 1]).to_expr()
        x*log(x + 1) + log(x + 1) + 1

        """

        return hyperexpand(self.to_hyper()).simplify()

    def change_ics(self, b, lenics=None):
        """
        Changes the point `x0` to ``b`` for initial conditions.

        Examples
        ========

        >>> from sympy.holonomic import expr_to_holonomic
        >>> from sympy import symbols, sin, exp
        >>> x = symbols('x')

        >>> expr_to_holonomic(sin(x)).change_ics(1)
        HolonomicFunction((1) + (1)*Dx**2, x, 1, [sin(1), cos(1)])

        >>> expr_to_holonomic(exp(x)).change_ics(2)
        HolonomicFunction((-1) + (1)*Dx, x, 2, [exp(2)])
        """

        symbolic = True

        if lenics is None and len(self.y0) > self.annihilator.order:
            lenics = len(self.y0)
        dom = self.annihilator.parent.base.domain

        try:
            sol = expr_to_holonomic(self.to_expr(), x=self.x, x0=b, lenics=lenics, domain=dom)
        except (NotPowerSeriesError, NotHyperSeriesError):
            symbolic = False

        if symbolic and sol.x0 == b:
            return sol

        y0 = self.evalf(b, derivatives=True)
        return HolonomicFunction(self.annihilator, self.x, b, y0)

    def to_meijerg(self):
        """
        Returns a linear combination of Meijer G-functions.

        Examples
        ========

        >>> from sympy.holonomic import expr_to_holonomic
        >>> from sympy import sin, cos, hyperexpand, log, symbols
        >>> x = symbols('x')
        >>> hyperexpand(expr_to_holonomic(cos(x) + sin(x)).to_meijerg())
        sin(x) + cos(x)
        >>> hyperexpand(expr_to_holonomic(log(x)).to_meijerg()).simplify()
        log(x)

        See Also
        ========

        to_hyper
        """

        # convert to hypergeometric first
        rep = self.to_hyper(as_list=True)
        sol = S.Zero

        for i in rep:
            if len(i) == 1:
                sol += i[0]

            elif len(i) == 2:
                sol += i[0] * _hyper_to_meijerg(i[1])

        return sol


def from_hyper(func, x0=0, evalf=False):
    r"""
    Converts a hypergeometric function to holonomic.
    ``func`` is the Hypergeometric Function and ``x0`` is the point at
    which initial conditions are required.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import from_hyper
    >>> from sympy import symbols, hyper, S
    >>> x = symbols('x')
    >>> from_hyper(hyper([], [S(3)/2], x**2/4))
    HolonomicFunction((-x) + (2)*Dx + (x)*Dx**2, x, 1, [sinh(1), -sinh(1) + cosh(1)])
    """

    a = func.ap
    b = func.bq
    z = func.args[2]
    x = z.atoms(Symbol).pop()
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # generalized hypergeometric differential equation
    xDx = x*Dx
    r1 = 1
    for ai in a:  # XXX gives sympify error if Mul is used with list of all factors
        r1 *= xDx + ai
    xDx_1 = xDx - 1
    # r2 = Mul(*([Dx] + [xDx_1 + bi for bi in b]))  # XXX gives sympify error
    r2 = Dx
    for bi in b:
        r2 *= xDx_1 + bi
    sol = r1 - r2

    simp = hyperexpand(func)

    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    # if the function is known symbolically
    if not isinstance(simp, hyper):
        y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)
        while not y0:
            # if values don't exist at 0, then try to find initial
            # conditions at 1. If it doesn't exist at 1 too then
            # try 2 and so on.
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    if isinstance(simp, hyper):
        x0 = 1
        # use evalf if the function can't be simplified
        y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        return HolonomicFunction(sol, x).composition(z, x0, y0)

    return HolonomicFunction(sol, x).composition(z)


def from_meijerg(func, x0=0, evalf=False, initcond=True, domain=QQ):
    """
    Converts a Meijer G-function to Holonomic.
    ``func`` is the G-Function and ``x0`` is the point at
    which initial conditions are required.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import from_meijerg
    >>> from sympy import symbols, meijerg, S
    >>> x = symbols('x')
    >>> from_meijerg(meijerg(([], []), ([S(1)/2], [0]), x**2/4))
    HolonomicFunction((1) + (1)*Dx**2, x, 0, [0, 1/sqrt(pi)])
    """

    a = func.ap
    b = func.bq
    n = len(func.an)
    m = len(func.bm)
    p = len(a)
    z = func.args[2]
    x = z.atoms(Symbol).pop()
    R, Dx = DifferentialOperators(domain.old_poly_ring(x), 'Dx')

    # compute the differential equation satisfied by the
    # Meijer G-function.
    xDx = x*Dx
    xDx1 = xDx + 1
    r1 = x*(-1)**(m + n - p)
    for ai in a:  # XXX gives sympify error if args given in list
        r1 *= xDx1 - ai
    # r2 = Mul(*[xDx - bi for bi in b])  # gives sympify error
    r2 = 1
    for bi in b:
        r2 *= xDx - bi
    sol = r1 - r2

    if not initcond:
        return HolonomicFunction(sol, x).composition(z)

    simp = hyperexpand(func)

    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    # computing initial conditions
    if not isinstance(simp, meijerg):
        y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    if isinstance(simp, meijerg):
        x0 = 1
        y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    return HolonomicFunction(sol, x).composition(z)


x_1 = Dummy('x_1')
_lookup_table = None
domain_for_table = None
from sympy.integrals.meijerint import _mytype


def expr_to_holonomic(func, x=None, x0=0, y0=None, lenics=None, domain=None, initcond=True):
    """
    Converts a function or an expression to a holonomic function.

    Parameters
    ==========

    func:
        The expression to be converted.
    x:
        variable for the function.
    x0:
        point at which initial condition must be computed.
    y0:
        One can optionally provide initial condition if the method
        is not able to do it automatically.
    lenics:
        Number of terms in the initial condition. By default it is
        equal to the order of the annihilator.
    domain:
        Ground domain for the polynomials in ``x`` appearing as coefficients
        in the annihilator.
    initcond:
        Set it false if you do not want the initial conditions to be computed.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import expr_to_holonomic
    >>> from sympy import sin, exp, symbols
    >>> x = symbols('x')
    >>> expr_to_holonomic(sin(x))
    HolonomicFunction((1) + (1)*Dx**2, x, 0, [0, 1])
    >>> expr_to_holonomic(exp(x))
    HolonomicFunction((-1) + (1)*Dx, x, 0, [1])

    See Also
    ========

    sympy.integrals.meijerint._rewrite1, _convert_poly_rat_alg, _create_table
    """
    func = sympify(func)
    syms = func.free_symbols

    if not x:
        if len(syms) == 1:
            x= syms.pop()
        else:
            raise ValueError("Specify the variable for the function")
    elif x in syms:
        syms.remove(x)

    extra_syms = list(syms)

    if domain is None:
        if func.has(Float):
            domain = RR
        else:
            domain = QQ
        if len(extra_syms) != 0:
            domain = domain[extra_syms].get_field()

    # try to convert if the function is polynomial or rational
    solpoly = _convert_poly_rat_alg(func, x, x0=x0, y0=y0, lenics=lenics, domain=domain, initcond=initcond)
    if solpoly:
        return solpoly

    # create the lookup table
    global _lookup_table, domain_for_table
    if not _lookup_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)
    elif domain != domain_for_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)

    # use the table directly to convert to Holonomic
    if func.is_Function:
        f = func.subs(x, x_1)
        t = _mytype(f, x_1)
        if t in _lookup_table:
            l = _lookup_table[t]
            sol = l[0][1].change_x(x)
        else:
            sol = _convert_meijerint(func, x, initcond=False, domain=domain)
            if not sol:
                raise NotImplementedError
            if y0:
                sol.y0 = y0
            if y0 or not initcond:
                sol.x0 = x0
                return sol
            if not lenics:
                lenics = sol.annihilator.order
            _y0 = _find_conditions(func, x, x0, lenics)
            while not _y0:
                x0 += 1
                _y0 = _find_conditions(func, x, x0, lenics)
            return HolonomicFunction(sol.annihilator, x, x0, _y0)

        if y0 or not initcond:
            sol = sol.composition(func.args[0])
            if y0:
                sol.y0 = y0
            sol.x0 = x0
            return sol
        if not lenics:
            lenics = sol.annihilator.order

        _y0 = _find_conditions(func, x, x0, lenics)
        while not _y0:
            x0 += 1
            _y0 = _find_conditions(func, x, x0, lenics)
        return sol.composition(func.args[0], x0, _y0)

    # iterate through the expression recursively
    args = func.args
    f = func.func
    sol = expr_to_holonomic(args[0], x=x, initcond=False, domain=domain)

    if f is Add:
        for i in range(1, len(args)):
            sol += expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)

    elif f is Mul:
        for i in range(1, len(args)):
            sol *= expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)

    elif f is Pow:
        sol = sol**args[1]
    sol.x0 = x0
    if not sol:
        raise NotImplementedError
    if y0:
        sol.y0 = y0
    if y0 or not initcond:
        return sol
    if sol.y0:
        return sol
    if not lenics:
        lenics = sol.annihilator.order
    if sol.annihilator.is_singular(x0):
        r = sol._indicial()
        l = list(r)
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            g = func / (x - x0)**r
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r:singular_ics}
            return HolonomicFunction(sol.annihilator, x, x0, y0)

    _y0 = _find_conditions(func, x, x0, lenics)
    while not _y0:
        x0 += 1
        _y0 = _find_conditions(func, x, x0, lenics)

    return HolonomicFunction(sol.annihilator, x, x0, _y0)


## Some helper functions ##

def _normalize(list_of, parent, negative=True):
    """
    Normalize a given annihilator
    """

    num = []
    denom = []
    base = parent.base
    K = base.get_field()
    lcm_denom = base.from_sympy(S.One)
    list_of_coeff = []

    # convert polynomials to the elements of associated
    # fraction field
    for i, j in enumerate(list_of):
        if isinstance(j, base.dtype):
            list_of_coeff.append(K.new(j.to_list()))
        elif not isinstance(j, K.dtype):
            list_of_coeff.append(K.from_sympy(sympify(j)))
        else:
            list_of_coeff.append(j)

        # corresponding numerators of the sequence of polynomials
        num.append(list_of_coeff[i].numer())

        # corresponding denominators
        denom.append(list_of_coeff[i].denom())

    # lcm of denominators in the coefficients
    for i in denom:
        lcm_denom = i.lcm(lcm_denom)

    if negative:
        lcm_denom = -lcm_denom

    lcm_denom = K.new(lcm_denom.to_list())

    # multiply the coefficients with lcm
    for i, j in enumerate(list_of_coeff):
        list_of_coeff[i] = j * lcm_denom

    gcd_numer = base((list_of_coeff[-1].numer() / list_of_coeff[-1].denom()).to_list())

    # gcd of numerators in the coefficients
    for i in num:
        gcd_numer = i.gcd(gcd_numer)

    gcd_numer = K.new(gcd_numer.to_list())

    # divide all the coefficients by the gcd
    for i, j in enumerate(list_of_coeff):
        frac_ans = j / gcd_numer
        list_of_coeff[i] = base((frac_ans.numer() / frac_ans.denom()).to_list())

    return DifferentialOperator(list_of_coeff, parent)


def _derivate_diff_eq(listofpoly, K):
    """
    Let a differential equation a0(x)y(x) + a1(x)y'(x) + ... = 0
    where a0, a1,... are polynomials or rational functions. The function
    returns b0, b1, b2... such that the differential equation
    b0(x)y(x) + b1(x)y'(x) +... = 0 is formed after differentiating the
    former equation.
    """

    sol = []
    a = len(listofpoly) - 1
    sol.append(DMFdiff(listofpoly[0], K))

    for i, j in enumerate(listofpoly[1:]):
        sol.append(DMFdiff(j, K) + listofpoly[i])

    sol.append(listofpoly[a])
    return sol


def _hyper_to_meijerg(func):
    """
    Converts a `hyper` to meijerg.
    """
    ap = func.ap
    bq = func.bq

    if any(i <= 0 and int(i) == i for i in ap):
        return hyperexpand(func)

    z = func.args[2]

    # parameters of the `meijerg` function.
    an = (1 - i for i in ap)
    anp = ()
    bm = (S.Zero, )
    bmq = (1 - i for i in bq)

    k = S.One

    for i in bq:
        k = k * gamma(i)

    for i in ap:
        k = k / gamma(i)

    return k * meijerg(an, anp, bm, bmq, -z)


def _add_lists(list1, list2):
    """Takes polynomial sequences of two annihilators a and b and returns
    the list of polynomials of sum of a and b.
    """
    if len(list1) <= len(list2):
        sol = [a + b for a, b in zip(list1, list2)] + list2[len(list1):]
    else:
        sol = [a + b for a, b in zip(list1, list2)] + list1[len(list2):]
    return sol


def _extend_y0(Holonomic, n):
    """
    Tries to find more initial conditions by substituting the initial
    value point in the differential equation.
    """

    if Holonomic.annihilator.is_singular(Holonomic.x0) or Holonomic.is_singularics() == True:
        return Holonomic.y0

    annihilator = Holonomic.annihilator
    a = annihilator.order

    listofpoly = []

    y0 = Holonomic.y0
    R = annihilator.parent.base
    K = R.get_field()

    for j in annihilator.listofpoly:
        if isinstance(j, annihilator.parent.base.dtype):
            listofpoly.append(K.new(j.to_list()))

    if len(y0) < a or n <= len(y0):
        return y0
    list_red = [-listofpoly[i] / listofpoly[a]
                for i in range(a)]
    y1 = y0[:min(len(y0), a)]
    for _ in range(n - a):
        sol = 0
        for a, b in zip(y1, list_red):
            r = DMFsubs(b, Holonomic.x0)
            if not getattr(r, 'is_finite', True):
                return y0
            if isinstance(r, (PolyElement, FracElement)):
                r = r.as_expr()
            sol += a * r
        y1.append(sol)
        list_red = _derivate_diff_eq(list_red, K)
    return y0 + y1[len(y0):]


def DMFdiff(frac, K):
    # differentiate a DMF object represented as p/q
    if not isinstance(frac, DMF):
        return frac.diff()

    p = K.numer(frac)
    q = K.denom(frac)
    sol_num = - p * q.diff() + q * p.diff()
    sol_denom = q**2
    return K((sol_num.to_list(), sol_denom.to_list()))


def DMFsubs(frac, x0, mpm=False):
    # substitute the point x0 in DMF object of the form p/q
    if not isinstance(frac, DMF):
        return frac

    p = frac.num
    q = frac.den
    sol_p = S.Zero
    sol_q = S.Zero

    if mpm:
        from mpmath import mp

    for i, j in enumerate(reversed(p)):
        if mpm:
            j = sympify(j)._to_mpmath(mp.prec)
        sol_p += j * x0**i

    for i, j in enumerate(reversed(q)):
        if mpm:
            j = sympify(j)._to_mpmath(mp.prec)
        sol_q += j * x0**i

    if isinstance(sol_p, (PolyElement, FracElement)):
        sol_p = sol_p.as_expr()
    if isinstance(sol_q, (PolyElement, FracElement)):
        sol_q = sol_q.as_expr()

    return sol_p / sol_q


def _convert_poly_rat_alg(func, x, x0=0, y0=None, lenics=None, domain=QQ, initcond=True):
    """
    Converts polynomials, rationals and algebraic functions to holonomic.
    """

    ispoly = func.is_polynomial()
    if not ispoly:
        israt = func.is_rational_function()
    else:
        israt = True

    if not (ispoly or israt):
        basepoly, ratexp = func.as_base_exp()
        if basepoly.is_polynomial() and ratexp.is_Number:
            if isinstance(ratexp, Float):
                ratexp = nsimplify(ratexp)
            m, n = ratexp.p, ratexp.q
            is_alg = True
        else:
            is_alg = False
    else:
        is_alg = True

    if not (ispoly or israt or is_alg):
        return None

    R = domain.old_poly_ring(x)
    _, Dx = DifferentialOperators(R, 'Dx')

    # if the function is constant
    if not func.has(x):
        return HolonomicFunction(Dx, x, 0, [func])

    if ispoly:
        # differential equation satisfied by polynomial
        sol = func * Dx - func.diff(x)
        sol = _normalize(sol.listofpoly, sol.parent, negative=False)
        is_singular = sol.is_singular(x0)

        # try to compute the conditions for singular points
        if y0 is None and x0 == 0 and is_singular:
            rep = R.from_sympy(func).to_list()
            for i, j in enumerate(reversed(rep)):
                if j == 0:
                    continue
                coeff = list(reversed(rep))[i:]
                indicial = i
                break
            for i, j in enumerate(coeff):
                if isinstance(j, (PolyElement, FracElement)):
                    coeff[i] = j.as_expr()
            y0 = {indicial: S(coeff)}

    elif israt:
        p, q = func.as_numer_denom()
        # differential equation satisfied by rational
        sol = p * q * Dx + p * q.diff(x) - q * p.diff(x)
        sol = _normalize(sol.listofpoly, sol.parent, negative=False)

    elif is_alg:
        sol = n * (x / m) * Dx - 1
        sol = HolonomicFunction(sol, x).composition(basepoly).annihilator
        is_singular = sol.is_singular(x0)

        # try to compute the conditions for singular points
        if y0 is None and x0 == 0 and is_singular and \
            (lenics is None or lenics <= 1):
            rep = R.from_sympy(basepoly).to_list()
            for i, j in enumerate(reversed(rep)):
                if j == 0:
                    continue
                if isinstance(j, (PolyElement, FracElement)):
                    j = j.as_expr()

                coeff = S(j)**ratexp
                indicial = S(i) * ratexp
                break
            if isinstance(coeff, (PolyElement, FracElement)):
                coeff = coeff.as_expr()
            y0 = {indicial: S([coeff])}

    if y0 or not initcond:
        return HolonomicFunction(sol, x, x0, y0)

    if not lenics:
        lenics = sol.order

    if sol.is_singular(x0):
        r = HolonomicFunction(sol, x, x0)._indicial()
        l = list(r)
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            g = func / (x - x0)**r
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r:singular_ics}
            return HolonomicFunction(sol, x, x0, y0)

    y0 = _find_conditions(func, x, x0, lenics)
    while not y0:
        x0 += 1
        y0 = _find_conditions(func, x, x0, lenics)

    return HolonomicFunction(sol, x, x0, y0)


def _convert_meijerint(func, x, initcond=True, domain=QQ):
    args = meijerint._rewrite1(func, x)

    if args:
        fac, po, g, _ = args
    else:
        return None

    # lists for sum of meijerg functions
    fac_list = [fac * i[0] for i in g]
    t = po.as_base_exp()
    s = t[1] if t[0] == x else S.Zero
    po_list = [s + i[1] for i in g]
    G_list = [i[2] for i in g]

    # finds meijerg representation of x**s * meijerg(a1 ... ap, b1 ... bq, z)
    def _shift(func, s):
        z = func.args[-1]
        if z.has(I):
            z = z.subs(exp_polar, exp)

        d = z.collect(x, evaluate=False)
        b = list(d)[0]
        a = d[b]

        t = b.as_base_exp()
        b = t[1] if t[0] == x else S.Zero
        r = s / b
        an = (i + r for i in func.args[0][0])
        ap = (i + r for i in func.args[0][1])
        bm = (i + r for i in func.args[1][0])
        bq = (i + r for i in func.args[1][1])

        return a**-r, meijerg((an, ap), (bm, bq), z)

    coeff, m = _shift(G_list[0], po_list[0])
    sol = fac_list[0] * coeff * from_meijerg(m, initcond=initcond, domain=domain)

    # add all the meijerg functions after converting to holonomic
    for i in range(1, len(G_list)):
        coeff, m = _shift(G_list[i], po_list[i])
        sol += fac_list[i] * coeff * from_meijerg(m, initcond=initcond, domain=domain)

    return sol


def _create_table(table, domain=QQ):
    """
    Creates the look-up table. For a similar implementation
    see meijerint._create_lookup_table.
    """

    def add(formula, annihilator, arg, x0=0, y0=()):
        """
        Adds a formula in the dictionary
        """
        table.setdefault(_mytype(formula, x_1), []).append((formula,
            HolonomicFunction(annihilator, arg, x0, y0)))

    R = domain.old_poly_ring(x_1)
    _, Dx = DifferentialOperators(R, 'Dx')

    # add some basic functions
    add(sin(x_1), Dx**2 + 1, x_1, 0, [0, 1])
    add(cos(x_1), Dx**2 + 1, x_1, 0, [1, 0])
    add(exp(x_1), Dx - 1, x_1, 0, 1)
    add(log(x_1), Dx + x_1*Dx**2, x_1, 1, [0, 1])

    add(erf(x_1), 2*x_1*Dx + Dx**2, x_1, 0, [0, 2/sqrt(pi)])
    add(erfc(x_1), 2*x_1*Dx + Dx**2, x_1, 0, [1, -2/sqrt(pi)])
    add(erfi(x_1), -2*x_1*Dx + Dx**2, x_1, 0, [0, 2/sqrt(pi)])

    add(sinh(x_1), Dx**2 - 1, x_1, 0, [0, 1])
    add(cosh(x_1), Dx**2 - 1, x_1, 0, [1, 0])

    add(sinc(x_1), x_1 + 2*Dx + x_1*Dx**2, x_1)

    add(Si(x_1), x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)
    add(Ci(x_1), x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)

    add(Shi(x_1), -x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)


def _find_conditions(func, x, x0, order, evalf=False, use_limit=True):
    y0 = []
    for _ in range(order):
        val = func.subs(x, x0)
        if evalf:
            val = val.evalf()
        if use_limit and isinstance(val, NaN):
            val = limit(func, x, x0)
        if val.is_finite is False or isinstance(val, NaN):
            return None
        y0.append(val)
        func = func.diff(x)
    return y0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\reshape\merge.py ===
"""
SQL-style merge routines
"""
from __future__ import annotations

from collections.abc import (
    Hashable,
    Sequence,
)
import datetime
from functools import partial
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    final,
)
import uuid
import warnings

import numpy as np

from pandas._libs import (
    Timedelta,
    hashtable as libhashtable,
    join as libjoin,
    lib,
)
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    IndexLabel,
    JoinHow,
    MergeHow,
    Shape,
    Suffixes,
    npt,
)
from pandas.errors import MergeError
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_object,
    is_bool,
    is_bool_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_number,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
)

from pandas import (
    ArrowDtype,
    Categorical,
    Index,
    MultiIndex,
    Series,
)
import pandas.core.algorithms as algos
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    ExtensionArray,
)
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
    get_group_index,
    is_int64_overflow_possible,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.core import groupby
    from pandas.core.arrays import DatetimeArray
    from pandas.core.indexes.frozen import FrozenList

_factorizers = {
    np.int64: libhashtable.Int64Factorizer,
    np.longlong: libhashtable.Int64Factorizer,
    np.int32: libhashtable.Int32Factorizer,
    np.int16: libhashtable.Int16Factorizer,
    np.int8: libhashtable.Int8Factorizer,
    np.uint64: libhashtable.UInt64Factorizer,
    np.uint32: libhashtable.UInt32Factorizer,
    np.uint16: libhashtable.UInt16Factorizer,
    np.uint8: libhashtable.UInt8Factorizer,
    np.bool_: libhashtable.UInt8Factorizer,
    np.float64: libhashtable.Float64Factorizer,
    np.float32: libhashtable.Float32Factorizer,
    np.complex64: libhashtable.Complex64Factorizer,
    np.complex128: libhashtable.Complex128Factorizer,
    np.object_: libhashtable.ObjectFactorizer,
}

# See https://github.com/pandas-dev/pandas/issues/52451
if np.intc is not np.int32:
    _factorizers[np.intc] = libhashtable.Int64Factorizer

_known = (np.ndarray, ExtensionArray, Index, ABCSeries)


@Substitution("\nleft : DataFrame or named Series")
@Appender(_merge_doc, indents=0)
def merge(
    left: DataFrame | Series,
    right: DataFrame | Series,
    how: MergeHow = "inner",
    on: IndexLabel | AnyArrayLike | None = None,
    left_on: IndexLabel | AnyArrayLike | None = None,
    right_on: IndexLabel | AnyArrayLike | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Suffixes = ("_x", "_y"),
    copy: bool | None = None,
    indicator: str | bool = False,
    validate: str | None = None,
) -> DataFrame:
    left_df = _validate_operand(left)
    right_df = _validate_operand(right)
    if how == "cross":
        return _cross_merge(
            left_df,
            right_df,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            indicator=indicator,
            validate=validate,
            copy=copy,
        )
    else:
        op = _MergeOperation(
            left_df,
            right_df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            indicator=indicator,
            validate=validate,
        )
        return op.get_result(copy=copy)


def _cross_merge(
    left: DataFrame,
    right: DataFrame,
    on: IndexLabel | AnyArrayLike | None = None,
    left_on: IndexLabel | AnyArrayLike | None = None,
    right_on: IndexLabel | AnyArrayLike | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Suffixes = ("_x", "_y"),
    copy: bool | None = None,
    indicator: str | bool = False,
    validate: str | None = None,
) -> DataFrame:
    """
    See merge.__doc__ with how='cross'
    """

    if (
        left_index
        or right_index
        or right_on is not None
        or left_on is not None
        or on is not None
    ):
        raise MergeError(
            "Can not pass on, right_on, left_on or set right_index=True or "
            "left_index=True"
        )

    cross_col = f"_cross_{uuid.uuid4()}"
    left = left.assign(**{cross_col: 1})
    right = right.assign(**{cross_col: 1})

    left_on = right_on = [cross_col]

    res = merge(
        left,
        right,
        how="inner",
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        indicator=indicator,
        validate=validate,
        copy=copy,
    )
    del res[cross_col]
    return res


def _groupby_and_merge(
    by, left: DataFrame | Series, right: DataFrame | Series, merge_pieces
):
    """
    groupby & merge; we are always performing a left-by type operation

    Parameters
    ----------
    by: field to group
    left: DataFrame
    right: DataFrame
    merge_pieces: function for merging
    """
    pieces = []
    if not isinstance(by, (list, tuple)):
        by = [by]

    lby = left.groupby(by, sort=False)
    rby: groupby.DataFrameGroupBy | groupby.SeriesGroupBy | None = None

    # if we can groupby the rhs
    # then we can get vastly better perf
    if all(item in right.columns for item in by):
        rby = right.groupby(by, sort=False)

    for key, lhs in lby._grouper.get_iterator(lby._selected_obj, axis=lby.axis):
        if rby is None:
            rhs = right
        else:
            try:
                rhs = right.take(rby.indices[key])
            except KeyError:
                # key doesn't exist in left
                lcols = lhs.columns.tolist()
                cols = lcols + [r for r in right.columns if r not in set(lcols)]
                merged = lhs.reindex(columns=cols)
                merged.index = range(len(merged))
                pieces.append(merged)
                continue

        merged = merge_pieces(lhs, rhs)

        # make sure join keys are in the merged
        # TODO, should merge_pieces do this?
        merged[by] = key

        pieces.append(merged)

    # preserve the original order
    # if we have a missing piece this can be reset
    from pandas.core.reshape.concat import concat

    result = concat(pieces, ignore_index=True)
    result = result.reindex(columns=pieces[0].columns, copy=False)
    return result, lby


def merge_ordered(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: IndexLabel | None = None,
    left_on: IndexLabel | None = None,
    right_on: IndexLabel | None = None,
    left_by=None,
    right_by=None,
    fill_method: str | None = None,
    suffixes: Suffixes = ("_x", "_y"),
    how: JoinHow = "outer",
) -> DataFrame:
    """
    Perform a merge for ordered data with optional filling/interpolation.

    Designed for ordered data like time series data. Optionally
    perform group-wise merge (see examples).

    Parameters
    ----------
    left : DataFrame or named Series
    right : DataFrame or named Series
    on : label or list
        Field names to join on. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector or list of
        vectors of the length of the DataFrame to use a particular vector as
        the join key instead of columns.
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of vectors per
        left_on docs.
    left_by : column name or list of column names
        Group left DataFrame by group columns and merge piece by piece with
        right DataFrame. Must be None if either left or right are a Series.
    right_by : column name or list of column names
        Group right DataFrame by group columns and merge piece by piece with
        left DataFrame. Must be None if either left or right are a Series.
    fill_method : {'ffill', None}, default None
        Interpolation method for data.
    suffixes : list-like, default is ("_x", "_y")
        A length-2 sequence where each element is optionally a string
        indicating the suffix to add to overlapping column names in
        `left` and `right` respectively. Pass a value of `None` instead
        of a string to indicate that the column name from `left` or
        `right` should be left as-is, with no suffix. At least one of the
        values must not be None.

    how : {'left', 'right', 'outer', 'inner'}, default 'outer'
        * left: use only keys from left frame (SQL: left outer join)
        * right: use only keys from right frame (SQL: right outer join)
        * outer: use union of keys from both frames (SQL: full outer join)
        * inner: use intersection of keys from both frames (SQL: inner join).

    Returns
    -------
    DataFrame
        The merged DataFrame output type will be the same as
        'left', if it is a subclass of DataFrame.

    See Also
    --------
    merge : Merge with a database-style join.
    merge_asof : Merge on nearest keys.

    Examples
    --------
    >>> from pandas import merge_ordered
    >>> df1 = pd.DataFrame(
    ...     {
    ...         "key": ["a", "c", "e", "a", "c", "e"],
    ...         "lvalue": [1, 2, 3, 1, 2, 3],
    ...         "group": ["a", "a", "a", "b", "b", "b"]
    ...     }
    ... )
    >>> df1
      key  lvalue group
    0   a       1     a
    1   c       2     a
    2   e       3     a
    3   a       1     b
    4   c       2     b
    5   e       3     b

    >>> df2 = pd.DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
    >>> df2
      key  rvalue
    0   b       1
    1   c       2
    2   d       3

    >>> merge_ordered(df1, df2, fill_method="ffill", left_by="group")
      key  lvalue group  rvalue
    0   a       1     a     NaN
    1   b       1     a     1.0
    2   c       2     a     2.0
    3   d       2     a     3.0
    4   e       3     a     3.0
    5   a       1     b     NaN
    6   b       1     b     1.0
    7   c       2     b     2.0
    8   d       2     b     3.0
    9   e       3     b     3.0
    """

    def _merger(x, y) -> DataFrame:
        # perform the ordered merge operation
        op = _OrderedMerge(
            x,
            y,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            fill_method=fill_method,
            how=how,
        )
        return op.get_result()

    if left_by is not None and right_by is not None:
        raise ValueError("Can only group either left or right frames")
    if left_by is not None:
        if isinstance(left_by, str):
            left_by = [left_by]
        check = set(left_by).difference(left.columns)
        if len(check) != 0:
            raise KeyError(f"{check} not found in left columns")
        result, _ = _groupby_and_merge(left_by, left, right, lambda x, y: _merger(x, y))
    elif right_by is not None:
        if isinstance(right_by, str):
            right_by = [right_by]
        check = set(right_by).difference(right.columns)
        if len(check) != 0:
            raise KeyError(f"{check} not found in right columns")
        result, _ = _groupby_and_merge(
            right_by, right, left, lambda x, y: _merger(y, x)
        )
    else:
        result = _merger(left, right)
    return result


def merge_asof(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: IndexLabel | None = None,
    left_on: IndexLabel | None = None,
    right_on: IndexLabel | None = None,
    left_index: bool = False,
    right_index: bool = False,
    by=None,
    left_by=None,
    right_by=None,
    suffixes: Suffixes = ("_x", "_y"),
    tolerance: int | Timedelta | None = None,
    allow_exact_matches: bool = True,
    direction: str = "backward",
) -> DataFrame:
    """
    Perform a merge by key distance.

    This is similar to a left-join except that we match on nearest
    key rather than equal keys. Both DataFrames must be sorted by the key.

    For each row in the left DataFrame:

      - A "backward" search selects the last row in the right DataFrame whose
        'on' key is less than or equal to the left's key.

      - A "forward" search selects the first row in the right DataFrame whose
        'on' key is greater than or equal to the left's key.

      - A "nearest" search selects the row in the right DataFrame whose 'on'
        key is closest in absolute distance to the left's key.

    Optionally match on equivalent keys with 'by' before searching with 'on'.

    Parameters
    ----------
    left : DataFrame or named Series
    right : DataFrame or named Series
    on : label
        Field name to join on. Must be found in both DataFrames.
        The data MUST be ordered. Furthermore this must be a numeric column,
        such as datetimelike, integer, or float. On or left_on/right_on
        must be given.
    left_on : label
        Field name to join on in left DataFrame.
    right_on : label
        Field name to join on in right DataFrame.
    left_index : bool
        Use the index of the left DataFrame as the join key.
    right_index : bool
        Use the index of the right DataFrame as the join key.
    by : column name or list of column names
        Match on these columns before performing merge operation.
    left_by : column name
        Field names to match on in the left DataFrame.
    right_by : column name
        Field names to match on in the right DataFrame.
    suffixes : 2-length sequence (tuple, list, ...)
        Suffix to apply to overlapping column names in the left and right
        side, respectively.
    tolerance : int or Timedelta, optional, default None
        Select asof tolerance within this range; must be compatible
        with the merge index.
    allow_exact_matches : bool, default True

        - If True, allow matching with the same 'on' value
          (i.e. less-than-or-equal-to / greater-than-or-equal-to)
        - If False, don't match the same 'on' value
          (i.e., strictly less-than / strictly greater-than).

    direction : 'backward' (default), 'forward', or 'nearest'
        Whether to search for prior, subsequent, or closest matches.

    Returns
    -------
    DataFrame

    See Also
    --------
    merge : Merge with a database-style join.
    merge_ordered : Merge with optional filling/interpolation.

    Examples
    --------
    >>> left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
    >>> left
        a left_val
    0   1        a
    1   5        b
    2  10        c

    >>> right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})
    >>> right
       a  right_val
    0  1          1
    1  2          2
    2  3          3
    3  6          6
    4  7          7

    >>> pd.merge_asof(left, right, on="a")
        a left_val  right_val
    0   1        a          1
    1   5        b          3
    2  10        c          7

    >>> pd.merge_asof(left, right, on="a", allow_exact_matches=False)
        a left_val  right_val
    0   1        a        NaN
    1   5        b        3.0
    2  10        c        7.0

    >>> pd.merge_asof(left, right, on="a", direction="forward")
        a left_val  right_val
    0   1        a        1.0
    1   5        b        6.0
    2  10        c        NaN

    >>> pd.merge_asof(left, right, on="a", direction="nearest")
        a left_val  right_val
    0   1        a          1
    1   5        b          6
    2  10        c          7

    We can use indexed DataFrames as well.

    >>> left = pd.DataFrame({"left_val": ["a", "b", "c"]}, index=[1, 5, 10])
    >>> left
       left_val
    1         a
    5         b
    10        c

    >>> right = pd.DataFrame({"right_val": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
    >>> right
       right_val
    1          1
    2          2
    3          3
    6          6
    7          7

    >>> pd.merge_asof(left, right, left_index=True, right_index=True)
       left_val  right_val
    1         a          1
    5         b          3
    10        c          7

    Here is a real-world times-series example

    >>> quotes = pd.DataFrame(
    ...     {
    ...         "time": [
    ...             pd.Timestamp("2016-05-25 13:30:00.023"),
    ...             pd.Timestamp("2016-05-25 13:30:00.023"),
    ...             pd.Timestamp("2016-05-25 13:30:00.030"),
    ...             pd.Timestamp("2016-05-25 13:30:00.041"),
    ...             pd.Timestamp("2016-05-25 13:30:00.048"),
    ...             pd.Timestamp("2016-05-25 13:30:00.049"),
    ...             pd.Timestamp("2016-05-25 13:30:00.072"),
    ...             pd.Timestamp("2016-05-25 13:30:00.075")
    ...         ],
    ...         "ticker": [
    ...                "GOOG",
    ...                "MSFT",
    ...                "MSFT",
    ...                "MSFT",
    ...                "GOOG",
    ...                "AAPL",
    ...                "GOOG",
    ...                "MSFT"
    ...            ],
    ...            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
    ...            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03]
    ...     }
    ... )
    >>> quotes
                         time ticker     bid     ask
    0 2016-05-25 13:30:00.023   GOOG  720.50  720.93
    1 2016-05-25 13:30:00.023   MSFT   51.95   51.96
    2 2016-05-25 13:30:00.030   MSFT   51.97   51.98
    3 2016-05-25 13:30:00.041   MSFT   51.99   52.00
    4 2016-05-25 13:30:00.048   GOOG  720.50  720.93
    5 2016-05-25 13:30:00.049   AAPL   97.99   98.01
    6 2016-05-25 13:30:00.072   GOOG  720.50  720.88
    7 2016-05-25 13:30:00.075   MSFT   52.01   52.03

    >>> trades = pd.DataFrame(
    ...        {
    ...            "time": [
    ...                pd.Timestamp("2016-05-25 13:30:00.023"),
    ...                pd.Timestamp("2016-05-25 13:30:00.038"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048")
    ...            ],
    ...            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
    ...            "price": [51.95, 51.95, 720.77, 720.92, 98.0],
    ...            "quantity": [75, 155, 100, 100, 100]
    ...        }
    ...    )
    >>> trades
                         time ticker   price  quantity
    0 2016-05-25 13:30:00.023   MSFT   51.95        75
    1 2016-05-25 13:30:00.038   MSFT   51.95       155
    2 2016-05-25 13:30:00.048   GOOG  720.77       100
    3 2016-05-25 13:30:00.048   GOOG  720.92       100
    4 2016-05-25 13:30:00.048   AAPL   98.00       100

    By default we are taking the asof of the quotes

    >>> pd.merge_asof(trades, quotes, on="time", by="ticker")
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96
    1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98
    2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93
    3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN

    We only asof within 2ms between the quote time and the trade time

    >>> pd.merge_asof(
    ...     trades, quotes, on="time", by="ticker", tolerance=pd.Timedelta("2ms")
    ... )
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96
    1 2016-05-25 13:30:00.038   MSFT   51.95       155     NaN     NaN
    2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93
    3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN

    We only asof within 10ms between the quote time and the trade time
    and we exclude exact matches on time. However *prior* data will
    propagate forward

    >>> pd.merge_asof(
    ...     trades,
    ...     quotes,
    ...     on="time",
    ...     by="ticker",
    ...     tolerance=pd.Timedelta("10ms"),
    ...     allow_exact_matches=False
    ... )
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75     NaN     NaN
    1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98
    2 2016-05-25 13:30:00.048   GOOG  720.77       100     NaN     NaN
    3 2016-05-25 13:30:00.048   GOOG  720.92       100     NaN     NaN
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN
    """
    op = _AsOfMerge(
        left,
        right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        by=by,
        left_by=left_by,
        right_by=right_by,
        suffixes=suffixes,
        how="asof",
        tolerance=tolerance,
        allow_exact_matches=allow_exact_matches,
        direction=direction,
    )
    return op.get_result()


# TODO: transformations??
# TODO: only copy DataFrames when modification necessary
class _MergeOperation:
    """
    Perform a database (SQL) merge operation between two DataFrame or Series
    objects using either columns as keys or their row indexes
    """

    _merge_type = "merge"
    how: JoinHow | Literal["asof"]
    on: IndexLabel | None
    # left_on/right_on may be None when passed, but in validate_specification
    #  get replaced with non-None.
    left_on: Sequence[Hashable | AnyArrayLike]
    right_on: Sequence[Hashable | AnyArrayLike]
    left_index: bool
    right_index: bool
    sort: bool
    suffixes: Suffixes
    copy: bool
    indicator: str | bool
    validate: str | None
    join_names: list[Hashable]
    right_join_keys: list[ArrayLike]
    left_join_keys: list[ArrayLike]

    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        how: JoinHow | Literal["asof"] = "inner",
        on: IndexLabel | AnyArrayLike | None = None,
        left_on: IndexLabel | AnyArrayLike | None = None,
        right_on: IndexLabel | AnyArrayLike | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = True,
        suffixes: Suffixes = ("_x", "_y"),
        indicator: str | bool = False,
        validate: str | None = None,
    ) -> None:
        _left = _validate_operand(left)
        _right = _validate_operand(right)
        self.left = self.orig_left = _left
        self.right = self.orig_right = _right
        self.how = how

        self.on = com.maybe_make_list(on)

        self.suffixes = suffixes
        self.sort = sort or how == "outer"

        self.left_index = left_index
        self.right_index = right_index

        self.indicator = indicator

        if not is_bool(left_index):
            raise ValueError(
                f"left_index parameter must be of type bool, not {type(left_index)}"
            )
        if not is_bool(right_index):
            raise ValueError(
                f"right_index parameter must be of type bool, not {type(right_index)}"
            )

        # GH 40993: raise when merging between different levels; enforced in 2.0
        if _left.columns.nlevels != _right.columns.nlevels:
            msg = (
                "Not allowed to merge between different levels. "
                f"({_left.columns.nlevels} levels on the left, "
                f"{_right.columns.nlevels} on the right)"
            )
            raise MergeError(msg)

        self.left_on, self.right_on = self._validate_left_right_on(left_on, right_on)

        (
            self.left_join_keys,
            self.right_join_keys,
            self.join_names,
            left_drop,
            right_drop,
        ) = self._get_merge_keys()

        if left_drop:
            self.left = self.left._drop_labels_or_levels(left_drop)

        if right_drop:
            self.right = self.right._drop_labels_or_levels(right_drop)

        self._maybe_require_matching_dtypes(self.left_join_keys, self.right_join_keys)
        self._validate_tolerance(self.left_join_keys)

        # validate the merge keys dtypes. We may need to coerce
        # to avoid incompatible dtypes
        self._maybe_coerce_merge_keys()

        # If argument passed to validate,
        # check if columns specified as unique
        # are in fact unique.
        if validate is not None:
            self._validate_validate_kwd(validate)

    def _maybe_require_matching_dtypes(
        self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]
    ) -> None:
        # Overridden by AsOfMerge
        pass

    def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None:
        # Overridden by AsOfMerge
        pass

    @final
    def _reindex_and_concat(
        self,
        join_index: Index,
        left_indexer: npt.NDArray[np.intp] | None,
        right_indexer: npt.NDArray[np.intp] | None,
        copy: bool | None,
    ) -> DataFrame:
        """
        reindex along index and concat along columns.
        """
        # Take views so we do not alter the originals
        left = self.left[:]
        right = self.right[:]

        llabels, rlabels = _items_overlap_with_suffix(
            self.left._info_axis, self.right._info_axis, self.suffixes
        )

        if left_indexer is not None and not is_range_indexer(left_indexer, len(left)):
            # Pinning the index here (and in the right code just below) is not
            #  necessary, but makes the `.take` more performant if we have e.g.
            #  a MultiIndex for left.index.
            lmgr = left._mgr.reindex_indexer(
                join_index,
                left_indexer,
                axis=1,
                copy=False,
                only_slice=True,
                allow_dups=True,
                use_na_proxy=True,
            )
            left = left._constructor_from_mgr(lmgr, axes=lmgr.axes)
        left.index = join_index

        if right_indexer is not None and not is_range_indexer(
            right_indexer, len(right)
        ):
            rmgr = right._mgr.reindex_indexer(
                join_index,
                right_indexer,
                axis=1,
                copy=False,
                only_slice=True,
                allow_dups=True,
                use_na_proxy=True,
            )
            right = right._constructor_from_mgr(rmgr, axes=rmgr.axes)
        right.index = join_index

        from pandas import concat

        left.columns = llabels
        right.columns = rlabels
        result = concat([left, right], axis=1, copy=copy)
        return result

    def get_result(self, copy: bool | None = True) -> DataFrame:
        if self.indicator:
            self.left, self.right = self._indicator_pre_merge(self.left, self.right)

        join_index, left_indexer, right_indexer = self._get_join_info()

        result = self._reindex_and_concat(
            join_index, left_indexer, right_indexer, copy=copy
        )
        result = result.__finalize__(self, method=self._merge_type)

        if self.indicator:
            result = self._indicator_post_merge(result)

        self._maybe_add_join_keys(result, left_indexer, right_indexer)

        self._maybe_restore_index_levels(result)

        return result.__finalize__(self, method="merge")

    @final
    @cache_readonly
    def _indicator_name(self) -> str | None:
        if isinstance(self.indicator, str):
            return self.indicator
        elif isinstance(self.indicator, bool):
            return "_merge" if self.indicator else None
        else:
            raise ValueError(
                "indicator option can only accept boolean or string arguments"
            )

    @final
    def _indicator_pre_merge(
        self, left: DataFrame, right: DataFrame
    ) -> tuple[DataFrame, DataFrame]:
        columns = left.columns.union(right.columns)

        for i in ["_left_indicator", "_right_indicator"]:
            if i in columns:
                raise ValueError(
                    "Cannot use `indicator=True` option when "
                    f"data contains a column named {i}"
                )
        if self._indicator_name in columns:
            raise ValueError(
                "Cannot use name of an existing column for indicator column"
            )

        left = left.copy()
        right = right.copy()

        left["_left_indicator"] = 1
        left["_left_indicator"] = left["_left_indicator"].astype("int8")

        right["_right_indicator"] = 2
        right["_right_indicator"] = right["_right_indicator"].astype("int8")

        return left, right

    @final
    def _indicator_post_merge(self, result: DataFrame) -> DataFrame:
        result["_left_indicator"] = result["_left_indicator"].fillna(0)
        result["_right_indicator"] = result["_right_indicator"].fillna(0)

        result[self._indicator_name] = Categorical(
            (result["_left_indicator"] + result["_right_indicator"]),
            categories=[1, 2, 3],
        )
        result[self._indicator_name] = result[
            self._indicator_name
        ].cat.rename_categories(["left_only", "right_only", "both"])

        result = result.drop(labels=["_left_indicator", "_right_indicator"], axis=1)
        return result

    @final
    def _maybe_restore_index_levels(self, result: DataFrame) -> None:
        """
        Restore index levels specified as `on` parameters

        Here we check for cases where `self.left_on` and `self.right_on` pairs
        each reference an index level in their respective DataFrames. The
        joined columns corresponding to these pairs are then restored to the
        index of `result`.

        **Note:** This method has side effects. It modifies `result` in-place

        Parameters
        ----------
        result: DataFrame
            merge result

        Returns
        -------
        None
        """
        names_to_restore = []
        for name, left_key, right_key in zip(
            self.join_names, self.left_on, self.right_on
        ):
            if (
                # Argument 1 to "_is_level_reference" of "NDFrame" has incompatible
                # type "Union[Hashable, ExtensionArray, Index, Series]"; expected
                # "Hashable"
                self.orig_left._is_level_reference(left_key)  # type: ignore[arg-type]
                # Argument 1 to "_is_level_reference" of "NDFrame" has incompatible
                # type "Union[Hashable, ExtensionArray, Index, Series]"; expected
                # "Hashable"
                and self.orig_right._is_level_reference(
                    right_key  # type: ignore[arg-type]
                )
                and left_key == right_key
                and name not in result.index.names
            ):
                names_to_restore.append(name)

        if names_to_restore:
            result.set_index(names_to_restore, inplace=True)

    @final
    def _maybe_add_join_keys(
        self,
        result: DataFrame,
        left_indexer: npt.NDArray[np.intp] | None,
        right_indexer: npt.NDArray[np.intp] | None,
    ) -> None:
        left_has_missing = None
        right_has_missing = None

        assert all(isinstance(x, _known) for x in self.left_join_keys)

        keys = zip(self.join_names, self.left_on, self.right_on)
        for i, (name, lname, rname) in enumerate(keys):
            if not _should_fill(lname, rname):
                continue

            take_left, take_right = None, None

            if name in result:
                if left_indexer is not None or right_indexer is not None:
                    if name in self.left:
                        if left_has_missing is None:
                            left_has_missing = (
                                False
                                if left_indexer is None
                                else (left_indexer == -1).any()
                            )

                        if left_has_missing:
                            take_right = self.right_join_keys[i]

                            if result[name].dtype != self.left[name].dtype:
                                take_left = self.left[name]._values

                    elif name in self.right:
                        if right_has_missing is None:
                            right_has_missing = (
                                False
                                if right_indexer is None
                                else (right_indexer == -1).any()
                            )

                        if right_has_missing:
                            take_left = self.left_join_keys[i]

                            if result[name].dtype != self.right[name].dtype:
                                take_right = self.right[name]._values

            else:
                take_left = self.left_join_keys[i]
                take_right = self.right_join_keys[i]

            if take_left is not None or take_right is not None:
                if take_left is None:
                    lvals = result[name]._values
                elif left_indexer is None:
                    lvals = take_left
                else:
                    # TODO: can we pin down take_left's type earlier?
                    take_left = extract_array(take_left, extract_numpy=True)
                    lfill = na_value_for_dtype(take_left.dtype)
                    lvals = algos.take_nd(take_left, left_indexer, fill_value=lfill)

                if take_right is None:
                    rvals = result[name]._values
                elif right_indexer is None:
                    rvals = take_right
                else:
                    # TODO: can we pin down take_right's type earlier?
                    taker = extract_array(take_right, extract_numpy=True)
                    rfill = na_value_for_dtype(taker.dtype)
                    rvals = algos.take_nd(taker, right_indexer, fill_value=rfill)

                # if we have an all missing left_indexer
                # make sure to just use the right values or vice-versa
                if left_indexer is not None and (left_indexer == -1).all():
                    key_col = Index(rvals)
                    result_dtype = rvals.dtype
                elif right_indexer is not None and (right_indexer == -1).all():
                    key_col = Index(lvals)
                    result_dtype = lvals.dtype
                else:
                    key_col = Index(lvals)
                    if left_indexer is not None:
                        mask_left = left_indexer == -1
                        key_col = key_col.where(~mask_left, rvals)
                    result_dtype = find_common_type([lvals.dtype, rvals.dtype])
                    if (
                        lvals.dtype.kind == "M"
                        and rvals.dtype.kind == "M"
                        and result_dtype.kind == "O"
                    ):
                        # TODO(non-nano) Workaround for common_type not dealing
                        # with different resolutions
                        result_dtype = key_col.dtype

                if result._is_label_reference(name):
                    result[name] = result._constructor_sliced(
                        key_col, dtype=result_dtype, index=result.index
                    )
                elif result._is_level_reference(name):
                    if isinstance(result.index, MultiIndex):
                        key_col.name = name
                        idx_list = [
                            result.index.get_level_values(level_name)
                            if level_name != name
                            else key_col
                            for level_name in result.index.names
                        ]

                        result.set_index(idx_list, inplace=True)
                    else:
                        result.index = Index(key_col, name=name)
                else:
                    result.insert(i, name or f"key_{i}", key_col)

    def _get_join_indexers(
        self,
    ) -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        """return the join indexers"""
        # make mypy happy
        assert self.how != "asof"
        return get_join_indexers(
            self.left_join_keys, self.right_join_keys, sort=self.sort, how=self.how
        )

    @final
    def _get_join_info(
        self,
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        left_ax = self.left.index
        right_ax = self.right.index

        if self.left_index and self.right_index and self.how != "asof":
            join_index, left_indexer, right_indexer = left_ax.join(
                right_ax, how=self.how, return_indexers=True, sort=self.sort
            )

        elif self.right_index and self.how == "left":
            join_index, left_indexer, right_indexer = _left_join_on_index(
                left_ax, right_ax, self.left_join_keys, sort=self.sort
            )

        elif self.left_index and self.how == "right":
            join_index, right_indexer, left_indexer = _left_join_on_index(
                right_ax, left_ax, self.right_join_keys, sort=self.sort
            )
        else:
            (left_indexer, right_indexer) = self._get_join_indexers()

            if self.right_index:
                if len(self.left) > 0:
                    join_index = self._create_join_index(
                        left_ax,
                        right_ax,
                        left_indexer,
                        how="right",
                    )
                elif right_indexer is None:
                    join_index = right_ax.copy()
                else:
                    join_index = right_ax.take(right_indexer)
            elif self.left_index:
                if self.how == "asof":
                    # GH#33463 asof should always behave like a left merge
                    join_index = self._create_join_index(
                        left_ax,
                        right_ax,
                        left_indexer,
                        how="left",
                    )

                elif len(self.right) > 0:
                    join_index = self._create_join_index(
                        right_ax,
                        left_ax,
                        right_indexer,
                        how="left",
                    )
                elif left_indexer is None:
                    join_index = left_ax.copy()
                else:
                    join_index = left_ax.take(left_indexer)
            else:
                n = len(left_ax) if left_indexer is None else len(left_indexer)
                join_index = default_index(n)

        return join_index, left_indexer, right_indexer

    @final
    def _create_join_index(
        self,
        index: Index,
        other_index: Index,
        indexer: npt.NDArray[np.intp] | None,
        how: JoinHow = "left",
    ) -> Index:
        """
        Create a join index by rearranging one index to match another

        Parameters
        ----------
        index : Index
            index being rearranged
        other_index : Index
            used to supply values not found in index
        indexer : np.ndarray[np.intp] or None
            how to rearrange index
        how : str
            Replacement is only necessary if indexer based on other_index.

        Returns
        -------
        Index
        """
        if self.how in (how, "outer") and not isinstance(other_index, MultiIndex):
            # if final index requires values in other_index but not target
            # index, indexer may hold missing (-1) values, causing Index.take
            # to take the final value in target index. So, we set the last
            # element to be the desired fill value. We do not use allow_fill
            # and fill_value because it throws a ValueError on integer indices
            mask = indexer == -1
            if np.any(mask):
                fill_value = na_value_for_dtype(index.dtype, compat=False)
                index = index.append(Index([fill_value]))
        if indexer is None:
            return index.copy()
        return index.take(indexer)

    @final
    def _get_merge_keys(
        self,
    ) -> tuple[
        list[ArrayLike],
        list[ArrayLike],
        list[Hashable],
        list[Hashable],
        list[Hashable],
    ]:
        """
        Returns
        -------
        left_keys, right_keys, join_names, left_drop, right_drop
        """
        left_keys: list[ArrayLike] = []
        right_keys: list[ArrayLike] = []
        join_names: list[Hashable] = []
        right_drop: list[Hashable] = []
        left_drop: list[Hashable] = []

        left, right = self.left, self.right

        is_lkey = lambda x: isinstance(x, _known) and len(x) == len(left)
        is_rkey = lambda x: isinstance(x, _known) and len(x) == len(right)

        # Note that pd.merge_asof() has separate 'on' and 'by' parameters. A
        # user could, for example, request 'left_index' and 'left_by'. In a
        # regular pd.merge(), users cannot specify both 'left_index' and
        # 'left_on'. (Instead, users have a MultiIndex). That means the
        # self.left_on in this function is always empty in a pd.merge(), but
        # a pd.merge_asof(left_index=True, left_by=...) will result in a
        # self.left_on array with a None in the middle of it. This requires
        # a work-around as designated in the code below.
        # See _validate_left_right_on() for where this happens.

        # ugh, spaghetti re #733
        if _any(self.left_on) and _any(self.right_on):
            for lk, rk in zip(self.left_on, self.right_on):
                lk = extract_array(lk, extract_numpy=True)
                rk = extract_array(rk, extract_numpy=True)
                if is_lkey(lk):
                    lk = cast(ArrayLike, lk)
                    left_keys.append(lk)
                    if is_rkey(rk):
                        rk = cast(ArrayLike, rk)
                        right_keys.append(rk)
                        join_names.append(None)  # what to do?
                    else:
                        # Then we're either Hashable or a wrong-length arraylike,
                        #  the latter of which will raise
                        rk = cast(Hashable, rk)
                        if rk is not None:
                            right_keys.append(right._get_label_or_level_values(rk))
                            join_names.append(rk)
                        else:
                            # work-around for merge_asof(right_index=True)
                            right_keys.append(right.index._values)
                            join_names.append(right.index.name)
                else:
                    if not is_rkey(rk):
                        # Then we're either Hashable or a wrong-length arraylike,
                        #  the latter of which will raise
                        rk = cast(Hashable, rk)
                        if rk is not None:
                            right_keys.append(right._get_label_or_level_values(rk))
                        else:
                            # work-around for merge_asof(right_index=True)
                            right_keys.append(right.index._values)
                        if lk is not None and lk == rk:  # FIXME: what about other NAs?
                            right_drop.append(rk)
                    else:
                        rk = cast(ArrayLike, rk)
                        right_keys.append(rk)
                    if lk is not None:
                        # Then we're either Hashable or a wrong-length arraylike,
                        #  the latter of which will raise
                        lk = cast(Hashable, lk)
                        left_keys.append(left._get_label_or_level_values(lk))
                        join_names.append(lk)
                    else:
                        # work-around for merge_asof(left_index=True)
                        left_keys.append(left.index._values)
                        join_names.append(left.index.name)
        elif _any(self.left_on):
            for k in self.left_on:
                if is_lkey(k):
                    k = extract_array(k, extract_numpy=True)
                    k = cast(ArrayLike, k)
                    left_keys.append(k)
                    join_names.append(None)
                else:
                    # Then we're either Hashable or a wrong-length arraylike,
                    #  the latter of which will raise
                    k = cast(Hashable, k)
                    left_keys.append(left._get_label_or_level_values(k))
                    join_names.append(k)
            if isinstance(self.right.index, MultiIndex):
                right_keys = [
                    lev._values.take(lev_codes)
                    for lev, lev_codes in zip(
                        self.right.index.levels, self.right.index.codes
                    )
                ]
            else:
                right_keys = [self.right.index._values]
        elif _any(self.right_on):
            for k in self.right_on:
                k = extract_array(k, extract_numpy=True)
                if is_rkey(k):
                    k = cast(ArrayLike, k)
                    right_keys.append(k)
                    join_names.append(None)
                else:
                    # Then we're either Hashable or a wrong-length arraylike,
                    #  the latter of which will raise
                    k = cast(Hashable, k)
                    right_keys.append(right._get_label_or_level_values(k))
                    join_names.append(k)
            if isinstance(self.left.index, MultiIndex):
                left_keys = [
                    lev._values.take(lev_codes)
                    for lev, lev_codes in zip(
                        self.left.index.levels, self.left.index.codes
                    )
                ]
            else:
                left_keys = [self.left.index._values]

        return left_keys, right_keys, join_names, left_drop, right_drop

    @final
    def _maybe_coerce_merge_keys(self) -> None:
        # we have valid merges but we may have to further
        # coerce these if they are originally incompatible types
        #
        # for example if these are categorical, but are not dtype_equal
        # or if we have object and integer dtypes

        for lk, rk, name in zip(
            self.left_join_keys, self.right_join_keys, self.join_names
        ):
            if (len(lk) and not len(rk)) or (not len(lk) and len(rk)):
                continue

            lk = extract_array(lk, extract_numpy=True)
            rk = extract_array(rk, extract_numpy=True)

            lk_is_cat = isinstance(lk.dtype, CategoricalDtype)
            rk_is_cat = isinstance(rk.dtype, CategoricalDtype)
            lk_is_object_or_string = is_object_dtype(lk.dtype) or is_string_dtype(
                lk.dtype
            )
            rk_is_object_or_string = is_object_dtype(rk.dtype) or is_string_dtype(
                rk.dtype
            )

            # if either left or right is a categorical
            # then the must match exactly in categories & ordered
            if lk_is_cat and rk_is_cat:
                lk = cast(Categorical, lk)
                rk = cast(Categorical, rk)
                if lk._categories_match_up_to_permutation(rk):
                    continue

            elif lk_is_cat or rk_is_cat:
                pass

            elif lk.dtype == rk.dtype:
                continue

            msg = (
                f"You are trying to merge on {lk.dtype} and {rk.dtype} columns "
                f"for key '{name}'. If you wish to proceed you should use pd.concat"
            )

            # if we are numeric, then allow differing
            # kinds to proceed, eg. int64 and int8, int and float
            # further if we are object, but we infer to
            # the same, then proceed
            if is_numeric_dtype(lk.dtype) and is_numeric_dtype(rk.dtype):
                if lk.dtype.kind == rk.dtype.kind:
                    continue

                if isinstance(lk.dtype, ExtensionDtype) and not isinstance(
                    rk.dtype, ExtensionDtype
                ):
                    ct = find_common_type([lk.dtype, rk.dtype])
                    if isinstance(ct, ExtensionDtype):
                        com_cls = ct.construct_array_type()
                        rk = com_cls._from_sequence(rk, dtype=ct, copy=False)
                    else:
                        rk = rk.astype(ct)
                elif isinstance(rk.dtype, ExtensionDtype):
                    ct = find_common_type([lk.dtype, rk.dtype])
                    if isinstance(ct, ExtensionDtype):
                        com_cls = ct.construct_array_type()
                        lk = com_cls._from_sequence(lk, dtype=ct, copy=False)
                    else:
                        lk = lk.astype(ct)

                # check whether ints and floats
                if is_integer_dtype(rk.dtype) and is_float_dtype(lk.dtype):
                    # GH 47391 numpy > 1.24 will raise a RuntimeError for nan -> int
                    with np.errstate(invalid="ignore"):
                        # error: Argument 1 to "astype" of "ndarray" has incompatible
                        # type "Union[ExtensionDtype, Any, dtype[Any]]"; expected
                        # "Union[dtype[Any], Type[Any], _SupportsDType[dtype[Any]]]"
                        casted = lk.astype(rk.dtype)  # type: ignore[arg-type]

                    mask = ~np.isnan(lk)
                    match = lk == casted
                    if not match[mask].all():
                        warnings.warn(
                            "You are merging on int and float "
                            "columns where the float values "
                            "are not equal to their int representation.",
                            UserWarning,
                            stacklevel=find_stack_level(),
                        )
                    continue

                if is_float_dtype(rk.dtype) and is_integer_dtype(lk.dtype):
                    # GH 47391 numpy > 1.24 will raise a RuntimeError for nan -> int
                    with np.errstate(invalid="ignore"):
                        # error: Argument 1 to "astype" of "ndarray" has incompatible
                        # type "Union[ExtensionDtype, Any, dtype[Any]]"; expected
                        # "Union[dtype[Any], Type[Any], _SupportsDType[dtype[Any]]]"
                        casted = rk.astype(lk.dtype)  # type: ignore[arg-type]

                    mask = ~np.isnan(rk)
                    match = rk == casted
                    if not match[mask].all():
                        warnings.warn(
                            "You are merging on int and float "
                            "columns where the float values "
                            "are not equal to their int representation.",
                            UserWarning,
                            stacklevel=find_stack_level(),
                        )
                    continue

                # let's infer and see if we are ok
                if lib.infer_dtype(lk, skipna=False) == lib.infer_dtype(
                    rk, skipna=False
                ):
                    continue

            # Check if we are trying to merge on obviously
            # incompatible dtypes GH 9780, GH 15800

            # bool values are coerced to object
            elif (lk_is_object_or_string and is_bool_dtype(rk.dtype)) or (
                is_bool_dtype(lk.dtype) and rk_is_object_or_string
            ):
                pass

            # object values are allowed to be merged
            elif (lk_is_object_or_string and is_numeric_dtype(rk.dtype)) or (
                is_numeric_dtype(lk.dtype) and rk_is_object_or_string
            ):
                inferred_left = lib.infer_dtype(lk, skipna=False)
                inferred_right = lib.infer_dtype(rk, skipna=False)
                bool_types = ["integer", "mixed-integer", "boolean", "empty"]
                string_types = ["string", "unicode", "mixed", "bytes", "empty"]

                # inferred bool
                if inferred_left in bool_types and inferred_right in bool_types:
                    pass

                # unless we are merging non-string-like with string-like
                elif (
                    inferred_left in string_types and inferred_right not in string_types
                ) or (
                    inferred_right in string_types and inferred_left not in string_types
                ):
                    raise ValueError(msg)

            # datetimelikes must match exactly
            elif needs_i8_conversion(lk.dtype) and not needs_i8_conversion(rk.dtype):
                raise ValueError(msg)
            elif not needs_i8_conversion(lk.dtype) and needs_i8_conversion(rk.dtype):
                raise ValueError(msg)
            elif isinstance(lk.dtype, DatetimeTZDtype) and not isinstance(
                rk.dtype, DatetimeTZDtype
            ):
                raise ValueError(msg)
            elif not isinstance(lk.dtype, DatetimeTZDtype) and isinstance(
                rk.dtype, DatetimeTZDtype
            ):
                raise ValueError(msg)
            elif (
                isinstance(lk.dtype, DatetimeTZDtype)
                and isinstance(rk.dtype, DatetimeTZDtype)
            ) or (lk.dtype.kind == "M" and rk.dtype.kind == "M"):
                # allows datetime with different resolutions
                continue
            # datetime and timedelta not allowed
            elif lk.dtype.kind == "M" and rk.dtype.kind == "m":
                raise ValueError(msg)
            elif lk.dtype.kind == "m" and rk.dtype.kind == "M":
                raise ValueError(msg)

            elif is_object_dtype(lk.dtype) and is_object_dtype(rk.dtype):
                continue

            # Houston, we have a problem!
            # let's coerce to object if the dtypes aren't
            # categorical, otherwise coerce to the category
            # dtype. If we coerced categories to object,
            # then we would lose type information on some
            # columns, and end up trying to merge
            # incompatible dtypes. See GH 16900.
            if name in self.left.columns:
                typ = cast(Categorical, lk).categories.dtype if lk_is_cat else object
                self.left = self.left.copy()
                self.left[name] = self.left[name].astype(typ)
            if name in self.right.columns:
                typ = cast(Categorical, rk).categories.dtype if rk_is_cat else object
                self.right = self.right.copy()
                self.right[name] = self.right[name].astype(typ)

    def _validate_left_right_on(self, left_on, right_on):
        left_on = com.maybe_make_list(left_on)
        right_on = com.maybe_make_list(right_on)

        # Hm, any way to make this logic less complicated??
        if self.on is None and left_on is None and right_on is None:
            if self.left_index and self.right_index:
                left_on, right_on = (), ()
            elif self.left_index:
                raise MergeError("Must pass right_on or right_index=True")
            elif self.right_index:
                raise MergeError("Must pass left_on or left_index=True")
            else:
                # use the common columns
                left_cols = self.left.columns
                right_cols = self.right.columns
                common_cols = left_cols.intersection(right_cols)
                if len(common_cols) == 0:
                    raise MergeError(
                        "No common columns to perform merge on. "
                        f"Merge options: left_on={left_on}, "
                        f"right_on={right_on}, "
                        f"left_index={self.left_index}, "
                        f"right_index={self.right_index}"
                    )
                if (
                    not left_cols.join(common_cols, how="inner").is_unique
                    or not right_cols.join(common_cols, how="inner").is_unique
                ):
                    raise MergeError(f"Data columns not unique: {repr(common_cols)}")
                left_on = right_on = common_cols
        elif self.on is not None:
            if left_on is not None or right_on is not None:
                raise MergeError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )
            if self.left_index or self.right_index:
                raise MergeError(
                    'Can only pass argument "on" OR "left_index" '
                    'and "right_index", not a combination of both.'
                )
            left_on = right_on = self.on
        elif left_on is not None:
            if self.left_index:
                raise MergeError(
                    'Can only pass argument "left_on" OR "left_index" not both.'
                )
            if not self.right_index and right_on is None:
                raise MergeError('Must pass "right_on" OR "right_index".')
            n = len(left_on)
            if self.right_index:
                if len(left_on) != self.right.index.nlevels:
                    raise ValueError(
                        "len(left_on) must equal the number "
                        'of levels in the index of "right"'
                    )
                right_on = [None] * n
        elif right_on is not None:
            if self.right_index:
                raise MergeError(
                    'Can only pass argument "right_on" OR "right_index" not both.'
                )
            if not self.left_index and left_on is None:
                raise MergeError('Must pass "left_on" OR "left_index".')
            n = len(right_on)
            if self.left_index:
                if len(right_on) != self.left.index.nlevels:
                    raise ValueError(
                        "len(right_on) must equal the number "
                        'of levels in the index of "left"'
                    )
                left_on = [None] * n
        if len(right_on) != len(left_on):
            raise ValueError("len(right_on) must equal len(left_on)")

        return left_on, right_on

    @final
    def _validate_validate_kwd(self, validate: str) -> None:
        # Check uniqueness of each
        if self.left_index:
            left_unique = self.orig_left.index.is_unique
        else:
            left_unique = MultiIndex.from_arrays(self.left_join_keys).is_unique

        if self.right_index:
            right_unique = self.orig_right.index.is_unique
        else:
            right_unique = MultiIndex.from_arrays(self.right_join_keys).is_unique

        # Check data integrity
        if validate in ["one_to_one", "1:1"]:
            if not left_unique and not right_unique:
                raise MergeError(
                    "Merge keys are not unique in either left "
                    "or right dataset; not a one-to-one merge"
                )
            if not left_unique:
                raise MergeError(
                    "Merge keys are not unique in left dataset; not a one-to-one merge"
                )
            if not right_unique:
                raise MergeError(
                    "Merge keys are not unique in right dataset; not a one-to-one merge"
                )

        elif validate in ["one_to_many", "1:m"]:
            if not left_unique:
                raise MergeError(
                    "Merge keys are not unique in left dataset; not a one-to-many merge"
                )

        elif validate in ["many_to_one", "m:1"]:
            if not right_unique:
                raise MergeError(
                    "Merge keys are not unique in right dataset; "
                    "not a many-to-one merge"
                )

        elif validate in ["many_to_many", "m:m"]:
            pass

        else:
            raise ValueError(
                f'"{validate}" is not a valid argument. '
                "Valid arguments are:\n"
                '- "1:1"\n'
                '- "1:m"\n'
                '- "m:1"\n'
                '- "m:m"\n'
                '- "one_to_one"\n'
                '- "one_to_many"\n'
                '- "many_to_one"\n'
                '- "many_to_many"'
            )


def get_join_indexers(
    left_keys: list[ArrayLike],
    right_keys: list[ArrayLike],
    sort: bool = False,
    how: JoinHow = "inner",
) -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
    """

    Parameters
    ----------
    left_keys : list[ndarray, ExtensionArray, Index, Series]
    right_keys : list[ndarray, ExtensionArray, Index, Series]
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp] or None
        Indexer into the left_keys.
    np.ndarray[np.intp] or None
        Indexer into the right_keys.
    """
    assert len(left_keys) == len(
        right_keys
    ), "left_keys and right_keys must be the same length"

    # fast-path for empty left/right
    left_n = len(left_keys[0])
    right_n = len(right_keys[0])
    if left_n == 0:
        if how in ["left", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["right", "outer"]:
            return _get_no_sort_one_missing_indexer(right_n, True)
    elif right_n == 0:
        if how in ["right", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["left", "outer"]:
            return _get_no_sort_one_missing_indexer(left_n, False)

    lkey: ArrayLike
    rkey: ArrayLike
    if len(left_keys) > 1:
        # get left & right join labels and num. of levels at each location
        mapped = (
            _factorize_keys(left_keys[n], right_keys[n], sort=sort)
            for n in range(len(left_keys))
        )
        zipped = zip(*mapped)
        llab, rlab, shape = (list(x) for x in zipped)

        # get flat i8 keys from label lists
        lkey, rkey = _get_join_keys(llab, rlab, tuple(shape), sort)
    else:
        lkey = left_keys[0]
        rkey = right_keys[0]

    left = Index(lkey)
    right = Index(rkey)

    if (
        left.is_monotonic_increasing
        and right.is_monotonic_increasing
        and (left.is_unique or right.is_unique)
    ):
        _, lidx, ridx = left.join(right, how=how, return_indexers=True, sort=sort)
    else:
        lidx, ridx = get_join_indexers_non_unique(
            left._values, right._values, sort, how
        )

    if lidx is not None and is_range_indexer(lidx, len(left)):
        lidx = None
    if ridx is not None and is_range_indexer(ridx, len(right)):
        ridx = None
    return lidx, ridx


def get_join_indexers_non_unique(
    left: ArrayLike,
    right: ArrayLike,
    sort: bool = False,
    how: JoinHow = "inner",
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Get join indexers for left and right.

    Parameters
    ----------
    left : ArrayLike
    right : ArrayLike
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp]
        Indexer into left.
    np.ndarray[np.intp]
        Indexer into right.
    """
    lkey, rkey, count = _factorize_keys(left, right, sort=sort)
    if how == "left":
        lidx, ridx = libjoin.left_outer_join(lkey, rkey, count, sort=sort)
    elif how == "right":
        ridx, lidx = libjoin.left_outer_join(rkey, lkey, count, sort=sort)
    elif how == "inner":
        lidx, ridx = libjoin.inner_join(lkey, rkey, count, sort=sort)
    elif how == "outer":
        lidx, ridx = libjoin.full_outer_join(lkey, rkey, count)
    return lidx, ridx


def restore_dropped_levels_multijoin(
    left: MultiIndex,
    right: MultiIndex,
    dropped_level_names,
    join_index: Index,
    lindexer: npt.NDArray[np.intp],
    rindexer: npt.NDArray[np.intp],
) -> tuple[FrozenList, FrozenList, FrozenList]:
    """
    *this is an internal non-public method*

    Returns the levels, labels and names of a multi-index to multi-index join.
    Depending on the type of join, this method restores the appropriate
    dropped levels of the joined multi-index.
    The method relies on lindexer, rindexer which hold the index positions of
    left and right, where a join was feasible

    Parameters
    ----------
    left : MultiIndex
        left index
    right : MultiIndex
        right index
    dropped_level_names : str array
        list of non-common level names
    join_index : Index
        the index of the join between the
        common levels of left and right
    lindexer : np.ndarray[np.intp]
        left indexer
    rindexer : np.ndarray[np.intp]
        right indexer

    Returns
    -------
    levels : list of Index
        levels of combined multiindexes
    labels : np.ndarray[np.intp]
        labels of combined multiindexes
    names : List[Hashable]
        names of combined multiindex levels

    """

    def _convert_to_multiindex(index: Index) -> MultiIndex:
        if isinstance(index, MultiIndex):
            return index
        else:
            return MultiIndex.from_arrays([index._values], names=[index.name])

    # For multi-multi joins with one overlapping level,
    # the returned index if of type Index
    # Assure that join_index is of type MultiIndex
    # so that dropped levels can be appended
    join_index = _convert_to_multiindex(join_index)

    join_levels = join_index.levels
    join_codes = join_index.codes
    join_names = join_index.names

    # Iterate through the levels that must be restored
    for dropped_level_name in dropped_level_names:
        if dropped_level_name in left.names:
            idx = left
            indexer = lindexer
        else:
            idx = right
            indexer = rindexer

        # The index of the level name to be restored
        name_idx = idx.names.index(dropped_level_name)

        restore_levels = idx.levels[name_idx]
        # Inject -1 in the codes list where a join was not possible
        # IOW indexer[i]=-1
        codes = idx.codes[name_idx]
        if indexer is None:
            restore_codes = codes
        else:
            restore_codes = algos.take_nd(codes, indexer, fill_value=-1)

        # error: Cannot determine type of "__add__"
        join_levels = join_levels + [restore_levels]  # type: ignore[has-type]
        join_codes = join_codes + [restore_codes]  # type: ignore[has-type]
        join_names = join_names + [dropped_level_name]

    return join_levels, join_codes, join_names


class _OrderedMerge(_MergeOperation):
    _merge_type = "ordered_merge"

    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        on: IndexLabel | None = None,
        left_on: IndexLabel | None = None,
        right_on: IndexLabel | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Suffixes = ("_x", "_y"),
        fill_method: str | None = None,
        how: JoinHow | Literal["asof"] = "outer",
    ) -> None:
        self.fill_method = fill_method
        _MergeOperation.__init__(
            self,
            left,
            right,
            on=on,
            left_on=left_on,
            left_index=left_index,
            right_index=right_index,
            right_on=right_on,
            how=how,
            suffixes=suffixes,
            sort=True,  # factorize sorts
        )

    def get_result(self, copy: bool | None = True) -> DataFrame:
        join_index, left_indexer, right_indexer = self._get_join_info()

        left_join_indexer: npt.NDArray[np.intp] | None
        right_join_indexer: npt.NDArray[np.intp] | None

        if self.fill_method == "ffill":
            if left_indexer is None:
                left_join_indexer = None
            else:
                left_join_indexer = libjoin.ffill_indexer(left_indexer)
            if right_indexer is None:
                right_join_indexer = None
            else:
                right_join_indexer = libjoin.ffill_indexer(right_indexer)
        elif self.fill_method is None:
            left_join_indexer = left_indexer
            right_join_indexer = right_indexer
        else:
            raise ValueError("fill_method must be 'ffill' or None")

        result = self._reindex_and_concat(
            join_index, left_join_indexer, right_join_indexer, copy=copy
        )
        self._maybe_add_join_keys(result, left_indexer, right_indexer)

        return result


def _asof_by_function(direction: str):
    name = f"asof_join_{direction}_on_X_by_Y"
    return getattr(libjoin, name, None)


class _AsOfMerge(_OrderedMerge):
    _merge_type = "asof_merge"

    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        on: IndexLabel | None = None,
        left_on: IndexLabel | None = None,
        right_on: IndexLabel | None = None,
        left_index: bool = False,
        right_index: bool = False,
        by=None,
        left_by=None,
        right_by=None,
        suffixes: Suffixes = ("_x", "_y"),
        how: Literal["asof"] = "asof",
        tolerance=None,
        allow_exact_matches: bool = True,
        direction: str = "backward",
    ) -> None:
        self.by = by
        self.left_by = left_by
        self.right_by = right_by
        self.tolerance = tolerance
        self.allow_exact_matches = allow_exact_matches
        self.direction = direction

        # check 'direction' is valid
        if self.direction not in ["backward", "forward", "nearest"]:
            raise MergeError(f"direction invalid: {self.direction}")

        # validate allow_exact_matches
        if not is_bool(self.allow_exact_matches):
            msg = (
                "allow_exact_matches must be boolean, "
                f"passed {self.allow_exact_matches}"
            )
            raise MergeError(msg)

        _OrderedMerge.__init__(
            self,
            left,
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            suffixes=suffixes,
            fill_method=None,
        )

    def _validate_left_right_on(self, left_on, right_on):
        left_on, right_on = super()._validate_left_right_on(left_on, right_on)

        # we only allow on to be a single item for on
        if len(left_on) != 1 and not self.left_index:
            raise MergeError("can only asof on a key for left")

        if len(right_on) != 1 and not self.right_index:
            raise MergeError("can only asof on a key for right")

        if self.left_index and isinstance(self.left.index, MultiIndex):
            raise MergeError("left can only have one index")

        if self.right_index and isinstance(self.right.index, MultiIndex):
            raise MergeError("right can only have one index")

        # set 'by' columns
        if self.by is not None:
            if self.left_by is not None or self.right_by is not None:
                raise MergeError("Can only pass by OR left_by and right_by")
            self.left_by = self.right_by = self.by
        if self.left_by is None and self.right_by is not None:
            raise MergeError("missing left_by")
        if self.left_by is not None and self.right_by is None:
            raise MergeError("missing right_by")

        # GH#29130 Check that merge keys do not have dtype object
        if not self.left_index:
            left_on_0 = left_on[0]
            if isinstance(left_on_0, _known):
                lo_dtype = left_on_0.dtype
            else:
                lo_dtype = (
                    self.left._get_label_or_level_values(left_on_0).dtype
                    if left_on_0 in self.left.columns
                    else self.left.index.get_level_values(left_on_0)
                )
        else:
            lo_dtype = self.left.index.dtype

        if not self.right_index:
            right_on_0 = right_on[0]
            if isinstance(right_on_0, _known):
                ro_dtype = right_on_0.dtype
            else:
                ro_dtype = (
                    self.right._get_label_or_level_values(right_on_0).dtype
                    if right_on_0 in self.right.columns
                    else self.right.index.get_level_values(right_on_0)
                )
        else:
            ro_dtype = self.right.index.dtype

        if (
            is_object_dtype(lo_dtype)
            or is_object_dtype(ro_dtype)
            or is_string_dtype(lo_dtype)
            or is_string_dtype(ro_dtype)
        ):
            raise MergeError(
                f"Incompatible merge dtype, {repr(ro_dtype)} and "
                f"{repr(lo_dtype)}, both sides must have numeric dtype"
            )

        # add 'by' to our key-list so we can have it in the
        # output as a key
        if self.left_by is not None:
            if not is_list_like(self.left_by):
                self.left_by = [self.left_by]
            if not is_list_like(self.right_by):
                self.right_by = [self.right_by]

            if len(self.left_by) != len(self.right_by):
                raise MergeError("left_by and right_by must be the same length")

            left_on = self.left_by + list(left_on)
            right_on = self.right_by + list(right_on)

        return left_on, right_on

    def _maybe_require_matching_dtypes(
        self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]
    ) -> None:
        # TODO: why do we do this for AsOfMerge but not the others?

        def _check_dtype_match(left: ArrayLike, right: ArrayLike, i: int):
            if left.dtype != right.dtype:
                if isinstance(left.dtype, CategoricalDtype) and isinstance(
                    right.dtype, CategoricalDtype
                ):
                    # The generic error message is confusing for categoricals.
                    #
                    # In this function, the join keys include both the original
                    # ones of the merge_asof() call, and also the keys passed
                    # to its by= argument. Unordered but equal categories
                    # are not supported for the former, but will fail
                    # later with a ValueError, so we don't *need* to check
                    # for them here.
                    msg = (
                        f"incompatible merge keys [{i}] {repr(left.dtype)} and "
                        f"{repr(right.dtype)}, both sides category, but not equal ones"
                    )
                else:
                    msg = (
                        f"incompatible merge keys [{i}] {repr(left.dtype)} and "
                        f"{repr(right.dtype)}, must be the same type"
                    )
                raise MergeError(msg)

        # validate index types are the same
        for i, (lk, rk) in enumerate(zip(left_join_keys, right_join_keys)):
            _check_dtype_match(lk, rk, i)

        if self.left_index:
            lt = self.left.index._values
        else:
            lt = left_join_keys[-1]

        if self.right_index:
            rt = self.right.index._values
        else:
            rt = right_join_keys[-1]

        _check_dtype_match(lt, rt, 0)

    def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None:
        # validate tolerance; datetime.timedelta or Timedelta if we have a DTI
        if self.tolerance is not None:
            if self.left_index:
                lt = self.left.index._values
            else:
                lt = left_join_keys[-1]

            msg = (
                f"incompatible tolerance {self.tolerance}, must be compat "
                f"with type {repr(lt.dtype)}"
            )

            if needs_i8_conversion(lt.dtype) or (
                isinstance(lt, ArrowExtensionArray) and lt.dtype.kind in "mM"
            ):
                if not isinstance(self.tolerance, datetime.timedelta):
                    raise MergeError(msg)
                if self.tolerance < Timedelta(0):
                    raise MergeError("tolerance must be positive")

            elif is_integer_dtype(lt.dtype):
                if not is_integer(self.tolerance):
                    raise MergeError(msg)
                if self.tolerance < 0:
                    raise MergeError("tolerance must be positive")

            elif is_float_dtype(lt.dtype):
                if not is_number(self.tolerance):
                    raise MergeError(msg)
                # error: Unsupported operand types for > ("int" and "Number")
                if self.tolerance < 0:  # type: ignore[operator]
                    raise MergeError("tolerance must be positive")

            else:
                raise MergeError("key must be integer, timestamp or float")

    def _convert_values_for_libjoin(
        self, values: AnyArrayLike, side: str
    ) -> np.ndarray:
        # we require sortedness and non-null values in the join keys
        if not Index(values).is_monotonic_increasing:
            if isna(values).any():
                raise ValueError(f"Merge keys contain null values on {side} side")
            raise ValueError(f"{side} keys must be sorted")

        if isinstance(values, ArrowExtensionArray):
            values = values._maybe_convert_datelike_array()

        if needs_i8_conversion(values.dtype):
            values = values.view("i8")

        elif isinstance(values, BaseMaskedArray):
            # we've verified above that no nulls exist
            values = values._data
        elif isinstance(values, ExtensionArray):
            values = values.to_numpy()

        # error: Incompatible return value type (got "Union[ExtensionArray,
        # Any, ndarray[Any, Any], ndarray[Any, dtype[Any]], Index, Series]",
        # expected "ndarray[Any, Any]")
        return values  # type: ignore[return-value]

    def _get_join_indexers(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """return the join indexers"""

        # values to compare
        left_values = (
            self.left.index._values if self.left_index else self.left_join_keys[-1]
        )
        right_values = (
            self.right.index._values if self.right_index else self.right_join_keys[-1]
        )

        # _maybe_require_matching_dtypes already checked for dtype matching
        assert left_values.dtype == right_values.dtype

        tolerance = self.tolerance
        if tolerance is not None:
            # TODO: can we reuse a tolerance-conversion function from
            #  e.g. TimedeltaIndex?
            if needs_i8_conversion(left_values.dtype) or (
                isinstance(left_values, ArrowExtensionArray)
                and left_values.dtype.kind in "mM"
            ):
                tolerance = Timedelta(tolerance)
                # TODO: we have no test cases with PeriodDtype here; probably
                #  need to adjust tolerance for that case.
                if left_values.dtype.kind in "mM":
                    # Make sure the i8 representation for tolerance
                    #  matches that for left_values/right_values.
                    if isinstance(left_values, ArrowExtensionArray):
                        unit = left_values.dtype.pyarrow_dtype.unit
                    else:
                        unit = ensure_wrapped_if_datetimelike(left_values).unit
                    tolerance = tolerance.as_unit(unit)

                tolerance = tolerance._value

        # initial type conversion as needed
        left_values = self._convert_values_for_libjoin(left_values, "left")
        right_values = self._convert_values_for_libjoin(right_values, "right")

        # a "by" parameter requires special handling
        if self.left_by is not None:
            # remove 'on' parameter from values if one existed
            if self.left_index and self.right_index:
                left_join_keys = self.left_join_keys
                right_join_keys = self.right_join_keys
            else:
                left_join_keys = self.left_join_keys[0:-1]
                right_join_keys = self.right_join_keys[0:-1]

            mapped = [
                _factorize_keys(
                    left_join_keys[n],
                    right_join_keys[n],
                    sort=False,
                )
                for n in range(len(left_join_keys))
            ]

            if len(left_join_keys) == 1:
                left_by_values = mapped[0][0]
                right_by_values = mapped[0][1]
            else:
                arrs = [np.concatenate(m[:2]) for m in mapped]
                shape = tuple(m[2] for m in mapped)
                group_index = get_group_index(
                    arrs, shape=shape, sort=False, xnull=False
                )
                left_len = len(left_join_keys[0])
                left_by_values = group_index[:left_len]
                right_by_values = group_index[left_len:]

            left_by_values = ensure_int64(left_by_values)
            right_by_values = ensure_int64(right_by_values)

            # choose appropriate function by type
            func = _asof_by_function(self.direction)
            return func(
                left_values,
                right_values,
                left_by_values,
                right_by_values,
                self.allow_exact_matches,
                tolerance,
            )
        else:
            # choose appropriate function by type
            func = _asof_by_function(self.direction)
            return func(
                left_values,
                right_values,
                None,
                None,
                self.allow_exact_matches,
                tolerance,
                False,
            )


def _get_multiindex_indexer(
    join_keys: list[ArrayLike], index: MultiIndex, sort: bool
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    # left & right join labels and num. of levels at each location
    mapped = (
        _factorize_keys(index.levels[n]._values, join_keys[n], sort=sort)
        for n in range(index.nlevels)
    )
    zipped = zip(*mapped)
    rcodes, lcodes, shape = (list(x) for x in zipped)
    if sort:
        rcodes = list(map(np.take, rcodes, index.codes))
    else:
        i8copy = lambda a: a.astype("i8", subok=False, copy=True)
        rcodes = list(map(i8copy, index.codes))

    # fix right labels if there were any nulls
    for i, join_key in enumerate(join_keys):
        mask = index.codes[i] == -1
        if mask.any():
            # check if there already was any nulls at this location
            # if there was, it is factorized to `shape[i] - 1`
            a = join_key[lcodes[i] == shape[i] - 1]
            if a.size == 0 or not a[0] != a[0]:
                shape[i] += 1

            rcodes[i][mask] = shape[i] - 1

    # get flat i8 join keys
    lkey, rkey = _get_join_keys(lcodes, rcodes, tuple(shape), sort)
    return lkey, rkey


def _get_empty_indexer() -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Return empty join indexers."""
    return (
        np.array([], dtype=np.intp),
        np.array([], dtype=np.intp),
    )


def _get_no_sort_one_missing_indexer(
    n: int, left_missing: bool
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Return join indexers where all of one side is selected without sorting
    and none of the other side is selected.

    Parameters
    ----------
    n : int
        Length of indexers to create.
    left_missing : bool
        If True, the left indexer will contain only -1's.
        If False, the right indexer will contain only -1's.

    Returns
    -------
    np.ndarray[np.intp]
        Left indexer
    np.ndarray[np.intp]
        Right indexer
    """
    idx = np.arange(n, dtype=np.intp)
    idx_missing = np.full(shape=n, fill_value=-1, dtype=np.intp)
    if left_missing:
        return idx_missing, idx
    return idx, idx_missing


def _left_join_on_index(
    left_ax: Index, right_ax: Index, join_keys: list[ArrayLike], sort: bool = False
) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp]]:
    if isinstance(right_ax, MultiIndex):
        lkey, rkey = _get_multiindex_indexer(join_keys, right_ax, sort=sort)
    else:
        # error: Incompatible types in assignment (expression has type
        # "Union[Union[ExtensionArray, ndarray[Any, Any]], Index, Series]",
        # variable has type "ndarray[Any, dtype[signedinteger[Any]]]")
        lkey = join_keys[0]  # type: ignore[assignment]
        # error: Incompatible types in assignment (expression has type "Index",
        # variable has type "ndarray[Any, dtype[signedinteger[Any]]]")
        rkey = right_ax._values  # type: ignore[assignment]

    left_key, right_key, count = _factorize_keys(lkey, rkey, sort=sort)
    left_indexer, right_indexer = libjoin.left_outer_join(
        left_key, right_key, count, sort=sort
    )

    if sort or len(left_ax) != len(left_indexer):
        # if asked to sort or there are 1-to-many matches
        join_index = left_ax.take(left_indexer)
        return join_index, left_indexer, right_indexer

    # left frame preserves order & length of its index
    return left_ax, None, right_indexer


def _factorize_keys(
    lk: ArrayLike, rk: ArrayLike, sort: bool = True
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
    """
    Encode left and right keys as enumerated types.

    This is used to get the join indexers to be used when merging DataFrames.

    Parameters
    ----------
    lk : ndarray, ExtensionArray
        Left key.
    rk : ndarray, ExtensionArray
        Right key.
    sort : bool, defaults to True
        If True, the encoding is done such that the unique elements in the
        keys are sorted.

    Returns
    -------
    np.ndarray[np.intp]
        Left (resp. right if called with `key='right'`) labels, as enumerated type.
    np.ndarray[np.intp]
        Right (resp. left if called with `key='right'`) labels, as enumerated type.
    int
        Number of unique elements in union of left and right labels.

    See Also
    --------
    merge : Merge DataFrame or named Series objects
        with a database-style join.
    algorithms.factorize : Encode the object as an enumerated type
        or categorical variable.

    Examples
    --------
    >>> lk = np.array(["a", "c", "b"])
    >>> rk = np.array(["a", "c"])

    Here, the unique values are `'a', 'b', 'c'`. With the default
    `sort=True`, the encoding will be `{0: 'a', 1: 'b', 2: 'c'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk)
    (array([0, 2, 1]), array([0, 2]), 3)

    With the `sort=False`, the encoding will correspond to the order
    in which the unique elements first appear: `{0: 'a', 1: 'c', 2: 'b'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk, sort=False)
    (array([0, 1, 2]), array([0, 1]), 3)
    """
    # TODO: if either is a RangeIndex, we can likely factorize more efficiently?

    if (
        isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype)
    ) or (lib.is_np_dtype(lk.dtype, "M") and lib.is_np_dtype(rk.dtype, "M")):
        # Extract the ndarray (UTC-localized) values
        # Note: we dont need the dtypes to match, as these can still be compared
        lk, rk = cast("DatetimeArray", lk)._ensure_matching_resos(rk)
        lk = cast("DatetimeArray", lk)._ndarray
        rk = cast("DatetimeArray", rk)._ndarray

    elif (
        isinstance(lk.dtype, CategoricalDtype)
        and isinstance(rk.dtype, CategoricalDtype)
        and lk.dtype == rk.dtype
    ):
        assert isinstance(lk, Categorical)
        assert isinstance(rk, Categorical)
        # Cast rk to encoding so we can compare codes with lk

        rk = lk._encode_with_my_categories(rk)

        lk = ensure_int64(lk.codes)
        rk = ensure_int64(rk.codes)

    elif isinstance(lk, ExtensionArray) and lk.dtype == rk.dtype:
        if (isinstance(lk.dtype, ArrowDtype) and is_string_dtype(lk.dtype)) or (
            isinstance(lk.dtype, StringDtype) and lk.dtype.storage == "pyarrow"
        ):
            import pyarrow as pa
            import pyarrow.compute as pc

            len_lk = len(lk)
            lk = lk._pa_array  # type: ignore[attr-defined]
            rk = rk._pa_array  # type: ignore[union-attr]
            dc = (
                pa.chunked_array(lk.chunks + rk.chunks)  # type: ignore[union-attr]
                .combine_chunks()
                .dictionary_encode()
            )

            llab, rlab, count = (
                pc.fill_null(dc.indices[slice(len_lk)], -1)
                .to_numpy()
                .astype(np.intp, copy=False),
                pc.fill_null(dc.indices[slice(len_lk, None)], -1)
                .to_numpy()
                .astype(np.intp, copy=False),
                len(dc.dictionary),
            )

            if sort:
                uniques = dc.dictionary.to_numpy(zero_copy_only=False)
                llab, rlab = _sort_labels(uniques, llab, rlab)

            if dc.null_count > 0:
                lmask = llab == -1
                lany = lmask.any()
                rmask = rlab == -1
                rany = rmask.any()
                if lany:
                    np.putmask(llab, lmask, count)
                if rany:
                    np.putmask(rlab, rmask, count)
                count += 1
            return llab, rlab, count

        if not isinstance(lk, BaseMaskedArray) and not (
            # exclude arrow dtypes that would get cast to object
            isinstance(lk.dtype, ArrowDtype)
            and (
                is_numeric_dtype(lk.dtype.numpy_dtype)
                or is_string_dtype(lk.dtype)
                and not sort
            )
        ):
            lk, _ = lk._values_for_factorize()

            # error: Item "ndarray" of "Union[Any, ndarray]" has no attribute
            # "_values_for_factorize"
            rk, _ = rk._values_for_factorize()  # type: ignore[union-attr]

    if needs_i8_conversion(lk.dtype) and lk.dtype == rk.dtype:
        # GH#23917 TODO: Needs tests for non-matching dtypes
        # GH#23917 TODO: needs tests for case where lk is integer-dtype
        #  and rk is datetime-dtype
        lk = np.asarray(lk, dtype=np.int64)
        rk = np.asarray(rk, dtype=np.int64)

    klass, lk, rk = _convert_arrays_and_get_rizer_klass(lk, rk)

    rizer = klass(max(len(lk), len(rk)))

    if isinstance(lk, BaseMaskedArray):
        assert isinstance(rk, BaseMaskedArray)
        llab = rizer.factorize(lk._data, mask=lk._mask)
        rlab = rizer.factorize(rk._data, mask=rk._mask)
    elif isinstance(lk, ArrowExtensionArray):
        assert isinstance(rk, ArrowExtensionArray)
        # we can only get here with numeric dtypes
        # TODO: Remove when we have a Factorizer for Arrow
        llab = rizer.factorize(
            lk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype), mask=lk.isna()
        )
        rlab = rizer.factorize(
            rk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype), mask=rk.isna()
        )
    else:
        # Argument 1 to "factorize" of "ObjectFactorizer" has incompatible type
        # "Union[ndarray[Any, dtype[signedinteger[_64Bit]]],
        # ndarray[Any, dtype[object_]]]"; expected "ndarray[Any, dtype[object_]]"
        llab = rizer.factorize(lk)  # type: ignore[arg-type]
        rlab = rizer.factorize(rk)  # type: ignore[arg-type]
    assert llab.dtype == np.dtype(np.intp), llab.dtype
    assert rlab.dtype == np.dtype(np.intp), rlab.dtype

    count = rizer.get_count()

    if sort:
        uniques = rizer.uniques.to_array()
        llab, rlab = _sort_labels(uniques, llab, rlab)

    # NA group
    lmask = llab == -1
    lany = lmask.any()
    rmask = rlab == -1
    rany = rmask.any()

    if lany or rany:
        if lany:
            np.putmask(llab, lmask, count)
        if rany:
            np.putmask(rlab, rmask, count)
        count += 1

    return llab, rlab, count


def _convert_arrays_and_get_rizer_klass(
    lk: ArrayLike, rk: ArrayLike
) -> tuple[type[libhashtable.Factorizer], ArrayLike, ArrayLike]:
    klass: type[libhashtable.Factorizer]
    if is_numeric_dtype(lk.dtype):
        if lk.dtype != rk.dtype:
            dtype = find_common_type([lk.dtype, rk.dtype])
            if isinstance(dtype, ExtensionDtype):
                cls = dtype.construct_array_type()
                if not isinstance(lk, ExtensionArray):
                    lk = cls._from_sequence(lk, dtype=dtype, copy=False)
                else:
                    lk = lk.astype(dtype, copy=False)

                if not isinstance(rk, ExtensionArray):
                    rk = cls._from_sequence(rk, dtype=dtype, copy=False)
                else:
                    rk = rk.astype(dtype, copy=False)
            else:
                lk = lk.astype(dtype, copy=False)
                rk = rk.astype(dtype, copy=False)
        if isinstance(lk, BaseMaskedArray):
            #  Invalid index type "type" for "Dict[Type[object], Type[Factorizer]]";
            #  expected type "Type[object]"
            klass = _factorizers[lk.dtype.type]  # type: ignore[index]
        elif isinstance(lk.dtype, ArrowDtype):
            klass = _factorizers[lk.dtype.numpy_dtype.type]
        else:
            klass = _factorizers[lk.dtype.type]

    else:
        klass = libhashtable.ObjectFactorizer
        lk = ensure_object(lk)
        rk = ensure_object(rk)
    return klass, lk, rk


def _sort_labels(
    uniques: np.ndarray, left: npt.NDArray[np.intp], right: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    llength = len(left)
    labels = np.concatenate([left, right])

    _, new_labels = algos.safe_sort(uniques, labels, use_na_sentinel=True)
    new_left, new_right = new_labels[:llength], new_labels[llength:]

    return new_left, new_right


def _get_join_keys(
    llab: list[npt.NDArray[np.int64 | np.intp]],
    rlab: list[npt.NDArray[np.int64 | np.intp]],
    shape: Shape,
    sort: bool,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    # how many levels can be done without overflow
    nlev = next(
        lev
        for lev in range(len(shape), 0, -1)
        if not is_int64_overflow_possible(shape[:lev])
    )

    # get keys for the first `nlev` levels
    stride = np.prod(shape[1:nlev], dtype="i8")
    lkey = stride * llab[0].astype("i8", subok=False, copy=False)
    rkey = stride * rlab[0].astype("i8", subok=False, copy=False)

    for i in range(1, nlev):
        with np.errstate(divide="ignore"):
            stride //= shape[i]
        lkey += llab[i] * stride
        rkey += rlab[i] * stride

    if nlev == len(shape):  # all done!
        return lkey, rkey

    # densify current keys to avoid overflow
    lkey, rkey, count = _factorize_keys(lkey, rkey, sort=sort)

    llab = [lkey] + llab[nlev:]
    rlab = [rkey] + rlab[nlev:]
    shape = (count,) + shape[nlev:]

    return _get_join_keys(llab, rlab, shape, sort)


def _should_fill(lname, rname) -> bool:
    if not isinstance(lname, str) or not isinstance(rname, str):
        return True
    return lname == rname


def _any(x) -> bool:
    return x is not None and com.any_not_none(*x)


def _validate_operand(obj: DataFrame | Series) -> DataFrame:
    if isinstance(obj, ABCDataFrame):
        return obj
    elif isinstance(obj, ABCSeries):
        if obj.name is None:
            raise ValueError("Cannot merge a Series without a name")
        return obj.to_frame()
    else:
        raise TypeError(
            f"Can only merge Series or DataFrame objects, a {type(obj)} was passed"
        )


def _items_overlap_with_suffix(
    left: Index, right: Index, suffixes: Suffixes
) -> tuple[Index, Index]:
    """
    Suffixes type validation.

    If two indices overlap, add suffixes to overlapping entries.

    If corresponding suffix is empty, the entry is simply converted to string.

    """
    if not is_list_like(suffixes, allow_sets=False) or isinstance(suffixes, dict):
        raise TypeError(
            f"Passing 'suffixes' as a {type(suffixes)}, is not supported. "
            "Provide 'suffixes' as a tuple instead."
        )

    to_rename = left.intersection(right)
    if len(to_rename) == 0:
        return left, right

    lsuffix, rsuffix = suffixes

    if not lsuffix and not rsuffix:
        raise ValueError(f"columns overlap but no suffix specified: {to_rename}")

    def renamer(x, suffix: str | None):
        """
        Rename the left and right indices.

        If there is overlap, and suffix is not None, add
        suffix, otherwise, leave it as-is.

        Parameters
        ----------
        x : original column name
        suffix : str or None

        Returns
        -------
        x : renamed column name
        """
        if x in to_rename and suffix is not None:
            return f"{x}{suffix}"
        return x

    lrenamer = partial(renamer, suffix=lsuffix)
    rrenamer = partial(renamer, suffix=rsuffix)

    llabels = left._transform_index(lrenamer)
    rlabels = right._transform_index(rrenamer)

    dups = []
    if not llabels.is_unique:
        # Only warn when duplicates are caused because of suffixes, already duplicated
        # columns in origin should not warn
        dups = llabels[(llabels.duplicated()) & (~left.duplicated())].tolist()
    if not rlabels.is_unique:
        dups.extend(rlabels[(rlabels.duplicated()) & (~right.duplicated())].tolist())
    if dups:
        raise MergeError(
            f"Passing 'suffixes' which cause duplicate columns {set(dups)} is "
            f"not allowed.",
        )

    return llabels, rlabels

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\numeric.py ===
import builtins
import functools
import itertools
import math
import numbers
import operator
import sys
import warnings

import numpy as np
from numpy.exceptions import AxisError

from . import multiarray, numerictypes, overrides, shape_base, umath
from . import numerictypes as nt
from ._ufunc_config import errstate
from .multiarray import (  # noqa: F401
    ALLOW_THREADS,
    BUFSIZE,
    CLIP,
    MAXDIMS,
    MAY_SHARE_BOUNDS,
    MAY_SHARE_EXACT,
    RAISE,
    WRAP,
    arange,
    array,
    asanyarray,
    asarray,
    ascontiguousarray,
    asfortranarray,
    broadcast,
    can_cast,
    concatenate,
    copyto,
    dot,
    dtype,
    empty,
    empty_like,
    flatiter,
    from_dlpack,
    frombuffer,
    fromfile,
    fromiter,
    fromstring,
    inner,
    lexsort,
    matmul,
    may_share_memory,
    min_scalar_type,
    ndarray,
    nditer,
    nested_iters,
    normalize_axis_index,
    promote_types,
    putmask,
    result_type,
    shares_memory,
    vdot,
    vecdot,
    where,
    zeros,
)
from .overrides import finalize_array_function_like, set_module
from .umath import NAN, PINF, invert, multiply, sin

bitwise_not = invert
ufunc = type(sin)
newaxis = None

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'newaxis', 'ndarray', 'flatiter', 'nditer', 'nested_iters', 'ufunc',
    'arange', 'array', 'asarray', 'asanyarray', 'ascontiguousarray',
    'asfortranarray', 'zeros', 'count_nonzero', 'empty', 'broadcast', 'dtype',
    'fromstring', 'fromfile', 'frombuffer', 'from_dlpack', 'where',
    'argwhere', 'copyto', 'concatenate', 'lexsort', 'astype',
    'can_cast', 'promote_types', 'min_scalar_type',
    'result_type', 'isfortran', 'empty_like', 'zeros_like', 'ones_like',
    'correlate', 'convolve', 'inner', 'dot', 'outer', 'vdot', 'roll',
    'rollaxis', 'moveaxis', 'cross', 'tensordot', 'little_endian',
    'fromiter', 'array_equal', 'array_equiv', 'indices', 'fromfunction',
    'isclose', 'isscalar', 'binary_repr', 'base_repr', 'ones',
    'identity', 'allclose', 'putmask',
    'flatnonzero', 'inf', 'nan', 'False_', 'True_', 'bitwise_not',
    'full', 'full_like', 'matmul', 'vecdot', 'shares_memory',
    'may_share_memory']


def _zeros_like_dispatcher(
    a, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    return (a,)


@array_function_dispatch(_zeros_like_dispatcher)
def zeros_like(
    a, dtype=None, order='K', subok=True, shape=None, *, device=None
):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> np.zeros_like(y)
    array([0.,  0.,  0.])

    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    # needed instead of a 0 to get same result as zeros for string dtypes
    z = zeros(1, dtype=res.dtype)
    multiarray.copyto(res, z, casting='unsafe')
    return res


@finalize_array_function_like
@set_module('numpy')
def ones(shape, dtype=None, order='C', *, device=None, like=None):
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: C
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and order.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty : Return a new uninitialized array.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import numpy as np
    >>> np.ones(5)
    array([1., 1., 1., 1., 1.])

    >>> np.ones((5,), dtype=int)
    array([1, 1, 1, 1, 1])

    >>> np.ones((2, 1))
    array([[1.],
           [1.]])

    >>> s = (2,2)
    >>> np.ones(s)
    array([[1.,  1.],
           [1.,  1.]])

    """
    if like is not None:
        return _ones_with_like(
            like, shape, dtype=dtype, order=order, device=device
        )

    a = empty(shape, dtype, order, device=device)
    multiarray.copyto(a, 1, casting='unsafe')
    return a


_ones_with_like = array_function_dispatch()(ones)


def _ones_like_dispatcher(
    a, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    return (a,)


@array_function_dispatch(_ones_like_dispatcher)
def ones_like(
    a, dtype=None, order='K', subok=True, shape=None, *, device=None
):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    ones : Return a new array setting values to one.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> np.ones_like(y)
    array([1.,  1.,  1.])

    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    multiarray.copyto(res, 1, casting='unsafe')
    return res


def _full_dispatcher(
    shape, fill_value, dtype=None, order=None, *, device=None, like=None
):
    return (like,)


@finalize_array_function_like
@set_module('numpy')
def full(shape, fill_value, dtype=None, order='C', *, device=None, like=None):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or array_like
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, None, means
         ``np.array(fill_value).dtype``.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> import numpy as np
    >>> np.full((2, 2), np.inf)
    array([[inf, inf],
           [inf, inf]])
    >>> np.full((2, 2), 10)
    array([[10, 10],
           [10, 10]])

    >>> np.full((2, 2), [1, 2])
    array([[1, 2],
           [1, 2]])

    """
    if like is not None:
        return _full_with_like(
            like, shape, fill_value, dtype=dtype, order=order, device=device
        )

    if dtype is None:
        fill_value = asarray(fill_value)
        dtype = fill_value.dtype
    a = empty(shape, dtype, order, device=device)
    multiarray.copyto(a, fill_value, casting='unsafe')
    return a


_full_with_like = array_function_dispatch()(full)


def _full_like_dispatcher(
    a, fill_value, dtype=None, order=None, subok=None, shape=None,
    *, device=None
):
    return (a,)


@array_function_dispatch(_full_like_dispatcher)
def full_like(
    a, fill_value, dtype=None, order='K', subok=True, shape=None,
    *, device=None
):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : array_like
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6, dtype=int)
    >>> np.full_like(x, 1)
    array([1, 1, 1, 1, 1, 1])
    >>> np.full_like(x, 0.1)
    array([0, 0, 0, 0, 0, 0])
    >>> np.full_like(x, 0.1, dtype=np.double)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    >>> np.full_like(x, np.nan, dtype=np.double)
    array([nan, nan, nan, nan, nan, nan])

    >>> y = np.arange(6, dtype=np.double)
    >>> np.full_like(y, 0.1)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    >>> y = np.zeros([2, 2, 3], dtype=int)
    >>> np.full_like(y, [0, 0, 255])
    array([[[  0,   0, 255],
            [  0,   0, 255]],
           [[  0,   0, 255],
            [  0,   0, 255]]])
    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    multiarray.copyto(res, fill_value, casting='unsafe')
    return res


def _count_nonzero_dispatcher(a, axis=None, *, keepdims=None):
    return (a,)


@array_function_dispatch(_count_nonzero_dispatcher)
def count_nonzero(a, axis=None, *, keepdims=False):
    """
    Counts the number of non-zero values in the array ``a``.

    The word "non-zero" is in reference to the Python 2.x
    built-in method ``__nonzero__()`` (renamed ``__bool__()``
    in Python 3.x) of Python objects that tests an object's
    "truthfulness". For example, any number is considered
    truthful if it is nonzero, whereas any string is considered
    truthful if it is not the empty string. Thus, this function
    (recursively) counts how many elements in ``a`` (and in
    sub-arrays thereof) have their ``__nonzero__()`` or ``__bool__()``
    method evaluated to ``True``.

    Parameters
    ----------
    a : array_like
        The array for which to count non-zeros.
    axis : int or tuple, optional
        Axis or tuple of axes along which to count non-zeros.
        Default is None, meaning that non-zeros will be counted
        along a flattened version of ``a``.
    keepdims : bool, optional
        If this is set to True, the axes that are counted are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    count : int or array of int
        Number of non-zero values in the array along a given axis.
        Otherwise, the total number of non-zero values in the array
        is returned.

    See Also
    --------
    nonzero : Return the coordinates of all the non-zero values.

    Examples
    --------
    >>> import numpy as np
    >>> np.count_nonzero(np.eye(4))
    4
    >>> a = np.array([[0, 1, 7, 0],
    ...               [3, 0, 2, 19]])
    >>> np.count_nonzero(a)
    5
    >>> np.count_nonzero(a, axis=0)
    array([1, 1, 2, 1])
    >>> np.count_nonzero(a, axis=1)
    array([2, 3])
    >>> np.count_nonzero(a, axis=1, keepdims=True)
    array([[2],
           [3]])
    """
    if axis is None and not keepdims:
        return multiarray.count_nonzero(a)

    a = asanyarray(a)

    # TODO: this works around .astype(bool) not working properly (gh-9847)
    if np.issubdtype(a.dtype, np.character):
        a_bool = a != a.dtype.type()
    else:
        a_bool = a.astype(np.bool, copy=False)

    return a_bool.sum(axis=axis, dtype=np.intp, keepdims=keepdims)


@set_module('numpy')
def isfortran(a):
    """
    Check if the array is Fortran contiguous but *not* C contiguous.

    This function is obsolete. If you only want to check if an array is Fortran
    contiguous use ``a.flags.f_contiguous`` instead.

    Parameters
    ----------
    a : ndarray
        Input array.

    Returns
    -------
    isfortran : bool
        Returns True if the array is Fortran contiguous but *not* C contiguous.


    Examples
    --------

    np.array allows to specify whether the array is written in C-contiguous
    order (last index varies the fastest), or FORTRAN-contiguous order in
    memory (first index varies the fastest).

    >>> import numpy as np
    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False

    >>> b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    >>> b
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(b)
    True


    The transpose of a C-ordered array is a FORTRAN-ordered array.

    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False
    >>> b = a.T
    >>> b
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.isfortran(b)
    True

    C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

    >>> np.isfortran(np.array([1, 2], order='F'))
    False

    """
    return a.flags.fnc


def _argwhere_dispatcher(a):
    return (a,)


@array_function_dispatch(_argwhere_dispatcher)
def argwhere(a):
    """
    Find the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_array : (N, a.ndim) ndarray
        Indices of elements that are non-zero. Indices are grouped by element.
        This array will have shape ``(N, a.ndim)`` where ``N`` is the number of
        non-zero items.

    See Also
    --------
    where, nonzero

    Notes
    -----
    ``np.argwhere(a)`` is almost the same as ``np.transpose(np.nonzero(a))``,
    but produces a result of the correct shape for a 0D array.

    The output of ``argwhere`` is not suitable for indexing arrays.
    For this purpose use ``nonzero(a)`` instead.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2,3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.argwhere(x>1)
    array([[0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])

    """
    # nonzero does not behave well on 0d, so promote to 1d
    if np.ndim(a) == 0:
        a = shape_base.atleast_1d(a)
        # then remove the added dimension
        return argwhere(a)[:, :0]
    return transpose(nonzero(a))


def _flatnonzero_dispatcher(a):
    return (a,)


@array_function_dispatch(_flatnonzero_dispatcher)
def flatnonzero(a):
    """
    Return indices that are non-zero in the flattened version of a.

    This is equivalent to ``np.nonzero(np.ravel(a))[0]``.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    res : ndarray
        Output array, containing the indices of the elements of ``a.ravel()``
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    ravel : Return a 1-D array containing the elements of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(-2, 3)
    >>> x
    array([-2, -1,  0,  1,  2])
    >>> np.flatnonzero(x)
    array([0, 1, 3, 4])

    Use the indices of the non-zero elements as an index array to extract
    these elements:

    >>> x.ravel()[np.flatnonzero(x)]
    array([-2, -1,  1,  2])

    """
    return np.nonzero(np.ravel(a))[0]


def _correlate_dispatcher(a, v, mode=None):
    return (a, v)


@array_function_dispatch(_correlate_dispatcher)
def correlate(a, v, mode='valid'):
    r"""
    Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal
    processing texts [1]_:

    .. math:: c_k = \sum_n a_{n+k} \cdot \overline{v}_n

    with a and v sequences being zero-padded where necessary and
    :math:`\overline v` denoting complex conjugation.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `convolve` docstring.  Note that the default
        is 'valid', unlike `convolve`, which uses 'full'.

    Returns
    -------
    out : ndarray
        Discrete cross-correlation of `a` and `v`.

    See Also
    --------
    convolve : Discrete, linear convolution of two one-dimensional sequences.
    scipy.signal.correlate : uses FFT which has superior performance
        on large arrays.

    Notes
    -----
    The definition of correlation above is not unique and sometimes
    correlation may be defined differently. Another common definition is [1]_:

    .. math:: c'_k = \sum_n a_{n} \cdot \overline{v_{n+k}}

    which is related to :math:`c_k` by :math:`c'_k = c_{-k}`.

    `numpy.correlate` may perform slowly in large arrays (i.e. n = 1e5)
    because it does not use the FFT to compute the convolution; in that case,
    `scipy.signal.correlate` might be preferable.

    References
    ----------
    .. [1] Wikipedia, "Cross-correlation",
           https://en.wikipedia.org/wiki/Cross-correlation

    Examples
    --------
    >>> import numpy as np
    >>> np.correlate([1, 2, 3], [0, 1, 0.5])
    array([3.5])
    >>> np.correlate([1, 2, 3], [0, 1, 0.5], "same")
    array([2. ,  3.5,  3. ])
    >>> np.correlate([1, 2, 3], [0, 1, 0.5], "full")
    array([0.5,  2. ,  3.5,  3. ,  0. ])

    Using complex sequences:

    >>> np.correlate([1+1j, 2, 3-1j], [0, 1, 0.5j], 'full')
    array([ 0.5-0.5j,  1.0+0.j ,  1.5-1.5j,  3.0-1.j ,  0.0+0.j ])

    Note that you get the time reversed, complex conjugated result
    (:math:`\overline{c_{-k}}`) when the two input sequences a and v change
    places:

    >>> np.correlate([0, 1, 0.5j], [1+1j, 2, 3-1j], 'full')
    array([ 0.0+0.j ,  3.0+1.j ,  1.5+1.5j,  1.0+0.j ,  0.5+0.5j])

    """
    return multiarray.correlate2(a, v, mode)


def _convolve_dispatcher(a, v, mode=None):
    return (a, v)


@array_function_dispatch(_convolve_dispatcher)
def convolve(a, v, mode='full'):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    The convolution operator is often seen in signal processing, where it
    models the effect of a linear time-invariant system on a signal [1]_.  In
    probability theory, the sum of two independent random variables is
    distributed according to the convolution of their individual
    distributions.

    If `v` is longer than `a`, the arrays are swapped before computation.

    Parameters
    ----------
    a : (N,) array_like
        First one-dimensional input array.
    v : (M,) array_like
        Second one-dimensional input array.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.

        'same':
          Mode 'same' returns output of length ``max(M, N)``.  Boundary
          effects are still visible.

        'valid':
          Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.

    Returns
    -------
    out : ndarray
        Discrete, linear convolution of `a` and `v`.

    See Also
    --------
    scipy.signal.fftconvolve : Convolve two arrays using the Fast Fourier
                               Transform.
    scipy.linalg.toeplitz : Used to construct the convolution operator.
    polymul : Polynomial multiplication. Same output as convolve, but also
              accepts poly1d objects as input.

    Notes
    -----
    The discrete convolution operation is defined as

    .. math:: (a * v)_n = \\sum_{m = -\\infty}^{\\infty} a_m v_{n - m}

    It can be shown that a convolution :math:`x(t) * y(t)` in time/space
    is equivalent to the multiplication :math:`X(f) Y(f)` in the Fourier
    domain, after appropriate padding (padding is necessary to prevent
    circular convolution).  Since multiplication is more efficient (faster)
    than convolution, the function `scipy.signal.fftconvolve` exploits the
    FFT to calculate the convolution of large data-sets.

    References
    ----------
    .. [1] Wikipedia, "Convolution",
        https://en.wikipedia.org/wiki/Convolution

    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:

    >>> import numpy as np
    >>> np.convolve([1, 2, 3], [0, 1, 0.5])
    array([0. , 1. , 2.5, 4. , 1.5])

    Only return the middle values of the convolution.
    Contains boundary effects, where zeros are taken
    into account:

    >>> np.convolve([1,2,3],[0,1,0.5], 'same')
    array([1. ,  2.5,  4. ])

    The two arrays are of the same length, so there
    is only one position where they completely overlap:

    >>> np.convolve([1,2,3],[0,1,0.5], 'valid')
    array([2.5])

    """
    a, v = array(a, copy=None, ndmin=1), array(v, copy=None, ndmin=1)
    if (len(v) > len(a)):
        a, v = v, a
    if len(a) == 0:
        raise ValueError('a cannot be empty')
    if len(v) == 0:
        raise ValueError('v cannot be empty')
    return multiarray.correlate(a, v[::-1], mode)


def _outer_dispatcher(a, b, out=None):
    return (a, b, out)


@array_function_dispatch(_outer_dispatcher)
def outer(a, b, out=None):
    """
    Compute the outer product of two vectors.

    Given two vectors `a` and `b` of length ``M`` and ``N``, respectively,
    the outer product [1]_ is::

      [[a_0*b_0  a_0*b_1 ... a_0*b_{N-1} ]
       [a_1*b_0    .
       [ ...          .
       [a_{M-1}*b_0            a_{M-1}*b_{N-1} ]]

    Parameters
    ----------
    a : (M,) array_like
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) array_like
        Second input vector.  Input is flattened if
        not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``

    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to dimensions other than 1D and other
                  operations. ``np.multiply.outer(a.ravel(), b.ravel())``
                  is the equivalent.
    linalg.outer : An Array API compatible variation of ``np.outer``,
                   which accepts 1-dimensional inputs only.
    tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``
                is the equivalent.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
           ed., Baltimore, MD, Johns Hopkins University Press, 1996,
           pg. 8.

    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:

    >>> import numpy as np
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    >>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
    >>> im
    array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
           [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
    >>> grid = rl + im
    >>> grid
    array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],
           [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
           [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
           [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
           [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])

    An example using a "vector" of letters:

    >>> x = np.array(['a', 'b', 'c'], dtype=object)
    >>> np.outer(x, [1, 2, 3])
    array([['a', 'aa', 'aaa'],
           ['b', 'bb', 'bbb'],
           ['c', 'cc', 'ccc']], dtype=object)

    """
    a = asarray(a)
    b = asarray(b)
    return multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out)


def _tensordot_dispatcher(a, b, axes=None):
    return (a, b)


@array_function_dispatch(_tensordot_dispatcher)
def tensordot(a, b, axes=2):
    """
    Compute tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : array_like
        Tensors to "dot".

    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    Returns
    -------
    output : ndarray
        The tensor dot product of the input.

    See Also
    --------
    dot, einsum

    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is integer_like, the sequence of axes for evaluation
    will be: from the -Nth axis to the -1th axis in `a`,
    and from the 0th axis to (N-1)th axis in `b`.
    For example, ``axes = 2`` is the equal to
    ``axes = [[-2, -1], [0, 1]]``.
    When N-1 is smaller than 0, or when -N is larger than -1,
    the element of `a` and `b` are defined as the `axes`.

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    The calculation can be referred to ``numpy.einsum``.

    The shape of the result consists of the non-contracted axes of the
    first tensor, followed by the non-contracted axes of the second.

    Examples
    --------
    An example on integer_like:

    >>> a_0 = np.array([[1, 2], [3, 4]])
    >>> b_0 = np.array([[5, 6], [7, 8]])
    >>> c_0 = np.tensordot(a_0, b_0, axes=0)
    >>> c_0.shape
    (2, 2, 2, 2)
    >>> c_0
    array([[[[ 5,  6],
             [ 7,  8]],
            [[10, 12],
             [14, 16]]],
           [[[15, 18],
             [21, 24]],
            [[20, 24],
             [28, 32]]]])

    An example on array_like:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    A slower but equivalent way of computing the same...

    >>> d = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i,j] += a[k,n,i] * b[n,k,j]
    >>> c == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]])

    An extended example taking advantage of the overloading of + and \\*:

    >>> a = np.array(range(1, 9))
    >>> a.shape = (2, 2, 2)
    >>> A = np.array(('a', 'b', 'c', 'd'), dtype=object)
    >>> A.shape = (2, 2)
    >>> a; A
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    array([['a', 'b'],
           ['c', 'd']], dtype=object)

    >>> np.tensordot(a, A) # third argument default is 2 for double-contraction
    array(['abbcccdddd', 'aaaaabbbbbbcccccccdddddddd'], dtype=object)

    >>> np.tensordot(a, A, 1)
    array([[['acc', 'bdd'],
            ['aaacccc', 'bbbdddd']],
           [['aaaaacccccc', 'bbbbbdddddd'],
            ['aaaaaaacccccccc', 'bbbbbbbdddddddd']]], dtype=object)

    >>> np.tensordot(a, A, 0) # tensor product (result too long to incl.)
    array([[[[['a', 'b'],
              ['c', 'd']],
              ...

    >>> np.tensordot(a, A, (0, 1))
    array([[['abbbbb', 'cddddd'],
            ['aabbbbbb', 'ccdddddd']],
           [['aaabbbbbbb', 'cccddddddd'],
            ['aaaabbbbbbbb', 'ccccdddddddd']]], dtype=object)

    >>> np.tensordot(a, A, (2, 1))
    array([[['abb', 'cdd'],
            ['aaabbbb', 'cccdddd']],
           [['aaaaabbbbbb', 'cccccdddddd'],
            ['aaaaaaabbbbbbbb', 'cccccccdddddddd']]], dtype=object)

    >>> np.tensordot(a, A, ((0, 1), (0, 1)))
    array(['abbbcccccddddddd', 'aabbbbccccccdddddddd'], dtype=object)

    >>> np.tensordot(a, A, ((2, 1), (1, 0)))
    array(['acccbbdddd', 'aaaaacccccccbbbbbbdddddddd'], dtype=object)

    """
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a, b = asarray(a), asarray(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = math.prod(as_[axis] for axis in axes_a)
    newshape_a = (math.prod(as_[ax] for ax in notin), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = math.prod(bs[axis] for axis in axes_b)
    newshape_b = (N2, math.prod(bs[ax] for ax in notin))
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = dot(at, bt)
    return res.reshape(olda + oldb)


def _roll_dispatcher(a, shift, axis=None):
    return (a,)


@array_function_dispatch(_roll_dispatcher)
def roll(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Notes
    -----
    Supports rolling over multiple dimensions simultaneously.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> np.roll(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> np.roll(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x2 = np.reshape(x, (2, 5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> np.roll(x2, 1)
    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> np.roll(x2, -1)
    array([[1, 2, 3, 4, 5],
           [6, 7, 8, 9, 0]])
    >>> np.roll(x2, 1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> np.roll(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> np.roll(x2, 1, axis=1)
    array([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])
    >>> np.roll(x2, -1, axis=1)
    array([[1, 2, 3, 4, 0],
           [6, 7, 8, 9, 5]])
    >>> np.roll(x2, (1, 1), axis=(1, 0))
    array([[9, 5, 6, 7, 8],
           [4, 0, 1, 2, 3]])
    >>> np.roll(x2, (2, 1), axis=(1, 0))
    array([[8, 9, 5, 6, 7],
           [3, 4, 0, 1, 2]])

    """
    a = asanyarray(a)
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)

    else:
        axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        broadcasted = broadcast(shift, axis)
        if broadcasted.ndim > 1:
            raise ValueError(
                "'shift' and 'axis' should be scalars or 1D sequences")
        shifts = dict.fromkeys(range(a.ndim), 0)
        for sh, ax in broadcasted:
            shifts[ax] += int(sh)

        rolls = [((slice(None), slice(None)),)] * a.ndim
        for ax, offset in shifts.items():
            offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
            if offset:
                # (original, result), (original, result)
                rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                             (slice(-offset, None), slice(None, offset)))

        result = empty_like(a)
        for indices in itertools.product(*rolls):
            arr_index, res_index = zip(*indices)
            result[res_index] = a[arr_index]

        return result


def _rollaxis_dispatcher(a, axis, start=None):
    return (a,)


@array_function_dispatch(_rollaxis_dispatcher)
def rollaxis(a, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    This function continues to be supported for backward compatibility, but you
    should prefer `moveaxis`. The `moveaxis` function was added in NumPy
    1.11.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int
        The axis to be rolled. The positions of the other axes do not
        change relative to one another.
    start : int, optional
        When ``start <= axis``, the axis is rolled back until it lies in
        this position. When ``start > axis``, the axis is rolled until it
        lies before this position. The default, 0, results in a "complete"
        roll. The following table describes how negative values of ``start``
        are interpreted:

        .. table::
           :align: left

           +-------------------+----------------------+
           |     ``start``     | Normalized ``start`` |
           +===================+======================+
           | ``-(arr.ndim+1)`` | raise ``AxisError``  |
           +-------------------+----------------------+
           | ``-arr.ndim``     | 0                    |
           +-------------------+----------------------+
           | |vdots|           | |vdots|              |
           +-------------------+----------------------+
           | ``-1``            | ``arr.ndim-1``       |
           +-------------------+----------------------+
           | ``0``             | ``0``                |
           +-------------------+----------------------+
           | |vdots|           | |vdots|              |
           +-------------------+----------------------+
           | ``arr.ndim``      | ``arr.ndim``         |
           +-------------------+----------------------+
           | ``arr.ndim + 1``  | raise ``AxisError``  |
           +-------------------+----------------------+

        .. |vdots|   unicode:: U+22EE .. Vertical Ellipsis

    Returns
    -------
    res : ndarray
        For NumPy >= 1.10.0 a view of `a` is always returned. For earlier
        NumPy versions a view of `a` is returned only if the order of the
        axes is changed, otherwise the input array is returned.

    See Also
    --------
    moveaxis : Move array axes to new positions.
    roll : Roll the elements of an array by a number of positions along a
        given axis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ones((3,4,5,6))
    >>> np.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> np.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> np.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)

    """
    n = a.ndim
    axis = normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise AxisError(msg % ('start', -n, 'start', n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        return a[...]
    axes = list(range(n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.transpose(axes)


@set_module("numpy.lib.array_utils")
def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.

    Used internally by multi-axis-checking logic.

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`

    Raises
    ------
    AxisError
        If any axis provided is out of range
    ValueError
        If an axis is repeated

    See also
    --------
    normalize_axis_index : normalizing a single scalar axis
    """
    # Optimization to speed-up the most common cases.
    if not isinstance(axis, (tuple, list)):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # Going via an iterator directly is slower than via list comprehension.
    axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError(f'repeated axis in `{argname}` argument')
        else:
            raise ValueError('repeated axis')
    return axis


def _moveaxis_dispatcher(a, source, destination):
    return (a,)


@array_function_dispatch(_moveaxis_dispatcher)
def moveaxis(a, source, destination):
    """
    Move axes of an array to new positions.

    Other axes remain in their original order.

    Parameters
    ----------
    a : np.ndarray
        The array whose axes should be reordered.
    source : int or sequence of int
        Original positions of the axes to move. These must be unique.
    destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : np.ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    transpose : Permute the dimensions of an array.
    swapaxes : Interchange two axes of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((3, 4, 5))
    >>> np.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> np.moveaxis(x, -1, 0).shape
    (5, 3, 4)

    These all achieve the same result:

    >>> np.transpose(x).shape
    (5, 4, 3)
    >>> np.swapaxes(x, 0, -1).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    (5, 4, 3)

    """
    try:
        # allow duck-array types if they define transpose
        transpose = a.transpose
    except AttributeError:
        a = asarray(a)
        transpose = a.transpose

    source = normalize_axis_tuple(source, a.ndim, 'source')
    destination = normalize_axis_tuple(destination, a.ndim, 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = transpose(order)
    return result


def _cross_dispatcher(a, b, axisa=None, axisb=None, axisc=None, axis=None):
    return (a, b)


@array_function_dispatch(_cross_dispatcher)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """
    Return the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector perpendicular
    to both `a` and `b`.  If `a` and `b` are arrays of vectors, the vectors
    are defined by the last axis of `a` and `b` by default, and these axes
    can have dimensions 2 or 3.  Where the dimension of either `a` or `b` is
    2, the third component of the input vector is assumed to be zero and the
    cross product calculated accordingly.  In cases where both input vectors
    have dimension 2, the z-component of the cross product is returned.

    Parameters
    ----------
    a : array_like
        Components of the first vector(s).
    b : array_like
        Components of the second vector(s).
    axisa : int, optional
        Axis of `a` that defines the vector(s).  By default, the last axis.
    axisb : int, optional
        Axis of `b` that defines the vector(s).  By default, the last axis.
    axisc : int, optional
        Axis of `c` containing the cross product vector(s).  Ignored if
        both input vectors have dimension 2, as the return is scalar.
        By default, the last axis.
    axis : int, optional
        If defined, the axis of `a`, `b` and `c` that defines the vector(s)
        and cross product(s).  Overrides `axisa`, `axisb` and `axisc`.

    Returns
    -------
    c : ndarray
        Vector cross product(s).

    Raises
    ------
    ValueError
        When the dimension of the vector(s) in `a` and/or `b` does not
        equal 2 or 3.

    See Also
    --------
    inner : Inner product
    outer : Outer product.
    linalg.cross : An Array API compatible variation of ``np.cross``,
                   which accepts (arrays of) 3-element vectors only.
    ix_ : Construct index arrays.

    Notes
    -----
    Supports full broadcasting of the inputs.

    Dimension-2 input arrays were deprecated in 2.0.0. If you do need this
    functionality, you can use::

        def cross2d(x, y):
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    Examples
    --------
    Vector cross-product.

    >>> import numpy as np
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([-3,  6, -3])

    One vector with dimension 2.

    >>> x = [1, 2]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([12, -6, -3])

    Equivalently:

    >>> x = [1, 2, 0]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([12, -6, -3])

    Both vectors with dimension 2.

    >>> x = [1,2]
    >>> y = [4,5]
    >>> np.cross(x, y)
    array(-3)

    Multiple vector cross-products. Note that the direction of the cross
    product vector is defined by the *right-hand rule*.

    >>> x = np.array([[1,2,3], [4,5,6]])
    >>> y = np.array([[4,5,6], [1,2,3]])
    >>> np.cross(x, y)
    array([[-3,  6, -3],
           [ 3, -6,  3]])

    The orientation of `c` can be changed using the `axisc` keyword.

    >>> np.cross(x, y, axisc=0)
    array([[-3,  3],
           [ 6, -6],
           [-3,  3]])

    Change the vector definition of `x` and `y` using `axisa` and `axisb`.

    >>> x = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
    >>> y = np.array([[7, 8, 9], [4,5,6], [1,2,3]])
    >>> np.cross(x, y)
    array([[ -6,  12,  -6],
           [  0,   0,   0],
           [  6, -12,   6]])
    >>> np.cross(x, y, axisa=0, axisb=0)
    array([[-24,  48, -24],
           [-30,  60, -30],
           [-36,  72, -36]])

    """
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3
    a = asarray(a)
    b = asarray(b)

    if (a.ndim < 1) or (b.ndim < 1):
        raise ValueError("At least one array has zero dimension")

    # Check axisa and axisb are within bounds
    axisa = normalize_axis_index(axisa, a.ndim, msg_prefix='axisa')
    axisb = normalize_axis_index(axisb, b.ndim, msg_prefix='axisb')

    # Move working axis to the end of the shape
    a = moveaxis(a, axisa, -1)
    b = moveaxis(b, axisb, -1)
    msg = ("incompatible dimensions for cross product\n"
           "(dimension must be 2 or 3)")
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)
    if a.shape[-1] == 2 or b.shape[-1] == 2:
        # Deprecated in NumPy 2.0, 2023-09-26
        warnings.warn(
            "Arrays of 2-dimensional vectors are deprecated. Use arrays of "
            "3-dimensional vectors instead. (deprecated in NumPy 2.0)",
            DeprecationWarning, stacklevel=2
        )

    # Create the output array
    shape = broadcast(a[..., 0], b[..., 0]).shape
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        # Check axisc is within bounds
        axisc = normalize_axis_index(axisc, len(shape), msg_prefix='axisc')
    dtype = promote_types(a.dtype, b.dtype)
    cp = empty(shape, dtype)

    # recast arrays as dtype
    a = a.astype(dtype)
    b = b.astype(dtype)

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            multiply(a0, b1, out=cp)
            cp -= a1 * b0
            return cp
        else:
            assert b.shape[-1] == 3
            # cp0 = a1 * b2 - 0  (a2 = 0)
            # cp1 = 0 - a0 * b2  (a2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            multiply(a1, b2, out=cp0)
            multiply(a0, b2, out=cp1)
            negative(cp1, out=cp1)
            multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    else:
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            # cp0 = a1 * b2 - a2 * b1
            # cp1 = a2 * b0 - a0 * b2
            # cp2 = a0 * b1 - a1 * b0
            multiply(a1, b2, out=cp0)
            tmp = np.multiply(a2, b1, out=...)
            cp0 -= tmp
            multiply(a2, b0, out=cp1)
            multiply(a0, b2, out=tmp)
            cp1 -= tmp
            multiply(a0, b1, out=cp2)
            multiply(a1, b0, out=tmp)
            cp2 -= tmp
        else:
            assert b.shape[-1] == 2
            # cp0 = 0 - a2 * b1  (b2 = 0)
            # cp1 = a2 * b0 - 0  (b2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            multiply(a2, b1, out=cp0)
            negative(cp0, out=cp0)
            multiply(a2, b0, out=cp1)
            multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0

    return moveaxis(cp, -1, axisc)


little_endian = (sys.byteorder == 'little')


@set_module('numpy')
def indices(dimensions, dtype=int, sparse=False):
    """
    Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0, 1, ...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : dtype, optional
        Data type of the result.
    sparse : boolean, optional
        Return a sparse representation of the grid instead of a dense
        representation. Default is False.

    Returns
    -------
    grid : one ndarray or tuple of ndarrays
        If sparse is False:
            Returns one array of grid indices,
            ``grid.shape = (len(dimensions),) + tuple(dimensions)``.
        If sparse is True:
            Returns a tuple of arrays, with
            ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` with
            dimensions[i] in the ith place

    See Also
    --------
    mgrid, ogrid, meshgrid

    Notes
    -----
    The output shape in the dense case is obtained by prepending the number
    of dimensions in front of the tuple of dimensions, i.e. if `dimensions`
    is a tuple ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N, r0, ..., rN-1)``.

    The subarrays ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly::

        grid[k, i0, i1, ..., iN-1] = ik

    Examples
    --------
    >>> import numpy as np
    >>> grid = np.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0]        # row indices
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]        # column indices
    array([[0, 1, 2],
           [0, 1, 2]])

    The indices can be used as an index into an array.

    >>> x = np.arange(20).reshape(5, 4)
    >>> row, col = np.indices((2, 3))
    >>> x[row, col]
    array([[0, 1, 2],
           [4, 5, 6]])

    Note that it would be more straightforward in the above example to
    extract the required elements directly with ``x[:2, :3]``.

    If sparse is set to true, the grid will be returned in a sparse
    representation.

    >>> i, j = np.indices((2, 3), sparse=True)
    >>> i.shape
    (2, 1)
    >>> j.shape
    (1, 3)
    >>> i        # row indices
    array([[0],
           [1]])
    >>> j        # column indices
    array([[0, 1, 2]])

    """
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = ()
    else:
        res = empty((N,) + dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = arange(dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i + 1:]
        )
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


@finalize_array_function_like
@set_module('numpy')
def fromfunction(function, shape, *, dtype=float, like=None, **kwargs):
    """
    Construct an array by executing a function over each coordinate.

    The resulting array therefore has a value ``fn(x, y, z)`` at
    coordinate ``(x, y, z)``.

    Parameters
    ----------
    function : callable
        The function is called with N parameters, where N is the rank of
        `shape`.  Each parameter represents the coordinates of the array
        varying along a specific axis.  For example, if `shape`
        were ``(2, 2)``, then the parameters would be
        ``array([[0, 0], [1, 1]])`` and ``array([[0, 1], [0, 1]])``
    shape : (N,) tuple of ints
        Shape of the output array, which also determines the shape of
        the coordinate arrays passed to `function`.
    dtype : data-type, optional
        Data-type of the coordinate arrays passed to `function`.
        By default, `dtype` is float.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    fromfunction : any
        The result of the call to `function` is passed back directly.
        Therefore the shape of `fromfunction` is completely determined by
        `function`.  If `function` returns a scalar value, the shape of
        `fromfunction` would not match the `shape` parameter.

    See Also
    --------
    indices, meshgrid

    Notes
    -----
    Keywords other than `dtype` and `like` are passed to `function`.

    Examples
    --------
    >>> import numpy as np
    >>> np.fromfunction(lambda i, j: i, (2, 2), dtype=float)
    array([[0., 0.],
           [1., 1.]])

    >>> np.fromfunction(lambda i, j: j, (2, 2), dtype=float)
    array([[0., 1.],
           [0., 1.]])

    >>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
    array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]])

    >>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

    """
    if like is not None:
        return _fromfunction_with_like(
                like, function, shape, dtype=dtype, **kwargs)

    args = indices(shape, dtype=dtype)
    return function(*args, **kwargs)


_fromfunction_with_like = array_function_dispatch()(fromfunction)


def _frombuffer(buf, dtype, shape, order, axis_order=None):
    array = frombuffer(buf, dtype=dtype)
    if order == 'K' and axis_order is not None:
        return array.reshape(shape, order='C').transpose(axis_order)
    return array.reshape(shape, order=order)


@set_module('numpy')
def isscalar(element):
    """
    Returns True if the type of `element` is a scalar type.

    Parameters
    ----------
    element : any
        Input argument, can be of any type and shape.

    Returns
    -------
    val : bool
        True if `element` is a scalar type, False if it is not.

    See Also
    --------
    ndim : Get the number of dimensions of an array

    Notes
    -----
    If you need a stricter way to identify a *numerical* scalar, use
    ``isinstance(x, numbers.Number)``, as that returns ``False`` for most
    non-numerical elements such as strings.

    In most cases ``np.ndim(x) == 0`` should be used instead of this function,
    as that will also return true for 0d arrays. This is how numpy overloads
    functions in the style of the ``dx`` arguments to `gradient` and
    the ``bins`` argument to `histogram`. Some key differences:

    +------------------------------------+---------------+-------------------+
    | x                                  |``isscalar(x)``|``np.ndim(x) == 0``|
    +====================================+===============+===================+
    | PEP 3141 numeric objects           | ``True``      | ``True``          |
    | (including builtins)               |               |                   |
    +------------------------------------+---------------+-------------------+
    | builtin string and buffer objects  | ``True``      | ``True``          |
    +------------------------------------+---------------+-------------------+
    | other builtin objects, like        | ``False``     | ``True``          |
    | `pathlib.Path`, `Exception`,       |               |                   |
    | the result of `re.compile`         |               |                   |
    +------------------------------------+---------------+-------------------+
    | third-party objects like           | ``False``     | ``True``          |
    | `matplotlib.figure.Figure`         |               |                   |
    +------------------------------------+---------------+-------------------+
    | zero-dimensional numpy arrays      | ``False``     | ``True``          |
    +------------------------------------+---------------+-------------------+
    | other numpy arrays                 | ``False``     | ``False``         |
    +------------------------------------+---------------+-------------------+
    | `list`, `tuple`, and other         | ``False``     | ``False``         |
    | sequence objects                   |               |                   |
    +------------------------------------+---------------+-------------------+

    Examples
    --------
    >>> import numpy as np

    >>> np.isscalar(3.1)
    True

    >>> np.isscalar(np.array(3.1))
    False

    >>> np.isscalar([3.1])
    False

    >>> np.isscalar(False)
    True

    >>> np.isscalar('numpy')
    True

    NumPy supports PEP 3141 numbers:

    >>> from fractions import Fraction
    >>> np.isscalar(Fraction(5, 17))
    True
    >>> from numbers import Number
    >>> np.isscalar(Number())
    True

    """
    return (isinstance(element, generic)
            or type(element) in ScalarType
            or isinstance(element, numbers.Number))


@set_module('numpy')
def binary_repr(num, width=None):
    """
    Return the binary representation of the input number as a string.

    For negative numbers, if width is not given, a minus sign is added to the
    front. If width is given, the two's complement of the number is
    returned, with respect to that width.

    In a two's-complement system negative numbers are represented by the two's
    complement of the absolute value. This is the most common method of
    representing signed integers on computers [1]_. A N-bit two's-complement
    system can represent every integer in the range
    :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    Parameters
    ----------
    num : int
        Only an integer decimal number can be used.
    width : int, optional
        The length of the returned string if `num` is positive, or the length
        of the two's complement if `num` is negative, provided that `width` is
        at least a sufficient number of bits for `num` to be represented in
        the designated form. If the `width` value is insufficient, an error is
        raised.

    Returns
    -------
    bin : str
        Binary representation of `num` or two's complement of `num`.

    See Also
    --------
    base_repr: Return a string representation of a number in the given base
               system.
    bin: Python's built-in binary representation generator of an integer.

    Notes
    -----
    `binary_repr` is equivalent to using `base_repr` with base 2, but about 25x
    faster.

    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        https://en.wikipedia.org/wiki/Two's_complement

    Examples
    --------
    >>> import numpy as np
    >>> np.binary_repr(3)
    '11'
    >>> np.binary_repr(-3)
    '-11'
    >>> np.binary_repr(3, width=4)
    '0011'

    The two's complement is returned when the input number is negative and
    width is specified:

    >>> np.binary_repr(-3, width=3)
    '101'
    >>> np.binary_repr(-3, width=5)
    '11101'

    """
    def err_if_insufficient(width, binwidth):
        if width is not None and width < binwidth:
            raise ValueError(
                f"Insufficient bit {width=} provided for {binwidth=}"
            )

    # Ensure that num is a Python integer to avoid overflow or unwanted
    # casts to floating point.
    num = operator.index(num)

    if num == 0:
        return '0' * (width or 1)

    elif num > 0:
        binary = f'{num:b}'
        binwidth = len(binary)
        outwidth = (binwidth if width is None
                    else builtins.max(binwidth, width))
        err_if_insufficient(width, binwidth)
        return binary.zfill(outwidth)

    elif width is None:
        return f'-{-num:b}'

    else:
        poswidth = len(f'{-num:b}')

        # See gh-8679: remove extra digit
        # for numbers at boundaries.
        if 2**(poswidth - 1) == -num:
            poswidth -= 1

        twocomp = 2**(poswidth + 1) + num
        binary = f'{twocomp:b}'
        binwidth = len(binary)

        outwidth = builtins.max(binwidth, width)
        err_if_insufficient(width, binwidth)
        return '1' * (outwidth - binwidth) + binary


@set_module('numpy')
def base_repr(number, base=2, padding=0):
    """
    Return a string representation of a number in the given base system.

    Parameters
    ----------
    number : int
        The value to convert. Positive and negative values are handled.
    base : int, optional
        Convert `number` to the `base` number system. The valid range is 2-36,
        the default value is 2.
    padding : int, optional
        Number of zeros padded on the left. Default is 0 (no padding).

    Returns
    -------
    out : str
        String representation of `number` in `base` system.

    See Also
    --------
    binary_repr : Faster version of `base_repr` for base 2.

    Examples
    --------
    >>> import numpy as np
    >>> np.base_repr(5)
    '101'
    >>> np.base_repr(6, 5)
    '11'
    >>> np.base_repr(7, base=5, padding=3)
    '00012'

    >>> np.base_repr(10, base=16)
    'A'
    >>> np.base_repr(32, base=16)
    '20'

    """
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if base > len(digits):
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    elif base < 2:
        raise ValueError("Bases less than 2 not handled in base_repr.")

    num = abs(int(number))
    res = []
    while num:
        res.append(digits[num % base])
        num //= base
    if padding:
        res.append('0' * padding)
    if number < 0:
        res.append('-')
    return ''.join(reversed(res or '0'))


# These are all essentially abbreviations
# These might wind up in a special abbreviations module


def _maketup(descr, val):
    dt = dtype(descr)
    # Place val in all scalar tuples:
    fields = dt.fields
    if fields is None:
        return val
    else:
        res = [_maketup(fields[name][0], val) for name in dt.names]
        return tuple(res)


@finalize_array_function_like
@set_module('numpy')
def identity(n, dtype=None, *, like=None):
    """
    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> import numpy as np
    >>> np.identity(3)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])

    """
    if like is not None:
        return _identity_with_like(like, n, dtype=dtype)

    from numpy import eye
    return eye(n, dtype=dtype, like=like)


_identity_with_like = array_function_dispatch()(identity)


def _allclose_dispatcher(a, b, rtol=None, atol=None, equal_nan=None):
    return (a, b, rtol, atol)


@array_function_dispatch(_allclose_dispatcher)
def allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    .. warning:: The default `atol` is not appropriate for comparing numbers
                 with magnitudes much smaller than one (see Notes).

    NaNs are treated as equal if they are in the same place and if
    ``equal_nan=True``.  Infs are treated as equal if they are in the same
    place and of the same sign in both arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : array_like
        The relative tolerance parameter (see Notes).
    atol : array_like
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    isclose, all, any, equal

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.::

     absolute(a - b) <= (atol + rtol * absolute(b))

    The above equation is not symmetric in `a` and `b`, so that
    ``allclose(a, b)`` might be different from ``allclose(b, a)`` in
    some rare cases.

    The default value of `atol` is not appropriate when the reference value
    `b` has magnitude smaller than one. For example, it is unlikely that
    ``a = 1e-9`` and ``b = 2e-9`` should be considered "close", yet
    ``allclose(1e-9, 2e-9)`` is ``True`` with default settings. Be sure
    to select `atol` for the use case at hand, especially for defining the
    threshold below which a non-zero value in `a` will be considered "close"
    to a very small or zero value in `b`.

    The comparison of `a` and `b` uses standard broadcasting, which
    means that `a` and `b` need not have the same shape in order for
    ``allclose(a, b)`` to evaluate to True.  The same is true for
    `equal` but not `array_equal`.

    `allclose` is not defined for non-numeric data types.
    `bool` is considered a numeric data-type for this purpose.

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose([1e10,1e-7], [1.00001e10,1e-8])
    False

    >>> np.allclose([1e10,1e-8], [1.00001e10,1e-9])
    True

    >>> np.allclose([1e10,1e-8], [1.0001e10,1e-9])
    False

    >>> np.allclose([1.0, np.nan], [1.0, np.nan])
    False

    >>> np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    True


    """
    res = all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
    return builtins.bool(res)


def _isclose_dispatcher(a, b, rtol=None, atol=None, equal_nan=None):
    return (a, b, rtol, atol)


@array_function_dispatch(_isclose_dispatcher)
def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    .. warning:: The default `atol` is not appropriate for comparing numbers
                 with magnitudes much smaller than one (see Notes).

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : array_like
        The relative tolerance parameter (see Notes).
    atol : array_like
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose
    math.isclose

    Notes
    -----
    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.::

     absolute(a - b) <= (atol + rtol * absolute(b))

    Unlike the built-in `math.isclose`, the above equation is not symmetric
    in `a` and `b` -- it assumes `b` is the reference value -- so that
    `isclose(a, b)` might be different from `isclose(b, a)`.

    The default value of `atol` is not appropriate when the reference value
    `b` has magnitude smaller than one. For example, it is unlikely that
    ``a = 1e-9`` and ``b = 2e-9`` should be considered "close", yet
    ``isclose(1e-9, 2e-9)`` is ``True`` with default settings. Be sure
    to select `atol` for the use case at hand, especially for defining the
    threshold below which a non-zero value in `a` will be considered "close"
    to a very small or zero value in `b`.

    `isclose` is not defined for non-numeric data types.
    :class:`bool` is considered a numeric data-type for this purpose.

    Examples
    --------
    >>> import numpy as np
    >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])
    array([ True, False])

    >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])
    array([ True, True])

    >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])
    array([False,  True])

    >>> np.isclose([1.0, np.nan], [1.0, np.nan])
    array([ True, False])

    >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    array([ True, True])

    >>> np.isclose([1e-8, 1e-7], [0.0, 0.0])
    array([ True, False])

    >>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
    array([False, False])

    >>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])
    array([ True,  True])

    >>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
    array([False,  True])

    """
    # Turn all but python scalars into arrays.
    x, y, atol, rtol = (
        a if isinstance(a, (int, float, complex)) else asanyarray(a)
        for a in (a, b, atol, rtol))

    # Make sure y is an inexact type to avoid bad behavior on abs(MIN_INT).
    # This will cause casting of x later. Also, make sure to allow subclasses
    # (e.g., for numpy.ma).
    # NOTE: We explicitly allow timedelta, which used to work. This could
    #       possibly be deprecated. See also gh-18286.
    #       timedelta works if `atol` is an integer or also a timedelta.
    #       Although, the default tolerances are unlikely to be useful
    if (dtype := getattr(y, "dtype", None)) is not None and dtype.kind != "m":
        dt = multiarray.result_type(y, 1.)
        y = asanyarray(y, dtype=dt)
    elif isinstance(y, int):
        y = float(y)

    # atol and rtol can be arrays
    if not (np.all(np.isfinite(atol)) and np.all(np.isfinite(rtol))):
        err_s = np.geterr()["invalid"]
        err_msg = f"One of rtol or atol is not valid, atol: {atol}, rtol: {rtol}"

        if err_s == "warn":
            warnings.warn(err_msg, RuntimeWarning, stacklevel=2)
        elif err_s == "raise":
            raise FloatingPointError(err_msg)
        elif err_s == "print":
            print(err_msg)

    with errstate(invalid='ignore'):

        result = (less_equal(abs(x - y), atol + rtol * abs(y))
                  & isfinite(y)
                  | (x == y))
        if equal_nan:
            result |= isnan(x) & isnan(y)

    return result[()]  # Flatten 0d arrays to scalars


def _array_equal_dispatcher(a1, a2, equal_nan=None):
    return (a1, a2)


_no_nan_types = {
    # should use np.dtype.BoolDType, but as of writing
    # that fails the reloading test.
    type(dtype(nt.bool)),
    type(dtype(nt.int8)),
    type(dtype(nt.int16)),
    type(dtype(nt.int32)),
    type(dtype(nt.int64)),
}


def _dtype_cannot_hold_nan(dtype):
    return type(dtype) in _no_nan_types


@array_function_dispatch(_array_equal_dispatcher)
def array_equal(a1, a2, equal_nan=False):
    """
    True if two arrays have the same shape and elements, False otherwise.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.
    equal_nan : bool
        Whether to compare NaN's as equal. If the dtype of a1 and a2 is
        complex, values will be considered equal if either the real or the
        imaginary component of a given value is ``nan``.

    Returns
    -------
    b : bool
        Returns True if the arrays are equal.

    See Also
    --------
    allclose: Returns True if two arrays are element-wise equal within a
              tolerance.
    array_equiv: Returns True if input arrays are shape consistent and all
                 elements equal.

    Examples
    --------
    >>> import numpy as np

    >>> np.array_equal([1, 2], [1, 2])
    True

    >>> np.array_equal(np.array([1, 2]), np.array([1, 2]))
    True

    >>> np.array_equal([1, 2], [1, 2, 3])
    False

    >>> np.array_equal([1, 2], [1, 4])
    False

    >>> a = np.array([1, np.nan])
    >>> np.array_equal(a, a)
    False

    >>> np.array_equal(a, a, equal_nan=True)
    True

    When ``equal_nan`` is True, complex values with nan components are
    considered equal if either the real *or* the imaginary components are nan.

    >>> a = np.array([1 + 1j])
    >>> b = a.copy()
    >>> a.real = np.nan
    >>> b.imag = np.nan
    >>> np.array_equal(a, b, equal_nan=True)
    True
    """
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        return False
    if a1.shape != a2.shape:
        return False
    if not equal_nan:
        return builtins.bool((asanyarray(a1 == a2)).all())

    if a1 is a2:
        # nan will compare equal so an array will compare equal to itself.
        return True

    cannot_have_nan = (_dtype_cannot_hold_nan(a1.dtype)
                       and _dtype_cannot_hold_nan(a2.dtype))
    if cannot_have_nan:
        return builtins.bool(asarray(a1 == a2).all())

    # Handling NaN values if equal_nan is True
    a1nan, a2nan = isnan(a1), isnan(a2)
    # NaN's occur at different locations
    if not (a1nan == a2nan).all():
        return False
    # Shapes of a1, a2 and masks are guaranteed to be consistent by this point
    return builtins.bool((a1[~a1nan] == a2[~a1nan]).all())


def _array_equiv_dispatcher(a1, a2):
    return (a1, a2)


@array_function_dispatch(_array_equiv_dispatcher)
def array_equiv(a1, a2):
    """
    Returns True if input arrays are shape consistent and all elements equal.

    Shape consistent means they are either the same shape, or one input array
    can be broadcasted to create the same shape as the other one.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    out : bool
        True if equivalent, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> np.array_equiv([1, 2], [1, 2])
    True
    >>> np.array_equiv([1, 2], [1, 3])
    False

    Showing the shape equivalence:

    >>> np.array_equiv([1, 2], [[1, 2], [1, 2]])
    True
    >>> np.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]])
    False

    >>> np.array_equiv([1, 2], [[1, 2], [1, 3]])
    False

    """
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        return False
    try:
        multiarray.broadcast(a1, a2)
    except Exception:
        return False

    return builtins.bool(asanyarray(a1 == a2).all())


def _astype_dispatcher(x, dtype, /, *, copy=None, device=None):
    return (x, dtype)


@array_function_dispatch(_astype_dispatcher)
def astype(x, dtype, /, *, copy=True, device=None):
    """
    Copies an array to a specified data type.

    This function is an Array API compatible alternative to
    `numpy.ndarray.astype`.

    Parameters
    ----------
    x : ndarray
        Input NumPy array to cast. ``array_likes`` are explicitly not
        supported here.
    dtype : dtype
        Data type of the result.
    copy : bool, optional
        Specifies whether to copy an array when the specified dtype matches
        the data type of the input array ``x``. If ``True``, a newly allocated
        array must always be returned. If ``False`` and the specified dtype
        matches the data type of the input array, the input array must be
        returned; otherwise, a newly allocated array must be returned.
        Defaults to ``True``.
    device : str, optional
        The device on which to place the returned array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.1.0

    Returns
    -------
    out : ndarray
        An array having the specified data type.

    See Also
    --------
    ndarray.astype

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3]); arr
    array([1, 2, 3])
    >>> np.astype(arr, np.float64)
    array([1., 2., 3.])

    Non-copy case:

    >>> arr = np.array([1, 2, 3])
    >>> arr_noncpy = np.astype(arr, arr.dtype, copy=False)
    >>> np.shares_memory(arr, arr_noncpy)
    True

    """
    if not (isinstance(x, np.ndarray) or isscalar(x)):
        raise TypeError(
            "Input should be a NumPy array or scalar. "
            f"It is a {type(x)} instead."
        )
    if device is not None and device != "cpu":
        raise ValueError(
            'Device not understood. Only "cpu" is allowed, but received:'
            f' {device}'
        )
    return x.astype(dtype, copy=copy)


inf = PINF
nan = NAN
False_ = nt.bool(False)
True_ = nt.bool(True)


def extend_all(module):
    existing = set(__all__)
    mall = module.__all__
    for a in mall:
        if a not in existing:
            __all__.append(a)


from . import _asarray, _ufunc_config, arrayprint, fromnumeric
from ._asarray import *
from ._ufunc_config import *
from .arrayprint import *
from .fromnumeric import *
from .numerictypes import *
from .umath import *

extend_all(fromnumeric)
extend_all(umath)
extend_all(numerictypes)
extend_all(arrayprint)
extend_all(_asarray)
extend_all(_ufunc_config)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\testing\_private\utils.py ===
"""
Utility function to facilitate testing.

"""
import concurrent.futures
import contextlib
import gc
import importlib.metadata
import operator
import os
import pathlib
import platform
import pprint
import re
import shutil
import sys
import sysconfig
import threading
import warnings
from functools import partial, wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from unittest.case import SkipTest
from warnings import WarningMessage

import numpy as np
import numpy.linalg._umath_linalg
from numpy import isfinite, isinf, isnan
from numpy._core import arange, array, array_repr, empty, float32, intp, isnat, ndarray

__all__ = [
        'assert_equal', 'assert_almost_equal', 'assert_approx_equal',
        'assert_array_equal', 'assert_array_less', 'assert_string_equal',
        'assert_array_almost_equal', 'assert_raises', 'build_err_msg',
        'decorate_methods', 'jiffies', 'memusage', 'print_assert_equal',
        'rundocs', 'runstring', 'verbose', 'measure',
        'assert_', 'assert_array_almost_equal_nulp', 'assert_raises_regex',
        'assert_array_max_ulp', 'assert_warns', 'assert_no_warnings',
        'assert_allclose', 'IgnoreException', 'clear_and_catch_warnings',
        'SkipTest', 'KnownFailureException', 'temppath', 'tempdir', 'IS_PYPY',
        'HAS_REFCOUNT', "IS_WASM", 'suppress_warnings', 'assert_array_compare',
        'assert_no_gc_cycles', 'break_cycles', 'HAS_LAPACK64', 'IS_PYSTON',
        'IS_MUSL', 'check_support_sve', 'NOGIL_BUILD',
        'IS_EDITABLE', 'IS_INSTALLED', 'NUMPY_ROOT', 'run_threaded', 'IS_64BIT',
        ]


class KnownFailureException(Exception):
    '''Raise this exception to mark a test as a known failing test.'''
    pass


KnownFailureTest = KnownFailureException  # backwards compat
verbose = 0

NUMPY_ROOT = pathlib.Path(np.__file__).parent

try:
    np_dist = importlib.metadata.distribution('numpy')
except importlib.metadata.PackageNotFoundError:
    IS_INSTALLED = IS_EDITABLE = False
else:
    IS_INSTALLED = True
    try:
        if sys.version_info >= (3, 13):
            IS_EDITABLE = np_dist.origin.dir_info.editable
        else:
            # Backport importlib.metadata.Distribution.origin
            import json  # noqa: E401
            import types
            origin = json.loads(
                np_dist.read_text('direct_url.json') or '{}',
                object_hook=lambda data: types.SimpleNamespace(**data),
            )
            IS_EDITABLE = origin.dir_info.editable
    except AttributeError:
        IS_EDITABLE = False

    # spin installs numpy directly via meson, instead of using meson-python, and
    # runs the module by setting PYTHONPATH. This is problematic because the
    # resulting installation lacks the Python metadata (.dist-info), and numpy
    # might already be installed on the environment, causing us to find its
    # metadata, even though we are not actually loading that package.
    # Work around this issue by checking if the numpy root matches.
    if not IS_EDITABLE and np_dist.locate_file('numpy') != NUMPY_ROOT:
        IS_INSTALLED = False

IS_WASM = platform.machine() in ["wasm32", "wasm64"]
IS_PYPY = sys.implementation.name == 'pypy'
IS_PYSTON = hasattr(sys, "pyston_version_info")
HAS_REFCOUNT = getattr(sys, 'getrefcount', None) is not None and not IS_PYSTON
HAS_LAPACK64 = numpy.linalg._umath_linalg._ilp64

IS_MUSL = False
# alternate way is
# from packaging.tags import sys_tags
#     _tags = list(sys_tags())
#     if 'musllinux' in _tags[0].platform:
_v = sysconfig.get_config_var('HOST_GNU_TYPE') or ''
if 'musl' in _v:
    IS_MUSL = True

NOGIL_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
IS_64BIT = np.dtype(np.intp).itemsize == 8

def assert_(val, msg=''):
    """
    Assert that works in release mode.
    Accepts callable msg to allow deferring evaluation until failure.

    The Python built-in ``assert`` does not work when executing code in
    optimized mode (the ``-O`` flag) - no byte-code is generated for it.

    For documentation on usage, refer to the Python documentation.

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if not val:
        try:
            smsg = msg()
        except TypeError:
            smsg = msg
        raise AssertionError(smsg)


if os.name == 'nt':
    # Code "stolen" from enthought/debug/memusage.py
    def GetPerformanceAttributes(object, counter, instance=None,
                                 inum=-1, format=None, machine=None):
        # NOTE: Many counters require 2 samples to give accurate results,
        # including "% Processor Time" (as by definition, at any instant, a
        # thread's CPU usage is either 0 or 100).  To read counters like this,
        # you should copy this function, but keep the counter open, and call
        # CollectQueryData() each time you need to know.
        # See http://msdn.microsoft.com/library/en-us/dnperfmo/html/perfmonpt2.asp
        # (dead link)
        # My older explanation for this was that the "AddCounter" process
        # forced the CPU to 100%, but the above makes more sense :)
        import win32pdh
        if format is None:
            format = win32pdh.PDH_FMT_LONG
        path = win32pdh.MakeCounterPath((machine, object, instance, None,
                                         inum, counter))
        hq = win32pdh.OpenQuery()
        try:
            hc = win32pdh.AddCounter(hq, path)
            try:
                win32pdh.CollectQueryData(hq)
                type, val = win32pdh.GetFormattedCounterValue(hc, format)
                return val
            finally:
                win32pdh.RemoveCounter(hc)
        finally:
            win32pdh.CloseQuery(hq)

    def memusage(processName="python", instance=0):
        # from win32pdhutil, part of the win32all package
        import win32pdh
        return GetPerformanceAttributes("Process", "Virtual Bytes",
                                        processName, instance,
                                        win32pdh.PDH_FMT_LONG, None)
elif sys.platform[:5] == 'linux':

    def memusage(_proc_pid_stat=None):
        """
        Return virtual memory size in bytes of the running python.

        """
        _proc_pid_stat = _proc_pid_stat or f'/proc/{os.getpid()}/stat'
        try:
            with open(_proc_pid_stat) as f:
                l = f.readline().split(' ')
            return int(l[22])
        except Exception:
            return
else:
    def memusage():
        """
        Return memory usage of running python. [Not implemented]

        """
        raise NotImplementedError


if sys.platform[:5] == 'linux':
    def jiffies(_proc_pid_stat=None, _load_time=None):
        """
        Return number of jiffies elapsed.

        Return number of jiffies (1/100ths of a second) that this
        process has been scheduled in user mode. See man 5 proc.

        """
        _proc_pid_stat = _proc_pid_stat or f'/proc/{os.getpid()}/stat'
        _load_time = _load_time or []
        import time
        if not _load_time:
            _load_time.append(time.time())
        try:
            with open(_proc_pid_stat) as f:
                l = f.readline().split(' ')
            return int(l[13])
        except Exception:
            return int(100 * (time.time() - _load_time[0]))
else:
    # os.getpid is not in all platforms available.
    # Using time is safe but inaccurate, especially when process
    # was suspended or sleeping.
    def jiffies(_load_time=[]):
        """
        Return number of jiffies elapsed.

        Return number of jiffies (1/100ths of a second) that this
        process has been scheduled in user mode. See man 5 proc.

        """
        import time
        if not _load_time:
            _load_time.append(time.time())
        return int(100 * (time.time() - _load_time[0]))


def build_err_msg(arrays, err_msg, header='Items are not equal:',
                  verbose=True, names=('ACTUAL', 'DESIRED'), precision=8):
    msg = ['\n' + header]
    err_msg = str(err_msg)
    if err_msg:
        if err_msg.find('\n') == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + ' ' + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):

            if isinstance(a, ndarray):
                # precision argument is only needed if the objects are ndarrays
                r_func = partial(array_repr, precision=precision)
            else:
                r_func = repr

            try:
                r = r_func(a)
            except Exception as exc:
                r = f'[repr failed for <{type(a).__name__}>: {exc}]'
            if r.count('\n') > 3:
                r = '\n'.join(r.splitlines()[:3])
                r += '...'
            msg.append(f' {names[i]}: {r}')
    return '\n'.join(msg)


def assert_equal(actual, desired, err_msg='', verbose=True, *, strict=False):
    """
    Raises an AssertionError if two objects are not equal.

    Given two objects (scalars, lists, tuples, dictionaries or numpy arrays),
    check that all elements of these objects are equal. An exception is raised
    at the first conflicting values.

    This function handles NaN comparisons as if NaN was a "normal" number.
    That is, AssertionError is not raised if both objects have NaNs in the same
    positions.  This is in contrast to the IEEE standard on NaNs, which says
    that NaN compared to anything must return False.

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    strict : bool, optional
        If True and either of the `actual` and `desired` arguments is an array,
        raise an ``AssertionError`` when either the shape or the data type of
        the arguments does not match. If neither argument is an array, this
        parameter has no effect.

        .. versionadded:: 2.0.0

    Raises
    ------
    AssertionError
        If actual and desired are not equal.

    See Also
    --------
    assert_allclose
    assert_array_almost_equal_nulp,
    assert_array_max_ulp,

    Notes
    -----
    By default, when one of `actual` and `desired` is a scalar and the other is
    an array, the function checks that each element of the array is equal to
    the scalar. This behaviour can be disabled by setting ``strict==True``.

    Examples
    --------
    >>> np.testing.assert_equal([4, 5], [4, 6])
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal:
    item=1
     ACTUAL: 5
     DESIRED: 6

    The following comparison does not raise an exception.  There are NaNs
    in the inputs, but they are in the same positions.

    >>> np.testing.assert_equal(np.array([1.0, 2.0, np.nan]), [1, 2, np.nan])

    As mentioned in the Notes section, `assert_equal` has special
    handling for scalars when one of the arguments is an array.
    Here, the test checks that each value in `x` is 3:

    >>> x = np.full((2, 5), fill_value=3)
    >>> np.testing.assert_equal(x, 3)

    Use `strict` to raise an AssertionError when comparing a scalar with an
    array of a different shape:

    >>> np.testing.assert_equal(x, 3, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (shapes (2, 5), () mismatch)
     ACTUAL: array([[3, 3, 3, 3, 3],
           [3, 3, 3, 3, 3]])
     DESIRED: array(3)

    The `strict` parameter also ensures that the array data types match:

    >>> x = np.array([2, 2, 2])
    >>> y = np.array([2., 2., 2.], dtype=np.float32)
    >>> np.testing.assert_equal(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (dtypes int64, float32 mismatch)
     ACTUAL: array([2, 2, 2])
     DESIRED: array([2., 2., 2.], dtype=float32)
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}',
                         verbose)
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}',
                         verbose)
        return
    from numpy import imag, iscomplexobj, real
    from numpy._core import isscalar, ndarray, signbit
    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_equal(actual, desired, err_msg, verbose,
                                  strict=strict)
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except (ValueError, TypeError):
        usecomplex = False

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_equal(actualr, desiredr)
            assert_equal(actuali, desiredi)
        except AssertionError:
            raise AssertionError(msg)

    # isscalar test to check cases such as [np.nan] != np.nan
    if isscalar(desired) != isscalar(actual):
        raise AssertionError(msg)

    try:
        isdesnat = isnat(desired)
        isactnat = isnat(actual)
        dtypes_match = (np.asarray(desired).dtype.type ==
                        np.asarray(actual).dtype.type)
        if isdesnat and isactnat:
            # If both are NaT (and have the same dtype -- datetime or
            # timedelta) they are considered equal.
            if dtypes_match:
                return
            else:
                raise AssertionError(msg)

    except (TypeError, ValueError, NotImplementedError):
        pass

    # Inf/nan/negative zero handling
    try:
        isdesnan = isnan(desired)
        isactnan = isnan(actual)
        if isdesnan and isactnan:
            return  # both nan, so equal

        # handle signed zero specially for floats
        array_actual = np.asarray(actual)
        array_desired = np.asarray(desired)
        if (array_actual.dtype.char in 'Mm' or
                array_desired.dtype.char in 'Mm'):
            # version 1.18
            # until this version, isnan failed for datetime64 and timedelta64.
            # Now it succeeds but comparison to scalar with a different type
            # emits a DeprecationWarning.
            # Avoid that by skipping the next check
            raise NotImplementedError('cannot compare to a scalar '
                                      'with a different type')

        if desired == 0 and actual == 0:
            if not signbit(desired) == signbit(actual):
                raise AssertionError(msg)

    except (TypeError, ValueError, NotImplementedError):
        pass

    try:
        # Explicitly use __eq__ for comparison, gh-2552
        if not (desired == actual):
            raise AssertionError(msg)

    except (DeprecationWarning, FutureWarning) as e:
        # this handles the case when the two types are not even comparable
        if 'elementwise == comparison' in e.args[0]:
            raise AssertionError(msg)
        else:
            raise


def print_assert_equal(test_string, actual, desired):
    """
    Test if two objects are equal, and print an error message if test fails.

    The test is performed with ``actual == desired``.

    Parameters
    ----------
    test_string : str
        The message supplied to AssertionError.
    actual : object
        The object to test for equality against `desired`.
    desired : object
        The expected result.

    Examples
    --------
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 1])
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 2])
    Traceback (most recent call last):
    ...
    AssertionError: Test XYZ of func xyz failed
    ACTUAL:
    [0, 1]
    DESIRED:
    [0, 2]

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import pprint

    if not (actual == desired):
        msg = StringIO()
        msg.write(test_string)
        msg.write(' failed\nACTUAL: \n')
        pprint.pprint(actual, msg)
        msg.write('DESIRED: \n')
        pprint.pprint(desired, msg)
        raise AssertionError(msg.getvalue())


def assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True):
    """
    Raises an AssertionError if two items are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies that the elements of `actual` and `desired` satisfy::

        abs(desired-actual) < float64(1.5 * 10**(-decimal))

    That is a looser test than originally documented, but agrees with what the
    actual implementation in `assert_array_almost_equal` did up to rounding
    vagaries. An exception is raised at conflicting values. For ndarrays this
    delegates to assert_array_almost_equal

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    decimal : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> from numpy.testing import assert_almost_equal
    >>> assert_almost_equal(2.3333333333333, 2.33333334)
    >>> assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 10 decimals
     ACTUAL: 2.3333333333333
     DESIRED: 2.33333334

    >>> assert_almost_equal(np.array([1.0,2.3333333333333]),
    ...                     np.array([1.0,2.33333334]), decimal=9)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 9 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference among violations: 6.66669964e-09
    Max relative difference among violations: 2.85715698e-09
     ACTUAL: array([1.         , 2.333333333])
     DESIRED: array([1.        , 2.33333334])

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from numpy import imag, iscomplexobj, real
    from numpy._core import ndarray

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except ValueError:
        usecomplex = False

    def _build_err_msg():
        header = ('Arrays are not almost equal to %d decimals' % decimal)
        return build_err_msg([actual, desired], err_msg, verbose=verbose,
                             header=header)

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_almost_equal(actualr, desiredr, decimal=decimal)
            assert_almost_equal(actuali, desiredi, decimal=decimal)
        except AssertionError:
            raise AssertionError(_build_err_msg())

    if isinstance(actual, (ndarray, tuple, list)) \
            or isinstance(desired, (ndarray, tuple, list)):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (isfinite(desired) and isfinite(actual)):
            if isnan(desired) or isnan(actual):
                if not (isnan(desired) and isnan(actual)):
                    raise AssertionError(_build_err_msg())
            elif not desired == actual:
                raise AssertionError(_build_err_msg())
            return
    except (NotImplementedError, TypeError):
        pass
    if abs(desired - actual) >= np.float64(1.5 * 10.0**(-decimal)):
        raise AssertionError(_build_err_msg())


def assert_approx_equal(actual, desired, significant=7, err_msg='',
                        verbose=True):
    """
    Raises an AssertionError if two items are not equal up to significant
    digits.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    Given two numbers, check that they are approximately equal.
    Approximately equal is defined as the number of significant digits
    that agree.

    Parameters
    ----------
    actual : scalar
        The object to check.
    desired : scalar
        The expected object.
    significant : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> np.testing.assert_approx_equal(0.12345677777777e-20, 0.1234567e-20)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345671e-20,
    ...                                significant=8)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345672e-20,
    ...                                significant=8)
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal to 8 significant digits:
     ACTUAL: 1.234567e-21
     DESIRED: 1.2345672e-21

    the evaluated condition that raises the exception is

    >>> abs(0.12345670e-20/1e-21 - 0.12345672e-20/1e-21) >= 10**-(8-1)
    True

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    (actual, desired) = map(float, (actual, desired))
    if desired == actual:
        return
    # Normalized the numbers to be in range (-10.0,10.0)
    # scale = float(pow(10,math.floor(math.log10(0.5*(abs(desired)+abs(actual))))))
    with np.errstate(invalid='ignore'):
        scale = 0.5 * (np.abs(desired) + np.abs(actual))
        scale = np.power(10, np.floor(np.log10(scale)))
    try:
        sc_desired = desired / scale
    except ZeroDivisionError:
        sc_desired = 0.0
    try:
        sc_actual = actual / scale
    except ZeroDivisionError:
        sc_actual = 0.0
    msg = build_err_msg(
        [actual, desired], err_msg,
        header='Items are not equal to %d significant digits:' % significant,
        verbose=verbose)
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (isfinite(desired) and isfinite(actual)):
            if isnan(desired) or isnan(actual):
                if not (isnan(desired) and isnan(actual)):
                    raise AssertionError(msg)
            elif not desired == actual:
                raise AssertionError(msg)
            return
    except (TypeError, NotImplementedError):
        pass
    if np.abs(sc_desired - sc_actual) >= np.power(10., -(significant - 1)):
        raise AssertionError(msg)


def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='',
                         precision=6, equal_nan=True, equal_inf=True,
                         *, strict=False, names=('ACTUAL', 'DESIRED')):
    __tracebackhide__ = True  # Hide traceback for py.test
    from numpy._core import all, array2string, errstate, inf, isnan, max, object_

    x = np.asanyarray(x)
    y = np.asanyarray(y)

    # original array for output formatting
    ox, oy = x, y

    def isnumber(x):
        return x.dtype.char in '?bhilqpBHILQPefdgFDG'

    def istime(x):
        return x.dtype.char in "Mm"

    def isvstring(x):
        return x.dtype.char == "T"

    def func_assert_same_pos(x, y, func=isnan, hasval='nan'):
        """Handling nan/inf.

        Combine results of running func on x and y, checking that they are True
        at the same locations.

        """
        __tracebackhide__ = True  # Hide traceback for py.test

        x_id = func(x)
        y_id = func(y)
        # We include work-arounds here to handle three types of slightly
        # pathological ndarray subclasses:
        # (1) all() on `masked` array scalars can return masked arrays, so we
        #     use != True
        # (2) __eq__ on some ndarray subclasses returns Python booleans
        #     instead of element-wise comparisons, so we cast to np.bool() and
        #     use isinstance(..., bool) checks
        # (3) subclasses with bare-bones __array_function__ implementations may
        #     not implement np.all(), so favor using the .all() method
        # We are not committed to supporting such subclasses, but it's nice to
        # support them if possible.
        if np.bool(x_id == y_id).all() != True:
            msg = build_err_msg(
                [x, y],
                err_msg + '\n%s location mismatch:'
                % (hasval), verbose=verbose, header=header,
                names=names,
                precision=precision)
            raise AssertionError(msg)
        # If there is a scalar, then here we know the array has the same
        # flag as it everywhere, so we should return the scalar flag.
        if isinstance(x_id, bool) or x_id.ndim == 0:
            return np.bool(x_id)
        elif isinstance(y_id, bool) or y_id.ndim == 0:
            return np.bool(y_id)
        else:
            return y_id

    try:
        if strict:
            cond = x.shape == y.shape and x.dtype == y.dtype
        else:
            cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
        if not cond:
            if x.shape != y.shape:
                reason = f'\n(shapes {x.shape}, {y.shape} mismatch)'
            else:
                reason = f'\n(dtypes {x.dtype}, {y.dtype} mismatch)'
            msg = build_err_msg([x, y],
                                err_msg
                                + reason,
                                verbose=verbose, header=header,
                                names=names,
                                precision=precision)
            raise AssertionError(msg)

        flagged = np.bool(False)
        if isnumber(x) and isnumber(y):
            if equal_nan:
                flagged = func_assert_same_pos(x, y, func=isnan, hasval='nan')

            if equal_inf:
                flagged |= func_assert_same_pos(x, y,
                                                func=lambda xy: xy == +inf,
                                                hasval='+inf')
                flagged |= func_assert_same_pos(x, y,
                                                func=lambda xy: xy == -inf,
                                                hasval='-inf')

        elif istime(x) and istime(y):
            # If one is datetime64 and the other timedelta64 there is no point
            if equal_nan and x.dtype.type == y.dtype.type:
                flagged = func_assert_same_pos(x, y, func=isnat, hasval="NaT")

        elif isvstring(x) and isvstring(y):
            dt = x.dtype
            if equal_nan and dt == y.dtype and hasattr(dt, 'na_object'):
                is_nan = (isinstance(dt.na_object, float) and
                          np.isnan(dt.na_object))
                bool_errors = 0
                try:
                    bool(dt.na_object)
                except TypeError:
                    bool_errors = 1
                if is_nan or bool_errors:
                    # nan-like NA object
                    flagged = func_assert_same_pos(
                        x, y, func=isnan, hasval=x.dtype.na_object)

        if flagged.ndim > 0:
            x, y = x[~flagged], y[~flagged]
            # Only do the comparison if actual values are left
            if x.size == 0:
                return
        elif flagged:
            # no sense doing comparison if everything is flagged.
            return

        val = comparison(x, y)
        invalids = np.logical_not(val)

        if isinstance(val, bool):
            cond = val
            reduced = array([val])
        else:
            reduced = val.ravel()
            cond = reduced.all()

        # The below comparison is a hack to ensure that fully masked
        # results, for which val.ravel().all() returns np.ma.masked,
        # do not trigger a failure (np.ma.masked != True evaluates as
        # np.ma.masked, which is falsy).
        if cond != True:
            n_mismatch = reduced.size - reduced.sum(dtype=intp)
            n_elements = flagged.size if flagged.ndim != 0 else reduced.size
            percent_mismatch = 100 * n_mismatch / n_elements
            remarks = [f'Mismatched elements: {n_mismatch} / {n_elements} '
                       f'({percent_mismatch:.3g}%)']

            with errstate(all='ignore'):
                # ignore errors for non-numeric types
                with contextlib.suppress(TypeError):
                    error = abs(x - y)
                    if np.issubdtype(x.dtype, np.unsignedinteger):
                        error2 = abs(y - x)
                        np.minimum(error, error2, out=error)

                    reduced_error = error[invalids]
                    max_abs_error = max(reduced_error)
                    if getattr(error, 'dtype', object_) == object_:
                        remarks.append(
                            'Max absolute difference among violations: '
                            + str(max_abs_error))
                    else:
                        remarks.append(
                            'Max absolute difference among violations: '
                            + array2string(max_abs_error))

                    # note: this definition of relative error matches that one
                    # used by assert_allclose (found in np.isclose)
                    # Filter values where the divisor would be zero
                    nonzero = np.bool(y != 0)
                    nonzero_and_invalid = np.logical_and(invalids, nonzero)

                    if all(~nonzero_and_invalid):
                        max_rel_error = array(inf)
                    else:
                        nonzero_invalid_error = error[nonzero_and_invalid]
                        broadcasted_y = np.broadcast_to(y, error.shape)
                        nonzero_invalid_y = broadcasted_y[nonzero_and_invalid]
                        max_rel_error = max(nonzero_invalid_error
                                            / abs(nonzero_invalid_y))

                    if getattr(error, 'dtype', object_) == object_:
                        remarks.append(
                            'Max relative difference among violations: '
                            + str(max_rel_error))
                    else:
                        remarks.append(
                            'Max relative difference among violations: '
                            + array2string(max_rel_error))
            err_msg = str(err_msg)
            err_msg += '\n' + '\n'.join(remarks)
            msg = build_err_msg([ox, oy], err_msg,
                                verbose=verbose, header=header,
                                names=names,
                                precision=precision)
            raise AssertionError(msg)
    except ValueError:
        import traceback
        efmt = traceback.format_exc()
        header = f'error during assertion:\n\n{efmt}\n\n{header}'

        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header,
                            names=names, precision=precision)
        raise ValueError(msg)


def assert_array_equal(actual, desired, err_msg='', verbose=True, *,
                       strict=False):
    """
    Raises an AssertionError if two array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are equal (but see the Notes for the special
    handling of a scalar). An exception is raised at shape mismatch or
    conflicting values. In contrast to the standard usage in numpy, NaNs
    are compared like numbers, no assertion is raised if both objects have
    NaNs in the same positions.

    The usual caution for verifying equality with floating point numbers is
    advised.

    .. note:: When either `actual` or `desired` is already an instance of
        `numpy.ndarray` and `desired` is not a ``dict``, the behavior of
        ``assert_equal(actual, desired)`` is identical to the behavior of this
        function. Otherwise, this function performs `np.asanyarray` on the
        inputs before comparison, whereas `assert_equal` defines special
        comparison rules for common Python types. For example, only
        `assert_equal` can be used to compare nested Python lists. In new code,
        consider using only `assert_equal`, explicitly converting either
        `actual` or `desired` to arrays if the behavior of `assert_array_equal`
        is desired.

    Parameters
    ----------
    actual : array_like
        The actual object to check.
    desired : array_like
        The desired, expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    strict : bool, optional
        If True, raise an AssertionError when either the shape or the data
        type of the array_like objects does not match. The special
        handling for scalars mentioned in the Notes section is disabled.

        .. versionadded:: 1.24.0

    Raises
    ------
    AssertionError
        If actual and desired objects are not equal.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Notes
    -----
    When one of `actual` and `desired` is a scalar and the other is array_like,
    the function checks that each element of the array_like object is equal to
    the scalar. This behaviour can be disabled with the `strict` parameter.

    Examples
    --------
    The first assert does not raise an exception:

    >>> np.testing.assert_array_equal([1.0,2.33333,np.nan],
    ...                               [np.exp(0),2.33333, np.nan])

    Assert fails with numerical imprecision with floats:

    >>> np.testing.assert_array_equal([1.0,np.pi,np.nan],
    ...                               [1, np.sqrt(np.pi)**2, np.nan])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference among violations: 4.4408921e-16
    Max relative difference among violations: 1.41357986e-16
     ACTUAL: array([1.      , 3.141593,      nan])
     DESIRED: array([1.      , 3.141593,      nan])

    Use `assert_allclose` or one of the nulp (number of floating point values)
    functions for these cases instead:

    >>> np.testing.assert_allclose([1.0,np.pi,np.nan],
    ...                            [1, np.sqrt(np.pi)**2, np.nan],
    ...                            rtol=1e-10, atol=0)

    As mentioned in the Notes section, `assert_array_equal` has special
    handling for scalars. Here the test checks that each value in `x` is 3:

    >>> x = np.full((2, 5), fill_value=3)
    >>> np.testing.assert_array_equal(x, 3)

    Use `strict` to raise an AssertionError when comparing a scalar with an
    array:

    >>> np.testing.assert_array_equal(x, 3, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (shapes (2, 5), () mismatch)
     ACTUAL: array([[3, 3, 3, 3, 3],
           [3, 3, 3, 3, 3]])
     DESIRED: array(3)

    The `strict` parameter also ensures that the array data types match:

    >>> x = np.array([2, 2, 2])
    >>> y = np.array([2., 2., 2.], dtype=np.float32)
    >>> np.testing.assert_array_equal(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (dtypes int64, float32 mismatch)
     ACTUAL: array([2, 2, 2])
     DESIRED: array([2., 2., 2.], dtype=float32)
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
                         verbose=verbose, header='Arrays are not equal',
                         strict=strict)


def assert_array_almost_equal(actual, desired, decimal=6, err_msg='',
                              verbose=True):
    """
    Raises an AssertionError if two objects are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies identical shapes and that the elements of ``actual`` and
    ``desired`` satisfy::

        abs(desired-actual) < 1.5 * 10**(-decimal)

    That is a looser test than originally documented, but agrees with what the
    actual implementation did up to rounding vagaries. An exception is raised
    at shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if both
    objects have NaNs in the same positions.

    Parameters
    ----------
    actual : array_like
        The actual object to check.
    desired : array_like
        The desired, expected object.
    decimal : int, optional
        Desired precision, default is 6.
    err_msg : str, optional
      The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    the first assert does not raise an exception

    >>> np.testing.assert_array_almost_equal([1.0,2.333,np.nan],
    ...                                      [1.0,2.333,np.nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33339,np.nan], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference among violations: 6.e-05
    Max relative difference among violations: 2.57136612e-05
     ACTUAL: array([1.     , 2.33333,     nan])
     DESIRED: array([1.     , 2.33339,     nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33333, 5], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    nan location mismatch:
     ACTUAL: array([1.     , 2.33333,     nan])
     DESIRED: array([1.     , 2.33333, 5.     ])

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from numpy._core import number, result_type
    from numpy._core.fromnumeric import any as npany
    from numpy._core.numerictypes import issubdtype

    def compare(x, y):
        try:
            if npany(isinf(x)) or npany(isinf(y)):
                xinfid = isinf(x)
                yinfid = isinf(y)
                if not (xinfid == yinfid).all():
                    return False
                # if one item, x and y is +- inf
                if x.size == y.size == 1:
                    return x == y
                x = x[~xinfid]
                y = y[~yinfid]
        except (TypeError, NotImplementedError):
            pass

        # make sure y is an inexact type to avoid abs(MIN_INT); will cause
        # casting of x later.
        dtype = result_type(y, 1.)
        y = np.asanyarray(y, dtype)
        z = abs(x - y)

        if not issubdtype(z.dtype, number):
            z = z.astype(np.float64)  # handle object arrays

        return z < 1.5 * 10.0**(-decimal)

    assert_array_compare(compare, actual, desired, err_msg=err_msg,
                         verbose=verbose,
             header=('Arrays are not almost equal to %d decimals' % decimal),
             precision=decimal)


def assert_array_less(x, y, err_msg='', verbose=True, *, strict=False):
    """
    Raises an AssertionError if two array_like objects are not ordered by less
    than.

    Given two array_like objects `x` and `y`, check that the shape is equal and
    all elements of `x` are strictly less than the corresponding elements of
    `y` (but see the Notes for the special handling of a scalar). An exception
    is raised at shape mismatch or values that are not correctly ordered. In
    contrast to the  standard usage in NumPy, no assertion is raised if both
    objects have NaNs in the same positions.

    Parameters
    ----------
    x : array_like
      The smaller object to check.
    y : array_like
      The larger object to compare.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.
    strict : bool, optional
        If True, raise an AssertionError when either the shape or the data
        type of the array_like objects does not match. The special
        handling for scalars mentioned in the Notes section is disabled.

        .. versionadded:: 2.0.0

    Raises
    ------
    AssertionError
      If x is not strictly smaller than y, element-wise.

    See Also
    --------
    assert_array_equal: tests objects for equality
    assert_array_almost_equal: test objects for equality up to precision

    Notes
    -----
    When one of `x` and `y` is a scalar and the other is array_like, the
    function performs the comparison as though the scalar were broadcasted
    to the shape of the array. This behaviour can be disabled with the `strict`
    parameter.

    Examples
    --------
    The following assertion passes because each finite element of `x` is
    strictly less than the corresponding element of `y`, and the NaNs are in
    corresponding locations.

    >>> x = [1.0, 1.0, np.nan]
    >>> y = [1.1, 2.0, np.nan]
    >>> np.testing.assert_array_less(x, y)

    The following assertion fails because the zeroth element of `x` is no
    longer strictly less than the zeroth element of `y`.

    >>> y[0] = 1
    >>> np.testing.assert_array_less(x, y)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not strictly ordered `x < y`
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference among violations: 0.
    Max relative difference among violations: 0.
     x: array([ 1.,  1., nan])
     y: array([ 1.,  2., nan])

    Here, `y` is a scalar, so each element of `x` is compared to `y`, and
    the assertion passes.

    >>> x = [1.0, 4.0]
    >>> y = 5.0
    >>> np.testing.assert_array_less(x, y)

    However, with ``strict=True``, the assertion will fail because the shapes
    do not match.

    >>> np.testing.assert_array_less(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not strictly ordered `x < y`
    <BLANKLINE>
    (shapes (2,), () mismatch)
     x: array([1., 4.])
     y: array(5.)

    With ``strict=True``, the assertion also fails if the dtypes of the two
    arrays do not match.

    >>> y = [5, 5]
    >>> np.testing.assert_array_less(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not strictly ordered `x < y`
    <BLANKLINE>
    (dtypes float64, int64 mismatch)
     x: array([1., 4.])
     y: array([5, 5])
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    assert_array_compare(operator.__lt__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not strictly ordered `x < y`',
                         equal_inf=False,
                         strict=strict,
                         names=('x', 'y'))


def runstring(astr, dict):
    exec(astr, dict)


def assert_string_equal(actual, desired):
    """
    Test if two strings are equal.

    If the given strings are equal, `assert_string_equal` does nothing.
    If they are not equal, an AssertionError is raised, and the diff
    between the strings is shown.

    Parameters
    ----------
    actual : str
        The string to test for equality against the expected string.
    desired : str
        The expected string.

    Examples
    --------
    >>> np.testing.assert_string_equal('abc', 'abc')
    >>> np.testing.assert_string_equal('abc', 'abcd')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ...
    AssertionError: Differences in strings:
    - abc+ abcd?    +

    """
    # delay import of difflib to reduce startup time
    __tracebackhide__ = True  # Hide traceback for py.test
    import difflib

    if not isinstance(actual, str):
        raise AssertionError(repr(type(actual)))
    if not isinstance(desired, str):
        raise AssertionError(repr(type(desired)))
    if desired == actual:
        return

    diff = list(difflib.Differ().compare(actual.splitlines(True),
                desired.splitlines(True)))
    diff_list = []
    while diff:
        d1 = diff.pop(0)
        if d1.startswith('  '):
            continue
        if d1.startswith('- '):
            l = [d1]
            d2 = diff.pop(0)
            if d2.startswith('? '):
                l.append(d2)
                d2 = diff.pop(0)
            if not d2.startswith('+ '):
                raise AssertionError(repr(d2))
            l.append(d2)
            if diff:
                d3 = diff.pop(0)
                if d3.startswith('? '):
                    l.append(d3)
                else:
                    diff.insert(0, d3)
            if d2[2:] == d1[2:]:
                continue
            diff_list.extend(l)
            continue
        raise AssertionError(repr(d1))
    if not diff_list:
        return
    msg = f"Differences in strings:\n{''.join(diff_list).rstrip()}"
    if actual != desired:
        raise AssertionError(msg)


def rundocs(filename=None, raise_on_error=True):
    """
    Run doctests found in the given file.

    By default `rundocs` raises an AssertionError on failure.

    Parameters
    ----------
    filename : str
        The path to the file for which the doctests are run.
    raise_on_error : bool
        Whether to raise an AssertionError when a doctest fails. Default is
        True.

    Notes
    -----
    The doctests can be run by the user/developer by adding the ``doctests``
    argument to the ``test()`` call. For example, to run all tests (including
    doctests) for ``numpy.lib``:

    >>> np.lib.test(doctests=True)  # doctest: +SKIP
    """
    import doctest

    from numpy.distutils.misc_util import exec_mod_from_location
    if filename is None:
        f = sys._getframe(1)
        filename = f.f_globals['__file__']
    name = os.path.splitext(os.path.basename(filename))[0]
    m = exec_mod_from_location(name, filename)

    tests = doctest.DocTestFinder().find(m)
    runner = doctest.DocTestRunner(verbose=False)

    msg = []
    if raise_on_error:
        out = msg.append
    else:
        out = None

    for test in tests:
        runner.run(test, out=out)

    if runner.failures > 0 and raise_on_error:
        raise AssertionError("Some doctests failed:\n%s" % "\n".join(msg))


def check_support_sve(__cache=[]):
    """
    gh-22982
    """

    if __cache:
        return __cache[0]

    import subprocess
    cmd = 'lscpu'
    try:
        output = subprocess.run(cmd, capture_output=True, text=True)
        result = 'sve' in output.stdout
    except (OSError, subprocess.SubprocessError):
        result = False
    __cache.append(result)
    return __cache[0]


#
# assert_raises and assert_raises_regex are taken from unittest.
#
import unittest


class _Dummy(unittest.TestCase):
    def nop(self):
        pass


_d = _Dummy('nop')


def assert_raises(*args, **kwargs):
    """
    assert_raises(exception_class, callable, *args, **kwargs)
    assert_raises(exception_class)

    Fail unless an exception of class exception_class is thrown
    by callable when invoked with arguments args and keyword
    arguments kwargs. If a different type of exception is
    thrown, it will not be caught, and the test case will be
    deemed to have suffered an error, exactly as for an
    unexpected exception.

    Alternatively, `assert_raises` can be used as a context manager:

    >>> from numpy.testing import assert_raises
    >>> with assert_raises(ZeroDivisionError):
    ...     1 / 0

    is equivalent to

    >>> def div(x, y):
    ...     return x / y
    >>> assert_raises(ZeroDivisionError, div, 1, 0)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    return _d.assertRaises(*args, **kwargs)


def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    """
    assert_raises_regex(exception_class, expected_regexp, callable, *args,
                        **kwargs)
    assert_raises_regex(exception_class, expected_regexp)

    Fail unless an exception of class exception_class and with message that
    matches expected_regexp is thrown by callable when invoked with arguments
    args and keyword arguments kwargs.

    Alternatively, can be used as a context manager like `assert_raises`.
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    return _d.assertRaisesRegex(exception_class, expected_regexp, *args, **kwargs)


def decorate_methods(cls, decorator, testmatch=None):
    """
    Apply a decorator to all methods in a class matching a regular expression.

    The given decorator is applied to all public methods of `cls` that are
    matched by the regular expression `testmatch`
    (``testmatch.search(methodname)``). Methods that are private, i.e. start
    with an underscore, are ignored.

    Parameters
    ----------
    cls : class
        Class whose methods to decorate.
    decorator : function
        Decorator to apply to methods
    testmatch : compiled regexp or str, optional
        The regular expression. Default value is None, in which case the
        nose default (``re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)``)
        is used.
        If `testmatch` is a string, it is compiled to a regular expression
        first.

    """
    if testmatch is None:
        testmatch = re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)
    else:
        testmatch = re.compile(testmatch)
    cls_attr = cls.__dict__

    # delayed import to reduce startup time
    from inspect import isfunction

    methods = [_m for _m in cls_attr.values() if isfunction(_m)]
    for function in methods:
        try:
            if hasattr(function, 'compat_func_name'):
                funcname = function.compat_func_name
            else:
                funcname = function.__name__
        except AttributeError:
            # not a function
            continue
        if testmatch.search(funcname) and not funcname.startswith('_'):
            setattr(cls, funcname, decorator(function))


def measure(code_str, times=1, label=None):
    """
    Return elapsed time for executing code in the namespace of the caller.

    The supplied code string is compiled with the Python builtin ``compile``.
    The precision of the timing is 10 milli-seconds. If the code will execute
    fast on this timescale, it can be executed many times to get reasonable
    timing accuracy.

    Parameters
    ----------
    code_str : str
        The code to be timed.
    times : int, optional
        The number of times the code is executed. Default is 1. The code is
        only compiled once.
    label : str, optional
        A label to identify `code_str` with. This is passed into ``compile``
        as the second argument (for run-time error messages).

    Returns
    -------
    elapsed : float
        Total elapsed time in seconds for executing `code_str` `times` times.

    Examples
    --------
    >>> times = 10
    >>> etime = np.testing.measure('for i in range(1000): np.sqrt(i**2)', times=times)
    >>> print("Time for a single execution : ", etime / times, "s")  # doctest: +SKIP
    Time for a single execution :  0.005 s

    """
    frame = sys._getframe(1)
    locs, globs = frame.f_locals, frame.f_globals

    code = compile(code_str, f'Test name: {label} ', 'exec')
    i = 0
    elapsed = jiffies()
    while i < times:
        i += 1
        exec(code, globs, locs)
    elapsed = jiffies() - elapsed
    return 0.01 * elapsed


def _assert_valid_refcount(op):
    """
    Check that ufuncs don't mishandle refcount of object `1`.
    Used in a few regression tests.
    """
    if not HAS_REFCOUNT:
        return True

    import gc

    import numpy as np

    b = np.arange(100 * 100).reshape(100, 100)
    c = b
    i = 1

    gc.disable()
    try:
        rc = sys.getrefcount(i)
        for j in range(15):
            d = op(b, c)
        assert_(sys.getrefcount(i) >= rc)
    finally:
        gc.enable()


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                    err_msg='', verbose=True, *, strict=False):
    """
    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    Given two array_like objects, check that their shapes and all elements
    are equal (but see the Notes for the special handling of a scalar). An
    exception is raised if the shapes mismatch or any values conflict. In
    contrast to the standard usage in numpy, NaNs are compared like numbers,
    no assertion is raised if both objects have NaNs in the same positions.

    The test is equivalent to ``allclose(actual, desired, rtol, atol)`` (note
    that ``allclose`` has different default values). It compares the difference
    between `actual` and `desired` to ``atol + rtol * abs(desired)``.

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    equal_nan : bool, optional.
        If True, NaNs will compare equal.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    strict : bool, optional
        If True, raise an ``AssertionError`` when either the shape or the data
        type of the arguments does not match. The special handling of scalars
        mentioned in the Notes section is disabled.

        .. versionadded:: 2.0.0

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_array_almost_equal_nulp, assert_array_max_ulp

    Notes
    -----
    When one of `actual` and `desired` is a scalar and the other is
    array_like, the function performs the comparison as if the scalar were
    broadcasted to the shape of the array.
    This behaviour can be disabled with the `strict` parameter.

    Examples
    --------
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)

    As mentioned in the Notes section, `assert_allclose` has special
    handling for scalars. Here, the test checks that the value of `numpy.sin`
    is nearly zero at integer multiples of π.

    >>> x = np.arange(3) * np.pi
    >>> np.testing.assert_allclose(np.sin(x), 0, atol=1e-15)

    Use `strict` to raise an ``AssertionError`` when comparing an array
    with one or more dimensions against a scalar.

    >>> np.testing.assert_allclose(np.sin(x), 0, atol=1e-15, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Not equal to tolerance rtol=1e-07, atol=1e-15
    <BLANKLINE>
    (shapes (3,), () mismatch)
     ACTUAL: array([ 0.000000e+00,  1.224647e-16, -2.449294e-16])
     DESIRED: array(0)

    The `strict` parameter also ensures that the array data types match:

    >>> y = np.zeros(3, dtype=np.float32)
    >>> np.testing.assert_allclose(np.sin(x), y, atol=1e-15, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Not equal to tolerance rtol=1e-07, atol=1e-15
    <BLANKLINE>
    (dtypes float64, float32 mismatch)
     ACTUAL: array([ 0.000000e+00,  1.224647e-16, -2.449294e-16])
     DESIRED: array([0., 0., 0.], dtype=float32)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    def compare(x, y):
        return np._core.numeric.isclose(x, y, rtol=rtol, atol=atol,
                                       equal_nan=equal_nan)

    actual, desired = np.asanyarray(actual), np.asanyarray(desired)
    header = f'Not equal to tolerance rtol={rtol:g}, atol={atol:g}'
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
                         verbose=verbose, header=header, equal_nan=equal_nan,
                         strict=strict)


def assert_array_almost_equal_nulp(x, y, nulp=1):
    """
    Compare two arrays relatively to their spacing.

    This is a relatively robust method to compare two arrays whose amplitude
    is variable.

    Parameters
    ----------
    x, y : array_like
        Input arrays.
    nulp : int, optional
        The maximum number of unit in the last place for tolerance (see Notes).
        Default is 1.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the spacing between `x` and `y` for one or more elements is larger
        than `nulp`.

    See Also
    --------
    assert_array_max_ulp : Check that all items of arrays differ in at most
        N Units in the Last Place.
    spacing : Return the distance between x and the nearest adjacent number.

    Notes
    -----
    An assertion is raised if the following condition is not met::

        abs(x - y) <= nulp * spacing(maximum(abs(x), abs(y)))

    Examples
    --------
    >>> x = np.array([1., 1e-10, 1e-20])
    >>> eps = np.finfo(x.dtype).eps
    >>> np.testing.assert_array_almost_equal_nulp(x, x*eps/2 + x)

    >>> np.testing.assert_array_almost_equal_nulp(x, x*eps + x)
    Traceback (most recent call last):
      ...
    AssertionError: Arrays are not equal to 1 ULP (max is 2)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np
    ax = np.abs(x)
    ay = np.abs(y)
    ref = nulp * np.spacing(np.where(ax > ay, ax, ay))
    if not np.all(np.abs(x - y) <= ref):
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            msg = f"Arrays are not equal to {nulp} ULP"
        else:
            max_nulp = np.max(nulp_diff(x, y))
            msg = f"Arrays are not equal to {nulp} ULP (max is {max_nulp:g})"
        raise AssertionError(msg)


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """
    Check that all items of arrays differ in at most N Units in the Last Place.

    Parameters
    ----------
    a, b : array_like
        Input arrays to be compared.
    maxulp : int, optional
        The maximum number of units in the last place that elements of `a` and
        `b` can differ. Default is 1.
    dtype : dtype, optional
        Data-type to convert `a` and `b` to if given. Default is None.

    Returns
    -------
    ret : ndarray
        Array containing number of representable floating point numbers between
        items in `a` and `b`.

    Raises
    ------
    AssertionError
        If one or more elements differ by more than `maxulp`.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    See Also
    --------
    assert_array_almost_equal_nulp : Compare two arrays relatively to their
        spacing.

    Examples
    --------
    >>> a = np.linspace(0., 1., 100)
    >>> res = np.testing.assert_array_max_ulp(a, np.arcsin(np.sin(a)))

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np
    ret = nulp_diff(a, b, dtype)
    if not np.all(ret <= maxulp):
        raise AssertionError("Arrays are not almost equal up to %g "
                             "ULP (max difference is %g ULP)" %
                             (maxulp, np.max(ret)))
    return ret


def nulp_diff(x, y, dtype=None):
    """For each item in x and y, return the number of representable floating
    points between them.

    Parameters
    ----------
    x : array_like
        first input array
    y : array_like
        second input array
    dtype : dtype, optional
        Data-type to convert `x` and `y` to if given. Default is None.

    Returns
    -------
    nulp : array_like
        number of representable floating point numbers between each item in x
        and y.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    Examples
    --------
    # By definition, epsilon is the smallest number such as 1 + eps != 1, so
    # there should be exactly one ULP between 1 and 1 + eps
    >>> nulp_diff(1, 1 + np.finfo(x.dtype).eps)
    1.0
    """
    import numpy as np
    if dtype:
        x = np.asarray(x, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
    else:
        x = np.asarray(x)
        y = np.asarray(y)

    t = np.common_type(x, y)
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise NotImplementedError("_nulp not implemented for complex array")

    x = np.array([x], dtype=t)
    y = np.array([y], dtype=t)

    x[np.isnan(x)] = np.nan
    y[np.isnan(y)] = np.nan

    if not x.shape == y.shape:
        raise ValueError(f"Arrays do not have the same shape: {x.shape} - {y.shape}")

    def _diff(rx, ry, vdt):
        diff = np.asarray(rx - ry, dtype=vdt)
        return np.abs(diff)

    rx = integer_repr(x)
    ry = integer_repr(y)
    return _diff(rx, ry, t)


def _integer_repr(x, vdt, comp):
    # Reinterpret binary representation of the float as sign-magnitude:
    # take into account two-complement representation
    # See also
    # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    rx = x.view(vdt)
    if not (rx.size == 1):
        rx[rx < 0] = comp - rx[rx < 0]
    elif rx < 0:
        rx = comp - rx

    return rx


def integer_repr(x):
    """Return the signed-magnitude interpretation of the binary representation
    of x."""
    import numpy as np
    if x.dtype == np.float16:
        return _integer_repr(x, np.int16, np.int16(-2**15))
    elif x.dtype == np.float32:
        return _integer_repr(x, np.int32, np.int32(-2**31))
    elif x.dtype == np.float64:
        return _integer_repr(x, np.int64, np.int64(-2**63))
    else:
        raise ValueError(f'Unsupported dtype {x.dtype}')


@contextlib.contextmanager
def _assert_warns_context(warning_class, name=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    with suppress_warnings() as sup:
        l = sup.record(warning_class)
        yield
        if not len(l) > 0:
            name_str = f' when calling {name}' if name is not None else ''
            raise AssertionError("No warning raised" + name_str)


def assert_warns(warning_class, *args, **kwargs):
    """
    Fail unless the given callable throws the specified warning.

    A warning of class warning_class should be thrown by the callable when
    invoked with arguments args and keyword arguments kwargs.
    If a different type of warning is thrown, it will not be caught.

    If called with all arguments other than the warning class omitted, may be
    used as a context manager::

        with assert_warns(SomeWarning):
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    Parameters
    ----------
    warning_class : class
        The class defining the warning that `func` is expected to throw.
    func : callable, optional
        Callable to test
    *args : Arguments
        Arguments for `func`.
    **kwargs : Kwargs
        Keyword arguments for `func`.

    Returns
    -------
    The value returned by `func`.

    Examples
    --------
    >>> import warnings
    >>> def deprecated_func(num):
    ...     warnings.warn("Please upgrade", DeprecationWarning)
    ...     return num*num
    >>> with np.testing.assert_warns(DeprecationWarning):
    ...     assert deprecated_func(4) == 16
    >>> # or passing a func
    >>> ret = np.testing.assert_warns(DeprecationWarning, deprecated_func, 4)
    >>> assert ret == 16
    """
    if not args and not kwargs:
        return _assert_warns_context(warning_class)
    elif len(args) < 1:
        if "match" in kwargs:
            raise RuntimeError(
                "assert_warns does not use 'match' kwarg, "
                "use pytest.warns instead"
                )
        raise RuntimeError("assert_warns(...) needs at least one arg")

    func = args[0]
    args = args[1:]
    with _assert_warns_context(warning_class, name=func.__name__):
        return func(*args, **kwargs)


@contextlib.contextmanager
def _assert_no_warnings_context(name=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    with warnings.catch_warnings(record=True) as l:
        warnings.simplefilter('always')
        yield
        if len(l) > 0:
            name_str = f' when calling {name}' if name is not None else ''
            raise AssertionError(f'Got warnings{name_str}: {l}')


def assert_no_warnings(*args, **kwargs):
    """
    Fail if the given callable produces any warnings.

    If called with all arguments omitted, may be used as a context manager::

        with assert_no_warnings():
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    Parameters
    ----------
    func : callable
        The callable to test.
    \\*args : Arguments
        Arguments passed to `func`.
    \\*\\*kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    The value returned by `func`.

    """
    if not args:
        return _assert_no_warnings_context()

    func = args[0]
    args = args[1:]
    with _assert_no_warnings_context(name=func.__name__):
        return func(*args, **kwargs)


def _gen_alignment_data(dtype=float32, type='binary', max_size=24):
    """
    generator producing data with different alignment and offsets
    to test simd vectorization

    Parameters
    ----------
    dtype : dtype
        data type to produce
    type : string
        'unary': create data for unary operations, creates one input
                 and output array
        'binary': create data for unary operations, creates two input
                 and output array
    max_size : integer
        maximum size of data to produce

    Returns
    -------
    if type is 'unary' yields one output, one input array and a message
    containing information on the data
    if type is 'binary' yields one output array, two input array and a message
    containing information on the data

    """
    ufmt = 'unary offset=(%d, %d), size=%d, dtype=%r, %s'
    bfmt = 'binary offset=(%d, %d, %d), size=%d, dtype=%r, %s'
    for o in range(3):
        for s in range(o + 2, max(o + 3, max_size)):
            if type == 'unary':
                inp = lambda: arange(s, dtype=dtype)[o:]
                out = empty((s,), dtype=dtype)[o:]
                yield out, inp(), ufmt % (o, o, s, dtype, 'out of place')
                d = inp()
                yield d, d, ufmt % (o, o, s, dtype, 'in place')
                yield out[1:], inp()[:-1], ufmt % \
                    (o + 1, o, s - 1, dtype, 'out of place')
                yield out[:-1], inp()[1:], ufmt % \
                    (o, o + 1, s - 1, dtype, 'out of place')
                yield inp()[:-1], inp()[1:], ufmt % \
                    (o, o + 1, s - 1, dtype, 'aliased')
                yield inp()[1:], inp()[:-1], ufmt % \
                    (o + 1, o, s - 1, dtype, 'aliased')
            if type == 'binary':
                inp1 = lambda: arange(s, dtype=dtype)[o:]
                inp2 = lambda: arange(s, dtype=dtype)[o:]
                out = empty((s,), dtype=dtype)[o:]
                yield out, inp1(), inp2(), bfmt % \
                    (o, o, o, s, dtype, 'out of place')
                d = inp1()
                yield d, d, inp2(), bfmt % \
                    (o, o, o, s, dtype, 'in place1')
                d = inp2()
                yield d, inp1(), d, bfmt % \
                    (o, o, o, s, dtype, 'in place2')
                yield out[1:], inp1()[:-1], inp2()[:-1], bfmt % \
                    (o + 1, o, o, s - 1, dtype, 'out of place')
                yield out[:-1], inp1()[1:], inp2()[:-1], bfmt % \
                    (o, o + 1, o, s - 1, dtype, 'out of place')
                yield out[:-1], inp1()[:-1], inp2()[1:], bfmt % \
                    (o, o, o + 1, s - 1, dtype, 'out of place')
                yield inp1()[1:], inp1()[:-1], inp2()[:-1], bfmt % \
                    (o + 1, o, o, s - 1, dtype, 'aliased')
                yield inp1()[:-1], inp1()[1:], inp2()[:-1], bfmt % \
                    (o, o + 1, o, s - 1, dtype, 'aliased')
                yield inp1()[:-1], inp1()[:-1], inp2()[1:], bfmt % \
                    (o, o, o + 1, s - 1, dtype, 'aliased')


class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"
    pass


@contextlib.contextmanager
def tempdir(*args, **kwargs):
    """Context manager to provide a temporary test folder.

    All arguments are passed as this to the underlying tempfile.mkdtemp
    function.

    """
    tmpdir = mkdtemp(*args, **kwargs)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


@contextlib.contextmanager
def temppath(*args, **kwargs):
    """Context manager for temporary files.

    Context manager that returns the path to a closed temporary file. Its
    parameters are the same as for tempfile.mkstemp and are passed directly
    to that function. The underlying file is removed when the context is
    exited, so it should be closed at that time.

    Windows does not allow a temporary file to be opened if it is already
    open, so the underlying file must be closed after opening before it
    can be opened again.

    """
    fd, path = mkstemp(*args, **kwargs)
    os.close(fd)
    try:
        yield path
    finally:
        os.remove(path)


class clear_and_catch_warnings(warnings.catch_warnings):
    """ Context manager that resets warning registry for catching warnings

    Warnings can be slippery, because, whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module.  This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters.  This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.

    For compatibility with Python, please consider all arguments to be
    keyword-only.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. The objects appended to the list are arguments whose
        attributes mirror the arguments to ``showwarning()``.
    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit. To work correctly, all 'ignore' filters should
        filter by one of these modules.

    Examples
    --------
    >>> import warnings
    >>> with np.testing.clear_and_catch_warnings(
    ...         modules=[np._core.fromnumeric]):
    ...     warnings.simplefilter('always')
    ...     warnings.filterwarnings('ignore', module='np._core.fromnumeric')
    ...     # do something that raises a warning but ignore those in
    ...     # np._core.fromnumeric
    """
    class_modules = ()

    def __init__(self, record=False, modules=()):
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super().__init__(record=record)

    def __enter__(self):
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super().__enter__()

    def __exit__(self, *exc_info):
        super().__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])


class suppress_warnings:
    """
    Context manager and decorator doing much the same as
    ``warnings.catch_warnings``.

    However, it also provides a filter mechanism to work around
    https://bugs.python.org/issue4180.

    This bug causes Python before 3.4 to not reliably show warnings again
    after they have been ignored once (even within catch_warnings). It
    means that no "ignore" filter can be used easily, since following
    tests might need to see the warning. Additionally it allows easier
    specificity for testing warnings and can be nested.

    Parameters
    ----------
    forwarding_rule : str, optional
        One of "always", "once", "module", or "location". Analogous to
        the usual warnings module filter mode, it is useful to reduce
        noise mostly on the outmost level. Unsuppressed and unrecorded
        warnings will be forwarded based on this rule. Defaults to "always".
        "location" is equivalent to the warnings "default", match by exact
        location the warning warning originated from.

    Notes
    -----
    Filters added inside the context manager will be discarded again
    when leaving it. Upon entering all filters defined outside a
    context will be applied automatically.

    When a recording filter is added, matching warnings are stored in the
    ``log`` attribute as well as in the list returned by ``record``.

    If filters are added and the ``module`` keyword is given, the
    warning registry of this module will additionally be cleared when
    applying it, entering the context, or exiting it. This could cause
    warnings to appear a second time after leaving the context if they
    were configured to be printed once (default) and were already
    printed before the context was entered.

    Nesting this context manager will work as expected when the
    forwarding rule is "always" (default). Unfiltered and unrecorded
    warnings will be passed out and be matched by the outer level.
    On the outmost level they will be printed (or caught by another
    warnings context). The forwarding rule argument can modify this
    behaviour.

    Like ``catch_warnings`` this context manager is not threadsafe.

    Examples
    --------

    With a context manager::

        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Some text")
            sup.filter(module=np.ma.core)
            log = sup.record(FutureWarning, "Does this occur?")
            command_giving_warnings()
            # The FutureWarning was given once, the filtered warnings were
            # ignored. All other warnings abide outside settings (may be
            # printed/error)
            assert_(len(log) == 1)
            assert_(len(sup.log) == 1)  # also stored in log attribute

    Or as a decorator::

        sup = np.testing.suppress_warnings()
        sup.filter(module=np.ma.core)  # module must match exactly
        @sup
        def some_function():
            # do something which causes a warning in np.ma.core
            pass
    """
    def __init__(self, forwarding_rule="always"):
        self._entered = False

        # Suppressions are either instance or defined inside one with block:
        self._suppressions = []

        if forwarding_rule not in {"always", "module", "once", "location"}:
            raise ValueError("unsupported forwarding rule.")
        self._forwarding_rule = forwarding_rule

    def _clear_registries(self):
        if hasattr(warnings, "_filters_mutated"):
            # clearing the registry should not be necessary on new pythons,
            # instead the filters should be mutated.
            warnings._filters_mutated()
            return
        # Simply clear the registry, this should normally be harmless,
        # note that on new pythons it would be invalidated anyway.
        for module in self._tmp_modules:
            if hasattr(module, "__warningregistry__"):
                module.__warningregistry__.clear()

    def _filter(self, category=Warning, message="", module=None, record=False):
        if record:
            record = []  # The log where to store warnings
        else:
            record = None
        if self._entered:
            if module is None:
                warnings.filterwarnings(
                    "always", category=category, message=message)
            else:
                module_regex = module.__name__.replace('.', r'\.') + '$'
                warnings.filterwarnings(
                    "always", category=category, message=message,
                    module=module_regex)
                self._tmp_modules.add(module)
                self._clear_registries()

            self._tmp_suppressions.append(
                (category, message, re.compile(message, re.I), module, record))
        else:
            self._suppressions.append(
                (category, message, re.compile(message, re.I), module, record))

        return record

    def filter(self, category=Warning, message="", module=None):
        """
        Add a new suppressing filter or apply it if the state is entered.

        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.

        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        self._filter(category=category, message=message, module=module,
                     record=False)

    def record(self, category=Warning, message="", module=None):
        """
        Append a new recording filter or apply it if the state is entered.

        All warnings matching will be appended to the ``log`` attribute.

        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.

        Returns
        -------
        log : list
            A list which will be filled with all matched warnings.

        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        return self._filter(category=category, message=message, module=module,
                            record=True)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("cannot enter suppress_warnings twice.")

        self._orig_show = warnings.showwarning
        self._filters = warnings.filters
        warnings.filters = self._filters[:]

        self._entered = True
        self._tmp_suppressions = []
        self._tmp_modules = set()
        self._forwarded = set()

        self.log = []  # reset global log (no need to keep same list)

        for cat, mess, _, mod, log in self._suppressions:
            if log is not None:
                del log[:]  # clear the log
            if mod is None:
                warnings.filterwarnings(
                    "always", category=cat, message=mess)
            else:
                module_regex = mod.__name__.replace('.', r'\.') + '$'
                warnings.filterwarnings(
                    "always", category=cat, message=mess,
                    module=module_regex)
                self._tmp_modules.add(mod)
        warnings.showwarning = self._showwarning
        self._clear_registries()

        return self

    def __exit__(self, *exc_info):
        warnings.showwarning = self._orig_show
        warnings.filters = self._filters
        self._clear_registries()
        self._entered = False
        del self._orig_show
        del self._filters

    def _showwarning(self, message, category, filename, lineno,
                     *args, use_warnmsg=None, **kwargs):
        for cat, _, pattern, mod, rec in (
                self._suppressions + self._tmp_suppressions)[::-1]:
            if (issubclass(category, cat) and
                    pattern.match(message.args[0]) is not None):
                if mod is None:
                    # Message and category match, either recorded or ignored
                    if rec is not None:
                        msg = WarningMessage(message, category, filename,
                                             lineno, **kwargs)
                        self.log.append(msg)
                        rec.append(msg)
                    return
                # Use startswith, because warnings strips the c or o from
                # .pyc/.pyo files.
                elif mod.__file__.startswith(filename):
                    # The message and module (filename) match
                    if rec is not None:
                        msg = WarningMessage(message, category, filename,
                                             lineno, **kwargs)
                        self.log.append(msg)
                        rec.append(msg)
                    return

        # There is no filter in place, so pass to the outside handler
        # unless we should only pass it once
        if self._forwarding_rule == "always":
            if use_warnmsg is None:
                self._orig_show(message, category, filename, lineno,
                                *args, **kwargs)
            else:
                self._orig_showmsg(use_warnmsg)
            return

        if self._forwarding_rule == "once":
            signature = (message.args, category)
        elif self._forwarding_rule == "module":
            signature = (message.args, category, filename)
        elif self._forwarding_rule == "location":
            signature = (message.args, category, filename, lineno)

        if signature in self._forwarded:
            return
        self._forwarded.add(signature)
        if use_warnmsg is None:
            self._orig_show(message, category, filename, lineno, *args,
                            **kwargs)
        else:
            self._orig_showmsg(use_warnmsg)

    def __call__(self, func):
        """
        Function decorator to apply certain suppressions to a whole
        function.
        """
        @wraps(func)
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return new_func


@contextlib.contextmanager
def _assert_no_gc_cycles_context(name=None):
    __tracebackhide__ = True  # Hide traceback for py.test

    # not meaningful to test if there is no refcounting
    if not HAS_REFCOUNT:
        yield
        return

    assert_(gc.isenabled())
    gc.disable()
    gc_debug = gc.get_debug()
    try:
        for i in range(100):
            if gc.collect() == 0:
                break
        else:
            raise RuntimeError(
                "Unable to fully collect garbage - perhaps a __del__ method "
                "is creating more reference cycles?")

        gc.set_debug(gc.DEBUG_SAVEALL)
        yield
        # gc.collect returns the number of unreachable objects in cycles that
        # were found -- we are checking that no cycles were created in the context
        n_objects_in_cycles = gc.collect()
        objects_in_cycles = gc.garbage[:]
    finally:
        del gc.garbage[:]
        gc.set_debug(gc_debug)
        gc.enable()

    if n_objects_in_cycles:
        name_str = f' when calling {name}' if name is not None else ''
        raise AssertionError(
            "Reference cycles were found{}: {} objects were collected, "
            "of which {} are shown below:{}"
            .format(
                name_str,
                n_objects_in_cycles,
                len(objects_in_cycles),
                ''.join(
                    "\n  {} object with id={}:\n    {}".format(
                        type(o).__name__,
                        id(o),
                        pprint.pformat(o).replace('\n', '\n    ')
                    ) for o in objects_in_cycles
                )
            )
        )


def assert_no_gc_cycles(*args, **kwargs):
    """
    Fail if the given callable produces any reference cycles.

    If called with all arguments omitted, may be used as a context manager::

        with assert_no_gc_cycles():
            do_something()

    Parameters
    ----------
    func : callable
        The callable to test.
    \\*args : Arguments
        Arguments passed to `func`.
    \\*\\*kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    Nothing. The result is deliberately discarded to ensure that all cycles
    are found.

    """
    if not args:
        return _assert_no_gc_cycles_context()

    func = args[0]
    args = args[1:]
    with _assert_no_gc_cycles_context(name=func.__name__):
        func(*args, **kwargs)


def break_cycles():
    """
    Break reference cycles by calling gc.collect
    Objects can call other objects' methods (for instance, another object's
     __del__) inside their own __del__. On PyPy, the interpreter only runs
    between calls to gc.collect, so multiple calls are needed to completely
    release all cycles.
    """

    gc.collect()
    if IS_PYPY:
        # a few more, just to make sure all the finalizers are called
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()


def requires_memory(free_bytes):
    """Decorator to skip a test if not enough memory is available"""
    import pytest

    def decorator(func):
        @wraps(func)
        def wrapper(*a, **kw):
            msg = check_free_memory(free_bytes)
            if msg is not None:
                pytest.skip(msg)

            try:
                return func(*a, **kw)
            except MemoryError:
                # Probably ran out of memory regardless: don't regard as failure
                pytest.xfail("MemoryError raised")

        return wrapper

    return decorator


def check_free_memory(free_bytes):
    """
    Check whether `free_bytes` amount of memory is currently free.
    Returns: None if enough memory available, otherwise error message
    """
    env_var = 'NPY_AVAILABLE_MEM'
    env_value = os.environ.get(env_var)
    if env_value is not None:
        try:
            mem_free = _parse_size(env_value)
        except ValueError as exc:
            raise ValueError(f'Invalid environment variable {env_var}: {exc}')

        msg = (f'{free_bytes / 1e9} GB memory required, but environment variable '
               f'NPY_AVAILABLE_MEM={env_value} set')
    else:
        mem_free = _get_mem_available()

        if mem_free is None:
            msg = ("Could not determine available memory; set NPY_AVAILABLE_MEM "
                   "environment variable (e.g. NPY_AVAILABLE_MEM=16GB) to run "
                   "the test.")
            mem_free = -1
        else:
            free_bytes_gb = free_bytes / 1e9
            mem_free_gb = mem_free / 1e9
            msg = f'{free_bytes_gb} GB memory required, but {mem_free_gb} GB available'

    return msg if mem_free < free_bytes else None


def _parse_size(size_str):
    """Convert memory size strings ('12 GB' etc.) to float"""
    suffixes = {'': 1, 'b': 1,
                'k': 1000, 'm': 1000**2, 'g': 1000**3, 't': 1000**4,
                'kb': 1000, 'mb': 1000**2, 'gb': 1000**3, 'tb': 1000**4,
                'kib': 1024, 'mib': 1024**2, 'gib': 1024**3, 'tib': 1024**4}

    pipe_suffixes = "|".join(suffixes.keys())

    size_re = re.compile(fr'^\s*(\d+|\d+\.\d+)\s*({pipe_suffixes})\s*$', re.I)

    m = size_re.match(size_str.lower())
    if not m or m.group(2) not in suffixes:
        raise ValueError(f'value {size_str!r} not a valid size')
    return int(float(m.group(1)) * suffixes[m.group(2)])


def _get_mem_available():
    """Return available memory in bytes, or None if unknown."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass

    if sys.platform.startswith('linux'):
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                p = line.split()
                info[p[0].strip(':').lower()] = int(p[1]) * 1024

        if 'memavailable' in info:
            # Linux >= 3.14
            return info['memavailable']
        else:
            return info['memfree'] + info['cached']

    return None


def _no_tracing(func):
    """
    Decorator to temporarily turn off tracing for the duration of a test.
    Needed in tests that check refcounting, otherwise the tracing itself
    influences the refcounts
    """
    if not hasattr(sys, 'gettrace'):
        return func
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_trace = sys.gettrace()
            try:
                sys.settrace(None)
                return func(*args, **kwargs)
            finally:
                sys.settrace(original_trace)
        return wrapper


def _get_glibc_version():
    try:
        ver = os.confstr('CS_GNU_LIBC_VERSION').rsplit(' ')[1]
    except Exception:
        ver = '0.0'

    return ver


_glibcver = _get_glibc_version()
_glibc_older_than = lambda x: (_glibcver != '0.0' and _glibcver < x)


def run_threaded(func, max_workers=8, pass_count=False,
                 pass_barrier=False, outer_iterations=1,
                 prepare_args=None):
    """Runs a function many times in parallel"""
    for _ in range(outer_iterations):
        with (concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
              as tpe):
            if prepare_args is None:
                args = []
            else:
                args = prepare_args()
            if pass_barrier:
                barrier = threading.Barrier(max_workers)
                args.append(barrier)
            if pass_count:
                all_args = [(func, i, *args) for i in range(max_workers)]
            else:
                all_args = [(func, *args) for i in range(max_workers)]
            try:
                futures = []
                for arg in all_args:
                    futures.append(tpe.submit(*arg))
            except RuntimeError as e:
                import pytest
                pytest.skip(f"Spawning {max_workers} threads failed with "
                            f"error {e!r} (likely due to resource limits on the "
                            "system running the tests)")
            finally:
                if len(futures) < max_workers and pass_barrier:
                    barrier.abort()
            for f in futures:
                f.result()