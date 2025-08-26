
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\systems.py ===
from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
                                 AppliedUndef, Function, Subs)
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
                             piecewise_fold, sqrt, log)
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
                                       strongly_connected_components)
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate


def _get_func_order(eqs, funcs):
    return {func: max(ode_order(eq, func) for eq in eqs) for func in funcs}


class ODEOrderError(ValueError):
    """Raised by linear_ode_to_matrix if the system has the wrong order"""
    pass


class ODENonlinearError(NonlinearError):
    """Raised by linear_ode_to_matrix if the system is nonlinear"""
    pass


def _simpsol(soleq):
    lhs = soleq.lhs
    sol = soleq.rhs
    sol = powsimp(sol)
    gens = list(sol.atoms(exp))
    p = Poly(sol, *gens, expand=False)
    gens = [factor_terms(g) for g in gens]
    if not gens:
        gens = p.gens
    syms = [Symbol('C1'), Symbol('C2')]
    terms = []
    for coeff, monom in zip(p.coeffs(), p.monoms()):
        coeff = piecewise_fold(coeff)
        if isinstance(coeff, Piecewise):
            coeff = Piecewise(*((ratsimp(coef).collect(syms), cond) for coef, cond in coeff.args))
        else:
            coeff = ratsimp(coeff).collect(syms)
        monom = Mul(*(g ** i for g, i in zip(gens, monom)))
        terms.append(coeff * monom)
    return Eq(lhs, Add(*terms))


def _solsimp(e, t):
    no_t, has_t = powsimp(expand_mul(e)).as_independent(t)

    no_t = ratsimp(no_t)
    has_t = has_t.replace(exp, lambda a: exp(factor_terms(a)))

    return no_t + has_t


def simpsol(sol, wrt1, wrt2, doit=True):
    """Simplify solutions from dsolve_system."""

    # The parameter sol is the solution as returned by dsolve (list of Eq).
    #
    # The parameters wrt1 and wrt2 are lists of symbols to be collected for
    # with those in wrt1 being collected for first. This allows for collecting
    # on any factors involving the independent variable before collecting on
    # the integration constants or vice versa using e.g.:
    #
    #     sol = simpsol(sol, [t], [C1, C2])  # t first, constants after
    #     sol = simpsol(sol, [C1, C2], [t])  # constants first, t after
    #
    # If doit=True (default) then simpsol will begin by evaluating any
    # unevaluated integrals. Since many integrals will appear multiple times
    # in the solutions this is done intelligently by computing each integral
    # only once.
    #
    # The strategy is to first perform simple cancellation with factor_terms
    # and then multiply out all brackets with expand_mul. This gives an Add
    # with many terms.
    #
    # We split each term into two multiplicative factors dep and coeff where
    # all factors that involve wrt1 are in dep and any constant factors are in
    # coeff e.g.
    #         sqrt(2)*C1*exp(t) -> ( exp(t), sqrt(2)*C1 )
    #
    # The dep factors are simplified using powsimp to combine expanded
    # exponential factors e.g.
    #              exp(a*t)*exp(b*t) -> exp(t*(a+b))
    #
    # We then collect coefficients for all terms having the same (simplified)
    # dep. The coefficients are then simplified using together and ratsimp and
    # lastly by recursively applying the same transformation to the
    # coefficients to collect on wrt2.
    #
    # Finally the result is recombined into an Add and signsimp is used to
    # normalise any minus signs.

    def simprhs(rhs, rep, wrt1, wrt2):
        """Simplify the rhs of an ODE solution"""
        if rep:
            rhs = rhs.subs(rep)
        rhs = factor_terms(rhs)
        rhs = simp_coeff_dep(rhs, wrt1, wrt2)
        rhs = signsimp(rhs)
        return rhs

    def simp_coeff_dep(expr, wrt1, wrt2=None):
        """Split rhs into terms, split terms into dep and coeff and collect on dep"""
        add_dep_terms = lambda e: e.is_Add and e.has(*wrt1)
        expandable = lambda e: e.is_Mul and any(map(add_dep_terms, e.args))
        expand_func = lambda e: expand_mul(e, deep=False)
        expand_mul_mod = lambda e: e.replace(expandable, expand_func)
        terms = Add.make_args(expand_mul_mod(expr))
        dc = {}
        for term in terms:
            coeff, dep = term.as_independent(*wrt1, as_Add=False)
            # Collect together the coefficients for terms that have the same
            # dependence on wrt1 (after dep is normalised using simpdep).
            dep = simpdep(dep, wrt1)

            # See if the dependence on t cancels out...
            if dep is not S.One:
                dep2 = factor_terms(dep)
                if not dep2.has(*wrt1):
                    coeff *= dep2
                    dep = S.One

            if dep not in dc:
                dc[dep] = coeff
            else:
                dc[dep] += coeff
        # Apply the method recursively to the coefficients but this time
        # collecting on wrt2 rather than wrt2.
        termpairs = ((simpcoeff(c, wrt2), d) for d, c in dc.items())
        if wrt2 is not None:
            termpairs = ((simp_coeff_dep(c, wrt2), d) for c, d in termpairs)
        return Add(*(c * d for c, d in termpairs))

    def simpdep(term, wrt1):
        """Normalise factors involving t with powsimp and recombine exp"""
        def canonicalise(a):
            # Using factor_terms here isn't quite right because it leads to things
            # like exp(t*(1+t)) that we don't want. We do want to cancel factors
            # and pull out a common denominator but ideally the numerator would be
            # expressed as a standard form polynomial in t so we expand_mul
            # and collect afterwards.
            a = factor_terms(a)
            num, den = a.as_numer_denom()
            num = expand_mul(num)
            num = collect(num, wrt1)
            return num / den

        term = powsimp(term)
        rep = {e: exp(canonicalise(e.args[0])) for e in term.atoms(exp)}
        term = term.subs(rep)
        return term

    def simpcoeff(coeff, wrt2):
        """Bring to a common fraction and cancel with ratsimp"""
        coeff = together(coeff)
        if coeff.is_polynomial():
            # Calling ratsimp can be expensive. The main reason is to simplify
            # sums of terms with irrational denominators so we limit ourselves
            # to the case where the expression is polynomial in any symbols.
            # Maybe there's a better approach...
            coeff = ratsimp(radsimp(coeff))
        # collect on secondary variables first and any remaining symbols after
        if wrt2 is not None:
            syms = list(wrt2) + list(ordered(coeff.free_symbols - set(wrt2)))
        else:
            syms = list(ordered(coeff.free_symbols))
        coeff = collect(coeff, syms)
        coeff = together(coeff)
        return coeff

    # There are often repeated integrals. Collect unique integrals and
    # evaluate each once and then substitute into the final result to replace
    # all occurrences in each of the solution equations.
    if doit:
        integrals = set().union(*(s.atoms(Integral) for s in sol))
        rep = {i: factor_terms(i).doit() for i in integrals}
    else:
        rep = {}

    sol = [Eq(s.lhs, simprhs(s.rhs, rep, wrt1, wrt2)) for s in sol]
    return sol


def linodesolve_type(A, t, b=None):
    r"""
    Helper function that determines the type of the system of ODEs for solving with :obj:`sympy.solvers.ode.systems.linodesolve()`

    Explanation
    ===========

    This function takes in the coefficient matrix and/or the non-homogeneous term
    and returns the type of the equation that can be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`.

    If the system is constant coefficient homogeneous, then "type1" is returned

    If the system is constant coefficient non-homogeneous, then "type2" is returned

    If the system is non-constant coefficient homogeneous, then "type3" is returned

    If the system is non-constant coefficient non-homogeneous, then "type4" is returned

    If the system has a non-constant coefficient matrix which can be factorized into constant
    coefficient matrix, then "type5" or "type6" is returned for when the system is homogeneous or
    non-homogeneous respectively.

    Note that, if the system of ODEs is of "type3" or "type4", then along with the type,
    the commutative antiderivative of the coefficient matrix is also returned.

    If the system cannot be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`, then
    NotImplementedError is raised.

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of ODEs
    b : Matrix or None
        Non-homogeneous term of the system. The default value is None.
        If this argument is None, then the system is assumed to be homogeneous.

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import linodesolve_type
    >>> t = symbols("t")
    >>> A = Matrix([[1, 1], [2, 3]])
    >>> b = Matrix([t, 1])

    >>> linodesolve_type(A, t)
    {'antiderivative': None, 'type_of_equation': 'type1'}

    >>> linodesolve_type(A, t, b=b)
    {'antiderivative': None, 'type_of_equation': 'type2'}

    >>> A_t = Matrix([[1, t], [-t, 1]])

    >>> linodesolve_type(A_t, t)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type3'}

    >>> linodesolve_type(A_t, t, b=b)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type4'}

    >>> A_non_commutative = Matrix([[1, t], [t, -1]])
    >>> linodesolve_type(A_non_commutative, t)
    Traceback (most recent call last):
    ...
    NotImplementedError:
    The system does not have a commutative antiderivative, it cannot be
    solved by linodesolve.

    Returns
    =======

    Dict

    Raises
    ======

    NotImplementedError
        When the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linodesolve: Function for which linodesolve_type gets the information

    """

    match = {}
    is_non_constant = not _matrix_is_constant(A, t)
    is_non_homogeneous = not (b is None or b.is_zero_matrix)
    type = "type{}".format(int("{}{}".format(int(is_non_constant), int(is_non_homogeneous)), 2) + 1)

    B = None
    match.update({"type_of_equation": type, "antiderivative": B})

    if is_non_constant:
        B, is_commuting = _is_commutative_anti_derivative(A, t)
        if not is_commuting:
            raise NotImplementedError(filldedent('''
                The system does not have a commutative antiderivative, it cannot be solved
                by linodesolve.
            '''))

        match['antiderivative'] = B
        match.update(_first_order_type5_6_subs(A, t, b=b))

    return match


def _first_order_type5_6_subs(A, t, b=None):
    match = {}

    factor_terms = _factor_matrix(A, t)
    is_homogeneous = b is None or b.is_zero_matrix

    if factor_terms is not None:
        t_ = Symbol("{}_".format(t))
        F_t = integrate(factor_terms[0], t)
        inverse = solveset(Eq(t_, F_t), t)

        # Note: A simple way to check if a function is invertible
        # or not.
        if isinstance(inverse, FiniteSet) and not inverse.has(Piecewise)\
            and len(inverse) == 1:

            A = factor_terms[1]
            if not is_homogeneous:
                b = b / factor_terms[0]
                b = b.subs(t, list(inverse)[0])
            type = "type{}".format(5 + (not is_homogeneous))
            match.update({'func_coeff': A, 'tau': F_t,
                          't_': t_, 'type_of_equation': type, 'rhs': b})

    return match


def linear_ode_to_matrix(eqs, funcs, t, order):
    r"""
    Convert a linear system of ODEs to matrix form

    Explanation
    ===========

    Express a system of linear ordinary differential equations as a single
    matrix differential equation [1]. For example the system $x' = x + y + 1$
    and $y' = x - y$ can be represented as

    .. math:: A_1 X' = A_0 X + b

    where $A_1$ and $A_0$ are $2 \times 2$ matrices and $b$, $X$ and $X'$ are
    $2 \times 1$ matrices with $X = [x, y]^T$.

    Higher-order systems are represented with additional matrices e.g. a
    second-order system would look like

    .. math:: A_2 X'' =  A_1 X' + A_0 X  + b

    Examples
    ========

    >>> from sympy import Function, Symbol, Matrix, Eq
    >>> from sympy.solvers.ode.systems import linear_ode_to_matrix
    >>> t = Symbol('t')
    >>> x = Function('x')
    >>> y = Function('y')

    We can create a system of linear ODEs like

    >>> eqs = [
    ...     Eq(x(t).diff(t), x(t) + y(t) + 1),
    ...     Eq(y(t).diff(t), x(t) - y(t)),
    ... ]
    >>> funcs = [x(t), y(t)]
    >>> order = 1 # 1st order system

    Now ``linear_ode_to_matrix`` can represent this as a matrix
    differential equation.

    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, order)
    >>> A1
    Matrix([
    [1, 0],
    [0, 1]])
    >>> A0
    Matrix([
    [1, 1],
    [1,  -1]])
    >>> b
    Matrix([
    [1],
    [0]])

    The original equations can be recovered from these matrices:

    >>> eqs_mat = Matrix([eq.lhs - eq.rhs for eq in eqs])
    >>> X = Matrix(funcs)
    >>> A1 * X.diff(t) - A0 * X - b == eqs_mat
    True

    If the system of equations has a maximum order greater than the
    order of the system specified, a ODEOrderError exception is raised.

    >>> eqs = [Eq(x(t).diff(t, 2), x(t).diff(t) + x(t)), Eq(y(t).diff(t), y(t) + x(t))]
    >>> linear_ode_to_matrix(eqs, funcs, t, 1)
    Traceback (most recent call last):
    ...
    ODEOrderError: Cannot represent system in 1-order form

    If the system of equations is nonlinear, then ODENonlinearError is
    raised.

    >>> eqs = [Eq(x(t).diff(t), x(t) + y(t)), Eq(y(t).diff(t), y(t)**2 + x(t))]
    >>> linear_ode_to_matrix(eqs, funcs, t, 1)
    Traceback (most recent call last):
    ...
    ODENonlinearError: The system of ODEs is nonlinear.

    Parameters
    ==========

    eqs : list of SymPy expressions or equalities
        The equations as expressions (assumed equal to zero).
    funcs : list of applied functions
        The dependent variables of the system of ODEs.
    t : symbol
        The independent variable.
    order : int
        The order of the system of ODEs.

    Returns
    =======

    The tuple ``(As, b)`` where ``As`` is a tuple of matrices and ``b`` is the
    the matrix representing the rhs of the matrix equation.

    Raises
    ======

    ODEOrderError
        When the system of ODEs have an order greater than what was specified
    ODENonlinearError
        When the system of ODEs is nonlinear

    See Also
    ========

    linear_eq_to_matrix: for systems of linear algebraic equations.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_differential_equation

    """
    from sympy.solvers.solveset import linear_eq_to_matrix

    if any(ode_order(eq, func) > order for eq in eqs for func in funcs):
        msg = "Cannot represent system in {}-order form"
        raise ODEOrderError(msg.format(order))

    As = []

    for o in range(order, -1, -1):
        # Work from the highest derivative down
        syms = [func.diff(t, o) for func in funcs]

        # Ai is the matrix for X(t).diff(t, o)
        # eqs is minus the remainder of the equations.
        try:
            Ai, b = linear_eq_to_matrix(eqs, syms)
        except NonlinearError:
            raise ODENonlinearError("The system of ODEs is nonlinear.")

        Ai = Ai.applyfunc(expand_mul)

        As.append(Ai if o == order else -Ai)

        if o:
            eqs = [-eq for eq in b]
        else:
            rhs = b

    return As, rhs


def matrix_exp(A, t):
    r"""
    Matrix exponential $\exp(A*t)$ for the matrix ``A`` and scalar ``t``.

    Explanation
    ===========

    This functions returns the $\exp(A*t)$ by doing a simple
    matrix multiplication:

    .. math:: \exp(A*t) = P * expJ * P^{-1}

    where $expJ$ is $\exp(J*t)$. $J$ is the Jordan normal
    form of $A$ and $P$ is matrix such that:

    .. math:: A = P * J * P^{-1}

    The matrix exponential $\exp(A*t)$ appears in the solution of linear
    differential equations. For example if $x$ is a vector and $A$ is a matrix
    then the initial value problem

    .. math:: \frac{dx(t)}{dt} = A \times x(t),   x(0) = x0

    has the unique solution

    .. math:: x(t) = \exp(A t) x0

    Examples
    ========

    >>> from sympy import Symbol, Matrix, pprint
    >>> from sympy.solvers.ode.systems import matrix_exp
    >>> t = Symbol('t')

    We will consider a 2x2 matrix for comupting the exponential

    >>> A = Matrix([[2, -5], [2, -4]])
    >>> pprint(A)
    [2  -5]
    [     ]
    [2  -4]

    Now, exp(A*t) is given as follows:

    >>> pprint(matrix_exp(A, t))
    [   -t           -t                    -t              ]
    [3*e  *sin(t) + e  *cos(t)         -5*e  *sin(t)       ]
    [                                                      ]
    [         -t                     -t           -t       ]
    [      2*e  *sin(t)         - 3*e  *sin(t) + e  *cos(t)]

    Parameters
    ==========

    A : Matrix
        The matrix $A$ in the expression $\exp(A*t)$
    t : Symbol
        The independent variable

    See Also
    ========

    matrix_exp_jordan_form: For exponential of Jordan normal form

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jordan_normal_form
    .. [2] https://en.wikipedia.org/wiki/Matrix_exponential

    """
    P, expJ = matrix_exp_jordan_form(A, t)
    return P * expJ * P.inv()


def matrix_exp_jordan_form(A, t):
    r"""
    Matrix exponential $\exp(A*t)$ for the matrix *A* and scalar *t*.

    Explanation
    ===========

    Returns the Jordan form of the $\exp(A*t)$ along with the matrix $P$ such that:

    .. math::
        \exp(A*t) = P * expJ * P^{-1}

    Examples
    ========

    >>> from sympy import Matrix, Symbol
    >>> from sympy.solvers.ode.systems import matrix_exp, matrix_exp_jordan_form
    >>> t = Symbol('t')

    We will consider a 2x2 defective matrix. This shows that our method
    works even for defective matrices.

    >>> A = Matrix([[1, 1], [0, 1]])

    It can be observed that this function gives us the Jordan normal form
    and the required invertible matrix P.

    >>> P, expJ = matrix_exp_jordan_form(A, t)

    Here, it is shown that P and expJ returned by this function is correct
    as they satisfy the formula: P * expJ * P_inverse = exp(A*t).

    >>> P * expJ * P.inv() == matrix_exp(A, t)
    True

    Parameters
    ==========

    A : Matrix
        The matrix $A$ in the expression $\exp(A*t)$
    t : Symbol
        The independent variable

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Defective_matrix
    .. [2] https://en.wikipedia.org/wiki/Jordan_matrix
    .. [3] https://en.wikipedia.org/wiki/Jordan_normal_form

    """

    N, M = A.shape
    if N != M:
        raise ValueError('Needed square matrix but got shape (%s, %s)' % (N, M))
    elif A.has(t):
        raise ValueError('Matrix A should not depend on t')

    def jordan_chains(A):
        '''Chains from Jordan normal form analogous to M.eigenvects().
        Returns a dict with eignevalues as keys like:
            {e1: [[v111,v112,...], [v121, v122,...]], e2:...}
        where vijk is the kth vector in the jth chain for eigenvalue i.
        '''
        P, blocks = A.jordan_cells()
        basis = [P[:,i] for i in range(P.shape[1])]
        n = 0
        chains = {}
        for b in blocks:
            eigval = b[0, 0]
            size = b.shape[0]
            if eigval not in chains:
                chains[eigval] = []
            chains[eigval].append(basis[n:n+size])
            n += size
        return chains

    eigenchains = jordan_chains(A)

    # Needed for consistency across Python versions
    eigenchains_iter = sorted(eigenchains.items(), key=default_sort_key)
    isreal = not A.has(I)

    blocks = []
    vectors = []
    seen_conjugate = set()
    for e, chains in eigenchains_iter:
        for chain in chains:
            n = len(chain)
            if isreal and e != e.conjugate() and e.conjugate() in eigenchains:
                if e in seen_conjugate:
                    continue
                seen_conjugate.add(e.conjugate())
                exprt = exp(re(e) * t)
                imrt = im(e) * t
                imblock = Matrix([[cos(imrt), sin(imrt)],
                                  [-sin(imrt), cos(imrt)]])
                expJblock2 = Matrix(n, n, lambda i,j:
                        imblock * t**(j-i) / factorial(j-i) if j >= i
                        else zeros(2, 2))
                expJblock = Matrix(2*n, 2*n, lambda i,j: expJblock2[i//2,j//2][i%2,j%2])

                blocks.append(exprt * expJblock)
                for i in range(n):
                    vectors.append(re(chain[i]))
                    vectors.append(im(chain[i]))
            else:
                vectors.extend(chain)
                fun = lambda i,j: t**(j-i)/factorial(j-i) if j >= i else 0
                expJblock = Matrix(n, n, fun)
                blocks.append(exp(e * t) * expJblock)

    expJ = Matrix.diag(*blocks)
    P = Matrix(N, N, lambda i,j: vectors[j][i])

    return P, expJ


# Note: To add a docstring example with tau
def linodesolve(A, t, b=None, B=None, type="auto", doit=False,
                tau=None):
    r"""
    System of n equations linear first-order differential equations

    Explanation
    ===========

    This solver solves the system of ODEs of the following form:

    .. math::
        X'(t) = A(t) X(t) +  b(t)

    Here, $A(t)$ is the coefficient matrix, $X(t)$ is the vector of n independent variables,
    $b(t)$ is the non-homogeneous term and $X'(t)$ is the derivative of $X(t)$

    Depending on the properties of $A(t)$ and $b(t)$, this solver evaluates the solution
    differently.

    When $A(t)$ is constant coefficient matrix and $b(t)$ is zero vector i.e. system is homogeneous,
    the system is "type1". The solution is:

    .. math::
        X(t) = \exp(A t) C

    Here, $C$ is a vector of constants and $A$ is the constant coefficient matrix.

    When $A(t)$ is constant coefficient matrix and $b(t)$ is non-zero i.e. system is non-homogeneous,
    the system is "type2". The solution is:

    .. math::
        X(t) = e^{A t} ( \int e^{- A t} b \,dt + C)

    When $A(t)$ is coefficient matrix such that its commutative with its antiderivative $B(t)$ and
    $b(t)$ is a zero vector i.e. system is homogeneous, the system is "type3". The solution is:

    .. math::
        X(t) = \exp(B(t)) C

    When $A(t)$ is commutative with its antiderivative $B(t)$ and $b(t)$ is non-zero i.e. system is
    non-homogeneous, the system is "type4". The solution is:

    .. math::
        X(t) =  e^{B(t)} ( \int e^{-B(t)} b(t) \,dt + C)

    When $A(t)$ is a coefficient matrix such that it can be factorized into a scalar and a constant
    coefficient matrix:

    .. math::
        A(t) = f(t) * A

    Where $f(t)$ is a scalar expression in the independent variable $t$ and $A$ is a constant matrix,
    then we can do the following substitutions:

    .. math::
        tau = \int f(t) dt, X(t) = Y(tau), b(t) = b(f^{-1}(tau))

    Here, the substitution for the non-homogeneous term is done only when its non-zero.
    Using these substitutions, our original system becomes:

    .. math::
        Y'(tau) = A * Y(tau) + b(tau)/f(tau)

    The above system can be easily solved using the solution for "type1" or "type2" depending
    on the homogeneity of the system. After we get the solution for $Y(tau)$, we substitute the
    solution for $tau$ as $t$ to get back $X(t)$

    .. math::
        X(t) = Y(tau)

    Systems of "type5" and "type6" have a commutative antiderivative but we use this solution
    because its faster to compute.

    The final solution is the general solution for all the four equations since a constant coefficient
    matrix is always commutative with its antidervative.

    An additional feature of this function is, if someone wants to substitute for value of the independent
    variable, they can pass the substitution `tau` and the solution will have the independent variable
    substituted with the passed expression(`tau`).

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of linear first order ODEs.
    t : Symbol
        Independent variable in the system of ODEs.
    b : Matrix or None
        Non-homogeneous term in the system of ODEs. If None is passed,
        a homogeneous system of ODEs is assumed.
    B : Matrix or None
        Antiderivative of the coefficient matrix. If the antiderivative
        is not passed and the solution requires the term, then the solver
        would compute it internally.
    type : String
        Type of the system of ODEs passed. Depending on the type, the
        solution is evaluated. The type values allowed and the corresponding
        system it solves are: "type1" for constant coefficient homogeneous
        "type2" for constant coefficient non-homogeneous, "type3" for non-constant
        coefficient homogeneous, "type4" for non-constant coefficient non-homogeneous,
        "type5" and "type6" for non-constant coefficient homogeneous and non-homogeneous
        systems respectively where the coefficient matrix can be factorized to a constant
        coefficient matrix.
        The default value is "auto" which will let the solver decide the correct type of
        the system passed.
    doit : Boolean
        Evaluate the solution if True, default value is False
    tau: Expression
        Used to substitute for the value of `t` after we get the solution of the system.

    Examples
    ========

    To solve the system of ODEs using this function directly, several things must be
    done in the right order. Wrong inputs to the function will lead to incorrect results.

    >>> from sympy import symbols, Function, Eq
    >>> from sympy.solvers.ode.systems import canonical_odes, linear_ode_to_matrix, linodesolve, linodesolve_type
    >>> from sympy.solvers.ode.subscheck import checkodesol
    >>> f, g = symbols("f, g", cls=Function)
    >>> x, a = symbols("x, a")
    >>> funcs = [f(x), g(x)]
    >>> eqs = [Eq(f(x).diff(x) - f(x), a*g(x) + 1), Eq(g(x).diff(x) + g(x), a*f(x))]

    Here, it is important to note that before we derive the coefficient matrix, it is
    important to get the system of ODEs into the desired form. For that we will use
    :obj:`sympy.solvers.ode.systems.canonical_odes()`.

    >>> eqs = canonical_odes(eqs, funcs, x)
    >>> eqs
    [[Eq(Derivative(f(x), x), a*g(x) + f(x) + 1), Eq(Derivative(g(x), x), a*f(x) - g(x))]]

    Now, we will use :obj:`sympy.solvers.ode.systems.linear_ode_to_matrix()` to get the coefficient matrix and the
    non-homogeneous term if it is there.

    >>> eqs = eqs[0]
    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)
    >>> A = A0

    We have the coefficient matrices and the non-homogeneous term ready. Now, we can use
    :obj:`sympy.solvers.ode.systems.linodesolve_type()` to get the information for the system of ODEs
    to finally pass it to the solver.

    >>> system_info = linodesolve_type(A, x, b=b)
    >>> sol_vector = linodesolve(A, x, b=b, B=system_info['antiderivative'], type=system_info['type_of_equation'])

    Now, we can prove if the solution is correct or not by using :obj:`sympy.solvers.ode.checkodesol()`

    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    We can also use the doit method to evaluate the solutions passed by the function.

    >>> sol_vector_evaluated = linodesolve(A, x, b=b, type="type2", doit=True)

    Now, we will look at a system of ODEs which is non-constant.

    >>> eqs = [Eq(f(x).diff(x), f(x) + x*g(x)), Eq(g(x).diff(x), -x*f(x) + g(x))]

    The system defined above is already in the desired form, so we do not have to convert it.

    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)
    >>> A = A0

    A user can also pass the commutative antiderivative required for type3 and type4 system of ODEs.
    Passing an incorrect one will lead to incorrect results. If the coefficient matrix is not commutative
    with its antiderivative, then :obj:`sympy.solvers.ode.systems.linodesolve_type()` raises a NotImplementedError.
    If it does have a commutative antiderivative, then the function just returns the information about the system.

    >>> system_info = linodesolve_type(A, x, b=b)

    Now, we can pass the antiderivative as an argument to get the solution. If the system information is not
    passed, then the solver will compute the required arguments internally.

    >>> sol_vector = linodesolve(A, x, b=b)

    Once again, we can verify the solution obtained.

    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    Returns
    =======

    List

    Raises
    ======

    ValueError
        This error is raised when the coefficient matrix, non-homogeneous term
        or the antiderivative, if passed, are not a matrix or
        do not have correct dimensions
    NonSquareMatrixError
        When the coefficient matrix or its antiderivative, if passed is not a
        square matrix
    NotImplementedError
        If the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linear_ode_to_matrix: Coefficient matrix computation function
    canonical_odes: System of ODEs representation change
    linodesolve_type: Getting information about systems of ODEs to pass in this solver

    """

    if not isinstance(A, MatrixBase):
        raise ValueError(filldedent('''\
            The coefficients of the system of ODEs should be of type Matrix
        '''))

    if not A.is_square:
        raise NonSquareMatrixError(filldedent('''\
            The coefficient matrix must be a square
        '''))

    if b is not None:
        if not isinstance(b, MatrixBase):
            raise ValueError(filldedent('''\
                The non-homogeneous terms of the system of ODEs should be of type Matrix
            '''))

        if A.rows != b.rows:
            raise ValueError(filldedent('''\
                The system of ODEs should have the same number of non-homogeneous terms and the number of
                equations
            '''))

    if B is not None:
        if not isinstance(B, MatrixBase):
            raise ValueError(filldedent('''\
                The antiderivative of coefficients of the system of ODEs should be of type Matrix
            '''))

        if not B.is_square:
            raise NonSquareMatrixError(filldedent('''\
                The antiderivative of the coefficient matrix must be a square
            '''))

        if A.rows != B.rows:
            raise ValueError(filldedent('''\
                        The coefficient matrix and its antiderivative should have same dimensions
                    '''))

    if not any(type == "type{}".format(i) for i in range(1, 7)) and not type == "auto":
        raise ValueError(filldedent('''\
                    The input type should be a valid one
                '''))

    n = A.rows

    # constants = numbered_symbols(prefix='C', cls=Dummy, start=const_idx+1)
    Cvect = Matrix([Dummy() for _ in range(n)])

    if b is None and any(type == typ for typ in ["type2", "type4", "type6"]):
        b = zeros(n, 1)

    is_transformed = tau is not None
    passed_type = type

    if type == "auto":
        system_info = linodesolve_type(A, t, b=b)
        type = system_info["type_of_equation"]
        B = system_info["antiderivative"]

    if type in ("type5", "type6"):
        is_transformed = True
        if passed_type != "auto":
            if tau is None:
                system_info = _first_order_type5_6_subs(A, t, b=b)
                if not system_info:
                    raise ValueError(filldedent('''
                        The system passed isn't {}.
                    '''.format(type)))

                tau = system_info['tau']
                t = system_info['t_']
                A = system_info['A']
                b = system_info['b']

    intx_wrtt = lambda x: Integral(x, t) if x else 0
    if type in ("type1", "type2", "type5", "type6"):
        P, J = matrix_exp_jordan_form(A, t)
        P = simplify(P)

        if type in ("type1", "type5"):
            sol_vector = P * (J * Cvect)
        else:
            Jinv = J.subs(t, -t)
            sol_vector = P * J * ((Jinv * P.inv() * b).applyfunc(intx_wrtt) + Cvect)
    else:
        if B is None:
            B, _ = _is_commutative_anti_derivative(A, t)

        if type == "type3":
            sol_vector = B.exp() * Cvect
        else:
            sol_vector = B.exp() * (((-B).exp() * b).applyfunc(intx_wrtt) + Cvect)

    if is_transformed:
        sol_vector = sol_vector.subs(t, tau)

    gens = sol_vector.atoms(exp)

    if type != "type1":
        sol_vector = [expand_mul(s) for s in sol_vector]

    sol_vector = [collect(s, ordered(gens), exact=True) for s in sol_vector]

    if doit:
        sol_vector = [s.doit() for s in sol_vector]

    return sol_vector


def _matrix_is_constant(M, t):
    """Checks if the matrix M is independent of t or not."""
    return all(coef.as_independent(t, as_Add=True)[1] == 0 for coef in M)


def canonical_odes(eqs, funcs, t):
    r"""
    Function that solves for highest order derivatives in a system

    Explanation
    ===========

    This function inputs a system of ODEs and based on the system,
    the dependent variables and their highest order, returns the system
    in the following form:

    .. math::
        X'(t) = A(t) X(t) + b(t)

    Here, $X(t)$ is the vector of dependent variables of lower order, $A(t)$ is
    the coefficient matrix, $b(t)$ is the non-homogeneous term and $X'(t)$ is the
    vector of dependent variables in their respective highest order. We use the term
    canonical form to imply the system of ODEs which is of the above form.

    If the system passed has a non-linear term with multiple solutions, then a list of
    systems is returned in its canonical form.

    Parameters
    ==========

    eqs : List
        List of the ODEs
    funcs : List
        List of dependent variables
    t : Symbol
        Independent variable

    Examples
    ========

    >>> from sympy import symbols, Function, Eq, Derivative
    >>> from sympy.solvers.ode.systems import canonical_odes
    >>> f, g = symbols("f g", cls=Function)
    >>> x, y = symbols("x y")
    >>> funcs = [f(x), g(x)]
    >>> eqs = [Eq(f(x).diff(x) - 7*f(x), 12*g(x)), Eq(g(x).diff(x) + g(x), 20*f(x))]

    >>> canonical_eqs = canonical_odes(eqs, funcs, x)
    >>> canonical_eqs
    [[Eq(Derivative(f(x), x), 7*f(x) + 12*g(x)), Eq(Derivative(g(x), x), 20*f(x) - g(x))]]

    >>> system = [Eq(Derivative(f(x), x)**2 - 2*Derivative(f(x), x) + 1, 4), Eq(-y*f(x) + Derivative(g(x), x), 0)]

    >>> canonical_system = canonical_odes(system, funcs, x)
    >>> canonical_system
    [[Eq(Derivative(f(x), x), -1), Eq(Derivative(g(x), x), y*f(x))], [Eq(Derivative(f(x), x), 3), Eq(Derivative(g(x), x), y*f(x))]]

    Returns
    =======

    List

    """
    from sympy.solvers.solvers import solve

    order = _get_func_order(eqs, funcs)

    canon_eqs = solve(eqs, *[func.diff(t, order[func]) for func in funcs], dict=True)

    systems = []
    for eq in canon_eqs:
        system = [Eq(func.diff(t, order[func]), eq[func.diff(t, order[func])]) for func in funcs]
        systems.append(system)

    return systems


def _is_commutative_anti_derivative(A, t):
    r"""
    Helper function for determining if the Matrix passed is commutative with its antiderivative

    Explanation
    ===========

    This function checks if the Matrix $A$ passed is commutative with its antiderivative with respect
    to the independent variable $t$.

    .. math::
        B(t) = \int A(t) dt

    The function outputs two values, first one being the antiderivative $B(t)$, second one being a
    boolean value, if True, then the matrix $A(t)$ passed is commutative with $B(t)$, else the matrix
    passed isn't commutative with $B(t)$.

    Parameters
    ==========

    A : Matrix
        The matrix which has to be checked
    t : Symbol
        Independent variable

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import _is_commutative_anti_derivative
    >>> t = symbols("t")
    >>> A = Matrix([[1, t], [-t, 1]])

    >>> B, is_commuting = _is_commutative_anti_derivative(A, t)
    >>> is_commuting
    True

    Returns
    =======

    Matrix, Boolean

    """
    B = integrate(A, t)
    is_commuting = (B*A - A*B).applyfunc(expand).applyfunc(factor_terms).is_zero_matrix

    is_commuting = False if is_commuting is None else is_commuting

    return B, is_commuting


def _factor_matrix(A, t):
    term = None
    for element in A:
        temp_term = element.as_independent(t)[1]
        if temp_term.has(t):
            term = temp_term
            break

    if term is not None:
        A_factored = (A/term).applyfunc(ratsimp)
        can_factor = _matrix_is_constant(A_factored, t)
        term = (term, A_factored) if can_factor else None

    return term


def _is_second_order_type2(A, t):
    term = _factor_matrix(A, t)
    is_type2 = False

    if term is not None:
        term = 1/term[0]
        is_type2 = term.is_polynomial()

    if is_type2:
        poly = Poly(term.expand(), t)
        monoms = poly.monoms()

        if monoms[0][0] in (2, 4):
            cs = _get_poly_coeffs(poly, 4)
            a, b, c, d, e = cs

            a1 = powdenest(sqrt(a), force=True)
            c1 = powdenest(sqrt(e), force=True)
            b1 = powdenest(sqrt(c - 2*a1*c1), force=True)

            is_type2 = (b == 2*a1*b1) and (d == 2*b1*c1)
            term = a1*t**2 + b1*t + c1

        else:
            is_type2 = False

    return is_type2, term


def _get_poly_coeffs(poly, order):
    cs = [0 for _ in range(order+1)]
    for c, m in zip(poly.coeffs(), poly.monoms()):
        cs[-1-m[0]] = c
    return cs


def _match_second_order_type(A1, A0, t, b=None):
    r"""
    Works only for second order system in its canonical form.

    Type 0: Constant coefficient matrix, can be simply solved by
            introducing dummy variables.
    Type 1: When the substitution: $U = t*X' - X$ works for reducing
            the second order system to first order system.
    Type 2: When the system is of the form: $poly * X'' = A*X$ where
            $poly$ is square of a quadratic polynomial with respect to
            *t* and $A$ is a constant coefficient matrix.

    """
    match = {"type_of_equation": "type0"}
    n = A1.shape[0]

    if _matrix_is_constant(A1, t) and _matrix_is_constant(A0, t):
        return match

    if (A1 + A0*t).applyfunc(expand_mul).is_zero_matrix:
        match.update({"type_of_equation": "type1", "A1": A1})

    elif A1.is_zero_matrix and (b is None or b.is_zero_matrix):
        is_type2, term = _is_second_order_type2(A0, t)
        if is_type2:
            a, b, c = _get_poly_coeffs(Poly(term, t), 2)
            A = (A0*(term**2).expand()).applyfunc(ratsimp) + (b**2/4 - a*c)*eye(n, n)
            tau = integrate(1/term, t)
            t_ = Symbol("{}_".format(t))
            match.update({"type_of_equation": "type2", "A0": A,
                          "g(t)": sqrt(term), "tau": tau, "is_transformed": True,
                          "t_": t_})

    return match


def _second_order_subs_type1(A, b, funcs, t):
    r"""
    For a linear, second order system of ODEs, a particular substitution.

    A system of the below form can be reduced to a linear first order system of
    ODEs:
    .. math::
        X'' = A(t) * (t*X' - X) + b(t)

    By substituting:
    .. math::  U = t*X' - X

    To get the system:
    .. math::  U' = t*(A(t)*U + b(t))

    Where $U$ is the vector of dependent variables, $X$ is the vector of dependent
    variables in `funcs` and $X'$ is the first order derivative of $X$ with respect to
    $t$. It may or may not reduce the system into linear first order system of ODEs.

    Then a check is made to determine if the system passed can be reduced or not, if
    this substitution works, then the system is reduced and its solved for the new
    substitution. After we get the solution for $U$:

    .. math::  U = a(t)

    We substitute and return the reduced system:

    .. math::
        a(t) = t*X' - X

    Parameters
    ==========

    A: Matrix
        Coefficient matrix($A(t)*t$) of the second order system of this form.
    b: Matrix
        Non-homogeneous term($b(t)$) of the system of ODEs.
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the system of ODEs.

    Returns
    =======

    List

    """

    U = Matrix([t*func.diff(t) - func for func in funcs])

    sol = linodesolve(A, t, t*b)
    reduced_eqs = [Eq(u, s) for s, u in zip(sol, U)]
    reduced_eqs = canonical_odes(reduced_eqs, funcs, t)[0]

    return reduced_eqs


def _second_order_subs_type2(A, funcs, t_):
    r"""
    Returns a second order system based on the coefficient matrix passed.

    Explanation
    ===========

    This function returns a system of second order ODE of the following form:

    .. math::
        X'' = A * X

    Here, $X$ is the vector of dependent variables, but a bit modified, $A$ is the
    coefficient matrix passed.

    Along with returning the second order system, this function also returns the new
    dependent variables with the new independent variable `t_` passed.

    Parameters
    ==========

    A: Matrix
        Coefficient matrix of the system
    funcs: List
        List of old dependent variables
    t_: Symbol
        New independent variable

    Returns
    =======

    List, List

    """
    func_names = [func.func.__name__ for func in funcs]
    new_funcs = [Function(Dummy("{}_".format(name)))(t_) for name in func_names]
    rhss = A * Matrix(new_funcs)
    new_eqs = [Eq(func.diff(t_, 2), rhs) for func, rhs in zip(new_funcs, rhss)]

    return new_eqs, new_funcs


def _is_euler_system(As, t):
    return all(_matrix_is_constant((A*t**i).applyfunc(ratsimp), t) for i, A in enumerate(As))


def _classify_linear_system(eqs, funcs, t, is_canon=False):
    r"""
    Returns a dictionary with details of the eqs if the system passed is linear
    and can be classified by this function else returns None

    Explanation
    ===========

    This function takes the eqs, converts it into a form Ax = b where x is a vector of terms
    containing dependent variables and their derivatives till their maximum order. If it is
    possible to convert eqs into Ax = b, then all the equations in eqs are linear otherwise
    they are non-linear.

    To check if the equations are constant coefficient, we need to check if all the terms in
    A obtained above are constant or not.

    To check if the equations are homogeneous or not, we need to check if b is a zero matrix
    or not.

    Parameters
    ==========

    eqs: List
        List of ODEs
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the equations in eqs
    is_canon: Boolean
        If True, then this function will not try to get the
        system in canonical form. Default value is False

    Returns
    =======

    match = {
        'no_of_equation': len(eqs),
        'eq': eqs,
        'func': funcs,
        'order': order,
        'is_linear': is_linear,
        'is_constant': is_constant,
        'is_homogeneous': is_homogeneous,
    }

    Dict or list of Dicts or None
        Dict with values for keys:
            1. no_of_equation: Number of equations
            2. eq: The set of equations
            3. func: List of dependent variables
            4. order: A dictionary that gives the order of the
                      dependent variable in eqs
            5. is_linear: Boolean value indicating if the set of
                          equations are linear or not.
            6. is_constant: Boolean value indicating if the set of
                          equations have constant coefficients or not.
            7. is_homogeneous: Boolean value indicating if the set of
                          equations are homogeneous or not.
            8. commutative_antiderivative: Antiderivative of the coefficient
                          matrix if the coefficient matrix is non-constant
                          and commutative with its antiderivative. This key
                          may or may not exist.
            9. is_general: Boolean value indicating if the system of ODEs is
                           solvable using one of the general case solvers or not.
            10. rhs: rhs of the non-homogeneous system of ODEs in Matrix form. This
                     key may or may not exist.
            11. is_higher_order: True if the system passed has an order greater than 1.
                                 This key may or may not exist.
            12. is_second_order: True if the system passed is a second order ODE. This
                                 key may or may not exist.
        This Dict is the answer returned if the eqs are linear and constant
        coefficient. Otherwise, None is returned.

    """

    # Error for i == 0 can be added but isn't for now

    # Check for len(funcs) == len(eqs)
    if len(funcs) != len(eqs):
        raise ValueError("Number of functions given is not equal to the number of equations %s" % funcs)

    # ValueError when functions have more than one arguments
    for func in funcs:
        if len(func.args) != 1:
            raise ValueError("dsolve() and classify_sysode() work with "
            "functions of one variable only, not %s" % func)

    # Getting the func_dict and order using the helper
    # function
    order = _get_func_order(eqs, funcs)
    system_order = max(order[func] for func in funcs)
    is_higher_order = system_order > 1
    is_second_order = system_order == 2 and all(order[func] == 2 for func in funcs)

    # Not adding the check if the len(func.args) for
    # every func in funcs is 1

    # Linearity check
    try:

        canon_eqs = canonical_odes(eqs, funcs, t) if not is_canon else [eqs]
        if len(canon_eqs) == 1:
            As, b = linear_ode_to_matrix(canon_eqs[0], funcs, t, system_order)
        else:

            match = {
                'is_implicit': True,
                'canon_eqs': canon_eqs
            }

            return match

    # When the system of ODEs is non-linear, an ODENonlinearError is raised.
    # This function catches the error and None is returned.
    except ODENonlinearError:
        return None

    is_linear = True

    # Homogeneous check
    is_homogeneous = True if b.is_zero_matrix else False

    # Is general key is used to identify if the system of ODEs can be solved by
    # one of the general case solvers or not.
    match = {
        'no_of_equation': len(eqs),
        'eq': eqs,
        'func': funcs,
        'order': order,
        'is_linear': is_linear,
        'is_homogeneous': is_homogeneous,
        'is_general': True
    }

    if not is_homogeneous:
        match['rhs'] = b

    is_constant = all(_matrix_is_constant(A_, t) for A_ in As)

    # The match['is_linear'] check will be added in the future when this
    # function becomes ready to deal with non-linear systems of ODEs

    if not is_higher_order:
        A = As[1]
        match['func_coeff'] = A

        # Constant coefficient check
        is_constant = _matrix_is_constant(A, t)
        match['is_constant'] = is_constant

        try:
            system_info = linodesolve_type(A, t, b=b)
        except NotImplementedError:
            return None

        match.update(system_info)
        antiderivative = match.pop("antiderivative")

        if not is_constant:
            match['commutative_antiderivative'] = antiderivative

        return match
    else:
        match['type_of_equation'] = "type0"

        if is_second_order:
            A1, A0 = As[1:]

            match_second_order = _match_second_order_type(A1, A0, t)
            match.update(match_second_order)

            match['is_second_order'] = True

        # If system is constant, then no need to check if its in euler
        # form or not. It will be easier and faster to directly proceed
        # to solve it.
        if match['type_of_equation'] == "type0" and not is_constant:
            is_euler = _is_euler_system(As, t)
            if is_euler:
                t_ = Symbol('{}_'.format(t))
                match.update({'is_transformed': True, 'type_of_equation': 'type1',
                              't_': t_})
            else:
                is_jordan = lambda M: M == Matrix.jordan_block(M.shape[0], M[0, 0])
                terms = _factor_matrix(As[-1], t)
                if all(A.is_zero_matrix for A in As[1:-1]) and terms is not None and not is_jordan(terms[1]):
                    P, J = terms[1].jordan_form()
                    match.update({'type_of_equation': 'type2', 'J': J,
                                  'f(t)': terms[0], 'P': P, 'is_transformed': True})

            if match['type_of_equation'] != 'type0' and is_second_order:
                match.pop('is_second_order', None)

        match['is_higher_order'] = is_higher_order

        return match

def _preprocess_eqs(eqs):
    processed_eqs = []
    for eq in eqs:
        processed_eqs.append(eq if isinstance(eq, Equality) else Eq(eq, 0))

    return processed_eqs


def _eqs2dict(eqs, funcs):
    eqsorig = {}
    eqsmap = {}
    funcset = set(funcs)
    for eq in eqs:
        f1, = eq.lhs.atoms(AppliedUndef)
        f2s = (eq.rhs.atoms(AppliedUndef) - {f1}) & funcset
        eqsmap[f1] = f2s
        eqsorig[f1] = eq
    return eqsmap, eqsorig


def _dict2graph(d):
    nodes = list(d)
    edges = [(f1, f2) for f1, f2s in d.items() for f2 in f2s]
    G = (nodes, edges)
    return G


def _is_type1(scc, t):
    eqs, funcs = scc

    try:
        (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, 1)
    except (ODENonlinearError, ODEOrderError):
        return False

    if _matrix_is_constant(A0, t) and b.is_zero_matrix:
        return True

    return False


def _combine_type1_subsystems(subsystem, funcs, t):
    indices = [i for i, sys in enumerate(zip(subsystem, funcs)) if _is_type1(sys, t)]
    remove = set()
    for ip, i in enumerate(indices):
        for j in indices[ip+1:]:
            if any(eq2.has(funcs[i]) for eq2 in subsystem[j]):
                subsystem[j] = subsystem[i] + subsystem[j]
                remove.add(i)
    subsystem = [sys for i, sys in enumerate(subsystem) if i not in remove]
    return subsystem


def _component_division(eqs, funcs, t):

    # Assuming that each eq in eqs is in canonical form,
    # that is, [f(x).diff(x) = .., g(x).diff(x) = .., etc]
    # and that the system passed is in its first order
    eqsmap, eqsorig = _eqs2dict(eqs, funcs)

    subsystems = []
    for cc in connected_components(_dict2graph(eqsmap)):
        eqsmap_c = {f: eqsmap[f] for f in cc}
        sccs = strongly_connected_components(_dict2graph(eqsmap_c))
        subsystem = [[eqsorig[f] for f in scc] for scc in sccs]
        subsystem = _combine_type1_subsystems(subsystem, sccs, t)
        subsystems.append(subsystem)

    return subsystems


# Returns: List of equations
def _linear_ode_solver(match):
    t = match['t']
    funcs = match['func']

    rhs = match.get('rhs', None)
    tau = match.get('tau', None)
    t = match['t_'] if 't_' in match else t
    A = match['func_coeff']

    # Note: To make B None when the matrix has constant
    # coefficient
    B = match.get('commutative_antiderivative', None)
    type = match['type_of_equation']

    sol_vector = linodesolve(A, t, b=rhs, B=B,
                             type=type, tau=tau)

    sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]

    return sol


def _select_equations(eqs, funcs, key=lambda x: x):
    eq_dict = {e.lhs: e.rhs for e in eqs}
    return [Eq(f, eq_dict[key(f)]) for f in funcs]


def _higher_order_ode_solver(match):
    eqs = match["eq"]
    funcs = match["func"]
    t = match["t"]
    sysorder = match['order']
    type = match.get('type_of_equation', "type0")

    is_second_order = match.get('is_second_order', False)
    is_transformed = match.get('is_transformed', False)
    is_euler = is_transformed and type == "type1"
    is_higher_order_type2 = is_transformed and type == "type2" and 'P' in match

    if is_second_order:
        new_eqs, new_funcs = _second_order_to_first_order(eqs, funcs, t,
                                                          A1=match.get("A1", None), A0=match.get("A0", None),
                                                          b=match.get("rhs", None), type=type,
                                                          t_=match.get("t_", None))
    else:
        new_eqs, new_funcs = _higher_order_to_first_order(eqs, sysorder, t, funcs=funcs,
                                                          type=type, J=match.get('J', None),
                                                          f_t=match.get('f(t)', None),
                                                          P=match.get('P', None), b=match.get('rhs', None))

    if is_transformed:
        t = match.get('t_', t)

    if not is_higher_order_type2:
        new_eqs = _select_equations(new_eqs, [f.diff(t) for f in new_funcs])

    sol = None

    # NotImplementedError may be raised when the system may be actually
    # solvable if it can be just divided into sub-systems
    try:
        if not is_higher_order_type2:
            sol = _strong_component_solver(new_eqs, new_funcs, t)
    except NotImplementedError:
        sol = None

    # Dividing the system only when it becomes essential
    if sol is None:
        try:
            sol = _component_solver(new_eqs, new_funcs, t)
        except NotImplementedError:
            sol = None

    if sol is None:
        return sol

    is_second_order_type2 = is_second_order and type == "type2"

    underscores = '__' if is_transformed else '_'

    sol = _select_equations(sol, funcs,
                            key=lambda x: Function(Dummy('{}{}0'.format(x.func.__name__, underscores)))(t))

    if match.get("is_transformed", False):
        if is_second_order_type2:
            g_t = match["g(t)"]
            tau = match["tau"]
            sol = [Eq(s.lhs, s.rhs.subs(t, tau) * g_t) for s in sol]
        elif is_euler:
            t = match['t']
            tau = match['t_']
            sol = [s.subs(tau, log(t)) for s in sol]
        elif is_higher_order_type2:
            P = match['P']
            sol_vector = P * Matrix([s.rhs for s in sol])
            sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]

    return sol


# Returns: List of equations or None
# If None is returned by this solver, then the system
# of ODEs cannot be solved directly by dsolve_system.
def _strong_component_solver(eqs, funcs, t):
    from sympy.solvers.ode.ode import dsolve, constant_renumber

    match = _classify_linear_system(eqs, funcs, t, is_canon=True)
    sol = None

    # Assuming that we can't get an implicit system
    # since we are already canonical equations from
    # dsolve_system
    if match:
        match['t'] = t

        if match.get('is_higher_order', False):
            sol = _higher_order_ode_solver(match)

        elif match.get('is_linear', False):
            sol = _linear_ode_solver(match)

        # Note: For now, only linear systems are handled by this function
        # hence, the match condition is added. This can be removed later.
        if sol is None and len(eqs) == 1:
            sol = dsolve(eqs[0], func=funcs[0])
            variables = Tuple(eqs[0]).free_symbols
            new_constants = [Dummy() for _ in range(ode_order(eqs[0], funcs[0]))]
            sol = constant_renumber(sol, variables=variables, newconstants=new_constants)
            sol = [sol]

        # To add non-linear case here in future

    return sol


def _get_funcs_from_canon(eqs):
    return [eq.lhs.args[0] for eq in eqs]


# Returns: List of Equations(a solution)
def _weak_component_solver(wcc, t):

    # We will divide the systems into sccs
    # only when the wcc cannot be solved as
    # a whole
    eqs = []
    for scc in wcc:
        eqs += scc
    funcs = _get_funcs_from_canon(eqs)

    sol = _strong_component_solver(eqs, funcs, t)
    if sol:
        return sol

    sol = []

    for scc in wcc:
        eqs = scc
        funcs = _get_funcs_from_canon(eqs)

        # Substituting solutions for the dependent
        # variables solved in previous SCC, if any solved.
        comp_eqs = [eq.subs({s.lhs: s.rhs for s in sol}) for eq in eqs]
        scc_sol = _strong_component_solver(comp_eqs, funcs, t)

        if scc_sol is None:
            raise NotImplementedError(filldedent('''
                The system of ODEs passed cannot be solved by dsolve_system.
            '''))

        # scc_sol: List of equations
        # scc_sol is a solution
        sol += scc_sol

    return sol


# Returns: List of Equations(a solution)
def _component_solver(eqs, funcs, t):
    components = _component_division(eqs, funcs, t)
    sol = []

    for wcc in components:

        # wcc_sol: List of Equations
        sol += _weak_component_solver(wcc, t)

    # sol: List of Equations
    return sol


def _second_order_to_first_order(eqs, funcs, t, type="auto", A1=None,
                                 A0=None, b=None, t_=None):
    r"""
    Expects the system to be in second order and in canonical form

    Explanation
    ===========

    Reduces a second order system into a first order one depending on the type of second
    order system.
    1. "type0": If this is passed, then the system will be reduced to first order by
                introducing dummy variables.
    2. "type1": If this is passed, then a particular substitution will be used to reduce the
                the system into first order.
    3. "type2": If this is passed, then the system will be transformed with new dependent
                variables and independent variables. This transformation is a part of solving
                the corresponding system of ODEs.

    `A1` and `A0` are the coefficient matrices from the system and it is assumed that the
    second order system has the form given below:

    .. math::
        A2 * X'' = A1 * X' + A0 * X + b

    Here, $A2$ is the coefficient matrix for the vector $X''$ and $b$ is the non-homogeneous
    term.

    Default value for `b` is None but if `A1` and `A0` are passed and `b` is not passed, then the
    system will be assumed homogeneous.

    """
    is_a1 = A1 is None
    is_a0 = A0 is None

    if (type == "type1" and is_a1) or (type == "type2" and is_a0)\
        or (type == "auto" and (is_a1 or is_a0)):
        (A2, A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, 2)

        if not A2.is_Identity:
            raise ValueError(filldedent('''
                The system must be in its canonical form.
            '''))

    if type == "auto":
        match = _match_second_order_type(A1, A0, t)
        type = match["type_of_equation"]
        A1 = match.get("A1", None)
        A0 = match.get("A0", None)

    sys_order = dict.fromkeys(funcs, 2)

    if type == "type1":
        if b is None:
            b = zeros(len(eqs))
        eqs = _second_order_subs_type1(A1, b, funcs, t)
        sys_order = dict.fromkeys(funcs, 1)

    if type == "type2":
        if t_ is None:
            t_ = Symbol("{}_".format(t))
        t = t_
        eqs, funcs = _second_order_subs_type2(A0, funcs, t_)
        sys_order = dict.fromkeys(funcs, 2)

    return _higher_order_to_first_order(eqs, sys_order, t, funcs=funcs)


def _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, b=None, P=None):

    # Note: To add a test for this ValueError
    if J is None or f_t is None or not _matrix_is_constant(J, t):
        raise ValueError(filldedent('''
            Correctly input for args 'A' and 'f_t' for Linear, Higher Order,
            Type 2
        '''))

    if P is None and b is not None and not b.is_zero_matrix:
        raise ValueError(filldedent('''
            Provide the keyword 'P' for matrix P in A = P * J * P-1.
        '''))

    new_funcs = Matrix([Function(Dummy('{}__0'.format(f.func.__name__)))(t) for f in funcs])
    new_eqs = new_funcs.diff(t, max_order) - f_t * J * new_funcs

    if b is not None and not b.is_zero_matrix:
        new_eqs -= P.inv() * b

    new_eqs = canonical_odes(new_eqs, new_funcs, t)[0]

    return new_eqs, new_funcs


def _higher_order_to_first_order(eqs, sys_order, t, funcs=None, type="type0", **kwargs):
    if funcs is None:
        funcs = sys_order.keys()

    # Standard Cauchy Euler system
    if type == "type1":
        t_ = Symbol('{}_'.format(t))
        new_funcs = [Function(Dummy('{}_'.format(f.func.__name__)))(t_) for f in funcs]
        max_order = max(sys_order[func] for func in funcs)
        subs_dict = dict(zip(funcs, new_funcs))
        subs_dict[t] = exp(t_)

        free_function = Function(Dummy())

        def _get_coeffs_from_subs_expression(expr):
            if isinstance(expr, Subs):
                free_symbol = expr.args[1][0]
                term = expr.args[0]
                return {ode_order(term, free_symbol): 1}

            if isinstance(expr, Mul):
                coeff = expr.args[0]
                order = list(_get_coeffs_from_subs_expression(expr.args[1]).keys())[0]
                return {order: coeff}

            if isinstance(expr, Add):
                coeffs = {}
                for arg in expr.args:

                    if isinstance(arg, Mul):
                        coeffs.update(_get_coeffs_from_subs_expression(arg))

                    else:
                        order = list(_get_coeffs_from_subs_expression(arg).keys())[0]
                        coeffs[order] = 1

                return coeffs

        for o in range(1, max_order + 1):
            expr = free_function(log(t_)).diff(t_, o)*t_**o
            coeff_dict = _get_coeffs_from_subs_expression(expr)
            coeffs = [coeff_dict[order] if order in coeff_dict else 0 for order in range(o + 1)]
            expr_to_subs = sum(free_function(t_).diff(t_, i) * c for i, c in
                        enumerate(coeffs)) / t**o
            subs_dict.update({f.diff(t, o): expr_to_subs.subs(free_function(t_), nf)
                              for f, nf in zip(funcs, new_funcs)})

        new_eqs = [eq.subs(subs_dict) for eq in eqs]
        new_sys_order = {nf: sys_order[f] for f, nf in zip(funcs, new_funcs)}

        new_eqs = canonical_odes(new_eqs, new_funcs, t_)[0]

        return _higher_order_to_first_order(new_eqs, new_sys_order, t_, funcs=new_funcs)

    # Systems of the form: X(n)(t) = f(t)*A*X + b
    # where X(n)(t) is the nth derivative of the vector of dependent variables
    # with respect to the independent variable and A is a constant matrix.
    if type == "type2":
        J = kwargs.get('J', None)
        f_t = kwargs.get('f_t', None)
        b = kwargs.get('b', None)
        P = kwargs.get('P', None)
        max_order = max(sys_order[func] for func in funcs)

        return _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, P=P, b=b)

        # Note: To be changed to this after doit option is disabled for default cases
        # new_sysorder = _get_func_order(new_eqs, new_funcs)
        #
        # return _higher_order_to_first_order(new_eqs, new_sysorder, t, funcs=new_funcs)

    new_funcs = []

    for prev_func in funcs:
        func_name = prev_func.func.__name__
        func = Function(Dummy('{}_0'.format(func_name)))(t)
        new_funcs.append(func)
        subs_dict = {prev_func: func}
        new_eqs = []

        for i in range(1, sys_order[prev_func]):
            new_func = Function(Dummy('{}_{}'.format(func_name, i)))(t)
            subs_dict[prev_func.diff(t, i)] = new_func
            new_funcs.append(new_func)

            prev_f = subs_dict[prev_func.diff(t, i-1)]
            new_eq = Eq(prev_f.diff(t), new_func)
            new_eqs.append(new_eq)

        eqs = [eq.subs(subs_dict) for eq in eqs] + new_eqs

    return eqs, new_funcs


def dsolve_system(eqs, funcs=None, t=None, ics=None, doit=False, simplify=True):
    r"""
    Solves any(supported) system of Ordinary Differential Equations

    Explanation
    ===========

    This function takes a system of ODEs as an input, determines if the
    it is solvable by this function, and returns the solution if found any.

    This function can handle:
    1. Linear, First Order, Constant coefficient homogeneous system of ODEs
    2. Linear, First Order, Constant coefficient non-homogeneous system of ODEs
    3. Linear, First Order, non-constant coefficient homogeneous system of ODEs
    4. Linear, First Order, non-constant coefficient non-homogeneous system of ODEs
    5. Any implicit system which can be divided into system of ODEs which is of the above 4 forms
    6. Any higher order linear system of ODEs that can be reduced to one of the 5 forms of systems described above.

    The types of systems described above are not limited by the number of equations, i.e. this
    function can solve the above types irrespective of the number of equations in the system passed.
    But, the bigger the system, the more time it will take to solve the system.

    This function returns a list of solutions. Each solution is a list of equations where LHS is
    the dependent variable and RHS is an expression in terms of the independent variable.

    Among the non constant coefficient types, not all the systems are solvable by this function. Only
    those which have either a coefficient matrix with a commutative antiderivative or those systems which
    may be divided further so that the divided systems may have coefficient matrix with commutative antiderivative.

    Parameters
    ==========

    eqs : List
        system of ODEs to be solved
    funcs : List or None
        List of dependent variables that make up the system of ODEs
    t : Symbol or None
        Independent variable in the system of ODEs
    ics : Dict or None
        Set of initial boundary/conditions for the system of ODEs
    doit : Boolean
        Evaluate the solutions if True. Default value is True. Can be
        set to false if the integral evaluation takes too much time and/or
        is not required.
    simplify: Boolean
        Simplify the solutions for the systems. Default value is True.
        Can be set to false if simplification takes too much time and/or
        is not required.

    Examples
    ========

    >>> from sympy import symbols, Eq, Function
    >>> from sympy.solvers.ode.systems import dsolve_system
    >>> f, g = symbols("f g", cls=Function)
    >>> x = symbols("x")

    >>> eqs = [Eq(f(x).diff(x), g(x)), Eq(g(x).diff(x), f(x))]
    >>> dsolve_system(eqs)
    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]

    You can also pass the initial conditions for the system of ODEs:

    >>> dsolve_system(eqs, ics={f(0): 1, g(0): 0})
    [[Eq(f(x), exp(x)/2 + exp(-x)/2), Eq(g(x), exp(x)/2 - exp(-x)/2)]]

    Optionally, you can pass the dependent variables and the independent
    variable for which the system is to be solved:

    >>> funcs = [f(x), g(x)]
    >>> dsolve_system(eqs, funcs=funcs, t=x)
    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]

    Lets look at an implicit system of ODEs:

    >>> eqs = [Eq(f(x).diff(x)**2, g(x)**2), Eq(g(x).diff(x), g(x))]
    >>> dsolve_system(eqs)
    [[Eq(f(x), C1 - C2*exp(x)), Eq(g(x), C2*exp(x))], [Eq(f(x), C1 + C2*exp(x)), Eq(g(x), C2*exp(x))]]

    Returns
    =======

    List of List of Equations

    Raises
    ======

    NotImplementedError
        When the system of ODEs is not solvable by this function.
    ValueError
        When the parameters passed are not in the required form.

    """
    from sympy.solvers.ode.ode import solve_ics, _extract_funcs, constant_renumber

    if not iterable(eqs):
        raise ValueError(filldedent('''
            List of equations should be passed. The input is not valid.
        '''))

    eqs = _preprocess_eqs(eqs)

    if funcs is not None and not isinstance(funcs, list):
        raise ValueError(filldedent('''
            Input to the funcs should be a list of functions.
        '''))

    if funcs is None:
        funcs = _extract_funcs(eqs)

    if any(len(func.args) != 1 for func in funcs):
        raise ValueError(filldedent('''
            dsolve_system can solve a system of ODEs with only one independent
            variable.
        '''))

    if len(eqs) != len(funcs):
        raise ValueError(filldedent('''
            Number of equations and number of functions do not match
        '''))

    if t is not None and not isinstance(t, Symbol):
        raise ValueError(filldedent('''
            The independent variable must be of type Symbol
        '''))

    if t is None:
        t = list(list(eqs[0].atoms(Derivative))[0].atoms(Symbol))[0]

    sols = []
    canon_eqs = canonical_odes(eqs, funcs, t)

    for canon_eq in canon_eqs:
        try:
            sol = _strong_component_solver(canon_eq, funcs, t)
        except NotImplementedError:
            sol = None

        if sol is None:
            sol = _component_solver(canon_eq, funcs, t)

        sols.append(sol)

    if sols:
        final_sols = []
        variables = Tuple(*eqs).free_symbols

        for sol in sols:

            sol = _select_equations(sol, funcs)
            sol = constant_renumber(sol, variables=variables)

            if ics:
                constants = Tuple(*sol).free_symbols - variables
                solved_constants = solve_ics(sol, funcs, constants, ics)
                sol = [s.subs(solved_constants) for s in sol]

            if simplify:
                constants = Tuple(*sol).free_symbols - variables
                sol = simpsol(sol, [t], constants, doit=doit)

            final_sols.append(sol)

        sols = final_sols

    return sols

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\ring_series.py ===
"""Power series evaluation and manipulation using sparse Polynomials

Implementing a new function
---------------------------

There are a few things to be kept in mind when adding a new function here::

    - The implementation should work on all possible input domains/rings.
      Special cases include the ``EX`` ring and a constant term in the series
      to be expanded. There can be two types of constant terms in the series:

        + A constant value or symbol.
        + A term of a multivariate series not involving the generator, with
          respect to which the series is to expanded.

      Strictly speaking, a generator of a ring should not be considered a
      constant. However, for series expansion both the cases need similar
      treatment (as the user does not care about inner details), i.e, use an
      addition formula to separate the constant part and the variable part (see
      rs_sin for reference).

    - All the algorithms used here are primarily designed to work for Taylor
      series (number of iterations in the algo equals the required order).
      Hence, it becomes tricky to get the series of the right order if a
      Puiseux series is input. Use rs_puiseux? in your function if your
      algorithm is not designed to handle fractional powers.

Extending rs_series
-------------------

To make a function work with rs_series you need to do two things::

    - Many sure it works with a constant term (as explained above).
    - If the series contains constant terms, you might need to extend its ring.
      You do so by adding the new terms to the rings as generators.
      ``PolyRing.compose`` and ``PolyRing.add_gens`` are two functions that do
      so and need to be called every time you expand a series containing a
      constant term.

Look at rs_sin and rs_series for further reference.

"""

from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.puiseux import PuiseuxPoly
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
                                   monomial_ldiv)
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational
from sympy.core.intfunc import igcd
from sympy.functions import (sin, cos, tan, atan, exp, atanh, asinh, tanh, log,
                             ceiling, sinh, cosh)
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math


def _invert_monoms(p1):
    """
    Compute ``x**n * p1(1/x)`` for a univariate polynomial ``p1`` in ``x``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _invert_monoms
    >>> R, x = ring('x', ZZ)
    >>> p = x**2 + 2*x + 3
    >>> _invert_monoms(p)
    3*x**2 + 2*x + 1

    See Also
    ========

    sympy.polys.densebasic.dup_reverse
    """
    terms = list(p1.items())
    terms.sort()
    deg = p1.degree()
    R = p1.ring
    p = R.zero
    cv = p1.listcoeffs()
    mv = p1.listmonoms()
    for mvi, cvi in zip(mv, cv):
        p[(deg - mvi[0],)] = cvi
    return p

def _giant_steps(target):
    """Return a list of precision steps for the Newton's method"""
    # We use ceil here because giant_steps cannot handle flint.fmpq
    res = giant_steps(2, math.ceil(target))
    if res[0] != 2:
        res = [2] + res
    return res

def rs_trunc(p1, x, prec):
    """
    Truncate the series in the ``x`` variable with precision ``prec``,
    that is, modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_trunc
    >>> R, x = ring('x', QQ)
    >>> p = x**10 + x**5 + x + 1
    >>> rs_trunc(p, x, 12)
    x**10 + x**5 + x + 1
    >>> rs_trunc(p, x, 10)
    x**5 + x + 1
    """
    R = p1.ring
    p = {}
    i = R.gens.index(x)
    for exp1 in p1:
        if exp1[i] >= prec:
            continue
        p[exp1] = p1[exp1]
    return R(p)

def rs_is_puiseux(p, x):
    """
    Test if ``p`` is Puiseux series in ``x``.

    Raise an exception if it has a negative power in ``x``.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_is_puiseux
    >>> R, x = puiseux_ring('x', QQ)
    >>> p = x**QQ(2,5) + x**QQ(2,3) + x
    >>> rs_is_puiseux(p, x)
    True
    """
    index = p.ring.gens.index(x)
    for k in p.itermonoms():
        if k[index] != int(k[index]):
            return True
        if k[index] < 0:
            raise ValueError('The series is not regular in %s' % x)
    return False

def rs_puiseux(f, p, x, prec):
    """
    Return the puiseux series for `f(p, x, prec)`.

    To be used when function ``f`` is implemented only for regular series.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_puiseux, rs_exp
    >>> R, x = puiseux_ring('x', QQ)
    >>> p = x**QQ(2,5) + x**QQ(2,3) + x
    >>> rs_puiseux(rs_exp,p, x, 1)
    1 + x**(2/5) + x**(2/3) + 1/2*x**(4/5)
    """
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = int(n*den // igcd(n, den))
        elif power != int(power):
            den = power.denominator
            n = int(n*den // igcd(n, den))
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, x, prec*n)
        n1 = QQ(1, n)
        if isinstance(r, tuple):
            r = tuple([pow_xin(rx, index, n1) for rx in r])
        else:
            r = pow_xin(r, index, n1)
    else:
        r = f(p, x, prec)
    return r

def rs_puiseux2(f, p, q, x, prec):
    """
    Return the puiseux series for `f(p, q, x, prec)`.

    To be used when function ``f`` is implemented only for regular series.
    """
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = n*den // igcd(n, den)
        elif power != int(power):
            den = power.denominator
            n = n*den // igcd(n, den)
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, q, x, prec*n)
        n1 = QQ(1, n)
        r = pow_xin(r, index, n1)
    else:
        r = f(p, q, x, prec)
    return r

def rs_mul(p1, p2, x, prec):
    """
    Return the product of the given two series, modulo ``O(x**prec)``.

    ``x`` is the series variable or its position in the generators.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_mul
    >>> R, x = ring('x', QQ)
    >>> p1 = x**2 + 2*x + 1
    >>> p2 = x + 1
    >>> rs_mul(p1, p2, x, 3)
    3*x**2 + 3*x + 1
    """
    R = p1.ring
    p = {}
    if R.__class__ != p2.ring.__class__ or R != p2.ring:
        raise ValueError('p1 and p2 must have the same ring')
    iv = R.gens.index(x)
    if not isinstance(p2, (PolyElement, PuiseuxPoly)):
        raise ValueError('p2 must be a polynomial')
    if R == p2.ring:
        get = p.get
        items2 = p2.terms()
        items2.sort(key=lambda e: e[0][iv])
        if R.ngens == 1:
            for exp1, v1 in p1.iterterms():
                for exp2, v2 in items2:
                    exp = exp1[0] + exp2[0]
                    if exp < prec:
                        exp = (exp, )
                        p[exp] = get(exp, 0) + v1*v2
                    else:
                        break
        else:
            monomial_mul = R.monomial_mul
            for exp1, v1 in p1.iterterms():
                for exp2, v2 in items2:
                    if exp1[iv] + exp2[iv] < prec:
                        exp = monomial_mul(exp1, exp2)
                        p[exp] = get(exp, 0) + v1*v2
                    else:
                        break

    return R(p)

def rs_square(p1, x, prec):
    """
    Square the series modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_square
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + 2*x + 1
    >>> rs_square(p, x, 3)
    6*x**2 + 4*x + 1
    """
    R = p1.ring
    p = {}
    iv = R.gens.index(x)
    get = p.get
    items = p1.terms()
    items.sort(key=lambda e: e[0][iv])
    monomial_mul = R.monomial_mul
    for i in range(len(items)):
        exp1, v1 = items[i]
        for j in range(i):
            exp2, v2 = items[j]
            if exp1[iv] + exp2[iv] < prec:
                exp = monomial_mul(exp1, exp2)
                p[exp] = get(exp, 0) + v1*v2
            else:
                break
    p = {m: 2*v for m, v in p.items()}
    get = p.get
    for expv, v in p1.iterterms():
        if 2*expv[iv] < prec:
            e2 = monomial_mul(expv, expv)
            p[e2] = get(e2, 0) + v**2
    return R(p)

def rs_pow(p1, n, x, prec):
    """
    Return ``p1**n`` modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_pow
    >>> R, x = ring('x', QQ)
    >>> p = x + 1
    >>> rs_pow(p, 4, x, 3)
    6*x**2 + 4*x + 1
    """
    R = p1.ring
    if isinstance(n, Rational):
        np = int(n.p)
        nq = int(n.q)
        if nq != 1:
            res = rs_nth_root(p1, nq, x, prec)
            if np != 1:
                res = rs_pow(res, np, x, prec)
        else:
            res = rs_pow(p1, np, x, prec)
        return res

    n = as_int(n)
    if n == 0:
        if p1:
            return R(1)
        else:
            raise ValueError('0**0 is undefined')
    if n < 0:
        p1 = rs_pow(p1, -n, x, prec)
        return rs_series_inversion(p1, x, prec)
    if n == 1:
        return rs_trunc(p1, x, prec)
    if n == 2:
        return rs_square(p1, x, prec)
    if n == 3:
        p2 = rs_square(p1, x, prec)
        return rs_mul(p1, p2, x, prec)
    p = R(1)
    while 1:
        if n & 1:
            p = rs_mul(p1, p, x, prec)
            n -= 1
            if not n:
                break
        p1 = rs_square(p1, x, prec)
        n = n // 2
    return p

def rs_subs(p, rules, x, prec):
    """
    Substitution with truncation according to the mapping in ``rules``.

    Return a series with precision ``prec`` in the generator ``x``

    Note that substitutions are not done one after the other

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_subs
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x**2 + y**2
    >>> rs_subs(p, {x: x+ y, y: x+ 2*y}, x, 3)
    2*x**2 + 6*x*y + 5*y**2
    >>> (x + y)**2 + (x + 2*y)**2
    2*x**2 + 6*x*y + 5*y**2

    which differs from

    >>> rs_subs(rs_subs(p, {x: x+ y}, x, 3), {y: x+ 2*y}, x, 3)
    5*x**2 + 12*x*y + 8*y**2

    Parameters
    ----------
    p : :class:`~.PolyElement` Input series.
    rules : ``dict`` with substitution mappings.
    x : :class:`~.PolyElement` in which the series truncation is to be done.
    prec : :class:`~.Integer` order of the series after truncation.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_subs
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_subs(x**2+y**2, {y: (x+y)**2}, x, 3)
     6*x**2*y**2 + x**2 + 4*x*y**3 + y**4
    """
    R = p.ring
    ngens = R.ngens
    d = R(0)
    for i in range(ngens):
        d[(i, 1)] = R.gens[i]
    for var in rules:
        d[(R.index(var), 1)] = rules[var]
    p1 = R(0)
    p_keys = sorted(p.keys())
    for expv in p_keys:
        p2 = R(1)
        for i in range(ngens):
            power = expv[i]
            if power == 0:
                continue
            if (i, power) not in d:
                q, r = divmod(power, 2)
                if r == 0 and (i, q) in d:
                    d[(i, power)] = rs_square(d[(i, q)], x, prec)
                elif (i, power - 1) in d:
                    d[(i, power)] = rs_mul(d[(i, power - 1)], d[(i, 1)],
                                           x, prec)
                else:
                    d[(i, power)] = rs_pow(d[(i, 1)], power, x, prec)
            p2 = rs_mul(p2, d[(i, power)], x, prec)
        p1 += p2*p[expv]
    return p1

def _has_constant_term(p, x):
    """
    Check if ``p`` has a constant term in ``x``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _has_constant_term
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + x + 1
    >>> _has_constant_term(p, x)
    True
    """
    R = p.ring
    iv = R.gens.index(x)
    zm = R.zero_monom
    a = [0]*R.ngens
    a[iv] = 1
    miv = tuple(a)
    return any(monomial_min(expv, miv) == zm for expv in p)

def _get_constant_term(p, x):
    """Return constant term in p with respect to x

    Note that it is not simply `p[R.zero_monom]` as there might be multiple
    generators in the ring R. We want the `x`-free term which can contain other
    generators.
    """
    R = p.ring
    i = R.gens.index(x)
    zm = R.zero_monom
    a = [0]*R.ngens
    a[i] = 1
    miv = tuple(a)
    c = 0
    for expv in p:
        if monomial_min(expv, miv) == zm:
            c += R({expv: p[expv]})
    return c

def _check_series_var(p, x, name):
    index = p.ring.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m < 0:
        raise PoleError("Asymptotic expansion of %s around [oo] not "
                        "implemented." % name)
    return index, m

def _series_inversion1(p, x, prec):
    """
    Univariate series inversion ``1/p`` modulo ``O(x**prec)``.

    The Newton method is used.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _series_inversion1
    >>> R, x = ring('x', QQ)
    >>> p = x + 1
    >>> _series_inversion1(p, x, 4)
    -x**3 + x**2 - x + 1
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(_series_inversion1, p, x, prec)
    R = p.ring
    zm = R.zero_monom
    c = p[zm]

    # giant_steps does not seem to work with PythonRational numbers with 1 as
    # denominator. This makes sure such a number is converted to integer.
    if prec == int(prec):
        prec = int(prec)

    if zm not in p:
        raise ValueError("No constant term in series")
    if _has_constant_term(p - c, x):
        raise ValueError("p cannot contain a constant term depending on "
                         "parameters")
    if not R.domain.is_unit(c):
        raise ValueError(f"Constant term {c} must be a unit in {R.domain}")

    one = R(1)
    if R.domain is EX:
        one = 1
    if c != one:
        p1 = R(1)/c
    else:
        p1 = R(1)
    for precx in _giant_steps(prec):
        t = 1 - rs_mul(p1, p, x, precx)
        p1 = p1 + rs_mul(p1, t, x, precx)
    return p1

def rs_series_inversion(p, x, prec):
    """
    Multivariate series inversion ``1/p`` modulo ``O(x**prec)``.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_inversion
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_series_inversion(1 + x*y**2, x, 4)
    -x**3*y**6 + x**2*y**4 - x*y**2 + 1
    >>> rs_series_inversion(1 + x*y**2, y, 4)
    -x*y**2 + 1
    >>> rs_series_inversion(x + x**2, x, 4)
    x**3 - x**2 + x - 1 + x**(-1)
    """
    R = p.ring
    if p == R.zero:
        raise ZeroDivisionError
    zm = R.zero_monom
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m:
        p = mul_xin(p, index, -m)
        prec = prec + m
    if zm not in p:
        raise NotImplementedError("No constant term in series")

    if _has_constant_term(p - p[zm], x):
        raise NotImplementedError("p - p[0] must not have a constant term in "
                                  "the series variables")
    r = _series_inversion1(p, x, prec)
    if m != 0:
        r = mul_xin(r, index, -m)
    return r

def _coefficient_t(p, t):
    r"""Coefficient of `x_i**j` in p, where ``t`` = (i, j)"""
    i, j = t
    R = p.ring
    expv1 = [0]*R.ngens
    expv1[i] = j
    expv1 = tuple(expv1)
    p1 = R(0)
    for expv in p:
        if expv[i] == j:
            p1[monomial_div(expv, expv1)] = p[expv]
    return p1

def rs_series_reversion(p, x, n, y):
    r"""
    Reversion of a series.

    ``p`` is a series with ``O(x**n)`` of the form $p = ax + f(x)$
    where $a$ is a number different from 0.

    $f(x) = \sum_{k=2}^{n-1} a_kx_k$

    Parameters
    ==========

      a_k : Can depend polynomially on other variables, not indicated.
      x : Variable with name x.
      y : Variable with name y.

    Returns
    =======

    Solve $p = y$, that is, given $ax + f(x) - y = 0$,
    find the solution $x = r(y)$ up to $O(y^n)$.

    Algorithm
    =========

    If $r_i$ is the solution at order $i$, then:
    $ar_i + f(r_i) - y = O\left(y^{i + 1}\right)$

    and if $r_{i + 1}$ is the solution at order $i + 1$, then:
    $ar_{i + 1} + f(r_{i + 1}) - y = O\left(y^{i + 2}\right)$

    We have, $r_{i + 1} = r_i + e$, such that,
    $ae + f(r_i) = O\left(y^{i + 2}\right)$
    or $e = -f(r_i)/a$

    So we use the recursion relation:
    $r_{i + 1} = r_i - f(r_i)/a$
    with the boundary condition: $r_1 = y$

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_reversion, rs_trunc
    >>> R, x, y, a, b = ring('x, y, a, b', QQ)
    >>> p = x - x**2 - 2*b*x**2 + 2*a*b*x**2
    >>> p1 = rs_series_reversion(p, x, 3, y); p1
    -2*y**2*a*b + 2*y**2*b + y**2 + y
    >>> rs_trunc(p.compose(x, p1), y, 3)
    y
    """
    if rs_is_puiseux(p, x):
        raise NotImplementedError
    R = p.ring
    nx = R.gens.index(x)
    y = R(y)
    ny = R.gens.index(y)
    if _has_constant_term(p, x):
        raise ValueError("p must not contain a constant term in the series "
                         "variable")
    a = _coefficient_t(p, (nx, 1))
    zm = R.zero_monom
    assert zm in a and len(a) == 1
    a = a[zm]
    r = y/a
    for i in range(2, n):
        sp = rs_subs(p, {x: r}, y, i + 1)
        sp = _coefficient_t(sp, (ny, i))*y**i
        r -= sp/a
    return r

def rs_series_from_list(p, c, x, prec, concur=1):
    """
    Return a series `sum c[n]*p**n` modulo `O(x**prec)`.

    It reduces the number of multiplications by summing concurrently.

    `ax = [1, p, p**2, .., p**(J - 1)]`
    `s = sum(c[i]*ax[i]` for i in `range(r, (r + 1)*J))*p**((K - 1)*J)`
    with `K >= (n + 1)/J`

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_from_list, rs_trunc
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + x + 1
    >>> c = [1, 2, 3]
    >>> rs_series_from_list(p, c, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> rs_trunc(1 + 2*p + 3*p**2, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> pc = R.from_list(list(reversed(c)))
    >>> rs_trunc(pc.compose(x, p), x, 4)
    6*x**3 + 11*x**2 + 8*x + 6

    See Also
    ========

    sympy.polys.rings.PolyRing.compose

    """
    R = p.ring
    n = len(c)
    if not concur:
        q = R(1)
        s = c[0]*q
        for i in range(1, n):
            q = rs_mul(q, p, x, prec)
            s += c[i]*q
        return s
    J = int(math.sqrt(n) + 1)
    K, r = divmod(n, J)
    if r:
        K += 1
    ax = [R(1)]
    q = R(1)
    if len(p) < 20:
        for i in range(1, J):
            q = rs_mul(q, p, x, prec)
            ax.append(q)
    else:
        for i in range(1, J):
            if i % 2 == 0:
                q = rs_square(ax[i//2], x, prec)
            else:
                q = rs_mul(q, p, x, prec)
            ax.append(q)
    # optimize using rs_square
    pj = rs_mul(ax[-1], p, x, prec)
    b = R(1)
    s = R(0)
    for k in range(K - 1):
        r = J*k
        s1 = c[r]
        for j in range(1, J):
            s1 += c[r + j]*ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
        b = rs_mul(b, pj, x, prec)
        if not b:
            break
    k = K - 1
    r = J*k
    if r < n:
        s1 = c[r]*R(1)
        for j in range(1, J):
            if r + j >= n:
                break
            s1 += c[r + j]*ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
    return s

def rs_diff(p, x):
    """
    Return partial derivative of ``p`` with respect to ``x``.

    Parameters
    ==========

    x : :class:`~.PolyElement` with respect to which ``p`` is differentiated.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_diff
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x**2*y**3
    >>> rs_diff(p, x)
    2*x*y**3 + 1
    """
    R = p.ring
    n = R.gens.index(x)
    p1 = {}
    mn = [0]*R.ngens
    mn[n] = 1
    mn = tuple(mn)
    for expv in p:
        if expv[n]:
            e = monomial_ldiv(expv, mn)
            p1[e] = R.domain_new(p[expv]*expv[n])
    return R(p1)

def rs_integrate(p, x):
    """
    Integrate ``p`` with respect to ``x``.

    Parameters
    ==========

    x : :class:`~.PolyElement` with respect to which ``p`` is integrated.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_integrate
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x**2*y**3
    >>> rs_integrate(p, x)
    1/3*x**3*y**3 + 1/2*x**2
    """
    R = p.ring
    p1 = {}
    n = R.gens.index(x)
    mn = [0]*R.ngens
    mn[n] = 1
    mn = tuple(mn)

    for expv in p:
        e = monomial_mul(expv, mn)
        p1[e] = R.domain_new(p[expv]/(expv[n] + 1))
    return R(p1)

def rs_fun(p, f, *args):
    r"""
    Function of a multivariate series computed by substitution.

    The case with f method name is used to compute `rs\_tan` and `rs\_nth\_root`
    of a multivariate series:

        `rs\_fun(p, tan, iv, prec)`

        tan series is first computed for a dummy variable _x,
        i.e, `rs\_tan(\_x, iv, prec)`. Then we substitute _x with p to get the
        desired series

    Parameters
    ==========

    p : :class:`~.PolyElement` The multivariate series to be expanded.
    f : `ring\_series` function to be applied on `p`.
    args[-2] : :class:`~.PolyElement` with respect to which, the series is to be expanded.
    args[-1] : Required order of the expanded series.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_fun, _tan1
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x*y + x**2*y + x**3*y**2
    >>> rs_fun(p, _tan1, x, 4)
    1/3*x**3*y**3 + 2*x**3*y**2 + x**3*y + 1/3*x**3 + x**2*y + x*y + x
    """
    _R = p.ring
    R1, _x = ring('_x', _R.domain)
    h = int(args[-1])
    args1 = args[:-2] + (_x, h)
    zm = _R.zero_monom
    # separate the constant term of the series
    # compute the univariate series f(_x, .., 'x', sum(nv))
    if zm in p:
        x1 = _x + p[zm]
        p1 = p - p[zm]
    else:
        x1 = _x
        p1 = p
    if isinstance(f, str):
        q = getattr(x1, f)(*args1)
    else:
        q = f(x1, *args1)
    a = sorted(q.items())
    c = [0]*h
    for x in a:
        c[x[0][0]] = x[1]
    p1 = rs_series_from_list(p1, c, args[-2], args[-1])
    return p1

def mul_xin(p, i, n):
    r"""
    Return `p*x_i**n`.

    `x\_i` is the ith variable in ``p``.
    """
    R = p.ring
    q = {}
    for k, v in p.terms():
        k1 = list(k)
        k1[i] += n
        q[tuple(k1)] = v
    return R(q)

def pow_xin(p, i, n):
    """
    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import pow_xin
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> p = x**QQ(2,5) + x + x**QQ(2,3)
    >>> index = p.ring.gens.index(x)
    >>> pow_xin(p, index, 15)
    x**6 + x**10 + x**15
    """
    R = p.ring
    q = {}
    for k, v in p.terms():
        k1 = list(k)
        k1[i] *= n
        q[tuple(k1)] = v
    return R(q)

def _nth_root1(p, n, x, prec):
    """
    Univariate series expansion of the nth root of ``p``.

    The Newton method is used.
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux2(_nth_root1, p, n, x, prec)
    R = p.ring
    zm = R.zero_monom
    if zm not in p:
        raise NotImplementedError('No constant term in series')
    n = as_int(n)
    assert p[zm] == 1
    p1 = R(1)
    if p == 1:
        return p
    if n == 0:
        return R(1)
    if n == 1:
        return p
    if n < 0:
        n = -n
        sign = 1
    else:
        sign = 0
    for precx in _giant_steps(prec):
        tmp = rs_pow(p1, n + 1, x, precx)
        tmp = rs_mul(tmp, p, x, precx)
        p1 += p1/n - tmp/n
    if sign:
        return p1
    else:
        return _series_inversion1(p1, x, prec)

def rs_nth_root(p, n, x, prec):
    """
    Multivariate series expansion of the nth root of ``p``.

    Parameters
    ==========

    p : Expr
        The polynomial to computer the root of.
    n : integer
        The order of the root to be computed.
    x : :class:`~.PolyElement`
    prec : integer
        Order of the expanded series.

    Notes
    =====

    The result of this function is dependent on the ring over which the
    polynomial has been defined. If the answer involves a root of a constant,
    make sure that the polynomial is over a real field. It cannot yet handle
    roots of symbols.

    Examples
    ========

    >>> from sympy.polys.domains import QQ, RR
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_nth_root
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_nth_root(1 + x + x*y, -3, x, 3)
    2/9*x**2*y**2 + 4/9*x**2*y + 2/9*x**2 - 1/3*x*y - 1/3*x + 1
    >>> R, x, y = ring('x, y', RR)
    >>> rs_nth_root(3 + x + x*y, 3, x, 2)
    0.160249952256379*x*y + 0.160249952256379*x + 1.44224957030741
    """
    if n == 0:
        if p == 0:
            raise ValueError('0**0 expression')
        else:
            return p.ring(1)
    if n == 1:
        return rs_trunc(p, x, prec)
    R = p.ring
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    p = mul_xin(p, index, -m)
    prec -= m

    if _has_constant_term(p - 1, x):
        zm = R.zero_monom
        c = p[zm]
        if isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(c_expr**(QQ(1, n)))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        else:
            try:                              # RealElement doesn't support
                const = R(c**Rational(1, n))  # exponentiation with mpq object
            except ValueError:                # as exponent
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        res = rs_nth_root(p/c, n, x, prec)*const
    else:
        res = _nth_root1(p, n, x, prec)
    if m:
        m = QQ(m) / n
        res = mul_xin(res, index, m)
    return res

def rs_log(p, x, prec):
    """
    The Logarithm of ``p`` modulo ``O(x**prec)``.

    Notes
    =====

    Truncation of ``integral dx p**-1*d p/dx`` is used.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_log
    >>> R, x = puiseux_ring('x', QQ)
    >>> rs_log(1 + x, x, 8)
    x + -1/2*x**2 + 1/3*x**3 + -1/4*x**4 + 1/5*x**5 + -1/6*x**6 + 1/7*x**7
    >>> rs_log(x**QQ(3, 2) + 1, x, 5)
    x**(3/2) + -1/2*x**3 + 1/3*x**(9/2)
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_log, p, x, prec)
    R = p.ring
    if p == 1:
        return R.zero
    c = _get_constant_term(p, x)
    if c:
        const = 0
        if c == 1:
            pass
        try:
            c_expr = c.as_expr()
            const = R(log(c_expr))
        except ValueError:
            R = R.add_gens([log(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(log(c_expr))

        dlog = p.diff(x)
        dlog = rs_mul(dlog, _series_inversion1(p, x, prec), x, prec - 1)
        return rs_integrate(dlog, x) + const
    else:
        raise NotImplementedError

def rs_LambertW(p, x, prec):
    """
    Calculate the series expansion of the principal branch of the Lambert W
    function.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_LambertW
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_LambertW(x + x*y, x, 3)
    -x**2*y**2 - 2*x**2*y - x**2 + x*y + x

    See Also
    ========

    LambertW
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_LambertW, p, x, prec)
    R = p.ring
    p1 = R(0)
    if _has_constant_term(p, x):
        raise NotImplementedError("Polynomial must not have constant term in "
                                  "the series variables")
    if x in R.gens:
        for precx in _giant_steps(prec):
            e = rs_exp(p1, x, precx)
            p2 = rs_mul(e, p1, x, precx) - p
            p3 = rs_mul(e, p1 + 1, x, precx)
            p3 = rs_series_inversion(p3, x, precx)
            tmp = rs_mul(p2, p3, x, precx)
            p1 -= tmp
        return p1
    else:
        raise NotImplementedError

def _exp1(p, x, prec):
    r"""Helper function for `rs\_exp`. """
    R = p.ring
    p1 = R(1)
    for precx in _giant_steps(prec):
        pt = p - rs_log(p1, x, precx)
        tmp = rs_mul(pt, p1, x, precx)
        p1 += tmp
    return p1

def rs_exp(p, x, prec):
    """
    Exponentiation of a series modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_exp
    >>> R, x = ring('x', QQ)
    >>> rs_exp(x**2, x, 7)
    1/6*x**6 + 1/2*x**4 + x**2 + 1
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_exp, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(exp(c_expr))
        except ValueError:
            R = R.add_gens([exp(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(exp(c_expr))

        p1 = p - c

    # Makes use of SymPy functions to evaluate the values of the cos/sin
    # of the constant term.
        return const*rs_exp(p1, x, prec)

    if len(p) > 20:
        return _exp1(p, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(prec):
        c.append(one/n)
        k += 1
        n *= k

    r = rs_series_from_list(p, c, x, prec)
    return r

def _atan(p, iv, prec):
    """
    Expansion using formula.

    Faster on very small and univariate series.
    """
    R = p.ring
    mo = R(-1)
    c = [-mo]
    p2 = rs_square(p, iv, prec)
    for k in range(1, prec):
        c.append(mo**k/(2*k + 1))
    s = rs_series_from_list(p2, c, iv, prec)
    s = rs_mul(s, p, iv, prec)
    return s

def rs_atan(p, x, prec):
    """
    The arctangent of a series

    Return the series expansion of the atan of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_atan
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_atan(x + x*y, x, 4)
    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x

    See Also
    ========

    atan
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atan, p, x, prec)
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(atan(c_expr))
        except ValueError:
            R = R.add_gens([atan(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(atan(c_expr))

    # Instead of using a closed form formula, we differentiate atan(p) to get
    # `1/(1+p**2) * dp`, whose series expansion is much easier to calculate.
    # Finally we integrate to get back atan
    dp = p.diff(x)
    p1 = rs_square(p, x, prec) + R(1)
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_asin(p, x, prec):
    """
    Arcsine of a series

    Return the series expansion of the asin of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_asin
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_asin(x, x, 8)
    5/112*x**7 + 3/40*x**5 + 1/6*x**3 + x

    See Also
    ========

    asin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_asin, p, x, prec)
    if _has_constant_term(p, x):
        raise NotImplementedError("Polynomial must not have constant term in "
                                    "series variables")
    R = p.ring
    if x in R.gens:
        # get a good value
        if len(p) > 20:
            dp = rs_diff(p, x)
            p1 = 1 - rs_square(p, x, prec - 1)
            p1 = rs_nth_root(p1, -2, x, prec - 1)
            p1 = rs_mul(dp, p1, x, prec - 1)
            return rs_integrate(p1, x)
        one = R(1)
        c = [0, one, 0]
        for k in range(3, prec, 2):
            c.append((k - 2)**2*c[-2]/(k*(k - 1)))
            c.append(0)
        return rs_series_from_list(p, c, x, prec)

    else:
        raise NotImplementedError

def _tan1(p, x, prec):
    r"""
    Helper function of :func:`rs_tan`.

    Return the series expansion of tan of a univariate series using Newton's
    method. It takes advantage of the fact that series expansion of atan is
    easier than that of tan.

    Consider `f(x) = y - \arctan(x)`
    Let r be a root of f(x) found using Newton's method.
    Then `f(r) = 0`
    Or `y = \arctan(x)` where `x = \tan(y)` as required.
    """
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atan(p1, x, precx)
        tmp = rs_mul(tmp, 1 + rs_square(p1, x, precx), x, precx)
        p1 += tmp
    return p1

def rs_tan(p, x, prec):
    """
    Tangent of a series.

    Return the series expansion of the tan of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_tan
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_tan(x + x*y, x, 4)
    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x

   See Also
   ========

   _tan1, tan
   """
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_tan, p, x, prec)
        return r
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(tan(c_expr))
        except ValueError:
            R = R.add_gens([tan(c_expr, )])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(tan(c_expr))

        p1 = p - c

    # Makes use of SymPy functions to evaluate the values of the cos/sin
    # of the constant term.
        t2 = rs_tan(p1, x, prec)
        t = rs_series_inversion(1 - const*t2, x, prec)
        return rs_mul(const + t2, t, x, prec)

    if R.ngens == 1:
        return _tan1(p, x, prec)
    else:
        return rs_fun(p, rs_tan, x, prec)

def rs_cot(p, x, prec):
    """
    Cotangent of a series

    Return the series expansion of the cot of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cot
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cot(x, x, 6)
    -2/945*x**5 - 1/45*x**3 - 1/3*x + x**(-1)

    See Also
    ========

    cot
    """
    # It can not handle series like `p = x + x*y` where the coefficient of the
    # linear term in the series variable is symbolic.
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_cot, p, x, prec)
        return r
    i, m = _check_series_var(p, x, 'cot')
    prec1 = int(prec + 2*m)
    c, s = rs_cos_sin(p, x, prec1)
    s = mul_xin(s, i, -m)
    s = rs_series_inversion(s, x, prec1)
    res = rs_mul(c, s, x, prec1)
    res = mul_xin(res, i, -m)
    res = rs_trunc(res, x, prec)
    return res

def rs_sin(p, x, prec):
    """
    Sine of a series

    Return the series expansion of the sin of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_sin
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> rs_sin(x + x*y, x, 4)
    x + x*y + -1/6*x**3 + -1/2*x**3*y + -1/2*x**3*y**2 + -1/6*x**3*y**3
    >>> rs_sin(x**QQ(3, 2) + x*y**QQ(7, 5), x, 4)
    x*y**(7/5) + x**(3/2) + -1/6*x**3*y**(21/5) + -1/2*x**(7/2)*y**(14/5)

    See Also
    ========

    sin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sin, p, x, prec)
    R = x.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        except ValueError:
            R = R.add_gens([sin(c_expr), cos(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))

        p1 = p - c

    # Makes use of SymPy cos, sin functions to evaluate the values of the
    # cos/sin of the constant term.
        p_cos, p_sin = rs_cos_sin(p1, x, prec)
        return p_sin*t2 + p_cos*t1

    # Series is calculated in terms of tan as its evaluation is fast.
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return rs_mul(p1, 2*t, x, prec)
    one = R(1)
    n = 1
    c = [0]
    for k in range(2, prec + 2, 2):
        c.append(one/n)
        c.append(0)
        n *= -k*(k + 1)
    return rs_series_from_list(p, c, x, prec)

def rs_cos(p, x, prec):
    """
    Cosine of a series

    Return the series expansion of the cos of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_cos
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> rs_cos(x + x*y, x, 4)
    1 + -1/2*x**2 + -1*x**2*y + -1/2*x**2*y**2
    >>> rs_cos(x + x*y, x, 4)/x**QQ(7, 5)
    x**(-7/5) + -1/2*x**(3/5) + -1*x**(3/5)*y + -1/2*x**(3/5)*y**2

    See Also
    ========

    cos
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        except ValueError:
            R = R.add_gens([sin(c_expr), cos(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))

        p1 = p - c
        # Makes use of SymPy cos, sin functions to evaluate the values of the
        # cos/sin of the constant term.
        p_cos, p_sin = rs_cos_sin(p1, x, prec)
        return p_cos*t2 - p_sin*t1

    # Series is calculated in terms of tan as its evaluation is fast.
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1+t2, x, prec)
        return rs_mul(p1, 1 - t2, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(2, prec + 2, 2):
        c.append(one/n)
        c.append(0)
        n *= -k*(k - 1)
    return rs_series_from_list(p, c, x, prec)

def rs_cos_sin(p, x, prec):
    """
    Cosine and sine of a series

    Return the series expansion of the cosine and sine of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cos_sin
    >>> R, x, y = ring('x, y', QQ)
    >>> c, s = rs_cos_sin(x + x*y, x, 4)
    >>> c
    -1/2*x**2*y**2 - x**2*y - 1/2*x**2 + 1
    >>> s
    -1/6*x**3*y**3 - 1/2*x**3*y**2 - 1/2*x**3*y - 1/6*x**3 + x*y + x

    See Also
    ========

    rs_cos, rs_sin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos_sin, p, x, prec)
    R = p.ring
    if not p:
        return R(0), R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        except ValueError:
            R = R.add_gens([sin(c_expr), cos(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))

        p1 = p - c
        p_cos, p_sin = rs_cos_sin(p1, x, prec)
        return p_cos*t2 - p_sin*t1, p_cos*t1 + p_sin*t2

    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return (rs_mul(p1, 1 - t2, x, prec), rs_mul(p1, 2*t, x, prec))

    one = R(1)
    coeffs = []
    cn, sn = 1, 1
    for k in range(2, prec+2, 2):
        coeffs.extend([(one/cn, 0), (0, one/sn)])
        cn, sn = -cn*k*(k - 1), -sn*k*(k + 1)

    c, s = zip(*coeffs)
    return (rs_series_from_list(p, c, x, prec), rs_series_from_list(p, s, x, prec))

def _atanh(p, x, prec):
    """
    Expansion using formula

    Faster for very small and univariate series
    """
    R = p.ring
    one = R(1)
    c = [one]
    p2 = rs_square(p, x, prec)
    for k in range(1, prec):
        c.append(one/(2*k + 1))
    s = rs_series_from_list(p2, c, x, prec)
    s = rs_mul(s, p, x, prec)
    return s

def rs_atanh(p, x, prec):
    """
    Hyperbolic arctangent of a series

    Return the series expansion of the atanh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_atanh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_atanh(x + x*y, x, 4)
    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x

    See Also
    ========

    atanh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atanh, p, x, prec)
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(atanh(c_expr))
        except ValueError:
            raise DomainError("The given series cannot be expanded in "
                "this domain.")

    # Instead of using a closed form formula, we differentiate atanh(p) to get
    # `1/(1-p**2) * dp`, whose series expansion is much easier to calculate.
    # Finally we integrate to get back atanh
    dp = rs_diff(p, x)
    p1 = - rs_square(p, x, prec) + 1
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_asinh(p, x, prec):
    """
    Hyperbolic arcsine of a series

    Return the series expansion of the arcsinh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_asinh
    >>> R, x = ring('x', QQ)
    >>> rs_asinh(x, x, 9)
    -5/112*x**7 + 3/40*x**5 - 1/6*x**3 + x

    See Also
    ========

    asinh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_asinh, p, x, prec)
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(asinh(c_expr))
        except ValueError:
            raise DomainError("The given series cannot be expanded in "
                "this domain.")

    # Instead of using a closed form formula, we differentiate asinh(p) to get
    # `1/sqrt(1+p**2) * dp`, whose series expansion is much easier to calculate.
    # Finally we integrate to get back asinh
    dp = rs_diff(p, x)
    p_squared = rs_square(p, x, prec)
    denom = p_squared + R(1)
    p1 = rs_nth_root(denom, -2, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_sinh(p, x, prec):
    """
    Hyperbolic sine of a series

    Return the series expansion of the sinh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_sinh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_sinh(x + x*y, x, 4)
    1/6*x**3*y**3 + 1/2*x**3*y**2 + 1/2*x**3*y + 1/6*x**3 + x*y + x

    See Also
    ========

    sinh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sinh, p, x, prec)
    R = p.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))
        except ValueError:
            R = R.add_gens([sinh(c_expr), cosh(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))

        p1 = p - c
        p_cosh, p_sinh = rs_cosh_sinh(p1, x, prec)
        return p_sinh * t2 + p_cosh * t1

    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t - t1)/2

def rs_cosh(p, x, prec):
    """
    Hyperbolic cosine of a series

    Return the series expansion of the cosh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cosh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cosh(x + x*y, x, 4)
    1/2*x**2*y**2 + x**2*y + 1/2*x**2 + 1

    See Also
    ========

    cosh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cosh, p, x, prec)
    R = p.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))
        except ValueError:
            R = R.add_gens([sinh(c_expr), cosh(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))

        p1 = p - c
        p_cosh, p_sinh = rs_cosh_sinh(p1, x, prec)
        return p_cosh * t2 + p_sinh * t1

    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t + t1)/2

def rs_cosh_sinh(p, x, prec):
    """
    Hyperbolic cosine and sine of a series

    Return the series expansion of the hyperbolic cosine and sine of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cosh_sinh
    >>> R, x, y = ring('x, y', QQ)
    >>> c, s = rs_cosh_sinh(x + x*y, x, 4)
    >>> c
    1/2*x**2*y**2 + x**2*y + 1/2*x**2 + 1
    >>> s
    1/6*x**3*y**3 + 1/2*x**3*y**2 + 1/2*x**3*y + 1/6*x**3 + x*y + x

    See Also
    ========

    rs_cosh, rs_sinh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cosh_sinh, p, x, prec)
    R = p.ring
    if not p:
        return R(0), R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))
        except ValueError:
            R = R.add_gens([sinh(c_expr), cosh(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))

        p1 = p - c
        p_cosh, p_sinh = rs_cosh_sinh(p1, x, prec)
        return p_cosh * t2 + p_sinh * t1, p_sinh * t2 + p_cosh * t1

    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t + t1)/2, (t - t1)/2


def _tanh(p, x, prec):
    r"""
    Helper function of :func:`rs_tanh`

    Return the series expansion of tanh of a univariate series using Newton's
    method. It takes advantage of the fact that series expansion of atanh is
    easier than that of tanh.

    See Also
    ========

    _tanh
    """
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atanh(p1, x, precx)
        tmp = rs_mul(tmp, 1 - rs_square(p1, x, prec), x, precx)
        p1 += tmp
    return p1

def rs_tanh(p, x, prec):
    """
    Hyperbolic tangent of a series

    Return the series expansion of the tanh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_tanh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_tanh(x + x*y, x, 4)
    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x

    See Also
    ========

    tanh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_tanh, p, x, prec)
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(tanh(c_expr))
        except ValueError:
            R = R.add_gens([tanh(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(tanh(c_expr))

        p1 = p - c
        t1 = rs_tanh(p1, x, prec)
        t = rs_series_inversion(1 + const*t1, x, prec)
        return rs_mul(const + t1, t, x, prec)

    if R.ngens == 1:
        return _tanh(p, x, prec)
    else:
        return rs_fun(p, _tanh, x, prec)

def rs_newton(p, x, prec):
    """
    Compute the truncated Newton sum of the polynomial ``p``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_newton
    >>> R, x = ring('x', QQ)
    >>> p = x**2 - 2
    >>> rs_newton(p, x, 5)
    8*x**4 + 4*x**2 + 2
    """
    deg = p.degree()
    p1 = _invert_monoms(p)
    p2 = rs_series_inversion(p1, x, prec)
    p3 = rs_mul(p1.diff(x), p2, x, prec)
    res = deg - p3*x
    return res

def rs_hadamard_exp(p1, inverse=False):
    """
    Return ``sum f_i/i!*x**i`` from ``sum f_i*x**i``,
    where ``x`` is the first variable.

    If ``inverse=True`` return ``sum f_i*i!*x**i``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_hadamard_exp
    >>> R, x = ring('x', QQ)
    >>> p = 1 + x + x**2 + x**3
    >>> rs_hadamard_exp(p)
    1/6*x**3 + 1/2*x**2 + x + 1
    """
    R = p1.ring
    if R.domain != QQ:
        raise NotImplementedError
    p = R.zero
    if not inverse:
        for exp1, v1 in p1.items():
            p[exp1] = v1/int(ifac(exp1[0]))
    else:
        for exp1, v1 in p1.items():
            p[exp1] = v1*int(ifac(exp1[0]))
    return p

def rs_compose_add(p1, p2):
    """
    compute the composed sum ``prod(p2(x - beta) for beta root of p1)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_compose_add
    >>> R, x = ring('x', QQ)
    >>> f = x**2 - 2
    >>> g = x**2 - 3
    >>> rs_compose_add(f, g)
    x**4 - 10*x**2 + 1

    References
    ==========

    .. [1] A. Bostan, P. Flajolet, B. Salvy and E. Schost
           "Fast Computation with Two Algebraic Numbers",
           (2002) Research Report 4579, Institut
           National de Recherche en Informatique et en Automatique
    """
    R = p1.ring
    x = R.gens[0]
    prec = p1.degree()*p2.degree() + 1
    np1 = rs_newton(p1, x, prec)
    np1e = rs_hadamard_exp(np1)
    np2 = rs_newton(p2, x, prec)
    np2e = rs_hadamard_exp(np2)
    np3e = rs_mul(np1e, np2e, x, prec)
    np3 = rs_hadamard_exp(np3e, True)
    np3a = (np3[(0,)] - np3) / x
    q = rs_integrate(np3a, x)
    q = rs_exp(q, x, prec)
    q = _invert_monoms(q)
    q = q.primitive()[1]
    dp = p1.degree()*p2.degree() - q.degree()
    # `dp` is the multiplicity of the zeroes of the resultant;
    # these zeroes are missed in this computation so they are put here.
    # if p1 and p2 are monic irreducible polynomials,
    # there are zeroes in the resultant
    # if and only if p1 = p2 ; in fact in that case p1 and p2 have a
    # root in common, so gcd(p1, p2) != 1; being p1 and p2 irreducible
    # this means p1 = p2
    if dp:
        q = q*x**dp
    return q


_convert_func = {
        'sin': 'rs_sin',
        'cos': 'rs_cos',
        'exp': 'rs_exp',
        'tan': 'rs_tan',
        'log': 'rs_log',
        'atan': 'rs_atan',
        'sinh': 'rs_sinh',
        'cosh': 'rs_cosh',
        'tanh': 'rs_tanh'
        }

def rs_min_pow(expr, series_rs, a):
    """Find the minimum power of `a` in the series expansion of expr"""
    series = 0
    n = 2
    while series == 0:
        series = _rs_series(expr, series_rs, a, n)
        n *= 2
    R = series.ring
    a = R(a)
    i = R.gens.index(a)
    return min(series, key=lambda t: t[i])[i]


def _rs_series(expr, series_rs, a, prec):
    # TODO Use _parallel_dict_from_expr instead of sring as sring is
    # inefficient. For details, read the todo in sring.
    args = expr.args
    R = series_rs.ring

    # expr does not contain any function to be expanded
    if not any(arg.has(Function) for arg in args) and not expr.is_Function:
        return series_rs

    if not expr.has(a):
        return series_rs

    elif expr.is_Function:
        arg = args[0]
        if len(args) > 1:
            raise NotImplementedError
        R1, series = sring(arg, domain=QQ, expand=False, series=True)
        series_inner = _rs_series(arg, series, a, prec)

        # Why do we need to compose these three rings?
        #
        # We want to use a simple domain (like ``QQ`` or ``RR``) but they don't
        # support symbolic coefficients. We need a ring that for example lets
        # us have `sin(1)` and `cos(1)` as coefficients if we are expanding
        # `sin(x + 1)`. The ``EX`` domain allows all symbolic coefficients, but
        # that makes it very complex and hence slow.
        #
        # To solve this problem, we add only those symbolic elements as
        # generators to our ring, that we need. Here, series_inner might
        # involve terms like `sin(4)`, `exp(a)`, etc, which are not there in
        # R1 or R. Hence, we compose these three rings to create one that has
        # the generators of all three.
        R = R.compose(R1).compose(series_inner.ring)
        series_inner = series_inner.set_ring(R)
        series = eval(_convert_func[str(expr.func)])(series_inner,
            R(a), prec)
        return series

    elif expr.is_Mul:
        n = len(args)
        for arg in args:    # XXX Looks redundant
            if not arg.is_Number:
                R1, _ = sring(arg, expand=False, series=True)
                R = R.compose(R1)
        min_pows = list(map(rs_min_pow, args, [R(arg) for arg in args],
            [a]*len(args)))
        sum_pows = sum(min_pows)
        series = R(1)

        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, ceiling(prec
                - sum_pows + min_pows[i]))
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series *= _series
        series = rs_trunc(series, R(a), prec)
        return series

    elif expr.is_Add:
        n = len(args)
        series = R(0)
        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec)
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series += _series
        return series

    elif expr.is_Pow:
        R1, _ = sring(expr.base, domain=QQ, expand=False, series=True)
        R = R.compose(R1)
        series_inner = _rs_series(expr.base, R(expr.base), a, prec)
        return rs_pow(series_inner, expr.exp, series_inner.ring(a), prec)

    # The `is_constant` method is buggy hence we check it at the end.
    # See issue #9786 for details.
    elif isinstance(expr, Expr) and expr.is_constant():
        return sring(expr, domain=QQ, expand=False, series=True)[1]

    else:
        raise NotImplementedError

def rs_series(expr, a, prec):
    """Return the series expansion of an expression about 0.

    Parameters
    ==========

    expr : :class:`~.Expr`
    a : :class:`~.Symbol` with respect to which expr is to be expanded
    prec : order of the series expansion

    Currently supports multivariate Taylor series expansion. This is much
    faster that SymPy's series method as it uses sparse polynomial operations.

    It automatically creates the simplest ring required to represent the series
    expansion through repeated calls to sring.

    Examples
    ========

    >>> from sympy.polys.ring_series import rs_series
    >>> from sympy import sin, cos, exp, tan, symbols, QQ
    >>> a, b, c = symbols('a, b, c')
    >>> rs_series(sin(a) + exp(a), a, 5)
    1/24*a**4 + 1/2*a**2 + 2*a + 1
    >>> series = rs_series(tan(a + b)*cos(a + c), a, 2)
    >>> series.as_expr()
    -a*sin(c)*tan(b) + a*cos(c)*tan(b)**2 + a*cos(c) + cos(c)*tan(b)
    >>> series = rs_series(exp(a**QQ(1,3) + a**QQ(2, 5)), a, 1)
    >>> series.as_expr()
    a**(11/15) + a**(4/5)/2 + a**(2/5) + a**(2/3)/2 + a**(1/3) + 1

    """
    R, series = sring(expr, domain=QQ, expand=False, series=True)
    if a not in R.symbols:
        R = R.add_gens([a, ])
    series = series.set_ring(R)
    series = _rs_series(expr, series, a, prec)
    R = series.ring
    gen = R(a)
    prec_got = series.degree(gen) + 1

    if prec_got >= prec:
        return rs_trunc(series, gen, prec)
    else:
        # increase the requested number of terms to get the desired
        # number keep increasing (up to 9) until the received order
        # is different than the original order and then predict how
        # many additional terms are needed
        for more in range(1, 9):
            p1 = _rs_series(expr, series, a, prec=prec + more)
            gen = gen.set_ring(p1.ring)
            new_prec = p1.degree(gen) + 1
            if new_prec != prec_got:
                prec_do = ceiling(prec + (prec - prec_got)*more/(new_prec -
                    prec_got))
                p1 = _rs_series(expr, series, a, prec=prec_do)
                while p1.degree(gen) + 1 < prec:
                    p1 = _rs_series(expr, series, a, prec=prec_do)
                    gen = gen.set_ring(p1.ring)
                    prec_do *= 2
                break
            else:
                break
        else:
            raise ValueError('Could not calculate %s terms for %s'
                             % (str(prec), expr))
        return rs_trunc(p1, gen, prec)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\core.py ===
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    final,
)
import warnings

import matplotlib as mpl
import numpy as np

from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_any_real_numeric_dtype,
    is_bool,
    is_float,
    is_float_dtype,
    is_hashable,
    is_integer,
    is_integer_dtype,
    is_iterator,
    is_list_like,
    is_number,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
    decorate_axes,
    format_dateaxis,
    maybe_convert_index,
    maybe_resample,
    use_dynamic_x,
)
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    format_date_labels,
    get_all_lines,
    get_xlim,
    handle_shared_axes,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure

    from pandas._typing import (
        IndexLabel,
        NDFrameT,
        PlottingOrientation,
        npt,
    )

    from pandas import Series


def _color_in_style(style: str) -> bool:
    """
    Check if there is a color letter in the style string.
    """
    from matplotlib.colors import BASE_COLORS

    return not set(BASE_COLORS).isdisjoint(style)


class MPLPlot(ABC):
    """
    Base class for assembling a pandas plot using matplotlib

    Parameters
    ----------
    data :

    """

    @property
    @abstractmethod
    def _kind(self) -> str:
        """Specify kind str. Must be overridden in child class"""
        raise NotImplementedError

    _layout_type = "vertical"
    _default_rot = 0

    @property
    def orientation(self) -> str | None:
        return None

    data: DataFrame

    def __init__(
        self,
        data,
        kind=None,
        by: IndexLabel | None = None,
        subplots: bool | Sequence[Sequence[str]] = False,
        sharex: bool | None = None,
        sharey: bool = False,
        use_index: bool = True,
        figsize: tuple[float, float] | None = None,
        grid=None,
        legend: bool | str = True,
        rot=None,
        ax=None,
        fig=None,
        title=None,
        xlim=None,
        ylim=None,
        xticks=None,
        yticks=None,
        xlabel: Hashable | None = None,
        ylabel: Hashable | None = None,
        fontsize: int | None = None,
        secondary_y: bool | tuple | list | np.ndarray = False,
        colormap=None,
        table: bool = False,
        layout=None,
        include_bool: bool = False,
        column: IndexLabel | None = None,
        *,
        logx: bool | None | Literal["sym"] = False,
        logy: bool | None | Literal["sym"] = False,
        loglog: bool | None | Literal["sym"] = False,
        mark_right: bool = True,
        stacked: bool = False,
        label: Hashable | None = None,
        style=None,
        **kwds,
    ) -> None:
        import matplotlib.pyplot as plt

        # if users assign an empty list or tuple, raise `ValueError`
        # similar to current `df.box` and `df.hist` APIs.
        if by in ([], ()):
            raise ValueError("No group keys passed!")
        self.by = com.maybe_make_list(by)

        # Assign the rest of columns into self.columns if by is explicitly defined
        # while column is not, only need `columns` in hist/box plot when it's DF
        # TODO: Might deprecate `column` argument in future PR (#28373)
        if isinstance(data, DataFrame):
            if column:
                self.columns = com.maybe_make_list(column)
            elif self.by is None:
                self.columns = [
                    col for col in data.columns if is_numeric_dtype(data[col])
                ]
            else:
                self.columns = [
                    col
                    for col in data.columns
                    if col not in self.by and is_numeric_dtype(data[col])
                ]

        # For `hist` plot, need to get grouped original data before `self.data` is
        # updated later
        if self.by is not None and self._kind == "hist":
            self._grouped = data.groupby(unpack_single_str_list(self.by))

        self.kind = kind

        self.subplots = type(self)._validate_subplots_kwarg(
            subplots, data, kind=self._kind
        )

        self.sharex = type(self)._validate_sharex(sharex, ax, by)
        self.sharey = sharey
        self.figsize = figsize
        self.layout = layout

        self.xticks = xticks
        self.yticks = yticks
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.use_index = use_index
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.fontsize = fontsize

        if rot is not None:
            self.rot = rot
            # need to know for format_date_labels since it's rotated to 30 by
            # default
            self._rot_set = True
        else:
            self._rot_set = False
            self.rot = self._default_rot

        if grid is None:
            grid = False if secondary_y else plt.rcParams["axes.grid"]

        self.grid = grid
        self.legend = legend
        self.legend_handles: list[Artist] = []
        self.legend_labels: list[Hashable] = []

        self.logx = type(self)._validate_log_kwd("logx", logx)
        self.logy = type(self)._validate_log_kwd("logy", logy)
        self.loglog = type(self)._validate_log_kwd("loglog", loglog)
        self.label = label
        self.style = style
        self.mark_right = mark_right
        self.stacked = stacked

        # ax may be an Axes object or (if self.subplots) an ndarray of
        #  Axes objects
        self.ax = ax
        # TODO: deprecate fig keyword as it is ignored, not passed in tests
        #  as of 2023-11-05

        # parse errorbar input if given
        xerr = kwds.pop("xerr", None)
        yerr = kwds.pop("yerr", None)
        nseries = self._get_nseries(data)
        xerr, data = type(self)._parse_errorbars("xerr", xerr, data, nseries)
        yerr, data = type(self)._parse_errorbars("yerr", yerr, data, nseries)
        self.errors = {"xerr": xerr, "yerr": yerr}
        self.data = data

        if not isinstance(secondary_y, (bool, tuple, list, np.ndarray, ABCIndex)):
            secondary_y = [secondary_y]
        self.secondary_y = secondary_y

        # ugly TypeError if user passes matplotlib's `cmap` name.
        # Probably better to accept either.
        if "cmap" in kwds and colormap:
            raise TypeError("Only specify one of `cmap` and `colormap`.")
        if "cmap" in kwds:
            self.colormap = kwds.pop("cmap")
        else:
            self.colormap = colormap

        self.table = table
        self.include_bool = include_bool

        self.kwds = kwds

        color = kwds.pop("color", lib.no_default)
        self.color = self._validate_color_args(color, self.colormap)
        assert "color" not in self.kwds

        self.data = self._ensure_frame(self.data)

    @final
    @staticmethod
    def _validate_sharex(sharex: bool | None, ax, by) -> bool:
        if sharex is None:
            # if by is defined, subplots are used and sharex should be False
            if ax is None and by is None:  # pylint: disable=simplifiable-if-statement
                sharex = True
            else:
                # if we get an axis, the users should do the visibility
                # setting...
                sharex = False
        elif not is_bool(sharex):
            raise TypeError("sharex must be a bool or None")
        return bool(sharex)

    @classmethod
    def _validate_log_kwd(
        cls,
        kwd: str,
        value: bool | None | Literal["sym"],
    ) -> bool | None | Literal["sym"]:
        if (
            value is None
            or isinstance(value, bool)
            or (isinstance(value, str) and value == "sym")
        ):
            return value
        raise ValueError(
            f"keyword '{kwd}' should be bool, None, or 'sym', not '{value}'"
        )

    @final
    @staticmethod
    def _validate_subplots_kwarg(
        subplots: bool | Sequence[Sequence[str]], data: Series | DataFrame, kind: str
    ) -> bool | list[tuple[int, ...]]:
        """
        Validate the subplots parameter

        - check type and content
        - check for duplicate columns
        - check for invalid column names
        - convert column names into indices
        - add missing columns in a group of their own
        See comments in code below for more details.

        Parameters
        ----------
        subplots : subplots parameters as passed to PlotAccessor

        Returns
        -------
        validated subplots : a bool or a list of tuples of column indices. Columns
        in the same tuple will be grouped together in the resulting plot.
        """

        if isinstance(subplots, bool):
            return subplots
        elif not isinstance(subplots, Iterable):
            raise ValueError("subplots should be a bool or an iterable")

        supported_kinds = (
            "line",
            "bar",
            "barh",
            "hist",
            "kde",
            "density",
            "area",
            "pie",
        )
        if kind not in supported_kinds:
            raise ValueError(
                "When subplots is an iterable, kind must be "
                f"one of {', '.join(supported_kinds)}. Got {kind}."
            )

        if isinstance(data, ABCSeries):
            raise NotImplementedError(
                "An iterable subplots for a Series is not supported."
            )

        columns = data.columns
        if isinstance(columns, ABCMultiIndex):
            raise NotImplementedError(
                "An iterable subplots for a DataFrame with a MultiIndex column "
                "is not supported."
            )

        if columns.nunique() != len(columns):
            raise NotImplementedError(
                "An iterable subplots for a DataFrame with non-unique column "
                "labels is not supported."
            )

        # subplots is a list of tuples where each tuple is a group of
        # columns to be grouped together (one ax per group).
        # we consolidate the subplots list such that:
        # - the tuples contain indices instead of column names
        # - the columns that aren't yet in the list are added in a group
        #   of their own.
        # For example with columns from a to g, and
        # subplots = [(a, c), (b, f, e)],
        # we end up with [(ai, ci), (bi, fi, ei), (di,), (gi,)]
        # This way, we can handle self.subplots in a homogeneous manner
        # later.
        # TODO: also accept indices instead of just names?

        out = []
        seen_columns: set[Hashable] = set()
        for group in subplots:
            if not is_list_like(group):
                raise ValueError(
                    "When subplots is an iterable, each entry "
                    "should be a list/tuple of column names."
                )
            idx_locs = columns.get_indexer_for(group)
            if (idx_locs == -1).any():
                bad_labels = np.extract(idx_locs == -1, group)
                raise ValueError(
                    f"Column label(s) {list(bad_labels)} not found in the DataFrame."
                )
            unique_columns = set(group)
            duplicates = seen_columns.intersection(unique_columns)
            if duplicates:
                raise ValueError(
                    "Each column should be in only one subplot. "
                    f"Columns {duplicates} were found in multiple subplots."
                )
            seen_columns = seen_columns.union(unique_columns)
            out.append(tuple(idx_locs))

        unseen_columns = columns.difference(seen_columns)
        for column in unseen_columns:
            idx_loc = columns.get_loc(column)
            out.append((idx_loc,))
        return out

    def _validate_color_args(self, color, colormap):
        if color is lib.no_default:
            # It was not provided by the user
            if "colors" in self.kwds and colormap is not None:
                warnings.warn(
                    "'color' and 'colormap' cannot be used simultaneously. "
                    "Using 'color'",
                    stacklevel=find_stack_level(),
                )
            return None
        if self.nseries == 1 and color is not None and not is_list_like(color):
            # support series.plot(color='green')
            color = [color]

        if isinstance(color, tuple) and self.nseries == 1 and len(color) in (3, 4):
            # support RGB and RGBA tuples in series plot
            color = [color]

        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )

        if self.style is not None:
            if is_list_like(self.style):
                styles = self.style
            else:
                styles = [self.style]
            # need only a single match
            for s in styles:
                if _color_in_style(s):
                    raise ValueError(
                        "Cannot pass 'style' string with a color symbol and "
                        "'color' keyword argument. Please use one or the "
                        "other or pass 'style' without a color symbol"
                    )
        return color

    @final
    @staticmethod
    def _iter_data(
        data: DataFrame | dict[Hashable, Series | DataFrame]
    ) -> Iterator[tuple[Hashable, np.ndarray]]:
        for col, values in data.items():
            # This was originally written to use values.values before EAs
            #  were implemented; adding np.asarray(...) to keep consistent
            #  typing.
            yield col, np.asarray(values.values)

    def _get_nseries(self, data: Series | DataFrame) -> int:
        # When `by` is explicitly assigned, grouped data size will be defined, and
        # this will determine number of subplots to have, aka `self.nseries`
        if data.ndim == 1:
            return 1
        elif self.by is not None and self._kind == "hist":
            return len(self._grouped)
        elif self.by is not None and self._kind == "box":
            return len(self.columns)
        else:
            return data.shape[1]

    @final
    @property
    def nseries(self) -> int:
        return self._get_nseries(self.data)

    @final
    def draw(self) -> None:
        self.plt.draw_if_interactive()

    @final
    def generate(self) -> None:
        self._compute_plot_data()
        fig = self.fig
        self._make_plot(fig)
        self._add_table()
        self._make_legend()
        self._adorn_subplots(fig)

        for ax in self.axes:
            self._post_plot_logic_common(ax)
            self._post_plot_logic(ax, self.data)

    @final
    @staticmethod
    def _has_plotted_object(ax: Axes) -> bool:
        """check whether ax has data"""
        return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0

    @final
    def _maybe_right_yaxis(self, ax: Axes, axes_num: int) -> Axes:
        if not self.on_right(axes_num):
            # secondary axes may be passed via ax kw
            return self._get_ax_layer(ax)

        if hasattr(ax, "right_ax"):
            # if it has right_ax property, ``ax`` must be left axes
            return ax.right_ax
        elif hasattr(ax, "left_ax"):
            # if it has left_ax property, ``ax`` must be right axes
            return ax
        else:
            # otherwise, create twin axes
            orig_ax, new_ax = ax, ax.twinx()
            # TODO: use Matplotlib public API when available
            new_ax._get_lines = orig_ax._get_lines  # type: ignore[attr-defined]
            # TODO #54485
            new_ax._get_patches_for_fill = (  # type: ignore[attr-defined]
                orig_ax._get_patches_for_fill  # type: ignore[attr-defined]
            )
            # TODO #54485
            orig_ax.right_ax, new_ax.left_ax = (  # type: ignore[attr-defined]
                new_ax,
                orig_ax,
            )

            if not self._has_plotted_object(orig_ax):  # no data on left y
                orig_ax.get_yaxis().set_visible(False)

            if self.logy is True or self.loglog is True:
                new_ax.set_yscale("log")
            elif self.logy == "sym" or self.loglog == "sym":
                new_ax.set_yscale("symlog")
            return new_ax

    @final
    @cache_readonly
    def fig(self) -> Figure:
        return self._axes_and_fig[1]

    @final
    @cache_readonly
    # TODO: can we annotate this as both a Sequence[Axes] and ndarray[object]?
    def axes(self) -> Sequence[Axes]:
        return self._axes_and_fig[0]

    @final
    @cache_readonly
    def _axes_and_fig(self) -> tuple[Sequence[Axes], Figure]:
        if self.subplots:
            naxes = (
                self.nseries if isinstance(self.subplots, bool) else len(self.subplots)
            )
            fig, axes = create_subplots(
                naxes=naxes,
                sharex=self.sharex,
                sharey=self.sharey,
                figsize=self.figsize,
                ax=self.ax,
                layout=self.layout,
                layout_type=self._layout_type,
            )
        elif self.ax is None:
            fig = self.plt.figure(figsize=self.figsize)
            axes = fig.add_subplot(111)
        else:
            fig = self.ax.get_figure()
            if self.figsize is not None:
                fig.set_size_inches(self.figsize)
            axes = self.ax

        axes = flatten_axes(axes)

        if self.logx is True or self.loglog is True:
            [a.set_xscale("log") for a in axes]
        elif self.logx == "sym" or self.loglog == "sym":
            [a.set_xscale("symlog") for a in axes]

        if self.logy is True or self.loglog is True:
            [a.set_yscale("log") for a in axes]
        elif self.logy == "sym" or self.loglog == "sym":
            [a.set_yscale("symlog") for a in axes]

        axes_seq = cast(Sequence["Axes"], axes)
        return axes_seq, fig

    @property
    def result(self):
        """
        Return result axes
        """
        if self.subplots:
            if self.layout is not None and not is_list_like(self.ax):
                # error: "Sequence[Any]" has no attribute "reshape"
                return self.axes.reshape(*self.layout)  # type: ignore[attr-defined]
            else:
                return self.axes
        else:
            sec_true = isinstance(self.secondary_y, bool) and self.secondary_y
            # error: Argument 1 to "len" has incompatible type "Union[bool,
            # Tuple[Any, ...], List[Any], ndarray[Any, Any]]"; expected "Sized"
            all_sec = (
                is_list_like(self.secondary_y)
                and len(self.secondary_y) == self.nseries  # type: ignore[arg-type]
            )
            if sec_true or all_sec:
                # if all data is plotted on secondary, return right axes
                return self._get_ax_layer(self.axes[0], primary=False)
            else:
                return self.axes[0]

    @final
    @staticmethod
    def _convert_to_ndarray(data):
        # GH31357: categorical columns are processed separately
        if isinstance(data.dtype, CategoricalDtype):
            return data

        # GH32073: cast to float if values contain nulled integers
        if (is_integer_dtype(data.dtype) or is_float_dtype(data.dtype)) and isinstance(
            data.dtype, ExtensionDtype
        ):
            return data.to_numpy(dtype="float", na_value=np.nan)

        # GH25587: cast ExtensionArray of pandas (IntegerArray, etc.) to
        # np.ndarray before plot.
        if len(data) > 0:
            return np.asarray(data)

        return data

    @final
    def _ensure_frame(self, data) -> DataFrame:
        if isinstance(data, ABCSeries):
            label = self.label
            if label is None and data.name is None:
                label = ""
            if label is None:
                # We'll end up with columns of [0] instead of [None]
                data = data.to_frame()
            else:
                data = data.to_frame(name=label)
        elif self._kind in ("hist", "box"):
            cols = self.columns if self.by is None else self.columns + self.by
            data = data.loc[:, cols]
        return data

    @final
    def _compute_plot_data(self) -> None:
        data = self.data

        # GH15079 reconstruct data if by is defined
        if self.by is not None:
            self.subplots = True
            data = reconstruct_data_with_by(self.data, by=self.by, cols=self.columns)

        # GH16953, infer_objects is needed as fallback, for ``Series``
        # with ``dtype == object``
        data = data.infer_objects(copy=False)
        include_type = [np.number, "datetime", "datetimetz", "timedelta"]

        # GH23719, allow plotting boolean
        if self.include_bool is True:
            include_type.append(np.bool_)

        # GH22799, exclude datetime-like type for boxplot
        exclude_type = None
        if self._kind == "box":
            # TODO: change after solving issue 27881
            include_type = [np.number]
            exclude_type = ["timedelta"]

        # GH 18755, include object and category type for scatter plot
        if self._kind == "scatter":
            include_type.extend(["object", "category", "string"])

        numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)

        is_empty = numeric_data.shape[-1] == 0
        # no non-numeric frames or series allowed
        if is_empty:
            raise TypeError("no numeric data to plot")

        self.data = numeric_data.apply(type(self)._convert_to_ndarray)

    def _make_plot(self, fig: Figure) -> None:
        raise AbstractMethodError(self)

    @final
    def _add_table(self) -> None:
        if self.table is False:
            return
        elif self.table is True:
            data = self.data.transpose()
        else:
            data = self.table
        ax = self._get_ax(0)
        tools.table(ax, data)

    @final
    def _post_plot_logic_common(self, ax: Axes) -> None:
        """Common post process for each axes"""
        if self.orientation == "vertical" or self.orientation is None:
            type(self)._apply_axis_properties(
                ax.xaxis, rot=self.rot, fontsize=self.fontsize
            )
            type(self)._apply_axis_properties(ax.yaxis, fontsize=self.fontsize)

            if hasattr(ax, "right_ax"):
                type(self)._apply_axis_properties(
                    ax.right_ax.yaxis, fontsize=self.fontsize
                )

        elif self.orientation == "horizontal":
            type(self)._apply_axis_properties(
                ax.yaxis, rot=self.rot, fontsize=self.fontsize
            )
            type(self)._apply_axis_properties(ax.xaxis, fontsize=self.fontsize)

            if hasattr(ax, "right_ax"):
                type(self)._apply_axis_properties(
                    ax.right_ax.yaxis, fontsize=self.fontsize
                )
        else:  # pragma no cover
            raise ValueError

    @abstractmethod
    def _post_plot_logic(self, ax: Axes, data) -> None:
        """Post process for each axes. Overridden in child classes"""

    @final
    def _adorn_subplots(self, fig: Figure) -> None:
        """Common post process unrelated to data"""
        if len(self.axes) > 0:
            all_axes = self._get_subplots(fig)
            nrows, ncols = self._get_axes_layout(fig)
            handle_shared_axes(
                axarr=all_axes,
                nplots=len(all_axes),
                naxes=nrows * ncols,
                nrows=nrows,
                ncols=ncols,
                sharex=self.sharex,
                sharey=self.sharey,
            )

        for ax in self.axes:
            ax = getattr(ax, "right_ax", ax)
            if self.yticks is not None:
                ax.set_yticks(self.yticks)

            if self.xticks is not None:
                ax.set_xticks(self.xticks)

            if self.ylim is not None:
                ax.set_ylim(self.ylim)

            if self.xlim is not None:
                ax.set_xlim(self.xlim)

            # GH9093, currently Pandas does not show ylabel, so if users provide
            # ylabel will set it as ylabel in the plot.
            if self.ylabel is not None:
                ax.set_ylabel(pprint_thing(self.ylabel))

            ax.grid(self.grid)

        if self.title:
            if self.subplots:
                if is_list_like(self.title):
                    if len(self.title) != self.nseries:
                        raise ValueError(
                            "The length of `title` must equal the number "
                            "of columns if using `title` of type `list` "
                            "and `subplots=True`.\n"
                            f"length of title = {len(self.title)}\n"
                            f"number of columns = {self.nseries}"
                        )

                    for ax, title in zip(self.axes, self.title):
                        ax.set_title(title)
                else:
                    fig.suptitle(self.title)
            else:
                if is_list_like(self.title):
                    msg = (
                        "Using `title` of type `list` is not supported "
                        "unless `subplots=True` is passed"
                    )
                    raise ValueError(msg)
                self.axes[0].set_title(self.title)

    @final
    @staticmethod
    def _apply_axis_properties(
        axis: Axis, rot=None, fontsize: int | None = None
    ) -> None:
        """
        Tick creation within matplotlib is reasonably expensive and is
        internally deferred until accessed as Ticks are created/destroyed
        multiple times per draw. It's therefore beneficial for us to avoid
        accessing unless we will act on the Tick.
        """
        if rot is not None or fontsize is not None:
            # rot=0 is a valid setting, hence the explicit None check
            labels = axis.get_majorticklabels() + axis.get_minorticklabels()
            for label in labels:
                if rot is not None:
                    label.set_rotation(rot)
                if fontsize is not None:
                    label.set_fontsize(fontsize)

    @final
    @property
    def legend_title(self) -> str | None:
        if not isinstance(self.data.columns, ABCMultiIndex):
            name = self.data.columns.name
            if name is not None:
                name = pprint_thing(name)
            return name
        else:
            stringified = map(pprint_thing, self.data.columns.names)
            return ",".join(stringified)

    @final
    def _mark_right_label(self, label: str, index: int) -> str:
        """
        Append ``(right)`` to the label of a line if it's plotted on the right axis.

        Note that ``(right)`` is only appended when ``subplots=False``.
        """
        if not self.subplots and self.mark_right and self.on_right(index):
            label += " (right)"
        return label

    @final
    def _append_legend_handles_labels(self, handle: Artist, label: str) -> None:
        """
        Append current handle and label to ``legend_handles`` and ``legend_labels``.

        These will be used to make the legend.
        """
        self.legend_handles.append(handle)
        self.legend_labels.append(label)

    def _make_legend(self) -> None:
        ax, leg = self._get_ax_legend(self.axes[0])

        handles = []
        labels = []
        title = ""

        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                # Replace leg.legend_handles because it misses marker info
                if Version(mpl.__version__) < Version("3.7"):
                    handles = leg.legendHandles
                else:
                    handles = leg.legend_handles
                labels = [x.get_text() for x in leg.get_texts()]

            if self.legend:
                if self.legend == "reverse":
                    handles += reversed(self.legend_handles)
                    labels += reversed(self.legend_labels)
                else:
                    handles += self.legend_handles
                    labels += self.legend_labels

                if self.legend_title is not None:
                    title = self.legend_title

            if len(handles) > 0:
                ax.legend(handles, labels, loc="best", title=title)

        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            "No artists with labels found to put in legend.",
                            UserWarning,
                        )
                        ax.legend(loc="best")

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes):
        """
        Take in axes and return ax and legend under different scenarios
        """
        leg = ax.get_legend()

        other_ax = getattr(ax, "left_ax", None) or getattr(ax, "right_ax", None)
        other_leg = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return ax, leg

    @final
    @cache_readonly
    def plt(self):
        import matplotlib.pyplot as plt

        return plt

    _need_to_set_index = False

    @final
    def _get_xticks(self):
        index = self.data.index
        is_datetype = index.inferred_type in ("datetime", "date", "datetime64", "time")

        # TODO: be stricter about x?
        x: list[int] | np.ndarray
        if self.use_index:
            if isinstance(index, ABCPeriodIndex):
                # test_mixed_freq_irreg_period
                x = index.to_timestamp()._mpl_repr()
                # TODO: why do we need to do to_timestamp() here but not other
                #  places where we call mpl_repr?
            elif is_any_real_numeric_dtype(index.dtype):
                # Matplotlib supports numeric values or datetime objects as
                # xaxis values. Taking LBYL approach here, by the time
                # matplotlib raises exception when using non numeric/datetime
                # values for xaxis, several actions are already taken by plt.
                x = index._mpl_repr()
            elif isinstance(index, ABCDatetimeIndex) or is_datetype:
                x = index._mpl_repr()
            else:
                self._need_to_set_index = True
                x = list(range(len(index)))
        else:
            x = list(range(len(index)))

        return x

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(
        cls, ax: Axes, x, y: np.ndarray, style=None, is_errorbar: bool = False, **kwds
    ):
        mask = isna(y)
        if mask.any():
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)

        if isinstance(x, ABCIndex):
            x = x._mpl_repr()

        if is_errorbar:
            if "xerr" in kwds:
                kwds["xerr"] = np.array(kwds.get("xerr"))
            if "yerr" in kwds:
                kwds["yerr"] = np.array(kwds.get("yerr"))
            return ax.errorbar(x, y, **kwds)
        else:
            # prevent style kwarg from going to errorbar, where it is unsupported
            args = (x, y, style) if style is not None else (x, y)
            return ax.plot(*args, **kwds)

    def _get_custom_index_name(self):
        """Specify whether xlabel/ylabel should be used to override index name"""
        return self.xlabel

    @final
    def _get_index_name(self) -> str | None:
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ",".join([pprint_thing(x) for x in name])
            else:
                name = None
        else:
            name = self.data.index.name
            if name is not None:
                name = pprint_thing(name)

        # GH 45145, override the default axis label if one is provided.
        index_name = self._get_custom_index_name()
        if index_name is not None:
            name = pprint_thing(index_name)

        return name

    @final
    @classmethod
    def _get_ax_layer(cls, ax, primary: bool = True):
        """get left (primary) or right (secondary) axes"""
        if primary:
            return getattr(ax, "left_ax", ax)
        else:
            return getattr(ax, "right_ax", ax)

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """Return the index of the axis where the column at col_idx should be plotted"""
        if isinstance(self.subplots, list):
            # Subplots is a list: some columns will be grouped together in the same ax
            return next(
                group_idx
                for (group_idx, group) in enumerate(self.subplots)
                if col_idx in group
            )
        else:
            # subplots is True: one ax per column
            return col_idx

    @final
    def _get_ax(self, i: int):
        # get the twinx ax if appropriate
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            ax = self.axes[i]
            ax = self._maybe_right_yaxis(ax, i)
            # error: Unsupported target for indexed assignment ("Sequence[Any]")
            self.axes[i] = ax  # type: ignore[index]
        else:
            ax = self.axes[0]
            ax = self._maybe_right_yaxis(ax, i)

        ax.get_yaxis().set_visible(True)
        return ax

    @final
    def on_right(self, i: int):
        if isinstance(self.secondary_y, bool):
            return self.secondary_y

        if isinstance(self.secondary_y, (tuple, list, np.ndarray, ABCIndex)):
            return self.data.columns[i] in self.secondary_y

    @final
    def _apply_style_colors(
        self, colors, kwds: dict[str, Any], col_num: int, label: str
    ):
        """
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        """
        style = None
        if self.style is not None:
            if isinstance(self.style, list):
                try:
                    style = self.style[col_num]
                except IndexError:
                    pass
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style

        has_color = "color" in kwds or self.colormap is not None
        nocolor_style = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[col_num % len(colors)]
        return style, kwds

    def _get_colors(
        self,
        num_colors: int | None = None,
        color_kwds: str = "color",
    ):
        if num_colors is None:
            num_colors = self.nseries
        if color_kwds == "color":
            color = self.color
        else:
            color = self.kwds.get(color_kwds)
        return get_standard_colors(
            num_colors=num_colors,
            colormap=self.colormap,
            color=color,
        )

    # TODO: tighter typing for first return?
    @final
    @staticmethod
    def _parse_errorbars(
        label: str, err, data: NDFrameT, nseries: int
    ) -> tuple[Any, NDFrameT]:
        """
        Look for error keyword arguments and return the actual errorbar data
        or return the error DataFrame/dict

        Error bars can be specified in several ways:
            Series: the user provides a pandas.Series object of the same
                    length as the data
            ndarray: provides a np.ndarray of the same length as the data
            DataFrame/dict: error values are paired with keys matching the
                    key in the plotted DataFrame
            str: the name of the column within the plotted DataFrame

        Asymmetrical error bars are also supported, however raw error values
        must be provided in this case. For a ``N`` length :class:`Series`, a
        ``2xN`` array should be provided indicating lower and upper (or left
        and right) errors. For a ``MxN`` :class:`DataFrame`, asymmetrical errors
        should be in a ``Mx2xN`` array.
        """
        if err is None:
            return None, data

        def match_labels(data, e):
            e = e.reindex(data.index)
            return e

        # key-matched DataFrame
        if isinstance(err, ABCDataFrame):
            err = match_labels(data, err)
        # key-matched dict
        elif isinstance(err, dict):
            pass

        # Series of error values
        elif isinstance(err, ABCSeries):
            # broadcast error series across data
            err = match_labels(data, err)
            err = np.atleast_2d(err)
            err = np.tile(err, (nseries, 1))

        # errors are a column in the dataframe
        elif isinstance(err, str):
            evalues = data[err].values
            data = data[data.columns.drop(err)]
            err = np.atleast_2d(evalues)
            err = np.tile(err, (nseries, 1))

        elif is_list_like(err):
            if is_iterator(err):
                err = np.atleast_2d(list(err))
            else:
                # raw error values
                err = np.atleast_2d(err)

            err_shape = err.shape

            # asymmetrical error bars
            if isinstance(data, ABCSeries) and err_shape[0] == 2:
                err = np.expand_dims(err, 0)
                err_shape = err.shape
                if err_shape[2] != len(data):
                    raise ValueError(
                        "Asymmetrical error bars should be provided "
                        f"with the shape (2, {len(data)})"
                    )
            elif isinstance(data, ABCDataFrame) and err.ndim == 3:
                if (
                    (err_shape[0] != nseries)
                    or (err_shape[1] != 2)
                    or (err_shape[2] != len(data))
                ):
                    raise ValueError(
                        "Asymmetrical error bars should be provided "
                        f"with the shape ({nseries}, 2, {len(data)})"
                    )

            # broadcast errors to each data series
            if len(err) == 1:
                err = np.tile(err, (nseries, 1))

        elif is_number(err):
            err = np.tile(
                [err],
                (nseries, len(data)),
            )

        else:
            msg = f"No valid {label} detected"
            raise ValueError(msg)

        return err, data

    @final
    def _get_errorbars(
        self, label=None, index=None, xerr: bool = True, yerr: bool = True
    ) -> dict[str, Any]:
        errors = {}

        for kw, flag in zip(["xerr", "yerr"], [xerr, yerr]):
            if flag:
                err = self.errors[kw]
                # user provided label-matched dataframe of errors
                if isinstance(err, (ABCDataFrame, dict)):
                    if label is not None and label in err.keys():
                        err = err[label]
                    else:
                        err = None
                elif index is not None and err is not None:
                    err = err[index]

                if err is not None:
                    errors[kw] = err
        return errors

    @final
    def _get_subplots(self, fig: Figure):
        if Version(mpl.__version__) < Version("3.8"):
            from matplotlib.axes import Subplot as Klass
        else:
            from matplotlib.axes import Axes as Klass

        return [
            ax
            for ax in fig.get_axes()
            if (isinstance(ax, Klass) and ax.get_subplotspec() is not None)
        ]

    @final
    def _get_axes_layout(self, fig: Figure) -> tuple[int, int]:
        axes = self._get_subplots(fig)
        x_set = set()
        y_set = set()
        for ax in axes:
            # check axes coordinates to estimate layout
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))


class PlanePlot(MPLPlot, ABC):
    """
    Abstract class for plotting on plane, currently scatter and hexbin.
    """

    _layout_type = "single"

    def __init__(self, data, x, y, **kwargs) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(self._kind + " requires an x and y column")
        if is_integer(x) and not self.data.columns._holds_integer():
            x = self.data.columns[x]
        if is_integer(y) and not self.data.columns._holds_integer():
            y = self.data.columns[y]

        self.x = x
        self.y = y

    @final
    def _get_nseries(self, data: Series | DataFrame) -> int:
        return 1

    @final
    def _post_plot_logic(self, ax: Axes, data) -> None:
        x, y = self.x, self.y
        xlabel = self.xlabel if self.xlabel is not None else pprint_thing(x)
        ylabel = self.ylabel if self.ylabel is not None else pprint_thing(y)
        # error: Argument 1 to "set_xlabel" of "_AxesBase" has incompatible
        # type "Hashable"; expected "str"
        ax.set_xlabel(xlabel)  # type: ignore[arg-type]
        ax.set_ylabel(ylabel)  # type: ignore[arg-type]

    @final
    def _plot_colorbar(self, ax: Axes, *, fig: Figure, **kwds):
        # Addresses issues #10611 and #10678:
        # When plotting scatterplots and hexbinplots in IPython
        # inline backend the colorbar axis height tends not to
        # exactly match the parent axis height.
        # The difference is due to small fractional differences
        # in floating points with similar representation.
        # To deal with this, this method forces the colorbar
        # height to take the height of the parent axes.
        # For a more detailed description of the issue
        # see the following link:
        # https://github.com/ipython/ipython/issues/11215

        # GH33389, if ax is used multiple times, we should always
        # use the last one which contains the latest information
        # about the ax
        img = ax.collections[-1]
        return fig.colorbar(img, ax=ax, **kwds)


class ScatterPlot(PlanePlot):
    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    def __init__(
        self,
        data,
        x,
        y,
        s=None,
        c=None,
        *,
        colorbar: bool | lib.NoDefault = lib.no_default,
        norm=None,
        **kwargs,
    ) -> None:
        if s is None:
            # hide the matplotlib default for size, in case we want to change
            # the handling of this argument later
            s = 20
        elif is_hashable(s) and s in data.columns:
            s = data[s]
        self.s = s

        self.colorbar = colorbar
        self.norm = norm

        super().__init__(data, x, y, **kwargs)
        if is_integer(c) and not self.data.columns._holds_integer():
            c = self.data.columns[c]
        self.c = c

    def _make_plot(self, fig: Figure) -> None:
        x, y, c, data = self.x, self.y, self.c, self.data
        ax = self.axes[0]

        c_is_column = is_hashable(c) and c in self.data.columns

        color_by_categorical = c_is_column and isinstance(
            self.data[c].dtype, CategoricalDtype
        )

        color = self.color
        c_values = self._get_c_values(color, color_by_categorical, c_is_column)
        norm, cmap = self._get_norm_and_cmap(c_values, color_by_categorical)
        cb = self._get_colorbar(c_values, c_is_column)

        if self.legend:
            label = self.label
        else:
            label = None
        scatter = ax.scatter(
            data[x].values,
            data[y].values,
            c=c_values,
            label=label,
            cmap=cmap,
            norm=norm,
            s=self.s,
            **self.kwds,
        )
        if cb:
            cbar_label = c if c_is_column else ""
            cbar = self._plot_colorbar(ax, fig=fig, label=cbar_label)
            if color_by_categorical:
                n_cats = len(self.data[c].cat.categories)
                cbar.set_ticks(np.linspace(0.5, n_cats - 0.5, n_cats))
                cbar.ax.set_yticklabels(self.data[c].cat.categories)

        if label is not None:
            self._append_legend_handles_labels(
                # error: Argument 2 to "_append_legend_handles_labels" of
                # "MPLPlot" has incompatible type "Hashable"; expected "str"
                scatter,
                label,  # type: ignore[arg-type]
            )

        errors_x = self._get_errorbars(label=x, index=0, yerr=False)
        errors_y = self._get_errorbars(label=y, index=0, xerr=False)
        if len(errors_x) > 0 or len(errors_y) > 0:
            err_kwds = dict(errors_x, **errors_y)
            err_kwds["ecolor"] = scatter.get_facecolor()[0]
            ax.errorbar(data[x].values, data[y].values, linestyle="none", **err_kwds)

    def _get_c_values(self, color, color_by_categorical: bool, c_is_column: bool):
        c = self.c
        if c is not None and color is not None:
            raise TypeError("Specify exactly one of `c` and `color`")
        if c is None and color is None:
            c_values = self.plt.rcParams["patch.facecolor"]
        elif color is not None:
            c_values = color
        elif color_by_categorical:
            c_values = self.data[c].cat.codes
        elif c_is_column:
            c_values = self.data[c].values
        else:
            c_values = c
        return c_values

    def _get_norm_and_cmap(self, c_values, color_by_categorical: bool):
        c = self.c
        if self.colormap is not None:
            cmap = mpl.colormaps.get_cmap(self.colormap)
        # cmap is only used if c_values are integers, otherwise UserWarning.
        # GH-53908: additionally call isinstance() because is_integer_dtype
        # returns True for "b" (meaning "blue" and not int8 in this context)
        elif not isinstance(c_values, str) and is_integer_dtype(c_values):
            # pandas uses colormap, matplotlib uses cmap.
            cmap = mpl.colormaps["Greys"]
        else:
            cmap = None

        if color_by_categorical and cmap is not None:
            from matplotlib import colors

            n_cats = len(self.data[c].cat.categories)
            cmap = colors.ListedColormap([cmap(i) for i in range(cmap.N)])
            bounds = np.linspace(0, n_cats, n_cats + 1)
            norm = colors.BoundaryNorm(bounds, cmap.N)
            # TODO: warn that we are ignoring self.norm if user specified it?
            #  Doesn't happen in any tests 2023-11-09
        else:
            norm = self.norm
        return norm, cmap

    def _get_colorbar(self, c_values, c_is_column: bool) -> bool:
        # plot colorbar if
        # 1. colormap is assigned, and
        # 2.`c` is a column containing only numeric values
        plot_colorbar = self.colormap or c_is_column
        cb = self.colorbar
        if cb is lib.no_default:
            return is_numeric_dtype(c_values) and plot_colorbar
        return cb


class HexBinPlot(PlanePlot):
    @property
    def _kind(self) -> Literal["hexbin"]:
        return "hexbin"

    def __init__(self, data, x, y, C=None, *, colorbar: bool = True, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)
        if is_integer(C) and not self.data.columns._holds_integer():
            C = self.data.columns[C]
        self.C = C

        self.colorbar = colorbar

        # Scatter plot allows to plot objects data
        if len(self.data[self.x]._get_numeric_data()) == 0:
            raise ValueError(self._kind + " requires x column to be numeric")
        if len(self.data[self.y]._get_numeric_data()) == 0:
            raise ValueError(self._kind + " requires y column to be numeric")

    def _make_plot(self, fig: Figure) -> None:
        x, y, data, C = self.x, self.y, self.data, self.C
        ax = self.axes[0]
        # pandas uses colormap, matplotlib uses cmap.
        cmap = self.colormap or "BuGn"
        cmap = mpl.colormaps.get_cmap(cmap)
        cb = self.colorbar

        if C is None:
            c_values = None
        else:
            c_values = data[C].values

        ax.hexbin(data[x].values, data[y].values, C=c_values, cmap=cmap, **self.kwds)
        if cb:
            self._plot_colorbar(ax, fig=fig)

    def _make_legend(self) -> None:
        pass


class LinePlot(MPLPlot):
    _default_rot = 0

    @property
    def orientation(self) -> PlottingOrientation:
        return "vertical"

    @property
    def _kind(self) -> Literal["line", "area", "hist", "kde", "box"]:
        return "line"

    def __init__(self, data, **kwargs) -> None:
        from pandas.plotting import plot_params

        MPLPlot.__init__(self, data, **kwargs)
        if self.stacked:
            self.data = self.data.fillna(value=0)
        self.x_compat = plot_params["x_compat"]
        if "x_compat" in self.kwds:
            self.x_compat = bool(self.kwds.pop("x_compat"))

    @final
    def _is_ts_plot(self) -> bool:
        # this is slightly deceptive
        return not self.x_compat and self.use_index and self._use_dynamic_x()

    @final
    def _use_dynamic_x(self) -> bool:
        return use_dynamic_x(self._get_ax(0), self.data)

    def _make_plot(self, fig: Figure) -> None:
        if self._is_ts_plot():
            data = maybe_convert_index(self._get_ax(0), self.data)

            x = data.index  # dummy, not used
            plotf = self._ts_plot
            it = data.items()
        else:
            x = self._get_xticks()
            # error: Incompatible types in assignment (expression has type
            # "Callable[[Any, Any, Any, Any, Any, Any, KwArg(Any)], Any]", variable has
            # type "Callable[[Any, Any, Any, Any, KwArg(Any)], Any]")
            plotf = self._plot  # type: ignore[assignment]
            # error: Incompatible types in assignment (expression has type
            # "Iterator[tuple[Hashable, ndarray[Any, Any]]]", variable has
            # type "Iterable[tuple[Hashable, Series]]")
            it = self._iter_data(data=self.data)  # type: ignore[assignment]

        stacking_id = self._get_stacking_id()
        is_errorbar = com.any_not_none(*self.errors.values())

        colors = self._get_colors()
        for i, (label, y) in enumerate(it):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self.color is not None:
                kwds["color"] = self.color
            style, kwds = self._apply_style_colors(
                colors,
                kwds,
                i,
                # error: Argument 4 to "_apply_style_colors" of "MPLPlot" has
                # incompatible type "Hashable"; expected "str"
                label,  # type: ignore[arg-type]
            )

            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)

            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label

            newlines = plotf(
                ax,
                x,
                y,
                style=style,
                column_num=i,
                stacking_id=stacking_id,
                is_errorbar=is_errorbar,
                **kwds,
            )
            self._append_legend_handles_labels(newlines[0], label)

            if self._is_ts_plot():
                # reset of xlim should be used for ts data
                # TODO: GH28021, should find a way to change view limit on xaxis
                lines = get_all_lines(ax)
                left, right = get_xlim(lines)
                ax.set_xlim(left, right)

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        style=None,
        column_num=None,
        stacking_id=None,
        **kwds,
    ):
        # column_num is used to get the target column from plotf in line and
        # area plots
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
        lines = MPLPlot._plot(ax, x, y_values, style=style, **kwds)
        cls._update_stacker(ax, stacking_id, y)
        return lines

    @final
    def _ts_plot(self, ax: Axes, x, data: Series, style=None, **kwds):
        # accept x to be consistent with normal plot func,
        # x is not passed to tsplot as it uses data.index as x coordinate
        # column_num must be in kwds for stacking purpose
        freq, data = maybe_resample(data, ax, kwds)

        # Set ax with freq info
        decorate_axes(ax, freq)
        # digging deeper
        if hasattr(ax, "left_ax"):
            decorate_axes(ax.left_ax, freq)
        if hasattr(ax, "right_ax"):
            decorate_axes(ax.right_ax, freq)
        # TODO #54485
        ax._plot_data.append((data, self._kind, kwds))  # type: ignore[attr-defined]

        lines = self._plot(ax, data.index, np.asarray(data.values), style=style, **kwds)
        # set date formatter, locators and rescale limits
        # TODO #54485
        format_dateaxis(ax, ax.freq, data.index)  # type: ignore[arg-type, attr-defined]
        return lines

    @final
    def _get_stacking_id(self) -> int | None:
        if self.stacked:
            return id(self.data)
        else:
            return None

    @final
    @classmethod
    def _initialize_stacker(cls, ax: Axes, stacking_id, n: int) -> None:
        if stacking_id is None:
            return
        if not hasattr(ax, "_stacker_pos_prior"):
            # TODO #54485
            ax._stacker_pos_prior = {}  # type: ignore[attr-defined]
        if not hasattr(ax, "_stacker_neg_prior"):
            # TODO #54485
            ax._stacker_neg_prior = {}  # type: ignore[attr-defined]
        # TODO #54485
        ax._stacker_pos_prior[stacking_id] = np.zeros(n)  # type: ignore[attr-defined]
        # TODO #54485
        ax._stacker_neg_prior[stacking_id] = np.zeros(n)  # type: ignore[attr-defined]

    @final
    @classmethod
    def _get_stacked_values(
        cls, ax: Axes, stacking_id: int | None, values: np.ndarray, label
    ) -> np.ndarray:
        if stacking_id is None:
            return values
        if not hasattr(ax, "_stacker_pos_prior"):
            # stacker may not be initialized for subplots
            cls._initialize_stacker(ax, stacking_id, len(values))

        if (values >= 0).all():
            # TODO #54485
            return (
                ax._stacker_pos_prior[stacking_id]  # type: ignore[attr-defined]
                + values
            )
        elif (values <= 0).all():
            # TODO #54485
            return (
                ax._stacker_neg_prior[stacking_id]  # type: ignore[attr-defined]
                + values
            )

        raise ValueError(
            "When stacked is True, each column must be either "
            "all positive or all negative. "
            f"Column '{label}' contains both positive and negative values"
        )

    @final
    @classmethod
    def _update_stacker(cls, ax: Axes, stacking_id: int | None, values) -> None:
        if stacking_id is None:
            return
        if (values >= 0).all():
            # TODO #54485
            ax._stacker_pos_prior[stacking_id] += values  # type: ignore[attr-defined]
        elif (values <= 0).all():
            # TODO #54485
            ax._stacker_neg_prior[stacking_id] += values  # type: ignore[attr-defined]

    def _post_plot_logic(self, ax: Axes, data) -> None:
        from matplotlib.ticker import FixedLocator

        def get_label(i):
            if is_float(i) and i.is_integer():
                i = int(i)
            try:
                return pprint_thing(data.index[i])
            except Exception:
                return ""

        if self._need_to_set_index:
            xticks = ax.get_xticks()
            xticklabels = [get_label(x) for x in xticks]
            # error: Argument 1 to "FixedLocator" has incompatible type "ndarray[Any,
            # Any]"; expected "Sequence[float]"
            ax.xaxis.set_major_locator(FixedLocator(xticks))  # type: ignore[arg-type]
            ax.set_xticklabels(xticklabels)

        # If the index is an irregular time series, then by default
        # we rotate the tick labels. The exception is if there are
        # subplots which don't share their x-axes, in which we case
        # we don't rotate the ticklabels as by default the subplots
        # would be too close together.
        condition = (
            not self._use_dynamic_x()
            and (data.index._is_all_dates and self.use_index)
            and (not self.subplots or (self.subplots and self.sharex))
        )

        index_name = self._get_index_name()

        if condition:
            # irregular TS rotated 30 deg. by default
            # probably a better place to check / set this.
            if not self._rot_set:
                self.rot = 30
            format_date_labels(ax, rot=self.rot)

        if index_name is not None and self.use_index:
            ax.set_xlabel(index_name)


class AreaPlot(LinePlot):
    @property
    def _kind(self) -> Literal["area"]:
        return "area"

    def __init__(self, data, **kwargs) -> None:
        kwargs.setdefault("stacked", True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Downcasting object dtype arrays",
                category=FutureWarning,
            )
            data = data.fillna(value=0)
        LinePlot.__init__(self, data, **kwargs)

        if not self.stacked:
            # use smaller alpha to distinguish overlap
            self.kwds.setdefault("alpha", 0.5)

        if self.logy or self.loglog:
            raise ValueError("Log-y scales are not supported in area plot")

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        style=None,
        column_num=None,
        stacking_id=None,
        is_errorbar: bool = False,
        **kwds,
    ):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])

        # need to remove label, because subplots uses mpl legend as it is
        line_kwds = kwds.copy()
        line_kwds.pop("label")
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)

        # get data from the line to get coordinates for fill_between
        xdata, y_values = lines[0].get_data(orig=False)

        # unable to use ``_get_stacked_values`` here to get starting point
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            # TODO #54485
            start = ax._stacker_pos_prior[stacking_id]  # type: ignore[attr-defined]
        elif (y <= 0).all():
            # TODO #54485
            start = ax._stacker_neg_prior[stacking_id]  # type: ignore[attr-defined]
        else:
            start = np.zeros(len(y))

        if "color" not in kwds:
            kwds["color"] = lines[0].get_color()

        rect = ax.fill_between(xdata, start, y_values, **kwds)
        cls._update_stacker(ax, stacking_id, y)

        # LinePlot expects list of artists
        res = [rect]
        return res

    def _post_plot_logic(self, ax: Axes, data) -> None:
        LinePlot._post_plot_logic(self, ax, data)

        is_shared_y = len(list(ax.get_shared_y_axes())) > 0
        # do not override the default axis behaviour in case of shared y axes
        if self.ylim is None and not is_shared_y:
            if (data >= 0).all().all():
                ax.set_ylim(0, None)
            elif (data <= 0).all().all():
                ax.set_ylim(None, 0)


class BarPlot(MPLPlot):
    @property
    def _kind(self) -> Literal["bar", "barh"]:
        return "bar"

    _default_rot = 90

    @property
    def orientation(self) -> PlottingOrientation:
        return "vertical"

    def __init__(
        self,
        data,
        *,
        align="center",
        bottom=0,
        left=0,
        width=0.5,
        position=0.5,
        log=False,
        **kwargs,
    ) -> None:
        # we have to treat a series differently than a
        # 1-column DataFrame w.r.t. color handling
        self._is_series = isinstance(data, ABCSeries)
        self.bar_width = width
        self._align = align
        self._position = position
        self.tick_pos = np.arange(len(data))

        if is_list_like(bottom):
            bottom = np.array(bottom)
        if is_list_like(left):
            left = np.array(left)
        self.bottom = bottom
        self.left = left

        self.log = log

        MPLPlot.__init__(self, data, **kwargs)

    @cache_readonly
    def ax_pos(self) -> np.ndarray:
        return self.tick_pos - self.tickoffset

    @cache_readonly
    def tickoffset(self):
        if self.stacked or self.subplots:
            return self.bar_width * self._position
        elif self._align == "edge":
            w = self.bar_width / self.nseries
            return self.bar_width * (self._position - 0.5) + w * 0.5
        else:
            return self.bar_width * self._position

    @cache_readonly
    def lim_offset(self):
        if self.stacked or self.subplots:
            if self._align == "edge":
                return self.bar_width / 2
            else:
                return 0
        elif self._align == "edge":
            w = self.bar_width / self.nseries
            return w * 0.5
        else:
            return 0

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        w,
        start: int | npt.NDArray[np.intp] = 0,
        log: bool = False,
        **kwds,
    ):
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)

    @property
    def _start_base(self):
        return self.bottom

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors()
        ncolors = len(colors)

        pos_prior = neg_prior = np.zeros(len(self.data))
        K = self.nseries

        data = self.data.fillna(0)
        for i, (label, y) in enumerate(self._iter_data(data=data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self._is_series:
                kwds["color"] = colors
            elif isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[i % ncolors]

            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)

            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)

            if (("yerr" in kwds) or ("xerr" in kwds)) and (kwds.get("ecolor") is None):
                kwds["ecolor"] = mpl.rcParams["xtick.color"]

            start = 0
            if self.log and (y >= 1).all():
                start = 1
            start = start + self._start_base

            kwds["align"] = self._align
            if self.subplots:
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                ax.set_title(label)
            elif self.stacked:
                mask = y > 0
                start = np.where(mask, pos_prior, neg_prior) + self._start_base
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                pos_prior = pos_prior + np.where(mask, y, 0)
                neg_prior = neg_prior + np.where(mask, 0, y)
            else:
                w = self.bar_width / K
                rect = self._plot(
                    ax,
                    self.ax_pos + (i + 0.5) * w,
                    y,
                    w,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
            self._append_legend_handles_labels(rect, label)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        if self.use_index:
            str_index = [pprint_thing(key) for key in data.index]
        else:
            str_index = [pprint_thing(key) for key in range(data.shape[0])]

        s_edge = self.ax_pos[0] - 0.25 + self.lim_offset
        e_edge = self.ax_pos[-1] + 0.25 + self.bar_width + self.lim_offset

        self._decorate_ticks(ax, self._get_index_name(), str_index, s_edge, e_edge)

    def _decorate_ticks(
        self,
        ax: Axes,
        name: str | None,
        ticklabels: list[str],
        start_edge: float,
        end_edge: float,
    ) -> None:
        ax.set_xlim((start_edge, end_edge))

        if self.xticks is not None:
            ax.set_xticks(np.array(self.xticks))
        else:
            ax.set_xticks(self.tick_pos)
            ax.set_xticklabels(ticklabels)

        if name is not None and self.use_index:
            ax.set_xlabel(name)


class BarhPlot(BarPlot):
    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

    _default_rot = 0

    @property
    def orientation(self) -> Literal["horizontal"]:
        return "horizontal"

    @property
    def _start_base(self):
        return self.left

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        w,
        start: int | npt.NDArray[np.intp] = 0,
        log: bool = False,
        **kwds,
    ):
        return ax.barh(x, y, w, left=start, log=log, **kwds)

    def _get_custom_index_name(self):
        return self.ylabel

    def _decorate_ticks(
        self,
        ax: Axes,
        name: str | None,
        ticklabels: list[str],
        start_edge: float,
        end_edge: float,
    ) -> None:
        # horizontal bars
        ax.set_ylim((start_edge, end_edge))
        ax.set_yticks(self.tick_pos)
        ax.set_yticklabels(ticklabels)
        if name is not None and self.use_index:
            ax.set_ylabel(name)
        # error: Argument 1 to "set_xlabel" of "_AxesBase" has incompatible type
        # "Hashable | None"; expected "str"
        ax.set_xlabel(self.xlabel)  # type: ignore[arg-type]


class PiePlot(MPLPlot):
    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    _layout_type = "horizontal"

    def __init__(self, data, kind=None, **kwargs) -> None:
        data = data.fillna(value=0)
        if (data < 0).any().any():
            raise ValueError(f"{self._kind} plot doesn't allow negative values")
        MPLPlot.__init__(self, data, kind=kind, **kwargs)

    @classmethod
    def _validate_log_kwd(
        cls,
        kwd: str,
        value: bool | None | Literal["sym"],
    ) -> bool | None | Literal["sym"]:
        super()._validate_log_kwd(kwd=kwd, value=value)
        if value is not False:
            warnings.warn(
                f"PiePlot ignores the '{kwd}' keyword",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        return False

    def _validate_color_args(self, color, colormap) -> None:
        # TODO: warn if color is passed and ignored?
        return None

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors(num_colors=len(self.data), color_kwds="colors")
        self.kwds.setdefault("colors", colors)

        for i, (label, y) in enumerate(self._iter_data(data=self.data)):
            ax = self._get_ax(i)
            if label is not None:
                label = pprint_thing(label)
                ax.set_ylabel(label)

            kwds = self.kwds.copy()

            def blank_labeler(label, value):
                if value == 0:
                    return ""
                else:
                    return label

            idx = [pprint_thing(v) for v in self.data.index]
            labels = kwds.pop("labels", idx)
            # labels is used for each wedge's labels
            # Blank out labels for values of 0 so they don't overlap
            # with nonzero wedges
            if labels is not None:
                blabels = [blank_labeler(left, value) for left, value in zip(labels, y)]
            else:
                blabels = None
            results = ax.pie(y, labels=blabels, **kwds)

            if kwds.get("autopct", None) is not None:
                patches, texts, autotexts = results
            else:
                patches, texts = results
                autotexts = []

            if self.fontsize is not None:
                for t in texts + autotexts:
                    t.set_fontsize(self.fontsize)

            # leglabels is used for legend labels
            leglabels = labels if labels is not None else idx
            for _patch, _leglabel in zip(patches, leglabels):
                self._append_legend_handles_labels(_patch, _leglabel)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\bulk_persistence.py ===
# orm/bulk_persistence.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""additional ORM persistence classes related to "bulk" operations,
specifically outside of the flush() process.

"""

from __future__ import annotations

from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal

if TYPE_CHECKING:
    from ._typing import DMLStrategyArgument
    from ._typing import OrmExecuteOptionsParameter
    from ._typing import SynchronizeSessionArgument
    from .mapper import Mapper
    from .session import _BindArguments
    from .session import ORMExecuteState
    from .session import Session
    from .session import SessionTransaction
    from .state import InstanceState
    from ..engine import Connection
    from ..engine import cursor
    from ..engine.interfaces import _CoreAnyExecuteParams

_O = TypeVar("_O", bound=object)


@overload
def _bulk_insert(
    mapper: Mapper[_O],
    mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]],
    session_transaction: SessionTransaction,
    *,
    isstates: bool,
    return_defaults: bool,
    render_nulls: bool,
    use_orm_insert_stmt: Literal[None] = ...,
    execution_options: Optional[OrmExecuteOptionsParameter] = ...,
) -> None: ...


@overload
def _bulk_insert(
    mapper: Mapper[_O],
    mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]],
    session_transaction: SessionTransaction,
    *,
    isstates: bool,
    return_defaults: bool,
    render_nulls: bool,
    use_orm_insert_stmt: Optional[dml.Insert] = ...,
    execution_options: Optional[OrmExecuteOptionsParameter] = ...,
) -> cursor.CursorResult[Any]: ...


def _bulk_insert(
    mapper: Mapper[_O],
    mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]],
    session_transaction: SessionTransaction,
    *,
    isstates: bool,
    return_defaults: bool,
    render_nulls: bool,
    use_orm_insert_stmt: Optional[dml.Insert] = None,
    execution_options: Optional[OrmExecuteOptionsParameter] = None,
) -> Optional[cursor.CursorResult[Any]]:
    base_mapper = mapper.base_mapper

    if session_transaction.session.connection_callable:
        raise NotImplementedError(
            "connection_callable / per-instance sharding "
            "not supported in bulk_insert()"
        )

    if isstates:
        if TYPE_CHECKING:
            mappings = cast(Iterable[InstanceState[_O]], mappings)

        if return_defaults:
            # list of states allows us to attach .key for return_defaults case
            states = [(state, state.dict) for state in mappings]
            mappings = [dict_ for (state, dict_) in states]
        else:
            mappings = [state.dict for state in mappings]
    else:
        if TYPE_CHECKING:
            mappings = cast(Iterable[Dict[str, Any]], mappings)

        if return_defaults:
            # use dictionaries given, so that newly populated defaults
            # can be delivered back to the caller (see #11661). This is **not**
            # compatible with other use cases such as a session-executed
            # insert() construct, as this will confuse the case of
            # insert-per-subclass for joined inheritance cases (see
            # test_bulk_statements.py::BulkDMLReturningJoinedInhTest).
            #
            # So in this conditional, we have **only** called
            # session.bulk_insert_mappings() which does not have this
            # requirement
            mappings = list(mappings)
        else:
            # for all other cases we need to establish a local dictionary
            # so that the incoming dictionaries aren't mutated
            mappings = [dict(m) for m in mappings]
        _expand_composites(mapper, mappings)

    connection = session_transaction.connection(base_mapper)

    return_result: Optional[cursor.CursorResult[Any]] = None

    mappers_to_run = [
        (table, mp)
        for table, mp in base_mapper._sorted_tables.items()
        if table in mapper._pks_by_table
    ]

    if return_defaults:
        # not used by new-style bulk inserts, only used for legacy
        bookkeeping = True
    elif len(mappers_to_run) > 1:
        # if we have more than one table, mapper to run where we will be
        # either horizontally splicing, or copying values between tables,
        # we need the "bookkeeping" / deterministic returning order
        bookkeeping = True
    else:
        bookkeeping = False

    for table, super_mapper in mappers_to_run:
        # find bindparams in the statement. For bulk, we don't really know if
        # a key in the params applies to a different table since we are
        # potentially inserting for multiple tables here; looking at the
        # bindparam() is a lot more direct.   in most cases this will
        # use _generate_cache_key() which is memoized, although in practice
        # the ultimate statement that's executed is probably not the same
        # object so that memoization might not matter much.
        extra_bp_names = (
            [
                b.key
                for b in use_orm_insert_stmt._get_embedded_bindparams()
                if b.key in mappings[0]
            ]
            if use_orm_insert_stmt is not None
            else ()
        )

        records = (
            (
                None,
                state_dict,
                params,
                mapper,
                connection,
                value_params,
                has_all_pks,
                has_all_defaults,
            )
            for (
                state,
                state_dict,
                params,
                mp,
                conn,
                value_params,
                has_all_pks,
                has_all_defaults,
            ) in persistence._collect_insert_commands(
                table,
                ((None, mapping, mapper, connection) for mapping in mappings),
                bulk=True,
                return_defaults=bookkeeping,
                render_nulls=render_nulls,
                include_bulk_keys=extra_bp_names,
            )
        )

        result = persistence._emit_insert_statements(
            base_mapper,
            None,
            super_mapper,
            table,
            records,
            bookkeeping=bookkeeping,
            use_orm_insert_stmt=use_orm_insert_stmt,
            execution_options=execution_options,
        )
        if use_orm_insert_stmt is not None:
            if not use_orm_insert_stmt._returning or return_result is None:
                return_result = result
            elif result.returns_rows:
                assert bookkeeping
                return_result = return_result.splice_horizontally(result)

    if return_defaults and isstates:
        identity_cls = mapper._identity_class
        identity_props = [p.key for p in mapper._identity_key_props]
        for state, dict_ in states:
            state.key = (
                identity_cls,
                tuple([dict_[key] for key in identity_props]),
                None,
            )

    if use_orm_insert_stmt is not None:
        assert return_result is not None
        return return_result


@overload
def _bulk_update(
    mapper: Mapper[Any],
    mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]],
    session_transaction: SessionTransaction,
    *,
    isstates: bool,
    update_changed_only: bool,
    use_orm_update_stmt: Literal[None] = ...,
    enable_check_rowcount: bool = True,
) -> None: ...


@overload
def _bulk_update(
    mapper: Mapper[Any],
    mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]],
    session_transaction: SessionTransaction,
    *,
    isstates: bool,
    update_changed_only: bool,
    use_orm_update_stmt: Optional[dml.Update] = ...,
    enable_check_rowcount: bool = True,
) -> _result.Result[Any]: ...


def _bulk_update(
    mapper: Mapper[Any],
    mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]],
    session_transaction: SessionTransaction,
    *,
    isstates: bool,
    update_changed_only: bool,
    use_orm_update_stmt: Optional[dml.Update] = None,
    enable_check_rowcount: bool = True,
) -> Optional[_result.Result[Any]]:
    base_mapper = mapper.base_mapper

    search_keys = mapper._primary_key_propkeys
    if mapper._version_id_prop:
        search_keys = {mapper._version_id_prop.key}.union(search_keys)

    def _changed_dict(mapper, state):
        return {
            k: v
            for k, v in state.dict.items()
            if k in state.committed_state or k in search_keys
        }

    if isstates:
        if update_changed_only:
            mappings = [_changed_dict(mapper, state) for state in mappings]
        else:
            mappings = [state.dict for state in mappings]
    else:
        mappings = [dict(m) for m in mappings]
        _expand_composites(mapper, mappings)

    if session_transaction.session.connection_callable:
        raise NotImplementedError(
            "connection_callable / per-instance sharding "
            "not supported in bulk_update()"
        )

    connection = session_transaction.connection(base_mapper)

    # find bindparams in the statement. see _bulk_insert for similar
    # notes for the insert case
    extra_bp_names = (
        [
            b.key
            for b in use_orm_update_stmt._get_embedded_bindparams()
            if b.key in mappings[0]
        ]
        if use_orm_update_stmt is not None
        else ()
    )

    for table, super_mapper in base_mapper._sorted_tables.items():
        if not mapper.isa(super_mapper) or table not in mapper._pks_by_table:
            continue

        records = persistence._collect_update_commands(
            None,
            table,
            (
                (
                    None,
                    mapping,
                    mapper,
                    connection,
                    (
                        mapping[mapper._version_id_prop.key]
                        if mapper._version_id_prop
                        else None
                    ),
                )
                for mapping in mappings
            ),
            bulk=True,
            use_orm_update_stmt=use_orm_update_stmt,
            include_bulk_keys=extra_bp_names,
        )
        persistence._emit_update_statements(
            base_mapper,
            None,
            super_mapper,
            table,
            records,
            bookkeeping=False,
            use_orm_update_stmt=use_orm_update_stmt,
            enable_check_rowcount=enable_check_rowcount,
        )

    if use_orm_update_stmt is not None:
        return _result.null_result()


def _expand_composites(mapper, mappings):
    composite_attrs = mapper.composites
    if not composite_attrs:
        return

    composite_keys = set(composite_attrs.keys())
    populators = {
        key: composite_attrs[key]._populate_composite_bulk_save_mappings_fn()
        for key in composite_keys
    }
    for mapping in mappings:
        for key in composite_keys.intersection(mapping):
            populators[key](mapping)


class ORMDMLState(AbstractORMCompileState):
    is_dml_returning = True
    from_statement_ctx: Optional[ORMFromStatementCompileState] = None

    @classmethod
    def _get_orm_crud_kv_pairs(
        cls, mapper, statement, kv_iterator, needs_to_be_cacheable
    ):
        core_get_crud_kv_pairs = UpdateDMLState._get_crud_kv_pairs

        for k, v in kv_iterator:
            k = coercions.expect(roles.DMLColumnRole, k)

            if isinstance(k, str):
                desc = _entity_namespace_key(mapper, k, default=NO_VALUE)
                if desc is NO_VALUE:
                    yield (
                        coercions.expect(roles.DMLColumnRole, k),
                        (
                            coercions.expect(
                                roles.ExpressionElementRole,
                                v,
                                type_=sqltypes.NullType(),
                                is_crud=True,
                            )
                            if needs_to_be_cacheable
                            else v
                        ),
                    )
                else:
                    yield from core_get_crud_kv_pairs(
                        statement,
                        desc._bulk_update_tuples(v),
                        needs_to_be_cacheable,
                    )
            elif "entity_namespace" in k._annotations:
                k_anno = k._annotations
                attr = _entity_namespace_key(
                    k_anno["entity_namespace"], k_anno["proxy_key"]
                )
                yield from core_get_crud_kv_pairs(
                    statement,
                    attr._bulk_update_tuples(v),
                    needs_to_be_cacheable,
                )
            else:
                yield (
                    k,
                    (
                        v
                        if not needs_to_be_cacheable
                        else coercions.expect(
                            roles.ExpressionElementRole,
                            v,
                            type_=sqltypes.NullType(),
                            is_crud=True,
                        )
                    ),
                )

    @classmethod
    def _get_multi_crud_kv_pairs(cls, statement, kv_iterator):
        plugin_subject = statement._propagate_attrs["plugin_subject"]

        if not plugin_subject or not plugin_subject.mapper:
            return UpdateDMLState._get_multi_crud_kv_pairs(
                statement, kv_iterator
            )

        return [
            dict(
                cls._get_orm_crud_kv_pairs(
                    plugin_subject.mapper, statement, value_dict.items(), False
                )
            )
            for value_dict in kv_iterator
        ]

    @classmethod
    def _get_crud_kv_pairs(cls, statement, kv_iterator, needs_to_be_cacheable):
        assert (
            needs_to_be_cacheable
        ), "no test coverage for needs_to_be_cacheable=False"

        plugin_subject = statement._propagate_attrs["plugin_subject"]

        if not plugin_subject or not plugin_subject.mapper:
            return UpdateDMLState._get_crud_kv_pairs(
                statement, kv_iterator, needs_to_be_cacheable
            )

        return list(
            cls._get_orm_crud_kv_pairs(
                plugin_subject.mapper,
                statement,
                kv_iterator,
                needs_to_be_cacheable,
            )
        )

    @classmethod
    def get_entity_description(cls, statement):
        ext_info = statement.table._annotations["parententity"]
        mapper = ext_info.mapper
        if ext_info.is_aliased_class:
            _label_name = ext_info.name
        else:
            _label_name = mapper.class_.__name__

        return {
            "name": _label_name,
            "type": mapper.class_,
            "expr": ext_info.entity,
            "entity": ext_info.entity,
            "table": mapper.local_table,
        }

    @classmethod
    def get_returning_column_descriptions(cls, statement):
        def _ent_for_col(c):
            return c._annotations.get("parententity", None)

        def _attr_for_col(c, ent):
            if ent is None:
                return c
            proxy_key = c._annotations.get("proxy_key", None)
            if not proxy_key:
                return c
            else:
                return getattr(ent.entity, proxy_key, c)

        return [
            {
                "name": c.key,
                "type": c.type,
                "expr": _attr_for_col(c, ent),
                "aliased": ent.is_aliased_class,
                "entity": ent.entity,
            }
            for c, ent in [
                (c, _ent_for_col(c)) for c in statement._all_selected_columns
            ]
        ]

    def _setup_orm_returning(
        self,
        compiler,
        orm_level_statement,
        dml_level_statement,
        dml_mapper,
        *,
        use_supplemental_cols=True,
    ):
        """establish ORM column handlers for an INSERT, UPDATE, or DELETE
        which uses explicit returning().

        called within compilation level create_for_statement.

        The _return_orm_returning() method then receives the Result
        after the statement was executed, and applies ORM loading to the
        state that we first established here.

        """

        if orm_level_statement._returning:
            fs = FromStatement(
                orm_level_statement._returning,
                dml_level_statement,
                _adapt_on_names=False,
            )
            fs = fs.execution_options(**orm_level_statement._execution_options)
            fs = fs.options(*orm_level_statement._with_options)
            self.select_statement = fs
            self.from_statement_ctx = fsc = (
                ORMFromStatementCompileState.create_for_statement(fs, compiler)
            )
            fsc.setup_dml_returning_compile_state(dml_mapper)

            dml_level_statement = dml_level_statement._generate()
            dml_level_statement._returning = ()

            cols_to_return = [c for c in fsc.primary_columns if c is not None]

            # since we are splicing result sets together, make sure there
            # are columns of some kind returned in each result set
            if not cols_to_return:
                cols_to_return.extend(dml_mapper.primary_key)

            if use_supplemental_cols:
                dml_level_statement = dml_level_statement.return_defaults(
                    # this is a little weird looking, but by passing
                    # primary key as the main list of cols, this tells
                    # return_defaults to omit server-default cols (and
                    # actually all cols, due to some weird thing we should
                    # clean up in crud.py).
                    # Since we have cols_to_return, just return what we asked
                    # for (plus primary key, which ORM persistence needs since
                    # we likely set bookkeeping=True here, which is another
                    # whole thing...).   We dont want to clutter the
                    # statement up with lots of other cols the user didn't
                    # ask for.  see #9685
                    *dml_mapper.primary_key,
                    supplemental_cols=cols_to_return,
                )
            else:
                dml_level_statement = dml_level_statement.returning(
                    *cols_to_return
                )

        return dml_level_statement

    @classmethod
    def _return_orm_returning(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        result,
    ):
        execution_context = result.context
        compile_state = execution_context.compiled.compile_state

        if (
            compile_state.from_statement_ctx
            and not compile_state.from_statement_ctx.compile_options._is_star
        ):
            load_options = execution_options.get(
                "_sa_orm_load_options", QueryContext.default_load_options
            )

            querycontext = QueryContext(
                compile_state.from_statement_ctx,
                compile_state.select_statement,
                statement,
                params,
                session,
                load_options,
                execution_options,
                bind_arguments,
            )
            return loading.instances(result, querycontext)
        else:
            return result


class BulkUDCompileState(ORMDMLState):
    class default_update_options(Options):
        _dml_strategy: DMLStrategyArgument = "auto"
        _synchronize_session: SynchronizeSessionArgument = "auto"
        _can_use_returning: bool = False
        _is_delete_using: bool = False
        _is_update_from: bool = False
        _autoflush: bool = True
        _subject_mapper: Optional[Mapper[Any]] = None
        _resolved_values = EMPTY_DICT
        _eval_condition = None
        _matched_rows = None
        _identity_token = None
        _populate_existing: bool = False

    @classmethod
    def can_use_returning(
        cls,
        dialect: Dialect,
        mapper: Mapper[Any],
        *,
        is_multitable: bool = False,
        is_update_from: bool = False,
        is_delete_using: bool = False,
        is_executemany: bool = False,
    ) -> bool:
        raise NotImplementedError()

    @classmethod
    def orm_pre_session_exec(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        is_pre_event,
    ):
        (
            update_options,
            execution_options,
        ) = BulkUDCompileState.default_update_options.from_execution_options(
            "_sa_orm_update_options",
            {
                "synchronize_session",
                "autoflush",
                "populate_existing",
                "identity_token",
                "is_delete_using",
                "is_update_from",
                "dml_strategy",
            },
            execution_options,
            statement._execution_options,
        )
        bind_arguments["clause"] = statement
        try:
            plugin_subject = statement._propagate_attrs["plugin_subject"]
        except KeyError:
            assert False, "statement had 'orm' plugin but no plugin_subject"
        else:
            if plugin_subject:
                bind_arguments["mapper"] = plugin_subject.mapper
                update_options += {"_subject_mapper": plugin_subject.mapper}

        if "parententity" not in statement.table._annotations:
            update_options += {"_dml_strategy": "core_only"}
        elif not isinstance(params, list):
            if update_options._dml_strategy == "auto":
                update_options += {"_dml_strategy": "orm"}
            elif update_options._dml_strategy == "bulk":
                raise sa_exc.InvalidRequestError(
                    'Can\'t use "bulk" ORM insert strategy without '
                    "passing separate parameters"
                )
        else:
            if update_options._dml_strategy == "auto":
                update_options += {"_dml_strategy": "bulk"}

        sync = update_options._synchronize_session
        if sync is not None:
            if sync not in ("auto", "evaluate", "fetch", False):
                raise sa_exc.ArgumentError(
                    "Valid strategies for session synchronization "
                    "are 'auto', 'evaluate', 'fetch', False"
                )
            if update_options._dml_strategy == "bulk" and sync == "fetch":
                raise sa_exc.InvalidRequestError(
                    "The 'fetch' synchronization strategy is not available "
                    "for 'bulk' ORM updates (i.e. multiple parameter sets)"
                )

        if not is_pre_event:
            if update_options._autoflush:
                session._autoflush()

            if update_options._dml_strategy == "orm":
                if update_options._synchronize_session == "auto":
                    update_options = cls._do_pre_synchronize_auto(
                        session,
                        statement,
                        params,
                        execution_options,
                        bind_arguments,
                        update_options,
                    )
                elif update_options._synchronize_session == "evaluate":
                    update_options = cls._do_pre_synchronize_evaluate(
                        session,
                        statement,
                        params,
                        execution_options,
                        bind_arguments,
                        update_options,
                    )
                elif update_options._synchronize_session == "fetch":
                    update_options = cls._do_pre_synchronize_fetch(
                        session,
                        statement,
                        params,
                        execution_options,
                        bind_arguments,
                        update_options,
                    )
            elif update_options._dml_strategy == "bulk":
                if update_options._synchronize_session == "auto":
                    update_options += {"_synchronize_session": "evaluate"}

            # indicators from the "pre exec" step that are then
            # added to the DML statement, which will also be part of the cache
            # key.  The compile level create_for_statement() method will then
            # consume these at compiler time.
            statement = statement._annotate(
                {
                    "synchronize_session": update_options._synchronize_session,
                    "is_delete_using": update_options._is_delete_using,
                    "is_update_from": update_options._is_update_from,
                    "dml_strategy": update_options._dml_strategy,
                    "can_use_returning": update_options._can_use_returning,
                }
            )

        return (
            statement,
            util.immutabledict(execution_options).union(
                {"_sa_orm_update_options": update_options}
            ),
        )

    @classmethod
    def orm_setup_cursor_result(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        result,
    ):
        # this stage of the execution is called after the
        # do_orm_execute event hook.  meaning for an extension like
        # horizontal sharding, this step happens *within* the horizontal
        # sharding event handler which calls session.execute() re-entrantly
        # and will occur for each backend individually.
        # the sharding extension then returns its own merged result from the
        # individual ones we return here.

        update_options = execution_options["_sa_orm_update_options"]
        if update_options._dml_strategy == "orm":
            if update_options._synchronize_session == "evaluate":
                cls._do_post_synchronize_evaluate(
                    session, statement, result, update_options
                )
            elif update_options._synchronize_session == "fetch":
                cls._do_post_synchronize_fetch(
                    session, statement, result, update_options
                )
        elif update_options._dml_strategy == "bulk":
            if update_options._synchronize_session == "evaluate":
                cls._do_post_synchronize_bulk_evaluate(
                    session, params, result, update_options
                )
            return result

        return cls._return_orm_returning(
            session,
            statement,
            params,
            execution_options,
            bind_arguments,
            result,
        )

    @classmethod
    def _adjust_for_extra_criteria(cls, global_attributes, ext_info):
        """Apply extra criteria filtering.

        For all distinct single-table-inheritance mappers represented in the
        table being updated or deleted, produce additional WHERE criteria such
        that only the appropriate subtypes are selected from the total results.

        Additionally, add WHERE criteria originating from LoaderCriteriaOptions
        collected from the statement.

        """

        return_crit = ()

        adapter = ext_info._adapter if ext_info.is_aliased_class else None

        if (
            "additional_entity_criteria",
            ext_info.mapper,
        ) in global_attributes:
            return_crit += tuple(
                ae._resolve_where_criteria(ext_info)
                for ae in global_attributes[
                    ("additional_entity_criteria", ext_info.mapper)
                ]
                if ae.include_aliases or ae.entity is ext_info
            )

        if ext_info.mapper._single_table_criterion is not None:
            return_crit += (ext_info.mapper._single_table_criterion,)

        if adapter:
            return_crit = tuple(adapter.traverse(crit) for crit in return_crit)

        return return_crit

    @classmethod
    def _interpret_returning_rows(cls, result, mapper, rows):
        """return rows that indicate PK cols in mapper.primary_key position
        for RETURNING rows.

        Prior to 2.0.36, this method seemed to be written for some kind of
        inheritance scenario but the scenario was unused for actual joined
        inheritance, and the function instead seemed to perform some kind of
        partial translation that would remove non-PK cols if the PK cols
        happened to be first in the row, but not otherwise.  The joined
        inheritance walk feature here seems to have never been used as it was
        always skipped by the "local_table" check.

        As of 2.0.36 the function strips away non-PK cols and provides the
        PK cols for the table in mapper PK order.

        """

        try:
            if mapper.local_table is not mapper.base_mapper.local_table:
                # TODO: dive more into how a local table PK is used for fetch
                # sync, not clear if this is correct as it depends on the
                # downstream routine to fetch rows using
                # local_table.primary_key order
                pk_keys = result._tuple_getter(mapper.local_table.primary_key)
            else:
                pk_keys = result._tuple_getter(mapper.primary_key)
        except KeyError:
            # can't use these rows, they don't have PK cols in them
            # this is an unusual case where the user would have used
            # .return_defaults()
            return []

        return [pk_keys(row) for row in rows]

    @classmethod
    def _get_matched_objects_on_criteria(cls, update_options, states):
        mapper = update_options._subject_mapper
        eval_condition = update_options._eval_condition

        raw_data = [
            (state.obj(), state, state.dict)
            for state in states
            if state.mapper.isa(mapper) and not state.expired
        ]

        identity_token = update_options._identity_token
        if identity_token is not None:
            raw_data = [
                (obj, state, dict_)
                for obj, state, dict_ in raw_data
                if state.identity_token == identity_token
            ]

        result = []
        for obj, state, dict_ in raw_data:
            evaled_condition = eval_condition(obj)

            # caution: don't use "in ()" or == here, _EXPIRE_OBJECT
            # evaluates as True for all comparisons
            if (
                evaled_condition is True
                or evaled_condition is evaluator._EXPIRED_OBJECT
            ):
                result.append(
                    (
                        obj,
                        state,
                        dict_,
                        evaled_condition is evaluator._EXPIRED_OBJECT,
                    )
                )
        return result

    @classmethod
    def _eval_condition_from_statement(cls, update_options, statement):
        mapper = update_options._subject_mapper
        target_cls = mapper.class_

        evaluator_compiler = evaluator._EvaluatorCompiler(target_cls)
        crit = ()
        if statement._where_criteria:
            crit += statement._where_criteria

        global_attributes = {}
        for opt in statement._with_options:
            if opt._is_criteria_option:
                opt.get_global_criteria(global_attributes)

        if global_attributes:
            crit += cls._adjust_for_extra_criteria(global_attributes, mapper)

        if crit:
            eval_condition = evaluator_compiler.process(*crit)
        else:
            # workaround for mypy https://github.com/python/mypy/issues/14027
            def _eval_condition(obj):
                return True

            eval_condition = _eval_condition

        return eval_condition

    @classmethod
    def _do_pre_synchronize_auto(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        update_options,
    ):
        """setup auto sync strategy


        "auto" checks if we can use "evaluate" first, then falls back
        to "fetch"

        evaluate is vastly more efficient for the common case
        where session is empty, only has a few objects, and the UPDATE
        statement can potentially match thousands/millions of rows.

        OTOH more complex criteria that fails to work with "evaluate"
        we would hope usually correlates with fewer net rows.

        """

        try:
            eval_condition = cls._eval_condition_from_statement(
                update_options, statement
            )

        except evaluator.UnevaluatableError:
            pass
        else:
            return update_options + {
                "_eval_condition": eval_condition,
                "_synchronize_session": "evaluate",
            }

        update_options += {"_synchronize_session": "fetch"}
        return cls._do_pre_synchronize_fetch(
            session,
            statement,
            params,
            execution_options,
            bind_arguments,
            update_options,
        )

    @classmethod
    def _do_pre_synchronize_evaluate(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        update_options,
    ):
        try:
            eval_condition = cls._eval_condition_from_statement(
                update_options, statement
            )

        except evaluator.UnevaluatableError as err:
            raise sa_exc.InvalidRequestError(
                'Could not evaluate current criteria in Python: "%s". '
                "Specify 'fetch' or False for the "
                "synchronize_session execution option." % err
            ) from err

        return update_options + {
            "_eval_condition": eval_condition,
        }

    @classmethod
    def _get_resolved_values(cls, mapper, statement):
        if statement._multi_values:
            return []
        elif statement._ordered_values:
            return list(statement._ordered_values)
        elif statement._values:
            return list(statement._values.items())
        else:
            return []

    @classmethod
    def _resolved_keys_as_propnames(cls, mapper, resolved_values):
        values = []
        for k, v in resolved_values:
            if mapper and isinstance(k, expression.ColumnElement):
                try:
                    attr = mapper._columntoproperty[k]
                except orm_exc.UnmappedColumnError:
                    pass
                else:
                    values.append((attr.key, v))
            else:
                raise sa_exc.InvalidRequestError(
                    "Attribute name not found, can't be "
                    "synchronized back to objects: %r" % k
                )
        return values

    @classmethod
    def _do_pre_synchronize_fetch(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        update_options,
    ):
        mapper = update_options._subject_mapper

        select_stmt = (
            select(*(mapper.primary_key + (mapper.select_identity_token,)))
            .select_from(mapper)
            .options(*statement._with_options)
        )
        select_stmt._where_criteria = statement._where_criteria

        # conditionally run the SELECT statement for pre-fetch, testing the
        # "bind" for if we can use RETURNING or not using the do_orm_execute
        # event.  If RETURNING is available, the do_orm_execute event
        # will cancel the SELECT from being actually run.
        #
        # The way this is organized seems strange, why don't we just
        # call can_use_returning() before invoking the statement and get
        # answer?, why does this go through the whole execute phase using an
        # event?  Answer: because we are integrating with extensions such
        # as the horizontal sharding extention that "multiplexes" an individual
        # statement run through multiple engines, and it uses
        # do_orm_execute() to do that.

        can_use_returning = None

        def skip_for_returning(orm_context: ORMExecuteState) -> Any:
            bind = orm_context.session.get_bind(**orm_context.bind_arguments)
            nonlocal can_use_returning

            per_bind_result = cls.can_use_returning(
                bind.dialect,
                mapper,
                is_update_from=update_options._is_update_from,
                is_delete_using=update_options._is_delete_using,
                is_executemany=orm_context.is_executemany,
            )

            if can_use_returning is not None:
                if can_use_returning != per_bind_result:
                    raise sa_exc.InvalidRequestError(
                        "For synchronize_session='fetch', can't mix multiple "
                        "backends where some support RETURNING and others "
                        "don't"
                    )
            elif orm_context.is_executemany and not per_bind_result:
                raise sa_exc.InvalidRequestError(
                    "For synchronize_session='fetch', can't use multiple "
                    "parameter sets in ORM mode, which this backend does not "
                    "support with RETURNING"
                )
            else:
                can_use_returning = per_bind_result

            if per_bind_result:
                return _result.null_result()
            else:
                return None

        result = session.execute(
            select_stmt,
            params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            _add_event=skip_for_returning,
        )
        matched_rows = result.fetchall()

        return update_options + {
            "_matched_rows": matched_rows,
            "_can_use_returning": can_use_returning,
        }


@CompileState.plugin_for("orm", "insert")
class BulkORMInsert(ORMDMLState, InsertDMLState):
    class default_insert_options(Options):
        _dml_strategy: DMLStrategyArgument = "auto"
        _render_nulls: bool = False
        _return_defaults: bool = False
        _subject_mapper: Optional[Mapper[Any]] = None
        _autoflush: bool = True
        _populate_existing: bool = False

    select_statement: Optional[FromStatement] = None

    @classmethod
    def orm_pre_session_exec(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        is_pre_event,
    ):
        (
            insert_options,
            execution_options,
        ) = BulkORMInsert.default_insert_options.from_execution_options(
            "_sa_orm_insert_options",
            {"dml_strategy", "autoflush", "populate_existing", "render_nulls"},
            execution_options,
            statement._execution_options,
        )
        bind_arguments["clause"] = statement
        try:
            plugin_subject = statement._propagate_attrs["plugin_subject"]
        except KeyError:
            assert False, "statement had 'orm' plugin but no plugin_subject"
        else:
            if plugin_subject:
                bind_arguments["mapper"] = plugin_subject.mapper
                insert_options += {"_subject_mapper": plugin_subject.mapper}

        if not params:
            if insert_options._dml_strategy == "auto":
                insert_options += {"_dml_strategy": "orm"}
            elif insert_options._dml_strategy == "bulk":
                raise sa_exc.InvalidRequestError(
                    'Can\'t use "bulk" ORM insert strategy without '
                    "passing separate parameters"
                )
        else:
            if insert_options._dml_strategy == "auto":
                insert_options += {"_dml_strategy": "bulk"}

        if insert_options._dml_strategy != "raw":
            # for ORM object loading, like ORMContext, we have to disable
            # result set adapt_to_context, because we will be generating a
            # new statement with specific columns that's cached inside of
            # an ORMFromStatementCompileState, which we will re-use for
            # each result.
            if not execution_options:
                execution_options = context._orm_load_exec_options
            else:
                execution_options = execution_options.union(
                    context._orm_load_exec_options
                )

        if not is_pre_event and insert_options._autoflush:
            session._autoflush()

        statement = statement._annotate(
            {"dml_strategy": insert_options._dml_strategy}
        )

        return (
            statement,
            util.immutabledict(execution_options).union(
                {"_sa_orm_insert_options": insert_options}
            ),
        )

    @classmethod
    def orm_execute_statement(
        cls,
        session: Session,
        statement: dml.Insert,
        params: _CoreAnyExecuteParams,
        execution_options: OrmExecuteOptionsParameter,
        bind_arguments: _BindArguments,
        conn: Connection,
    ) -> _result.Result:
        insert_options = execution_options.get(
            "_sa_orm_insert_options", cls.default_insert_options
        )

        if insert_options._dml_strategy not in (
            "raw",
            "bulk",
            "orm",
            "auto",
        ):
            raise sa_exc.ArgumentError(
                "Valid strategies for ORM insert strategy "
                "are 'raw', 'orm', 'bulk', 'auto"
            )

        result: _result.Result[Any]

        if insert_options._dml_strategy == "raw":
            result = conn.execute(
                statement, params or {}, execution_options=execution_options
            )
            return result

        if insert_options._dml_strategy == "bulk":
            mapper = insert_options._subject_mapper

            if (
                statement._post_values_clause is not None
                and mapper._multiple_persistence_tables
            ):
                raise sa_exc.InvalidRequestError(
                    "bulk INSERT with a 'post values' clause "
                    "(typically upsert) not supported for multi-table "
                    f"mapper {mapper}"
                )

            assert mapper is not None
            assert session._transaction is not None
            result = _bulk_insert(
                mapper,
                cast(
                    "Iterable[Dict[str, Any]]",
                    [params] if isinstance(params, dict) else params,
                ),
                session._transaction,
                isstates=False,
                return_defaults=insert_options._return_defaults,
                render_nulls=insert_options._render_nulls,
                use_orm_insert_stmt=statement,
                execution_options=execution_options,
            )
        elif insert_options._dml_strategy == "orm":
            result = conn.execute(
                statement, params or {}, execution_options=execution_options
            )
        else:
            raise AssertionError()

        if not bool(statement._returning):
            return result

        if insert_options._populate_existing:
            load_options = execution_options.get(
                "_sa_orm_load_options", QueryContext.default_load_options
            )
            load_options += {"_populate_existing": True}
            execution_options = execution_options.union(
                {"_sa_orm_load_options": load_options}
            )

        return cls._return_orm_returning(
            session,
            statement,
            params,
            execution_options,
            bind_arguments,
            result,
        )

    @classmethod
    def create_for_statement(cls, statement, compiler, **kw) -> BulkORMInsert:
        self = cast(
            BulkORMInsert,
            super().create_for_statement(statement, compiler, **kw),
        )

        if compiler is not None:
            toplevel = not compiler.stack
        else:
            toplevel = True
        if not toplevel:
            return self

        mapper = statement._propagate_attrs["plugin_subject"]
        dml_strategy = statement._annotations.get("dml_strategy", "raw")
        if dml_strategy == "bulk":
            self._setup_for_bulk_insert(compiler)
        elif dml_strategy == "orm":
            self._setup_for_orm_insert(compiler, mapper)

        return self

    @classmethod
    def _resolved_keys_as_col_keys(cls, mapper, resolved_value_dict):
        return {
            col.key if col is not None else k: v
            for col, k, v in (
                (mapper.c.get(k), k, v) for k, v in resolved_value_dict.items()
            )
        }

    def _setup_for_orm_insert(self, compiler, mapper):
        statement = orm_level_statement = cast(dml.Insert, self.statement)

        statement = self._setup_orm_returning(
            compiler,
            orm_level_statement,
            statement,
            dml_mapper=mapper,
            use_supplemental_cols=False,
        )
        self.statement = statement

    def _setup_for_bulk_insert(self, compiler):
        """establish an INSERT statement within the context of
        bulk insert.

        This method will be within the "conn.execute()" call that is invoked
        by persistence._emit_insert_statement().

        """
        statement = orm_level_statement = cast(dml.Insert, self.statement)
        an = statement._annotations

        emit_insert_table, emit_insert_mapper = (
            an["_emit_insert_table"],
            an["_emit_insert_mapper"],
        )

        statement = statement._clone()

        statement.table = emit_insert_table
        if self._dict_parameters:
            self._dict_parameters = {
                col: val
                for col, val in self._dict_parameters.items()
                if col.table is emit_insert_table
            }

        statement = self._setup_orm_returning(
            compiler,
            orm_level_statement,
            statement,
            dml_mapper=emit_insert_mapper,
            use_supplemental_cols=True,
        )

        if (
            self.from_statement_ctx is not None
            and self.from_statement_ctx.compile_options._is_star
        ):
            raise sa_exc.CompileError(
                "Can't use RETURNING * with bulk ORM INSERT.  "
                "Please use a different INSERT form, such as INSERT..VALUES "
                "or INSERT with a Core Connection"
            )

        self.statement = statement


@CompileState.plugin_for("orm", "update")
class BulkORMUpdate(BulkUDCompileState, UpdateDMLState):
    @classmethod
    def create_for_statement(cls, statement, compiler, **kw):
        self = cls.__new__(cls)

        dml_strategy = statement._annotations.get(
            "dml_strategy", "unspecified"
        )

        toplevel = not compiler.stack

        if toplevel and dml_strategy == "bulk":
            self._setup_for_bulk_update(statement, compiler)
        elif (
            dml_strategy == "core_only"
            or dml_strategy == "unspecified"
            and "parententity" not in statement.table._annotations
        ):
            UpdateDMLState.__init__(self, statement, compiler, **kw)
        elif not toplevel or dml_strategy in ("orm", "unspecified"):
            self._setup_for_orm_update(statement, compiler)

        return self

    def _setup_for_orm_update(self, statement, compiler, **kw):
        orm_level_statement = statement

        toplevel = not compiler.stack

        ext_info = statement.table._annotations["parententity"]

        self.mapper = mapper = ext_info.mapper

        self._resolved_values = self._get_resolved_values(mapper, statement)

        self._init_global_attributes(
            statement,
            compiler,
            toplevel=toplevel,
            process_criteria_for_toplevel=toplevel,
        )

        if statement._values:
            self._resolved_values = dict(self._resolved_values)

        new_stmt = statement._clone()

        if new_stmt.table._annotations["parententity"] is mapper:
            new_stmt.table = mapper.local_table

        # note if the statement has _multi_values, these
        # are passed through to the new statement, which will then raise
        # InvalidRequestError because UPDATE doesn't support multi_values
        # right now.
        if statement._ordered_values:
            new_stmt._ordered_values = self._resolved_values
        elif statement._values:
            new_stmt._values = self._resolved_values

        new_crit = self._adjust_for_extra_criteria(
            self.global_attributes, mapper
        )
        if new_crit:
            new_stmt = new_stmt.where(*new_crit)

        # if we are against a lambda statement we might not be the
        # topmost object that received per-execute annotations

        # do this first as we need to determine if there is
        # UPDATE..FROM

        UpdateDMLState.__init__(self, new_stmt, compiler, **kw)

        use_supplemental_cols = False

        if not toplevel:
            synchronize_session = None
        else:
            synchronize_session = compiler._annotations.get(
                "synchronize_session", None
            )
        can_use_returning = compiler._annotations.get(
            "can_use_returning", None
        )
        if can_use_returning is not False:
            # even though pre_exec has determined basic
            # can_use_returning for the dialect, if we are to use
            # RETURNING we need to run can_use_returning() at this level
            # unconditionally because is_delete_using was not known
            # at the pre_exec level
            can_use_returning = (
                synchronize_session == "fetch"
                and self.can_use_returning(
                    compiler.dialect, mapper, is_multitable=self.is_multitable
                )
            )

        if synchronize_session == "fetch" and can_use_returning:
            use_supplemental_cols = True

            # NOTE: we might want to RETURNING the actual columns to be
            # synchronized also.  however this is complicated and difficult
            # to align against the behavior of "evaluate".  Additionally,
            # in a large number (if not the majority) of cases, we have the
            # "evaluate" answer, usually a fixed value, in memory already and
            # there's no need to re-fetch the same value
            # over and over again.   so perhaps if it could be RETURNING just
            # the elements that were based on a SQL expression and not
            # a constant.   For now it doesn't quite seem worth it
            new_stmt = new_stmt.return_defaults(*new_stmt.table.primary_key)

        if toplevel:
            new_stmt = self._setup_orm_returning(
                compiler,
                orm_level_statement,
                new_stmt,
                dml_mapper=mapper,
                use_supplemental_cols=use_supplemental_cols,
            )

        self.statement = new_stmt

    def _setup_for_bulk_update(self, statement, compiler, **kw):
        """establish an UPDATE statement within the context of
        bulk insert.

        This method will be within the "conn.execute()" call that is invoked
        by persistence._emit_update_statement().

        """
        statement = cast(dml.Update, statement)
        an = statement._annotations

        emit_update_table, _ = (
            an["_emit_update_table"],
            an["_emit_update_mapper"],
        )

        statement = statement._clone()
        statement.table = emit_update_table

        UpdateDMLState.__init__(self, statement, compiler, **kw)

        if self._ordered_values:
            raise sa_exc.InvalidRequestError(
                "bulk ORM UPDATE does not support ordered_values() for "
                "custom UPDATE statements with bulk parameter sets.  Use a "
                "non-bulk UPDATE statement or use values()."
            )

        if self._dict_parameters:
            self._dict_parameters = {
                col: val
                for col, val in self._dict_parameters.items()
                if col.table is emit_update_table
            }
        self.statement = statement

    @classmethod
    def orm_execute_statement(
        cls,
        session: Session,
        statement: dml.Update,
        params: _CoreAnyExecuteParams,
        execution_options: OrmExecuteOptionsParameter,
        bind_arguments: _BindArguments,
        conn: Connection,
    ) -> _result.Result:

        update_options = execution_options.get(
            "_sa_orm_update_options", cls.default_update_options
        )

        if update_options._populate_existing:
            load_options = execution_options.get(
                "_sa_orm_load_options", QueryContext.default_load_options
            )
            load_options += {"_populate_existing": True}
            execution_options = execution_options.union(
                {"_sa_orm_load_options": load_options}
            )

        if update_options._dml_strategy not in (
            "orm",
            "auto",
            "bulk",
            "core_only",
        ):
            raise sa_exc.ArgumentError(
                "Valid strategies for ORM UPDATE strategy "
                "are 'orm', 'auto', 'bulk', 'core_only'"
            )

        result: _result.Result[Any]

        if update_options._dml_strategy == "bulk":
            enable_check_rowcount = not statement._where_criteria

            assert update_options._synchronize_session != "fetch"

            if (
                statement._where_criteria
                and update_options._synchronize_session == "evaluate"
            ):
                raise sa_exc.InvalidRequestError(
                    "bulk synchronize of persistent objects not supported "
                    "when using bulk update with additional WHERE "
                    "criteria right now.  add synchronize_session=None "
                    "execution option to bypass synchronize of persistent "
                    "objects."
                )
            mapper = update_options._subject_mapper
            assert mapper is not None
            assert session._transaction is not None
            result = _bulk_update(
                mapper,
                cast(
                    "Iterable[Dict[str, Any]]",
                    [params] if isinstance(params, dict) else params,
                ),
                session._transaction,
                isstates=False,
                update_changed_only=False,
                use_orm_update_stmt=statement,
                enable_check_rowcount=enable_check_rowcount,
            )
            return cls.orm_setup_cursor_result(
                session,
                statement,
                params,
                execution_options,
                bind_arguments,
                result,
            )
        else:
            return super().orm_execute_statement(
                session,
                statement,
                params,
                execution_options,
                bind_arguments,
                conn,
            )

    @classmethod
    def can_use_returning(
        cls,
        dialect: Dialect,
        mapper: Mapper[Any],
        *,
        is_multitable: bool = False,
        is_update_from: bool = False,
        is_delete_using: bool = False,
        is_executemany: bool = False,
    ) -> bool:
        # normal answer for "should we use RETURNING" at all.
        normal_answer = (
            dialect.update_returning and mapper.local_table.implicit_returning
        )
        if not normal_answer:
            return False

        if is_executemany:
            return dialect.update_executemany_returning

        # these workarounds are currently hypothetical for UPDATE,
        # unlike DELETE where they impact MariaDB
        if is_update_from:
            return dialect.update_returning_multifrom

        elif is_multitable and not dialect.update_returning_multifrom:
            raise sa_exc.CompileError(
                f'Dialect "{dialect.name}" does not support RETURNING '
                "with UPDATE..FROM; for synchronize_session='fetch', "
                "please add the additional execution option "
                "'is_update_from=True' to the statement to indicate that "
                "a separate SELECT should be used for this backend."
            )

        return True

    @classmethod
    def _do_post_synchronize_bulk_evaluate(
        cls, session, params, result, update_options
    ):
        if not params:
            return

        mapper = update_options._subject_mapper
        pk_keys = [prop.key for prop in mapper._identity_key_props]

        identity_map = session.identity_map

        for param in params:
            identity_key = mapper.identity_key_from_primary_key(
                (param[key] for key in pk_keys),
                update_options._identity_token,
            )
            state = identity_map.fast_get_state(identity_key)
            if not state:
                continue

            evaluated_keys = set(param).difference(pk_keys)

            dict_ = state.dict
            # only evaluate unmodified attributes
            to_evaluate = state.unmodified.intersection(evaluated_keys)
            for key in to_evaluate:
                if key in dict_:
                    dict_[key] = param[key]

            state.manager.dispatch.refresh(state, None, to_evaluate)

            state._commit(dict_, list(to_evaluate))

            # attributes that were formerly modified instead get expired.
            # this only gets hit if the session had pending changes
            # and autoflush were set to False.
            to_expire = evaluated_keys.intersection(dict_).difference(
                to_evaluate
            )
            if to_expire:
                state._expire_attributes(dict_, to_expire)

    @classmethod
    def _do_post_synchronize_evaluate(
        cls, session, statement, result, update_options
    ):
        matched_objects = cls._get_matched_objects_on_criteria(
            update_options,
            session.identity_map.all_states(),
        )

        cls._apply_update_set_values_to_objects(
            session,
            update_options,
            statement,
            result.context.compiled_parameters[0],
            [(obj, state, dict_) for obj, state, dict_, _ in matched_objects],
            result.prefetch_cols(),
            result.postfetch_cols(),
        )

    @classmethod
    def _do_post_synchronize_fetch(
        cls, session, statement, result, update_options
    ):
        target_mapper = update_options._subject_mapper

        returned_defaults_rows = result.returned_defaults_rows
        if returned_defaults_rows:
            pk_rows = cls._interpret_returning_rows(
                result, target_mapper, returned_defaults_rows
            )
            matched_rows = [
                tuple(row) + (update_options._identity_token,)
                for row in pk_rows
            ]
        else:
            matched_rows = update_options._matched_rows

        objs = [
            session.identity_map[identity_key]
            for identity_key in [
                target_mapper.identity_key_from_primary_key(
                    list(primary_key),
                    identity_token=identity_token,
                )
                for primary_key, identity_token in [
                    (row[0:-1], row[-1]) for row in matched_rows
                ]
                if update_options._identity_token is None
                or identity_token == update_options._identity_token
            ]
            if identity_key in session.identity_map
        ]

        if not objs:
            return

        cls._apply_update_set_values_to_objects(
            session,
            update_options,
            statement,
            result.context.compiled_parameters[0],
            [
                (
                    obj,
                    attributes.instance_state(obj),
                    attributes.instance_dict(obj),
                )
                for obj in objs
            ],
            result.prefetch_cols(),
            result.postfetch_cols(),
        )

    @classmethod
    def _apply_update_set_values_to_objects(
        cls,
        session,
        update_options,
        statement,
        effective_params,
        matched_objects,
        prefetch_cols,
        postfetch_cols,
    ):
        """apply values to objects derived from an update statement, e.g.
        UPDATE..SET <values>

        """

        mapper = update_options._subject_mapper
        target_cls = mapper.class_
        evaluator_compiler = evaluator._EvaluatorCompiler(target_cls)
        resolved_values = cls._get_resolved_values(mapper, statement)
        resolved_keys_as_propnames = cls._resolved_keys_as_propnames(
            mapper, resolved_values
        )
        value_evaluators = {}
        for key, value in resolved_keys_as_propnames:
            try:
                _evaluator = evaluator_compiler.process(
                    coercions.expect(roles.ExpressionElementRole, value)
                )
            except evaluator.UnevaluatableError:
                pass
            else:
                value_evaluators[key] = _evaluator

        evaluated_keys = list(value_evaluators.keys())
        attrib = {k for k, v in resolved_keys_as_propnames}

        states = set()

        to_prefetch = {
            c
            for c in prefetch_cols
            if c.key in effective_params
            and c in mapper._columntoproperty
            and c.key not in evaluated_keys
        }
        to_expire = {
            mapper._columntoproperty[c].key
            for c in postfetch_cols
            if c in mapper._columntoproperty
        }.difference(evaluated_keys)

        prefetch_transfer = [
            (mapper._columntoproperty[c].key, c.key) for c in to_prefetch
        ]

        for obj, state, dict_ in matched_objects:

            dict_.update(
                {
                    col_to_prop: effective_params[c_key]
                    for col_to_prop, c_key in prefetch_transfer
                }
            )

            state._expire_attributes(state.dict, to_expire)

            to_evaluate = state.unmodified.intersection(evaluated_keys)

            for key in to_evaluate:
                if key in dict_:
                    # only run eval for attributes that are present.
                    dict_[key] = value_evaluators[key](obj)

            state.manager.dispatch.refresh(state, None, to_evaluate)

            state._commit(dict_, list(to_evaluate))

            # attributes that were formerly modified instead get expired.
            # this only gets hit if the session had pending changes
            # and autoflush were set to False.
            to_expire = attrib.intersection(dict_).difference(to_evaluate)
            if to_expire:
                state._expire_attributes(dict_, to_expire)

            states.add(state)
        session._register_altered(states)


@CompileState.plugin_for("orm", "delete")
class BulkORMDelete(BulkUDCompileState, DeleteDMLState):
    @classmethod
    def create_for_statement(cls, statement, compiler, **kw):
        self = cls.__new__(cls)

        dml_strategy = statement._annotations.get(
            "dml_strategy", "unspecified"
        )

        if (
            dml_strategy == "core_only"
            or dml_strategy == "unspecified"
            and "parententity" not in statement.table._annotations
        ):
            DeleteDMLState.__init__(self, statement, compiler, **kw)
            return self

        toplevel = not compiler.stack

        orm_level_statement = statement

        ext_info = statement.table._annotations["parententity"]
        self.mapper = mapper = ext_info.mapper

        self._init_global_attributes(
            statement,
            compiler,
            toplevel=toplevel,
            process_criteria_for_toplevel=toplevel,
        )

        new_stmt = statement._clone()

        if new_stmt.table._annotations["parententity"] is mapper:
            new_stmt.table = mapper.local_table

        new_crit = cls._adjust_for_extra_criteria(
            self.global_attributes, mapper
        )
        if new_crit:
            new_stmt = new_stmt.where(*new_crit)

        # do this first as we need to determine if there is
        # DELETE..FROM
        DeleteDMLState.__init__(self, new_stmt, compiler, **kw)

        use_supplemental_cols = False

        if not toplevel:
            synchronize_session = None
        else:
            synchronize_session = compiler._annotations.get(
                "synchronize_session", None
            )
        can_use_returning = compiler._annotations.get(
            "can_use_returning", None
        )
        if can_use_returning is not False:
            # even though pre_exec has determined basic
            # can_use_returning for the dialect, if we are to use
            # RETURNING we need to run can_use_returning() at this level
            # unconditionally because is_delete_using was not known
            # at the pre_exec level
            can_use_returning = (
                synchronize_session == "fetch"
                and self.can_use_returning(
                    compiler.dialect,
                    mapper,
                    is_multitable=self.is_multitable,
                    is_delete_using=compiler._annotations.get(
                        "is_delete_using", False
                    ),
                )
            )

        if can_use_returning:
            use_supplemental_cols = True

            new_stmt = new_stmt.return_defaults(*new_stmt.table.primary_key)

        if toplevel:
            new_stmt = self._setup_orm_returning(
                compiler,
                orm_level_statement,
                new_stmt,
                dml_mapper=mapper,
                use_supplemental_cols=use_supplemental_cols,
            )

        self.statement = new_stmt

        return self

    @classmethod
    def orm_execute_statement(
        cls,
        session: Session,
        statement: dml.Delete,
        params: _CoreAnyExecuteParams,
        execution_options: OrmExecuteOptionsParameter,
        bind_arguments: _BindArguments,
        conn: Connection,
    ) -> _result.Result:
        update_options = execution_options.get(
            "_sa_orm_update_options", cls.default_update_options
        )

        if update_options._dml_strategy == "bulk":
            raise sa_exc.InvalidRequestError(
                "Bulk ORM DELETE not supported right now. "
                "Statement may be invoked at the "
                "Core level using "
                "session.connection().execute(stmt, parameters)"
            )

        if update_options._dml_strategy not in ("orm", "auto", "core_only"):
            raise sa_exc.ArgumentError(
                "Valid strategies for ORM DELETE strategy are 'orm', 'auto', "
                "'core_only'"
            )

        return super().orm_execute_statement(
            session, statement, params, execution_options, bind_arguments, conn
        )

    @classmethod
    def can_use_returning(
        cls,
        dialect: Dialect,
        mapper: Mapper[Any],
        *,
        is_multitable: bool = False,
        is_update_from: bool = False,
        is_delete_using: bool = False,
        is_executemany: bool = False,
    ) -> bool:
        # normal answer for "should we use RETURNING" at all.
        normal_answer = (
            dialect.delete_returning and mapper.local_table.implicit_returning
        )
        if not normal_answer:
            return False

        # now get into special workarounds because MariaDB supports
        # DELETE...RETURNING but not DELETE...USING...RETURNING.
        if is_delete_using:
            # is_delete_using hint was passed.   use
            # additional dialect feature (True for PG, False for MariaDB)
            return dialect.delete_returning_multifrom

        elif is_multitable and not dialect.delete_returning_multifrom:
            # is_delete_using hint was not passed, but we determined
            # at compile time that this is in fact a DELETE..USING.
            # it's too late to continue since we did not pre-SELECT.
            # raise that we need that hint up front.

            raise sa_exc.CompileError(
                f'Dialect "{dialect.name}" does not support RETURNING '
                "with DELETE..USING; for synchronize_session='fetch', "
                "please add the additional execution option "
                "'is_delete_using=True' to the statement to indicate that "
                "a separate SELECT should be used for this backend."
            )

        return True

    @classmethod
    def _do_post_synchronize_evaluate(
        cls, session, statement, result, update_options
    ):
        matched_objects = cls._get_matched_objects_on_criteria(
            update_options,
            session.identity_map.all_states(),
        )

        to_delete = []

        for _, state, dict_, is_partially_expired in matched_objects:
            if is_partially_expired:
                state._expire(dict_, session.identity_map._modified)
            else:
                to_delete.append(state)

        if to_delete:
            session._remove_newly_deleted(to_delete)

    @classmethod
    def _do_post_synchronize_fetch(
        cls, session, statement, result, update_options
    ):
        target_mapper = update_options._subject_mapper

        returned_defaults_rows = result.returned_defaults_rows

        if returned_defaults_rows:
            pk_rows = cls._interpret_returning_rows(
                result, target_mapper, returned_defaults_rows
            )

            matched_rows = [
                tuple(row) + (update_options._identity_token,)
                for row in pk_rows
            ]
        else:
            matched_rows = update_options._matched_rows

        for row in matched_rows:
            primary_key = row[0:-1]
            identity_token = row[-1]

            # TODO: inline this and call remove_newly_deleted
            # once
            identity_key = target_mapper.identity_key_from_primary_key(
                list(primary_key),
                identity_token=identity_token,
            )
            if identity_key in session.identity_map:
                session._remove_newly_deleted(
                    [
                        attributes.instance_state(
                            session.identity_map[identity_key]
                        )
                    ]
                )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\storage.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Storage (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page


class SerializedStorageKey(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> SerializedStorageKey:
        return cls(json)

    def __repr__(self):
        return 'SerializedStorageKey({})'.format(super().__repr__())


class StorageType(enum.Enum):
    '''
    Enum of possible storage types.
    '''
    COOKIES = "cookies"
    FILE_SYSTEMS = "file_systems"
    INDEXEDDB = "indexeddb"
    LOCAL_STORAGE = "local_storage"
    SHADER_CACHE = "shader_cache"
    WEBSQL = "websql"
    SERVICE_WORKERS = "service_workers"
    CACHE_STORAGE = "cache_storage"
    INTEREST_GROUPS = "interest_groups"
    SHARED_STORAGE = "shared_storage"
    STORAGE_BUCKETS = "storage_buckets"
    ALL_ = "all"
    OTHER = "other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class UsageForType:
    '''
    Usage for a storage type.
    '''
    #: Name of storage type.
    storage_type: StorageType

    #: Storage usage (bytes).
    usage: float

    def to_json(self):
        json = dict()
        json['storageType'] = self.storage_type.to_json()
        json['usage'] = self.usage
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            storage_type=StorageType.from_json(json['storageType']),
            usage=float(json['usage']),
        )


@dataclass
class TrustTokens:
    '''
    Pair of issuer origin and number of available (signed, but not used) Trust
    Tokens from that issuer.
    '''
    issuer_origin: str

    count: float

    def to_json(self):
        json = dict()
        json['issuerOrigin'] = self.issuer_origin
        json['count'] = self.count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            issuer_origin=str(json['issuerOrigin']),
            count=float(json['count']),
        )


class InterestGroupAuctionId(str):
    '''
    Protected audience interest group auction identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> InterestGroupAuctionId:
        return cls(json)

    def __repr__(self):
        return 'InterestGroupAuctionId({})'.format(super().__repr__())


class InterestGroupAccessType(enum.Enum):
    '''
    Enum of interest group access types.
    '''
    JOIN = "join"
    LEAVE = "leave"
    UPDATE = "update"
    LOADED = "loaded"
    BID = "bid"
    WIN = "win"
    ADDITIONAL_BID = "additionalBid"
    ADDITIONAL_BID_WIN = "additionalBidWin"
    TOP_LEVEL_BID = "topLevelBid"
    TOP_LEVEL_ADDITIONAL_BID = "topLevelAdditionalBid"
    CLEAR = "clear"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class InterestGroupAuctionEventType(enum.Enum):
    '''
    Enum of auction events.
    '''
    STARTED = "started"
    CONFIG_RESOLVED = "configResolved"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class InterestGroupAuctionFetchType(enum.Enum):
    '''
    Enum of network fetches auctions can do.
    '''
    BIDDER_JS = "bidderJs"
    BIDDER_WASM = "bidderWasm"
    SELLER_JS = "sellerJs"
    BIDDER_TRUSTED_SIGNALS = "bidderTrustedSignals"
    SELLER_TRUSTED_SIGNALS = "sellerTrustedSignals"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SharedStorageAccessType(enum.Enum):
    '''
    Enum of shared storage access types.
    '''
    DOCUMENT_ADD_MODULE = "documentAddModule"
    DOCUMENT_SELECT_URL = "documentSelectURL"
    DOCUMENT_RUN = "documentRun"
    DOCUMENT_SET = "documentSet"
    DOCUMENT_APPEND = "documentAppend"
    DOCUMENT_DELETE = "documentDelete"
    DOCUMENT_CLEAR = "documentClear"
    DOCUMENT_GET = "documentGet"
    WORKLET_SET = "workletSet"
    WORKLET_APPEND = "workletAppend"
    WORKLET_DELETE = "workletDelete"
    WORKLET_CLEAR = "workletClear"
    WORKLET_GET = "workletGet"
    WORKLET_KEYS = "workletKeys"
    WORKLET_ENTRIES = "workletEntries"
    WORKLET_LENGTH = "workletLength"
    WORKLET_REMAINING_BUDGET = "workletRemainingBudget"
    HEADER_SET = "headerSet"
    HEADER_APPEND = "headerAppend"
    HEADER_DELETE = "headerDelete"
    HEADER_CLEAR = "headerClear"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SharedStorageEntry:
    '''
    Struct for a single key-value pair in an origin's shared storage.
    '''
    key: str

    value: str

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=str(json['key']),
            value=str(json['value']),
        )


@dataclass
class SharedStorageMetadata:
    '''
    Details for an origin's shared storage.
    '''
    #: Time when the origin's shared storage was last created.
    creation_time: network.TimeSinceEpoch

    #: Number of key-value pairs stored in origin's shared storage.
    length: int

    #: Current amount of bits of entropy remaining in the navigation budget.
    remaining_budget: float

    #: Total number of bytes stored as key-value pairs in origin's shared
    #: storage.
    bytes_used: int

    def to_json(self):
        json = dict()
        json['creationTime'] = self.creation_time.to_json()
        json['length'] = self.length
        json['remainingBudget'] = self.remaining_budget
        json['bytesUsed'] = self.bytes_used
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            creation_time=network.TimeSinceEpoch.from_json(json['creationTime']),
            length=int(json['length']),
            remaining_budget=float(json['remainingBudget']),
            bytes_used=int(json['bytesUsed']),
        )


@dataclass
class SharedStorageReportingMetadata:
    '''
    Pair of reporting metadata details for a candidate URL for ``selectURL()``.
    '''
    event_type: str

    reporting_url: str

    def to_json(self):
        json = dict()
        json['eventType'] = self.event_type
        json['reportingUrl'] = self.reporting_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            event_type=str(json['eventType']),
            reporting_url=str(json['reportingUrl']),
        )


@dataclass
class SharedStorageUrlWithMetadata:
    '''
    Bundles a candidate URL with its reporting metadata.
    '''
    #: Spec of candidate URL.
    url: str

    #: Any associated reporting metadata.
    reporting_metadata: typing.List[SharedStorageReportingMetadata]

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['reportingMetadata'] = [i.to_json() for i in self.reporting_metadata]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            reporting_metadata=[SharedStorageReportingMetadata.from_json(i) for i in json['reportingMetadata']],
        )


@dataclass
class SharedStorageAccessParams:
    '''
    Bundles the parameters for shared storage access events whose
    presence/absence can vary according to SharedStorageAccessType.
    '''
    #: Spec of the module script URL.
    #: Present only for SharedStorageAccessType.documentAddModule.
    script_source_url: typing.Optional[str] = None

    #: Name of the registered operation to be run.
    #: Present only for SharedStorageAccessType.documentRun and
    #: SharedStorageAccessType.documentSelectURL.
    operation_name: typing.Optional[str] = None

    #: The operation's serialized data in bytes (converted to a string).
    #: Present only for SharedStorageAccessType.documentRun and
    #: SharedStorageAccessType.documentSelectURL.
    serialized_data: typing.Optional[str] = None

    #: Array of candidate URLs' specs, along with any associated metadata.
    #: Present only for SharedStorageAccessType.documentSelectURL.
    urls_with_metadata: typing.Optional[typing.List[SharedStorageUrlWithMetadata]] = None

    #: Key for a specific entry in an origin's shared storage.
    #: Present only for SharedStorageAccessType.documentSet,
    #: SharedStorageAccessType.documentAppend,
    #: SharedStorageAccessType.documentDelete,
    #: SharedStorageAccessType.workletSet,
    #: SharedStorageAccessType.workletAppend,
    #: SharedStorageAccessType.workletDelete,
    #: SharedStorageAccessType.workletGet,
    #: SharedStorageAccessType.headerSet,
    #: SharedStorageAccessType.headerAppend, and
    #: SharedStorageAccessType.headerDelete.
    key: typing.Optional[str] = None

    #: Value for a specific entry in an origin's shared storage.
    #: Present only for SharedStorageAccessType.documentSet,
    #: SharedStorageAccessType.documentAppend,
    #: SharedStorageAccessType.workletSet,
    #: SharedStorageAccessType.workletAppend,
    #: SharedStorageAccessType.headerSet, and
    #: SharedStorageAccessType.headerAppend.
    value: typing.Optional[str] = None

    #: Whether or not to set an entry for a key if that key is already present.
    #: Present only for SharedStorageAccessType.documentSet,
    #: SharedStorageAccessType.workletSet, and
    #: SharedStorageAccessType.headerSet.
    ignore_if_present: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.script_source_url is not None:
            json['scriptSourceUrl'] = self.script_source_url
        if self.operation_name is not None:
            json['operationName'] = self.operation_name
        if self.serialized_data is not None:
            json['serializedData'] = self.serialized_data
        if self.urls_with_metadata is not None:
            json['urlsWithMetadata'] = [i.to_json() for i in self.urls_with_metadata]
        if self.key is not None:
            json['key'] = self.key
        if self.value is not None:
            json['value'] = self.value
        if self.ignore_if_present is not None:
            json['ignoreIfPresent'] = self.ignore_if_present
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_source_url=str(json['scriptSourceUrl']) if 'scriptSourceUrl' in json else None,
            operation_name=str(json['operationName']) if 'operationName' in json else None,
            serialized_data=str(json['serializedData']) if 'serializedData' in json else None,
            urls_with_metadata=[SharedStorageUrlWithMetadata.from_json(i) for i in json['urlsWithMetadata']] if 'urlsWithMetadata' in json else None,
            key=str(json['key']) if 'key' in json else None,
            value=str(json['value']) if 'value' in json else None,
            ignore_if_present=bool(json['ignoreIfPresent']) if 'ignoreIfPresent' in json else None,
        )


class StorageBucketsDurability(enum.Enum):
    RELAXED = "relaxed"
    STRICT = "strict"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class StorageBucket:
    storage_key: SerializedStorageKey

    #: If not specified, it is the default bucket of the storageKey.
    name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['storageKey'] = self.storage_key.to_json()
        if self.name is not None:
            json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            storage_key=SerializedStorageKey.from_json(json['storageKey']),
            name=str(json['name']) if 'name' in json else None,
        )


@dataclass
class StorageBucketInfo:
    bucket: StorageBucket

    id_: str

    expiration: network.TimeSinceEpoch

    #: Storage quota (bytes).
    quota: float

    persistent: bool

    durability: StorageBucketsDurability

    def to_json(self):
        json = dict()
        json['bucket'] = self.bucket.to_json()
        json['id'] = self.id_
        json['expiration'] = self.expiration.to_json()
        json['quota'] = self.quota
        json['persistent'] = self.persistent
        json['durability'] = self.durability.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            bucket=StorageBucket.from_json(json['bucket']),
            id_=str(json['id']),
            expiration=network.TimeSinceEpoch.from_json(json['expiration']),
            quota=float(json['quota']),
            persistent=bool(json['persistent']),
            durability=StorageBucketsDurability.from_json(json['durability']),
        )


class AttributionReportingSourceType(enum.Enum):
    NAVIGATION = "navigation"
    EVENT = "event"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class UnsignedInt64AsBase10(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UnsignedInt64AsBase10:
        return cls(json)

    def __repr__(self):
        return 'UnsignedInt64AsBase10({})'.format(super().__repr__())


class UnsignedInt128AsBase16(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UnsignedInt128AsBase16:
        return cls(json)

    def __repr__(self):
        return 'UnsignedInt128AsBase16({})'.format(super().__repr__())


class SignedInt64AsBase10(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> SignedInt64AsBase10:
        return cls(json)

    def __repr__(self):
        return 'SignedInt64AsBase10({})'.format(super().__repr__())


@dataclass
class AttributionReportingFilterDataEntry:
    key: str

    values: typing.List[str]

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['values'] = [i for i in self.values]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=str(json['key']),
            values=[str(i) for i in json['values']],
        )


@dataclass
class AttributionReportingFilterConfig:
    filter_values: typing.List[AttributionReportingFilterDataEntry]

    #: duration in seconds
    lookback_window: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['filterValues'] = [i.to_json() for i in self.filter_values]
        if self.lookback_window is not None:
            json['lookbackWindow'] = self.lookback_window
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filter_values=[AttributionReportingFilterDataEntry.from_json(i) for i in json['filterValues']],
            lookback_window=int(json['lookbackWindow']) if 'lookbackWindow' in json else None,
        )


@dataclass
class AttributionReportingFilterPair:
    filters: typing.List[AttributionReportingFilterConfig]

    not_filters: typing.List[AttributionReportingFilterConfig]

    def to_json(self):
        json = dict()
        json['filters'] = [i.to_json() for i in self.filters]
        json['notFilters'] = [i.to_json() for i in self.not_filters]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filters=[AttributionReportingFilterConfig.from_json(i) for i in json['filters']],
            not_filters=[AttributionReportingFilterConfig.from_json(i) for i in json['notFilters']],
        )


@dataclass
class AttributionReportingAggregationKeysEntry:
    key: str

    value: UnsignedInt128AsBase16

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=str(json['key']),
            value=UnsignedInt128AsBase16.from_json(json['value']),
        )


@dataclass
class AttributionReportingEventReportWindows:
    #: duration in seconds
    start: int

    #: duration in seconds
    ends: typing.List[int]

    def to_json(self):
        json = dict()
        json['start'] = self.start
        json['ends'] = [i for i in self.ends]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            start=int(json['start']),
            ends=[int(i) for i in json['ends']],
        )


@dataclass
class AttributionReportingTriggerSpec:
    #: number instead of integer because not all uint32 can be represented by
    #: int
    trigger_data: typing.List[float]

    event_report_windows: AttributionReportingEventReportWindows

    def to_json(self):
        json = dict()
        json['triggerData'] = [i for i in self.trigger_data]
        json['eventReportWindows'] = self.event_report_windows.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            trigger_data=[float(i) for i in json['triggerData']],
            event_report_windows=AttributionReportingEventReportWindows.from_json(json['eventReportWindows']),
        )


class AttributionReportingTriggerDataMatching(enum.Enum):
    EXACT = "exact"
    MODULUS = "modulus"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AttributionReportingAggregatableDebugReportingData:
    key_piece: UnsignedInt128AsBase16

    #: number instead of integer because not all uint32 can be represented by
    #: int
    value: float

    types: typing.List[str]

    def to_json(self):
        json = dict()
        json['keyPiece'] = self.key_piece.to_json()
        json['value'] = self.value
        json['types'] = [i for i in self.types]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key_piece=UnsignedInt128AsBase16.from_json(json['keyPiece']),
            value=float(json['value']),
            types=[str(i) for i in json['types']],
        )


@dataclass
class AttributionReportingAggregatableDebugReportingConfig:
    key_piece: UnsignedInt128AsBase16

    debug_data: typing.List[AttributionReportingAggregatableDebugReportingData]

    #: number instead of integer because not all uint32 can be represented by
    #: int, only present for source registrations
    budget: typing.Optional[float] = None

    aggregation_coordinator_origin: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['keyPiece'] = self.key_piece.to_json()
        json['debugData'] = [i.to_json() for i in self.debug_data]
        if self.budget is not None:
            json['budget'] = self.budget
        if self.aggregation_coordinator_origin is not None:
            json['aggregationCoordinatorOrigin'] = self.aggregation_coordinator_origin
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key_piece=UnsignedInt128AsBase16.from_json(json['keyPiece']),
            debug_data=[AttributionReportingAggregatableDebugReportingData.from_json(i) for i in json['debugData']],
            budget=float(json['budget']) if 'budget' in json else None,
            aggregation_coordinator_origin=str(json['aggregationCoordinatorOrigin']) if 'aggregationCoordinatorOrigin' in json else None,
        )


@dataclass
class AttributionScopesData:
    values: typing.List[str]

    #: number instead of integer because not all uint32 can be represented by
    #: int
    limit: float

    max_event_states: float

    def to_json(self):
        json = dict()
        json['values'] = [i for i in self.values]
        json['limit'] = self.limit
        json['maxEventStates'] = self.max_event_states
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            values=[str(i) for i in json['values']],
            limit=float(json['limit']),
            max_event_states=float(json['maxEventStates']),
        )


@dataclass
class AttributionReportingSourceRegistration:
    time: network.TimeSinceEpoch

    #: duration in seconds
    expiry: int

    trigger_specs: typing.List[AttributionReportingTriggerSpec]

    #: duration in seconds
    aggregatable_report_window: int

    type_: AttributionReportingSourceType

    source_origin: str

    reporting_origin: str

    destination_sites: typing.List[str]

    event_id: UnsignedInt64AsBase10

    priority: SignedInt64AsBase10

    filter_data: typing.List[AttributionReportingFilterDataEntry]

    aggregation_keys: typing.List[AttributionReportingAggregationKeysEntry]

    trigger_data_matching: AttributionReportingTriggerDataMatching

    destination_limit_priority: SignedInt64AsBase10

    aggregatable_debug_reporting_config: AttributionReportingAggregatableDebugReportingConfig

    max_event_level_reports: int

    debug_key: typing.Optional[UnsignedInt64AsBase10] = None

    scopes_data: typing.Optional[AttributionScopesData] = None

    def to_json(self):
        json = dict()
        json['time'] = self.time.to_json()
        json['expiry'] = self.expiry
        json['triggerSpecs'] = [i.to_json() for i in self.trigger_specs]
        json['aggregatableReportWindow'] = self.aggregatable_report_window
        json['type'] = self.type_.to_json()
        json['sourceOrigin'] = self.source_origin
        json['reportingOrigin'] = self.reporting_origin
        json['destinationSites'] = [i for i in self.destination_sites]
        json['eventId'] = self.event_id.to_json()
        json['priority'] = self.priority.to_json()
        json['filterData'] = [i.to_json() for i in self.filter_data]
        json['aggregationKeys'] = [i.to_json() for i in self.aggregation_keys]
        json['triggerDataMatching'] = self.trigger_data_matching.to_json()
        json['destinationLimitPriority'] = self.destination_limit_priority.to_json()
        json['aggregatableDebugReportingConfig'] = self.aggregatable_debug_reporting_config.to_json()
        json['maxEventLevelReports'] = self.max_event_level_reports
        if self.debug_key is not None:
            json['debugKey'] = self.debug_key.to_json()
        if self.scopes_data is not None:
            json['scopesData'] = self.scopes_data.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            time=network.TimeSinceEpoch.from_json(json['time']),
            expiry=int(json['expiry']),
            trigger_specs=[AttributionReportingTriggerSpec.from_json(i) for i in json['triggerSpecs']],
            aggregatable_report_window=int(json['aggregatableReportWindow']),
            type_=AttributionReportingSourceType.from_json(json['type']),
            source_origin=str(json['sourceOrigin']),
            reporting_origin=str(json['reportingOrigin']),
            destination_sites=[str(i) for i in json['destinationSites']],
            event_id=UnsignedInt64AsBase10.from_json(json['eventId']),
            priority=SignedInt64AsBase10.from_json(json['priority']),
            filter_data=[AttributionReportingFilterDataEntry.from_json(i) for i in json['filterData']],
            aggregation_keys=[AttributionReportingAggregationKeysEntry.from_json(i) for i in json['aggregationKeys']],
            trigger_data_matching=AttributionReportingTriggerDataMatching.from_json(json['triggerDataMatching']),
            destination_limit_priority=SignedInt64AsBase10.from_json(json['destinationLimitPriority']),
            aggregatable_debug_reporting_config=AttributionReportingAggregatableDebugReportingConfig.from_json(json['aggregatableDebugReportingConfig']),
            max_event_level_reports=int(json['maxEventLevelReports']),
            debug_key=UnsignedInt64AsBase10.from_json(json['debugKey']) if 'debugKey' in json else None,
            scopes_data=AttributionScopesData.from_json(json['scopesData']) if 'scopesData' in json else None,
        )


class AttributionReportingSourceRegistrationResult(enum.Enum):
    SUCCESS = "success"
    INTERNAL_ERROR = "internalError"
    INSUFFICIENT_SOURCE_CAPACITY = "insufficientSourceCapacity"
    INSUFFICIENT_UNIQUE_DESTINATION_CAPACITY = "insufficientUniqueDestinationCapacity"
    EXCESSIVE_REPORTING_ORIGINS = "excessiveReportingOrigins"
    PROHIBITED_BY_BROWSER_POLICY = "prohibitedByBrowserPolicy"
    SUCCESS_NOISED = "successNoised"
    DESTINATION_REPORTING_LIMIT_REACHED = "destinationReportingLimitReached"
    DESTINATION_GLOBAL_LIMIT_REACHED = "destinationGlobalLimitReached"
    DESTINATION_BOTH_LIMITS_REACHED = "destinationBothLimitsReached"
    REPORTING_ORIGINS_PER_SITE_LIMIT_REACHED = "reportingOriginsPerSiteLimitReached"
    EXCEEDS_MAX_CHANNEL_CAPACITY = "exceedsMaxChannelCapacity"
    EXCEEDS_MAX_SCOPES_CHANNEL_CAPACITY = "exceedsMaxScopesChannelCapacity"
    EXCEEDS_MAX_TRIGGER_STATE_CARDINALITY = "exceedsMaxTriggerStateCardinality"
    EXCEEDS_MAX_EVENT_STATES_LIMIT = "exceedsMaxEventStatesLimit"
    DESTINATION_PER_DAY_REPORTING_LIMIT_REACHED = "destinationPerDayReportingLimitReached"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AttributionReportingSourceRegistrationTimeConfig(enum.Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AttributionReportingAggregatableValueDictEntry:
    key: str

    #: number instead of integer because not all uint32 can be represented by
    #: int
    value: float

    filtering_id: UnsignedInt64AsBase10

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['value'] = self.value
        json['filteringId'] = self.filtering_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=str(json['key']),
            value=float(json['value']),
            filtering_id=UnsignedInt64AsBase10.from_json(json['filteringId']),
        )


@dataclass
class AttributionReportingAggregatableValueEntry:
    values: typing.List[AttributionReportingAggregatableValueDictEntry]

    filters: AttributionReportingFilterPair

    def to_json(self):
        json = dict()
        json['values'] = [i.to_json() for i in self.values]
        json['filters'] = self.filters.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            values=[AttributionReportingAggregatableValueDictEntry.from_json(i) for i in json['values']],
            filters=AttributionReportingFilterPair.from_json(json['filters']),
        )


@dataclass
class AttributionReportingEventTriggerData:
    data: UnsignedInt64AsBase10

    priority: SignedInt64AsBase10

    filters: AttributionReportingFilterPair

    dedup_key: typing.Optional[UnsignedInt64AsBase10] = None

    def to_json(self):
        json = dict()
        json['data'] = self.data.to_json()
        json['priority'] = self.priority.to_json()
        json['filters'] = self.filters.to_json()
        if self.dedup_key is not None:
            json['dedupKey'] = self.dedup_key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            data=UnsignedInt64AsBase10.from_json(json['data']),
            priority=SignedInt64AsBase10.from_json(json['priority']),
            filters=AttributionReportingFilterPair.from_json(json['filters']),
            dedup_key=UnsignedInt64AsBase10.from_json(json['dedupKey']) if 'dedupKey' in json else None,
        )


@dataclass
class AttributionReportingAggregatableTriggerData:
    key_piece: UnsignedInt128AsBase16

    source_keys: typing.List[str]

    filters: AttributionReportingFilterPair

    def to_json(self):
        json = dict()
        json['keyPiece'] = self.key_piece.to_json()
        json['sourceKeys'] = [i for i in self.source_keys]
        json['filters'] = self.filters.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key_piece=UnsignedInt128AsBase16.from_json(json['keyPiece']),
            source_keys=[str(i) for i in json['sourceKeys']],
            filters=AttributionReportingFilterPair.from_json(json['filters']),
        )


@dataclass
class AttributionReportingAggregatableDedupKey:
    filters: AttributionReportingFilterPair

    dedup_key: typing.Optional[UnsignedInt64AsBase10] = None

    def to_json(self):
        json = dict()
        json['filters'] = self.filters.to_json()
        if self.dedup_key is not None:
            json['dedupKey'] = self.dedup_key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filters=AttributionReportingFilterPair.from_json(json['filters']),
            dedup_key=UnsignedInt64AsBase10.from_json(json['dedupKey']) if 'dedupKey' in json else None,
        )


@dataclass
class AttributionReportingTriggerRegistration:
    filters: AttributionReportingFilterPair

    aggregatable_dedup_keys: typing.List[AttributionReportingAggregatableDedupKey]

    event_trigger_data: typing.List[AttributionReportingEventTriggerData]

    aggregatable_trigger_data: typing.List[AttributionReportingAggregatableTriggerData]

    aggregatable_values: typing.List[AttributionReportingAggregatableValueEntry]

    aggregatable_filtering_id_max_bytes: int

    debug_reporting: bool

    source_registration_time_config: AttributionReportingSourceRegistrationTimeConfig

    aggregatable_debug_reporting_config: AttributionReportingAggregatableDebugReportingConfig

    scopes: typing.List[str]

    debug_key: typing.Optional[UnsignedInt64AsBase10] = None

    aggregation_coordinator_origin: typing.Optional[str] = None

    trigger_context_id: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['filters'] = self.filters.to_json()
        json['aggregatableDedupKeys'] = [i.to_json() for i in self.aggregatable_dedup_keys]
        json['eventTriggerData'] = [i.to_json() for i in self.event_trigger_data]
        json['aggregatableTriggerData'] = [i.to_json() for i in self.aggregatable_trigger_data]
        json['aggregatableValues'] = [i.to_json() for i in self.aggregatable_values]
        json['aggregatableFilteringIdMaxBytes'] = self.aggregatable_filtering_id_max_bytes
        json['debugReporting'] = self.debug_reporting
        json['sourceRegistrationTimeConfig'] = self.source_registration_time_config.to_json()
        json['aggregatableDebugReportingConfig'] = self.aggregatable_debug_reporting_config.to_json()
        json['scopes'] = [i for i in self.scopes]
        if self.debug_key is not None:
            json['debugKey'] = self.debug_key.to_json()
        if self.aggregation_coordinator_origin is not None:
            json['aggregationCoordinatorOrigin'] = self.aggregation_coordinator_origin
        if self.trigger_context_id is not None:
            json['triggerContextId'] = self.trigger_context_id
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filters=AttributionReportingFilterPair.from_json(json['filters']),
            aggregatable_dedup_keys=[AttributionReportingAggregatableDedupKey.from_json(i) for i in json['aggregatableDedupKeys']],
            event_trigger_data=[AttributionReportingEventTriggerData.from_json(i) for i in json['eventTriggerData']],
            aggregatable_trigger_data=[AttributionReportingAggregatableTriggerData.from_json(i) for i in json['aggregatableTriggerData']],
            aggregatable_values=[AttributionReportingAggregatableValueEntry.from_json(i) for i in json['aggregatableValues']],
            aggregatable_filtering_id_max_bytes=int(json['aggregatableFilteringIdMaxBytes']),
            debug_reporting=bool(json['debugReporting']),
            source_registration_time_config=AttributionReportingSourceRegistrationTimeConfig.from_json(json['sourceRegistrationTimeConfig']),
            aggregatable_debug_reporting_config=AttributionReportingAggregatableDebugReportingConfig.from_json(json['aggregatableDebugReportingConfig']),
            scopes=[str(i) for i in json['scopes']],
            debug_key=UnsignedInt64AsBase10.from_json(json['debugKey']) if 'debugKey' in json else None,
            aggregation_coordinator_origin=str(json['aggregationCoordinatorOrigin']) if 'aggregationCoordinatorOrigin' in json else None,
            trigger_context_id=str(json['triggerContextId']) if 'triggerContextId' in json else None,
        )


class AttributionReportingEventLevelResult(enum.Enum):
    SUCCESS = "success"
    SUCCESS_DROPPED_LOWER_PRIORITY = "successDroppedLowerPriority"
    INTERNAL_ERROR = "internalError"
    NO_CAPACITY_FOR_ATTRIBUTION_DESTINATION = "noCapacityForAttributionDestination"
    NO_MATCHING_SOURCES = "noMatchingSources"
    DEDUPLICATED = "deduplicated"
    EXCESSIVE_ATTRIBUTIONS = "excessiveAttributions"
    PRIORITY_TOO_LOW = "priorityTooLow"
    NEVER_ATTRIBUTED_SOURCE = "neverAttributedSource"
    EXCESSIVE_REPORTING_ORIGINS = "excessiveReportingOrigins"
    NO_MATCHING_SOURCE_FILTER_DATA = "noMatchingSourceFilterData"
    PROHIBITED_BY_BROWSER_POLICY = "prohibitedByBrowserPolicy"
    NO_MATCHING_CONFIGURATIONS = "noMatchingConfigurations"
    EXCESSIVE_REPORTS = "excessiveReports"
    FALSELY_ATTRIBUTED_SOURCE = "falselyAttributedSource"
    REPORT_WINDOW_PASSED = "reportWindowPassed"
    NOT_REGISTERED = "notRegistered"
    REPORT_WINDOW_NOT_STARTED = "reportWindowNotStarted"
    NO_MATCHING_TRIGGER_DATA = "noMatchingTriggerData"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AttributionReportingAggregatableResult(enum.Enum):
    SUCCESS = "success"
    INTERNAL_ERROR = "internalError"
    NO_CAPACITY_FOR_ATTRIBUTION_DESTINATION = "noCapacityForAttributionDestination"
    NO_MATCHING_SOURCES = "noMatchingSources"
    EXCESSIVE_ATTRIBUTIONS = "excessiveAttributions"
    EXCESSIVE_REPORTING_ORIGINS = "excessiveReportingOrigins"
    NO_HISTOGRAMS = "noHistograms"
    INSUFFICIENT_BUDGET = "insufficientBudget"
    INSUFFICIENT_NAMED_BUDGET = "insufficientNamedBudget"
    NO_MATCHING_SOURCE_FILTER_DATA = "noMatchingSourceFilterData"
    NOT_REGISTERED = "notRegistered"
    PROHIBITED_BY_BROWSER_POLICY = "prohibitedByBrowserPolicy"
    DEDUPLICATED = "deduplicated"
    REPORT_WINDOW_PASSED = "reportWindowPassed"
    EXCESSIVE_REPORTS = "excessiveReports"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class RelatedWebsiteSet:
    '''
    A single Related Website Set object.
    '''
    #: The primary site of this set, along with the ccTLDs if there is any.
    primary_sites: typing.List[str]

    #: The associated sites of this set, along with the ccTLDs if there is any.
    associated_sites: typing.List[str]

    #: The service sites of this set, along with the ccTLDs if there is any.
    service_sites: typing.List[str]

    def to_json(self):
        json = dict()
        json['primarySites'] = [i for i in self.primary_sites]
        json['associatedSites'] = [i for i in self.associated_sites]
        json['serviceSites'] = [i for i in self.service_sites]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            primary_sites=[str(i) for i in json['primarySites']],
            associated_sites=[str(i) for i in json['associatedSites']],
            service_sites=[str(i) for i in json['serviceSites']],
        )


def get_storage_key_for_frame(
        frame_id: page.FrameId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SerializedStorageKey]:
    '''
    Returns a storage key given a frame id.

    :param frame_id:
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getStorageKeyForFrame',
        'params': params,
    }
    json = yield cmd_dict
    return SerializedStorageKey.from_json(json['storageKey'])


def clear_data_for_origin(
        origin: str,
        storage_types: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears storage for origin.

    :param origin: Security origin.
    :param storage_types: Comma separated list of StorageType to clear.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['storageTypes'] = storage_types
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearDataForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def clear_data_for_storage_key(
        storage_key: str,
        storage_types: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears storage for storage key.

    :param storage_key: Storage key.
    :param storage_types: Comma separated list of StorageType to clear.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    params['storageTypes'] = storage_types
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearDataForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def get_cookies(
        browser_context_id: typing.Optional[browser.BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[network.Cookie]]:
    '''
    Returns all browser cookies.

    :param browser_context_id: *(Optional)* Browser context to use when called on the browser endpoint.
    :returns: Array of cookie objects.
    '''
    params: T_JSON_DICT = dict()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getCookies',
        'params': params,
    }
    json = yield cmd_dict
    return [network.Cookie.from_json(i) for i in json['cookies']]


def set_cookies(
        cookies: typing.List[network.CookieParam],
        browser_context_id: typing.Optional[browser.BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets given cookies.

    :param cookies: Cookies to be set.
    :param browser_context_id: *(Optional)* Browser context to use when called on the browser endpoint.
    '''
    params: T_JSON_DICT = dict()
    params['cookies'] = [i.to_json() for i in cookies]
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setCookies',
        'params': params,
    }
    json = yield cmd_dict


def clear_cookies(
        browser_context_id: typing.Optional[browser.BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears cookies.

    :param browser_context_id: *(Optional)* Browser context to use when called on the browser endpoint.
    '''
    params: T_JSON_DICT = dict()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearCookies',
        'params': params,
    }
    json = yield cmd_dict


def get_usage_and_quota(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[float, float, bool, typing.List[UsageForType]]]:
    '''
    Returns usage and quota in bytes.

    :param origin: Security origin.
    :returns: A tuple with the following items:

        0. **usage** - Storage usage (bytes).
        1. **quota** - Storage quota (bytes).
        2. **overrideActive** - Whether or not the origin has an active storage quota override
        3. **usageBreakdown** - Storage usage per type (bytes).
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getUsageAndQuota',
        'params': params,
    }
    json = yield cmd_dict
    return (
        float(json['usage']),
        float(json['quota']),
        bool(json['overrideActive']),
        [UsageForType.from_json(i) for i in json['usageBreakdown']]
    )


def override_quota_for_origin(
        origin: str,
        quota_size: typing.Optional[float] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Override quota for the specified origin

    **EXPERIMENTAL**

    :param origin: Security origin.
    :param quota_size: *(Optional)* The quota size (in bytes) to override the original quota with. If this is called multiple times, the overridden quota will be equal to the quotaSize provided in the final call. If this is called without specifying a quotaSize, the quota will be reset to the default value for the specified origin. If this is called multiple times with different origins, the override will be maintained for each origin until it is disabled (called without a quotaSize).
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    if quota_size is not None:
        params['quotaSize'] = quota_size
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.overrideQuotaForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def track_cache_storage_for_origin(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Registers origin to be notified when an update occurs to its cache storage list.

    :param origin: Security origin.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.trackCacheStorageForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def track_cache_storage_for_storage_key(
        storage_key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Registers storage key to be notified when an update occurs to its cache storage list.

    :param storage_key: Storage key.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.trackCacheStorageForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def track_indexed_db_for_origin(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Registers origin to be notified when an update occurs to its IndexedDB.

    :param origin: Security origin.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.trackIndexedDBForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def track_indexed_db_for_storage_key(
        storage_key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Registers storage key to be notified when an update occurs to its IndexedDB.

    :param storage_key: Storage key.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.trackIndexedDBForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def untrack_cache_storage_for_origin(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Unregisters origin from receiving notifications for cache storage.

    :param origin: Security origin.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.untrackCacheStorageForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def untrack_cache_storage_for_storage_key(
        storage_key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Unregisters storage key from receiving notifications for cache storage.

    :param storage_key: Storage key.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.untrackCacheStorageForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def untrack_indexed_db_for_origin(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Unregisters origin from receiving notifications for IndexedDB.

    :param origin: Security origin.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.untrackIndexedDBForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def untrack_indexed_db_for_storage_key(
        storage_key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Unregisters storage key from receiving notifications for IndexedDB.

    :param storage_key: Storage key.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.untrackIndexedDBForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def get_trust_tokens() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[TrustTokens]]:
    '''
    Returns the number of stored Trust Tokens per issuer for the
    current browsing context.

    **EXPERIMENTAL**

    :returns:
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getTrustTokens',
    }
    json = yield cmd_dict
    return [TrustTokens.from_json(i) for i in json['tokens']]


def clear_trust_tokens(
        issuer_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Removes all Trust Tokens issued by the provided issuerOrigin.
    Leaves other stored data, including the issuer's Redemption Records, intact.

    **EXPERIMENTAL**

    :param issuer_origin:
    :returns: True if any tokens were deleted, false otherwise.
    '''
    params: T_JSON_DICT = dict()
    params['issuerOrigin'] = issuer_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearTrustTokens',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['didDeleteTokens'])


def get_interest_group_details(
        owner_origin: str,
        name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    Gets details for a named interest group.

    **EXPERIMENTAL**

    :param owner_origin:
    :param name:
    :returns: This largely corresponds to: https://wicg.github.io/turtledove/#dictdef-generatebidinterestgroup but has absolute expirationTime instead of relative lifetimeMs and also adds joiningOrigin.
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['name'] = name
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getInterestGroupDetails',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['details'])


def set_interest_group_tracking(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables/Disables issuing of interestGroupAccessed events.

    **EXPERIMENTAL**

    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setInterestGroupTracking',
        'params': params,
    }
    json = yield cmd_dict


def set_interest_group_auction_tracking(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables/Disables issuing of interestGroupAuctionEventOccurred and
    interestGroupAuctionNetworkRequestCreated.

    **EXPERIMENTAL**

    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setInterestGroupAuctionTracking',
        'params': params,
    }
    json = yield cmd_dict


def get_shared_storage_metadata(
        owner_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SharedStorageMetadata]:
    '''
    Gets metadata for an origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getSharedStorageMetadata',
        'params': params,
    }
    json = yield cmd_dict
    return SharedStorageMetadata.from_json(json['metadata'])


def get_shared_storage_entries(
        owner_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[SharedStorageEntry]]:
    '''
    Gets the entries in an given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getSharedStorageEntries',
        'params': params,
    }
    json = yield cmd_dict
    return [SharedStorageEntry.from_json(i) for i in json['entries']]


def set_shared_storage_entry(
        owner_origin: str,
        key: str,
        value: str,
        ignore_if_present: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets entry with ``key`` and ``value`` for a given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :param key:
    :param value:
    :param ignore_if_present: *(Optional)* If ```ignoreIfPresent```` is included and true, then only sets the entry if ````key``` doesn't already exist.
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['key'] = key
    params['value'] = value
    if ignore_if_present is not None:
        params['ignoreIfPresent'] = ignore_if_present
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setSharedStorageEntry',
        'params': params,
    }
    json = yield cmd_dict


def delete_shared_storage_entry(
        owner_origin: str,
        key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes entry for ``key`` (if it exists) for a given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :param key:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['key'] = key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.deleteSharedStorageEntry',
        'params': params,
    }
    json = yield cmd_dict


def clear_shared_storage_entries(
        owner_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all entries for a given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearSharedStorageEntries',
        'params': params,
    }
    json = yield cmd_dict


def reset_shared_storage_budget(
        owner_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets the budget for ``ownerOrigin`` by clearing all budget withdrawals.

    **EXPERIMENTAL**

    :param owner_origin:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.resetSharedStorageBudget',
        'params': params,
    }
    json = yield cmd_dict


def set_shared_storage_tracking(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables/disables issuing of sharedStorageAccessed events.

    **EXPERIMENTAL**

    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setSharedStorageTracking',
        'params': params,
    }
    json = yield cmd_dict


def set_storage_bucket_tracking(
        storage_key: str,
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set tracking for a storage key's buckets.

    **EXPERIMENTAL**

    :param storage_key:
    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setStorageBucketTracking',
        'params': params,
    }
    json = yield cmd_dict


def delete_storage_bucket(
        bucket: StorageBucket
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes the Storage Bucket with the given storage key and bucket name.

    **EXPERIMENTAL**

    :param bucket:
    '''
    params: T_JSON_DICT = dict()
    params['bucket'] = bucket.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.deleteStorageBucket',
        'params': params,
    }
    json = yield cmd_dict


def run_bounce_tracking_mitigations() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Deletes state for sites identified as potential bounce trackers, immediately.

    **EXPERIMENTAL**

    :returns:
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.runBounceTrackingMitigations',
    }
    json = yield cmd_dict
    return [str(i) for i in json['deletedSites']]


def set_attribution_reporting_local_testing_mode(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    https://wicg.github.io/attribution-reporting-api/

    **EXPERIMENTAL**

    :param enabled: If enabled, noise is suppressed and reports are sent immediately.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setAttributionReportingLocalTestingMode',
        'params': params,
    }
    json = yield cmd_dict


def set_attribution_reporting_tracking(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables/disables issuing of Attribution Reporting events.

    **EXPERIMENTAL**

    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setAttributionReportingTracking',
        'params': params,
    }
    json = yield cmd_dict


def send_pending_attribution_reports() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,int]:
    '''
    Sends all pending Attribution Reports immediately, regardless of their
    scheduled report time.

    **EXPERIMENTAL**

    :returns: The number of reports that were sent.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.sendPendingAttributionReports',
    }
    json = yield cmd_dict
    return int(json['numSent'])


def get_related_website_sets() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[RelatedWebsiteSet]]:
    '''
    Returns the effective Related Website Sets in use by this profile for the browser
    session. The effective Related Website Sets will not change during a browser session.

    **EXPERIMENTAL**

    :returns:
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getRelatedWebsiteSets',
    }
    json = yield cmd_dict
    return [RelatedWebsiteSet.from_json(i) for i in json['sets']]


def get_affected_urls_for_third_party_cookie_metadata(
        first_party_url: str,
        third_party_urls: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Returns the list of URLs from a page and its embedded resources that match
    existing grace period URL pattern rules.
    https://developers.google.com/privacy-sandbox/cookies/temporary-exceptions/grace-period

    **EXPERIMENTAL**

    :param first_party_url: The URL of the page currently being visited.
    :param third_party_urls: The list of embedded resource URLs from the page.
    :returns: Array of matching URLs. If there is a primary pattern match for the first- party URL, only the first-party URL is returned in the array.
    '''
    params: T_JSON_DICT = dict()
    params['firstPartyUrl'] = first_party_url
    params['thirdPartyUrls'] = [i for i in third_party_urls]
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getAffectedUrlsForThirdPartyCookieMetadata',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['matchedUrls']]


@event_class('Storage.cacheStorageContentUpdated')
@dataclass
class CacheStorageContentUpdated:
    '''
    A cache's contents have been modified.
    '''
    #: Origin to update.
    origin: str
    #: Storage key to update.
    storage_key: str
    #: Storage bucket to update.
    bucket_id: str
    #: Name of cache in origin.
    cache_name: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CacheStorageContentUpdated:
        return cls(
            origin=str(json['origin']),
            storage_key=str(json['storageKey']),
            bucket_id=str(json['bucketId']),
            cache_name=str(json['cacheName'])
        )


@event_class('Storage.cacheStorageListUpdated')
@dataclass
class CacheStorageListUpdated:
    '''
    A cache has been added/deleted.
    '''
    #: Origin to update.
    origin: str
    #: Storage key to update.
    storage_key: str
    #: Storage bucket to update.
    bucket_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CacheStorageListUpdated:
        return cls(
            origin=str(json['origin']),
            storage_key=str(json['storageKey']),
            bucket_id=str(json['bucketId'])
        )


@event_class('Storage.indexedDBContentUpdated')
@dataclass
class IndexedDBContentUpdated:
    '''
    The origin's IndexedDB object store has been modified.
    '''
    #: Origin to update.
    origin: str
    #: Storage key to update.
    storage_key: str
    #: Storage bucket to update.
    bucket_id: str
    #: Database to update.
    database_name: str
    #: ObjectStore to update.
    object_store_name: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IndexedDBContentUpdated:
        return cls(
            origin=str(json['origin']),
            storage_key=str(json['storageKey']),
            bucket_id=str(json['bucketId']),
            database_name=str(json['databaseName']),
            object_store_name=str(json['objectStoreName'])
        )


@event_class('Storage.indexedDBListUpdated')
@dataclass
class IndexedDBListUpdated:
    '''
    The origin's IndexedDB database list has been modified.
    '''
    #: Origin to update.
    origin: str
    #: Storage key to update.
    storage_key: str
    #: Storage bucket to update.
    bucket_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IndexedDBListUpdated:
        return cls(
            origin=str(json['origin']),
            storage_key=str(json['storageKey']),
            bucket_id=str(json['bucketId'])
        )


@event_class('Storage.interestGroupAccessed')
@dataclass
class InterestGroupAccessed:
    '''
    One of the interest groups was accessed. Note that these events are global
    to all targets sharing an interest group store.
    '''
    access_time: network.TimeSinceEpoch
    type_: InterestGroupAccessType
    owner_origin: str
    name: str
    #: For topLevelBid/topLevelAdditionalBid, and when appropriate,
    #: win and additionalBidWin
    component_seller_origin: typing.Optional[str]
    #: For bid or somethingBid event, if done locally and not on a server.
    bid: typing.Optional[float]
    bid_currency: typing.Optional[str]
    #: For non-global events --- links to interestGroupAuctionEvent
    unique_auction_id: typing.Optional[InterestGroupAuctionId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAccessed:
        return cls(
            access_time=network.TimeSinceEpoch.from_json(json['accessTime']),
            type_=InterestGroupAccessType.from_json(json['type']),
            owner_origin=str(json['ownerOrigin']),
            name=str(json['name']),
            component_seller_origin=str(json['componentSellerOrigin']) if 'componentSellerOrigin' in json else None,
            bid=float(json['bid']) if 'bid' in json else None,
            bid_currency=str(json['bidCurrency']) if 'bidCurrency' in json else None,
            unique_auction_id=InterestGroupAuctionId.from_json(json['uniqueAuctionId']) if 'uniqueAuctionId' in json else None
        )


@event_class('Storage.interestGroupAuctionEventOccurred')
@dataclass
class InterestGroupAuctionEventOccurred:
    '''
    An auction involving interest groups is taking place. These events are
    target-specific.
    '''
    event_time: network.TimeSinceEpoch
    type_: InterestGroupAuctionEventType
    unique_auction_id: InterestGroupAuctionId
    #: Set for child auctions.
    parent_auction_id: typing.Optional[InterestGroupAuctionId]
    #: Set for started and configResolved
    auction_config: typing.Optional[dict]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAuctionEventOccurred:
        return cls(
            event_time=network.TimeSinceEpoch.from_json(json['eventTime']),
            type_=InterestGroupAuctionEventType.from_json(json['type']),
            unique_auction_id=InterestGroupAuctionId.from_json(json['uniqueAuctionId']),
            parent_auction_id=InterestGroupAuctionId.from_json(json['parentAuctionId']) if 'parentAuctionId' in json else None,
            auction_config=dict(json['auctionConfig']) if 'auctionConfig' in json else None
        )


@event_class('Storage.interestGroupAuctionNetworkRequestCreated')
@dataclass
class InterestGroupAuctionNetworkRequestCreated:
    '''
    Specifies which auctions a particular network fetch may be related to, and
    in what role. Note that it is not ordered with respect to
    Network.requestWillBeSent (but will happen before loadingFinished
    loadingFailed).
    '''
    type_: InterestGroupAuctionFetchType
    request_id: network.RequestId
    #: This is the set of the auctions using the worklet that issued this
    #: request.  In the case of trusted signals, it's possible that only some of
    #: them actually care about the keys being queried.
    auctions: typing.List[InterestGroupAuctionId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAuctionNetworkRequestCreated:
        return cls(
            type_=InterestGroupAuctionFetchType.from_json(json['type']),
            request_id=network.RequestId.from_json(json['requestId']),
            auctions=[InterestGroupAuctionId.from_json(i) for i in json['auctions']]
        )


@event_class('Storage.sharedStorageAccessed')
@dataclass
class SharedStorageAccessed:
    '''
    Shared storage was accessed by the associated page.
    The following parameters are included in all events.
    '''
    #: Time of the access.
    access_time: network.TimeSinceEpoch
    #: Enum value indicating the Shared Storage API method invoked.
    type_: SharedStorageAccessType
    #: DevTools Frame Token for the primary frame tree's root.
    main_frame_id: page.FrameId
    #: Serialized origin for the context that invoked the Shared Storage API.
    owner_origin: str
    #: The sub-parameters wrapped by ``params`` are all optional and their
    #: presence/absence depends on ``type``.
    params: SharedStorageAccessParams

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SharedStorageAccessed:
        return cls(
            access_time=network.TimeSinceEpoch.from_json(json['accessTime']),
            type_=SharedStorageAccessType.from_json(json['type']),
            main_frame_id=page.FrameId.from_json(json['mainFrameId']),
            owner_origin=str(json['ownerOrigin']),
            params=SharedStorageAccessParams.from_json(json['params'])
        )


@event_class('Storage.storageBucketCreatedOrUpdated')
@dataclass
class StorageBucketCreatedOrUpdated:
    bucket_info: StorageBucketInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> StorageBucketCreatedOrUpdated:
        return cls(
            bucket_info=StorageBucketInfo.from_json(json['bucketInfo'])
        )


@event_class('Storage.storageBucketDeleted')
@dataclass
class StorageBucketDeleted:
    bucket_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> StorageBucketDeleted:
        return cls(
            bucket_id=str(json['bucketId'])
        )


@event_class('Storage.attributionReportingSourceRegistered')
@dataclass
class AttributionReportingSourceRegistered:
    '''
    **EXPERIMENTAL**


    '''
    registration: AttributionReportingSourceRegistration
    result: AttributionReportingSourceRegistrationResult

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttributionReportingSourceRegistered:
        return cls(
            registration=AttributionReportingSourceRegistration.from_json(json['registration']),
            result=AttributionReportingSourceRegistrationResult.from_json(json['result'])
        )


@event_class('Storage.attributionReportingTriggerRegistered')
@dataclass
class AttributionReportingTriggerRegistered:
    '''
    **EXPERIMENTAL**


    '''
    registration: AttributionReportingTriggerRegistration
    event_level: AttributionReportingEventLevelResult
    aggregatable: AttributionReportingAggregatableResult

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttributionReportingTriggerRegistered:
        return cls(
            registration=AttributionReportingTriggerRegistration.from_json(json['registration']),
            event_level=AttributionReportingEventLevelResult.from_json(json['eventLevel']),
            aggregatable=AttributionReportingAggregatableResult.from_json(json['aggregatable'])
        )