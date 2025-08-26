
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\cg.py ===
#TODO:
# -Implement Clebsch-Gordan symmetries
# -Improve simplification method
# -Implement new simplifications
"""Clebsch-Gordon Coefficients."""

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict

from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE

__all__ = [
    'CG',
    'Wigner3j',
    'Wigner6j',
    'Wigner9j',
    'cg_simp'
]

#-----------------------------------------------------------------------------
# CG Coefficients
#-----------------------------------------------------------------------------


class Wigner3j(Expr):
    """Class for the Wigner-3j symbols.

    Explanation
    ===========

    Wigner 3j-symbols are coefficients determined by the coupling of
    two angular momenta. When created, they are expressed as symbolic
    quantities that, for numerical parameters, can be evaluated using the
    ``.doit()`` method [1]_.

    Parameters
    ==========

    j1, m1, j2, m2, j3, m3 : Number, Symbol
        Terms determining the angular momentum of coupled angular momentum
        systems.

    Examples
    ========

    Declare a Wigner-3j coefficient and calculate its value

        >>> from sympy.physics.quantum.cg import Wigner3j
        >>> w3j = Wigner3j(6,0,4,0,2,0)
        >>> w3j
        Wigner3j(6, 0, 4, 0, 2, 0)
        >>> w3j.doit()
        sqrt(715)/143

    See Also
    ========

    CG: Clebsch-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    is_commutative = True

    def __new__(cls, j1, m1, j2, m2, j3, m3):
        args = map(sympify, (j1, m1, j2, m2, j3, m3))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def m1(self):
        return self.args[1]

    @property
    def j2(self):
        return self.args[2]

    @property
    def m2(self):
        return self.args[3]

    @property
    def j3(self):
        return self.args[4]

    @property
    def m3(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.m1)),
            (printer._print(self.j2), printer._print(self.m2)),
            (printer._print(self.j3), printer._print(self.m3)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(2))
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens())
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j3,
                    self.m1, self.m2, self.m3))
        return r'\left(\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right)' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_3j(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)


class CG(Wigner3j):
    r"""Class for Clebsch-Gordan coefficient.

    Explanation
    ===========

    Clebsch-Gordan coefficients describe the angular momentum coupling between
    two systems. The coefficients give the expansion of a coupled total angular
    momentum state and an uncoupled tensor product state. The Clebsch-Gordan
    coefficients are defined as [1]_:

    .. math ::
        C^{j_3,m_3}_{j_1,m_1,j_2,m_2} = \left\langle j_1,m_1;j_2,m_2 | j_3,m_3\right\rangle

    Parameters
    ==========

    j1, m1, j2, m2 : Number, Symbol
        Angular momenta of states 1 and 2.

    j3, m3: Number, Symbol
        Total angular momentum of the coupled system.

    Examples
    ========

    Define a Clebsch-Gordan coefficient and evaluate its value

        >>> from sympy.physics.quantum.cg import CG
        >>> from sympy import S
        >>> cg = CG(S(3)/2, S(3)/2, S(1)/2, -S(1)/2, 1, 1)
        >>> cg
        CG(3/2, 3/2, 1/2, -1/2, 1, 1)
        >>> cg.doit()
        sqrt(3)/2
        >>> CG(j1=S(1)/2, m1=-S(1)/2, j2=S(1)/2, m2=+S(1)/2, j3=1, m3=0).doit()
        sqrt(2)/2


    Compare [2]_.

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    .. [2] `Clebsch-Gordan Coefficients, Spherical Harmonics, and d Functions
        <https://pdg.lbl.gov/2020/reviews/rpp2020-rev-clebsch-gordan-coefs.pdf>`_
        in P.A. Zyla *et al.* (Particle Data Group), Prog. Theor. Exp. Phys.
        2020, 083C01 (2020).
    """
    precedence = PRECEDENCE["Pow"] - 1

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return clebsch_gordan(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)

    def _pretty(self, printer, *args):
        bot = printer._print_seq(
            (self.j1, self.m1, self.j2, self.m2), delimiter=',')
        top = printer._print_seq((self.j3, self.m3), delimiter=',')

        pad = max(top.width(), bot.width())
        bot = prettyForm(*bot.left(' '))
        top = prettyForm(*top.left(' '))

        if not pad == bot.width():
            bot = prettyForm(*bot.right(' '*(pad - bot.width())))
        if not pad == top.width():
            top = prettyForm(*top.right(' '*(pad - top.width())))
        s = stringPict('C' + ' '*pad)
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.above(top))
        return s

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j3, self.m3, self.j1,
                    self.m1, self.j2, self.m2))
        return r'C^{%s,%s}_{%s,%s,%s,%s}' % tuple(label)


class Wigner6j(Expr):
    """Class for the Wigner-6j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j, j23):
        args = map(sympify, (j1, j2, j12, j3, j, j23))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j(self):
        return self.args[4]

    @property
    def j23(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.j3)),
            (printer._print(self.j2), printer._print(self.j)),
            (printer._print(self.j12), printer._print(self.j23)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(2))
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12,
                    self.j3, self.j, self.j23))
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_6j(self.j1, self.j2, self.j12, self.j3, self.j, self.j23)


class Wigner9j(Expr):
    """Class for the Wigner-9j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j4, j34, j13, j24, j):
        args = map(sympify, (j1, j2, j12, j3, j4, j34, j13, j24, j))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j4(self):
        return self.args[4]

    @property
    def j34(self):
        return self.args[5]

    @property
    def j13(self):
        return self.args[6]

    @property
    def j24(self):
        return self.args[7]

    @property
    def j(self):
        return self.args[8]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = (
            (printer._print(
                self.j1), printer._print(self.j3), printer._print(self.j13)),
            (printer._print(
                self.j2), printer._print(self.j4), printer._print(self.j24)),
            (printer._print(self.j12), printer._print(self.j34), printer._print(self.j)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(3))
        D = None
        for i in range(3):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12, self.j3,
                self.j4, self.j34, self.j13, self.j24, self.j))
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_9j(self.j1, self.j2, self.j12, self.j3, self.j4, self.j34, self.j13, self.j24, self.j)


def cg_simp(e):
    """Simplify and combine CG coefficients.

    Explanation
    ===========

    This function uses various symmetry and properties of sums and
    products of Clebsch-Gordan coefficients to simplify statements
    involving these terms [1]_.

    Examples
    ========

    Simplify the sum over CG(a,alpha,0,0,a,alpha) for all alpha to
    2*a+1

        >>> from sympy.physics.quantum.cg import CG, cg_simp
        >>> a = CG(1,1,0,0,1,1)
        >>> b = CG(1,0,0,0,1,0)
        >>> c = CG(1,-1,0,0,1,-1)
        >>> cg_simp(a+b+c)
        3

    See Also
    ========

    CG: Clebsh-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """
    if isinstance(e, Add):
        return _cg_simp_add(e)
    elif isinstance(e, Sum):
        return _cg_simp_sum(e)
    elif isinstance(e, Mul):
        return Mul(*[cg_simp(arg) for arg in e.args])
    elif isinstance(e, Pow):
        return Pow(cg_simp(e.base), e.exp)
    else:
        return e


def _cg_simp_add(e):
    #TODO: Improve simplification method
    """Takes a sum of terms involving Clebsch-Gordan coefficients and
    simplifies the terms.

    Explanation
    ===========

    First, we create two lists, cg_part, which is all the terms involving CG
    coefficients, and other_part, which is all other terms. The cg_part list
    is then passed to the simplification methods, which return the new cg_part
    and any additional terms that are added to other_part
    """
    cg_part = []
    other_part = []

    e = expand(e)
    for arg in e.args:
        if arg.has(CG):
            if isinstance(arg, Sum):
                other_part.append(_cg_simp_sum(arg))
            elif isinstance(arg, Mul):
                terms = 1
                for term in arg.args:
                    if isinstance(term, Sum):
                        terms *= _cg_simp_sum(term)
                    else:
                        terms *= term
                if terms.has(CG):
                    cg_part.append(terms)
                else:
                    other_part.append(terms)
            else:
                cg_part.append(arg)
        else:
            other_part.append(arg)

    cg_part, other = _check_varsh_871_1(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_871_2(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_872_9(cg_part)
    other_part.append(other)
    return Add(*cg_part) + Add(*other_part)


def _check_varsh_871_1(term_list):
    # Sum( CG(a,alpha,b,0,a,alpha), (alpha, -a, a)) == KroneckerDelta(b,0)
    a, alpha, b, lt = map(Wild, ('a', 'alpha', 'b', 'lt'))
    expr = lt*CG(a, alpha, b, 0, a, alpha)
    simp = (2*a + 1)*KroneckerDelta(b, 0)
    sign = lt/abs(lt)
    build_expr = 2*a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, lt), (a, b), build_expr, index_expr)


def _check_varsh_871_2(term_list):
    # Sum((-1)**(a-alpha)*CG(a,alpha,a,-alpha,c,0),(alpha,-a,a))
    a, alpha, c, lt = map(Wild, ('a', 'alpha', 'c', 'lt'))
    expr = lt*CG(a, alpha, a, -alpha, c, 0)
    simp = sqrt(2*a + 1)*KroneckerDelta(c, 0)
    sign = (-1)**(a - alpha)*lt/abs(lt)
    build_expr = 2*a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, c, lt), (a, c), build_expr, index_expr)


def _check_varsh_872_9(term_list):
    # Sum( CG(a,alpha,b,beta,c,gamma)*CG(a,alpha',b,beta',c,gamma), (gamma, -c, c), (c, abs(a-b), a+b))
    a, alpha, alphap, b, beta, betap, c, gamma, lt = map(Wild, (
        'a', 'alpha', 'alphap', 'b', 'beta', 'betap', 'c', 'gamma', 'lt'))
    # Case alpha==alphap, beta==betap

    # For numerical alpha,beta
    expr = lt*CG(a, alpha, b, beta, c, gamma)**2
    simp = S.One
    sign = lt/abs(lt)
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    term_list, other1 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)

    # For symbolic alpha,beta
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x)*(x + y + 1)
    index_expr = (c - x)*(x + c) + c + gamma
    term_list, other2 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)

    # Case alpha!=alphap or beta!=betap
    # Note: this only works with leading term of 1, pattern matching is unable to match when there is a Wild leading term
    # For numerical alpha,alphap,beta,betap
    expr = CG(a, alpha, b, beta, c, gamma)*CG(a, alphap, b, betap, c, gamma)
    simp = KroneckerDelta(alpha, alphap)*KroneckerDelta(beta, betap)
    sign = S.One
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    term_list, other3 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)

    # For symbolic alpha,alphap,beta,betap
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x)*(x + y + 1)
    index_expr = (c - x)*(x + c) + c + gamma
    term_list, other4 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)

    return term_list, other1 + other2 + other4


def _check_cg_simp(expr, simp, sign, lt, term_list, variables, dep_variables, build_index_expr, index_expr):
    """ Checks for simplifications that can be made, returning a tuple of the
    simplified list of terms and any terms generated by simplification.

    Parameters
    ==========

    expr: expression
        The expression with Wild terms that will be matched to the terms in
        the sum

    simp: expression
        The expression with Wild terms that is substituted in place of the CG
        terms in the case of simplification

    sign: expression
        The expression with Wild terms denoting the sign that is on expr that
        must match

    lt: expression
        The expression with Wild terms that gives the leading term of the
        matched expr

    term_list: list
        A list of all of the terms is the sum to be simplified

    variables: list
        A list of all the variables that appears in expr

    dep_variables: list
        A list of the variables that must match for all the terms in the sum,
        i.e. the dependent variables

    build_index_expr: expression
        Expression with Wild terms giving the number of elements in cg_index

    index_expr: expression
        Expression with Wild terms giving the index terms have when storing
        them to cg_index

    """
    other_part = 0
    i = 0
    while i < len(term_list):
        sub_1 = _check_cg(term_list[i], expr, len(variables))
        if sub_1 is None:
            i += 1
            continue
        if not build_index_expr.subs(sub_1).is_number:
            i += 1
            continue
        sub_dep = [(x, sub_1[x]) for x in dep_variables]
        cg_index = [None]*build_index_expr.subs(sub_1)
        for j in range(i, len(term_list)):
            sub_2 = _check_cg(term_list[j], expr.subs(sub_dep), len(variables) - len(dep_variables), sign=(sign.subs(sub_1), sign.subs(sub_dep)))
            if sub_2 is None:
                continue
            if not index_expr.subs(sub_dep).subs(sub_2).is_number:
                continue
            cg_index[index_expr.subs(sub_dep).subs(sub_2)] = j, expr.subs(lt, 1).subs(sub_dep).subs(sub_2), lt.subs(sub_2), sign.subs(sub_dep).subs(sub_2)
        if not any(i is None for i in cg_index):
            min_lt = min(*[ abs(term[2]) for term in cg_index ])
            indices = [ term[0] for term in cg_index]
            indices.sort()
            indices.reverse()
            [ term_list.pop(j) for j in indices ]
            for term in cg_index:
                if abs(term[2]) > min_lt:
                    term_list.append( (term[2] - min_lt*term[3])*term[1] )
            other_part += min_lt*(sign*simp).subs(sub_1)
        else:
            i += 1
    return term_list, other_part


def _check_cg(cg_term, expr, length, sign=None):
    """Checks whether a term matches the given expression"""
    # TODO: Check for symmetries
    matches = cg_term.match(expr)
    if matches is None:
        return
    if sign is not None:
        if not isinstance(sign, tuple):
            raise TypeError('sign must be a tuple')
        if not sign[0] == (sign[1]).subs(matches):
            return
    if len(matches) == length:
        return matches


def _cg_simp_sum(e):
    e = _check_varsh_sum_871_1(e)
    e = _check_varsh_sum_871_2(e)
    e = _check_varsh_sum_872_4(e)
    return e


def _check_varsh_sum_871_1(e):
    a = Wild('a')
    alpha = symbols('alpha')
    b = Wild('b')
    match = e.match(Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)))
    if match is not None and len(match) == 2:
        return ((2*a + 1)*KroneckerDelta(b, 0)).subs(match)
    return e


def _check_varsh_sum_871_2(e):
    a = Wild('a')
    alpha = symbols('alpha')
    c = Wild('c')
    match = e.match(
        Sum((-1)**(a - alpha)*CG(a, alpha, a, -alpha, c, 0), (alpha, -a, a)))
    if match is not None and len(match) == 2:
        return (sqrt(2*a + 1)*KroneckerDelta(c, 0)).subs(match)
    return e


def _check_varsh_sum_872_4(e):
    alpha = symbols('alpha')
    beta = symbols('beta')
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    cp = Wild('cp')
    gamma = Wild('gamma')
    gammap = Wild('gammap')
    cg1 = CG(a, alpha, b, beta, c, gamma)
    cg2 = CG(a, alpha, b, beta, cp, gammap)
    match1 = e.match(Sum(cg1*cg2, (alpha, -a, a), (beta, -b, b)))
    if match1 is not None and len(match1) == 6:
        return (KroneckerDelta(c, cp)*KroneckerDelta(gamma, gammap)).subs(match1)
    match2 = e.match(Sum(cg1**2, (alpha, -a, a), (beta, -b, b)))
    if match2 is not None and len(match2) == 4:
        return S.One
    return e


def _cg_list(term):
    if isinstance(term, CG):
        return (term,), 1, 1
    cg = []
    coeff = 1
    if not isinstance(term, (Mul, Pow)):
        raise NotImplementedError('term must be CG, Add, Mul or Pow')
    if isinstance(term, Pow) and term.exp.is_number:
        if term.exp.is_number:
            [ cg.append(term.base) for _ in range(term.exp) ]
        else:
            return (term,), 1, 1
    if isinstance(term, Mul):
        for arg in term.args:
            if isinstance(arg, CG):
                cg.append(arg)
            else:
                coeff *= arg
        return cg, coeff, coeff/abs(coeff)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\target.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Target
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page


class TargetID(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> TargetID:
        return cls(json)

    def __repr__(self):
        return 'TargetID({})'.format(super().__repr__())


class SessionID(str):
    '''
    Unique identifier of attached debugging session.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> SessionID:
        return cls(json)

    def __repr__(self):
        return 'SessionID({})'.format(super().__repr__())


@dataclass
class TargetInfo:
    target_id: TargetID

    #: List of types: https://source.chromium.org/chromium/chromium/src/+/main:content/browser/devtools/devtools_agent_host_impl.cc?ss=chromium&q=f:devtools%20-f:out%20%22::kTypeTab%5B%5D%22
    type_: str

    title: str

    url: str

    #: Whether the target has an attached client.
    attached: bool

    #: Whether the target has access to the originating window.
    can_access_opener: bool

    #: Opener target Id
    opener_id: typing.Optional[TargetID] = None

    #: Frame id of originating window (is only set if target has an opener).
    opener_frame_id: typing.Optional[page.FrameId] = None

    browser_context_id: typing.Optional[browser.BrowserContextID] = None

    #: Provides additional details for specific target types. For example, for
    #: the type of "page", this may be set to "prerender".
    subtype: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['targetId'] = self.target_id.to_json()
        json['type'] = self.type_
        json['title'] = self.title
        json['url'] = self.url
        json['attached'] = self.attached
        json['canAccessOpener'] = self.can_access_opener
        if self.opener_id is not None:
            json['openerId'] = self.opener_id.to_json()
        if self.opener_frame_id is not None:
            json['openerFrameId'] = self.opener_frame_id.to_json()
        if self.browser_context_id is not None:
            json['browserContextId'] = self.browser_context_id.to_json()
        if self.subtype is not None:
            json['subtype'] = self.subtype
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            target_id=TargetID.from_json(json['targetId']),
            type_=str(json['type']),
            title=str(json['title']),
            url=str(json['url']),
            attached=bool(json['attached']),
            can_access_opener=bool(json['canAccessOpener']),
            opener_id=TargetID.from_json(json['openerId']) if 'openerId' in json else None,
            opener_frame_id=page.FrameId.from_json(json['openerFrameId']) if 'openerFrameId' in json else None,
            browser_context_id=browser.BrowserContextID.from_json(json['browserContextId']) if 'browserContextId' in json else None,
            subtype=str(json['subtype']) if 'subtype' in json else None,
        )


@dataclass
class FilterEntry:
    '''
    A filter used by target query/discovery/auto-attach operations.
    '''
    #: If set, causes exclusion of matching targets from the list.
    exclude: typing.Optional[bool] = None

    #: If not present, matches any type.
    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.exclude is not None:
            json['exclude'] = self.exclude
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            exclude=bool(json['exclude']) if 'exclude' in json else None,
            type_=str(json['type']) if 'type' in json else None,
        )


class TargetFilter(list):
    '''
    The entries in TargetFilter are matched sequentially against targets and
    the first entry that matches determines if the target is included or not,
    depending on the value of ``exclude`` field in the entry.
    If filter is not specified, the one assumed is
    [{type: "browser", exclude: true}, {type: "tab", exclude: true}, {}]
    (i.e. include everything but ``browser`` and ``tab``).
    '''
    def to_json(self) -> typing.List[FilterEntry]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[FilterEntry]) -> TargetFilter:
        return cls(json)

    def __repr__(self):
        return 'TargetFilter({})'.format(super().__repr__())


@dataclass
class RemoteLocation:
    host: str

    port: int

    def to_json(self):
        json = dict()
        json['host'] = self.host
        json['port'] = self.port
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            host=str(json['host']),
            port=int(json['port']),
        )


class WindowState(enum.Enum):
    '''
    The state of the target window.
    '''
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    FULLSCREEN = "fullscreen"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def activate_target(
        target_id: TargetID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Activates (focuses) the target.

    :param target_id:
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.activateTarget',
        'params': params,
    }
    json = yield cmd_dict


def attach_to_target(
        target_id: TargetID,
        flatten: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SessionID]:
    '''
    Attaches to the target with given id.

    :param target_id:
    :param flatten: *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.
    :returns: Id assigned to the session.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    if flatten is not None:
        params['flatten'] = flatten
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.attachToTarget',
        'params': params,
    }
    json = yield cmd_dict
    return SessionID.from_json(json['sessionId'])


def attach_to_browser_target() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SessionID]:
    '''
    Attaches to the browser target, only uses flat sessionId mode.

    **EXPERIMENTAL**

    :returns: Id assigned to the session.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.attachToBrowserTarget',
    }
    json = yield cmd_dict
    return SessionID.from_json(json['sessionId'])


def close_target(
        target_id: TargetID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Closes the target. If the target is a page that gets closed too.

    :param target_id:
    :returns: Always set to true. If an error occurs, the response indicates protocol error.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.closeTarget',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['success'])


def expose_dev_tools_protocol(
        target_id: TargetID,
        binding_name: typing.Optional[str] = None,
        inherit_permissions: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Inject object to the target's main frame that provides a communication
    channel with browser target.

    Injected object will be available as ``window[bindingName]``.

    The object has the following API:
    - ``binding.send(json)`` - a method to send messages over the remote debugging protocol
    - ``binding.onmessage = json => handleMessage(json)`` - a callback that will be called for the protocol notifications and command responses.

    **EXPERIMENTAL**

    :param target_id:
    :param binding_name: *(Optional)* Binding name, 'cdp' if not specified.
    :param inherit_permissions: *(Optional)* If true, inherits the current root session's permissions (default: false).
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    if binding_name is not None:
        params['bindingName'] = binding_name
    if inherit_permissions is not None:
        params['inheritPermissions'] = inherit_permissions
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.exposeDevToolsProtocol',
        'params': params,
    }
    json = yield cmd_dict


def create_browser_context(
        dispose_on_detach: typing.Optional[bool] = None,
        proxy_server: typing.Optional[str] = None,
        proxy_bypass_list: typing.Optional[str] = None,
        origins_with_universal_network_access: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,browser.BrowserContextID]:
    '''
    Creates a new empty BrowserContext. Similar to an incognito profile but you can have more than
    one.

    :param dispose_on_detach: **(EXPERIMENTAL)** *(Optional)* If specified, disposes this context when debugging session disconnects.
    :param proxy_server: **(EXPERIMENTAL)** *(Optional)* Proxy server, similar to the one passed to --proxy-server
    :param proxy_bypass_list: **(EXPERIMENTAL)** *(Optional)* Proxy bypass list, similar to the one passed to --proxy-bypass-list
    :param origins_with_universal_network_access: **(EXPERIMENTAL)** *(Optional)* An optional list of origins to grant unlimited cross-origin access to. Parts of the URL other than those constituting origin are ignored.
    :returns: The id of the context created.
    '''
    params: T_JSON_DICT = dict()
    if dispose_on_detach is not None:
        params['disposeOnDetach'] = dispose_on_detach
    if proxy_server is not None:
        params['proxyServer'] = proxy_server
    if proxy_bypass_list is not None:
        params['proxyBypassList'] = proxy_bypass_list
    if origins_with_universal_network_access is not None:
        params['originsWithUniversalNetworkAccess'] = [i for i in origins_with_universal_network_access]
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.createBrowserContext',
        'params': params,
    }
    json = yield cmd_dict
    return browser.BrowserContextID.from_json(json['browserContextId'])


def get_browser_contexts() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[browser.BrowserContextID]]:
    '''
    Returns all browser contexts created with ``Target.createBrowserContext`` method.

    :returns: An array of browser context ids.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getBrowserContexts',
    }
    json = yield cmd_dict
    return [browser.BrowserContextID.from_json(i) for i in json['browserContextIds']]


def create_target(
        url: str,
        left: typing.Optional[int] = None,
        top: typing.Optional[int] = None,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        window_state: typing.Optional[WindowState] = None,
        browser_context_id: typing.Optional[browser.BrowserContextID] = None,
        enable_begin_frame_control: typing.Optional[bool] = None,
        new_window: typing.Optional[bool] = None,
        background: typing.Optional[bool] = None,
        for_tab: typing.Optional[bool] = None,
        hidden: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,TargetID]:
    '''
    Creates a new page.

    :param url: The initial URL the page will be navigated to. An empty string indicates about:blank.
    :param left: **(EXPERIMENTAL)** *(Optional)* Frame left origin in DIP (requires newWindow to be true or headless shell).
    :param top: **(EXPERIMENTAL)** *(Optional)* Frame top origin in DIP (requires newWindow to be true or headless shell).
    :param width: *(Optional)* Frame width in DIP (requires newWindow to be true or headless shell).
    :param height: *(Optional)* Frame height in DIP (requires newWindow to be true or headless shell).
    :param window_state: *(Optional)* Frame window state (requires newWindow to be true or headless shell). Default is normal.
    :param browser_context_id: **(EXPERIMENTAL)** *(Optional)* The browser context to create the page in.
    :param enable_begin_frame_control: **(EXPERIMENTAL)** *(Optional)* Whether BeginFrames for this target will be controlled via DevTools (headless shell only, not supported on MacOS yet, false by default).
    :param new_window: *(Optional)* Whether to create a new Window or Tab (false by default, not supported by headless shell).
    :param background: *(Optional)* Whether to create the target in background or foreground (false by default, not supported by headless shell).
    :param for_tab: **(EXPERIMENTAL)** *(Optional)* Whether to create the target of type "tab".
    :param hidden: **(EXPERIMENTAL)** *(Optional)* Whether to create a hidden target. The hidden target is observable via protocol, but not present in the tab UI strip. Cannot be created with ```forTab: true````, ````newWindow: true```` or ````background: false```. The life-time of the tab is limited to the life-time of the session.
    :returns: The id of the page opened.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    if left is not None:
        params['left'] = left
    if top is not None:
        params['top'] = top
    if width is not None:
        params['width'] = width
    if height is not None:
        params['height'] = height
    if window_state is not None:
        params['windowState'] = window_state.to_json()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    if enable_begin_frame_control is not None:
        params['enableBeginFrameControl'] = enable_begin_frame_control
    if new_window is not None:
        params['newWindow'] = new_window
    if background is not None:
        params['background'] = background
    if for_tab is not None:
        params['forTab'] = for_tab
    if hidden is not None:
        params['hidden'] = hidden
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.createTarget',
        'params': params,
    }
    json = yield cmd_dict
    return TargetID.from_json(json['targetId'])


def detach_from_target(
        session_id: typing.Optional[SessionID] = None,
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Detaches session with given id.

    :param session_id: *(Optional)* Session to detach.
    :param target_id: *(Optional)* Deprecated.
    '''
    params: T_JSON_DICT = dict()
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.detachFromTarget',
        'params': params,
    }
    json = yield cmd_dict


def dispose_browser_context(
        browser_context_id: browser.BrowserContextID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a BrowserContext. All the belonging pages will be closed without calling their
    beforeunload hooks.

    :param browser_context_id:
    '''
    params: T_JSON_DICT = dict()
    params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.disposeBrowserContext',
        'params': params,
    }
    json = yield cmd_dict


def get_target_info(
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,TargetInfo]:
    '''
    Returns information about a target.

    **EXPERIMENTAL**

    :param target_id: *(Optional)*
    :returns:
    '''
    params: T_JSON_DICT = dict()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getTargetInfo',
        'params': params,
    }
    json = yield cmd_dict
    return TargetInfo.from_json(json['targetInfo'])


def get_targets(
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[TargetInfo]]:
    '''
    Retrieves a list of available targets.

    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be reported. If filter is not specified and target discovery is currently enabled, a filter used for target discovery is used for consistency.
    :returns: The list of targets.
    '''
    params: T_JSON_DICT = dict()
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getTargets',
        'params': params,
    }
    json = yield cmd_dict
    return [TargetInfo.from_json(i) for i in json['targetInfos']]


def send_message_to_target(
        message: str,
        session_id: typing.Optional[SessionID] = None,
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sends protocol message over session with given id.
    Consider using flat mode instead; see commands attachToTarget, setAutoAttach,
    and crbug.com/991325.

    :param message:
    :param session_id: *(Optional)* Identifier of the session.
    :param target_id: *(Optional)* Deprecated.
    '''
    params: T_JSON_DICT = dict()
    params['message'] = message
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.sendMessageToTarget',
        'params': params,
    }
    json = yield cmd_dict


def set_auto_attach(
        auto_attach: bool,
        wait_for_debugger_on_start: bool,
        flatten: typing.Optional[bool] = None,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Controls whether to automatically attach to new targets which are considered
    to be directly related to this one (for example, iframes or workers).
    When turned on, attaches to all existing related targets as well. When turned off,
    automatically detaches from all currently attached targets.
    This also clears all targets added by ``autoAttachRelated`` from the list of targets to watch
    for creation of related targets.
    You might want to call this recursively for auto-attached targets to attach
    to all available targets.

    :param auto_attach: Whether to auto-attach to related targets.
    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.
    :param flatten: **(EXPERIMENTAL)** *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.
    '''
    params: T_JSON_DICT = dict()
    params['autoAttach'] = auto_attach
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if flatten is not None:
        params['flatten'] = flatten
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setAutoAttach',
        'params': params,
    }
    json = yield cmd_dict


def auto_attach_related(
        target_id: TargetID,
        wait_for_debugger_on_start: bool,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Adds the specified target to the list of targets that will be monitored for any related target
    creation (such as child frames, child workers and new versions of service worker) and reported
    through ``attachedToTarget``. The specified target is also auto-attached.
    This cancels the effect of any previous ``setAutoAttach`` and is also cancelled by subsequent
    ``setAutoAttach``. Only available at the Browser target.

    **EXPERIMENTAL**

    :param target_id:
    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.autoAttachRelated',
        'params': params,
    }
    json = yield cmd_dict


def set_discover_targets(
        discover: bool,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Controls whether to discover available targets and notify via
    ``targetCreated/targetInfoChanged/targetDestroyed`` events.

    :param discover: Whether to discover available targets.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached. If ```discover```` is false, ````filter``` must be omitted or empty.
    '''
    params: T_JSON_DICT = dict()
    params['discover'] = discover
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setDiscoverTargets',
        'params': params,
    }
    json = yield cmd_dict


def set_remote_locations(
        locations: typing.List[RemoteLocation]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables target discovery for the specified locations, when ``setDiscoverTargets`` was set to
    ``true``.

    **EXPERIMENTAL**

    :param locations: List of remote locations.
    '''
    params: T_JSON_DICT = dict()
    params['locations'] = [i.to_json() for i in locations]
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setRemoteLocations',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Target.attachedToTarget')
@dataclass
class AttachedToTarget:
    '''
    **EXPERIMENTAL**

    Issued when attached to target because of auto-attach or ``attachToTarget`` command.
    '''
    #: Identifier assigned to the session used to send/receive messages.
    session_id: SessionID
    target_info: TargetInfo
    waiting_for_debugger: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttachedToTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            target_info=TargetInfo.from_json(json['targetInfo']),
            waiting_for_debugger=bool(json['waitingForDebugger'])
        )


@event_class('Target.detachedFromTarget')
@dataclass
class DetachedFromTarget:
    '''
    **EXPERIMENTAL**

    Issued when detached from target for any reason (including ``detachFromTarget`` command). Can be
    issued multiple times per target if multiple sessions have been attached to it.
    '''
    #: Detached session identifier.
    session_id: SessionID
    #: Deprecated.
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DetachedFromTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None
        )


@event_class('Target.receivedMessageFromTarget')
@dataclass
class ReceivedMessageFromTarget:
    '''
    Notifies about a new protocol message received from the session (as reported in
    ``attachedToTarget`` event).
    '''
    #: Identifier of a session which sends a message.
    session_id: SessionID
    message: str
    #: Deprecated.
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReceivedMessageFromTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            message=str(json['message']),
            target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None
        )


@event_class('Target.targetCreated')
@dataclass
class TargetCreated:
    '''
    Issued when a possible inspection target is created.
    '''
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCreated:
        return cls(
            target_info=TargetInfo.from_json(json['targetInfo'])
        )


@event_class('Target.targetDestroyed')
@dataclass
class TargetDestroyed:
    '''
    Issued when a target is destroyed.
    '''
    target_id: TargetID

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetDestroyed:
        return cls(
            target_id=TargetID.from_json(json['targetId'])
        )


@event_class('Target.targetCrashed')
@dataclass
class TargetCrashed:
    '''
    Issued when a target has crashed.
    '''
    target_id: TargetID
    #: Termination status type.
    status: str
    #: Termination error code.
    error_code: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCrashed:
        return cls(
            target_id=TargetID.from_json(json['targetId']),
            status=str(json['status']),
            error_code=int(json['errorCode'])
        )


@event_class('Target.targetInfoChanged')
@dataclass
class TargetInfoChanged:
    '''
    Issued when some information about a target has changed. This only happens between
    ``targetCreated`` and ``targetDestroyed``.
    '''
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetInfoChanged:
        return cls(
            target_info=TargetInfo.from_json(json['targetInfo'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\segment.py ===
from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from .cells import (
    _is_single_cell_widths,
    cached_cell_len,
    cell_len,
    get_character_cell_size,
    set_cell_size,
)
from .repr import Result, rich_repr
from .style import Style

if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderResult

log = getLogger("rich")


class ControlType(IntEnum):
    """Non-printable control codes which typically translate to ANSI codes."""

    BELL = 1
    CARRIAGE_RETURN = 2
    HOME = 3
    CLEAR = 4
    SHOW_CURSOR = 5
    HIDE_CURSOR = 6
    ENABLE_ALT_SCREEN = 7
    DISABLE_ALT_SCREEN = 8
    CURSOR_UP = 9
    CURSOR_DOWN = 10
    CURSOR_FORWARD = 11
    CURSOR_BACKWARD = 12
    CURSOR_MOVE_TO_COLUMN = 13
    CURSOR_MOVE_TO = 14
    ERASE_IN_LINE = 15
    SET_WINDOW_TITLE = 16


ControlCode = Union[
    Tuple[ControlType],
    Tuple[ControlType, Union[int, str]],
    Tuple[ControlType, int, int],
]


@rich_repr()
class Segment(NamedTuple):
    """A piece of text with associated style. Segments are produced by the Console render process and
    are ultimately converted in to strings to be written to the terminal.

    Args:
        text (str): A piece of text.
        style (:class:`~rich.style.Style`, optional): An optional style to apply to the text.
        control (Tuple[ControlCode], optional): Optional sequence of control codes.

    Attributes:
        cell_length (int): The cell length of this Segment.
    """

    text: str
    style: Optional[Style] = None
    control: Optional[Sequence[ControlCode]] = None

    @property
    def cell_length(self) -> int:
        """The number of terminal cells required to display self.text.

        Returns:
            int: A number of cells.
        """
        text, _style, control = self
        return 0 if control else cell_len(text)

    def __rich_repr__(self) -> Result:
        yield self.text
        if self.control is None:
            if self.style is not None:
                yield self.style
        else:
            yield self.style
            yield self.control

    def __bool__(self) -> bool:
        """Check if the segment contains text."""
        return bool(self.text)

    @property
    def is_control(self) -> bool:
        """Check if the segment contains control codes."""
        return self.control is not None

    @classmethod
    @lru_cache(1024 * 16)
    def _split_cells(cls, segment: "Segment", cut: int) -> Tuple["Segment", "Segment"]:
        """Split a segment in to two at a given cell position.

        Note that splitting a double-width character, may result in that character turning
        into two spaces.

        Args:
            segment (Segment): A segment to split.
            cut (int): A cell position to cut on.

        Returns:
            A tuple of two segments.
        """
        text, style, control = segment
        _Segment = Segment
        cell_length = segment.cell_length
        if cut >= cell_length:
            return segment, _Segment("", style, control)

        cell_size = get_character_cell_size

        pos = int((cut / cell_length) * len(text))

        while True:
            before = text[:pos]
            cell_pos = cell_len(before)
            out_by = cell_pos - cut
            if not out_by:
                return (
                    _Segment(before, style, control),
                    _Segment(text[pos:], style, control),
                )
            if out_by == -1 and cell_size(text[pos]) == 2:
                return (
                    _Segment(text[:pos] + " ", style, control),
                    _Segment(" " + text[pos + 1 :], style, control),
                )
            if out_by == +1 and cell_size(text[pos - 1]) == 2:
                return (
                    _Segment(text[: pos - 1] + " ", style, control),
                    _Segment(" " + text[pos:], style, control),
                )
            if cell_pos < cut:
                pos += 1
            else:
                pos -= 1

    def split_cells(self, cut: int) -> Tuple["Segment", "Segment"]:
        """Split segment in to two segments at the specified column.

        If the cut point falls in the middle of a 2-cell wide character then it is replaced
        by two spaces, to preserve the display width of the parent segment.

        Args:
            cut (int): Offset within the segment to cut.

        Returns:
            Tuple[Segment, Segment]: Two segments.
        """
        text, style, control = self
        assert cut >= 0

        if _is_single_cell_widths(text):
            # Fast path with all 1 cell characters
            if cut >= len(text):
                return self, Segment("", style, control)
            return (
                Segment(text[:cut], style, control),
                Segment(text[cut:], style, control),
            )

        return self._split_cells(self, cut)

    @classmethod
    def line(cls) -> "Segment":
        """Make a new line segment."""
        return cls("\n")

    @classmethod
    def apply_style(
        cls,
        segments: Iterable["Segment"],
        style: Optional[Style] = None,
        post_style: Optional[Style] = None,
    ) -> Iterable["Segment"]:
        """Apply style(s) to an iterable of segments.

        Returns an iterable of segments where the style is replaced by ``style + segment.style + post_style``.

        Args:
            segments (Iterable[Segment]): Segments to process.
            style (Style, optional): Base style. Defaults to None.
            post_style (Style, optional): Style to apply on top of segment style. Defaults to None.

        Returns:
            Iterable[Segments]: A new iterable of segments (possibly the same iterable).
        """
        result_segments = segments
        if style:
            apply = style.__add__
            result_segments = (
                cls(text, None if control else apply(_style), control)
                for text, _style, control in result_segments
            )
        if post_style:
            result_segments = (
                cls(
                    text,
                    (
                        None
                        if control
                        else (_style + post_style if _style else post_style)
                    ),
                    control,
                )
                for text, _style, control in result_segments
            )
        return result_segments

    @classmethod
    def filter_control(
        cls, segments: Iterable["Segment"], is_control: bool = False
    ) -> Iterable["Segment"]:
        """Filter segments by ``is_control`` attribute.

        Args:
            segments (Iterable[Segment]): An iterable of Segment instances.
            is_control (bool, optional): is_control flag to match in search.

        Returns:
            Iterable[Segment]: And iterable of Segment instances.

        """
        if is_control:
            return filter(attrgetter("control"), segments)
        else:
            return filterfalse(attrgetter("control"), segments)

    @classmethod
    def split_lines(cls, segments: Iterable["Segment"]) -> Iterable[List["Segment"]]:
        """Split a sequence of segments in to a list of lines.

        Args:
            segments (Iterable[Segment]): Segments potentially containing line feeds.

        Yields:
            Iterable[List[Segment]]: Iterable of segment lists, one per line.
        """
        line: List[Segment] = []
        append = line.append

        for segment in segments:
            if "\n" in segment.text and not segment.control:
                text, style, _ = segment
                while text:
                    _text, new_line, text = text.partition("\n")
                    if _text:
                        append(cls(_text, style))
                    if new_line:
                        yield line
                        line = []
                        append = line.append
            else:
                append(segment)
        if line:
            yield line

    @classmethod
    def split_and_crop_lines(
        cls,
        segments: Iterable["Segment"],
        length: int,
        style: Optional[Style] = None,
        pad: bool = True,
        include_new_lines: bool = True,
    ) -> Iterable[List["Segment"]]:
        """Split segments in to lines, and crop lines greater than a given length.

        Args:
            segments (Iterable[Segment]): An iterable of segments, probably
                generated from console.render.
            length (int): Desired line length.
            style (Style, optional): Style to use for any padding.
            pad (bool): Enable padding of lines that are less than `length`.

        Returns:
            Iterable[List[Segment]]: An iterable of lines of segments.
        """
        line: List[Segment] = []
        append = line.append

        adjust_line_length = cls.adjust_line_length
        new_line_segment = cls("\n")

        for segment in segments:
            if "\n" in segment.text and not segment.control:
                text, segment_style, _ = segment
                while text:
                    _text, new_line, text = text.partition("\n")
                    if _text:
                        append(cls(_text, segment_style))
                    if new_line:
                        cropped_line = adjust_line_length(
                            line, length, style=style, pad=pad
                        )
                        if include_new_lines:
                            cropped_line.append(new_line_segment)
                        yield cropped_line
                        line.clear()
            else:
                append(segment)
        if line:
            yield adjust_line_length(line, length, style=style, pad=pad)

    @classmethod
    def adjust_line_length(
        cls,
        line: List["Segment"],
        length: int,
        style: Optional[Style] = None,
        pad: bool = True,
    ) -> List["Segment"]:
        """Adjust a line to a given width (cropping or padding as required).

        Args:
            segments (Iterable[Segment]): A list of segments in a single line.
            length (int): The desired width of the line.
            style (Style, optional): The style of padding if used (space on the end). Defaults to None.
            pad (bool, optional): Pad lines with spaces if they are shorter than `length`. Defaults to True.

        Returns:
            List[Segment]: A line of segments with the desired length.
        """
        line_length = sum(segment.cell_length for segment in line)
        new_line: List[Segment]

        if line_length < length:
            if pad:
                new_line = line + [cls(" " * (length - line_length), style)]
            else:
                new_line = line[:]
        elif line_length > length:
            new_line = []
            append = new_line.append
            line_length = 0
            for segment in line:
                segment_length = segment.cell_length
                if line_length + segment_length < length or segment.control:
                    append(segment)
                    line_length += segment_length
                else:
                    text, segment_style, _ = segment
                    text = set_cell_size(text, length - line_length)
                    append(cls(text, segment_style))
                    break
        else:
            new_line = line[:]
        return new_line

    @classmethod
    def get_line_length(cls, line: List["Segment"]) -> int:
        """Get the length of list of segments.

        Args:
            line (List[Segment]): A line encoded as a list of Segments (assumes no '\\\\n' characters),

        Returns:
            int: The length of the line.
        """
        _cell_len = cell_len
        return sum(_cell_len(text) for text, style, control in line if not control)

    @classmethod
    def get_shape(cls, lines: List[List["Segment"]]) -> Tuple[int, int]:
        """Get the shape (enclosing rectangle) of a list of lines.

        Args:
            lines (List[List[Segment]]): A list of lines (no '\\\\n' characters).

        Returns:
            Tuple[int, int]: Width and height in characters.
        """
        get_line_length = cls.get_line_length
        max_width = max(get_line_length(line) for line in lines) if lines else 0
        return (max_width, len(lines))

    @classmethod
    def set_shape(
        cls,
        lines: List[List["Segment"]],
        width: int,
        height: Optional[int] = None,
        style: Optional[Style] = None,
        new_lines: bool = False,
    ) -> List[List["Segment"]]:
        """Set the shape of a list of lines (enclosing rectangle).

        Args:
            lines (List[List[Segment]]): A list of lines.
            width (int): Desired width.
            height (int, optional): Desired height or None for no change.
            style (Style, optional): Style of any padding added.
            new_lines (bool, optional): Padded lines should include "\n". Defaults to False.

        Returns:
            List[List[Segment]]: New list of lines.
        """
        _height = height or len(lines)

        blank = (
            [cls(" " * width + "\n", style)] if new_lines else [cls(" " * width, style)]
        )

        adjust_line_length = cls.adjust_line_length
        shaped_lines = lines[:_height]
        shaped_lines[:] = [
            adjust_line_length(line, width, style=style) for line in lines
        ]
        if len(shaped_lines) < _height:
            shaped_lines.extend([blank] * (_height - len(shaped_lines)))
        return shaped_lines

    @classmethod
    def align_top(
        cls: Type["Segment"],
        lines: List[List["Segment"]],
        width: int,
        height: int,
        style: Style,
        new_lines: bool = False,
    ) -> List[List["Segment"]]:
        """Aligns lines to top (adds extra lines to bottom as required).

        Args:
            lines (List[List[Segment]]): A list of lines.
            width (int): Desired width.
            height (int, optional): Desired height or None for no change.
            style (Style): Style of any padding added.
            new_lines (bool, optional): Padded lines should include "\n". Defaults to False.

        Returns:
            List[List[Segment]]: New list of lines.
        """
        extra_lines = height - len(lines)
        if not extra_lines:
            return lines[:]
        lines = lines[:height]
        blank = cls(" " * width + "\n", style) if new_lines else cls(" " * width, style)
        lines = lines + [[blank]] * extra_lines
        return lines

    @classmethod
    def align_bottom(
        cls: Type["Segment"],
        lines: List[List["Segment"]],
        width: int,
        height: int,
        style: Style,
        new_lines: bool = False,
    ) -> List[List["Segment"]]:
        """Aligns render to bottom (adds extra lines above as required).

        Args:
            lines (List[List[Segment]]): A list of lines.
            width (int): Desired width.
            height (int, optional): Desired height or None for no change.
            style (Style): Style of any padding added. Defaults to None.
            new_lines (bool, optional): Padded lines should include "\n". Defaults to False.

        Returns:
            List[List[Segment]]: New list of lines.
        """
        extra_lines = height - len(lines)
        if not extra_lines:
            return lines[:]
        lines = lines[:height]
        blank = cls(" " * width + "\n", style) if new_lines else cls(" " * width, style)
        lines = [[blank]] * extra_lines + lines
        return lines

    @classmethod
    def align_middle(
        cls: Type["Segment"],
        lines: List[List["Segment"]],
        width: int,
        height: int,
        style: Style,
        new_lines: bool = False,
    ) -> List[List["Segment"]]:
        """Aligns lines to middle (adds extra lines to above and below as required).

        Args:
            lines (List[List[Segment]]): A list of lines.
            width (int): Desired width.
            height (int, optional): Desired height or None for no change.
            style (Style): Style of any padding added.
            new_lines (bool, optional): Padded lines should include "\n". Defaults to False.

        Returns:
            List[List[Segment]]: New list of lines.
        """
        extra_lines = height - len(lines)
        if not extra_lines:
            return lines[:]
        lines = lines[:height]
        blank = cls(" " * width + "\n", style) if new_lines else cls(" " * width, style)
        top_lines = extra_lines // 2
        bottom_lines = extra_lines - top_lines
        lines = [[blank]] * top_lines + lines + [[blank]] * bottom_lines
        return lines

    @classmethod
    def simplify(cls, segments: Iterable["Segment"]) -> Iterable["Segment"]:
        """Simplify an iterable of segments by combining contiguous segments with the same style.

        Args:
            segments (Iterable[Segment]): An iterable of segments.

        Returns:
            Iterable[Segment]: A possibly smaller iterable of segments that will render the same way.
        """
        iter_segments = iter(segments)
        try:
            last_segment = next(iter_segments)
        except StopIteration:
            return

        _Segment = Segment
        for segment in iter_segments:
            if last_segment.style == segment.style and not segment.control:
                last_segment = _Segment(
                    last_segment.text + segment.text, last_segment.style
                )
            else:
                yield last_segment
                last_segment = segment
        yield last_segment

    @classmethod
    def strip_links(cls, segments: Iterable["Segment"]) -> Iterable["Segment"]:
        """Remove all links from an iterable of styles.

        Args:
            segments (Iterable[Segment]): An iterable segments.

        Yields:
            Segment: Segments with link removed.
        """
        for segment in segments:
            if segment.control or segment.style is None:
                yield segment
            else:
                text, style, _control = segment
                yield cls(text, style.update_link(None) if style else None)

    @classmethod
    def strip_styles(cls, segments: Iterable["Segment"]) -> Iterable["Segment"]:
        """Remove all styles from an iterable of segments.

        Args:
            segments (Iterable[Segment]): An iterable segments.

        Yields:
            Segment: Segments with styles replace with None
        """
        for text, _style, control in segments:
            yield cls(text, None, control)

    @classmethod
    def remove_color(cls, segments: Iterable["Segment"]) -> Iterable["Segment"]:
        """Remove all color from an iterable of segments.

        Args:
            segments (Iterable[Segment]): An iterable segments.

        Yields:
            Segment: Segments with colorless style.
        """

        cache: Dict[Style, Style] = {}
        for text, style, control in segments:
            if style:
                colorless_style = cache.get(style)
                if colorless_style is None:
                    colorless_style = style.without_color
                    cache[style] = colorless_style
                yield cls(text, colorless_style, control)
            else:
                yield cls(text, None, control)

    @classmethod
    def divide(
        cls, segments: Iterable["Segment"], cuts: Iterable[int]
    ) -> Iterable[List["Segment"]]:
        """Divides an iterable of segments in to portions.

        Args:
            cuts (Iterable[int]): Cell positions where to divide.

        Yields:
            [Iterable[List[Segment]]]: An iterable of Segments in List.
        """
        split_segments: List["Segment"] = []
        add_segment = split_segments.append

        iter_cuts = iter(cuts)

        while True:
            cut = next(iter_cuts, -1)
            if cut == -1:
                return
            if cut != 0:
                break
            yield []
        pos = 0

        segments_clear = split_segments.clear
        segments_copy = split_segments.copy

        _cell_len = cached_cell_len
        for segment in segments:
            text, _style, control = segment
            while text:
                end_pos = pos if control else pos + _cell_len(text)
                if end_pos < cut:
                    add_segment(segment)
                    pos = end_pos
                    break

                if end_pos == cut:
                    add_segment(segment)
                    yield segments_copy()
                    segments_clear()
                    pos = end_pos

                    cut = next(iter_cuts, -1)
                    if cut == -1:
                        if split_segments:
                            yield segments_copy()
                        return

                    break

                else:
                    before, segment = segment.split_cells(cut - pos)
                    text, _style, control = segment
                    add_segment(before)
                    yield segments_copy()
                    segments_clear()
                    pos = cut

                cut = next(iter_cuts, -1)
                if cut == -1:
                    if split_segments:
                        yield segments_copy()
                    return

        yield segments_copy()


class Segments:
    """A simple renderable to render an iterable of segments. This class may be useful if
    you want to print segments outside of a __rich_console__ method.

    Args:
        segments (Iterable[Segment]): An iterable of segments.
        new_lines (bool, optional): Add new lines between segments. Defaults to False.
    """

    def __init__(self, segments: Iterable[Segment], new_lines: bool = False) -> None:
        self.segments = list(segments)
        self.new_lines = new_lines

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        if self.new_lines:
            line = Segment.line()
            for segment in self.segments:
                yield segment
                yield line
        else:
            yield from self.segments


class SegmentLines:
    def __init__(self, lines: Iterable[List[Segment]], new_lines: bool = False) -> None:
        """A simple renderable containing a number of lines of segments. May be used as an intermediate
        in rendering process.

        Args:
            lines (Iterable[List[Segment]]): Lists of segments forming lines.
            new_lines (bool, optional): Insert new lines after each line. Defaults to False.
        """
        self.lines = list(lines)
        self.new_lines = new_lines

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        if self.new_lines:
            new_line = Segment.line()
            for line in self.lines:
                yield from line
                yield new_line
        else:
            for line in self.lines:
                yield from line


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich.console import Console
    from pip._vendor.rich.syntax import Syntax
    from pip._vendor.rich.text import Text

    code = """from rich.console import Console
console = Console()
text = Text.from_markup("Hello, [bold magenta]World[/]!")
console.print(text)"""

    text = Text.from_markup("Hello, [bold magenta]World[/]!")

    console = Console()

    console.rule("rich.Segment")
    console.print(
        "A Segment is the last step in the Rich render process before generating text with ANSI codes."
    )
    console.print("\nConsider the following code:\n")
    console.print(Syntax(code, "python", line_numbers=True))
    console.print()
    console.print(
        "When you call [b]print()[/b], Rich [i]renders[/i] the object in to the following:\n"
    )
    fragments = list(console.render(text))
    console.print(fragments)
    console.print()
    console.print("The Segments are then processed to produce the following output:\n")
    console.print(text)
    console.print(
        "\nYou will only need to know this if you are implementing your own Rich renderables."
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\version.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012-2023 The Python Software Foundation.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
"""
Implementation of a flexible versioning scheme providing support for PEP-440,
setuptools-compatible and semantic versioning.
"""

import logging
import re

from .compat import string_types
from .util import parse_requirement

__all__ = ['NormalizedVersion', 'NormalizedMatcher',
           'LegacyVersion', 'LegacyMatcher',
           'SemanticVersion', 'SemanticMatcher',
           'UnsupportedVersionError', 'get_scheme']

logger = logging.getLogger(__name__)


class UnsupportedVersionError(ValueError):
    """This is an unsupported version."""
    pass


class Version(object):
    def __init__(self, s):
        self._string = s = s.strip()
        self._parts = parts = self.parse(s)
        assert isinstance(parts, tuple)
        assert len(parts) > 0

    def parse(self, s):
        raise NotImplementedError('please implement in a subclass')

    def _check_compatible(self, other):
        if type(self) != type(other):
            raise TypeError('cannot compare %r and %r' % (self, other))

    def __eq__(self, other):
        self._check_compatible(other)
        return self._parts == other._parts

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        self._check_compatible(other)
        return self._parts < other._parts

    def __gt__(self, other):
        return not (self.__lt__(other) or self.__eq__(other))

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    # See http://docs.python.org/reference/datamodel#object.__hash__
    def __hash__(self):
        return hash(self._parts)

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self._string)

    def __str__(self):
        return self._string

    @property
    def is_prerelease(self):
        raise NotImplementedError('Please implement in subclasses.')


class Matcher(object):
    version_class = None

    # value is either a callable or the name of a method
    _operators = {
        '<': lambda v, c, p: v < c,
        '>': lambda v, c, p: v > c,
        '<=': lambda v, c, p: v == c or v < c,
        '>=': lambda v, c, p: v == c or v > c,
        '==': lambda v, c, p: v == c,
        '===': lambda v, c, p: v == c,
        # by default, compatible => >=.
        '~=': lambda v, c, p: v == c or v > c,
        '!=': lambda v, c, p: v != c,
    }

    # this is a method only to support alternative implementations
    # via overriding
    def parse_requirement(self, s):
        return parse_requirement(s)

    def __init__(self, s):
        if self.version_class is None:
            raise ValueError('Please specify a version class')
        self._string = s = s.strip()
        r = self.parse_requirement(s)
        if not r:
            raise ValueError('Not valid: %r' % s)
        self.name = r.name
        self.key = self.name.lower()    # for case-insensitive comparisons
        clist = []
        if r.constraints:
            # import pdb; pdb.set_trace()
            for op, s in r.constraints:
                if s.endswith('.*'):
                    if op not in ('==', '!='):
                        raise ValueError('\'.*\' not allowed for '
                                         '%r constraints' % op)
                    # Could be a partial version (e.g. for '2.*') which
                    # won't parse as a version, so keep it as a string
                    vn, prefix = s[:-2], True
                    # Just to check that vn is a valid version
                    self.version_class(vn)
                else:
                    # Should parse as a version, so we can create an
                    # instance for the comparison
                    vn, prefix = self.version_class(s), False
                clist.append((op, vn, prefix))
        self._parts = tuple(clist)

    def match(self, version):
        """
        Check if the provided version matches the constraints.

        :param version: The version to match against this instance.
        :type version: String or :class:`Version` instance.
        """
        if isinstance(version, string_types):
            version = self.version_class(version)
        for operator, constraint, prefix in self._parts:
            f = self._operators.get(operator)
            if isinstance(f, string_types):
                f = getattr(self, f)
            if not f:
                msg = ('%r not implemented '
                       'for %s' % (operator, self.__class__.__name__))
                raise NotImplementedError(msg)
            if not f(version, constraint, prefix):
                return False
        return True

    @property
    def exact_version(self):
        result = None
        if len(self._parts) == 1 and self._parts[0][0] in ('==', '==='):
            result = self._parts[0][1]
        return result

    def _check_compatible(self, other):
        if type(self) != type(other) or self.name != other.name:
            raise TypeError('cannot compare %s and %s' % (self, other))

    def __eq__(self, other):
        self._check_compatible(other)
        return self.key == other.key and self._parts == other._parts

    def __ne__(self, other):
        return not self.__eq__(other)

    # See http://docs.python.org/reference/datamodel#object.__hash__
    def __hash__(self):
        return hash(self.key) + hash(self._parts)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._string)

    def __str__(self):
        return self._string


PEP440_VERSION_RE = re.compile(r'^v?(\d+!)?(\d+(\.\d+)*)((a|alpha|b|beta|c|rc|pre|preview)(\d+)?)?'
                               r'(\.(post|r|rev)(\d+)?)?([._-]?(dev)(\d+)?)?'
                               r'(\+([a-zA-Z\d]+(\.[a-zA-Z\d]+)?))?$', re.I)


def _pep_440_key(s):
    s = s.strip()
    m = PEP440_VERSION_RE.match(s)
    if not m:
        raise UnsupportedVersionError('Not a valid version: %s' % s)
    groups = m.groups()
    nums = tuple(int(v) for v in groups[1].split('.'))
    while len(nums) > 1 and nums[-1] == 0:
        nums = nums[:-1]

    if not groups[0]:
        epoch = 0
    else:
        epoch = int(groups[0][:-1])
    pre = groups[4:6]
    post = groups[7:9]
    dev = groups[10:12]
    local = groups[13]
    if pre == (None, None):
        pre = ()
    else:
        if pre[1] is None:
            pre = pre[0], 0
        else:
            pre = pre[0], int(pre[1])
    if post == (None, None):
        post = ()
    else:
        if post[1] is None:
            post = post[0], 0
        else:
            post = post[0], int(post[1])
    if dev == (None, None):
        dev = ()
    else:
        if dev[1] is None:
            dev = dev[0], 0
        else:
            dev = dev[0], int(dev[1])
    if local is None:
        local = ()
    else:
        parts = []
        for part in local.split('.'):
            # to ensure that numeric compares as > lexicographic, avoid
            # comparing them directly, but encode a tuple which ensures
            # correct sorting
            if part.isdigit():
                part = (1, int(part))
            else:
                part = (0, part)
            parts.append(part)
        local = tuple(parts)
    if not pre:
        # either before pre-release, or final release and after
        if not post and dev:
            # before pre-release
            pre = ('a', -1)     # to sort before a0
        else:
            pre = ('z',)        # to sort after all pre-releases
    # now look at the state of post and dev.
    if not post:
        post = ('_',)   # sort before 'a'
    if not dev:
        dev = ('final',)

    return epoch, nums, pre, post, dev, local


_normalized_key = _pep_440_key


class NormalizedVersion(Version):
    """A rational version.

    Good:
        1.2         # equivalent to "1.2.0"
        1.2.0
        1.2a1
        1.2.3a2
        1.2.3b1
        1.2.3c1
        1.2.3.4
        TODO: fill this out

    Bad:
        1           # minimum two numbers
        1.2a        # release level must have a release serial
        1.2.3b
    """
    def parse(self, s):
        result = _normalized_key(s)
        # _normalized_key loses trailing zeroes in the release
        # clause, since that's needed to ensure that X.Y == X.Y.0 == X.Y.0.0
        # However, PEP 440 prefix matching needs it: for example,
        # (~= 1.4.5.0) matches differently to (~= 1.4.5.0.0).
        m = PEP440_VERSION_RE.match(s)      # must succeed
        groups = m.groups()
        self._release_clause = tuple(int(v) for v in groups[1].split('.'))
        return result

    PREREL_TAGS = set(['a', 'b', 'c', 'rc', 'dev'])

    @property
    def is_prerelease(self):
        return any(t[0] in self.PREREL_TAGS for t in self._parts if t)


def _match_prefix(x, y):
    x = str(x)
    y = str(y)
    if x == y:
        return True
    if not x.startswith(y):
        return False
    n = len(y)
    return x[n] == '.'


class NormalizedMatcher(Matcher):
    version_class = NormalizedVersion

    # value is either a callable or the name of a method
    _operators = {
        '~=': '_match_compatible',
        '<': '_match_lt',
        '>': '_match_gt',
        '<=': '_match_le',
        '>=': '_match_ge',
        '==': '_match_eq',
        '===': '_match_arbitrary',
        '!=': '_match_ne',
    }

    def _adjust_local(self, version, constraint, prefix):
        if prefix:
            strip_local = '+' not in constraint and version._parts[-1]
        else:
            # both constraint and version are
            # NormalizedVersion instances.
            # If constraint does not have a local component,
            # ensure the version doesn't, either.
            strip_local = not constraint._parts[-1] and version._parts[-1]
        if strip_local:
            s = version._string.split('+', 1)[0]
            version = self.version_class(s)
        return version, constraint

    def _match_lt(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version >= constraint:
            return False
        release_clause = constraint._release_clause
        pfx = '.'.join([str(i) for i in release_clause])
        return not _match_prefix(version, pfx)

    def _match_gt(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version <= constraint:
            return False
        release_clause = constraint._release_clause
        pfx = '.'.join([str(i) for i in release_clause])
        return not _match_prefix(version, pfx)

    def _match_le(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        return version <= constraint

    def _match_ge(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        return version >= constraint

    def _match_eq(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if not prefix:
            result = (version == constraint)
        else:
            result = _match_prefix(version, constraint)
        return result

    def _match_arbitrary(self, version, constraint, prefix):
        return str(version) == str(constraint)

    def _match_ne(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if not prefix:
            result = (version != constraint)
        else:
            result = not _match_prefix(version, constraint)
        return result

    def _match_compatible(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version == constraint:
            return True
        if version < constraint:
            return False
#        if not prefix:
#            return True
        release_clause = constraint._release_clause
        if len(release_clause) > 1:
            release_clause = release_clause[:-1]
        pfx = '.'.join([str(i) for i in release_clause])
        return _match_prefix(version, pfx)


_REPLACEMENTS = (
    (re.compile('[.+-]$'), ''),                     # remove trailing puncts
    (re.compile(r'^[.](\d)'), r'0.\1'),             # .N -> 0.N at start
    (re.compile('^[.-]'), ''),                      # remove leading puncts
    (re.compile(r'^\((.*)\)$'), r'\1'),             # remove parentheses
    (re.compile(r'^v(ersion)?\s*(\d+)'), r'\2'),    # remove leading v(ersion)
    (re.compile(r'^r(ev)?\s*(\d+)'), r'\2'),        # remove leading v(ersion)
    (re.compile('[.]{2,}'), '.'),                   # multiple runs of '.'
    (re.compile(r'\b(alfa|apha)\b'), 'alpha'),      # misspelt alpha
    (re.compile(r'\b(pre-alpha|prealpha)\b'),
        'pre.alpha'),                               # standardise
    (re.compile(r'\(beta\)$'), 'beta'),             # remove parentheses
)

_SUFFIX_REPLACEMENTS = (
    (re.compile('^[:~._+-]+'), ''),                   # remove leading puncts
    (re.compile('[,*")([\\]]'), ''),                  # remove unwanted chars
    (re.compile('[~:+_ -]'), '.'),                    # replace illegal chars
    (re.compile('[.]{2,}'), '.'),                   # multiple runs of '.'
    (re.compile(r'\.$'), ''),                       # trailing '.'
)

_NUMERIC_PREFIX = re.compile(r'(\d+(\.\d+)*)')


def _suggest_semantic_version(s):
    """
    Try to suggest a semantic form for a version for which
    _suggest_normalized_version couldn't come up with anything.
    """
    result = s.strip().lower()
    for pat, repl in _REPLACEMENTS:
        result = pat.sub(repl, result)
    if not result:
        result = '0.0.0'

    # Now look for numeric prefix, and separate it out from
    # the rest.
    # import pdb; pdb.set_trace()
    m = _NUMERIC_PREFIX.match(result)
    if not m:
        prefix = '0.0.0'
        suffix = result
    else:
        prefix = m.groups()[0].split('.')
        prefix = [int(i) for i in prefix]
        while len(prefix) < 3:
            prefix.append(0)
        if len(prefix) == 3:
            suffix = result[m.end():]
        else:
            suffix = '.'.join([str(i) for i in prefix[3:]]) + result[m.end():]
            prefix = prefix[:3]
        prefix = '.'.join([str(i) for i in prefix])
        suffix = suffix.strip()
    if suffix:
        # import pdb; pdb.set_trace()
        # massage the suffix.
        for pat, repl in _SUFFIX_REPLACEMENTS:
            suffix = pat.sub(repl, suffix)

    if not suffix:
        result = prefix
    else:
        sep = '-' if 'dev' in suffix else '+'
        result = prefix + sep + suffix
    if not is_semver(result):
        result = None
    return result


def _suggest_normalized_version(s):
    """Suggest a normalized version close to the given version string.

    If you have a version string that isn't rational (i.e. NormalizedVersion
    doesn't like it) then you might be able to get an equivalent (or close)
    rational version from this function.

    This does a number of simple normalizations to the given string, based
    on observation of versions currently in use on PyPI. Given a dump of
    those version during PyCon 2009, 4287 of them:
    - 2312 (53.93%) match NormalizedVersion without change
      with the automatic suggestion
    - 3474 (81.04%) match when using this suggestion method

    @param s {str} An irrational version string.
    @returns A rational version string, or None, if couldn't determine one.
    """
    try:
        _normalized_key(s)
        return s   # already rational
    except UnsupportedVersionError:
        pass

    rs = s.lower()

    # part of this could use maketrans
    for orig, repl in (('-alpha', 'a'), ('-beta', 'b'), ('alpha', 'a'),
                       ('beta', 'b'), ('rc', 'c'), ('-final', ''),
                       ('-pre', 'c'),
                       ('-release', ''), ('.release', ''), ('-stable', ''),
                       ('+', '.'), ('_', '.'), (' ', ''), ('.final', ''),
                       ('final', '')):
        rs = rs.replace(orig, repl)

    # if something ends with dev or pre, we add a 0
    rs = re.sub(r"pre$", r"pre0", rs)
    rs = re.sub(r"dev$", r"dev0", rs)

    # if we have something like "b-2" or "a.2" at the end of the
    # version, that is probably beta, alpha, etc
    # let's remove the dash or dot
    rs = re.sub(r"([abc]|rc)[\-\.](\d+)$", r"\1\2", rs)

    # 1.0-dev-r371 -> 1.0.dev371
    # 0.1-dev-r79 -> 0.1.dev79
    rs = re.sub(r"[\-\.](dev)[\-\.]?r?(\d+)$", r".\1\2", rs)

    # Clean: 2.0.a.3, 2.0.b1, 0.9.0~c1
    rs = re.sub(r"[.~]?([abc])\.?", r"\1", rs)

    # Clean: v0.3, v1.0
    if rs.startswith('v'):
        rs = rs[1:]

    # Clean leading '0's on numbers.
    # TODO: unintended side-effect on, e.g., "2003.05.09"
    # PyPI stats: 77 (~2%) better
    rs = re.sub(r"\b0+(\d+)(?!\d)", r"\1", rs)

    # Clean a/b/c with no version. E.g. "1.0a" -> "1.0a0". Setuptools infers
    # zero.
    # PyPI stats: 245 (7.56%) better
    rs = re.sub(r"(\d+[abc])$", r"\g<1>0", rs)

    # the 'dev-rNNN' tag is a dev tag
    rs = re.sub(r"\.?(dev-r|dev\.r)\.?(\d+)$", r".dev\2", rs)

    # clean the - when used as a pre delimiter
    rs = re.sub(r"-(a|b|c)(\d+)$", r"\1\2", rs)

    # a terminal "dev" or "devel" can be changed into ".dev0"
    rs = re.sub(r"[\.\-](dev|devel)$", r".dev0", rs)

    # a terminal "dev" can be changed into ".dev0"
    rs = re.sub(r"(?![\.\-])dev$", r".dev0", rs)

    # a terminal "final" or "stable" can be removed
    rs = re.sub(r"(final|stable)$", "", rs)

    # The 'r' and the '-' tags are post release tags
    #   0.4a1.r10       ->  0.4a1.post10
    #   0.9.33-17222    ->  0.9.33.post17222
    #   0.9.33-r17222   ->  0.9.33.post17222
    rs = re.sub(r"\.?(r|-|-r)\.?(\d+)$", r".post\2", rs)

    # Clean 'r' instead of 'dev' usage:
    #   0.9.33+r17222   ->  0.9.33.dev17222
    #   1.0dev123       ->  1.0.dev123
    #   1.0.git123      ->  1.0.dev123
    #   1.0.bzr123      ->  1.0.dev123
    #   0.1a0dev.123    ->  0.1a0.dev123
    # PyPI stats:  ~150 (~4%) better
    rs = re.sub(r"\.?(dev|git|bzr)\.?(\d+)$", r".dev\2", rs)

    # Clean '.pre' (normalized from '-pre' above) instead of 'c' usage:
    #   0.2.pre1        ->  0.2c1
    #   0.2-c1         ->  0.2c1
    #   1.0preview123   ->  1.0c123
    # PyPI stats: ~21 (0.62%) better
    rs = re.sub(r"\.?(pre|preview|-c)(\d+)$", r"c\g<2>", rs)

    # Tcl/Tk uses "px" for their post release markers
    rs = re.sub(r"p(\d+)$", r".post\1", rs)

    try:
        _normalized_key(rs)
    except UnsupportedVersionError:
        rs = None
    return rs

#
#   Legacy version processing (distribute-compatible)
#


_VERSION_PART = re.compile(r'([a-z]+|\d+|[\.-])', re.I)
_VERSION_REPLACE = {
    'pre': 'c',
    'preview': 'c',
    '-': 'final-',
    'rc': 'c',
    'dev': '@',
    '': None,
    '.': None,
}


def _legacy_key(s):
    def get_parts(s):
        result = []
        for p in _VERSION_PART.split(s.lower()):
            p = _VERSION_REPLACE.get(p, p)
            if p:
                if '0' <= p[:1] <= '9':
                    p = p.zfill(8)
                else:
                    p = '*' + p
                result.append(p)
        result.append('*final')
        return result

    result = []
    for p in get_parts(s):
        if p.startswith('*'):
            if p < '*final':
                while result and result[-1] == '*final-':
                    result.pop()
            while result and result[-1] == '00000000':
                result.pop()
        result.append(p)
    return tuple(result)


class LegacyVersion(Version):
    def parse(self, s):
        return _legacy_key(s)

    @property
    def is_prerelease(self):
        result = False
        for x in self._parts:
            if (isinstance(x, string_types) and x.startswith('*') and x < '*final'):
                result = True
                break
        return result


class LegacyMatcher(Matcher):
    version_class = LegacyVersion

    _operators = dict(Matcher._operators)
    _operators['~='] = '_match_compatible'

    numeric_re = re.compile(r'^(\d+(\.\d+)*)')

    def _match_compatible(self, version, constraint, prefix):
        if version < constraint:
            return False
        m = self.numeric_re.match(str(constraint))
        if not m:
            logger.warning('Cannot compute compatible match for version %s '
                           ' and constraint %s', version, constraint)
            return True
        s = m.groups()[0]
        if '.' in s:
            s = s.rsplit('.', 1)[0]
        return _match_prefix(version, s)

#
#   Semantic versioning
#


_SEMVER_RE = re.compile(r'^(\d+)\.(\d+)\.(\d+)'
                        r'(-[a-z0-9]+(\.[a-z0-9-]+)*)?'
                        r'(\+[a-z0-9]+(\.[a-z0-9-]+)*)?$', re.I)


def is_semver(s):
    return _SEMVER_RE.match(s)


def _semantic_key(s):
    def make_tuple(s, absent):
        if s is None:
            result = (absent,)
        else:
            parts = s[1:].split('.')
            # We can't compare ints and strings on Python 3, so fudge it
            # by zero-filling numeric values so simulate a numeric comparison
            result = tuple([p.zfill(8) if p.isdigit() else p for p in parts])
        return result

    m = is_semver(s)
    if not m:
        raise UnsupportedVersionError(s)
    groups = m.groups()
    major, minor, patch = [int(i) for i in groups[:3]]
    # choose the '|' and '*' so that versions sort correctly
    pre, build = make_tuple(groups[3], '|'), make_tuple(groups[5], '*')
    return (major, minor, patch), pre, build


class SemanticVersion(Version):
    def parse(self, s):
        return _semantic_key(s)

    @property
    def is_prerelease(self):
        return self._parts[1][0] != '|'


class SemanticMatcher(Matcher):
    version_class = SemanticVersion


class VersionScheme(object):
    def __init__(self, key, matcher, suggester=None):
        self.key = key
        self.matcher = matcher
        self.suggester = suggester

    def is_valid_version(self, s):
        try:
            self.matcher.version_class(s)
            result = True
        except UnsupportedVersionError:
            result = False
        return result

    def is_valid_matcher(self, s):
        try:
            self.matcher(s)
            result = True
        except UnsupportedVersionError:
            result = False
        return result

    def is_valid_constraint_list(self, s):
        """
        Used for processing some metadata fields
        """
        # See issue #140. Be tolerant of a single trailing comma.
        if s.endswith(','):
            s = s[:-1]
        return self.is_valid_matcher('dummy_name (%s)' % s)

    def suggest(self, s):
        if self.suggester is None:
            result = None
        else:
            result = self.suggester(s)
        return result


_SCHEMES = {
    'normalized': VersionScheme(_normalized_key, NormalizedMatcher,
                                _suggest_normalized_version),
    'legacy': VersionScheme(_legacy_key, LegacyMatcher, lambda self, s: s),
    'semantic': VersionScheme(_semantic_key, SemanticMatcher,
                              _suggest_semantic_version),
}

_SCHEMES['default'] = _SCHEMES['normalized']


def get_scheme(name):
    if name not in _SCHEMES:
        raise ValueError('unknown scheme name: %r' % name)
    return _SCHEMES[name]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\getlimits.py ===
"""Machine limits for Float32 and Float64 and (long double) if available...

"""
__all__ = ['finfo', 'iinfo']

import types
import warnings

from numpy._utils import set_module

from . import numeric
from . import numerictypes as ntypes
from ._machar import MachAr
from .numeric import array, inf, nan
from .umath import exp2, isnan, log10, nextafter


def _fr0(a):
    """fix rank-0 --> rank-1"""
    if a.ndim == 0:
        a = a.copy()
        a.shape = (1,)
    return a


def _fr1(a):
    """fix rank > 0 --> rank-0"""
    if a.size == 1:
        a = a.copy()
        a.shape = ()
    return a


class MachArLike:
    """ Object to simulate MachAr instance """
    def __init__(self, ftype, *, eps, epsneg, huge, tiny,
                 ibeta, smallest_subnormal=None, **kwargs):
        self.params = _MACHAR_PARAMS[ftype]
        self.ftype = ftype
        self.title = self.params['title']
        # Parameter types same as for discovered MachAr object.
        if not smallest_subnormal:
            self._smallest_subnormal = nextafter(
                self.ftype(0), self.ftype(1), dtype=self.ftype)
        else:
            self._smallest_subnormal = smallest_subnormal
        self.epsilon = self.eps = self._float_to_float(eps)
        self.epsneg = self._float_to_float(epsneg)
        self.xmax = self.huge = self._float_to_float(huge)
        self.xmin = self._float_to_float(tiny)
        self.smallest_normal = self.tiny = self._float_to_float(tiny)
        self.ibeta = self.params['itype'](ibeta)
        self.__dict__.update(kwargs)
        self.precision = int(-log10(self.eps))
        self.resolution = self._float_to_float(
            self._float_conv(10) ** (-self.precision))
        self._str_eps = self._float_to_str(self.eps)
        self._str_epsneg = self._float_to_str(self.epsneg)
        self._str_xmin = self._float_to_str(self.xmin)
        self._str_xmax = self._float_to_str(self.xmax)
        self._str_resolution = self._float_to_str(self.resolution)
        self._str_smallest_normal = self._float_to_str(self.xmin)

    @property
    def smallest_subnormal(self):
        """Return the value for the smallest subnormal.

        Returns
        -------
        smallest_subnormal : float
            value for the smallest subnormal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest subnormal is zero.
        """
        # Check that the calculated value is not zero, in case it raises a
        # warning.
        value = self._smallest_subnormal
        if self.ftype(0) == value:
            warnings.warn(
                f'The value of the smallest subnormal for {self.ftype} type is zero.',
                UserWarning, stacklevel=2)

        return self._float_to_float(value)

    @property
    def _str_smallest_subnormal(self):
        """Return the string representation of the smallest subnormal."""
        return self._float_to_str(self.smallest_subnormal)

    def _float_to_float(self, value):
        """Converts float to float.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        return _fr1(self._float_conv(value))

    def _float_conv(self, value):
        """Converts float to conv.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        return array([value], self.ftype)

    def _float_to_str(self, value):
        """Converts float to str.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        return self.params['fmt'] % array(_fr0(value)[0], self.ftype)


_convert_to_float = {
    ntypes.csingle: ntypes.single,
    ntypes.complex128: ntypes.float64,
    ntypes.clongdouble: ntypes.longdouble
    }

# Parameters for creating MachAr / MachAr-like objects
_title_fmt = 'numpy {} precision floating point number'
_MACHAR_PARAMS = {
    ntypes.double: {
        'itype': ntypes.int64,
        'fmt': '%24.16e',
        'title': _title_fmt.format('double')},
    ntypes.single: {
        'itype': ntypes.int32,
        'fmt': '%15.7e',
        'title': _title_fmt.format('single')},
    ntypes.longdouble: {
        'itype': ntypes.longlong,
        'fmt': '%s',
        'title': _title_fmt.format('long double')},
    ntypes.half: {
        'itype': ntypes.int16,
        'fmt': '%12.5e',
        'title': _title_fmt.format('half')}}

# Key to identify the floating point type.  Key is result of
#
#    ftype = np.longdouble        # or float64, float32, etc.
#    v = (ftype(-1.0) / ftype(10.0))
#    v.view(v.dtype.newbyteorder('<')).tobytes()
#
# Uses division to work around deficiencies in strtold on some platforms.
# See:
# https://perl5.git.perl.org/perl.git/blob/3118d7d684b56cbeb702af874f4326683c45f045:/Configure

_KNOWN_TYPES = {}
def _register_type(machar, bytepat):
    _KNOWN_TYPES[bytepat] = machar


_float_ma = {}


def _register_known_types():
    # Known parameters for float16
    # See docstring of MachAr class for description of parameters.
    f16 = ntypes.float16
    float16_ma = MachArLike(f16,
                            machep=-10,
                            negep=-11,
                            minexp=-14,
                            maxexp=16,
                            it=10,
                            iexp=5,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(f16(-10)),
                            epsneg=exp2(f16(-11)),
                            huge=f16(65504),
                            tiny=f16(2 ** -14))
    _register_type(float16_ma, b'f\xae')
    _float_ma[16] = float16_ma

    # Known parameters for float32
    f32 = ntypes.float32
    float32_ma = MachArLike(f32,
                            machep=-23,
                            negep=-24,
                            minexp=-126,
                            maxexp=128,
                            it=23,
                            iexp=8,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(f32(-23)),
                            epsneg=exp2(f32(-24)),
                            huge=f32((1 - 2 ** -24) * 2**128),
                            tiny=exp2(f32(-126)))
    _register_type(float32_ma, b'\xcd\xcc\xcc\xbd')
    _float_ma[32] = float32_ma

    # Known parameters for float64
    f64 = ntypes.float64
    epsneg_f64 = 2.0 ** -53.0
    tiny_f64 = 2.0 ** -1022.0
    float64_ma = MachArLike(f64,
                            machep=-52,
                            negep=-53,
                            minexp=-1022,
                            maxexp=1024,
                            it=52,
                            iexp=11,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=2.0 ** -52.0,
                            epsneg=epsneg_f64,
                            huge=(1.0 - epsneg_f64) / tiny_f64 * f64(4),
                            tiny=tiny_f64)
    _register_type(float64_ma, b'\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    _float_ma[64] = float64_ma

    # Known parameters for IEEE 754 128-bit binary float
    ld = ntypes.longdouble
    epsneg_f128 = exp2(ld(-113))
    tiny_f128 = exp2(ld(-16382))
    # Ignore runtime error when this is not f128
    with numeric.errstate(all='ignore'):
        huge_f128 = (ld(1) - epsneg_f128) / tiny_f128 * ld(4)
    float128_ma = MachArLike(ld,
                             machep=-112,
                             negep=-113,
                             minexp=-16382,
                             maxexp=16384,
                             it=112,
                             iexp=15,
                             ibeta=2,
                             irnd=5,
                             ngrd=0,
                             eps=exp2(ld(-112)),
                             epsneg=epsneg_f128,
                             huge=huge_f128,
                             tiny=tiny_f128)
    # IEEE 754 128-bit binary float
    _register_type(float128_ma,
        b'\x9a\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\xfb\xbf')
    _float_ma[128] = float128_ma

    # Known parameters for float80 (Intel 80-bit extended precision)
    epsneg_f80 = exp2(ld(-64))
    tiny_f80 = exp2(ld(-16382))
    # Ignore runtime error when this is not f80
    with numeric.errstate(all='ignore'):
        huge_f80 = (ld(1) - epsneg_f80) / tiny_f80 * ld(4)
    float80_ma = MachArLike(ld,
                            machep=-63,
                            negep=-64,
                            minexp=-16382,
                            maxexp=16384,
                            it=63,
                            iexp=15,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(ld(-63)),
                            epsneg=epsneg_f80,
                            huge=huge_f80,
                            tiny=tiny_f80)
    # float80, first 10 bytes containing actual storage
    _register_type(float80_ma, b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf')
    _float_ma[80] = float80_ma

    # Guessed / known parameters for double double; see:
    # https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic
    # These numbers have the same exponent range as float64, but extended
    # number of digits in the significand.
    huge_dd = nextafter(ld(inf), ld(0), dtype=ld)
    # As the smallest_normal in double double is so hard to calculate we set
    # it to NaN.
    smallest_normal_dd = nan
    # Leave the same value for the smallest subnormal as double
    smallest_subnormal_dd = ld(nextafter(0., 1.))
    float_dd_ma = MachArLike(ld,
                             machep=-105,
                             negep=-106,
                             minexp=-1022,
                             maxexp=1024,
                             it=105,
                             iexp=11,
                             ibeta=2,
                             irnd=5,
                             ngrd=0,
                             eps=exp2(ld(-105)),
                             epsneg=exp2(ld(-106)),
                             huge=huge_dd,
                             tiny=smallest_normal_dd,
                             smallest_subnormal=smallest_subnormal_dd)
    # double double; low, high order (e.g. PPC 64)
    _register_type(float_dd_ma,
        b'\x9a\x99\x99\x99\x99\x99Y<\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    # double double; high, low order (e.g. PPC 64 le)
    _register_type(float_dd_ma,
        b'\x9a\x99\x99\x99\x99\x99\xb9\xbf\x9a\x99\x99\x99\x99\x99Y<')
    _float_ma['dd'] = float_dd_ma


def _get_machar(ftype):
    """ Get MachAr instance or MachAr-like instance

    Get parameters for floating point type, by first trying signatures of
    various known floating point types, then, if none match, attempting to
    identify parameters by analysis.

    Parameters
    ----------
    ftype : class
        Numpy floating point type class (e.g. ``np.float64``)

    Returns
    -------
    ma_like : instance of :class:`MachAr` or :class:`MachArLike`
        Object giving floating point parameters for `ftype`.

    Warns
    -----
    UserWarning
        If the binary signature of the float type is not in the dictionary of
        known float types.
    """
    params = _MACHAR_PARAMS.get(ftype)
    if params is None:
        raise ValueError(repr(ftype))
    # Detect known / suspected types
    # ftype(-1.0) / ftype(10.0) is better than ftype('-0.1') because stold
    # may be deficient
    key = (ftype(-1.0) / ftype(10.))
    key = key.view(key.dtype.newbyteorder("<")).tobytes()
    ma_like = None
    if ftype == ntypes.longdouble:
        # Could be 80 bit == 10 byte extended precision, where last bytes can
        # be random garbage.
        # Comparing first 10 bytes to pattern first to avoid branching on the
        # random garbage.
        ma_like = _KNOWN_TYPES.get(key[:10])
    if ma_like is None:
        # see if the full key is known.
        ma_like = _KNOWN_TYPES.get(key)
    if ma_like is None and len(key) == 16:
        # machine limits could be f80 masquerading as np.float128,
        # find all keys with length 16 and make new dict, but make the keys
        # only 10 bytes long, the last bytes can be random garbage
        _kt = {k[:10]: v for k, v in _KNOWN_TYPES.items() if len(k) == 16}
        ma_like = _kt.get(key[:10])
    if ma_like is not None:
        return ma_like
    # Fall back to parameter discovery
    warnings.warn(
        f'Signature {key} for {ftype} does not match any known type: '
        'falling back to type probe function.\n'
        'This warnings indicates broken support for the dtype!',
        UserWarning, stacklevel=2)
    return _discovered_machar(ftype)


def _discovered_machar(ftype):
    """ Create MachAr instance with found information on float types

    TODO: MachAr should be retired completely ideally.  We currently only
          ever use it system with broken longdouble (valgrind, WSL).
    """
    params = _MACHAR_PARAMS[ftype]
    return MachAr(lambda v: array([v], ftype),
                  lambda v: _fr0(v.astype(params['itype']))[0],
                  lambda v: array(_fr0(v)[0], ftype),
                  lambda v: params['fmt'] % array(_fr0(v)[0], ftype),
                  params['title'])


@set_module('numpy')
class finfo:
    """
    finfo(dtype)

    Machine limits for floating point types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    dtype : dtype
        Returns the dtype for which `finfo` returns information. For complex
        input, the returned dtype is the associated ``float*`` dtype for its
        real and complex components.
    eps : float
        The difference between 1.0 and the next smallest representable float
        larger than 1.0. For example, for 64-bit binary floats in the IEEE-754
        standard, ``eps = 2**-52``, approximately 2.22e-16.
    epsneg : float
        The difference between 1.0 and the next smallest representable float
        less than 1.0. For example, for 64-bit binary floats in the IEEE-754
        standard, ``epsneg = 2**-53``, approximately 1.11e-16.
    iexp : int
        The number of bits in the exponent portion of the floating point
        representation.
    machep : int
        The exponent that yields `eps`.
    max : floating point number of the appropriate type
        The largest representable number.
    maxexp : int
        The smallest positive power of the base (2) that causes overflow.
    min : floating point number of the appropriate type
        The smallest representable number, typically ``-max``.
    minexp : int
        The most negative power of the base (2) consistent with there
        being no leading 0's in the mantissa.
    negep : int
        The exponent that yields `epsneg`.
    nexp : int
        The number of bits in the exponent including its sign and bias.
    nmant : int
        The number of bits in the mantissa.
    precision : int
        The approximate number of decimal digits to which this kind of
        float is precise.
    resolution : floating point number of the appropriate type
        The approximate decimal resolution of this type, i.e.,
        ``10**-precision``.
    tiny : float
        An alias for `smallest_normal`, kept for backwards compatibility.
    smallest_normal : float
        The smallest positive floating point number with 1 as leading bit in
        the mantissa following IEEE-754 (see Notes).
    smallest_subnormal : float
        The smallest positive floating point number with 0 as leading bit in
        the mantissa following IEEE-754.

    Parameters
    ----------
    dtype : float, dtype, or instance
        Kind of floating point or complex floating point
        data-type about which to get information.

    See Also
    --------
    iinfo : The equivalent for integer data types.
    spacing : The distance between a value and the nearest adjacent number
    nextafter : The next floating point value after x1 towards x2

    Notes
    -----
    For developers of NumPy: do not instantiate this at the module level.
    The initial calculation of these parameters is expensive and negatively
    impacts import times.  These objects are cached, so calling ``finfo()``
    repeatedly inside your functions is not a problem.

    Note that ``smallest_normal`` is not actually the smallest positive
    representable value in a NumPy floating point type. As in the IEEE-754
    standard [1]_, NumPy floating point types make use of subnormal numbers to
    fill the gap between 0 and ``smallest_normal``. However, subnormal numbers
    may have significantly reduced precision [2]_.

    This function can also be used for complex data types as well. If used,
    the output will be the same as the corresponding real float type
    (e.g. numpy.finfo(numpy.csingle) is the same as numpy.finfo(numpy.single)).
    However, the output is true for the real and imaginary components.

    References
    ----------
    .. [1] IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2008,
           pp.1-70, 2008, https://doi.org/10.1109/IEEESTD.2008.4610935
    .. [2] Wikipedia, "Denormal Numbers",
           https://en.wikipedia.org/wiki/Denormal_number

    Examples
    --------
    >>> import numpy as np
    >>> np.finfo(np.float64).dtype
    dtype('float64')
    >>> np.finfo(np.complex64).dtype
    dtype('float32')

    """

    _finfo_cache = {}

    __class_getitem__ = classmethod(types.GenericAlias)

    def __new__(cls, dtype):
        try:
            obj = cls._finfo_cache.get(dtype)  # most common path
            if obj is not None:
                return obj
        except TypeError:
            pass

        if dtype is None:
            # Deprecated in NumPy 1.25, 2023-01-16
            warnings.warn(
                "finfo() dtype cannot be None. This behavior will "
                "raise an error in the future. (Deprecated in NumPy 1.25)",
                DeprecationWarning,
                stacklevel=2
            )

        try:
            dtype = numeric.dtype(dtype)
        except TypeError:
            # In case a float instance was given
            dtype = numeric.dtype(type(dtype))

        obj = cls._finfo_cache.get(dtype)
        if obj is not None:
            return obj
        dtypes = [dtype]
        newdtype = ntypes.obj2sctype(dtype)
        if newdtype is not dtype:
            dtypes.append(newdtype)
            dtype = newdtype
        if not issubclass(dtype, numeric.inexact):
            raise ValueError(f"data type {dtype!r} not inexact")
        obj = cls._finfo_cache.get(dtype)
        if obj is not None:
            return obj
        if not issubclass(dtype, numeric.floating):
            newdtype = _convert_to_float[dtype]
            if newdtype is not dtype:
                # dtype changed, for example from complex128 to float64
                dtypes.append(newdtype)
                dtype = newdtype

                obj = cls._finfo_cache.get(dtype, None)
                if obj is not None:
                    # the original dtype was not in the cache, but the new
                    # dtype is in the cache. we add the original dtypes to
                    # the cache and return the result
                    for dt in dtypes:
                        cls._finfo_cache[dt] = obj
                    return obj
        obj = object.__new__(cls)._init(dtype)
        for dt in dtypes:
            cls._finfo_cache[dt] = obj
        return obj

    def _init(self, dtype):
        self.dtype = numeric.dtype(dtype)
        machar = _get_machar(dtype)

        for word in ['precision', 'iexp',
                     'maxexp', 'minexp', 'negep',
                     'machep']:
            setattr(self, word, getattr(machar, word))
        for word in ['resolution', 'epsneg', 'smallest_subnormal']:
            setattr(self, word, getattr(machar, word).flat[0])
        self.bits = self.dtype.itemsize * 8
        self.max = machar.huge.flat[0]
        self.min = -self.max
        self.eps = machar.eps.flat[0]
        self.nexp = machar.iexp
        self.nmant = machar.it
        self._machar = machar
        self._str_tiny = machar._str_xmin.strip()
        self._str_max = machar._str_xmax.strip()
        self._str_epsneg = machar._str_epsneg.strip()
        self._str_eps = machar._str_eps.strip()
        self._str_resolution = machar._str_resolution.strip()
        self._str_smallest_normal = machar._str_smallest_normal.strip()
        self._str_smallest_subnormal = machar._str_smallest_subnormal.strip()
        return self

    def __str__(self):
        fmt = (
            'Machine parameters for %(dtype)s\n'
            '---------------------------------------------------------------\n'
            'precision = %(precision)3s   resolution = %(_str_resolution)s\n'
            'machep = %(machep)6s   eps =        %(_str_eps)s\n'
            'negep =  %(negep)6s   epsneg =     %(_str_epsneg)s\n'
            'minexp = %(minexp)6s   tiny =       %(_str_tiny)s\n'
            'maxexp = %(maxexp)6s   max =        %(_str_max)s\n'
            'nexp =   %(nexp)6s   min =        -max\n'
            'smallest_normal = %(_str_smallest_normal)s   '
            'smallest_subnormal = %(_str_smallest_subnormal)s\n'
            '---------------------------------------------------------------\n'
            )
        return fmt % self.__dict__

    def __repr__(self):
        c = self.__class__.__name__
        d = self.__dict__.copy()
        d['klass'] = c
        return (("%(klass)s(resolution=%(resolution)s, min=-%(_str_max)s,"
                 " max=%(_str_max)s, dtype=%(dtype)s)") % d)

    @property
    def smallest_normal(self):
        """Return the value for the smallest normal.

        Returns
        -------
        smallest_normal : float
            Value for the smallest normal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest normal is requested for
            double-double.
        """
        # This check is necessary because the value for smallest_normal is
        # platform dependent for longdouble types.
        if isnan(self._machar.smallest_normal.flat[0]):
            warnings.warn(
                'The value of smallest normal is undefined for double double',
                UserWarning, stacklevel=2)
        return self._machar.smallest_normal.flat[0]

    @property
    def tiny(self):
        """Return the value for tiny, alias of smallest_normal.

        Returns
        -------
        tiny : float
            Value for the smallest normal, alias of smallest_normal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest normal is requested for
            double-double.
        """
        return self.smallest_normal


@set_module('numpy')
class iinfo:
    """
    iinfo(type)

    Machine limits for integer types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    dtype : dtype
        Returns the dtype for which `iinfo` returns information.
    min : int
        The smallest integer expressible by the type.
    max : int
        The largest integer expressible by the type.

    Parameters
    ----------
    int_type : integer type, dtype, or instance
        The kind of integer data type to get information about.

    See Also
    --------
    finfo : The equivalent for floating point data types.

    Examples
    --------
    With types:

    >>> import numpy as np
    >>> ii16 = np.iinfo(np.int16)
    >>> ii16.min
    -32768
    >>> ii16.max
    32767
    >>> ii32 = np.iinfo(np.int32)
    >>> ii32.min
    -2147483648
    >>> ii32.max
    2147483647

    With instances:

    >>> ii32 = np.iinfo(np.int32(10))
    >>> ii32.min
    -2147483648
    >>> ii32.max
    2147483647

    """

    _min_vals = {}
    _max_vals = {}

    __class_getitem__ = classmethod(types.GenericAlias)

    def __init__(self, int_type):
        try:
            self.dtype = numeric.dtype(int_type)
        except TypeError:
            self.dtype = numeric.dtype(type(int_type))
        self.kind = self.dtype.kind
        self.bits = self.dtype.itemsize * 8
        self.key = "%s%d" % (self.kind, self.bits)
        if self.kind not in 'iu':
            raise ValueError(f"Invalid integer data type {self.kind!r}.")

    @property
    def min(self):
        """Minimum value of given dtype."""
        if self.kind == 'u':
            return 0
        else:
            try:
                val = iinfo._min_vals[self.key]
            except KeyError:
                val = int(-(1 << (self.bits - 1)))
                iinfo._min_vals[self.key] = val
            return val

    @property
    def max(self):
        """Maximum value of given dtype."""
        try:
            val = iinfo._max_vals[self.key]
        except KeyError:
            if self.kind == 'u':
                val = int((1 << self.bits) - 1)
            else:
                val = int((1 << (self.bits - 1)) - 1)
            iinfo._max_vals[self.key] = val
        return val

    def __str__(self):
        """String representation."""
        fmt = (
            'Machine parameters for %(dtype)s\n'
            '---------------------------------------------------------------\n'
            'min = %(min)s\n'
            'max = %(max)s\n'
            '---------------------------------------------------------------\n'
            )
        return fmt % {'dtype': self.dtype, 'min': self.min, 'max': self.max}

    def __repr__(self):
        return "%s(min=%s, max=%s, dtype=%s)" % (self.__class__.__name__,
                                    self.min, self.max, self.dtype)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\sorting.py ===
""" miscellaneous sorting / groupby utilities """
from __future__ import annotations

from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Callable,
    DefaultDict,
    cast,
)

import numpy as np

from pandas._libs import (
    algos,
    hashtable,
    lib,
)
from pandas._libs.hashtable import unique_label_indices

from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
)
from pandas.core.dtypes.generic import (
    ABCMultiIndex,
    ABCRangeIndex,
)
from pandas.core.dtypes.missing import isna

from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        IndexKeyFunc,
        Level,
        NaPosition,
        Shape,
        SortKind,
        npt,
    )

    from pandas import (
        MultiIndex,
        Series,
    )
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index


def get_indexer_indexer(
    target: Index,
    level: Level | list[Level] | None,
    ascending: list[bool] | bool,
    kind: SortKind,
    na_position: NaPosition,
    sort_remaining: bool,
    key: IndexKeyFunc,
) -> npt.NDArray[np.intp] | None:
    """
    Helper method that return the indexer according to input parameters for
    the sort_index method of DataFrame and Series.

    Parameters
    ----------
    target : Index
    level : int or level name or list of ints or list of level names
    ascending : bool or list of bools, default True
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}
    na_position : {'first', 'last'}
    sort_remaining : bool
    key : callable, optional

    Returns
    -------
    Optional[ndarray[intp]]
        The indexer for the new index.
    """

    # error: Incompatible types in assignment (expression has type
    # "Union[ExtensionArray, ndarray[Any, Any], Index, Series]", variable has
    # type "Index")
    target = ensure_key_mapped(target, key, levels=level)  # type: ignore[assignment]
    target = target._sort_levels_monotonic()

    if level is not None:
        _, indexer = target.sortlevel(
            level,
            ascending=ascending,
            sort_remaining=sort_remaining,
            na_position=na_position,
        )
    elif (np.all(ascending) and target.is_monotonic_increasing) or (
        not np.any(ascending) and target.is_monotonic_decreasing
    ):
        # Check monotonic-ness before sort an index (GH 11080)
        return None
    elif isinstance(target, ABCMultiIndex):
        codes = [lev.codes for lev in target._get_codes_for_sorting()]
        indexer = lexsort_indexer(
            codes, orders=ascending, na_position=na_position, codes_given=True
        )
    else:
        # ascending can only be a Sequence for MultiIndex
        indexer = nargsort(
            target,
            kind=kind,
            ascending=cast(bool, ascending),
            na_position=na_position,
        )
    return indexer


def get_group_index(
    labels, shape: Shape, sort: bool, xnull: bool
) -> npt.NDArray[np.int64]:
    """
    For the particular label_list, gets the offsets into the hypothetical list
    representing the totally ordered cartesian product of all possible label
    combinations, *as long as* this space fits within int64 bounds;
    otherwise, though group indices identify unique combinations of
    labels, they cannot be deconstructed.
    - If `sort`, rank of returned ids preserve lexical ranks of labels.
      i.e. returned id's can be used to do lexical sort on labels;
    - If `xnull` nulls (-1 labels) are passed through.

    Parameters
    ----------
    labels : sequence of arrays
        Integers identifying levels at each location
    shape : tuple[int, ...]
        Number of unique levels at each location
    sort : bool
        If the ranks of returned ids should match lexical ranks of labels
    xnull : bool
        If true nulls are excluded. i.e. -1 values in the labels are
        passed through.

    Returns
    -------
    An array of type int64 where two elements are equal if their corresponding
    labels are equal at all location.

    Notes
    -----
    The length of `labels` and `shape` must be identical.
    """

    def _int64_cut_off(shape) -> int:
        acc = 1
        for i, mul in enumerate(shape):
            acc *= int(mul)
            if not acc < lib.i8max:
                return i
        return len(shape)

    def maybe_lift(lab, size: int) -> tuple[np.ndarray, int]:
        # promote nan values (assigned -1 label in lab array)
        # so that all output values are non-negative
        return (lab + 1, size + 1) if (lab == -1).any() else (lab, size)

    labels = [ensure_int64(x) for x in labels]
    lshape = list(shape)
    if not xnull:
        for i, (lab, size) in enumerate(zip(labels, shape)):
            labels[i], lshape[i] = maybe_lift(lab, size)

    labels = list(labels)

    # Iteratively process all the labels in chunks sized so less
    # than lib.i8max unique int ids will be required for each chunk
    while True:
        # how many levels can be done without overflow:
        nlev = _int64_cut_off(lshape)

        # compute flat ids for the first `nlev` levels
        stride = np.prod(lshape[1:nlev], dtype="i8")
        out = stride * labels[0].astype("i8", subok=False, copy=False)

        for i in range(1, nlev):
            if lshape[i] == 0:
                stride = np.int64(0)
            else:
                stride //= lshape[i]
            out += labels[i] * stride

        if xnull:  # exclude nulls
            mask = labels[0] == -1
            for lab in labels[1:nlev]:
                mask |= lab == -1
            out[mask] = -1

        if nlev == len(lshape):  # all levels done!
            break

        # compress what has been done so far in order to avoid overflow
        # to retain lexical ranks, obs_ids should be sorted
        comp_ids, obs_ids = compress_group_index(out, sort=sort)

        labels = [comp_ids] + labels[nlev:]
        lshape = [len(obs_ids)] + lshape[nlev:]

    return out


def get_compressed_ids(
    labels, sizes: Shape
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters
    ----------
    labels : list of label arrays
    sizes : tuple[int] of size of the levels

    Returns
    -------
    np.ndarray[np.intp]
        comp_ids
    np.ndarray[np.int64]
        obs_group_ids
    """
    ids = get_group_index(labels, sizes, sort=True, xnull=False)
    return compress_group_index(ids, sort=True)


def is_int64_overflow_possible(shape: Shape) -> bool:
    the_prod = 1
    for x in shape:
        the_prod *= int(x)

    return the_prod >= lib.i8max


def _decons_group_index(
    comp_labels: npt.NDArray[np.intp], shape: Shape
) -> list[npt.NDArray[np.intp]]:
    # reconstruct labels
    if is_int64_overflow_possible(shape):
        # at some point group indices are factorized,
        # and may not be deconstructed here! wrong path!
        raise ValueError("cannot deconstruct factorized group indices!")

    label_list = []
    factor = 1
    y = np.array(0)
    x = comp_labels
    for i in reversed(range(len(shape))):
        labels = (x - y) % (factor * shape[i]) // factor
        np.putmask(labels, comp_labels < 0, -1)
        label_list.append(labels)
        y = labels * factor
        factor *= shape[i]
    return label_list[::-1]


def decons_obs_group_ids(
    comp_ids: npt.NDArray[np.intp],
    obs_ids: npt.NDArray[np.intp],
    shape: Shape,
    labels: Sequence[npt.NDArray[np.signedinteger]],
    xnull: bool,
) -> list[npt.NDArray[np.intp]]:
    """
    Reconstruct labels from observed group ids.

    Parameters
    ----------
    comp_ids : np.ndarray[np.intp]
    obs_ids: np.ndarray[np.intp]
    shape : tuple[int]
    labels : Sequence[np.ndarray[np.signedinteger]]
    xnull : bool
        If nulls are excluded; i.e. -1 labels are passed through.
    """
    if not xnull:
        lift = np.fromiter(((a == -1).any() for a in labels), dtype=np.intp)
        arr_shape = np.asarray(shape, dtype=np.intp) + lift
        shape = tuple(arr_shape)

    if not is_int64_overflow_possible(shape):
        # obs ids are deconstructable! take the fast route!
        out = _decons_group_index(obs_ids, shape)
        return out if xnull or not lift.any() else [x - y for x, y in zip(out, lift)]

    indexer = unique_label_indices(comp_ids)
    return [lab[indexer].astype(np.intp, subok=False, copy=True) for lab in labels]


def lexsort_indexer(
    keys: Sequence[ArrayLike | Index | Series],
    orders=None,
    na_position: str = "last",
    key: Callable | None = None,
    codes_given: bool = False,
) -> npt.NDArray[np.intp]:
    """
    Performs lexical sorting on a set of keys

    Parameters
    ----------
    keys : Sequence[ArrayLike | Index | Series]
        Sequence of arrays to be sorted by the indexer
        Sequence[Series] is only if key is not None.
    orders : bool or list of booleans, optional
        Determines the sorting order for each element in keys. If a list,
        it must be the same length as keys. This determines whether the
        corresponding element in keys should be sorted in ascending
        (True) or descending (False) order. if bool, applied to all
        elements as above. if None, defaults to True.
    na_position : {'first', 'last'}, default 'last'
        Determines placement of NA elements in the sorted list ("last" or "first")
    key : Callable, optional
        Callable key function applied to every element in keys before sorting
    codes_given: bool, False
        Avoid categorical materialization if codes are already provided.

    Returns
    -------
    np.ndarray[np.intp]
    """
    from pandas.core.arrays import Categorical

    if na_position not in ["last", "first"]:
        raise ValueError(f"invalid na_position: {na_position}")

    if isinstance(orders, bool):
        orders = [orders] * len(keys)
    elif orders is None:
        orders = [True] * len(keys)

    labels = []

    for k, order in zip(keys, orders):
        k = ensure_key_mapped(k, key)
        if codes_given:
            codes = cast(np.ndarray, k)
            n = codes.max() + 1 if len(codes) else 0
        else:
            cat = Categorical(k, ordered=True)
            codes = cat.codes
            n = len(cat.categories)

        mask = codes == -1

        if na_position == "last" and mask.any():
            codes = np.where(mask, n, codes)

        # not order means descending
        if not order:
            codes = np.where(mask, codes, n - codes - 1)

        labels.append(codes)

    return np.lexsort(labels[::-1])


def nargsort(
    items: ArrayLike | Index | Series,
    kind: SortKind = "quicksort",
    ascending: bool = True,
    na_position: str = "last",
    key: Callable | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.intp]:
    """
    Intended to be a drop-in replacement for np.argsort which handles NaNs.

    Adds ascending, na_position, and key parameters.

    (GH #6399, #5231, #27237)

    Parameters
    ----------
    items : np.ndarray, ExtensionArray, Index, or Series
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
    ascending : bool, default True
    na_position : {'first', 'last'}, default 'last'
    key : Optional[Callable], default None
    mask : Optional[np.ndarray[bool]], default None
        Passed when called by ExtensionArray.argsort.

    Returns
    -------
    np.ndarray[np.intp]
    """

    if key is not None:
        # see TestDataFrameSortKey, TestRangeIndex::test_sort_values_key
        items = ensure_key_mapped(items, key)
        return nargsort(
            items,
            kind=kind,
            ascending=ascending,
            na_position=na_position,
            key=None,
            mask=mask,
        )

    if isinstance(items, ABCRangeIndex):
        return items.argsort(ascending=ascending)
    elif not isinstance(items, ABCMultiIndex):
        items = extract_array(items)
    else:
        raise TypeError(
            "nargsort does not support MultiIndex. Use index.sort_values instead."
        )

    if mask is None:
        mask = np.asarray(isna(items))

    if not isinstance(items, np.ndarray):
        # i.e. ExtensionArray
        return items.argsort(
            ascending=ascending,
            kind=kind,
            na_position=na_position,
        )

    idx = np.arange(len(items))
    non_nans = items[~mask]
    non_nan_idx = idx[~mask]

    nan_idx = np.nonzero(mask)[0]
    if not ascending:
        non_nans = non_nans[::-1]
        non_nan_idx = non_nan_idx[::-1]
    indexer = non_nan_idx[non_nans.argsort(kind=kind)]
    if not ascending:
        indexer = indexer[::-1]
    # Finally, place the NaNs at the end or the beginning according to
    # na_position
    if na_position == "last":
        indexer = np.concatenate([indexer, nan_idx])
    elif na_position == "first":
        indexer = np.concatenate([nan_idx, indexer])
    else:
        raise ValueError(f"invalid na_position: {na_position}")
    return ensure_platform_int(indexer)


def nargminmax(values: ExtensionArray, method: str, axis: AxisInt = 0):
    """
    Implementation of np.argmin/argmax but for ExtensionArray and which
    handles missing values.

    Parameters
    ----------
    values : ExtensionArray
    method : {"argmax", "argmin"}
    axis : int, default 0

    Returns
    -------
    int
    """
    assert method in {"argmax", "argmin"}
    func = np.argmax if method == "argmax" else np.argmin

    mask = np.asarray(isna(values))
    arr_values = values._values_for_argsort()

    if arr_values.ndim > 1:
        if mask.any():
            if axis == 1:
                zipped = zip(arr_values, mask)
            else:
                zipped = zip(arr_values.T, mask.T)
            return np.array([_nanargminmax(v, m, func) for v, m in zipped])
        return func(arr_values, axis=axis)

    return _nanargminmax(arr_values, mask, func)


def _nanargminmax(values: np.ndarray, mask: npt.NDArray[np.bool_], func) -> int:
    """
    See nanargminmax.__doc__.
    """
    idx = np.arange(values.shape[0])
    non_nans = values[~mask]
    non_nan_idx = idx[~mask]

    return non_nan_idx[func(non_nans)]


def _ensure_key_mapped_multiindex(
    index: MultiIndex, key: Callable, level=None
) -> MultiIndex:
    """
    Returns a new MultiIndex in which key has been applied
    to all levels specified in level (or all levels if level
    is None). Used for key sorting for MultiIndex.

    Parameters
    ----------
    index : MultiIndex
        Index to which to apply the key function on the
        specified levels.
    key : Callable
        Function that takes an Index and returns an Index of
        the same shape. This key is applied to each level
        separately. The name of the level can be used to
        distinguish different levels for application.
    level : list-like, int or str, default None
        Level or list of levels to apply the key function to.
        If None, key function is applied to all levels. Other
        levels are left unchanged.

    Returns
    -------
    labels : MultiIndex
        Resulting MultiIndex with modified levels.
    """

    if level is not None:
        if isinstance(level, (str, int)):
            sort_levels = [level]
        else:
            sort_levels = level

        sort_levels = [index._get_level_number(lev) for lev in sort_levels]
    else:
        sort_levels = list(range(index.nlevels))  # satisfies mypy

    mapped = [
        ensure_key_mapped(index._get_level_values(level), key)
        if level in sort_levels
        else index._get_level_values(level)
        for level in range(index.nlevels)
    ]

    return type(index).from_arrays(mapped)


def ensure_key_mapped(
    values: ArrayLike | Index | Series, key: Callable | None, levels=None
) -> ArrayLike | Index | Series:
    """
    Applies a callable key function to the values function and checks
    that the resulting value has the same shape. Can be called on Index
    subclasses, Series, DataFrames, or ndarrays.

    Parameters
    ----------
    values : Series, DataFrame, Index subclass, or ndarray
    key : Optional[Callable], key to be called on the values array
    levels : Optional[List], if values is a MultiIndex, list of levels to
    apply the key to.
    """
    from pandas.core.indexes.api import Index

    if not key:
        return values

    if isinstance(values, ABCMultiIndex):
        return _ensure_key_mapped_multiindex(values, key, level=levels)

    result = key(values.copy())
    if len(result) != len(values):
        raise ValueError(
            "User-provided `key` function must not change the shape of the array."
        )

    try:
        if isinstance(
            values, Index
        ):  # convert to a new Index subclass, not necessarily the same
            result = Index(result)
        else:
            # try to revert to original type otherwise
            type_of_values = type(values)
            #  error: Too many arguments for "ExtensionArray"
            result = type_of_values(result)  # type: ignore[call-arg]
    except TypeError:
        raise TypeError(
            f"User-provided `key` function returned an invalid type {type(result)} \
            which could not be converted to {type(values)}."
        )

    return result


def get_flattened_list(
    comp_ids: npt.NDArray[np.intp],
    ngroups: int,
    levels: Iterable[Index],
    labels: Iterable[np.ndarray],
) -> list[tuple]:
    """Map compressed group id -> key tuple."""
    comp_ids = comp_ids.astype(np.int64, copy=False)
    arrays: DefaultDict[int, list[int]] = defaultdict(list)
    for labs, level in zip(labels, levels):
        table = hashtable.Int64HashTable(ngroups)
        table.map_keys_to_values(comp_ids, labs.astype(np.int64, copy=False))
        for i in range(ngroups):
            arrays[i].append(level[table.get_item(i)])
    return [tuple(array) for array in arrays.values()]


def get_indexer_dict(
    label_list: list[np.ndarray], keys: list[Index]
) -> dict[Hashable, npt.NDArray[np.intp]]:
    """
    Returns
    -------
    dict:
        Labels mapped to indexers.
    """
    shape = tuple(len(x) for x in keys)

    group_index = get_group_index(label_list, shape, sort=True, xnull=True)
    if np.all(group_index == -1):
        # Short-circuit, lib.indices_fast will return the same
        return {}
    ngroups = (
        ((group_index.size and group_index.max()) + 1)
        if is_int64_overflow_possible(shape)
        else np.prod(shape, dtype="i8")
    )

    sorter = get_group_index_sorter(group_index, ngroups)

    sorted_labels = [lab.take(sorter) for lab in label_list]
    group_index = group_index.take(sorter)

    return lib.indices_fast(sorter, group_index, keys, sorted_labels)


# ----------------------------------------------------------------------
# sorting levels...cleverly?


def get_group_index_sorter(
    group_index: npt.NDArray[np.intp], ngroups: int | None = None
) -> npt.NDArray[np.intp]:
    """
    algos.groupsort_indexer implements `counting sort` and it is at least
    O(ngroups), where
        ngroups = prod(shape)
        shape = map(len, keys)
    that is, linear in the number of combinations (cartesian product) of unique
    values of groupby keys. This can be huge when doing multi-key groupby.
    np.argsort(kind='mergesort') is O(count x log(count)) where count is the
    length of the data-frame;
    Both algorithms are `stable` sort and that is necessary for correctness of
    groupby operations. e.g. consider:
        df.groupby(key)[col].transform('first')

    Parameters
    ----------
    group_index : np.ndarray[np.intp]
        signed integer dtype
    ngroups : int or None, default None

    Returns
    -------
    np.ndarray[np.intp]
    """
    if ngroups is None:
        ngroups = 1 + group_index.max()
    count = len(group_index)
    alpha = 0.0  # taking complexities literally; there may be
    beta = 1.0  # some room for fine-tuning these parameters
    do_groupsort = count > 0 and ((alpha + beta * ngroups) < (count * np.log(count)))
    if do_groupsort:
        sorter, _ = algos.groupsort_indexer(
            ensure_platform_int(group_index),
            ngroups,
        )
        # sorter _should_ already be intp, but mypy is not yet able to verify
    else:
        sorter = group_index.argsort(kind="mergesort")
    return ensure_platform_int(sorter)


def compress_group_index(
    group_index: npt.NDArray[np.int64], sort: bool = True
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).
    """
    if len(group_index) and np.all(group_index[1:] >= group_index[:-1]):
        # GH 53806: fast path for sorted group_index
        unique_mask = np.concatenate(
            [group_index[:1] > -1, group_index[1:] != group_index[:-1]]
        )
        comp_ids = unique_mask.cumsum()
        comp_ids -= 1
        obs_group_ids = group_index[unique_mask]
    else:
        size_hint = len(group_index)
        table = hashtable.Int64HashTable(size_hint)

        group_index = ensure_int64(group_index)

        # note, group labels come out ascending (ie, 1,2,3 etc)
        comp_ids, obs_group_ids = table.get_labels_groupby(group_index)

        if sort and len(obs_group_ids) > 0:
            obs_group_ids, comp_ids = _reorder_by_uniques(obs_group_ids, comp_ids)

    return ensure_int64(comp_ids), ensure_int64(obs_group_ids)


def _reorder_by_uniques(
    uniques: npt.NDArray[np.int64], labels: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp]]:
    """
    Parameters
    ----------
    uniques : np.ndarray[np.int64]
    labels : np.ndarray[np.intp]

    Returns
    -------
    np.ndarray[np.int64]
    np.ndarray[np.intp]
    """
    # sorter is index where elements ought to go
    sorter = uniques.argsort()

    # reverse_indexer is where elements came from
    reverse_indexer = np.empty(len(sorter), dtype=np.intp)
    reverse_indexer.put(sorter, np.arange(len(sorter)))

    mask = labels < 0

    # move labels to right locations (ie, unsort ascending labels)
    labels = reverse_indexer.take(labels)
    np.putmask(labels, mask, -1)

    # sort observed ids
    uniques = uniques.take(sorter)

    return uniques, labels

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\clipboard\__init__.py ===
"""
Pyperclip

A cross-platform clipboard module for Python,
with copy & paste functions for plain text.
By Al Sweigart al@inventwithpython.com
Licence at LICENSES/PYPERCLIP_LICENSE

Usage:
  import pyperclip
  pyperclip.copy('The text to be copied to the clipboard.')
  spam = pyperclip.paste()

  if not pyperclip.is_available():
    print("Copy functionality unavailable!")

On Windows, no additional modules are needed.
On Mac, the pyobjc module is used, falling back to the pbcopy and pbpaste cli
    commands. (These commands should come with OS X.).
On Linux, install xclip, xsel, or wl-clipboard (for "wayland" sessions) via
package manager.
For example, in Debian:
    sudo apt-get install xclip
    sudo apt-get install xsel
    sudo apt-get install wl-clipboard

Otherwise on Linux, you will need the PyQt5 modules installed.

This module does not work with PyGObject yet.

Cygwin is currently not supported.

Security Note: This module runs programs with these names:
    - pbcopy
    - pbpaste
    - xclip
    - xsel
    - wl-copy/wl-paste
    - klipper
    - qdbus
A malicious user could rename or add programs with these names, tricking
Pyperclip into running them with whatever permissions the Python process has.

"""

__version__ = "1.8.2"


import contextlib
import ctypes
from ctypes import (
    c_size_t,
    c_wchar,
    c_wchar_p,
    get_errno,
    sizeof,
)
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings

from pandas.errors import (
    PyperclipException,
    PyperclipWindowsException,
)
from pandas.util._exceptions import find_stack_level

# `import PyQt4` sys.exit()s if DISPLAY is not in the environment.
# Thus, we need to detect the presence of $DISPLAY manually
# and not load PyQt4 if it is absent.
HAS_DISPLAY = os.getenv("DISPLAY")

EXCEPT_MSG = """
    Pyperclip could not find a copy/paste mechanism for your system.
    For more information, please visit
    https://pyperclip.readthedocs.io/en/latest/index.html#not-implemented-error
    """

ENCODING = "utf-8"


class PyperclipTimeoutException(PyperclipException):
    pass


def _stringifyText(text) -> str:
    acceptedTypes = (str, int, float, bool)
    if not isinstance(text, acceptedTypes):
        raise PyperclipException(
            f"only str, int, float, and bool values "
            f"can be copied to the clipboard, not {type(text).__name__}"
        )
    return str(text)


def init_osx_pbcopy_clipboard():
    def copy_osx_pbcopy(text):
        text = _stringifyText(text)  # Converts non-str values to str.
        with subprocess.Popen(
            ["pbcopy", "w"], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_osx_pbcopy():
        with subprocess.Popen(
            ["pbpaste", "r"], stdout=subprocess.PIPE, close_fds=True
        ) as p:
            stdout = p.communicate()[0]
        return stdout.decode(ENCODING)

    return copy_osx_pbcopy, paste_osx_pbcopy


def init_osx_pyobjc_clipboard():
    def copy_osx_pyobjc(text):
        """Copy string argument to clipboard"""
        text = _stringifyText(text)  # Converts non-str values to str.
        newStr = Foundation.NSString.stringWithString_(text).nsstring()
        newData = newStr.dataUsingEncoding_(Foundation.NSUTF8StringEncoding)
        board = AppKit.NSPasteboard.generalPasteboard()
        board.declareTypes_owner_([AppKit.NSStringPboardType], None)
        board.setData_forType_(newData, AppKit.NSStringPboardType)

    def paste_osx_pyobjc():
        """Returns contents of clipboard"""
        board = AppKit.NSPasteboard.generalPasteboard()
        content = board.stringForType_(AppKit.NSStringPboardType)
        return content

    return copy_osx_pyobjc, paste_osx_pyobjc


def init_qt_clipboard():
    global QApplication
    # $DISPLAY should exist

    # Try to import from qtpy, but if that fails try PyQt5 then PyQt4
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            from PyQt4.QtGui import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    def copy_qt(text):
        text = _stringifyText(text)  # Converts non-str values to str.
        cb = app.clipboard()
        cb.setText(text)

    def paste_qt() -> str:
        cb = app.clipboard()
        return str(cb.text())

    return copy_qt, paste_qt


def init_xclip_clipboard():
    DEFAULT_SELECTION = "c"
    PRIMARY_SELECTION = "p"

    def copy_xclip(text, primary=False):
        text = _stringifyText(text)  # Converts non-str values to str.
        selection = DEFAULT_SELECTION
        if primary:
            selection = PRIMARY_SELECTION
        with subprocess.Popen(
            ["xclip", "-selection", selection], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_xclip(primary=False):
        selection = DEFAULT_SELECTION
        if primary:
            selection = PRIMARY_SELECTION
        with subprocess.Popen(
            ["xclip", "-selection", selection, "-o"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]
        # Intentionally ignore extraneous output on stderr when clipboard is empty
        return stdout.decode(ENCODING)

    return copy_xclip, paste_xclip


def init_xsel_clipboard():
    DEFAULT_SELECTION = "-b"
    PRIMARY_SELECTION = "-p"

    def copy_xsel(text, primary=False):
        text = _stringifyText(text)  # Converts non-str values to str.
        selection_flag = DEFAULT_SELECTION
        if primary:
            selection_flag = PRIMARY_SELECTION
        with subprocess.Popen(
            ["xsel", selection_flag, "-i"], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_xsel(primary=False):
        selection_flag = DEFAULT_SELECTION
        if primary:
            selection_flag = PRIMARY_SELECTION
        with subprocess.Popen(
            ["xsel", selection_flag, "-o"], stdout=subprocess.PIPE, close_fds=True
        ) as p:
            stdout = p.communicate()[0]
        return stdout.decode(ENCODING)

    return copy_xsel, paste_xsel


def init_wl_clipboard():
    PRIMARY_SELECTION = "-p"

    def copy_wl(text, primary=False):
        text = _stringifyText(text)  # Converts non-str values to str.
        args = ["wl-copy"]
        if primary:
            args.append(PRIMARY_SELECTION)
        if not text:
            args.append("--clear")
            subprocess.check_call(args, close_fds=True)
        else:
            p = subprocess.Popen(args, stdin=subprocess.PIPE, close_fds=True)
            p.communicate(input=text.encode(ENCODING))

    def paste_wl(primary=False):
        args = ["wl-paste", "-n"]
        if primary:
            args.append(PRIMARY_SELECTION)
        p = subprocess.Popen(args, stdout=subprocess.PIPE, close_fds=True)
        stdout, _stderr = p.communicate()
        return stdout.decode(ENCODING)

    return copy_wl, paste_wl


def init_klipper_clipboard():
    def copy_klipper(text):
        text = _stringifyText(text)  # Converts non-str values to str.
        with subprocess.Popen(
            [
                "qdbus",
                "org.kde.klipper",
                "/klipper",
                "setClipboardContents",
                text.encode(ENCODING),
            ],
            stdin=subprocess.PIPE,
            close_fds=True,
        ) as p:
            p.communicate(input=None)

    def paste_klipper():
        with subprocess.Popen(
            ["qdbus", "org.kde.klipper", "/klipper", "getClipboardContents"],
            stdout=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]

        # Workaround for https://bugs.kde.org/show_bug.cgi?id=342874
        # TODO: https://github.com/asweigart/pyperclip/issues/43
        clipboardContents = stdout.decode(ENCODING)
        # even if blank, Klipper will append a newline at the end
        assert len(clipboardContents) > 0
        # make sure that newline is there
        assert clipboardContents.endswith("\n")
        if clipboardContents.endswith("\n"):
            clipboardContents = clipboardContents[:-1]
        return clipboardContents

    return copy_klipper, paste_klipper


def init_dev_clipboard_clipboard():
    def copy_dev_clipboard(text):
        text = _stringifyText(text)  # Converts non-str values to str.
        if text == "":
            warnings.warn(
                "Pyperclip cannot copy a blank string to the clipboard on Cygwin. "
                "This is effectively a no-op.",
                stacklevel=find_stack_level(),
            )
        if "\r" in text:
            warnings.warn(
                "Pyperclip cannot handle \\r characters on Cygwin.",
                stacklevel=find_stack_level(),
            )

        with open("/dev/clipboard", "w", encoding="utf-8") as fd:
            fd.write(text)

    def paste_dev_clipboard() -> str:
        with open("/dev/clipboard", encoding="utf-8") as fd:
            content = fd.read()
        return content

    return copy_dev_clipboard, paste_dev_clipboard


def init_no_clipboard():
    class ClipboardUnavailable:
        def __call__(self, *args, **kwargs):
            raise PyperclipException(EXCEPT_MSG)

        def __bool__(self) -> bool:
            return False

    return ClipboardUnavailable(), ClipboardUnavailable()


# Windows-related clipboard functions:
class CheckedCall:
    def __init__(self, f) -> None:
        super().__setattr__("f", f)

    def __call__(self, *args):
        ret = self.f(*args)
        if not ret and get_errno():
            raise PyperclipWindowsException("Error calling " + self.f.__name__)
        return ret

    def __setattr__(self, key, value):
        setattr(self.f, key, value)


def init_windows_clipboard():
    global HGLOBAL, LPVOID, DWORD, LPCSTR, INT
    global HWND, HINSTANCE, HMENU, BOOL, UINT, HANDLE
    from ctypes.wintypes import (
        BOOL,
        DWORD,
        HANDLE,
        HGLOBAL,
        HINSTANCE,
        HMENU,
        HWND,
        INT,
        LPCSTR,
        LPVOID,
        UINT,
    )

    windll = ctypes.windll
    msvcrt = ctypes.CDLL("msvcrt")

    safeCreateWindowExA = CheckedCall(windll.user32.CreateWindowExA)
    safeCreateWindowExA.argtypes = [
        DWORD,
        LPCSTR,
        LPCSTR,
        DWORD,
        INT,
        INT,
        INT,
        INT,
        HWND,
        HMENU,
        HINSTANCE,
        LPVOID,
    ]
    safeCreateWindowExA.restype = HWND

    safeDestroyWindow = CheckedCall(windll.user32.DestroyWindow)
    safeDestroyWindow.argtypes = [HWND]
    safeDestroyWindow.restype = BOOL

    OpenClipboard = windll.user32.OpenClipboard
    OpenClipboard.argtypes = [HWND]
    OpenClipboard.restype = BOOL

    safeCloseClipboard = CheckedCall(windll.user32.CloseClipboard)
    safeCloseClipboard.argtypes = []
    safeCloseClipboard.restype = BOOL

    safeEmptyClipboard = CheckedCall(windll.user32.EmptyClipboard)
    safeEmptyClipboard.argtypes = []
    safeEmptyClipboard.restype = BOOL

    safeGetClipboardData = CheckedCall(windll.user32.GetClipboardData)
    safeGetClipboardData.argtypes = [UINT]
    safeGetClipboardData.restype = HANDLE

    safeSetClipboardData = CheckedCall(windll.user32.SetClipboardData)
    safeSetClipboardData.argtypes = [UINT, HANDLE]
    safeSetClipboardData.restype = HANDLE

    safeGlobalAlloc = CheckedCall(windll.kernel32.GlobalAlloc)
    safeGlobalAlloc.argtypes = [UINT, c_size_t]
    safeGlobalAlloc.restype = HGLOBAL

    safeGlobalLock = CheckedCall(windll.kernel32.GlobalLock)
    safeGlobalLock.argtypes = [HGLOBAL]
    safeGlobalLock.restype = LPVOID

    safeGlobalUnlock = CheckedCall(windll.kernel32.GlobalUnlock)
    safeGlobalUnlock.argtypes = [HGLOBAL]
    safeGlobalUnlock.restype = BOOL

    wcslen = CheckedCall(msvcrt.wcslen)
    wcslen.argtypes = [c_wchar_p]
    wcslen.restype = UINT

    GMEM_MOVEABLE = 0x0002
    CF_UNICODETEXT = 13

    @contextlib.contextmanager
    def window():
        """
        Context that provides a valid Windows hwnd.
        """
        # we really just need the hwnd, so setting "STATIC"
        # as predefined lpClass is just fine.
        hwnd = safeCreateWindowExA(
            0, b"STATIC", None, 0, 0, 0, 0, 0, None, None, None, None
        )
        try:
            yield hwnd
        finally:
            safeDestroyWindow(hwnd)

    @contextlib.contextmanager
    def clipboard(hwnd):
        """
        Context manager that opens the clipboard and prevents
        other applications from modifying the clipboard content.
        """
        # We may not get the clipboard handle immediately because
        # some other application is accessing it (?)
        # We try for at least 500ms to get the clipboard.
        t = time.time() + 0.5
        success = False
        while time.time() < t:
            success = OpenClipboard(hwnd)
            if success:
                break
            time.sleep(0.01)
        if not success:
            raise PyperclipWindowsException("Error calling OpenClipboard")

        try:
            yield
        finally:
            safeCloseClipboard()

    def copy_windows(text):
        # This function is heavily based on
        # http://msdn.com/ms649016#_win32_Copying_Information_to_the_Clipboard

        text = _stringifyText(text)  # Converts non-str values to str.

        with window() as hwnd:
            # http://msdn.com/ms649048
            # If an application calls OpenClipboard with hwnd set to NULL,
            # EmptyClipboard sets the clipboard owner to NULL;
            # this causes SetClipboardData to fail.
            # => We need a valid hwnd to copy something.
            with clipboard(hwnd):
                safeEmptyClipboard()

                if text:
                    # http://msdn.com/ms649051
                    # If the hMem parameter identifies a memory object,
                    # the object must have been allocated using the
                    # function with the GMEM_MOVEABLE flag.
                    count = wcslen(text) + 1
                    handle = safeGlobalAlloc(GMEM_MOVEABLE, count * sizeof(c_wchar))
                    locked_handle = safeGlobalLock(handle)

                    ctypes.memmove(
                        c_wchar_p(locked_handle),
                        c_wchar_p(text),
                        count * sizeof(c_wchar),
                    )

                    safeGlobalUnlock(handle)
                    safeSetClipboardData(CF_UNICODETEXT, handle)

    def paste_windows():
        with clipboard(None):
            handle = safeGetClipboardData(CF_UNICODETEXT)
            if not handle:
                # GetClipboardData may return NULL with errno == NO_ERROR
                # if the clipboard is empty.
                # (Also, it may return a handle to an empty buffer,
                # but technically that's not empty)
                return ""
            return c_wchar_p(handle).value

    return copy_windows, paste_windows


def init_wsl_clipboard():
    def copy_wsl(text):
        text = _stringifyText(text)  # Converts non-str values to str.
        with subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE, close_fds=True) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_wsl():
        with subprocess.Popen(
            ["powershell.exe", "-command", "Get-Clipboard"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]
        # WSL appends "\r\n" to the contents.
        return stdout[:-2].decode(ENCODING)

    return copy_wsl, paste_wsl


# Automatic detection of clipboard mechanisms
# and importing is done in determine_clipboard():
def determine_clipboard():
    """
    Determine the OS/platform and set the copy() and paste() functions
    accordingly.
    """
    global Foundation, AppKit, qtpy, PyQt4, PyQt5

    # Setup for the CYGWIN platform:
    if (
        "cygwin" in platform.system().lower()
    ):  # Cygwin has a variety of values returned by platform.system(),
        # such as 'CYGWIN_NT-6.1'
        # FIXME(pyperclip#55): pyperclip currently does not support Cygwin,
        # see https://github.com/asweigart/pyperclip/issues/55
        if os.path.exists("/dev/clipboard"):
            warnings.warn(
                "Pyperclip's support for Cygwin is not perfect, "
                "see https://github.com/asweigart/pyperclip/issues/55",
                stacklevel=find_stack_level(),
            )
            return init_dev_clipboard_clipboard()

    # Setup for the WINDOWS platform:
    elif os.name == "nt" or platform.system() == "Windows":
        return init_windows_clipboard()

    if platform.system() == "Linux":
        if _executable_exists("wslconfig.exe"):
            return init_wsl_clipboard()

    # Setup for the macOS platform:
    if os.name == "mac" or platform.system() == "Darwin":
        try:
            import AppKit
            import Foundation  # check if pyobjc is installed
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()

    # Setup for the LINUX platform:
    if HAS_DISPLAY:
        if os.environ.get("WAYLAND_DISPLAY") and _executable_exists("wl-copy"):
            return init_wl_clipboard()
        if _executable_exists("xsel"):
            return init_xsel_clipboard()
        if _executable_exists("xclip"):
            return init_xclip_clipboard()
        if _executable_exists("klipper") and _executable_exists("qdbus"):
            return init_klipper_clipboard()

        try:
            # qtpy is a small abstraction layer that lets you write applications
            # using a single api call to either PyQt or PySide.
            # https://pypi.python.org/project/QtPy
            import qtpy  # check if qtpy is installed
        except ImportError:
            # If qtpy isn't installed, fall back on importing PyQt4.
            try:
                import PyQt5  # check if PyQt5 is installed
            except ImportError:
                try:
                    import PyQt4  # check if PyQt4 is installed
                except ImportError:
                    pass  # We want to fail fast for all non-ImportError exceptions.
                else:
                    return init_qt_clipboard()
            else:
                return init_qt_clipboard()
        else:
            return init_qt_clipboard()

    return init_no_clipboard()


def set_clipboard(clipboard):
    """
    Explicitly sets the clipboard mechanism. The "clipboard mechanism" is how
    the copy() and paste() functions interact with the operating system to
    implement the copy/paste feature. The clipboard parameter must be one of:
        - pbcopy
        - pyobjc (default on macOS)
        - qt
        - xclip
        - xsel
        - klipper
        - windows (default on Windows)
        - no (this is what is set when no clipboard mechanism can be found)
    """
    global copy, paste

    clipboard_types = {
        "pbcopy": init_osx_pbcopy_clipboard,
        "pyobjc": init_osx_pyobjc_clipboard,
        "qt": init_qt_clipboard,  # TODO - split this into 'qtpy', 'pyqt4', and 'pyqt5'
        "xclip": init_xclip_clipboard,
        "xsel": init_xsel_clipboard,
        "wl-clipboard": init_wl_clipboard,
        "klipper": init_klipper_clipboard,
        "windows": init_windows_clipboard,
        "no": init_no_clipboard,
    }

    if clipboard not in clipboard_types:
        allowed_clipboard_types = [repr(_) for _ in clipboard_types]
        raise ValueError(
            f"Argument must be one of {', '.join(allowed_clipboard_types)}"
        )

    # Sets pyperclip's copy() and paste() functions:
    copy, paste = clipboard_types[clipboard]()


def lazy_load_stub_copy(text):
    """
    A stub function for copy(), which will load the real copy() function when
    called so that the real copy() function is used for later calls.

    This allows users to import pyperclip without having determine_clipboard()
    automatically run, which will automatically select a clipboard mechanism.
    This could be a problem if it selects, say, the memory-heavy PyQt4 module
    but the user was just going to immediately call set_clipboard() to use a
    different clipboard mechanism.

    The lazy loading this stub function implements gives the user a chance to
    call set_clipboard() to pick another clipboard mechanism. Or, if the user
    simply calls copy() or paste() without calling set_clipboard() first,
    will fall back on whatever clipboard mechanism that determine_clipboard()
    automatically chooses.
    """
    global copy, paste
    copy, paste = determine_clipboard()
    return copy(text)


def lazy_load_stub_paste():
    """
    A stub function for paste(), which will load the real paste() function when
    called so that the real paste() function is used for later calls.

    This allows users to import pyperclip without having determine_clipboard()
    automatically run, which will automatically select a clipboard mechanism.
    This could be a problem if it selects, say, the memory-heavy PyQt4 module
    but the user was just going to immediately call set_clipboard() to use a
    different clipboard mechanism.

    The lazy loading this stub function implements gives the user a chance to
    call set_clipboard() to pick another clipboard mechanism. Or, if the user
    simply calls copy() or paste() without calling set_clipboard() first,
    will fall back on whatever clipboard mechanism that determine_clipboard()
    automatically chooses.
    """
    global copy, paste
    copy, paste = determine_clipboard()
    return paste()


def is_available() -> bool:
    return copy != lazy_load_stub_copy and paste != lazy_load_stub_paste


# Initially, copy() and paste() are set to lazy loading wrappers which will
# set `copy` and `paste` to real functions the first time they're used, unless
# set_clipboard() or determine_clipboard() is called first.
copy, paste = lazy_load_stub_copy, lazy_load_stub_paste


def waitForPaste(timeout=None):
    """This function call blocks until a non-empty text string exists on the
    clipboard. It returns this text.

    This function raises PyperclipTimeoutException if timeout was set to
    a number of seconds that has elapsed without non-empty text being put on
    the clipboard."""
    startTime = time.time()
    while True:
        clipboardText = paste()
        if clipboardText != "":
            return clipboardText
        time.sleep(0.01)

        if timeout is not None and time.time() > startTime + timeout:
            raise PyperclipTimeoutException(
                "waitForPaste() timed out after " + str(timeout) + " seconds."
            )


def waitForNewPaste(timeout=None):
    """This function call blocks until a new text string exists on the
    clipboard that is different from the text that was there when the function
    was first called. It returns this text.

    This function raises PyperclipTimeoutException if timeout was set to
    a number of seconds that has elapsed without non-empty text being put on
    the clipboard."""
    startTime = time.time()
    originalText = paste()
    while True:
        currentText = paste()
        if currentText != originalText:
            return currentText
        time.sleep(0.01)

        if timeout is not None and time.time() > startTime + timeout:
            raise PyperclipTimeoutException(
                "waitForNewPaste() timed out after " + str(timeout) + " seconds."
            )


__all__ = [
    "copy",
    "paste",
    "waitForPaste",
    "waitForNewPaste",
    "set_clipboard",
    "determine_clipboard",
]

# pandas aliases
clipboard_get = paste
clipboard_set = copy

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\c.py ===
"""
C code printer

The C89CodePrinter & C99CodePrinter converts single SymPy expressions into
single C expressions, using the functions defined in math.h where possible.

A complete code generator, which uses ccode extensively, can be found in
sympy.utilities.codegen. The codegen module can be used to generate complete
source code files that are compilable without further modifications.


"""

from __future__ import annotations
from typing import Any

from functools import wraps
from itertools import chain

from sympy.core import S
from sympy.core.numbers import equal_valued, Float
from sympy.codegen.ast import (
    Assignment, Pointer, Variable, Declaration, Type,
    real, complex_, integer, bool_, float32, float64, float80,
    complex64, complex128, intc, value_const, pointer_const,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64, untyped,
    none
)
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401

# dictionary mapping SymPy function to (argument_conditions, C_function).
# Used in C89CodePrinter._print_Function(self)
known_functions_C89 = {
    "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "log10": "log10",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "floor": "floor",
    "ceiling": "ceil",
    "sqrt": "sqrt", # To enable automatic rewrites
}

known_functions_C99 = dict(known_functions_C89, **{
    'exp2': 'exp2',
    'expm1': 'expm1',
    'log2': 'log2',
    'log1p': 'log1p',
    'Cbrt': 'cbrt',
    'hypot': 'hypot',
    'fma': 'fma',
    'loggamma': 'lgamma',
    'erfc': 'erfc',
    'Max': 'fmax',
    'Min': 'fmin',
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "erf": "erf",
    "gamma": "tgamma",
})

# These are the core reserved words in the C language. Taken from:
# https://en.cppreference.com/w/c/keyword

reserved_words = [
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int',
    'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
    'struct', 'entry',  # never standardized, we'll leave it here anyway
    'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'
]

reserved_words_c99 = ['inline', 'restrict']

def get_math_macros():
    """ Returns a dictionary with math-related macros from math.h/cmath

    Note that these macros are not strictly required by the C/C++-standard.
    For MSVC they are enabled by defining "_USE_MATH_DEFINES" (preferably
    via a compilation flag).

    Returns
    =======

    Dictionary mapping SymPy expressions to strings (macro names)

    """
    from sympy.codegen.cfunctions import log2, Sqrt
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.miscellaneous import sqrt

    return {
        S.Exp1: 'M_E',
        log2(S.Exp1): 'M_LOG2E',
        1/log(2): 'M_LOG2E',
        log(2): 'M_LN2',
        log(10): 'M_LN10',
        S.Pi: 'M_PI',
        S.Pi/2: 'M_PI_2',
        S.Pi/4: 'M_PI_4',
        1/S.Pi: 'M_1_PI',
        2/S.Pi: 'M_2_PI',
        2/sqrt(S.Pi): 'M_2_SQRTPI',
        2/Sqrt(S.Pi): 'M_2_SQRTPI',
        sqrt(2): 'M_SQRT2',
        Sqrt(2): 'M_SQRT2',
        1/sqrt(2): 'M_SQRT1_2',
        1/Sqrt(2): 'M_SQRT1_2'
    }


def _as_macro_if_defined(meth):
    """ Decorator for printer methods

    When a Printer's method is decorated using this decorator the expressions printed
    will first be looked for in the attribute ``math_macros``, and if present it will
    print the macro name in ``math_macros`` followed by a type suffix for the type
    ``real``. e.g. printing ``sympy.pi`` would print ``M_PIl`` if real is mapped to float80.

    """
    @wraps(meth)
    def _meth_wrapper(self, expr, **kwargs):
        if expr in self.math_macros:
            return '%s%s' % (self.math_macros[expr], self._get_math_macro_suffix(real))
        else:
            return meth(self, expr, **kwargs)

    return _meth_wrapper


class C89CodePrinter(CodePrinter):
    """A printer to convert Python expressions to strings of C code"""
    printmethod = "_ccode"
    language = "C"
    standard = "C89"
    reserved_words = set(reserved_words)

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
        'error_on_reserved': False,
    })

    type_aliases = {
        real: float64,
        complex_: complex128,
        integer: intc
    }

    type_mappings: dict[Type, Any] = {
        real: 'double',
        intc: 'int',
        float32: 'float',
        float64: 'double',
        integer: 'int',
        bool_: 'bool',
        int8: 'int8_t',
        int16: 'int16_t',
        int32: 'int32_t',
        int64: 'int64_t',
        uint8: 'int8_t',
        uint16: 'int16_t',
        uint32: 'int32_t',
        uint64: 'int64_t',
    }

    type_headers = {
        bool_: {'stdbool.h'},
        int8: {'stdint.h'},
        int16: {'stdint.h'},
        int32: {'stdint.h'},
        int64: {'stdint.h'},
        uint8: {'stdint.h'},
        uint16: {'stdint.h'},
        uint32: {'stdint.h'},
        uint64: {'stdint.h'},
    }

    # Macros needed to be defined when using a Type
    type_macros: dict[Type, tuple[str, ...]] = {}

    type_func_suffixes = {
        float32: 'f',
        float64: '',
        float80: 'l'
    }

    type_literal_suffixes = {
        float32: 'F',
        float64: '',
        float80: 'L'
    }

    type_math_macro_suffixes = {
        float80: 'l'
    }

    math_macros = None

    _ns = ''  # namespace, C++ uses 'std::'
    # known_functions-dict to copy
    _kf: dict[str, Any] = known_functions_C89

    def __init__(self, settings=None):
        settings = settings or {}
        if self.math_macros is None:
            self.math_macros = settings.pop('math_macros', get_math_macros())
        self.type_aliases = dict(chain(self.type_aliases.items(),
                                       settings.pop('type_aliases', {}).items()))
        self.type_mappings = dict(chain(self.type_mappings.items(),
                                        settings.pop('type_mappings', {}).items()))
        self.type_headers = dict(chain(self.type_headers.items(),
                                       settings.pop('type_headers', {}).items()))
        self.type_macros = dict(chain(self.type_macros.items(),
                                       settings.pop('type_macros', {}).items()))
        self.type_func_suffixes = dict(chain(self.type_func_suffixes.items(),
                                        settings.pop('type_func_suffixes', {}).items()))
        self.type_literal_suffixes = dict(chain(self.type_literal_suffixes.items(),
                                        settings.pop('type_literal_suffixes', {}).items()))
        self.type_math_macro_suffixes = dict(chain(self.type_math_macro_suffixes.items(),
                                        settings.pop('type_math_macro_suffixes', {}).items()))
        super().__init__(settings)
        self.known_functions = dict(self._kf, **settings.get('user_functions', {}))
        self._dereference = set(settings.get('dereference', []))
        self.headers = set()
        self.libraries = set()
        self.macros = set()

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        """ Get code string as a statement - i.e. ending with a semicolon. """
        return codestring if codestring.endswith(';') else codestring + ';'

    def _get_comment(self, text):
        return "/* {} */".format(text)

    def _declare_number_const(self, name, value):
        type_ = self.type_aliases[real]
        var = Variable(name, type=type_, value=value.evalf(type_.decimal_dig), attrs={value_const})
        decl = Declaration(var)
        return self._get_statement(self._print(decl))

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    @_as_macro_if_defined
    def _print_Mul(self, expr, **kwargs):
        return super()._print_Mul(expr, **kwargs)

    @_as_macro_if_defined
    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)
        if equal_valued(expr.exp, -1):
            return '%s/%s' % (self._print_Float(Float(1.0)), self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        elif expr.exp == S.One/3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base),
                                   self._print(expr.exp))

    def _print_Mod(self, expr):
        num, den = expr.args
        if num.is_integer and den.is_integer:
            PREC = precedence(expr)
            snum, sden = [self.parenthesize(arg, PREC) for arg in expr.args]
            # % is remainder (same sign as numerator), not modulo (same sign as
            # denominator), in C. Hence, % only works as modulo if both numbers
            # have the same sign
            if (num.is_nonnegative and den.is_nonnegative or
                num.is_nonpositive and den.is_nonpositive):
                return f"{snum} % {sden}"
            return f"(({snum} % {sden}) + {sden}) % {sden}"
        # Not guaranteed integer
        return self._print_math_func(expr, known='fmod')

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        suffix = self._get_literal_suffix(real)
        return '%d.0%s/%d.0%s' % (p, suffix, q, suffix)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        offset = getattr(expr.base, 'offset', S.Zero)
        strides = getattr(expr.base, 'strides', None)
        indices = expr.indices

        if strides is None or isinstance(strides, str):
            dims = expr.shape
            shift = S.One
            temp = ()
            if strides == 'C' or strides is None:
                traversal = reversed(range(expr.rank))
                indices = indices[::-1]
            elif strides == 'F':
                traversal = range(expr.rank)

            for i in traversal:
                temp += (shift,)
                shift *= dims[i]
            strides = temp
        flat_index = sum(x[0]*x[1] for x in zip(indices, strides)) + offset
        return "%s[%s]" % (self._print(expr.base.label),
                           self._print(flat_index))

    @_as_macro_if_defined
    def _print_NumberSymbol(self, expr):
        return super()._print_NumberSymbol(expr)

    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                else:
                    lines.append("else if (%s) {" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            return "\n".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c),
                                               self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixElement(self, expr):
        return "{}[{}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
            strict=True), expr.j + expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if expr in self._settings['dereference']:
            return '(*{})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return ('for ({target} = {start}; {target} < {stop}; {target} += '
                '{step}) {{\n{body}\n}}').format(target=target, start=start,
                stop=stop, step=step, body=body)

    def _print_sign(self, func):
        return '((({0}) > 0) - (({0}) < 0))'.format(self._print(func.args[0]))

    def _print_Max(self, expr):
        if "Max" in self.known_functions:
            return self._print_Function(expr)
        def inner_print_max(args): # The more natural abstraction of creating
            if len(args) == 1:     # and printing smaller Max objects is slow
                return self._print(args[0]) # when there are many arguments.
            half = len(args) // 2
            return "((%(a)s > %(b)s) ? %(a)s : %(b)s)" % {
                'a': inner_print_max(args[:half]),
                'b': inner_print_max(args[half:])
            }
        return inner_print_max(expr.args)

    def _print_Min(self, expr):
        if "Min" in self.known_functions:
            return self._print_Function(expr)
        def inner_print_min(args): # The more natural abstraction of creating
            if len(args) == 1:     # and printing smaller Min objects is slow
                return self._print(args[0]) # when there are many arguments.
            half = len(args) // 2
            return "((%(a)s < %(b)s) ? %(a)s : %(b)s)" % {
                'a': inner_print_min(args[:half]),
                'b': inner_print_min(args[half:])
            }
        return inner_print_min(expr.args)

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [line.lstrip(' \t') for line in code]

        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token))) for line in code]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty

    def _get_func_suffix(self, type_):
        return self.type_func_suffixes[self.type_aliases.get(type_, type_)]

    def _get_literal_suffix(self, type_):
        return self.type_literal_suffixes[self.type_aliases.get(type_, type_)]

    def _get_math_macro_suffix(self, type_):
        alias = self.type_aliases.get(type_, type_)
        dflt = self.type_math_macro_suffixes.get(alias, '')
        return self.type_math_macro_suffixes.get(type_, dflt)

    def _print_Tuple(self, expr):
        return '{'+', '.join(self._print(e) for e in expr)+'}'

    _print_List = _print_Tuple

    def _print_Type(self, type_):
        self.headers.update(self.type_headers.get(type_, set()))
        self.macros.update(self.type_macros.get(type_, set()))
        return self._print(self.type_mappings.get(type_, type_.name))

    def _print_Declaration(self, decl):
        from sympy.codegen.cnodes import restrict
        var = decl.variable
        val = var.value
        if var.type == untyped:
            raise ValueError("C does not support untyped variables")

        if isinstance(var, Pointer):
            result = '{vc}{t} *{pc} {r}{s}'.format(
                vc='const ' if value_const in var.attrs else '',
                t=self._print(var.type),
                pc=' const' if pointer_const in var.attrs else '',
                r='restrict ' if restrict in var.attrs else '',
                s=self._print(var.symbol)
            )
        elif isinstance(var, Variable):
            result = '{vc}{t} {s}'.format(
                vc='const ' if value_const in var.attrs else '',
                t=self._print(var.type),
                s=self._print(var.symbol)
            )
        else:
            raise NotImplementedError("Unknown type of var: %s" % type(var))
        if val != None: # Must be "!= None", cannot be "is not None"
            result += ' = %s' % self._print(val)
        return result

    def _print_Float(self, flt):
        type_ = self.type_aliases.get(real, real)
        self.macros.update(self.type_macros.get(type_, set()))
        suffix = self._get_literal_suffix(type_)
        num = str(flt.evalf(type_.decimal_dig))
        if 'e' not in num and '.' not in num:
            num += '.0'
        num_parts = num.split('e')
        num_parts[0] = num_parts[0].rstrip('0')
        if num_parts[0].endswith('.'):
            num_parts[0] += '0'
        return 'e'.join(num_parts) + suffix

    @requires(headers={'stdbool.h'})
    def _print_BooleanTrue(self, expr):
        return 'true'

    @requires(headers={'stdbool.h'})
    def _print_BooleanFalse(self, expr):
        return 'false'

    def _print_Element(self, elem):
        if elem.strides == None: # Must be "== None", cannot be "is None"
            if elem.offset != None: # Must be "!= None", cannot be "is not None"
                raise ValueError("Expected strides when offset is given")
            idxs = ']['.join((self._print(arg) for arg in elem.indices))
        else:
            global_idx = sum(i*s for i, s in zip(elem.indices, elem.strides))
            if elem.offset != None: # Must be "!= None", cannot be "is not None"
                global_idx += elem.offset
            idxs = self._print(global_idx)

        return "{symb}[{idxs}]".format(
            symb=self._print(elem.symbol),
            idxs=idxs
        )

    def _print_CodeBlock(self, expr):
        """ Elements of code blocks printed as statements. """
        return '\n'.join([self._get_statement(self._print(i)) for i in expr.args])

    def _print_While(self, expr):
        return 'while ({condition}) {{\n{body}\n}}'.format(**expr.kwargs(
            apply=lambda arg: self._print(arg)))

    def _print_Scope(self, expr):
        return '{\n%s\n}' % self._print_CodeBlock(expr.body)

    @requires(headers={'stdio.h'})
    def _print_Print(self, expr):
        if expr.file == none:
            template = 'printf({fmt}, {pargs})'
        else:
            template = 'fprintf(%(out)s, {fmt}, {pargs})' % {
                'out': self._print(expr.file)
            }
        return template.format(
            fmt="%s\n" if expr.format_string == none else self._print(expr.format_string),
            pargs=', '.join((self._print(arg) for arg in expr.print_args))
        )

    def _print_Stream(self, strm):
        return strm.name

    def _print_FunctionPrototype(self, expr):
        pars = ', '.join((self._print(Declaration(arg)) for arg in expr.parameters))
        return "%s %s(%s)" % (
            tuple((self._print(arg) for arg in (expr.return_type, expr.name))) + (pars,)
        )

    def _print_FunctionDefinition(self, expr):
        return "%s%s" % (self._print_FunctionPrototype(expr),
                         self._print_Scope(expr))

    def _print_Return(self, expr):
        arg, = expr.args
        return 'return %s' % self._print(arg)

    def _print_CommaOperator(self, expr):
        return '(%s)' % ', '.join((self._print(arg) for arg in expr.args))

    def _print_Label(self, expr):
        if expr.body == none:
            return '%s:' % str(expr.name)
        if len(expr.body.args) == 1:
            return '%s:\n%s' % (str(expr.name), self._print_CodeBlock(expr.body))
        return '%s:\n{\n%s\n}' % (str(expr.name), self._print_CodeBlock(expr.body))

    def _print_goto(self, expr):
        return 'goto %s' % expr.label.name

    def _print_PreIncrement(self, expr):
        arg, = expr.args
        return '++(%s)' % self._print(arg)

    def _print_PostIncrement(self, expr):
        arg, = expr.args
        return '(%s)++' % self._print(arg)

    def _print_PreDecrement(self, expr):
        arg, = expr.args
        return '--(%s)' % self._print(arg)

    def _print_PostDecrement(self, expr):
        arg, = expr.args
        return '(%s)--' % self._print(arg)

    def _print_struct(self, expr):
        return "%(keyword)s %(name)s {\n%(lines)s}" % {
            "keyword": expr.__class__.__name__, "name": expr.name, "lines": ';\n'.join(
                [self._print(decl) for decl in expr.declarations] + [''])
        }

    def _print_BreakToken(self, _):
        return 'break'

    def _print_ContinueToken(self, _):
        return 'continue'

    _print_union = _print_struct

class C99CodePrinter(C89CodePrinter):
    standard = 'C99'
    reserved_words = set(reserved_words + reserved_words_c99)
    type_mappings=dict(chain(C89CodePrinter.type_mappings.items(), {
        complex64: 'float complex',
        complex128: 'double complex',
    }.items()))
    type_headers = dict(chain(C89CodePrinter.type_headers.items(), {
        complex64: {'complex.h'},
        complex128: {'complex.h'}
    }.items()))

    # known_functions-dict to copy
    _kf: dict[str, Any] = known_functions_C99

    # functions with versions with 'f' and 'l' suffixes:
    _prec_funcs = ('fabs fmod remainder remquo fma fmax fmin fdim nan exp exp2'
                   ' expm1 log log10 log2 log1p pow sqrt cbrt hypot sin cos tan'
                   ' asin acos atan atan2 sinh cosh tanh asinh acosh atanh erf'
                   ' erfc tgamma lgamma ceil floor trunc round nearbyint rint'
                   ' frexp ldexp modf scalbn ilogb logb nextafter copysign').split()

    def _print_Infinity(self, expr):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr):
        return '-INFINITY'

    def _print_NaN(self, expr):
        return 'NAN'

    # tgamma was already covered by 'known_functions' dict

    @requires(headers={'math.h'}, libraries={'m'})
    @_as_macro_if_defined
    def _print_math_func(self, expr, nest=False, known=None):
        if known is None:
            known = self.known_functions[expr.__class__.__name__]
        if not isinstance(known, str):
            for cb, name in known:
                if cb(*expr.args):
                    known = name
                    break
            else:
                raise ValueError("No matching printer")
        try:
            return known(self, *expr.args)
        except TypeError:
            suffix = self._get_func_suffix(real) if self._ns + known in self._prec_funcs else ''

        if nest:
            args = self._print(expr.args[0])
            if len(expr.args) > 1:
                paren_pile = ''
                for curr_arg in expr.args[1:-1]:
                    paren_pile += ')'
                    args += ', {ns}{name}{suffix}({next}'.format(
                        ns=self._ns,
                        name=known,
                        suffix=suffix,
                        next = self._print(curr_arg)
                    )
                args += ', %s%s' % (
                    self._print(expr.func(expr.args[-1])),
                    paren_pile
                )
        else:
            args = ', '.join((self._print(arg) for arg in expr.args))
        return '{ns}{name}{suffix}({args})'.format(
            ns=self._ns,
            name=known,
            suffix=suffix,
            args=args
        )

    def _print_Max(self, expr):
        return self._print_math_func(expr, nest=True)

    def _print_Min(self, expr):
        return self._print_math_func(expr, nest=True)

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (int %(var)s=%(start)s; %(var)s<%(end)s; %(var)s++){"  # C99
        for i in indices:
            # C arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'var': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines


for k in ('Abs Sqrt exp exp2 expm1 log log10 log2 log1p Cbrt hypot fma'
          ' loggamma sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh '
          'atanh erf erfc loggamma gamma ceiling floor').split():
    setattr(C99CodePrinter, '_print_%s' % k, C99CodePrinter._print_math_func)


class C11CodePrinter(C99CodePrinter):

    @requires(headers={'stdalign.h'})
    def _print_alignof(self, expr):
        arg, = expr.args
        return 'alignof(%s)' % self._print(arg)


c_code_printers = {
    'c89': C89CodePrinter,
    'c99': C99CodePrinter,
    'c11': C11CodePrinter
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\target.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Target
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page


class TargetID(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> TargetID:
        return cls(json)

    def __repr__(self):
        return 'TargetID({})'.format(super().__repr__())


class SessionID(str):
    '''
    Unique identifier of attached debugging session.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> SessionID:
        return cls(json)

    def __repr__(self):
        return 'SessionID({})'.format(super().__repr__())


@dataclass
class TargetInfo:
    target_id: TargetID

    #: List of types: https://source.chromium.org/chromium/chromium/src/+/main:content/browser/devtools/devtools_agent_host_impl.cc?ss=chromium&q=f:devtools%20-f:out%20%22::kTypeTab%5B%5D%22
    type_: str

    title: str

    url: str

    #: Whether the target has an attached client.
    attached: bool

    #: Whether the target has access to the originating window.
    can_access_opener: bool

    #: Opener target Id
    opener_id: typing.Optional[TargetID] = None

    #: Frame id of originating window (is only set if target has an opener).
    opener_frame_id: typing.Optional[page.FrameId] = None

    browser_context_id: typing.Optional[browser.BrowserContextID] = None

    #: Provides additional details for specific target types. For example, for
    #: the type of "page", this may be set to "prerender".
    subtype: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['targetId'] = self.target_id.to_json()
        json['type'] = self.type_
        json['title'] = self.title
        json['url'] = self.url
        json['attached'] = self.attached
        json['canAccessOpener'] = self.can_access_opener
        if self.opener_id is not None:
            json['openerId'] = self.opener_id.to_json()
        if self.opener_frame_id is not None:
            json['openerFrameId'] = self.opener_frame_id.to_json()
        if self.browser_context_id is not None:
            json['browserContextId'] = self.browser_context_id.to_json()
        if self.subtype is not None:
            json['subtype'] = self.subtype
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            target_id=TargetID.from_json(json['targetId']),
            type_=str(json['type']),
            title=str(json['title']),
            url=str(json['url']),
            attached=bool(json['attached']),
            can_access_opener=bool(json['canAccessOpener']),
            opener_id=TargetID.from_json(json['openerId']) if 'openerId' in json else None,
            opener_frame_id=page.FrameId.from_json(json['openerFrameId']) if 'openerFrameId' in json else None,
            browser_context_id=browser.BrowserContextID.from_json(json['browserContextId']) if 'browserContextId' in json else None,
            subtype=str(json['subtype']) if 'subtype' in json else None,
        )


@dataclass
class FilterEntry:
    '''
    A filter used by target query/discovery/auto-attach operations.
    '''
    #: If set, causes exclusion of matching targets from the list.
    exclude: typing.Optional[bool] = None

    #: If not present, matches any type.
    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.exclude is not None:
            json['exclude'] = self.exclude
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            exclude=bool(json['exclude']) if 'exclude' in json else None,
            type_=str(json['type']) if 'type' in json else None,
        )


class TargetFilter(list):
    '''
    The entries in TargetFilter are matched sequentially against targets and
    the first entry that matches determines if the target is included or not,
    depending on the value of ``exclude`` field in the entry.
    If filter is not specified, the one assumed is
    [{type: "browser", exclude: true}, {type: "tab", exclude: true}, {}]
    (i.e. include everything but ``browser`` and ``tab``).
    '''
    def to_json(self) -> typing.List[FilterEntry]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[FilterEntry]) -> TargetFilter:
        return cls(json)

    def __repr__(self):
        return 'TargetFilter({})'.format(super().__repr__())


@dataclass
class RemoteLocation:
    host: str

    port: int

    def to_json(self):
        json = dict()
        json['host'] = self.host
        json['port'] = self.port
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            host=str(json['host']),
            port=int(json['port']),
        )


class WindowState(enum.Enum):
    '''
    The state of the target window.
    '''
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    FULLSCREEN = "fullscreen"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def activate_target(
        target_id: TargetID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Activates (focuses) the target.

    :param target_id:
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.activateTarget',
        'params': params,
    }
    json = yield cmd_dict


def attach_to_target(
        target_id: TargetID,
        flatten: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SessionID]:
    '''
    Attaches to the target with given id.

    :param target_id:
    :param flatten: *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.
    :returns: Id assigned to the session.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    if flatten is not None:
        params['flatten'] = flatten
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.attachToTarget',
        'params': params,
    }
    json = yield cmd_dict
    return SessionID.from_json(json['sessionId'])


def attach_to_browser_target() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SessionID]:
    '''
    Attaches to the browser target, only uses flat sessionId mode.

    **EXPERIMENTAL**

    :returns: Id assigned to the session.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.attachToBrowserTarget',
    }
    json = yield cmd_dict
    return SessionID.from_json(json['sessionId'])


def close_target(
        target_id: TargetID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Closes the target. If the target is a page that gets closed too.

    :param target_id:
    :returns: Always set to true. If an error occurs, the response indicates protocol error.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.closeTarget',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['success'])


def expose_dev_tools_protocol(
        target_id: TargetID,
        binding_name: typing.Optional[str] = None,
        inherit_permissions: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Inject object to the target's main frame that provides a communication
    channel with browser target.

    Injected object will be available as ``window[bindingName]``.

    The object has the following API:
    - ``binding.send(json)`` - a method to send messages over the remote debugging protocol
    - ``binding.onmessage = json => handleMessage(json)`` - a callback that will be called for the protocol notifications and command responses.

    **EXPERIMENTAL**

    :param target_id:
    :param binding_name: *(Optional)* Binding name, 'cdp' if not specified.
    :param inherit_permissions: *(Optional)* If true, inherits the current root session's permissions (default: false).
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    if binding_name is not None:
        params['bindingName'] = binding_name
    if inherit_permissions is not None:
        params['inheritPermissions'] = inherit_permissions
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.exposeDevToolsProtocol',
        'params': params,
    }
    json = yield cmd_dict


def create_browser_context(
        dispose_on_detach: typing.Optional[bool] = None,
        proxy_server: typing.Optional[str] = None,
        proxy_bypass_list: typing.Optional[str] = None,
        origins_with_universal_network_access: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,browser.BrowserContextID]:
    '''
    Creates a new empty BrowserContext. Similar to an incognito profile but you can have more than
    one.

    :param dispose_on_detach: **(EXPERIMENTAL)** *(Optional)* If specified, disposes this context when debugging session disconnects.
    :param proxy_server: **(EXPERIMENTAL)** *(Optional)* Proxy server, similar to the one passed to --proxy-server
    :param proxy_bypass_list: **(EXPERIMENTAL)** *(Optional)* Proxy bypass list, similar to the one passed to --proxy-bypass-list
    :param origins_with_universal_network_access: **(EXPERIMENTAL)** *(Optional)* An optional list of origins to grant unlimited cross-origin access to. Parts of the URL other than those constituting origin are ignored.
    :returns: The id of the context created.
    '''
    params: T_JSON_DICT = dict()
    if dispose_on_detach is not None:
        params['disposeOnDetach'] = dispose_on_detach
    if proxy_server is not None:
        params['proxyServer'] = proxy_server
    if proxy_bypass_list is not None:
        params['proxyBypassList'] = proxy_bypass_list
    if origins_with_universal_network_access is not None:
        params['originsWithUniversalNetworkAccess'] = [i for i in origins_with_universal_network_access]
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.createBrowserContext',
        'params': params,
    }
    json = yield cmd_dict
    return browser.BrowserContextID.from_json(json['browserContextId'])


def get_browser_contexts() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[browser.BrowserContextID]]:
    '''
    Returns all browser contexts created with ``Target.createBrowserContext`` method.

    :returns: An array of browser context ids.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getBrowserContexts',
    }
    json = yield cmd_dict
    return [browser.BrowserContextID.from_json(i) for i in json['browserContextIds']]


def create_target(
        url: str,
        left: typing.Optional[int] = None,
        top: typing.Optional[int] = None,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        window_state: typing.Optional[WindowState] = None,
        browser_context_id: typing.Optional[browser.BrowserContextID] = None,
        enable_begin_frame_control: typing.Optional[bool] = None,
        new_window: typing.Optional[bool] = None,
        background: typing.Optional[bool] = None,
        for_tab: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,TargetID]:
    '''
    Creates a new page.

    :param url: The initial URL the page will be navigated to. An empty string indicates about:blank.
    :param left: **(EXPERIMENTAL)** *(Optional)* Frame left origin in DIP (requires newWindow to be true or headless shell).
    :param top: **(EXPERIMENTAL)** *(Optional)* Frame top origin in DIP (requires newWindow to be true or headless shell).
    :param width: *(Optional)* Frame width in DIP (requires newWindow to be true or headless shell).
    :param height: *(Optional)* Frame height in DIP (requires newWindow to be true or headless shell).
    :param window_state: *(Optional)* Frame window state (requires newWindow to be true or headless shell). Default is normal.
    :param browser_context_id: **(EXPERIMENTAL)** *(Optional)* The browser context to create the page in.
    :param enable_begin_frame_control: **(EXPERIMENTAL)** *(Optional)* Whether BeginFrames for this target will be controlled via DevTools (headless shell only, not supported on MacOS yet, false by default).
    :param new_window: *(Optional)* Whether to create a new Window or Tab (false by default, not supported by headless shell).
    :param background: *(Optional)* Whether to create the target in background or foreground (false by default, not supported by headless shell).
    :param for_tab: **(EXPERIMENTAL)** *(Optional)* Whether to create the target of type "tab".
    :returns: The id of the page opened.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    if left is not None:
        params['left'] = left
    if top is not None:
        params['top'] = top
    if width is not None:
        params['width'] = width
    if height is not None:
        params['height'] = height
    if window_state is not None:
        params['windowState'] = window_state.to_json()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    if enable_begin_frame_control is not None:
        params['enableBeginFrameControl'] = enable_begin_frame_control
    if new_window is not None:
        params['newWindow'] = new_window
    if background is not None:
        params['background'] = background
    if for_tab is not None:
        params['forTab'] = for_tab
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.createTarget',
        'params': params,
    }
    json = yield cmd_dict
    return TargetID.from_json(json['targetId'])


def detach_from_target(
        session_id: typing.Optional[SessionID] = None,
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Detaches session with given id.

    :param session_id: *(Optional)* Session to detach.
    :param target_id: *(Optional)* Deprecated.
    '''
    params: T_JSON_DICT = dict()
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.detachFromTarget',
        'params': params,
    }
    json = yield cmd_dict


def dispose_browser_context(
        browser_context_id: browser.BrowserContextID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a BrowserContext. All the belonging pages will be closed without calling their
    beforeunload hooks.

    :param browser_context_id:
    '''
    params: T_JSON_DICT = dict()
    params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.disposeBrowserContext',
        'params': params,
    }
    json = yield cmd_dict


def get_target_info(
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,TargetInfo]:
    '''
    Returns information about a target.

    **EXPERIMENTAL**

    :param target_id: *(Optional)*
    :returns:
    '''
    params: T_JSON_DICT = dict()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getTargetInfo',
        'params': params,
    }
    json = yield cmd_dict
    return TargetInfo.from_json(json['targetInfo'])


def get_targets(
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[TargetInfo]]:
    '''
    Retrieves a list of available targets.

    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be reported. If filter is not specified and target discovery is currently enabled, a filter used for target discovery is used for consistency.
    :returns: The list of targets.
    '''
    params: T_JSON_DICT = dict()
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getTargets',
        'params': params,
    }
    json = yield cmd_dict
    return [TargetInfo.from_json(i) for i in json['targetInfos']]


def send_message_to_target(
        message: str,
        session_id: typing.Optional[SessionID] = None,
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sends protocol message over session with given id.
    Consider using flat mode instead; see commands attachToTarget, setAutoAttach,
    and crbug.com/991325.

    :param message:
    :param session_id: *(Optional)* Identifier of the session.
    :param target_id: *(Optional)* Deprecated.
    '''
    params: T_JSON_DICT = dict()
    params['message'] = message
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.sendMessageToTarget',
        'params': params,
    }
    json = yield cmd_dict


def set_auto_attach(
        auto_attach: bool,
        wait_for_debugger_on_start: bool,
        flatten: typing.Optional[bool] = None,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Controls whether to automatically attach to new targets which are considered to be related to
    this one. When turned on, attaches to all existing related targets as well. When turned off,
    automatically detaches from all currently attached targets.
    This also clears all targets added by ``autoAttachRelated`` from the list of targets to watch
    for creation of related targets.

    :param auto_attach: Whether to auto-attach to related targets.
    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.
    :param flatten: **(EXPERIMENTAL)** *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.
    '''
    params: T_JSON_DICT = dict()
    params['autoAttach'] = auto_attach
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if flatten is not None:
        params['flatten'] = flatten
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setAutoAttach',
        'params': params,
    }
    json = yield cmd_dict


def auto_attach_related(
        target_id: TargetID,
        wait_for_debugger_on_start: bool,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Adds the specified target to the list of targets that will be monitored for any related target
    creation (such as child frames, child workers and new versions of service worker) and reported
    through ``attachedToTarget``. The specified target is also auto-attached.
    This cancels the effect of any previous ``setAutoAttach`` and is also cancelled by subsequent
    ``setAutoAttach``. Only available at the Browser target.

    **EXPERIMENTAL**

    :param target_id:
    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.autoAttachRelated',
        'params': params,
    }
    json = yield cmd_dict


def set_discover_targets(
        discover: bool,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Controls whether to discover available targets and notify via
    ``targetCreated/targetInfoChanged/targetDestroyed`` events.

    :param discover: Whether to discover available targets.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached. If ```discover```` is false, ````filter``` must be omitted or empty.
    '''
    params: T_JSON_DICT = dict()
    params['discover'] = discover
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setDiscoverTargets',
        'params': params,
    }
    json = yield cmd_dict


def set_remote_locations(
        locations: typing.List[RemoteLocation]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables target discovery for the specified locations, when ``setDiscoverTargets`` was set to
    ``true``.

    **EXPERIMENTAL**

    :param locations: List of remote locations.
    '''
    params: T_JSON_DICT = dict()
    params['locations'] = [i.to_json() for i in locations]
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setRemoteLocations',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Target.attachedToTarget')
@dataclass
class AttachedToTarget:
    '''
    **EXPERIMENTAL**

    Issued when attached to target because of auto-attach or ``attachToTarget`` command.
    '''
    #: Identifier assigned to the session used to send/receive messages.
    session_id: SessionID
    target_info: TargetInfo
    waiting_for_debugger: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttachedToTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            target_info=TargetInfo.from_json(json['targetInfo']),
            waiting_for_debugger=bool(json['waitingForDebugger'])
        )


@event_class('Target.detachedFromTarget')
@dataclass
class DetachedFromTarget:
    '''
    **EXPERIMENTAL**

    Issued when detached from target for any reason (including ``detachFromTarget`` command). Can be
    issued multiple times per target if multiple sessions have been attached to it.
    '''
    #: Detached session identifier.
    session_id: SessionID
    #: Deprecated.
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DetachedFromTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None
        )


@event_class('Target.receivedMessageFromTarget')
@dataclass
class ReceivedMessageFromTarget:
    '''
    Notifies about a new protocol message received from the session (as reported in
    ``attachedToTarget`` event).
    '''
    #: Identifier of a session which sends a message.
    session_id: SessionID
    message: str
    #: Deprecated.
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReceivedMessageFromTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            message=str(json['message']),
            target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None
        )


@event_class('Target.targetCreated')
@dataclass
class TargetCreated:
    '''
    Issued when a possible inspection target is created.
    '''
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCreated:
        return cls(
            target_info=TargetInfo.from_json(json['targetInfo'])
        )


@event_class('Target.targetDestroyed')
@dataclass
class TargetDestroyed:
    '''
    Issued when a target is destroyed.
    '''
    target_id: TargetID

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetDestroyed:
        return cls(
            target_id=TargetID.from_json(json['targetId'])
        )


@event_class('Target.targetCrashed')
@dataclass
class TargetCrashed:
    '''
    Issued when a target has crashed.
    '''
    target_id: TargetID
    #: Termination status type.
    status: str
    #: Termination error code.
    error_code: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCrashed:
        return cls(
            target_id=TargetID.from_json(json['targetId']),
            status=str(json['status']),
            error_code=int(json['errorCode'])
        )


@event_class('Target.targetInfoChanged')
@dataclass
class TargetInfoChanged:
    '''
    Issued when some information about a target has changed. This only happens between
    ``targetCreated`` and ``targetDestroyed``.
    '''
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetInfoChanged:
        return cls(
            target_info=TargetInfo.from_json(json['targetInfo'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\target.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Target
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page


class TargetID(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> TargetID:
        return cls(json)

    def __repr__(self):
        return 'TargetID({})'.format(super().__repr__())


class SessionID(str):
    '''
    Unique identifier of attached debugging session.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> SessionID:
        return cls(json)

    def __repr__(self):
        return 'SessionID({})'.format(super().__repr__())


@dataclass
class TargetInfo:
    target_id: TargetID

    #: List of types: https://source.chromium.org/chromium/chromium/src/+/main:content/browser/devtools/devtools_agent_host_impl.cc?ss=chromium&q=f:devtools%20-f:out%20%22::kTypeTab%5B%5D%22
    type_: str

    title: str

    url: str

    #: Whether the target has an attached client.
    attached: bool

    #: Whether the target has access to the originating window.
    can_access_opener: bool

    #: Opener target Id
    opener_id: typing.Optional[TargetID] = None

    #: Frame id of originating window (is only set if target has an opener).
    opener_frame_id: typing.Optional[page.FrameId] = None

    browser_context_id: typing.Optional[browser.BrowserContextID] = None

    #: Provides additional details for specific target types. For example, for
    #: the type of "page", this may be set to "prerender".
    subtype: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['targetId'] = self.target_id.to_json()
        json['type'] = self.type_
        json['title'] = self.title
        json['url'] = self.url
        json['attached'] = self.attached
        json['canAccessOpener'] = self.can_access_opener
        if self.opener_id is not None:
            json['openerId'] = self.opener_id.to_json()
        if self.opener_frame_id is not None:
            json['openerFrameId'] = self.opener_frame_id.to_json()
        if self.browser_context_id is not None:
            json['browserContextId'] = self.browser_context_id.to_json()
        if self.subtype is not None:
            json['subtype'] = self.subtype
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            target_id=TargetID.from_json(json['targetId']),
            type_=str(json['type']),
            title=str(json['title']),
            url=str(json['url']),
            attached=bool(json['attached']),
            can_access_opener=bool(json['canAccessOpener']),
            opener_id=TargetID.from_json(json['openerId']) if 'openerId' in json else None,
            opener_frame_id=page.FrameId.from_json(json['openerFrameId']) if 'openerFrameId' in json else None,
            browser_context_id=browser.BrowserContextID.from_json(json['browserContextId']) if 'browserContextId' in json else None,
            subtype=str(json['subtype']) if 'subtype' in json else None,
        )


@dataclass
class FilterEntry:
    '''
    A filter used by target query/discovery/auto-attach operations.
    '''
    #: If set, causes exclusion of matching targets from the list.
    exclude: typing.Optional[bool] = None

    #: If not present, matches any type.
    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.exclude is not None:
            json['exclude'] = self.exclude
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            exclude=bool(json['exclude']) if 'exclude' in json else None,
            type_=str(json['type']) if 'type' in json else None,
        )


class TargetFilter(list):
    '''
    The entries in TargetFilter are matched sequentially against targets and
    the first entry that matches determines if the target is included or not,
    depending on the value of ``exclude`` field in the entry.
    If filter is not specified, the one assumed is
    [{type: "browser", exclude: true}, {type: "tab", exclude: true}, {}]
    (i.e. include everything but ``browser`` and ``tab``).
    '''
    def to_json(self) -> typing.List[FilterEntry]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[FilterEntry]) -> TargetFilter:
        return cls(json)

    def __repr__(self):
        return 'TargetFilter({})'.format(super().__repr__())


@dataclass
class RemoteLocation:
    host: str

    port: int

    def to_json(self):
        json = dict()
        json['host'] = self.host
        json['port'] = self.port
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            host=str(json['host']),
            port=int(json['port']),
        )


class WindowState(enum.Enum):
    '''
    The state of the target window.
    '''
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    FULLSCREEN = "fullscreen"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def activate_target(
        target_id: TargetID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Activates (focuses) the target.

    :param target_id:
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.activateTarget',
        'params': params,
    }
    json = yield cmd_dict


def attach_to_target(
        target_id: TargetID,
        flatten: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SessionID]:
    '''
    Attaches to the target with given id.

    :param target_id:
    :param flatten: *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.
    :returns: Id assigned to the session.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    if flatten is not None:
        params['flatten'] = flatten
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.attachToTarget',
        'params': params,
    }
    json = yield cmd_dict
    return SessionID.from_json(json['sessionId'])


def attach_to_browser_target() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SessionID]:
    '''
    Attaches to the browser target, only uses flat sessionId mode.

    **EXPERIMENTAL**

    :returns: Id assigned to the session.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.attachToBrowserTarget',
    }
    json = yield cmd_dict
    return SessionID.from_json(json['sessionId'])


def close_target(
        target_id: TargetID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Closes the target. If the target is a page that gets closed too.

    :param target_id:
    :returns: Always set to true. If an error occurs, the response indicates protocol error.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.closeTarget',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['success'])


def expose_dev_tools_protocol(
        target_id: TargetID,
        binding_name: typing.Optional[str] = None,
        inherit_permissions: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Inject object to the target's main frame that provides a communication
    channel with browser target.

    Injected object will be available as ``window[bindingName]``.

    The object has the following API:
    - ``binding.send(json)`` - a method to send messages over the remote debugging protocol
    - ``binding.onmessage = json => handleMessage(json)`` - a callback that will be called for the protocol notifications and command responses.

    **EXPERIMENTAL**

    :param target_id:
    :param binding_name: *(Optional)* Binding name, 'cdp' if not specified.
    :param inherit_permissions: *(Optional)* If true, inherits the current root session's permissions (default: false).
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    if binding_name is not None:
        params['bindingName'] = binding_name
    if inherit_permissions is not None:
        params['inheritPermissions'] = inherit_permissions
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.exposeDevToolsProtocol',
        'params': params,
    }
    json = yield cmd_dict


def create_browser_context(
        dispose_on_detach: typing.Optional[bool] = None,
        proxy_server: typing.Optional[str] = None,
        proxy_bypass_list: typing.Optional[str] = None,
        origins_with_universal_network_access: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,browser.BrowserContextID]:
    '''
    Creates a new empty BrowserContext. Similar to an incognito profile but you can have more than
    one.

    :param dispose_on_detach: **(EXPERIMENTAL)** *(Optional)* If specified, disposes this context when debugging session disconnects.
    :param proxy_server: **(EXPERIMENTAL)** *(Optional)* Proxy server, similar to the one passed to --proxy-server
    :param proxy_bypass_list: **(EXPERIMENTAL)** *(Optional)* Proxy bypass list, similar to the one passed to --proxy-bypass-list
    :param origins_with_universal_network_access: **(EXPERIMENTAL)** *(Optional)* An optional list of origins to grant unlimited cross-origin access to. Parts of the URL other than those constituting origin are ignored.
    :returns: The id of the context created.
    '''
    params: T_JSON_DICT = dict()
    if dispose_on_detach is not None:
        params['disposeOnDetach'] = dispose_on_detach
    if proxy_server is not None:
        params['proxyServer'] = proxy_server
    if proxy_bypass_list is not None:
        params['proxyBypassList'] = proxy_bypass_list
    if origins_with_universal_network_access is not None:
        params['originsWithUniversalNetworkAccess'] = [i for i in origins_with_universal_network_access]
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.createBrowserContext',
        'params': params,
    }
    json = yield cmd_dict
    return browser.BrowserContextID.from_json(json['browserContextId'])


def get_browser_contexts() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[browser.BrowserContextID]]:
    '''
    Returns all browser contexts created with ``Target.createBrowserContext`` method.

    :returns: An array of browser context ids.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getBrowserContexts',
    }
    json = yield cmd_dict
    return [browser.BrowserContextID.from_json(i) for i in json['browserContextIds']]


def create_target(
        url: str,
        left: typing.Optional[int] = None,
        top: typing.Optional[int] = None,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        window_state: typing.Optional[WindowState] = None,
        browser_context_id: typing.Optional[browser.BrowserContextID] = None,
        enable_begin_frame_control: typing.Optional[bool] = None,
        new_window: typing.Optional[bool] = None,
        background: typing.Optional[bool] = None,
        for_tab: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,TargetID]:
    '''
    Creates a new page.

    :param url: The initial URL the page will be navigated to. An empty string indicates about:blank.
    :param left: **(EXPERIMENTAL)** *(Optional)* Frame left origin in DIP (requires newWindow to be true or headless shell).
    :param top: **(EXPERIMENTAL)** *(Optional)* Frame top origin in DIP (requires newWindow to be true or headless shell).
    :param width: *(Optional)* Frame width in DIP (requires newWindow to be true or headless shell).
    :param height: *(Optional)* Frame height in DIP (requires newWindow to be true or headless shell).
    :param window_state: *(Optional)* Frame window state (requires newWindow to be true or headless shell). Default is normal.
    :param browser_context_id: **(EXPERIMENTAL)** *(Optional)* The browser context to create the page in.
    :param enable_begin_frame_control: **(EXPERIMENTAL)** *(Optional)* Whether BeginFrames for this target will be controlled via DevTools (headless shell only, not supported on MacOS yet, false by default).
    :param new_window: *(Optional)* Whether to create a new Window or Tab (false by default, not supported by headless shell).
    :param background: *(Optional)* Whether to create the target in background or foreground (false by default, not supported by headless shell).
    :param for_tab: **(EXPERIMENTAL)** *(Optional)* Whether to create the target of type "tab".
    :returns: The id of the page opened.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    if left is not None:
        params['left'] = left
    if top is not None:
        params['top'] = top
    if width is not None:
        params['width'] = width
    if height is not None:
        params['height'] = height
    if window_state is not None:
        params['windowState'] = window_state.to_json()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    if enable_begin_frame_control is not None:
        params['enableBeginFrameControl'] = enable_begin_frame_control
    if new_window is not None:
        params['newWindow'] = new_window
    if background is not None:
        params['background'] = background
    if for_tab is not None:
        params['forTab'] = for_tab
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.createTarget',
        'params': params,
    }
    json = yield cmd_dict
    return TargetID.from_json(json['targetId'])


def detach_from_target(
        session_id: typing.Optional[SessionID] = None,
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Detaches session with given id.

    :param session_id: *(Optional)* Session to detach.
    :param target_id: *(Optional)* Deprecated.
    '''
    params: T_JSON_DICT = dict()
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.detachFromTarget',
        'params': params,
    }
    json = yield cmd_dict


def dispose_browser_context(
        browser_context_id: browser.BrowserContextID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a BrowserContext. All the belonging pages will be closed without calling their
    beforeunload hooks.

    :param browser_context_id:
    '''
    params: T_JSON_DICT = dict()
    params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.disposeBrowserContext',
        'params': params,
    }
    json = yield cmd_dict


def get_target_info(
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,TargetInfo]:
    '''
    Returns information about a target.

    **EXPERIMENTAL**

    :param target_id: *(Optional)*
    :returns:
    '''
    params: T_JSON_DICT = dict()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getTargetInfo',
        'params': params,
    }
    json = yield cmd_dict
    return TargetInfo.from_json(json['targetInfo'])


def get_targets(
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[TargetInfo]]:
    '''
    Retrieves a list of available targets.

    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be reported. If filter is not specified and target discovery is currently enabled, a filter used for target discovery is used for consistency.
    :returns: The list of targets.
    '''
    params: T_JSON_DICT = dict()
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.getTargets',
        'params': params,
    }
    json = yield cmd_dict
    return [TargetInfo.from_json(i) for i in json['targetInfos']]


def send_message_to_target(
        message: str,
        session_id: typing.Optional[SessionID] = None,
        target_id: typing.Optional[TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sends protocol message over session with given id.
    Consider using flat mode instead; see commands attachToTarget, setAutoAttach,
    and crbug.com/991325.

    :param message:
    :param session_id: *(Optional)* Identifier of the session.
    :param target_id: *(Optional)* Deprecated.
    '''
    params: T_JSON_DICT = dict()
    params['message'] = message
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.sendMessageToTarget',
        'params': params,
    }
    json = yield cmd_dict


def set_auto_attach(
        auto_attach: bool,
        wait_for_debugger_on_start: bool,
        flatten: typing.Optional[bool] = None,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Controls whether to automatically attach to new targets which are considered to be related to
    this one. When turned on, attaches to all existing related targets as well. When turned off,
    automatically detaches from all currently attached targets.
    This also clears all targets added by ``autoAttachRelated`` from the list of targets to watch
    for creation of related targets.

    :param auto_attach: Whether to auto-attach to related targets.
    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.
    :param flatten: **(EXPERIMENTAL)** *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.
    '''
    params: T_JSON_DICT = dict()
    params['autoAttach'] = auto_attach
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if flatten is not None:
        params['flatten'] = flatten
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setAutoAttach',
        'params': params,
    }
    json = yield cmd_dict


def auto_attach_related(
        target_id: TargetID,
        wait_for_debugger_on_start: bool,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Adds the specified target to the list of targets that will be monitored for any related target
    creation (such as child frames, child workers and new versions of service worker) and reported
    through ``attachedToTarget``. The specified target is also auto-attached.
    This cancels the effect of any previous ``setAutoAttach`` and is also cancelled by subsequent
    ``setAutoAttach``. Only available at the Browser target.

    **EXPERIMENTAL**

    :param target_id:
    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.
    '''
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.autoAttachRelated',
        'params': params,
    }
    json = yield cmd_dict


def set_discover_targets(
        discover: bool,
        filter_: typing.Optional[TargetFilter] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Controls whether to discover available targets and notify via
    ``targetCreated/targetInfoChanged/targetDestroyed`` events.

    :param discover: Whether to discover available targets.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached. If ```discover```` is false, ````filter``` must be omitted or empty.
    '''
    params: T_JSON_DICT = dict()
    params['discover'] = discover
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setDiscoverTargets',
        'params': params,
    }
    json = yield cmd_dict


def set_remote_locations(
        locations: typing.List[RemoteLocation]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables target discovery for the specified locations, when ``setDiscoverTargets`` was set to
    ``true``.

    **EXPERIMENTAL**

    :param locations: List of remote locations.
    '''
    params: T_JSON_DICT = dict()
    params['locations'] = [i.to_json() for i in locations]
    cmd_dict: T_JSON_DICT = {
        'method': 'Target.setRemoteLocations',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Target.attachedToTarget')
@dataclass
class AttachedToTarget:
    '''
    **EXPERIMENTAL**

    Issued when attached to target because of auto-attach or ``attachToTarget`` command.
    '''
    #: Identifier assigned to the session used to send/receive messages.
    session_id: SessionID
    target_info: TargetInfo
    waiting_for_debugger: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttachedToTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            target_info=TargetInfo.from_json(json['targetInfo']),
            waiting_for_debugger=bool(json['waitingForDebugger'])
        )


@event_class('Target.detachedFromTarget')
@dataclass
class DetachedFromTarget:
    '''
    **EXPERIMENTAL**

    Issued when detached from target for any reason (including ``detachFromTarget`` command). Can be
    issued multiple times per target if multiple sessions have been attached to it.
    '''
    #: Detached session identifier.
    session_id: SessionID
    #: Deprecated.
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DetachedFromTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None
        )


@event_class('Target.receivedMessageFromTarget')
@dataclass
class ReceivedMessageFromTarget:
    '''
    Notifies about a new protocol message received from the session (as reported in
    ``attachedToTarget`` event).
    '''
    #: Identifier of a session which sends a message.
    session_id: SessionID
    message: str
    #: Deprecated.
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReceivedMessageFromTarget:
        return cls(
            session_id=SessionID.from_json(json['sessionId']),
            message=str(json['message']),
            target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None
        )


@event_class('Target.targetCreated')
@dataclass
class TargetCreated:
    '''
    Issued when a possible inspection target is created.
    '''
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCreated:
        return cls(
            target_info=TargetInfo.from_json(json['targetInfo'])
        )


@event_class('Target.targetDestroyed')
@dataclass
class TargetDestroyed:
    '''
    Issued when a target is destroyed.
    '''
    target_id: TargetID

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetDestroyed:
        return cls(
            target_id=TargetID.from_json(json['targetId'])
        )


@event_class('Target.targetCrashed')
@dataclass
class TargetCrashed:
    '''
    Issued when a target has crashed.
    '''
    target_id: TargetID
    #: Termination status type.
    status: str
    #: Termination error code.
    error_code: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCrashed:
        return cls(
            target_id=TargetID.from_json(json['targetId']),
            status=str(json['status']),
            error_code=int(json['errorCode'])
        )


@event_class('Target.targetInfoChanged')
@dataclass
class TargetInfoChanged:
    '''
    Issued when some information about a target has changed. This only happens between
    ``targetCreated`` and ``targetDestroyed``.
    '''
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetInfoChanged:
        return cls(
            target_info=TargetInfo.from_json(json['targetInfo'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\partitions.py ===
from sympy.core import Basic, Dict, sympify, Tuple
from sympy.core.numbers import Integer
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.numbers import bell
from sympy.matrices import zeros
from sympy.sets.sets import FiniteSet, Union
from sympy.utilities.iterables import flatten, group
from sympy.utilities.misc import as_int


from collections import defaultdict


class Partition(FiniteSet):
    """
    This class represents an abstract partition.

    A partition is a set of disjoint sets whose union equals a given set.

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions
    """

    _rank = None
    _partition = None

    def __new__(cls, *partition):
        """
        Generates a new partition object.

        This method also verifies if the arguments passed are
        valid and raises a ValueError if they are not.

        Examples
        ========

        Creating Partition from Python lists:

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a
        Partition({3}, {1, 2})
        >>> a.partition
        [[1, 2], [3]]
        >>> len(a)
        2
        >>> a.members
        (1, 2, 3)

        Creating Partition from Python sets:

        >>> Partition({1, 2, 3}, {4, 5})
        Partition({4, 5}, {1, 2, 3})

        Creating Partition from SymPy finite sets:

        >>> from sympy import FiniteSet
        >>> a = FiniteSet(1, 2, 3)
        >>> b = FiniteSet(4, 5)
        >>> Partition(a, b)
        Partition({4, 5}, {1, 2, 3})
        """
        args = []
        dups = False
        for arg in partition:
            if isinstance(arg, list):
                as_set = set(arg)
                if len(as_set) < len(arg):
                    dups = True
                    break  # error below
                arg = as_set
            args.append(_sympify(arg))

        if not all(isinstance(part, FiniteSet) for part in args):
            raise ValueError(
                "Each argument to Partition should be " \
                "a list, set, or a FiniteSet")

        # sort so we have a canonical reference for RGS
        U = Union(*args)
        if dups or len(U) < sum(len(arg) for arg in args):
            raise ValueError("Partition contained duplicate elements.")

        obj = FiniteSet.__new__(cls, *args)
        obj.members = tuple(U)
        obj.size = len(U)
        return obj

    def sort_key(self, order=None):
        """Return a canonical key that can be used for sorting.

        Ordering is based on the size and sorted elements of the partition
        and ties are broken with the rank.

        Examples
        ========

        >>> from sympy import default_sort_key
        >>> from sympy.combinatorics import Partition
        >>> from sympy.abc import x
        >>> a = Partition([1, 2])
        >>> b = Partition([3, 4])
        >>> c = Partition([1, x])
        >>> d = Partition(list(range(4)))
        >>> l = [d, b, a + 1, a, c]
        >>> l.sort(key=default_sort_key); l
        [Partition({1, 2}), Partition({1}, {2}), Partition({1, x}), Partition({3, 4}), Partition({0, 1, 2, 3})]
        """
        if order is None:
            members = self.members
        else:
            members = tuple(sorted(self.members,
                             key=lambda w: default_sort_key(w, order)))
        return tuple(map(default_sort_key, (self.size, members, self.rank)))

    @property
    def partition(self):
        """Return partition as a sorted list of lists.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> Partition([1], [2, 3]).partition
        [[1], [2, 3]]
        """
        if self._partition is None:
            self._partition = sorted([sorted(p, key=default_sort_key)
                                      for p in self.args])
        return self._partition

    def __add__(self, other):
        """
        Return permutation whose rank is ``other`` greater than current rank,
        (mod the maximum rank for the set).

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a.rank
        1
        >>> (a + 1).rank
        2
        >>> (a + 100).rank
        1
        """
        other = as_int(other)
        offset = self.rank + other
        result = RGS_unrank((offset) %
                            RGS_enum(self.size),
                            self.size)
        return Partition.from_rgs(result, self.members)

    def __sub__(self, other):
        """
        Return permutation whose rank is ``other`` less than current rank,
        (mod the maximum rank for the set).

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a.rank
        1
        >>> (a - 1).rank
        0
        >>> (a - 100).rank
        1
        """
        return self.__add__(-other)

    def __le__(self, other):
        """
        Checks if a partition is less than or equal to
        the other based on rank.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3, 4, 5])
        >>> b = Partition([1], [2, 3], [4], [5])
        >>> a.rank, b.rank
        (9, 34)
        >>> a <= a
        True
        >>> a <= b
        True
        """
        return self.sort_key() <= sympify(other).sort_key()

    def __lt__(self, other):
        """
        Checks if a partition is less than the other.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3, 4, 5])
        >>> b = Partition([1], [2, 3], [4], [5])
        >>> a.rank, b.rank
        (9, 34)
        >>> a < b
        True
        """
        return self.sort_key() < sympify(other).sort_key()

    @property
    def rank(self):
        """
        Gets the rank of a partition.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3], [4, 5])
        >>> a.rank
        13
        """
        if self._rank is not None:
            return self._rank
        self._rank = RGS_rank(self.RGS)
        return self._rank

    @property
    def RGS(self):
        """
        Returns the "restricted growth string" of the partition.

        Explanation
        ===========

        The RGS is returned as a list of indices, L, where L[i] indicates
        the block in which element i appears. For example, in a partition
        of 3 elements (a, b, c) into 2 blocks ([c], [a, b]) the RGS is
        [1, 1, 0]: "a" is in block 1, "b" is in block 1 and "c" is in block 0.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3], [4, 5])
        >>> a.members
        (1, 2, 3, 4, 5)
        >>> a.RGS
        (0, 0, 1, 2, 2)
        >>> a + 1
        Partition({3}, {4}, {5}, {1, 2})
        >>> _.RGS
        (0, 0, 1, 2, 3)
        """
        rgs = {}
        partition = self.partition
        for i, part in enumerate(partition):
            for j in part:
                rgs[j] = i
        return tuple([rgs[i] for i in sorted(
            [i for p in partition for i in p], key=default_sort_key)])

    @classmethod
    def from_rgs(self, rgs, elements):
        """
        Creates a set partition from a restricted growth string.

        Explanation
        ===========

        The indices given in rgs are assumed to be the index
        of the element as given in elements *as provided* (the
        elements are not sorted by this routine). Block numbering
        starts from 0. If any block was not referenced in ``rgs``
        an error will be raised.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))
        Partition({c}, {a, d}, {b, e})
        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))
        Partition({e}, {a, c}, {b, d})
        >>> a = Partition([1, 4], [2], [3, 5])
        >>> Partition.from_rgs(a.RGS, a.members)
        Partition({2}, {1, 4}, {3, 5})
        """
        if len(rgs) != len(elements):
            raise ValueError('mismatch in rgs and element lengths')
        max_elem = max(rgs) + 1
        partition = [[] for i in range(max_elem)]
        j = 0
        for i in rgs:
            partition[i].append(elements[j])
            j += 1
        if not all(p for p in partition):
            raise ValueError('some blocks of the partition were empty.')
        return Partition(*partition)


class IntegerPartition(Basic):
    """
    This class represents an integer partition.

    Explanation
    ===========

    In number theory and combinatorics, a partition of a positive integer,
    ``n``, also called an integer partition, is a way of writing ``n`` as a
    list of positive integers that sum to n. Two partitions that differ only
    in the order of summands are considered to be the same partition; if order
    matters then the partitions are referred to as compositions. For example,
    4 has five partitions: [4], [3, 1], [2, 2], [2, 1, 1], and [1, 1, 1, 1];
    the compositions [1, 2, 1] and [1, 1, 2] are the same as partition
    [2, 1, 1].

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_%28number_theory%29
    """

    _dict = None
    _keys = None

    def __new__(cls, partition, integer=None):
        """
        Generates a new IntegerPartition object from a list or dictionary.

        Explanation
        ===========

        The partition can be given as a list of positive integers or a
        dictionary of (integer, multiplicity) items. If the partition is
        preceded by an integer an error will be raised if the partition
        does not sum to that given integer.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([5, 4, 3, 1, 1])
        >>> a
        IntegerPartition(14, (5, 4, 3, 1, 1))
        >>> print(a)
        [5, 4, 3, 1, 1]
        >>> IntegerPartition({1:3, 2:1})
        IntegerPartition(5, (2, 1, 1, 1))

        If the value that the partition should sum to is given first, a check
        will be made to see n error will be raised if there is a discrepancy:

        >>> IntegerPartition(10, [5, 4, 3, 1])
        Traceback (most recent call last):
        ...
        ValueError: The partition is not valid

        """
        if integer is not None:
            integer, partition = partition, integer
        if isinstance(partition, (dict, Dict)):
            _ = []
            for k, v in sorted(partition.items(), reverse=True):
                if not v:
                    continue
                k, v = as_int(k), as_int(v)
                _.extend([k]*v)
            partition = tuple(_)
        else:
            partition = tuple(sorted(map(as_int, partition), reverse=True))
        sum_ok = False
        if integer is None:
            integer = sum(partition)
            sum_ok = True
        else:
            integer = as_int(integer)

        if not sum_ok and sum(partition) != integer:
            raise ValueError("Partition did not add to %s" % integer)
        if any(i < 1 for i in partition):
            raise ValueError("All integer summands must be greater than one")

        obj = Basic.__new__(cls, Integer(integer), Tuple(*partition))
        obj.partition = list(partition)
        obj.integer = integer
        return obj

    def prev_lex(self):
        """Return the previous partition of the integer, n, in lexical order,
        wrapping around to [1, ..., 1] if the partition is [n].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([4])
        >>> print(p.prev_lex())
        [3, 1]
        >>> p.partition > p.prev_lex().partition
        True
        """
        d = defaultdict(int)
        d.update(self.as_dict())
        keys = self._keys
        if keys == [1]:
            return IntegerPartition({self.integer: 1})
        if keys[-1] != 1:
            d[keys[-1]] -= 1
            if keys[-1] == 2:
                d[1] = 2
            else:
                d[keys[-1] - 1] = d[1] = 1
        else:
            d[keys[-2]] -= 1
            left = d[1] + keys[-2]
            new = keys[-2]
            d[1] = 0
            while left:
                new -= 1
                if left - new >= 0:
                    d[new] += left//new
                    left -= d[new]*new
        return IntegerPartition(self.integer, d)

    def next_lex(self):
        """Return the next partition of the integer, n, in lexical order,
        wrapping around to [n] if the partition is [1, ..., 1].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([3, 1])
        >>> print(p.next_lex())
        [4]
        >>> p.partition < p.next_lex().partition
        True
        """
        d = defaultdict(int)
        d.update(self.as_dict())
        key = self._keys
        a = key[-1]
        if a == self.integer:
            d.clear()
            d[1] = self.integer
        elif a == 1:
            if d[a] > 1:
                d[a + 1] += 1
                d[a] -= 2
            else:
                b = key[-2]
                d[b + 1] += 1
                d[1] = (d[b] - 1)*b
                d[b] = 0
        else:
            if d[a] > 1:
                if len(key) == 1:
                    d.clear()
                    d[a + 1] = 1
                    d[1] = self.integer - a - 1
                else:
                    a1 = a + 1
                    d[a1] += 1
                    d[1] = d[a]*a - a1
                    d[a] = 0
            else:
                b = key[-2]
                b1 = b + 1
                d[b1] += 1
                need = d[b]*b + d[a]*a - b1
                d[a] = d[b] = 0
                d[1] = need
        return IntegerPartition(self.integer, d)

    def as_dict(self):
        """Return the partition as a dictionary whose keys are the
        partition integers and the values are the multiplicity of that
        integer.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> IntegerPartition([1]*3 + [2] + [3]*4).as_dict()
        {1: 3, 2: 1, 3: 4}
        """
        if self._dict is None:
            groups = group(self.partition, multiple=False)
            self._keys = [g[0] for g in groups]
            self._dict = dict(groups)
        return self._dict

    @property
    def conjugate(self):
        """
        Computes the conjugate partition of itself.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([6, 3, 3, 2, 1])
        >>> a.conjugate
        [5, 4, 3, 1, 1, 1]
        """
        j = 1
        temp_arr = list(self.partition) + [0]
        k = temp_arr[0]
        b = [0]*k
        while k > 0:
            while k > temp_arr[j]:
                b[k - 1] = j
                k -= 1
            j += 1
        return b

    def __lt__(self, other):
        """Return True if self is less than other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([3, 1])
        >>> a < a
        False
        >>> b = a.next_lex()
        >>> a < b
        True
        >>> a == b
        False
        """
        return list(reversed(self.partition)) < list(reversed(other.partition))

    def __le__(self, other):
        """Return True if self is less than other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([4])
        >>> a <= a
        True
        """
        return list(reversed(self.partition)) <= list(reversed(other.partition))

    def as_ferrers(self, char='#'):
        """
        Prints the ferrer diagram of a partition.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> print(IntegerPartition([1, 1, 5]).as_ferrers())
        #####
        #
        #
        """
        return "\n".join([char*i for i in self.partition])

    def __str__(self):
        return str(list(self.partition))


def random_integer_partition(n, seed=None):
    """
    Generates a random integer partition summing to ``n`` as a list
    of reverse-sorted integers.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import random_integer_partition

    For the following, a seed is given so a known value can be shown; in
    practice, the seed would not be given.

    >>> random_integer_partition(100, seed=[1, 1, 12, 1, 2, 1, 85, 1])
    [85, 12, 2, 1]
    >>> random_integer_partition(10, seed=[1, 2, 3, 1, 5, 1])
    [5, 3, 1, 1]
    >>> random_integer_partition(1)
    [1]
    """
    from sympy.core.random import _randint

    n = as_int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')

    randint = _randint(seed)

    partition = []
    while (n > 0):
        k = randint(1, n)
        mult = randint(1, n//k)
        partition.append((k, mult))
        n -= k*mult
    partition.sort(reverse=True)
    partition = flatten([[k]*m for k, m in partition])
    return partition


def RGS_generalized(m):
    """
    Computes the m + 1 generalized unrestricted growth strings
    and returns them as rows in matrix.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_generalized
    >>> RGS_generalized(6)
    Matrix([
    [  1,   1,   1,  1,  1, 1, 1],
    [  1,   2,   3,  4,  5, 6, 0],
    [  2,   5,  10, 17, 26, 0, 0],
    [  5,  15,  37, 77,  0, 0, 0],
    [ 15,  52, 151,  0,  0, 0, 0],
    [ 52, 203,   0,  0,  0, 0, 0],
    [203,   0,   0,  0,  0, 0, 0]])
    """
    d = zeros(m + 1)
    for i in range(m + 1):
        d[0, i] = 1

    for i in range(1, m + 1):
        for j in range(m):
            if j <= m - i:
                d[i, j] = j * d[i - 1, j] + d[i - 1, j + 1]
            else:
                d[i, j] = 0
    return d


def RGS_enum(m):
    """
    RGS_enum computes the total number of restricted growth strings
    possible for a superset of size m.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_enum
    >>> from sympy.combinatorics import Partition
    >>> RGS_enum(4)
    15
    >>> RGS_enum(5)
    52
    >>> RGS_enum(6)
    203

    We can check that the enumeration is correct by actually generating
    the partitions. Here, the 15 partitions of 4 items are generated:

    >>> a = Partition(list(range(4)))
    >>> s = set()
    >>> for i in range(20):
    ...     s.add(a)
    ...     a += 1
    ...
    >>> assert len(s) == 15

    """
    if (m < 1):
        return 0
    elif (m == 1):
        return 1
    else:
        return bell(m)


def RGS_unrank(rank, m):
    """
    Gives the unranked restricted growth string for a given
    superset size.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_unrank
    >>> RGS_unrank(14, 4)
    [0, 1, 2, 3]
    >>> RGS_unrank(0, 4)
    [0, 0, 0, 0]
    """
    if m < 1:
        raise ValueError("The superset size must be >= 1")
    if rank < 0 or RGS_enum(m) <= rank:
        raise ValueError("Invalid arguments")

    L = [1] * (m + 1)
    j = 1
    D = RGS_generalized(m)
    for i in range(2, m + 1):
        v = D[m - i, j]
        cr = j*v
        if cr <= rank:
            L[i] = j + 1
            rank -= cr
            j += 1
        else:
            L[i] = int(rank / v + 1)
            rank %= v
    return [x - 1 for x in L[1:]]


def RGS_rank(rgs):
    """
    Computes the rank of a restricted growth string.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_rank, RGS_unrank
    >>> RGS_rank([0, 1, 2, 1, 3])
    42
    >>> RGS_rank(RGS_unrank(4, 7))
    4
    """
    rgs_size = len(rgs)
    rank = 0
    D = RGS_generalized(rgs_size)
    for i in range(1, rgs_size):
        n = len(rgs[(i + 1):])
        m = max(rgs[0:i])
        rank += D[n, m + 1] * rgs[i]
    return rank

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\operations.py ===
from __future__ import annotations

from typing import overload, TYPE_CHECKING

from operator import attrgetter
from collections import defaultdict

from sympy.utilities.exceptions import sympy_deprecation_warning

from .sympify import _sympify as _sympify_, sympify
from .basic import Basic
from .cache import cacheit
from .sorting import ordered
from .logic import fuzzy_and
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from sympy.multipledispatch.dispatcher import (Dispatcher,
    ambiguity_register_error_ignore_dup,
    str_signature, RaiseNotImplementedError)


if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    from sympy.logic.boolalg import Boolean, And, Or


class AssocOp(Basic):
    """ Associative operations, can separate noncommutative and
    commutative parts.

    (a op b) op c == a op (b op c) == a op b op c.

    Base class for Add and Mul.

    This is an abstract base class, concrete derived classes must define
    the attribute `identity`.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Parameters
    ==========

    *args :
        Arguments which are operated

    evaluate : bool, optional
        Evaluate the operation. If not passed, refer to ``global_parameters.evaluate``.
    """

    # for performance reason, we don't let is_commutative go to assumptions,
    # and keep it right here
    __slots__: tuple[str, ...] = ('is_commutative',)

    _args_type: type[Basic] | None = None

    @cacheit
    def __new__(cls, *args, evaluate=None, _sympify=True):
        # Allow faster processing by passing ``_sympify=False``, if all arguments
        # are already sympified.
        if _sympify:
            args = list(map(_sympify_, args))

        # Disallow non-Expr args in Add/Mul
        typ = cls._args_type
        if typ is not None:
            from .relational import Relational
            if any(isinstance(arg, Relational) for arg in args):
                raise TypeError("Relational cannot be used in %s" % cls.__name__)

            # This should raise TypeError once deprecation period is over:
            for arg in args:
                if not isinstance(arg, typ):
                    sympy_deprecation_warning(
                        f"""

Using non-Expr arguments in {cls.__name__} is deprecated (in this case, one of
the arguments has type {type(arg).__name__!r}).

If you really did intend to use a multiplication or addition operation with
this object, use the * or + operator instead.

                        """,
                        deprecated_since_version="1.7",
                        active_deprecations_target="non-expr-args-deprecated",
                        stacklevel=4,
                    )

        if evaluate is None:
            evaluate = global_parameters.evaluate
        if not evaluate:
            obj = cls._from_args(args)
            obj = cls._exec_constructor_postprocessors(obj)
            return obj

        args = [a for a in args if a is not cls.identity]

        if len(args) == 0:
            return cls.identity
        if len(args) == 1:
            return args[0]

        c_part, nc_part, order_symbols = cls.flatten(args)
        is_commutative = not nc_part
        obj = cls._from_args(c_part + nc_part, is_commutative)
        obj = cls._exec_constructor_postprocessors(obj)

        if order_symbols is not None:
            from sympy.series.order import Order
            return Order(obj, *order_symbols)
        return obj

    @classmethod
    def _from_args(cls, args, is_commutative=None):
        """Create new instance with already-processed args.
        If the args are not in canonical order, then a non-canonical
        result will be returned, so use with caution. The order of
        args may change if the sign of the args is changed."""
        if len(args) == 0:
            return cls.identity
        elif len(args) == 1:
            return args[0]

        obj = super().__new__(cls, *args)
        if is_commutative is None:
            is_commutative = fuzzy_and(a.is_commutative for a in args)
        obj.is_commutative = is_commutative
        return obj

    def _new_rawargs(self, *args, reeval=True, **kwargs):
        """Create new instance of own class with args exactly as provided by
        caller but returning the self class identity if args is empty.

        Examples
        ========

           This is handy when we want to optimize things, e.g.

               >>> from sympy import Mul, S
               >>> from sympy.abc import x, y
               >>> e = Mul(3, x, y)
               >>> e.args
               (3, x, y)
               >>> Mul(*e.args[1:])
               x*y
               >>> e._new_rawargs(*e.args[1:])  # the same as above, but faster
               x*y

           Note: use this with caution. There is no checking of arguments at
           all. This is best used when you are rebuilding an Add or Mul after
           simply removing one or more args. If, for example, modifications,
           result in extra 1s being inserted they will show up in the result:

               >>> m = (x*y)._new_rawargs(S.One, x); m
               1*x
               >>> m == x
               False
               >>> m.is_Mul
               True

           Another issue to be aware of is that the commutativity of the result
           is based on the commutativity of self. If you are rebuilding the
           terms that came from a commutative object then there will be no
           problem, but if self was non-commutative then what you are
           rebuilding may now be commutative.

           Although this routine tries to do as little as possible with the
           input, getting the commutativity right is important, so this level
           of safety is enforced: commutativity will always be recomputed if
           self is non-commutative and kwarg `reeval=False` has not been
           passed.
        """
        if reeval and self.is_commutative is False:
            is_commutative = None
        else:
            is_commutative = self.is_commutative
        return self._from_args(args, is_commutative)

    @classmethod
    def flatten(cls, seq):
        """Return seq so that none of the elements are of type `cls`. This is
        the vanilla routine that will be used if a class derived from AssocOp
        does not define its own flatten routine."""
        # apply associativity, no commutativity property is used
        new_seq = []
        while seq:
            o = seq.pop()
            if o.__class__ is cls:  # classes must match exactly
                seq.extend(o.args)
            else:
                new_seq.append(o)
        new_seq.reverse()

        # c_part, nc_part, order_symbols
        return [], new_seq, None

    def _matches_commutative(self, expr, repl_dict=None, old=False):
        """
        Matches Add/Mul "pattern" to an expression "expr".

        repl_dict ... a dictionary of (wild: expression) pairs, that get
                      returned with the results

        This function is the main workhorse for Add/Mul.

        Examples
        ========

        >>> from sympy import symbols, Wild, sin
        >>> a = Wild("a")
        >>> b = Wild("b")
        >>> c = Wild("c")
        >>> x, y, z = symbols("x y z")
        >>> (a+sin(b)*c)._matches_commutative(x+sin(y)*z)
        {a_: x, b_: y, c_: z}

        In the example above, "a+sin(b)*c" is the pattern, and "x+sin(y)*z" is
        the expression.

        The repl_dict contains parts that were already matched. For example
        here:

        >>> (x+sin(b)*c)._matches_commutative(x+sin(y)*z, repl_dict={a: x})
        {a_: x, b_: y, c_: z}

        the only function of the repl_dict is to return it in the
        result, e.g. if you omit it:

        >>> (x+sin(b)*c)._matches_commutative(x+sin(y)*z)
        {b_: y, c_: z}

        the "a: x" is not returned in the result, but otherwise it is
        equivalent.

        """
        from .function import _coeff_isneg
        # make sure expr is Expr if pattern is Expr
        from .expr import Expr
        if isinstance(self, Expr) and not isinstance(expr, Expr):
            return None

        if repl_dict is None:
            repl_dict = {}

        # handle simple patterns
        if self == expr:
            return repl_dict

        d = self._matches_simple(expr, repl_dict)
        if d is not None:
            return d

        # eliminate exact part from pattern: (2+a+w1+w2).matches(expr) -> (w1+w2).matches(expr-a-2)
        from .function import WildFunction
        from .symbol import Wild
        wild_part, exact_part = sift(self.args, lambda p:
            p.has(Wild, WildFunction) and not expr.has(p),
            binary=True)
        if not exact_part:
            wild_part = list(ordered(wild_part))
            if self.is_Add:
                # in addition to normal ordered keys, impose
                # sorting on Muls with leading Number to put
                # them in order
                wild_part = sorted(wild_part, key=lambda x:
                    x.args[0] if x.is_Mul and x.args[0].is_Number else
                    0)
        else:
            exact = self._new_rawargs(*exact_part)
            free = expr.free_symbols
            if free and (exact.free_symbols - free):
                # there are symbols in the exact part that are not
                # in the expr; but if there are no free symbols, let
                # the matching continue
                return None
            newexpr = self._combine_inverse(expr, exact)
            if not old and (expr.is_Add or expr.is_Mul):
                check = newexpr
                if _coeff_isneg(check):
                    check = -check
                if check.count_ops() > expr.count_ops():
                    return None
            newpattern = self._new_rawargs(*wild_part)
            return newpattern.matches(newexpr, repl_dict)

        # now to real work ;)
        i = 0
        saw = set()
        while expr not in saw:
            saw.add(expr)
            args = tuple(ordered(self.make_args(expr)))
            if self.is_Add and expr.is_Add:
                # in addition to normal ordered keys, impose
                # sorting on Muls with leading Number to put
                # them in order
                args = tuple(sorted(args, key=lambda x:
                    x.args[0] if x.is_Mul and x.args[0].is_Number else
                    0))
            expr_list = (self.identity,) + args
            for last_op in reversed(expr_list):
                for w in reversed(wild_part):
                    d1 = w.matches(last_op, repl_dict)
                    if d1 is not None:
                        d2 = self.xreplace(d1).matches(expr, d1)
                        if d2 is not None:
                            return d2

            if i == 0:
                if self.is_Mul:
                    # make e**i look like Mul
                    if expr.is_Pow and expr.exp.is_Integer:
                        from .mul import Mul
                        if expr.exp > 0:
                            expr = Mul(*[expr.base, expr.base**(expr.exp - 1)], evaluate=False)
                        else:
                            expr = Mul(*[1/expr.base, expr.base**(expr.exp + 1)], evaluate=False)
                        i += 1
                        continue

                elif self.is_Add:
                    # make i*e look like Add
                    c, e = expr.as_coeff_Mul()
                    if abs(c) > 1:
                        from .add import Add
                        if c > 0:
                            expr = Add(*[e, (c - 1)*e], evaluate=False)
                        else:
                            expr = Add(*[-e, (c + 1)*e], evaluate=False)
                        i += 1
                        continue

                    # try collection on non-Wild symbols
                    from sympy.simplify.radsimp import collect
                    was = expr
                    did = set()
                    for w in reversed(wild_part):
                        c, w = w.as_coeff_mul(Wild)
                        free = c.free_symbols - did
                        if free:
                            did.update(free)
                            expr = collect(expr, free)
                    if expr != was:
                        i += 0
                        continue

                break  # if we didn't continue, there is nothing more to do

        return

    def _has_matcher(self):
        """Helper for .has() that checks for containment of
        subexpressions within an expr by using sets of args
        of similar nodes, e.g. x + 1 in x + y + 1 checks
        to see that {x, 1} & {x, y, 1} == {x, 1}
        """
        def _ncsplit(expr):
            # this is not the same as args_cnc because here
            # we don't assume expr is a Mul -- hence deal with args --
            # and always return a set.
            cpart, ncpart = sift(expr.args,
                lambda arg: arg.is_commutative is True, binary=True)
            return set(cpart), ncpart

        c, nc = _ncsplit(self)
        cls = self.__class__

        def is_in(expr):
            if isinstance(expr, cls):
                if expr == self:
                    return True
                _c, _nc = _ncsplit(expr)
                if (c & _c) == c:
                    if not nc:
                        return True
                    elif len(nc) <= len(_nc):
                        for i in range(len(_nc) - len(nc) + 1):
                            if _nc[i:i + len(nc)] == nc:
                                return True
            return False
        return is_in

    def _eval_evalf(self, prec):
        """
        Evaluate the parts of self that are numbers; if the whole thing
        was a number with no functions it would have been evaluated, but
        it wasn't so we must judiciously extract the numbers and reconstruct
        the object. This is *not* simply replacing numbers with evaluated
        numbers. Numbers should be handled in the largest pure-number
        expression as possible. So the code below separates ``self`` into
        number and non-number parts and evaluates the number parts and
        walks the args of the non-number part recursively (doing the same
        thing).
        """
        from .add import Add
        from .mul import Mul
        from .symbol import Symbol
        from .function import AppliedUndef
        if isinstance(self, (Mul, Add)):
            x, tail = self.as_independent(Symbol, AppliedUndef)
            # if x is an AssocOp Function then the _evalf below will
            # call _eval_evalf (here) so we must break the recursion
            if not (tail is self.identity or
                    isinstance(x, AssocOp) and x.is_Function or
                    x is self.identity and isinstance(tail, AssocOp)):
                # here, we have a number so we just call to _evalf with prec;
                # prec is not the same as n, it is the binary precision so
                # that's why we don't call to evalf.
                x = x._evalf(prec) if x is not self.identity else self.identity
                args = []
                tail_args = tuple(self.func.make_args(tail))
                for a in tail_args:
                    # here we call to _eval_evalf since we don't know what we
                    # are dealing with and all other _eval_evalf routines should
                    # be doing the same thing (i.e. taking binary prec and
                    # finding the evalf-able args)
                    newa = a._eval_evalf(prec)
                    if newa is None:
                        args.append(a)
                    else:
                        args.append(newa)
                return self.func(x, *args)

        # this is the same as above, but there were no pure-number args to
        # deal with
        args = []
        for a in self.args:
            newa = a._eval_evalf(prec)
            if newa is None:
                args.append(a)
            else:
                args.append(newa)
        return self.func(*args)

    @overload
    @classmethod
    def make_args(cls: type[Add], expr: Expr) -> tuple[Expr, ...]: ... # type: ignore
    @overload
    @classmethod
    def make_args(cls: type[Mul], expr: Expr) -> tuple[Expr, ...]: ... # type: ignore
    @overload
    @classmethod
    def make_args(cls: type[And], expr: Boolean) -> tuple[Boolean, ...]: ... # type: ignore
    @overload
    @classmethod
    def make_args(cls: type[Or], expr: Boolean) -> tuple[Boolean, ...]: ... # type: ignore

    @classmethod
    def make_args(cls: type[Basic], expr: Basic) -> tuple[Basic, ...]:
        """
        Return a sequence of elements `args` such that cls(*args) == expr

        Examples
        ========

        >>> from sympy import Symbol, Mul, Add
        >>> x, y = map(Symbol, 'xy')

        >>> Mul.make_args(x*y)
        (x, y)
        >>> Add.make_args(x*y)
        (x*y,)
        >>> set(Add.make_args(x*y + y)) == set([y, x*y])
        True

        """
        if isinstance(expr, cls):
            return expr.args
        else:
            return (sympify(expr),)

    def doit(self, **hints):
        if hints.get('deep', True):
            terms = [term.doit(**hints) for term in self.args]
        else:
            terms = self.args
        return self.func(*terms, evaluate=True)

class ShortCircuit(Exception):
    pass


class LatticeOp(AssocOp):
    """
    Join/meet operations of an algebraic lattice[1].

    Explanation
    ===========

    These binary operations are associative (op(op(a, b), c) = op(a, op(b, c))),
    commutative (op(a, b) = op(b, a)) and idempotent (op(a, a) = op(a) = a).
    Common examples are AND, OR, Union, Intersection, max or min. They have an
    identity element (op(identity, a) = a) and an absorbing element
    conventionally called zero (op(zero, a) = zero).

    This is an abstract base class, concrete derived classes must declare
    attributes zero and identity. All defining properties are then respected.

    Examples
    ========

    >>> from sympy import Integer
    >>> from sympy.core.operations import LatticeOp
    >>> class my_join(LatticeOp):
    ...     zero = Integer(0)
    ...     identity = Integer(1)
    >>> my_join(2, 3) == my_join(3, 2)
    True
    >>> my_join(2, my_join(3, 4)) == my_join(2, 3, 4)
    True
    >>> my_join(0, 1, 4, 2, 3, 4)
    0
    >>> my_join(1, 2)
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lattice_%28order%29
    """

    is_commutative = True

    def __new__(cls, *args, **options):
        args = (_sympify_(arg) for arg in args)

        try:
            # /!\ args is a generator and _new_args_filter
            # must be careful to handle as such; this
            # is done so short-circuiting can be done
            # without having to sympify all values
            _args = frozenset(cls._new_args_filter(args))
        except ShortCircuit:
            return sympify(cls.zero)
        if not _args:
            return sympify(cls.identity)
        elif len(_args) == 1:
            return set(_args).pop()
        else:
            # XXX in almost every other case for __new__, *_args is
            # passed along, but the expectation here is for _args
            obj = super(AssocOp, cls).__new__(cls, *ordered(_args))
            obj._argset = _args
            return obj

    @classmethod
    def _new_args_filter(cls, arg_sequence, call_cls=None):
        """Generator filtering args"""
        ncls = call_cls or cls
        for arg in arg_sequence:
            if arg == ncls.zero:
                raise ShortCircuit(arg)
            elif arg == ncls.identity:
                continue
            elif arg.func == ncls:
                yield from arg.args
            else:
                yield arg

    @classmethod
    def make_args(cls, expr):
        """
        Return a set of args such that cls(*arg_set) == expr.
        """
        if isinstance(expr, cls):
            return expr._argset
        else:
            return frozenset([sympify(expr)])


class AssocOpDispatcher:
    """
    Handler dispatcher for associative operators

    .. notes::
       This approach is experimental, and can be replaced or deleted in the future.
       See https://github.com/sympy/sympy/pull/19463.

    Explanation
    ===========

    If arguments of different types are passed, the classes which handle the operation for each type
    are collected. Then, a class which performs the operation is selected by recursive binary dispatching.
    Dispatching relation can be registered by ``register_handlerclass`` method.

    Priority registration is unordered. You cannot make ``A*B`` and ``B*A`` refer to
    different handler classes. All logic dealing with the order of arguments must be implemented
    in the handler class.

    Examples
    ========

    >>> from sympy import Add, Expr, Symbol
    >>> from sympy.core.add import add

    >>> class NewExpr(Expr):
    ...     @property
    ...     def _add_handler(self):
    ...         return NewAdd
    >>> class NewAdd(NewExpr, Add):
    ...     pass
    >>> add.register_handlerclass((Add, NewAdd), NewAdd)

    >>> a, b = Symbol('a'), NewExpr()
    >>> add(a, b) == NewAdd(a, b)
    True

    """
    def __init__(self, name, doc=None):
        self.name = name
        self.doc = doc
        self.handlerattr = "_%s_handler" % name
        self._handlergetter = attrgetter(self.handlerattr)
        self._dispatcher = Dispatcher(name)

    def __repr__(self):
        return "<dispatched %s>" % self.name

    def register_handlerclass(self, classes, typ, on_ambiguity=ambiguity_register_error_ignore_dup):
        """
        Register the handler class for two classes, in both straight and reversed order.

        Paramteters
        ===========

        classes : tuple of two types
            Classes who are compared with each other.

        typ:
            Class which is registered to represent *cls1* and *cls2*.
            Handler method of *self* must be implemented in this class.
        """
        if not len(classes) == 2:
            raise RuntimeError(
                "Only binary dispatch is supported, but got %s types: <%s>." % (
                len(classes), str_signature(classes)
            ))
        if len(set(classes)) == 1:
            raise RuntimeError(
                "Duplicate types <%s> cannot be dispatched." % str_signature(classes)
            )
        self._dispatcher.add(tuple(classes), typ, on_ambiguity=on_ambiguity)
        self._dispatcher.add(tuple(reversed(classes)), typ, on_ambiguity=on_ambiguity)

    @cacheit
    def __call__(self, *args, _sympify=True, **kwargs):
        """
        Parameters
        ==========

        *args :
            Arguments which are operated
        """
        if _sympify:
            args = tuple(map(_sympify_, args))
        handlers = frozenset(map(self._handlergetter, args))

        # no need to sympify again
        return self.dispatch(handlers)(*args, _sympify=False, **kwargs)

    @cacheit
    def dispatch(self, handlers):
        """
        Select the handler class, and return its handler method.
        """

        # Quick exit for the case where all handlers are same
        if len(handlers) == 1:
            h, = handlers
            if not isinstance(h, type):
                raise RuntimeError("Handler {!r} is not a type.".format(h))
            return h

        # Recursively select with registered binary priority
        for i, typ in enumerate(handlers):

            if not isinstance(typ, type):
                raise RuntimeError("Handler {!r} is not a type.".format(typ))

            if i == 0:
                handler = typ
            else:
                prev_handler = handler
                handler = self._dispatcher.dispatch(prev_handler, typ)

                if not isinstance(handler, type):
                    raise RuntimeError(
                        "Dispatcher for {!r} and {!r} must return a type, but got {!r}".format(
                        prev_handler, typ, handler
                    ))

        # return handler class
        return handler

    @property
    def __doc__(self):
        docs = [
            "Multiply dispatched associative operator: %s" % self.name,
            "Note that support for this is experimental, see the docs for :class:`AssocOpDispatcher` for details"
        ]

        if self.doc:
            docs.append(self.doc)

        s = "Registered handler classes\n"
        s += '=' * len(s)
        docs.append(s)

        amb_sigs = []

        typ_sigs = defaultdict(list)
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)

        for typ, sigs in typ_sigs.items():

            sigs_str = ', '.join('<%s>' % str_signature(sig) for sig in sigs)

            if isinstance(typ, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue

            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            s += typ.__name__
            docs.append(s)

        if amb_sigs:
            s = "Ambiguous handler classes\n"
            s += '=' * len(s)
            docs.append(s)

            s = '\n'.join(amb_sigs)
            docs.append(s)

        return '\n\n'.join(docs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\distributedmodules.py ===
r"""
Sparse distributed elements of free modules over multivariate (generalized)
polynomial rings.

This code and its data structures are very much like the distributed
polynomials, except that the first "exponent" of the monomial is
a module generator index. That is, the multi-exponent ``(i, e_1, ..., e_n)``
represents the "monomial" `x_1^{e_1} \cdots x_n^{e_n} f_i` of the free module
`F` generated by `f_1, \ldots, f_r` over (a localization of) the ring
`K[x_1, \ldots, x_n]`. A module element is simply stored as a list of terms
ordered by the monomial order. Here a term is a pair of a multi-exponent and a
coefficient. In general, this coefficient should never be zero (since it can
then be omitted). The zero module element is stored as an empty list.

The main routines are ``sdm_nf_mora`` and ``sdm_groebner`` which can be used
to compute, respectively, weak normal forms and standard bases. They work with
arbitrary (not necessarily global) monomial orders.

In general, product orders have to be used to construct valid monomial orders
for modules. However, ``lex`` can be used as-is.

Note that the "level" (number of variables, i.e. parameter u+1 in
distributedpolys.py) is never needed in this code.

The main reference for this file is [SCA],
"A Singular Introduction to Commutative Algebra".
"""


from itertools import permutations

from sympy.polys.monomials import (
    monomial_mul, monomial_lcm, monomial_div, monomial_deg
)

from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify

# Additional monomial tools.


def sdm_monomial_mul(M, X):
    """
    Multiply tuple ``X`` representing a monomial of `K[X]` into the tuple
    ``M`` representing a monomial of `F`.

    Examples
    ========

    Multiplying `xy^3` into `x f_1` yields `x^2 y^3 f_1`:

    >>> from sympy.polys.distributedmodules import sdm_monomial_mul
    >>> sdm_monomial_mul((1, 1, 0), (1, 3))
    (1, 2, 3)
    """
    return (M[0],) + monomial_mul(X, M[1:])


def sdm_monomial_deg(M):
    """
    Return the total degree of ``M``.

    Examples
    ========

    For example, the total degree of `x^2 y f_5` is 3:

    >>> from sympy.polys.distributedmodules import sdm_monomial_deg
    >>> sdm_monomial_deg((5, 2, 1))
    3
    """
    return monomial_deg(M[1:])


def sdm_monomial_lcm(A, B):
    r"""
    Return the "least common multiple" of ``A`` and ``B``.

    IF `A = M e_j` and `B = N e_j`, where `M` and `N` are polynomial monomials,
    this returns `\lcm(M, N) e_j`. Note that ``A`` and ``B`` involve distinct
    monomials.

    Otherwise the result is undefined.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_monomial_lcm
    >>> sdm_monomial_lcm((1, 2, 3), (1, 0, 5))
    (1, 2, 5)
    """
    return (A[0],) + monomial_lcm(A[1:], B[1:])


def sdm_monomial_divides(A, B):
    """
    Does there exist a (polynomial) monomial X such that XA = B?

    Examples
    ========

    Positive examples:

    In the following examples, the monomial is given in terms of x, y and the
    generator(s), f_1, f_2 etc. The tuple form of that monomial is used in
    the call to sdm_monomial_divides.
    Note: the generator appears last in the expression but first in the tuple
    and other factors appear in the same order that they appear in the monomial
    expression.

    `A = f_1` divides `B = f_1`

    >>> from sympy.polys.distributedmodules import sdm_monomial_divides
    >>> sdm_monomial_divides((1, 0, 0), (1, 0, 0))
    True

    `A = f_1` divides `B = x^2 y f_1`

    >>> sdm_monomial_divides((1, 0, 0), (1, 2, 1))
    True

    `A = xy f_5` divides `B = x^2 y f_5`

    >>> sdm_monomial_divides((5, 1, 1), (5, 2, 1))
    True

    Negative examples:

    `A = f_1` does not divide `B = f_2`

    >>> sdm_monomial_divides((1, 0, 0), (2, 0, 0))
    False

    `A = x f_1` does not divide `B = f_1`

    >>> sdm_monomial_divides((1, 1, 0), (1, 0, 0))
    False

    `A = xy^2 f_5` does not divide `B = y f_5`

    >>> sdm_monomial_divides((5, 1, 2), (5, 0, 1))
    False
    """
    return A[0] == B[0] and all(a <= b for a, b in zip(A[1:], B[1:]))


# The actual distributed modules code.

def sdm_LC(f, K):
    """Returns the leading coefficient of ``f``. """
    if not f:
        return K.zero
    else:
        return f[0][1]


def sdm_to_dict(f):
    """Make a dictionary from a distributed polynomial. """
    return dict(f)


def sdm_from_dict(d, O):
    """
    Create an sdm from a dictionary.

    Here ``O`` is the monomial order to use.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_from_dict
    >>> from sympy.polys import QQ, lex
    >>> dic = {(1, 1, 0): QQ(1), (1, 0, 0): QQ(2), (0, 1, 0): QQ(0)}
    >>> sdm_from_dict(dic, lex)
    [((1, 1, 0), 1), ((1, 0, 0), 2)]
    """
    return sdm_strip(sdm_sort(list(d.items()), O))


def sdm_sort(f, O):
    """Sort terms in ``f`` using the given monomial order ``O``. """
    return sorted(f, key=lambda term: O(term[0]), reverse=True)


def sdm_strip(f):
    """Remove terms with zero coefficients from ``f`` in ``K[X]``. """
    return [ (monom, coeff) for monom, coeff in f if coeff ]


def sdm_add(f, g, O, K):
    """
    Add two module elements ``f``, ``g``.

    Addition is done over the ground field ``K``, monomials are ordered
    according to ``O``.

    Examples
    ========

    All examples use lexicographic order.

    `(xy f_1) + (f_2) = f_2 + xy f_1`

    >>> from sympy.polys.distributedmodules import sdm_add
    >>> from sympy.polys import lex, QQ
    >>> sdm_add([((1, 1, 1), QQ(1))], [((2, 0, 0), QQ(1))], lex, QQ)
    [((2, 0, 0), 1), ((1, 1, 1), 1)]

    `(xy f_1) + (-xy f_1)` = 0`

    >>> sdm_add([((1, 1, 1), QQ(1))], [((1, 1, 1), QQ(-1))], lex, QQ)
    []

    `(f_1) + (2f_1) = 3f_1`

    >>> sdm_add([((1, 0, 0), QQ(1))], [((1, 0, 0), QQ(2))], lex, QQ)
    [((1, 0, 0), 3)]

    `(yf_1) + (xf_1) = xf_1 + yf_1`

    >>> sdm_add([((1, 0, 1), QQ(1))], [((1, 1, 0), QQ(1))], lex, QQ)
    [((1, 1, 0), 1), ((1, 0, 1), 1)]
    """
    h = dict(f)

    for monom, c in g:
        if monom in h:
            coeff = h[monom] + c

            if not coeff:
                del h[monom]
            else:
                h[monom] = coeff
        else:
            h[monom] = c

    return sdm_from_dict(h, O)


def sdm_LM(f):
    r"""
    Returns the leading monomial of ``f``.

    Only valid if `f \ne 0`.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_LM, sdm_from_dict
    >>> from sympy.polys import QQ, lex
    >>> dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(1), (4, 0, 1): QQ(1)}
    >>> sdm_LM(sdm_from_dict(dic, lex))
    (4, 0, 1)
    """
    return f[0][0]


def sdm_LT(f):
    r"""
    Returns the leading term of ``f``.

    Only valid if `f \ne 0`.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_LT, sdm_from_dict
    >>> from sympy.polys import QQ, lex
    >>> dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(2), (4, 0, 1): QQ(3)}
    >>> sdm_LT(sdm_from_dict(dic, lex))
    ((4, 0, 1), 3)
    """
    return f[0]


def sdm_mul_term(f, term, O, K):
    """
    Multiply a distributed module element ``f`` by a (polynomial) term ``term``.

    Multiplication of coefficients is done over the ground field ``K``, and
    monomials are ordered according to ``O``.

    Examples
    ========

    `0 f_1 = 0`

    >>> from sympy.polys.distributedmodules import sdm_mul_term
    >>> from sympy.polys import lex, QQ
    >>> sdm_mul_term([((1, 0, 0), QQ(1))], ((0, 0), QQ(0)), lex, QQ)
    []

    `x 0 = 0`

    >>> sdm_mul_term([], ((1, 0), QQ(1)), lex, QQ)
    []

    `(x) (f_1) = xf_1`

    >>> sdm_mul_term([((1, 0, 0), QQ(1))], ((1, 0), QQ(1)), lex, QQ)
    [((1, 1, 0), 1)]

    `(2xy) (3x f_1 + 4y f_2) = 8xy^2 f_2 + 6x^2y f_1`

    >>> f = [((2, 0, 1), QQ(4)), ((1, 1, 0), QQ(3))]
    >>> sdm_mul_term(f, ((1, 1), QQ(2)), lex, QQ)
    [((2, 1, 2), 8), ((1, 2, 1), 6)]
    """
    X, c = term

    if not f or not c:
        return []
    else:
        if K.is_one(c):
            return [ (sdm_monomial_mul(f_M, X), f_c) for f_M, f_c in f ]
        else:
            return [ (sdm_monomial_mul(f_M, X), f_c * c) for f_M, f_c in f ]


def sdm_zero():
    """Return the zero module element."""
    return []


def sdm_deg(f):
    """
    Degree of ``f``.

    This is the maximum of the degrees of all its monomials.
    Invalid if ``f`` is zero.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_deg
    >>> sdm_deg([((1, 2, 3), 1), ((10, 0, 1), 1), ((2, 3, 4), 4)])
    7
    """
    return max(sdm_monomial_deg(M[0]) for M in f)


# Conversion

def sdm_from_vector(vec, O, K, **opts):
    """
    Create an sdm from an iterable of expressions.

    Coefficients are created in the ground field ``K``, and terms are ordered
    according to monomial order ``O``. Named arguments are passed on to the
    polys conversion code and can be used to specify for example generators.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_from_vector
    >>> from sympy.abc import x, y, z
    >>> from sympy.polys import QQ, lex
    >>> sdm_from_vector([x**2+y**2, 2*z], lex, QQ)
    [((1, 0, 0, 1), 2), ((0, 2, 0, 0), 1), ((0, 0, 2, 0), 1)]
    """
    dics, gens = parallel_dict_from_expr(sympify(vec), **opts)
    dic = {}
    for i, d in enumerate(dics):
        for k, v in d.items():
            dic[(i,) + k] = K.convert(v)
    return sdm_from_dict(dic, O)


def sdm_to_vector(f, gens, K, n=None):
    """
    Convert sdm ``f`` into a list of polynomial expressions.

    The generators for the polynomial ring are specified via ``gens``. The rank
    of the module is guessed, or passed via ``n``. The ground field is assumed
    to be ``K``.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_to_vector
    >>> from sympy.abc import x, y, z
    >>> from sympy.polys import QQ
    >>> f = [((1, 0, 0, 1), QQ(2)), ((0, 2, 0, 0), QQ(1)), ((0, 0, 2, 0), QQ(1))]
    >>> sdm_to_vector(f, [x, y, z], QQ)
    [x**2 + y**2, 2*z]
    """
    dic = sdm_to_dict(f)
    dics = {}
    for k, v in dic.items():
        dics.setdefault(k[0], []).append((k[1:], v))
    n = n or len(dics)
    res = []
    for k in range(n):
        if k in dics:
            res.append(Poly(dict(dics[k]), gens=gens, domain=K).as_expr())
        else:
            res.append(S.Zero)
    return res

# Algorithms.


def sdm_spoly(f, g, O, K, phantom=None):
    """
    Compute the generalized s-polynomial of ``f`` and ``g``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    This is invalid if either of ``f`` or ``g`` is zero.

    If the leading terms of `f` and `g` involve different basis elements of
    `F`, their s-poly is defined to be zero. Otherwise it is a certain linear
    combination of `f` and `g` in which the leading terms cancel.
    See [SCA, defn 2.3.6] for details.

    If ``phantom`` is not ``None``, it should be a pair of module elements on
    which to perform the same operation(s) as on ``f`` and ``g``. The in this
    case both results are returned.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_spoly
    >>> from sympy.polys import QQ, lex
    >>> f = [((2, 1, 1), QQ(1)), ((1, 0, 1), QQ(1))]
    >>> g = [((2, 3, 0), QQ(1))]
    >>> h = [((1, 2, 3), QQ(1))]
    >>> sdm_spoly(f, h, lex, QQ)
    []
    >>> sdm_spoly(f, g, lex, QQ)
    [((1, 2, 1), 1)]
    """
    if not f or not g:
        return sdm_zero()
    LM1 = sdm_LM(f)
    LM2 = sdm_LM(g)
    if LM1[0] != LM2[0]:
        return sdm_zero()
    LM1 = LM1[1:]
    LM2 = LM2[1:]
    lcm = monomial_lcm(LM1, LM2)
    m1 = monomial_div(lcm, LM1)
    m2 = monomial_div(lcm, LM2)
    c = K.quo(-sdm_LC(f, K), sdm_LC(g, K))
    r1 = sdm_add(sdm_mul_term(f, (m1, K.one), O, K),
                 sdm_mul_term(g, (m2, c), O, K), O, K)
    if phantom is None:
        return r1
    r2 = sdm_add(sdm_mul_term(phantom[0], (m1, K.one), O, K),
                 sdm_mul_term(phantom[1], (m2, c), O, K), O, K)
    return r1, r2


def sdm_ecart(f):
    """
    Compute the ecart of ``f``.

    This is defined to be the difference of the total degree of `f` and the
    total degree of the leading monomial of `f` [SCA, defn 2.3.7].

    Invalid if f is zero.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_ecart
    >>> sdm_ecart([((1, 2, 3), 1), ((1, 0, 1), 1)])
    0
    >>> sdm_ecart([((2, 2, 1), 1), ((1, 5, 1), 1)])
    3
    """
    return sdm_deg(f) - sdm_monomial_deg(sdm_LM(f))


def sdm_nf_mora(f, G, O, K, phantom=None):
    r"""
    Compute a weak normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    Weak normal forms are defined in [SCA, defn 2.3.3]. They are not unique.
    This function deterministically computes a weak normal form, depending on
    the order of `G`.

    The most important property of a weak normal form is the following: if
    `R` is the ring associated with the monomial ordering (if the ordering is
    global, we just have `R = K[x_1, \ldots, x_n]`, otherwise it is a certain
    localization thereof), `I` any ideal of `R` and `G` a standard basis for
    `I`, then for any `f \in R`, we have `f \in I` if and only if
    `NF(f | G) = 0`.

    This is the generalized Mora algorithm for computing weak normal forms with
    respect to arbitrary monomial orders [SCA, algorithm 2.3.9].

    If ``phantom`` is not ``None``, it should be a pair of "phantom" arguments
    on which to perform the same computations as on ``f``, ``G``, both results
    are then returned.
    """
    from itertools import repeat
    h = f
    T = list(G)
    if phantom is not None:
        # "phantom" variables with suffix p
        hp = phantom[0]
        Tp = list(phantom[1])
        phantom = True
    else:
        Tp = repeat([])
        phantom = False
    while h:
        # TODO better data structure!!!
        Th = [(g, sdm_ecart(g), gp) for g, gp in zip(T, Tp)
              if sdm_monomial_divides(sdm_LM(g), sdm_LM(h))]
        if not Th:
            break
        g, _, gp = min(Th, key=lambda x: x[1])
        if sdm_ecart(g) > sdm_ecart(h):
            T.append(h)
            if phantom:
                Tp.append(hp)
        if phantom:
            h, hp = sdm_spoly(h, g, O, K, phantom=(hp, gp))
        else:
            h = sdm_spoly(h, g, O, K)
    if phantom:
        return h, hp
    return h


def sdm_nf_buchberger(f, G, O, K, phantom=None):
    r"""
    Compute a weak normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    This is the standard Buchberger algorithm for computing weak normal forms with
    respect to *global* monomial orders [SCA, algorithm 1.6.10].

    If ``phantom`` is not ``None``, it should be a pair of "phantom" arguments
    on which to perform the same computations as on ``f``, ``G``, both results
    are then returned.
    """
    from itertools import repeat
    h = f
    T = list(G)
    if phantom is not None:
        # "phantom" variables with suffix p
        hp = phantom[0]
        Tp = list(phantom[1])
        phantom = True
    else:
        Tp = repeat([])
        phantom = False
    while h:
        try:
            g, gp = next((g, gp) for g, gp in zip(T, Tp)
                         if sdm_monomial_divides(sdm_LM(g), sdm_LM(h)))
        except StopIteration:
            break
        if phantom:
            h, hp = sdm_spoly(h, g, O, K, phantom=(hp, gp))
        else:
            h = sdm_spoly(h, g, O, K)
    if phantom:
        return h, hp
    return h


def sdm_nf_buchberger_reduced(f, G, O, K):
    r"""
    Compute a reduced normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    In contrast to weak normal forms, reduced normal forms *are* unique, but
    their computation is more expensive.

    This is the standard Buchberger algorithm for computing reduced normal forms
    with respect to *global* monomial orders [SCA, algorithm 1.6.11].

    The ``pantom`` option is not supported, so this normal form cannot be used
    as a normal form for the "extended" groebner algorithm.
    """
    h = sdm_zero()
    g = f
    while g:
        g = sdm_nf_buchberger(g, G, O, K)
        if g:
            h = sdm_add(h, [sdm_LT(g)], O, K)
            g = g[1:]
    return h


def sdm_groebner(G, NF, O, K, extended=False):
    """
    Compute a minimal standard basis of ``G`` with respect to order ``O``.

    The algorithm uses a normal form ``NF``, for example ``sdm_nf_mora``.
    The ground field is assumed to be ``K``, and monomials ordered according
    to ``O``.

    Let `N` denote the submodule generated by elements of `G`. A standard
    basis for `N` is a subset `S` of `N`, such that `in(S) = in(N)`, where for
    any subset `X` of `F`, `in(X)` denotes the submodule generated by the
    initial forms of elements of `X`. [SCA, defn 2.3.2]

    A standard basis is called minimal if no subset of it is a standard basis.

    One may show that standard bases are always generating sets.

    Minimal standard bases are not unique. This algorithm computes a
    deterministic result, depending on the particular order of `G`.

    If ``extended=True``, also compute the transition matrix from the initial
    generators to the groebner basis. That is, return a list of coefficient
    vectors, expressing the elements of the groebner basis in terms of the
    elements of ``G``.

    This functions implements the "sugar" strategy, see

    Giovini et al: "One sugar cube, please" OR Selection strategies in
    Buchberger algorithm.
    """

    # The critical pair set.
    # A critical pair is stored as (i, j, s, t) where (i, j) defines the pair
    # (by indexing S), s is the sugar of the pair, and t is the lcm of their
    # leading monomials.
    P = []

    # The eventual standard basis.
    S = []
    Sugars = []

    def Ssugar(i, j):
        """Compute the sugar of the S-poly corresponding to (i, j)."""
        LMi = sdm_LM(S[i])
        LMj = sdm_LM(S[j])
        return max(Sugars[i] - sdm_monomial_deg(LMi),
                   Sugars[j] - sdm_monomial_deg(LMj)) \
            + sdm_monomial_deg(sdm_monomial_lcm(LMi, LMj))

    ourkey = lambda p: (p[2], O(p[3]), p[1])

    def update(f, sugar, P):
        """Add f with sugar ``sugar`` to S, update P."""
        if not f:
            return P
        k = len(S)
        S.append(f)
        Sugars.append(sugar)

        LMf = sdm_LM(f)

        def removethis(pair):
            i, j, s, t = pair
            if LMf[0] != t[0]:
                return False
            tik = sdm_monomial_lcm(LMf, sdm_LM(S[i]))
            tjk = sdm_monomial_lcm(LMf, sdm_LM(S[j]))
            return tik != t and tjk != t and sdm_monomial_divides(tik, t) and \
                sdm_monomial_divides(tjk, t)
        # apply the chain criterion
        P = [p for p in P if not removethis(p)]

        # new-pair set
        N = [(i, k, Ssugar(i, k), sdm_monomial_lcm(LMf, sdm_LM(S[i])))
             for i in range(k) if LMf[0] == sdm_LM(S[i])[0]]
        # TODO apply the product criterion?
        N.sort(key=ourkey)
        remove = set()
        for i, p in enumerate(N):
            for j in range(i + 1, len(N)):
                if sdm_monomial_divides(p[3], N[j][3]):
                    remove.add(j)

        # TODO mergesort?
        P.extend(reversed([p for i, p in enumerate(N) if i not in remove]))
        P.sort(key=ourkey, reverse=True)
        # NOTE reverse-sort, because we want to pop from the end
        return P

    # Figure out the number of generators in the ground ring.
    try:
        # NOTE: we look for the first non-zero vector, take its first monomial
        #       the number of generators in the ring is one less than the length
        #       (since the zeroth entry is for the module generators)
        numgens = len(next(x[0] for x in G if x)[0]) - 1
    except StopIteration:
        # No non-zero elements in G ...
        if extended:
            return [], []
        return []

    # This list will store expressions of the elements of S in terms of the
    # initial generators
    coefficients = []

    # First add all the elements of G to S
    for i, f in enumerate(G):
        P = update(f, sdm_deg(f), P)
        if extended and f:
            coefficients.append(sdm_from_dict({(i,) + (0,)*numgens: K(1)}, O))

    # Now carry out the buchberger algorithm.
    while P:
        i, j, s, t = P.pop()
        f, g = S[i], S[j]
        if extended:
            sp, coeff = sdm_spoly(f, g, O, K,
                                  phantom=(coefficients[i], coefficients[j]))
            h, hcoeff = NF(sp, S, O, K, phantom=(coeff, coefficients))
            if h:
                coefficients.append(hcoeff)
        else:
            h = NF(sdm_spoly(f, g, O, K), S, O, K)
        P = update(h, Ssugar(i, j), P)

    # Finally interreduce the standard basis.
    # (TODO again, better data structures)
    S = {(tuple(f), i) for i, f in enumerate(S)}
    for (a, ai), (b, bi) in permutations(S, 2):
        A = sdm_LM(a)
        B = sdm_LM(b)
        if sdm_monomial_divides(A, B) and (b, bi) in S and (a, ai) in S:
            S.remove((b, bi))

    L = sorted(((list(f), i) for f, i in S), key=lambda p: O(sdm_LM(p[0])),
               reverse=True)
    res = [x[0] for x in L]
    if extended:
        return res, [coefficients[i] for _, i in L]
    return res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\mobile_helpers\usability_checker.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import tempfile
from collections import deque
from enum import IntEnum

import onnx

from ..onnx_model_utils import ModelProtoWithShapeInfo, get_producer_consumer_maps, is_fixed_size_tensor, optimize_model


class _SupportedOpsChecker:
    """
    Class to process the md file with list of supported ops and caveats for an execution provider.
    e.g. /tools/ci_build/github/android/nnapi_supported_ops.md
         /tools/ci_build/github/apple/coreml_supported_mlprogram_ops.md
         /tools/ci_build/github/apple/coreml_supported_neuralnetwork_ops.md
    """

    def __init__(self, filename):
        self._filename = filename
        self._ops = {}  # op to caveats
        self._ops_seen = set()

        with open(filename) as f:
            for line in f:
                # we're looking for a markdown table with 2 columns. first is op name. second is caveats
                # op name is domain:op
                if line.startswith("|"):
                    pieces = line.strip().split("|")
                    if len(pieces) == 4:  # pre-first '|'. op, caveat, post-last '|'
                        domain_op = pieces[1]
                        caveat = pieces[2]
                        caveat = caveat.replace("<br/>", " ")  # remove some HTML tags
                        # skip lines that don't have the ':' which separates the domain and op
                        # e.g. the table header will fail this check
                        if ":" in domain_op:
                            self._ops[domain_op] = caveat

    def is_op_supported(self, node):
        domain = node.domain if node.domain else "ai.onnx"
        domain_op = domain + ":" + node.op_type

        is_supported = domain_op in self._ops
        if is_supported:
            self._ops_seen.add(domain_op)

        return is_supported

    def get_caveats(self):
        caveats = []
        for op in sorted(self._ops_seen):
            caveat = self._ops[op]
            if caveat:
                caveats.append(f"{op}:{caveat}")

        return caveats


class PartitioningInfo:
    class TryWithEP(IntEnum):
        NO = (0,)
        MAYBE = (1,)
        YES = 2

    def __init__(
        self,
        num_nodes: int,
        num_supported_nodes: int,
        num_partitions: int,
        supported_ops_checker: _SupportedOpsChecker,
        supported_groups: list[onnx.NodeProto],
        unsupported_ops: set[str],
        nodes_unsupported_due_to_op: int,
        nodes_unsupported_due_to_dynamic_input: int,
        num_unsupported_nodes_due_to_rank: int,
        ops_with_unsupported_rank: set[str],
    ):
        self.num_nodes = num_nodes
        self.num_supported_nodes = num_supported_nodes
        self.num_partitions = num_partitions
        self.supported_ops_checker = supported_ops_checker
        self.supported_groups = supported_groups
        self.unsupported_ops = unsupported_ops
        self.nodes_unsupported_due_to_op = nodes_unsupported_due_to_op
        self.nodes_unsupported_due_to_dynamic_input = nodes_unsupported_due_to_dynamic_input
        self.num_unsupported_nodes_due_to_rank = num_unsupported_nodes_due_to_rank
        self.ops_with_unsupported_rank = ops_with_unsupported_rank

        self.num_subgraphs = 0
        self.num_nodes_in_subgraphs = 0

    def merge(self, other: PartitioningInfo):
        """
        Merge the information from another PartitioningInfo instance into this one.
        """
        self.num_nodes += other.num_nodes
        self.num_supported_nodes += other.num_supported_nodes
        self.num_partitions += other.num_partitions
        self.supported_groups.extend(other.supported_groups)
        self.unsupported_ops.update(other.unsupported_ops)
        self.nodes_unsupported_due_to_op += other.nodes_unsupported_due_to_op
        self.nodes_unsupported_due_to_dynamic_input += other.nodes_unsupported_due_to_dynamic_input
        self.num_unsupported_nodes_due_to_rank += other.num_unsupported_nodes_due_to_rank
        self.ops_with_unsupported_rank.update(other.ops_with_unsupported_rank)

        # hard assumption that we merge into the main graph partitioning info
        self.num_subgraphs += 1
        self.num_nodes_in_subgraphs += other.num_nodes

    def suitability(self):
        # semi-arbitrary choices that err on the side of MAYBE.
        # having 1 partition is always preferred, but if that is small it may not be useful.
        # having 2 partitions may be okay if they cover most nodes
        # more than 2 partitions and the device copy cost is almost guaranteed to outweigh the benefit of using the NPU
        # NOTE: This assumes the EP is not CPU based and there is device copy overhead to consider
        pct_supported = self.num_supported_nodes / self.num_nodes * 100
        if self.num_partitions == 1:
            if pct_supported > 75:
                return PartitioningInfo.TryWithEP.YES
            elif pct_supported > 50:
                return PartitioningInfo.TryWithEP.MAYBE
            else:
                return PartitioningInfo.TryWithEP.NO

        if self.num_partitions == 2:
            if pct_supported > 75:
                return PartitioningInfo.TryWithEP.MAYBE
            else:
                return PartitioningInfo.TryWithEP.NO

        return PartitioningInfo.TryWithEP.NO

    def print_analysis(self, logger: logging.Logger, ep_name: str):
        """
        Analyze the partitioning information and log the analysis
        :param logger: Logger to use
        :param ep_name: Execution provider name to use in the log messages
        """

        logger.info(
            f"{self.num_partitions} partitions with a total of {self.num_supported_nodes}/{self.num_nodes} "
            f"nodes can be handled by the {ep_name} EP."
        )

        if self.supported_groups:
            logger.info(
                f"\tPartition sizes: [{', '.join([str(len(partition)) for partition in self.supported_groups])}]"
            )

            # dump full groups if debug output is enabled
            for group in self.supported_groups:
                logger.debug(f"Nodes in group: {','.join([f'{node.op_type}:{node.name}' for node in group])}")

        logger.info(f"Unsupported nodes due to operator={self.nodes_unsupported_due_to_op}")
        if self.unsupported_ops:
            logger.info(f"\tUnsupported ops: {','.join(sorted(self.unsupported_ops))}")

        caveats = self.supported_ops_checker.get_caveats()
        if caveats:
            indent = " " * 5
            logger.info(
                "\tCaveats that have not been checked and may result in a node not actually being supported:  "
                f"{''.join([os.linesep + indent + caveat for caveat in caveats])}"
            )

        if self.nodes_unsupported_due_to_dynamic_input:
            logger.info(
                "Unsupported nodes due to input having a dynamic shape=%d",
                self.nodes_unsupported_due_to_dynamic_input,
            )

        if self.num_unsupported_nodes_due_to_rank:
            logger.info(f"Unsupported nodes due to rank of input data={self.num_unsupported_nodes_due_to_rank}")
            logger.info(f"\tOps with unsupported rank: {','.join(sorted(self.ops_with_unsupported_rank))}")

        if self.num_subgraphs > 0:
            # TODO: CoreML has a flag. NNAPI doesn't. Either should be able to support a subgraph when treated as a
            # separate graph (only extra detail would be making sure implicit inputs are handled).
            # Merging the subgraph into the parent graph would be more complex.
            #   e.g. for CoreML we could potentially convert Loop to while_loop and If to cond if the subgraphs in the
            #        control flow node are fully supported.
            #        NNAPI also has While and If.

            # It most likely will be necessary to support merging in If nodes with fully supported subgraphs,
            # as the subgraphs in those are often very simple, so the performance cost of going to the CPU EP and back
            # is high.
            logger.info(
                f"{self.num_nodes_in_subgraphs} nodes are in {self.num_subgraphs} subgraphs. "
                "Check EP as to whether subgraphs are supported."
            )

        pct_nodes_using_ep = self.num_supported_nodes / self.num_nodes * 100
        if self.num_partitions == 0:
            logger.info(f"{ep_name} cannot run any nodes in this model.")
        elif self.num_partitions == 1:
            if pct_nodes_using_ep > 75:
                logger.info(
                    f"{ep_name} should work well for this model as there is one partition "
                    f"covering {pct_nodes_using_ep:.1f}% of the nodes in the model."
                )
            elif pct_nodes_using_ep > 50:
                logger.info(
                    f"{ep_name} may work well for this model, however only {pct_nodes_using_ep:.1f}% of nodes "
                    "will use it. Performance testing is required to validate."
                )
            else:
                logger.info(
                    f"{ep_name} will probably not work will for this model as only {pct_nodes_using_ep:.2f}% "
                    "of nodes will use it."
                )

        elif self.num_partitions == 2 and pct_nodes_using_ep > 75:
            logger.info(
                f"{ep_name} can be considered for this model as there are two partitions "
                f"covering {pct_nodes_using_ep:.1f}% of the nodes. "
                "Performance testing is required to validate."
            )
        else:
            logger.info(
                f"{ep_name} is not recommended with this model as there are {self.num_partitions} partitions "
                f"covering {pct_nodes_using_ep:.1f}% of the nodes in the model. "
                "This will most likely result in worse performance than just using the CPU EP."
            )


def _check_partitioning_for_graph(
    graph: onnx.GraphProto,
    node_to_producers: dict[onnx.NodeProto, set[onnx.NodeProto]],
    node_to_consumers: dict[onnx.NodeProto, set[onnx.NodeProto]],
    supported_ops_checker: _SupportedOpsChecker,
    outer_scope_initializers: set[str],
    require_fixed_input_sizes: bool,
    value_info: dict[str, onnx.ValueInfoProto],
    max_rank: int = 999,  # max rank if EP has a limitation
):
    # initializers have fixed sizes.
    initializers = [i.name for i in graph.initializer]

    def _is_fixed_shape_value(value):
        if value in value_info:
            return is_fixed_size_tensor(value_info[value])

        if value in initializers or value in outer_scope_initializers:
            return True

        # if something has an unknown shape (e.g. something downstream of a Reshape with dynamic input for the shape)
        # it won't have an entry in value_info
        return False

    #
    # Replicate logic from /onnxruntime/core/providers/partitioning_utils.cc:CreateSupportedPartitionNodeGroups
    # to roughly estimate number of partitions for nodes that is_node_supported_fn returns true for.
    #
    # We keep the structure and variable names as close as possible to the C++ implementation to simplify keeping them
    # in sync if future updates are needed.
    #
    # NOTE: CreateSupportedPartitionNodeGroups was recently updated to be QDQ aware so that partitions did not split
    # QDQ node groups. This code does not need to be QDQ aware as splitting a QDQ node group does not affect the total
    # number of partitions or supported nodes.
    #

    # we don't currently support a callback for additional group closure checks in the python implementation
    on_group_closed_fn = None

    supported_groups = []
    # number of inputs from unprocessed nodes (in-degree) per node
    in_degree = {}
    # nodes that are ready to process
    nodes_to_process = deque()  # deque of Node instances
    # nodes that will be processed when considering the next partition node group
    nodes_to_process_with_next_group = deque()

    # initialize in-degrees and find root nodes
    for node in graph.node:
        node_input_edge_count = len(node_to_producers[node]) if node in node_to_producers else 0
        in_degree[node] = node_input_edge_count
        if node_input_edge_count == 0:
            # node is only dependent on graph input or initializers
            nodes_to_process.append(node)

    supported_group = []
    # the partition node group's border is the aggregate of its nodes' output nodes
    supported_group_border = set()
    num_supported_nodes = 0
    num_unsupported_nodes_due_to_op = 0
    num_unsupported_nodes_due_to_dynamic_input = 0
    num_unsupported_nodes_due_to_rank = 0
    unsupported_ops = set()
    ops_with_unsupported_rank = set()

    def close_group():
        if supported_group:
            keep_partition = not on_group_closed_fn or on_group_closed_fn(supported_group)

            if keep_partition:
                supported_groups.append(supported_group.copy())

            supported_group.clear()
            supported_group_border.clear()

    while nodes_to_process or nodes_to_process_with_next_group:
        if not nodes_to_process:
            close_group()
            nodes_to_process = nodes_to_process_with_next_group
            nodes_to_process_with_next_group = deque()
            continue

        node = nodes_to_process.popleft()

        is_op_supported = supported_ops_checker.is_op_supported(node)
        is_input_shape_supported = not require_fixed_input_sizes or all(_is_fixed_shape_value(i) for i in node.input)

        is_rank_supported = True
        if value_info:
            for node_input in node.input:
                if node_input and node_input in value_info and value_info[node_input].type.HasField("tensor_type"):
                    input_rank = len(value_info[node_input].type.tensor_type.shape.dim)
                    if input_rank > max_rank:
                        is_rank_supported = False
                        break

        # special-case if we can infer the rank from the length of the 'perms' Transpose attribute
        # e.g. this works with SegmentAnything where dynamic Reshape operators result in no shape info.
        if node.op_type == "Transpose" and len(node.attribute[0].ints) > max_rank:
            is_rank_supported = False

        is_node_supported = is_op_supported and is_input_shape_supported and is_rank_supported

        if not is_node_supported:
            if node in supported_group_border:
                # an unsupported node on the border will be processed after the current partition node group
                # so skip any additional processing/counting here
                nodes_to_process_with_next_group.append(node)
                continue

            if not is_op_supported:
                unsupported_ops.add(f"{node.domain if node.domain else 'ai.onnx'}:{node.op_type}")
                num_unsupported_nodes_due_to_op += 1

            if not is_input_shape_supported:
                num_unsupported_nodes_due_to_dynamic_input += 1

            if not is_rank_supported:
                num_unsupported_nodes_due_to_rank += 1
                ops_with_unsupported_rank.add(f"{node.domain if node.domain else 'ai.onnx'}:{node.op_type}")

        if is_node_supported:
            num_supported_nodes += 1

            # add node to the partition node group
            supported_group.append(node)

            # remove node from the border and add its outputs to the border
            if node in supported_group_border:
                supported_group_border.remove(node)

            # for each consumer node add to supported_group_border
            if node in node_to_consumers:
                for consumer in node_to_consumers[node]:
                    supported_group_border.add(consumer)

        # adjust in-degrees of the node outputs and add any new nodes to process
        if node in node_to_consumers:
            for consumer in node_to_consumers[node]:
                consumer_node_in_degree = in_degree[consumer]
                consumer_node_in_degree -= 1
                if consumer_node_in_degree == 0:
                    nodes_to_process.append(consumer)

                in_degree[consumer] = consumer_node_in_degree

    close_group()

    num_nodes = len(graph.node)
    num_partitions = len(supported_groups)

    info = PartitioningInfo(
        num_nodes,
        num_supported_nodes,
        num_partitions,
        supported_ops_checker,
        supported_groups,
        unsupported_ops,
        num_unsupported_nodes_due_to_op,
        num_unsupported_nodes_due_to_dynamic_input,
        num_unsupported_nodes_due_to_rank,
        ops_with_unsupported_rank,
    )

    return info


def check_partitioning(
    main_graph: onnx.GraphProto,
    supported_ops_checker: _SupportedOpsChecker,
    require_fixed_input_sizes: bool,
    max_rank: int = 999,
) -> PartitioningInfo:
    """
    Estimate the partitions the graph will be split into for nodes that is_node_supported_fn returns true for.

    The check on whether a node is supported is purely based on the operator type. Additional limitations
    (e.g. NNAPI EP only supports 2D Conv) are not checked, so partitions may not be 100% accurate. The limitations
    for operators in the partitions are printed so the user can manually check.
    :param main_graph: Graph to process
    :param supported_ops_checker: Checker with info on supported ops.
    :param require_fixed_input_sizes: If True, require that the inputs to a potentially supported node are fixed size
                                      tensors for it to be considered as supported. This requires
                                      onnx.shape_inference.infer_shapes to have been run on the model to populate the
                                      shape information.
                                      If False, shapes are ignored during the check.
    :param max_rank: Set if EP has a limitation on the rank of tensors it supports.
    :return PartitioningInfo instance with details
    """

    if require_fixed_input_sizes and len(main_graph.value_info) == 0 and len(main_graph.node) > 1:
        raise ValueError("Run onnx.shape_inference.infer_shapes on the model to populate the shape information.")

    # create lookup map from ValueInfo for efficiency
    def _update_value_info(graph: onnx.GraphProto, value_to_shape: dict[str, onnx.ValueInfoProto]):
        for v in graph.input:
            value_to_shape[v.name] = v
        for v in graph.output:
            value_to_shape[v.name] = v
        for v in graph.value_info:
            value_to_shape[v.name] = v

    # the producer/consumer maps are for the entire model
    node_to_producers, node_to_consumers = get_producer_consumer_maps(main_graph)

    def _check_graph(
        graph: onnx.GraphProto,
        outer_scope_value_info: dict[str, onnx.ValueInfoProto] | None,
        outer_scope_initializers: set[str] | None = None,
        partitioning_info: PartitioningInfo | None = None,
    ) -> PartitioningInfo:
        if outer_scope_value_info is not None:
            # extend value info if we're using it. we replace any value shadowed with a local one
            value_info = outer_scope_value_info.copy()
            _update_value_info(graph, value_info)
        else:
            value_info = {}

        if outer_scope_initializers is None:
            outer_scope_initializers = set()

        info = _check_partitioning_for_graph(
            graph,
            node_to_producers,
            node_to_consumers,
            supported_ops_checker,
            outer_scope_initializers,
            require_fixed_input_sizes,
            value_info,
            max_rank,
        )

        if partitioning_info:
            # merge in subgraph info
            partitioning_info.merge(info)
        else:
            # main graph info
            partitioning_info = info

        # setup outer scope initializers. we copy the input set as a model may have multiple subgraphs
        # on multiple levels, so we need to keep the set for each descent separate
        subgraph_outer_scope_initializers = set(outer_scope_initializers)
        for initializer in graph.initializer:
            subgraph_outer_scope_initializers.add(initializer.name)

        for node in graph.node:
            # recurse into nodes with subgraphs
            for attr in node.attribute:
                if attr.HasField("g"):
                    subgraph = attr.g
                    partitioning_info = _check_graph(
                        subgraph, value_info, subgraph_outer_scope_initializers, partitioning_info
                    )

        return partitioning_info

    aggregated_partitioning_info = _check_graph(main_graph, {} if require_fixed_input_sizes else None)

    return aggregated_partitioning_info


def _check_ep_partitioning(
    model: onnx.ModelProto, supported_ops_config: pathlib.Path, require_fixed_input_sizes: bool, max_rank: int = 999
):
    supported_ops = _SupportedOpsChecker(supported_ops_config)
    partition_info = check_partitioning(model.graph, supported_ops, require_fixed_input_sizes, max_rank)
    return partition_info


def check_nnapi_partitions(model, require_fixed_input_sizes: bool):
    # if we're running in the ORT python package the file should be local. otherwise assume we're running from the
    # ORT repo
    script_dir = pathlib.Path(__file__).parent
    local_config = script_dir / "nnapi_supported_ops.md"
    if local_config.exists():
        config_path = local_config
    else:
        ort_root = script_dir.parents[3]
        config_path = ort_root / "tools" / "ci_build" / "github" / "android" / "nnapi_supported_ops.md"

    return _check_ep_partitioning(model, config_path, require_fixed_input_sizes)


def check_coreml_partitions(model: onnx.ModelProto, require_fixed_input_sizes: bool, config_filename: str):
    # if we're running in the ORT python package the file should be local. otherwise assume we're running from the
    # ORT repo
    script_dir = pathlib.Path(__file__).parent
    local_config = script_dir / config_filename
    if local_config.exists():
        config_path = local_config
    else:
        ort_root = script_dir.parents[3]
        config_path = ort_root / "tools" / "ci_build" / "github" / "apple" / config_filename

    max_rank = 5
    return _check_ep_partitioning(model, config_path, require_fixed_input_sizes, max_rank)


def check_shapes(graph: onnx.GraphProto, logger: logging.Logger | None = None):
    """
    Check the shapes of graph inputs, values and graph outputs to determine if they have static or dynamic sizes.
    NNAPI does not support dynamically sized values. CoreML does, but it will most likely cost performance.
    :param graph: Graph to check. If shape inferencing has been run the checks on values will be meaningful.
    :param logger: Optional logger for diagnostic information.
    :return: Tuple of List of inputs with dynamic shapes, Number of dynamic values found
    """

    # it's OK if the input is dynamically sized and we do a Resize early to a fixed size.
    # it's not good if lots of ops have dynamic inputs

    num_fixed_values = 0
    num_dynamic_values = 0

    dynamic_inputs = []
    for i in graph.input:
        if not is_fixed_size_tensor(i):
            dynamic_inputs.append(i)
            # split/join to remove repeated whitespace and newlines from str(i)
            if logger:
                logger.info(f"Input is not a fixed size tensor: {' '.join(str(i).split())}")
            num_dynamic_values += 1
        else:
            num_fixed_values += 1

    dynamic_outputs = []
    for o in graph.output:
        if not is_fixed_size_tensor(o):
            dynamic_outputs.append(o)
            if logger:
                logger.info(f"Output is not a fixed size tensor: {' '.join(str(o).split())}")
            num_dynamic_values += 1
        else:
            num_fixed_values += 1

    # check we have value info.
    # special case some test graphs with a single node which only have graph input and output values, and
    # a model where all inputs are dynamic (results in no value_info)
    if not graph.value_info and not (len(graph.node) == 1 or len(dynamic_inputs) == len(graph.input)):
        logger.warning(
            "Unable to check shapes within model. ONNX shape inferencing should be run on the model prior to checking."
        )

    for vi in graph.value_info:
        if is_fixed_size_tensor(vi):
            num_fixed_values += 1
        else:
            num_dynamic_values += 1

    if logger:
        logger.info(
            f"Num values with fixed shape={num_fixed_values}. Num values with dynamic shape={num_dynamic_values}"
        )

        if dynamic_inputs:
            if dynamic_outputs:
                logger.info(
                    "Model has dynamic inputs and outputs. Consider re-exporting model with fixed sizes "
                    "if NNAPI or CoreML can be used with this model."
                )
            else:
                logger.info(
                    """Model has dynamically sized inputs but fixed sized outputs.
                       If the sizes become fixed early in the model (e.g. pre-processing of a dynamic input size
                       results in a fixed input size for the majority of the model) performance with NNAPI and CoreML,
                       if applicable, should not be significantly impacted."""
                )

    return dynamic_inputs, num_dynamic_values


def checker(model_path: pathlib.Path, logger: logging.Logger):
    model_with_shape_info_wrapper = ModelProtoWithShapeInfo(model_path)
    model_with_shape_info = model_with_shape_info_wrapper.model_with_shape_info

    dynamic_inputs, num_dynamic_values = check_shapes(model_with_shape_info.graph)

    def check_ep(ep_name, checker_func):
        logger.info(f"Checking {ep_name}")

        # check with shape info first so supported nodes takes into account values with dynamic shapes
        require_fixed_input_sizes = True
        partition_info = checker_func(model_with_shape_info, require_fixed_input_sizes)
        if logger.getEffectiveLevel() <= logging.INFO:
            partition_info.print_analysis(logger, ep_name)

        suitability = partition_info.suitability()
        logger.info(f"Model should perform well with {ep_name} as is: {suitability.name}")

        if suitability != PartitioningInfo.TryWithEP.YES and dynamic_inputs:
            logger.info("--------")
            logger.info("Checking if model will perform better if the dynamic shapes are fixed...")
            require_fixed_input_sizes = False
            partition_info_with_fixed_shapes = checker_func(model_with_shape_info, require_fixed_input_sizes)

            if logger.getEffectiveLevel() <= logging.INFO:
                # analyze and log detailed info
                logger.info("Partition information if the model was updated to make the shapes fixed:")
                partition_info_with_fixed_shapes.print_analysis(logger, ep_name)

            fixed_shape_suitability = partition_info_with_fixed_shapes.suitability()
            logger.info(
                f"Model should perform well with {ep_name} if modified to have fixed input shapes: "
                f"{fixed_shape_suitability.name}"
            )

            if fixed_shape_suitability != PartitioningInfo.TryWithEP.NO:
                logger.info("Shapes can be altered using python -m onnxruntime.tools.make_dynamic_shape_fixed")

            if fixed_shape_suitability.value > suitability.value:
                suitability = fixed_shape_suitability

        logger.info("================")
        logger.info("")

        return suitability

    nnapi_suitability = check_ep("NNAPI", check_nnapi_partitions)

    # Check for NeuralNetwork CoreML model
    def check_nn_coreml(model: onnx.ModelProto, require_fixed_input_sizes):
        return check_coreml_partitions(model, require_fixed_input_sizes, "coreml_supported_neuralnetwork_ops.md")

    # Check for MLProgram CoreML model
    def check_mlprogram_coreml(model: onnx.ModelProto, require_fixed_input_sizes):
        return check_coreml_partitions(model, require_fixed_input_sizes, "coreml_supported_mlprogram_ops.md")

    coreml_nn_suitability = check_ep("CoreML NeuralNetwork", check_nn_coreml)
    coreml_mlprogram_suitability = check_ep("CoreML MLProgram", check_mlprogram_coreml)

    if (
        nnapi_suitability != PartitioningInfo.TryWithEP.YES
        or coreml_nn_suitability != PartitioningInfo.TryWithEP.YES
        or coreml_mlprogram_suitability != PartitioningInfo.TryWithEP.YES
    ) and logger.getEffectiveLevel() > logging.INFO:
        logger.info("Re-run with log level of INFO for more details on the NNAPI/CoreML issues.")

    return (
        nnapi_suitability != PartitioningInfo.TryWithEP.NO
        or coreml_nn_suitability != PartitioningInfo.TryWithEP.NO
        or coreml_mlprogram_suitability != PartitioningInfo.TryWithEP.NO
    )


def analyze_model(model_path: pathlib.Path, skip_optimize: bool = False, logger: logging.Logger | None = None):
    """
    Analyze the provided model to determine if it's likely to work well with the NNAPI or CoreML Execution Providers
    :param model_path: Model to analyze.
    :param skip_optimize: Skip optimizing to BASIC level before checking. When exporting to ORT format we will do this
                          optimization..
    :param logger: Logger for output
    :return: True if either the NNAPI or CoreML Execution Providers may work well with this model.
    """
    if not logger:
        logger = logging.getLogger("usability_checker")
        logger.setLevel(logging.INFO)

    logger.info(f"Checking {model_path} for usability with ORT Mobile.")

    with tempfile.TemporaryDirectory() as tmp:
        if not skip_optimize:
            tmp_path = pathlib.Path(tmp) / model_path.name
            optimize_model(model_path, tmp_path, use_external_initializers=True)
            model_path = tmp_path

        try_eps = checker(model_path.resolve(strict=True), logger)

    return try_eps


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="""Analyze an ONNX model for usage with the ORT mobile"""
    )

    parser.add_argument("--log_level", choices=["debug", "info"], default="info", help="Logging level")
    parser.add_argument(
        "--skip_optimize",
        action="store_true",
        help="Don't optimize the model to BASIC level prior to analyzing. "
        "Optimization will occur when exporting the model to ORT format, so in general "
        "should not be skipped unless you have a specific reason to do so.",
    )
    parser.add_argument("model_path", type=pathlib.Path, help="Provide path to ONNX model")

    return parser.parse_args()


def run_analyze_model():
    args = parse_args()
    logger = logging.getLogger("default")

    if args.log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif args.log_level == "info":
        logger.setLevel(logging.INFO)
    elif args.log_level == "warning":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)

    model_path = args.model_path.resolve()
    analyze_model(model_path, args.skip_optimize, logger)


if __name__ == "__main__":
    run_analyze_model()