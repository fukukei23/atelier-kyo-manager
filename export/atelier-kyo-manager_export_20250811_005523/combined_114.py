
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\julia.py ===
"""
Julia code printer

The `JuliaCodePrinter` converts SymPy expressions into Julia expressions.

A complete code generator, which uses `julia_code` extensively, can be found
in `sympy.utilities.codegen`.  The `codegen` module can be used to generate
complete source code files.

"""

from __future__ import annotations
from typing import Any

from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search

# List of known functions.  First, those that have the same name in
# SymPy and Julia. This is almost certainly incomplete!
known_fcns_src1 = ["sin", "cos", "tan", "cot", "sec", "csc",
                   "asin", "acos", "atan", "acot", "asec", "acsc",
                   "sinh", "cosh", "tanh", "coth", "sech", "csch",
                   "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
                   "atan2", "sign", "floor", "log", "exp",
                   "cbrt", "sqrt", "erf", "erfc", "erfi",
                   "factorial", "gamma", "digamma", "trigamma",
                   "polygamma", "beta",
                   "airyai", "airyaiprime", "airybi", "airybiprime",
                   "besselj", "bessely", "besseli", "besselk",
                   "erfinv", "erfcinv"]
# These functions have different names ("SymPy": "Julia"), more
# generally a mapping to (argument_conditions, julia_function).
known_fcns_src2 = {
    "Abs": "abs",
    "ceiling": "ceil",
    "conjugate": "conj",
    "hankel1": "hankelh1",
    "hankel2": "hankelh2",
    "im": "imag",
    "re": "real"
}


class JuliaCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Julia code.
    """
    printmethod = "_julia"
    language = "Julia"

    _operators = {
        'and': '&&',
        'or': '||',
        'not': '!',
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'inline': True,
    })
    # Note: contract is for expressing tensors as loops (if True), or just
    # assignment (if False).  FIXME: this should be looked a more carefully
    # for Julia.

    def __init__(self, settings={}):
        super().__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)


    def _rate_index_position(self, p):
        return p*5


    def _get_statement(self, codestring):
        return "%s" % codestring


    def _get_comment(self, text):
        return "# {}".format(text)


    def _declare_number_const(self, name, value):
        return "const {} = {}".format(name, value)


    def _format_code(self, lines):
        return self.indent_code(lines)


    def _traverse_matrix_indices(self, mat):
        # Julia uses Fortran order (column-major)
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))


    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # Julia arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("for %s = %s:%s" % (var, start, stop))
            close_lines.append("end")
        return open_lines, close_lines


    def _print_Mul(self, expr):
        # print complex numbers nicely in Julia
        if (expr.is_number and expr.is_imaginary and
                expr.as_coeff_Mul()[0].is_integer):
            return "%sim" % self._print(-S.ImaginaryUnit*expr)

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity and item.p == 1:
                # Save the Rational type in julia Unless the numerator is 1.
                # For example:
                # julia_code(Rational(3, 7)*x) --> (3 // 7) * x
                # julia_code(x/3) --> x / 3 but not x * (1 // 3)
                b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # from here it differs from str.py to deal with "*" and ".*"
        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = "%s %s %s" % (r, mulsym, a_str[i])
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return "%s %s %s" % (sign+multjoin(a, a_str), divsym, b_str[0])
        else:
            divsym = '/' if all(bi.is_number for bi in b) else './'
            return "%s %s (%s)" % (sign + multjoin(a, a_str), divsym, multjoin(b, b_str))

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Pow(self, expr):
        powsymbol = '^' if all(x.is_number for x in expr.args) else '.^'

        PREC = precedence(expr)

        if equal_valued(expr.exp, 0.5):
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if equal_valued(expr.exp, -0.5):
                sym = '/' if expr.base.is_number else './'
                return "1 %s sqrt(%s)" % (sym, self._print(expr.base))
            if equal_valued(expr.exp, -1):
                sym = '/' if expr.base.is_number else './'
                return  "1 %s %s" % (sym, self.parenthesize(expr.base, PREC))

        return '%s %s %s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))


    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s ^ %s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))


    def _print_Pi(self, expr):
        if self._settings["inline"]:
            return "pi"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_ImaginaryUnit(self, expr):
        return "im"


    def _print_Exp1(self, expr):
        if self._settings["inline"]:
            return "e"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_EulerGamma(self, expr):
        if self._settings["inline"]:
            return "eulergamma"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_Catalan(self, expr):
        if self._settings["inline"]:
            return "catalan"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_GoldenRatio(self, expr):
        if self._settings["inline"]:
            return "golden"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.tensor.indexed import IndexedBase
        # Copied from codeprinter, but remove special MatrixSymbol treatment
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if not self._settings["inline"] and isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))


    def _print_Infinity(self, expr):
        return 'Inf'


    def _print_NegativeInfinity(self, expr):
        return '-Inf'


    def _print_NaN(self, expr):
        return 'NaN'


    def _print_list(self, expr):
        return 'Any[' + ', '.join(self._print(a) for a in expr) + ']'


    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.stringify(expr, ", ")
    _print_Tuple = _print_tuple


    def _print_BooleanTrue(self, expr):
        return "true"


    def _print_BooleanFalse(self, expr):
        return "false"


    def _print_bool(self, expr):
        return str(expr).lower()


    # Could generate quadrature code for definite Integrals?
    #_print_Integral = _print_not_supported


    def _print_MatrixBase(self, A):
        # Handle zero dimensions:
        if S.Zero in A.shape:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            return "[%s]" % A[0, 0]
        elif A.rows == 1:
            return "[%s]" % A.table(self, rowstart='', rowend='', colsep=' ')
        elif A.cols == 1:
            # note .table would unnecessarily equispace the rows
            return "[%s]" % ", ".join([self._print(a) for a in A])
        return "[%s]" % A.table(self, rowstart='', rowend='',
                                rowsep=';\n', colsep=' ')


    def _print_SparseRepMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list()
        # make row vectors of the indices and entries
        I = Matrix([k[0] + 1 for k in L])
        J = Matrix([k[1] + 1 for k in L])
        AIJ = Matrix([k[2] for k in L])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                            self._print(AIJ), A.rows, A.cols)


    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s,%s]' % (expr.i + 1, expr.j + 1)


    def _print_MatrixSlice(self, expr):
        def strslice(x, lim):
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        return (self._print(expr.parent) + '[' +
                strslice(expr.rowslice, expr.parent.shape[0]) + ',' +
                strslice(expr.colslice, expr.parent.shape[1]) + ']')


    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s[%s]" % (self._print(expr.base.label), ",".join(inds))

    def _print_Identity(self, expr):
        return "eye(%s)" % self._print(expr.shape[0])

    def _print_HadamardProduct(self, expr):
        return ' .* '.join([self.parenthesize(arg, precedence(expr))
                          for arg in expr.args])

    def _print_HadamardPower(self, expr):
        PREC = precedence(expr)
        return '.**'.join([
            self.parenthesize(expr.base, PREC),
            self.parenthesize(expr.exp, PREC)
            ])

    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        return "%s // %s" % (expr.p, expr.q)

    # Note: as of 2022, Julia doesn't have spherical Bessel functions
    def _print_jn(self, expr):
        from sympy.functions import sqrt, besselj
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*besselj(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_yn(self, expr):
        from sympy.functions import sqrt, bessely
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*bessely(expr.order + S.Half, x)
        return self._print(expr2)

    def _print_sinc(self, expr):
        # Julia has the normalized sinc function
        return "sinc({})".format(self._print(expr.args[0] / S.Pi))

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
        if self._settings["inline"]:
            # Express each (cond, expr) pair in a nested Horner form:
            #   (condition) .* (expr) + (not cond) .* (<others>)
            # Expressions that result in multiple statements won't work here.
            ecpairs = ["({}) ? ({}) :".format
                       (self._print(c), self._print(e))
                       for e, c in expr.args[:-1]]
            elast = " (%s)" % self._print(expr.args[-1].expr)
            pw = "\n".join(ecpairs) + elast
            # Note: current need these outer brackets for 2*pw.  Would be
            # nicer to teach parenthesize() to do this for us when needed!
            return "(" + pw + ")"
        else:
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s)" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("elseif (%s)" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                if i == len(expr.args) - 1:
                    lines.append("end")
            return "\n".join(lines)

    def _print_MatMul(self, expr):
        c, m = expr.as_coeff_mmul()

        sign = ""
        if c.is_number:
            re, im = c.as_real_imag()
            if im.is_zero and re.is_negative:
                expr = _keep_coeff(-c, m)
                sign = "-"
            elif re.is_zero and im.is_negative:
                expr = _keep_coeff(-c, m)
                sign = "-"

        return sign + ' * '.join(
            (self.parenthesize(arg, precedence(expr)) for arg in expr.args)
        )


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # code mostly copied from ccode
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "    "
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        dec_regex = ('^end$', '^elseif ', '^else$')

        # pre-strip left-space from the code
        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(search(re, line) for re in inc_regex))
                     for line in code ]
        decrease = [ int(any(search(re, line) for re in dec_regex))
                     for line in code ]

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


def julia_code(expr, assign_to=None, **settings):
    r"""Converts `expr` to a string of Julia code.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    Examples
    ========

    >>> from sympy import julia_code, symbols, sin, pi
    >>> x = symbols('x')
    >>> julia_code(sin(x).series(x).removeO())
    'x .^ 5 / 120 - x .^ 3 / 6 + x'

    >>> from sympy import Rational, ceiling
    >>> x, y, tau = symbols("x, y, tau")
    >>> julia_code((2*tau)**Rational(7, 2))
    '8 * sqrt(2) * tau .^ (7 // 2)'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its possible in Julia to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> julia_code(sin(pi*x*y), assign_to="s")
    's = sin(pi * x .* y)'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> julia_code(3*pi*A**3)
    '(3 * pi) * A ^ 3'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify
    scalar symbols, so sometimes the code might have some minor cosmetic
    issues.  For example, suppose x and y are scalars and A is a Matrix, then
    while a human programmer might write "(x^2*y)*A^3", we generate:

    >>> julia_code(x**2*y*A**3)
    '(x .^ 2 .* y) * A ^ 3'

    Matrices are supported using Julia inline notation.  When using
    ``assign_to`` with matrices, the name can be specified either as a string
    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> julia_code(mat, assign_to='A')
    'A = [x .^ 2 sin(x) ceil(x)]'

    ``Piecewise`` expressions are implemented with logical masking by default.
    Alternatively, you can pass "inline=False" to use if-else conditionals.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> julia_code(pw, assign_to=tau)
    'tau = ((x > 0) ? (x + 1) : (x))'

    Note that any expression that can be generated normally can also exist
    inside a Matrix:

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> julia_code(mat, assign_to='A')
    'A = [x .^ 2 ((x > 0) ? (x + 1) : (x)) sin(x)]'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the
    dictionary value can be a list of tuples i.e., [(argument_test,
    cfunction_string)].  This can be used to call a custom Julia function.

    >>> from sympy import Function
    >>> f = Function('f')
    >>> g = Function('g')
    >>> custom_functions = {
    ...   "f": "existing_julia_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> mat = Matrix([[1, x]])
    >>> julia_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    'existing_julia_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])'

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> julia_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i]) ./ (t[i + 1] - t[i])'
    """
    return JuliaCodePrinter(settings).doprint(expr, assign_to)


def print_julia_code(expr, **settings):
    """Prints the Julia representation of the given expression.

    See `julia_code` for the meaning of the optional arguments.
    """
    print(julia_code(expr, **settings))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\ask.py ===
"""Module for querying SymPy objects about assumptions."""

from sympy.assumptions.assume import (global_assumptions, Predicate,
        AppliedPredicate)
from sympy.assumptions.cnf import CNF, EncodedCNF, Literal
from sympy.core import sympify
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.inference import satisfiable
from sympy.utilities.decorator import memoize_property
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning,
                                        ignore_warnings)


# Memoization is necessary for the properties of AssumptionKeys to
# ensure that only one object of Predicate objects are created.
# This is because assumption handlers are registered on those objects.


class AssumptionKeys:
    """
    This class contains all the supported keys by ``ask``.
    It should be accessed via the instance ``sympy.Q``.

    """

    # DO NOT add methods or properties other than predicate keys.
    # SAT solver checks the properties of Q and use them to compute the
    # fact system. Non-predicate attributes will break this.

    @memoize_property
    def hermitian(self):
        from .handlers.sets import HermitianPredicate
        return HermitianPredicate()

    @memoize_property
    def antihermitian(self):
        from .handlers.sets import AntihermitianPredicate
        return AntihermitianPredicate()

    @memoize_property
    def real(self):
        from .handlers.sets import RealPredicate
        return RealPredicate()

    @memoize_property
    def extended_real(self):
        from .handlers.sets import ExtendedRealPredicate
        return ExtendedRealPredicate()

    @memoize_property
    def imaginary(self):
        from .handlers.sets import ImaginaryPredicate
        return ImaginaryPredicate()

    @memoize_property
    def complex(self):
        from .handlers.sets import ComplexPredicate
        return ComplexPredicate()

    @memoize_property
    def algebraic(self):
        from .handlers.sets import AlgebraicPredicate
        return AlgebraicPredicate()

    @memoize_property
    def transcendental(self):
        from .predicates.sets import TranscendentalPredicate
        return TranscendentalPredicate()

    @memoize_property
    def integer(self):
        from .handlers.sets import IntegerPredicate
        return IntegerPredicate()

    @memoize_property
    def noninteger(self):
        from .predicates.sets import NonIntegerPredicate
        return NonIntegerPredicate()

    @memoize_property
    def rational(self):
        from .handlers.sets import RationalPredicate
        return RationalPredicate()

    @memoize_property
    def irrational(self):
        from .handlers.sets import IrrationalPredicate
        return IrrationalPredicate()

    @memoize_property
    def finite(self):
        from .handlers.calculus import FinitePredicate
        return FinitePredicate()

    @memoize_property
    def infinite(self):
        from .handlers.calculus import InfinitePredicate
        return InfinitePredicate()

    @memoize_property
    def positive_infinite(self):
        from .handlers.calculus import PositiveInfinitePredicate
        return PositiveInfinitePredicate()

    @memoize_property
    def negative_infinite(self):
        from .handlers.calculus import NegativeInfinitePredicate
        return NegativeInfinitePredicate()

    @memoize_property
    def positive(self):
        from .handlers.order import PositivePredicate
        return PositivePredicate()

    @memoize_property
    def negative(self):
        from .handlers.order import NegativePredicate
        return NegativePredicate()

    @memoize_property
    def zero(self):
        from .handlers.order import ZeroPredicate
        return ZeroPredicate()

    @memoize_property
    def extended_positive(self):
        from .handlers.order import ExtendedPositivePredicate
        return ExtendedPositivePredicate()

    @memoize_property
    def extended_negative(self):
        from .handlers.order import ExtendedNegativePredicate
        return ExtendedNegativePredicate()

    @memoize_property
    def nonzero(self):
        from .handlers.order import NonZeroPredicate
        return NonZeroPredicate()

    @memoize_property
    def nonpositive(self):
        from .handlers.order import NonPositivePredicate
        return NonPositivePredicate()

    @memoize_property
    def nonnegative(self):
        from .handlers.order import NonNegativePredicate
        return NonNegativePredicate()

    @memoize_property
    def extended_nonzero(self):
        from .handlers.order import ExtendedNonZeroPredicate
        return ExtendedNonZeroPredicate()

    @memoize_property
    def extended_nonpositive(self):
        from .handlers.order import ExtendedNonPositivePredicate
        return ExtendedNonPositivePredicate()

    @memoize_property
    def extended_nonnegative(self):
        from .handlers.order import ExtendedNonNegativePredicate
        return ExtendedNonNegativePredicate()

    @memoize_property
    def even(self):
        from .handlers.ntheory import EvenPredicate
        return EvenPredicate()

    @memoize_property
    def odd(self):
        from .handlers.ntheory import OddPredicate
        return OddPredicate()

    @memoize_property
    def prime(self):
        from .handlers.ntheory import PrimePredicate
        return PrimePredicate()

    @memoize_property
    def composite(self):
        from .handlers.ntheory import CompositePredicate
        return CompositePredicate()

    @memoize_property
    def commutative(self):
        from .handlers.common import CommutativePredicate
        return CommutativePredicate()

    @memoize_property
    def is_true(self):
        from .handlers.common import IsTruePredicate
        return IsTruePredicate()

    @memoize_property
    def symmetric(self):
        from .handlers.matrices import SymmetricPredicate
        return SymmetricPredicate()

    @memoize_property
    def invertible(self):
        from .handlers.matrices import InvertiblePredicate
        return InvertiblePredicate()

    @memoize_property
    def orthogonal(self):
        from .handlers.matrices import OrthogonalPredicate
        return OrthogonalPredicate()

    @memoize_property
    def unitary(self):
        from .handlers.matrices import UnitaryPredicate
        return UnitaryPredicate()

    @memoize_property
    def positive_definite(self):
        from .handlers.matrices import PositiveDefinitePredicate
        return PositiveDefinitePredicate()

    @memoize_property
    def upper_triangular(self):
        from .handlers.matrices import UpperTriangularPredicate
        return UpperTriangularPredicate()

    @memoize_property
    def lower_triangular(self):
        from .handlers.matrices import LowerTriangularPredicate
        return LowerTriangularPredicate()

    @memoize_property
    def diagonal(self):
        from .handlers.matrices import DiagonalPredicate
        return DiagonalPredicate()

    @memoize_property
    def fullrank(self):
        from .handlers.matrices import FullRankPredicate
        return FullRankPredicate()

    @memoize_property
    def square(self):
        from .handlers.matrices import SquarePredicate
        return SquarePredicate()

    @memoize_property
    def integer_elements(self):
        from .handlers.matrices import IntegerElementsPredicate
        return IntegerElementsPredicate()

    @memoize_property
    def real_elements(self):
        from .handlers.matrices import RealElementsPredicate
        return RealElementsPredicate()

    @memoize_property
    def complex_elements(self):
        from .handlers.matrices import ComplexElementsPredicate
        return ComplexElementsPredicate()

    @memoize_property
    def singular(self):
        from .predicates.matrices import SingularPredicate
        return SingularPredicate()

    @memoize_property
    def normal(self):
        from .predicates.matrices import NormalPredicate
        return NormalPredicate()

    @memoize_property
    def triangular(self):
        from .predicates.matrices import TriangularPredicate
        return TriangularPredicate()

    @memoize_property
    def unit_triangular(self):
        from .predicates.matrices import UnitTriangularPredicate
        return UnitTriangularPredicate()

    @memoize_property
    def eq(self):
        from .relation.equality import EqualityPredicate
        return EqualityPredicate()

    @memoize_property
    def ne(self):
        from .relation.equality import UnequalityPredicate
        return UnequalityPredicate()

    @memoize_property
    def gt(self):
        from .relation.equality import StrictGreaterThanPredicate
        return StrictGreaterThanPredicate()

    @memoize_property
    def ge(self):
        from .relation.equality import GreaterThanPredicate
        return GreaterThanPredicate()

    @memoize_property
    def lt(self):
        from .relation.equality import StrictLessThanPredicate
        return StrictLessThanPredicate()

    @memoize_property
    def le(self):
        from .relation.equality import LessThanPredicate
        return LessThanPredicate()


Q = AssumptionKeys()

def _extract_all_facts(assump, exprs):
    """
    Extract all relevant assumptions from *assump* with respect to given *exprs*.

    Parameters
    ==========

    assump : sympy.assumptions.cnf.CNF

    exprs : tuple of expressions

    Returns
    =======

    sympy.assumptions.cnf.CNF

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.ask import _extract_all_facts
    >>> from sympy.abc import x, y
    >>> assump = CNF.from_prop(Q.positive(x) & Q.integer(y))
    >>> exprs = (x,)
    >>> cnf = _extract_all_facts(assump, exprs)
    >>> cnf.clauses
    {frozenset({Literal(Q.positive, False)})}

    """
    facts = set()

    for clause in assump.clauses:
        args = []
        for literal in clause:
            if isinstance(literal.lit, AppliedPredicate) and len(literal.lit.arguments) == 1:
                if literal.lit.arg in exprs:
                    # Add literal if it has matching in it
                    args.append(Literal(literal.lit.function, literal.is_Not))
                else:
                    # If any of the literals doesn't have matching expr don't add the whole clause.
                    break
            else:
                # If any of the literals aren't unary predicate don't add the whole clause.
                break

        else:
            if args:
                facts.add(frozenset(args))
    return CNF(facts)


def ask(proposition, assumptions=True, context=global_assumptions):
    """
    Function to evaluate the proposition with assumptions.

    Explanation
    ===========

    This function evaluates the proposition to ``True`` or ``False`` if
    the truth value can be determined. If not, it returns ``None``.

    It should be discerned from :func:`~.refine` which, when applied to a
    proposition, simplifies the argument to symbolic ``Boolean`` instead of
    Python built-in ``True``, ``False`` or ``None``.

    **Syntax**

        * ask(proposition)
            Evaluate the *proposition* in global assumption context.

        * ask(proposition, assumptions)
            Evaluate the *proposition* with respect to *assumptions* in
            global assumption context.

    Parameters
    ==========

    proposition : Boolean
        Proposition which will be evaluated to boolean value. If this is
        not ``AppliedPredicate``, it will be wrapped by ``Q.is_true``.

    assumptions : Boolean, optional
        Local assumptions to evaluate the *proposition*.

    context : AssumptionsContext, optional
        Default assumptions to evaluate the *proposition*. By default,
        this is ``sympy.assumptions.global_assumptions`` variable.

    Returns
    =======

    ``True``, ``False``, or ``None``

    Raises
    ======

    TypeError : *proposition* or *assumptions* is not valid logical expression.

    ValueError : assumptions are inconsistent.

    Examples
    ========

    >>> from sympy import ask, Q, pi
    >>> from sympy.abc import x, y
    >>> ask(Q.rational(pi))
    False
    >>> ask(Q.even(x*y), Q.even(x) & Q.integer(y))
    True
    >>> ask(Q.prime(4*x), Q.integer(x))
    False

    If the truth value cannot be determined, ``None`` will be returned.

    >>> print(ask(Q.odd(3*x))) # cannot determine unless we know x
    None

    ``ValueError`` is raised if assumptions are inconsistent.

    >>> ask(Q.integer(x), Q.even(x) & Q.odd(x))
    Traceback (most recent call last):
      ...
    ValueError: inconsistent assumptions Q.even(x) & Q.odd(x)

    Notes
    =====

    Relations in assumptions are not implemented (yet), so the following
    will not give a meaningful result.

    >>> ask(Q.positive(x), x > 0)

    It is however a work in progress.

    See Also
    ========

    sympy.assumptions.refine.refine : Simplification using assumptions.
        Proposition is not reduced to ``None`` if the truth value cannot
        be determined.
    """
    from sympy.assumptions.satask import satask
    from sympy.assumptions.lra_satask import lra_satask
    from sympy.logic.algorithms.lra_theory import UnhandledInput

    proposition = sympify(proposition)
    assumptions = sympify(assumptions)

    if isinstance(proposition, Predicate) or proposition.kind is not BooleanKind:
        raise TypeError("proposition must be a valid logical expression")

    if isinstance(assumptions, Predicate) or assumptions.kind is not BooleanKind:
        raise TypeError("assumptions must be a valid logical expression")

    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
    if isinstance(proposition, AppliedPredicate):
        key, args = proposition.function, proposition.arguments
    elif proposition.func in binrelpreds:
        key, args = binrelpreds[type(proposition)], proposition.args
    else:
        key, args = Q.is_true, (proposition,)

    # convert local and global assumptions to CNF
    assump_cnf = CNF.from_prop(assumptions)
    assump_cnf.extend(context)

    # extract the relevant facts from assumptions with respect to args
    local_facts = _extract_all_facts(assump_cnf, args)

    # convert default facts and assumed facts to encoded CNF
    known_facts_cnf = get_all_known_facts()
    enc_cnf = EncodedCNF()
    enc_cnf.from_cnf(CNF(known_facts_cnf))
    enc_cnf.add_from_cnf(local_facts)

    # check the satisfiability of given assumptions
    if local_facts.clauses and satisfiable(enc_cnf) is False:
        raise ValueError("inconsistent assumptions %s" % assumptions)

    # quick computation for single fact
    res = _ask_single_fact(key, local_facts)
    if res is not None:
        return res

    # direct resolution method, no logic
    res = key(*args)._eval_ask(assumptions)
    if res is not None:
        return bool(res)

    # using satask (still costly)
    res = satask(proposition, assumptions=assumptions, context=context)
    if res is not None:
        return res

    try:
        res = lra_satask(proposition, assumptions=assumptions, context=context)
    except UnhandledInput:
        return None

    return res


def _ask_single_fact(key, local_facts):
    """
    Compute the truth value of single predicate using assumptions.

    Parameters
    ==========

    key : sympy.assumptions.assume.Predicate
        Proposition predicate.

    local_facts : sympy.assumptions.cnf.CNF
        Local assumption in CNF form.

    Returns
    =======

    ``True``, ``False`` or ``None``

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.ask import _ask_single_fact

    If prerequisite of proposition is rejected by the assumption,
    return ``False``.

    >>> key, assump = Q.zero, ~Q.zero
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False
    >>> key, assump = Q.zero, ~Q.even
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False

    If assumption implies the proposition, return ``True``.

    >>> key, assump = Q.even, Q.zero
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    True

    If proposition rejects the assumption, return ``False``.

    >>> key, assump = Q.even, Q.odd
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False
    """
    if local_facts.clauses:

        known_facts_dict = get_known_facts_dict()

        if len(local_facts.clauses) == 1:
            cl, = local_facts.clauses
            if len(cl) == 1:
                f, = cl
                prop_facts = known_facts_dict.get(key, None)
                prop_req = prop_facts[0] if prop_facts is not None else set()
                if f.is_Not and f.arg in prop_req:
                    # the prerequisite of proposition is rejected
                    return False

        for clause in local_facts.clauses:
            if len(clause) == 1:
                f, = clause
                prop_facts = known_facts_dict.get(f.arg, None) if not f.is_Not else None
                if prop_facts is None:
                    continue

                prop_req, prop_rej = prop_facts
                if key in prop_req:
                    # assumption implies the proposition
                    return True
                elif key in prop_rej:
                    # proposition rejects the assumption
                    return False

    return None


def register_handler(key, handler):
    """
    Register a handler in the ask system. key must be a string and handler a
    class inheriting from AskHandler.

    .. deprecated:: 1.8.
        Use multipledispatch handler instead. See :obj:`~.Predicate`.

    """
    sympy_deprecation_warning(
        """
        The AskHandler system is deprecated. The register_handler() function
        should be replaced with the multipledispatch handler of Predicate.
        """,
        deprecated_since_version="1.8",
        active_deprecations_target='deprecated-askhandler',
    )
    if isinstance(key, Predicate):
        key = key.name.name
    Qkey = getattr(Q, key, None)
    if Qkey is not None:
        Qkey.add_handler(handler)
    else:
        setattr(Q, key, Predicate(key, handlers=[handler]))


def remove_handler(key, handler):
    """
    Removes a handler from the ask system.

    .. deprecated:: 1.8.
        Use multipledispatch handler instead. See :obj:`~.Predicate`.

    """
    sympy_deprecation_warning(
        """
        The AskHandler system is deprecated. The remove_handler() function
        should be replaced with the multipledispatch handler of Predicate.
        """,
        deprecated_since_version="1.8",
        active_deprecations_target='deprecated-askhandler',
    )
    if isinstance(key, Predicate):
        key = key.name.name
    # Don't show the same warning again recursively
    with ignore_warnings(SymPyDeprecationWarning):
        getattr(Q, key).remove_handler(handler)


from sympy.assumptions.ask_generated import (get_all_known_facts,
    get_known_facts_dict)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\functions.py ===
from functools import reduce

from sympy import (sympify, diff, sin, cos, Matrix, symbols,
                                Function, S, Symbol, linear_eq_to_matrix)
from sympy.integrals.integrals import integrate
from sympy.simplify.trigsimp import trigsimp
from .vector import Vector, _check_vector
from .frame import CoordinateSym, _check_frame
from .dyadic import Dyadic
from .printing import vprint, vsprint, vpprint, vlatex, init_vprinting
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import translate

__all__ = ['cross', 'dot', 'express', 'time_derivative', 'outer',
           'kinematic_equations', 'get_motion_params', 'partial_velocity',
           'dynamicsymbols', 'vprint', 'vsprint', 'vpprint', 'vlatex',
           'init_vprinting']


def cross(vec1, vec2):
    """Cross product convenience wrapper for Vector.cross(): \n"""
    if not isinstance(vec1, (Vector, Dyadic)):
        raise TypeError('Cross product is between two vectors')
    return vec1 ^ vec2


cross.__doc__ += Vector.cross.__doc__  # type: ignore


def dot(vec1, vec2):
    """Dot product convenience wrapper for Vector.dot(): \n"""
    if not isinstance(vec1, (Vector, Dyadic)):
        raise TypeError('Dot product is between two vectors')
    return vec1 & vec2


dot.__doc__ += Vector.dot.__doc__  # type: ignore


def express(expr, frame, frame2=None, variables=False):
    """
    Global function for 'express' functionality.

    Re-expresses a Vector, scalar(sympyfiable) or Dyadic in given frame.

    Refer to the local methods of Vector and Dyadic for details.
    If 'variables' is True, then the coordinate variables (CoordinateSym
    instances) of other frames present in the vector/scalar field or
    dyadic expression are also substituted in terms of the base scalars of
    this frame.

    Parameters
    ==========

    expr : Vector/Dyadic/scalar(sympyfiable)
        The expression to re-express in ReferenceFrame 'frame'

    frame: ReferenceFrame
        The reference frame to express expr in

    frame2 : ReferenceFrame
        The other frame required for re-expression(only for Dyadic expr)

    variables : boolean
        Specifies whether to substitute the coordinate variables present
        in expr, in terms of those of frame

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> B = N.orientnew('B', 'Axis', [q, N.z])
    >>> d = outer(N.x, N.x)
    >>> from sympy.physics.vector import express
    >>> express(d, B, N)
    cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)
    >>> express(B.x, N)
    cos(q)*N.x + sin(q)*N.y
    >>> express(N[0], B, variables=True)
    B_x*cos(q) - B_y*sin(q)

    """

    _check_frame(frame)

    if expr == 0:
        return expr

    if isinstance(expr, Vector):
        # Given expr is a Vector
        if variables:
            # If variables attribute is True, substitute the coordinate
            # variables in the Vector
            frame_list = [x[-1] for x in expr.args]
            subs_dict = {}
            for f in frame_list:
                subs_dict.update(f.variable_map(frame))
            expr = expr.subs(subs_dict)
        # Re-express in this frame
        outvec = Vector([])
        for v in expr.args:
            if v[1] != frame:
                temp = frame.dcm(v[1]) * v[0]
                if Vector.simp:
                    temp = temp.applyfunc(lambda x:
                                          trigsimp(x, method='fu'))
                outvec += Vector([(temp, frame)])
            else:
                outvec += Vector([v])
        return outvec

    if isinstance(expr, Dyadic):
        if frame2 is None:
            frame2 = frame
        _check_frame(frame2)
        ol = Dyadic(0)
        for v in expr.args:
            ol += express(v[0], frame, variables=variables) * \
                  (express(v[1], frame, variables=variables) |
                   express(v[2], frame2, variables=variables))
        return ol

    else:
        if variables:
            # Given expr is a scalar field
            frame_set = set()
            expr = sympify(expr)
            # Substitute all the coordinate variables
            for x in expr.free_symbols:
                if isinstance(x, CoordinateSym) and x.frame != frame:
                    frame_set.add(x.frame)
            subs_dict = {}
            for f in frame_set:
                subs_dict.update(f.variable_map(frame))
            return expr.subs(subs_dict)
        return expr


def time_derivative(expr, frame, order=1):
    """
    Calculate the time derivative of a vector/scalar field function
    or dyadic expression in given frame.

    References
    ==========

    https://en.wikipedia.org/wiki/Rotating_reference_frame#Time_derivatives_in_the_two_frames

    Parameters
    ==========

    expr : Vector/Dyadic/sympifyable
        The expression whose time derivative is to be calculated

    frame : ReferenceFrame
        The reference frame to calculate the time derivative in

    order : integer
        The order of the derivative to be calculated

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import Symbol
    >>> q1 = Symbol('q1')
    >>> u1 = dynamicsymbols('u1')
    >>> N = ReferenceFrame('N')
    >>> A = N.orientnew('A', 'Axis', [q1, N.x])
    >>> v = u1 * N.x
    >>> A.set_ang_vel(N, 10*A.x)
    >>> from sympy.physics.vector import time_derivative
    >>> time_derivative(v, N)
    u1'*N.x
    >>> time_derivative(u1*A[0], N)
    N_x*u1'
    >>> B = N.orientnew('B', 'Axis', [u1, N.z])
    >>> from sympy.physics.vector import outer
    >>> d = outer(N.x, N.x)
    >>> time_derivative(d, B)
    - u1'*(N.y|N.x) - u1'*(N.x|N.y)

    """

    t = dynamicsymbols._t
    _check_frame(frame)

    if order == 0:
        return expr
    if order % 1 != 0 or order < 0:
        raise ValueError("Unsupported value of order entered")

    if isinstance(expr, Vector):
        outlist = []
        for v in expr.args:
            if v[1] == frame:
                outlist += [(express(v[0], frame, variables=True).diff(t),
                             frame)]
            else:
                outlist += (time_derivative(Vector([v]), v[1]) +
                            (v[1].ang_vel_in(frame) ^ Vector([v]))).args
        outvec = Vector(outlist)
        return time_derivative(outvec, frame, order - 1)

    if isinstance(expr, Dyadic):
        ol = Dyadic(0)
        for v in expr.args:
            ol += (v[0].diff(t) * (v[1] | v[2]))
            ol += (v[0] * (time_derivative(v[1], frame) | v[2]))
            ol += (v[0] * (v[1] | time_derivative(v[2], frame)))
        return time_derivative(ol, frame, order - 1)

    else:
        return diff(express(expr, frame, variables=True), t, order)


def outer(vec1, vec2):
    """Outer product convenience wrapper for Vector.outer():\n"""
    if not isinstance(vec1, Vector):
        raise TypeError('Outer product is between two Vectors')
    return vec1.outer(vec2)


outer.__doc__ += Vector.outer.__doc__  # type: ignore


def kinematic_equations(speeds, coords, rot_type, rot_order=''):
    """Gives equations relating the qdot's to u's for a rotation type.

    Supply rotation type and order as in orient. Speeds are assumed to be
    body-fixed; if we are defining the orientation of B in A using by rot_type,
    the angular velocity of B in A is assumed to be in the form: speed[0]*B.x +
    speed[1]*B.y + speed[2]*B.z

    Parameters
    ==========

    speeds : list of length 3
        The body fixed angular velocity measure numbers.
    coords : list of length 3 or 4
        The coordinates used to define the orientation of the two frames.
    rot_type : str
        The type of rotation used to create the equations. Body, Space, or
        Quaternion only
    rot_order : str or int
        If applicable, the order of a series of rotations.

    Examples
    ========

    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy.physics.vector import kinematic_equations, vprint
    >>> u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    >>> q1, q2, q3 = dynamicsymbols('q1 q2 q3')
    >>> vprint(kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', '313'),
    ...     order=None)
    [-(u1*sin(q3) + u2*cos(q3))/sin(q2) + q1', -u1*cos(q3) + u2*sin(q3) + q2', (u1*sin(q3) + u2*cos(q3))*cos(q2)/sin(q2) - u3 + q3']

    """

    # Code below is checking and sanitizing input
    approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131',
                       '212', '232', '313', '323', '1', '2', '3', '')
    # make sure XYZ => 123 and rot_type is in lower case
    rot_order = translate(str(rot_order), 'XYZxyz', '123123')
    rot_type = rot_type.lower()

    if not isinstance(speeds, (list, tuple)):
        raise TypeError('Need to supply speeds in a list')
    if len(speeds) != 3:
        raise TypeError('Need to supply 3 body-fixed speeds')
    if not isinstance(coords, (list, tuple)):
        raise TypeError('Need to supply coordinates in a list')
    if rot_type in ['body', 'space']:
        if rot_order not in approved_orders:
            raise ValueError('Not an acceptable rotation order')
        if len(coords) != 3:
            raise ValueError('Need 3 coordinates for body or space')
        # Actual hard-coded kinematic differential equations
        w1, w2, w3 = speeds
        if w1 == w2 == w3 == 0:
            return [S.Zero]*3
        q1, q2, q3 = coords
        q1d, q2d, q3d = [diff(i, dynamicsymbols._t) for i in coords]
        s1, s2, s3 = [sin(q1), sin(q2), sin(q3)]
        c1, c2, c3 = [cos(q1), cos(q2), cos(q3)]
        if rot_type == 'body':
            if rot_order == '123':
                return [q1d - (w1 * c3 - w2 * s3) / c2, q2d - w1 * s3 - w2 *
                        c3, q3d - (-w1 * c3 + w2 * s3) * s2 / c2 - w3]
            if rot_order == '231':
                return [q1d - (w2 * c3 - w3 * s3) / c2, q2d - w2 * s3 - w3 *
                        c3, q3d - w1 - (- w2 * c3 + w3 * s3) * s2 / c2]
            if rot_order == '312':
                return [q1d - (-w1 * s3 + w3 * c3) / c2, q2d - w1 * c3 - w3 *
                        s3, q3d - (w1 * s3 - w3 * c3) * s2 / c2 - w2]
            if rot_order == '132':
                return [q1d - (w1 * c3 + w3 * s3) / c2, q2d + w1 * s3 - w3 *
                        c3, q3d - (w1 * c3 + w3 * s3) * s2 / c2 - w2]
            if rot_order == '213':
                return [q1d - (w1 * s3 + w2 * c3) / c2, q2d - w1 * c3 + w2 *
                        s3, q3d - (w1 * s3 + w2 * c3) * s2 / c2 - w3]
            if rot_order == '321':
                return [q1d - (w2 * s3 + w3 * c3) / c2, q2d - w2 * c3 + w3 *
                        s3, q3d - w1 - (w2 * s3 + w3 * c3) * s2 / c2]
            if rot_order == '121':
                return [q1d - (w2 * s3 + w3 * c3) / s2, q2d - w2 * c3 + w3 *
                        s3, q3d - w1 + (w2 * s3 + w3 * c3) * c2 / s2]
            if rot_order == '131':
                return [q1d - (-w2 * c3 + w3 * s3) / s2, q2d - w2 * s3 - w3 *
                        c3, q3d - w1 - (w2 * c3 - w3 * s3) * c2 / s2]
            if rot_order == '212':
                return [q1d - (w1 * s3 - w3 * c3) / s2, q2d - w1 * c3 - w3 *
                        s3, q3d - (-w1 * s3 + w3 * c3) * c2 / s2 - w2]
            if rot_order == '232':
                return [q1d - (w1 * c3 + w3 * s3) / s2, q2d + w1 * s3 - w3 *
                        c3, q3d + (w1 * c3 + w3 * s3) * c2 / s2 - w2]
            if rot_order == '313':
                return [q1d - (w1 * s3 + w2 * c3) / s2, q2d - w1 * c3 + w2 *
                        s3, q3d + (w1 * s3 + w2 * c3) * c2 / s2 - w3]
            if rot_order == '323':
                return [q1d - (-w1 * c3 + w2 * s3) / s2, q2d - w1 * s3 - w2 *
                        c3, q3d - (w1 * c3 - w2 * s3) * c2 / s2 - w3]
        if rot_type == 'space':
            if rot_order == '123':
                return [q1d - w1 - (w2 * s1 + w3 * c1) * s2 / c2, q2d - w2 *
                        c1 + w3 * s1, q3d - (w2 * s1 + w3 * c1) / c2]
            if rot_order == '231':
                return [q1d - (w1 * c1 + w3 * s1) * s2 / c2 - w2, q2d + w1 *
                        s1 - w3 * c1, q3d - (w1 * c1 + w3 * s1) / c2]
            if rot_order == '312':
                return [q1d - (w1 * s1 + w2 * c1) * s2 / c2 - w3, q2d - w1 *
                        c1 + w2 * s1, q3d - (w1 * s1 + w2 * c1) / c2]
            if rot_order == '132':
                return [q1d - w1 - (-w2 * c1 + w3 * s1) * s2 / c2, q2d - w2 *
                        s1 - w3 * c1, q3d - (w2 * c1 - w3 * s1) / c2]
            if rot_order == '213':
                return [q1d - (w1 * s1 - w3 * c1) * s2 / c2 - w2, q2d - w1 *
                        c1 - w3 * s1, q3d - (-w1 * s1 + w3 * c1) / c2]
            if rot_order == '321':
                return [q1d - (-w1 * c1 + w2 * s1) * s2 / c2 - w3, q2d - w1 *
                        s1 - w2 * c1, q3d - (w1 * c1 - w2 * s1) / c2]
            if rot_order == '121':
                return [q1d - w1 + (w2 * s1 + w3 * c1) * c2 / s2, q2d - w2 *
                        c1 + w3 * s1, q3d - (w2 * s1 + w3 * c1) / s2]
            if rot_order == '131':
                return [q1d - w1 - (w2 * c1 - w3 * s1) * c2 / s2, q2d - w2 *
                        s1 - w3 * c1, q3d - (-w2 * c1 + w3 * s1) / s2]
            if rot_order == '212':
                return [q1d - (-w1 * s1 + w3 * c1) * c2 / s2 - w2, q2d - w1 *
                        c1 - w3 * s1, q3d - (w1 * s1 - w3 * c1) / s2]
            if rot_order == '232':
                return [q1d + (w1 * c1 + w3 * s1) * c2 / s2 - w2, q2d + w1 *
                        s1 - w3 * c1, q3d - (w1 * c1 + w3 * s1) / s2]
            if rot_order == '313':
                return [q1d + (w1 * s1 + w2 * c1) * c2 / s2 - w3, q2d - w1 *
                        c1 + w2 * s1, q3d - (w1 * s1 + w2 * c1) / s2]
            if rot_order == '323':
                return [q1d - (w1 * c1 - w2 * s1) * c2 / s2 - w3, q2d - w1 *
                        s1 - w2 * c1, q3d - (-w1 * c1 + w2 * s1) / s2]
    elif rot_type == 'quaternion':
        if rot_order != '':
            raise ValueError('Cannot have rotation order for quaternion')
        if len(coords) != 4:
            raise ValueError('Need 4 coordinates for quaternion')
        # Actual hard-coded kinematic differential equations
        e0, e1, e2, e3 = coords
        w = Matrix(speeds + [0])
        E = Matrix([[e0, -e3, e2, e1],
                    [e3, e0, -e1, e2],
                    [-e2, e1, e0, e3],
                    [-e1, -e2, -e3, e0]])
        edots = Matrix([diff(i, dynamicsymbols._t) for i in [e1, e2, e3, e0]])
        return list(edots.T - 0.5 * w.T * E.T)
    else:
        raise ValueError('Not an approved rotation type for this function')


def get_motion_params(frame, **kwargs):
    """
    Returns the three motion parameters - (acceleration, velocity, and
    position) as vectorial functions of time in the given frame.

    If a higher order differential function is provided, the lower order
    functions are used as boundary conditions. For example, given the
    acceleration, the velocity and position parameters are taken as
    boundary conditions.

    The values of time at which the boundary conditions are specified
    are taken from timevalue1(for position boundary condition) and
    timevalue2(for velocity boundary condition).

    If any of the boundary conditions are not provided, they are taken
    to be zero by default (zero vectors, in case of vectorial inputs). If
    the boundary conditions are also functions of time, they are converted
    to constants by substituting the time values in the dynamicsymbols._t
    time Symbol.

    This function can also be used for calculating rotational motion
    parameters. Have a look at the Parameters and Examples for more clarity.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame to express the motion parameters in

    acceleration : Vector
        Acceleration of the object/frame as a function of time

    velocity : Vector
        Velocity as function of time or as boundary condition
        of velocity at time = timevalue1

    position : Vector
        Velocity as function of time or as boundary condition
        of velocity at time = timevalue1

    timevalue1 : sympyfiable
        Value of time for position boundary condition

    timevalue2 : sympyfiable
        Value of time for velocity boundary condition

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, get_motion_params, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import symbols
    >>> R = ReferenceFrame('R')
    >>> v1, v2, v3 = dynamicsymbols('v1 v2 v3')
    >>> v = v1*R.x + v2*R.y + v3*R.z
    >>> get_motion_params(R, position = v)
    (v1''*R.x + v2''*R.y + v3''*R.z, v1'*R.x + v2'*R.y + v3'*R.z, v1*R.x + v2*R.y + v3*R.z)
    >>> a, b, c = symbols('a b c')
    >>> v = a*R.x + b*R.y + c*R.z
    >>> get_motion_params(R, velocity = v)
    (0, a*R.x + b*R.y + c*R.z, a*t*R.x + b*t*R.y + c*t*R.z)
    >>> parameters = get_motion_params(R, acceleration = v)
    >>> parameters[1]
    a*t*R.x + b*t*R.y + c*t*R.z
    >>> parameters[2]
    a*t**2/2*R.x + b*t**2/2*R.y + c*t**2/2*R.z

    """

    def _process_vector_differential(vectdiff, condition, variable, ordinate,
                                     frame):
        """
        Helper function for get_motion methods. Finds derivative of vectdiff
        wrt variable, and its integral using the specified boundary condition
        at value of variable = ordinate.
        Returns a tuple of - (derivative, function and integral) wrt vectdiff

        """

        # Make sure boundary condition is independent of 'variable'
        if condition != 0:
            condition = express(condition, frame, variables=True)
        # Special case of vectdiff == 0
        if vectdiff == Vector(0):
            return (0, 0, condition)
        # Express vectdiff completely in condition's frame to give vectdiff1
        vectdiff1 = express(vectdiff, frame)
        # Find derivative of vectdiff
        vectdiff2 = time_derivative(vectdiff, frame)
        # Integrate and use boundary condition
        vectdiff0 = Vector(0)
        lims = (variable, ordinate, variable)
        for dim in frame:
            function1 = vectdiff1.dot(dim)
            abscissa = dim.dot(condition).subs({variable: ordinate})
            # Indefinite integral of 'function1' wrt 'variable', using
            # the given initial condition (ordinate, abscissa).
            vectdiff0 += (integrate(function1, lims) + abscissa) * dim
        # Return tuple
        return (vectdiff2, vectdiff, vectdiff0)

    _check_frame(frame)
    # Decide mode of operation based on user's input
    if 'acceleration' in kwargs:
        mode = 2
    elif 'velocity' in kwargs:
        mode = 1
    else:
        mode = 0
    # All the possible parameters in kwargs
    # Not all are required for every case
    # If not specified, set to default values(may or may not be used in
    # calculations)
    conditions = ['acceleration', 'velocity', 'position',
                  'timevalue', 'timevalue1', 'timevalue2']
    for i, x in enumerate(conditions):
        if x not in kwargs:
            if i < 3:
                kwargs[x] = Vector(0)
            else:
                kwargs[x] = S.Zero
        elif i < 3:
            _check_vector(kwargs[x])
        else:
            kwargs[x] = sympify(kwargs[x])
    if mode == 2:
        vel = _process_vector_differential(kwargs['acceleration'],
                                           kwargs['velocity'],
                                           dynamicsymbols._t,
                                           kwargs['timevalue2'], frame)[2]
        pos = _process_vector_differential(vel, kwargs['position'],
                                           dynamicsymbols._t,
                                           kwargs['timevalue1'], frame)[2]
        return (kwargs['acceleration'], vel, pos)
    elif mode == 1:
        return _process_vector_differential(kwargs['velocity'],
                                            kwargs['position'],
                                            dynamicsymbols._t,
                                            kwargs['timevalue1'], frame)
    else:
        vel = time_derivative(kwargs['position'], frame)
        acc = time_derivative(vel, frame)
        return (acc, vel, kwargs['position'])


def partial_velocity(vel_vecs, gen_speeds, frame):
    """Returns a list of partial velocities with respect to the provided
    generalized speeds in the given reference frame for each of the supplied
    velocity vectors.

    The output is a list of lists. The outer list has a number of elements
    equal to the number of supplied velocity vectors. The inner lists are, for
    each velocity vector, the partial derivatives of that velocity vector with
    respect to the generalized speeds supplied.

    Parameters
    ==========

    vel_vecs : iterable
        An iterable of velocity vectors (angular or linear).
    gen_speeds : iterable
        An iterable of generalized speeds.
    frame : ReferenceFrame
        The reference frame that the partial derivatives are going to be taken
        in.

    Examples
    ========

    >>> from sympy.physics.vector import Point, ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy.physics.vector import partial_velocity
    >>> u = dynamicsymbols('u')
    >>> N = ReferenceFrame('N')
    >>> P = Point('P')
    >>> P.set_vel(N, u * N.x)
    >>> vel_vecs = [P.vel(N)]
    >>> gen_speeds = [u]
    >>> partial_velocity(vel_vecs, gen_speeds, N)
    [[N.x]]

    """

    if not iterable(vel_vecs):
        raise TypeError('Velocity vectors must be contained in an iterable.')

    if not iterable(gen_speeds):
        raise TypeError('Generalized speeds must be contained in an iterable')

    vec_partials = []
    gen_speeds = list(gen_speeds)
    for vel in vel_vecs:
        partials = [Vector(0) for _ in gen_speeds]
        for components, ref in vel.args:
            mat, _ = linear_eq_to_matrix(components, gen_speeds)
            for i in range(len(gen_speeds)):
                for dim, direction in enumerate(ref):
                    if mat[dim, i] != 0:
                        partials[i] += direction * mat[dim, i]

        vec_partials.append(partials)

    return vec_partials


def dynamicsymbols(names, level=0, **assumptions):
    """Uses symbols and Function for functions of time.

    Creates a SymPy UndefinedFunction, which is then initialized as a function
    of a variable, the default being Symbol('t').

    Parameters
    ==========

    names : str
        Names of the dynamic symbols you want to create; works the same way as
        inputs to symbols
    level : int
        Level of differentiation of the returned function; d/dt once of t,
        twice of t, etc.
    assumptions :
        - real(bool) : This is used to set the dynamicsymbol as real,
                    by default is False.
        - positive(bool) : This is used to set the dynamicsymbol as positive,
                    by default is False.
        - commutative(bool) : This is used to set the commutative property of
                    a dynamicsymbol, by default is True.
        - integer(bool) : This is used to set the dynamicsymbol as integer,
                    by default is False.

    Examples
    ========

    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy import diff, Symbol
    >>> q1 = dynamicsymbols('q1')
    >>> q1
    q1(t)
    >>> q2 = dynamicsymbols('q2', real=True)
    >>> q2.is_real
    True
    >>> q3 = dynamicsymbols('q3', positive=True)
    >>> q3.is_positive
    True
    >>> q4, q5 = dynamicsymbols('q4,q5', commutative=False)
    >>> bool(q4*q5 != q5*q4)
    True
    >>> q6 = dynamicsymbols('q6', integer=True)
    >>> q6.is_integer
    True
    >>> diff(q1, Symbol('t'))
    Derivative(q1(t), t)

    """
    esses = symbols(names, cls=Function, **assumptions)
    t = dynamicsymbols._t
    if iterable(esses):
        esses = [reduce(diff, [t] * level, e(t)) for e in esses]
        return esses
    else:
        return reduce(diff, [t] * level, esses(t))


dynamicsymbols._t = Symbol('t')  # type: ignore
dynamicsymbols._str = '\''  # type: ignore

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\wrappers\request.py ===
from __future__ import annotations

import collections.abc as cabc
import functools
import json
import typing as t
from io import BytesIO

from .._internal import _wsgi_decoding_dance
from ..datastructures import CombinedMultiDict
from ..datastructures import EnvironHeaders
from ..datastructures import FileStorage
from ..datastructures import ImmutableMultiDict
from ..datastructures import iter_multi_items
from ..datastructures import MultiDict
from ..exceptions import BadRequest
from ..exceptions import UnsupportedMediaType
from ..formparser import default_stream_factory
from ..formparser import FormDataParser
from ..sansio.request import Request as _SansIORequest
from ..utils import cached_property
from ..utils import environ_property
from ..wsgi import _get_server
from ..wsgi import get_input_stream

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class Request(_SansIORequest):
    """Represents an incoming WSGI HTTP request, with headers and body
    taken from the WSGI environment. Has properties and methods for
    using the functionality defined by various HTTP specs. The data in
    requests object is read-only.

    Text data is assumed to use UTF-8 encoding, which should be true for
    the vast majority of modern clients. Using an encoding set by the
    client is unsafe in Python due to extra encodings it provides, such
    as ``zip``. To change the assumed encoding, subclass and replace
    :attr:`charset`.

    :param environ: The WSGI environ is generated by the WSGI server and
        contains information about the server configuration and client
        request.
    :param populate_request: Add this request object to the WSGI environ
        as ``environ['werkzeug.request']``. Can be useful when
        debugging.
    :param shallow: Makes reading from :attr:`stream` (and any method
        that would read from it) raise a :exc:`RuntimeError`. Useful to
        prevent consuming the form data in middleware, which would make
        it unavailable to the final application.

    .. versionchanged:: 3.0
        The ``charset``, ``url_charset``, and ``encoding_errors`` parameters
        were removed.

    .. versionchanged:: 2.1
        Old ``BaseRequest`` and mixin classes were removed.

    .. versionchanged:: 2.1
        Remove the ``disable_data_descriptor`` attribute.

    .. versionchanged:: 2.0
        Combine ``BaseRequest`` and mixins into a single ``Request``
        class.

    .. versionchanged:: 0.5
        Read-only mode is enforced with immutable classes for all data.
    """

    #: the maximum content length.  This is forwarded to the form data
    #: parsing function (:func:`parse_form_data`).  When set and the
    #: :attr:`form` or :attr:`files` attribute is accessed and the
    #: parsing fails because more than the specified value is transmitted
    #: a :exc:`~werkzeug.exceptions.RequestEntityTooLarge` exception is raised.
    #:
    #: .. versionadded:: 0.5
    max_content_length: int | None = None

    #: the maximum form field size.  This is forwarded to the form data
    #: parsing function (:func:`parse_form_data`).  When set and the
    #: :attr:`form` or :attr:`files` attribute is accessed and the
    #: data in memory for post data is longer than the specified value a
    #: :exc:`~werkzeug.exceptions.RequestEntityTooLarge` exception is raised.
    #:
    #: .. versionchanged:: 3.1
    #:     Defaults to 500kB instead of unlimited.
    #:
    #: .. versionadded:: 0.5
    max_form_memory_size: int | None = 500_000

    #: The maximum number of multipart parts to parse, passed to
    #: :attr:`form_data_parser_class`. Parsing form data with more than this
    #: many parts will raise :exc:`~.RequestEntityTooLarge`.
    #:
    #: .. versionadded:: 2.2.3
    max_form_parts = 1000

    #: The form data parser that should be used.  Can be replaced to customize
    #: the form date parsing.
    form_data_parser_class: type[FormDataParser] = FormDataParser

    #: The WSGI environment containing HTTP headers and information from
    #: the WSGI server.
    environ: WSGIEnvironment

    #: Set when creating the request object. If ``True``, reading from
    #: the request body will cause a ``RuntimeException``. Useful to
    #: prevent modifying the stream from middleware.
    shallow: bool

    def __init__(
        self,
        environ: WSGIEnvironment,
        populate_request: bool = True,
        shallow: bool = False,
    ) -> None:
        super().__init__(
            method=environ.get("REQUEST_METHOD", "GET"),
            scheme=environ.get("wsgi.url_scheme", "http"),
            server=_get_server(environ),
            root_path=_wsgi_decoding_dance(environ.get("SCRIPT_NAME") or ""),
            path=_wsgi_decoding_dance(environ.get("PATH_INFO") or ""),
            query_string=environ.get("QUERY_STRING", "").encode("latin1"),
            headers=EnvironHeaders(environ),
            remote_addr=environ.get("REMOTE_ADDR"),
        )
        self.environ = environ
        self.shallow = shallow

        if populate_request and not shallow:
            self.environ["werkzeug.request"] = self

    @classmethod
    def from_values(cls, *args: t.Any, **kwargs: t.Any) -> Request:
        """Create a new request object based on the values provided.  If
        environ is given missing values are filled from there.  This method is
        useful for small scripts when you need to simulate a request from an URL.
        Do not use this method for unittesting, there is a full featured client
        object (:class:`Client`) that allows to create multipart requests,
        support for cookies etc.

        This accepts the same options as the
        :class:`~werkzeug.test.EnvironBuilder`.

        .. versionchanged:: 0.5
           This method now accepts the same arguments as
           :class:`~werkzeug.test.EnvironBuilder`.  Because of this the
           `environ` parameter is now called `environ_overrides`.

        :return: request object
        """
        from ..test import EnvironBuilder

        builder = EnvironBuilder(*args, **kwargs)
        try:
            return builder.get_request(cls)
        finally:
            builder.close()

    @classmethod
    def application(cls, f: t.Callable[[Request], WSGIApplication]) -> WSGIApplication:
        """Decorate a function as responder that accepts the request as
        the last argument.  This works like the :func:`responder`
        decorator but the function is passed the request object as the
        last argument and the request object will be closed
        automatically::

            @Request.application
            def my_wsgi_app(request):
                return Response('Hello World!')

        As of Werkzeug 0.14 HTTP exceptions are automatically caught and
        converted to responses instead of failing.

        :param f: the WSGI callable to decorate
        :return: a new WSGI callable
        """
        #: return a callable that wraps the -2nd argument with the request
        #: and calls the function with all the arguments up to that one and
        #: the request.  The return value is then called with the latest
        #: two arguments.  This makes it possible to use this decorator for
        #: both standalone WSGI functions as well as bound methods and
        #: partially applied functions.
        from ..exceptions import HTTPException

        @functools.wraps(f)
        def application(*args: t.Any) -> cabc.Iterable[bytes]:
            request = cls(args[-2])
            with request:
                try:
                    resp = f(*args[:-2] + (request,))
                except HTTPException as e:
                    resp = t.cast("WSGIApplication", e.get_response(args[-2]))
                return resp(*args[-2:])

        return t.cast("WSGIApplication", application)

    def _get_file_stream(
        self,
        total_content_length: int | None,
        content_type: str | None,
        filename: str | None = None,
        content_length: int | None = None,
    ) -> t.IO[bytes]:
        """Called to get a stream for the file upload.

        This must provide a file-like class with `read()`, `readline()`
        and `seek()` methods that is both writeable and readable.

        The default implementation returns a temporary file if the total
        content length is higher than 500KB.  Because many browsers do not
        provide a content length for the files only the total content
        length matters.

        :param total_content_length: the total content length of all the
                                     data in the request combined.  This value
                                     is guaranteed to be there.
        :param content_type: the mimetype of the uploaded file.
        :param filename: the filename of the uploaded file.  May be `None`.
        :param content_length: the length of this file.  This value is usually
                               not provided because webbrowsers do not provide
                               this value.
        """
        return default_stream_factory(
            total_content_length=total_content_length,
            filename=filename,
            content_type=content_type,
            content_length=content_length,
        )

    @property
    def want_form_data_parsed(self) -> bool:
        """``True`` if the request method carries content. By default
        this is true if a ``Content-Type`` is sent.

        .. versionadded:: 0.8
        """
        return bool(self.environ.get("CONTENT_TYPE"))

    def make_form_data_parser(self) -> FormDataParser:
        """Creates the form data parser. Instantiates the
        :attr:`form_data_parser_class` with some parameters.

        .. versionadded:: 0.8
        """
        return self.form_data_parser_class(
            stream_factory=self._get_file_stream,
            max_form_memory_size=self.max_form_memory_size,
            max_content_length=self.max_content_length,
            max_form_parts=self.max_form_parts,
            cls=self.parameter_storage_class,
        )

    def _load_form_data(self) -> None:
        """Method used internally to retrieve submitted data.  After calling
        this sets `form` and `files` on the request object to multi dicts
        filled with the incoming form data.  As a matter of fact the input
        stream will be empty afterwards.  You can also call this method to
        force the parsing of the form data.

        .. versionadded:: 0.8
        """
        # abort early if we have already consumed the stream
        if "form" in self.__dict__:
            return

        if self.want_form_data_parsed:
            parser = self.make_form_data_parser()
            data = parser.parse(
                self._get_stream_for_parsing(),
                self.mimetype,
                self.content_length,
                self.mimetype_params,
            )
        else:
            data = (
                self.stream,
                self.parameter_storage_class(),
                self.parameter_storage_class(),
            )

        # inject the values into the instance dict so that we bypass
        # our cached_property non-data descriptor.
        d = self.__dict__
        d["stream"], d["form"], d["files"] = data

    def _get_stream_for_parsing(self) -> t.IO[bytes]:
        """This is the same as accessing :attr:`stream` with the difference
        that if it finds cached data from calling :meth:`get_data` first it
        will create a new stream out of the cached data.

        .. versionadded:: 0.9.3
        """
        cached_data = getattr(self, "_cached_data", None)
        if cached_data is not None:
            return BytesIO(cached_data)
        return self.stream

    def close(self) -> None:
        """Closes associated resources of this request object.  This
        closes all file handles explicitly.  You can also use the request
        object in a with statement which will automatically close it.

        .. versionadded:: 0.9
        """
        files = self.__dict__.get("files")
        for _key, value in iter_multi_items(files or ()):
            value.close()

    def __enter__(self) -> Request:
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:  # type: ignore
        self.close()

    @cached_property
    def stream(self) -> t.IO[bytes]:
        """The WSGI input stream, with safety checks. This stream can only be consumed
        once.

        Use :meth:`get_data` to get the full data as bytes or text. The :attr:`data`
        attribute will contain the full bytes only if they do not represent form data.
        The :attr:`form` attribute will contain the parsed form data in that case.

        Unlike :attr:`input_stream`, this stream guards against infinite streams or
        reading past :attr:`content_length` or :attr:`max_content_length`.

        If ``max_content_length`` is set, it can be enforced on streams if
        ``wsgi.input_terminated`` is set. Otherwise, an empty stream is returned.

        If the limit is reached before the underlying stream is exhausted (such as a
        file that is too large, or an infinite stream), the remaining contents of the
        stream cannot be read safely. Depending on how the server handles this, clients
        may show a "connection reset" failure instead of seeing the 413 response.

        .. versionchanged:: 2.3
            Check ``max_content_length`` preemptively and while reading.

        .. versionchanged:: 0.9
            The stream is always set (but may be consumed) even if form parsing was
            accessed first.
        """
        if self.shallow:
            raise RuntimeError(
                "This request was created with 'shallow=True', reading"
                " from the input stream is disabled."
            )

        return get_input_stream(
            self.environ, max_content_length=self.max_content_length
        )

    input_stream = environ_property[t.IO[bytes]](
        "wsgi.input",
        doc="""The raw WSGI input stream, without any safety checks.

        This is dangerous to use. It does not guard against infinite streams or reading
        past :attr:`content_length` or :attr:`max_content_length`.

        Use :attr:`stream` instead.
        """,
    )

    @cached_property
    def data(self) -> bytes:
        """The raw data read from :attr:`stream`. Will be empty if the request
        represents form data.

        To get the raw data even if it represents form data, use :meth:`get_data`.
        """
        return self.get_data(parse_form_data=True)

    @t.overload
    def get_data(
        self,
        cache: bool = True,
        as_text: t.Literal[False] = False,
        parse_form_data: bool = False,
    ) -> bytes: ...

    @t.overload
    def get_data(
        self,
        cache: bool = True,
        as_text: t.Literal[True] = ...,
        parse_form_data: bool = False,
    ) -> str: ...

    def get_data(
        self, cache: bool = True, as_text: bool = False, parse_form_data: bool = False
    ) -> bytes | str:
        """This reads the buffered incoming data from the client into one
        bytes object.  By default this is cached but that behavior can be
        changed by setting `cache` to `False`.

        Usually it's a bad idea to call this method without checking the
        content length first as a client could send dozens of megabytes or more
        to cause memory problems on the server.

        Note that if the form data was already parsed this method will not
        return anything as form data parsing does not cache the data like
        this method does.  To implicitly invoke form data parsing function
        set `parse_form_data` to `True`.  When this is done the return value
        of this method will be an empty string if the form parser handles
        the data.  This generally is not necessary as if the whole data is
        cached (which is the default) the form parser will used the cached
        data to parse the form data.  Please be generally aware of checking
        the content length first in any case before calling this method
        to avoid exhausting server memory.

        If `as_text` is set to `True` the return value will be a decoded
        string.

        .. versionadded:: 0.9
        """
        rv = getattr(self, "_cached_data", None)
        if rv is None:
            if parse_form_data:
                self._load_form_data()
            rv = self.stream.read()
            if cache:
                self._cached_data = rv
        if as_text:
            rv = rv.decode(errors="replace")
        return rv

    @cached_property
    def form(self) -> ImmutableMultiDict[str, str]:
        """The form parameters.  By default an
        :class:`~werkzeug.datastructures.ImmutableMultiDict`
        is returned from this function.  This can be changed by setting
        :attr:`parameter_storage_class` to a different type.  This might
        be necessary if the order of the form data is important.

        Please keep in mind that file uploads will not end up here, but instead
        in the :attr:`files` attribute.

        .. versionchanged:: 0.9

            Previous to Werkzeug 0.9 this would only contain form data for POST
            and PUT requests.
        """
        self._load_form_data()
        return self.form

    @cached_property
    def values(self) -> CombinedMultiDict[str, str]:
        """A :class:`werkzeug.datastructures.CombinedMultiDict` that
        combines :attr:`args` and :attr:`form`.

        For GET requests, only ``args`` are present, not ``form``.

        .. versionchanged:: 2.0
            For GET requests, only ``args`` are present, not ``form``.
        """
        sources = [self.args]

        if self.method != "GET":
            # GET requests can have a body, and some caching proxies
            # might not treat that differently than a normal GET
            # request, allowing form data to "invisibly" affect the
            # cache without indication in the query string / URL.
            sources.append(self.form)

        args = []

        for d in sources:
            if not isinstance(d, MultiDict):
                d = MultiDict(d)

            args.append(d)

        return CombinedMultiDict(args)

    @cached_property
    def files(self) -> ImmutableMultiDict[str, FileStorage]:
        """:class:`~werkzeug.datastructures.MultiDict` object containing
        all uploaded files.  Each key in :attr:`files` is the name from the
        ``<input type="file" name="">``.  Each value in :attr:`files` is a
        Werkzeug :class:`~werkzeug.datastructures.FileStorage` object.

        It basically behaves like a standard file object you know from Python,
        with the difference that it also has a
        :meth:`~werkzeug.datastructures.FileStorage.save` function that can
        store the file on the filesystem.

        Note that :attr:`files` will only contain data if the request method was
        POST, PUT or PATCH and the ``<form>`` that posted to the request had
        ``enctype="multipart/form-data"``.  It will be empty otherwise.

        See the :class:`~werkzeug.datastructures.MultiDict` /
        :class:`~werkzeug.datastructures.FileStorage` documentation for
        more details about the used data structure.
        """
        self._load_form_data()
        return self.files

    @property
    def script_root(self) -> str:
        """Alias for :attr:`self.root_path`. ``environ["SCRIPT_ROOT"]``
        without a trailing slash.
        """
        return self.root_path

    @cached_property
    def url_root(self) -> str:
        """Alias for :attr:`root_url`. The URL with scheme, host, and
        root path. For example, ``https://example.com/app/``.
        """
        return self.root_url

    remote_user = environ_property[str](
        "REMOTE_USER",
        doc="""If the server supports user authentication, and the
        script is protected, this attribute contains the username the
        user has authenticated as.""",
    )
    is_multithread = environ_property[bool](
        "wsgi.multithread",
        doc="""boolean that is `True` if the application is served by a
        multithreaded WSGI server.""",
    )
    is_multiprocess = environ_property[bool](
        "wsgi.multiprocess",
        doc="""boolean that is `True` if the application is served by a
        WSGI server that spawns multiple processes.""",
    )
    is_run_once = environ_property[bool](
        "wsgi.run_once",
        doc="""boolean that is `True` if the application will be
        executed only once in a process lifetime.  This is the case for
        CGI for example, but it's not guaranteed that the execution only
        happens one time.""",
    )

    # JSON

    #: A module or other object that has ``dumps`` and ``loads``
    #: functions that match the API of the built-in :mod:`json` module.
    json_module = json

    @property
    def json(self) -> t.Any | None:
        """The parsed JSON data if :attr:`mimetype` indicates JSON
        (:mimetype:`application/json`, see :attr:`is_json`).

        Calls :meth:`get_json` with default arguments.

        If the request content type is not ``application/json``, this
        will raise a 415 Unsupported Media Type error.

        .. versionchanged:: 2.3
            Raise a 415 error instead of 400.

        .. versionchanged:: 2.1
            Raise a 400 error if the content type is incorrect.
        """
        return self.get_json()

    # Cached values for ``(silent=False, silent=True)``. Initialized
    # with sentinel values.
    _cached_json: tuple[t.Any, t.Any] = (Ellipsis, Ellipsis)

    @t.overload
    def get_json(
        self, force: bool = ..., silent: t.Literal[False] = ..., cache: bool = ...
    ) -> t.Any: ...

    @t.overload
    def get_json(
        self, force: bool = ..., silent: bool = ..., cache: bool = ...
    ) -> t.Any | None: ...

    def get_json(
        self, force: bool = False, silent: bool = False, cache: bool = True
    ) -> t.Any | None:
        """Parse :attr:`data` as JSON.

        If the mimetype does not indicate JSON
        (:mimetype:`application/json`, see :attr:`is_json`), or parsing
        fails, :meth:`on_json_loading_failed` is called and
        its return value is used as the return value. By default this
        raises a 415 Unsupported Media Type resp.

        :param force: Ignore the mimetype and always try to parse JSON.
        :param silent: Silence mimetype and parsing errors, and
            return ``None`` instead.
        :param cache: Store the parsed JSON to return for subsequent
            calls.

        .. versionchanged:: 2.3
            Raise a 415 error instead of 400.

        .. versionchanged:: 2.1
            Raise a 400 error if the content type is incorrect.
        """
        if cache and self._cached_json[silent] is not Ellipsis:
            return self._cached_json[silent]

        if not (force or self.is_json):
            if not silent:
                return self.on_json_loading_failed(None)
            else:
                return None

        data = self.get_data(cache=cache)

        try:
            rv = self.json_module.loads(data)
        except ValueError as e:
            if silent:
                rv = None

                if cache:
                    normal_rv, _ = self._cached_json
                    self._cached_json = (normal_rv, rv)
            else:
                rv = self.on_json_loading_failed(e)

                if cache:
                    _, silent_rv = self._cached_json
                    self._cached_json = (rv, silent_rv)
        else:
            if cache:
                self._cached_json = (rv, rv)

        return rv

    def on_json_loading_failed(self, e: ValueError | None) -> t.Any:
        """Called if :meth:`get_json` fails and isn't silenced.

        If this method returns a value, it is used as the return value
        for :meth:`get_json`. The default implementation raises
        :exc:`~werkzeug.exceptions.BadRequest`.

        :param e: If parsing failed, this is the exception. It will be
            ``None`` if the content type wasn't ``application/json``.

        .. versionchanged:: 2.3
            Raise a 415 error instead of 400.
        """
        if e is not None:
            raise BadRequest(f"Failed to decode JSON object: {e}")

        raise UnsupportedMediaType(
            "Did not attempt to load JSON data because the request"
            " Content-Type was not 'application/json'."
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\accessibility.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Accessibility (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime


class AXNodeId(str):
    '''
    Unique accessibility node identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> AXNodeId:
        return cls(json)

    def __repr__(self):
        return 'AXNodeId({})'.format(super().__repr__())


class AXValueType(enum.Enum):
    '''
    Enum of possible property types.
    '''
    BOOLEAN = "boolean"
    TRISTATE = "tristate"
    BOOLEAN_OR_UNDEFINED = "booleanOrUndefined"
    IDREF = "idref"
    IDREF_LIST = "idrefList"
    INTEGER = "integer"
    NODE = "node"
    NODE_LIST = "nodeList"
    NUMBER = "number"
    STRING = "string"
    COMPUTED_STRING = "computedString"
    TOKEN = "token"
    TOKEN_LIST = "tokenList"
    DOM_RELATION = "domRelation"
    ROLE = "role"
    INTERNAL_ROLE = "internalRole"
    VALUE_UNDEFINED = "valueUndefined"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AXValueSourceType(enum.Enum):
    '''
    Enum of possible property sources.
    '''
    ATTRIBUTE = "attribute"
    IMPLICIT = "implicit"
    STYLE = "style"
    CONTENTS = "contents"
    PLACEHOLDER = "placeholder"
    RELATED_ELEMENT = "relatedElement"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AXValueNativeSourceType(enum.Enum):
    '''
    Enum of possible native property sources (as a subtype of a particular AXValueSourceType).
    '''
    DESCRIPTION = "description"
    FIGCAPTION = "figcaption"
    LABEL = "label"
    LABELFOR = "labelfor"
    LABELWRAPPED = "labelwrapped"
    LEGEND = "legend"
    RUBYANNOTATION = "rubyannotation"
    TABLECAPTION = "tablecaption"
    TITLE = "title"
    OTHER = "other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AXValueSource:
    '''
    A single source for a computed AX property.
    '''
    #: What type of source this is.
    type_: AXValueSourceType

    #: The value of this property source.
    value: typing.Optional[AXValue] = None

    #: The name of the relevant attribute, if any.
    attribute: typing.Optional[str] = None

    #: The value of the relevant attribute, if any.
    attribute_value: typing.Optional[AXValue] = None

    #: Whether this source is superseded by a higher priority source.
    superseded: typing.Optional[bool] = None

    #: The native markup source for this value, e.g. a ``<label>`` element.
    native_source: typing.Optional[AXValueNativeSourceType] = None

    #: The value, such as a node or node list, of the native source.
    native_source_value: typing.Optional[AXValue] = None

    #: Whether the value for this property is invalid.
    invalid: typing.Optional[bool] = None

    #: Reason for the value being invalid, if it is.
    invalid_reason: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.attribute is not None:
            json['attribute'] = self.attribute
        if self.attribute_value is not None:
            json['attributeValue'] = self.attribute_value.to_json()
        if self.superseded is not None:
            json['superseded'] = self.superseded
        if self.native_source is not None:
            json['nativeSource'] = self.native_source.to_json()
        if self.native_source_value is not None:
            json['nativeSourceValue'] = self.native_source_value.to_json()
        if self.invalid is not None:
            json['invalid'] = self.invalid
        if self.invalid_reason is not None:
            json['invalidReason'] = self.invalid_reason
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=AXValueSourceType.from_json(json['type']),
            value=AXValue.from_json(json['value']) if 'value' in json else None,
            attribute=str(json['attribute']) if 'attribute' in json else None,
            attribute_value=AXValue.from_json(json['attributeValue']) if 'attributeValue' in json else None,
            superseded=bool(json['superseded']) if 'superseded' in json else None,
            native_source=AXValueNativeSourceType.from_json(json['nativeSource']) if 'nativeSource' in json else None,
            native_source_value=AXValue.from_json(json['nativeSourceValue']) if 'nativeSourceValue' in json else None,
            invalid=bool(json['invalid']) if 'invalid' in json else None,
            invalid_reason=str(json['invalidReason']) if 'invalidReason' in json else None,
        )


@dataclass
class AXRelatedNode:
    #: The BackendNodeId of the related DOM node.
    backend_dom_node_id: dom.BackendNodeId

    #: The IDRef value provided, if any.
    idref: typing.Optional[str] = None

    #: The text alternative of this node in the current context.
    text: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['backendDOMNodeId'] = self.backend_dom_node_id.to_json()
        if self.idref is not None:
            json['idref'] = self.idref
        if self.text is not None:
            json['text'] = self.text
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            backend_dom_node_id=dom.BackendNodeId.from_json(json['backendDOMNodeId']),
            idref=str(json['idref']) if 'idref' in json else None,
            text=str(json['text']) if 'text' in json else None,
        )


@dataclass
class AXProperty:
    #: The name of this property.
    name: AXPropertyName

    #: The value of this property.
    value: AXValue

    def to_json(self):
        json = dict()
        json['name'] = self.name.to_json()
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=AXPropertyName.from_json(json['name']),
            value=AXValue.from_json(json['value']),
        )


@dataclass
class AXValue:
    '''
    A single computed AX property.
    '''
    #: The type of this value.
    type_: AXValueType

    #: The computed value of this property.
    value: typing.Optional[typing.Any] = None

    #: One or more related nodes, if applicable.
    related_nodes: typing.Optional[typing.List[AXRelatedNode]] = None

    #: The sources which contributed to the computation of this property.
    sources: typing.Optional[typing.List[AXValueSource]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.value is not None:
            json['value'] = self.value
        if self.related_nodes is not None:
            json['relatedNodes'] = [i.to_json() for i in self.related_nodes]
        if self.sources is not None:
            json['sources'] = [i.to_json() for i in self.sources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=AXValueType.from_json(json['type']),
            value=json['value'] if 'value' in json else None,
            related_nodes=[AXRelatedNode.from_json(i) for i in json['relatedNodes']] if 'relatedNodes' in json else None,
            sources=[AXValueSource.from_json(i) for i in json['sources']] if 'sources' in json else None,
        )


class AXPropertyName(enum.Enum):
    '''
    Values of AXProperty name:
    - from 'busy' to 'roledescription': states which apply to every AX node
    - from 'live' to 'root': attributes which apply to nodes in live regions
    - from 'autocomplete' to 'valuetext': attributes which apply to widgets
    - from 'checked' to 'selected': states which apply to widgets
    - from 'activedescendant' to 'owns' - relationships between elements other than parent/child/sibling.
    '''
    ACTIONS = "actions"
    BUSY = "busy"
    DISABLED = "disabled"
    EDITABLE = "editable"
    FOCUSABLE = "focusable"
    FOCUSED = "focused"
    HIDDEN = "hidden"
    HIDDEN_ROOT = "hiddenRoot"
    INVALID = "invalid"
    KEYSHORTCUTS = "keyshortcuts"
    SETTABLE = "settable"
    ROLEDESCRIPTION = "roledescription"
    LIVE = "live"
    ATOMIC = "atomic"
    RELEVANT = "relevant"
    ROOT = "root"
    AUTOCOMPLETE = "autocomplete"
    HAS_POPUP = "hasPopup"
    LEVEL = "level"
    MULTISELECTABLE = "multiselectable"
    ORIENTATION = "orientation"
    MULTILINE = "multiline"
    READONLY = "readonly"
    REQUIRED = "required"
    VALUEMIN = "valuemin"
    VALUEMAX = "valuemax"
    VALUETEXT = "valuetext"
    CHECKED = "checked"
    EXPANDED = "expanded"
    MODAL = "modal"
    PRESSED = "pressed"
    SELECTED = "selected"
    ACTIVEDESCENDANT = "activedescendant"
    CONTROLS = "controls"
    DESCRIBEDBY = "describedby"
    DETAILS = "details"
    ERRORMESSAGE = "errormessage"
    FLOWTO = "flowto"
    LABELLEDBY = "labelledby"
    OWNS = "owns"
    URL = "url"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AXNode:
    '''
    A node in the accessibility tree.
    '''
    #: Unique identifier for this node.
    node_id: AXNodeId

    #: Whether this node is ignored for accessibility
    ignored: bool

    #: Collection of reasons why this node is hidden.
    ignored_reasons: typing.Optional[typing.List[AXProperty]] = None

    #: This ``Node``'s role, whether explicit or implicit.
    role: typing.Optional[AXValue] = None

    #: This ``Node``'s Chrome raw role.
    chrome_role: typing.Optional[AXValue] = None

    #: The accessible name for this ``Node``.
    name: typing.Optional[AXValue] = None

    #: The accessible description for this ``Node``.
    description: typing.Optional[AXValue] = None

    #: The value for this ``Node``.
    value: typing.Optional[AXValue] = None

    #: All other properties
    properties: typing.Optional[typing.List[AXProperty]] = None

    #: ID for this node's parent.
    parent_id: typing.Optional[AXNodeId] = None

    #: IDs for each of this node's child nodes.
    child_ids: typing.Optional[typing.List[AXNodeId]] = None

    #: The backend ID for the associated DOM node, if any.
    backend_dom_node_id: typing.Optional[dom.BackendNodeId] = None

    #: The frame ID for the frame associated with this nodes document.
    frame_id: typing.Optional[page.FrameId] = None

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['ignored'] = self.ignored
        if self.ignored_reasons is not None:
            json['ignoredReasons'] = [i.to_json() for i in self.ignored_reasons]
        if self.role is not None:
            json['role'] = self.role.to_json()
        if self.chrome_role is not None:
            json['chromeRole'] = self.chrome_role.to_json()
        if self.name is not None:
            json['name'] = self.name.to_json()
        if self.description is not None:
            json['description'] = self.description.to_json()
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.properties is not None:
            json['properties'] = [i.to_json() for i in self.properties]
        if self.parent_id is not None:
            json['parentId'] = self.parent_id.to_json()
        if self.child_ids is not None:
            json['childIds'] = [i.to_json() for i in self.child_ids]
        if self.backend_dom_node_id is not None:
            json['backendDOMNodeId'] = self.backend_dom_node_id.to_json()
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_id=AXNodeId.from_json(json['nodeId']),
            ignored=bool(json['ignored']),
            ignored_reasons=[AXProperty.from_json(i) for i in json['ignoredReasons']] if 'ignoredReasons' in json else None,
            role=AXValue.from_json(json['role']) if 'role' in json else None,
            chrome_role=AXValue.from_json(json['chromeRole']) if 'chromeRole' in json else None,
            name=AXValue.from_json(json['name']) if 'name' in json else None,
            description=AXValue.from_json(json['description']) if 'description' in json else None,
            value=AXValue.from_json(json['value']) if 'value' in json else None,
            properties=[AXProperty.from_json(i) for i in json['properties']] if 'properties' in json else None,
            parent_id=AXNodeId.from_json(json['parentId']) if 'parentId' in json else None,
            child_ids=[AXNodeId.from_json(i) for i in json['childIds']] if 'childIds' in json else None,
            backend_dom_node_id=dom.BackendNodeId.from_json(json['backendDOMNodeId']) if 'backendDOMNodeId' in json else None,
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the accessibility domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the accessibility domain which causes ``AXNodeId``'s to remain consistent between method calls.
    This turns on accessibility for the page, which can impact performance until accessibility is disabled.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.enable',
    }
    json = yield cmd_dict


def get_partial_ax_tree(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        fetch_relatives: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches the accessibility node and partial accessibility tree for this DOM node, if it exists.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node to get the partial accessibility tree for.
    :param backend_node_id: *(Optional)* Identifier of the backend node to get the partial accessibility tree for.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper to get the partial accessibility tree for.
    :param fetch_relatives: *(Optional)* Whether to fetch this node's ancestors, siblings and children. Defaults to true.
    :returns: The ``Accessibility.AXNode`` for this DOM node, if it exists, plus its ancestors, siblings and children, if requested.
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if fetch_relatives is not None:
        params['fetchRelatives'] = fetch_relatives
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getPartialAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_full_ax_tree(
        depth: typing.Optional[int] = None,
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches the entire accessibility tree for the root Document

    **EXPERIMENTAL**

    :param depth: *(Optional)* The maximum depth at which descendants of the root node should be retrieved. If omitted, the full tree is returned.
    :param frame_id: *(Optional)* The frame for whose document the AX tree should be retrieved. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if depth is not None:
        params['depth'] = depth
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getFullAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_root_ax_node(
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,AXNode]:
    '''
    Fetches the root node.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param frame_id: *(Optional)* The frame in whose document the node resides. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getRootAXNode',
        'params': params,
    }
    json = yield cmd_dict
    return AXNode.from_json(json['node'])


def get_ax_node_and_ancestors(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches a node and all ancestors up to and including the root.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node to get.
    :param backend_node_id: *(Optional)* Identifier of the backend node to get.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper to get.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getAXNodeAndAncestors',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_child_ax_nodes(
        id_: AXNodeId,
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches a particular accessibility node by AXNodeId.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param id_:
    :param frame_id: *(Optional)* The frame in whose document the node resides. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getChildAXNodes',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def query_ax_tree(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        accessible_name: typing.Optional[str] = None,
        role: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Query a DOM node's accessibility subtree for accessible name and role.
    This command computes the name and role for all nodes in the subtree, including those that are
    ignored for accessibility, and returns those that match the specified name and role. If no DOM
    node is specified, or the DOM node does not exist, the command returns an error. If neither
    ``accessibleName`` or ``role`` is specified, it returns all the accessibility nodes in the subtree.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node for the root to query.
    :param backend_node_id: *(Optional)* Identifier of the backend node for the root to query.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper for the root to query.
    :param accessible_name: *(Optional)* Find nodes with this computed name.
    :param role: *(Optional)* Find nodes with this computed role.
    :returns: A list of ``Accessibility.AXNode`` matching the specified attributes, including nodes that are ignored for accessibility.
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if accessible_name is not None:
        params['accessibleName'] = accessible_name
    if role is not None:
        params['role'] = role
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.queryAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


@event_class('Accessibility.loadComplete')
@dataclass
class LoadComplete:
    '''
    **EXPERIMENTAL**

    The loadComplete event mirrors the load complete event sent by the browser to assistive
    technology when the web page has finished loading.
    '''
    #: New document root node.
    root: AXNode

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadComplete:
        return cls(
            root=AXNode.from_json(json['root'])
        )


@event_class('Accessibility.nodesUpdated')
@dataclass
class NodesUpdated:
    '''
    **EXPERIMENTAL**

    The nodesUpdated event is sent every time a previously requested node has changed the in tree.
    '''
    #: Updated node data.
    nodes: typing.List[AXNode]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesUpdated:
        return cls(
            nodes=[AXNode.from_json(i) for i in json['nodes']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\accessibility.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Accessibility (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime


class AXNodeId(str):
    '''
    Unique accessibility node identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> AXNodeId:
        return cls(json)

    def __repr__(self):
        return 'AXNodeId({})'.format(super().__repr__())


class AXValueType(enum.Enum):
    '''
    Enum of possible property types.
    '''
    BOOLEAN = "boolean"
    TRISTATE = "tristate"
    BOOLEAN_OR_UNDEFINED = "booleanOrUndefined"
    IDREF = "idref"
    IDREF_LIST = "idrefList"
    INTEGER = "integer"
    NODE = "node"
    NODE_LIST = "nodeList"
    NUMBER = "number"
    STRING = "string"
    COMPUTED_STRING = "computedString"
    TOKEN = "token"
    TOKEN_LIST = "tokenList"
    DOM_RELATION = "domRelation"
    ROLE = "role"
    INTERNAL_ROLE = "internalRole"
    VALUE_UNDEFINED = "valueUndefined"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AXValueSourceType(enum.Enum):
    '''
    Enum of possible property sources.
    '''
    ATTRIBUTE = "attribute"
    IMPLICIT = "implicit"
    STYLE = "style"
    CONTENTS = "contents"
    PLACEHOLDER = "placeholder"
    RELATED_ELEMENT = "relatedElement"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AXValueNativeSourceType(enum.Enum):
    '''
    Enum of possible native property sources (as a subtype of a particular AXValueSourceType).
    '''
    DESCRIPTION = "description"
    FIGCAPTION = "figcaption"
    LABEL = "label"
    LABELFOR = "labelfor"
    LABELWRAPPED = "labelwrapped"
    LEGEND = "legend"
    RUBYANNOTATION = "rubyannotation"
    TABLECAPTION = "tablecaption"
    TITLE = "title"
    OTHER = "other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AXValueSource:
    '''
    A single source for a computed AX property.
    '''
    #: What type of source this is.
    type_: AXValueSourceType

    #: The value of this property source.
    value: typing.Optional[AXValue] = None

    #: The name of the relevant attribute, if any.
    attribute: typing.Optional[str] = None

    #: The value of the relevant attribute, if any.
    attribute_value: typing.Optional[AXValue] = None

    #: Whether this source is superseded by a higher priority source.
    superseded: typing.Optional[bool] = None

    #: The native markup source for this value, e.g. a ``<label>`` element.
    native_source: typing.Optional[AXValueNativeSourceType] = None

    #: The value, such as a node or node list, of the native source.
    native_source_value: typing.Optional[AXValue] = None

    #: Whether the value for this property is invalid.
    invalid: typing.Optional[bool] = None

    #: Reason for the value being invalid, if it is.
    invalid_reason: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.attribute is not None:
            json['attribute'] = self.attribute
        if self.attribute_value is not None:
            json['attributeValue'] = self.attribute_value.to_json()
        if self.superseded is not None:
            json['superseded'] = self.superseded
        if self.native_source is not None:
            json['nativeSource'] = self.native_source.to_json()
        if self.native_source_value is not None:
            json['nativeSourceValue'] = self.native_source_value.to_json()
        if self.invalid is not None:
            json['invalid'] = self.invalid
        if self.invalid_reason is not None:
            json['invalidReason'] = self.invalid_reason
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=AXValueSourceType.from_json(json['type']),
            value=AXValue.from_json(json['value']) if 'value' in json else None,
            attribute=str(json['attribute']) if 'attribute' in json else None,
            attribute_value=AXValue.from_json(json['attributeValue']) if 'attributeValue' in json else None,
            superseded=bool(json['superseded']) if 'superseded' in json else None,
            native_source=AXValueNativeSourceType.from_json(json['nativeSource']) if 'nativeSource' in json else None,
            native_source_value=AXValue.from_json(json['nativeSourceValue']) if 'nativeSourceValue' in json else None,
            invalid=bool(json['invalid']) if 'invalid' in json else None,
            invalid_reason=str(json['invalidReason']) if 'invalidReason' in json else None,
        )


@dataclass
class AXRelatedNode:
    #: The BackendNodeId of the related DOM node.
    backend_dom_node_id: dom.BackendNodeId

    #: The IDRef value provided, if any.
    idref: typing.Optional[str] = None

    #: The text alternative of this node in the current context.
    text: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['backendDOMNodeId'] = self.backend_dom_node_id.to_json()
        if self.idref is not None:
            json['idref'] = self.idref
        if self.text is not None:
            json['text'] = self.text
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            backend_dom_node_id=dom.BackendNodeId.from_json(json['backendDOMNodeId']),
            idref=str(json['idref']) if 'idref' in json else None,
            text=str(json['text']) if 'text' in json else None,
        )


@dataclass
class AXProperty:
    #: The name of this property.
    name: AXPropertyName

    #: The value of this property.
    value: AXValue

    def to_json(self):
        json = dict()
        json['name'] = self.name.to_json()
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=AXPropertyName.from_json(json['name']),
            value=AXValue.from_json(json['value']),
        )


@dataclass
class AXValue:
    '''
    A single computed AX property.
    '''
    #: The type of this value.
    type_: AXValueType

    #: The computed value of this property.
    value: typing.Optional[typing.Any] = None

    #: One or more related nodes, if applicable.
    related_nodes: typing.Optional[typing.List[AXRelatedNode]] = None

    #: The sources which contributed to the computation of this property.
    sources: typing.Optional[typing.List[AXValueSource]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.value is not None:
            json['value'] = self.value
        if self.related_nodes is not None:
            json['relatedNodes'] = [i.to_json() for i in self.related_nodes]
        if self.sources is not None:
            json['sources'] = [i.to_json() for i in self.sources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=AXValueType.from_json(json['type']),
            value=json['value'] if 'value' in json else None,
            related_nodes=[AXRelatedNode.from_json(i) for i in json['relatedNodes']] if 'relatedNodes' in json else None,
            sources=[AXValueSource.from_json(i) for i in json['sources']] if 'sources' in json else None,
        )


class AXPropertyName(enum.Enum):
    '''
    Values of AXProperty name:
    - from 'busy' to 'roledescription': states which apply to every AX node
    - from 'live' to 'root': attributes which apply to nodes in live regions
    - from 'autocomplete' to 'valuetext': attributes which apply to widgets
    - from 'checked' to 'selected': states which apply to widgets
    - from 'activedescendant' to 'owns' - relationships between elements other than parent/child/sibling.
    '''
    ACTIONS = "actions"
    BUSY = "busy"
    DISABLED = "disabled"
    EDITABLE = "editable"
    FOCUSABLE = "focusable"
    FOCUSED = "focused"
    HIDDEN = "hidden"
    HIDDEN_ROOT = "hiddenRoot"
    INVALID = "invalid"
    KEYSHORTCUTS = "keyshortcuts"
    SETTABLE = "settable"
    ROLEDESCRIPTION = "roledescription"
    LIVE = "live"
    ATOMIC = "atomic"
    RELEVANT = "relevant"
    ROOT = "root"
    AUTOCOMPLETE = "autocomplete"
    HAS_POPUP = "hasPopup"
    LEVEL = "level"
    MULTISELECTABLE = "multiselectable"
    ORIENTATION = "orientation"
    MULTILINE = "multiline"
    READONLY = "readonly"
    REQUIRED = "required"
    VALUEMIN = "valuemin"
    VALUEMAX = "valuemax"
    VALUETEXT = "valuetext"
    CHECKED = "checked"
    EXPANDED = "expanded"
    MODAL = "modal"
    PRESSED = "pressed"
    SELECTED = "selected"
    ACTIVEDESCENDANT = "activedescendant"
    CONTROLS = "controls"
    DESCRIBEDBY = "describedby"
    DETAILS = "details"
    ERRORMESSAGE = "errormessage"
    FLOWTO = "flowto"
    LABELLEDBY = "labelledby"
    OWNS = "owns"
    URL = "url"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AXNode:
    '''
    A node in the accessibility tree.
    '''
    #: Unique identifier for this node.
    node_id: AXNodeId

    #: Whether this node is ignored for accessibility
    ignored: bool

    #: Collection of reasons why this node is hidden.
    ignored_reasons: typing.Optional[typing.List[AXProperty]] = None

    #: This ``Node``'s role, whether explicit or implicit.
    role: typing.Optional[AXValue] = None

    #: This ``Node``'s Chrome raw role.
    chrome_role: typing.Optional[AXValue] = None

    #: The accessible name for this ``Node``.
    name: typing.Optional[AXValue] = None

    #: The accessible description for this ``Node``.
    description: typing.Optional[AXValue] = None

    #: The value for this ``Node``.
    value: typing.Optional[AXValue] = None

    #: All other properties
    properties: typing.Optional[typing.List[AXProperty]] = None

    #: ID for this node's parent.
    parent_id: typing.Optional[AXNodeId] = None

    #: IDs for each of this node's child nodes.
    child_ids: typing.Optional[typing.List[AXNodeId]] = None

    #: The backend ID for the associated DOM node, if any.
    backend_dom_node_id: typing.Optional[dom.BackendNodeId] = None

    #: The frame ID for the frame associated with this nodes document.
    frame_id: typing.Optional[page.FrameId] = None

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['ignored'] = self.ignored
        if self.ignored_reasons is not None:
            json['ignoredReasons'] = [i.to_json() for i in self.ignored_reasons]
        if self.role is not None:
            json['role'] = self.role.to_json()
        if self.chrome_role is not None:
            json['chromeRole'] = self.chrome_role.to_json()
        if self.name is not None:
            json['name'] = self.name.to_json()
        if self.description is not None:
            json['description'] = self.description.to_json()
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.properties is not None:
            json['properties'] = [i.to_json() for i in self.properties]
        if self.parent_id is not None:
            json['parentId'] = self.parent_id.to_json()
        if self.child_ids is not None:
            json['childIds'] = [i.to_json() for i in self.child_ids]
        if self.backend_dom_node_id is not None:
            json['backendDOMNodeId'] = self.backend_dom_node_id.to_json()
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_id=AXNodeId.from_json(json['nodeId']),
            ignored=bool(json['ignored']),
            ignored_reasons=[AXProperty.from_json(i) for i in json['ignoredReasons']] if 'ignoredReasons' in json else None,
            role=AXValue.from_json(json['role']) if 'role' in json else None,
            chrome_role=AXValue.from_json(json['chromeRole']) if 'chromeRole' in json else None,
            name=AXValue.from_json(json['name']) if 'name' in json else None,
            description=AXValue.from_json(json['description']) if 'description' in json else None,
            value=AXValue.from_json(json['value']) if 'value' in json else None,
            properties=[AXProperty.from_json(i) for i in json['properties']] if 'properties' in json else None,
            parent_id=AXNodeId.from_json(json['parentId']) if 'parentId' in json else None,
            child_ids=[AXNodeId.from_json(i) for i in json['childIds']] if 'childIds' in json else None,
            backend_dom_node_id=dom.BackendNodeId.from_json(json['backendDOMNodeId']) if 'backendDOMNodeId' in json else None,
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the accessibility domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the accessibility domain which causes ``AXNodeId``'s to remain consistent between method calls.
    This turns on accessibility for the page, which can impact performance until accessibility is disabled.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.enable',
    }
    json = yield cmd_dict


def get_partial_ax_tree(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        fetch_relatives: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches the accessibility node and partial accessibility tree for this DOM node, if it exists.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node to get the partial accessibility tree for.
    :param backend_node_id: *(Optional)* Identifier of the backend node to get the partial accessibility tree for.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper to get the partial accessibility tree for.
    :param fetch_relatives: *(Optional)* Whether to fetch this node's ancestors, siblings and children. Defaults to true.
    :returns: The ``Accessibility.AXNode`` for this DOM node, if it exists, plus its ancestors, siblings and children, if requested.
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if fetch_relatives is not None:
        params['fetchRelatives'] = fetch_relatives
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getPartialAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_full_ax_tree(
        depth: typing.Optional[int] = None,
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches the entire accessibility tree for the root Document

    **EXPERIMENTAL**

    :param depth: *(Optional)* The maximum depth at which descendants of the root node should be retrieved. If omitted, the full tree is returned.
    :param frame_id: *(Optional)* The frame for whose document the AX tree should be retrieved. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if depth is not None:
        params['depth'] = depth
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getFullAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_root_ax_node(
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,AXNode]:
    '''
    Fetches the root node.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param frame_id: *(Optional)* The frame in whose document the node resides. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getRootAXNode',
        'params': params,
    }
    json = yield cmd_dict
    return AXNode.from_json(json['node'])


def get_ax_node_and_ancestors(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches a node and all ancestors up to and including the root.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node to get.
    :param backend_node_id: *(Optional)* Identifier of the backend node to get.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper to get.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getAXNodeAndAncestors',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_child_ax_nodes(
        id_: AXNodeId,
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches a particular accessibility node by AXNodeId.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param id_:
    :param frame_id: *(Optional)* The frame in whose document the node resides. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getChildAXNodes',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def query_ax_tree(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        accessible_name: typing.Optional[str] = None,
        role: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Query a DOM node's accessibility subtree for accessible name and role.
    This command computes the name and role for all nodes in the subtree, including those that are
    ignored for accessibility, and returns those that match the specified name and role. If no DOM
    node is specified, or the DOM node does not exist, the command returns an error. If neither
    ``accessibleName`` or ``role`` is specified, it returns all the accessibility nodes in the subtree.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node for the root to query.
    :param backend_node_id: *(Optional)* Identifier of the backend node for the root to query.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper for the root to query.
    :param accessible_name: *(Optional)* Find nodes with this computed name.
    :param role: *(Optional)* Find nodes with this computed role.
    :returns: A list of ``Accessibility.AXNode`` matching the specified attributes, including nodes that are ignored for accessibility.
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if accessible_name is not None:
        params['accessibleName'] = accessible_name
    if role is not None:
        params['role'] = role
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.queryAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


@event_class('Accessibility.loadComplete')
@dataclass
class LoadComplete:
    '''
    **EXPERIMENTAL**

    The loadComplete event mirrors the load complete event sent by the browser to assistive
    technology when the web page has finished loading.
    '''
    #: New document root node.
    root: AXNode

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadComplete:
        return cls(
            root=AXNode.from_json(json['root'])
        )


@event_class('Accessibility.nodesUpdated')
@dataclass
class NodesUpdated:
    '''
    **EXPERIMENTAL**

    The nodesUpdated event is sent every time a previously requested node has changed the in tree.
    '''
    #: Updated node data.
    nodes: typing.List[AXNode]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesUpdated:
        return cls(
            nodes=[AXNode.from_json(i) for i in json['nodes']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\accessibility.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Accessibility (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime


class AXNodeId(str):
    '''
    Unique accessibility node identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> AXNodeId:
        return cls(json)

    def __repr__(self):
        return 'AXNodeId({})'.format(super().__repr__())


class AXValueType(enum.Enum):
    '''
    Enum of possible property types.
    '''
    BOOLEAN = "boolean"
    TRISTATE = "tristate"
    BOOLEAN_OR_UNDEFINED = "booleanOrUndefined"
    IDREF = "idref"
    IDREF_LIST = "idrefList"
    INTEGER = "integer"
    NODE = "node"
    NODE_LIST = "nodeList"
    NUMBER = "number"
    STRING = "string"
    COMPUTED_STRING = "computedString"
    TOKEN = "token"
    TOKEN_LIST = "tokenList"
    DOM_RELATION = "domRelation"
    ROLE = "role"
    INTERNAL_ROLE = "internalRole"
    VALUE_UNDEFINED = "valueUndefined"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AXValueSourceType(enum.Enum):
    '''
    Enum of possible property sources.
    '''
    ATTRIBUTE = "attribute"
    IMPLICIT = "implicit"
    STYLE = "style"
    CONTENTS = "contents"
    PLACEHOLDER = "placeholder"
    RELATED_ELEMENT = "relatedElement"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AXValueNativeSourceType(enum.Enum):
    '''
    Enum of possible native property sources (as a subtype of a particular AXValueSourceType).
    '''
    DESCRIPTION = "description"
    FIGCAPTION = "figcaption"
    LABEL = "label"
    LABELFOR = "labelfor"
    LABELWRAPPED = "labelwrapped"
    LEGEND = "legend"
    RUBYANNOTATION = "rubyannotation"
    TABLECAPTION = "tablecaption"
    TITLE = "title"
    OTHER = "other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AXValueSource:
    '''
    A single source for a computed AX property.
    '''
    #: What type of source this is.
    type_: AXValueSourceType

    #: The value of this property source.
    value: typing.Optional[AXValue] = None

    #: The name of the relevant attribute, if any.
    attribute: typing.Optional[str] = None

    #: The value of the relevant attribute, if any.
    attribute_value: typing.Optional[AXValue] = None

    #: Whether this source is superseded by a higher priority source.
    superseded: typing.Optional[bool] = None

    #: The native markup source for this value, e.g. a ``<label>`` element.
    native_source: typing.Optional[AXValueNativeSourceType] = None

    #: The value, such as a node or node list, of the native source.
    native_source_value: typing.Optional[AXValue] = None

    #: Whether the value for this property is invalid.
    invalid: typing.Optional[bool] = None

    #: Reason for the value being invalid, if it is.
    invalid_reason: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.attribute is not None:
            json['attribute'] = self.attribute
        if self.attribute_value is not None:
            json['attributeValue'] = self.attribute_value.to_json()
        if self.superseded is not None:
            json['superseded'] = self.superseded
        if self.native_source is not None:
            json['nativeSource'] = self.native_source.to_json()
        if self.native_source_value is not None:
            json['nativeSourceValue'] = self.native_source_value.to_json()
        if self.invalid is not None:
            json['invalid'] = self.invalid
        if self.invalid_reason is not None:
            json['invalidReason'] = self.invalid_reason
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=AXValueSourceType.from_json(json['type']),
            value=AXValue.from_json(json['value']) if 'value' in json else None,
            attribute=str(json['attribute']) if 'attribute' in json else None,
            attribute_value=AXValue.from_json(json['attributeValue']) if 'attributeValue' in json else None,
            superseded=bool(json['superseded']) if 'superseded' in json else None,
            native_source=AXValueNativeSourceType.from_json(json['nativeSource']) if 'nativeSource' in json else None,
            native_source_value=AXValue.from_json(json['nativeSourceValue']) if 'nativeSourceValue' in json else None,
            invalid=bool(json['invalid']) if 'invalid' in json else None,
            invalid_reason=str(json['invalidReason']) if 'invalidReason' in json else None,
        )


@dataclass
class AXRelatedNode:
    #: The BackendNodeId of the related DOM node.
    backend_dom_node_id: dom.BackendNodeId

    #: The IDRef value provided, if any.
    idref: typing.Optional[str] = None

    #: The text alternative of this node in the current context.
    text: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['backendDOMNodeId'] = self.backend_dom_node_id.to_json()
        if self.idref is not None:
            json['idref'] = self.idref
        if self.text is not None:
            json['text'] = self.text
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            backend_dom_node_id=dom.BackendNodeId.from_json(json['backendDOMNodeId']),
            idref=str(json['idref']) if 'idref' in json else None,
            text=str(json['text']) if 'text' in json else None,
        )


@dataclass
class AXProperty:
    #: The name of this property.
    name: AXPropertyName

    #: The value of this property.
    value: AXValue

    def to_json(self):
        json = dict()
        json['name'] = self.name.to_json()
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=AXPropertyName.from_json(json['name']),
            value=AXValue.from_json(json['value']),
        )


@dataclass
class AXValue:
    '''
    A single computed AX property.
    '''
    #: The type of this value.
    type_: AXValueType

    #: The computed value of this property.
    value: typing.Optional[typing.Any] = None

    #: One or more related nodes, if applicable.
    related_nodes: typing.Optional[typing.List[AXRelatedNode]] = None

    #: The sources which contributed to the computation of this property.
    sources: typing.Optional[typing.List[AXValueSource]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.value is not None:
            json['value'] = self.value
        if self.related_nodes is not None:
            json['relatedNodes'] = [i.to_json() for i in self.related_nodes]
        if self.sources is not None:
            json['sources'] = [i.to_json() for i in self.sources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=AXValueType.from_json(json['type']),
            value=json['value'] if 'value' in json else None,
            related_nodes=[AXRelatedNode.from_json(i) for i in json['relatedNodes']] if 'relatedNodes' in json else None,
            sources=[AXValueSource.from_json(i) for i in json['sources']] if 'sources' in json else None,
        )


class AXPropertyName(enum.Enum):
    '''
    Values of AXProperty name:
    - from 'busy' to 'roledescription': states which apply to every AX node
    - from 'live' to 'root': attributes which apply to nodes in live regions
    - from 'autocomplete' to 'valuetext': attributes which apply to widgets
    - from 'checked' to 'selected': states which apply to widgets
    - from 'activedescendant' to 'owns' - relationships between elements other than parent/child/sibling.
    '''
    ACTIONS = "actions"
    BUSY = "busy"
    DISABLED = "disabled"
    EDITABLE = "editable"
    FOCUSABLE = "focusable"
    FOCUSED = "focused"
    HIDDEN = "hidden"
    HIDDEN_ROOT = "hiddenRoot"
    INVALID = "invalid"
    KEYSHORTCUTS = "keyshortcuts"
    SETTABLE = "settable"
    ROLEDESCRIPTION = "roledescription"
    LIVE = "live"
    ATOMIC = "atomic"
    RELEVANT = "relevant"
    ROOT = "root"
    AUTOCOMPLETE = "autocomplete"
    HAS_POPUP = "hasPopup"
    LEVEL = "level"
    MULTISELECTABLE = "multiselectable"
    ORIENTATION = "orientation"
    MULTILINE = "multiline"
    READONLY = "readonly"
    REQUIRED = "required"
    VALUEMIN = "valuemin"
    VALUEMAX = "valuemax"
    VALUETEXT = "valuetext"
    CHECKED = "checked"
    EXPANDED = "expanded"
    MODAL = "modal"
    PRESSED = "pressed"
    SELECTED = "selected"
    ACTIVEDESCENDANT = "activedescendant"
    CONTROLS = "controls"
    DESCRIBEDBY = "describedby"
    DETAILS = "details"
    ERRORMESSAGE = "errormessage"
    FLOWTO = "flowto"
    LABELLEDBY = "labelledby"
    OWNS = "owns"
    URL = "url"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AXNode:
    '''
    A node in the accessibility tree.
    '''
    #: Unique identifier for this node.
    node_id: AXNodeId

    #: Whether this node is ignored for accessibility
    ignored: bool

    #: Collection of reasons why this node is hidden.
    ignored_reasons: typing.Optional[typing.List[AXProperty]] = None

    #: This ``Node``'s role, whether explicit or implicit.
    role: typing.Optional[AXValue] = None

    #: This ``Node``'s Chrome raw role.
    chrome_role: typing.Optional[AXValue] = None

    #: The accessible name for this ``Node``.
    name: typing.Optional[AXValue] = None

    #: The accessible description for this ``Node``.
    description: typing.Optional[AXValue] = None

    #: The value for this ``Node``.
    value: typing.Optional[AXValue] = None

    #: All other properties
    properties: typing.Optional[typing.List[AXProperty]] = None

    #: ID for this node's parent.
    parent_id: typing.Optional[AXNodeId] = None

    #: IDs for each of this node's child nodes.
    child_ids: typing.Optional[typing.List[AXNodeId]] = None

    #: The backend ID for the associated DOM node, if any.
    backend_dom_node_id: typing.Optional[dom.BackendNodeId] = None

    #: The frame ID for the frame associated with this nodes document.
    frame_id: typing.Optional[page.FrameId] = None

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['ignored'] = self.ignored
        if self.ignored_reasons is not None:
            json['ignoredReasons'] = [i.to_json() for i in self.ignored_reasons]
        if self.role is not None:
            json['role'] = self.role.to_json()
        if self.chrome_role is not None:
            json['chromeRole'] = self.chrome_role.to_json()
        if self.name is not None:
            json['name'] = self.name.to_json()
        if self.description is not None:
            json['description'] = self.description.to_json()
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.properties is not None:
            json['properties'] = [i.to_json() for i in self.properties]
        if self.parent_id is not None:
            json['parentId'] = self.parent_id.to_json()
        if self.child_ids is not None:
            json['childIds'] = [i.to_json() for i in self.child_ids]
        if self.backend_dom_node_id is not None:
            json['backendDOMNodeId'] = self.backend_dom_node_id.to_json()
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_id=AXNodeId.from_json(json['nodeId']),
            ignored=bool(json['ignored']),
            ignored_reasons=[AXProperty.from_json(i) for i in json['ignoredReasons']] if 'ignoredReasons' in json else None,
            role=AXValue.from_json(json['role']) if 'role' in json else None,
            chrome_role=AXValue.from_json(json['chromeRole']) if 'chromeRole' in json else None,
            name=AXValue.from_json(json['name']) if 'name' in json else None,
            description=AXValue.from_json(json['description']) if 'description' in json else None,
            value=AXValue.from_json(json['value']) if 'value' in json else None,
            properties=[AXProperty.from_json(i) for i in json['properties']] if 'properties' in json else None,
            parent_id=AXNodeId.from_json(json['parentId']) if 'parentId' in json else None,
            child_ids=[AXNodeId.from_json(i) for i in json['childIds']] if 'childIds' in json else None,
            backend_dom_node_id=dom.BackendNodeId.from_json(json['backendDOMNodeId']) if 'backendDOMNodeId' in json else None,
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the accessibility domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the accessibility domain which causes ``AXNodeId``'s to remain consistent between method calls.
    This turns on accessibility for the page, which can impact performance until accessibility is disabled.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.enable',
    }
    json = yield cmd_dict


def get_partial_ax_tree(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        fetch_relatives: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches the accessibility node and partial accessibility tree for this DOM node, if it exists.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node to get the partial accessibility tree for.
    :param backend_node_id: *(Optional)* Identifier of the backend node to get the partial accessibility tree for.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper to get the partial accessibility tree for.
    :param fetch_relatives: *(Optional)* Whether to fetch this node's ancestors, siblings and children. Defaults to true.
    :returns: The ``Accessibility.AXNode`` for this DOM node, if it exists, plus its ancestors, siblings and children, if requested.
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if fetch_relatives is not None:
        params['fetchRelatives'] = fetch_relatives
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getPartialAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_full_ax_tree(
        depth: typing.Optional[int] = None,
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches the entire accessibility tree for the root Document

    **EXPERIMENTAL**

    :param depth: *(Optional)* The maximum depth at which descendants of the root node should be retrieved. If omitted, the full tree is returned.
    :param frame_id: *(Optional)* The frame for whose document the AX tree should be retrieved. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if depth is not None:
        params['depth'] = depth
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getFullAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_root_ax_node(
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,AXNode]:
    '''
    Fetches the root node.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param frame_id: *(Optional)* The frame in whose document the node resides. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getRootAXNode',
        'params': params,
    }
    json = yield cmd_dict
    return AXNode.from_json(json['node'])


def get_ax_node_and_ancestors(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches a node and all ancestors up to and including the root.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node to get.
    :param backend_node_id: *(Optional)* Identifier of the backend node to get.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper to get.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getAXNodeAndAncestors',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def get_child_ax_nodes(
        id_: AXNodeId,
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Fetches a particular accessibility node by AXNodeId.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param id_:
    :param frame_id: *(Optional)* The frame in whose document the node resides. If omitted, the root frame is used.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.getChildAXNodes',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


def query_ax_tree(
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        accessible_name: typing.Optional[str] = None,
        role: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[AXNode]]:
    '''
    Query a DOM node's accessibility subtree for accessible name and role.
    This command computes the name and role for all nodes in the subtree, including those that are
    ignored for accessibility, and returns those that match the specified name and role. If no DOM
    node is specified, or the DOM node does not exist, the command returns an error. If neither
    ``accessibleName`` or ``role`` is specified, it returns all the accessibility nodes in the subtree.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node for the root to query.
    :param backend_node_id: *(Optional)* Identifier of the backend node for the root to query.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper for the root to query.
    :param accessible_name: *(Optional)* Find nodes with this computed name.
    :param role: *(Optional)* Find nodes with this computed role.
    :returns: A list of ``Accessibility.AXNode`` matching the specified attributes, including nodes that are ignored for accessibility.
    '''
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if accessible_name is not None:
        params['accessibleName'] = accessible_name
    if role is not None:
        params['role'] = role
    cmd_dict: T_JSON_DICT = {
        'method': 'Accessibility.queryAXTree',
        'params': params,
    }
    json = yield cmd_dict
    return [AXNode.from_json(i) for i in json['nodes']]


@event_class('Accessibility.loadComplete')
@dataclass
class LoadComplete:
    '''
    **EXPERIMENTAL**

    The loadComplete event mirrors the load complete event sent by the browser to assistive
    technology when the web page has finished loading.
    '''
    #: New document root node.
    root: AXNode

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadComplete:
        return cls(
            root=AXNode.from_json(json['root'])
        )


@event_class('Accessibility.nodesUpdated')
@dataclass
class NodesUpdated:
    '''
    **EXPERIMENTAL**

    The nodesUpdated event is sent every time a previously requested node has changed the in tree.
    '''
    #: Updated node data.
    nodes: typing.List[AXNode]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesUpdated:
        return cls(
            nodes=[AXNode.from_json(i) for i in json['nodes']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\calculus\differentiation.py ===
from ..libmp.backend import xrange
from .calculus import defun

try:
    iteritems = dict.iteritems
except AttributeError:
    iteritems = dict.items

#----------------------------------------------------------------------------#
#                                Differentiation                             #
#----------------------------------------------------------------------------#

@defun
def difference(ctx, s, n):
    r"""
    Given a sequence `(s_k)` containing at least `n+1` items, returns the
    `n`-th forward difference,

    .. math ::

        \Delta^n = \sum_{k=0}^{\infty} (-1)^{k+n} {n \choose k} s_k.
    """
    n = int(n)
    d = ctx.zero
    b = (-1) ** (n & 1)
    for k in xrange(n+1):
        d += b * s[k]
        b = (b * (k-n)) // (k+1)
    return d

def hsteps(ctx, f, x, n, prec, **options):
    singular = options.get('singular')
    addprec = options.get('addprec', 10)
    direction = options.get('direction', 0)
    workprec = (prec+2*addprec) * (n+1)
    orig = ctx.prec
    try:
        ctx.prec = workprec
        h = options.get('h')
        if h is None:
            if options.get('relative'):
                hextramag = int(ctx.mag(x))
            else:
                hextramag = 0
            h = ctx.ldexp(1, -prec-addprec-hextramag)
        else:
            h = ctx.convert(h)
        # Directed: steps x, x+h, ... x+n*h
        direction = options.get('direction', 0)
        if direction:
            h *= ctx.sign(direction)
            steps = xrange(n+1)
            norm = h
        # Central: steps x-n*h, x-(n-2)*h ..., x, ..., x+(n-2)*h, x+n*h
        else:
            steps = xrange(-n, n+1, 2)
            norm = (2*h)
        # Perturb
        if singular:
            x += 0.5*h
        values = [f(x+k*h) for k in steps]
        return values, norm, workprec
    finally:
        ctx.prec = orig


@defun
def diff(ctx, f, x, n=1, **options):
    r"""
    Numerically computes the derivative of `f`, `f'(x)`, or generally for
    an integer `n \ge 0`, the `n`-th derivative `f^{(n)}(x)`.
    A few basic examples are::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> diff(lambda x: x**2 + x, 1.0)
        3.0
        >>> diff(lambda x: x**2 + x, 1.0, 2)
        2.0
        >>> diff(lambda x: x**2 + x, 1.0, 3)
        0.0
        >>> nprint([diff(exp, 3, n) for n in range(5)])   # exp'(x) = exp(x)
        [20.0855, 20.0855, 20.0855, 20.0855, 20.0855]

    Even more generally, given a tuple of arguments `(x_1, \ldots, x_k)`
    and order `(n_1, \ldots, n_k)`, the partial derivative
    `f^{(n_1,\ldots,n_k)}(x_1,\ldots,x_k)` is evaluated. For example::

        >>> diff(lambda x,y: 3*x*y + 2*y - x, (0.25, 0.5), (0,1))
        2.75
        >>> diff(lambda x,y: 3*x*y + 2*y - x, (0.25, 0.5), (1,1))
        3.0

    **Options**

    The following optional keyword arguments are recognized:

    ``method``
        Supported methods are ``'step'`` or ``'quad'``: derivatives may be
        computed using either a finite difference with a small step
        size `h` (default), or numerical quadrature.
    ``direction``
        Direction of finite difference: can be -1 for a left
        difference, 0 for a central difference (default), or +1
        for a right difference; more generally can be any complex number.
    ``addprec``
        Extra precision for `h` used to account for the function's
        sensitivity to perturbations (default = 10).
    ``relative``
        Choose `h` relative to the magnitude of `x`, rather than an
        absolute value; useful for large or tiny `x` (default = False).
    ``h``
        As an alternative to ``addprec`` and ``relative``, manually
        select the step size `h`.
    ``singular``
        If True, evaluation exactly at the point `x` is avoided; this is
        useful for differentiating functions with removable singularities.
        Default = False.
    ``radius``
        Radius of integration contour (with ``method = 'quad'``).
        Default = 0.25. A larger radius typically is faster and more
        accurate, but it must be chosen so that `f` has no
        singularities within the radius from the evaluation point.

    A finite difference requires `n+1` function evaluations and must be
    performed at `(n+1)` times the target precision. Accordingly, `f` must
    support fast evaluation at high precision.

    With integration, a larger number of function evaluations is
    required, but not much extra precision is required. For high order
    derivatives, this method may thus be faster if f is very expensive to
    evaluate at high precision.

    **Further examples**

    The direction option is useful for computing left- or right-sided
    derivatives of nonsmooth functions::

        >>> diff(abs, 0, direction=0)
        0.0
        >>> diff(abs, 0, direction=1)
        1.0
        >>> diff(abs, 0, direction=-1)
        -1.0

    More generally, if the direction is nonzero, a right difference
    is computed where the step size is multiplied by sign(direction).
    For example, with direction=+j, the derivative from the positive
    imaginary direction will be computed::

        >>> diff(abs, 0, direction=j)
        (0.0 - 1.0j)

    With integration, the result may have a small imaginary part
    even even if the result is purely real::

        >>> diff(sqrt, 1, method='quad')    # doctest:+ELLIPSIS
        (0.5 - 4.59...e-26j)
        >>> chop(_)
        0.5

    Adding precision to obtain an accurate value::

        >>> diff(cos, 1e-30)
        0.0
        >>> diff(cos, 1e-30, h=0.0001)
        -9.99999998328279e-31
        >>> diff(cos, 1e-30, addprec=100)
        -1.0e-30

    """
    partial = False
    try:
        orders = list(n)
        x = list(x)
        partial = True
    except TypeError:
        pass
    if partial:
        x = [ctx.convert(_) for _ in x]
        return _partial_diff(ctx, f, x, orders, options)
    method = options.get('method', 'step')
    if n == 0 and method != 'quad' and not options.get('singular'):
        return f(ctx.convert(x))
    prec = ctx.prec
    try:
        if method == 'step':
            values, norm, workprec = hsteps(ctx, f, x, n, prec, **options)
            ctx.prec = workprec
            v = ctx.difference(values, n) / norm**n
        elif method == 'quad':
            ctx.prec += 10
            radius = ctx.convert(options.get('radius', 0.25))
            def g(t):
                rei = radius*ctx.expj(t)
                z = x + rei
                return f(z) / rei**n
            d = ctx.quadts(g, [0, 2*ctx.pi])
            v = d * ctx.factorial(n) / (2*ctx.pi)
        else:
            raise ValueError("unknown method: %r" % method)
    finally:
        ctx.prec = prec
    return +v

def _partial_diff(ctx, f, xs, orders, options):
    if not orders:
        return f()
    if not sum(orders):
        return f(*xs)
    i = 0
    for i in range(len(orders)):
        if orders[i]:
            break
    order = orders[i]
    def fdiff_inner(*f_args):
        def inner(t):
            return f(*(f_args[:i] + (t,) + f_args[i+1:]))
        return ctx.diff(inner, f_args[i], order, **options)
    orders[i] = 0
    return _partial_diff(ctx, fdiff_inner, xs, orders, options)

@defun
def diffs(ctx, f, x, n=None, **options):
    r"""
    Returns a generator that yields the sequence of derivatives

    .. math ::

        f(x), f'(x), f''(x), \ldots, f^{(k)}(x), \ldots

    With ``method='step'``, :func:`~mpmath.diffs` uses only `O(k)`
    function evaluations to generate the first `k` derivatives,
    rather than the roughly `O(k^2)` evaluations
    required if one calls :func:`~mpmath.diff` `k` separate times.

    With `n < \infty`, the generator stops as soon as the
    `n`-th derivative has been generated. If the exact number of
    needed derivatives is known in advance, this is further
    slightly more efficient.

    Options are the same as for :func:`~mpmath.diff`.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15
        >>> nprint(list(diffs(cos, 1, 5)))
        [0.540302, -0.841471, -0.540302, 0.841471, 0.540302, -0.841471]
        >>> for i, d in zip(range(6), diffs(cos, 1)):
        ...     print("%s %s" % (i, d))
        ...
        0 0.54030230586814
        1 -0.841470984807897
        2 -0.54030230586814
        3 0.841470984807897
        4 0.54030230586814
        5 -0.841470984807897

    """
    if n is None:
        n = ctx.inf
    else:
        n = int(n)
    if options.get('method', 'step') != 'step':
        k = 0
        while k < n + 1:
            yield ctx.diff(f, x, k, **options)
            k += 1
        return
    singular = options.get('singular')
    if singular:
        yield ctx.diff(f, x, 0, singular=True)
    else:
        yield f(ctx.convert(x))
    if n < 1:
        return
    if n == ctx.inf:
        A, B = 1, 2
    else:
        A, B = 1, n+1
    while 1:
        callprec = ctx.prec
        y, norm, workprec = hsteps(ctx, f, x, B, callprec, **options)
        for k in xrange(A, B):
            try:
                ctx.prec = workprec
                d = ctx.difference(y, k) / norm**k
            finally:
                ctx.prec = callprec
            yield +d
            if k >= n:
                return
        A, B = B, int(A*1.4+1)
        B = min(B, n)

def iterable_to_function(gen):
    gen = iter(gen)
    data = []
    def f(k):
        for i in xrange(len(data), k+1):
            data.append(next(gen))
        return data[k]
    return f

@defun
def diffs_prod(ctx, factors):
    r"""
    Given a list of `N` iterables or generators yielding
    `f_k(x), f'_k(x), f''_k(x), \ldots` for `k = 1, \ldots, N`,
    generate `g(x), g'(x), g''(x), \ldots` where
    `g(x) = f_1(x) f_2(x) \cdots f_N(x)`.

    At high precision and for large orders, this is typically more efficient
    than numerical differentiation if the derivatives of each `f_k(x)`
    admit direct computation.

    Note: This function does not increase the working precision internally,
    so guard digits may have to be added externally for full accuracy.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> f = lambda x: exp(x)*cos(x)*sin(x)
        >>> u = diffs(f, 1)
        >>> v = mp.diffs_prod([diffs(exp,1), diffs(cos,1), diffs(sin,1)])
        >>> next(u); next(v)
        1.23586333600241
        1.23586333600241
        >>> next(u); next(v)
        0.104658952245596
        0.104658952245596
        >>> next(u); next(v)
        -5.96999877552086
        -5.96999877552086
        >>> next(u); next(v)
        -12.4632923122697
        -12.4632923122697

    """
    N = len(factors)
    if N == 1:
        for c in factors[0]:
            yield c
    else:
        u = iterable_to_function(ctx.diffs_prod(factors[:N//2]))
        v = iterable_to_function(ctx.diffs_prod(factors[N//2:]))
        n = 0
        while 1:
            #yield sum(binomial(n,k)*u(n-k)*v(k) for k in xrange(n+1))
            s = u(n) * v(0)
            a = 1
            for k in xrange(1,n+1):
                a = a * (n-k+1) // k
                s += a * u(n-k) * v(k)
            yield s
            n += 1

def dpoly(n, _cache={}):
    """
    nth differentiation polynomial for exp (Faa di Bruno's formula).

    TODO: most exponents are zero, so maybe a sparse representation
    would be better.
    """
    if n in _cache:
        return _cache[n]
    if not _cache:
        _cache[0] = {(0,):1}
    R = dpoly(n-1)
    R = dict((c+(0,),v) for (c,v) in iteritems(R))
    Ra = {}
    for powers, count in iteritems(R):
        powers1 = (powers[0]+1,) + powers[1:]
        if powers1 in Ra:
            Ra[powers1] += count
        else:
            Ra[powers1] = count
    for powers, count in iteritems(R):
        if not sum(powers):
            continue
        for k,p in enumerate(powers):
            if p:
                powers2 = powers[:k] + (p-1,powers[k+1]+1) + powers[k+2:]
                if powers2 in Ra:
                    Ra[powers2] += p*count
                else:
                    Ra[powers2] = p*count
    _cache[n] = Ra
    return _cache[n]

@defun
def diffs_exp(ctx, fdiffs):
    r"""
    Given an iterable or generator yielding `f(x), f'(x), f''(x), \ldots`
    generate `g(x), g'(x), g''(x), \ldots` where `g(x) = \exp(f(x))`.

    At high precision and for large orders, this is typically more efficient
    than numerical differentiation if the derivatives of `f(x)`
    admit direct computation.

    Note: This function does not increase the working precision internally,
    so guard digits may have to be added externally for full accuracy.

    **Examples**

    The derivatives of the gamma function can be computed using
    logarithmic differentiation::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>>
        >>> def diffs_loggamma(x):
        ...     yield loggamma(x)
        ...     i = 0
        ...     while 1:
        ...         yield psi(i,x)
        ...         i += 1
        ...
        >>> u = diffs_exp(diffs_loggamma(3))
        >>> v = diffs(gamma, 3)
        >>> next(u); next(v)
        2.0
        2.0
        >>> next(u); next(v)
        1.84556867019693
        1.84556867019693
        >>> next(u); next(v)
        2.49292999190269
        2.49292999190269
        >>> next(u); next(v)
        3.44996501352367
        3.44996501352367

    """
    fn = iterable_to_function(fdiffs)
    f0 = ctx.exp(fn(0))
    yield f0
    i = 1
    while 1:
        s = ctx.mpf(0)
        for powers, c in iteritems(dpoly(i)):
            s += c*ctx.fprod(fn(k+1)**p for (k,p) in enumerate(powers) if p)
        yield s * f0
        i += 1

@defun
def differint(ctx, f, x, n=1, x0=0):
    r"""
    Calculates the Riemann-Liouville differintegral, or fractional
    derivative, defined by

    .. math ::

        \,_{x_0}{\mathbb{D}}^n_xf(x) = \frac{1}{\Gamma(m-n)} \frac{d^m}{dx^m}
        \int_{x_0}^{x}(x-t)^{m-n-1}f(t)dt

    where `f` is a given (presumably well-behaved) function,
    `x` is the evaluation point, `n` is the order, and `x_0` is
    the reference point of integration (`m` is an arbitrary
    parameter selected automatically).

    With `n = 1`, this is just the standard derivative `f'(x)`; with `n = 2`,
    the second derivative `f''(x)`, etc. With `n = -1`, it gives
    `\int_{x_0}^x f(t) dt`, with `n = -2`
    it gives `\int_{x_0}^x \left( \int_{x_0}^t f(u) du \right) dt`, etc.

    As `n` is permitted to be any number, this operator generalizes
    iterated differentiation and iterated integration to a single
    operator with a continuous order parameter.

    **Examples**

    There is an exact formula for the fractional derivative of a
    monomial `x^p`, which may be used as a reference. For example,
    the following gives a half-derivative (order 0.5)::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> x = mpf(3); p = 2; n = 0.5
        >>> differint(lambda t: t**p, x, n)
        7.81764019044672
        >>> gamma(p+1)/gamma(p-n+1) * x**(p-n)
        7.81764019044672

    Another useful test function is the exponential function, whose
    integration / differentiation formula easy generalizes
    to arbitrary order. Here we first compute a third derivative,
    and then a triply nested integral. (The reference point `x_0`
    is set to `-\infty` to avoid nonzero endpoint terms.)::

        >>> differint(lambda x: exp(pi*x), -1.5, 3)
        0.278538406900792
        >>> exp(pi*-1.5) * pi**3
        0.278538406900792
        >>> differint(lambda x: exp(pi*x), 3.5, -3, -inf)
        1922.50563031149
        >>> exp(pi*3.5) / pi**3
        1922.50563031149

    However, for noninteger `n`, the differentiation formula for the
    exponential function must be modified to give the same result as the
    Riemann-Liouville differintegral::

        >>> x = mpf(3.5)
        >>> c = pi
        >>> n = 1+2*j
        >>> differint(lambda x: exp(c*x), x, n)
        (-123295.005390743 + 140955.117867654j)
        >>> x**(-n) * exp(c)**x * (x*c)**n * gammainc(-n, 0, x*c) / gamma(-n)
        (-123295.005390743 + 140955.117867654j)


    """
    m = max(int(ctx.ceil(ctx.re(n)))+1, 1)
    r = m-n-1
    g = lambda x: ctx.quad(lambda t: (x-t)**r * f(t), [x0, x])
    return ctx.diff(g, x, m) / ctx.gamma(m-n)

@defun
def diffun(ctx, f, n=1, **options):
    r"""
    Given a function `f`, returns a function `g(x)` that evaluates the nth
    derivative `f^{(n)}(x)`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> cos2 = diffun(sin)
        >>> sin2 = diffun(sin, 4)
        >>> cos(1.3), cos2(1.3)
        (0.267498828624587, 0.267498828624587)
        >>> sin(1.3), sin2(1.3)
        (0.963558185417193, 0.963558185417193)

    The function `f` must support arbitrary precision evaluation.
    See :func:`~mpmath.diff` for additional details and supported
    keyword options.
    """
    if n == 0:
        return f
    def g(x):
        return ctx.diff(f, x, n, **options)
    return g

@defun
def taylor(ctx, f, x, n, **options):
    r"""
    Produces a degree-`n` Taylor polynomial around the point `x` of the
    given function `f`. The coefficients are returned as a list.

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nprint(chop(taylor(sin, 0, 5)))
        [0.0, 1.0, 0.0, -0.166667, 0.0, 0.00833333]

    The coefficients are computed using high-order numerical
    differentiation. The function must be possible to evaluate
    to arbitrary precision. See :func:`~mpmath.diff` for additional details
    and supported keyword options.

    Note that to evaluate the Taylor polynomial as an approximation
    of `f`, e.g. with :func:`~mpmath.polyval`, the coefficients must be reversed,
    and the point of the Taylor expansion must be subtracted from
    the argument:

        >>> p = taylor(exp, 2.0, 10)
        >>> polyval(p[::-1], 2.5 - 2.0)
        12.1824939606092
        >>> exp(2.5)
        12.1824939607035

    """
    gen = enumerate(ctx.diffs(f, x, n, **options))
    if options.get("chop", True):
        return [ctx.chop(d)/ctx.factorial(i) for i, d in gen]
    else:
        return [d/ctx.factorial(i) for i, d in gen]

@defun
def pade(ctx, a, L, M):
    r"""
    Computes a Pade approximation of degree `(L, M)` to a function.
    Given at least `L+M+1` Taylor coefficients `a` approximating
    a function `A(x)`, :func:`~mpmath.pade` returns coefficients of
    polynomials `P, Q` satisfying

    .. math ::

        P = \sum_{k=0}^L p_k x^k

        Q = \sum_{k=0}^M q_k x^k

        Q_0 = 1

        A(x) Q(x) = P(x) + O(x^{L+M+1})

    `P(x)/Q(x)` can provide a good approximation to an analytic function
    beyond the radius of convergence of its Taylor series (example
    from G.A. Baker 'Essentials of Pade Approximants' Academic Press,
    Ch.1A)::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> one = mpf(1)
        >>> def f(x):
        ...     return sqrt((one + 2*x)/(one + x))
        ...
        >>> a = taylor(f, 0, 6)
        >>> p, q = pade(a, 3, 3)
        >>> x = 10
        >>> polyval(p[::-1], x)/polyval(q[::-1], x)
        1.38169105566806
        >>> f(x)
        1.38169855941551

    """
    # To determine L+1 coefficients of P and M coefficients of Q
    # L+M+1 coefficients of A must be provided
    if len(a) < L+M+1:
        raise ValueError("L+M+1 Coefficients should be provided")

    if M == 0:
        if L == 0:
            return [ctx.one], [ctx.one]
        else:
            return a[:L+1], [ctx.one]

    # Solve first
    # a[L]*q[1] + ... + a[L-M+1]*q[M] = -a[L+1]
    # ...
    # a[L+M-1]*q[1] + ... + a[L]*q[M] = -a[L+M]
    A = ctx.matrix(M)
    for j in range(M):
        for i in range(min(M, L+j+1)):
            A[j, i] = a[L+j-i]
    v = -ctx.matrix(a[(L+1):(L+M+1)])
    x = ctx.lu_solve(A, v)
    q = [ctx.one] + list(x)
    # compute p
    p = [0]*(L+1)
    for i in range(L+1):
        s = a[i]
        for j in range(1, min(M,i) + 1):
            s += q[j]*a[i-j]
        p[i] = s
    return p, q

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\benchmark_helper.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import csv
import logging
import os
import random
import sys
import time
import timeit
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from time import sleep
from typing import Any

import coloredlogs
import numpy
import torch
import transformers
from packaging import version

import onnxruntime

logger = logging.getLogger(__name__)


class Precision(Enum):
    FLOAT32 = "fp32"
    FLOAT16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    def __str__(self):
        return self.value


class OptimizerInfo(Enum):
    # no_opt means using the raw ONNX model, but OnnxRuntime might still apply optimization as long as
    # graph optimization level is not 0 (disable all).
    NOOPT = "no_opt"
    BYORT = "by_ort"
    BYSCRIPT = "by_script"

    def __str__(self):
        return self.value


class ConfigModifier:
    def __init__(self, num_layers):
        self.num_layers = num_layers

    def modify(self, config):
        if self.num_layers is None:
            return
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = self.num_layers
            logger.info(f"Modifying pytorch model's number of hidden layers to: {self.num_layers}")
        if hasattr(config, "encoder_layers"):
            config.encoder_layers = self.num_layers
            logger.info(f"Modifying pytorch model's number of encoder layers to: {self.num_layers}")
        if hasattr(config, "decoder_layers "):
            config.decoder_layers = self.num_layers
            logger.info(f"Modifying pytorch model's number of decoder layers to: {self.num_layers}")

    def get_layer_num(self):
        return self.num_layers


IO_BINDING_DATA_TYPE_MAP = {
    "float32": numpy.float32,
    # TODO: Add more.
}


def create_onnxruntime_session(
    onnx_model_path,
    use_gpu,
    provider=None,
    enable_all_optimization=True,
    num_threads=-1,
    enable_profiling=False,
    verbose=False,
    enable_mlas_gemm_fastmath_arm64_bfloat16=False,
    provider_options={},  # map execution provider name to its option  # noqa: B006
):
    sess_options = onnxruntime.SessionOptions()

    if enable_all_optimization:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC

    if enable_profiling:
        sess_options.enable_profiling = True

    if num_threads > 0:
        sess_options.intra_op_num_threads = num_threads
        logger.debug(f"Session option: intra_op_num_threads={sess_options.intra_op_num_threads}")

    if verbose:
        sess_options.log_severity_level = 0
    else:
        sess_options.log_severity_level = 4

    if provider in onnxruntime.get_available_providers():
        providers = [provider]
    elif use_gpu:
        if provider == "dml":
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif provider == "rocm":
            providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
        elif provider == "migraphx":
            providers = [
                "MIGraphXExecutionProvider",
                "ROCMExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif provider == "cuda" or provider is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "tensorrt":
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            raise RuntimeError(f"The execution provider is not supported: {provider}")
    else:
        providers = ["CPUExecutionProvider"]

    if provider_options:
        providers = [(name, provider_options[name]) if name in provider_options else name for name in providers]

    if enable_mlas_gemm_fastmath_arm64_bfloat16:
        sess_options.add_session_config_entry("mlas.enable_gemm_fastmath_arm64_bfloat16", "1")

    session = None
    try:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)
    except Exception:
        logger.exception(f"Failed to create session for {onnx_model_path} with providers={providers}")

    return session


def setup_logger(verbose=True):
    if verbose:
        coloredlogs.install(
            level="DEBUG",
            fmt="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s",
        )
    else:
        coloredlogs.install(fmt="%(message)s")
        logging.getLogger("transformers").setLevel(logging.WARNING)


def prepare_environment(cache_dir, output_dir, use_gpu, provider=None):
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if use_gpu:
        if provider == "dml":
            assert "DmlExecutionProvider" in onnxruntime.get_available_providers(), (
                "Please install onnxruntime-directml package to test GPU inference."
            )

        else:
            assert not set(onnxruntime.get_available_providers()).isdisjoint(
                ["CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider"]
            ), "Please install onnxruntime-gpu package, or install ROCm support, to test GPU inference."

    logger.info(f"PyTorch Version:{torch.__version__}")
    logger.info(f"Transformers Version:{transformers.__version__}")
    logger.info(f"OnnxRuntime Version:{onnxruntime.__version__}")

    # Support three major versions of PyTorch and OnnxRuntime, and up to 9 months of transformers.
    assert version.parse(torch.__version__) >= version.parse("1.10.0")
    assert version.parse(transformers.__version__) >= version.parse("4.12.0")
    assert version.parse(onnxruntime.__version__) >= version.parse("1.10.0")


def get_latency_result(latency_list, batch_size):
    latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    latency_variance = numpy.var(latency_list, dtype=numpy.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)

    return {
        "test_times": len(latency_list),
        "latency_variance": f"{latency_variance:.2f}",
        "latency_90_percentile": f"{numpy.percentile(latency_list, 90) * 1000.0:.2f}",
        "latency_95_percentile": f"{numpy.percentile(latency_list, 95) * 1000.0:.2f}",
        "latency_99_percentile": f"{numpy.percentile(latency_list, 99) * 1000.0:.2f}",
        "average_latency_ms": f"{latency_ms:.2f}",
        "QPS": f"{throughput:.2f}",
    }


def output_details(results, csv_filename):
    with open(csv_filename, mode="a", newline="", encoding="ascii") as csv_file:
        column_names = [
            "engine",
            "version",
            "providers",
            "device",
            "precision",
            "optimizer",
            "io_binding",
            "model_name",
            "inputs",
            "threads",
            "batch_size",
            "sequence_length",
            "custom_layer_num",
            "datetime",
            "test_times",
            "QPS",
            "average_latency_ms",
            "latency_variance",
            "latency_90_percentile",
            "latency_95_percentile",
            "latency_99_percentile",
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    logger.info(f"Detail results are saved to csv file: {csv_filename}")


def output_summary(results, csv_filename, args):
    with open(csv_filename, mode="a", newline="", encoding="ascii") as csv_file:
        header_names = [
            "model_name",
            "inputs",
            "custom_layer_num",
            "engine",
            "version",
            "providers",
            "device",
            "precision",
            "optimizer",
            "io_binding",
            "threads",
        ]
        data_names = []
        for batch_size in args.batch_sizes:
            if args.sequence_lengths == [""]:
                data_names.append(f"b{batch_size}")
            else:
                for sequence_length in args.sequence_lengths:
                    data_names.append(f"b{batch_size}_s{sequence_length}")

        csv_writer = csv.DictWriter(csv_file, fieldnames=header_names + data_names)
        csv_writer.writeheader()
        for model_name in args.models:
            for input_count in [1, 2, 3]:
                for engine_name in args.engines:
                    for io_binding in [True, False, ""]:
                        for threads in args.num_threads:
                            row = {}
                            for result in results:
                                if (
                                    result["model_name"] == model_name
                                    and result["inputs"] == input_count
                                    and result["engine"] == engine_name
                                    and result["io_binding"] == io_binding
                                    and result["threads"] == threads
                                ):
                                    headers = {k: v for k, v in result.items() if k in header_names}
                                    if not row:
                                        row.update(headers)
                                        row.update({k: "" for k in data_names})
                                    else:
                                        for k in header_names:
                                            assert row[k] == headers[k]
                                    b = result["batch_size"]
                                    s = result["sequence_length"]
                                    if s:
                                        row[f"b{b}_s{s}"] = result["average_latency_ms"]
                                    else:
                                        row[f"b{b}"] = result["average_latency_ms"]
                            if row:
                                csv_writer.writerow(row)

    logger.info(f"Summary results are saved to csv file: {csv_filename}")


def output_fusion_statistics(model_fusion_statistics, csv_filename):
    with open(csv_filename, mode="a", newline="", encoding="ascii") as csv_file:
        column_names = [
            "model_filename",
            "datetime",
            "transformers",
            "torch",
            *list(next(iter(model_fusion_statistics.values())).keys()),
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for key in model_fusion_statistics:
            model_fusion_statistics[key]["datetime"] = str(datetime.now())
            model_fusion_statistics[key]["transformers"] = transformers.__version__
            model_fusion_statistics[key]["torch"] = torch.__version__
            model_fusion_statistics[key]["model_filename"] = key
            csv_writer.writerow(model_fusion_statistics[key])
    logger.info(f"Fusion statistics is saved to csv file: {csv_filename}")


def inference_ort(ort_session, ort_inputs, result_template, repeat_times, batch_size, warm_up_repeat=0):
    result = {}
    timeit.repeat(lambda: ort_session.run(None, ort_inputs), number=1, repeat=warm_up_repeat)  # Dry run
    latency_list = timeit.repeat(lambda: ort_session.run(None, ort_inputs), number=1, repeat=repeat_times)
    result.update(result_template)
    result.update({"io_binding": False})
    result.update(get_latency_result(latency_list, batch_size))
    return result


def inference_ort_with_io_binding(
    ort_session,
    ort_inputs,
    result_template,
    repeat_times,
    ort_output_names,
    ort_outputs,
    output_buffers,
    output_buffer_max_sizes,
    batch_size,
    device,
    data_type=numpy.longlong,
    warm_up_repeat=0,
):
    result = {}

    # Bind inputs and outputs to onnxruntime session
    io_binding = ort_session.io_binding()
    # Bind inputs to device
    for name in ort_inputs:
        np_input = torch.from_numpy(ort_inputs[name]).to(device)
        input_type = IO_BINDING_DATA_TYPE_MAP.get(str(ort_inputs[name].dtype), data_type)
        io_binding.bind_input(
            name,
            np_input.device.type,
            0,
            input_type,
            np_input.shape,
            np_input.data_ptr(),
        )
    # Bind outputs buffers with the sizes needed if not allocated already
    if len(output_buffers) == 0:
        allocateOutputBuffers(output_buffers, output_buffer_max_sizes, device)

    for i, ort_output_name in enumerate(ort_output_names):
        io_binding.bind_output(
            ort_output_name,
            output_buffers[i].device.type,
            0,
            numpy.float32,
            ort_outputs[i].shape,
            output_buffers[i].data_ptr(),
        )

    timeit.repeat(
        lambda: ort_session.run_with_iobinding(io_binding),
        number=1,
        repeat=warm_up_repeat,
    )  # Dry run

    latency_list = timeit.repeat(
        lambda: ort_session.run_with_iobinding(io_binding),
        number=1,
        repeat=repeat_times,
    )
    result.update(result_template)
    result.update({"io_binding": True})
    result.update(get_latency_result(latency_list, batch_size))
    return result


def allocateOutputBuffers(output_buffers, output_buffer_max_sizes, device):  # noqa: N802
    # Allocate output tensors with the largest test size needed. So the allocated memory can be reused
    # for each test run.

    for i in output_buffer_max_sizes:
        output_buffers.append(torch.empty(i, dtype=torch.float32, device=device))


def set_random_seed(seed=123):
    """Set random seed manually to get deterministic results"""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def get_gpu_info() -> list[dict[str, Any]] | None:
    from py3nvml.py3nvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlInit,
        nvmlShutdown,
    )

    try:
        nvmlInit()
        result = []
        device_count = nvmlDeviceGetCount()
        if not isinstance(device_count, int):
            return None

        for i in range(device_count):
            info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i))
            if isinstance(info, str):
                return None
            result.append(
                {
                    "id": i,
                    "name": nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)),
                    "total": info.total,
                    "free": info.free,
                    "used": info.used,
                }
            )
        nvmlShutdown()
        return result
    except NVMLError as error:
        print("Error fetching GPU information using nvml: %s", error)
        return None


class MemoryMonitor(ABC):
    def __init__(self, keep_measuring=True):
        self.keep_measuring = keep_measuring

    def measure_cpu_usage(self):
        import psutil

        max_usage = 0
        while True:
            max_usage = max(max_usage, psutil.Process(os.getpid()).memory_info().rss / 1024**2)
            sleep(0.005)  # 5ms
            if not self.keep_measuring:
                break
        return max_usage

    @abstractmethod
    def measure_gpu_usage(self) -> list[dict[str, Any]] | None:
        raise NotImplementedError()


class CudaMemoryMonitor(MemoryMonitor):
    def __init__(self, keep_measuring=True):
        super().__init__(keep_measuring)

    def measure_gpu_usage(self) -> list[dict[str, Any]] | None:
        from py3nvml.py3nvml import (
            NVMLError,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
        )

        max_gpu_usage = []
        gpu_name = []
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            if not isinstance(device_count, int):
                logger.error(f"nvmlDeviceGetCount result is not integer: {device_count}")
                return None

            max_gpu_usage = [0 for i in range(device_count)]
            gpu_name = [nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)) for i in range(device_count)]
            while True:
                for i in range(device_count):
                    info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i))
                    if isinstance(info, str):
                        logger.error(f"nvmlDeviceGetMemoryInfo returns str: {info}")
                        return None
                    max_gpu_usage[i] = max(max_gpu_usage[i], info.used / 1024**2)
                sleep(0.005)  # 5ms
                if not self.keep_measuring:
                    break
            nvmlShutdown()
            return [
                {
                    "device_id": i,
                    "name": gpu_name[i],
                    "max_used_MB": max_gpu_usage[i],
                }
                for i in range(device_count)
            ]
        except NVMLError as error:
            logger.error("Error fetching GPU information using nvml: %s", error)
            return None


class RocmMemoryMonitor(MemoryMonitor):
    def __init__(self, keep_measuring=True):
        super().__init__(keep_measuring)
        rocm_smi_path = "/opt/rocm/libexec/rocm_smi"
        if os.path.exists(rocm_smi_path):
            if rocm_smi_path not in sys.path:
                sys.path.append(rocm_smi_path)
        try:
            import rocm_smi

            self.rocm_smi = rocm_smi
            self.rocm_smi.initializeRsmi()
        except ImportError:
            self.rocm_smi = None

    def get_used_memory(self, dev):
        if self.rocm_smi is None:
            return -1
        return self.rocm_smi.getMemInfo(dev, "VRAM")[0] / 1024 / 1024

    def measure_gpu_usage(self):
        if self.rocm_smi is None:
            return None

        device_count = len(self.rocm_smi.listDevices()) if self.rocm_smi is not None else 0
        max_gpu_usage = [0 for i in range(device_count)]
        gpu_name = [f"GPU{i}" for i in range(device_count)]
        while True:
            for i in range(device_count):
                max_gpu_usage[i] = max(max_gpu_usage[i], self.get_used_memory(i))
            time.sleep(0.005)  # 5ms
            if not self.keep_measuring:
                break
        return [
            {
                "device_id": i,
                "name": gpu_name[i],
                "max_used_MB": max_gpu_usage[i],
            }
            for i in range(device_count)
        ]


def measure_memory(is_gpu, func, monitor_type="cuda", start_memory=None):
    memory_monitor_type = None
    if monitor_type == "rocm":
        memory_monitor_type = RocmMemoryMonitor
    else:
        memory_monitor_type = CudaMemoryMonitor

    monitor = memory_monitor_type(False)

    if is_gpu:
        if start_memory is not None:
            memory_before_test = start_memory
        else:
            memory_before_test = monitor.measure_gpu_usage()
        if memory_before_test is None:
            return None

        if func is None:
            return memory_before_test

        with ThreadPoolExecutor() as executor:
            monitor = memory_monitor_type()
            mem_thread = executor.submit(monitor.measure_gpu_usage)
            try:
                fn_thread = executor.submit(func)
                _ = fn_thread.result()
            finally:
                monitor.keep_measuring = False
                max_usage = mem_thread.result()

            if max_usage is None:
                return None

            logger.info(f"GPU memory usage: before={memory_before_test}  peak={max_usage}")
            if len(memory_before_test) >= 1 and len(max_usage) >= 1 and len(memory_before_test) == len(max_usage):
                # When there are multiple GPUs, we will check the one with maximum usage.
                max_used = 0
                for i, memory_before in enumerate(memory_before_test):
                    before = memory_before["max_used_MB"]
                    after = max_usage[i]["max_used_MB"]
                    used = after - before
                    max_used = max(max_used, used)
                return max_used
        return None

    # CPU memory
    if start_memory is not None:
        memory_before_test = start_memory
    else:
        memory_before_test = monitor.measure_cpu_usage()

    if func is None:
        return memory_before_test

    with ThreadPoolExecutor() as executor:
        monitor = memory_monitor_type()
        mem_thread = executor.submit(monitor.measure_cpu_usage)
        try:
            fn_thread = executor.submit(func)
            _ = fn_thread.result()
        finally:
            monitor.keep_measuring = False
            max_usage = mem_thread.result()

        logger.info(f"CPU memory usage: before={memory_before_test:.1f} MB, peak={max_usage:.1f} MB")
        return max_usage - memory_before_test


def get_ort_environment_variables():
    # Environment variables might impact ORT performance on transformer models. Note that they are for testing only.
    env_names = [
        "ORT_DISABLE_FUSED_ATTENTION",
        "ORT_ENABLE_FUSED_CAUSAL_ATTENTION",
        "ORT_DISABLE_FUSED_CROSS_ATTENTION",
        "ORT_DISABLE_TRT_FLASH_ATTENTION",
        "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION",
        "ORT_TRANSFORMER_OPTIONS",
        "ORT_CUDA_GEMM_OPTIONS",
    ]
    env = ""
    for name in env_names:
        value = os.getenv(name)
        if value is None:
            continue
        if env:
            env += ","
        env += f"{name}={value}"
    return env

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_core.py ===
import socket
import struct
import threading
import time
from typing import Optional, Union

# websocket modules
from ._abnf import ABNF, STATUS_NORMAL, continuous_frame, frame_buffer
from ._exceptions import WebSocketProtocolException, WebSocketConnectionClosedException
from ._handshake import SUPPORTED_REDIRECT_STATUSES, handshake
from ._http import connect, proxy_info
from ._logging import debug, error, trace, isEnabledForError, isEnabledForTrace
from ._socket import getdefaulttimeout, recv, send, sock_opt
from ._ssl_compat import ssl
from ._utils import NoLock

"""
_core.py
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

__all__ = ["WebSocket", "create_connection"]


class WebSocket:
    """
    Low level WebSocket interface.

    This class is based on the WebSocket protocol `draft-hixie-thewebsocketprotocol-76 <http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol-76>`_

    We can connect to the websocket server and send/receive data.
    The following example is an echo client.

    >>> import websocket
    >>> ws = websocket.WebSocket()
    >>> ws.connect("ws://echo.websocket.events")
    >>> ws.recv()
    'echo.websocket.events sponsored by Lob.com'
    >>> ws.send("Hello, Server")
    19
    >>> ws.recv()
    'Hello, Server'
    >>> ws.close()

    Parameters
    ----------
    get_mask_key: func
        A callable function to get new mask keys, see the
        WebSocket.set_mask_key's docstring for more information.
    sockopt: tuple
        Values for socket.setsockopt.
        sockopt must be tuple and each element is argument of sock.setsockopt.
    sslopt: dict
        Optional dict object for ssl socket options. See FAQ for details.
    fire_cont_frame: bool
        Fire recv event for each cont frame. Default is False.
    enable_multithread: bool
        If set to True, lock send method.
    skip_utf8_validation: bool
        Skip utf8 validation.
    """

    def __init__(
        self,
        get_mask_key=None,
        sockopt=None,
        sslopt=None,
        fire_cont_frame: bool = False,
        enable_multithread: bool = True,
        skip_utf8_validation: bool = False,
        **_,
    ):
        """
        Initialize WebSocket object.

        Parameters
        ----------
        sslopt: dict
            Optional dict object for ssl socket options. See FAQ for details.
        """
        self.sock_opt = sock_opt(sockopt, sslopt)
        self.handshake_response = None
        self.sock: Optional[socket.socket] = None

        self.connected = False
        self.get_mask_key = get_mask_key
        # These buffer over the build-up of a single frame.
        self.frame_buffer = frame_buffer(self._recv, skip_utf8_validation)
        self.cont_frame = continuous_frame(fire_cont_frame, skip_utf8_validation)

        if enable_multithread:
            self.lock = threading.Lock()
            self.readlock = threading.Lock()
        else:
            self.lock = NoLock()
            self.readlock = NoLock()

    def __iter__(self):
        """
        Allow iteration over websocket, implying sequential `recv` executions.
        """
        while True:
            yield self.recv()

    def __next__(self):
        return self.recv()

    def next(self):
        return self.__next__()

    def fileno(self):
        return self.sock.fileno()

    def set_mask_key(self, func):
        """
        Set function to create mask key. You can customize mask key generator.
        Mainly, this is for testing purpose.

        Parameters
        ----------
        func: func
            callable object. the func takes 1 argument as integer.
            The argument means length of mask key.
            This func must return string(byte array),
            which length is argument specified.
        """
        self.get_mask_key = func

    def gettimeout(self) -> Union[float, int, None]:
        """
        Get the websocket timeout (in seconds) as an int or float

        Returns
        ----------
        timeout: int or float
             returns timeout value (in seconds). This value could be either float/integer.
        """
        return self.sock_opt.timeout

    def settimeout(self, timeout: Union[float, int, None]):
        """
        Set the timeout to the websocket.

        Parameters
        ----------
        timeout: int or float
            timeout time (in seconds). This value could be either float/integer.
        """
        self.sock_opt.timeout = timeout
        if self.sock:
            self.sock.settimeout(timeout)

    timeout = property(gettimeout, settimeout)

    def getsubprotocol(self):
        """
        Get subprotocol
        """
        if self.handshake_response:
            return self.handshake_response.subprotocol
        else:
            return None

    subprotocol = property(getsubprotocol)

    def getstatus(self):
        """
        Get handshake status
        """
        if self.handshake_response:
            return self.handshake_response.status
        else:
            return None

    status = property(getstatus)

    def getheaders(self):
        """
        Get handshake response header
        """
        if self.handshake_response:
            return self.handshake_response.headers
        else:
            return None

    def is_ssl(self):
        try:
            return isinstance(self.sock, ssl.SSLSocket)
        except:
            return False

    headers = property(getheaders)

    def connect(self, url, **options):
        """
        Connect to url. url is websocket url scheme.
        ie. ws://host:port/resource
        You can customize using 'options'.
        If you set "header" list object, you can set your own custom header.

        >>> ws = WebSocket()
        >>> ws.connect("ws://echo.websocket.events",
                ...     header=["User-Agent: MyProgram",
                ...             "x-custom: header"])

        Parameters
        ----------
        header: list or dict
            Custom http header list or dict.
        cookie: str
            Cookie value.
        origin: str
            Custom origin url.
        connection: str
            Custom connection header value.
            Default value "Upgrade" set in _handshake.py
        suppress_origin: bool
            Suppress outputting origin header.
        host: str
            Custom host header string.
        timeout: int or float
            Socket timeout time. This value is an integer or float.
            If you set None for this value, it means "use default_timeout value"
        http_proxy_host: str
            HTTP proxy host name.
        http_proxy_port: str or int
            HTTP proxy port. Default is 80.
        http_no_proxy: list
            Whitelisted host names that don't use the proxy.
        http_proxy_auth: tuple
            HTTP proxy auth information. Tuple of username and password. Default is None.
        http_proxy_timeout: int or float
            HTTP proxy timeout, default is 60 sec as per python-socks.
        redirect_limit: int
            Number of redirects to follow.
        subprotocols: list
            List of available subprotocols. Default is None.
        socket: socket
            Pre-initialized stream socket.
        """
        self.sock_opt.timeout = options.get("timeout", self.sock_opt.timeout)
        self.sock, addrs = connect(
            url, self.sock_opt, proxy_info(**options), options.pop("socket", None)
        )

        try:
            self.handshake_response = handshake(self.sock, url, *addrs, **options)
            for _ in range(options.pop("redirect_limit", 3)):
                if self.handshake_response.status in SUPPORTED_REDIRECT_STATUSES:
                    url = self.handshake_response.headers["location"]
                    self.sock.close()
                    self.sock, addrs = connect(
                        url,
                        self.sock_opt,
                        proxy_info(**options),
                        options.pop("socket", None),
                    )
                    self.handshake_response = handshake(
                        self.sock, url, *addrs, **options
                    )
            self.connected = True
        except:
            if self.sock:
                self.sock.close()
                self.sock = None
            raise

    def send(self, payload: Union[bytes, str], opcode: int = ABNF.OPCODE_TEXT) -> int:
        """
        Send the data as string.

        Parameters
        ----------
        payload: str
            Payload must be utf-8 string or unicode,
            If the opcode is OPCODE_TEXT.
            Otherwise, it must be string(byte array).
        opcode: int
            Operation code (opcode) to send.
        """

        frame = ABNF.create_frame(payload, opcode)
        return self.send_frame(frame)

    def send_text(self, text_data: str) -> int:
        """
        Sends UTF-8 encoded text.
        """
        return self.send(text_data, ABNF.OPCODE_TEXT)

    def send_bytes(self, data: Union[bytes, bytearray]) -> int:
        """
        Sends a sequence of bytes.
        """
        return self.send(data, ABNF.OPCODE_BINARY)

    def send_frame(self, frame) -> int:
        """
        Send the data frame.

        >>> ws = create_connection("ws://echo.websocket.events")
        >>> frame = ABNF.create_frame("Hello", ABNF.OPCODE_TEXT)
        >>> ws.send_frame(frame)
        >>> cont_frame = ABNF.create_frame("My name is ", ABNF.OPCODE_CONT, 0)
        >>> ws.send_frame(frame)
        >>> cont_frame = ABNF.create_frame("Foo Bar", ABNF.OPCODE_CONT, 1)
        >>> ws.send_frame(frame)

        Parameters
        ----------
        frame: ABNF frame
            frame data created by ABNF.create_frame
        """
        if self.get_mask_key:
            frame.get_mask_key = self.get_mask_key
        data = frame.format()
        length = len(data)
        if isEnabledForTrace():
            trace(f"++Sent raw: {repr(data)}")
            trace(f"++Sent decoded: {frame.__str__()}")
        with self.lock:
            while data:
                l = self._send(data)
                data = data[l:]

        return length

    def send_binary(self, payload: bytes) -> int:
        """
        Send a binary message (OPCODE_BINARY).

        Parameters
        ----------
        payload: bytes
            payload of message to send.
        """
        return self.send(payload, ABNF.OPCODE_BINARY)

    def ping(self, payload: Union[str, bytes] = ""):
        """
        Send ping data.

        Parameters
        ----------
        payload: str
            data payload to send server.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.send(payload, ABNF.OPCODE_PING)

    def pong(self, payload: Union[str, bytes] = ""):
        """
        Send pong data.

        Parameters
        ----------
        payload: str
            data payload to send server.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.send(payload, ABNF.OPCODE_PONG)

    def recv(self) -> Union[str, bytes]:
        """
        Receive string data(byte array) from the server.

        Returns
        ----------
        data: string (byte array) value.
        """
        with self.readlock:
            opcode, data = self.recv_data()
        if opcode == ABNF.OPCODE_TEXT:
            data_received: Union[bytes, str] = data
            if isinstance(data_received, bytes):
                return data_received.decode("utf-8")
            elif isinstance(data_received, str):
                return data_received
        elif opcode == ABNF.OPCODE_BINARY:
            data_binary: bytes = data
            return data_binary
        else:
            return ""

    def recv_data(self, control_frame: bool = False) -> tuple:
        """
        Receive data with operation code.

        Parameters
        ----------
        control_frame: bool
            a boolean flag indicating whether to return control frame
            data, defaults to False

        Returns
        -------
        opcode, frame.data: tuple
            tuple of operation code and string(byte array) value.
        """
        opcode, frame = self.recv_data_frame(control_frame)
        return opcode, frame.data

    def recv_data_frame(self, control_frame: bool = False) -> tuple:
        """
        Receive data with operation code.

        If a valid ping message is received, a pong response is sent.

        Parameters
        ----------
        control_frame: bool
            a boolean flag indicating whether to return control frame
            data, defaults to False

        Returns
        -------
        frame.opcode, frame: tuple
            tuple of operation code and string(byte array) value.
        """
        while True:
            frame = self.recv_frame()
            if isEnabledForTrace():
                trace(f"++Rcv raw: {repr(frame.format())}")
                trace(f"++Rcv decoded: {frame.__str__()}")
            if not frame:
                # handle error:
                # 'NoneType' object has no attribute 'opcode'
                raise WebSocketProtocolException(f"Not a valid frame {frame}")
            elif frame.opcode in (
                ABNF.OPCODE_TEXT,
                ABNF.OPCODE_BINARY,
                ABNF.OPCODE_CONT,
            ):
                self.cont_frame.validate(frame)
                self.cont_frame.add(frame)

                if self.cont_frame.is_fire(frame):
                    return self.cont_frame.extract(frame)

            elif frame.opcode == ABNF.OPCODE_CLOSE:
                self.send_close()
                return frame.opcode, frame
            elif frame.opcode == ABNF.OPCODE_PING:
                if len(frame.data) < 126:
                    self.pong(frame.data)
                else:
                    raise WebSocketProtocolException("Ping message is too long")
                if control_frame:
                    return frame.opcode, frame
            elif frame.opcode == ABNF.OPCODE_PONG:
                if control_frame:
                    return frame.opcode, frame

    def recv_frame(self):
        """
        Receive data as frame from server.

        Returns
        -------
        self.frame_buffer.recv_frame(): ABNF frame object
        """
        return self.frame_buffer.recv_frame()

    def send_close(self, status: int = STATUS_NORMAL, reason: bytes = b""):
        """
        Send close data to the server.

        Parameters
        ----------
        status: int
            Status code to send. See STATUS_XXX.
        reason: str or bytes
            The reason to close. This must be string or UTF-8 bytes.
        """
        if status < 0 or status >= ABNF.LENGTH_16:
            raise ValueError("code is invalid range")
        self.connected = False
        self.send(struct.pack("!H", status) + reason, ABNF.OPCODE_CLOSE)

    def close(self, status: int = STATUS_NORMAL, reason: bytes = b"", timeout: int = 3):
        """
        Close Websocket object

        Parameters
        ----------
        status: int
            Status code to send. See VALID_CLOSE_STATUS in ABNF.
        reason: bytes
            The reason to close in UTF-8.
        timeout: int or float
            Timeout until receive a close frame.
            If None, it will wait forever until receive a close frame.
        """
        if not self.connected:
            return
        if status < 0 or status >= ABNF.LENGTH_16:
            raise ValueError("code is invalid range")

        try:
            self.connected = False
            self.send(struct.pack("!H", status) + reason, ABNF.OPCODE_CLOSE)
            sock_timeout = self.sock.gettimeout()
            self.sock.settimeout(timeout)
            start_time = time.time()
            while timeout is None or time.time() - start_time < timeout:
                try:
                    frame = self.recv_frame()
                    if frame.opcode != ABNF.OPCODE_CLOSE:
                        continue
                    if isEnabledForError():
                        recv_status = struct.unpack("!H", frame.data[0:2])[0]
                        if recv_status >= 3000 and recv_status <= 4999:
                            debug(f"close status: {repr(recv_status)}")
                        elif recv_status != STATUS_NORMAL:
                            error(f"close status: {repr(recv_status)}")
                    break
                except:
                    break
            self.sock.settimeout(sock_timeout)
            self.sock.shutdown(socket.SHUT_RDWR)
        except:
            pass

        self.shutdown()

    def abort(self):
        """
        Low-level asynchronous abort, wakes up other threads that are waiting in recv_*
        """
        if self.connected:
            self.sock.shutdown(socket.SHUT_RDWR)

    def shutdown(self):
        """
        close socket, immediately.
        """
        if self.sock:
            self.sock.close()
            self.sock = None
            self.connected = False

    def _send(self, data: Union[str, bytes]):
        return send(self.sock, data)

    def _recv(self, bufsize):
        try:
            return recv(self.sock, bufsize)
        except WebSocketConnectionClosedException:
            if self.sock:
                self.sock.close()
            self.sock = None
            self.connected = False
            raise


def create_connection(url: str, timeout=None, class_=WebSocket, **options):
    """
    Connect to url and return websocket object.

    Connect to url and return the WebSocket object.
    Passing optional timeout parameter will set the timeout on the socket.
    If no timeout is supplied,
    the global default timeout setting returned by getdefaulttimeout() is used.
    You can customize using 'options'.
    If you set "header" list object, you can set your own custom header.

    >>> conn = create_connection("ws://echo.websocket.events",
         ...     header=["User-Agent: MyProgram",
         ...             "x-custom: header"])

    Parameters
    ----------
    class_: class
        class to instantiate when creating the connection. It has to implement
        settimeout and connect. It's __init__ should be compatible with
        WebSocket.__init__, i.e. accept all of it's kwargs.
    header: list or dict
        custom http header list or dict.
    cookie: str
        Cookie value.
    origin: str
        custom origin url.
    suppress_origin: bool
        suppress outputting origin header.
    host: str
        custom host header string.
    timeout: int or float
        socket timeout time. This value could be either float/integer.
        If set to None, it uses the default_timeout value.
    http_proxy_host: str
        HTTP proxy host name.
    http_proxy_port: str or int
        HTTP proxy port. If not set, set to 80.
    http_no_proxy: list
        Whitelisted host names that don't use the proxy.
    http_proxy_auth: tuple
        HTTP proxy auth information. tuple of username and password. Default is None.
    http_proxy_timeout: int or float
        HTTP proxy timeout, default is 60 sec as per python-socks.
    enable_multithread: bool
        Enable lock for multithread.
    redirect_limit: int
        Number of redirects to follow.
    sockopt: tuple
        Values for socket.setsockopt.
        sockopt must be a tuple and each element is an argument of sock.setsockopt.
    sslopt: dict
        Optional dict object for ssl socket options. See FAQ for details.
    subprotocols: list
        List of available subprotocols. Default is None.
    skip_utf8_validation: bool
        Skip utf8 validation.
    socket: socket
        Pre-initialized stream socket.
    """
    sockopt = options.pop("sockopt", [])
    sslopt = options.pop("sslopt", {})
    fire_cont_frame = options.pop("fire_cont_frame", False)
    enable_multithread = options.pop("enable_multithread", True)
    skip_utf8_validation = options.pop("skip_utf8_validation", False)
    websock = class_(
        sockopt=sockopt,
        sslopt=sslopt,
        fire_cont_frame=fire_cont_frame,
        enable_multithread=enable_multithread,
        skip_utf8_validation=skip_utf8_validation,
        **options,
    )
    websock.settimeout(timeout if timeout is not None else getdefaulttimeout())
    websock.connect(url, **options)
    return websock

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\html.py ===
"""
Module for formatting output data in HTML.
"""
from __future__ import annotations

from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    cast,
)

from pandas._config import get_option

from pandas._libs import lib

from pandas import (
    MultiIndex,
    option_context,
)

from pandas.io.common import is_url
from pandas.io.formats.format import (
    DataFrameFormatter,
    get_level_lengths,
)
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
    )


class HTMLFormatter:
    """
    Internal class for formatting output data in html.
    This class is intended for shared functionality between
    DataFrame.to_html() and DataFrame._repr_html_().
    Any logic in common with other output formatting methods
    should ideally be inherited from classes in format.py
    and this class responsible for only producing html markup.
    """

    indent_delta: Final = 2

    def __init__(
        self,
        formatter: DataFrameFormatter,
        classes: str | list[str] | tuple[str, ...] | None = None,
        border: int | bool | None = None,
        table_id: str | None = None,
        render_links: bool = False,
    ) -> None:
        self.fmt = formatter
        self.classes = classes

        self.frame = self.fmt.frame
        self.columns = self.fmt.tr_frame.columns
        self.elements: list[str] = []
        self.bold_rows = self.fmt.bold_rows
        self.escape = self.fmt.escape
        self.show_dimensions = self.fmt.show_dimensions
        if border is None or border is True:
            border = cast(int, get_option("display.html.border"))
        elif not border:
            border = None

        self.border = border
        self.table_id = table_id
        self.render_links = render_links

        self.col_space = {}
        is_multi_index = isinstance(self.columns, MultiIndex)
        for column, value in self.fmt.col_space.items():
            col_space_value = f"{value}px" if isinstance(value, int) else value
            self.col_space[column] = col_space_value
            # GH 53885: Handling case where column is index
            # Flatten the data in the multi index and add in the map
            if is_multi_index and isinstance(column, tuple):
                for column_index in column:
                    self.col_space[str(column_index)] = col_space_value

    def to_string(self) -> str:
        lines = self.render()
        if any(isinstance(x, str) for x in lines):
            lines = [str(x) for x in lines]
        return "\n".join(lines)

    def render(self) -> list[str]:
        self._write_table()

        if self.should_show_dimensions:
            by = chr(215)  # ×  # noqa: RUF003
            self.write(
                f"<p>{len(self.frame)} rows {by} {len(self.frame.columns)} columns</p>"
            )

        return self.elements

    @property
    def should_show_dimensions(self) -> bool:
        return self.fmt.should_show_dimensions

    @property
    def show_row_idx_names(self) -> bool:
        return self.fmt.show_row_idx_names

    @property
    def show_col_idx_names(self) -> bool:
        return self.fmt.show_col_idx_names

    @property
    def row_levels(self) -> int:
        if self.fmt.index:
            # showing (row) index
            return self.frame.index.nlevels
        elif self.show_col_idx_names:
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # If the row index is not displayed a column of
            # blank cells need to be included before the DataFrame values.
            return 1
        # not showing (row) index
        return 0

    def _get_columns_formatted_values(self) -> Iterable:
        return self.columns

    @property
    def is_truncated(self) -> bool:
        return self.fmt.is_truncated

    @property
    def ncols(self) -> int:
        return len(self.fmt.tr_frame.columns)

    def write(self, s: Any, indent: int = 0) -> None:
        rs = pprint_thing(s)
        self.elements.append(" " * indent + rs)

    def write_th(
        self, s: Any, header: bool = False, indent: int = 0, tags: str | None = None
    ) -> None:
        """
        Method for writing a formatted <th> cell.

        If col_space is set on the formatter then that is used for
        the value of min-width.

        Parameters
        ----------
        s : object
            The data to be written inside the cell.
        header : bool, default False
            Set to True if the <th> is for use inside <thead>.  This will
            cause min-width to be set if there is one.
        indent : int, default 0
            The indentation level of the cell.
        tags : str, default None
            Tags to include in the cell.

        Returns
        -------
        A written <th> cell.
        """
        col_space = self.col_space.get(s, None)

        if header and col_space is not None:
            tags = tags or ""
            tags += f'style="min-width: {col_space};"'

        self._write_cell(s, kind="th", indent=indent, tags=tags)

    def write_td(self, s: Any, indent: int = 0, tags: str | None = None) -> None:
        self._write_cell(s, kind="td", indent=indent, tags=tags)

    def _write_cell(
        self, s: Any, kind: str = "td", indent: int = 0, tags: str | None = None
    ) -> None:
        if tags is not None:
            start_tag = f"<{kind} {tags}>"
        else:
            start_tag = f"<{kind}>"

        if self.escape:
            # escape & first to prevent double escaping of &
            esc = {"&": r"&amp;", "<": r"&lt;", ">": r"&gt;"}
        else:
            esc = {}

        rs = pprint_thing(s, escape_chars=esc).strip()

        if self.render_links and is_url(rs):
            rs_unescaped = pprint_thing(s, escape_chars={}).strip()
            start_tag += f'<a href="{rs_unescaped}" target="_blank">'
            end_a = "</a>"
        else:
            end_a = ""

        self.write(f"{start_tag}{rs}{end_a}</{kind}>", indent)

    def write_tr(
        self,
        line: Iterable,
        indent: int = 0,
        indent_delta: int = 0,
        header: bool = False,
        align: str | None = None,
        tags: dict[int, str] | None = None,
        nindex_levels: int = 0,
    ) -> None:
        if tags is None:
            tags = {}

        if align is None:
            self.write("<tr>", indent)
        else:
            self.write(f'<tr style="text-align: {align};">', indent)
        indent += indent_delta

        for i, s in enumerate(line):
            val_tag = tags.get(i, None)
            if header or (self.bold_rows and i < nindex_levels):
                self.write_th(s, indent=indent, header=header, tags=val_tag)
            else:
                self.write_td(s, indent, tags=val_tag)

        indent -= indent_delta
        self.write("</tr>", indent)

    def _write_table(self, indent: int = 0) -> None:
        _classes = ["dataframe"]  # Default class.
        use_mathjax = get_option("display.html.use_mathjax")
        if not use_mathjax:
            _classes.append("tex2jax_ignore")
        if self.classes is not None:
            if isinstance(self.classes, str):
                self.classes = self.classes.split()
            if not isinstance(self.classes, (list, tuple)):
                raise TypeError(
                    "classes must be a string, list, "
                    f"or tuple, not {type(self.classes)}"
                )
            _classes.extend(self.classes)

        if self.table_id is None:
            id_section = ""
        else:
            id_section = f' id="{self.table_id}"'

        if self.border is None:
            border_attr = ""
        else:
            border_attr = f' border="{self.border}"'

        self.write(
            f'<table{border_attr} class="{" ".join(_classes)}"{id_section}>',
            indent,
        )

        if self.fmt.header or self.show_row_idx_names:
            self._write_header(indent + self.indent_delta)

        self._write_body(indent + self.indent_delta)

        self.write("</table>", indent)

    def _write_col_header(self, indent: int) -> None:
        row: list[Hashable]
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        if isinstance(self.columns, MultiIndex):
            template = 'colspan="{span:d}" halign="left"'

            sentinel: lib.NoDefault | bool
            if self.fmt.sparsify:
                # GH3547
                sentinel = lib.no_default
            else:
                sentinel = False
            levels = self.columns._format_multi(sparsify=sentinel, include_names=False)
            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            for lnum, (records, values) in enumerate(zip(level_lengths, levels)):
                if is_truncated_horizontally:
                    # modify the header lines
                    ins_col = self.fmt.tr_col_num
                    if self.fmt.sparsify:
                        recs_new = {}
                        # Increment tags after ... col.
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            elif tag + span > ins_col:
                                recs_new[tag] = span + 1
                                if lnum == inner_lvl:
                                    values = (
                                        values[:ins_col] + ("...",) + values[ins_col:]
                                    )
                                else:
                                    # sparse col headers do not receive a ...
                                    values = (
                                        values[:ins_col]
                                        + (values[ins_col - 1],)
                                        + values[ins_col:]
                                    )
                            else:
                                recs_new[tag] = span
                            # if ins_col lies between tags, all col headers
                            # get ...
                            if tag + span == ins_col:
                                recs_new[ins_col] = 1
                                values = values[:ins_col] + ("...",) + values[ins_col:]
                        records = recs_new
                        inner_lvl = len(level_lengths) - 1
                        if lnum == inner_lvl:
                            records[ins_col] = 1
                    else:
                        recs_new = {}
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            else:
                                recs_new[tag] = span
                        recs_new[ins_col] = 1
                        records = recs_new
                        values = values[:ins_col] + ["..."] + values[ins_col:]

                # see gh-22579
                # Column Offset Bug with to_html(index=False) with
                # MultiIndex Columns and Index.
                # Initially fill row with blank cells before column names.
                # TODO: Refactor to remove code duplication with code
                # block below for standard columns index.
                row = [""] * (self.row_levels - 1)
                if self.fmt.index or self.show_col_idx_names:
                    # see gh-22747
                    # If to_html(index_names=False) do not show columns
                    # index names.
                    # TODO: Refactor to use _get_column_name_list from
                    # DataFrameFormatter class and create a
                    # _get_formatted_column_labels function for code
                    # parity with DataFrameFormatter class.
                    if self.fmt.show_index_names:
                        name = self.columns.names[lnum]
                        row.append(pprint_thing(name or ""))
                    else:
                        row.append("")

                tags = {}
                j = len(row)
                for i, v in enumerate(values):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template.format(span=records[i])
                    else:
                        continue
                    j += 1
                    row.append(v)
                self.write_tr(row, indent, self.indent_delta, tags=tags, header=True)
        else:
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # Initially fill row with blank cells before column names.
            # TODO: Refactor to remove code duplication with code block
            # above for columns MultiIndex.
            row = [""] * (self.row_levels - 1)
            if self.fmt.index or self.show_col_idx_names:
                # see gh-22747
                # If to_html(index_names=False) do not show columns
                # index names.
                # TODO: Refactor to use _get_column_name_list from
                # DataFrameFormatter class.
                if self.fmt.show_index_names:
                    row.append(self.columns.name or "")
                else:
                    row.append("")
            row.extend(self._get_columns_formatted_values())
            align = self.fmt.justify

            if is_truncated_horizontally:
                ins_col = self.row_levels + self.fmt.tr_col_num
                row.insert(ins_col, "...")

            self.write_tr(row, indent, self.indent_delta, header=True, align=align)

    def _write_row_header(self, indent: int) -> None:
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        row = [x if x is not None else "" for x in self.frame.index.names] + [""] * (
            self.ncols + (1 if is_truncated_horizontally else 0)
        )
        self.write_tr(row, indent, self.indent_delta, header=True)

    def _write_header(self, indent: int) -> None:
        self.write("<thead>", indent)

        if self.fmt.header:
            self._write_col_header(indent + self.indent_delta)

        if self.show_row_idx_names:
            self._write_row_header(indent + self.indent_delta)

        self.write("</thead>", indent)

    def _get_formatted_values(self) -> dict[int, list[str]]:
        with option_context("display.max_colwidth", None):
            fmt_values = {i: self.fmt.format_col(i) for i in range(self.ncols)}
        return fmt_values

    def _write_body(self, indent: int) -> None:
        self.write("<tbody>", indent)
        fmt_values = self._get_formatted_values()

        # write values
        if self.fmt.index and isinstance(self.frame.index, MultiIndex):
            self._write_hierarchical_rows(fmt_values, indent + self.indent_delta)
        else:
            self._write_regular_rows(fmt_values, indent + self.indent_delta)

        self.write("</tbody>", indent)

    def _write_regular_rows(
        self, fmt_values: Mapping[int, list[str]], indent: int
    ) -> None:
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        is_truncated_vertically = self.fmt.is_truncated_vertically

        nrows = len(self.fmt.tr_frame)

        if self.fmt.index:
            fmt = self.fmt._get_formatter("__index__")
            if fmt is not None:
                index_values = self.fmt.tr_frame.index.map(fmt)
            else:
                # only reached with non-Multi index
                index_values = self.fmt.tr_frame.index._format_flat(include_name=False)

        row: list[str] = []
        for i in range(nrows):
            if is_truncated_vertically and i == (self.fmt.tr_row_num):
                str_sep_row = ["..."] * len(row)
                self.write_tr(
                    str_sep_row,
                    indent,
                    self.indent_delta,
                    tags=None,
                    nindex_levels=self.row_levels,
                )

            row = []
            if self.fmt.index:
                row.append(index_values[i])
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # Add blank cell before data cells.
            elif self.show_col_idx_names:
                row.append("")
            row.extend(fmt_values[j][i] for j in range(self.ncols))

            if is_truncated_horizontally:
                dot_col_ix = self.fmt.tr_col_num + self.row_levels
                row.insert(dot_col_ix, "...")
            self.write_tr(
                row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels
            )

    def _write_hierarchical_rows(
        self, fmt_values: Mapping[int, list[str]], indent: int
    ) -> None:
        template = 'rowspan="{span}" valign="top"'

        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        is_truncated_vertically = self.fmt.is_truncated_vertically
        frame = self.fmt.tr_frame
        nrows = len(frame)

        assert isinstance(frame.index, MultiIndex)
        idx_values = frame.index._format_multi(sparsify=False, include_names=False)
        idx_values = list(zip(*idx_values))

        if self.fmt.sparsify:
            # GH3547
            sentinel = lib.no_default
            levels = frame.index._format_multi(sparsify=sentinel, include_names=False)

            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            if is_truncated_vertically:
                # Insert ... row and adjust idx_values and
                # level_lengths to take this into account.
                ins_row = self.fmt.tr_row_num
                inserted = False
                for lnum, records in enumerate(level_lengths):
                    rec_new = {}
                    for tag, span in list(records.items()):
                        if tag >= ins_row:
                            rec_new[tag + 1] = span
                        elif tag + span > ins_row:
                            rec_new[tag] = span + 1

                            # GH 14882 - Make sure insertion done once
                            if not inserted:
                                dot_row = list(idx_values[ins_row - 1])
                                dot_row[-1] = "..."
                                idx_values.insert(ins_row, tuple(dot_row))
                                inserted = True
                            else:
                                dot_row = list(idx_values[ins_row])
                                dot_row[inner_lvl - lnum] = "..."
                                idx_values[ins_row] = tuple(dot_row)
                        else:
                            rec_new[tag] = span
                        # If ins_row lies between tags, all cols idx cols
                        # receive ...
                        if tag + span == ins_row:
                            rec_new[ins_row] = 1
                            if lnum == 0:
                                idx_values.insert(
                                    ins_row, tuple(["..."] * len(level_lengths))
                                )

                            # GH 14882 - Place ... in correct level
                            elif inserted:
                                dot_row = list(idx_values[ins_row])
                                dot_row[inner_lvl - lnum] = "..."
                                idx_values[ins_row] = tuple(dot_row)
                    level_lengths[lnum] = rec_new

                level_lengths[inner_lvl][ins_row] = 1
                for ix_col in fmt_values:
                    fmt_values[ix_col].insert(ins_row, "...")
                nrows += 1

            for i in range(nrows):
                row = []
                tags = {}

                sparse_offset = 0
                j = 0
                for records, v in zip(level_lengths, idx_values[i]):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template.format(span=records[i])
                    else:
                        sparse_offset += 1
                        continue

                    j += 1
                    row.append(v)

                row.extend(fmt_values[j][i] for j in range(self.ncols))
                if is_truncated_horizontally:
                    row.insert(
                        self.row_levels - sparse_offset + self.fmt.tr_col_num, "..."
                    )
                self.write_tr(
                    row,
                    indent,
                    self.indent_delta,
                    tags=tags,
                    nindex_levels=len(levels) - sparse_offset,
                )
        else:
            row = []
            for i in range(len(frame)):
                if is_truncated_vertically and i == (self.fmt.tr_row_num):
                    str_sep_row = ["..."] * len(row)
                    self.write_tr(
                        str_sep_row,
                        indent,
                        self.indent_delta,
                        tags=None,
                        nindex_levels=self.row_levels,
                    )

                idx_values = list(
                    zip(*frame.index._format_multi(sparsify=False, include_names=False))
                )
                row = []
                row.extend(idx_values[i])
                row.extend(fmt_values[j][i] for j in range(self.ncols))
                if is_truncated_horizontally:
                    row.insert(self.row_levels + self.fmt.tr_col_num, "...")
                self.write_tr(
                    row,
                    indent,
                    self.indent_delta,
                    tags=None,
                    nindex_levels=frame.index.nlevels,
                )


class NotebookFormatter(HTMLFormatter):
    """
    Internal class for formatting output data in html for display in Jupyter
    Notebooks. This class is intended for functionality specific to
    DataFrame._repr_html_() and DataFrame.to_html(notebook=True)
    """

    def _get_formatted_values(self) -> dict[int, list[str]]:
        return {i: self.fmt.format_col(i) for i in range(self.ncols)}

    def _get_columns_formatted_values(self) -> list[str]:
        # only reached with non-Multi Index
        return self.columns._format_flat(include_name=False)

    def write_style(self) -> None:
        # We use the "scoped" attribute here so that the desired
        # style properties for the data frame are not then applied
        # throughout the entire notebook.
        template_first = """\
            <style scoped>"""
        template_last = """\
            </style>"""
        template_select = """\
                .dataframe %s {
                    %s: %s;
                }"""
        element_props = [
            ("tbody tr th:only-of-type", "vertical-align", "middle"),
            ("tbody tr th", "vertical-align", "top"),
        ]
        if isinstance(self.columns, MultiIndex):
            element_props.append(("thead tr th", "text-align", "left"))
            if self.show_row_idx_names:
                element_props.append(
                    ("thead tr:last-of-type th", "text-align", "right")
                )
        else:
            element_props.append(("thead th", "text-align", "right"))
        template_mid = "\n\n".join(template_select % t for t in element_props)
        template = dedent(f"{template_first}\n{template_mid}\n{template_last}")
        self.write(template)

    def render(self) -> list[str]:
        self.write("<div>")
        self.write_style()
        super().render()
        self.write("</div>")
        return self.elements

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\sympify.py ===
"""sympify -- convert objects SymPy internal format"""

from __future__ import annotations

from typing import Any, Callable, overload, TYPE_CHECKING, TypeVar

import mpmath.libmp as mlib

from inspect import getmro
import string
from sympy.core.random import choice

from .parameters import global_parameters

from sympy.utilities.iterables import iterable


if TYPE_CHECKING:

    from sympy.core.basic import Basic
    from sympy.core.expr import Expr
    from sympy.core.numbers import Integer, Float

    Tbasic = TypeVar('Tbasic', bound=Basic)


class SympifyError(ValueError):
    def __init__(self, expr, base_exc=None):
        self.expr = expr
        self.base_exc = base_exc

    def __str__(self):
        if self.base_exc is None:
            return "SympifyError: %r" % (self.expr,)

        return ("Sympify of expression '%s' failed, because of exception being "
            "raised:\n%s: %s" % (self.expr, self.base_exc.__class__.__name__,
            str(self.base_exc)))


converter: dict[type[Any], Callable[[Any], Basic]] = {}

#holds the conversions defined in SymPy itself, i.e. non-user defined conversions
_sympy_converter: dict[type[Any], Callable[[Any], Basic]] = {}

#alias for clearer use in the library
_external_converter = converter

class CantSympify:
    """
    Mix in this trait to a class to disallow sympification of its instances.

    Examples
    ========

    >>> from sympy import sympify
    >>> from sympy.core.sympify import CantSympify

    >>> class Something(dict):
    ...     pass
    ...
    >>> sympify(Something())
    {}

    >>> class Something(dict, CantSympify):
    ...     pass
    ...
    >>> sympify(Something())
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: {}

    """

    __slots__ = ()


def _is_numpy_instance(a):
    """
    Checks if an object is an instance of a type from the numpy module.
    """
    # This check avoids unnecessarily importing NumPy.  We check the whole
    # __mro__ in case any base type is a numpy type.
    return any(type_.__module__ == 'numpy'
               for type_ in type(a).__mro__)


def _convert_numpy_types(a, **sympify_args):
    """
    Converts a numpy datatype input to an appropriate SymPy type.
    """
    import numpy as np
    if not isinstance(a, np.floating):
        if np.iscomplex(a):
            return _sympy_converter[complex](a.item())
        else:
            return sympify(a.item(), **sympify_args)
    else:
        from .numbers import Float
        prec = np.finfo(a).nmant + 1
        # E.g. double precision means prec=53 but nmant=52
        # Leading bit of mantissa is always 1, so is not stored
        if np.isposinf(a):
            return Float('inf')
        elif np.isneginf(a):
            return Float('-inf')
        else:
            p, q = a.as_integer_ratio()
            a = mlib.from_rational(p, q, prec)
            return Float(a, precision=prec)


@overload
def sympify(a: int, *, strict: bool = False) -> Integer: ... # type: ignore
@overload
def sympify(a: float, *, strict: bool = False) -> Float: ...
@overload
def sympify(a: Expr | complex, *, strict: bool = False) -> Expr: ...
@overload
def sympify(a: Tbasic, *, strict: bool = False) -> Tbasic: ...
@overload
def sympify(a: Any, *, strict: bool = False) -> Basic: ...

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False,
        evaluate=None):
    """
    Converts an arbitrary expression to a type that can be used inside SymPy.

    Explanation
    ===========

    It will convert Python ints into instances of :class:`~.Integer`, floats
    into instances of :class:`~.Float`, etc. It is also able to coerce
    symbolic expressions which inherit from :class:`~.Basic`. This can be
    useful in cooperation with SAGE.

    .. warning::
        Note that this function uses ``eval``, and thus shouldn't be used on
        unsanitized input.

    If the argument is already a type that SymPy understands, it will do
    nothing but return that value. This can be used at the beginning of a
    function to ensure you are working with the correct type.

    Examples
    ========

    >>> from sympy import sympify

    >>> sympify(2).is_integer
    True
    >>> sympify(2).is_real
    True

    >>> sympify(2.0).is_real
    True
    >>> sympify("2.0").is_real
    True
    >>> sympify("2e-45").is_real
    True

    If the expression could not be converted, a SympifyError is raised.

    >>> sympify("x***2")
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: "could not parse 'x***2'"

    When attempting to parse non-Python syntax using ``sympify``, it raises a
    ``SympifyError``:

    >>> sympify("2x+1")
    Traceback (most recent call last):
    ...
    SympifyError: Sympify of expression 'could not parse '2x+1'' failed

    To parse non-Python syntax, use ``parse_expr`` from ``sympy.parsing.sympy_parser``.

    >>> from sympy.parsing.sympy_parser import parse_expr
    >>> parse_expr("2x+1", transformations="all")
    2*x + 1

    For more details about ``transformations``: see :func:`~sympy.parsing.sympy_parser.parse_expr`

    Locals
    ------

    The sympification happens with access to everything that is loaded
    by ``from sympy import *``; anything used in a string that is not
    defined by that import will be converted to a symbol. In the following,
    the ``bitcount`` function is treated as a symbol and the ``O`` is
    interpreted as the :class:`~.Order` object (used with series) and it raises
    an error when used improperly:

    >>> s = 'bitcount(42)'
    >>> sympify(s)
    bitcount(42)
    >>> sympify("O(x)")
    O(x)
    >>> sympify("O + 1")
    Traceback (most recent call last):
    ...
    TypeError: unbound method...

    In order to have ``bitcount`` be recognized it can be imported into a
    namespace dictionary and passed as locals:

    >>> ns = {}
    >>> exec('from sympy.core.evalf import bitcount', ns)
    >>> sympify(s, locals=ns)
    6

    In order to have the ``O`` interpreted as a Symbol, identify it as such
    in the namespace dictionary. This can be done in a variety of ways; all
    three of the following are possibilities:

    >>> from sympy import Symbol
    >>> ns["O"] = Symbol("O")  # method 1
    >>> exec('from sympy.abc import O', ns)  # method 2
    >>> ns.update(dict(O=Symbol("O")))  # method 3
    >>> sympify("O + 1", locals=ns)
    O + 1

    If you want *all* single-letter and Greek-letter variables to be symbols
    then you can use the clashing-symbols dictionaries that have been defined
    there as private variables: ``_clash1`` (single-letter variables),
    ``_clash2`` (the multi-letter Greek names) or ``_clash`` (both single and
    multi-letter names that are defined in ``abc``).

    >>> from sympy.abc import _clash1
    >>> set(_clash1)  # if this fails, see issue #23903
    {'E', 'I', 'N', 'O', 'Q', 'S'}
    >>> sympify('I & Q', _clash1)
    I & Q

    Strict
    ------

    If the option ``strict`` is set to ``True``, only the types for which an
    explicit conversion has been defined are converted. In the other
    cases, a SympifyError is raised.

    >>> print(sympify(None))
    None
    >>> sympify(None, strict=True)
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: None

    .. deprecated:: 1.6

       ``sympify(obj)`` automatically falls back to ``str(obj)`` when all
       other conversion methods fail, but this is deprecated. ``strict=True``
       will disable this deprecated behavior. See
       :ref:`deprecated-sympify-string-fallback`.

    Evaluation
    ----------

    If the option ``evaluate`` is set to ``False``, then arithmetic and
    operators will be converted into their SymPy equivalents and the
    ``evaluate=False`` option will be added. Nested ``Add`` or ``Mul`` will
    be denested first. This is done via an AST transformation that replaces
    operators with their SymPy equivalents, so if an operand redefines any
    of those operations, the redefined operators will not be used. If
    argument a is not a string, the mathematical expression is evaluated
    before being passed to sympify, so adding ``evaluate=False`` will still
    return the evaluated result of expression.

    >>> sympify('2**2 / 3 + 5')
    19/3
    >>> sympify('2**2 / 3 + 5', evaluate=False)
    2**2/3 + 5
    >>> sympify('4/2+7', evaluate=True)
    9
    >>> sympify('4/2+7', evaluate=False)
    4/2 + 7
    >>> sympify(4/2+7, evaluate=False)
    9.00000000000000

    Extending
    ---------

    To extend ``sympify`` to convert custom objects (not derived from ``Basic``),
    just define a ``_sympy_`` method to your class. You can do that even to
    classes that you do not own by subclassing or adding the method at runtime.

    >>> from sympy import Matrix
    >>> class MyList1(object):
    ...     def __iter__(self):
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    ...     def _sympy_(self): return Matrix(self)
    >>> sympify(MyList1())
    Matrix([
    [1],
    [2]])

    If you do not have control over the class definition you could also use the
    ``converter`` global dictionary. The key is the class and the value is a
    function that takes a single argument and returns the desired SymPy
    object, e.g. ``converter[MyList] = lambda x: Matrix(x)``.

    >>> class MyList2(object):   # XXX Do not do this if you control the class!
    ...     def __iter__(self):  #     Use _sympy_!
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    >>> from sympy.core.sympify import converter
    >>> converter[MyList2] = lambda x: Matrix(x)
    >>> sympify(MyList2())
    Matrix([
    [1],
    [2]])

    Notes
    =====

    The keywords ``rational`` and ``convert_xor`` are only used
    when the input is a string.

    convert_xor
    -----------

    >>> sympify('x^y',convert_xor=True)
    x**y
    >>> sympify('x^y',convert_xor=False)
    x ^ y

    rational
    --------

    >>> sympify('0.1',rational=False)
    0.1
    >>> sympify('0.1',rational=True)
    1/10

    Sometimes autosimplification during sympification results in expressions
    that are very different in structure than what was entered. Until such
    autosimplification is no longer done, the ``kernS`` function might be of
    some use. In the example below you can see how an expression reduces to
    $-1$ by autosimplification, but does not do so when ``kernS`` is used.

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x
    >>> -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1
    -1
    >>> s = '-2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1'
    >>> sympify(s)
    -1
    >>> kernS(s)
    -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1

    Parameters
    ==========

    a :
        - any object defined in SymPy
        - standard numeric Python types: ``int``, ``long``, ``float``, ``Decimal``
        - strings (like ``"0.09"``, ``"2e-19"`` or ``'sin(x)'``)
        - booleans, including ``None`` (will leave ``None`` unchanged)
        - dicts, lists, sets or tuples containing any of the above

    convert_xor : bool, optional
        If true, treats ``^`` as exponentiation.
        If False, treats ``^`` as XOR itself.
        Used only when input is a string.

    locals : any object defined in SymPy, optional
        In order to have strings be recognized it can be imported
        into a namespace dictionary and passed as locals.

    strict : bool, optional
        If the option strict is set to ``True``, only the types for which
        an explicit conversion has been defined are converted. In the
        other cases, a SympifyError is raised.

    rational : bool, optional
        If ``True``, converts floats into :class:`~.Rational`.
        If ``False``, it lets floats remain as it is.
        Used only when input is a string.

    evaluate : bool, optional
        If False, then arithmetic and operators will be converted into
        their SymPy equivalents. If True the expression will be evaluated
        and the result will be returned.

    """
    # XXX: If a is a Basic subclass rather than instance (e.g. sin rather than
    # sin(x)) then a.__sympy__ will be the property. Only on the instance will
    # a.__sympy__ give the *value* of the property (True). Since sympify(sin)
    # was used for a long time we allow it to pass. However if strict=True as
    # is the case in internal calls to _sympify then we only allow
    # is_sympy=True.
    #
    # https://github.com/sympy/sympy/issues/20124
    is_sympy = getattr(a, '__sympy__', None)
    if is_sympy is True:
        return a
    elif is_sympy is not None:
        if not strict:
            return a
        else:
            raise SympifyError(a)

    if isinstance(a, CantSympify):
        raise SympifyError(a)

    cls = getattr(a, "__class__", None)

    #Check if there exists a converter for any of the types in the mro
    for superclass in getmro(cls):
        #First check for user defined converters
        conv = _external_converter.get(superclass)
        if conv is None:
            #if none exists, check for SymPy defined converters
            conv = _sympy_converter.get(superclass)
        if conv is not None:
            return conv(a)

    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a

    if evaluate is None:
        evaluate = global_parameters.evaluate

    # Support for basic numpy datatypes
    if _is_numpy_instance(a):
        import numpy as np
        if np.isscalar(a):
            return _convert_numpy_types(a, locals=locals,
                convert_xor=convert_xor, strict=strict, rational=rational,
                evaluate=evaluate)

    _sympy_ = getattr(a, "_sympy_", None)
    if _sympy_ is not None:
        return a._sympy_()

    if not strict:
        # Put numpy array conversion _before_ float/int, see
        # <https://github.com/sympy/sympy/issues/13924>.
        flat = getattr(a, "flat", None)
        if flat is not None:
            shape = getattr(a, "shape", None)
            if shape is not None:
                from sympy.tensor.array import Array
                return Array(a.flat, a.shape)  # works with e.g. NumPy arrays

    if not isinstance(a, str):
        if _is_numpy_instance(a):
            import numpy as np
            assert not isinstance(a, np.number)
            if isinstance(a, np.ndarray):
                # Scalar arrays (those with zero dimensions) have sympify
                # called on the scalar element.
                if a.ndim == 0:
                    try:
                        return sympify(a.item(),
                                       locals=locals,
                                       convert_xor=convert_xor,
                                       strict=strict,
                                       rational=rational,
                                       evaluate=evaluate)
                    except SympifyError:
                        pass
        elif hasattr(a, '__float__'):
            # float and int can coerce size-one numpy arrays to their lone
            # element.  See issue https://github.com/numpy/numpy/issues/10404.
            return sympify(float(a))
        elif hasattr(a, '__int__'):
            return sympify(int(a))

    if strict:
        raise SympifyError(a)

    if iterable(a):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor,
                rational=rational, evaluate=evaluate) for x in a])
        except TypeError:
            # Not all iterables are rebuildable with their type.
            pass

    if not isinstance(a, str):
        raise SympifyError('cannot sympify object of type %r' % type(a))

    from sympy.parsing.sympy_parser import (parse_expr, TokenError,
                                            standard_transformations)
    from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
    from sympy.parsing.sympy_parser import rationalize as t_rationalize

    transformations = standard_transformations

    if rational:
        transformations += (t_rationalize,)
    if convert_xor:
        transformations += (t_convert_xor,)

    try:
        a = a.replace('\n', '')
        expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
    except (TokenError, SyntaxError) as exc:
        raise SympifyError('could not parse %r' % a, exc)

    return expr


def _sympify(a):
    """
    Short version of :func:`~.sympify` for internal usage for ``__add__`` and
    ``__eq__`` methods where it is ok to allow some things (like Python
    integers and floats) in the expression. This excludes things (like strings)
    that are unwise to allow into such an expression.

    >>> from sympy import Integer
    >>> Integer(1) == 1
    True

    >>> Integer(1) == '1'
    False

    >>> from sympy.abc import x
    >>> x + 1
    x + 1

    >>> x + '1'
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for +: 'Symbol' and 'str'

    see: sympify

    """
    return sympify(a, strict=True)


def kernS(s):
    """Use a hack to try keep autosimplification from distributing a
    a number into an Add; this modification does not
    prevent the 2-arg Mul from becoming an Add, however.

    Examples
    ========

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x, y

    The 2-arg Mul distributes a number (or minus sign) across the terms
    of an expression, but kernS will prevent that:

    >>> 2*(x + y), -(x + 1)
    (2*x + 2*y, -x - 1)
    >>> kernS('2*(x + y)')
    2*(x + y)
    >>> kernS('-(x + 1)')
    -(x + 1)

    If use of the hack fails, the un-hacked string will be passed to sympify...
    and you get what you get.

    XXX This hack should not be necessary once issue 4596 has been resolved.
    """
    hit = False
    quoted = '"' in s or "'" in s
    if '(' in s and not quoted:
        if s.count('(') != s.count(")"):
            raise SympifyError('unmatched left parenthesis')

        # strip all space from s
        s = ''.join(s.split())
        olds = s
        # now use space to represent a symbol that
        # will
        # step 1. turn potential 2-arg Muls into 3-arg versions
        # 1a. *( -> * *(
        s = s.replace('*(', '* *(')
        # 1b. close up exponentials
        s = s.replace('** *', '**')
        # 2. handle the implied multiplication of a negated
        # parenthesized expression in two steps
        # 2a:  -(...)  -->  -( *(...)
        target = '-( *('
        s = s.replace('-(', target)
        # 2b: double the matching closing parenthesis
        # -( *(...)  -->  -( *(...))
        i = nest = 0
        assert target.endswith('(')  # assumption below
        while True:
            j = s.find(target, i)
            if j == -1:
                break
            j += len(target) - 1
            for j in range(j, len(s)):
                if s[j] == "(":
                    nest += 1
                elif s[j] == ")":
                    nest -= 1
                if nest == 0:
                    break
            s = s[:j] + ")" + s[j:]
            i = j + 2  # the first char after 2nd )
        if ' ' in s:
            # get a unique kern
            kern = '_'
            while kern in s:
                kern += choice(string.ascii_letters + string.digits)
            s = s.replace(' ', kern)
            hit = kern in s
        else:
            hit = False

    for i in range(2):
        try:
            expr = sympify(s)
            break
        except TypeError:  # the kern might cause unknown errors...
            if hit:
                s = olds  # maybe it didn't like the kern; use un-kerned s
                hit = False
                continue
            expr = sympify(s)  # let original error raise

    if not hit:
        return expr

    from .symbol import Symbol
    rep = {Symbol(kern): 1}
    def _clear(expr):
        if isinstance(expr, (list, tuple, set)):
            return type(expr)([_clear(e) for e in expr])
        if hasattr(expr, 'subs'):
            return expr.subs(rep, hack2=True)
        return expr
    expr = _clear(expr)
    # hope that kern is not there anymore
    return expr


# Avoid circular import
from .basic import Basic

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\functions.py ===
from ..libmp.backend import xrange

class SpecialFunctions(object):
    """
    This class implements special functions using high-level code.

    Elementary and some other functions (e.g. gamma function, basecase
    hypergeometric series) are assumed to be predefined by the context as
    "builtins" or "low-level" functions.
    """
    defined_functions = {}

    # The series for the Jacobi theta functions converge for |q| < 1;
    # in the current implementation they throw a ValueError for
    # abs(q) > THETA_Q_LIM
    THETA_Q_LIM = 1 - 10**-7

    def __init__(self):
        cls = self.__class__
        for name in cls.defined_functions:
            f, wrap = cls.defined_functions[name]
            cls._wrap_specfun(name, f, wrap)

        self.mpq_1 = self._mpq((1,1))
        self.mpq_0 = self._mpq((0,1))
        self.mpq_1_2 = self._mpq((1,2))
        self.mpq_3_2 = self._mpq((3,2))
        self.mpq_1_4 = self._mpq((1,4))
        self.mpq_1_16 = self._mpq((1,16))
        self.mpq_3_16 = self._mpq((3,16))
        self.mpq_5_2 = self._mpq((5,2))
        self.mpq_3_4 = self._mpq((3,4))
        self.mpq_7_4 = self._mpq((7,4))
        self.mpq_5_4 = self._mpq((5,4))
        self.mpq_1_3 = self._mpq((1,3))
        self.mpq_2_3 = self._mpq((2,3))
        self.mpq_4_3 = self._mpq((4,3))
        self.mpq_1_6 = self._mpq((1,6))
        self.mpq_5_6 = self._mpq((5,6))
        self.mpq_5_3 = self._mpq((5,3))

        self._misc_const_cache = {}

        self._aliases.update({
            'phase' : 'arg',
            'conjugate' : 'conj',
            'nthroot' : 'root',
            'polygamma' : 'psi',
            'hurwitz' : 'zeta',
            #'digamma' : 'psi0',
            #'trigamma' : 'psi1',
            #'tetragamma' : 'psi2',
            #'pentagamma' : 'psi3',
            'fibonacci' : 'fib',
            'factorial' : 'fac',
        })

        self.zetazero_memoized = self.memoize(self.zetazero)

    # Default -- do nothing
    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        setattr(cls, name, f)

    # Optional fast versions of common functions in common cases.
    # If not overridden, default (generic hypergeometric series)
    # implementations will be used
    def _besselj(ctx, n, z): raise NotImplementedError
    def _erf(ctx, z): raise NotImplementedError
    def _erfc(ctx, z): raise NotImplementedError
    def _gamma_upper_int(ctx, z, a): raise NotImplementedError
    def _expint_int(ctx, n, z): raise NotImplementedError
    def _zeta(ctx, s): raise NotImplementedError
    def _zetasum_fast(ctx, s, a, n, derivatives, reflect): raise NotImplementedError
    def _ei(ctx, z): raise NotImplementedError
    def _e1(ctx, z): raise NotImplementedError
    def _ci(ctx, z): raise NotImplementedError
    def _si(ctx, z): raise NotImplementedError
    def _altzeta(ctx, s): raise NotImplementedError

def defun_wrapped(f):
    SpecialFunctions.defined_functions[f.__name__] = f, True
    return f

def defun(f):
    SpecialFunctions.defined_functions[f.__name__] = f, False
    return f

def defun_static(f):
    setattr(SpecialFunctions, f.__name__, f)
    return f

@defun_wrapped
def cot(ctx, z): return ctx.one / ctx.tan(z)

@defun_wrapped
def sec(ctx, z): return ctx.one / ctx.cos(z)

@defun_wrapped
def csc(ctx, z): return ctx.one / ctx.sin(z)

@defun_wrapped
def coth(ctx, z): return ctx.one / ctx.tanh(z)

@defun_wrapped
def sech(ctx, z): return ctx.one / ctx.cosh(z)

@defun_wrapped
def csch(ctx, z): return ctx.one / ctx.sinh(z)

@defun_wrapped
def acot(ctx, z):
    if not z:
        return ctx.pi * 0.5
    else:
        return ctx.atan(ctx.one / z)

@defun_wrapped
def asec(ctx, z): return ctx.acos(ctx.one / z)

@defun_wrapped
def acsc(ctx, z): return ctx.asin(ctx.one / z)

@defun_wrapped
def acoth(ctx, z):
    if not z:
        return ctx.pi * 0.5j
    else:
        return ctx.atanh(ctx.one / z)


@defun_wrapped
def asech(ctx, z): return ctx.acosh(ctx.one / z)

@defun_wrapped
def acsch(ctx, z): return ctx.asinh(ctx.one / z)

@defun
def sign(ctx, x):
    x = ctx.convert(x)
    if not x or ctx.isnan(x):
        return x
    if ctx._is_real_type(x):
        if x > 0:
            return ctx.one
        else:
            return -ctx.one
    return x / abs(x)

@defun
def agm(ctx, a, b=1):
    if b == 1:
        return ctx.agm1(a)
    a = ctx.convert(a)
    b = ctx.convert(b)
    return ctx._agm(a, b)

@defun_wrapped
def sinc(ctx, x):
    if ctx.isinf(x):
        return 1/x
    if not x:
        return x+1
    return ctx.sin(x)/x

@defun_wrapped
def sincpi(ctx, x):
    if ctx.isinf(x):
        return 1/x
    if not x:
        return x+1
    return ctx.sinpi(x)/(ctx.pi*x)

# TODO: tests; improve implementation
@defun_wrapped
def expm1(ctx, x):
    if not x:
        return ctx.zero
    # exp(x) - 1 ~ x
    if ctx.mag(x) < -ctx.prec:
        return x + 0.5*x**2
    # TODO: accurately eval the smaller of the real/imag parts
    return ctx.sum_accurately(lambda: iter([ctx.exp(x),-1]),1)

@defun_wrapped
def log1p(ctx, x):
    if not x:
        return ctx.zero
    if ctx.mag(x) < -ctx.prec:
        return x - 0.5*x**2
    return ctx.log(ctx.fadd(1, x, prec=2*ctx.prec))

@defun_wrapped
def powm1(ctx, x, y):
    mag = ctx.mag
    one = ctx.one
    w = x**y - one
    M = mag(w)
    # Only moderate cancellation
    if M > -8:
        return w
    # Check for the only possible exact cases
    if not w:
        if (not y) or (x in (1, -1, 1j, -1j) and ctx.isint(y)):
            return w
    x1 = x - one
    magy = mag(y)
    lnx = ctx.ln(x)
    # Small y: x^y - 1 ~ log(x)*y + O(log(x)^2 * y^2)
    if magy + mag(lnx) < -ctx.prec:
        return lnx*y + (lnx*y)**2/2
    # TODO: accurately eval the smaller of the real/imag part
    return ctx.sum_accurately(lambda: iter([x**y, -1]), 1)

@defun
def _rootof1(ctx, k, n):
    k = int(k)
    n = int(n)
    k %= n
    if not k:
        return ctx.one
    elif 2*k == n:
        return -ctx.one
    elif 4*k == n:
        return ctx.j
    elif 4*k == 3*n:
        return -ctx.j
    return ctx.expjpi(2*ctx.mpf(k)/n)

@defun
def root(ctx, x, n, k=0):
    n = int(n)
    x = ctx.convert(x)
    if k:
        # Special case: there is an exact real root
        if (n & 1 and 2*k == n-1) and (not ctx.im(x)) and (ctx.re(x) < 0):
            return -ctx.root(-x, n)
        # Multiply by root of unity
        prec = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.root(x, n, 0) * ctx._rootof1(k, n)
        finally:
            ctx.prec = prec
        return +v
    return ctx._nthroot(x, n)

@defun
def unitroots(ctx, n, primitive=False):
    gcd = ctx._gcd
    prec = ctx.prec
    try:
        ctx.prec += 10
        if primitive:
            v = [ctx._rootof1(k,n) for k in range(n) if gcd(k,n) == 1]
        else:
            # TODO: this can be done *much* faster
            v = [ctx._rootof1(k,n) for k in range(n)]
    finally:
        ctx.prec = prec
    return [+x for x in v]

@defun
def arg(ctx, x):
    x = ctx.convert(x)
    re = ctx._re(x)
    im = ctx._im(x)
    return ctx.atan2(im, re)

@defun
def fabs(ctx, x):
    return abs(ctx.convert(x))

@defun
def re(ctx, x):
    x = ctx.convert(x)
    if hasattr(x, "real"):    # py2.5 doesn't have .real/.imag for all numbers
        return x.real
    return x

@defun
def im(ctx, x):
    x = ctx.convert(x)
    if hasattr(x, "imag"):    # py2.5 doesn't have .real/.imag for all numbers
        return x.imag
    return ctx.zero

@defun
def conj(ctx, x):
    x = ctx.convert(x)
    try:
        return x.conjugate()
    except AttributeError:
        return x

@defun
def polar(ctx, z):
    return (ctx.fabs(z), ctx.arg(z))

@defun_wrapped
def rect(ctx, r, phi):
    return r * ctx.mpc(*ctx.cos_sin(phi))

@defun
def log(ctx, x, b=None):
    if b is None:
        return ctx.ln(x)
    wp = ctx.prec + 20
    return ctx.ln(x, prec=wp) / ctx.ln(b, prec=wp)

@defun
def log10(ctx, x):
    return ctx.log(x, 10)

@defun
def fmod(ctx, x, y):
    return ctx.convert(x) % ctx.convert(y)

@defun
def degrees(ctx, x):
    return x / ctx.degree

@defun
def radians(ctx, x):
    return x * ctx.degree

def _lambertw_special(ctx, z, k):
    # W(0,0) = 0; all other branches are singular
    if not z:
        if not k:
            return z
        return ctx.ninf + z
    if z == ctx.inf:
        if k == 0:
            return z
        else:
            return z + 2*k*ctx.pi*ctx.j
    if z == ctx.ninf:
        return (-z) + (2*k+1)*ctx.pi*ctx.j
    # Some kind of nan or complex inf/nan?
    return ctx.ln(z)

import math
import cmath

def _lambertw_approx_hybrid(z, k):
    imag_sign = 0
    if hasattr(z, "imag"):
        x = float(z.real)
        y = z.imag
        if y:
            imag_sign = (-1) ** (y < 0)
        y = float(y)
    else:
        x = float(z)
        y = 0.0
        imag_sign = 0
    # hack to work regardless of whether Python supports -0.0
    if not y:
        y = 0.0
    z = complex(x,y)
    if k == 0:
        if -4.0 < y < 4.0 and -1.0 < x < 2.5:
            if imag_sign:
                # Taylor series in upper/lower half-plane
                if y > 1.00: return (0.876+0.645j) + (0.118-0.174j)*(z-(0.75+2.5j))
                if y > 0.25: return (0.505+0.204j) + (0.375-0.132j)*(z-(0.75+0.5j))
                if y < -1.00: return (0.876-0.645j) + (0.118+0.174j)*(z-(0.75-2.5j))
                if y < -0.25: return (0.505-0.204j) + (0.375+0.132j)*(z-(0.75-0.5j))
            # Taylor series near -1
            if x < -0.5:
                if imag_sign >= 0:
                    return (-0.318+1.34j) + (-0.697-0.593j)*(z+1)
                else:
                    return (-0.318-1.34j) + (-0.697+0.593j)*(z+1)
            # return real type
            r = -0.367879441171442
            if (not imag_sign) and x > r:
                z = x
            # Singularity near -1/e
            if x < -0.2:
                return -1 + 2.33164398159712*(z-r)**0.5 - 1.81218788563936*(z-r)
            # Taylor series near 0
            if x < 0.5: return z
            # Simple linear approximation
            return 0.2 + 0.3*z
        if (not imag_sign) and x > 0.0:
            L1 = math.log(x); L2 = math.log(L1)
        else:
            L1 = cmath.log(z); L2 = cmath.log(L1)
    elif k == -1:
        # return real type
        r = -0.367879441171442
        if (not imag_sign) and r < x < 0.0:
            z = x
        if (imag_sign >= 0) and y < 0.1 and -0.6 < x < -0.2:
            return -1 - 2.33164398159712*(z-r)**0.5 - 1.81218788563936*(z-r)
        if (not imag_sign) and -0.2 <= x < 0.0:
            L1 = math.log(-x)
            return L1 - math.log(-L1)
        else:
            if imag_sign == -1 and (not y) and x < 0.0:
                L1 = cmath.log(z) - 3.1415926535897932j
            else:
                L1 = cmath.log(z) - 6.2831853071795865j
            L2 = cmath.log(L1)
    return L1 - L2 + L2/L1 + L2*(L2-2)/(2*L1**2)

def _lambertw_series(ctx, z, k, tol):
    """
    Return rough approximation for W_k(z) from an asymptotic series,
    sufficiently accurate for the Halley iteration to converge to
    the correct value.
    """
    magz = ctx.mag(z)
    if (-10 < magz < 900) and (-1000 < k < 1000):
        # Near the branch point at -1/e
        if magz < 1 and abs(z+0.36787944117144) < 0.05:
            if k == 0 or (k == -1 and ctx._im(z) >= 0) or \
                         (k == 1  and ctx._im(z) < 0):
                delta = ctx.sum_accurately(lambda: [z, ctx.exp(-1)])
                cancellation = -ctx.mag(delta)
                ctx.prec += cancellation
                # Use series given in Corless et al.
                p = ctx.sqrt(2*(ctx.e*z+1))
                ctx.prec -= cancellation
                u = {0:ctx.mpf(-1), 1:ctx.mpf(1)}
                a = {0:ctx.mpf(2), 1:ctx.mpf(-1)}
                if k != 0:
                    p = -p
                s = ctx.zero
                # The series converges, so we could use it directly, but unless
                # *extremely* close, it is better to just use the first few
                # terms to get a good approximation for the iteration
                for l in xrange(max(2,cancellation)):
                    if l not in u:
                        a[l] = ctx.fsum(u[j]*u[l+1-j] for j in xrange(2,l))
                        u[l] = (l-1)*(u[l-2]/2+a[l-2]/4)/(l+1)-a[l]/2-u[l-1]/(l+1)
                    term = u[l] * p**l
                    s += term
                    if ctx.mag(term) < -tol:
                        return s, True
                    l += 1
                ctx.prec += cancellation//2
                return s, False
        if k == 0 or k == -1:
            return _lambertw_approx_hybrid(z, k), False
    if k == 0:
        if magz < -1:
            return z*(1-z), False
        L1 = ctx.ln(z)
        L2 = ctx.ln(L1)
    elif k == -1 and (not ctx._im(z)) and (-0.36787944117144 < ctx._re(z) < 0):
        L1 = ctx.ln(-z)
        return L1 - ctx.ln(-L1), False
    else:
        # This holds both as z -> 0 and z -> inf.
        # Relative error is O(1/log(z)).
        L1 = ctx.ln(z) + 2j*ctx.pi*k
        L2 = ctx.ln(L1)
    return L1 - L2 + L2/L1 + L2*(L2-2)/(2*L1**2), False

@defun
def lambertw(ctx, z, k=0):
    z = ctx.convert(z)
    k = int(k)
    if not ctx.isnormal(z):
        return _lambertw_special(ctx, z, k)
    prec = ctx.prec
    ctx.prec += 20 + ctx.mag(k or 1)
    wp = ctx.prec
    tol = wp - 5
    w, done = _lambertw_series(ctx, z, k, tol)
    if not done:
        # Use Halley iteration to solve w*exp(w) = z
        two = ctx.mpf(2)
        for i in xrange(100):
            ew = ctx.exp(w)
            wew = w*ew
            wewz = wew-z
            wn = w - wewz/(wew+ew-(w+two)*wewz/(two*w+two))
            if ctx.mag(wn-w) <= ctx.mag(wn) - tol:
                w = wn
                break
            else:
                w = wn
        if i == 100:
            ctx.warn("Lambert W iteration failed to converge for z = %s" % z)
    ctx.prec = prec
    return +w

@defun_wrapped
def bell(ctx, n, x=1):
    x = ctx.convert(x)
    if not n:
        if ctx.isnan(x):
            return x
        return type(x)(1)
    if ctx.isinf(x) or ctx.isinf(n) or ctx.isnan(x) or ctx.isnan(n):
        return x**n
    if n == 1: return x
    if n == 2: return x*(x+1)
    if x == 0: return ctx.sincpi(n)
    return _polyexp(ctx, n, x, True) / ctx.exp(x)

def _polyexp(ctx, n, x, extra=False):
    def _terms():
        if extra:
            yield ctx.sincpi(n)
        t = x
        k = 1
        while 1:
            yield k**n * t
            k += 1
            t = t*x/k
    return ctx.sum_accurately(_terms, check_step=4)

@defun_wrapped
def polyexp(ctx, s, z):
    if ctx.isinf(z) or ctx.isinf(s) or ctx.isnan(z) or ctx.isnan(s):
        return z**s
    if z == 0: return z*s
    if s == 0: return ctx.expm1(z)
    if s == 1: return ctx.exp(z)*z
    if s == 2: return ctx.exp(z)*z*(z+1)
    return _polyexp(ctx, s, z)

@defun_wrapped
def cyclotomic(ctx, n, z):
    n = int(n)
    if n < 0:
        raise ValueError("n cannot be negative")
    p = ctx.one
    if n == 0:
        return p
    if n == 1:
        return z - p
    if n == 2:
        return z + p
    # Use divisor product representation. Unfortunately, this sometimes
    # includes singularities for roots of unity, which we have to cancel out.
    # Matching zeros/poles pairwise, we have (1-z^a)/(1-z^b) ~ a/b + O(z-1).
    a_prod = 1
    b_prod = 1
    num_zeros = 0
    num_poles = 0
    for d in range(1,n+1):
        if not n % d:
            w = ctx.moebius(n//d)
            # Use powm1 because it is important that we get 0 only
            # if it really is exactly 0
            b = -ctx.powm1(z, d)
            if b:
                p *= b**w
            else:
                if w == 1:
                    a_prod *= d
                    num_zeros += 1
                elif w == -1:
                    b_prod *= d
                    num_poles += 1
    #print n, num_zeros, num_poles
    if num_zeros:
        if num_zeros > num_poles:
            p *= 0
        else:
            p *= a_prod
            p /= b_prod
    return p

@defun
def mangoldt(ctx, n):
    r"""
    Evaluates the von Mangoldt function `\Lambda(n) = \log p`
    if `n = p^k` a power of a prime, and `\Lambda(n) = 0` otherwise.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> [mangoldt(n) for n in range(-2,3)]
        [0.0, 0.0, 0.0, 0.0, 0.6931471805599453094172321]
        >>> mangoldt(6)
        0.0
        >>> mangoldt(7)
        1.945910149055313305105353
        >>> mangoldt(8)
        0.6931471805599453094172321
        >>> fsum(mangoldt(n) for n in range(101))
        94.04531122935739224600493
        >>> fsum(mangoldt(n) for n in range(10001))
        10013.39669326311478372032

    """
    n = int(n)
    if n < 2:
        return ctx.zero
    if n % 2 == 0:
        # Must be a power of two
        if n & (n-1) == 0:
            return +ctx.ln2
        else:
            return ctx.zero
    # TODO: the following could be generalized into a perfect
    # power testing function
    # ---
    # Look for a small factor
    for p in (3,5,7,11,13,17,19,23,29,31):
        if not n % p:
            q, r = n // p, 0
            while q > 1:
                q, r = divmod(q, p)
                if r:
                    return ctx.zero
            return ctx.ln(p)
    if ctx.isprime(n):
        return ctx.ln(n)
    # Obviously, we could use arbitrary-precision arithmetic for this...
    if n > 10**30:
        raise NotImplementedError
    k = 2
    while 1:
        p = int(n**(1./k) + 0.5)
        if p < 2:
            return ctx.zero
        if p ** k == n:
            if ctx.isprime(p):
                return ctx.ln(p)
        k += 1

@defun
def stirling1(ctx, n, k, exact=False):
    v = ctx._stirling1(int(n), int(k))
    if exact:
        return int(v)
    else:
        return ctx.mpf(v)

@defun
def stirling2(ctx, n, k, exact=False):
    v = ctx._stirling2(int(n), int(k))
    if exact:
        return int(v)
    else:
        return ctx.mpf(v)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\shell_completion.py ===
from __future__ import annotations

import collections.abc as cabc
import os
import re
import typing as t
from gettext import gettext as _

from .core import Argument
from .core import Command
from .core import Context
from .core import Group
from .core import Option
from .core import Parameter
from .core import ParameterSource
from .utils import echo


def shell_complete(
    cli: Command,
    ctx_args: cabc.MutableMapping[str, t.Any],
    prog_name: str,
    complete_var: str,
    instruction: str,
) -> int:
    """Perform shell completion for the given CLI program.

    :param cli: Command being called.
    :param ctx_args: Extra arguments to pass to
        ``cli.make_context``.
    :param prog_name: Name of the executable in the shell.
    :param complete_var: Name of the environment variable that holds
        the completion instruction.
    :param instruction: Value of ``complete_var`` with the completion
        instruction and shell, in the form ``instruction_shell``.
    :return: Status code to exit with.
    """
    shell, _, instruction = instruction.partition("_")
    comp_cls = get_completion_class(shell)

    if comp_cls is None:
        return 1

    comp = comp_cls(cli, ctx_args, prog_name, complete_var)

    if instruction == "source":
        echo(comp.source())
        return 0

    if instruction == "complete":
        echo(comp.complete())
        return 0

    return 1


class CompletionItem:
    """Represents a completion value and metadata about the value. The
    default metadata is ``type`` to indicate special shell handling,
    and ``help`` if a shell supports showing a help string next to the
    value.

    Arbitrary parameters can be passed when creating the object, and
    accessed using ``item.attr``. If an attribute wasn't passed,
    accessing it returns ``None``.

    :param value: The completion suggestion.
    :param type: Tells the shell script to provide special completion
        support for the type. Click uses ``"dir"`` and ``"file"``.
    :param help: String shown next to the value if supported.
    :param kwargs: Arbitrary metadata. The built-in implementations
        don't use this, but custom type completions paired with custom
        shell support could use it.
    """

    __slots__ = ("value", "type", "help", "_info")

    def __init__(
        self,
        value: t.Any,
        type: str = "plain",
        help: str | None = None,
        **kwargs: t.Any,
    ) -> None:
        self.value: t.Any = value
        self.type: str = type
        self.help: str | None = help
        self._info = kwargs

    def __getattr__(self, name: str) -> t.Any:
        return self._info.get(name)


# Only Bash >= 4.4 has the nosort option.
_SOURCE_BASH = """\
%(complete_func)s() {
    local IFS=$'\\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD \
%(complete_var)s=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"

        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}

%(complete_func)s_setup() {
    complete -o nosort -F %(complete_func)s %(prog_name)s
}

%(complete_func)s_setup;
"""

_SOURCE_ZSH = """\
#compdef %(prog_name)s

%(complete_func)s() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    (( ! $+commands[%(prog_name)s] )) && return 1

    response=("${(@f)$(env COMP_WORDS="${words[*]}" COMP_CWORD=$((CURRENT-1)) \
%(complete_var)s=zsh_complete %(prog_name)s)}")

    for type key descr in ${response}; do
        if [[ "$type" == "plain" ]]; then
            if [[ "$descr" == "_" ]]; then
                completions+=("$key")
            else
                completions_with_descriptions+=("$key":"$descr")
            fi
        elif [[ "$type" == "dir" ]]; then
            _path_files -/
        elif [[ "$type" == "file" ]]; then
            _path_files -f
        fi
    done

    if [ -n "$completions_with_descriptions" ]; then
        _describe -V unsorted completions_with_descriptions -U
    fi

    if [ -n "$completions" ]; then
        compadd -U -V unsorted -a completions
    fi
}

if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
    # autoload from fpath, call function directly
    %(complete_func)s "$@"
else
    # eval/source/. command, register function for later
    compdef %(complete_func)s %(prog_name)s
fi
"""

_SOURCE_FISH = """\
function %(complete_func)s;
    set -l response (env %(complete_var)s=fish_complete COMP_WORDS=(commandline -cp) \
COMP_CWORD=(commandline -t) %(prog_name)s);

    for completion in $response;
        set -l metadata (string split "," $completion);

        if test $metadata[1] = "dir";
            __fish_complete_directories $metadata[2];
        else if test $metadata[1] = "file";
            __fish_complete_path $metadata[2];
        else if test $metadata[1] = "plain";
            echo $metadata[2];
        end;
    end;
end;

complete --no-files --command %(prog_name)s --arguments \
"(%(complete_func)s)";
"""


class ShellComplete:
    """Base class for providing shell completion support. A subclass for
    a given shell will override attributes and methods to implement the
    completion instructions (``source`` and ``complete``).

    :param cli: Command being called.
    :param prog_name: Name of the executable in the shell.
    :param complete_var: Name of the environment variable that holds
        the completion instruction.

    .. versionadded:: 8.0
    """

    name: t.ClassVar[str]
    """Name to register the shell as with :func:`add_completion_class`.
    This is used in completion instructions (``{name}_source`` and
    ``{name}_complete``).
    """

    source_template: t.ClassVar[str]
    """Completion script template formatted by :meth:`source`. This must
    be provided by subclasses.
    """

    def __init__(
        self,
        cli: Command,
        ctx_args: cabc.MutableMapping[str, t.Any],
        prog_name: str,
        complete_var: str,
    ) -> None:
        self.cli = cli
        self.ctx_args = ctx_args
        self.prog_name = prog_name
        self.complete_var = complete_var

    @property
    def func_name(self) -> str:
        """The name of the shell function defined by the completion
        script.
        """
        safe_name = re.sub(r"\W*", "", self.prog_name.replace("-", "_"), flags=re.ASCII)
        return f"_{safe_name}_completion"

    def source_vars(self) -> dict[str, t.Any]:
        """Vars for formatting :attr:`source_template`.

        By default this provides ``complete_func``, ``complete_var``,
        and ``prog_name``.
        """
        return {
            "complete_func": self.func_name,
            "complete_var": self.complete_var,
            "prog_name": self.prog_name,
        }

    def source(self) -> str:
        """Produce the shell script that defines the completion
        function. By default this ``%``-style formats
        :attr:`source_template` with the dict returned by
        :meth:`source_vars`.
        """
        return self.source_template % self.source_vars()

    def get_completion_args(self) -> tuple[list[str], str]:
        """Use the env vars defined by the shell script to return a
        tuple of ``args, incomplete``. This must be implemented by
        subclasses.
        """
        raise NotImplementedError

    def get_completions(self, args: list[str], incomplete: str) -> list[CompletionItem]:
        """Determine the context and last complete command or parameter
        from the complete args. Call that object's ``shell_complete``
        method to get the completions for the incomplete value.

        :param args: List of complete args before the incomplete value.
        :param incomplete: Value being completed. May be empty.
        """
        ctx = _resolve_context(self.cli, self.ctx_args, self.prog_name, args)
        obj, incomplete = _resolve_incomplete(ctx, args, incomplete)
        return obj.shell_complete(ctx, incomplete)

    def format_completion(self, item: CompletionItem) -> str:
        """Format a completion item into the form recognized by the
        shell script. This must be implemented by subclasses.

        :param item: Completion item to format.
        """
        raise NotImplementedError

    def complete(self) -> str:
        """Produce the completion data to send back to the shell.

        By default this calls :meth:`get_completion_args`, gets the
        completions, then calls :meth:`format_completion` for each
        completion.
        """
        args, incomplete = self.get_completion_args()
        completions = self.get_completions(args, incomplete)
        out = [self.format_completion(item) for item in completions]
        return "\n".join(out)


class BashComplete(ShellComplete):
    """Shell completion for Bash."""

    name = "bash"
    source_template = _SOURCE_BASH

    @staticmethod
    def _check_version() -> None:
        import shutil
        import subprocess

        bash_exe = shutil.which("bash")

        if bash_exe is None:
            match = None
        else:
            output = subprocess.run(
                [bash_exe, "--norc", "-c", 'echo "${BASH_VERSION}"'],
                stdout=subprocess.PIPE,
            )
            match = re.search(r"^(\d+)\.(\d+)\.\d+", output.stdout.decode())

        if match is not None:
            major, minor = match.groups()

            if major < "4" or major == "4" and minor < "4":
                echo(
                    _(
                        "Shell completion is not supported for Bash"
                        " versions older than 4.4."
                    ),
                    err=True,
                )
        else:
            echo(
                _("Couldn't detect Bash version, shell completion is not supported."),
                err=True,
            )

    def source(self) -> str:
        self._check_version()
        return super().source()

    def get_completion_args(self) -> tuple[list[str], str]:
        cwords = split_arg_string(os.environ["COMP_WORDS"])
        cword = int(os.environ["COMP_CWORD"])
        args = cwords[1:cword]

        try:
            incomplete = cwords[cword]
        except IndexError:
            incomplete = ""

        return args, incomplete

    def format_completion(self, item: CompletionItem) -> str:
        return f"{item.type},{item.value}"


class ZshComplete(ShellComplete):
    """Shell completion for Zsh."""

    name = "zsh"
    source_template = _SOURCE_ZSH

    def get_completion_args(self) -> tuple[list[str], str]:
        cwords = split_arg_string(os.environ["COMP_WORDS"])
        cword = int(os.environ["COMP_CWORD"])
        args = cwords[1:cword]

        try:
            incomplete = cwords[cword]
        except IndexError:
            incomplete = ""

        return args, incomplete

    def format_completion(self, item: CompletionItem) -> str:
        return f"{item.type}\n{item.value}\n{item.help if item.help else '_'}"


class FishComplete(ShellComplete):
    """Shell completion for Fish."""

    name = "fish"
    source_template = _SOURCE_FISH

    def get_completion_args(self) -> tuple[list[str], str]:
        cwords = split_arg_string(os.environ["COMP_WORDS"])
        incomplete = os.environ["COMP_CWORD"]
        args = cwords[1:]

        # Fish stores the partial word in both COMP_WORDS and
        # COMP_CWORD, remove it from complete args.
        if incomplete and args and args[-1] == incomplete:
            args.pop()

        return args, incomplete

    def format_completion(self, item: CompletionItem) -> str:
        if item.help:
            return f"{item.type},{item.value}\t{item.help}"

        return f"{item.type},{item.value}"


ShellCompleteType = t.TypeVar("ShellCompleteType", bound="type[ShellComplete]")


_available_shells: dict[str, type[ShellComplete]] = {
    "bash": BashComplete,
    "fish": FishComplete,
    "zsh": ZshComplete,
}


def add_completion_class(
    cls: ShellCompleteType, name: str | None = None
) -> ShellCompleteType:
    """Register a :class:`ShellComplete` subclass under the given name.
    The name will be provided by the completion instruction environment
    variable during completion.

    :param cls: The completion class that will handle completion for the
        shell.
    :param name: Name to register the class under. Defaults to the
        class's ``name`` attribute.
    """
    if name is None:
        name = cls.name

    _available_shells[name] = cls

    return cls


def get_completion_class(shell: str) -> type[ShellComplete] | None:
    """Look up a registered :class:`ShellComplete` subclass by the name
    provided by the completion instruction environment variable. If the
    name isn't registered, returns ``None``.

    :param shell: Name the class is registered under.
    """
    return _available_shells.get(shell)


def split_arg_string(string: str) -> list[str]:
    """Split an argument string as with :func:`shlex.split`, but don't
    fail if the string is incomplete. Ignores a missing closing quote or
    incomplete escape sequence and uses the partial token as-is.

    .. code-block:: python

        split_arg_string("example 'my file")
        ["example", "my file"]

        split_arg_string("example my\\")
        ["example", "my"]

    :param string: String to split.

    .. versionchanged:: 8.2
        Moved to ``shell_completion`` from ``parser``.
    """
    import shlex

    lex = shlex.shlex(string, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    out = []

    try:
        for token in lex:
            out.append(token)
    except ValueError:
        # Raised when end-of-string is reached in an invalid state. Use
        # the partial token as-is. The quote or escape character is in
        # lex.state, not lex.token.
        out.append(lex.token)

    return out


def _is_incomplete_argument(ctx: Context, param: Parameter) -> bool:
    """Determine if the given parameter is an argument that can still
    accept values.

    :param ctx: Invocation context for the command represented by the
        parsed complete args.
    :param param: Argument object being checked.
    """
    if not isinstance(param, Argument):
        return False

    assert param.name is not None
    # Will be None if expose_value is False.
    value = ctx.params.get(param.name)
    return (
        param.nargs == -1
        or ctx.get_parameter_source(param.name) is not ParameterSource.COMMANDLINE
        or (
            param.nargs > 1
            and isinstance(value, (tuple, list))
            and len(value) < param.nargs
        )
    )


def _start_of_option(ctx: Context, value: str) -> bool:
    """Check if the value looks like the start of an option."""
    if not value:
        return False

    c = value[0]
    return c in ctx._opt_prefixes


def _is_incomplete_option(ctx: Context, args: list[str], param: Parameter) -> bool:
    """Determine if the given parameter is an option that needs a value.

    :param args: List of complete args before the incomplete value.
    :param param: Option object being checked.
    """
    if not isinstance(param, Option):
        return False

    if param.is_flag or param.count:
        return False

    last_option = None

    for index, arg in enumerate(reversed(args)):
        if index + 1 > param.nargs:
            break

        if _start_of_option(ctx, arg):
            last_option = arg

    return last_option is not None and last_option in param.opts


def _resolve_context(
    cli: Command,
    ctx_args: cabc.MutableMapping[str, t.Any],
    prog_name: str,
    args: list[str],
) -> Context:
    """Produce the context hierarchy starting with the command and
    traversing the complete arguments. This only follows the commands,
    it doesn't trigger input prompts or callbacks.

    :param cli: Command being called.
    :param prog_name: Name of the executable in the shell.
    :param args: List of complete args before the incomplete value.
    """
    ctx_args["resilient_parsing"] = True
    with cli.make_context(prog_name, args.copy(), **ctx_args) as ctx:
        args = ctx._protected_args + ctx.args

        while args:
            command = ctx.command

            if isinstance(command, Group):
                if not command.chain:
                    name, cmd, args = command.resolve_command(ctx, args)

                    if cmd is None:
                        return ctx

                    with cmd.make_context(
                        name, args, parent=ctx, resilient_parsing=True
                    ) as sub_ctx:
                        ctx = sub_ctx
                        args = ctx._protected_args + ctx.args
                else:
                    sub_ctx = ctx

                    while args:
                        name, cmd, args = command.resolve_command(ctx, args)

                        if cmd is None:
                            return ctx

                        with cmd.make_context(
                            name,
                            args,
                            parent=ctx,
                            allow_extra_args=True,
                            allow_interspersed_args=False,
                            resilient_parsing=True,
                        ) as sub_sub_ctx:
                            sub_ctx = sub_sub_ctx
                            args = sub_ctx.args

                    ctx = sub_ctx
                    args = [*sub_ctx._protected_args, *sub_ctx.args]
            else:
                break

    return ctx


def _resolve_incomplete(
    ctx: Context, args: list[str], incomplete: str
) -> tuple[Command | Parameter, str]:
    """Find the Click object that will handle the completion of the
    incomplete value. Return the object and the incomplete value.

    :param ctx: Invocation context for the command represented by
        the parsed complete args.
    :param args: List of complete args before the incomplete value.
    :param incomplete: Value being completed. May be empty.
    """
    # Different shells treat an "=" between a long option name and
    # value differently. Might keep the value joined, return the "="
    # as a separate item, or return the split name and value. Always
    # split and discard the "=" to make completion easier.
    if incomplete == "=":
        incomplete = ""
    elif "=" in incomplete and _start_of_option(ctx, incomplete):
        name, _, incomplete = incomplete.partition("=")
        args.append(name)

    # The "--" marker tells Click to stop treating values as options
    # even if they start with the option character. If it hasn't been
    # given and the incomplete arg looks like an option, the current
    # command will provide option name completions.
    if "--" not in args and _start_of_option(ctx, incomplete):
        return ctx.command, incomplete

    params = ctx.command.get_params(ctx)

    # If the last complete arg is an option name with an incomplete
    # value, the option will provide value completions.
    for param in params:
        if _is_incomplete_option(ctx, args, param):
            return param, incomplete

    # It's not an option name or value. The first argument without a
    # parsed value will provide value completions.
    for param in params:
        if _is_incomplete_argument(ctx, param):
            return param, incomplete

    # There were no unparsed arguments, the command may be a group that
    # will provide command name completions.
    return ctx.command, incomplete

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_bart_attention.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import numpy as np
from fusion_attention import AttentionMask, FusionAttention
from onnx import TensorProto, helper
from onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class FusionBartAttention(FusionAttention):
    """
    Fuse Bart Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        super().__init__(model, hidden_size, num_heads, attention_mask)

    def check_runtime_shape_path(
        self,
        reshape_qkv_2,
        reshape_qkv_1,
        reshape_q_2,
        reshape_k_2,
        reshape_v_2,
        root_input,
    ):
        concat_qkv_2_path = self.model.match_parent_path(reshape_qkv_2, ["Concat"], [1])
        if concat_qkv_2_path is None:
            return False
        concat_qkv_2 = concat_qkv_2_path[0]

        reshape_qkv_2_path_1 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_2_path_2 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_qkv_2_path_1 is None or reshape_qkv_2_path_2 is None:
            return False

        _, gather_1, shape_1 = reshape_qkv_2_path_1
        _, gather_2, shape_2 = reshape_qkv_2_path_2

        if shape_1.input[0] != root_input or shape_2.input[0] != root_input:
            return False

        reshape_qkv_1_path_1 = self.model.match_parent_path(reshape_qkv_1, ["Concat", "Unsqueeze", "Gather"], [1, 0, 0])
        reshape_qkv_1_path_2 = self.model.match_parent_path(reshape_qkv_1, ["Concat", "Unsqueeze", "Gather"], [1, 2, 0])
        if reshape_qkv_1_path_1 is None or reshape_qkv_1_path_2 is None:
            return False
        if reshape_qkv_1_path_1[-1].name != gather_1.name or reshape_qkv_1_path_2[-1].name != gather_2.name:
            return False

        reshape_q_2_path = self.model.match_parent_path(reshape_q_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        reshape_k_2_path = self.model.match_parent_path(reshape_k_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        reshape_v_2_path = self.model.match_parent_path(reshape_v_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        if reshape_q_2_path is None or reshape_k_2_path is None or reshape_v_2_path is None:
            return False

        mul_q = reshape_q_2_path[-1]
        mul_k = reshape_k_2_path[-1]
        mul_v = reshape_v_2_path[-1]

        gather_1_out = gather_1.output[0]
        if mul_q.input[0] != gather_1_out or mul_k.input[0] != gather_1_out or mul_v.input[0] != gather_1_out:
            return False

        return True

    def check_runtime_shape_path_openai(
        self,
        reshape_qkv_2,
        matmul_qkv,
        add_qk,
        matmul_qk,
        add_q,
    ):
        reshape_qkv_path = self.model.match_parent_path(
            reshape_qkv_2, ["Concat", "Slice", "Shape", "Transpose"], [1, 0, 0, 0]
        )
        if reshape_qkv_path is None or reshape_qkv_path[-1].input[0] != matmul_qkv.output[0]:
            return False

        matmul_qk_path_1 = self.model.match_parent_path(
            matmul_qk, ["Mul", "Pow", "Cast", "Div", "Gather", "Shape"], [0, 1, 0, 0, 0, 0]
        )
        matmul_qk_path_2 = self.model.match_parent_path(
            matmul_qk, ["Mul", "Pow", "Cast", "Div", "Gather", "Shape"], [1, 1, 0, 0, 0, 0]
        )
        if matmul_qk_path_1 is None or matmul_qk_path_2 is None:
            return False

        mul_1 = matmul_qk_path_1[0]
        mul_2 = matmul_qk_path_2[0]
        if mul_1.input[1] != mul_2.input[1]:
            return False
        if matmul_qk_path_1[-1].input[0] != add_q.output[0] and matmul_qk_path_2[-1].input[0] != add_q.output[0]:
            return False

        # For decoder attentions only
        if add_qk is not None:
            add_qk_path = self.model.match_parent_path(add_qk, ["Slice"], [1])
            if add_qk_path is None:
                return False
            slice_q_path_1 = self.model.match_parent_path(
                add_qk_path[0], ["Slice", "Unsqueeze", "Gather", "Shape"], [0, 2, 0, 0]
            )
            slice_q_path_2 = self.model.match_parent_path(add_qk_path[0], ["Unsqueeze", "Gather", "Shape"], [2, 0, 0])
            if slice_q_path_1 is None and slice_q_path_2 is None:
                return False
            _, unsqueeze_1, _, _ = slice_q_path_1
            unsqueeze_2, _, _ = slice_q_path_2
            if unsqueeze_1.input[0] != unsqueeze_2.input[0]:
                return False
            if slice_q_path_1[-1].input[0] != add_q.output[0] and slice_q_path_2[-1].input[0] != add_q.output[0]:
                return False

        return True

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Track if fusion is occurring for OpenAI implementation of Whisper
        model_impl_openai = False

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, 1, 0, 0, 0, 0],
        )
        qkv_nodes_openai = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 1, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (
                add_out,
                matmul_out,
                reshape_qkv_2,
                transpose_qkv,
                reshape_qkv_1,
                matmul_qkv,
            ) = qkv_nodes
        elif qkv_nodes_openai is not None:
            qkv_nodes = qkv_nodes_openai
            (
                add_out,
                matmul_out,
                reshape_qkv_2,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes
            # Set model implementation to openai
            model_impl_openai = True
        else:
            return

        other_inputs = []
        for input in normalize_node.input:
            if input not in output_name_to_node:
                continue
            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return
        root_input = other_inputs[0]

        # Sometimes the input name to the attention MatMul nodes does not match the input name to the end
        # SkipLayerNormalization node (name saved in root_input). We find the true input name to the MatMul
        # nodes by getting the initial SkipLayerNormalization node and checking how many MatMul nodes are
        # children nodes for each of its output names.
        """
                                        root_input
                    +---------------------------------------------------+
                    |                                                   |
                    |                                                   |
        SkipLayerNormalization --> Attention --> MatMul --> SkipLayerNormalization
        """
        skip_layernorm = output_name_to_node[root_input]
        # For some attention blocks, the end SkipLayerNormalization node may point to an Add node whose
        # child is the LayerNormalization node.
        if skip_layernorm.op_type == "Add":
            skip_layernorm = self.model.get_children(skip_layernorm)[0]
        for output in skip_layernorm.output:
            if not output:
                continue
            children = input_name_to_nodes[output]
            children_types = [child.op_type for child in children]
            if children_types.count("MatMul") >= 1:
                root_input = output
                break

        graph_input_names = {node.name for node in self.model.graph().input}
        graph_output_names = {node.name for node in self.model.graph().output}

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Reshape", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, 0, None],
        )
        v_nodes_openai = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, None],
        )
        v_nodes_with_past_self_attn = self.model.match_parent_path(
            # Decoder attention with past value concatenated before MatMul
            matmul_qkv,
            ["Reshape", "Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 1, 0, 0, None],
        )
        v_nodes_with_past_cross_attn = self.model.match_parent_path(
            # Decoder attention with past value directly used in MatMul
            matmul_qkv,
            ["Reshape"],
            [1],
        )
        v_nodes_with_past_cross_attn_openai = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "Reshape", "Transpose"],
            [1, 0, 0, 0],
        )
        past_v, present_v = "", ""
        reshape_v_2, add_v = None, None
        if v_nodes is not None:
            (reshape_v_2, transpose_v, reshape_v_1, add_v, matmul_v) = v_nodes
            # For initial pass through encoder-decoder_with_past to get starting past values (beam search)
            present_v = transpose_v.output[0]
        elif v_nodes_openai is not None:
            v_nodes = v_nodes_openai
            (transpose_v, reshape_v_1, add_v, matmul_v) = v_nodes
            # For initial pass through encoder-decoder_with_past to get starting past values (beam search)

            # Find the child path to access the correct present_v values
            # Openai impl provides present/past v values in 3D format
            # whereas ort MultiHeadAttention expects v values in 4D, hence the
            # additional Reshape and Transpose nodes are added
            # For encoder attention types
            # Add -> Reshape -> Transpose -> Present_V
            reshape_path = self.model.match_child_path(
                add_v,
                ["Reshape", "Transpose"],
                exclude=[reshape_v_1],
            )
            # For decoder attention types
            # add_v_node                     Reshape <- Transpose <-Past_V
            #           \                  /
            #             \              /
            #               -> Concat <-
            #                    |
            #                    |--> Reshape -> Transpose -> Present_V
            concat_path = self.model.match_child_path(add_v, ["Concat", "Reshape", "Transpose"])
            if reshape_path is not None:
                (_, transpose_add_v) = reshape_path
                if transpose_add_v.output[0] in graph_output_names:
                    present_v = transpose_add_v.output[0]
            if concat_path is not None:
                (concat_v, _, transpose_concat_v) = concat_path
                if transpose_concat_v.output[0] in graph_output_names:
                    present_v = transpose_concat_v.output[0]
                concat_nodes = self.model.match_parent_path(concat_v, ["Reshape", "Transpose"], [0, 0])
                _, transpose_concat_v_in = concat_nodes
                past_v = transpose_concat_v_in.input[0]
        elif v_nodes_with_past_self_attn is not None:
            (reshape_v_2, concat_v, transpose_v, reshape_v_1, add_v, matmul_v) = v_nodes_with_past_self_attn
            v_nodes = v_nodes_with_past_self_attn
            past_v = concat_v.input[0]
            present_v = concat_v.output[0]
        elif (
            v_nodes_with_past_cross_attn is not None and v_nodes_with_past_cross_attn[-1].input[0] in graph_input_names
        ):
            v_nodes = v_nodes_with_past_cross_attn
            past_v = v_nodes[-1].input[0]
            present_v = v_nodes[-1].output[0]
            if present_v not in graph_output_names:
                identity_node_v = list(
                    filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_v])
                )
                present_v = identity_node_v[0].output[0] if len(identity_node_v) == 1 else ""
        elif (
            v_nodes_with_past_cross_attn_openai is not None
            and v_nodes_with_past_cross_attn_openai[-1].input[0] in graph_input_names
        ):
            v_nodes = v_nodes_with_past_cross_attn_openai
            past_v = v_nodes[-1].input[0]
            present_v = v_nodes[-1].output[0]
            if present_v not in graph_output_names:
                identity_node_v = list(
                    filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_v])
                )
                present_v = identity_node_v[0].output[0] if len(identity_node_v) == 1 else ""
        else:
            logger.debug("fuse_attention: failed to match v path")
            return
        past_v = past_v if past_v in graph_input_names else ""
        present_v = present_v if present_v in graph_output_names else ""

        qk_nodes_1 = self.model.match_parent_path(matmul_qkv, ["Softmax", "MatMul"], [0, 0])
        qk_nodes_2 = self.model.match_parent_path(
            matmul_qkv, ["Softmax", "Reshape", "Add", "Reshape", "MatMul"], [0, 0, 0, 0, 0]
        )
        qk_nodes_2_openai = self.model.match_parent_path(matmul_qkv, ["Softmax", "Add", "MatMul"], [0, 0, 0])
        add_qk = None
        if qk_nodes_1 is not None:
            _, matmul_qk = qk_nodes_1
            qk_nodes = qk_nodes_1
        elif qk_nodes_2 is not None:
            _, _, add_qk, _, matmul_qk = qk_nodes_2
            qk_nodes = qk_nodes_2
        elif qk_nodes_2_openai is not None:
            _, add_qk, matmul_qk = qk_nodes_2_openai
            qk_nodes = qk_nodes_2_openai
        else:
            return

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "Reshape", "Mul", "Add", "MatMul"],
            [0, 0, 0, 0, 0, 1],
        )
        q_nodes_openai = self.model.match_parent_path(
            matmul_qk,
            ["Mul", "Transpose", "Reshape", "Add", "MatMul"],
            [0, 0, 0, 0, 1],
        )
        reshape_q_2 = None
        if q_nodes is not None:
            reshape_q_2, transpose_q, reshape_q_1, mul_q, add_q, matmul_q = q_nodes
        elif q_nodes_openai is not None:
            q_nodes = q_nodes_openai
            mul_q, transpose_q, reshape_q_1, add_q, matmul_q = q_nodes
        else:
            return

        k_nodes_with_bias = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, 0, 0, 1],
        )
        k_nodes_no_bias_openai = self.model.match_parent_path(
            matmul_qk,
            ["Mul", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 0],
        )
        k_nodes_no_bias = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 0, 0],
        )
        k_nodes_no_bias_with_past_self_attn = self.model.match_parent_path(
            # Decoder attention with past key concatenated before MatMul
            matmul_qk,
            ["Transpose", "Reshape", "Concat", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 1, 0, 0],
        )
        k_nodes_no_bias_with_past_cross_attn = self.model.match_parent_path(
            # Decoder attention with past key directly used in MatMul
            matmul_qk,
            ["Transpose", "Reshape"],
            [1, 0],
        )
        k_nodes_no_bias_with_past_cross_attn_openai = self.model.match_parent_path(
            # Decoder attention with past key directly used in MatMul
            matmul_qk,
            ["Mul", "Transpose", "Reshape", "Reshape", "Transpose"],
            [1, 0, 0, 0, 0],
        )
        past_k, present_k = "", ""
        reshape_k_2, reshape_k_1, matmul_k = None, None, None
        if k_nodes_with_bias is not None:
            _, reshape_k_2, transpose_k_1, reshape_k_1, add_k, matmul_k = k_nodes_with_bias
            k_nodes = k_nodes_with_bias
        elif k_nodes_no_bias_openai is not None:
            mul_k, transpose_k_1, reshape_k_1, matmul_k = k_nodes_no_bias_openai
            k_nodes = k_nodes_no_bias_openai
            present_k = matmul_k.output[0]

            # Find the child path to access the correct present_k values
            # Openai impl provides present/past k values in 3D format
            # whereas ort MultiHeadAttention expects k values in 4D, hence the
            # additional Reshape and Transpose nodes are added
            # For encoder attention types
            # Matmul -> Reshape -> Transpose -> Present_K
            reshape_path = self.model.match_child_path(
                matmul_k,
                ["Reshape", "Transpose"],
                exclude=[reshape_k_1],
            )
            # For decoder attention types
            # matmul_k_node                  Reshape <- Transpose <- Past_K
            #           \                  /
            #             \              /
            #               -> Concat <-
            #                    |
            #                    +--> Reshape -> Transpose -> Present_K
            concat_path = self.model.match_child_path(matmul_k, ["Concat", "Reshape", "Transpose"])
            if reshape_path is not None:
                (_, transpose_matmul_k) = reshape_path
                if transpose_matmul_k.output[0] in graph_output_names:
                    present_k = transpose_matmul_k.output[0]
            if concat_path is not None:
                (concat_k, _, transpose_concat_k) = concat_path
                if transpose_concat_k.output[0] in graph_output_names:
                    present_k = transpose_concat_k.output[0]
                concat_nodes = self.model.match_parent_path(concat_k, ["Reshape", "Transpose"], [0, 0])
                _, transpose_concat_k_in = concat_nodes
                past_k = transpose_concat_k_in.input[0]
        elif k_nodes_no_bias is not None:
            _, reshape_k_2, transpose_k_1, reshape_k_1, matmul_k = k_nodes_no_bias
            k_nodes = k_nodes_no_bias
            # For initial pass through encoder-decoder_with_past to get starting past values (beam search)
            present_k = transpose_k_1.output[0]
        elif k_nodes_no_bias_with_past_self_attn is not None:
            _, reshape_k_2, concat_k, _, reshape_k_1, matmul_k = k_nodes_no_bias_with_past_self_attn
            k_nodes = k_nodes_no_bias_with_past_self_attn
            past_k = concat_k.input[0]
            present_k = concat_k.output[0]
        elif (
            k_nodes_no_bias_with_past_cross_attn is not None
            and k_nodes_no_bias_with_past_cross_attn[-1].input[0] in graph_input_names
        ):
            k_nodes = k_nodes_no_bias_with_past_cross_attn
            past_k = k_nodes[-1].input[0]
            present_k = k_nodes[-1].output[0]
            if present_k not in graph_output_names:
                identity_node_k = list(
                    filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_k])
                )
                present_k = identity_node_k[0].output[0] if len(identity_node_k) == 1 else ""
        elif (
            k_nodes_no_bias_with_past_cross_attn_openai is not None
            and k_nodes_no_bias_with_past_cross_attn_openai[-1].input[0] in graph_input_names
        ):
            k_nodes = k_nodes_no_bias_with_past_cross_attn_openai
            past_k = k_nodes[-1].input[0]
            present_k = k_nodes[-1].output[0]
            if present_k not in graph_output_names:
                identity_node_k = list(
                    filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_k])
                )
                present_k = identity_node_k[0].output[0] if len(identity_node_k) == 1 else ""
        else:
            return
        past_k = past_k if past_k in graph_input_names else ""
        present_k = present_k if present_k in graph_output_names else ""

        if k_nodes in (k_nodes_no_bias_openai, k_nodes_no_bias, k_nodes_no_bias_with_past_self_attn):
            # Create empty Add node for attention graph
            bias_dim = self.model.get_initializer(add_v.input[0]).dims[0]
            empty_bias_name = "empty_bias"
            empty_tensor = self.model.get_initializer(empty_bias_name)
            if empty_tensor is None:
                self.add_initializer(
                    empty_bias_name,
                    TensorProto.FLOAT,
                    dims=[bias_dim],
                    vals=np.array([0.0] * bias_dim, dtype=np.float32),
                )

            add_name = self.model.create_node_name("Add")
            add_k = helper.make_node("Add", [empty_bias_name, matmul_k.output[0]], [reshape_k_1.name], add_name)

        if (
            model_impl_openai
            and not bool(past_k)
            and not self.check_runtime_shape_path_openai(
                reshape_qkv_2,
                matmul_qkv,
                add_qk,
                matmul_qk,
                add_q,
            )
        ):
            return
        elif (
            not model_impl_openai
            and not bool(past_k)
            and not self.check_runtime_shape_path(
                reshape_qkv_2,
                reshape_qkv_1,
                reshape_q_2,
                reshape_k_2,
                reshape_v_2,
                root_input,
            )
        ):
            return

        three_root_inputs = bool(past_k) and bool(past_v) and matmul_k is None and "matmul_v" not in locals()
        one_root_input = (
            not three_root_inputs
            and matmul_k.input[0] == root_input
            and matmul_q.input[0] == root_input
            and matmul_v.input[0] == root_input
        )
        two_root_inputs = (
            not three_root_inputs
            and matmul_q.input[0] == root_input
            and matmul_k.input[0] == matmul_v.input[0]
            and matmul_k.input[0] != matmul_q.input[0]
        )

        # There are 5 types of attention:
        # 1) Encoder attention with one_root_input=True and qk_nodes=qk_nodes_1
        # 2) Decoder attention with one_root_input=True and qk_nodes=qk_nodes_2
        # 3) Decoder attention with past with one_root_input=True and qk_nodes=qk_nodes_1 and past_k=past_decoder_key and past_v=past_decoder_value
        # 4) Decoder cross attention with two_root_inputs=True and qk_nodes=qk_nodes_1
        # 5) Decoder cross attention with past with three_root_inputs=True and qk_nodes=qk_nodes_1
        encoder_attention = one_root_input and qk_nodes == qk_nodes_1
        decoder_attention = one_root_input and qk_nodes in (qk_nodes_2, qk_nodes_2_openai)
        decoder_attention_with_past = (
            (encoder_attention if not model_impl_openai else decoder_attention) and bool(past_k) and bool(past_v)
        )
        decoder_cross_attention = two_root_inputs and qk_nodes == qk_nodes_1
        decoder_cross_attention_with_past = three_root_inputs and qk_nodes == qk_nodes_1

        # For decoder_attention, the attention mask needs to be included in the attention node
        mask_index, mask_nodes = None, []
        if decoder_attention:
            mask_nodes_bart = self.model.match_parent_path(
                add_qk,
                ["Where"],
                [1],
            )
            mask_nodes_whisper = self.model.match_parent_path(
                add_qk,
                ["Expand", "Unsqueeze", "Unsqueeze", "Where"],
                [1, 0, 0, 0],
            )
            if mask_nodes_whisper is not None:
                mask_index = mask_nodes_whisper[0].output[-1]
                mask_nodes = mask_nodes_whisper
            elif mask_nodes_bart is not None:
                mask_index = mask_nodes_bart[0].output[-1]
                mask_nodes = mask_nodes_bart

        if (
            encoder_attention
            or decoder_attention
            or decoder_attention_with_past
            or decoder_cross_attention
            or decoder_cross_attention_with_past
        ):
            attention_last_node = reshape_qkv_2
            num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q_1)

            if num_heads <= 0 or hidden_size <= 0 or (hidden_size % num_heads) != 0:
                logger.debug("fuse_attention: failed to detect num_heads or hidden_size")
                return

            new_node = None
            if decoder_attention_with_past or decoder_cross_attention or decoder_cross_attention_with_past:
                # Note: Decoder attention with past key and past value is fused as multihead attention
                # rather than attention because multihead attention supports separate past key and past
                # value whereas attention supports concatenated past key and past value.
                new_node = (
                    self.create_multihead_attention_node(
                        q_matmul=matmul_q,
                        k_matmul=matmul_k if decoder_cross_attention or decoder_attention_with_past else past_k,
                        v_matmul=matmul_v if decoder_cross_attention or decoder_attention_with_past else past_v,
                        q_add=add_q,
                        k_add=add_k if decoder_cross_attention or decoder_attention_with_past else None,
                        v_add=add_v if decoder_cross_attention or decoder_attention_with_past else None,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        output=attention_last_node.output[0],
                        unidirectional=decoder_attention_with_past,
                        past_k=past_k if decoder_attention_with_past else "",
                        past_v=past_v if decoder_attention_with_past else "",
                        present_k=present_k,
                        present_v=present_v,
                        packed_qkv=decoder_attention_with_past,
                    )
                    if self.use_multi_head_attention
                    else None
                )
            else:
                # Temporarily set multihead attention flag to false
                use_multi_head_attention_ground_truth = self.use_multi_head_attention
                self.use_multi_head_attention = False
                add_qk_str = mask_index if decoder_attention and mask_index else ""
                new_node = self.create_attention_node(
                    mask_index=None,
                    q_matmul=matmul_q,
                    k_matmul=matmul_k,
                    v_matmul=matmul_v,
                    q_add=add_q,
                    k_add=add_k,
                    v_add=add_v,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    first_input=root_input,
                    output=attention_last_node.output[0],
                    add_qk_str=(
                        None if len(mask_nodes) > 1 else add_qk_str
                    ),  # deprecate and use is_unidirectional attr instead for Whisper
                    past_k=past_k,
                    past_v=past_v,
                    present_k=present_k,
                    present_v=present_v,
                    causal=decoder_attention,
                )
                self.use_multi_head_attention = use_multi_head_attention_ground_truth
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)

            # When using multihead attention, keep MatMul nodes in original graph
            if decoder_attention_with_past or decoder_cross_attention or decoder_cross_attention_with_past:
                if q_nodes[-1].op_type == "MatMul":
                    q_nodes.pop()
                if k_nodes[-1].op_type == "MatMul":
                    k_nodes.pop()
                if v_nodes[-1].op_type == "MatMul":
                    v_nodes.pop()
                if self.disable_multi_head_attention_bias and (
                    decoder_cross_attention or decoder_cross_attention_with_past
                ):
                    if q_nodes[-1].op_type == "Add":
                        q_nodes.pop()
                    if k_nodes[-1].op_type == "Add":
                        k_nodes.pop()
                    if v_nodes[-1].op_type == "Add":
                        v_nodes.pop()

            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\accessors.py ===
"""
datetimelike delegation
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import ABCSeries

from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.base import (
    NoNewAttributesMixin,
    PandasObject,
)
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Series,
    )


class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _hidden_attrs = PandasObject._hidden_attrs | {
        "orig",
        "name",
    }

    def __init__(self, data: Series, orig) -> None:
        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        self._parent = data
        self.orig = orig
        self.name = getattr(data, "name", None)
        self._freeze()

    def _get_values(self):
        data = self._parent
        if lib.is_np_dtype(data.dtype, "M"):
            return DatetimeIndex(data, copy=False, name=self.name)

        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeIndex(data, copy=False, name=self.name)

        elif lib.is_np_dtype(data.dtype, "m"):
            return TimedeltaIndex(data, copy=False, name=self.name)

        elif isinstance(data.dtype, PeriodDtype):
            return PeriodArray(data, copy=False)

        raise TypeError(
            f"cannot convert an object of type {type(data)} to a datetimelike index"
        )

    def _delegate_property_get(self, name: str):
        from pandas import Series

        values = self._get_values()

        result = getattr(values, name)

        # maybe need to upcast (ints)
        if isinstance(result, np.ndarray):
            if is_integer_dtype(result):
                result = result.astype("int64")
        elif not is_list_like(result):
            return result

        result = np.asarray(result)

        if self.orig is not None:
            index = self.orig.index
        else:
            index = self._parent.index
        # return the result as a Series
        result = Series(result, index=index, name=self.name).__finalize__(self._parent)

        # setting this object will show a SettingWithCopyWarning/Error
        result._is_copy = (
            "modifications to a property of a datetimelike "
            "object are not supported and are discarded. "
            "Change values on the original."
        )

        return result

    def _delegate_property_set(self, name: str, value, *args, **kwargs):
        raise ValueError(
            "modifications to a property of a datetimelike object are not supported. "
            "Change values on the original."
        )

    def _delegate_method(self, name: str, *args, **kwargs):
        from pandas import Series

        values = self._get_values()

        method = getattr(values, name)
        result = method(*args, **kwargs)

        if not is_list_like(result):
            return result

        result = Series(result, index=self._parent.index, name=self.name).__finalize__(
            self._parent
        )

        # setting this object will show a SettingWithCopyWarning/Error
        result._is_copy = (
            "modifications to a method of a datetimelike "
            "object are not supported and are discarded. "
            "Change values on the original."
        )

        return result


@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=TimedeltaArray._datetimelike_ops,
    typ="property",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=TimedeltaArray._datetimelike_methods,
    typ="method",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=DatetimeArray._datetimelike_ops,
    typ="property",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=DatetimeArray._datetimelike_methods,
    typ="method",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    def __init__(self, data: Series, orig) -> None:
        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        self._parent = data
        self._orig = orig
        self._freeze()

    def _delegate_property_get(self, name: str):
        if not hasattr(self._parent.array, f"_dt_{name}"):
            raise NotImplementedError(
                f"dt.{name} is not supported for {self._parent.dtype}"
            )
        result = getattr(self._parent.array, f"_dt_{name}")

        if not is_list_like(result):
            return result

        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        # return the result as a Series, which is by definition a copy
        result = type(self._parent)(
            result, index=index, name=self._parent.name
        ).__finalize__(self._parent)

        return result

    def _delegate_method(self, name: str, *args, **kwargs):
        if not hasattr(self._parent.array, f"_dt_{name}"):
            raise NotImplementedError(
                f"dt.{name} is not supported for {self._parent.dtype}"
            )

        result = getattr(self._parent.array, f"_dt_{name}")(*args, **kwargs)

        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        # return the result as a Series, which is by definition a copy
        result = type(self._parent)(
            result, index=index, name=self._parent.name
        ).__finalize__(self._parent)

        return result

    def to_pytimedelta(self):
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pytimedelta()

    def to_pydatetime(self):
        # GH#20306
        warnings.warn(
            f"The behavior of {type(self).__name__}.to_pydatetime is deprecated, "
            "in a future version this will return a Series containing python "
            "datetime objects instead of an ndarray. To retain the old behavior, "
            "call `np.array` on the result",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pydatetime()

    def isocalendar(self) -> DataFrame:
        from pandas import DataFrame

        result = (
            cast(ArrowExtensionArray, self._parent.array)
            ._dt_isocalendar()
            ._pa_array.combine_chunks()
        )
        iso_calendar_df = DataFrame(
            {
                col: type(self._parent.array)(result.field(i))  # type: ignore[call-arg]
                for i, col in enumerate(["year", "week", "day"])
            }
        )
        return iso_calendar_df

    @property
    def components(self) -> DataFrame:
        from pandas import DataFrame

        components_df = DataFrame(
            {
                col: getattr(self._parent.array, f"_dt_{col}")
                for col in [
                    "days",
                    "hours",
                    "minutes",
                    "seconds",
                    "milliseconds",
                    "microseconds",
                    "nanoseconds",
                ]
            }
        )
        return components_df


@delegate_names(
    delegate=DatetimeArray,
    accessors=DatetimeArray._datetimelike_ops + ["unit"],
    typ="property",
)
@delegate_names(
    delegate=DatetimeArray,
    accessors=DatetimeArray._datetimelike_methods + ["as_unit"],
    typ="method",
)
class DatetimeProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Examples
    --------
    >>> seconds_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="s"))
    >>> seconds_series
    0   2000-01-01 00:00:00
    1   2000-01-01 00:00:01
    2   2000-01-01 00:00:02
    dtype: datetime64[ns]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    dtype: int32

    >>> hours_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="h"))
    >>> hours_series
    0   2000-01-01 00:00:00
    1   2000-01-01 01:00:00
    2   2000-01-01 02:00:00
    dtype: datetime64[ns]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    dtype: int32

    >>> quarters_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="QE"))
    >>> quarters_series
    0   2000-03-31
    1   2000-06-30
    2   2000-09-30
    dtype: datetime64[ns]
    >>> quarters_series.dt.quarter
    0    1
    1    2
    2    3
    dtype: int32

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.
    """

    def to_pydatetime(self) -> np.ndarray:
        """
        Return the data as an array of :class:`datetime.datetime` objects.

        .. deprecated:: 2.1.0

            The current behavior of dt.to_pydatetime is deprecated.
            In a future version this will return a Series containing python
            datetime objects instead of a ndarray.

        Timezone information is retained if present.

        .. warning::

           Python's datetime uses microsecond resolution, which is lower than
           pandas (nanosecond). The values are truncated.

        Returns
        -------
        numpy.ndarray
            Object dtype array containing native Python datetime objects.

        See Also
        --------
        datetime.datetime : Standard library value for a datetime.

        Examples
        --------
        >>> s = pd.Series(pd.date_range('20180310', periods=2))
        >>> s
        0   2018-03-10
        1   2018-03-11
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        array([datetime.datetime(2018, 3, 10, 0, 0),
               datetime.datetime(2018, 3, 11, 0, 0)], dtype=object)

        pandas' nanosecond precision is truncated to microseconds.

        >>> s = pd.Series(pd.date_range('20180310', periods=2, freq='ns'))
        >>> s
        0   2018-03-10 00:00:00.000000000
        1   2018-03-10 00:00:00.000000001
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        array([datetime.datetime(2018, 3, 10, 0, 0),
               datetime.datetime(2018, 3, 10, 0, 0)], dtype=object)
        """
        # GH#20306
        warnings.warn(
            f"The behavior of {type(self).__name__}.to_pydatetime is deprecated, "
            "in a future version this will return a Series containing python "
            "datetime objects instead of an ndarray. To retain the old behavior, "
            "call `np.array` on the result",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._get_values().to_pydatetime()

    @property
    def freq(self):
        return self._get_values().inferred_freq

    def isocalendar(self) -> DataFrame:
        """
        Calculate year, week, and day according to the ISO 8601 standard.

        Returns
        -------
        DataFrame
            With columns year, week and day.

        See Also
        --------
        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,
            week number, and weekday for the given Timestamp object.
        datetime.date.isocalendar : Return a named tuple object with
            three components: year, week and weekday.

        Examples
        --------
        >>> ser = pd.to_datetime(pd.Series(["2010-01-01", pd.NaT]))
        >>> ser.dt.isocalendar()
           year  week  day
        0  2009    53     5
        1  <NA>  <NA>  <NA>
        >>> ser.dt.isocalendar().week
        0      53
        1    <NA>
        Name: week, dtype: UInt32
        """
        return self._get_values().isocalendar().set_index(self._parent.index)


@delegate_names(
    delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=TimedeltaArray,
    accessors=TimedeltaArray._datetimelike_methods,
    typ="method",
)
class TimedeltaProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.

    Examples
    --------
    >>> seconds_series = pd.Series(
    ...     pd.timedelta_range(start="1 second", periods=3, freq="s")
    ... )
    >>> seconds_series
    0   0 days 00:00:01
    1   0 days 00:00:02
    2   0 days 00:00:03
    dtype: timedelta64[ns]
    >>> seconds_series.dt.seconds
    0    1
    1    2
    2    3
    dtype: int32
    """

    def to_pytimedelta(self) -> np.ndarray:
        """
        Return an array of native :class:`datetime.timedelta` objects.

        Python's standard `datetime` library uses a different representation
        timedelta's. This method converts a Series of pandas Timedeltas
        to `datetime.timedelta` format with the same length as the original
        Series.

        Returns
        -------
        numpy.ndarray
            Array of 1D containing data with `datetime.timedelta` type.

        See Also
        --------
        datetime.timedelta : A duration expressing the difference
            between two date, time, or datetime.

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="d"))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.to_pytimedelta()
        array([datetime.timedelta(0), datetime.timedelta(days=1),
        datetime.timedelta(days=2), datetime.timedelta(days=3),
        datetime.timedelta(days=4)], dtype=object)
        """
        return self._get_values().to_pytimedelta()

    @property
    def components(self):
        """
        Return a Dataframe of the components of the Timedeltas.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='s'))
        >>> s
        0   0 days 00:00:00
        1   0 days 00:00:01
        2   0 days 00:00:02
        3   0 days 00:00:03
        4   0 days 00:00:04
        dtype: timedelta64[ns]
        >>> s.dt.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     0      0        0        0             0             0            0
        1     0      0        0        1             0             0            0
        2     0      0        0        2             0             0            0
        3     0      0        0        3             0             0            0
        4     0      0        0        4             0             0            0
        """
        return (
            self._get_values()
            .components.set_index(self._parent.index)
            .__finalize__(self._parent)
        )

    @property
    def freq(self):
        return self._get_values().inferred_freq


@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_methods, typ="method"
)
class PeriodProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.

    Examples
    --------
    >>> seconds_series = pd.Series(
    ...     pd.period_range(
    ...         start="2000-01-01 00:00:00", end="2000-01-01 00:00:03", freq="s"
    ...     )
    ... )
    >>> seconds_series
    0    2000-01-01 00:00:00
    1    2000-01-01 00:00:01
    2    2000-01-01 00:00:02
    3    2000-01-01 00:00:03
    dtype: period[s]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    3    3
    dtype: int64

    >>> hours_series = pd.Series(
    ...     pd.period_range(start="2000-01-01 00:00", end="2000-01-01 03:00", freq="h")
    ... )
    >>> hours_series
    0    2000-01-01 00:00
    1    2000-01-01 01:00
    2    2000-01-01 02:00
    3    2000-01-01 03:00
    dtype: period[h]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    3    3
    dtype: int64

    >>> quarters_series = pd.Series(
    ...     pd.period_range(start="2000-01-01", end="2000-12-31", freq="Q-DEC")
    ... )
    >>> quarters_series
    0    2000Q1
    1    2000Q2
    2    2000Q3
    3    2000Q4
    dtype: period[Q-DEC]
    >>> quarters_series.dt.quarter
    0    1
    1    2
    2    3
    3    4
    dtype: int64
    """


class CombinedDatetimelikeProperties(
    DatetimeProperties, TimedeltaProperties, PeriodProperties
):
    def __new__(cls, data: Series):  # pyright: ignore[reportInconsistentConstructor]
        # CombinedDatetimelikeProperties isn't really instantiated. Instead
        # we need to choose which parent (datetime or timedelta) is
        # appropriate. Since we're checking the dtypes anyway, we'll just
        # do all the validation here.

        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        orig = data if isinstance(data.dtype, CategoricalDtype) else None
        if orig is not None:
            data = data._constructor(
                orig.array,
                name=orig.name,
                copy=False,
                dtype=orig._values.categories.dtype,
                index=orig.index,
            )

        if isinstance(data.dtype, ArrowDtype) and data.dtype.kind in "Mm":
            return ArrowTemporalProperties(data, orig)
        if lib.is_np_dtype(data.dtype, "M"):
            return DatetimeProperties(data, orig)
        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeProperties(data, orig)
        elif lib.is_np_dtype(data.dtype, "m"):
            return TimedeltaProperties(data, orig)
        elif isinstance(data.dtype, PeriodDtype):
            return PeriodProperties(data, orig)

        raise AttributeError("Can only use .dt accessor with datetimelike values")