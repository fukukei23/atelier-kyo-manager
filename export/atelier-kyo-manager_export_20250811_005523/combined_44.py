
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_utils.py ===
from io import StringIO

import pytest

import numpy as np
import numpy.lib._utils_impl as _utils_impl
from numpy.testing import assert_raises_regex


def test_assert_raises_regex_context_manager():
    with assert_raises_regex(ValueError, 'no deprecation warning'):
        raise ValueError('no deprecation warning')


def test_info_method_heading():
    # info(class) should only print "Methods:" heading if methods exist

    class NoPublicMethods:
        pass

    class WithPublicMethods:
        def first_method():
            pass

    def _has_method_heading(cls):
        out = StringIO()
        np.info(cls, output=out)
        return 'Methods:' in out.getvalue()

    assert _has_method_heading(WithPublicMethods)
    assert not _has_method_heading(NoPublicMethods)


def test_drop_metadata():
    def _compare_dtypes(dt1, dt2):
        return np.can_cast(dt1, dt2, casting='no')

    # structured dtype
    dt = np.dtype([('l1', [('l2', np.dtype('S8', metadata={'msg': 'toto'}))])],
                  metadata={'msg': 'titi'})
    dt_m = _utils_impl.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None
    assert dt_m['l1'].metadata is None
    assert dt_m['l1']['l2'].metadata is None

    # alignment
    dt = np.dtype([('x', '<f8'), ('y', '<i4')],
                  align=True,
                  metadata={'msg': 'toto'})
    dt_m = _utils_impl.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None

    # subdtype
    dt = np.dtype('8f',
                  metadata={'msg': 'toto'})
    dt_m = _utils_impl.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None

    # scalar
    dt = np.dtype('uint32',
                  metadata={'msg': 'toto'})
    dt_m = _utils_impl.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None


@pytest.mark.parametrize("dtype",
        [np.dtype("i,i,i,i")[["f1", "f3"]],
        np.dtype("f8"),
        np.dtype("10i")])
def test_drop_metadata_identity_and_copy(dtype):
    # If there is no metadata, the identity is preserved:
    assert _utils_impl.drop_metadata(dtype) is dtype

    # If there is any, it is dropped (subforms are checked above)
    dtype = np.dtype(dtype, metadata={1: 2})
    assert _utils_impl.drop_metadata(dtype).metadata is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\series.py ===
### The base class for all series
from collections.abc import Callable
from sympy.calculus.util import continuous_domain
from sympy.concrete import Sum, Product
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions import atan2, zeta, frac, ceiling, floor, im
from sympy.core.relational import (Equality, GreaterThan,
    LessThan, Relational, Ne)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.utils import _get_free_symbols, extract_solution
from sympy.printing.latex import latex
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.precedence import precedence
from sympy.sets.sets import Set, Interval, Union
from sympy.simplify.simplify import nsimplify
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.lambdify import lambdify
from .intervalmath import interval
import warnings


class IntervalMathPrinter(PythonCodePrinter):
    """A printer to be used inside `plot_implicit` when `adaptive=True`,
    in which case the interval arithmetic module is going to be used, which
    requires the following edits.
    """
    def _print_And(self, expr):
        PREC = precedence(expr)
        return " & ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return " | ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))


def _uniform_eval(f1, f2, *args, modules=None,
    force_real_eval=False, has_sum=False):
    """
    Note: this is an experimental function, as such it is prone to changes.
    Please, do not use it in your code.
    """
    np = import_module('numpy')

    def wrapper_func(func, *args):
        try:
            return complex(func(*args))
        except (ZeroDivisionError, OverflowError):
            return complex(np.nan, np.nan)

    # NOTE: np.vectorize is much slower than numpy vectorized operations.
    # However, this modules must be able to evaluate functions also with
    # mpmath or sympy.
    wrapper_func = np.vectorize(wrapper_func, otypes=[complex])

    def _eval_with_sympy(err=None):
        if f2 is None:
            msg = "Impossible to evaluate the provided numerical function"
            if err is None:
                msg += "."
            else:
                msg += "because the following exception was raised:\n"
                "{}: {}".format(type(err).__name__, err)
            raise RuntimeError(msg)
        if err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not modules else modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
        return wrapper_func(f2, *args)

    if modules == "sympy":
        return _eval_with_sympy()

    try:
        return wrapper_func(f1, *args)
    except Exception as err:
        return _eval_with_sympy(err)


def _adaptive_eval(f, x):
    """Evaluate f(x) with an adaptive algorithm. Post-process the result.
    If a symbolic expression is evaluated with SymPy, it might returns
    another symbolic expression, containing additions, ...
    Force evaluation to a float.

    Parameters
    ==========
    f : callable
    x : float
    """
    np = import_module('numpy')

    y = f(x)
    if isinstance(y, Expr) and (not y.is_Number):
        y = y.evalf()
    y = complex(y)
    if y.imag > 1e-08:
        return np.nan
    return y.real


def _get_wrapper_for_expr(ret):
    wrapper = "%s"
    if ret == "real":
        wrapper = "re(%s)"
    elif ret == "imag":
        wrapper = "im(%s)"
    elif ret == "abs":
        wrapper = "abs(%s)"
    elif ret == "arg":
        wrapper = "arg(%s)"
    return wrapper


class BaseSeries:
    """Base class for the data objects containing stuff to be plotted.

    Notes
    =====

    The backend should check if it supports the data series that is given.
    (e.g. TextBackend supports only LineOver1DRangeSeries).
    It is the backend responsibility to know how to use the class of
    data series that is given.

    Some data series classes are grouped (using a class attribute like is_2Dline)
    according to the api they present (based only on convention). The backend is
    not obliged to use that api (e.g. LineOver1DRangeSeries belongs to the
    is_2Dline group and presents the get_points method, but the
    TextBackend does not use the get_points method).

    BaseSeries
    """

    # Some flags follow. The rationale for using flags instead of checking base
    # classes is that setting multiple flags is simpler than multiple
    # inheritance.

    is_2Dline = False
    # Some of the backends expect:
    #  - get_points returning 1D np.arrays list_x, list_y
    #  - get_color_array returning 1D np.array (done in Line2DBaseSeries)
    # with the colors calculated at the points from get_points

    is_3Dline = False
    # Some of the backends expect:
    #  - get_points returning 1D np.arrays list_x, list_y, list_y
    #  - get_color_array returning 1D np.array (done in Line2DBaseSeries)
    # with the colors calculated at the points from get_points

    is_3Dsurface = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes

    is_contour = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes

    is_implicit = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x (1D array), mesh_y(1D array,
    #     mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes
    # Different from is_contour as the colormap in backend will be
    # different

    is_interactive = False
    # An interactive series can update its data.

    is_parametric = False
    # The calculation of aesthetics expects:
    #   - get_parameter_points returning one or two np.arrays (1D or 2D)
    # used for calculation aesthetics

    is_generic = False
    # Represent generic user-provided numerical data

    is_vector = False
    is_2Dvector = False
    is_3Dvector = False
    # Represents a 2D or 3D vector data series

    _N = 100
    # default number of discretization points for uniform sampling. Each
    # subclass can set its number.

    def __init__(self, *args, **kwargs):
        kwargs = _set_discretization_points(kwargs.copy(), type(self))
        # discretize the domain using only integer numbers
        self.only_integers = kwargs.get("only_integers", False)
        # represents the evaluation modules to be used by lambdify
        self.modules = kwargs.get("modules", None)
        # plot functions might create data series that might not be useful to
        # be shown on the legend, for example wireframe lines on 3D plots.
        self.show_in_legend = kwargs.get("show_in_legend", True)
        # line and surface series can show data with a colormap, hence a
        # colorbar is essential to understand the data. However, sometime it
        # is useful to hide it on series-by-series base. The following keyword
        # controls whether the series should show a colorbar or not.
        self.colorbar = kwargs.get("colorbar", True)
        # Some series might use a colormap as default coloring. Setting this
        # attribute to False will inform the backends to use solid color.
        self.use_cm = kwargs.get("use_cm", False)
        # If True, the backend will attempt to render it on a polar-projection
        # axis, or using a polar discretization if a 3D plot is requested
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        # If True, the rendering will use points, not lines.
        self.is_point = kwargs.get("is_point", kwargs.get("point", False))
        # some backend is able to render latex, other needs standard text
        self._label = self._latex_label = ""

        self._ranges = []
        self._n = [
            int(kwargs.get("n1", self._N)),
            int(kwargs.get("n2", self._N)),
            int(kwargs.get("n3", self._N))
        ]
        self._scales = [
            kwargs.get("xscale", "linear"),
            kwargs.get("yscale", "linear"),
            kwargs.get("zscale", "linear")
        ]

        # enable interactive widget plots
        self._params = kwargs.get("params", {})
        if not isinstance(self._params, dict):
            raise TypeError("`params` must be a dictionary mapping symbols "
                "to numeric values.")
        if len(self._params) > 0:
            self.is_interactive = True

        # contains keyword arguments that will be passed to the rendering
        # function of the chosen plotting library
        self.rendering_kw = kwargs.get("rendering_kw", {})

        # numerical transformation functions to be applied to the output data:
        # x, y, z (coordinates), p (parameter on parametric plots)
        self._tx = kwargs.get("tx", None)
        self._ty = kwargs.get("ty", None)
        self._tz = kwargs.get("tz", None)
        self._tp = kwargs.get("tp", None)
        if not all(callable(t) or (t is None) for t in
            [self._tx, self._ty, self._tz, self._tp]):
            raise TypeError("`tx`, `ty`, `tz`, `tp` must be functions.")

        # list of numerical functions representing the expressions to evaluate
        self._functions = []
        # signature for the numerical functions
        self._signature = []
        # some expressions don't like to be evaluated over complex data.
        # if that's the case, set this to True
        self._force_real_eval = kwargs.get("force_real_eval", None)
        # this attribute will eventually contain a dictionary with the
        # discretized ranges
        self._discretized_domain = None
        # whether the series contains any interactive range, which is a range
        # where the minimum and maximum values can be changed with an
        # interactive widget
        self._interactive_ranges = False
        # NOTE: consider a generic summation, for example:
        #   s = Sum(cos(pi * x), (x, 1, y))
        # This gets lambdified to something:
        #   sum(cos(pi*x) for x in range(1, y+1))
        # Hence, y needs to be an integer, otherwise it raises:
        #   TypeError: 'complex' object cannot be interpreted as an integer
        # This list will contains symbols that are upper bound to summations
        # or products
        self._needs_to_be_int = []
        # a color function will be responsible to set the line/surface color
        # according to some logic. Each data series will et an appropriate
        # default value.
        self.color_func = None
        # NOTE: color_func usually receives numerical functions that are going
        # to be evaluated over the coordinates of the computed points (or the
        # discretized meshes).
        # However, if an expression is given to color_func, then it will be
        # lambdified with symbols in self._signature, and it will be evaluated
        # with the same data used to evaluate the plotted expression.
        self._eval_color_func_with_signature = False

    def _block_lambda_functions(self, *exprs):
        """Some data series can be used to plot numerical functions, others
        cannot. Execute this method inside the `__init__` to prevent the
        processing of numerical functions.
        """
        if any(callable(e) for e in exprs):
            raise TypeError(type(self).__name__ + " requires a symbolic "
                "expression.")

    def _check_fs(self):
        """ Checks if there are enough parameters and free symbols.
        """
        exprs, ranges = self.expr, self.ranges
        params, label = self.params, self.label
        exprs = exprs if hasattr(exprs, "__iter__") else [exprs]
        if any(callable(e) for e in exprs):
            return

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = _get_free_symbols(exprs)
        fs = fs.difference(params.keys())
        if ranges is not None:
            fs = fs.difference([r[0] for r in ranges])

        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n"
                + "Expression: {}\n".format(
                    (exprs, ranges, label) if ranges is not None else (exprs, label))
                + "params: {}\n".format(params)
                + "Specify what these symbols represent: {}\n".format(fs)
                + "Are they ranges or parameters?"
            )

        # verify that all symbols are known (they either represent plotting
        # ranges or parameters)
        range_symbols = [r[0] for r in ranges]
        for r in ranges:
            fs = set().union(*[e.free_symbols for e in r[1:]])
            if any(t in fs for t in range_symbols):
                # ranges can't depend on each other, for example this are
                # not allowed:
                # (x, 0, y), (y, 0, 3)
                # (x, 0, y), (y, x + 2, 3)
                raise ValueError("Range symbols can't be included into "
                    "minimum and maximum of a range. "
                    "Received range: %s" % str(r))
            if len(fs) > 0:
                self._interactive_ranges = True
            remaining_fs = fs.difference(params.keys())
            if len(remaining_fs) > 0:
                raise ValueError(
                    "Unknown symbols found in plotting range: %s. " % (r,) +
                    "Are the following parameters? %s" % remaining_fs)

    def _create_lambda_func(self):
        """Create the lambda functions to be used by the uniform meshing
        strategy.

        Notes
        =====
        The old sympy.plotting used experimental_lambdify. It created one
        lambda function each time an evaluation was requested. If that failed,
        it went on to create a different lambda function and evaluated it,
        and so on.

        This new module changes strategy: it creates right away the default
        lambda function as well as the backup one. The reason is that the
        series could be interactive, hence the numerical function will be
        evaluated multiple times. So, let's create the functions just once.

        This approach works fine for the majority of cases, in which the
        symbolic expression is relatively short, hence the lambdification
        is fast. If the expression is very long, this approach takes twice
        the time to create the lambda functions. Be aware of that!
        """
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if not any(callable(e) for e in exprs):
            fs = _get_free_symbols(exprs)
            self._signature = sorted(fs, key=lambda t: t.name)

            # Generate a list of lambda functions, two for each expression:
            # 1. the default one.
            # 2. the backup one, in case of failures with the default one.
            self._functions = []
            for e in exprs:
                # TODO: set cse=True once this issue is solved:
                # https://github.com/sympy/sympy/issues/24246
                self._functions.append([
                    lambdify(self._signature, e, modules=self.modules),
                    lambdify(self._signature, e, modules="sympy", dummify=True),
                ])
        else:
            self._signature = sorted([r[0] for r in self.ranges], key=lambda t: t.name)
            self._functions = [(e, None) for e in exprs]

        # deal with symbolic color_func
        if isinstance(self.color_func, Expr):
            self.color_func = lambdify(self._signature, self.color_func)
            self._eval_color_func_with_signature = True

    def _update_range_value(self, t):
        """If the value of a plotting range is a symbolic expression,
        substitute the parameters in order to get a numerical value.
        """
        if not self._interactive_ranges:
            return complex(t)
        return complex(t.subs(self.params))

    def _create_discretized_domain(self):
        """Discretize the ranges for uniform meshing strategy.
        """
        # NOTE: the goal is to create a dictionary stored in
        # self._discretized_domain, mapping symbols to a numpy array
        # representing the discretization
        discr_symbols = []
        discretizations = []

        # create a 1D discretization
        for i, r in enumerate(self.ranges):
            discr_symbols.append(r[0])
            c_start = self._update_range_value(r[1])
            c_end = self._update_range_value(r[2])
            start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
            end = c_end.real if c_start.imag == c_end.imag == 0 else c_end
            needs_integer_discr = self.only_integers or (r[0] in self._needs_to_be_int)
            d = BaseSeries._discretize(start, end, self.n[i],
                scale=self.scales[i],
                only_integers=needs_integer_discr)

            if ((not self._force_real_eval) and (not needs_integer_discr) and
                (d.dtype != "complex")):
                d = d + 1j * c_start.imag

            if needs_integer_discr:
                d = d.astype(int)

            discretizations.append(d)

        # create 2D or 3D
        self._create_discretized_domain_helper(discr_symbols, discretizations)

    def _create_discretized_domain_helper(self, discr_symbols, discretizations):
        """Create 2D or 3D discretized grids.

        Subclasses should override this method in order to implement a
        different behaviour.
        """
        np = import_module('numpy')

        # discretization suitable for 2D line plots, 3D surface plots,
        # contours plots, vector plots
        # NOTE: why indexing='ij'? Because it produces consistent results with
        # np.mgrid. This is important as Mayavi requires this indexing
        # to correctly compute 3D streamlines. While VTK is able to compute
        # streamlines regardless of the indexing, with indexing='xy' it
        # produces "strange" results with "voids" into the
        # discretization volume. indexing='ij' solves the problem.
        # Also note that matplotlib 2D streamlines requires indexing='xy'.
        indexing = "xy"
        if self.is_3Dvector or (self.is_3Dsurface and self.is_implicit):
            indexing = "ij"
        meshes = np.meshgrid(*discretizations, indexing=indexing)
        self._discretized_domain = dict(zip(discr_symbols, meshes))

    def _evaluate(self, cast_to_real=True):
        """Evaluation of the symbolic expression (or expressions) with the
        uniform meshing strategy, based on current values of the parameters.
        """
        np = import_module('numpy')

        # create lambda functions
        if not self._functions:
            self._create_lambda_func()
        # create (or update) the discretized domain
        if (not self._discretized_domain) or self._interactive_ranges:
            self._create_discretized_domain()
        # ensure that discretized domains are returned with the proper order
        discr = [self._discretized_domain[s[0]] for s in self.ranges]

        args = self._aggregate_args()

        results = []
        for f in self._functions:
            r = _uniform_eval(*f, *args)
            # the evaluation might produce an int/float. Need this correction.
            r = self._correct_shape(np.array(r), discr[0])
            # sometime the evaluation is performed over arrays of type object.
            # hence, `result` might be of type object, which don't work well
            # with numpy real and imag functions.
            r = r.astype(complex)
            results.append(r)

        if cast_to_real:
            discr = [np.real(d.astype(complex)) for d in discr]
        return [*discr, *results]

    def _aggregate_args(self):
        """Create a list of arguments to be passed to the lambda function,
        sorted according to self._signature.
        """
        args = []
        for s in self._signature:
            if s in self._params.keys():
                args.append(
                    int(self._params[s]) if s in self._needs_to_be_int else
                    self._params[s] if self._force_real_eval
                    else complex(self._params[s]))
            else:
                args.append(self._discretized_domain[s])
        return args

    @property
    def expr(self):
        """Return the expression (or expressions) of the series."""
        return self._expr

    @expr.setter
    def expr(self, e):
        """Set the expression (or expressions) of the series."""
        is_iter = hasattr(e, "__iter__")
        is_callable = callable(e) if not is_iter else any(callable(t) for t in e)
        if is_callable:
            self._expr = e
        else:
            self._expr = sympify(e) if not is_iter else Tuple(*e)

            # look for the upper bound of summations and products
            s = set()
            for e in self._expr.atoms(Sum, Product):
                for a in e.args[1:]:
                    if isinstance(a[-1], Symbol):
                        s.add(a[-1])
            self._needs_to_be_int = list(s)

            # list of sympy functions that when lambdified, the corresponding
            # numpy functions don't like complex-type arguments
            pf = [ceiling, floor, atan2, frac, zeta]
            if self._force_real_eval is not True:
                check_res = [self._expr.has(f) for f in pf]
                self._force_real_eval = any(check_res)
                if self._force_real_eval and ((self.modules is None) or
                    (isinstance(self.modules, str) and "numpy" in self.modules)):
                    funcs = [f for f, c in zip(pf, check_res) if c]
                    warnings.warn("NumPy is unable to evaluate with complex "
                        "numbers some of the functions included in this "
                        "symbolic expression: %s. " % funcs +
                        "Hence, the evaluation will use real numbers. "
                        "If you believe the resulting plot is incorrect, "
                        "change the evaluation module by setting the "
                        "`modules` keyword argument.")
            if self._functions:
                # update lambda functions
                self._create_lambda_func()

    @property
    def is_3D(self):
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    def _line_surface_color(self, prop, val):
        """This method enables back-compatibility with old sympy.plotting"""
        # NOTE: color_func is set inside the init method of the series.
        # If line_color/surface_color is not a callable, then color_func will
        # be set to None.
        setattr(self, prop, val)
        if callable(val) or isinstance(val, Expr):
            self.color_func = val
            setattr(self, prop, None)
        elif val is not None:
            self.color_func = None

    @property
    def line_color(self):
        return self._line_color

    @line_color.setter
    def line_color(self, val):
        self._line_surface_color("_line_color", val)

    @property
    def n(self):
        """Returns a list [n1, n2, n3] of numbers of discratization points.
        """
        return self._n

    @n.setter
    def n(self, v):
        """Set the numbers of discretization points. ``v`` must be an int or
        a list.

        Let ``s`` be a series. Then:

        * to set the number of discretization points along the x direction (or
          first parameter): ``s.n = 10``
        * to set the number of discretization points along the x and y
          directions (or first and second parameters): ``s.n = [10, 15]``
        * to set the number of discretization points along the x, y and z
          directions: ``s.n = [10, 15, 20]``

        The following is highly unreccomended, because it prevents
        the execution of necessary code in order to keep updated data:
        ``s.n[1] = 15``
        """
        if not hasattr(v, "__iter__"):
            self._n[0] = v
        else:
            self._n[:len(v)] = v
        if self._discretized_domain:
            # update the discretized domain
            self._create_discretized_domain()

    @property
    def params(self):
        """Get or set the current parameters dictionary.

        Parameters
        ==========

        p : dict

            * key: symbol associated to the parameter
            * val: the numeric value
        """
        return self._params

    @params.setter
    def params(self, p):
        self._params = p

    def _post_init(self):
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if any(callable(e) for e in exprs) and self.params:
            raise TypeError("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "user-provided numerical functions.")

        # if the expressions is a lambda function and no label has been
        # provided, then its better to do the following in order to avoid
        # surprises on the backend
        if any(callable(e) for e in exprs):
            if self._label == str(self.expr):
                self.label = ""

        self._check_fs()

        if hasattr(self, "adaptive") and self.adaptive and self.params:
            warnings.warn("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "adaptive evaluation. Automatically switched to "
                "adaptive=False.")
            self.adaptive = False

    @property
    def scales(self):
        return self._scales

    @scales.setter
    def scales(self, v):
        if isinstance(v, str):
            self._scales[0] = v
        else:
            self._scales[:len(v)] = v

    @property
    def surface_color(self):
        return self._surface_color

    @surface_color.setter
    def surface_color(self, val):
        self._line_surface_color("_surface_color", val)

    @property
    def rendering_kw(self):
        return self._rendering_kw

    @rendering_kw.setter
    def rendering_kw(self, kwargs):
        if isinstance(kwargs, dict):
            self._rendering_kw = kwargs
        else:
            self._rendering_kw = {}
            if kwargs is not None:
                warnings.warn(
                    "`rendering_kw` must be a dictionary, instead an "
                    "object of type %s was received. " % type(kwargs) +
                    "Automatically setting `rendering_kw` to an empty "
                    "dictionary")

    @staticmethod
    def _discretize(start, end, N, scale="linear", only_integers=False):
        """Discretize a 1D domain.

        Returns
        =======

        domain : np.ndarray with dtype=float or complex
            The domain's dtype will be float or complex (depending on the
            type of start/end) even if only_integers=True. It is left for
            the downstream code to perform further casting, if necessary.
        """
        np = import_module('numpy')

        if only_integers is True:
            start, end = int(start), int(end)
            N = end - start + 1

        if scale == "linear":
            return np.linspace(start, end, N)
        return np.geomspace(start, end, N)

    @staticmethod
    def _correct_shape(a, b):
        """Convert ``a`` to a np.ndarray of the same shape of ``b``.

        Parameters
        ==========

        a : int, float, complex, np.ndarray
            Usually, this is the result of a numerical evaluation of a
            symbolic expression. Even if a discretized domain was used to
            evaluate the function, the result can be a scalar (int, float,
            complex). Think for example to ``expr = Float(2)`` and
            ``f = lambdify(x, expr)``. No matter the shape of the numerical
            array representing x, the result of the evaluation will be
            a single value.

        b : np.ndarray
            It represents the correct shape that ``a`` should have.

        Returns
        =======
        new_a : np.ndarray
            An array with the correct shape.
        """
        np = import_module('numpy')

        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if a.shape != b.shape:
            if a.shape == ():
                a = a * np.ones_like(b)
            else:
                a = a.reshape(b.shape)
        return a

    def eval_color_func(self, *args):
        """Evaluate the color function.

        Parameters
        ==========

        args : tuple
            Arguments to be passed to the coloring function. Can be coordinates
            or parameters or both.

        Notes
        =====

        The backend will request the data series to generate the numerical
        data. Depending on the data series, either the data series itself or
        the backend will eventually execute this function to generate the
        appropriate coloring value.
        """
        np = import_module('numpy')
        if self.color_func is None:
            # NOTE: with the line_color and surface_color attributes
            # (back-compatibility with the old sympy.plotting module) it is
            # possible to create a plot with a callable line_color (or
            # surface_color). For example:
            # p = plot(sin(x), line_color=lambda x, y: -y)
            # This creates a ColoredLineOver1DRangeSeries with line_color=None
            # and color_func=lambda x, y: -y, which effectively is a
            # parametric series. Later we could change it to a string value:
            # p[0].line_color = "red"
            # However, this sets ine_color="red" and color_func=None, but the
            # series is still ColoredLineOver1DRangeSeries (a parametric
            # series), which will render using a color_func...
            warnings.warn("This is likely not the result you were "
                "looking for. Please, re-execute the plot command, this time "
                "with the appropriate an appropriate value to line_color "
                "or surface_color.")
            return np.ones_like(args[0])

        if self._eval_color_func_with_signature:
            args = self._aggregate_args()
            color = self.color_func(*args)
            _re, _im = np.real(color), np.imag(color)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            return _re

        nargs = arity(self.color_func)
        if nargs == 1:
            if self.is_2Dline and self.is_parametric:
                if len(args) == 2:
                    # ColoredLineOver1DRangeSeries
                    return self._correct_shape(self.color_func(args[0]), args[0])
                # Parametric2DLineSeries
                return self._correct_shape(self.color_func(args[2]), args[2])
            elif self.is_3Dline and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            elif self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            return self._correct_shape(self.color_func(args[0]), args[0])
        elif nargs == 2:
            if self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(*args[3:]), args[3])
            return self._correct_shape(self.color_func(*args[:2]), args[0])
        return self._correct_shape(self.color_func(*args[:nargs]), args[0])

    def get_data(self):
        """Compute and returns the numerical data.

        The number of parameters returned by this method depends on the
        specific instance. If ``s`` is the series, make sure to read
        ``help(s.get_data)`` to understand what it returns.
        """
        raise NotImplementedError

    def _get_wrapped_label(self, label, wrapper):
        """Given a latex representation of an expression, wrap it inside
        some characters. Matplotlib needs "$%s%$", K3D-Jupyter needs "%s".
        """
        return wrapper % label

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        if use_latex is False:
            return self._label
        if self._label == str(self.expr):
            # when the backend requests a latex label and user didn't provide
            # any label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label

    @property
    def label(self):
        return self.get_label()

    @label.setter
    def label(self, val):
        """Set the labels associated to this series."""
        # NOTE: the init method of any series requires a label. If the user do
        # not provide it, the preprocessing function will set label=None, which
        # informs the series to initialize two attributes:
        # _label contains the string representation of the expression.
        # _latex_label contains the latex representation of the expression.
        self._label = self._latex_label = val

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, val):
        new_vals = []
        for v in val:
            if v is not None:
                new_vals.append(tuple([sympify(t) for t in v]))
        self._ranges = new_vals

    def _apply_transform(self, *args):
        """Apply transformations to the results of numerical evaluation.

        Parameters
        ==========
        args : tuple
            Results of numerical evaluation.

        Returns
        =======
        transformed_args : tuple
            Tuple containing the transformed results.
        """
        t = lambda x, transform: x if transform is None else transform(x)
        x, y, z = None, None, None
        if len(args) == 2:
            x, y = args
            return t(x, self._tx), t(y, self._ty)
        elif (len(args) == 3) and isinstance(self, Parametric2DLineSeries):
            x, y, u = args
            return (t(x, self._tx), t(y, self._ty), t(u, self._tp))
        elif len(args) == 3:
            x, y, z = args
            return t(x, self._tx), t(y, self._ty), t(z, self._tz)
        elif (len(args) == 4) and isinstance(self, Parametric3DLineSeries):
            x, y, z, u = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), t(u, self._tp))
        elif len(args) == 4: # 2D vector plot
            x, y, u, v = args
            return (
                t(x, self._tx), t(y, self._ty),
                t(u, self._tx), t(v, self._ty)
            )
        elif (len(args) == 5) and isinstance(self, ParametricSurfaceSeries):
            x, y, z, u, v = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), u, v)
        elif (len(args) == 6) and self.is_3Dvector: # 3D vector plot
            x, y, z, u, v, w = args
            return (
                t(x, self._tx), t(y, self._ty), t(z, self._tz),
                t(u, self._tx), t(v, self._ty), t(w, self._tz)
            )
        elif len(args) == 6: # complex plot
            x, y, _abs, _arg, img, colors = args
            return (
                x, y, t(_abs, self._tz), _arg, img, colors)
        return args

    def _str_helper(self, s):
        pre, post = "", ""
        if self.is_interactive:
            pre = "interactive "
            post = " and parameters " + str(tuple(self.params.keys()))
        return pre + s + post


def _detect_poles_numerical_helper(x, y, eps=0.01, expr=None, symb=None, symbolic=False):
    """Compute the steepness of each segment. If it's greater than a
    threshold, set the right-point y-value non NaN and record the
    corresponding x-location for further processing.

    Returns
    =======
    x : np.ndarray
        Unchanged x-data.
    yy : np.ndarray
        Modified y-data with NaN values.
    """
    np = import_module('numpy')

    yy = y.copy()
    threshold = np.pi / 2 - eps
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = abs(y[i + 1] - y[i])
        angle = np.arctan(dy / dx)
        if abs(angle) >= threshold:
            yy[i + 1] = np.nan

    return x, yy

def _detect_poles_symbolic_helper(expr, symb, start, end):
    """Attempts to compute symbolic discontinuities.

    Returns
    =======
    pole : list
        List of symbolic poles, possibly empty.
    """
    poles = []
    interval = Interval(nsimplify(start), nsimplify(end))
    res = continuous_domain(expr, symb, interval)
    res = res.simplify()
    if res == interval:
        pass
    elif (isinstance(res, Union) and
        all(isinstance(t, Interval) for t in res.args)):
        poles = []
        for s in res.args:
            if s.left_open:
                poles.append(s.left)
            if s.right_open:
                poles.append(s.right)
        poles = list(set(poles))
    else:
        raise ValueError(
            f"Could not parse the following object: {res} .\n"
            "Please, submit this as a bug. Consider also to set "
            "`detect_poles=True`."
        )
    return poles


### 2D lines
class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines.

    - adding the label, steps and only_integers options
    - making is_2Dline true
    - defining get_segments and get_color_array
    """

    is_2Dline = True
    _dim = 2
    _N = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = kwargs.get("steps", False)
        self.is_point = kwargs.get("is_point", kwargs.get("point", False))
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))
        self.adaptive = kwargs.get("adaptive", False)
        self.depth = kwargs.get('depth', 12)
        self.use_cm = kwargs.get("use_cm", False)
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        self.detect_poles = kwargs.get("detect_poles", False)
        self.eps = kwargs.get("eps", 0.01)
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.unwrap = kwargs.get("unwrap", False)
        # when detect_poles="symbolic", stores the location of poles so that
        # they can be appropriately rendered
        self.poles_locations = []
        exclude = kwargs.get("exclude", [])
        if isinstance(exclude, Set):
            exclude = list(extract_solution(exclude, n=100))
        if not hasattr(exclude, "__iter__"):
            exclude = [exclude]
        exclude = [float(e) for e in exclude]
        self.exclude = sorted(exclude)

    def get_data(self):
        """Return coordinates for plotting the line.

        Returns
        =======

        x: np.ndarray
            x-coordinates

        y: np.ndarray
            y-coordinates

        z: np.ndarray (optional)
            z-coordinates in case of Parametric3DLineSeries,
            Parametric3DLineInteractiveSeries

        param : np.ndarray (optional)
            The parameter in case of Parametric2DLineSeries,
            Parametric3DLineSeries or AbsArgLineSeries (and their
            corresponding interactive series).
        """
        np = import_module('numpy')
        points = self._get_data_helper()

        if (isinstance(self, LineOver1DRangeSeries) and
            (self.detect_poles == "symbolic")):
            poles = _detect_poles_symbolic_helper(
                self.expr.subs(self.params), *self.ranges[0])
            poles = np.array([float(t) for t in poles])
            t = lambda x, transform: x if transform is None else transform(x)
            self.poles_locations = t(np.array(poles), self._tx)

        # postprocessing
        points = self._apply_transform(*points)

        if self.is_2Dline and self.detect_poles:
            if len(points) == 2:
                x, y = points
                x, y = _detect_poles_numerical_helper(
                    x, y, self.eps)
                points = (x, y)
            else:
                x, y, p = points
                x, y = _detect_poles_numerical_helper(x, y, self.eps)
                points = (x, y, p)

        if self.unwrap:
            kw = {}
            if self.unwrap is not True:
                kw = self.unwrap
            if self.is_2Dline:
                if len(points) == 2:
                    x, y = points
                    y = np.unwrap(y, **kw)
                    points = (x, y)
                else:
                    x, y, p = points
                    y = np.unwrap(y, **kw)
                    points = (x, y, p)

        if self.steps is True:
            if self.is_2Dline:
                x, y = points[0], points[1]
                x = np.array((x, x)).T.flatten()[1:]
                y = np.array((y, y)).T.flatten()[:-1]
                if self.is_parametric:
                    points = (x, y, points[2])
                else:
                    points = (x, y)
            elif self.is_3Dline:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                if len(points) > 3:
                    points = (x, y, z, points[3])
                else:
                    points = (x, y, z)

        if len(self.exclude) > 0:
            points = self._insert_exclusions(points)
        return points

    def get_segments(self):
        sympy_deprecation_warning(
            """
            The Line2DBaseSeries.get_segments() method is deprecated.

            Instead, use the MatplotlibBackend.get_segments() method, or use
            The get_points() or get_data() methods.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-get-segments")

        np = import_module('numpy')
        points = type(self).get_data(self)
        points = np.ma.array(points).T.reshape(-1, 1, self._dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _insert_exclusions(self, points):
        """Add NaN to each of the exclusion point. Practically, this adds a
        NaN to the exclusion point, plus two other nearby points evaluated with
        the numerical functions associated to this data series.
        These nearby points are important when the number of discretization
        points is low, or the scale is logarithm.

        NOTE: it would be easier to just add exclusion points to the
        discretized domain before evaluation, then after evaluation add NaN
        to the exclusion points. But that's only work with adaptive=False.
        The following approach work even with adaptive=True.
        """
        np = import_module("numpy")
        points = list(points)
        n = len(points)
        # index of the x-coordinate (for 2d plots) or parameter (for 2d/3d
        # parametric plots)
        k = n - 1
        if n == 2:
            k = 0
        # indices of the other coordinates
        j_indeces = sorted(set(range(n)).difference([k]))
        # TODO: for now, I assume that numpy functions are going to succeed
        funcs = [f[0] for f in self._functions]

        for e in self.exclude:
            res = points[k] - e >= 0
            # if res contains both True and False, ie, if e is found
            if any(res) and any(~res):
                idx = np.nanargmax(res)
                # select the previous point with respect to e
                idx -= 1
                # TODO: what if points[k][idx]==e or points[k][idx+1]==e?

                if idx > 0 and idx < len(points[k]) - 1:
                    delta_prev = abs(e - points[k][idx])
                    delta_post = abs(e - points[k][idx + 1])
                    delta = min(delta_prev, delta_post) / 100
                    prev = e - delta
                    post = e + delta

                    # add points to the x-coord or the parameter
                    points[k] = np.concatenate(
                        (points[k][:idx], [prev, e, post], points[k][idx+1:]))

                    # add points to the other coordinates
                    c = 0
                    for j in j_indeces:
                        values = funcs[c](np.array([prev, post]))
                        c += 1
                        points[j] = np.concatenate(
                            (points[j][:idx], [values[0], np.nan, values[1]], points[j][idx+1:]))
        return points

    @property
    def var(self):
        return None if not self.ranges else self.ranges[0][0]

    @property
    def start(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def xscale(self):
        return self._scales[0]

    @xscale.setter
    def xscale(self, v):
        self.scales = v

    def get_color_array(self):
        np = import_module('numpy')
        c = self.line_color
        if hasattr(c, '__call__'):
            f = np.vectorize(c)
            nargs = arity(c)
            if nargs == 1 and self.is_parametric:
                x = self.get_parameter_points()
                return f(centers_of_segments(x))
            else:
                variables = list(map(centers_of_segments, self.get_points()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables[:2])
                else:  # only if the line is 3D (otherwise raises an error)
                    return f(*variables)
        else:
            return c*np.ones(self.nb_of_points)


class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    def __init__(self, list_x, list_y, label="", **kwargs):
        super().__init__(**kwargs)
        np = import_module('numpy')
        if len(list_x) != len(list_y):
            raise ValueError(
                "The two lists of coordinates must have the same "
                "number of elements.\n"
                "Received: len(list_x) = {} ".format(len(list_x)) +
                "and len(list_y) = {}".format(len(list_y))
            )
        self._block_lambda_functions(list_x, list_y)
        check = lambda l: [isinstance(t, Expr) and (not t.is_number) for t in l]
        if any(check(list_x) + check(list_y)) or self.params:
            if not self.params:
                raise ValueError("Some or all elements of the provided lists "
                    "are symbolic expressions, but the ``params`` dictionary "
                    "was not provided: those elements can't be evaluated.")
            self.list_x = Tuple(*list_x)
            self.list_y = Tuple(*list_y)
        else:
            self.list_x = np.array(list_x, dtype=np.float64)
            self.list_y = np.array(list_y, dtype=np.float64)

        self._expr = (self.list_x, self.list_y)
        if not any(isinstance(t, np.ndarray) for t in [self.list_x, self.list_y]):
            self._check_fs()
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.label = label
        self.rendering_kw = kwargs.get("rendering_kw", {})
        if self.use_cm and self.color_func:
            self.is_parametric = True
            if isinstance(self.color_func, Expr):
                raise TypeError(
                    "%s don't support symbolic " % self.__class__.__name__ +
                    "expression for `color_func`.")

    def __str__(self):
        return "2D list plot"

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        lx, ly = self.list_x, self.list_y

        if not self.is_interactive:
            return self._eval_color_func_and_return(lx, ly)

        np = import_module('numpy')
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)
        return self._eval_color_func_and_return(lx, ly)

    def _eval_color_func_and_return(self, *data):
        if self.use_cm and callable(self.color_func):
            return [*data, self.eval_color_func(*data)]
        return data


class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a range."""

    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self._label = str(self.expr) if label is None else label
        self._latex_label = latex(self.expr) if label is None else label
        self.ranges = [var_start_end]
        self._cast = complex
        # for complex-related data series, this determines what data to return
        # on the y-axis
        self._return = kwargs.get("return", None)
        self._post_init()

        if not self._interactive_ranges:
            # NOTE: the following check is only possible when the minimum and
            # maximum values of a plotting range are numeric
            start, end = [complex(t) for t in self.ranges[0][1:]]
            if im(start) != im(end):
                raise ValueError(
                    "%s requires the imaginary " % self.__class__.__name__ +
                    "part of the start and end values of the range "
                    "to be the same.")

        if self.adaptive and self._return:
            warnings.warn("The adaptive algorithm is unable to deal with "
                "complex numbers. Automatically switching to uniform meshing.")
            self.adaptive = False

    @property
    def nb_of_points(self):
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        self.n = v

    def __str__(self):
        def f(t):
            if isinstance(t, complex):
                if t.imag != 0:
                    return t
                return t.real
            return t
        pre = "interactive " if self.is_interactive else ""
        post = ""
        if self.is_interactive:
            post = " and parameters " + str(tuple(self.params.keys()))
        wrapper = _get_wrapper_for_expr(self._return)
        return pre + "cartesian line: %s for %s over %s" % (
            wrapper % self.expr,
            str(self.var),
            str((f(self.start), f(self.end))),
        ) + post

    def get_points(self):
        """Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates
        """
        return self._get_data_helper()

    def _adaptive_sampling(self):
        try:
            if callable(self.expr):
                f = self.expr
            else:
                f = lambdify([self.var], self.expr, self.modules)
            x, y = self._adaptive_sampling_helper(f)
        except Exception as err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not self.modules else self.modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            f = lambdify([self.var], self.expr, "sympy")
            x, y = self._adaptive_sampling_helper(f)
        return x, y

    def _adaptive_sampling_helper(self, f):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
               Luiz Henrique de Figueiredo.
        """
        np = import_module('numpy')

        x_coords = []
        y_coords = []
        def sample(p, q, depth):
            """ Samples recursively if three points are almost collinear.
            For depth < 6, points are added irrespective of whether they
            satisfy the collinearity condition or not. The maximum depth
            allowed is 12.
            """
            # Randomly sample to avoid aliasing.
            random = 0.45 + np.random.rand() * 0.1
            if self.xscale == 'log':
                xnew = 10**(np.log10(p[0]) + random * (np.log10(q[0]) -
                                                        np.log10(p[0])))
            else:
                xnew = p[0] + random * (q[0] - p[0])
            ynew = _adaptive_eval(f, xnew)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])

            # Sample to depth of 6 (whether the line is flat or not)
            # without using linspace (to avoid aliasing).
            elif depth < 6:
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif p[1] is None and q[1] is None:
                if self.xscale == 'log':
                    xarray = np.logspace(p[0], q[0], 10)
                else:
                    xarray = np.linspace(p[0], q[0], 10)
                yarray = list(map(f, xarray))
                if not all(y is None for y in yarray):
                    for i in range(len(yarray) - 1):
                        if not (yarray[i] is None and yarray[i + 1] is None):
                            sample([xarray[i], yarray[i]],
                                [xarray[i + 1], yarray[i + 1]], depth + 1)

            # Sample further if one of the end points in None (i.e. a
            # complex value) or the three points are not almost collinear.
            elif (p[1] is None or q[1] is None or new_point[1] is None
                    or not flat(p, new_point, q)):
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])

        f_start = _adaptive_eval(f, self.start.real)
        f_end = _adaptive_eval(f, self.end.real)
        x_coords.append(self.start.real)
        y_coords.append(f_start)
        sample(np.array([self.start.real, f_start]),
                np.array([self.end.real, f_end]), 0)

        return (x_coords, y_coords)

    def _uniform_sampling(self):
        np = import_module('numpy')

        x, result = self._evaluate()
        _re, _im = np.real(result), np.imag(result)
        _re = self._correct_shape(_re, x)
        _im = self._correct_shape(_im, x)
        return x, _re, _im

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        """
        np = import_module('numpy')
        if self.adaptive and (not self.only_integers):
            x, y = self._adaptive_sampling()
            return [np.array(t) for t in [x, y]]

        x, _re, _im = self._uniform_sampling()

        if self._return is None:
            # The evaluation could produce complex numbers. Set real elements
            # to NaN where there are non-zero imaginary elements
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        elif self._return == "real":
            pass
        elif self._return == "imag":
            _re = _im
        elif self._return == "abs":
            _re = np.sqrt(_re**2 + _im**2)
        elif self._return == "arg":
            _re = np.arctan2(_im, _re)
        else:
            raise ValueError("`_return` not recognized. "
                "Received: %s" % self._return)

        return x, _re


class ParametricLineBaseSeries(Line2DBaseSeries):
    is_parametric = True

    def _set_parametric_line_label(self, label):
        """Logic to set the correct label to be shown on the plot.
        If `use_cm=True` there will be a colorbar, so we show the parameter.
        If `use_cm=False`, there might be a legend, so we show the expressions.

        Parameters
        ==========
        label : str
            label passed in by the pre-processor or the user
        """
        self._label = str(self.var) if label is None else label
        self._latex_label = latex(self.var) if label is None else label
        if (self.use_cm is False) and (self._label == str(self.var)):
            self._label = str(self.expr)
            self._latex_label = latex(self.expr)
        # if the expressions is a lambda function and use_cm=False and no label
        # has been provided, then its better to do the following in order to
        # avoid surprises on the backend
        if any(callable(e) for e in self.expr) and (not self.use_cm):
            if self._label == str(self.expr):
                self._label = ""

    def get_label(self, use_latex=False, wrapper="$%s$"):
        # parametric lines returns the representation of the parameter to be
        # shown on the colorbar if `use_cm=True`, otherwise it returns the
        # representation of the expression to be placed on the legend.
        if self.use_cm:
            if str(self.var) == self._label:
                if use_latex:
                    return self._get_wrapped_label(latex(self.var), wrapper)
                return str(self.var)
            # here the user has provided a custom label
            return self._label
        if use_latex:
            if self._label != str(self.expr):
                return self._latex_label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._label

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        if self.adaptive:
            np = import_module("numpy")
            coords = self._adaptive_sampling()
            coords = [np.array(t) for t in coords]
        else:
            coords = self._uniform_sampling()

        if self.is_2Dline and self.is_polar:
            # when plot_polar is executed with polar_axis=True
            np = import_module('numpy')
            x, y, _ = coords
            r = np.sqrt(x**2 + y**2)
            t = np.arctan2(y, x)
            coords = [t, r, coords[-1]]

        if callable(self.color_func):
            coords = list(coords)
            coords[-1] = self.eval_color_func(*coords)

        return coords

    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        return [*results[1:], results[0]]

    def get_parameter_points(self):
        return self.get_data()[-1]

    def get_points(self):
        """ Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates
            y : list
                List of y-coordinates
            z : list
                List of z-coordinates, only for 3D parametric line plot.
        """
        return self._get_data_helper()[:-1]

    @property
    def nb_of_points(self):
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        self.n = v


class Parametric2DLineSeries(ParametricLineBaseSeries):
    """Representation for a line consisting of two parametric SymPy expressions
    over a range."""

    is_2Dline = True

    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr = (self.expr_x, self.expr_y)
        self.ranges = [var_start_end]
        self._cast = float
        self.use_cm = kwargs.get("use_cm", True)
        self._set_parametric_line_label(label)
        self._post_init()

    def __str__(self):
        return self._str_helper(
            "parametric cartesian line: (%s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.var),
            str((self.start, self.end))
        ))

    def _adaptive_sampling(self):
        try:
            if callable(self.expr_x) and callable(self.expr_y):
                f_x = self.expr_x
                f_y = self.expr_y
            else:
                f_x = lambdify([self.var], self.expr_x)
                f_y = lambdify([self.var], self.expr_y)
            x, y, p = self._adaptive_sampling_helper(f_x, f_y)
        except Exception as err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not self.modules else self.modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            f_x = lambdify([self.var], self.expr_x, "sympy")
            f_y = lambdify([self.var], self.expr_y, "sympy")
            x, y, p = self._adaptive_sampling_helper(f_x, f_y)
        return x, y, p

    def _adaptive_sampling_helper(self, f_x, f_y):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
            Luiz Henrique de Figueiredo.
        """
        x_coords = []
        y_coords = []
        param = []

        def sample(param_p, param_q, p, q, depth):
            """ Samples recursively if three points are almost collinear.
            For depth < 6, points are added irrespective of whether they
            satisfy the collinearity condition or not. The maximum depth
            allowed is 12.
            """
            # Randomly sample to avoid aliasing.
            np = import_module('numpy')
            random = 0.45 + np.random.rand() * 0.1
            param_new = param_p + random * (param_q - param_p)
            xnew = _adaptive_eval(f_x, param_new)
            ynew = _adaptive_eval(f_y, param_new)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

            # Sample irrespective of whether the line is flat till the
            # depth of 6. We are not using linspace to avoid aliasing.
            elif depth < 6:
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif ((p[0] is None and q[1] is None) or
                    (p[1] is None and q[1] is None)):
                param_array = np.linspace(param_p, param_q, 10)
                x_array = [_adaptive_eval(f_x, t) for t in param_array]
                y_array = [_adaptive_eval(f_y, t) for t in param_array]
                if not all(x is None and y is None
                           for x, y in zip(x_array, y_array)):
                    for i in range(len(y_array) - 1):
                        if ((x_array[i] is not None and y_array[i] is not None) or
                                (x_array[i + 1] is not None and y_array[i + 1] is not None)):
                            point_a = [x_array[i], y_array[i]]
                            point_b = [x_array[i + 1], y_array[i + 1]]
                            sample(param_array[i], param_array[i], point_a,
                                   point_b, depth + 1)

            # Sample further if one of the end points in None (i.e. a complex
            # value) or the three points are not almost collinear.
            elif (p[0] is None or p[1] is None
                    or q[1] is None or q[0] is None
                    or not flat(p, new_point, q)):
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

        f_start_x = _adaptive_eval(f_x, self.start)
        f_start_y = _adaptive_eval(f_y, self.start)
        start = [f_start_x, f_start_y]
        f_end_x = _adaptive_eval(f_x, self.end)
        f_end_y = _adaptive_eval(f_y, self.end)
        end = [f_end_x, f_end_y]
        x_coords.append(f_start_x)
        y_coords.append(f_start_y)
        param.append(self.start)
        sample(self.start, self.end, start, end, 0)

        return x_coords, y_coords, param


### 3D lines
class Line3DBaseSeries(Line2DBaseSeries):
    """A base class for 3D lines.

    Most of the stuff is derived from Line2DBaseSeries."""

    is_2Dline = False
    is_3Dline = True
    _dim = 3

    def __init__(self):
        super().__init__()


class Parametric3DLineSeries(ParametricLineBaseSeries):
    """Representation for a 3D line consisting of three parametric SymPy
    expressions and a range."""

    is_2Dline = False
    is_3Dline = True

    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end]
        self._cast = float
        self.adaptive = False
        self.use_cm = kwargs.get("use_cm", True)
        self._set_parametric_line_label(label)
        self._post_init()
        # TODO: remove this
        self._xlim = None
        self._ylim = None
        self._zlim = None

    def __str__(self):
        return self._str_helper(
            "3D parametric cartesian line: (%s, %s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var),
            str((self.start, self.end))
        ))

    def get_data(self):
        # TODO: remove this
        np = import_module("numpy")
        x, y, z, p = super().get_data()
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))
        return x, y, z, p


### Surfaces
class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""

    is_3Dsurface = True

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.use_cm = kwargs.get("use_cm", False)
        # NOTE: why should SurfaceOver2DRangeSeries support is polar?
        # After all, the same result can be achieve with
        # ParametricSurfaceSeries. For example:
        # sin(r) for (r, 0, 2 * pi) and (theta, 0, pi/2) can be parameterized
        # as (r * cos(theta), r * sin(theta), sin(t)) for (r, 0, 2 * pi) and
        # (theta, 0, pi/2).
        # Because it is faster to evaluate (important for interactive plots).
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.surface_color = kwargs.get("surface_color", None)
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        if callable(self.surface_color):
            self.color_func = self.surface_color
            self.surface_color = None

    def _set_surface_label(self, label):
        exprs = self.expr
        self._label = str(exprs) if label is None else label
        self._latex_label = latex(exprs) if label is None else label
        # if the expressions is a lambda function and no label
        # has been provided, then its better to do the following to avoid
        # surprises on the backend
        is_lambda = (callable(exprs) if not hasattr(exprs, "__iter__")
            else any(callable(e) for e in exprs))
        if is_lambda and (self._label == str(exprs)):
                self._label = ""
                self._latex_label = ""

    def get_color_array(self):
        np = import_module('numpy')
        c = self.surface_color
        if isinstance(c, Callable):
            f = np.vectorize(c)
            nargs = arity(c)
            if self.is_parametric:
                variables = list(map(centers_of_faces, self.get_parameter_meshes()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables)
            variables = list(map(centers_of_faces, self.get_meshes()))
            if nargs == 1:
                return f(variables[0])
            elif nargs == 2:
                return f(*variables[:2])
            else:
                return f(*variables)
        else:
            if isinstance(self, SurfaceOver2DRangeSeries):
                return c*np.ones(min(self.nb_of_points_x, self.nb_of_points_y))
            else:
                return c*np.ones(min(self.nb_of_points_u, self.nb_of_points_v))


class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a SymPy expression and 2D
    range."""

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.ranges = [var_start_end_x, var_start_end_y]
        self._set_surface_label(label)
        self._post_init()
        # TODO: remove this
        self._xlim = (self.start_x, self.end_x)
        self._ylim = (self.start_y, self.end_y)

    @property
    def var_x(self):
        return self.ranges[0][0]

    @property
    def var_y(self):
        return self.ranges[1][0]

    @property
    def start_x(self):
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end_x(self):
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def start_y(self):
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    def end_y(self):
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    def nb_of_points_x(self):
        return self.n[0]

    @nb_of_points_x.setter
    def nb_of_points_x(self, v):
        n = self.n
        self.n = [v, n[1:]]

    @property
    def nb_of_points_y(self):
        return self.n[1]

    @nb_of_points_y.setter
    def nb_of_points_y(self, v):
        n = self.n
        self.n = [n[0], v, n[2]]

    def __str__(self):
        series_type = "cartesian surface" if self.is_3Dsurface else "contour"
        return self._str_helper(
            series_type + ": %s for" " %s over %s and %s over %s" % (
            str(self.expr),
            str(self.var_x), str((self.start_x, self.end_x)),
            str(self.var_y), str((self.start_y, self.end_y)),
        ))

    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
        return self.get_data()

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        mesh_x : np.ndarray
            Discretized x-domain.
        mesh_y : np.ndarray
            Discretized y-domain.
        mesh_z : np.ndarray
            Results of the evaluation.
        """
        np = import_module('numpy')

        results = self._evaluate()
        # mask out complex values
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        x, y, z = results
        if self.is_polar and self.is_3Dsurface:
            r = x.copy()
            x = r * np.cos(y)
            y = r * np.sin(y)

        # TODO: remove this
        self._zlim = (np.amin(z), np.amax(z))

        return self._apply_transform(x, y, z)


class ParametricSurfaceSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of three parametric SymPy
    expressions and a range."""

    is_parametric = True

    def __init__(self, expr_x, expr_y, expr_z,
        var_start_end_u, var_start_end_v, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end_u, var_start_end_v]
        self.color_func = kwargs.get("color_func", lambda x, y, z, u, v: z)
        self._set_surface_label(label)
        self._post_init()

    @property
    def var_u(self):
        return self.ranges[0][0]

    @property
    def var_v(self):
        return self.ranges[1][0]

    @property
    def start_u(self):
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end_u(self):
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def start_v(self):
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    def end_v(self):
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    def nb_of_points_u(self):
        return self.n[0]

    @nb_of_points_u.setter
    def nb_of_points_u(self, v):
        n = self.n
        self.n = [v, n[1:]]

    @property
    def nb_of_points_v(self):
        return self.n[1]

    @nb_of_points_v.setter
    def nb_of_points_v(self, v):
        n = self.n
        self.n = [n[0], v, n[2]]

    def __str__(self):
        return self._str_helper(
            "parametric cartesian surface: (%s, %s, %s) for"
            " %s over %s and %s over %s" % (
            str(self.expr_x), str(self.expr_y), str(self.expr_z),
            str(self.var_u), str((self.start_u, self.end_u)),
            str(self.var_v), str((self.start_v, self.end_v)),
        ))

    def get_parameter_meshes(self):
        return self.get_data()[3:]

    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
        return self.get_data()[:3]

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        x : np.ndarray [n2 x n1]
            x-coordinates.
        y : np.ndarray [n2 x n1]
            y-coordinates.
        z : np.ndarray [n2 x n1]
            z-coordinates.
        mesh_u : np.ndarray [n2 x n1]
            Discretized u range.
        mesh_v : np.ndarray [n2 x n1]
            Discretized v range.
        """
        np = import_module('numpy')

        results = self._evaluate()
        # mask out complex values
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        # TODO: remove this
        x, y, z = results[2:]
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))

        return self._apply_transform(*results[2:], *results[:2])


### Contours
class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    is_3Dsurface = False
    is_contour = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))
        self.show_clabels = kwargs.get("clabels", True)

        # NOTE: contour plots are used by plot_contour, plot_vector and
        # plot_complex_vector. By implementing contour_kw we are able to
        # quickly target the contour plot.
        self.rendering_kw = kwargs.get("contour_kw",
            kwargs.get("rendering_kw", {}))


class GenericDataSeries(BaseSeries):
    """Represents generic numerical data.

    Notes
    =====
    This class serves the purpose of back-compatibility with the "markers,
    annotations, fill, rectangles" keyword arguments that represent
    user-provided numerical data. In particular, it solves the problem of
    combining together two or more plot-objects with the ``extend`` or
    ``append`` methods: user-provided numerical data is also taken into
    consideration because it is stored in this series class.

    Also note that the current implementation is far from optimal, as each
    keyword argument is stored into an attribute in the ``Plot`` class, which
    requires a hard-coded if-statement in the ``MatplotlibBackend`` class.
    The implementation suggests that it is ok to add attributes and
    if-statements to provide more and more functionalities for user-provided
    numerical data (e.g. adding horizontal lines, or vertical lines, or bar
    plots, etc). However, in doing so one would reinvent the wheel: plotting
    libraries (like Matplotlib) already implements the necessary API.

    Instead of adding more keyword arguments and attributes, users interested
    in adding custom numerical data to a plot should retrieve the figure
    created by this plotting module. For example, this code:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import Symbol, plot, cos
       x = Symbol("x")
       p = plot(cos(x), markers=[{"args": [[0, 1, 2], [0, 1, -1], "*"]}])

    Becomes:

    .. plot::
       :context: close-figs
       :include-source: True

       p = plot(cos(x), backend="matplotlib")
       fig, ax = p._backend.fig, p._backend.ax
       ax.plot([0, 1, 2], [0, 1, -1], "*")
       fig

    Which is far better in terms of readability. Also, it gives access to the
    full plotting library capabilities, without the need to reinvent the wheel.
    """
    is_generic = True

    def __init__(self, tp, *args, **kwargs):
        self.type = tp
        self.args = args
        self.rendering_kw = kwargs

    def get_data(self):
        return self.args


class ImplicitSeries(BaseSeries):
    """Representation for 2D Implicit plot."""

    is_implicit = True
    use_cm = False
    _N = 100

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self.adaptive = kwargs.get("adaptive", False)
        self.expr = expr
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self.ranges = [var_start_end_x, var_start_end_y]
        self.var_x, self.start_x, self.end_x = self.ranges[0]
        self.var_y, self.start_y, self.end_y = self.ranges[1]
        self._color = kwargs.get("color", kwargs.get("line_color", None))

        if self.is_interactive and self.adaptive:
            raise NotImplementedError("Interactive plot with `adaptive=True` "
                "is not supported.")

        # Check whether the depth is greater than 4 or less than 0.
        depth = kwargs.get("depth", 0)
        if depth > 4:
            depth = 4
        elif depth < 0:
            depth = 0
        self.depth = 4 + depth
        self._post_init()

    @property
    def expr(self):
        if self.adaptive:
            return self._adaptive_expr
        return self._non_adaptive_expr

    @expr.setter
    def expr(self, expr):
        self._block_lambda_functions(expr)
        # these are needed for adaptive evaluation
        expr, has_equality = self._has_equality(sympify(expr))
        self._adaptive_expr = expr
        self.has_equality = has_equality
        self._label = str(expr)
        self._latex_label = latex(expr)

        if isinstance(expr, (BooleanFunction, Ne)) and (not self.adaptive):
            self.adaptive = True
            msg = "contains Boolean functions. "
            if isinstance(expr, Ne):
                msg = "is an unequality. "
            warnings.warn(
                "The provided expression " + msg
                + "In order to plot the expression, the algorithm "
                + "automatically switched to an adaptive sampling."
            )

        if isinstance(expr, BooleanFunction):
            self._non_adaptive_expr = None
            self._is_equality = False
        else:
            # these are needed for uniform meshing evaluation
            expr, is_equality = self._preprocess_meshgrid_expression(expr, self.adaptive)
            self._non_adaptive_expr = expr
            self._is_equality = is_equality

    @property
    def line_color(self):
        return self._color

    @line_color.setter
    def line_color(self, v):
        self._color = v

    color = line_color

    def _has_equality(self, expr):
        # Represents whether the expression contains an Equality, GreaterThan
        # or LessThan
        has_equality = False

        def arg_expand(bool_expr):
            """Recursively expands the arguments of an Boolean Function"""
            for arg in bool_expr.args:
                if isinstance(arg, BooleanFunction):
                    arg_expand(arg)
                elif isinstance(arg, Relational):
                    arg_list.append(arg)

        arg_list = []
        if isinstance(expr, BooleanFunction):
            arg_expand(expr)
            # Check whether there is an equality in the expression provided.
            if any(isinstance(e, (Equality, GreaterThan, LessThan)) for e in arg_list):
                has_equality = True
        elif not isinstance(expr, Relational):
            expr = Equality(expr, 0)
            has_equality = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            has_equality = True

        return expr, has_equality

    def __str__(self):
        f = lambda t: float(t) if len(t.free_symbols) == 0 else t

        return self._str_helper(
            "Implicit expression: %s for %s over %s and %s over %s") % (
            str(self._adaptive_expr),
            str(self.var_x),
            str((f(self.start_x), f(self.end_x))),
            str(self.var_y),
            str((f(self.start_y), f(self.end_y))),
        )

    def get_data(self):
        """Returns numerical data.

        Returns
        =======

        If the series is evaluated with the `adaptive=True` it returns:

        interval_list : list
            List of bounding rectangular intervals to be postprocessed and
            eventually used with Matplotlib's ``fill`` command.
        dummy : str
            A string containing ``"fill"``.

        Otherwise, it returns 2D numpy arrays to be used with Matplotlib's
        ``contour`` or ``contourf`` commands:

        x_array : np.ndarray
        y_array : np.ndarray
        z_array : np.ndarray
        plot_type : str
            A string specifying which plot command to use, ``"contour"``
            or ``"contourf"``.
        """
        if self.adaptive:
            data = self._adaptive_eval()
            if data is not None:
                return data

        return self._get_meshes_grid()

    def _adaptive_eval(self):
        """
        References
        ==========

        .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
        Mathematical Formulae with Two Free Variables.

        .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
        Arithmetic. Master's thesis. University of Toronto, 1996
        """
        import sympy.plotting.intervalmath.lib_interval as li

        user_functions = {}
        printer = IntervalMathPrinter({
            'fully_qualified_modules': False, 'inline': True,
            'allow_unknown_functions': True,
            'user_functions': user_functions})

        keys = [t for t in dir(li) if ("__" not in t) and (t not in ["import_module", "interval"])]
        vals = [getattr(li, k) for k in keys]
        d = dict(zip(keys, vals))
        func = lambdify((self.var_x, self.var_y), self.expr, modules=[d], printer=printer)
        data = None

        try:
            data = self._get_raster_interval(func)
        except NameError as err:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression, as some functions are not yet implemented"
                " in the interval math module:\n\n"
                "NameError: %s\n\n" % err +
                "Proceeding with uniform meshing."
                )
            self.adaptive = False
        except TypeError:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression. Using uniform meshing.")
            self.adaptive = False

        return data

    def _get_raster_interval(self, func):
        """Uses interval math to adaptively mesh and obtain the plot"""
        np = import_module('numpy')

        k = self.depth
        interval_list = []
        sx, sy = [float(t) for t in [self.start_x, self.start_y]]
        ex, ey = [float(t) for t in [self.end_x, self.end_y]]
        # Create initial 32 divisions
        xsample = np.linspace(sx, ex, 33)
        ysample = np.linspace(sy, ey, 33)

        # Add a small jitter so that there are no false positives for equality.
        # Ex: y==x becomes True for x interval(1, 2) and y interval(1, 2)
        # which will draw a rectangle.
        jitterx = (
            (np.random.rand(len(xsample)) * 2 - 1)
            * (ex - sx)
            / 2 ** 20
        )
        jittery = (
            (np.random.rand(len(ysample)) * 2 - 1)
            * (ey - sy)
            / 2 ** 20
        )
        xsample += jitterx
        ysample += jittery

        xinter = [interval(x1, x2) for x1, x2 in zip(xsample[:-1], xsample[1:])]
        yinter = [interval(y1, y2) for y1, y2 in zip(ysample[:-1], ysample[1:])]
        interval_list = [[x, y] for x in xinter for y in yinter]
        plot_list = []

        # recursive call refinepixels which subdivides the intervals which are
        # neither True nor False according to the expression.
        def refine_pixels(interval_list):
            """Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
            temp_interval_list = []
            plot_list = []
            for intervals in interval_list:

                # Convert the array indices to x and y values
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                # The expression is valid in the interval. Change the contour
                # array values to 1.
                if func_eval[1] is False or func_eval[0] is False:
                    pass
                elif func_eval == (True, True):
                    plot_list.append([intervalx, intervaly])
                elif func_eval[1] is None or func_eval[0] is None:
                    # Subdivide
                    avgx = intervalx.mid
                    avgy = intervaly.mid
                    a = interval(intervalx.start, avgx)
                    b = interval(avgx, intervalx.end)
                    c = interval(intervaly.start, avgy)
                    d = interval(avgy, intervaly.end)
                    temp_interval_list.append([a, c])
                    temp_interval_list.append([a, d])
                    temp_interval_list.append([b, c])
                    temp_interval_list.append([b, d])
            return temp_interval_list, plot_list

        while k >= 0 and len(interval_list):
            interval_list, plot_list_temp = refine_pixels(interval_list)
            plot_list.extend(plot_list_temp)
            k = k - 1
        # Check whether the expression represents an equality
        # If it represents an equality, then none of the intervals
        # would have satisfied the expression due to floating point
        # differences. Add all the undecided values to the plot.
        if self.has_equality:
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] and func_eval[0] is not False:
                    plot_list.append([intervalx, intervaly])
        return plot_list, "fill"

    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
        np = import_module('numpy')

        xarray, yarray, z_grid = self._evaluate()
        _re, _im = np.real(z_grid), np.imag(z_grid)
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        if self._is_equality:
            return xarray, yarray, _re, 'contour'
        return xarray, yarray, _re, 'contourf'

    @staticmethod
    def _preprocess_meshgrid_expression(expr, adaptive):
        """If the expression is a Relational, rewrite it as a single
        expression.

        Returns
        =======

        expr : Expr
            The rewritten expression

        equality : Boolean
            Whether the original expression was an Equality or not.
        """
        equality = False
        if isinstance(expr, Equality):
            expr = expr.lhs - expr.rhs
            equality = True
        elif isinstance(expr, Relational):
            expr = expr.gts - expr.lts
        elif not adaptive:
            raise NotImplementedError(
                "The expression is not supported for "
                "plotting in uniform meshed plot."
            )
        return expr, equality

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        if use_latex is False:
            return self._label
        if self._label == str(self._adaptive_expr):
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label


##############################################################################
# Finding the centers of line segments or mesh faces
##############################################################################

def centers_of_segments(array):
    np = import_module('numpy')
    return np.mean(np.vstack((array[:-1], array[1:])), 0)


def centers_of_faces(array):
    np = import_module('numpy')
    return np.mean(np.dstack((array[:-1, :-1],
                             array[1:, :-1],
                             array[:-1, 1:],
                             array[:-1, :-1],
                             )), 2)


def flat(x, y, z, eps=1e-3):
    """Checks whether three points are almost collinear"""
    np = import_module('numpy')
    # Workaround plotting piecewise (#8577)
    vector_a = (x - y).astype(float)
    vector_b = (z - y).astype(float)
    dot_product = np.dot(vector_a, vector_b)
    vector_a_norm = np.linalg.norm(vector_a)
    vector_b_norm = np.linalg.norm(vector_b)
    cos_theta = dot_product / (vector_a_norm * vector_b_norm)
    return abs(cos_theta + 1) < eps


def _set_discretization_points(kwargs, pt):
    """Allow the use of the keyword arguments ``n, n1, n2`` to
    specify the number of discretization points in one and two
    directions, while keeping back-compatibility with older keyword arguments
    like, ``nb_of_points, nb_of_points_*, points``.

    Parameters
    ==========

    kwargs : dict
        Dictionary of keyword arguments passed into a plotting function.
    pt : type
        The type of the series, which indicates the kind of plot we are
        trying to create.
    """
    replace_old_keywords = {
        "nb_of_points": "n",
        "nb_of_points_x": "n1",
        "nb_of_points_y": "n2",
        "nb_of_points_u": "n1",
        "nb_of_points_v": "n2",
        "points": "n"
    }
    for k, v in replace_old_keywords.items():
        if k in kwargs.keys():
            kwargs[v] = kwargs.pop(k)

    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries,
        Parametric3DLineSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 0):
                kwargs["n1"] = kwargs["n"][0]
    elif pt in [SurfaceOver2DRangeSeries, ContourSeries,
        ParametricSurfaceSeries, ImplicitSeries]:
        if "n" in kwargs.keys():
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 1):
                kwargs["n1"] = kwargs["n"][0]
                kwargs["n2"] = kwargs["n"][1]
            else:
                kwargs["n1"] = kwargs["n2"] = kwargs["n"]
    return kwargs

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\_orm_constructors.py ===
# orm/_orm_constructors.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

import typing
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

from . import mapperlib as mapperlib
from ._typing import _O
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .interfaces import _AttributeOptions
from .properties import MappedColumn
from .properties import MappedSQLExpression
from .query import AliasOption
from .relationships import _RelationshipArgumentType
from .relationships import _RelationshipDeclared
from .relationships import _RelationshipSecondaryArgument
from .relationships import RelationshipProperty
from .session import Session
from .util import _ORMJoin
from .util import AliasedClass
from .util import AliasedInsp
from .util import LoaderCriteriaOption
from .. import sql
from .. import util
from ..exc import InvalidRequestError
from ..sql._typing import _no_kw
from ..sql.base import _NoArg
from ..sql.base import SchemaEventTarget
from ..sql.schema import _InsertSentinelColumnDefault
from ..sql.schema import SchemaConst
from ..sql.selectable import FromClause
from ..util.typing import Annotated
from ..util.typing import Literal

if TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _ORMColumnExprArgument
    from .descriptor_props import _CC
    from .descriptor_props import _CompositeAttrType
    from .interfaces import PropComparator
    from .mapper import Mapper
    from .query import Query
    from .relationships import _LazyLoadArgumentType
    from .relationships import _ORMColCollectionArgument
    from .relationships import _ORMOrderByArgument
    from .relationships import _RelationshipJoinConditionArgument
    from .relationships import ORMBackrefArgument
    from .session import _SessionBind
    from ..sql._typing import _AutoIncrementType
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _FromClauseArgument
    from ..sql._typing import _InfoType
    from ..sql._typing import _OnClauseArgument
    from ..sql._typing import _TypeEngineArgument
    from ..sql.elements import ColumnElement
    from ..sql.schema import _ServerDefaultArgument
    from ..sql.schema import _ServerOnUpdateArgument
    from ..sql.selectable import Alias
    from ..sql.selectable import Subquery


_T = typing.TypeVar("_T")


@util.deprecated(
    "1.4",
    "The :class:`.AliasOption` object is not necessary "
    "for entities to be matched up to a query that is established "
    "via :meth:`.Query.from_statement` and now does nothing.",
    enable_warnings=False,  # AliasOption itself warns
)
def contains_alias(alias: Union[Alias, Subquery]) -> AliasOption:
    r"""Return a :class:`.MapperOption` that will indicate to the
    :class:`_query.Query`
    that the main table has been aliased.

    """
    return AliasOption(alias)


def mapped_column(
    __name_pos: Optional[
        Union[str, _TypeEngineArgument[Any], SchemaEventTarget]
    ] = None,
    __type_pos: Optional[
        Union[_TypeEngineArgument[Any], SchemaEventTarget]
    ] = None,
    *args: SchemaEventTarget,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Optional[Any] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    nullable: Optional[
        Union[bool, Literal[SchemaConst.NULL_UNSPECIFIED]]
    ] = SchemaConst.NULL_UNSPECIFIED,
    primary_key: Optional[bool] = False,
    deferred: Union[_NoArg, bool] = _NoArg.NO_ARG,
    deferred_group: Optional[str] = None,
    deferred_raiseload: Optional[bool] = None,
    use_existing_column: bool = False,
    name: Optional[str] = None,
    type_: Optional[_TypeEngineArgument[Any]] = None,
    autoincrement: _AutoIncrementType = "auto",
    doc: Optional[str] = None,
    key: Optional[str] = None,
    index: Optional[bool] = None,
    unique: Optional[bool] = None,
    info: Optional[_InfoType] = None,
    onupdate: Optional[Any] = None,
    insert_default: Optional[Any] = _NoArg.NO_ARG,
    server_default: Optional[_ServerDefaultArgument] = None,
    server_onupdate: Optional[_ServerOnUpdateArgument] = None,
    active_history: bool = False,
    quote: Optional[bool] = None,
    system: bool = False,
    comment: Optional[str] = None,
    sort_order: Union[_NoArg, int] = _NoArg.NO_ARG,
    **kw: Any,
) -> MappedColumn[Any]:
    r"""declare a new ORM-mapped :class:`_schema.Column` construct
    for use within :ref:`Declarative Table <orm_declarative_table>`
    configuration.

    The :func:`_orm.mapped_column` function provides an ORM-aware and
    Python-typing-compatible construct which is used with
    :ref:`declarative <orm_declarative_mapping>` mappings to indicate an
    attribute that's mapped to a Core :class:`_schema.Column` object.  It
    provides the equivalent feature as mapping an attribute to a
    :class:`_schema.Column` object directly when using Declarative,
    specifically when using :ref:`Declarative Table <orm_declarative_table>`
    configuration.

    .. versionadded:: 2.0

    :func:`_orm.mapped_column` is normally used with explicit typing along with
    the :class:`_orm.Mapped` annotation type, where it can derive the SQL
    type and nullability for the column based on what's present within the
    :class:`_orm.Mapped` annotation.   It also may be used without annotations
    as a drop-in replacement for how :class:`_schema.Column` is used in
    Declarative mappings in SQLAlchemy 1.x style.

    For usage examples of :func:`_orm.mapped_column`, see the documentation
    at :ref:`orm_declarative_table`.

    .. seealso::

        :ref:`orm_declarative_table` - complete documentation

        :ref:`whatsnew_20_orm_declarative_typing` - migration notes for
        Declarative mappings using 1.x style mappings

    :param __name: String name to give to the :class:`_schema.Column`.  This
     is an optional, positional only argument that if present must be the
     first positional argument passed.  If omitted, the attribute name to
     which the :func:`_orm.mapped_column`  is mapped will be used as the SQL
     column name.
    :param __type: :class:`_types.TypeEngine` type or instance which will
     indicate the datatype to be associated with the :class:`_schema.Column`.
     This is an optional, positional-only argument that if present must
     immediately follow the ``__name`` parameter if present also, or otherwise
     be the first positional parameter.  If omitted, the ultimate type for
     the column may be derived either from the annotated type, or if a
     :class:`_schema.ForeignKey` is present, from the datatype of the
     referenced column.
    :param \*args: Additional positional arguments include constructs such
     as :class:`_schema.ForeignKey`, :class:`_schema.CheckConstraint`,
     and :class:`_schema.Identity`, which are passed through to the constructed
     :class:`_schema.Column`.
    :param nullable: Optional bool, whether the column should be "NULL" or
     "NOT NULL". If omitted, the nullability is derived from the type
     annotation based on whether or not ``typing.Optional`` is present.
     ``nullable`` defaults to ``True`` otherwise for non-primary key columns,
     and ``False`` for primary key columns.
    :param primary_key: optional bool, indicates the :class:`_schema.Column`
     would be part of the table's primary key or not.
    :param deferred: Optional bool - this keyword argument is consumed by the
     ORM declarative process, and is not part of the :class:`_schema.Column`
     itself; instead, it indicates that this column should be "deferred" for
     loading as though mapped by :func:`_orm.deferred`.

     .. seealso::

        :ref:`orm_queryguide_deferred_declarative`

    :param deferred_group: Implies :paramref:`_orm.mapped_column.deferred`
     to ``True``, and set the :paramref:`_orm.deferred.group` parameter.

     .. seealso::

        :ref:`orm_queryguide_deferred_group`

    :param deferred_raiseload: Implies :paramref:`_orm.mapped_column.deferred`
     to ``True``, and set the :paramref:`_orm.deferred.raiseload` parameter.

     .. seealso::

        :ref:`orm_queryguide_deferred_raiseload`

    :param use_existing_column: if True, will attempt to locate the given
     column name on an inherited superclass (typically single inheriting
     superclass), and if present, will not produce a new column, mapping
     to the superclass column as though it were omitted from this class.
     This is used for mixins that add new columns to an inherited superclass.

     .. seealso::

        :ref:`orm_inheritance_column_conflicts`

     .. versionadded:: 2.0.0b4

    :param default: Passed directly to the
     :paramref:`_schema.Column.default` parameter if the
     :paramref:`_orm.mapped_column.insert_default` parameter is not present.
     Additionally, when used with :ref:`orm_declarative_native_dataclasses`,
     indicates a default Python value that should be applied to the keyword
     constructor within the generated ``__init__()`` method.

     Note that in the case of dataclass generation when
     :paramref:`_orm.mapped_column.insert_default` is not present, this means
     the :paramref:`_orm.mapped_column.default` value is used in **two**
     places, both the ``__init__()`` method as well as the
     :paramref:`_schema.Column.default` parameter. While this behavior may
     change in a future release, for the moment this tends to "work out"; a
     default of ``None`` will mean that the :class:`_schema.Column` gets no
     default generator, whereas a default that refers to a non-``None`` Python
     or SQL expression value will be assigned up front on the object when
     ``__init__()`` is called, which is the same value that the Core
     :class:`_sql.Insert` construct would use in any case, leading to the same
     end result.

     .. note:: When using Core level column defaults that are callables to
        be interpreted by the underlying :class:`_schema.Column` in conjunction
        with :ref:`ORM-mapped dataclasses
        <orm_declarative_native_dataclasses>`, especially those that are
        :ref:`context-aware default functions <context_default_functions>`,
        **the** :paramref:`_orm.mapped_column.insert_default` **parameter must
        be used instead**.  This is necessary to disambiguate the callable from
        being interpreted as a dataclass level default.

     .. seealso::

        :ref:`defaults_default_factory_insert_default`

        :paramref:`_orm.mapped_column.insert_default`

        :paramref:`_orm.mapped_column.default_factory`

    :param insert_default: Passed directly to the
     :paramref:`_schema.Column.default` parameter; will supersede the value
     of :paramref:`_orm.mapped_column.default` when present, however
     :paramref:`_orm.mapped_column.default` will always apply to the
     constructor default for a dataclasses mapping.

     .. seealso::

        :ref:`defaults_default_factory_insert_default`

        :paramref:`_orm.mapped_column.default`

        :paramref:`_orm.mapped_column.default_factory`

    :param sort_order: An integer that indicates how this mapped column
     should be sorted compared to the others when the ORM is creating a
     :class:`_schema.Table`. Among mapped columns that have the same
     value the default ordering is used, placing first the mapped columns
     defined in the main class, then the ones in the super classes.
     Defaults to 0. The sort is ascending.

     .. versionadded:: 2.0.4

    :param active_history=False:

        When ``True``, indicates that the "previous" value for a
        scalar attribute should be loaded when replaced, if not
        already loaded. Normally, history tracking logic for
        simple non-primary-key scalar values only needs to be
        aware of the "new" value in order to perform a flush. This
        flag is available for applications that make use of
        :func:`.attributes.get_history` or :meth:`.Session.is_modified`
        which also need to know the "previous" value of the attribute.

        .. versionadded:: 2.0.10


    :param init: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__init__()``
     method as generated by the dataclass process.
    :param repr: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__repr__()``
     method as generated by the dataclass process.
    :param default_factory: Specific to
     :ref:`orm_declarative_native_dataclasses`,
     specifies a default-value generation function that will take place
     as part of the ``__init__()``
     method as generated by the dataclass process.

     .. seealso::

        :ref:`defaults_default_factory_insert_default`

        :paramref:`_orm.mapped_column.default`

        :paramref:`_orm.mapped_column.insert_default`

    :param compare: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be included in comparison operations when generating the
     ``__eq__()`` and ``__ne__()`` methods for the mapped class.

     .. versionadded:: 2.0.0b4

    :param kw_only: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be marked as keyword-only when generating the ``__init__()``.

    :param hash: Specific to
     :ref:`orm_declarative_native_dataclasses`, controls if this field
     is included when generating the ``__hash__()`` method for the mapped
     class.

     .. versionadded:: 2.0.36

    :param \**kw: All remaining keyword arguments are passed through to the
     constructor for the :class:`_schema.Column`.

    """

    return MappedColumn(
        __name_pos,
        __type_pos,
        *args,
        name=name,
        type_=type_,
        autoincrement=autoincrement,
        insert_default=insert_default,
        attribute_options=_AttributeOptions(
            init, repr, default, default_factory, compare, kw_only, hash
        ),
        doc=doc,
        key=key,
        index=index,
        unique=unique,
        info=info,
        active_history=active_history,
        nullable=nullable,
        onupdate=onupdate,
        primary_key=primary_key,
        server_default=server_default,
        server_onupdate=server_onupdate,
        use_existing_column=use_existing_column,
        quote=quote,
        comment=comment,
        system=system,
        deferred=deferred,
        deferred_group=deferred_group,
        deferred_raiseload=deferred_raiseload,
        sort_order=sort_order,
        **kw,
    )


def orm_insert_sentinel(
    name: Optional[str] = None,
    type_: Optional[_TypeEngineArgument[Any]] = None,
    *,
    default: Optional[Any] = None,
    omit_from_statements: bool = True,
) -> MappedColumn[Any]:
    """Provides a surrogate :func:`_orm.mapped_column` that generates
    a so-called :term:`sentinel` column, allowing efficient bulk
    inserts with deterministic RETURNING sorting for tables that don't
    otherwise have qualifying primary key configurations.

    Use of :func:`_orm.orm_insert_sentinel` is analogous to the use of the
    :func:`_schema.insert_sentinel` construct within a Core
    :class:`_schema.Table` construct.

    Guidelines for adding this construct to a Declarative mapped class
    are the same as that of the :func:`_schema.insert_sentinel` construct;
    the database table itself also needs to have a column with this name
    present.

    For background on how this object is used, see the section
    :ref:`engine_insertmanyvalues_sentinel_columns` as part of the
    section :ref:`engine_insertmanyvalues`.

    .. seealso::

        :func:`_schema.insert_sentinel`

        :ref:`engine_insertmanyvalues`

        :ref:`engine_insertmanyvalues_sentinel_columns`


    .. versionadded:: 2.0.10

    """

    return mapped_column(
        name=name,
        default=(
            default if default is not None else _InsertSentinelColumnDefault()
        ),
        _omit_from_statements=omit_from_statements,
        insert_sentinel=True,
        use_existing_column=True,
        nullable=True,
    )


@util.deprecated_params(
    **{
        arg: (
            "2.0",
            f"The :paramref:`_orm.column_property.{arg}` parameter is "
            "deprecated for :func:`_orm.column_property`.  This parameter "
            "applies to a writeable-attribute in a Declarative Dataclasses "
            "configuration only, and :func:`_orm.column_property` is treated "
            "as a read-only attribute in this context.",
        )
        for arg in ("init", "kw_only", "default", "default_factory")
    }
)
def column_property(
    column: _ORMColumnExprArgument[_T],
    *additional_columns: _ORMColumnExprArgument[Any],
    group: Optional[str] = None,
    deferred: bool = False,
    raiseload: bool = False,
    comparator_factory: Optional[Type[PropComparator[_T]]] = None,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Optional[Any] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    active_history: bool = False,
    expire_on_flush: bool = True,
    info: Optional[_InfoType] = None,
    doc: Optional[str] = None,
) -> MappedSQLExpression[_T]:
    r"""Provide a column-level property for use with a mapping.

    With Declarative mappings, :func:`_orm.column_property` is used to
    map read-only SQL expressions to a mapped class.

    When using Imperative mappings, :func:`_orm.column_property` also
    takes on the role of mapping table columns with additional features.
    When using fully Declarative mappings, the :func:`_orm.mapped_column`
    construct should be used for this purpose.

    With Declarative Dataclass mappings, :func:`_orm.column_property`
    is considered to be **read only**, and will not be included in the
    Dataclass ``__init__()`` constructor.

    The :func:`_orm.column_property` function returns an instance of
    :class:`.ColumnProperty`.

    .. seealso::

        :ref:`mapper_column_property_sql_expressions` - general use of
        :func:`_orm.column_property` to map SQL expressions

        :ref:`orm_imperative_table_column_options` - usage of
        :func:`_orm.column_property` with Imperative Table mappings to apply
        additional options to a plain :class:`_schema.Column` object

    :param \*cols:
        list of Column objects to be mapped.

    :param active_history=False:

        Used only for Imperative Table mappings, or legacy-style Declarative
        mappings (i.e. which have not been upgraded to
        :func:`_orm.mapped_column`), for column-based attributes that are
        expected to be writeable; use :func:`_orm.mapped_column` with
        :paramref:`_orm.mapped_column.active_history` for Declarative mappings.
        See that parameter for functional details.

    :param comparator_factory: a class which extends
        :class:`.ColumnProperty.Comparator` which provides custom SQL
        clause generation for comparison operations.

    :param group:
        a group name for this property when marked as deferred.

    :param deferred:
        when True, the column property is "deferred", meaning that
        it does not load immediately, and is instead loaded when the
        attribute is first accessed on an instance.  See also
        :func:`~sqlalchemy.orm.deferred`.

    :param doc:
        optional string that will be applied as the doc on the
        class-bound descriptor.

    :param expire_on_flush=True:
        Disable expiry on flush.   A column_property() which refers
        to a SQL expression (and not a single table-bound column)
        is considered to be a "read only" property; populating it
        has no effect on the state of data, and it can only return
        database state.   For this reason a column_property()'s value
        is expired whenever the parent object is involved in a
        flush, that is, has any kind of "dirty" state within a flush.
        Setting this parameter to ``False`` will have the effect of
        leaving any existing value present after the flush proceeds.
        Note that the :class:`.Session` with default expiration
        settings still expires
        all attributes after a :meth:`.Session.commit` call, however.

    :param info: Optional data dictionary which will be populated into the
        :attr:`.MapperProperty.info` attribute of this object.

    :param raiseload: if True, indicates the column should raise an error
        when undeferred, rather than loading the value.  This can be
        altered at query time by using the :func:`.deferred` option with
        raiseload=False.

        .. versionadded:: 1.4

        .. seealso::

            :ref:`orm_queryguide_deferred_raiseload`

    :param init: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__init__()``
     method as generated by the dataclass process.
    :param repr: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__repr__()``
     method as generated by the dataclass process.
    :param default_factory: Specific to
     :ref:`orm_declarative_native_dataclasses`,
     specifies a default-value generation function that will take place
     as part of the ``__init__()``
     method as generated by the dataclass process.

     .. seealso::

        :ref:`defaults_default_factory_insert_default`

        :paramref:`_orm.mapped_column.default`

        :paramref:`_orm.mapped_column.insert_default`

    :param compare: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be included in comparison operations when generating the
     ``__eq__()`` and ``__ne__()`` methods for the mapped class.

     .. versionadded:: 2.0.0b4

    :param kw_only: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be marked as keyword-only when generating the ``__init__()``.

    :param hash: Specific to
     :ref:`orm_declarative_native_dataclasses`, controls if this field
     is included when generating the ``__hash__()`` method for the mapped
     class.

     .. versionadded:: 2.0.36

    """
    return MappedSQLExpression(
        column,
        *additional_columns,
        attribute_options=_AttributeOptions(
            False if init is _NoArg.NO_ARG else init,
            repr,
            default,
            default_factory,
            compare,
            kw_only,
            hash,
        ),
        group=group,
        deferred=deferred,
        raiseload=raiseload,
        comparator_factory=comparator_factory,
        active_history=active_history,
        expire_on_flush=expire_on_flush,
        info=info,
        doc=doc,
        _assume_readonly_dc_attributes=True,
    )


@overload
def composite(
    _class_or_attr: _CompositeAttrType[Any],
    *attrs: _CompositeAttrType[Any],
    group: Optional[str] = None,
    deferred: bool = False,
    raiseload: bool = False,
    comparator_factory: Optional[Type[Composite.Comparator[_T]]] = None,
    active_history: bool = False,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Optional[Any] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    info: Optional[_InfoType] = None,
    doc: Optional[str] = None,
    **__kw: Any,
) -> Composite[Any]: ...


@overload
def composite(
    _class_or_attr: Type[_CC],
    *attrs: _CompositeAttrType[Any],
    group: Optional[str] = None,
    deferred: bool = False,
    raiseload: bool = False,
    comparator_factory: Optional[Type[Composite.Comparator[_T]]] = None,
    active_history: bool = False,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Optional[Any] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    info: Optional[_InfoType] = None,
    doc: Optional[str] = None,
    **__kw: Any,
) -> Composite[_CC]: ...


@overload
def composite(
    _class_or_attr: Callable[..., _CC],
    *attrs: _CompositeAttrType[Any],
    group: Optional[str] = None,
    deferred: bool = False,
    raiseload: bool = False,
    comparator_factory: Optional[Type[Composite.Comparator[_T]]] = None,
    active_history: bool = False,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Optional[Any] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    info: Optional[_InfoType] = None,
    doc: Optional[str] = None,
    **__kw: Any,
) -> Composite[_CC]: ...


def composite(
    _class_or_attr: Union[
        None, Type[_CC], Callable[..., _CC], _CompositeAttrType[Any]
    ] = None,
    *attrs: _CompositeAttrType[Any],
    group: Optional[str] = None,
    deferred: bool = False,
    raiseload: bool = False,
    comparator_factory: Optional[Type[Composite.Comparator[_T]]] = None,
    active_history: bool = False,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Optional[Any] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    info: Optional[_InfoType] = None,
    doc: Optional[str] = None,
    **__kw: Any,
) -> Composite[Any]:
    r"""Return a composite column-based property for use with a Mapper.

    See the mapping documentation section :ref:`mapper_composite` for a
    full usage example.

    The :class:`.MapperProperty` returned by :func:`.composite`
    is the :class:`.Composite`.

    :param class\_:
      The "composite type" class, or any classmethod or callable which
      will produce a new instance of the composite object given the
      column values in order.

    :param \*attrs:
      List of elements to be mapped, which may include:

      * :class:`_schema.Column` objects
      * :func:`_orm.mapped_column` constructs
      * string names of other attributes on the mapped class, which may be
        any other SQL or object-mapped attribute.  This can for
        example allow a composite that refers to a many-to-one relationship

    :param active_history=False:
      When ``True``, indicates that the "previous" value for a
      scalar attribute should be loaded when replaced, if not
      already loaded.  See the same flag on :func:`.column_property`.

    :param group:
      A group name for this property when marked as deferred.

    :param deferred:
      When True, the column property is "deferred", meaning that it does
      not load immediately, and is instead loaded when the attribute is
      first accessed on an instance.  See also
      :func:`~sqlalchemy.orm.deferred`.

    :param comparator_factory:  a class which extends
      :class:`.Composite.Comparator` which provides custom SQL
      clause generation for comparison operations.

    :param doc:
      optional string that will be applied as the doc on the
      class-bound descriptor.

    :param info: Optional data dictionary which will be populated into the
        :attr:`.MapperProperty.info` attribute of this object.

    :param init: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__init__()``
     method as generated by the dataclass process.
    :param repr: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__repr__()``
     method as generated by the dataclass process.
    :param default_factory: Specific to
     :ref:`orm_declarative_native_dataclasses`,
     specifies a default-value generation function that will take place
     as part of the ``__init__()``
     method as generated by the dataclass process.

    :param compare: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be included in comparison operations when generating the
     ``__eq__()`` and ``__ne__()`` methods for the mapped class.

     .. versionadded:: 2.0.0b4

    :param kw_only: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be marked as keyword-only when generating the ``__init__()``.

    :param hash: Specific to
     :ref:`orm_declarative_native_dataclasses`, controls if this field
     is included when generating the ``__hash__()`` method for the mapped
     class.

     .. versionadded:: 2.0.36
    """
    if __kw:
        raise _no_kw()

    return Composite(
        _class_or_attr,
        *attrs,
        attribute_options=_AttributeOptions(
            init, repr, default, default_factory, compare, kw_only, hash
        ),
        group=group,
        deferred=deferred,
        raiseload=raiseload,
        comparator_factory=comparator_factory,
        active_history=active_history,
        info=info,
        doc=doc,
    )


def with_loader_criteria(
    entity_or_base: _EntityType[Any],
    where_criteria: Union[
        _ColumnExpressionArgument[bool],
        Callable[[Any], _ColumnExpressionArgument[bool]],
    ],
    loader_only: bool = False,
    include_aliases: bool = False,
    propagate_to_loaders: bool = True,
    track_closure_variables: bool = True,
) -> LoaderCriteriaOption:
    """Add additional WHERE criteria to the load for all occurrences of
    a particular entity.

    .. versionadded:: 1.4

    The :func:`_orm.with_loader_criteria` option is intended to add
    limiting criteria to a particular kind of entity in a query,
    **globally**, meaning it will apply to the entity as it appears
    in the SELECT query as well as within any subqueries, join
    conditions, and relationship loads, including both eager and lazy
    loaders, without the need for it to be specified in any particular
    part of the query.    The rendering logic uses the same system used by
    single table inheritance to ensure a certain discriminator is applied
    to a table.

    E.g., using :term:`2.0-style` queries, we can limit the way the
    ``User.addresses`` collection is loaded, regardless of the kind
    of loading used::

        from sqlalchemy.orm import with_loader_criteria

        stmt = select(User).options(
            selectinload(User.addresses),
            with_loader_criteria(Address, Address.email_address != "foo"),
        )

    Above, the "selectinload" for ``User.addresses`` will apply the
    given filtering criteria to the WHERE clause.

    Another example, where the filtering will be applied to the
    ON clause of the join, in this example using :term:`1.x style`
    queries::

        q = (
            session.query(User)
            .outerjoin(User.addresses)
            .options(with_loader_criteria(Address, Address.email_address != "foo"))
        )

    The primary purpose of :func:`_orm.with_loader_criteria` is to use
    it in the :meth:`_orm.SessionEvents.do_orm_execute` event handler
    to ensure that all occurrences of a particular entity are filtered
    in a certain way, such as filtering for access control roles.    It
    also can be used to apply criteria to relationship loads.  In the
    example below, we can apply a certain set of rules to all queries
    emitted by a particular :class:`_orm.Session`::

        session = Session(bind=engine)


        @event.listens_for("do_orm_execute", session)
        def _add_filtering_criteria(execute_state):

            if (
                execute_state.is_select
                and not execute_state.is_column_load
                and not execute_state.is_relationship_load
            ):
                execute_state.statement = execute_state.statement.options(
                    with_loader_criteria(
                        SecurityRole,
                        lambda cls: cls.role.in_(["some_role"]),
                        include_aliases=True,
                    )
                )

    In the above example, the :meth:`_orm.SessionEvents.do_orm_execute`
    event will intercept all queries emitted using the
    :class:`_orm.Session`. For those queries which are SELECT statements
    and are not attribute or relationship loads a custom
    :func:`_orm.with_loader_criteria` option is added to the query.    The
    :func:`_orm.with_loader_criteria` option will be used in the given
    statement and will also be automatically propagated to all relationship
    loads that descend from this query.

    The criteria argument given is a ``lambda`` that accepts a ``cls``
    argument.  The given class will expand to include all mapped subclass
    and need not itself be a mapped class.

    .. tip::

       When using :func:`_orm.with_loader_criteria` option in
       conjunction with the :func:`_orm.contains_eager` loader option,
       it's important to note that :func:`_orm.with_loader_criteria` only
       affects the part of the query that determines what SQL is rendered
       in terms of the WHERE and FROM clauses. The
       :func:`_orm.contains_eager` option does not affect the rendering of
       the SELECT statement outside of the columns clause, so does not have
       any interaction with the :func:`_orm.with_loader_criteria` option.
       However, the way things "work" is that :func:`_orm.contains_eager`
       is meant to be used with a query that is already selecting from the
       additional entities in some way, where
       :func:`_orm.with_loader_criteria` can apply it's additional
       criteria.

       In the example below, assuming a mapping relationship as
       ``A -> A.bs -> B``, the given :func:`_orm.with_loader_criteria`
       option will affect the way in which the JOIN is rendered::

            stmt = (
                select(A)
                .join(A.bs)
                .options(contains_eager(A.bs), with_loader_criteria(B, B.flag == 1))
            )

       Above, the given :func:`_orm.with_loader_criteria` option will
       affect the ON clause of the JOIN that is specified by
       ``.join(A.bs)``, so is applied as expected. The
       :func:`_orm.contains_eager` option has the effect that columns from
       ``B`` are added to the columns clause:

       .. sourcecode:: sql

            SELECT
                b.id, b.a_id, b.data, b.flag,
                a.id AS id_1,
                a.data AS data_1
            FROM a JOIN b ON a.id = b.a_id AND b.flag = :flag_1


       The use of the :func:`_orm.contains_eager` option within the above
       statement has no effect on the behavior of the
       :func:`_orm.with_loader_criteria` option. If the
       :func:`_orm.contains_eager` option were omitted, the SQL would be
       the same as regards the FROM and WHERE clauses, where
       :func:`_orm.with_loader_criteria` continues to add its criteria to
       the ON clause of the JOIN. The addition of
       :func:`_orm.contains_eager` only affects the columns clause, in that
       additional columns against ``b`` are added which are then consumed
       by the ORM to produce ``B`` instances.

    .. warning:: The use of a lambda inside of the call to
      :func:`_orm.with_loader_criteria` is only invoked **once per unique
      class**. Custom functions should not be invoked within this lambda.
      See :ref:`engine_lambda_caching` for an overview of the "lambda SQL"
      feature, which is for advanced use only.

    :param entity_or_base: a mapped class, or a class that is a super
     class of a particular set of mapped classes, to which the rule
     will apply.

    :param where_criteria: a Core SQL expression that applies limiting
     criteria.   This may also be a "lambda:" or Python function that
     accepts a target class as an argument, when the given class is
     a base with many different mapped subclasses.

     .. note:: To support pickling, use a module-level Python function to
        produce the SQL expression instead of a lambda or a fixed SQL
        expression, which tend to not be picklable.

    :param include_aliases: if True, apply the rule to :func:`_orm.aliased`
     constructs as well.

    :param propagate_to_loaders: defaults to True, apply to relationship
     loaders such as lazy loaders.   This indicates that the
     option object itself including SQL expression is carried along with
     each loaded instance.  Set to ``False`` to prevent the object from
     being assigned to individual instances.


     .. seealso::

        :ref:`examples_session_orm_events` - includes examples of using
        :func:`_orm.with_loader_criteria`.

        :ref:`do_orm_execute_global_criteria` - basic example on how to
        combine :func:`_orm.with_loader_criteria` with the
        :meth:`_orm.SessionEvents.do_orm_execute` event.

    :param track_closure_variables: when False, closure variables inside
     of a lambda expression will not be used as part of
     any cache key.    This allows more complex expressions to be used
     inside of a lambda expression but requires that the lambda ensures
     it returns the identical SQL every time given a particular class.

     .. versionadded:: 1.4.0b2

    """  # noqa: E501
    return LoaderCriteriaOption(
        entity_or_base,
        where_criteria,
        loader_only,
        include_aliases,
        propagate_to_loaders,
        track_closure_variables,
    )


def relationship(
    argument: Optional[_RelationshipArgumentType[Any]] = None,
    secondary: Optional[_RelationshipSecondaryArgument] = None,
    *,
    uselist: Optional[bool] = None,
    collection_class: Optional[
        Union[Type[Collection[Any]], Callable[[], Collection[Any]]]
    ] = None,
    primaryjoin: Optional[_RelationshipJoinConditionArgument] = None,
    secondaryjoin: Optional[_RelationshipJoinConditionArgument] = None,
    back_populates: Optional[str] = None,
    order_by: _ORMOrderByArgument = False,
    backref: Optional[ORMBackrefArgument] = None,
    overlaps: Optional[str] = None,
    post_update: bool = False,
    cascade: str = "save-update, merge",
    viewonly: bool = False,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Union[_NoArg, _T] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    lazy: _LazyLoadArgumentType = "select",
    passive_deletes: Union[Literal["all"], bool] = False,
    passive_updates: bool = True,
    active_history: bool = False,
    enable_typechecks: bool = True,
    foreign_keys: Optional[_ORMColCollectionArgument] = None,
    remote_side: Optional[_ORMColCollectionArgument] = None,
    join_depth: Optional[int] = None,
    comparator_factory: Optional[
        Type[RelationshipProperty.Comparator[Any]]
    ] = None,
    single_parent: bool = False,
    innerjoin: bool = False,
    distinct_target_key: Optional[bool] = None,
    load_on_pending: bool = False,
    query_class: Optional[Type[Query[Any]]] = None,
    info: Optional[_InfoType] = None,
    omit_join: Literal[None, False] = None,
    sync_backref: Optional[bool] = None,
    **kw: Any,
) -> _RelationshipDeclared[Any]:
    """Provide a relationship between two mapped classes.

    This corresponds to a parent-child or associative table relationship.
    The constructed class is an instance of :class:`.Relationship`.

    .. seealso::

        :ref:`tutorial_orm_related_objects` - tutorial introduction
        to :func:`_orm.relationship` in the :ref:`unified_tutorial`

        :ref:`relationship_config_toplevel` - narrative documentation

    :param argument:
      This parameter refers to the class that is to be related.   It
      accepts several forms, including a direct reference to the target
      class itself, the :class:`_orm.Mapper` instance for the target class,
      a Python callable / lambda that will return a reference to the
      class or :class:`_orm.Mapper` when called, and finally a string
      name for the class, which will be resolved from the
      :class:`_orm.registry` in use in order to locate the class, e.g.::

            class SomeClass(Base):
                # ...

                related = relationship("RelatedClass")

      The :paramref:`_orm.relationship.argument` may also be omitted from the
      :func:`_orm.relationship` construct entirely, and instead placed inside
      a :class:`_orm.Mapped` annotation on the left side, which should
      include a Python collection type if the relationship is expected
      to be a collection, such as::

            class SomeClass(Base):
                # ...

                related_items: Mapped[List["RelatedItem"]] = relationship()

      Or for a many-to-one or one-to-one relationship::

            class SomeClass(Base):
                # ...

                related_item: Mapped["RelatedItem"] = relationship()

      .. seealso::

        :ref:`orm_declarative_properties` - further detail
        on relationship configuration when using Declarative.

    :param secondary:
      For a many-to-many relationship, specifies the intermediary
      table, and is typically an instance of :class:`_schema.Table`.
      In less common circumstances, the argument may also be specified
      as an :class:`_expression.Alias` construct, or even a
      :class:`_expression.Join` construct.

      :paramref:`_orm.relationship.secondary` may
      also be passed as a callable function which is evaluated at
      mapper initialization time.  When using Declarative, it may also
      be a string argument noting the name of a :class:`_schema.Table`
      that is
      present in the :class:`_schema.MetaData`
      collection associated with the
      parent-mapped :class:`_schema.Table`.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      The :paramref:`_orm.relationship.secondary` keyword argument is
      typically applied in the case where the intermediary
      :class:`_schema.Table`
      is not otherwise expressed in any direct class mapping. If the
      "secondary" table is also explicitly mapped elsewhere (e.g. as in
      :ref:`association_pattern`), one should consider applying the
      :paramref:`_orm.relationship.viewonly` flag so that this
      :func:`_orm.relationship`
      is not used for persistence operations which
      may conflict with those of the association object pattern.

      .. seealso::

          :ref:`relationships_many_to_many` - Reference example of "many
          to many".

          :ref:`self_referential_many_to_many` - Specifics on using
          many-to-many in a self-referential case.

          :ref:`declarative_many_to_many` - Additional options when using
          Declarative.

          :ref:`association_pattern` - an alternative to
          :paramref:`_orm.relationship.secondary`
          when composing association
          table relationships, allowing additional attributes to be
          specified on the association table.

          :ref:`composite_secondary_join` - a lesser-used pattern which
          in some cases can enable complex :func:`_orm.relationship` SQL
          conditions to be used.

    :param active_history=False:
      When ``True``, indicates that the "previous" value for a
      many-to-one reference should be loaded when replaced, if
      not already loaded. Normally, history tracking logic for
      simple many-to-ones only needs to be aware of the "new"
      value in order to perform a flush. This flag is available
      for applications that make use of
      :func:`.attributes.get_history` which also need to know
      the "previous" value of the attribute.

    :param backref:
      A reference to a string relationship name, or a :func:`_orm.backref`
      construct, which will be used to automatically generate a new
      :func:`_orm.relationship` on the related class, which then refers to this
      one using a bi-directional :paramref:`_orm.relationship.back_populates`
      configuration.

      In modern Python, explicit use of :func:`_orm.relationship`
      with :paramref:`_orm.relationship.back_populates` should be preferred,
      as it is more robust in terms of mapper configuration as well as
      more conceptually straightforward.  It also integrates with
      new :pep:`484` typing features introduced in SQLAlchemy 2.0 which
      is not possible with dynamically generated attributes.

      .. seealso::

        :ref:`relationships_backref` - notes on using
        :paramref:`_orm.relationship.backref`

        :ref:`tutorial_orm_related_objects` - in the :ref:`unified_tutorial`,
        presents an overview of bi-directional relationship configuration
        and behaviors using :paramref:`_orm.relationship.back_populates`

        :func:`.backref` - allows control over :func:`_orm.relationship`
        configuration when using :paramref:`_orm.relationship.backref`.


    :param back_populates:
      Indicates the name of a :func:`_orm.relationship` on the related
      class that will be synchronized with this one.   It is usually
      expected that the :func:`_orm.relationship` on the related class
      also refer to this one.  This allows objects on both sides of
      each :func:`_orm.relationship` to synchronize in-Python state
      changes and also provides directives to the :term:`unit of work`
      flush process how changes along these relationships should
      be persisted.

      .. seealso::

        :ref:`tutorial_orm_related_objects` - in the :ref:`unified_tutorial`,
        presents an overview of bi-directional relationship configuration
        and behaviors.

        :ref:`relationship_patterns` - includes many examples of
        :paramref:`_orm.relationship.back_populates`.

        :paramref:`_orm.relationship.backref` - legacy form which allows
        more succinct configuration, but does not support explicit typing

    :param overlaps:
       A string name or comma-delimited set of names of other relationships
       on either this mapper, a descendant mapper, or a target mapper with
       which this relationship may write to the same foreign keys upon
       persistence.   The only effect this has is to eliminate the
       warning that this relationship will conflict with another upon
       persistence.   This is used for such relationships that are truly
       capable of conflicting with each other on write, but the application
       will ensure that no such conflicts occur.

       .. versionadded:: 1.4

       .. seealso::

            :ref:`error_qzyx` - usage example

    :param cascade:
      A comma-separated list of cascade rules which determines how
      Session operations should be "cascaded" from parent to child.
      This defaults to ``False``, which means the default cascade
      should be used - this default cascade is ``"save-update, merge"``.

      The available cascades are ``save-update``, ``merge``,
      ``expunge``, ``delete``, ``delete-orphan``, and ``refresh-expire``.
      An additional option, ``all`` indicates shorthand for
      ``"save-update, merge, refresh-expire,
      expunge, delete"``, and is often used as in ``"all, delete-orphan"``
      to indicate that related objects should follow along with the
      parent object in all cases, and be deleted when de-associated.

      .. seealso::

        :ref:`unitofwork_cascades` - Full detail on each of the available
        cascade options.

    :param cascade_backrefs=False:
      Legacy; this flag is always False.

      .. versionchanged:: 2.0 "cascade_backrefs" functionality has been
         removed.

    :param collection_class:
      A class or callable that returns a new list-holding object. will
      be used in place of a plain list for storing elements.

      .. seealso::

        :ref:`custom_collections` - Introductory documentation and
        examples.

    :param comparator_factory:
      A class which extends :class:`.Relationship.Comparator`
      which provides custom SQL clause generation for comparison
      operations.

      .. seealso::

        :class:`.PropComparator` - some detail on redefining comparators
        at this level.

        :ref:`custom_comparators` - Brief intro to this feature.


    :param distinct_target_key=None:
      Indicate if a "subquery" eager load should apply the DISTINCT
      keyword to the innermost SELECT statement.  When left as ``None``,
      the DISTINCT keyword will be applied in those cases when the target
      columns do not comprise the full primary key of the target table.
      When set to ``True``, the DISTINCT keyword is applied to the
      innermost SELECT unconditionally.

      It may be desirable to set this flag to False when the DISTINCT is
      reducing performance of the innermost subquery beyond that of what
      duplicate innermost rows may be causing.

      .. seealso::

        :ref:`loading_toplevel` - includes an introduction to subquery
        eager loading.

    :param doc:
      Docstring which will be applied to the resulting descriptor.

    :param foreign_keys:

      A list of columns which are to be used as "foreign key"
      columns, or columns which refer to the value in a remote
      column, within the context of this :func:`_orm.relationship`
      object's :paramref:`_orm.relationship.primaryjoin` condition.
      That is, if the :paramref:`_orm.relationship.primaryjoin`
      condition of this :func:`_orm.relationship` is ``a.id ==
      b.a_id``, and the values in ``b.a_id`` are required to be
      present in ``a.id``, then the "foreign key" column of this
      :func:`_orm.relationship` is ``b.a_id``.

      In normal cases, the :paramref:`_orm.relationship.foreign_keys`
      parameter is **not required.** :func:`_orm.relationship` will
      automatically determine which columns in the
      :paramref:`_orm.relationship.primaryjoin` condition are to be
      considered "foreign key" columns based on those
      :class:`_schema.Column` objects that specify
      :class:`_schema.ForeignKey`,
      or are otherwise listed as referencing columns in a
      :class:`_schema.ForeignKeyConstraint` construct.
      :paramref:`_orm.relationship.foreign_keys` is only needed when:

        1. There is more than one way to construct a join from the local
           table to the remote table, as there are multiple foreign key
           references present.  Setting ``foreign_keys`` will limit the
           :func:`_orm.relationship`
           to consider just those columns specified
           here as "foreign".

        2. The :class:`_schema.Table` being mapped does not actually have
           :class:`_schema.ForeignKey` or
           :class:`_schema.ForeignKeyConstraint`
           constructs present, often because the table
           was reflected from a database that does not support foreign key
           reflection (MySQL MyISAM).

        3. The :paramref:`_orm.relationship.primaryjoin`
           argument is used to
           construct a non-standard join condition, which makes use of
           columns or expressions that do not normally refer to their
           "parent" column, such as a join condition expressed by a
           complex comparison using a SQL function.

      The :func:`_orm.relationship` construct will raise informative
      error messages that suggest the use of the
      :paramref:`_orm.relationship.foreign_keys` parameter when
      presented with an ambiguous condition.   In typical cases,
      if :func:`_orm.relationship` doesn't raise any exceptions, the
      :paramref:`_orm.relationship.foreign_keys` parameter is usually
      not needed.

      :paramref:`_orm.relationship.foreign_keys` may also be passed as a
      callable function which is evaluated at mapper initialization time,
      and may be passed as a Python-evaluable string when using
      Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      .. seealso::

        :ref:`relationship_foreign_keys`

        :ref:`relationship_custom_foreign`

        :func:`.foreign` - allows direct annotation of the "foreign"
        columns within a :paramref:`_orm.relationship.primaryjoin`
        condition.

    :param info: Optional data dictionary which will be populated into the
        :attr:`.MapperProperty.info` attribute of this object.

    :param innerjoin=False:
      When ``True``, joined eager loads will use an inner join to join
      against related tables instead of an outer join.  The purpose
      of this option is generally one of performance, as inner joins
      generally perform better than outer joins.

      This flag can be set to ``True`` when the relationship references an
      object via many-to-one using local foreign keys that are not
      nullable, or when the reference is one-to-one or a collection that
      is guaranteed to have one or at least one entry.

      The option supports the same "nested" and "unnested" options as
      that of :paramref:`_orm.joinedload.innerjoin`.  See that flag
      for details on nested / unnested behaviors.

      .. seealso::

        :paramref:`_orm.joinedload.innerjoin` - the option as specified by
        loader option, including detail on nesting behavior.

        :ref:`what_kind_of_loading` - Discussion of some details of
        various loader options.


    :param join_depth:
      When non-``None``, an integer value indicating how many levels
      deep "eager" loaders should join on a self-referring or cyclical
      relationship.  The number counts how many times the same Mapper
      shall be present in the loading condition along a particular join
      branch.  When left at its default of ``None``, eager loaders
      will stop chaining when they encounter a the same target mapper
      which is already higher up in the chain.  This option applies
      both to joined- and subquery- eager loaders.

      .. seealso::

        :ref:`self_referential_eager_loading` - Introductory documentation
        and examples.

    :param lazy='select': specifies
      How the related items should be loaded.  Default value is
      ``select``.  Values include:

      * ``select`` - items should be loaded lazily when the property is
        first accessed, using a separate SELECT statement, or identity map
        fetch for simple many-to-one references.

      * ``immediate`` - items should be loaded as the parents are loaded,
        using a separate SELECT statement, or identity map fetch for
        simple many-to-one references.

      * ``joined`` - items should be loaded "eagerly" in the same query as
        that of the parent, using a JOIN or LEFT OUTER JOIN.  Whether
        the join is "outer" or not is determined by the
        :paramref:`_orm.relationship.innerjoin` parameter.

      * ``subquery`` - items should be loaded "eagerly" as the parents are
        loaded, using one additional SQL statement, which issues a JOIN to
        a subquery of the original statement, for each collection
        requested.

      * ``selectin`` - items should be loaded "eagerly" as the parents
        are loaded, using one or more additional SQL statements, which
        issues a JOIN to the immediate parent object, specifying primary
        key identifiers using an IN clause.

      * ``noload`` - no loading should occur at any time.  The related
        collection will remain empty.   The ``noload`` strategy is not
        recommended for general use.  For a general use "never load"
        approach, see :ref:`write_only_relationship`

      * ``raise`` - lazy loading is disallowed; accessing
        the attribute, if its value were not already loaded via eager
        loading, will raise an :exc:`~sqlalchemy.exc.InvalidRequestError`.
        This strategy can be used when objects are to be detached from
        their attached :class:`.Session` after they are loaded.

      * ``raise_on_sql`` - lazy loading that emits SQL is disallowed;
        accessing the attribute, if its value were not already loaded via
        eager loading, will raise an
        :exc:`~sqlalchemy.exc.InvalidRequestError`, **if the lazy load
        needs to emit SQL**.  If the lazy load can pull the related value
        from the identity map or determine that it should be None, the
        value is loaded.  This strategy can be used when objects will
        remain associated with the attached :class:`.Session`, however
        additional SELECT statements should be blocked.

      * ``write_only`` - the attribute will be configured with a special
        "virtual collection" that may receive
        :meth:`_orm.WriteOnlyCollection.add` and
        :meth:`_orm.WriteOnlyCollection.remove` commands to add or remove
        individual objects, but will not under any circumstances load or
        iterate the full set of objects from the database directly. Instead,
        methods such as :meth:`_orm.WriteOnlyCollection.select`,
        :meth:`_orm.WriteOnlyCollection.insert`,
        :meth:`_orm.WriteOnlyCollection.update` and
        :meth:`_orm.WriteOnlyCollection.delete` are provided which generate SQL
        constructs that may be used to load and modify rows in bulk. Used for
        large collections that are never appropriate to load at once into
        memory.

        The ``write_only`` loader style is configured automatically when
        the :class:`_orm.WriteOnlyMapped` annotation is provided on the
        left hand side within a Declarative mapping.  See the section
        :ref:`write_only_relationship` for examples.

        .. versionadded:: 2.0

        .. seealso::

            :ref:`write_only_relationship` - in the :ref:`queryguide_toplevel`

      * ``dynamic`` - the attribute will return a pre-configured
        :class:`_query.Query` object for all read
        operations, onto which further filtering operations can be
        applied before iterating the results.

        The ``dynamic`` loader style is configured automatically when
        the :class:`_orm.DynamicMapped` annotation is provided on the
        left hand side within a Declarative mapping.  See the section
        :ref:`dynamic_relationship` for examples.

        .. legacy::  The "dynamic" lazy loader strategy is the legacy form of
           what is now the "write_only" strategy described in the section
           :ref:`write_only_relationship`.

        .. seealso::

            :ref:`dynamic_relationship` - in the :ref:`queryguide_toplevel`

            :ref:`write_only_relationship` - more generally useful approach
            for large collections that should not fully load into memory

      * True - a synonym for 'select'

      * False - a synonym for 'joined'

      * None - a synonym for 'noload'

      .. seealso::

        :ref:`orm_queryguide_relationship_loaders` - Full documentation on
        relationship loader configuration in the :ref:`queryguide_toplevel`.


    :param load_on_pending=False:
      Indicates loading behavior for transient or pending parent objects.

      When set to ``True``, causes the lazy-loader to
      issue a query for a parent object that is not persistent, meaning it
      has never been flushed.  This may take effect for a pending object
      when autoflush is disabled, or for a transient object that has been
      "attached" to a :class:`.Session` but is not part of its pending
      collection.

      The :paramref:`_orm.relationship.load_on_pending`
      flag does not improve
      behavior when the ORM is used normally - object references should be
      constructed at the object level, not at the foreign key level, so
      that they are present in an ordinary way before a flush proceeds.
      This flag is not not intended for general use.

      .. seealso::

          :meth:`.Session.enable_relationship_loading` - this method
          establishes "load on pending" behavior for the whole object, and
          also allows loading on objects that remain transient or
          detached.

    :param order_by:
      Indicates the ordering that should be applied when loading these
      items.  :paramref:`_orm.relationship.order_by`
      is expected to refer to
      one of the :class:`_schema.Column`
      objects to which the target class is
      mapped, or the attribute itself bound to the target class which
      refers to the column.

      :paramref:`_orm.relationship.order_by`
      may also be passed as a callable
      function which is evaluated at mapper initialization time, and may
      be passed as a Python-evaluable string when using Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

    :param passive_deletes=False:
       Indicates loading behavior during delete operations.

       A value of True indicates that unloaded child items should not
       be loaded during a delete operation on the parent.  Normally,
       when a parent item is deleted, all child items are loaded so
       that they can either be marked as deleted, or have their
       foreign key to the parent set to NULL.  Marking this flag as
       True usually implies an ON DELETE <CASCADE|SET NULL> rule is in
       place which will handle updating/deleting child rows on the
       database side.

       Additionally, setting the flag to the string value 'all' will
       disable the "nulling out" of the child foreign keys, when the parent
       object is deleted and there is no delete or delete-orphan cascade
       enabled.  This is typically used when a triggering or error raise
       scenario is in place on the database side.  Note that the foreign
       key attributes on in-session child objects will not be changed after
       a flush occurs so this is a very special use-case setting.
       Additionally, the "nulling out" will still occur if the child
       object is de-associated with the parent.

       .. seealso::

            :ref:`passive_deletes` - Introductory documentation
            and examples.

    :param passive_updates=True:
      Indicates the persistence behavior to take when a referenced
      primary key value changes in place, indicating that the referencing
      foreign key columns will also need their value changed.

      When True, it is assumed that ``ON UPDATE CASCADE`` is configured on
      the foreign key in the database, and that the database will
      handle propagation of an UPDATE from a source column to
      dependent rows.  When False, the SQLAlchemy
      :func:`_orm.relationship`
      construct will attempt to emit its own UPDATE statements to
      modify related targets.  However note that SQLAlchemy **cannot**
      emit an UPDATE for more than one level of cascade.  Also,
      setting this flag to False is not compatible in the case where
      the database is in fact enforcing referential integrity, unless
      those constraints are explicitly "deferred", if the target backend
      supports it.

      It is highly advised that an application which is employing
      mutable primary keys keeps ``passive_updates`` set to True,
      and instead uses the referential integrity features of the database
      itself in order to handle the change efficiently and fully.

      .. seealso::

          :ref:`passive_updates` - Introductory documentation and
          examples.

          :paramref:`.mapper.passive_updates` - a similar flag which
          takes effect for joined-table inheritance mappings.

    :param post_update:
      This indicates that the relationship should be handled by a
      second UPDATE statement after an INSERT or before a
      DELETE. This flag is used to handle saving bi-directional
      dependencies between two individual rows (i.e. each row
      references the other), where it would otherwise be impossible to
      INSERT or DELETE both rows fully since one row exists before the
      other. Use this flag when a particular mapping arrangement will
      incur two rows that are dependent on each other, such as a table
      that has a one-to-many relationship to a set of child rows, and
      also has a column that references a single child row within that
      list (i.e. both tables contain a foreign key to each other). If
      a flush operation returns an error that a "cyclical
      dependency" was detected, this is a cue that you might want to
      use :paramref:`_orm.relationship.post_update` to "break" the cycle.

      .. seealso::

          :ref:`post_update` - Introductory documentation and examples.

    :param primaryjoin:
      A SQL expression that will be used as the primary
      join of the child object against the parent object, or in a
      many-to-many relationship the join of the parent object to the
      association table. By default, this value is computed based on the
      foreign key relationships of the parent and child tables (or
      association table).

      :paramref:`_orm.relationship.primaryjoin` may also be passed as a
      callable function which is evaluated at mapper initialization time,
      and may be passed as a Python-evaluable string when using
      Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      .. seealso::

          :ref:`relationship_primaryjoin`

    :param remote_side:
      Used for self-referential relationships, indicates the column or
      list of columns that form the "remote side" of the relationship.

      :paramref:`_orm.relationship.remote_side` may also be passed as a
      callable function which is evaluated at mapper initialization time,
      and may be passed as a Python-evaluable string when using
      Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      .. seealso::

        :ref:`self_referential` - in-depth explanation of how
        :paramref:`_orm.relationship.remote_side`
        is used to configure self-referential relationships.

        :func:`.remote` - an annotation function that accomplishes the
        same purpose as :paramref:`_orm.relationship.remote_side`,
        typically
        when a custom :paramref:`_orm.relationship.primaryjoin` condition
        is used.

    :param query_class:
      A :class:`_query.Query`
      subclass that will be used internally by the
      ``AppenderQuery`` returned by a "dynamic" relationship, that
      is, a relationship that specifies ``lazy="dynamic"`` or was
      otherwise constructed using the :func:`_orm.dynamic_loader`
      function.

      .. seealso::

        :ref:`dynamic_relationship` - Introduction to "dynamic"
        relationship loaders.

    :param secondaryjoin:
      A SQL expression that will be used as the join of
      an association table to the child object. By default, this value is
      computed based on the foreign key relationships of the association
      and child tables.

      :paramref:`_orm.relationship.secondaryjoin` may also be passed as a
      callable function which is evaluated at mapper initialization time,
      and may be passed as a Python-evaluable string when using
      Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      .. seealso::

          :ref:`relationship_primaryjoin`

    :param single_parent:
      When True, installs a validator which will prevent objects
      from being associated with more than one parent at a time.
      This is used for many-to-one or many-to-many relationships that
      should be treated either as one-to-one or one-to-many.  Its usage
      is optional, except for :func:`_orm.relationship` constructs which
      are many-to-one or many-to-many and also
      specify the ``delete-orphan`` cascade option.  The
      :func:`_orm.relationship` construct itself will raise an error
      instructing when this option is required.

      .. seealso::

        :ref:`unitofwork_cascades` - includes detail on when the
        :paramref:`_orm.relationship.single_parent`
        flag may be appropriate.

    :param uselist:
      A boolean that indicates if this property should be loaded as a
      list or a scalar. In most cases, this value is determined
      automatically by :func:`_orm.relationship` at mapper configuration
      time.  When using explicit :class:`_orm.Mapped` annotations,
      :paramref:`_orm.relationship.uselist` may be derived from the
      whether or not the annotation within :class:`_orm.Mapped` contains
      a collection class.
      Otherwise, :paramref:`_orm.relationship.uselist` may be derived from
      the type and direction
      of the relationship - one to many forms a list, many to one
      forms a scalar, many to many is a list. If a scalar is desired
      where normally a list would be present, such as a bi-directional
      one-to-one relationship, use an appropriate :class:`_orm.Mapped`
      annotation or set :paramref:`_orm.relationship.uselist` to False.

      The :paramref:`_orm.relationship.uselist`
      flag is also available on an
      existing :func:`_orm.relationship`
      construct as a read-only attribute,
      which can be used to determine if this :func:`_orm.relationship`
      deals
      with collections or scalar attributes::

          >>> User.addresses.property.uselist
          True

      .. seealso::

          :ref:`relationships_one_to_one` - Introduction to the "one to
          one" relationship pattern, which is typically when an alternate
          setting for :paramref:`_orm.relationship.uselist` is involved.

    :param viewonly=False:
      When set to ``True``, the relationship is used only for loading
      objects, and not for any persistence operation.  A
      :func:`_orm.relationship` which specifies
      :paramref:`_orm.relationship.viewonly` can work
      with a wider range of SQL operations within the
      :paramref:`_orm.relationship.primaryjoin` condition, including
      operations that feature the use of a variety of comparison operators
      as well as SQL functions such as :func:`_expression.cast`.  The
      :paramref:`_orm.relationship.viewonly`
      flag is also of general use when defining any kind of
      :func:`_orm.relationship` that doesn't represent
      the full set of related objects, to prevent modifications of the
      collection from resulting in persistence operations.

      .. seealso::

        :ref:`relationship_viewonly_notes` - more details on best practices
        when using :paramref:`_orm.relationship.viewonly`.

    :param sync_backref:
      A boolean that enables the events used to synchronize the in-Python
      attributes when this relationship is target of either
      :paramref:`_orm.relationship.backref` or
      :paramref:`_orm.relationship.back_populates`.

      Defaults to ``None``, which indicates that an automatic value should
      be selected based on the value of the
      :paramref:`_orm.relationship.viewonly` flag.  When left at its
      default, changes in state will be back-populated only if neither
      sides of a relationship is viewonly.

      .. versionadded:: 1.3.17

      .. versionchanged:: 1.4 - A relationship that specifies
         :paramref:`_orm.relationship.viewonly` automatically implies
         that :paramref:`_orm.relationship.sync_backref` is ``False``.

      .. seealso::

        :paramref:`_orm.relationship.viewonly`

    :param omit_join:
      Allows manual control over the "selectin" automatic join
      optimization.  Set to ``False`` to disable the "omit join" feature
      added in SQLAlchemy 1.3; or leave as ``None`` to leave automatic
      optimization in place.

      .. note:: This flag may only be set to ``False``.   It is not
         necessary to set it to ``True`` as the "omit_join" optimization is
         automatically detected; if it is not detected, then the
         optimization is not supported.

         .. versionchanged:: 1.3.11  setting ``omit_join`` to True will now
            emit a warning as this was not the intended use of this flag.

      .. versionadded:: 1.3

    :param init: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__init__()``
     method as generated by the dataclass process.
    :param repr: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__repr__()``
     method as generated by the dataclass process.
    :param default_factory: Specific to
     :ref:`orm_declarative_native_dataclasses`,
     specifies a default-value generation function that will take place
     as part of the ``__init__()``
     method as generated by the dataclass process.
    :param compare: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be included in comparison operations when generating the
     ``__eq__()`` and ``__ne__()`` methods for the mapped class.

     .. versionadded:: 2.0.0b4

    :param kw_only: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be marked as keyword-only when generating the ``__init__()``.

    :param hash: Specific to
     :ref:`orm_declarative_native_dataclasses`, controls if this field
     is included when generating the ``__hash__()`` method for the mapped
     class.

     .. versionadded:: 2.0.36
    """

    return _RelationshipDeclared(
        argument,
        secondary=secondary,
        uselist=uselist,
        collection_class=collection_class,
        primaryjoin=primaryjoin,
        secondaryjoin=secondaryjoin,
        back_populates=back_populates,
        order_by=order_by,
        backref=backref,
        overlaps=overlaps,
        post_update=post_update,
        cascade=cascade,
        viewonly=viewonly,
        attribute_options=_AttributeOptions(
            init, repr, default, default_factory, compare, kw_only, hash
        ),
        lazy=lazy,
        passive_deletes=passive_deletes,
        passive_updates=passive_updates,
        active_history=active_history,
        enable_typechecks=enable_typechecks,
        foreign_keys=foreign_keys,
        remote_side=remote_side,
        join_depth=join_depth,
        comparator_factory=comparator_factory,
        single_parent=single_parent,
        innerjoin=innerjoin,
        distinct_target_key=distinct_target_key,
        load_on_pending=load_on_pending,
        query_class=query_class,
        info=info,
        omit_join=omit_join,
        sync_backref=sync_backref,
        **kw,
    )


def synonym(
    name: str,
    *,
    map_column: Optional[bool] = None,
    descriptor: Optional[Any] = None,
    comparator_factory: Optional[Type[PropComparator[_T]]] = None,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Union[_NoArg, _T] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    info: Optional[_InfoType] = None,
    doc: Optional[str] = None,
) -> Synonym[Any]:
    """Denote an attribute name as a synonym to a mapped property,
    in that the attribute will mirror the value and expression behavior
    of another attribute.

    e.g.::

        class MyClass(Base):
            __tablename__ = "my_table"

            id = Column(Integer, primary_key=True)
            job_status = Column(String(50))

            status = synonym("job_status")

    :param name: the name of the existing mapped property.  This
      can refer to the string name ORM-mapped attribute
      configured on the class, including column-bound attributes
      and relationships.

    :param descriptor: a Python :term:`descriptor` that will be used
      as a getter (and potentially a setter) when this attribute is
      accessed at the instance level.

    :param map_column: **For classical mappings and mappings against
      an existing Table object only**.  if ``True``, the :func:`.synonym`
      construct will locate the :class:`_schema.Column`
      object upon the mapped
      table that would normally be associated with the attribute name of
      this synonym, and produce a new :class:`.ColumnProperty` that instead
      maps this :class:`_schema.Column`
      to the alternate name given as the "name"
      argument of the synonym; in this way, the usual step of redefining
      the mapping of the :class:`_schema.Column`
      to be under a different name is
      unnecessary. This is usually intended to be used when a
      :class:`_schema.Column`
      is to be replaced with an attribute that also uses a
      descriptor, that is, in conjunction with the
      :paramref:`.synonym.descriptor` parameter::

        my_table = Table(
            "my_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("job_status", String(50)),
        )


        class MyClass:
            @property
            def _job_status_descriptor(self):
                return "Status: %s" % self._job_status


        mapper(
            MyClass,
            my_table,
            properties={
                "job_status": synonym(
                    "_job_status",
                    map_column=True,
                    descriptor=MyClass._job_status_descriptor,
                )
            },
        )

      Above, the attribute named ``_job_status`` is automatically
      mapped to the ``job_status`` column::

        >>> j1 = MyClass()
        >>> j1._job_status = "employed"
        >>> j1.job_status
        Status: employed

      When using Declarative, in order to provide a descriptor in
      conjunction with a synonym, use the
      :func:`sqlalchemy.ext.declarative.synonym_for` helper.  However,
      note that the :ref:`hybrid properties <mapper_hybrids>` feature
      should usually be preferred, particularly when redefining attribute
      behavior.

    :param info: Optional data dictionary which will be populated into the
        :attr:`.InspectionAttr.info` attribute of this object.

    :param comparator_factory: A subclass of :class:`.PropComparator`
      that will provide custom comparison behavior at the SQL expression
      level.

      .. note::

        For the use case of providing an attribute which redefines both
        Python-level and SQL-expression level behavior of an attribute,
        please refer to the Hybrid attribute introduced at
        :ref:`mapper_hybrids` for a more effective technique.

    .. seealso::

        :ref:`synonyms` - Overview of synonyms

        :func:`.synonym_for` - a helper oriented towards Declarative

        :ref:`mapper_hybrids` - The Hybrid Attribute extension provides an
        updated approach to augmenting attribute behavior more flexibly
        than can be achieved with synonyms.

    """
    return Synonym(
        name,
        map_column=map_column,
        descriptor=descriptor,
        comparator_factory=comparator_factory,
        attribute_options=_AttributeOptions(
            init, repr, default, default_factory, compare, kw_only, hash
        ),
        doc=doc,
        info=info,
    )


def create_session(
    bind: Optional[_SessionBind] = None, **kwargs: Any
) -> Session:
    r"""Create a new :class:`.Session`
    with no automation enabled by default.

    This function is used primarily for testing.   The usual
    route to :class:`.Session` creation is via its constructor
    or the :func:`.sessionmaker` function.

    :param bind: optional, a single Connectable to use for all
      database access in the created
      :class:`~sqlalchemy.orm.session.Session`.

    :param \*\*kwargs: optional, passed through to the
      :class:`.Session` constructor.

    :returns: an :class:`~sqlalchemy.orm.session.Session` instance

    The defaults of create_session() are the opposite of that of
    :func:`sessionmaker`; ``autoflush`` and ``expire_on_commit`` are
    False.

    Usage::

      >>> from sqlalchemy.orm import create_session
      >>> session = create_session()

    It is recommended to use :func:`sessionmaker` instead of
    create_session().

    """

    kwargs.setdefault("autoflush", False)
    kwargs.setdefault("expire_on_commit", False)
    return Session(bind=bind, **kwargs)


def _mapper_fn(*arg: Any, **kw: Any) -> NoReturn:
    """Placeholder for the now-removed ``mapper()`` function.

    Classical mappings should be performed using the
    :meth:`_orm.registry.map_imperatively` method.

    This symbol remains in SQLAlchemy 2.0 to suit the deprecated use case
    of using the ``mapper()`` function as a target for ORM event listeners,
    which failed to be marked as deprecated in the 1.4 series.

    Global ORM mapper listeners should instead use the :class:`_orm.Mapper`
    class as the target.

    .. versionchanged:: 2.0  The ``mapper()`` function was removed; the
       symbol remains temporarily as a placeholder for the event listening
       use case.

    """
    raise InvalidRequestError(
        "The 'sqlalchemy.orm.mapper()' function is removed as of "
        "SQLAlchemy 2.0.  Use the "
        "'sqlalchemy.orm.registry.map_imperatively()` "
        "method of the ``sqlalchemy.orm.registry`` class to perform "
        "classical mapping."
    )


def dynamic_loader(
    argument: Optional[_RelationshipArgumentType[Any]] = None, **kw: Any
) -> RelationshipProperty[Any]:
    """Construct a dynamically-loading mapper property.

    This is essentially the same as
    using the ``lazy='dynamic'`` argument with :func:`relationship`::

        dynamic_loader(SomeClass)

        # is the same as

        relationship(SomeClass, lazy="dynamic")

    See the section :ref:`dynamic_relationship` for more details
    on dynamic loading.

    """
    kw["lazy"] = "dynamic"
    return relationship(argument, **kw)


def backref(name: str, **kwargs: Any) -> ORMBackrefArgument:
    """When using the :paramref:`_orm.relationship.backref` parameter,
    provides specific parameters to be used when the new
    :func:`_orm.relationship` is generated.

    E.g.::

        "items": relationship(SomeItem, backref=backref("parent", lazy="subquery"))

    The :paramref:`_orm.relationship.backref` parameter is generally
    considered to be legacy; for modern applications, using
    explicit :func:`_orm.relationship` constructs linked together using
    the :paramref:`_orm.relationship.back_populates` parameter should be
    preferred.

    .. seealso::

        :ref:`relationships_backref` - background on backrefs

    """  # noqa: E501

    return (name, kwargs)


def deferred(
    column: _ORMColumnExprArgument[_T],
    *additional_columns: _ORMColumnExprArgument[Any],
    group: Optional[str] = None,
    raiseload: bool = False,
    comparator_factory: Optional[Type[PropComparator[_T]]] = None,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Optional[Any] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
    active_history: bool = False,
    expire_on_flush: bool = True,
    info: Optional[_InfoType] = None,
    doc: Optional[str] = None,
) -> MappedSQLExpression[_T]:
    r"""Indicate a column-based mapped attribute that by default will
    not load unless accessed.

    When using :func:`_orm.mapped_column`, the same functionality as
    that of :func:`_orm.deferred` construct is provided by using the
    :paramref:`_orm.mapped_column.deferred` parameter.

    :param \*columns: columns to be mapped.  This is typically a single
     :class:`_schema.Column` object,
     however a collection is supported in order
     to support multiple columns mapped under the same attribute.

    :param raiseload: boolean, if True, indicates an exception should be raised
     if the load operation is to take place.

     .. versionadded:: 1.4


    Additional arguments are the same as that of :func:`_orm.column_property`.

    .. seealso::

        :ref:`orm_queryguide_deferred_imperative`

    """
    return MappedSQLExpression(
        column,
        *additional_columns,
        attribute_options=_AttributeOptions(
            init, repr, default, default_factory, compare, kw_only, hash
        ),
        group=group,
        deferred=True,
        raiseload=raiseload,
        comparator_factory=comparator_factory,
        active_history=active_history,
        expire_on_flush=expire_on_flush,
        info=info,
        doc=doc,
    )


def query_expression(
    default_expr: _ORMColumnExprArgument[_T] = sql.null(),
    *,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    expire_on_flush: bool = True,
    info: Optional[_InfoType] = None,
    doc: Optional[str] = None,
) -> MappedSQLExpression[_T]:
    """Indicate an attribute that populates from a query-time SQL expression.

    :param default_expr: Optional SQL expression object that will be used in
        all cases if not assigned later with :func:`_orm.with_expression`.

    .. versionadded:: 1.2

    .. seealso::

        :ref:`orm_queryguide_with_expression` - background and usage examples

    """
    prop = MappedSQLExpression(
        default_expr,
        attribute_options=_AttributeOptions(
            False,
            repr,
            _NoArg.NO_ARG,
            _NoArg.NO_ARG,
            compare,
            _NoArg.NO_ARG,
            _NoArg.NO_ARG,
        ),
        expire_on_flush=expire_on_flush,
        info=info,
        doc=doc,
        _assume_readonly_dc_attributes=True,
    )

    prop.strategy_key = (("query_expression", True),)
    return prop


def clear_mappers() -> None:
    """Remove all mappers from all classes.

    .. versionchanged:: 1.4  This function now locates all
       :class:`_orm.registry` objects and calls upon the
       :meth:`_orm.registry.dispose` method of each.

    This function removes all instrumentation from classes and disposes
    of their associated mappers.  Once called, the classes are unmapped
    and can be later re-mapped with new mappers.

    :func:`.clear_mappers` is *not* for normal use, as there is literally no
    valid usage for it outside of very specific testing scenarios. Normally,
    mappers are permanent structural components of user-defined classes, and
    are never discarded independently of their class.  If a mapped class
    itself is garbage collected, its mapper is automatically disposed of as
    well. As such, :func:`.clear_mappers` is only for usage in test suites
    that re-use the same classes with different mappings, which is itself an
    extremely rare use case - the only such use case is in fact SQLAlchemy's
    own test suite, and possibly the test suites of other ORM extension
    libraries which intend to test various combinations of mapper construction
    upon a fixed set of classes.

    """

    mapperlib._dispose_registries(mapperlib._all_registries(), False)


# I would really like a way to get the Type[] here that shows up
# in a different way in typing tools, however there is no current method
# that is accepted by mypy (subclass of Type[_O] works in pylance, rejected
# by mypy).
AliasedType = Annotated[Type[_O], "aliased"]


@overload
def aliased(
    element: Type[_O],
    alias: Optional[FromClause] = None,
    name: Optional[str] = None,
    flat: bool = False,
    adapt_on_names: bool = False,
) -> AliasedType[_O]: ...


@overload
def aliased(
    element: Union[AliasedClass[_O], Mapper[_O], AliasedInsp[_O]],
    alias: Optional[FromClause] = None,
    name: Optional[str] = None,
    flat: bool = False,
    adapt_on_names: bool = False,
) -> AliasedClass[_O]: ...


@overload
def aliased(
    element: FromClause,
    alias: None = None,
    name: Optional[str] = None,
    flat: bool = False,
    adapt_on_names: bool = False,
) -> FromClause: ...


def aliased(
    element: Union[_EntityType[_O], FromClause],
    alias: Optional[FromClause] = None,
    name: Optional[str] = None,
    flat: bool = False,
    adapt_on_names: bool = False,
) -> Union[AliasedClass[_O], FromClause, AliasedType[_O]]:
    """Produce an alias of the given element, usually an :class:`.AliasedClass`
    instance.

    E.g.::

        my_alias = aliased(MyClass)

        stmt = select(MyClass, my_alias).filter(MyClass.id > my_alias.id)
        result = session.execute(stmt)

    The :func:`.aliased` function is used to create an ad-hoc mapping of a
    mapped class to a new selectable.  By default, a selectable is generated
    from the normally mapped selectable (typically a :class:`_schema.Table`
    ) using the
    :meth:`_expression.FromClause.alias` method. However, :func:`.aliased`
    can also be
    used to link the class to a new :func:`_expression.select` statement.
    Also, the :func:`.with_polymorphic` function is a variant of
    :func:`.aliased` that is intended to specify a so-called "polymorphic
    selectable", that corresponds to the union of several joined-inheritance
    subclasses at once.

    For convenience, the :func:`.aliased` function also accepts plain
    :class:`_expression.FromClause` constructs, such as a
    :class:`_schema.Table` or
    :func:`_expression.select` construct.   In those cases, the
    :meth:`_expression.FromClause.alias`
    method is called on the object and the new
    :class:`_expression.Alias` object returned.  The returned
    :class:`_expression.Alias` is not
    ORM-mapped in this case.

    .. seealso::

        :ref:`tutorial_orm_entity_aliases` - in the :ref:`unified_tutorial`

        :ref:`orm_queryguide_orm_aliases` - in the :ref:`queryguide_toplevel`

    :param element: element to be aliased.  Is normally a mapped class,
     but for convenience can also be a :class:`_expression.FromClause`
     element.

    :param alias: Optional selectable unit to map the element to.  This is
     usually used to link the object to a subquery, and should be an aliased
     select construct as one would produce from the
     :meth:`_query.Query.subquery` method or
     the :meth:`_expression.Select.subquery` or
     :meth:`_expression.Select.alias` methods of the :func:`_expression.select`
     construct.

    :param name: optional string name to use for the alias, if not specified
     by the ``alias`` parameter.  The name, among other things, forms the
     attribute name that will be accessible via tuples returned by a
     :class:`_query.Query` object.  Not supported when creating aliases
     of :class:`_sql.Join` objects.

    :param flat: Boolean, will be passed through to the
     :meth:`_expression.FromClause.alias` call so that aliases of
     :class:`_expression.Join` objects will alias the individual tables
     inside the join, rather than creating a subquery.  This is generally
     supported by all modern databases with regards to right-nested joins
     and generally produces more efficient queries.

     When :paramref:`_orm.aliased.flat` is combined with
     :paramref:`_orm.aliased.name`, the resulting joins will alias individual
     tables using a naming scheme similar to ``<prefix>_<tablename>``.  This
     naming scheme is for visibility / debugging purposes only and the
     specific scheme is subject to change without notice.

     .. versionadded:: 2.0.32 added support for combining
        :paramref:`_orm.aliased.name` with :paramref:`_orm.aliased.flat`.
        Previously, this would raise ``NotImplementedError``.

    :param adapt_on_names: if True, more liberal "matching" will be used when
     mapping the mapped columns of the ORM entity to those of the
     given selectable - a name-based match will be performed if the
     given selectable doesn't otherwise have a column that corresponds
     to one on the entity.  The use case for this is when associating
     an entity with some derived selectable such as one that uses
     aggregate functions::

        class UnitPrice(Base):
            __tablename__ = "unit_price"
            ...
            unit_id = Column(Integer)
            price = Column(Numeric)


        aggregated_unit_price = (
            Session.query(func.sum(UnitPrice.price).label("price"))
            .group_by(UnitPrice.unit_id)
            .subquery()
        )

        aggregated_unit_price = aliased(
            UnitPrice, alias=aggregated_unit_price, adapt_on_names=True
        )

     Above, functions on ``aggregated_unit_price`` which refer to
     ``.price`` will return the
     ``func.sum(UnitPrice.price).label('price')`` column, as it is
     matched on the name "price".  Ordinarily, the "price" function
     wouldn't have any "column correspondence" to the actual
     ``UnitPrice.price`` column as it is not a proxy of the original.

    """
    return AliasedInsp._alias_factory(
        element,
        alias=alias,
        name=name,
        flat=flat,
        adapt_on_names=adapt_on_names,
    )


def with_polymorphic(
    base: Union[Type[_O], Mapper[_O]],
    classes: Union[Literal["*"], Iterable[Type[Any]]],
    selectable: Union[Literal[False, None], FromClause] = False,
    flat: bool = False,
    polymorphic_on: Optional[ColumnElement[Any]] = None,
    aliased: bool = False,
    innerjoin: bool = False,
    adapt_on_names: bool = False,
    name: Optional[str] = None,
    _use_mapper_path: bool = False,
) -> AliasedClass[_O]:
    """Produce an :class:`.AliasedClass` construct which specifies
    columns for descendant mappers of the given base.

    Using this method will ensure that each descendant mapper's
    tables are included in the FROM clause, and will allow filter()
    criterion to be used against those tables.  The resulting
    instances will also have those columns already loaded so that
    no "post fetch" of those columns will be required.

    .. seealso::

        :ref:`with_polymorphic` - full discussion of
        :func:`_orm.with_polymorphic`.

    :param base: Base class to be aliased.

    :param classes: a single class or mapper, or list of
        class/mappers, which inherit from the base class.
        Alternatively, it may also be the string ``'*'``, in which case
        all descending mapped classes will be added to the FROM clause.

    :param aliased: when True, the selectable will be aliased.   For a
        JOIN, this means the JOIN will be SELECTed from inside of a subquery
        unless the :paramref:`_orm.with_polymorphic.flat` flag is set to
        True, which is recommended for simpler use cases.

    :param flat: Boolean, will be passed through to the
     :meth:`_expression.FromClause.alias` call so that aliases of
     :class:`_expression.Join` objects will alias the individual tables
     inside the join, rather than creating a subquery.  This is generally
     supported by all modern databases with regards to right-nested joins
     and generally produces more efficient queries.  Setting this flag is
     recommended as long as the resulting SQL is functional.

    :param selectable: a table or subquery that will
        be used in place of the generated FROM clause. This argument is
        required if any of the desired classes use concrete table
        inheritance, since SQLAlchemy currently cannot generate UNIONs
        among tables automatically. If used, the ``selectable`` argument
        must represent the full set of tables and columns mapped by every
        mapped class. Otherwise, the unaccounted mapped columns will
        result in their table being appended directly to the FROM clause
        which will usually lead to incorrect results.

        When left at its default value of ``False``, the polymorphic
        selectable assigned to the base mapper is used for selecting rows.
        However, it may also be passed as ``None``, which will bypass the
        configured polymorphic selectable and instead construct an ad-hoc
        selectable for the target classes given; for joined table inheritance
        this will be a join that includes all target mappers and their
        subclasses.

    :param polymorphic_on: a column to be used as the "discriminator"
        column for the given selectable. If not given, the polymorphic_on
        attribute of the base classes' mapper will be used, if any. This
        is useful for mappings that don't have polymorphic loading
        behavior by default.

    :param innerjoin: if True, an INNER JOIN will be used.  This should
       only be specified if querying for one specific subtype only

    :param adapt_on_names: Passes through the
      :paramref:`_orm.aliased.adapt_on_names`
      parameter to the aliased object.  This may be useful in situations where
      the given selectable is not directly related to the existing mapped
      selectable.

      .. versionadded:: 1.4.33

    :param name: Name given to the generated :class:`.AliasedClass`.

      .. versionadded:: 2.0.31

    """
    return AliasedInsp._with_polymorphic_factory(
        base,
        classes,
        selectable=selectable,
        flat=flat,
        polymorphic_on=polymorphic_on,
        adapt_on_names=adapt_on_names,
        aliased=aliased,
        innerjoin=innerjoin,
        name=name,
        _use_mapper_path=_use_mapper_path,
    )


def join(
    left: _FromClauseArgument,
    right: _FromClauseArgument,
    onclause: Optional[_OnClauseArgument] = None,
    isouter: bool = False,
    full: bool = False,
) -> _ORMJoin:
    r"""Produce an inner join between left and right clauses.

    :func:`_orm.join` is an extension to the core join interface
    provided by :func:`_expression.join()`, where the
    left and right selectable may be not only core selectable
    objects such as :class:`_schema.Table`, but also mapped classes or
    :class:`.AliasedClass` instances.   The "on" clause can
    be a SQL expression or an ORM mapped attribute
    referencing a configured :func:`_orm.relationship`.

    :func:`_orm.join` is not commonly needed in modern usage,
    as its functionality is encapsulated within that of the
    :meth:`_sql.Select.join` and :meth:`_query.Query.join`
    methods. which feature a
    significant amount of automation beyond :func:`_orm.join`
    by itself.  Explicit use of :func:`_orm.join`
    with ORM-enabled SELECT statements involves use of the
    :meth:`_sql.Select.select_from` method, as in::

        from sqlalchemy.orm import join

        stmt = (
            select(User)
            .select_from(join(User, Address, User.addresses))
            .filter(Address.email_address == "foo@bar.com")
        )

    In modern SQLAlchemy the above join can be written more
    succinctly as::

        stmt = (
            select(User)
            .join(User.addresses)
            .filter(Address.email_address == "foo@bar.com")
        )

    .. warning:: using :func:`_orm.join` directly may not work properly
       with modern ORM options such as :func:`_orm.with_loader_criteria`.
       It is strongly recommended to use the idiomatic join patterns
       provided by methods such as :meth:`.Select.join` and
       :meth:`.Select.join_from` when creating ORM joins.

    .. seealso::

        :ref:`orm_queryguide_joins` - in the :ref:`queryguide_toplevel` for
        background on idiomatic ORM join patterns

    """
    return _ORMJoin(left, right, onclause, isouter, full)


def outerjoin(
    left: _FromClauseArgument,
    right: _FromClauseArgument,
    onclause: Optional[_OnClauseArgument] = None,
    full: bool = False,
) -> _ORMJoin:
    """Produce a left outer join between left and right clauses.

    This is the "outer join" version of the :func:`_orm.join` function,
    featuring the same behavior except that an OUTER JOIN is generated.
    See that function's documentation for other usage details.

    """
    return _ORMJoin(left, right, onclause, True, full)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\datetimelike.py ===
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from functools import wraps
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Union,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import (
    algos,
    lib,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
    BaseOffset,
    IncompatibleFrequency,
    NaT,
    NaTType,
    Period,
    Resolution,
    Tick,
    Timedelta,
    Timestamp,
    add_overflowsafe,
    astype_overflowsafe,
    get_unit_from_dtype,
    iNaT,
    ints_to_pydatetime,
    ints_to_pytimedelta,
    periods_per_day,
    to_offset,
)
from pandas._libs.tslibs.fields import (
    RoundTo,
    round_nsint64,
)
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import (
    ArrayLike,
    AxisInt,
    DatetimeLikeScalar,
    Dtype,
    DtypeObj,
    F,
    InterpolateOptions,
    NpDtype,
    PositionalIndexer2D,
    PositionalIndexerTuple,
    ScalarIndexer,
    Self,
    SequenceIndexer,
    TimeAmbiguous,
    TimeNonexistent,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    AbstractMethodError,
    InvalidComparison,
    PerformanceWarning,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_all_strings,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCCategorical,
    ABCMultiIndex,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    algorithms,
    missing,
    nanops,
    ops,
)
from pandas.core.algorithms import (
    isin,
    map_array,
    unique1d,
)
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,
    ravel_compat,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    check_setitem_lengths,
)
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import (
    invalid_comparison,
    make_invalid_op,
)

from pandas.tseries import frequencies

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

    from pandas import Index
    from pandas.core.arrays import (
        DatetimeArray,
        PeriodArray,
        TimedeltaArray,
    )

DTScalarOrNaT = Union[DatetimeLikeScalar, NaTType]


def _make_unpacked_invalid_op(op_name: str):
    op = make_invalid_op(op_name)
    return unpack_zerodim_and_defer(op_name)(op)


def _period_dispatch(meth: F) -> F:
    """
    For PeriodArray methods, dispatch to DatetimeArray and re-wrap the results
    in PeriodArray.  We cannot use ._ndarray directly for the affected
    methods because the i8 data has different semantics on NaT values.
    """

    @wraps(meth)
    def new_meth(self, *args, **kwargs):
        if not isinstance(self.dtype, PeriodDtype):
            return meth(self, *args, **kwargs)

        arr = self.view("M8[ns]")
        result = meth(arr, *args, **kwargs)
        if result is NaT:
            return NaT
        elif isinstance(result, Timestamp):
            return self._box_func(result._value)

        res_i8 = result.view("i8")
        return self._from_backing_data(res_i8)

    return cast(F, new_meth)


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class DatetimeLikeArrayMixin(  # type: ignore[misc]
    OpsMixin, NDArrayBackedExtensionArray
):
    """
    Shared Base/Mixin class for DatetimeArray, TimedeltaArray, PeriodArray

    Assumes that __new__/__init__ defines:
        _ndarray

    and that inheriting subclass implements:
        freq
    """

    # _infer_matches -> which infer_dtype strings are close enough to our own
    _infer_matches: tuple[str, ...]
    _is_recognized_dtype: Callable[[DtypeObj], bool]
    _recognized_scalars: tuple[type, ...]
    _ndarray: np.ndarray
    freq: BaseOffset | None

    @cache_readonly
    def _can_hold_na(self) -> bool:
        return True

    def __init__(
        self, data, dtype: Dtype | None = None, freq=None, copy: bool = False
    ) -> None:
        raise AbstractMethodError(self)

    @property
    def _scalar_type(self) -> type[DatetimeLikeScalar]:
        """
        The scalar associated with this datelike

        * PeriodArray : Period
        * DatetimeArray : Timestamp
        * TimedeltaArray : Timedelta
        """
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value: str) -> DTScalarOrNaT:
        """
        Construct a scalar type from a string.

        Parameters
        ----------
        value : str

        Returns
        -------
        Period, Timestamp, or Timedelta, or NaT
            Whatever the type of ``self._scalar_type`` is.

        Notes
        -----
        This should call ``self._check_compatible_with`` before
        unboxing the result.
        """
        raise AbstractMethodError(self)

    def _unbox_scalar(
        self, value: DTScalarOrNaT
    ) -> np.int64 | np.datetime64 | np.timedelta64:
        """
        Unbox the integer value of a scalar `value`.

        Parameters
        ----------
        value : Period, Timestamp, Timedelta, or NaT
            Depending on subclass.

        Returns
        -------
        int

        Examples
        --------
        >>> arr = pd.array(np.array(['1970-01-01'], 'datetime64[ns]'))
        >>> arr._unbox_scalar(arr[0])
        numpy.datetime64('1970-01-01T00:00:00.000000000')
        """
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other: DTScalarOrNaT) -> None:
        """
        Verify that `self` and `other` are compatible.

        * DatetimeArray verifies that the timezones (if any) match
        * PeriodArray verifies that the freq matches
        * Timedelta has no verification

        In each case, NaT is considered compatible.

        Parameters
        ----------
        other

        Raises
        ------
        Exception
        """
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------

    def _box_func(self, x):
        """
        box function to get object from internal representation
        """
        raise AbstractMethodError(self)

    def _box_values(self, values) -> np.ndarray:
        """
        apply box func to passed values
        """
        return lib.map_infer(values, self._box_func, convert=False)

    def __iter__(self) -> Iterator:
        if self.ndim > 1:
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        """
        Integer representation of the values.

        Returns
        -------
        ndarray
            An ndarray with int64 dtype.
        """
        # do not cache or you'll create a memory leak
        return self._ndarray.view("i8")

    # ----------------------------------------------------------------
    # Rendering Methods

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None
    ) -> npt.NDArray[np.object_]:
        """
        Helper method for astype when converting to strings.

        Returns
        -------
        ndarray[str]
        """
        raise AbstractMethodError(self)

    def _formatter(self, boxed: bool = False):
        # TODO: Remove Datetime & DatetimeTZ formatters.
        return "'{}'".format

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # used for Timedelta/DatetimeArray, overwritten by PeriodArray
        if is_object_dtype(dtype):
            if copy is False:
                warnings.warn(
                    "Starting with NumPy 2.0, the behavior of the 'copy' keyword has "
                    "changed and passing 'copy=False' raises an error when returning "
                    "a zero-copy NumPy array is not possible. pandas will follow this "
                    "behavior starting with pandas 3.0.\nThis conversion to NumPy "
                    "requires a copy, but 'copy=False' was passed. Consider using "
                    "'np.asarray(..)' instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

            return np.array(list(self), dtype=object)

        if copy is True:
            return np.array(self._ndarray, dtype=dtype)
        return self._ndarray

    @overload
    def __getitem__(self, item: ScalarIndexer) -> DTScalarOrNaT:
        ...

    @overload
    def __getitem__(
        self,
        item: SequenceIndexer | PositionalIndexerTuple,
    ) -> Self:
        ...

    def __getitem__(self, key: PositionalIndexer2D) -> Self | DTScalarOrNaT:
        """
        This getitem defers to the underlying array, which by-definition can
        only handle list-likes, slices, and integer scalars
        """
        # Use cast as we know we will get back a DatetimeLikeArray or DTScalar,
        # but skip evaluating the Union at runtime for performance
        # (see https://github.com/pandas-dev/pandas/pull/44624)
        result = cast("Union[Self, DTScalarOrNaT]", super().__getitem__(key))
        if lib.is_scalar(result):
            return result
        else:
            # At this point we know the result is an array.
            result = cast(Self, result)
        result._freq = self._get_getitem_freq(key)
        return result

    def _get_getitem_freq(self, key) -> BaseOffset | None:
        """
        Find the `freq` attribute to assign to the result of a __getitem__ lookup.
        """
        is_period = isinstance(self.dtype, PeriodDtype)
        if is_period:
            freq = self.freq
        elif self.ndim != 1:
            freq = None
        else:
            key = check_array_indexer(self, key)  # maybe ndarray[bool] -> slice
            freq = None
            if isinstance(key, slice):
                if self.freq is not None and key.step is not None:
                    freq = key.step * self.freq
                else:
                    freq = self.freq
            elif key is Ellipsis:
                # GH#21282 indexing with Ellipsis is similar to a full slice,
                #  should preserve `freq` attribute
                freq = self.freq
            elif com.is_bool_indexer(key):
                new_key = lib.maybe_booleans_to_slice(key.view(np.uint8))
                if isinstance(new_key, slice):
                    return self._get_getitem_freq(new_key)
        return freq

    # error: Argument 1 of "__setitem__" is incompatible with supertype
    # "ExtensionArray"; supertype defines the argument type as "Union[int,
    # ndarray]"
    def __setitem__(
        self,
        key: int | Sequence[int] | Sequence[bool] | slice,
        value: NaTType | Any | Sequence[Any],
    ) -> None:
        # I'm fudging the types a bit here. "Any" above really depends
        # on type(self). For PeriodArray, it's Period (or stuff coercible
        # to a period in from_sequence). For DatetimeArray, it's Timestamp...
        # I don't know if mypy can do that, possibly with Generics.
        # https://mypy.readthedocs.io/en/latest/generics.html

        no_op = check_setitem_lengths(key, value, self)

        # Calling super() before the no_op short-circuit means that we raise
        #  on invalid 'value' even if this is a no-op, e.g. wrong-dtype empty array.
        super().__setitem__(key, value)

        if no_op:
            return

        self._maybe_clear_freq()

    def _maybe_clear_freq(self) -> None:
        # inplace operations like __setitem__ may invalidate the freq of
        # DatetimeArray and TimedeltaArray
        pass

    def astype(self, dtype, copy: bool = True):
        # Some notes on cases we don't have to handle here in the base class:
        #   1. PeriodArray.astype handles period -> period
        #   2. DatetimeArray.astype handles conversion between tz.
        #   3. DatetimeArray.astype handles datetime -> period
        dtype = pandas_dtype(dtype)

        if dtype == object:
            if self.dtype.kind == "M":
                self = cast("DatetimeArray", self)
                # *much* faster than self._box_values
                #  for e.g. test_get_loc_tuple_monotonic_above_size_cutoff
                i8data = self.asi8
                converted = ints_to_pydatetime(
                    i8data,
                    tz=self.tz,
                    box="timestamp",
                    reso=self._creso,
                )
                return converted

            elif self.dtype.kind == "m":
                return ints_to_pytimedelta(self._ndarray, box=True)

            return self._box_values(self.asi8.ravel()).reshape(self.shape)

        elif is_string_dtype(dtype):
            if isinstance(dtype, ExtensionDtype):
                arr_object = self._format_native_types(na_rep=dtype.na_value)  # type: ignore[arg-type]
                cls = dtype.construct_array_type()
                return cls._from_sequence(arr_object, dtype=dtype, copy=False)
            else:
                return self._format_native_types()

        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        elif dtype.kind in "iu":
            # we deliberately ignore int32 vs. int64 here.
            # See https://github.com/pandas-dev/pandas/issues/24381 for more.
            values = self.asi8
            if dtype != np.int64:
                raise TypeError(
                    f"Converting from {self.dtype} to {dtype} is not supported. "
                    "Do obj.astype('int64').astype(dtype) instead"
                )

            if copy:
                values = values.copy()
            return values
        elif (dtype.kind in "mM" and self.dtype != dtype) or dtype.kind == "f":
            # disallow conversion between datetime/timedelta,
            # and conversions for any datetimelike to float
            msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
            raise TypeError(msg)
        else:
            return np.asarray(self, dtype=dtype)

    @overload
    def view(self) -> Self:
        ...

    @overload
    def view(self, dtype: Literal["M8[ns]"]) -> DatetimeArray:
        ...

    @overload
    def view(self, dtype: Literal["m8[ns]"]) -> TimedeltaArray:
        ...

    @overload
    def view(self, dtype: Dtype | None = ...) -> ArrayLike:
        ...

    # pylint: disable-next=useless-parent-delegation
    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        # we need to explicitly call super() method as long as the `@overload`s
        #  are present in this file.
        return super().view(dtype)

    # ------------------------------------------------------------------
    # Validation Methods
    # TODO: try to de-duplicate these, ensure identical behavior

    def _validate_comparison_value(self, other):
        if isinstance(other, str):
            try:
                # GH#18435 strings get a pass from tzawareness compat
                other = self._scalar_from_string(other)
            except (ValueError, IncompatibleFrequency):
                # failed to parse as Timestamp/Timedelta/Period
                raise InvalidComparison(other)

        if isinstance(other, self._recognized_scalars) or other is NaT:
            other = self._scalar_type(other)
            try:
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                # e.g. tzawareness mismatch
                raise InvalidComparison(other) from err

        elif not is_list_like(other):
            raise InvalidComparison(other)

        elif len(other) != len(self):
            raise ValueError("Lengths must match")

        else:
            try:
                other = self._validate_listlike(other, allow_object=True)
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                if is_object_dtype(getattr(other, "dtype", None)):
                    # We will have to operate element-wise
                    pass
                else:
                    raise InvalidComparison(other) from err

        return other

    def _validate_scalar(
        self,
        value,
        *,
        allow_listlike: bool = False,
        unbox: bool = True,
    ):
        """
        Validate that the input value can be cast to our scalar_type.

        Parameters
        ----------
        value : object
        allow_listlike: bool, default False
            When raising an exception, whether the message should say
            listlike inputs are allowed.
        unbox : bool, default True
            Whether to unbox the result before returning.  Note: unbox=False
            skips the setitem compatibility check.

        Returns
        -------
        self._scalar_type or NaT
        """
        if isinstance(value, self._scalar_type):
            pass

        elif isinstance(value, str):
            # NB: Careful about tzawareness
            try:
                value = self._scalar_from_string(value)
            except ValueError as err:
                msg = self._validation_error_message(value, allow_listlike)
                raise TypeError(msg) from err

        elif is_valid_na_for_dtype(value, self.dtype):
            # GH#18295
            value = NaT

        elif isna(value):
            # if we are dt64tz and value is dt64("NaT"), dont cast to NaT,
            #  or else we'll fail to raise in _unbox_scalar
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)

        elif isinstance(value, self._recognized_scalars):
            # error: Argument 1 to "Timestamp" has incompatible type "object"; expected
            # "integer[Any] | float | str | date | datetime | datetime64"
            value = self._scalar_type(value)  # type: ignore[arg-type]

        else:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)

        if not unbox:
            # NB: In general NDArrayBackedExtensionArray will unbox here;
            #  this option exists to prevent a performance hit in
            #  TimedeltaIndex.get_loc
            return value
        return self._unbox_scalar(value)

    def _validation_error_message(self, value, allow_listlike: bool = False) -> str:
        """
        Construct an exception message on validation error.

        Some methods allow only scalar inputs, while others allow either scalar
        or listlike.

        Parameters
        ----------
        allow_listlike: bool, default False

        Returns
        -------
        str
        """
        if hasattr(value, "dtype") and getattr(value, "ndim", 0) > 0:
            msg_got = f"{value.dtype} array"
        else:
            msg_got = f"'{type(value).__name__}'"
        if allow_listlike:
            msg = (
                f"value should be a '{self._scalar_type.__name__}', 'NaT', "
                f"or array of those. Got {msg_got} instead."
            )
        else:
            msg = (
                f"value should be a '{self._scalar_type.__name__}' or 'NaT'. "
                f"Got {msg_got} instead."
            )
        return msg

    def _validate_listlike(self, value, allow_object: bool = False):
        if isinstance(value, type(self)):
            if self.dtype.kind in "mM" and not allow_object:
                # error: "DatetimeLikeArrayMixin" has no attribute "as_unit"
                value = value.as_unit(self.unit, round_ok=False)  # type: ignore[attr-defined]
            return value

        if isinstance(value, list) and len(value) == 0:
            # We treat empty list as our own dtype.
            return type(self)._from_sequence([], dtype=self.dtype)

        if hasattr(value, "dtype") and value.dtype == object:
            # `array` below won't do inference if value is an Index or Series.
            #  so do so here.  in the Index case, inferred_type may be cached.
            if lib.infer_dtype(value) in self._infer_matches:
                try:
                    value = type(self)._from_sequence(value)
                except (ValueError, TypeError):
                    if allow_object:
                        return value
                    msg = self._validation_error_message(value, True)
                    raise TypeError(msg)

        # Do type inference if necessary up front (after unpacking
        # NumpyExtensionArray)
        # e.g. we passed PeriodIndex.values and got an ndarray of Periods
        value = extract_array(value, extract_numpy=True)
        value = pd_array(value)
        value = extract_array(value, extract_numpy=True)

        if is_all_strings(value):
            # We got a StringArray
            try:
                # TODO: Could use from_sequence_of_strings if implemented
                # Note: passing dtype is necessary for PeriodArray tests
                value = type(self)._from_sequence(value, dtype=self.dtype)
            except ValueError:
                pass

        if isinstance(value.dtype, CategoricalDtype):
            # e.g. we have a Categorical holding self.dtype
            if value.categories.dtype == self.dtype:
                # TODO: do we need equal dtype or just comparable?
                value = value._internal_get_values()
                value = extract_array(value, extract_numpy=True)

        if allow_object and is_object_dtype(value.dtype):
            pass

        elif not type(self)._is_recognized_dtype(value.dtype):
            msg = self._validation_error_message(value, True)
            raise TypeError(msg)

        if self.dtype.kind in "mM" and not allow_object:
            # error: "DatetimeLikeArrayMixin" has no attribute "as_unit"
            value = value.as_unit(self.unit, round_ok=False)  # type: ignore[attr-defined]
        return value

    def _validate_setitem_value(self, value):
        if is_list_like(value):
            value = self._validate_listlike(value)
        else:
            return self._validate_scalar(value, allow_listlike=True)

        return self._unbox(value)

    @final
    def _unbox(self, other) -> np.int64 | np.datetime64 | np.timedelta64 | np.ndarray:
        """
        Unbox either a scalar with _unbox_scalar or an instance of our own type.
        """
        if lib.is_scalar(other):
            other = self._unbox_scalar(other)
        else:
            # same type as self
            self._check_compatible_with(other)
            other = other._ndarray
        return other

    # ------------------------------------------------------------------
    # Additional array methods
    #  These are not part of the EA API, but we implement them because
    #  pandas assumes they're there.

    @ravel_compat
    def map(self, mapper, na_action=None):
        from pandas import Index

        result = map_array(self, mapper, na_action=na_action)
        result = Index(result)

        if isinstance(result, ABCMultiIndex):
            return result.to_numpy()
        else:
            return result.array

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Compute boolean array of whether each value is found in the
        passed set of values.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray

        Returns
        -------
        ndarray[bool]
        """
        if values.dtype.kind in "fiuc":
            # TODO: de-duplicate with equals, validate_comparison_value
            return np.zeros(self.shape, dtype=bool)

        values = ensure_wrapped_if_datetimelike(values)

        if not isinstance(values, type(self)):
            inferable = [
                "timedelta",
                "timedelta64",
                "datetime",
                "datetime64",
                "date",
                "period",
            ]
            if values.dtype == object:
                values = lib.maybe_convert_objects(
                    values,  # type: ignore[arg-type]
                    convert_non_numeric=True,
                    dtype_if_all_nat=self.dtype,
                )
                if values.dtype != object:
                    return self.isin(values)

                inferred = lib.infer_dtype(values, skipna=False)
                if inferred not in inferable:
                    if inferred == "string":
                        pass

                    elif "mixed" in inferred:
                        return isin(self.astype(object), values)
                    else:
                        return np.zeros(self.shape, dtype=bool)

            try:
                values = type(self)._from_sequence(values)
            except ValueError:
                return isin(self.astype(object), values)
            else:
                warnings.warn(
                    # GH#53111
                    f"The behavior of 'isin' with dtype={self.dtype} and "
                    "castable values (e.g. strings) is deprecated. In a "
                    "future version, these will not be considered matching "
                    "by isin. Explicitly cast to the appropriate dtype before "
                    "calling isin instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

        if self.dtype.kind in "mM":
            self = cast("DatetimeArray | TimedeltaArray", self)
            # error: Item "ExtensionArray" of "ExtensionArray | ndarray[Any, Any]"
            # has no attribute "as_unit"
            values = values.as_unit(self.unit)  # type: ignore[union-attr]

        try:
            # error: Argument 1 to "_check_compatible_with" of "DatetimeLikeArrayMixin"
            # has incompatible type "ExtensionArray | ndarray[Any, Any]"; expected
            # "Period | Timestamp | Timedelta | NaTType"
            self._check_compatible_with(values)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Includes tzawareness mismatch and IncompatibleFrequencyError
            return np.zeros(self.shape, dtype=bool)

        # error: Item "ExtensionArray" of "ExtensionArray | ndarray[Any, Any]"
        # has no attribute "asi8"
        return isin(self.asi8, values.asi8)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Null Handling

    def isna(self) -> npt.NDArray[np.bool_]:
        return self._isnan

    @property  # NB: override with cache_readonly in immutable subclasses
    def _isnan(self) -> npt.NDArray[np.bool_]:
        """
        return if each value is nan
        """
        return self.asi8 == iNaT

    @property  # NB: override with cache_readonly in immutable subclasses
    def _hasna(self) -> bool:
        """
        return if I have any nans; enables various perf speedups
        """
        return bool(self._isnan.any())

    def _maybe_mask_results(
        self, result: np.ndarray, fill_value=iNaT, convert=None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        result : np.ndarray
        fill_value : object, default iNaT
        convert : str, dtype or None

        Returns
        -------
        result : ndarray with values replace by the fill_value

        mask the result if needed, convert to the provided dtype if its not
        None

        This is an internal routine.
        """
        if self._hasna:
            if convert:
                result = result.astype(convert)
            if fill_value is None:
                fill_value = np.nan
            np.putmask(result, self._isnan, fill_value)
        return result

    # ------------------------------------------------------------------
    # Frequency Properties/Methods

    @property
    def freqstr(self) -> str | None:
        """
        Return the frequency object as a string if it's set, otherwise None.

        Examples
        --------
        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00"], freq="D")
        >>> idx.freqstr
        'D'

        The frequency can be inferred if there are more than 2 points:

        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"],
        ...                        freq="infer")
        >>> idx.freqstr
        '2D'

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-1", "2023-2", "2023-3"], freq="M")
        >>> idx.freqstr
        'M'
        """
        if self.freq is None:
            return None
        return self.freq.freqstr

    @property  # NB: override with cache_readonly in immutable subclasses
    def inferred_freq(self) -> str | None:
        """
        Tries to return a string representing a frequency generated by infer_freq.

        Returns None if it can't autodetect the frequency.

        Examples
        --------
        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"])
        >>> idx.inferred_freq
        '2D'

        For TimedeltaIndex:

        >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])
        >>> tdelta_idx
        TimedeltaIndex(['0 days', '10 days', '20 days'],
                       dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.inferred_freq
        '10D'
        """
        if self.ndim != 1:
            return None
        try:
            return frequencies.infer_freq(self)
        except ValueError:
            return None

    @property  # NB: override with cache_readonly in immutable subclasses
    def _resolution_obj(self) -> Resolution | None:
        freqstr = self.freqstr
        if freqstr is None:
            return None
        try:
            return Resolution.get_reso_from_freqstr(freqstr)
        except KeyError:
            return None

    @property  # NB: override with cache_readonly in immutable subclasses
    def resolution(self) -> str:
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        # error: Item "None" of "Optional[Any]" has no attribute "attrname"
        return self._resolution_obj.attrname  # type: ignore[union-attr]

    # monotonicity/uniqueness properties are called via frequencies.infer_freq,
    #  see GH#23789

    @property
    def _is_monotonic_increasing(self) -> bool:
        return algos.is_monotonic(self.asi8, timelike=True)[0]

    @property
    def _is_monotonic_decreasing(self) -> bool:
        return algos.is_monotonic(self.asi8, timelike=True)[1]

    @property
    def _is_unique(self) -> bool:
        return len(unique1d(self.asi8.ravel("K"))) == self.size

    # ------------------------------------------------------------------
    # Arithmetic Methods

    def _cmp_method(self, other, op):
        if self.ndim > 1 and getattr(other, "shape", None) == self.shape:
            # TODO: handle 2D-like listlikes
            return op(self.ravel(), other.ravel()).reshape(self.shape)

        try:
            other = self._validate_comparison_value(other)
        except InvalidComparison:
            return invalid_comparison(self, other, op)

        dtype = getattr(other, "dtype", None)
        if is_object_dtype(dtype):
            # We have to use comp_method_OBJECT_ARRAY instead of numpy
            #  comparison otherwise it would raise when comparing to None
            result = ops.comp_method_OBJECT_ARRAY(
                op, np.asarray(self.astype(object)), other
            )
            return result
        if other is NaT:
            if op is operator.ne:
                result = np.ones(self.shape, dtype=bool)
            else:
                result = np.zeros(self.shape, dtype=bool)
            return result

        if not isinstance(self.dtype, PeriodDtype):
            self = cast(TimelikeOps, self)
            if self._creso != other._creso:
                if not isinstance(other, type(self)):
                    # i.e. Timedelta/Timestamp, cast to ndarray and let
                    #  compare_mismatched_resolutions handle broadcasting
                    try:
                        # GH#52080 see if we can losslessly cast to shared unit
                        other = other.as_unit(self.unit, round_ok=False)
                    except ValueError:
                        other_arr = np.array(other.asm8)
                        return compare_mismatched_resolutions(
                            self._ndarray, other_arr, op
                        )
                else:
                    other_arr = other._ndarray
                    return compare_mismatched_resolutions(self._ndarray, other_arr, op)

        other_vals = self._unbox(other)
        # GH#37462 comparison on i8 values is almost 2x faster than M8/m8
        result = op(self._ndarray.view("i8"), other_vals.view("i8"))

        o_mask = isna(other)
        mask = self._isnan | o_mask
        if mask.any():
            nat_result = op is operator.ne
            np.putmask(result, mask, nat_result)

        return result

    # pow is invalid for all three subclasses; TimedeltaArray will override
    #  the multiplication and division ops
    __pow__ = _make_unpacked_invalid_op("__pow__")
    __rpow__ = _make_unpacked_invalid_op("__rpow__")
    __mul__ = _make_unpacked_invalid_op("__mul__")
    __rmul__ = _make_unpacked_invalid_op("__rmul__")
    __truediv__ = _make_unpacked_invalid_op("__truediv__")
    __rtruediv__ = _make_unpacked_invalid_op("__rtruediv__")
    __floordiv__ = _make_unpacked_invalid_op("__floordiv__")
    __rfloordiv__ = _make_unpacked_invalid_op("__rfloordiv__")
    __mod__ = _make_unpacked_invalid_op("__mod__")
    __rmod__ = _make_unpacked_invalid_op("__rmod__")
    __divmod__ = _make_unpacked_invalid_op("__divmod__")
    __rdivmod__ = _make_unpacked_invalid_op("__rdivmod__")

    @final
    def _get_i8_values_and_mask(
        self, other
    ) -> tuple[int | npt.NDArray[np.int64], None | npt.NDArray[np.bool_]]:
        """
        Get the int64 values and b_mask to pass to add_overflowsafe.
        """
        if isinstance(other, Period):
            i8values = other.ordinal
            mask = None
        elif isinstance(other, (Timestamp, Timedelta)):
            i8values = other._value
            mask = None
        else:
            # PeriodArray, DatetimeArray, TimedeltaArray
            mask = other._isnan
            i8values = other.asi8
        return i8values, mask

    @final
    def _get_arithmetic_result_freq(self, other) -> BaseOffset | None:
        """
        Check if we can preserve self.freq in addition or subtraction.
        """
        # Adding or subtracting a Timedelta/Timestamp scalar is freq-preserving
        #  whenever self.freq is a Tick
        if isinstance(self.dtype, PeriodDtype):
            return self.freq
        elif not lib.is_scalar(other):
            return None
        elif isinstance(self.freq, Tick):
            # In these cases
            return self.freq
        return None

    @final
    def _add_datetimelike_scalar(self, other) -> DatetimeArray:
        if not lib.is_np_dtype(self.dtype, "m"):
            raise TypeError(
                f"cannot add {type(self).__name__} and {type(other).__name__}"
            )

        self = cast("TimedeltaArray", self)

        from pandas.core.arrays import DatetimeArray
        from pandas.core.arrays.datetimes import tz_to_dtype

        assert other is not NaT
        if isna(other):
            # i.e. np.datetime64("NaT")
            # In this case we specifically interpret NaT as a datetime, not
            # the timedelta interpretation we would get by returning self + NaT
            result = self._ndarray + NaT.to_datetime64().astype(f"M8[{self.unit}]")
            # Preserve our resolution
            return DatetimeArray._simple_new(result, dtype=result.dtype)

        other = Timestamp(other)
        self, other = self._ensure_matching_resos(other)
        self = cast("TimedeltaArray", self)

        other_i8, o_mask = self._get_i8_values_and_mask(other)
        result = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype="i8"))
        res_values = result.view(f"M8[{self.unit}]")

        dtype = tz_to_dtype(tz=other.tz, unit=self.unit)
        res_values = result.view(f"M8[{self.unit}]")
        new_freq = self._get_arithmetic_result_freq(other)
        return DatetimeArray._simple_new(res_values, dtype=dtype, freq=new_freq)

    @final
    def _add_datetime_arraylike(self, other: DatetimeArray) -> DatetimeArray:
        if not lib.is_np_dtype(self.dtype, "m"):
            raise TypeError(
                f"cannot add {type(self).__name__} and {type(other).__name__}"
            )

        # defer to DatetimeArray.__add__
        return other + self

    @final
    def _sub_datetimelike_scalar(
        self, other: datetime | np.datetime64
    ) -> TimedeltaArray:
        if self.dtype.kind != "M":
            raise TypeError(f"cannot subtract a datelike from a {type(self).__name__}")

        self = cast("DatetimeArray", self)
        # subtract a datetime from myself, yielding a ndarray[timedelta64[ns]]

        if isna(other):
            # i.e. np.datetime64("NaT")
            return self - NaT

        ts = Timestamp(other)

        self, ts = self._ensure_matching_resos(ts)
        return self._sub_datetimelike(ts)

    @final
    def _sub_datetime_arraylike(self, other: DatetimeArray) -> TimedeltaArray:
        if self.dtype.kind != "M":
            raise TypeError(f"cannot subtract a datelike from a {type(self).__name__}")

        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        self = cast("DatetimeArray", self)

        self, other = self._ensure_matching_resos(other)
        return self._sub_datetimelike(other)

    @final
    def _sub_datetimelike(self, other: Timestamp | DatetimeArray) -> TimedeltaArray:
        self = cast("DatetimeArray", self)

        from pandas.core.arrays import TimedeltaArray

        try:
            self._assert_tzawareness_compat(other)
        except TypeError as err:
            new_message = str(err).replace("compare", "subtract")
            raise type(err)(new_message) from err

        other_i8, o_mask = self._get_i8_values_and_mask(other)
        res_values = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype="i8"))
        res_m8 = res_values.view(f"timedelta64[{self.unit}]")

        new_freq = self._get_arithmetic_result_freq(other)
        new_freq = cast("Tick | None", new_freq)
        return TimedeltaArray._simple_new(res_m8, dtype=res_m8.dtype, freq=new_freq)

    @final
    def _add_period(self, other: Period) -> PeriodArray:
        if not lib.is_np_dtype(self.dtype, "m"):
            raise TypeError(f"cannot add Period to a {type(self).__name__}")

        # We will wrap in a PeriodArray and defer to the reversed operation
        from pandas.core.arrays.period import PeriodArray

        i8vals = np.broadcast_to(other.ordinal, self.shape)
        dtype = PeriodDtype(other.freq)
        parr = PeriodArray(i8vals, dtype=dtype)
        return parr + self

    def _add_offset(self, offset):
        raise AbstractMethodError(self)

    def _add_timedeltalike_scalar(self, other):
        """
        Add a delta of a timedeltalike

        Returns
        -------
        Same type as self
        """
        if isna(other):
            # i.e np.timedelta64("NaT")
            new_values = np.empty(self.shape, dtype="i8").view(self._ndarray.dtype)
            new_values.fill(iNaT)
            return type(self)._simple_new(new_values, dtype=self.dtype)

        # PeriodArray overrides, so we only get here with DTA/TDA
        self = cast("DatetimeArray | TimedeltaArray", self)
        other = Timedelta(other)
        self, other = self._ensure_matching_resos(other)
        return self._add_timedeltalike(other)

    def _add_timedelta_arraylike(self, other: TimedeltaArray):
        """
        Add a delta of a TimedeltaIndex

        Returns
        -------
        Same type as self
        """
        # overridden by PeriodArray

        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        self = cast("DatetimeArray | TimedeltaArray", self)

        self, other = self._ensure_matching_resos(other)
        return self._add_timedeltalike(other)

    @final
    def _add_timedeltalike(self, other: Timedelta | TimedeltaArray):
        self = cast("DatetimeArray | TimedeltaArray", self)

        other_i8, o_mask = self._get_i8_values_and_mask(other)
        new_values = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype="i8"))
        res_values = new_values.view(self._ndarray.dtype)

        new_freq = self._get_arithmetic_result_freq(other)

        # error: Argument "dtype" to "_simple_new" of "DatetimeArray" has
        # incompatible type "Union[dtype[datetime64], DatetimeTZDtype,
        # dtype[timedelta64]]"; expected "Union[dtype[datetime64], DatetimeTZDtype]"
        return type(self)._simple_new(
            res_values, dtype=self.dtype, freq=new_freq  # type: ignore[arg-type]
        )

    @final
    def _add_nat(self):
        """
        Add pd.NaT to self
        """
        if isinstance(self.dtype, PeriodDtype):
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(NaT).__name__}"
            )
        self = cast("TimedeltaArray | DatetimeArray", self)

        # GH#19124 pd.NaT is treated like a timedelta for both timedelta
        # and datetime dtypes
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        result = result.view(self._ndarray.dtype)  # preserve reso
        # error: Argument "dtype" to "_simple_new" of "DatetimeArray" has
        # incompatible type "Union[dtype[timedelta64], dtype[datetime64],
        # DatetimeTZDtype]"; expected "Union[dtype[datetime64], DatetimeTZDtype]"
        return type(self)._simple_new(
            result, dtype=self.dtype, freq=None  # type: ignore[arg-type]
        )

    @final
    def _sub_nat(self):
        """
        Subtract pd.NaT from self
        """
        # GH#19124 Timedelta - datetime is not in general well-defined.
        # We make an exception for pd.NaT, which in this case quacks
        # like a timedelta.
        # For datetime64 dtypes by convention we treat NaT as a datetime, so
        # this subtraction returns a timedelta64 dtype.
        # For period dtype, timedelta64 is a close-enough return dtype.
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        if self.dtype.kind in "mM":
            # We can retain unit in dtype
            self = cast("DatetimeArray| TimedeltaArray", self)
            return result.view(f"timedelta64[{self.unit}]")
        else:
            return result.view("timedelta64[ns]")

    @final
    def _sub_periodlike(self, other: Period | PeriodArray) -> npt.NDArray[np.object_]:
        # If the operation is well-defined, we return an object-dtype ndarray
        # of DateOffsets.  Null entries are filled with pd.NaT
        if not isinstance(self.dtype, PeriodDtype):
            raise TypeError(
                f"cannot subtract {type(other).__name__} from {type(self).__name__}"
            )

        self = cast("PeriodArray", self)
        self._check_compatible_with(other)

        other_i8, o_mask = self._get_i8_values_and_mask(other)
        new_i8_data = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype="i8"))
        new_data = np.array([self.freq.base * x for x in new_i8_data])

        if o_mask is None:
            # i.e. Period scalar
            mask = self._isnan
        else:
            # i.e. PeriodArray
            mask = self._isnan | o_mask
        new_data[mask] = NaT
        return new_data

    @final
    def _addsub_object_array(self, other: npt.NDArray[np.object_], op):
        """
        Add or subtract array-like of DateOffset objects

        Parameters
        ----------
        other : np.ndarray[object]
        op : {operator.add, operator.sub}

        Returns
        -------
        np.ndarray[object]
            Except in fastpath case with length 1 where we operate on the
            contained scalar.
        """
        assert op in [operator.add, operator.sub]
        if len(other) == 1 and self.ndim == 1:
            # Note: without this special case, we could annotate return type
            #  as ndarray[object]
            # If both 1D then broadcasting is unambiguous
            return op(self, other[0])

        warnings.warn(
            "Adding/subtracting object-dtype array to "
            f"{type(self).__name__} not vectorized.",
            PerformanceWarning,
            stacklevel=find_stack_level(),
        )

        # Caller is responsible for broadcasting if necessary
        assert self.shape == other.shape, (self.shape, other.shape)

        res_values = op(self.astype("O"), np.asarray(other))
        return res_values

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> Self:
        if name not in {"cummin", "cummax"}:
            raise TypeError(f"Accumulation {name} not supported for {type(self)}")

        op = getattr(datetimelike_accumulations, name)
        result = op(self.copy(), skipna=skipna, **kwargs)

        return type(self)._simple_new(result, dtype=self.dtype)

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other):
        other_dtype = getattr(other, "dtype", None)
        other = ensure_wrapped_if_datetimelike(other)

        # scalar others
        if other is NaT:
            result = self._add_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(other)
        elif isinstance(other, BaseOffset):
            # specifically _not_ a Tick
            result = self._add_offset(other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._add_datetimelike_scalar(other)
        elif isinstance(other, Period) and lib.is_np_dtype(self.dtype, "m"):
            result = self._add_period(other)
        elif lib.is_integer(other):
            # This check must come after the check for np.timedelta64
            # as is_integer returns True for these
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)

        # array-like others
        elif lib.is_np_dtype(other_dtype, "m"):
            # TimedeltaIndex, ndarray[timedelta64]
            result = self._add_timedelta_arraylike(other)
        elif is_object_dtype(other_dtype):
            # e.g. Array/Index of DateOffset objects
            result = self._addsub_object_array(other, operator.add)
        elif lib.is_np_dtype(other_dtype, "M") or isinstance(
            other_dtype, DatetimeTZDtype
        ):
            # DatetimeIndex, ndarray[datetime64]
            return self._add_datetime_arraylike(other)
        elif is_integer_dtype(other_dtype):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)
        else:
            # Includes Categorical, other ExtensionArrays
            # For PeriodDtype, if self is a TimedeltaArray and other is a
            #  PeriodArray with  a timedelta-like (i.e. Tick) freq, this
            #  operation is valid.  Defer to the PeriodArray implementation.
            #  In remaining cases, this will end up raising TypeError.
            return NotImplemented

        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, "m"):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._from_sequence(result)
        return result

    def __radd__(self, other):
        # alias for __add__
        return self.__add__(other)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        other_dtype = getattr(other, "dtype", None)
        other = ensure_wrapped_if_datetimelike(other)

        # scalar others
        if other is NaT:
            result = self._sub_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(-other)
        elif isinstance(other, BaseOffset):
            # specifically _not_ a Tick
            result = self._add_offset(-other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._sub_datetimelike_scalar(other)
        elif lib.is_integer(other):
            # This check must come after the check for np.timedelta64
            # as is_integer returns True for these
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)

        elif isinstance(other, Period):
            result = self._sub_periodlike(other)

        # array-like others
        elif lib.is_np_dtype(other_dtype, "m"):
            # TimedeltaIndex, ndarray[timedelta64]
            result = self._add_timedelta_arraylike(-other)
        elif is_object_dtype(other_dtype):
            # e.g. Array/Index of DateOffset objects
            result = self._addsub_object_array(other, operator.sub)
        elif lib.is_np_dtype(other_dtype, "M") or isinstance(
            other_dtype, DatetimeTZDtype
        ):
            # DatetimeIndex, ndarray[datetime64]
            result = self._sub_datetime_arraylike(other)
        elif isinstance(other_dtype, PeriodDtype):
            # PeriodIndex
            result = self._sub_periodlike(other)
        elif is_integer_dtype(other_dtype):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast("PeriodArray", self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)
        else:
            # Includes ExtensionArrays, float_dtype
            return NotImplemented

        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, "m"):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._from_sequence(result)
        return result

    def __rsub__(self, other):
        other_dtype = getattr(other, "dtype", None)
        other_is_dt64 = lib.is_np_dtype(other_dtype, "M") or isinstance(
            other_dtype, DatetimeTZDtype
        )

        if other_is_dt64 and lib.is_np_dtype(self.dtype, "m"):
            # ndarray[datetime64] cannot be subtracted from self, so
            # we need to wrap in DatetimeArray/Index and flip the operation
            if lib.is_scalar(other):
                # i.e. np.datetime64 object
                return Timestamp(other) - self
            if not isinstance(other, DatetimeLikeArrayMixin):
                # Avoid down-casting DatetimeIndex
                from pandas.core.arrays import DatetimeArray

                other = DatetimeArray._from_sequence(other)
            return other - self
        elif self.dtype.kind == "M" and hasattr(other, "dtype") and not other_is_dt64:
            # GH#19959 datetime - datetime is well-defined as timedelta,
            # but any other type - datetime is not well-defined.
            raise TypeError(
                f"cannot subtract {type(self).__name__} from {type(other).__name__}"
            )
        elif isinstance(self.dtype, PeriodDtype) and lib.is_np_dtype(other_dtype, "m"):
            # TODO: Can we simplify/generalize these cases at all?
            raise TypeError(f"cannot subtract {type(self).__name__} from {other.dtype}")
        elif lib.is_np_dtype(self.dtype, "m"):
            self = cast("TimedeltaArray", self)
            return (-self) + other

        # We get here with e.g. datetime objects
        return -(self - other)

    def __iadd__(self, other) -> Self:
        result = self + other
        self[:] = result[:]

        if not isinstance(self.dtype, PeriodDtype):
            # restore freq, which is invalidated by setitem
            self._freq = result.freq
        return self

    def __isub__(self, other) -> Self:
        result = self - other
        self[:] = result[:]

        if not isinstance(self.dtype, PeriodDtype):
            # restore freq, which is invalidated by setitem
            self._freq = result.freq
        return self

    # --------------------------------------------------------------
    # Reductions

    @_period_dispatch
    def _quantile(
        self,
        qs: npt.NDArray[np.float64],
        interpolation: str,
    ) -> Self:
        return super()._quantile(qs=qs, interpolation=interpolation)

    @_period_dispatch
    def min(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        """
        Return the minimum value of the Array or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Index.min : Return the minimum value in an Index.
        Series.min : Return the minimum value in a Series.
        """
        nv.validate_min((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)

        result = nanops.nanmin(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def max(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        """
        Return the maximum value of the Array or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Index.max : Return the maximum value in an Index.
        Series.max : Return the maximum value in a Series.
        """
        nv.validate_max((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)

        result = nanops.nanmax(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def mean(self, *, skipna: bool = True, axis: AxisInt | None = 0):
        """
        Return the mean value of the Array.

        Parameters
        ----------
        skipna : bool, default True
            Whether to ignore any NaT elements.
        axis : int, optional, default 0

        Returns
        -------
        scalar
            Timestamp or Timedelta.

        See Also
        --------
        numpy.ndarray.mean : Returns the average of array elements along a given axis.
        Series.mean : Return the mean value in a Series.

        Notes
        -----
        mean is only defined for Datetime and Timedelta dtypes, not for Period.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range('2001-01-01 00:00', periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.mean()
        Timestamp('2001-01-02 00:00:00')

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='D')
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.mean()
        Timedelta('2 days 00:00:00')
        """
        if isinstance(self.dtype, PeriodDtype):
            # See discussion in GH#24757
            raise TypeError(
                f"mean is not implemented for {type(self).__name__} since the "
                "meaning is ambiguous.  An alternative is "
                "obj.to_timestamp(how='start').mean()"
            )

        result = nanops.nanmean(
            self._ndarray, axis=axis, skipna=skipna, mask=self.isna()
        )
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def median(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs):
        nv.validate_median((), kwargs)

        if axis is not None and abs(axis) >= self.ndim:
            raise ValueError("abs(axis) must be less than ndim")

        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def _mode(self, dropna: bool = True):
        mask = None
        if dropna:
            mask = self.isna()

        i8modes = algorithms.mode(self.view("i8"), mask=mask)
        npmodes = i8modes.view(self._ndarray.dtype)
        npmodes = cast(np.ndarray, npmodes)
        return self._from_backing_data(npmodes)

    # ------------------------------------------------------------------
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
        dtype = self.dtype
        if dtype.kind == "M":
            # Adding/multiplying datetimes is not valid
            if how in ["sum", "prod", "cumsum", "cumprod", "var", "skew"]:
                raise TypeError(f"datetime64 type does not support {how} operations")
            if how in ["any", "all"]:
                # GH#34479
                warnings.warn(
                    f"'{how}' with datetime64 dtypes is deprecated and will raise in a "
                    f"future version. Use (obj != pd.Timestamp(0)).{how}() instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

        elif isinstance(dtype, PeriodDtype):
            # Adding/multiplying Periods is not valid
            if how in ["sum", "prod", "cumsum", "cumprod", "var", "skew"]:
                raise TypeError(f"Period type does not support {how} operations")
            if how in ["any", "all"]:
                # GH#34479
                warnings.warn(
                    f"'{how}' with PeriodDtype is deprecated and will raise in a "
                    f"future version. Use (obj != pd.Period(0, freq)).{how}() instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
        else:
            # timedeltas we can add but not multiply
            if how in ["prod", "cumprod", "skew", "var"]:
                raise TypeError(f"timedelta64 type does not support {how} operations")

        # All of the functions implemented here are ordinal, so we can
        #  operate on the tz-naive equivalents
        npvalues = self._ndarray.view("M8[ns]")

        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=None,
            **kwargs,
        )

        if op.how in op.cast_blocklist:
            # i.e. how in ["rank"], since other cast_blocklist methods don't go
            #  through cython_operation
            return res_values

        # We did a view to M8[ns] above, now we go the other direction
        assert res_values.dtype == "M8[ns]"
        if how in ["std", "sem"]:
            from pandas.core.arrays import TimedeltaArray

            if isinstance(self.dtype, PeriodDtype):
                raise TypeError("'std' and 'sem' are not valid for PeriodDtype")
            self = cast("DatetimeArray | TimedeltaArray", self)
            new_dtype = f"m8[{self.unit}]"
            res_values = res_values.view(new_dtype)
            return TimedeltaArray._simple_new(res_values, dtype=res_values.dtype)

        res_values = res_values.view(self._ndarray.dtype)
        return self._from_backing_data(res_values)


class DatelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for DatetimeIndex/PeriodIndex, but not TimedeltaIndex.
    """

    @Substitution(
        URL="https://docs.python.org/3/library/datetime.html"
        "#strftime-and-strptime-behavior"
    )
    def strftime(self, date_format: str) -> npt.NDArray[np.object_]:
        """
        Convert to Index using specified date_format.

        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library. Details
        of the string format can be found in `python string format
        doc <%(URL)s>`__.

        Formats supported by the C `strftime` API but not by the python string format
        doc (such as `"%%R"`, `"%%r"`) are not officially supported and should be
        preferably replaced with their supported equivalents (such as `"%%H:%%M"`,
        `"%%I:%%M:%%S %%p"`).

        Note that `PeriodIndex` support additional directives, detailed in
        `Period.strftime`.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%%Y-%%m-%%d").

        Returns
        -------
        ndarray[object]
            NumPy ndarray of formatted strings.

        See Also
        --------
        to_datetime : Convert the given argument to datetime.
        DatetimeIndex.normalize : Return DatetimeIndex with times to midnight.
        DatetimeIndex.round : Round the DatetimeIndex to the specified freq.
        DatetimeIndex.floor : Floor the DatetimeIndex to the specified freq.
        Timestamp.strftime : Format a single Timestamp.
        Period.strftime : Format a single Period.

        Examples
        --------
        >>> rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"),
        ...                     periods=3, freq='s')
        >>> rng.strftime('%%B %%d, %%Y, %%r')
        Index(['March 10, 2018, 09:00:00 AM', 'March 10, 2018, 09:00:01 AM',
               'March 10, 2018, 09:00:02 AM'],
              dtype='object')
        """
        result = self._format_native_types(date_format=date_format, na_rep=np.nan)
        if using_string_dtype():
            from pandas import StringDtype

            return pd_array(result, dtype=StringDtype(na_value=np.nan))  # type: ignore[return-value]
        return result.astype(object, copy=False)


_round_doc = """
    Perform {op} operation on the data to the specified `freq`.

    Parameters
    ----------
    freq : str or Offset
        The frequency level to {op} the index to. Must be a fixed
        frequency like 'S' (second) not 'ME' (month end). See
        :ref:`frequency aliases <timeseries.offset_aliases>` for
        a list of possible `freq` values.
    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        Only relevant for DatetimeIndex:

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False designates
          a non-DST time (note that this flag is only applicable for
          ambiguous times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous
          times.

    nonexistent : 'shift_forward', 'shift_backward', 'NaT', timedelta, default 'raise'
        A nonexistent time does not exist in a particular timezone
        where clocks moved forward due to DST.

        - 'shift_forward' will shift the nonexistent time forward to the
          closest existing time
        - 'shift_backward' will shift the nonexistent time backward to the
          closest existing time
        - 'NaT' will return NaT where there are nonexistent times
        - timedelta objects will shift nonexistent times by the timedelta
        - 'raise' will raise an NonExistentTimeError if there are
          nonexistent times.

    Returns
    -------
    DatetimeIndex, TimedeltaIndex, or Series
        Index of the same type for a DatetimeIndex or TimedeltaIndex,
        or a Series with the same index for a Series.

    Raises
    ------
    ValueError if the `freq` cannot be converted.

    Notes
    -----
    If the timestamps have a timezone, {op}ing will take place relative to the
    local ("wall") time and re-localized to the same timezone. When {op}ing
    near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
    control the re-localization behavior.

    Examples
    --------
    **DatetimeIndex**

    >>> rng = pd.date_range('1/1/2018 11:59:00', periods=3, freq='min')
    >>> rng
    DatetimeIndex(['2018-01-01 11:59:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:01:00'],
                  dtype='datetime64[ns]', freq='min')
    """

_round_example = """>>> rng.round('h')
    DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.round("h")
    0   2018-01-01 12:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 12:00:00
    dtype: datetime64[ns]

    When rounding near a daylight savings time transition, use ``ambiguous`` or
    ``nonexistent`` to control how the timestamp should be re-localized.

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.floor("2h", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)

    >>> rng_tz.floor("2h", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)
    """

_floor_example = """>>> rng.floor('h')
    DatetimeIndex(['2018-01-01 11:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.floor("h")
    0   2018-01-01 11:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 12:00:00
    dtype: datetime64[ns]

    When rounding near a daylight savings time transition, use ``ambiguous`` or
    ``nonexistent`` to control how the timestamp should be re-localized.

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.floor("2h", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                 dtype='datetime64[ns, Europe/Amsterdam]', freq=None)

    >>> rng_tz.floor("2h", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)
    """

_ceil_example = """>>> rng.ceil('h')
    DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 13:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.ceil("h")
    0   2018-01-01 12:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 13:00:00
    dtype: datetime64[ns]

    When rounding near a daylight savings time transition, use ``ambiguous`` or
    ``nonexistent`` to control how the timestamp should be re-localized.

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 01:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.ceil("h", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)

    >>> rng_tz.ceil("h", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)
    """


class TimelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for TimedeltaIndex/DatetimeIndex, but not PeriodIndex.
    """

    _default_dtype: np.dtype

    def __init__(
        self, values, dtype=None, freq=lib.no_default, copy: bool = False
    ) -> None:
        warnings.warn(
            # GH#55623
            f"{type(self).__name__}.__init__ is deprecated and will be "
            "removed in a future version. Use pd.array instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        if dtype is not None:
            dtype = pandas_dtype(dtype)

        values = extract_array(values, extract_numpy=True)
        if isinstance(values, IntegerArray):
            values = values.to_numpy("int64", na_value=iNaT)

        inferred_freq = getattr(values, "_freq", None)
        explicit_none = freq is None
        freq = freq if freq is not lib.no_default else None

        if isinstance(values, type(self)):
            if explicit_none:
                # don't inherit from values
                pass
            elif freq is None:
                freq = values.freq
            elif freq and values.freq:
                freq = to_offset(freq)
                freq = _validate_inferred_freq(freq, values.freq)

            if dtype is not None and dtype != values.dtype:
                # TODO: we only have tests for this for DTA, not TDA (2022-07-01)
                raise TypeError(
                    f"dtype={dtype} does not match data dtype {values.dtype}"
                )

            dtype = values.dtype
            values = values._ndarray

        elif dtype is None:
            if isinstance(values, np.ndarray) and values.dtype.kind in "Mm":
                dtype = values.dtype
            else:
                dtype = self._default_dtype
                if isinstance(values, np.ndarray) and values.dtype == "i8":
                    values = values.view(dtype)

        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"Unexpected type '{type(values).__name__}'. 'values' must be a "
                f"{type(self).__name__}, ndarray, or Series or Index "
                "containing one of those."
            )
        if values.ndim not in [1, 2]:
            raise ValueError("Only 1-dimensional input arrays are supported.")

        if values.dtype == "i8":
            # for compat with datetime/timedelta/period shared methods,
            #  we can sometimes get here with int64 values.  These represent
            #  nanosecond UTC (or tz-naive) unix timestamps
            if dtype is None:
                dtype = self._default_dtype
                values = values.view(self._default_dtype)
            elif lib.is_np_dtype(dtype, "mM"):
                values = values.view(dtype)
            elif isinstance(dtype, DatetimeTZDtype):
                kind = self._default_dtype.kind
                new_dtype = f"{kind}8[{dtype.unit}]"
                values = values.view(new_dtype)

        dtype = self._validate_dtype(values, dtype)

        if freq == "infer":
            raise ValueError(
                f"Frequency inference not allowed in {type(self).__name__}.__init__. "
                "Use 'pd.array()' instead."
            )

        if copy:
            values = values.copy()
        if freq:
            freq = to_offset(freq)
            if values.dtype.kind == "m" and not isinstance(freq, Tick):
                raise TypeError("TimedeltaArray/Index freq must be a Tick")

        NDArrayBacked.__init__(self, values=values, dtype=dtype)
        self._freq = freq

        if inferred_freq is None and freq is not None:
            type(self)._validate_frequency(self, freq)

    @classmethod
    def _validate_dtype(cls, values, dtype):
        raise AbstractMethodError(cls)

    @property
    def freq(self):
        """
        Return the frequency object if it is set, otherwise None.
        """
        return self._freq

    @freq.setter
    def freq(self, value) -> None:
        if value is not None:
            value = to_offset(value)
            self._validate_frequency(self, value)
            if self.dtype.kind == "m" and not isinstance(value, Tick):
                raise TypeError("TimedeltaArray/Index freq must be a Tick")

            if self.ndim > 1:
                raise ValueError("Cannot set freq with ndim > 1")

        self._freq = value

    @final
    def _maybe_pin_freq(self, freq, validate_kwds: dict):
        """
        Constructor helper to pin the appropriate `freq` attribute.  Assumes
        that self._freq is currently set to any freq inferred in
        _from_sequence_not_strict.
        """
        if freq is None:
            # user explicitly passed None -> override any inferred_freq
            self._freq = None
        elif freq == "infer":
            # if self._freq is *not* None then we already inferred a freq
            #  and there is nothing left to do
            if self._freq is None:
                # Set _freq directly to bypass duplicative _validate_frequency
                # check.
                self._freq = to_offset(self.inferred_freq)
        elif freq is lib.no_default:
            # user did not specify anything, keep inferred freq if the original
            #  data had one, otherwise do nothing
            pass
        elif self._freq is None:
            # We cannot inherit a freq from the data, so we need to validate
            #  the user-passed freq
            freq = to_offset(freq)
            type(self)._validate_frequency(self, freq, **validate_kwds)
            self._freq = freq
        else:
            # Otherwise we just need to check that the user-passed freq
            #  doesn't conflict with the one we already have.
            freq = to_offset(freq)
            _validate_inferred_freq(freq, self._freq)

    @final
    @classmethod
    def _validate_frequency(cls, index, freq: BaseOffset, **kwargs):
        """
        Validate that a frequency is compatible with the values of a given
        Datetime Array/Index or Timedelta Array/Index

        Parameters
        ----------
        index : DatetimeIndex or TimedeltaIndex
            The index on which to determine if the given frequency is valid
        freq : DateOffset
            The frequency to validate
        """
        inferred = index.inferred_freq
        if index.size == 0 or inferred == freq.freqstr:
            return None

        try:
            on_freq = cls._generate_range(
                start=index[0],
                end=None,
                periods=len(index),
                freq=freq,
                unit=index.unit,
                **kwargs,
            )
            if not np.array_equal(index.asi8, on_freq.asi8):
                raise ValueError
        except ValueError as err:
            if "non-fixed" in str(err):
                # non-fixed frequencies are not meaningful for timedelta64;
                #  we retain that error message
                raise err
            # GH#11587 the main way this is reached is if the `np.array_equal`
            #  check above is False.  This can also be reached if index[0]
            #  is `NaT`, in which case the call to `cls._generate_range` will
            #  raise a ValueError, which we re-raise with a more targeted
            #  message.
            raise ValueError(
                f"Inferred frequency {inferred} from passed values "
                f"does not conform to passed frequency {freq.freqstr}"
            ) from err

    @classmethod
    def _generate_range(
        cls, start, end, periods: int | None, freq, *args, **kwargs
    ) -> Self:
        raise AbstractMethodError(cls)

    # --------------------------------------------------------------

    @cache_readonly
    def _creso(self) -> int:
        return get_unit_from_dtype(self._ndarray.dtype)

    @cache_readonly
    def unit(self) -> str:
        # e.g. "ns", "us", "ms"
        # error: Argument 1 to "dtype_to_unit" has incompatible type
        # "ExtensionDtype"; expected "Union[DatetimeTZDtype, dtype[Any]]"
        return dtype_to_unit(self.dtype)  # type: ignore[arg-type]

    def as_unit(self, unit: str, round_ok: bool = True) -> Self:
        if unit not in ["s", "ms", "us", "ns"]:
            raise ValueError("Supported units are 's', 'ms', 'us', 'ns'")

        dtype = np.dtype(f"{self.dtype.kind}8[{unit}]")
        new_values = astype_overflowsafe(self._ndarray, dtype, round_ok=round_ok)

        if isinstance(self.dtype, np.dtype):
            new_dtype = new_values.dtype
        else:
            tz = cast("DatetimeArray", self).tz
            new_dtype = DatetimeTZDtype(tz=tz, unit=unit)

        # error: Unexpected keyword argument "freq" for "_simple_new" of
        # "NDArrayBacked"  [call-arg]
        return type(self)._simple_new(
            new_values, dtype=new_dtype, freq=self.freq  # type: ignore[call-arg]
        )

    # TODO: annotate other as DatetimeArray | TimedeltaArray | Timestamp | Timedelta
    #  with the return type matching input type.  TypeVar?
    def _ensure_matching_resos(self, other):
        if self._creso != other._creso:
            # Just as with Timestamp/Timedelta, we cast to the higher resolution
            if self._creso < other._creso:
                self = self.as_unit(other.unit)
            else:
                other = other.as_unit(self.unit)
        return self, other

    # --------------------------------------------------------------

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if (
            ufunc in [np.isnan, np.isinf, np.isfinite]
            and len(inputs) == 1
            and inputs[0] is self
        ):
            # numpy 1.18 changed isinf and isnan to not raise on dt64/td64
            return getattr(ufunc, method)(self._ndarray, **kwargs)

        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def _round(self, freq, mode, ambiguous, nonexistent):
        # round the local times
        if isinstance(self.dtype, DatetimeTZDtype):
            # operate on naive timestamps, then convert back to aware
            self = cast("DatetimeArray", self)
            naive = self.tz_localize(None)
            result = naive._round(freq, mode, ambiguous, nonexistent)
            return result.tz_localize(
                self.tz, ambiguous=ambiguous, nonexistent=nonexistent
            )

        values = self.view("i8")
        values = cast(np.ndarray, values)
        nanos = get_unit_for_round(freq, self._creso)
        if nanos == 0:
            # GH 52761
            return self.copy()
        result_i8 = round_nsint64(values, mode, nanos)
        result = self._maybe_mask_results(result_i8, fill_value=iNaT)
        result = result.view(self._ndarray.dtype)
        return self._simple_new(result, dtype=self.dtype)

    @Appender((_round_doc + _round_example).format(op="round"))
    def round(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent)

    @Appender((_round_doc + _floor_example).format(op="floor"))
    def floor(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    @Appender((_round_doc + _ceil_example).format(op="ceil"))
    def ceil(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    # --------------------------------------------------------------
    # Reductions

    def any(self, *, axis: AxisInt | None = None, skipna: bool = True) -> bool:
        # GH#34479 the nanops call will issue a FutureWarning for non-td64 dtype
        return nanops.nanany(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def all(self, *, axis: AxisInt | None = None, skipna: bool = True) -> bool:
        # GH#34479 the nanops call will issue a FutureWarning for non-td64 dtype

        return nanops.nanall(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    # --------------------------------------------------------------
    # Frequency Methods

    def _maybe_clear_freq(self) -> None:
        self._freq = None

    def _with_freq(self, freq) -> Self:
        """
        Helper to get a view on the same data, with a new freq.

        Parameters
        ----------
        freq : DateOffset, None, or "infer"

        Returns
        -------
        Same type as self
        """
        # GH#29843
        if freq is None:
            # Always valid
            pass
        elif len(self) == 0 and isinstance(freq, BaseOffset):
            # Always valid.  In the TimedeltaArray case, we require a Tick offset
            if self.dtype.kind == "m" and not isinstance(freq, Tick):
                raise TypeError("TimedeltaArray/Index freq must be a Tick")
        else:
            # As an internal method, we can ensure this assertion always holds
            assert freq == "infer"
            freq = to_offset(self.inferred_freq)

        arr = self.view()
        arr._freq = freq
        return arr

    # --------------------------------------------------------------
    # ExtensionArray Interface

    def _values_for_json(self) -> np.ndarray:
        # Small performance bump vs the base class which calls np.asarray(self)
        if isinstance(self.dtype, np.dtype):
            return self._ndarray
        return super()._values_for_json()

    def factorize(
        self,
        use_na_sentinel: bool = True,
        sort: bool = False,
    ):
        if self.freq is not None:
            # We must be unique, so can short-circuit (and retain freq)
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self.copy()  # TODO: copy or view?
            if sort and self.freq.n < 0:
                codes = codes[::-1]
                uniques = uniques[::-1]
            return codes, uniques

        if sort:
            # algorithms.factorize only passes sort=True here when freq is
            #  not None, so this should not be reached.
            raise NotImplementedError(
                f"The 'sort' keyword in {type(self).__name__}.factorize is "
                "ignored unless arr.freq is not None. To factorize with sort, "
                "call pd.factorize(obj, sort=True) instead."
            )
        return super().factorize(use_na_sentinel=use_na_sentinel)

    @classmethod
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        new_obj = super()._concat_same_type(to_concat, axis)

        obj = to_concat[0]

        if axis == 0:
            # GH 3232: If the concat result is evenly spaced, we can retain the
            # original frequency
            to_concat = [x for x in to_concat if len(x)]

            if obj.freq is not None and all(x.freq == obj.freq for x in to_concat):
                pairs = zip(to_concat[:-1], to_concat[1:])
                if all(pair[0][-1] + obj.freq == pair[1][0] for pair in pairs):
                    new_freq = obj.freq
                    new_obj._freq = new_freq
        return new_obj

    def copy(self, order: str = "C") -> Self:
        new_obj = super().copy(order=order)
        new_obj._freq = self.freq
        return new_obj

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if method != "linear":
            raise NotImplementedError

        if not copy:
            out_data = self._ndarray
        else:
            out_data = self._ndarray.copy()

        missing.interpolate_2d_inplace(
            out_data,
            method=method,
            axis=axis,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            **kwargs,
        )
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    # --------------------------------------------------------------
    # Unsorted

    @property
    def _is_dates_only(self) -> bool:
        """
        Check if we are round times at midnight (and no timezone), which will
        be given a more compact __repr__ than other cases. For TimedeltaArray
        we are checking for multiples of 24H.
        """
        if not lib.is_np_dtype(self.dtype):
            # i.e. we have a timezone
            return False

        values_int = self.asi8
        consider_values = values_int != iNaT
        reso = get_unit_from_dtype(self.dtype)
        ppd = periods_per_day(reso)

        # TODO: can we reuse is_date_array_normalized?  would need a skipna kwd
        #  (first attempt at this was less performant than this implementation)
        even_days = np.logical_and(consider_values, values_int % ppd != 0).sum() == 0
        return even_days


# -------------------------------------------------------------------
# Shared Constructor Helpers


def ensure_arraylike_for_datetimelike(
    data, copy: bool, cls_name: str
) -> tuple[ArrayLike, bool]:
    if not hasattr(data, "dtype"):
        # e.g. list, tuple
        if not isinstance(data, (list, tuple)) and np.ndim(data) == 0:
            # i.e. generator
            data = list(data)

        data = construct_1d_object_array_from_listlike(data)
        copy = False
    elif isinstance(data, ABCMultiIndex):
        raise TypeError(f"Cannot create a {cls_name} from a MultiIndex.")
    else:
        data = extract_array(data, extract_numpy=True)

    if isinstance(data, IntegerArray) or (
        isinstance(data, ArrowExtensionArray) and data.dtype.kind in "iu"
    ):
        data = data.to_numpy("int64", na_value=iNaT)
        copy = False
    elif isinstance(data, ArrowExtensionArray):
        data = data._maybe_convert_datelike_array()
        data = data.to_numpy()
        copy = False
    elif not isinstance(data, (np.ndarray, ExtensionArray)):
        # GH#24539 e.g. xarray, dask object
        data = np.asarray(data)

    elif isinstance(data, ABCCategorical):
        # GH#18664 preserve tz in going DTI->Categorical->DTI
        # TODO: cases where we need to do another pass through maybe_convert_dtype,
        #  e.g. the categories are timedelta64s
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False

    return data, copy


@overload
def validate_periods(periods: None) -> None:
    ...


@overload
def validate_periods(periods: int | float) -> int:
    ...


def validate_periods(periods: int | float | None) -> int | None:
    """
    If a `periods` argument is passed to the Datetime/Timedelta Array/Index
    constructor, cast it to an integer.

    Parameters
    ----------
    periods : None, float, int

    Returns
    -------
    periods : None or int

    Raises
    ------
    TypeError
        if periods is None, float, or int
    """
    if periods is not None:
        if lib.is_float(periods):
            warnings.warn(
                # GH#56036
                "Non-integer 'periods' in pd.date_range, pd.timedelta_range, "
                "pd.period_range, and pd.interval_range are deprecated and "
                "will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            periods = int(periods)
        elif not lib.is_integer(periods):
            raise TypeError(f"periods must be a number, got {periods}")
    return periods


def _validate_inferred_freq(
    freq: BaseOffset | None, inferred_freq: BaseOffset | None
) -> BaseOffset | None:
    """
    If the user passes a freq and another freq is inferred from passed data,
    require that they match.

    Parameters
    ----------
    freq : DateOffset or None
    inferred_freq : DateOffset or None

    Returns
    -------
    freq : DateOffset or None
    """
    if inferred_freq is not None:
        if freq is not None and freq != inferred_freq:
            raise ValueError(
                f"Inferred frequency {inferred_freq} from passed "
                "values does not conform to passed frequency "
                f"{freq.freqstr}"
            )
        if freq is None:
            freq = inferred_freq

    return freq


def dtype_to_unit(dtype: DatetimeTZDtype | np.dtype | ArrowDtype) -> str:
    """
    Return the unit str corresponding to the dtype's resolution.

    Parameters
    ----------
    dtype : DatetimeTZDtype or np.dtype
        If np.dtype, we assume it is a datetime64 dtype.

    Returns
    -------
    str
    """
    if isinstance(dtype, DatetimeTZDtype):
        return dtype.unit
    elif isinstance(dtype, ArrowDtype):
        if dtype.kind not in "mM":
            raise ValueError(f"{dtype=} does not have a resolution.")
        return dtype.pyarrow_dtype.unit
    return np.datetime_data(dtype)[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\categories\diagram_drawing.py ===
r"""
This module contains the functionality to arrange the nodes of a
diagram on an abstract grid, and then to produce a graphical
representation of the grid.

The currently supported back-ends are Xy-pic [Xypic].

Layout Algorithm
================

This section provides an overview of the algorithms implemented in
:class:`DiagramGrid` to lay out diagrams.

The first step of the algorithm is the removal composite and identity
morphisms which do not have properties in the supplied diagram.  The
premises and conclusions of the diagram are then merged.

The generic layout algorithm begins with the construction of the
"skeleton" of the diagram.  The skeleton is an undirected graph which
has the objects of the diagram as vertices and has an (undirected)
edge between each pair of objects between which there exist morphisms.
The direction of the morphisms does not matter at this stage.  The
skeleton also includes an edge between each pair of vertices `A` and
`C` such that there exists an object `B` which is connected via
a morphism to `A`, and via a morphism to `C`.

The skeleton constructed in this way has the property that every
object is a vertex of a triangle formed by three edges of the
skeleton.  This property lies at the base of the generic layout
algorithm.

After the skeleton has been constructed, the algorithm lists all
triangles which can be formed.  Note that some triangles will not have
all edges corresponding to morphisms which will actually be drawn.
Triangles which have only one edge or less which will actually be
drawn are immediately discarded.

The list of triangles is sorted according to the number of edges which
correspond to morphisms, then the triangle with the least number of such
edges is selected.  One of such edges is picked and the corresponding
objects are placed horizontally, on a grid.  This edge is recorded to
be in the fringe.  The algorithm then finds a "welding" of a triangle
to the fringe.  A welding is an edge in the fringe where a triangle
could be attached.  If the algorithm succeeds in finding such a
welding, it adds to the grid that vertex of the triangle which was not
yet included in any edge in the fringe and records the two new edges in
the fringe.  This process continues iteratively until all objects of
the diagram has been placed or until no more weldings can be found.

An edge is only removed from the fringe when a welding to this edge
has been found, and there is no room around this edge to place
another vertex.

When no more weldings can be found, but there are still triangles
left, the algorithm searches for a possibility of attaching one of the
remaining triangles to the existing structure by a vertex.  If such a
possibility is found, the corresponding edge of the found triangle is
placed in the found space and the iterative process of welding
triangles restarts.

When logical groups are supplied, each of these groups is laid out
independently.  Then a diagram is constructed in which groups are
objects and any two logical groups between which there exist morphisms
are connected via a morphism.  This diagram is laid out.  Finally,
the grid which includes all objects of the initial diagram is
constructed by replacing the cells which contain logical groups with
the corresponding laid out grids, and by correspondingly expanding the
rows and columns.

The sequential layout algorithm begins by constructing the
underlying undirected graph defined by the morphisms obtained after
simplifying premises and conclusions and merging them (see above).
The vertex with the minimal degree is then picked up and depth-first
search is started from it.  All objects which are located at distance
`n` from the root in the depth-first search tree, are positioned in
the `n`-th column of the resulting grid.  The sequential layout will
therefore attempt to lay the objects out along a line.

References
==========

.. [Xypic] https://xy-pic.sourceforge.net/

"""
from sympy.categories import (CompositeMorphism, IdentityMorphism,
                              NamedMorphism, Diagram)
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on

from itertools import chain


__doctest_requires__ = {('preview_diagram',): 'pyglet'}


class _GrowableGrid:
    """
    Holds a growable grid of objects.

    Explanation
    ===========

    It is possible to append or prepend a row or a column to the grid
    using the corresponding methods.  Prepending rows or columns has
    the effect of changing the coordinates of the already existing
    elements.

    This class currently represents a naive implementation of the
    functionality with little attempt at optimisation.
    """
    def __init__(self, width, height):
        self._width = width
        self._height = height

        self._array = [[None for j in range(width)] for i in range(height)]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __getitem__(self, i_j):
        """
        Returns the element located at in the i-th line and j-th
        column.
        """
        i, j = i_j
        return self._array[i][j]

    def __setitem__(self, i_j, newvalue):
        """
        Sets the element located at in the i-th line and j-th
        column.
        """
        i, j = i_j
        self._array[i][j] = newvalue

    def append_row(self):
        """
        Appends an empty row to the grid.
        """
        self._height += 1
        self._array.append([None for j in range(self._width)])

    def append_column(self):
        """
        Appends an empty column to the grid.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].append(None)

    def prepend_row(self):
        """
        Prepends the grid with an empty row.
        """
        self._height += 1
        self._array.insert(0, [None for j in range(self._width)])

    def prepend_column(self):
        """
        Prepends the grid with an empty column.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].insert(0, None)


class DiagramGrid:
    r"""
    Constructs and holds the fitting of the diagram into a grid.

    Explanation
    ===========

    The mission of this class is to analyse the structure of the
    supplied diagram and to place its objects on a grid such that,
    when the objects and the morphisms are actually drawn, the diagram
    would be "readable", in the sense that there will not be many
    intersections of moprhisms.  This class does not perform any
    actual drawing.  It does strive nevertheless to offer sufficient
    metadata to draw a diagram.

    Consider the following simple diagram.

    >>> from sympy.categories import Object, NamedMorphism
    >>> from sympy.categories import Diagram, DiagramGrid
    >>> from sympy import pprint
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g])

    The simplest way to have a diagram laid out is the following:

    >>> grid = DiagramGrid(diagram)
    >>> (grid.width, grid.height)
    (2, 2)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C

    Sometimes one sees the diagram as consisting of logical groups.
    One can advise ``DiagramGrid`` as to such groups by employing the
    ``groups`` keyword argument.

    Consider the following diagram:

    >>> D = Object("D")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])

    Lay it out with generic layout:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B  D
    <BLANKLINE>
       C

    Now, we can group the objects `A` and `D` to have them near one
    another:

    >>> grid = DiagramGrid(diagram, groups=[[A, D], B, C])
    >>> pprint(grid)
    B     C
    <BLANKLINE>
    A  D

    Note how the positioning of the other objects changes.

    Further indications can be supplied to the constructor of
    :class:`DiagramGrid` using keyword arguments.  The currently
    supported hints are explained in the following paragraphs.

    :class:`DiagramGrid` does not automatically guess which layout
    would suit the supplied diagram better.  Consider, for example,
    the following linear diagram:

    >>> E = Object("E")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(C, D, "h")
    >>> i = NamedMorphism(D, E, "i")
    >>> diagram = Diagram([f, g, h, i])

    When laid out with the generic layout, it does not get to look
    linear:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C  D
    <BLANKLINE>
          E

    To get it laid out in a line, use ``layout="sequential"``:

    >>> grid = DiagramGrid(diagram, layout="sequential")
    >>> pprint(grid)
    A  B  C  D  E

    One may sometimes need to transpose the resulting layout.  While
    this can always be done by hand, :class:`DiagramGrid` provides a
    hint for that purpose:

    >>> grid = DiagramGrid(diagram, layout="sequential", transpose=True)
    >>> pprint(grid)
    A
    <BLANKLINE>
    B
    <BLANKLINE>
    C
    <BLANKLINE>
    D
    <BLANKLINE>
    E

    Separate hints can also be provided for each group.  For an
    example, refer to ``tests/test_drawing.py``, and see the different
    ways in which the five lemma [FiveLemma] can be laid out.

    See Also
    ========

    Diagram

    References
    ==========

    .. [FiveLemma] https://en.wikipedia.org/wiki/Five_lemma
    """
    @staticmethod
    def _simplify_morphisms(morphisms):
        """
        Given a dictionary mapping morphisms to their properties,
        returns a new dictionary in which there are no morphisms which
        do not have properties, and which are compositions of other
        morphisms included in the dictionary.  Identities are dropped
        as well.
        """
        newmorphisms = {}
        for morphism, props in morphisms.items():
            if isinstance(morphism, CompositeMorphism) and not props:
                continue
            elif isinstance(morphism, IdentityMorphism):
                continue
            else:
                newmorphisms[morphism] = props
        return newmorphisms

    @staticmethod
    def _merge_premises_conclusions(premises, conclusions):
        """
        Given two dictionaries of morphisms and their properties,
        produces a single dictionary which includes elements from both
        dictionaries.  If a morphism has some properties in premises
        and also in conclusions, the properties in conclusions take
        priority.
        """
        return dict(chain(premises.items(), conclusions.items()))

    @staticmethod
    def _juxtapose_edges(edge1, edge2):
        """
        If ``edge1`` and ``edge2`` have precisely one common endpoint,
        returns an edge which would form a triangle with ``edge1`` and
        ``edge2``.

        If ``edge1`` and ``edge2`` do not have a common endpoint,
        returns ``None``.

        If ``edge1`` and ``edge`` are the same edge, returns ``None``.
        """
        intersection = edge1 & edge2
        if len(intersection) != 1:
            # The edges either have no common points or are equal.
            return None

        # The edges have a common endpoint.  Extract the different
        # endpoints and set up the new edge.
        return (edge1 - intersection) | (edge2 - intersection)

    @staticmethod
    def _add_edge_append(dictionary, edge, elem):
        """
        If ``edge`` is not in ``dictionary``, adds ``edge`` to the
        dictionary and sets its value to ``[elem]``.  Otherwise
        appends ``elem`` to the value of existing entry.

        Note that edges are undirected, thus `(A, B) = (B, A)`.
        """
        if edge in dictionary:
            dictionary[edge].append(elem)
        else:
            dictionary[edge] = [elem]

    @staticmethod
    def _build_skeleton(morphisms):
        """
        Creates a dictionary which maps edges to corresponding
        morphisms.  Thus for a morphism `f:A\rightarrow B`, the edge
        `(A, B)` will be associated with `f`.  This function also adds
        to the list those edges which are formed by juxtaposition of
        two edges already in the list.  These new edges are not
        associated with any morphism and are only added to assure that
        the diagram can be decomposed into triangles.
        """
        edges = {}
        # Create edges for morphisms.
        for morphism in morphisms:
            DiagramGrid._add_edge_append(
                edges, frozenset([morphism.domain, morphism.codomain]), morphism)

        # Create new edges by juxtaposing existing edges.
        edges1 = dict(edges)
        for w in edges1:
            for v in edges1:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv not in edges:
                    edges[wv] = []

        return edges

    @staticmethod
    def _list_triangles(edges):
        """
        Builds the set of triangles formed by the supplied edges.  The
        triangles are arbitrary and need not be commutative.  A
        triangle is a set that contains all three of its sides.
        """
        triangles = set()

        for w in edges:
            for v in edges:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv in edges:
                    triangles.add(frozenset([w, v, wv]))

        return triangles

    @staticmethod
    def _drop_redundant_triangles(triangles, skeleton):
        """
        Returns a list which contains only those triangles who have
        morphisms associated with at least two edges.
        """
        return [tri for tri in triangles
                if len([e for e in tri if skeleton[e]]) >= 2]

    @staticmethod
    def _morphism_length(morphism):
        """
        Returns the length of a morphism.  The length of a morphism is
        the number of components it consists of.  A non-composite
        morphism is of length 1.
        """
        if isinstance(morphism, CompositeMorphism):
            return len(morphism.components)
        else:
            return 1

    @staticmethod
    def _compute_triangle_min_sizes(triangles, edges):
        r"""
        Returns a dictionary mapping triangles to their minimal sizes.
        The minimal size of a triangle is the sum of maximal lengths
        of morphisms associated to the sides of the triangle.  The
        length of a morphism is the number of components it consists
        of.  A non-composite morphism is of length 1.

        Sorting triangles by this metric attempts to address two
        aspects of layout.  For triangles with only simple morphisms
        in the edge, this assures that triangles with all three edges
        visible will get typeset after triangles with less visible
        edges, which sometimes minimizes the necessity in diagonal
        arrows.  For triangles with composite morphisms in the edges,
        this assures that objects connected with shorter morphisms
        will be laid out first, resulting the visual proximity of
        those objects which are connected by shorter morphisms.
        """
        triangle_sizes = {}
        for triangle in triangles:
            size = 0
            for e in triangle:
                morphisms = edges[e]
                if morphisms:
                    size += max(DiagramGrid._morphism_length(m)
                                for m in morphisms)
            triangle_sizes[triangle] = size
        return triangle_sizes

    @staticmethod
    def _triangle_objects(triangle):
        """
        Given a triangle, returns the objects included in it.
        """
        # A triangle is a frozenset of three two-element frozensets
        # (the edges).  This chains the three edges together and
        # creates a frozenset from the iterator, thus producing a
        # frozenset of objects of the triangle.
        return frozenset(chain(*tuple(triangle)))

    @staticmethod
    def _other_vertex(triangle, edge):
        """
        Given a triangle and an edge of it, returns the vertex which
        opposes the edge.
        """
        # This gets the set of objects of the triangle and then
        # subtracts the set of objects employed in ``edge`` to get the
        # vertex opposite to ``edge``.
        return list(DiagramGrid._triangle_objects(triangle) - set(edge))[0]

    @staticmethod
    def _empty_point(pt, grid):
        """
        Checks if the cell at coordinates ``pt`` is either empty or
        out of the bounds of the grid.
        """
        if (pt[0] < 0) or (pt[1] < 0) or \
           (pt[0] >= grid.height) or (pt[1] >= grid.width):
            return True
        return grid[pt] is None

    @staticmethod
    def _put_object(coords, obj, grid, fringe):
        """
        Places an object at the coordinate ``cords`` in ``grid``,
        growing the grid and updating ``fringe``, if necessary.
        Returns (0, 0) if no row or column has been prepended, (1, 0)
        if a row was prepended, (0, 1) if a column was prepended and
        (1, 1) if both a column and a row were prepended.
        """
        (i, j) = coords
        offset = (0, 0)
        if i == -1:
            grid.prepend_row()
            i = 0
            offset = (1, 0)
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1 + 1, j1), (i2 + 1, j2))
        elif i == grid.height:
            grid.append_row()

        if j == -1:
            j = 0
            offset = (offset[0], 1)
            grid.prepend_column()
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1, j1 + 1), (i2, j2 + 1))
        elif j == grid.width:
            grid.append_column()

        grid[i, j] = obj
        return offset

    @staticmethod
    def _choose_target_cell(pt1, pt2, edge, obj, skeleton, grid):
        """
        Given two points, ``pt1`` and ``pt2``, and the welding edge
        ``edge``, chooses one of the two points to place the opposing
        vertex ``obj`` of the triangle.  If neither of this points
        fits, returns ``None``.
        """
        pt1_empty = DiagramGrid._empty_point(pt1, grid)
        pt2_empty = DiagramGrid._empty_point(pt2, grid)

        if pt1_empty and pt2_empty:
            # Both cells are empty.  Of these two, choose that cell
            # which will assure that a visible edge of the triangle
            # will be drawn perpendicularly to the current welding
            # edge.

            A = grid[edge[0]]

            if skeleton.get(frozenset([A, obj])):
                return pt1
            else:
                return pt2
        if pt1_empty:
            return pt1
        elif pt2_empty:
            return pt2
        else:
            return None

    @staticmethod
    def _find_triangle_to_weld(triangles, fringe, grid):
        """
        Finds, if possible, a triangle and an edge in the ``fringe`` to
        which the triangle could be attached.  Returns the tuple
        containing the triangle and the index of the corresponding
        edge in the ``fringe``.

        This function relies on the fact that objects are unique in
        the diagram.
        """
        for triangle in triangles:
            for (a, b) in fringe:
                if frozenset([grid[a], grid[b]]) in triangle:
                    return (triangle, (a, b))
        return None

    @staticmethod
    def _weld_triangle(tri, welding_edge, fringe, grid, skeleton):
        """
        If possible, welds the triangle ``tri`` to ``fringe`` and
        returns ``False``.  If this method encounters a degenerate
        situation in the fringe and corrects it such that a restart of
        the search is required, it returns ``True`` (which means that
        a restart in finding triangle weldings is required).

        A degenerate situation is a situation when an edge listed in
        the fringe does not belong to the visual boundary of the
        diagram.
        """
        a, b = welding_edge
        target_cell = None

        obj = DiagramGrid._other_vertex(tri, (grid[a], grid[b]))

        # We now have a triangle and an edge where it can be welded to
        # the fringe.  Decide where to place the other vertex of the
        # triangle and check for degenerate situations en route.

        if (abs(a[0] - b[0]) == 1) and (abs(a[1] - b[1]) == 1):
            # A diagonal edge.
            target_cell = (a[0], b[1])
            if grid[target_cell]:
                # That cell is already occupied.
                target_cell = (b[0], a[1])

                if grid[target_cell]:
                    # Degenerate situation, this edge is not
                    # on the actual fringe.  Correct the
                    # fringe and go on.
                    fringe.remove((a, b))
                    return True
        elif a[0] == b[0]:
            # A horizontal edge.  We first attempt to build the
            # triangle in the downward direction.

            down_left = a[0] + 1, a[1]
            down_right = a[0] + 1, b[1]

            target_cell = DiagramGrid._choose_target_cell(
                down_left, down_right, (a, b), obj, skeleton, grid)

            if not target_cell:
                # No room below this edge.  Check above.
                up_left = a[0] - 1, a[1]
                up_right = a[0] - 1, b[1]

                target_cell = DiagramGrid._choose_target_cell(
                    up_left, up_right, (a, b), obj, skeleton, grid)

                if not target_cell:
                    # This edge is not in the fringe, remove it
                    # and restart.
                    fringe.remove((a, b))
                    return True
        elif a[1] == b[1]:
            # A vertical edge.  We will attempt to place the other
            # vertex of the triangle to the right of this edge.
            right_up = a[0], a[1] + 1
            right_down = b[0], a[1] + 1

            target_cell = DiagramGrid._choose_target_cell(
                right_up, right_down, (a, b), obj, skeleton, grid)

            if not target_cell:
                # No room to the left.  See what's to the right.
                left_up = a[0], a[1] - 1
                left_down = b[0], a[1] - 1

                target_cell = DiagramGrid._choose_target_cell(
                    left_up, left_down, (a, b), obj, skeleton, grid)

                if not target_cell:
                    # This edge is not in the fringe, remove it
                    # and restart.
                    fringe.remove((a, b))
                    return True

        # We now know where to place the other vertex of the
        # triangle.
        offset = DiagramGrid._put_object(target_cell, obj, grid, fringe)

        # Take care of the displacement of coordinates if a row or
        # a column was prepended.
        target_cell = (target_cell[0] + offset[0],
                       target_cell[1] + offset[1])
        a = (a[0] + offset[0], a[1] + offset[1])
        b = (b[0] + offset[0], b[1] + offset[1])

        fringe.extend([(a, target_cell), (b, target_cell)])

        # No restart is required.
        return False

    @staticmethod
    def _triangle_key(tri, triangle_sizes):
        """
        Returns a key for the supplied triangle.  It should be the
        same independently of the hash randomisation.
        """
        objects = sorted(
            DiagramGrid._triangle_objects(tri), key=default_sort_key)
        return (triangle_sizes[tri], default_sort_key(objects))

    @staticmethod
    def _pick_root_edge(tri, skeleton):
        """
        For a given triangle always picks the same root edge.  The
        root edge is the edge that will be placed first on the grid.
        """
        candidates = [sorted(e, key=default_sort_key)
                      for e in tri if skeleton[e]]
        sorted_candidates = sorted(candidates, key=default_sort_key)
        # Don't forget to assure the proper ordering of the vertices
        # in this edge.
        return tuple(sorted(sorted_candidates[0], key=default_sort_key))

    @staticmethod
    def _drop_irrelevant_triangles(triangles, placed_objects):
        """
        Returns only those triangles whose set of objects is not
        completely included in ``placed_objects``.
        """
        return [tri for tri in triangles if not placed_objects.issuperset(
            DiagramGrid._triangle_objects(tri))]

    @staticmethod
    def _grow_pseudopod(triangles, fringe, grid, skeleton, placed_objects):
        """
        Starting from an object in the existing structure on the ``grid``,
        adds an edge to which a triangle from ``triangles`` could be
        welded.  If this method has found a way to do so, it returns
        the object it has just added.

        This method should be applied when ``_weld_triangle`` cannot
        find weldings any more.
        """
        for i in range(grid.height):
            for j in range(grid.width):
                obj = grid[i, j]
                if not obj:
                    continue

                # Here we need to choose a triangle which has only
                # ``obj`` in common with the existing structure.  The
                # situations when this is not possible should be
                # handled elsewhere.

                def good_triangle(tri):
                    objs = DiagramGrid._triangle_objects(tri)
                    return obj in objs and \
                        placed_objects & (objs - {obj}) == set()

                tris = [tri for tri in triangles if good_triangle(tri)]
                if not tris:
                    # This object is not interesting.
                    continue

                # Pick the "simplest" of the triangles which could be
                # attached.  Remember that the list of triangles is
                # sorted according to their "simplicity" (see
                # _compute_triangle_min_sizes for the metric).
                #
                # Note that ``tris`` are sequentially built from
                # ``triangles``, so we don't have to worry about hash
                # randomisation.
                tri = tris[0]

                # We have found a triangle which could be attached to
                # the existing structure by a vertex.

                candidates = sorted([e for e in tri if skeleton[e]],
                                    key=lambda e: FiniteSet(*e).sort_key())
                edges = [e for e in candidates if obj in e]

                # Note that a meaningful edge (i.e., and edge that is
                # associated with a morphism) containing ``obj``
                # always exists.  That's because all triangles are
                # guaranteed to have at least two meaningful edges.
                # See _drop_redundant_triangles.

                # Get the object at the other end of the edge.
                edge = edges[0]
                other_obj = tuple(edge - frozenset([obj]))[0]

                # Now check for free directions.  When checking for
                # free directions, prefer the horizontal and vertical
                # directions.
                neighbours = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1),
                              (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]

                for pt in neighbours:
                    if DiagramGrid._empty_point(pt, grid):
                        # We have a found a place to grow the
                        # pseudopod into.
                        offset = DiagramGrid._put_object(
                            pt, other_obj, grid, fringe)

                        i += offset[0]
                        j += offset[1]
                        pt = (pt[0] + offset[0], pt[1] + offset[1])
                        fringe.append(((i, j), pt))

                        return other_obj

        # This diagram is actually cooler that I can handle.  Fail cowardly.
        return None

    @staticmethod
    def _handle_groups(diagram, groups, merged_morphisms, hints):
        """
        Given the slightly preprocessed morphisms of the diagram,
        produces a grid laid out according to ``groups``.

        If a group has hints, it is laid out with those hints only,
        without any influence from ``hints``.  Otherwise, it is laid
        out with ``hints``.
        """
        def lay_out_group(group, local_hints):
            """
            If ``group`` is a set of objects, uses a ``DiagramGrid``
            to lay it out and returns the grid.  Otherwise returns the
            object (i.e., ``group``).  If ``local_hints`` is not
            empty, it is supplied to ``DiagramGrid`` as the dictionary
            of hints.  Otherwise, the ``hints`` argument of
            ``_handle_groups`` is used.
            """
            if isinstance(group, FiniteSet):
                # Set up the corresponding object-to-group
                # mappings.
                for obj in group:
                    obj_groups[obj] = group

                # Lay out the current group.
                if local_hints:
                    groups_grids[group] = DiagramGrid(
                        diagram.subdiagram_from_objects(group), **local_hints)
                else:
                    groups_grids[group] = DiagramGrid(
                        diagram.subdiagram_from_objects(group), **hints)
            else:
                obj_groups[group] = group

        def group_to_finiteset(group):
            """
            Converts ``group`` to a :class:``FiniteSet`` if it is an
            iterable.
            """
            if iterable(group):
                return FiniteSet(*group)
            else:
                return group

        obj_groups = {}
        groups_grids = {}

        # We would like to support various containers to represent
        # groups.  To achieve that, before laying each group out, it
        # should be converted to a FiniteSet, because that is what the
        # following code expects.

        if isinstance(groups, (dict, Dict)):
            finiteset_groups = {}
            for group, local_hints in groups.items():
                finiteset_group = group_to_finiteset(group)
                finiteset_groups[finiteset_group] = local_hints
                lay_out_group(group, local_hints)
            groups = finiteset_groups
        else:
            finiteset_groups = []
            for group in groups:
                finiteset_group = group_to_finiteset(group)
                finiteset_groups.append(finiteset_group)
                lay_out_group(finiteset_group, None)
            groups = finiteset_groups

        new_morphisms = []
        for morphism in merged_morphisms:
            dom = obj_groups[morphism.domain]
            cod = obj_groups[morphism.codomain]
            # Note that we are not really interested in morphisms
            # which do not employ two different groups, because
            # these do not influence the layout.
            if dom != cod:
                # These are essentially unnamed morphisms; they are
                # not going to mess in the final layout.  By giving
                # them the same names, we avoid unnecessary
                # duplicates.
                new_morphisms.append(NamedMorphism(dom, cod, "dummy"))

        # Lay out the new diagram.  Since these are dummy morphisms,
        # properties and conclusions are irrelevant.
        top_grid = DiagramGrid(Diagram(new_morphisms))

        # We now have to substitute the groups with the corresponding
        # grids, laid out at the beginning of this function.  Compute
        # the size of each row and column in the grid, so that all
        # nested grids fit.

        def group_size(group):
            """
            For the supplied group (or object, eventually), returns
            the size of the cell that will hold this group (object).
            """
            if group in groups_grids:
                grid = groups_grids[group]
                return (grid.height, grid.width)
            else:
                return (1, 1)

        row_heights = [max(group_size(top_grid[i, j])[0]
                           for j in range(top_grid.width))
                       for i in range(top_grid.height)]

        column_widths = [max(group_size(top_grid[i, j])[1]
                             for i in range(top_grid.height))
                         for j in range(top_grid.width)]

        grid = _GrowableGrid(sum(column_widths), sum(row_heights))

        real_row = 0
        real_column = 0
        for logical_row in range(top_grid.height):
            for logical_column in range(top_grid.width):
                obj = top_grid[logical_row, logical_column]

                if obj in groups_grids:
                    # This is a group.  Copy the corresponding grid in
                    # place.
                    local_grid = groups_grids[obj]
                    for i in range(local_grid.height):
                        for j in range(local_grid.width):
                            grid[real_row + i,
                                real_column + j] = local_grid[i, j]
                else:
                    # This is an object.  Just put it there.
                    grid[real_row, real_column] = obj

                real_column += column_widths[logical_column]
            real_column = 0
            real_row += row_heights[logical_row]

        return grid

    @staticmethod
    def _generic_layout(diagram, merged_morphisms):
        """
        Produces the generic layout for the supplied diagram.
        """
        all_objects = set(diagram.objects)
        if len(all_objects) == 1:
            # There only one object in the diagram, just put in on 1x1
            # grid.
            grid = _GrowableGrid(1, 1)
            grid[0, 0] = tuple(all_objects)[0]
            return grid

        skeleton = DiagramGrid._build_skeleton(merged_morphisms)

        grid = _GrowableGrid(2, 1)

        if len(skeleton) == 1:
            # This diagram contains only one morphism.  Draw it
            # horizontally.
            objects = sorted(all_objects, key=default_sort_key)
            grid[0, 0] = objects[0]
            grid[0, 1] = objects[1]

            return grid

        triangles = DiagramGrid._list_triangles(skeleton)
        triangles = DiagramGrid._drop_redundant_triangles(triangles, skeleton)
        triangle_sizes = DiagramGrid._compute_triangle_min_sizes(
            triangles, skeleton)

        triangles = sorted(triangles, key=lambda tri:
                           DiagramGrid._triangle_key(tri, triangle_sizes))

        # Place the first edge on the grid.
        root_edge = DiagramGrid._pick_root_edge(triangles[0], skeleton)
        grid[0, 0], grid[0, 1] = root_edge
        fringe = [((0, 0), (0, 1))]

        # Record which objects we now have on the grid.
        placed_objects = set(root_edge)

        while placed_objects != all_objects:
            welding = DiagramGrid._find_triangle_to_weld(
                triangles, fringe, grid)

            if welding:
                (triangle, welding_edge) = welding

                restart_required = DiagramGrid._weld_triangle(
                    triangle, welding_edge, fringe, grid, skeleton)
                if restart_required:
                    continue

                placed_objects.update(
                    DiagramGrid._triangle_objects(triangle))
            else:
                # No more weldings found.  Try to attach triangles by
                # vertices.
                new_obj = DiagramGrid._grow_pseudopod(
                    triangles, fringe, grid, skeleton, placed_objects)

                if not new_obj:
                    # No more triangles can be attached, not even by
                    # the edge.  We will set up a new diagram out of
                    # what has been left, laid it out independently,
                    # and then attach it to this one.

                    remaining_objects = all_objects - placed_objects

                    remaining_diagram = diagram.subdiagram_from_objects(
                        FiniteSet(*remaining_objects))
                    remaining_grid = DiagramGrid(remaining_diagram)

                    # Now, let's glue ``remaining_grid`` to ``grid``.
                    final_width = grid.width + remaining_grid.width
                    final_height = max(grid.height, remaining_grid.height)
                    final_grid = _GrowableGrid(final_width, final_height)

                    for i in range(grid.width):
                        for j in range(grid.height):
                            final_grid[i, j] = grid[i, j]

                    start_j = grid.width
                    for i in range(remaining_grid.height):
                        for j in range(remaining_grid.width):
                            final_grid[i, start_j + j] = remaining_grid[i, j]

                    return final_grid

                placed_objects.add(new_obj)

            triangles = DiagramGrid._drop_irrelevant_triangles(
                triangles, placed_objects)

        return grid

    @staticmethod
    def _get_undirected_graph(objects, merged_morphisms):
        """
        Given the objects and the relevant morphisms of a diagram,
        returns the adjacency lists of the underlying undirected
        graph.
        """
        adjlists = {obj: [] for obj in objects}

        for morphism in merged_morphisms:
            adjlists[morphism.domain].append(morphism.codomain)
            adjlists[morphism.codomain].append(morphism.domain)

        # Assure that the objects in the adjacency list are always in
        # the same order.
        for obj in adjlists.keys():
            adjlists[obj].sort(key=default_sort_key)

        return adjlists

    @staticmethod
    def _sequential_layout(diagram, merged_morphisms):
        r"""
        Lays out the diagram in "sequential" layout.  This method
        will attempt to produce a result as close to a line as
        possible.  For linear diagrams, the result will actually be a
        line.
        """
        objects = diagram.objects
        sorted_objects = sorted(objects, key=default_sort_key)

        # Set up the adjacency lists of the underlying undirected
        # graph of ``merged_morphisms``.
        adjlists = DiagramGrid._get_undirected_graph(objects, merged_morphisms)

        root = min(sorted_objects, key=lambda x: len(adjlists[x]))
        grid = _GrowableGrid(1, 1)
        grid[0, 0] = root

        placed_objects = {root}

        def place_objects(pt, placed_objects):
            """
            Does depth-first search in the underlying graph of the
            diagram and places the objects en route.
            """
            # We will start placing new objects from here.
            new_pt = (pt[0], pt[1] + 1)

            for adjacent_obj in adjlists[grid[pt]]:
                if adjacent_obj in placed_objects:
                    # This object has already been placed.
                    continue

                DiagramGrid._put_object(new_pt, adjacent_obj, grid, [])
                placed_objects.add(adjacent_obj)
                placed_objects.update(place_objects(new_pt, placed_objects))

                new_pt = (new_pt[0] + 1, new_pt[1])

            return placed_objects

        place_objects((0, 0), placed_objects)

        return grid

    @staticmethod
    def _drop_inessential_morphisms(merged_morphisms):
        r"""
        Removes those morphisms which should appear in the diagram,
        but which have no relevance to object layout.

        Currently this removes "loop" morphisms: the non-identity
        morphisms with the same domains and codomains.
        """
        morphisms = [m for m in merged_morphisms if m.domain != m.codomain]
        return morphisms

    @staticmethod
    def _get_connected_components(objects, merged_morphisms):
        """
        Given a container of morphisms, returns a list of connected
        components formed by these morphisms.  A connected component
        is represented by a diagram consisting of the corresponding
        morphisms.
        """
        component_index = {}
        for o in objects:
            component_index[o] = None

        # Get the underlying undirected graph of the diagram.
        adjlist = DiagramGrid._get_undirected_graph(objects, merged_morphisms)

        def traverse_component(object, current_index):
            """
            Does a depth-first search traversal of the component
            containing ``object``.
            """
            component_index[object] = current_index
            for o in adjlist[object]:
                if component_index[o] is None:
                    traverse_component(o, current_index)

        # Traverse all components.
        current_index = 0
        for o in adjlist:
            if component_index[o] is None:
                traverse_component(o, current_index)
                current_index += 1

        # List the objects of the components.
        component_objects = [[] for i in range(current_index)]
        for o, idx in component_index.items():
            component_objects[idx].append(o)

        # Finally, list the morphisms belonging to each component.
        #
        # Note: If some objects are isolated, they will not get any
        # morphisms at this stage, and since the layout algorithm
        # relies, we are essentially going to lose this object.
        # Therefore, check if there are isolated objects and, for each
        # of them, provide the trivial identity morphism.  It will get
        # discarded later, but the object will be there.

        component_morphisms = []
        for component in component_objects:
            current_morphisms = {}
            for m in merged_morphisms:
                if (m.domain in component) and (m.codomain in component):
                    current_morphisms[m] = merged_morphisms[m]

            if len(component) == 1:
                # Let's add an identity morphism, for the sake of
                # surely having morphisms in this component.
                current_morphisms[IdentityMorphism(component[0])] = FiniteSet()

            component_morphisms.append(Diagram(current_morphisms))

        return component_morphisms

    def __init__(self, diagram, groups=None, **hints):
        premises = DiagramGrid._simplify_morphisms(diagram.premises)
        conclusions = DiagramGrid._simplify_morphisms(diagram.conclusions)
        all_merged_morphisms = DiagramGrid._merge_premises_conclusions(
            premises, conclusions)
        merged_morphisms = DiagramGrid._drop_inessential_morphisms(
            all_merged_morphisms)

        # Store the merged morphisms for later use.
        self._morphisms = all_merged_morphisms

        components = DiagramGrid._get_connected_components(
            diagram.objects, all_merged_morphisms)

        if groups and (groups != diagram.objects):
            # Lay out the diagram according to the groups.
            self._grid = DiagramGrid._handle_groups(
                diagram, groups, merged_morphisms, hints)
        elif len(components) > 1:
            # Note that we check for connectedness _before_ checking
            # the layout hints because the layout strategies don't
            # know how to deal with disconnected diagrams.

            # The diagram is disconnected.  Lay out the components
            # independently.
            grids = []

            # Sort the components to eventually get the grids arranged
            # in a fixed, hash-independent order.
            components = sorted(components, key=default_sort_key)

            for component in components:
                grid = DiagramGrid(component, **hints)
                grids.append(grid)

            # Throw the grids together, in a line.
            total_width = sum(g.width for g in grids)
            total_height = max(g.height for g in grids)

            grid = _GrowableGrid(total_width, total_height)
            start_j = 0
            for g in grids:
                for i in range(g.height):
                    for j in range(g.width):
                        grid[i, start_j + j] = g[i, j]

                start_j += g.width

            self._grid = grid
        elif "layout" in hints:
            if hints["layout"] == "sequential":
                self._grid = DiagramGrid._sequential_layout(
                    diagram, merged_morphisms)
        else:
            self._grid = DiagramGrid._generic_layout(diagram, merged_morphisms)

        if hints.get("transpose"):
            # Transpose the resulting grid.
            grid = _GrowableGrid(self._grid.height, self._grid.width)
            for i in range(self._grid.height):
                for j in range(self._grid.width):
                    grid[j, i] = self._grid[i, j]
            self._grid = grid

    @property
    def width(self):
        """
        Returns the number of columns in this diagram layout.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.width
        2

        """
        return self._grid.width

    @property
    def height(self):
        """
        Returns the number of rows in this diagram layout.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.height
        2

        """
        return self._grid.height

    def __getitem__(self, i_j):
        """
        Returns the object placed in the row ``i`` and column ``j``.
        The indices are 0-based.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> (grid[0, 0], grid[0, 1])
        (Object("A"), Object("B"))
        >>> (grid[1, 0], grid[1, 1])
        (None, Object("C"))

        """
        i, j = i_j
        return self._grid[i, j]

    @property
    def morphisms(self):
        """
        Returns those morphisms (and their properties) which are
        sufficiently meaningful to be drawn.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.morphisms
        {NamedMorphism(Object("A"), Object("B"), "f"): EmptySet,
        NamedMorphism(Object("B"), Object("C"), "g"): EmptySet}

        """
        return self._morphisms

    def __str__(self):
        """
        Produces a string representation of this class.

        This method returns a string representation of the underlying
        list of lists of objects.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> print(grid)
        [[Object("A"), Object("B")],
        [None, Object("C")]]

        """
        return repr(self._grid._array)


class ArrowStringDescription:
    r"""
    Stores the information necessary for producing an Xy-pic
    description of an arrow.

    The principal goal of this class is to abstract away the string
    representation of an arrow and to also provide the functionality
    to produce the actual Xy-pic string.

    ``unit`` sets the unit which will be used to specify the amount of
    curving and other distances.  ``horizontal_direction`` should be a
    string of ``"r"`` or ``"l"`` specifying the horizontal offset of the
    target cell of the arrow relatively to the current one.
    ``vertical_direction`` should  specify the vertical offset using a
    series of either ``"d"`` or ``"u"``.  ``label_position`` should be
    either ``"^"``, ``"_"``,  or ``"|"`` to specify that the label should
    be positioned above the arrow, below the arrow or just over the arrow,
    in a break.  Note that the notions "above" and "below" are relative
    to arrow direction.  ``label`` stores the morphism label.

    This works as follows (disregard the yet unexplained arguments):

    >>> from sympy.categories.diagram_drawing import ArrowStringDescription
    >>> astr = ArrowStringDescription(
    ... unit="mm", curving=None, curving_amount=None,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> print(str(astr))
    \ar[dr]_{f}

    ``curving`` should be one of ``"^"``, ``"_"`` to specify in which
    direction the arrow is going to curve. ``curving_amount`` is a number
    describing how many ``unit``'s the morphism is going to curve:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> print(str(astr))
    \ar@/^12mm/[dr]_{f}

    ``looping_start`` and ``looping_end`` are currently only used for
    loop morphisms, those which have the same domain and codomain.
    These two attributes should store a valid Xy-pic direction and
    specify, correspondingly, the direction the arrow gets out into
    and the direction the arrow gets back from:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving=None, curving_amount=None,
    ... looping_start="u", looping_end="l", horizontal_direction="",
    ... vertical_direction="", label_position="_", label="f")
    >>> print(str(astr))
    \ar@(u,l)[]_{f}

    ``label_displacement`` controls how far the arrow label is from
    the ends of the arrow.  For example, to position the arrow label
    near the arrow head, use ">":

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> astr.label_displacement = ">"
    >>> print(str(astr))
    \ar@/^12mm/[dr]_>{f}

    Finally, ``arrow_style`` is used to specify the arrow style.  To
    get a dashed arrow, for example, use "{-->}" as arrow style:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> astr.arrow_style = "{-->}"
    >>> print(str(astr))
    \ar@/^12mm/@{-->}[dr]_{f}

    Notes
    =====

    Instances of :class:`ArrowStringDescription` will be constructed
    by :class:`XypicDiagramDrawer` and provided for further use in
    formatters.  The user is not expected to construct instances of
    :class:`ArrowStringDescription` themselves.

    To be able to properly utilise this class, the reader is encouraged
    to checkout the Xy-pic user guide, available at [Xypic].

    See Also
    ========

    XypicDiagramDrawer

    References
    ==========

    .. [Xypic] https://xy-pic.sourceforge.net/
    """
    def __init__(self, unit, curving, curving_amount, looping_start,
                 looping_end, horizontal_direction, vertical_direction,
                 label_position, label):
        self.unit = unit
        self.curving = curving
        self.curving_amount = curving_amount
        self.looping_start = looping_start
        self.looping_end = looping_end
        self.horizontal_direction = horizontal_direction
        self.vertical_direction = vertical_direction
        self.label_position = label_position
        self.label = label

        self.label_displacement = ""
        self.arrow_style = ""

        # This flag shows that the position of the label of this
        # morphism was set while typesetting a curved morphism and
        # should not be modified later.
        self.forced_label_position = False

    def __str__(self):
        if self.curving:
            curving_str = "@/%s%d%s/" % (self.curving, self.curving_amount,
                                         self.unit)
        else:
            curving_str = ""

        if self.looping_start and self.looping_end:
            looping_str = "@(%s,%s)" % (self.looping_start, self.looping_end)
        else:
            looping_str = ""

        if self.arrow_style:

            style_str = "@" + self.arrow_style
        else:
            style_str = ""

        return "\\ar%s%s%s[%s%s]%s%s{%s}" % \
               (curving_str, looping_str, style_str, self.horizontal_direction,
                self.vertical_direction, self.label_position,
                self.label_displacement, self.label)


class XypicDiagramDrawer:
    r"""
    Given a :class:`~.Diagram` and the corresponding
    :class:`DiagramGrid`, produces the Xy-pic representation of the
    diagram.

    The most important method in this class is ``draw``.  Consider the
    following triangle diagram:

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g], {g * f: "unique"})

    To draw this diagram, its objects need to be laid out with a
    :class:`DiagramGrid`::

    >>> grid = DiagramGrid(diagram)

    Finally, the drawing:

    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    For further details see the docstring of this method.

    To control the appearance of the arrows, formatters are used.  The
    dictionary ``arrow_formatters`` maps morphisms to formatter
    functions.  A formatter is accepts an
    :class:`ArrowStringDescription` and is allowed to modify any of
    the arrow properties exposed thereby.  For example, to have all
    morphisms with the property ``unique`` appear as dashed arrows,
    and to have their names prepended with `\exists !`, the following
    should be done:

    >>> def formatter(astr):
    ...   astr.label = r"\exists !" + astr.label
    ...   astr.arrow_style = "{-->}"
    >>> drawer.arrow_formatters["unique"] = formatter
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar@{-->}[d]_{\exists !g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    To modify the appearance of all arrows in the diagram, set
    ``default_arrow_formatter``.  For example, to place all morphism
    labels a little bit farther from the arrow head so that they look
    more centred, do as follows:

    >>> def default_formatter(astr):
    ...   astr.label_displacement = "(0.45)"
    >>> drawer.default_arrow_formatter = default_formatter
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar@{-->}[d]_(0.45){\exists !g\circ f} \ar[r]^(0.45){f} & B \ar[ld]^(0.45){g} \\
    C &
    }

    In some diagrams some morphisms are drawn as curved arrows.
    Consider the following diagram:

    >>> D = Object("D")
    >>> E = Object("E")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])
    >>> grid = DiagramGrid(diagram)
    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_3mm/[ll]_{h} \\
    & C &
    }

    To control how far the morphisms are curved by default, one can
    use the ``unit`` and ``default_curving_amount`` attributes:

    >>> drawer.unit = "cm"
    >>> drawer.default_curving_amount = 1
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_1cm/[ll]_{h} \\
    & C &
    }

    In some diagrams, there are multiple curved morphisms between the
    same two objects.  To control by how much the curving changes
    between two such successive morphisms, use
    ``default_curving_step``:

    >>> drawer.default_curving_step = 1
    >>> h1 = NamedMorphism(A, D, "h1")
    >>> diagram = Diagram([f, g, h, k, h1])
    >>> grid = DiagramGrid(diagram)
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} \ar@/^1cm/[rr]^{h_{1}} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_2cm/[ll]_{h} \\
    & C &
    }

    The default value of ``default_curving_step`` is 4 units.

    See Also
    ========

    draw, ArrowStringDescription
    """
    def __init__(self):
        self.unit = "mm"
        self.default_curving_amount = 3
        self.default_curving_step = 4

        # This dictionary maps properties to the corresponding arrow
        # formatters.
        self.arrow_formatters = {}

        # This is the default arrow formatter which will be applied to
        # each arrow independently of its properties.
        self.default_arrow_formatter = None

    @staticmethod
    def _process_loop_morphism(i, j, grid, morphisms_str_info, object_coords):
        """
        Produces the information required for constructing the string
        representation of a loop morphism.  This function is invoked
        from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        curving = ""
        label_pos = "^"
        looping_start = ""
        looping_end = ""

        # This is a loop morphism.  Count how many morphisms stick
        # in each of the four quadrants.  Note that straight
        # vertical and horizontal morphisms count in two quadrants
        # at the same time (i.e., a morphism going up counts both
        # in the first and the second quadrants).

        # The usual numbering (counterclockwise) of quadrants
        # applies.
        quadrant = [0, 0, 0, 0]

        obj = grid[i, j]

        for m, m_str_info in morphisms_str_info.items():
            if (m.domain == obj) and (m.codomain == obj):
                # That's another loop morphism.  Check how it
                # loops and mark the corresponding quadrants as
                # busy.
                (l_s, l_e) = (m_str_info.looping_start, m_str_info.looping_end)

                if (l_s, l_e) == ("r", "u"):
                    quadrant[0] += 1
                elif (l_s, l_e) == ("u", "l"):
                    quadrant[1] += 1
                elif (l_s, l_e) == ("l", "d"):
                    quadrant[2] += 1
                elif (l_s, l_e) == ("d", "r"):
                    quadrant[3] += 1

                continue
            if m.domain == obj:
                (end_i, end_j) = object_coords[m.codomain]
                goes_out = True
            elif m.codomain == obj:
                (end_i, end_j) = object_coords[m.domain]
                goes_out = False
            else:
                continue

            d_i = end_i - i
            d_j = end_j - j
            m_curving = m_str_info.curving

            if (d_i != 0) and (d_j != 0):
                # This is really a diagonal morphism.  Detect the
                # quadrant.
                if (d_i > 0) and (d_j > 0):
                    quadrant[0] += 1
                elif (d_i > 0) and (d_j < 0):
                    quadrant[1] += 1
                elif (d_i < 0) and (d_j < 0):
                    quadrant[2] += 1
                elif (d_i < 0) and (d_j > 0):
                    quadrant[3] += 1
            elif d_i == 0:
                # Knowing where the other end of the morphism is
                # and which way it goes, we now have to decide
                # which quadrant is now the upper one and which is
                # the lower one.
                if d_j > 0:
                    if goes_out:
                        upper_quadrant = 0
                        lower_quadrant = 3
                    else:
                        upper_quadrant = 3
                        lower_quadrant = 0
                else:
                    if goes_out:
                        upper_quadrant = 2
                        lower_quadrant = 1
                    else:
                        upper_quadrant = 1
                        lower_quadrant = 2

                if m_curving:
                    if m_curving == "^":
                        quadrant[upper_quadrant] += 1
                    elif m_curving == "_":
                        quadrant[lower_quadrant] += 1
                else:
                    # This morphism counts in both upper and lower
                    # quadrants.
                    quadrant[upper_quadrant] += 1
                    quadrant[lower_quadrant] += 1
            elif d_j == 0:
                # Knowing where the other end of the morphism is
                # and which way it goes, we now have to decide
                # which quadrant is now the left one and which is
                # the right one.
                if d_i < 0:
                    if goes_out:
                        left_quadrant = 1
                        right_quadrant = 0
                    else:
                        left_quadrant = 0
                        right_quadrant = 1
                else:
                    if goes_out:
                        left_quadrant = 3
                        right_quadrant = 2
                    else:
                        left_quadrant = 2
                        right_quadrant = 3

                if m_curving:
                    if m_curving == "^":
                        quadrant[left_quadrant] += 1
                    elif m_curving == "_":
                        quadrant[right_quadrant] += 1
                else:
                    # This morphism counts in both upper and lower
                    # quadrants.
                    quadrant[left_quadrant] += 1
                    quadrant[right_quadrant] += 1

        # Pick the freest quadrant to curve our morphism into.
        freest_quadrant = 0
        for i in range(4):
            if quadrant[i] < quadrant[freest_quadrant]:
                freest_quadrant = i

        # Now set up proper looping.
        (looping_start, looping_end) = [("r", "u"), ("u", "l"), ("l", "d"),
                                        ("d", "r")][freest_quadrant]

        return (curving, label_pos, looping_start, looping_end)

    @staticmethod
    def _process_horizontal_morphism(i, j, target_j, grid, morphisms_str_info,
                                     object_coords):
        """
        Produces the information required for constructing the string
        representation of a horizontal morphism.  This function is
        invoked from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        # The arrow is horizontal.  Check if it goes from left to
        # right (``backwards == False``) or from right to left
        # (``backwards == True``).
        backwards = False
        start = j
        end = target_j
        if end < start:
            (start, end) = (end, start)
            backwards = True

        # Let's see which objects are there between ``start`` and
        # ``end``, and then count how many morphisms stick out
        # upwards, and how many stick out downwards.
        #
        # For example, consider the situation:
        #
        #    B1 C1
        #    |  |
        # A--B--C--D
        #    |
        #    B2
        #
        # Between the objects `A` and `D` there are two objects:
        # `B` and `C`.  Further, there are two morphisms which
        # stick out upward (the ones between `B1` and `B` and
        # between `C` and `C1`) and one morphism which sticks out
        # downward (the one between `B and `B2`).
        #
        # We need this information to decide how to curve the
        # arrow between `A` and `D`.  First of all, since there
        # are two objects between `A` and `D``, we must curve the
        # arrow.  Then, we will have it curve downward, because
        # there is more space (less morphisms stick out downward
        # than upward).
        up = []
        down = []
        straight_horizontal = []
        for k in range(start + 1, end):
            obj = grid[i, k]
            if not obj:
                continue

            for m in morphisms_str_info:
                if m.domain == obj:
                    (end_i, end_j) = object_coords[m.codomain]
                elif m.codomain == obj:
                    (end_i, end_j) = object_coords[m.domain]
                else:
                    continue

                if end_i > i:
                    down.append(m)
                elif end_i < i:
                    up.append(m)
                elif not morphisms_str_info[m].curving:
                    # This is a straight horizontal morphism,
                    # because it has no curving.
                    straight_horizontal.append(m)

        if len(up) < len(down):
            # More morphisms stick out downward than upward, let's
            # curve the morphism up.
            if backwards:
                curving = "_"
                label_pos = "_"
            else:
                curving = "^"
                label_pos = "^"

            # Assure that the straight horizontal morphisms have
            # their labels on the lower side of the arrow.
            for m in straight_horizontal:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]

                m_str_info = morphisms_str_info[m]
                if j1 < j2:
                    m_str_info.label_position = "_"
                else:
                    m_str_info.label_position = "^"

                # Don't allow any further modifications of the
                # position of this label.
                m_str_info.forced_label_position = True
        else:
            # More morphisms stick out downward than upward, let's
            # curve the morphism up.
            if backwards:
                curving = "^"
                label_pos = "^"
            else:
                curving = "_"
                label_pos = "_"

            # Assure that the straight horizontal morphisms have
            # their labels on the upper side of the arrow.
            for m in straight_horizontal:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]

                m_str_info = morphisms_str_info[m]
                if j1 < j2:
                    m_str_info.label_position = "^"
                else:
                    m_str_info.label_position = "_"

                # Don't allow any further modifications of the
                # position of this label.
                m_str_info.forced_label_position = True

        return (curving, label_pos)

    @staticmethod
    def _process_vertical_morphism(i, j, target_i, grid, morphisms_str_info,
                                   object_coords):
        """
        Produces the information required for constructing the string
        representation of a vertical morphism.  This function is
        invoked from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        # This arrow is vertical.  Check if it goes from top to
        # bottom (``backwards == False``) or from bottom to top
        # (``backwards == True``).
        backwards = False
        start = i
        end = target_i
        if end < start:
            (start, end) = (end, start)
            backwards = True

        # Let's see which objects are there between ``start`` and
        # ``end``, and then count how many morphisms stick out to
        # the left, and how many stick out to the right.
        #
        # See the corresponding comment in the previous branch of
        # this if-statement for more details.
        left = []
        right = []
        straight_vertical = []
        for k in range(start + 1, end):
            obj = grid[k, j]
            if not obj:
                continue

            for m in morphisms_str_info:
                if m.domain == obj:
                    (end_i, end_j) = object_coords[m.codomain]
                elif m.codomain == obj:
                    (end_i, end_j) = object_coords[m.domain]
                else:
                    continue

                if end_j > j:
                    right.append(m)
                elif end_j < j:
                    left.append(m)
                elif not morphisms_str_info[m].curving:
                    # This is a straight vertical morphism,
                    # because it has no curving.
                    straight_vertical.append(m)

        if len(left) < len(right):
            # More morphisms stick out to the left than to the
            # right, let's curve the morphism to the right.
            if backwards:
                curving = "^"
                label_pos = "^"
            else:
                curving = "_"
                label_pos = "_"

            # Assure that the straight vertical morphisms have
            # their labels on the left side of the arrow.
            for m in straight_vertical:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]

                m_str_info = morphisms_str_info[m]
                if i1 < i2:
                    m_str_info.label_position = "^"
                else:
                    m_str_info.label_position = "_"

                # Don't allow any further modifications of the
                # position of this label.
                m_str_info.forced_label_position = True
        else:
            # More morphisms stick out to the right than to the
            # left, let's curve the morphism to the left.
            if backwards:
                curving = "_"
                label_pos = "_"
            else:
                curving = "^"
                label_pos = "^"

            # Assure that the straight vertical morphisms have
            # their labels on the right side of the arrow.
            for m in straight_vertical:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]

                m_str_info = morphisms_str_info[m]
                if i1 < i2:
                    m_str_info.label_position = "_"
                else:
                    m_str_info.label_position = "^"

                # Don't allow any further modifications of the
                # position of this label.
                m_str_info.forced_label_position = True

        return (curving, label_pos)

    def _process_morphism(self, diagram, grid, morphism, object_coords,
                          morphisms, morphisms_str_info):
        """
        Given the required information, produces the string
        representation of ``morphism``.
        """
        def repeat_string_cond(times, str_gt, str_lt):
            """
            If ``times > 0``, repeats ``str_gt`` ``times`` times.
            Otherwise, repeats ``str_lt`` ``-times`` times.
            """
            if times > 0:
                return str_gt * times
            else:
                return str_lt * (-times)

        def count_morphisms_undirected(A, B):
            """
            Counts how many processed morphisms there are between the
            two supplied objects.
            """
            return len([m for m in morphisms_str_info
                        if {m.domain, m.codomain} == {A, B}])

        def count_morphisms_filtered(dom, cod, curving):
            """
            Counts the processed morphisms which go out of ``dom``
            into ``cod`` with curving ``curving``.
            """
            return len([m for m, m_str_info in morphisms_str_info.items()
                        if (m.domain, m.codomain) == (dom, cod) and
                        (m_str_info.curving == curving)])

        (i, j) = object_coords[morphism.domain]
        (target_i, target_j) = object_coords[morphism.codomain]

        # We now need to determine the direction of
        # the arrow.
        delta_i = target_i - i
        delta_j = target_j - j
        vertical_direction = repeat_string_cond(delta_i,
                                                "d", "u")
        horizontal_direction = repeat_string_cond(delta_j,
                                                  "r", "l")

        curving = ""
        label_pos = "^"
        looping_start = ""
        looping_end = ""

        if (delta_i == 0) and (delta_j == 0):
            # This is a loop morphism.
            (curving, label_pos, looping_start,
             looping_end) = XypicDiagramDrawer._process_loop_morphism(
                 i, j, grid, morphisms_str_info, object_coords)
        elif (delta_i == 0) and (abs(j - target_j) > 1):
            # This is a horizontal morphism.
            (curving, label_pos) = XypicDiagramDrawer._process_horizontal_morphism(
                i, j, target_j, grid, morphisms_str_info, object_coords)
        elif (delta_j == 0) and (abs(i - target_i) > 1):
            # This is a vertical morphism.
            (curving, label_pos) = XypicDiagramDrawer._process_vertical_morphism(
                i, j, target_i, grid, morphisms_str_info, object_coords)

        count = count_morphisms_undirected(morphism.domain, morphism.codomain)
        curving_amount = ""
        if curving:
            # This morphisms should be curved anyway.
            curving_amount = self.default_curving_amount + count * \
                self.default_curving_step
        elif count:
            # There are no objects between the domain and codomain of
            # the current morphism, but this is not there already are
            # some morphisms with the same domain and codomain, so we
            # have to curve this one.
            curving = "^"
            filtered_morphisms = count_morphisms_filtered(
                morphism.domain, morphism.codomain, curving)
            curving_amount = self.default_curving_amount + \
                filtered_morphisms * \
                self.default_curving_step

        # Let's now get the name of the morphism.
        morphism_name = ""
        if isinstance(morphism, IdentityMorphism):
            morphism_name = "id_{%s}" + latex(grid[i, j])
        elif isinstance(morphism, CompositeMorphism):
            component_names = [latex(Symbol(component.name)) for
                               component in morphism.components]
            component_names.reverse()
            morphism_name = "\\circ ".join(component_names)
        elif isinstance(morphism, NamedMorphism):
            morphism_name = latex(Symbol(morphism.name))

        return ArrowStringDescription(
            self.unit, curving, curving_amount, looping_start,
            looping_end, horizontal_direction, vertical_direction,
            label_pos, morphism_name)

    @staticmethod
    def _check_free_space_horizontal(dom_i, dom_j, cod_j, grid):
        """
        For a horizontal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """
        if dom_j < cod_j:
            (start, end) = (dom_j, cod_j)
            backwards = False
        else:
            (start, end) = (cod_j, dom_j)
            backwards = True

        # Check for free space above.
        if dom_i == 0:
            free_up = True
        else:
            free_up = all(grid[dom_i - 1, j] for j in
                          range(start, end + 1))

        # Check for free space below.
        if dom_i == grid.height - 1:
            free_down = True
        else:
            free_down = not any(grid[dom_i + 1, j] for j in
                                range(start, end + 1))

        return (free_up, free_down, backwards)

    @staticmethod
    def _check_free_space_vertical(dom_i, cod_i, dom_j, grid):
        """
        For a vertical morphism, checks whether there is free space
        (i.e., space not occupied by any objects) to the left of the
        morphism or to the right of it.
        """
        if dom_i < cod_i:
            (start, end) = (dom_i, cod_i)
            backwards = False
        else:
            (start, end) = (cod_i, dom_i)
            backwards = True

        # Check if there's space to the left.
        if dom_j == 0:
            free_left = True
        else:
            free_left = not any(grid[i, dom_j - 1] for i in
                                range(start, end + 1))

        if dom_j == grid.width - 1:
            free_right = True
        else:
            free_right = not any(grid[i, dom_j + 1] for i in
                                 range(start, end + 1))

        return (free_left, free_right, backwards)

    @staticmethod
    def _check_free_space_diagonal(dom_i, cod_i, dom_j, cod_j, grid):
        """
        For a diagonal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """
        def abs_xrange(start, end):
            if start < end:
                return range(start, end + 1)
            else:
                return range(end, start + 1)

        if dom_i < cod_i and dom_j < cod_j:
            # This morphism goes from top-left to
            # bottom-right.
            (start_i, start_j) = (dom_i, dom_j)
            (end_i, end_j) = (cod_i, cod_j)
            backwards = False
        elif dom_i > cod_i and dom_j > cod_j:
            # This morphism goes from bottom-right to
            # top-left.
            (start_i, start_j) = (cod_i, cod_j)
            (end_i, end_j) = (dom_i, dom_j)
            backwards = True
        if dom_i < cod_i and dom_j > cod_j:
            # This morphism goes from top-right to
            # bottom-left.
            (start_i, start_j) = (dom_i, dom_j)
            (end_i, end_j) = (cod_i, cod_j)
            backwards = True
        elif dom_i > cod_i and dom_j < cod_j:
            # This morphism goes from bottom-left to
            # top-right.
            (start_i, start_j) = (cod_i, cod_j)
            (end_i, end_j) = (dom_i, dom_j)
            backwards = False

        # This is an attempt at a fast and furious strategy to
        # decide where there is free space on the two sides of
        # a diagonal morphism.  For a diagonal morphism
        # starting at ``(start_i, start_j)`` and ending at
        # ``(end_i, end_j)`` the rectangle defined by these
        # two points is considered.  The slope of the diagonal
        # ``alpha`` is then computed.  Then, for every cell
        # ``(i, j)`` within the rectangle, the slope
        # ``alpha1`` of the line through ``(start_i,
        # start_j)`` and ``(i, j)`` is considered.  If
        # ``alpha1`` is between 0 and ``alpha``, the point
        # ``(i, j)`` is above the diagonal, if ``alpha1`` is
        # between ``alpha`` and infinity, the point is below
        # the diagonal.  Also note that, with some beforehand
        # precautions, this trick works for both the main and
        # the secondary diagonals of the rectangle.

        # I have considered the possibility to only follow the
        # shorter diagonals immediately above and below the
        # main (or secondary) diagonal.  This, however,
        # wouldn't have resulted in much performance gain or
        # better detection of outer edges, because of
        # relatively small sizes of diagram grids, while the
        # code would have become harder to understand.

        alpha = float(end_i - start_i)/(end_j - start_j)
        free_up = True
        free_down = True
        for i in abs_xrange(start_i, end_i):
            if not free_up and not free_down:
                break

            for j in abs_xrange(start_j, end_j):
                if not free_up and not free_down:
                    break

                if (i, j) == (start_i, start_j):
                    continue

                if j == start_j:
                    alpha1 = "inf"
                else:
                    alpha1 = float(i - start_i)/(j - start_j)

                if grid[i, j]:
                    if (alpha1 == "inf") or (abs(alpha1) > abs(alpha)):
                        free_down = False
                    elif abs(alpha1) < abs(alpha):
                        free_up = False

        return (free_up, free_down, backwards)

    def _push_labels_out(self, morphisms_str_info, grid, object_coords):
        """
        For all straight morphisms which form the visual boundary of
        the laid out diagram, puts their labels on their outer sides.
        """
        def set_label_position(free1, free2, pos1, pos2, backwards, m_str_info):
            """
            Given the information about room available to one side and
            to the other side of a morphism (``free1`` and ``free2``),
            sets the position of the morphism label in such a way that
            it is on the freer side.  This latter operations involves
            choice between ``pos1`` and ``pos2``, taking ``backwards``
            in consideration.

            Thus this function will do nothing if either both ``free1
            == True`` and ``free2 == True`` or both ``free1 == False``
            and ``free2 == False``.  In either case, choosing one side
            over the other presents no advantage.
            """
            if backwards:
                (pos1, pos2) = (pos2, pos1)

            if free1 and not free2:
                m_str_info.label_position = pos1
            elif free2 and not free1:
                m_str_info.label_position = pos2

        for m, m_str_info in morphisms_str_info.items():
            if m_str_info.curving or m_str_info.forced_label_position:
                # This is either a curved morphism, and curved
                # morphisms have other magic, or the position of this
                # label has already been fixed.
                continue

            if m.domain == m.codomain:
                # This is a loop morphism, their labels, again have a
                # different magic.
                continue

            (dom_i, dom_j) = object_coords[m.domain]
            (cod_i, cod_j) = object_coords[m.codomain]

            if dom_i == cod_i:
                # Horizontal morphism.
                (free_up, free_down,
                 backwards) = XypicDiagramDrawer._check_free_space_horizontal(
                     dom_i, dom_j, cod_j, grid)

                set_label_position(free_up, free_down, "^", "_",
                                   backwards, m_str_info)
            elif dom_j == cod_j:
                # Vertical morphism.
                (free_left, free_right,
                 backwards) = XypicDiagramDrawer._check_free_space_vertical(
                     dom_i, cod_i, dom_j, grid)

                set_label_position(free_left, free_right, "_", "^",
                                   backwards, m_str_info)
            else:
                # A diagonal morphism.
                (free_up, free_down,
                 backwards) = XypicDiagramDrawer._check_free_space_diagonal(
                     dom_i, cod_i, dom_j, cod_j, grid)

                set_label_position(free_up, free_down, "^", "_",
                                   backwards, m_str_info)

    @staticmethod
    def _morphism_sort_key(morphism, object_coords):
        """
        Provides a morphism sorting key such that horizontal or
        vertical morphisms between neighbouring objects come
        first, then horizontal or vertical morphisms between more
        far away objects, and finally, all other morphisms.
        """
        (i, j) = object_coords[morphism.domain]
        (target_i, target_j) = object_coords[morphism.codomain]

        if morphism.domain == morphism.codomain:
            # Loop morphisms should get after diagonal morphisms
            # so that the proper direction in which to curve the
            # loop can be determined.
            return (3, 0, default_sort_key(morphism))

        if target_i == i:
            return (1, abs(target_j - j), default_sort_key(morphism))

        if target_j == j:
            return (1, abs(target_i - i), default_sort_key(morphism))

        # Diagonal morphism.
        return (2, 0, default_sort_key(morphism))

    @staticmethod
    def _build_xypic_string(diagram, grid, morphisms,
                            morphisms_str_info, diagram_format):
        """
        Given a collection of :class:`ArrowStringDescription`
        describing the morphisms of a diagram and the object layout
        information of a diagram, produces the final Xy-pic picture.
        """
        # Build the mapping between objects and morphisms which have
        # them as domains.
        object_morphisms = {}
        for obj in diagram.objects:
            object_morphisms[obj] = []
        for morphism in morphisms:
            object_morphisms[morphism.domain].append(morphism)

        result = "\\xymatrix%s{\n" % diagram_format

        for i in range(grid.height):
            for j in range(grid.width):
                obj = grid[i, j]
                if obj:
                    result += latex(obj) + " "

                    morphisms_to_draw = object_morphisms[obj]
                    for morphism in morphisms_to_draw:
                        result += str(morphisms_str_info[morphism]) + " "

                # Don't put the & after the last column.
                if j < grid.width - 1:
                    result += "& "

            # Don't put the line break after the last row.
            if i < grid.height - 1:
                result += "\\\\"
            result += "\n"

        result += "}\n"

        return result

    def draw(self, diagram, grid, masked=None, diagram_format=""):
        r"""
        Returns the Xy-pic representation of ``diagram`` laid out in
        ``grid``.

        Consider the following simple triangle diagram.

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g], {g * f: "unique"})

        To draw this diagram, its objects need to be laid out with a
        :class:`DiagramGrid`::

        >>> grid = DiagramGrid(diagram)

        Finally, the drawing:

        >>> drawer = XypicDiagramDrawer()
        >>> print(drawer.draw(diagram, grid))
        \xymatrix{
        A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
        C &
        }

        The argument ``masked`` can be used to skip morphisms in the
        presentation of the diagram:

        >>> print(drawer.draw(diagram, grid, masked=[g * f]))
        \xymatrix{
        A \ar[r]^{f} & B \ar[ld]^{g} \\
        C &
        }

        Finally, the ``diagram_format`` argument can be used to
        specify the format string of the diagram.  For example, to
        increase the spacing by 1 cm, proceeding as follows:

        >>> print(drawer.draw(diagram, grid, diagram_format="@+1cm"))
        \xymatrix@+1cm{
        A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
        C &
        }

        """
        # This method works in several steps.  It starts by removing
        # the masked morphisms, if necessary, and then maps objects to
        # their positions in the grid (coordinate tuples).  Remember
        # that objects are unique in ``Diagram`` and in the layout
        # produced by ``DiagramGrid``, so every object is mapped to a
        # single coordinate pair.
        #
        # The next step is the central step and is concerned with
        # analysing the morphisms of the diagram and deciding how to
        # draw them.  For example, how to curve the arrows is decided
        # at this step.  The bulk of the analysis is implemented in
        # ``_process_morphism``, to the result of which the
        # appropriate formatters are applied.
        #
        # The result of the previous step is a list of
        # ``ArrowStringDescription``.  After the analysis and
        # application of formatters, some extra logic tries to assure
        # better positioning of morphism labels (for example, an
        # attempt is made to avoid the situations when arrows cross
        # labels).  This functionality constitutes the next step and
        # is implemented in ``_push_labels_out``.  Note that label
        # positions which have been set via a formatter are not
        # affected in this step.
        #
        # Finally, at the closing step, the array of
        # ``ArrowStringDescription`` and the layout information
        # incorporated in ``DiagramGrid`` are combined to produce the
        # resulting Xy-pic picture.  This part of code lies in
        # ``_build_xypic_string``.

        if not masked:
            morphisms_props = grid.morphisms
        else:
            morphisms_props = {}
            for m, props in grid.morphisms.items():
                if m in masked:
                    continue
                morphisms_props[m] = props

        # Build the mapping between objects and their position in the
        # grid.
        object_coords = {}
        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    object_coords[grid[i, j]] = (i, j)

        morphisms = sorted(morphisms_props,
                           key=lambda m: XypicDiagramDrawer._morphism_sort_key(
                               m, object_coords))

        # Build the tuples defining the string representations of
        # morphisms.
        morphisms_str_info = {}
        for morphism in morphisms:
            string_description = self._process_morphism(
                diagram, grid, morphism, object_coords, morphisms,
                morphisms_str_info)

            if self.default_arrow_formatter:
                self.default_arrow_formatter(string_description)

            for prop in morphisms_props[morphism]:
                # prop is a Symbol.  TODO: Find out why.
                if prop.name in self.arrow_formatters:
                    formatter = self.arrow_formatters[prop.name]
                    formatter(string_description)

            morphisms_str_info[morphism] = string_description

        # Reposition the labels a bit.
        self._push_labels_out(morphisms_str_info, grid, object_coords)

        return XypicDiagramDrawer._build_xypic_string(
            diagram, grid, morphisms, morphisms_str_info, diagram_format)


def xypic_draw_diagram(diagram, masked=None, diagram_format="",
                       groups=None, **hints):
    r"""
    Provides a shortcut combining :class:`DiagramGrid` and
    :class:`XypicDiagramDrawer`.  Returns an Xy-pic presentation of
    ``diagram``.  The argument ``masked`` is a list of morphisms which
    will be not be drawn.  The argument ``diagram_format`` is the
    format string inserted after "\xymatrix".  ``groups`` should be a
    set of logical groups.  The ``hints`` will be passed directly to
    the constructor of :class:`DiagramGrid`.

    For more information about the arguments, see the docstrings of
    :class:`DiagramGrid` and ``XypicDiagramDrawer.draw``.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import xypic_draw_diagram
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g], {g * f: "unique"})
    >>> print(xypic_draw_diagram(diagram))
    \xymatrix{
    A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    See Also
    ========

    XypicDiagramDrawer, DiagramGrid
    """
    grid = DiagramGrid(diagram, groups, **hints)
    drawer = XypicDiagramDrawer()
    return drawer.draw(diagram, grid, masked, diagram_format)


@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',))
def preview_diagram(diagram, masked=None, diagram_format="", groups=None,
                    output='png', viewer=None, euler=True, **hints):
    """
    Combines the functionality of ``xypic_draw_diagram`` and
    ``sympy.printing.preview``.  The arguments ``masked``,
    ``diagram_format``, ``groups``, and ``hints`` are passed to
    ``xypic_draw_diagram``, while ``output``, ``viewer, and ``euler``
    are passed to ``preview``.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import preview_diagram
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g], {g * f: "unique"})
    >>> preview_diagram(d)

    See Also
    ========

    XypicDiagramDrawer
    """
    from sympy.printing import preview
    latex_output = xypic_draw_diagram(diagram, masked, diagram_format,
                                      groups, **hints)
    preview(latex_output, output, viewer, euler, ("xypic",))