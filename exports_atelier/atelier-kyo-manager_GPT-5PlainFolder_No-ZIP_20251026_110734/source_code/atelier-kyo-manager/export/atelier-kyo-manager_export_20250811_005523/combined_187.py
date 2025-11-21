
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\tests\test_plot.py ===
import os
from tempfile import TemporaryDirectory
import pytest
from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, oo, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import (real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.plotting.plot import (
    Plot, plot, plot_parametric, plot3d_parametric_line, plot3d,
    plot3d_parametric_surface)
from sympy.plotting.plot import (
    unset_show, plot_contour, PlotGrid, MatplotlibBackend, TextBackend)
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries)
from sympy.testing.pytest import skip, skip_under_pyodide, warns, raises, warns_deprecated_sympy
from sympy.utilities import lambdify as lambdify_
from sympy.utilities.exceptions import ignore_warnings

unset_show()


matplotlib = import_module(
    'matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))


class DummyBackendNotOk(Plot):
    """ Used to verify if users can create their own backends.
    This backend is meant to raise NotImplementedError for methods `show`,
    `save`, `close`.
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)


class DummyBackendOk(Plot):
    """ Used to verify if users can create their own backends.
    This backend is meant to pass all tests.
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def show(self):
        pass

    def save(self):
        pass

    def close(self):
        pass

def test_basic_plotting_backend():
    x = Symbol('x')
    plot(x, (x, 0, 3), backend='text')
    plot(x**2 + 1, (x, 0, 3), backend='text')

@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_1(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        ###
        # Examples from the 'introduction' notebook
        ###
        p = plot(x, legend=True, label='f1', adaptive=adaptive, n=10)
        p = plot(x*sin(x), x*cos(x), label='f2', adaptive=adaptive, n=10)
        p.extend(p)
        p[0].line_color = lambda a: a
        p[1].line_color = 'b'
        p.title = 'Big title'
        p.xlabel = 'the x axis'
        p[1].label = 'straight line'
        p.legend = True
        p.aspect_ratio = (1, 1)
        p.xlim = (-15, 20)
        filename = 'test_basic_options_and_colors.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p.extend(plot(x + 1, adaptive=adaptive, n=10))
        p.append(plot(x + 3, x**2, adaptive=adaptive, n=10)[1])
        filename = 'test_plot_extend_append.png'
        p.save(os.path.join(tmpdir, filename))

        p[2] = plot(x**2, (x, -2, 3), adaptive=adaptive, n=10)
        filename = 'test_plot_setitem.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(sin(x), (x, -2*pi, 4*pi), adaptive=adaptive, n=10)
        filename = 'test_line_explicit.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(sin(x), adaptive=adaptive, n=10)
        filename = 'test_line_default_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot((x**2, (x, -5, 5)), (x**3, (x, -3, 3)), adaptive=adaptive, n=10)
        filename = 'test_line_multiple_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        raises(ValueError, lambda: plot(x, y))

        #Piecewise plots
        p = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1), adaptive=adaptive, n=10)
        filename = 'test_plot_piecewise.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(Piecewise((x, x < 1), (x**2, True)), (x, -3, 3), adaptive=adaptive, n=10)
        filename = 'test_plot_piecewise_2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # test issue 7471
        p1 = plot(x, adaptive=adaptive, n=10)
        p2 = plot(3, adaptive=adaptive, n=10)
        p1.extend(p2)
        filename = 'test_horizontal_line.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # test issue 10925
        f = Piecewise((-1, x < -1), (x, And(-1 <= x, x < 0)), \
            (x**2, And(0 <= x, x < 1)), (x**3, x >= 1))
        p = plot(f, (x, -3, 3), adaptive=adaptive, n=10)
        filename = 'test_plot_piecewise_3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_2(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        #parametric 2d plots.
        #Single plot with default range.
        p = plot_parametric(sin(x), cos(x), adaptive=adaptive, n=10)
        filename = 'test_parametric.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #Single plot with range.
        p = plot_parametric(
            sin(x), cos(x), (x, -5, 5), legend=True, label='parametric_plot',
            adaptive=adaptive, n=10)
        filename = 'test_parametric_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #Multiple plots with same range.
        p = plot_parametric((sin(x), cos(x)), (x, sin(x)),
            adaptive=adaptive, n=10)
        filename = 'test_parametric_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #Multiple plots with different ranges.
        p = plot_parametric(
            (sin(x), cos(x), (x, -3, 3)), (x, sin(x), (x, -5, 5)),
            adaptive=adaptive, n=10)
        filename = 'test_parametric_multiple_ranges.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #depth of recursion specified.
        p = plot_parametric(x, sin(x), depth=13,
            adaptive=adaptive, n=10)
        filename = 'test_recursion_depth.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #No adaptive sampling.
        p = plot_parametric(cos(x), sin(x), adaptive=False, n=500)
        filename = 'test_adaptive.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #3d parametric plots
        p = plot3d_parametric_line(
            sin(x), cos(x), x, legend=True, label='3d_parametric_plot',
            adaptive=adaptive, n=10)
        filename = 'test_3d_line.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d_parametric_line(
            (sin(x), cos(x), x, (x, -5, 5)), (cos(x), sin(x), x, (x, -3, 3)),
            adaptive=adaptive, n=10)
        filename = 'test_3d_line_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d_parametric_line(sin(x), cos(x), x, n=30,
            adaptive=adaptive)
        filename = 'test_3d_line_points.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # 3d surface single plot.
        p = plot3d(x * y, adaptive=adaptive, n=10)
        filename = 'test_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple 3D plots with same range.
        p = plot3d(-x * y, x * y, (x, -5, 5), adaptive=adaptive, n=10)
        filename = 'test_surface_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple 3D plots with different ranges.
        p = plot3d(
            (x * y, (x, -3, 3), (y, -3, 3)), (-x * y, (x, -3, 3), (y, -3, 3)),
            adaptive=adaptive, n=10)
        filename = 'test_surface_multiple_ranges.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Single Parametric 3D plot
        p = plot3d_parametric_surface(sin(x + y), cos(x - y), x - y,
            adaptive=adaptive, n=10)
        filename = 'test_parametric_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple Parametric 3D plots.
        p = plot3d_parametric_surface(
            (x*sin(z), x*cos(z), z, (x, -5, 5), (z, -5, 5)),
            (sin(x + y), cos(x - y), x - y, (x, -5, 5), (y, -5, 5)),
            adaptive=adaptive, n=10)
        filename = 'test_parametric_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Single Contour plot.
        p = plot_contour(sin(x)*sin(y), (x, -5, 5), (y, -5, 5),
            adaptive=adaptive, n=10)
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple Contour plots with same range.
        p = plot_contour(x**2 + y**2, x**3 + y**3, (x, -5, 5), (y, -5, 5),
            adaptive=adaptive, n=10)
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple Contour plots with different range.
        p = plot_contour(
            (x**2 + y**2, (x, -5, 5), (y, -5, 5)),
            (x**3 + y**3, (x, -3, 3), (y, -3, 3)),
            adaptive=adaptive, n=10)
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_3(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        ###
        # Examples from the 'colors' notebook
        ###

        p = plot(sin(x), adaptive=adaptive, n=10)
        p[0].line_color = lambda a: a
        filename = 'test_colors_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))

        p[0].line_color = lambda a, b: b
        filename = 'test_colors_line_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(x*sin(x), x*cos(x), (x, 0, 10), adaptive=adaptive, n=10)
        p[0].line_color = lambda a: a
        filename = 'test_colors_param_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))

        p[0].line_color = lambda a, b: a
        filename = 'test_colors_param_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))

        p[0].line_color = lambda a, b: b
        filename = 'test_colors_param_line_arity2b.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d_parametric_line(
            sin(x) + 0.1*sin(x)*cos(7*x),
            cos(x) + 0.1*cos(x)*cos(7*x),
            0.1*sin(7*x),
            (x, 0, 2*pi), adaptive=adaptive, n=10)
        p[0].line_color = lambdify_(x, sin(4*x))
        filename = 'test_colors_3d_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b: b
        filename = 'test_colors_3d_line_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b, c: c
        filename = 'test_colors_3d_line_arity3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d(sin(x)*y, (x, 0, 6*pi), (y, -5, 5), adaptive=adaptive, n=10)
        p[0].surface_color = lambda a: a
        filename = 'test_colors_surface_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b: b
        filename = 'test_colors_surface_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b, c: c
        filename = 'test_colors_surface_arity3a.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambdify_((x, y, z), sqrt((x - 3*pi)**2 + y**2))
        filename = 'test_colors_surface_arity3b.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d_parametric_surface(x * cos(4 * y), x * sin(4 * y), y,
                (x, -1, 1), (y, -1, 1), adaptive=adaptive, n=10)
        p[0].surface_color = lambda a: a
        filename = 'test_colors_param_surf_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b: a*b
        filename = 'test_colors_param_surf_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambdify_((x, y, z), sqrt(x**2 + y**2 + z**2))
        filename = 'test_colors_param_surf_arity3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True])
def test_plot_and_save_4(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')

    ###
    # Examples from the 'advanced' notebook
    ###

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        i = Integral(log((sin(x)**2 + 1)*sqrt(x**2 + 1)), (x, 0, y))
        p = plot(i, (y, 1, 5), adaptive=adaptive, n=10, force_real_eval=True)
        filename = 'test_advanced_integral.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_5(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        s = Sum(1/x**y, (x, 1, oo))
        p = plot(s, (y, 2, 10), adaptive=adaptive, n=10)
        filename = 'test_advanced_inf_sum.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(Sum(1/x, (x, 1, y)), (y, 2, 10), show=False,
            adaptive=adaptive, n=10)
        p[0].only_integers = True
        p[0].steps = True
        filename = 'test_advanced_fin_sum.png'

        # XXX: This should be fixed in experimental_lambdify or by using
        # ordinary lambdify so that it doesn't warn. The error results from
        # passing an array of values as the integration limit.
        #
        # UserWarning: The evaluation of the expression is problematic. We are
        # trying a failback method that may still work. Please report this as a
        # bug.
        with ignore_warnings(UserWarning):
            p.save(os.path.join(tmpdir, filename))

        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_6(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        filename = 'test.png'
        ###
        # Test expressions that can not be translated to np and generate complex
        # results.
        ###
        p = plot(sin(x) + I*cos(x))
        p.save(os.path.join(tmpdir, filename))

        with ignore_warnings(RuntimeWarning):
            p = plot(sqrt(sqrt(-x)))
            p.save(os.path.join(tmpdir, filename))

        p = plot(LambertW(x))
        p.save(os.path.join(tmpdir, filename))
        p = plot(sqrt(LambertW(x)))
        p.save(os.path.join(tmpdir, filename))

        #Characteristic function of a StudentT distribution with nu=10
        x1 = 5 * x**2 * exp_polar(-I*pi)/2
        m1 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x1)
        x2 = 5*x**2 * exp_polar(I*pi)/2
        m2 = meijerg(((1/2,), ()), ((5, 0, 1/2), ()), x2)
        expr = (m1 + m2) / (48 * pi)
        with warns(
            UserWarning,
            match="The evaluation with NumPy/SciPy failed",
            test_stacklevel=False,
        ):
            p = plot(expr, (x, 1e-6, 1e-2), adaptive=adaptive, n=10)
            p.save(os.path.join(tmpdir, filename))


@pytest.mark.parametrize("adaptive", [True, False])
def test_plotgrid_and_save(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        p1 = plot(x, adaptive=adaptive, n=10)
        p2 = plot_parametric((sin(x), cos(x)), (x, sin(x)), show=False,
            adaptive=adaptive, n=10)
        p3 = plot_parametric(
            cos(x), sin(x), adaptive=adaptive, n=10, show=False)
        p4 = plot3d_parametric_line(sin(x), cos(x), x, show=False,
            adaptive=adaptive, n=10)
        # symmetric grid
        p = PlotGrid(2, 2, p1, p2, p3, p4)
        filename = 'test_grid1.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # grid size greater than the number of subplots
        p = PlotGrid(3, 4, p1, p2, p3, p4)
        filename = 'test_grid2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p5 = plot(cos(x),(x, -pi, pi), show=False, adaptive=adaptive, n=10)
        p5[0].line_color = lambda a: a
        p6 = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1), show=False,
            adaptive=adaptive, n=10)
        p7 = plot_contour(
            (x**2 + y**2, (x, -5, 5), (y, -5, 5)),
            (x**3 + y**3, (x, -3, 3), (y, -3, 3)), show=False,
            adaptive=adaptive, n=10)
        # unsymmetric grid (subplots in one line)
        p = PlotGrid(1, 3, p5, p6, p7)
        filename = 'test_grid3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_append_issue_7140(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p1 = plot(x, adaptive=adaptive, n=10)
    p2 = plot(x**2, adaptive=adaptive, n=10)
    plot(x + 2, adaptive=adaptive, n=10)

    # append a series
    p2.append(p1[0])
    assert len(p2._series) == 2

    with raises(TypeError):
        p1.append(p2)

    with raises(TypeError):
        p1.append(p2._series)


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_15265(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    eqn = sin(x)

    p = plot(eqn, xlim=(-S.Pi, S.Pi), ylim=(-1, 1), adaptive=adaptive, n=10)
    p._backend.close()

    p = plot(eqn, xlim=(-1, 1), ylim=(-S.Pi, S.Pi), adaptive=adaptive, n=10)
    p._backend.close()

    p = plot(eqn, xlim=(-1, 1), adaptive=adaptive, n=10,
        ylim=(sympify('-3.14'), sympify('3.14')))
    p._backend.close()

    p = plot(eqn, adaptive=adaptive, n=10,
        xlim=(sympify('-3.14'), sympify('3.14')), ylim=(-1, 1))
    p._backend.close()

    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-S.ImaginaryUnit, 1), ylim=(-1, 1)))

    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-1, 1), ylim=(-1, S.ImaginaryUnit)))

    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(S.NegativeInfinity, 1), ylim=(-1, 1)))

    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-1, 1), ylim=(-1, S.Infinity)))


def test_empty_Plot():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # No exception showing an empty plot
    plot()
    # Plot is only a base class: doesn't implement any logic for showing
    # images
    p = Plot()
    raises(NotImplementedError, lambda: p.show())


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_17405(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    f = x**0.3 - 10*x**3 + x**2
    p = plot(f, (x, -10, 10), adaptive=adaptive, n=30, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present

    # RuntimeWarning: invalid value encountered in double_scalars
    with ignore_warnings(RuntimeWarning):
        assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_logplot_PR_16796(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(x, (x, .001, 100), adaptive=adaptive, n=30,
        xscale='log', show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30
    assert p[0].end == 100.0
    assert p[0].start == .001


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_16572(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(LambertW(x), show=False, adaptive=adaptive, n=30)
    # Random number of segments, probably more than 50, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_11865(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    k = Symbol('k', integer=True)
    f = Piecewise((-I*exp(I*pi*k)/k + I*exp(-I*pi*k)/k, Ne(k, 0)), (2*pi, True))
    p = plot(f, show=False, adaptive=adaptive, n=30)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    # and that there are no exceptions.
    assert len(p[0].get_data()[0]) >= 30


@skip_under_pyodide("Warnings not emitted in Pyodide because of lack of WASM fp exception support")
def test_issue_11461():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(real_root((log(x/(x-2))), 3), show=False, adaptive=True)
    with warns(
        RuntimeWarning,
        match="invalid value encountered in",
        test_stacklevel=False,
    ):
        # Random number of segments, probably more than 100, but we want to see
        # that there are segments generated, as opposed to when the bug was present
        # and that there are no exceptions.
        assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_11764(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect_ratio=(1,1), show=False, adaptive=adaptive, n=30)
    assert p.aspect_ratio == (1, 1)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_13516(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    pm = plot(sin(x), backend="matplotlib", show=False, adaptive=adaptive, n=30)
    assert pm.backend == MatplotlibBackend
    assert len(pm[0].get_data()[0]) >= 30

    pt = plot(sin(x), backend="text", show=False, adaptive=adaptive, n=30)
    assert pt.backend == TextBackend
    assert len(pt[0].get_data()[0]) >= 30

    pd = plot(sin(x), backend="default", show=False, adaptive=adaptive, n=30)
    assert pd.backend == MatplotlibBackend
    assert len(pd[0].get_data()[0]) >= 30

    p = plot(sin(x), show=False, adaptive=adaptive, n=30)
    assert p.backend == MatplotlibBackend
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_limits(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(x, x**2, (x, -10, 10), adaptive=adaptive, n=10)
    backend = p._backend

    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 10) < 2
    assert abs(xmax - 10) < 2
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 10) < 10
    assert abs(ymax - 100) < 10


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot3d_parametric_line_limits(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    v1 = (2*cos(x), 2*sin(x), 2*x, (x, -5, 5))
    v2 = (sin(x), cos(x), x, (x, -5, 5))
    p = plot3d_parametric_line(v1, v2, adaptive=adaptive, n=60)
    backend = p._backend

    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = backend.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2

    p = plot3d_parametric_line(v2, v1, adaptive=adaptive, n=60)
    backend = p._backend

    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = backend.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_size(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    p1 = plot(sin(x), backend="matplotlib", size=(8, 4),
        adaptive=adaptive, n=10)
    s1 = p1._backend.fig.get_size_inches()
    assert (s1[0] == 8) and (s1[1] == 4)
    p2 = plot(sin(x), backend="matplotlib", size=(5, 10),
        adaptive=adaptive, n=10)
    s2 = p2._backend.fig.get_size_inches()
    assert (s2[0] == 5) and (s2[1] == 10)
    p3 = PlotGrid(2, 1, p1, p2, size=(6, 2),
        adaptive=adaptive, n=10)
    s3 = p3._backend.fig.get_size_inches()
    assert (s3[0] == 6) and (s3[1] == 2)

    with raises(ValueError):
        plot(sin(x), backend="matplotlib", size=(-1, 3))


def test_issue_20113():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    # verify the capability to use custom backends
    plot(sin(x), backend=Plot, show=False)
    p2 = plot(sin(x), backend=MatplotlibBackend, show=False)
    assert p2.backend == MatplotlibBackend
    assert len(p2[0].get_data()[0]) >= 30
    p3 = plot(sin(x), backend=DummyBackendOk, show=False)
    assert p3.backend == DummyBackendOk
    assert len(p3[0].get_data()[0]) >= 30

    # test for an improper coded backend
    p4 = plot(sin(x), backend=DummyBackendNotOk, show=False)
    assert p4.backend == DummyBackendNotOk
    assert len(p4[0].get_data()[0]) >= 30
    with raises(NotImplementedError):
        p4.show()
    with raises(NotImplementedError):
        p4.save("test/path")
    with raises(NotImplementedError):
        p4._backend.close()


def test_custom_coloring():
    x = Symbol('x')
    y = Symbol('y')
    plot(cos(x), line_color=lambda a: a)
    plot(cos(x), line_color=1)
    plot(cos(x), line_color="r")
    plot_parametric(cos(x), sin(x), line_color=lambda a: a)
    plot_parametric(cos(x), sin(x), line_color=1)
    plot_parametric(cos(x), sin(x), line_color="r")
    plot3d_parametric_line(cos(x), sin(x), x, line_color=lambda a: a)
    plot3d_parametric_line(cos(x), sin(x), x, line_color=1)
    plot3d_parametric_line(cos(x), sin(x), x, line_color="r")
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color=lambda a, b: a**2 + b**2)
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color=1)
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color="r")
    plot3d(x*y, (x, -5, 5), (y, -5, 5),
            surface_color=lambda a, b: a**2 + b**2)
    plot3d(x*y, (x, -5, 5), (y, -5, 5), surface_color=1)
    plot3d(x*y, (x, -5, 5), (y, -5, 5), surface_color="r")


@pytest.mark.parametrize("adaptive", [True, False])
def test_deprecated_get_segments(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    f = sin(x)
    p = plot(f, (x, -10, 10), show=False, adaptive=adaptive, n=10)
    with warns_deprecated_sympy():
        p[0].get_segments()


@pytest.mark.parametrize("adaptive", [True, False])
def test_generic_data_series(adaptive):
    # verify that no errors are raised when generic data series are used
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol("x")
    p = plot(x,
        markers=[{"args":[[0, 1], [0, 1]], "marker": "*", "linestyle": "none"}],
        annotations=[{"text": "test", "xy": (0, 0)}],
        fill={"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]},
        rectangles=[{"xy": (0, 0), "width": 5, "height": 1}],
        adaptive=adaptive, n=10)
    assert len(p._backend.ax.collections) == 1
    assert len(p._backend.ax.patches) == 1
    assert len(p._backend.ax.lines) == 2
    assert len(p._backend.ax.texts) == 1


def test_deprecated_markers_annotations_rectangles_fill():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(sin(x), (x, -10, 10), show=False)
    with warns_deprecated_sympy():
        p.markers = [{"args":[[0, 1], [0, 1]], "marker": "*", "linestyle": "none"}]
    assert len(p._series) == 2
    with warns_deprecated_sympy():
        p.annotations = [{"text": "test", "xy": (0, 0)}]
    assert len(p._series) == 3
    with warns_deprecated_sympy():
        p.fill = {"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]}
    assert len(p._series) == 4
    with warns_deprecated_sympy():
        p.rectangles = [{"xy": (0, 0), "width": 5, "height": 1}]
    assert len(p._series) == 5


def test_back_compatibility():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')
    p = plot(sin(x), adaptive=False, n=5)
    assert len(p[0].get_points()) == 2
    assert len(p[0].get_data()) == 2
    p = plot_parametric(cos(x), sin(x), (x, 0, 2), adaptive=False, n=5)
    assert len(p[0].get_points()) == 2
    assert len(p[0].get_data()) == 3
    p = plot3d_parametric_line(cos(x), sin(x), x, (x, 0, 2),
        adaptive=False, n=5)
    assert len(p[0].get_points()) == 3
    assert len(p[0].get_data()) == 4
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n=5)
    assert len(p[0].get_meshes()) == 3
    assert len(p[0].get_data()) == 3
    p = plot_contour(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n=5)
    assert len(p[0].get_meshes()) == 3
    assert len(p[0].get_data()) == 3
    p = plot3d_parametric_surface(x * cos(y), x * sin(y), x * cos(4 * y) / 2,
        (x, 0, pi), (y, 0, 2*pi), n=5)
    assert len(p[0].get_meshes()) == 3
    assert len(p[0].get_data()) == 5


def test_plot_arguments():
    ### Test arguments for plot()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single expressions
    p = plot(x + 1)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}

    # single expressions custom label
    p = plot(x + 1, "label")
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "label"
    assert p[0].rendering_kw == {}

    # single expressions with range
    p = plot(x + 1, (x, -2, 2))
    assert p[0].ranges == [(x, -2, 2)]

    # single expressions with range, label and rendering-kw dictionary
    p = plot(x + 1, (x, -2, 2), "test", {"color": "r"})
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"color": "r"}

    # multiple expressions
    p = plot(x + 1, x**2)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}
    assert isinstance(p[1], LineOver1DRangeSeries)
    assert p[1].expr == x**2
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x**2"
    assert p[1].rendering_kw == {}

    # multiple expressions over the same range
    p = plot(x + 1, x**2, (x, 0, 5))
    assert p[0].ranges == [(x, 0, 5)]
    assert p[1].ranges == [(x, 0, 5)]

    # multiple expressions over the same range with the same rendering kws
    p = plot(x + 1, x**2, (x, 0, 5), {"color": "r"})
    assert p[0].ranges == [(x, 0, 5)]
    assert p[1].ranges == [(x, 0, 5)]
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "r"}

    # multiple expressions with different ranges, labels and rendering kws
    p = plot(
        (x + 1, (x, 0, 5)),
        (x**2, (x, -2, 2), "test", {"color": "r"}))
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, 0, 5)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}
    assert isinstance(p[1], LineOver1DRangeSeries)
    assert p[1].expr == x**2
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"color": "r"}

    # single argument: lambda function
    f = lambda t: t
    p = plot(lambda t: t)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range and label
    p = plot(f, ("t", -5, 6), "test")
    assert p[0].ranges[0][1:] == (-5, 6)
    assert p[0].get_label(False) == "test"


def test_plot_parametric_arguments():
    ### Test arguments for plot_parametric()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single parametric expression
    p = plot_parametric(x + 1, x)
    assert isinstance(p[0], Parametric2DLineSeries)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}

    # single parametric expression with custom range, label and rendering kws
    p = plot_parametric(x + 1, x, (x, -2, 2), "test",
        {"cmap": "Reds"})
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot_parametric((x + 1, x), (x, -2, 2), "test")
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # multiple parametric expressions same symbol
    p = plot_parametric((x + 1, x), (x ** 2, x + 1))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions different symbols
    p = plot_parametric((x + 1, x), (y ** 2, y + 1, "test"))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (y ** 2, y + 1)
    assert p[1].ranges == [(y, -10, 10)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions same range
    p = plot_parametric((x + 1, x), (x ** 2, x + 1), (x, -2, 2))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions, custom ranges and labels
    p = plot_parametric(
        (x + 1, x, (x, -2, 2), "test1"),
        (x ** 2, x + 1, (x, -3, 3), "test2", {"cmap": "Reds"}))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test1"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -3, 3)]
    assert p[1].get_label(False) == "test2"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single argument: lambda function
    fx = lambda t: t
    fy = lambda t: 2 * t
    p = plot_parametric(fx, fy)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert "Dummy" in p[0].get_label(False)
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range + label
    p = plot_parametric(fx, fy, ("t", 0, 2), "test")
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}


def test_plot3d_parametric_line_arguments():
    ### Test arguments for plot3d_parametric_line()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single parametric expression
    p = plot3d_parametric_line(x + 1, x, sin(x))
    assert isinstance(p[0], Parametric3DLineSeries)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}

    # single parametric expression with custom range, label and rendering kws
    p = plot3d_parametric_line(x + 1, x, sin(x), (x, -2, 2),
        "test", {"cmap": "Reds"})
    assert isinstance(p[0], Parametric3DLineSeries)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot3d_parametric_line((x + 1, x, sin(x)), (x, -2, 2), "test")
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # multiple parametric expression same symbol
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)), (x ** 2, 1, cos(x), {"cmap": "Reds"}))
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, 1, cos(x))
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # multiple parametric expression different symbols
    p = plot3d_parametric_line((x + 1, x, sin(x)), (y ** 2, 1, cos(y)))
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (y ** 2, 1, cos(y))
    assert p[1].ranges == [(y, -10, 10)]
    assert p[1].get_label(False) == "y"
    assert p[1].rendering_kw == {}

    # multiple parametric expression, custom ranges and labels
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)),
        (x ** 2, 1, cos(x), (x, -2, 2), "test", {"cmap": "Reds"}))
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, 1, cos(x))
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single argument: lambda function
    fx = lambda t: t
    fy = lambda t: 2 * t
    fz = lambda t: 3 * t
    p = plot3d_parametric_line(fx, fy, fz)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert "Dummy" in p[0].get_label(False)
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range + label
    p = plot3d_parametric_line(fx, fy, fz, ("t", 0, 2), "test")
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}


def test_plot3d_plot_contour_arguments():
    ### Test arguments for plot3d() and plot_contour()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single expression
    p = plot3d(x + y)
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}

    # single expression, custom range, label and rendering kws
    p = plot3d(x + y, (x, -2, 2), "test", {"cmap": "Reds"})
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -10, 10)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot3d(x + y, (x, -2, 2), (y, -4, 4), "test")
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)

    # multiple expressions
    p = plot3d(x + y, x * y)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[1].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[1].get_label(False) == "x*y"
    assert p[1].rendering_kw == {}

    # multiple expressions, same custom ranges
    p = plot3d(x + y, x * y, (x, -2, 2), (y, -4, 4))
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -2, 2)
    assert p[1].ranges[1] == (y, -4, 4)
    assert p[1].get_label(False) == "x*y"
    assert p[1].rendering_kw == {}

    # multiple expressions, custom ranges, labels and rendering kws
    p = plot3d(
        (x + y, (x, -2, 2), (y, -4, 4)),
        (x * y, (x, -3, 3), (y, -6, 6), "test", {"cmap": "Reds"}))
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -3, 3)
    assert p[1].ranges[1] == (y, -6, 6)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single expression: lambda function
    f = lambda x, y: x + y
    p = plot3d(f)
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert p[0].ranges[1][1:] == (-10, 10)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # single expression: lambda function + custom ranges + label
    p = plot3d(f, ("a", -5, 3), ("b", -2, 1), "test")
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-5, 3)
    assert p[0].ranges[1][1:] == (-2, 1)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # test issue 25818
    # single expression, custom range, min/max functions
    p = plot3d(Min(x, y), (x, 0, 10), (y, 0, 10))
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == Min(x, y)
    assert p[0].ranges[0] == (x, 0, 10)
    assert p[0].ranges[1] == (y, 0, 10)
    assert p[0].get_label(False) == "Min(x, y)"
    assert p[0].rendering_kw == {}


def test_plot3d_parametric_surface_arguments():
    ### Test arguments for plot3d_parametric_surface()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single parametric expression
    p = plot3d_parametric_surface(x + y, cos(x + y), sin(x + y))
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "(x + y, cos(x + y), sin(x + y))"
    assert p[0].rendering_kw == {}

    # single parametric expression, custom ranges, labels and rendering kws
    p = plot3d_parametric_surface(x + y, cos(x + y), sin(x + y),
        (x, -2, 2), (y, -4, 4), "test", {"cmap": "Reds"})
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    # multiple parametric expressions
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y)),
        (x - y, cos(x - y), sin(x - y), "test"))
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "(x + y, cos(x + y), sin(x + y))"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x - y, cos(x - y), sin(x - y))
    assert p[1].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[1].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions, custom ranges and labels
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y), (x, -2, 2), "test"),
        (x - y, cos(x - y), sin(x - y), (x, -3, 3), (y, -4, 4),
            "test2", {"cmap": "Reds"}))
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -10, 10)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x - y, cos(x - y), sin(x - y))
    assert p[1].ranges[0] == (x, -3, 3)
    assert p[1].ranges[1] == (y, -4, 4)
    assert p[1].get_label(False) == "test2"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # lambda functions instead of symbolic expressions for a single 3D
    # parametric surface
    p = plot3d_parametric_surface(
        lambda u, v: u, lambda u, v: v, lambda u, v: u + v,
        ("u", 0, 2), ("v", -3, 4))
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # lambda functions instead of symbolic expressions for multiple 3D
    # parametric surfaces
    p = plot3d_parametric_surface(
        (lambda u, v: u, lambda u, v: v, lambda u, v: u + v,
        ("u", 0, 2), ("v", -3, 4)),
        (lambda u, v: v, lambda u, v: u, lambda u, v: u - v,
        ("u", -2, 3), ("v", -4, 5), "test"))
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}
    assert all(callable(t) for t in p[1].expr)
    assert p[1].ranges[0][1:] == (-2, 3)
    assert p[1].ranges[1][1:] == (-4, 5)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

# === atelier-kyo-manager/test_11.py ===
# Generated by Selenium IDE
import pytest
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

class Test11():
  def setup_method(self, method):
    self.driver = webdriver.Firefox()
    self.vars = {}

  def teardown_method(self, method):
    self.driver.quit()

  def test_11(self):
    self.driver.get("https://www.buyma.com/")
    self.driver.set_window_size(1215, 775)
    self.driver.find_element(By.ID, "js-pc-header-login-link").click()
    self.driver.execute_script("window.scrollTo(0,182)")
    self.driver.execute_script("window.scrollTo(0,280.6666564941406)")
    self.driver.find_element(By.ID, "txtLoginId").click()
    self.driver.find_element(By.ID, "txtLoginId").click()
    self.driver.find_element(By.ID, "txtLoginId").send_keys("y.n.441622@gmail.com")
    self.driver.find_element(By.ID, "Sale_Bg").click()
    self.driver.find_element(By.ID, "txtLoginPass").click()
    self.driver.find_element(By.ID, "txtLoginPass").send_keys("kenshin4416")
    self.driver.find_element(By.ID, "login_do").click()
    self.driver.find_element(By.CSS_SELECTOR, ".my-page-profile__btn .fab-icon-dashboard-exhibit").click()
    element = self.driver.find_element(By.CSS_SELECTOR, ".my-page-profile__btn .fab-icon-dashboard-exhibit")
    actions = ActionChains(self.driver)
    actions.move_to_element(element).perform()
    self.driver.find_element(By.LINK_TEXT, "カタログから出品する").click()
    self.driver.find_element(By.CSS_SELECTOR, ".catalogs-table__row:nth-child(5) .catalogs-table__image-item:nth-child(1) > .catalogs-table__image").click()
    self.driver.find_element(By.LINK_TEXT, "利用特約に同意し、画像を保存する").click()


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\dtypes.py ===
"""
This module is home to specific dtypes related functionality and their classes.
For more general information about dtypes, also see `numpy.dtype` and
:ref:`arrays.dtypes`.

Similar to the builtin ``types`` module, this submodule defines types (classes)
that are not widely used directly.

.. versionadded:: NumPy 1.25

    The dtypes module is new in NumPy 1.25.  Previously DType classes were
    only accessible indirectly.


DType classes
-------------

The following are the classes of the corresponding NumPy dtype instances and
NumPy scalar types.  The classes can be used in ``isinstance`` checks and can
also be instantiated or used directly.  Direct use of these classes is not
typical, since their scalar counterparts (e.g. ``np.float64``) or strings
like ``"float64"`` can be used.
"""

# See doc/source/reference/routines.dtypes.rst for module-level docs

__all__ = []


def _add_dtype_helper(DType, alias):
    # Function to add DTypes a bit more conveniently without channeling them
    # through `numpy._core._multiarray_umath` namespace or similar.
    from numpy import dtypes

    setattr(dtypes, DType.__name__, DType)
    __all__.append(DType.__name__)

    if alias:
        alias = alias.removeprefix("numpy.dtypes.")
        setattr(dtypes, alias, DType)
        __all__.append(alias)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\series_factory.py ===
# Copyright (c) 2010-2024 openpyxl

from .data_source import NumDataSource, NumRef, AxDataSource
from .reference import Reference
from .series import Series, XYSeries, SeriesLabel, StrRef
from  openpyxl.utils import rows_from_range, quote_sheetname


def SeriesFactory(values, xvalues=None, zvalues=None, title=None, title_from_data=False):
    """
    Convenience Factory for creating chart data series.
    """

    if not isinstance(values, Reference):
        values = Reference(range_string=values)

    if title_from_data:
        cell = values.pop()
        title = u"{0}!{1}".format(values.sheetname, cell)
        title = SeriesLabel(strRef=StrRef(title))
    elif title is not None:
        title = SeriesLabel(v=title)

    source = NumDataSource(numRef=NumRef(f=values))
    if xvalues is not None:
        if not isinstance(xvalues, Reference):
            xvalues = Reference(range_string=xvalues)
        series = XYSeries()
        series.yVal = source
        series.xVal = AxDataSource(numRef=NumRef(f=xvalues))
        if zvalues is not None:
            if not isinstance(zvalues, Reference):
                zvalues = Reference(range_string=zvalues)
            series.zVal = NumDataSource(NumRef(f=zvalues))
    else:
        series = Series()
        series.val = source

    if title is not None:
        series.title = title
    return series

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chartsheet\protection.py ===
import hashlib

from openpyxl.descriptors import (Bool, Integer, String)
from openpyxl.descriptors.excel import Base64Binary
from openpyxl.descriptors.serialisable import Serialisable

from openpyxl.worksheet.protection import (
    hash_password,
    _Protected
)


class ChartsheetProtection(Serialisable, _Protected):
    tagname = "sheetProtection"

    algorithmName = String(allow_none=True)
    hashValue = Base64Binary(allow_none=True)
    saltValue = Base64Binary(allow_none=True)
    spinCount = Integer(allow_none=True)
    content = Bool(allow_none=True)
    objects = Bool(allow_none=True)

    __attrs__ = ("content", "objects", "password", "hashValue", "spinCount", "saltValue", "algorithmName")

    def __init__(self,
                 content=None,
                 objects=None,
                 hashValue=None,
                 spinCount=None,
                 saltValue=None,
                 algorithmName=None,
                 password=None,
                 ):
        self.content = content
        self.objects = objects
        self.hashValue = hashValue
        self.spinCount = spinCount
        self.saltValue = saltValue
        self.algorithmName = algorithmName
        if password is not None:
            self.password = password

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\descriptors\container.py ===
# Copyright (c) 2010-2024 openpyxl

"""
Utility list for top level containers that contain one type of element

Provides the necessary API to read and write XML
"""

from openpyxl.xml.functions import Element


class ElementList(list):


    @property
    def tagname(self):
        raise NotImplementedError


    @property
    def expected_type(self):
        raise NotImplementedError


    @classmethod
    def from_tree(cls, tree):
        l = [cls.expected_type.from_tree(el) for el in tree]
        return cls(l)


    def to_tree(self):
        container = Element(self.tagname)
        for el in self:
            container.append(el.to_tree())
        return container


    def append(self, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"Value must of type {self.expected_type} {type(value)} provided")
        super().append(value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\reshape\api.py ===
from pandas.core.reshape.concat import concat
from pandas.core.reshape.encoding import (
    from_dummies,
    get_dummies,
)
from pandas.core.reshape.melt import (
    lreshape,
    melt,
    wide_to_long,
)
from pandas.core.reshape.merge import (
    merge,
    merge_asof,
    merge_ordered,
)
from pandas.core.reshape.pivot import (
    crosstab,
    pivot,
    pivot_table,
)
from pandas.core.reshape.tile import (
    cut,
    qcut,
)

__all__ = [
    "concat",
    "crosstab",
    "cut",
    "from_dummies",
    "get_dummies",
    "lreshape",
    "melt",
    "merge",
    "merge_asof",
    "merge_ordered",
    "pivot",
    "pivot_table",
    "qcut",
    "wide_to_long",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\help.py ===
from optparse import Values
from typing import List

from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.exceptions import CommandError


class HelpCommand(Command):
    """Show help for commands"""

    usage = """
      %prog <command>"""
    ignore_require_venv = True

    def run(self, options: Values, args: List[str]) -> int:
        from pip._internal.commands import (
            commands_dict,
            create_command,
            get_similar_commands,
        )

        try:
            # 'pip help' with no args is handled by pip.__init__.parseopt()
            cmd_name = args[0]  # the command we need help for
        except IndexError:
            return SUCCESS

        if cmd_name not in commands_dict:
            guess = get_similar_commands(cmd_name)

            msg = [f'unknown command "{cmd_name}"']
            if guess:
                msg.append(f'maybe you meant "{guess}"')

            raise CommandError(" - ".join(msg))

        command = create_command(cmd_name)
        command.parser.print_help()

        return SUCCESS

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\build\metadata_editable.py ===
"""Metadata generation logic for source distributions."""

import os

from pip._vendor.pyproject_hooks import BuildBackendHookCaller

from pip._internal.build_env import BuildEnvironment
from pip._internal.exceptions import (
    InstallationSubprocessError,
    MetadataGenerationFailed,
)
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory


def generate_editable_metadata(
    build_env: BuildEnvironment, backend: BuildBackendHookCaller, details: str
) -> str:
    """Generate metadata using mechanisms described in PEP 660.

    Returns the generated metadata directory.
    """
    metadata_tmpdir = TempDirectory(kind="modern-metadata", globally_managed=True)

    metadata_dir = metadata_tmpdir.path

    with build_env:
        # Note that BuildBackendHookCaller implements a fallback for
        # prepare_metadata_for_build_wheel/editable, so we don't have to
        # consider the possibility that this hook doesn't exist.
        runner = runner_with_spinner_message(
            "Preparing editable metadata (pyproject.toml)"
        )
        with backend.subprocess_runner(runner):
            try:
                distinfo_dir = backend.prepare_metadata_for_build_editable(metadata_dir)
            except InstallationSubprocessError as error:
                raise MetadataGenerationFailed(package_details=details) from error

    assert distinfo_dir is not None
    return os.path.join(metadata_dir, distinfo_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\algorithms\pycosat_wrapper.py ===
from sympy.assumptions.cnf import EncodedCNF


def pycosat_satisfiable(expr, all_models=False):
    import pycosat
    if not isinstance(expr, EncodedCNF):
        exprs = EncodedCNF()
        exprs.add_prop(expr)
        expr = exprs

    # Return UNSAT when False (encoded as 0) is present in the CNF
    if {0} in expr.data:
        if all_models:
            return (f for f in [False])
        return False

    if not all_models:
        r = pycosat.solve(expr.data)
        result = (r != "UNSAT")
        if not result:
            return result
        return {expr.symbols[abs(lit) - 1]: lit > 0 for lit in r}
    else:
        r = pycosat.itersolve(expr.data)
        result = (r != "UNSAT")
        if not result:
            return result

        # Make solutions SymPy compatible by creating a generator
        def _gen(results):
            satisfiable = False
            try:
                while True:
                    sol = next(results)
                    yield {expr.symbols[abs(lit) - 1]: lit > 0 for lit in sol}
                    satisfiable = True
            except StopIteration:
                if not satisfiable:
                    yield False

        return _gen(r)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\config.py ===
import os
from dotenv import load_dotenv


def str2bool(value):
    return value.lower() in ['true', '1']


load_dotenv()


def ssl_verify():
    return str2bool(os.getenv("WDM_SSL_VERIFY", "true"))


def gh_token():
    return os.getenv("GH_TOKEN", None)


def wdm_local():
    return str2bool(os.getenv("WDM_LOCAL", "false"))


def wdm_log_level():
    default_level = 20
    try:
        return int(os.getenv("WDM_LOG", default_level))
    except Exception:
        return default_level


def wdm_progress_bar():
    default_level = 1
    try:
        return int(os.getenv("WDM_PROGRESS_BAR", default_level))
    except Exception:
        return default_level


def get_xdist_worker_id():
    return os.getenv("PYTEST_XDIST_WORKER", '')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_simd.py ===
# NOTE: Please avoid the use of numpy.testing since NPYV intrinsics
# may be involved in their functionality.
import itertools
import math
import operator
import re

import pytest
from numpy._core._multiarray_umath import __cpu_baseline__

from numpy._core._simd import clear_floatstatus, get_floatstatus, targets


def check_floatstatus(divbyzero=False, overflow=False,
                      underflow=False, invalid=False,
                      all=False):
    #define NPY_FPE_DIVIDEBYZERO  1
    #define NPY_FPE_OVERFLOW      2
    #define NPY_FPE_UNDERFLOW     4
    #define NPY_FPE_INVALID       8
    err = get_floatstatus()
    ret = (all or divbyzero) and (err & 1) != 0
    ret |= (all or overflow) and (err & 2) != 0
    ret |= (all or underflow) and (err & 4) != 0
    ret |= (all or invalid) and (err & 8) != 0
    return ret

class _Test_Utility:
    # submodule of the desired SIMD extension, e.g. targets["AVX512F"]
    npyv = None
    # the current data type suffix e.g. 's8'
    sfx = None
    # target name can be 'baseline' or one or more of CPU features
    target_name = None

    def __getattr__(self, attr):
        """
        To call NPV intrinsics without the attribute 'npyv' and
        auto suffixing intrinsics according to class attribute 'sfx'
        """
        return getattr(self.npyv, attr + "_" + self.sfx)

    def _x2(self, intrin_name):
        return getattr(self.npyv, f"{intrin_name}_{self.sfx}x2")

    def _data(self, start=None, count=None, reverse=False):
        """
        Create list of consecutive numbers according to number of vector's lanes.
        """
        if start is None:
            start = 1
        if count is None:
            count = self.nlanes
        rng = range(start, start + count)
        if reverse:
            rng = reversed(rng)
        if self._is_fp():
            return [x / 1.0 for x in rng]
        return list(rng)

    def _is_unsigned(self):
        return self.sfx[0] == 'u'

    def _is_signed(self):
        return self.sfx[0] == 's'

    def _is_fp(self):
        return self.sfx[0] == 'f'

    def _scalar_size(self):
        return int(self.sfx[1:])

    def _int_clip(self, seq):
        if self._is_fp():
            return seq
        max_int = self._int_max()
        min_int = self._int_min()
        return [min(max(v, min_int), max_int) for v in seq]

    def _int_max(self):
        if self._is_fp():
            return None
        max_u = self._to_unsigned(self.setall(-1))[0]
        if self._is_signed():
            return max_u // 2
        return max_u

    def _int_min(self):
        if self._is_fp():
            return None
        if self._is_unsigned():
            return 0
        return -(self._int_max() + 1)

    def _true_mask(self):
        max_unsig = getattr(self.npyv, "setall_u" + self.sfx[1:])(-1)
        return max_unsig[0]

    def _to_unsigned(self, vector):
        if isinstance(vector, (list, tuple)):
            return getattr(self.npyv, "load_u" + self.sfx[1:])(vector)
        else:
            sfx = vector.__name__.replace("npyv_", "")
            if sfx[0] == "b":
                cvt_intrin = "cvt_u{0}_b{0}"
            else:
                cvt_intrin = "reinterpret_u{0}_{1}"
            return getattr(self.npyv, cvt_intrin.format(sfx[1:], sfx))(vector)

    def _pinfinity(self):
        return float("inf")

    def _ninfinity(self):
        return -float("inf")

    def _nan(self):
        return float("nan")

    def _cpu_features(self):
        target = self.target_name
        if target == "baseline":
            target = __cpu_baseline__
        else:
            target = target.split('__')  # multi-target separator
        return ' '.join(target)

class _SIMD_BOOL(_Test_Utility):
    """
    To test all boolean vector types at once
    """
    def _nlanes(self):
        return getattr(self.npyv, "nlanes_u" + self.sfx[1:])

    def _data(self, start=None, count=None, reverse=False):
        true_mask = self._true_mask()
        rng = range(self._nlanes())
        if reverse:
            rng = reversed(rng)
        return [true_mask if x % 2 else 0 for x in rng]

    def _load_b(self, data):
        len_str = self.sfx[1:]
        load = getattr(self.npyv, "load_u" + len_str)
        cvt = getattr(self.npyv, f"cvt_b{len_str}_u{len_str}")
        return cvt(load(data))

    def test_operators_logical(self):
        """
        Logical operations for boolean types.
        Test intrinsics:
            npyv_xor_##SFX, npyv_and_##SFX, npyv_or_##SFX, npyv_not_##SFX,
            npyv_andc_b8, npvy_orc_b8, nvpy_xnor_b8
        """
        data_a = self._data()
        data_b = self._data(reverse=True)
        vdata_a = self._load_b(data_a)
        vdata_b = self._load_b(data_b)

        data_and = [a & b for a, b in zip(data_a, data_b)]
        vand = getattr(self, "and")(vdata_a, vdata_b)
        assert vand == data_and

        data_or = [a | b for a, b in zip(data_a, data_b)]
        vor = getattr(self, "or")(vdata_a, vdata_b)
        assert vor == data_or

        data_xor = [a ^ b for a, b in zip(data_a, data_b)]
        vxor = self.xor(vdata_a, vdata_b)
        assert vxor == data_xor

        vnot = getattr(self, "not")(vdata_a)
        assert vnot == data_b

        # among the boolean types, andc, orc and xnor only support b8
        if self.sfx not in ("b8"):
            return

        data_andc = [(a & ~b) & 0xFF for a, b in zip(data_a, data_b)]
        vandc = self.andc(vdata_a, vdata_b)
        assert data_andc == vandc

        data_orc = [(a | ~b) & 0xFF for a, b in zip(data_a, data_b)]
        vorc = self.orc(vdata_a, vdata_b)
        assert data_orc == vorc

        data_xnor = [~(a ^ b) & 0xFF for a, b in zip(data_a, data_b)]
        vxnor = self.xnor(vdata_a, vdata_b)
        assert data_xnor == vxnor

    def test_tobits(self):
        data2bits = lambda data: sum(int(x != 0) << i for i, x in enumerate(data, 0))
        for data in (self._data(), self._data(reverse=True)):
            vdata = self._load_b(data)
            data_bits = data2bits(data)
            tobits = self.tobits(vdata)
            bin_tobits = bin(tobits)
            assert bin_tobits == bin(data_bits)

    def test_pack(self):
        """
        Pack multiple vectors into one
        Test intrinsics:
            npyv_pack_b8_b16
            npyv_pack_b8_b32
            npyv_pack_b8_b64
        """
        if self.sfx not in ("b16", "b32", "b64"):
            return
        # create the vectors
        data = self._data()
        rdata = self._data(reverse=True)
        vdata = self._load_b(data)
        vrdata = self._load_b(rdata)
        pack_simd = getattr(self.npyv, f"pack_b8_{self.sfx}")
        # for scalar execution, concatenate the elements of the multiple lists
        # into a single list (spack) and then iterate over the elements of
        # the created list applying a mask to capture the first byte of them.
        if self.sfx == "b16":
            spack = [(i & 0xFF) for i in (list(rdata) + list(data))]
            vpack = pack_simd(vrdata, vdata)
        elif self.sfx == "b32":
            spack = [(i & 0xFF) for i in (2 * list(rdata) + 2 * list(data))]
            vpack = pack_simd(vrdata, vrdata, vdata, vdata)
        elif self.sfx == "b64":
            spack = [(i & 0xFF) for i in (4 * list(rdata) + 4 * list(data))]
            vpack = pack_simd(vrdata, vrdata, vrdata, vrdata,
                               vdata,  vdata,  vdata,  vdata)
        assert vpack == spack

    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [-1, 0],
        [0, -1],
        [-1],
        [0]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        data_a = self._load_b(data * self._nlanes())
        func = eval(intrin)
        intrin = getattr(self, intrin)
        desired = func(data_a)
        simd = intrin(data_a)
        assert not not simd == desired

class _SIMD_INT(_Test_Utility):
    """
    To test all integer vector types at once
    """
    def test_operators_shift(self):
        if self.sfx in ("u8", "s8"):
            return

        data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        for count in range(self._scalar_size()):
            # load to cast
            data_shl_a = self.load([a << count for a in data_a])
            # left shift
            shl = self.shl(vdata_a, count)
            assert shl == data_shl_a
            # load to cast
            data_shr_a = self.load([a >> count for a in data_a])
            # right shift
            shr = self.shr(vdata_a, count)
            assert shr == data_shr_a

        # shift by zero or max or out-range immediate constant is not applicable and illogical
        for count in range(1, self._scalar_size()):
            # load to cast
            data_shl_a = self.load([a << count for a in data_a])
            # left shift by an immediate constant
            shli = self.shli(vdata_a, count)
            assert shli == data_shl_a
            # load to cast
            data_shr_a = self.load([a >> count for a in data_a])
            # right shift by an immediate constant
            shri = self.shri(vdata_a, count)
            assert shri == data_shr_a

    def test_arithmetic_subadd_saturated(self):
        if self.sfx in ("u32", "s32", "u64", "s64"):
            return

        data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        data_adds = self._int_clip([a + b for a, b in zip(data_a, data_b)])
        adds = self.adds(vdata_a, vdata_b)
        assert adds == data_adds

        data_subs = self._int_clip([a - b for a, b in zip(data_a, data_b)])
        subs = self.subs(vdata_a, vdata_b)
        assert subs == data_subs

    def test_math_max_min(self):
        data_a = self._data()
        data_b = self._data(self.nlanes)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        data_max = [max(a, b) for a, b in zip(data_a, data_b)]
        simd_max = self.max(vdata_a, vdata_b)
        assert simd_max == data_max

        data_min = [min(a, b) for a, b in zip(data_a, data_b)]
        simd_min = self.min(vdata_a, vdata_b)
        assert simd_min == data_min

    @pytest.mark.parametrize("start", [-100, -10000, 0, 100, 10000])
    def test_reduce_max_min(self, start):
        """
        Test intrinsics:
            npyv_reduce_max_##sfx
            npyv_reduce_min_##sfx
        """
        vdata_a = self.load(self._data(start))
        assert self.reduce_max(vdata_a) == max(vdata_a)
        assert self.reduce_min(vdata_a) == min(vdata_a)


class _SIMD_FP32(_Test_Utility):
    """
    To only test single precision
    """
    def test_conversions(self):
        """
        Round to nearest even integer, assume CPU control register is set to rounding.
        Test intrinsics:
            npyv_round_s32_##SFX
        """
        features = self._cpu_features()
        if not self.npyv.simd_f64 and re.match(r".*(NEON|ASIMD)", features):
            # very costly to emulate nearest even on Armv7
            # instead we round halves to up. e.g. 0.5 -> 1, -0.5 -> -1
            _round = lambda v: int(v + (0.5 if v >= 0 else -0.5))
        else:
            _round = round
        vdata_a = self.load(self._data())
        vdata_a = self.sub(vdata_a, self.setall(0.5))
        data_round = [_round(x) for x in vdata_a]
        vround = self.round_s32(vdata_a)
        assert vround == data_round

class _SIMD_FP64(_Test_Utility):
    """
    To only test double precision
    """
    def test_conversions(self):
        """
        Round to nearest even integer, assume CPU control register is set to rounding.
        Test intrinsics:
            npyv_round_s32_##SFX
        """
        vdata_a = self.load(self._data())
        vdata_a = self.sub(vdata_a, self.setall(0.5))
        vdata_b = self.mul(vdata_a, self.setall(-1.5))
        data_round = [round(x) for x in list(vdata_a) + list(vdata_b)]
        vround = self.round_s32(vdata_a, vdata_b)
        assert vround == data_round

class _SIMD_FP(_Test_Utility):
    """
    To test all float vector types at once
    """
    def test_arithmetic_fused(self):
        vdata_a, vdata_b, vdata_c = [self.load(self._data())] * 3
        vdata_cx2 = self.add(vdata_c, vdata_c)
        # multiply and add, a*b + c
        data_fma = self.load([a * b + c for a, b, c in zip(vdata_a, vdata_b, vdata_c)])
        fma = self.muladd(vdata_a, vdata_b, vdata_c)
        assert fma == data_fma
        # multiply and subtract, a*b - c
        fms = self.mulsub(vdata_a, vdata_b, vdata_c)
        data_fms = self.sub(data_fma, vdata_cx2)
        assert fms == data_fms
        # negate multiply and add, -(a*b) + c
        nfma = self.nmuladd(vdata_a, vdata_b, vdata_c)
        data_nfma = self.sub(vdata_cx2, data_fma)
        assert nfma == data_nfma
        # negate multiply and subtract, -(a*b) - c
        nfms = self.nmulsub(vdata_a, vdata_b, vdata_c)
        data_nfms = self.mul(data_fma, self.setall(-1))
        assert nfms == data_nfms
        # multiply, add for odd elements and subtract even elements.
        # (a * b) -+ c
        fmas = list(self.muladdsub(vdata_a, vdata_b, vdata_c))
        assert fmas[0::2] == list(data_fms)[0::2]
        assert fmas[1::2] == list(data_fma)[1::2]

    def test_abs(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())

        abs_cases = ((-0, 0), (ninf, pinf), (pinf, pinf), (nan, nan))
        for case, desired in abs_cases:
            data_abs = [desired] * self.nlanes
            vabs = self.abs(self.setall(case))
            assert vabs == pytest.approx(data_abs, nan_ok=True)

        vabs = self.abs(self.mul(vdata, self.setall(-1)))
        assert vabs == data

    def test_sqrt(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())

        sqrt_cases = ((-0.0, -0.0), (0.0, 0.0), (-1.0, nan), (ninf, nan), (pinf, pinf))
        for case, desired in sqrt_cases:
            data_sqrt = [desired] * self.nlanes
            sqrt = self.sqrt(self.setall(case))
            assert sqrt == pytest.approx(data_sqrt, nan_ok=True)

        data_sqrt = self.load([math.sqrt(x) for x in data])  # load to truncate precision
        sqrt = self.sqrt(vdata)
        assert sqrt == data_sqrt

    def test_square(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())
        # square
        square_cases = ((nan, nan), (pinf, pinf), (ninf, pinf))
        for case, desired in square_cases:
            data_square = [desired] * self.nlanes
            square = self.square(self.setall(case))
            assert square == pytest.approx(data_square, nan_ok=True)

        data_square = [x * x for x in data]
        square = self.square(vdata)
        assert square == data_square

    @pytest.mark.parametrize("intrin, func", [("ceil", math.ceil),
    ("trunc", math.trunc), ("floor", math.floor), ("rint", round)])
    def test_rounding(self, intrin, func):
        """
        Test intrinsics:
            npyv_rint_##SFX
            npyv_ceil_##SFX
            npyv_trunc_##SFX
            npyv_floor##SFX
        """
        intrin_name = intrin
        intrin = getattr(self, intrin)
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        # special cases
        round_cases = ((nan, nan), (pinf, pinf), (ninf, ninf))
        for case, desired in round_cases:
            data_round = [desired] * self.nlanes
            _round = intrin(self.setall(case))
            assert _round == pytest.approx(data_round, nan_ok=True)

        for x in range(0, 2**20, 256**2):
            for w in (-1.05, -1.10, -1.15, 1.05, 1.10, 1.15):
                data = self.load([(x + a) * w for a in range(self.nlanes)])
                data_round = [func(x) for x in data]
                _round = intrin(data)
                assert _round == data_round

        # test large numbers
        for i in (
            1.1529215045988576e+18, 4.6116860183954304e+18,
            5.902958103546122e+20, 2.3611832414184488e+21
        ):
            x = self.setall(i)
            y = intrin(x)
            data_round = [func(n) for n in x]
            assert y == data_round

        # signed zero
        if intrin_name == "floor":
            data_szero = (-0.0,)
        else:
            data_szero = (-0.0, -0.25, -0.30, -0.45, -0.5)

        for w in data_szero:
            _round = self._to_unsigned(intrin(self.setall(w)))
            data_round = self._to_unsigned(self.setall(-0.0))
            assert _round == data_round

    @pytest.mark.parametrize("intrin", [
        "max", "maxp", "maxn", "min", "minp", "minn"
    ])
    def test_max_min(self, intrin):
        """
        Test intrinsics:
            npyv_max_##sfx
            npyv_maxp_##sfx
            npyv_maxn_##sfx
            npyv_min_##sfx
            npyv_minp_##sfx
            npyv_minn_##sfx
            npyv_reduce_max_##sfx
            npyv_reduce_maxp_##sfx
            npyv_reduce_maxn_##sfx
            npyv_reduce_min_##sfx
            npyv_reduce_minp_##sfx
            npyv_reduce_minn_##sfx
        """
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        chk_nan = {"xp": 1, "np": 1, "nn": 2, "xn": 2}.get(intrin[-2:], 0)
        func = eval(intrin[:3])
        reduce_intrin = getattr(self, "reduce_" + intrin)
        intrin = getattr(self, intrin)
        hf_nlanes = self.nlanes // 2

        cases = (
            ([0.0, -0.0], [-0.0, 0.0]),
            ([10, -10],  [10, -10]),
            ([pinf, 10], [10, ninf]),
            ([10, pinf], [ninf, 10]),
            ([10, -10], [10, -10]),
            ([-10, 10], [-10, 10])
        )
        for op1, op2 in cases:
            vdata_a = self.load(op1 * hf_nlanes)
            vdata_b = self.load(op2 * hf_nlanes)
            data = func(vdata_a, vdata_b)
            simd = intrin(vdata_a, vdata_b)
            assert simd == data
            data = func(vdata_a)
            simd = reduce_intrin(vdata_a)
            assert simd == data

        if not chk_nan:
            return
        if chk_nan == 1:
            test_nan = lambda a, b: (
                b if math.isnan(a) else a if math.isnan(b) else b
            )
        else:
            test_nan = lambda a, b: (
                nan if math.isnan(a) or math.isnan(b) else b
            )
        cases = (
            (nan, 10),
            (10, nan),
            (nan, pinf),
            (pinf, nan),
            (nan, nan)
        )
        for op1, op2 in cases:
            vdata_ab = self.load([op1, op2] * hf_nlanes)
            data = test_nan(op1, op2)
            simd = reduce_intrin(vdata_ab)
            assert simd == pytest.approx(data, nan_ok=True)
            vdata_a = self.setall(op1)
            vdata_b = self.setall(op2)
            data = [data] * self.nlanes
            simd = intrin(vdata_a, vdata_b)
            assert simd == pytest.approx(data, nan_ok=True)

    def test_reciprocal(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())

        recip_cases = ((nan, nan), (pinf, 0.0), (ninf, -0.0), (0.0, pinf), (-0.0, ninf))
        for case, desired in recip_cases:
            data_recip = [desired] * self.nlanes
            recip = self.recip(self.setall(case))
            assert recip == pytest.approx(data_recip, nan_ok=True)

        data_recip = self.load([1 / x for x in data])  # load to truncate precision
        recip = self.recip(vdata)
        assert recip == data_recip

    def test_special_cases(self):
        """
        Compare Not NaN. Test intrinsics:
            npyv_notnan_##SFX
        """
        nnan = self.notnan(self.setall(self._nan()))
        assert nnan == [0] * self.nlanes

    @pytest.mark.parametrize("intrin_name", [
        "rint", "trunc", "ceil", "floor"
    ])
    def test_unary_invalid_fpexception(self, intrin_name):
        intrin = getattr(self, intrin_name)
        for d in [float("nan"), float("inf"), -float("inf")]:
            v = self.setall(d)
            clear_floatstatus()
            intrin(v)
            assert check_floatstatus(invalid=True) is False

    @pytest.mark.parametrize('py_comp,np_comp', [
        (operator.lt, "cmplt"),
        (operator.le, "cmple"),
        (operator.gt, "cmpgt"),
        (operator.ge, "cmpge"),
        (operator.eq, "cmpeq"),
        (operator.ne, "cmpneq")
    ])
    def test_comparison_with_nan(self, py_comp, np_comp):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        mask_true = self._true_mask()

        def to_bool(vector):
            return [lane == mask_true for lane in vector]

        intrin = getattr(self, np_comp)
        cmp_cases = ((0, nan), (nan, 0), (nan, nan), (pinf, nan),
                     (ninf, nan), (-0.0, +0.0))
        for case_operand1, case_operand2 in cmp_cases:
            data_a = [case_operand1] * self.nlanes
            data_b = [case_operand2] * self.nlanes
            vdata_a = self.setall(case_operand1)
            vdata_b = self.setall(case_operand2)
            vcmp = to_bool(intrin(vdata_a, vdata_b))
            data_cmp = [py_comp(a, b) for a, b in zip(data_a, data_b)]
            assert vcmp == data_cmp

    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [float("nan"), 0],
        [0, float("nan")],
        [float("nan"), 1],
        [1, float("nan")],
        [float("nan"), float("nan")],
        [0.0, -0.0],
        [-0.0, 0.0],
        [1.0, -0.0]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        data_a = self.load(data * self.nlanes)
        func = eval(intrin)
        intrin = getattr(self, intrin)
        desired = func(data_a)
        simd = intrin(data_a)
        assert not not simd == desired

class _SIMD_ALL(_Test_Utility):
    """
    To test all vector types at once
    """
    def test_memory_load(self):
        data = self._data()
        # unaligned load
        load_data = self.load(data)
        assert load_data == data
        # aligned load
        loada_data = self.loada(data)
        assert loada_data == data
        # stream load
        loads_data = self.loads(data)
        assert loads_data == data
        # load lower part
        loadl = self.loadl(data)
        loadl_half = list(loadl)[:self.nlanes // 2]
        data_half = data[:self.nlanes // 2]
        assert loadl_half == data_half
        assert loadl != data  # detect overflow

    def test_memory_store(self):
        data = self._data()
        vdata = self.load(data)
        # unaligned store
        store = [0] * self.nlanes
        self.store(store, vdata)
        assert store == data
        # aligned store
        store_a = [0] * self.nlanes
        self.storea(store_a, vdata)
        assert store_a == data
        # stream store
        store_s = [0] * self.nlanes
        self.stores(store_s, vdata)
        assert store_s == data
        # store lower part
        store_l = [0] * self.nlanes
        self.storel(store_l, vdata)
        assert store_l[:self.nlanes // 2] == data[:self.nlanes // 2]
        assert store_l != vdata  # detect overflow
        # store higher part
        store_h = [0] * self.nlanes
        self.storeh(store_h, vdata)
        assert store_h[:self.nlanes // 2] == data[self.nlanes // 2:]
        assert store_h != vdata  # detect overflow

    @pytest.mark.parametrize("intrin, elsizes, scale, fill", [
        ("self.load_tillz, self.load_till", (32, 64), 1, [0xffff]),
        ("self.load2_tillz, self.load2_till", (32, 64), 2, [0xffff, 0x7fff]),
    ])
    def test_memory_partial_load(self, intrin, elsizes, scale, fill):
        if self._scalar_size() not in elsizes:
            return
        npyv_load_tillz, npyv_load_till = eval(intrin)
        data = self._data()
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]  # test out of range
        for n in lanes:
            load_till = npyv_load_till(data, n, *fill)
            load_tillz = npyv_load_tillz(data, n)
            n *= scale
            data_till = data[:n] + fill * ((self.nlanes - n) // scale)
            assert load_till == data_till
            data_tillz = data[:n] + [0] * (self.nlanes - n)
            assert load_tillz == data_tillz

    @pytest.mark.parametrize("intrin, elsizes, scale", [
        ("self.store_till", (32, 64), 1),
        ("self.store2_till", (32, 64), 2),
    ])
    def test_memory_partial_store(self, intrin, elsizes, scale):
        if self._scalar_size() not in elsizes:
            return
        npyv_store_till = eval(intrin)
        data = self._data()
        data_rev = self._data(reverse=True)
        vdata = self.load(data)
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]
        for n in lanes:
            data_till = data_rev.copy()
            data_till[:n * scale] = data[:n * scale]
            store_till = self._data(reverse=True)
            npyv_store_till(store_till, n, vdata)
            assert store_till == data_till

    @pytest.mark.parametrize("intrin, elsizes, scale", [
        ("self.loadn", (32, 64), 1),
        ("self.loadn2", (32, 64), 2),
    ])
    def test_memory_noncont_load(self, intrin, elsizes, scale):
        if self._scalar_size() not in elsizes:
            return
        npyv_loadn = eval(intrin)
        for stride in range(-64, 64):
            if stride < 0:
                data = self._data(stride, -stride * self.nlanes)
                data_stride = list(itertools.chain(
                    *zip(*[data[-i::stride] for i in range(scale, 0, -1)])
                ))
            elif stride == 0:
                data = self._data()
                data_stride = data[0:scale] * (self.nlanes // scale)
            else:
                data = self._data(count=stride * self.nlanes)
                data_stride = list(itertools.chain(
                    *zip(*[data[i::stride] for i in range(scale)]))
                )
            data_stride = self.load(data_stride)  # cast unsigned
            loadn = npyv_loadn(data, stride)
            assert loadn == data_stride

    @pytest.mark.parametrize("intrin, elsizes, scale, fill", [
        ("self.loadn_tillz, self.loadn_till", (32, 64), 1, [0xffff]),
        ("self.loadn2_tillz, self.loadn2_till", (32, 64), 2, [0xffff, 0x7fff]),
    ])
    def test_memory_noncont_partial_load(self, intrin, elsizes, scale, fill):
        if self._scalar_size() not in elsizes:
            return
        npyv_loadn_tillz, npyv_loadn_till = eval(intrin)
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]
        for stride in range(-64, 64):
            if stride < 0:
                data = self._data(stride, -stride * self.nlanes)
                data_stride = list(itertools.chain(
                    *zip(*[data[-i::stride] for i in range(scale, 0, -1)])
                ))
            elif stride == 0:
                data = self._data()
                data_stride = data[0:scale] * (self.nlanes // scale)
            else:
                data = self._data(count=stride * self.nlanes)
                data_stride = list(itertools.chain(
                    *zip(*[data[i::stride] for i in range(scale)])
                ))
            data_stride = list(self.load(data_stride))  # cast unsigned
            for n in lanes:
                nscale = n * scale
                llanes = self.nlanes - nscale
                data_stride_till = (
                    data_stride[:nscale] + fill * (llanes // scale)
                )
                loadn_till = npyv_loadn_till(data, stride, n, *fill)
                assert loadn_till == data_stride_till
                data_stride_tillz = data_stride[:nscale] + [0] * llanes
                loadn_tillz = npyv_loadn_tillz(data, stride, n)
                assert loadn_tillz == data_stride_tillz

    @pytest.mark.parametrize("intrin, elsizes, scale", [
        ("self.storen", (32, 64), 1),
        ("self.storen2", (32, 64), 2),
    ])
    def test_memory_noncont_store(self, intrin, elsizes, scale):
        if self._scalar_size() not in elsizes:
            return
        npyv_storen = eval(intrin)
        data = self._data()
        vdata = self.load(data)
        hlanes = self.nlanes // scale
        for stride in range(1, 64):
            data_storen = [0xff] * stride * self.nlanes
            for s in range(0, hlanes * stride, stride):
                i = (s // stride) * scale
                data_storen[s:s + scale] = data[i:i + scale]
            storen = [0xff] * stride * self.nlanes
            storen += [0x7f] * 64
            npyv_storen(storen, stride, vdata)
            assert storen[:-64] == data_storen
            assert storen[-64:] == [0x7f] * 64  # detect overflow

        for stride in range(-64, 0):
            data_storen = [0xff] * -stride * self.nlanes
            for s in range(0, hlanes * stride, stride):
                i = (s // stride) * scale
                data_storen[s - scale:s or None] = data[i:i + scale]
            storen = [0x7f] * 64
            storen += [0xff] * -stride * self.nlanes
            npyv_storen(storen, stride, vdata)
            assert storen[64:] == data_storen
            assert storen[:64] == [0x7f] * 64  # detect overflow
        # stride 0
        data_storen = [0x7f] * self.nlanes
        storen = data_storen.copy()
        data_storen[0:scale] = data[-scale:]
        npyv_storen(storen, 0, vdata)
        assert storen == data_storen

    @pytest.mark.parametrize("intrin, elsizes, scale", [
        ("self.storen_till", (32, 64), 1),
        ("self.storen2_till", (32, 64), 2),
    ])
    def test_memory_noncont_partial_store(self, intrin, elsizes, scale):
        if self._scalar_size() not in elsizes:
            return
        npyv_storen_till = eval(intrin)
        data = self._data()
        vdata = self.load(data)
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]
        hlanes = self.nlanes // scale
        for stride in range(1, 64):
            for n in lanes:
                data_till = [0xff] * stride * self.nlanes
                tdata = data[:n * scale] + [0xff] * (self.nlanes - n * scale)
                for s in range(0, hlanes * stride, stride)[:n]:
                    i = (s // stride) * scale
                    data_till[s:s + scale] = tdata[i:i + scale]
                storen_till = [0xff] * stride * self.nlanes
                storen_till += [0x7f] * 64
                npyv_storen_till(storen_till, stride, n, vdata)
                assert storen_till[:-64] == data_till
                assert storen_till[-64:] == [0x7f] * 64  # detect overflow

        for stride in range(-64, 0):
            for n in lanes:
                data_till = [0xff] * -stride * self.nlanes
                tdata = data[:n * scale] + [0xff] * (self.nlanes - n * scale)
                for s in range(0, hlanes * stride, stride)[:n]:
                    i = (s // stride) * scale
                    data_till[s - scale:s or None] = tdata[i:i + scale]
                storen_till = [0x7f] * 64
                storen_till += [0xff] * -stride * self.nlanes
                npyv_storen_till(storen_till, stride, n, vdata)
                assert storen_till[64:] == data_till
                assert storen_till[:64] == [0x7f] * 64  # detect overflow

        # stride 0
        for n in lanes:
            data_till = [0x7f] * self.nlanes
            storen_till = data_till.copy()
            data_till[0:scale] = data[:n * scale][-scale:]
            npyv_storen_till(storen_till, 0, n, vdata)
            assert storen_till == data_till

    @pytest.mark.parametrize("intrin, table_size, elsize", [
        ("self.lut32", 32, 32),
        ("self.lut16", 16, 64)
    ])
    def test_lut(self, intrin, table_size, elsize):
        """
        Test lookup table intrinsics:
            npyv_lut32_##sfx
            npyv_lut16_##sfx
        """
        if elsize != self._scalar_size():
            return
        intrin = eval(intrin)
        idx_itrin = getattr(self.npyv, f"setall_u{elsize}")
        table = range(table_size)
        for i in table:
            broadi = self.setall(i)
            idx = idx_itrin(i)
            lut = intrin(table, idx)
            assert lut == broadi

    def test_misc(self):
        broadcast_zero = self.zero()
        assert broadcast_zero == [0] * self.nlanes
        for i in range(1, 10):
            broadcasti = self.setall(i)
            assert broadcasti == [i] * self.nlanes

        data_a, data_b = self._data(), self._data(reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # py level of npyv_set_* don't support ignoring the extra specified lanes or
        # fill non-specified lanes with zero.
        vset = self.set(*data_a)
        assert vset == data_a
        # py level of npyv_setf_* don't support ignoring the extra specified lanes or
        # fill non-specified lanes with the specified scalar.
        vsetf = self.setf(10, *data_a)
        assert vsetf == data_a

        # We're testing the sanity of _simd's type-vector,
        # reinterpret* intrinsics itself are tested via compiler
        # during the build of _simd module
        sfxes = ["u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64"]
        if self.npyv.simd_f64:
            sfxes.append("f64")
        if self.npyv.simd_f32:
            sfxes.append("f32")
        for sfx in sfxes:
            vec_name = getattr(self, "reinterpret_" + sfx)(vdata_a).__name__
            assert vec_name == "npyv_" + sfx

        # select & mask operations
        select_a = self.select(self.cmpeq(self.zero(), self.zero()), vdata_a, vdata_b)
        assert select_a == data_a
        select_b = self.select(self.cmpneq(self.zero(), self.zero()), vdata_a, vdata_b)
        assert select_b == data_b

        # test extract elements
        assert self.extract0(vdata_b) == vdata_b[0]

        # cleanup intrinsic is only used with AVX for
        # zeroing registers to avoid the AVX-SSE transition penalty,
        # so nothing to test here
        self.npyv.cleanup()

    def test_reorder(self):
        data_a, data_b = self._data(), self._data(reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)
        # lower half part
        data_a_lo = data_a[:self.nlanes // 2]
        data_b_lo = data_b[:self.nlanes // 2]
        # higher half part
        data_a_hi = data_a[self.nlanes // 2:]
        data_b_hi = data_b[self.nlanes // 2:]
        # combine two lower parts
        combinel = self.combinel(vdata_a, vdata_b)
        assert combinel == data_a_lo + data_b_lo
        # combine two higher parts
        combineh = self.combineh(vdata_a, vdata_b)
        assert combineh == data_a_hi + data_b_hi
        # combine x2
        combine = self.combine(vdata_a, vdata_b)
        assert combine == (data_a_lo + data_b_lo, data_a_hi + data_b_hi)

        # zip(interleave)
        data_zipl = self.load([
            v for p in zip(data_a_lo, data_b_lo) for v in p
        ])
        data_ziph = self.load([
            v for p in zip(data_a_hi, data_b_hi) for v in p
        ])
        vzip = self.zip(vdata_a, vdata_b)
        assert vzip == (data_zipl, data_ziph)
        vzip = [0] * self.nlanes * 2
        self._x2("store")(vzip, (vdata_a, vdata_b))
        assert vzip == list(data_zipl) + list(data_ziph)

        # unzip(deinterleave)
        unzip = self.unzip(data_zipl, data_ziph)
        assert unzip == (data_a, data_b)
        unzip = self._x2("load")(list(data_zipl) + list(data_ziph))
        assert unzip == (data_a, data_b)

    def test_reorder_rev64(self):
        # Reverse elements of each 64-bit lane
        ssize = self._scalar_size()
        if ssize == 64:
            return
        data_rev64 = [
            y for x in range(0, self.nlanes, 64 // ssize)
              for y in reversed(range(x, x + 64 // ssize))
        ]
        rev64 = self.rev64(self.load(range(self.nlanes)))
        assert rev64 == data_rev64

    def test_reorder_permi128(self):
        """
        Test permuting elements for each 128-bit lane.
        npyv_permi128_##sfx
        """
        ssize = self._scalar_size()
        if ssize < 32:
            return
        data = self.load(self._data())
        permn = 128 // ssize
        permd = permn - 1
        nlane128 = self.nlanes // permn
        shfl = [0, 1] if ssize == 64 else [0, 2, 4, 6]
        for i in range(permn):
            indices = [(i >> shf) & permd for shf in shfl]
            vperm = self.permi128(data, *indices)
            data_vperm = [
                data[j + (e & -permn)]
                for e, j in enumerate(indices * nlane128)
            ]
            assert vperm == data_vperm

    @pytest.mark.parametrize('func, intrin', [
        (operator.lt, "cmplt"),
        (operator.le, "cmple"),
        (operator.gt, "cmpgt"),
        (operator.ge, "cmpge"),
        (operator.eq, "cmpeq")
    ])
    def test_operators_comparison(self, func, intrin):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)
        intrin = getattr(self, intrin)

        mask_true = self._true_mask()

        def to_bool(vector):
            return [lane == mask_true for lane in vector]

        data_cmp = [func(a, b) for a, b in zip(data_a, data_b)]
        cmp = to_bool(intrin(vdata_a, vdata_b))
        assert cmp == data_cmp

    def test_operators_logical(self):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        if self._is_fp():
            data_cast_a = self._to_unsigned(vdata_a)
            data_cast_b = self._to_unsigned(vdata_b)
            cast, cast_data = self._to_unsigned, self._to_unsigned
        else:
            data_cast_a, data_cast_b = data_a, data_b
            cast, cast_data = lambda a: a, self.load

        data_xor = cast_data([a ^ b for a, b in zip(data_cast_a, data_cast_b)])
        vxor = cast(self.xor(vdata_a, vdata_b))
        assert vxor == data_xor

        data_or = cast_data([a | b for a, b in zip(data_cast_a, data_cast_b)])
        vor = cast(getattr(self, "or")(vdata_a, vdata_b))
        assert vor == data_or

        data_and = cast_data([a & b for a, b in zip(data_cast_a, data_cast_b)])
        vand = cast(getattr(self, "and")(vdata_a, vdata_b))
        assert vand == data_and

        data_not = cast_data([~a for a in data_cast_a])
        vnot = cast(getattr(self, "not")(vdata_a))
        assert vnot == data_not

        if self.sfx not in ("u8"):
            return
        data_andc = [a & ~b for a, b in zip(data_cast_a, data_cast_b)]
        vandc = cast(self.andc(vdata_a, vdata_b))
        assert vandc == data_andc

    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
        [0, 1, 2, 3, 4],
        [0x7f, 0x7fff, 0x7fffffff, 0x7fffffffffffffff],
        [0, -1, -2, -3, 4],
        [0],
        [1],
        [-1]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        data_a = self.load(data * self.nlanes)
        func = eval(intrin)
        intrin = getattr(self, intrin)
        desired = func(data_a)
        simd = intrin(data_a)
        assert not not simd == desired

    def test_conversion_boolean(self):
        bsfx = "b" + self.sfx[1:]
        to_boolean = getattr(self.npyv, f"cvt_{bsfx}_{self.sfx}")
        from_boolean = getattr(self.npyv, f"cvt_{self.sfx}_{bsfx}")

        false_vb = to_boolean(self.setall(0))
        true_vb = self.cmpeq(self.setall(0), self.setall(0))
        assert false_vb != true_vb

        false_vsfx = from_boolean(false_vb)
        true_vsfx = from_boolean(true_vb)
        assert false_vsfx != true_vsfx

    def test_conversion_expand(self):
        """
        Test expand intrinsics:
            npyv_expand_u16_u8
            npyv_expand_u32_u16
        """
        if self.sfx not in ("u8", "u16"):
            return
        totype = self.sfx[0] + str(int(self.sfx[1:]) * 2)
        expand = getattr(self.npyv, f"expand_{totype}_{self.sfx}")
        # close enough from the edge to detect any deviation
        data = self._data(self._int_max() - self.nlanes)
        vdata = self.load(data)
        edata = expand(vdata)
        # lower half part
        data_lo = data[:self.nlanes // 2]
        # higher half part
        data_hi = data[self.nlanes // 2:]
        assert edata == (data_lo, data_hi)

    def test_arithmetic_subadd(self):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # non-saturated
        data_add = self.load([a + b for a, b in zip(data_a, data_b)])  # load to cast
        add = self.add(vdata_a, vdata_b)
        assert add == data_add
        data_sub = self.load([a - b for a, b in zip(data_a, data_b)])
        sub = self.sub(vdata_a, vdata_b)
        assert sub == data_sub

    def test_arithmetic_mul(self):
        if self.sfx in ("u64", "s64"):
            return

        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        data_mul = self.load([a * b for a, b in zip(data_a, data_b)])
        mul = self.mul(vdata_a, vdata_b)
        assert mul == data_mul

    def test_arithmetic_div(self):
        if not self._is_fp():
            return

        data_a, data_b = self._data(), self._data(reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # load to truncate f64 to precision of f32
        data_div = self.load([a / b for a, b in zip(data_a, data_b)])
        div = self.div(vdata_a, vdata_b)
        assert div == data_div

    def test_arithmetic_intdiv(self):
        """
        Test integer division intrinsics:
            npyv_divisor_##sfx
            npyv_divc_##sfx
        """
        if self._is_fp():
            return

        int_min = self._int_min()

        def trunc_div(a, d):
            """
            Divide towards zero works with large integers > 2^53,
            and wrap around overflow similar to what C does.
            """
            if d == -1 and a == int_min:
                return a
            sign_a, sign_d = a < 0, d < 0
            if a == 0 or sign_a == sign_d:
                return a // d
            return (a + sign_d - sign_a) // d + 1

        data = [1, -int_min]  # to test overflow
        data += range(0, 2**8, 2**5)
        data += range(0, 2**8, 2**5 - 1)
        bsize = self._scalar_size()
        if bsize > 8:
            data += range(2**8, 2**16, 2**13)
            data += range(2**8, 2**16, 2**13 - 1)
        if bsize > 16:
            data += range(2**16, 2**32, 2**29)
            data += range(2**16, 2**32, 2**29 - 1)
        if bsize > 32:
            data += range(2**32, 2**64, 2**61)
            data += range(2**32, 2**64, 2**61 - 1)
        # negate
        data += [-x for x in data]
        for dividend, divisor in itertools.product(data, data):
            divisor = self.setall(divisor)[0]  # cast
            if divisor == 0:
                continue
            dividend = self.load(self._data(dividend))
            data_divc = [trunc_div(a, divisor) for a in dividend]
            divisor_parms = self.divisor(divisor)
            divc = self.divc(dividend, divisor_parms)
            assert divc == data_divc

    def test_arithmetic_reduce_sum(self):
        """
        Test reduce sum intrinsics:
            npyv_sum_##sfx
        """
        if self.sfx not in ("u32", "u64", "f32", "f64"):
            return
        # reduce sum
        data = self._data()
        vdata = self.load(data)

        data_sum = sum(data)
        vsum = self.sum(vdata)
        assert vsum == data_sum

    def test_arithmetic_reduce_sumup(self):
        """
        Test extend reduce sum intrinsics:
            npyv_sumup_##sfx
        """
        if self.sfx not in ("u8", "u16"):
            return
        rdata = (0, self.nlanes, self._int_min(), self._int_max() - self.nlanes)
        for r in rdata:
            data = self._data(r)
            vdata = self.load(data)
            data_sum = sum(data)
            vsum = self.sumup(vdata)
            assert vsum == data_sum

    def test_mask_conditional(self):
        """
        Conditional addition and subtraction for all supported data types.
        Test intrinsics:
            npyv_ifadd_##SFX, npyv_ifsub_##SFX
        """
        vdata_a = self.load(self._data())
        vdata_b = self.load(self._data(reverse=True))
        true_mask = self.cmpeq(self.zero(), self.zero())
        false_mask = self.cmpneq(self.zero(), self.zero())

        data_sub = self.sub(vdata_b, vdata_a)
        ifsub = self.ifsub(true_mask, vdata_b, vdata_a, vdata_b)
        assert ifsub == data_sub
        ifsub = self.ifsub(false_mask, vdata_a, vdata_b, vdata_b)
        assert ifsub == vdata_b

        data_add = self.add(vdata_b, vdata_a)
        ifadd = self.ifadd(true_mask, vdata_b, vdata_a, vdata_b)
        assert ifadd == data_add
        ifadd = self.ifadd(false_mask, vdata_a, vdata_b, vdata_b)
        assert ifadd == vdata_b

        if not self._is_fp():
            return
        data_div = self.div(vdata_b, vdata_a)
        ifdiv = self.ifdiv(true_mask, vdata_b, vdata_a, vdata_b)
        assert ifdiv == data_div
        ifdivz = self.ifdivz(true_mask, vdata_b, vdata_a)
        assert ifdivz == data_div
        ifdiv = self.ifdiv(false_mask, vdata_a, vdata_b, vdata_b)
        assert ifdiv == vdata_b
        ifdivz = self.ifdivz(false_mask, vdata_a, vdata_b)
        assert ifdivz == self.zero()


bool_sfx = ("b8", "b16", "b32", "b64")
int_sfx = ("u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64")
fp_sfx = ("f32", "f64")
all_sfx = int_sfx + fp_sfx
tests_registry = {
    bool_sfx: _SIMD_BOOL,
    int_sfx:  _SIMD_INT,
    fp_sfx:   _SIMD_FP,
    ("f32",): _SIMD_FP32,
    ("f64",): _SIMD_FP64,
    all_sfx:  _SIMD_ALL
}
for target_name, npyv in targets.items():
    simd_width = npyv.simd if npyv else ''
    pretty_name = target_name.split('__')  # multi-target separator
    if len(pretty_name) > 1:
        # multi-target
        pretty_name = f"({' '.join(pretty_name)})"
    else:
        pretty_name = pretty_name[0]

    skip = ""
    skip_sfx = {}
    if not npyv:
        skip = f"target '{pretty_name}' isn't supported by current machine"
    elif not npyv.simd:
        skip = f"target '{pretty_name}' isn't supported by NPYV"
    else:
        if not npyv.simd_f32:
            skip_sfx["f32"] = f"target '{pretty_name}' "\
                               "doesn't support single-precision"
        if not npyv.simd_f64:
            skip_sfx["f64"] = f"target '{pretty_name}' doesn't"\
                               "support double-precision"

    for sfxes, cls in tests_registry.items():
        for sfx in sfxes:
            skip_m = skip_sfx.get(sfx, skip)
            inhr = (cls,)
            attr = {"npyv": targets[target_name], "sfx": sfx, "target_name": target_name}
            tcls = type(f"Test{cls.__name__}_{simd_width}_{target_name}_{sfx}", inhr, attr)
            if skip_m:
                pytest.mark.skip(reason=skip_m)(tcls)
            globals()[tcls.__name__] = tcls

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_assumptions.py ===
from sympy.core.mod import Mod
from sympy.core.numbers import (I, oo, pi)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, sin)
from sympy.simplify.simplify import simplify
from sympy.core import Symbol, S, Rational, Integer, Dummy, Wild, Pow
from sympy.core.assumptions import (assumptions, check_assumptions,
    failing_assumptions, common_assumptions, _generate_assumption_rules,
    _load_pre_generated_assumption_rules)
from sympy.core.facts import InconsistentAssumptions
from sympy.core.random import seed
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

from sympy.testing.pytest import raises, XFAIL


def test_symbol_unset():
    x = Symbol('x', real=True, integer=True)
    assert x.is_real is True
    assert x.is_integer is True
    assert x.is_imaginary is False
    assert x.is_noninteger is False
    assert x.is_number is False


def test_zero():
    z = Integer(0)
    assert z.is_commutative is True
    assert z.is_integer is True
    assert z.is_rational is True
    assert z.is_algebraic is True
    assert z.is_transcendental is False
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is False
    assert z.is_irrational is False
    assert z.is_imaginary is False
    assert z.is_positive is False
    assert z.is_negative is False
    assert z.is_nonpositive is True
    assert z.is_nonnegative is True
    assert z.is_even is True
    assert z.is_odd is False
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_composite is False
    assert z.is_number is True


def test_one():
    z = Integer(1)
    assert z.is_commutative is True
    assert z.is_integer is True
    assert z.is_rational is True
    assert z.is_algebraic is True
    assert z.is_transcendental is False
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is False
    assert z.is_irrational is False
    assert z.is_imaginary is False
    assert z.is_positive is True
    assert z.is_negative is False
    assert z.is_nonpositive is False
    assert z.is_nonnegative is True
    assert z.is_even is False
    assert z.is_odd is True
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_number is True
    assert z.is_composite is False  # issue 8807


def test_negativeone():
    z = Integer(-1)
    assert z.is_commutative is True
    assert z.is_integer is True
    assert z.is_rational is True
    assert z.is_algebraic is True
    assert z.is_transcendental is False
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is False
    assert z.is_irrational is False
    assert z.is_imaginary is False
    assert z.is_positive is False
    assert z.is_negative is True
    assert z.is_nonpositive is True
    assert z.is_nonnegative is False
    assert z.is_even is False
    assert z.is_odd is True
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_composite is False
    assert z.is_number is True


def test_infinity():
    oo = S.Infinity

    assert oo.is_commutative is True
    assert oo.is_integer is False
    assert oo.is_rational is False
    assert oo.is_algebraic is False
    assert oo.is_transcendental is False
    assert oo.is_extended_real is True
    assert oo.is_real is False
    assert oo.is_complex is False
    assert oo.is_noninteger is True
    assert oo.is_irrational is False
    assert oo.is_imaginary is False
    assert oo.is_nonzero is False
    assert oo.is_positive is False
    assert oo.is_negative is False
    assert oo.is_nonpositive is False
    assert oo.is_nonnegative is False
    assert oo.is_extended_nonzero is True
    assert oo.is_extended_positive is True
    assert oo.is_extended_negative is False
    assert oo.is_extended_nonpositive is False
    assert oo.is_extended_nonnegative is True
    assert oo.is_even is False
    assert oo.is_odd is False
    assert oo.is_finite is False
    assert oo.is_infinite is True
    assert oo.is_comparable is True
    assert oo.is_prime is False
    assert oo.is_composite is False
    assert oo.is_number is True


def test_neg_infinity():
    mm = S.NegativeInfinity

    assert mm.is_commutative is True
    assert mm.is_integer is False
    assert mm.is_rational is False
    assert mm.is_algebraic is False
    assert mm.is_transcendental is False
    assert mm.is_extended_real is True
    assert mm.is_real is False
    assert mm.is_complex is False
    assert mm.is_noninteger is True
    assert mm.is_irrational is False
    assert mm.is_imaginary is False
    assert mm.is_nonzero is False
    assert mm.is_positive is False
    assert mm.is_negative is False
    assert mm.is_nonpositive is False
    assert mm.is_nonnegative is False
    assert mm.is_extended_nonzero is True
    assert mm.is_extended_positive is False
    assert mm.is_extended_negative is True
    assert mm.is_extended_nonpositive is True
    assert mm.is_extended_nonnegative is False
    assert mm.is_even is False
    assert mm.is_odd is False
    assert mm.is_finite is False
    assert mm.is_infinite is True
    assert mm.is_comparable is True
    assert mm.is_prime is False
    assert mm.is_composite is False
    assert mm.is_number is True


def test_zoo():
    zoo = S.ComplexInfinity
    assert zoo.is_complex is False
    assert zoo.is_real is False
    assert zoo.is_prime is False


def test_nan():
    nan = S.NaN

    assert nan.is_commutative is True
    assert nan.is_integer is None
    assert nan.is_rational is None
    assert nan.is_algebraic is None
    assert nan.is_transcendental is None
    assert nan.is_real is None
    assert nan.is_complex is None
    assert nan.is_noninteger is None
    assert nan.is_irrational is None
    assert nan.is_imaginary is None
    assert nan.is_positive is None
    assert nan.is_negative is None
    assert nan.is_nonpositive is None
    assert nan.is_nonnegative is None
    assert nan.is_even is None
    assert nan.is_odd is None
    assert nan.is_finite is None
    assert nan.is_infinite is None
    assert nan.is_comparable is False
    assert nan.is_prime is None
    assert nan.is_composite is None
    assert nan.is_number is True


def test_pos_rational():
    r = Rational(3, 4)
    assert r.is_commutative is True
    assert r.is_integer is False
    assert r.is_rational is True
    assert r.is_algebraic is True
    assert r.is_transcendental is False
    assert r.is_real is True
    assert r.is_complex is True
    assert r.is_noninteger is True
    assert r.is_irrational is False
    assert r.is_imaginary is False
    assert r.is_positive is True
    assert r.is_negative is False
    assert r.is_nonpositive is False
    assert r.is_nonnegative is True
    assert r.is_even is False
    assert r.is_odd is False
    assert r.is_finite is True
    assert r.is_infinite is False
    assert r.is_comparable is True
    assert r.is_prime is False
    assert r.is_composite is False

    r = Rational(1, 4)
    assert r.is_nonpositive is False
    assert r.is_positive is True
    assert r.is_negative is False
    assert r.is_nonnegative is True
    r = Rational(5, 4)
    assert r.is_negative is False
    assert r.is_positive is True
    assert r.is_nonpositive is False
    assert r.is_nonnegative is True
    r = Rational(5, 3)
    assert r.is_nonnegative is True
    assert r.is_positive is True
    assert r.is_negative is False
    assert r.is_nonpositive is False


def test_neg_rational():
    r = Rational(-3, 4)
    assert r.is_positive is False
    assert r.is_nonpositive is True
    assert r.is_negative is True
    assert r.is_nonnegative is False
    r = Rational(-1, 4)
    assert r.is_nonpositive is True
    assert r.is_positive is False
    assert r.is_negative is True
    assert r.is_nonnegative is False
    r = Rational(-5, 4)
    assert r.is_negative is True
    assert r.is_positive is False
    assert r.is_nonpositive is True
    assert r.is_nonnegative is False
    r = Rational(-5, 3)
    assert r.is_nonnegative is False
    assert r.is_positive is False
    assert r.is_negative is True
    assert r.is_nonpositive is True


def test_pi():
    z = S.Pi
    assert z.is_commutative is True
    assert z.is_integer is False
    assert z.is_rational is False
    assert z.is_algebraic is False
    assert z.is_transcendental is True
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is True
    assert z.is_irrational is True
    assert z.is_imaginary is False
    assert z.is_positive is True
    assert z.is_negative is False
    assert z.is_nonpositive is False
    assert z.is_nonnegative is True
    assert z.is_even is False
    assert z.is_odd is False
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_composite is False


def test_E():
    z = S.Exp1
    assert z.is_commutative is True
    assert z.is_integer is False
    assert z.is_rational is False
    assert z.is_algebraic is False
    assert z.is_transcendental is True
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is True
    assert z.is_irrational is True
    assert z.is_imaginary is False
    assert z.is_positive is True
    assert z.is_negative is False
    assert z.is_nonpositive is False
    assert z.is_nonnegative is True
    assert z.is_even is False
    assert z.is_odd is False
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_composite is False


def test_I():
    z = S.ImaginaryUnit
    assert z.is_commutative is True
    assert z.is_integer is False
    assert z.is_rational is False
    assert z.is_algebraic is True
    assert z.is_transcendental is False
    assert z.is_real is False
    assert z.is_complex is True
    assert z.is_noninteger is False
    assert z.is_irrational is False
    assert z.is_imaginary is True
    assert z.is_positive is False
    assert z.is_negative is False
    assert z.is_nonpositive is False
    assert z.is_nonnegative is False
    assert z.is_even is False
    assert z.is_odd is False
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is False
    assert z.is_prime is False
    assert z.is_composite is False


def test_symbol_real_false():
    # issue 3848
    a = Symbol('a', real=False)

    assert a.is_real is False
    assert a.is_integer is False
    assert a.is_zero is False

    assert a.is_negative is False
    assert a.is_positive is False
    assert a.is_nonnegative is False
    assert a.is_nonpositive is False
    assert a.is_nonzero is False

    assert a.is_extended_negative is None
    assert a.is_extended_positive is None
    assert a.is_extended_nonnegative is None
    assert a.is_extended_nonpositive is None
    assert a.is_extended_nonzero is None


def test_symbol_extended_real_false():
    # issue 3848
    a = Symbol('a', extended_real=False)

    assert a.is_real is False
    assert a.is_integer is False
    assert a.is_zero is False

    assert a.is_negative is False
    assert a.is_positive is False
    assert a.is_nonnegative is False
    assert a.is_nonpositive is False
    assert a.is_nonzero is False

    assert a.is_extended_negative is False
    assert a.is_extended_positive is False
    assert a.is_extended_nonnegative is False
    assert a.is_extended_nonpositive is False
    assert a.is_extended_nonzero is False


def test_symbol_imaginary():
    a = Symbol('a', imaginary=True)

    assert a.is_real is False
    assert a.is_integer is False
    assert a.is_negative is False
    assert a.is_positive is False
    assert a.is_nonnegative is False
    assert a.is_nonpositive is False
    assert a.is_zero is False
    assert a.is_nonzero is False  # since nonzero -> real


def test_symbol_zero():
    x = Symbol('x', zero=True)
    assert x.is_positive is False
    assert x.is_nonpositive
    assert x.is_negative is False
    assert x.is_nonnegative
    assert x.is_zero is True
    # TODO Change to x.is_nonzero is None
    # See https://github.com/sympy/sympy/pull/9583
    assert x.is_nonzero is False
    assert x.is_finite is True


def test_symbol_positive():
    x = Symbol('x', positive=True)
    assert x.is_positive is True
    assert x.is_nonpositive is False
    assert x.is_negative is False
    assert x.is_nonnegative is True
    assert x.is_zero is False
    assert x.is_nonzero is True


def test_neg_symbol_positive():
    x = -Symbol('x', positive=True)
    assert x.is_positive is False
    assert x.is_nonpositive is True
    assert x.is_negative is True
    assert x.is_nonnegative is False
    assert x.is_zero is False
    assert x.is_nonzero is True


def test_symbol_nonpositive():
    x = Symbol('x', nonpositive=True)
    assert x.is_positive is False
    assert x.is_nonpositive is True
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None


def test_neg_symbol_nonpositive():
    x = -Symbol('x', nonpositive=True)
    assert x.is_positive is None
    assert x.is_nonpositive is None
    assert x.is_negative is False
    assert x.is_nonnegative is True
    assert x.is_zero is None
    assert x.is_nonzero is None


def test_symbol_falsepositive():
    x = Symbol('x', positive=False)
    assert x.is_positive is False
    assert x.is_nonpositive is None
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None


def test_symbol_falsepositive_mul():
    # To test pull request 9379
    # Explicit handling of arg.is_positive=False was added to Mul._eval_is_positive
    x = 2*Symbol('x', positive=False)
    assert x.is_positive is False  # This was None before
    assert x.is_nonpositive is None
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None


@XFAIL
def test_symbol_infinitereal_mul():
    ix = Symbol('ix', infinite=True, extended_real=True)
    assert (-ix).is_extended_positive is None


def test_neg_symbol_falsepositive():
    x = -Symbol('x', positive=False)
    assert x.is_positive is None
    assert x.is_nonpositive is None
    assert x.is_negative is False
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None


def test_neg_symbol_falsenegative():
    # To test pull request 9379
    # Explicit handling of arg.is_negative=False was added to Mul._eval_is_positive
    x = -Symbol('x', negative=False)
    assert x.is_positive is False  # This was None before
    assert x.is_nonpositive is None
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None


def test_symbol_falsepositive_real():
    x = Symbol('x', positive=False, real=True)
    assert x.is_positive is False
    assert x.is_nonpositive is True
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None


def test_neg_symbol_falsepositive_real():
    x = -Symbol('x', positive=False, real=True)
    assert x.is_positive is None
    assert x.is_nonpositive is None
    assert x.is_negative is False
    assert x.is_nonnegative is True
    assert x.is_zero is None
    assert x.is_nonzero is None


def test_symbol_falsenonnegative():
    x = Symbol('x', nonnegative=False)
    assert x.is_positive is False
    assert x.is_nonpositive is None
    assert x.is_negative is None
    assert x.is_nonnegative is False
    assert x.is_zero is False
    assert x.is_nonzero is None


@XFAIL
def test_neg_symbol_falsenonnegative():
    x = -Symbol('x', nonnegative=False)
    assert x.is_positive is None
    assert x.is_nonpositive is False  # this currently returns None
    assert x.is_negative is False  # this currently returns None
    assert x.is_nonnegative is None
    assert x.is_zero is False  # this currently returns None
    assert x.is_nonzero is True  # this currently returns None


def test_symbol_falsenonnegative_real():
    x = Symbol('x', nonnegative=False, real=True)
    assert x.is_positive is False
    assert x.is_nonpositive is True
    assert x.is_negative is True
    assert x.is_nonnegative is False
    assert x.is_zero is False
    assert x.is_nonzero is True


def test_neg_symbol_falsenonnegative_real():
    x = -Symbol('x', nonnegative=False, real=True)
    assert x.is_positive is True
    assert x.is_nonpositive is False
    assert x.is_negative is False
    assert x.is_nonnegative is True
    assert x.is_zero is False
    assert x.is_nonzero is True


def test_prime():
    assert S.NegativeOne.is_prime is False
    assert S(-2).is_prime is False
    assert S(-4).is_prime is False
    assert S.Zero.is_prime is False
    assert S.One.is_prime is False
    assert S(2).is_prime is True
    assert S(17).is_prime is True
    assert S(4).is_prime is False


def test_composite():
    assert S.NegativeOne.is_composite is False
    assert S(-2).is_composite is False
    assert S(-4).is_composite is False
    assert S.Zero.is_composite is False
    assert S(2).is_composite is False
    assert S(17).is_composite is False
    assert S(4).is_composite is True
    x = Dummy(integer=True, positive=True, prime=False)
    assert x.is_composite is None # x could be 1
    assert (x + 1).is_composite is None
    x = Dummy(positive=True, even=True, prime=False)
    assert x.is_integer is True
    assert x.is_composite is True


def test_prime_symbol():
    x = Symbol('x', prime=True)
    assert x.is_prime is True
    assert x.is_integer is True
    assert x.is_positive is True
    assert x.is_negative is False
    assert x.is_nonpositive is False
    assert x.is_nonnegative is True

    x = Symbol('x', prime=False)
    assert x.is_prime is False
    assert x.is_integer is None
    assert x.is_positive is None
    assert x.is_negative is None
    assert x.is_nonpositive is None
    assert x.is_nonnegative is None


def test_symbol_noncommutative():
    x = Symbol('x', commutative=True)
    assert x.is_complex is None

    x = Symbol('x', commutative=False)
    assert x.is_integer is False
    assert x.is_rational is False
    assert x.is_algebraic is False
    assert x.is_irrational is False
    assert x.is_real is False
    assert x.is_complex is False


def test_other_symbol():
    x = Symbol('x', integer=True)
    assert x.is_integer is True
    assert x.is_real is True
    assert x.is_finite is True

    x = Symbol('x', integer=True, nonnegative=True)
    assert x.is_integer is True
    assert x.is_nonnegative is True
    assert x.is_negative is False
    assert x.is_positive is None
    assert x.is_finite is True

    x = Symbol('x', integer=True, nonpositive=True)
    assert x.is_integer is True
    assert x.is_nonpositive is True
    assert x.is_positive is False
    assert x.is_negative is None
    assert x.is_finite is True

    x = Symbol('x', odd=True)
    assert x.is_odd is True
    assert x.is_even is False
    assert x.is_integer is True
    assert x.is_finite is True

    x = Symbol('x', odd=False)
    assert x.is_odd is False
    assert x.is_even is None
    assert x.is_integer is None
    assert x.is_finite is None

    x = Symbol('x', even=True)
    assert x.is_even is True
    assert x.is_odd is False
    assert x.is_integer is True
    assert x.is_finite is True

    x = Symbol('x', even=False)
    assert x.is_even is False
    assert x.is_odd is None
    assert x.is_integer is None
    assert x.is_finite is None

    x = Symbol('x', integer=True, nonnegative=True)
    assert x.is_integer is True
    assert x.is_nonnegative is True
    assert x.is_finite is True

    x = Symbol('x', integer=True, nonpositive=True)
    assert x.is_integer is True
    assert x.is_nonpositive is True
    assert x.is_finite is True

    x = Symbol('x', rational=True)
    assert x.is_real is True
    assert x.is_finite is True

    x = Symbol('x', rational=False)
    assert x.is_real is None
    assert x.is_finite is None

    x = Symbol('x', irrational=True)
    assert x.is_real is True
    assert x.is_finite is True

    x = Symbol('x', irrational=False)
    assert x.is_real is None
    assert x.is_finite is None

    with raises(AttributeError):
        x.is_real = False

    x = Symbol('x', algebraic=True)
    assert x.is_transcendental is False
    x = Symbol('x', transcendental=True)
    assert x.is_algebraic is False
    assert x.is_rational is False
    assert x.is_integer is False


def test_evaluate_false():
    # Previously this failed because the assumptions query would make new
    # expressions and some of the evaluation logic would fail under
    # evaluate(False).
    from sympy.core.parameters import evaluate
    from sympy.abc import x, h
    f = 2**x**7
    with evaluate(False):
        fh = f.xreplace({x: x+h})
        assert fh.exp.is_rational is None


def test_issue_3825():
    """catch: hash instability"""
    x = Symbol("x")
    y = Symbol("y")
    a1 = x + y
    a2 = y + x
    a2.is_comparable

    h1 = hash(a1)
    h2 = hash(a2)
    assert h1 == h2


def test_issue_4822():
    z = (-1)**Rational(1, 3)*(1 - I*sqrt(3))
    assert z.is_real in [True, None]


def test_hash_vs_typeinfo():
    """seemingly different typeinfo, but in fact equal"""

    # the following two are semantically equal
    x1 = Symbol('x', even=True)
    x2 = Symbol('x', integer=True, odd=False)

    assert hash(x1) == hash(x2)
    assert x1 == x2


def test_hash_vs_typeinfo_2():
    """different typeinfo should mean !eq"""
    # the following two are semantically different
    x = Symbol('x')
    x1 = Symbol('x', even=True)

    assert x != x1
    assert hash(x) != hash(x1)  # This might fail with very low probability


def test_hash_vs_eq():
    """catch: different hash for equal objects"""
    a = 1 + S.Pi    # important: do not fold it into a Number instance
    ha = hash(a)  # it should be Add/Mul/... to trigger the bug

    a.is_positive   # this uses .evalf() and deduces it is positive
    assert a.is_positive is True

    # be sure that hash stayed the same
    assert ha == hash(a)

    # now b should be the same expression
    b = a.expand(trig=True)
    hb = hash(b)

    assert a == b
    assert ha == hb


def test_Add_is_pos_neg():
    # these cover lines not covered by the rest of tests in core
    n = Symbol('n', extended_negative=True, infinite=True)
    nn = Symbol('n', extended_nonnegative=True, infinite=True)
    np = Symbol('n', extended_nonpositive=True, infinite=True)
    p = Symbol('p', extended_positive=True, infinite=True)
    r = Dummy(extended_real=True, finite=False)
    x = Symbol('x')
    xf = Symbol('xf', finite=True)
    assert (n + p).is_extended_positive is None
    assert (n + x).is_extended_positive is None
    assert (p + x).is_extended_positive is None
    assert (n + p).is_extended_negative is None
    assert (n + x).is_extended_negative is None
    assert (p + x).is_extended_negative is None

    assert (n + xf).is_extended_positive is False
    assert (p + xf).is_extended_positive is True
    assert (n + xf).is_extended_negative is True
    assert (p + xf).is_extended_negative is False

    assert (x - S.Infinity).is_extended_negative is None  # issue 7798
    # issue 8046, 16.2
    assert (p + nn).is_extended_positive
    assert (n + np).is_extended_negative
    assert (p + r).is_extended_positive is None


def test_Add_is_imaginary():
    nn = Dummy(nonnegative=True)
    assert (I*nn + I).is_imaginary  # issue 8046, 17


def test_Add_is_algebraic():
    a = Symbol('a', algebraic=True)
    b = Symbol('a', algebraic=True)
    na = Symbol('na', algebraic=False)
    nb = Symbol('nb', algebraic=False)
    x = Symbol('x')
    assert (a + b).is_algebraic
    assert (na + nb).is_algebraic is None
    assert (a + na).is_algebraic is False
    assert (a + x).is_algebraic is None
    assert (na + x).is_algebraic is None


def test_Mul_is_algebraic():
    a = Symbol('a', algebraic=True)
    b = Symbol('b', algebraic=True)
    na = Symbol('na', algebraic=False)
    an = Symbol('an', algebraic=True, nonzero=True)
    nb = Symbol('nb', algebraic=False)
    x = Symbol('x')
    assert (a*b).is_algebraic is True
    assert (na*nb).is_algebraic is None
    assert (a*na).is_algebraic is None
    assert (an*na).is_algebraic is False
    assert (a*x).is_algebraic is None
    assert (na*x).is_algebraic is None


def test_Pow_is_algebraic():
    e = Symbol('e', algebraic=True)

    assert Pow(1, e, evaluate=False).is_algebraic
    assert Pow(0, e, evaluate=False).is_algebraic

    a = Symbol('a', algebraic=True)
    azf = Symbol('azf', algebraic=True, zero=False)
    na = Symbol('na', algebraic=False)
    ia = Symbol('ia', algebraic=True, irrational=True)
    ib = Symbol('ib', algebraic=True, irrational=True)
    r = Symbol('r', rational=True)
    x = Symbol('x')
    assert (a**2).is_algebraic is True
    assert (a**r).is_algebraic is None
    assert (azf**r).is_algebraic is True
    assert (a**x).is_algebraic is None
    assert (na**r).is_algebraic is None
    assert (ia**r).is_algebraic is True
    assert (ia**ib).is_algebraic is False

    assert (a**e).is_algebraic is None

    # Gelfond-Schneider constant:
    assert Pow(2, sqrt(2), evaluate=False).is_algebraic is False

    assert Pow(S.GoldenRatio, sqrt(3), evaluate=False).is_algebraic is False

    # issue 8649
    t = Symbol('t', real=True, transcendental=True)
    n = Symbol('n', integer=True)
    assert (t**n).is_algebraic is None
    assert (t**n).is_integer is None

    assert (pi**3).is_algebraic is False
    r = Symbol('r', zero=True)
    assert (pi**r).is_algebraic is True


def test_Mul_is_prime_composite():
    x = Symbol('x', positive=True, integer=True)
    y = Symbol('y', positive=True, integer=True)
    assert (x*y).is_prime is None
    assert ( (x+1)*(y+1) ).is_prime is False
    assert ( (x+1)*(y+1) ).is_composite is True

    x = Symbol('x', positive=True)
    assert ( (x+1)*(y+1) ).is_prime is None
    assert ( (x+1)*(y+1) ).is_composite is None


def test_Pow_is_pos_neg():
    z = Symbol('z', real=True)
    w = Symbol('w', nonpositive=True)

    assert (S.NegativeOne**S(2)).is_positive is True
    assert (S.One**z).is_positive is True
    assert (S.NegativeOne**S(3)).is_positive is False
    assert (S.Zero**S.Zero).is_positive is True  # 0**0 is 1
    assert (w**S(3)).is_positive is False
    assert (w**S(2)).is_positive is None
    assert (I**2).is_positive is False
    assert (I**4).is_positive is True

    # tests emerging from #16332 issue
    p = Symbol('p', zero=True)
    q = Symbol('q', zero=False, real=True)
    j = Symbol('j', zero=False, even=True)
    x = Symbol('x', zero=True)
    y = Symbol('y', zero=True)
    assert (p**q).is_positive is False
    assert (p**q).is_negative is False
    assert (p**j).is_positive is False
    assert (x**y).is_positive is True   # 0**0
    assert (x**y).is_negative is False


def test_Pow_is_prime_composite():
    x = Symbol('x', positive=True, integer=True)
    y = Symbol('y', positive=True, integer=True)
    assert (x**y).is_prime is None
    assert ( x**(y+1) ).is_prime is False
    assert ( x**(y+1) ).is_composite is None
    assert ( (x+1)**(y+1) ).is_composite is True
    assert ( (-x-1)**(2*y) ).is_composite is True

    x = Symbol('x', positive=True)
    assert (x**y).is_prime is None


def test_Mul_is_infinite():
    x = Symbol('x')
    f = Symbol('f', finite=True)
    i = Symbol('i', infinite=True)
    z = Dummy(zero=True)
    nzf = Dummy(finite=True, zero=False)
    from sympy.core.mul import Mul
    assert (x*f).is_finite is None
    assert (x*i).is_finite is None
    assert (f*i).is_finite is None
    assert (x*f*i).is_finite is None
    assert (z*i).is_finite is None
    assert (nzf*i).is_finite is False
    assert (z*f).is_finite is True
    assert Mul(0, f, evaluate=False).is_finite is True
    assert Mul(0, i, evaluate=False).is_finite is None

    assert (x*f).is_infinite is None
    assert (x*i).is_infinite is None
    assert (f*i).is_infinite is None
    assert (x*f*i).is_infinite is None
    assert (z*i).is_infinite is S.NaN.is_infinite
    assert (nzf*i).is_infinite is True
    assert (z*f).is_infinite is False
    assert Mul(0, f, evaluate=False).is_infinite is False
    assert Mul(0, i, evaluate=False).is_infinite is S.NaN.is_infinite


def test_Add_is_infinite():
    x = Symbol('x')
    f = Symbol('f', finite=True)
    i = Symbol('i', infinite=True)
    i2 = Symbol('i2', infinite=True)
    z = Dummy(zero=True)
    nzf = Dummy(finite=True, zero=False)
    from sympy.core.add import Add
    assert (x+f).is_finite is None
    assert (x+i).is_finite is None
    assert (f+i).is_finite is False
    assert (x+f+i).is_finite is None
    assert (z+i).is_finite is False
    assert (nzf+i).is_finite is False
    assert (z+f).is_finite is True
    assert (i+i2).is_finite is None
    assert Add(0, f, evaluate=False).is_finite is True
    assert Add(0, i, evaluate=False).is_finite is False

    assert (x+f).is_infinite is None
    assert (x+i).is_infinite is None
    assert (f+i).is_infinite is True
    assert (x+f+i).is_infinite is None
    assert (z+i).is_infinite is True
    assert (nzf+i).is_infinite is True
    assert (z+f).is_infinite is False
    assert (i+i2).is_infinite is None
    assert Add(0, f, evaluate=False).is_infinite is False
    assert Add(0, i, evaluate=False).is_infinite is True


def test_special_is_rational():
    i = Symbol('i', integer=True)
    i2 = Symbol('i2', integer=True)
    ni = Symbol('ni', integer=True, nonzero=True)
    r = Symbol('r', rational=True)
    rn = Symbol('r', rational=True, nonzero=True)
    nr = Symbol('nr', irrational=True)
    x = Symbol('x')
    assert sqrt(3).is_rational is False
    assert (3 + sqrt(3)).is_rational is False
    assert (3*sqrt(3)).is_rational is False
    assert exp(3).is_rational is False
    assert exp(ni).is_rational is False
    assert exp(rn).is_rational is False
    assert exp(x).is_rational is None
    assert exp(log(3), evaluate=False).is_rational is True
    assert log(exp(3), evaluate=False).is_rational is True
    assert log(3).is_rational is False
    assert log(ni + 1).is_rational is False
    assert log(rn + 1).is_rational is False
    assert log(x).is_rational is None
    assert (sqrt(3) + sqrt(5)).is_rational is None
    assert (sqrt(3) + S.Pi).is_rational is False
    assert (x**i).is_rational is None
    assert (i**i).is_rational is True
    assert (i**i2).is_rational is None
    assert (r**i).is_rational is None
    assert (r**r).is_rational is None
    assert (r**x).is_rational is None
    assert (nr**i).is_rational is None  # issue 8598
    assert (nr**Symbol('z', zero=True)).is_rational
    assert sin(1).is_rational is False
    assert sin(ni).is_rational is False
    assert sin(rn).is_rational is False
    assert sin(x).is_rational is None
    assert asin(r).is_rational is False
    assert sin(asin(3), evaluate=False).is_rational is True


@XFAIL
def test_issue_6275():
    x = Symbol('x')
    # both zero or both Muls...but neither "change would be very appreciated.
    # This is similar to x/x => 1 even though if x = 0, it is really nan.
    assert isinstance(x*0, type(0*S.Infinity))
    if 0*S.Infinity is S.NaN:
        b = Symbol('b', finite=None)
        assert (b*0).is_zero is None


def test_sanitize_assumptions():
    # issue 6666
    for cls in (Symbol, Dummy, Wild):
        x = cls('x', real=1, positive=0)
        assert x.is_real is True
        assert x.is_positive is False
        assert cls('', real=True, positive=None).is_positive is None
        raises(ValueError, lambda: cls('', commutative=None))
    raises(ValueError, lambda: Symbol._sanitize({"commutative": None}))


def test_special_assumptions():
    e = -3 - sqrt(5) + (-sqrt(10)/2 - sqrt(2)/2)**2
    assert simplify(e < 0) is S.false
    assert simplify(e > 0) is S.false
    assert (e == 0) is False  # it's not a literal 0
    assert e.equals(0) is True


def test_inconsistent():
    # cf. issues 5795 and 5545
    raises(InconsistentAssumptions, lambda: Symbol('x', real=True,
           commutative=False))


def test_issue_6631():
    assert ((-1)**(I)).is_real is True
    assert ((-1)**(I*2)).is_real is True
    assert ((-1)**(I/2)).is_real is True
    assert ((-1)**(I*S.Pi)).is_real is True
    assert (I**(I + 2)).is_real is True


def test_issue_2730():
    assert (1/(1 + I)).is_real is False


def test_issue_4149():
    assert (3 + I).is_complex
    assert (3 + I).is_imaginary is False
    assert (3*I + S.Pi*I).is_imaginary
    # as Zero.is_imaginary is False, see issue 7649
    y = Symbol('y', real=True)
    assert (3*I + S.Pi*I + y*I).is_imaginary is None
    p = Symbol('p', positive=True)
    assert (3*I + S.Pi*I + p*I).is_imaginary
    n = Symbol('n', negative=True)
    assert (-3*I - S.Pi*I + n*I).is_imaginary

    i = Symbol('i', imaginary=True)
    assert ([(i**a).is_imaginary for a in range(4)] ==
            [False, True, False, True])

    # tests from the PR #7887:
    e = S("-sqrt(3)*I/2 + 0.866025403784439*I")
    assert e.is_real is False
    assert e.is_imaginary


def test_issue_2920():
    n = Symbol('n', negative=True)
    assert sqrt(n).is_imaginary


def test_issue_7899():
    x = Symbol('x', real=True)
    assert (I*x).is_real is None
    assert ((x - I)*(x - 1)).is_zero is None
    assert ((x - I)*(x - 1)).is_real is None


@XFAIL
def test_issue_7993():
    x = Dummy(integer=True)
    y = Dummy(noninteger=True)
    assert (x - y).is_zero is False


def test_issue_8075():
    raises(InconsistentAssumptions, lambda: Dummy(zero=True, finite=False))
    raises(InconsistentAssumptions, lambda: Dummy(zero=True, infinite=True))


def test_issue_8642():
    x = Symbol('x', real=True, integer=False)
    assert (x*2).is_integer is None, (x*2).is_integer


def test_issues_8632_8633_8638_8675_8992():
    p = Dummy(integer=True, positive=True)
    nn = Dummy(integer=True, nonnegative=True)
    assert (p - S.Half).is_positive
    assert (p - 1).is_nonnegative
    assert (nn + 1).is_positive
    assert (-p + 1).is_nonpositive
    assert (-nn - 1).is_negative
    prime = Dummy(prime=True)
    assert (prime - 2).is_nonnegative
    assert (prime - 3).is_nonnegative is None
    even = Dummy(positive=True, even=True)
    assert (even - 2).is_nonnegative

    p = Dummy(positive=True)
    assert (p/(p + 1) - 1).is_negative
    assert ((p + 2)**3 - S.Half).is_positive
    n = Dummy(negative=True)
    assert (n - 3).is_nonpositive


def test_issue_9115_9150():
    n = Dummy('n', integer=True, nonnegative=True)
    assert (factorial(n) >= 1) == True
    assert (factorial(n) < 1) == False

    assert factorial(n + 1).is_even is None
    assert factorial(n + 2).is_even is True
    assert factorial(n + 2) >= 2


def test_issue_9165():
    z = Symbol('z', zero=True)
    f = Symbol('f', finite=False)
    assert 0/z is S.NaN
    assert 0*(1/z) is S.NaN
    assert 0*f is S.NaN


def test_issue_10024():
    x = Dummy('x')
    assert Mod(x, 2*pi).is_zero is None


def test_issue_10302():
    x = Symbol('x')
    r = Symbol('r', real=True)
    u = -(3*2**pi)**(1/pi) + 2*3**(1/pi)
    i = u + u*I

    assert i.is_real is None  # w/o simplification this should fail
    assert (u + i).is_zero is None
    assert (1 + i).is_zero is False

    a = Dummy('a', zero=True)
    assert (a + I).is_zero is False
    assert (a + r*I).is_zero is None
    assert (a + I).is_imaginary
    assert (a + x + I).is_imaginary is None
    assert (a + r*I + I).is_imaginary is None


def test_complex_reciprocal_imaginary():
    assert (1 / (4 + 3*I)).is_imaginary is False


def test_issue_16313():
    x = Symbol('x', extended_real=False)
    k = Symbol('k', real=True)
    l = Symbol('l', real=True, zero=False)
    assert (-x).is_real is False
    assert (k*x).is_real is None  # k can be zero also
    assert (l*x).is_real is False
    assert (l*x*x).is_real is None  # since x*x can be a real number
    assert (-x).is_positive is False


def test_issue_16579():
    # extended_real -> finite | infinite
    x = Symbol('x', extended_real=True, infinite=False)
    y = Symbol('y', extended_real=True, finite=False)
    assert x.is_finite is True
    assert y.is_infinite is True

    # With PR 16978, complex now implies finite
    c = Symbol('c', complex=True)
    assert c.is_finite is True
    raises(InconsistentAssumptions, lambda: Dummy(complex=True, finite=False))

    # Now infinite == !finite
    nf = Symbol('nf', finite=False)
    assert nf.is_infinite is True


def test_issue_17556():
    z = I*oo
    assert z.is_imaginary is False
    assert z.is_finite is False


def test_issue_21651():
    k = Symbol('k', positive=True, integer=True)
    exp = 2*2**(-k)
    assert exp.is_integer is None


def test_assumptions_copy():
    assert assumptions(Symbol('x'), {"commutative": True}
        ) == {'commutative': True}
    assert assumptions(Symbol('x'), ['integer']) == {}
    assert assumptions(Symbol('x'), ['commutative']
        ) == {'commutative': True}
    assert assumptions(Symbol('x')) == {'commutative': True}
    assert assumptions(1)['positive']
    assert assumptions(3 + I) == {
        'algebraic': True,
        'commutative': True,
        'complex': True,
        'composite': False,
        'even': False,
        'extended_negative': False,
        'extended_nonnegative': False,
        'extended_nonpositive': False,
        'extended_nonzero': False,
        'extended_positive': False,
        'extended_real': False,
        'finite': True,
        'imaginary': False,
        'infinite': False,
        'integer': False,
        'irrational': False,
        'negative': False,
        'noninteger': False,
        'nonnegative': False,
        'nonpositive': False,
        'nonzero': False,
        'odd': False,
        'positive': False,
        'prime': False,
        'rational': False,
        'real': False,
        'transcendental': False,
        'zero': False}


def test_check_assumptions():
    assert check_assumptions(1, 0) is False
    x = Symbol('x', positive=True)
    assert check_assumptions(1, x) is True
    assert check_assumptions(1, 1) is True
    assert check_assumptions(-1, 1) is False
    i = Symbol('i', integer=True)
    # don't know if i is positive (or prime, etc...)
    assert check_assumptions(i, 1) is None
    assert check_assumptions(Dummy(integer=None), integer=True) is None
    assert check_assumptions(Dummy(integer=None), integer=False) is None
    assert check_assumptions(Dummy(integer=False), integer=True) is False
    assert check_assumptions(Dummy(integer=True), integer=False) is False
    # no T/F assumptions to check
    assert check_assumptions(Dummy(integer=False), integer=None) is True
    raises(ValueError, lambda: check_assumptions(2*x, x, positive=True))


def test_failing_assumptions():
    x = Symbol('x', positive=True)
    y = Symbol('y')
    assert failing_assumptions(6*x + y, **x.assumptions0) == \
    {'real': None, 'imaginary': None, 'complex': None, 'hermitian': None,
    'positive': None, 'nonpositive': None, 'nonnegative': None, 'nonzero': None,
    'negative': None, 'zero': None, 'extended_real': None, 'finite': None,
    'infinite': None, 'extended_negative': None, 'extended_nonnegative': None,
    'extended_nonpositive': None, 'extended_nonzero': None,
    'extended_positive': None }


def test_common_assumptions():
    assert common_assumptions([0, 1, 2]
        ) == {'algebraic': True, 'irrational': False, 'hermitian':
        True, 'extended_real': True, 'real': True, 'extended_negative':
        False, 'extended_nonnegative': True, 'integer': True,
        'rational': True, 'imaginary': False, 'complex': True,
        'commutative': True,'noninteger': False, 'composite': False,
        'infinite': False, 'nonnegative': True, 'finite': True,
        'transcendental': False,'negative': False}
    assert common_assumptions([0, 1, 2], 'positive integer'.split()
        ) == {'integer': True}
    assert common_assumptions([0, 1, 2], []) == {}
    assert common_assumptions([], ['integer']) == {}
    assert common_assumptions([0], ['integer']) == {'integer': True}

def test_pre_generated_assumption_rules_are_valid():
    # check the pre-generated assumptions match freshly generated assumptions
    # if this check fails, consider updating the assumptions
    # see sympy.core.assumptions._generate_assumption_rules
    pre_generated_assumptions =_load_pre_generated_assumption_rules()
    generated_assumptions =_generate_assumption_rules()
    assert pre_generated_assumptions._to_python() == generated_assumptions._to_python(), "pre-generated assumptions are invalid, see sympy.core.assumptions._generate_assumption_rules"


def test_ask_shuffle():
    grp = PermutationGroup(Permutation(1, 0, 2), Permutation(2, 1, 3))

    seed(123)
    first = grp.random()
    seed(123)
    simplify(I)
    second = grp.random()
    seed(123)
    simplify(-I)
    third = grp.random()

    assert first == second == third

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_arrayprint.py ===
import gc
import sys
import textwrap

import pytest
from hypothesis import given
from hypothesis.extra import numpy as hynp

import numpy as np
from numpy._core.arrayprint import _typelessdata
from numpy.testing import (
    HAS_REFCOUNT,
    IS_WASM,
    assert_,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
)
from numpy.testing._private.utils import run_threaded


class TestArrayRepr:
    def test_nan_inf(self):
        x = np.array([np.nan, np.inf])
        assert_equal(repr(x), 'array([nan, inf])')

    def test_subclass(self):
        class sub(np.ndarray):
            pass

        # one dimensional
        x1d = np.array([1, 2]).view(sub)
        assert_equal(repr(x1d), 'sub([1, 2])')

        # two dimensional
        x2d = np.array([[1, 2], [3, 4]]).view(sub)
        assert_equal(repr(x2d),
            'sub([[1, 2],\n'
            '     [3, 4]])')

        # two dimensional with flexible dtype
        xstruct = np.ones((2, 2), dtype=[('a', '<i4')]).view(sub)
        assert_equal(repr(xstruct),
            "sub([[(1,), (1,)],\n"
            "     [(1,), (1,)]], dtype=[('a', '<i4')])"
        )

    @pytest.mark.xfail(reason="See gh-10544")
    def test_object_subclass(self):
        class sub(np.ndarray):
            def __new__(cls, inp):
                obj = np.asarray(inp).view(cls)
                return obj

            def __getitem__(self, ind):
                ret = super().__getitem__(ind)
                return sub(ret)

        # test that object + subclass is OK:
        x = sub([None, None])
        assert_equal(repr(x), 'sub([None, None], dtype=object)')
        assert_equal(str(x), '[None None]')

        x = sub([None, sub([None, None])])
        assert_equal(repr(x),
            'sub([None, sub([None, None], dtype=object)], dtype=object)')
        assert_equal(str(x), '[None sub([None, None], dtype=object)]')

    def test_0d_object_subclass(self):
        # make sure that subclasses which return 0ds instead
        # of scalars don't cause infinite recursion in str
        class sub(np.ndarray):
            def __new__(cls, inp):
                obj = np.asarray(inp).view(cls)
                return obj

            def __getitem__(self, ind):
                ret = super().__getitem__(ind)
                return sub(ret)

        x = sub(1)
        assert_equal(repr(x), 'sub(1)')
        assert_equal(str(x), '1')

        x = sub([1, 1])
        assert_equal(repr(x), 'sub([1, 1])')
        assert_equal(str(x), '[1 1]')

        # check it works properly with object arrays too
        x = sub(None)
        assert_equal(repr(x), 'sub(None, dtype=object)')
        assert_equal(str(x), 'None')

        # plus recursive object arrays (even depth > 1)
        y = sub(None)
        x[()] = y
        y[()] = x
        assert_equal(repr(x),
            'sub(sub(sub(..., dtype=object), dtype=object), dtype=object)')
        assert_equal(str(x), '...')
        x[()] = 0  # resolve circular references for garbage collector

        # nested 0d-subclass-object
        x = sub(None)
        x[()] = sub(None)
        assert_equal(repr(x), 'sub(sub(None, dtype=object), dtype=object)')
        assert_equal(str(x), 'None')

        # gh-10663
        class DuckCounter(np.ndarray):
            def __getitem__(self, item):
                result = super().__getitem__(item)
                if not isinstance(result, DuckCounter):
                    result = result[...].view(DuckCounter)
                return result

            def to_string(self):
                return {0: 'zero', 1: 'one', 2: 'two'}.get(self.item(), 'many')

            def __str__(self):
                if self.shape == ():
                    return self.to_string()
                else:
                    fmt = {'all': lambda x: x.to_string()}
                    return np.array2string(self, formatter=fmt)

        dc = np.arange(5).view(DuckCounter)
        assert_equal(str(dc), "[zero one two many many]")
        assert_equal(str(dc[0]), "zero")

    def test_self_containing(self):
        arr0d = np.array(None)
        arr0d[()] = arr0d
        assert_equal(repr(arr0d),
            'array(array(..., dtype=object), dtype=object)')
        arr0d[()] = 0  # resolve recursion for garbage collector

        arr1d = np.array([None, None])
        arr1d[1] = arr1d
        assert_equal(repr(arr1d),
            'array([None, array(..., dtype=object)], dtype=object)')
        arr1d[1] = 0  # resolve recursion for garbage collector

        first = np.array(None)
        second = np.array(None)
        first[()] = second
        second[()] = first
        assert_equal(repr(first),
            'array(array(array(..., dtype=object), dtype=object), dtype=object)')
        first[()] = 0  # resolve circular references for garbage collector

    def test_containing_list(self):
        # printing square brackets directly would be ambiguous
        arr1d = np.array([None, None])
        arr1d[0] = [1, 2]
        arr1d[1] = [3]
        assert_equal(repr(arr1d),
            'array([list([1, 2]), list([3])], dtype=object)')

    def test_void_scalar_recursion(self):
        # gh-9345
        repr(np.void(b'test'))  # RecursionError ?

    def test_fieldless_structured(self):
        # gh-10366
        no_fields = np.dtype([])
        arr_no_fields = np.empty(4, dtype=no_fields)
        assert_equal(repr(arr_no_fields), 'array([(), (), (), ()], dtype=[])')


class TestComplexArray:
    def test_str(self):
        rvals = [0, 1, -1, np.inf, -np.inf, np.nan]
        cvals = [complex(rp, ip) for rp in rvals for ip in rvals]
        dtypes = [np.complex64, np.cdouble, np.clongdouble]
        actual = [str(np.array([c], dt)) for c in cvals for dt in dtypes]
        wanted = [
            '[0.+0.j]',    '[0.+0.j]',    '[0.+0.j]',
            '[0.+1.j]',    '[0.+1.j]',    '[0.+1.j]',
            '[0.-1.j]',    '[0.-1.j]',    '[0.-1.j]',
            '[0.+infj]',   '[0.+infj]',   '[0.+infj]',
            '[0.-infj]',   '[0.-infj]',   '[0.-infj]',
            '[0.+nanj]',   '[0.+nanj]',   '[0.+nanj]',
            '[1.+0.j]',    '[1.+0.j]',    '[1.+0.j]',
            '[1.+1.j]',    '[1.+1.j]',    '[1.+1.j]',
            '[1.-1.j]',    '[1.-1.j]',    '[1.-1.j]',
            '[1.+infj]',   '[1.+infj]',   '[1.+infj]',
            '[1.-infj]',   '[1.-infj]',   '[1.-infj]',
            '[1.+nanj]',   '[1.+nanj]',   '[1.+nanj]',
            '[-1.+0.j]',   '[-1.+0.j]',   '[-1.+0.j]',
            '[-1.+1.j]',   '[-1.+1.j]',   '[-1.+1.j]',
            '[-1.-1.j]',   '[-1.-1.j]',   '[-1.-1.j]',
            '[-1.+infj]',  '[-1.+infj]',  '[-1.+infj]',
            '[-1.-infj]',  '[-1.-infj]',  '[-1.-infj]',
            '[-1.+nanj]',  '[-1.+nanj]',  '[-1.+nanj]',
            '[inf+0.j]',   '[inf+0.j]',   '[inf+0.j]',
            '[inf+1.j]',   '[inf+1.j]',   '[inf+1.j]',
            '[inf-1.j]',   '[inf-1.j]',   '[inf-1.j]',
            '[inf+infj]',  '[inf+infj]',  '[inf+infj]',
            '[inf-infj]',  '[inf-infj]',  '[inf-infj]',
            '[inf+nanj]',  '[inf+nanj]',  '[inf+nanj]',
            '[-inf+0.j]',  '[-inf+0.j]',  '[-inf+0.j]',
            '[-inf+1.j]',  '[-inf+1.j]',  '[-inf+1.j]',
            '[-inf-1.j]',  '[-inf-1.j]',  '[-inf-1.j]',
            '[-inf+infj]', '[-inf+infj]', '[-inf+infj]',
            '[-inf-infj]', '[-inf-infj]', '[-inf-infj]',
            '[-inf+nanj]', '[-inf+nanj]', '[-inf+nanj]',
            '[nan+0.j]',   '[nan+0.j]',   '[nan+0.j]',
            '[nan+1.j]',   '[nan+1.j]',   '[nan+1.j]',
            '[nan-1.j]',   '[nan-1.j]',   '[nan-1.j]',
            '[nan+infj]',  '[nan+infj]',  '[nan+infj]',
            '[nan-infj]',  '[nan-infj]',  '[nan-infj]',
            '[nan+nanj]',  '[nan+nanj]',  '[nan+nanj]']

        for res, val in zip(actual, wanted):
            assert_equal(res, val)

class TestArray2String:
    def test_basic(self):
        """Basic test of array2string."""
        a = np.arange(3)
        assert_(np.array2string(a) == '[0 1 2]')
        assert_(np.array2string(a, max_line_width=4, legacy='1.13') == '[0 1\n 2]')
        assert_(np.array2string(a, max_line_width=4) == '[0\n 1\n 2]')

    def test_unexpected_kwarg(self):
        # ensure than an appropriate TypeError
        # is raised when array2string receives
        # an unexpected kwarg

        with assert_raises_regex(TypeError, 'nonsense'):
            np.array2string(np.array([1, 2, 3]),
                            nonsense=None)

    def test_format_function(self):
        """Test custom format function for each element in array."""
        def _format_function(x):
            if np.abs(x) < 1:
                return '.'
            elif np.abs(x) < 2:
                return 'o'
            else:
                return 'O'

        x = np.arange(3)
        x_hex = "[0x0 0x1 0x2]"
        x_oct = "[0o0 0o1 0o2]"
        assert_(np.array2string(x, formatter={'all': _format_function}) ==
                "[. o O]")
        assert_(np.array2string(x, formatter={'int_kind': _format_function}) ==
                "[. o O]")
        assert_(np.array2string(x, formatter={'all': lambda x: f"{x:.4f}"}) ==
                "[0.0000 1.0000 2.0000]")
        assert_equal(np.array2string(x, formatter={'int': hex}),
                x_hex)
        assert_equal(np.array2string(x, formatter={'int': oct}),
                x_oct)

        x = np.arange(3.)
        assert_(np.array2string(x, formatter={'float_kind': lambda x: f"{x:.2f}"}) ==
                "[0.00 1.00 2.00]")
        assert_(np.array2string(x, formatter={'float': lambda x: f"{x:.2f}"}) ==
                "[0.00 1.00 2.00]")

        s = np.array(['abc', 'def'])
        assert_(np.array2string(s, formatter={'numpystr': lambda s: s * 2}) ==
                '[abcabc defdef]')

    def test_structure_format_mixed(self):
        dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
        x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        assert_equal(np.array2string(x),
                "[('Sarah', [8., 7.]) ('John', [6., 7.])]")

        np.set_printoptions(legacy='1.13')
        try:
            # for issue #5692
            A = np.zeros(shape=10, dtype=[("A", "M8[s]")])
            A[5:].fill(np.datetime64('NaT'))
            assert_equal(
                np.array2string(A),
                textwrap.dedent("""\
                [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
                 ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('NaT',) ('NaT',)
                 ('NaT',) ('NaT',) ('NaT',)]""")
            )
        finally:
            np.set_printoptions(legacy=False)

        # same again, but with non-legacy behavior
        assert_equal(
            np.array2string(A),
            textwrap.dedent("""\
            [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
             ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
             ('1970-01-01T00:00:00',) (                'NaT',)
             (                'NaT',) (                'NaT',)
             (                'NaT',) (                'NaT',)]""")
        )

        # and again, with timedeltas
        A = np.full(10, 123456, dtype=[("A", "m8[s]")])
        A[5:].fill(np.datetime64('NaT'))
        assert_equal(
            np.array2string(A),
            textwrap.dedent("""\
            [(123456,) (123456,) (123456,) (123456,) (123456,) ( 'NaT',) ( 'NaT',)
             ( 'NaT',) ( 'NaT',) ( 'NaT',)]""")
        )

    def test_structure_format_int(self):
        # See #8160
        struct_int = np.array([([1, -1],), ([123, 1],)],
                dtype=[('B', 'i4', 2)])
        assert_equal(np.array2string(struct_int),
                "[([  1,  -1],) ([123,   1],)]")
        struct_2dint = np.array([([[0, 1], [2, 3]],), ([[12, 0], [0, 0]],)],
                dtype=[('B', 'i4', (2, 2))])
        assert_equal(np.array2string(struct_2dint),
                "[([[ 0,  1], [ 2,  3]],) ([[12,  0], [ 0,  0]],)]")

    def test_structure_format_float(self):
        # See #8172
        array_scalar = np.array(
                (1., 2.1234567890123456789, 3.), dtype=('f8,f8,f8'))
        assert_equal(np.array2string(array_scalar), "(1., 2.12345679, 3.)")

    def test_unstructured_void_repr(self):
        a = np.array([27, 91, 50, 75, 7, 65, 10, 8, 27, 91, 51, 49, 109, 82, 101, 100],
                      dtype='u1').view('V8')
        assert_equal(repr(a[0]),
            r"np.void(b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08')")
        assert_equal(str(a[0]), r"b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08'")
        assert_equal(repr(a),
            r"array([b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08',"
            "\n"
            r"       b'\x1B\x5B\x33\x31\x6D\x52\x65\x64'], dtype='|V8')")

        assert_equal(eval(repr(a), vars(np)), a)
        assert_equal(eval(repr(a[0]), {'np': np}), a[0])

    def test_edgeitems_kwarg(self):
        # previously the global print options would be taken over the kwarg
        arr = np.zeros(3, int)
        assert_equal(
            np.array2string(arr, edgeitems=1, threshold=0),
            "[0 ... 0]"
        )

    def test_summarize_1d(self):
        A = np.arange(1001)
        strA = '[   0    1    2 ...  998  999 1000]'
        assert_equal(str(A), strA)

        reprA = 'array([   0,    1,    2, ...,  998,  999, 1000])'
        try:
            np.set_printoptions(legacy='2.1')
            assert_equal(repr(A), reprA)
        finally:
            np.set_printoptions(legacy=False)

        assert_equal(repr(A), reprA.replace(')', ', shape=(1001,))'))

    def test_summarize_2d(self):
        A = np.arange(1002).reshape(2, 501)
        strA = '[[   0    1    2 ...  498  499  500]\n' \
               ' [ 501  502  503 ...  999 1000 1001]]'
        assert_equal(str(A), strA)

        reprA = 'array([[   0,    1,    2, ...,  498,  499,  500],\n' \
                '       [ 501,  502,  503, ...,  999, 1000, 1001]])'
        try:
            np.set_printoptions(legacy='2.1')
            assert_equal(repr(A), reprA)
        finally:
            np.set_printoptions(legacy=False)

        assert_equal(repr(A), reprA.replace(')', ', shape=(2, 501))'))

    def test_summarize_2d_dtype(self):
        A = np.arange(1002, dtype='i2').reshape(2, 501)
        strA = '[[   0    1    2 ...  498  499  500]\n' \
               ' [ 501  502  503 ...  999 1000 1001]]'
        assert_equal(str(A), strA)

        reprA = ('array([[   0,    1,    2, ...,  498,  499,  500],\n'
                 '       [ 501,  502,  503, ...,  999, 1000, 1001]],\n'
                 '      shape=(2, 501), dtype=int16)')
        assert_equal(repr(A), reprA)

    def test_summarize_structure(self):
        A = (np.arange(2002, dtype="<i8").reshape(2, 1001)
             .view([('i', "<i8", (1001,))]))
        strA = ("[[([   0,    1,    2, ...,  998,  999, 1000],)]\n"
                " [([1001, 1002, 1003, ..., 1999, 2000, 2001],)]]")
        assert_equal(str(A), strA)

        reprA = ("array([[([   0,    1,    2, ...,  998,  999, 1000],)],\n"
                 "       [([1001, 1002, 1003, ..., 1999, 2000, 2001],)]],\n"
                 "      dtype=[('i', '<i8', (1001,))])")
        assert_equal(repr(A), reprA)

        B = np.ones(2002, dtype=">i8").view([('i', ">i8", (2, 1001))])
        strB = "[([[1, 1, 1, ..., 1, 1, 1], [1, 1, 1, ..., 1, 1, 1]],)]"
        assert_equal(str(B), strB)

        reprB = (
            "array([([[1, 1, 1, ..., 1, 1, 1], [1, 1, 1, ..., 1, 1, 1]],)],\n"
            "      dtype=[('i', '>i8', (2, 1001))])"
        )
        assert_equal(repr(B), reprB)

        C = (np.arange(22, dtype="<i8").reshape(2, 11)
             .view([('i1', "<i8"), ('i10', "<i8", (10,))]))
        strC = "[[( 0, [ 1, ..., 10])]\n [(11, [12, ..., 21])]]"
        assert_equal(np.array2string(C, threshold=1, edgeitems=1), strC)

    def test_linewidth(self):
        a = np.full(6, 1)

        def make_str(a, width, **kw):
            return np.array2string(a, separator="", max_line_width=width, **kw)

        assert_equal(make_str(a, 8, legacy='1.13'), '[111111]')
        assert_equal(make_str(a, 7, legacy='1.13'), '[111111]')
        assert_equal(make_str(a, 5, legacy='1.13'), '[1111\n'
                                                    ' 11]')

        assert_equal(make_str(a, 8), '[111111]')
        assert_equal(make_str(a, 7), '[11111\n'
                                     ' 1]')
        assert_equal(make_str(a, 5), '[111\n'
                                     ' 111]')

        b = a[None, None, :]

        assert_equal(make_str(b, 12, legacy='1.13'), '[[[111111]]]')
        assert_equal(make_str(b,  9, legacy='1.13'), '[[[111111]]]')
        assert_equal(make_str(b,  8, legacy='1.13'), '[[[11111\n'
                                                     '   1]]]')

        assert_equal(make_str(b, 12), '[[[111111]]]')
        assert_equal(make_str(b,  9), '[[[111\n'
                                      '   111]]]')
        assert_equal(make_str(b,  8), '[[[11\n'
                                      '   11\n'
                                      '   11]]]')

    def test_wide_element(self):
        a = np.array(['xxxxx'])
        assert_equal(
            np.array2string(a, max_line_width=5),
            "['xxxxx']"
        )
        assert_equal(
            np.array2string(a, max_line_width=5, legacy='1.13'),
            "[ 'xxxxx']"
        )

    def test_multiline_repr(self):
        class MultiLine:
            def __repr__(self):
                return "Line 1\nLine 2"

        a = np.array([[None, MultiLine()], [MultiLine(), None]])

        assert_equal(
            np.array2string(a),
            '[[None Line 1\n'
            '       Line 2]\n'
            ' [Line 1\n'
            '  Line 2 None]]'
        )
        assert_equal(
            np.array2string(a, max_line_width=5),
            '[[None\n'
            '  Line 1\n'
            '  Line 2]\n'
            ' [Line 1\n'
            '  Line 2\n'
            '  None]]'
        )
        assert_equal(
            repr(a),
            'array([[None, Line 1\n'
            '              Line 2],\n'
            '       [Line 1\n'
            '        Line 2, None]], dtype=object)'
        )

        class MultiLineLong:
            def __repr__(self):
                return "Line 1\nLooooooooooongestLine2\nLongerLine 3"

        a = np.array([[None, MultiLineLong()], [MultiLineLong(), None]])
        assert_equal(
            repr(a),
            'array([[None, Line 1\n'
            '              LooooooooooongestLine2\n'
            '              LongerLine 3          ],\n'
            '       [Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          , None]], dtype=object)'
        )
        assert_equal(
            np.array_repr(a, 20),
            'array([[None,\n'
            '        Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          ],\n'
            '       [Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          ,\n'
            '        None]],\n'
            '      dtype=object)'
        )

    def test_nested_array_repr(self):
        a = np.empty((2, 2), dtype=object)
        a[0, 0] = np.eye(2)
        a[0, 1] = np.eye(3)
        a[1, 0] = None
        a[1, 1] = np.ones((3, 1))
        assert_equal(
            repr(a),
            'array([[array([[1., 0.],\n'
            '               [0., 1.]]), array([[1., 0., 0.],\n'
            '                                  [0., 1., 0.],\n'
            '                                  [0., 0., 1.]])],\n'
            '       [None, array([[1.],\n'
            '                     [1.],\n'
            '                     [1.]])]], dtype=object)'
        )

    @given(hynp.from_dtype(np.dtype("U")))
    def test_any_text(self, text):
        # This test checks that, given any value that can be represented in an
        # array of dtype("U") (i.e. unicode string), ...
        a = np.array([text, text, text])
        # casting a list of them to an array does not e.g. truncate the value
        assert_equal(a[0], text)
        text = text.item()  # use raw python strings for repr below
        # and that np.array2string puts a newline in the expected location
        expected_repr = f"[{text!r} {text!r}\n {text!r}]"
        result = np.array2string(a, max_line_width=len(repr(text)) * 2 + 3)
        assert_equal(result, expected_repr)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_refcount(self):
        # make sure we do not hold references to the array due to a recursive
        # closure (gh-10620)
        gc.disable()
        a = np.arange(2)
        r1 = sys.getrefcount(a)
        np.array2string(a)
        np.array2string(a)
        r2 = sys.getrefcount(a)
        gc.collect()
        gc.enable()
        assert_(r1 == r2)

    def test_with_sign(self):
        # mixed negative and positive value array
        a = np.array([-2, 0, 3])
        assert_equal(
            np.array2string(a, sign='+'),
            '[-2 +0 +3]'
        )
        assert_equal(
            np.array2string(a, sign='-'),
            '[-2  0  3]'
        )
        assert_equal(
            np.array2string(a, sign=' '),
            '[-2  0  3]'
        )
        # all non-negative array
        a = np.array([2, 0, 3])
        assert_equal(
            np.array2string(a, sign='+'),
            '[+2 +0 +3]'
        )
        assert_equal(
            np.array2string(a, sign='-'),
            '[2 0 3]'
        )
        assert_equal(
            np.array2string(a, sign=' '),
            '[ 2  0  3]'
        )
        # all negative array
        a = np.array([-2, -1, -3])
        assert_equal(
            np.array2string(a, sign='+'),
            '[-2 -1 -3]'
        )
        assert_equal(
            np.array2string(a, sign='-'),
            '[-2 -1 -3]'
        )
        assert_equal(
            np.array2string(a, sign=' '),
            '[-2 -1 -3]'
        )
        # 2d array mixed negative and positive
        a = np.array([[10, -1, 1, 1], [10, 10, 10, 10]])
        assert_equal(
            np.array2string(a, sign='+'),
            '[[+10  -1  +1  +1]\n [+10 +10 +10 +10]]'
        )
        assert_equal(
            np.array2string(a, sign='-'),
            '[[10 -1  1  1]\n [10 10 10 10]]'
        )
        assert_equal(
            np.array2string(a, sign=' '),
            '[[10 -1  1  1]\n [10 10 10 10]]'
        )
        # 2d array all positive
        a = np.array([[10, 0, 1, 1], [10, 10, 10, 10]])
        assert_equal(
            np.array2string(a, sign='+'),
            '[[+10  +0  +1  +1]\n [+10 +10 +10 +10]]'
        )
        assert_equal(
            np.array2string(a, sign='-'),
            '[[10  0  1  1]\n [10 10 10 10]]'
        )
        assert_equal(
            np.array2string(a, sign=' '),
            '[[ 10   0   1   1]\n [ 10  10  10  10]]'
        )
        # 2d array all negative
        a = np.array([[-10, -1, -1, -1], [-10, -10, -10, -10]])
        assert_equal(
            np.array2string(a, sign='+'),
            '[[-10  -1  -1  -1]\n [-10 -10 -10 -10]]'
        )
        assert_equal(
            np.array2string(a, sign='-'),
            '[[-10  -1  -1  -1]\n [-10 -10 -10 -10]]'
        )
        assert_equal(
            np.array2string(a, sign=' '),
            '[[-10  -1  -1  -1]\n [-10 -10 -10 -10]]'
        )


class TestPrintOptions:
    """Test getting and setting global print options."""

    def setup_method(self):
        self.oldopts = np.get_printoptions()

    def teardown_method(self):
        np.set_printoptions(**self.oldopts)

    def test_basic(self):
        x = np.array([1.5, 0, 1.234567890])
        assert_equal(repr(x), "array([1.5       , 0.        , 1.23456789])")
        ret = np.set_printoptions(precision=4)
        assert_equal(repr(x), "array([1.5   , 0.    , 1.2346])")
        assert ret is None

    def test_precision_zero(self):
        np.set_printoptions(precision=0)
        for values, string in (
                ([0.], "0."), ([.3], "0."), ([-.3], "-0."), ([.7], "1."),
                ([1.5], "2."), ([-1.5], "-2."), ([-15.34], "-15."),
                ([100.], "100."), ([.2, -1, 122.51], "  0.,  -1., 123."),
                ([0], "0"), ([-12], "-12"), ([complex(.3, -.7)], "0.-1.j")):
            x = np.array(values)
            assert_equal(repr(x), f"array([{string}])")

    def test_formatter(self):
        x = np.arange(3)
        np.set_printoptions(formatter={'all': lambda x: str(x - 1)})
        assert_equal(repr(x), "array([-1, 0, 1])")

    def test_formatter_reset(self):
        x = np.arange(3)
        np.set_printoptions(formatter={'all': lambda x: str(x - 1)})
        assert_equal(repr(x), "array([-1, 0, 1])")
        np.set_printoptions(formatter={'int': None})
        assert_equal(repr(x), "array([0, 1, 2])")

        np.set_printoptions(formatter={'all': lambda x: str(x - 1)})
        assert_equal(repr(x), "array([-1, 0, 1])")
        np.set_printoptions(formatter={'all': None})
        assert_equal(repr(x), "array([0, 1, 2])")

        np.set_printoptions(formatter={'int': lambda x: str(x - 1)})
        assert_equal(repr(x), "array([-1, 0, 1])")
        np.set_printoptions(formatter={'int_kind': None})
        assert_equal(repr(x), "array([0, 1, 2])")

        x = np.arange(3.)
        np.set_printoptions(formatter={'float': lambda x: str(x - 1)})
        assert_equal(repr(x), "array([-1.0, 0.0, 1.0])")
        np.set_printoptions(formatter={'float_kind': None})
        assert_equal(repr(x), "array([0., 1., 2.])")

    def test_override_repr(self):
        x = np.arange(3)
        np.set_printoptions(override_repr=lambda x: "FOO")
        assert_equal(repr(x), "FOO")
        np.set_printoptions(override_repr=None)
        assert_equal(repr(x), "array([0, 1, 2])")

        with np.printoptions(override_repr=lambda x: "BAR"):
            assert_equal(repr(x), "BAR")
        assert_equal(repr(x), "array([0, 1, 2])")

    def test_0d_arrays(self):
        assert_equal(str(np.array('café', '<U4')), 'café')

        assert_equal(repr(np.array('café', '<U4')),
                     "array('café', dtype='<U4')")
        assert_equal(str(np.array('test', np.str_)), 'test')

        a = np.zeros(1, dtype=[('a', '<i4', (3,))])
        assert_equal(str(a[0]), '([0, 0, 0],)')

        assert_equal(repr(np.datetime64('2005-02-25')[...]),
                     "array('2005-02-25', dtype='datetime64[D]')")

        assert_equal(repr(np.timedelta64('10', 'Y')[...]),
                     "array(10, dtype='timedelta64[Y]')")

        # repr of 0d arrays is affected by printoptions
        x = np.array(1)
        np.set_printoptions(formatter={'all': lambda x: "test"})
        assert_equal(repr(x), "array(test)")
        # str is unaffected
        assert_equal(str(x), "1")

        # check `style` arg raises
        assert_warns(DeprecationWarning, np.array2string,
                                         np.array(1.), style=repr)
        # but not in legacy mode
        np.array2string(np.array(1.), style=repr, legacy='1.13')
        # gh-10934 style was broken in legacy mode, check it works
        np.array2string(np.array(1.), legacy='1.13')

    def test_float_spacing(self):
        x = np.array([1., 2., 3.])
        y = np.array([1., 2., -10.])
        z = np.array([100., 2., -1.])
        w = np.array([-100., 2., 1.])

        assert_equal(repr(x), 'array([1., 2., 3.])')
        assert_equal(repr(y), 'array([  1.,   2., -10.])')
        assert_equal(repr(np.array(y[0])), 'array(1.)')
        assert_equal(repr(np.array(y[-1])), 'array(-10.)')
        assert_equal(repr(z), 'array([100.,   2.,  -1.])')
        assert_equal(repr(w), 'array([-100.,    2.,    1.])')

        assert_equal(repr(np.array([np.nan, np.inf])), 'array([nan, inf])')
        assert_equal(repr(np.array([np.nan, -np.inf])), 'array([ nan, -inf])')

        x = np.array([np.inf, 100000, 1.1234])
        y = np.array([np.inf, 100000, -1.1234])
        z = np.array([np.inf, 1.1234, -1e120])
        np.set_printoptions(precision=2)
        assert_equal(repr(x), 'array([     inf, 1.00e+05, 1.12e+00])')
        assert_equal(repr(y), 'array([      inf,  1.00e+05, -1.12e+00])')
        assert_equal(repr(z), 'array([       inf,  1.12e+000, -1.00e+120])')

    def test_bool_spacing(self):
        assert_equal(repr(np.array([True, True])),
                     'array([ True,  True])')
        assert_equal(repr(np.array([True, False])),
                     'array([ True, False])')
        assert_equal(repr(np.array([True])),
                     'array([ True])')
        assert_equal(repr(np.array(True)),
                     'array(True)')
        assert_equal(repr(np.array(False)),
                     'array(False)')

    def test_sign_spacing(self):
        a = np.arange(4.)
        b = np.array([1.234e9])
        c = np.array([1.0 + 1.0j, 1.123456789 + 1.123456789j], dtype='c16')

        assert_equal(repr(a), 'array([0., 1., 2., 3.])')
        assert_equal(repr(np.array(1.)), 'array(1.)')
        assert_equal(repr(b), 'array([1.234e+09])')
        assert_equal(repr(np.array([0.])), 'array([0.])')
        assert_equal(repr(c),
            "array([1.        +1.j        , 1.12345679+1.12345679j])")
        assert_equal(repr(np.array([0., -0.])), 'array([ 0., -0.])')

        np.set_printoptions(sign=' ')
        assert_equal(repr(a), 'array([ 0.,  1.,  2.,  3.])')
        assert_equal(repr(np.array(1.)), 'array( 1.)')
        assert_equal(repr(b), 'array([ 1.234e+09])')
        assert_equal(repr(c),
            "array([ 1.        +1.j        ,  1.12345679+1.12345679j])")
        assert_equal(repr(np.array([0., -0.])), 'array([ 0., -0.])')

        np.set_printoptions(sign='+')
        assert_equal(repr(a), 'array([+0., +1., +2., +3.])')
        assert_equal(repr(np.array(1.)), 'array(+1.)')
        assert_equal(repr(b), 'array([+1.234e+09])')
        assert_equal(repr(c),
            "array([+1.        +1.j        , +1.12345679+1.12345679j])")

        np.set_printoptions(legacy='1.13')
        assert_equal(repr(a), 'array([ 0.,  1.,  2.,  3.])')
        assert_equal(repr(b),  'array([  1.23400000e+09])')
        assert_equal(repr(-b), 'array([ -1.23400000e+09])')
        assert_equal(repr(np.array(1.)), 'array(1.0)')
        assert_equal(repr(np.array([0.])), 'array([ 0.])')
        assert_equal(repr(c),
            "array([ 1.00000000+1.j        ,  1.12345679+1.12345679j])")
        # gh-10383
        assert_equal(str(np.array([-1., 10])), "[ -1.  10.]")

        assert_raises(TypeError, np.set_printoptions, wrongarg=True)

    def test_float_overflow_nowarn(self):
        # make sure internal computations in FloatingFormat don't
        # warn about overflow
        repr(np.array([1e4, 0.1], dtype='f2'))

    def test_sign_spacing_structured(self):
        a = np.ones(2, dtype='<f,<f')
        assert_equal(repr(a),
            "array([(1., 1.), (1., 1.)], dtype=[('f0', '<f4'), ('f1', '<f4')])")
        assert_equal(repr(a[0]),
            "np.void((1.0, 1.0), dtype=[('f0', '<f4'), ('f1', '<f4')])")

    def test_floatmode(self):
        x = np.array([0.6104, 0.922, 0.457, 0.0906, 0.3733, 0.007244,
                      0.5933, 0.947, 0.2383, 0.4226], dtype=np.float16)
        y = np.array([0.2918820979355541, 0.5064172631089138,
                      0.2848750619642916, 0.4342965294660567,
                      0.7326538397312751, 0.3459503329096204,
                      0.0862072768214508, 0.39112753029631175],
                      dtype=np.float64)
        z = np.arange(6, dtype=np.float16) / 10
        c = np.array([1.0 + 1.0j, 1.123456789 + 1.123456789j], dtype='c16')

        # also make sure 1e23 is right (is between two fp numbers)
        w = np.array([f'1e{i}' for i in range(25)], dtype=np.float64)
        # note: we construct w from the strings `1eXX` instead of doing
        # `10.**arange(24)` because it turns out the two are not equivalent in
        # python. On some architectures `1e23 != 10.**23`.
        wp = np.array([1.234e1, 1e2, 1e123])

        # unique mode
        np.set_printoptions(floatmode='unique')
        assert_equal(repr(x),
            "array([0.6104  , 0.922   , 0.457   , 0.0906  , 0.3733  , 0.007244,\n"
            "       0.5933  , 0.947   , 0.2383  , 0.4226  ], dtype=float16)")
        assert_equal(repr(y),
            "array([0.2918820979355541 , 0.5064172631089138 , 0.2848750619642916 ,\n"
            "       0.4342965294660567 , 0.7326538397312751 , 0.3459503329096204 ,\n"
            "       0.0862072768214508 , 0.39112753029631175])")
        assert_equal(repr(z),
            "array([0. , 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float16)")
        assert_equal(repr(w),
            "array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06, 1.e+07,\n"
            "       1.e+08, 1.e+09, 1.e+10, 1.e+11, 1.e+12, 1.e+13, 1.e+14, 1.e+15,\n"
            "       1.e+16, 1.e+17, 1.e+18, 1.e+19, 1.e+20, 1.e+21, 1.e+22, 1.e+23,\n"
            "       1.e+24])")
        assert_equal(repr(wp), "array([1.234e+001, 1.000e+002, 1.000e+123])")
        assert_equal(repr(c),
            "array([1.         +1.j         , 1.123456789+1.123456789j])")

        # maxprec mode, precision=8
        np.set_printoptions(floatmode='maxprec', precision=8)
        assert_equal(repr(x),
            "array([0.6104  , 0.922   , 0.457   , 0.0906  , 0.3733  , 0.007244,\n"
            "       0.5933  , 0.947   , 0.2383  , 0.4226  ], dtype=float16)")
        assert_equal(repr(y),
            "array([0.2918821 , 0.50641726, 0.28487506, 0.43429653, 0.73265384,\n"
            "       0.34595033, 0.08620728, 0.39112753])")
        assert_equal(repr(z),
            "array([0. , 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float16)")
        assert_equal(repr(w[::5]),
            "array([1.e+00, 1.e+05, 1.e+10, 1.e+15, 1.e+20])")
        assert_equal(repr(wp), "array([1.234e+001, 1.000e+002, 1.000e+123])")
        assert_equal(repr(c),
            "array([1.        +1.j        , 1.12345679+1.12345679j])")

        # fixed mode, precision=4
        np.set_printoptions(floatmode='fixed', precision=4)
        assert_equal(repr(x),
            "array([0.6104, 0.9219, 0.4570, 0.0906, 0.3733, 0.0072, 0.5933, 0.9468,\n"
            "       0.2383, 0.4226], dtype=float16)")
        assert_equal(repr(y),
            "array([0.2919, 0.5064, 0.2849, 0.4343, 0.7327, 0.3460, 0.0862, 0.3911])")
        assert_equal(repr(z),
            "array([0.0000, 0.1000, 0.2000, 0.3000, 0.3999, 0.5000], dtype=float16)")
        assert_equal(repr(w[::5]),
            "array([1.0000e+00, 1.0000e+05, 1.0000e+10, 1.0000e+15, 1.0000e+20])")
        assert_equal(repr(wp), "array([1.2340e+001, 1.0000e+002, 1.0000e+123])")
        assert_equal(repr(np.zeros(3)), "array([0.0000, 0.0000, 0.0000])")
        assert_equal(repr(c),
            "array([1.0000+1.0000j, 1.1235+1.1235j])")
        # for larger precision, representation error becomes more apparent:
        np.set_printoptions(floatmode='fixed', precision=8)
        assert_equal(repr(z),
            "array([0.00000000, 0.09997559, 0.19995117, 0.30004883, 0.39990234,\n"
            "       0.50000000], dtype=float16)")

        # maxprec_equal  mode, precision=8
        np.set_printoptions(floatmode='maxprec_equal', precision=8)
        assert_equal(repr(x),
            "array([0.610352, 0.921875, 0.457031, 0.090576, 0.373291, 0.007244,\n"
            "       0.593262, 0.946777, 0.238281, 0.422607], dtype=float16)")
        assert_equal(repr(y),
            "array([0.29188210, 0.50641726, 0.28487506, 0.43429653, 0.73265384,\n"
            "       0.34595033, 0.08620728, 0.39112753])")
        assert_equal(repr(z),
            "array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float16)")
        assert_equal(repr(w[::5]),
            "array([1.e+00, 1.e+05, 1.e+10, 1.e+15, 1.e+20])")
        assert_equal(repr(wp), "array([1.234e+001, 1.000e+002, 1.000e+123])")
        assert_equal(repr(c),
            "array([1.00000000+1.00000000j, 1.12345679+1.12345679j])")

        # test unique special case (gh-18609)
        a = np.float64.fromhex('-1p-97')
        assert_equal(np.float64(np.array2string(a, floatmode='unique')), a)

    test_cases_gh_28679 = [
        (np.half([999, 999]), "[999. 999.]"),
        (np.half([999, 1000]), "[9.99e+02 1.00e+03]"),
        (np.single([999999, 999999]), "[999999. 999999.]"),
        (np.single([999999, -1000000]), "[ 9.99999e+05 -1.00000e+06]"),
        (
            np.complex64([999999 + 999999j, 999999 + 999999j]),
            "[999999.+999999.j 999999.+999999.j]"
        ),
        (
            np.complex64([999999 + 999999j, 999999 + -1000000j]),
            "[999999.+9.99999e+05j 999999.-1.00000e+06j]"
        ),
    ]

    @pytest.mark.parametrize("input_array, expected_str", test_cases_gh_28679)
    def test_gh_28679(self, input_array, expected_str):
        # test cutoff to exponent notation for half, single, and complex64
        assert_equal(str(input_array), expected_str)

    test_cases_legacy_2_2 = [
        (np.half([1.e3, 1.e4, 65504]), "[ 1000. 10000. 65504.]"),
        (np.single([1.e6, 1.e7]), "[ 1000000. 10000000.]"),
        (np.single([1.e7, 1.e8]), "[1.e+07 1.e+08]"),
    ]

    @pytest.mark.parametrize("input_array, expected_str", test_cases_legacy_2_2)
    def test_legacy_2_2_mode(self, input_array, expected_str):
        # test legacy cutoff to exponent notation for half and single
        with np.printoptions(legacy='2.2'):
            assert_equal(str(input_array), expected_str)

    @pytest.mark.parametrize("legacy", ['1.13', '1.21', '1.25', '2.1', '2.2'])
    def test_legacy_get_options(self, legacy):
        # test legacy get options works okay
        with np.printoptions(legacy=legacy):
            p_opt = np.get_printoptions()
            assert_equal(p_opt["legacy"], legacy)

    def test_legacy_mode_scalars(self):
        # in legacy mode, str of floats get truncated, and complex scalars
        # use * for non-finite imaginary part
        np.set_printoptions(legacy='1.13')
        assert_equal(str(np.float64(1.123456789123456789)), '1.12345678912')
        assert_equal(str(np.complex128(complex(1, np.nan))), '(1+nan*j)')

        np.set_printoptions(legacy=False)
        assert_equal(str(np.float64(1.123456789123456789)),
                     '1.1234567891234568')
        assert_equal(str(np.complex128(complex(1, np.nan))), '(1+nanj)')

    def test_legacy_stray_comma(self):
        np.set_printoptions(legacy='1.13')
        assert_equal(str(np.arange(10000)), '[   0    1    2 ..., 9997 9998 9999]')

        np.set_printoptions(legacy=False)
        assert_equal(str(np.arange(10000)), '[   0    1    2 ... 9997 9998 9999]')

    def test_dtype_linewidth_wrapping(self):
        np.set_printoptions(linewidth=75)
        assert_equal(repr(np.arange(10, 20., dtype='f4')),
            "array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19.], dtype=float32)")
        assert_equal(repr(np.arange(10, 23., dtype='f4')), textwrap.dedent("""\
            array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.],
                  dtype=float32)"""))

        styp = '<U4'
        assert_equal(repr(np.ones(3, dtype=styp)),
            f"array(['1', '1', '1'], dtype='{styp}')")
        assert_equal(repr(np.ones(12, dtype=styp)), textwrap.dedent(f"""\
            array(['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                  dtype='{styp}')"""))

    @pytest.mark.parametrize(
        ['native'],
        [
            ('bool',),
            ('uint8',),
            ('uint16',),
            ('uint32',),
            ('uint64',),
            ('int8',),
            ('int16',),
            ('int32',),
            ('int64',),
            ('float16',),
            ('float32',),
            ('float64',),
            ('U1',),     # 4-byte width string
        ],
    )
    def test_dtype_endianness_repr(self, native):
        '''
        there was an issue where
        repr(array([0], dtype='<u2')) and repr(array([0], dtype='>u2'))
        both returned the same thing:
        array([0], dtype=uint16)
        even though their dtypes have different endianness.
        '''
        native_dtype = np.dtype(native)
        non_native_dtype = native_dtype.newbyteorder()
        non_native_repr = repr(np.array([1], non_native_dtype))
        native_repr = repr(np.array([1], native_dtype))
        # preserve the sensible default of only showing dtype if nonstandard
        assert ('dtype' in native_repr) ^ (native_dtype in _typelessdata),\
                ("an array's repr should show dtype if and only if the type "
                 'of the array is NOT one of the standard types '
                 '(e.g., int32, bool, float64).')
        if non_native_dtype.itemsize > 1:
            # if the type is >1 byte, the non-native endian version
            # must show endianness.
            assert non_native_repr != native_repr
            assert f"dtype='{non_native_dtype.byteorder}" in non_native_repr

    def test_linewidth_repr(self):
        a = np.full(7, fill_value=2)
        np.set_printoptions(linewidth=17)
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2,
                   2])""")
        )
        np.set_printoptions(linewidth=17, legacy='1.13')
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2, 2])""")
        )

        a = np.full(8, fill_value=2)

        np.set_printoptions(linewidth=18, legacy=False)
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2,
                   2, 2])""")
        )

        np.set_printoptions(linewidth=18, legacy='1.13')
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2, 2,
                   2, 2, 2, 2])""")
        )

    def test_linewidth_str(self):
        a = np.full(18, fill_value=2)
        np.set_printoptions(linewidth=18)
        assert_equal(
            str(a),
            textwrap.dedent("""\
            [2 2 2 2 2 2 2 2
             2 2 2 2 2 2 2 2
             2 2]""")
        )
        np.set_printoptions(linewidth=18, legacy='1.13')
        assert_equal(
            str(a),
            textwrap.dedent("""\
            [2 2 2 2 2 2 2 2 2
             2 2 2 2 2 2 2 2 2]""")
        )

    def test_edgeitems(self):
        np.set_printoptions(edgeitems=1, threshold=1)
        a = np.arange(27).reshape((3, 3, 3))
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([[[ 0, ...,  2],
                    ...,
                    [ 6, ...,  8]],

                   ...,

                   [[18, ..., 20],
                    ...,
                    [24, ..., 26]]], shape=(3, 3, 3))""")
        )

        b = np.zeros((3, 3, 1, 1))
        assert_equal(
            repr(b),
            textwrap.dedent("""\
            array([[[[0.]],

                    ...,

                    [[0.]]],


                   ...,


                   [[[0.]],

                    ...,

                    [[0.]]]], shape=(3, 3, 1, 1))""")
        )

        # 1.13 had extra trailing spaces, and was missing newlines
        try:
            np.set_printoptions(legacy='1.13')
            assert_equal(repr(a), (
                "array([[[ 0, ...,  2],\n"
                "        ..., \n"
                "        [ 6, ...,  8]],\n"
                "\n"
                "       ..., \n"
                "       [[18, ..., 20],\n"
                "        ..., \n"
                "        [24, ..., 26]]])")
            )
            assert_equal(repr(b), (
                "array([[[[ 0.]],\n"
                "\n"
                "        ..., \n"
                "        [[ 0.]]],\n"
                "\n"
                "\n"
                "       ..., \n"
                "       [[[ 0.]],\n"
                "\n"
                "        ..., \n"
                "        [[ 0.]]]])")
            )
        finally:
            np.set_printoptions(legacy=False)

    def test_edgeitems_structured(self):
        np.set_printoptions(edgeitems=1, threshold=1)
        A = np.arange(5 * 2 * 3, dtype="<i8").view([('i', "<i8", (5, 2, 3))])
        reprA = (
            "array([([[[ 0, ...,  2], [ 3, ...,  5]], ..., "
            "[[24, ..., 26], [27, ..., 29]]],)],\n"
            "      dtype=[('i', '<i8', (5, 2, 3))])"
        )
        assert_equal(repr(A), reprA)

    def test_bad_args(self):
        assert_raises(ValueError, np.set_printoptions, threshold=float('nan'))
        assert_raises(TypeError, np.set_printoptions, threshold='1')
        assert_raises(TypeError, np.set_printoptions, threshold=b'1')

        assert_raises(TypeError, np.set_printoptions, precision='1')
        assert_raises(TypeError, np.set_printoptions, precision=1.5)

def test_unicode_object_array():
    expected = "array(['é'], dtype=object)"
    x = np.array(['\xe9'], dtype=object)
    assert_equal(repr(x), expected)


class TestContextManager:
    def test_ctx_mgr(self):
        # test that context manager actually works
        with np.printoptions(precision=2):
            s = str(np.array([2.0]) / 3)
        assert_equal(s, '[0.67]')

    def test_ctx_mgr_restores(self):
        # test that print options are actually restored
        opts = np.get_printoptions()
        with np.printoptions(precision=opts['precision'] - 1,
                             linewidth=opts['linewidth'] - 4):
            pass
        assert_equal(np.get_printoptions(), opts)

    def test_ctx_mgr_exceptions(self):
        # test that print options are restored even if an exception is raised
        opts = np.get_printoptions()
        try:
            with np.printoptions(precision=2, linewidth=11):
                raise ValueError
        except ValueError:
            pass
        assert_equal(np.get_printoptions(), opts)

    def test_ctx_mgr_as_smth(self):
        opts = {"precision": 2}
        with np.printoptions(**opts) as ctx:
            saved_opts = ctx.copy()
        assert_equal({k: saved_opts[k] for k in opts}, opts)


@pytest.mark.parametrize("dtype", "bhilqpBHILQPefdgFDG")
@pytest.mark.parametrize("value", [0, 1])
def test_scalar_repr_numbers(dtype, value):
    # Test NEP 51 scalar repr (and legacy option) for numeric types
    dtype = np.dtype(dtype)
    scalar = np.array(value, dtype=dtype)[()]
    assert isinstance(scalar, np.generic)

    string = str(scalar)
    repr_string = string.strip("()")  # complex may have extra brackets
    representation = repr(scalar)
    if dtype.char == "g":
        assert representation == f"np.longdouble('{repr_string}')"
    elif dtype.char == 'G':
        assert representation == f"np.clongdouble('{repr_string}')"
    else:
        normalized_name = np.dtype(f"{dtype.kind}{dtype.itemsize}").type.__name__
        assert representation == f"np.{normalized_name}({repr_string})"

    with np.printoptions(legacy="1.25"):
        assert repr(scalar) == string


@pytest.mark.parametrize("scalar, legacy_repr, representation", [
        (np.True_, "True", "np.True_"),
        (np.bytes_(b'a'), "b'a'", "np.bytes_(b'a')"),
        (np.str_('a'), "'a'", "np.str_('a')"),
        (np.datetime64("2012"),
            "numpy.datetime64('2012')", "np.datetime64('2012')"),
        (np.timedelta64(1), "numpy.timedelta64(1)", "np.timedelta64(1)"),
        (np.void((True, 2), dtype="?,<i8"),
            "(True, 2)",
            "np.void((True, 2), dtype=[('f0', '?'), ('f1', '<i8')])"),
        (np.void((1, 2), dtype="<f8,>f4"),
            "(1., 2.)",
            "np.void((1.0, 2.0), dtype=[('f0', '<f8'), ('f1', '>f4')])"),
        (np.void(b'a'), r"void(b'\x61')", r"np.void(b'\x61')"),
    ])
def test_scalar_repr_special(scalar, legacy_repr, representation):
    # Test NEP 51 scalar repr (and legacy option) for numeric types
    assert repr(scalar) == representation

    with np.printoptions(legacy="1.25"):
        assert repr(scalar) == legacy_repr

def test_scalar_void_float_str():
    # Note that based on this currently we do not print the same as a tuple
    # would, since the tuple would include the repr() inside for floats, but
    # we do not do that.
    scalar = np.void((1.0, 2.0), dtype=[('f0', '<f8'), ('f1', '>f4')])
    assert str(scalar) == "(1.0, 2.0)"

@pytest.mark.skipif(IS_WASM, reason="wasm doesn't support asyncio")
@pytest.mark.skipif(sys.version_info < (3, 11),
                    reason="asyncio.barrier was added in Python 3.11")
def test_printoptions_asyncio_safe():
    asyncio = pytest.importorskip("asyncio")

    b = asyncio.Barrier(2)

    async def legacy_113():
        np.set_printoptions(legacy='1.13', precision=12)
        await b.wait()
        po = np.get_printoptions()
        assert po['legacy'] == '1.13'
        assert po['precision'] == 12
        orig_linewidth = po['linewidth']
        with np.printoptions(linewidth=34, legacy='1.21'):
            po = np.get_printoptions()
            assert po['legacy'] == '1.21'
            assert po['precision'] == 12
            assert po['linewidth'] == 34
        po = np.get_printoptions()
        assert po['linewidth'] == orig_linewidth
        assert po['legacy'] == '1.13'
        assert po['precision'] == 12

    async def legacy_125():
        np.set_printoptions(legacy='1.25', precision=7)
        await b.wait()
        po = np.get_printoptions()
        assert po['legacy'] == '1.25'
        assert po['precision'] == 7
        orig_linewidth = po['linewidth']
        with np.printoptions(linewidth=6, legacy='1.13'):
            po = np.get_printoptions()
            assert po['legacy'] == '1.13'
            assert po['precision'] == 7
            assert po['linewidth'] == 6
        po = np.get_printoptions()
        assert po['linewidth'] == orig_linewidth
        assert po['legacy'] == '1.25'
        assert po['precision'] == 7

    async def main():
        await asyncio.gather(legacy_125(), legacy_125())

    loop = asyncio.new_event_loop()
    asyncio.run(main())
    loop.close()

@pytest.mark.skipif(IS_WASM, reason="wasm doesn't support threads")
def test_multithreaded_array_printing():
    # the dragon4 implementation uses a static scratch space for performance
    # reasons this test makes sure it is set up in a thread-safe manner

    run_threaded(TestPrintOptions().test_floatmode, 500)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_greenlet.py ===
import gc
import sys
import time
import threading
import unittest

from abc import ABCMeta
from abc import abstractmethod

import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from . import RUNNING_ON_MANYLINUX
from . import PY313
from . import PY314
from .leakcheck import fails_leakcheck


# We manually manage locks in many tests
# pylint:disable=consider-using-with
# pylint:disable=too-many-public-methods
# This module is quite large.
# TODO: Refactor into separate test files. For example,
# put all the regression tests that used to produce
# crashes in test_greenlet_no_crash; put tests that DO deliberately crash
# the interpreter into test_greenlet_crash.
# pylint:disable=too-many-lines

class SomeError(Exception):
    pass


def fmain(seen):
    try:
        greenlet.getcurrent().parent.switch()
    except:
        seen.append(sys.exc_info()[0])
        raise
    raise SomeError


def send_exception(g, exc):
    # note: send_exception(g, exc)  can be now done with  g.throw(exc).
    # the purpose of this test is to explicitly check the propagation rules.
    def crasher(exc):
        raise exc
    g1 = RawGreenlet(crasher, parent=g)
    g1.switch(exc)


class TestGreenlet(TestCase):

    def _do_simple_test(self):
        lst = []

        def f():
            lst.append(1)
            greenlet.getcurrent().parent.switch()
            lst.append(3)
        g = RawGreenlet(f)
        lst.append(0)
        g.switch()
        lst.append(2)
        g.switch()
        lst.append(4)
        self.assertEqual(lst, list(range(5)))

    def test_simple(self):
        self._do_simple_test()

    def test_switch_no_run_raises_AttributeError(self):
        g = RawGreenlet()
        with self.assertRaises(AttributeError) as exc:
            g.switch()

        self.assertIn("run", str(exc.exception))

    def test_throw_no_run_raises_AttributeError(self):
        g = RawGreenlet()
        with self.assertRaises(AttributeError) as exc:
            g.throw(SomeError)

        self.assertIn("run", str(exc.exception))

    def test_parent_equals_None(self):
        g = RawGreenlet(parent=None)
        self.assertIsNotNone(g)
        self.assertIs(g.parent, greenlet.getcurrent())

    def test_run_equals_None(self):
        g = RawGreenlet(run=None)
        self.assertIsNotNone(g)
        self.assertIsNone(g.run)

    def test_two_children(self):
        lst = []

        def f():
            lst.append(1)
            greenlet.getcurrent().parent.switch()
            lst.extend([1, 1])
        g = RawGreenlet(f)
        h = RawGreenlet(f)
        g.switch()
        self.assertEqual(len(lst), 1)
        h.switch()
        self.assertEqual(len(lst), 2)
        h.switch()
        self.assertEqual(len(lst), 4)
        self.assertEqual(h.dead, True)
        g.switch()
        self.assertEqual(len(lst), 6)
        self.assertEqual(g.dead, True)

    def test_two_recursive_children(self):
        lst = []

        def f():
            lst.append('b')
            greenlet.getcurrent().parent.switch()

        def g():
            lst.append('a')
            g = RawGreenlet(f)
            g.switch()
            lst.append('c')
        self.assertEqual(sys.getrefcount(g), 2 if not PY314 else 1)
        g = RawGreenlet(g)
        # Python 3.14 elides reference counting operations
        # in some cases. See https://github.com/python/cpython/pull/130708
        self.assertEqual(sys.getrefcount(g), 2 if not PY314 else 1)
        g.switch()
        self.assertEqual(lst, ['a', 'b', 'c'])
        # Just the one in this frame, plus the one on the stack we pass to the function
        self.assertEqual(sys.getrefcount(g), 2 if not PY314 else 1)

    def test_threads(self):
        success = []

        def f():
            self._do_simple_test()
            success.append(True)
        ths = [threading.Thread(target=f) for i in range(10)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(10)
        self.assertEqual(len(success), len(ths))

    def test_exception(self):
        seen = []
        g1 = RawGreenlet(fmain)
        g2 = RawGreenlet(fmain)
        g1.switch(seen)
        g2.switch(seen)
        g2.parent = g1

        self.assertEqual(seen, [])
        #with self.assertRaises(SomeError):
        #    p("***Switching back")
        #    g2.switch()
        # Creating this as a bound method can reveal bugs that
        # are hidden on newer versions of Python that avoid creating
        # bound methods for direct expressions; IOW, don't use the `with`
        # form!
        self.assertRaises(SomeError, g2.switch)
        self.assertEqual(seen, [SomeError])

        value = g2.switch()
        self.assertEqual(value, ())
        self.assertEqual(seen, [SomeError])

        value = g2.switch(25)
        self.assertEqual(value, 25)
        self.assertEqual(seen, [SomeError])


    def test_send_exception(self):
        seen = []
        g1 = RawGreenlet(fmain)
        g1.switch(seen)
        self.assertRaises(KeyError, send_exception, g1, KeyError)
        self.assertEqual(seen, [KeyError])

    def test_dealloc(self):
        seen = []
        g1 = RawGreenlet(fmain)
        g2 = RawGreenlet(fmain)
        g1.switch(seen)
        g2.switch(seen)
        self.assertEqual(seen, [])
        del g1
        gc.collect()
        self.assertEqual(seen, [greenlet.GreenletExit])
        del g2
        gc.collect()
        self.assertEqual(seen, [greenlet.GreenletExit, greenlet.GreenletExit])

    def test_dealloc_catches_GreenletExit_throws_other(self):
        def run():
            try:
                greenlet.getcurrent().parent.switch()
            except greenlet.GreenletExit:
                raise SomeError from None

        g = RawGreenlet(run)
        g.switch()
        # Destroying the only reference to the greenlet causes it
        # to get GreenletExit; when it in turn raises, even though we're the parent
        # we don't get the exception, it just gets printed.
        # When we run on 3.8 only, we can use sys.unraisablehook
        oldstderr = sys.stderr
        from io import StringIO
        stderr = sys.stderr = StringIO()
        try:
            del g
        finally:
            sys.stderr = oldstderr

        v = stderr.getvalue()
        self.assertIn("Exception", v)
        self.assertIn('ignored', v)
        self.assertIn("SomeError", v)


    @unittest.skipIf(
        PY313 and RUNNING_ON_MANYLINUX,
        "Sometimes flaky (getting one GreenletExit in the second list)"
        # Probably due to funky timing interactions?
        # TODO: FIXME Make that work.
    )

    def test_dealloc_other_thread(self):
        seen = []
        someref = []

        bg_glet_created_running_and_no_longer_ref_in_bg = threading.Event()
        fg_ref_released = threading.Event()
        bg_should_be_clear = threading.Event()
        ok_to_exit_bg_thread = threading.Event()

        def f():
            g1 = RawGreenlet(fmain)
            g1.switch(seen)
            someref.append(g1)
            del g1
            gc.collect()

            bg_glet_created_running_and_no_longer_ref_in_bg.set()
            fg_ref_released.wait(3)

            RawGreenlet()   # trigger release
            bg_should_be_clear.set()
            ok_to_exit_bg_thread.wait(3)
            RawGreenlet() # One more time

        t = threading.Thread(target=f)
        t.start()
        bg_glet_created_running_and_no_longer_ref_in_bg.wait(10)

        self.assertEqual(seen, [])
        self.assertEqual(len(someref), 1)
        del someref[:]
        gc.collect()
        # g1 is not released immediately because it's from another thread
        self.assertEqual(seen, [])
        fg_ref_released.set()
        bg_should_be_clear.wait(3)
        try:
            self.assertEqual(seen, [greenlet.GreenletExit])
        finally:
            ok_to_exit_bg_thread.set()
            t.join(10)
            del seen[:]
            del someref[:]

    def test_frame(self):
        def f1():
            f = sys._getframe(0) # pylint:disable=protected-access
            self.assertEqual(f.f_back, None)
            greenlet.getcurrent().parent.switch(f)
            return "meaning of life"
        g = RawGreenlet(f1)
        frame = g.switch()
        self.assertTrue(frame is g.gr_frame)
        self.assertTrue(g)

        from_g = g.switch()
        self.assertFalse(g)
        self.assertEqual(from_g, 'meaning of life')
        self.assertEqual(g.gr_frame, None)

    def test_thread_bug(self):
        def runner(x):
            g = RawGreenlet(lambda: time.sleep(x))
            g.switch()
        t1 = threading.Thread(target=runner, args=(0.2,))
        t2 = threading.Thread(target=runner, args=(0.3,))
        t1.start()
        t2.start()
        t1.join(10)
        t2.join(10)

    def test_switch_kwargs(self):
        def run(a, b):
            self.assertEqual(a, 4)
            self.assertEqual(b, 2)
            return 42
        x = RawGreenlet(run).switch(a=4, b=2)
        self.assertEqual(x, 42)

    def test_switch_kwargs_to_parent(self):
        def run(x):
            greenlet.getcurrent().parent.switch(x=x)
            greenlet.getcurrent().parent.switch(2, x=3)
            return x, x ** 2
        g = RawGreenlet(run)
        self.assertEqual({'x': 3}, g.switch(3))
        self.assertEqual(((2,), {'x': 3}), g.switch())
        self.assertEqual((3, 9), g.switch())

    def test_switch_to_another_thread(self):
        data = {}
        created_event = threading.Event()
        done_event = threading.Event()

        def run():
            data['g'] = RawGreenlet(lambda: None)
            created_event.set()
            done_event.wait(10)
        thread = threading.Thread(target=run)
        thread.start()
        created_event.wait(10)
        with self.assertRaises(greenlet.error):
            data['g'].switch()
        done_event.set()
        thread.join(10)
        # XXX: Should handle this automatically
        data.clear()

    def test_exc_state(self):
        def f():
            try:
                raise ValueError('fun')
            except: # pylint:disable=bare-except
                exc_info = sys.exc_info()
                RawGreenlet(h).switch()
                self.assertEqual(exc_info, sys.exc_info())

        def h():
            self.assertEqual(sys.exc_info(), (None, None, None))

        RawGreenlet(f).switch()

    def test_instance_dict(self):
        def f():
            greenlet.getcurrent().test = 42
        def deldict(g):
            del g.__dict__
        def setdict(g, value):
            g.__dict__ = value
        g = RawGreenlet(f)
        self.assertEqual(g.__dict__, {})
        g.switch()
        self.assertEqual(g.test, 42)
        self.assertEqual(g.__dict__, {'test': 42})
        g.__dict__ = g.__dict__
        self.assertEqual(g.__dict__, {'test': 42})
        self.assertRaises(TypeError, deldict, g)
        self.assertRaises(TypeError, setdict, g, 42)

    def test_running_greenlet_has_no_run(self):
        has_run = []
        def func():
            has_run.append(
                hasattr(greenlet.getcurrent(), 'run')
            )

        g = RawGreenlet(func)
        g.switch()
        self.assertEqual(has_run, [False])

    def test_deepcopy(self):
        import copy
        self.assertRaises(TypeError, copy.copy, RawGreenlet())
        self.assertRaises(TypeError, copy.deepcopy, RawGreenlet())

    def test_parent_restored_on_kill(self):
        hub = RawGreenlet(lambda: None)
        main = greenlet.getcurrent()
        result = []
        def worker():
            try:
                # Wait to be killed by going back to the test.
                main.switch()
            except greenlet.GreenletExit:
                # Resurrect and switch to parent
                result.append(greenlet.getcurrent().parent)
                result.append(greenlet.getcurrent())
                hub.switch()
        g = RawGreenlet(worker, parent=hub)
        g.switch()
        # delete the only reference, thereby raising GreenletExit
        del g
        self.assertTrue(result)
        self.assertIs(result[0], main)
        self.assertIs(result[1].parent, hub)
        # Delete them, thereby breaking the cycle between the greenlet
        # and the frame, which otherwise would never be collectable
        # XXX: We should be able to automatically fix this.
        del result[:]
        hub = None
        main = None

    def test_parent_return_failure(self):
        # No run causes AttributeError on switch
        g1 = RawGreenlet()
        # Greenlet that implicitly switches to parent
        g2 = RawGreenlet(lambda: None, parent=g1)
        # AttributeError should propagate to us, no fatal errors
        with self.assertRaises(AttributeError):
            g2.switch()

    def test_throw_exception_not_lost(self):
        class mygreenlet(RawGreenlet):
            def __getattribute__(self, name):
                try:
                    raise Exception # pylint:disable=broad-exception-raised
                except: # pylint:disable=bare-except
                    pass
                return RawGreenlet.__getattribute__(self, name)
        g = mygreenlet(lambda: None)
        self.assertRaises(SomeError, g.throw, SomeError())

    @fails_leakcheck
    def _do_test_throw_to_dead_thread_doesnt_crash(self, wait_for_cleanup=False):
        result = []
        def worker():
            greenlet.getcurrent().parent.switch()

        def creator():
            g = RawGreenlet(worker)
            g.switch()
            result.append(g)
            if wait_for_cleanup:
                # Let this greenlet eventually be cleaned up.
                g.switch()
                greenlet.getcurrent()
        t = threading.Thread(target=creator)
        t.start()
        t.join(10)
        del t
        # But, depending on the operating system, the thread
        # deallocator may not actually have run yet! So we can't be
        # sure about the error message unless we wait.
        if wait_for_cleanup:
            self.wait_for_pending_cleanups()
        with self.assertRaises(greenlet.error) as exc:
            result[0].throw(SomeError)

        if not wait_for_cleanup:
            s = str(exc.exception)
            self.assertTrue(
                s == "cannot switch to a different thread (which happens to have exited)"
                or 'Cannot switch' in s
            )
        else:
            self.assertEqual(
                str(exc.exception),
                "cannot switch to a different thread (which happens to have exited)",
            )

        if hasattr(result[0].gr_frame, 'clear'):
            # The frame is actually executing (it thinks), we can't clear it.
            with self.assertRaises(RuntimeError):
                result[0].gr_frame.clear()
        # Unfortunately, this doesn't actually clear the references, they're in the
        # fast local array.
        if not wait_for_cleanup:
            # f_locals has no clear method in Python 3.13
            if hasattr(result[0].gr_frame.f_locals, 'clear'):
                result[0].gr_frame.f_locals.clear()
        else:
            self.assertIsNone(result[0].gr_frame)

        del creator
        worker = None
        del result[:]
        # XXX: we ought to be able to automatically fix this.
        # See issue 252
        self.expect_greenlet_leak = True # direct us not to wait for it to go away

    @fails_leakcheck
    def test_throw_to_dead_thread_doesnt_crash(self):
        self._do_test_throw_to_dead_thread_doesnt_crash()

    def test_throw_to_dead_thread_doesnt_crash_wait(self):
        self._do_test_throw_to_dead_thread_doesnt_crash(True)

    @fails_leakcheck
    def test_recursive_startup(self):
        class convoluted(RawGreenlet):
            def __init__(self):
                RawGreenlet.__init__(self)
                self.count = 0
            def __getattribute__(self, name):
                if name == 'run' and self.count == 0:
                    self.count = 1
                    self.switch(43)
                return RawGreenlet.__getattribute__(self, name)
            def run(self, value):
                while True:
                    self.parent.switch(value)
        g = convoluted()
        self.assertEqual(g.switch(42), 43)
        # Exits the running greenlet, otherwise it leaks
        # XXX: We should be able to automatically fix this
        #g.throw(greenlet.GreenletExit)
        #del g
        self.expect_greenlet_leak = True

    def test_threaded_updatecurrent(self):
        # released when main thread should execute
        lock1 = threading.Lock()
        lock1.acquire()
        # released when another thread should execute
        lock2 = threading.Lock()
        lock2.acquire()
        class finalized(object):
            def __del__(self):
                # happens while in green_updatecurrent() in main greenlet
                # should be very careful not to accidentally call it again
                # at the same time we must make sure another thread executes
                lock2.release()
                lock1.acquire()
                # now ts_current belongs to another thread
        def deallocator():
            greenlet.getcurrent().parent.switch()
        def fthread():
            lock2.acquire()
            greenlet.getcurrent()
            del g[0]
            lock1.release()
            lock2.acquire()
            greenlet.getcurrent()
            lock1.release()
        main = greenlet.getcurrent()
        g = [RawGreenlet(deallocator)]
        g[0].bomb = finalized()
        g[0].switch()
        t = threading.Thread(target=fthread)
        t.start()
        # let another thread grab ts_current and deallocate g[0]
        lock2.release()
        lock1.acquire()
        # this is the corner stone
        # getcurrent() will notice that ts_current belongs to another thread
        # and start the update process, which would notice that g[0] should
        # be deallocated, and that will execute an object's finalizer. Now,
        # that object will let another thread run so it can grab ts_current
        # again, which would likely crash the interpreter if there's no
        # check for this case at the end of green_updatecurrent(). This test
        # passes if getcurrent() returns correct result, but it's likely
        # to randomly crash if it's not anyway.
        self.assertEqual(greenlet.getcurrent(), main)
        # wait for another thread to complete, just in case
        t.join(10)

    def test_dealloc_switch_args_not_lost(self):
        seen = []
        def worker():
            # wait for the value
            value = greenlet.getcurrent().parent.switch()
            # delete all references to ourself
            del worker[0]
            initiator.parent = greenlet.getcurrent().parent
            # switch to main with the value, but because
            # ts_current is the last reference to us we
            # return here immediately, where we resurrect ourself.
            try:
                greenlet.getcurrent().parent.switch(value)
            finally:
                seen.append(greenlet.getcurrent())
        def initiator():
            return 42 # implicitly falls thru to parent

        worker = [RawGreenlet(worker)]

        worker[0].switch() # prime worker
        initiator = RawGreenlet(initiator, worker[0])
        value = initiator.switch()
        self.assertTrue(seen)
        self.assertEqual(value, 42)

    def test_tuple_subclass(self):
        # The point of this test is to see what happens when a custom
        # tuple subclass is used as an object passed directly to the C
        # function ``green_switch``; part of ``green_switch`` checks
        # the ``len()`` of the ``args`` tuple, and that can call back
        # into Python. Here, when it calls back into Python, we
        # recursively enter ``green_switch`` again.

        # This test is really only relevant on Python 2. The builtin
        # `apply` function directly passes the given args tuple object
        # to the underlying function, whereas the Python 3 version
        # unpacks and repacks into an actual tuple. This could still
        # happen using the C API on Python 3 though. We should write a
        # builtin version of apply() ourself.
        def _apply(func, a, k):
            func(*a, **k)

        class mytuple(tuple):
            def __len__(self):
                greenlet.getcurrent().switch()
                return tuple.__len__(self)
        args = mytuple()
        kwargs = dict(a=42)
        def switchapply():
            _apply(greenlet.getcurrent().parent.switch, args, kwargs)
        g = RawGreenlet(switchapply)
        self.assertEqual(g.switch(), kwargs)

    def test_abstract_subclasses(self):
        AbstractSubclass = ABCMeta(
            'AbstractSubclass',
            (RawGreenlet,),
            {'run': abstractmethod(lambda self: None)})

        class BadSubclass(AbstractSubclass):
            pass

        class GoodSubclass(AbstractSubclass):
            def run(self):
                pass

        GoodSubclass() # should not raise
        self.assertRaises(TypeError, BadSubclass)

    def test_implicit_parent_with_threads(self):
        if not gc.isenabled():
            return # cannot test with disabled gc
        N = gc.get_threshold()[0]
        if N < 50:
            return # cannot test with such a small N
        def attempt():
            lock1 = threading.Lock()
            lock1.acquire()
            lock2 = threading.Lock()
            lock2.acquire()
            recycled = [False]
            def another_thread():
                lock1.acquire() # wait for gc
                greenlet.getcurrent() # update ts_current
                lock2.release() # release gc
            t = threading.Thread(target=another_thread)
            t.start()
            class gc_callback(object):
                def __del__(self):
                    lock1.release()
                    lock2.acquire()
                    recycled[0] = True
            class garbage(object):
                def __init__(self):
                    self.cycle = self
                    self.callback = gc_callback()
            l = []
            x = range(N*2)
            current = greenlet.getcurrent()
            g = garbage()
            for _ in x:
                g = None # lose reference to garbage
                if recycled[0]:
                    # gc callback called prematurely
                    t.join(10)
                    return False
                last = RawGreenlet()
                if recycled[0]:
                    break # yes! gc called in green_new
                l.append(last) # increase allocation counter
            else:
                # gc callback not called when expected
                gc.collect()
                if recycled[0]:
                    t.join(10)
                return False
            self.assertEqual(last.parent, current)
            for g in l:
                self.assertEqual(g.parent, current)
            return True
        for _ in range(5):
            if attempt():
                break

    def test_issue_245_reference_counting_subclass_no_threads(self):
        # https://github.com/python-greenlet/greenlet/issues/245
        # Before the fix, this crashed pretty reliably on
        # Python 3.10, at least on macOS; but much less reliably on other
        # interpreters (memory layout must have changed).
        # The threaded test crashed more reliably on more interpreters.
        from greenlet import getcurrent
        from greenlet import GreenletExit

        class Greenlet(RawGreenlet):
            pass

        initial_refs = sys.getrefcount(Greenlet)
        # This has to be an instance variable because
        # Python 2 raises a SyntaxError if we delete a local
        # variable referenced in an inner scope.
        self.glets = [] # pylint:disable=attribute-defined-outside-init

        def greenlet_main():
            try:
                getcurrent().parent.switch()
            except GreenletExit:
                self.glets.append(getcurrent())

        # Before the
        for _ in range(10):
            Greenlet(greenlet_main).switch()

        del self.glets
        self.assertEqual(sys.getrefcount(Greenlet), initial_refs)

    @unittest.skipIf(
        PY313 and RUNNING_ON_MANYLINUX,
        "The manylinux images appear to hang on this test on 3.13rc2"
        # Or perhaps I just got tired of waiting for the 450s timeout.
        # Still, it shouldn't take anywhere near that long. Does not reproduce in
        # Ubuntu images, on macOS or Windows.
    )
    def test_issue_245_reference_counting_subclass_threads(self):
        # https://github.com/python-greenlet/greenlet/issues/245
        from threading import Thread
        from threading import Event

        from greenlet import getcurrent

        class MyGreenlet(RawGreenlet):
            pass

        glets = []
        ref_cleared = Event()

        def greenlet_main():
            getcurrent().parent.switch()

        def thread_main(greenlet_running_event):
            mine = MyGreenlet(greenlet_main)
            glets.append(mine)
            # The greenlets being deleted must be active
            mine.switch()
            # Don't keep any reference to it in this thread
            del mine
            # Let main know we published our greenlet.
            greenlet_running_event.set()
            # Wait for main to let us know the references are
            # gone and the greenlet objects no longer reachable
            ref_cleared.wait(10)
            # The creating thread must call getcurrent() (or a few other
            # greenlet APIs) because that's when the thread-local list of dead
            # greenlets gets cleared.
            getcurrent()

        # We start with 3 references to the subclass:
        # - This module
        # - Its __mro__
        # - The __subclassess__ attribute of greenlet
        # - (If we call gc.get_referents(), we find four entries, including
        #   some other tuple ``(greenlet)`` that I'm not sure about but must be part
        #   of the machinery.)
        #
        # On Python 3.10 it's often enough to just run 3 threads; on Python 2.7,
        # more threads are needed, and the results are still
        # non-deterministic. Presumably the memory layouts are different
        initial_refs = sys.getrefcount(MyGreenlet)
        thread_ready_events = []
        for _ in range(
                initial_refs + 45
        ):
            event = Event()
            thread = Thread(target=thread_main, args=(event,))
            thread_ready_events.append(event)
            thread.start()


        for done_event in thread_ready_events:
            done_event.wait(10)


        del glets[:]
        ref_cleared.set()
        # Let any other thread run; it will crash the interpreter
        # if not fixed (or silently corrupt memory and we possibly crash
        # later).
        self.wait_for_pending_cleanups()
        self.assertEqual(sys.getrefcount(MyGreenlet), initial_refs)

    def test_falling_off_end_switches_to_unstarted_parent_raises_error(self):
        def no_args():
            return 13

        parent_never_started = RawGreenlet(no_args)

        def leaf():
            return 42

        child = RawGreenlet(leaf, parent_never_started)

        # Because the run function takes to arguments
        with self.assertRaises(TypeError):
            child.switch()

    def test_falling_off_end_switches_to_unstarted_parent_works(self):
        def one_arg(x):
            return (x, 24)

        parent_never_started = RawGreenlet(one_arg)

        def leaf():
            return 42

        child = RawGreenlet(leaf, parent_never_started)

        result = child.switch()
        self.assertEqual(result, (42, 24))

    def test_switch_to_dead_greenlet_with_unstarted_perverse_parent(self):
        class Parent(RawGreenlet):
            def __getattribute__(self, name):
                if name == 'run':
                    raise SomeError


        parent_never_started = Parent()
        seen = []
        child = RawGreenlet(lambda: seen.append(42), parent_never_started)
        # Because we automatically start the parent when the child is
        # finished
        with self.assertRaises(SomeError):
            child.switch()

        self.assertEqual(seen, [42])

        with self.assertRaises(SomeError):
            child.switch()
        self.assertEqual(seen, [42])

    def test_switch_to_dead_greenlet_reparent(self):
        seen = []
        parent_never_started = RawGreenlet(lambda: seen.append(24))
        child = RawGreenlet(lambda: seen.append(42))

        child.switch()
        self.assertEqual(seen, [42])

        child.parent = parent_never_started
        # This actually is the same as switching to the parent.
        result = child.switch()
        self.assertIsNone(result)
        self.assertEqual(seen, [42, 24])

    def test_can_access_f_back_of_suspended_greenlet(self):
        # This tests our frame rewriting to work around Python 3.12+ having
        # some interpreter frames on the C stack. It will crash in the absence
        # of that logic.
        main = greenlet.getcurrent()

        def outer():
            inner()

        def inner():
            main.switch(sys._getframe(0))

        hub = RawGreenlet(outer)
        # start it
        hub.switch()

        # start another greenlet to make sure we aren't relying on
        # anything in `hub` still being on the C stack
        unrelated = RawGreenlet(lambda: None)
        unrelated.switch()

        # now it is suspended
        self.assertIsNotNone(hub.gr_frame)
        self.assertEqual(hub.gr_frame.f_code.co_name, "inner")
        self.assertIsNotNone(hub.gr_frame.f_back)
        self.assertEqual(hub.gr_frame.f_back.f_code.co_name, "outer")
        # The next line is what would crash
        self.assertIsNone(hub.gr_frame.f_back.f_back)

    def test_get_stack_with_nested_c_calls(self):
        from functools import partial
        from . import _test_extension_cpp

        def recurse(v):
            if v > 0:
                return v * _test_extension_cpp.test_call(partial(recurse, v - 1))
            return greenlet.getcurrent().parent.switch()

        gr = RawGreenlet(recurse)
        gr.switch(5)
        frame = gr.gr_frame
        for i in range(5):
            self.assertEqual(frame.f_locals["v"], i)
            frame = frame.f_back
        self.assertEqual(frame.f_locals["v"], 5)
        self.assertIsNone(frame.f_back)
        self.assertEqual(gr.switch(10), 1200)  # 1200 = 5! * 10

    def test_frames_always_exposed(self):
        # On Python 3.12 this will crash if we don't set the
        # gr_frames_always_exposed attribute. More background:
        # https://github.com/python-greenlet/greenlet/issues/388
        main = greenlet.getcurrent()

        def outer():
            inner(sys._getframe(0))

        def inner(frame):
            main.switch(frame)

        gr = RawGreenlet(outer)
        frame = gr.switch()

        # Do something else to clobber the part of the C stack used by `gr`,
        # so we can't skate by on "it just happened to still be there"
        unrelated = RawGreenlet(lambda: None)
        unrelated.switch()

        self.assertEqual(frame.f_code.co_name, "outer")
        # The next line crashes on 3.12 if we haven't exposed the frames.
        self.assertIsNone(frame.f_back)


class TestGreenletSetParentErrors(TestCase):
    def test_threaded_reparent(self):
        data = {}
        created_event = threading.Event()
        done_event = threading.Event()

        def run():
            data['g'] = RawGreenlet(lambda: None)
            created_event.set()
            done_event.wait(10)

        def blank():
            greenlet.getcurrent().parent.switch()

        thread = threading.Thread(target=run)
        thread.start()
        created_event.wait(10)
        g = RawGreenlet(blank)
        g.switch()
        with self.assertRaises(ValueError) as exc:
            g.parent = data['g']
        done_event.set()
        thread.join(10)

        self.assertEqual(str(exc.exception), "parent cannot be on a different thread")

    def test_unexpected_reparenting(self):
        another = []
        def worker():
            g = RawGreenlet(lambda: None)
            another.append(g)
            g.switch()
        t = threading.Thread(target=worker)
        t.start()
        t.join(10)
        # The first time we switch (running g_initialstub(), which is
        # when we look up the run attribute) we attempt to change the
        # parent to one from another thread (which also happens to be
        # dead). ``g_initialstub()`` should detect this and raise a
        # greenlet error.
        #
        # EXCEPT: With the fix for #252, this is actually detected
        # sooner, when setting the parent itself. Prior to that fix,
        # the main greenlet from the background thread kept a valid
        # value for ``run_info``, and appeared to be a valid parent
        # until we actually started the greenlet. But now that it's
        # cleared, this test is catching whether ``green_setparent``
        # can detect the dead thread.
        #
        # Further refactoring once again changes this back to a greenlet.error
        #
        # We need to wait for the cleanup to happen, but we're
        # deliberately leaking a main greenlet here.
        self.wait_for_pending_cleanups(initial_main_greenlets=self.main_greenlets_before_test + 1)

        class convoluted(RawGreenlet):
            def __getattribute__(self, name):
                if name == 'run':
                    self.parent = another[0] # pylint:disable=attribute-defined-outside-init
                return RawGreenlet.__getattribute__(self, name)
        g = convoluted(lambda: None)
        with self.assertRaises(greenlet.error) as exc:
            g.switch()
        self.assertEqual(str(exc.exception),
                         "cannot switch to a different thread (which happens to have exited)")
        del another[:]

    def test_unexpected_reparenting_thread_running(self):
        # Like ``test_unexpected_reparenting``, except the background thread is
        # actually still alive.
        another = []
        switched_to_greenlet = threading.Event()
        keep_main_alive = threading.Event()
        def worker():
            g = RawGreenlet(lambda: None)
            another.append(g)
            g.switch()
            switched_to_greenlet.set()
            keep_main_alive.wait(10)
        class convoluted(RawGreenlet):
            def __getattribute__(self, name):
                if name == 'run':
                    self.parent = another[0] # pylint:disable=attribute-defined-outside-init
                return RawGreenlet.__getattribute__(self, name)

        t = threading.Thread(target=worker)
        t.start()

        switched_to_greenlet.wait(10)
        try:
            g = convoluted(lambda: None)

            with self.assertRaises(greenlet.error) as exc:
                g.switch()
            self.assertIn("Cannot switch to a different thread", str(exc.exception))
            self.assertIn("Expected", str(exc.exception))
            self.assertIn("Current", str(exc.exception))
        finally:
            keep_main_alive.set()
            t.join(10)
            # XXX: Should handle this automatically.
            del another[:]

    def test_cannot_delete_parent(self):
        worker = RawGreenlet(lambda: None)
        self.assertIs(worker.parent, greenlet.getcurrent())

        with self.assertRaises(AttributeError) as exc:
            del worker.parent
        self.assertEqual(str(exc.exception), "can't delete attribute")

    def test_cannot_delete_parent_of_main(self):
        with self.assertRaises(AttributeError) as exc:
            del greenlet.getcurrent().parent
        self.assertEqual(str(exc.exception), "can't delete attribute")


    def test_main_greenlet_parent_is_none(self):
        # assuming we're in a main greenlet here.
        self.assertIsNone(greenlet.getcurrent().parent)

    def test_set_parent_wrong_types(self):
        def bg():
            # Go back to main.
            greenlet.getcurrent().parent.switch()

        def check(glet):
            for p in None, 1, self, "42":
                with self.assertRaises(TypeError) as exc:
                    glet.parent = p

                self.assertEqual(
                    str(exc.exception),
                    "GreenletChecker: Expected any type of greenlet, not " + type(p).__name__)

        # First, not running
        g = RawGreenlet(bg)
        self.assertFalse(g)
        check(g)

        # Then when running.
        g.switch()
        self.assertTrue(g)
        check(g)

        # Let it finish
        g.switch()


    def test_trivial_cycle(self):
        glet = RawGreenlet(lambda: None)
        with self.assertRaises(ValueError) as exc:
            glet.parent = glet
        self.assertEqual(str(exc.exception), "cyclic parent chain")

    def test_trivial_cycle_main(self):
        # This used to produce a ValueError, but we catch it earlier than that now.
        with self.assertRaises(AttributeError) as exc:
            greenlet.getcurrent().parent = greenlet.getcurrent()
        self.assertEqual(str(exc.exception), "cannot set the parent of a main greenlet")

    def test_deeper_cycle(self):
        g1 = RawGreenlet(lambda: None)
        g2 = RawGreenlet(lambda: None)
        g3 = RawGreenlet(lambda: None)

        g1.parent = g2
        g2.parent = g3
        with self.assertRaises(ValueError) as exc:
            g3.parent = g1
        self.assertEqual(str(exc.exception), "cyclic parent chain")


class TestRepr(TestCase):

    def assertEndsWith(self, got, suffix):
        self.assertTrue(got.endswith(suffix), (got, suffix))

    def test_main_while_running(self):
        r = repr(greenlet.getcurrent())
        self.assertEndsWith(r, " current active started main>")

    def test_main_in_background(self):
        main = greenlet.getcurrent()
        def run():
            return repr(main)

        g = RawGreenlet(run)
        r = g.switch()
        self.assertEndsWith(r, ' suspended active started main>')

    def test_initial(self):
        r = repr(RawGreenlet())
        self.assertEndsWith(r, ' pending>')

    def test_main_from_other_thread(self):
        main = greenlet.getcurrent()

        class T(threading.Thread):
            original_main = thread_main = None
            main_glet = None
            def run(self):
                self.original_main = repr(main)
                self.main_glet = greenlet.getcurrent()
                self.thread_main = repr(self.main_glet)

        t = T()
        t.start()
        t.join(10)

        self.assertEndsWith(t.original_main, ' suspended active started main>')
        self.assertEndsWith(t.thread_main, ' current active started main>')
        # give the machinery time to notice the death of the thread,
        # and clean it up. Note that we don't use
        # ``expect_greenlet_leak`` or wait_for_pending_cleanups,
        # because at this point we know we have an extra greenlet
        # still reachable.
        for _ in range(3):
            time.sleep(0.001)

        # In the past, main greenlets, even from dead threads, never
        # really appear dead. We have fixed that, and we also report
        # that the thread is dead in the repr. (Do this multiple times
        # to make sure that we don't self-modify and forget our state
        # in the C++ code).
        for _ in range(3):
            self.assertTrue(t.main_glet.dead)
            r = repr(t.main_glet)
            self.assertEndsWith(r, ' (thread exited) dead>')

    def test_dead(self):
        g = RawGreenlet(lambda: None)
        g.switch()
        self.assertEndsWith(repr(g), ' dead>')
        self.assertNotIn('suspended', repr(g))
        self.assertNotIn('started', repr(g))
        self.assertNotIn('active', repr(g))

    def test_formatting_produces_native_str(self):
        # https://github.com/python-greenlet/greenlet/issues/218
        # %s formatting on Python 2 was producing unicode, not str.

        g_dead = RawGreenlet(lambda: None)
        g_not_started = RawGreenlet(lambda: None)
        g_cur = greenlet.getcurrent()

        for g in g_dead, g_not_started, g_cur:

            self.assertIsInstance(
                '%s' % (g,),
                str
            )
            self.assertIsInstance(
                '%r' % (g,),
                str,
            )


class TestMainGreenlet(TestCase):
    # Tests some implementation details, and relies on some
    # implementation details.

    def _check_current_is_main(self):
        # implementation detail
        assert 'main' in repr(greenlet.getcurrent())

        t = type(greenlet.getcurrent())
        assert 'main' not in repr(t)
        return t

    def test_main_greenlet_type_can_be_subclassed(self):
        main_type = self._check_current_is_main()
        subclass = type('subclass', (main_type,), {})
        self.assertIsNotNone(subclass)

    def test_main_greenlet_is_greenlet(self):
        self._check_current_is_main()
        self.assertIsInstance(greenlet.getcurrent(), RawGreenlet)



class TestBrokenGreenlets(TestCase):
    # Tests for things that used to, or still do, terminate the interpreter.
    # This often means doing unsavory things.

    def test_failed_to_initialstub(self):
        def func():
            raise AssertionError("Never get here")


        g = greenlet._greenlet.UnswitchableGreenlet(func)
        g.force_switch_error = True

        with self.assertRaisesRegex(SystemError,
                                    "Failed to switch stacks into a greenlet for the first time."):
            g.switch()

    def test_failed_to_switch_into_running(self):
        runs = []
        def func():
            runs.append(1)
            greenlet.getcurrent().parent.switch()
            runs.append(2)
            greenlet.getcurrent().parent.switch()
            runs.append(3) # pragma: no cover

        g = greenlet._greenlet.UnswitchableGreenlet(func)
        g.switch()
        self.assertEqual(runs, [1])
        g.switch()
        self.assertEqual(runs, [1, 2])
        g.force_switch_error = True

        with self.assertRaisesRegex(SystemError,
                                    "Failed to switch stacks into a running greenlet."):
            g.switch()

        # If we stopped here, we would fail the leakcheck, because we've left
        # the ``inner_bootstrap()`` C frame and its descendents hanging around,
        # which have a bunch of Python references. They'll never get cleaned up
        # if we don't let the greenlet finish.
        g.force_switch_error = False
        g.switch()
        self.assertEqual(runs, [1, 2, 3])

    def test_failed_to_slp_switch_into_running(self):
        ex = self.assertScriptRaises('fail_slp_switch.py')

        self.assertIn('fail_slp_switch is running', ex.output)
        self.assertIn(ex.returncode, self.get_expected_returncodes_for_aborted_process())

    def test_reentrant_switch_two_greenlets(self):
        # Before we started capturing the arguments in g_switch_finish, this could crash.
        output = self.run_script('fail_switch_two_greenlets.py')
        self.assertIn('In g1_run', output)
        self.assertIn('TRACE', output)
        self.assertIn('LEAVE TRACE', output)
        self.assertIn('Falling off end of main', output)
        self.assertIn('Falling off end of g1_run', output)
        self.assertIn('Falling off end of g2', output)

    def test_reentrant_switch_three_greenlets(self):
        # On debug builds of greenlet, this used to crash with an assertion error;
        # on non-debug versions, it ran fine (which it should not do!).
        # Now it always crashes correctly with a TypeError
        ex = self.assertScriptRaises('fail_switch_three_greenlets.py', exitcodes=(1,))

        self.assertIn('TypeError', ex.output)
        self.assertIn('positional arguments', ex.output)

    def test_reentrant_switch_three_greenlets2(self):
        # This actually passed on debug and non-debug builds. It
        # should probably have been triggering some debug assertions
        # but it didn't.
        #
        # I think the fixes for the above test also kicked in here.
        output = self.run_script('fail_switch_three_greenlets2.py')
        self.assertIn(
            "RESULTS: [('trace', 'switch'), "
            "('trace', 'switch'), ('g2 arg', 'g2 from tracefunc'), "
            "('trace', 'switch'), ('main g1', 'from g2_run'), ('trace', 'switch'), "
            "('g1 arg', 'g1 from main'), ('trace', 'switch'), ('main g2', 'from g1_run'), "
            "('trace', 'switch'), ('g1 from parent', 'g1 from main 2'), ('trace', 'switch'), "
            "('main g1.2', 'g1 done'), ('trace', 'switch'), ('g2 from parent', ()), "
            "('trace', 'switch'), ('main g2.2', 'g2 done')]",
            output
        )

    def test_reentrant_switch_GreenletAlreadyStartedInPython(self):
        output = self.run_script('fail_initialstub_already_started.py')

        self.assertIn(
            "RESULTS: ['Begin C', 'Switch to b from B.__getattribute__ in C', "
            "('Begin B', ()), '_B_run switching to main', ('main from c', 'From B'), "
            "'B.__getattribute__ back from main in C', ('Begin A', (None,)), "
            "('A dead?', True, 'B dead?', True, 'C dead?', False), "
            "'C done', ('main from c.2', None)]",
            output
        )

    def test_reentrant_switch_run_callable_has_del(self):
        output = self.run_script('fail_clearing_run_switches.py')
        self.assertIn(
             "RESULTS ["
            "('G.__getattribute__', 'run'), ('RunCallable', '__del__'), "
            "('main: g.switch()', 'from RunCallable'), ('run_func', 'enter')"
            "]",
            output
        )

if __name__ == '__main__':
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_reindex.py ===
from datetime import (
    datetime,
    timedelta,
)
import inspect

import numpy as np
import pytest

from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype


class TestReindexSetIndex:
    # Tests that check both reindex and set_index

    def test_dti_set_index_reindex_datetimeindex(self):
        # GH#6631
        df = DataFrame(np.random.default_rng(2).random(6))
        idx1 = date_range("2011/01/01", periods=6, freq="ME", tz="US/Eastern")
        idx2 = date_range("2013", periods=6, freq="YE", tz="Asia/Tokyo")

        df = df.set_index(idx1)
        tm.assert_index_equal(df.index, idx1)
        df = df.reindex(idx2)
        tm.assert_index_equal(df.index, idx2)

    def test_dti_set_index_reindex_freq_with_tz(self):
        # GH#11314 with tz
        index = date_range(
            datetime(2015, 10, 1), datetime(2015, 10, 1, 23), freq="h", tz="US/Eastern"
        )
        df = DataFrame(
            np.random.default_rng(2).standard_normal((24, 1)),
            columns=["a"],
            index=index,
        )
        new_index = date_range(
            datetime(2015, 10, 2), datetime(2015, 10, 2, 23), freq="h", tz="US/Eastern"
        )

        result = df.set_index(new_index)
        assert result.index.freq == index.freq

    def test_set_reset_index_intervalindex(self):
        df = DataFrame({"A": range(10)})
        ser = pd.cut(df.A, 5)
        df["B"] = ser
        df = df.set_index("B")

        df = df.reset_index()

    def test_setitem_reset_index_dtypes(self):
        # GH 22060
        df = DataFrame(columns=["a", "b", "c"]).astype(
            {"a": "datetime64[ns]", "b": np.int64, "c": np.float64}
        )
        df1 = df.set_index(["a"])
        df1["d"] = []
        result = df1.reset_index()
        expected = DataFrame(columns=["a", "b", "c", "d"], index=range(0)).astype(
            {"a": "datetime64[ns]", "b": np.int64, "c": np.float64, "d": np.float64}
        )
        tm.assert_frame_equal(result, expected)

        df2 = df.set_index(["a", "b"])
        df2["d"] = []
        result = df2.reset_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "timezone, year, month, day, hour",
        [["America/Chicago", 2013, 11, 3, 1], ["America/Santiago", 2021, 4, 3, 23]],
    )
    def test_reindex_timestamp_with_fold(self, timezone, year, month, day, hour):
        # see gh-40817
        test_timezone = gettz(timezone)
        transition_1 = pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=0,
            tzinfo=test_timezone,
        )
        transition_2 = pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=1,
            tzinfo=test_timezone,
        )
        df = (
            DataFrame({"index": [transition_1, transition_2], "vals": ["a", "b"]})
            .set_index("index")
            .reindex(["1", "2"])
        )
        exp = DataFrame({"index": ["1", "2"], "vals": [np.nan, np.nan]}).set_index(
            "index"
        )
        exp = exp.astype(df.vals.dtype)
        tm.assert_frame_equal(
            df,
            exp,
        )


class TestDataFrameSelectReindex:
    # These are specific reindex-based tests; other indexing tests should go in
    # test_indexing

    @pytest.mark.xfail(
        not IS64 or (is_platform_windows() and not np_version_gt2),
        reason="Passes int32 values to DatetimeArray in make_na_array on "
        "windows, 32bit linux builds",
    )
    @td.skip_array_manager_not_yet_implemented
    def test_reindex_tzaware_fill_value(self):
        # GH#52586
        df = DataFrame([[1]])

        ts = pd.Timestamp("2023-04-10 17:32", tz="US/Pacific")
        res = df.reindex([0, 1], axis=1, fill_value=ts)
        assert res.dtypes[1] == pd.DatetimeTZDtype(unit="s", tz="US/Pacific")
        expected = DataFrame({0: [1], 1: [ts]})
        expected[1] = expected[1].astype(res.dtypes[1])
        tm.assert_frame_equal(res, expected)

        per = ts.tz_localize(None).to_period("s")
        res = df.reindex([0, 1], axis=1, fill_value=per)
        assert res.dtypes[1] == pd.PeriodDtype("s")
        expected = DataFrame({0: [1], 1: [per]})
        tm.assert_frame_equal(res, expected)

        interval = pd.Interval(ts, ts + pd.Timedelta(seconds=1))
        res = df.reindex([0, 1], axis=1, fill_value=interval)
        assert res.dtypes[1] == pd.IntervalDtype("datetime64[s, US/Pacific]", "right")
        expected = DataFrame({0: [1], 1: [interval]})
        expected[1] = expected[1].astype(res.dtypes[1])
        tm.assert_frame_equal(res, expected)

    def test_reindex_copies(self):
        # based on asv time_reindex_axis1
        N = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((N * 10, N)))
        cols = np.arange(N)
        np.random.default_rng(2).shuffle(cols)

        result = df.reindex(columns=cols, copy=True)
        assert not np.shares_memory(result[0]._values, df[0]._values)

        # pass both columns and index
        result2 = df.reindex(columns=cols, index=df.index, copy=True)
        assert not np.shares_memory(result2[0]._values, df[0]._values)

    def test_reindex_copies_ea(self, using_copy_on_write):
        # https://github.com/pandas-dev/pandas/pull/51197
        # also ensure to honor copy keyword for ExtensionDtypes
        N = 10
        df = DataFrame(
            np.random.default_rng(2).standard_normal((N * 10, N)), dtype="Float64"
        )
        cols = np.arange(N)
        np.random.default_rng(2).shuffle(cols)

        result = df.reindex(columns=cols, copy=True)
        if using_copy_on_write:
            assert np.shares_memory(result[0].array._data, df[0].array._data)
        else:
            assert not np.shares_memory(result[0].array._data, df[0].array._data)

        # pass both columns and index
        result2 = df.reindex(columns=cols, index=df.index, copy=True)
        if using_copy_on_write:
            assert np.shares_memory(result2[0].array._data, df[0].array._data)
        else:
            assert not np.shares_memory(result2[0].array._data, df[0].array._data)

    @td.skip_array_manager_not_yet_implemented
    def test_reindex_date_fill_value(self):
        # passing date to dt64 is deprecated; enforced in 2.0 to cast to object
        arr = date_range("2016-01-01", periods=6).values.reshape(3, 2)
        df = DataFrame(arr, columns=["A", "B"], index=range(3))

        ts = df.iloc[0, 0]
        fv = ts.date()

        res = df.reindex(index=range(4), columns=["A", "B", "C"], fill_value=fv)

        expected = DataFrame(
            {"A": df["A"].tolist() + [fv], "B": df["B"].tolist() + [fv], "C": [fv] * 4},
            dtype=object,
        )
        tm.assert_frame_equal(res, expected)

        # only reindexing rows
        res = df.reindex(index=range(4), fill_value=fv)
        tm.assert_frame_equal(res, expected[["A", "B"]])

        # same with a datetime-castable str
        res = df.reindex(
            index=range(4), columns=["A", "B", "C"], fill_value="2016-01-01"
        )
        expected = DataFrame(
            {"A": df["A"].tolist() + [ts], "B": df["B"].tolist() + [ts], "C": [ts] * 4},
        )
        tm.assert_frame_equal(res, expected)

    def test_reindex_with_multi_index(self):
        # https://github.com/pandas-dev/pandas/issues/29896
        # tests for reindexing a multi-indexed DataFrame with a new MultiIndex
        #
        # confirms that we can reindex a multi-indexed DataFrame with a new
        # MultiIndex object correctly when using no filling, backfilling, and
        # padding
        #
        # The DataFrame, `df`, used in this test is:
        #       c
        #  a b
        # -1 0  A
        #    1  B
        #    2  C
        #    3  D
        #    4  E
        #    5  F
        #    6  G
        #  0 0  A
        #    1  B
        #    2  C
        #    3  D
        #    4  E
        #    5  F
        #    6  G
        #  1 0  A
        #    1  B
        #    2  C
        #    3  D
        #    4  E
        #    5  F
        #    6  G
        #
        # and the other MultiIndex, `new_multi_index`, is:
        # 0: 0 0.5
        # 1:   2.0
        # 2:   5.0
        # 3:   5.8
        df = DataFrame(
            {
                "a": [-1] * 7 + [0] * 7 + [1] * 7,
                "b": list(range(7)) * 3,
                "c": ["A", "B", "C", "D", "E", "F", "G"] * 3,
            }
        ).set_index(["a", "b"])
        new_index = [0.5, 2.0, 5.0, 5.8]
        new_multi_index = MultiIndex.from_product([[0], new_index], names=["a", "b"])

        # reindexing w/o a `method` value
        reindexed = df.reindex(new_multi_index)
        expected = DataFrame(
            {"a": [0] * 4, "b": new_index, "c": [np.nan, "C", "F", np.nan]}
        ).set_index(["a", "b"])
        tm.assert_frame_equal(expected, reindexed)

        # reindexing with backfilling
        expected = DataFrame(
            {"a": [0] * 4, "b": new_index, "c": ["B", "C", "F", "G"]}
        ).set_index(["a", "b"])
        reindexed_with_backfilling = df.reindex(new_multi_index, method="bfill")
        tm.assert_frame_equal(expected, reindexed_with_backfilling)

        reindexed_with_backfilling = df.reindex(new_multi_index, method="backfill")
        tm.assert_frame_equal(expected, reindexed_with_backfilling)

        # reindexing with padding
        expected = DataFrame(
            {"a": [0] * 4, "b": new_index, "c": ["A", "C", "F", "F"]}
        ).set_index(["a", "b"])
        reindexed_with_padding = df.reindex(new_multi_index, method="pad")
        tm.assert_frame_equal(expected, reindexed_with_padding)

        reindexed_with_padding = df.reindex(new_multi_index, method="ffill")
        tm.assert_frame_equal(expected, reindexed_with_padding)

    @pytest.mark.parametrize(
        "method,expected_values",
        [
            ("nearest", [0, 1, 1, 2]),
            ("pad", [np.nan, 0, 1, 1]),
            ("backfill", [0, 1, 2, 2]),
        ],
    )
    def test_reindex_methods(self, method, expected_values):
        df = DataFrame({"x": list(range(5))})
        target = np.array([-0.1, 0.9, 1.1, 1.5])

        expected = DataFrame({"x": expected_values}, index=target)
        actual = df.reindex(target, method=method)
        tm.assert_frame_equal(expected, actual)

        actual = df.reindex(target, method=method, tolerance=1)
        tm.assert_frame_equal(expected, actual)
        actual = df.reindex(target, method=method, tolerance=[1, 1, 1, 1])
        tm.assert_frame_equal(expected, actual)

        e2 = expected[::-1]
        actual = df.reindex(target[::-1], method=method)
        tm.assert_frame_equal(e2, actual)

        new_order = [3, 0, 2, 1]
        e2 = expected.iloc[new_order]
        actual = df.reindex(target[new_order], method=method)
        tm.assert_frame_equal(e2, actual)

        switched_method = (
            "pad" if method == "backfill" else "backfill" if method == "pad" else method
        )
        actual = df[::-1].reindex(target, method=switched_method)
        tm.assert_frame_equal(expected, actual)

    def test_reindex_methods_nearest_special(self):
        df = DataFrame({"x": list(range(5))})
        target = np.array([-0.1, 0.9, 1.1, 1.5])

        expected = DataFrame({"x": [0, 1, 1, np.nan]}, index=target)
        actual = df.reindex(target, method="nearest", tolerance=0.2)
        tm.assert_frame_equal(expected, actual)

        expected = DataFrame({"x": [0, np.nan, 1, np.nan]}, index=target)
        actual = df.reindex(target, method="nearest", tolerance=[0.5, 0.01, 0.4, 0.1])
        tm.assert_frame_equal(expected, actual)

    def test_reindex_nearest_tz(self, tz_aware_fixture):
        # GH26683
        tz = tz_aware_fixture
        idx = date_range("2019-01-01", periods=5, tz=tz)
        df = DataFrame({"x": list(range(5))}, index=idx)

        expected = df.head(3)
        actual = df.reindex(idx[:3], method="nearest")
        tm.assert_frame_equal(expected, actual)

    def test_reindex_nearest_tz_empty_frame(self):
        # https://github.com/pandas-dev/pandas/issues/31964
        dti = pd.DatetimeIndex(["2016-06-26 14:27:26+00:00"])
        df = DataFrame(index=pd.DatetimeIndex(["2016-07-04 14:00:59+00:00"]))
        expected = DataFrame(index=dti)
        result = df.reindex(dti, method="nearest")
        tm.assert_frame_equal(result, expected)

    def test_reindex_frame_add_nat(self):
        rng = date_range("1/1/2000 00:00:00", periods=10, freq="10s")
        df = DataFrame(
            {"A": np.random.default_rng(2).standard_normal(len(rng)), "B": rng}
        )

        result = df.reindex(range(15))
        assert np.issubdtype(result["B"].dtype, np.dtype("M8[ns]"))

        mask = isna(result)["B"]
        assert mask[-5:].all()
        assert not mask[:-5].any()

    @pytest.mark.parametrize(
        "method, exp_values",
        [("ffill", [0, 1, 2, 3]), ("bfill", [1.0, 2.0, 3.0, np.nan])],
    )
    def test_reindex_frame_tz_ffill_bfill(self, frame_or_series, method, exp_values):
        # GH#38566
        obj = frame_or_series(
            [0, 1, 2, 3],
            index=date_range("2020-01-01 00:00:00", periods=4, freq="h", tz="UTC"),
        )
        new_index = date_range("2020-01-01 00:01:00", periods=4, freq="h", tz="UTC")
        result = obj.reindex(new_index, method=method, tolerance=pd.Timedelta("1 hour"))
        expected = frame_or_series(exp_values, index=new_index)
        tm.assert_equal(result, expected)

    def test_reindex_limit(self):
        # GH 28631
        data = [["A", "A", "A"], ["B", "B", "B"], ["C", "C", "C"], ["D", "D", "D"]]
        exp_data = [
            ["A", "A", "A"],
            ["B", "B", "B"],
            ["C", "C", "C"],
            ["D", "D", "D"],
            ["D", "D", "D"],
            [np.nan, np.nan, np.nan],
        ]
        df = DataFrame(data)
        result = df.reindex([0, 1, 2, 3, 4, 5], method="ffill", limit=1)
        expected = DataFrame(exp_data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "idx, check_index_type",
        [
            [["C", "B", "A"], True],
            [["F", "C", "A", "D"], True],
            [["A"], True],
            [["A", "B", "C"], True],
            [["C", "A", "B"], True],
            [["C", "B"], True],
            [["C", "A"], True],
            [["A", "B"], True],
            [["B", "A", "C"], True],
            # reindex by these causes different MultiIndex levels
            [["D", "F"], False],
            [["A", "C", "B"], False],
        ],
    )
    def test_reindex_level_verify_first_level(self, idx, check_index_type):
        df = DataFrame(
            {
                "jim": list("B" * 4 + "A" * 2 + "C" * 3),
                "joe": list("abcdeabcd")[::-1],
                "jolie": [10, 20, 30] * 3,
                "joline": np.random.default_rng(2).integers(0, 1000, 9),
            }
        )
        icol = ["jim", "joe", "jolie"]

        def f(val):
            return np.nonzero((df["jim"] == val).to_numpy())[0]

        i = np.concatenate(list(map(f, idx)))
        left = df.set_index(icol).reindex(idx, level="jim")
        right = df.iloc[i].set_index(icol)
        tm.assert_frame_equal(left, right, check_index_type=check_index_type)

    @pytest.mark.parametrize(
        "idx",
        [
            ("mid",),
            ("mid", "btm"),
            ("mid", "btm", "top"),
            ("mid",),
            ("mid", "top"),
            ("mid", "top", "btm"),
            ("btm",),
            ("btm", "mid"),
            ("btm", "mid", "top"),
            ("btm",),
            ("btm", "top"),
            ("btm", "top", "mid"),
            ("top",),
            ("top", "mid"),
            ("top", "mid", "btm"),
            ("top",),
            ("top", "btm"),
            ("top", "btm", "mid"),
        ],
    )
    def test_reindex_level_verify_first_level_repeats(self, idx):
        df = DataFrame(
            {
                "jim": ["mid"] * 5 + ["btm"] * 8 + ["top"] * 7,
                "joe": ["3rd"] * 2
                + ["1st"] * 3
                + ["2nd"] * 3
                + ["1st"] * 2
                + ["3rd"] * 3
                + ["1st"] * 2
                + ["3rd"] * 3
                + ["2nd"] * 2,
                # this needs to be jointly unique with jim and joe or
                # reindexing will fail ~1.5% of the time, this works
                # out to needing unique groups of same size as joe
                "jolie": np.concatenate(
                    [
                        np.random.default_rng(2).choice(1000, x, replace=False)
                        for x in [2, 3, 3, 2, 3, 2, 3, 2]
                    ]
                ),
                "joline": np.random.default_rng(2).standard_normal(20).round(3) * 10,
            }
        )
        icol = ["jim", "joe", "jolie"]

        def f(val):
            return np.nonzero((df["jim"] == val).to_numpy())[0]

        i = np.concatenate(list(map(f, idx)))
        left = df.set_index(icol).reindex(idx, level="jim")
        right = df.iloc[i].set_index(icol)
        tm.assert_frame_equal(left, right)

    @pytest.mark.parametrize(
        "idx, indexer",
        [
            [
                ["1st", "2nd", "3rd"],
                [2, 3, 4, 0, 1, 8, 9, 5, 6, 7, 10, 11, 12, 13, 14, 18, 19, 15, 16, 17],
            ],
            [
                ["3rd", "2nd", "1st"],
                [0, 1, 2, 3, 4, 10, 11, 12, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 13, 14],
            ],
            [["2nd", "3rd"], [0, 1, 5, 6, 7, 10, 11, 12, 18, 19, 15, 16, 17]],
            [["3rd", "1st"], [0, 1, 2, 3, 4, 10, 11, 12, 8, 9, 15, 16, 17, 13, 14]],
        ],
    )
    def test_reindex_level_verify_repeats(self, idx, indexer):
        df = DataFrame(
            {
                "jim": ["mid"] * 5 + ["btm"] * 8 + ["top"] * 7,
                "joe": ["3rd"] * 2
                + ["1st"] * 3
                + ["2nd"] * 3
                + ["1st"] * 2
                + ["3rd"] * 3
                + ["1st"] * 2
                + ["3rd"] * 3
                + ["2nd"] * 2,
                # this needs to be jointly unique with jim and joe or
                # reindexing will fail ~1.5% of the time, this works
                # out to needing unique groups of same size as joe
                "jolie": np.concatenate(
                    [
                        np.random.default_rng(2).choice(1000, x, replace=False)
                        for x in [2, 3, 3, 2, 3, 2, 3, 2]
                    ]
                ),
                "joline": np.random.default_rng(2).standard_normal(20).round(3) * 10,
            }
        )
        icol = ["jim", "joe", "jolie"]
        left = df.set_index(icol).reindex(idx, level="joe")
        right = df.iloc[indexer].set_index(icol)
        tm.assert_frame_equal(left, right)

    @pytest.mark.parametrize(
        "idx, indexer, check_index_type",
        [
            [list("abcde"), [3, 2, 1, 0, 5, 4, 8, 7, 6], True],
            [list("abcd"), [3, 2, 1, 0, 5, 8, 7, 6], True],
            [list("abc"), [3, 2, 1, 8, 7, 6], True],
            [list("eca"), [1, 3, 4, 6, 8], True],
            [list("edc"), [0, 1, 4, 5, 6], True],
            [list("eadbc"), [3, 0, 2, 1, 4, 5, 8, 7, 6], True],
            [list("edwq"), [0, 4, 5], True],
            [list("wq"), [], False],
        ],
    )
    def test_reindex_level_verify(self, idx, indexer, check_index_type):
        df = DataFrame(
            {
                "jim": list("B" * 4 + "A" * 2 + "C" * 3),
                "joe": list("abcdeabcd")[::-1],
                "jolie": [10, 20, 30] * 3,
                "joline": np.random.default_rng(2).integers(0, 1000, 9),
            }
        )
        icol = ["jim", "joe", "jolie"]
        left = df.set_index(icol).reindex(idx, level="joe")
        right = df.iloc[indexer].set_index(icol)
        tm.assert_frame_equal(left, right, check_index_type=check_index_type)

    def test_non_monotonic_reindex_methods(self):
        dr = date_range("2013-08-01", periods=6, freq="B")
        data = np.random.default_rng(2).standard_normal((6, 1))
        df = DataFrame(data, index=dr, columns=list("A"))
        df_rev = DataFrame(data, index=dr[[3, 4, 5] + [0, 1, 2]], columns=list("A"))
        # index is not monotonic increasing or decreasing
        msg = "index must be monotonic increasing or decreasing"
        with pytest.raises(ValueError, match=msg):
            df_rev.reindex(df.index, method="pad")
        with pytest.raises(ValueError, match=msg):
            df_rev.reindex(df.index, method="ffill")
        with pytest.raises(ValueError, match=msg):
            df_rev.reindex(df.index, method="bfill")
        with pytest.raises(ValueError, match=msg):
            df_rev.reindex(df.index, method="nearest")

    def test_reindex_sparse(self):
        # https://github.com/pandas-dev/pandas/issues/35286
        df = DataFrame(
            {"A": [0, 1], "B": pd.array([0, 1], dtype=pd.SparseDtype("int64", 0))}
        )
        result = df.reindex([0, 2])
        expected = DataFrame(
            {
                "A": [0.0, np.nan],
                "B": pd.array([0.0, np.nan], dtype=pd.SparseDtype("float64", 0.0)),
            },
            index=[0, 2],
        )
        tm.assert_frame_equal(result, expected)

    def test_reindex(self, float_frame, using_copy_on_write):
        datetime_series = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )

        newFrame = float_frame.reindex(datetime_series.index)

        for col in newFrame.columns:
            for idx, val in newFrame[col].items():
                if idx in float_frame.index:
                    if np.isnan(val):
                        assert np.isnan(float_frame[col][idx])
                    else:
                        assert val == float_frame[col][idx]
                else:
                    assert np.isnan(val)

        for col, series in newFrame.items():
            tm.assert_index_equal(series.index, newFrame.index)
        emptyFrame = float_frame.reindex(Index([]))
        assert len(emptyFrame.index) == 0

        # Cython code should be unit-tested directly
        nonContigFrame = float_frame.reindex(datetime_series.index[::2])

        for col in nonContigFrame.columns:
            for idx, val in nonContigFrame[col].items():
                if idx in float_frame.index:
                    if np.isnan(val):
                        assert np.isnan(float_frame[col][idx])
                    else:
                        assert val == float_frame[col][idx]
                else:
                    assert np.isnan(val)

        for col, series in nonContigFrame.items():
            tm.assert_index_equal(series.index, nonContigFrame.index)

        # corner cases

        # Same index, copies values but not index if copy=False
        newFrame = float_frame.reindex(float_frame.index, copy=False)
        if using_copy_on_write:
            assert newFrame.index.is_(float_frame.index)
        else:
            assert newFrame.index is float_frame.index

        # length zero
        newFrame = float_frame.reindex([])
        assert newFrame.empty
        assert len(newFrame.columns) == len(float_frame.columns)

        # length zero with columns reindexed with non-empty index
        newFrame = float_frame.reindex([])
        newFrame = newFrame.reindex(float_frame.index)
        assert len(newFrame.index) == len(float_frame.index)
        assert len(newFrame.columns) == len(float_frame.columns)

        # pass non-Index
        newFrame = float_frame.reindex(list(datetime_series.index))
        expected = datetime_series.index._with_freq(None)
        tm.assert_index_equal(newFrame.index, expected)

        # copy with no axes
        result = float_frame.reindex()
        tm.assert_frame_equal(result, float_frame)
        assert result is not float_frame

    def test_reindex_nan(self):
        df = DataFrame(
            [[1, 2], [3, 5], [7, 11], [9, 23]],
            index=[2, np.nan, 1, 5],
            columns=["joe", "jim"],
        )

        i, j = [np.nan, 5, 5, np.nan, 1, 2, np.nan], [1, 3, 3, 1, 2, 0, 1]
        tm.assert_frame_equal(df.reindex(i), df.iloc[j])

        df.index = df.index.astype("object")
        tm.assert_frame_equal(df.reindex(i), df.iloc[j], check_index_type=False)

        # GH10388
        df = DataFrame(
            {
                "other": ["a", "b", np.nan, "c"],
                "date": ["2015-03-22", np.nan, "2012-01-08", np.nan],
                "amount": [2, 3, 4, 5],
            }
        )

        df["date"] = pd.to_datetime(df.date)
        df["delta"] = (pd.to_datetime("2015-06-18") - df["date"]).shift(1)

        left = df.set_index(["delta", "other", "date"]).reset_index()
        right = df.reindex(columns=["delta", "other", "date", "amount"])
        tm.assert_frame_equal(left, right)

    def test_reindex_name_remains(self):
        s = Series(np.random.default_rng(2).random(10))
        df = DataFrame(s, index=np.arange(len(s)))
        i = Series(np.arange(10), name="iname")

        df = df.reindex(i)
        assert df.index.name == "iname"

        df = df.reindex(Index(np.arange(10), name="tmpname"))
        assert df.index.name == "tmpname"

        s = Series(np.random.default_rng(2).random(10))
        df = DataFrame(s.T, index=np.arange(len(s)))
        i = Series(np.arange(10), name="iname")
        df = df.reindex(columns=i)
        assert df.columns.name == "iname"

    def test_reindex_int(self, int_frame):
        smaller = int_frame.reindex(int_frame.index[::2])

        assert smaller["A"].dtype == np.int64

        bigger = smaller.reindex(int_frame.index)
        assert bigger["A"].dtype == np.float64

        smaller = int_frame.reindex(columns=["A", "B"])
        assert smaller["A"].dtype == np.int64

    def test_reindex_columns(self, float_frame):
        new_frame = float_frame.reindex(columns=["A", "B", "E"])

        tm.assert_series_equal(new_frame["B"], float_frame["B"])
        assert np.isnan(new_frame["E"]).all()
        assert "C" not in new_frame

        # Length zero
        new_frame = float_frame.reindex(columns=[])
        assert new_frame.empty

    def test_reindex_columns_method(self):
        # GH 14992, reindexing over columns ignored method
        df = DataFrame(
            data=[[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            index=[1, 2, 4],
            columns=[1, 2, 4],
            dtype=float,
        )

        # default method
        result = df.reindex(columns=range(6))
        expected = DataFrame(
            data=[
                [np.nan, 11, 12, np.nan, 13, np.nan],
                [np.nan, 21, 22, np.nan, 23, np.nan],
                [np.nan, 31, 32, np.nan, 33, np.nan],
            ],
            index=[1, 2, 4],
            columns=range(6),
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)

        # method='ffill'
        result = df.reindex(columns=range(6), method="ffill")
        expected = DataFrame(
            data=[
                [np.nan, 11, 12, 12, 13, 13],
                [np.nan, 21, 22, 22, 23, 23],
                [np.nan, 31, 32, 32, 33, 33],
            ],
            index=[1, 2, 4],
            columns=range(6),
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)

        # method='bfill'
        result = df.reindex(columns=range(6), method="bfill")
        expected = DataFrame(
            data=[
                [11, 11, 12, 13, 13, np.nan],
                [21, 21, 22, 23, 23, np.nan],
                [31, 31, 32, 33, 33, np.nan],
            ],
            index=[1, 2, 4],
            columns=range(6),
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)

    def test_reindex_axes(self):
        # GH 3317, reindexing by both axes loses freq of the index
        df = DataFrame(
            np.ones((3, 3)),
            index=[datetime(2012, 1, 1), datetime(2012, 1, 2), datetime(2012, 1, 3)],
            columns=["a", "b", "c"],
        )
        time_freq = date_range("2012-01-01", "2012-01-03", freq="d")
        some_cols = ["a", "b"]

        index_freq = df.reindex(index=time_freq).index.freq
        both_freq = df.reindex(index=time_freq, columns=some_cols).index.freq
        seq_freq = df.reindex(index=time_freq).reindex(columns=some_cols).index.freq
        assert index_freq == both_freq
        assert index_freq == seq_freq

    def test_reindex_fill_value(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        # axis=0
        result = df.reindex(list(range(15)))
        assert np.isnan(result.values[-5:]).all()

        result = df.reindex(range(15), fill_value=0)
        expected = df.reindex(range(15)).fillna(0)
        tm.assert_frame_equal(result, expected)

        # axis=1
        result = df.reindex(columns=range(5), fill_value=0.0)
        expected = df.copy()
        expected[4] = 0.0
        tm.assert_frame_equal(result, expected)

        result = df.reindex(columns=range(5), fill_value=0)
        expected = df.copy()
        expected[4] = 0
        tm.assert_frame_equal(result, expected)

        result = df.reindex(columns=range(5), fill_value="foo")
        expected = df.copy()
        expected[4] = "foo"
        tm.assert_frame_equal(result, expected)

        # other dtypes
        df["foo"] = "foo"
        result = df.reindex(range(15), fill_value="0")
        expected = df.reindex(range(15)).fillna("0")
        tm.assert_frame_equal(result, expected)

    def test_reindex_uint_dtypes_fill_value(self, any_unsigned_int_numpy_dtype):
        # GH#48184
        df = DataFrame({"a": [1, 2], "b": [1, 2]}, dtype=any_unsigned_int_numpy_dtype)
        result = df.reindex(columns=list("abcd"), index=[0, 1, 2, 3], fill_value=10)
        expected = DataFrame(
            {"a": [1, 2, 10, 10], "b": [1, 2, 10, 10], "c": 10, "d": 10},
            dtype=any_unsigned_int_numpy_dtype,
        )
        tm.assert_frame_equal(result, expected)

    def test_reindex_single_column_ea_index_and_columns(self, any_numeric_ea_dtype):
        # GH#48190
        df = DataFrame({"a": [1, 2]}, dtype=any_numeric_ea_dtype)
        result = df.reindex(columns=list("ab"), index=[0, 1, 2], fill_value=10)
        expected = DataFrame(
            {"a": Series([1, 2, 10], dtype=any_numeric_ea_dtype), "b": 10}
        )
        tm.assert_frame_equal(result, expected)

    def test_reindex_dups(self):
        # GH4746, reindex on duplicate index error messages
        arr = np.random.default_rng(2).standard_normal(10)
        df = DataFrame(arr, index=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

        # set index is ok
        result = df.copy()
        result.index = list(range(len(df)))
        expected = DataFrame(arr, index=list(range(len(df))))
        tm.assert_frame_equal(result, expected)

        # reindex fails
        msg = "cannot reindex on an axis with duplicate labels"
        with pytest.raises(ValueError, match=msg):
            df.reindex(index=list(range(len(df))))

    def test_reindex_with_duplicate_columns(self):
        # reindex is invalid!
        df = DataFrame(
            [[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=["bar", "a", "a"]
        )
        msg = "cannot reindex on an axis with duplicate labels"
        with pytest.raises(ValueError, match=msg):
            df.reindex(columns=["bar"])
        with pytest.raises(ValueError, match=msg):
            df.reindex(columns=["bar", "foo"])

    def test_reindex_axis_style(self):
        # https://github.com/pandas-dev/pandas/issues/12392
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        expected = DataFrame(
            {"A": [1, 2, np.nan], "B": [4, 5, np.nan]}, index=[0, 1, 3]
        )
        result = df.reindex([0, 1, 3])
        tm.assert_frame_equal(result, expected)

        result = df.reindex([0, 1, 3], axis=0)
        tm.assert_frame_equal(result, expected)

        result = df.reindex([0, 1, 3], axis="index")
        tm.assert_frame_equal(result, expected)

    def test_reindex_positional_raises(self):
        # https://github.com/pandas-dev/pandas/issues/12392
        # Enforced in 2.0
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        msg = r"reindex\(\) takes from 1 to 2 positional arguments but 3 were given"
        with pytest.raises(TypeError, match=msg):
            df.reindex([0, 1], ["A", "B", "C"])

    def test_reindex_axis_style_raises(self):
        # https://github.com/pandas-dev/pandas/issues/12392
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex([0, 1], columns=["A"], axis=1)

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex([0, 1], columns=["A"], axis="index")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis="index")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis="columns")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(columns=[0, 1], axis="columns")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], columns=[0, 1], axis="columns")

        with pytest.raises(TypeError, match="Cannot specify all"):
            df.reindex(labels=[0, 1], index=[0], columns=["A"])

        # Mixing styles
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis="index")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis="columns")

        # Duplicates
        with pytest.raises(TypeError, match="multiple values"):
            df.reindex([0, 1], labels=[0, 1])

    def test_reindex_single_named_indexer(self):
        # https://github.com/pandas-dev/pandas/issues/12392
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})
        result = df.reindex([0, 1], columns=["A"])
        expected = DataFrame({"A": [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_reindex_api_equivalence(self):
        # https://github.com/pandas-dev/pandas/issues/12392
        # equivalence of the labels/axis and index/columns API's
        df = DataFrame(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            index=["a", "b", "c"],
            columns=["d", "e", "f"],
        )

        res1 = df.reindex(["b", "a"])
        res2 = df.reindex(index=["b", "a"])
        res3 = df.reindex(labels=["b", "a"])
        res4 = df.reindex(labels=["b", "a"], axis=0)
        res5 = df.reindex(["b", "a"], axis=0)
        for res in [res2, res3, res4, res5]:
            tm.assert_frame_equal(res1, res)

        res1 = df.reindex(columns=["e", "d"])
        res2 = df.reindex(["e", "d"], axis=1)
        res3 = df.reindex(labels=["e", "d"], axis=1)
        for res in [res2, res3]:
            tm.assert_frame_equal(res1, res)

        res1 = df.reindex(index=["b", "a"], columns=["e", "d"])
        res2 = df.reindex(columns=["e", "d"], index=["b", "a"])
        res3 = df.reindex(labels=["b", "a"], axis=0).reindex(labels=["e", "d"], axis=1)
        for res in [res2, res3]:
            tm.assert_frame_equal(res1, res)

    def test_reindex_boolean(self):
        frame = DataFrame(
            np.ones((10, 2), dtype=bool), index=np.arange(0, 20, 2), columns=[0, 2]
        )

        reindexed = frame.reindex(np.arange(10))
        assert reindexed.values.dtype == np.object_
        assert isna(reindexed[0][1])

        reindexed = frame.reindex(columns=range(3))
        assert reindexed.values.dtype == np.object_
        assert isna(reindexed[1]).all()

    def test_reindex_objects(self, float_string_frame):
        reindexed = float_string_frame.reindex(columns=["foo", "A", "B"])
        assert "foo" in reindexed

        reindexed = float_string_frame.reindex(columns=["A", "B"])
        assert "foo" not in reindexed

    def test_reindex_corner(self, int_frame):
        index = Index(["a", "b", "c"])
        dm = DataFrame({}).reindex(index=[1, 2, 3])
        reindexed = dm.reindex(columns=index)
        tm.assert_index_equal(reindexed.columns, index)

        # ints are weird
        smaller = int_frame.reindex(columns=["A", "B", "E"])
        assert smaller["E"].dtype == np.float64

    def test_reindex_with_nans(self):
        df = DataFrame(
            [[1, 2], [3, 4], [np.nan, np.nan], [7, 8], [9, 10]],
            columns=["a", "b"],
            index=[100.0, 101.0, np.nan, 102.0, 103.0],
        )

        result = df.reindex(index=[101.0, 102.0, 103.0])
        expected = df.iloc[[1, 3, 4]]
        tm.assert_frame_equal(result, expected)

        result = df.reindex(index=[103.0])
        expected = df.iloc[[4]]
        tm.assert_frame_equal(result, expected)

        result = df.reindex(index=[101.0])
        expected = df.iloc[[1]]
        tm.assert_frame_equal(result, expected)

    def test_reindex_multi(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))

        result = df.reindex(index=range(4), columns=range(4))
        expected = df.reindex(list(range(4))).reindex(columns=range(4))

        tm.assert_frame_equal(result, expected)

        df = DataFrame(np.random.default_rng(2).integers(0, 10, (3, 3)))

        result = df.reindex(index=range(4), columns=range(4))
        expected = df.reindex(list(range(4))).reindex(columns=range(4))

        tm.assert_frame_equal(result, expected)

        df = DataFrame(np.random.default_rng(2).integers(0, 10, (3, 3)))

        result = df.reindex(index=range(2), columns=range(2))
        expected = df.reindex(range(2)).reindex(columns=range(2))

        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)) + 1j,
            columns=["a", "b", "c"],
        )

        result = df.reindex(index=[0, 1], columns=["a", "b"])
        expected = df.reindex([0, 1]).reindex(columns=["a", "b"])

        tm.assert_frame_equal(result, expected)

    def test_reindex_multi_categorical_time(self):
        # https://github.com/pandas-dev/pandas/issues/21390
        midx = MultiIndex.from_product(
            [
                Categorical(["a", "b", "c"]),
                Categorical(date_range("2012-01-01", periods=3, freq="h")),
            ]
        )
        df = DataFrame({"a": range(len(midx))}, index=midx)
        df2 = df.iloc[[0, 1, 2, 3, 4, 5, 6, 8]]

        result = df2.reindex(midx)
        expected = DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, np.nan, 8]}, index=midx)
        tm.assert_frame_equal(result, expected)

    def test_reindex_with_categoricalindex(self):
        df = DataFrame(
            {
                "A": np.arange(3, dtype="int64"),
            },
            index=CategoricalIndex(
                list("abc"), dtype=CategoricalDtype(list("cabe")), name="B"
            ),
        )

        # reindexing
        # convert to a regular index
        result = df.reindex(["a", "b", "e"])
        expected = DataFrame({"A": [0, 1, np.nan], "B": Series(list("abe"))}).set_index(
            "B"
        )
        tm.assert_frame_equal(result, expected, check_index_type=True)

        result = df.reindex(["a", "b"])
        expected = DataFrame({"A": [0, 1], "B": Series(list("ab"))}).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        result = df.reindex(["e"])
        expected = DataFrame({"A": [np.nan], "B": Series(["e"])}).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        result = df.reindex(["d"])
        expected = DataFrame({"A": [np.nan], "B": Series(["d"])}).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        # since we are actually reindexing with a Categorical
        # then return a Categorical
        cats = list("cabe")

        result = df.reindex(Categorical(["a", "e"], categories=cats))
        expected = DataFrame(
            {"A": [0, np.nan], "B": Series(list("ae")).astype(CategoricalDtype(cats))}
        ).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        result = df.reindex(Categorical(["a"], categories=cats))
        expected = DataFrame(
            {"A": [0], "B": Series(list("a")).astype(CategoricalDtype(cats))}
        ).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        result = df.reindex(["a", "b", "e"])
        expected = DataFrame({"A": [0, 1, np.nan], "B": Series(list("abe"))}).set_index(
            "B"
        )
        tm.assert_frame_equal(result, expected, check_index_type=True)

        result = df.reindex(["a", "b"])
        expected = DataFrame({"A": [0, 1], "B": Series(list("ab"))}).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        result = df.reindex(["e"])
        expected = DataFrame({"A": [np.nan], "B": Series(["e"])}).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        # give back the type of categorical that we received
        result = df.reindex(Categorical(["a", "e"], categories=cats, ordered=True))
        expected = DataFrame(
            {
                "A": [0, np.nan],
                "B": Series(list("ae")).astype(CategoricalDtype(cats, ordered=True)),
            }
        ).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        result = df.reindex(Categorical(["a", "d"], categories=["a", "d"]))
        expected = DataFrame(
            {
                "A": [0, np.nan],
                "B": Series(list("ad")).astype(CategoricalDtype(["a", "d"])),
            }
        ).set_index("B")
        tm.assert_frame_equal(result, expected, check_index_type=True)

        df2 = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
            },
            index=CategoricalIndex(
                list("aabbca"), dtype=CategoricalDtype(list("cabe")), name="B"
            ),
        )
        # passed duplicate indexers are not allowed
        msg = "cannot reindex on an axis with duplicate labels"
        with pytest.raises(ValueError, match=msg):
            df2.reindex(["a", "b"])

        # args NotImplemented ATM
        msg = r"argument {} is not implemented for CategoricalIndex\.reindex"
        with pytest.raises(NotImplementedError, match=msg.format("method")):
            df.reindex(["a"], method="ffill")
        with pytest.raises(NotImplementedError, match=msg.format("level")):
            df.reindex(["a"], level=1)
        with pytest.raises(NotImplementedError, match=msg.format("limit")):
            df.reindex(["a"], limit=2)

    def test_reindex_signature(self):
        sig = inspect.signature(DataFrame.reindex)
        parameters = set(sig.parameters)
        assert parameters == {
            "self",
            "labels",
            "index",
            "columns",
            "axis",
            "limit",
            "copy",
            "level",
            "method",
            "fill_value",
            "tolerance",
        }

    def test_reindex_multiindex_ffill_added_rows(self):
        # GH#23693
        # reindex added rows with nan values even when fill method was specified
        mi = MultiIndex.from_tuples([("a", "b"), ("d", "e")])
        df = DataFrame([[0, 7], [3, 4]], index=mi, columns=["x", "y"])
        mi2 = MultiIndex.from_tuples([("a", "b"), ("d", "e"), ("h", "i")])
        result = df.reindex(mi2, axis=0, method="ffill")
        expected = DataFrame([[0, 7], [3, 4], [3, 4]], index=mi2, columns=["x", "y"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"method": "pad", "tolerance": timedelta(seconds=9)},
            {"method": "backfill", "tolerance": timedelta(seconds=9)},
            {"method": "nearest"},
            {"method": None},
        ],
    )
    def test_reindex_empty_frame(self, kwargs):
        # GH#27315
        idx = date_range(start="2020", freq="30s", periods=3)
        df = DataFrame([], index=Index([], name="time"), columns=["a"])
        result = df.reindex(idx, **kwargs)
        expected = DataFrame({"a": [np.nan] * 3}, index=idx, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "src_idx",
        [
            Index([]),
            CategoricalIndex([]),
        ],
    )
    @pytest.mark.parametrize(
        "cat_idx",
        [
            # No duplicates
            Index([]),
            CategoricalIndex([]),
            Index(["A", "B"]),
            CategoricalIndex(["A", "B"]),
            # Duplicates: GH#38906
            Index(["A", "A"]),
            CategoricalIndex(["A", "A"]),
        ],
    )
    def test_reindex_empty(self, src_idx, cat_idx):
        df = DataFrame(columns=src_idx, index=["K"], dtype="f8")

        result = df.reindex(columns=cat_idx)
        expected = DataFrame(index=["K"], columns=cat_idx, dtype="f8")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]"])
    def test_reindex_datetimelike_to_object(self, dtype):
        # GH#39755 dont cast dt64/td64 to ints
        mi = MultiIndex.from_product([list("ABCDE"), range(2)])

        dti = date_range("2016-01-01", periods=10)
        fv = np.timedelta64("NaT", "ns")
        if dtype == "m8[ns]":
            dti = dti - dti[0]
            fv = np.datetime64("NaT", "ns")

        ser = Series(dti, index=mi)
        ser[::3] = pd.NaT

        df = ser.unstack()

        index = df.index.append(Index([1]))
        columns = df.columns.append(Index(["foo"]))

        res = df.reindex(index=index, columns=columns, fill_value=fv)

        expected = DataFrame(
            {
                0: df[0].tolist() + [fv],
                1: df[1].tolist() + [fv],
                "foo": np.array(["NaT"] * 6, dtype=fv.dtype),
            },
            index=index,
        )
        assert (res.dtypes[[0, 1]] == object).all()
        assert res.iloc[0, 0] is pd.NaT
        assert res.iloc[-1, 0] is fv
        assert res.iloc[-1, 1] is fv
        tm.assert_frame_equal(res, expected)

    @pytest.mark.parametrize(
        "index_df,index_res,index_exp",
        [
            (
                CategoricalIndex([], categories=["A"]),
                Index(["A"]),
                Index(["A"]),
            ),
            (
                CategoricalIndex([], categories=["A"]),
                Index(["B"]),
                Index(["B"]),
            ),
            (
                CategoricalIndex([], categories=["A"]),
                CategoricalIndex(["A"]),
                CategoricalIndex(["A"]),
            ),
            (
                CategoricalIndex([], categories=["A"]),
                CategoricalIndex(["B"]),
                CategoricalIndex(["B"]),
            ),
        ],
    )
    def test_reindex_not_category(self, index_df, index_res, index_exp):
        # GH#28690
        df = DataFrame(index=index_df)
        result = df.reindex(index=index_res)
        expected = DataFrame(index=index_exp)
        tm.assert_frame_equal(result, expected)

    def test_invalid_method(self):
        df = DataFrame({"A": [1, np.nan, 2]})

        msg = "Invalid fill method"
        with pytest.raises(ValueError, match=msg):
            df.reindex([1, 0, 2], method="asfreq")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\__init__.py ===
# encoding: utf-8
"""Helper classes for tests."""

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

import pickle
import importlib
import copy
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
    AttributeValueList,
    CharsetMetaAttributeValue,
    Comment,
    ContentMetaAttributeValue,
    Doctype,
    PageElement,
    PYTHON_SPECIFIC_ENCODINGS,
    Script,
    Stylesheet,
    Tag,
)
from bs4.filter import SoupStrainer
from bs4.builder import (
    XMLParsedAsHTMLWarning,
)
from bs4._typing import _IncomingMarkup

from bs4.builder import TreeBuilder
from bs4.builder._htmlparser import HTMLParserTreeBuilder

from typing import (
    Any,
    Iterable,
    Optional,
    Tuple,
    Type,
)

# Some tests depend on specific third-party libraries. We use
# @pytest.mark.skipIf on the following conditionals to skip them
# if the libraries are not installed.
try:
    from soupsieve import SelectorSyntaxError

    SOUP_SIEVE_PRESENT = True
except ImportError:
    SOUP_SIEVE_PRESENT = False

HTML5LIB_PRESENT = importlib.util.find_spec("html5lib") is not None

try:
    import lxml.etree
    LXML_PRESENT = True
    LXML_VERSION = lxml.etree.LXML_VERSION
except ImportError:
    LXML_PRESENT = False
    LXML_VERSION = (0,)

default_builder: Type[TreeBuilder] = HTMLParserTreeBuilder

BAD_DOCUMENT: str = """A bare string
<!DOCTYPE xsl:stylesheet SYSTEM "htmlent.dtd">
<!DOCTYPE xsl:stylesheet PUBLIC "htmlent.dtd">
<div><![CDATA[A CDATA section where it doesn't belong]]></div>
<div><svg><![CDATA[HTML5 does allow CDATA sections in SVG]]></svg></div>
<div>A <meta> tag</div>
<div>A <br> tag that supposedly has contents.</br></div>
<div>AT&T</div>
<div><textarea>Within a textarea, markup like <b> tags and <&<&amp; should be treated as literal</textarea></div>
<div><script>if (i < 2) { alert("<b>Markup within script tags should be treated as literal.</b>"); }</script></div>
<div>This numeric entity is missing the final semicolon: <x t="pi&#241ata"></div>
<div><a href="http://example.com/</a> that attribute value never got closed</div>
<div><a href="foo</a>, </a><a href="bar">that attribute value was closed by the subsequent tag</a></div>
<! This document starts with a bogus declaration ><div>a</div>
<div>This document contains <!an incomplete declaration <div>(do you see it?)</div>
<div>This document ends with <!an incomplete declaration
<div><a style={height:21px;}>That attribute value was bogus</a></div>
<! DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">The doctype is invalid because it contains extra whitespace
<div><table><td nowrap>That boolean attribute had no value</td></table></div>
<div>Here's a nonexistent entity: &#foo; (do you see it?)</div>
<div>This document ends before the entity finishes: &gt
<div><p>Paragraphs shouldn't contain block display elements, but this one does: <dl><dt>you see?</dt></p>
<b b="20" a="1" b="10" a="2" a="3" a="4">Multiple values for the same attribute.</b>
<div><table><tr><td>Here's a table</td></tr></table></div>
<div><table id="1"><tr><td>Here's a nested table:<table id="2"><tr><td>foo</td></tr></table></td></div>
<div>This tag contains nothing but whitespace: <b>    </b></div>
<div><blockquote><p><b>This p tag is cut off by</blockquote></p>the end of the blockquote tag</div>
<div><table><div>This table contains bare markup</div></table></div>
<div><div id="1">\n <a href="link1">This link is never closed.\n</div>\n<div id="2">\n <div id="3">\n   <a href="link2">This link is closed.</a>\n  </div>\n</div></div>
<div>This document contains a <!DOCTYPE surprise>surprise doctype</div>
<div><a><B><Cd><EFG>Mixed case tags are folded to lowercase</efg></CD></b></A></div>
<div><our\u2603>Tag name contains Unicode characters</our\u2603></div>
<div><a \u2603="snowman">Attribute name contains Unicode characters</a></div>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
"""


class SoupTest(object):
    @property
    def default_builder(self) -> Type[TreeBuilder]:
        return default_builder

    def soup(self, markup: _IncomingMarkup, **kwargs: Any) -> BeautifulSoup:
        """Build a Beautiful Soup object from markup."""
        builder = kwargs.pop("builder", self.default_builder)
        return BeautifulSoup(markup, builder=builder, **kwargs)

    def document_for(self, markup: str, **kwargs: Any) -> str:
        """Turn an HTML fragment into a document.

        The details depend on the builder.
        """
        return self.default_builder(**kwargs).test_fragment_to_document(markup)

    def assert_soup(
        self, to_parse: _IncomingMarkup, compare_parsed_to: Optional[str] = None
    ) -> None:
        """Parse some markup using Beautiful Soup and verify that
        the output markup is as expected.
        """
        builder = self.default_builder
        obj = BeautifulSoup(to_parse, builder=builder)
        if compare_parsed_to is None:
            assert isinstance(to_parse, str)
            compare_parsed_to = to_parse

        # Verify that the documents come out the same.
        assert obj.decode() == self.document_for(compare_parsed_to)

        # Also run some checks on the BeautifulSoup object itself:

        # Verify that every tag that was opened was eventually closed.

        # There are no tags in the open tag counter.
        assert all(v == 0 for v in list(obj.open_tag_counter.values()))

        # The only tag in the tag stack is the one for the root
        # document.
        assert [obj.ROOT_TAG_NAME] == [x.name for x in obj.tagStack]

    assertSoupEquals = assert_soup

    def assertConnectedness(self, element: Tag) -> None:
        """Ensure that next_element and previous_element are properly
        set for all descendants of the given element.
        """
        earlier = None
        for e in element.descendants:
            if earlier:
                assert e == earlier.next_element
                assert earlier == e.previous_element
            earlier = e

    def linkage_validator(
        self, el: Tag, _recursive_call: bool = False
    ) -> Optional[PageElement]:
        """Ensure proper linkage throughout the document."""
        descendant = None
        # Document element should have no previous element or previous sibling.
        # It also shouldn't have a next sibling.
        if el.parent is None:
            assert (
                el.previous_element is None
            ), "Bad previous_element\nNODE: {}\nPREV: {}\nEXPECTED: {}".format(
                el, el.previous_element, None
            )
            assert (
                el.previous_sibling is None
            ), "Bad previous_sibling\nNODE: {}\nPREV: {}\nEXPECTED: {}".format(
                el, el.previous_sibling, None
            )
            assert (
                el.next_sibling is None
            ), "Bad next_sibling\nNODE: {}\nNEXT: {}\nEXPECTED: {}".format(
                el, el.next_sibling, None
            )

        idx = 0
        child = None
        last_child = None
        last_idx = len(el.contents) - 1
        for child in el.contents:
            descendant = None

            # Parent should link next element to their first child
            # That child should have no previous sibling
            if idx == 0:
                if el.parent is not None:
                    assert (
                        el.next_element is child
                    ), "Bad next_element\nNODE: {}\nNEXT: {}\nEXPECTED: {}".format(
                        el, el.next_element, child
                    )
                    assert (
                        child.previous_element is el
                    ), "Bad previous_element\nNODE: {}\nPREV: {}\nEXPECTED: {}".format(
                        child, child.previous_element, el
                    )
                    assert (
                        child.previous_sibling is None
                    ), "Bad previous_sibling\nNODE: {}\nPREV {}\nEXPECTED: {}".format(
                        child, child.previous_sibling, None
                    )

            # If not the first child, previous index should link as sibling to this index
            # Previous element should match the last index or the last bubbled up descendant
            else:
                assert (
                    child.previous_sibling is el.contents[idx - 1]
                ), "Bad previous_sibling\nNODE: {}\nPREV {}\nEXPECTED {}".format(
                    child, child.previous_sibling, el.contents[idx - 1]
                )
                assert (
                    el.contents[idx - 1].next_sibling is child
                ), "Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                    el.contents[idx - 1], el.contents[idx - 1].next_sibling, child
                )

                if last_child is not None:
                    assert (
                        child.previous_element is last_child
                    ), "Bad previous_element\nNODE: {}\nPREV {}\nEXPECTED {}\nCONTENTS {}".format(
                        child, child.previous_element, last_child, child.parent.contents
                    )
                    assert (
                        last_child.next_element is child
                    ), "Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                        last_child, last_child.next_element, child
                    )

            if isinstance(child, Tag) and child.contents:
                descendant = self.linkage_validator(child, True)
                assert descendant is not None
                # A bubbled up descendant should have no next siblings
                assert (
                    descendant.next_sibling is None
                ), "Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                    descendant, descendant.next_sibling, None
                )

            # Mark last child as either the bubbled up descendant or the current child
            if descendant is not None:
                last_child = descendant
            else:
                last_child = child

            # If last child, there are non next siblings
            if idx == last_idx:
                assert (
                    child.next_sibling is None
                ), "Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                    child, child.next_sibling, None
                )
            idx += 1

        child = descendant if descendant is not None else child
        if child is None:
            child = el

        if not _recursive_call and child is not None:
            target: Optional[Tag] = el
            while True:
                if target is None:
                    assert (
                        child.next_element is None
                    ), "Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                        child, child.next_element, None
                    )
                    break
                elif target.next_sibling is not None:
                    assert (
                        child.next_element is target.next_sibling
                    ), "Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                        child, child.next_element, target.next_sibling
                    )
                    break
                target = target.parent

            # We are done, so nothing to return
            return None
        else:
            # Return the child to the recursive caller
            return child

    def assert_selects(self, tags: Iterable[Tag], should_match: Iterable[str]) -> None:
        """Make sure that the given tags have the correct text.

        This is used in tests that define a bunch of tags, each
        containing a single string, and then select certain strings by
        some mechanism.
        """
        assert [tag.string for tag in tags] == should_match

    def assert_selects_ids(
        self, tags: Iterable[Tag], should_match: Iterable[str]
    ) -> None:
        """Make sure that the given tags have the correct IDs.

        This is used in tests that define a bunch of tags, each
        containing a single string, and then select certain strings by
        some mechanism.
        """
        assert [tag["id"] for tag in tags] == should_match


class TreeBuilderSmokeTest(SoupTest):
    # Tests that are common to HTML and XML tree builders.

    @pytest.mark.parametrize(
        "multi_valued_attributes", [None, {}, dict(b=["class"]), {"*": ["notclass"]}]
    )
    def test_attribute_not_multi_valued(self, multi_valued_attributes):
        markup = '<html xmlns="http://www.w3.org/1999/xhtml"><a class="a b c"></html>'
        soup = self.soup(markup, multi_valued_attributes=multi_valued_attributes)
        assert soup.a["class"] == "a b c"

    @pytest.mark.parametrize(
        "multi_valued_attributes", [dict(a=["class"]), {"*": ["class"]}]
    )
    def test_attribute_multi_valued(self, multi_valued_attributes):
        markup = '<a class="a b c">'
        soup = self.soup(markup, multi_valued_attributes=multi_valued_attributes)
        assert soup.a["class"] == ["a", "b", "c"]

    def test_invalid_doctype(self):
        # We don't have an official opinion on how these are parsed,
        # but they shouldn't crash any of the parsers.
        markup = "<![if word]>content<![endif]>"
        self.soup(markup)
        markup = "<!DOCTYPE html]ff>"
        self.soup(markup)

    def test_doctype_filtered(self):
        markup = "<!DOCTYPE html>\n<html>\n</html>"
        soup = self.soup(markup, parse_only=SoupStrainer(name="html"))
        assert not any(isinstance(x, Doctype) for x in soup.descendants)

    def test_custom_attribute_dict_class(self):
        class MyAttributeDict(dict):
            def __setitem__(self, key: str, value: Any):
                # Ignore the provided value and substitute a
                # hard-coded one.
                super().__setitem__(key, "OVERRIDDEN")

        markup = '<a attr1="val1" attr2="val2">f</a>'
        builder = self.default_builder(attribute_dict_class=MyAttributeDict)
        soup = self.soup(markup, builder=builder)
        tag = soup.a
        assert isinstance(tag.attrs, MyAttributeDict)
        assert "OVERRIDDEN" == tag["attr1"]
        tag["attr3"] = True
        assert "OVERRIDDEN" == tag["attr3"]

        expect = '<a attr1="OVERRIDDEN" attr2="OVERRIDDEN" attr3="OVERRIDDEN">f</a>'
        assert expect == tag.decode()

    def test_custom_attribute_value_list_class(self):
        class MyCustomAttributeValueList(AttributeValueList):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.append("extra")

        builder = self.default_builder(
            multi_valued_attributes={"*": set(["attr2"])},
            attribute_value_list_class=MyCustomAttributeValueList,
        )
        markup = '<a attr1="val1" attr2="val2">f</a>'
        soup = self.soup(markup, builder=builder)
        tag = soup.a
        assert tag["attr1"] == "val1"
        assert tag["attr2"] == ["val2", "extra"]
        assert isinstance(tag["attr2"], MyCustomAttributeValueList)


class HTMLTreeBuilderSmokeTest(TreeBuilderSmokeTest):
    """A basic test of a treebuilder's competence.

    Any HTML treebuilder, present or future, should be able to pass
    these tests. With invalid markup, there's room for interpretation,
    and different parsers can handle it differently. But with the
    markup in these tests, there's not much room for interpretation.
    """

    def test_empty_element_tags(self):
        """Verify that all HTML4 and HTML5 empty element (aka void element) tags
        are handled correctly.
        """
        for name in [
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "keygen",
            "link",
            "menuitem",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
            "spacer",
            "frame",
        ]:
            soup = self.soup("")
            new_tag = soup.new_tag(name)
            assert new_tag.is_empty_element is True

        self.assert_soup("<br/><br/><br/>", "<br/><br/><br/>")
        self.assert_soup("<br /><br /><br />", "<br/><br/><br/>")

    def test_special_string_containers(self):
        soup = self.soup("<style>Some CSS</style><script>Some Javascript</script>")
        assert isinstance(soup.style.string, Stylesheet)
        assert isinstance(soup.script.string, Script)

        soup = self.soup("<style><!--Some CSS--></style>")
        assert isinstance(soup.style.string, Stylesheet)
        # The contents of the style tag resemble an HTML comment, but
        # it's not treated as a comment.
        assert soup.style.string == "<!--Some CSS-->"
        assert isinstance(soup.style.string, Stylesheet)

    def test_pickle_and_unpickle_identity(self):
        # Pickling a tree, then unpickling it, yields a tree identical
        # to the original.
        tree = self.soup("<a><b>foo</a>")
        dumped = pickle.dumps(tree, pickle.HIGHEST_PROTOCOL)
        loaded = pickle.loads(dumped)
        assert loaded.__class__ == BeautifulSoup
        assert loaded.decode() == tree.decode()

    def test_pickle_and_unpickle_bad_markup(self):
        markup = """
<!DOCTYPE html>
<html lang="en">
<head><title>blabla</title></head>
<body><?xml encoding="utf-8" ?><html></html></body>
</html>
"""
        soup = self.soup(markup)
        pickled = pickle.dumps(soup, pickle.HIGHEST_PROTOCOL)
        soup = pickle.loads(pickled)
        assert soup.builder.is_xml is False

    def assertDoctypeHandled(self, doctype_fragment: str) -> None:
        """Assert that a given doctype string is handled correctly."""
        doctype_str, soup = self._document_with_doctype(doctype_fragment)

        # Make sure a Doctype object was created.
        doctype = soup.contents[0]
        assert doctype.__class__ == Doctype
        assert doctype == doctype_fragment
        assert soup.encode("utf8")[: len(doctype_str)] == doctype_str

        # Make sure that the doctype was correctly associated with the
        # parse tree and that the rest of the document parsed.
        assert soup.p is not None
        assert soup.p.contents[0] == "foo"

    def _document_with_doctype(
        self, doctype_fragment: str, doctype_string: str = "DOCTYPE"
    ) -> Tuple[bytes, BeautifulSoup]:
        """Generate and parse a document with the given doctype."""
        doctype = "<!%s %s>" % (doctype_string, doctype_fragment)
        markup = doctype + "\n<p>foo</p>"
        soup = self.soup(markup)
        return doctype.encode("utf8"), soup

    def test_normal_doctypes(self):
        """Make sure normal, everyday HTML doctypes are handled correctly."""
        self.assertDoctypeHandled("html")
        self.assertDoctypeHandled(
            'html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"'
        )

    def test_empty_doctype(self):
        soup = self.soup("<!DOCTYPE>")
        doctype = soup.contents[0]
        assert "" == doctype.strip()

    def test_mixed_case_doctype(self):
        # A lowercase or mixed-case doctype becomes a Doctype.
        for doctype_fragment in ("doctype", "DocType"):
            doctype_str, soup = self._document_with_doctype("html", doctype_fragment)

            # Make sure a Doctype object was created and that the DOCTYPE
            # is uppercase.
            doctype = soup.contents[0]
            assert doctype.__class__ == Doctype
            assert doctype == "html"
            assert soup.encode("utf8")[: len(doctype_str)] == b"<!DOCTYPE html>"

            # Make sure that the doctype was correctly associated with the
            # parse tree and that the rest of the document parsed.
            assert soup.p.contents[0] == "foo"

    def test_public_doctype_with_url(self):
        doctype = 'html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"'
        self.assertDoctypeHandled(doctype)

    def test_system_doctype(self):
        self.assertDoctypeHandled('foo SYSTEM "http://www.example.com/"')

    def test_namespaced_system_doctype(self):
        # We can handle a namespaced doctype with a system ID.
        self.assertDoctypeHandled('xsl:stylesheet SYSTEM "htmlent.dtd"')

    def test_namespaced_public_doctype(self):
        # Test a namespaced doctype with a public id.
        self.assertDoctypeHandled('xsl:stylesheet PUBLIC "htmlent.dtd"')

    def test_real_xhtml_document(self):
        """A real XHTML document should come out more or less the same as it went in."""
        markup = b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Hello.</title></head>
<body>Goodbye.</body>
</html>"""
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)
        assert soup.encode("utf-8").replace(b"\n", b"") == markup.replace(b"\n", b"")

        # No warning was issued about parsing an XML document as HTML,
        # because XHTML is both.
        assert w == []

    def test_namespaced_html(self):
        # When a namespaced XML document is parsed as HTML it should
        # be treated as HTML with weird tag names.
        markup = b"""<ns1:foo>content</ns1:foo><ns1:foo/><ns2:foo/>"""
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)

        assert 2 == len(soup.find_all("ns1:foo"))

        # n.b. no "you're parsing XML as HTML" warning was given
        # because there was no XML declaration.
        assert [] == w

    def test_detect_xml_parsed_as_html(self):
        # A warning is issued when parsing an XML document as HTML,
        # but basic stuff should still work.
        markup = b"""<?xml version="1.0" encoding="utf-8"?><tag>string</tag>"""
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)
            assert soup.tag.string == "string"
        [warning] = w
        assert isinstance(warning.message, XMLParsedAsHTMLWarning)
        assert str(warning.message) == XMLParsedAsHTMLWarning.MESSAGE

        # NOTE: the warning is not issued if the document appears to
        # be XHTML (tested with test_real_xhtml_document in the
        # superclass) or if there is no XML declaration (tested with
        # test_namespaced_html in the superclass).

    def test_processing_instruction(self):
        # We test both Unicode and bytestring to verify that
        # process_markup correctly sets processing_instruction_class
        # even when the markup is already Unicode and there is no
        # need to process anything.
        markup = """<?PITarget PIContent?>"""
        soup = self.soup(markup)
        assert markup == soup.decode()

        markup = b"""<?PITarget PIContent?>"""
        soup = self.soup(markup)
        assert markup == soup.encode("utf8")

    def test_deepcopy(self):
        """Make sure you can copy the tree builder.

        This is important because the builder is part of a
        BeautifulSoup object, and we want to be able to copy that.
        """
        copy.deepcopy(self.default_builder)

    def test_p_tag_is_never_empty_element(self):
        """A <p> tag is never designated as an empty-element tag.

        Even if the markup shows it as an empty-element tag, it
        shouldn't be presented that way.
        """
        soup = self.soup("<p/>")
        assert not soup.p.is_empty_element
        assert str(soup.p) == "<p></p>"

    def test_unclosed_tags_get_closed(self):
        """A tag that's not closed by the end of the document should be closed.

        This applies to all tags except empty-element tags.
        """
        self.assert_soup("<p>", "<p></p>")
        self.assert_soup("<b>", "<b></b>")

        self.assert_soup("<br>", "<br/>")

    def test_br_is_always_empty_element_tag(self):
        """A <br> tag is designated as an empty-element tag.

        Some parsers treat <br></br> as one <br/> tag, some parsers as
        two tags, but it should always be an empty-element tag.
        """
        soup = self.soup("<br></br>")
        assert soup.br.is_empty_element
        assert str(soup.br) == "<br/>"

    def test_nested_formatting_elements(self):
        self.assert_soup("<em><em></em></em>")

    def test_double_head(self):
        html = """<!DOCTYPE html>
<html>
<head>
<title>Ordinary HEAD element test</title>
</head>
<script type="text/javascript">
alert("Help!");
</script>
<body>
Hello, world!
</body>
</html>
"""
        soup = self.soup(html)
        assert "text/javascript" == soup.find("script")["type"]

    def test_comment(self):
        # Comments are represented as Comment objects.
        markup = "<p>foo<!--foobar-->baz</p>"
        self.assert_soup(markup)

        soup = self.soup(markup)
        comment = soup.find(string="foobar")
        assert comment.__class__ == Comment

        # The comment is properly integrated into the tree.
        foo = soup.find(string="foo")
        assert comment == foo.next_element
        baz = soup.find(string="baz")
        assert comment == baz.previous_element

    def test_preserved_whitespace_in_pre_and_textarea(self):
        """Whitespace must be preserved in <pre> and <textarea> tags,
        even if that would mean not prettifying the markup.
        """
        pre_markup = "<pre>a   z</pre>\n"
        textarea_markup = "<textarea> woo\nwoo  </textarea>\n"
        self.assert_soup(pre_markup)
        self.assert_soup(textarea_markup)

        soup = self.soup(pre_markup)
        assert soup.pre.prettify() == pre_markup

        soup = self.soup(textarea_markup)
        assert soup.textarea.prettify() == textarea_markup

        soup = self.soup("<textarea></textarea>")
        assert soup.textarea.prettify() == "<textarea></textarea>\n"

    def test_nested_inline_elements(self):
        """Inline elements can be nested indefinitely."""
        b_tag = "<b>Inside a B tag</b>"
        self.assert_soup(b_tag)

        nested_b_tag = "<p>A <i>nested <b>tag</b></i></p>"
        self.assert_soup(nested_b_tag)

        double_nested_b_tag = "<p>A <a>doubly <i>nested <b>tag</b></i></a></p>"
        self.assert_soup(double_nested_b_tag)

    def test_nested_block_level_elements(self):
        """Block elements can be nested."""
        soup = self.soup("<blockquote><p><b>Foo</b></p></blockquote>")
        blockquote = soup.blockquote
        assert blockquote.p.b.string == "Foo"
        assert blockquote.b.string == "Foo"

    def test_correctly_nested_tables(self):
        """One table can go inside another one."""
        markup = (
            '<table id="1">'
            "<tr>"
            "<td>Here's another table:"
            '<table id="2">'
            "<tr><td>foo</td></tr>"
            "</table></td>"
        )

        self.assert_soup(
            markup,
            '<table id="1"><tr><td>Here\'s another table:'
            '<table id="2"><tr><td>foo</td></tr></table>'
            "</td></tr></table>",
        )

        self.assert_soup(
            "<table><thead><tr><td>Foo</td></tr></thead>"
            "<tbody><tr><td>Bar</td></tr></tbody>"
            "<tfoot><tr><td>Baz</td></tr></tfoot></table>"
        )

    def test_multivalued_attribute_with_whitespace(self):
        # Whitespace separating the values of a multi-valued attribute
        # should be ignored.

        markup = '<div class=" foo bar	 "></a>'
        soup = self.soup(markup)
        assert ["foo", "bar"] == soup.div["class"]

        # If you search by the literal name of the class it's like the whitespace
        # wasn't there.
        assert soup.div == soup.find("div", class_="foo bar")

    def test_deeply_nested_multivalued_attribute(self):
        # html5lib can set the attributes of the same tag many times
        # as it rearranges the tree. This has caused problems with
        # multivalued attributes.
        markup = '<table><div><div class="css"></div></div></table>'
        soup = self.soup(markup)
        assert ["css"] == soup.div.div["class"]

    def test_multivalued_attribute_on_html(self):
        # html5lib uses a different API to set the attributes ot the
        # <html> tag. This has caused problems with multivalued
        # attributes.
        markup = '<html class="a b"></html>'
        soup = self.soup(markup)
        assert ["a", "b"] == soup.html["class"]

    def test_angle_brackets_in_attribute_values_are_escaped(self):
        self.assert_soup('<a b="<a>"></a>', '<a b="&lt;a&gt;"></a>')

    def test_strings_resembling_character_entity_references(self):
        # "&T" and "&p" look like incomplete character entities, but they are
        # not.
        self.assert_soup(
            "<p>&bull; AT&T is in the s&p 500</p>",
            "<p>\u2022 AT&amp;T is in the s&amp;p 500</p>",
        )

    def test_apos_entity(self):
        self.assert_soup(
            "<p>Bob&apos;s Bar</p>",
            "<p>Bob's Bar</p>",
        )

    def test_entities_in_foreign_document_encoding(self):
        # &#147; and &#148; are invalid numeric entities referencing
        # Windows-1252 characters. &#45; references a character common
        # to Windows-1252 and Unicode, and &#9731; references a
        # character only found in Unicode.
        #
        # All of these entities should be converted to Unicode
        # characters.
        markup = "<p>&#147;Hello&#148; &#45;&#9731;</p>"
        soup = self.soup(markup)
        assert "“Hello” -☃" == soup.p.string

    def test_entities_in_attributes_converted_to_unicode(self):
        expect = '<p id="pi\N{LATIN SMALL LETTER N WITH TILDE}ata"></p>'
        self.assert_soup('<p id="pi&#241;ata"></p>', expect)
        self.assert_soup('<p id="pi&#xf1;ata"></p>', expect)
        self.assert_soup('<p id="pi&#Xf1;ata"></p>', expect)
        self.assert_soup('<p id="pi&ntilde;ata"></p>', expect)

    def test_entities_in_text_converted_to_unicode(self):
        expect = "<p>pi\N{LATIN SMALL LETTER N WITH TILDE}ata</p>"
        self.assert_soup("<p>pi&#241;ata</p>", expect)
        self.assert_soup("<p>pi&#xf1;ata</p>", expect)
        self.assert_soup("<p>pi&#Xf1;ata</p>", expect)
        self.assert_soup("<p>pi&ntilde;ata</p>", expect)

    def test_quot_entity_converted_to_quotation_mark(self):
        self.assert_soup(
            "<p>I said &quot;good day!&quot;</p>", '<p>I said "good day!"</p>'
        )

    def test_out_of_range_entity(self):
        expect = "\N{REPLACEMENT CHARACTER}"
        self.assert_soup("&#10000000000000;", expect)
        self.assert_soup("&#x10000000000000;", expect)
        self.assert_soup("&#1000000000;", expect)

    def test_multipart_strings(self):
        "Mostly to prevent a recurrence of a bug in the html5lib treebuilder."
        soup = self.soup("<html><h2>\nfoo</h2><p></p></html>")
        assert "p" == soup.h2.string.next_element.name
        assert "p" == soup.p.name
        self.assertConnectedness(soup)

    def test_invalid_html_entity(self):
        # The html.parser treebuilder can't distinguish between an
        # invalid HTML entity with a semicolon and an invalid HTML
        # entity with no semicolon (see its subclass for the tested
        # behavior). But the other treebuilders can.
        markup = "<p>a &nosuchentity b</p>"
        soup = self.soup(markup)
        assert "<p>a &amp;nosuchentity b</p>" == soup.p.decode()

        markup = "<p>a &nosuchentity; b</p>"
        soup = self.soup(markup)
        assert "<p>a &amp;nosuchentity; b</p>" == soup.p.decode()

    def test_head_tag_between_head_and_body(self):
        "Prevent recurrence of a bug in the html5lib treebuilder."
        content = """<html><head></head>
  <link></link>
  <body>foo</body>
</html>
"""
        soup = self.soup(content)
        assert soup.html.body is not None
        self.assertConnectedness(soup)

    def test_multiple_copies_of_a_tag(self):
        "Prevent recurrence of a bug in the html5lib treebuilder."
        content = """<!DOCTYPE html>
<html>
 <body>
   <article id="a" >
   <div><a href="1"></div>
   <footer>
     <a href="2"></a>
   </footer>
  </article>
  </body>
</html>
"""
        soup = self.soup(content)
        self.assertConnectedness(soup.article)

    def test_basic_namespaces(self):
        """Parsers don't need to *understand* namespaces, but at the
        very least they should not choke on namespaces or lose
        data."""

        markup = b'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:mathml="http://www.w3.org/1998/Math/MathML" xmlns:svg="http://www.w3.org/2000/svg"><head></head><body><mathml:msqrt>4</mathml:msqrt><b svg:fill="red"></b></body></html>'
        soup = self.soup(markup)
        assert markup == soup.encode()
        assert "http://www.w3.org/1999/xhtml" == soup.html["xmlns"]
        assert "http://www.w3.org/1998/Math/MathML" == soup.html["xmlns:mathml"]
        assert "http://www.w3.org/2000/svg" == soup.html["xmlns:svg"]

    def test_multivalued_attribute_value_becomes_list(self):
        markup = b'<a class="foo bar">'
        soup = self.soup(markup)
        assert ["foo", "bar"] == soup.a["class"]

    #
    # Generally speaking, tests below this point are more tests of
    # Beautiful Soup than tests of the tree builders. But parsers are
    # weird, so we run these tests separately for every tree builder
    # to detect any differences between them.
    #

    def test_can_parse_unicode_document(self):
        # A seemingly innocuous document... but it's in Unicode! And
        # it contains characters that can't be represented in the
        # encoding found in the  declaration! The horror!
        markup = '<html><head><meta encoding="euc-jp"></head><body>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</body>'
        soup = self.soup(markup)
        assert "Sacr\xe9 bleu!" == soup.body.string

    def test_soupstrainer(self):
        """Parsers should be able to work with SoupStrainers."""
        strainer = SoupStrainer("b")
        soup = self.soup("A <b>bold</b> <meta/> <i>statement</i>", parse_only=strainer)
        assert soup.decode() == "<b>bold</b>"

    def test_single_quote_attribute_values_become_double_quotes(self):
        self.assert_soup("<foo attr='bar'></foo>", '<foo attr="bar"></foo>')

    def test_attribute_values_with_nested_quotes_are_left_alone(self):
        text = """<foo attr='bar "brawls" happen'>a</foo>"""
        self.assert_soup(text)

    def test_attribute_values_with_double_nested_quotes_get_quoted(self):
        text = """<foo attr='bar "brawls" happen'>a</foo>"""
        soup = self.soup(text)
        soup.foo["attr"] = 'Brawls happen at "Bob\'s Bar"'
        self.assert_soup(
            soup.foo.decode(),
            """<foo attr="Brawls happen at &quot;Bob\'s Bar&quot;">a</foo>""",
        )

    def test_ampersand_in_attribute_value_gets_escaped(self):
        self.assert_soup(
            '<this is="really messed up & stuff"></this>',
            '<this is="really messed up &amp; stuff"></this>',
        )

        self.assert_soup(
            '<a href="http://example.org?a=1&b=2;3">foo</a>',
            '<a href="http://example.org?a=1&amp;b=2;3">foo</a>',
        )

    def test_escaped_ampersand_in_attribute_value_is_left_alone(self):
        self.assert_soup('<a href="http://example.org?a=1&amp;b=2;3"></a>')

    def test_entities_in_strings_converted_during_parsing(self):
        # Both XML and HTML entities are converted to Unicode characters
        # during parsing.
        text = "<p>&lt;&lt;sacr&eacute;&#32;bleu!&gt;&gt;</p>"
        expected = (
            "<p>&lt;&lt;sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</p>"
        )
        self.assert_soup(text, expected)

    def test_smart_quotes_converted_on_the_way_in(self):
        # Microsoft smart quotes are converted to Unicode characters during
        # parsing.
        quote = b"<p>\x91Foo\x92</p>"
        soup = self.soup(quote, from_encoding="windows-1252")
        assert (
            soup.p.string
            == "\N{LEFT SINGLE QUOTATION MARK}Foo\N{RIGHT SINGLE QUOTATION MARK}"
        )

    def test_non_breaking_spaces_converted_on_the_way_in(self):
        soup = self.soup("<a>&nbsp;&nbsp;</a>")
        assert soup.a.string == "\N{NO-BREAK SPACE}" * 2

    def test_entities_converted_on_the_way_out(self):
        text = "<p>&lt;&lt;sacr&eacute;&#32;bleu!&gt;&gt;</p>"
        expected = "<p>&lt;&lt;sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</p>".encode(
            "utf-8"
        )
        soup = self.soup(text)
        assert soup.p.encode("utf-8") == expected

    def test_real_iso_8859_document(self):
        # Smoke test of interrelated functionality, using an
        # easy-to-understand document.

        # Here it is in Unicode. Note that it claims to be in ISO-8859-1.
        unicode_html = '<html><head><meta content="text/html; charset=ISO-8859-1" http-equiv="Content-type"/></head><body><p>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</p></body></html>'

        # That's because we're going to encode it into ISO-8859-1,
        # and use that to test.
        iso_latin_html = unicode_html.encode("iso-8859-1")

        # Parse the ISO-8859-1 HTML.
        soup = self.soup(iso_latin_html)

        # Encode it to UTF-8.
        result = soup.encode("utf-8")

        # What do we expect the result to look like? Well, it would
        # look like unicode_html, except that the META tag would say
        # UTF-8 instead of ISO-8859-1.
        expected = unicode_html.replace("ISO-8859-1", "utf-8")

        # And, of course, it would be in UTF-8, not Unicode.
        expected = expected.encode("utf-8")

        # Ta-da!
        assert result == expected

    def test_real_shift_jis_document(self):
        # Smoke test to make sure the parser can handle a document in
        # Shift-JIS encoding, without choking.
        shift_jis_html = (
            b"<html><head></head><body><pre>"
            b"\x82\xb1\x82\xea\x82\xcdShift-JIS\x82\xc5\x83R\x81[\x83f"
            b"\x83B\x83\x93\x83O\x82\xb3\x82\xea\x82\xbd\x93\xfa\x96{\x8c"
            b"\xea\x82\xcc\x83t\x83@\x83C\x83\x8b\x82\xc5\x82\xb7\x81B"
            b"</pre></body></html>"
        )
        unicode_html = shift_jis_html.decode("shift-jis")
        soup = self.soup(unicode_html)

        # Make sure the parse tree is correctly encoded to various
        # encodings.
        assert soup.encode("utf-8") == unicode_html.encode("utf-8")
        assert soup.encode("euc_jp") == unicode_html.encode("euc_jp")

    def test_real_hebrew_document(self):
        # A real-world test to make sure we can convert ISO-8859-9 (a
        # Hebrew encoding) to UTF-8.
        hebrew_document = b"<html><head><title>Hebrew (ISO 8859-8) in Visual Directionality</title></head><body><h1>Hebrew (ISO 8859-8) in Visual Directionality</h1>\xed\xe5\xec\xf9</body></html>"
        soup = self.soup(hebrew_document, from_encoding="iso8859-8")
        # Some tree builders call it iso8859-8, others call it iso-8859-9.
        # That's not a difference we really care about.
        assert soup.original_encoding in ("iso8859-8", "iso-8859-8")
        assert soup.encode("utf-8") == (
            hebrew_document.decode("iso8859-8").encode("utf-8")
        )

    def test_meta_tag_reflects_current_encoding(self):
        # Here's the <meta> tag saying that a document is
        # encoded in Shift-JIS.
        meta_tag = (
            '<meta content="text/html; charset=x-sjis" ' 'http-equiv="Content-type"/>'
        )

        # Here's a document incorporating that meta tag.
        shift_jis_html = (
            "<html><head>\n%s\n"
            '<meta http-equiv="Content-language" content="ja"/>'
            "</head><body>Shift-JIS markup goes here."
        ) % meta_tag
        soup = self.soup(shift_jis_html)

        # Parse the document, and the charset is seemingly unaffected.
        parsed_meta = soup.find("meta", {"http-equiv": "Content-type"})
        content = parsed_meta["content"]
        assert "text/html; charset=x-sjis" == content

        # But that value is actually a ContentMetaAttributeValue object.
        assert isinstance(content, ContentMetaAttributeValue)

        # And it will take on a value that reflects its current
        # encoding.
        assert "text/html; charset=utf8" == content.substitute_encoding("utf8")

        # No matter how the <meta> tag is encoded, its charset attribute
        # will always be accurate.
        assert b"charset=utf8" in parsed_meta.encode("utf8")
        assert b"charset=shift-jis" in parsed_meta.encode("shift-jis")

        # For the rest of the story, see TestSubstitutions in
        # test_tree.py.

    def test_html5_style_meta_tag_reflects_current_encoding(self):
        # Here's the <meta> tag saying that a document is
        # encoded in Shift-JIS.
        meta_tag = '<meta id="encoding" charset="x-sjis" />'

        # Here's a document incorporating that meta tag.
        shift_jis_html = (
            "<html><head>\n%s\n"
            '<meta http-equiv="Content-language" content="ja"/>'
            "</head><body>Shift-JIS markup goes here."
        ) % meta_tag
        soup = self.soup(shift_jis_html)

        # Parse the document, and the charset is seemingly unaffected.
        parsed_meta = soup.find("meta", id="encoding")
        charset = parsed_meta["charset"]
        assert "x-sjis" == charset

        # But that value is actually a CharsetMetaAttributeValue object.
        assert isinstance(charset, CharsetMetaAttributeValue)

        # And it will take on a value that reflects its current
        # encoding.
        assert "utf8" == charset.substitute_encoding("utf8")

        # No matter how the <meta> tag is encoded, its charset attribute
        # will always be accurate.
        assert b'charset="utf8"' in parsed_meta.encode("utf8")
        assert b'charset="shift-jis"' in parsed_meta.encode("shift-jis")

    def test_python_specific_encodings_not_used_in_charset(self):
        # You can encode an HTML document using a Python-specific
        # encoding, but that encoding won't be mentioned _inside_ the
        # resulting document. Instead, the document will appear to
        # have no encoding.
        for markup in [
            b'<meta charset="utf8"></head>' b'<meta id="encoding" charset="utf-8" />'
        ]:
            soup = self.soup(markup)
            for encoding in PYTHON_SPECIFIC_ENCODINGS:
                if encoding in (
                    "idna",
                    "mbcs",
                    "oem",
                    "undefined",
                    "string_escape",
                    "string-escape",
                ):
                    # For one reason or another, these will raise an
                    # exception if we actually try to use them, so don't
                    # bother.
                    continue
                encoded = soup.encode(encoding)
                assert b'meta charset=""' in encoded
                assert encoding.encode("ascii") not in encoded

    def test_tag_with_no_attributes_can_have_attributes_added(self):
        data = self.soup("<a>text</a>")
        data.a["foo"] = "bar"
        assert '<a foo="bar">text</a>' == data.a.decode()

    def test_closing_tag_with_no_opening_tag(self):
        # Without BeautifulSoup.open_tag_counter, the </span> tag will
        # cause _popToTag to be called over and over again as we look
        # for a <span> tag that wasn't there. The result is that 'text2'
        # will show up outside the body of the document.
        soup = self.soup("<body><div><p>text1</p></span>text2</div></body>")
        assert "<body><div><p>text1</p>text2</div></body>" == soup.body.decode()

    def test_worst_case(self):
        """Test the worst case (currently) for linking issues."""

        soup = self.soup(BAD_DOCUMENT)
        self.linkage_validator(soup)


class XMLTreeBuilderSmokeTest(TreeBuilderSmokeTest):
    def test_pickle_and_unpickle_identity(self):
        # Pickling a tree, then unpickling it, yields a tree identical
        # to the original.
        tree = self.soup("<a><b>foo</a>")
        dumped = pickle.dumps(tree, 2)
        loaded = pickle.loads(dumped)
        assert loaded.__class__ == BeautifulSoup
        assert loaded.decode() == tree.decode()

    def test_docstring_generated(self):
        soup = self.soup("<root/>")
        assert soup.encode() == b'<?xml version="1.0" encoding="utf-8"?>\n<root/>'

    def test_xml_declaration(self):
        markup = b"""<?xml version="1.0" encoding="utf8"?>\n<foo/>"""
        soup = self.soup(markup)
        assert markup == soup.encode("utf8")

    def test_python_specific_encodings_not_used_in_xml_declaration(self):
        # You can encode an XML document using a Python-specific
        # encoding, but that encoding won't be mentioned _inside_ the
        # resulting document.
        markup = b"""<?xml version="1.0"?>\n<foo/>"""
        soup = self.soup(markup)
        for encoding in PYTHON_SPECIFIC_ENCODINGS:
            if encoding in (
                "idna",
                "mbcs",
                "oem",
                "undefined",
                "string_escape",
                "string-escape",
            ):
                # For one reason or another, these will raise an
                # exception if we actually try to use them, so don't
                # bother.
                continue
            encoded = soup.encode(encoding)
            assert b'<?xml version="1.0"?>' in encoded
            assert encoding.encode("ascii") not in encoded

    def test_processing_instruction(self):
        markup = b"""<?xml version="1.0" encoding="utf8"?>\n<?PITarget PIContent?>"""
        soup = self.soup(markup)
        assert markup == soup.encode("utf8")

    def test_real_xhtml_document(self):
        """A real XHTML document should come out *exactly* the same as it went in."""
        markup = b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Hello.</title></head>
<body>Goodbye.</body>
</html>"""
        soup = self.soup(markup)
        assert soup.encode("utf-8") == markup

    def test_nested_namespaces(self):
        doc = b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<parent xmlns="http://ns1/">
<child xmlns="http://ns2/" xmlns:ns3="http://ns3/">
<grandchild ns3:attr="value" xmlns="http://ns4/"/>
</child>
</parent>"""
        soup = self.soup(doc)
        assert doc == soup.encode()

    def test_formatter_processes_script_tag_for_xml_documents(self):
        doc = """
  <script type="text/javascript">
  </script>
"""
        soup = BeautifulSoup(doc, "lxml-xml")
        # lxml would have stripped this while parsing, but we can add
        # it later.
        soup.script.string = 'console.log("< < hey > > ");'
        encoded = soup.encode()
        assert b"&lt; &lt; hey &gt; &gt;" in encoded

    def test_can_parse_unicode_document(self):
        markup = '<?xml version="1.0" encoding="euc-jp"><root>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</root>'
        soup = self.soup(markup)
        assert "Sacr\xe9 bleu!" == soup.root.string

    def test_can_parse_unicode_document_begining_with_bom(self):
        markup = '\N{BYTE ORDER MARK}<?xml version="1.0" encoding="euc-jp"><root>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</root>'
        soup = self.soup(markup)
        assert "Sacr\xe9 bleu!" == soup.root.string

    def test_popping_namespaced_tag(self):
        markup = '<rss xmlns:dc="foo"><dc:creator>b</dc:creator><dc:date>2012-07-02T20:33:42Z</dc:date><dc:rights>c</dc:rights><image>d</image></rss>'
        soup = self.soup(markup)
        assert str(soup.rss) == markup

    def test_docstring_includes_correct_encoding(self):
        soup = self.soup("<root/>")
        assert (
            soup.encode("latin1") == b'<?xml version="1.0" encoding="latin1"?>\n<root/>'
        )

    def test_large_xml_document(self):
        """A large XML document should come out the same as it went in."""
        markup = (
            b'<?xml version="1.0" encoding="utf-8"?>\n<root>'
            + b"0" * (2**12)
            + b"</root>"
        )
        soup = self.soup(markup)
        assert soup.encode("utf-8") == markup

    def test_tags_are_empty_element_if_and_only_if_they_are_empty(self):
        self.assert_soup("<p>", "<p/>")
        self.assert_soup("<p>foo</p>")

    def test_namespaces_are_preserved(self):
        markup = '<root xmlns:a="http://example.com/" xmlns:b="http://example.net/"><a:foo>This tag is in the a namespace</a:foo><b:foo>This tag is in the b namespace</b:foo></root>'
        soup = self.soup(markup)
        root = soup.root
        assert "http://example.com/" == root["xmlns:a"]
        assert "http://example.net/" == root["xmlns:b"]

    def test_closing_namespaced_tag(self):
        markup = '<p xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:date>20010504</dc:date></p>'
        soup = self.soup(markup)
        assert str(soup.p) == markup

    def test_namespaced_attributes(self):
        markup = '<foo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><bar xsi:schemaLocation="http://www.example.com"/></foo>'
        soup = self.soup(markup)
        assert str(soup.foo) == markup

    def test_namespaced_attributes_xml_namespace(self):
        markup = '<foo xml:lang="fr">bar</foo>'
        soup = self.soup(markup)
        assert str(soup.foo) == markup

    def test_find_by_prefixed_name(self):
        doc = """<?xml version="1.0" encoding="utf-8"?>
<Document xmlns="http://example.com/ns0"
    xmlns:ns1="http://example.com/ns1"
    xmlns:ns2="http://example.com/ns2">
    <ns1:tag>foo</ns1:tag>
    <ns1:tag>bar</ns1:tag>
    <ns2:tag key="value">baz</ns2:tag>
</Document>
"""
        soup = self.soup(doc)

        # There are three <tag> tags.
        assert 3 == len(soup.find_all("tag"))

        # But two of them are ns1:tag and one of them is ns2:tag.
        assert 2 == len(soup.find_all("ns1:tag"))
        assert 1 == len(soup.find_all("ns2:tag"))

        assert 1, len(soup.find_all("ns2:tag", key="value"))
        assert 3, len(soup.find_all(["ns1:tag", "ns2:tag"]))

    def test_copy_tag_preserves_namespace(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://example.com/ns0"/>"""

        soup = self.soup(xml)
        tag = soup.document
        duplicate = copy.copy(tag)

        # The two tags have the same namespace prefix.
        assert tag.prefix == duplicate.prefix

    def test_worst_case(self):
        """Test the worst case (currently) for linking issues."""

        soup = self.soup(BAD_DOCUMENT)
        self.linkage_validator(soup)


class HTML5TreeBuilderSmokeTest(HTMLTreeBuilderSmokeTest):
    """Smoke test for a tree builder that supports HTML5."""

    def test_real_xhtml_document(self):
        # Since XHTML is not HTML5, HTML5 parsers are not tested to handle
        # XHTML documents in any particular way.
        pass

    def test_html_tags_have_namespace(self):
        markup = "<a>"
        soup = self.soup(markup)
        assert "http://www.w3.org/1999/xhtml" == soup.a.namespace

    def test_svg_tags_have_namespace(self):
        markup = "<svg><circle/></svg>"
        soup = self.soup(markup)
        namespace = "http://www.w3.org/2000/svg"
        assert namespace == soup.svg.namespace
        assert namespace == soup.circle.namespace

    def test_mathml_tags_have_namespace(self):
        markup = "<math><msqrt>5</msqrt></math>"
        soup = self.soup(markup)
        namespace = "http://www.w3.org/1998/Math/MathML"
        assert namespace == soup.math.namespace
        assert namespace == soup.msqrt.namespace

    def test_xml_declaration_becomes_comment(self):
        markup = '<?xml version="1.0" encoding="utf-8"?><html></html>'
        soup = self.soup(markup)
        assert isinstance(soup.contents[0], Comment)
        assert soup.contents[0] == '?xml version="1.0" encoding="utf-8"?'
        assert "html" == soup.contents[0].next_element.name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_groupby.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby


@pytest.fixture
def times_frame():
    """Frame for testing times argument in EWM groupby."""
    return DataFrame(
        {
            "A": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "B": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            "C": to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-10",
                    "2020-01-22",
                    "2020-01-03",
                    "2020-01-23",
                    "2020-01-23",
                    "2020-01-04",
                ]
            ),
        }
    )


@pytest.fixture
def roll_frame():
    return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})


class TestRolling:
    def test_groupby_unsupported_argument(self, roll_frame):
        msg = r"groupby\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            roll_frame.groupby("A", foo=1)

    def test_getitem(self, roll_frame):
        g = roll_frame.groupby("A")
        g_mutated = get_groupby(roll_frame, by="A")

        expected = g_mutated.B.apply(lambda x: x.rolling(2).mean())

        result = g.rolling(2).mean().B
        tm.assert_series_equal(result, expected)

        result = g.rolling(2).B.mean()
        tm.assert_series_equal(result, expected)

        result = g.B.rolling(2).mean()
        tm.assert_series_equal(result, expected)

        result = roll_frame.B.groupby(roll_frame.A).rolling(2).mean()
        tm.assert_series_equal(result, expected)

    def test_getitem_multiple(self, roll_frame):
        # GH 13174
        g = roll_frame.groupby("A")
        r = g.rolling(2, min_periods=0)
        g_mutated = get_groupby(roll_frame, by="A")
        expected = g_mutated.B.apply(lambda x: x.rolling(2, min_periods=0).count())

        result = r.B.count()
        tm.assert_series_equal(result, expected)

        result = r.B.count()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "f",
        [
            "sum",
            "mean",
            "min",
            "max",
            "count",
            "kurt",
            "skew",
        ],
    )
    def test_rolling(self, f, roll_frame):
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result = getattr(r, f)()
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(lambda x: getattr(x.rolling(4), f)())
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["std", "var"])
    def test_rolling_ddof(self, f, roll_frame):
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result = getattr(r, f)(ddof=1)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(lambda x: getattr(x.rolling(4), f)(ddof=1))
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    def test_rolling_quantile(self, interpolation, roll_frame):
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result = r.quantile(0.4, interpolation=interpolation)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(
                lambda x: x.rolling(4).quantile(0.4, interpolation=interpolation)
            )
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f, expected_val", [["corr", 1], ["cov", 0.5]])
    def test_rolling_corr_cov_other_same_size_as_groups(self, f, expected_val):
        # GH 42915
        df = DataFrame(
            {"value": range(10), "idx1": [1] * 5 + [2] * 5, "idx2": [1, 2, 3, 4, 5] * 2}
        ).set_index(["idx1", "idx2"])
        other = DataFrame({"value": range(5), "idx2": [1, 2, 3, 4, 5]}).set_index(
            "idx2"
        )
        result = getattr(df.groupby(level=0).rolling(2), f)(other)
        expected_data = ([np.nan] + [expected_val] * 4) * 2
        expected = DataFrame(
            expected_data,
            columns=["value"],
            index=MultiIndex.from_arrays(
                [
                    [1] * 5 + [2] * 5,
                    [1] * 5 + [2] * 5,
                    list(range(1, 6)) * 2,
                ],
                names=["idx1", "idx1", "idx2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_other_diff_size_as_groups(self, f, roll_frame):
        g = roll_frame.groupby("A")
        r = g.rolling(window=4)

        result = getattr(r, f)(roll_frame)

        def func(x):
            return getattr(x.rolling(4), f)(roll_frame)

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(func)
        # GH 39591: The grouped column should be all np.nan
        # (groupby.apply inserts 0s for cov)
        expected["A"] = np.nan
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_pairwise(self, f, roll_frame):
        g = roll_frame.groupby("A")
        r = g.rolling(window=4)

        result = getattr(r.B, f)(pairwise=True)

        def func(x):
            return getattr(x.B.rolling(4), f)(pairwise=True)

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(func)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func, expected_values",
        [("cov", [[1.0, 1.0], [1.0, 4.0]]), ("corr", [[1.0, 0.5], [0.5, 1.0]])],
    )
    def test_rolling_corr_cov_unordered(self, func, expected_values):
        # GH 43386
        df = DataFrame(
            {
                "a": ["g1", "g2", "g1", "g1"],
                "b": [0, 0, 1, 2],
                "c": [2, 0, 6, 4],
            }
        )
        rol = df.groupby("a").rolling(3)
        result = getattr(rol, func)()
        expected = DataFrame(
            {
                "b": 4 * [np.nan] + expected_values[0] + 2 * [np.nan],
                "c": 4 * [np.nan] + expected_values[1] + 2 * [np.nan],
            },
            index=MultiIndex.from_tuples(
                [
                    ("g1", 0, "b"),
                    ("g1", 0, "c"),
                    ("g1", 2, "b"),
                    ("g1", 2, "c"),
                    ("g1", 3, "b"),
                    ("g1", 3, "c"),
                    ("g2", 1, "b"),
                    ("g2", 1, "c"),
                ],
                names=["a", None, None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply(self, raw, roll_frame):
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        # reduction
        result = r.apply(lambda x: x.sum(), raw=raw)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(lambda x: x.rolling(4).apply(lambda y: y.sum(), raw=raw))
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply_mutability(self):
        # GH 14013
        df = DataFrame({"A": ["foo"] * 3 + ["bar"] * 3, "B": [1] * 6})
        g = df.groupby("A")

        mi = MultiIndex.from_tuples(
            [("bar", 3), ("bar", 4), ("bar", 5), ("foo", 0), ("foo", 1), ("foo", 2)]
        )

        mi.names = ["A", None]
        # Grouped column should not be a part of the output
        expected = DataFrame([np.nan, 2.0, 2.0] * 2, columns=["B"], index=mi)

        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

        # Call an arbitrary function on the groupby
        g.sum()

        # Make sure nothing has been mutated
        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("expected_value,raw_value", [[1.0, True], [0.0, False]])
    def test_groupby_rolling(self, expected_value, raw_value):
        # GH 31754

        def isnumpyarray(x):
            return int(isinstance(x, np.ndarray))

        df = DataFrame({"id": [1, 1, 1], "value": [1, 2, 3]})
        result = df.groupby("id").value.rolling(1).apply(isnumpyarray, raw=raw_value)
        expected = Series(
            [expected_value] * 3,
            index=MultiIndex.from_tuples(((1, 0), (1, 1), (1, 2)), names=["id", None]),
            name="value",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_center_center(self):
        # GH 35552
        series = Series(range(1, 6))
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series(
            [np.nan] * 5,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3), (5, 4))),
        )
        tm.assert_series_equal(result, expected)

        series = Series(range(1, 5))
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series(
            [np.nan] * 4,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3))),
        )
        tm.assert_series_equal(result, expected)

        df = DataFrame({"a": ["a"] * 5 + ["b"] * 6, "b": range(11)})
        result = df.groupby("a").rolling(center=True, window=3).mean()
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                    ("b", 10),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": ["a"] * 5 + ["b"] * 5, "b": range(10)})
        result = df.groupby("a").rolling(center=True, window=3).mean()
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_center_on(self):
        # GH 37141
        df = DataFrame(
            data={
                "Date": date_range("2020-01-01", "2020-01-10"),
                "gb": ["group_1"] * 6 + ["group_2"] * 4,
                "value": range(10),
            }
        )
        result = (
            df.groupby("gb")
            .rolling(6, on="Date", center=True, min_periods=1)
            .value.mean()
        )
        mi = MultiIndex.from_arrays([df["gb"], df["Date"]], names=["gb", "Date"])
        expected = Series(
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0, 7.5, 7.5, 7.5],
            name="value",
            index=mi,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("min_periods", [5, 4, 3])
    def test_groupby_rolling_center_min_periods(self, min_periods):
        # GH 36040
        df = DataFrame({"group": ["A"] * 10 + ["B"] * 10, "data": range(20)})

        window_size = 5
        result = (
            df.groupby("group")
            .rolling(window_size, center=True, min_periods=min_periods)
            .mean()
        )
        result = result.reset_index()[["group", "data"]]

        grp_A_mean = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]
        grp_B_mean = [x + 10.0 for x in grp_A_mean]

        num_nans = max(0, min_periods - 3)  # For window_size of 5
        nans = [np.nan] * num_nans
        grp_A_expected = nans + grp_A_mean[num_nans : 10 - num_nans] + nans
        grp_B_expected = nans + grp_B_mean[num_nans : 10 - num_nans] + nans

        expected = DataFrame(
            {"group": ["A"] * 10 + ["B"] * 10, "data": grp_A_expected + grp_B_expected}
        )

        tm.assert_frame_equal(result, expected)

    def test_groupby_subselect_rolling(self):
        # GH 35486
        df = DataFrame(
            {"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0], "c": [10, 20, 30, 20]}
        )
        result = df.groupby("a")[["b"]].rolling(2).max()
        expected = DataFrame(
            [np.nan, np.nan, 2.0, np.nan],
            columns=["b"],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
        )
        tm.assert_frame_equal(result, expected)

        result = df.groupby("a")["b"].rolling(2).max()
        expected = Series(
            [np.nan, np.nan, 2.0, np.nan],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
            name="b",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_custom_indexer(self):
        # GH 35557
        class SimpleIndexer(BaseIndexer):
            def get_window_bounds(
                self,
                num_values=0,
                min_periods=None,
                center=None,
                closed=None,
                step=None,
            ):
                min_periods = self.window_size if min_periods is None else 0
                end = np.arange(num_values, dtype=np.int64) + 1
                start = end.copy() - self.window_size
                start[start < 0] = min_periods
                return start, end

        df = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 3}, index=[0] * 5 + [1] * 5 + [2] * 5
        )
        result = (
            df.groupby(df.index)
            .rolling(SimpleIndexer(window_size=3), min_periods=1)
            .sum()
        )
        expected = df.groupby(df.index).rolling(window=3, min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_subset_with_closed(self):
        # GH 35549
        df = DataFrame(
            {
                "column1": range(8),
                "column2": range(8),
                "group": ["A"] * 4 + ["B"] * 4,
                "date": [
                    Timestamp(date)
                    for date in ["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02"]
                ]
                * 2,
            }
        )
        result = (
            df.groupby("group").rolling("1D", on="date", closed="left")["column1"].sum()
        )
        expected = Series(
            [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0],
            index=MultiIndex.from_frame(
                df[["group", "date"]],
                names=["group", "date"],
            ),
            name="column1",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_subset_rolling_subset_with_closed(self):
        # GH 35549
        df = DataFrame(
            {
                "column1": range(8),
                "column2": range(8),
                "group": ["A"] * 4 + ["B"] * 4,
                "date": [
                    Timestamp(date)
                    for date in ["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02"]
                ]
                * 2,
            }
        )

        result = (
            df.groupby("group")[["column1", "date"]]
            .rolling("1D", on="date", closed="left")["column1"]
            .sum()
        )
        expected = Series(
            [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0],
            index=MultiIndex.from_frame(
                df[["group", "date"]],
                names=["group", "date"],
            ),
            name="column1",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", ["max", "min"])
    def test_groupby_rolling_index_changed(self, func):
        # GH: #36018 nlevels of MultiIndex changed
        ds = Series(
            [1, 2, 2],
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("c", "z")], names=["1", "2"]
            ),
            name="a",
        )

        result = getattr(ds.groupby(ds).rolling(2), func)()
        expected = Series(
            [np.nan, np.nan, 2.0],
            index=MultiIndex.from_tuples(
                [(1, "a", "x"), (2, "a", "y"), (2, "c", "z")], names=["a", "1", "2"]
            ),
            name="a",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_empty_frame(self):
        # GH 36197
        expected = DataFrame({"s1": []})
        result = expected.groupby("s1").rolling(window=1).sum()
        # GH 32262
        expected = expected.drop(columns="s1")
        # GH-38057 from_tuples gives empty object dtype, we now get float/int levels
        # expected.index = MultiIndex.from_tuples([], names=["s1", None])
        expected.index = MultiIndex.from_product(
            [Index([], dtype="float64"), Index([], dtype="int64")], names=["s1", None]
        )
        tm.assert_frame_equal(result, expected)

        expected = DataFrame({"s1": [], "s2": []})
        result = expected.groupby(["s1", "s2"]).rolling(window=1).sum()
        # GH 32262
        expected = expected.drop(columns=["s1", "s2"])
        expected.index = MultiIndex.from_product(
            [
                Index([], dtype="float64"),
                Index([], dtype="float64"),
                Index([], dtype="int64"),
            ],
            names=["s1", "s2", None],
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_string_index(self):
        # GH: 36727
        df = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9)],
                ["B", "group_1", Timestamp(2019, 1, 2, 9)],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9)],
                ["H", "group_1", Timestamp(2019, 1, 6, 9)],
                ["E", "group_2", Timestamp(2019, 1, 20, 9)],
            ],
            columns=["index", "group", "eventTime"],
        ).set_index("index")

        groups = df.groupby("group")
        df["count_to_date"] = groups.cumcount()
        rolling_groups = groups.rolling("10d", on="eventTime")
        result = rolling_groups.apply(lambda df: df.shape[0])
        expected = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9), 1.0],
                ["B", "group_1", Timestamp(2019, 1, 2, 9), 2.0],
                ["H", "group_1", Timestamp(2019, 1, 6, 9), 3.0],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9), 1.0],
                ["E", "group_2", Timestamp(2019, 1, 20, 9), 1.0],
            ],
            columns=["index", "group", "eventTime", "count_to_date"],
        ).set_index(["group", "index"])
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_no_sort(self):
        # GH 36889
        result = (
            DataFrame({"foo": [2, 1], "bar": [2, 1]})
            .groupby("foo", sort=False)
            .rolling(1)
            .min()
        )
        expected = DataFrame(
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            columns=["foo", "bar"],
            index=MultiIndex.from_tuples([(2, 0), (1, 1)], names=["foo", None]),
        )
        # GH 32262
        expected = expected.drop(columns="foo")
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_count_closed_on(self, unit):
        # GH 35869
        df = DataFrame(
            {
                "column1": range(6),
                "column2": range(6),
                "group": 3 * ["A", "B"],
                "date": date_range(end="20190101", periods=6, unit=unit),
            }
        )
        result = (
            df.groupby("group")
            .rolling("3d", on="date", closed="left")["column1"]
            .count()
        )
        dti = DatetimeIndex(
            [
                "2018-12-27",
                "2018-12-29",
                "2018-12-31",
                "2018-12-28",
                "2018-12-30",
                "2019-01-01",
            ],
            dtype=f"M8[{unit}]",
        )
        mi = MultiIndex.from_arrays(
            [
                ["A", "A", "A", "B", "B", "B"],
                dti,
            ],
            names=["group", "date"],
        )
        expected = Series(
            [np.nan, 1.0, 1.0, np.nan, 1.0, 1.0],
            name="column1",
            index=mi,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        ("func", "kwargs"),
        [("rolling", {"window": 2, "min_periods": 1}), ("expanding", {})],
    )
    def test_groupby_rolling_sem(self, func, kwargs):
        # GH: 26476
        df = DataFrame(
            [["a", 1], ["a", 2], ["b", 1], ["b", 2], ["b", 3]], columns=["a", "b"]
        )
        result = getattr(df.groupby("a"), func)(**kwargs).sem()
        expected = DataFrame(
            {"a": [np.nan] * 5, "b": [np.nan, 0.70711, np.nan, 0.70711, 0.70711]},
            index=MultiIndex.from_tuples(
                [("a", 0), ("a", 1), ("b", 2), ("b", 3), ("b", 4)], names=["a", None]
            ),
        )
        # GH 32262
        expected = expected.drop(columns="a")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("rollings", "key"), [({"on": "a"}, "a"), ({"on": None}, "index")]
    )
    def test_groupby_rolling_nans_in_index(self, rollings, key):
        # GH: 34617
        df = DataFrame(
            {
                "a": to_datetime(["2020-06-01 12:00", "2020-06-01 14:00", np.nan]),
                "b": [1, 2, 3],
                "c": [1, 1, 1],
            }
        )
        if key == "index":
            df = df.set_index("a")
        with pytest.raises(ValueError, match=f"{key} values must not have NaT"):
            df.groupby("c").rolling("60min", **rollings)

    @pytest.mark.parametrize("group_keys", [True, False])
    def test_groupby_rolling_group_keys(self, group_keys):
        # GH 37641
        # GH 38523: GH 37641 actually was not a bug.
        # group_keys only applies to groupby.apply directly
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        s = Series([1, 2, 3], index=index)
        result = s.groupby(["idx1", "idx2"], group_keys=group_keys).rolling(1).mean()
        expected = Series(
            [1.0, 2.0, 3.0],
            index=MultiIndex.from_tuples(
                [
                    ("val1", "val1", "val1", "val1"),
                    ("val1", "val1", "val1", "val1"),
                    ("val2", "val2", "val2", "val2"),
                ],
                names=["idx1", "idx2", "idx1", "idx2"],
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_index_level_and_column_label(self):
        # The groupby keys should not appear as a resulting column
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        df = DataFrame({"A": [1, 1, 2], "B": range(3)}, index=index)
        result = df.groupby(["idx1", "A"]).rolling(1).mean()
        expected = DataFrame(
            {"B": [0.0, 1.0, 2.0]},
            index=MultiIndex.from_tuples(
                [
                    ("val1", 1, "val1", "val1"),
                    ("val1", 1, "val1", "val1"),
                    ("val2", 2, "val2", "val2"),
                ],
                names=["idx1", "A", "idx1", "idx2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_resulting_multiindex(self):
        # a few different cases checking the created MultiIndex of the result
        # https://github.com/pandas-dev/pandas/pull/38057

        # grouping by 1 columns -> 2-level MI as result
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4})
        result = df.groupby("b").rolling(3).mean()
        expected_index = MultiIndex.from_tuples(
            [(1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)],
            names=["b", None],
        )
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex2(self):
        # grouping by 2 columns -> 3-level MI as result
        df = DataFrame({"a": np.arange(12.0), "b": [1, 2] * 6, "c": [1, 2, 3, 4] * 3})
        result = df.groupby(["b", "c"]).rolling(2).sum()
        expected_index = MultiIndex.from_tuples(
            [
                (1, 1, 0),
                (1, 1, 4),
                (1, 1, 8),
                (1, 3, 2),
                (1, 3, 6),
                (1, 3, 10),
                (2, 2, 1),
                (2, 2, 5),
                (2, 2, 9),
                (2, 4, 3),
                (2, 4, 7),
                (2, 4, 11),
            ],
            names=["b", "c", None],
        )
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex3(self):
        # grouping with 1 level on dataframe with 2-level MI -> 3-level MI as result
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4, "c": [1, 2, 3, 4] * 2})
        df = df.set_index("c", append=True)
        result = df.groupby("b").rolling(3).mean()
        expected_index = MultiIndex.from_tuples(
            [
                (1, 0, 1),
                (1, 2, 3),
                (1, 4, 1),
                (1, 6, 3),
                (2, 1, 2),
                (2, 3, 4),
                (2, 5, 2),
                (2, 7, 4),
            ],
            names=["b", None, "c"],
        )
        tm.assert_index_equal(result.index, expected_index, exact="equiv")

    def test_groupby_rolling_object_doesnt_affect_groupby_apply(self, roll_frame):
        # GH 39732
        g = roll_frame.groupby("A", group_keys=False)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(lambda x: x.rolling(4).sum()).index
        _ = g.rolling(window=4)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = g.apply(lambda x: x.rolling(4).sum()).index
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        ("window", "min_periods", "closed", "expected"),
        [
            (2, 0, "left", [None, 0.0, 1.0, 1.0, None, 0.0, 1.0, 1.0]),
            (2, 2, "left", [None, None, 1.0, 1.0, None, None, 1.0, 1.0]),
            (4, 4, "left", [None, None, None, None, None, None, None, None]),
            (4, 4, "right", [None, None, None, 5.0, None, None, None, 5.0]),
        ],
    )
    def test_groupby_rolling_var(self, window, min_periods, closed, expected):
        df = DataFrame([1, 2, 3, 4, 5, 6, 7, 8])
        result = (
            df.groupby([1, 2, 1, 2, 1, 2, 1, 2])
            .rolling(window=window, min_periods=min_periods, closed=closed)
            .var(0)
        )
        expected_result = DataFrame(
            np.array(expected, dtype="float64"),
            index=MultiIndex(
                levels=[np.array([1, 2]), [0, 1, 2, 3, 4, 5, 6, 7]],
                codes=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 2, 4, 6, 1, 3, 5, 7]],
            ),
        )
        tm.assert_frame_equal(result, expected_result)

    @pytest.mark.parametrize(
        "columns", [MultiIndex.from_tuples([("A", ""), ("B", "C")]), ["A", "B"]]
    )
    def test_by_column_not_in_values(self, columns):
        # GH 32262
        df = DataFrame([[1, 0]] * 20 + [[2, 0]] * 12 + [[3, 0]] * 8, columns=columns)
        g = df.groupby("A")
        original_obj = g.obj.copy(deep=True)
        r = g.rolling(4)
        result = r.sum()
        assert "A" not in result.columns
        tm.assert_frame_equal(g.obj, original_obj)

    def test_groupby_level(self):
        # GH 38523, 38787
        arrays = [
            ["Falcon", "Falcon", "Parrot", "Parrot"],
            ["Captive", "Wild", "Captive", "Wild"],
        ]
        index = MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
        df = DataFrame({"Max Speed": [390.0, 350.0, 30.0, 20.0]}, index=index)
        result = df.groupby(level=0)["Max Speed"].rolling(2).sum()
        expected = Series(
            [np.nan, 740.0, np.nan, 50.0],
            index=MultiIndex.from_tuples(
                [
                    ("Falcon", "Falcon", "Captive"),
                    ("Falcon", "Falcon", "Wild"),
                    ("Parrot", "Parrot", "Captive"),
                    ("Parrot", "Parrot", "Wild"),
                ],
                names=["Animal", "Animal", "Type"],
            ),
            name="Max Speed",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "by, expected_data",
        [
            [["id"], {"num": [100.0, 150.0, 150.0, 200.0]}],
            [
                ["id", "index"],
                {
                    "date": [
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                    ],
                    "num": [100.0, 200.0, 150.0, 250.0],
                },
            ],
        ],
    )
    def test_as_index_false(self, by, expected_data, unit):
        # GH 39433
        data = [
            ["A", "2018-01-01", 100.0],
            ["A", "2018-01-02", 200.0],
            ["B", "2018-01-01", 150.0],
            ["B", "2018-01-02", 250.0],
        ]
        df = DataFrame(data, columns=["id", "date", "num"])
        df["date"] = df["date"].astype(f"M8[{unit}]")
        df = df.set_index(["date"])

        gp_by = [getattr(df, attr) for attr in by]
        result = (
            df.groupby(gp_by, as_index=False).rolling(window=2, min_periods=1).mean()
        )

        expected = {"id": ["A", "A", "B", "B"]}
        expected.update(expected_data)
        expected = DataFrame(
            expected,
            index=df.index,
        )
        if "date" in expected_data:
            expected["date"] = expected["date"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(result, expected)

    def test_nan_and_zero_endpoints(self, any_int_numpy_dtype):
        # https://github.com/twosigma/pandas/issues/53
        typ = np.dtype(any_int_numpy_dtype).type
        size = 1000
        idx = np.repeat(typ(0), size)
        idx[-1] = 1

        val = 5e25
        arr = np.repeat(val, size)
        arr[0] = np.nan
        arr[-1] = 0

        df = DataFrame(
            {
                "index": idx,
                "adl2": arr,
            }
        ).set_index("index")
        result = df.groupby("index")["adl2"].rolling(window=10, min_periods=1).mean()
        expected = Series(
            arr,
            name="adl2",
            index=MultiIndex.from_arrays(
                [
                    Index([0] * 999 + [1], dtype=typ, name="index"),
                    Index([0] * 999 + [1], dtype=typ, name="index"),
                ],
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_non_monotonic(self):
        # GH 43909

        shuffled = [3, 0, 1, 2]
        sec = 1_000
        df = DataFrame(
            [{"t": Timestamp(2 * x * sec), "x": x + 1, "c": 42} for x in shuffled]
        )
        with pytest.raises(ValueError, match=r".* must be monotonic"):
            df.groupby("c").rolling(on="t", window="3s")

    def test_groupby_monotonic(self):
        # GH 15130
        # we don't need to validate monotonicity when grouping

        # GH 43909 we should raise an error here to match
        # behaviour of non-groupby rolling.

        data = [
            ["David", "1/1/2015", 100],
            ["David", "1/5/2015", 500],
            ["David", "5/30/2015", 50],
            ["David", "7/25/2015", 50],
            ["Ryan", "1/4/2014", 100],
            ["Ryan", "1/19/2015", 500],
            ["Ryan", "3/31/2016", 50],
            ["Joe", "7/1/2015", 100],
            ["Joe", "9/9/2015", 500],
            ["Joe", "10/15/2015", 50],
        ]

        df = DataFrame(data=data, columns=["name", "date", "amount"])
        df["date"] = to_datetime(df["date"])
        df = df.sort_values("date")

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = (
                df.set_index("date")
                .groupby("name")
                .apply(lambda x: x.rolling("180D")["amount"].sum())
            )
        result = df.groupby("name").rolling("180D", on="date")["amount"].sum()
        tm.assert_series_equal(result, expected)

    def test_datelike_on_monotonic_within_each_group(self):
        # GH 13966 (similar to #15130, closed by #15175)

        # superseded by 43909
        # GH 46061: OK if the on is monotonic relative to each each group

        dates = date_range(start="2016-01-01 09:30:00", periods=20, freq="s")
        df = DataFrame(
            {
                "A": [1] * 20 + [2] * 12 + [3] * 8,
                "B": np.concatenate((dates, dates)),
                "C": np.arange(40),
            }
        )

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = (
                df.set_index("B")
                .groupby("A")
                .apply(lambda x: x.rolling("4s")["C"].mean())
            )
        result = df.groupby("A").rolling("4s", on="B").C.mean()
        tm.assert_series_equal(result, expected)

    def test_datelike_on_not_monotonic_within_each_group(self):
        # GH 46061
        df = DataFrame(
            {
                "A": [1] * 3 + [2] * 3,
                "B": [Timestamp(year, 1, 1) for year in [2020, 2021, 2019]] * 2,
                "C": range(6),
            }
        )
        with pytest.raises(ValueError, match="Each group within B must be monotonic."):
            df.groupby("A").rolling("365D", on="B")


class TestExpanding:
    @pytest.fixture
    def frame(self):
        return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})

    @pytest.mark.parametrize(
        "f", ["sum", "mean", "min", "max", "count", "kurt", "skew"]
    )
    def test_expanding(self, f, frame):
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result = getattr(r, f)()
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(lambda x: getattr(x.expanding(), f)())
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["std", "var"])
    def test_expanding_ddof(self, f, frame):
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result = getattr(r, f)(ddof=0)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(lambda x: getattr(x.expanding(), f)(ddof=0))
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    def test_expanding_quantile(self, interpolation, frame):
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result = r.quantile(0.4, interpolation=interpolation)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(
                lambda x: x.expanding().quantile(0.4, interpolation=interpolation)
            )
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_expanding_corr_cov(self, f, frame):
        g = frame.groupby("A")
        r = g.expanding()

        result = getattr(r, f)(frame)

        def func_0(x):
            return getattr(x.expanding(), f)(frame)

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(func_0)
        # GH 39591: groupby.apply returns 1 instead of nan for windows
        # with all nan values
        null_idx = list(range(20, 61)) + list(range(72, 113))
        expected.iloc[null_idx, 1] = np.nan
        # GH 39591: The grouped column should be all np.nan
        # (groupby.apply inserts 0s for cov)
        expected["A"] = np.nan
        tm.assert_frame_equal(result, expected)

        result = getattr(r.B, f)(pairwise=True)

        def func_1(x):
            return getattr(x.B.expanding(), f)(pairwise=True)

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(func_1)
        tm.assert_series_equal(result, expected)

    def test_expanding_apply(self, raw, frame):
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        # reduction
        result = r.apply(lambda x: x.sum(), raw=raw)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = g.apply(
                lambda x: x.expanding().apply(lambda y: y.sum(), raw=raw)
            )
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)


class TestEWM:
    @pytest.mark.parametrize(
        "method, expected_data",
        [
            ["mean", [0.0, 0.6666666666666666, 1.4285714285714286, 2.2666666666666666]],
            ["std", [np.nan, 0.707107, 0.963624, 1.177164]],
            ["var", [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857]],
        ],
    )
    def test_methods(self, method, expected_data):
        # GH 16037
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        result = getattr(df.groupby("A").ewm(com=1.0), method)()
        expected = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "method, expected_data",
        [["corr", [np.nan, 1.0, 1.0, 1]], ["cov", [np.nan, 0.5, 0.928571, 1.385714]]],
    )
    def test_pairwise_methods(self, method, expected_data):
        # GH 16037
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        result = getattr(df.groupby("A").ewm(com=1.0), method)()
        expected = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0, "B"),
                    ("a", 1, "B"),
                    ("a", 2, "B"),
                    ("a", 3, "B"),
                ],
                names=["A", None, None],
            ),
        )
        tm.assert_frame_equal(result, expected)

        expected = df.groupby("A")[["B"]].apply(
            lambda x: getattr(x.ewm(com=1.0), method)()
        )
        tm.assert_frame_equal(result, expected)

    def test_times(self, times_frame):
        # GH 40951
        halflife = "23 days"
        # GH#42738
        times = times_frame.pop("C")
        result = times_frame.groupby("A").ewm(halflife=halflife, times=times).mean()
        expected = DataFrame(
            {
                "B": [
                    0.0,
                    0.507534,
                    1.020088,
                    1.537661,
                    0.0,
                    0.567395,
                    1.221209,
                    0.0,
                    0.653141,
                    1.195003,
                ]
            },
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 3),
                    ("a", 6),
                    ("a", 9),
                    ("b", 1),
                    ("b", 4),
                    ("b", 7),
                    ("c", 2),
                    ("c", 5),
                    ("c", 8),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_times_array(self, times_frame):
        # GH 40951
        halflife = "23 days"
        times = times_frame.pop("C")
        gb = times_frame.groupby("A")
        result = gb.ewm(halflife=halflife, times=times).mean()
        expected = gb.ewm(halflife=halflife, times=times.values).mean()
        tm.assert_frame_equal(result, expected)

    def test_dont_mutate_obj_after_slicing(self):
        # GH 43355
        df = DataFrame(
            {
                "id": ["a", "a", "b", "b", "b"],
                "timestamp": date_range("2021-9-1", periods=5, freq="h"),
                "y": range(5),
            }
        )
        grp = df.groupby("id").rolling("1h", on="timestamp")
        result = grp.count()
        expected_df = DataFrame(
            {
                "timestamp": date_range("2021-9-1", periods=5, freq="h"),
                "y": [1.0] * 5,
            },
            index=MultiIndex.from_arrays(
                [["a", "a", "b", "b", "b"], list(range(5))], names=["id", None]
            ),
        )
        tm.assert_frame_equal(result, expected_df)

        result = grp["y"].count()
        expected_series = Series(
            [1.0] * 5,
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "b", "b", "b"],
                    date_range("2021-9-1", periods=5, freq="h"),
                ],
                names=["id", "timestamp"],
            ),
            name="y",
        )
        tm.assert_series_equal(result, expected_series)
        # This is the key test
        result = grp.count()
        tm.assert_frame_equal(result, expected_df)


def test_rolling_corr_with_single_integer_in_index():
    # GH 44078
    df = DataFrame({"a": [(1,), (1,), (1,)], "b": [4, 5, 6]})
    gb = df.groupby(["a"])
    result = gb.rolling(2).corr(other=df)
    index = MultiIndex.from_tuples([((1,), 0), ((1,), 1), ((1,), 2)], names=["a", None])
    expected = DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [np.nan, 1.0, 1.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_corr_with_tuples_in_index():
    # GH 44078
    df = DataFrame(
        {
            "a": [
                (
                    1,
                    2,
                ),
                (
                    1,
                    2,
                ),
                (
                    1,
                    2,
                ),
            ],
            "b": [4, 5, 6],
        }
    )
    gb = df.groupby(["a"])
    result = gb.rolling(2).corr(other=df)
    index = MultiIndex.from_tuples(
        [((1, 2), 0), ((1, 2), 1), ((1, 2), 2)], names=["a", None]
    )
    expected = DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [np.nan, 1.0, 1.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)