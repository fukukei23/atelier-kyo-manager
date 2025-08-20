
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\control\control_plots.py ===
from sympy.core.numbers import I, pi
from sympy.functions.elementary.exponential import (exp, log)
from sympy.polys.partfrac import apart
from sympy.core.symbol import Dummy
from sympy.external import import_module
from sympy.functions import arg, Abs
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from sympy.plotting.series import LineOver1DRangeSeries
from sympy.plotting.plot import plot_parametric
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import Poly
from sympy.printing.latex import latex
from sympy.geometry.polygon import deg

__all__ = ['pole_zero_numerical_data', 'pole_zero_plot',
    'step_response_numerical_data', 'step_response_plot',
    'impulse_response_numerical_data', 'impulse_response_plot',
    'ramp_response_numerical_data', 'ramp_response_plot',
    'bode_magnitude_numerical_data', 'bode_phase_numerical_data',
    'bode_magnitude_plot', 'bode_phase_plot', 'bode_plot',
    'nyquist_plot_expr', 'nyquist_plot', 'nichols_plot_expr',
    'nichols_plot']


matplotlib = import_module(
        'matplotlib', import_kwargs={'fromlist': ['pyplot']},
        catch=(RuntimeError,))

if matplotlib:
    plt = matplotlib.pyplot


def _check_system(system):
    """Function to check whether the dynamical system passed for plots is
    compatible or not."""
    if not isinstance(system, SISOLinearTimeInvariant):
        raise NotImplementedError("Only SISO LTI systems are currently supported.")
    sys = system.to_expr()
    len_free_symbols = len(sys.free_symbols)
    if len_free_symbols > 1:
        raise ValueError("Extra degree of freedom found. Make sure"
            " that there are no free symbols in the dynamical system other"
            " than the variable of Laplace transform.")
    if sys.has(exp):
        # Should test that exp is not part of a constant, in which case
        # no exception is required, compare exp(s) with s*exp(1)
        raise NotImplementedError("Time delay terms are not supported.")


def _poly_roots(poly):
    """Function to get the roots of a polynomial."""
    def _eval(l):
        return [float(i) if i.is_real else complex(i) for i in l]
    if poly.domain in (QQ, ZZ):
        return _eval(poly.all_roots())
    # XXX: Use all_roots() for irrational coefficients when possible
    # See https://github.com/sympy/sympy/issues/22943
    return _eval(poly.nroots())


def pole_zero_numerical_data(system):
    """
    Returns the numerical data of poles and zeros of the system.
    It is internally used by ``pole_zero_plot`` to get the data
    for plotting poles and zeros. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the pole-zero data is to be computed.

    Returns
    =======

    tuple : (zeros, poles)
        zeros = Zeros of the system as a list of Python float/complex.
        poles = Poles of the system as a list of Python float/complex.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import pole_zero_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> pole_zero_numerical_data(tf1)
    ([-1j, 1j], [-2.0, -1.0, (-0.5-0.8660254037844386j), (-0.5+0.8660254037844386j)])

    See Also
    ========

    pole_zero_plot

    """
    _check_system(system)
    system = system.doit()  # Get the equivalent TransferFunction object.

    num_poly = Poly(system.num, system.var)
    den_poly = Poly(system.den, system.var)

    return _poly_roots(num_poly), _poly_roots(den_poly)


def pole_zero_plot(system, pole_color='blue', pole_markersize=10,
    zero_color='orange', zero_markersize=7, grid=True, show_axes=True,
    show=True, **kwargs):
    r"""
    Returns the Pole-Zero plot (also known as PZ Plot or PZ Map) of a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
    pole_color : str, tuple, optional
        The color of the pole points on the plot. Default color
        is blue. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    pole_markersize : Number, optional
        The size of the markers used to mark the poles in the plot.
        Default pole markersize is 10.
    zero_color : str, tuple, optional
        The color of the zero points on the plot. Default color
        is orange. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    zero_markersize : Number, optional
        The size of the markers used to mark the zeros in the plot.
        Default zero markersize is 7.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import pole_zero_plot
        >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        >>> pole_zero_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    pole_zero_numerical_data

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot

    """
    zeros, poles = pole_zero_numerical_data(system)

    zero_real = [i.real for i in zeros]
    zero_imag = [i.imag for i in zeros]

    pole_real = [i.real for i in poles]
    pole_imag = [i.imag for i in poles]

    plt.plot(pole_real, pole_imag, 'x', mfc='none',
        markersize=pole_markersize, color=pole_color)
    plt.plot(zero_real, zero_imag, 'o', markersize=zero_markersize,
        color=zero_color)
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title(f'Poles and Zeros of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def step_response_numerical_data(system, prec=8, lower_limit=0,
    upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the step response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the unit step response data is to be computed.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the step response. NumPy array.
        y = Amplitude-axis values of the points in the step response. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import step_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> step_response_numerical_data(tf1)   # doctest: +SKIP
    ([0.0, 0.025413462339411542, 0.0484508722725343, ... , 9.670250533855183, 9.844291913708725, 10.0],
    [0.0, 0.023844582399907256, 0.042894276802320226, ..., 6.828770759094287e-12, 6.456457160755703e-12])

    See Also
    ========

    step_response_plot

    """
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = system.to_expr()/(system.var)
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def step_response_plot(system, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the unit step response of a continuous-time system. It is
    the response of the system when the input signal is a step function.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Step Response is to be computed.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import step_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> step_response_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    impulse_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html

    """
    x, y = step_response_numerical_data(system, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Unit Step Response of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def impulse_response_numerical_data(system, prec=8, lower_limit=0,
    upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the impulse response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the impulse response data is to be computed.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the impulse response. NumPy array.
        y = Amplitude-axis values of the points in the impulse response. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import impulse_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> impulse_response_numerical_data(tf1)   # doctest: +SKIP
    ([0.0, 0.06616480200395854,... , 9.854500743565858, 10.0],
    [0.9999999799999999, 0.7042848373025861,...,7.170748906965121e-13, -5.1901263495547205e-12])

    See Also
    ========

    impulse_response_plot

    """
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = system.to_expr()
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def impulse_response_plot(system, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the unit impulse response (Input is the Dirac-Delta Function) of a
    continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Impulse Response is to be computed.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import impulse_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> impulse_response_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    step_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/dynamicsystem.impulse.html

    """
    x, y = impulse_response_numerical_data(system, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Impulse Response of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def ramp_response_numerical_data(system, slope=1, prec=8,
    lower_limit=0, upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the ramp response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the ramp response data is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the ramp response plot. NumPy array.
        y = Amplitude-axis values of the points in the ramp response plot. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

        When ``slope`` is negative.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import ramp_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> ramp_response_numerical_data(tf1)   # doctest: +SKIP
    (([0.0, 0.12166980856813935,..., 9.861246379582118, 10.0],
    [1.4504508011325967e-09, 0.006046440489058766,..., 0.12499999999568202, 0.12499999999661349]))

    See Also
    ========

    ramp_response_plot

    """
    if slope < 0:
        raise ValueError("Slope must be greater than or equal"
            " to zero.")
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = (slope*system.to_expr())/((system.var)**2)
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def ramp_response_plot(system, slope=1, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line
    passing through origin ($f(x) = mx$). The slope of
    the ramp function can be varied by the user and
    the default value is 1.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Ramp Response is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import ramp_response_plot
        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)
        >>> ramp_response_plot(tf1, upper_limit=2)   # doctest: +SKIP

    See Also
    ========

    step_response_plot, impulse_response_plot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    x, y = ramp_response_numerical_data(system, slope=slope, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Ramp Response of ${latex(system)}$ [Slope = {slope}]', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_magnitude_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', **kwargs):
    """
    Returns the numerical data of the Bode magnitude plot of the system.
    It is internally used by ``bode_magnitude_plot`` to get the data
    for plotting Bode magnitude plot. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the data is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.

    Returns
    =======

    tuple : (x, y)
        x = x-axis values of the Bode magnitude plot.
        y = y-axis values of the Bode magnitude plot.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When incorrect frequency units are given as input.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import bode_magnitude_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> bode_magnitude_numerical_data(tf1)   # doctest: +SKIP
    ([1e-05, 1.5148378120533502e-05,..., 68437.36188804005, 100000.0],
    [-6.020599914256786, -6.0205999155219505,..., -193.4117304087953, -200.00000000260573])

    See Also
    ========

    bode_magnitude_plot, bode_phase_numerical_data

    """
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')

    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    mag = 20*log(Abs(w_expr), 10)

    x, y = LineOver1DRangeSeries(mag,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()

    return x, y


def bode_magnitude_plot(system, initial_exp=-5, final_exp=5,
    color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', **kwargs):
    r"""
    Returns the Bode magnitude plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    x, y = bode_magnitude_numerical_data(system, initial_exp=initial_exp,
        final_exp=final_exp, freq_unit=freq_unit)
    plt.plot(x, y, color=color, **kwargs)
    plt.xscale('log')


    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Bode Plot (Magnitude) of ${latex(system)}$', pad=20)

    if grid:
        plt.grid(True)
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_phase_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', phase_unit='rad', phase_unwrap = True, **kwargs):
    """
    Returns the numerical data of the Bode phase plot of the system.
    It is internally used by ``bode_phase_plot`` to get the data
    for plotting Bode phase plot. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the Bode phase plot data is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and '``'Hz'`` (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.
    phase_unwrap : bool, optional
        Set to ``True`` by default.

    Returns
    =======

    tuple : (x, y)
        x = x-axis values of the Bode phase plot.
        y = y-axis values of the Bode phase plot.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When incorrect frequency or phase units are given as input.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import bode_phase_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> bode_phase_numerical_data(tf1)   # doctest: +SKIP
    ([1e-05, 1.4472354033813751e-05, 2.035581932165858e-05,..., 47577.3248186011, 67884.09326036123, 100000.0],
    [-2.5000000000291665e-05, -3.6180885085e-05, -5.08895483066e-05,...,-3.1415085799262523, -3.14155265358979])

    See Also
    ========

    bode_magnitude_plot, bode_phase_numerical_data

    """
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    phase_units = ('rad', 'deg')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')
    if phase_unit not in phase_units:
        raise ValueError('Only "rad" and "deg" are accepted phase units.')

    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    if phase_unit == 'deg':
        phase = arg(w_expr)*180/pi
    else:
        phase = arg(w_expr)

    x, y = LineOver1DRangeSeries(phase,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()

    half = None
    if phase_unwrap:
        if(phase_unit == 'rad'):
            half = pi
        elif(phase_unit == 'deg'):
            half = 180
    if half:
        unit = 2*half
        for i in range(1, len(y)):
            diff = y[i] - y[i - 1]
            if diff > half:      # Jump from -half to half
                y[i] = (y[i] - unit)
            elif diff < -half:   # Jump from half to -half
                y[i] = (y[i] + unit)

    return x, y


def bode_phase_plot(system, initial_exp=-5, final_exp=5,
    color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    r"""
    Returns the Bode phase plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    x, y = bode_phase_numerical_data(system, initial_exp=initial_exp,
        final_exp=final_exp, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap)
    plt.plot(x, y, color=color, **kwargs)
    plt.xscale('log')

    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    plt.ylabel('Phase (%s)' % phase_unit)
    plt.title(f'Bode Plot (Phase) of ${latex(system)}$', pad=20)

    if grid:
        plt.grid(True)
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_plot(system, initial_exp=-5, final_exp=5,
    grid=True, show_axes=False, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    r"""
    Returns the Bode phase and magnitude plots of a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Bode Plot is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import bode_plot
        >>> tf1 = TransferFunction(1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        >>> bode_plot(tf1, initial_exp=0.2, final_exp=0.7)   # doctest: +SKIP

    See Also
    ========

    bode_magnitude_plot, bode_phase_plot

    """
    plt.subplot(211)
    mag = bode_magnitude_plot(system, initial_exp=initial_exp, final_exp=final_exp,
        show=False, grid=grid, show_axes=show_axes,
        freq_unit=freq_unit, **kwargs)
    mag.title(f'Bode Plot of ${latex(system)}$', pad=20)
    mag.xlabel(None)
    plt.subplot(212)
    bode_phase_plot(system, initial_exp=initial_exp, final_exp=final_exp,
        show=False, grid=grid, show_axes=show_axes, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap, **kwargs).title(None)

    if show:
        plt.show()
        return

    return plt


def nyquist_plot_expr(system):
    """Function to get the expression for Nyquist plot."""
    s = system.var
    w = Dummy('w', real=True)
    repl = I * w
    expr = system.to_expr()
    w_expr = expr.subs({s: repl})
    w_expr = w_expr.as_real_imag()
    real_expr = w_expr[0]
    imag_expr = w_expr[1]
    return real_expr, imag_expr, w


def nichols_plot_expr(system):
    """Function to get the expression for Nichols plot."""
    s = system.var
    w = Dummy('w', real=True)
    sys_expr = system.to_expr()
    H_jw = sys_expr.subs(s, I*w)
    mag_expr = Abs(H_jw)
    mag_dB_expr = 20*log(mag_expr, 10)
    phase_expr = arg(H_jw)
    phase_deg_expr = deg(phase_expr)
    return mag_dB_expr, phase_deg_expr, w


def nyquist_plot(system, initial_omega=0.01, final_omega=100, show=True,
                color='b', **kwargs):
    r"""
    Generates the Nyquist plot for a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The LTI SISO system for which the Nyquist plot is to be generated.
    initial_omega : float, optional
        The starting frequency value. Defaults to 0.01.
    final_omega : float, optional
        The ending frequency value. Defaults to 100.
    show : bool, optional
        If True, the plot is displayed. Default is True.
    color : str, optional
        The color of the Nyquist plot. Default is 'b' (blue).
    grid : bool, optional
        If True, grid lines are displayed. Default is False.
    **kwargs
        Additional keyword arguments for customization.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import nyquist_plot
        >>> tf1 = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)
        >>> nyquist_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    nichols_plot, bode_plot

    """
    _check_system(system)
    real_expr, imag_expr, w = nyquist_plot_expr(system)
    w_values = [(w, initial_omega, final_omega)]
    p = plot_parametric(
        (real_expr, imag_expr),   # The curve
        (real_expr, -imag_expr),  # Its mirror image
        *w_values,
        show=False,
        line_color=color,
        adaptive=True,
        title=f'Nyquist Plot of ${latex(system)}$',
        xlabel='Real Axis',
        ylabel='Imaginary Axis',
        size=(6, 5),
        kwargs=kwargs)
    if show:
        p.show()
        return
    return p


def nichols_plot(system, initial_omega=0.01, final_omega=100, show=True, color='b', **kwargs):
    r"""
    Generates the Nichols plot for a LTI system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The LTI SISO system for which the Nyquist plot is to be generated.
    initial_omega : float, optional
        The starting frequency value. Defaults to 0.01.
    final_omega : float, optional
        The ending frequency value. Defaults to 100.
    show : bool, optional
        If True, the plot is displayed. Default is True.
    color : str, optional
        The color of the Nyquist plot. Default is 'b' (blue).
    grid : bool, optional
        If True, grid lines are displayed. Default is False.
    **kwargs
        Additional keyword arguments for customization.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import nichols_plot
        >>> tf1 = TransferFunction(1.5, s**2+14*s+40.02, s)
        >>> nichols_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    nyquist_plot, bode_plot

    """
    _check_system(system)
    magnitude_dB_expr, phase_deg_expr, w = nichols_plot_expr(system)
    w_values = [(w, initial_omega, final_omega)]
    p = plot_parametric(
        (phase_deg_expr, magnitude_dB_expr),
        *w_values,
        show=False,
        line_color=color,
        title=f'Nichols Plot of ${latex(system)}$',
        xlabel='Phase [deg]',
        ylabel='Magnitude [dB]',
        size=(6,5),
        kwargs=kwargs)
    if show:
        p.show()
        return
    return p

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\cmdoptions.py ===
"""
shared options and groups

The principle here is to define options once, but *not* instantiate them
globally. One reason being that options with action='append' can carry state
between parses. pip parses general options twice internally, and shouldn't
pass on state. To be consistent, all options will follow this design.
"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False

import importlib.util
import logging
import os
import pathlib
import textwrap
from functools import partial
from optparse import SUPPRESS_HELP, Option, OptionGroup, OptionParser, Values
from textwrap import dedent
from typing import Any, Callable, Dict, Optional, Tuple

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.cli.parser import ConfigOptionParser
from pip._internal.exceptions import CommandError
from pip._internal.locations import USER_CACHE_DIR, get_src_prefix
from pip._internal.models.format_control import FormatControl
from pip._internal.models.index import PyPI
from pip._internal.models.target_python import TargetPython
from pip._internal.utils.hashes import STRONG_HASHES
from pip._internal.utils.misc import strtobool

logger = logging.getLogger(__name__)


def raise_option_error(parser: OptionParser, option: Option, msg: str) -> None:
    """
    Raise an option parsing error using parser.error().

    Args:
      parser: an OptionParser instance.
      option: an Option instance.
      msg: the error text.
    """
    msg = f"{option} error: {msg}"
    msg = textwrap.fill(" ".join(msg.split()))
    parser.error(msg)


def make_option_group(group: Dict[str, Any], parser: ConfigOptionParser) -> OptionGroup:
    """
    Return an OptionGroup object
    group  -- assumed to be dict with 'name' and 'options' keys
    parser -- an optparse Parser
    """
    option_group = OptionGroup(parser, group["name"])
    for option in group["options"]:
        option_group.add_option(option())
    return option_group


def check_dist_restriction(options: Values, check_target: bool = False) -> None:
    """Function for determining if custom platform options are allowed.

    :param options: The OptionParser options.
    :param check_target: Whether or not to check if --target is being used.
    """
    dist_restriction_set = any(
        [
            options.python_version,
            options.platforms,
            options.abis,
            options.implementation,
        ]
    )

    binary_only = FormatControl(set(), {":all:"})
    sdist_dependencies_allowed = (
        options.format_control != binary_only and not options.ignore_dependencies
    )

    # Installations or downloads using dist restrictions must not combine
    # source distributions and dist-specific wheels, as they are not
    # guaranteed to be locally compatible.
    if dist_restriction_set and sdist_dependencies_allowed:
        raise CommandError(
            "When restricting platform and interpreter constraints using "
            "--python-version, --platform, --abi, or --implementation, "
            "either --no-deps must be set, or --only-binary=:all: must be "
            "set and --no-binary must not be set (or must be set to "
            ":none:)."
        )

    if check_target:
        if not options.dry_run and dist_restriction_set and not options.target_dir:
            raise CommandError(
                "Can not use any platform or abi specific options unless "
                "installing via '--target' or using '--dry-run'"
            )


def _path_option_check(option: Option, opt: str, value: str) -> str:
    return os.path.expanduser(value)


def _package_name_option_check(option: Option, opt: str, value: str) -> str:
    return canonicalize_name(value)


class PipOption(Option):
    TYPES = Option.TYPES + ("path", "package_name")
    TYPE_CHECKER = Option.TYPE_CHECKER.copy()
    TYPE_CHECKER["package_name"] = _package_name_option_check
    TYPE_CHECKER["path"] = _path_option_check


###########
# options #
###########

help_: Callable[..., Option] = partial(
    Option,
    "-h",
    "--help",
    dest="help",
    action="help",
    help="Show help.",
)

debug_mode: Callable[..., Option] = partial(
    Option,
    "--debug",
    dest="debug_mode",
    action="store_true",
    default=False,
    help=(
        "Let unhandled exceptions propagate outside the main subroutine, "
        "instead of logging them to stderr."
    ),
)

isolated_mode: Callable[..., Option] = partial(
    Option,
    "--isolated",
    dest="isolated_mode",
    action="store_true",
    default=False,
    help=(
        "Run pip in an isolated mode, ignoring environment variables and user "
        "configuration."
    ),
)

require_virtualenv: Callable[..., Option] = partial(
    Option,
    "--require-virtualenv",
    "--require-venv",
    dest="require_venv",
    action="store_true",
    default=False,
    help=(
        "Allow pip to only run in a virtual environment; "
        "exit with an error otherwise."
    ),
)

override_externally_managed: Callable[..., Option] = partial(
    Option,
    "--break-system-packages",
    dest="override_externally_managed",
    action="store_true",
    help="Allow pip to modify an EXTERNALLY-MANAGED Python installation",
)

python: Callable[..., Option] = partial(
    Option,
    "--python",
    dest="python",
    help="Run pip with the specified Python interpreter.",
)

verbose: Callable[..., Option] = partial(
    Option,
    "-v",
    "--verbose",
    dest="verbose",
    action="count",
    default=0,
    help="Give more output. Option is additive, and can be used up to 3 times.",
)

no_color: Callable[..., Option] = partial(
    Option,
    "--no-color",
    dest="no_color",
    action="store_true",
    default=False,
    help="Suppress colored output.",
)

version: Callable[..., Option] = partial(
    Option,
    "-V",
    "--version",
    dest="version",
    action="store_true",
    help="Show version and exit.",
)

quiet: Callable[..., Option] = partial(
    Option,
    "-q",
    "--quiet",
    dest="quiet",
    action="count",
    default=0,
    help=(
        "Give less output. Option is additive, and can be used up to 3"
        " times (corresponding to WARNING, ERROR, and CRITICAL logging"
        " levels)."
    ),
)

progress_bar: Callable[..., Option] = partial(
    Option,
    "--progress-bar",
    dest="progress_bar",
    type="choice",
    choices=["on", "off", "raw"],
    default="on",
    help="Specify whether the progress bar should be used [on, off, raw] (default: on)",
)

log: Callable[..., Option] = partial(
    PipOption,
    "--log",
    "--log-file",
    "--local-log",
    dest="log",
    metavar="path",
    type="path",
    help="Path to a verbose appending log.",
)

no_input: Callable[..., Option] = partial(
    Option,
    # Don't ask for input
    "--no-input",
    dest="no_input",
    action="store_true",
    default=False,
    help="Disable prompting for input.",
)

keyring_provider: Callable[..., Option] = partial(
    Option,
    "--keyring-provider",
    dest="keyring_provider",
    choices=["auto", "disabled", "import", "subprocess"],
    default="auto",
    help=(
        "Enable the credential lookup via the keyring library if user input is allowed."
        " Specify which mechanism to use [auto, disabled, import, subprocess]."
        " (default: %default)"
    ),
)

proxy: Callable[..., Option] = partial(
    Option,
    "--proxy",
    dest="proxy",
    type="str",
    default="",
    help="Specify a proxy in the form scheme://[user:passwd@]proxy.server:port.",
)

retries: Callable[..., Option] = partial(
    Option,
    "--retries",
    dest="retries",
    type="int",
    default=5,
    help="Maximum attempts to establish a new HTTP connection. (default: %default)",
)

resume_retries: Callable[..., Option] = partial(
    Option,
    "--resume-retries",
    dest="resume_retries",
    type="int",
    default=0,
    help="Maximum attempts to resume or restart an incomplete download. "
    "(default: %default)",
)

timeout: Callable[..., Option] = partial(
    Option,
    "--timeout",
    "--default-timeout",
    metavar="sec",
    dest="timeout",
    type="float",
    default=15,
    help="Set the socket timeout (default %default seconds).",
)


def exists_action() -> Option:
    return Option(
        # Option when path already exist
        "--exists-action",
        dest="exists_action",
        type="choice",
        choices=["s", "i", "w", "b", "a"],
        default=[],
        action="append",
        metavar="action",
        help="Default action when a path already exists: "
        "(s)witch, (i)gnore, (w)ipe, (b)ackup, (a)bort.",
    )


cert: Callable[..., Option] = partial(
    PipOption,
    "--cert",
    dest="cert",
    type="path",
    metavar="path",
    help=(
        "Path to PEM-encoded CA certificate bundle. "
        "If provided, overrides the default. "
        "See 'SSL Certificate Verification' in pip documentation "
        "for more information."
    ),
)

client_cert: Callable[..., Option] = partial(
    PipOption,
    "--client-cert",
    dest="client_cert",
    type="path",
    default=None,
    metavar="path",
    help="Path to SSL client certificate, a single file containing the "
    "private key and the certificate in PEM format.",
)

index_url: Callable[..., Option] = partial(
    Option,
    "-i",
    "--index-url",
    "--pypi-url",
    dest="index_url",
    metavar="URL",
    default=PyPI.simple_url,
    help="Base URL of the Python Package Index (default %default). "
    "This should point to a repository compliant with PEP 503 "
    "(the simple repository API) or a local directory laid out "
    "in the same format.",
)


def extra_index_url() -> Option:
    return Option(
        "--extra-index-url",
        dest="extra_index_urls",
        metavar="URL",
        action="append",
        default=[],
        help="Extra URLs of package indexes to use in addition to "
        "--index-url. Should follow the same rules as "
        "--index-url.",
    )


no_index: Callable[..., Option] = partial(
    Option,
    "--no-index",
    dest="no_index",
    action="store_true",
    default=False,
    help="Ignore package index (only looking at --find-links URLs instead).",
)


def find_links() -> Option:
    return Option(
        "-f",
        "--find-links",
        dest="find_links",
        action="append",
        default=[],
        metavar="url",
        help="If a URL or path to an html file, then parse for links to "
        "archives such as sdist (.tar.gz) or wheel (.whl) files. "
        "If a local path or file:// URL that's a directory, "
        "then look for archives in the directory listing. "
        "Links to VCS project URLs are not supported.",
    )


def trusted_host() -> Option:
    return Option(
        "--trusted-host",
        dest="trusted_hosts",
        action="append",
        metavar="HOSTNAME",
        default=[],
        help="Mark this host or host:port pair as trusted, even though it "
        "does not have valid or any HTTPS.",
    )


def constraints() -> Option:
    return Option(
        "-c",
        "--constraint",
        dest="constraints",
        action="append",
        default=[],
        metavar="file",
        help="Constrain versions using the given constraints file. "
        "This option can be used multiple times.",
    )


def requirements() -> Option:
    return Option(
        "-r",
        "--requirement",
        dest="requirements",
        action="append",
        default=[],
        metavar="file",
        help="Install from the given requirements file. "
        "This option can be used multiple times.",
    )


def editable() -> Option:
    return Option(
        "-e",
        "--editable",
        dest="editables",
        action="append",
        default=[],
        metavar="path/url",
        help=(
            "Install a project in editable mode (i.e. setuptools "
            '"develop mode") from a local project path or a VCS url.'
        ),
    )


def _handle_src(option: Option, opt_str: str, value: str, parser: OptionParser) -> None:
    value = os.path.abspath(value)
    setattr(parser.values, option.dest, value)


src: Callable[..., Option] = partial(
    PipOption,
    "--src",
    "--source",
    "--source-dir",
    "--source-directory",
    dest="src_dir",
    type="path",
    metavar="dir",
    default=get_src_prefix(),
    action="callback",
    callback=_handle_src,
    help="Directory to check out editable projects into. "
    'The default in a virtualenv is "<venv path>/src". '
    'The default for global installs is "<current dir>/src".',
)


def _get_format_control(values: Values, option: Option) -> Any:
    """Get a format_control object."""
    return getattr(values, option.dest)


def _handle_no_binary(
    option: Option, opt_str: str, value: str, parser: OptionParser
) -> None:
    existing = _get_format_control(parser.values, option)
    FormatControl.handle_mutual_excludes(
        value,
        existing.no_binary,
        existing.only_binary,
    )


def _handle_only_binary(
    option: Option, opt_str: str, value: str, parser: OptionParser
) -> None:
    existing = _get_format_control(parser.values, option)
    FormatControl.handle_mutual_excludes(
        value,
        existing.only_binary,
        existing.no_binary,
    )


def no_binary() -> Option:
    format_control = FormatControl(set(), set())
    return Option(
        "--no-binary",
        dest="format_control",
        action="callback",
        callback=_handle_no_binary,
        type="str",
        default=format_control,
        help="Do not use binary packages. Can be supplied multiple times, and "
        'each time adds to the existing value. Accepts either ":all:" to '
        'disable all binary packages, ":none:" to empty the set (notice '
        "the colons), or one or more package names with commas between "
        "them (no colons). Note that some packages are tricky to compile "
        "and may fail to install when this option is used on them.",
    )


def only_binary() -> Option:
    format_control = FormatControl(set(), set())
    return Option(
        "--only-binary",
        dest="format_control",
        action="callback",
        callback=_handle_only_binary,
        type="str",
        default=format_control,
        help="Do not use source packages. Can be supplied multiple times, and "
        'each time adds to the existing value. Accepts either ":all:" to '
        'disable all source packages, ":none:" to empty the set, or one '
        "or more package names with commas between them. Packages "
        "without binary distributions will fail to install when this "
        "option is used on them.",
    )


platforms: Callable[..., Option] = partial(
    Option,
    "--platform",
    dest="platforms",
    metavar="platform",
    action="append",
    default=None,
    help=(
        "Only use wheels compatible with <platform>. Defaults to the "
        "platform of the running system. Use this option multiple times to "
        "specify multiple platforms supported by the target interpreter."
    ),
)


# This was made a separate function for unit-testing purposes.
def _convert_python_version(value: str) -> Tuple[Tuple[int, ...], Optional[str]]:
    """
    Convert a version string like "3", "37", or "3.7.3" into a tuple of ints.

    :return: A 2-tuple (version_info, error_msg), where `error_msg` is
        non-None if and only if there was a parsing error.
    """
    if not value:
        # The empty string is the same as not providing a value.
        return (None, None)

    parts = value.split(".")
    if len(parts) > 3:
        return ((), "at most three version parts are allowed")

    if len(parts) == 1:
        # Then we are in the case of "3" or "37".
        value = parts[0]
        if len(value) > 1:
            parts = [value[0], value[1:]]

    try:
        version_info = tuple(int(part) for part in parts)
    except ValueError:
        return ((), "each version part must be an integer")

    return (version_info, None)


def _handle_python_version(
    option: Option, opt_str: str, value: str, parser: OptionParser
) -> None:
    """
    Handle a provided --python-version value.
    """
    version_info, error_msg = _convert_python_version(value)
    if error_msg is not None:
        msg = f"invalid --python-version value: {value!r}: {error_msg}"
        raise_option_error(parser, option=option, msg=msg)

    parser.values.python_version = version_info


python_version: Callable[..., Option] = partial(
    Option,
    "--python-version",
    dest="python_version",
    metavar="python_version",
    action="callback",
    callback=_handle_python_version,
    type="str",
    default=None,
    help=dedent(
        """\
    The Python interpreter version to use for wheel and "Requires-Python"
    compatibility checks. Defaults to a version derived from the running
    interpreter. The version can be specified using up to three dot-separated
    integers (e.g. "3" for 3.0.0, "3.7" for 3.7.0, or "3.7.3"). A major-minor
    version can also be given as a string without dots (e.g. "37" for 3.7.0).
    """
    ),
)


implementation: Callable[..., Option] = partial(
    Option,
    "--implementation",
    dest="implementation",
    metavar="implementation",
    default=None,
    help=(
        "Only use wheels compatible with Python "
        "implementation <implementation>, e.g. 'pp', 'jy', 'cp', "
        " or 'ip'. If not specified, then the current "
        "interpreter implementation is used.  Use 'py' to force "
        "implementation-agnostic wheels."
    ),
)


abis: Callable[..., Option] = partial(
    Option,
    "--abi",
    dest="abis",
    metavar="abi",
    action="append",
    default=None,
    help=(
        "Only use wheels compatible with Python abi <abi>, e.g. 'pypy_41'. "
        "If not specified, then the current interpreter abi tag is used. "
        "Use this option multiple times to specify multiple abis supported "
        "by the target interpreter. Generally you will need to specify "
        "--implementation, --platform, and --python-version when using this "
        "option."
    ),
)


def add_target_python_options(cmd_opts: OptionGroup) -> None:
    cmd_opts.add_option(platforms())
    cmd_opts.add_option(python_version())
    cmd_opts.add_option(implementation())
    cmd_opts.add_option(abis())


def make_target_python(options: Values) -> TargetPython:
    target_python = TargetPython(
        platforms=options.platforms,
        py_version_info=options.python_version,
        abis=options.abis,
        implementation=options.implementation,
    )

    return target_python


def prefer_binary() -> Option:
    return Option(
        "--prefer-binary",
        dest="prefer_binary",
        action="store_true",
        default=False,
        help=(
            "Prefer binary packages over source packages, even if the "
            "source packages are newer."
        ),
    )


cache_dir: Callable[..., Option] = partial(
    PipOption,
    "--cache-dir",
    dest="cache_dir",
    default=USER_CACHE_DIR,
    metavar="dir",
    type="path",
    help="Store the cache data in <dir>.",
)


def _handle_no_cache_dir(
    option: Option, opt: str, value: str, parser: OptionParser
) -> None:
    """
    Process a value provided for the --no-cache-dir option.

    This is an optparse.Option callback for the --no-cache-dir option.
    """
    # The value argument will be None if --no-cache-dir is passed via the
    # command-line, since the option doesn't accept arguments.  However,
    # the value can be non-None if the option is triggered e.g. by an
    # environment variable, like PIP_NO_CACHE_DIR=true.
    if value is not None:
        # Then parse the string value to get argument error-checking.
        try:
            strtobool(value)
        except ValueError as exc:
            raise_option_error(parser, option=option, msg=str(exc))

    # Originally, setting PIP_NO_CACHE_DIR to a value that strtobool()
    # converted to 0 (like "false" or "no") caused cache_dir to be disabled
    # rather than enabled (logic would say the latter).  Thus, we disable
    # the cache directory not just on values that parse to True, but (for
    # backwards compatibility reasons) also on values that parse to False.
    # In other words, always set it to False if the option is provided in
    # some (valid) form.
    parser.values.cache_dir = False


no_cache: Callable[..., Option] = partial(
    Option,
    "--no-cache-dir",
    dest="cache_dir",
    action="callback",
    callback=_handle_no_cache_dir,
    help="Disable the cache.",
)

no_deps: Callable[..., Option] = partial(
    Option,
    "--no-deps",
    "--no-dependencies",
    dest="ignore_dependencies",
    action="store_true",
    default=False,
    help="Don't install package dependencies.",
)


def _handle_dependency_group(
    option: Option, opt: str, value: str, parser: OptionParser
) -> None:
    """
    Process a value provided for the --group option.

    Splits on the rightmost ":", and validates that the path (if present) ends
    in `pyproject.toml`. Defaults the path to `pyproject.toml` when one is not given.

    `:` cannot appear in dependency group names, so this is a safe and simple parse.

    This is an optparse.Option callback for the dependency_groups option.
    """
    path, sep, groupname = value.rpartition(":")
    if not sep:
        path = "pyproject.toml"
    else:
        # check for 'pyproject.toml' filenames using pathlib
        if pathlib.PurePath(path).name != "pyproject.toml":
            msg = "group paths use 'pyproject.toml' filenames"
            raise_option_error(parser, option=option, msg=msg)

    parser.values.dependency_groups.append((path, groupname))


dependency_groups: Callable[..., Option] = partial(
    Option,
    "--group",
    dest="dependency_groups",
    default=[],
    type=str,
    action="callback",
    callback=_handle_dependency_group,
    metavar="[path:]group",
    help='Install a named dependency-group from a "pyproject.toml" file. '
    'If a path is given, the name of the file must be "pyproject.toml". '
    'Defaults to using "pyproject.toml" in the current directory.',
)

ignore_requires_python: Callable[..., Option] = partial(
    Option,
    "--ignore-requires-python",
    dest="ignore_requires_python",
    action="store_true",
    help="Ignore the Requires-Python information.",
)

no_build_isolation: Callable[..., Option] = partial(
    Option,
    "--no-build-isolation",
    dest="build_isolation",
    action="store_false",
    default=True,
    help="Disable isolation when building a modern source distribution. "
    "Build dependencies specified by PEP 518 must be already installed "
    "if this option is used.",
)

check_build_deps: Callable[..., Option] = partial(
    Option,
    "--check-build-dependencies",
    dest="check_build_deps",
    action="store_true",
    default=False,
    help="Check the build dependencies when PEP517 is used.",
)


def _handle_no_use_pep517(
    option: Option, opt: str, value: str, parser: OptionParser
) -> None:
    """
    Process a value provided for the --no-use-pep517 option.

    This is an optparse.Option callback for the no_use_pep517 option.
    """
    # Since --no-use-pep517 doesn't accept arguments, the value argument
    # will be None if --no-use-pep517 is passed via the command-line.
    # However, the value can be non-None if the option is triggered e.g.
    # by an environment variable, for example "PIP_NO_USE_PEP517=true".
    if value is not None:
        msg = """A value was passed for --no-use-pep517,
        probably using either the PIP_NO_USE_PEP517 environment variable
        or the "no-use-pep517" config file option. Use an appropriate value
        of the PIP_USE_PEP517 environment variable or the "use-pep517"
        config file option instead.
        """
        raise_option_error(parser, option=option, msg=msg)

    # If user doesn't wish to use pep517, we check if setuptools is installed
    # and raise error if it is not.
    packages = ("setuptools",)
    if not all(importlib.util.find_spec(package) for package in packages):
        msg = (
            f"It is not possible to use --no-use-pep517 "
            f"without {' and '.join(packages)} installed."
        )
        raise_option_error(parser, option=option, msg=msg)

    # Otherwise, --no-use-pep517 was passed via the command-line.
    parser.values.use_pep517 = False


use_pep517: Any = partial(
    Option,
    "--use-pep517",
    dest="use_pep517",
    action="store_true",
    default=None,
    help="Use PEP 517 for building source distributions "
    "(use --no-use-pep517 to force legacy behaviour).",
)

no_use_pep517: Any = partial(
    Option,
    "--no-use-pep517",
    dest="use_pep517",
    action="callback",
    callback=_handle_no_use_pep517,
    default=None,
    help=SUPPRESS_HELP,
)


def _handle_config_settings(
    option: Option, opt_str: str, value: str, parser: OptionParser
) -> None:
    key, sep, val = value.partition("=")
    if sep != "=":
        parser.error(f"Arguments to {opt_str} must be of the form KEY=VAL")
    dest = getattr(parser.values, option.dest)
    if dest is None:
        dest = {}
        setattr(parser.values, option.dest, dest)
    if key in dest:
        if isinstance(dest[key], list):
            dest[key].append(val)
        else:
            dest[key] = [dest[key], val]
    else:
        dest[key] = val


config_settings: Callable[..., Option] = partial(
    Option,
    "-C",
    "--config-settings",
    dest="config_settings",
    type=str,
    action="callback",
    callback=_handle_config_settings,
    metavar="settings",
    help="Configuration settings to be passed to the PEP 517 build backend. "
    "Settings take the form KEY=VALUE. Use multiple --config-settings options "
    "to pass multiple keys to the backend.",
)

build_options: Callable[..., Option] = partial(
    Option,
    "--build-option",
    dest="build_options",
    metavar="options",
    action="append",
    help="Extra arguments to be supplied to 'setup.py bdist_wheel'.",
)

global_options: Callable[..., Option] = partial(
    Option,
    "--global-option",
    dest="global_options",
    action="append",
    metavar="options",
    help="Extra global options to be supplied to the setup.py "
    "call before the install or bdist_wheel command.",
)

no_clean: Callable[..., Option] = partial(
    Option,
    "--no-clean",
    action="store_true",
    default=False,
    help="Don't clean up build directories.",
)

pre: Callable[..., Option] = partial(
    Option,
    "--pre",
    action="store_true",
    default=False,
    help="Include pre-release and development versions. By default, "
    "pip only finds stable versions.",
)

json: Callable[..., Option] = partial(
    Option,
    "--json",
    action="store_true",
    default=False,
    help="Output data in a machine-readable JSON format.",
)

disable_pip_version_check: Callable[..., Option] = partial(
    Option,
    "--disable-pip-version-check",
    dest="disable_pip_version_check",
    action="store_true",
    default=False,
    help="Don't periodically check PyPI to determine whether a new version "
    "of pip is available for download. Implied with --no-index.",
)

root_user_action: Callable[..., Option] = partial(
    Option,
    "--root-user-action",
    dest="root_user_action",
    default="warn",
    choices=["warn", "ignore"],
    help="Action if pip is run as a root user [warn, ignore] (default: warn)",
)


def _handle_merge_hash(
    option: Option, opt_str: str, value: str, parser: OptionParser
) -> None:
    """Given a value spelled "algo:digest", append the digest to a list
    pointed to in a dict by the algo name."""
    if not parser.values.hashes:
        parser.values.hashes = {}
    try:
        algo, digest = value.split(":", 1)
    except ValueError:
        parser.error(
            f"Arguments to {opt_str} must be a hash name "
            "followed by a value, like --hash=sha256:"
            "abcde..."
        )
    if algo not in STRONG_HASHES:
        parser.error(
            "Allowed hash algorithms for {} are {}.".format(
                opt_str, ", ".join(STRONG_HASHES)
            )
        )
    parser.values.hashes.setdefault(algo, []).append(digest)


hash: Callable[..., Option] = partial(
    Option,
    "--hash",
    # Hash values eventually end up in InstallRequirement.hashes due to
    # __dict__ copying in process_line().
    dest="hashes",
    action="callback",
    callback=_handle_merge_hash,
    type="string",
    help="Verify that the package's archive matches this "
    "hash before installing. Example: --hash=sha256:abcdef...",
)


require_hashes: Callable[..., Option] = partial(
    Option,
    "--require-hashes",
    dest="require_hashes",
    action="store_true",
    default=False,
    help="Require a hash to check each requirement against, for "
    "repeatable installs. This option is implied when any package in a "
    "requirements file has a --hash option.",
)


list_path: Callable[..., Option] = partial(
    PipOption,
    "--path",
    dest="path",
    type="path",
    action="append",
    help="Restrict to the specified installation path for listing "
    "packages (can be used multiple times).",
)


def check_list_path_option(options: Values) -> None:
    if options.path and (options.user or options.local):
        raise CommandError("Cannot combine '--path' with '--user' or '--local'")


list_exclude: Callable[..., Option] = partial(
    PipOption,
    "--exclude",
    dest="excludes",
    action="append",
    metavar="package",
    type="package_name",
    help="Exclude specified package from the output",
)


no_python_version_warning: Callable[..., Option] = partial(
    Option,
    "--no-python-version-warning",
    dest="no_python_version_warning",
    action="store_true",
    default=False,
    help=SUPPRESS_HELP,  # No-op, a hold-over from the Python 2->3 transition.
)


# Features that are now always on. A warning is printed if they are used.
ALWAYS_ENABLED_FEATURES = [
    "truststore",  # always on since 24.2
    "no-binary-enable-wheel-cache",  # always on since 23.1
]

use_new_feature: Callable[..., Option] = partial(
    Option,
    "--use-feature",
    dest="features_enabled",
    metavar="feature",
    action="append",
    default=[],
    choices=[
        "fast-deps",
    ]
    + ALWAYS_ENABLED_FEATURES,
    help="Enable new functionality, that may be backward incompatible.",
)

use_deprecated_feature: Callable[..., Option] = partial(
    Option,
    "--use-deprecated",
    dest="deprecated_features_enabled",
    metavar="feature",
    action="append",
    default=[],
    choices=[
        "legacy-resolver",
        "legacy-certs",
    ],
    help=("Enable deprecated functionality, that will be removed in the future."),
)

##########
# groups #
##########

general_group: Dict[str, Any] = {
    "name": "General Options",
    "options": [
        help_,
        debug_mode,
        isolated_mode,
        require_virtualenv,
        python,
        verbose,
        version,
        quiet,
        log,
        no_input,
        keyring_provider,
        proxy,
        retries,
        timeout,
        exists_action,
        trusted_host,
        cert,
        client_cert,
        cache_dir,
        no_cache,
        disable_pip_version_check,
        no_color,
        no_python_version_warning,
        use_new_feature,
        use_deprecated_feature,
        resume_retries,
    ],
}

index_group: Dict[str, Any] = {
    "name": "Package Index Options",
    "options": [
        index_url,
        extra_index_url,
        no_index,
        find_links,
    ],
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\combinatorial\factorials.py ===
from __future__ import annotations
from functools import reduce

from sympy.core import S, sympify, Dummy, Mod
from sympy.core.cache import cacheit
from sympy.core.function import DefinedFunction, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and
from sympy.core.numbers import Integer, pi, I
from sympy.core.relational import Eq
from sympy.external.gmpy import gmpy as _gmpy
from sympy.ntheory import sieve
from sympy.ntheory.residue_ntheory import binomial_mod
from sympy.polys.polytools import Poly

from math import factorial as _factorial, prod, sqrt as _sqrt

class CombinatorialFunction(DefinedFunction):
    """Base class for combinatorial functions. """

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.combsimp import combsimp
        # combinatorial function with non-integer arguments is
        # automatically passed to gammasimp
        expr = combsimp(self)
        measure = kwargs['measure']
        if measure(expr) <= kwargs['ratio']*measure(self):
            return expr
        return self


###############################################################################
######################## FACTORIAL and MULTI-FACTORIAL ########################
###############################################################################


class factorial(CombinatorialFunction):
    r"""Implementation of factorial function over nonnegative integers.
       By convention (consistent with the gamma function and the binomial
       coefficients), factorial of a negative integer is complex infinity.

       The factorial is very important in combinatorics where it gives
       the number of ways in which `n` objects can be permuted. It also
       arises in calculus, probability, number theory, etc.

       There is strict relation of factorial with gamma function. In
       fact `n! = gamma(n+1)` for nonnegative integers. Rewrite of this
       kind is very useful in case of combinatorial simplification.

       Computation of the factorial is done using two algorithms. For
       small arguments a precomputed look up table is used. However for bigger
       input algorithm Prime-Swing is used. It is the fastest algorithm
       known and computes `n!` via prime factorization of special class
       of numbers, called here the 'Swing Numbers'.

       Examples
       ========

       >>> from sympy import Symbol, factorial, S
       >>> n = Symbol('n', integer=True)

       >>> factorial(0)
       1

       >>> factorial(7)
       5040

       >>> factorial(-2)
       zoo

       >>> factorial(n)
       factorial(n)

       >>> factorial(2*n)
       factorial(2*n)

       >>> factorial(S(1)/2)
       factorial(1/2)

       See Also
       ========

       factorial2, RisingFactorial, FallingFactorial
    """

    def fdiff(self, argindex=1):
        from sympy.functions.special.gamma_functions import (gamma, polygamma)
        if argindex == 1:
            return gamma(self.args[0] + 1)*polygamma(0, self.args[0] + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    _small_swing = [
        1, 1, 1, 3, 3, 15, 5, 35, 35, 315, 63, 693, 231, 3003, 429, 6435, 6435, 109395,
        12155, 230945, 46189, 969969, 88179, 2028117, 676039, 16900975, 1300075,
        35102025, 5014575, 145422675, 9694845, 300540195, 300540195
    ]

    _small_factorials: list[int] = []

    @classmethod
    def _swing(cls, n):
        if n < 33:
            return cls._small_swing[n]
        else:
            N, primes = int(_sqrt(n)), []

            for prime in sieve.primerange(3, N + 1):
                p, q = 1, n

                while True:
                    q //= prime

                    if q > 0:
                        if q & 1 == 1:
                            p *= prime
                    else:
                        break

                if p > 1:
                    primes.append(p)

            for prime in sieve.primerange(N + 1, n//3 + 1):
                if (n // prime) & 1 == 1:
                    primes.append(prime)

            L_product = prod(sieve.primerange(n//2 + 1, n + 1))
            R_product = prod(primes)

            return L_product*R_product

    @classmethod
    def _recursive(cls, n):
        if n < 2:
            return 1
        else:
            return (cls._recursive(n//2)**2)*cls._swing(n)

    @classmethod
    def eval(cls, n):
        n = sympify(n)

        if n.is_Number:
            if n.is_zero:
                return S.One
            elif n is S.Infinity:
                return S.Infinity
            elif n.is_Integer:
                if n.is_negative:
                    return S.ComplexInfinity
                else:
                    n = n.p

                    if n < 20:
                        if not cls._small_factorials:
                            result = 1
                            for i in range(1, 20):
                                result *= i
                                cls._small_factorials.append(result)
                        result = cls._small_factorials[n-1]

                    # GMPY factorial is faster, use it when available
                    #
                    # XXX: There is a sympy.external.gmpy.factorial function
                    # which provides gmpy.fac if available or the flint version
                    # if flint is used. It could be used here to avoid the
                    # conditional logic but it needs to be checked whether the
                    # pure Python fallback used there is as fast as the
                    # fallback used here (perhaps the fallback here should be
                    # moved to sympy.external.ntheory).
                    elif _gmpy is not None:
                        result = _gmpy.fac(n)

                    else:
                        bits = bin(n).count('1')
                        result = cls._recursive(n)*2**(n - bits)

                    return Integer(result)

    def _facmod(self, n, q):
        res, N = 1, int(_sqrt(n))

        # Exponent of prime p in n! is e_p(n) = [n/p] + [n/p**2] + ...
        # for p > sqrt(n), e_p(n) < sqrt(n), the primes with [n/p] = m,
        # occur consecutively and are grouped together in pw[m] for
        # simultaneous exponentiation at a later stage
        pw = [1]*N

        m = 2 # to initialize the if condition below
        for prime in sieve.primerange(2, n + 1):
            if m > 1:
                m, y = 0, n // prime
                while y:
                    m += y
                    y //= prime
            if m < N:
                pw[m] = pw[m]*prime % q
            else:
                res = res*pow(prime, m, q) % q

        for ex, bs in enumerate(pw):
            if ex == 0 or bs == 1:
                continue
            if bs == 0:
                return 0
            res = res*pow(bs, ex, q) % q

        return res

    def _eval_Mod(self, q):
        n = self.args[0]
        if n.is_integer and n.is_nonnegative and q.is_integer:
            aq = abs(q)
            d = aq - n
            if d.is_nonpositive:
                return S.Zero
            else:
                isprime = aq.is_prime
                if d == 1:
                    # Apply Wilson's theorem (if a natural number n > 1
                    # is a prime number, then (n-1)! = -1 mod n) and
                    # its inverse (if n > 4 is a composite number, then
                    # (n-1)! = 0 mod n)
                    if isprime:
                        return -1 % q
                    elif isprime is False and (aq - 6).is_nonnegative:
                        return S.Zero
                elif n.is_Integer and q.is_Integer:
                    n, d, aq = map(int, (n, d, aq))
                    if isprime and (d - 1 < n):
                        fc = self._facmod(d - 1, aq)
                        fc = pow(fc, aq - 2, aq)
                        if d%2:
                            fc = -fc
                    else:
                        fc = self._facmod(n, aq)

                    return fc % q

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return gamma(n + 1)

    def _eval_rewrite_as_Product(self, n, **kwargs):
        from sympy.concrete.products import Product
        if n.is_nonnegative and n.is_integer:
            i = Dummy('i', integer=True)
            return Product(i, (i, 1, n))

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_even(self):
        x = self.args[0]
        if x.is_integer and x.is_nonnegative:
            return (x - 2).is_nonnegative

    def _eval_is_composite(self):
        x = self.args[0]
        if x.is_integer and x.is_nonnegative:
            return (x - 3).is_nonnegative

    def _eval_is_real(self):
        x = self.args[0]
        if x.is_nonnegative or x.is_noninteger:
            return True

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x)
        arg0 = arg.subs(x, 0)
        if arg0.is_zero:
            return S.One
        elif not arg0.is_infinite:
            return self.func(arg)
        raise PoleError("Cannot expand %s around 0" % (self))

class MultiFactorial(CombinatorialFunction):
    pass


class subfactorial(CombinatorialFunction):
    r"""The subfactorial counts the derangements of $n$ items and is
    defined for non-negative integers as:

    .. math:: !n = \begin{cases} 1 & n = 0 \\ 0 & n = 1 \\
                    (n-1)(!(n-1) + !(n-2)) & n > 1 \end{cases}

    It can also be written as ``int(round(n!/exp(1)))`` but the
    recursive definition with caching is implemented for this function.

    An interesting analytic expression is the following [2]_

    .. math:: !x = \Gamma(x + 1, -1)/e

    which is valid for non-negative integers `x`. The above formula
    is not very useful in case of non-integers. `\Gamma(x + 1, -1)` is
    single-valued only for integral arguments `x`, elsewhere on the positive
    real axis it has an infinite number of branches none of which are real.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Subfactorial
    .. [2] https://mathworld.wolfram.com/Subfactorial.html

    Examples
    ========

    >>> from sympy import subfactorial
    >>> from sympy.abc import n
    >>> subfactorial(n + 1)
    subfactorial(n + 1)
    >>> subfactorial(5)
    44

    See Also
    ========

    factorial, uppergamma,
    sympy.utilities.iterables.generate_derangements
    """

    @classmethod
    @cacheit
    def _eval(self, n):
        if not n:
            return S.One
        elif n == 1:
            return S.Zero
        else:
            z1, z2 = 1, 0
            for i in range(2, n + 1):
                z1, z2 = z2, (i - 1)*(z2 + z1)
            return z2

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg.is_Integer and arg.is_nonnegative:
                return cls._eval(arg)
            elif arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity

    def _eval_is_even(self):
        if self.args[0].is_odd and self.args[0].is_nonnegative:
            return True

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_rewrite_as_factorial(self, arg, **kwargs):
        from sympy.concrete.summations import summation
        i = Dummy('i')
        f = S.NegativeOne**i / factorial(i)
        return factorial(arg) * summation(f, (i, 0, arg))

    def _eval_rewrite_as_gamma(self, arg, piecewise=True, **kwargs):
        from sympy.functions.elementary.exponential import exp
        from sympy.functions.special.gamma_functions import (gamma, lowergamma)
        return (S.NegativeOne**(arg + 1)*exp(-I*pi*arg)*lowergamma(arg + 1, -1)
                + gamma(arg + 1))*exp(-1)

    def _eval_rewrite_as_uppergamma(self, arg, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return uppergamma(arg + 1, -1)/S.Exp1

    def _eval_is_nonnegative(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_odd(self):
        if self.args[0].is_even and self.args[0].is_nonnegative:
            return True


class factorial2(CombinatorialFunction):
    r"""The double factorial `n!!`, not to be confused with `(n!)!`

    The double factorial is defined for nonnegative integers and for odd
    negative integers as:

    .. math:: n!! = \begin{cases} 1 & n = 0 \\
                    n(n-2)(n-4) \cdots 1 & n\ \text{positive odd} \\
                    n(n-2)(n-4) \cdots 2 & n\ \text{positive even} \\
                    (n+2)!!/(n+2) & n\ \text{negative odd} \end{cases}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Double_factorial

    Examples
    ========

    >>> from sympy import factorial2, var
    >>> n = var('n')
    >>> n
    n
    >>> factorial2(n + 1)
    factorial2(n + 1)
    >>> factorial2(5)
    15
    >>> factorial2(-1)
    1
    >>> factorial2(-5)
    1/3

    See Also
    ========

    factorial, RisingFactorial, FallingFactorial
    """

    @classmethod
    def eval(cls, arg):
        # TODO: extend this to complex numbers?

        if arg.is_Number:
            if not arg.is_Integer:
                raise ValueError("argument must be nonnegative integer "
                                    "or negative odd integer")

            # This implementation is faster than the recursive one
            # It also avoids "maximum recursion depth exceeded" runtime error
            if arg.is_nonnegative:
                if arg.is_even:
                    k = arg / 2
                    return 2**k * factorial(k)
                return factorial(arg) / factorial2(arg - 1)


            if arg.is_odd:
                return arg*(S.NegativeOne)**((1 - arg)/2) / factorial2(-arg)
            raise ValueError("argument must be nonnegative integer "
                                "or negative odd integer")


    def _eval_is_even(self):
        # Double factorial is even for every positive even input
        n = self.args[0]
        if n.is_integer:
            if n.is_odd:
                return False
            if n.is_even:
                if n.is_positive:
                    return True
                if n.is_zero:
                    return False

    def _eval_is_integer(self):
        # Double factorial is an integer for every nonnegative input, and for
        # -1 and -3
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return (n + 3).is_nonnegative

    def _eval_is_odd(self):
        # Double factorial is odd for every odd input not smaller than -3, and
        # for 0
        n = self.args[0]
        if n.is_odd:
            return (n + 3).is_nonnegative
        if n.is_even:
            if n.is_positive:
                return False
            if n.is_zero:
                return True

    def _eval_is_positive(self):
        # Double factorial is positive for every nonnegative input, and for
        # every odd negative input which is of the form -1-4k for an
        # nonnegative integer k
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return ((n + 1) / 2).is_even

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        return 2**(n/2)*gamma(n/2 + 1) * Piecewise((1, Eq(Mod(n, 2), 0)),
                (sqrt(2/pi), Eq(Mod(n, 2), 1)))


###############################################################################
######################## RISING and FALLING FACTORIALS ########################
###############################################################################


class RisingFactorial(CombinatorialFunction):
    r"""
    Rising factorial (also called Pochhammer symbol [1]_) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by:

    .. math:: \texttt{rf(y, k)} = (x)^k = x \cdot (x+1) \cdots (x+k-1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or visit https://mathworld.wolfram.com/RisingFactorial.html page.

    When `x` is a `~.Poly` instance of degree $\ge 1$ with a single variable,
    `(x)^k = x(y) \cdot x(y+1) \cdots x(y+k-1)`, where `y` is the
    variable of `x`. This is as described in [2]_.

    Examples
    ========

    >>> from sympy import rf, Poly
    >>> from sympy.abc import x
    >>> rf(x, 0)
    1
    >>> rf(1, 5)
    120
    >>> rf(x, 5) == x*(1 + x)*(2 + x)*(3 + x)*(4 + x)
    True
    >>> rf(Poly(x**3, x), 2)
    Poly(x**6 + 3*x**5 + 3*x**4 + x**3, x, domain='ZZ')

    Rewriting is complicated unless the relationship between
    the arguments is known, but rising factorial can
    be rewritten in terms of gamma, factorial, binomial,
    and falling factorial.

    >>> from sympy import Symbol, factorial, ff, binomial, gamma
    >>> n = Symbol('n', integer=True, positive=True)
    >>> R = rf(n, n + 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  R.rewrite(i)
    ...
    RisingFactorial(n, n + 2)
    FallingFactorial(2*n + 1, n + 2)
    factorial(2*n + 1)/factorial(n - 1)
    binomial(2*n + 1, n + 2)*factorial(n + 2)
    gamma(2*n + 2)/gamma(n)

    See Also
    ========

    factorial, factorial2, FallingFactorial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pochhammer_symbol
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    """

    @classmethod
    def eval(cls, x, k):
        x = sympify(x)
        k = sympify(k)

        if x is S.NaN or k is S.NaN:
            return S.NaN
        elif x is S.One:
            return factorial(k)
        elif k.is_Integer:
            if k.is_zero:
                return S.One
            else:
                if k.is_positive:
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        if k.is_odd:
                            return S.NegativeInfinity
                        else:
                            return S.Infinity
                    else:
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens)!= 1:
                                raise ValueError("rf only defined for "
                                            "polynomials on one generator")
                            else:
                                return reduce(lambda r, i:
                                              r*(x.shift(i)),
                                              range(int(k)), 1)
                        else:
                            return reduce(lambda r, i: r*(x + i),
                                          range(int(k)), 1)

                else:
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        return S.Infinity
                    else:
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens)!= 1:
                                raise ValueError("rf only defined for "
                                            "polynomials on one generator")
                            else:
                                return 1/reduce(lambda r, i:
                                                r*(x.shift(-i)),
                                                range(1, abs(int(k)) + 1), 1)
                        else:
                            return 1/reduce(lambda r, i:
                                            r*(x - i),
                                            range(1, abs(int(k)) + 1), 1)

        if k.is_integer == False:
            if x.is_integer and x.is_negative:
                return S.Zero

    def _eval_rewrite_as_gamma(self, x, k, piecewise=True, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        if not piecewise:
            if (x <= 0) == True:
                return S.NegativeOne**k*gamma(1 - x) / gamma(-k - x + 1)
            return gamma(x + k) / gamma(x)
        return Piecewise(
            (gamma(x + k) / gamma(x), x > 0),
            (S.NegativeOne**k*gamma(1 - x) / gamma(-k - x + 1), True))

    def _eval_rewrite_as_FallingFactorial(self, x, k, **kwargs):
        return FallingFactorial(x + k - 1, k)

    def _eval_rewrite_as_factorial(self, x, k, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        if x.is_integer and k.is_integer:
            return Piecewise(
                (factorial(k + x - 1)/factorial(x - 1), x > 0),
                (S.NegativeOne**k*factorial(-x)/factorial(-k - x), True))

    def _eval_rewrite_as_binomial(self, x, k, **kwargs):
        if k.is_integer:
            return factorial(k) * binomial(x + k - 1, k)

    def _eval_rewrite_as_tractable(self, x, k, limitvar=None, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        if limitvar:
            k_lim = k.subs(limitvar, S.Infinity)
            if k_lim is S.Infinity:
                return (gamma(x + k).rewrite('tractable', deep=True) / gamma(x))
            elif k_lim is S.NegativeInfinity:
                return (S.NegativeOne**k*gamma(1 - x) / gamma(-k - x + 1).rewrite('tractable', deep=True))
        return self.rewrite(gamma).rewrite('tractable', deep=True)

    def _eval_is_integer(self):
        return fuzzy_and((self.args[0].is_integer, self.args[1].is_integer,
                          self.args[1].is_nonnegative))


class FallingFactorial(CombinatorialFunction):
    r"""
    Falling factorial (related to rising factorial) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by

    .. math:: \texttt{ff(x, k)} = (x)_k = x \cdot (x-1) \cdots (x-k+1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or [1]_.

    When `x` is a `~.Poly` instance of degree $\ge 1$ with single variable,
    `(x)_k = x(y) \cdot x(y-1) \cdots x(y-k+1)`, where `y` is the
    variable of `x`. This is as described in

    >>> from sympy import ff, Poly, Symbol
    >>> from sympy.abc import x
    >>> n = Symbol('n', integer=True)

    >>> ff(x, 0)
    1
    >>> ff(5, 5)
    120
    >>> ff(x, 5) == x*(x - 1)*(x - 2)*(x - 3)*(x - 4)
    True
    >>> ff(Poly(x**2, x), 2)
    Poly(x**4 - 2*x**3 + x**2, x, domain='ZZ')
    >>> ff(n, n)
    factorial(n)

    Rewriting is complicated unless the relationship between
    the arguments is known, but falling factorial can
    be rewritten in terms of gamma, factorial and binomial
    and rising factorial.

    >>> from sympy import factorial, rf, gamma, binomial, Symbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> F = ff(n, n - 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  F.rewrite(i)
    ...
    RisingFactorial(3, n - 2)
    FallingFactorial(n, n - 2)
    factorial(n)/2
    binomial(n, n - 2)*factorial(n - 2)
    gamma(n + 1)/2

    See Also
    ========

    factorial, factorial2, RisingFactorial

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FallingFactorial.html
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    """

    @classmethod
    def eval(cls, x, k):
        x = sympify(x)
        k = sympify(k)

        if x is S.NaN or k is S.NaN:
            return S.NaN
        elif k.is_integer and x == k:
            return factorial(x)
        elif k.is_Integer:
            if k.is_zero:
                return S.One
            else:
                if k.is_positive:
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        if k.is_odd:
                            return S.NegativeInfinity
                        else:
                            return S.Infinity
                    else:
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens)!= 1:
                                raise ValueError("ff only defined for "
                                            "polynomials on one generator")
                            else:
                                return reduce(lambda r, i:
                                              r*(x.shift(-i)),
                                              range(int(k)), 1)
                        else:
                            return reduce(lambda r, i: r*(x - i),
                                          range(int(k)), 1)
                else:
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        return S.Infinity
                    else:
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens)!= 1:
                                raise ValueError("rf only defined for "
                                            "polynomials on one generator")
                            else:
                                return 1/reduce(lambda r, i:
                                                r*(x.shift(i)),
                                                range(1, abs(int(k)) + 1), 1)
                        else:
                            return 1/reduce(lambda r, i: r*(x + i),
                                            range(1, abs(int(k)) + 1), 1)

    def _eval_rewrite_as_gamma(self, x, k, piecewise=True, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        if not piecewise:
            if (x < 0) == True:
                return S.NegativeOne**k*gamma(k - x) / gamma(-x)
            return gamma(x + 1) / gamma(x - k + 1)
        return Piecewise(
            (gamma(x + 1) / gamma(x - k + 1), x >= 0),
            (S.NegativeOne**k*gamma(k - x) / gamma(-x), True))

    def _eval_rewrite_as_RisingFactorial(self, x, k, **kwargs):
        return rf(x - k + 1, k)

    def _eval_rewrite_as_binomial(self, x, k, **kwargs):
        if k.is_integer:
            return factorial(k) * binomial(x, k)

    def _eval_rewrite_as_factorial(self, x, k, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        if x.is_integer and k.is_integer:
            return Piecewise(
                (factorial(x)/factorial(-k + x), x >= 0),
                (S.NegativeOne**k*factorial(k - x - 1)/factorial(-x - 1), True))

    def _eval_rewrite_as_tractable(self, x, k, limitvar=None, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        if limitvar:
            k_lim = k.subs(limitvar, S.Infinity)
            if k_lim is S.Infinity:
                return (S.NegativeOne**k*gamma(k - x).rewrite('tractable', deep=True) / gamma(-x))
            elif k_lim is S.NegativeInfinity:
                return (gamma(x + 1) / gamma(x - k + 1).rewrite('tractable', deep=True))
        return self.rewrite(gamma).rewrite('tractable', deep=True)

    def _eval_is_integer(self):
        return fuzzy_and((self.args[0].is_integer, self.args[1].is_integer,
                          self.args[1].is_nonnegative))


rf = RisingFactorial
ff = FallingFactorial

###############################################################################
########################### BINOMIAL COEFFICIENTS #############################
###############################################################################


class binomial(CombinatorialFunction):
    r"""Implementation of the binomial coefficient. It can be defined
    in two ways depending on its desired interpretation:

    .. math:: \binom{n}{k} = \frac{n!}{k!(n-k)!}\ \text{or}\
                \binom{n}{k} = \frac{(n)_k}{k!}

    First, in a strict combinatorial sense it defines the
    number of ways we can choose `k` elements from a set of
    `n` elements. In this case both arguments are nonnegative
    integers and binomial is computed using an efficient
    algorithm based on prime factorization.

    The other definition is generalization for arbitrary `n`,
    however `k` must also be nonnegative. This case is very
    useful when evaluating summations.

    For the sake of convenience, for negative integer `k` this function
    will return zero no matter the other argument.

    To expand the binomial when `n` is a symbol, use either
    ``expand_func()`` or ``expand(func=True)``. The former will keep
    the polynomial in factored form while the latter will expand the
    polynomial itself. See examples for details.

    Examples
    ========

    >>> from sympy import Symbol, Rational, binomial, expand_func
    >>> n = Symbol('n', integer=True, positive=True)

    >>> binomial(15, 8)
    6435

    >>> binomial(n, -1)
    0

    Rows of Pascal's triangle can be generated with the binomial function:

    >>> for N in range(8):
    ...     print([binomial(N, i) for i in range(N + 1)])
    ...
    [1]
    [1, 1]
    [1, 2, 1]
    [1, 3, 3, 1]
    [1, 4, 6, 4, 1]
    [1, 5, 10, 10, 5, 1]
    [1, 6, 15, 20, 15, 6, 1]
    [1, 7, 21, 35, 35, 21, 7, 1]

    As can a given diagonal, e.g. the 4th diagonal:

    >>> N = -4
    >>> [binomial(N, i) for i in range(1 - N)]
    [1, -4, 10, -20, 35]

    >>> binomial(Rational(5, 4), 3)
    -5/128
    >>> binomial(Rational(-5, 4), 3)
    -195/128

    >>> binomial(n, 3)
    binomial(n, 3)

    >>> binomial(n, 3).expand(func=True)
    n**3/6 - n**2/2 + n/3

    >>> expand_func(binomial(n, 3))
    n*(n - 2)*(n - 1)/6

    In many cases, we can also compute binomial coefficients modulo a
    prime p quickly using Lucas' Theorem [2]_, though we need to include
    `evaluate=False` to postpone evaluation:

    >>> from sympy import Mod
    >>> Mod(binomial(156675, 4433, evaluate=False), 10**5 + 3)
    28625

    Using a generalisation of Lucas's Theorem given by Granville [3]_,
    we can extend this to arbitrary n:

    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    References
    ==========

    .. [1] https://www.johndcook.com/blog/binomial_coefficients/
    .. [2] https://en.wikipedia.org/wiki/Lucas%27s_theorem
    .. [3] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """

    def fdiff(self, argindex=1):
        from sympy.functions.special.gamma_functions import polygamma
        if argindex == 1:
            # https://functions.wolfram.com/GammaBetaErf/Binomial/20/01/01/
            n, k = self.args
            return binomial(n, k)*(polygamma(0, n + 1) - \
                polygamma(0, n - k + 1))
        elif argindex == 2:
            # https://functions.wolfram.com/GammaBetaErf/Binomial/20/01/02/
            n, k = self.args
            return binomial(n, k)*(polygamma(0, n - k + 1) - \
                polygamma(0, k + 1))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def _eval(self, n, k):
        # n.is_Number and k.is_Integer and k != 1 and n != k

        if k.is_Integer:
            if n.is_Integer and n >= 0:
                n, k = int(n), int(k)

                if k > n:
                    return S.Zero
                elif k > n // 2:
                    k = n - k

                # XXX: This conditional logic should be moved to
                # sympy.external.gmpy and the pure Python version of bincoef
                # should be moved to sympy.external.ntheory.
                if _gmpy is not None:
                    return Integer(_gmpy.bincoef(n, k))

                d, result = n - k, 1
                for i in range(1, k + 1):
                    d += 1
                    result = result * d // i
                return Integer(result)
            else:
                d, result = n - k, 1
                for i in range(1, k + 1):
                    d += 1
                    result *= d
                return result / _factorial(k)

    @classmethod
    def eval(cls, n, k):
        n, k = map(sympify, (n, k))
        d = n - k
        n_nonneg, n_isint = n.is_nonnegative, n.is_integer
        if k.is_zero or ((n_nonneg or n_isint is False)
                and d.is_zero):
            return S.One
        if (k - 1).is_zero or ((n_nonneg or n_isint is False)
                and (d - 1).is_zero):
            return n
        if k.is_integer:
            if k.is_negative or (n_nonneg and n_isint and d.is_negative):
                return S.Zero
            elif n.is_number:
                res = cls._eval(n, k)
                return res.expand(basic=True) if res else res
        elif n_nonneg is False and n_isint:
            # a special case when binomial evaluates to complex infinity
            return S.ComplexInfinity
        elif k.is_number:
            from sympy.functions.special.gamma_functions import gamma
            return gamma(n + 1)/(gamma(k + 1)*gamma(n - k + 1))

    def _eval_Mod(self, q):
        n, k = self.args

        if any(x.is_integer is False for x in (n, k, q)):
            raise ValueError("Integers expected for binomial Mod")

        if all(x.is_Integer for x in (n, k, q)):
            n, k = map(int, (n, k))
            aq, res = abs(q), 1

            # handle negative integers k or n
            if k < 0:
                return S.Zero
            if n < 0:
                n = -n + k - 1
                res = -1 if k%2 else 1

            # non negative integers k and n
            if k > n:
                return S.Zero

            isprime = aq.is_prime
            aq = int(aq)
            if isprime:
                if aq < n:
                    # use Lucas Theorem
                    N, K = n, k
                    while N or K:
                        res = res*binomial(N % aq, K % aq) % aq
                        N, K = N // aq, K // aq

                else:
                    # use Factorial Modulo
                    d = n - k
                    if k > d:
                        k, d = d, k
                    kf = 1
                    for i in range(2, k + 1):
                        kf = kf*i % aq
                    df = kf
                    for i in range(k + 1, d + 1):
                        df = df*i % aq
                    res *= df
                    for i in range(d + 1, n + 1):
                        res = res*i % aq

                    res *= pow(kf*df % aq, aq - 2, aq)
                    res %= aq

            elif _sqrt(q) < k and q != 1:
                res = binomial_mod(n, k, q)

            else:
                # Binomial Factorization is performed by calculating the
                # exponents of primes <= n in `n! /(k! (n - k)!)`,
                # for non-negative integers n and k. As the exponent of
                # prime in n! is e_p(n) = [n/p] + [n/p**2] + ...
                # the exponent of prime in binomial(n, k) would be
                # e_p(n) - e_p(k) - e_p(n - k)
                M = int(_sqrt(n))
                for prime in sieve.primerange(2, n + 1):
                    if prime > n - k:
                        res = res*prime % aq
                    elif prime > n // 2:
                        continue
                    elif prime > M:
                        if n % prime < k % prime:
                            res = res*prime % aq
                    else:
                        N, K = n, k
                        exp = a = 0

                        while N > 0:
                            a = int((N % prime) < (K % prime + a))
                            N, K = N // prime, K // prime
                            exp += a

                        if exp > 0:
                            res *= pow(prime, exp, aq)
                            res %= aq

            return S(res % q)

    def _eval_expand_func(self, **hints):
        """
        Function to expand binomial(n, k) when m is positive integer
        Also,
        n is self.args[0] and k is self.args[1] while using binomial(n, k)
        """
        n = self.args[0]
        if n.is_Number:
            return binomial(*self.args)

        k = self.args[1]
        if (n-k).is_Integer:
            k = n - k

        if k.is_Integer:
            if k.is_zero:
                return S.One
            elif k.is_negative:
                return S.Zero
            else:
                n, result = self.args[0], 1
                for i in range(1, k + 1):
                    result *= n - k + i
                return result / _factorial(k)
        else:
            return binomial(*self.args)

    def _eval_rewrite_as_factorial(self, n, k, **kwargs):
        return factorial(n)/(factorial(k)*factorial(n - k))

    def _eval_rewrite_as_gamma(self, n, k, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return gamma(n + 1)/(gamma(k + 1)*gamma(n - k + 1))

    def _eval_rewrite_as_tractable(self, n, k, limitvar=None, **kwargs):
        return self._eval_rewrite_as_gamma(n, k).rewrite('tractable')

    def _eval_rewrite_as_FallingFactorial(self, n, k, **kwargs):
        if k.is_integer:
            return ff(n, k) / factorial(k)

    def _eval_is_integer(self):
        n, k = self.args
        if n.is_integer and k.is_integer:
            return True
        elif k.is_integer is False:
            return False

    def _eval_is_nonnegative(self):
        n, k = self.args
        if n.is_integer and k.is_integer:
            if n.is_nonnegative or k.is_negative or k.is_even:
                return True
            elif k.is_even is False:
                return  False

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.functions.special.gamma_functions import gamma
        return self.rewrite(gamma)._eval_as_leading_term(x, logx=logx, cdir=cdir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\datetimes.py ===
from __future__ import annotations

import datetime as dt
import operator
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytz

from pandas._libs import (
    NaT,
    Period,
    Timestamp,
    index as libindex,
    lib,
)
from pandas._libs.tslibs import (
    Resolution,
    Tick,
    Timedelta,
    periods_per_day,
    timezones,
    to_offset,
)
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas.util._decorators import (
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.arrays.datetimes import (
    DatetimeArray,
    tz_to_dtype,
)
import pandas.core.common as com
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Frequency,
        IntervalClosedType,
        Self,
        TimeAmbiguous,
        TimeNonexistent,
        npt,
    )

    from pandas.core.api import (
        DataFrame,
        PeriodIndex,
    )

from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR


def _new_DatetimeIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't
    have arguments and breaks __new__
    """
    if "data" in d and not isinstance(d["data"], DatetimeIndex):
        # Avoid need to verify integrity by calling simple_new directly
        data = d.pop("data")
        if not isinstance(data, DatetimeArray):
            # For backward compat with older pickles, we may need to construct
            #  a DatetimeArray to adapt to the newer _simple_new signature
            tz = d.pop("tz")
            freq = d.pop("freq")
            dta = DatetimeArray._simple_new(data, dtype=tz_to_dtype(tz), freq=freq)
        else:
            dta = data
            for key in ["tz", "freq"]:
                # These are already stored in our DatetimeArray; if they are
                #  also in the pickle and don't match, we have a problem.
                if key in d:
                    assert d[key] == getattr(dta, key)
                    d.pop(key)
        result = cls._simple_new(dta, **d)
    else:
        with warnings.catch_warnings():
            # TODO: If we knew what was going in to **d, we might be able to
            #  go through _simple_new instead
            warnings.simplefilter("ignore")
            result = cls.__new__(cls, **d)

    return result


@inherit_names(
    DatetimeArray._field_ops
    + [
        method
        for method in DatetimeArray._datetimelike_methods
        if method not in ("tz_localize", "tz_convert", "strftime")
    ],
    DatetimeArray,
    wrap=True,
)
@inherit_names(["is_normalized"], DatetimeArray, cache=True)
@inherit_names(
    [
        "tz",
        "tzinfo",
        "dtype",
        "to_pydatetime",
        "date",
        "time",
        "timetz",
        "std",
    ]
    + DatetimeArray._bool_ops,
    DatetimeArray,
)
class DatetimeIndex(DatetimeTimedeltaMixin):
    """
    Immutable ndarray-like of datetime64 data.

    Represented internally as int64, and which can be boxed to Timestamp objects
    that are subclasses of datetime and carry metadata.

    .. versionchanged:: 2.0.0
        The various numeric date/time attributes (:attr:`~DatetimeIndex.day`,
        :attr:`~DatetimeIndex.month`, :attr:`~DatetimeIndex.year` etc.) now have dtype
        ``int32``. Previously they had dtype ``int64``.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Datetime-like data to construct index with.
    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        'infer' can be passed in order to set the frequency of the index as the
        inferred frequency upon creation.
    tz : pytz.timezone or dateutil.tz.tzfile or datetime.tzinfo or str
        Set the Timezone of the data.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

        .. deprecated:: 2.1.0

    closed : {'left', 'right'}, optional
        Set whether to include `start` and `end` that are on the
        boundary. The default includes boundary points on either end.

        .. deprecated:: 2.1.0

    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from 03:00
        DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
        and at 01:30:00 UTC. In such a situation, the `ambiguous` parameter
        dictates how ambiguous times should be handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False signifies a
          non-DST time (note that this flag is only applicable for ambiguous
          times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous times.
    dayfirst : bool, default False
        If True, parse dates in `data` with the day first order.
    yearfirst : bool, default False
        If True parse dates in `data` with the year first order.
    dtype : numpy.dtype or DatetimeTZDtype or str, default None
        Note that the only NumPy dtype allowed is `datetime64[ns]`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : label, default None
        Name to be stored in the index.

    Attributes
    ----------
    year
    month
    day
    hour
    minute
    second
    microsecond
    nanosecond
    date
    time
    timetz
    dayofyear
    day_of_year
    dayofweek
    day_of_week
    weekday
    quarter
    tz
    freq
    freqstr
    is_month_start
    is_month_end
    is_quarter_start
    is_quarter_end
    is_year_start
    is_year_end
    is_leap_year
    inferred_freq

    Methods
    -------
    normalize
    strftime
    snap
    tz_convert
    tz_localize
    round
    floor
    ceil
    to_period
    to_pydatetime
    to_series
    to_frame
    month_name
    day_name
    mean
    std

    See Also
    --------
    Index : The base pandas Index type.
    TimedeltaIndex : Index of timedelta64 data.
    PeriodIndex : Index of Period data.
    to_datetime : Convert argument to datetime.
    date_range : Create a fixed-frequency DatetimeIndex.

    Notes
    -----
    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
    >>> idx
    DatetimeIndex(['2020-01-01 10:00:00+00:00', '2020-02-01 11:00:00+00:00'],
    dtype='datetime64[ns, UTC]', freq=None)
    """

    _typ = "datetimeindex"

    _data_cls = DatetimeArray
    _supports_partial_string_indexing = True

    @property
    def _engine_type(self) -> type[libindex.DatetimeEngine]:
        return libindex.DatetimeEngine

    _data: DatetimeArray
    _values: DatetimeArray
    tz: dt.tzinfo | None

    # --------------------------------------------------------------------
    # methods that dispatch to DatetimeArray and wrap result

    @doc(DatetimeArray.strftime)
    def strftime(self, date_format) -> Index:
        arr = self._data.strftime(date_format)
        return Index(arr, name=self.name, dtype=arr.dtype)

    @doc(DatetimeArray.tz_convert)
    def tz_convert(self, tz) -> Self:
        arr = self._data.tz_convert(tz)
        return type(self)._simple_new(arr, name=self.name, refs=self._references)

    @doc(DatetimeArray.tz_localize)
    def tz_localize(
        self,
        tz,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        arr = self._data.tz_localize(tz, ambiguous, nonexistent)
        return type(self)._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_period)
    def to_period(self, freq=None) -> PeriodIndex:
        from pandas.core.indexes.api import PeriodIndex

        arr = self._data.to_period(freq)
        return PeriodIndex._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_julian_date)
    def to_julian_date(self) -> Index:
        arr = self._data.to_julian_date()
        return Index._simple_new(arr, name=self.name)

    @doc(DatetimeArray.isocalendar)
    def isocalendar(self) -> DataFrame:
        df = self._data.isocalendar()
        return df.set_index(self)

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        return self._data._resolution_obj

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        freq: Frequency | lib.NoDefault = lib.no_default,
        tz=lib.no_default,
        normalize: bool | lib.NoDefault = lib.no_default,
        closed=lib.no_default,
        ambiguous: TimeAmbiguous = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        if closed is not lib.no_default:
            # GH#52628
            warnings.warn(
                f"The 'closed' keyword in {cls.__name__} construction is "
                "deprecated and will be removed in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if normalize is not lib.no_default:
            # GH#52628
            warnings.warn(
                f"The 'normalize' keyword in {cls.__name__} construction is "
                "deprecated and will be removed in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        if is_scalar(data):
            cls._raise_scalar_data_error(data)

        # - Cases checked above all return/raise before reaching here - #

        name = maybe_extract_name(name, data, cls)

        if (
            isinstance(data, DatetimeArray)
            and freq is lib.no_default
            and tz is lib.no_default
            and dtype is None
        ):
            # fastpath, similar logic in TimedeltaIndex.__new__;
            # Note in this particular case we retain non-nano.
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)

        dtarr = DatetimeArray._from_sequence_not_strict(
            data,
            dtype=dtype,
            copy=copy,
            tz=tz,
            freq=freq,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            ambiguous=ambiguous,
        )
        refs = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references

        subarr = cls._simple_new(dtarr, name=name, refs=refs)
        return subarr

    # --------------------------------------------------------------------

    @cache_readonly
    def _is_dates_only(self) -> bool:
        """
        Return a boolean if we are only dates (and don't have a timezone)

        Returns
        -------
        bool
        """
        if isinstance(self.freq, Tick):
            delta = Timedelta(self.freq)

            if delta % dt.timedelta(days=1) != dt.timedelta(days=0):
                return False

        return self._values._is_dates_only

    def __reduce__(self):
        d = {"data": self._data, "name": self.name}
        return _new_DatetimeIndex, (type(self), d), None

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        if self.tz is not None:
            # If we have tz, we can compare to tzaware
            return isinstance(dtype, DatetimeTZDtype)
        # if we dont have tz, we can only compare to tznaive
        return lib.is_np_dtype(dtype, "M")

    # --------------------------------------------------------------------
    # Rendering Methods

    @cache_readonly
    def _formatter_func(self):
        # Note this is equivalent to the DatetimeIndexOpsMixin method but
        #  uses the maybe-cached self._is_dates_only instead of re-computing it.
        from pandas.io.formats.format import get_format_datetime64

        formatter = get_format_datetime64(is_dates_only=self._is_dates_only)
        return lambda x: f"'{formatter(x)}'"

    # --------------------------------------------------------------------
    # Set Operation Methods

    def _can_range_setop(self, other) -> bool:
        # GH 46702: If self or other have non-UTC tzs, DST transitions prevent
        # range representation due to no singular step
        if (
            self.tz is not None
            and not timezones.is_utc(self.tz)
            and not timezones.is_fixed_offset(self.tz)
        ):
            return False
        if (
            other.tz is not None
            and not timezones.is_utc(other.tz)
            and not timezones.is_fixed_offset(other.tz)
        ):
            return False
        return super()._can_range_setop(other)

    # --------------------------------------------------------------------

    def _get_time_micros(self) -> npt.NDArray[np.int64]:
        """
        Return the number of microseconds since midnight.

        Returns
        -------
        ndarray[int64_t]
        """
        values = self._data._local_timestamps()

        ppd = periods_per_day(self._data._creso)

        frac = values % ppd
        if self.unit == "ns":
            micros = frac // 1000
        elif self.unit == "us":
            micros = frac
        elif self.unit == "ms":
            micros = frac * 1000
        elif self.unit == "s":
            micros = frac * 1_000_000
        else:  # pragma: no cover
            raise NotImplementedError(self.unit)

        micros[self._isnan] = -1
        return micros

    def snap(self, freq: Frequency = "S") -> DatetimeIndex:
        """
        Snap time stamps to nearest occurring frequency.

        Returns
        -------
        DatetimeIndex

        Examples
        --------
        >>> idx = pd.DatetimeIndex(['2023-01-01', '2023-01-02',
        ...                        '2023-02-01', '2023-02-02'])
        >>> idx
        DatetimeIndex(['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02'],
        dtype='datetime64[ns]', freq=None)
        >>> idx.snap('MS')
        DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01'],
        dtype='datetime64[ns]', freq=None)
        """
        # Superdumb, punting on any optimizing
        freq = to_offset(freq)

        dta = self._data.copy()

        for i, v in enumerate(self):
            s = v
            if not freq.is_on_offset(s):
                t0 = freq.rollback(s)
                t1 = freq.rollforward(s)
                if abs(s - t0) < abs(t1 - s):
                    s = t0
                else:
                    s = t1
            dta[i] = s

        return DatetimeIndex._simple_new(dta, name=self.name)

    # --------------------------------------------------------------------
    # Indexing Methods

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: dt.datetime):
        """
        Calculate datetime bounds for parsed time string and its resolution.

        Parameters
        ----------
        reso : Resolution
            Resolution provided by parsed string.
        parsed : datetime
            Datetime from parsed string.

        Returns
        -------
        lower, upper: pd.Timestamp
        """
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        per = Period(parsed, freq=freq)
        start, end = per.start_time, per.end_time

        # GH 24076
        # If an incoming date string contained a UTC offset, need to localize
        # the parsed date to this offset first before aligning with the index's
        # timezone
        start = start.tz_localize(parsed.tzinfo)
        end = end.tz_localize(parsed.tzinfo)

        if parsed.tzinfo is not None:
            if self.tz is None:
                raise ValueError(
                    "The index must be timezone aware when indexing "
                    "with a date string with a UTC offset"
                )
        # The flipped case with parsed.tz is None and self.tz is not None
        #  is ruled out bc parsed and reso are produced by _parse_with_reso,
        #  which localizes parsed.
        return start, end

    def _parse_with_reso(self, label: str):
        parsed, reso = super()._parse_with_reso(label)

        parsed = Timestamp(parsed)

        if self.tz is not None and parsed.tzinfo is None:
            # we special-case timezone-naive strings and timezone-aware
            #  DatetimeIndex
            # https://github.com/pandas-dev/pandas/pull/36148#issuecomment-687883081
            parsed = parsed.tz_localize(self.tz)

        return parsed, reso

    def _disallow_mismatched_indexing(self, key) -> None:
        """
        Check for mismatched-tzawareness indexing and re-raise as KeyError.
        """
        # we get here with isinstance(key, self._data._recognized_scalars)
        try:
            # GH#36148
            self._data._assert_tzawareness_compat(key)
        except TypeError as err:
            raise KeyError(key) from err

    def get_loc(self, key):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int
        """
        self._check_indexing_error(key)

        orig_key = key
        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT

        if isinstance(key, self._data._recognized_scalars):
            # needed to localize naive datetimes
            self._disallow_mismatched_indexing(key)
            key = Timestamp(key)

        elif isinstance(key, str):
            try:
                parsed, reso = self._parse_with_reso(key)
            except (ValueError, pytz.NonExistentTimeError) as err:
                raise KeyError(key) from err
            self._disallow_mismatched_indexing(parsed)

            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err

            key = parsed

        elif isinstance(key, dt.timedelta):
            # GH#20464
            raise TypeError(
                f"Cannot index {type(self).__name__} with {type(key).__name__}"
            )

        elif isinstance(key, dt.time):
            return self.indexer_at_time(key)

        else:
            # unrecognized type
            raise KeyError(key)

        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            raise KeyError(orig_key) from err

    @doc(DatetimeTimedeltaMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side: str):
        # GH#42855 handle date here instead of get_slice_bound
        if isinstance(label, dt.date) and not isinstance(label, dt.datetime):
            # Pandas supports slicing with dates, treated as datetimes at midnight.
            # https://github.com/pandas-dev/pandas/issues/31501
            label = Timestamp(label).to_pydatetime()

        label = super()._maybe_cast_slice_bound(label, side)
        self._data._assert_tzawareness_compat(label)
        return Timestamp(label)

    def slice_indexer(self, start=None, end=None, step=None):
        """
        Return indexer for specified label slice.
        Index.slice_indexer, customized to handle time slicing.

        In addition to functionality provided by Index.slice_indexer, does the
        following:

        - if both `start` and `end` are instances of `datetime.time`, it
          invokes `indexer_between_time`
        - if `start` and `end` are both either string or None perform
          value-based selection in non-monotonic cases.

        """
        # For historical reasons DatetimeIndex supports slices between two
        # instances of datetime.time as if it were applying a slice mask to
        # an array of (self.hour, self.minute, self.seconds, self.microsecond).
        if isinstance(start, dt.time) and isinstance(end, dt.time):
            if step is not None and step != 1:
                raise ValueError("Must have step size of 1 with time slices")
            return self.indexer_between_time(start, end)

        if isinstance(start, dt.time) or isinstance(end, dt.time):
            raise KeyError("Cannot mix time and non-time slice keys")

        def check_str_or_none(point) -> bool:
            return point is not None and not isinstance(point, str)

        # GH#33146 if start and end are combinations of str and None and Index is not
        # monotonic, we can not use Index.slice_indexer because it does not honor the
        # actual elements, is only searching for start and end
        if (
            check_str_or_none(start)
            or check_str_or_none(end)
            or self.is_monotonic_increasing
        ):
            return Index.slice_indexer(self, start, end, step)

        mask = np.array(True)
        in_index = True
        if start is not None:
            start_casted = self._maybe_cast_slice_bound(start, "left")
            mask = start_casted <= self
            in_index &= (start_casted == self).any()

        if end is not None:
            end_casted = self._maybe_cast_slice_bound(end, "right")
            mask = (self <= end_casted) & mask
            in_index &= (end_casted == self).any()

        if not in_index:
            raise KeyError(
                "Value based partial slicing on non-monotonic DatetimeIndexes "
                "with non-existing keys is not allowed.",
            )
        indexer = mask.nonzero()[0][::step]
        if len(indexer) == len(self):
            return slice(None)
        else:
            return indexer

    # --------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        # b/c datetime is represented as microseconds since the epoch, make
        # sure we can't have ambiguous indexing
        return "datetime64"

    def indexer_at_time(self, time, asof: bool = False) -> npt.NDArray[np.intp]:
        """
        Return index locations of values at particular time of day.

        Parameters
        ----------
        time : datetime.time or str
            Time passed in either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p", "%I%M%S%p").

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_between_time : Get index locations of values between particular
            times of day.
        DataFrame.at_time : Select values at particular time of day.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00", "2/1/2020 11:00",
        ...                         "3/1/2020 10:00"])
        >>> idx.indexer_at_time("10:00")
        array([0, 2])
        """
        if asof:
            raise NotImplementedError("'asof' argument is not supported")

        if isinstance(time, str):
            from dateutil.parser import parse

            time = parse(time).time()

        if time.tzinfo:
            if self.tz is None:
                raise ValueError("Index must be timezone aware.")
            time_micros = self.tz_convert(time.tzinfo)._get_time_micros()
        else:
            time_micros = self._get_time_micros()
        micros = _time_to_micros(time)
        return (time_micros == micros).nonzero()[0]

    def indexer_between_time(
        self, start_time, end_time, include_start: bool = True, include_end: bool = True
    ) -> npt.NDArray[np.intp]:
        """
        Return index locations of values between particular times of day.

        Parameters
        ----------
        start_time, end_time : datetime.time, str
            Time passed either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p","%I%M%S%p").
        include_start : bool, default True
        include_end : bool, default True

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_at_time : Get index locations of values at particular time of day.
        DataFrame.between_time : Select values between particular times of day.

        Examples
        --------
        >>> idx = pd.date_range("2023-01-01", periods=4, freq="h")
        >>> idx
        DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 01:00:00',
                           '2023-01-01 02:00:00', '2023-01-01 03:00:00'],
                          dtype='datetime64[ns]', freq='h')
        >>> idx.indexer_between_time("00:00", "2:00", include_end=False)
        array([0, 1])
        """
        start_time = to_time(start_time)
        end_time = to_time(end_time)
        time_micros = self._get_time_micros()
        start_micros = _time_to_micros(start_time)
        end_micros = _time_to_micros(end_time)

        if include_start and include_end:
            lop = rop = operator.le
        elif include_start:
            lop = operator.le
            rop = operator.lt
        elif include_end:
            lop = operator.lt
            rop = operator.le
        else:
            lop = rop = operator.lt

        if start_time <= end_time:
            join_op = operator.and_
        else:
            join_op = operator.or_

        mask = join_op(lop(start_micros, time_micros), rop(time_micros, end_micros))

        return mask.nonzero()[0]


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize: bool = False,
    name: Hashable | None = None,
    inclusive: IntervalClosedType = "both",
    *,
    unit: str | None = None,
    **kwargs,
) -> DatetimeIndex:
    """
    Return a fixed frequency DatetimeIndex.

    Returns the range of equally spaced time points (where the difference between any
    two adjacent points is specified by the given frequency) such that they all
    satisfy `start <[=] x <[=] end`, where the first one and the last one are, resp.,
    the first and last time points in that range that fall on the boundary of ``freq``
    (if given as a frequency string) or that are valid for ``freq`` (if given as a
    :class:`pandas.tseries.offsets.DateOffset`). (If exactly one of ``start``,
    ``end``, or ``freq`` is *not* specified, this missing parameter can be computed
    given ``periods``, the number of timesteps in the range. See the note below.)

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5h'. See
        :ref:`here <timeseries.offset_aliases>` for a list of
        frequency aliases.
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        'Asia/Hong_Kong'. By default, the resulting DatetimeIndex is
        timezone-naive unless timezone-aware datetime-likes are passed.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; Whether to set each bound as closed or open.

        .. versionadded:: 1.4.0
    unit : str, default None
        Specify the desired resolution of the result.

        .. versionadded:: 2.0.0
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    DatetimeIndex

    See Also
    --------
    DatetimeIndex : An immutable container for datetimes.
    timedelta_range : Return a fixed frequency TimedeltaIndex.
    period_range : Return a fixed frequency PeriodIndex.
    interval_range : Return a fixed frequency IntervalIndex.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    **Specifying the values**

    The next four examples generate the same `DatetimeIndex`, but vary
    the combination of `start`, `end` and `periods`.

    Specify `start` and `end`, with the default daily frequency.

    >>> pd.date_range(start='1/1/2018', end='1/08/2018')
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                  dtype='datetime64[ns]', freq='D')

    Specify timezone-aware `start` and `end`, with the default daily frequency.

    >>> pd.date_range(
    ...     start=pd.to_datetime("1/1/2018").tz_localize("Europe/Berlin"),
    ...     end=pd.to_datetime("1/08/2018").tz_localize("Europe/Berlin"),
    ... )
    DatetimeIndex(['2018-01-01 00:00:00+01:00', '2018-01-02 00:00:00+01:00',
                   '2018-01-03 00:00:00+01:00', '2018-01-04 00:00:00+01:00',
                   '2018-01-05 00:00:00+01:00', '2018-01-06 00:00:00+01:00',
                   '2018-01-07 00:00:00+01:00', '2018-01-08 00:00:00+01:00'],
                  dtype='datetime64[ns, Europe/Berlin]', freq='D')

    Specify `start` and `periods`, the number of periods (days).

    >>> pd.date_range(start='1/1/2018', periods=8)
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                  dtype='datetime64[ns]', freq='D')

    Specify `end` and `periods`, the number of periods (days).

    >>> pd.date_range(end='1/1/2018', periods=8)
    DatetimeIndex(['2017-12-25', '2017-12-26', '2017-12-27', '2017-12-28',
                   '2017-12-29', '2017-12-30', '2017-12-31', '2018-01-01'],
                  dtype='datetime64[ns]', freq='D')

    Specify `start`, `end`, and `periods`; the frequency is generated
    automatically (linearly spaced).

    >>> pd.date_range(start='2018-04-24', end='2018-04-27', periods=3)
    DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00',
                   '2018-04-27 00:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Other Parameters**

    Changed the `freq` (frequency) to ``'ME'`` (month end frequency).

    >>> pd.date_range(start='1/1/2018', periods=5, freq='ME')
    DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30',
                   '2018-05-31'],
                  dtype='datetime64[ns]', freq='ME')

    Multiples are allowed

    >>> pd.date_range(start='1/1/2018', periods=5, freq='3ME')
    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                   '2019-01-31'],
                  dtype='datetime64[ns]', freq='3ME')

    `freq` can also be specified as an Offset object.

    >>> pd.date_range(start='1/1/2018', periods=5, freq=pd.offsets.MonthEnd(3))
    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                   '2019-01-31'],
                  dtype='datetime64[ns]', freq='3ME')

    Specify `tz` to set the timezone.

    >>> pd.date_range(start='1/1/2018', periods=5, tz='Asia/Tokyo')
    DatetimeIndex(['2018-01-01 00:00:00+09:00', '2018-01-02 00:00:00+09:00',
                   '2018-01-03 00:00:00+09:00', '2018-01-04 00:00:00+09:00',
                   '2018-01-05 00:00:00+09:00'],
                  dtype='datetime64[ns, Asia/Tokyo]', freq='D')

    `inclusive` controls whether to include `start` and `end` that are on the
    boundary. The default, "both", includes boundary points on either end.

    >>> pd.date_range(start='2017-01-01', end='2017-01-04', inclusive="both")
    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04'],
                  dtype='datetime64[ns]', freq='D')

    Use ``inclusive='left'`` to exclude `end` if it falls on the boundary.

    >>> pd.date_range(start='2017-01-01', end='2017-01-04', inclusive='left')
    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03'],
                  dtype='datetime64[ns]', freq='D')

    Use ``inclusive='right'`` to exclude `start` if it falls on the boundary, and
    similarly ``inclusive='neither'`` will exclude both `start` and `end`.

    >>> pd.date_range(start='2017-01-01', end='2017-01-04', inclusive='right')
    DatetimeIndex(['2017-01-02', '2017-01-03', '2017-01-04'],
                  dtype='datetime64[ns]', freq='D')

    **Specify a unit**

    >>> pd.date_range(start="2017-01-01", periods=10, freq="100YS", unit="s")
    DatetimeIndex(['2017-01-01', '2117-01-01', '2217-01-01', '2317-01-01',
                   '2417-01-01', '2517-01-01', '2617-01-01', '2717-01-01',
                   '2817-01-01', '2917-01-01'],
                  dtype='datetime64[s]', freq='100YS-JAN')
    """
    if freq is None and com.any_none(periods, start, end):
        freq = "D"

    dtarr = DatetimeArray._generate_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        inclusive=inclusive,
        unit=unit,
        **kwargs,
    )
    return DatetimeIndex._simple_new(dtarr, name=name)


def bdate_range(
    start=None,
    end=None,
    periods: int | None = None,
    freq: Frequency | dt.timedelta = "B",
    tz=None,
    normalize: bool = True,
    name: Hashable | None = None,
    weekmask=None,
    holidays=None,
    inclusive: IntervalClosedType = "both",
    **kwargs,
) -> DatetimeIndex:
    """
    Return a fixed frequency DatetimeIndex with business day as the default.

    Parameters
    ----------
    start : str or datetime-like, default None
        Left bound for generating dates.
    end : str or datetime-like, default None
        Right bound for generating dates.
    periods : int, default None
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'B'
        Frequency strings can have multiples, e.g. '5h'. The default is
        business daily ('B').
    tz : str or None
        Time zone name for returning localized DatetimeIndex, for example
        Asia/Beijing.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    weekmask : str or None, default None
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``,
        only used when custom frequency strings are passed.  The default
        value None is equivalent to 'Mon Tue Wed Thu Fri'.
    holidays : list-like or None, default None
        Dates to exclude from the set of valid business days, passed to
        ``numpy.busdaycalendar``, only used when custom frequency strings
        are passed.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; Whether to set each bound as closed or open.

        .. versionadded:: 1.4.0
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    DatetimeIndex

    Notes
    -----
    Of the four parameters: ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified.  Specifying ``freq`` is a requirement
    for ``bdate_range``.  Use ``date_range`` if specifying ``freq`` is not
    desired.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    Note how the two weekend days are skipped in the result.

    >>> pd.bdate_range(start='1/1/2018', end='1/08/2018')
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-08'],
              dtype='datetime64[ns]', freq='B')
    """
    if freq is None:
        msg = "freq must be specified for bdate_range; use date_range instead"
        raise TypeError(msg)

    if isinstance(freq, str) and freq.startswith("C"):
        try:
            weekmask = weekmask or "Mon Tue Wed Thu Fri"
            freq = prefix_mapping[freq](holidays=holidays, weekmask=weekmask)
        except (KeyError, TypeError) as err:
            msg = f"invalid custom frequency string: {freq}"
            raise ValueError(msg) from err
    elif holidays or weekmask:
        msg = (
            "a custom frequency string is required when holidays or "
            f"weekmask are passed, got frequency {freq}"
        )
        raise ValueError(msg)

    return date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        inclusive=inclusive,
        **kwargs,
    )


def _time_to_micros(time_obj: dt.time) -> int:
    seconds = time_obj.hour * 60 * 60 + 60 * time_obj.minute + time_obj.second
    return 1_000_000 * seconds + time_obj.microsecond

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\c_ast.py ===
#-----------------------------------------------------------------
# ** ATTENTION **
# This code was automatically generated from the file:
# _c_ast.cfg
#
# Do not modify it directly. Modify the configuration file and
# run the generator again.
# ** ** *** ** **
#
# pycparser: c_ast.py
#
# AST Node classes.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#-----------------------------------------------------------------


import sys

def _repr(obj):
    """
    Get the representation of an object, with dedicated pprint-like format for lists.
    """
    if isinstance(obj, list):
        return '[' + (',\n '.join((_repr(e).replace('\n', '\n ') for e in obj))) + '\n]'
    else:
        return repr(obj)

class Node(object):
    __slots__ = ()
    """ Abstract base class for AST nodes.
    """
    def __repr__(self):
        """ Generates a python representation of the current node
        """
        result = self.__class__.__name__ + '('

        indent = ''
        separator = ''
        for name in self.__slots__[:-2]:
            result += separator
            result += indent
            result += name + '=' + (_repr(getattr(self, name)).replace('\n', '\n  ' + (' ' * (len(name) + len(self.__class__.__name__)))))

            separator = ','
            indent = '\n ' + (' ' * len(self.__class__.__name__))

        result += indent + ')'

        return result

    def children(self):
        """ A sequence of all children that are Nodes
        """
        pass

    def show(self, buf=sys.stdout, offset=0, attrnames=False, nodenames=False, showcoord=False, _my_node_name=None):
        """ Pretty print the Node and all its attributes and
            children (recursively) to a buffer.

            buf:
                Open IO buffer into which the Node is printed.

            offset:
                Initial offset (amount of leading spaces)

            attrnames:
                True if you want to see the attribute names in
                name=value pairs. False to only see the values.

            nodenames:
                True if you want to see the actual node names
                within their parents.

            showcoord:
                Do you want the coordinates of each Node to be
                displayed.
        """
        lead = ' ' * offset
        if nodenames and _my_node_name is not None:
            buf.write(lead + self.__class__.__name__+ ' <' + _my_node_name + '>: ')
        else:
            buf.write(lead + self.__class__.__name__+ ': ')

        if self.attr_names:
            if attrnames:
                nvlist = [(n, getattr(self,n)) for n in self.attr_names]
                attrstr = ', '.join('%s=%s' % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ', '.join('%s' % v for v in vlist)
            buf.write(attrstr)

        if showcoord:
            buf.write(' (at %s)' % self.coord)
        buf.write('\n')

        for (child_name, child) in self.children():
            child.show(
                buf,
                offset=offset + 2,
                attrnames=attrnames,
                nodenames=nodenames,
                showcoord=showcoord,
                _my_node_name=child_name)


class NodeVisitor(object):
    """ A base NodeVisitor class for visiting c_ast nodes.
        Subclass it and define your own visit_XXX methods, where
        XXX is the class name you want to visit with these
        methods.

        For example:

        class ConstantVisitor(NodeVisitor):
            def __init__(self):
                self.values = []

            def visit_Constant(self, node):
                self.values.append(node.value)

        Creates a list of values of all the constant nodes
        encountered below the given node. To use it:

        cv = ConstantVisitor()
        cv.visit(node)

        Notes:

        *   generic_visit() will be called for AST nodes for which
            no visit_XXX method was defined.
        *   The children of nodes for which a visit_XXX was
            defined will not be visited - if you need this, call
            generic_visit() on the node.
            You can use:
                NodeVisitor.generic_visit(self, node)
        *   Modeled after Python's own AST visiting facilities
            (the ast module of Python 3.0)
    """

    _method_cache = None

    def visit(self, node):
        """ Visit a node.
        """

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__, None)
        if visitor is None:
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node)

    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a
            node. Implements preorder visiting of the node.
        """
        for c in node:
            self.visit(c)

class ArrayDecl(Node):
    __slots__ = ('type', 'dim', 'dim_quals', 'coord', '__weakref__')
    def __init__(self, type, dim, dim_quals, coord=None):
        self.type = type
        self.dim = dim
        self.dim_quals = dim_quals
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.dim is not None: nodelist.append(("dim", self.dim))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.dim is not None:
            yield self.dim

    attr_names = ('dim_quals', )

class ArrayRef(Node):
    __slots__ = ('name', 'subscript', 'coord', '__weakref__')
    def __init__(self, name, subscript, coord=None):
        self.name = name
        self.subscript = subscript
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.subscript is not None: nodelist.append(("subscript", self.subscript))
        return tuple(nodelist)

    def __iter__(self):
        if self.name is not None:
            yield self.name
        if self.subscript is not None:
            yield self.subscript

    attr_names = ()

class Assignment(Node):
    __slots__ = ('op', 'lvalue', 'rvalue', 'coord', '__weakref__')
    def __init__(self, op, lvalue, rvalue, coord=None):
        self.op = op
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.coord = coord

    def children(self):
        nodelist = []
        if self.lvalue is not None: nodelist.append(("lvalue", self.lvalue))
        if self.rvalue is not None: nodelist.append(("rvalue", self.rvalue))
        return tuple(nodelist)

    def __iter__(self):
        if self.lvalue is not None:
            yield self.lvalue
        if self.rvalue is not None:
            yield self.rvalue

    attr_names = ('op', )

class Alignas(Node):
    __slots__ = ('alignment', 'coord', '__weakref__')
    def __init__(self, alignment, coord=None):
        self.alignment = alignment
        self.coord = coord

    def children(self):
        nodelist = []
        if self.alignment is not None: nodelist.append(("alignment", self.alignment))
        return tuple(nodelist)

    def __iter__(self):
        if self.alignment is not None:
            yield self.alignment

    attr_names = ()

class BinaryOp(Node):
    __slots__ = ('op', 'left', 'right', 'coord', '__weakref__')
    def __init__(self, op, left, right, coord=None):
        self.op = op
        self.left = left
        self.right = right
        self.coord = coord

    def children(self):
        nodelist = []
        if self.left is not None: nodelist.append(("left", self.left))
        if self.right is not None: nodelist.append(("right", self.right))
        return tuple(nodelist)

    def __iter__(self):
        if self.left is not None:
            yield self.left
        if self.right is not None:
            yield self.right

    attr_names = ('op', )

class Break(Node):
    __slots__ = ('coord', '__weakref__')
    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()

class Case(Node):
    __slots__ = ('expr', 'stmts', 'coord', '__weakref__')
    def __init__(self, expr, stmts, coord=None):
        self.expr = expr
        self.stmts = stmts
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        for i, child in enumerate(self.stmts or []):
            nodelist.append(("stmts[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr
        for child in (self.stmts or []):
            yield child

    attr_names = ()

class Cast(Node):
    __slots__ = ('to_type', 'expr', 'coord', '__weakref__')
    def __init__(self, to_type, expr, coord=None):
        self.to_type = to_type
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.to_type is not None: nodelist.append(("to_type", self.to_type))
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    def __iter__(self):
        if self.to_type is not None:
            yield self.to_type
        if self.expr is not None:
            yield self.expr

    attr_names = ()

class Compound(Node):
    __slots__ = ('block_items', 'coord', '__weakref__')
    def __init__(self, block_items, coord=None):
        self.block_items = block_items
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.block_items or []):
            nodelist.append(("block_items[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.block_items or []):
            yield child

    attr_names = ()

class CompoundLiteral(Node):
    __slots__ = ('type', 'init', 'coord', '__weakref__')
    def __init__(self, type, init, coord=None):
        self.type = type
        self.init = init
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.init is not None: nodelist.append(("init", self.init))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.init is not None:
            yield self.init

    attr_names = ()

class Constant(Node):
    __slots__ = ('type', 'value', 'coord', '__weakref__')
    def __init__(self, type, value, coord=None):
        self.type = type
        self.value = value
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    def __iter__(self):
        return
        yield

    attr_names = ('type', 'value', )

class Continue(Node):
    __slots__ = ('coord', '__weakref__')
    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()

class Decl(Node):
    __slots__ = ('name', 'quals', 'align', 'storage', 'funcspec', 'type', 'init', 'bitsize', 'coord', '__weakref__')
    def __init__(self, name, quals, align, storage, funcspec, type, init, bitsize, coord=None):
        self.name = name
        self.quals = quals
        self.align = align
        self.storage = storage
        self.funcspec = funcspec
        self.type = type
        self.init = init
        self.bitsize = bitsize
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        if self.init is not None: nodelist.append(("init", self.init))
        if self.bitsize is not None: nodelist.append(("bitsize", self.bitsize))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.init is not None:
            yield self.init
        if self.bitsize is not None:
            yield self.bitsize

    attr_names = ('name', 'quals', 'align', 'storage', 'funcspec', )

class DeclList(Node):
    __slots__ = ('decls', 'coord', '__weakref__')
    def __init__(self, decls, coord=None):
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.decls or []):
            yield child

    attr_names = ()

class Default(Node):
    __slots__ = ('stmts', 'coord', '__weakref__')
    def __init__(self, stmts, coord=None):
        self.stmts = stmts
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.stmts or []):
            nodelist.append(("stmts[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.stmts or []):
            yield child

    attr_names = ()

class DoWhile(Node):
    __slots__ = ('cond', 'stmt', 'coord', '__weakref__')
    def __init__(self, cond, stmt, coord=None):
        self.cond = cond
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.stmt is not None:
            yield self.stmt

    attr_names = ()

class EllipsisParam(Node):
    __slots__ = ('coord', '__weakref__')
    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()

class EmptyStatement(Node):
    __slots__ = ('coord', '__weakref__')
    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()

class Enum(Node):
    __slots__ = ('name', 'values', 'coord', '__weakref__')
    def __init__(self, name, values, coord=None):
        self.name = name
        self.values = values
        self.coord = coord

    def children(self):
        nodelist = []
        if self.values is not None: nodelist.append(("values", self.values))
        return tuple(nodelist)

    def __iter__(self):
        if self.values is not None:
            yield self.values

    attr_names = ('name', )

class Enumerator(Node):
    __slots__ = ('name', 'value', 'coord', '__weakref__')
    def __init__(self, name, value, coord=None):
        self.name = name
        self.value = value
        self.coord = coord

    def children(self):
        nodelist = []
        if self.value is not None: nodelist.append(("value", self.value))
        return tuple(nodelist)

    def __iter__(self):
        if self.value is not None:
            yield self.value

    attr_names = ('name', )

class EnumeratorList(Node):
    __slots__ = ('enumerators', 'coord', '__weakref__')
    def __init__(self, enumerators, coord=None):
        self.enumerators = enumerators
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.enumerators or []):
            nodelist.append(("enumerators[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.enumerators or []):
            yield child

    attr_names = ()

class ExprList(Node):
    __slots__ = ('exprs', 'coord', '__weakref__')
    def __init__(self, exprs, coord=None):
        self.exprs = exprs
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(("exprs[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.exprs or []):
            yield child

    attr_names = ()

class FileAST(Node):
    __slots__ = ('ext', 'coord', '__weakref__')
    def __init__(self, ext, coord=None):
        self.ext = ext
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.ext or []):
            nodelist.append(("ext[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.ext or []):
            yield child

    attr_names = ()

class For(Node):
    __slots__ = ('init', 'cond', 'next', 'stmt', 'coord', '__weakref__')
    def __init__(self, init, cond, next, stmt, coord=None):
        self.init = init
        self.cond = cond
        self.next = next
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.init is not None: nodelist.append(("init", self.init))
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.next is not None: nodelist.append(("next", self.next))
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.init is not None:
            yield self.init
        if self.cond is not None:
            yield self.cond
        if self.next is not None:
            yield self.next
        if self.stmt is not None:
            yield self.stmt

    attr_names = ()

class FuncCall(Node):
    __slots__ = ('name', 'args', 'coord', '__weakref__')
    def __init__(self, name, args, coord=None):
        self.name = name
        self.args = args
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.args is not None: nodelist.append(("args", self.args))
        return tuple(nodelist)

    def __iter__(self):
        if self.name is not None:
            yield self.name
        if self.args is not None:
            yield self.args

    attr_names = ()

class FuncDecl(Node):
    __slots__ = ('args', 'type', 'coord', '__weakref__')
    def __init__(self, args, type, coord=None):
        self.args = args
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.args is not None: nodelist.append(("args", self.args))
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.args is not None:
            yield self.args
        if self.type is not None:
            yield self.type

    attr_names = ()

class FuncDef(Node):
    __slots__ = ('decl', 'param_decls', 'body', 'coord', '__weakref__')
    def __init__(self, decl, param_decls, body, coord=None):
        self.decl = decl
        self.param_decls = param_decls
        self.body = body
        self.coord = coord

    def children(self):
        nodelist = []
        if self.decl is not None: nodelist.append(("decl", self.decl))
        if self.body is not None: nodelist.append(("body", self.body))
        for i, child in enumerate(self.param_decls or []):
            nodelist.append(("param_decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        if self.decl is not None:
            yield self.decl
        if self.body is not None:
            yield self.body
        for child in (self.param_decls or []):
            yield child

    attr_names = ()

class Goto(Node):
    __slots__ = ('name', 'coord', '__weakref__')
    def __init__(self, name, coord=None):
        self.name = name
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    def __iter__(self):
        return
        yield

    attr_names = ('name', )

class ID(Node):
    __slots__ = ('name', 'coord', '__weakref__')
    def __init__(self, name, coord=None):
        self.name = name
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    def __iter__(self):
        return
        yield

    attr_names = ('name', )

class IdentifierType(Node):
    __slots__ = ('names', 'coord', '__weakref__')
    def __init__(self, names, coord=None):
        self.names = names
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    def __iter__(self):
        return
        yield

    attr_names = ('names', )

class If(Node):
    __slots__ = ('cond', 'iftrue', 'iffalse', 'coord', '__weakref__')
    def __init__(self, cond, iftrue, iffalse, coord=None):
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.iftrue is not None: nodelist.append(("iftrue", self.iftrue))
        if self.iffalse is not None: nodelist.append(("iffalse", self.iffalse))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.iftrue is not None:
            yield self.iftrue
        if self.iffalse is not None:
            yield self.iffalse

    attr_names = ()

class InitList(Node):
    __slots__ = ('exprs', 'coord', '__weakref__')
    def __init__(self, exprs, coord=None):
        self.exprs = exprs
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(("exprs[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.exprs or []):
            yield child

    attr_names = ()

class Label(Node):
    __slots__ = ('name', 'stmt', 'coord', '__weakref__')
    def __init__(self, name, stmt, coord=None):
        self.name = name
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.stmt is not None:
            yield self.stmt

    attr_names = ('name', )

class NamedInitializer(Node):
    __slots__ = ('name', 'expr', 'coord', '__weakref__')
    def __init__(self, name, expr, coord=None):
        self.name = name
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        for i, child in enumerate(self.name or []):
            nodelist.append(("name[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr
        for child in (self.name or []):
            yield child

    attr_names = ()

class ParamList(Node):
    __slots__ = ('params', 'coord', '__weakref__')
    def __init__(self, params, coord=None):
        self.params = params
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.params or []):
            nodelist.append(("params[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.params or []):
            yield child

    attr_names = ()

class PtrDecl(Node):
    __slots__ = ('quals', 'type', 'coord', '__weakref__')
    def __init__(self, quals, type, coord=None):
        self.quals = quals
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type

    attr_names = ('quals', )

class Return(Node):
    __slots__ = ('expr', 'coord', '__weakref__')
    def __init__(self, expr, coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr

    attr_names = ()

class StaticAssert(Node):
    __slots__ = ('cond', 'message', 'coord', '__weakref__')
    def __init__(self, cond, message, coord=None):
        self.cond = cond
        self.message = message
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.message is not None: nodelist.append(("message", self.message))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.message is not None:
            yield self.message

    attr_names = ()

class Struct(Node):
    __slots__ = ('name', 'decls', 'coord', '__weakref__')
    def __init__(self, name, decls, coord=None):
        self.name = name
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.decls or []):
            yield child

    attr_names = ('name', )

class StructRef(Node):
    __slots__ = ('name', 'type', 'field', 'coord', '__weakref__')
    def __init__(self, name, type, field, coord=None):
        self.name = name
        self.type = type
        self.field = field
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None: nodelist.append(("name", self.name))
        if self.field is not None: nodelist.append(("field", self.field))
        return tuple(nodelist)

    def __iter__(self):
        if self.name is not None:
            yield self.name
        if self.field is not None:
            yield self.field

    attr_names = ('type', )

class Switch(Node):
    __slots__ = ('cond', 'stmt', 'coord', '__weakref__')
    def __init__(self, cond, stmt, coord=None):
        self.cond = cond
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.stmt is not None:
            yield self.stmt

    attr_names = ()

class TernaryOp(Node):
    __slots__ = ('cond', 'iftrue', 'iffalse', 'coord', '__weakref__')
    def __init__(self, cond, iftrue, iffalse, coord=None):
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.iftrue is not None: nodelist.append(("iftrue", self.iftrue))
        if self.iffalse is not None: nodelist.append(("iffalse", self.iffalse))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.iftrue is not None:
            yield self.iftrue
        if self.iffalse is not None:
            yield self.iffalse

    attr_names = ()

class TypeDecl(Node):
    __slots__ = ('declname', 'quals', 'align', 'type', 'coord', '__weakref__')
    def __init__(self, declname, quals, align, type, coord=None):
        self.declname = declname
        self.quals = quals
        self.align = align
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type

    attr_names = ('declname', 'quals', 'align', )

class Typedef(Node):
    __slots__ = ('name', 'quals', 'storage', 'type', 'coord', '__weakref__')
    def __init__(self, name, quals, storage, type, coord=None):
        self.name = name
        self.quals = quals
        self.storage = storage
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type

    attr_names = ('name', 'quals', 'storage', )

class Typename(Node):
    __slots__ = ('name', 'quals', 'align', 'type', 'coord', '__weakref__')
    def __init__(self, name, quals, align, type, coord=None):
        self.name = name
        self.quals = quals
        self.align = align
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None: nodelist.append(("type", self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type

    attr_names = ('name', 'quals', 'align', )

class UnaryOp(Node):
    __slots__ = ('op', 'expr', 'coord', '__weakref__')
    def __init__(self, op, expr, coord=None):
        self.op = op
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None: nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr

    attr_names = ('op', )

class Union(Node):
    __slots__ = ('name', 'decls', 'coord', '__weakref__')
    def __init__(self, name, decls, coord=None):
        self.name = name
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in (self.decls or []):
            yield child

    attr_names = ('name', )

class While(Node):
    __slots__ = ('cond', 'stmt', 'coord', '__weakref__')
    def __init__(self, cond, stmt, coord=None):
        self.cond = cond
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None: nodelist.append(("cond", self.cond))
        if self.stmt is not None: nodelist.append(("stmt", self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.stmt is not None:
            yield self.stmt

    attr_names = ()

class Pragma(Node):
    __slots__ = ('string', 'coord', '__weakref__')
    def __init__(self, string, coord=None):
        self.string = string
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    def __iter__(self):
        return
        yield

    attr_names = ('string', )


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\serving.py ===
"""A WSGI and HTTP server for use **during development only**. This
server is convenient to use, but is not designed to be particularly
stable, secure, or efficient. Use a dedicate WSGI server and HTTP
server when deploying to production.

It provides features like interactive debugging and code reloading. Use
``run_simple`` to start the server. Put this in a ``run.py`` script:

.. code-block:: python

    from myapp import create_app
    from werkzeug import run_simple
"""

from __future__ import annotations

import errno
import io
import os
import selectors
import socket
import socketserver
import sys
import typing as t
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib.parse import unquote
from urllib.parse import urlsplit

from ._internal import _log
from ._internal import _wsgi_encoding_dance
from .exceptions import InternalServerError
from .urls import uri_to_iri

try:
    import ssl

    connection_dropped_errors: tuple[type[Exception], ...] = (
        ConnectionError,
        socket.timeout,
        ssl.SSLEOFError,
    )
except ImportError:

    class _SslDummy:
        def __getattr__(self, name: str) -> t.Any:
            raise RuntimeError(  # noqa: B904
                "SSL is unavailable because this Python runtime was not"
                " compiled with SSL/TLS support."
            )

    ssl = _SslDummy()  # type: ignore
    connection_dropped_errors = (ConnectionError, socket.timeout)

_log_add_style = True

if os.name == "nt":
    try:
        __import__("colorama")
    except ImportError:
        _log_add_style = False

can_fork = hasattr(os, "fork")

if can_fork:
    ForkingMixIn = socketserver.ForkingMixIn
else:

    class ForkingMixIn:  # type: ignore
        pass


try:
    af_unix = socket.AF_UNIX
except AttributeError:
    af_unix = None  # type: ignore

LISTEN_QUEUE = 128

_TSSLContextArg = t.Optional[
    t.Union["ssl.SSLContext", tuple[str, t.Optional[str]], t.Literal["adhoc"]]
]

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment
    from cryptography.hazmat.primitives.asymmetric.rsa import (
        RSAPrivateKeyWithSerialization,
    )
    from cryptography.x509 import Certificate


class DechunkedInput(io.RawIOBase):
    """An input stream that handles Transfer-Encoding 'chunked'"""

    def __init__(self, rfile: t.IO[bytes]) -> None:
        self._rfile = rfile
        self._done = False
        self._len = 0

    def readable(self) -> bool:
        return True

    def read_chunk_len(self) -> int:
        try:
            line = self._rfile.readline().decode("latin1")
            _len = int(line.strip(), 16)
        except ValueError as e:
            raise OSError("Invalid chunk header") from e
        if _len < 0:
            raise OSError("Negative chunk length not allowed")
        return _len

    def readinto(self, buf: bytearray) -> int:  # type: ignore
        read = 0
        while not self._done and read < len(buf):
            if self._len == 0:
                # This is the first chunk or we fully consumed the previous
                # one. Read the next length of the next chunk
                self._len = self.read_chunk_len()

            if self._len == 0:
                # Found the final chunk of size 0. The stream is now exhausted,
                # but there is still a final newline that should be consumed
                self._done = True

            if self._len > 0:
                # There is data (left) in this chunk, so append it to the
                # buffer. If this operation fully consumes the chunk, this will
                # reset self._len to 0.
                n = min(len(buf), self._len)

                # If (read + chunk size) becomes more than len(buf), buf will
                # grow beyond the original size and read more data than
                # required. So only read as much data as can fit in buf.
                if read + n > len(buf):
                    buf[read:] = self._rfile.read(len(buf) - read)
                    self._len -= len(buf) - read
                    read = len(buf)
                else:
                    buf[read : read + n] = self._rfile.read(n)
                    self._len -= n
                    read += n

            if self._len == 0:
                # Skip the terminating newline of a chunk that has been fully
                # consumed. This also applies to the 0-sized final chunk
                terminator = self._rfile.readline()
                if terminator not in (b"\n", b"\r\n", b"\r"):
                    raise OSError("Missing chunk terminating newline")

        return read


class WSGIRequestHandler(BaseHTTPRequestHandler):
    """A request handler that implements WSGI dispatching."""

    server: BaseWSGIServer

    @property
    def server_version(self) -> str:  # type: ignore
        return self.server._server_version

    def make_environ(self) -> WSGIEnvironment:
        request_url = urlsplit(self.path)
        url_scheme = "http" if self.server.ssl_context is None else "https"

        if not self.client_address:
            self.client_address = ("<local>", 0)
        elif isinstance(self.client_address, str):
            self.client_address = (self.client_address, 0)

        # If there was no scheme but the path started with two slashes,
        # the first segment may have been incorrectly parsed as the
        # netloc, prepend it to the path again.
        if not request_url.scheme and request_url.netloc:
            path_info = f"/{request_url.netloc}{request_url.path}"
        else:
            path_info = request_url.path

        path_info = unquote(path_info)

        environ: WSGIEnvironment = {
            "wsgi.version": (1, 0),
            "wsgi.url_scheme": url_scheme,
            "wsgi.input": self.rfile,
            "wsgi.errors": sys.stderr,
            "wsgi.multithread": self.server.multithread,
            "wsgi.multiprocess": self.server.multiprocess,
            "wsgi.run_once": False,
            "werkzeug.socket": self.connection,
            "SERVER_SOFTWARE": self.server_version,
            "REQUEST_METHOD": self.command,
            "SCRIPT_NAME": "",
            "PATH_INFO": _wsgi_encoding_dance(path_info),
            "QUERY_STRING": _wsgi_encoding_dance(request_url.query),
            # Non-standard, added by mod_wsgi, uWSGI
            "REQUEST_URI": _wsgi_encoding_dance(self.path),
            # Non-standard, added by gunicorn
            "RAW_URI": _wsgi_encoding_dance(self.path),
            "REMOTE_ADDR": self.address_string(),
            "REMOTE_PORT": self.port_integer(),
            "SERVER_NAME": self.server.server_address[0],
            "SERVER_PORT": str(self.server.server_address[1]),
            "SERVER_PROTOCOL": self.request_version,
        }

        for key, value in self.headers.items():
            if "_" in key:
                continue

            key = key.upper().replace("-", "_")
            value = value.replace("\r\n", "")
            if key not in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                key = f"HTTP_{key}"
                if key in environ:
                    value = f"{environ[key]},{value}"
            environ[key] = value

        if environ.get("HTTP_TRANSFER_ENCODING", "").strip().lower() == "chunked":
            environ["wsgi.input_terminated"] = True
            environ["wsgi.input"] = DechunkedInput(environ["wsgi.input"])

        # Per RFC 2616, if the URL is absolute, use that as the host.
        # We're using "has a scheme" to indicate an absolute URL.
        if request_url.scheme and request_url.netloc:
            environ["HTTP_HOST"] = request_url.netloc

        try:
            # binary_form=False gives nicer information, but wouldn't be compatible with
            # what Nginx or Apache could return.
            peer_cert = self.connection.getpeercert(binary_form=True)
            if peer_cert is not None:
                # Nginx and Apache use PEM format.
                environ["SSL_CLIENT_CERT"] = ssl.DER_cert_to_PEM_cert(peer_cert)
        except ValueError:
            # SSL handshake hasn't finished.
            self.server.log("error", "Cannot fetch SSL peer certificate info")
        except AttributeError:
            # Not using TLS, the socket will not have getpeercert().
            pass

        return environ

    def run_wsgi(self) -> None:
        if self.headers.get("Expect", "").lower().strip() == "100-continue":
            self.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")

        self.environ = environ = self.make_environ()
        status_set: str | None = None
        headers_set: list[tuple[str, str]] | None = None
        status_sent: str | None = None
        headers_sent: list[tuple[str, str]] | None = None
        chunk_response: bool = False

        def write(data: bytes) -> None:
            nonlocal status_sent, headers_sent, chunk_response
            assert status_set is not None, "write() before start_response"
            assert headers_set is not None, "write() before start_response"
            if status_sent is None:
                status_sent = status_set
                headers_sent = headers_set
                try:
                    code_str, msg = status_sent.split(None, 1)
                except ValueError:
                    code_str, msg = status_sent, ""
                code = int(code_str)
                self.send_response(code, msg)
                header_keys = set()
                for key, value in headers_sent:
                    self.send_header(key, value)
                    header_keys.add(key.lower())

                # Use chunked transfer encoding if there is no content
                # length. Do not use for 1xx and 204 responses. 304
                # responses and HEAD requests are also excluded, which
                # is the more conservative behavior and matches other
                # parts of the code.
                # https://httpwg.org/specs/rfc7230.html#rfc.section.3.3.1
                if (
                    not (
                        "content-length" in header_keys
                        or environ["REQUEST_METHOD"] == "HEAD"
                        or (100 <= code < 200)
                        or code in {204, 304}
                    )
                    and self.protocol_version >= "HTTP/1.1"
                ):
                    chunk_response = True
                    self.send_header("Transfer-Encoding", "chunked")

                # Always close the connection. This disables HTTP/1.1
                # keep-alive connections. They aren't handled well by
                # Python's http.server because it doesn't know how to
                # drain the stream before the next request line.
                self.send_header("Connection", "close")
                self.end_headers()

            assert isinstance(data, bytes), "applications must write bytes"

            if data:
                if chunk_response:
                    self.wfile.write(hex(len(data))[2:].encode())
                    self.wfile.write(b"\r\n")

                self.wfile.write(data)

                if chunk_response:
                    self.wfile.write(b"\r\n")

            self.wfile.flush()

        def start_response(status, headers, exc_info=None):  # type: ignore
            nonlocal status_set, headers_set
            if exc_info:
                try:
                    if headers_sent:
                        raise exc_info[1].with_traceback(exc_info[2])
                finally:
                    exc_info = None
            elif headers_set:
                raise AssertionError("Headers already set")
            status_set = status
            headers_set = headers
            return write

        def execute(app: WSGIApplication) -> None:
            application_iter = app(environ, start_response)
            try:
                for data in application_iter:
                    write(data)
                if not headers_sent:
                    write(b"")
                if chunk_response:
                    self.wfile.write(b"0\r\n\r\n")
            finally:
                # Check for any remaining data in the read socket, and discard it. This
                # will read past request.max_content_length, but lets the client see a
                # 413 response instead of a connection reset failure. If we supported
                # keep-alive connections, this naive approach would break by reading the
                # next request line. Since we know that write (above) closes every
                # connection we can read everything.
                selector = selectors.DefaultSelector()
                selector.register(self.connection, selectors.EVENT_READ)
                total_size = 0
                total_reads = 0

                # A timeout of 0 tends to fail because a client needs a small amount of
                # time to continue sending its data.
                while selector.select(timeout=0.01):
                    # Only read 10MB into memory at a time.
                    data = self.rfile.read(10_000_000)
                    total_size += len(data)
                    total_reads += 1

                    # Stop reading on no data, >=10GB, or 1000 reads. If a client sends
                    # more than that, they'll get a connection reset failure.
                    if not data or total_size >= 10_000_000_000 or total_reads > 1000:
                        break

                selector.close()

                if hasattr(application_iter, "close"):
                    application_iter.close()

        try:
            execute(self.server.app)
        except connection_dropped_errors as e:
            self.connection_dropped(e, environ)
        except Exception as e:
            if self.server.passthrough_errors:
                raise

            if status_sent is not None and chunk_response:
                self.close_connection = True

            try:
                # if we haven't yet sent the headers but they are set
                # we roll back to be able to set them again.
                if status_sent is None:
                    status_set = None
                    headers_set = None
                execute(InternalServerError())
            except Exception:
                pass

            from .debug.tbtools import DebugTraceback

            msg = DebugTraceback(e).render_traceback_text()
            self.server.log("error", f"Error on request:\n{msg}")

    def handle(self) -> None:
        """Handles a request ignoring dropped connections."""
        try:
            super().handle()
        except (ConnectionError, socket.timeout) as e:
            self.connection_dropped(e)
        except Exception as e:
            if self.server.ssl_context is not None and is_ssl_error(e):
                self.log_error("SSL error occurred: %s", e)
            else:
                raise

    def connection_dropped(
        self, error: BaseException, environ: WSGIEnvironment | None = None
    ) -> None:
        """Called if the connection was closed by the client.  By default
        nothing happens.
        """

    def __getattr__(self, name: str) -> t.Any:
        # All HTTP methods are handled by run_wsgi.
        if name.startswith("do_"):
            return self.run_wsgi

        # All other attributes are forwarded to the base class.
        return getattr(super(), name)

    def address_string(self) -> str:
        if getattr(self, "environ", None):
            return self.environ["REMOTE_ADDR"]  # type: ignore

        if not self.client_address:
            return "<local>"

        return self.client_address[0]

    def port_integer(self) -> int:
        return self.client_address[1]

    # Escape control characters. This is defined (but private) in Python 3.12.
    _control_char_table = str.maketrans(
        {c: rf"\x{c:02x}" for c in [*range(0x20), *range(0x7F, 0xA0)]}
    )
    _control_char_table[ord("\\")] = r"\\"

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        try:
            path = uri_to_iri(self.path)
            msg = f"{self.command} {path} {self.request_version}"
        except AttributeError:
            # path isn't set if the requestline was bad
            msg = self.requestline

        # Escape control characters that may be in the decoded path.
        msg = msg.translate(self._control_char_table)
        code = str(code)

        if code[0] == "1":  # 1xx - Informational
            msg = _ansi_style(msg, "bold")
        elif code == "200":  # 2xx - Success
            pass
        elif code == "304":  # 304 - Resource Not Modified
            msg = _ansi_style(msg, "cyan")
        elif code[0] == "3":  # 3xx - Redirection
            msg = _ansi_style(msg, "green")
        elif code == "404":  # 404 - Resource Not Found
            msg = _ansi_style(msg, "yellow")
        elif code[0] == "4":  # 4xx - Client Error
            msg = _ansi_style(msg, "bold", "red")
        else:  # 5xx, or any other response
            msg = _ansi_style(msg, "bold", "magenta")

        self.log("info", '"%s" %s %s', msg, code, size)

    def log_error(self, format: str, *args: t.Any) -> None:
        self.log("error", format, *args)

    def log_message(self, format: str, *args: t.Any) -> None:
        self.log("info", format, *args)

    def log(self, type: str, message: str, *args: t.Any) -> None:
        # an IPv6 scoped address contains "%" which breaks logging
        address_string = self.address_string().replace("%", "%%")
        _log(
            type,
            f"{address_string} - - [{self.log_date_time_string()}] {message}\n",
            *args,
        )


def _ansi_style(value: str, *styles: str) -> str:
    if not _log_add_style:
        return value

    codes = {
        "bold": 1,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "magenta": 35,
        "cyan": 36,
    }

    for style in styles:
        value = f"\x1b[{codes[style]}m{value}"

    return f"{value}\x1b[0m"


def generate_adhoc_ssl_pair(
    cn: str | None = None,
) -> tuple[Certificate, RSAPrivateKeyWithSerialization]:
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        raise TypeError(
            "Using ad-hoc certificates requires the cryptography library."
        ) from None

    backend = default_backend()
    pkey = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=backend
    )

    # pretty damn sure that this is not actually accepted by anyone
    if cn is None:
        cn = "*"

    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Dummy Certificate"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )

    backend = default_backend()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(pkey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(dt.now(timezone.utc))
        .not_valid_after(dt.now(timezone.utc) + timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([x509.OID_SERVER_AUTH]), critical=False)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(cn), x509.DNSName(f"*.{cn}")]),
            critical=False,
        )
        .sign(pkey, hashes.SHA256(), backend)
    )
    return cert, pkey


def make_ssl_devcert(
    base_path: str, host: str | None = None, cn: str | None = None
) -> tuple[str, str]:
    """Creates an SSL key for development.  This should be used instead of
    the ``'adhoc'`` key which generates a new cert on each server start.
    It accepts a path for where it should store the key and cert and
    either a host or CN.  If a host is given it will use the CN
    ``*.host/CN=host``.

    For more information see :func:`run_simple`.

    .. versionadded:: 0.9

    :param base_path: the path to the certificate and key.  The extension
                      ``.crt`` is added for the certificate, ``.key`` is
                      added for the key.
    :param host: the name of the host.  This can be used as an alternative
                 for the `cn`.
    :param cn: the `CN` to use.
    """

    if host is not None:
        cn = host
    cert, pkey = generate_adhoc_ssl_pair(cn=cn)

    from cryptography.hazmat.primitives import serialization

    cert_file = f"{base_path}.crt"
    pkey_file = f"{base_path}.key"

    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(pkey_file, "wb") as f:
        f.write(
            pkey.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    return cert_file, pkey_file


def generate_adhoc_ssl_context() -> ssl.SSLContext:
    """Generates an adhoc SSL context for the development server."""
    import atexit
    import tempfile

    cert, pkey = generate_adhoc_ssl_pair()

    from cryptography.hazmat.primitives import serialization

    cert_handle, cert_file = tempfile.mkstemp()
    pkey_handle, pkey_file = tempfile.mkstemp()
    atexit.register(os.remove, pkey_file)
    atexit.register(os.remove, cert_file)

    os.write(cert_handle, cert.public_bytes(serialization.Encoding.PEM))
    os.write(
        pkey_handle,
        pkey.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )

    os.close(cert_handle)
    os.close(pkey_handle)
    ctx = load_ssl_context(cert_file, pkey_file)
    return ctx


def load_ssl_context(
    cert_file: str, pkey_file: str | None = None, protocol: int | None = None
) -> ssl.SSLContext:
    """Loads SSL context from cert/private key files and optional protocol.
    Many parameters are directly taken from the API of
    :py:class:`ssl.SSLContext`.

    :param cert_file: Path of the certificate to use.
    :param pkey_file: Path of the private key to use. If not given, the key
                      will be obtained from the certificate file.
    :param protocol: A ``PROTOCOL`` constant from the :mod:`ssl` module.
        Defaults to :data:`ssl.PROTOCOL_TLS_SERVER`.
    """
    if protocol is None:
        protocol = ssl.PROTOCOL_TLS_SERVER

    ctx = ssl.SSLContext(protocol)
    ctx.load_cert_chain(cert_file, pkey_file)
    return ctx


def is_ssl_error(error: Exception | None = None) -> bool:
    """Checks if the given error (or the current one) is an SSL error."""
    if error is None:
        error = t.cast(Exception, sys.exc_info()[1])
    return isinstance(error, ssl.SSLError)


def select_address_family(host: str, port: int) -> socket.AddressFamily:
    """Return ``AF_INET4``, ``AF_INET6``, or ``AF_UNIX`` depending on
    the host and port."""
    if host.startswith("unix://"):
        return socket.AF_UNIX
    elif ":" in host and hasattr(socket, "AF_INET6"):
        return socket.AF_INET6
    return socket.AF_INET


def get_sockaddr(
    host: str, port: int, family: socket.AddressFamily
) -> tuple[str, int] | str:
    """Return a fully qualified socket address that can be passed to
    :func:`socket.bind`."""
    if family == af_unix:
        # Absolute path avoids IDNA encoding error when path starts with dot.
        return os.path.abspath(host.partition("://")[2])
    try:
        res = socket.getaddrinfo(
            host, port, family, socket.SOCK_STREAM, socket.IPPROTO_TCP
        )
    except socket.gaierror:
        return host, port
    return res[0][4]  # type: ignore


def get_interface_ip(family: socket.AddressFamily) -> str:
    """Get the IP address of an external interface. Used when binding to
    0.0.0.0 or ::1 to show a more useful URL.

    :meta private:
    """
    # arbitrary private address
    host = "fd31:f903:5ab5:1::1" if family == socket.AF_INET6 else "10.253.155.219"

    with socket.socket(family, socket.SOCK_DGRAM) as s:
        try:
            s.connect((host, 58162))
        except OSError:
            return "::1" if family == socket.AF_INET6 else "127.0.0.1"

        return s.getsockname()[0]  # type: ignore


class BaseWSGIServer(HTTPServer):
    """A WSGI server that that handles one request at a time.

    Use :func:`make_server` to create a server instance.
    """

    multithread = False
    multiprocess = False
    request_queue_size = LISTEN_QUEUE
    allow_reuse_address = True

    def __init__(
        self,
        host: str,
        port: int,
        app: WSGIApplication,
        handler: type[WSGIRequestHandler] | None = None,
        passthrough_errors: bool = False,
        ssl_context: _TSSLContextArg | None = None,
        fd: int | None = None,
    ) -> None:
        if handler is None:
            handler = WSGIRequestHandler

        # If the handler doesn't directly set a protocol version and
        # thread or process workers are used, then allow chunked
        # responses and keep-alive connections by enabling HTTP/1.1.
        if "protocol_version" not in vars(handler) and (
            self.multithread or self.multiprocess
        ):
            handler.protocol_version = "HTTP/1.1"

        self.host = host
        self.port = port
        self.app = app
        self.passthrough_errors = passthrough_errors

        self.address_family = address_family = select_address_family(host, port)
        server_address = get_sockaddr(host, int(port), address_family)

        # Remove a leftover Unix socket file from a previous run. Don't
        # remove a file that was set up by run_simple.
        if address_family == af_unix and fd is None:
            server_address = t.cast(str, server_address)

            if os.path.exists(server_address):
                os.unlink(server_address)

        # Bind and activate will be handled manually, it should only
        # happen if we're not using a socket that was already set up.
        super().__init__(
            server_address,  # type: ignore[arg-type]
            handler,
            bind_and_activate=False,
        )

        if fd is None:
            # No existing socket descriptor, do bind_and_activate=True.
            try:
                self.server_bind()
                self.server_activate()
            except OSError as e:
                # Catch connection issues and show them without the traceback. Show
                # extra instructions for address not found, and for macOS.
                self.server_close()
                print(e.strerror, file=sys.stderr)

                if e.errno == errno.EADDRINUSE:
                    print(
                        f"Port {port} is in use by another program. Either identify and"
                        " stop that program, or start the server with a different"
                        " port.",
                        file=sys.stderr,
                    )

                    if sys.platform == "darwin" and port == 5000:
                        print(
                            "On macOS, try disabling the 'AirPlay Receiver' service"
                            " from System Preferences -> General -> AirDrop & Handoff.",
                            file=sys.stderr,
                        )

                sys.exit(1)
            except BaseException:
                self.server_close()
                raise
        else:
            # TCPServer automatically opens a socket even if bind_and_activate is False.
            # Close it to silence a ResourceWarning.
            self.server_close()

            # Use the passed in socket directly.
            self.socket = socket.fromfd(fd, address_family, socket.SOCK_STREAM)
            self.server_address = self.socket.getsockname()

        if address_family != af_unix:
            # If port was 0, this will record the bound port.
            self.port = self.server_address[1]

        if ssl_context is not None:
            if isinstance(ssl_context, tuple):
                ssl_context = load_ssl_context(*ssl_context)
            elif ssl_context == "adhoc":
                ssl_context = generate_adhoc_ssl_context()

            self.socket = ssl_context.wrap_socket(self.socket, server_side=True)
            self.ssl_context: ssl.SSLContext | None = ssl_context
        else:
            self.ssl_context = None

        import importlib.metadata

        self._server_version = f"Werkzeug/{importlib.metadata.version('werkzeug')}"

    def log(self, type: str, message: str, *args: t.Any) -> None:
        _log(type, message, *args)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        try:
            super().serve_forever(poll_interval=poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.server_close()

    def handle_error(
        self, request: t.Any, client_address: tuple[str, int] | str
    ) -> None:
        if self.passthrough_errors:
            raise

        return super().handle_error(request, client_address)

    def log_startup(self) -> None:
        """Show information about the address when starting the server."""
        dev_warning = (
            "WARNING: This is a development server. Do not use it in a production"
            " deployment. Use a production WSGI server instead."
        )
        dev_warning = _ansi_style(dev_warning, "bold", "red")
        messages = [dev_warning]

        if self.address_family == af_unix:
            messages.append(f" * Running on {self.host}")
        else:
            scheme = "http" if self.ssl_context is None else "https"
            display_hostname = self.host

            if self.host in {"0.0.0.0", "::"}:
                messages.append(f" * Running on all addresses ({self.host})")

                if self.host == "0.0.0.0":
                    localhost = "127.0.0.1"
                    display_hostname = get_interface_ip(socket.AF_INET)
                else:
                    localhost = "[::1]"
                    display_hostname = get_interface_ip(socket.AF_INET6)

                messages.append(f" * Running on {scheme}://{localhost}:{self.port}")

            if ":" in display_hostname:
                display_hostname = f"[{display_hostname}]"

            messages.append(f" * Running on {scheme}://{display_hostname}:{self.port}")

        _log("info", "\n".join(messages))


class ThreadedWSGIServer(socketserver.ThreadingMixIn, BaseWSGIServer):
    """A WSGI server that handles concurrent requests in separate
    threads.

    Use :func:`make_server` to create a server instance.
    """

    multithread = True
    daemon_threads = True


class ForkingWSGIServer(ForkingMixIn, BaseWSGIServer):
    """A WSGI server that handles concurrent requests in separate forked
    processes.

    Use :func:`make_server` to create a server instance.
    """

    multiprocess = True

    def __init__(
        self,
        host: str,
        port: int,
        app: WSGIApplication,
        processes: int = 40,
        handler: type[WSGIRequestHandler] | None = None,
        passthrough_errors: bool = False,
        ssl_context: _TSSLContextArg | None = None,
        fd: int | None = None,
    ) -> None:
        if not can_fork:
            raise ValueError("Your platform does not support forking.")

        super().__init__(host, port, app, handler, passthrough_errors, ssl_context, fd)
        self.max_children = processes


def make_server(
    host: str,
    port: int,
    app: WSGIApplication,
    threaded: bool = False,
    processes: int = 1,
    request_handler: type[WSGIRequestHandler] | None = None,
    passthrough_errors: bool = False,
    ssl_context: _TSSLContextArg | None = None,
    fd: int | None = None,
) -> BaseWSGIServer:
    """Create an appropriate WSGI server instance based on the value of
    ``threaded`` and ``processes``.

    This is called from :func:`run_simple`, but can be used separately
    to have access to the server object, such as to run it in a separate
    thread.

    See :func:`run_simple` for parameter docs.
    """
    if threaded and processes > 1:
        raise ValueError("Cannot have a multi-thread and multi-process server.")

    if threaded:
        return ThreadedWSGIServer(
            host, port, app, request_handler, passthrough_errors, ssl_context, fd=fd
        )

    if processes > 1:
        return ForkingWSGIServer(
            host,
            port,
            app,
            processes,
            request_handler,
            passthrough_errors,
            ssl_context,
            fd=fd,
        )

    return BaseWSGIServer(
        host, port, app, request_handler, passthrough_errors, ssl_context, fd=fd
    )


def is_running_from_reloader() -> bool:
    """Check if the server is running as a subprocess within the
    Werkzeug reloader.

    .. versionadded:: 0.10
    """
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def run_simple(
    hostname: str,
    port: int,
    application: WSGIApplication,
    use_reloader: bool = False,
    use_debugger: bool = False,
    use_evalex: bool = True,
    extra_files: t.Iterable[str] | None = None,
    exclude_patterns: t.Iterable[str] | None = None,
    reloader_interval: int = 1,
    reloader_type: str = "auto",
    threaded: bool = False,
    processes: int = 1,
    request_handler: type[WSGIRequestHandler] | None = None,
    static_files: dict[str, str | tuple[str, str]] | None = None,
    passthrough_errors: bool = False,
    ssl_context: _TSSLContextArg | None = None,
) -> None:
    """Start a development server for a WSGI application. Various
    optional features can be enabled.

    .. warning::

        Do not use the development server when deploying to production.
        It is intended for use only during local development. It is not
        designed to be particularly efficient, stable, or secure.

    :param hostname: The host to bind to, for example ``'localhost'``.
        Can be a domain, IPv4 or IPv6 address, or file path starting
        with ``unix://`` for a Unix socket.
    :param port: The port to bind to, for example ``8080``. Using ``0``
        tells the OS to pick a random free port.
    :param application: The WSGI application to run.
    :param use_reloader: Use a reloader process to restart the server
        process when files are changed.
    :param use_debugger: Use Werkzeug's debugger, which will show
        formatted tracebacks on unhandled exceptions.
    :param use_evalex: Make the debugger interactive. A Python terminal
        can be opened for any frame in the traceback. Some protection is
        provided by requiring a PIN, but this should never be enabled
        on a publicly visible server.
    :param extra_files: The reloader will watch these files for changes
        in addition to Python modules. For example, watch a
        configuration file.
    :param exclude_patterns: The reloader will ignore changes to any
        files matching these :mod:`fnmatch` patterns. For example,
        ignore cache files.
    :param reloader_interval: How often the reloader tries to check for
        changes.
    :param reloader_type: The reloader to use. The ``'stat'`` reloader
        is built in, but may require significant CPU to watch files. The
        ``'watchdog'`` reloader is much more efficient but requires
        installing the ``watchdog`` package first.
    :param threaded: Handle concurrent requests using threads. Cannot be
        used with ``processes``.
    :param processes: Handle concurrent requests using up to this number
        of processes. Cannot be used with ``threaded``.
    :param request_handler: Use a different
        :class:`~BaseHTTPServer.BaseHTTPRequestHandler` subclass to
        handle requests.
    :param static_files: A dict mapping URL prefixes to directories to
        serve static files from using
        :class:`~werkzeug.middleware.SharedDataMiddleware`.
    :param passthrough_errors: Don't catch unhandled exceptions at the
        server level, let the server crash instead. If ``use_debugger``
        is enabled, the debugger will still catch such errors.
    :param ssl_context: Configure TLS to serve over HTTPS. Can be an
        :class:`ssl.SSLContext` object, a ``(cert_file, key_file)``
        tuple to create a typical context, or the string ``'adhoc'`` to
        generate a temporary self-signed certificate.

    .. versionchanged:: 2.1
        Instructions are shown for dealing with an "address already in
        use" error.

    .. versionchanged:: 2.1
        Running on ``0.0.0.0`` or ``::`` shows the loopback IP in
        addition to a real IP.

    .. versionchanged:: 2.1
        The command-line interface was removed.

    .. versionchanged:: 2.0
        Running on ``0.0.0.0`` or ``::`` shows a real IP address that
        was bound as well as a warning not to run the development server
        in production.

    .. versionchanged:: 2.0
        The ``exclude_patterns`` parameter was added.

    .. versionchanged:: 0.15
        Bind to a Unix socket by passing a ``hostname`` that starts with
        ``unix://``.

    .. versionchanged:: 0.10
        Improved the reloader and added support for changing the backend
        through the ``reloader_type`` parameter.

    .. versionchanged:: 0.9
        A command-line interface was added.

    .. versionchanged:: 0.8
        ``ssl_context`` can be a tuple of paths to the certificate and
        private key files.

    .. versionchanged:: 0.6
        The ``ssl_context`` parameter was added.

    .. versionchanged:: 0.5
       The ``static_files`` and ``passthrough_errors`` parameters were
       added.
    """
    if not isinstance(port, int):
        raise TypeError("port must be an integer")

    if static_files:
        from .middleware.shared_data import SharedDataMiddleware

        application = SharedDataMiddleware(application, static_files)

    if use_debugger:
        from .debug import DebuggedApplication

        application = DebuggedApplication(application, evalex=use_evalex)
        # Allow the specified hostname to use the debugger, in addition to
        # localhost domains.
        application.trusted_hosts.append(hostname)

    if not is_running_from_reloader():
        fd = None
    else:
        fd = int(os.environ["WERKZEUG_SERVER_FD"])

    srv = make_server(
        hostname,
        port,
        application,
        threaded,
        processes,
        request_handler,
        passthrough_errors,
        ssl_context,
        fd=fd,
    )
    srv.socket.set_inheritable(True)
    os.environ["WERKZEUG_SERVER_FD"] = str(srv.fileno())

    if not is_running_from_reloader():
        srv.log_startup()
        _log("info", _ansi_style("Press CTRL+C to quit", "yellow"))

    if use_reloader:
        from ._reloader import run_with_reloader

        try:
            run_with_reloader(
                srv.serve_forever,
                extra_files=extra_files,
                exclude_patterns=exclude_patterns,
                interval=reloader_interval,
                reloader_type=reloader_type,
            )
        finally:
            srv.server_close()
    else:
        srv.serve_forever()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\backend_ctypes.py ===
import ctypes, ctypes.util, operator, sys
from . import model

if sys.version_info < (3,):
    bytechr = chr
else:
    unicode = str
    long = int
    xrange = range
    bytechr = lambda num: bytes([num])

class CTypesType(type):
    pass

class CTypesData(object):
    __metaclass__ = CTypesType
    __slots__ = ['__weakref__']
    __name__ = '<cdata>'

    def __init__(self, *args):
        raise TypeError("cannot instantiate %r" % (self.__class__,))

    @classmethod
    def _newp(cls, init):
        raise TypeError("expected a pointer or array ctype, got '%s'"
                        % (cls._get_c_name(),))

    @staticmethod
    def _to_ctypes(value):
        raise TypeError

    @classmethod
    def _arg_to_ctypes(cls, *value):
        try:
            ctype = cls._ctype
        except AttributeError:
            raise TypeError("cannot create an instance of %r" % (cls,))
        if value:
            res = cls._to_ctypes(*value)
            if not isinstance(res, ctype):
                res = cls._ctype(res)
        else:
            res = cls._ctype()
        return res

    @classmethod
    def _create_ctype_obj(cls, init):
        if init is None:
            return cls._arg_to_ctypes()
        else:
            return cls._arg_to_ctypes(init)

    @staticmethod
    def _from_ctypes(ctypes_value):
        raise TypeError

    @classmethod
    def _get_c_name(cls, replace_with=''):
        return cls._reftypename.replace(' &', replace_with)

    @classmethod
    def _fix_class(cls):
        cls.__name__ = 'CData<%s>' % (cls._get_c_name(),)
        cls.__qualname__ = 'CData<%s>' % (cls._get_c_name(),)
        cls.__module__ = 'ffi'

    def _get_own_repr(self):
        raise NotImplementedError

    def _addr_repr(self, address):
        if address == 0:
            return 'NULL'
        else:
            if address < 0:
                address += 1 << (8*ctypes.sizeof(ctypes.c_void_p))
            return '0x%x' % address

    def __repr__(self, c_name=None):
        own = self._get_own_repr()
        return '<cdata %r %s>' % (c_name or self._get_c_name(), own)

    def _convert_to_address(self, BClass):
        if BClass is None:
            raise TypeError("cannot convert %r to an address" % (
                self._get_c_name(),))
        else:
            raise TypeError("cannot convert %r to %r" % (
                self._get_c_name(), BClass._get_c_name()))

    @classmethod
    def _get_size(cls):
        return ctypes.sizeof(cls._ctype)

    def _get_size_of_instance(self):
        return ctypes.sizeof(self._ctype)

    @classmethod
    def _cast_from(cls, source):
        raise TypeError("cannot cast to %r" % (cls._get_c_name(),))

    def _cast_to_integer(self):
        return self._convert_to_address(None)

    @classmethod
    def _alignment(cls):
        return ctypes.alignment(cls._ctype)

    def __iter__(self):
        raise TypeError("cdata %r does not support iteration" % (
            self._get_c_name()),)

    def _make_cmp(name):
        cmpfunc = getattr(operator, name)
        def cmp(self, other):
            v_is_ptr = not isinstance(self, CTypesGenericPrimitive)
            w_is_ptr = (isinstance(other, CTypesData) and
                           not isinstance(other, CTypesGenericPrimitive))
            if v_is_ptr and w_is_ptr:
                return cmpfunc(self._convert_to_address(None),
                               other._convert_to_address(None))
            elif v_is_ptr or w_is_ptr:
                return NotImplemented
            else:
                if isinstance(self, CTypesGenericPrimitive):
                    self = self._value
                if isinstance(other, CTypesGenericPrimitive):
                    other = other._value
                return cmpfunc(self, other)
        cmp.func_name = name
        return cmp

    __eq__ = _make_cmp('__eq__')
    __ne__ = _make_cmp('__ne__')
    __lt__ = _make_cmp('__lt__')
    __le__ = _make_cmp('__le__')
    __gt__ = _make_cmp('__gt__')
    __ge__ = _make_cmp('__ge__')

    def __hash__(self):
        return hash(self._convert_to_address(None))

    def _to_string(self, maxlen):
        raise TypeError("string(): %r" % (self,))


class CTypesGenericPrimitive(CTypesData):
    __slots__ = []

    def __hash__(self):
        return hash(self._value)

    def _get_own_repr(self):
        return repr(self._from_ctypes(self._value))


class CTypesGenericArray(CTypesData):
    __slots__ = []

    @classmethod
    def _newp(cls, init):
        return cls(init)

    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]

    def _get_own_repr(self):
        return self._addr_repr(ctypes.addressof(self._blob))


class CTypesGenericPtr(CTypesData):
    __slots__ = ['_address', '_as_ctype_ptr']
    _automatic_casts = False
    kind = "pointer"

    @classmethod
    def _newp(cls, init):
        return cls(init)

    @classmethod
    def _cast_from(cls, source):
        if source is None:
            address = 0
        elif isinstance(source, CTypesData):
            address = source._cast_to_integer()
        elif isinstance(source, (int, long)):
            address = source
        else:
            raise TypeError("bad type for cast to %r: %r" %
                            (cls, type(source).__name__))
        return cls._new_pointer_at(address)

    @classmethod
    def _new_pointer_at(cls, address):
        self = cls.__new__(cls)
        self._address = address
        self._as_ctype_ptr = ctypes.cast(address, cls._ctype)
        return self

    def _get_own_repr(self):
        try:
            return self._addr_repr(self._address)
        except AttributeError:
            return '???'

    def _cast_to_integer(self):
        return self._address

    def __nonzero__(self):
        return bool(self._address)
    __bool__ = __nonzero__

    @classmethod
    def _to_ctypes(cls, value):
        if not isinstance(value, CTypesData):
            raise TypeError("unexpected %s object" % type(value).__name__)
        address = value._convert_to_address(cls)
        return ctypes.cast(address, cls._ctype)

    @classmethod
    def _from_ctypes(cls, ctypes_ptr):
        address = ctypes.cast(ctypes_ptr, ctypes.c_void_p).value or 0
        return cls._new_pointer_at(address)

    @classmethod
    def _initialize(cls, ctypes_ptr, value):
        if value:
            ctypes_ptr.contents = cls._to_ctypes(value).contents

    def _convert_to_address(self, BClass):
        if (BClass in (self.__class__, None) or BClass._automatic_casts
            or self._automatic_casts):
            return self._address
        else:
            return CTypesData._convert_to_address(self, BClass)


class CTypesBaseStructOrUnion(CTypesData):
    __slots__ = ['_blob']

    @classmethod
    def _create_ctype_obj(cls, init):
        # may be overridden
        raise TypeError("cannot instantiate opaque type %s" % (cls,))

    def _get_own_repr(self):
        return self._addr_repr(ctypes.addressof(self._blob))

    @classmethod
    def _offsetof(cls, fieldname):
        return getattr(cls._ctype, fieldname).offset

    def _convert_to_address(self, BClass):
        if getattr(BClass, '_BItem', None) is self.__class__:
            return ctypes.addressof(self._blob)
        else:
            return CTypesData._convert_to_address(self, BClass)

    @classmethod
    def _from_ctypes(cls, ctypes_struct_or_union):
        self = cls.__new__(cls)
        self._blob = ctypes_struct_or_union
        return self

    @classmethod
    def _to_ctypes(cls, value):
        return value._blob

    def __repr__(self, c_name=None):
        return CTypesData.__repr__(self, c_name or self._get_c_name(' &'))


class CTypesBackend(object):

    PRIMITIVE_TYPES = {
        'char': ctypes.c_char,
        'short': ctypes.c_short,
        'int': ctypes.c_int,
        'long': ctypes.c_long,
        'long long': ctypes.c_longlong,
        'signed char': ctypes.c_byte,
        'unsigned char': ctypes.c_ubyte,
        'unsigned short': ctypes.c_ushort,
        'unsigned int': ctypes.c_uint,
        'unsigned long': ctypes.c_ulong,
        'unsigned long long': ctypes.c_ulonglong,
        'float': ctypes.c_float,
        'double': ctypes.c_double,
        '_Bool': ctypes.c_bool,
        }

    for _name in ['unsigned long long', 'unsigned long',
                  'unsigned int', 'unsigned short', 'unsigned char']:
        _size = ctypes.sizeof(PRIMITIVE_TYPES[_name])
        PRIMITIVE_TYPES['uint%d_t' % (8*_size)] = PRIMITIVE_TYPES[_name]
        if _size == ctypes.sizeof(ctypes.c_void_p):
            PRIMITIVE_TYPES['uintptr_t'] = PRIMITIVE_TYPES[_name]
        if _size == ctypes.sizeof(ctypes.c_size_t):
            PRIMITIVE_TYPES['size_t'] = PRIMITIVE_TYPES[_name]

    for _name in ['long long', 'long', 'int', 'short', 'signed char']:
        _size = ctypes.sizeof(PRIMITIVE_TYPES[_name])
        PRIMITIVE_TYPES['int%d_t' % (8*_size)] = PRIMITIVE_TYPES[_name]
        if _size == ctypes.sizeof(ctypes.c_void_p):
            PRIMITIVE_TYPES['intptr_t'] = PRIMITIVE_TYPES[_name]
            PRIMITIVE_TYPES['ptrdiff_t'] = PRIMITIVE_TYPES[_name]
        if _size == ctypes.sizeof(ctypes.c_size_t):
            PRIMITIVE_TYPES['ssize_t'] = PRIMITIVE_TYPES[_name]


    def __init__(self):
        self.RTLD_LAZY = 0   # not supported anyway by ctypes
        self.RTLD_NOW  = 0
        self.RTLD_GLOBAL = ctypes.RTLD_GLOBAL
        self.RTLD_LOCAL = ctypes.RTLD_LOCAL

    def set_ffi(self, ffi):
        self.ffi = ffi

    def _get_types(self):
        return CTypesData, CTypesType

    def load_library(self, path, flags=0):
        cdll = ctypes.CDLL(path, flags)
        return CTypesLibrary(self, cdll)

    def new_void_type(self):
        class CTypesVoid(CTypesData):
            __slots__ = []
            _reftypename = 'void &'
            @staticmethod
            def _from_ctypes(novalue):
                return None
            @staticmethod
            def _to_ctypes(novalue):
                if novalue is not None:
                    raise TypeError("None expected, got %s object" %
                                    (type(novalue).__name__,))
                return None
        CTypesVoid._fix_class()
        return CTypesVoid

    def new_primitive_type(self, name):
        if name == 'wchar_t':
            raise NotImplementedError(name)
        ctype = self.PRIMITIVE_TYPES[name]
        if name == 'char':
            kind = 'char'
        elif name in ('float', 'double'):
            kind = 'float'
        else:
            if name in ('signed char', 'unsigned char'):
                kind = 'byte'
            elif name == '_Bool':
                kind = 'bool'
            else:
                kind = 'int'
            is_signed = (ctype(-1).value == -1)
        #
        def _cast_source_to_int(source):
            if isinstance(source, (int, long, float)):
                source = int(source)
            elif isinstance(source, CTypesData):
                source = source._cast_to_integer()
            elif isinstance(source, bytes):
                source = ord(source)
            elif source is None:
                source = 0
            else:
                raise TypeError("bad type for cast to %r: %r" %
                                (CTypesPrimitive, type(source).__name__))
            return source
        #
        kind1 = kind
        class CTypesPrimitive(CTypesGenericPrimitive):
            __slots__ = ['_value']
            _ctype = ctype
            _reftypename = '%s &' % name
            kind = kind1

            def __init__(self, value):
                self._value = value

            @staticmethod
            def _create_ctype_obj(init):
                if init is None:
                    return ctype()
                return ctype(CTypesPrimitive._to_ctypes(init))

            if kind == 'int' or kind == 'byte':
                @classmethod
                def _cast_from(cls, source):
                    source = _cast_source_to_int(source)
                    source = ctype(source).value     # cast within range
                    return cls(source)
                def __int__(self):
                    return self._value

            if kind == 'bool':
                @classmethod
                def _cast_from(cls, source):
                    if not isinstance(source, (int, long, float)):
                        source = _cast_source_to_int(source)
                    return cls(bool(source))
                def __int__(self):
                    return int(self._value)

            if kind == 'char':
                @classmethod
                def _cast_from(cls, source):
                    source = _cast_source_to_int(source)
                    source = bytechr(source & 0xFF)
                    return cls(source)
                def __int__(self):
                    return ord(self._value)

            if kind == 'float':
                @classmethod
                def _cast_from(cls, source):
                    if isinstance(source, float):
                        pass
                    elif isinstance(source, CTypesGenericPrimitive):
                        if hasattr(source, '__float__'):
                            source = float(source)
                        else:
                            source = int(source)
                    else:
                        source = _cast_source_to_int(source)
                    source = ctype(source).value     # fix precision
                    return cls(source)
                def __int__(self):
                    return int(self._value)
                def __float__(self):
                    return self._value

            _cast_to_integer = __int__

            if kind == 'int' or kind == 'byte' or kind == 'bool':
                @staticmethod
                def _to_ctypes(x):
                    if not isinstance(x, (int, long)):
                        if isinstance(x, CTypesData):
                            x = int(x)
                        else:
                            raise TypeError("integer expected, got %s" %
                                            type(x).__name__)
                    if ctype(x).value != x:
                        if not is_signed and x < 0:
                            raise OverflowError("%s: negative integer" % name)
                        else:
                            raise OverflowError("%s: integer out of bounds"
                                                % name)
                    return x

            if kind == 'char':
                @staticmethod
                def _to_ctypes(x):
                    if isinstance(x, bytes) and len(x) == 1:
                        return x
                    if isinstance(x, CTypesPrimitive):    # <CData <char>>
                        return x._value
                    raise TypeError("character expected, got %s" %
                                    type(x).__name__)
                def __nonzero__(self):
                    return ord(self._value) != 0
            else:
                def __nonzero__(self):
                    return self._value != 0
            __bool__ = __nonzero__

            if kind == 'float':
                @staticmethod
                def _to_ctypes(x):
                    if not isinstance(x, (int, long, float, CTypesData)):
                        raise TypeError("float expected, got %s" %
                                        type(x).__name__)
                    return ctype(x).value

            @staticmethod
            def _from_ctypes(value):
                return getattr(value, 'value', value)

            @staticmethod
            def _initialize(blob, init):
                blob.value = CTypesPrimitive._to_ctypes(init)

            if kind == 'char':
                def _to_string(self, maxlen):
                    return self._value
            if kind == 'byte':
                def _to_string(self, maxlen):
                    return chr(self._value & 0xff)
        #
        CTypesPrimitive._fix_class()
        return CTypesPrimitive

    def new_pointer_type(self, BItem):
        getbtype = self.ffi._get_cached_btype
        if BItem is getbtype(model.PrimitiveType('char')):
            kind = 'charp'
        elif BItem in (getbtype(model.PrimitiveType('signed char')),
                       getbtype(model.PrimitiveType('unsigned char'))):
            kind = 'bytep'
        elif BItem is getbtype(model.void_type):
            kind = 'voidp'
        else:
            kind = 'generic'
        #
        class CTypesPtr(CTypesGenericPtr):
            __slots__ = ['_own']
            if kind == 'charp':
                __slots__ += ['__as_strbuf']
            _BItem = BItem
            if hasattr(BItem, '_ctype'):
                _ctype = ctypes.POINTER(BItem._ctype)
                _bitem_size = ctypes.sizeof(BItem._ctype)
            else:
                _ctype = ctypes.c_void_p
            if issubclass(BItem, CTypesGenericArray):
                _reftypename = BItem._get_c_name('(* &)')
            else:
                _reftypename = BItem._get_c_name(' * &')

            def __init__(self, init):
                ctypeobj = BItem._create_ctype_obj(init)
                if kind == 'charp':
                    self.__as_strbuf = ctypes.create_string_buffer(
                        ctypeobj.value + b'\x00')
                    self._as_ctype_ptr = ctypes.cast(
                        self.__as_strbuf, self._ctype)
                else:
                    self._as_ctype_ptr = ctypes.pointer(ctypeobj)
                self._address = ctypes.cast(self._as_ctype_ptr,
                                            ctypes.c_void_p).value
                self._own = True

            def __add__(self, other):
                if isinstance(other, (int, long)):
                    return self._new_pointer_at(self._address +
                                                other * self._bitem_size)
                else:
                    return NotImplemented

            def __sub__(self, other):
                if isinstance(other, (int, long)):
                    return self._new_pointer_at(self._address -
                                                other * self._bitem_size)
                elif type(self) is type(other):
                    return (self._address - other._address) // self._bitem_size
                else:
                    return NotImplemented

            def __getitem__(self, index):
                if getattr(self, '_own', False) and index != 0:
                    raise IndexError
                return BItem._from_ctypes(self._as_ctype_ptr[index])

            def __setitem__(self, index, value):
                self._as_ctype_ptr[index] = BItem._to_ctypes(value)

            if kind == 'charp' or kind == 'voidp':
                @classmethod
                def _arg_to_ctypes(cls, *value):
                    if value and isinstance(value[0], bytes):
                        return ctypes.c_char_p(value[0])
                    else:
                        return super(CTypesPtr, cls)._arg_to_ctypes(*value)

            if kind == 'charp' or kind == 'bytep':
                def _to_string(self, maxlen):
                    if maxlen < 0:
                        maxlen = sys.maxsize
                    p = ctypes.cast(self._as_ctype_ptr,
                                    ctypes.POINTER(ctypes.c_char))
                    n = 0
                    while n < maxlen and p[n] != b'\x00':
                        n += 1
                    return b''.join([p[i] for i in range(n)])

            def _get_own_repr(self):
                if getattr(self, '_own', False):
                    return 'owning %d bytes' % (
                        ctypes.sizeof(self._as_ctype_ptr.contents),)
                return super(CTypesPtr, self)._get_own_repr()
        #
        if (BItem is self.ffi._get_cached_btype(model.void_type) or
            BItem is self.ffi._get_cached_btype(model.PrimitiveType('char'))):
            CTypesPtr._automatic_casts = True
        #
        CTypesPtr._fix_class()
        return CTypesPtr

    def new_array_type(self, CTypesPtr, length):
        if length is None:
            brackets = ' &[]'
        else:
            brackets = ' &[%d]' % length
        BItem = CTypesPtr._BItem
        getbtype = self.ffi._get_cached_btype
        if BItem is getbtype(model.PrimitiveType('char')):
            kind = 'char'
        elif BItem in (getbtype(model.PrimitiveType('signed char')),
                       getbtype(model.PrimitiveType('unsigned char'))):
            kind = 'byte'
        else:
            kind = 'generic'
        #
        class CTypesArray(CTypesGenericArray):
            __slots__ = ['_blob', '_own']
            if length is not None:
                _ctype = BItem._ctype * length
            else:
                __slots__.append('_ctype')
            _reftypename = BItem._get_c_name(brackets)
            _declared_length = length
            _CTPtr = CTypesPtr

            def __init__(self, init):
                if length is None:
                    if isinstance(init, (int, long)):
                        len1 = init
                        init = None
                    elif kind == 'char' and isinstance(init, bytes):
                        len1 = len(init) + 1    # extra null
                    else:
                        init = tuple(init)
                        len1 = len(init)
                    self._ctype = BItem._ctype * len1
                self._blob = self._ctype()
                self._own = True
                if init is not None:
                    self._initialize(self._blob, init)

            @staticmethod
            def _initialize(blob, init):
                if isinstance(init, bytes):
                    init = [init[i:i+1] for i in range(len(init))]
                else:
                    if isinstance(init, CTypesGenericArray):
                        if (len(init) != len(blob) or
                            not isinstance(init, CTypesArray)):
                            raise TypeError("length/type mismatch: %s" % (init,))
                    init = tuple(init)
                if len(init) > len(blob):
                    raise IndexError("too many initializers")
                addr = ctypes.cast(blob, ctypes.c_void_p).value
                PTR = ctypes.POINTER(BItem._ctype)
                itemsize = ctypes.sizeof(BItem._ctype)
                for i, value in enumerate(init):
                    p = ctypes.cast(addr + i * itemsize, PTR)
                    BItem._initialize(p.contents, value)

            def __len__(self):
                return len(self._blob)

            def __getitem__(self, index):
                if not (0 <= index < len(self._blob)):
                    raise IndexError
                return BItem._from_ctypes(self._blob[index])

            def __setitem__(self, index, value):
                if not (0 <= index < len(self._blob)):
                    raise IndexError
                self._blob[index] = BItem._to_ctypes(value)

            if kind == 'char' or kind == 'byte':
                def _to_string(self, maxlen):
                    if maxlen < 0:
                        maxlen = len(self._blob)
                    p = ctypes.cast(self._blob,
                                    ctypes.POINTER(ctypes.c_char))
                    n = 0
                    while n < maxlen and p[n] != b'\x00':
                        n += 1
                    return b''.join([p[i] for i in range(n)])

            def _get_own_repr(self):
                if getattr(self, '_own', False):
                    return 'owning %d bytes' % (ctypes.sizeof(self._blob),)
                return super(CTypesArray, self)._get_own_repr()

            def _convert_to_address(self, BClass):
                if BClass in (CTypesPtr, None) or BClass._automatic_casts:
                    return ctypes.addressof(self._blob)
                else:
                    return CTypesData._convert_to_address(self, BClass)

            @staticmethod
            def _from_ctypes(ctypes_array):
                self = CTypesArray.__new__(CTypesArray)
                self._blob = ctypes_array
                return self

            @staticmethod
            def _arg_to_ctypes(value):
                return CTypesPtr._arg_to_ctypes(value)

            def __add__(self, other):
                if isinstance(other, (int, long)):
                    return CTypesPtr._new_pointer_at(
                        ctypes.addressof(self._blob) +
                        other * ctypes.sizeof(BItem._ctype))
                else:
                    return NotImplemented

            @classmethod
            def _cast_from(cls, source):
                raise NotImplementedError("casting to %r" % (
                    cls._get_c_name(),))
        #
        CTypesArray._fix_class()
        return CTypesArray

    def _new_struct_or_union(self, kind, name, base_ctypes_class):
        #
        class struct_or_union(base_ctypes_class):
            pass
        struct_or_union.__name__ = '%s_%s' % (kind, name)
        kind1 = kind
        #
        class CTypesStructOrUnion(CTypesBaseStructOrUnion):
            __slots__ = ['_blob']
            _ctype = struct_or_union
            _reftypename = '%s &' % (name,)
            _kind = kind = kind1
        #
        CTypesStructOrUnion._fix_class()
        return CTypesStructOrUnion

    def new_struct_type(self, name):
        return self._new_struct_or_union('struct', name, ctypes.Structure)

    def new_union_type(self, name):
        return self._new_struct_or_union('union', name, ctypes.Union)

    def complete_struct_or_union(self, CTypesStructOrUnion, fields, tp,
                                 totalsize=-1, totalalignment=-1, sflags=0,
                                 pack=0):
        if totalsize >= 0 or totalalignment >= 0:
            raise NotImplementedError("the ctypes backend of CFFI does not support "
                                      "structures completed by verify(); please "
                                      "compile and install the _cffi_backend module.")
        struct_or_union = CTypesStructOrUnion._ctype
        fnames = [fname for (fname, BField, bitsize) in fields]
        btypes = [BField for (fname, BField, bitsize) in fields]
        bitfields = [bitsize for (fname, BField, bitsize) in fields]
        #
        bfield_types = {}
        cfields = []
        for (fname, BField, bitsize) in fields:
            if bitsize < 0:
                cfields.append((fname, BField._ctype))
                bfield_types[fname] = BField
            else:
                cfields.append((fname, BField._ctype, bitsize))
                bfield_types[fname] = Ellipsis
        if sflags & 8:
            struct_or_union._pack_ = 1
        elif pack:
            struct_or_union._pack_ = pack
        struct_or_union._fields_ = cfields
        CTypesStructOrUnion._bfield_types = bfield_types
        #
        @staticmethod
        def _create_ctype_obj(init):
            result = struct_or_union()
            if init is not None:
                initialize(result, init)
            return result
        CTypesStructOrUnion._create_ctype_obj = _create_ctype_obj
        #
        def initialize(blob, init):
            if is_union:
                if len(init) > 1:
                    raise ValueError("union initializer: %d items given, but "
                                    "only one supported (use a dict if needed)"
                                     % (len(init),))
            if not isinstance(init, dict):
                if isinstance(init, (bytes, unicode)):
                    raise TypeError("union initializer: got a str")
                init = tuple(init)
                if len(init) > len(fnames):
                    raise ValueError("too many values for %s initializer" %
                                     CTypesStructOrUnion._get_c_name())
                init = dict(zip(fnames, init))
            addr = ctypes.addressof(blob)
            for fname, value in init.items():
                BField, bitsize = name2fieldtype[fname]
                assert bitsize < 0, \
                       "not implemented: initializer with bit fields"
                offset = CTypesStructOrUnion._offsetof(fname)
                PTR = ctypes.POINTER(BField._ctype)
                p = ctypes.cast(addr + offset, PTR)
                BField._initialize(p.contents, value)
        is_union = CTypesStructOrUnion._kind == 'union'
        name2fieldtype = dict(zip(fnames, zip(btypes, bitfields)))
        #
        for fname, BField, bitsize in fields:
            if fname == '':
                raise NotImplementedError("nested anonymous structs/unions")
            if hasattr(CTypesStructOrUnion, fname):
                raise ValueError("the field name %r conflicts in "
                                 "the ctypes backend" % fname)
            if bitsize < 0:
                def getter(self, fname=fname, BField=BField,
                           offset=CTypesStructOrUnion._offsetof(fname),
                           PTR=ctypes.POINTER(BField._ctype)):
                    addr = ctypes.addressof(self._blob)
                    p = ctypes.cast(addr + offset, PTR)
                    return BField._from_ctypes(p.contents)
                def setter(self, value, fname=fname, BField=BField):
                    setattr(self._blob, fname, BField._to_ctypes(value))
                #
                if issubclass(BField, CTypesGenericArray):
                    setter = None
                    if BField._declared_length == 0:
                        def getter(self, fname=fname, BFieldPtr=BField._CTPtr,
                                   offset=CTypesStructOrUnion._offsetof(fname),
                                   PTR=ctypes.POINTER(BField._ctype)):
                            addr = ctypes.addressof(self._blob)
                            p = ctypes.cast(addr + offset, PTR)
                            return BFieldPtr._from_ctypes(p)
                #
            else:
                def getter(self, fname=fname, BField=BField):
                    return BField._from_ctypes(getattr(self._blob, fname))
                def setter(self, value, fname=fname, BField=BField):
                    # xxx obscure workaround
                    value = BField._to_ctypes(value)
                    oldvalue = getattr(self._blob, fname)
                    setattr(self._blob, fname, value)
                    if value != getattr(self._blob, fname):
                        setattr(self._blob, fname, oldvalue)
                        raise OverflowError("value too large for bitfield")
            setattr(CTypesStructOrUnion, fname, property(getter, setter))
        #
        CTypesPtr = self.ffi._get_cached_btype(model.PointerType(tp))
        for fname in fnames:
            if hasattr(CTypesPtr, fname):
                raise ValueError("the field name %r conflicts in "
                                 "the ctypes backend" % fname)
            def getter(self, fname=fname):
                return getattr(self[0], fname)
            def setter(self, value, fname=fname):
                setattr(self[0], fname, value)
            setattr(CTypesPtr, fname, property(getter, setter))

    def new_function_type(self, BArgs, BResult, has_varargs):
        nameargs = [BArg._get_c_name() for BArg in BArgs]
        if has_varargs:
            nameargs.append('...')
        nameargs = ', '.join(nameargs)
        #
        class CTypesFunctionPtr(CTypesGenericPtr):
            __slots__ = ['_own_callback', '_name']
            _ctype = ctypes.CFUNCTYPE(getattr(BResult, '_ctype', None),
                                      *[BArg._ctype for BArg in BArgs],
                                      use_errno=True)
            _reftypename = BResult._get_c_name('(* &)(%s)' % (nameargs,))

            def __init__(self, init, error=None):
                # create a callback to the Python callable init()
                import traceback
                assert not has_varargs, "varargs not supported for callbacks"
                if getattr(BResult, '_ctype', None) is not None:
                    error = BResult._from_ctypes(
                        BResult._create_ctype_obj(error))
                else:
                    error = None
                def callback(*args):
                    args2 = []
                    for arg, BArg in zip(args, BArgs):
                        args2.append(BArg._from_ctypes(arg))
                    try:
                        res2 = init(*args2)
                        res2 = BResult._to_ctypes(res2)
                    except:
                        traceback.print_exc()
                        res2 = error
                    if issubclass(BResult, CTypesGenericPtr):
                        if res2:
                            res2 = ctypes.cast(res2, ctypes.c_void_p).value
                                # .value: http://bugs.python.org/issue1574593
                        else:
                            res2 = None
                    #print repr(res2)
                    return res2
                if issubclass(BResult, CTypesGenericPtr):
                    # The only pointers callbacks can return are void*s:
                    # http://bugs.python.org/issue5710
                    callback_ctype = ctypes.CFUNCTYPE(
                        ctypes.c_void_p,
                        *[BArg._ctype for BArg in BArgs],
                        use_errno=True)
                else:
                    callback_ctype = CTypesFunctionPtr._ctype
                self._as_ctype_ptr = callback_ctype(callback)
                self._address = ctypes.cast(self._as_ctype_ptr,
                                            ctypes.c_void_p).value
                self._own_callback = init

            @staticmethod
            def _initialize(ctypes_ptr, value):
                if value:
                    raise NotImplementedError("ctypes backend: not supported: "
                                          "initializers for function pointers")

            def __repr__(self):
                c_name = getattr(self, '_name', None)
                if c_name:
                    i = self._reftypename.index('(* &)')
                    if self._reftypename[i-1] not in ' )*':
                        c_name = ' ' + c_name
                    c_name = self._reftypename.replace('(* &)', c_name)
                return CTypesData.__repr__(self, c_name)

            def _get_own_repr(self):
                if getattr(self, '_own_callback', None) is not None:
                    return 'calling %r' % (self._own_callback,)
                return super(CTypesFunctionPtr, self)._get_own_repr()

            def __call__(self, *args):
                if has_varargs:
                    assert len(args) >= len(BArgs)
                    extraargs = args[len(BArgs):]
                    args = args[:len(BArgs)]
                else:
                    assert len(args) == len(BArgs)
                ctypes_args = []
                for arg, BArg in zip(args, BArgs):
                    ctypes_args.append(BArg._arg_to_ctypes(arg))
                if has_varargs:
                    for i, arg in enumerate(extraargs):
                        if arg is None:
                            ctypes_args.append(ctypes.c_void_p(0))  # NULL
                            continue
                        if not isinstance(arg, CTypesData):
                            raise TypeError(
                                "argument %d passed in the variadic part "
                                "needs to be a cdata object (got %s)" %
                                (1 + len(BArgs) + i, type(arg).__name__))
                        ctypes_args.append(arg._arg_to_ctypes(arg))
                result = self._as_ctype_ptr(*ctypes_args)
                return BResult._from_ctypes(result)
        #
        CTypesFunctionPtr._fix_class()
        return CTypesFunctionPtr

    def new_enum_type(self, name, enumerators, enumvalues, CTypesInt):
        assert isinstance(name, str)
        reverse_mapping = dict(zip(reversed(enumvalues),
                                   reversed(enumerators)))
        #
        class CTypesEnum(CTypesInt):
            __slots__ = []
            _reftypename = '%s &' % name

            def _get_own_repr(self):
                value = self._value
                try:
                    return '%d: %s' % (value, reverse_mapping[value])
                except KeyError:
                    return str(value)

            def _to_string(self, maxlen):
                value = self._value
                try:
                    return reverse_mapping[value]
                except KeyError:
                    return str(value)
        #
        CTypesEnum._fix_class()
        return CTypesEnum

    def get_errno(self):
        return ctypes.get_errno()

    def set_errno(self, value):
        ctypes.set_errno(value)

    def string(self, b, maxlen=-1):
        return b._to_string(maxlen)

    def buffer(self, bptr, size=-1):
        raise NotImplementedError("buffer() with ctypes backend")

    def sizeof(self, cdata_or_BType):
        if isinstance(cdata_or_BType, CTypesData):
            return cdata_or_BType._get_size_of_instance()
        else:
            assert issubclass(cdata_or_BType, CTypesData)
            return cdata_or_BType._get_size()

    def alignof(self, BType):
        assert issubclass(BType, CTypesData)
        return BType._alignment()

    def newp(self, BType, source):
        if not issubclass(BType, CTypesData):
            raise TypeError
        return BType._newp(source)

    def cast(self, BType, source):
        return BType._cast_from(source)

    def callback(self, BType, source, error, onerror):
        assert onerror is None   # XXX not implemented
        return BType(source, error)

    _weakref_cache_ref = None

    def gcp(self, cdata, destructor, size=0):
        if self._weakref_cache_ref is None:
            import weakref
            class MyRef(weakref.ref):
                def __eq__(self, other):
                    myref = self()
                    return self is other or (
                        myref is not None and myref is other())
                def __ne__(self, other):
                    return not (self == other)
                def __hash__(self):
                    try:
                        return self._hash
                    except AttributeError:
                        self._hash = hash(self())
                        return self._hash
            self._weakref_cache_ref = {}, MyRef
        weak_cache, MyRef = self._weakref_cache_ref

        if destructor is None:
            try:
                del weak_cache[MyRef(cdata)]
            except KeyError:
                raise TypeError("Can remove destructor only on a object "
                                "previously returned by ffi.gc()")
            return None

        def remove(k):
            cdata, destructor = weak_cache.pop(k, (None, None))
            if destructor is not None:
                destructor(cdata)

        new_cdata = self.cast(self.typeof(cdata), cdata)
        assert new_cdata is not cdata
        weak_cache[MyRef(new_cdata, remove)] = (cdata, destructor)
        return new_cdata

    typeof = type

    def getcname(self, BType, replace_with):
        return BType._get_c_name(replace_with)

    def typeoffsetof(self, BType, fieldname, num=0):
        if isinstance(fieldname, str):
            if num == 0 and issubclass(BType, CTypesGenericPtr):
                BType = BType._BItem
            if not issubclass(BType, CTypesBaseStructOrUnion):
                raise TypeError("expected a struct or union ctype")
            BField = BType._bfield_types[fieldname]
            if BField is Ellipsis:
                raise TypeError("not supported for bitfields")
            return (BField, BType._offsetof(fieldname))
        elif isinstance(fieldname, (int, long)):
            if issubclass(BType, CTypesGenericArray):
                BType = BType._CTPtr
            if not issubclass(BType, CTypesGenericPtr):
                raise TypeError("expected an array or ptr ctype")
            BItem = BType._BItem
            offset = BItem._get_size() * fieldname
            if offset > sys.maxsize:
                raise OverflowError
            return (BItem, offset)
        else:
            raise TypeError(type(fieldname))

    def rawaddressof(self, BTypePtr, cdata, offset=None):
        if isinstance(cdata, CTypesBaseStructOrUnion):
            ptr = ctypes.pointer(type(cdata)._to_ctypes(cdata))
        elif isinstance(cdata, CTypesGenericPtr):
            if offset is None or not issubclass(type(cdata)._BItem,
                                                CTypesBaseStructOrUnion):
                raise TypeError("unexpected cdata type")
            ptr = type(cdata)._to_ctypes(cdata)
        elif isinstance(cdata, CTypesGenericArray):
            ptr = type(cdata)._to_ctypes(cdata)
        else:
            raise TypeError("expected a <cdata 'struct-or-union'>")
        if offset:
            ptr = ctypes.cast(
                ctypes.c_void_p(
                    ctypes.cast(ptr, ctypes.c_void_p).value + offset),
                type(ptr))
        return BTypePtr._from_ctypes(ptr)


class CTypesLibrary(object):

    def __init__(self, backend, cdll):
        self.backend = backend
        self.cdll = cdll

    def load_function(self, BType, name):
        c_func = getattr(self.cdll, name)
        funcobj = BType._from_ctypes(c_func)
        funcobj._name = name
        return funcobj

    def read_variable(self, BType, name):
        try:
            ctypes_obj = BType._ctype.in_dll(self.cdll, name)
        except AttributeError as e:
            raise NotImplementedError(e)
        return BType._from_ctypes(ctypes_obj)

    def write_variable(self, BType, name, value):
        new_ctypes_obj = BType._to_ctypes(value)
        ctypes_obj = BType._ctype.in_dll(self.cdll, name)
        ctypes.memmove(ctypes.addressof(ctypes_obj),
                       ctypes.addressof(new_ctypes_obj),
                       ctypes.sizeof(BType._ctype))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\defmatrix.py ===
__all__ = ['matrix', 'bmat', 'asmatrix']

import ast
import sys
import warnings

import numpy._core.numeric as N
from numpy._core.numeric import concatenate, isscalar
from numpy._utils import set_module

# While not in __all__, matrix_power used to be defined here, so we import
# it for backward compatibility.
from numpy.linalg import matrix_power


def _convert_from_string(data):
    for char in '[]':
        data = data.replace(char, '')

    rows = data.split(';')
    newdata = []
    for count, row in enumerate(rows):
        trow = row.split(',')
        newrow = []
        for col in trow:
            temp = col.split()
            newrow.extend(map(ast.literal_eval, temp))
        if count == 0:
            Ncols = len(newrow)
        elif len(newrow) != Ncols:
            raise ValueError("Rows not the same size.")
        newdata.append(newrow)
    return newdata


@set_module('numpy')
def asmatrix(data, dtype=None):
    """
    Interpret the input as a matrix.

    Unlike `matrix`, `asmatrix` does not make a copy if the input is already
    a matrix or an ndarray.  Equivalent to ``matrix(data, copy=False)``.

    Parameters
    ----------
    data : array_like
        Input data.
    dtype : data-type
       Data-type of the output matrix.

    Returns
    -------
    mat : matrix
        `data` interpreted as a matrix.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 4]])

    >>> m = np.asmatrix(x)

    >>> x[0,0] = 5

    >>> m
    matrix([[5, 2],
            [3, 4]])

    """
    return matrix(data, dtype=dtype, copy=False)


@set_module('numpy')
class matrix(N.ndarray):
    """
    matrix(data, dtype=None, copy=True)

    Returns a matrix from an array-like object, or from a string of data.

    A matrix is a specialized 2-D array that retains its 2-D nature
    through operations.  It has certain special operators, such as ``*``
    (matrix multiplication) and ``**`` (matrix power).

    .. note:: It is no longer recommended to use this class, even for linear
              algebra. Instead use regular arrays. The class may be removed
              in the future.

    Parameters
    ----------
    data : array_like or string
       If `data` is a string, it is interpreted as a matrix with commas
       or spaces separating columns, and semicolons separating rows.
    dtype : data-type
       Data-type of the output matrix.
    copy : bool
       If `data` is already an `ndarray`, then this flag determines
       whether the data is copied (the default), or whether a view is
       constructed.

    See Also
    --------
    array

    Examples
    --------
    >>> import numpy as np
    >>> a = np.matrix('1 2; 3 4')
    >>> a
    matrix([[1, 2],
            [3, 4]])

    >>> np.matrix([[1, 2], [3, 4]])
    matrix([[1, 2],
            [3, 4]])

    """
    __array_priority__ = 10.0

    def __new__(subtype, data, dtype=None, copy=True):
        warnings.warn('the matrix subclass is not the recommended way to '
                      'represent matrices or deal with linear algebra (see '
                      'https://docs.scipy.org/doc/numpy/user/'
                      'numpy-for-matlab-users.html). '
                      'Please adjust your code to use regular ndarray.',
                      PendingDeprecationWarning, stacklevel=2)
        if isinstance(data, matrix):
            dtype2 = data.dtype
            if (dtype is None):
                dtype = dtype2
            if (dtype2 == dtype) and (not copy):
                return data
            return data.astype(dtype)

        if isinstance(data, N.ndarray):
            if dtype is None:
                intype = data.dtype
            else:
                intype = N.dtype(dtype)
            new = data.view(subtype)
            if intype != data.dtype:
                return new.astype(intype)
            if copy:
                return new.copy()
            else:
                return new

        if isinstance(data, str):
            data = _convert_from_string(data)

        # now convert data to an array
        copy = None if not copy else True
        arr = N.array(data, dtype=dtype, copy=copy)
        ndim = arr.ndim
        shape = arr.shape
        if (ndim > 2):
            raise ValueError("matrix must be 2-dimensional")
        elif ndim == 0:
            shape = (1, 1)
        elif ndim == 1:
            shape = (1, shape[0])

        order = 'C'
        if (ndim == 2) and arr.flags.fortran:
            order = 'F'

        if not (order or arr.flags.contiguous):
            arr = arr.copy()

        ret = N.ndarray.__new__(subtype, shape, arr.dtype,
                                buffer=arr,
                                order=order)
        return ret

    def __array_finalize__(self, obj):
        self._getitem = False
        if (isinstance(obj, matrix) and obj._getitem):
            return
        ndim = self.ndim
        if (ndim == 2):
            return
        if (ndim > 2):
            newshape = tuple(x for x in self.shape if x > 1)
            ndim = len(newshape)
            if ndim == 2:
                self.shape = newshape
                return
            elif (ndim > 2):
                raise ValueError("shape too large to be a matrix.")
        else:
            newshape = self.shape
        if ndim == 0:
            self.shape = (1, 1)
        elif ndim == 1:
            self.shape = (1, newshape[0])
        return

    def __getitem__(self, index):
        self._getitem = True

        try:
            out = N.ndarray.__getitem__(self, index)
        finally:
            self._getitem = False

        if not isinstance(out, N.ndarray):
            return out

        if out.ndim == 0:
            return out[()]
        if out.ndim == 1:
            sh = out.shape[0]
            # Determine when we should have a column array
            try:
                n = len(index)
            except Exception:
                n = 0
            if n > 1 and isscalar(index[1]):
                out.shape = (sh, 1)
            else:
                out.shape = (1, sh)
        return out

    def __mul__(self, other):
        if isinstance(other, (N.ndarray, list, tuple)):
            # This promotes 1-D vectors to row vectors
            return N.dot(self, asmatrix(other))
        if isscalar(other) or not hasattr(other, '__rmul__'):
            return N.dot(self, other)
        return NotImplemented

    def __rmul__(self, other):
        return N.dot(other, self)

    def __imul__(self, other):
        self[:] = self * other
        return self

    def __pow__(self, other):
        return matrix_power(self, other)

    def __ipow__(self, other):
        self[:] = self ** other
        return self

    def __rpow__(self, other):
        return NotImplemented

    def _align(self, axis):
        """A convenience function for operations that need to preserve axis
        orientation.
        """
        if axis is None:
            return self[0, 0]
        elif axis == 0:
            return self
        elif axis == 1:
            return self.transpose()
        else:
            raise ValueError("unsupported axis")

    def _collapse(self, axis):
        """A convenience function for operations that want to collapse
        to a scalar like _align, but are using keepdims=True
        """
        if axis is None:
            return self[0, 0]
        else:
            return self

    # Necessary because base-class tolist expects dimension
    #  reduction by x[0]
    def tolist(self):
        """
        Return the matrix as a (possibly nested) list.

        See `ndarray.tolist` for full documentation.

        See Also
        --------
        ndarray.tolist

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.tolist()
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

        """
        return self.__array__().tolist()

    # To preserve orientation of result...
    def sum(self, axis=None, dtype=None, out=None):
        """
        Returns the sum of the matrix elements, along the given axis.

        Refer to `numpy.sum` for full documentation.

        See Also
        --------
        numpy.sum

        Notes
        -----
        This is the same as `ndarray.sum`, except that where an `ndarray` would
        be returned, a `matrix` object is returned instead.

        Examples
        --------
        >>> x = np.matrix([[1, 2], [4, 3]])
        >>> x.sum()
        10
        >>> x.sum(axis=1)
        matrix([[3],
                [7]])
        >>> x.sum(axis=1, dtype='float')
        matrix([[3.],
                [7.]])
        >>> out = np.zeros((2, 1), dtype='float')
        >>> x.sum(axis=1, dtype='float', out=np.asmatrix(out))
        matrix([[3.],
                [7.]])

        """
        return N.ndarray.sum(self, axis, dtype, out, keepdims=True)._collapse(axis)

    # To update docstring from array to matrix...
    def squeeze(self, axis=None):
        """
        Return a possibly reshaped matrix.

        Refer to `numpy.squeeze` for more documentation.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Selects a subset of the axes of length one in the shape.
            If an axis is selected with shape entry greater than one,
            an error is raised.

        Returns
        -------
        squeezed : matrix
            The matrix, but as a (1, N) matrix if it had shape (N, 1).

        See Also
        --------
        numpy.squeeze : related function

        Notes
        -----
        If `m` has a single column then that column is returned
        as the single row of a matrix.  Otherwise `m` is returned.
        The returned matrix is always either `m` itself or a view into `m`.
        Supplying an axis keyword argument will not affect the returned matrix
        but it may cause an error to be raised.

        Examples
        --------
        >>> c = np.matrix([[1], [2]])
        >>> c
        matrix([[1],
                [2]])
        >>> c.squeeze()
        matrix([[1, 2]])
        >>> r = c.T
        >>> r
        matrix([[1, 2]])
        >>> r.squeeze()
        matrix([[1, 2]])
        >>> m = np.matrix([[1, 2], [3, 4]])
        >>> m.squeeze()
        matrix([[1, 2],
                [3, 4]])

        """
        return N.ndarray.squeeze(self, axis=axis)

    # To update docstring from array to matrix...
    def flatten(self, order='C'):
        """
        Return a flattened copy of the matrix.

        All `N` elements of the matrix are placed into a single row.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            'C' means to flatten in row-major (C-style) order. 'F' means to
            flatten in column-major (Fortran-style) order. 'A' means to
            flatten in column-major order if `m` is Fortran *contiguous* in
            memory, row-major order otherwise. 'K' means to flatten `m` in
            the order the elements occur in memory. The default is 'C'.

        Returns
        -------
        y : matrix
            A copy of the matrix, flattened to a `(1, N)` matrix where `N`
            is the number of elements in the original matrix.

        See Also
        --------
        ravel : Return a flattened array.
        flat : A 1-D flat iterator over the matrix.

        Examples
        --------
        >>> m = np.matrix([[1,2], [3,4]])
        >>> m.flatten()
        matrix([[1, 2, 3, 4]])
        >>> m.flatten('F')
        matrix([[1, 3, 2, 4]])

        """
        return N.ndarray.flatten(self, order=order)

    def mean(self, axis=None, dtype=None, out=None):
        """
        Returns the average of the matrix elements along the given axis.

        Refer to `numpy.mean` for full documentation.

        See Also
        --------
        numpy.mean

        Notes
        -----
        Same as `ndarray.mean` except that, where that returns an `ndarray`,
        this returns a `matrix` object.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3, 4)))
        >>> x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.mean()
        5.5
        >>> x.mean(0)
        matrix([[4., 5., 6., 7.]])
        >>> x.mean(1)
        matrix([[ 1.5],
                [ 5.5],
                [ 9.5]])

        """
        return N.ndarray.mean(self, axis, dtype, out, keepdims=True)._collapse(axis)

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        """
        Return the standard deviation of the array elements along the given axis.

        Refer to `numpy.std` for full documentation.

        See Also
        --------
        numpy.std

        Notes
        -----
        This is the same as `ndarray.std`, except that where an `ndarray` would
        be returned, a `matrix` object is returned instead.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3, 4)))
        >>> x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.std()
        3.4520525295346629 # may vary
        >>> x.std(0)
        matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]]) # may vary
        >>> x.std(1)
        matrix([[ 1.11803399],
                [ 1.11803399],
                [ 1.11803399]])

        """
        return N.ndarray.std(self, axis, dtype, out, ddof,
                             keepdims=True)._collapse(axis)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        """
        Returns the variance of the matrix elements, along the given axis.

        Refer to `numpy.var` for full documentation.

        See Also
        --------
        numpy.var

        Notes
        -----
        This is the same as `ndarray.var`, except that where an `ndarray` would
        be returned, a `matrix` object is returned instead.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3, 4)))
        >>> x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.var()
        11.916666666666666
        >>> x.var(0)
        matrix([[ 10.66666667,  10.66666667,  10.66666667,  10.66666667]]) # may vary
        >>> x.var(1)
        matrix([[1.25],
                [1.25],
                [1.25]])

        """
        return N.ndarray.var(self, axis, dtype, out, ddof,
                             keepdims=True)._collapse(axis)

    def prod(self, axis=None, dtype=None, out=None):
        """
        Return the product of the array elements over the given axis.

        Refer to `prod` for full documentation.

        See Also
        --------
        prod, ndarray.prod

        Notes
        -----
        Same as `ndarray.prod`, except, where that returns an `ndarray`, this
        returns a `matrix` object instead.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.prod()
        0
        >>> x.prod(0)
        matrix([[  0,  45, 120, 231]])
        >>> x.prod(1)
        matrix([[   0],
                [ 840],
                [7920]])

        """
        return N.ndarray.prod(self, axis, dtype, out, keepdims=True)._collapse(axis)

    def any(self, axis=None, out=None):
        """
        Test whether any array element along a given axis evaluates to True.

        Refer to `numpy.any` for full documentation.

        Parameters
        ----------
        axis : int, optional
            Axis along which logical OR is performed
        out : ndarray, optional
            Output to existing array instead of creating new one, must have
            same shape as expected output

        Returns
        -------
            any : bool, ndarray
                Returns a single bool if `axis` is ``None``; otherwise,
                returns `ndarray`

        """
        return N.ndarray.any(self, axis, out, keepdims=True)._collapse(axis)

    def all(self, axis=None, out=None):
        """
        Test whether all matrix elements along a given axis evaluate to True.

        Parameters
        ----------
        See `numpy.all` for complete descriptions

        See Also
        --------
        numpy.all

        Notes
        -----
        This is the same as `ndarray.all`, but it returns a `matrix` object.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> y = x[0]; y
        matrix([[0, 1, 2, 3]])
        >>> (x == y)
        matrix([[ True,  True,  True,  True],
                [False, False, False, False],
                [False, False, False, False]])
        >>> (x == y).all()
        False
        >>> (x == y).all(0)
        matrix([[False, False, False, False]])
        >>> (x == y).all(1)
        matrix([[ True],
                [False],
                [False]])

        """
        return N.ndarray.all(self, axis, out, keepdims=True)._collapse(axis)

    def max(self, axis=None, out=None):
        """
        Return the maximum value along an axis.

        Parameters
        ----------
        See `amax` for complete descriptions

        See Also
        --------
        amax, ndarray.max

        Notes
        -----
        This is the same as `ndarray.max`, but returns a `matrix` object
        where `ndarray.max` would return an ndarray.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.max()
        11
        >>> x.max(0)
        matrix([[ 8,  9, 10, 11]])
        >>> x.max(1)
        matrix([[ 3],
                [ 7],
                [11]])

        """
        return N.ndarray.max(self, axis, out, keepdims=True)._collapse(axis)

    def argmax(self, axis=None, out=None):
        """
        Indexes of the maximum values along an axis.

        Return the indexes of the first occurrences of the maximum values
        along the specified axis.  If axis is None, the index is for the
        flattened matrix.

        Parameters
        ----------
        See `numpy.argmax` for complete descriptions

        See Also
        --------
        numpy.argmax

        Notes
        -----
        This is the same as `ndarray.argmax`, but returns a `matrix` object
        where `ndarray.argmax` would return an `ndarray`.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.argmax()
        11
        >>> x.argmax(0)
        matrix([[2, 2, 2, 2]])
        >>> x.argmax(1)
        matrix([[3],
                [3],
                [3]])

        """
        return N.ndarray.argmax(self, axis, out)._align(axis)

    def min(self, axis=None, out=None):
        """
        Return the minimum value along an axis.

        Parameters
        ----------
        See `amin` for complete descriptions.

        See Also
        --------
        amin, ndarray.min

        Notes
        -----
        This is the same as `ndarray.min`, but returns a `matrix` object
        where `ndarray.min` would return an ndarray.

        Examples
        --------
        >>> x = -np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[  0,  -1,  -2,  -3],
                [ -4,  -5,  -6,  -7],
                [ -8,  -9, -10, -11]])
        >>> x.min()
        -11
        >>> x.min(0)
        matrix([[ -8,  -9, -10, -11]])
        >>> x.min(1)
        matrix([[ -3],
                [ -7],
                [-11]])

        """
        return N.ndarray.min(self, axis, out, keepdims=True)._collapse(axis)

    def argmin(self, axis=None, out=None):
        """
        Indexes of the minimum values along an axis.

        Return the indexes of the first occurrences of the minimum values
        along the specified axis.  If axis is None, the index is for the
        flattened matrix.

        Parameters
        ----------
        See `numpy.argmin` for complete descriptions.

        See Also
        --------
        numpy.argmin

        Notes
        -----
        This is the same as `ndarray.argmin`, but returns a `matrix` object
        where `ndarray.argmin` would return an `ndarray`.

        Examples
        --------
        >>> x = -np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[  0,  -1,  -2,  -3],
                [ -4,  -5,  -6,  -7],
                [ -8,  -9, -10, -11]])
        >>> x.argmin()
        11
        >>> x.argmin(0)
        matrix([[2, 2, 2, 2]])
        >>> x.argmin(1)
        matrix([[3],
                [3],
                [3]])

        """
        return N.ndarray.argmin(self, axis, out)._align(axis)

    def ptp(self, axis=None, out=None):
        """
        Peak-to-peak (maximum - minimum) value along the given axis.

        Refer to `numpy.ptp` for full documentation.

        See Also
        --------
        numpy.ptp

        Notes
        -----
        Same as `ndarray.ptp`, except, where that would return an `ndarray` object,
        this returns a `matrix` object.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.ptp()
        11
        >>> x.ptp(0)
        matrix([[8, 8, 8, 8]])
        >>> x.ptp(1)
        matrix([[3],
                [3],
                [3]])

        """
        return N.ptp(self, axis, out)._align(axis)

    @property
    def I(self):  # noqa: E743
        """
        Returns the (multiplicative) inverse of invertible `self`.

        Parameters
        ----------
        None

        Returns
        -------
        ret : matrix object
            If `self` is non-singular, `ret` is such that ``ret * self`` ==
            ``self * ret`` == ``np.matrix(np.eye(self[0,:].size))`` all return
            ``True``.

        Raises
        ------
        numpy.linalg.LinAlgError: Singular matrix
            If `self` is singular.

        See Also
        --------
        linalg.inv

        Examples
        --------
        >>> m = np.matrix('[1, 2; 3, 4]'); m
        matrix([[1, 2],
                [3, 4]])
        >>> m.getI()
        matrix([[-2. ,  1. ],
                [ 1.5, -0.5]])
        >>> m.getI() * m
        matrix([[ 1.,  0.], # may vary
                [ 0.,  1.]])

        """
        M, N = self.shape
        if M == N:
            from numpy.linalg import inv as func
        else:
            from numpy.linalg import pinv as func
        return asmatrix(func(self))

    @property
    def A(self):
        """
        Return `self` as an `ndarray` object.

        Equivalent to ``np.asarray(self)``.

        Parameters
        ----------
        None

        Returns
        -------
        ret : ndarray
            `self` as an `ndarray`

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.getA()
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])

        """
        return self.__array__()

    @property
    def A1(self):
        """
        Return `self` as a flattened `ndarray`.

        Equivalent to ``np.asarray(x).ravel()``

        Parameters
        ----------
        None

        Returns
        -------
        ret : ndarray
            `self`, 1-D, as an `ndarray`

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.getA1()
        array([ 0,  1,  2, ...,  9, 10, 11])


        """
        return self.__array__().ravel()

    def ravel(self, order='C'):
        """
        Return a flattened matrix.

        Refer to `numpy.ravel` for more documentation.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The elements of `m` are read using this index order. 'C' means to
            index the elements in C-like order, with the last axis index
            changing fastest, back to the first axis index changing slowest.
            'F' means to index the elements in Fortran-like index order, with
            the first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of the
            memory layout of the underlying array, and only refer to the order
            of axis indexing.  'A' means to read the elements in Fortran-like
            index order if `m` is Fortran *contiguous* in memory, C-like order
            otherwise.  'K' means to read the elements in the order they occur
            in memory, except for reversing the data when strides are negative.
            By default, 'C' index order is used.

        Returns
        -------
        ret : matrix
            Return the matrix flattened to shape `(1, N)` where `N`
            is the number of elements in the original matrix.
            A copy is made only if necessary.

        See Also
        --------
        matrix.flatten : returns a similar output matrix but always a copy
        matrix.flat : a flat iterator on the array.
        numpy.ravel : related function which returns an ndarray

        """
        return N.ndarray.ravel(self, order=order)

    @property
    def T(self):
        """
        Returns the transpose of the matrix.

        Does *not* conjugate!  For the complex conjugate transpose, use ``.H``.

        Parameters
        ----------
        None

        Returns
        -------
        ret : matrix object
            The (non-conjugated) transpose of the matrix.

        See Also
        --------
        transpose, getH

        Examples
        --------
        >>> m = np.matrix('[1, 2; 3, 4]')
        >>> m
        matrix([[1, 2],
                [3, 4]])
        >>> m.getT()
        matrix([[1, 3],
                [2, 4]])

        """
        return self.transpose()

    @property
    def H(self):
        """
        Returns the (complex) conjugate transpose of `self`.

        Equivalent to ``np.transpose(self)`` if `self` is real-valued.

        Parameters
        ----------
        None

        Returns
        -------
        ret : matrix object
            complex conjugate transpose of `self`

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4)))
        >>> z = x - 1j*x; z
        matrix([[  0. +0.j,   1. -1.j,   2. -2.j,   3. -3.j],
                [  4. -4.j,   5. -5.j,   6. -6.j,   7. -7.j],
                [  8. -8.j,   9. -9.j,  10.-10.j,  11.-11.j]])
        >>> z.getH()
        matrix([[ 0. -0.j,  4. +4.j,  8. +8.j],
                [ 1. +1.j,  5. +5.j,  9. +9.j],
                [ 2. +2.j,  6. +6.j, 10.+10.j],
                [ 3. +3.j,  7. +7.j, 11.+11.j]])

        """
        if issubclass(self.dtype.type, N.complexfloating):
            return self.transpose().conjugate()
        else:
            return self.transpose()

    # kept for compatibility
    getT = T.fget
    getA = A.fget
    getA1 = A1.fget
    getH = H.fget
    getI = I.fget

def _from_string(str, gdict, ldict):
    rows = str.split(';')
    rowtup = []
    for row in rows:
        trow = row.split(',')
        newrow = []
        for x in trow:
            newrow.extend(x.split())
        trow = newrow
        coltup = []
        for col in trow:
            col = col.strip()
            try:
                thismat = ldict[col]
            except KeyError:
                try:
                    thismat = gdict[col]
                except KeyError as e:
                    raise NameError(f"name {col!r} is not defined") from None

            coltup.append(thismat)
        rowtup.append(concatenate(coltup, axis=-1))
    return concatenate(rowtup, axis=0)


@set_module('numpy')
def bmat(obj, ldict=None, gdict=None):
    """
    Build a matrix object from a string, nested sequence, or array.

    Parameters
    ----------
    obj : str or array_like
        Input data. If a string, variables in the current scope may be
        referenced by name.
    ldict : dict, optional
        A dictionary that replaces local operands in current frame.
        Ignored if `obj` is not a string or `gdict` is None.
    gdict : dict, optional
        A dictionary that replaces global operands in current frame.
        Ignored if `obj` is not a string.

    Returns
    -------
    out : matrix
        Returns a matrix object, which is a specialized 2-D array.

    See Also
    --------
    block :
        A generalization of this function for N-d arrays, that returns normal
        ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.asmatrix('1 1; 1 1')
    >>> B = np.asmatrix('2 2; 2 2')
    >>> C = np.asmatrix('3 4; 5 6')
    >>> D = np.asmatrix('7 8; 9 0')

    All the following expressions construct the same block matrix:

    >>> np.bmat([[A, B], [C, D]])
    matrix([[1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 4, 7, 8],
            [5, 6, 9, 0]])
    >>> np.bmat(np.r_[np.c_[A, B], np.c_[C, D]])
    matrix([[1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 4, 7, 8],
            [5, 6, 9, 0]])
    >>> np.bmat('A,B; C,D')
    matrix([[1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 4, 7, 8],
            [5, 6, 9, 0]])

    """
    if isinstance(obj, str):
        if gdict is None:
            # get previous frame
            frame = sys._getframe().f_back
            glob_dict = frame.f_globals
            loc_dict = frame.f_locals
        else:
            glob_dict = gdict
            loc_dict = ldict

        return matrix(_from_string(obj, glob_dict, loc_dict))

    if isinstance(obj, (tuple, list)):
        # [[A,B],[C,D]]
        arr_rows = []
        for row in obj:
            if isinstance(row, N.ndarray):  # not 2-d
                return matrix(concatenate(obj, axis=-1))
            else:
                arr_rows.append(concatenate(row, axis=-1))
        return matrix(concatenate(arr_rows, axis=0))
    if isinstance(obj, N.ndarray):
        return matrix(obj)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_testing_raisesgroup.py ===
from __future__ import annotations

import re
import sys
from types import TracebackType
from typing import TYPE_CHECKING

import pytest

import trio
from trio.testing import Matcher, RaisesGroup
from trio.testing._raises_group import repr_callable

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup

if TYPE_CHECKING:
    from _pytest.python_api import RaisesContext


def wrap_escape(s: str) -> str:
    return "^" + re.escape(s) + "$"


def fails_raises_group(
    msg: str, add_prefix: bool = True
) -> RaisesContext[AssertionError]:
    assert (
        msg[-1] != "\n"
    ), "developer error, expected string should not end with newline"
    prefix = "Raised exception group did not match: " if add_prefix else ""
    return pytest.raises(AssertionError, match=wrap_escape(prefix + msg))


def test_raises_group() -> None:
    with pytest.raises(
        ValueError,
        match=wrap_escape(
            f'Invalid argument "{TypeError()!r}" must be exception type, Matcher, or RaisesGroup.',
        ),
    ):
        RaisesGroup(TypeError())  # type: ignore[call-overload]
    with RaisesGroup(ValueError):
        raise ExceptionGroup("foo", (ValueError(),))

    with (
        fails_raises_group("'SyntaxError' is not of type 'ValueError'"),
        RaisesGroup(ValueError),
    ):
        raise ExceptionGroup("foo", (SyntaxError(),))

    # multiple exceptions
    with RaisesGroup(ValueError, SyntaxError):
        raise ExceptionGroup("foo", (ValueError(), SyntaxError()))

    # order doesn't matter
    with RaisesGroup(SyntaxError, ValueError):
        raise ExceptionGroup("foo", (ValueError(), SyntaxError()))

    # nested exceptions
    with RaisesGroup(RaisesGroup(ValueError)):
        raise ExceptionGroup("foo", (ExceptionGroup("bar", (ValueError(),)),))

    with RaisesGroup(
        SyntaxError,
        RaisesGroup(ValueError),
        RaisesGroup(RuntimeError),
    ):
        raise ExceptionGroup(
            "foo",
            (
                SyntaxError(),
                ExceptionGroup("bar", (ValueError(),)),
                ExceptionGroup("", (RuntimeError(),)),
            ),
        )


def test_incorrect_number_exceptions() -> None:
    # We previously gave an error saying the number of exceptions was wrong,
    # but we now instead indicate excess/missing exceptions
    with (
        fails_raises_group(
            "1 matched exception. Unexpected exception(s): [RuntimeError()]"
        ),
        RaisesGroup(ValueError),
    ):
        raise ExceptionGroup("", (RuntimeError(), ValueError()))

    # will error if there's missing exceptions
    with (
        fails_raises_group(
            "1 matched exception. Too few exceptions raised, found no match for: ['SyntaxError']"
        ),
        RaisesGroup(ValueError, SyntaxError),
    ):
        raise ExceptionGroup("", (ValueError(),))

    with (
        fails_raises_group(
            "\n"
            "1 matched exception. \n"
            "Too few exceptions raised!\n"
            "The following expected exceptions did not find a match:\n"
            "  'ValueError'\n"
            "    It matches ValueError() which was paired with 'ValueError'"
        ),
        RaisesGroup(ValueError, ValueError),
    ):
        raise ExceptionGroup("", (ValueError(),))

    with (
        fails_raises_group(
            "\n"
            "1 matched exception. \n"
            "Unexpected exception(s)!\n"
            "The following raised exceptions did not find a match\n"
            "  ValueError():\n"
            "    It matches 'ValueError' which was paired with ValueError()"
        ),
        RaisesGroup(ValueError),
    ):
        raise ExceptionGroup("", (ValueError(), ValueError()))

    with (
        fails_raises_group(
            "\n"
            "1 matched exception. \n"
            "The following expected exceptions did not find a match:\n"
            "  'ValueError'\n"
            "    It matches ValueError() which was paired with 'ValueError'\n"
            "The following raised exceptions did not find a match\n"
            "  SyntaxError():\n"
            "    'SyntaxError' is not of type 'ValueError'"
        ),
        RaisesGroup(ValueError, ValueError),
    ):
        raise ExceptionGroup("", [ValueError(), SyntaxError()])


def test_flatten_subgroups() -> None:
    # loose semantics, as with expect*
    with RaisesGroup(ValueError, flatten_subgroups=True):
        raise ExceptionGroup("", (ExceptionGroup("", (ValueError(),)),))

    with RaisesGroup(ValueError, TypeError, flatten_subgroups=True):
        raise ExceptionGroup("", (ExceptionGroup("", (ValueError(), TypeError())),))
    with RaisesGroup(ValueError, TypeError, flatten_subgroups=True):
        raise ExceptionGroup("", [ExceptionGroup("", [ValueError()]), TypeError()])

    # mixed loose is possible if you want it to be at least N deep
    with RaisesGroup(RaisesGroup(ValueError, flatten_subgroups=True)):
        raise ExceptionGroup("", (ExceptionGroup("", (ValueError(),)),))
    with RaisesGroup(RaisesGroup(ValueError, flatten_subgroups=True)):
        raise ExceptionGroup(
            "",
            (ExceptionGroup("", (ExceptionGroup("", (ValueError(),)),)),),
        )

    # but not the other way around
    with pytest.raises(
        ValueError,
        match=r"^You cannot specify a nested structure inside a RaisesGroup with",
    ):
        RaisesGroup(RaisesGroup(ValueError), flatten_subgroups=True)  # type: ignore[call-overload]

    # flatten_subgroups is not sufficient to catch fully unwrapped
    with (
        fails_raises_group(
            "'ValueError' is not an exception group, but would match with `allow_unwrapped=True`"
        ),
        RaisesGroup(ValueError, flatten_subgroups=True),
    ):
        raise ValueError
    with (
        fails_raises_group(
            "RaisesGroup(ValueError, flatten_subgroups=True): 'ValueError' is not an exception group, but would match with `allow_unwrapped=True`"
        ),
        RaisesGroup(RaisesGroup(ValueError, flatten_subgroups=True)),
    ):
        raise ExceptionGroup("", (ValueError(),))

    # helpful suggestion if flatten_subgroups would make it pass
    with (
        fails_raises_group(
            "Raised exception group did not match: \n"
            "The following expected exceptions did not find a match:\n"
            "  'ValueError'\n"
            "  'TypeError'\n"
            "The following raised exceptions did not find a match\n"
            "  ExceptionGroup('', [ValueError(), TypeError()]):\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "    Unexpected nested 'ExceptionGroup', expected 'TypeError'\n"
            "Did you mean to use `flatten_subgroups=True`?",
            add_prefix=False,
        ),
        RaisesGroup(ValueError, TypeError),
    ):
        raise ExceptionGroup("", [ExceptionGroup("", [ValueError(), TypeError()])])
    # but doesn't consider check (otherwise we'd break typing guarantees)
    with (
        fails_raises_group(
            "Raised exception group did not match: \n"
            "The following expected exceptions did not find a match:\n"
            "  'ValueError'\n"
            "  'TypeError'\n"
            "The following raised exceptions did not find a match\n"
            "  ExceptionGroup('', [ValueError(), TypeError()]):\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "    Unexpected nested 'ExceptionGroup', expected 'TypeError'\n"
            "Did you mean to use `flatten_subgroups=True`?",
            add_prefix=False,
        ),
        RaisesGroup(
            ValueError,
            TypeError,
            check=lambda eg: len(eg.exceptions) == 1,
        ),
    ):
        raise ExceptionGroup("", [ExceptionGroup("", [ValueError(), TypeError()])])
    # correct number of exceptions, and flatten_subgroups would make it pass
    # This now doesn't print a repr of the caught exception at all, but that can be found in the traceback
    with (
        fails_raises_group(
            "Raised exception group did not match: Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "  Did you mean to use `flatten_subgroups=True`?",
            add_prefix=False,
        ),
        RaisesGroup(ValueError),
    ):
        raise ExceptionGroup("", [ExceptionGroup("", [ValueError()])])
    # correct number of exceptions, but flatten_subgroups wouldn't help, so we don't suggest it
    with (
        fails_raises_group("Unexpected nested 'ExceptionGroup', expected 'ValueError'"),
        RaisesGroup(ValueError),
    ):
        raise ExceptionGroup("", [ExceptionGroup("", [TypeError()])])

    # flatten_subgroups can be suggested if nested. This will implicitly ask the user to
    # do `RaisesGroup(RaisesGroup(ValueError, flatten_subgroups=True))` which is unlikely
    # to be what they actually want - but I don't think it's worth trying to special-case
    with (
        fails_raises_group(
            "RaisesGroup(ValueError): Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "  Did you mean to use `flatten_subgroups=True`?",
        ),
        RaisesGroup(RaisesGroup(ValueError)),
    ):
        raise ExceptionGroup(
            "",
            [ExceptionGroup("", [ExceptionGroup("", [ValueError()])])],
        )

    # Don't mention "unexpected nested" if expecting an ExceptionGroup.
    # Although it should perhaps be an error to specify `RaisesGroup(ExceptionGroup)` in
    # favor of doing `RaisesGroup(RaisesGroup(...))`.
    with (
        fails_raises_group("'BaseExceptionGroup' is not of type 'ExceptionGroup'"),
        RaisesGroup(ExceptionGroup),
    ):
        raise BaseExceptionGroup("", [BaseExceptionGroup("", [KeyboardInterrupt()])])


def test_catch_unwrapped_exceptions() -> None:
    # Catches lone exceptions with strict=False
    # just as except* would
    with RaisesGroup(ValueError, allow_unwrapped=True):
        raise ValueError

    # expecting multiple unwrapped exceptions is not possible
    with pytest.raises(
        ValueError,
        match=r"^You cannot specify multiple exceptions with",
    ):
        RaisesGroup(SyntaxError, ValueError, allow_unwrapped=True)  # type: ignore[call-overload]
    # if users want one of several exception types they need to use a Matcher
    # (which the error message suggests)
    with RaisesGroup(
        Matcher(check=lambda e: isinstance(e, (SyntaxError, ValueError))),
        allow_unwrapped=True,
    ):
        raise ValueError

    # Unwrapped nested `RaisesGroup` is likely a user error, so we raise an error.
    with pytest.raises(ValueError, match="has no effect when expecting"):
        RaisesGroup(RaisesGroup(ValueError), allow_unwrapped=True)  # type: ignore[call-overload]

    # But it *can* be used to check for nesting level +- 1 if they move it to
    # the nested RaisesGroup. Users should probably use `Matcher`s instead though.
    with RaisesGroup(RaisesGroup(ValueError, allow_unwrapped=True)):
        raise ExceptionGroup("", [ExceptionGroup("", [ValueError()])])
    with RaisesGroup(RaisesGroup(ValueError, allow_unwrapped=True)):
        raise ExceptionGroup("", [ValueError()])

    # with allow_unwrapped=False (default) it will not be caught
    with (
        fails_raises_group(
            "'ValueError' is not an exception group, but would match with `allow_unwrapped=True`"
        ),
        RaisesGroup(ValueError),
    ):
        raise ValueError("value error text")

    # allow_unwrapped on its own won't match against nested groups
    with (
        fails_raises_group(
            "Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "  Did you mean to use `flatten_subgroups=True`?",
        ),
        RaisesGroup(ValueError, allow_unwrapped=True),
    ):
        raise ExceptionGroup("foo", [ExceptionGroup("bar", [ValueError()])])

    # you need both allow_unwrapped and flatten_subgroups to fully emulate except*
    with RaisesGroup(ValueError, allow_unwrapped=True, flatten_subgroups=True):
        raise ExceptionGroup("", [ExceptionGroup("", [ValueError()])])

    # code coverage
    with (
        fails_raises_group(
            "Raised exception (group) did not match: 'TypeError' is not of type 'ValueError'",
            add_prefix=False,
        ),
        RaisesGroup(ValueError, allow_unwrapped=True),
    ):
        raise TypeError("this text doesn't show up in the error message")
    with (
        fails_raises_group(
            "Raised exception (group) did not match: Matcher(ValueError): 'TypeError' is not of type 'ValueError'",
            add_prefix=False,
        ),
        RaisesGroup(Matcher(ValueError), allow_unwrapped=True),
    ):
        raise TypeError

    # check we don't suggest unwrapping with nested RaisesGroup
    with (
        fails_raises_group("'ValueError' is not an exception group"),
        RaisesGroup(RaisesGroup(ValueError)),
    ):
        raise ValueError


def test_match() -> None:
    # supports match string
    with RaisesGroup(ValueError, match="bar"):
        raise ExceptionGroup("bar", (ValueError(),))

    # now also works with ^$
    with RaisesGroup(ValueError, match="^bar$"):
        raise ExceptionGroup("bar", (ValueError(),))

    # it also includes notes
    with RaisesGroup(ValueError, match="my note"):
        e = ExceptionGroup("bar", (ValueError(),))
        e.add_note("my note")
        raise e

    # and technically you can match it all with ^$
    # but you're probably better off using a Matcher at that point
    with RaisesGroup(ValueError, match="^bar\nmy note$"):
        e = ExceptionGroup("bar", (ValueError(),))
        e.add_note("my note")
        raise e

    with (
        fails_raises_group(
            "Regex pattern 'foo' did not match 'bar' of 'ExceptionGroup'"
        ),
        RaisesGroup(ValueError, match="foo"),
    ):
        raise ExceptionGroup("bar", (ValueError(),))

    # Suggest a fix for easy pitfall of adding match to the RaisesGroup instead of
    # using a Matcher.
    # This requires a single expected & raised exception, the expected is a type,
    # and `isinstance(raised, expected_type)`.
    with (
        fails_raises_group(
            "Regex pattern 'foo' did not match 'bar' of 'ExceptionGroup', but matched the expected 'ValueError'. You might want RaisesGroup(Matcher(ValueError, match='foo'))"
        ),
        RaisesGroup(ValueError, match="foo"),
    ):
        raise ExceptionGroup("bar", [ValueError("foo")])


def test_check() -> None:
    exc = ExceptionGroup("", (ValueError(),))

    def is_exc(e: ExceptionGroup[ValueError]) -> bool:
        return e is exc

    is_exc_repr = repr_callable(is_exc)
    with RaisesGroup(ValueError, check=is_exc):
        raise exc

    with (
        fails_raises_group(f"check {is_exc_repr} did not return True"),
        RaisesGroup(ValueError, check=is_exc),
    ):
        raise ExceptionGroup("", (ValueError(),))


def test_unwrapped_match_check() -> None:
    def my_check(e: object) -> bool:  # pragma: no cover
        return True

    msg = (
        "`allow_unwrapped=True` bypasses the `match` and `check` parameters"
        " if the exception is unwrapped. If you intended to match/check the"
        " exception you should use a `Matcher` object. If you want to match/check"
        " the exceptiongroup when the exception *is* wrapped you need to"
        " do e.g. `if isinstance(exc.value, ExceptionGroup):"
        " assert RaisesGroup(...).matches(exc.value)` afterwards."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        RaisesGroup(ValueError, allow_unwrapped=True, match="foo")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match=re.escape(msg)):
        RaisesGroup(ValueError, allow_unwrapped=True, check=my_check)  # type: ignore[call-overload]

    # Users should instead use a Matcher
    rg = RaisesGroup(Matcher(ValueError, match="^foo$"), allow_unwrapped=True)
    with rg:
        raise ValueError("foo")
    with rg:
        raise ExceptionGroup("", [ValueError("foo")])

    # or if they wanted to match/check the group, do a conditional `.matches()`
    with RaisesGroup(ValueError, allow_unwrapped=True) as exc:
        raise ExceptionGroup("bar", [ValueError("foo")])
    if isinstance(exc.value, ExceptionGroup):  # pragma: no branch
        assert RaisesGroup(ValueError, match="bar").matches(exc.value)


def test_RaisesGroup_matches() -> None:
    rg = RaisesGroup(ValueError)
    assert not rg.matches(None)
    assert not rg.matches(ValueError())
    assert rg.matches(ExceptionGroup("", (ValueError(),)))


def test_message() -> None:
    def check_message(
        message: str,
        body: RaisesGroup[BaseException],
    ) -> None:
        with (
            pytest.raises(
                AssertionError,
                match=f"^DID NOT RAISE any exception, expected {re.escape(message)}$",
            ),
            body,
        ):
            ...

    # basic
    check_message("ExceptionGroup(ValueError)", RaisesGroup(ValueError))
    # multiple exceptions
    check_message(
        "ExceptionGroup(ValueError, ValueError)",
        RaisesGroup(ValueError, ValueError),
    )
    # nested
    check_message(
        "ExceptionGroup(ExceptionGroup(ValueError))",
        RaisesGroup(RaisesGroup(ValueError)),
    )

    # Matcher
    check_message(
        "ExceptionGroup(Matcher(ValueError, match='my_str'))",
        RaisesGroup(Matcher(ValueError, "my_str")),
    )
    check_message(
        "ExceptionGroup(Matcher(match='my_str'))",
        RaisesGroup(Matcher(match="my_str")),
    )

    # BaseExceptionGroup
    check_message(
        "BaseExceptionGroup(KeyboardInterrupt)",
        RaisesGroup(KeyboardInterrupt),
    )
    # BaseExceptionGroup with type inside Matcher
    check_message(
        "BaseExceptionGroup(Matcher(KeyboardInterrupt))",
        RaisesGroup(Matcher(KeyboardInterrupt)),
    )
    # Base-ness transfers to parent containers
    check_message(
        "BaseExceptionGroup(BaseExceptionGroup(KeyboardInterrupt))",
        RaisesGroup(RaisesGroup(KeyboardInterrupt)),
    )
    # but not to child containers
    check_message(
        "BaseExceptionGroup(BaseExceptionGroup(KeyboardInterrupt), ExceptionGroup(ValueError))",
        RaisesGroup(RaisesGroup(KeyboardInterrupt), RaisesGroup(ValueError)),
    )


def test_assert_message() -> None:
    # the message does not need to list all parameters to RaisesGroup, nor all exceptions
    # in the exception group, as those are both visible in the traceback.
    # first fails to match
    with (
        fails_raises_group("'TypeError' is not of type 'ValueError'"),
        RaisesGroup(ValueError),
    ):
        raise ExceptionGroup("a", [TypeError()])
    with (
        fails_raises_group(
            "Raised exception group did not match: \n"
            "The following expected exceptions did not find a match:\n"
            "  RaisesGroup(ValueError)\n"
            "  RaisesGroup(ValueError, match='a')\n"
            "The following raised exceptions did not find a match\n"
            "  ExceptionGroup('', [RuntimeError()]):\n"
            "    RaisesGroup(ValueError): 'RuntimeError' is not of type 'ValueError'\n"
            "    RaisesGroup(ValueError, match='a'): Regex pattern 'a' did not match '' of 'ExceptionGroup'\n"
            "  RuntimeError():\n"
            "    RaisesGroup(ValueError): 'RuntimeError' is not an exception group\n"
            "    RaisesGroup(ValueError, match='a'): 'RuntimeError' is not an exception group",
            add_prefix=False,  # to see the full structure
        ),
        RaisesGroup(RaisesGroup(ValueError), RaisesGroup(ValueError, match="a")),
    ):
        raise ExceptionGroup(
            "",
            [ExceptionGroup("", [RuntimeError()]), RuntimeError()],
        )

    with (
        fails_raises_group(
            "Raised exception group did not match: \n"
            "2 matched exceptions. \n"
            "The following expected exceptions did not find a match:\n"
            "  RaisesGroup(RuntimeError)\n"
            "  RaisesGroup(ValueError)\n"
            "The following raised exceptions did not find a match\n"
            "  RuntimeError():\n"
            # "    'RuntimeError' is not of type 'ValueError'\n"
            # "    Matcher(TypeError): 'RuntimeError' is not of type 'TypeError'\n"
            "    RaisesGroup(RuntimeError): 'RuntimeError' is not an exception group, but would match with `allow_unwrapped=True`\n"
            "    RaisesGroup(ValueError): 'RuntimeError' is not an exception group\n"
            "  ValueError('bar'):\n"
            "    It matches 'ValueError' which was paired with ValueError('foo')\n"
            "    RaisesGroup(RuntimeError): 'ValueError' is not an exception group\n"
            "    RaisesGroup(ValueError): 'ValueError' is not an exception group, but would match with `allow_unwrapped=True`",
            add_prefix=False,  # to see the full structure
        ),
        RaisesGroup(
            ValueError,
            Matcher(TypeError),
            RaisesGroup(RuntimeError),
            RaisesGroup(ValueError),
        ),
    ):
        raise ExceptionGroup(
            "a",
            [RuntimeError(), TypeError(), ValueError("foo"), ValueError("bar")],
        )

    with (
        fails_raises_group(
            "1 matched exception. 'AssertionError' is not of type 'TypeError'"
        ),
        RaisesGroup(ValueError, TypeError),
    ):
        raise ExceptionGroup("a", [ValueError(), AssertionError()])

    with (
        fails_raises_group(
            "Matcher(ValueError): 'TypeError' is not of type 'ValueError'"
        ),
        RaisesGroup(Matcher(ValueError)),
    ):
        raise ExceptionGroup("a", [TypeError()])

    # suggest escaping
    with (
        fails_raises_group(
            "Raised exception group did not match: Regex pattern 'h(ell)o' did not match 'h(ell)o' of 'ExceptionGroup'\n"
            "  Did you mean to `re.escape()` the regex?",
            add_prefix=False,  # to see the full structure
        ),
        RaisesGroup(ValueError, match="h(ell)o"),
    ):
        raise ExceptionGroup("h(ell)o", [ValueError()])
    with (
        fails_raises_group(
            "Matcher(match='h(ell)o'): Regex pattern 'h(ell)o' did not match 'h(ell)o'\n"
            "  Did you mean to `re.escape()` the regex?",
        ),
        RaisesGroup(Matcher(match="h(ell)o")),
    ):
        raise ExceptionGroup("", [ValueError("h(ell)o")])

    with (
        fails_raises_group(
            "Raised exception group did not match: \n"
            "The following expected exceptions did not find a match:\n"
            "  'ValueError'\n"
            "  'ValueError'\n"
            "  'ValueError'\n"
            "  'ValueError'\n"
            "The following raised exceptions did not find a match\n"
            "  ExceptionGroup('', [ValueError(), TypeError()]):\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'",
            add_prefix=False,  # to see the full structure
        ),
        RaisesGroup(ValueError, ValueError, ValueError, ValueError),
    ):
        raise ExceptionGroup("", [ExceptionGroup("", [ValueError(), TypeError()])])


def test_message_indent() -> None:
    with (
        fails_raises_group(
            "Raised exception group did not match: \n"
            "The following expected exceptions did not find a match:\n"
            "  RaisesGroup(ValueError, ValueError)\n"
            "  'ValueError'\n"
            "The following raised exceptions did not find a match\n"
            "  ExceptionGroup('', [TypeError(), RuntimeError()]):\n"
            "    RaisesGroup(ValueError, ValueError): \n"
            "      The following expected exceptions did not find a match:\n"
            "        'ValueError'\n"
            "        'ValueError'\n"
            "      The following raised exceptions did not find a match\n"
            "        TypeError():\n"
            "          'TypeError' is not of type 'ValueError'\n"
            "          'TypeError' is not of type 'ValueError'\n"
            "        RuntimeError():\n"
            "          'RuntimeError' is not of type 'ValueError'\n"
            "          'RuntimeError' is not of type 'ValueError'\n"
            # TODO: this line is not great, should maybe follow the same format as the other and say
            # 'ValueError': Unexpected nested 'ExceptionGroup' (?)
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "  TypeError():\n"
            "    RaisesGroup(ValueError, ValueError): 'TypeError' is not an exception group\n"
            "    'TypeError' is not of type 'ValueError'",
            add_prefix=False,
        ),
        RaisesGroup(
            RaisesGroup(ValueError, ValueError),
            ValueError,
        ),
    ):
        raise ExceptionGroup(
            "",
            [
                ExceptionGroup("", [TypeError(), RuntimeError()]),
                TypeError(),
            ],
        )
    with (
        fails_raises_group(
            "Raised exception group did not match: \n"
            "RaisesGroup(ValueError, ValueError): \n"
            "  The following expected exceptions did not find a match:\n"
            "    'ValueError'\n"
            "    'ValueError'\n"
            "  The following raised exceptions did not find a match\n"
            "    TypeError():\n"
            "      'TypeError' is not of type 'ValueError'\n"
            "      'TypeError' is not of type 'ValueError'\n"
            "    RuntimeError():\n"
            "      'RuntimeError' is not of type 'ValueError'\n"
            "      'RuntimeError' is not of type 'ValueError'",
            add_prefix=False,
        ),
        RaisesGroup(
            RaisesGroup(ValueError, ValueError),
        ),
    ):
        raise ExceptionGroup(
            "",
            [
                ExceptionGroup("", [TypeError(), RuntimeError()]),
            ],
        )


def test_suggestion_on_nested_and_brief_error() -> None:
    # Make sure "Did you mean" suggestion gets indented iff it follows a single-line error
    with (
        fails_raises_group(
            "\n"
            "The following expected exceptions did not find a match:\n"
            "  RaisesGroup(ValueError)\n"
            "  'ValueError'\n"
            "The following raised exceptions did not find a match\n"
            "  ExceptionGroup('', [ExceptionGroup('', [ValueError()])]):\n"
            "    RaisesGroup(ValueError): Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "      Did you mean to use `flatten_subgroups=True`?\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'",
        ),
        RaisesGroup(RaisesGroup(ValueError), ValueError),
    ):
        raise ExceptionGroup(
            "",
            [ExceptionGroup("", [ExceptionGroup("", [ValueError()])])],
        )
    # if indented here it would look like another raised exception
    with (
        fails_raises_group(
            "\n"
            "The following expected exceptions did not find a match:\n"
            "  RaisesGroup(ValueError, ValueError)\n"
            "  'ValueError'\n"
            "The following raised exceptions did not find a match\n"
            "  ExceptionGroup('', [ValueError(), ExceptionGroup('', [ValueError()])]):\n"
            "    RaisesGroup(ValueError, ValueError): \n"
            "      1 matched exception. \n"
            "      The following expected exceptions did not find a match:\n"
            "        'ValueError'\n"
            "          It matches ValueError() which was paired with 'ValueError'\n"
            "      The following raised exceptions did not find a match\n"
            "        ExceptionGroup('', [ValueError()]):\n"
            "          Unexpected nested 'ExceptionGroup', expected 'ValueError'\n"
            "      Did you mean to use `flatten_subgroups=True`?\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'"
        ),
        RaisesGroup(RaisesGroup(ValueError, ValueError), ValueError),
    ):
        raise ExceptionGroup(
            "",
            [ExceptionGroup("", [ValueError(), ExceptionGroup("", [ValueError()])])],
        )

    # re.escape always comes after single-line errors
    with (
        fails_raises_group(
            "\n"
            "The following expected exceptions did not find a match:\n"
            "  RaisesGroup(Exception, match='^hello')\n"
            "  'ValueError'\n"
            "The following raised exceptions did not find a match\n"
            "  ExceptionGroup('^hello', [Exception()]):\n"
            "    RaisesGroup(Exception, match='^hello'): Regex pattern '^hello' did not match '^hello' of 'ExceptionGroup'\n"
            "      Did you mean to `re.escape()` the regex?\n"
            "    Unexpected nested 'ExceptionGroup', expected 'ValueError'"
        ),
        RaisesGroup(RaisesGroup(Exception, match="^hello"), ValueError),
    ):
        raise ExceptionGroup("", [ExceptionGroup("^hello", [Exception()])])


def test_assert_message_nested() -> None:
    # we only get one instance of aaaaaaaaaa... and bbbbbb..., but we do get multiple instances of ccccc... and dddddd..
    # but I think this now only prints the full repr when that is necessary to disambiguate exceptions
    with (
        fails_raises_group(
            "Raised exception group did not match: \n"
            "The following expected exceptions did not find a match:\n"
            "  RaisesGroup(ValueError)\n"
            "  RaisesGroup(RaisesGroup(ValueError))\n"
            "  RaisesGroup(Matcher(TypeError, match='foo'))\n"
            "  RaisesGroup(TypeError, ValueError)\n"
            "The following raised exceptions did not find a match\n"
            "  TypeError('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'):\n"
            "    RaisesGroup(ValueError): 'TypeError' is not an exception group\n"
            "    RaisesGroup(RaisesGroup(ValueError)): 'TypeError' is not an exception group\n"
            "    RaisesGroup(Matcher(TypeError, match='foo')): 'TypeError' is not an exception group\n"
            "    RaisesGroup(TypeError, ValueError): 'TypeError' is not an exception group\n"
            "  ExceptionGroup('Exceptions from Trio nursery', [TypeError('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')]):\n"
            "    RaisesGroup(ValueError): 'TypeError' is not of type 'ValueError'\n"
            "    RaisesGroup(RaisesGroup(ValueError)): RaisesGroup(ValueError): 'TypeError' is not an exception group\n"
            "    RaisesGroup(Matcher(TypeError, match='foo')): Matcher(TypeError, match='foo'): Regex pattern 'foo' did not match 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'\n"
            "    RaisesGroup(TypeError, ValueError): 1 matched exception. Too few exceptions raised, found no match for: ['ValueError']\n"
            "  ExceptionGroup('Exceptions from Trio nursery', [TypeError('cccccccccccccccccccccccccccccccccccccccc'), TypeError('dddddddddddddddddddddddddddddddddddddddd')]):\n"
            "    RaisesGroup(ValueError): \n"
            "      The following expected exceptions did not find a match:\n"
            "        'ValueError'\n"
            "      The following raised exceptions did not find a match\n"
            "        TypeError('cccccccccccccccccccccccccccccccccccccccc'):\n"
            "          'TypeError' is not of type 'ValueError'\n"
            "        TypeError('dddddddddddddddddddddddddddddddddddddddd'):\n"
            "          'TypeError' is not of type 'ValueError'\n"
            "    RaisesGroup(RaisesGroup(ValueError)): \n"
            "      The following expected exceptions did not find a match:\n"
            "        RaisesGroup(ValueError)\n"
            "      The following raised exceptions did not find a match\n"
            "        TypeError('cccccccccccccccccccccccccccccccccccccccc'):\n"
            "          RaisesGroup(ValueError): 'TypeError' is not an exception group\n"
            "        TypeError('dddddddddddddddddddddddddddddddddddddddd'):\n"
            "          RaisesGroup(ValueError): 'TypeError' is not an exception group\n"
            "    RaisesGroup(Matcher(TypeError, match='foo')): \n"
            "      The following expected exceptions did not find a match:\n"
            "        Matcher(TypeError, match='foo')\n"
            "      The following raised exceptions did not find a match\n"
            "        TypeError('cccccccccccccccccccccccccccccccccccccccc'):\n"
            "          Matcher(TypeError, match='foo'): Regex pattern 'foo' did not match 'cccccccccccccccccccccccccccccccccccccccc'\n"
            "        TypeError('dddddddddddddddddddddddddddddddddddddddd'):\n"
            "          Matcher(TypeError, match='foo'): Regex pattern 'foo' did not match 'dddddddddddddddddddddddddddddddddddddddd'\n"
            "    RaisesGroup(TypeError, ValueError): \n"
            "      1 matched exception. \n"
            "      The following expected exceptions did not find a match:\n"
            "        'ValueError'\n"
            "      The following raised exceptions did not find a match\n"
            "        TypeError('dddddddddddddddddddddddddddddddddddddddd'):\n"
            "          It matches 'TypeError' which was paired with TypeError('cccccccccccccccccccccccccccccccccccccccc')\n"
            "          'TypeError' is not of type 'ValueError'",
            add_prefix=False,  # to see the full structure
        ),
        RaisesGroup(
            RaisesGroup(ValueError),
            RaisesGroup(RaisesGroup(ValueError)),
            RaisesGroup(Matcher(TypeError, match="foo")),
            RaisesGroup(TypeError, ValueError),
        ),
    ):
        raise ExceptionGroup(
            "",
            [
                TypeError("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                ExceptionGroup(
                    "Exceptions from Trio nursery",
                    [TypeError("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")],
                ),
                ExceptionGroup(
                    "Exceptions from Trio nursery",
                    [
                        TypeError("cccccccccccccccccccccccccccccccccccccccc"),
                        TypeError("dddddddddddddddddddddddddddddddddddddddd"),
                    ],
                ),
            ],
        )


@pytest.mark.skipif(
    "hypothesis" in sys.modules,
    reason="hypothesis may have monkeypatched _check_repr",
)
def test_check_no_patched_repr() -> None:
    # We make `_check_repr` monkeypatchable to avoid this very ugly and verbose
    # repr. The other tests that use `check` make use of `_check_repr` so they'll
    # continue passing in case it is patched - but we have this one test that
    # demonstrates just how nasty it gets otherwise.
    with (
        pytest.raises(
            AssertionError,
            match=(
                r"^Raised exception group did not match: \n"
                r"The following expected exceptions did not find a match:\n"
                r"  Matcher\(check=<function test_check_no_patched_repr.<locals>.<lambda> at .*>\)\n"
                r"  'TypeError'\n"
                r"The following raised exceptions did not find a match\n"
                r"  ValueError\('foo'\):\n"
                r"    Matcher\(check=<function test_check_no_patched_repr.<locals>.<lambda> at .*>\): check did not return True\n"
                r"    'ValueError' is not of type 'TypeError'\n"
                r"  ValueError\('bar'\):\n"
                r"    Matcher\(check=<function test_check_no_patched_repr.<locals>.<lambda> at .*>\): check did not return True\n"
                r"    'ValueError' is not of type 'TypeError'$"
            ),
        ),
        RaisesGroup(Matcher(check=lambda x: False), TypeError),
    ):
        raise ExceptionGroup("", [ValueError("foo"), ValueError("bar")])


def test_misordering_example() -> None:
    with (
        fails_raises_group(
            "\n"
            "3 matched exceptions. \n"
            "The following expected exceptions did not find a match:\n"
            "  Matcher(ValueError, match='foo')\n"
            "    It matches ValueError('foo') which was paired with 'ValueError'\n"
            "    It matches ValueError('foo') which was paired with 'ValueError'\n"
            "    It matches ValueError('foo') which was paired with 'ValueError'\n"
            "The following raised exceptions did not find a match\n"
            "  ValueError('bar'):\n"
            "    It matches 'ValueError' which was paired with ValueError('foo')\n"
            "    It matches 'ValueError' which was paired with ValueError('foo')\n"
            "    It matches 'ValueError' which was paired with ValueError('foo')\n"
            "    Matcher(ValueError, match='foo'): Regex pattern 'foo' did not match 'bar'\n"
            "There exist a possible match when attempting an exhaustive check, but RaisesGroup uses a greedy algorithm. Please make your expected exceptions more stringent with `Matcher` etc so the greedy algorithm can function."
        ),
        RaisesGroup(
            ValueError, ValueError, ValueError, Matcher(ValueError, match="foo")
        ),
    ):
        raise ExceptionGroup(
            "",
            [
                ValueError("foo"),
                ValueError("foo"),
                ValueError("foo"),
                ValueError("bar"),
            ],
        )


def test_brief_error_on_one_fail() -> None:
    """if only one raised and one expected fail to match up, we print a full table iff
    the raised exception would match one of the expected that previously got matched"""
    # no also-matched
    with (
        fails_raises_group(
            "1 matched exception. 'TypeError' is not of type 'RuntimeError'"
        ),
        RaisesGroup(ValueError, RuntimeError),
    ):
        raise ExceptionGroup("", [ValueError(), TypeError()])

    # raised would match an expected
    with (
        fails_raises_group(
            "\n"
            "1 matched exception. \n"
            "The following expected exceptions did not find a match:\n"
            "  'RuntimeError'\n"
            "The following raised exceptions did not find a match\n"
            "  TypeError():\n"
            "    It matches 'Exception' which was paired with ValueError()\n"
            "    'TypeError' is not of type 'RuntimeError'"
        ),
        RaisesGroup(Exception, RuntimeError),
    ):
        raise ExceptionGroup("", [ValueError(), TypeError()])

    # expected would match a raised
    with (
        fails_raises_group(
            "\n"
            "1 matched exception. \n"
            "The following expected exceptions did not find a match:\n"
            "  'ValueError'\n"
            "    It matches ValueError() which was paired with 'ValueError'\n"
            "The following raised exceptions did not find a match\n"
            "  TypeError():\n"
            "    'TypeError' is not of type 'ValueError'"
        ),
        RaisesGroup(ValueError, ValueError),
    ):
        raise ExceptionGroup("", [ValueError(), TypeError()])


def test_identity_oopsies() -> None:
    # it's both possible to have several instances of the same exception in the same group
    # and to expect multiple of the same type
    # this previously messed up the logic

    with (
        fails_raises_group(
            "3 matched exceptions. 'RuntimeError' is not of type 'TypeError'"
        ),
        RaisesGroup(ValueError, ValueError, ValueError, TypeError),
    ):
        raise ExceptionGroup(
            "", [ValueError(), ValueError(), ValueError(), RuntimeError()]
        )

    e = ValueError("foo")
    m = Matcher(match="bar")
    with (
        fails_raises_group(
            "\n"
            "The following expected exceptions did not find a match:\n"
            "  Matcher(match='bar')\n"
            "  Matcher(match='bar')\n"
            "  Matcher(match='bar')\n"
            "The following raised exceptions did not find a match\n"
            "  ValueError('foo'):\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'\n"
            "  ValueError('foo'):\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'\n"
            "  ValueError('foo'):\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'\n"
            "    Matcher(match='bar'): Regex pattern 'bar' did not match 'foo'"
        ),
        RaisesGroup(m, m, m),
    ):
        raise ExceptionGroup("", [e, e, e])


def test_matcher() -> None:
    with pytest.raises(
        ValueError,
        match=r"^You must specify at least one parameter to match on.$",
    ):
        Matcher()  # type: ignore[call-overload]
    with pytest.raises(
        ValueError,
        match=f"^exception_type {re.escape(repr(object))} must be a subclass of BaseException$",
    ):
        Matcher(object)  # type: ignore[type-var]

    with RaisesGroup(Matcher(ValueError)):
        raise ExceptionGroup("", (ValueError(),))
    with (
        fails_raises_group(
            "Matcher(TypeError): 'ValueError' is not of type 'TypeError'"
        ),
        RaisesGroup(Matcher(TypeError)),
    ):
        raise ExceptionGroup("", (ValueError(),))


def test_matcher_match() -> None:
    with RaisesGroup(Matcher(ValueError, "foo")):
        raise ExceptionGroup("", (ValueError("foo"),))
    with (
        fails_raises_group(
            "Matcher(ValueError, match='foo'): Regex pattern 'foo' did not match 'bar'"
        ),
        RaisesGroup(Matcher(ValueError, "foo")),
    ):
        raise ExceptionGroup("", (ValueError("bar"),))

    # Can be used without specifying the type
    with RaisesGroup(Matcher(match="foo")):
        raise ExceptionGroup("", (ValueError("foo"),))
    with (
        fails_raises_group(
            "Matcher(match='foo'): Regex pattern 'foo' did not match 'bar'"
        ),
        RaisesGroup(Matcher(match="foo")),
    ):
        raise ExceptionGroup("", (ValueError("bar"),))

    # check ^$
    with RaisesGroup(Matcher(ValueError, match="^bar$")):
        raise ExceptionGroup("", [ValueError("bar")])
    with (
        fails_raises_group(
            "Matcher(ValueError, match='^bar$'): Regex pattern '^bar$' did not match 'barr'"
        ),
        RaisesGroup(Matcher(ValueError, match="^bar$")),
    ):
        raise ExceptionGroup("", [ValueError("barr")])


def test_Matcher_check() -> None:
    def check_oserror_and_errno_is_5(e: BaseException) -> bool:
        return isinstance(e, OSError) and e.errno == 5

    with RaisesGroup(Matcher(check=check_oserror_and_errno_is_5)):
        raise ExceptionGroup("", (OSError(5, ""),))

    # specifying exception_type narrows the parameter type to the callable
    def check_errno_is_5(e: OSError) -> bool:
        return e.errno == 5

    with RaisesGroup(Matcher(OSError, check=check_errno_is_5)):
        raise ExceptionGroup("", (OSError(5, ""),))

    # avoid printing overly verbose repr multiple times
    with (
        fails_raises_group(
            f"Matcher(OSError, check={check_errno_is_5!r}): check did not return True"
        ),
        RaisesGroup(Matcher(OSError, check=check_errno_is_5)),
    ):
        raise ExceptionGroup("", (OSError(6, ""),))

    # in nested cases you still get it multiple times though
    # to address this you'd need logic in Matcher.__repr__ and RaisesGroup.__repr__
    with (
        fails_raises_group(
            f"RaisesGroup(Matcher(OSError, check={check_errno_is_5!r})): Matcher(OSError, check={check_errno_is_5!r}): check did not return True"
        ),
        RaisesGroup(RaisesGroup(Matcher(OSError, check=check_errno_is_5))),
    ):
        raise ExceptionGroup("", [ExceptionGroup("", [OSError(6, "")])])


def test_matcher_tostring() -> None:
    assert str(Matcher(ValueError)) == "Matcher(ValueError)"
    assert str(Matcher(match="[a-z]")) == "Matcher(match='[a-z]')"
    pattern_no_flags = re.compile(r"noflag", 0)
    assert str(Matcher(match=pattern_no_flags)) == "Matcher(match='noflag')"
    pattern_flags = re.compile(r"noflag", re.IGNORECASE)
    assert str(Matcher(match=pattern_flags)) == f"Matcher(match={pattern_flags!r})"
    assert (
        str(Matcher(ValueError, match="re", check=bool))
        == f"Matcher(ValueError, match='re', check={bool!r})"
    )


def test_raisesgroup_tostring() -> None:
    def check_str_and_repr(s: str) -> None:
        evaled = eval(s)
        assert s == str(evaled) == repr(evaled)

    check_str_and_repr("RaisesGroup(ValueError)")
    check_str_and_repr("RaisesGroup(RaisesGroup(ValueError))")
    check_str_and_repr("RaisesGroup(Matcher(ValueError))")
    check_str_and_repr("RaisesGroup(ValueError, allow_unwrapped=True)")
    check_str_and_repr("RaisesGroup(ValueError, match='aoeu')")

    assert (
        str(RaisesGroup(ValueError, match="[a-z]", check=bool))
        == f"RaisesGroup(ValueError, match='[a-z]', check={bool!r})"
    )


def test__ExceptionInfo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trio.testing._raises_group,
        "ExceptionInfo",
        trio.testing._raises_group._ExceptionInfo,
    )
    with trio.testing.RaisesGroup(ValueError) as excinfo:
        raise ExceptionGroup("", (ValueError("hello"),))
    assert excinfo.type is ExceptionGroup
    assert excinfo.value.exceptions[0].args == ("hello",)
    assert isinstance(excinfo.tb, TracebackType)