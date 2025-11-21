
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\calculus\quadrature.py ===
import math

from ..libmp.backend import xrange

class QuadratureRule(object):
    """
    Quadrature rules are implemented using this class, in order to
    simplify the code and provide a common infrastructure
    for tasks such as error estimation and node caching.

    You can implement a custom quadrature rule by subclassing
    :class:`QuadratureRule` and implementing the appropriate
    methods. The subclass can then be used by :func:`~mpmath.quad` by
    passing it as the *method* argument.

    :class:`QuadratureRule` instances are supposed to be singletons.
    :class:`QuadratureRule` therefore implements instance caching
    in :func:`~mpmath.__new__`.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.standard_cache = {}
        self.transformed_cache = {}
        self.interval_count = {}

    def clear(self):
        """
        Delete cached node data.
        """
        self.standard_cache = {}
        self.transformed_cache = {}
        self.interval_count = {}

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        Compute nodes for the standard interval `[-1, 1]`. Subclasses
        should probably implement only this method, and use
        :func:`~mpmath.get_nodes` method to retrieve the nodes.
        """
        raise NotImplementedError

    def get_nodes(self, a, b, degree, prec, verbose=False):
        """
        Return nodes for given interval, degree and precision. The
        nodes are retrieved from a cache if already computed;
        otherwise they are computed by calling :func:`~mpmath.calc_nodes`
        and are then cached.

        Subclasses should probably not implement this method,
        but just implement :func:`~mpmath.calc_nodes` for the actual
        node computation.
        """
        key = (a, b, degree, prec)
        if key in self.transformed_cache:
            return self.transformed_cache[key]
        orig = self.ctx.prec
        try:
            self.ctx.prec = prec+20
            # Get nodes on standard interval
            if (degree, prec) in self.standard_cache:
                nodes = self.standard_cache[degree, prec]
            else:
                nodes = self.calc_nodes(degree, prec, verbose)
                self.standard_cache[degree, prec] = nodes
            # Transform to general interval
            nodes = self.transform_nodes(nodes, a, b, verbose)
            if key in self.interval_count:
                self.transformed_cache[key] = nodes
            else:
                self.interval_count[key] = True
        finally:
            self.ctx.prec = orig
        return nodes

    def transform_nodes(self, nodes, a, b, verbose=False):
        r"""
        Rescale standardized nodes (for `[-1, 1]`) to a general
        interval `[a, b]`. For a finite interval, a simple linear
        change of variables is used. Otherwise, the following
        transformations are used:

        .. math ::

            \lbrack a, \infty \rbrack : t = \frac{1}{x} + (a-1)

            \lbrack -\infty, b \rbrack : t = (b+1) - \frac{1}{x}

            \lbrack -\infty, \infty \rbrack : t = \frac{x}{\sqrt{1-x^2}}

        """
        ctx = self.ctx
        a = ctx.convert(a)
        b = ctx.convert(b)
        one = ctx.one
        if (a, b) == (-one, one):
            return nodes
        half = ctx.mpf(0.5)
        new_nodes = []
        if ctx.isinf(a) or ctx.isinf(b):
            if (a, b) == (ctx.ninf, ctx.inf):
                p05 = -half
                for x, w in nodes:
                    x2 = x*x
                    px1 = one-x2
                    spx1 = px1**p05
                    x = x*spx1
                    w *= spx1/px1
                    new_nodes.append((x, w))
            elif a == ctx.ninf:
                b1 = b+1
                for x, w in nodes:
                    u = 2/(x+one)
                    x = b1-u
                    w *= half*u**2
                    new_nodes.append((x, w))
            elif b == ctx.inf:
                a1 = a-1
                for x, w in nodes:
                    u = 2/(x+one)
                    x = a1+u
                    w *= half*u**2
                    new_nodes.append((x, w))
            elif a == ctx.inf or b == ctx.ninf:
                return [(x,-w) for (x,w) in self.transform_nodes(nodes, b, a, verbose)]
            else:
                raise NotImplementedError
        else:
            # Simple linear change of variables
            C = (b-a)/2
            D = (b+a)/2
            for x, w in nodes:
                new_nodes.append((D+C*x, C*w))
        return new_nodes

    def guess_degree(self, prec):
        """
        Given a desired precision `p` in bits, estimate the degree `m`
        of the quadrature required to accomplish full accuracy for
        typical integrals. By default, :func:`~mpmath.quad` will perform up
        to `m` iterations. The value of `m` should be a slight
        overestimate, so that "slightly bad" integrals can be dealt
        with automatically using a few extra iterations. On the
        other hand, it should not be too big, so :func:`~mpmath.quad` can
        quit within a reasonable amount of time when it is given
        an "unsolvable" integral.

        The default formula used by :func:`~mpmath.guess_degree` is tuned
        for both :class:`TanhSinh` and :class:`GaussLegendre`.
        The output is roughly as follows:

            +---------+---------+
            | `p`     | `m`     |
            +=========+=========+
            | 50      | 6       |
            +---------+---------+
            | 100     | 7       |
            +---------+---------+
            | 500     | 10      |
            +---------+---------+
            | 3000    | 12      |
            +---------+---------+

        This formula is based purely on a limited amount of
        experimentation and will sometimes be wrong.
        """
        # Expected degree
        # XXX: use mag
        g = int(4 + max(0, self.ctx.log(prec/30.0, 2)))
        # Reasonable "worst case"
        g += 2
        return g

    def estimate_error(self, results, prec, epsilon):
        r"""
        Given results from integrations `[I_1, I_2, \ldots, I_k]` done
        with a quadrature of rule of degree `1, 2, \ldots, k`, estimate
        the error of `I_k`.

        For `k = 2`, we estimate  `|I_{\infty}-I_2|` as `|I_2-I_1|`.

        For `k > 2`, we extrapolate `|I_{\infty}-I_k| \approx |I_{k+1}-I_k|`
        from `|I_k-I_{k-1}|` and `|I_k-I_{k-2}|` under the assumption
        that each degree increment roughly doubles the accuracy of
        the quadrature rule (this is true for both :class:`TanhSinh`
        and :class:`GaussLegendre`). The extrapolation formula is given
        by Borwein, Bailey & Girgensohn. Although not very conservative,
        this method seems to be very robust in practice.
        """
        if len(results) == 2:
            return abs(results[0]-results[1])
        try:
            if results[-1] == results[-2] == results[-3]:
                return self.ctx.zero
            D1 = self.ctx.log(abs(results[-1]-results[-2]), 10)
            D2 = self.ctx.log(abs(results[-1]-results[-3]), 10)
        except ValueError:
            return epsilon
        D3 = -prec
        D4 = min(0, max(D1**2/D2, 2*D1, D3))
        return self.ctx.mpf(10) ** int(D4)

    def summation(self, f, points, prec, epsilon, max_degree, verbose=False):
        """
        Main integration function. Computes the 1D integral over
        the interval specified by *points*. For each subinterval,
        performs quadrature of degree from 1 up to *max_degree*
        until :func:`~mpmath.estimate_error` signals convergence.

        :func:`~mpmath.summation` transforms each subintegration to
        the standard interval and then calls :func:`~mpmath.sum_next`.
        """
        ctx = self.ctx
        I = total_err = ctx.zero
        for i in xrange(len(points)-1):
            a, b = points[i], points[i+1]
            if a == b:
                continue
            # XXX: we could use a single variable transformation,
            # but this is not good in practice. We get better accuracy
            # by having 0 as an endpoint.
            if (a, b) == (ctx.ninf, ctx.inf):
                _f = f
                f = lambda x: _f(-x) + _f(x)
                a, b = (ctx.zero, ctx.inf)
            results = []
            err = ctx.zero
            for degree in xrange(1, max_degree+1):
                nodes = self.get_nodes(a, b, degree, prec, verbose)
                if verbose:
                    print("Integrating from %s to %s (degree %s of %s)" % \
                        (ctx.nstr(a), ctx.nstr(b), degree, max_degree))
                result = self.sum_next(f, nodes, degree, prec, results, verbose)
                results.append(result)
                if degree > 1:
                    err = self.estimate_error(results, prec, epsilon)
                    if verbose:
                        print("Estimated error:", ctx.nstr(err), " epsilon:", ctx.nstr(epsilon), " result: ", ctx.nstr(result))
                    if err <= epsilon:
                        break
            I += results[-1]
            total_err += err
        if total_err > epsilon:
            if verbose:
                print("Failed to reach full accuracy. Estimated error:", ctx.nstr(total_err))
        return I, total_err

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        r"""
        Evaluates the step sum `\sum w_k f(x_k)` where the *nodes* list
        contains the `(w_k, x_k)` pairs.

        :func:`~mpmath.summation` will supply the list *results* of
        values computed by :func:`~mpmath.sum_next` at previous degrees, in
        case the quadrature rule is able to reuse them.
        """
        return self.ctx.fdot((w, f(x)) for (x,w) in nodes)


class TanhSinh(QuadratureRule):
    r"""
    This class implements "tanh-sinh" or "doubly exponential"
    quadrature. This quadrature rule is based on the Euler-Maclaurin
    integral formula. By performing a change of variables involving
    nested exponentials / hyperbolic functions (hence the name), the
    derivatives at the endpoints vanish rapidly. Since the error term
    in the Euler-Maclaurin formula depends on the derivatives at the
    endpoints, a simple step sum becomes extremely accurate. In
    practice, this means that doubling the number of evaluation
    points roughly doubles the number of accurate digits.

    Comparison to Gauss-Legendre:
      * Initial computation of nodes is usually faster
      * Handles endpoint singularities better
      * Handles infinite integration intervals better
      * Is slower for smooth integrands once nodes have been computed

    The implementation of the tanh-sinh algorithm is based on the
    description given in Borwein, Bailey & Girgensohn, "Experimentation
    in Mathematics - Computational Paths to Discovery", A K Peters,
    2003, pages 312-313. In the present implementation, a few
    improvements have been made:

      * A more efficient scheme is used to compute nodes (exploiting
        recurrence for the exponential function)
      * The nodes are computed successively instead of all at once

    **References**

    * [Bailey]_
    * http://users.cs.dal.ca/~jborwein/tanh-sinh.pdf

    """

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        """
        Step sum for tanh-sinh quadrature of degree `m`. We exploit the
        fact that half of the abscissas at degree `m` are precisely the
        abscissas from degree `m-1`. Thus reusing the result from
        the previous level allows a 2x speedup.
        """
        h = self.ctx.mpf(2)**(-degree)
        # Abscissas overlap, so reusing saves half of the time
        if previous:
            S = previous[-1]/(h*2)
        else:
            S = self.ctx.zero
        S += self.ctx.fdot((w,f(x)) for (x,w) in nodes)
        return h*S

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        The abscissas and weights for tanh-sinh quadrature of degree
        `m` are given by

        .. math::

            x_k = \tanh(\pi/2 \sinh(t_k))

            w_k = \pi/2 \cosh(t_k) / \cosh(\pi/2 \sinh(t_k))^2

        where `t_k = t_0 + hk` for a step length `h \sim 2^{-m}`. The
        list of nodes is actually infinite, but the weights die off so
        rapidly that only a few are needed.
        """
        ctx = self.ctx
        nodes = []

        extra = 20
        ctx.prec += extra
        tol = ctx.ldexp(1, -prec-10)
        pi4 = ctx.pi/4

        # For simplicity, we work in steps h = 1/2^n, with the first point
        # offset so that we can reuse the sum from the previous degree

        # We define degree 1 to include the "degree 0" steps, including
        # the point x = 0. (It doesn't work well otherwise; not sure why.)
        t0 = ctx.ldexp(1, -degree)
        if degree == 1:
            #nodes.append((mpf(0), pi4))
            #nodes.append((-mpf(0), pi4))
            nodes.append((ctx.zero, ctx.pi/2))
            h = t0
        else:
            h = t0*2

        # Since h is fixed, we can compute the next exponential
        # by simply multiplying by exp(h)
        expt0 = ctx.exp(t0)
        a = pi4 * expt0
        b = pi4 / expt0
        udelta = ctx.exp(h)
        urdelta = 1/udelta

        for k in xrange(0, 20*2**degree+1):
            # Reference implementation:
            # t = t0 + k*h
            # x = tanh(pi/2 * sinh(t))
            # w = pi/2 * cosh(t) / cosh(pi/2 * sinh(t))**2

            # Fast implementation. Note that c = exp(pi/2 * sinh(t))
            c = ctx.exp(a-b)
            d = 1/c
            co = (c+d)/2
            si = (c-d)/2
            x = si / co
            w = (a+b) / co**2
            diff = abs(x-1)
            if diff <= tol:
                break

            nodes.append((x, w))
            nodes.append((-x, w))

            a *= udelta
            b *= urdelta

            if verbose and k % 300 == 150:
                # Note: the number displayed is rather arbitrary. Should
                # figure out how to print something that looks more like a
                # percentage
                print("Calculating nodes:", ctx.nstr(-ctx.log(diff, 10) / prec))

        ctx.prec -= extra
        return nodes


class GaussLegendre(QuadratureRule):
    r"""
    This class implements Gauss-Legendre quadrature, which is
    exceptionally efficient for polynomials and polynomial-like (i.e.
    very smooth) integrands.

    The abscissas and weights are given by roots and values of
    Legendre polynomials, which are the orthogonal polynomials
    on `[-1, 1]` with respect to the unit weight
    (see :func:`~mpmath.legendre`).

    In this implementation, we take the "degree" `m` of the quadrature
    to denote a Gauss-Legendre rule of degree `3 \cdot 2^m` (following
    Borwein, Bailey & Girgensohn). This way we get quadratic, rather
    than linear, convergence as the degree is incremented.

    Comparison to tanh-sinh quadrature:
      * Is faster for smooth integrands once nodes have been computed
      * Initial computation of nodes is usually slower
      * Handles endpoint singularities worse
      * Handles infinite integration intervals worse

    """

    def calc_nodes(self, degree, prec, verbose=False):
        r"""
        Calculates the abscissas and weights for Gauss-Legendre
        quadrature of degree of given degree (actually `3 \cdot 2^m`).
        """
        ctx = self.ctx
        # It is important that the epsilon is set lower than the
        # "real" epsilon
        epsilon = ctx.ldexp(1, -prec-8)
        # Fairly high precision might be required for accurate
        # evaluation of the roots
        orig = ctx.prec
        ctx.prec = int(prec*1.5)
        if degree == 1:
            x = ctx.sqrt(ctx.mpf(3)/5)
            w = ctx.mpf(5)/9
            nodes = [(-x,w),(ctx.zero,ctx.mpf(8)/9),(x,w)]
            ctx.prec = orig
            return nodes
        nodes = []
        n = 3*2**(degree-1)
        upto = n//2 + 1
        for j in xrange(1, upto):
            # Asymptotic formula for the roots
            r = ctx.mpf(math.cos(math.pi*(j-0.25)/(n+0.5)))
            # Newton iteration
            while 1:
                t1, t2 = 1, 0
                # Evaluates the Legendre polynomial using its defining
                # recurrence relation
                for j1 in xrange(1,n+1):
                    t3, t2, t1 = t2, t1, ((2*j1-1)*r*t1 - (j1-1)*t2)/j1
                t4 = n*(r*t1-t2)/(r**2-1)
                a = t1/t4
                r = r - a
                if abs(a) < epsilon:
                    break
            x = r
            w = 2/((1-r**2)*t4**2)
            if verbose  and j % 30 == 15:
                print("Computing nodes (%i of %i)" % (j, upto))
            nodes.append((x, w))
            nodes.append((-x, w))
        ctx.prec = orig
        return nodes

class QuadratureMethods(object):

    def __init__(ctx, *args, **kwargs):
        ctx._gauss_legendre = GaussLegendre(ctx)
        ctx._tanh_sinh = TanhSinh(ctx)

    def quad(ctx, f, *points, **kwargs):
        r"""
        Computes a single, double or triple integral over a given
        1D interval, 2D rectangle, or 3D cuboid. A basic example::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> quad(sin, [0, pi])
            2.0

        A basic 2D integral::

            >>> f = lambda x, y: cos(x+y/2)
            >>> quad(f, [-pi/2, pi/2], [0, pi])
            4.0

        **Interval format**

        The integration range for each dimension may be specified
        using a list or tuple. Arguments are interpreted as follows:

        ``quad(f, [x1, x2])`` -- calculates
        `\int_{x_1}^{x_2} f(x) \, dx`

        ``quad(f, [x1, x2], [y1, y2])`` -- calculates
        `\int_{x_1}^{x_2} \int_{y_1}^{y_2} f(x,y) \, dy \, dx`

        ``quad(f, [x1, x2], [y1, y2], [z1, z2])`` -- calculates
        `\int_{x_1}^{x_2} \int_{y_1}^{y_2} \int_{z_1}^{z_2} f(x,y,z)
        \, dz \, dy \, dx`

        Endpoints may be finite or infinite. An interval descriptor
        may also contain more than two points. In this
        case, the integration is split into subintervals, between
        each pair of consecutive points. This is useful for
        dealing with mid-interval discontinuities, or integrating
        over large intervals where the function is irregular or
        oscillates.

        **Options**

        :func:`~mpmath.quad` recognizes the following keyword arguments:

        *method*
            Chooses integration algorithm (described below).
        *error*
            If set to true, :func:`~mpmath.quad` returns `(v, e)` where `v` is the
            integral and `e` is the estimated error.
        *maxdegree*
            Maximum degree of the quadrature rule to try before
            quitting.
        *verbose*
            Print details about progress.

        **Algorithms**

        Mpmath presently implements two integration algorithms: tanh-sinh
        quadrature and Gauss-Legendre quadrature. These can be selected
        using *method='tanh-sinh'* or *method='gauss-legendre'* or by
        passing the classes *method=TanhSinh*, *method=GaussLegendre*.
        The functions :func:`~mpmath.quadts` and :func:`~mpmath.quadgl` are also available
        as shortcuts.

        Both algorithms have the property that doubling the number of
        evaluation points roughly doubles the accuracy, so both are ideal
        for high precision quadrature (hundreds or thousands of digits).

        At high precision, computing the nodes and weights for the
        integration can be expensive (more expensive than computing the
        function values). To make repeated integrations fast, nodes
        are automatically cached.

        The advantages of the tanh-sinh algorithm are that it tends to
        handle endpoint singularities well, and that the nodes are cheap
        to compute on the first run. For these reasons, it is used by
        :func:`~mpmath.quad` as the default algorithm.

        Gauss-Legendre quadrature often requires fewer function
        evaluations, and is therefore often faster for repeated use, but
        the algorithm does not handle endpoint singularities as well and
        the nodes are more expensive to compute. Gauss-Legendre quadrature
        can be a better choice if the integrand is smooth and repeated
        integrations are required (e.g. for multiple integrals).

        See the documentation for :class:`TanhSinh` and
        :class:`GaussLegendre` for additional details.

        **Examples of 1D integrals**

        Intervals may be infinite or half-infinite. The following two
        examples evaluate the limits of the inverse tangent function
        (`\int 1/(1+x^2) = \tan^{-1} x`), and the Gaussian integral
        `\int_{\infty}^{\infty} \exp(-x^2)\,dx = \sqrt{\pi}`::

            >>> mp.dps = 15
            >>> quad(lambda x: 2/(x**2+1), [0, inf])
            3.14159265358979
            >>> quad(lambda x: exp(-x**2), [-inf, inf])**2
            3.14159265358979

        Integrals can typically be resolved to high precision.
        The following computes 50 digits of `\pi` by integrating the
        area of the half-circle defined by `x^2 + y^2 \le 1`,
        `-1 \le x \le 1`, `y \ge 0`::

            >>> mp.dps = 50
            >>> 2*quad(lambda x: sqrt(1-x**2), [-1, 1])
            3.1415926535897932384626433832795028841971693993751

        One can just as well compute 1000 digits (output truncated)::

            >>> mp.dps = 1000
            >>> 2*quad(lambda x: sqrt(1-x**2), [-1, 1])  #doctest:+ELLIPSIS
            3.141592653589793238462643383279502884...216420199

        Complex integrals are supported. The following computes
        a residue at `z = 0` by integrating counterclockwise along the
        diamond-shaped path from `1` to `+i` to `-1` to `-i` to `1`::

            >>> mp.dps = 15
            >>> chop(quad(lambda z: 1/z, [1,j,-1,-j,1]))
            (0.0 + 6.28318530717959j)

        **Examples of 2D and 3D integrals**

        Here are several nice examples of analytically solvable
        2D integrals (taken from MathWorld [1]) that can be evaluated
        to high precision fairly rapidly by :func:`~mpmath.quad`::

            >>> mp.dps = 30
            >>> f = lambda x, y: (x-1)/((1-x*y)*log(x*y))
            >>> quad(f, [0, 1], [0, 1])
            0.577215664901532860606512090082
            >>> +euler
            0.577215664901532860606512090082

            >>> f = lambda x, y: 1/sqrt(1+x**2+y**2)
            >>> quad(f, [-1, 1], [-1, 1])
            3.17343648530607134219175646705
            >>> 4*log(2+sqrt(3))-2*pi/3
            3.17343648530607134219175646705

            >>> f = lambda x, y: 1/(1-x**2 * y**2)
            >>> quad(f, [0, 1], [0, 1])
            1.23370055013616982735431137498
            >>> pi**2 / 8
            1.23370055013616982735431137498

            >>> quad(lambda x, y: 1/(1-x*y), [0, 1], [0, 1])
            1.64493406684822643647241516665
            >>> pi**2 / 6
            1.64493406684822643647241516665

        Multiple integrals may be done over infinite ranges::

            >>> mp.dps = 15
            >>> print(quad(lambda x,y: exp(-x-y), [0, inf], [1, inf]))
            0.367879441171442
            >>> print(1/e)
            0.367879441171442

        For nonrectangular areas, one can call :func:`~mpmath.quad` recursively.
        For example, we can replicate the earlier example of calculating
        `\pi` by integrating over the unit-circle, and actually use double
        quadrature to actually measure the area circle::

            >>> f = lambda x: quad(lambda y: 1, [-sqrt(1-x**2), sqrt(1-x**2)])
            >>> quad(f, [-1, 1])
            3.14159265358979

        Here is a simple triple integral::

            >>> mp.dps = 15
            >>> f = lambda x,y,z: x*y/(1+z)
            >>> quad(f, [0,1], [0,1], [1,2], method='gauss-legendre')
            0.101366277027041
            >>> (log(3)-log(2))/4
            0.101366277027041

        **Singularities**

        Both tanh-sinh and Gauss-Legendre quadrature are designed to
        integrate smooth (infinitely differentiable) functions. Neither
        algorithm copes well with mid-interval singularities (such as
        mid-interval discontinuities in `f(x)` or `f'(x)`).
        The best solution is to split the integral into parts::

            >>> mp.dps = 15
            >>> quad(lambda x: abs(sin(x)), [0, 2*pi])   # Bad
            3.99900894176779
            >>> quad(lambda x: abs(sin(x)), [0, pi, 2*pi])  # Good
            4.0

        The tanh-sinh rule often works well for integrands having a
        singularity at one or both endpoints::

            >>> mp.dps = 15
            >>> quad(log, [0, 1], method='tanh-sinh')  # Good
            -1.0
            >>> quad(log, [0, 1], method='gauss-legendre')  # Bad
            -0.999932197413801

        However, the result may still be inaccurate for some functions::

            >>> quad(lambda x: 1/sqrt(x), [0, 1], method='tanh-sinh')
            1.99999999946942

        This problem is not due to the quadrature rule per se, but to
        numerical amplification of errors in the nodes. The problem can be
        circumvented by temporarily increasing the precision::

            >>> mp.dps = 30
            >>> a = quad(lambda x: 1/sqrt(x), [0, 1], method='tanh-sinh')
            >>> mp.dps = 15
            >>> +a
            2.0

        **Highly variable functions**

        For functions that are smooth (in the sense of being infinitely
        differentiable) but contain sharp mid-interval peaks or many
        "bumps", :func:`~mpmath.quad` may fail to provide full accuracy. For
        example, with default settings, :func:`~mpmath.quad` is able to integrate
        `\sin(x)` accurately over an interval of length 100 but not over
        length 1000::

            >>> quad(sin, [0, 100]); 1-cos(100)   # Good
            0.137681127712316
            0.137681127712316
            >>> quad(sin, [0, 1000]); 1-cos(1000)   # Bad
            -37.8587612408485
            0.437620923709297

        One solution is to break the integration into 10 intervals of
        length 100::

            >>> quad(sin, linspace(0, 1000, 10))   # Good
            0.437620923709297

        Another is to increase the degree of the quadrature::

            >>> quad(sin, [0, 1000], maxdegree=10)   # Also good
            0.437620923709297

        Whether splitting the interval or increasing the degree is
        more efficient differs from case to case. Another example is the
        function `1/(1+x^2)`, which has a sharp peak centered around
        `x = 0`::

            >>> f = lambda x: 1/(1+x**2)
            >>> quad(f, [-100, 100])   # Bad
            3.64804647105268
            >>> quad(f, [-100, 100], maxdegree=10)   # Good
            3.12159332021646
            >>> quad(f, [-100, 0, 100])   # Also good
            3.12159332021646

        **References**

        1. http://mathworld.wolfram.com/DoubleIntegral.html

        """
        rule = kwargs.get('method', 'tanh-sinh')
        if type(rule) is str:
            if rule == 'tanh-sinh':
                rule = ctx._tanh_sinh
            elif rule == 'gauss-legendre':
                rule = ctx._gauss_legendre
            else:
                raise ValueError("unknown quadrature rule: %s" % rule)
        else:
            rule = rule(ctx)
        verbose = kwargs.get('verbose')
        dim = len(points)
        orig = prec = ctx.prec
        epsilon = ctx.eps/8
        m = kwargs.get('maxdegree') or rule.guess_degree(prec)
        points = [ctx._as_points(p) for p in points]
        try:
            ctx.prec += 20
            if dim == 1:
                v, err = rule.summation(f, points[0], prec, epsilon, m, verbose)
            elif dim == 2:
                v, err = rule.summation(lambda x: \
                        rule.summation(lambda y: f(x,y), \
                        points[1], prec, epsilon, m)[0],
                    points[0], prec, epsilon, m, verbose)
            elif dim == 3:
                v, err = rule.summation(lambda x: \
                        rule.summation(lambda y: \
                            rule.summation(lambda z: f(x,y,z), \
                            points[2], prec, epsilon, m)[0],
                        points[1], prec, epsilon, m)[0],
                    points[0], prec, epsilon, m, verbose)
            else:
                raise NotImplementedError("quadrature must have dim 1, 2 or 3")
        finally:
            ctx.prec = orig
        if kwargs.get("error"):
            return +v, err
        return +v

    def quadts(ctx, *args, **kwargs):
        """
        Performs tanh-sinh quadrature. The call

            quadts(func, *points, ...)

        is simply a shortcut for:

            quad(func, *points, ..., method=TanhSinh)

        For example, a single integral and a double integral:

            quadts(lambda x: exp(cos(x)), [0, 1])
            quadts(lambda x, y: exp(cos(x+y)), [0, 1], [0, 1])

        See the documentation for quad for information about how points
        arguments and keyword arguments are parsed.

        See documentation for TanhSinh for algorithmic information about
        tanh-sinh quadrature.
        """
        kwargs['method'] = 'tanh-sinh'
        return ctx.quad(*args, **kwargs)

    def quadgl(ctx, *args, **kwargs):
        """
        Performs Gauss-Legendre quadrature. The call

            quadgl(func, *points, ...)

        is simply a shortcut for:

            quad(func, *points, ..., method=GaussLegendre)

        For example, a single integral and a double integral:

            quadgl(lambda x: exp(cos(x)), [0, 1])
            quadgl(lambda x, y: exp(cos(x+y)), [0, 1], [0, 1])

        See the documentation for quad for information about how points
        arguments and keyword arguments are parsed.

        See documentation for TanhSinh for algorithmic information about
        tanh-sinh quadrature.
        """
        kwargs['method'] = 'gauss-legendre'
        return ctx.quad(*args, **kwargs)

    def quadosc(ctx, f, interval, omega=None, period=None, zeros=None):
        r"""
        Calculates

        .. math ::

            I = \int_a^b f(x) dx

        where at least one of `a` and `b` is infinite and where
        `f(x) = g(x) \cos(\omega x  + \phi)` for some slowly
        decreasing function `g(x)`. With proper input, :func:`~mpmath.quadosc`
        can also handle oscillatory integrals where the oscillation
        rate is different from a pure sine or cosine wave.

        In the standard case when `|a| < \infty, b = \infty`,
        :func:`~mpmath.quadosc` works by evaluating the infinite series

        .. math ::

            I = \int_a^{x_1} f(x) dx +
            \sum_{k=1}^{\infty} \int_{x_k}^{x_{k+1}} f(x) dx

        where `x_k` are consecutive zeros (alternatively
        some other periodic reference point) of `f(x)`.
        Accordingly, :func:`~mpmath.quadosc` requires information about the
        zeros of `f(x)`. For a periodic function, you can specify
        the zeros by either providing the angular frequency `\omega`
        (*omega*) or the *period* `2 \pi/\omega`. In general, you can
        specify the `n`-th zero by providing the *zeros* arguments.
        Below is an example of each::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> f = lambda x: sin(3*x)/(x**2+1)
            >>> quadosc(f, [0,inf], omega=3)
            0.37833007080198
            >>> quadosc(f, [0,inf], period=2*pi/3)
            0.37833007080198
            >>> quadosc(f, [0,inf], zeros=lambda n: pi*n/3)
            0.37833007080198
            >>> (ei(3)*exp(-3)-exp(3)*ei(-3))/2  # Computed by Mathematica
            0.37833007080198

        Note that *zeros* was specified to multiply `n` by the
        *half-period*, not the full period. In theory, it does not matter
        whether each partial integral is done over a half period or a full
        period. However, if done over half-periods, the infinite series
        passed to :func:`~mpmath.nsum` becomes an *alternating series* and this
        typically makes the extrapolation much more efficient.

        Here is an example of an integration over the entire real line,
        and a half-infinite integration starting at `-\infty`::

            >>> quadosc(lambda x: cos(x)/(1+x**2), [-inf, inf], omega=1)
            1.15572734979092
            >>> pi/e
            1.15572734979092
            >>> quadosc(lambda x: cos(x)/x**2, [-inf, -1], period=2*pi)
            -0.0844109505595739
            >>> cos(1)+si(1)-pi/2
            -0.0844109505595738

        Of course, the integrand may contain a complex exponential just as
        well as a real sine or cosine::

            >>> quadosc(lambda x: exp(3*j*x)/(1+x**2), [-inf,inf], omega=3)
            (0.156410688228254 + 0.0j)
            >>> pi/e**3
            0.156410688228254
            >>> quadosc(lambda x: exp(3*j*x)/(2+x+x**2), [-inf,inf], omega=3)
            (0.00317486988463794 - 0.0447701735209082j)
            >>> 2*pi/sqrt(7)/exp(3*(j+sqrt(7))/2)
            (0.00317486988463794 - 0.0447701735209082j)

        **Non-periodic functions**

        If `f(x) = g(x) h(x)` for some function `h(x)` that is not
        strictly periodic, *omega* or *period* might not work, and it might
        be necessary to use *zeros*.

        A notable exception can be made for Bessel functions which, though not
        periodic, are "asymptotically periodic" in a sufficiently strong sense
        that the sum extrapolation will work out::

            >>> quadosc(j0, [0, inf], period=2*pi)
            1.0
            >>> quadosc(j1, [0, inf], period=2*pi)
            1.0

        More properly, one should provide the exact Bessel function zeros::

            >>> j0zero = lambda n: findroot(j0, pi*(n-0.25))
            >>> quadosc(j0, [0, inf], zeros=j0zero)
            1.0

        For an example where *zeros* becomes necessary, consider the
        complete Fresnel integrals

        .. math ::

            \int_0^{\infty} \cos x^2\,dx = \int_0^{\infty} \sin x^2\,dx
            = \sqrt{\frac{\pi}{8}}.

        Although the integrands do not decrease in magnitude as
        `x \to \infty`, the integrals are convergent since the oscillation
        rate increases (causing consecutive periods to asymptotically
        cancel out). These integrals are virtually impossible to calculate
        to any kind of accuracy using standard quadrature rules. However,
        if one provides the correct asymptotic distribution of zeros
        (`x_n \sim \sqrt{n}`), :func:`~mpmath.quadosc` works::

            >>> mp.dps = 30
            >>> f = lambda x: cos(x**2)
            >>> quadosc(f, [0,inf], zeros=lambda n:sqrt(pi*n))
            0.626657068657750125603941321203
            >>> f = lambda x: sin(x**2)
            >>> quadosc(f, [0,inf], zeros=lambda n:sqrt(pi*n))
            0.626657068657750125603941321203
            >>> sqrt(pi/8)
            0.626657068657750125603941321203

        (Interestingly, these integrals can still be evaluated if one
        places some other constant than `\pi` in the square root sign.)

        In general, if `f(x) \sim g(x) \cos(h(x))`, the zeros follow
        the inverse-function distribution `h^{-1}(x)`::

            >>> mp.dps = 15
            >>> f = lambda x: sin(exp(x))
            >>> quadosc(f, [1,inf], zeros=lambda n: log(n))
            -0.25024394235267
            >>> pi/2-si(e)
            -0.250243942352671

        **Non-alternating functions**

        If the integrand oscillates around a positive value, without
        alternating signs, the extrapolation might fail. A simple trick
        that sometimes works is to multiply or divide the frequency by 2::

            >>> f = lambda x: 1/x**2+sin(x)/x**4
            >>> quadosc(f, [1,inf], omega=1)  # Bad
            1.28642190869861
            >>> quadosc(f, [1,inf], omega=0.5)  # Perfect
            1.28652953559617
            >>> 1+(cos(1)+ci(1)+sin(1))/6
            1.28652953559617

        **Fast decay**

        :func:`~mpmath.quadosc` is primarily useful for slowly decaying
        integrands. If the integrand decreases exponentially or faster,
        :func:`~mpmath.quad` will likely handle it without trouble (and generally be
        much faster than :func:`~mpmath.quadosc`)::

            >>> quadosc(lambda x: cos(x)/exp(x), [0, inf], omega=1)
            0.5
            >>> quad(lambda x: cos(x)/exp(x), [0, inf])
            0.5

        """
        a, b = ctx._as_points(interval)
        a = ctx.convert(a)
        b = ctx.convert(b)
        if [omega, period, zeros].count(None) != 2:
            raise ValueError( \
                "must specify exactly one of omega, period, zeros")
        if a == ctx.ninf and b == ctx.inf:
            s1 = ctx.quadosc(f, [a, 0], omega=omega, zeros=zeros, period=period)
            s2 = ctx.quadosc(f, [0, b], omega=omega, zeros=zeros, period=period)
            return s1 + s2
        if a == ctx.ninf:
            if zeros:
                return ctx.quadosc(lambda x:f(-x), [-b,-a], lambda n: zeros(-n))
            else:
                return ctx.quadosc(lambda x:f(-x), [-b,-a], omega=omega, period=period)
        if b != ctx.inf:
            raise ValueError("quadosc requires an infinite integration interval")
        if not zeros:
            if omega:
                period = 2*ctx.pi/omega
            zeros = lambda n: n*period/2
        #for n in range(1,10):
        #    p = zeros(n)
        #    if p > a:
        #        break
        #if n >= 9:
        #    raise ValueError("zeros do not appear to be correctly indexed")
        n = 1
        s = ctx.quadgl(f, [a, zeros(n)])
        def term(k):
            return ctx.quadgl(f, [zeros(k), zeros(k+1)])
        s += ctx.nsum(term, [n, ctx.inf])
        return s

    def quadsubdiv(ctx, f, interval, tol=None, maxintervals=None, **kwargs):
        """
        Computes the integral of *f* over the interval or path specified
        by *interval*, using :func:`~mpmath.quad` together with adaptive
        subdivision of the interval.

        This function gives an accurate answer for some integrals where
        :func:`~mpmath.quad` fails::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> quad(lambda x: abs(sin(x)), [0, 2*pi])
            3.99900894176779
            >>> quadsubdiv(lambda x: abs(sin(x)), [0, 2*pi])
            4.0
            >>> quadsubdiv(sin, [0, 1000])
            0.437620923709297
            >>> quadsubdiv(lambda x: 1/(1+x**2), [-100, 100])
            3.12159332021646
            >>> quadsubdiv(lambda x: ceil(x), [0, 100])
            5050.0
            >>> quadsubdiv(lambda x: sin(x+exp(x)), [0,8])
            0.347400172657248

        The argument *maxintervals* can be set to limit the permissible
        subdivision::

            >>> quadsubdiv(lambda x: sin(x**2), [0,100], maxintervals=5, error=True)
            (-5.40487904307774, 5.011)
            >>> quadsubdiv(lambda x: sin(x**2), [0,100], maxintervals=100, error=True)
            (0.631417921866934, 1.10101120134116e-17)

        Subdivision does not guarantee a correct answer since, the error
        estimate on subintervals may be inaccurate::

            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, [0,1], error=True)
            (0.210802735500549, 1.0001111101e-17)
            >>> mp.dps = 20
            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, [0,1], error=True)
            (0.21080273550054927738, 2.200000001e-24)

        The second answer is correct. We can get an accurate result at lower
        precision by forcing a finer initial subdivision::

            >>> mp.dps = 15
            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, linspace(0,1,5))
            0.210802735500549

        The following integral is too oscillatory for convergence, but we can get a
        reasonable estimate::

            >>> v, err = fp.quadsubdiv(lambda x: fp.sin(1/x), [0,1], error=True)
            >>> round(v, 6), round(err, 6)
            (0.504067, 1e-06)
            >>> sin(1) - ci(1)
            0.504067061906928

        """
        queue = []
        for i in range(len(interval)-1):
            queue.append((interval[i], interval[i+1]))
        total = ctx.zero
        total_error = ctx.zero
        if maxintervals is None:
            maxintervals = 10 * ctx.prec
        count = 0
        quad_args = kwargs.copy()
        quad_args["verbose"] = False
        quad_args["error"] = True
        if tol is None:
            tol = +ctx.eps
        orig = ctx.prec
        try:
            ctx.prec += 5
            while queue:
                a, b = queue.pop()
                s, err = ctx.quad(f, [a, b], **quad_args)
                if kwargs.get("verbose"):
                    print("subinterval", count, a, b, err)
                if err < tol or count > maxintervals:
                    total += s
                    total_error += err
                else:
                    count += 1
                    if count == maxintervals and kwargs.get("verbose"):
                        print("warning: number of intervals exceeded maxintervals")
                    if a == -ctx.inf and b == ctx.inf:
                        m = 0
                    elif a == -ctx.inf:
                        m = min(b-1, 2*b)
                    elif b == ctx.inf:
                        m = max(a+1, 2*a)
                    else:
                        m = a + (b - a) / 2
                    queue.append((a, m))
                    queue.append((m, b))
        finally:
            ctx.prec = orig
        if kwargs.get("error"):
            return +total, +total_error
        else:
            return +total

if __name__ == '__main__':
    import doctest
    doctest.testmod()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\bessel.py ===
from .functions import defun, defun_wrapped

@defun
def j0(ctx, x):
    """Computes the Bessel function `J_0(x)`. See :func:`~mpmath.besselj`."""
    return ctx.besselj(0, x)

@defun
def j1(ctx, x):
    """Computes the Bessel function `J_1(x)`.  See :func:`~mpmath.besselj`."""
    return ctx.besselj(1, x)

@defun
def besselj(ctx, n, z, derivative=0, **kwargs):
    if type(n) is int:
        n_isint = True
    else:
        n = ctx.convert(n)
        n_isint = ctx.isint(n)
        if n_isint:
            n = int(ctx._re(n))
    if n_isint and n < 0:
        return (-1)**n * ctx.besselj(-n, z, derivative, **kwargs)
    z = ctx.convert(z)
    M = ctx.mag(z)
    if derivative:
        d = ctx.convert(derivative)
        # TODO: the integer special-casing shouldn't be necessary.
        # However, the hypergeometric series gets inaccurate for large d
        # because of inaccurate pole cancellation at a pole far from
        # zero (needs to be fixed in hypercomb or hypsum)
        if ctx.isint(d) and d >= 0:
            d = int(d)
            orig = ctx.prec
            try:
                ctx.prec += 15
                v = ctx.fsum((-1)**k * ctx.binomial(d,k) * ctx.besselj(2*k+n-d,z)
                    for k in range(d+1))
            finally:
                ctx.prec = orig
            v *= ctx.mpf(2)**(-d)
        else:
            def h(n,d):
                r = ctx.fmul(ctx.fmul(z, z, prec=ctx.prec+M), -0.25, exact=True)
                B = [0.5*(n-d+1), 0.5*(n-d+2)]
                T = [([2,ctx.pi,z],[d-2*n,0.5,n-d],[],B,[(n+1)*0.5,(n+2)*0.5],B+[n+1],r)]
                return T
            v = ctx.hypercomb(h, [n,d], **kwargs)
    else:
        # Fast case: J_n(x), n int, appropriate magnitude for fixed-point calculation
        if (not derivative) and n_isint and abs(M) < 10 and abs(n) < 20:
            try:
                return ctx._besselj(n, z)
            except NotImplementedError:
                pass
        if not z:
            if not n:
                v = ctx.one + n+z
            elif ctx.re(n) > 0:
                v = n*z
            else:
                v = ctx.inf + z + n
        else:
            #v = 0
            orig = ctx.prec
            try:
                # XXX: workaround for accuracy in low level hypergeometric series
                # when alternating, large arguments
                ctx.prec += min(3*abs(M), ctx.prec)
                w = ctx.fmul(z, 0.5, exact=True)
                def h(n):
                    r = ctx.fneg(ctx.fmul(w, w, prec=max(0,ctx.prec+M)), exact=True)
                    return [([w], [n], [], [n+1], [], [n+1], r)]
                v = ctx.hypercomb(h, [n], **kwargs)
            finally:
                ctx.prec = orig
        v = +v
    return v

@defun
def besseli(ctx, n, z, derivative=0, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    if not z:
        if derivative:
            raise ValueError
        if not n:
            # I(0,0) = 1
            return 1+n+z
        if ctx.isint(n):
            return 0*(n+z)
        r = ctx.re(n)
        if r == 0:
            return ctx.nan*(n+z)
        elif r > 0:
            return 0*(n+z)
        else:
            return ctx.inf+(n+z)
    M = ctx.mag(z)
    if derivative:
        d = ctx.convert(derivative)
        def h(n,d):
            r = ctx.fmul(ctx.fmul(z, z, prec=ctx.prec+M), 0.25, exact=True)
            B = [0.5*(n-d+1), 0.5*(n-d+2), n+1]
            T = [([2,ctx.pi,z],[d-2*n,0.5,n-d],[n+1],B,[(n+1)*0.5,(n+2)*0.5],B,r)]
            return T
        v = ctx.hypercomb(h, [n,d], **kwargs)
    else:
        def h(n):
            w = ctx.fmul(z, 0.5, exact=True)
            r = ctx.fmul(w, w, prec=max(0,ctx.prec+M))
            return [([w], [n], [], [n+1], [], [n+1], r)]
        v = ctx.hypercomb(h, [n], **kwargs)
    return v

@defun_wrapped
def bessely(ctx, n, z, derivative=0, **kwargs):
    if not z:
        if derivative:
            # Not implemented
            raise ValueError
        if not n:
            # ~ log(z/2)
            return -ctx.inf + (n+z)
        if ctx.im(n):
            return ctx.nan * (n+z)
        r = ctx.re(n)
        q = n+0.5
        if ctx.isint(q):
            if n > 0:
                return -ctx.inf + (n+z)
            else:
                return 0 * (n+z)
        if r < 0 and int(ctx.floor(q)) % 2:
            return ctx.inf + (n+z)
        else:
            return ctx.ninf + (n+z)
    # XXX: use hypercomb
    ctx.prec += 10
    m, d = ctx.nint_distance(n)
    if d < -ctx.prec:
        h = +ctx.eps
        ctx.prec *= 2
        n += h
    elif d < 0:
        ctx.prec -= d
    # TODO: avoid cancellation for imaginary arguments
    cos, sin = ctx.cospi_sinpi(n)
    return (ctx.besselj(n,z,derivative,**kwargs)*cos - \
        ctx.besselj(-n,z,derivative,**kwargs))/sin

@defun_wrapped
def besselk(ctx, n, z, **kwargs):
    if not z:
        return ctx.inf
    M = ctx.mag(z)
    if M < 1:
        # Represent as limit definition
        def h(n):
            r = (z/2)**2
            T1 = [z, 2], [-n, n-1], [n], [], [], [1-n], r
            T2 = [z, 2], [n, -n-1], [-n], [], [], [1+n], r
            return T1, T2
    # We could use the limit definition always, but it leads
    # to very bad cancellation (of exponentially large terms)
    # for large real z
    # Instead represent in terms of 2F0
    else:
        ctx.prec += M
        def h(n):
            return [([ctx.pi/2, z, ctx.exp(-z)], [0.5,-0.5,1], [], [], \
                [n+0.5, 0.5-n], [], -1/(2*z))]
    return ctx.hypercomb(h, [n], **kwargs)

@defun_wrapped
def hankel1(ctx,n,x,**kwargs):
    return ctx.besselj(n,x,**kwargs) + ctx.j*ctx.bessely(n,x,**kwargs)

@defun_wrapped
def hankel2(ctx,n,x,**kwargs):
    return ctx.besselj(n,x,**kwargs) - ctx.j*ctx.bessely(n,x,**kwargs)

@defun_wrapped
def whitm(ctx,k,m,z,**kwargs):
    if z == 0:
        # M(k,m,z) = 0^(1/2+m)
        if ctx.re(m) > -0.5:
            return z
        elif ctx.re(m) < -0.5:
            return ctx.inf + z
        else:
            return ctx.nan * z
    x = ctx.fmul(-0.5, z, exact=True)
    y = 0.5+m
    return ctx.exp(x) * z**y * ctx.hyp1f1(y-k, 1+2*m, z, **kwargs)

@defun_wrapped
def whitw(ctx,k,m,z,**kwargs):
    if z == 0:
        g = abs(ctx.re(m))
        if g < 0.5:
            return z
        elif g > 0.5:
            return ctx.inf + z
        else:
            return ctx.nan * z
    x = ctx.fmul(-0.5, z, exact=True)
    y = 0.5+m
    return ctx.exp(x) * z**y * ctx.hyperu(y-k, 1+2*m, z, **kwargs)

@defun
def hyperu(ctx, a, b, z, **kwargs):
    a, atype = ctx._convert_param(a)
    b, btype = ctx._convert_param(b)
    z = ctx.convert(z)
    if not z:
        if ctx.re(b) <= 1:
            return ctx.gammaprod([1-b],[a-b+1])
        else:
            return ctx.inf + z
    bb = 1+a-b
    bb, bbtype = ctx._convert_param(bb)
    try:
        orig = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.hypsum(2, 0, (atype, bbtype), [a, bb], -1/z, maxterms=ctx.prec)
            return v / z**a
        finally:
            ctx.prec = orig
    except ctx.NoConvergence:
        pass
    def h(a,b):
        w = ctx.sinpi(b)
        T1 = ([ctx.pi,w],[1,-1],[],[a-b+1,b],[a],[b],z)
        T2 = ([-ctx.pi,w,z],[1,-1,1-b],[],[a,2-b],[a-b+1],[2-b],z)
        return T1, T2
    return ctx.hypercomb(h, [a,b], **kwargs)

@defun
def struveh(ctx,n,z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/StruveH/26/01/02/
    def h(n):
        return [([z/2, 0.5*ctx.sqrt(ctx.pi)], [n+1, -1], [], [n+1.5], [1], [1.5, n+1.5], -(z/2)**2)]
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def struvel(ctx,n,z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/StruveL/26/01/02/
    def h(n):
        return [([z/2, 0.5*ctx.sqrt(ctx.pi)], [n+1, -1], [], [n+1.5], [1], [1.5, n+1.5], (z/2)**2)]
    return ctx.hypercomb(h, [n], **kwargs)

def _anger(ctx,which,v,z,**kwargs):
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    def h(v):
        b = ctx.mpq_1_2
        u = v*b
        m = b*3
        a1,a2,b1,b2 = m-u, m+u, 1-u, 1+u
        c, s = ctx.cospi_sinpi(u)
        if which == 0:
            A, B = [b*z, s], [c]
        if which == 1:
            A, B = [b*z, -c], [s]
        w = ctx.square_exp_arg(z, mult=-0.25)
        T1 = A, [1, 1], [], [a1,a2], [1], [a1,a2], w
        T2 = B, [1], [], [b1,b2], [1], [b1,b2], w
        return T1, T2
    return ctx.hypercomb(h, [v], **kwargs)

@defun
def angerj(ctx, v, z, **kwargs):
    return _anger(ctx, 0, v, z, **kwargs)

@defun
def webere(ctx, v, z, **kwargs):
    return _anger(ctx, 1, v, z, **kwargs)

@defun
def lommels1(ctx, u, v, z, **kwargs):
    u = ctx._convert_param(u)[0]
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    def h(u,v):
        b = ctx.mpq_1_2
        w = ctx.square_exp_arg(z, mult=-0.25)
        return ([u-v+1, u+v+1, z], [-1, -1, u+1], [], [], [1], \
            [b*(u-v+3),b*(u+v+3)], w),
    return ctx.hypercomb(h, [u,v], **kwargs)

@defun
def lommels2(ctx, u, v, z, **kwargs):
    u = ctx._convert_param(u)[0]
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)
    # Asymptotic expansion (GR p. 947) -- need to be careful
    # not to use for small arguments
    # def h(u,v):
    #    b = ctx.mpq_1_2
    #    w = -(z/2)**(-2)
    #    return ([z], [u-1], [], [], [b*(1-u+v)], [b*(1-u-v)], w),
    def h(u,v):
        b = ctx.mpq_1_2
        w = ctx.square_exp_arg(z, mult=-0.25)
        T1 = [u-v+1, u+v+1, z], [-1, -1, u+1], [], [], [1], [b*(u-v+3),b*(u+v+3)], w
        T2 = [2, z], [u+v-1, -v], [v, b*(u+v+1)], [b*(v-u+1)], [], [1-v], w
        T3 = [2, z], [u-v-1, v], [-v, b*(u-v+1)], [b*(1-u-v)], [], [1+v], w
        #c1 = ctx.cospi((u-v)*b)
        #c2 = ctx.cospi((u+v)*b)
        #s = ctx.sinpi(v)
        #r1 = (u-v+1)*b
        #r2 = (u+v+1)*b
        #T2 = [c1, s, z, 2], [1, -1, -v, v], [], [-v+1], [], [-v+1], w
        #T3 = [-c2, s, z, 2], [1, -1, v, -v], [], [v+1], [], [v+1], w
        #T2 = [c1, s, z, 2], [1, -1, -v, v+u-1], [r1, r2], [-v+1], [], [-v+1], w
        #T3 = [-c2, s, z, 2], [1, -1, v, -v+u-1], [r1, r2], [v+1], [], [v+1], w
        return T1, T2, T3
    return ctx.hypercomb(h, [u,v], **kwargs)

@defun
def ber(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinBer2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos, sin = ctx.cospi_sinpi(-0.75*n)
        T1 = [cos, z/2], [1, n], [], [n+1], [], [0.5, 0.5*(n+1), 0.5*n+1], r
        T2 = [sin, z/2], [1, n+2], [], [n+2], [], [1.5, 0.5*(n+3), 0.5*n+1], r
        return T1, T2
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def bei(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinBei2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos, sin = ctx.cospi_sinpi(0.75*n)
        T1 = [cos, z/2], [1, n+2], [], [n+2], [], [1.5, 0.5*(n+3), 0.5*n+1], r
        T2 = [sin, z/2], [1, n], [], [n+1], [], [0.5, 0.5*(n+1), 0.5*n+1], r
        return T1, T2
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def ker(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinKer2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos1, sin1 = ctx.cospi_sinpi(0.25*n)
        cos2, sin2 = ctx.cospi_sinpi(0.75*n)
        T1 = [2, z, 4*cos1], [-n-3, n, 1], [-n], [], [], [0.5, 0.5*(1+n), 0.5*(n+2)], r
        T2 = [2, z, -sin1], [-n-3, 2+n, 1], [-n-1], [], [], [1.5, 0.5*(3+n), 0.5*(n+2)], r
        T3 = [2, z, 4*cos2], [n-3, -n, 1], [n], [], [], [0.5, 0.5*(1-n), 1-0.5*n], r
        T4 = [2, z, -sin2], [n-3, 2-n, 1], [n-1], [], [], [1.5, 0.5*(3-n), 1-0.5*n], r
        return T1, T2, T3, T4
    return ctx.hypercomb(h, [n], **kwargs)

@defun
def kei(ctx, n, z, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    # http://functions.wolfram.com/Bessel-TypeFunctions/KelvinKei2/26/01/02/0001/
    def h(n):
        r = -(z/4)**4
        cos1, sin1 = ctx.cospi_sinpi(0.75*n)
        cos2, sin2 = ctx.cospi_sinpi(0.25*n)
        T1 = [-cos1, 2, z], [1, n-3, 2-n], [n-1], [], [], [1.5, 0.5*(3-n), 1-0.5*n], r
        T2 = [-sin1, 2, z], [1, n-1, -n], [n], [], [], [0.5, 0.5*(1-n), 1-0.5*n], r
        T3 = [-sin2, 2, z], [1, -n-1, n], [-n], [], [], [0.5, 0.5*(n+1), 0.5*(n+2)], r
        T4 = [-cos2, 2, z], [1, -n-3, n+2], [-n-1], [], [], [1.5, 0.5*(n+3), 0.5*(n+2)], r
        return T1, T2, T3, T4
    return ctx.hypercomb(h, [n], **kwargs)

# TODO: do this more generically?
def c_memo(f):
    name = f.__name__
    def f_wrapped(ctx):
        cache = ctx._misc_const_cache
        prec = ctx.prec
        p,v = cache.get(name, (-1,0))
        if p >= prec:
            return +v
        else:
            cache[name] = (prec, f(ctx))
            return cache[name][1]
    return f_wrapped

@c_memo
def _airyai_C1(ctx):
    return 1 / (ctx.cbrt(9) * ctx.gamma(ctx.mpf(2)/3))

@c_memo
def _airyai_C2(ctx):
    return -1 / (ctx.cbrt(3) * ctx.gamma(ctx.mpf(1)/3))

@c_memo
def _airybi_C1(ctx):
    return 1 / (ctx.nthroot(3,6) * ctx.gamma(ctx.mpf(2)/3))

@c_memo
def _airybi_C2(ctx):
    return ctx.nthroot(3,6) / ctx.gamma(ctx.mpf(1)/3)

def _airybi_n2_inf(ctx):
    prec = ctx.prec
    try:
        v = ctx.power(3,'2/3')*ctx.gamma('2/3')/(2*ctx.pi)
    finally:
        ctx.prec = prec
    return +v

# Derivatives at z = 0
# TODO: could be expressed more elegantly using triple factorials
def _airyderiv_0(ctx, z, n, ntype, which):
    if ntype == 'Z':
        if n < 0:
            return z
        r = ctx.mpq_1_3
        prec = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.gamma((n+1)*r) * ctx.power(3,n*r) / ctx.pi
            if which == 0:
                v *= ctx.sinpi(2*(n+1)*r)
                v /= ctx.power(3,'2/3')
            else:
                v *= abs(ctx.sinpi(2*(n+1)*r))
                v /= ctx.power(3,'1/6')
        finally:
            ctx.prec = prec
        return +v + z
    else:
        # singular (does the limit exist?)
        raise NotImplementedError

@defun
def airyai(ctx, z, derivative=0, **kwargs):
    z = ctx.convert(z)
    if derivative:
        n, ntype = ctx._convert_param(derivative)
    else:
        n = 0
    # Values at infinities
    if not ctx.isnormal(z) and z:
        if n and ntype == 'Z':
            if n == -1:
                if z == ctx.inf:
                    return ctx.mpf(1)/3 + 1/z
                if z == ctx.ninf:
                    return ctx.mpf(-2)/3 + 1/z
            if n < -1:
                if z == ctx.inf:
                    return z
                if z == ctx.ninf:
                    return (-1)**n * (-z)
        if (not n) and z == ctx.inf or z == ctx.ninf:
            return 1/z
        # TODO: limits
        raise ValueError("essential singularity of Ai(z)")
    # Account for exponential scaling
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if n:
        if n == 1:
            def h():
                # http://functions.wolfram.com/03.07.06.0005.01
                if ctx._re(z) > 4:
                    ctx.prec += extraprec
                    w = z**1.5; r = -0.75/w; u = -2*w/3
                    ctx.prec -= extraprec
                    C = -ctx.exp(u)/(2*ctx.sqrt(ctx.pi))*ctx.nthroot(z,4)
                    return ([C],[1],[],[],[(-1,6),(7,6)],[],r),
                # http://functions.wolfram.com/03.07.26.0001.01
                else:
                    ctx.prec += extraprec
                    w = z**3 / 9
                    ctx.prec -= extraprec
                    C1 = _airyai_C1(ctx) * 0.5
                    C2 = _airyai_C2(ctx)
                    T1 = [C1,z],[1,2],[],[],[],[ctx.mpq_5_3],w
                    T2 = [C2],[1],[],[],[],[ctx.mpq_1_3],w
                    return T1, T2
            return ctx.hypercomb(h, [], **kwargs)
        else:
            if z == 0:
                return _airyderiv_0(ctx, z, n, ntype, 0)
            # http://functions.wolfram.com/03.05.20.0004.01
            def h(n):
                ctx.prec += extraprec
                w = z**3/9
                ctx.prec -= extraprec
                q13,q23,q43 = ctx.mpq_1_3, ctx.mpq_2_3, ctx.mpq_4_3
                a1=q13; a2=1; b1=(1-n)*q13; b2=(2-n)*q13; b3=1-n*q13
                T1 = [3, z], [n-q23, -n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                a1=q23; b1=(2-n)*q13; b2=1-n*q13; b3=(4-n)*q13
                T2 = [3, z, -z], [n-q43, -n, 1], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                return T1, T2
            v = ctx.hypercomb(h, [n], **kwargs)
            if ctx._is_real_type(z) and ctx.isint(n):
                v = ctx._re(v)
            return v
    else:
        def h():
            if ctx._re(z) > 4:
                # We could use 1F1, but it results in huge cancellation;
                # the following expansion is better.
                # TODO: asymptotic series for derivatives
                ctx.prec += extraprec
                w = z**1.5; r = -0.75/w; u = -2*w/3
                ctx.prec -= extraprec
                C = ctx.exp(u)/(2*ctx.sqrt(ctx.pi)*ctx.nthroot(z,4))
                return ([C],[1],[],[],[(1,6),(5,6)],[],r),
            else:
                ctx.prec += extraprec
                w = z**3 / 9
                ctx.prec -= extraprec
                C1 = _airyai_C1(ctx)
                C2 = _airyai_C2(ctx)
                T1 = [C1],[1],[],[],[],[ctx.mpq_2_3],w
                T2 = [z*C2],[1],[],[],[],[ctx.mpq_4_3],w
                return T1, T2
        return ctx.hypercomb(h, [], **kwargs)

@defun
def airybi(ctx, z, derivative=0, **kwargs):
    z = ctx.convert(z)
    if derivative:
        n, ntype = ctx._convert_param(derivative)
    else:
        n = 0
    # Values at infinities
    if not ctx.isnormal(z) and z:
        if n and ntype == 'Z':
            if z == ctx.inf:
                return z
            if z == ctx.ninf:
                if n == -1:
                    return 1/z
                if n == -2:
                    return _airybi_n2_inf(ctx)
                if n < -2:
                    return (-1)**n * (-z)
        if not n:
            if z == ctx.inf:
                return z
            if z == ctx.ninf:
                return 1/z
        # TODO: limits
        raise ValueError("essential singularity of Bi(z)")
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if n:
        if n == 1:
            # http://functions.wolfram.com/03.08.26.0001.01
            def h():
                ctx.prec += extraprec
                w = z**3 / 9
                ctx.prec -= extraprec
                C1 = _airybi_C1(ctx)*0.5
                C2 = _airybi_C2(ctx)
                T1 = [C1,z],[1,2],[],[],[],[ctx.mpq_5_3],w
                T2 = [C2],[1],[],[],[],[ctx.mpq_1_3],w
                return T1, T2
            return ctx.hypercomb(h, [], **kwargs)
        else:
            if z == 0:
                return _airyderiv_0(ctx, z, n, ntype, 1)
            def h(n):
                ctx.prec += extraprec
                w = z**3/9
                ctx.prec -= extraprec
                q13,q23,q43 = ctx.mpq_1_3, ctx.mpq_2_3, ctx.mpq_4_3
                q16 = ctx.mpq_1_6
                q56 = ctx.mpq_5_6
                a1=q13; a2=1; b1=(1-n)*q13; b2=(2-n)*q13; b3=1-n*q13
                T1 = [3, z], [n-q16, -n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                a1=q23; b1=(2-n)*q13; b2=1-n*q13; b3=(4-n)*q13
                T2 = [3, z], [n-q56, 1-n], [a1], [b1,b2,b3], \
                    [a1,a2], [b1,b2,b3], w
                return T1, T2
            v = ctx.hypercomb(h, [n], **kwargs)
            if ctx._is_real_type(z) and ctx.isint(n):
                v = ctx._re(v)
            return v
    else:
        def h():
            ctx.prec += extraprec
            w = z**3 / 9
            ctx.prec -= extraprec
            C1 = _airybi_C1(ctx)
            C2 = _airybi_C2(ctx)
            T1 = [C1],[1],[],[],[],[ctx.mpq_2_3],w
            T2 = [z*C2],[1],[],[],[],[ctx.mpq_4_3],w
            return T1, T2
        return ctx.hypercomb(h, [], **kwargs)

def _airy_zero(ctx, which, k, derivative, complex=False):
    # Asymptotic formulas are given in DLMF section 9.9
    def U(t): return t**(2/3.)*(1-7/(t**2*48))
    def T(t): return t**(2/3.)*(1+5/(t**2*48))
    k = int(k)
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if not derivative in (0,1):
        raise ValueError("Derivative should lie between 0 and 1")
    if which == 0:
        if derivative:
            return ctx.findroot(lambda z: ctx.airyai(z,1),
                -U(3*ctx.pi*(4*k-3)/8))
        return ctx.findroot(ctx.airyai, -T(3*ctx.pi*(4*k-1)/8))
    if which == 1 and complex == False:
        if derivative:
            return ctx.findroot(lambda z: ctx.airybi(z,1),
                -U(3*ctx.pi*(4*k-1)/8))
        return ctx.findroot(ctx.airybi, -T(3*ctx.pi*(4*k-3)/8))
    if which == 1 and complex == True:
        if derivative:
            t = 3*ctx.pi*(4*k-3)/8 + 0.75j*ctx.ln2
            s = ctx.expjpi(ctx.mpf(1)/3) * T(t)
            return ctx.findroot(lambda z: ctx.airybi(z,1), s)
        t = 3*ctx.pi*(4*k-1)/8 + 0.75j*ctx.ln2
        s = ctx.expjpi(ctx.mpf(1)/3) * U(t)
        return ctx.findroot(ctx.airybi, s)

@defun
def airyaizero(ctx, k, derivative=0):
    return _airy_zero(ctx, 0, k, derivative, False)

@defun
def airybizero(ctx, k, derivative=0, complex=False):
    return _airy_zero(ctx, 1, k, derivative, complex)

def _scorer(ctx, z, which, kwargs):
    z = ctx.convert(z)
    if ctx.isinf(z):
        if z == ctx.inf:
            if which == 0: return 1/z
            if which == 1: return z
        if z == ctx.ninf:
            return 1/z
        raise ValueError("essential singularity")
    if z:
        extraprec = max(0, int(1.5*ctx.mag(z)))
    else:
        extraprec = 0
    if kwargs.get('derivative'):
        raise NotImplementedError
    # Direct asymptotic expansions, to avoid
    # exponentially large cancellation
    try:
        if ctx.mag(z) > 3:
            if which == 0 and abs(ctx.arg(z)) < ctx.pi/3 * 0.999:
                def h():
                    return (([ctx.pi,z],[-1,-1],[],[],[(1,3),(2,3),1],[],9/z**3),)
                return ctx.hypercomb(h, [], maxterms=ctx.prec, force_series=True)
            if which == 1 and abs(ctx.arg(-z)) < 2*ctx.pi/3 * 0.999:
                def h():
                    return (([-ctx.pi,z],[-1,-1],[],[],[(1,3),(2,3),1],[],9/z**3),)
                return ctx.hypercomb(h, [], maxterms=ctx.prec, force_series=True)
    except ctx.NoConvergence:
        pass
    def h():
        A = ctx.airybi(z, **kwargs)/3
        B = -2*ctx.pi
        if which == 1:
            A *= 2
            B *= -1
        ctx.prec += extraprec
        w = z**3/9
        ctx.prec -= extraprec
        T1 = [A], [1], [], [], [], [], 0
        T2 = [B,z], [-1,2], [], [], [1], [ctx.mpq_4_3,ctx.mpq_5_3], w
        return T1, T2
    return ctx.hypercomb(h, [], **kwargs)

@defun
def scorergi(ctx, z, **kwargs):
    return _scorer(ctx, z, 0, kwargs)

@defun
def scorerhi(ctx, z, **kwargs):
    return _scorer(ctx, z, 1, kwargs)

@defun_wrapped
def coulombc(ctx, l, eta, _cache={}):
    if (l, eta) in _cache and _cache[l,eta][0] >= ctx.prec:
        return +_cache[l,eta][1]
    G3 = ctx.loggamma(2*l+2)
    G1 = ctx.loggamma(1+l+ctx.j*eta)
    G2 = ctx.loggamma(1+l-ctx.j*eta)
    v = 2**l * ctx.exp((-ctx.pi*eta+G1+G2)/2 - G3)
    if not (ctx.im(l) or ctx.im(eta)):
        v = ctx.re(v)
    _cache[l,eta] = (ctx.prec, v)
    return v

@defun_wrapped
def coulombf(ctx, l, eta, z, w=1, chop=True, **kwargs):
    # Regular Coulomb wave function
    # Note: w can be either 1 or -1; the other may be better in some cases
    # TODO: check that chop=True chops when and only when it should
    #ctx.prec += 10
    def h(l, eta):
        try:
            jw = ctx.j*w
            jwz = ctx.fmul(jw, z, exact=True)
            jwz2 = ctx.fmul(jwz, -2, exact=True)
            C = ctx.coulombc(l, eta)
            T1 = [C, z, ctx.exp(jwz)], [1, l+1, 1], [], [], [1+l+jw*eta], \
                [2*l+2], jwz2
        except ValueError:
            T1 = [0], [-1], [], [], [], [], 0
        return (T1,)
    v = ctx.hypercomb(h, [l,eta], **kwargs)
    if chop and (not ctx.im(l)) and (not ctx.im(eta)) and (not ctx.im(z)) and \
        (ctx.re(z) >= 0):
        v = ctx.re(v)
    return v

@defun_wrapped
def _coulomb_chi(ctx, l, eta, _cache={}):
    if (l, eta) in _cache and _cache[l,eta][0] >= ctx.prec:
        return _cache[l,eta][1]
    def terms():
        l2 = -l-1
        jeta = ctx.j*eta
        return [ctx.loggamma(1+l+jeta) * (-0.5j),
            ctx.loggamma(1+l-jeta) * (0.5j),
            ctx.loggamma(1+l2+jeta) * (0.5j),
            ctx.loggamma(1+l2-jeta) * (-0.5j),
            -(l+0.5)*ctx.pi]
    v = ctx.sum_accurately(terms, 1)
    _cache[l,eta] = (ctx.prec, v)
    return v

@defun_wrapped
def coulombg(ctx, l, eta, z, w=1, chop=True, **kwargs):
    # Irregular Coulomb wave function
    # Note: w can be either 1 or -1; the other may be better in some cases
    # TODO: check that chop=True chops when and only when it should
    if not ctx._im(l):
        l = ctx._re(l)  # XXX: for isint
    def h(l, eta):
        # Force perturbation for integers and half-integers
        if ctx.isint(l*2):
            T1 = [0], [-1], [], [], [], [], 0
            return (T1,)
        l2 = -l-1
        try:
            chi = ctx._coulomb_chi(l, eta)
            jw = ctx.j*w
            s = ctx.sin(chi); c = ctx.cos(chi)
            C1 = ctx.coulombc(l,eta)
            C2 = ctx.coulombc(l2,eta)
            u = ctx.exp(jw*z)
            x = -2*jw*z
            T1 = [s, C1, z, u, c], [-1, 1, l+1, 1, 1], [], [], \
                [1+l+jw*eta], [2*l+2], x
            T2 = [-s, C2, z, u],   [-1, 1, l2+1, 1],    [], [], \
                [1+l2+jw*eta], [2*l2+2], x
            return T1, T2
        except ValueError:
            T1 = [0], [-1], [], [], [], [], 0
            return (T1,)
    v = ctx.hypercomb(h, [l,eta], **kwargs)
    if chop and (not ctx._im(l)) and (not ctx._im(eta)) and (not ctx._im(z)) and \
        (ctx._re(z) >= 0):
        v = ctx._re(v)
    return v

def mcmahon(ctx,kind,prime,v,m):
    """
    Computes an estimate for the location of the Bessel function zero
    j_{v,m}, y_{v,m}, j'_{v,m} or y'_{v,m} using McMahon's asymptotic
    expansion (Abramowitz & Stegun 9.5.12-13, DLMF 20.21(vi)).

    Returns (r,err) where r is the estimated location of the root
    and err is a positive number estimating the error of the
    asymptotic expansion.
    """
    u = 4*v**2
    if kind == 1 and not prime: b = (4*m+2*v-1)*ctx.pi/4
    if kind == 2 and not prime: b = (4*m+2*v-3)*ctx.pi/4
    if kind == 1 and prime: b = (4*m+2*v-3)*ctx.pi/4
    if kind == 2 and prime: b = (4*m+2*v-1)*ctx.pi/4
    if not prime:
        s1 = b
        s2 = -(u-1)/(8*b)
        s3 = -4*(u-1)*(7*u-31)/(3*(8*b)**3)
        s4 = -32*(u-1)*(83*u**2-982*u+3779)/(15*(8*b)**5)
        s5 = -64*(u-1)*(6949*u**3-153855*u**2+1585743*u-6277237)/(105*(8*b)**7)
    if prime:
        s1 = b
        s2 = -(u+3)/(8*b)
        s3 = -4*(7*u**2+82*u-9)/(3*(8*b)**3)
        s4 = -32*(83*u**3+2075*u**2-3039*u+3537)/(15*(8*b)**5)
        s5 = -64*(6949*u**4+296492*u**3-1248002*u**2+7414380*u-5853627)/(105*(8*b)**7)
    terms = [s1,s2,s3,s4,s5]
    s = s1
    err = 0.0
    for i in range(1,len(terms)):
        if abs(terms[i]) < abs(terms[i-1]):
            s += terms[i]
        else:
            err = abs(terms[i])
    if i == len(terms)-1:
        err = abs(terms[-1])
    return s, err

def generalized_bisection(ctx,f,a,b,n):
    """
    Given f known to have exactly n simple roots within [a,b],
    return a list of n intervals isolating the roots
    and having opposite signs at the endpoints.

    TODO: this can be optimized, e.g. by reusing evaluation points.
    """
    if n < 1:
        raise ValueError("n cannot be less than 1")
    N = n+1
    points = []
    signs = []
    while 1:
        points = ctx.linspace(a,b,N)
        signs = [ctx.sign(f(x)) for x in points]
        ok_intervals = [(points[i],points[i+1]) for i in range(N-1) \
            if signs[i]*signs[i+1] == -1]
        if len(ok_intervals) == n:
            return ok_intervals
        N = N*2

def find_in_interval(ctx, f, ab):
    return ctx.findroot(f, ab, solver='illinois', verify=False)

def bessel_zero(ctx, kind, prime, v, m, isoltol=0.01, _interval_cache={}):
    prec = ctx.prec
    workprec = max(prec, ctx.mag(v), ctx.mag(m))+10
    try:
        ctx.prec = workprec
        v = ctx.mpf(v)
        m = int(m)
        prime = int(prime)
        if v < 0:
            raise ValueError("v cannot be negative")
        if m < 1:
            raise ValueError("m cannot be less than 1")
        if not prime in (0,1):
            raise ValueError("prime should lie between 0 and 1")
        if kind == 1:
            if prime: f = lambda x: ctx.besselj(v,x,derivative=1)
            else:     f = lambda x: ctx.besselj(v,x)
        if kind == 2:
            if prime: f = lambda x: ctx.bessely(v,x,derivative=1)
            else:     f = lambda x: ctx.bessely(v,x)
        # The first root of J' is very close to 0 for small
        # orders, and this needs to be special-cased
        if kind == 1 and prime and m == 1:
            if v == 0:
                return ctx.zero
            if v <= 1:
                # TODO: use v <= j'_{v,1} < y_{v,1}?
                r = 2*ctx.sqrt(v*(1+v)/(v+2))
                return find_in_interval(ctx, f, (r/10, 2*r))
        if (kind,prime,v,m) in _interval_cache:
            return find_in_interval(ctx, f, _interval_cache[kind,prime,v,m])
        r, err = mcmahon(ctx, kind, prime, v, m)
        if err < isoltol:
            return find_in_interval(ctx, f, (r-isoltol, r+isoltol))
        # An x such that 0 < x < r_{v,1}
        if kind == 1 and not prime: low = 2.4
        if kind == 1 and prime: low = 1.8
        if kind == 2 and not prime: low = 0.8
        if kind == 2 and prime: low = 2.0
        n = m+1
        while 1:
            r1, err = mcmahon(ctx, kind, prime, v, n)
            if err < isoltol:
                r2, err2 = mcmahon(ctx, kind, prime, v, n+1)
                intervals = generalized_bisection(ctx, f, low, 0.5*(r1+r2), n)
                for k, ab in enumerate(intervals):
                    _interval_cache[kind,prime,v,k+1] = ab
                return find_in_interval(ctx, f, intervals[m-1])
            else:
                n = n*2
    finally:
        ctx.prec = prec

@defun
def besseljzero(ctx, v, m, derivative=0):
    r"""
    For a real order `\nu \ge 0` and a positive integer `m`, returns
    `j_{\nu,m}`, the `m`-th positive zero of the Bessel function of the
    first kind `J_{\nu}(z)` (see :func:`~mpmath.besselj`). Alternatively,
    with *derivative=1*, gives the first nonnegative simple zero
    `j'_{\nu,m}` of `J'_{\nu}(z)`.

    The indexing convention is that used by Abramowitz & Stegun
    and the DLMF. Note the special case `j'_{0,1} = 0`, while all other
    zeros are positive. In effect, only simple zeros are counted
    (all zeros of Bessel functions are simple except possibly `z = 0`)
    and `j_{\nu,m}` becomes a monotonic function of both `\nu`
    and `m`.

    The zeros are interlaced according to the inequalities

    .. math ::

        j'_{\nu,k} < j_{\nu,k} < j'_{\nu,k+1}

        j_{\nu,1} < j_{\nu+1,2} < j_{\nu,2} < j_{\nu+1,2} < j_{\nu,3} < \cdots

    **Examples**

    Initial zeros of the Bessel functions `J_0(z), J_1(z), J_2(z)`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> besseljzero(0,1); besseljzero(0,2); besseljzero(0,3)
        2.404825557695772768621632
        5.520078110286310649596604
        8.653727912911012216954199
        >>> besseljzero(1,1); besseljzero(1,2); besseljzero(1,3)
        3.831705970207512315614436
        7.01558666981561875353705
        10.17346813506272207718571
        >>> besseljzero(2,1); besseljzero(2,2); besseljzero(2,3)
        5.135622301840682556301402
        8.417244140399864857783614
        11.61984117214905942709415

    Initial zeros of `J'_0(z), J'_1(z), J'_2(z)`::

        0.0
        3.831705970207512315614436
        7.01558666981561875353705
        >>> besseljzero(1,1,1); besseljzero(1,2,1); besseljzero(1,3,1)
        1.84118378134065930264363
        5.331442773525032636884016
        8.536316366346285834358961
        >>> besseljzero(2,1,1); besseljzero(2,2,1); besseljzero(2,3,1)
        3.054236928227140322755932
        6.706133194158459146634394
        9.969467823087595793179143

    Zeros with large index::

        >>> besseljzero(0,100); besseljzero(0,1000); besseljzero(0,10000)
        313.3742660775278447196902
        3140.807295225078628895545
        31415.14114171350798533666
        >>> besseljzero(5,100); besseljzero(5,1000); besseljzero(5,10000)
        321.1893195676003157339222
        3148.657306813047523500494
        31422.9947255486291798943
        >>> besseljzero(0,100,1); besseljzero(0,1000,1); besseljzero(0,10000,1)
        311.8018681873704508125112
        3139.236339643802482833973
        31413.57032947022399485808

    Zeros of functions with large order::

        >>> besseljzero(50,1)
        57.11689916011917411936228
        >>> besseljzero(50,2)
        62.80769876483536093435393
        >>> besseljzero(50,100)
        388.6936600656058834640981
        >>> besseljzero(50,1,1)
        52.99764038731665010944037
        >>> besseljzero(50,2,1)
        60.02631933279942589882363
        >>> besseljzero(50,100,1)
        387.1083151608726181086283

    Zeros of functions with fractional order::

        >>> besseljzero(0.5,1); besseljzero(1.5,1); besseljzero(2.25,4)
        3.141592653589793238462643
        4.493409457909064175307881
        15.15657692957458622921634

    Both `J_{\nu}(z)` and `J'_{\nu}(z)` can be expressed as infinite
    products over their zeros::

        >>> v,z = 2, mpf(1)
        >>> (z/2)**v/gamma(v+1) * \
        ...     nprod(lambda k: 1-(z/besseljzero(v,k))**2, [1,inf])
        ...
        0.1149034849319004804696469
        >>> besselj(v,z)
        0.1149034849319004804696469
        >>> (z/2)**(v-1)/2/gamma(v) * \
        ...     nprod(lambda k: 1-(z/besseljzero(v,k,1))**2, [1,inf])
        ...
        0.2102436158811325550203884
        >>> besselj(v,z,1)
        0.2102436158811325550203884

    """
    return +bessel_zero(ctx, 1, derivative, v, m)

@defun
def besselyzero(ctx, v, m, derivative=0):
    r"""
    For a real order `\nu \ge 0` and a positive integer `m`, returns
    `y_{\nu,m}`, the `m`-th positive zero of the Bessel function of the
    second kind `Y_{\nu}(z)` (see :func:`~mpmath.bessely`). Alternatively,
    with *derivative=1*, gives the first positive zero `y'_{\nu,m}` of
    `Y'_{\nu}(z)`.

    The zeros are interlaced according to the inequalities

    .. math ::

        y_{\nu,k} < y'_{\nu,k} < y_{\nu,k+1}

        y_{\nu,1} < y_{\nu+1,2} < y_{\nu,2} < y_{\nu+1,2} < y_{\nu,3} < \cdots

    **Examples**

    Initial zeros of the Bessel functions `Y_0(z), Y_1(z), Y_2(z)`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> besselyzero(0,1); besselyzero(0,2); besselyzero(0,3)
        0.8935769662791675215848871
        3.957678419314857868375677
        7.086051060301772697623625
        >>> besselyzero(1,1); besselyzero(1,2); besselyzero(1,3)
        2.197141326031017035149034
        5.429681040794135132772005
        8.596005868331168926429606
        >>> besselyzero(2,1); besselyzero(2,2); besselyzero(2,3)
        3.384241767149593472701426
        6.793807513268267538291167
        10.02347797936003797850539

    Initial zeros of `Y'_0(z), Y'_1(z), Y'_2(z)`::

        >>> besselyzero(0,1,1); besselyzero(0,2,1); besselyzero(0,3,1)
        2.197141326031017035149034
        5.429681040794135132772005
        8.596005868331168926429606
        >>> besselyzero(1,1,1); besselyzero(1,2,1); besselyzero(1,3,1)
        3.683022856585177699898967
        6.941499953654175655751944
        10.12340465543661307978775
        >>> besselyzero(2,1,1); besselyzero(2,2,1); besselyzero(2,3,1)
        5.002582931446063945200176
        8.350724701413079526349714
        11.57419546521764654624265

    Zeros with large index::

        >>> besselyzero(0,100); besselyzero(0,1000); besselyzero(0,10000)
        311.8034717601871549333419
        3139.236498918198006794026
        31413.57034538691205229188
        >>> besselyzero(5,100); besselyzero(5,1000); besselyzero(5,10000)
        319.6183338562782156235062
        3147.086508524556404473186
        31421.42392920214673402828
        >>> besselyzero(0,100,1); besselyzero(0,1000,1); besselyzero(0,10000,1)
        313.3726705426359345050449
        3140.807136030340213610065
        31415.14112579761578220175

    Zeros of functions with large order::

        >>> besselyzero(50,1)
        53.50285882040036394680237
        >>> besselyzero(50,2)
        60.11244442774058114686022
        >>> besselyzero(50,100)
        387.1096509824943957706835
        >>> besselyzero(50,1,1)
        56.96290427516751320063605
        >>> besselyzero(50,2,1)
        62.74888166945933944036623
        >>> besselyzero(50,100,1)
        388.6923300548309258355475

    Zeros of functions with fractional order::

        >>> besselyzero(0.5,1); besselyzero(1.5,1); besselyzero(2.25,4)
        1.570796326794896619231322
        2.798386045783887136720249
        13.56721208770735123376018

    """
    return +bessel_zero(ctx, 2, derivative, v, m)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\truss.py ===
"""
This module can be used to solve problems related
to 2D Trusses.
"""


from cmath import atan, inf
from sympy.core.add import Add
from sympy.core.evalf import INF
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.external.importtools import import_module
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
import math
from sympy.physics.units.quantities import Quantity
from sympy.plotting import plot
from sympy.utilities.decorator import doctest_depends_on
from sympy import sin, cos


__doctest_requires__ = {('Truss.draw'): ['matplotlib']}


numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})


class Truss:
    """
    A Truss is an assembly of members such as beams,
    connected by nodes, that create a rigid structure.
    In engineering, a truss is a structure that
    consists of two-force members only.

    Trusses are extremely important in engineering applications
    and can be seen in numerous real-world applications like bridges.

    Examples
    ========

    There is a Truss consisting of four nodes and five
    members connecting the nodes. A force P acts
    downward on the node D and there also exist pinned
    and roller joints on the nodes A and B respectively.

    .. image:: truss_example.png

    >>> from sympy.physics.continuum_mechanics.truss import Truss
    >>> t = Truss()
    >>> t.add_node(("node_1", 0, 0), ("node_2", 6, 0), ("node_3", 2, 2), ("node_4", 2, 0))
    >>> t.add_member(("member_1", "node_1", "node_4"), ("member_2", "node_2", "node_4"), ("member_3", "node_1", "node_3"))
    >>> t.add_member(("member_4", "node_2", "node_3"), ("member_5", "node_3", "node_4"))
    >>> t.apply_load(("node_4", 10, 270))
    >>> t.apply_support(("node_1", "pinned"), ("node_2", "roller"))
    """

    def __init__(self):
        """
        Initializes the class
        """
        self._nodes = []
        self._members = {}
        self._loads = {}
        self._supports = {}
        self._node_labels = []
        self._node_positions = []
        self._node_position_x = []
        self._node_position_y = []
        self._nodes_occupied = {}
        self._member_lengths = {}
        self._reaction_loads = {}
        self._internal_forces = {}
        self._node_coordinates = {}

    @property
    def nodes(self):
        """
        Returns the nodes of the truss along with their positions.
        """
        return self._nodes

    @property
    def node_labels(self):
        """
        Returns the node labels of the truss.
        """
        return self._node_labels

    @property
    def node_positions(self):
        """
        Returns the positions of the nodes of the truss.
        """
        return self._node_positions

    @property
    def members(self):
        """
        Returns the members of the truss along with the start and end points.
        """
        return self._members

    @property
    def member_lengths(self):
        """
        Returns the length of each member of the truss.
        """
        return self._member_lengths

    @property
    def supports(self):
        """
        Returns the nodes with provided supports along with the kind of support provided i.e.
        pinned or roller.
        """
        return self._supports

    @property
    def loads(self):
        """
        Returns the loads acting on the truss.
        """
        return self._loads

    @property
    def reaction_loads(self):
        """
        Returns the reaction forces for all supports which are all initialized to 0.
        """
        return self._reaction_loads

    @property
    def internal_forces(self):
        """
        Returns the internal forces for all members which are all initialized to 0.
        """
        return self._internal_forces

    def add_node(self, *args):
        """
        This method adds a node to the truss along with its name/label and its location.
        Multiple nodes can be added at the same time.

        Parameters
        ==========
        The input(s) for this method are tuples of the form (label, x, y).

        label:  String or a Symbol
            The label for a node. It is the only way to identify a particular node.

        x: Sympifyable
            The x-coordinate of the position of the node.

        y: Sympifyable
            The y-coordinate of the position of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0))
        >>> t.nodes
        [('A', 0, 0)]
        >>> t.add_node(('B', 3, 0), ('C', 4, 1))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 4, 1)]
        """

        for i in args:
            label = i[0]
            x = i[1]
            x = sympify(x)
            y=i[2]
            y = sympify(y)
            if label in self._node_coordinates:
                raise ValueError("Node needs to have a unique label")

            elif [x, y] in self._node_coordinates.values():
                raise ValueError("A node already exists at the given position")

            else :
                self._nodes.append((label, x, y))
                self._node_labels.append(label)
                self._node_positions.append((x, y))
                self._node_position_x.append(x)
                self._node_position_y.append(y)
                self._node_coordinates[label] = [x, y]



    def remove_node(self, *args):
        """
        This method removes a node from the truss.
        Multiple nodes can be removed at the same time.

        Parameters
        ==========
        The input(s) for this method are the labels of the nodes to be removed.

        label:  String or Symbol
            The label of the node to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 5, 0)]
        >>> t.remove_node('A', 'C')
        >>> t.nodes
        [('B', 3, 0)]
        """
        for label in args:
            for i in range(len(self.nodes)):
                if self._node_labels[i] == label:
                    x = self._node_position_x[i]
                    y = self._node_position_y[i]

            if label not in self._node_coordinates:
                raise ValueError("No such node exists in the truss")

            else:
                members_duplicate = self._members.copy()
                for member in members_duplicate:
                    if label == self._members[member][0] or label == self._members[member][1]:
                        raise ValueError("The given node already has member attached to it")
                self._nodes.remove((label, x, y))
                self._node_labels.remove(label)
                self._node_positions.remove((x, y))
                self._node_position_x.remove(x)
                self._node_position_y.remove(y)
                if label in self._loads:
                    self._loads.pop(label)
                if label in self._supports:
                    self._supports.pop(label)
                self._node_coordinates.pop(label)



    def add_member(self, *args):
        """
        This method adds a member between any two nodes in the given truss.

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (label, start, end).

        label: String or Symbol
            The label for a member. It is the only way to identify a particular member.

        start: String or Symbol
            The label of the starting point/node of the member.

        end: String or Symbol
            The label of the ending point/node of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'BC': ['B', 'C']}
        """
        for i in args:
            label = i[0]
            start = i[1]
            end = i[2]

            if start not in self._node_coordinates or end not in self._node_coordinates or start==end:
                raise ValueError("The start and end points of the member must be unique nodes")

            elif label in self._members:
                raise ValueError("A member with the same label already exists for the truss")

            elif self._nodes_occupied.get((start, end)):
                raise ValueError("A member already exists between the two nodes")

            else:
                self._members[label] = [start, end]
                self._member_lengths[label] = sqrt((self._node_coordinates[end][0]-self._node_coordinates[start][0])**2 + (self._node_coordinates[end][1]-self._node_coordinates[start][1])**2)
                self._nodes_occupied[start, end] = True
                self._nodes_occupied[end, start] = True
                self._internal_forces[label] = 0

    def remove_member(self, *args):
        """
        This method removes members from the given truss.

        Parameters
        ==========
        labels: String or Symbol
            The label for the member to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('AC', 'A', 'C'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'AC': ['A', 'C'], 'BC': ['B', 'C']}
        >>> t.remove_member('AC', 'BC')
        >>> t.members
        {'AB': ['A', 'B']}
        """
        for label in args:
            if label not in self._members:
                raise ValueError("No such member exists in the Truss")

            else:
                self._nodes_occupied.pop((self._members[label][0], self._members[label][1]))
                self._nodes_occupied.pop((self._members[label][1], self._members[label][0]))
                self._members.pop(label)
                self._member_lengths.pop(label)
                self._internal_forces.pop(label)

    def change_node_label(self, *args):
        """
        This method changes the label(s) of the specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label).

        label: String or Symbol
            The label of the node for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0)]
        >>> t.change_node_label(('A', 'C'), ('B', 'D'))
        >>> t.nodes
        [('C', 0, 0), ('D', 3, 0)]
        """
        for i in args:
            label = i[0]
            new_label = i[1]
            if label not in self._node_coordinates:
                raise ValueError("No such node exists for the Truss")
            elif new_label in self._node_coordinates:
                raise ValueError("A node with the given label already exists")
            else:
                for node in self._nodes:
                    if node[0] == label:
                        self._nodes[self._nodes.index((label, node[1], node[2]))] = (new_label, node[1], node[2])
                        self._node_labels[self._node_labels.index(node[0])] = new_label
                        self._node_coordinates[new_label] = self._node_coordinates[label]
                        self._node_coordinates.pop(label)
                        if node[0] in self._supports:
                            self._supports[new_label] = self._supports[node[0]]
                            self._supports.pop(node[0])
                        if new_label in self._supports:
                            if self._supports[new_label] == 'pinned':
                                if 'R_'+str(label)+'_x' in self._reaction_loads and 'R_'+str(label)+'_y' in self._reaction_loads:
                                    self._reaction_loads['R_'+str(new_label)+'_x'] = self._reaction_loads['R_'+str(label)+'_x']
                                    self._reaction_loads['R_'+str(new_label)+'_y'] = self._reaction_loads['R_'+str(label)+'_y']
                                    self._reaction_loads.pop('R_'+str(label)+'_x')
                                    self._reaction_loads.pop('R_'+str(label)+'_y')
                                self._loads[new_label] = self._loads[label]
                                for load in self._loads[new_label]:
                                    if load[1] == 90:
                                        load[0] -= Symbol('R_'+str(label)+'_y')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                for load in self._loads[new_label]:
                                    if load[1] == 0:
                                        load[0] -= Symbol('R_'+str(label)+'_x')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_x'), 0)
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_y'), 90)
                                self._loads.pop(label)
                            elif self._supports[new_label] == 'roller':
                                self._loads[new_label] = self._loads[label]
                                for load in self._loads[label]:
                                    if load[1] == 90:
                                        load[0] -= Symbol('R_'+str(label)+'_y')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_y'), 90)
                                self._loads.pop(label)
                        else:
                            if label in self._loads:
                                self._loads[new_label] = self._loads[label]
                                self._loads.pop(label)
                        for member in self._members:
                            if self._members[member][0] == node[0]:
                                self._members[member][0] = new_label
                                self._nodes_occupied[(new_label, self._members[member][1])] = True
                                self._nodes_occupied[(self._members[member][1], new_label)] = True
                                self._nodes_occupied.pop((label, self._members[member][1]))
                                self._nodes_occupied.pop((self._members[member][1], label))
                            elif self._members[member][1] == node[0]:
                                self._members[member][1] = new_label
                                self._nodes_occupied[(self._members[member][0], new_label)] = True
                                self._nodes_occupied[(new_label, self._members[member][0])] = True
                                self._nodes_occupied.pop((self._members[member][0], label))
                                self._nodes_occupied.pop((label, self._members[member][0]))

    def change_member_label(self, *args):
        """
        This method changes the label(s) of the specified member(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label)

        label: String or Symbol
            The label of the member for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('D', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.change_node_label(('A', 'C'))
        >>> t.nodes
        [('C', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.add_member(('BC', 'B', 'C'), ('BD', 'B', 'D'))
        >>> t.members
        {'BC': ['B', 'C'], 'BD': ['B', 'D']}
        >>> t.change_member_label(('BC', 'BC_new'), ('BD', 'BD_new'))
        >>> t.members
        {'BC_new': ['B', 'C'], 'BD_new': ['B', 'D']}
        """
        for i in args:
            label = i[0]
            new_label = i[1]
            if label not in self._members:
                raise ValueError("No such member exists for the Truss")
            else:
                members_duplicate = list(self._members).copy()
                for member in members_duplicate:
                    if member == label:
                        self._members[new_label] = [self._members[member][0], self._members[member][1]]
                        self._members.pop(label)
                        self._member_lengths[new_label] = self._member_lengths[label]
                        self._member_lengths.pop(label)
                        self._internal_forces[new_label] = self._internal_forces[label]
                        self._internal_forces.pop(label)

    def apply_load(self, *args):
        """
        This method applies external load(s) at the specified node(s).

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied.

        magnitude: Sympifyable
            Magnitude of the load applied. It must always be positive and any changes in
            the direction of the load are not reflected here.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        """
        for i in args:
            location = i[0]
            magnitude = i[1]
            direction = i[2]
            magnitude = sympify(magnitude)
            direction = sympify(direction)

            if location not in self._node_coordinates:
                raise ValueError("Load must be applied at a known node")

            else:
                if location in self._loads:
                    self._loads[location].append([magnitude, direction])
                else:
                    self._loads[location] = [[magnitude, direction]]

    def remove_load(self, *args):
        """
        This method removes already
        present external load(s) at specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied and is to be removed.

        magnitude: Sympifyable
            Magnitude of the load applied.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        >>> t.remove_load(('A', P/4, 90), ('A', P/2, 45))
        >>> t.loads
        {'A': [[P, 90]]}
        """
        for i in args:
            location = i[0]
            magnitude = i[1]
            direction = i[2]
            magnitude = sympify(magnitude)
            direction = sympify(direction)

            if location not in self._node_coordinates:
                raise ValueError("Load must be removed from a known node")

            else:
                if [magnitude, direction] not in self._loads[location]:
                    raise ValueError("No load of this magnitude and direction has been applied at this node")
                else:
                    self._loads[location].remove([magnitude, direction])
            if self._loads[location] == []:
                self._loads.pop(location)

    def apply_support(self, *args):
        """
        This method adds a pinned or roller support at specified node(s).

        Parameters
        ==========
        The input(s) of this method are of the form (location, type).

        location: String or Symbol
            Label of the Node at which support is added.

        type: String
            Type of the support being provided at the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        """
        for i in args:
            location = i[0]
            type = i[1]
            if location not in self._node_coordinates:
                raise ValueError("Support must be added on a known node")

            else:
                if location not in self._supports:
                    if type == 'pinned':
                        self.apply_load((location, Symbol('R_'+str(location)+'_x'), 0))
                        self.apply_load((location, Symbol('R_'+str(location)+'_y'), 90))
                    elif type == 'roller':
                        self.apply_load((location, Symbol('R_'+str(location)+'_y'), 90))
                elif self._supports[location] == 'pinned':
                    if type == 'roller':
                        self.remove_load((location, Symbol('R_'+str(location)+'_x'), 0))
                elif self._supports[location] == 'roller':
                    if type == 'pinned':
                        self.apply_load((location, Symbol('R_'+str(location)+'_x'), 0))
                self._supports[location] = type

    def remove_support(self, *args):
        """
        This method removes support from specified node(s.)

        Parameters
        ==========

        locations: String or Symbol
            Label of the Node(s) at which support is to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        >>> t.remove_support('A','B')
        >>> t.supports
        {}
        """
        for location in args:

            if location not in self._node_coordinates:
                raise ValueError("No such node exists in the Truss")

            elif location not in self._supports:
                raise ValueError("No support has been added to the given node")

            else:
                if self._supports[location] == 'pinned':
                    self.remove_load((location, Symbol('R_'+str(location)+'_x'), 0))
                    self.remove_load((location, Symbol('R_'+str(location)+'_y'), 90))
                elif self._supports[location] == 'roller':
                    self.remove_load((location, Symbol('R_'+str(location)+'_y'), 90))
                self._supports.pop(location)

    def solve(self):
        """
        This method solves for all reaction forces of all supports and all internal forces
        of all the members in the truss, provided the Truss is solvable.

        A Truss is solvable if the following condition is met,

        2n >= r + m

        Where n is the number of nodes, r is the number of reaction forces, where each pinned
        support has 2 reaction forces and each roller has 1, and m is the number of members.

        The given condition is derived from the fact that a system of equations is solvable
        only when the number of variables is lesser than or equal to the number of equations.
        Equilibrium Equations in x and y directions give two equations per node giving 2n number
        equations. However, the truss needs to be stable as well and may be unstable if 2n > r + m.
        The number of variables is simply the sum of the number of reaction forces and member
        forces.

        .. note::
           The sign convention for the internal forces present in a member revolves around whether each
           force is compressive or tensile. While forming equations for each node, internal force due
           to a member on the node is assumed to be away from the node i.e. each force is assumed to
           be compressive by default. Hence, a positive value for an internal force implies the
           presence of compressive force in the member and a negative value implies a tensile force.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(("node_1", 0, 0), ("node_2", 6, 0), ("node_3", 2, 2), ("node_4", 2, 0))
        >>> t.add_member(("member_1", "node_1", "node_4"), ("member_2", "node_2", "node_4"), ("member_3", "node_1", "node_3"))
        >>> t.add_member(("member_4", "node_2", "node_3"), ("member_5", "node_3", "node_4"))
        >>> t.apply_load(("node_4", 10, 270))
        >>> t.apply_support(("node_1", "pinned"), ("node_2", "roller"))
        >>> t.solve()
        >>> t.reaction_loads
        {'R_node_1_x': 0, 'R_node_1_y': 20/3, 'R_node_2_y': 10/3}
        >>> t.internal_forces
        {'member_1': 20/3, 'member_2': 20/3, 'member_3': -20*sqrt(2)/3, 'member_4': -10*sqrt(5)/3, 'member_5': 10}
        """
        count_reaction_loads = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    count_reaction_loads += 2
                elif self._supports[node[0]]=='roller':
                    count_reaction_loads += 1
        if 2*len(self._nodes) != len(self._members) + count_reaction_loads:
            raise ValueError("The given truss cannot be solved")
        coefficients_matrix = [[0 for i in range(2*len(self._nodes))] for j in range(2*len(self._nodes))]
        load_matrix = zeros(2*len(self.nodes), 1)
        load_matrix_row = 0
        for node in self._nodes:
            if node[0] in self._loads:
                for load in self._loads[node[0]]:
                    if load[0]!=Symbol('R_'+str(node[0])+'_x') and load[0]!=Symbol('R_'+str(node[0])+'_y'):
                        load_matrix[load_matrix_row] -= load[0]*cos(pi*load[1]/180)
                        load_matrix[load_matrix_row + 1] -= load[0]*sin(pi*load[1]/180)
            load_matrix_row += 2
        cols = 0
        row = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    coefficients_matrix[row][cols] += 1
                    coefficients_matrix[row+1][cols+1] += 1
                    cols += 2
                elif self._supports[node[0]]=='roller':
                    coefficients_matrix[row+1][cols] += 1
                    cols += 1
            row += 2
        for member in self._members:
            start = self._members[member][0]
            end = self._members[member][1]
            length = sqrt((self._node_coordinates[start][0]-self._node_coordinates[end][0])**2 + (self._node_coordinates[start][1]-self._node_coordinates[end][1])**2)
            start_index = self._node_labels.index(start)
            end_index = self._node_labels.index(end)
            horizontal_component_start = (self._node_coordinates[end][0]-self._node_coordinates[start][0])/length
            vertical_component_start = (self._node_coordinates[end][1]-self._node_coordinates[start][1])/length
            horizontal_component_end = (self._node_coordinates[start][0]-self._node_coordinates[end][0])/length
            vertical_component_end = (self._node_coordinates[start][1]-self._node_coordinates[end][1])/length
            coefficients_matrix[start_index*2][cols] += horizontal_component_start
            coefficients_matrix[start_index*2+1][cols] += vertical_component_start
            coefficients_matrix[end_index*2][cols] += horizontal_component_end
            coefficients_matrix[end_index*2+1][cols] += vertical_component_end
            cols += 1
        forces_matrix = (Matrix(coefficients_matrix)**-1)*load_matrix
        self._reaction_loads = {}
        i = 0
        min_load = inf
        for node in self._nodes:
            if node[0] in self._loads:
                for load in self._loads[node[0]]:
                    if type(load[0]) not in [Symbol, Mul, Add]:
                        min_load = min(min_load, load[0])
        for j in range(len(forces_matrix)):
            if type(forces_matrix[j]) not in [Symbol, Mul, Add]:
                if abs(forces_matrix[j]/min_load) <1E-10:
                    forces_matrix[j] = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    self._reaction_loads['R_'+str(node[0])+'_x'] = forces_matrix[i]
                    self._reaction_loads['R_'+str(node[0])+'_y'] = forces_matrix[i+1]
                    i += 2
                elif self._supports[node[0]]=='roller':
                    self._reaction_loads['R_'+str(node[0])+'_y'] = forces_matrix[i]
                    i += 1
        for member in self._members:
            self._internal_forces[member] = forces_matrix[i]
            i += 1
        return

    @doctest_depends_on(modules=('numpy',))
    def draw(self, subs_dict=None):
        """
        Returns a plot object of the Truss with all its nodes, members,
        supports and loads.

        .. note::
            The user must be careful while entering load values in their
            directions. The draw function assumes a sign convention that
            is used for plotting loads.

            Given a right-handed coordinate system with XYZ coordinates,
            the supports are assumed to be such that the reaction forces of a
            pinned support is in the +X and +Y direction while those of a
            roller support is in the +Y direction. For the load, the range
            of angles, one can input goes all the way to 360 degrees which, in the
            the plot is the angle that the load vector makes with the positive x-axis in the anticlockwise direction.

            For example, for a 90-degree angle, the load will be a vertically
            directed along +Y while a 270-degree angle denotes a vertical
            load as well but along -Y.

        Examples
        ========

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.truss import Truss
            >>> import math
            >>> t = Truss()
            >>> t.add_node(("A", -4, 0), ("B", 0, 0), ("C", 4, 0), ("D", 8, 0))
            >>> t.add_node(("E", 6, 2/math.sqrt(3)))
            >>> t.add_node(("F", 2, 2*math.sqrt(3)))
            >>> t.add_node(("G", -2, 2/math.sqrt(3)))
            >>> t.add_member(("AB","A","B"), ("BC","B","C"), ("CD","C","D"))
            >>> t.add_member(("AG","A","G"), ("GB","G","B"), ("GF","G","F"))
            >>> t.add_member(("BF","B","F"), ("FC","F","C"), ("CE","C","E"))
            >>> t.add_member(("FE","F","E"), ("DE","D","E"))
            >>> t.apply_support(("A","pinned"), ("D","roller"))
            >>> t.apply_load(("G", 3, 90), ("E", 3, 90), ("F", 2, 90))
            >>> p = t.draw()
            >>> p  # doctest: +ELLIPSIS
            Plot object containing:
            [0]: cartesian line: 1 for x over (1.0, 1.0)
            ...
            >>> p.show()
        """
        if not numpy:
            raise ImportError("To use this function numpy module is required")

        x = Symbol('x')

        markers = []
        annotations = []
        rectangles = []

        node_markers = self._draw_nodes(subs_dict)
        markers += node_markers

        member_rectangles = self._draw_members()
        rectangles += member_rectangles

        support_markers = self._draw_supports()
        markers += support_markers

        load_annotations = self._draw_loads()
        annotations += load_annotations

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        lim = max(xmax*1.1-xmin*0.8+1, ymax*1.1-ymin*0.8+1)

        if lim==xmax*1.1-xmin*0.8+1:
            sing_plot = plot(1, (x, 1, 1), markers=markers, show=False, annotations=annotations, xlim=(xmin-0.05*lim, xmax*1.1), ylim=(xmin-0.05*lim, xmax*1.1), axis=False, rectangles=rectangles)

        else:
            sing_plot = plot(1, (x, 1, 1), markers=markers, show=False, annotations=annotations, xlim=(ymin-0.05*lim, ymax*1.1), ylim=(ymin-0.05*lim, ymax*1.1), axis=False, rectangles=rectangles)

        return sing_plot


    def _draw_nodes(self, subs_dict):
        node_markers = []

        for node in self._node_coordinates:
            if (type(self._node_coordinates[node][0]) in (Symbol, Quantity)):
                if self._node_coordinates[node][0] in subs_dict:
                    self._node_coordinates[node][0] = subs_dict[self._node_coordinates[node][0]]
                else:
                    raise ValueError("provided substituted dictionary is not adequate")
            elif (type(self._node_coordinates[node][0]) == Mul):
                objects = self._node_coordinates[node][0].as_coeff_Mul()
                for object in objects:
                    if type(object) in (Symbol, Quantity):
                        if subs_dict==None or object not in subs_dict:
                            raise ValueError("provided substituted dictionary is not adequate")
                        else:
                            self._node_coordinates[node][0] /= object
                            self._node_coordinates[node][0] *= subs_dict[object]

            if (type(self._node_coordinates[node][1]) in (Symbol, Quantity)):
                if self._node_coordinates[node][1] in subs_dict:
                    self._node_coordinates[node][1] = subs_dict[self._node_coordinates[node][1]]
                else:
                    raise ValueError("provided substituted dictionary is not adequate")
            elif (type(self._node_coordinates[node][1]) == Mul):
                objects = self._node_coordinates[node][1].as_coeff_Mul()
                for object in objects:
                    if type(object) in (Symbol, Quantity):
                        if subs_dict==None or object not in subs_dict:
                            raise ValueError("provided substituted dictionary is not adequate")
                        else:
                            self._node_coordinates[node][1] /= object
                            self._node_coordinates[node][1] *= subs_dict[object]

        for node in self._node_coordinates:
            node_markers.append(
                {
                    'args':[[self._node_coordinates[node][0]], [self._node_coordinates[node][1]]],
                    'marker':'o',
                    'markersize':5,
                    'color':'black'
                }
            )
        return node_markers

    def _draw_members(self):

        member_rectangles = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for member in self._members:
            x1 = self._node_coordinates[self._members[member][0]][0]
            y1 = self._node_coordinates[self._members[member][0]][1]
            x2 = self._node_coordinates[self._members[member][1]][0]
            y2 = self._node_coordinates[self._members[member][1]][1]
            if x2!=x1 and y2!=y1:
                if x2>x1:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff*cos(pi/4+atan((y2-y1)/(x2-x1)))/2, y1-0.005*max_diff*sin(pi/4+atan((y2-y1)/(x2-x1)))/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/math.sqrt(2),
                            'height':0.005*max_diff,
                            'angle':180*atan((y2-y1)/(x2-x1))/pi,
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x2-0.005*max_diff*cos(pi/4+atan((y2-y1)/(x2-x1)))/2, y2-0.005*max_diff*sin(pi/4+atan((y2-y1)/(x2-x1)))/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/math.sqrt(2),
                            'height':0.005*max_diff,
                            'angle':180*atan((y2-y1)/(x2-x1))/pi,
                            'color':'brown'
                        }
                    )
            elif y2==y1:
                if x2>x1:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2),
                            'height':0.005*max_diff,
                            'angle':90*(1-math.copysign(1, x2-x1)),
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2),
                            'height':-0.005*max_diff,
                            'angle':90*(1-math.copysign(1, x2-x1)),
                            'color':'brown'
                        }
                    )
            else:
                if y1<y2:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/2,
                            'height':0.005*max_diff,
                            'angle':90*math.copysign(1, y2-y1),
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x2-0.005*max_diff/2, y2-0.005*max_diff/2),
                            'width':-(sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/2),
                            'height':0.005*max_diff,
                            'angle':90*math.copysign(1, y2-y1),
                            'color':'brown'
                        }
                    )

        return member_rectangles

    def _draw_supports(self):
        support_markers = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])
        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for node in self._supports:
            if self._supports[node]=='pinned':
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]]
                        ],
                        'marker':6,
                        'markersize':15,
                        'color':'black',
                        'markerfacecolor':'none'
                    }
                )
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.035*max_diff]
                        ],
                        'marker':'_',
                        'markersize':14,
                        'color':'black'
                    }
                )

            elif self._supports[node]=='roller':
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.02*max_diff]
                        ],
                        'marker':'o',
                        'markersize':11,
                        'color':'black',
                        'markerfacecolor':'none'
                    }
                )
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.0375*max_diff]
                        ],
                        'marker':'_',
                        'markersize':14,
                        'color':'black'
                    }
                )
        return support_markers

    def _draw_loads(self):
        load_annotations = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin+5
        else:
            max_diff = 1.1*ymax-0.8*ymin+5

        for node in self._loads:
            for load in self._loads[node]:
                if load[0] in [Symbol('R_'+str(node)+'_x'), Symbol('R_'+str(node)+'_y')]:
                    continue
                x = self._node_coordinates[node][0]
                y = self._node_coordinates[node][1]
                load_annotations.append(
                    {
                        'text':'',
                        'xy':(
                            x-math.cos(pi*load[1]/180)*(max_diff/100),
                            y-math.sin(pi*load[1]/180)*(max_diff/100)
                        ),
                        'xytext':(
                            x-(max_diff/100+abs(xmax-xmin)+abs(ymax-ymin))*math.cos(pi*load[1]/180)/20,
                            y-(max_diff/100+abs(xmax-xmin)+abs(ymax-ymin))*math.sin(pi*load[1]/180)/20
                        ),
                        'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'black'}
                    }
                )
        return load_annotations

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\simplex.py ===
"""Tools for optimizing a linear function for a given simplex.

For the linear objective function ``f`` with linear constraints
expressed using `Le`, `Ge` or `Eq` can be found with ``lpmin`` or
``lpmax``. The symbols are **unbounded** unless specifically
constrained.

As an alternative, the matrices describing the objective and the
constraints, and an optional list of bounds can be passed to
``linprog`` which will solve for the minimization of ``C*x``
under constraints ``A*x <= b`` and/or ``Aeq*x = beq``, and
individual bounds for variables given as ``(lo, hi)``. The values
returned are **nonnegative** unless bounds are provided that
indicate otherwise.

Errors that might be raised are UnboundedLPError when there is no
finite solution for the system or InfeasibleLPError when the
constraints represent impossible conditions (i.e. a non-existent
 simplex).

Here is a simple 1-D system: minimize `x` given that ``x >= 1``.

    >>> from sympy.solvers.simplex import lpmin, linprog
    >>> from sympy.abc import x

    The function and a list with the constraint is passed directly
    to `lpmin`:

    >>> lpmin(x, [x >= 1])
    (1, {x: 1})

    For `linprog` the matrix for the objective is `[1]` and the
    uivariate constraint can be passed as a bound with None acting
    as infinity:

    >>> linprog([1], bounds=(1, None))
    (1, [1])

    Or the matrices, corresponding to ``x >= 1`` expressed as
    ``-x <= -1`` as required by the routine, can be passed:

    >>> linprog([1], [-1], [-1])
    (1, [1])

    If there is no limit for the objective, an error is raised.
    In this case there is a valid region of interest (simplex)
    but no limit to how small ``x`` can be:

    >>> lpmin(x, [])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    An error is raised if there is no possible solution:

    >>> lpmin(x,[x<=1,x>=2])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError:
    Inconsistent/False constraint
"""

from sympy.core import sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Le, Ge, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sorting import ordered
from sympy.functions.elementary.complexes import sign
from sympy.matrices.dense import Matrix, zeros
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.utilities.iterables import numbered_symbols
from sympy.utilities.misc import filldedent


class UnboundedLPError(Exception):
    """
    A linear programming problem is said to be unbounded if its objective
    function can assume arbitrarily large values.

    Example
    =======

    Suppose you want to maximize
        2x
    subject to
        x >= 0

    There's no upper limit that 2x can take.
    """

    pass


class InfeasibleLPError(Exception):
    """
    A linear programming problem is considered infeasible if its
    constraint set is empty. That is, if the set of all vectors
    satisfying the constraints is empty, then the problem is infeasible.

    Example
    =======

    Suppose you want to maximize
        x
    subject to
        x >= 10
        x <= 9

    No x can satisfy those constraints.
    """

    pass


def _pivot(M, i, j):
    """
    The pivot element `M[i, j]` is inverted and the rest of the matrix
    modified and returned as a new matrix; original is left unmodified.

    Example
    =======

    >>> from sympy.matrices.dense import Matrix
    >>> from sympy.solvers.simplex import _pivot
    >>> from sympy import var
    >>> Matrix(3, 3, var('a:i'))
    Matrix([
    [a, b, c],
    [d, e, f],
    [g, h, i]])
    >>> _pivot(_, 1, 0)
    Matrix([
    [-a/d, -a*e/d + b, -a*f/d + c],
    [ 1/d,        e/d,        f/d],
    [-g/d,  h - e*g/d,  i - f*g/d]])
    """
    Mi, Mj, Mij = M[i, :], M[:, j], M[i, j]
    if Mij == 0:
        raise ZeroDivisionError(
            "Tried to pivot about zero-valued entry.")
    A = M - Mj * (Mi / Mij)
    A[i, :] = Mi / Mij
    A[:, j] = -Mj / Mij
    A[i, j] = 1 / Mij
    return A


def _choose_pivot_row(A, B, candidate_rows, pivot_col, Y):
    # Choose row with smallest ratio
    # If there are ties, pick using Bland's rule
    return min(candidate_rows, key=lambda i: (B[i] / A[i, pivot_col], Y[i]))


def _simplex(A, B, C, D=None, dual=False):
    """Return ``(o, x, y)`` obtained from the two-phase simplex method
    using Bland's rule: ``o`` is the minimum value of primal,
    ``Cx - D``, under constraints ``Ax <= B`` (with ``x >= 0``) and
    the maximum of the dual, ``y^{T}B - D``, under constraints
    ``A^{T}*y >= C^{T}`` (with ``y >= 0``). To compute the dual of
    the system, pass `dual=True` and ``(o, y, x)`` will be returned.

    Note: the nonnegative constraints for ``x`` and ``y`` supercede
    any values of ``A`` and ``B`` that are inconsistent with that
    assumption, so if a constraint of ``x >= -1`` is represented
    in ``A`` and ``B``, no value will be obtained that is negative; if
    a constraint of ``x <= -1`` is represented, an error will be
    raised since no solution is possible.

    This routine relies on the ability of determining whether an
    expression is 0 or not. This is guaranteed if the input contains
    only Float or Rational entries. It will raise a TypeError if
    a relationship does not evaluate to True or False.

    Examples
    ========

    >>> from sympy.solvers.simplex import _simplex
    >>> from sympy import Matrix

    Consider the simple minimization of ``f = x + y + 1`` under the
    constraint that ``y + 2*x >= 4``. This is the "standard form" of
    a minimization.

    In the nonnegative quadrant, this inequality describes a area above
    a triangle with vertices at (0, 4), (0, 0) and (2, 0). The minimum
    of ``f`` occurs at (2, 0). Define A, B, C, D for the standard
    minimization:

    >>> A = Matrix([[2, 1]])
    >>> B = Matrix([4])
    >>> C = Matrix([[1, 1]])
    >>> D = Matrix([-1])

    Confirm that this is the system of interest:

    >>> from sympy.abc import x, y
    >>> X = Matrix([x, y])
    >>> (C*X - D)[0]
    x + y + 1
    >>> [i >= j for i, j in zip(A*X, B)]
    [2*x + y >= 4]

    Since `_simplex` will do a minimization for constraints given as
    ``A*x <= B``, the signs of ``A`` and ``B`` must be negated since
    the currently correspond to a greater-than inequality:

    >>> _simplex(-A, -B, C, D)
    (3, [2, 0], [1/2])

    The dual of minimizing ``f`` is maximizing ``F = c*y - d`` for
    ``a*y <= b`` where ``a``, ``b``, ``c``, ``d`` are derived from the
    transpose of the matrix representation of the standard minimization:

    >>> tr = lambda a, b, c, d: [i.T for i in (a, c, b, d)]
    >>> a, b, c, d = tr(A, B, C, D)

    This time ``a*x <= b`` is the expected inequality for the `_simplex`
    method, but to maximize ``F``, the sign of ``c`` and ``d`` must be
    changed (so that minimizing the negative will give the negative of
    the maximum of ``F``):

    >>> _simplex(a, b, -c, -d)
    (-3, [1/2], [2, 0])

    The negative of ``F`` and the min of ``f`` are the same. The dual
    point `[1/2]` is the value of ``y`` that minimized ``F = c*y - d``
    under constraints a*x <= b``:

    >>> y = Matrix(['y'])
    >>> (c*y - d)[0]
    4*y + 1
    >>> [i <= j for i, j in zip(a*y,b)]
    [2*y <= 1, y <= 1]

    In this 1-dimensional dual system, the more restrictive constraint is
    the first which limits ``y`` between 0 and 1/2 and the maximum of
    ``F`` is attained at the nonzero value, hence is ``4*(1/2) + 1 = 3``.

    In this case the values for ``x`` and ``y`` were the same when the
    dual representation was solved. This is not always the case (though
    the value of the function will be the same).

    >>> l = [[1, 1], [-1, 1], [0, 1], [-1, 0]], [5, 1, 2, -1], [[1, 1]], [-1]
    >>> A, B, C, D = [Matrix(i) for i in l]
    >>> _simplex(A, B, -C, -D)
    (-6, [3, 2], [1, 0, 0, 0])
    >>> _simplex(A, B, -C, -D, dual=True)  # [5, 0] != [3, 2]
    (-6, [1, 0, 0, 0], [5, 0])

    In both cases the function has the same value:

    >>> Matrix(C)*Matrix([3, 2]) == Matrix(C)*Matrix([5, 0])
    True

    See Also
    ========
    _lp - poses min/max problem in form compatible with _simplex
    lpmin - minimization which calls _lp
    lpmax - maximimzation which calls _lp

    References
    ==========

    .. [1] Thomas S. Ferguson, LINEAR PROGRAMMING: A Concise Introduction
           web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_lp.pdf

    """
    A, B, C, D = [Matrix(i) for i in (A, B, C, D or [0])]
    if dual:
        _o, d, p = _simplex(-A.T, C.T, B.T, -D)
        return -_o, d, p

    if A and B:
        M = Matrix([[A, B], [C, D]])
    else:
        if A or B:
            raise ValueError("must give A and B")
        # no constraints given
        M = Matrix([[C, D]])
    n = M.cols - 1
    m = M.rows - 1

    if not all(i.is_Float or i.is_Rational for i in M):
        # with literal Float and Rational we are guaranteed the
        # ability of determining whether an expression is 0 or not
        raise TypeError(filldedent("""
            Only rationals and floats are allowed.
            """
            )
        )

    # x variables have priority over y variables during Bland's rule
    # since False < True
    X = [(False, j) for j in range(n)]
    Y = [(True, i) for i in range(m)]

    # Phase 1: find a feasible solution or determine none exist

    ## keep track of last pivot row and column
    last = None

    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        if all(B[i] >= 0 for i in range(B.rows)):
            # We have found a feasible solution
            break

        # Find k: first row with a negative rightmost entry
        for k in range(B.rows):
            if B[k] < 0:
                break  # use current value of k below
        else:
            pass  # error will raise below

        # Choose pivot column, c
        piv_cols = [_ for _ in range(A.cols) if A[k, _] < 0]
        if not piv_cols:
            raise InfeasibleLPError(filldedent("""
                The constraint set is empty!"""))
        _, c = min((X[i], i) for i in piv_cols) # Bland's rule

        # Choose pivot row, r
        piv_rows = [_ for _ in range(A.rows) if A[_, c] > 0 and B[_] > 0]
        piv_rows.append(k)
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        # check for oscillation
        if (r, c) == last:
            # Not sure what to do here; it looks like there will be
            # oscillations; see o1 test added at this commit to
            # see a system with no solution and the o2 for one
            # with a solution. In the case of o2, the solution
            # from linprog is the same as the one from lpmin, but
            # the matrices created in the lpmin case are different
            # than those created without replacements in linprog and
            # the matrices in the linprog case lead to oscillations.
            # If the matrices could be re-written in linprog like
            # lpmin does, this behavior could be avoided and then
            # perhaps the oscillating case would only occur when
            # there is no solution. For now, the output is checked
            # before exit if oscillations were detected and an
            # error is raised there if the solution was invalid.
            #
            # cf section 6 of Ferguson for a non-cycling modification
            last = True
            break
        last = r, c

        M = _pivot(M, r, c)
        X[c], Y[r] = Y[r], X[c]

    # Phase 2: from a feasible solution, pivot to optimal
    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        C = M[-1, :-1]

        # Choose a pivot column, c
        piv_cols = [_ for _ in range(n) if C[_] < 0]
        if not piv_cols:
            break
        _, c = min((X[i], i) for i in piv_cols)  # Bland's rule

        # Choose a pivot row, r
        piv_rows = [_ for _ in range(m) if A[_, c] > 0]
        if not piv_rows:
            raise UnboundedLPError(filldedent("""
                Objective function can assume
                arbitrarily large values!"""))
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        M = _pivot(M, r, c)
        X[c], Y[r] = Y[r], X[c]

    argmax = [None] * n
    argmin_dual = [None] * m

    for i, (v, n) in enumerate(X):
        if v == False:
            argmax[n] = 0
        else:
            argmin_dual[n] = M[-1, i]

    for i, (v, n) in enumerate(Y):
        if v == True:
            argmin_dual[n] = 0
        else:
            argmax[n] = M[i, -1]

    if last and not all(i >= 0 for i in argmax + argmin_dual):
        raise InfeasibleLPError(filldedent("""
            Oscillating system led to invalid solution.
            If you believe there was a valid solution, please
            report this as a bug."""))
    return -M[-1, -1], argmax, argmin_dual


## routines that use _simplex or support those that do


def _abcd(M, list=False):
    """return parts of M as matrices or lists

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd

    >>> m = Matrix(3, 3, range(9)); m
    Matrix([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]])
    >>> a, b, c, d = _abcd(m)
    >>> a
    Matrix([
    [0, 1],
    [3, 4]])
    >>> b
    Matrix([
    [2],
    [5]])
    >>> c
    Matrix([[6, 7]])
    >>> d
    Matrix([[8]])

    The matrices can be returned as compact lists, too:

    >>> L = a, b, c, d = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    """

    def aslist(i):
        l = i.tolist()
        if len(l[0]) == 1:  # col vector
            return [i[0] for i in l]
        return l

    m = M[:-1, :-1], M[:-1, -1], M[-1, :-1], M[-1:, -1:]
    if not list:
        return m
    return tuple([aslist(i) for i in m])


def _m(a, b, c, d=None):
    """return Matrix([[a, b], [c, d]]) from matrices
    in Matrix or list form.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd, _m
    >>> m = Matrix(3, 3, range(9))
    >>> L = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    >>> _abcd(m)
    (Matrix([
    [0, 1],
    [3, 4]]), Matrix([
    [2],
    [5]]), Matrix([[6, 7]]), Matrix([[8]]))
    >>> assert m == _m(*L) == _m(*_)
    """
    a, b, c, d = [Matrix(i) for i in (a, b, c, d or [0])]
    return Matrix([[a, b], [c, d]])


def _primal_dual(M, factor=True):
    """return primal and dual function and constraints
    assuming that ``M = Matrix([[A, b], [c, d]])`` and the
    function ``c*x - d`` is being minimized with ``Ax >= b``
    for nonnegative values of ``x``. The dual and its
    constraints will be for maximizing `b.T*y - d` subject
    to ``A.T*y <= c.T``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _primal_dual, lpmin, lpmax
    >>> from sympy import Matrix

    The following matrix represents the primal task of
    minimizing x + y + 7 for y >= x + 1 and y >= -2*x + 3.
    The dual task seeks to maximize x + 3*y + 7 with
    2*y - x <= 1 and and x + y <= 1:

    >>> M = Matrix([
    ...     [-1, 1,  1],
    ...     [ 2, 1,  3],
    ...     [ 1, 1, -7]])
    >>> p, d = _primal_dual(M)

    The minimum of the primal and maximum of the dual are the same
    (though they occur at different points):

    >>> lpmin(*p)
    (28/3, {x1: 2/3, x2: 5/3})
    >>> lpmax(*d)
    (28/3, {y1: 1/3, y2: 2/3})

    If the equivalent (but canonical) inequalities are
    desired, leave `factor=True`, otherwise the unmodified
    inequalities for M will be returned.

    >>> m = Matrix([
    ... [-3, -2,  4, -2],
    ... [ 2,  0,  0, -2],
    ... [ 0,  1, -3,  0]])

    >>> _primal_dual(m, False)  # last condition is 2*x1 >= -2
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, 2*x1 >= -2]),
    (-2*y1 - 2*y2,
        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))

    >>> _primal_dual(m)  # condition now x1 >= -1
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, x1 >= -1]),
    (-2*y1 - 2*y2,
        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))

    If you pass the transpose of the matrix, the primal will be
    identified as the standard minimization problem and the
    dual as the standard maximization:

    >>> _primal_dual(m.T)
    ((-2*x1 - 2*x2,
        [-3*x1 + 2*x2 >= 0, -2*x1 >= 1, 4*x1 >= -3]),
    (y2 - 3*y3,
        [-3*y1 - 2*y2 + 4*y3 <= -2, y1 <= -1]))

    A matrix must have some size or else None will be returned for
    the functions:

    >>> _primal_dual(Matrix([[1, 2]]))
    ((x1 - 2, []), (-2, []))

    >>> _primal_dual(Matrix([]))
    ((None, []), (None, []))

    References
    ==========

    .. [1] David Galvin, Relations between Primal and Dual
           www3.nd.edu/~dgalvin1/30210/30210_F07/presentations/dual_opt.pdf
    """
    if not M:
        return (None, []), (None, [])
    if not hasattr(M, "shape"):
        if len(M) not in (3, 4):
            raise ValueError("expecting Matrix or 3 or 4 lists")
        M = _m(*M)
    m, n = [i - 1 for i in M.shape]
    A, b, c, d = _abcd(M)
    d = d[0]
    _ = lambda x: numbered_symbols(x, start=1)
    x = Matrix([i for i, j in zip(_("x"), range(n))])
    yT = Matrix([i for i, j in zip(_("y"), range(m))]).T

    def ineq(L, r, op):
        rv = []
        for r in (op(i, j) for i, j in zip(L, r)):
            if r == True:
                continue
            elif r == False:
                return [False]
            if factor:
                f = factor_terms(r)
                if f.lhs.is_Mul and f.rhs % f.lhs.args[0] == 0:
                    assert len(f.lhs.args) == 2, f.lhs
                    k = f.lhs.args[0]
                    r = r.func(sign(k) * f.lhs.args[1], f.rhs // abs(k))
            rv.append(r)
        return rv

    eq = lambda x, d: x[0] - d if x else -d
    F = eq(c * x, d)
    f = eq(yT * b, d)
    return (F, ineq(A * x, b, Ge)), (f, ineq(yT * A, c, Le))


def _rel_as_nonpos(constr, syms):
    """return `(np, d, aux)` where `np` is a list of nonpositive
    expressions that represent the given constraints (possibly
    rewritten in terms of auxilliary variables) expressible with
    nonnegative symbols, and `d` is a dictionary mapping a given
    symbols to an expression with an auxilliary variable. In some
    cases a symbol will be used as part of the change of variables,
    e.g. x: x - z1 instead of x: z1 - z2.

    If any constraint is False/empty, return None. All variables in
    ``constr`` are assumed to be unbounded unless explicitly indicated
    otherwise with a univariate constraint, e.g. ``x >= 0`` will
    restrict ``x`` to nonnegative values.

    The ``syms`` must be included so all symbols can be given an
    unbounded assumption if they are not otherwise bound with
    univariate conditions like ``x <= 3``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _rel_as_nonpos
    >>> from sympy.abc import x, y
    >>> _rel_as_nonpos([x >= y, x >= 0, y >= 0], (x, y))
    ([-x + y], {}, [])
    >>> _rel_as_nonpos([x >= 3, x <= 5], [x])
    ([_z1 - 2], {x: _z1 + 3}, [_z1])
    >>> _rel_as_nonpos([x <= 5], [x])
    ([], {x: 5 - _z1}, [_z1])
    >>> _rel_as_nonpos([x >= 1], [x])
    ([], {x: _z1 + 1}, [_z1])
    """
    r = {}  # replacements to handle change of variables
    np = []  # nonpositive expressions
    aux = []  # auxilliary symbols added
    ui = numbered_symbols("z", start=1, cls=Dummy)  # auxilliary symbols
    univariate = {}  # {x: interval} for univariate constraints
    unbound = []  # symbols designated as unbound
    syms = set(syms)  # the expected syms of the system

    # separate out univariates
    for i in constr:
        if i == True:
            continue  # ignore
        if i == False:
            return  # no solution
        if i.has(S.Infinity, S.NegativeInfinity):
            raise ValueError("only finite bounds are permitted")
        if isinstance(i, (Le, Ge)):
            i = i.lts - i.gts
            freei = i.free_symbols
            if freei - syms:
                raise ValueError(
                    "unexpected symbol(s) in constraint: %s" % (freei - syms)
                )
            if len(freei) > 1:
                np.append(i)
            elif freei:
                x = freei.pop()
                if x in unbound:
                    continue  # will handle later
                ivl = Le(i, 0, evaluate=False).as_set()
                if x not in univariate:
                    univariate[x] = ivl
                else:
                    univariate[x] &= ivl
            elif i:
                return False
        else:
            raise TypeError(filldedent("""
                only equalities like Eq(x, y) or non-strict
                inequalities like x >= y are allowed in lp, not %s""" % i))

    # introduce auxilliary variables as needed for univariate
    # inequalities
    for x in syms:
        i = univariate.get(x, True)
        if not i:
            return None  # no solution possible
        if i == True:
            unbound.append(x)
            continue
        a, b = i.inf, i.sup
        if a.is_infinite:
            u = next(ui)
            r[x] = b - u
            aux.append(u)
        elif b.is_infinite:
            if a:
                u = next(ui)
                r[x] = a + u
                aux.append(u)
            else:
                # standard nonnegative relationship
                pass
        else:
            u = next(ui)
            aux.append(u)
            # shift so u = x - a => x = u + a
            r[x] = u + a
            # add constraint for u <= b - a
            # since when u = b-a then x = u + a = b - a + a = b:
            # the upper limit for x
            np.append(u - (b - a))

    # make change of variables for unbound variables
    for x in unbound:
        u = next(ui)
        r[x] = u - x  # reusing x
        aux.append(u)

    return np, r, aux


def _lp_matrices(objective, constraints):
    """return A, B, C, D, r, x+X, X for maximizing
    objective = Cx - D with constraints Ax <= B, introducing
    introducing auxilliary variables, X, as necessary to make
    replacements of symbols as given in r, {xi: expression with Xj},
    so all variables in x+X will take on nonnegative values.

    Every univariate condition creates a semi-infinite
    condition, e.g. a single ``x <= 3`` creates the
    interval ``[-oo, 3]`` while ``x <= 3`` and ``x >= 2``
    create an interval ``[2, 3]``. Variables not in a univariate
    expression will take on nonnegative values.
    """

    # sympify input and collect free symbols
    F = sympify(objective)
    np = [sympify(i) for i in constraints]
    syms = set.union(*[i.free_symbols for i in [F] + np], set())

    # change Eq(x, y) to x - y <= 0 and y - x <= 0
    for i in range(len(np)):
        if isinstance(np[i], Eq):
            np[i] = np[i].lhs - np[i].rhs <= 0
            np.append(-np[i].lhs <= 0)

    # convert constraints to nonpositive expressions
    _ = _rel_as_nonpos(np, syms)
    if _ is None:
        raise InfeasibleLPError(filldedent("""
            Inconsistent/False constraint"""))
    np, r, aux = _

    # do change of variables
    F = F.xreplace(r)
    np = [i.xreplace(r) for i in np]

    # convert to matrices
    xx = list(ordered(syms)) + aux
    A, B = linear_eq_to_matrix(np, xx)
    C, D = linear_eq_to_matrix([F], xx)
    return A, B, C, D, r, xx, aux


def _lp(min_max, f, constr):
    """Return the optimization (min or max) of ``f`` with the given
    constraints. All variables are unbounded unless constrained.

    If `min_max` is 'max' then the results corresponding to the
    maximization of ``f`` will be returned, else the minimization.
    The constraints can be given as Le, Ge or Eq expressions.

    Examples
    ========

    >>> from sympy.solvers.simplex import _lp as lp
    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z
    >>> f = x + y - 2*z
    >>> c = [7*x + 4*y - 7*z <= 3, 3*x - y + 10*z <= 6]
    >>> c += [i >= 0 for i in (x, y, z)]
    >>> lp(min, f, c)
    (-6/5, {x: 0, y: 0, z: 3/5})

    By passing max, the maximum value for f under the constraints
    is returned (if possible):

    >>> lp(max, f, c)
    (3/4, {x: 0, y: 3/4, z: 0})

    Constraints that are equalities will require that the solution
    also satisfy them:

    >>> lp(max, f, c + [Eq(y - 9*x, 1)])
    (5/7, {x: 0, y: 1, z: 1/7})

    All symbols are reported, even if they are not in the objective
    function:

    >>> lp(min, x, [y + x >= 3, x >= 0])
    (0, {x: 0, y: 3})
    """
    # get the matrix components for the system expressed
    # in terms of only nonnegative variables
    A, B, C, D, r, xx, aux = _lp_matrices(f, constr)

    how = str(min_max).lower()
    if "max" in how:
        # _simplex minimizes for Ax <= B so we
        # have to change the sign of the function
        # and negate the optimal value returned
        _o, p, d = _simplex(A, B, -C, -D)
        o = -_o
    elif "min" in how:
        o, p, d = _simplex(A, B, C, D)
    else:
        raise ValueError("expecting min or max")

    # restore original variables and remove aux from p
    p = dict(zip(xx, p))
    if r:  # p has original symbols and auxilliary symbols
        # if r has x: x - z1 use values from p to update
        r = {k: v.xreplace(p) for k, v in r.items()}
        # then use the actual value of x (= x - z1) in p
        p.update(r)
        # don't show aux
        p = {k: p[k] for k in ordered(p) if k not in aux}

    # not returning dual since there may be extra constraints
    # when a variable has finite bounds
    return o, p


def lpmin(f, constr):
    """return minimum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmin
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmin(x, [2*x - 3*y >= -1, Eq(x + 3*y, 2), x <= 2*y])
    (1/3, {x: 1/3, y: 5/9})

    Negative values for variables are permitted unless explicitly
    excluding, so minimizing ``x`` for ``x <= 3`` is an
    unbounded problem while the following has a bounded solution:

    >>> lpmin(x, [x >= 0, x <= 3])
    (0, {x: 0})

    Without indicating that ``x`` is nonnegative, there
    is no minimum for this objective:

    >>> lpmin(x, [x <= 3])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    See Also
    ========
    linprog, lpmax
    """
    return _lp(min, f, constr)


def lpmax(f, constr):
    """return maximum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmax
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmax(x, [2*x - 3*y >= -1, Eq(x+ 3*y,2), x <= 2*y])
    (4/5, {x: 4/5, y: 2/5})

    Negative values for variables are permitted unless explicitly
    excluding:

    >>> lpmax(x, [x <= -1])
    (-1, {x: -1})

    If a non-negative constraint is added for x, there is no
    possible solution:

    >>> lpmax(x, [x <= -1, x >= 0])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError: inconsistent/False constraint

    See Also
    ========
    linprog, lpmin
    """
    return _lp(max, f, constr)


def _handle_bounds(bounds):
    # introduce auxiliary variables as needed for univariate
    # inequalities

    def _make_list(length: int, index_value_pairs):
        li = [0] * length
        for idx, val in index_value_pairs:
            li[idx] = val
        return li

    unbound = []
    row = []
    row2 = []
    b_len = len(bounds)
    for x, (a, b) in enumerate(bounds):
        if a is None and b is None:
            unbound.append(x)
        elif a is None:
            # r[x] = b - u
            b_len += 1
            row.append(_make_list(b_len, [(x, 1), (-1, 1)]))
            row.append(_make_list(b_len, [(x, -1), (-1, -1)]))
            row2.extend([[b], [-b]])
        elif b is None:
            if a:
                # r[x] = a + u
                b_len += 1
                row.append(_make_list(b_len, [(x, 1), (-1, -1)]))
                row.append(_make_list(b_len, [(x, -1), (-1, 1)]))
                row2.extend([[a], [-a]])
            else:
                # standard nonnegative relationship
                pass
        else:
            # r[x] = u + a
            b_len += 1
            row.append(_make_list(b_len, [(x, 1), (-1, -1)]))
            row.append(_make_list(b_len, [(x, -1), (-1, 1)]))
            # u <= b - a
            row.append(_make_list(b_len, [(-1, 1)]))
            row2.extend([[a], [-a], [b - a]])

    # make change of variables for unbound variables
    for x in unbound:
        # r[x] = u - v
        b_len += 2
        row.append(_make_list(b_len, [(x, 1), (-1, 1), (-2, -1)]))
        row.append(_make_list(b_len, [(x, -1), (-1, -1), (-2, 1)]))
        row2.extend([[0], [0]])

    return Matrix([r + [0]*(b_len - len(r)) for r in row]), Matrix(row2)


def linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    """Return the minimization of ``c*x`` with the given
    constraints ``A*x <= b`` and ``A_eq*x = b_eq``. Unless bounds
    are given, variables will have nonnegative values in the solution.

    If ``A`` is not given, then the dimension of the system will
    be determined by the length of ``C``.

    By default, all variables will be nonnegative. If ``bounds``
    is given as a single tuple, ``(lo, hi)``, then all variables
    will be constrained to be between ``lo`` and ``hi``. Use
    None for a ``lo`` or ``hi`` if it is unconstrained in the
    negative or positive direction, respectively, e.g.
    ``(None, 0)`` indicates nonpositive values. To set
    individual ranges, pass a list with length equal to the
    number of columns in ``A``, each element being a tuple; if
    only a few variables take on non-default values they can be
    passed as a dictionary with keys giving the corresponding
    column to which the variable is assigned, e.g. ``bounds={2:
    (1, 4)}`` would limit the 3rd variable to have a value in
    range ``[1, 4]``.

    Examples
    ========

    >>> from sympy.solvers.simplex import linprog
    >>> from sympy import symbols, Eq, linear_eq_to_matrix as M, Matrix
    >>> x = x1, x2, x3, x4 = symbols('x1:5')
    >>> X = Matrix(x)
    >>> c, d = M(5*x2 + x3 + 4*x4 - x1, x)
    >>> a, b = M([5*x2 + 2*x3 + 5*x4 - (x1 + 5)], x)
    >>> aeq, beq = M([Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)], x)
    >>> constr = [i <= j for i,j in zip(a*X, b)]
    >>> constr += [Eq(i, j) for i,j in zip(aeq*X, beq)]
    >>> linprog(c, a, b, aeq, beq)
    (9/2, [0, 1/2, 0, 1/2])
    >>> assert all(i.subs(dict(zip(x, _[1]))) for i in constr)

    See Also
    ========
    lpmin, lpmax
    """

    ## the objective
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError("C must be a single row.")

    ## the inequalities
    if not A:
        if b:
            raise ValueError("A and b must both be given")
        # the governing equations will be simple constraints
        # on variables
        A, b = zeros(0, C.cols), zeros(C.cols, 1)
    else:
        A, b = [Matrix(i) for i in (A, b)]

    if A.cols != C.cols:
        raise ValueError("number of columns in A and C must match")

    ## the equalities
    if A_eq is None:
        if not b_eq is None:
            raise ValueError("A_eq and b_eq must both be given")
    else:
        A_eq, b_eq = [Matrix(i) for i in (A_eq, b_eq)]
        # if x == y then x <= y and x >= y (-x <= -y)
        A = A.col_join(A_eq)
        A = A.col_join(-A_eq)
        b = b.col_join(b_eq)
        b = b.col_join(-b_eq)

    if not (bounds is None or bounds == {} or bounds == (0, None)):
        ## the bounds are interpreted
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all(
                type(i) is tuple and len(i) == 2 for i in bounds):
            pass # individual bounds
        elif type(bounds) is dict and all(
                type(i) is tuple and len(i) == 2
                for i in bounds.values()):
            # sparse bounds
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                i, j = db.popitem()
                bounds[i] = j  # IndexError if out-of-bounds indices
        else:
            raise ValueError("unexpected bounds %s" % bounds)
        A_, b_ = _handle_bounds(bounds)
        aux = A_.cols - A.cols
        if A:
            A = Matrix([[A, zeros(A.rows, aux)], [A_]])
            b = b.col_join(b_)
        else:
            A = A_
            b = b_
        C = C.row_join(zeros(1, aux))
    else:
        aux = -A.cols  # set so -aux will give all cols below

    o, p, d = _simplex(A, b, C)
    return o, p[:-aux]  # don't include aux values

def show_linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    from sympy import symbols
    ## the objective
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError("C must be a single row.")

    ## the inequalities
    if not A:
        if b:
            raise ValueError("A and b must both be given")
        # the governing equations will be simple constraints
        # on variables
        A, b = zeros(0, C.cols), zeros(C.cols, 1)
    else:
        A, b = [Matrix(i) for i in (A, b)]

    if A.cols != C.cols:
        raise ValueError("number of columns in A and C must match")

    ## the equalities
    if A_eq is None:
        if not b_eq is None:
            raise ValueError("A_eq and b_eq must both be given")
    else:
        A_eq, b_eq = [Matrix(i) for i in (A_eq, b_eq)]

    if not (bounds is None or bounds == {} or bounds == (0, None)):
        ## the bounds are interpreted
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all(
                type(i) is tuple and len(i) == 2 for i in bounds):
            pass # individual bounds
        elif type(bounds) is dict and all(
                type(i) is tuple and len(i) == 2
                for i in bounds.values()):
            # sparse bounds
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                i, j = db.popitem()
                bounds[i] = j  # IndexError if out-of-bounds indices
        else:
            raise ValueError("unexpected bounds %s" % bounds)

    x = Matrix(symbols('x1:%s' % (A.cols+1)))
    f,c = (C*x)[0], [i<=j for i,j in zip(A*x, b)] + [Eq(i,j) for i,j in zip(A_eq*x,b_eq)]
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            c.append(x[i]>=lo)
        if hi is not None:
            c.append(x[i]<=hi)
    return f,c

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\calculus\optimization.py ===
from __future__ import print_function

from copy import copy

from ..libmp.backend import xrange

class OptimizationMethods(object):
    def __init__(ctx):
        pass

##############
# 1D-SOLVERS #
##############

class Newton:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting points x0 close to the root.

    Pro:

    * converges fast
    * sometimes more robust than secant with bad second starting point

    Contra:

    * converges slowly for multiple roots
    * needs first derivative
    * 2 function evaluations per iteration
    """
    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) == 1:
            self.x0 = x0[0]
        else:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.f = f
        if not 'df' in kwargs:
            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df

    def __iter__(self):
        f = self.f
        df = self.df
        x0 = self.x0
        while True:
            x1 = x0 - f(x0) / df(x0)
            error = abs(x1 - x0)
            x0 = x1
            yield (x1, error)

class Secant:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting points x0 and x1 close to the root.
    x1 defaults to x0 + 0.25.

    Pro:

    * converges fast

    Contra:

    * converges slowly for multiple roots
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) == 1:
            self.x0 = x0[0]
            self.x1 = self.x0 + 0.25
        elif len(x0) == 2:
            self.x0 = x0[0]
            self.x1 = x0[1]
        else:
            raise ValueError('expected 1 or 2 starting points, got %i' % len(x0))
        self.f = f

    def __iter__(self):
        f = self.f
        x0 = self.x0
        x1 = self.x1
        f0 = f(x0)
        while True:
            f1 = f(x1)
            l = x1 - x0
            if not l:
                break
            s = (f1 - f0) / l
            if not s:
                break
            x0, x1 = x1, x1 - f1/s
            f0 = f1
            yield x1, abs(l)

class MNewton:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting point x0 close to the root.
    Uses modified Newton's method that converges fast regardless of the
    multiplicity of the root.

    Pro:

    * converges fast for multiple roots

    Contra:

    * needs first and second derivative of f
    * 3 function evaluations per iteration
    """
    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if not len(x0) == 1:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.x0 = x0[0]
        self.f = f
        if not 'df' in kwargs:
            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df
        if not 'd2f' in kwargs:
            def d2f(x):
                return self.ctx.diff(df, x)
        else:
            d2f = kwargs['df']
        self.d2f = d2f

    def __iter__(self):
        x = self.x0
        f = self.f
        df = self.df
        d2f = self.d2f
        while True:
            prevx = x
            fx = f(x)
            if fx == 0:
                break
            dfx = df(x)
            d2fx = d2f(x)
            # x = x - F(x)/F'(x) with F(x) = f(x)/f'(x)
            x -= fx / (dfx - fx * d2fx / dfx)
            error = abs(x - prevx)
            yield x, error

class Halley:
    """
    1d-solver generating pairs of approximative root and error.

    Needs a starting point x0 close to the root.
    Uses Halley's method with cubic convergence rate.

    Pro:

    * converges even faster the Newton's method
    * useful when computing with *many* digits

    Contra:

    * needs first and second derivative of f
    * 3 function evaluations per iteration
    * converges slowly for multiple roots
    """

    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if not len(x0) == 1:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.x0 = x0[0]
        self.f = f
        if not 'df' in kwargs:
            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df
        if not 'd2f' in kwargs:
            def d2f(x):
                return self.ctx.diff(df, x)
        else:
            d2f = kwargs['df']
        self.d2f = d2f

    def __iter__(self):
        x = self.x0
        f = self.f
        df = self.df
        d2f = self.d2f
        while True:
            prevx = x
            fx = f(x)
            dfx = df(x)
            d2fx = d2f(x)
            x -=  2*fx*dfx / (2*dfx**2 - fx*d2fx)
            error = abs(x - prevx)
            yield x, error

class Muller:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting points x0, x1 and x2 close to the root.
    x1 defaults to x0 + 0.25; x2 to x1 + 0.25.
    Uses Muller's method that converges towards complex roots.

    Pro:

    * converges fast (somewhat faster than secant)
    * can find complex roots

    Contra:

    * converges slowly for multiple roots
    * may have complex values for real starting points and real roots

    http://en.wikipedia.org/wiki/Muller's_method
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) == 1:
            self.x0 = x0[0]
            self.x1 = self.x0 + 0.25
            self.x2 = self.x1 + 0.25
        elif len(x0) == 2:
            self.x0 = x0[0]
            self.x1 = x0[1]
            self.x2 = self.x1 + 0.25
        elif len(x0) == 3:
            self.x0 = x0[0]
            self.x1 = x0[1]
            self.x2 = x0[2]
        else:
            raise ValueError('expected 1, 2 or 3 starting points, got %i'
                             % len(x0))
        self.f = f
        self.verbose = kwargs['verbose']

    def __iter__(self):
        f = self.f
        x0 = self.x0
        x1 = self.x1
        x2 = self.x2
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)
        while True:
            # TODO: maybe refactoring with function for divided differences
            # calculate divided differences
            fx2x1 = (fx1 - fx2) / (x1 - x2)
            fx2x0 = (fx0 - fx2) / (x0 - x2)
            fx1x0 = (fx0 - fx1) / (x0 - x1)
            w = fx2x1 + fx2x0 - fx1x0
            fx2x1x0 = (fx1x0 - fx2x1) / (x0 - x2)
            if w == 0 and fx2x1x0 == 0:
                if self.verbose:
                    print('canceled with')
                    print('x0 =', x0, ', x1 =', x1, 'and x2 =', x2)
                break
            x0 = x1
            fx0 = fx1
            x1 = x2
            fx1 = fx2
            # denominator should be as large as possible => choose sign
            r = self.ctx.sqrt(w**2 - 4*fx2*fx2x1x0)
            if abs(w - r) > abs(w + r):
                r = -r
            x2 -= 2*fx2 / (w + r)
            fx2 = f(x2)
            error = abs(x2 - x1)
            yield x2, error

# TODO: consider raising a ValueError when there's no sign change in a and b
class Bisection:
    """
    1d-solver generating pairs of approximative root and error.

    Uses bisection method to find a root of f in [a, b].
    Might fail for multiple roots (needs sign change).

    Pro:

    * robust and reliable

    Contra:

    * converges slowly
    * needs sign change
    """
    maxsteps = 100

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.f = f
        self.a = x0[0]
        self.b = x0[1]

    def __iter__(self):
        f = self.f
        a = self.a
        b = self.b
        l = b - a
        fb = f(b)
        while True:
            m = self.ctx.ldexp(a + b, -1)
            fm = f(m)
            sign = fm * fb
            if sign < 0:
                a = m
            elif sign > 0:
                b = m
                fb = fm
            else:
                yield m, self.ctx.zero
            l /= 2
            yield (a + b)/2, abs(l)

def _getm(method):
    """
    Return a function to calculate m for Illinois-like methods.
    """
    if method == 'illinois':
        def getm(fz, fb):
            return 0.5
    elif method == 'pegasus':
        def getm(fz, fb):
            return fb/(fb + fz)
    elif method == 'anderson':
        def getm(fz, fb):
            m = 1 - fz/fb
            if m > 0:
                return m
            else:
                return 0.5
    else:
        raise ValueError("method '%s' not recognized" % method)
    return getm

class Illinois:
    """
    1d-solver generating pairs of approximative root and error.

    Uses Illinois method or similar to find a root of f in [a, b].
    Might fail for multiple roots (needs sign change).
    Combines bisect with secant (improved regula falsi).

    The only difference between the methods is the scaling factor m, which is
    used to ensure convergence (you can choose one using the 'method' keyword):

    Illinois method ('illinois'):
        m = 0.5

    Pegasus method ('pegasus'):
        m = fb/(fb + fz)

    Anderson-Bjoerk method ('anderson'):
        m = 1 - fz/fb if positive else 0.5

    Pro:

    * converges very fast

    Contra:

    * has problems with multiple roots
    * needs sign change
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.a = x0[0]
        self.b = x0[1]
        self.f = f
        self.tol = kwargs['tol']
        self.verbose = kwargs['verbose']
        self.method = kwargs.get('method', 'illinois')
        self.getm = _getm(self.method)
        if self.verbose:
            print('using %s method' % self.method)

    def __iter__(self):
        method = self.method
        f = self.f
        a = self.a
        b = self.b
        fa = f(a)
        fb = f(b)
        m = None
        while True:
            l = b - a
            if l == 0:
                break
            s = (fb - fa) / l
            z = a - fa/s
            fz = f(z)
            if abs(fz) < self.tol:
                # TODO: better condition (when f is very flat)
                if self.verbose:
                    print('canceled with z =', z)
                yield z, l
                break
            if fz * fb < 0: # root in [z, b]
                a = b
                fa = fb
                b = z
                fb = fz
            else: # root in [a, z]
                m = self.getm(fz, fb)
                b = z
                fb = fz
                fa = m*fa # scale down to ensure convergence
            if self.verbose and m and not method == 'illinois':
                print('m:', m)
            yield (a + b)/2, abs(l)

def Pegasus(*args, **kwargs):
    """
    1d-solver generating pairs of approximative root and error.

    Uses Pegasus method to find a root of f in [a, b].
    Wrapper for illinois to use method='pegasus'.
    """
    kwargs['method'] = 'pegasus'
    return Illinois(*args, **kwargs)

def Anderson(*args, **kwargs):
    """
    1d-solver generating pairs of approximative root and error.

    Uses Anderson-Bjoerk method to find a root of f in [a, b].
    Wrapper for illinois to use method='pegasus'.
    """
    kwargs['method'] = 'anderson'
    return Illinois(*args, **kwargs)

# TODO: check whether it's possible to combine it with Illinois stuff
class Ridder:
    """
    1d-solver generating pairs of approximative root and error.

    Ridders' method to find a root of f in [a, b].
    Is told to perform as well as Brent's method while being simpler.

    Pro:

    * very fast
    * simpler than Brent's method

    Contra:

    * two function evaluations per step
    * has problems with multiple roots
    * needs sign change

    http://en.wikipedia.org/wiki/Ridders'_method
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        self.f = f
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.x1 = x0[0]
        self.x2 = x0[1]
        self.verbose = kwargs['verbose']
        self.tol = kwargs['tol']

    def __iter__(self):
        ctx = self.ctx
        f = self.f
        x1 = self.x1
        fx1 = f(x1)
        x2 = self.x2
        fx2 = f(x2)
        while True:
            x3 = 0.5*(x1 + x2)
            fx3 = f(x3)
            x4 = x3 + (x3 - x1) * ctx.sign(fx1 - fx2) * fx3 / ctx.sqrt(fx3**2 - fx1*fx2)
            fx4 = f(x4)
            if abs(fx4) < self.tol:
                # TODO: better condition (when f is very flat)
                if self.verbose:
                    print('canceled with f(x4) =', fx4)
                yield x4, abs(x1 - x2)
                break
            if fx4 * fx2 < 0: # root in [x4, x2]
                x1 = x4
                fx1 = fx4
            else: # root in [x1, x4]
                x2 = x4
                fx2 = fx4
            error = abs(x1 - x2)
            yield (x1 + x2)/2, error

class ANewton:
    """
    EXPERIMENTAL 1d-solver generating pairs of approximative root and error.

    Uses Newton's method modified to use Steffensens method when convergence is
    slow. (I.e. for multiple roots.)
    """
    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if not len(x0) == 1:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.x0 = x0[0]
        self.f = f
        if not 'df' in kwargs:
            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df
        def phi(x):
            return x - f(x) / df(x)
        self.phi = phi
        self.verbose = kwargs['verbose']

    def __iter__(self):
        x0 = self.x0
        f = self.f
        df = self.df
        phi = self.phi
        error = 0
        counter = 0
        while True:
            prevx = x0
            try:
                x0 = phi(x0)
            except ZeroDivisionError:
                if self.verbose:
                    print('ZeroDivisionError: canceled with x =', x0)
                break
            preverror = error
            error = abs(prevx - x0)
            # TODO: decide not to use convergence acceleration
            if error and abs(error - preverror) / error < 1:
                if self.verbose:
                    print('converging slowly')
                counter += 1
            if counter >= 3:
                # accelerate convergence
                phi = steffensen(phi)
                counter = 0
                if self.verbose:
                    print('accelerating convergence')
            yield x0, error

# TODO: add Brent

############################
# MULTIDIMENSIONAL SOLVERS #
############################

def jacobian(ctx, f, x):
    """
    Calculate the Jacobian matrix of a function at the point x0.

    This is the first derivative of a vectorial function:

        f : R^m -> R^n with m >= n
    """
    x = ctx.matrix(x)
    h = ctx.sqrt(ctx.eps)
    fx = ctx.matrix(f(*x))
    m = len(fx)
    n = len(x)
    J = ctx.matrix(m, n)
    for j in xrange(n):
        xj = x.copy()
        xj[j] += h
        Jj = (ctx.matrix(f(*xj)) - fx) / h
        for i in xrange(m):
            J[i,j] = Jj[i]
    return J

# TODO: test with user-specified jacobian matrix
class MDNewton:
    """
    Find the root of a vector function numerically using Newton's method.

    f is a vector function representing a nonlinear equation system.

    x0 is the starting point close to the root.

    J is a function returning the Jacobian matrix for a point.

    Supports overdetermined systems.

    Use the 'norm' keyword to specify which norm to use. Defaults to max-norm.
    The function to calculate the Jacobian matrix can be given using the
    keyword 'J'. Otherwise it will be calculated numerically.

    Please note that this method converges only locally. Especially for high-
    dimensional systems it is not trivial to find a good starting point being
    close enough to the root.

    It is recommended to use a faster, low-precision solver from SciPy [1] or
    OpenOpt [2] to get an initial guess. Afterwards you can use this method for
    root-polishing to any precision.

    [1] http://scipy.org

    [2] http://openopt.org/Welcome
    """
    maxsteps = 10

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        self.f = f
        if isinstance(x0, (tuple, list)):
            x0 = ctx.matrix(x0)
        assert x0.cols == 1, 'need a vector'
        self.x0 = x0
        if 'J' in kwargs:
            self.J = kwargs['J']
        else:
            def J(*x):
                return ctx.jacobian(f, x)
            self.J = J
        self.norm = kwargs['norm']
        self.verbose = kwargs['verbose']

    def __iter__(self):
        f = self.f
        x0 = self.x0
        norm = self.norm
        J = self.J
        fx = self.ctx.matrix(f(*x0))
        fxnorm = norm(fx)
        cancel = False
        while not cancel:
            # get direction of descent
            fxn = -fx
            Jx = J(*x0)
            s = self.ctx.lu_solve(Jx, fxn)
            if self.verbose:
                print('Jx:')
                print(Jx)
                print('s:', s)
            # damping step size TODO: better strategy (hard task)
            l = self.ctx.one
            x1 = x0 + s
            while True:
                if x1 == x0:
                    if self.verbose:
                        print("canceled, won't get more excact")
                    cancel = True
                    break
                fx = self.ctx.matrix(f(*x1))
                newnorm = norm(fx)
                if newnorm < fxnorm:
                    # new x accepted
                    fxnorm = newnorm
                    x0 = x1
                    break
                l /= 2
                x1 = x0 + l*s
            yield (x0, fxnorm)

#############
# UTILITIES #
#############

str2solver = {'newton':Newton, 'secant':Secant, 'mnewton':MNewton,
              'halley':Halley, 'muller':Muller, 'bisect':Bisection,
              'illinois':Illinois, 'pegasus':Pegasus, 'anderson':Anderson,
              'ridder':Ridder, 'anewton':ANewton, 'mdnewton':MDNewton}

def findroot(ctx, f, x0, solver='secant', tol=None, verbose=False, verify=True, **kwargs):
    r"""
    Find an approximate solution to `f(x) = 0`, using *x0* as starting point or
    interval for *x*.

    Multidimensional overdetermined systems are supported.
    You can specify them using a function or a list of functions.

    Mathematically speaking, this function returns `x` such that
    `|f(x)|^2 \leq \mathrm{tol}` is true within the current working precision.
    If the computed value does not meet this criterion, an exception is raised.
    This exception can be disabled with *verify=False*.

    For interval arithmetic (``iv.findroot()``), please note that
    the returned interval ``x`` is not guaranteed to contain `f(x)=0`!
    It is only some `x` for which `|f(x)|^2 \leq \mathrm{tol}` certainly holds
    regardless of numerical error. This may be improved in the future.

    **Arguments**

    *f*
        one dimensional function
    *x0*
        starting point, several starting points or interval (depends on solver)
    *tol*
        the returned solution has an error smaller than this
    *verbose*
        print additional information for each iteration if true
    *verify*
        verify the solution and raise a ValueError if `|f(x)|^2 > \mathrm{tol}`
    *solver*
        a generator for *f* and *x0* returning approximative solution and error
    *maxsteps*
        after how many steps the solver will cancel
    *df*
        first derivative of *f* (used by some solvers)
    *d2f*
        second derivative of *f* (used by some solvers)
    *multidimensional*
        force multidimensional solving
    *J*
        Jacobian matrix of *f* (used by multidimensional solvers)
    *norm*
        used vector norm (used by multidimensional solvers)

    solver has to be callable with ``(f, x0, **kwargs)`` and return an generator
    yielding pairs of approximative solution and estimated error (which is
    expected to be positive).
    You can use the following string aliases:
    'secant', 'mnewton', 'halley', 'muller', 'illinois', 'pegasus', 'anderson',
    'ridder', 'anewton', 'bisect'

    See mpmath.calculus.optimization for their documentation.

    **Examples**

    The function :func:`~mpmath.findroot` locates a root of a given function using the
    secant method by default. A simple example use of the secant method is to
    compute `\pi` as the root of `\sin x` closest to `x_0 = 3`::

        >>> from mpmath import *
        >>> mp.dps = 30; mp.pretty = True
        >>> findroot(sin, 3)
        3.14159265358979323846264338328

    The secant method can be used to find complex roots of analytic functions,
    although it must in that case generally be given a nonreal starting value
    (or else it will never leave the real line)::

        >>> mp.dps = 15
        >>> findroot(lambda x: x**3 + 2*x + 1, j)
        (0.226698825758202 + 1.46771150871022j)

    A nice application is to compute nontrivial roots of the Riemann zeta
    function with many digits (good initial values are needed for convergence)::

        >>> mp.dps = 30
        >>> findroot(zeta, 0.5+14j)
        (0.5 + 14.1347251417346937904572519836j)

    The secant method can also be used as an optimization algorithm, by passing
    it a derivative of a function. The following example locates the positive
    minimum of the gamma function::

        >>> mp.dps = 20
        >>> findroot(lambda x: diff(gamma, x), 1)
        1.4616321449683623413

    Finally, a useful application is to compute inverse functions, such as the
    Lambert W function which is the inverse of `w e^w`, given the first
    term of the solution's asymptotic expansion as the initial value. In basic
    cases, this gives identical results to mpmath's built-in ``lambertw``
    function::

        >>> def lambert(x):
        ...     return findroot(lambda w: w*exp(w) - x, log(1+x))
        ...
        >>> mp.dps = 15
        >>> lambert(1); lambertw(1)
        0.567143290409784
        0.567143290409784
        >>> lambert(1000); lambert(1000)
        5.2496028524016
        5.2496028524016

    Multidimensional functions are also supported::

        >>> f = [lambda x1, x2: x1**2 + x2,
        ...      lambda x1, x2: 5*x1**2 - 3*x1 + 2*x2 - 3]
        >>> findroot(f, (0, 0))
        [-0.618033988749895]
        [-0.381966011250105]
        >>> findroot(f, (10, 10))
        [ 1.61803398874989]
        [-2.61803398874989]

    You can verify this by solving the system manually.

    Please note that the following (more general) syntax also works::

        >>> def f(x1, x2):
        ...     return x1**2 + x2, 5*x1**2 - 3*x1 + 2*x2 - 3
        ...
        >>> findroot(f, (0, 0))
        [-0.618033988749895]
        [-0.381966011250105]


    **Multiple roots**

    For multiple roots all methods of the Newtonian family (including secant)
    converge slowly. Consider this example::

        >>> f = lambda x: (x - 1)**99
        >>> findroot(f, 0.9, verify=False)
        0.918073542444929

    Even for a very close starting point the secant method converges very
    slowly. Use ``verbose=True`` to illustrate this.

    It is possible to modify Newton's method to make it converge regardless of
    the root's multiplicity::

        >>> findroot(f, -10, solver='mnewton')
        1.0

    This variant uses the first and second derivative of the function, which is
    not very efficient.

    Alternatively you can use an experimental Newtonian solver that keeps track
    of the speed of convergence and accelerates it using Steffensen's method if
    necessary::

        >>> findroot(f, -10, solver='anewton', verbose=True)
        x:     -9.88888888888888888889
        error: 0.111111111111111111111
        converging slowly
        x:     -9.77890011223344556678
        error: 0.10998877665544332211
        converging slowly
        x:     -9.67002233332199662166
        error: 0.108877778911448945119
        converging slowly
        accelerating convergence
        x:     -9.5622443299551077669
        error: 0.107778003366888854764
        converging slowly
        x:     0.99999999999999999214
        error: 10.562244329955107759
        x:     1.0
        error: 7.8598304758094664213e-18
        ZeroDivisionError: canceled with x = 1.0
        1.0

    **Complex roots**

    For complex roots it's recommended to use Muller's method as it converges
    even for real starting points very fast::

        >>> findroot(lambda x: x**4 + x + 1, (0, 1, 2), solver='muller')
        (0.727136084491197 + 0.934099289460529j)


    **Intersection methods**

    When you need to find a root in a known interval, it's highly recommended to
    use an intersection-based solver like ``'anderson'`` or ``'ridder'``.
    Usually they converge faster and more reliable. They have however problems
    with multiple roots and usually need a sign change to find a root::

        >>> findroot(lambda x: x**3, (-1, 1), solver='anderson')
        0.0

    Be careful with symmetric functions::

        >>> findroot(lambda x: x**2, (-1, 1), solver='anderson') #doctest:+ELLIPSIS
        Traceback (most recent call last):
          ...
        ZeroDivisionError

    It fails even for better starting points, because there is no sign change::

        >>> findroot(lambda x: x**2, (-1, .5), solver='anderson')
        Traceback (most recent call last):
          ...
        ValueError: Could not find root within given tolerance. (1.0 > 2.16840434497100886801e-19)
        Try another starting point or tweak arguments.

    """
    prec = ctx.prec
    try:
        ctx.prec += 20

        # initialize arguments
        if tol is None:
            tol = ctx.eps * 2**10

        kwargs['verbose'] = kwargs.get('verbose', verbose)

        if 'd1f' in kwargs:
            kwargs['df'] = kwargs['d1f']

        kwargs['tol'] = tol
        if isinstance(x0, (list, tuple)):
            x0 = [ctx.convert(x) for x in x0]
        else:
            x0 = [ctx.convert(x0)]

        if isinstance(solver, str):
            try:
                solver = str2solver[solver]
            except KeyError:
                raise ValueError('could not recognize solver')

        # accept list of functions
        if isinstance(f, (list, tuple)):
            f2 = copy(f)
            def tmp(*args):
                return [fn(*args) for fn in f2]
            f = tmp

        # detect multidimensional functions
        try:
            fx = f(*x0)
            multidimensional = isinstance(fx, (list, tuple, ctx.matrix))
        except TypeError:
            fx = f(x0[0])
            multidimensional = False
        if 'multidimensional' in kwargs:
            multidimensional = kwargs['multidimensional']
        if multidimensional:
            # only one multidimensional solver available at the moment
            solver = MDNewton
            if not 'norm' in kwargs:
                norm = lambda x: ctx.norm(x, 'inf')
                kwargs['norm'] = norm
            else:
                norm = kwargs['norm']
        else:
            norm = abs

        # happily return starting point if it's a root
        if norm(fx) == 0:
            if multidimensional:
                return ctx.matrix(x0)
            else:
                return x0[0]

        # use solver
        iterations = solver(ctx, f, x0, **kwargs)
        if 'maxsteps' in kwargs:
            maxsteps = kwargs['maxsteps']
        else:
            maxsteps = iterations.maxsteps
        i = 0
        for x, error in iterations:
            if verbose:
                print('x:    ', x)
                print('error:', error)
            i += 1
            if error < tol * max(1, norm(x)) or i >= maxsteps:
                break
        else:
            if not i:
                raise ValueError('Could not find root using the given solver.\n'
                                 'Try another starting point or tweak arguments.')
        if not isinstance(x, (list, tuple, ctx.matrix)):
            xl = [x]
        else:
            xl = x
        if verify and norm(f(*xl))**2 > tol: # TODO: better condition?
            raise ValueError('Could not find root within given tolerance. '
                             '(%s > %s)\n'
                             'Try another starting point or tweak arguments.'
                             % (norm(f(*xl))**2, tol))
        return x
    finally:
        ctx.prec = prec


def multiplicity(ctx, f, root, tol=None, maxsteps=10, **kwargs):
    """
    Return the multiplicity of a given root of f.

    Internally, numerical derivatives are used. This might be inefficient for
    higher order derviatives. Due to this, ``multiplicity`` cancels after
    evaluating 10 derivatives by default. You can be specify the n-th derivative
    using the dnf keyword.

    >>> from mpmath import *
    >>> multiplicity(lambda x: sin(x) - 1, pi/2)
    2

    """
    if tol is None:
        tol = ctx.eps ** 0.8
    kwargs['d0f'] = f
    for i in xrange(maxsteps):
        dfstr = 'd' + str(i) + 'f'
        if dfstr in kwargs:
            df = kwargs[dfstr]
        else:
            df = lambda x: ctx.diff(f, x, i)
        if not abs(df(root)) < tol:
            break
    return i

def steffensen(f):
    """
    linear convergent function -> quadratic convergent function

    Steffensen's method for quadratic convergence of a linear converging
    sequence.
    Don not use it for higher rates of convergence.
    It may even work for divergent sequences.

    Definition:
    F(x) = (x*f(f(x)) - f(x)**2) / (f(f(x)) - 2*f(x) + x)

    Example
    .......

    You can use Steffensen's method to accelerate a fixpoint iteration of linear
    (or less) convergence.

    x* is a fixpoint of the iteration x_{k+1} = phi(x_k) if x* = phi(x*). For
    phi(x) = x**2 there are two fixpoints: 0 and 1.

    Let's try Steffensen's method:

    >>> f = lambda x: x**2
    >>> from mpmath.calculus.optimization import steffensen
    >>> F = steffensen(f)
    >>> for x in [0.5, 0.9, 2.0]:
    ...     fx = Fx = x
    ...     for i in xrange(9):
    ...         try:
    ...             fx = f(fx)
    ...         except OverflowError:
    ...             pass
    ...         try:
    ...             Fx = F(Fx)
    ...         except ZeroDivisionError:
    ...             pass
    ...         print('%20g  %20g' % (fx, Fx))
                    0.25                  -0.5
                  0.0625                   0.1
              0.00390625            -0.0011236
             1.52588e-05           1.41691e-09
             2.32831e-10          -2.84465e-27
             5.42101e-20           2.30189e-80
             2.93874e-39          -1.2197e-239
             8.63617e-78                     0
            7.45834e-155                     0
                    0.81               1.02676
                  0.6561               1.00134
                0.430467                     1
                0.185302                     1
               0.0343368                     1
              0.00117902                     1
             1.39008e-06                     1
             1.93233e-12                     1
             3.73392e-24                     1
                       4                   1.6
                      16                1.2962
                     256               1.10194
                   65536               1.01659
             4.29497e+09               1.00053
             1.84467e+19                     1
             3.40282e+38                     1
             1.15792e+77                     1
            1.34078e+154                     1

    Unmodified, the iteration converges only towards 0. Modified it converges
    not only much faster, it converges even to the repelling fixpoint 1.
    """
    def F(x):
        fx = f(x)
        ffx = f(fx)
        return (x*ffx - fx**2) / (ffx - 2*fx + x)
    return F

OptimizationMethods.jacobian = jacobian
OptimizationMethods.findroot = findroot
OptimizationMethods.multiplicity = multiplicity

if __name__ == '__main__':
    import doctest
    doctest.testmod()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\groupby\grouper.py ===
"""
Provide user facing operators for doing the split part of the
split-apply-combine paradigm.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    final,
)
import warnings

import numpy as np

from pandas._config import (
    using_copy_on_write,
    warn_copy_on_write,
)

from pandas._libs import lib
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.core import algorithms
from pandas.core.arrays import (
    Categorical,
    ExtensionArray,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby
from pandas.core.indexes.api import (
    CategoricalIndex,
    Index,
    MultiIndex,
)
from pandas.core.series import Series

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

    from pandas._typing import (
        ArrayLike,
        Axis,
        NDFrameT,
        npt,
    )

    from pandas.core.generic import NDFrame


class Grouper:
    """
    A Grouper allows the user to specify a groupby instruction for an object.

    This specification will select a column via the key parameter, or if the
    level and/or axis parameters are given, a level of the index of the target
    object.

    If `axis` and/or `level` are passed as keywords to both `Grouper` and
    `groupby`, the values passed to `Grouper` take precedence.

    Parameters
    ----------
    key : str, defaults to None
        Groupby key, which selects the grouping column of the target.
    level : name/number, defaults to None
        The level for the target index.
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.
    axis : str, int, defaults to 0
        Number/name of the axis.
    sort : bool, default to False
        Whether to sort the resulting labels.
    closed : {'left' or 'right'}
        Closed end of interval. Only when `freq` parameter is passed.
    label : {'left' or 'right'}
        Interval boundary to use for labeling.
        Only when `freq` parameter is passed.
    convention : {'start', 'end', 'e', 's'}
        If grouper is PeriodIndex and `freq` parameter is passed.

    origin : Timestamp or str, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If string, must be one of the following:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries

        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day

        .. versionadded:: 1.3.0

    offset : Timedelta or str, default is None
        An offset timedelta added to the origin.

    dropna : bool, default True
        If True, and if group keys contain NA values, NA values together with
        row/column will be dropped. If False, NA values will also be treated as
        the key in groups.

    Returns
    -------
    Grouper or pandas.api.typing.TimeGrouper
        A TimeGrouper is returned if ``freq`` is not ``None``. Otherwise, a Grouper
        is returned.

    Examples
    --------
    ``df.groupby(pd.Grouper(key="Animal"))`` is equivalent to ``df.groupby('Animal')``

    >>> df = pd.DataFrame(
    ...     {
    ...         "Animal": ["Falcon", "Parrot", "Falcon", "Falcon", "Parrot"],
    ...         "Speed": [100, 5, 200, 300, 15],
    ...     }
    ... )
    >>> df
       Animal  Speed
    0  Falcon    100
    1  Parrot      5
    2  Falcon    200
    3  Falcon    300
    4  Parrot     15
    >>> df.groupby(pd.Grouper(key="Animal")).mean()
            Speed
    Animal
    Falcon  200.0
    Parrot   10.0

    Specify a resample operation on the column 'Publish date'

    >>> df = pd.DataFrame(
    ...    {
    ...        "Publish date": [
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-09"),
    ...             pd.Timestamp("2000-01-16")
    ...         ],
    ...         "ID": [0, 1, 2, 3],
    ...         "Price": [10, 20, 30, 40]
    ...     }
    ... )
    >>> df
      Publish date  ID  Price
    0   2000-01-02   0     10
    1   2000-01-02   1     20
    2   2000-01-09   2     30
    3   2000-01-16   3     40
    >>> df.groupby(pd.Grouper(key="Publish date", freq="1W")).mean()
                   ID  Price
    Publish date
    2000-01-02    0.5   15.0
    2000-01-09    2.0   30.0
    2000-01-16    3.0   40.0

    If you want to adjust the start of the bins based on a fixed timestamp:

    >>> start, end = '2000-10-01 23:30:00', '2000-10-02 00:30:00'
    >>> rng = pd.date_range(start, end, freq='7min')
    >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)
    >>> ts
    2000-10-01 23:30:00     0
    2000-10-01 23:37:00     3
    2000-10-01 23:44:00     6
    2000-10-01 23:51:00     9
    2000-10-01 23:58:00    12
    2000-10-02 00:05:00    15
    2000-10-02 00:12:00    18
    2000-10-02 00:19:00    21
    2000-10-02 00:26:00    24
    Freq: 7min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min')).sum()
    2000-10-01 23:14:00     0
    2000-10-01 23:31:00     9
    2000-10-01 23:48:00    21
    2000-10-02 00:05:00    54
    2000-10-02 00:22:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', origin='epoch')).sum()
    2000-10-01 23:18:00     0
    2000-10-01 23:35:00    18
    2000-10-01 23:52:00    27
    2000-10-02 00:09:00    39
    2000-10-02 00:26:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', origin='2000-01-01')).sum()
    2000-10-01 23:24:00     3
    2000-10-01 23:41:00    15
    2000-10-01 23:58:00    45
    2000-10-02 00:15:00    45
    Freq: 17min, dtype: int64

    If you want to adjust the start of the bins with an `offset` Timedelta, the two
    following lines are equivalent:

    >>> ts.groupby(pd.Grouper(freq='17min', origin='start')).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq='17min', offset='23h30min')).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    To replace the use of the deprecated `base` argument, you can now use `offset`,
    in this example it is equivalent to have `base=2`:

    >>> ts.groupby(pd.Grouper(freq='17min', offset='2min')).sum()
    2000-10-01 23:16:00     0
    2000-10-01 23:33:00     9
    2000-10-01 23:50:00    36
    2000-10-02 00:07:00    39
    2000-10-02 00:24:00    24
    Freq: 17min, dtype: int64
    """

    sort: bool
    dropna: bool
    _gpr_index: Index | None
    _grouper: Index | None

    _attributes: tuple[str, ...] = ("key", "level", "freq", "axis", "sort", "dropna")

    def __new__(cls, *args, **kwargs):
        if kwargs.get("freq") is not None:
            from pandas.core.resample import TimeGrouper

            cls = TimeGrouper
        return super().__new__(cls)

    def __init__(
        self,
        key=None,
        level=None,
        freq=None,
        axis: Axis | lib.NoDefault = lib.no_default,
        sort: bool = False,
        dropna: bool = True,
    ) -> None:
        if type(self) is Grouper:
            # i.e. not TimeGrouper
            if axis is not lib.no_default:
                warnings.warn(
                    "Grouper axis keyword is deprecated and will be removed in a "
                    "future version. To group on axis=1, use obj.T.groupby(...) "
                    "instead",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            else:
                axis = 0
        if axis is lib.no_default:
            axis = 0

        self.key = key
        self.level = level
        self.freq = freq
        self.axis = axis
        self.sort = sort
        self.dropna = dropna

        self._grouper_deprecated = None
        self._indexer_deprecated: npt.NDArray[np.intp] | None = None
        self._obj_deprecated = None
        self._gpr_index = None
        self.binner = None
        self._grouper = None
        self._indexer: npt.NDArray[np.intp] | None = None

    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True
    ) -> tuple[ops.BaseGrouper, NDFrameT]:
        """
        Parameters
        ----------
        obj : Series or DataFrame
        validate : bool, default True
            if True, validate the grouper

        Returns
        -------
        a tuple of grouper, obj (possibly sorted)
        """
        obj, _, _ = self._set_grouper(obj)
        grouper, _, obj = get_grouper(
            obj,
            [self.key],
            axis=self.axis,
            level=self.level,
            sort=self.sort,
            validate=validate,
            dropna=self.dropna,
        )
        # Without setting this, subsequent lookups to .groups raise
        # error: Incompatible types in assignment (expression has type "BaseGrouper",
        # variable has type "None")
        self._grouper_deprecated = grouper  # type: ignore[assignment]

        return grouper, obj

    def _set_grouper(
        self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None
    ) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        """
        given an object and the specifications, setup the internal grouper
        for this particular specification

        Parameters
        ----------
        obj : Series or DataFrame
        sort : bool, default False
            whether the resulting grouper should be sorted
        gpr_index : Index or None, default None

        Returns
        -------
        NDFrame
        Index
        np.ndarray[np.intp] | None
        """
        assert obj is not None

        if self.key is not None and self.level is not None:
            raise ValueError("The Grouper cannot specify both a key and a level!")

        # Keep self._grouper value before overriding
        if self._grouper is None:
            # TODO: What are we assuming about subsequent calls?
            self._grouper = gpr_index
            self._indexer = self._indexer_deprecated

        # the key must be a valid info item
        if self.key is not None:
            key = self.key
            # The 'on' is already defined
            if getattr(gpr_index, "name", None) == key and isinstance(obj, Series):
                # Sometimes self._grouper will have been resorted while
                # obj has not. In this case there is a mismatch when we
                # call self._grouper.take(obj.index) so we need to undo the sorting
                # before we call _grouper.take.
                assert self._grouper is not None
                if self._indexer is not None:
                    reverse_indexer = self._indexer.argsort()
                    unsorted_ax = self._grouper.take(reverse_indexer)
                    ax = unsorted_ax.take(obj.index)
                else:
                    ax = self._grouper.take(obj.index)
            else:
                if key not in obj._info_axis:
                    raise KeyError(f"The grouper name {key} is not found")
                ax = Index(obj[key], name=key)

        else:
            ax = obj._get_axis(self.axis)
            if self.level is not None:
                level = self.level

                # if a level is given it must be a mi level or
                # equivalent to the axis name
                if isinstance(ax, MultiIndex):
                    level = ax._get_level_number(level)
                    ax = Index(ax._get_level_values(level), name=ax.names[level])

                else:
                    if level not in (0, ax.name):
                        raise ValueError(f"The level {level} is not valid")

        # possibly sort
        indexer: npt.NDArray[np.intp] | None = None
        if (self.sort or sort) and not ax.is_monotonic_increasing:
            # use stable sort to support first, last, nth
            # TODO: why does putting na_position="first" fix datetimelike cases?
            indexer = self._indexer_deprecated = ax.array.argsort(
                kind="mergesort", na_position="first"
            )
            ax = ax.take(indexer)
            obj = obj.take(indexer, axis=self.axis)

        # error: Incompatible types in assignment (expression has type
        # "NDFrameT", variable has type "None")
        self._obj_deprecated = obj  # type: ignore[assignment]
        self._gpr_index = ax
        return obj, ax, indexer

    @final
    @property
    def ax(self) -> Index:
        warnings.warn(
            f"{type(self).__name__}.ax is deprecated and will be removed in a "
            "future version. Use Resampler.ax instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        index = self._gpr_index
        if index is None:
            raise ValueError("_set_grouper must be called before ax is accessed")
        return index

    @final
    @property
    def indexer(self):
        warnings.warn(
            f"{type(self).__name__}.indexer is deprecated and will be removed "
            "in a future version. Use Resampler.indexer instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._indexer_deprecated

    @final
    @property
    def obj(self):
        # TODO(3.0): enforcing these deprecations on Grouper should close
        #  GH#25564, GH#41930
        warnings.warn(
            f"{type(self).__name__}.obj is deprecated and will be removed "
            "in a future version. Use GroupBy.indexer instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._obj_deprecated

    @final
    @property
    def grouper(self):
        warnings.warn(
            f"{type(self).__name__}.grouper is deprecated and will be removed "
            "in a future version. Use GroupBy.grouper instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._grouper_deprecated

    @final
    @property
    def groups(self):
        warnings.warn(
            f"{type(self).__name__}.groups is deprecated and will be removed "
            "in a future version. Use GroupBy.groups instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        # error: "None" has no attribute "groups"
        return self._grouper_deprecated.groups  # type: ignore[attr-defined]

    @final
    def __repr__(self) -> str:
        attrs_list = (
            f"{attr_name}={repr(getattr(self, attr_name))}"
            for attr_name in self._attributes
            if getattr(self, attr_name) is not None
        )
        attrs = ", ".join(attrs_list)
        cls_name = type(self).__name__
        return f"{cls_name}({attrs})"


@final
class Grouping:
    """
    Holds the grouping information for a single key

    Parameters
    ----------
    index : Index
    grouper :
    obj : DataFrame or Series
    name : Label
    level :
    observed : bool, default False
        If we are a Categorical, use the observed values
    in_axis : if the Grouping is a column in self.obj and hence among
        Groupby.exclusions list
    dropna : bool, default True
        Whether to drop NA groups.
    uniques : Array-like, optional
        When specified, will be used for unique values. Enables including empty groups
        in the result for a BinGrouper. Must not contain duplicates.

    Attributes
    -------
    indices : dict
        Mapping of {group -> index_list}
    codes : ndarray
        Group codes
    group_index : Index or None
        unique groups
    groups : dict
        Mapping of {group -> label_list}
    """

    _codes: npt.NDArray[np.signedinteger] | None = None
    _all_grouper: Categorical | None
    _orig_cats: Index | None
    _index: Index

    def __init__(
        self,
        index: Index,
        grouper=None,
        obj: NDFrame | None = None,
        level=None,
        sort: bool = True,
        observed: bool = False,
        in_axis: bool = False,
        dropna: bool = True,
        uniques: ArrayLike | None = None,
    ) -> None:
        self.level = level
        self._orig_grouper = grouper
        grouping_vector = _convert_grouper(index, grouper)
        self._all_grouper = None
        self._orig_cats = None
        self._index = index
        self._sort = sort
        self.obj = obj
        self._observed = observed
        self.in_axis = in_axis
        self._dropna = dropna
        self._uniques = uniques

        # we have a single grouper which may be a myriad of things,
        # some of which are dependent on the passing in level

        ilevel = self._ilevel
        if ilevel is not None:
            # In extant tests, the new self.grouping_vector matches
            #  `index.get_level_values(ilevel)` whenever
            #  mapper is None and isinstance(index, MultiIndex)
            if isinstance(index, MultiIndex):
                index_level = index.get_level_values(ilevel)
            else:
                index_level = index

            if grouping_vector is None:
                grouping_vector = index_level
            else:
                mapper = grouping_vector
                grouping_vector = index_level.map(mapper)

        # a passed Grouper like, directly get the grouper in the same way
        # as single grouper groupby, use the group_info to get codes
        elif isinstance(grouping_vector, Grouper):
            # get the new grouper; we already have disambiguated
            # what key/level refer to exactly, don't need to
            # check again as we have by this point converted these
            # to an actual value (rather than a pd.Grouper)
            assert self.obj is not None  # for mypy
            newgrouper, newobj = grouping_vector._get_grouper(self.obj, validate=False)
            self.obj = newobj

            if isinstance(newgrouper, ops.BinGrouper):
                # TODO: can we unwrap this and get a tighter typing
                #  for self.grouping_vector?
                grouping_vector = newgrouper
            else:
                # ops.BaseGrouper
                # TODO: 2023-02-03 no test cases with len(newgrouper.groupings) > 1.
                #  If that were to occur, would we be throwing out information?
                # error: Cannot determine type of "grouping_vector"  [has-type]
                ng = newgrouper.groupings[0].grouping_vector  # type: ignore[has-type]
                # use Index instead of ndarray so we can recover the name
                grouping_vector = Index(ng, name=newgrouper.result_index.name)

        elif not isinstance(
            grouping_vector, (Series, Index, ExtensionArray, np.ndarray)
        ):
            # no level passed
            if getattr(grouping_vector, "ndim", 1) != 1:
                t = str(type(grouping_vector))
                raise ValueError(f"Grouper for '{t}' not 1-dimensional")

            grouping_vector = index.map(grouping_vector)

            if not (
                hasattr(grouping_vector, "__len__")
                and len(grouping_vector) == len(index)
            ):
                grper = pprint_thing(grouping_vector)
                errmsg = (
                    "Grouper result violates len(labels) == "
                    f"len(data)\nresult: {grper}"
                )
                raise AssertionError(errmsg)

        if isinstance(grouping_vector, np.ndarray):
            if grouping_vector.dtype.kind in "mM":
                # if we have a date/time-like grouper, make sure that we have
                # Timestamps like
                # TODO 2022-10-08 we only have one test that gets here and
                #  values are already in nanoseconds in that case.
                grouping_vector = Series(grouping_vector).to_numpy()
        elif isinstance(getattr(grouping_vector, "dtype", None), CategoricalDtype):
            # a passed Categorical
            self._orig_cats = grouping_vector.categories
            grouping_vector, self._all_grouper = recode_for_groupby(
                grouping_vector, sort, observed
            )

        self.grouping_vector = grouping_vector

    def __repr__(self) -> str:
        return f"Grouping({self.name})"

    def __iter__(self) -> Iterator:
        return iter(self.indices)

    @cache_readonly
    def _passed_categorical(self) -> bool:
        dtype = getattr(self.grouping_vector, "dtype", None)
        return isinstance(dtype, CategoricalDtype)

    @cache_readonly
    def name(self) -> Hashable:
        ilevel = self._ilevel
        if ilevel is not None:
            return self._index.names[ilevel]

        if isinstance(self._orig_grouper, (Index, Series)):
            return self._orig_grouper.name

        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.result_index.name

        elif isinstance(self.grouping_vector, Index):
            return self.grouping_vector.name

        # otherwise we have ndarray or ExtensionArray -> no name
        return None

    @cache_readonly
    def _ilevel(self) -> int | None:
        """
        If necessary, converted index level name to index level position.
        """
        level = self.level
        if level is None:
            return None
        if not isinstance(level, int):
            index = self._index
            if level not in index.names:
                raise AssertionError(f"Level {level} not in index")
            return index.names.index(level)
        return level

    @property
    def ngroups(self) -> int:
        return len(self._group_index)

    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        # we have a list of groupers
        if isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.indices

        values = Categorical(self.grouping_vector)
        return values._reverse_indexer()

    @property
    def codes(self) -> npt.NDArray[np.signedinteger]:
        return self._codes_and_uniques[0]

    @cache_readonly
    def _group_arraylike(self) -> ArrayLike:
        """
        Analogous to result_index, but holding an ArrayLike to ensure
        we can retain ExtensionDtypes.
        """
        if self._all_grouper is not None:
            # retain dtype for categories, including unobserved ones
            return self._result_index._values

        elif self._passed_categorical:
            return self._group_index._values

        return self._codes_and_uniques[1]

    @property
    def group_arraylike(self) -> ArrayLike:
        """
        Analogous to result_index, but holding an ArrayLike to ensure
        we can retain ExtensionDtypes.
        """
        warnings.warn(
            "group_arraylike is deprecated and will be removed in a future "
            "version of pandas",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._group_arraylike

    @cache_readonly
    def _result_index(self) -> Index:
        # result_index retains dtype for categories, including unobserved ones,
        #  which group_index does not
        if self._all_grouper is not None:
            group_idx = self._group_index
            assert isinstance(group_idx, CategoricalIndex)
            cats = self._orig_cats
            # set_categories is dynamically added
            return group_idx.set_categories(cats)  # type: ignore[attr-defined]
        return self._group_index

    @property
    def result_index(self) -> Index:
        warnings.warn(
            "result_index is deprecated and will be removed in a future "
            "version of pandas",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._result_index

    @cache_readonly
    def _group_index(self) -> Index:
        codes, uniques = self._codes_and_uniques
        if not self._dropna and self._passed_categorical:
            assert isinstance(uniques, Categorical)
            if self._sort and (codes == len(uniques)).any():
                # Add NA value on the end when sorting
                uniques = Categorical.from_codes(
                    np.append(uniques.codes, [-1]), uniques.categories, validate=False
                )
            elif len(codes) > 0:
                # Need to determine proper placement of NA value when not sorting
                cat = self.grouping_vector
                na_idx = (cat.codes < 0).argmax()
                if cat.codes[na_idx] < 0:
                    # count number of unique codes that comes before the nan value
                    na_unique_idx = algorithms.nunique_ints(cat.codes[:na_idx])
                    new_codes = np.insert(uniques.codes, na_unique_idx, -1)
                    uniques = Categorical.from_codes(
                        new_codes, uniques.categories, validate=False
                    )
        return Index._with_infer(uniques, name=self.name)

    @property
    def group_index(self) -> Index:
        warnings.warn(
            "group_index is deprecated and will be removed in a future "
            "version of pandas",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._group_index

    @cache_readonly
    def _codes_and_uniques(self) -> tuple[npt.NDArray[np.signedinteger], ArrayLike]:
        uniques: ArrayLike
        if self._passed_categorical:
            # we make a CategoricalIndex out of the cat grouper
            # preserving the categories / ordered attributes;
            # doesn't (yet - GH#46909) handle dropna=False
            cat = self.grouping_vector
            categories = cat.categories

            if self._observed:
                ucodes = algorithms.unique1d(cat.codes)
                ucodes = ucodes[ucodes != -1]
                if self._sort:
                    ucodes = np.sort(ucodes)
            else:
                ucodes = np.arange(len(categories))

            uniques = Categorical.from_codes(
                codes=ucodes, categories=categories, ordered=cat.ordered, validate=False
            )

            codes = cat.codes
            if not self._dropna:
                na_mask = codes < 0
                if np.any(na_mask):
                    if self._sort:
                        # Replace NA codes with `largest code + 1`
                        na_code = len(categories)
                        codes = np.where(na_mask, na_code, codes)
                    else:
                        # Insert NA code into the codes based on first appearance
                        # A negative code must exist, no need to check codes[na_idx] < 0
                        na_idx = na_mask.argmax()
                        # count number of unique codes that comes before the nan value
                        na_code = algorithms.nunique_ints(codes[:na_idx])
                        codes = np.where(codes >= na_code, codes + 1, codes)
                        codes = np.where(na_mask, na_code, codes)

            if not self._observed:
                uniques = uniques.reorder_categories(self._orig_cats)

            return codes, uniques

        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            # we have a list of groupers
            codes = self.grouping_vector.codes_info
            uniques = self.grouping_vector.result_index._values
        elif self._uniques is not None:
            # GH#50486 Code grouping_vector using _uniques; allows
            # including uniques that are not present in grouping_vector.
            cat = Categorical(self.grouping_vector, categories=self._uniques)
            codes = cat.codes
            uniques = self._uniques
        else:
            # GH35667, replace dropna=False with use_na_sentinel=False
            # error: Incompatible types in assignment (expression has type "Union[
            # ndarray[Any, Any], Index]", variable has type "Categorical")
            codes, uniques = algorithms.factorize(  # type: ignore[assignment]
                self.grouping_vector, sort=self._sort, use_na_sentinel=self._dropna
            )
        return codes, uniques

    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]:
        cats = Categorical.from_codes(self.codes, self._group_index, validate=False)
        return self._index.groupby(cats)


def get_grouper(
    obj: NDFrameT,
    key=None,
    axis: Axis = 0,
    level=None,
    sort: bool = True,
    observed: bool = False,
    validate: bool = True,
    dropna: bool = True,
) -> tuple[ops.BaseGrouper, frozenset[Hashable], NDFrameT]:
    """
    Create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to axis,level,sort, while
    the passed in axis, level, and sort are 'global'.

    This routine tries to figure out what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    If observed & we have a categorical grouper, only show the observed
    values.

    If validate, then check for key/level overlaps.

    """
    group_axis = obj._get_axis(axis)

    # validate that the passed single level is compatible with the passed
    # axis of the object
    if level is not None:
        # TODO: These if-block and else-block are almost same.
        # MultiIndex instance check is removable, but it seems that there are
        # some processes only for non-MultiIndex in else-block,
        # eg. `obj.index.name != level`. We have to consider carefully whether
        # these are applicable for MultiIndex. Even if these are applicable,
        # we need to check if it makes no side effect to subsequent processes
        # on the outside of this condition.
        # (GH 17621)
        if isinstance(group_axis, MultiIndex):
            if is_list_like(level) and len(level) == 1:
                level = level[0]

            if key is None and is_scalar(level):
                # Get the level values from group_axis
                key = group_axis.get_level_values(level)
                level = None

        else:
            # allow level to be a length-one list-like object
            # (e.g., level=[0])
            # GH 13901
            if is_list_like(level):
                nlevels = len(level)
                if nlevels == 1:
                    level = level[0]
                elif nlevels == 0:
                    raise ValueError("No group keys passed!")
                else:
                    raise ValueError("multiple levels only valid with MultiIndex")

            if isinstance(level, str):
                if obj._get_axis(axis).name != level:
                    raise ValueError(
                        f"level name {level} is not the name "
                        f"of the {obj._get_axis_name(axis)}"
                    )
            elif level > 0 or level < -1:
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")

            # NOTE: `group_axis` and `group_axis.get_level_values(level)`
            # are same in this section.
            level = None
            key = group_axis

    # a passed-in Grouper, directly convert
    if isinstance(key, Grouper):
        grouper, obj = key._get_grouper(obj, validate=False)
        if key.key is None:
            return grouper, frozenset(), obj
        else:
            return grouper, frozenset({key.key}), obj

    # already have a BaseGrouper, just return it
    elif isinstance(key, ops.BaseGrouper):
        return key, frozenset(), obj

    if not isinstance(key, list):
        keys = [key]
        match_axis_length = False
    else:
        keys = key
        match_axis_length = len(keys) == len(group_axis)

    # what are we after, exactly?
    any_callable = any(callable(g) or isinstance(g, dict) for g in keys)
    any_groupers = any(isinstance(g, (Grouper, Grouping)) for g in keys)
    any_arraylike = any(
        isinstance(g, (list, tuple, Series, Index, np.ndarray)) for g in keys
    )

    # is this an index replacement?
    if (
        not any_callable
        and not any_arraylike
        and not any_groupers
        and match_axis_length
        and level is None
    ):
        if isinstance(obj, DataFrame):
            all_in_columns_index = all(
                g in obj.columns or g in obj.index.names for g in keys
            )
        else:
            assert isinstance(obj, Series)
            all_in_columns_index = all(g in obj.index.names for g in keys)

        if not all_in_columns_index:
            keys = [com.asarray_tuplesafe(keys)]

    if isinstance(level, (tuple, list)):
        if key is None:
            keys = [None] * len(level)
        levels = level
    else:
        levels = [level] * len(keys)

    groupings: list[Grouping] = []
    exclusions: set[Hashable] = set()

    # if the actual grouper should be obj[key]
    def is_in_axis(key) -> bool:
        if not _is_label_like(key):
            if obj.ndim == 1:
                return False

            # items -> .columns for DataFrame, .index for Series
            items = obj.axes[-1]
            try:
                items.get_loc(key)
            except (KeyError, TypeError, InvalidIndexError):
                # TypeError shows up here if we pass e.g. an Index
                return False

        return True

    # if the grouper is obj[name]
    def is_in_obj(gpr) -> bool:
        if not hasattr(gpr, "name"):
            return False
        if using_copy_on_write() or warn_copy_on_write():
            # For the CoW case, we check the references to determine if the
            # series is part of the object
            try:
                obj_gpr_column = obj[gpr.name]
            except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
                return False
            if isinstance(gpr, Series) and isinstance(obj_gpr_column, Series):
                return gpr._mgr.references_same_values(  # type: ignore[union-attr]
                    obj_gpr_column._mgr, 0  # type: ignore[arg-type]
                )
            return False
        try:
            return gpr is obj[gpr.name]
        except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
            # IndexError reached in e.g. test_skip_group_keys when we pass
            #  lambda here
            # InvalidIndexError raised on key-types inappropriate for index,
            #  e.g. DatetimeIndex.get_loc(tuple())
            # OutOfBoundsDatetime raised when obj is a Series with DatetimeIndex
            # and gpr.name is month str
            return False

    for gpr, level in zip(keys, levels):
        if isinstance(obj, DataFrame) and is_in_obj(gpr):  # df.groupby(df['name'])
            in_axis = True
            exclusions.add(gpr.name)

        elif is_in_axis(gpr):  # df.groupby('name')
            if obj.ndim != 1 and gpr in obj:
                if validate:
                    obj._check_label_or_level_ambiguity(gpr, axis=axis)
                in_axis, name, gpr = True, gpr, obj[gpr]
                if gpr.ndim != 1:
                    # non-unique columns; raise here to get the name in the
                    # exception message
                    raise ValueError(f"Grouper for '{name}' not 1-dimensional")
                exclusions.add(name)
            elif obj._is_level_reference(gpr, axis=axis):
                in_axis, level, gpr = False, gpr, None
            else:
                raise KeyError(gpr)
        elif isinstance(gpr, Grouper) and gpr.key is not None:
            # Add key to exclusions
            exclusions.add(gpr.key)
            in_axis = True
        else:
            in_axis = False

        # create the Grouping
        # allow us to passing the actual Grouping as the gpr
        ping = (
            Grouping(
                group_axis,
                gpr,
                obj=obj,
                level=level,
                sort=sort,
                observed=observed,
                in_axis=in_axis,
                dropna=dropna,
            )
            if not isinstance(gpr, Grouping)
            else gpr
        )

        groupings.append(ping)

    if len(groupings) == 0 and len(obj):
        raise ValueError("No group keys passed!")
    if len(groupings) == 0:
        groupings.append(Grouping(Index([], dtype="int"), np.array([], dtype=np.intp)))

    # create the internals grouper
    grouper = ops.BaseGrouper(group_axis, groupings, sort=sort, dropna=dropna)
    return grouper, frozenset(exclusions), obj


def _is_label_like(val) -> bool:
    return isinstance(val, (str, tuple)) or (val is not None and is_scalar(val))


def _convert_grouper(axis: Index, grouper):
    if isinstance(grouper, dict):
        return grouper.get
    elif isinstance(grouper, Series):
        if grouper.index.equals(axis):
            return grouper._values
        else:
            return grouper.reindex(axis)._values
    elif isinstance(grouper, MultiIndex):
        return grouper._values
    elif isinstance(grouper, (list, tuple, Index, Categorical, np.ndarray)):
        if len(grouper) != len(axis):
            raise ValueError("Grouper and axis must be same length")

        if isinstance(grouper, (list, tuple)):
            grouper = com.asarray_tuplesafe(grouper)
        return grouper
    else:
        return grouper

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\info.py ===
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import sys
from textwrap import dedent
from typing import TYPE_CHECKING

from pandas._config import get_option

from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        Dtype,
        WriteBuffer,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


frame_max_cols_sub = dedent(
    """\
    max_cols : int, optional
        When to switch from the verbose to the truncated output. If the
        DataFrame has more than `max_cols` columns, the truncated output
        is used. By default, the setting in
        ``pandas.options.display.max_info_columns`` is used."""
)


show_counts_sub = dedent(
    """\
    show_counts : bool, optional
        Whether to show the non-null counts. By default, this is shown
        only if the DataFrame is smaller than
        ``pandas.options.display.max_info_rows`` and
        ``pandas.options.display.max_info_columns``. A value of True always
        shows the counts, and False never shows the counts."""
)


frame_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,
    ...                   "float_col": float_values})
    >>> df
        int_col text_col  float_col
    0        1    alpha       0.00
    1        2     beta       0.25
    2        3    gamma       0.50
    3        4    delta       0.75
    4        5  epsilon       1.00

    Prints information of all columns:

    >>> df.info(verbose=True)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   int_col    5 non-null      int64
     1   text_col   5 non-null      object
     2   float_col  5 non-null      float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 248.0+ bytes

    Prints a summary of columns count and its dtypes but not per column
    information:

    >>> df.info(verbose=False)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Columns: 3 entries, int_col to float_col
    dtypes: float64(1), int64(1), object(1)
    memory usage: 248.0+ bytes

    Pipe output of DataFrame.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> df.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260

    The `memory_usage` parameter allows deep introspection mode, specially
    useful for big DataFrames and fine-tune memory optimization:

    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    >>> df = pd.DataFrame({
    ...     'column_1': np.random.choice(['a', 'b', 'c'], 10 ** 6),
    ...     'column_2': np.random.choice(['a', 'b', 'c'], 10 ** 6),
    ...     'column_3': np.random.choice(['a', 'b', 'c'], 10 ** 6)
    ... })
    >>> df.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 3 columns):
     #   Column    Non-Null Count    Dtype
    ---  ------    --------------    -----
     0   column_1  1000000 non-null  object
     1   column_2  1000000 non-null  object
     2   column_3  1000000 non-null  object
    dtypes: object(3)
    memory usage: 22.9+ MB

    >>> df.info(memory_usage='deep')
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 3 columns):
     #   Column    Non-Null Count    Dtype
    ---  ------    --------------    -----
     0   column_1  1000000 non-null  object
     1   column_2  1000000 non-null  object
     2   column_3  1000000 non-null  object
    dtypes: object(3)
    memory usage: 165.9 MB"""
)


frame_see_also_sub = dedent(
    """\
    DataFrame.describe: Generate descriptive statistics of DataFrame
        columns.
    DataFrame.memory_usage: Memory usage of DataFrame columns."""
)


frame_sub_kwargs = {
    "klass": "DataFrame",
    "type_sub": " and columns",
    "max_cols_sub": frame_max_cols_sub,
    "show_counts_sub": show_counts_sub,
    "examples_sub": frame_examples_sub,
    "see_also_sub": frame_see_also_sub,
    "version_added_sub": "",
}


series_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> s = pd.Series(text_values, index=int_values)
    >>> s.info()
    <class 'pandas.core.series.Series'>
    Index: 5 entries, 1 to 5
    Series name: None
    Non-Null Count  Dtype
    --------------  -----
    5 non-null      object
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Prints a summary excluding information about its values:

    >>> s.info(verbose=False)
    <class 'pandas.core.series.Series'>
    Index: 5 entries, 1 to 5
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Pipe output of Series.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> s.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260

    The `memory_usage` parameter allows deep introspection mode, specially
    useful for big Series and fine-tune memory optimization:

    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    >>> s = pd.Series(np.random.choice(['a', 'b', 'c'], 10 ** 6))
    >>> s.info()
    <class 'pandas.core.series.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 7.6+ MB

    >>> s.info(memory_usage='deep')
    <class 'pandas.core.series.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 55.3 MB"""
)


series_see_also_sub = dedent(
    """\
    Series.describe: Generate descriptive statistics of Series.
    Series.memory_usage: Memory usage of Series."""
)


series_sub_kwargs = {
    "klass": "Series",
    "type_sub": "",
    "max_cols_sub": "",
    "show_counts_sub": show_counts_sub,
    "examples_sub": series_examples_sub,
    "see_also_sub": series_see_also_sub,
    "version_added_sub": "\n.. versionadded:: 1.4.0\n",
}


INFO_DOCSTRING = dedent(
    """
    Print a concise summary of a {klass}.

    This method prints information about a {klass} including
    the index dtype{type_sub}, non-null values and memory usage.
    {version_added_sub}\

    Parameters
    ----------
    verbose : bool, optional
        Whether to print the full summary. By default, the setting in
        ``pandas.options.display.max_info_columns`` is followed.
    buf : writable buffer, defaults to sys.stdout
        Where to send the output. By default, the output is printed to
        sys.stdout. Pass a writable buffer if you need to further process
        the output.
    {max_cols_sub}
    memory_usage : bool, str, optional
        Specifies whether total memory usage of the {klass}
        elements (including the index) should be displayed. By default,
        this follows the ``pandas.options.display.memory_usage`` setting.

        True always show memory usage. False never shows memory usage.
        A value of 'deep' is equivalent to "True with deep introspection".
        Memory usage is shown in human-readable units (base-2
        representation). Without deep introspection a memory estimation is
        made based in column dtype and number of rows assuming values
        consume the same memory amount for corresponding dtypes. With deep
        memory introspection, a real memory usage calculation is performed
        at the cost of computational resources. See the
        :ref:`Frequently Asked Questions <df-memory-usage>` for more
        details.
    {show_counts_sub}

    Returns
    -------
    None
        This method prints a summary of a {klass} and returns None.

    See Also
    --------
    {see_also_sub}

    Examples
    --------
    {examples_sub}
    """
)


def _put_str(s: str | Dtype, space: int) -> str:
    """
    Make string of specified length, padding to the right if necessary.

    Parameters
    ----------
    s : Union[str, Dtype]
        String to be formatted.
    space : int
        Length to force string to be of.

    Returns
    -------
    str
        String coerced to given length.

    Examples
    --------
    >>> pd.io.formats.info._put_str("panda", 6)
    'panda '
    >>> pd.io.formats.info._put_str("panda", 4)
    'pand'
    """
    return str(s)[:space].ljust(space)


def _sizeof_fmt(num: float, size_qualifier: str) -> str:
    """
    Return size in human readable format.

    Parameters
    ----------
    num : int
        Size in bytes.
    size_qualifier : str
        Either empty, or '+' (if lower bound).

    Returns
    -------
    str
        Size in human readable format.

    Examples
    --------
    >>> _sizeof_fmt(23028, '')
    '22.5 KB'

    >>> _sizeof_fmt(23028, '+')
    '22.5+ KB'
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f}{size_qualifier} {x}"
        num /= 1024.0
    return f"{num:3.1f}{size_qualifier} PB"


def _initialize_memory_usage(
    memory_usage: bool | str | None = None,
) -> bool | str:
    """Get memory usage based on inputs and display options."""
    if memory_usage is None:
        memory_usage = get_option("display.memory_usage")
    return memory_usage


class _BaseInfo(ABC):
    """
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    memory_usage : bool or str, optional
        If "deep", introspect the data deeply by interrogating object dtypes
        for system-level memory consumption, and include it in the returned
        values.
    """

    data: DataFrame | Series
    memory_usage: bool | str

    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""

    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""

    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """
        Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """

    @property
    def memory_usage_string(self) -> str:
        """Memory usage in a form of human readable string."""
        return f"{_sizeof_fmt(self.memory_usage_bytes, self.size_qualifier)}\n"

    @property
    def size_qualifier(self) -> str:
        size_qualifier = ""
        if self.memory_usage:
            if self.memory_usage != "deep":
                # size_qualifier is just a best effort; not guaranteed to catch
                # all cases (e.g., it misses categorical data even with object
                # categories)
                if (
                    "object" in self.dtype_counts
                    or self.data.index._is_memory_usage_qualified()
                ):
                    size_qualifier = "+"
        return size_qualifier

    @abstractmethod
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        pass


class DataFrameInfo(_BaseInfo):
    """
    Class storing dataframe-specific info.
    """

    def __init__(
        self,
        data: DataFrame,
        memory_usage: bool | str | None = None,
    ) -> None:
        self.data: DataFrame = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        return self.data.dtypes

    @property
    def ids(self) -> Index:
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
        return self.data.columns

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return len(self.ids)

    @property
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
        return self.data.count()

    @property
    def memory_usage_bytes(self) -> int:
        deep = self.memory_usage == "deep"
        return self.data.memory_usage(index=True, deep=deep).sum()

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        printer = _DataFrameInfoPrinter(
            info=self,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)


class SeriesInfo(_BaseInfo):
    """
    Class storing series-specific info.
    """

    def __init__(
        self,
        data: Series,
        memory_usage: bool | str | None = None,
    ) -> None:
        self.data: Series = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None = None,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        if max_cols is not None:
            raise ValueError(
                "Argument `max_cols` can only be passed "
                "in DataFrame.info, not Series.info"
            )
        printer = _SeriesInfoPrinter(
            info=self,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)

    @property
    def non_null_counts(self) -> Sequence[int]:
        return [self.data.count()]

    @property
    def dtypes(self) -> Iterable[Dtype]:
        return [self.data.dtypes]

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        from pandas.core.frame import DataFrame

        return _get_dataframe_dtype_counts(DataFrame(self.data))

    @property
    def memory_usage_bytes(self) -> int:
        """Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """
        deep = self.memory_usage == "deep"
        return self.data.memory_usage(index=True, deep=deep)


class _InfoPrinterAbstract:
    """
    Class for printing dataframe or series info.
    """

    def to_buffer(self, buf: WriteBuffer[str] | None = None) -> None:
        """Save dataframe info into buffer."""
        table_builder = self._create_table_builder()
        lines = table_builder.get_lines()
        if buf is None:  # pragma: no cover
            buf = sys.stdout
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self) -> _TableBuilderAbstract:
        """Create instance of table builder."""


class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(
        self,
        info: DataFrameInfo,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.max_cols = self._initialize_max_cols(max_cols)
        self.show_counts = self._initialize_show_counts(show_counts)

    @property
    def max_rows(self) -> int:
        """Maximum info rows to be displayed."""
        return get_option("display.max_info_rows", len(self.data) + 1)

    @property
    def exceeds_info_cols(self) -> bool:
        """Check if number of columns to be summarized does not exceed maximum."""
        return bool(self.col_count > self.max_cols)

    @property
    def exceeds_info_rows(self) -> bool:
        """Check if number of rows to be summarized does not exceed maximum."""
        return bool(len(self.data) > self.max_rows)

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return self.info.col_count

    def _initialize_max_cols(self, max_cols: int | None) -> int:
        if max_cols is None:
            return get_option("display.max_info_columns", self.col_count + 1)
        return max_cols

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        if show_counts is None:
            return bool(not self.exceeds_info_cols and not self.exceeds_info_rows)
        else:
            return show_counts

    def _create_table_builder(self) -> _DataFrameTableBuilder:
        """
        Create instance of table builder based on verbosity and display settings.
        """
        if self.verbose:
            return _DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        elif self.verbose is False:  # specifically set to False, not necessarily None
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        elif self.exceeds_info_cols:
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        else:
            return _DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )


class _SeriesInfoPrinter(_InfoPrinterAbstract):
    """Class for printing series info.

    Parameters
    ----------
    info : SeriesInfo
        Instance of SeriesInfo.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(
        self,
        info: SeriesInfo,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.show_counts = self._initialize_show_counts(show_counts)

    def _create_table_builder(self) -> _SeriesTableBuilder:
        """
        Create instance of table builder based on verbosity.
        """
        if self.verbose or self.verbose is None:
            return _SeriesTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        else:
            return _SeriesTableBuilderNonVerbose(info=self.info)

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        if show_counts is None:
            return True
        else:
            return show_counts


class _TableBuilderAbstract(ABC):
    """
    Abstract builder for info table.
    """

    _lines: list[str]
    info: _BaseInfo

    @abstractmethod
    def get_lines(self) -> list[str]:
        """Product in a form of list of lines (strings)."""

    @property
    def data(self) -> DataFrame | Series:
        return self.info.data

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """Dtypes of each of the DataFrame's columns."""
        return self.info.dtypes

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
        return self.info.dtype_counts

    @property
    def display_memory_usage(self) -> bool:
        """Whether to display memory usage."""
        return bool(self.info.memory_usage)

    @property
    def memory_usage_string(self) -> str:
        """Memory usage string with proper size qualifier."""
        return self.info.memory_usage_string

    @property
    def non_null_counts(self) -> Sequence[int]:
        return self.info.non_null_counts

    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
        self._lines.append(str(type(self.data)))

    def add_index_range_line(self) -> None:
        """Add line with range of indices to the table."""
        self._lines.append(self.data.index._summary())

    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""
        collected_dtypes = [
            f"{key}({val:d})" for key, val in sorted(self.dtype_counts.items())
        ]
        self._lines.append(f"dtypes: {', '.join(collected_dtypes)}")


class _DataFrameTableBuilder(_TableBuilderAbstract):
    """
    Abstract builder for dataframe info table.

    Parameters
    ----------
    info : DataFrameInfo.
        Instance of DataFrameInfo.
    """

    def __init__(self, *, info: DataFrameInfo) -> None:
        self.info: DataFrameInfo = info

    def get_lines(self) -> list[str]:
        self._lines = []
        if self.col_count == 0:
            self._fill_empty_info()
        else:
            self._fill_non_empty_info()
        return self._lines

    def _fill_empty_info(self) -> None:
        """Add lines to the info table, pertaining to empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self._lines.append(f"Empty {type(self.data).__name__}\n")

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""

    @property
    def data(self) -> DataFrame:
        """DataFrame."""
        return self.info.data

    @property
    def ids(self) -> Index:
        """Dataframe columns."""
        return self.info.ids

    @property
    def col_count(self) -> int:
        """Number of dataframe columns to be summarized."""
        return self.info.col_count

    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        self._lines.append(f"memory usage: {self.memory_usage_string}")


class _DataFrameTableBuilderNonVerbose(_DataFrameTableBuilder):
    """
    Dataframe info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_columns_summary_line(self) -> None:
        self._lines.append(self.ids._summary(name="Columns"))


class _TableBuilderVerboseMixin(_TableBuilderAbstract):
    """
    Mixin for verbose info output.
    """

    SPACING: str = " " * 2
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool

    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""

    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only titles)."""
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> Sequence[int]:
        """Get widths of columns containing both headers and actual content."""
        body_column_widths = self._get_body_column_widths()
        return [
            max(*widths)
            for widths in zip(self.header_column_widths, body_column_widths)
        ]

    def _get_body_column_widths(self) -> Sequence[int]:
        """Get widths of table content columns."""
        strcols: Sequence[Sequence[str]] = list(zip(*self.strrows))
        return [max(len(x) for x in col) for col in strcols]

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
        if self.with_counts:
            return self._gen_rows_with_counts()
        else:
            return self._gen_rows_without_counts()

    @abstractmethod
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""

    @abstractmethod
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""

    def add_header_line(self) -> None:
        header_line = self.SPACING.join(
            [
                _put_str(header, col_width)
                for header, col_width in zip(self.headers, self.gross_column_widths)
            ]
        )
        self._lines.append(header_line)

    def add_separator_line(self) -> None:
        separator_line = self.SPACING.join(
            [
                _put_str("-" * header_colwidth, gross_colwidth)
                for header_colwidth, gross_colwidth in zip(
                    self.header_column_widths, self.gross_column_widths
                )
            ]
        )
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        for row in self.strrows:
            body_line = self.SPACING.join(
                [
                    _put_str(col, gross_colwidth)
                    for col, gross_colwidth in zip(row, self.gross_column_widths)
                ]
            )
            self._lines.append(body_line)

    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
        for count in self.non_null_counts:
            yield f"{count} non-null"

    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
        for dtype in self.dtypes:
            yield pprint_thing(dtype)


class _DataFrameTableBuilderVerbose(_DataFrameTableBuilder, _TableBuilderVerboseMixin):
    """
    Dataframe info table builder for verbose output.
    """

    def __init__(
        self,
        *,
        info: DataFrameInfo,
        with_counts: bool,
    ) -> None:
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return [" # ", "Column", "Non-Null Count", "Dtype"]
        return [" # ", "Column", "Dtype"]

    def add_columns_summary_line(self) -> None:
        self._lines.append(f"Data columns (total {self.col_count} columns):")

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
        yield from zip(
            self._gen_line_numbers(),
            self._gen_columns(),
            self._gen_dtypes(),
        )

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
        yield from zip(
            self._gen_line_numbers(),
            self._gen_columns(),
            self._gen_non_null_counts(),
            self._gen_dtypes(),
        )

    def _gen_line_numbers(self) -> Iterator[str]:
        """Iterator with string representation of column numbers."""
        for i, _ in enumerate(self.ids):
            yield f" {i}"

    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""
        for col in self.ids:
            yield pprint_thing(col)


class _SeriesTableBuilder(_TableBuilderAbstract):
    """
    Abstract builder for series info table.

    Parameters
    ----------
    info : SeriesInfo.
        Instance of SeriesInfo.
    """

    def __init__(self, *, info: SeriesInfo) -> None:
        self.info: SeriesInfo = info

    def get_lines(self) -> list[str]:
        self._lines = []
        self._fill_non_empty_info()
        return self._lines

    @property
    def data(self) -> Series:
        """Series."""
        return self.info.data

    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        self._lines.append(f"memory usage: {self.memory_usage_string}")

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""


class _SeriesTableBuilderNonVerbose(_SeriesTableBuilder):
    """
    Series info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()


class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    """
    Series info table builder for verbose output.
    """

    def __init__(
        self,
        *,
        info: SeriesInfo,
        with_counts: bool,
    ) -> None:
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_series_name_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_series_name_line(self) -> None:
        self._lines.append(f"Series name: {self.data.name}")

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return ["Non-Null Count", "Dtype"]
        return ["Dtype"]

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
        yield from self._gen_dtypes()

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
        yield from zip(
            self._gen_non_null_counts(),
            self._gen_dtypes(),
        )


def _get_dataframe_dtype_counts(df: DataFrame) -> Mapping[str, int]:
    """
    Create mapping between datatypes and their number of occurrences.
    """
    # groupby dtype.name to collect e.g. Categorical columns
    return df.dtypes.value_counts().groupby(lambda x: x.name).sum()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\wheel.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013-2023 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
from __future__ import unicode_literals

import base64
import codecs
import datetime
from email import message_from_file
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import sys
import tempfile
import zipfile

from . import __version__, DistlibException
from .compat import sysconfig, ZipFile, fsdecode, text_type, filter
from .database import InstalledDistribution
from .metadata import Metadata, WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME
from .util import (FileOperator, convert_path, CSVReader, CSVWriter, Cache, cached_property, get_cache_base,
                   read_exports, tempdir, get_platform)
from .version import NormalizedVersion, UnsupportedVersionError

logger = logging.getLogger(__name__)

cache = None  # created when needed

if hasattr(sys, 'pypy_version_info'):  # pragma: no cover
    IMP_PREFIX = 'pp'
elif sys.platform.startswith('java'):  # pragma: no cover
    IMP_PREFIX = 'jy'
elif sys.platform == 'cli':  # pragma: no cover
    IMP_PREFIX = 'ip'
else:
    IMP_PREFIX = 'cp'

VER_SUFFIX = sysconfig.get_config_var('py_version_nodot')
if not VER_SUFFIX:  # pragma: no cover
    VER_SUFFIX = '%s%s' % sys.version_info[:2]
PYVER = 'py' + VER_SUFFIX
IMPVER = IMP_PREFIX + VER_SUFFIX

ARCH = get_platform().replace('-', '_').replace('.', '_')

ABI = sysconfig.get_config_var('SOABI')
if ABI and ABI.startswith('cpython-'):
    ABI = ABI.replace('cpython-', 'cp').split('-')[0]
else:

    def _derive_abi():
        parts = ['cp', VER_SUFFIX]
        if sysconfig.get_config_var('Py_DEBUG'):
            parts.append('d')
        if IMP_PREFIX == 'cp':
            vi = sys.version_info[:2]
            if vi < (3, 8):
                wpm = sysconfig.get_config_var('WITH_PYMALLOC')
                if wpm is None:
                    wpm = True
                if wpm:
                    parts.append('m')
                if vi < (3, 3):
                    us = sysconfig.get_config_var('Py_UNICODE_SIZE')
                    if us == 4 or (us is None and sys.maxunicode == 0x10FFFF):
                        parts.append('u')
        return ''.join(parts)

    ABI = _derive_abi()
    del _derive_abi

FILENAME_RE = re.compile(
    r'''
(?P<nm>[^-]+)
-(?P<vn>\d+[^-]*)
(-(?P<bn>\d+[^-]*))?
-(?P<py>\w+\d+(\.\w+\d+)*)
-(?P<bi>\w+)
-(?P<ar>\w+(\.\w+)*)
\.whl$
''', re.IGNORECASE | re.VERBOSE)

NAME_VERSION_RE = re.compile(r'''
(?P<nm>[^-]+)
-(?P<vn>\d+[^-]*)
(-(?P<bn>\d+[^-]*))?$
''', re.IGNORECASE | re.VERBOSE)

SHEBANG_RE = re.compile(br'\s*#![^\r\n]*')
SHEBANG_DETAIL_RE = re.compile(br'^(\s*#!("[^"]+"|\S+))\s+(.*)$')
SHEBANG_PYTHON = b'#!python'
SHEBANG_PYTHONW = b'#!pythonw'

if os.sep == '/':
    to_posix = lambda o: o
else:
    to_posix = lambda o: o.replace(os.sep, '/')

if sys.version_info[0] < 3:
    import imp
else:
    imp = None
    import importlib.machinery
    import importlib.util


def _get_suffixes():
    if imp:
        return [s[0] for s in imp.get_suffixes()]
    else:
        return importlib.machinery.EXTENSION_SUFFIXES


def _load_dynamic(name, path):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    if imp:
        return imp.load_dynamic(name, path)
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


class Mounter(object):

    def __init__(self):
        self.impure_wheels = {}
        self.libs = {}

    def add(self, pathname, extensions):
        self.impure_wheels[pathname] = extensions
        self.libs.update(extensions)

    def remove(self, pathname):
        extensions = self.impure_wheels.pop(pathname)
        for k, v in extensions:
            if k in self.libs:
                del self.libs[k]

    def find_module(self, fullname, path=None):
        if fullname in self.libs:
            result = self
        else:
            result = None
        return result

    def load_module(self, fullname):
        if fullname in sys.modules:
            result = sys.modules[fullname]
        else:
            if fullname not in self.libs:
                raise ImportError('unable to find extension for %s' % fullname)
            result = _load_dynamic(fullname, self.libs[fullname])
            result.__loader__ = self
            parts = fullname.rsplit('.', 1)
            if len(parts) > 1:
                result.__package__ = parts[0]
        return result


_hook = Mounter()


class Wheel(object):
    """
    Class to build and install from Wheel files (PEP 427).
    """

    wheel_version = (1, 1)
    hash_kind = 'sha256'

    def __init__(self, filename=None, sign=False, verify=False):
        """
        Initialise an instance using a (valid) filename.
        """
        self.sign = sign
        self.should_verify = verify
        self.buildver = ''
        self.pyver = [PYVER]
        self.abi = ['none']
        self.arch = ['any']
        self.dirname = os.getcwd()
        if filename is None:
            self.name = 'dummy'
            self.version = '0.1'
            self._filename = self.filename
        else:
            m = NAME_VERSION_RE.match(filename)
            if m:
                info = m.groupdict('')
                self.name = info['nm']
                # Reinstate the local version separator
                self.version = info['vn'].replace('_', '-')
                self.buildver = info['bn']
                self._filename = self.filename
            else:
                dirname, filename = os.path.split(filename)
                m = FILENAME_RE.match(filename)
                if not m:
                    raise DistlibException('Invalid name or '
                                           'filename: %r' % filename)
                if dirname:
                    self.dirname = os.path.abspath(dirname)
                self._filename = filename
                info = m.groupdict('')
                self.name = info['nm']
                self.version = info['vn']
                self.buildver = info['bn']
                self.pyver = info['py'].split('.')
                self.abi = info['bi'].split('.')
                self.arch = info['ar'].split('.')

    @property
    def filename(self):
        """
        Build and return a filename from the various components.
        """
        if self.buildver:
            buildver = '-' + self.buildver
        else:
            buildver = ''
        pyver = '.'.join(self.pyver)
        abi = '.'.join(self.abi)
        arch = '.'.join(self.arch)
        # replace - with _ as a local version separator
        version = self.version.replace('-', '_')
        return '%s-%s%s-%s-%s-%s.whl' % (self.name, version, buildver, pyver, abi, arch)

    @property
    def exists(self):
        path = os.path.join(self.dirname, self.filename)
        return os.path.isfile(path)

    @property
    def tags(self):
        for pyver in self.pyver:
            for abi in self.abi:
                for arch in self.arch:
                    yield pyver, abi, arch

    @cached_property
    def metadata(self):
        pathname = os.path.join(self.dirname, self.filename)
        name_ver = '%s-%s' % (self.name, self.version)
        info_dir = '%s.dist-info' % name_ver
        wrapper = codecs.getreader('utf-8')
        with ZipFile(pathname, 'r') as zf:
            self.get_wheel_metadata(zf)
            # wv = wheel_metadata['Wheel-Version'].split('.', 1)
            # file_version = tuple([int(i) for i in wv])
            # if file_version < (1, 1):
            # fns = [WHEEL_METADATA_FILENAME, METADATA_FILENAME,
            # LEGACY_METADATA_FILENAME]
            # else:
            # fns = [WHEEL_METADATA_FILENAME, METADATA_FILENAME]
            fns = [WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME]
            result = None
            for fn in fns:
                try:
                    metadata_filename = posixpath.join(info_dir, fn)
                    with zf.open(metadata_filename) as bf:
                        wf = wrapper(bf)
                        result = Metadata(fileobj=wf)
                        if result:
                            break
                except KeyError:
                    pass
            if not result:
                raise ValueError('Invalid wheel, because metadata is '
                                 'missing: looked in %s' % ', '.join(fns))
        return result

    def get_wheel_metadata(self, zf):
        name_ver = '%s-%s' % (self.name, self.version)
        info_dir = '%s.dist-info' % name_ver
        metadata_filename = posixpath.join(info_dir, 'WHEEL')
        with zf.open(metadata_filename) as bf:
            wf = codecs.getreader('utf-8')(bf)
            message = message_from_file(wf)
        return dict(message)

    @cached_property
    def info(self):
        pathname = os.path.join(self.dirname, self.filename)
        with ZipFile(pathname, 'r') as zf:
            result = self.get_wheel_metadata(zf)
        return result

    def process_shebang(self, data):
        m = SHEBANG_RE.match(data)
        if m:
            end = m.end()
            shebang, data_after_shebang = data[:end], data[end:]
            # Preserve any arguments after the interpreter
            if b'pythonw' in shebang.lower():
                shebang_python = SHEBANG_PYTHONW
            else:
                shebang_python = SHEBANG_PYTHON
            m = SHEBANG_DETAIL_RE.match(shebang)
            if m:
                args = b' ' + m.groups()[-1]
            else:
                args = b''
            shebang = shebang_python + args
            data = shebang + data_after_shebang
        else:
            cr = data.find(b'\r')
            lf = data.find(b'\n')
            if cr < 0 or cr > lf:
                term = b'\n'
            else:
                if data[cr:cr + 2] == b'\r\n':
                    term = b'\r\n'
                else:
                    term = b'\r'
            data = SHEBANG_PYTHON + term + data
        return data

    def get_hash(self, data, hash_kind=None):
        if hash_kind is None:
            hash_kind = self.hash_kind
        try:
            hasher = getattr(hashlib, hash_kind)
        except AttributeError:
            raise DistlibException('Unsupported hash algorithm: %r' % hash_kind)
        result = hasher(data).digest()
        result = base64.urlsafe_b64encode(result).rstrip(b'=').decode('ascii')
        return hash_kind, result

    def write_record(self, records, record_path, archive_record_path):
        records = list(records)  # make a copy, as mutated
        records.append((archive_record_path, '', ''))
        with CSVWriter(record_path) as writer:
            for row in records:
                writer.writerow(row)

    def write_records(self, info, libdir, archive_paths):
        records = []
        distinfo, info_dir = info
        # hasher = getattr(hashlib, self.hash_kind)
        for ap, p in archive_paths:
            with open(p, 'rb') as f:
                data = f.read()
            digest = '%s=%s' % self.get_hash(data)
            size = os.path.getsize(p)
            records.append((ap, digest, size))

        p = os.path.join(distinfo, 'RECORD')
        ap = to_posix(os.path.join(info_dir, 'RECORD'))
        self.write_record(records, p, ap)
        archive_paths.append((ap, p))

    def build_zip(self, pathname, archive_paths):
        with ZipFile(pathname, 'w', zipfile.ZIP_DEFLATED) as zf:
            for ap, p in archive_paths:
                logger.debug('Wrote %s to %s in wheel', p, ap)
                zf.write(p, ap)

    def build(self, paths, tags=None, wheel_version=None):
        """
        Build a wheel from files in specified paths, and use any specified tags
        when determining the name of the wheel.
        """
        if tags is None:
            tags = {}

        libkey = list(filter(lambda o: o in paths, ('purelib', 'platlib')))[0]
        if libkey == 'platlib':
            is_pure = 'false'
            default_pyver = [IMPVER]
            default_abi = [ABI]
            default_arch = [ARCH]
        else:
            is_pure = 'true'
            default_pyver = [PYVER]
            default_abi = ['none']
            default_arch = ['any']

        self.pyver = tags.get('pyver', default_pyver)
        self.abi = tags.get('abi', default_abi)
        self.arch = tags.get('arch', default_arch)

        libdir = paths[libkey]

        name_ver = '%s-%s' % (self.name, self.version)
        data_dir = '%s.data' % name_ver
        info_dir = '%s.dist-info' % name_ver

        archive_paths = []

        # First, stuff which is not in site-packages
        for key in ('data', 'headers', 'scripts'):
            if key not in paths:
                continue
            path = paths[key]
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for fn in files:
                        p = fsdecode(os.path.join(root, fn))
                        rp = os.path.relpath(p, path)
                        ap = to_posix(os.path.join(data_dir, key, rp))
                        archive_paths.append((ap, p))
                        if key == 'scripts' and not p.endswith('.exe'):
                            with open(p, 'rb') as f:
                                data = f.read()
                            data = self.process_shebang(data)
                            with open(p, 'wb') as f:
                                f.write(data)

        # Now, stuff which is in site-packages, other than the
        # distinfo stuff.
        path = libdir
        distinfo = None
        for root, dirs, files in os.walk(path):
            if root == path:
                # At the top level only, save distinfo for later
                # and skip it for now
                for i, dn in enumerate(dirs):
                    dn = fsdecode(dn)
                    if dn.endswith('.dist-info'):
                        distinfo = os.path.join(root, dn)
                        del dirs[i]
                        break
                assert distinfo, '.dist-info directory expected, not found'

            for fn in files:
                # comment out next suite to leave .pyc files in
                if fsdecode(fn).endswith(('.pyc', '.pyo')):
                    continue
                p = os.path.join(root, fn)
                rp = to_posix(os.path.relpath(p, path))
                archive_paths.append((rp, p))

        # Now distinfo. Assumed to be flat, i.e. os.listdir is enough.
        files = os.listdir(distinfo)
        for fn in files:
            if fn not in ('RECORD', 'INSTALLER', 'SHARED', 'WHEEL'):
                p = fsdecode(os.path.join(distinfo, fn))
                ap = to_posix(os.path.join(info_dir, fn))
                archive_paths.append((ap, p))

        wheel_metadata = [
            'Wheel-Version: %d.%d' % (wheel_version or self.wheel_version),
            'Generator: distlib %s' % __version__,
            'Root-Is-Purelib: %s' % is_pure,
        ]
        for pyver, abi, arch in self.tags:
            wheel_metadata.append('Tag: %s-%s-%s' % (pyver, abi, arch))
        p = os.path.join(distinfo, 'WHEEL')
        with open(p, 'w') as f:
            f.write('\n'.join(wheel_metadata))
        ap = to_posix(os.path.join(info_dir, 'WHEEL'))
        archive_paths.append((ap, p))

        # sort the entries by archive path. Not needed by any spec, but it
        # keeps the archive listing and RECORD tidier than they would otherwise
        # be. Use the number of path segments to keep directory entries together,
        # and keep the dist-info stuff at the end.
        def sorter(t):
            ap = t[0]
            n = ap.count('/')
            if '.dist-info' in ap:
                n += 10000
            return (n, ap)

        archive_paths = sorted(archive_paths, key=sorter)

        # Now, at last, RECORD.
        # Paths in here are archive paths - nothing else makes sense.
        self.write_records((distinfo, info_dir), libdir, archive_paths)
        # Now, ready to build the zip file
        pathname = os.path.join(self.dirname, self.filename)
        self.build_zip(pathname, archive_paths)
        return pathname

    def skip_entry(self, arcname):
        """
        Determine whether an archive entry should be skipped when verifying
        or installing.
        """
        # The signature file won't be in RECORD,
        # and we  don't currently don't do anything with it
        # We also skip directories, as they won't be in RECORD
        # either. See:
        #
        # https://github.com/pypa/wheel/issues/294
        # https://github.com/pypa/wheel/issues/287
        # https://github.com/pypa/wheel/pull/289
        #
        return arcname.endswith(('/', '/RECORD.jws'))

    def install(self, paths, maker, **kwargs):
        """
        Install a wheel to the specified paths. If kwarg ``warner`` is
        specified, it should be a callable, which will be called with two
        tuples indicating the wheel version of this software and the wheel
        version in the file, if there is a discrepancy in the versions.
        This can be used to issue any warnings to raise any exceptions.
        If kwarg ``lib_only`` is True, only the purelib/platlib files are
        installed, and the headers, scripts, data and dist-info metadata are
        not written. If kwarg ``bytecode_hashed_invalidation`` is True, written
        bytecode will try to use file-hash based invalidation (PEP-552) on
        supported interpreter versions (CPython 3.7+).

        The return value is a :class:`InstalledDistribution` instance unless
        ``options.lib_only`` is True, in which case the return value is ``None``.
        """

        dry_run = maker.dry_run
        warner = kwargs.get('warner')
        lib_only = kwargs.get('lib_only', False)
        bc_hashed_invalidation = kwargs.get('bytecode_hashed_invalidation', False)

        pathname = os.path.join(self.dirname, self.filename)
        name_ver = '%s-%s' % (self.name, self.version)
        data_dir = '%s.data' % name_ver
        info_dir = '%s.dist-info' % name_ver

        metadata_name = posixpath.join(info_dir, LEGACY_METADATA_FILENAME)
        wheel_metadata_name = posixpath.join(info_dir, 'WHEEL')
        record_name = posixpath.join(info_dir, 'RECORD')

        wrapper = codecs.getreader('utf-8')

        with ZipFile(pathname, 'r') as zf:
            with zf.open(wheel_metadata_name) as bwf:
                wf = wrapper(bwf)
                message = message_from_file(wf)
            wv = message['Wheel-Version'].split('.', 1)
            file_version = tuple([int(i) for i in wv])
            if (file_version != self.wheel_version) and warner:
                warner(self.wheel_version, file_version)

            if message['Root-Is-Purelib'] == 'true':
                libdir = paths['purelib']
            else:
                libdir = paths['platlib']

            records = {}
            with zf.open(record_name) as bf:
                with CSVReader(stream=bf) as reader:
                    for row in reader:
                        p = row[0]
                        records[p] = row

            data_pfx = posixpath.join(data_dir, '')
            info_pfx = posixpath.join(info_dir, '')
            script_pfx = posixpath.join(data_dir, 'scripts', '')

            # make a new instance rather than a copy of maker's,
            # as we mutate it
            fileop = FileOperator(dry_run=dry_run)
            fileop.record = True  # so we can rollback if needed

            bc = not sys.dont_write_bytecode  # Double negatives. Lovely!

            outfiles = []  # for RECORD writing

            # for script copying/shebang processing
            workdir = tempfile.mkdtemp()
            # set target dir later
            # we default add_launchers to False, as the
            # Python Launcher should be used instead
            maker.source_dir = workdir
            maker.target_dir = None
            try:
                for zinfo in zf.infolist():
                    arcname = zinfo.filename
                    if isinstance(arcname, text_type):
                        u_arcname = arcname
                    else:
                        u_arcname = arcname.decode('utf-8')
                    if self.skip_entry(u_arcname):
                        continue
                    row = records[u_arcname]
                    if row[2] and str(zinfo.file_size) != row[2]:
                        raise DistlibException('size mismatch for '
                                               '%s' % u_arcname)
                    if row[1]:
                        kind, value = row[1].split('=', 1)
                        with zf.open(arcname) as bf:
                            data = bf.read()
                        _, digest = self.get_hash(data, kind)
                        if digest != value:
                            raise DistlibException('digest mismatch for '
                                                   '%s' % arcname)

                    if lib_only and u_arcname.startswith((info_pfx, data_pfx)):
                        logger.debug('lib_only: skipping %s', u_arcname)
                        continue
                    is_script = (u_arcname.startswith(script_pfx) and not u_arcname.endswith('.exe'))

                    if u_arcname.startswith(data_pfx):
                        _, where, rp = u_arcname.split('/', 2)
                        outfile = os.path.join(paths[where], convert_path(rp))
                    else:
                        # meant for site-packages.
                        if u_arcname in (wheel_metadata_name, record_name):
                            continue
                        outfile = os.path.join(libdir, convert_path(u_arcname))
                    if not is_script:
                        with zf.open(arcname) as bf:
                            fileop.copy_stream(bf, outfile)
                        # Issue #147: permission bits aren't preserved. Using
                        # zf.extract(zinfo, libdir) should have worked, but didn't,
                        # see https://www.thetopsites.net/article/53834422.shtml
                        # So ... manually preserve permission bits as given in zinfo
                        if os.name == 'posix':
                            # just set the normal permission bits
                            os.chmod(outfile, (zinfo.external_attr >> 16) & 0x1FF)
                        outfiles.append(outfile)
                        # Double check the digest of the written file
                        if not dry_run and row[1]:
                            with open(outfile, 'rb') as bf:
                                data = bf.read()
                                _, newdigest = self.get_hash(data, kind)
                                if newdigest != digest:
                                    raise DistlibException('digest mismatch '
                                                           'on write for '
                                                           '%s' % outfile)
                        if bc and outfile.endswith('.py'):
                            try:
                                pyc = fileop.byte_compile(outfile, hashed_invalidation=bc_hashed_invalidation)
                                outfiles.append(pyc)
                            except Exception:
                                # Don't give up if byte-compilation fails,
                                # but log it and perhaps warn the user
                                logger.warning('Byte-compilation failed', exc_info=True)
                    else:
                        fn = os.path.basename(convert_path(arcname))
                        workname = os.path.join(workdir, fn)
                        with zf.open(arcname) as bf:
                            fileop.copy_stream(bf, workname)

                        dn, fn = os.path.split(outfile)
                        maker.target_dir = dn
                        filenames = maker.make(fn)
                        fileop.set_executable_mode(filenames)
                        outfiles.extend(filenames)

                if lib_only:
                    logger.debug('lib_only: returning None')
                    dist = None
                else:
                    # Generate scripts

                    # Try to get pydist.json so we can see if there are
                    # any commands to generate. If this fails (e.g. because
                    # of a legacy wheel), log a warning but don't give up.
                    commands = None
                    file_version = self.info['Wheel-Version']
                    if file_version == '1.0':
                        # Use legacy info
                        ep = posixpath.join(info_dir, 'entry_points.txt')
                        try:
                            with zf.open(ep) as bwf:
                                epdata = read_exports(bwf)
                            commands = {}
                            for key in ('console', 'gui'):
                                k = '%s_scripts' % key
                                if k in epdata:
                                    commands['wrap_%s' % key] = d = {}
                                    for v in epdata[k].values():
                                        s = '%s:%s' % (v.prefix, v.suffix)
                                        if v.flags:
                                            s += ' [%s]' % ','.join(v.flags)
                                        d[v.name] = s
                        except Exception:
                            logger.warning('Unable to read legacy script '
                                           'metadata, so cannot generate '
                                           'scripts')
                    else:
                        try:
                            with zf.open(metadata_name) as bwf:
                                wf = wrapper(bwf)
                                commands = json.load(wf).get('extensions')
                                if commands:
                                    commands = commands.get('python.commands')
                        except Exception:
                            logger.warning('Unable to read JSON metadata, so '
                                           'cannot generate scripts')
                    if commands:
                        console_scripts = commands.get('wrap_console', {})
                        gui_scripts = commands.get('wrap_gui', {})
                        if console_scripts or gui_scripts:
                            script_dir = paths.get('scripts', '')
                            if not os.path.isdir(script_dir):
                                raise ValueError('Valid script path not '
                                                 'specified')
                            maker.target_dir = script_dir
                            for k, v in console_scripts.items():
                                script = '%s = %s' % (k, v)
                                filenames = maker.make(script)
                                fileop.set_executable_mode(filenames)

                            if gui_scripts:
                                options = {'gui': True}
                                for k, v in gui_scripts.items():
                                    script = '%s = %s' % (k, v)
                                    filenames = maker.make(script, options)
                                    fileop.set_executable_mode(filenames)

                    p = os.path.join(libdir, info_dir)
                    dist = InstalledDistribution(p)

                    # Write SHARED
                    paths = dict(paths)  # don't change passed in dict
                    del paths['purelib']
                    del paths['platlib']
                    paths['lib'] = libdir
                    p = dist.write_shared_locations(paths, dry_run)
                    if p:
                        outfiles.append(p)

                    # Write RECORD
                    dist.write_installed_files(outfiles, paths['prefix'], dry_run)
                return dist
            except Exception:  # pragma: no cover
                logger.exception('installation failed.')
                fileop.rollback()
                raise
            finally:
                shutil.rmtree(workdir)

    def _get_dylib_cache(self):
        global cache
        if cache is None:
            # Use native string to avoid issues on 2.x: see Python #20140.
            base = os.path.join(get_cache_base(), str('dylib-cache'), '%s.%s' % sys.version_info[:2])
            cache = Cache(base)
        return cache

    def _get_extensions(self):
        pathname = os.path.join(self.dirname, self.filename)
        name_ver = '%s-%s' % (self.name, self.version)
        info_dir = '%s.dist-info' % name_ver
        arcname = posixpath.join(info_dir, 'EXTENSIONS')
        wrapper = codecs.getreader('utf-8')
        result = []
        with ZipFile(pathname, 'r') as zf:
            try:
                with zf.open(arcname) as bf:
                    wf = wrapper(bf)
                    extensions = json.load(wf)
                    cache = self._get_dylib_cache()
                    prefix = cache.prefix_to_dir(self.filename, use_abspath=False)
                    cache_base = os.path.join(cache.base, prefix)
                    if not os.path.isdir(cache_base):
                        os.makedirs(cache_base)
                    for name, relpath in extensions.items():
                        dest = os.path.join(cache_base, convert_path(relpath))
                        if not os.path.exists(dest):
                            extract = True
                        else:
                            file_time = os.stat(dest).st_mtime
                            file_time = datetime.datetime.fromtimestamp(file_time)
                            info = zf.getinfo(relpath)
                            wheel_time = datetime.datetime(*info.date_time)
                            extract = wheel_time > file_time
                        if extract:
                            zf.extract(relpath, cache_base)
                        result.append((name, dest))
            except KeyError:
                pass
        return result

    def is_compatible(self):
        """
        Determine if a wheel is compatible with the running system.
        """
        return is_compatible(self)

    def is_mountable(self):
        """
        Determine if a wheel is asserted as mountable by its metadata.
        """
        return True  # for now - metadata details TBD

    def mount(self, append=False):
        pathname = os.path.abspath(os.path.join(self.dirname, self.filename))
        if not self.is_compatible():
            msg = 'Wheel %s not compatible with this Python.' % pathname
            raise DistlibException(msg)
        if not self.is_mountable():
            msg = 'Wheel %s is marked as not mountable.' % pathname
            raise DistlibException(msg)
        if pathname in sys.path:
            logger.debug('%s already in path', pathname)
        else:
            if append:
                sys.path.append(pathname)
            else:
                sys.path.insert(0, pathname)
            extensions = self._get_extensions()
            if extensions:
                if _hook not in sys.meta_path:
                    sys.meta_path.append(_hook)
                _hook.add(pathname, extensions)

    def unmount(self):
        pathname = os.path.abspath(os.path.join(self.dirname, self.filename))
        if pathname not in sys.path:
            logger.debug('%s not in path', pathname)
        else:
            sys.path.remove(pathname)
            if pathname in _hook.impure_wheels:
                _hook.remove(pathname)
            if not _hook.impure_wheels:
                if _hook in sys.meta_path:
                    sys.meta_path.remove(_hook)

    def verify(self):
        pathname = os.path.join(self.dirname, self.filename)
        name_ver = '%s-%s' % (self.name, self.version)
        # data_dir = '%s.data' % name_ver
        info_dir = '%s.dist-info' % name_ver

        # metadata_name = posixpath.join(info_dir, LEGACY_METADATA_FILENAME)
        wheel_metadata_name = posixpath.join(info_dir, 'WHEEL')
        record_name = posixpath.join(info_dir, 'RECORD')

        wrapper = codecs.getreader('utf-8')

        with ZipFile(pathname, 'r') as zf:
            with zf.open(wheel_metadata_name) as bwf:
                wf = wrapper(bwf)
                message_from_file(wf)
            # wv = message['Wheel-Version'].split('.', 1)
            # file_version = tuple([int(i) for i in wv])
            # TODO version verification

            records = {}
            with zf.open(record_name) as bf:
                with CSVReader(stream=bf) as reader:
                    for row in reader:
                        p = row[0]
                        records[p] = row

            for zinfo in zf.infolist():
                arcname = zinfo.filename
                if isinstance(arcname, text_type):
                    u_arcname = arcname
                else:
                    u_arcname = arcname.decode('utf-8')
                # See issue #115: some wheels have .. in their entries, but
                # in the filename ... e.g. __main__..py ! So the check is
                # updated to look for .. in the directory portions
                p = u_arcname.split('/')
                if '..' in p:
                    raise DistlibException('invalid entry in '
                                           'wheel: %r' % u_arcname)

                if self.skip_entry(u_arcname):
                    continue
                row = records[u_arcname]
                if row[2] and str(zinfo.file_size) != row[2]:
                    raise DistlibException('size mismatch for '
                                           '%s' % u_arcname)
                if row[1]:
                    kind, value = row[1].split('=', 1)
                    with zf.open(arcname) as bf:
                        data = bf.read()
                    _, digest = self.get_hash(data, kind)
                    if digest != value:
                        raise DistlibException('digest mismatch for '
                                               '%s' % arcname)

    def update(self, modifier, dest_dir=None, **kwargs):
        """
        Update the contents of a wheel in a generic way. The modifier should
        be a callable which expects a dictionary argument: its keys are
        archive-entry paths, and its values are absolute filesystem paths
        where the contents the corresponding archive entries can be found. The
        modifier is free to change the contents of the files pointed to, add
        new entries and remove entries, before returning. This method will
        extract the entire contents of the wheel to a temporary location, call
        the modifier, and then use the passed (and possibly updated)
        dictionary to write a new wheel. If ``dest_dir`` is specified, the new
        wheel is written there -- otherwise, the original wheel is overwritten.

        The modifier should return True if it updated the wheel, else False.
        This method returns the same value the modifier returns.
        """

        def get_version(path_map, info_dir):
            version = path = None
            key = '%s/%s' % (info_dir, LEGACY_METADATA_FILENAME)
            if key not in path_map:
                key = '%s/PKG-INFO' % info_dir
            if key in path_map:
                path = path_map[key]
                version = Metadata(path=path).version
            return version, path

        def update_version(version, path):
            updated = None
            try:
                NormalizedVersion(version)
                i = version.find('-')
                if i < 0:
                    updated = '%s+1' % version
                else:
                    parts = [int(s) for s in version[i + 1:].split('.')]
                    parts[-1] += 1
                    updated = '%s+%s' % (version[:i], '.'.join(str(i) for i in parts))
            except UnsupportedVersionError:
                logger.debug('Cannot update non-compliant (PEP-440) '
                             'version %r', version)
            if updated:
                md = Metadata(path=path)
                md.version = updated
                legacy = path.endswith(LEGACY_METADATA_FILENAME)
                md.write(path=path, legacy=legacy)
                logger.debug('Version updated from %r to %r', version, updated)

        pathname = os.path.join(self.dirname, self.filename)
        name_ver = '%s-%s' % (self.name, self.version)
        info_dir = '%s.dist-info' % name_ver
        record_name = posixpath.join(info_dir, 'RECORD')
        with tempdir() as workdir:
            with ZipFile(pathname, 'r') as zf:
                path_map = {}
                for zinfo in zf.infolist():
                    arcname = zinfo.filename
                    if isinstance(arcname, text_type):
                        u_arcname = arcname
                    else:
                        u_arcname = arcname.decode('utf-8')
                    if u_arcname == record_name:
                        continue
                    if '..' in u_arcname:
                        raise DistlibException('invalid entry in '
                                               'wheel: %r' % u_arcname)
                    zf.extract(zinfo, workdir)
                    path = os.path.join(workdir, convert_path(u_arcname))
                    path_map[u_arcname] = path

            # Remember the version.
            original_version, _ = get_version(path_map, info_dir)
            # Files extracted. Call the modifier.
            modified = modifier(path_map, **kwargs)
            if modified:
                # Something changed - need to build a new wheel.
                current_version, path = get_version(path_map, info_dir)
                if current_version and (current_version == original_version):
                    # Add or update local version to signify changes.
                    update_version(current_version, path)
                # Decide where the new wheel goes.
                if dest_dir is None:
                    fd, newpath = tempfile.mkstemp(suffix='.whl', prefix='wheel-update-', dir=workdir)
                    os.close(fd)
                else:
                    if not os.path.isdir(dest_dir):
                        raise DistlibException('Not a directory: %r' % dest_dir)
                    newpath = os.path.join(dest_dir, self.filename)
                archive_paths = list(path_map.items())
                distinfo = os.path.join(workdir, info_dir)
                info = distinfo, info_dir
                self.write_records(info, workdir, archive_paths)
                self.build_zip(newpath, archive_paths)
                if dest_dir is None:
                    shutil.copyfile(newpath, pathname)
        return modified


def _get_glibc_version():
    import platform
    ver = platform.libc_ver()
    result = []
    if ver[0] == 'glibc':
        for s in ver[1].split('.'):
            result.append(int(s) if s.isdigit() else 0)
        result = tuple(result)
    return result


def compatible_tags():
    """
    Return (pyver, abi, arch) tuples compatible with this Python.
    """
    class _Version:
        def __init__(self, major, minor):
            self.major = major
            self.major_minor = (major, minor)
            self.string = ''.join((str(major), str(minor)))

        def __str__(self):
            return self.string


    versions = [
        _Version(sys.version_info.major, minor_version)
        for minor_version in range(sys.version_info.minor, -1, -1)
    ]
    abis = []
    for suffix in _get_suffixes():
        if suffix.startswith('.abi'):
            abis.append(suffix.split('.', 2)[1])
    abis.sort()
    if ABI != 'none':
        abis.insert(0, ABI)
    abis.append('none')
    result = []

    arches = [ARCH]
    if sys.platform == 'darwin':
        m = re.match(r'(\w+)_(\d+)_(\d+)_(\w+)$', ARCH)
        if m:
            name, major, minor, arch = m.groups()
            minor = int(minor)
            matches = [arch]
            if arch in ('i386', 'ppc'):
                matches.append('fat')
            if arch in ('i386', 'ppc', 'x86_64'):
                matches.append('fat3')
            if arch in ('ppc64', 'x86_64'):
                matches.append('fat64')
            if arch in ('i386', 'x86_64'):
                matches.append('intel')
            if arch in ('i386', 'x86_64', 'intel', 'ppc', 'ppc64'):
                matches.append('universal')
            while minor >= 0:
                for match in matches:
                    s = '%s_%s_%s_%s' % (name, major, minor, match)
                    if s != ARCH:  # already there
                        arches.append(s)
                minor -= 1

    # Most specific - our Python version, ABI and arch
    for i, version_object in enumerate(versions):
        version = str(version_object)
        add_abis = []

        if i == 0:
            add_abis = abis

        if IMP_PREFIX == 'cp' and version_object.major_minor >= (3, 2):
            limited_api_abi = 'abi' + str(version_object.major)
            if limited_api_abi not in add_abis:
                add_abis.append(limited_api_abi)

        for abi in add_abis:
            for arch in arches:
                result.append((''.join((IMP_PREFIX, version)), abi, arch))
                # manylinux
                if abi != 'none' and sys.platform.startswith('linux'):
                    arch = arch.replace('linux_', '')
                    parts = _get_glibc_version()
                    if len(parts) == 2:
                        if parts >= (2, 5):
                            result.append((''.join((IMP_PREFIX, version)), abi, 'manylinux1_%s' % arch))
                        if parts >= (2, 12):
                            result.append((''.join((IMP_PREFIX, version)), abi, 'manylinux2010_%s' % arch))
                        if parts >= (2, 17):
                            result.append((''.join((IMP_PREFIX, version)), abi, 'manylinux2014_%s' % arch))
                        result.append((''.join(
                            (IMP_PREFIX, version)), abi, 'manylinux_%s_%s_%s' % (parts[0], parts[1], arch)))

    # where no ABI / arch dependency, but IMP_PREFIX dependency
    for i, version_object in enumerate(versions):
        version = str(version_object)
        result.append((''.join((IMP_PREFIX, version)), 'none', 'any'))
        if i == 0:
            result.append((''.join((IMP_PREFIX, version[0])), 'none', 'any'))

    # no IMP_PREFIX, ABI or arch dependency
    for i, version_object in enumerate(versions):
        version = str(version_object)
        result.append((''.join(('py', version)), 'none', 'any'))
        if i == 0:
            result.append((''.join(('py', version[0])), 'none', 'any'))

    return set(result)


COMPATIBLE_TAGS = compatible_tags()

del compatible_tags


def is_compatible(wheel, tags=None):
    if not isinstance(wheel, Wheel):
        wheel = Wheel(wheel)  # assume it's a filename
    result = False
    if tags is None:
        tags = COMPATIBLE_TAGS
    for ver, abi, arch in tags:
        if ver in wheel.pyver and abi in wheel.abi and arch in wheel.arch:
            result = True
            break
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\ply\lex.py ===
# -----------------------------------------------------------------------------
# ply: lex.py
#
# Copyright (C) 2001-2017
# David M. Beazley (Dabeaz LLC)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the David Beazley or Dabeaz LLC may be used to
#   endorse or promote products derived from this software without
#  specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

__version__    = '3.10'
__tabversion__ = '3.10'

import re
import sys
import types
import copy
import os
import inspect

# This tuple contains known string types
try:
    # Python 2.6
    StringTypes = (types.StringType, types.UnicodeType)
except AttributeError:
    # Python 3.0
    StringTypes = (str, bytes)

# This regular expression is used to match valid token names
_is_identifier = re.compile(r'^[a-zA-Z0-9_]+$')

# Exception thrown when invalid token encountered and no default error
# handler is defined.
class LexError(Exception):
    def __init__(self, message, s):
        self.args = (message,)
        self.text = s


# Token class.  This class is used to represent the tokens produced.
class LexToken(object):
    def __str__(self):
        return 'LexToken(%s,%r,%d,%d)' % (self.type, self.value, self.lineno, self.lexpos)

    def __repr__(self):
        return str(self)


# This object is a stand-in for a logging object created by the
# logging module.

class PlyLogger(object):
    def __init__(self, f):
        self.f = f

    def critical(self, msg, *args, **kwargs):
        self.f.write((msg % args) + '\n')

    def warning(self, msg, *args, **kwargs):
        self.f.write('WARNING: ' + (msg % args) + '\n')

    def error(self, msg, *args, **kwargs):
        self.f.write('ERROR: ' + (msg % args) + '\n')

    info = critical
    debug = critical


# Null logger is used when no output is generated. Does nothing.
class NullLogger(object):
    def __getattribute__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


# -----------------------------------------------------------------------------
#                        === Lexing Engine ===
#
# The following Lexer class implements the lexer runtime.   There are only
# a few public methods and attributes:
#
#    input()          -  Store a new string in the lexer
#    token()          -  Get the next token
#    clone()          -  Clone the lexer
#
#    lineno           -  Current line number
#    lexpos           -  Current position in the input string
# -----------------------------------------------------------------------------

class Lexer:
    def __init__(self):
        self.lexre = None             # Master regular expression. This is a list of
                                      # tuples (re, findex) where re is a compiled
                                      # regular expression and findex is a list
                                      # mapping regex group numbers to rules
        self.lexretext = None         # Current regular expression strings
        self.lexstatere = {}          # Dictionary mapping lexer states to master regexs
        self.lexstateretext = {}      # Dictionary mapping lexer states to regex strings
        self.lexstaterenames = {}     # Dictionary mapping lexer states to symbol names
        self.lexstate = 'INITIAL'     # Current lexer state
        self.lexstatestack = []       # Stack of lexer states
        self.lexstateinfo = None      # State information
        self.lexstateignore = {}      # Dictionary of ignored characters for each state
        self.lexstateerrorf = {}      # Dictionary of error functions for each state
        self.lexstateeoff = {}        # Dictionary of eof functions for each state
        self.lexreflags = 0           # Optional re compile flags
        self.lexdata = None           # Actual input data (as a string)
        self.lexpos = 0               # Current position in input text
        self.lexlen = 0               # Length of the input text
        self.lexerrorf = None         # Error rule (if any)
        self.lexeoff = None           # EOF rule (if any)
        self.lextokens = None         # List of valid tokens
        self.lexignore = ''           # Ignored characters
        self.lexliterals = ''         # Literal characters that can be passed through
        self.lexmodule = None         # Module
        self.lineno = 1               # Current line number
        self.lexoptimize = False      # Optimized mode

    def clone(self, object=None):
        c = copy.copy(self)

        # If the object parameter has been supplied, it means we are attaching the
        # lexer to a new object.  In this case, we have to rebind all methods in
        # the lexstatere and lexstateerrorf tables.

        if object:
            newtab = {}
            for key, ritem in self.lexstatere.items():
                newre = []
                for cre, findex in ritem:
                    newfindex = []
                    for f in findex:
                        if not f or not f[0]:
                            newfindex.append(f)
                            continue
                        newfindex.append((getattr(object, f[0].__name__), f[1]))
                newre.append((cre, newfindex))
                newtab[key] = newre
            c.lexstatere = newtab
            c.lexstateerrorf = {}
            for key, ef in self.lexstateerrorf.items():
                c.lexstateerrorf[key] = getattr(object, ef.__name__)
            c.lexmodule = object
        return c

    # ------------------------------------------------------------
    # writetab() - Write lexer information to a table file
    # ------------------------------------------------------------
    def writetab(self, lextab, outputdir=''):
        if isinstance(lextab, types.ModuleType):
            raise IOError("Won't overwrite existing lextab module")
        basetabmodule = lextab.split('.')[-1]
        filename = os.path.join(outputdir, basetabmodule) + '.py'
        with open(filename, 'w') as tf:
            tf.write('# %s.py. This file automatically created by PLY (version %s). Don\'t edit!\n' % (basetabmodule, __version__))
            tf.write('_tabversion   = %s\n' % repr(__tabversion__))
            tf.write('_lextokens    = set(%s)\n' % repr(tuple(sorted(self.lextokens))))
            tf.write('_lexreflags   = %s\n' % repr(self.lexreflags))
            tf.write('_lexliterals  = %s\n' % repr(self.lexliterals))
            tf.write('_lexstateinfo = %s\n' % repr(self.lexstateinfo))

            # Rewrite the lexstatere table, replacing function objects with function names
            tabre = {}
            for statename, lre in self.lexstatere.items():
                titem = []
                for (pat, func), retext, renames in zip(lre, self.lexstateretext[statename], self.lexstaterenames[statename]):
                    titem.append((retext, _funcs_to_names(func, renames)))
                tabre[statename] = titem

            tf.write('_lexstatere   = %s\n' % repr(tabre))
            tf.write('_lexstateignore = %s\n' % repr(self.lexstateignore))

            taberr = {}
            for statename, ef in self.lexstateerrorf.items():
                taberr[statename] = ef.__name__ if ef else None
            tf.write('_lexstateerrorf = %s\n' % repr(taberr))

            tabeof = {}
            for statename, ef in self.lexstateeoff.items():
                tabeof[statename] = ef.__name__ if ef else None
            tf.write('_lexstateeoff = %s\n' % repr(tabeof))

    # ------------------------------------------------------------
    # readtab() - Read lexer information from a tab file
    # ------------------------------------------------------------
    def readtab(self, tabfile, fdict):
        if isinstance(tabfile, types.ModuleType):
            lextab = tabfile
        else:
            exec('import %s' % tabfile)
            lextab = sys.modules[tabfile]

        if getattr(lextab, '_tabversion', '0.0') != __tabversion__:
            raise ImportError('Inconsistent PLY version')

        self.lextokens      = lextab._lextokens
        self.lexreflags     = lextab._lexreflags
        self.lexliterals    = lextab._lexliterals
        self.lextokens_all  = self.lextokens | set(self.lexliterals)
        self.lexstateinfo   = lextab._lexstateinfo
        self.lexstateignore = lextab._lexstateignore
        self.lexstatere     = {}
        self.lexstateretext = {}
        for statename, lre in lextab._lexstatere.items():
            titem = []
            txtitem = []
            for pat, func_name in lre:
                titem.append((re.compile(pat, lextab._lexreflags), _names_to_funcs(func_name, fdict)))

            self.lexstatere[statename] = titem
            self.lexstateretext[statename] = txtitem

        self.lexstateerrorf = {}
        for statename, ef in lextab._lexstateerrorf.items():
            self.lexstateerrorf[statename] = fdict[ef]

        self.lexstateeoff = {}
        for statename, ef in lextab._lexstateeoff.items():
            self.lexstateeoff[statename] = fdict[ef]

        self.begin('INITIAL')

    # ------------------------------------------------------------
    # input() - Push a new string into the lexer
    # ------------------------------------------------------------
    def input(self, s):
        # Pull off the first character to see if s looks like a string
        c = s[:1]
        if not isinstance(c, StringTypes):
            raise ValueError('Expected a string')
        self.lexdata = s
        self.lexpos = 0
        self.lexlen = len(s)

    # ------------------------------------------------------------
    # begin() - Changes the lexing state
    # ------------------------------------------------------------
    def begin(self, state):
        if state not in self.lexstatere:
            raise ValueError('Undefined state')
        self.lexre = self.lexstatere[state]
        self.lexretext = self.lexstateretext[state]
        self.lexignore = self.lexstateignore.get(state, '')
        self.lexerrorf = self.lexstateerrorf.get(state, None)
        self.lexeoff = self.lexstateeoff.get(state, None)
        self.lexstate = state

    # ------------------------------------------------------------
    # push_state() - Changes the lexing state and saves old on stack
    # ------------------------------------------------------------
    def push_state(self, state):
        self.lexstatestack.append(self.lexstate)
        self.begin(state)

    # ------------------------------------------------------------
    # pop_state() - Restores the previous state
    # ------------------------------------------------------------
    def pop_state(self):
        self.begin(self.lexstatestack.pop())

    # ------------------------------------------------------------
    # current_state() - Returns the current lexing state
    # ------------------------------------------------------------
    def current_state(self):
        return self.lexstate

    # ------------------------------------------------------------
    # skip() - Skip ahead n characters
    # ------------------------------------------------------------
    def skip(self, n):
        self.lexpos += n

    # ------------------------------------------------------------
    # opttoken() - Return the next token from the Lexer
    #
    # Note: This function has been carefully implemented to be as fast
    # as possible.  Don't make changes unless you really know what
    # you are doing
    # ------------------------------------------------------------
    def token(self):
        # Make local copies of frequently referenced attributes
        lexpos    = self.lexpos
        lexlen    = self.lexlen
        lexignore = self.lexignore
        lexdata   = self.lexdata

        while lexpos < lexlen:
            # This code provides some short-circuit code for whitespace, tabs, and other ignored characters
            if lexdata[lexpos] in lexignore:
                lexpos += 1
                continue

            # Look for a regular expression match
            for lexre, lexindexfunc in self.lexre:
                m = lexre.match(lexdata, lexpos)
                if not m:
                    continue

                # Create a token for return
                tok = LexToken()
                tok.value = m.group()
                tok.lineno = self.lineno
                tok.lexpos = lexpos

                i = m.lastindex
                func, tok.type = lexindexfunc[i]

                if not func:
                    # If no token type was set, it's an ignored token
                    if tok.type:
                        self.lexpos = m.end()
                        return tok
                    else:
                        lexpos = m.end()
                        break

                lexpos = m.end()

                # If token is processed by a function, call it

                tok.lexer = self      # Set additional attributes useful in token rules
                self.lexmatch = m
                self.lexpos = lexpos

                newtok = func(tok)

                # Every function must return a token, if nothing, we just move to next token
                if not newtok:
                    lexpos    = self.lexpos         # This is here in case user has updated lexpos.
                    lexignore = self.lexignore      # This is here in case there was a state change
                    break

                # Verify type of the token.  If not in the token map, raise an error
                if not self.lexoptimize:
                    if newtok.type not in self.lextokens_all:
                        raise LexError("%s:%d: Rule '%s' returned an unknown token type '%s'" % (
                            func.__code__.co_filename, func.__code__.co_firstlineno,
                            func.__name__, newtok.type), lexdata[lexpos:])

                return newtok
            else:
                # No match, see if in literals
                if lexdata[lexpos] in self.lexliterals:
                    tok = LexToken()
                    tok.value = lexdata[lexpos]
                    tok.lineno = self.lineno
                    tok.type = tok.value
                    tok.lexpos = lexpos
                    self.lexpos = lexpos + 1
                    return tok

                # No match. Call t_error() if defined.
                if self.lexerrorf:
                    tok = LexToken()
                    tok.value = self.lexdata[lexpos:]
                    tok.lineno = self.lineno
                    tok.type = 'error'
                    tok.lexer = self
                    tok.lexpos = lexpos
                    self.lexpos = lexpos
                    newtok = self.lexerrorf(tok)
                    if lexpos == self.lexpos:
                        # Error method didn't change text position at all. This is an error.
                        raise LexError("Scanning error. Illegal character '%s'" % (lexdata[lexpos]), lexdata[lexpos:])
                    lexpos = self.lexpos
                    if not newtok:
                        continue
                    return newtok

                self.lexpos = lexpos
                raise LexError("Illegal character '%s' at index %d" % (lexdata[lexpos], lexpos), lexdata[lexpos:])

        if self.lexeoff:
            tok = LexToken()
            tok.type = 'eof'
            tok.value = ''
            tok.lineno = self.lineno
            tok.lexpos = lexpos
            tok.lexer = self
            self.lexpos = lexpos
            newtok = self.lexeoff(tok)
            return newtok

        self.lexpos = lexpos + 1
        if self.lexdata is None:
            raise RuntimeError('No input string given with input()')
        return None

    # Iterator interface
    def __iter__(self):
        return self

    def next(self):
        t = self.token()
        if t is None:
            raise StopIteration
        return t

    __next__ = next

# -----------------------------------------------------------------------------
#                           ==== Lex Builder ===
#
# The functions and classes below are used to collect lexing information
# and build a Lexer object from it.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# _get_regex(func)
#
# Returns the regular expression assigned to a function either as a doc string
# or as a .regex attribute attached by the @TOKEN decorator.
# -----------------------------------------------------------------------------
def _get_regex(func):
    return getattr(func, 'regex', func.__doc__)

# -----------------------------------------------------------------------------
# get_caller_module_dict()
#
# This function returns a dictionary containing all of the symbols defined within
# a caller further down the call stack.  This is used to get the environment
# associated with the yacc() call if none was provided.
# -----------------------------------------------------------------------------
def get_caller_module_dict(levels):
    f = sys._getframe(levels)
    ldict = f.f_globals.copy()
    if f.f_globals != f.f_locals:
        ldict.update(f.f_locals)
    return ldict

# -----------------------------------------------------------------------------
# _funcs_to_names()
#
# Given a list of regular expression functions, this converts it to a list
# suitable for output to a table file
# -----------------------------------------------------------------------------
def _funcs_to_names(funclist, namelist):
    result = []
    for f, name in zip(funclist, namelist):
        if f and f[0]:
            result.append((name, f[1]))
        else:
            result.append(f)
    return result

# -----------------------------------------------------------------------------
# _names_to_funcs()
#
# Given a list of regular expression function names, this converts it back to
# functions.
# -----------------------------------------------------------------------------
def _names_to_funcs(namelist, fdict):
    result = []
    for n in namelist:
        if n and n[0]:
            result.append((fdict[n[0]], n[1]))
        else:
            result.append(n)
    return result

# -----------------------------------------------------------------------------
# _form_master_re()
#
# This function takes a list of all of the regex components and attempts to
# form the master regular expression.  Given limitations in the Python re
# module, it may be necessary to break the master regex into separate expressions.
# -----------------------------------------------------------------------------
def _form_master_re(relist, reflags, ldict, toknames):
    if not relist:
        return []
    regex = '|'.join(relist)
    try:
        lexre = re.compile(regex, reflags)

        # Build the index to function map for the matching engine
        lexindexfunc = [None] * (max(lexre.groupindex.values()) + 1)
        lexindexnames = lexindexfunc[:]

        for f, i in lexre.groupindex.items():
            handle = ldict.get(f, None)
            if type(handle) in (types.FunctionType, types.MethodType):
                lexindexfunc[i] = (handle, toknames[f])
                lexindexnames[i] = f
            elif handle is not None:
                lexindexnames[i] = f
                if f.find('ignore_') > 0:
                    lexindexfunc[i] = (None, None)
                else:
                    lexindexfunc[i] = (None, toknames[f])

        return [(lexre, lexindexfunc)], [regex], [lexindexnames]
    except Exception:
        m = int(len(relist)/2)
        if m == 0:
            m = 1
        llist, lre, lnames = _form_master_re(relist[:m], reflags, ldict, toknames)
        rlist, rre, rnames = _form_master_re(relist[m:], reflags, ldict, toknames)
        return (llist+rlist), (lre+rre), (lnames+rnames)

# -----------------------------------------------------------------------------
# def _statetoken(s,names)
#
# Given a declaration name s of the form "t_" and a dictionary whose keys are
# state names, this function returns a tuple (states,tokenname) where states
# is a tuple of state names and tokenname is the name of the token.  For example,
# calling this with s = "t_foo_bar_SPAM" might return (('foo','bar'),'SPAM')
# -----------------------------------------------------------------------------
def _statetoken(s, names):
    nonstate = 1
    parts = s.split('_')
    for i, part in enumerate(parts[1:], 1):
        if part not in names and part != 'ANY':
            break

    if i > 1:
        states = tuple(parts[1:i])
    else:
        states = ('INITIAL',)

    if 'ANY' in states:
        states = tuple(names)

    tokenname = '_'.join(parts[i:])
    return (states, tokenname)


# -----------------------------------------------------------------------------
# LexerReflect()
#
# This class represents information needed to build a lexer as extracted from a
# user's input file.
# -----------------------------------------------------------------------------
class LexerReflect(object):
    def __init__(self, ldict, log=None, reflags=0):
        self.ldict      = ldict
        self.error_func = None
        self.tokens     = []
        self.reflags    = reflags
        self.stateinfo  = {'INITIAL': 'inclusive'}
        self.modules    = set()
        self.error      = False
        self.log        = PlyLogger(sys.stderr) if log is None else log

    # Get all of the basic information
    def get_all(self):
        self.get_tokens()
        self.get_literals()
        self.get_states()
        self.get_rules()

    # Validate all of the information
    def validate_all(self):
        self.validate_tokens()
        self.validate_literals()
        self.validate_rules()
        return self.error

    # Get the tokens map
    def get_tokens(self):
        tokens = self.ldict.get('tokens', None)
        if not tokens:
            self.log.error('No token list is defined')
            self.error = True
            return

        if not isinstance(tokens, (list, tuple)):
            self.log.error('tokens must be a list or tuple')
            self.error = True
            return

        if not tokens:
            self.log.error('tokens is empty')
            self.error = True
            return

        self.tokens = tokens

    # Validate the tokens
    def validate_tokens(self):
        terminals = {}
        for n in self.tokens:
            if not _is_identifier.match(n):
                self.log.error("Bad token name '%s'", n)
                self.error = True
            if n in terminals:
                self.log.warning("Token '%s' multiply defined", n)
            terminals[n] = 1

    # Get the literals specifier
    def get_literals(self):
        self.literals = self.ldict.get('literals', '')
        if not self.literals:
            self.literals = ''

    # Validate literals
    def validate_literals(self):
        try:
            for c in self.literals:
                if not isinstance(c, StringTypes) or len(c) > 1:
                    self.log.error('Invalid literal %s. Must be a single character', repr(c))
                    self.error = True

        except TypeError:
            self.log.error('Invalid literals specification. literals must be a sequence of characters')
            self.error = True

    def get_states(self):
        self.states = self.ldict.get('states', None)
        # Build statemap
        if self.states:
            if not isinstance(self.states, (tuple, list)):
                self.log.error('states must be defined as a tuple or list')
                self.error = True
            else:
                for s in self.states:
                    if not isinstance(s, tuple) or len(s) != 2:
                        self.log.error("Invalid state specifier %s. Must be a tuple (statename,'exclusive|inclusive')", repr(s))
                        self.error = True
                        continue
                    name, statetype = s
                    if not isinstance(name, StringTypes):
                        self.log.error('State name %s must be a string', repr(name))
                        self.error = True
                        continue
                    if not (statetype == 'inclusive' or statetype == 'exclusive'):
                        self.log.error("State type for state %s must be 'inclusive' or 'exclusive'", name)
                        self.error = True
                        continue
                    if name in self.stateinfo:
                        self.log.error("State '%s' already defined", name)
                        self.error = True
                        continue
                    self.stateinfo[name] = statetype

    # Get all of the symbols with a t_ prefix and sort them into various
    # categories (functions, strings, error functions, and ignore characters)

    def get_rules(self):
        tsymbols = [f for f in self.ldict if f[:2] == 't_']

        # Now build up a list of functions and a list of strings
        self.toknames = {}        # Mapping of symbols to token names
        self.funcsym  = {}        # Symbols defined as functions
        self.strsym   = {}        # Symbols defined as strings
        self.ignore   = {}        # Ignore strings by state
        self.errorf   = {}        # Error functions by state
        self.eoff     = {}        # EOF functions by state

        for s in self.stateinfo:
            self.funcsym[s] = []
            self.strsym[s] = []

        if len(tsymbols) == 0:
            self.log.error('No rules of the form t_rulename are defined')
            self.error = True
            return

        for f in tsymbols:
            t = self.ldict[f]
            states, tokname = _statetoken(f, self.stateinfo)
            self.toknames[f] = tokname

            if hasattr(t, '__call__'):
                if tokname == 'error':
                    for s in states:
                        self.errorf[s] = t
                elif tokname == 'eof':
                    for s in states:
                        self.eoff[s] = t
                elif tokname == 'ignore':
                    line = t.__code__.co_firstlineno
                    file = t.__code__.co_filename
                    self.log.error("%s:%d: Rule '%s' must be defined as a string", file, line, t.__name__)
                    self.error = True
                else:
                    for s in states:
                        self.funcsym[s].append((f, t))
            elif isinstance(t, StringTypes):
                if tokname == 'ignore':
                    for s in states:
                        self.ignore[s] = t
                    if '\\' in t:
                        self.log.warning("%s contains a literal backslash '\\'", f)

                elif tokname == 'error':
                    self.log.error("Rule '%s' must be defined as a function", f)
                    self.error = True
                else:
                    for s in states:
                        self.strsym[s].append((f, t))
            else:
                self.log.error('%s not defined as a function or string', f)
                self.error = True

        # Sort the functions by line number
        for f in self.funcsym.values():
            f.sort(key=lambda x: x[1].__code__.co_firstlineno)

        # Sort the strings by regular expression length
        for s in self.strsym.values():
            s.sort(key=lambda x: len(x[1]), reverse=True)

    # Validate all of the t_rules collected
    def validate_rules(self):
        for state in self.stateinfo:
            # Validate all rules defined by functions

            for fname, f in self.funcsym[state]:
                line = f.__code__.co_firstlineno
                file = f.__code__.co_filename
                module = inspect.getmodule(f)
                self.modules.add(module)

                tokname = self.toknames[fname]
                if isinstance(f, types.MethodType):
                    reqargs = 2
                else:
                    reqargs = 1
                nargs = f.__code__.co_argcount
                if nargs > reqargs:
                    self.log.error("%s:%d: Rule '%s' has too many arguments", file, line, f.__name__)
                    self.error = True
                    continue

                if nargs < reqargs:
                    self.log.error("%s:%d: Rule '%s' requires an argument", file, line, f.__name__)
                    self.error = True
                    continue

                if not _get_regex(f):
                    self.log.error("%s:%d: No regular expression defined for rule '%s'", file, line, f.__name__)
                    self.error = True
                    continue

                try:
                    c = re.compile('(?P<%s>%s)' % (fname, _get_regex(f)), self.reflags)
                    if c.match(''):
                        self.log.error("%s:%d: Regular expression for rule '%s' matches empty string", file, line, f.__name__)
                        self.error = True
                except re.error as e:
                    self.log.error("%s:%d: Invalid regular expression for rule '%s'. %s", file, line, f.__name__, e)
                    if '#' in _get_regex(f):
                        self.log.error("%s:%d. Make sure '#' in rule '%s' is escaped with '\\#'", file, line, f.__name__)
                    self.error = True

            # Validate all rules defined by strings
            for name, r in self.strsym[state]:
                tokname = self.toknames[name]
                if tokname == 'error':
                    self.log.error("Rule '%s' must be defined as a function", name)
                    self.error = True
                    continue

                if tokname not in self.tokens and tokname.find('ignore_') < 0:
                    self.log.error("Rule '%s' defined for an unspecified token %s", name, tokname)
                    self.error = True
                    continue

                try:
                    c = re.compile('(?P<%s>%s)' % (name, r), self.reflags)
                    if (c.match('')):
                        self.log.error("Regular expression for rule '%s' matches empty string", name)
                        self.error = True
                except re.error as e:
                    self.log.error("Invalid regular expression for rule '%s'. %s", name, e)
                    if '#' in r:
                        self.log.error("Make sure '#' in rule '%s' is escaped with '\\#'", name)
                    self.error = True

            if not self.funcsym[state] and not self.strsym[state]:
                self.log.error("No rules defined for state '%s'", state)
                self.error = True

            # Validate the error function
            efunc = self.errorf.get(state, None)
            if efunc:
                f = efunc
                line = f.__code__.co_firstlineno
                file = f.__code__.co_filename
                module = inspect.getmodule(f)
                self.modules.add(module)

                if isinstance(f, types.MethodType):
                    reqargs = 2
                else:
                    reqargs = 1
                nargs = f.__code__.co_argcount
                if nargs > reqargs:
                    self.log.error("%s:%d: Rule '%s' has too many arguments", file, line, f.__name__)
                    self.error = True

                if nargs < reqargs:
                    self.log.error("%s:%d: Rule '%s' requires an argument", file, line, f.__name__)
                    self.error = True

        for module in self.modules:
            self.validate_module(module)

    # -----------------------------------------------------------------------------
    # validate_module()
    #
    # This checks to see if there are duplicated t_rulename() functions or strings
    # in the parser input file.  This is done using a simple regular expression
    # match on each line in the source code of the given module.
    # -----------------------------------------------------------------------------

    def validate_module(self, module):
        try:
            lines, linen = inspect.getsourcelines(module)
        except IOError:
            return

        fre = re.compile(r'\s*def\s+(t_[a-zA-Z_0-9]*)\(')
        sre = re.compile(r'\s*(t_[a-zA-Z_0-9]*)\s*=')

        counthash = {}
        linen += 1
        for line in lines:
            m = fre.match(line)
            if not m:
                m = sre.match(line)
            if m:
                name = m.group(1)
                prev = counthash.get(name)
                if not prev:
                    counthash[name] = linen
                else:
                    filename = inspect.getsourcefile(module)
                    self.log.error('%s:%d: Rule %s redefined. Previously defined on line %d', filename, linen, name, prev)
                    self.error = True
            linen += 1

# -----------------------------------------------------------------------------
# lex(module)
#
# Build all of the regular expression rules from definitions in the supplied module
# -----------------------------------------------------------------------------
def lex(module=None, object=None, debug=False, optimize=False, lextab='lextab',
        reflags=int(re.VERBOSE), nowarn=False, outputdir=None, debuglog=None, errorlog=None):

    if lextab is None:
        lextab = 'lextab'

    global lexer

    ldict = None
    stateinfo  = {'INITIAL': 'inclusive'}
    lexobj = Lexer()
    lexobj.lexoptimize = optimize
    global token, input

    if errorlog is None:
        errorlog = PlyLogger(sys.stderr)

    if debug:
        if debuglog is None:
            debuglog = PlyLogger(sys.stderr)

    # Get the module dictionary used for the lexer
    if object:
        module = object

    # Get the module dictionary used for the parser
    if module:
        _items = [(k, getattr(module, k)) for k in dir(module)]
        ldict = dict(_items)
        # If no __file__ attribute is available, try to obtain it from the __module__ instead
        if '__file__' not in ldict:
            ldict['__file__'] = sys.modules[ldict['__module__']].__file__
    else:
        ldict = get_caller_module_dict(2)

    # Determine if the module is package of a package or not.
    # If so, fix the tabmodule setting so that tables load correctly
    pkg = ldict.get('__package__')
    if pkg and isinstance(lextab, str):
        if '.' not in lextab:
            lextab = pkg + '.' + lextab

    # Collect parser information from the dictionary
    linfo = LexerReflect(ldict, log=errorlog, reflags=reflags)
    linfo.get_all()
    if not optimize:
        if linfo.validate_all():
            raise SyntaxError("Can't build lexer")

    if optimize and lextab:
        try:
            lexobj.readtab(lextab, ldict)
            token = lexobj.token
            input = lexobj.input
            lexer = lexobj
            return lexobj

        except ImportError:
            pass

    # Dump some basic debugging information
    if debug:
        debuglog.info('lex: tokens   = %r', linfo.tokens)
        debuglog.info('lex: literals = %r', linfo.literals)
        debuglog.info('lex: states   = %r', linfo.stateinfo)

    # Build a dictionary of valid token names
    lexobj.lextokens = set()
    for n in linfo.tokens:
        lexobj.lextokens.add(n)

    # Get literals specification
    if isinstance(linfo.literals, (list, tuple)):
        lexobj.lexliterals = type(linfo.literals[0])().join(linfo.literals)
    else:
        lexobj.lexliterals = linfo.literals

    lexobj.lextokens_all = lexobj.lextokens | set(lexobj.lexliterals)

    # Get the stateinfo dictionary
    stateinfo = linfo.stateinfo

    regexs = {}
    # Build the master regular expressions
    for state in stateinfo:
        regex_list = []

        # Add rules defined by functions first
        for fname, f in linfo.funcsym[state]:
            line = f.__code__.co_firstlineno
            file = f.__code__.co_filename
            regex_list.append('(?P<%s>%s)' % (fname, _get_regex(f)))
            if debug:
                debuglog.info("lex: Adding rule %s -> '%s' (state '%s')", fname, _get_regex(f), state)

        # Now add all of the simple rules
        for name, r in linfo.strsym[state]:
            regex_list.append('(?P<%s>%s)' % (name, r))
            if debug:
                debuglog.info("lex: Adding rule %s -> '%s' (state '%s')", name, r, state)

        regexs[state] = regex_list

    # Build the master regular expressions

    if debug:
        debuglog.info('lex: ==== MASTER REGEXS FOLLOW ====')

    for state in regexs:
        lexre, re_text, re_names = _form_master_re(regexs[state], reflags, ldict, linfo.toknames)
        lexobj.lexstatere[state] = lexre
        lexobj.lexstateretext[state] = re_text
        lexobj.lexstaterenames[state] = re_names
        if debug:
            for i, text in enumerate(re_text):
                debuglog.info("lex: state '%s' : regex[%d] = '%s'", state, i, text)

    # For inclusive states, we need to add the regular expressions from the INITIAL state
    for state, stype in stateinfo.items():
        if state != 'INITIAL' and stype == 'inclusive':
            lexobj.lexstatere[state].extend(lexobj.lexstatere['INITIAL'])
            lexobj.lexstateretext[state].extend(lexobj.lexstateretext['INITIAL'])
            lexobj.lexstaterenames[state].extend(lexobj.lexstaterenames['INITIAL'])

    lexobj.lexstateinfo = stateinfo
    lexobj.lexre = lexobj.lexstatere['INITIAL']
    lexobj.lexretext = lexobj.lexstateretext['INITIAL']
    lexobj.lexreflags = reflags

    # Set up ignore variables
    lexobj.lexstateignore = linfo.ignore
    lexobj.lexignore = lexobj.lexstateignore.get('INITIAL', '')

    # Set up error functions
    lexobj.lexstateerrorf = linfo.errorf
    lexobj.lexerrorf = linfo.errorf.get('INITIAL', None)
    if not lexobj.lexerrorf:
        errorlog.warning('No t_error rule is defined')

    # Set up eof functions
    lexobj.lexstateeoff = linfo.eoff
    lexobj.lexeoff = linfo.eoff.get('INITIAL', None)

    # Check state information for ignore and error rules
    for s, stype in stateinfo.items():
        if stype == 'exclusive':
            if s not in linfo.errorf:
                errorlog.warning("No error rule is defined for exclusive state '%s'", s)
            if s not in linfo.ignore and lexobj.lexignore:
                errorlog.warning("No ignore rule is defined for exclusive state '%s'", s)
        elif stype == 'inclusive':
            if s not in linfo.errorf:
                linfo.errorf[s] = linfo.errorf.get('INITIAL', None)
            if s not in linfo.ignore:
                linfo.ignore[s] = linfo.ignore.get('INITIAL', '')

    # Create global versions of the token() and input() functions
    token = lexobj.token
    input = lexobj.input
    lexer = lexobj

    # If in optimize mode, we write the lextab
    if lextab and optimize:
        if outputdir is None:
            # If no output directory is set, the location of the output files
            # is determined according to the following rules:
            #     - If lextab specifies a package, files go into that package directory
            #     - Otherwise, files go in the same directory as the specifying module
            if isinstance(lextab, types.ModuleType):
                srcfile = lextab.__file__
            else:
                if '.' not in lextab:
                    srcfile = ldict['__file__']
                else:
                    parts = lextab.split('.')
                    pkgname = '.'.join(parts[:-1])
                    exec('import %s' % pkgname)
                    srcfile = getattr(sys.modules[pkgname], '__file__', '')
            outputdir = os.path.dirname(srcfile)
        try:
            lexobj.writetab(lextab, outputdir)
        except IOError as e:
            errorlog.warning("Couldn't write lextab module %r. %s" % (lextab, e))

    return lexobj

# -----------------------------------------------------------------------------
# runmain()
#
# This runs the lexer as a main program
# -----------------------------------------------------------------------------

def runmain(lexer=None, data=None):
    if not data:
        try:
            filename = sys.argv[1]
            f = open(filename)
            data = f.read()
            f.close()
        except IndexError:
            sys.stdout.write('Reading from standard input (type EOF to end):\n')
            data = sys.stdin.read()

    if lexer:
        _input = lexer.input
    else:
        _input = input
    _input(data)
    if lexer:
        _token = lexer.token
    else:
        _token = token

    while True:
        tok = _token()
        if not tok:
            break
        sys.stdout.write('(%s,%r,%d,%d)\n' % (tok.type, tok.value, tok.lineno, tok.lexpos))

# -----------------------------------------------------------------------------
# @TOKEN(regex)
#
# This decorator function can be used to set the regex expression on a function
# when its docstring might need to be set in an alternative way
# -----------------------------------------------------------------------------

def TOKEN(r):
    def set_regex(f):
        if hasattr(r, '__call__'):
            f.regex = _get_regex(r)
        else:
            f.regex = r
        return f
    return set_regex

# Alternative spelling of the TOKEN decorator
Token = TOKEN

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\llama\convert_to_onnx.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from itertools import chain

import onnx
import torch
from benchmark_helper import Precision, prepare_environment, setup_logger
from convert_generation import replace_mha_with_gqa
from dist_settings import barrier, get_rank, get_size, init_dist
from llama_inputs import get_merged_sample_with_past_kv_inputs, get_sample_inputs, get_sample_with_past_kv_inputs
from llama_parity import main as parity_check
from llama_torch import setup_torch_model

# to patch transformers before exporting for transformers >= 4.45
from models.torch_export_patches import bypass_export_some_errors
from models.torch_export_patches.patch_inputs import convert_dynamic_axes_into_dynamic_shapes, replace_dynamic_shapes
from onnx_model import OnnxModel
from optimizer import optimize_model
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM

from onnxruntime import __version__ as ort_version
from onnxruntime import quantization as ort_quantization

if version.parse(ort_version) < version.parse("1.22.0"):
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as MatMulNBitsQuantizer
else:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

torch_export_onnx_opset_version = 14
logger = logging.getLogger("")
init_dist()


def get_model_dynamic_axes(input_names: list[str], output_names: list[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in input_names:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "sequence_length"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes


def get_model_with_past_kv_dynamic_axes(input_names: list[str], output_names: list[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"input_ids", "position_ids"}:
            # shape is (batch_size, 1)
            dynamic_axes[name] = {0: "batch_size"}
        elif name == "attention_mask":
            # shape is (batch_size, past_sequence_length + 1)
            dynamic_axes[name] = {0: "batch_size", 1: "past_sequence_length + 1"}
        elif "past" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, 1, vocab_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + 1, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length + 1"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes


def get_merged_model_dynamic_axes(input_names: list[str], output_names: list[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"input_ids", "position_ids"}:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "attention_mask":
            # shape is (batch_size, past_sequence_length + sequence_length) = (batch_size, total_sequence_length)
            # for prompt generation, past_sequence_length = 0
            # for token generation, sequence_length = 1
            dynamic_axes[name] = {0: "batch_size", 1: "total_sequence_length"}
        elif "past" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + sequence_length, head_size) = (batch_size, num_heads, total_sequence_length, head_size)
            # for prompt generation, past_sequence_length = 0
            # for token generation, sequence_length = 1
            dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes


def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False,
    )


def run_dynamo_export(
    args: argparse.Namespace, l_config: AutoConfig, llama: AutoModelForCausalLM, rank: int = 0, world_size: int = 1
):
    from torch._dynamo import config

    config.capture_scalar_outputs = True

    # Dummy values for export
    batch_size, sequence_length, past_sequence_length = 2, 8, 3
    device = llama.device if args.model_name == "Llama-2-70b-hf" else torch.device("cpu")

    temp_name = args.model_name.lower().replace("-", "").replace("_", "")
    max_sequence_length = 16384 if "codellama" in temp_name else 4096 if "llama2" in temp_name else 2048

    # Export decoder_with_past_model.onnx
    input_ids, attn_mask, pos_ids, past_kv = get_merged_sample_with_past_kv_inputs(
        l_config,
        device,
        batch_size,
        sequence_length,
        past_sequence_length,
        max_seq_len=max_sequence_length,
        use_fp16=False,
        world_size=world_size,
    )
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, "temp.onnx")

    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = [
        "logits",
        *list(
            chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(l_config.num_hidden_layers))
        ),
    ]
    dynamic_axes = get_model_dynamic_axes(input_names, output_names)

    model_args = (input_ids, attn_mask, pos_ids, past_kv)
    model_args, model_kwargs, dynamic_shapes = convert_dynamic_axes_into_dynamic_shapes(
        llama, args=model_args, dynamic_axes=dynamic_axes, prefix_mapping={"present": "past_key_values"}
    )

    if version.Version(torch.__version__) < version.Version("2.7"):
        # This section is only needed for torch==2.6. The workaround implemented here
        # to fix bugs is not necessary with torch>=2.7.
        # - strings are not allowed with torch 2.6, so we replace them by DYNAMIC
        # - TypePromotion was fixed in torch==2.7
        from onnxscript import opset18 as op

        dynamic_shapes = replace_dynamic_shapes(
            dynamic_shapes,
            dict(batch_size=torch.export.Dim("batch_size")),
            default_value=torch.export.Dim.DYNAMIC,
        )

        # TypePromotion cannot fix a type issue after the conversion.
        # We insert an additional CastLike when the exporter
        def custom_aten_ge(self, other):
            if isinstance(other, (int, float)):
                return op.GreaterOrEqual(self, op.CastLike(other, self))
            return op.GreaterOrEqual(self, other)

        with bypass_export_some_errors(patch_transformers=True):
            # ONNX pass TypePromotion crashes for torch 2.6.
            # It can be bypassed by exporting first into an exported program.
            # We then need to apply run_decompositions() before onnx conversion starts.
            ep = torch.export.export(
                llama,
                (),
                kwargs=model_kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            ep = ep.run_decompositions()
            torch.onnx.export(
                ep,
                (),
                temp_path,
                kwargs=model_kwargs,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                verbose=args.verbose,
                optimize=True,
                custom_translation_table={torch.ops.aten.ge.Scalar: custom_aten_ge},
            )
    else:
        with bypass_export_some_errors(patch_transformers=True):
            torch.onnx.export(
                llama,
                (),
                temp_path,
                kwargs=model_kwargs,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                verbose=args.verbose,
                optimize=True,
            )

    # Check decoder_with_past_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = os.path.join(args.output, f"rank_{rank}_{args.model_name}_decoder_with_past_model_fp32.onnx")
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(onnx_model, output_path, f"rank_{rank}_{args.model_name}_decoder_with_past_model_fp32.onnx.data")
    del onnx_model
    temp_dir.cleanup()

    logger.info(f"The {args.model_name} ONNX model has been successfully created with the Dynamo exporter!")


def _prepare_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def run_torchscript_separate_export(
    args: argparse.Namespace, l_config: AutoConfig, llama: AutoModelForCausalLM, rank: int = 0, world_size: int = 1
):
    # Dummy values for export
    batch_size, sequence_length = 2, 8

    # set device used to export model
    # for llama-2-70b we will use current gpus to speed up export process
    # for other models, we will use CPU to make sure we have enough memory to do export
    device = llama.device if args.model_name == "Llama-2-70b-hf" else torch.device("cpu")

    # Export decoder_model.onnx
    decoder_inputs = get_sample_inputs(l_config, device, batch_size, sequence_length)

    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = [
        "logits",
        *list(
            chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(l_config.num_hidden_layers))
        ),
    ]
    dynamic_axes = get_model_dynamic_axes(input_names, output_names)

    # Avoid using system temp dir to avoid overflood on hard disk as 70b model is very large.
    # Use temp folder per rank to avoid race condition here.
    temp_dir = f"./temp_{rank}"
    _prepare_dir(temp_dir)
    temp_path = os.path.join(temp_dir, "temp.onnx")
    torch.onnx.export(
        llama,
        args=decoder_inputs,
        f=temp_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=torch_export_onnx_opset_version,
        do_constant_folding=True,
        verbose=args.verbose,
    )

    # Check decoder_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = os.path.join(args.output, f"rank_{rank}_{args.model_name}_decoder_model_fp32.onnx")
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(
        onnx_model,
        output_path,
        f"rank_{rank}_{args.model_name}_decoder_model_fp32.onnx.data",
    )
    del onnx_model
    shutil.rmtree(temp_dir)

    # Export decoder_with_past_model.onnx
    decoder_with_past_inputs = get_sample_with_past_kv_inputs(
        l_config,
        device,
        batch_size,
        sequence_length,
        use_fp16=False,
        world_size=world_size,
    )
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *list(
            chain.from_iterable(
                (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(l_config.num_hidden_layers)
            )
        ),
    ]
    output_names = [
        "logits",
        *list(
            chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(l_config.num_hidden_layers))
        ),
    ]
    dynamic_axes = get_model_with_past_kv_dynamic_axes(input_names, output_names)

    # Avoid using system temp dir to avoid overflood on hard disk as 70b model is very large.
    # Use temp folder per rank to avoid race condition here.
    temp_dir = f"./temp_past_{rank}"
    _prepare_dir(temp_dir)
    temp_path = os.path.join(temp_dir, "temp.onnx")
    torch.onnx.export(
        llama,
        args=decoder_with_past_inputs,
        f=temp_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=torch_export_onnx_opset_version,
        do_constant_folding=True,
        verbose=args.verbose,
    )

    # Check decoder_with_past_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = os.path.join(args.output, f"rank_{rank}_{args.model_name}_decoder_with_past_model_fp32.onnx")
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(
        onnx_model,
        output_path,
        f"rank_{rank}_{args.model_name}_decoder_with_past_model_fp32.onnx.data",
    )
    del onnx_model
    shutil.rmtree(temp_dir)

    logger.info(
        f"The {args.model_name} separate ONNX model has been successfully created with the TorchScript exporter!"
    )


def run_torchscript_merged_export(
    args: argparse.Namespace, l_config: AutoConfig, llama: AutoModelForCausalLM, rank: int = 0, world_size: int = 1
):
    # Dummy values for export
    batch_size, sequence_length, past_sequence_length = 2, 8, 0

    # set device used to export model
    # for llama-2-70b we will use current gpus to speed up export process
    # for other models, we will use CPU to make sure we have enough memory to do export
    device = llama.device if args.model_name == "Llama-2-70b-hf" else torch.device("cpu")

    temp_name = args.model_name.lower().replace("-", "").replace("_", "")
    max_sequence_length = 16384 if "codellama" in temp_name else 4096 if "llama2" in temp_name else 2048

    # Export decoder_merged_model.onnx
    decoder_merged_inputs = get_merged_sample_with_past_kv_inputs(
        l_config,
        device,
        batch_size,
        sequence_length,
        past_sequence_length,
        max_seq_len=max_sequence_length,
        use_fp16=False,
        world_size=world_size,
    )
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *list(
            chain.from_iterable(
                (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(l_config.num_hidden_layers)
            )
        ),
    ]
    output_names = [
        "logits",
        *list(
            chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(l_config.num_hidden_layers))
        ),
    ]
    dynamic_axes = get_merged_model_dynamic_axes(input_names, output_names)

    # Avoid using system temp dir to avoid overflood on hard disk as 70b model is very large.
    # Use temp folder per rank to avoid race condition here.
    temp_dir = f"./temp_{rank}"
    _prepare_dir(temp_dir)
    temp_path = os.path.join(temp_dir, "temp.onnx")

    torch.onnx.export(
        llama,
        args=decoder_merged_inputs,
        f=temp_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=torch_export_onnx_opset_version,
        do_constant_folding=True,
        verbose=args.verbose,
        dynamo=False,
    )

    # Check decoder_merged_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = os.path.join(args.output, f"rank_{rank}_{args.model_name}_decoder_merged_model_fp32.onnx")
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(
        onnx_model,
        output_path,
        f"rank_{rank}_{args.model_name}_decoder_merged_model_fp32.onnx.data",
    )
    del onnx_model
    shutil.rmtree(temp_dir)

    logger.info(f"The {args.model_name} merged ONNX model has been successfully created with the TorchScript exporter!")


# Optimize the model as FP32
def optimize_export(
    args: argparse.Namespace,
    config: AutoConfig,
    input_path: str,
    output_path: str,
    remove_model: bool = True,
    world_size: int = 1,
    window_size: int = -1,
):
    from fusion_options import FusionOptions

    optimization_options = FusionOptions("gpt2")

    model_opt = optimize_model(
        input_path,
        model_type="gpt2",
        num_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
        opt_level=0,
        optimization_options=optimization_options,
        only_onnxruntime=False,
    )
    if args.use_gqa:
        model_opt = use_group_query_attention(config, model_opt, world_size, window_size)
    model_opt.save_model_to_file(output_path, use_external_data_format=True)

    # Run symbolic shape inference on optimized model to avoid shape errors during runtime
    # Ex: Before attention fusion, RotaryEmbedding assumes a 4D input and produces a 4D output.
    # After attention fusion, RotaryEmbedding expects a 3D input and produces a 3D output.
    wheel_cmd = [sys.executable, "-m", "onnxruntime.tools.symbolic_shape_infer"]
    source_cmd = [sys.executable, "../symbolic_shape_infer.py"]
    symbolic_shape_infer_args = [
        "--input",
        output_path,
        "--output",
        output_path,
        "--auto_merge",
        "--save_as_external_data",
        "--all_tensors_to_one_file",
        "--external_data_location",
        os.path.basename(output_path) + ".data",
    ]

    file_path = os.path.dirname(__file__)
    if os.path.exists(os.path.join(file_path, "../../../tools/symbolic_shape_infer.py")):
        main_cmd = wheel_cmd
    else:
        main_cmd = source_cmd
    subprocess.run(main_cmd + symbolic_shape_infer_args)  # noqa: PLW1510

    logger.info(f"The ONNX model at {input_path} has been successfully optimized and saved at {output_path}!")
    if remove_model:
        remove_existing_model(input_path)


def convert_to_float16(args: argparse.Namespace, old_paths: list[str], rank: int = 0):
    decoder_model_fp16_path = os.path.join(args.output, f"rank_{rank}_{args.model_name}_decoder_model_fp16.onnx")
    decoder_with_past_model_fp16_path = os.path.join(
        args.output, f"rank_{rank}_{args.model_name}_decoder_with_past_model_fp16.onnx"
    )
    decoder_merged_model_fp16_path = os.path.join(
        args.output, f"rank_{rank}_{args.model_name}_decoder_merged_model_fp16.onnx"
    )
    new_paths = [decoder_model_fp16_path, decoder_with_past_model_fp16_path, decoder_merged_model_fp16_path]

    logger.info("Converting to float16...")
    for fp32_path, fp16_path in zip(old_paths, new_paths, strict=False):
        if os.path.exists(fp32_path):
            model = OnnxModel(onnx.load_model(fp32_path, load_external_data=True))
            model.convert_float_to_float16(keep_io_types=False)
            model.save_model_to_file(fp16_path, use_external_data_format=True)
            del model
            logger.info(f"The ONNX model at {fp32_path} has been converted to float16 and saved at {fp16_path}!")
            remove_existing_model(fp32_path)

    logger.info(f"The {args.model_name} ONNX model has been successfully converted to float16!")
    return new_paths


def use_group_query_attention(config: AutoConfig, model_opt: OnnxModel, world_size: int = 1, window_size: int = -1):
    # Replace MultiHeadAttention with GroupQueryAttention
    model_opt = replace_mha_with_gqa(model_opt, "attention_mask", config.num_key_value_heads, world_size, window_size)
    model_opt.prune_graph()
    model_opt.update_graph(allow_remove_graph_inputs=True)
    return model_opt


def smooth_quant(
    args: argparse.Namespace,
    decoder_model_fp32_path: str,
    decoder_with_past_model_fp32_path: str,
    decoder_model_int8_path: str,
    decoder_with_past_model_int8_path: str,
):
    from neural_compressor import PostTrainingQuantConfig, set_workspace
    from neural_compressor import quantization as intel_quantization
    from onnx.external_data_helper import load_external_data_for_model
    from quant_kv_dataloader import QuantKVDataLoader

    set_workspace(args.nc_workspace)
    quantization_config = PostTrainingQuantConfig(
        calibration_sampling_size=[args.calibration_sampling_size],
        recipes={
            "optypes_to_exclude_output_quant": ["MatMul"],
            "smooth_quant": True,
            "smooth_quant_args": {"alpha": args.smooth_quant_alpha},
        },
        op_type_dict={
            "^((?!(MatMul|Gather|Conv)).)*$": {
                "weight": {"dtype": ["fp32"]},
                "activation": {"dtype": ["fp32"]},
            }
        },
    )

    # Convert decoder_model.onnx to INT8
    decoder_model_int8 = intel_quantization.fit(
        decoder_model_fp32_path,
        quantization_config,
        calib_dataloader=QuantKVDataLoader(args),
    )
    load_external_data_for_model(
        decoder_model_int8._model,
        os.path.split(decoder_model_int8._model_path)[0],
    )
    save_onnx_model(
        decoder_model_int8._model,
        decoder_model_int8_path,
        f"{args.model_name}_decoder_model_int8.onnx.data",
    )
    del decoder_model_int8
    logger.info(
        f"The ONNX model at {decoder_model_fp32_path} has been quantized to int8 and saved at {decoder_model_int8_path}!"
    )
    remove_existing_model(decoder_model_fp32_path)

    # Convert decoder_with_past_model.onnx to INT8
    decoder_with_past_model_int8 = intel_quantization.fit(
        decoder_with_past_model_fp32_path,
        quantization_config,
        calib_dataloader=QuantKVDataLoader(args, onnx_model_path=decoder_model_fp32_path),
    )
    load_external_data_for_model(
        decoder_with_past_model_int8._model,
        os.path.split(decoder_with_past_model_int8._model_path)[0],
    )
    save_onnx_model(
        decoder_with_past_model_int8._model,
        decoder_with_past_model_int8_path,
        f"{args.model_name}_decoder_with_past_model_int8.onnx.data",
    )
    del decoder_with_past_model_int8
    logger.info(
        f"The ONNX model at {decoder_with_past_model_fp32_path} has been quantized to int8 and saved at {decoder_with_past_model_int8_path}!"
    )
    remove_existing_model(decoder_with_past_model_fp32_path)

    logger.info(f"The {args.model_name} ONNX model has been successfully quantized to int8!")

    logger.warning(f"Removing {args.nc_workspace}")
    shutil.rmtree(args.nc_workspace)


def remove_existing_model(model_path: str):
    # Remove ONNX model and its external data
    data_path = os.path.join(model_path + ".data")
    os.remove(model_path)
    os.remove(data_path)
    logger.warning(f"Removed {model_path} and {data_path}")


def remove_existing_files(output_path: str):
    for filename in os.listdir(output_path):
        filepath = os.path.join(output_path, filename)
        if ".onnx" in filename or ".onnx.data" in filename:
            os.remove(filepath)
            logger.warning(f"Removed {filepath}")


def optimize_optimum(config: AutoConfig, args: argparse.Namespace):
    tmp_file = os.path.join(args.output, args.model_name + ".tmp.onnx")
    output_file = os.path.join(args.output, args.model_name + ".onnx")
    window_size = -1 if not hasattr(config, "sliding_window") else config.sliding_window
    optimize_export(args, config, args.input, tmp_file, remove_model=False, window_size=window_size)
    logger.info(f"Model successfully optimized to {tmp_file}")
    opt_model = OnnxModel(onnx.load_model(tmp_file, load_external_data=True))
    if args.precision == Precision.FLOAT16:
        opt_model.convert_float_to_float16(keep_io_types=False)
        logger.info("Model successfully fused and quantized to FP16!")
    opt_model.save_model_to_file(output_file, use_external_data_format=True)
    logger.info(f"Output model successfully saved to {output_file}")
    logger.info(f"Removing {tmp_file}")
    remove_existing_model(tmp_file)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help="Model name in Hugging Face",
    )

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default=os.path.join("."),
        help="Directory path to PyTorch model and associated files if saved on disk, or ONNX model file location if optimize_optimum is passed.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=os.path.join(".", "llama_onnx_models"),
        help="Directory path to save exported model files in",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=[Precision.FLOAT32, Precision.FLOAT16, Precision.INT8, Precision.INT4],
        help="Precision to export model in",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        required=False,
        default="cpu",
        choices=["cpu", "cuda", "rocm"],
        help="Execution provider to verify parity with",
    )

    parser.add_argument(
        "-r",
        "--reexport",
        required=False,
        action="store_true",
        help="Re-export models and overwrite existing models in output folder",
    )
    parser.set_defaults(reexport=False)

    parser.add_argument(
        "--use_gqa",
        required=False,
        action="store_true",
        help="Use GroupQueryAttention instead of MultiHeadAttention",
    )
    parser.set_defaults(use_gqa=False)

    parser.add_argument(
        "--no_merged",
        required=False,
        action="store_true",
        help="Export models into 2 ONNX files instead of 1. Deprecated in favor of exporting into 1 ONNX file.",
    )
    parser.set_defaults(no_merged=False)

    parser.add_argument(
        "-q",
        "--quantization_method",
        default="",
        choices=["blockwise", "smooth_quant", "quantize_dynamic"],
        help="Run a specific quantization algorithm (blockwise for int4, smooth_quant for int8, quantize_dynamic for int8). Blockwise is recommended. Need to install extra packages in `requirements-quant.txt` for SmoothQuant.",
    )

    blockwise_group = parser.add_argument_group("blockwise (4-bit quantization)")

    blockwise_group.add_argument(
        "--block_size",
        required=False,
        default=32,
        type=int,
        help="Block size to quantize with. See https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py for details.",
    )

    blockwise_group.add_argument(
        "--int4_accuracy_level",
        required=False,
        type=int,
        help="Accuracy level of the 4-bit quantized MatMul computation. "
        "Refer to the MatMulNBits contrib op's 'accuracy_level' attribute for details "
        "(https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits).",
    )

    smooth_quant_group = parser.add_argument_group("smooth_quant (8-bit quantization)")

    smooth_quant_group.add_argument(
        "--smooth_quant_alpha",
        required=False,
        default=0.8,
        type=float,
        help="Strength to control migration difficulty from activation to weights. Default is 0.8 to match value \
              used in original paper for LLaMA. Paper recommends using values in [0.4, 0.6] range. \
              Link to paper: https://arxiv.org/pdf/2211.10438.pdf",
    )

    smooth_quant_group.add_argument(
        "--smooth_quant_dataset",
        required=False,
        default="NeelNanda/pile-10k",
        help="Path to dataset for calibration during quantization",
    )

    smooth_quant_group.add_argument(
        "--pad_max",
        required=False,
        default=196,
        type=int,
        help="Max padding size",
    )

    smooth_quant_group.add_argument(
        "--calibration_sampling_size",
        required=False,
        type=int,
        default=8,
        help="Calibration sampling size for quantization config",
    )

    smooth_quant_group.add_argument(
        "--nc_workspace",
        required=False,
        type=str,
        default=os.path.join(".", "nc_workspace"),
        help="Workspace to save intermediate files generated by Intel's Neural Compressor package.",
    )

    quantize_dynamic_group = parser.add_argument_group("quantize_dynamic (8-bit quantization)")

    quantize_dynamic_group.add_argument(
        "--quantize_embedding_layer",
        required=False,
        action="store_true",
        help="Quantize MatMul, GEMM, and Gather.",
    )
    quantize_dynamic_group.set_defaults(quantize_embedding_layer=False)

    quantize_dynamic_group.add_argument(
        "--quantize_per_channel",
        required=False,
        action="store_true",
        help="Quantize weights per each channel.",
    )
    quantize_dynamic_group.set_defaults(quantize_per_channel=False)

    quantize_dynamic_group.add_argument(
        "--quantize_reduce_range",
        required=False,
        action="store_true",
        help="Quantize weights with 7 bits.",
    )
    quantize_dynamic_group.set_defaults(quantize_reduce_range=False)

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logs",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "-d",
        "--use_dynamo_export",
        action="store_true",
        help="Use the new Dynamo exporter instead of the old TorchScript exporter",
    )
    parser.set_defaults(use_dynamo_export=False)

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default="./model_cache",
        help="model cache dir to override default HF cache dir to avoid overflood the /home dir",
    )

    parser.add_argument(
        "--optimize_optimum",
        action="store_true",
        help="Avoid exporting model, only apply quantizations and optimizations to existing model exported from optimum.",
    )

    parser.add_argument(
        "--small_gpu",
        action="store_true",
        help="Load the llama in GPU every time for parity_check if it's running in a machine which GPU memory < 36GB.",
    )

    parser.set_defaults(optimize_optimum=False)

    args = parser.parse_args()
    return args


def main():
    if version.parse(torch.__version__) < version.parse("2.2.0"):
        logger.error(f"Detected PyTorch version {torch.__version__}. Please upgrade and use v2.2.0 or newer.")
        return

    args = get_args()
    setup_logger(args.verbose)
    prepare_environment(args.input, args.output, args.execution_provider != "cpu")
    if args.reexport:
        remove_existing_files(args.output)
    logger.info(f"Arguments: {args}")

    world_size = get_size()
    rank = get_rank()
    args.world_size = world_size

    # Load model and config
    use_auth_token = args.input == os.path.join(".")
    setattr(args, "use_auth_token", use_auth_token)  # noqa: B010

    original_model_name = args.model_name
    setattr(args, "original_model_name", original_model_name)  # noqa: B010
    args.model_name = args.model_name.split("/")[-1]

    setattr(args, "device_name", "cpu" if args.execution_provider == "cpu" else f"cuda:{rank}")  # noqa: B010
    setattr(args, "device", torch.device(args.device_name))  # noqa: B010

    location = args.original_model_name if use_auth_token else args.input

    if args.optimize_optimum:
        config = AutoConfig.from_pretrained(args.original_model_name, cache_dir=args.cache_dir)
        optimize_optimum(config, args)
        return

    # Use CUDA for LLaMA-2-70B to speed up export and CPU for other models
    l_config, llama = setup_torch_model(
        args, location, use_auth_token, device=args.device if args.model_name == "Llama-2-70b-hf" else None
    )

    assert l_config.num_attention_heads % world_size == 0 and l_config.num_key_value_heads % world_size == 0

    barrier()
    for i in range(world_size):
        if i == rank:
            # Set model paths for FP32 model
            decoder_model_fp32_path = os.path.join(
                args.output, f"rank_{rank}_{args.model_name}_decoder_model_fp32.onnx"
            )
            decoder_with_past_model_fp32_path = os.path.join(
                args.output, f"rank_{rank}_{args.model_name}_decoder_with_past_model_fp32.onnx"
            )
            decoder_merged_model_fp32_path = os.path.join(
                args.output, f"rank_{rank}_{args.model_name}_decoder_merged_model_fp32.onnx"
            )
            old_paths = [decoder_model_fp32_path, decoder_with_past_model_fp32_path, decoder_merged_model_fp32_path]

            missing_separate_exports = (
                args.no_merged
                and not os.path.exists(decoder_model_fp32_path)
                and not os.path.exists(decoder_with_past_model_fp32_path)
            )
            missing_merged_export = not args.no_merged and not os.path.exists(decoder_merged_model_fp32_path)

            # Export to ONNX
            if missing_separate_exports or missing_merged_export:
                if args.use_dynamo_export:
                    logger.warning("Please ensure you have installed PyTorch, ONNX, and ONNX Script as follows.")
                    logger.warning("Step 1 - PyTorch nightly: https://pytorch.org/get-started/locally/")
                    logger.warning("Step 2 - ONNX weekly: https://pypi.org/project/onnx-weekly/")
                    logger.warning(
                        "Step 3 - ONNX Script from source: https://github.com/microsoft/onnxscript#installing-onnx-script"
                    )
                    logger.warning(
                        "Note: After you install ONNX weekly, omit `onnx` when running the first line for installing ONNX Script. This is because you already installed `onnx-weekly` in the previous step."
                    )
                    run_dynamo_export(args, l_config, llama)
                elif args.no_merged:
                    run_torchscript_separate_export(args, l_config, llama, rank, world_size)
                else:
                    run_torchscript_merged_export(args, l_config, llama, rank, world_size)
            del llama  # Delete LLaMA model from memory since it will be loaded again during parity check

            # Set model paths to store FP32 optimized model
            decoder_model_fp32_opt_path = os.path.join(
                args.output, f"rank_{rank}_{args.model_name}_decoder_model_fp32_opt.onnx"
            )
            decoder_with_past_model_fp32_opt_path = os.path.join(
                args.output, f"rank_{rank}_{args.model_name}_decoder_with_past_model_fp32_opt.onnx"
            )
            decoder_merged_model_fp32_opt_path = os.path.join(
                args.output, f"rank_{rank}_{args.model_name}_decoder_merged_model_fp32_opt.onnx"
            )
            new_paths = [
                decoder_model_fp32_opt_path,
                decoder_with_past_model_fp32_opt_path,
                decoder_merged_model_fp32_opt_path,
            ]

            # Run the optimizer script.
            logger.info("Optimizing models...")
            for orig_path, opt_path in zip(old_paths, new_paths, strict=False):
                if os.path.exists(orig_path):
                    optimize_export(args, l_config, input_path=orig_path, output_path=opt_path, world_size=world_size)

            # Re-assign default FP32 model paths as their optimized versions
            decoder_model_fp32_path = decoder_model_fp32_opt_path
            decoder_with_past_model_fp32_path = decoder_with_past_model_fp32_opt_path
            decoder_merged_model_fp32_path = decoder_merged_model_fp32_opt_path
            old_paths = [decoder_model_fp32_path, decoder_with_past_model_fp32_path, decoder_merged_model_fp32_path]

            logger.info(
                f"The {args.model_name} ONNX model has been successfully optimized with the ORT transformer optimizer script!"
            )

            # Change precision of exported models from FP32
            if args.precision == Precision.FLOAT16:
                new_paths = convert_to_float16(args, old_paths, rank)

            elif args.precision == Precision.INT8:
                decoder_model_int8_path = os.path.join(
                    args.output, f"rank_{rank}_{args.model_name}_decoder_model_int8.onnx"
                )
                decoder_with_past_model_int8_path = os.path.join(
                    args.output, f"rank_{rank}_{args.model_name}_decoder_with_past_model_int8.onnx"
                )
                decoder_merged_model_int8_path = os.path.join(
                    args.output, f"rank_{rank}_{args.model_name}_decoder_merged_model_int8.onnx"
                )
                new_paths = [decoder_model_int8_path, decoder_with_past_model_int8_path, decoder_merged_model_int8_path]

                if args.quantization_method == "smooth_quant":
                    if not args.no_merged:
                        logger.error("SmoothQuant must be used on separately exported models")
                    else:
                        logger.info(
                            f"Quantizing {decoder_model_fp32_path} and {decoder_with_past_model_fp32_path} to int8"
                        )
                        smooth_quant(args, old_paths[0], old_paths[1], new_paths[0], new_paths[1])

                elif args.quantization_method == "quantize_dynamic":
                    logger.warning(
                        "The `quantize_dynamic` method is deprecated in favor of `smooth_quant` instead. Precision loss may be high with `quantize_dynamic`."
                    )

                    logger.info("Quantizing to int8...")
                    for fp32_path, int8_path in zip(old_paths, new_paths, strict=False):
                        if os.path.exists(fp32_path):
                            ort_quantization.quantize_dynamic(
                                fp32_path,
                                int8_path,
                                op_types_to_quantize=(
                                    ["MatMul", "Gemm", "Gather"]
                                    if args.quantize_embedding_layer
                                    else ["MatMul", "Gemm"]
                                ),
                                per_channel=args.quantize_per_channel,
                                reduce_range=args.quantize_reduce_range,
                                use_external_data_format=True,
                                extra_options={"MatMulConstBOnly": True},
                            )
                            logger.info(
                                f"The ONNX model at {fp32_path} has been quantized to int8 and saved at {int8_path}!"
                            )
                            remove_existing_model(decoder_model_fp32_path)

                    logger.info(f"The {args.model_name} ONNX model has been successfully quantized to int8!")

                else:
                    raise Exception(f"Could not recognize {args.quantization_method} as a quantization method")

            elif args.precision == Precision.INT4:
                if args.execution_provider != "cpu":
                    old_paths = convert_to_float16(args, old_paths, rank)

                decoder_model_int4_path = os.path.join(
                    args.output, f"rank_{rank}_{args.model_name}_decoder_model_int4.onnx"
                )
                decoder_with_past_model_int4_path = os.path.join(
                    args.output, f"rank_{rank}_{args.model_name}_decoder_with_past_model_int4.onnx"
                )
                decoder_merged_model_int4_path = os.path.join(
                    args.output, f"rank_{rank}_{args.model_name}_decoder_merged_model_int4.onnx"
                )
                new_paths = [decoder_model_int4_path, decoder_with_past_model_int4_path, decoder_merged_model_int4_path]

                for fp_path, int4_path in zip(old_paths, new_paths, strict=False):
                    if os.path.exists(fp_path):
                        model = onnx.load_model(fp_path, load_external_data=True)
                        quant = MatMulNBitsQuantizer(
                            model=model,
                            block_size=args.block_size,
                            is_symmetric=True,
                            accuracy_level=args.int4_accuracy_level,
                            nodes_to_exclude=[],
                        )
                        quant.process()
                        quant.model.save_model_to_file(int4_path, use_external_data_format=True)
                        del model
                        del quant
                        logger.info(f"The ONNX model at {fp_path} has been quantized to int4 and saved at {int4_path}!")
                        remove_existing_model(fp_path)
        barrier()

    logger.info("Verifying parity on all ONNX models created")

    # Use FP32 precision for FP32, INT8, INT4 CPU models, use FP16 precision for FP16 and INT4 GPU models
    args.precision = (
        "fp32"
        if args.precision in {Precision.INT8, Precision.FLOAT32}
        or (args.precision == Precision.INT4 and args.execution_provider == "cpu")
        else "fp16"
    )

    # Verify parity on all saved ONNX models
    for filename in os.listdir(args.output):
        if (
            ".data" in filename
            or ".onnx" not in filename
            or args.precision not in filename
            or f"rank_{rank}" not in filename
        ):
            continue

        parity_cmd = [
            "-m",
            original_model_name,
            "-o",
            os.path.join(args.output, filename),
            "-ep",
            args.execution_provider,
            "--precision",
            args.precision,
            "--cache_dir",
            args.cache_dir,
            "--torch_model_directory",
            args.input,
        ]
        if args.small_gpu:
            parity_cmd.append("--small_gpu")
        if "with_past" in filename:
            parity_cmd.append("--use_past_kv")
        if "merged" in filename:
            parity_cmd.append("--merged")

        try:
            logger.info(f"check parity with cmd: {parity_cmd}")
            parity_check(parity_cmd)
        except Exception as e:
            logger.exception(f"An error occurred while verifying parity: {e}")
            sys.exit(-1)


if __name__ == "__main__":
    main()