
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\beam.py ===
"""
This module can be used to solve 2D beam bending problems with
singularity functions in mechanics.
"""

from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
import warnings


__doctest_requires__ = {
    ('Beam.draw',
     'Beam.plot_bending_moment',
     'Beam.plot_deflection',
     'Beam.plot_ild_moment',
     'Beam.plot_ild_shear',
     'Beam.plot_shear_force',
     'Beam.plot_shear_stress',
     'Beam.plot_slope'): ['matplotlib'],
}


numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})


class Beam:
    """
    A Beam is a structural element that is capable of withstanding load
    primarily by resisting against bending. Beams are characterized by
    their cross sectional profile(Second moment of area), their length
    and their material.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention. However, the
       chosen sign convention must respect the rule that, on the positive
       side of beam's axis (in respect to current section), a loading force
       giving positive shear yields a negative moment, as below (the
       curved arrow shows the positive moment and rotation):

    .. image:: allowed-sign-conventions.png

    Examples
    ========
    There is a beam of length 4 meters. A constant distributed load of 6 N/m
    is applied from half of the beam till the end. There are two simple supports
    below the beam, one at the starting point and another at the ending point
    of the beam. The deflection of the beam at the end is restricted.

    Using the sign convention of downwards forces being positive.

    >>> from sympy.physics.continuum_mechanics.beam import Beam
    >>> from sympy import symbols, Piecewise
    >>> E, I = symbols('E, I')
    >>> R1, R2 = symbols('R1, R2')
    >>> b = Beam(4, E, I)
    >>> b.apply_load(R1, 0, -1)
    >>> b.apply_load(6, 2, 0)
    >>> b.apply_load(R2, 4, -1)
    >>> b.bc_deflection = [(0, 0), (4, 0)]
    >>> b.boundary_conditions
    {'bending_moment': [], 'deflection': [(0, 0), (4, 0)], 'shear_force': [], 'slope': []}
    >>> b.load
    R1*SingularityFunction(x, 0, -1) + R2*SingularityFunction(x, 4, -1) + 6*SingularityFunction(x, 2, 0)
    >>> b.solve_for_reaction_loads(R1, R2)
    >>> b.load
    -3*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 2, 0) - 9*SingularityFunction(x, 4, -1)
    >>> b.shear_force()
    3*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 2, 1) + 9*SingularityFunction(x, 4, 0)
    >>> b.bending_moment()
    3*SingularityFunction(x, 0, 1) - 3*SingularityFunction(x, 2, 2) + 9*SingularityFunction(x, 4, 1)
    >>> b.slope()
    (-3*SingularityFunction(x, 0, 2)/2 + SingularityFunction(x, 2, 3) - 9*SingularityFunction(x, 4, 2)/2 + 7)/(E*I)
    >>> b.deflection()
    (7*x - SingularityFunction(x, 0, 3)/2 + SingularityFunction(x, 2, 4)/4 - 3*SingularityFunction(x, 4, 3)/2)/(E*I)
    >>> b.deflection().rewrite(Piecewise)
    (7*x - Piecewise((x**3, x >= 0), (0, True))/2
         - 3*Piecewise(((x - 4)**3, x >= 4), (0, True))/2
         + Piecewise(((x - 2)**4, x >= 2), (0, True))/4)/(E*I)

    Calculate the support reactions for a fully symbolic beam of length L.
    There are two simple supports below the beam, one at the starting point
    and another at the ending point of the beam. The deflection of the beam
    at the end is restricted. The beam is loaded with:

    * a downward point load P1 applied at L/4
    * an upward point load P2 applied at L/8
    * a counterclockwise moment M1 applied at L/2
    * a clockwise moment M2 applied at 3*L/4
    * a distributed constant load q1, applied downward, starting from L/2
      up to 3*L/4
    * a distributed constant load q2, applied upward, starting from 3*L/4
      up to L

    No assumptions are needed for symbolic loads. However, defining a positive
    length will help the algorithm to compute the solution.

    >>> E, I = symbols('E, I')
    >>> L = symbols("L", positive=True)
    >>> P1, P2, M1, M2, q1, q2 = symbols("P1, P2, M1, M2, q1, q2")
    >>> R1, R2 = symbols('R1, R2')
    >>> b = Beam(L, E, I)
    >>> b.apply_load(R1, 0, -1)
    >>> b.apply_load(R2, L, -1)
    >>> b.apply_load(P1, L/4, -1)
    >>> b.apply_load(-P2, L/8, -1)
    >>> b.apply_load(M1, L/2, -2)
    >>> b.apply_load(-M2, 3*L/4, -2)
    >>> b.apply_load(q1, L/2, 0, 3*L/4)
    >>> b.apply_load(-q2, 3*L/4, 0, L)
    >>> b.bc_deflection = [(0, 0), (L, 0)]
    >>> b.solve_for_reaction_loads(R1, R2)
    >>> print(b.reaction_loads[R1])
    (-3*L**2*q1 + L**2*q2 - 24*L*P1 + 28*L*P2 - 32*M1 + 32*M2)/(32*L)
    >>> print(b.reaction_loads[R2])
    (-5*L**2*q1 + 7*L**2*q2 - 8*L*P1 + 4*L*P2 + 32*M1 - 32*M2)/(32*L)
    """

    def __init__(self, length, elastic_modulus, second_moment, area=Symbol('A'), variable=Symbol('x'), base_char='C', ild_variable=Symbol('a')):
        """Initializes the class.

        Parameters
        ==========

        length : Sympifyable
            A Symbol or value representing the Beam's length.

        elastic_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of Elasticity.
            It is a measure of the stiffness of the Beam material. It can
            also be a continuous function of position along the beam.

        second_moment : Sympifyable or Geometry object
            Describes the cross-section of the beam via a SymPy expression
            representing the Beam's second moment of area. It is a geometrical
            property of an area which reflects how its points are distributed
            with respect to its neutral axis. It can also be a continuous
            function of position along the beam. Alternatively ``second_moment``
            can be a shape object such as a ``Polygon`` from the geometry module
            representing the shape of the cross-section of the beam. In such cases,
            it is assumed that the x-axis of the shape object is aligned with the
            bending axis of the beam. The second moment of area will be computed
            from the shape object internally.

        area : Symbol/float
            Represents the cross-section area of beam

        variable : Symbol, optional
            A Symbol object that will be used as the variable along the beam
            while representing the load, shear, moment, slope and deflection
            curve. By default, it is set to ``Symbol('x')``.

        base_char : String, optional
            A String that will be used as base character to generate sequential
            symbols for integration constants in cases where boundary conditions
            are not sufficient to solve them.

        ild_variable : Symbol, optional
            A Symbol object that will be used as the variable specifying the
            location of the moving load in ILD calculations. By default, it
            is set to ``Symbol('a')``.
        """
        self.length = length
        self.elastic_modulus = elastic_modulus
        if isinstance(second_moment, GeometryEntity):
            self.cross_section = second_moment
        else:
            self.cross_section = None
            self.second_moment = second_moment
        self.variable = variable
        self.ild_variable = ild_variable
        self._base_char = base_char
        self._boundary_conditions = {'deflection': [], 'slope': [], 'bending_moment': [], 'shear_force': []}
        self._load = 0
        self.area = area
        self._applied_supports = []
        self._applied_rotation_hinges = []
        self._applied_sliding_hinges = []
        self._rotation_hinge_symbols = []
        self._sliding_hinge_symbols = []
        self._support_as_loads = []
        self._applied_loads = []
        self._reaction_loads = {}
        self._ild_reactions = {}
        self._ild_shear = 0
        self._ild_moment = 0
        # _original_load is a copy of _load equations with unsubstituted reaction
        # forces. It is used for calculating reaction forces in case of I.L.D.
        self._original_load = 0
        self._joined_beam = False

    def __str__(self):
        shape_description = self._cross_section if self._cross_section else self._second_moment
        str_sol = 'Beam({}, {}, {})'.format(sstr(self._length), sstr(self._elastic_modulus), sstr(shape_description))
        return str_sol

    @property
    def reaction_loads(self):
        """ Returns the reaction forces in a dictionary."""
        return self._reaction_loads

    @property
    def rotation_jumps(self):
        """
        Returns the value for the rotation jumps in rotation hinges in a dictionary.
        The rotation jump is the rotation (in radian) in a rotation hinge. This can
        be seen as a jump in the slope plot.
        """
        return self._rotation_jumps

    @property
    def deflection_jumps(self):
        """
        Returns the deflection jumps in sliding hinges in a dictionary.
        The deflection jump is the deflection (in meters) in a sliding hinge.
        This can be seen as a jump in the deflection plot.
        """
        return self._deflection_jumps

    @property
    def ild_shear(self):
        """ Returns the I.L.D. shear equation."""
        return self._ild_shear

    @property
    def ild_reactions(self):
        """ Returns the I.L.D. reaction forces in a dictionary."""
        return self._ild_reactions

    @property
    def ild_rotation_jumps(self):
        """
        Returns the I.L.D. rotation jumps in rotation hinges in a dictionary.
        The rotation jump is the rotation (in radian) in a rotation hinge. This can
        be seen as a jump in the slope plot.
        """
        return self._ild_rotations_jumps

    @property
    def ild_deflection_jumps(self):
        """
        Returns the I.L.D. deflection jumps in sliding hinges in a dictionary.
        The deflection jump is the deflection (in meters) in a sliding hinge.
        This can be seen as a jump in the deflection plot.
        """
        return self._ild_deflection_jumps

    @property
    def ild_moment(self):
        """ Returns the I.L.D. moment equation."""
        return self._ild_moment

    @property
    def length(self):
        """Length of the Beam."""
        return self._length

    @length.setter
    def length(self, l):
        self._length = sympify(l)

    @property
    def area(self):
        """Cross-sectional area of the Beam. """
        return self._area

    @area.setter
    def area(self, a):
        self._area = sympify(a)

    @property
    def variable(self):
        """
        A symbol that can be used as a variable along the length of the beam
        while representing load distribution, shear force curve, bending
        moment, slope curve and the deflection curve. By default, it is set
        to ``Symbol('x')``, but this property is mutable.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I, A = symbols('E, I, A')
        >>> x, y, z = symbols('x, y, z')
        >>> b = Beam(4, E, I)
        >>> b.variable
        x
        >>> b.variable = y
        >>> b.variable
        y
        >>> b = Beam(4, E, I, A, z)
        >>> b.variable
        z
        """
        return self._variable

    @variable.setter
    def variable(self, v):
        if isinstance(v, Symbol):
            self._variable = v
        else:
            raise TypeError("""The variable should be a Symbol object.""")

    @property
    def elastic_modulus(self):
        """Young's Modulus of the Beam. """
        return self._elastic_modulus

    @elastic_modulus.setter
    def elastic_modulus(self, e):
        self._elastic_modulus = sympify(e)

    @property
    def second_moment(self):
        """Second moment of area of the Beam. """
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        self._cross_section = None
        if isinstance(i, GeometryEntity):
            raise ValueError("To update cross-section geometry use `cross_section` attribute")
        else:
            self._second_moment = sympify(i)

    @property
    def cross_section(self):
        """Cross-section of the beam"""
        return self._cross_section

    @cross_section.setter
    def cross_section(self, s):
        if s:
            self._second_moment = s.second_moment_of_area()[0]
        self._cross_section = s

    @property
    def boundary_conditions(self):
        """
        Returns a dictionary of boundary conditions applied on the beam.
        The dictionary has three keywords namely moment, slope and deflection.
        The value of each keyword is a list of tuple, where each tuple
        contains location and value of a boundary condition in the format
        (location, value).

        Examples
        ========
        There is a beam of length 4 meters. The bending moment at 0 should be 4
        and at 4 it should be 0. The slope of the beam should be 1 at 0. The
        deflection should be 2 at 0.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.bc_deflection = [(0, 2)]
        >>> b.bc_slope = [(0, 1)]
        >>> b.boundary_conditions
        {'bending_moment': [], 'deflection': [(0, 2)], 'shear_force': [], 'slope': [(0, 1)]}

        Here the deflection of the beam should be ``2`` at ``0``.
        Similarly, the slope of the beam should be ``1`` at ``0``.
        """
        return self._boundary_conditions

    @property
    def bc_shear_force(self):
        return self._boundary_conditions['shear_force']

    @bc_shear_force.setter
    def bc_shear_force(self, sf_bcs):
        self._boundary_conditions['shear_force'] = sf_bcs

    @property
    def bc_bending_moment(self):
        return self._boundary_conditions['bending_moment']

    @bc_bending_moment.setter
    def bc_bending_moment(self, bm_bcs):
        self._boundary_conditions['bending_moment'] = bm_bcs

    @property
    def bc_slope(self):
        return self._boundary_conditions['slope']

    @bc_slope.setter
    def bc_slope(self, s_bcs):
        self._boundary_conditions['slope'] = s_bcs

    @property
    def bc_deflection(self):
        return self._boundary_conditions['deflection']

    @bc_deflection.setter
    def bc_deflection(self, d_bcs):
        self._boundary_conditions['deflection'] = d_bcs

    def join(self, beam, via="fixed"):
        """
        This method joins two beams to make a new composite beam system.
        Passed Beam class instance is attached to the right end of calling
        object. This method can be used to form beams having Discontinuous
        values of Elastic modulus or Second moment.

        Parameters
        ==========
        beam : Beam class object
            The Beam object which would be connected to the right of calling
            object.
        via : String
            States the way two Beam object would get connected
            - For axially fixed Beams, via="fixed"
            - For Beams connected via rotation hinge, via="hinge"

        Examples
        ========
        There is a cantilever beam of length 4 meters. For first 2 meters
        its moment of inertia is `1.5*I` and `I` for the other end.
        A pointload of magnitude 4 N is applied from the top at its free end.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b1 = Beam(2, E, 1.5*I)
        >>> b2 = Beam(2, E, I)
        >>> b = b1.join(b2, "fixed")
        >>> b.apply_load(20, 4, -1)
        >>> b.apply_load(R1, 0, -1)
        >>> b.apply_load(R2, 0, -2)
        >>> b.bc_slope = [(0, 0)]
        >>> b.bc_deflection = [(0, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.load
        80*SingularityFunction(x, 0, -2) - 20*SingularityFunction(x, 0, -1) + 20*SingularityFunction(x, 4, -1)
        >>> b.slope()
        (-((-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))/I + 120/I)/E + 80.0/(E*I))*SingularityFunction(x, 2, 0)
        - 0.666666666666667*(-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))*SingularityFunction(x, 0, 0)/(E*I)
        + 0.666666666666667*(-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))*SingularityFunction(x, 2, 0)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        new_length = self.length + beam.length
        if self.elastic_modulus != beam.elastic_modulus:
            raise NotImplementedError('Joining beams with different Elastic modulus is not implemented.')

        if self.second_moment != beam.second_moment:
            new_second_moment = Piecewise((self.second_moment, x<=self.length),
                                    (beam.second_moment, x<=new_length))
        else:
            new_second_moment = self.second_moment

        if via == "fixed":
            new_beam = Beam(new_length, E, new_second_moment, x)
            new_beam._joined_beam = True
            return new_beam

        if via == "hinge":
            new_beam = Beam(new_length, E, new_second_moment, x)
            new_beam._joined_beam = True
            new_beam.apply_rotation_hinge(self.length)
            return new_beam

    def apply_support(self, loc, type="fixed"):
        """
        This method applies support to a particular beam object and returns
        the symbol of the unknown reaction load(s).

        Parameters
        ==========
        loc : Sympifyable
            Location of point at which support is applied.
        type : String
            Determines type of Beam support applied. To apply support structure
            with
            - zero degree of freedom, type = "fixed"
            - one degree of freedom, type = "pin"
            - two degrees of freedom, type = "roller"

        Returns
        =======
        Symbol or tuple of Symbol
            The unknown reaction load as a symbol.
            - Symbol(reaction_force) if type = "pin" or "roller"
            - Symbol(reaction_force), Symbol(reaction_moment) if type = "fixed"

        Examples
        ========
        There is a beam of length 20 meters. A moment of magnitude 100 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at a distance of 10 meters.
        There is one fixed support at the start of the beam and a roller at the end.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(20, E, I)
        >>> p0, m0 = b.apply_support(0, 'fixed')
        >>> p1 = b.apply_support(20, 'roller')
        >>> b.apply_load(-8, 10, -1)
        >>> b.apply_load(100, 20, -2)
        >>> b.solve_for_reaction_loads(p0, m0, p1)
        >>> b.reaction_loads
        {M_0: 20, R_0: -2, R_20: 10}
        >>> b.reaction_loads[p0]
        -2
        >>> b.load
        20*SingularityFunction(x, 0, -2) - 2*SingularityFunction(x, 0, -1)
        - 8*SingularityFunction(x, 10, -1) + 100*SingularityFunction(x, 20, -2)
        + 10*SingularityFunction(x, 20, -1)
        """
        loc = sympify(loc)

        self._applied_supports.append((loc, type))
        if type in ("pin", "roller"):
            reaction_load = Symbol('R_'+str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.bc_deflection.append((loc, 0))
        else:
            reaction_load = Symbol('R_'+str(loc))
            reaction_moment = Symbol('M_'+str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.apply_load(reaction_moment, loc, -2)
            self.bc_deflection.append((loc, 0))
            self.bc_slope.append((loc, 0))
            self._support_as_loads.append((reaction_moment, loc, -2, None))

        self._support_as_loads.append((reaction_load, loc, -1, None))

        if type in ("pin", "roller"):
            return reaction_load
        else:
            return reaction_load, reaction_moment

    def _get_I(self, loc):
        """
        Helper function that returns the Second moment (I) at a location in the beam.
        """
        I = self.second_moment
        if not isinstance(I, Piecewise):
            return I
        else:
            for i in range(len(I.args)):
                if loc <= I.args[i][1].args[1]:
                    return I.args[i][0]

    def apply_rotation_hinge(self, loc):
        """
        This method applies a rotation hinge at a single location on the beam.

        Parameters
        ----------
        loc : Sympifyable
            Location of point at which hinge is applied.

        Returns
        =======
        Symbol
            The unknown rotation jump multiplied by the elastic modulus and second moment as a symbol.

        Examples
        ========
        There is a beam of length 15 meters. Pin supports are placed at distances
        of 0 and 10 meters. There is a fixed support at the end. There are two rotation hinges
        in the structure, one at 5 meters and one at 10 meters. A pointload of magnitude
        10 kN is applied on the hinge at 5 meters. A distributed load of 5 kN works on
        the structure from 10 meters to the end.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import Symbol
        >>> E = Symbol('E')
        >>> I = Symbol('I')
        >>> b = Beam(15, E, I)
        >>> r0 = b.apply_support(0, type='pin')
        >>> r10 = b.apply_support(10, type='pin')
        >>> r15, m15 = b.apply_support(15, type='fixed')
        >>> p5 = b.apply_rotation_hinge(5)
        >>> p12 = b.apply_rotation_hinge(12)
        >>> b.apply_load(-10, 5, -1)
        >>> b.apply_load(-5, 10, 0, 15)
        >>> b.solve_for_reaction_loads(r0, r10, r15, m15)
        >>> b.reaction_loads
        {M_15: -75/2, R_0: 0, R_10: 40, R_15: -5}
        >>> b.rotation_jumps
        {P_12: -1875/(16*E*I), P_5: 9625/(24*E*I)}
        >>> b.rotation_jumps[p12]
        -1875/(16*E*I)
        >>> b.bending_moment()
        -9625*SingularityFunction(x, 5, -1)/24 + 10*SingularityFunction(x, 5, 1)
        - 40*SingularityFunction(x, 10, 1) + 5*SingularityFunction(x, 10, 2)/2
        + 1875*SingularityFunction(x, 12, -1)/16 + 75*SingularityFunction(x, 15, 0)/2
        + 5*SingularityFunction(x, 15, 1) - 5*SingularityFunction(x, 15, 2)/2
        """
        loc = sympify(loc)
        E = self.elastic_modulus
        I = self._get_I(loc)

        rotation_jump = Symbol('P_'+str(loc))
        self._applied_rotation_hinges.append(loc)
        self._rotation_hinge_symbols.append(rotation_jump)
        self.apply_load(E * I * rotation_jump, loc, -3)
        self.bc_bending_moment.append((loc, 0))
        return rotation_jump

    def apply_sliding_hinge(self, loc):
        """
        This method applies a sliding hinge at a single location on the beam.

        Parameters
        ----------
        loc : Sympifyable
            Location of point at which hinge is applied.

        Returns
        =======
        Symbol
            The unknown deflection jump multiplied by the elastic modulus and second moment as a symbol.

        Examples
        ========
        There is a beam of length 13 meters. A fixed support is placed at the beginning.
        There is a pin support at the end. There is a sliding hinge at a location of 8 meters.
        A pointload of magnitude 10 kN is applied on the hinge at 5 meters.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> b = Beam(13, 20, 20)
        >>> r0, m0 = b.apply_support(0, type="fixed")
        >>> s8 = b.apply_sliding_hinge(8)
        >>> r13 = b.apply_support(13, type="pin")
        >>> b.apply_load(-10, 5, -1)
        >>> b.solve_for_reaction_loads(r0, m0, r13)
        >>> b.reaction_loads
        {M_0: -50, R_0: 10, R_13: 0}
        >>> b.deflection_jumps
        {W_8: 85/24}
        >>> b.deflection_jumps[s8]
        85/24
        >>> b.bending_moment()
        50*SingularityFunction(x, 0, 0) - 10*SingularityFunction(x, 0, 1)
        + 10*SingularityFunction(x, 5, 1) - 4250*SingularityFunction(x, 8, -2)/3
        >>> b.deflection()
        -SingularityFunction(x, 0, 2)/16 + SingularityFunction(x, 0, 3)/240
        - SingularityFunction(x, 5, 3)/240 + 85*SingularityFunction(x, 8, 0)/24
        """
        loc = sympify(loc)
        E = self.elastic_modulus
        I = self._get_I(loc)

        deflection_jump = Symbol('W_' + str(loc))
        self._applied_sliding_hinges.append(loc)
        self._sliding_hinge_symbols.append(deflection_jump)
        self.apply_load(E * I * deflection_jump, loc, -4)
        self.bc_shear_force.append((loc, 0))
        return deflection_jump

    def apply_load(self, value, start, order, end=None):
        """
        This method adds up the loads given to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The value inserted should have the units [Force/(Distance**(n+1)]
            where n is the order of applied load.
            Units for applied loads:

               - For moments, unit = kN*m
               - For point loads, unit = kN
               - For constant distributed load, unit = kN/m
               - For ramp loads, unit = kN/m/m
               - For parabolic ramp loads, unit = kN/m/m/m
               - ... so on.

        start : Sympifyable
            The starting point of the applied load. For point moments and
            point forces this is the location of application.
        order : Integer
            The order of the applied load.

               - For moments, order = -2
               - For point loads, order =-1
               - For constant distributed load, order = 0
               - For ramp loads, order = 1
               - For parabolic ramp loads, order = 2
               - ... so on.

        end : Sympifyable, optional
            An optional argument that can be used if the load has an end point
            within the length of the beam.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A point load of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 2 meters to 3 meters
        away from the starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 2, 2, end=3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)

        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        self._applied_loads.append((value, start, order, end))
        self._load += value*SingularityFunction(x, start, order)
        self._original_load += value*SingularityFunction(x, start, order)

        if end:
            # load has an end point within the length of the beam.
            self._handle_end(x, value, start, order, end, type="apply")

    def remove_load(self, value, start, order, end=None):
        """
        This method removes a particular load present on the beam object.
        Returns a ValueError if the load passed as an argument is not
        present on the beam.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        start : Sympifyable
            The starting point of the applied load. For point moments and
            point forces this is the location of application.
        order : Integer
            The order of the applied load.
            - For moments, order= -2
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        end : Sympifyable, optional
            An optional argument that can be used if the load has an end point
            within the length of the beam.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A pointload of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 2 meters to 3 meters
        away from the starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 2, 2, end=3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)
        >>> b.remove_load(-2, 2, 2, end = 3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1)
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if (value, start, order, end) in self._applied_loads:
            self._load -= value*SingularityFunction(x, start, order)
            self._original_load -= value*SingularityFunction(x, start, order)
            self._applied_loads.remove((value, start, order, end))
        else:
            msg = "No such load distribution exists on the beam object."
            raise ValueError(msg)

        if end:
            # load has an end point within the length of the beam.
            self._handle_end(x, value, start, order, end, type="remove")

    def _handle_end(self, x, value, start, order, end, type):
        """
        This functions handles the optional `end` value in the
        `apply_load` and `remove_load` functions. When the value
        of end is not NULL, this function will be executed.
        """
        if order.is_negative:
            msg = ("If 'end' is provided the 'order' of the load cannot "
                    "be negative, i.e. 'end' is only valid for distributed "
                    "loads.")
            raise ValueError(msg)
        # NOTE : A Taylor series can be used to define the summation of
        # singularity functions that subtract from the load past the end
        # point such that it evaluates to zero past 'end'.
        f = value*x**order

        if type == "apply":
            # iterating for "apply_load" method
            for i in range(0, order + 1):
                self._load -= (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
                self._original_load -= (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
        elif type == "remove":
            # iterating for "remove_load" method
            for i in range(0, order + 1):
                self._load += (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
                self._original_load += (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))


    @property
    def load(self):
        """
        Returns a Singularity Function expression which represents
        the load distribution curve of the Beam object.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A point load of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 3 meters away from the
        starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 3, 2)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 3, 2)
        """
        return self._load

    @property
    def applied_loads(self):
        """
        Returns a list of all loads applied on the beam object.
        Each load in the list is a tuple of form (value, start, order, end).

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A pointload of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point. Another pointload of magnitude 5 N
        is applied at same position.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(5, 2, -1)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 9*SingularityFunction(x, 2, -1)
        >>> b.applied_loads
        [(-3, 0, -2, None), (4, 2, -1, None), (5, 2, -1, None)]
        """
        return self._applied_loads

    def solve_for_reaction_loads(self, *reactions):
        """
        Solves for the reaction forces.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)  # Reaction force at x = 10
        >>> b.apply_load(R2, 30, -1)  # Reaction force at x = 30
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.load
        R1*SingularityFunction(x, 10, -1) + R2*SingularityFunction(x, 30, -1)
            - 8*SingularityFunction(x, 0, -1) + 120*SingularityFunction(x, 30, -2)
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.reaction_loads
        {R1: 6, R2: 2}
        >>> b.load
        -8*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 10, -1)
            + 120*SingularityFunction(x, 30, -2) + 2*SingularityFunction(x, 30, -1)
        """

        x = self.variable
        l = self.length
        C3 = Symbol('C3')
        C4 = Symbol('C4')
        rotation_jumps = tuple(self._rotation_hinge_symbols)
        deflection_jumps = tuple(self._sliding_hinge_symbols)

        shear_curve = limit(self.shear_force(), x, l)
        moment_curve = limit(self.bending_moment(), x, l)

        shear_force_eqs = []
        bending_moment_eqs = []
        slope_eqs = []
        deflection_eqs = []

        for position, value in self._boundary_conditions['shear_force']:
            eqs = self.shear_force().subs(x, position) - value
            new_eqs = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            shear_force_eqs.append(new_eqs)

        for position, value in self._boundary_conditions['bending_moment']:
            eqs = self.bending_moment().subs(x, position) - value
            new_eqs = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            bending_moment_eqs.append(new_eqs)

        slope_curve = integrate(self.bending_moment(), x) + C3
        for position, value in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            slope_eqs.append(eqs)

        deflection_curve = integrate(slope_curve, x) + C4
        for position, value in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            deflection_eqs.append(eqs)

        solution = list((linsolve([shear_curve, moment_curve] + shear_force_eqs + bending_moment_eqs + slope_eqs
                            + deflection_eqs, (C3, C4) + reactions + rotation_jumps + deflection_jumps).args)[0])
        reaction_index = 2+len(reactions)
        rotation_index = reaction_index + len(rotation_jumps)
        reaction_solution = solution[2:reaction_index]
        rotation_solution = solution[reaction_index:rotation_index]
        deflection_solution = solution[rotation_index:]

        self._reaction_loads = dict(zip(reactions, reaction_solution))
        self._rotation_jumps = dict(zip(rotation_jumps, rotation_solution))
        self._deflection_jumps = dict(zip(deflection_jumps, deflection_solution))
        self._load = self._load.subs(self._reaction_loads)
        self._load = self._load.subs(self._rotation_jumps)
        self._load = self._load.subs(self._deflection_jumps)

    def shear_force(self):
        """
        Returns a Singularity Function expression which represents
        the shear force curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.shear_force()
        8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) - 120*SingularityFunction(x, 30, -1) - 2*SingularityFunction(x, 30, 0)
        """
        x = self.variable
        return -integrate(self.load, x)

    def max_shear_force(self):
        """Returns maximum Shear force and its coordinate
        in the Beam object."""
        shear_curve = self.shear_force()
        x = self.variable

        terms = shear_curve.args
        singularity = []        # Points at which shear function changes
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]    # SingularityFunction in the term
            singularity.append(term.args[1])
        singularity = list(set(singularity))
        singularity.sort()

        intervals = []    # List of Intervals with discrete value of shear force
        shear_values = []   # List of values of shear force in each interval
        for i, s in enumerate(singularity):
            if s == 0:
                continue
            try:
                shear_slope = Piecewise((float("nan"), x<=singularity[i-1]),(self._load.rewrite(Piecewise), x<s), (float("nan"), True))
                points = solve(shear_slope, x)
                val = []
                for point in points:
                    val.append(abs(shear_curve.subs(x, point)))
                points.extend([singularity[i-1], s])
                val += [abs(limit(shear_curve, x, singularity[i-1], '+')), abs(limit(shear_curve, x, s, '-'))]
                max_shear = max(val)
                shear_values.append(max_shear)
                intervals.append(points[val.index(max_shear)])
            # If shear force in a particular Interval has zero or constant
            # slope, then above block gives NotImplementedError as
            # solve can't represent Interval solutions.
            except NotImplementedError:
                initial_shear = limit(shear_curve, x, singularity[i-1], '+')
                final_shear = limit(shear_curve, x, s, '-')
                # If shear_curve has a constant slope(it is a line).
                if shear_curve.subs(x, (singularity[i-1] + s)/2) == (initial_shear + final_shear)/2 and initial_shear != final_shear:
                    shear_values.extend([initial_shear, final_shear])
                    intervals.extend([singularity[i-1], s])
                else:    # shear_curve has same value in whole Interval
                    shear_values.append(final_shear)
                    intervals.append(Interval(singularity[i-1], s))

        shear_values = list(map(abs, shear_values))
        maximum_shear = max(shear_values)
        point = intervals[shear_values.index(maximum_shear)]
        return (point, maximum_shear)

    def bending_moment(self):
        """
        Returns a Singularity Function expression which represents
        the bending moment curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.bending_moment()
        8*SingularityFunction(x, 0, 1) - 6*SingularityFunction(x, 10, 1) - 120*SingularityFunction(x, 30, 0) - 2*SingularityFunction(x, 30, 1)
        """
        x = self.variable
        return integrate(self.shear_force(), x)

    def max_bmoment(self):
        """Returns maximum Shear force and its coordinate
        in the Beam object."""
        bending_curve = self.bending_moment()
        x = self.variable

        terms = bending_curve.args
        singularity = []        # Points at which bending moment changes
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]    # SingularityFunction in the term
            singularity.append(term.args[1])
        singularity = list(set(singularity))
        singularity.sort()

        intervals = []    # List of Intervals with discrete value of bending moment
        moment_values = []   # List of values of bending moment in each interval
        for i, s in enumerate(singularity):
            if s == 0:
                continue
            try:
                moment_slope = Piecewise(
                    (float("nan"), x <= singularity[i - 1]),
                    (self.shear_force().rewrite(Piecewise), x < s),
                    (float("nan"), True))
                points = solve(moment_slope, x)
                val = []
                for point in points:
                    val.append(abs(bending_curve.subs(x, point)))
                points.extend([singularity[i-1], s])
                val += [abs(limit(bending_curve, x, singularity[i-1], '+')), abs(limit(bending_curve, x, s, '-'))]
                max_moment = max(val)
                moment_values.append(max_moment)
                intervals.append(points[val.index(max_moment)])

            # If bending moment in a particular Interval has zero or constant
            # slope, then above block gives NotImplementedError as solve
            # can't represent Interval solutions.
            except NotImplementedError:
                initial_moment = limit(bending_curve, x, singularity[i-1], '+')
                final_moment = limit(bending_curve, x, s, '-')
                # If bending_curve has a constant slope(it is a line).
                if bending_curve.subs(x, (singularity[i-1] + s)/2) == (initial_moment + final_moment)/2 and initial_moment != final_moment:
                    moment_values.extend([initial_moment, final_moment])
                    intervals.extend([singularity[i-1], s])
                else:    # bending_curve has same value in whole Interval
                    moment_values.append(final_moment)
                    intervals.append(Interval(singularity[i-1], s))

        moment_values = list(map(abs, moment_values))
        maximum_moment = max(moment_values)
        point = intervals[moment_values.index(maximum_moment)]
        return (point, maximum_moment)

    def point_cflexure(self):
        """
        Returns a Set of point(s) with zero bending moment and
        where bending moment curve of the beam object changes
        its sign from negative to positive or vice versa.

        Examples
        ========
        There is is 10 meter long overhanging beam. There are
        two simple supports below the beam. One at the start
        and another one at a distance of 6 meters from the start.
        Point loads of magnitude 10KN and 20KN are applied at
        2 meters and 4 meters from start respectively. A Uniformly
        distribute load of magnitude of magnitude 3KN/m is also
        applied on top starting from 6 meters away from starting
        point till end.
        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(10, E, I)
        >>> b.apply_load(-4, 0, -1)
        >>> b.apply_load(-46, 6, -1)
        >>> b.apply_load(10, 2, -1)
        >>> b.apply_load(20, 4, -1)
        >>> b.apply_load(3, 6, 0)
        >>> b.point_cflexure()
        [10/3]
        """
        #Removes the singularity functions of order < 0 from the bending moment equation used in this method
        non_singular_bending_moment = sum(arg for arg in self.bending_moment().args if not arg.args[1].args[2] < 0)

        # To restrict the range within length of the Beam
        moment_curve = Piecewise((float("nan"), self.variable<=0),
                (non_singular_bending_moment, self.variable<self.length),
                (float("nan"), True))
        try:
            points = solve(moment_curve.rewrite(Piecewise), self.variable,
                           domain=S.Reals)
        except NotImplementedError as e:
            if "An expression is already zero when" in str(e):
                raise NotImplementedError("This method cannot be used when a whole region of "
                                          "the bending moment line is equal to 0.")
            else:
                raise

        return points

    def slope(self):
        """
        Returns a Singularity Function expression which represents
        the slope the elastic curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.slope()
        (-4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2)
            + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + 4000/3)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        I = self.second_moment

        if not self._boundary_conditions['slope']:
            return diff(self.deflection(), x)
        if isinstance(I, Piecewise) and self._joined_beam:
            args = I.args
            slope = 0
            prev_slope = 0
            prev_end = 0
            for i in range(len(args)):
                if i != 0:
                    prev_end = args[i-1][1].args[1]
                slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                if i != len(args) - 1:
                    slope += (prev_slope + slope_value)*SingularityFunction(x, prev_end, 0) - \
                        (prev_slope + slope_value)*SingularityFunction(x, args[i][1].args[1], 0)
                else:
                    slope += (prev_slope + slope_value)*SingularityFunction(x, prev_end, 0)
                prev_slope = slope_value.subs(x, args[i][1].args[1])
            return slope

        C3 = Symbol('C3')
        slope_curve = -integrate(S.One/(E*I)*self.bending_moment(), x) + C3

        bc_eqs = []
        for position, value in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            bc_eqs.append(eqs)
        constants = list(linsolve(bc_eqs, C3))
        slope_curve = slope_curve.subs({C3: constants[0][0]})
        return slope_curve

    def deflection(self):
        """
        Returns a Singularity Function expression which represents
        the elastic curve or deflection of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.deflection()
        (4000*x/3 - 4*SingularityFunction(x, 0, 3)/3 + SingularityFunction(x, 10, 3)
            + 60*SingularityFunction(x, 30, 2) + SingularityFunction(x, 30, 3)/3 - 12000)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        I = self.second_moment
        if not self._boundary_conditions['deflection'] and not self._boundary_conditions['slope']:
            if isinstance(I, Piecewise) and self._joined_beam:
                args = I.args
                prev_slope = 0
                prev_def = 0
                prev_end = 0
                deflection = 0
                for i in range(len(args)):
                    if i != 0:
                        prev_end = args[i-1][1].args[1]
                    slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                    recent_segment_slope = prev_slope + slope_value
                    deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                    if i != len(args) - 1:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                            - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                    else:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                    prev_slope = slope_value.subs(x, args[i][1].args[1])
                    prev_def = deflection_value.subs(x, args[i][1].args[1])
                return deflection
            base_char = self._base_char
            constants = symbols(base_char + '3:5')
            return S.One/(E*I)*integrate(-integrate(self.bending_moment(), x), x) + constants[0]*x + constants[1]
        elif not self._boundary_conditions['deflection']:
            base_char = self._base_char
            constant = symbols(base_char + '4')
            return integrate(self.slope(), x) + constant
        elif not self._boundary_conditions['slope'] and self._boundary_conditions['deflection']:
            if isinstance(I, Piecewise) and self._joined_beam:
                args = I.args
                prev_slope = 0
                prev_def = 0
                prev_end = 0
                deflection = 0
                for i in range(len(args)):
                    if i != 0:
                        prev_end = args[i-1][1].args[1]
                    slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                    recent_segment_slope = prev_slope + slope_value
                    deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                    if i != len(args) - 1:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                            - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                    else:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                    prev_slope = slope_value.subs(x, args[i][1].args[1])
                    prev_def = deflection_value.subs(x, args[i][1].args[1])
                return deflection
            base_char = self._base_char
            C3, C4 = symbols(base_char + '3:5')    # Integration constants
            slope_curve = -integrate(self.bending_moment(), x) + C3
            deflection_curve = integrate(slope_curve, x) + C4
            bc_eqs = []
            for position, value in self._boundary_conditions['deflection']:
                eqs = deflection_curve.subs(x, position) - value
                bc_eqs.append(eqs)
            constants = list(linsolve(bc_eqs, (C3, C4)))
            deflection_curve = deflection_curve.subs({C3: constants[0][0], C4: constants[0][1]})
            return S.One/(E*I)*deflection_curve

        if isinstance(I, Piecewise) and self._joined_beam:
            args = I.args
            prev_slope = 0
            prev_def = 0
            prev_end = 0
            deflection = 0
            for i in range(len(args)):
                if i != 0:
                    prev_end = args[i-1][1].args[1]
                slope_value = S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                recent_segment_slope = prev_slope + slope_value
                deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                if i != len(args) - 1:
                    deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                        - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                else:
                    deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                prev_slope = slope_value.subs(x, args[i][1].args[1])
                prev_def = deflection_value.subs(x, args[i][1].args[1])
            return deflection

        C4 = Symbol('C4')
        deflection_curve = integrate(self.slope(), x) + C4

        bc_eqs = []
        for position, value in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            bc_eqs.append(eqs)

        constants = list(linsolve(bc_eqs, C4))
        deflection_curve = deflection_curve.subs({C4: constants[0][0]})
        return deflection_curve

    def max_deflection(self):
        """
        Returns point of max deflection and its corresponding deflection value
        in a Beam object.
        """

        # To restrict the range within length of the Beam
        slope_curve = Piecewise((float("nan"), self.variable<=0),
                (self.slope(), self.variable<self.length),
                (float("nan"), True))

        points = solve(slope_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        deflection_curve = self.deflection()
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))
        if len(deflections) != 0:
            max_def = max(deflections)
            return (points[deflections.index(max_def)], max_def)
        else:
            return None

    def shear_stress(self):
        """
        Returns an expression representing the Shear Stress
        curve of the Beam object.
        """
        return self.shear_force()/self._area

    def plot_shear_stress(self, subs=None):
        """

        Returns a plot of shear stress present in the beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters and area of cross section 2 square
        meters. A constant distributed load of 10 KN/m is applied from half of
        the beam till the end. There are two simple supports below the beam,
        one at the starting point and another at the ending point of the beam.
        A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6), 2)
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_shear_stress()
            Plot object containing:
            [0]: cartesian line: 6875*SingularityFunction(x, 0, 0) - 2500*SingularityFunction(x, 2, 0)
            - 5000*SingularityFunction(x, 4, 1) + 15625*SingularityFunction(x, 8, 0)
            + 5000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)
        """

        shear_stress = self.shear_stress()
        x = self.variable
        length = self.length

        if subs is None:
            subs = {}
        for sym in shear_stress.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('value of %s was not passed.' %sym)

        if length in subs:
            length = subs[length]

        # Returns Plot of Shear Stress
        return plot (shear_stress.subs(subs), (x, 0, length),
        title='Shear Stress', xlabel=r'$\mathrm{x}$', ylabel=r'$\tau$',
        line_color='r')


    def plot_shear_force(self, subs=None):
        """

        Returns a plot for Shear force present in the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_shear_force()
            Plot object containing:
            [0]: cartesian line: 13750*SingularityFunction(x, 0, 0) - 5000*SingularityFunction(x, 2, 0)
            - 10000*SingularityFunction(x, 4, 1) + 31250*SingularityFunction(x, 8, 0)
            + 10000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)
        """
        shear_force = self.shear_force()
        if subs is None:
            subs = {}
        for sym in shear_force.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(shear_force.subs(subs), (self.variable, 0, length), title='Shear Force',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{V}$', line_color='g')

    def plot_bending_moment(self, subs=None):
        """

        Returns a plot for Bending moment present in the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_bending_moment()
            Plot object containing:
            [0]: cartesian line: 13750*SingularityFunction(x, 0, 1) - 5000*SingularityFunction(x, 2, 1)
            - 5000*SingularityFunction(x, 4, 2) + 31250*SingularityFunction(x, 8, 1)
            + 5000*SingularityFunction(x, 8, 2) for x over (0.0, 8.0)
        """
        bending_moment = self.bending_moment()
        if subs is None:
            subs = {}
        for sym in bending_moment.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(bending_moment.subs(subs), (self.variable, 0, length), title='Bending Moment',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{M}$', line_color='b')

    def plot_slope(self, subs=None):
        """

        Returns a plot for slope of deflection curve of the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_slope()
            Plot object containing:
            [0]: cartesian line: -8.59375e-5*SingularityFunction(x, 0, 2) + 3.125e-5*SingularityFunction(x, 2, 2)
            + 2.08333333333333e-5*SingularityFunction(x, 4, 3) - 0.0001953125*SingularityFunction(x, 8, 2)
            - 2.08333333333333e-5*SingularityFunction(x, 8, 3) + 0.00138541666666667 for x over (0.0, 8.0)
        """
        slope = self.slope()
        if subs is None:
            subs = {}
        for sym in slope.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(slope.subs(subs), (self.variable, 0, length), title='Slope',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\theta$', line_color='m')

    def plot_deflection(self, subs=None):
        """

        Returns a plot for deflection curve of the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_deflection()
            Plot object containing:
            [0]: cartesian line: 0.00138541666666667*x - 2.86458333333333e-5*SingularityFunction(x, 0, 3)
            + 1.04166666666667e-5*SingularityFunction(x, 2, 3) + 5.20833333333333e-6*SingularityFunction(x, 4, 4)
            - 6.51041666666667e-5*SingularityFunction(x, 8, 3) - 5.20833333333333e-6*SingularityFunction(x, 8, 4)
            for x over (0.0, 8.0)
        """
        deflection = self.deflection()
        if subs is None:
            subs = {}
        for sym in deflection.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(deflection.subs(subs), (self.variable, 0, length),
                    title='Deflection', xlabel=r'$\mathrm{x}$', ylabel=r'$\delta$',
                    line_color='r')


    def plot_loading_results(self, subs=None):
        """
        Returns a subplot of Shear Force, Bending Moment,
        Slope and Deflection of the Beam object.

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========

        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> axes = b.plot_loading_results()
        """
        length = self.length
        variable = self.variable
        if subs is None:
            subs = {}
        for sym in self.deflection().atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if length in subs:
            length = subs[length]
        ax1 = plot(self.shear_force().subs(subs), (variable, 0, length),
                   title="Shear Force", xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{V}$',
                   line_color='g', show=False)
        ax2 = plot(self.bending_moment().subs(subs), (variable, 0, length),
                   title="Bending Moment", xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{M}$',
                   line_color='b', show=False)
        ax3 = plot(self.slope().subs(subs), (variable, 0, length),
                   title="Slope", xlabel=r'$\mathrm{x}$', ylabel=r'$\theta$',
                   line_color='m', show=False)
        ax4 = plot(self.deflection().subs(subs), (variable, 0, length),
                   title="Deflection", xlabel=r'$\mathrm{x}$', ylabel=r'$\delta$',
                   line_color='r', show=False)

        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _solve_for_ild_equations(self, value):
        """

        Helper function for I.L.D. It takes the unsubstituted
        copy of the load equation and uses it to calculate shear force and bending
        moment equations.
        """
        x = self.variable
        a = self.ild_variable
        load = self._load + value * SingularityFunction(x, a, -1)
        shear_force = -integrate(load, x)
        bending_moment = integrate(shear_force, x)

        return shear_force, bending_moment

    def solve_for_ild_reactions(self, value, *reactions):
        """

        Determines the Influence Line Diagram equations for reaction
        forces under the effect of a moving load.

        Parameters
        ==========
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 10 meters. There are two simple supports
        below the beam, one at the starting point and another at the ending
        point of the beam. Calculate the I.L.D. equations for reaction forces
        under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_10 = symbols('R_0, R_10')
            >>> b = Beam(10, E, I)
            >>> p0 = b.apply_support(0, 'pin')
            >>> p10 = b.apply_support(10, 'roller')
            >>> b.solve_for_ild_reactions(1,R_0,R_10)
            >>> b.ild_reactions
            {R_0: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/10 - SingularityFunction(a, 10, 1)/10,
            R_10: -SingularityFunction(a, 0, 1)/10 + SingularityFunction(a, 10, 0) + SingularityFunction(a, 10, 1)/10}

        """
        shear_force, bending_moment = self._solve_for_ild_equations(value)
        x = self.variable
        l = self.length
        a = self.ild_variable

        rotation_jumps = tuple(self._rotation_hinge_symbols)
        deflection_jumps = tuple(self._sliding_hinge_symbols)

        C3 = Symbol('C3')
        C4 = Symbol('C4')

        shear_curve = limit(shear_force, x, l) - value*(SingularityFunction(a, 0, 0) - SingularityFunction(a, l, 0))
        moment_curve = (limit(bending_moment, x, l) - value * (l * SingularityFunction(a, 0, 0)
                                                               - SingularityFunction(a, 0, 1)
                                                               + SingularityFunction(a, l, 1)))

        shear_force_eqs = []
        bending_moment_eqs = []
        slope_eqs = []
        deflection_eqs = []

        for position, val in self._boundary_conditions['shear_force']:
            eqs = self.shear_force().subs(x, position) - val
            eqs_without_inf = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            shear_sinc = value * (SingularityFunction(- a, - position, 0) - SingularityFunction(-a, 0, 0))
            eqs_with_shear_sinc = eqs_without_inf - shear_sinc
            shear_force_eqs.append(eqs_with_shear_sinc)

        for position, val in self._boundary_conditions['bending_moment']:
            eqs = self.bending_moment().subs(x, position) - val
            eqs_without_inf = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            moment_sinc = value * (position * SingularityFunction(a, 0, 0)
                                   - SingularityFunction(a, 0, 1) + SingularityFunction(a, position, 1))
            eqs_with_moment_sinc = eqs_without_inf - moment_sinc
            bending_moment_eqs.append(eqs_with_moment_sinc)

        slope_curve = integrate(bending_moment, x) + C3
        for position, val in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - val + value * (SingularityFunction(-a, 0, 1) + position * SingularityFunction(-a, 0, 0))**2 / 2
            slope_eqs.append(eqs)

        deflection_curve = integrate(slope_curve, x) + C4
        for position, val in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - val + value * (SingularityFunction(-a, 0, 1) + position * SingularityFunction(-a, 0, 0)) ** 3 / 6
            deflection_eqs.append(eqs)

        solution = list((linsolve([shear_curve, moment_curve] + shear_force_eqs + bending_moment_eqs + slope_eqs
                                  + deflection_eqs, (C3, C4) + reactions + rotation_jumps + deflection_jumps).args)[0])

        reaction_index = 2 + len(reactions)
        rotation_index = reaction_index + len(rotation_jumps)
        reaction_solution = solution[2:reaction_index]
        rotation_solution = solution[reaction_index:rotation_index]
        deflection_solution = solution[rotation_index:]

        self._ild_reactions = dict(zip(reactions, reaction_solution))
        self._ild_rotations_jumps = dict(zip(rotation_jumps, rotation_solution))
        self._ild_deflection_jumps = dict(zip(deflection_jumps, deflection_solution))

    def plot_ild_reactions(self, subs=None):
        """

        Plots the Influence Line Diagram of Reaction Forces
        under the effect of a moving load. This function
        should be called after calling solve_for_ild_reactions().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 10 meters. A point load of magnitude 5KN
        is also applied from top of the beam, at a distance of 4 meters
        from the starting point. There are two simple supports below the
        beam, located at the starting point and at a distance of 7 meters
        from the starting point. Plot the I.L.D. equations for reactions
        at both support points under the effect of a moving load
        of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_7 = symbols('R_0, R_7')
            >>> b = Beam(10, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p7 = b.apply_support(7, 'roller')
            >>> b.apply_load(5,4,-1)
            >>> b.solve_for_ild_reactions(1,R_0,R_7)
            >>> b.ild_reactions
            {R_0: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/7
            - 3*SingularityFunction(a, 10, 0)/7  - SingularityFunction(a, 10, 1)/7 - 15/7,
            R_7: -SingularityFunction(a, 0, 1)/7 + 10*SingularityFunction(a, 10, 0)/7 + SingularityFunction(a, 10, 1)/7 - 20/7}
            >>> b.plot_ild_reactions()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/7
            - 3*SingularityFunction(a, 10, 0)/7 - SingularityFunction(a, 10, 1)/7 - 15/7 for a over (0.0, 10.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -SingularityFunction(a, 0, 1)/7 + 10*SingularityFunction(a, 10, 0)/7
            + SingularityFunction(a, 10, 1)/7 - 20/7 for a over (0.0, 10.0)

        """
        if not self._ild_reactions:
            raise ValueError("I.L.D. reaction equations not found. Please use solve_for_ild_reactions() to generate the I.L.D. reaction equations.")

        a = self.ild_variable
        ildplots = []

        if subs is None:
            subs = {}

        for reaction in self._ild_reactions:
            for sym in self._ild_reactions[reaction].atoms(Symbol):
                if sym != a and sym not in subs:
                    raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for reaction in self._ild_reactions:
            ildplots.append(plot(self._ild_reactions[reaction].subs(subs),
            (a, 0, self._length.subs(subs)), title='I.L.D. for Reactions',
            xlabel=a, ylabel=reaction, line_color='blue', show=False))

        return PlotGrid(len(ildplots), 1, *ildplots)

    def solve_for_ild_shear(self, distance, value, *reactions):
        """

        Determines the Influence Line Diagram equations for shear at a
        specified point under the effect of a moving load.

        Parameters
        ==========
        distance : Integer
            Distance of the point from the start of the beam
            for which equations are to be determined
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Calculate the I.L.D. equations for Shear at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)
            >>> b.ild_shear
            -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1

        """

        x = self.variable
        l = self.length
        a = self.ild_variable

        shear_force, _ = self._solve_for_ild_equations(value)

        shear_curve1 = value - limit(shear_force, x, distance)
        shear_curve2 = (limit(shear_force, x, l) - limit(shear_force, x, distance)) - value

        for reaction in reactions:
            shear_curve1 = shear_curve1.subs(reaction,self._ild_reactions[reaction])
            shear_curve2 = shear_curve2.subs(reaction,self._ild_reactions[reaction])

        shear_eq = (shear_curve1 - (shear_curve1 - shear_curve2) * SingularityFunction(a, distance, 0)
                    - value * SingularityFunction(-a, 0, 0) + value * SingularityFunction(a, l, 0))

        self._ild_shear = shear_eq

    def plot_ild_shear(self,subs=None):
        """

        Plots the Influence Line Diagram for Shear under the effect
        of a moving load. This function should be called after
        calling solve_for_ild_shear().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Plot the I.L.D. for Shear at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)
            >>> b.ild_shear
            -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1
            >>> b.plot_ild_shear()
            Plot object containing:
            [0]: cartesian line: -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1 for a over (0.0, 12.0)

        """

        if not self._ild_shear:
            raise ValueError("I.L.D. shear equation not found. Please use solve_for_ild_shear() to generate the I.L.D. shear equations.")

        l = self._length
        a = self.ild_variable

        if subs is None:
            subs = {}

        for sym in self._ild_shear.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        return plot(self._ild_shear.subs(subs), (a, 0, l),  title='I.L.D. for Shear',
               xlabel=r'$\mathrm{a}$', ylabel=r'$\mathrm{V}$', line_color='blue',show=True)

    def solve_for_ild_moment(self, distance, value, *reactions):
        """

        Determines the Influence Line Diagram equations for moment at a
        specified point under the effect of a moving load.

        Parameters
        ==========
        distance : Integer
            Distance of the point from the start of the beam
            for which equations are to be determined
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Calculate the I.L.D. equations for Moment at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)
            >>> b.ild_moment
            -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1) + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0)
            - SingularityFunction(a, 0, 1)/2 + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0)
            - SingularityFunction(a, 12, 1)/2

        """

        x = self.variable
        l = self.length
        a = self.ild_variable

        _, moment = self._solve_for_ild_equations(value)

        moment_curve1 = value*(distance * SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
                               + SingularityFunction(a, distance, 1)) - limit(moment, x, distance)
        moment_curve2 = (limit(moment, x, l)-limit(moment, x, distance)
                         - value * (l * SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
                                    + SingularityFunction(a, l, 1)))

        for reaction in reactions:
            moment_curve1 = moment_curve1.subs(reaction, self._ild_reactions[reaction])
            moment_curve2 = moment_curve2.subs(reaction, self._ild_reactions[reaction])

        moment_eq = moment_curve1 - (moment_curve1 - moment_curve2) * SingularityFunction(a, distance, 0)

        self._ild_moment = moment_eq

    def plot_ild_moment(self,subs=None):
        """

        Plots the Influence Line Diagram for Moment under the effect
        of a moving load. This function should be called after
        calling solve_for_ild_moment().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Plot the I.L.D. for Moment at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)
            >>> b.ild_moment
            -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1) + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0)
            - SingularityFunction(a, 0, 1)/2 + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0)
            - SingularityFunction(a, 12, 1)/2
            >>> b.plot_ild_moment()
            Plot object containing:
            [0]: cartesian line: -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
            + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0) - SingularityFunction(a, 0, 1)/2
            + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0) - SingularityFunction(a, 12, 1)/2 for a over (0.0, 12.0)

        """

        if not self._ild_moment:
            raise ValueError("I.L.D. moment equation not found. Please use solve_for_ild_moment() to generate the I.L.D. moment equations.")

        a = self.ild_variable

        if subs is None:
            subs = {}

        for sym in self._ild_moment.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        return plot(self._ild_moment.subs(subs), (a, 0, self._length), title='I.L.D. for Moment',
               xlabel=r'$\mathrm{a}$', ylabel=r'$\mathrm{M}$', line_color='blue', show=True)

    @doctest_depends_on(modules=('numpy',))
    def draw(self, pictorial=True):
        """
        Returns a plot object representing the beam diagram of the beam.
        In particular, the diagram might include:

        * the beam.
        * vertical black arrows represent point loads and support reaction
          forces (the latter if they have been added with the ``apply_load``
          method).
        * circular arrows represent moments.
        * shaded areas represent distributed loads.
        * the support, if ``apply_support`` has been executed.
        * if a composite beam has been created with the ``join`` method and
          a hinge has been specified, it will be shown with a white disc.

        The diagram shows positive loads on the upper side of the beam,
        and negative loads on the lower side. If two or more distributed
        loads acts along the same direction over the same region, the
        function will add them up together.

        .. note::
            The user must be careful while entering load values.
            The draw function assumes a sign convention which is used
            for plotting loads.
            Given a right handed coordinate system with XYZ coordinates,
            the beam's length is assumed to be along the positive X axis.
            The draw function recognizes positive loads(with n>-2) as loads
            acting along negative Y direction and positive moments acting
            along positive Z direction.

        Parameters
        ==========

        pictorial: Boolean (default=True)
            Setting ``pictorial=True`` would simply create a pictorial (scaled)
            view of the beam diagram. On the other hand, ``pictorial=False``
            would create a beam diagram with the exact dimensions on the plot.

        Examples
        ========

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> P1, P2, M = symbols('P1, P2, M')
            >>> E, I = symbols('E, I')
            >>> b = Beam(50, 20, 30)
            >>> b.apply_load(-10, 2, -1)
            >>> b.apply_load(15, 26, -1)
            >>> b.apply_load(P1, 10, -1)
            >>> b.apply_load(-P2, 40, -1)
            >>> b.apply_load(90, 5, 0, 23)
            >>> b.apply_load(10, 30, 1, 50)
            >>> b.apply_load(M, 15, -2)
            >>> b.apply_load(-M, 30, -2)
            >>> p50 = b.apply_support(50, "pin")
            >>> p0, m0 = b.apply_support(0, "fixed")
            >>> p20 = b.apply_support(20, "roller")
            >>> p = b.draw()  # doctest: +SKIP
            >>> p  # doctest: +ELLIPSIS,+SKIP
            Plot object containing:
            [0]: cartesian line: 25*SingularityFunction(x, 5, 0) - 25*SingularityFunction(x, 23, 0)
            + SingularityFunction(x, 30, 1) - 20*SingularityFunction(x, 50, 0)
            - SingularityFunction(x, 50, 1) + 5 for x over (0.0, 50.0)
            [1]: cartesian line: 5 for x over (0.0, 50.0)
            ...
            >>> p.show() # doctest: +SKIP

        """
        if not numpy:
            raise ImportError("To use this function numpy module is required")

        loads = list(set(self.applied_loads) - set(self._support_as_loads))
        if (not pictorial) and any((len(l[0].free_symbols) > 0) and (l[2] >= 0) for l in loads):
            raise ValueError("`pictorial=False` requires numerical "
                "distributed loads. Instead, symbolic loads were found. "
                "Cannot continue.")

        x = self.variable

        # checking whether length is an expression in terms of any Symbol.
        if isinstance(self.length, Expr):
            l = list(self.length.atoms(Symbol))
            # assigning every Symbol a default value of 10
            l = dict.fromkeys(l, 10)
            length = self.length.subs(l)
        else:
            l = {}
            length = self.length
        height = length/10

        rectangles = []
        rectangles.append({'xy':(0, 0), 'width':length, 'height': height, 'facecolor':"brown"})
        annotations, markers, load_eq,load_eq1, fill = self._draw_load(pictorial, length, l)
        support_markers, support_rectangles = self._draw_supports(length, l)

        rectangles += support_rectangles
        markers += support_markers

        for loc in self._applied_rotation_hinges:
            ratio = loc / self.length
            x_pos = float(ratio) * length
            markers += [{'args':[[x_pos], [height / 2]], 'marker':'o', 'markersize':6, 'color':"white"}]

        for loc in self._applied_sliding_hinges:
            ratio = loc / self.length
            x_pos = float(ratio) * length
            markers += [{'args': [[x_pos], [height / 2]], 'marker':'|', 'markersize':12, 'color':"white"}]

        ylim = (-length, 1.25*length)
        if fill:
            # when distributed loads are presents, they might get clipped out
            # in the figure by the ylim settings.
            # It might be necessary to compute new limits.
            _min = min(min(fill["y2"]), min(r["xy"][1] for r in rectangles))
            _max = max(max(fill["y1"]), max(r["xy"][1] for r in rectangles))
            if (_min < ylim[0]) or (_max > ylim[1]):
                offset = abs(_max - _min) * 0.1
                ylim = (_min - offset, _max + offset)

        sing_plot = plot(height + load_eq, height + load_eq1, (x, 0, length),
            xlim=(-height, length + height), ylim=ylim,
            annotations=annotations, markers=markers, rectangles=rectangles,
            line_color='brown', fill=fill, axis=False, show=False)

        return sing_plot


    def _is_load_negative(self, load):
        """Try to determine if a load is negative or positive, using
        expansion and doit if necessary.

        Returns
        =======
        True: if the load is negative
        False: if the load is positive
        None: if it is indeterminate

        """
        rv = load.is_negative
        if load.is_Atom or rv is not None:
            return rv
        return load.doit().expand().is_negative

    def _draw_load(self, pictorial, length, l):
        loads = list(set(self.applied_loads) - set(self._support_as_loads))
        height = length/10
        x = self.variable

        annotations = []
        markers = []
        load_args = []
        scaled_load = 0
        load_args1 = []
        scaled_load1 = 0
        load_eq = S.Zero     # For positive valued higher order loads
        load_eq1 = S.Zero    # For negative valued higher order loads
        fill = None

        # schematic view should use the class convention as much as possible.
        # However, users can add expressions as symbolic loads, for example
        # P1 - P2: is this load positive or negative? We can't say.
        # On these occasions it is better to inform users about the
        # indeterminate state of those loads.
        warning_head = "Please, note that this schematic view might not be " \
            "in agreement with the sign convention used by the Beam class " \
            "for load-related computations, because it was not possible " \
            "to determine the sign (hence, the direction) of the " \
            "following loads:\n"
        warning_body = ""

        for load in loads:
            # check if the position of load is in terms of the beam length.
            if l:
                pos =  load[1].subs(l)
            else:
                pos = load[1]

            # point loads
            if load[2] == -1:
                iln = self._is_load_negative(load[0])
                if iln is None:
                    warning_body += "* Point load %s located at %s\n" % (load[0], load[1])
                if iln:
                    annotations.append({'text':'', 'xy':(pos, 0), 'xytext':(pos, height - 4*height), 'arrowprops':{'width': 1.5, 'headlength': 5, 'headwidth': 5, 'facecolor': 'black'}})
                else:
                    annotations.append({'text':'', 'xy':(pos, height),  'xytext':(pos, height*4), 'arrowprops':{"width": 1.5, "headlength": 4, "headwidth": 4, "facecolor": 'black'}})
            # moment loads
            elif load[2] == -2:
                iln = self._is_load_negative(load[0])
                if iln is None:
                    warning_body += "* Moment %s located at %s\n" % (load[0], load[1])
                if self._is_load_negative(load[0]):
                    markers.append({'args':[[pos], [height/2]], 'marker': r'$\circlearrowright$', 'markersize':15})
                else:
                    markers.append({'args':[[pos], [height/2]], 'marker': r'$\circlearrowleft$', 'markersize':15})
            # higher order loads
            elif load[2] >= 0:
                # `fill` will be assigned only when higher order loads are present
                value, start, order, end = load

                iln = self._is_load_negative(value)
                if iln is None:
                    warning_body += "* Distributed load %s from %s to %s\n" % (value, start, end)

                # Positive loads have their separate equations
                if not iln:
                    # if pictorial is True we remake the load equation again with
                    # some constant magnitude values.
                    if pictorial:
                        # remake the load equation again with some constant
                        # magnitude values.
                        value = 10**(1-order) if order > 0 else length/2
                    scaled_load += value*SingularityFunction(x, start, order)
                    if end:
                        f2 = value*x**order if order >= 0 else length/2*x**order
                        for i in range(0, order + 1):
                            scaled_load -= (f2.diff(x, i).subs(x, end - start)*
                                            SingularityFunction(x, end, i)/factorial(i))

                    if isinstance(scaled_load, Add):
                        load_args = scaled_load.args
                    else:
                        # when the load equation consists of only a single term
                        load_args = (scaled_load,)
                    load_eq = Add(*[i.subs(l) for i in load_args])

                # For loads with negative value
                else:
                    if pictorial:
                        # remake the load equation again with some constant
                        # magnitude values.
                        value = 10**(1-order) if order > 0 else length/2
                    scaled_load1 += abs(value)*SingularityFunction(x, start, order)
                    if end:
                        f2 = abs(value)*x**order if order >= 0 else length/2*x**order
                        for i in range(0, order + 1):
                            scaled_load1 -= (f2.diff(x, i).subs(x, end - start)*
                                            SingularityFunction(x, end, i)/factorial(i))

                    if isinstance(scaled_load1, Add):
                        load_args1 = scaled_load1.args
                    else:
                        # when the load equation consists of only a single term
                        load_args1 = (scaled_load1,)
                    load_eq1 = [i.subs(l) for i in load_args1]
                    load_eq1 = -Add(*load_eq1) - height

        if len(warning_body) > 0:
            warnings.warn(warning_head + warning_body)

        xx = numpy.arange(0, float(length), 0.001)
        yy1 = lambdify([x], height + load_eq.rewrite(Piecewise))(xx)
        yy2 = lambdify([x], height + load_eq1.rewrite(Piecewise))(xx)
        if not isinstance(yy1, numpy.ndarray):
            yy1 *= numpy.ones_like(xx)
        if not isinstance(yy2, numpy.ndarray):
            yy2 *= numpy.ones_like(xx)
        fill = {'x': xx, 'y1': yy1, 'y2': yy2,
            'color':'darkkhaki', "zorder": -1}
        return annotations, markers, load_eq, load_eq1, fill


    def _draw_supports(self, length, l):
        height = float(length/10)

        support_markers = []
        support_rectangles = []
        for support in self._applied_supports:
            if l:
                pos =  support[0].subs(l)
            else:
                pos = support[0]

            if support[1] == "pin":
                support_markers.append({'args':[pos, [0]], 'marker':6, 'markersize':13, 'color':"black"})

            elif support[1] == "roller":
                support_markers.append({'args':[pos, [-height/2.5]], 'marker':'o', 'markersize':11, 'color':"black"})

            elif support[1] == "fixed":
                if pos == 0:
                    support_rectangles.append({'xy':(0, -3*height), 'width':-length/20, 'height':6*height + height, 'fill':False, 'hatch':'/////'})
                else:
                    support_rectangles.append({'xy':(length, -3*height), 'width':length/20, 'height': 6*height + height, 'fill':False, 'hatch':'/////'})

        return support_markers, support_rectangles


class Beam3D(Beam):
    """
    This class handles loads applied in any direction of a 3D space along
    with unequal values of Second moment along different axes.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention.
       This class assumes that any kind of distributed load/moment is
       applied through out the span of a beam.

    Examples
    ========
    There is a beam of l meters long. A constant distributed load of magnitude q
    is applied along y-axis from start till the end of beam. A constant distributed
    moment of magnitude m is also applied along z-axis from start till the end of beam.
    Beam is fixed at both of its end. So, deflection of the beam at the both ends
    is restricted.

    >>> from sympy.physics.continuum_mechanics.beam import Beam3D
    >>> from sympy import symbols, simplify, collect, factor
    >>> l, E, G, I, A = symbols('l, E, G, I, A')
    >>> b = Beam3D(l, E, G, I, A)
    >>> x, q, m = symbols('x, q, m')
    >>> b.apply_load(q, 0, 0, dir="y")
    >>> b.apply_moment_load(m, 0, -1, dir="z")
    >>> b.shear_force()
    [0, -q*x, 0]
    >>> b.bending_moment()
    [0, 0, -m*x + q*x**2/2]
    >>> b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.solve_slope_deflection()
    >>> factor(b.slope())
    [0, 0, x*(-l + x)*(-A*G*l**3*q + 2*A*G*l**2*q*x - 12*E*I*l*q
        - 72*E*I*m + 24*E*I*q*x)/(12*E*I*(A*G*l**2 + 12*E*I))]
    >>> dx, dy, dz = b.deflection()
    >>> dy = collect(simplify(dy), x)
    >>> dx == dz == 0
    True
    >>> dy == (x*(12*E*I*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q)
    ... + x*(A*G*l*(3*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q) + x*(-2*A*G*l**2*q + 4*A*G*l*m - 24*E*I*q))
    ... + A*G*(A*G*l**2 + 12*E*I)*(-2*l**2*q + 6*l*m - 4*m*x + q*x**2)
    ... - 12*E*I*q*(A*G*l**2 + 12*E*I)))/(24*A*E*G*I*(A*G*l**2 + 12*E*I)))
    True

    References
    ==========

    .. [1] https://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf

    """

    def __init__(self, length, elastic_modulus, shear_modulus, second_moment,
                 area, variable=Symbol('x')):
        """Initializes the class.

        Parameters
        ==========
        length : Sympifyable
            A Symbol or value representing the Beam's length.
        elastic_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of Elasticity.
            It is a measure of the stiffness of the Beam material.
        shear_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of rigidity.
            It is a measure of rigidity of the Beam material.
        second_moment : Sympifyable or list
            A list of two elements having SymPy expression representing the
            Beam's Second moment of area. First value represent Second moment
            across y-axis and second across z-axis.
            Single SymPy expression can be passed if both values are same
        area : Sympifyable
            A SymPy expression representing the Beam's cross-sectional area
            in a plane perpendicular to length of the Beam.
        variable : Symbol, optional
            A Symbol object that will be used as the variable along the beam
            while representing the load, shear, moment, slope and deflection
            curve. By default, it is set to ``Symbol('x')``.
        """
        super().__init__(length, elastic_modulus, second_moment, variable)
        self.shear_modulus = shear_modulus
        self.area = area
        self._load_vector = [0, 0, 0]
        self._moment_load_vector = [0, 0, 0]
        self._torsion_moment = {}
        self._load_Singularity = [0, 0, 0]
        self._slope = [0, 0, 0]
        self._deflection = [0, 0, 0]
        self._angular_deflection = 0

    @property
    def shear_modulus(self):
        """Young's Modulus of the Beam. """
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, e):
        self._shear_modulus = sympify(e)

    @property
    def second_moment(self):
        """Second moment of area of the Beam. """
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        if isinstance(i, list):
            i = [sympify(x) for x in i]
            self._second_moment = i
        else:
            self._second_moment = sympify(i)

    @property
    def area(self):
        """Cross-sectional area of the Beam. """
        return self._area

    @area.setter
    def area(self, a):
        self._area = sympify(a)

    @property
    def load_vector(self):
        """
        Returns a three element list representing the load vector.
        """
        return self._load_vector

    @property
    def moment_load_vector(self):
        """
        Returns a three element list representing moment loads on Beam.
        """
        return self._moment_load_vector

    @property
    def boundary_conditions(self):
        """
        Returns a dictionary of boundary conditions applied on the beam.
        The dictionary has two keywords namely slope and deflection.
        The value of each keyword is a list of tuple, where each tuple
        contains location and value of a boundary condition in the format
        (location, value). Further each value is a list corresponding to
        slope or deflection(s) values along three axes at that location.

        Examples
        ========
        There is a beam of length 4 meters. The slope at 0 should be 4 along
        the x-axis and 0 along others. At the other end of beam, deflection
        along all the three axes should be zero.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.bc_slope = [(0, (4, 0, 0))]
        >>> b.bc_deflection = [(4, [0, 0, 0])]
        >>> b.boundary_conditions
        {'bending_moment': [], 'deflection': [(4, [0, 0, 0])], 'shear_force': [], 'slope': [(0, (4, 0, 0))]}

        Here the deflection of the beam should be ``0`` along all the three axes at ``4``.
        Similarly, the slope of the beam should be ``4`` along x-axis and ``0``
        along y and z axis at ``0``.
        """
        return self._boundary_conditions

    def polar_moment(self):
        """
        Returns the polar moment of area of the beam
        about the X axis with respect to the centroid.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A = symbols('l, E, G, I, A')
        >>> b = Beam3D(l, E, G, I, A)
        >>> b.polar_moment()
        2*I
        >>> I1 = [9, 15]
        >>> b = Beam3D(l, E, G, I1, A)
        >>> b.polar_moment()
        24
        """
        if not iterable(self.second_moment):
            return 2*self.second_moment
        return sum(self.second_moment)

    def apply_load(self, value, start, order, dir="y"):
        """
        This method adds up the force load to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        dir : String
            Axis along which load is applied.
        order : Integer
            The order of the applied load.
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if dir == "x":
            if not order == -1:
                self._load_vector[0] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)

        elif dir == "y":
            if not order == -1:
                self._load_vector[1] += value
            self._load_Singularity[1] += value*SingularityFunction(x, start, order)

        else:
            if not order == -1:
                self._load_vector[2] += value
            self._load_Singularity[2] += value*SingularityFunction(x, start, order)

    def apply_moment_load(self, value, start, order, dir="y"):
        """
        This method adds up the moment loads to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied moment.
        dir : String
            Axis along which moment is applied.
        order : Integer
            The order of the applied load.
            - For point moments, order=-2
            - For constant distributed moment, order=-1
            - For ramp moments, order=0
            - For parabolic ramp moments, order=1
            - ... so on.
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if dir == "x":
            if not order == -2:
                self._moment_load_vector[0] += value
            else:
                if start in list(self._torsion_moment):
                    self._torsion_moment[start] += value
                else:
                    self._torsion_moment[start] = value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)
        elif dir == "y":
            if not order == -2:
                self._moment_load_vector[1] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)
        else:
            if not order == -2:
                self._moment_load_vector[2] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)

    def apply_support(self, loc, type="fixed"):
        if type in ("pin", "roller"):
            reaction_load = Symbol('R_'+str(loc))
            self._reaction_loads[reaction_load] = reaction_load
            self.bc_deflection.append((loc, [0, 0, 0]))
        else:
            reaction_load = Symbol('R_'+str(loc))
            reaction_moment = Symbol('M_'+str(loc))
            self._reaction_loads[reaction_load] = [reaction_load, reaction_moment]
            self.bc_deflection.append((loc, [0, 0, 0]))
            self.bc_slope.append((loc, [0, 0, 0]))

    def solve_for_reaction_loads(self, *reaction):
        """
        Solves for the reaction forces.

        Examples
        ========
        There is a beam of length 30 meters. It it supported by rollers at
        of its end. A constant distributed load of magnitude 8 N is applied
        from start till its end along y-axis. Another linear load having
        slope equal to 9 is applied along z-axis.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.apply_load(8, start=0, order=0, dir="y")
        >>> b.apply_load(9*x, start=0, order=0, dir="z")
        >>> b.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]
        >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
        >>> b.apply_load(R1, start=0, order=-1, dir="y")
        >>> b.apply_load(R2, start=30, order=-1, dir="y")
        >>> b.apply_load(R3, start=0, order=-1, dir="z")
        >>> b.apply_load(R4, start=30, order=-1, dir="z")
        >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
        >>> b.reaction_loads
        {R1: -120, R2: -120, R3: -1350, R4: -2700}
        """
        x = self.variable
        l = self.length
        q = self._load_Singularity
        shear_curves = [integrate(load, x) for load in q]
        moment_curves = [integrate(shear, x) for shear in shear_curves]
        for i in range(3):
            react = [r for r in reaction if (shear_curves[i].has(r) or moment_curves[i].has(r))]
            if len(react) == 0:
                continue
            shear_curve = limit(shear_curves[i], x, l)
            moment_curve = limit(moment_curves[i], x, l)
            sol = list((linsolve([shear_curve, moment_curve], react).args)[0])
            sol_dict = dict(zip(react, sol))
            reaction_loads = self._reaction_loads
            # Check if any of the evaluated reaction exists in another direction
            # and if it exists then it should have same value.
            for key in sol_dict:
                if key in reaction_loads and sol_dict[key] != reaction_loads[key]:
                    raise ValueError("Ambiguous solution for %s in different directions." % key)
            self._reaction_loads.update(sol_dict)

    def shear_force(self):
        """
        Returns a list of three expressions which represents the shear force
        curve of the Beam object along all three axes.
        """
        x = self.variable
        q = self._load_vector
        return [integrate(-q[0], x), integrate(-q[1], x), integrate(-q[2], x)]

    def axial_force(self):
        """
        Returns expression of Axial shear force present inside the Beam object.
        """
        return self.shear_force()[0]

    def shear_stress(self):
        """
        Returns a list of three expressions which represents the shear stress
        curve of the Beam object along all three axes.
        """
        return [self.shear_force()[0]/self._area, self.shear_force()[1]/self._area, self.shear_force()[2]/self._area]

    def axial_stress(self):
        """
        Returns expression of Axial stress present inside the Beam object.
        """
        return self.axial_force()/self._area

    def bending_moment(self):
        """
        Returns a list of three expressions which represents the bending moment
        curve of the Beam object along all three axes.
        """
        x = self.variable
        m = self._moment_load_vector
        shear = self.shear_force()

        return [integrate(-m[0], x), integrate(-m[1] + shear[2], x),
                integrate(-m[2] - shear[1], x) ]

    def torsional_moment(self):
        """
        Returns expression of Torsional moment present inside the Beam object.
        """
        return self.bending_moment()[0]

    def solve_for_torsion(self):
        """
        Solves for the angular deflection due to the torsional effects of
        moments being applied in the x-direction i.e. out of or into the beam.

        Here, a positive torque means the direction of the torque is positive
        i.e. out of the beam along the beam-axis. Likewise, a negative torque
        signifies a torque into the beam cross-section.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(20, E, G, I, A, x)
        >>> b.apply_moment_load(4, 4, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.solve_for_torsion()
        >>> b.angular_deflection().subs(x, 3)
        18/(G*I)
        """
        x = self.variable
        sum_moments = 0
        for point in list(self._torsion_moment):
            sum_moments += self._torsion_moment[point]
        list(self._torsion_moment).sort()
        pointsList = list(self._torsion_moment)
        torque_diagram = Piecewise((sum_moments, x<=pointsList[0]), (0, x>=pointsList[0]))
        for i in range(len(pointsList))[1:]:
            sum_moments -= self._torsion_moment[pointsList[i-1]]
            torque_diagram += Piecewise((0, x<=pointsList[i-1]), (sum_moments, x<=pointsList[i]), (0, x>=pointsList[i]))
        integrated_torque_diagram = integrate(torque_diagram)
        self._angular_deflection =  integrated_torque_diagram/(self.shear_modulus*self.polar_moment())

    def solve_slope_deflection(self):
        x = self.variable
        l = self.length
        E = self.elastic_modulus
        G = self.shear_modulus
        I = self.second_moment
        if isinstance(I, list):
            I_y, I_z = I[0], I[1]
        else:
            I_y = I_z = I
        A = self._area
        load = self._load_vector
        moment = self._moment_load_vector
        defl = Function('defl')
        theta = Function('theta')

        # Finding deflection along x-axis(and corresponding slope value by differentiating it)
        # Equation used: Derivative(E*A*Derivative(def_x(x), x), x) + load_x = 0
        eq = Derivative(E*A*Derivative(defl(x), x), x) + load[0]
        def_x = dsolve(Eq(eq, 0), defl(x)).args[1]
        # Solving constants originated from dsolve
        C1 = Symbol('C1')
        C2 = Symbol('C2')
        constants = list((linsolve([def_x.subs(x, 0), def_x.subs(x, l)], C1, C2).args)[0])
        def_x = def_x.subs({C1:constants[0], C2:constants[1]})
        slope_x = def_x.diff(x)
        self._deflection[0] = def_x
        self._slope[0] = slope_x

        # Finding deflection along y-axis and slope across z-axis. System of equation involved:
        # 1: Derivative(E*I_z*Derivative(theta_z(x), x), x) + G*A*(Derivative(defl_y(x), x) - theta_z(x)) + moment_z = 0
        # 2: Derivative(G*A*(Derivative(defl_y(x), x) - theta_z(x)), x) + load_y = 0
        C_i = Symbol('C_i')
        # Substitute value of `G*A*(Derivative(defl_y(x), x) - theta_z(x))` from (2) in (1)
        eq1 = Derivative(E*I_z*Derivative(theta(x), x), x) + (integrate(-load[1], x) + C_i) + moment[2]
        slope_z = dsolve(Eq(eq1, 0)).args[1]

        # Solve for constants originated from using dsolve on eq1
        constants = list((linsolve([slope_z.subs(x, 0), slope_z.subs(x, l)], C1, C2).args)[0])
        slope_z = slope_z.subs({C1:constants[0], C2:constants[1]})

        # Put value of slope obtained back in (2) to solve for `C_i` and find deflection across y-axis
        eq2 = G*A*(Derivative(defl(x), x)) + load[1]*x - C_i - G*A*slope_z
        def_y = dsolve(Eq(eq2, 0), defl(x)).args[1]
        # Solve for constants originated from using dsolve on eq2
        constants = list((linsolve([def_y.subs(x, 0), def_y.subs(x, l)], C1, C_i).args)[0])
        self._deflection[1] = def_y.subs({C1:constants[0], C_i:constants[1]})
        self._slope[2] = slope_z.subs(C_i, constants[1])

        # Finding deflection along z-axis and slope across y-axis. System of equation involved:
        # 1: Derivative(E*I_y*Derivative(theta_y(x), x), x) - G*A*(Derivative(defl_z(x), x) + theta_y(x)) + moment_y = 0
        # 2: Derivative(G*A*(Derivative(defl_z(x), x) + theta_y(x)), x) + load_z = 0

        # Substitute value of `G*A*(Derivative(defl_y(x), x) + theta_z(x))` from (2) in (1)
        eq1 = Derivative(E*I_y*Derivative(theta(x), x), x) + (integrate(load[2], x) - C_i) + moment[1]
        slope_y = dsolve(Eq(eq1, 0)).args[1]
        # Solve for constants originated from using dsolve on eq1
        constants = list((linsolve([slope_y.subs(x, 0), slope_y.subs(x, l)], C1, C2).args)[0])
        slope_y = slope_y.subs({C1:constants[0], C2:constants[1]})

        # Put value of slope obtained back in (2) to solve for `C_i` and find deflection across z-axis
        eq2 = G*A*(Derivative(defl(x), x)) + load[2]*x - C_i + G*A*slope_y
        def_z = dsolve(Eq(eq2,0)).args[1]
        # Solve for constants originated from using dsolve on eq2
        constants = list((linsolve([def_z.subs(x, 0), def_z.subs(x, l)], C1, C_i).args)[0])
        self._deflection[2] = def_z.subs({C1:constants[0], C_i:constants[1]})
        self._slope[1] = slope_y.subs(C_i, constants[1])

    def slope(self):
        """
        Returns a three element list representing slope of deflection curve
        along all the three axes.
        """
        return self._slope

    def deflection(self):
        """
        Returns a three element list representing deflection curve along all
        the three axes.
        """
        return self._deflection

    def angular_deflection(self):
        """
        Returns a function in x depicting how the angular deflection, due to moments
        in the x-axis on the beam, varies with x.
        """
        return self._angular_deflection

    def _plot_shear_force(self, dir, subs=None):

        shear_force = self.shear_force()

        if dir == 'x':
            dir_num = 0
            color = 'r'

        elif dir == 'y':
            dir_num = 1
            color = 'g'

        elif dir == 'z':
            dir_num = 2
            color = 'b'

        if subs is None:
            subs = {}

        for sym in shear_force[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(shear_force[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Shear Force along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{V(%c)}$'%dir, line_color=color)

    def plot_shear_force(self, dir="all", subs=None):

        """

        Returns a plot for Shear force along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear force plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_force()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For shear force along x direction
        if dir == "x":
            Px = self._plot_shear_force('x', subs)
            return Px.show()
        # For shear force along y direction
        elif dir == "y":
            Py = self._plot_shear_force('y', subs)
            return Py.show()
        # For shear force along z direction
        elif dir == "z":
            Pz = self._plot_shear_force('z', subs)
            return Pz.show()
        # For shear force along all direction
        else:
            Px = self._plot_shear_force('x', subs)
            Py = self._plot_shear_force('y', subs)
            Pz = self._plot_shear_force('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_bending_moment(self, dir, subs=None):

        bending_moment = self.bending_moment()

        if dir == 'x':
            dir_num = 0
            color = 'g'

        elif dir == 'y':
            dir_num = 1
            color = 'c'

        elif dir == 'z':
            dir_num = 2
            color = 'm'

        if subs is None:
            subs = {}

        for sym in bending_moment[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(bending_moment[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Bending Moment along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{M(%c)}$'%dir, line_color=color)

    def plot_bending_moment(self, dir="all", subs=None):

        """

        Returns a plot for bending moment along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which bending moment plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_bending_moment()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: 2*x**3 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For bending moment along x direction
        if dir == "x":
            Px = self._plot_bending_moment('x', subs)
            return Px.show()
        # For bending moment along y direction
        elif dir == "y":
            Py = self._plot_bending_moment('y', subs)
            return Py.show()
        # For bending moment along z direction
        elif dir == "z":
            Pz = self._plot_bending_moment('z', subs)
            return Pz.show()
        # For bending moment along all direction
        else:
            Px = self._plot_bending_moment('x', subs)
            Py = self._plot_bending_moment('y', subs)
            Pz = self._plot_bending_moment('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_slope(self, dir, subs=None):

        slope = self.slope()

        if dir == 'x':
            dir_num = 0
            color = 'b'

        elif dir == 'y':
            dir_num = 1
            color = 'm'

        elif dir == 'z':
            dir_num = 2
            color = 'g'

        if subs is None:
            subs = {}

        for sym in slope[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length


        return plot(slope[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Slope along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{\theta(%c)}$'%dir, line_color=color)

    def plot_slope(self, dir="all", subs=None):

        """

        Returns a plot for Slope along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which Slope plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_slope()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/8000 - 19*x**2/172 + 52*x/43 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For Slope along x direction
        if dir == "x":
            Px = self._plot_slope('x', subs)
            return Px.show()
        # For Slope along y direction
        elif dir == "y":
            Py = self._plot_slope('y', subs)
            return Py.show()
        # For Slope along z direction
        elif dir == "z":
            Pz = self._plot_slope('z', subs)
            return Pz.show()
        # For Slope along all direction
        else:
            Px = self._plot_slope('x', subs)
            Py = self._plot_slope('y', subs)
            Pz = self._plot_slope('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_deflection(self, dir, subs=None):

        deflection = self.deflection()

        if dir == 'x':
            dir_num = 0
            color = 'm'

        elif dir == 'y':
            dir_num = 1
            color = 'r'

        elif dir == 'z':
            dir_num = 2
            color = 'c'

        if subs is None:
            subs = {}

        for sym in deflection[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(deflection[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Deflection along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{\delta(%c)}$'%dir, line_color=color)

    def plot_deflection(self, dir="all", subs=None):

        """

        Returns a plot for Deflection along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which deflection plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_deflection()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/6400 - x**3/160 + 27*x**2/560 + 2*x/7 for x over (0.0, 20.0)


        """

        dir = dir.lower()
        # For deflection along x direction
        if dir == "x":
            Px = self._plot_deflection('x', subs)
            return Px.show()
        # For deflection along y direction
        elif dir == "y":
            Py = self._plot_deflection('y', subs)
            return Py.show()
        # For deflection along z direction
        elif dir == "z":
            Pz = self._plot_deflection('z', subs)
            return Pz.show()
        # For deflection along all direction
        else:
            Px = self._plot_deflection('x', subs)
            Py = self._plot_deflection('y', subs)
            Pz = self._plot_deflection('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def plot_loading_results(self, dir='x', subs=None):

        """

        Returns a subplot of Shear Force, Bending Moment,
        Slope and Deflection of the Beam object along the direction specified.

        Parameters
        ==========

        dir : string (default : "x")
               Direction along which plots are required.
               If no direction is specified, plots along x-axis are displayed.
        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> subs = {E:40, G:21, I:100, A:25}
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_loading_results('y',subs)
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[3]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        if subs is None:
            subs = {}

        ax1 = self._plot_shear_force(dir, subs)
        ax2 = self._plot_bending_moment(dir, subs)
        ax3 = self._plot_slope(dir, subs)
        ax4 = self._plot_deflection(dir, subs)

        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _plot_shear_stress(self, dir, subs=None):

        shear_stress = self.shear_stress()

        if dir == 'x':
            dir_num = 0
            color = 'r'

        elif dir == 'y':
            dir_num = 1
            color = 'g'

        elif dir == 'z':
            dir_num = 2
            color = 'b'

        if subs is None:
            subs = {}

        for sym in shear_stress[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(shear_stress[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Shear stress along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\tau(%c)$'%dir, line_color=color)

    def plot_shear_stress(self, dir="all", subs=None):

        """

        Returns a plot for Shear Stress along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear stress plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters and area of cross section 2 square
        meters. It is supported by rollers at both of its ends. A linear load having
        slope equal to 12 is applied along y-axis. A constant distributed load
        of magnitude 15 N is applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, 2, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_stress()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -3*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x/2 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For shear stress along x direction
        if dir == "x":
            Px = self._plot_shear_stress('x', subs)
            return Px.show()
        # For shear stress along y direction
        elif dir == "y":
            Py = self._plot_shear_stress('y', subs)
            return Py.show()
        # For shear stress along z direction
        elif dir == "z":
            Pz = self._plot_shear_stress('z', subs)
            return Pz.show()
        # For shear stress along all direction
        else:
            Px = self._plot_shear_stress('x', subs)
            Py = self._plot_shear_stress('y', subs)
            Pz = self._plot_shear_stress('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _max_shear_force(self, dir):
        """
        Helper function for max_shear_force().
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.shear_force()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        load_curve = Piecewise((float("nan"), self.variable<=0),
                (self._load_vector[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(load_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self.length)
        shear_curve = self.shear_force()[dir_num]
        shear_values = [shear_curve.subs(self.variable, x) for x in points]
        shear_values = list(map(abs, shear_values))

        max_shear = max(shear_values)
        return (points[shear_values.index(max_shear)], max_shear)

    def max_shear_force(self):
        """
        Returns point of max shear force and its corresponding shear value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_shear_force()
            [(0, 0), (20, 2400), (20, 300)]
        """

        max_shear = []
        max_shear.append(self._max_shear_force('x'))
        max_shear.append(self._max_shear_force('y'))
        max_shear.append(self._max_shear_force('z'))
        return max_shear

    def _max_bending_moment(self, dir):
        """
        Helper function for max_bending_moment().
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.bending_moment()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        shear_curve = Piecewise((float("nan"), self.variable<=0),
                (self.shear_force()[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(shear_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self.length)
        bending_moment_curve = self.bending_moment()[dir_num]
        bending_moments = [bending_moment_curve.subs(self.variable, x) for x in points]
        bending_moments = list(map(abs, bending_moments))

        max_bending_moment = max(bending_moments)
        return (points[bending_moments.index(max_bending_moment)], max_bending_moment)

    def max_bending_moment(self):
        """
        Returns point of max bending moment and its corresponding bending moment value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_bending_moment()
            [(0, 0), (20, 3000), (20, 16000)]
        """

        max_bmoment = []
        max_bmoment.append(self._max_bending_moment('x'))
        max_bmoment.append(self._max_bending_moment('y'))
        max_bmoment.append(self._max_bending_moment('z'))
        return max_bmoment

    max_bmoment = max_bending_moment

    def _max_deflection(self, dir):
        """
        Helper function for max_Deflection()
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.deflection()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        slope_curve = Piecewise((float("nan"), self.variable<=0),
                (self.slope()[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(slope_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self._length)
        deflection_curve = self.deflection()[dir_num]
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))

        max_def = max(deflections)
        return (points[deflections.index(max_def)], max_def)

    def max_deflection(self):
        """
        Returns point of max deflection and its corresponding deflection value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() and solve_slope_deflection() must be called
        before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.max_deflection()
            [(0, 0), (10, 495/14), (-10 + 10*sqrt(10793)/43, (10 - 10*sqrt(10793)/43)**3/160 - 20/7 + (10 - 10*sqrt(10793)/43)**4/6400 + 20*sqrt(10793)/301 + 27*(10 - 10*sqrt(10793)/43)**2/560)]
        """

        max_def = []
        max_def.append(self._max_deflection('x'))
        max_def.append(self._max_deflection('y'))
        max_def.append(self._max_deflection('z'))
        return max_def

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\stata.py ===
"""
Module contains tools for processing Stata files into DataFrames

The StataReader below was originally written by Joe Presbrey as part of PyDTA.
It has been extended and improved by Skipper Seabold from the Statsmodels
project who also developed the StataWriter and was finally added to pandas in
a once again improved version.

You can find more information on http://presbrey.mit.edu/PyDTA and
https://www.statsmodels.org/devel/
"""
from __future__ import annotations

from collections import abc
from datetime import (
    datetime,
    timedelta,
)
from io import BytesIO
import os
import struct
import sys
from typing import (
    IO,
    TYPE_CHECKING,
    AnyStr,
    Callable,
    Final,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
    CategoricalConversionWarning,
    InvalidColumnName,
    PossiblePrecisionLoss,
    ValueLabelTypeMismatch,
)
from pandas.util._decorators import (
    Appender,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    ensure_object,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas import (
    Categorical,
    DatetimeIndex,
    NaT,
    Timestamp,
    isna,
    to_datetime,
    to_timedelta,
)
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )
    from types import TracebackType
    from typing import Literal

    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
        Self,
        StorageOptions,
        WriteBuffer,
    )

_version_error = (
    "Version of given Stata file is {version}. pandas supports importing "
    "versions 105, 108, 111 (Stata 7SE), 113 (Stata 8/9), "
    "114 (Stata 10/11), 115 (Stata 12), 117 (Stata 13), 118 (Stata 14/15/16),"
    "and 119 (Stata 15/16, over 32,767 variables)."
)

_statafile_processing_params1 = """\
convert_dates : bool, default True
    Convert date variables to DataFrame time values.
convert_categoricals : bool, default True
    Read value labels and convert columns to Categorical/Factor variables."""

_statafile_processing_params2 = """\
index_col : str, optional
    Column to set as index.
convert_missing : bool, default False
    Flag indicating whether to convert missing values to their Stata
    representations.  If False, missing values are replaced with nan.
    If True, columns containing missing values are returned with
    object data types and missing values are represented by
    StataMissingValue objects.
preserve_dtypes : bool, default True
    Preserve Stata datatypes. If False, numeric data are upcast to pandas
    default types for foreign data (float64 or int64).
columns : list or None
    Columns to retain.  Columns will be returned in the given order.  None
    returns all columns.
order_categoricals : bool, default True
    Flag indicating whether converted categorical data are ordered."""

_chunksize_params = """\
chunksize : int, default None
    Return StataReader object for iterations, returns chunks with
    given number of lines."""

_iterator_params = """\
iterator : bool, default False
    Return StataReader object."""

_reader_notes = """\
Notes
-----
Categorical variables read through an iterator may not have the same
categories and dtype. This occurs when  a variable stored in a DTA
file is associated to an incomplete set of value labels that only
label a strict subset of the values."""

_read_stata_doc = f"""
Read Stata file into DataFrame.

Parameters
----------
filepath_or_buffer : str, path object or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, and file. For file URLs, a host is
    expected. A local file could be: ``file://localhost/path/to/table.dta``.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method,
    such as a file handle (e.g. via builtin ``open`` function)
    or ``StringIO``.
{_statafile_processing_params1}
{_statafile_processing_params2}
{_chunksize_params}
{_iterator_params}
{_shared_docs["decompression_options"] % "filepath_or_buffer"}
{_shared_docs["storage_options"]}

Returns
-------
DataFrame or pandas.api.typing.StataReader

See Also
--------
io.stata.StataReader : Low-level reader for Stata data files.
DataFrame.to_stata: Export Stata data files.

{_reader_notes}

Examples
--------

Creating a dummy stata for this example

>>> df = pd.DataFrame({{'animal': ['falcon', 'parrot', 'falcon', 'parrot'],
...                     'speed': [350, 18, 361, 15]}})  # doctest: +SKIP
>>> df.to_stata('animals.dta')  # doctest: +SKIP

Read a Stata dta file:

>>> df = pd.read_stata('animals.dta')  # doctest: +SKIP

Read a Stata dta file in 10,000 line chunks:

>>> values = np.random.randint(0, 10, size=(20_000, 1), dtype="uint8")  # doctest: +SKIP
>>> df = pd.DataFrame(values, columns=["i"])  # doctest: +SKIP
>>> df.to_stata('filename.dta')  # doctest: +SKIP

>>> with pd.read_stata('filename.dta', chunksize=10000) as itr: # doctest: +SKIP
>>>     for chunk in itr:
...         # Operate on a single chunk, e.g., chunk.mean()
...         pass  # doctest: +SKIP
"""

_read_method_doc = f"""\
Reads observations from Stata file, converting them into a dataframe

Parameters
----------
nrows : int
    Number of lines to read from data file, if None read whole file.
{_statafile_processing_params1}
{_statafile_processing_params2}

Returns
-------
DataFrame
"""

_stata_reader_doc = f"""\
Class for reading Stata dta files.

Parameters
----------
path_or_buf : path (string), buffer or path object
    string, path object (pathlib.Path or py._path.local.LocalPath) or object
    implementing a binary read() functions.
{_statafile_processing_params1}
{_statafile_processing_params2}
{_chunksize_params}
{_shared_docs["decompression_options"]}
{_shared_docs["storage_options"]}

{_reader_notes}
"""


_date_formats = ["%tc", "%tC", "%td", "%d", "%tw", "%tm", "%tq", "%th", "%ty"]


stata_epoch: Final = datetime(1960, 1, 1)


def _stata_elapsed_date_to_datetime_vec(dates: Series, fmt: str) -> Series:
    """
    Convert from SIF to datetime. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        The Stata Internal Format date to convert to datetime according to fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
        Returns

    Returns
    -------
    converted : Series
        The converted dates

    Examples
    --------
    >>> dates = pd.Series([52])
    >>> _stata_elapsed_date_to_datetime_vec(dates , "%tw")
    0   1961-01-01
    dtype: datetime64[ns]

    Notes
    -----
    datetime/c - tc
        milliseconds since 01jan1960 00:00:00.000, assuming 86,400 s/day
    datetime/C - tC - NOT IMPLEMENTED
        milliseconds since 01jan1960 00:00:00.000, adjusted for leap seconds
    date - td
        days since 01jan1960 (01jan1960 = 0)
    weekly date - tw
        weeks since 1960w1
        This assumes 52 weeks in a year, then adds 7 * remainder of the weeks.
        The datetime value is the start of the week in terms of days in the
        year, not ISO calendar weeks.
    monthly date - tm
        months since 1960m1
    quarterly date - tq
        quarters since 1960q1
    half-yearly date - th
        half-years since 1960h1 yearly
    date - ty
        years since 0000
    """
    MIN_YEAR, MAX_YEAR = Timestamp.min.year, Timestamp.max.year
    MAX_DAY_DELTA = (Timestamp.max - datetime(1960, 1, 1)).days
    MIN_DAY_DELTA = (Timestamp.min - datetime(1960, 1, 1)).days
    MIN_MS_DELTA = MIN_DAY_DELTA * 24 * 3600 * 1000
    MAX_MS_DELTA = MAX_DAY_DELTA * 24 * 3600 * 1000

    def convert_year_month_safe(year, month) -> Series:
        """
        Convert year and month to datetimes, using pandas vectorized versions
        when the date range falls within the range supported by pandas.
        Otherwise it falls back to a slower but more robust method
        using datetime.
        """
        if year.max() < MAX_YEAR and year.min() > MIN_YEAR:
            return to_datetime(100 * year + month, format="%Y%m")
        else:
            index = getattr(year, "index", None)
            return Series([datetime(y, m, 1) for y, m in zip(year, month)], index=index)

    def convert_year_days_safe(year, days) -> Series:
        """
        Converts year (e.g. 1999) and days since the start of the year to a
        datetime or datetime64 Series
        """
        if year.max() < (MAX_YEAR - 1) and year.min() > MIN_YEAR:
            return to_datetime(year, format="%Y") + to_timedelta(days, unit="d")
        else:
            index = getattr(year, "index", None)
            value = [
                datetime(y, 1, 1) + timedelta(days=int(d)) for y, d in zip(year, days)
            ]
            return Series(value, index=index)

    def convert_delta_safe(base, deltas, unit) -> Series:
        """
        Convert base dates and deltas to datetimes, using pandas vectorized
        versions if the deltas satisfy restrictions required to be expressed
        as dates in pandas.
        """
        index = getattr(deltas, "index", None)
        if unit == "d":
            if deltas.max() > MAX_DAY_DELTA or deltas.min() < MIN_DAY_DELTA:
                values = [base + timedelta(days=int(d)) for d in deltas]
                return Series(values, index=index)
        elif unit == "ms":
            if deltas.max() > MAX_MS_DELTA or deltas.min() < MIN_MS_DELTA:
                values = [
                    base + timedelta(microseconds=(int(d) * 1000)) for d in deltas
                ]
                return Series(values, index=index)
        else:
            raise ValueError("format not understood")
        base = to_datetime(base)
        deltas = to_timedelta(deltas, unit=unit)
        return base + deltas

    # TODO(non-nano): If/when pandas supports more than datetime64[ns], this
    #  should be improved to use correct range, e.g. datetime[Y] for yearly
    bad_locs = np.isnan(dates)
    has_bad_values = False
    if bad_locs.any():
        has_bad_values = True
        dates._values[bad_locs] = 1.0  # Replace with NaT
    dates = dates.astype(np.int64)

    if fmt.startswith(("%tc", "tc")):  # Delta ms relative to base
        base = stata_epoch
        ms = dates
        conv_dates = convert_delta_safe(base, ms, "ms")
    elif fmt.startswith(("%tC", "tC")):
        warnings.warn(
            "Encountered %tC format. Leaving in Stata Internal Format.",
            stacklevel=find_stack_level(),
        )
        conv_dates = Series(dates, dtype=object)
        if has_bad_values:
            conv_dates[bad_locs] = NaT
        return conv_dates
    # Delta days relative to base
    elif fmt.startswith(("%td", "td", "%d", "d")):
        base = stata_epoch
        days = dates
        conv_dates = convert_delta_safe(base, days, "d")
    # does not count leap days - 7 days is a week.
    # 52nd week may have more than 7 days
    elif fmt.startswith(("%tw", "tw")):
        year = stata_epoch.year + dates // 52
        days = (dates % 52) * 7
        conv_dates = convert_year_days_safe(year, days)
    elif fmt.startswith(("%tm", "tm")):  # Delta months relative to base
        year = stata_epoch.year + dates // 12
        month = (dates % 12) + 1
        conv_dates = convert_year_month_safe(year, month)
    elif fmt.startswith(("%tq", "tq")):  # Delta quarters relative to base
        year = stata_epoch.year + dates // 4
        quarter_month = (dates % 4) * 3 + 1
        conv_dates = convert_year_month_safe(year, quarter_month)
    elif fmt.startswith(("%th", "th")):  # Delta half-years relative to base
        year = stata_epoch.year + dates // 2
        month = (dates % 2) * 6 + 1
        conv_dates = convert_year_month_safe(year, month)
    elif fmt.startswith(("%ty", "ty")):  # Years -- not delta
        year = dates
        first_month = np.ones_like(dates)
        conv_dates = convert_year_month_safe(year, first_month)
    else:
        raise ValueError(f"Date fmt {fmt} not understood")

    if has_bad_values:  # Restore NaT for bad values
        conv_dates[bad_locs] = NaT

    return conv_dates


def _datetime_to_stata_elapsed_vec(dates: Series, fmt: str) -> Series:
    """
    Convert from datetime to SIF. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        Series or array containing datetime or datetime64[ns] to
        convert to the Stata Internal Format given by fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
    """
    index = dates.index
    NS_PER_DAY = 24 * 3600 * 1000 * 1000 * 1000
    US_PER_DAY = NS_PER_DAY / 1000

    def parse_dates_safe(
        dates: Series, delta: bool = False, year: bool = False, days: bool = False
    ):
        d = {}
        if lib.is_np_dtype(dates.dtype, "M"):
            if delta:
                time_delta = dates - Timestamp(stata_epoch).as_unit("ns")
                d["delta"] = time_delta._values.view(np.int64) // 1000  # microseconds
            if days or year:
                date_index = DatetimeIndex(dates)
                d["year"] = date_index._data.year
                d["month"] = date_index._data.month
            if days:
                days_in_ns = dates._values.view(np.int64) - to_datetime(
                    d["year"], format="%Y"
                )._values.view(np.int64)
                d["days"] = days_in_ns // NS_PER_DAY

        elif infer_dtype(dates, skipna=False) == "datetime":
            if delta:
                delta = dates._values - stata_epoch

                def f(x: timedelta) -> float:
                    return US_PER_DAY * x.days + 1000000 * x.seconds + x.microseconds

                v = np.vectorize(f)
                d["delta"] = v(delta)
            if year:
                year_month = dates.apply(lambda x: 100 * x.year + x.month)
                d["year"] = year_month._values // 100
                d["month"] = year_month._values - d["year"] * 100
            if days:

                def g(x: datetime) -> int:
                    return (x - datetime(x.year, 1, 1)).days

                v = np.vectorize(g)
                d["days"] = v(dates)
        else:
            raise ValueError(
                "Columns containing dates must contain either "
                "datetime64, datetime or null values."
            )

        return DataFrame(d, index=index)

    bad_loc = isna(dates)
    index = dates.index
    if bad_loc.any():
        if lib.is_np_dtype(dates.dtype, "M"):
            dates._values[bad_loc] = to_datetime(stata_epoch)
        else:
            dates._values[bad_loc] = stata_epoch

    if fmt in ["%tc", "tc"]:
        d = parse_dates_safe(dates, delta=True)
        conv_dates = d.delta / 1000
    elif fmt in ["%tC", "tC"]:
        warnings.warn(
            "Stata Internal Format tC not supported.",
            stacklevel=find_stack_level(),
        )
        conv_dates = dates
    elif fmt in ["%td", "td"]:
        d = parse_dates_safe(dates, delta=True)
        conv_dates = d.delta // US_PER_DAY
    elif fmt in ["%tw", "tw"]:
        d = parse_dates_safe(dates, year=True, days=True)
        conv_dates = 52 * (d.year - stata_epoch.year) + d.days // 7
    elif fmt in ["%tm", "tm"]:
        d = parse_dates_safe(dates, year=True)
        conv_dates = 12 * (d.year - stata_epoch.year) + d.month - 1
    elif fmt in ["%tq", "tq"]:
        d = parse_dates_safe(dates, year=True)
        conv_dates = 4 * (d.year - stata_epoch.year) + (d.month - 1) // 3
    elif fmt in ["%th", "th"]:
        d = parse_dates_safe(dates, year=True)
        conv_dates = 2 * (d.year - stata_epoch.year) + (d.month > 6).astype(int)
    elif fmt in ["%ty", "ty"]:
        d = parse_dates_safe(dates, year=True)
        conv_dates = d.year
    else:
        raise ValueError(f"Format {fmt} is not a known Stata date format")

    conv_dates = Series(conv_dates, dtype=np.float64, copy=False)
    missing_value = struct.unpack("<d", b"\x00\x00\x00\x00\x00\x00\xe0\x7f")[0]
    conv_dates[bad_loc] = missing_value

    return Series(conv_dates, index=index, copy=False)


excessive_string_length_error: Final = """
Fixed width strings in Stata .dta files are limited to 244 (or fewer)
characters.  Column '{0}' does not satisfy this restriction. Use the
'version=117' parameter to write the newer (Stata 13 and later) format.
"""


precision_loss_doc: Final = """
Column converted from {0} to {1}, and some data are outside of the lossless
conversion range. This may result in a loss of precision in the saved data.
"""


value_label_mismatch_doc: Final = """
Stata value labels (pandas categories) must be strings. Column {0} contains
non-string labels which will be converted to strings.  Please check that the
Stata data file created has not lost information due to duplicate labels.
"""


invalid_name_doc: Final = """
Not all pandas column names were valid Stata variable names.
The following replacements have been made:

    {0}

If this is not what you expect, please make sure you have Stata-compliant
column names in your DataFrame (strings only, max 32 characters, only
alphanumerics and underscores, no Stata reserved words)
"""


categorical_conversion_warning: Final = """
One or more series with value labels are not fully labeled. Reading this
dataset with an iterator results in categorical variable with different
categories. This occurs since it is not possible to know all possible values
until the entire dataset has been read. To avoid this warning, you can either
read dataset without an iterator, or manually convert categorical data by
``convert_categoricals`` to False and then accessing the variable labels
through the value_labels method of the reader.
"""


def _cast_to_stata_types(data: DataFrame) -> DataFrame:
    """
    Checks the dtypes of the columns of a pandas DataFrame for
    compatibility with the data types and ranges supported by Stata, and
    converts if necessary.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to check and convert

    Notes
    -----
    Numeric columns in Stata must be one of int8, int16, int32, float32 or
    float64, with some additional value restrictions.  int8 and int16 columns
    are checked for violations of the value restrictions and upcast if needed.
    int64 data is not usable in Stata, and so it is downcast to int32 whenever
    the value are in the int32 range, and sidecast to float64 when larger than
    this range.  If the int64 values are outside of the range of those
    perfectly representable as float64 values, a warning is raised.

    bool columns are cast to int8.  uint columns are converted to int of the
    same size if there is no loss in precision, otherwise are upcast to a
    larger type.  uint64 is currently not supported since it is concerted to
    object in a DataFrame.
    """
    ws = ""
    # original, if small, if large
    conversion_data: tuple[
        tuple[type, type, type],
        tuple[type, type, type],
        tuple[type, type, type],
        tuple[type, type, type],
        tuple[type, type, type],
    ] = (
        (np.bool_, np.int8, np.int8),
        (np.uint8, np.int8, np.int16),
        (np.uint16, np.int16, np.int32),
        (np.uint32, np.int32, np.int64),
        (np.uint64, np.int64, np.float64),
    )

    float32_max = struct.unpack("<f", b"\xff\xff\xff\x7e")[0]
    float64_max = struct.unpack("<d", b"\xff\xff\xff\xff\xff\xff\xdf\x7f")[0]

    for col in data:
        # Cast from unsupported types to supported types
        is_nullable_int = (
            isinstance(data[col].dtype, ExtensionDtype)
            and data[col].dtype.kind in "iub"
        )
        # We need to find orig_missing before altering data below
        orig_missing = data[col].isna()
        if is_nullable_int:
            fv = 0 if data[col].dtype.kind in "iu" else False
            # Replace with NumPy-compatible column
            data[col] = data[col].fillna(fv).astype(data[col].dtype.numpy_dtype)
        elif isinstance(data[col].dtype, ExtensionDtype):
            if getattr(data[col].dtype, "numpy_dtype", None) is not None:
                data[col] = data[col].astype(data[col].dtype.numpy_dtype)
            elif is_string_dtype(data[col].dtype):
                # TODO could avoid converting string dtype to object here,
                # but handle string dtype in _encode_strings
                data[col] = data[col].astype("object")
                # generate_table checks for None values
                data.loc[data[col].isna(), col] = None

        dtype = data[col].dtype
        empty_df = data.shape[0] == 0
        for c_data in conversion_data:
            if dtype == c_data[0]:
                if empty_df or data[col].max() <= np.iinfo(c_data[1]).max:
                    dtype = c_data[1]
                else:
                    dtype = c_data[2]
                if c_data[2] == np.int64:  # Warn if necessary
                    if data[col].max() >= 2**53:
                        ws = precision_loss_doc.format("uint64", "float64")

                data[col] = data[col].astype(dtype)

        # Check values and upcast if necessary

        if dtype == np.int8 and not empty_df:
            if data[col].max() > 100 or data[col].min() < -127:
                data[col] = data[col].astype(np.int16)
        elif dtype == np.int16 and not empty_df:
            if data[col].max() > 32740 or data[col].min() < -32767:
                data[col] = data[col].astype(np.int32)
        elif dtype == np.int64:
            if empty_df or (
                data[col].max() <= 2147483620 and data[col].min() >= -2147483647
            ):
                data[col] = data[col].astype(np.int32)
            else:
                data[col] = data[col].astype(np.float64)
                if data[col].max() >= 2**53 or data[col].min() <= -(2**53):
                    ws = precision_loss_doc.format("int64", "float64")
        elif dtype in (np.float32, np.float64):
            if np.isinf(data[col]).any():
                raise ValueError(
                    f"Column {col} contains infinity or -infinity"
                    "which is outside the range supported by Stata."
                )
            value = data[col].max()
            if dtype == np.float32 and value > float32_max:
                data[col] = data[col].astype(np.float64)
            elif dtype == np.float64:
                if value > float64_max:
                    raise ValueError(
                        f"Column {col} has a maximum value ({value}) outside the range "
                        f"supported by Stata ({float64_max})"
                    )
        if is_nullable_int:
            if orig_missing.any():
                # Replace missing by Stata sentinel value
                sentinel = StataMissingValue.BASE_MISSING_VALUES[data[col].dtype.name]
                data.loc[orig_missing, col] = sentinel
    if ws:
        warnings.warn(
            ws,
            PossiblePrecisionLoss,
            stacklevel=find_stack_level(),
        )

    return data


class StataValueLabel:
    """
    Parse a categorical column and prepare formatted output

    Parameters
    ----------
    catarray : Series
        Categorical Series to encode
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    """

    def __init__(
        self, catarray: Series, encoding: Literal["latin-1", "utf-8"] = "latin-1"
    ) -> None:
        if encoding not in ("latin-1", "utf-8"):
            raise ValueError("Only latin-1 and utf-8 are supported.")
        self.labname = catarray.name
        self._encoding = encoding
        categories = catarray.cat.categories
        self.value_labels = enumerate(categories)

        self._prepare_value_labels()

    def _prepare_value_labels(self) -> None:
        """Encode value labels."""

        self.text_len = 0
        self.txt: list[bytes] = []
        self.n = 0
        # Offsets (length of categories), converted to int32
        self.off = np.array([], dtype=np.int32)
        # Values, converted to int32
        self.val = np.array([], dtype=np.int32)
        self.len = 0

        # Compute lengths and setup lists of offsets and labels
        offsets: list[int] = []
        values: list[float] = []
        for vl in self.value_labels:
            category: str | bytes = vl[1]
            if not isinstance(category, str):
                category = str(category)
                warnings.warn(
                    value_label_mismatch_doc.format(self.labname),
                    ValueLabelTypeMismatch,
                    stacklevel=find_stack_level(),
                )
            category = category.encode(self._encoding)
            offsets.append(self.text_len)
            self.text_len += len(category) + 1  # +1 for the padding
            values.append(vl[0])
            self.txt.append(category)
            self.n += 1

        if self.text_len > 32000:
            raise ValueError(
                "Stata value labels for a single variable must "
                "have a combined length less than 32,000 characters."
            )

        # Ensure int32
        self.off = np.array(offsets, dtype=np.int32)
        self.val = np.array(values, dtype=np.int32)

        # Total length
        self.len = 4 + 4 + 4 * self.n + 4 * self.n + self.text_len

    def generate_value_label(self, byteorder: str) -> bytes:
        """
        Generate the binary representation of the value labels.

        Parameters
        ----------
        byteorder : str
            Byte order of the output

        Returns
        -------
        value_label : bytes
            Bytes containing the formatted value label
        """
        encoding = self._encoding
        bio = BytesIO()
        null_byte = b"\x00"

        # len
        bio.write(struct.pack(byteorder + "i", self.len))

        # labname
        labname = str(self.labname)[:32].encode(encoding)
        lab_len = 32 if encoding not in ("utf-8", "utf8") else 128
        labname = _pad_bytes(labname, lab_len + 1)
        bio.write(labname)

        # padding - 3 bytes
        for i in range(3):
            bio.write(struct.pack("c", null_byte))

        # value_label_table
        # n - int32
        bio.write(struct.pack(byteorder + "i", self.n))

        # textlen  - int32
        bio.write(struct.pack(byteorder + "i", self.text_len))

        # off - int32 array (n elements)
        for offset in self.off:
            bio.write(struct.pack(byteorder + "i", offset))

        # val - int32 array (n elements)
        for value in self.val:
            bio.write(struct.pack(byteorder + "i", value))

        # txt - Text labels, null terminated
        for text in self.txt:
            bio.write(text + null_byte)

        return bio.getvalue()


class StataNonCatValueLabel(StataValueLabel):
    """
    Prepare formatted version of value labels

    Parameters
    ----------
    labname : str
        Value label name
    value_labels: Dictionary
        Mapping of values to labels
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    """

    def __init__(
        self,
        labname: str,
        value_labels: dict[float, str],
        encoding: Literal["latin-1", "utf-8"] = "latin-1",
    ) -> None:
        if encoding not in ("latin-1", "utf-8"):
            raise ValueError("Only latin-1 and utf-8 are supported.")

        self.labname = labname
        self._encoding = encoding
        self.value_labels = sorted(  # type: ignore[assignment]
            value_labels.items(), key=lambda x: x[0]
        )
        self._prepare_value_labels()


class StataMissingValue:
    """
    An observation's missing value.

    Parameters
    ----------
    value : {int, float}
        The Stata missing value code

    Notes
    -----
    More information: <https://www.stata.com/help.cgi?missing>

    Integer missing values make the code '.', '.a', ..., '.z' to the ranges
    101 ... 127 (for int8), 32741 ... 32767  (for int16) and 2147483621 ...
    2147483647 (for int32).  Missing values for floating point data types are
    more complex but the pattern is simple to discern from the following table.

    np.float32 missing values (float in Stata)
    0000007f    .
    0008007f    .a
    0010007f    .b
    ...
    00c0007f    .x
    00c8007f    .y
    00d0007f    .z

    np.float64 missing values (double in Stata)
    000000000000e07f    .
    000000000001e07f    .a
    000000000002e07f    .b
    ...
    000000000018e07f    .x
    000000000019e07f    .y
    00000000001ae07f    .z
    """

    # Construct a dictionary of missing values
    MISSING_VALUES: dict[float, str] = {}
    bases: Final = (101, 32741, 2147483621)
    for b in bases:
        # Conversion to long to avoid hash issues on 32 bit platforms #8968
        MISSING_VALUES[b] = "."
        for i in range(1, 27):
            MISSING_VALUES[i + b] = "." + chr(96 + i)

    float32_base: bytes = b"\x00\x00\x00\x7f"
    increment_32: int = struct.unpack("<i", b"\x00\x08\x00\x00")[0]
    for i in range(27):
        key = struct.unpack("<f", float32_base)[0]
        MISSING_VALUES[key] = "."
        if i > 0:
            MISSING_VALUES[key] += chr(96 + i)
        int_value = struct.unpack("<i", struct.pack("<f", key))[0] + increment_32
        float32_base = struct.pack("<i", int_value)

    float64_base: bytes = b"\x00\x00\x00\x00\x00\x00\xe0\x7f"
    increment_64 = struct.unpack("q", b"\x00\x00\x00\x00\x00\x01\x00\x00")[0]
    for i in range(27):
        key = struct.unpack("<d", float64_base)[0]
        MISSING_VALUES[key] = "."
        if i > 0:
            MISSING_VALUES[key] += chr(96 + i)
        int_value = struct.unpack("q", struct.pack("<d", key))[0] + increment_64
        float64_base = struct.pack("q", int_value)

    BASE_MISSING_VALUES: Final = {
        "int8": 101,
        "int16": 32741,
        "int32": 2147483621,
        "float32": struct.unpack("<f", float32_base)[0],
        "float64": struct.unpack("<d", float64_base)[0],
    }

    def __init__(self, value: float) -> None:
        self._value = value
        # Conversion to int to avoid hash issues on 32 bit platforms #8968
        value = int(value) if value < 2147483648 else float(value)
        self._str = self.MISSING_VALUES[value]

    @property
    def string(self) -> str:
        """
        The Stata representation of the missing value: '.', '.a'..'.z'

        Returns
        -------
        str
            The representation of the missing value.
        """
        return self._str

    @property
    def value(self) -> float:
        """
        The binary representation of the missing value.

        Returns
        -------
        {int, float}
            The binary representation of the missing value.
        """
        return self._value

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f"{type(self)}({self})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.string == other.string
            and self.value == other.value
        )

    @classmethod
    def get_base_missing_value(cls, dtype: np.dtype) -> float:
        if dtype.type is np.int8:
            value = cls.BASE_MISSING_VALUES["int8"]
        elif dtype.type is np.int16:
            value = cls.BASE_MISSING_VALUES["int16"]
        elif dtype.type is np.int32:
            value = cls.BASE_MISSING_VALUES["int32"]
        elif dtype.type is np.float32:
            value = cls.BASE_MISSING_VALUES["float32"]
        elif dtype.type is np.float64:
            value = cls.BASE_MISSING_VALUES["float64"]
        else:
            raise ValueError("Unsupported dtype")
        return value


class StataParser:
    def __init__(self) -> None:
        # type          code.
        # --------------------
        # str1        1 = 0x01
        # str2        2 = 0x02
        # ...
        # str244    244 = 0xf4
        # byte      251 = 0xfb  (sic)
        # int       252 = 0xfc
        # long      253 = 0xfd
        # float     254 = 0xfe
        # double    255 = 0xff
        # --------------------
        # NOTE: the byte type seems to be reserved for categorical variables
        # with a label, but the underlying variable is -127 to 100
        # we're going to drop the label and cast to int
        self.DTYPE_MAP = dict(
            [(i, np.dtype(f"S{i}")) for i in range(1, 245)]
            + [
                (251, np.dtype(np.int8)),
                (252, np.dtype(np.int16)),
                (253, np.dtype(np.int32)),
                (254, np.dtype(np.float32)),
                (255, np.dtype(np.float64)),
            ]
        )
        self.DTYPE_MAP_XML: dict[int, np.dtype] = {
            32768: np.dtype(np.uint8),  # Keys to GSO
            65526: np.dtype(np.float64),
            65527: np.dtype(np.float32),
            65528: np.dtype(np.int32),
            65529: np.dtype(np.int16),
            65530: np.dtype(np.int8),
        }
        self.TYPE_MAP = list(tuple(range(251)) + tuple("bhlfd"))
        self.TYPE_MAP_XML = {
            # Not really a Q, unclear how to handle byteswap
            32768: "Q",
            65526: "d",
            65527: "f",
            65528: "l",
            65529: "h",
            65530: "b",
        }
        # NOTE: technically, some of these are wrong. there are more numbers
        # that can be represented. it's the 27 ABOVE and BELOW the max listed
        # numeric data type in [U] 12.2.2 of the 11.2 manual
        float32_min = b"\xff\xff\xff\xfe"
        float32_max = b"\xff\xff\xff\x7e"
        float64_min = b"\xff\xff\xff\xff\xff\xff\xef\xff"
        float64_max = b"\xff\xff\xff\xff\xff\xff\xdf\x7f"
        self.VALID_RANGE = {
            "b": (-127, 100),
            "h": (-32767, 32740),
            "l": (-2147483647, 2147483620),
            "f": (
                np.float32(struct.unpack("<f", float32_min)[0]),
                np.float32(struct.unpack("<f", float32_max)[0]),
            ),
            "d": (
                np.float64(struct.unpack("<d", float64_min)[0]),
                np.float64(struct.unpack("<d", float64_max)[0]),
            ),
        }

        self.OLD_TYPE_MAPPING = {
            98: 251,  # byte
            105: 252,  # int
            108: 253,  # long
            102: 254,  # float
            100: 255,  # double
        }

        # These missing values are the generic '.' in Stata, and are used
        # to replace nans
        self.MISSING_VALUES = {
            "b": 101,
            "h": 32741,
            "l": 2147483621,
            "f": np.float32(struct.unpack("<f", b"\x00\x00\x00\x7f")[0]),
            "d": np.float64(
                struct.unpack("<d", b"\x00\x00\x00\x00\x00\x00\xe0\x7f")[0]
            ),
        }
        self.NUMPY_TYPE_MAP = {
            "b": "i1",
            "h": "i2",
            "l": "i4",
            "f": "f4",
            "d": "f8",
            "Q": "u8",
        }

        # Reserved words cannot be used as variable names
        self.RESERVED_WORDS = {
            "aggregate",
            "array",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "class",
            "colvector",
            "complex",
            "const",
            "continue",
            "default",
            "delegate",
            "delete",
            "do",
            "double",
            "else",
            "eltypedef",
            "end",
            "enum",
            "explicit",
            "export",
            "external",
            "float",
            "for",
            "friend",
            "function",
            "global",
            "goto",
            "if",
            "inline",
            "int",
            "local",
            "long",
            "NULL",
            "pragma",
            "protected",
            "quad",
            "rowvector",
            "short",
            "typedef",
            "typename",
            "virtual",
            "_all",
            "_N",
            "_skip",
            "_b",
            "_pi",
            "str#",
            "in",
            "_pred",
            "strL",
            "_coef",
            "_rc",
            "using",
            "_cons",
            "_se",
            "with",
            "_n",
        }


class StataReader(StataParser, abc.Iterator):
    __doc__ = _stata_reader_doc

    _path_or_buf: IO[bytes]  # Will be assigned by `_open_file`.

    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        convert_dates: bool = True,
        convert_categoricals: bool = True,
        index_col: str | None = None,
        convert_missing: bool = False,
        preserve_dtypes: bool = True,
        columns: Sequence[str] | None = None,
        order_categoricals: bool = True,
        chunksize: int | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
    ) -> None:
        super().__init__()

        # Arguments to the reader (can be temporarily overridden in
        # calls to read).
        self._convert_dates = convert_dates
        self._convert_categoricals = convert_categoricals
        self._index_col = index_col
        self._convert_missing = convert_missing
        self._preserve_dtypes = preserve_dtypes
        self._columns = columns
        self._order_categoricals = order_categoricals
        self._original_path_or_buf = path_or_buf
        self._compression = compression
        self._storage_options = storage_options
        self._encoding = ""
        self._chunksize = chunksize
        self._using_iterator = False
        self._entered = False
        if self._chunksize is None:
            self._chunksize = 1
        elif not isinstance(chunksize, int) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer when set.")

        # State variables for the file
        self._close_file: Callable[[], None] | None = None
        self._missing_values = False
        self._can_read_value_labels = False
        self._column_selector_set = False
        self._value_labels_read = False
        self._data_read = False
        self._dtype: np.dtype | None = None
        self._lines_read = 0

        self._native_byteorder = _set_endianness(sys.byteorder)

    def _ensure_open(self) -> None:
        """
        Ensure the file has been opened and its header data read.
        """
        if not hasattr(self, "_path_or_buf"):
            self._open_file()

    def _open_file(self) -> None:
        """
        Open the file (with compression options, etc.), and read header information.
        """
        if not self._entered:
            warnings.warn(
                "StataReader is being used without using a context manager. "
                "Using StataReader as a context manager is the only supported method.",
                ResourceWarning,
                stacklevel=find_stack_level(),
            )
        handles = get_handle(
            self._original_path_or_buf,
            "rb",
            storage_options=self._storage_options,
            is_text=False,
            compression=self._compression,
        )
        if hasattr(handles.handle, "seekable") and handles.handle.seekable():
            # If the handle is directly seekable, use it without an extra copy.
            self._path_or_buf = handles.handle
            self._close_file = handles.close
        else:
            # Copy to memory, and ensure no encoding.
            with handles:
                self._path_or_buf = BytesIO(handles.handle.read())
            self._close_file = self._path_or_buf.close

        self._read_header()
        self._setup_dtype()

    def __enter__(self) -> Self:
        """enter context manager"""
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._close_file:
            self._close_file()

    def close(self) -> None:
        """Close the handle if its open.

        .. deprecated: 2.0.0

           The close method is not part of the public API.
           The only supported way to use StataReader is to use it as a context manager.
        """
        warnings.warn(
            "The StataReader.close() method is not part of the public API and "
            "will be removed in a future version without notice. "
            "Using StataReader as a context manager is the only supported method.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        if self._close_file:
            self._close_file()

    def _set_encoding(self) -> None:
        """
        Set string encoding which depends on file version
        """
        if self._format_version < 118:
            self._encoding = "latin-1"
        else:
            self._encoding = "utf-8"

    def _read_int8(self) -> int:
        return struct.unpack("b", self._path_or_buf.read(1))[0]

    def _read_uint8(self) -> int:
        return struct.unpack("B", self._path_or_buf.read(1))[0]

    def _read_uint16(self) -> int:
        return struct.unpack(f"{self._byteorder}H", self._path_or_buf.read(2))[0]

    def _read_uint32(self) -> int:
        return struct.unpack(f"{self._byteorder}I", self._path_or_buf.read(4))[0]

    def _read_uint64(self) -> int:
        return struct.unpack(f"{self._byteorder}Q", self._path_or_buf.read(8))[0]

    def _read_int16(self) -> int:
        return struct.unpack(f"{self._byteorder}h", self._path_or_buf.read(2))[0]

    def _read_int32(self) -> int:
        return struct.unpack(f"{self._byteorder}i", self._path_or_buf.read(4))[0]

    def _read_int64(self) -> int:
        return struct.unpack(f"{self._byteorder}q", self._path_or_buf.read(8))[0]

    def _read_char8(self) -> bytes:
        return struct.unpack("c", self._path_or_buf.read(1))[0]

    def _read_int16_count(self, count: int) -> tuple[int, ...]:
        return struct.unpack(
            f"{self._byteorder}{'h' * count}",
            self._path_or_buf.read(2 * count),
        )

    def _read_header(self) -> None:
        first_char = self._read_char8()
        if first_char == b"<":
            self._read_new_header()
        else:
            self._read_old_header(first_char)

    def _read_new_header(self) -> None:
        # The first part of the header is common to 117 - 119.
        self._path_or_buf.read(27)  # stata_dta><header><release>
        self._format_version = int(self._path_or_buf.read(3))
        if self._format_version not in [117, 118, 119]:
            raise ValueError(_version_error.format(version=self._format_version))
        self._set_encoding()
        self._path_or_buf.read(21)  # </release><byteorder>
        self._byteorder = ">" if self._path_or_buf.read(3) == b"MSF" else "<"
        self._path_or_buf.read(15)  # </byteorder><K>
        self._nvar = (
            self._read_uint16() if self._format_version <= 118 else self._read_uint32()
        )
        self._path_or_buf.read(7)  # </K><N>

        self._nobs = self._get_nobs()
        self._path_or_buf.read(11)  # </N><label>
        self._data_label = self._get_data_label()
        self._path_or_buf.read(19)  # </label><timestamp>
        self._time_stamp = self._get_time_stamp()
        self._path_or_buf.read(26)  # </timestamp></header><map>
        self._path_or_buf.read(8)  # 0x0000000000000000
        self._path_or_buf.read(8)  # position of <map>

        self._seek_vartypes = self._read_int64() + 16
        self._seek_varnames = self._read_int64() + 10
        self._seek_sortlist = self._read_int64() + 10
        self._seek_formats = self._read_int64() + 9
        self._seek_value_label_names = self._read_int64() + 19

        # Requires version-specific treatment
        self._seek_variable_labels = self._get_seek_variable_labels()

        self._path_or_buf.read(8)  # <characteristics>
        self._data_location = self._read_int64() + 6
        self._seek_strls = self._read_int64() + 7
        self._seek_value_labels = self._read_int64() + 14

        self._typlist, self._dtyplist = self._get_dtypes(self._seek_vartypes)

        self._path_or_buf.seek(self._seek_varnames)
        self._varlist = self._get_varlist()

        self._path_or_buf.seek(self._seek_sortlist)
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]

        self._path_or_buf.seek(self._seek_formats)
        self._fmtlist = self._get_fmtlist()

        self._path_or_buf.seek(self._seek_value_label_names)
        self._lbllist = self._get_lbllist()

        self._path_or_buf.seek(self._seek_variable_labels)
        self._variable_labels = self._get_variable_labels()

    # Get data type information, works for versions 117-119.
    def _get_dtypes(
        self, seek_vartypes: int
    ) -> tuple[list[int | str], list[str | np.dtype]]:
        self._path_or_buf.seek(seek_vartypes)
        typlist = []
        dtyplist = []
        for _ in range(self._nvar):
            typ = self._read_uint16()
            if typ <= 2045:
                typlist.append(typ)
                dtyplist.append(str(typ))
            else:
                try:
                    typlist.append(self.TYPE_MAP_XML[typ])  # type: ignore[arg-type]
                    dtyplist.append(self.DTYPE_MAP_XML[typ])  # type: ignore[arg-type]
                except KeyError as err:
                    raise ValueError(f"cannot convert stata types [{typ}]") from err

        return typlist, dtyplist  # type: ignore[return-value]

    def _get_varlist(self) -> list[str]:
        # 33 in order formats, 129 in formats 118 and 119
        b = 33 if self._format_version < 118 else 129
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    # Returns the format list
    def _get_fmtlist(self) -> list[str]:
        if self._format_version >= 118:
            b = 57
        elif self._format_version > 113:
            b = 49
        elif self._format_version > 104:
            b = 12
        else:
            b = 7

        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    # Returns the label list
    def _get_lbllist(self) -> list[str]:
        if self._format_version >= 118:
            b = 129
        elif self._format_version > 108:
            b = 33
        else:
            b = 9
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    def _get_variable_labels(self) -> list[str]:
        if self._format_version >= 118:
            vlblist = [
                self._decode(self._path_or_buf.read(321)) for _ in range(self._nvar)
            ]
        elif self._format_version > 105:
            vlblist = [
                self._decode(self._path_or_buf.read(81)) for _ in range(self._nvar)
            ]
        else:
            vlblist = [
                self._decode(self._path_or_buf.read(32)) for _ in range(self._nvar)
            ]
        return vlblist

    def _get_nobs(self) -> int:
        if self._format_version >= 118:
            return self._read_uint64()
        else:
            return self._read_uint32()

    def _get_data_label(self) -> str:
        if self._format_version >= 118:
            strlen = self._read_uint16()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version == 117:
            strlen = self._read_int8()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version > 105:
            return self._decode(self._path_or_buf.read(81))
        else:
            return self._decode(self._path_or_buf.read(32))

    def _get_time_stamp(self) -> str:
        if self._format_version >= 118:
            strlen = self._read_int8()
            return self._path_or_buf.read(strlen).decode("utf-8")
        elif self._format_version == 117:
            strlen = self._read_int8()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version > 104:
            return self._decode(self._path_or_buf.read(18))
        else:
            raise ValueError()

    def _get_seek_variable_labels(self) -> int:
        if self._format_version == 117:
            self._path_or_buf.read(8)  # <variable_labels>, throw away
            # Stata 117 data files do not follow the described format.  This is
            # a work around that uses the previous label, 33 bytes for each
            # variable, 20 for the closing tag and 17 for the opening tag
            return self._seek_value_label_names + (33 * self._nvar) + 20 + 17
        elif self._format_version >= 118:
            return self._read_int64() + 17
        else:
            raise ValueError()

    def _read_old_header(self, first_char: bytes) -> None:
        self._format_version = int(first_char[0])
        if self._format_version not in [104, 105, 108, 111, 113, 114, 115]:
            raise ValueError(_version_error.format(version=self._format_version))
        self._set_encoding()
        self._byteorder = ">" if self._read_int8() == 0x1 else "<"
        self._filetype = self._read_int8()
        self._path_or_buf.read(1)  # unused

        self._nvar = self._read_uint16()
        self._nobs = self._get_nobs()

        self._data_label = self._get_data_label()

        self._time_stamp = self._get_time_stamp()

        # descriptors
        if self._format_version > 108:
            typlist = [int(c) for c in self._path_or_buf.read(self._nvar)]
        else:
            buf = self._path_or_buf.read(self._nvar)
            typlistb = np.frombuffer(buf, dtype=np.uint8)
            typlist = []
            for tp in typlistb:
                if tp in self.OLD_TYPE_MAPPING:
                    typlist.append(self.OLD_TYPE_MAPPING[tp])
                else:
                    typlist.append(tp - 127)  # bytes

        try:
            self._typlist = [self.TYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_types = ",".join([str(x) for x in typlist])
            raise ValueError(f"cannot convert stata types [{invalid_types}]") from err
        try:
            self._dtyplist = [self.DTYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_dtypes = ",".join([str(x) for x in typlist])
            raise ValueError(f"cannot convert stata dtypes [{invalid_dtypes}]") from err

        if self._format_version > 108:
            self._varlist = [
                self._decode(self._path_or_buf.read(33)) for _ in range(self._nvar)
            ]
        else:
            self._varlist = [
                self._decode(self._path_or_buf.read(9)) for _ in range(self._nvar)
            ]
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]

        self._fmtlist = self._get_fmtlist()

        self._lbllist = self._get_lbllist()

        self._variable_labels = self._get_variable_labels()

        # ignore expansion fields (Format 105 and later)
        # When reading, read five bytes; the last four bytes now tell you
        # the size of the next read, which you discard.  You then continue
        # like this until you read 5 bytes of zeros.

        if self._format_version > 104:
            while True:
                data_type = self._read_int8()
                if self._format_version > 108:
                    data_len = self._read_int32()
                else:
                    data_len = self._read_int16()
                if data_type == 0:
                    break
                self._path_or_buf.read(data_len)

        # necessary data to continue parsing
        self._data_location = self._path_or_buf.tell()

    def _setup_dtype(self) -> np.dtype:
        """Map between numpy and state dtypes"""
        if self._dtype is not None:
            return self._dtype

        dtypes = []  # Convert struct data types to numpy data type
        for i, typ in enumerate(self._typlist):
            if typ in self.NUMPY_TYPE_MAP:
                typ = cast(str, typ)  # only strs in NUMPY_TYPE_MAP
                dtypes.append((f"s{i}", f"{self._byteorder}{self.NUMPY_TYPE_MAP[typ]}"))
            else:
                dtypes.append((f"s{i}", f"S{typ}"))
        self._dtype = np.dtype(dtypes)

        return self._dtype

    def _decode(self, s: bytes) -> str:
        # have bytes not strings, so must decode
        s = s.partition(b"\0")[0]
        try:
            return s.decode(self._encoding)
        except UnicodeDecodeError:
            # GH 25960, fallback to handle incorrect format produced when 117
            # files are converted to 118 files in Stata
            encoding = self._encoding
            msg = f"""
One or more strings in the dta file could not be decoded using {encoding}, and
so the fallback encoding of latin-1 is being used.  This can happen when a file
has been incorrectly encoded by Stata or some other software. You should verify
the string values returned are correct."""
            warnings.warn(
                msg,
                UnicodeWarning,
                stacklevel=find_stack_level(),
            )
            return s.decode("latin-1")

    def _read_value_labels(self) -> None:
        self._ensure_open()
        if self._value_labels_read:
            # Don't read twice
            return
        if self._format_version <= 108:
            # Value labels are not supported in version 108 and earlier.
            self._value_labels_read = True
            self._value_label_dict: dict[str, dict[float, str]] = {}
            return

        if self._format_version >= 117:
            self._path_or_buf.seek(self._seek_value_labels)
        else:
            assert self._dtype is not None
            offset = self._nobs * self._dtype.itemsize
            self._path_or_buf.seek(self._data_location + offset)

        self._value_labels_read = True
        self._value_label_dict = {}

        while True:
            if self._format_version >= 117:
                if self._path_or_buf.read(5) == b"</val":  # <lbl>
                    break  # end of value label table

            slength = self._path_or_buf.read(4)
            if not slength:
                break  # end of value label table (format < 117)
            if self._format_version <= 117:
                labname = self._decode(self._path_or_buf.read(33))
            else:
                labname = self._decode(self._path_or_buf.read(129))
            self._path_or_buf.read(3)  # padding

            n = self._read_uint32()
            txtlen = self._read_uint32()
            off = np.frombuffer(
                self._path_or_buf.read(4 * n), dtype=f"{self._byteorder}i4", count=n
            )
            val = np.frombuffer(
                self._path_or_buf.read(4 * n), dtype=f"{self._byteorder}i4", count=n
            )
            ii = np.argsort(off)
            off = off[ii]
            val = val[ii]
            txt = self._path_or_buf.read(txtlen)
            self._value_label_dict[labname] = {}
            for i in range(n):
                end = off[i + 1] if i < n - 1 else txtlen
                self._value_label_dict[labname][val[i]] = self._decode(
                    txt[off[i] : end]
                )
            if self._format_version >= 117:
                self._path_or_buf.read(6)  # </lbl>
        self._value_labels_read = True

    def _read_strls(self) -> None:
        self._path_or_buf.seek(self._seek_strls)
        # Wrap v_o in a string to allow uint64 values as keys on 32bit OS
        self.GSO = {"0": ""}
        while True:
            if self._path_or_buf.read(3) != b"GSO":
                break

            if self._format_version == 117:
                v_o = self._read_uint64()
            else:
                buf = self._path_or_buf.read(12)
                # Only tested on little endian file on little endian machine.
                v_size = 2 if self._format_version == 118 else 3
                if self._byteorder == "<":
                    buf = buf[0:v_size] + buf[4 : (12 - v_size)]
                else:
                    # This path may not be correct, impossible to test
                    buf = buf[0:v_size] + buf[(4 + v_size) :]
                v_o = struct.unpack("Q", buf)[0]
            typ = self._read_uint8()
            length = self._read_uint32()
            va = self._path_or_buf.read(length)
            if typ == 130:
                decoded_va = va[0:-1].decode(self._encoding)
            else:
                # Stata says typ 129 can be binary, so use str
                decoded_va = str(va)
                # Wrap v_o in a string to allow uint64 values as keys on 32bit OS
            self.GSO[str(v_o)] = decoded_va

    def __next__(self) -> DataFrame:
        self._using_iterator = True
        return self.read(nrows=self._chunksize)

    def get_chunk(self, size: int | None = None) -> DataFrame:
        """
        Reads lines from Stata file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
        if size is None:
            size = self._chunksize
        return self.read(nrows=size)

    @Appender(_read_method_doc)
    def read(
        self,
        nrows: int | None = None,
        convert_dates: bool | None = None,
        convert_categoricals: bool | None = None,
        index_col: str | None = None,
        convert_missing: bool | None = None,
        preserve_dtypes: bool | None = None,
        columns: Sequence[str] | None = None,
        order_categoricals: bool | None = None,
    ) -> DataFrame:
        self._ensure_open()

        # Handle options
        if convert_dates is None:
            convert_dates = self._convert_dates
        if convert_categoricals is None:
            convert_categoricals = self._convert_categoricals
        if convert_missing is None:
            convert_missing = self._convert_missing
        if preserve_dtypes is None:
            preserve_dtypes = self._preserve_dtypes
        if columns is None:
            columns = self._columns
        if order_categoricals is None:
            order_categoricals = self._order_categoricals
        if index_col is None:
            index_col = self._index_col
        if nrows is None:
            nrows = self._nobs

        # Handle empty file or chunk.  If reading incrementally raise
        # StopIteration.  If reading the whole thing return an empty
        # data frame.
        if (self._nobs == 0) and nrows == 0:
            self._can_read_value_labels = True
            self._data_read = True
            data = DataFrame(columns=self._varlist)
            # Apply dtypes correctly
            for i, col in enumerate(data.columns):
                dt = self._dtyplist[i]
                if isinstance(dt, np.dtype):
                    if dt.char != "S":
                        data[col] = data[col].astype(dt)
            if columns is not None:
                data = self._do_select_columns(data, columns)
            return data

        if (self._format_version >= 117) and (not self._value_labels_read):
            self._can_read_value_labels = True
            self._read_strls()

        # Read data
        assert self._dtype is not None
        dtype = self._dtype
        max_read_len = (self._nobs - self._lines_read) * dtype.itemsize
        read_len = nrows * dtype.itemsize
        read_len = min(read_len, max_read_len)
        if read_len <= 0:
            # Iterator has finished, should never be here unless
            # we are reading the file incrementally
            if convert_categoricals:
                self._read_value_labels()
            raise StopIteration
        offset = self._lines_read * dtype.itemsize
        self._path_or_buf.seek(self._data_location + offset)
        read_lines = min(nrows, self._nobs - self._lines_read)
        raw_data = np.frombuffer(
            self._path_or_buf.read(read_len), dtype=dtype, count=read_lines
        )

        self._lines_read += read_lines
        if self._lines_read == self._nobs:
            self._can_read_value_labels = True
            self._data_read = True
        # if necessary, swap the byte order to native here
        if self._byteorder != self._native_byteorder:
            raw_data = raw_data.byteswap().view(raw_data.dtype.newbyteorder())

        if convert_categoricals:
            self._read_value_labels()

        if len(raw_data) == 0:
            data = DataFrame(columns=self._varlist)
        else:
            data = DataFrame.from_records(raw_data)
            data.columns = Index(self._varlist)

        # If index is not specified, use actual row number rather than
        # restarting at 0 for each chunk.
        if index_col is None:
            data.index = RangeIndex(
                self._lines_read - read_lines, self._lines_read
            )  # set attr instead of set_index to avoid copy

        if columns is not None:
            data = self._do_select_columns(data, columns)

        # Decode strings
        for col, typ in zip(data, self._typlist):
            if isinstance(typ, int):
                data[col] = data[col].apply(self._decode)

        data = self._insert_strls(data)

        # Convert columns (if needed) to match input type
        valid_dtypes = [i for i, dtyp in enumerate(self._dtyplist) if dtyp is not None]
        object_type = np.dtype(object)
        for idx in valid_dtypes:
            dtype = data.iloc[:, idx].dtype
            if dtype not in (object_type, self._dtyplist[idx]):
                data.isetitem(idx, data.iloc[:, idx].astype(dtype))

        data = self._do_convert_missing(data, convert_missing)

        if convert_dates:
            for i, fmt in enumerate(self._fmtlist):
                if any(fmt.startswith(date_fmt) for date_fmt in _date_formats):
                    data.isetitem(
                        i, _stata_elapsed_date_to_datetime_vec(data.iloc[:, i], fmt)
                    )

        if convert_categoricals and self._format_version > 108:
            data = self._do_convert_categoricals(
                data, self._value_label_dict, self._lbllist, order_categoricals
            )

        if not preserve_dtypes:
            retyped_data = []
            convert = False
            for col in data:
                dtype = data[col].dtype
                if dtype in (np.dtype(np.float16), np.dtype(np.float32)):
                    dtype = np.dtype(np.float64)
                    convert = True
                elif dtype in (
                    np.dtype(np.int8),
                    np.dtype(np.int16),
                    np.dtype(np.int32),
                ):
                    dtype = np.dtype(np.int64)
                    convert = True
                retyped_data.append((col, data[col].astype(dtype)))
            if convert:
                data = DataFrame.from_dict(dict(retyped_data))

        if index_col is not None:
            data = data.set_index(data.pop(index_col))

        return data

    def _do_convert_missing(self, data: DataFrame, convert_missing: bool) -> DataFrame:
        # Check for missing values, and replace if found
        replacements = {}
        for i in range(len(data.columns)):
            fmt = self._typlist[i]
            if fmt not in self.VALID_RANGE:
                continue

            fmt = cast(str, fmt)  # only strs in VALID_RANGE
            nmin, nmax = self.VALID_RANGE[fmt]
            series = data.iloc[:, i]

            # appreciably faster to do this with ndarray instead of Series
            svals = series._values
            missing = (svals < nmin) | (svals > nmax)

            if not missing.any():
                continue

            if convert_missing:  # Replacement follows Stata notation
                missing_loc = np.nonzero(np.asarray(missing))[0]
                umissing, umissing_loc = np.unique(series[missing], return_inverse=True)
                replacement = Series(series, dtype=object)
                for j, um in enumerate(umissing):
                    missing_value = StataMissingValue(um)

                    loc = missing_loc[umissing_loc == j]
                    replacement.iloc[loc] = missing_value
            else:  # All replacements are identical
                dtype = series.dtype
                if dtype not in (np.float32, np.float64):
                    dtype = np.float64
                replacement = Series(series, dtype=dtype)
                if not replacement._values.flags["WRITEABLE"]:
                    # only relevant for ArrayManager; construction
                    #  path for BlockManager ensures writeability
                    replacement = replacement.copy()
                # Note: operating on ._values is much faster than directly
                # TODO: can we fix that?
                replacement._values[missing] = np.nan
            replacements[i] = replacement
        if replacements:
            for idx, value in replacements.items():
                data.isetitem(idx, value)
        return data

    def _insert_strls(self, data: DataFrame) -> DataFrame:
        if not hasattr(self, "GSO") or len(self.GSO) == 0:
            return data
        for i, typ in enumerate(self._typlist):
            if typ != "Q":
                continue
            # Wrap v_o in a string to allow uint64 values as keys on 32bit OS
            data.isetitem(i, [self.GSO[str(k)] for k in data.iloc[:, i]])
        return data

    def _do_select_columns(self, data: DataFrame, columns: Sequence[str]) -> DataFrame:
        if not self._column_selector_set:
            column_set = set(columns)
            if len(column_set) != len(columns):
                raise ValueError("columns contains duplicate entries")
            unmatched = column_set.difference(data.columns)
            if unmatched:
                joined = ", ".join(list(unmatched))
                raise ValueError(
                    "The following columns were not "
                    f"found in the Stata data set: {joined}"
                )
            # Copy information for retained columns for later processing
            dtyplist = []
            typlist = []
            fmtlist = []
            lbllist = []
            for col in columns:
                i = data.columns.get_loc(col)
                dtyplist.append(self._dtyplist[i])
                typlist.append(self._typlist[i])
                fmtlist.append(self._fmtlist[i])
                lbllist.append(self._lbllist[i])

            self._dtyplist = dtyplist
            self._typlist = typlist
            self._fmtlist = fmtlist
            self._lbllist = lbllist
            self._column_selector_set = True

        return data[columns]

    def _do_convert_categoricals(
        self,
        data: DataFrame,
        value_label_dict: dict[str, dict[float, str]],
        lbllist: Sequence[str],
        order_categoricals: bool,
    ) -> DataFrame:
        """
        Converts categorical columns to Categorical type.
        """
        if not value_label_dict:
            return data
        cat_converted_data = []
        for col, label in zip(data, lbllist):
            if label in value_label_dict:
                # Explicit call with ordered=True
                vl = value_label_dict[label]
                keys = np.array(list(vl.keys()))
                column = data[col]
                key_matches = column.isin(keys)
                if self._using_iterator and key_matches.all():
                    initial_categories: np.ndarray | None = keys
                    # If all categories are in the keys and we are iterating,
                    # use the same keys for all chunks. If some are missing
                    # value labels, then we will fall back to the categories
                    # varying across chunks.
                else:
                    if self._using_iterator:
                        # warn is using an iterator
                        warnings.warn(
                            categorical_conversion_warning,
                            CategoricalConversionWarning,
                            stacklevel=find_stack_level(),
                        )
                    initial_categories = None
                cat_data = Categorical(
                    column, categories=initial_categories, ordered=order_categoricals
                )
                if initial_categories is None:
                    # If None here, then we need to match the cats in the Categorical
                    categories = []
                    for category in cat_data.categories:
                        if category in vl:
                            categories.append(vl[category])
                        else:
                            categories.append(category)
                else:
                    # If all cats are matched, we can use the values
                    categories = list(vl.values())
                try:
                    # Try to catch duplicate categories
                    # TODO: if we get a non-copying rename_categories, use that
                    cat_data = cat_data.rename_categories(categories)
                except ValueError as err:
                    vc = Series(categories, copy=False).value_counts()
                    repeated_cats = list(vc.index[vc > 1])
                    repeats = "-" * 80 + "\n" + "\n".join(repeated_cats)
                    # GH 25772
                    msg = f"""
Value labels for column {col} are not unique. These cannot be converted to
pandas categoricals.

Either read the file with `convert_categoricals` set to False or use the
low level interface in `StataReader` to separately read the values and the
value_labels.

The repeated labels are:
{repeats}
"""
                    raise ValueError(msg) from err
                # TODO: is the next line needed above in the data(...) method?
                cat_series = Series(cat_data, index=data.index, copy=False)
                cat_converted_data.append((col, cat_series))
            else:
                cat_converted_data.append((col, data[col]))
        data = DataFrame(dict(cat_converted_data), copy=False)
        return data

    @property
    def data_label(self) -> str:
        """
        Return data label of Stata file.

        Examples
        --------
        >>> df = pd.DataFrame([(1,)], columns=["variable"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> data_label = "This is a data file."
        >>> path = "/My_path/filename.dta"
        >>> df.to_stata(path, time_stamp=time_stamp,    # doctest: +SKIP
        ...             data_label=data_label,  # doctest: +SKIP
        ...             version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.data_label)  # doctest: +SKIP
        This is a data file.
        """
        self._ensure_open()
        return self._data_label

    @property
    def time_stamp(self) -> str:
        """
        Return time stamp of Stata file.
        """
        self._ensure_open()
        return self._time_stamp

    def variable_labels(self) -> dict[str, str]:
        """
        Return a dict associating each variable name with corresponding label.

        Returns
        -------
        dict

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> variable_labels = {"col_1": "This is an example"}
        >>> df.to_stata(path, time_stamp=time_stamp,  # doctest: +SKIP
        ...             variable_labels=variable_labels, version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.variable_labels())  # doctest: +SKIP
        {'index': '', 'col_1': 'This is an example', 'col_2': ''}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    3    4
        """
        self._ensure_open()
        return dict(zip(self._varlist, self._variable_labels))

    def value_labels(self) -> dict[str, dict[float, str]]:
        """
        Return a nested dict associating each variable name to its value and label.

        Returns
        -------
        dict

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> value_labels = {"col_1": {3: "x"}}
        >>> df.to_stata(path, time_stamp=time_stamp,  # doctest: +SKIP
        ...             value_labels=value_labels, version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.value_labels())  # doctest: +SKIP
        {'col_1': {3: 'x'}}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    x    4
        """
        if not self._value_labels_read:
            self._read_value_labels()

        return self._value_label_dict


@Appender(_read_stata_doc)
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: str | None = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: Sequence[str] | None = None,
    order_categoricals: bool = True,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions | None = None,
) -> DataFrame | StataReader:
    reader = StataReader(
        filepath_or_buffer,
        convert_dates=convert_dates,
        convert_categoricals=convert_categoricals,
        index_col=index_col,
        convert_missing=convert_missing,
        preserve_dtypes=preserve_dtypes,
        columns=columns,
        order_categoricals=order_categoricals,
        chunksize=chunksize,
        storage_options=storage_options,
        compression=compression,
    )

    if iterator or chunksize:
        return reader

    with reader:
        return reader.read()


def _set_endianness(endianness: str) -> str:
    if endianness.lower() in ["<", "little"]:
        return "<"
    elif endianness.lower() in [">", "big"]:
        return ">"
    else:  # pragma : no cover
        raise ValueError(f"Endianness {endianness} not understood")


def _pad_bytes(name: AnyStr, length: int) -> AnyStr:
    """
    Take a char string and pads it with null bytes until it's length chars.
    """
    if isinstance(name, bytes):
        return name + b"\x00" * (length - len(name))
    return name + "\x00" * (length - len(name))


def _convert_datetime_to_stata_type(fmt: str) -> np.dtype:
    """
    Convert from one of the stata date formats to a type in TYPE_MAP.
    """
    if fmt in [
        "tc",
        "%tc",
        "td",
        "%td",
        "tw",
        "%tw",
        "tm",
        "%tm",
        "tq",
        "%tq",
        "th",
        "%th",
        "ty",
        "%ty",
    ]:
        return np.dtype(np.float64)  # Stata expects doubles for SIFs
    else:
        raise NotImplementedError(f"Format {fmt} not implemented")


def _maybe_convert_to_int_keys(convert_dates: dict, varlist: list[Hashable]) -> dict:
    new_dict = {}
    for key in convert_dates:
        if not convert_dates[key].startswith("%"):  # make sure proper fmts
            convert_dates[key] = "%" + convert_dates[key]
        if key in varlist:
            new_dict.update({varlist.index(key): convert_dates[key]})
        else:
            if not isinstance(key, int):
                raise ValueError("convert_dates key must be a column or an integer")
            new_dict.update({key: convert_dates[key]})
    return new_dict


def _dtype_to_stata_type(dtype: np.dtype, column: Series) -> int:
    """
    Convert dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 244 are strings of this length
                         Pandas    Stata
    251 - for int8      byte
    252 - for int16     int
    253 - for int32     long
    254 - for float32   float
    255 - for double    double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
    # TODO: expand to handle datetime to integer conversion
    if dtype.type is np.object_:  # try to coerce it to the biggest string
        # not memory efficient, what else could we
        # do?
        itemsize = max_len_string_array(ensure_object(column._values))
        return max(itemsize, 1)
    elif dtype.type is np.float64:
        return 255
    elif dtype.type is np.float32:
        return 254
    elif dtype.type is np.int32:
        return 253
    elif dtype.type is np.int16:
        return 252
    elif dtype.type is np.int8:
        return 251
    else:  # pragma : no cover
        raise NotImplementedError(f"Data type {dtype} not supported.")


def _dtype_to_default_stata_fmt(
    dtype, column: Series, dta_version: int = 114, force_strl: bool = False
) -> str:
    """
    Map numpy dtype to stata's default format for this type. Not terribly
    important since users can change this in Stata. Semantics are

    object  -> "%DDs" where DD is the length of the string.  If not a string,
                raise ValueError
    float64 -> "%10.0g"
    float32 -> "%9.0g"
    int64   -> "%9.0g"
    int32   -> "%12.0g"
    int16   -> "%8.0g"
    int8    -> "%8.0g"
    strl    -> "%9s"
    """
    # TODO: Refactor to combine type with format
    # TODO: expand this to handle a default datetime format?
    if dta_version < 117:
        max_str_len = 244
    else:
        max_str_len = 2045
        if force_strl:
            return "%9s"
    if dtype.type is np.object_:
        itemsize = max_len_string_array(ensure_object(column._values))
        if itemsize > max_str_len:
            if dta_version >= 117:
                return "%9s"
            else:
                raise ValueError(excessive_string_length_error.format(column.name))
        return "%" + str(max(itemsize, 1)) + "s"
    elif dtype == np.float64:
        return "%10.0g"
    elif dtype == np.float32:
        return "%9.0g"
    elif dtype == np.int32:
        return "%12.0g"
    elif dtype in (np.int8, np.int16):
        return "%8.0g"
    else:  # pragma : no cover
        raise NotImplementedError(f"Data type {dtype} not supported.")


@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "fname",
)
class StataWriter(StataParser):
    """
    A class for writing Stata binary dta files

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter instance
        The StataWriter instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1]], columns=['a', 'b'])
    >>> writer = StataWriter('./data_file.dta', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {{"method": "zip", "archive_name": "data_file.dta"}}
    >>> writer = StataWriter('./data_file.zip', data, compression=compression)
    >>> writer.write_file()

    Save a DataFrame with dates
    >>> from datetime import datetime
    >>> data = pd.DataFrame([[datetime(2000,1,1)]], columns=['date'])
    >>> writer = StataWriter('./date_data_file.dta', data, {{'date' : 'tw'}})
    >>> writer.write_file()
    """

    _max_string_length = 244
    _encoding: Literal["latin-1", "utf-8"] = "latin-1"

    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Hashable, str] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
        *,
        value_labels: dict[Hashable, dict[float, str]] | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self._convert_dates = {} if convert_dates is None else convert_dates
        self._write_index = write_index
        self._time_stamp = time_stamp
        self._data_label = data_label
        self._variable_labels = variable_labels
        self._non_cat_value_labels = value_labels
        self._value_labels: list[StataValueLabel] = []
        self._has_value_labels = np.array([], dtype=bool)
        self._compression = compression
        self._output_file: IO[bytes] | None = None
        self._converted_names: dict[Hashable, str] = {}
        # attach nobs, nvars, data, varlist, typlist
        self._prepare_pandas(data)
        self.storage_options = storage_options

        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder = _set_endianness(byteorder)
        self._fname = fname
        self.type_converters = {253: np.int32, 252: np.int16, 251: np.int8}

    def _write(self, to_write: str) -> None:
        """
        Helper to call encode before writing to file for Python 3 compat.
        """
        self.handles.handle.write(to_write.encode(self._encoding))

    def _write_bytes(self, value: bytes) -> None:
        """
        Helper to assert file is open before writing.
        """
        self.handles.handle.write(value)

    def _prepare_non_cat_value_labels(
        self, data: DataFrame
    ) -> list[StataNonCatValueLabel]:
        """
        Check for value labels provided for non-categorical columns. Value
        labels
        """
        non_cat_value_labels: list[StataNonCatValueLabel] = []
        if self._non_cat_value_labels is None:
            return non_cat_value_labels

        for labname, labels in self._non_cat_value_labels.items():
            if labname in self._converted_names:
                colname = self._converted_names[labname]
            elif labname in data.columns:
                colname = str(labname)
            else:
                raise KeyError(
                    f"Can't create value labels for {labname}, it wasn't "
                    "found in the dataset."
                )

            if not is_numeric_dtype(data[colname].dtype):
                # Labels should not be passed explicitly for categorical
                # columns that will be converted to int
                raise ValueError(
                    f"Can't create value labels for {labname}, value labels "
                    "can only be applied to numeric columns."
                )
            svl = StataNonCatValueLabel(colname, labels, self._encoding)
            non_cat_value_labels.append(svl)
        return non_cat_value_labels

    def _prepare_categoricals(self, data: DataFrame) -> DataFrame:
        """
        Check for categorical columns, retain categorical information for
        Stata file and convert categorical data to int
        """
        is_cat = [isinstance(dtype, CategoricalDtype) for dtype in data.dtypes]
        if not any(is_cat):
            return data

        self._has_value_labels |= np.array(is_cat)

        get_base_missing_value = StataMissingValue.get_base_missing_value
        data_formatted = []
        for col, col_is_cat in zip(data, is_cat):
            if col_is_cat:
                svl = StataValueLabel(data[col], encoding=self._encoding)
                self._value_labels.append(svl)
                dtype = data[col].cat.codes.dtype
                if dtype == np.int64:
                    raise ValueError(
                        "It is not possible to export "
                        "int64-based categorical data to Stata."
                    )
                values = data[col].cat.codes._values.copy()

                # Upcast if needed so that correct missing values can be set
                if values.max() >= get_base_missing_value(dtype):
                    if dtype == np.int8:
                        dtype = np.dtype(np.int16)
                    elif dtype == np.int16:
                        dtype = np.dtype(np.int32)
                    else:
                        dtype = np.dtype(np.float64)
                    values = np.array(values, dtype=dtype)

                # Replace missing values with Stata missing value for type
                values[values == -1] = get_base_missing_value(dtype)
                data_formatted.append((col, values))
            else:
                data_formatted.append((col, data[col]))
        return DataFrame.from_dict(dict(data_formatted))

    def _replace_nans(self, data: DataFrame) -> DataFrame:
        # return data
        """
        Checks floating point data columns for nans, and replaces these with
        the generic Stata for missing value (.)
        """
        for c in data:
            dtype = data[c].dtype
            if dtype in (np.float32, np.float64):
                if dtype == np.float32:
                    replacement = self.MISSING_VALUES["f"]
                else:
                    replacement = self.MISSING_VALUES["d"]
                data[c] = data[c].fillna(replacement)

        return data

    def _update_strl_names(self) -> None:
        """No-op, forward compatibility"""

    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 114 and 117 support ascii characters in a-z, A-Z, 0-9
        and _.
        """
        for c in name:
            if (
                (c < "A" or c > "Z")
                and (c < "a" or c > "z")
                and (c < "0" or c > "9")
                and c != "_"
            ):
                name = name.replace(c, "_")
        return name

    def _check_column_names(self, data: DataFrame) -> DataFrame:
        """
        Checks column names to ensure that they are valid Stata column names.
        This includes checks for:
            * Non-string names
            * Stata keywords
            * Variables that start with numbers
            * Variables with names that are too long

        When an illegal variable name is detected, it is converted, and if
        dates are exported, the variable name is propagated to the date
        conversion dictionary
        """
        converted_names: dict[Hashable, str] = {}
        columns = list(data.columns)
        original_columns = columns[:]

        duplicate_var_id = 0
        for j, name in enumerate(columns):
            orig_name = name
            if not isinstance(name, str):
                name = str(name)

            name = self._validate_variable_name(name)

            # Variable name must not be a reserved word
            if name in self.RESERVED_WORDS:
                name = "_" + name

            # Variable name may not start with a number
            if "0" <= name[0] <= "9":
                name = "_" + name

            name = name[: min(len(name), 32)]

            if not name == orig_name:
                # check for duplicates
                while columns.count(name) > 0:
                    # prepend ascending number to avoid duplicates
                    name = "_" + str(duplicate_var_id) + name
                    name = name[: min(len(name), 32)]
                    duplicate_var_id += 1
                converted_names[orig_name] = name

            columns[j] = name

        data.columns = Index(columns)

        # Check date conversion, and fix key if needed
        if self._convert_dates:
            for c, o in zip(columns, original_columns):
                if c != o:
                    self._convert_dates[c] = self._convert_dates[o]
                    del self._convert_dates[o]

        if converted_names:
            conversion_warning = []
            for orig_name, name in converted_names.items():
                msg = f"{orig_name}   ->   {name}"
                conversion_warning.append(msg)

            ws = invalid_name_doc.format("\n    ".join(conversion_warning))
            warnings.warn(
                ws,
                InvalidColumnName,
                stacklevel=find_stack_level(),
            )

        self._converted_names = converted_names
        self._update_strl_names()

        return data

    def _set_formats_and_types(self, dtypes: Series) -> None:
        self.fmtlist: list[str] = []
        self.typlist: list[int] = []
        for col, dtype in dtypes.items():
            self.fmtlist.append(_dtype_to_default_stata_fmt(dtype, self.data[col]))
            self.typlist.append(_dtype_to_stata_type(dtype, self.data[col]))

    def _prepare_pandas(self, data: DataFrame) -> None:
        # NOTE: we might need a different API / class for pandas objects so
        # we can set different semantics - handle this with a PR to pandas.io

        data = data.copy()

        if self._write_index:
            temp = data.reset_index()
            if isinstance(temp, DataFrame):
                data = temp

        # Ensure column names are strings
        data = self._check_column_names(data)

        # Check columns for compatibility with stata, upcast if necessary
        # Raise if outside the supported range
        data = _cast_to_stata_types(data)

        # Replace NaNs with Stata missing values
        data = self._replace_nans(data)

        # Set all columns to initially unlabelled
        self._has_value_labels = np.repeat(False, data.shape[1])

        # Create value labels for non-categorical data
        non_cat_value_labels = self._prepare_non_cat_value_labels(data)

        non_cat_columns = [svl.labname for svl in non_cat_value_labels]
        has_non_cat_val_labels = data.columns.isin(non_cat_columns)
        self._has_value_labels |= has_non_cat_val_labels
        self._value_labels.extend(non_cat_value_labels)

        # Convert categoricals to int data, and strip labels
        data = self._prepare_categoricals(data)

        self.nobs, self.nvar = data.shape
        self.data = data
        self.varlist = data.columns.tolist()

        dtypes = data.dtypes

        # Ensure all date columns are converted
        for col in data:
            if col in self._convert_dates:
                continue
            if lib.is_np_dtype(data[col].dtype, "M"):
                self._convert_dates[col] = "tc"

        self._convert_dates = _maybe_convert_to_int_keys(
            self._convert_dates, self.varlist
        )
        for key in self._convert_dates:
            new_type = _convert_datetime_to_stata_type(self._convert_dates[key])
            dtypes.iloc[key] = np.dtype(new_type)

        # Verify object arrays are strings and encode to bytes
        self._encode_strings()

        self._set_formats_and_types(dtypes)

        # set the given format for the datetime cols
        if self._convert_dates is not None:
            for key in self._convert_dates:
                if isinstance(key, int):
                    self.fmtlist[key] = self._convert_dates[key]

    def _encode_strings(self) -> None:
        """
        Encode strings in dta-specific encoding

        Do not encode columns marked for date conversion or for strL
        conversion. The strL converter independently handles conversion and
        also accepts empty string arrays.
        """
        convert_dates = self._convert_dates
        # _convert_strl is not available in dta 114
        convert_strl = getattr(self, "_convert_strl", [])
        for i, col in enumerate(self.data):
            # Skip columns marked for date conversion or strl conversion
            if i in convert_dates or col in convert_strl:
                continue
            column = self.data[col]
            dtype = column.dtype
            # TODO could also handle string dtype here specifically
            if dtype.type is np.object_:
                inferred_dtype = infer_dtype(column, skipna=True)
                if not ((inferred_dtype == "string") or len(column) == 0):
                    col = column.name
                    raise ValueError(
                        f"""\
Column `{col}` cannot be exported.\n\nOnly string-like object arrays
containing all strings or a mix of strings and None can be exported.
Object arrays containing only null values are prohibited. Other object
types cannot be exported and must first be converted to one of the
supported types."""
                    )
                encoded = self.data[col].str.encode(self._encoding)
                # If larger than _max_string_length do nothing
                if (
                    max_len_string_array(ensure_object(encoded._values))
                    <= self._max_string_length
                ):
                    self.data[col] = encoded

    def write_file(self) -> None:
        """
        Export DataFrame object to Stata dta format.

        Examples
        --------
        >>> df = pd.DataFrame({"fully_labelled": [1, 2, 3, 3, 1],
        ...                    "partially_labelled": [1.0, 2.0, np.nan, 9.0, np.nan],
        ...                    "Y": [7, 7, 9, 8, 10],
        ...                    "Z": pd.Categorical(["j", "k", "l", "k", "j"]),
        ...                    })
        >>> path = "/My_path/filename.dta"
        >>> labels = {"fully_labelled": {1: "one", 2: "two", 3: "three"},
        ...           "partially_labelled": {1.0: "one", 2.0: "two"},
        ...           }
        >>> writer = pd.io.stata.StataWriter(path,
        ...                                  df,
        ...                                  value_labels=labels)  # doctest: +SKIP
        >>> writer.write_file()  # doctest: +SKIP
        >>> df = pd.read_stata(path)  # doctest: +SKIP
        >>> df  # doctest: +SKIP
            index fully_labelled  partially_labeled  Y  Z
        0       0            one                one  7  j
        1       1            two                two  7  k
        2       2          three                NaN  9  l
        3       3          three                9.0  8  k
        4       4            one                NaN 10  j
        """
        with get_handle(
            self._fname,
            "wb",
            compression=self._compression,
            is_text=False,
            storage_options=self.storage_options,
        ) as self.handles:
            if self.handles.compression["method"] is not None:
                # ZipFile creates a file (with the same name) for each write call.
                # Write it first into a buffer and then write the buffer to the ZipFile.
                self._output_file, self.handles.handle = self.handles.handle, BytesIO()
                self.handles.created_handles.append(self.handles.handle)

            try:
                self._write_header(
                    data_label=self._data_label, time_stamp=self._time_stamp
                )
                self._write_map()
                self._write_variable_types()
                self._write_varnames()
                self._write_sortlist()
                self._write_formats()
                self._write_value_label_names()
                self._write_variable_labels()
                self._write_expansion_fields()
                self._write_characteristics()
                records = self._prepare_data()
                self._write_data(records)
                self._write_strls()
                self._write_value_labels()
                self._write_file_close_tag()
                self._write_map()
                self._close()
            except Exception as exc:
                self.handles.close()
                if isinstance(self._fname, (str, os.PathLike)) and os.path.isfile(
                    self._fname
                ):
                    try:
                        os.unlink(self._fname)
                    except OSError:
                        warnings.warn(
                            f"This save was not successful but {self._fname} could not "
                            "be deleted. This file is not valid.",
                            ResourceWarning,
                            stacklevel=find_stack_level(),
                        )
                raise exc

    def _close(self) -> None:
        """
        Close the file if it was created by the writer.

        If a buffer or file-like object was passed in, for example a GzipFile,
        then leave this file open for the caller to close.
        """
        # write compression
        if self._output_file is not None:
            assert isinstance(self.handles.handle, BytesIO)
            bio, self.handles.handle = self.handles.handle, self._output_file
            self.handles.handle.write(bio.getvalue())

    def _write_map(self) -> None:
        """No-op, future compatibility"""

    def _write_file_close_tag(self) -> None:
        """No-op, future compatibility"""

    def _write_characteristics(self) -> None:
        """No-op, future compatibility"""

    def _write_strls(self) -> None:
        """No-op, future compatibility"""

    def _write_expansion_fields(self) -> None:
        """Write 5 zeros for expansion fields"""
        self._write(_pad_bytes("", 5))

    def _write_value_labels(self) -> None:
        for vl in self._value_labels:
            self._write_bytes(vl.generate_value_label(self._byteorder))

    def _write_header(
        self,
        data_label: str | None = None,
        time_stamp: datetime | None = None,
    ) -> None:
        byteorder = self._byteorder
        # ds_format - just use 114
        self._write_bytes(struct.pack("b", 114))
        # byteorder
        self._write(byteorder == ">" and "\x01" or "\x02")
        # filetype
        self._write("\x01")
        # unused
        self._write("\x00")
        # number of vars, 2 bytes
        self._write_bytes(struct.pack(byteorder + "h", self.nvar)[:2])
        # number of obs, 4 bytes
        self._write_bytes(struct.pack(byteorder + "i", self.nobs)[:4])
        # data label 81 bytes, char, null terminated
        if data_label is None:
            self._write_bytes(self._null_terminate_bytes(_pad_bytes("", 80)))
        else:
            self._write_bytes(
                self._null_terminate_bytes(_pad_bytes(data_label[:80], 80))
            )
        # time stamp, 18 bytes, char, null terminated
        # format dd Mon yyyy hh:mm
        if time_stamp is None:
            time_stamp = datetime.now()
        elif not isinstance(time_stamp, datetime):
            raise ValueError("time_stamp should be datetime type")
        # GH #13856
        # Avoid locale-specific month conversion
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        month_lookup = {i + 1: month for i, month in enumerate(months)}
        ts = (
            time_stamp.strftime("%d ")
            + month_lookup[time_stamp.month]
            + time_stamp.strftime(" %Y %H:%M")
        )
        self._write_bytes(self._null_terminate_bytes(ts))

    def _write_variable_types(self) -> None:
        for typ in self.typlist:
            self._write_bytes(struct.pack("B", typ))

    def _write_varnames(self) -> None:
        # varlist names are checked by _check_column_names
        # varlist, requires null terminated
        for name in self.varlist:
            name = self._null_terminate_str(name)
            name = _pad_bytes(name[:32], 33)
            self._write(name)

    def _write_sortlist(self) -> None:
        # srtlist, 2*(nvar+1), int array, encoded by byteorder
        srtlist = _pad_bytes("", 2 * (self.nvar + 1))
        self._write(srtlist)

    def _write_formats(self) -> None:
        # fmtlist, 49*nvar, char array
        for fmt in self.fmtlist:
            self._write(_pad_bytes(fmt, 49))

    def _write_value_label_names(self) -> None:
        # lbllist, 33*nvar, char array
        for i in range(self.nvar):
            # Use variable name when categorical
            if self._has_value_labels[i]:
                name = self.varlist[i]
                name = self._null_terminate_str(name)
                name = _pad_bytes(name[:32], 33)
                self._write(name)
            else:  # Default is empty label
                self._write(_pad_bytes("", 33))

    def _write_variable_labels(self) -> None:
        # Missing labels are 80 blank characters plus null termination
        blank = _pad_bytes("", 81)

        if self._variable_labels is None:
            for i in range(self.nvar):
                self._write(blank)
            return

        for col in self.data:
            if col in self._variable_labels:
                label = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError("Variable labels must be 80 characters or fewer")
                is_latin1 = all(ord(c) < 256 for c in label)
                if not is_latin1:
                    raise ValueError(
                        "Variable labels must contain only characters that "
                        "can be encoded in Latin-1"
                    )
                self._write(_pad_bytes(label, 81))
            else:
                self._write(blank)

    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """No-op, future compatibility"""
        return data

    def _prepare_data(self) -> np.rec.recarray:
        data = self.data
        typlist = self.typlist
        convert_dates = self._convert_dates
        # 1. Convert dates
        if self._convert_dates is not None:
            for i, col in enumerate(data):
                if i in convert_dates:
                    data[col] = _datetime_to_stata_elapsed_vec(
                        data[col], self.fmtlist[i]
                    )
        # 2. Convert strls
        data = self._convert_strls(data)

        # 3. Convert bad string data to '' and pad to correct length
        dtypes = {}
        native_byteorder = self._byteorder == _set_endianness(sys.byteorder)
        for i, col in enumerate(data):
            typ = typlist[i]
            if typ <= self._max_string_length:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "Downcasting object dtype arrays",
                        category=FutureWarning,
                    )
                    dc = data[col].fillna("")
                data[col] = dc.apply(_pad_bytes, args=(typ,))
                stype = f"S{typ}"
                dtypes[col] = stype
                data[col] = data[col].astype(stype)
            else:
                dtype = data[col].dtype
                if not native_byteorder:
                    dtype = dtype.newbyteorder(self._byteorder)
                dtypes[col] = dtype

        return data.to_records(index=False, column_dtypes=dtypes)

    def _write_data(self, records: np.rec.recarray) -> None:
        self._write_bytes(records.tobytes())

    @staticmethod
    def _null_terminate_str(s: str) -> str:
        s += "\x00"
        return s

    def _null_terminate_bytes(self, s: str) -> bytes:
        return self._null_terminate_str(s).encode(self._encoding)


def _dtype_to_stata_type_117(dtype: np.dtype, column: Series, force_strl: bool) -> int:
    """
    Converts dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 2045 are strings of this length
                Pandas    Stata
    32768 - for object    strL
    65526 - for int8      byte
    65527 - for int16     int
    65528 - for int32     long
    65529 - for float32   float
    65530 - for double    double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
    # TODO: expand to handle datetime to integer conversion
    if force_strl:
        return 32768
    if dtype.type is np.object_:  # try to coerce it to the biggest string
        # not memory efficient, what else could we
        # do?
        itemsize = max_len_string_array(ensure_object(column._values))
        itemsize = max(itemsize, 1)
        if itemsize <= 2045:
            return itemsize
        return 32768
    elif dtype.type is np.float64:
        return 65526
    elif dtype.type is np.float32:
        return 65527
    elif dtype.type is np.int32:
        return 65528
    elif dtype.type is np.int16:
        return 65529
    elif dtype.type is np.int8:
        return 65530
    else:  # pragma : no cover
        raise NotImplementedError(f"Data type {dtype} not supported.")


def _pad_bytes_new(name: str | bytes, length: int) -> bytes:
    """
    Takes a bytes instance and pads it with null bytes until it's length chars.
    """
    if isinstance(name, str):
        name = bytes(name, "utf-8")
    return name + b"\x00" * (length - len(name))


class StataStrLWriter:
    """
    Converter for Stata StrLs

    Stata StrLs map 8 byte values to strings which are stored using a
    dictionary-like format where strings are keyed to two values.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert
    columns : Sequence[str]
        List of columns names to convert to StrL
    version : int, optional
        dta version.  Currently supports 117, 118 and 119
    byteorder : str, optional
        Can be ">", "<", "little", or "big". default is `sys.byteorder`

    Notes
    -----
    Supports creation of the StrL block of a dta file for dta versions
    117, 118 and 119.  These differ in how the GSO is stored.  118 and
    119 store the GSO lookup value as a uint32 and a uint64, while 117
    uses two uint32s. 118 and 119 also encode all strings as unicode
    which is required by the format.  117 uses 'latin-1' a fixed width
    encoding that extends the 7-bit ascii table with an additional 128
    characters.
    """

    def __init__(
        self,
        df: DataFrame,
        columns: Sequence[str],
        version: int = 117,
        byteorder: str | None = None,
    ) -> None:
        if version not in (117, 118, 119):
            raise ValueError("Only dta versions 117, 118 and 119 supported")
        self._dta_ver = version

        self.df = df
        self.columns = columns
        self._gso_table = {"": (0, 0)}
        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder = _set_endianness(byteorder)

        gso_v_type = "I"  # uint32
        gso_o_type = "Q"  # uint64
        self._encoding = "utf-8"
        if version == 117:
            o_size = 4
            gso_o_type = "I"  # 117 used uint32
            self._encoding = "latin-1"
        elif version == 118:
            o_size = 6
        else:  # version == 119
            o_size = 5
        self._o_offet = 2 ** (8 * (8 - o_size))
        self._gso_o_type = gso_o_type
        self._gso_v_type = gso_v_type

    def _convert_key(self, key: tuple[int, int]) -> int:
        v, o = key
        return v + self._o_offet * o

    def generate_table(self) -> tuple[dict[str, tuple[int, int]], DataFrame]:
        """
        Generates the GSO lookup table for the DataFrame

        Returns
        -------
        gso_table : dict
            Ordered dictionary using the string found as keys
            and their lookup position (v,o) as values
        gso_df : DataFrame
            DataFrame where strl columns have been converted to
            (v,o) values

        Notes
        -----
        Modifies the DataFrame in-place.

        The DataFrame returned encodes the (v,o) values as uint64s. The
        encoding depends on the dta version, and can be expressed as

        enc = v + o * 2 ** (o_size * 8)

        so that v is stored in the lower bits and o is in the upper
        bits. o_size is

          * 117: 4
          * 118: 6
          * 119: 5
        """
        gso_table = self._gso_table
        gso_df = self.df
        columns = list(gso_df.columns)
        selected = gso_df[self.columns]
        col_index = [(col, columns.index(col)) for col in self.columns]
        keys = np.empty(selected.shape, dtype=np.uint64)
        for o, (idx, row) in enumerate(selected.iterrows()):
            for j, (col, v) in enumerate(col_index):
                val = row[col]
                # Allow columns with mixed str and None (GH 23633)
                val = "" if val is None else val
                key = gso_table.get(val, None)
                if key is None:
                    # Stata prefers human numbers
                    key = (v + 1, o + 1)
                    gso_table[val] = key
                keys[o, j] = self._convert_key(key)
        for i, col in enumerate(self.columns):
            gso_df[col] = keys[:, i]

        return gso_table, gso_df

    def generate_blob(self, gso_table: dict[str, tuple[int, int]]) -> bytes:
        """
        Generates the binary blob of GSOs that is written to the dta file.

        Parameters
        ----------
        gso_table : dict
            Ordered dictionary (str, vo)

        Returns
        -------
        gso : bytes
            Binary content of dta file to be placed between strl tags

        Notes
        -----
        Output format depends on dta version.  117 uses two uint32s to
        express v and o while 118+ uses a uint32 for v and a uint64 for o.
        """
        # Format information
        # Length includes null term
        # 117
        # GSOvvvvooootllllxxxxxxxxxxxxxxx...x
        #  3  u4  u4 u1 u4  string + null term
        #
        # 118, 119
        # GSOvvvvooooooootllllxxxxxxxxxxxxxxx...x
        #  3  u4   u8   u1 u4    string + null term

        bio = BytesIO()
        gso = bytes("GSO", "ascii")
        gso_type = struct.pack(self._byteorder + "B", 130)
        null = struct.pack(self._byteorder + "B", 0)
        v_type = self._byteorder + self._gso_v_type
        o_type = self._byteorder + self._gso_o_type
        len_type = self._byteorder + "I"
        for strl, vo in gso_table.items():
            if vo == (0, 0):
                continue
            v, o = vo

            # GSO
            bio.write(gso)

            # vvvv
            bio.write(struct.pack(v_type, v))

            # oooo / oooooooo
            bio.write(struct.pack(o_type, o))

            # t
            bio.write(gso_type)

            # llll
            utf8_string = bytes(strl, "utf-8")
            bio.write(struct.pack(len_type, len(utf8_string) + 1))

            # xxx...xxx
            bio.write(utf8_string)
            bio.write(null)

        return bio.getvalue()


class StataWriter117(StataWriter):
    """
    A class for writing Stata binary dta files in Stata 13 format (117)

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter117 instance
        The StataWriter117 instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1, 'a']], columns=['a', 'b', 'c'])
    >>> writer = pd.io.stata.StataWriter117('./data_file.dta', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = pd.io.stata.StataWriter117(
    ...     './data_file.zip', data, compression=compression
    ...     )
    >>> writer.write_file()

    Or with long strings stored in strl format
    >>> data = pd.DataFrame([['A relatively long string'], [''], ['']],
    ...                     columns=['strls'])
    >>> writer = pd.io.stata.StataWriter117(
    ...     './data_file_with_long_strings.dta', data, convert_strl=['strls'])
    >>> writer.write_file()
    """

    _max_string_length = 2045
    _dta_version = 117

    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Hashable, str] | None = None,
        convert_strl: Sequence[Hashable] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
        *,
        value_labels: dict[Hashable, dict[float, str]] | None = None,
    ) -> None:
        # Copy to new list since convert_strl might be modified later
        self._convert_strl: list[Hashable] = []
        if convert_strl is not None:
            self._convert_strl.extend(convert_strl)

        super().__init__(
            fname,
            data,
            convert_dates,
            write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            value_labels=value_labels,
            compression=compression,
            storage_options=storage_options,
        )
        self._map: dict[str, int] = {}
        self._strl_blob = b""

    @staticmethod
    def _tag(val: str | bytes, tag: str) -> bytes:
        """Surround val with <tag></tag>"""
        if isinstance(val, str):
            val = bytes(val, "utf-8")
        return bytes("<" + tag + ">", "utf-8") + val + bytes("</" + tag + ">", "utf-8")

    def _update_map(self, tag: str) -> None:
        """Update map location for tag with file position"""
        assert self.handles.handle is not None
        self._map[tag] = self.handles.handle.tell()

    def _write_header(
        self,
        data_label: str | None = None,
        time_stamp: datetime | None = None,
    ) -> None:
        """Write the file header"""
        byteorder = self._byteorder
        self._write_bytes(bytes("<stata_dta>", "utf-8"))
        bio = BytesIO()
        # ds_format - 117
        bio.write(self._tag(bytes(str(self._dta_version), "utf-8"), "release"))
        # byteorder
        bio.write(self._tag(byteorder == ">" and "MSF" or "LSF", "byteorder"))
        # number of vars, 2 bytes in 117 and 118, 4 byte in 119
        nvar_type = "H" if self._dta_version <= 118 else "I"
        bio.write(self._tag(struct.pack(byteorder + nvar_type, self.nvar), "K"))
        # 117 uses 4 bytes, 118 uses 8
        nobs_size = "I" if self._dta_version == 117 else "Q"
        bio.write(self._tag(struct.pack(byteorder + nobs_size, self.nobs), "N"))
        # data label 81 bytes, char, null terminated
        label = data_label[:80] if data_label is not None else ""
        encoded_label = label.encode(self._encoding)
        label_size = "B" if self._dta_version == 117 else "H"
        label_len = struct.pack(byteorder + label_size, len(encoded_label))
        encoded_label = label_len + encoded_label
        bio.write(self._tag(encoded_label, "label"))
        # time stamp, 18 bytes, char, null terminated
        # format dd Mon yyyy hh:mm
        if time_stamp is None:
            time_stamp = datetime.now()
        elif not isinstance(time_stamp, datetime):
            raise ValueError("time_stamp should be datetime type")
        # Avoid locale-specific month conversion
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        month_lookup = {i + 1: month for i, month in enumerate(months)}
        ts = (
            time_stamp.strftime("%d ")
            + month_lookup[time_stamp.month]
            + time_stamp.strftime(" %Y %H:%M")
        )
        # '\x11' added due to inspection of Stata file
        stata_ts = b"\x11" + bytes(ts, "utf-8")
        bio.write(self._tag(stata_ts, "timestamp"))
        self._write_bytes(self._tag(bio.getvalue(), "header"))

    def _write_map(self) -> None:
        """
        Called twice during file write. The first populates the values in
        the map with 0s.  The second call writes the final map locations when
        all blocks have been written.
        """
        if not self._map:
            self._map = {
                "stata_data": 0,
                "map": self.handles.handle.tell(),
                "variable_types": 0,
                "varnames": 0,
                "sortlist": 0,
                "formats": 0,
                "value_label_names": 0,
                "variable_labels": 0,
                "characteristics": 0,
                "data": 0,
                "strls": 0,
                "value_labels": 0,
                "stata_data_close": 0,
                "end-of-file": 0,
            }
        # Move to start of map
        self.handles.handle.seek(self._map["map"])
        bio = BytesIO()
        for val in self._map.values():
            bio.write(struct.pack(self._byteorder + "Q", val))
        self._write_bytes(self._tag(bio.getvalue(), "map"))

    def _write_variable_types(self) -> None:
        self._update_map("variable_types")
        bio = BytesIO()
        for typ in self.typlist:
            bio.write(struct.pack(self._byteorder + "H", typ))
        self._write_bytes(self._tag(bio.getvalue(), "variable_types"))

    def _write_varnames(self) -> None:
        self._update_map("varnames")
        bio = BytesIO()
        # 118 scales by 4 to accommodate utf-8 data worst case encoding
        vn_len = 32 if self._dta_version == 117 else 128
        for name in self.varlist:
            name = self._null_terminate_str(name)
            name = _pad_bytes_new(name[:32].encode(self._encoding), vn_len + 1)
            bio.write(name)
        self._write_bytes(self._tag(bio.getvalue(), "varnames"))

    def _write_sortlist(self) -> None:
        self._update_map("sortlist")
        sort_size = 2 if self._dta_version < 119 else 4
        self._write_bytes(self._tag(b"\x00" * sort_size * (self.nvar + 1), "sortlist"))

    def _write_formats(self) -> None:
        self._update_map("formats")
        bio = BytesIO()
        fmt_len = 49 if self._dta_version == 117 else 57
        for fmt in self.fmtlist:
            bio.write(_pad_bytes_new(fmt.encode(self._encoding), fmt_len))
        self._write_bytes(self._tag(bio.getvalue(), "formats"))

    def _write_value_label_names(self) -> None:
        self._update_map("value_label_names")
        bio = BytesIO()
        # 118 scales by 4 to accommodate utf-8 data worst case encoding
        vl_len = 32 if self._dta_version == 117 else 128
        for i in range(self.nvar):
            # Use variable name when categorical
            name = ""  # default name
            if self._has_value_labels[i]:
                name = self.varlist[i]
            name = self._null_terminate_str(name)
            encoded_name = _pad_bytes_new(name[:32].encode(self._encoding), vl_len + 1)
            bio.write(encoded_name)
        self._write_bytes(self._tag(bio.getvalue(), "value_label_names"))

    def _write_variable_labels(self) -> None:
        # Missing labels are 80 blank characters plus null termination
        self._update_map("variable_labels")
        bio = BytesIO()
        # 118 scales by 4 to accommodate utf-8 data worst case encoding
        vl_len = 80 if self._dta_version == 117 else 320
        blank = _pad_bytes_new("", vl_len + 1)

        if self._variable_labels is None:
            for _ in range(self.nvar):
                bio.write(blank)
            self._write_bytes(self._tag(bio.getvalue(), "variable_labels"))
            return

        for col in self.data:
            if col in self._variable_labels:
                label = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError("Variable labels must be 80 characters or fewer")
                try:
                    encoded = label.encode(self._encoding)
                except UnicodeEncodeError as err:
                    raise ValueError(
                        "Variable labels must contain only characters that "
                        f"can be encoded in {self._encoding}"
                    ) from err

                bio.write(_pad_bytes_new(encoded, vl_len + 1))
            else:
                bio.write(blank)
        self._write_bytes(self._tag(bio.getvalue(), "variable_labels"))

    def _write_characteristics(self) -> None:
        self._update_map("characteristics")
        self._write_bytes(self._tag(b"", "characteristics"))

    def _write_data(self, records) -> None:
        self._update_map("data")
        self._write_bytes(b"<data>")
        self._write_bytes(records.tobytes())
        self._write_bytes(b"</data>")

    def _write_strls(self) -> None:
        self._update_map("strls")
        self._write_bytes(self._tag(self._strl_blob, "strls"))

    def _write_expansion_fields(self) -> None:
        """No-op in dta 117+"""

    def _write_value_labels(self) -> None:
        self._update_map("value_labels")
        bio = BytesIO()
        for vl in self._value_labels:
            lab = vl.generate_value_label(self._byteorder)
            lab = self._tag(lab, "lbl")
            bio.write(lab)
        self._write_bytes(self._tag(bio.getvalue(), "value_labels"))

    def _write_file_close_tag(self) -> None:
        self._update_map("stata_data_close")
        self._write_bytes(bytes("</stata_dta>", "utf-8"))
        self._update_map("end-of-file")

    def _update_strl_names(self) -> None:
        """
        Update column names for conversion to strl if they might have been
        changed to comply with Stata naming rules
        """
        # Update convert_strl if names changed
        for orig, new in self._converted_names.items():
            if orig in self._convert_strl:
                idx = self._convert_strl.index(orig)
                self._convert_strl[idx] = new

    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """
        Convert columns to StrLs if either very large or in the
        convert_strl variable
        """
        convert_cols = [
            col
            for i, col in enumerate(data)
            if self.typlist[i] == 32768 or col in self._convert_strl
        ]

        if convert_cols:
            ssw = StataStrLWriter(data, convert_cols, version=self._dta_version)
            tab, new_data = ssw.generate_table()
            data = new_data
            self._strl_blob = ssw.generate_blob(tab)
        return data

    def _set_formats_and_types(self, dtypes: Series) -> None:
        self.typlist = []
        self.fmtlist = []
        for col, dtype in dtypes.items():
            force_strl = col in self._convert_strl
            fmt = _dtype_to_default_stata_fmt(
                dtype,
                self.data[col],
                dta_version=self._dta_version,
                force_strl=force_strl,
            )
            self.fmtlist.append(fmt)
            self.typlist.append(
                _dtype_to_stata_type_117(dtype, self.data[col], force_strl)
            )


class StataWriterUTF8(StataWriter117):
    """
    Stata binary dta file writing in Stata 15 (118) and 16 (119) formats

    DTA 118 and 119 format files support unicode string data (both fixed
    and strL) format. Unicode is also supported in value labels, variable
    labels and the dataset label. Format 119 is automatically used if the
    file contains more than 32,767 variables.

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict, default None
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool, default True
        Write the index to Stata dataset.
    byteorder : str, default None
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime, default None
        A datetime to use as file creation date.  Default is the current time
    data_label : str, default None
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict, default None
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list, default None
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    version : int, default None
        The dta version to use. By default, uses the size of data to determine
        the version. 118 is used if data.shape[1] <= 32767, and 119 is used
        for storing larger DataFrames.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    StataWriterUTF8
        The instance has a write_file method, which will write the file to the
        given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    Using Unicode data and column names

    >>> from pandas.io.stata import StataWriterUTF8
    >>> data = pd.DataFrame([[1.0, 1, 'ᴬ']], columns=['a', 'β', 'ĉ'])
    >>> writer = StataWriterUTF8('./data_file.dta', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = StataWriterUTF8('./data_file.zip', data, compression=compression)
    >>> writer.write_file()

    Or with long strings stored in strl format

    >>> data = pd.DataFrame([['ᴀ relatively long ŝtring'], [''], ['']],
    ...                     columns=['strls'])
    >>> writer = StataWriterUTF8('./data_file_with_long_strings.dta', data,
    ...                          convert_strl=['strls'])
    >>> writer.write_file()
    """

    _encoding: Literal["utf-8"] = "utf-8"

    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Hashable, str] | None = None,
        convert_strl: Sequence[Hashable] | None = None,
        version: int | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
        *,
        value_labels: dict[Hashable, dict[float, str]] | None = None,
    ) -> None:
        if version is None:
            version = 118 if data.shape[1] <= 32767 else 119
        elif version not in (118, 119):
            raise ValueError("version must be either 118 or 119.")
        elif version == 118 and data.shape[1] > 32767:
            raise ValueError(
                "You must use version 119 for data sets containing more than"
                "32,767 variables"
            )

        super().__init__(
            fname,
            data,
            convert_dates=convert_dates,
            write_index=write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            value_labels=value_labels,
            convert_strl=convert_strl,
            compression=compression,
            storage_options=storage_options,
        )
        # Override version set in StataWriter117 init
        self._dta_version = version

    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 118+ support most unicode characters. The only limitation is in
        the ascii range where the characters supported are a-z, A-Z, 0-9 and _.
        """
        # High code points appear to be acceptable
        for c in name:
            if (
                (
                    ord(c) < 128
                    and (c < "A" or c > "Z")
                    and (c < "a" or c > "z")
                    and (c < "0" or c > "9")
                    and c != "_"
                )
                or 128 <= ord(c) < 192
                or c in {"×", "÷"}  # noqa: RUF001
            ):
                name = name.replace(c, "_")

        return name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\oracle\base.py ===
# dialects/oracle/base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


r"""
.. dialect:: oracle
    :name: Oracle Database
    :normal_support: 11+
    :best_effort: 9+


Auto Increment Behavior
-----------------------

SQLAlchemy Table objects which include integer primary keys are usually assumed
to have "autoincrementing" behavior, meaning they can generate their own
primary key values upon INSERT. For use within Oracle Database, two options are
available, which are the use of IDENTITY columns (Oracle Database 12 and above
only) or the association of a SEQUENCE with the column.

Specifying GENERATED AS IDENTITY (Oracle Database 12 and above)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from version 12, Oracle Database can make use of identity columns
using the :class:`_sql.Identity` to specify the autoincrementing behavior::

    t = Table(
        "mytable",
        metadata,
        Column("id", Integer, Identity(start=3), primary_key=True),
        Column(...),
        ...,
    )

The CREATE TABLE for the above :class:`_schema.Table` object would be:

.. sourcecode:: sql

    CREATE TABLE mytable (
        id INTEGER GENERATED BY DEFAULT AS IDENTITY (START WITH 3),
        ...,
        PRIMARY KEY (id)
    )

The :class:`_schema.Identity` object support many options to control the
"autoincrementing" behavior of the column, like the starting value, the
incrementing value, etc.  In addition to the standard options, Oracle Database
supports setting :paramref:`_schema.Identity.always` to ``None`` to use the
default generated mode, rendering GENERATED AS IDENTITY in the DDL. It also supports
setting :paramref:`_schema.Identity.on_null` to ``True`` to specify ON NULL
in conjunction with a 'BY DEFAULT' identity column.

Using a SEQUENCE (all Oracle Database versions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Older version of Oracle Database had no "autoincrement" feature: SQLAlchemy
relies upon sequences to produce these values.  With the older Oracle Database
versions, *a sequence must always be explicitly specified to enable
autoincrement*.  This is divergent with the majority of documentation examples
which assume the usage of an autoincrement-capable database.  To specify
sequences, use the sqlalchemy.schema.Sequence object which is passed to a
Column construct::

  t = Table(
      "mytable",
      metadata,
      Column("id", Integer, Sequence("id_seq", start=1), primary_key=True),
      Column(...),
      ...,
  )

This step is also required when using table reflection, i.e. autoload_with=engine::

  t = Table(
      "mytable",
      metadata,
      Column("id", Integer, Sequence("id_seq", start=1), primary_key=True),
      autoload_with=engine,
  )

.. versionchanged::  1.4   Added :class:`_schema.Identity` construct
   in a :class:`_schema.Column` to specify the option of an autoincrementing
   column.

.. _oracle_isolation_level:

Transaction Isolation Level / Autocommit
----------------------------------------

Oracle Database supports "READ COMMITTED" and "SERIALIZABLE" modes of
isolation. The AUTOCOMMIT isolation level is also supported by the
python-oracledb and cx_Oracle dialects.

To set using per-connection execution options::

    connection = engine.connect()
    connection = connection.execution_options(isolation_level="AUTOCOMMIT")

For ``READ COMMITTED`` and ``SERIALIZABLE``, the Oracle Database dialects sets
the level at the session level using ``ALTER SESSION``, which is reverted back
to its default setting when the connection is returned to the connection pool.

Valid values for ``isolation_level`` include:

* ``READ COMMITTED``
* ``AUTOCOMMIT``
* ``SERIALIZABLE``

.. note:: The implementation for the
   :meth:`_engine.Connection.get_isolation_level` method as implemented by the
   Oracle Database dialects necessarily force the start of a transaction using the
   Oracle Database DBMS_TRANSACTION.LOCAL_TRANSACTION_ID function; otherwise no
   level is normally readable.

   Additionally, the :meth:`_engine.Connection.get_isolation_level` method will
   raise an exception if the ``v$transaction`` view is not available due to
   permissions or other reasons, which is a common occurrence in Oracle Database
   installations.

   The python-oracledb and cx_Oracle dialects attempt to call the
   :meth:`_engine.Connection.get_isolation_level` method when the dialect makes
   its first connection to the database in order to acquire the
   "default"isolation level.  This default level is necessary so that the level
   can be reset on a connection after it has been temporarily modified using
   :meth:`_engine.Connection.execution_options` method.  In the common event
   that the :meth:`_engine.Connection.get_isolation_level` method raises an
   exception due to ``v$transaction`` not being readable as well as any other
   database-related failure, the level is assumed to be "READ COMMITTED".  No
   warning is emitted for this initial first-connect condition as it is
   expected to be a common restriction on Oracle databases.

.. versionadded:: 1.3.16 added support for AUTOCOMMIT to the cx_Oracle dialect
   as well as the notion of a default isolation level

.. versionadded:: 1.3.21 Added support for SERIALIZABLE as well as live
   reading of the isolation level.

.. versionchanged:: 1.3.22 In the event that the default isolation
   level cannot be read due to permissions on the v$transaction view as
   is common in Oracle installations, the default isolation level is hardcoded
   to "READ COMMITTED" which was the behavior prior to 1.3.21.

.. seealso::

    :ref:`dbapi_autocommit`

Identifier Casing
-----------------

In Oracle Database, the data dictionary represents all case insensitive
identifier names using UPPERCASE text.  This is in contradiction to the
expectations of SQLAlchemy, which assume a case insensitive name is represented
as lowercase text.

As an example of case insensitive identifier names, consider the following table:

.. sourcecode:: sql

    CREATE TABLE MyTable (Identifier INTEGER PRIMARY KEY)

If you were to ask Oracle Database for information about this table, the
table name would be reported as ``MYTABLE`` and the column name would
be reported as ``IDENTIFIER``.    Compare to most other databases such as
PostgreSQL and MySQL which would report these names as ``mytable`` and
``identifier``.   The names are **not quoted, therefore are case insensitive**.
The special casing of ``MyTable`` and ``Identifier`` would only be maintained
if they were quoted in the table definition:

.. sourcecode:: sql

    CREATE TABLE "MyTable" ("Identifier" INTEGER PRIMARY KEY)

When constructing a SQLAlchemy :class:`.Table` object, **an all lowercase name
is considered to be case insensitive**.   So the following table assumes
case insensitive names::

    Table("mytable", metadata, Column("identifier", Integer, primary_key=True))

Whereas when mixed case or UPPERCASE names are used, case sensitivity is
assumed::

    Table("MyTable", metadata, Column("Identifier", Integer, primary_key=True))

A similar situation occurs at the database driver level when emitting a
textual SQL SELECT statement and looking at column names in the DBAPI
``cursor.description`` attribute.  A database like PostgreSQL will normalize
case insensitive names to be lowercase::

    >>> pg_engine = create_engine("postgresql://scott:tiger@localhost/test")
    >>> pg_connection = pg_engine.connect()
    >>> result = pg_connection.exec_driver_sql("SELECT 1 AS SomeName")
    >>> result.cursor.description
    (Column(name='somename', type_code=23),)

Whereas Oracle normalizes them to UPPERCASE::

    >>> oracle_engine = create_engine("oracle+oracledb://scott:tiger@oracle18c/xe")
    >>> oracle_connection = oracle_engine.connect()
    >>> result = oracle_connection.exec_driver_sql(
    ...     "SELECT 1 AS SomeName FROM DUAL"
    ... )
    >>> result.cursor.description
    [('SOMENAME', <DbType DB_TYPE_NUMBER>, 127, None, 0, -127, True)]

In order to achieve cross-database parity for the two cases of a. table
reflection and b. textual-only SQL statement round trips, SQLAlchemy performs a step
called **name normalization** when using the Oracle dialect.  This process may
also apply to other third party dialects that have similar UPPERCASE handling
of case insensitive names.

When using name normalization, SQLAlchemy attempts to detect if a name is
case insensitive by checking if all characters are UPPERCASE letters only;
if so, then it assumes this is a case insensitive name and is delivered as
a lowercase name.

For table reflection, a tablename that is seen represented as all UPPERCASE
in Oracle Database's catalog tables will be assumed to have a case insensitive
name.  This is what allows the ``Table`` definition to use lower case names
and be equally compatible from a reflection point of view on Oracle Database
and all other databases such as PostgreSQL and MySQL::

    # matches a table created with CREATE TABLE mytable
    Table("mytable", metadata, autoload_with=some_engine)

Above, the all lowercase name ``"mytable"`` is case insensitive; it will match
a table reported by PostgreSQL as ``"mytable"`` and a table reported by
Oracle as ``"MYTABLE"``.  If name normalization were not present, it would
not be possible for the above :class:`.Table` definition to be introspectable
in a cross-database way, since we are dealing with a case insensitive name
that is not reported by each database in the same way.

Case sensitivity can be forced on in this case, such as if we wanted to represent
the quoted tablename ``"MYTABLE"`` with that exact casing, most simply by using
that casing directly, which will be seen as a case sensitive name::

    # matches a table created with CREATE TABLE "MYTABLE"
    Table("MYTABLE", metadata, autoload_with=some_engine)

For the unusual case of a quoted all-lowercase name, the :class:`.quoted_name`
construct may be used::

    from sqlalchemy import quoted_name

    # matches a table created with CREATE TABLE "mytable"
    Table(
        quoted_name("mytable", quote=True), metadata, autoload_with=some_engine
    )

Name normalization also takes place when handling result sets from **purely
textual SQL strings**, that have no other :class:`.Table` or :class:`.Column`
metadata associated with them. This includes SQL strings executed using
:meth:`.Connection.exec_driver_sql` and SQL strings executed using the
:func:`.text` construct which do not include :class:`.Column` metadata.

Returning to the Oracle Database SELECT statement, we see that even though
``cursor.description`` reports the column name as ``SOMENAME``, SQLAlchemy
name normalizes this to ``somename``::

    >>> oracle_engine = create_engine("oracle+oracledb://scott:tiger@oracle18c/xe")
    >>> oracle_connection = oracle_engine.connect()
    >>> result = oracle_connection.exec_driver_sql(
    ...     "SELECT 1 AS SomeName FROM DUAL"
    ... )
    >>> result.cursor.description
    [('SOMENAME', <DbType DB_TYPE_NUMBER>, 127, None, 0, -127, True)]
    >>> result.keys()
    RMKeyView(['somename'])

The single scenario where the above behavior produces inaccurate results
is when using an all-uppercase, quoted name.  SQLAlchemy has no way to determine
that a particular name in ``cursor.description`` was quoted, and is therefore
case sensitive, or was not quoted, and should be name normalized::

    >>> result = oracle_connection.exec_driver_sql(
    ...     'SELECT 1 AS "SOMENAME" FROM DUAL'
    ... )
    >>> result.cursor.description
    [('SOMENAME', <DbType DB_TYPE_NUMBER>, 127, None, 0, -127, True)]
    >>> result.keys()
    RMKeyView(['somename'])

For this case, a new feature will be available in SQLAlchemy 2.1 to disable
the name normalization behavior in specific cases.


.. _oracle_max_identifier_lengths:

Maximum Identifier Lengths
--------------------------

SQLAlchemy is sensitive to the maximum identifier length supported by Oracle
Database. This affects generated SQL label names as well as the generation of
constraint names, particularly in the case where the constraint naming
convention feature described at :ref:`constraint_naming_conventions` is being
used.

Oracle Database 12.2 increased the default maximum identifier length from 30 to
128. As of SQLAlchemy 1.4, the default maximum identifier length for the Oracle
dialects is 128 characters.  Upon first connection, the maximum length actually
supported by the database is obtained. In all cases, setting the
:paramref:`_sa.create_engine.max_identifier_length` parameter will bypass this
change and the value given will be used as is::

    engine = create_engine(
        "oracle+oracledb://scott:tiger@localhost:1521?service_name=freepdb1",
        max_identifier_length=30,
    )

If :paramref:`_sa.create_engine.max_identifier_length` is not set, the oracledb
dialect internally uses the ``max_identifier_length`` attribute available on
driver connections since python-oracledb version 2.5. When using an older
driver version, or using the cx_Oracle dialect, SQLAlchemy will instead attempt
to use the query ``SELECT value FROM v$parameter WHERE name = 'compatible'``
upon first connect in order to determine the effective compatibility version of
the database. The "compatibility" version is a version number that is
independent of the actual database version. It is used to assist database
migration. It is configured by an Oracle Database initialization parameter. The
compatibility version then determines the maximum allowed identifier length for
the database. If the V$ view is not available, the database version information
is used instead.

The maximum identifier length comes into play both when generating anonymized
SQL labels in SELECT statements, but more crucially when generating constraint
names from a naming convention.  It is this area that has created the need for
SQLAlchemy to change this default conservatively.  For example, the following
naming convention produces two very different constraint names based on the
identifier length::

    from sqlalchemy import Column
    from sqlalchemy import Index
    from sqlalchemy import Integer
    from sqlalchemy import MetaData
    from sqlalchemy import Table
    from sqlalchemy.dialects import oracle
    from sqlalchemy.schema import CreateIndex

    m = MetaData(naming_convention={"ix": "ix_%(column_0N_name)s"})

    t = Table(
        "t",
        m,
        Column("some_column_name_1", Integer),
        Column("some_column_name_2", Integer),
        Column("some_column_name_3", Integer),
    )

    ix = Index(
        None,
        t.c.some_column_name_1,
        t.c.some_column_name_2,
        t.c.some_column_name_3,
    )

    oracle_dialect = oracle.dialect(max_identifier_length=30)
    print(CreateIndex(ix).compile(dialect=oracle_dialect))

With an identifier length of 30, the above CREATE INDEX looks like:

.. sourcecode:: sql

    CREATE INDEX ix_some_column_name_1s_70cd ON t
    (some_column_name_1, some_column_name_2, some_column_name_3)

However with length of 128, it becomes::

.. sourcecode:: sql

    CREATE INDEX ix_some_column_name_1some_column_name_2some_column_name_3 ON t
    (some_column_name_1, some_column_name_2, some_column_name_3)

Applications which have run versions of SQLAlchemy prior to 1.4 on Oracle
Database version 12.2 or greater are therefore subject to the scenario of a
database migration that wishes to "DROP CONSTRAINT" on a name that was
previously generated with the shorter length.  This migration will fail when
the identifier length is changed without the name of the index or constraint
first being adjusted.  Such applications are strongly advised to make use of
:paramref:`_sa.create_engine.max_identifier_length` in order to maintain
control of the generation of truncated names, and to fully review and test all
database migrations in a staging environment when changing this value to ensure
that the impact of this change has been mitigated.

.. versionchanged:: 1.4 the default max_identifier_length for Oracle Database
   is 128 characters, which is adjusted down to 30 upon first connect if the
   Oracle Database, or its compatibility setting, are lower than version 12.2.


LIMIT/OFFSET/FETCH Support
--------------------------

Methods like :meth:`_sql.Select.limit` and :meth:`_sql.Select.offset` make use
of ``FETCH FIRST N ROW / OFFSET N ROWS`` syntax assuming Oracle Database 12c or
above, and assuming the SELECT statement is not embedded within a compound
statement like UNION.  This syntax is also available directly by using the
:meth:`_sql.Select.fetch` method.

.. versionchanged:: 2.0 the Oracle Database dialects now use ``FETCH FIRST N
   ROW / OFFSET N ROWS`` for all :meth:`_sql.Select.limit` and
   :meth:`_sql.Select.offset` usage including within the ORM and legacy
   :class:`_orm.Query`.  To force the legacy behavior using window functions,
   specify the ``enable_offset_fetch=False`` dialect parameter to
   :func:`_sa.create_engine`.

The use of ``FETCH FIRST / OFFSET`` may be disabled on any Oracle Database
version by passing ``enable_offset_fetch=False`` to :func:`_sa.create_engine`,
which will force the use of "legacy" mode that makes use of window functions.
This mode is also selected automatically when using a version of Oracle
Database prior to 12c.

When using legacy mode, or when a :class:`.Select` statement with limit/offset
is embedded in a compound statement, an emulated approach for LIMIT / OFFSET
based on window functions is used, which involves creation of a subquery using
``ROW_NUMBER`` that is prone to performance issues as well as SQL construction
issues for complex statements. However, this approach is supported by all
Oracle Database versions. See notes below.

Notes on LIMIT / OFFSET emulation (when fetch() method cannot be used)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If using :meth:`_sql.Select.limit` and :meth:`_sql.Select.offset`, or with the
ORM the :meth:`_orm.Query.limit` and :meth:`_orm.Query.offset` methods on an
Oracle Database version prior to 12c, the following notes apply:

* SQLAlchemy currently makes use of ROWNUM to achieve
  LIMIT/OFFSET; the exact methodology is taken from
  https://blogs.oracle.com/oraclemagazine/on-rownum-and-limiting-results .

* the "FIRST_ROWS()" optimization keyword is not used by default.  To enable
  the usage of this optimization directive, specify ``optimize_limits=True``
  to :func:`_sa.create_engine`.

  .. versionchanged:: 1.4

      The Oracle Database dialect renders limit/offset integer values using a
      "post compile" scheme which renders the integer directly before passing
      the statement to the cursor for execution.  The ``use_binds_for_limits``
      flag no longer has an effect.

      .. seealso::

          :ref:`change_4808`.

.. _oracle_returning:

RETURNING Support
-----------------

Oracle Database supports RETURNING fully for INSERT, UPDATE and DELETE
statements that are invoked with a single collection of bound parameters (that
is, a ``cursor.execute()`` style statement; SQLAlchemy does not generally
support RETURNING with :term:`executemany` statements).  Multiple rows may be
returned as well.

.. versionchanged:: 2.0 the Oracle Database backend has full support for
   RETURNING on parity with other backends.


ON UPDATE CASCADE
-----------------

Oracle Database doesn't have native ON UPDATE CASCADE functionality.  A trigger
based solution is available at
https://web.archive.org/web/20090317041251/https://asktom.oracle.com/tkyte/update_cascade/index.html

When using the SQLAlchemy ORM, the ORM has limited ability to manually issue
cascading updates - specify ForeignKey objects using the
"deferrable=True, initially='deferred'" keyword arguments,
and specify "passive_updates=False" on each relationship().

Oracle Database 8 Compatibility
-------------------------------

.. warning:: The status of Oracle Database 8 compatibility is not known for
   SQLAlchemy 2.0.

When Oracle Database 8 is detected, the dialect internally configures itself to
the following behaviors:

* the use_ansi flag is set to False.  This has the effect of converting all
  JOIN phrases into the WHERE clause, and in the case of LEFT OUTER JOIN
  makes use of Oracle's (+) operator.

* the NVARCHAR2 and NCLOB datatypes are no longer generated as DDL when
  the :class:`~sqlalchemy.types.Unicode` is used - VARCHAR2 and CLOB are issued
  instead. This because these types don't seem to work correctly on Oracle 8
  even though they are available. The :class:`~sqlalchemy.types.NVARCHAR` and
  :class:`~sqlalchemy.dialects.oracle.NCLOB` types will always generate
  NVARCHAR2 and NCLOB.


Synonym/DBLINK Reflection
-------------------------

When using reflection with Table objects, the dialect can optionally search
for tables indicated by synonyms, either in local or remote schemas or
accessed over DBLINK, by passing the flag ``oracle_resolve_synonyms=True`` as
a keyword argument to the :class:`_schema.Table` construct::

    some_table = Table(
        "some_table", autoload_with=some_engine, oracle_resolve_synonyms=True
    )

When this flag is set, the given name (such as ``some_table`` above) will be
searched not just in the ``ALL_TABLES`` view, but also within the
``ALL_SYNONYMS`` view to see if this name is actually a synonym to another
name.  If the synonym is located and refers to a DBLINK, the Oracle Database
dialects know how to locate the table's information using DBLINK syntax(e.g.
``@dblink``).

``oracle_resolve_synonyms`` is accepted wherever reflection arguments are
accepted, including methods such as :meth:`_schema.MetaData.reflect` and
:meth:`_reflection.Inspector.get_columns`.

If synonyms are not in use, this flag should be left disabled.

.. _oracle_constraint_reflection:

Constraint Reflection
---------------------

The Oracle Database dialects can return information about foreign key, unique,
and CHECK constraints, as well as indexes on tables.

Raw information regarding these constraints can be acquired using
:meth:`_reflection.Inspector.get_foreign_keys`,
:meth:`_reflection.Inspector.get_unique_constraints`,
:meth:`_reflection.Inspector.get_check_constraints`, and
:meth:`_reflection.Inspector.get_indexes`.

.. versionchanged:: 1.2 The Oracle Database dialect can now reflect UNIQUE and
   CHECK constraints.

When using reflection at the :class:`_schema.Table` level, the
:class:`_schema.Table`
will also include these constraints.

Note the following caveats:

* When using the :meth:`_reflection.Inspector.get_check_constraints` method,
  Oracle Database builds a special "IS NOT NULL" constraint for columns that
  specify "NOT NULL".  This constraint is **not** returned by default; to
  include the "IS NOT NULL" constraints, pass the flag ``include_all=True``::

      from sqlalchemy import create_engine, inspect

      engine = create_engine(
          "oracle+oracledb://scott:tiger@localhost:1521?service_name=freepdb1"
      )
      inspector = inspect(engine)
      all_check_constraints = inspector.get_check_constraints(
          "some_table", include_all=True
      )

* in most cases, when reflecting a :class:`_schema.Table`, a UNIQUE constraint
  will **not** be available as a :class:`.UniqueConstraint` object, as Oracle
  Database mirrors unique constraints with a UNIQUE index in most cases (the
  exception seems to be when two or more unique constraints represent the same
  columns); the :class:`_schema.Table` will instead represent these using
  :class:`.Index` with the ``unique=True`` flag set.

* Oracle Database creates an implicit index for the primary key of a table;
  this index is **excluded** from all index results.

* the list of columns reflected for an index will not include column names
  that start with SYS_NC.

Table names with SYSTEM/SYSAUX tablespaces
-------------------------------------------

The :meth:`_reflection.Inspector.get_table_names` and
:meth:`_reflection.Inspector.get_temp_table_names`
methods each return a list of table names for the current engine. These methods
are also part of the reflection which occurs within an operation such as
:meth:`_schema.MetaData.reflect`.  By default,
these operations exclude the ``SYSTEM``
and ``SYSAUX`` tablespaces from the operation.   In order to change this, the
default list of tablespaces excluded can be changed at the engine level using
the ``exclude_tablespaces`` parameter::

    # exclude SYSAUX and SOME_TABLESPACE, but not SYSTEM
    e = create_engine(
        "oracle+oracledb://scott:tiger@localhost:1521/?service_name=freepdb1",
        exclude_tablespaces=["SYSAUX", "SOME_TABLESPACE"],
    )

.. _oracle_float_support:

FLOAT / DOUBLE Support and Behaviors
------------------------------------

The SQLAlchemy :class:`.Float` and :class:`.Double` datatypes are generic
datatypes that resolve to the "least surprising" datatype for a given backend.
For Oracle Database, this means they resolve to the ``FLOAT`` and ``DOUBLE``
types::

    >>> from sqlalchemy import cast, literal, Float
    >>> from sqlalchemy.dialects import oracle
    >>> float_datatype = Float()
    >>> print(cast(literal(5.0), float_datatype).compile(dialect=oracle.dialect()))
    CAST(:param_1 AS FLOAT)

Oracle's ``FLOAT`` / ``DOUBLE`` datatypes are aliases for ``NUMBER``.   Oracle
Database stores ``NUMBER`` values with full precision, not floating point
precision, which means that ``FLOAT`` / ``DOUBLE`` do not actually behave like
native FP values. Oracle Database instead offers special datatypes
``BINARY_FLOAT`` and ``BINARY_DOUBLE`` to deliver real 4- and 8- byte FP
values.

SQLAlchemy supports these datatypes directly using :class:`.BINARY_FLOAT` and
:class:`.BINARY_DOUBLE`.   To use the :class:`.Float` or :class:`.Double`
datatypes in a database agnostic way, while allowing Oracle backends to utilize
one of these types, use the :meth:`.TypeEngine.with_variant` method to set up a
variant::

    >>> from sqlalchemy import cast, literal, Float
    >>> from sqlalchemy.dialects import oracle
    >>> float_datatype = Float().with_variant(oracle.BINARY_FLOAT(), "oracle")
    >>> print(cast(literal(5.0), float_datatype).compile(dialect=oracle.dialect()))
    CAST(:param_1 AS BINARY_FLOAT)

E.g. to use this datatype in a :class:`.Table` definition::

    my_table = Table(
        "my_table",
        metadata,
        Column(
            "fp_data", Float().with_variant(oracle.BINARY_FLOAT(), "oracle")
        ),
    )

DateTime Compatibility
----------------------

Oracle Database has no datatype known as ``DATETIME``, it instead has only
``DATE``, which can actually store a date and time value.  For this reason, the
Oracle Database dialects provide a type :class:`_oracle.DATE` which is a
subclass of :class:`.DateTime`.  This type has no special behavior, and is only
present as a "marker" for this type; additionally, when a database column is
reflected and the type is reported as ``DATE``, the time-supporting
:class:`_oracle.DATE` type is used.

.. _oracle_table_options:

Oracle Database Table Options
-----------------------------

The CREATE TABLE phrase supports the following options with Oracle Database
dialects in conjunction with the :class:`_schema.Table` construct:


* ``ON COMMIT``::

    Table(
        "some_table",
        metadata,
        ...,
        prefixes=["GLOBAL TEMPORARY"],
        oracle_on_commit="PRESERVE ROWS",
    )

*
  ``COMPRESS``::

     Table(
         "mytable", metadata, Column("data", String(32)), oracle_compress=True
     )

     Table("mytable", metadata, Column("data", String(32)), oracle_compress=6)

  The ``oracle_compress`` parameter accepts either an integer compression
  level, or ``True`` to use the default compression level.

*
  ``TABLESPACE``::

     Table("mytable", metadata, ..., oracle_tablespace="EXAMPLE_TABLESPACE")

  The ``oracle_tablespace`` parameter specifies the tablespace in which the
  table is to be created. This is useful when you want to create a table in a
  tablespace other than the default tablespace of the user.

  .. versionadded:: 2.0.37

.. _oracle_index_options:

Oracle Database Specific Index Options
--------------------------------------

Bitmap Indexes
~~~~~~~~~~~~~~

You can specify the ``oracle_bitmap`` parameter to create a bitmap index
instead of a B-tree index::

    Index("my_index", my_table.c.data, oracle_bitmap=True)

Bitmap indexes cannot be unique and cannot be compressed. SQLAlchemy will not
check for such limitations, only the database will.

Index compression
~~~~~~~~~~~~~~~~~

Oracle Database has a more efficient storage mode for indexes containing lots
of repeated values. Use the ``oracle_compress`` parameter to turn on key
compression::

    Index("my_index", my_table.c.data, oracle_compress=True)

    Index(
        "my_index",
        my_table.c.data1,
        my_table.c.data2,
        unique=True,
        oracle_compress=1,
    )

The ``oracle_compress`` parameter accepts either an integer specifying the
number of prefix columns to compress, or ``True`` to use the default (all
columns for non-unique indexes, all but the last column for unique indexes).

.. _oracle_vector_datatype:

VECTOR Datatype
---------------

Oracle Database 23ai introduced a new VECTOR datatype for artificial intelligence
and machine learning search operations. The VECTOR datatype is a homogeneous array
of 8-bit signed integers, 8-bit unsigned integers (binary), 32-bit floating-point numbers,
or 64-bit floating-point numbers.

.. seealso::

    `Using VECTOR Data
    <https://python-oracledb.readthedocs.io/en/latest/user_guide/vector_data_type.html>`_ - in the documentation
    for the :ref:`oracledb` driver.

.. versionadded:: 2.0.41

CREATE TABLE support for VECTOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the :class:`.VECTOR` datatype, you can specify the dimension for the data
and the storage format. Valid values for storage format are enum values from
:class:`.VectorStorageFormat`. To create a table that includes a
:class:`.VECTOR` column::

    from sqlalchemy.dialects.oracle import VECTOR, VectorStorageFormat

    t = Table(
        "t1",
        metadata,
        Column("id", Integer, primary_key=True),
        Column(
            "embedding",
            VECTOR(dim=3, storage_format=VectorStorageFormat.FLOAT32),
        ),
        Column(...),
        ...,
    )

Vectors can also be defined with an arbitrary number of dimensions and formats.
This allows you to specify vectors of different dimensions with the various
storage formats mentioned above.

**Examples**

* In this case, the storage format is flexible, allowing any vector type data to be inserted,
  such as INT8 or BINARY etc::

    vector_col: Mapped[array.array] = mapped_column(VECTOR(dim=3))

* The dimension is flexible in this case, meaning that any dimension vector can be used::

    vector_col: Mapped[array.array] = mapped_column(
        VECTOR(storage_format=VectorStorageType.INT8)
    )

* Both the dimensions and the storage format are flexible::

    vector_col: Mapped[array.array] = mapped_column(VECTOR)

Python Datatypes for VECTOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~

VECTOR data can be inserted using Python list or Python ``array.array()`` objects.
Python arrays of type FLOAT (32-bit), DOUBLE (64-bit), or INT (8-bit signed integer)
are used as bind values when inserting VECTOR columns::

    from sqlalchemy import insert, select

    with engine.begin() as conn:
        conn.execute(
            insert(t1),
            {"id": 1, "embedding": [1, 2, 3]},
        )

VECTOR Indexes
~~~~~~~~~~~~~~

The VECTOR feature supports an Oracle-specific parameter ``oracle_vector``
on the :class:`.Index` construct, which allows the construction of VECTOR
indexes.

To utilize VECTOR indexing, set the ``oracle_vector`` parameter to True to use
the default values provided by Oracle. HNSW is the default indexing method::

    from sqlalchemy import Index

    Index(
        "vector_index",
        t1.c.embedding,
        oracle_vector=True,
    )

The full range of parameters for vector indexes are available by using the
:class:`.VectorIndexConfig` dataclass in place of a boolean; this dataclass
allows full configuration of the index::

    Index(
        "hnsw_vector_index",
        t1.c.embedding,
        oracle_vector=VectorIndexConfig(
            index_type=VectorIndexType.HNSW,
            distance=VectorDistanceType.COSINE,
            accuracy=90,
            hnsw_neighbors=5,
            hnsw_efconstruction=20,
            parallel=10,
        ),
    )

    Index(
        "ivf_vector_index",
        t1.c.embedding,
        oracle_vector=VectorIndexConfig(
            index_type=VectorIndexType.IVF,
            distance=VectorDistanceType.DOT,
            accuracy=90,
            ivf_neighbor_partitions=5,
        ),
    )

For complete explanation of these parameters, see the Oracle documentation linked
below.

.. seealso::

    `CREATE VECTOR INDEX <https://www.oracle.com/pls/topic/lookup?ctx=dblatest&id=GUID-B396C369-54BB-4098-A0DD-7C54B3A0D66F>`_ - in the Oracle documentation



Similarity Searching
~~~~~~~~~~~~~~~~~~~~

When using the :class:`_oracle.VECTOR` datatype with a :class:`.Column` or similar
ORM mapped construct, additional comparison functions are available, including:

* ``l2_distance``
* ``cosine_distance``
* ``inner_product``

Example Usage::

    result_vector = connection.scalars(
        select(t1).order_by(t1.embedding.l2_distance([2, 3, 4])).limit(3)
    )

    for user in vector:
        print(user.id, user.embedding)

FETCH APPROXIMATE support
~~~~~~~~~~~~~~~~~~~~~~~~~

Approximate vector search can only be performed when all syntax and semantic
rules are satisfied, the corresponding vector index is available, and the
query optimizer determines to perform it. If any of these conditions are
unmet, then an approximate search is not performed. In this case the query
returns exact results.

To enable approximate searching during similarity searches on VECTORS, the
``oracle_fetch_approximate`` parameter may be used with the :meth:`.Select.fetch`
clause to add ``FETCH APPROX`` to the SELECT statement::

    select(users_table).fetch(5, oracle_fetch_approximate=True)

"""  # noqa

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
from functools import lru_cache
from functools import wraps
import re

from . import dictionary
from .types import _OracleBoolean
from .types import _OracleDate
from .types import BFILE
from .types import BINARY_DOUBLE
from .types import BINARY_FLOAT
from .types import DATE
from .types import FLOAT
from .types import INTERVAL
from .types import LONG
from .types import NCLOB
from .types import NUMBER
from .types import NVARCHAR2  # noqa
from .types import OracleRaw  # noqa
from .types import RAW
from .types import ROWID  # noqa
from .types import TIMESTAMP
from .types import VARCHAR2  # noqa
from .vector import VECTOR
from .vector import VectorIndexConfig
from .vector import VectorIndexType
from ... import Computed
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import default
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import and_
from ...sql import bindparam
from ...sql import compiler
from ...sql import expression
from ...sql import func
from ...sql import null
from ...sql import or_
from ...sql import select
from ...sql import selectable as sa_selectable
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.visitors import InternalTraversal
from ...types import BLOB
from ...types import CHAR
from ...types import CLOB
from ...types import DOUBLE_PRECISION
from ...types import INTEGER
from ...types import NCHAR
from ...types import NVARCHAR
from ...types import REAL
from ...types import VARCHAR

RESERVED_WORDS = set(
    "SHARE RAW DROP BETWEEN FROM DESC OPTION PRIOR LONG THEN "
    "DEFAULT ALTER IS INTO MINUS INTEGER NUMBER GRANT IDENTIFIED "
    "ALL TO ORDER ON FLOAT DATE HAVING CLUSTER NOWAIT RESOURCE "
    "ANY TABLE INDEX FOR UPDATE WHERE CHECK SMALLINT WITH DELETE "
    "BY ASC REVOKE LIKE SIZE RENAME NOCOMPRESS NULL GROUP VALUES "
    "AS IN VIEW EXCLUSIVE COMPRESS SYNONYM SELECT INSERT EXISTS "
    "NOT TRIGGER ELSE CREATE INTERSECT PCTFREE DISTINCT USER "
    "CONNECT SET MODE OF UNIQUE VARCHAR2 VARCHAR LOCK OR CHAR "
    "DECIMAL UNION PUBLIC AND START UID COMMENT CURRENT LEVEL".split()
)

NO_ARG_FNS = set(
    "UID CURRENT_DATE SYSDATE USER CURRENT_TIME CURRENT_TIMESTAMP".split()
)


colspecs = {
    sqltypes.Boolean: _OracleBoolean,
    sqltypes.Interval: INTERVAL,
    sqltypes.DateTime: DATE,
    sqltypes.Date: _OracleDate,
}

ischema_names = {
    "VARCHAR2": VARCHAR,
    "NVARCHAR2": NVARCHAR,
    "CHAR": CHAR,
    "NCHAR": NCHAR,
    "DATE": DATE,
    "NUMBER": NUMBER,
    "BLOB": BLOB,
    "BFILE": BFILE,
    "CLOB": CLOB,
    "NCLOB": NCLOB,
    "TIMESTAMP": TIMESTAMP,
    "TIMESTAMP WITH TIME ZONE": TIMESTAMP,
    "TIMESTAMP WITH LOCAL TIME ZONE": TIMESTAMP,
    "INTERVAL DAY TO SECOND": INTERVAL,
    "RAW": RAW,
    "FLOAT": FLOAT,
    "DOUBLE PRECISION": DOUBLE_PRECISION,
    "REAL": REAL,
    "LONG": LONG,
    "BINARY_DOUBLE": BINARY_DOUBLE,
    "BINARY_FLOAT": BINARY_FLOAT,
    "ROWID": ROWID,
    "VECTOR": VECTOR,
}


class OracleTypeCompiler(compiler.GenericTypeCompiler):
    # Note:
    # Oracle DATE == DATETIME
    # Oracle does not allow milliseconds in DATE
    # Oracle does not support TIME columns

    def visit_datetime(self, type_, **kw):
        return self.visit_DATE(type_, **kw)

    def visit_float(self, type_, **kw):
        return self.visit_FLOAT(type_, **kw)

    def visit_double(self, type_, **kw):
        return self.visit_DOUBLE_PRECISION(type_, **kw)

    def visit_unicode(self, type_, **kw):
        if self.dialect._use_nchar_for_unicode:
            return self.visit_NVARCHAR2(type_, **kw)
        else:
            return self.visit_VARCHAR2(type_, **kw)

    def visit_INTERVAL(self, type_, **kw):
        return "INTERVAL DAY%s TO SECOND%s" % (
            type_.day_precision is not None
            and "(%d)" % type_.day_precision
            or "",
            type_.second_precision is not None
            and "(%d)" % type_.second_precision
            or "",
        )

    def visit_LONG(self, type_, **kw):
        return "LONG"

    def visit_TIMESTAMP(self, type_, **kw):
        if getattr(type_, "local_timezone", False):
            return "TIMESTAMP WITH LOCAL TIME ZONE"
        elif type_.timezone:
            return "TIMESTAMP WITH TIME ZONE"
        else:
            return "TIMESTAMP"

    def visit_DOUBLE_PRECISION(self, type_, **kw):
        return self._generate_numeric(type_, "DOUBLE PRECISION", **kw)

    def visit_BINARY_DOUBLE(self, type_, **kw):
        return self._generate_numeric(type_, "BINARY_DOUBLE", **kw)

    def visit_BINARY_FLOAT(self, type_, **kw):
        return self._generate_numeric(type_, "BINARY_FLOAT", **kw)

    def visit_FLOAT(self, type_, **kw):
        kw["_requires_binary_precision"] = True
        return self._generate_numeric(type_, "FLOAT", **kw)

    def visit_NUMBER(self, type_, **kw):
        return self._generate_numeric(type_, "NUMBER", **kw)

    def _generate_numeric(
        self,
        type_,
        name,
        precision=None,
        scale=None,
        _requires_binary_precision=False,
        **kw,
    ):
        if precision is None:
            precision = getattr(type_, "precision", None)

        if _requires_binary_precision:
            binary_precision = getattr(type_, "binary_precision", None)

            if precision and binary_precision is None:
                # https://www.oracletutorial.com/oracle-basics/oracle-float/
                estimated_binary_precision = int(precision / 0.30103)
                raise exc.ArgumentError(
                    "Oracle Database FLOAT types use 'binary precision', "
                    "which does not convert cleanly from decimal "
                    "'precision'.  Please specify "
                    "this type with a separate Oracle Database variant, such "
                    f"as {type_.__class__.__name__}(precision={precision})."
                    f"with_variant(oracle.FLOAT"
                    f"(binary_precision="
                    f"{estimated_binary_precision}), 'oracle'), so that the "
                    "Oracle Database specific 'binary_precision' may be "
                    "specified accurately."
                )
            else:
                precision = binary_precision

        if scale is None:
            scale = getattr(type_, "scale", None)

        if precision is None:
            return name
        elif scale is None:
            n = "%(name)s(%(precision)s)"
            return n % {"name": name, "precision": precision}
        else:
            n = "%(name)s(%(precision)s, %(scale)s)"
            return n % {"name": name, "precision": precision, "scale": scale}

    def visit_string(self, type_, **kw):
        return self.visit_VARCHAR2(type_, **kw)

    def visit_VARCHAR2(self, type_, **kw):
        return self._visit_varchar(type_, "", "2")

    def visit_NVARCHAR2(self, type_, **kw):
        return self._visit_varchar(type_, "N", "2")

    visit_NVARCHAR = visit_NVARCHAR2

    def visit_VARCHAR(self, type_, **kw):
        return self._visit_varchar(type_, "", "")

    def _visit_varchar(self, type_, n, num):
        if not type_.length:
            return "%(n)sVARCHAR%(two)s" % {"two": num, "n": n}
        elif not n and self.dialect._supports_char_length:
            varchar = "VARCHAR%(two)s(%(length)s CHAR)"
            return varchar % {"length": type_.length, "two": num}
        else:
            varchar = "%(n)sVARCHAR%(two)s(%(length)s)"
            return varchar % {"length": type_.length, "two": num, "n": n}

    def visit_text(self, type_, **kw):
        return self.visit_CLOB(type_, **kw)

    def visit_unicode_text(self, type_, **kw):
        if self.dialect._use_nchar_for_unicode:
            return self.visit_NCLOB(type_, **kw)
        else:
            return self.visit_CLOB(type_, **kw)

    def visit_large_binary(self, type_, **kw):
        return self.visit_BLOB(type_, **kw)

    def visit_big_integer(self, type_, **kw):
        return self.visit_NUMBER(type_, precision=19, **kw)

    def visit_boolean(self, type_, **kw):
        return self.visit_SMALLINT(type_, **kw)

    def visit_RAW(self, type_, **kw):
        if type_.length:
            return "RAW(%(length)s)" % {"length": type_.length}
        else:
            return "RAW"

    def visit_ROWID(self, type_, **kw):
        return "ROWID"

    def visit_VECTOR(self, type_, **kw):
        if type_.dim is None and type_.storage_format is None:
            return "VECTOR(*,*)"
        elif type_.storage_format is None:
            return f"VECTOR({type_.dim},*)"
        elif type_.dim is None:
            return f"VECTOR(*,{type_.storage_format.value})"
        else:
            return f"VECTOR({type_.dim},{type_.storage_format.value})"


class OracleCompiler(compiler.SQLCompiler):
    """Oracle compiler modifies the lexical structure of Select
    statements to work under non-ANSI configured Oracle databases, if
    the use_ansi flag is False.
    """

    compound_keywords = util.update_copy(
        compiler.SQLCompiler.compound_keywords,
        {expression.CompoundSelect.EXCEPT: "MINUS"},
    )

    def __init__(self, *args, **kwargs):
        self.__wheres = {}
        super().__init__(*args, **kwargs)

    def visit_mod_binary(self, binary, operator, **kw):
        return "mod(%s, %s)" % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    def visit_now_func(self, fn, **kw):
        return "CURRENT_TIMESTAMP"

    def visit_char_length_func(self, fn, **kw):
        return "LENGTH" + self.function_argspec(fn, **kw)

    def visit_match_op_binary(self, binary, operator, **kw):
        return "CONTAINS (%s, %s)" % (
            self.process(binary.left),
            self.process(binary.right),
        )

    def visit_true(self, expr, **kw):
        return "1"

    def visit_false(self, expr, **kw):
        return "0"

    def get_cte_preamble(self, recursive):
        return "WITH"

    def get_select_hint_text(self, byfroms):
        return " ".join("/*+ %s */" % text for table, text in byfroms.items())

    def function_argspec(self, fn, **kw):
        if len(fn.clauses) > 0 or fn.name.upper() not in NO_ARG_FNS:
            return compiler.SQLCompiler.function_argspec(self, fn, **kw)
        else:
            return ""

    def visit_function(self, func, **kw):
        text = super().visit_function(func, **kw)
        if kw.get("asfrom", False) and func.name.lower() != "table":
            text = "TABLE (%s)" % text
        return text

    def visit_table_valued_column(self, element, **kw):
        text = super().visit_table_valued_column(element, **kw)
        text = text + ".COLUMN_VALUE"
        return text

    def default_from(self):
        """Called when a ``SELECT`` statement has no froms,
        and no ``FROM`` clause is to be appended.

        The Oracle compiler tacks a "FROM DUAL" to the statement.
        """

        return " FROM DUAL"

    def visit_join(self, join, from_linter=None, **kwargs):
        if self.dialect.use_ansi:
            return compiler.SQLCompiler.visit_join(
                self, join, from_linter=from_linter, **kwargs
            )
        else:
            if from_linter:
                from_linter.edges.add((join.left, join.right))

            kwargs["asfrom"] = True
            if isinstance(join.right, expression.FromGrouping):
                right = join.right.element
            else:
                right = join.right
            return (
                self.process(join.left, from_linter=from_linter, **kwargs)
                + ", "
                + self.process(right, from_linter=from_linter, **kwargs)
            )

    def _get_nonansi_join_whereclause(self, froms):
        clauses = []

        def visit_join(join):
            if join.isouter:
                # https://docs.oracle.com/database/121/SQLRF/queries006.htm#SQLRF52354
                # "apply the outer join operator (+) to all columns of B in
                # the join condition in the WHERE clause" - that is,
                # unconditionally regardless of operator or the other side
                def visit_binary(binary):
                    if isinstance(
                        binary.left, expression.ColumnClause
                    ) and join.right.is_derived_from(binary.left.table):
                        binary.left = _OuterJoinColumn(binary.left)
                    elif isinstance(
                        binary.right, expression.ColumnClause
                    ) and join.right.is_derived_from(binary.right.table):
                        binary.right = _OuterJoinColumn(binary.right)

                clauses.append(
                    visitors.cloned_traverse(
                        join.onclause, {}, {"binary": visit_binary}
                    )
                )
            else:
                clauses.append(join.onclause)

            for j in join.left, join.right:
                if isinstance(j, expression.Join):
                    visit_join(j)
                elif isinstance(j, expression.FromGrouping):
                    visit_join(j.element)

        for f in froms:
            if isinstance(f, expression.Join):
                visit_join(f)

        if not clauses:
            return None
        else:
            return sql.and_(*clauses)

    def visit_outer_join_column(self, vc, **kw):
        return self.process(vc.column, **kw) + "(+)"

    def visit_sequence(self, seq, **kw):
        return self.preparer.format_sequence(seq) + ".nextval"

    def get_render_as_alias_suffix(self, alias_name_text):
        """Oracle doesn't like ``FROM table AS alias``"""

        return " " + alias_name_text

    def returning_clause(
        self, stmt, returning_cols, *, populate_result_map, **kw
    ):
        columns = []
        binds = []

        for i, column in enumerate(
            expression._select_iterables(returning_cols)
        ):
            if (
                self.isupdate
                and isinstance(column, sa_schema.Column)
                and isinstance(column.server_default, Computed)
                and not self.dialect._supports_update_returning_computed_cols
            ):
                util.warn(
                    "Computed columns don't work with Oracle Database UPDATE "
                    "statements that use RETURNING; the value of the column "
                    "*before* the UPDATE takes place is returned.   It is "
                    "advised to not use RETURNING with an Oracle Database "
                    "computed column.  Consider setting implicit_returning "
                    "to False on the Table object in order to avoid implicit "
                    "RETURNING clauses from being generated for this Table."
                )
            if column.type._has_column_expression:
                col_expr = column.type.column_expression(column)
            else:
                col_expr = column

            outparam = sql.outparam("ret_%d" % i, type_=column.type)
            self.binds[outparam.key] = outparam
            binds.append(
                self.bindparam_string(self._truncate_bindparam(outparam))
            )

            # has_out_parameters would in a normal case be set to True
            # as a result of the compiler visiting an outparam() object.
            # in this case, the above outparam() objects are not being
            # visited.   Ensure the statement itself didn't have other
            # outparam() objects independently.
            # technically, this could be supported, but as it would be
            # a very strange use case without a clear rationale, disallow it
            if self.has_out_parameters:
                raise exc.InvalidRequestError(
                    "Using explicit outparam() objects with "
                    "UpdateBase.returning() in the same Core DML statement "
                    "is not supported in the Oracle Database dialects."
                )

            self._oracle_returning = True

            columns.append(self.process(col_expr, within_columns_clause=False))
            if populate_result_map:
                self._add_to_result_map(
                    getattr(col_expr, "name", col_expr._anon_name_label),
                    getattr(col_expr, "name", col_expr._anon_name_label),
                    (
                        column,
                        getattr(column, "name", None),
                        getattr(column, "key", None),
                    ),
                    column.type,
                )

        return "RETURNING " + ", ".join(columns) + " INTO " + ", ".join(binds)

    def _row_limit_clause(self, select, **kw):
        """Oracle Database 12c supports OFFSET/FETCH operators
        Use it instead subquery with row_number

        """

        if (
            select._fetch_clause is not None
            or not self.dialect._supports_offset_fetch
        ):
            return super()._row_limit_clause(
                select, use_literal_execute_for_simple_int=True, **kw
            )
        else:
            return self.fetch_clause(
                select,
                fetch_clause=self._get_limit_or_fetch(select),
                use_literal_execute_for_simple_int=True,
                **kw,
            )

    def _get_limit_or_fetch(self, select):
        if select._fetch_clause is None:
            return select._limit_clause
        else:
            return select._fetch_clause

    def fetch_clause(
        self,
        select,
        fetch_clause=None,
        require_offset=False,
        use_literal_execute_for_simple_int=False,
        **kw,
    ):
        text = super().fetch_clause(
            select,
            fetch_clause=fetch_clause,
            require_offset=require_offset,
            use_literal_execute_for_simple_int=(
                use_literal_execute_for_simple_int
            ),
            **kw,
        )

        if select.dialect_options["oracle"]["fetch_approximate"]:
            text = re.sub("FETCH FIRST", "FETCH APPROX FIRST", text)

        return text

    def translate_select_structure(self, select_stmt, **kwargs):
        select = select_stmt

        if not getattr(select, "_oracle_visit", None):
            if not self.dialect.use_ansi:
                froms = self._display_froms_for_select(
                    select, kwargs.get("asfrom", False)
                )
                whereclause = self._get_nonansi_join_whereclause(froms)
                if whereclause is not None:
                    select = select.where(whereclause)
                    select._oracle_visit = True

            # if fetch is used this is not needed
            if (
                select._has_row_limiting_clause
                and not self.dialect._supports_offset_fetch
                and select._fetch_clause is None
            ):
                limit_clause = select._limit_clause
                offset_clause = select._offset_clause

                if select._simple_int_clause(limit_clause):
                    limit_clause = limit_clause.render_literal_execute()

                if select._simple_int_clause(offset_clause):
                    offset_clause = offset_clause.render_literal_execute()

                # currently using form at:
                # https://blogs.oracle.com/oraclemagazine/\
                # on-rownum-and-limiting-results

                orig_select = select
                select = select._generate()
                select._oracle_visit = True

                # add expressions to accommodate FOR UPDATE OF
                for_update = select._for_update_arg
                if for_update is not None and for_update.of:
                    for_update = for_update._clone()
                    for_update._copy_internals()

                    for elem in for_update.of:
                        if not select.selected_columns.contains_column(elem):
                            select = select.add_columns(elem)

                # Wrap the middle select and add the hint
                inner_subquery = select.alias()
                limitselect = sql.select(
                    *[
                        c
                        for c in inner_subquery.c
                        if orig_select.selected_columns.corresponding_column(c)
                        is not None
                    ]
                )

                if (
                    limit_clause is not None
                    and self.dialect.optimize_limits
                    and select._simple_int_clause(limit_clause)
                ):
                    limitselect = limitselect.prefix_with(
                        expression.text(
                            "/*+ FIRST_ROWS(%s) */"
                            % self.process(limit_clause, **kwargs)
                        )
                    )

                limitselect._oracle_visit = True
                limitselect._is_wrapper = True

                # add expressions to accommodate FOR UPDATE OF
                if for_update is not None and for_update.of:
                    adapter = sql_util.ClauseAdapter(inner_subquery)
                    for_update.of = [
                        adapter.traverse(elem) for elem in for_update.of
                    ]

                # If needed, add the limiting clause
                if limit_clause is not None:
                    if select._simple_int_clause(limit_clause) and (
                        offset_clause is None
                        or select._simple_int_clause(offset_clause)
                    ):
                        max_row = limit_clause

                        if offset_clause is not None:
                            max_row = max_row + offset_clause

                    else:
                        max_row = limit_clause

                        if offset_clause is not None:
                            max_row = max_row + offset_clause
                    limitselect = limitselect.where(
                        sql.literal_column("ROWNUM") <= max_row
                    )

                # If needed, add the ora_rn, and wrap again with offset.
                if offset_clause is None:
                    limitselect._for_update_arg = for_update
                    select = limitselect
                else:
                    limitselect = limitselect.add_columns(
                        sql.literal_column("ROWNUM").label("ora_rn")
                    )
                    limitselect._oracle_visit = True
                    limitselect._is_wrapper = True

                    if for_update is not None and for_update.of:
                        limitselect_cols = limitselect.selected_columns
                        for elem in for_update.of:
                            if (
                                limitselect_cols.corresponding_column(elem)
                                is None
                            ):
                                limitselect = limitselect.add_columns(elem)

                    limit_subquery = limitselect.alias()
                    origselect_cols = orig_select.selected_columns
                    offsetselect = sql.select(
                        *[
                            c
                            for c in limit_subquery.c
                            if origselect_cols.corresponding_column(c)
                            is not None
                        ]
                    )

                    offsetselect._oracle_visit = True
                    offsetselect._is_wrapper = True

                    if for_update is not None and for_update.of:
                        adapter = sql_util.ClauseAdapter(limit_subquery)
                        for_update.of = [
                            adapter.traverse(elem) for elem in for_update.of
                        ]

                    offsetselect = offsetselect.where(
                        sql.literal_column("ora_rn") > offset_clause
                    )

                    offsetselect._for_update_arg = for_update
                    select = offsetselect

        return select

    def limit_clause(self, select, **kw):
        return ""

    def visit_empty_set_expr(self, type_, **kw):
        return "SELECT 1 FROM DUAL WHERE 1!=1"

    def for_update_clause(self, select, **kw):
        if self.is_subquery():
            return ""

        tmp = " FOR UPDATE"

        if select._for_update_arg.of:
            tmp += " OF " + ", ".join(
                self.process(elem, **kw) for elem in select._for_update_arg.of
            )

        if select._for_update_arg.nowait:
            tmp += " NOWAIT"
        if select._for_update_arg.skip_locked:
            tmp += " SKIP LOCKED"

        return tmp

    def visit_is_distinct_from_binary(self, binary, operator, **kw):
        return "DECODE(%s, %s, 0, 1) = 1" % (
            self.process(binary.left),
            self.process(binary.right),
        )

    def visit_is_not_distinct_from_binary(self, binary, operator, **kw):
        return "DECODE(%s, %s, 0, 1) = 0" % (
            self.process(binary.left),
            self.process(binary.right),
        )

    def visit_regexp_match_op_binary(self, binary, operator, **kw):
        string = self.process(binary.left, **kw)
        pattern = self.process(binary.right, **kw)
        flags = binary.modifiers["flags"]
        if flags is None:
            return "REGEXP_LIKE(%s, %s)" % (string, pattern)
        else:
            return "REGEXP_LIKE(%s, %s, %s)" % (
                string,
                pattern,
                self.render_literal_value(flags, sqltypes.STRINGTYPE),
            )

    def visit_not_regexp_match_op_binary(self, binary, operator, **kw):
        return "NOT %s" % self.visit_regexp_match_op_binary(
            binary, operator, **kw
        )

    def visit_regexp_replace_op_binary(self, binary, operator, **kw):
        string = self.process(binary.left, **kw)
        pattern_replace = self.process(binary.right, **kw)
        flags = binary.modifiers["flags"]
        if flags is None:
            return "REGEXP_REPLACE(%s, %s)" % (
                string,
                pattern_replace,
            )
        else:
            return "REGEXP_REPLACE(%s, %s, %s)" % (
                string,
                pattern_replace,
                self.render_literal_value(flags, sqltypes.STRINGTYPE),
            )

    def visit_aggregate_strings_func(self, fn, **kw):
        return "LISTAGG%s" % self.function_argspec(fn, **kw)

    def _visit_bitwise(self, binary, fn_name, custom_right=None, **kw):
        left = self.process(binary.left, **kw)
        right = self.process(
            custom_right if custom_right is not None else binary.right, **kw
        )
        return f"{fn_name}({left}, {right})"

    def visit_bitwise_xor_op_binary(self, binary, operator, **kw):
        return self._visit_bitwise(binary, "BITXOR", **kw)

    def visit_bitwise_or_op_binary(self, binary, operator, **kw):
        return self._visit_bitwise(binary, "BITOR", **kw)

    def visit_bitwise_and_op_binary(self, binary, operator, **kw):
        return self._visit_bitwise(binary, "BITAND", **kw)

    def visit_bitwise_rshift_op_binary(self, binary, operator, **kw):
        raise exc.CompileError("Cannot compile bitwise_rshift in oracle")

    def visit_bitwise_lshift_op_binary(self, binary, operator, **kw):
        raise exc.CompileError("Cannot compile bitwise_lshift in oracle")

    def visit_bitwise_not_op_unary_operator(self, element, operator, **kw):
        raise exc.CompileError("Cannot compile bitwise_not in oracle")


class OracleDDLCompiler(compiler.DDLCompiler):

    def _build_vector_index_config(
        self, vector_index_config: VectorIndexConfig
    ) -> str:
        parts = []
        sql_param_name = {
            "hnsw_neighbors": "neighbors",
            "hnsw_efconstruction": "efconstruction",
            "ivf_neighbor_partitions": "neighbor partitions",
            "ivf_sample_per_partition": "sample_per_partition",
            "ivf_min_vectors_per_partition": "min_vectors_per_partition",
        }
        if vector_index_config.index_type == VectorIndexType.HNSW:
            parts.append("ORGANIZATION INMEMORY NEIGHBOR GRAPH")
        elif vector_index_config.index_type == VectorIndexType.IVF:
            parts.append("ORGANIZATION NEIGHBOR PARTITIONS")
        if vector_index_config.distance is not None:
            parts.append(f"DISTANCE {vector_index_config.distance.value}")

        if vector_index_config.accuracy is not None:
            parts.append(
                f"WITH TARGET ACCURACY {vector_index_config.accuracy}"
            )

        parameters_str = [f"type {vector_index_config.index_type.name}"]
        prefix = vector_index_config.index_type.name.lower() + "_"

        for field in fields(vector_index_config):
            if field.name.startswith(prefix):
                key = sql_param_name.get(field.name)
                value = getattr(vector_index_config, field.name)
                if value is not None:
                    parameters_str.append(f"{key} {value}")

        parameters_str = ", ".join(parameters_str)
        parts.append(f"PARAMETERS ({parameters_str})")

        if vector_index_config.parallel is not None:
            parts.append(f"PARALLEL {vector_index_config.parallel}")

        return " ".join(parts)

    def define_constraint_cascades(self, constraint):
        text = ""
        if constraint.ondelete is not None:
            text += " ON DELETE %s" % constraint.ondelete

        # oracle has no ON UPDATE CASCADE -
        # its only available via triggers
        # https://web.archive.org/web/20090317041251/https://asktom.oracle.com/tkyte/update_cascade/index.html
        if constraint.onupdate is not None:
            util.warn(
                "Oracle Database does not contain native UPDATE CASCADE "
                "functionality - onupdates will not be rendered for foreign "
                "keys.  Consider using deferrable=True, initially='deferred' "
                "or triggers."
            )

        return text

    def visit_drop_table_comment(self, drop, **kw):
        return "COMMENT ON TABLE %s IS ''" % self.preparer.format_table(
            drop.element
        )

    def visit_create_index(self, create, **kw):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = "CREATE "
        if index.unique:
            text += "UNIQUE "
        if index.dialect_options["oracle"]["bitmap"]:
            text += "BITMAP "
        vector_options = index.dialect_options["oracle"]["vector"]
        if vector_options:
            text += "VECTOR "
        text += "INDEX %s ON %s (%s)" % (
            self._prepared_index_name(index, include_schema=True),
            preparer.format_table(index.table, use_schema=True),
            ", ".join(
                self.sql_compiler.process(
                    expr, include_table=False, literal_binds=True
                )
                for expr in index.expressions
            ),
        )
        if index.dialect_options["oracle"]["compress"] is not False:
            if index.dialect_options["oracle"]["compress"] is True:
                text += " COMPRESS"
            else:
                text += " COMPRESS %d" % (
                    index.dialect_options["oracle"]["compress"]
                )
        if vector_options:
            if vector_options is True:
                vector_options = VectorIndexConfig()

            text += " " + self._build_vector_index_config(vector_options)
        return text

    def post_create_table(self, table):
        table_opts = []
        opts = table.dialect_options["oracle"]

        if opts["on_commit"]:
            on_commit_options = opts["on_commit"].replace("_", " ").upper()
            table_opts.append("\n ON COMMIT %s" % on_commit_options)

        if opts["compress"]:
            if opts["compress"] is True:
                table_opts.append("\n COMPRESS")
            else:
                table_opts.append("\n COMPRESS FOR %s" % (opts["compress"]))
        if opts["tablespace"]:
            table_opts.append(
                "\n TABLESPACE %s" % self.preparer.quote(opts["tablespace"])
            )
        return "".join(table_opts)

    def get_identity_options(self, identity_options):
        text = super().get_identity_options(identity_options)
        text = text.replace("NO MINVALUE", "NOMINVALUE")
        text = text.replace("NO MAXVALUE", "NOMAXVALUE")
        text = text.replace("NO CYCLE", "NOCYCLE")
        if identity_options.order is not None:
            text += " ORDER" if identity_options.order else " NOORDER"
        return text.strip()

    def visit_computed_column(self, generated, **kw):
        text = "GENERATED ALWAYS AS (%s)" % self.sql_compiler.process(
            generated.sqltext, include_table=False, literal_binds=True
        )
        if generated.persisted is True:
            raise exc.CompileError(
                "Oracle Database computed columns do not support 'stored' "
                "persistence; set the 'persisted' flag to None or False for "
                "Oracle Database support."
            )
        elif generated.persisted is False:
            text += " VIRTUAL"
        return text

    def visit_identity_column(self, identity, **kw):
        if identity.always is None:
            kind = ""
        else:
            kind = "ALWAYS" if identity.always else "BY DEFAULT"
        text = "GENERATED %s" % kind
        if identity.on_null:
            text += " ON NULL"
        text += " AS IDENTITY"
        options = self.get_identity_options(identity)
        if options:
            text += " (%s)" % options
        return text


class OracleIdentifierPreparer(compiler.IdentifierPreparer):
    reserved_words = {x.lower() for x in RESERVED_WORDS}
    illegal_initial_characters = {str(dig) for dig in range(0, 10)}.union(
        ["_", "$"]
    )

    def _bindparam_requires_quotes(self, value):
        """Return True if the given identifier requires quoting."""
        lc_value = value.lower()
        return (
            lc_value in self.reserved_words
            or value[0] in self.illegal_initial_characters
            or not self.legal_characters.match(str(value))
        )

    def format_savepoint(self, savepoint):
        name = savepoint.ident.lstrip("_")
        return super().format_savepoint(savepoint, name)


class OracleExecutionContext(default.DefaultExecutionContext):
    def fire_sequence(self, seq, type_):
        return self._execute_scalar(
            "SELECT "
            + self.identifier_preparer.format_sequence(seq)
            + ".nextval FROM DUAL",
            type_,
        )

    def pre_exec(self):
        if self.statement and "_oracle_dblink" in self.execution_options:
            self.statement = self.statement.replace(
                dictionary.DB_LINK_PLACEHOLDER,
                self.execution_options["_oracle_dblink"],
            )


class OracleDialect(default.DefaultDialect):
    name = "oracle"
    supports_statement_cache = True
    supports_alter = True
    max_identifier_length = 128

    _supports_offset_fetch = True

    insert_returning = True
    update_returning = True
    delete_returning = True

    div_is_floordiv = False

    supports_simple_order_by_label = False
    cte_follows_insert = True
    returns_native_bytes = True

    supports_sequences = True
    sequences_optional = False
    postfetch_lastrowid = False

    default_paramstyle = "named"
    colspecs = colspecs
    ischema_names = ischema_names
    requires_name_normalize = True

    supports_comments = True

    supports_default_values = False
    supports_default_metavalue = True
    supports_empty_insert = False
    supports_identity_columns = True

    statement_compiler = OracleCompiler
    ddl_compiler = OracleDDLCompiler
    type_compiler_cls = OracleTypeCompiler
    preparer = OracleIdentifierPreparer
    execution_ctx_cls = OracleExecutionContext

    reflection_options = ("oracle_resolve_synonyms",)

    _use_nchar_for_unicode = False

    construct_arguments = [
        (
            sa_schema.Table,
            {
                "resolve_synonyms": False,
                "on_commit": None,
                "compress": False,
                "tablespace": None,
            },
        ),
        (
            sa_schema.Index,
            {
                "bitmap": False,
                "compress": False,
                "vector": False,
            },
        ),
        (sa_selectable.Select, {"fetch_approximate": False}),
        (sa_selectable.CompoundSelect, {"fetch_approximate": False}),
    ]

    @util.deprecated_params(
        use_binds_for_limits=(
            "1.4",
            "The ``use_binds_for_limits`` Oracle Database dialect parameter "
            "is deprecated. The dialect now renders LIMIT / OFFSET integers "
            "inline in all cases using a post-compilation hook, so that the "
            "value is still represented by a 'bound parameter' on the Core "
            "Expression side.",
        )
    )
    def __init__(
        self,
        use_ansi=True,
        optimize_limits=False,
        use_binds_for_limits=None,
        use_nchar_for_unicode=False,
        exclude_tablespaces=("SYSTEM", "SYSAUX"),
        enable_offset_fetch=True,
        **kwargs,
    ):
        default.DefaultDialect.__init__(self, **kwargs)
        self._use_nchar_for_unicode = use_nchar_for_unicode
        self.use_ansi = use_ansi
        self.optimize_limits = optimize_limits
        self.exclude_tablespaces = exclude_tablespaces
        self.enable_offset_fetch = self._supports_offset_fetch = (
            enable_offset_fetch
        )

    def initialize(self, connection):
        super().initialize(connection)

        # Oracle 8i has RETURNING:
        # https://docs.oracle.com/cd/A87860_01/doc/index.htm

        # so does Oracle8:
        # https://docs.oracle.com/cd/A64702_01/doc/index.htm

        if self._is_oracle_8:
            self.colspecs = self.colspecs.copy()
            self.colspecs.pop(sqltypes.Interval)
            self.use_ansi = False

        self.supports_identity_columns = self.server_version_info >= (12,)
        self._supports_offset_fetch = (
            self.enable_offset_fetch and self.server_version_info >= (12,)
        )

    def _get_effective_compat_server_version_info(self, connection):
        # dialect does not need compat levels below 12.2, so don't query
        # in those cases

        if self.server_version_info < (12, 2):
            return self.server_version_info
        try:
            compat = connection.exec_driver_sql(
                "SELECT value FROM v$parameter WHERE name = 'compatible'"
            ).scalar()
        except exc.DBAPIError:
            compat = None

        if compat:
            try:
                return tuple(int(x) for x in compat.split("."))
            except:
                return self.server_version_info
        else:
            return self.server_version_info

    @property
    def _is_oracle_8(self):
        return self.server_version_info and self.server_version_info < (9,)

    @property
    def _supports_table_compression(self):
        return self.server_version_info and self.server_version_info >= (10, 1)

    @property
    def _supports_table_compress_for(self):
        return self.server_version_info and self.server_version_info >= (11,)

    @property
    def _supports_char_length(self):
        return not self._is_oracle_8

    @property
    def _supports_update_returning_computed_cols(self):
        # on version 18 this error is no longet present while it happens on 11
        # it may work also on versions before the 18
        return self.server_version_info and self.server_version_info >= (18,)

    @property
    def _supports_except_all(self):
        return self.server_version_info and self.server_version_info >= (21,)

    def do_release_savepoint(self, connection, name):
        # Oracle does not support RELEASE SAVEPOINT
        pass

    def _check_max_identifier_length(self, connection):
        if self._get_effective_compat_server_version_info(connection) < (
            12,
            2,
        ):
            return 30
        else:
            # use the default
            return None

    def get_isolation_level_values(self, dbapi_connection):
        return ["READ COMMITTED", "SERIALIZABLE"]

    def get_default_isolation_level(self, dbapi_conn):
        try:
            return self.get_isolation_level(dbapi_conn)
        except NotImplementedError:
            raise
        except:
            return "READ COMMITTED"

    def _execute_reflection(
        self, connection, query, dblink, returns_long, params=None
    ):
        if dblink and not dblink.startswith("@"):
            dblink = f"@{dblink}"
        execution_options = {
            # handle db links
            "_oracle_dblink": dblink or "",
            # override any schema translate map
            "schema_translate_map": None,
        }

        if dblink and returns_long:
            # Oracle seems to error with
            # "ORA-00997: illegal use of LONG datatype" when returning
            # LONG columns via a dblink in a query with bind params
            # This type seems to be very hard to cast into something else
            # so it seems easier to just use bind param in this case
            def visit_bindparam(bindparam):
                bindparam.literal_execute = True

            query = visitors.cloned_traverse(
                query, {}, {"bindparam": visit_bindparam}
            )
        return connection.execute(
            query, params, execution_options=execution_options
        )

    @util.memoized_property
    def _has_table_query(self):
        # materialized views are returned by all_tables
        tables = (
            select(
                dictionary.all_tables.c.table_name,
                dictionary.all_tables.c.owner,
            )
            .union_all(
                select(
                    dictionary.all_views.c.view_name.label("table_name"),
                    dictionary.all_views.c.owner,
                )
            )
            .subquery("tables_and_views")
        )

        query = select(tables.c.table_name).where(
            tables.c.table_name == bindparam("table_name"),
            tables.c.owner == bindparam("owner"),
        )
        return query

    @reflection.cache
    def has_table(
        self, connection, table_name, schema=None, dblink=None, **kw
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        self._ensure_has_table_connection(connection)

        if not schema:
            schema = self.default_schema_name

        params = {
            "table_name": self.denormalize_name(table_name),
            "owner": self.denormalize_schema_name(schema),
        }
        cursor = self._execute_reflection(
            connection,
            self._has_table_query,
            dblink,
            returns_long=False,
            params=params,
        )
        return bool(cursor.scalar())

    @reflection.cache
    def has_sequence(
        self, connection, sequence_name, schema=None, dblink=None, **kw
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if not schema:
            schema = self.default_schema_name

        query = select(dictionary.all_sequences.c.sequence_name).where(
            dictionary.all_sequences.c.sequence_name
            == self.denormalize_schema_name(sequence_name),
            dictionary.all_sequences.c.sequence_owner
            == self.denormalize_schema_name(schema),
        )

        cursor = self._execute_reflection(
            connection, query, dblink, returns_long=False
        )
        return bool(cursor.scalar())

    def _get_default_schema_name(self, connection):
        return self.normalize_name(
            connection.exec_driver_sql(
                "select sys_context( 'userenv', 'current_schema' ) from dual"
            ).scalar()
        )

    def denormalize_schema_name(self, name):
        # look for quoted_name
        force = getattr(name, "quote", None)
        if force is None and name == "public":
            # look for case insensitive, no quoting specified, "public"
            return "PUBLIC"
        return super().denormalize_name(name)

    @reflection.flexi_cache(
        ("schema", InternalTraversal.dp_string),
        ("filter_names", InternalTraversal.dp_string_list),
        ("dblink", InternalTraversal.dp_string),
    )
    def _get_synonyms(self, connection, schema, filter_names, dblink, **kw):
        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )

        has_filter_names, params = self._prepare_filter_names(filter_names)
        query = select(
            dictionary.all_synonyms.c.synonym_name,
            dictionary.all_synonyms.c.table_name,
            dictionary.all_synonyms.c.table_owner,
            dictionary.all_synonyms.c.db_link,
        ).where(dictionary.all_synonyms.c.owner == owner)
        if has_filter_names:
            query = query.where(
                dictionary.all_synonyms.c.synonym_name.in_(
                    params["filter_names"]
                )
            )
        result = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).mappings()
        return result.all()

    @lru_cache()
    def _all_objects_query(
        self, owner, scope, kind, has_filter_names, has_mat_views
    ):
        query = (
            select(dictionary.all_objects.c.object_name)
            .select_from(dictionary.all_objects)
            .where(dictionary.all_objects.c.owner == owner)
        )

        # NOTE: materialized views are listed in all_objects twice;
        # once as MATERIALIZE VIEW and once as TABLE
        if kind is ObjectKind.ANY:
            # materilaized view are listed also as tables so there is no
            # need to add them to the in_.
            query = query.where(
                dictionary.all_objects.c.object_type.in_(("TABLE", "VIEW"))
            )
        else:
            object_type = []
            if ObjectKind.VIEW in kind:
                object_type.append("VIEW")
            if (
                ObjectKind.MATERIALIZED_VIEW in kind
                and ObjectKind.TABLE not in kind
            ):
                # materilaized view are listed also as tables so there is no
                # need to add them to the in_ if also selecting tables.
                object_type.append("MATERIALIZED VIEW")
            if ObjectKind.TABLE in kind:
                object_type.append("TABLE")
                if has_mat_views and ObjectKind.MATERIALIZED_VIEW not in kind:
                    # materialized view are listed also as tables,
                    # so they need to be filtered out
                    # EXCEPT ALL / MINUS profiles as faster than using
                    # NOT EXISTS or NOT IN with a subquery, but it's in
                    # general faster to get the mat view names and exclude
                    # them only when needed
                    query = query.where(
                        dictionary.all_objects.c.object_name.not_in(
                            bindparam("mat_views")
                        )
                    )
            query = query.where(
                dictionary.all_objects.c.object_type.in_(object_type)
            )

        # handles scope
        if scope is ObjectScope.DEFAULT:
            query = query.where(dictionary.all_objects.c.temporary == "N")
        elif scope is ObjectScope.TEMPORARY:
            query = query.where(dictionary.all_objects.c.temporary == "Y")

        if has_filter_names:
            query = query.where(
                dictionary.all_objects.c.object_name.in_(
                    bindparam("filter_names")
                )
            )
        return query

    @reflection.flexi_cache(
        ("schema", InternalTraversal.dp_string),
        ("scope", InternalTraversal.dp_plain_obj),
        ("kind", InternalTraversal.dp_plain_obj),
        ("filter_names", InternalTraversal.dp_string_list),
        ("dblink", InternalTraversal.dp_string),
    )
    def _get_all_objects(
        self, connection, schema, scope, kind, filter_names, dblink, **kw
    ):
        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )

        has_filter_names, params = self._prepare_filter_names(filter_names)
        has_mat_views = False
        if (
            ObjectKind.TABLE in kind
            and ObjectKind.MATERIALIZED_VIEW not in kind
        ):
            # see note in _all_objects_query
            mat_views = self.get_materialized_view_names(
                connection, schema, dblink, _normalize=False, **kw
            )
            if mat_views:
                params["mat_views"] = mat_views
                has_mat_views = True

        query = self._all_objects_query(
            owner, scope, kind, has_filter_names, has_mat_views
        )

        result = self._execute_reflection(
            connection, query, dblink, returns_long=False, params=params
        ).scalars()

        return result.all()

    def _handle_synonyms_decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            return self._handle_synonyms(fn, *args, **kwargs)

        return wrapper

    def _handle_synonyms(self, fn, connection, *args, **kwargs):
        if not kwargs.get("oracle_resolve_synonyms", False):
            return fn(self, connection, *args, **kwargs)

        original_kw = kwargs.copy()
        schema = kwargs.pop("schema", None)
        result = self._get_synonyms(
            connection,
            schema=schema,
            filter_names=kwargs.pop("filter_names", None),
            dblink=kwargs.pop("dblink", None),
            info_cache=kwargs.get("info_cache", None),
        )

        dblinks_owners = defaultdict(dict)
        for row in result:
            key = row["db_link"], row["table_owner"]
            tn = self.normalize_name(row["table_name"])
            dblinks_owners[key][tn] = row["synonym_name"]

        if not dblinks_owners:
            # No synonym, do the plain thing
            return fn(self, connection, *args, **original_kw)

        data = {}
        for (dblink, table_owner), mapping in dblinks_owners.items():
            call_kw = {
                **original_kw,
                "schema": table_owner,
                "dblink": self.normalize_name(dblink),
                "filter_names": mapping.keys(),
            }
            call_result = fn(self, connection, *args, **call_kw)
            for (_, tn), value in call_result:
                synonym_name = self.normalize_name(mapping[tn])
                data[(schema, synonym_name)] = value
        return data.items()

    @reflection.cache
    def get_schema_names(self, connection, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        query = select(dictionary.all_users.c.username).order_by(
            dictionary.all_users.c.username
        )
        result = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).scalars()
        return [self.normalize_name(row) for row in result]

    @reflection.cache
    def get_table_names(self, connection, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        # note that table_names() isn't loading DBLINKed or synonym'ed tables
        if schema is None:
            schema = self.default_schema_name

        den_schema = self.denormalize_schema_name(schema)
        if kw.get("oracle_resolve_synonyms", False):
            tables = (
                select(
                    dictionary.all_tables.c.table_name,
                    dictionary.all_tables.c.owner,
                    dictionary.all_tables.c.iot_name,
                    dictionary.all_tables.c.duration,
                    dictionary.all_tables.c.tablespace_name,
                )
                .union_all(
                    select(
                        dictionary.all_synonyms.c.synonym_name.label(
                            "table_name"
                        ),
                        dictionary.all_synonyms.c.owner,
                        dictionary.all_tables.c.iot_name,
                        dictionary.all_tables.c.duration,
                        dictionary.all_tables.c.tablespace_name,
                    )
                    .select_from(dictionary.all_tables)
                    .join(
                        dictionary.all_synonyms,
                        and_(
                            dictionary.all_tables.c.table_name
                            == dictionary.all_synonyms.c.table_name,
                            dictionary.all_tables.c.owner
                            == func.coalesce(
                                dictionary.all_synonyms.c.table_owner,
                                dictionary.all_synonyms.c.owner,
                            ),
                        ),
                    )
                )
                .subquery("available_tables")
            )
        else:
            tables = dictionary.all_tables

        query = select(tables.c.table_name)
        if self.exclude_tablespaces:
            query = query.where(
                func.coalesce(
                    tables.c.tablespace_name, "no tablespace"
                ).not_in(self.exclude_tablespaces)
            )
        query = query.where(
            tables.c.owner == den_schema,
            tables.c.iot_name.is_(null()),
            tables.c.duration.is_(null()),
        )

        # remove materialized views
        mat_query = select(
            dictionary.all_mviews.c.mview_name.label("table_name")
        ).where(dictionary.all_mviews.c.owner == den_schema)

        query = (
            query.except_all(mat_query)
            if self._supports_except_all
            else query.except_(mat_query)
        )

        result = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).scalars()
        return [self.normalize_name(row) for row in result]

    @reflection.cache
    def get_temp_table_names(self, connection, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        schema = self.denormalize_schema_name(self.default_schema_name)

        query = select(dictionary.all_tables.c.table_name)
        if self.exclude_tablespaces:
            query = query.where(
                func.coalesce(
                    dictionary.all_tables.c.tablespace_name, "no tablespace"
                ).not_in(self.exclude_tablespaces)
            )
        query = query.where(
            dictionary.all_tables.c.owner == schema,
            dictionary.all_tables.c.iot_name.is_(null()),
            dictionary.all_tables.c.duration.is_not(null()),
        )

        result = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).scalars()
        return [self.normalize_name(row) for row in result]

    @reflection.cache
    def get_materialized_view_names(
        self, connection, schema=None, dblink=None, _normalize=True, **kw
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if not schema:
            schema = self.default_schema_name

        query = select(dictionary.all_mviews.c.mview_name).where(
            dictionary.all_mviews.c.owner
            == self.denormalize_schema_name(schema)
        )
        result = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).scalars()
        if _normalize:
            return [self.normalize_name(row) for row in result]
        else:
            return result.all()

    @reflection.cache
    def get_view_names(self, connection, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if not schema:
            schema = self.default_schema_name

        query = select(dictionary.all_views.c.view_name).where(
            dictionary.all_views.c.owner
            == self.denormalize_schema_name(schema)
        )
        result = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).scalars()
        return [self.normalize_name(row) for row in result]

    @reflection.cache
    def get_sequence_names(self, connection, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if not schema:
            schema = self.default_schema_name
        query = select(dictionary.all_sequences.c.sequence_name).where(
            dictionary.all_sequences.c.sequence_owner
            == self.denormalize_schema_name(schema)
        )

        result = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).scalars()
        return [self.normalize_name(row) for row in result]

    def _value_or_raise(self, data, table, schema):
        table = self.normalize_name(str(table))
        try:
            return dict(data)[(schema, table)]
        except KeyError:
            raise exc.NoSuchTableError(
                f"{schema}.{table}" if schema else table
            ) from None

    def _prepare_filter_names(self, filter_names):
        if filter_names:
            fn = [self.denormalize_name(name) for name in filter_names]
            return True, {"filter_names": fn}
        else:
            return False, {}

    @reflection.cache
    def get_table_options(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_table_options(
            connection,
            schema=schema,
            filter_names=[table_name],
            scope=ObjectScope.ANY,
            kind=ObjectKind.ANY,
            **kw,
        )
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _table_options_query(
        self, owner, scope, kind, has_filter_names, has_mat_views
    ):
        query = select(
            dictionary.all_tables.c.table_name,
            (
                dictionary.all_tables.c.compression
                if self._supports_table_compression
                else sql.null().label("compression")
            ),
            (
                dictionary.all_tables.c.compress_for
                if self._supports_table_compress_for
                else sql.null().label("compress_for")
            ),
            dictionary.all_tables.c.tablespace_name,
        ).where(dictionary.all_tables.c.owner == owner)
        if has_filter_names:
            query = query.where(
                dictionary.all_tables.c.table_name.in_(
                    bindparam("filter_names")
                )
            )
        if scope is ObjectScope.DEFAULT:
            query = query.where(dictionary.all_tables.c.duration.is_(null()))
        elif scope is ObjectScope.TEMPORARY:
            query = query.where(
                dictionary.all_tables.c.duration.is_not(null())
            )

        if (
            has_mat_views
            and ObjectKind.TABLE in kind
            and ObjectKind.MATERIALIZED_VIEW not in kind
        ):
            # cant use EXCEPT ALL / MINUS here because we don't have an
            # excludable row vs. the query above
            # outerjoin + where null works better on oracle 21 but 11 does
            # not like it at all. this is the next best thing

            query = query.where(
                dictionary.all_tables.c.table_name.not_in(
                    bindparam("mat_views")
                )
            )
        elif (
            ObjectKind.TABLE not in kind
            and ObjectKind.MATERIALIZED_VIEW in kind
        ):
            query = query.where(
                dictionary.all_tables.c.table_name.in_(bindparam("mat_views"))
            )
        return query

    @_handle_synonyms_decorator
    def get_multi_table_options(
        self,
        connection,
        *,
        schema,
        filter_names,
        scope,
        kind,
        dblink=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )

        has_filter_names, params = self._prepare_filter_names(filter_names)
        has_mat_views = False

        if (
            ObjectKind.TABLE in kind
            and ObjectKind.MATERIALIZED_VIEW not in kind
        ):
            # see note in _table_options_query
            mat_views = self.get_materialized_view_names(
                connection, schema, dblink, _normalize=False, **kw
            )
            if mat_views:
                params["mat_views"] = mat_views
                has_mat_views = True
        elif (
            ObjectKind.TABLE not in kind
            and ObjectKind.MATERIALIZED_VIEW in kind
        ):
            mat_views = self.get_materialized_view_names(
                connection, schema, dblink, _normalize=False, **kw
            )
            params["mat_views"] = mat_views

        options = {}
        default = ReflectionDefaults.table_options

        if ObjectKind.TABLE in kind or ObjectKind.MATERIALIZED_VIEW in kind:
            query = self._table_options_query(
                owner, scope, kind, has_filter_names, has_mat_views
            )
            result = self._execute_reflection(
                connection, query, dblink, returns_long=False, params=params
            )

            for table, compression, compress_for, tablespace in result:
                data = default()
                if compression == "ENABLED":
                    data["oracle_compress"] = compress_for
                if tablespace:
                    data["oracle_tablespace"] = tablespace
                options[(schema, self.normalize_name(table))] = data
        if ObjectKind.VIEW in kind and ObjectScope.DEFAULT in scope:
            # add the views (no temporary views)
            for view in self.get_view_names(connection, schema, dblink, **kw):
                if not filter_names or view in filter_names:
                    options[(schema, view)] = default()

        return options.items()

    @reflection.cache
    def get_columns(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """

        data = self.get_multi_columns(
            connection,
            schema=schema,
            filter_names=[table_name],
            scope=ObjectScope.ANY,
            kind=ObjectKind.ANY,
            **kw,
        )
        return self._value_or_raise(data, table_name, schema)

    def _run_batches(
        self, connection, query, dblink, returns_long, mappings, all_objects
    ):
        each_batch = 500
        batches = list(all_objects)
        while batches:
            batch = batches[0:each_batch]
            batches[0:each_batch] = []

            result = self._execute_reflection(
                connection,
                query,
                dblink,
                returns_long=returns_long,
                params={"all_objects": batch},
            )
            if mappings:
                yield from result.mappings()
            else:
                yield from result

    @lru_cache()
    def _column_query(self, owner):
        all_cols = dictionary.all_tab_cols
        all_comments = dictionary.all_col_comments
        all_ids = dictionary.all_tab_identity_cols

        if self.server_version_info >= (12,):
            add_cols = (
                all_cols.c.default_on_null,
                sql.case(
                    (all_ids.c.table_name.is_(None), sql.null()),
                    else_=all_ids.c.generation_type
                    + ","
                    + all_ids.c.identity_options,
                ).label("identity_options"),
            )
            join_identity_cols = True
        else:
            add_cols = (
                sql.null().label("default_on_null"),
                sql.null().label("identity_options"),
            )
            join_identity_cols = False

        # NOTE: on oracle cannot create tables/views without columns and
        # a table cannot have all column hidden:
        # ORA-54039: table must have at least one column that is not invisible
        # all_tab_cols returns data for tables/views/mat-views.
        # all_tab_cols does not return recycled tables

        query = (
            select(
                all_cols.c.table_name,
                all_cols.c.column_name,
                all_cols.c.data_type,
                all_cols.c.char_length,
                all_cols.c.data_precision,
                all_cols.c.data_scale,
                all_cols.c.nullable,
                all_cols.c.data_default,
                all_comments.c.comments,
                all_cols.c.virtual_column,
                *add_cols,
            ).select_from(all_cols)
            # NOTE: all_col_comments has a row for each column even if no
            # comment is present, so a join could be performed, but there
            # seems to be no difference compared to an outer join
            .outerjoin(
                all_comments,
                and_(
                    all_cols.c.table_name == all_comments.c.table_name,
                    all_cols.c.column_name == all_comments.c.column_name,
                    all_cols.c.owner == all_comments.c.owner,
                ),
            )
        )
        if join_identity_cols:
            query = query.outerjoin(
                all_ids,
                and_(
                    all_cols.c.table_name == all_ids.c.table_name,
                    all_cols.c.column_name == all_ids.c.column_name,
                    all_cols.c.owner == all_ids.c.owner,
                ),
            )

        query = query.where(
            all_cols.c.table_name.in_(bindparam("all_objects")),
            all_cols.c.hidden_column == "NO",
            all_cols.c.owner == owner,
        ).order_by(all_cols.c.table_name, all_cols.c.column_id)
        return query

    @_handle_synonyms_decorator
    def get_multi_columns(
        self,
        connection,
        *,
        schema,
        filter_names,
        scope,
        kind,
        dblink=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )
        query = self._column_query(owner)

        if (
            filter_names
            and kind is ObjectKind.ANY
            and scope is ObjectScope.ANY
        ):
            all_objects = [self.denormalize_name(n) for n in filter_names]
        else:
            all_objects = self._get_all_objects(
                connection, schema, scope, kind, filter_names, dblink, **kw
            )

        columns = defaultdict(list)

        # all_tab_cols.data_default is LONG
        result = self._run_batches(
            connection,
            query,
            dblink,
            returns_long=True,
            mappings=True,
            all_objects=all_objects,
        )

        def maybe_int(value):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            else:
                return value

        remove_size = re.compile(r"\(\d+\)")

        for row_dict in result:
            table_name = self.normalize_name(row_dict["table_name"])
            orig_colname = row_dict["column_name"]
            colname = self.normalize_name(orig_colname)
            coltype = row_dict["data_type"]
            precision = maybe_int(row_dict["data_precision"])

            if coltype == "NUMBER":
                scale = maybe_int(row_dict["data_scale"])
                if precision is None and scale == 0:
                    coltype = INTEGER()
                else:
                    coltype = NUMBER(precision, scale)
            elif coltype == "FLOAT":
                # https://docs.oracle.com/cd/B14117_01/server.101/b10758/sqlqr06.htm
                if precision == 126:
                    # The DOUBLE PRECISION datatype is a floating-point
                    # number with binary precision 126.
                    coltype = DOUBLE_PRECISION()
                elif precision == 63:
                    # The REAL datatype is a floating-point number with a
                    # binary precision of 63, or 18 decimal.
                    coltype = REAL()
                else:
                    # non standard precision
                    coltype = FLOAT(binary_precision=precision)

            elif coltype in ("VARCHAR2", "NVARCHAR2", "CHAR", "NCHAR"):
                char_length = maybe_int(row_dict["char_length"])
                coltype = self.ischema_names.get(coltype)(char_length)
            elif "WITH TIME ZONE" in coltype:
                coltype = TIMESTAMP(timezone=True)
            elif "WITH LOCAL TIME ZONE" in coltype:
                coltype = TIMESTAMP(local_timezone=True)
            else:
                coltype = re.sub(remove_size, "", coltype)
                try:
                    coltype = self.ischema_names[coltype]
                except KeyError:
                    util.warn(
                        "Did not recognize type '%s' of column '%s'"
                        % (coltype, colname)
                    )
                    coltype = sqltypes.NULLTYPE

            default = row_dict["data_default"]
            if row_dict["virtual_column"] == "YES":
                computed = dict(sqltext=default)
                default = None
            else:
                computed = None

            identity_options = row_dict["identity_options"]
            if identity_options is not None:
                identity = self._parse_identity_options(
                    identity_options, row_dict["default_on_null"]
                )
                default = None
            else:
                identity = None

            cdict = {
                "name": colname,
                "type": coltype,
                "nullable": row_dict["nullable"] == "Y",
                "default": default,
                "comment": row_dict["comments"],
            }
            if orig_colname.lower() == orig_colname:
                cdict["quote"] = True
            if computed is not None:
                cdict["computed"] = computed
            if identity is not None:
                cdict["identity"] = identity

            columns[(schema, table_name)].append(cdict)

        # NOTE: default not needed since all tables have columns
        # default = ReflectionDefaults.columns
        # return (
        #     (key, value if value else default())
        #     for key, value in columns.items()
        # )
        return columns.items()

    def _parse_identity_options(self, identity_options, default_on_null):
        # identity_options is a string that starts with 'ALWAYS,' or
        # 'BY DEFAULT,' and continues with
        # START WITH: 1, INCREMENT BY: 1, MAX_VALUE: 123, MIN_VALUE: 1,
        # CYCLE_FLAG: N, CACHE_SIZE: 1, ORDER_FLAG: N, SCALE_FLAG: N,
        # EXTEND_FLAG: N, SESSION_FLAG: N, KEEP_VALUE: N
        parts = [p.strip() for p in identity_options.split(",")]
        identity = {
            "always": parts[0] == "ALWAYS",
            "on_null": default_on_null == "YES",
        }

        for part in parts[1:]:
            option, value = part.split(":")
            value = value.strip()

            if "START WITH" in option:
                identity["start"] = int(value)
            elif "INCREMENT BY" in option:
                identity["increment"] = int(value)
            elif "MAX_VALUE" in option:
                identity["maxvalue"] = int(value)
            elif "MIN_VALUE" in option:
                identity["minvalue"] = int(value)
            elif "CYCLE_FLAG" in option:
                identity["cycle"] = value == "Y"
            elif "CACHE_SIZE" in option:
                identity["cache"] = int(value)
            elif "ORDER_FLAG" in option:
                identity["order"] = value == "Y"
        return identity

    @reflection.cache
    def get_table_comment(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_table_comment(
            connection,
            schema=schema,
            filter_names=[table_name],
            scope=ObjectScope.ANY,
            kind=ObjectKind.ANY,
            **kw,
        )
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _comment_query(self, owner, scope, kind, has_filter_names):
        # NOTE: all_tab_comments / all_mview_comments have a row for all
        # object even if they don't have comments
        queries = []
        if ObjectKind.TABLE in kind or ObjectKind.VIEW in kind:
            # all_tab_comments returns also plain views
            tbl_view = select(
                dictionary.all_tab_comments.c.table_name,
                dictionary.all_tab_comments.c.comments,
            ).where(
                dictionary.all_tab_comments.c.owner == owner,
                dictionary.all_tab_comments.c.table_name.not_like("BIN$%"),
            )
            if ObjectKind.VIEW not in kind:
                tbl_view = tbl_view.where(
                    dictionary.all_tab_comments.c.table_type == "TABLE"
                )
            elif ObjectKind.TABLE not in kind:
                tbl_view = tbl_view.where(
                    dictionary.all_tab_comments.c.table_type == "VIEW"
                )
            queries.append(tbl_view)
        if ObjectKind.MATERIALIZED_VIEW in kind:
            mat_view = select(
                dictionary.all_mview_comments.c.mview_name.label("table_name"),
                dictionary.all_mview_comments.c.comments,
            ).where(
                dictionary.all_mview_comments.c.owner == owner,
                dictionary.all_mview_comments.c.mview_name.not_like("BIN$%"),
            )
            queries.append(mat_view)
        if len(queries) == 1:
            query = queries[0]
        else:
            union = sql.union_all(*queries).subquery("tables_and_views")
            query = select(union.c.table_name, union.c.comments)

        name_col = query.selected_columns.table_name

        if scope in (ObjectScope.DEFAULT, ObjectScope.TEMPORARY):
            temp = "Y" if scope is ObjectScope.TEMPORARY else "N"
            # need distinct since materialized view are listed also
            # as tables in all_objects
            query = query.distinct().join(
                dictionary.all_objects,
                and_(
                    dictionary.all_objects.c.owner == owner,
                    dictionary.all_objects.c.object_name == name_col,
                    dictionary.all_objects.c.temporary == temp,
                ),
            )
        if has_filter_names:
            query = query.where(name_col.in_(bindparam("filter_names")))
        return query

    @_handle_synonyms_decorator
    def get_multi_table_comment(
        self,
        connection,
        *,
        schema,
        filter_names,
        scope,
        kind,
        dblink=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )
        has_filter_names, params = self._prepare_filter_names(filter_names)
        query = self._comment_query(owner, scope, kind, has_filter_names)

        result = self._execute_reflection(
            connection, query, dblink, returns_long=False, params=params
        )
        default = ReflectionDefaults.table_comment
        # materialized views by default seem to have a comment like
        # "snapshot table for snapshot owner.mat_view_name"
        ignore_mat_view = "snapshot table for snapshot "
        return (
            (
                (schema, self.normalize_name(table)),
                (
                    {"text": comment}
                    if comment is not None
                    and not comment.startswith(ignore_mat_view)
                    else default()
                ),
            )
            for table, comment in result
        )

    @reflection.cache
    def get_indexes(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_indexes(
            connection,
            schema=schema,
            filter_names=[table_name],
            scope=ObjectScope.ANY,
            kind=ObjectKind.ANY,
            **kw,
        )
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _index_query(self, owner):
        return (
            select(
                dictionary.all_ind_columns.c.table_name,
                dictionary.all_ind_columns.c.index_name,
                dictionary.all_ind_columns.c.column_name,
                dictionary.all_indexes.c.index_type,
                dictionary.all_indexes.c.uniqueness,
                dictionary.all_indexes.c.compression,
                dictionary.all_indexes.c.prefix_length,
                dictionary.all_ind_columns.c.descend,
                dictionary.all_ind_expressions.c.column_expression,
            )
            .select_from(dictionary.all_ind_columns)
            .join(
                dictionary.all_indexes,
                sql.and_(
                    dictionary.all_ind_columns.c.index_name
                    == dictionary.all_indexes.c.index_name,
                    dictionary.all_ind_columns.c.index_owner
                    == dictionary.all_indexes.c.owner,
                ),
            )
            .outerjoin(
                # NOTE: this adds about 20% to the query time. Using a
                # case expression with a scalar subquery only when needed
                # with the assumption that most indexes are not expression
                # would be faster but oracle does not like that with
                # LONG datatype. It errors with:
                # ORA-00997: illegal use of LONG datatype
                dictionary.all_ind_expressions,
                sql.and_(
                    dictionary.all_ind_expressions.c.index_name
                    == dictionary.all_ind_columns.c.index_name,
                    dictionary.all_ind_expressions.c.index_owner
                    == dictionary.all_ind_columns.c.index_owner,
                    dictionary.all_ind_expressions.c.column_position
                    == dictionary.all_ind_columns.c.column_position,
                ),
            )
            .where(
                dictionary.all_indexes.c.table_owner == owner,
                dictionary.all_indexes.c.table_name.in_(
                    bindparam("all_objects")
                ),
            )
            .order_by(
                dictionary.all_ind_columns.c.index_name,
                dictionary.all_ind_columns.c.column_position,
            )
        )

    @reflection.flexi_cache(
        ("schema", InternalTraversal.dp_string),
        ("dblink", InternalTraversal.dp_string),
        ("all_objects", InternalTraversal.dp_string_list),
    )
    def _get_indexes_rows(self, connection, schema, dblink, all_objects, **kw):
        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )

        query = self._index_query(owner)

        pks = {
            row_dict["constraint_name"]
            for row_dict in self._get_all_constraint_rows(
                connection, schema, dblink, all_objects, **kw
            )
            if row_dict["constraint_type"] == "P"
        }

        # all_ind_expressions.column_expression is LONG
        result = self._run_batches(
            connection,
            query,
            dblink,
            returns_long=True,
            mappings=True,
            all_objects=all_objects,
        )

        return [
            row_dict
            for row_dict in result
            if row_dict["index_name"] not in pks
        ]

    @_handle_synonyms_decorator
    def get_multi_indexes(
        self,
        connection,
        *,
        schema,
        filter_names,
        scope,
        kind,
        dblink=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(
            connection, schema, scope, kind, filter_names, dblink, **kw
        )

        uniqueness = {"NONUNIQUE": False, "UNIQUE": True}
        enabled = {"DISABLED": False, "ENABLED": True}
        is_bitmap = {"BITMAP", "FUNCTION-BASED BITMAP"}

        indexes = defaultdict(dict)

        for row_dict in self._get_indexes_rows(
            connection, schema, dblink, all_objects, **kw
        ):
            index_name = self.normalize_name(row_dict["index_name"])
            table_name = self.normalize_name(row_dict["table_name"])
            table_indexes = indexes[(schema, table_name)]

            if index_name not in table_indexes:
                table_indexes[index_name] = index_dict = {
                    "name": index_name,
                    "column_names": [],
                    "dialect_options": {},
                    "unique": uniqueness.get(row_dict["uniqueness"], False),
                }
                do = index_dict["dialect_options"]
                if row_dict["index_type"] in is_bitmap:
                    do["oracle_bitmap"] = True
                if enabled.get(row_dict["compression"], False):
                    do["oracle_compress"] = row_dict["prefix_length"]

            else:
                index_dict = table_indexes[index_name]

            expr = row_dict["column_expression"]
            if expr is not None:
                index_dict["column_names"].append(None)
                if "expressions" in index_dict:
                    index_dict["expressions"].append(expr)
                else:
                    index_dict["expressions"] = index_dict["column_names"][:-1]
                    index_dict["expressions"].append(expr)

                if row_dict["descend"].lower() != "asc":
                    assert row_dict["descend"].lower() == "desc"
                    cs = index_dict.setdefault("column_sorting", {})
                    cs[expr] = ("desc",)
            else:
                assert row_dict["descend"].lower() == "asc"
                cn = self.normalize_name(row_dict["column_name"])
                index_dict["column_names"].append(cn)
                if "expressions" in index_dict:
                    index_dict["expressions"].append(cn)

        default = ReflectionDefaults.indexes

        return (
            (key, list(indexes[key].values()) if key in indexes else default())
            for key in (
                (schema, self.normalize_name(obj_name))
                for obj_name in all_objects
            )
        )

    @reflection.cache
    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_pk_constraint(
            connection,
            schema=schema,
            filter_names=[table_name],
            scope=ObjectScope.ANY,
            kind=ObjectKind.ANY,
            **kw,
        )
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _constraint_query(self, owner):
        local = dictionary.all_cons_columns.alias("local")
        remote = dictionary.all_cons_columns.alias("remote")
        return (
            select(
                dictionary.all_constraints.c.table_name,
                dictionary.all_constraints.c.constraint_type,
                dictionary.all_constraints.c.constraint_name,
                local.c.column_name.label("local_column"),
                remote.c.table_name.label("remote_table"),
                remote.c.column_name.label("remote_column"),
                remote.c.owner.label("remote_owner"),
                dictionary.all_constraints.c.search_condition,
                dictionary.all_constraints.c.delete_rule,
            )
            .select_from(dictionary.all_constraints)
            .join(
                local,
                and_(
                    local.c.owner == dictionary.all_constraints.c.owner,
                    dictionary.all_constraints.c.constraint_name
                    == local.c.constraint_name,
                ),
            )
            .outerjoin(
                remote,
                and_(
                    dictionary.all_constraints.c.r_owner == remote.c.owner,
                    dictionary.all_constraints.c.r_constraint_name
                    == remote.c.constraint_name,
                    or_(
                        remote.c.position.is_(sql.null()),
                        local.c.position == remote.c.position,
                    ),
                ),
            )
            .where(
                dictionary.all_constraints.c.owner == owner,
                dictionary.all_constraints.c.table_name.in_(
                    bindparam("all_objects")
                ),
                dictionary.all_constraints.c.constraint_type.in_(
                    ("R", "P", "U", "C")
                ),
            )
            .order_by(
                dictionary.all_constraints.c.constraint_name, local.c.position
            )
        )

    @reflection.flexi_cache(
        ("schema", InternalTraversal.dp_string),
        ("dblink", InternalTraversal.dp_string),
        ("all_objects", InternalTraversal.dp_string_list),
    )
    def _get_all_constraint_rows(
        self, connection, schema, dblink, all_objects, **kw
    ):
        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )
        query = self._constraint_query(owner)

        # since the result is cached a list must be created
        values = list(
            self._run_batches(
                connection,
                query,
                dblink,
                returns_long=False,
                mappings=True,
                all_objects=all_objects,
            )
        )
        return values

    @_handle_synonyms_decorator
    def get_multi_pk_constraint(
        self,
        connection,
        *,
        scope,
        schema,
        filter_names,
        kind,
        dblink=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(
            connection, schema, scope, kind, filter_names, dblink, **kw
        )

        primary_keys = defaultdict(dict)
        default = ReflectionDefaults.pk_constraint

        for row_dict in self._get_all_constraint_rows(
            connection, schema, dblink, all_objects, **kw
        ):
            if row_dict["constraint_type"] != "P":
                continue
            table_name = self.normalize_name(row_dict["table_name"])
            constraint_name = self.normalize_name(row_dict["constraint_name"])
            column_name = self.normalize_name(row_dict["local_column"])

            table_pk = primary_keys[(schema, table_name)]
            if not table_pk:
                table_pk["name"] = constraint_name
                table_pk["constrained_columns"] = [column_name]
            else:
                table_pk["constrained_columns"].append(column_name)

        return (
            (key, primary_keys[key] if key in primary_keys else default())
            for key in (
                (schema, self.normalize_name(obj_name))
                for obj_name in all_objects
            )
        )

    @reflection.cache
    def get_foreign_keys(
        self,
        connection,
        table_name,
        schema=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_foreign_keys(
            connection,
            schema=schema,
            filter_names=[table_name],
            scope=ObjectScope.ANY,
            kind=ObjectKind.ANY,
            **kw,
        )
        return self._value_or_raise(data, table_name, schema)

    @_handle_synonyms_decorator
    def get_multi_foreign_keys(
        self,
        connection,
        *,
        scope,
        schema,
        filter_names,
        kind,
        dblink=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(
            connection, schema, scope, kind, filter_names, dblink, **kw
        )

        resolve_synonyms = kw.get("oracle_resolve_synonyms", False)

        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )

        all_remote_owners = set()
        fkeys = defaultdict(dict)

        for row_dict in self._get_all_constraint_rows(
            connection, schema, dblink, all_objects, **kw
        ):
            if row_dict["constraint_type"] != "R":
                continue

            table_name = self.normalize_name(row_dict["table_name"])
            constraint_name = self.normalize_name(row_dict["constraint_name"])
            table_fkey = fkeys[(schema, table_name)]

            assert constraint_name is not None

            local_column = self.normalize_name(row_dict["local_column"])
            remote_table = self.normalize_name(row_dict["remote_table"])
            remote_column = self.normalize_name(row_dict["remote_column"])
            remote_owner_orig = row_dict["remote_owner"]
            remote_owner = self.normalize_name(remote_owner_orig)
            if remote_owner_orig is not None:
                all_remote_owners.add(remote_owner_orig)

            if remote_table is None:
                # ticket 363
                if dblink and not dblink.startswith("@"):
                    dblink = f"@{dblink}"
                util.warn(
                    "Got 'None' querying 'table_name' from "
                    f"all_cons_columns{dblink or ''} - does the user have "
                    "proper rights to the table?"
                )
                continue

            if constraint_name not in table_fkey:
                table_fkey[constraint_name] = fkey = {
                    "name": constraint_name,
                    "constrained_columns": [],
                    "referred_schema": None,
                    "referred_table": remote_table,
                    "referred_columns": [],
                    "options": {},
                }

                if resolve_synonyms:
                    # will be removed below
                    fkey["_ref_schema"] = remote_owner

                if schema is not None or remote_owner_orig != owner:
                    fkey["referred_schema"] = remote_owner

                delete_rule = row_dict["delete_rule"]
                if delete_rule != "NO ACTION":
                    fkey["options"]["ondelete"] = delete_rule

            else:
                fkey = table_fkey[constraint_name]

            fkey["constrained_columns"].append(local_column)
            fkey["referred_columns"].append(remote_column)

        if resolve_synonyms and all_remote_owners:
            query = select(
                dictionary.all_synonyms.c.owner,
                dictionary.all_synonyms.c.table_name,
                dictionary.all_synonyms.c.table_owner,
                dictionary.all_synonyms.c.synonym_name,
            ).where(dictionary.all_synonyms.c.owner.in_(all_remote_owners))

            result = self._execute_reflection(
                connection, query, dblink, returns_long=False
            ).mappings()

            remote_owners_lut = {}
            for row in result:
                synonym_owner = self.normalize_name(row["owner"])
                table_name = self.normalize_name(row["table_name"])

                remote_owners_lut[(synonym_owner, table_name)] = (
                    row["table_owner"],
                    row["synonym_name"],
                )

            empty = (None, None)
            for table_fkeys in fkeys.values():
                for table_fkey in table_fkeys.values():
                    key = (
                        table_fkey.pop("_ref_schema"),
                        table_fkey["referred_table"],
                    )
                    remote_owner, syn_name = remote_owners_lut.get(key, empty)
                    if syn_name:
                        sn = self.normalize_name(syn_name)
                        table_fkey["referred_table"] = sn
                        if schema is not None or remote_owner != owner:
                            ro = self.normalize_name(remote_owner)
                            table_fkey["referred_schema"] = ro
                        else:
                            table_fkey["referred_schema"] = None
        default = ReflectionDefaults.foreign_keys

        return (
            (key, list(fkeys[key].values()) if key in fkeys else default())
            for key in (
                (schema, self.normalize_name(obj_name))
                for obj_name in all_objects
            )
        )

    @reflection.cache
    def get_unique_constraints(
        self, connection, table_name, schema=None, **kw
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_unique_constraints(
            connection,
            schema=schema,
            filter_names=[table_name],
            scope=ObjectScope.ANY,
            kind=ObjectKind.ANY,
            **kw,
        )
        return self._value_or_raise(data, table_name, schema)

    @_handle_synonyms_decorator
    def get_multi_unique_constraints(
        self,
        connection,
        *,
        scope,
        schema,
        filter_names,
        kind,
        dblink=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(
            connection, schema, scope, kind, filter_names, dblink, **kw
        )

        unique_cons = defaultdict(dict)

        index_names = {
            row_dict["index_name"]
            for row_dict in self._get_indexes_rows(
                connection, schema, dblink, all_objects, **kw
            )
        }

        for row_dict in self._get_all_constraint_rows(
            connection, schema, dblink, all_objects, **kw
        ):
            if row_dict["constraint_type"] != "U":
                continue
            table_name = self.normalize_name(row_dict["table_name"])
            constraint_name_orig = row_dict["constraint_name"]
            constraint_name = self.normalize_name(constraint_name_orig)
            column_name = self.normalize_name(row_dict["local_column"])
            table_uc = unique_cons[(schema, table_name)]

            assert constraint_name is not None

            if constraint_name not in table_uc:
                table_uc[constraint_name] = uc = {
                    "name": constraint_name,
                    "column_names": [],
                    "duplicates_index": (
                        constraint_name
                        if constraint_name_orig in index_names
                        else None
                    ),
                }
            else:
                uc = table_uc[constraint_name]

            uc["column_names"].append(column_name)

        default = ReflectionDefaults.unique_constraints

        return (
            (
                key,
                (
                    list(unique_cons[key].values())
                    if key in unique_cons
                    else default()
                ),
            )
            for key in (
                (schema, self.normalize_name(obj_name))
                for obj_name in all_objects
            )
        )

    @reflection.cache
    def get_view_definition(
        self,
        connection,
        view_name,
        schema=None,
        dblink=None,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        if kw.get("oracle_resolve_synonyms", False):
            synonyms = self._get_synonyms(
                connection, schema, filter_names=[view_name], dblink=dblink
            )
            if synonyms:
                assert len(synonyms) == 1
                row_dict = synonyms[0]
                dblink = self.normalize_name(row_dict["db_link"])
                schema = row_dict["table_owner"]
                view_name = row_dict["table_name"]

        name = self.denormalize_name(view_name)
        owner = self.denormalize_schema_name(
            schema or self.default_schema_name
        )
        query = (
            select(dictionary.all_views.c.text)
            .where(
                dictionary.all_views.c.view_name == name,
                dictionary.all_views.c.owner == owner,
            )
            .union_all(
                select(dictionary.all_mviews.c.query).where(
                    dictionary.all_mviews.c.mview_name == name,
                    dictionary.all_mviews.c.owner == owner,
                )
            )
        )

        rp = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).scalar()
        if rp is None:
            raise exc.NoSuchTableError(
                f"{schema}.{view_name}" if schema else view_name
            )
        else:
            return rp

    @reflection.cache
    def get_check_constraints(
        self, connection, table_name, schema=None, include_all=False, **kw
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_check_constraints(
            connection,
            schema=schema,
            filter_names=[table_name],
            scope=ObjectScope.ANY,
            include_all=include_all,
            kind=ObjectKind.ANY,
            **kw,
        )
        return self._value_or_raise(data, table_name, schema)

    @_handle_synonyms_decorator
    def get_multi_check_constraints(
        self,
        connection,
        *,
        schema,
        filter_names,
        dblink=None,
        scope,
        kind,
        include_all=False,
        **kw,
    ):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(
            connection, schema, scope, kind, filter_names, dblink, **kw
        )

        not_null = re.compile(r"..+?. IS NOT NULL$")

        check_constraints = defaultdict(list)

        for row_dict in self._get_all_constraint_rows(
            connection, schema, dblink, all_objects, **kw
        ):
            if row_dict["constraint_type"] != "C":
                continue
            table_name = self.normalize_name(row_dict["table_name"])
            constraint_name = self.normalize_name(row_dict["constraint_name"])
            search_condition = row_dict["search_condition"]

            table_checks = check_constraints[(schema, table_name)]
            if constraint_name is not None and (
                include_all or not not_null.match(search_condition)
            ):
                table_checks.append(
                    {"name": constraint_name, "sqltext": search_condition}
                )

        default = ReflectionDefaults.check_constraints

        return (
            (
                key,
                (
                    check_constraints[key]
                    if key in check_constraints
                    else default()
                ),
            )
            for key in (
                (schema, self.normalize_name(obj_name))
                for obj_name in all_objects
            )
        )

    def _list_dblinks(self, connection, dblink=None):
        query = select(dictionary.all_db_links.c.db_link)
        links = self._execute_reflection(
            connection, query, dblink, returns_long=False
        ).scalars()
        return [self.normalize_name(link) for link in links]


class _OuterJoinColumn(sql.ClauseElement):
    __visit_name__ = "outer_join_column"

    def __init__(self, column):
        self.column = column