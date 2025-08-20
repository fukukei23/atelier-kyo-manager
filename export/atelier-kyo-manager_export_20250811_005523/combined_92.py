
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\coordsysrect.py ===
from collections.abc import Callable

from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda
from sympy.core.symbol import Str
from sympy.core.symbol import symbols
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.matrices.matrixbase import MatrixBase
from sympy.solvers import solve
from sympy.vector.scalar import BaseScalar
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import eye
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
import sympy.vector
from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)


class CoordSys3D(Basic):
    """
    Represents a coordinate system in 3-D space.
    """

    def __new__(cls, name, transformation=None, parent=None, location=None,
                rotation_matrix=None, vector_names=None, variable_names=None):
        """
        The orientation/location parameters are necessary if this system
        is being defined at a certain orientation or location wrt another.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        location : Vector
            The position vector of the new system's origin wrt the parent
            instance.

        rotation_matrix : SymPy ImmutableMatrix
            The rotation matrix of the new coordinate system with respect
            to the parent. In other words, the output of
            new_system.rotation_matrix(parent).

        parent : CoordSys3D
            The coordinate system wrt which the orientation/location
            (or both) is being defined.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        """

        name = str(name)
        Vector = sympy.vector.Vector
        Point = sympy.vector.Point

        if not isinstance(name, str):
            raise TypeError("name should be a string")

        if transformation is not None:
            if (location is not None) or (rotation_matrix is not None):
                raise ValueError("specify either `transformation` or "
                                 "`location`/`rotation_matrix`")
            if isinstance(transformation, (Tuple, tuple, list)):
                if isinstance(transformation[0], MatrixBase):
                    rotation_matrix = transformation[0]
                    location = transformation[1]
                else:
                    transformation = Lambda(transformation[0],
                                            transformation[1])
            elif isinstance(transformation, Callable):
                x1, x2, x3 = symbols('x1 x2 x3', cls=Dummy)
                transformation = Lambda((x1, x2, x3),
                                        transformation(x1, x2, x3))
            elif isinstance(transformation, str):
                transformation = Str(transformation)
            elif isinstance(transformation, (Str, Lambda)):
                pass
            else:
                raise TypeError("transformation: "
                                "wrong type {}".format(type(transformation)))

        # If orientation information has been provided, store
        # the rotation matrix accordingly
        if rotation_matrix is None:
            rotation_matrix = ImmutableDenseMatrix(eye(3))
        else:
            if not isinstance(rotation_matrix, MatrixBase):
                raise TypeError("rotation_matrix should be an Immutable" +
                                "Matrix instance")
            rotation_matrix = rotation_matrix.as_immutable()

        # If location information is not given, adjust the default
        # location as Vector.zero
        if parent is not None:
            if not isinstance(parent, CoordSys3D):
                raise TypeError("parent should be a " +
                                "CoordSys3D/None")
            if location is None:
                location = Vector.zero
            else:
                if not isinstance(location, Vector):
                    raise TypeError("location should be a Vector")
                # Check that location does not contain base
                # scalars
                for x in location.free_symbols:
                    if isinstance(x, BaseScalar):
                        raise ValueError("location should not contain" +
                                         " BaseScalars")
            origin = parent.origin.locate_new(name + '.origin',
                                              location)
        else:
            location = Vector.zero
            origin = Point(name + '.origin')

        if transformation is None:
            transformation = Tuple(rotation_matrix, location)

        if isinstance(transformation, Tuple):
            lambda_transformation = CoordSys3D._compose_rotation_and_translation(
                transformation[0],
                transformation[1],
                parent
            )
            r, l = transformation
            l = l._projections
            lambda_lame = CoordSys3D._get_lame_coeff('cartesian')
            lambda_inverse = lambda x, y, z: r.inv()*Matrix(
                [x-l[0], y-l[1], z-l[2]])
        elif isinstance(transformation, Str):
            trname = transformation.name
            lambda_transformation = CoordSys3D._get_transformation_lambdas(trname)
            if parent is not None:
                if parent.lame_coefficients() != (S.One, S.One, S.One):
                    raise ValueError('Parent for pre-defined coordinate '
                                 'system should be Cartesian.')
            lambda_lame = CoordSys3D._get_lame_coeff(trname)
            lambda_inverse = CoordSys3D._set_inv_trans_equations(trname)
        elif isinstance(transformation, Lambda):
            if not CoordSys3D._check_orthogonality(transformation):
                raise ValueError("The transformation equation does not "
                                 "create orthogonal coordinate system")
            lambda_transformation = transformation
            lambda_lame = CoordSys3D._calculate_lame_coeff(lambda_transformation)
            lambda_inverse = None
        else:
            lambda_transformation = lambda x, y, z: transformation(x, y, z)
            lambda_lame = CoordSys3D._get_lame_coeff(transformation)
            lambda_inverse = None

        if variable_names is None:
            if isinstance(transformation, Lambda):
                variable_names = ["x1", "x2", "x3"]
            elif isinstance(transformation, Str):
                if transformation.name == 'spherical':
                    variable_names = ["r", "theta", "phi"]
                elif transformation.name == 'cylindrical':
                    variable_names = ["r", "theta", "z"]
                else:
                    variable_names = ["x", "y", "z"]
            else:
                variable_names = ["x", "y", "z"]
        if vector_names is None:
            vector_names = ["i", "j", "k"]

        # All systems that are defined as 'roots' are unequal, unless
        # they have the same name.
        # Systems defined at same orientation/position wrt the same
        # 'parent' are equal, irrespective of the name.
        # This is true even if the same orientation is provided via
        # different methods like Axis/Body/Space/Quaternion.
        # However, coincident systems may be seen as unequal if
        # positioned/oriented wrt different parents, even though
        # they may actually be 'coincident' wrt the root system.
        if parent is not None:
            obj = super().__new__(
                cls, Str(name), transformation, parent)
        else:
            obj = super().__new__(
                cls, Str(name), transformation)
        obj._name = name
        # Initialize the base vectors

        _check_strings('vector_names', vector_names)
        vector_names = list(vector_names)
        latex_vects = [(r'\mathbf{\hat{%s}_{%s}}' % (x, name)) for
                           x in vector_names]
        pretty_vects = ['%s_%s' % (x, name) for x in vector_names]

        obj._vector_names = vector_names

        v1 = BaseVector(0, obj, pretty_vects[0], latex_vects[0])
        v2 = BaseVector(1, obj, pretty_vects[1], latex_vects[1])
        v3 = BaseVector(2, obj, pretty_vects[2], latex_vects[2])

        obj._base_vectors = (v1, v2, v3)

        # Initialize the base scalars

        _check_strings('variable_names', vector_names)
        variable_names = list(variable_names)
        latex_scalars = [(r"\mathbf{{%s}_{%s}}" % (x, name)) for
                         x in variable_names]
        pretty_scalars = ['%s_%s' % (x, name) for x in variable_names]

        obj._variable_names = variable_names
        obj._vector_names = vector_names

        x1 = BaseScalar(0, obj, pretty_scalars[0], latex_scalars[0])
        x2 = BaseScalar(1, obj, pretty_scalars[1], latex_scalars[1])
        x3 = BaseScalar(2, obj, pretty_scalars[2], latex_scalars[2])

        obj._base_scalars = (x1, x2, x3)

        obj._transformation = transformation
        obj._transformation_lambda = lambda_transformation
        obj._lame_coefficients = lambda_lame(x1, x2, x3)
        obj._transformation_from_parent_lambda = lambda_inverse

        setattr(obj, variable_names[0], x1)
        setattr(obj, variable_names[1], x2)
        setattr(obj, variable_names[2], x3)

        setattr(obj, vector_names[0], v1)
        setattr(obj, vector_names[1], v2)
        setattr(obj, vector_names[2], v3)

        # Assign params
        obj._parent = parent
        if obj._parent is not None:
            obj._root = obj._parent._root
        else:
            obj._root = obj

        obj._parent_rotation_matrix = rotation_matrix
        obj._origin = origin

        # Return the instance
        return obj

    def _sympystr(self, printer):
        return self._name

    def __iter__(self):
        return iter(self.base_vectors())

    @staticmethod
    def _check_orthogonality(equations):
        """
        Helper method for _connect_to_cartesian. It checks if
        set of transformation equations create orthogonal curvilinear
        coordinate system

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations

        """

        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy)
        equations = equations(x1, x2, x3)
        v1 = Matrix([diff(equations[0], x1),
                     diff(equations[1], x1), diff(equations[2], x1)])

        v2 = Matrix([diff(equations[0], x2),
                     diff(equations[1], x2), diff(equations[2], x2)])

        v3 = Matrix([diff(equations[0], x3),
                     diff(equations[1], x3), diff(equations[2], x3)])

        if any(simplify(i[0] + i[1] + i[2]) == 0 for i in (v1, v2, v3)):
            return False
        else:
            if simplify(v1.dot(v2)) == 0 and simplify(v2.dot(v3)) == 0 \
                and simplify(v3.dot(v1)) == 0:
                return True
            else:
                return False

    @staticmethod
    def _set_inv_trans_equations(curv_coord_name):
        """
        Store information about inverse transformation equations for
        pre-defined coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if curv_coord_name == 'cartesian':
            return lambda x, y, z: (x, y, z)

        if curv_coord_name == 'spherical':
            return lambda x, y, z: (
                sqrt(x**2 + y**2 + z**2),
                acos(z/sqrt(x**2 + y**2 + z**2)),
                atan2(y, x)
            )
        if curv_coord_name == 'cylindrical':
            return lambda x, y, z: (
                sqrt(x**2 + y**2),
                atan2(y, x),
                z
            )
        raise ValueError('Wrong set of parameters.'
                         'Type of coordinate system is defined')

    def _calculate_inv_trans_equations(self):
        """
        Helper method for set_coordinate_type. It calculates inverse
        transformation equations for given transformations equations.

        """
        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy, reals=True)
        x, y, z = symbols("x, y, z", cls=Dummy)

        equations = self._transformation(x1, x2, x3)

        solved = solve([equations[0] - x,
                        equations[1] - y,
                        equations[2] - z], (x1, x2, x3), dict=True)[0]
        solved = solved[x1], solved[x2], solved[x3]
        self._transformation_from_parent_lambda = \
            lambda x1, x2, x3: tuple(i.subs(list(zip((x, y, z), (x1, x2, x3)))) for i in solved)

    @staticmethod
    def _get_lame_coeff(curv_coord_name):
        """
        Store information about Lame coefficients for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if isinstance(curv_coord_name, str):
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (S.One, S.One, S.One)
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (S.One, r, r*sin(theta))
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (S.One, r, S.One)
            raise ValueError('Wrong set of parameters.'
                             ' Type of coordinate system is not defined')
        return CoordSys3D._calculate_lame_coefficients(curv_coord_name)

    @staticmethod
    def _calculate_lame_coeff(equations):
        """
        It calculates Lame coefficients
        for given transformations equations.

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations.

        """
        return lambda x1, x2, x3: (
                          sqrt(diff(equations(x1, x2, x3)[0], x1)**2 +
                               diff(equations(x1, x2, x3)[1], x1)**2 +
                               diff(equations(x1, x2, x3)[2], x1)**2),
                          sqrt(diff(equations(x1, x2, x3)[0], x2)**2 +
                               diff(equations(x1, x2, x3)[1], x2)**2 +
                               diff(equations(x1, x2, x3)[2], x2)**2),
                          sqrt(diff(equations(x1, x2, x3)[0], x3)**2 +
                               diff(equations(x1, x2, x3)[1], x3)**2 +
                               diff(equations(x1, x2, x3)[2], x3)**2)
                      )

    def _inverse_rotation_matrix(self):
        """
        Returns inverse rotation matrix.
        """
        return simplify(self._parent_rotation_matrix**-1)

    @staticmethod
    def _get_transformation_lambdas(curv_coord_name):
        """
        Store information about transformation equations for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if isinstance(curv_coord_name, str):
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (x, y, z)
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (
                    r*sin(theta)*cos(phi),
                    r*sin(theta)*sin(phi),
                    r*cos(theta)
                )
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (
                    r*cos(theta),
                    r*sin(theta),
                    h
                )
            raise ValueError('Wrong set of parameters.'
                             'Type of coordinate system is defined')

    @classmethod
    def _rotation_trans_equations(cls, matrix, equations):
        """
        Returns the transformation equations obtained from rotation matrix.

        Parameters
        ==========

        matrix : Matrix
            Rotation matrix

        equations : tuple
            Transformation equations

        """
        return tuple(matrix * Matrix(equations))

    @property
    def origin(self):
        return self._origin

    def base_vectors(self):
        return self._base_vectors

    def base_scalars(self):
        return self._base_scalars

    def lame_coefficients(self):
        return self._lame_coefficients

    def transformation_to_parent(self):
        return self._transformation_lambda(*self.base_scalars())

    def transformation_from_parent(self):
        if self._parent is None:
            raise ValueError("no parent coordinate system, use "
                             "`transformation_from_parent_function()`")
        return self._transformation_from_parent_lambda(
                            *self._parent.base_scalars())

    def transformation_from_parent_function(self):
        return self._transformation_from_parent_lambda

    def rotation_matrix(self, other):
        """
        Returns the direction cosine matrix(DCM), also known as the
        'rotation matrix' of this coordinate system with respect to
        another system.

        If v_a is a vector defined in system 'A' (in matrix format)
        and v_b is the same vector defined in system 'B', then
        v_a = A.rotation_matrix(B) * v_b.

        A SymPy Matrix is returned.

        Parameters
        ==========

        other : CoordSys3D
            The system which the DCM is generated to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> A = N.orient_new_axis('A', q1, N.i)
        >>> N.rotation_matrix(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        """
        from sympy.vector.functions import _path
        if not isinstance(other, CoordSys3D):
            raise TypeError(str(other) +
                            " is not a CoordSys3D")
        # Handle special cases
        if other == self:
            return eye(3)
        elif other == self._parent:
            return self._parent_rotation_matrix
        elif other._parent == self:
            return other._parent_rotation_matrix.T
        # Else, use tree to calculate position
        rootindex, path = _path(self, other)
        result = eye(3)
        for i in range(rootindex):
            result *= path[i]._parent_rotation_matrix
        for i in range(rootindex + 1, len(path)):
            result *= path[i]._parent_rotation_matrix.T
        return result

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of the origin of this coordinate
        system with respect to another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this system's origin
            wrt it is returned. If its an instance of CoordSyRect,
            the position wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N1 = N.locate_new('N1', 10 * N.i)
        >>> N.position_wrt(N1)
        (-10)*N.i

        """
        return self.origin.position_wrt(other)

    def scalar_map(self, other):
        """
        Returns a dictionary which expresses the coordinate variables
        (base scalars) of this frame in terms of the variables of
        otherframe.

        Parameters
        ==========

        otherframe : CoordSys3D
            The other system to map the variables to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import Symbol
        >>> A = CoordSys3D('A')
        >>> q = Symbol('q')
        >>> B = A.orient_new_axis('B', q, A.k)
        >>> A.scalar_map(B)
        {A.x: B.x*cos(q) - B.y*sin(q), A.y: B.x*sin(q) + B.y*cos(q), A.z: B.z}

        """

        origin_coords = tuple(self.position_wrt(other).to_matrix(other))
        relocated_scalars = [x - origin_coords[i]
                             for i, x in enumerate(other.base_scalars())]

        vars_matrix = (self.rotation_matrix(other) *
                       Matrix(relocated_scalars))
        return {x: trigsimp(vars_matrix[i])
                for i, x in enumerate(self.base_scalars())}

    def locate_new(self, name, position, vector_names=None,
                   variable_names=None):
        """
        Returns a CoordSys3D with its origin located at the given
        position wrt this coordinate system's origin.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        position : Vector
            The position vector of the new system's origin wrt this
            one.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> A = CoordSys3D('A')
        >>> B = A.locate_new('B', 10 * A.i)
        >>> B.origin.position_wrt(A.origin)
        10*A.i

        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        return CoordSys3D(name, location=position,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          parent=self)

    def orient_new(self, name, orienters, location=None,
                   vector_names=None, variable_names=None):
        """
        Creates a new CoordSys3D oriented in the user-specified way
        with respect to this system.

        Please refer to the documentation of the orienter classes
        for more information about the orientation procedure.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        orienters : iterable/Orienter
            An Orienter or an iterable of Orienters for orienting the
            new coordinate system.
            If an Orienter is provided, it is applied to get the new
            system.
            If an iterable is provided, the orienters will be applied
            in the order in which they appear in the iterable.

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')

        Using an AxisOrienter

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> A = N.orient_new('A', (axis_orienter, ))

        Using a BodyOrienter

        >>> from sympy.vector import BodyOrienter
        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')
        >>> B = N.orient_new('B', (body_orienter, ))

        Using a SpaceOrienter

        >>> from sympy.vector import SpaceOrienter
        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')
        >>> C = N.orient_new('C', (space_orienter, ))

        Using a QuaternionOrienter

        >>> from sympy.vector import QuaternionOrienter
        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)
        >>> D = N.orient_new('D', (q_orienter, ))
        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        if isinstance(orienters, Orienter):
            if isinstance(orienters, AxisOrienter):
                final_matrix = orienters.rotation_matrix(self)
            else:
                final_matrix = orienters.rotation_matrix()
            # TODO: trigsimp is needed here so that the matrix becomes
            # canonical (scalar_map also calls trigsimp; without this, you can
            # end up with the same CoordinateSystem that compares differently
            # due to a differently formatted matrix). However, this is
            # probably not so good for performance.
            final_matrix = trigsimp(final_matrix)
        else:
            final_matrix = Matrix(eye(3))
            for orienter in orienters:
                if isinstance(orienter, AxisOrienter):
                    final_matrix *= orienter.rotation_matrix(self)
                else:
                    final_matrix *= orienter.rotation_matrix()

        return CoordSys3D(name, rotation_matrix=final_matrix,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          location=location,
                          parent=self)

    def orient_new_axis(self, name, angle, axis, location=None,
                        vector_names=None, variable_names=None):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_axis('B', q1, N.i + 2 * N.j)

        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        orienter = AxisOrienter(angle, axis)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_body(self, name, angle1, angle2, angle3,
                        rotation_order, location=None,
                        vector_names=None, variable_names=None):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> D = N.orient_new_body('D', q1, q2, q3, '123')

        is same as

        >>> D = N.orient_new_axis('D', q1, N.i)
        >>> D = D.orient_new_axis('D', q2, D.j)
        >>> D = D.orient_new_axis('D', q3, D.k)

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> B = N.orient_new_body('B', q1, q2, q3, '123')
        >>> B = N.orient_new_body('B', q1, q2, 0, 'ZXZ')
        >>> B = N.orient_new_body('B', 0, 0, 0, 'XYX')

        """

        orienter = BodyOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_space(self, name, angle1, angle2, angle3,
                         rotation_order, location=None,
                         vector_names=None, variable_names=None):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        See Also
        ========

        CoordSys3D.orient_new_body : method to orient via Euler
            angles

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> D = N.orient_new_space('D', q1, q2, q3, '312')

        is same as

        >>> B = N.orient_new_axis('B', q1, N.i)
        >>> C = B.orient_new_axis('C', q2, N.j)
        >>> D = C.orient_new_axis('D', q3, N.k)

        """

        orienter = SpaceOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_quaternion(self, name, q0, q1, q2, q3, location=None,
                              vector_names=None, variable_names=None):
        """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_quaternion('B', q0, q1, q2, q3)

        """

        orienter = QuaternionOrienter(q0, q1, q2, q3)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def create_new(self, name, transformation, variable_names=None, vector_names=None):
        """
        Returns a CoordSys3D which is connected to self by transformation.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> a = CoordSys3D('a')
        >>> b = a.create_new('b', transformation='spherical')
        >>> b.transformation_to_parent()
        (b.r*sin(b.theta)*cos(b.phi), b.r*sin(b.phi)*sin(b.theta), b.r*cos(b.theta))
        >>> b.transformation_from_parent()
        (sqrt(a.x**2 + a.y**2 + a.z**2), acos(a.z/sqrt(a.x**2 + a.y**2 + a.z**2)), atan2(a.y, a.x))

        """
        return CoordSys3D(name, parent=self, transformation=transformation,
                          variable_names=variable_names, vector_names=vector_names)

    def __init__(self, name, location=None, rotation_matrix=None,
                 parent=None, vector_names=None, variable_names=None,
                 latex_vects=None, pretty_vects=None, latex_scalars=None,
                 pretty_scalars=None, transformation=None):
        # Dummy initializer for setting docstring
        pass

    __init__.__doc__ = __new__.__doc__

    @staticmethod
    def _compose_rotation_and_translation(rot, translation, parent):
        r = lambda x, y, z: CoordSys3D._rotation_trans_equations(rot, (x, y, z))
        if parent is None:
            return r

        dx, dy, dz = [translation.dot(i) for i in parent.base_vectors()]
        t = lambda x, y, z: (
            x + dx,
            y + dy,
            z + dz,
        )
        return lambda x, y, z: t(*r(x, y, z))


def _check_strings(arg_name, arg):
    errorstr = arg_name + " must be an iterable of 3 string-types"
    if len(arg) != 3:
        raise ValueError(errorstr)
    for s in arg:
        if not isinstance(s, str):
            raise TypeError(errorstr)


# Delayed import to avoid cyclic import problems:
from sympy.vector.vector import BaseVector

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\arch.py ===
"""
This module can be used to solve probelsm related to 2D parabolic arches
"""
from sympy.core.sympify import sympify
from sympy.core.symbol import Symbol,symbols
from sympy import diff, sqrt, cos , sin, atan, rad, Min
from sympy.core.relational import Eq
from sympy.solvers.solvers import solve
from sympy.functions import Piecewise
from sympy.plotting import plot
from sympy import limit
from sympy.utilities.decorator import doctest_depends_on
from sympy.external.importtools import import_module

numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})

class Arch:
    """
    This class is used to solve problems related to a three hinged arch(determinate) structure.\n
    An arch is a curved vertical structure spanning an open space underneath it.\n
    Arches can be used to reduce the bending moments in long-span structures.\n

    Arches are used in structural engineering(over windows, door and even bridges)\n
    because they can support a very large mass placed on top of them.

    Example
    ========
    >>> from sympy.physics.continuum_mechanics.arch import Arch
    >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
    >>> a.get_shape_eqn
    5 - (x - 5)**2/5

    >>> from sympy.physics.continuum_mechanics.arch import Arch
    >>> a = Arch((0,0),(10,1),crown_x=6)
    >>> a.get_shape_eqn
    9/5 - (x - 6)**2/20
    """
    def __init__(self,left_support,right_support,**kwargs):
        self._shape_eqn = None
        self._left_support  = (sympify(left_support[0]),sympify(left_support[1]))
        self._right_support  = (sympify(right_support[0]),sympify(right_support[1]))
        self._crown_x = None
        self._crown_y = None
        if 'crown_x' in kwargs:
            self._crown_x = sympify(kwargs['crown_x'])
        if 'crown_y' in kwargs:
            self._crown_y = sympify(kwargs['crown_y'])
        self._shape_eqn = self.get_shape_eqn
        self._conc_loads = {}
        self._distributed_loads = {}
        self._loads = {'concentrated': self._conc_loads, 'distributed':self._distributed_loads}
        self._loads_applied = {}
        self._supports = {'left':'hinge', 'right':'hinge'}
        self._member = None
        self._member_force = None
        self._reaction_force = {Symbol('R_A_x'):0, Symbol('R_A_y'):0, Symbol('R_B_x'):0, Symbol('R_B_y'):0}
        self._points_disc_x = set()
        self._points_disc_y = set()
        self._moment_x  = {}
        self._moment_y = {}
        self._load_x = {}
        self._load_y = {}
        self._moment_x_func = Piecewise((0,True))
        self._moment_y_func = Piecewise((0,True))
        self._load_x_func = Piecewise((0,True))
        self._load_y_func = Piecewise((0,True))
        self._bending_moment = None
        self._shear_force = None
        self._axial_force = None
        # self._crown = (sympify(crown[0]),sympify(crown[1]))

    @property
    def get_shape_eqn(self):
        "returns the equation of the shape of arch developed"
        if self._shape_eqn:
            return self._shape_eqn

        x,y,c = symbols('x y c')
        a = Symbol('a',positive=False)
        if self._crown_x and self._crown_y:
            x0  = self._crown_x
            y0 = self._crown_y
            parabola_eqn = a*(x-x0)**2 + y0 - y
            eq1 = parabola_eqn.subs({x:self._left_support[0], y:self._left_support[1]})
            solution = solve((eq1),(a))
            parabola_eqn = solution[0]*(x-x0)**2 + y0
            if(parabola_eqn.subs({x:self._right_support[0]}) != self._right_support[1]):
                raise ValueError("provided coordinates of crown and supports are not consistent with parabolic arch")

        elif self._crown_x:
            x0  = self._crown_x
            parabola_eqn = a*(x-x0)**2 + c - y
            eq1 = parabola_eqn.subs({x:self._left_support[0], y:self._left_support[1]})
            eq2 = parabola_eqn.subs({x:self._right_support[0], y:self._right_support[1]})
            solution = solve((eq1,eq2),(a,c))
            if len(solution) <2 or solution[a] == 0:
                raise ValueError("parabolic arch cannot be constructed with the provided coordinates, try providing crown_y")
            parabola_eqn = solution[a]*(x-x0)**2+ solution[c]
            self._crown_y = solution[c]

        else:
            raise KeyError("please provide crown_x to construct arch")

        return parabola_eqn

    @property
    def get_loads(self):
        """
        return the position of the applied load and angle (for concentrated loads)
        """
        return self._loads

    @property
    def supports(self):
        """
        Returns the type of support
        """
        return self._supports

    @property
    def left_support(self):
        """
        Returns the position of the left support.
        """
        return self._left_support

    @property
    def right_support(self):
        """
        Returns the position of the right support.
        """
        return self._right_support

    @property
    def reaction_force(self):
        """
        return the reaction forces generated
        """
        return self._reaction_force

    def apply_load(self,order,label,start,mag,end=None,angle=None):
        """
        This method adds load to the Arch.

        Parameters
        ==========

            order : Integer
                Order of the applied load.

                    - For point/concentrated loads, order = -1
                    - For distributed load, order = 0

            label : String or Symbol
                The label of the load
                - should not use 'A' or 'B' as it is used for supports.

            start : Float

                    - For concentrated/point loads, start is the x coordinate
                    - For distributed loads, start is the starting position of distributed load

            mag : Sympifyable
                Magnitude of the applied load. Must be positive

            end : Float
                Required for distributed loads

                    - For concentrated/point load , end is None(may not be given)
                    - For distributed loads, end is the end position of distributed load

            angle: Sympifyable
                The angle in degrees, the load vector makes with the horizontal
                in the counter-clockwise direction.

        Examples
        ========
        For applying distributed load

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)

        For applying point/concentrated_loads

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(-1,'C',start=2,mag=15,angle=45)

        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')
        # y0 = Symbol('y0')
        order= sympify(order)
        mag = sympify(mag)
        angle = sympify(angle)

        if label in self._loads_applied:
            raise ValueError("load with the given label already exists")

        if label in ['A','B']:
            raise ValueError("cannot use the given label, reserved for supports")

        if order == 0:
            if end is None or end<start:
                raise KeyError("provide end greater than start")

            self._distributed_loads[label] = {'start':start, 'end':end, 'f_y': mag}
            self._points_disc_y.add(start)

            if start in self._moment_y:
                self._moment_y[start] -= mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
                self._load_y[start] += mag*(Min(end,x)-start)
            else:
                self._moment_y[start] = -mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
                self._load_y[start] = mag*(Min(end,x)-start)

            self._loads_applied[label] = 'distributed'

        if order == -1:

            if angle is None:
                raise TypeError("please provide direction of force")
            height = self._shape_eqn.subs({'x':start})

            self._conc_loads[label] = {'x':start, 'y':height, 'f_x':mag*cos(rad(angle)), 'f_y': mag*sin(rad(angle)), 'mag':mag, 'angle':angle}
            self._points_disc_x.add(start)
            self._points_disc_y.add(start)

            if start in self._moment_x:
                self._moment_x[start] += self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
                self._load_x[start] += self._conc_loads[label]['f_x']
            else:
                self._moment_x[start] = self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
                self._load_x[start] = self._conc_loads[label]['f_x']

            if start in self._moment_y:
                self._moment_y[start] -= self._conc_loads[label]['f_y']*(x0-start)
                self._load_y[start] += self._conc_loads[label]['f_y']
            else:
                self._moment_y[start] = -self._conc_loads[label]['f_y']*(x0-start)
                self._load_y[start] = self._conc_loads[label]['f_y']

            self._loads_applied[label] = 'concentrated'


    def remove_load(self,label):
        """
        This methods removes the load applied to the arch

        Parameters
        ==========

        label : String or Symbol
            The label of the applied load

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)
        >>> a.remove_load('C')
        removed load C: {'start': 3, 'end': 5, 'f_y': -10}
        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')

        if label in self._distributed_loads :

            self._loads_applied.pop(label)
            start = self._distributed_loads[label]['start']
            end = self._distributed_loads[label]['end']
            mag  = self._distributed_loads[label]['f_y']
            self._points_disc_y.remove(start)
            self._load_y[start] -= mag*(Min(x,end)-start)
            self._moment_y[start] += mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
            val = self._distributed_loads.pop(label)
            print(f"removed load {label}: {val}")

        elif label in self._conc_loads :

            self._loads_applied.pop(label)
            start = self._conc_loads[label]['x']
            self._points_disc_x.remove(start)
            self._points_disc_y.remove(start)
            self._moment_y[start] += self._conc_loads[label]['f_y']*(x0-start)
            self._moment_x[start] -= self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
            self._load_x[start] -= self._conc_loads[label]['f_x']
            self._load_y[start] -= self._conc_loads[label]['f_y']
            val = self._conc_loads.pop(label)
            print(f"removed load {label}: {val}")

        else :
            raise ValueError("label not found")

    def change_support_position(self, left_support=None, right_support=None):
        """
        Change position of supports.
        If not provided , defaults to the old value.
        Parameters
        ==========

            left_support: tuple (x, y)
                x: float
                    x-coordinate value of the left_support

                y: float
                    y-coordinate value of the left_support

            right_support: tuple (x, y)
                x: float
                    x-coordinate value of the right_support

                y: float
                    y-coordinate value of the right_support
        """
        if left_support is not None:
            self._left_support = (left_support[0],left_support[1])

        if right_support is not None:
            self._right_support = (right_support[0],right_support[1])

        self._shape_eqn = None
        self._shape_eqn = self.get_shape_eqn

    def change_crown_position(self,crown_x=None,crown_y=None):
        """
        Change the position of the crown/hinge of the arch

        Parameters
        ==========

            crown_x: Float
                The x coordinate of the position of the hinge
                - if not provided, defaults to old value

            crown_y: Float
                The y coordinate of the position of the hinge
                - if not provided defaults to None
        """
        self._crown_x = crown_x
        self._crown_y = crown_y
        self._shape_eqn = None
        self._shape_eqn = self.get_shape_eqn

    def change_support_type(self,left_support=None,right_support=None):
        """
        Add the type for support at each end.
        Can use roller or hinge support at each end.

        Parameters
        ==========

            left_support, right_support : string
                Type of support at respective end

                    - For roller support , left_support/right_support = "roller"
                    - For hinged support, left_support/right_support = "hinge"
                    - defaults to hinge if value not provided

        Examples
        ========

        For applying roller support at right end

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.change_support_type(right_support="roller")

        """
        support_types = ['roller','hinge']
        if left_support:
            if left_support not in support_types:
                raise ValueError("supports must only be roller or hinge")

            self._supports['left'] = left_support

        if right_support:
            if right_support not in support_types:
                raise ValueError("supports must only be roller or hinge")

            self._supports['right'] = right_support

    def add_member(self,y):
        """
        This method adds a member/rod at a particular height y.
        A rod is used for stability of the structure in case of a roller support.
        """
        if y>self._crown_y or y<min(self._left_support[1],  self._right_support[1]):
            raise ValueError(f"position of support must be between y={min(self._left_support[1],  self._right_support[1])} and y={self._crown_y}")
        x = Symbol('x')
        a = diff(self._shape_eqn,x).subs(x,self._crown_x+1)/2
        x_diff = sqrt((y - self._crown_y)/a)
        x1 = self._crown_x + x_diff
        x2 = self._crown_x - x_diff
        self._member = (x1,x2,y)

    def shear_force_at(self, pos = None, **kwargs):
        """
        return the shear at some x-coordinates
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._shear_force
        else:
            x = Symbol('x')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._shear_force,x,pos,dir=dir)
            return self._shear_force.subs(x,pos)

    def bending_moment_at(self, pos = None, **kwargs):
        """
        return the bending moment at some x-coordinates
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._bending_moment
        else:
            x0 = Symbol('x0')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._bending_moment,x0,pos,dir=dir)
            return self._bending_moment.subs(x0,pos)


    def axial_force_at(self,pos = None, **kwargs):
        """
        return the axial/normal force generated at some x-coordinate
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._axial_force
        else:
            x = Symbol('x')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._axial_force,x,pos,dir=dir)
            return self._axial_force.subs(x,pos)

    def solve(self):
        """
        This method solves for the reaction forces generated at the supports,\n
        and bending moment and generated in the arch and tension produced in the member if used.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)
        >>> a.solve()
        >>> a.reaction_force
        {R_A_x: 8, R_A_y: 12, R_B_x: -8, R_B_y: 8}

        >>> from sympy import Symbol
        >>> t = Symbol('t')
        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(16,0),crown_x=8,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=t)
        >>> a.solve()
        >>> a.reaction_force
        {R_A_x: -4*t/5, R_A_y: -3*t/2, R_B_x: 4*t/5, R_B_y: -t/2}

        >>> a.bending_moment_at(4)
        -5*t/2
        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')

        discontinuity_points_x = sorted(self._points_disc_x)
        discontinuity_points_y = sorted(self._points_disc_y)

        self._moment_x_func = Piecewise((0,True))
        self._moment_y_func = Piecewise((0,True))

        self._load_x_func = Piecewise((0,True))
        self._load_y_func = Piecewise((0,True))

        accumulated_x_moment = 0
        accumulated_y_moment = 0

        accumulated_x_load = 0
        accumulated_y_load = 0

        for point in discontinuity_points_x:
            cond = (x >= point)
            accumulated_x_load += self._load_x[point]
            accumulated_x_moment += self._moment_x[point]
            self._load_x_func = Piecewise((accumulated_x_load,cond),(self._load_x_func,True))
            self._moment_x_func = Piecewise((accumulated_x_moment,cond),(self._moment_x_func,True))

        for point in discontinuity_points_y:
            cond = (x >= point)
            accumulated_y_moment += self._moment_y[point]
            accumulated_y_load += self._load_y[point]
            self._load_y_func = Piecewise((accumulated_y_load,cond),(self._load_y_func,True))
            self._moment_y_func = Piecewise((accumulated_y_moment,cond),(self._moment_y_func,True))

        moment_A = self._moment_y_func.subs(x,self._right_support[0]).subs(x0,self._left_support[0]) +\
                   self._moment_x_func.subs(x,self._right_support[0]).subs(y,self._left_support[1])

        moment_hinge_left = self._moment_y_func.subs(x,self._crown_x).subs(x0,self._crown_x) +\
                            self._moment_x_func.subs(x,self._crown_x).subs(y,self._crown_y)

        moment_hinge_right = self._moment_y_func.subs(x,self._right_support[0]).subs(x0,self._crown_x)- \
                             self._moment_y_func.subs(x,self._crown_x).subs(x0,self._crown_x) +\
                             self._moment_x_func.subs(x,self._right_support[0]).subs(y,self._crown_y) -\
                             self._moment_x_func.subs(x,self._crown_x).subs(y,self._crown_y)

        net_x = self._load_x_func.subs(x,self._right_support[0])
        net_y = self._load_y_func.subs(x,self._right_support[0])

        if (self._supports['left']=='roller' or self._supports['right']=='roller') and not self._member:
            print("member must be added if any of the supports is roller")
            return

        R_A_x, R_A_y, R_B_x, R_B_y, T = symbols('R_A_x R_A_y R_B_x R_B_y T')

        if self._supports['left'] == 'roller' and self._supports['right'] == 'roller':

            if self._member[2]>=max(self._left_support[1],self._right_support[1]):

                if net_x!=0:
                    raise ValueError("net force in x direction not possible under the specified conditions")

                else:
                    eq1 = Eq(R_A_x ,0)
                    eq2 = Eq(R_B_x, 0)
                    eq3 = Eq(R_A_y + R_B_y + net_y,0)

                    eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                             R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)

                    eq5 = Eq(moment_hinge_right + R_B_y*(self._right_support[0]-self._crown_x) +\
                             T*(self._member[2]-self._crown_y),0)
                    solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x, 0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         T*(self._member[2]-self._left_support[1])+moment_A,0)
                eq5 = Eq(T+net_x,0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x, 0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])+\
                         T*(self._member[2]-self._left_support[1])+moment_A,0)
                eq5 = Eq(T-net_x,0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

        elif self._supports['left'] == 'roller':
            if self._member[2]>=max(self._left_support[1], self._right_support[1]):
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x+net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)
                eq5 = Eq(moment_hinge_left + R_A_y*(self._left_support[0]-self._crown_x) -\
                         T*(self._member[2]-self._crown_y),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x+ T +net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])-\
                         T*(self._member[2]-self._left_support[0])+moment_A,0)
                eq5 = Eq(moment_hinge_left + R_A_y*(self._left_support[0]-self._crown_x)-\
                         T*(self._member[2]-self._crown_y),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[0]:
                eq1 = Eq(R_A_x,0)
                eq2 = Eq(R_B_x- T +net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(moment_hinge_left+R_A_y*(self._left_support[0]-self._crown_x),0)
                eq5 = Eq(moment_A+R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])+\
                         T*(self._member[2]-self._left_support[1]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

        elif self._supports['right'] == 'roller':
            if self._member[2]>=max(self._left_support[1], self._right_support[1]):
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x)+\
                         T*(self._member[2]-self._crown_y),0)
                eq5 = Eq(moment_A+R_B_y*(self._right_support[0]-self._left_support[0]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x+T+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x),0)
                eq5 = Eq(moment_A-T*(self._member[2]-self._left_support[1])+\
                         R_B_y*(self._right_support[0]-self._left_support[0]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[1]:
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x-T+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x)+\
                         T*(self._member[2]-self._crown_y),0)
                eq5 = Eq(moment_A+T*(self._member[2]-self._left_support[1])+\
                         R_B_y*(self._right_support[0]-self._left_support[0]))
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))
        else:
            eq1 = Eq(R_A_x + R_B_x + net_x,0)
            eq2 = Eq(R_A_y + R_B_y + net_y,0)
            eq3 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                     R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)
            eq4 = Eq(moment_hinge_right + R_B_y*(self._right_support[0]-self._crown_x) -\
                     R_B_x*(self._right_support[1]-self._crown_y),0)
            solution = solve((eq1,eq2,eq3,eq4),(R_A_x,R_A_y,R_B_x,R_B_y))

        for symb in self._reaction_force:
            self._reaction_force[symb] = solution[symb]

        self._bending_moment = - (self._moment_x_func.subs(x,x0) + self._moment_y_func.subs(x,x0) -\
                                  solution[R_A_y]*(x0-self._left_support[0]) +\
                                  solution[R_A_x]*(self._shape_eqn.subs({x:x0})-self._left_support[1]))

        angle  = atan(diff(self._shape_eqn,x))

        fx = (self._load_x_func+solution[R_A_x])
        fy = (self._load_y_func+solution[R_A_y])

        axial_force = fx*cos(angle) + fy*sin(angle)
        shear_force = -fx*sin(angle) + fy*cos(angle)

        self._axial_force = axial_force
        self._shear_force = shear_force

    @doctest_depends_on(modules=('numpy',))
    def draw(self):
        """
        This method returns a plot object containing the diagram of the specified arch along with the supports
        and forces applied to the structure.

        Examples
        ========

        >>> from sympy import Symbol
        >>> t = Symbol('t')
        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(40,0),crown_x=20,crown_y=12)
        >>> a.apply_load(-1,'C',8,150,angle=270)
        >>> a.apply_load(0,'D',start=20,end=40,mag=-4)
        >>> a.apply_load(-1,'E',10,t,angle=300)
        >>> p = a.draw()
        >>> p # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: 11.325 - 3*(x - 20)**2/100 for x over (0.0, 40.0)
        [1]: cartesian line: 12 - 3*(x - 20)**2/100 for x over (0.0, 40.0)
        ...
        >>> p.show()

        """
        x = Symbol('x')
        markers = []
        annotations = self._draw_loads()
        rectangles = []
        supports = self._draw_supports()
        markers+=supports

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        lim = max(xmax*1.1-xmin*0.8+1, ymax*1.1-ymin*0.8+1)

        rectangles = self._draw_rectangles()

        filler = self._draw_filler()
        rectangles+=filler

        if self._member is not None:
            if(self._member[2]>=self._right_support[1]):
                markers.append(
                    {
                        'args':[[self._member[1]+0.005*lim],[self._member[2]]],
                        'marker':'o',
                        'markersize': 4,
                        'color': 'white',
                        'markerfacecolor':'none'
                    }
                )

            if(self._member[2]>=self._left_support[1]):
                markers.append(
                    {
                        'args':[[self._member[0]-0.005*lim],[self._member[2]]],
                        'marker':'o',
                        'markersize': 4,
                        'color': 'white',
                        'markerfacecolor':'none'
                    }
                )



        markers.append({
            'args':[[self._crown_x],[self._crown_y-0.005*lim]],
            'marker':'o',
            'markersize': 5,
            'color':'white',
            'markerfacecolor':'none',
        })

        if lim==xmax*1.1-xmin*0.8+1:

            sing_plot = plot(self._shape_eqn-0.015*lim,
                             self._shape_eqn,
                             (x, self._left_support[0], self._right_support[0]),
                             markers=markers,
                             show=False,
                             annotations=annotations,
                             rectangles = rectangles,
                             xlim=(xmin-0.05*lim, xmax*1.1),
                             ylim=(xmin-0.05*lim, xmax*1.1),
                             axis=False,
                             line_color='brown')

        else:
            sing_plot = plot(self._shape_eqn-0.015*lim,
                             self._shape_eqn,
                             (x, self._left_support[0], self._right_support[0]),
                             markers=markers,
                             show=False,
                             annotations=annotations,
                             rectangles = rectangles,
                             xlim=(ymin-0.05*lim, ymax*1.1),
                             ylim=(ymin-0.05*lim, ymax*1.1),
                             axis=False,
                             line_color='brown')

        return sing_plot


    def _draw_supports(self):
        support_markers = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        if self._supports['left']=='roller':
            support_markers.append(
                {
                    'args':[
                        [self._left_support[0]],
                        [self._left_support[1]-0.02*max_diff]
                    ],
                    'marker':'o',
                    'markersize':11,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )
        else:
            support_markers.append(
                {
                    'args':[
                        [self._left_support[0]],
                        [self._left_support[1]-0.007*max_diff]
                    ],
                    'marker':6,
                    'markersize':15,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )

        if self._supports['right']=='roller':
            support_markers.append(
                {
                    'args':[
                        [self._right_support[0]],
                        [self._right_support[1]-0.02*max_diff]
                    ],
                    'marker':'o',
                    'markersize':11,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )
        else:
            support_markers.append(
                {
                    'args':[
                        [self._right_support[0]],
                        [self._right_support[1]-0.007*max_diff]
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
                    [self._right_support[0]],
                    [self._right_support[1]-0.036*max_diff]
                ],
                'marker':'_',
                'markersize':15,
                'color':'black',
                'markerfacecolor':'none'
            }
        )

        support_markers.append(
            {
                'args':[
                    [self._left_support[0]],
                    [self._left_support[1]-0.036*max_diff]
                ],
                'marker':'_',
                'markersize':15,
                'color':'black',
                'markerfacecolor':'none'
            }
        )

        return support_markers

    def _draw_rectangles(self):
        member = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        if self._member is not None:
            if self._member[2]>= max(self._left_support[1],self._right_support[1]):
                member.append(
                    {
                        'xy':(self._member[0],self._member[2]-0.005*max_diff),
                        'width':self._member[1]-self._member[0],
                        'height': 0.01*max_diff,
                        'angle': 0,
                        'color':'brown',
                    }
                )

            elif self._member[2]>=self._left_support[1]:
                member.append(
                    {
                        'xy':(self._member[0],self._member[2]-0.005*max_diff),
                        'width':self._right_support[0]-self._member[0],
                        'height': 0.01*max_diff,
                        'angle': 0,
                        'color':'brown',
                    }
                )

            else:
                member.append(
                    {
                        'xy':(self._member[1],self._member[2]-0.005*max_diff),
                        'width':abs(self._left_support[0]-self._member[1]),
                        'height': 0.01*max_diff,
                        'angle': 180,
                        'color':'brown',
                    }
                )

        if self._distributed_loads:
            for loads in self._distributed_loads:

                start = self._distributed_loads[loads]['start']
                end = self._distributed_loads[loads]['end']

                member.append(
                    {
                        'xy':(start,self._crown_y+max_diff*0.15),
                        'width': (end-start),
                        'height': max_diff*0.01,
                        'color': 'orange'
                    }
                )


        return member

    def _draw_loads(self):
        load_annotations = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for load in self._conc_loads:
            x = self._conc_loads[load]['x']
            y = self._conc_loads[load]['y']
            angle = self._conc_loads[load]['angle']
            mag = self._conc_loads[load]['mag']
            load_annotations.append(
                {
                    'text':'',
                    'xy':(
                        x+cos(rad(angle))*max_diff*0.08,
                        y+sin(rad(angle))*max_diff*0.08
                    ),
                    'xytext':(x,y),
                    'fontsize':10,
                    'fontweight': 'bold',
                    'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'blue','edgecolor':'blue'}
                }
            )
            load_annotations.append(
                {
                    'text':f'{load}: {mag} N',
                    'fontsize':10,
                    'fontweight': 'bold',
                    'xy': (x+cos(rad(angle))*max_diff*0.12,y+sin(rad(angle))*max_diff*0.12)
                }
            )

        for load in self._distributed_loads:
            start = self._distributed_loads[load]['start']
            end = self._distributed_loads[load]['end']
            mag = self._distributed_loads[load]['f_y']
            x_points = numpy.arange(start,end,(end-start)/(max_diff*0.25))
            x_points = numpy.append(x_points,end)
            for point in x_points:
                if(mag<0):
                    load_annotations.append(
                        {
                            'text':'',
                            'xy':(point,self._crown_y+max_diff*0.05),
                            'xytext': (point,self._crown_y+max_diff*0.15),
                            'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'orange','edgecolor':'orange'}
                        }
                    )
                else:
                    load_annotations.append(
                        {
                            'text':'',
                            'xy':(point,self._crown_y+max_diff*0.2),
                            'xytext': (point,self._crown_y+max_diff*0.15),
                            'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'orange','edgecolor':'orange'}
                        }
                    )
            if(mag<0):
                load_annotations.append(
                    {
                        'text':f'{load}: {abs(mag)} N/m',
                        'fontsize':10,
                        'fontweight': 'bold',
                        'xy':((start+end)/2,self._crown_y+max_diff*0.175)
                    }
                )
            else:
                load_annotations.append(
                    {
                        'text':f'{load}: {abs(mag)} N/m',
                        'fontsize':10,
                        'fontweight': 'bold',
                        'xy':((start+end)/2,self._crown_y+max_diff*0.125)
                    }
                )
        return load_annotations

    def _draw_filler(self):
        x = Symbol('x')
        filler = []
        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        x_points = numpy.arange(self._left_support[0],self._right_support[0],(self._right_support[0]-self._left_support[0])/(max_diff*max_diff))

        for point in x_points:
            filler.append(
                    {
                        'xy':(point,self._shape_eqn.subs(x,point)-max_diff*0.015),
                        'width': (self._right_support[0]-self._left_support[0])/(max_diff*max_diff),
                        'height': max_diff*0.015,
                        'color': 'brown'
                    }
            )

        return filler

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\traversals.py ===
# sql/traversals.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

from __future__ import annotations

from collections import deque
import collections.abc as collections_abc
import itertools
from itertools import zip_longest
import operator
import typing
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type

from . import operators
from .cache_key import HasCacheKey
from .visitors import _TraverseInternalsType
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .. import util
from ..util import langhelpers
from ..util.typing import Self


SKIP_TRAVERSE = util.symbol("skip_traverse")
COMPARE_FAILED = False
COMPARE_SUCCEEDED = True


def compare(obj1: Any, obj2: Any, **kw: Any) -> bool:
    strategy: TraversalComparatorStrategy
    if kw.get("use_proxies", False):
        strategy = ColIdentityComparatorStrategy()
    else:
        strategy = TraversalComparatorStrategy()

    return strategy.compare(obj1, obj2, **kw)


def _preconfigure_traversals(target_hierarchy: Type[Any]) -> None:
    for cls in util.walk_subclasses(target_hierarchy):
        if hasattr(cls, "_generate_cache_attrs") and hasattr(
            cls, "_traverse_internals"
        ):
            cls._generate_cache_attrs()
            _copy_internals.generate_dispatch(
                cls,
                cls._traverse_internals,
                "_generated_copy_internals_traversal",
            )
            _get_children.generate_dispatch(
                cls,
                cls._traverse_internals,
                "_generated_get_children_traversal",
            )


class HasShallowCopy(HasTraverseInternals):
    """attribute-wide operations that are useful for classes that use
    __slots__ and therefore can't operate on their attributes in a dictionary.


    """

    __slots__ = ()

    if typing.TYPE_CHECKING:

        def _generated_shallow_copy_traversal(self, other: Self) -> None: ...

        def _generated_shallow_from_dict_traversal(
            self, d: Dict[str, Any]
        ) -> None: ...

        def _generated_shallow_to_dict_traversal(self) -> Dict[str, Any]: ...

    @classmethod
    def _generate_shallow_copy(
        cls,
        internal_dispatch: _TraverseInternalsType,
        method_name: str,
    ) -> Callable[[Self, Self], None]:
        code = "\n".join(
            f"    other.{attrname} = self.{attrname}"
            for attrname, _ in internal_dispatch
        )
        meth_text = f"def {method_name}(self, other):\n{code}\n"
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    @classmethod
    def _generate_shallow_to_dict(
        cls,
        internal_dispatch: _TraverseInternalsType,
        method_name: str,
    ) -> Callable[[Self], Dict[str, Any]]:
        code = ",\n".join(
            f"    '{attrname}': self.{attrname}"
            for attrname, _ in internal_dispatch
        )
        meth_text = f"def {method_name}(self):\n    return {{{code}}}\n"
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    @classmethod
    def _generate_shallow_from_dict(
        cls,
        internal_dispatch: _TraverseInternalsType,
        method_name: str,
    ) -> Callable[[Self, Dict[str, Any]], None]:
        code = "\n".join(
            f"    self.{attrname} = d['{attrname}']"
            for attrname, _ in internal_dispatch
        )
        meth_text = f"def {method_name}(self, d):\n{code}\n"
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    def _shallow_from_dict(self, d: Dict[str, Any]) -> None:
        cls = self.__class__

        shallow_from_dict: Callable[[HasShallowCopy, Dict[str, Any]], None]
        try:
            shallow_from_dict = cls.__dict__[
                "_generated_shallow_from_dict_traversal"
            ]
        except KeyError:
            shallow_from_dict = self._generate_shallow_from_dict(
                cls._traverse_internals,
                "_generated_shallow_from_dict_traversal",
            )

            cls._generated_shallow_from_dict_traversal = shallow_from_dict  # type: ignore  # noqa: E501

        shallow_from_dict(self, d)

    def _shallow_to_dict(self) -> Dict[str, Any]:
        cls = self.__class__

        shallow_to_dict: Callable[[HasShallowCopy], Dict[str, Any]]

        try:
            shallow_to_dict = cls.__dict__[
                "_generated_shallow_to_dict_traversal"
            ]
        except KeyError:
            shallow_to_dict = self._generate_shallow_to_dict(
                cls._traverse_internals, "_generated_shallow_to_dict_traversal"
            )

            cls._generated_shallow_to_dict_traversal = shallow_to_dict  # type: ignore  # noqa: E501
        return shallow_to_dict(self)

    def _shallow_copy_to(self, other: Self) -> None:
        cls = self.__class__

        shallow_copy: Callable[[Self, Self], None]
        try:
            shallow_copy = cls.__dict__["_generated_shallow_copy_traversal"]
        except KeyError:
            shallow_copy = self._generate_shallow_copy(
                cls._traverse_internals, "_generated_shallow_copy_traversal"
            )

            cls._generated_shallow_copy_traversal = shallow_copy  # type: ignore  # noqa: E501
        shallow_copy(self, other)

    def _clone(self, **kw: Any) -> Self:
        """Create a shallow copy"""
        c = self.__class__.__new__(self.__class__)
        self._shallow_copy_to(c)
        return c


class GenerativeOnTraversal(HasShallowCopy):
    """Supplies Generative behavior but making use of traversals to shallow
    copy.

    .. seealso::

        :class:`sqlalchemy.sql.base.Generative`


    """

    __slots__ = ()

    def _generate(self) -> Self:
        cls = self.__class__
        s = cls.__new__(cls)
        self._shallow_copy_to(s)
        return s


def _clone(element, **kw):
    return element._clone()


class HasCopyInternals(HasTraverseInternals):
    __slots__ = ()

    def _clone(self, **kw):
        raise NotImplementedError()

    def _copy_internals(
        self, *, omit_attrs: Iterable[str] = (), **kw: Any
    ) -> None:
        """Reassign internal elements to be clones of themselves.

        Called during a copy-and-traverse operation on newly
        shallow-copied elements to create a deep copy.

        The given clone function should be used, which may be applying
        additional transformations to the element (i.e. replacement
        traversal, cloned traversal, annotations).

        """

        try:
            traverse_internals = self._traverse_internals
        except AttributeError:
            # user-defined classes may not have a _traverse_internals
            return

        for attrname, obj, meth in _copy_internals.run_generated_dispatch(
            self, traverse_internals, "_generated_copy_internals_traversal"
        ):
            if attrname in omit_attrs:
                continue

            if obj is not None:
                result = meth(attrname, self, obj, **kw)
                if result is not None:
                    setattr(self, attrname, result)


class _CopyInternalsTraversal(HasTraversalDispatch):
    """Generate a _copy_internals internal traversal dispatch for classes
    with a _traverse_internals collection."""

    def visit_clauseelement(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return clone(element, **kw)

    def visit_clauseelement_list(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return [clone(clause, **kw) for clause in element]

    def visit_clauseelement_tuple(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return tuple([clone(clause, **kw) for clause in element])

    def visit_executable_options(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return tuple([clone(clause, **kw) for clause in element])

    def visit_clauseelement_unordered_set(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return {clone(clause, **kw) for clause in element}

    def visit_clauseelement_tuples(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return [
            tuple(clone(tup_elem, **kw) for tup_elem in elem)
            for elem in element
        ]

    def visit_string_clauseelement_dict(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return {key: clone(value, **kw) for key, value in element.items()}

    def visit_setup_join_tuple(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return tuple(
            (
                clone(target, **kw) if target is not None else None,
                clone(onclause, **kw) if onclause is not None else None,
                clone(from_, **kw) if from_ is not None else None,
                flags,
            )
            for (target, onclause, from_, flags) in element
        )

    def visit_memoized_select_entities(self, attrname, parent, element, **kw):
        return self.visit_clauseelement_tuple(attrname, parent, element, **kw)

    def visit_dml_ordered_values(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        # sequence of 2-tuples
        return [
            (
                (
                    clone(key, **kw)
                    if hasattr(key, "__clause_element__")
                    else key
                ),
                clone(value, **kw),
            )
            for key, value in element
        ]

    def visit_dml_values(self, attrname, parent, element, clone=_clone, **kw):
        return {
            (
                clone(key, **kw) if hasattr(key, "__clause_element__") else key
            ): clone(value, **kw)
            for key, value in element.items()
        }

    def visit_dml_multi_values(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        # sequence of sequences, each sequence contains a list/dict/tuple

        def copy(elem):
            if isinstance(elem, (list, tuple)):
                return [
                    (
                        clone(value, **kw)
                        if hasattr(value, "__clause_element__")
                        else value
                    )
                    for value in elem
                ]
            elif isinstance(elem, dict):
                return {
                    (
                        clone(key, **kw)
                        if hasattr(key, "__clause_element__")
                        else key
                    ): (
                        clone(value, **kw)
                        if hasattr(value, "__clause_element__")
                        else value
                    )
                    for key, value in elem.items()
                }
            else:
                # TODO: use abc classes
                assert False

        return [
            [copy(sub_element) for sub_element in sequence]
            for sequence in element
        ]

    def visit_propagate_attrs(
        self, attrname, parent, element, clone=_clone, **kw
    ):
        return element


_copy_internals = _CopyInternalsTraversal()


def _flatten_clauseelement(element):
    while hasattr(element, "__clause_element__") and not getattr(
        element, "is_clause_element", False
    ):
        element = element.__clause_element__()

    return element


class _GetChildrenTraversal(HasTraversalDispatch):
    """Generate a _children_traversal internal traversal dispatch for classes
    with a _traverse_internals collection."""

    def visit_has_cache_key(self, element, **kw):
        # the GetChildren traversal refers explicitly to ClauseElement
        # structures.  Within these, a plain HasCacheKey is not a
        # ClauseElement, so don't include these.
        return ()

    def visit_clauseelement(self, element, **kw):
        return (element,)

    def visit_clauseelement_list(self, element, **kw):
        return element

    def visit_clauseelement_tuple(self, element, **kw):
        return element

    def visit_clauseelement_tuples(self, element, **kw):
        return itertools.chain.from_iterable(element)

    def visit_fromclause_canonical_column_collection(self, element, **kw):
        return ()

    def visit_string_clauseelement_dict(self, element, **kw):
        return element.values()

    def visit_fromclause_ordered_set(self, element, **kw):
        return element

    def visit_clauseelement_unordered_set(self, element, **kw):
        return element

    def visit_setup_join_tuple(self, element, **kw):
        for target, onclause, from_, flags in element:
            if from_ is not None:
                yield from_

            if not isinstance(target, str):
                yield _flatten_clauseelement(target)

            if onclause is not None and not isinstance(onclause, str):
                yield _flatten_clauseelement(onclause)

    def visit_memoized_select_entities(self, element, **kw):
        return self.visit_clauseelement_tuple(element, **kw)

    def visit_dml_ordered_values(self, element, **kw):
        for k, v in element:
            if hasattr(k, "__clause_element__"):
                yield k
            yield v

    def visit_dml_values(self, element, **kw):
        expr_values = {k for k in element if hasattr(k, "__clause_element__")}
        str_values = expr_values.symmetric_difference(element)

        for k in sorted(str_values):
            yield element[k]
        for k in expr_values:
            yield k
            yield element[k]

    def visit_dml_multi_values(self, element, **kw):
        return ()

    def visit_propagate_attrs(self, element, **kw):
        return ()


_get_children = _GetChildrenTraversal()


@util.preload_module("sqlalchemy.sql.elements")
def _resolve_name_for_compare(element, name, anon_map, **kw):
    if isinstance(name, util.preloaded.sql_elements._anonymous_label):
        name = name.apply_map(anon_map)

    return name


class TraversalComparatorStrategy(HasTraversalDispatch, util.MemoizedSlots):
    __slots__ = "stack", "cache", "anon_map"

    def __init__(self):
        self.stack: Deque[
            Tuple[
                Optional[ExternallyTraversible],
                Optional[ExternallyTraversible],
            ]
        ] = deque()
        self.cache = set()

    def _memoized_attr_anon_map(self):
        return (anon_map(), anon_map())

    def compare(
        self,
        obj1: ExternallyTraversible,
        obj2: ExternallyTraversible,
        **kw: Any,
    ) -> bool:
        stack = self.stack
        cache = self.cache

        compare_annotations = kw.get("compare_annotations", False)

        stack.append((obj1, obj2))

        while stack:
            left, right = stack.popleft()

            if left is right:
                continue
            elif left is None or right is None:
                # we know they are different so no match
                return False
            elif (left, right) in cache:
                continue
            cache.add((left, right))

            visit_name = left.__visit_name__
            if visit_name != right.__visit_name__:
                return False

            meth = getattr(self, "compare_%s" % visit_name, None)

            if meth:
                attributes_compared = meth(left, right, **kw)
                if attributes_compared is COMPARE_FAILED:
                    return False
                elif attributes_compared is SKIP_TRAVERSE:
                    continue

                # attributes_compared is returned as a list of attribute
                # names that were "handled" by the comparison method above.
                # remaining attribute names in the _traverse_internals
                # will be compared.
            else:
                attributes_compared = ()

            for (
                (left_attrname, left_visit_sym),
                (right_attrname, right_visit_sym),
            ) in zip_longest(
                left._traverse_internals,
                right._traverse_internals,
                fillvalue=(None, None),
            ):
                if not compare_annotations and (
                    (left_attrname == "_annotations")
                    or (right_attrname == "_annotations")
                ):
                    continue

                if (
                    left_attrname != right_attrname
                    or left_visit_sym is not right_visit_sym
                ):
                    return False
                elif left_attrname in attributes_compared:
                    continue

                assert left_visit_sym is not None
                assert left_attrname is not None
                assert right_attrname is not None

                dispatch = self.dispatch(left_visit_sym)
                assert dispatch is not None, (
                    f"{self.__class__} has no dispatch for "
                    f"'{self._dispatch_lookup[left_visit_sym]}'"
                )
                left_child = operator.attrgetter(left_attrname)(left)
                right_child = operator.attrgetter(right_attrname)(right)
                if left_child is None:
                    if right_child is not None:
                        return False
                    else:
                        continue
                elif right_child is None:
                    return False

                comparison = dispatch(
                    left_attrname, left, left_child, right, right_child, **kw
                )
                if comparison is COMPARE_FAILED:
                    return False

        return True

    def compare_inner(self, obj1, obj2, **kw):
        comparator = self.__class__()
        return comparator.compare(obj1, obj2, **kw)

    def visit_has_cache_key(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        if left._gen_cache_key(self.anon_map[0], []) != right._gen_cache_key(
            self.anon_map[1], []
        ):
            return COMPARE_FAILED

    def visit_propagate_attrs(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return self.compare_inner(
            left.get("plugin_subject", None), right.get("plugin_subject", None)
        )

    def visit_has_cache_key_list(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for l, r in zip_longest(left, right, fillvalue=None):
            if l is None:
                if r is not None:
                    return COMPARE_FAILED
                else:
                    continue
            elif r is None:
                return COMPARE_FAILED

            if l._gen_cache_key(self.anon_map[0], []) != r._gen_cache_key(
                self.anon_map[1], []
            ):
                return COMPARE_FAILED

    def visit_executable_options(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for l, r in zip_longest(left, right, fillvalue=None):
            if l is None:
                if r is not None:
                    return COMPARE_FAILED
                else:
                    continue
            elif r is None:
                return COMPARE_FAILED

            if (
                l._gen_cache_key(self.anon_map[0], [])
                if l._is_has_cache_key
                else l
            ) != (
                r._gen_cache_key(self.anon_map[1], [])
                if r._is_has_cache_key
                else r
            ):
                return COMPARE_FAILED

    def visit_clauseelement(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        self.stack.append((left, right))

    def visit_fromclause_canonical_column_collection(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for lcol, rcol in zip_longest(left, right, fillvalue=None):
            self.stack.append((lcol, rcol))

    def visit_fromclause_derived_column_collection(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        pass

    def visit_string_clauseelement_dict(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for lstr, rstr in zip_longest(
            sorted(left), sorted(right), fillvalue=None
        ):
            if lstr != rstr:
                return COMPARE_FAILED
            self.stack.append((left[lstr], right[rstr]))

    def visit_clauseelement_tuples(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for ltup, rtup in zip_longest(left, right, fillvalue=None):
            if ltup is None or rtup is None:
                return COMPARE_FAILED

            for l, r in zip_longest(ltup, rtup, fillvalue=None):
                self.stack.append((l, r))

    def visit_clauseelement_list(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for l, r in zip_longest(left, right, fillvalue=None):
            self.stack.append((l, r))

    def visit_clauseelement_tuple(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for l, r in zip_longest(left, right, fillvalue=None):
            self.stack.append((l, r))

    def _compare_unordered_sequences(self, seq1, seq2, **kw):
        if seq1 is None:
            return seq2 is None

        completed: Set[object] = set()
        for clause in seq1:
            for other_clause in set(seq2).difference(completed):
                if self.compare_inner(clause, other_clause, **kw):
                    completed.add(other_clause)
                    break
        return len(completed) == len(seq1) == len(seq2)

    def visit_clauseelement_unordered_set(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return self._compare_unordered_sequences(left, right, **kw)

    def visit_fromclause_ordered_set(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for l, r in zip_longest(left, right, fillvalue=None):
            self.stack.append((l, r))

    def visit_string(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left == right

    def visit_string_list(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left == right

    def visit_string_multi_dict(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for lk, rk in zip_longest(
            sorted(left.keys()), sorted(right.keys()), fillvalue=(None, None)
        ):
            if lk != rk:
                return COMPARE_FAILED

            lv, rv = left[lk], right[rk]

            lhc = isinstance(left, HasCacheKey)
            rhc = isinstance(right, HasCacheKey)
            if lhc and rhc:
                if lv._gen_cache_key(
                    self.anon_map[0], []
                ) != rv._gen_cache_key(self.anon_map[1], []):
                    return COMPARE_FAILED
            elif lhc != rhc:
                return COMPARE_FAILED
            elif lv != rv:
                return COMPARE_FAILED

    def visit_multi(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        lhc = isinstance(left, HasCacheKey)
        rhc = isinstance(right, HasCacheKey)
        if lhc and rhc:
            if left._gen_cache_key(
                self.anon_map[0], []
            ) != right._gen_cache_key(self.anon_map[1], []):
                return COMPARE_FAILED
        elif lhc != rhc:
            return COMPARE_FAILED
        else:
            return left == right

    def visit_anon_name(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return _resolve_name_for_compare(
            left_parent, left, self.anon_map[0], **kw
        ) == _resolve_name_for_compare(
            right_parent, right, self.anon_map[1], **kw
        )

    def visit_boolean(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left == right

    def visit_operator(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left == right

    def visit_type(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left._compare_type_affinity(right)

    def visit_plain_dict(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left == right

    def visit_dialect_options(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left == right

    def visit_annotations_key(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        if left and right:
            return (
                left_parent._annotations_cache_key
                == right_parent._annotations_cache_key
            )
        else:
            return left == right

    def visit_with_context_options(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return tuple((fn.__code__, c_key) for fn, c_key in left) == tuple(
            (fn.__code__, c_key) for fn, c_key in right
        )

    def visit_plain_obj(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left == right

    def visit_named_ddl_element(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        if left is None:
            if right is not None:
                return COMPARE_FAILED

        return left.name == right.name

    def visit_prefix_sequence(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for (l_clause, l_str), (r_clause, r_str) in zip_longest(
            left, right, fillvalue=(None, None)
        ):
            if l_str != r_str:
                return COMPARE_FAILED
            else:
                self.stack.append((l_clause, r_clause))

    def visit_setup_join_tuple(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        # TODO: look at attrname for "legacy_join" and use different structure
        for (
            (l_target, l_onclause, l_from, l_flags),
            (r_target, r_onclause, r_from, r_flags),
        ) in zip_longest(left, right, fillvalue=(None, None, None, None)):
            if l_flags != r_flags:
                return COMPARE_FAILED
            self.stack.append((l_target, r_target))
            self.stack.append((l_onclause, r_onclause))
            self.stack.append((l_from, r_from))

    def visit_memoized_select_entities(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return self.visit_clauseelement_tuple(
            attrname, left_parent, left, right_parent, right, **kw
        )

    def visit_table_hint_list(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        left_keys = sorted(left, key=lambda elem: (elem[0].fullname, elem[1]))
        right_keys = sorted(
            right, key=lambda elem: (elem[0].fullname, elem[1])
        )
        for (ltable, ldialect), (rtable, rdialect) in zip_longest(
            left_keys, right_keys, fillvalue=(None, None)
        ):
            if ldialect != rdialect:
                return COMPARE_FAILED
            elif left[(ltable, ldialect)] != right[(rtable, rdialect)]:
                return COMPARE_FAILED
            else:
                self.stack.append((ltable, rtable))

    def visit_statement_hint_list(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        return left == right

    def visit_unknown_structure(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        raise NotImplementedError()

    def visit_dml_ordered_values(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        # sequence of tuple pairs

        for (lk, lv), (rk, rv) in zip_longest(
            left, right, fillvalue=(None, None)
        ):
            if not self._compare_dml_values_or_ce(lk, rk, **kw):
                return COMPARE_FAILED

    def _compare_dml_values_or_ce(self, lv, rv, **kw):
        lvce = hasattr(lv, "__clause_element__")
        rvce = hasattr(rv, "__clause_element__")
        if lvce != rvce:
            return False
        elif lvce and not self.compare_inner(lv, rv, **kw):
            return False
        elif not lvce and lv != rv:
            return False
        elif not self.compare_inner(lv, rv, **kw):
            return False

        return True

    def visit_dml_values(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        if left is None or right is None or len(left) != len(right):
            return COMPARE_FAILED

        if isinstance(left, collections_abc.Sequence):
            for lv, rv in zip(left, right):
                if not self._compare_dml_values_or_ce(lv, rv, **kw):
                    return COMPARE_FAILED
        elif isinstance(right, collections_abc.Sequence):
            return COMPARE_FAILED
        else:
            # dictionaries guaranteed to support insert ordering in
            # py37 so that we can compare the keys in order.  without
            # this, we can't compare SQL expression keys because we don't
            # know which key is which
            for (lk, lv), (rk, rv) in zip(left.items(), right.items()):
                if not self._compare_dml_values_or_ce(lk, rk, **kw):
                    return COMPARE_FAILED
                if not self._compare_dml_values_or_ce(lv, rv, **kw):
                    return COMPARE_FAILED

    def visit_dml_multi_values(
        self, attrname, left_parent, left, right_parent, right, **kw
    ):
        for lseq, rseq in zip_longest(left, right, fillvalue=None):
            if lseq is None or rseq is None:
                return COMPARE_FAILED

            for ld, rd in zip_longest(lseq, rseq, fillvalue=None):
                if (
                    self.visit_dml_values(
                        attrname, left_parent, ld, right_parent, rd, **kw
                    )
                    is COMPARE_FAILED
                ):
                    return COMPARE_FAILED

    def compare_expression_clauselist(self, left, right, **kw):
        if left.operator is right.operator:
            if operators.is_associative(left.operator):
                if self._compare_unordered_sequences(
                    left.clauses, right.clauses, **kw
                ):
                    return ["operator", "clauses"]
                else:
                    return COMPARE_FAILED
            else:
                return ["operator"]
        else:
            return COMPARE_FAILED

    def compare_clauselist(self, left, right, **kw):
        return self.compare_expression_clauselist(left, right, **kw)

    def compare_binary(self, left, right, **kw):
        if left.operator == right.operator:
            if operators.is_commutative(left.operator):
                if (
                    self.compare_inner(left.left, right.left, **kw)
                    and self.compare_inner(left.right, right.right, **kw)
                ) or (
                    self.compare_inner(left.left, right.right, **kw)
                    and self.compare_inner(left.right, right.left, **kw)
                ):
                    return ["operator", "negate", "left", "right"]
                else:
                    return COMPARE_FAILED
            else:
                return ["operator", "negate"]
        else:
            return COMPARE_FAILED

    def compare_bindparam(self, left, right, **kw):
        compare_keys = kw.pop("compare_keys", True)
        compare_values = kw.pop("compare_values", True)

        if compare_values:
            omit = []
        else:
            # this means, "skip these, we already compared"
            omit = ["callable", "value"]

        if not compare_keys:
            omit.append("key")

        return omit


class ColIdentityComparatorStrategy(TraversalComparatorStrategy):
    def compare_column_element(
        self, left, right, use_proxies=True, equivalents=(), **kw
    ):
        """Compare ColumnElements using proxies and equivalent collections.

        This is a comparison strategy specific to the ORM.
        """

        to_compare = (right,)
        if equivalents and right in equivalents:
            to_compare = equivalents[right].union(to_compare)

        for oth in to_compare:
            if use_proxies and left.shares_lineage(oth):
                return SKIP_TRAVERSE
            elif hash(left) == hash(right):
                return SKIP_TRAVERSE
        else:
            return COMPARE_FAILED

    def compare_column(self, left, right, **kw):
        return self.compare_column_element(left, right, **kw)

    def compare_label(self, left, right, **kw):
        return self.compare_column_element(left, right, **kw)

    def compare_table(self, left, right, **kw):
        # tables compare on identity, since it's not really feasible to
        # compare them column by column with the above rules
        return SKIP_TRAVERSE if left is right else COMPARE_FAILED

# === atelier-kyo-manager/app\routes\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\quant_utils.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
import logging
import os
import tempfile
from enum import Enum
from pathlib import Path

import numpy
import onnx
from onnx import ModelProto, TensorProto, external_data_helper
from onnx import onnx_pb as onnx_proto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

try:
    from onnx.reference.custom_element_types import float8e4m3fn
except ImportError:
    float8e4m3fn = None

# INT4 np.dtypes added in ONNX 1.16. These map to np.int8/np.uint8 because numpy
# does not support sub-byte types.
try:
    from onnx.reference.custom_element_types import int4, uint4
except ImportError:
    int4 = None
    uint4 = None

try:
    from onnx.reference.op_run import to_array_extended
except ImportError:
    # old version of onnx.
    to_array_extended = None


__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"
QUANT_OP_NAME = "QuantizeLinear"
QUANT_INPUT_SUFFIX = "_QuantizeLinear_Input"
DEQUANT_OP_NAME = "DequantizeLinear"
DEQUANT_OUTPUT_SUFFIX = "_DequantizeLinear_Output"
TENSOR_NAME_QUANT_SUFFIX = "_quantized"
MODEL_SIZE_THRESHOLD = 2147483648  # Quant model should use external data if >= 2GB

FLOAT8_DISTRIBUTIONS = {}

type_to_name = {getattr(TensorProto, k): k for k in dir(TensorProto) if isinstance(getattr(TensorProto, k), int)}

# Quantization mode
# IntegerOps: Use IntegerOps in quantized model. Only ConvInteger and MatMulInteger ops are supported now.
# QLinearOps: Use QLinearOps in quantized model. Only QLinearConv and QLinearMatMul ops are supported now.


class QuantizationMode(Enum):
    IntegerOps = 0
    QLinearOps = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(mode):
        try:
            return QuantizationMode[mode]
        except KeyError:
            raise ValueError()  # noqa: B904


class QuantizedValueType(Enum):
    Input = 0
    Initializer = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(v):
        try:
            return QuantizedValueType[v]
        except KeyError:
            raise ValueError()  # noqa: B904


class QuantType(Enum):
    QInt8 = 0
    QUInt8 = 1
    QFLOAT8E4M3FN = 2
    QInt16 = 3
    QUInt16 = 4
    QInt4 = 5
    QUInt4 = 6

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(t):
        try:
            return QuantType[t]
        except KeyError:
            raise ValueError()  # noqa: B904

    @property
    def tensor_type(self):
        if self == QuantType.QInt8:
            return TensorProto.INT8
        if self == QuantType.QUInt8:
            return TensorProto.UINT8
        if self == QuantType.QUInt16:
            return TensorProto.UINT16
        if self == QuantType.QInt16:
            return TensorProto.INT16
        if self == QuantType.QFLOAT8E4M3FN:
            return TensorProto.FLOAT8E4M3FN
        if self == QuantType.QUInt4:
            return TensorProto.UINT4
        if self == QuantType.QInt4:
            return TensorProto.INT4
        raise ValueError(f"Unexpected value qtype={self!r}.")


class QuantFormat(Enum):
    QOperator = 0
    QDQ = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(format):
        try:
            return QuantFormat[format]
        except KeyError:
            raise ValueError()  # noqa: B904


ONNX_TYPE_TO_NP_TYPE = {
    onnx_proto.TensorProto.INT8: numpy.dtype("int8"),
    onnx_proto.TensorProto.UINT8: numpy.dtype("uint8"),
    onnx_proto.TensorProto.INT16: numpy.dtype("int16"),
    onnx_proto.TensorProto.UINT16: numpy.dtype("uint16"),
    onnx_proto.TensorProto.FLOAT8E4M3FN: float8e4m3fn,
    onnx_proto.TensorProto.INT4: int4,  # base_dtype is np.int8
    onnx_proto.TensorProto.UINT4: uint4,  # base_dtype is np.uint8
}

ONNX_INT_TYPE_RANGE = {
    onnx_proto.TensorProto.UINT8: (numpy.array(0, dtype=numpy.uint8), numpy.array(255, dtype=numpy.uint8)),
    onnx_proto.TensorProto.INT8: (numpy.array(-128, dtype=numpy.int8), numpy.array(127, dtype=numpy.int8)),
    onnx_proto.TensorProto.UINT16: (numpy.array(0, dtype=numpy.uint16), numpy.array(65535, dtype=numpy.uint16)),
    onnx_proto.TensorProto.INT16: (numpy.array(-32768, dtype=numpy.int16), numpy.array(32767, dtype=numpy.int16)),
    onnx_proto.TensorProto.UINT4: (numpy.array(0, dtype=uint4), numpy.array(15, dtype=uint4)),
    onnx_proto.TensorProto.INT4: (numpy.array(-8, dtype=int4), numpy.array(7, dtype=int4)),
}

ONNX_INT_TYPE_SYMMETRIC_RANGE = {
    onnx_proto.TensorProto.UINT8: (numpy.array(0, dtype=numpy.uint8), numpy.array(254, dtype=numpy.uint8)),
    onnx_proto.TensorProto.INT8: (numpy.array(-127, dtype=numpy.int8), numpy.array(127, dtype=numpy.int8)),
    onnx_proto.TensorProto.UINT16: (numpy.array(0, dtype=numpy.uint16), numpy.array(65534, dtype=numpy.uint16)),
    onnx_proto.TensorProto.INT16: (numpy.array(-32767, dtype=numpy.int16), numpy.array(32767, dtype=numpy.int16)),
}

ONNX_INT_TYPE_REDUCED_RANGE = {
    onnx_proto.TensorProto.UINT8: (numpy.array(0, dtype=numpy.uint8), numpy.array(127, dtype=numpy.uint8)),
    onnx_proto.TensorProto.INT8: (numpy.array(-64, dtype=numpy.int8), numpy.array(64, dtype=numpy.int8)),
    onnx_proto.TensorProto.UINT16: (numpy.array(0, dtype=numpy.uint16), numpy.array(32767, dtype=numpy.uint16)),
    onnx_proto.TensorProto.INT16: (numpy.array(-16384, dtype=numpy.int16), numpy.array(16384, dtype=numpy.int16)),
    onnx_proto.TensorProto.UINT4: (numpy.array(0, dtype=int4), numpy.array(7, dtype=int4)),
    onnx_proto.TensorProto.INT4: (numpy.array(-4, dtype=int4), numpy.array(3, dtype=int4)),
}


def _check_type(*args, zero_point_index=-1):
    new_args = []
    for i, a in enumerate(args):
        if numpy.issubdtype(type(a), numpy.number):
            new_args.append(numpy.array(a))
        elif isinstance(a, numpy.ndarray):
            new_args.append(a)
        else:
            raise TypeError(f"arg {i} is not an array: {a}")
        if i == zero_point_index:
            v = new_args[-1]
            if v.dtype == numpy.float32 or v.dtype == numpy.float16:
                raise TypeError(f"zero_point cannot be {v.dtype}")
    return tuple(new_args) if len(new_args) > 1 else new_args[0]


def quantize_nparray(qType, arr, scale, zero_point, low=None, high=None):
    assert qType in ONNX_TYPE_TO_NP_TYPE, (
        f"Unexpected data type {qType} requested. Only INT8, UINT8, INT16, and UINT16 are supported."
    )
    if qType in (
        onnx_proto.TensorProto.FLOAT8E4M3FN,
        onnx_proto.TensorProto.FLOAT8E4M3FNUZ,
        onnx_proto.TensorProto.FLOAT8E5M2,
        onnx_proto.TensorProto.FLOAT8E5M2FNUZ,
    ):
        if zero_point != 0:
            raise NotImplementedError(f"zero_point is expected to be null for float 8 not {zero_point!r}.")
        if arr.dtype == numpy.float32:
            onnx_type = TensorProto.FLOAT
        elif arr.dtype == numpy.float16:
            onnx_type = TensorProto.FLOAT16
        else:
            raise ValueError(f"Unexpected dtype {arr.dtype}.")
        onnx_model = make_model(
            make_graph(
                [
                    make_node(
                        "Constant", [], ["zero_point"], value=onnx.helper.make_tensor("zero_point", qType, [], [0])
                    ),
                    make_node("QuantizeLinear", ["X", "scale", "zero_point"], ["Y"]),
                ],
                "qu",
                [
                    make_tensor_value_info("X", onnx_type, None),
                    make_tensor_value_info("scale", onnx_type, None),
                ],
                [make_tensor_value_info("Y", qType, None)],
            )
        )
        ref = ReferenceEvaluator(onnx_model)
        return _check_type(ref.run(None, {"X": arr, "scale": scale})[0])
    else:
        # Quantizes data for all integer types.
        #
        # For int4 types, the quantized data is returned as either np.int8 or np.uint8,
        # which matches the python reference ONNX implementation of QuantizeLinear.
        # This data can be packed into 4-bit elements by using pack_bytes_to_4bit().
        dtype = ONNX_TYPE_TO_NP_TYPE[qType]
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range=False, symmetric=False)

        cliplow = max(qmin, low) if low is not None else qmin
        cliphigh = min(qmax, high) if high is not None else qmax
        arr_fp32 = numpy.asarray((arr.astype(numpy.float32) / scale).round() + zero_point)
        numpy.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
        return _check_type(arr_fp32.astype(dtype))


def compute_scale_zp(rmin, rmax, qmin, qmax, symmetric=False, min_real_range=None):
    """Calculate the scale s and zero point z for the quantization relation
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values.

    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).

    :parameter rmin: minimum value of r
    :parameter rmax: maximum value of r
    :parameter qmin: minimum value representable by the target quantization data type
    :parameter qmax: maximum value representable by the target quantization data type
    :parameter symmetric: True if the floating-point range should be made symmetric. Defaults to False.
    :parameter min_real_range: Minimum floating-point range (i.e., rmax - rmin) to enforce. Defaults to None.
    :return: zero and scale [z, s]

    """
    if qmin > 0 or qmax < 0:
        raise ValueError(f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin}, qmmax:{qmax}")

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = numpy.minimum(rmin, numpy.array(0, dtype=rmin.dtype))
    rmax = numpy.maximum(rmax, numpy.array(0, dtype=rmax.dtype))

    # Ensure a minimum float-point range if specified.
    if min_real_range is not None:
        rmax = max(rmax, rmin + numpy.asarray(min_real_range, dtype=rmin.dtype))

    if symmetric:
        absmax = numpy.maximum(numpy.abs(rmin), numpy.abs(rmax))
        rmin = -absmax
        rmax = +absmax

    assert qmin <= qmax, f"qmin={rmin} > qmax={rmax}"
    dr = numpy.array(rmax - rmin, dtype=numpy.float64)
    dq = numpy.array(qmax, dtype=numpy.float64) - numpy.array(qmin, dtype=numpy.float64)
    scale = numpy.array(dr / dq)
    assert scale >= 0, "scale issue"
    if scale < numpy.finfo(rmax.dtype).tiny:
        scale = numpy.array(1.0, dtype=rmax.dtype)
        zero_point = numpy.array(0, dtype=qmin.dtype)
    else:
        if symmetric:
            # When symmetric (i.e., rmax == -rmin), the zero_point formula reduces to round((qmax + qmin) / 2.0).
            # This simpler formula doesn't depend on scale and guarantees that the zero point values
            # for int8, uint8, int16, and uint16 are always 0, 128, 0, and 32768, respectively.
            # This is important for per-channel/symmetric QLinearConv on CPU EP, which requires all channels to have
            # the exact same zero_point values.
            zero_point = numpy.array(
                numpy.round((qmin + qmax) / numpy.array(2.0, dtype=numpy.float64)), dtype=qmin.dtype
            )
        else:
            zero_point = numpy.array(numpy.round(qmin - rmin / scale), dtype=qmin.dtype)
        scale = scale.astype(rmax.dtype)

    return [zero_point, scale]


def compute_scale_zp_float8(element_type, std):
    """Calculate the scale s for a float8 type (E4M3FN).
    The function assumes the coefficient distribution and the float 8
    distribution are similar to two gaussian laws.

    :return: zero and scale [z, s]

    More details in notebook `quantization_fp8.ipynb
    <https://github.com/microsoft/onnxruntime/blob/main/docs/python/notebooks/quantization_fp8.ipynb>`_.
    """
    zp_dtype = None
    if element_type not in FLOAT8_DISTRIBUTIONS:
        if element_type == TensorProto.FLOAT8E4M3FN:
            from onnx.numpy_helper import float8e4m3_to_float32
            from onnx.reference.custom_element_types import float8e4m3fn

            zp_dtype = float8e4m3fn
            all_values = [float8e4m3_to_float32(i) for i in range(256)]
            values = numpy.array(
                [f for f in all_values if not numpy.isnan(f) and not numpy.isinf(f)], dtype=numpy.float32
            )
        else:
            raise ValueError(f"Quantization to element_type={element_type} not implemented.")
        FLOAT8_DISTRIBUTIONS[element_type] = values
    elif element_type == TensorProto.FLOAT8E4M3FN:
        from onnx.reference.custom_element_types import float8e4m3fn

        zp_dtype = float8e4m3fn

    if zp_dtype is None:
        raise TypeError(f"Unexpected element_type {element_type}.")
    std_f8 = numpy.std(FLOAT8_DISTRIBUTIONS[element_type])
    zero = numpy.array(0, dtype=zp_dtype)
    scale = numpy.array(std / std_f8, dtype=std.dtype)
    return [zero, scale]


def compute_data_quant_params(
    data: numpy.ndarray,
    quant_type: onnx.TensorProto.DataType,
    symmetric: bool,
    reduce_range: bool = False,
    min_real_range: float | None = None,
    rmin_override: float | None = None,
    rmax_override: float | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Returns the zero_point and scale for the given data.

    :param data: The data for which to compute quantization parameters.
    :param quant_type: The quantization data type.
    :param symmetric: whether symmetric quantization is used or not.
    :parameter reduce_range: True if the quantization range should be reduced. Defaults to False.
    :parameter min_real_range: Minimum floating-point range (i.e., rmax - rmin) to enforce. Defaults to None.
    :parameter rmin_override: The value of rmin to use if not None. Otherwise, uses min(data).
    :parameter rmax_override: The value of rmax to use if not None. Otherwise, uses max(data).
    :return: zero point and scale
    """
    if not isinstance(data, numpy.ndarray):
        raise TypeError(f"Weight must be given as an array not {type(data)}.")
    if rmin_override is not None:
        rmin = rmin_override
    else:
        rmin = data.min() if len(data) else 0.0

    if rmax_override is not None:
        rmax = rmax_override
    else:
        rmax = data.max() if len(data) else 0.0

    rmin = numpy.array(rmin, dtype=data.dtype)
    rmax = numpy.array(rmax, dtype=data.dtype)
    scale = numpy.array(1.0, dtype=data.dtype)

    if quant_type == TensorProto.FLOAT8E4M3FN:
        if reduce_range:
            raise RuntimeError("Unsupported option reduce_range=True for float 8.")
        std = numpy.std(data)
        zero_point, scale = compute_scale_zp_float8(quant_type, std)
        return _check_type(zero_point, scale, zero_point_index=0)

    if quant_type in (
        TensorProto.INT8,
        TensorProto.UINT8,
        TensorProto.INT16,
        TensorProto.UINT16,
        TensorProto.INT4,
        TensorProto.UINT4,
    ):
        qmin, qmax = get_qmin_qmax_for_qType(quant_type, reduce_range, symmetric=symmetric)
        if len(data):
            zero_point, scale = compute_scale_zp(rmin, rmax, qmin, qmax, symmetric, min_real_range)
        else:
            zero_point = numpy.array(0, dtype=qmin.dtype)
        return _check_type(zero_point, scale, zero_point_index=0)

    raise ValueError(f"Unexpected value for quant_type={quant_type}.")


def quantize_data(
    data, qType, symmetric, reduce_range=False, min_real_range=None, rmin_override=None, rmax_override=None
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    :param data: data to quantize
    :param qType: data type to quantize to.
    :param symmetric: whether symmetric quantization is used or not.
    :parameter reduce_range: True if the quantization range should be reduced. Defaults to False.
    :parameter min_real_range: Minimum floating-point range (i.e., rmax - rmin) to enforce. Defaults to None.
    :parameter rmin_override: The value of rmin to use if not None. Otherwise, uses min(data).
    :parameter rmax_override: The value of rmax to use if not None. Otherwise, uses max(data).
    :return: minimum, maximum, zero point, scale, and quantized weights

    To pack weights, we compute a linear transformation

    - when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
    - when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
        `m = max(abs(rmin), abs(rmax))`

    and add necessary intermediate nodes to transform quantized weight to full weight using the equation

    :math:`r = S(q-z)`, where

    - *r*: real original value
    - *q*: quantized value
    - *S*: scale
    - *z*: zero point
    """
    zero_point, scale = compute_data_quant_params(
        data,
        qType,
        symmetric,
        reduce_range,
        min_real_range,
        rmin_override,
        rmax_override,
    )
    if qType == TensorProto.FLOAT8E4M3FN:
        quantized_data = quantize_nparray(qType, data, scale, zero_point)
        if any((quantized_data.astype(numpy.uint8).ravel() & 127) == 127):
            np_data = numpy.asarray(data)
            raise RuntimeError(
                f"One of the quantized value is NaN data in [{np_data.min()}, {np_data.max()}], "
                f"quantized_data in [{quantized_data.min()}, {quantized_data.max()}]."
            )
        return zero_point, scale, quantized_data

    if qType in (
        TensorProto.INT8,
        TensorProto.UINT8,
        TensorProto.INT16,
        TensorProto.UINT16,
        TensorProto.INT4,
        TensorProto.UINT4,
    ):
        quantized_data = quantize_nparray(qType, data, scale, zero_point)
        return zero_point, scale, quantized_data

    raise ValueError(f"Unexpected value for qType={qType}.")


def quantize_onnx_initializer(
    weight: onnx.TensorProto,
    quant_type: onnx.TensorProto.DataType,
    zero_point: numpy.ndarray,
    scale: numpy.ndarray,
    axis: int | None = None,
    quant_weight_name: str | None = None,
) -> onnx.TensorProto:
    """
    Returns a quantized version of the given ONNX initializer.

    :param weight: The ONNX initializer to quantize.
    :param quant_type: The final quantized data type.
    :param zero_point: The zero-point value to use for quantization.
    :param scale: The scale value to use for quantization.
    :param axis: The quantization axis if quantizing per-channel. Defaults to None.
    :param quant_weight_name: The name of the quantized initializer.
                              If not specified, the quantized name is generated.
    :return: The quantized ONNX initializer.
    """
    weight_data = tensor_proto_to_array(weight)
    q_weight_data: numpy.ndarray | None = None

    if axis is None:  # Per-tensor quantization
        q_weight_data = quantize_nparray(quant_type, weight_data.ravel(), scale, zero_point)
    else:  # Per-channel quantization
        channel_count = weight_data.shape[axis]
        channel_dims = list(weight_data.shape)  # deep copy
        channel_dims[axis] = 1  # only one per channel for reshape
        quantized_channel_data_list = []

        for i in range(channel_count):
            channel_data = weight_data.take(i, axis)
            channel_scale = scale[i]
            channel_zero_point = zero_point[i]
            quantized_channel_data = quantize_nparray(
                quant_type, channel_data.ravel(), channel_scale, channel_zero_point
            )
            quantized_channel_data_list.append(numpy.asarray(quantized_channel_data).reshape(channel_dims))

        q_weight_data = numpy.concatenate(quantized_channel_data_list, axis)

    q_weight_name = quant_weight_name if quant_weight_name else f"{weight.name}{TENSOR_NAME_QUANT_SUFFIX}"

    if quant_type == onnx.TensorProto.FLOAT8E4M3FN:
        q_weight_initializer = onnx.TensorProto()
        q_weight_initializer.data_type = quant_type
        q_weight_initializer.dims.extend(weight.dims)
        q_weight_initializer.name = q_weight_name
        # Do not remove .flatten().copy() numpy is not clear about data persistence.
        q_weight_initializer.raw_data = q_weight_data.flatten().copy().tobytes()
        if to_array_extended is not None:
            # This test should not be needed but it helped catch some issues
            # with data persistence and tobytes.
            check = to_array_extended(q_weight_initializer)
            if check.shape != weight_data.shape or check.tobytes() != q_weight_data.tobytes():
                raise RuntimeError(
                    f"The initializer of shape {weight_data.shape} could not be created, expecting "
                    f"{q_weight_data.tobytes()[:10]}, got {check.tobytes()[:10]} and shape={weight.shape}"
                    f"\nraw={str(q_weight_initializer)[:200]}."
                )
    elif quant_type in (onnx.TensorProto.INT4, onnx.TensorProto.UINT4):
        if q_weight_data.dtype not in (numpy.int8, numpy.uint8):
            raise RuntimeError(f"Quantized weights for {q_weight_name} must be 8-bit before packing as 4-bit values.")

        # We do not use onnx.helper.pack_float32_to_4bit() due to performance.
        # This can be the difference between a large model taking 30 minutes to quantize vs 5 minutes.
        packed_data = bytes(pack_bytes_to_4bit(q_weight_data.tobytes()))

        # We only use onnx.helper.make_tensor with raw data due to bug: https://github.com/onnx/onnx/pull/6161
        q_weight_initializer = onnx.helper.make_tensor(q_weight_name, quant_type, weight.dims, packed_data, raw=True)
    else:
        quant_np_dtype = onnx.helper.tensor_dtype_to_np_dtype(quant_type)
        q_weight_data = numpy.asarray(q_weight_data, dtype=quant_np_dtype).reshape(weight.dims)
        q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)

    return q_weight_initializer


def get_qmin_qmax_for_qType(qType, reduce_range=False, symmetric=False):  # noqa: N802
    """
    Return qmin and qmax, the minimum and maximum value representable by the given qType
    :parameter qType: onnx.onnx_pb.TensorProto.UINT8 or onnx.onnx_pb.TensorProto.UINT8
    :return: qmin, qmax
    """
    if qType == onnx_proto.TensorProto.FLOAT8E4M3FN:
        raise NotImplementedError("This function is not implemented for float 8 as not needed.")

    qrange = None

    if reduce_range:
        qrange = ONNX_INT_TYPE_REDUCED_RANGE.get(qType)
    elif symmetric and qType in ONNX_INT_TYPE_SYMMETRIC_RANGE:
        qrange = ONNX_INT_TYPE_SYMMETRIC_RANGE[qType]
    else:
        qrange = ONNX_INT_TYPE_RANGE.get(qType)

    if not qrange:
        raise ValueError(f"Unexpected data type {qType} requested. Only INT8, UINT8, INT16, and UINT16 are supported.")

    qmin, qmax = qrange
    if qmin > 0 or qmax < 0:
        raise ValueError(
            f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while "
            f"qmin:{qmin}, qmmax:{qmax}, dtype={qmin.dtype}, reduce_range={reduce_range}, "
            f"symmetric={symmetric}, qType={qType}"
        )

    return qrange


def get_qrange_for_qType(qType, reduce_range=False, symmetric=False):  # noqa: N802
    """
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    """
    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)
    return qmax - qmin


def normalize_axis(axis: int, rank: int) -> tuple[bool, int]:
    """
    Helper function that tries to return a normalized axis in the range [0, rank - 1].
    :parameter axis: The axis to normalize.
    :parameter rank: The tensor rank (number of dimensions).
    :return (is_valid, axis_norm)
    """
    axis_norm = axis + rank if axis < 0 else axis
    is_valid = axis_norm >= 0 and axis_norm < rank
    return is_valid, axis_norm


def pack_bytes_to_4bit(src_8bit: bytes) -> bytearray:
    """
    Copies a source array of 8-bit values into a destination bytearray of packed 4-bit values.
    Assumes that the source values are already in the appropriate int4 range.
    :parameter src_8bit: The 8-bit element values to pack.
    :return A bytearray with every two 8-bit src elements packed into a single byte.
    """
    num_elems = len(src_8bit)
    if num_elems == 0:
        return bytearray()

    dst_size = (num_elems + 1) // 2  # Ex: 5 8-bit elems packed into 3 bytes
    dst = bytearray(dst_size)

    src_i: int = 0
    dst_i: int = 0

    # Pack two 8-bit elements into a single byte in each iteration.
    while src_i < num_elems - 1:
        dst[dst_i] = ((src_8bit[src_i + 1] & 0xF) << 4) | (src_8bit[src_i] & 0xF)
        dst_i += 1
        src_i += 2

    if src_i < num_elems:
        # Odd number of elements.
        dst[dst_i] = src_8bit[src_i] & 0xF

    return dst


class QuantizedInitializer:
    """
    Represents a linearly quantized weight input from ONNX operators
    """

    def __init__(
        self,
        name,
        initializer,
        rmins,
        rmaxs,
        zero_points,
        scales,
        data=[],  # noqa: B006
        quantized_data=[],  # noqa: B006
        axis=None,
    ):
        self.name = name
        self.initializer = initializer  # TensorProto initializer in ONNX graph
        self.rmins = rmins  # List of minimum range for each axis
        self.rmaxs = rmaxs  # List of maximum range for each axis
        # 1D tensor of zero points computed for each axis. scalar if axis is empty
        self.zero_points = zero_points
        self.scales = scales  # 1D tensor of scales computed for each axis. scalar if axis is empty
        self.data = data  # original data from initializer TensorProto
        self.quantized_data = quantized_data  # weight-packed data from data
        # Scalar to specify which dimension in the initializer to weight pack.
        self.axis = axis
        # If empty, single zero point and scales computed from a single rmin and rmax


class QuantizedValue:
    """
    Represents a linearly quantized value (input\\output\\intializer)
    """

    def __init__(
        self,
        name,
        new_quantized_name,
        scale_name,
        zero_point_name,
        quantized_value_type,
        axis=None,
        node_type=None,
        node_qtype=None,
        scale_type=None,
    ):
        self.original_name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.value_type = quantized_value_type
        self.axis = axis
        self.node_type = node_type
        self.node_qtype = node_qtype
        self.scale_type = scale_type


class BiasToQuantize:
    """
    Represents a bias to be quantized
    """

    def __init__(self, bias_name, input_name, weight_name):
        self.bias_name = bias_name
        self.input_name = input_name
        self.weight_name = weight_name


def attribute_to_kwarg(attribute):
    """
    Convert attribute to kwarg format for use with onnx.helper.make_node.
        :parameter attribute: attribute in AttributeProto format.
        :return: attribute in {key: value} format.
    """
    if attribute.type == 0:
        raise ValueError(f"attribute {attribute.name} does not have type specified.")

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
    if attribute.type == 1:
        value = attribute.f
    elif attribute.type == 2:
        value = attribute.i
    elif attribute.type == 3:
        value = attribute.s
    elif attribute.type == 4:
        value = attribute.t
    elif attribute.type == 5:
        value = attribute.g
    elif attribute.type == 6:
        value = attribute.floats
    elif attribute.type == 7:
        value = attribute.ints
    elif attribute.type == 8:
        value = attribute.strings
    elif attribute.type == 9:
        value = attribute.tensors
    elif attribute.type == 10:
        value = attribute.graphs
    else:
        raise ValueError(f"attribute {attribute.name} has unsupported type {attribute.type}.")

    return {attribute.name: value}


def find_by_name(item_name, item_list):
    """
    Helper function to find item by name in a list.
        parameter item_name: name of the item.
        parameter item_list: list of items.
        return: item if found. None otherwise.
    """
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None


def get_elem_index(elem_name, elem_list):
    """
    Helper function to return index of an item in a node list
    """
    elem_idx = -1
    for i in range(len(elem_list)):
        if elem_list[i] == elem_name:
            elem_idx = i
    return elem_idx


def get_mul_node(inputs, output, name):
    """
    Helper function to create a Mul node.
        parameter inputs: list of input names.
        parameter output: output name.
        parameter name: name of the node.
        return: Mul node in NodeProto format.
    """
    return onnx.helper.make_node("Mul", inputs, [output], name)


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Helper function to generate a identifiable filepath by concatenating the given identifier as a suffix.
    """
    return filename.parent.joinpath(filename.stem + identifier + filename.suffix)


def apply_plot(hist, hist_edges):
    import sys

    import matplotlib.pyplot as plt
    import numpy

    numpy.set_printoptions(threshold=sys.maxsize)
    print("Histogram:")
    print(hist)
    print("Histogram Edges:")
    print(hist_edges)
    plt.stairs(hist, hist_edges, fill=True)
    plt.xlabel("Tensor value")
    plt.ylabel("Counts")
    plt.title("Tensor value V.S. Counts")
    plt.show()


def write_calibration_table(calibration_cache, dir="."):
    """
    Helper function to write calibration table to files.
    """

    import json

    import flatbuffers
    import numpy as np

    import onnxruntime.quantization.CalTableFlatBuffers.KeyValue as KeyValue
    import onnxruntime.quantization.CalTableFlatBuffers.TrtTable as TrtTable
    from onnxruntime.quantization.calibrate import CalibrationMethod, TensorData, TensorsData

    logging.info(f"calibration cache: {calibration_cache}")

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (TensorData, TensorsData)):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return {"data": obj.tolist(), "dtype": str(obj.dtype), "CLS": "numpy.array"}
            if isinstance(obj, CalibrationMethod):
                return {"CLS": obj.__class__.__name__, "value": str(obj)}
            return json.JSONEncoder.default(self, obj)

    json_data = json.dumps(calibration_cache, cls=MyEncoder)

    with open(os.path.join(dir, "calibration.json"), "w") as file:
        file.write(json_data)  # use `json.loads` to do the reverse

    # Serialize data using FlatBuffers
    zero = np.array(0)
    builder = flatbuffers.Builder(1024)
    key_value_list = []
    for key in sorted(calibration_cache.keys()):
        values = calibration_cache[key]
        d_values = values.to_dict()
        floats = [
            float(d_values.get("highest", zero).item()),
            float(d_values.get("lowest", zero).item()),
        ]
        value = str(max(floats))

        flat_key = builder.CreateString(key)
        flat_value = builder.CreateString(value)

        KeyValue.KeyValueStart(builder)
        KeyValue.KeyValueAddKey(builder, flat_key)
        KeyValue.KeyValueAddValue(builder, flat_value)
        key_value = KeyValue.KeyValueEnd(builder)

        key_value_list.append(key_value)

    TrtTable.TrtTableStartDictVector(builder, len(key_value_list))
    for key_value in key_value_list:
        builder.PrependUOffsetTRelative(key_value)
    main_dict = builder.EndVector()

    TrtTable.TrtTableStart(builder)
    TrtTable.TrtTableAddDict(builder, main_dict)
    cal_table = TrtTable.TrtTableEnd(builder)

    builder.Finish(cal_table)
    buf = builder.Output()

    with open(os.path.join(dir, "calibration.flatbuffers"), "wb") as file:
        file.write(buf)

    # Deserialize data (for validation)
    if os.environ.get("QUANTIZATION_DEBUG", 0) in (1, "1"):
        cal_table = TrtTable.TrtTable.GetRootAsTrtTable(buf, 0)
        dict_len = cal_table.DictLength()
        for i in range(dict_len):
            key_value = cal_table.Dict(i)
            logging.info(key_value.Key())
            logging.info(key_value.Value())

    # write plain text
    with open(os.path.join(dir, "calibration.cache"), "w") as file:
        for key in sorted(calibration_cache.keys()):
            values = calibration_cache[key]
            d_values = values.to_dict()
            floats = [
                float(d_values.get("highest", zero).item()),
                float(d_values.get("lowest", zero).item()),
            ]
            value = key + " " + str(max(floats))
            file.write(value)
            file.write("\n")


def smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor
    and taking the corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
         https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    """
    is_zeros = (p == 0).astype(numpy.float32)
    is_nonzeros = (p != 0).astype(numpy.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros

    if not n_nonzeros:
        # raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
        return None
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, f"n_zeros={n_zeros}, n_nonzeros={n_nonzeros}, eps1={eps1}"

    hist = p.astype(numpy.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0

    return hist


def model_has_external_data(model_path: Path):
    model = onnx.load(model_path.as_posix(), load_external_data=False)
    return any(external_data_helper.uses_external_data(intializer) for intializer in model.graph.initializer)


def optimize_model(model_path: Path, opt_model_path: Path):
    """
        Generate model that applies graph optimization (constant folding, etc.)
        parameter model_path: path to the original onnx model
        parameter opt_model_path: path to the optimized onnx model
    :return: optimized onnx model
    """
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    kwargs = {}
    # This will rename constant initializer names, disable it to make test pass.
    kwargs["disabled_optimizers"] = ["ConstantSharing"]
    _ = InferenceSession(model_path.as_posix(), sess_option, providers=["CPUExecutionProvider"], **kwargs)


def add_pre_process_metadata(model: ModelProto):
    """Tag the model that it went through quantization pre-processing"""
    metadata_props = {"onnx.quant.pre_process": "onnxruntime.quant"}
    if model.metadata_props:
        for prop in model.metadata_props:
            metadata_props.update({prop.key: prop.value})
    onnx.helper.set_model_props(model, metadata_props)


def model_has_pre_process_metadata(model: ModelProto) -> bool:
    """Check the model whether it went through quantization pre-processing"""
    if model.metadata_props:
        for prop in model.metadata_props:
            if prop.key == "onnx.quant.pre_process" and prop.value == "onnxruntime.quant":
                return True
    return False


def add_infer_metadata(model: ModelProto):
    metadata_props = {"onnx.infer": "onnxruntime.quant"}
    if model.metadata_props:
        for p in model.metadata_props:
            metadata_props.update({p.key: p.value})
    onnx.helper.set_model_props(model, metadata_props)


def model_has_infer_metadata(model: ModelProto) -> bool:
    if model.metadata_props:
        for p in model.metadata_props:
            if p.key == "onnx.infer" and p.value == "onnxruntime.quant":
                return True
    return False


def load_model_with_shape_infer(model_path: Path) -> ModelProto:
    inferred_model_path = generate_identified_filename(model_path, "-inferred")
    onnx.shape_inference.infer_shapes_path(str(model_path), str(inferred_model_path))
    model = onnx.load(inferred_model_path.as_posix())
    add_infer_metadata(model)
    inferred_model_path.unlink()
    return model


def save_and_reload_model_with_shape_infer(model: ModelProto) -> ModelProto:
    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        model_copy = copy.deepcopy(model)
        model_path = Path(quant_tmp_dir).joinpath("model.onnx")
        onnx.save_model(model_copy, model_path.as_posix(), save_as_external_data=True)
        return load_model_with_shape_infer(model_path)


def tensor_proto_to_array(initializer: TensorProto) -> numpy.ndarray:
    if initializer.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
        return onnx.numpy_helper.to_array(initializer)

    raise ValueError(
        f"Only float type is supported. Weights {initializer.name} is {type_to_name[initializer.data_type]}"
    )


def add_quant_suffix(tensor_name: str) -> str:
    return tensor_name + "_QuantizeLinear"


def add_quant_input_suffix(tensor_name: str) -> str:
    return tensor_name + QUANT_INPUT_SUFFIX


def add_quant_output_suffix(tensor_name) -> str:
    return tensor_name + "_QuantizeLinear_Output"


def add_dequant_suffix(tensor_name) -> str:
    return tensor_name + "_DequantizeLinear"


def add_dequant_input_suffix(tensor_name) -> str:
    return tensor_name + "_DequantizeLinear_Input"


def add_dequant_output_suffix(tensor_name) -> str:
    return tensor_name + DEQUANT_OUTPUT_SUFFIX

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\determinant.py ===
from types import FunctionType

from sympy.core.cache import cacheit
from sympy.core.numbers import Float, Integer
from sympy.core.singleton import S
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.mul import Mul
from sympy.polys import PurePoly, cancel
from sympy.functions.combinatorial.numbers import nC
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.ddm import DDM

from .exceptions import NonSquareMatrixError
from .utilities import (
    _get_intermediate_simp, _get_intermediate_simp_bool,
    _iszero, _is_zero_after_expand_mul, _dotprodsimp, _simplify)


def _find_reasonable_pivot(col, iszerofunc=_iszero, simpfunc=_simplify):
    """ Find the lowest index of an item in ``col`` that is
    suitable for a pivot.  If ``col`` consists only of
    Floats, the pivot with the largest norm is returned.
    Otherwise, the first element where ``iszerofunc`` returns
    False is used.  If ``iszerofunc`` does not return false,
    items are simplified and retested until a suitable
    pivot is found.

    Returns a 4-tuple
        (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    where pivot_offset is the index of the pivot, pivot_val is
    the (possibly simplified) value of the pivot, assumed_nonzero
    is True if an assumption that the pivot was non-zero
    was made without being proved, and newly_determined are
    elements that were simplified during the process of pivot
    finding."""

    newly_determined = []
    col = list(col)
    # a column that contains a mix of floats and integers
    # but at least one float is considered a numerical
    # column, and so we do partial pivoting
    if all(isinstance(x, (Float, Integer)) for x in col) and any(
            isinstance(x, Float) for x in col):
        col_abs = [abs(x) for x in col]
        max_value = max(col_abs)
        if iszerofunc(max_value):
            # just because iszerofunc returned True, doesn't
            # mean the value is numerically zero.  Make sure
            # to replace all entries with numerical zeros
            if max_value != 0:
                newly_determined = [(i, 0) for i, x in enumerate(col) if x != 0]
            return (None, None, False, newly_determined)
        index = col_abs.index(max_value)
        return (index, col[index], False, newly_determined)

    # PASS 1 (iszerofunc directly)
    possible_zeros = []
    for i, x in enumerate(col):
        is_zero = iszerofunc(x)
        # is someone wrote a custom iszerofunc, it may return
        # BooleanFalse or BooleanTrue instead of True or False,
        # so use == for comparison instead of `is`
        if is_zero == False:
            # we found something that is definitely not zero
            return (i, x, False, newly_determined)
        possible_zeros.append(is_zero)

    # by this point, we've found no certain non-zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 2 (iszerofunc after simplify)
    # we haven't found any for-sure non-zeros, so
    # go through the elements iszerofunc couldn't
    # make a determination about and opportunistically
    # simplify to see if we find something
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        simped = simpfunc(x)
        is_zero = iszerofunc(simped)
        if is_zero in (True, False):
            newly_determined.append((i, simped))
        if is_zero == False:
            return (i, simped, False, newly_determined)
        possible_zeros[i] = is_zero

    # after simplifying, some things that were recognized
    # as zeros might be zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 3 (.equals(0))
    # some expressions fail to simplify to zero, but
    # ``.equals(0)`` evaluates to True.  As a last-ditch
    # attempt, apply ``.equals`` to these expressions
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        if x.equals(S.Zero):
            # ``.iszero`` may return False with
            # an implicit assumption (e.g., ``x.equals(0)``
            # when ``x`` is a symbol), so only treat it
            # as proved when ``.equals(0)`` returns True
            possible_zeros[i] = True
            newly_determined.append((i, S.Zero))

    if all(possible_zeros):
        return (None, None, False, newly_determined)

    # at this point there is nothing that could definitely
    # be a pivot.  To maintain compatibility with existing
    # behavior, we'll assume that an illdetermined thing is
    # non-zero.  We should probably raise a warning in this case
    i = possible_zeros.index(None)
    return (i, col[i], True, newly_determined)


def _find_reasonable_pivot_naive(col, iszerofunc=_iszero, simpfunc=None):
    """
    Helper that computes the pivot value and location from a
    sequence of contiguous matrix column elements. As a side effect
    of the pivot search, this function may simplify some of the elements
    of the input column. A list of these simplified entries and their
    indices are also returned.
    This function mimics the behavior of _find_reasonable_pivot(),
    but does less work trying to determine if an indeterminate candidate
    pivot simplifies to zero. This more naive approach can be much faster,
    with the trade-off that it may erroneously return a pivot that is zero.

    ``col`` is a sequence of contiguous column entries to be searched for
    a suitable pivot.
    ``iszerofunc`` is a callable that returns a Boolean that indicates
    if its input is zero, or None if no such determination can be made.
    ``simpfunc`` is a callable that simplifies its input. It must return
    its input if it does not simplify its input. Passing in
    ``simpfunc=None`` indicates that the pivot search should not attempt
    to simplify any candidate pivots.

    Returns a 4-tuple:
    (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    ``pivot_offset`` is the sequence index of the pivot.
    ``pivot_val`` is the value of the pivot.
    pivot_val and col[pivot_index] are equivalent, but will be different
    when col[pivot_index] was simplified during the pivot search.
    ``assumed_nonzero`` is a boolean indicating if the pivot cannot be
    guaranteed to be zero. If assumed_nonzero is true, then the pivot
    may or may not be non-zero. If assumed_nonzero is false, then
    the pivot is non-zero.
    ``newly_determined`` is a list of index-value pairs of pivot candidates
    that were simplified during the pivot search.
    """

    # indeterminates holds the index-value pairs of each pivot candidate
    # that is neither zero or non-zero, as determined by iszerofunc().
    # If iszerofunc() indicates that a candidate pivot is guaranteed
    # non-zero, or that every candidate pivot is zero then the contents
    # of indeterminates are unused.
    # Otherwise, the only viable candidate pivots are symbolic.
    # In this case, indeterminates will have at least one entry,
    # and all but the first entry are ignored when simpfunc is None.
    indeterminates = []
    for i, col_val in enumerate(col):
        col_val_is_zero = iszerofunc(col_val)
        if col_val_is_zero == False:
            # This pivot candidate is non-zero.
            return i, col_val, False, []
        elif col_val_is_zero is None:
            # The candidate pivot's comparison with zero
            # is indeterminate.
            indeterminates.append((i, col_val))

    if len(indeterminates) == 0:
        # All candidate pivots are guaranteed to be zero, i.e. there is
        # no pivot.
        return None, None, False, []

    if simpfunc is None:
        # Caller did not pass in a simplification function that might
        # determine if an indeterminate pivot candidate is guaranteed
        # to be nonzero, so assume the first indeterminate candidate
        # is non-zero.
        return indeterminates[0][0], indeterminates[0][1], True, []

    # newly_determined holds index-value pairs of candidate pivots
    # that were simplified during the search for a non-zero pivot.
    newly_determined = []
    for i, col_val in indeterminates:
        tmp_col_val = simpfunc(col_val)
        if id(col_val) != id(tmp_col_val):
            # simpfunc() simplified this candidate pivot.
            newly_determined.append((i, tmp_col_val))
            if iszerofunc(tmp_col_val) == False:
                # Candidate pivot simplified to a guaranteed non-zero value.
                return i, tmp_col_val, False, newly_determined

    return indeterminates[0][0], indeterminates[0][1], True, newly_determined


# This functions is a candidate for caching if it gets implemented for matrices.
def _berkowitz_toeplitz_matrix(M):
    """Return (A,T) where T the Toeplitz matrix used in the Berkowitz algorithm
    corresponding to ``M`` and A is the first principal submatrix.
    """

    # the 0 x 0 case is trivial
    if M.rows == 0 and M.cols == 0:
        return M._new(1,1, [M.one])

    #
    # Partition M = [ a_11  R ]
    #                  [ C     A ]
    #

    a, R = M[0,0],   M[0, 1:]
    C, A = M[1:, 0], M[1:,1:]

    #
    # The Toeplitz matrix looks like
    #
    #  [ 1                                     ]
    #  [ -a         1                          ]
    #  [ -RC       -a        1                 ]
    #  [ -RAC     -RC       -a       1         ]
    #  [ -RA**2C -RAC      -RC      -a       1 ]
    #  etc.

    # Compute the diagonal entries.
    # Because multiplying matrix times vector is so much
    # more efficient than matrix times matrix, recursively
    # compute -R * A**n * C.
    diags = [C]
    for i in range(M.rows - 2):
        diags.append(A.multiply(diags[i], dotprodsimp=None))
    diags = [(-R).multiply(d, dotprodsimp=None)[0, 0] for d in diags]
    diags = [M.one, -a] + diags

    def entry(i,j):
        if j > i:
            return M.zero
        return diags[i - j]

    toeplitz = M._new(M.cols + 1, M.rows, entry)
    return (A, toeplitz)


# This functions is a candidate for caching if it gets implemented for matrices.
def _berkowitz_vector(M):
    """ Run the Berkowitz algorithm and return a vector whose entries
        are the coefficients of the characteristic polynomial of ``M``.

        Given N x N matrix, efficiently compute
        coefficients of characteristic polynomials of ``M``
        without division in the ground domain.

        This method is particularly useful for computing determinant,
        principal minors and characteristic polynomial when ``M``
        has complicated coefficients e.g. polynomials. Semi-direct
        usage of this algorithm is also important in computing
        efficiently sub-resultant PRS.

        Assuming that M is a square matrix of dimension N x N and
        I is N x N identity matrix, then the Berkowitz vector is
        an N x 1 vector whose entries are coefficients of the
        polynomial

                        charpoly(M) = det(t*I - M)

        As a consequence, all polynomials generated by Berkowitz
        algorithm are monic.

        For more information on the implemented algorithm refer to:

        [1] S.J. Berkowitz, On computing the determinant in small
            parallel time using a small number of processors, ACM,
            Information Processing Letters 18, 1984, pp. 147-150

        [2] M. Keber, Division-Free computation of sub-resultants
            using Bezout matrices, Tech. Report MPI-I-2006-1-006,
            Saarbrucken, 2006
    """

    # handle the trivial cases
    if M.rows == 0 and M.cols == 0:
        return M._new(1, 1, [M.one])
    elif M.rows == 1 and M.cols == 1:
        return M._new(2, 1, [M.one, -M[0,0]])

    submat, toeplitz = _berkowitz_toeplitz_matrix(M)

    return toeplitz.multiply(_berkowitz_vector(submat), dotprodsimp=None)


def _adjugate(M, method="berkowitz"):
    """Returns the adjugate, or classical adjoint, of
    a matrix.  That is, the transpose of the matrix of cofactors.

    https://en.wikipedia.org/wiki/Adjugate

    Parameters
    ==========

    method : string, optional
        Method to use to find the cofactors, can be "bareiss", "berkowitz",
        "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.adjugate()
    Matrix([
    [ 4, -2],
    [-3,  1]])

    See Also
    ========

    cofactor_matrix
    sympy.matrices.matrixbase.MatrixBase.transpose
    """

    return M.cofactor_matrix(method=method).transpose()


# This functions is a candidate for caching if it gets implemented for matrices.
def _charpoly(M, x='lambda', simplify=_simplify):
    """Computes characteristic polynomial det(x*I - M) where I is
    the identity matrix.

    A PurePoly is returned, so using different variables for ``x`` does
    not affect the comparison or the polynomials:

    Parameters
    ==========

    x : string, optional
        Name for the "lambda" variable, defaults to "lambda".

    simplify : function, optional
        Simplification function to use on the characteristic polynomial
        calculated. Defaults to ``simplify``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x, y
    >>> M = Matrix([[1, 3], [2, 0]])
    >>> M.charpoly()
    PurePoly(lambda**2 - lambda - 6, lambda, domain='ZZ')
    >>> M.charpoly(x) == M.charpoly(y)
    True
    >>> M.charpoly(x) == M.charpoly(y)
    True

    Specifying ``x`` is optional; a symbol named ``lambda`` is used by
    default (which looks good when pretty-printed in unicode):

    >>> M.charpoly().as_expr()
    lambda**2 - lambda - 6

    And if ``x`` clashes with an existing symbol, underscores will
    be prepended to the name to make it unique:

    >>> M = Matrix([[1, 2], [x, 0]])
    >>> M.charpoly(x).as_expr()
    _x**2 - _x - 2*x

    Whether you pass a symbol or not, the generator can be obtained
    with the gen attribute since it may not be the same as the symbol
    that was passed:

    >>> M.charpoly(x).gen
    _x
    >>> M.charpoly(x).gen == x
    False

    Notes
    =====

    The Samuelson-Berkowitz algorithm is used to compute
    the characteristic polynomial efficiently and without any
    division operations.  Thus the characteristic polynomial over any
    commutative ring without zero divisors can be computed.

    If the determinant det(x*I - M) can be found out easily as
    in the case of an upper or a lower triangular matrix, then
    instead of Samuelson-Berkowitz algorithm, eigenvalues are computed
    and the characteristic polynomial with their help.

    See Also
    ========

    det
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    # Use DomainMatrix. We are already going to convert this to a Poly so there
    # is no need to worry about expanding powers etc. Also since this algorithm
    # does not require division or zero detection it is fine to use EX.
    #
    # M.to_DM() will fall back on EXRAW rather than EX. EXRAW is a lot faster
    # for elementary arithmetic because it does not call cancel for each
    # operation but it generates large unsimplified results that are slow in
    # the subsequent call to simplify. Using EX instead is faster overall
    # but at least in some cases EXRAW+simplify gives a simpler result so we
    # preserve that existing behaviour of charpoly for now...
    dM = M.to_DM()

    K = dM.domain

    cp = dM.charpoly()

    x = uniquely_named_symbol(x, [M], modify=lambda s: '_' + s)

    if K.is_EXRAW or simplify is not _simplify:
        # XXX: Converting back to Expr is expensive. We only do it if the
        # caller supplied a custom simplify function for backwards
        # compatibility or otherwise if the domain was EX. For any other domain
        # there should be no benefit in simplifying at this stage because Poly
        # will put everything into canonical form anyway.
        berk_vector = [K.to_sympy(c) for c in cp]
        berk_vector = [simplify(a) for a in berk_vector]
        p = PurePoly(berk_vector, x)

    else:
        # Convert from the list of domain elements directly to Poly.
        p = PurePoly(cp, x, domain=K)

    return p


def _cofactor(M, i, j, method="berkowitz"):
    """Calculate the cofactor of an element.

    Parameters
    ==========

    method : string, optional
        Method to use to find the cofactors, can be "bareiss", "berkowitz",
        "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.cofactor(0, 1)
    -3

    See Also
    ========

    cofactor_matrix
    minor
    minor_submatrix
    """

    if not M.is_square or M.rows < 1:
        raise NonSquareMatrixError()

    return S.NegativeOne**((i + j) % 2) * M.minor(i, j, method)


def _cofactor_matrix(M, method="berkowitz"):
    """Return a matrix containing the cofactor of each element.

    Parameters
    ==========

    method : string, optional
        Method to use to find the cofactors, can be "bareiss", "berkowitz",
        "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.cofactor_matrix()
    Matrix([
    [ 4, -3],
    [-2,  1]])

    See Also
    ========

    cofactor
    minor
    minor_submatrix
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    return M._new(M.rows, M.cols,
            lambda i, j: M.cofactor(i, j, method))

def _per(M):
    """Returns the permanent of a matrix. Unlike determinant,
    permanent is defined for both square and non-square matrices.

    For an m x n matrix, with m less than or equal to n,
    it is given as the sum over the permutations s of size
    less than or equal to m on [1, 2, . . . n] of the product
    from i = 1 to m of M[i, s[i]]. Taking the transpose will
    not affect the value of the permanent.

    In the case of a square matrix, this is the same as the permutation
    definition of the determinant, but it does not take the sign of the
    permutation into account. Computing the permanent with this definition
    is quite inefficient, so here the Ryser formula is used.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.per()
    450
    >>> M = Matrix([1, 5, 7])
    >>> M.per()
    13

    References
    ==========

    .. [1] Prof. Frank Ben's notes: https://math.berkeley.edu/~bernd/ban275.pdf
    .. [2] Wikipedia article on Permanent: https://en.wikipedia.org/wiki/Permanent_%28mathematics%29
    .. [3] https://reference.wolfram.com/language/ref/Permanent.html
    .. [4] Permanent of a rectangular matrix : https://arxiv.org/pdf/0904.3251.pdf
    """
    import itertools

    m, n = M.shape
    if m > n:
        M = M.T
        m, n = n, m
    s = list(range(n))

    subsets = []
    for i in range(1, m + 1):
        subsets += list(map(list, itertools.combinations(s, i)))

    perm = 0
    for subset in subsets:
        prod = 1
        sub_len = len(subset)
        for i in range(m):
            prod *= sum(M[i, j] for j in subset)
        perm += prod * S.NegativeOne**sub_len * nC(n - sub_len, m - sub_len)
    perm *= S.NegativeOne**m
    return perm.simplify()

def _det_DOM(M):
    DOM = DomainMatrix.from_Matrix(M, field=True, extension=True)
    K = DOM.domain
    return K.to_sympy(DOM.det())

# This functions is a candidate for caching if it gets implemented for matrices.
def _det(M, method="bareiss", iszerofunc=None):
    """Computes the determinant of a matrix if ``M`` is a concrete matrix object
    otherwise return an expressions ``Determinant(M)`` if ``M`` is a
    ``MatrixSymbol`` or other expression.

    Parameters
    ==========

    method : string, optional
        Specifies the algorithm used for computing the matrix determinant.

        If the matrix is at most 3x3, a hard-coded formula is used and the
        specified method is ignored. Otherwise, it defaults to
        ``'bareiss'``.

        Also, if the matrix is an upper or a lower triangular matrix, determinant
        is computed by simple multiplication of diagonal elements, and the
        specified method is ignored.

        If it is set to ``'domain-ge'``, then Gaussian elimination method will
        be used via using DomainMatrix.

        If it is set to ``'bareiss'``, Bareiss' fraction-free algorithm will
        be used.

        If it is set to ``'berkowitz'``, Berkowitz' algorithm will be used.

        If it is set to ``'bird'``, Bird's algorithm will be used [1]_.

        If it is set to ``'laplace'``, Laplace's algorithm will be used.

        Otherwise, if it is set to ``'lu'``, LU decomposition will be used.

        .. note::
            For backward compatibility, legacy keys like "bareis" and
            "det_lu" can still be used to indicate the corresponding
            methods.
            And the keys are also case-insensitive for now. However, it is
            suggested to use the precise keys for specifying the method.

    iszerofunc : FunctionType or None, optional
        If it is set to ``None``, it will be defaulted to ``_iszero`` if the
        method is set to ``'bareiss'``, and ``_is_zero_after_expand_mul`` if
        the method is set to ``'lu'``.

        It can also accept any user-specified zero testing function, if it
        is formatted as a function which accepts a single symbolic argument
        and returns ``True`` if it is tested as zero and ``False`` if it
        tested as non-zero, and also ``None`` if it is undecidable.

    Returns
    =======

    det : Basic
        Result of determinant.

    Raises
    ======

    ValueError
        If unrecognized keys are given for ``method`` or ``iszerofunc``.

    NonSquareMatrixError
        If attempted to calculate determinant from a non-square matrix.

    Examples
    ========

    >>> from sympy import Matrix, eye, det
    >>> I3 = eye(3)
    >>> det(I3)
    1
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> det(M)
    -2
    >>> det(M) == M.det()
    True
    >>> M.det(method="domain-ge")
    -2

    References
    ==========

    .. [1] Bird, R. S. (2011). A simple division-free algorithm for computing
           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:
           10.1016/j.ipl.2011.08.006
    """

    # sanitize `method`
    method = method.lower()

    if method == "bareis":
        method = "bareiss"
    elif method == "det_lu":
        method = "lu"

    if method not in ("bareiss", "berkowitz", "lu", "domain-ge", "bird",
                      "laplace"):
        raise ValueError("Determinant method '%s' unrecognized" % method)

    if iszerofunc is None:
        if method == "bareiss":
            iszerofunc = _is_zero_after_expand_mul
        elif method == "lu":
            iszerofunc = _iszero

    elif not isinstance(iszerofunc, FunctionType):
        raise ValueError("Zero testing method '%s' unrecognized" % iszerofunc)

    n = M.rows

    if n == M.cols: # square check is done in individual method functions
        if n == 0:
            return M.one
        elif n == 1:
            return M[0, 0]
        elif n == 2:
            m = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            return _get_intermediate_simp(_dotprodsimp)(m)
        elif n == 3:
            m =  (M[0, 0] * M[1, 1] * M[2, 2]
                + M[0, 1] * M[1, 2] * M[2, 0]
                + M[0, 2] * M[1, 0] * M[2, 1]
                - M[0, 2] * M[1, 1] * M[2, 0]
                - M[0, 0] * M[1, 2] * M[2, 1]
                - M[0, 1] * M[1, 0] * M[2, 2])
            return _get_intermediate_simp(_dotprodsimp)(m)

    dets = []
    for b in M.strongly_connected_components():
        if method == "domain-ge": # uses DomainMatrix to evaluate determinant
            det = _det_DOM(M[b, b])
        elif method == "bareiss":
            det = M[b, b]._eval_det_bareiss(iszerofunc=iszerofunc)
        elif method == "berkowitz":
            det = M[b, b]._eval_det_berkowitz()
        elif method == "lu":
            det = M[b, b]._eval_det_lu(iszerofunc=iszerofunc)
        elif method == "bird":
            det = M[b, b]._eval_det_bird()
        elif method == "laplace":
            det = M[b, b]._eval_det_laplace()
        dets.append(det)
    return Mul(*dets)


# This functions is a candidate for caching if it gets implemented for matrices.
def _det_bareiss(M, iszerofunc=_is_zero_after_expand_mul):
    """Compute matrix determinant using Bareiss' fraction-free
    algorithm which is an extension of the well known Gaussian
    elimination method. This approach is best suited for dense
    symbolic matrices and will result in a determinant with
    minimal number of fractions. It means that less term
    rewriting is needed on resulting formulae.

    Parameters
    ==========

    iszerofunc : function, optional
        The function to use to determine zeros when doing an LU decomposition.
        Defaults to ``lambda x: x.is_zero``.

    TODO: Implement algorithm for sparse matrices (SFF),
    http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.
    """

    # Recursively implemented Bareiss' algorithm as per Deanna Richelle Leggett's
    # thesis http://www.math.usm.edu/perry/Research/Thesis_DRL.pdf
    def bareiss(mat, cumm=1):
        if mat.rows == 0:
            return mat.one
        elif mat.rows == 1:
            return mat[0, 0]

        # find a pivot and extract the remaining matrix
        # With the default iszerofunc, _find_reasonable_pivot slows down
        # the computation by the factor of 2.5 in one test.
        # Relevant issues: #10279 and #13877.
        pivot_pos, pivot_val, _, _ = _find_reasonable_pivot(mat[:, 0], iszerofunc=iszerofunc)
        if pivot_pos is None:
            return mat.zero

        # if we have a valid pivot, we'll do a "row swap", so keep the
        # sign of the det
        sign = (-1) ** (pivot_pos % 2)

        # we want every row but the pivot row and every column
        rows = [i for i in range(mat.rows) if i != pivot_pos]
        cols = list(range(mat.cols))
        tmp_mat = mat.extract(rows, cols)

        def entry(i, j):
            ret = (pivot_val*tmp_mat[i, j + 1] - mat[pivot_pos, j + 1]*tmp_mat[i, 0]) / cumm
            if _get_intermediate_simp_bool(True):
                return _dotprodsimp(ret)
            elif not ret.is_Atom:
                return cancel(ret)
            return ret

        return sign*bareiss(M._new(mat.rows - 1, mat.cols - 1, entry), pivot_val)

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    return bareiss(M)


def _det_berkowitz(M):
    """ Use the Berkowitz algorithm to compute the determinant."""

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    berk_vector = _berkowitz_vector(M)
    return (-1)**(len(berk_vector) - 1) * berk_vector[-1]


# This functions is a candidate for caching if it gets implemented for matrices.
def _det_LU(M, iszerofunc=_iszero, simpfunc=None):
    """ Computes the determinant of a matrix from its LU decomposition.
    This function uses the LU decomposition computed by
    LUDecomposition_Simple().

    The keyword arguments iszerofunc and simpfunc are passed to
    LUDecomposition_Simple().
    iszerofunc is a callable that returns a boolean indicating if its
    input is zero, or None if it cannot make the determination.
    simpfunc is a callable that simplifies its input.
    The default is simpfunc=None, which indicate that the pivot search
    algorithm should not attempt to simplify any candidate pivots.
    If simpfunc fails to simplify its input, then it must return its input
    instead of a copy.

    Parameters
    ==========

    iszerofunc : function, optional
        The function to use to determine zeros when doing an LU decomposition.
        Defaults to ``lambda x: x.is_zero``.

    simpfunc : function, optional
        The simplification function to use when looking for zeros for pivots.
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    lu, row_swaps = M.LUdecomposition_Simple(iszerofunc=iszerofunc,
            simpfunc=simpfunc)
    # P*A = L*U => det(A) = det(L)*det(U)/det(P) = det(P)*det(U).
    # Lower triangular factor L encoded in lu has unit diagonal => det(L) = 1.
    # P is a permutation matrix => det(P) in {-1, 1} => 1/det(P) = det(P).
    # LUdecomposition_Simple() returns a list of row exchange index pairs, rather
    # than a permutation matrix, but det(P) = (-1)**len(row_swaps).

    # Avoid forming the potentially time consuming  product of U's diagonal entries
    # if the product is zero.
    # Bottom right entry of U is 0 => det(A) = 0.
    # It may be impossible to determine if this entry of U is zero when it is symbolic.
    if iszerofunc(lu[lu.rows-1, lu.rows-1]):
        return M.zero

    # Compute det(P)
    det = -M.one if len(row_swaps)%2 else M.one

    # Compute det(U) by calculating the product of U's diagonal entries.
    # The upper triangular portion of lu is the upper triangular portion of the
    # U factor in the LU decomposition.
    for k in range(lu.rows):
        det *= lu[k, k]

    # return det(P)*det(U)
    return det


@cacheit
def __det_laplace(M):
    """Compute the determinant of a matrix using Laplace expansion.

    This is a recursive function, and it should not be called directly.
    Use _det_laplace() instead. The reason for splitting this function
    into two is to allow caching of determinants of submatrices. While
    one could also define this function inside _det_laplace(), that
    would remove the advantage of using caching in Cramer Solve.
    """
    n = M.shape[0]
    if n == 1:
        return M[0]
    elif n == 2:
        return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    else:
        return sum((-1) ** i * M[0, i] *
                   __det_laplace(M.minor_submatrix(0, i)) for i in range(n))


def _det_laplace(M):
    """Compute the determinant of a matrix using Laplace expansion.

    While Laplace expansion is not the most efficient method of computing
    a determinant, it is a simple one, and it has the advantage of
    being division free. To improve efficiency, this function uses
    caching to avoid recomputing determinants of submatrices.
    """
    if not M.is_square:
        raise NonSquareMatrixError()
    if M.shape[0] == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.
    return __det_laplace(M.as_immutable())


def _det_bird(M):
    r"""Compute the determinant of a matrix using Bird's algorithm.

    Bird's algorithm is a simple division-free algorithm for computing, which
    is of lower order than the Laplace's algorithm. It is described in [1]_.

    References
    ==========

    .. [1] Bird, R. S. (2011). A simple division-free algorithm for computing
           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:
           10.1016/j.ipl.2011.08.006
    """
    def mu(X):
        n = X.shape[0]
        zero = X.domain.zero

        total = zero
        diag_sums = [zero]
        for i in reversed(range(1, n)):
            total -= X[i][i]
            diag_sums.append(total)
        diag_sums = diag_sums[::-1]

        elems = [[zero] * i + [diag_sums[i]] + X_i[i + 1:] for i, X_i in
                 enumerate(X)]
        return DDM(elems, X.shape, X.domain)

    Mddm = M._rep.to_ddm()
    n = M.shape[0]
    if n == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.
    Fn1 = Mddm
    for _ in range(n - 1):
        Fn1 = mu(Fn1).matmul(Mddm)
    detA = Fn1[0][0]
    if n % 2 == 0:
        detA = -detA

    return Mddm.domain.to_sympy(detA)


def _minor(M, i, j, method="berkowitz"):
    """Return the (i,j) minor of ``M``.  That is,
    return the determinant of the matrix obtained by deleting
    the `i`th row and `j`th column from ``M``.

    Parameters
    ==========

    i, j : int
        The row and column to exclude to obtain the submatrix.

    method : string, optional
        Method to use to find the determinant of the submatrix, can be
        "bareiss", "berkowitz", "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.minor(1, 1)
    -12

    See Also
    ========

    minor_submatrix
    cofactor
    det
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    return M.minor_submatrix(i, j).det(method=method)


def _minor_submatrix(M, i, j):
    """Return the submatrix obtained by removing the `i`th row
    and `j`th column from ``M`` (works with Pythonic negative indices).

    Parameters
    ==========

    i, j : int
        The row and column to exclude to obtain the submatrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.minor_submatrix(1, 1)
    Matrix([
    [1, 3],
    [7, 9]])

    See Also
    ========

    minor
    cofactor
    """

    if i < 0:
        i += M.rows
    if j < 0:
        j += M.cols

    if not 0 <= i < M.rows or not 0 <= j < M.cols:
        raise ValueError("`i` and `j` must satisfy 0 <= i < ``M.rows`` "
                            "(%d)" % M.rows + "and 0 <= j < ``M.cols`` (%d)." % M.cols)

    rows = [a for a in range(M.rows) if a != i]
    cols = [a for a in range(M.cols) if a != j]

    return M.extract(rows, cols)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\str.py ===
"""
A Printer for generating readable representation of most SymPy classes.
"""

from __future__ import annotations
from typing import Any

from sympy.core import S, Rational, Pow, Basic, Mul, Number
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import Integer
from sympy.core.relational import Relational
from sympy.core.sorting import default_sort_key
from sympy.utilities.iterables import sift
from .precedence import precedence, PRECEDENCE
from .printer import Printer, print_function

from mpmath.libmp import prec_to_dps, to_str as mlib_to_str


class StrPrinter(Printer):
    printmethod = "_sympystr"
    _default_settings: dict[str, Any] = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
        "abbrev": False,
        "perm_cyclic": True,
        "min": None,
        "max": None,
        "dps" : None
    }

    _relationals: dict[str, str] = {}

    def parenthesize(self, item, level, strict=False):
        if (precedence(item) < level) or ((not strict) and precedence(item) <= level):
            return "(%s)" % self._print(item)
        else:
            return self._print(item)

    def stringify(self, args, sep, level=0):
        return sep.join([self.parenthesize(item, level) for item in args])

    def emptyPrinter(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, Basic):
            return repr(expr)
        else:
            return str(expr)

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)

        prec = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if t.startswith('-') and not term.is_Add:
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < prec or term.is_Add:
                l.extend([sign, "(%s)" % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    def _print_BooleanTrue(self, expr):
        return "True"

    def _print_BooleanFalse(self, expr):
        return "False"

    def _print_Not(self, expr):
        return '~%s' %(self.parenthesize(expr.args[0],PRECEDENCE["Not"]))

    def _print_And(self, expr):
        args = list(expr.args)
        for j, i in enumerate(args):
            if isinstance(i, Relational) and (
                    i.canonical.rhs is S.NegativeInfinity):
                args.insert(0, args.pop(j))
        return self.stringify(args, " & ", PRECEDENCE["BitwiseAnd"])

    def _print_Or(self, expr):
        return self.stringify(expr.args, " | ", PRECEDENCE["BitwiseOr"])

    def _print_Xor(self, expr):
        return self.stringify(expr.args, " ^ ", PRECEDENCE["BitwiseXor"])

    def _print_AppliedPredicate(self, expr):
        return '%s(%s)' % (
            self._print(expr.function), self.stringify(expr.arguments, ", "))

    def _print_Basic(self, expr):
        l = [self._print(o) for o in expr.args]
        return expr.__class__.__name__ + "(%s)" % ", ".join(l)

    def _print_BlockMatrix(self, B):
        if B.blocks.shape == (1, 1):
            self._print(B.blocks[0, 0])
        return self._print(B.blocks)

    def _print_Catalan(self, expr):
        return 'Catalan'

    def _print_ComplexInfinity(self, expr):
        return 'zoo'

    def _print_ConditionSet(self, s):
        args = tuple([self._print(i) for i in (s.sym, s.condition)])
        if s.base_set is S.UniversalSet:
            return 'ConditionSet(%s, %s)' % args
        args += (self._print(s.base_set),)
        return 'ConditionSet(%s, %s, %s)' % args

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return 'Derivative(%s)' % ", ".join((self._print(arg) for arg in [dexpr] + dvars))

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            item = "%s: %s" % (self._print(key), self._print(d[key]))
            items.append(item)

        return "{%s}" % ", ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('Domain: ' + self._print(d.symbols) + ' in ' +
                    self._print(d.set))
        else:
            return 'Domain on ' + self._print(d.symbols)

    def _print_Dummy(self, expr):
        return '_' + expr.name

    def _print_EulerGamma(self, expr):
        return 'EulerGamma'

    def _print_Exp1(self, expr):
        return 'E'

    def _print_ExprCondPair(self, expr):
        return '(%s, %s)' % (self._print(expr.expr), self._print(expr.cond))

    def _print_Function(self, expr):
        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_Heaviside(self, expr):
        # Same as _print_Function but uses pargs to suppress default 1/2 for
        # 2nd args
        return expr.func.__name__ + "(%s)" % self.stringify(expr.pargs, ", ")

    def _print_TribonacciConstant(self, expr):
        return 'TribonacciConstant'

    def _print_ImaginaryUnit(self, expr):
        return 'I'

    def _print_Infinity(self, expr):
        return 'oo'

    def _print_Integral(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Integral(%s, %s)' % (self._print(expr.function), L)

    def _print_Interval(self, i):
        fin =  'Interval{m}({a}, {b})'
        a, b, l, r = i.args
        if a.is_infinite and b.is_infinite:
            m = ''
        elif a.is_infinite and not r:
            m = ''
        elif b.is_infinite and not l:
            m = ''
        elif not l and not r:
            m = ''
        elif l and r:
            m = '.open'
        elif l:
            m = '.Lopen'
        else:
            m = '.Ropen'
        return fin.format(**{'a': a, 'b': b, 'm': m})

    def _print_AccumulationBounds(self, i):
        return "AccumBounds(%s, %s)" % (self._print(i.min),
                                        self._print(i.max))

    def _print_Inverse(self, I):
        return "%s**(-1)" % self.parenthesize(I.arg, PRECEDENCE["Pow"])

    def _print_Lambda(self, obj):
        expr = obj.expr
        sig = obj.signature
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        return "Lambda(%s, %s)" % (self._print(sig), self._print(expr))

    def _print_LatticeOp(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return expr.func.__name__ + "(%s)" % ", ".join(self._print(arg) for arg in args)

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args
        return "Limit(%s, %s, %s, dir='%s')" % tuple(map(self._print, (e, z, z0, dir)))


    def _print_list(self, expr):
        return "[%s]" % self.stringify(expr, ", ")

    def _print_List(self, expr):
        return self._print_list(expr)

    def _print_MatrixBase(self, expr):
        return expr._format_str(self)

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))

    def _print_MatrixSlice(self, expr):
        def strslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return ':'.join((self._print(arg) for arg in x))
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + '[' +
                strslice(expr.rowslice, expr.parent.rows) + ', ' +
                strslice(expr.colslice, expr.parent.cols) + ']')

    def _print_DeferredVector(self, expr):
        return expr.name

    def _print_Mul(self, expr):

        prec = precedence(expr)

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        args = expr.args
        if args[0] is S.One or any(
                isinstance(a, Number) or
                a.is_Pow and all(ai.is_Integer for ai in a.args)
                for a in args[1:]):
            d, n = sift(args, lambda x:
                isinstance(x, Pow) and bool(x.exp.as_coeff_Mul()[0] < 0),
                binary=True)
            for i, di in enumerate(d):
                if di.exp.is_Number:
                    e = -di.exp
                else:
                    dargs = list(di.exp.args)
                    dargs[0] = -dargs[0]
                    e = Mul._from_args(dargs)
                d[i] = Pow(di.base, e, evaluate=False) if e - 1 else di.base

            pre = []
            # don't parenthesize first factor if negative
            if n and not n[0].is_Add and n[0].could_extract_minus_sign():
                pre = [self._print(n.pop(0))]

            nfactors = pre + [self.parenthesize(a, prec, strict=False)
                for a in n]
            if not nfactors:
                nfactors = ['1']

            # don't parenthesize first of denominator unless singleton
            if len(d) > 1 and d[0].could_extract_minus_sign():
                pre = [self._print(d.pop(0))]
            else:
                pre = []
            dfactors = pre + [self.parenthesize(a, prec, strict=False)
                for a in d]

            n = '*'.join(nfactors)
            d = '*'.join(dfactors)
            if len(dfactors) > 1:
                return '%s/(%s)' % (n, d)
            elif dfactors:
                return '%s/%s' % (n, d)
            return n

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
        def apow(i):
            b, e = i.as_base_exp()
            eargs = list(Mul.make_args(e))
            if eargs[0] is S.NegativeOne:
                eargs = eargs[1:]
            else:
                eargs[0] = -eargs[0]
            e = Mul._from_args(eargs)
            if isinstance(i, Pow):
                return i.func(b, e, evaluate=False)
            return i.func(e, evaluate=False)
        for item in args:
            if (item.is_commutative and
                    isinstance(item, Pow) and
                    bool(item.exp.as_coeff_Mul()[0] < 0)):
                if item.exp is not S.NegativeOne:
                    b.append(apow(item))
                else:
                    if (len(item.args[0].args) != 1 and
                            isinstance(item.base, (Mul, Pow))):
                        # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(item.base)
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec, strict=False) for x in a]
        b_str = [self.parenthesize(x, prec, strict=False) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)

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

        return sign + '*'.join(
            [self.parenthesize(arg, precedence(expr)) for arg in expr.args]
        )

    def _print_ElementwiseApplyFunction(self, expr):
        return "{}.({})".format(
            expr.function,
            self._print(expr.expr),
        )

    def _print_NaN(self, expr):
        return 'nan'

    def _print_NegativeInfinity(self, expr):
        return '-oo'

    def _print_Order(self, expr):
        if not expr.variables or all(p is S.Zero for p in expr.point):
            if len(expr.variables) <= 1:
                return 'O(%s)' % self._print(expr.expr)
            else:
                return 'O(%s)' % self.stringify((expr.expr,) + expr.variables, ', ', 0)
        else:
            return 'O(%s)' % self.stringify(expr.args, ', ', 0)

    def _print_Ordinal(self, expr):
        return expr.__str__()

    def _print_Cycle(self, expr):
        return expr.__str__()

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
        from sympy.utilities.exceptions import sympy_deprecation_warning

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=7,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            if not expr.size:
                return '()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            s = s.replace(',', '')
            return s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % self._print(expr.array_form)
                return 'Permutation([], size=%s)' % self._print(expr.size)
            trim = self._print(expr.array_form[:s[-1] + 1]) + ', size=%s' % self._print(expr.size)
            use = full = self._print(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use

    def _print_Subs(self, obj):
        expr, old, new = obj.args
        if len(obj.point) == 1:
            old = old[0]
            new = new[0]
        return "Subs(%s, %s, %s)" % (
            self._print(expr), self._print(old), self._print(new))

    def _print_TensorIndex(self, expr):
        return expr._print()

    def _print_TensorHead(self, expr):
        return expr._print()

    def _print_Tensor(self, expr):
        return expr._print()

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "*".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        return expr._print()

    def _print_ArraySymbol(self, expr):
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        return "%s[%s]" % (
            self.parenthesize(expr.name, PRECEDENCE["Func"], True), ", ".join([self._print(i) for i in expr.indices]))

    def _print_PermutationGroup(self, expr):
        p = ['    %s' % self._print(a) for a in expr.args]
        return 'PermutationGroup([\n%s])' % ',\n'.join(p)

    def _print_Pi(self, expr):
        return 'pi'

    def _print_PolyRing(self, ring):
        return "Polynomial ring in %s over %s with %s order" % \
            (", ".join((self._print(rs) for rs in ring.symbols)),
            self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        return "Rational function field in %s over %s with %s order" % \
            (", ".join((self._print(fs) for fs in field.symbols)),
            self._print(field.domain), self._print(field.order))

    def _print_FreeGroupElement(self, elm):
        return elm.__str__()

    def _print_GaussianElement(self, poly):
        return "(%s + %s*I)" % (poly.x, poly.y)

    def _print_PolyElement(self, poly):
        return poly.str(self, PRECEDENCE, "%s**%s", "*")

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self.parenthesize(frac.numer, PRECEDENCE["Mul"], strict=True)
            denom = self.parenthesize(frac.denom, PRECEDENCE["Atom"], strict=True)
            return numer + "/" + denom

    def _print_Poly(self, expr):
        ATOM_PREC = PRECEDENCE["Atom"] - 1
        terms, gens = [], [ self.parenthesize(s, ATOM_PREC) for s in expr.gens ]

        for monom, coeff in expr.terms():
            s_monom = []

            for i, e in enumerate(monom):
                if e > 0:
                    if e == 1:
                        s_monom.append(gens[i])
                    else:
                        s_monom.append(gens[i] + "**%d" % e)

            s_monom = "*".join(s_monom)

            if coeff.is_Add:
                if s_monom:
                    s_coeff = "(" + self._print(coeff) + ")"
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + "*" + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        format = expr.__class__.__name__ + "(%s, %s"

        from sympy.polys.polyerrors import PolynomialError

        try:
            format += ", modulus=%s" % expr.get_modulus()
        except PolynomialError:
            format += ", domain='%s'" % expr.get_domain()

        format += ")"

        for index, item in enumerate(gens):
            if len(item) > 2 and (item[:1] == "(" and item[len(item) - 1:] == ")"):
                gens[index] = item[1:len(item) - 1]

        return format % (' '.join(terms), ', '.join(gens))

    def _print_UniversalSet(self, p):
        return 'UniversalSet'

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_Pow(self, expr, rational=False):
        """Printing helper function for ``Pow``

        Parameters
        ==========

        rational : bool, optional
            If ``True``, it will not attempt printing ``sqrt(x)`` or
            ``x**S.Half`` as ``sqrt``, and will use ``x**(1/2)``
            instead.

            See examples for additional details

        Examples
        ========

        >>> from sympy import sqrt, StrPrinter
        >>> from sympy.abc import x

        How ``rational`` keyword works with ``sqrt``:

        >>> printer = StrPrinter()
        >>> printer._print_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._print_Pow(sqrt(x), rational=False)
        'sqrt(x)'
        >>> printer._print_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._print_Pow(1/sqrt(x), rational=False)
        '1/sqrt(x)'

        Notes
        =====

        ``sqrt(x)`` is canonicalized as ``Pow(x, S.Half)`` in SymPy,
        so there is no need of defining a separate printer for ``sqrt``.
        Instead, it should be handled here as well.
        """
        PREC = precedence(expr)

        if expr.exp is S.Half and not rational:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # Note: Don't test "expr.exp == -S.Half" here, because that will
                # match -0.5, which we don't want.
                return "%s/sqrt(%s)" % tuple((self._print(arg) for arg in (S.One, expr.base)))
            if expr.exp is -S.One:
                # Similarly to the S.Half case, don't test with "==" here.
                return '%s/%s' % (self._print(S.One),
                                  self.parenthesize(expr.base, PREC, strict=False))

        e = self.parenthesize(expr.exp, PREC, strict=False)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and expr.exp.q != 1:
            # the parenthesized exp should be '(Rational(a, b))' so strip parens,
            # but just check to be sure.
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e[1:-1])
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False),
                         self.parenthesize(expr.exp, PREC, strict=False))

    def _print_Integer(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(%s)" % (expr)
        return str(expr.p)

    def _print_Integers(self, expr):
        return 'Integers'

    def _print_Naturals(self, expr):
        return 'Naturals'

    def _print_Naturals0(self, expr):
        return 'Naturals0'

    def _print_Rationals(self, expr):
        return 'Rationals'

    def _print_Reals(self, expr):
        return 'Reals'

    def _print_Complexes(self, expr):
        return 'Complexes'

    def _print_EmptySet(self, expr):
        return 'EmptySet'

    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            if self._settings.get("sympy_integers", False):
                return "S(%s)/%s" % (expr.p, expr.q)
            return "%s/%s" % (expr.p, expr.q)

    def _print_PythonRational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            return "%d/%d" % (expr.p, expr.q)

    def _print_Fraction(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_mpq(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_Float(self, expr):
        prec = expr._prec
        dps = self._settings.get('dps', None)
        if dps is None:
            dps = 0 if prec < 5 else prec_to_dps(expr._prec)
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1
        low = self._settings["min"] if "min" in self._settings else None
        high = self._settings["max"] if "max" in self._settings else None
        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        rv = rv.removeprefix('+') # e.g., +inf -> inf
        return rv

    def _print_Relational(self, expr):

        charmap = {
            "==": "Eq",
            "!=": "Ne",
            ":=": "Assignment",
            '+=': "AddAugmentedAssignment",
            "-=": "SubAugmentedAssignment",
            "*=": "MulAugmentedAssignment",
            "/=": "DivAugmentedAssignment",
            "%=": "ModAugmentedAssignment",
        }

        if expr.rel_op in charmap:
            return '%s(%s, %s)' % (charmap[expr.rel_op], self._print(expr.lhs),
                                   self._print(expr.rhs))

        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
                           self._relationals.get(expr.rel_op) or expr.rel_op,
                           self.parenthesize(expr.rhs, precedence(expr)))

    def _print_ComplexRootOf(self, expr):
        return "CRootOf(%s, %d)" % (self._print_Add(expr.expr,  order='lex'),
                                    expr.index)

    def _print_RootSum(self, expr):
        args = [self._print_Add(expr.expr, order='lex')]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        return "RootSum(%s)" % ", ".join(args)

    def _print_GroebnerBasis(self, basis):
        cls = basis.__class__.__name__

        exprs = [self._print_Add(arg, order=basis.order) for arg in basis.exprs]
        exprs = "[%s]" % ", ".join(exprs)

        gens = [ self._print(gen) for gen in basis.gens ]
        domain = "domain='%s'" % self._print(basis.domain)
        order = "order='%s'" % self._print(basis.order)

        args = [exprs] + gens + [domain, order]

        return "%s(%s)" % (cls, ", ".join(args))

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(item) for item in items)
        if not args:
            return "set()"
        return '{%s}' % args

    def _print_FiniteSet(self, s):
        from sympy.sets.sets import FiniteSet
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(item) for item in items)
        if any(item.has(FiniteSet) for item in items):
            return 'FiniteSet({})'.format(args)
        return '{{{}}}'.format(args)

    def _print_Partition(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(arg) for arg in items)
        return 'Partition({})'.format(args)

    def _print_frozenset(self, s):
        if not s:
            return "frozenset()"
        return "frozenset(%s)" % self._print_set(s)

    def _print_Sum(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Sum(%s, %s)' % (self._print(expr.function), L)

    def _print_Symbol(self, expr):
        return expr.name
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Identity(self, expr):
        return "I"

    def _print_ZeroMatrix(self, expr):
        return "0"

    def _print_OneMatrix(self, expr):
        return "1"

    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    def _print_str(self, expr):
        return str(expr)

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.stringify(expr, ", ")

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_Transpose(self, T):
        return "%s.T" % self.parenthesize(T.arg, PRECEDENCE["Pow"])

    def _print_Uniform(self, expr):
        return "Uniform(%s, %s)" % (self._print(expr.a), self._print(expr.b))

    def _print_Quantity(self, expr):
        if self._settings.get("abbrev", False):
            return "%s" % expr.abbrev
        return "%s" % expr.name

    def _print_Quaternion(self, expr):
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True) for i in expr.args]
        a = [s[0]] + [i+"*"+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_Dimension(self, expr):
        return str(expr)

    def _print_Wild(self, expr):
        return expr.name + '_'

    def _print_WildFunction(self, expr):
        return expr.name + '_'

    def _print_WildDot(self, expr):
        return expr.name

    def _print_WildPlus(self, expr):
        return expr.name

    def _print_WildStar(self, expr):
        return expr.name

    def _print_Zero(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(0)"
        return self._print_Integer(Integer(0))

    def _print_DMP(self, p):
        cls = p.__class__.__name__
        rep = self._print(p.to_list())
        dom = self._print(p.dom)

        return "%s(%s, %s)" % (cls, rep, dom)

    def _print_DMF(self, expr):
        cls = expr.__class__.__name__
        num = self._print(expr.num)
        den = self._print(expr.den)
        dom = self._print(expr.dom)

        return "%s(%s, %s, %s)" % (cls, num, den, dom)

    def _print_Object(self, obj):
        return 'Object("%s")' % obj.name

    def _print_IdentityMorphism(self, morphism):
        return 'IdentityMorphism(%s)' % morphism.domain

    def _print_NamedMorphism(self, morphism):
        return 'NamedMorphism(%s, %s, "%s")' % \
               (morphism.domain, morphism.codomain, morphism.name)

    def _print_Category(self, category):
        return 'Category("%s")' % category.name

    def _print_Manifold(self, manifold):
        return manifold.name.name

    def _print_Patch(self, patch):
        return patch.name.name

    def _print_CoordSystem(self, coords):
        return coords.name.name

    def _print_BaseScalarField(self, field):
        return field._coord_sys.symbols[field._index].name

    def _print_BaseVectorField(self, field):
        return 'e_%s' % field._coord_sys.symbols[field._index].name

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            return 'd%s' % field._coord_sys.symbols[field._index].name
        else:
            return 'd(%s)' % self._print(field)

    def _print_Tr(self, expr):
        #TODO : Handle indices
        return "%s(%s)" % ("Tr", self._print(expr.args[0]))

    def _print_Str(self, s):
        return self._print(s.name)

    def _print_AppliedBinaryRelation(self, expr):
        rel = expr.function
        return '%s(%s, %s)' % (self._print(rel),
                               self._print(expr.lhs),
                               self._print(expr.rhs))


@print_function(StrPrinter)
def sstr(expr, **settings):
    """Returns the expression as a string.

    For large expressions where speed is a concern, use the setting
    order='none'. If abbrev=True setting is used then units are printed in
    abbreviated form.

    Examples
    ========

    >>> from sympy import symbols, Eq, sstr
    >>> a, b = symbols('a b')
    >>> sstr(Eq(a + b, 0))
    'Eq(a + b, 0)'
    """

    p = StrPrinter(settings)
    s = p.doprint(expr)

    return s


class StrReprPrinter(StrPrinter):
    """(internal) -- see sstrrepr"""

    def _print_str(self, s):
        return repr(s)

    def _print_Str(self, s):
        # Str does not to be printed same as str here
        return "%s(%s)" % (s.__class__.__name__, self._print(s.name))

@print_function(StrReprPrinter)
def sstrrepr(expr, **settings):
    """return expr in mixed str/repr form

       i.e. strings are returned in repr form with quotes, and everything else
       is returned in str form.

       This function could be useful for hooking into sys.displayhook
    """

    p = StrReprPrinter(settings)
    s = p.doprint(expr)

    return s

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\testing\_raises_group.py ===
from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from re import Pattern
from textwrap import indent
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    cast,
    overload,
)

from trio._util import final

if TYPE_CHECKING:
    import builtins

    # sphinx will *only* work if we use types.TracebackType, and import
    # *inside* TYPE_CHECKING. No other combination works.....
    import types
    from collections.abc import Callable, Sequence

    from _pytest._code.code import ExceptionChainRepr, ReprExceptionInfo, Traceback
    from typing_extensions import TypeGuard, TypeVar

    # this conditional definition is because we want to allow a TypeVar default
    MatchE = TypeVar(
        "MatchE",
        bound=BaseException,
        default=BaseException,
        covariant=True,
    )
else:
    from typing import TypeVar

    MatchE = TypeVar("MatchE", bound=BaseException, covariant=True)

# RaisesGroup doesn't work with a default.
BaseExcT_co = TypeVar("BaseExcT_co", bound=BaseException, covariant=True)
BaseExcT_1 = TypeVar("BaseExcT_1", bound=BaseException)
BaseExcT_2 = TypeVar("BaseExcT_2", bound=BaseException)
ExcT_1 = TypeVar("ExcT_1", bound=Exception)
ExcT_2 = TypeVar("ExcT_2", bound=Exception)

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup


@final
class _ExceptionInfo(Generic[MatchE]):
    """Minimal re-implementation of pytest.ExceptionInfo, only used if pytest is not available. Supports a subset of its features necessary for functionality of :class:`trio.testing.RaisesGroup` and :class:`trio.testing.Matcher`."""

    _excinfo: tuple[type[MatchE], MatchE, types.TracebackType] | None

    def __init__(
        self,
        excinfo: tuple[type[MatchE], MatchE, types.TracebackType] | None,
    ) -> None:
        self._excinfo = excinfo

    def fill_unfilled(
        self,
        exc_info: tuple[type[MatchE], MatchE, types.TracebackType],
    ) -> None:
        """Fill an unfilled ExceptionInfo created with ``for_later()``."""
        assert self._excinfo is None, "ExceptionInfo was already filled"
        self._excinfo = exc_info

    @classmethod
    def for_later(cls) -> _ExceptionInfo[MatchE]:
        """Return an unfilled ExceptionInfo."""
        return cls(None)

    # Note, special cased in sphinx config, since "type" conflicts.
    @property
    def type(self) -> type[MatchE]:
        """The exception class."""
        assert (
            self._excinfo is not None
        ), ".type can only be used after the context manager exits"
        return self._excinfo[0]

    @property
    def value(self) -> MatchE:
        """The exception value."""
        assert (
            self._excinfo is not None
        ), ".value can only be used after the context manager exits"
        return self._excinfo[1]

    @property
    def tb(self) -> types.TracebackType:
        """The exception raw traceback."""
        assert (
            self._excinfo is not None
        ), ".tb can only be used after the context manager exits"
        return self._excinfo[2]

    def exconly(self, tryshort: bool = False) -> str:
        raise NotImplementedError(
            "This is a helper method only available if you use RaisesGroup with the pytest package installed",
        )

    def errisinstance(
        self,
        exc: builtins.type[BaseException] | tuple[builtins.type[BaseException], ...],
    ) -> bool:
        raise NotImplementedError(
            "This is a helper method only available if you use RaisesGroup with the pytest package installed",
        )

    def getrepr(
        self,
        showlocals: bool = False,
        style: str = "long",
        abspath: bool = False,
        tbfilter: bool | Callable[[_ExceptionInfo], Traceback] = True,
        funcargs: bool = False,
        truncate_locals: bool = True,
        chain: bool = True,
    ) -> ReprExceptionInfo | ExceptionChainRepr:
        raise NotImplementedError(
            "This is a helper method only available if you use RaisesGroup with the pytest package installed",
        )


# Type checkers are not able to do conditional types depending on installed packages, so
# we've added signatures for all helpers to _ExceptionInfo, and then always use that.
# If this ends up leading to problems, we can resort to always using _ExceptionInfo and
# users that want to use getrepr/errisinstance/exconly can write helpers on their own, or
# we reimplement them ourselves...or get this merged in upstream pytest.
if TYPE_CHECKING:
    ExceptionInfo = _ExceptionInfo

else:
    try:
        from pytest import ExceptionInfo  # noqa: PT013
    except ImportError:  # pragma: no cover
        ExceptionInfo = _ExceptionInfo


# copied from pytest.ExceptionInfo
def _stringify_exception(exc: BaseException) -> str:
    return "\n".join(
        [
            getattr(exc, "message", str(exc)),
            *getattr(exc, "__notes__", []),
        ],
    )


# String patterns default to including the unicode flag.
_REGEX_NO_FLAGS = re.compile(r"").flags


def _match_pattern(match: Pattern[str]) -> str | Pattern[str]:
    """helper function to remove redundant `re.compile` calls when printing regex"""
    return match.pattern if match.flags == _REGEX_NO_FLAGS else match


def repr_callable(fun: Callable[[BaseExcT_1], bool]) -> str:
    """Get the repr of a ``check`` parameter.

    Split out so it can be monkeypatched (e.g. by our hypothesis plugin)
    """
    return repr(fun)


def _exception_type_name(e: type[BaseException]) -> str:
    return repr(e.__name__)


def _check_raw_type(
    expected_type: type[BaseException] | None,
    exception: BaseException,
) -> str | None:
    if expected_type is None:
        return None

    if not isinstance(
        exception,
        expected_type,
    ):
        actual_type_str = _exception_type_name(type(exception))
        expected_type_str = _exception_type_name(expected_type)
        if isinstance(exception, BaseExceptionGroup) and not issubclass(
            expected_type, BaseExceptionGroup
        ):
            return f"Unexpected nested {actual_type_str}, expected {expected_type_str}"
        return f"{actual_type_str} is not of type {expected_type_str}"
    return None


class AbstractMatcher(ABC, Generic[BaseExcT_co]):
    """ABC with common functionality shared between Matcher and RaisesGroup"""

    def __init__(
        self,
        match: str | Pattern[str] | None,
        check: Callable[[BaseExcT_co], bool] | None,
    ) -> None:
        if isinstance(match, str):
            self.match: Pattern[str] | None = re.compile(match)
        else:
            self.match = match
        self.check = check
        self._fail_reason: str | None = None

        # used to suppress repeated printing of `repr(self.check)`
        self._nested: bool = False

    @property
    def fail_reason(self) -> str | None:
        """Set after a call to `matches` to give a human-readable
        reason for why the match failed.
        When used as a context manager the string will be given as the text of an
        `AssertionError`"""
        return self._fail_reason

    def _check_check(
        self: AbstractMatcher[BaseExcT_1],
        exception: BaseExcT_1,
    ) -> bool:
        if self.check is None:
            return True

        if self.check(exception):
            return True

        check_repr = "" if self._nested else " " + repr_callable(self.check)
        self._fail_reason = f"check{check_repr} did not return True"
        return False

    def _check_match(self, e: BaseException) -> bool:
        if self.match is None or re.search(
            self.match,
            stringified_exception := _stringify_exception(e),
        ):
            return True

        maybe_specify_type = (
            f" of {_exception_type_name(type(e))}"
            if isinstance(e, BaseExceptionGroup)
            else ""
        )
        self._fail_reason = f"Regex pattern {_match_pattern(self.match)!r} did not match {stringified_exception!r}{maybe_specify_type}"
        if _match_pattern(self.match) == stringified_exception:
            self._fail_reason += "\n  Did you mean to `re.escape()` the regex?"
        return False

    # TODO: when transitioning to pytest, harmonize Matcher and RaisesGroup
    # signatures. One names the parameter `exc_val` and the other `exception`
    @abstractmethod
    def matches(
        self: AbstractMatcher[BaseExcT_1], exc_val: BaseException
    ) -> TypeGuard[BaseExcT_1]:
        """Check if an exception matches the requirements of this AbstractMatcher.
        If it fails, `AbstractMatcher.fail_reason` should be set.
        """


@final
class Matcher(AbstractMatcher[MatchE]):
    """Helper class to be used together with RaisesGroups when you want to specify requirements on sub-exceptions. Only specifying the type is redundant, and it's also unnecessary when the type is a nested `RaisesGroup` since it supports the same arguments.
    The type is checked with `isinstance`, and does not need to be an exact match. If that is wanted you can use the ``check`` parameter.
    :meth:`Matcher.matches` can also be used standalone to check individual exceptions.

    Examples::

        with RaisesGroups(Matcher(ValueError, match="string")):
            ...
        with RaisesGroups(Matcher(check=lambda x: x.args == (3, "hello"))):
            ...
        with RaisesGroups(Matcher(check=lambda x: type(x) is ValueError)):
            ...

    Tip: if you install ``hypothesis`` and import it in ``conftest.py`` you will get
    readable ``repr``s of ``check`` callables in the output.
    """

    # At least one of the three parameters must be passed.
    @overload
    def __init__(
        self,
        exception_type: type[MatchE],
        match: str | Pattern[str] = ...,
        check: Callable[[MatchE], bool] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: Matcher[BaseException],  # Give E a value.
        *,
        match: str | Pattern[str],
        # If exception_type is not provided, check() must do any typechecks itself.
        check: Callable[[BaseException], bool] = ...,
    ) -> None: ...

    @overload
    def __init__(self, *, check: Callable[[BaseException], bool]) -> None: ...

    def __init__(
        self,
        exception_type: type[MatchE] | None = None,
        match: str | Pattern[str] | None = None,
        check: Callable[[MatchE], bool] | None = None,
    ):
        super().__init__(match, check)
        if exception_type is None and match is None and check is None:
            raise ValueError("You must specify at least one parameter to match on.")
        if exception_type is not None and not issubclass(exception_type, BaseException):
            raise ValueError(
                f"exception_type {exception_type} must be a subclass of BaseException",
            )
        self.exception_type = exception_type

    def matches(
        self,
        exception: BaseException,
    ) -> TypeGuard[MatchE]:
        """Check if an exception matches the requirements of this Matcher.
        If it fails, `Matcher.fail_reason` will be set.

        Examples::

            assert Matcher(ValueError).matches(my_exception)
            # is equivalent to
            assert isinstance(my_exception, ValueError)

            # this can be useful when checking e.g. the ``__cause__`` of an exception.
            with pytest.raises(ValueError) as excinfo:
                ...
            assert Matcher(SyntaxError, match="foo").matches(excinfo.value.__cause__)
            # above line is equivalent to
            assert isinstance(excinfo.value.__cause__, SyntaxError)
            assert re.search("foo", str(excinfo.value.__cause__))

        """
        if not self._check_type(exception):
            return False

        if not self._check_match(exception):
            return False

        return self._check_check(exception)

    def __repr__(self) -> str:
        parameters = []
        if self.exception_type is not None:
            parameters.append(self.exception_type.__name__)
        if self.match is not None:
            # If no flags were specified, discard the redundant re.compile() here.
            parameters.append(
                f"match={_match_pattern(self.match)!r}",
            )
        if self.check is not None:
            parameters.append(f"check={repr_callable(self.check)}")
        return f'Matcher({", ".join(parameters)})'

    def _check_type(self, exception: BaseException) -> TypeGuard[MatchE]:
        self._fail_reason = _check_raw_type(self.exception_type, exception)
        return self._fail_reason is None


@final
class RaisesGroup(AbstractMatcher[BaseExceptionGroup[BaseExcT_co]]):
    """Contextmanager for checking for an expected `ExceptionGroup`.
    This works similar to ``pytest.raises``, and a version of it will hopefully be added upstream, after which this can be deprecated and removed. See https://github.com/pytest-dev/pytest/issues/11538


    The catching behaviour differs from :ref:`except* <except_star>` in multiple different ways, being much stricter by default. By using ``allow_unwrapped=True`` and ``flatten_subgroups=True`` you can match ``except*`` fully when expecting a single exception.

    #. All specified exceptions must be present, *and no others*.

       * If you expect a variable number of exceptions you need to use ``pytest.raises(ExceptionGroup)`` and manually check the contained exceptions. Consider making use of :func:`Matcher.matches`.

    #. It will only catch exceptions wrapped in an exceptiongroup by default.

       * With ``allow_unwrapped=True`` you can specify a single expected exception or `Matcher` and it will match the exception even if it is not inside an `ExceptionGroup`. If you expect one of several different exception types you need to use a `Matcher` object.

    #. By default it cares about the full structure with nested `ExceptionGroup`'s. You can specify nested `ExceptionGroup`'s by passing `RaisesGroup` objects as expected exceptions.

       * With ``flatten_subgroups=True`` it will "flatten" the raised `ExceptionGroup`, extracting all exceptions inside any nested :class:`ExceptionGroup`, before matching.

    It does not care about the order of the exceptions, so ``RaisesGroups(ValueError, TypeError)`` is equivalent to ``RaisesGroups(TypeError, ValueError)``.

    Examples::

        with RaisesGroups(ValueError):
            raise ExceptionGroup("", (ValueError(),))
        with RaisesGroups(ValueError, ValueError, Matcher(TypeError, match="expected int")):
            ...
        with RaisesGroups(KeyboardInterrupt, match="hello", check=lambda x: type(x) is BaseExceptionGroup):
            ...
        with RaisesGroups(RaisesGroups(ValueError)):
            raise ExceptionGroup("", (ExceptionGroup("", (ValueError(),)),))

        # flatten_subgroups
        with RaisesGroups(ValueError, flatten_subgroups=True):
            raise ExceptionGroup("", (ExceptionGroup("", (ValueError(),)),))

        # allow_unwrapped
        with RaisesGroups(ValueError, allow_unwrapped=True):
            raise ValueError


    `RaisesGroup.matches` can also be used directly to check a standalone exception group.


    The matching algorithm is greedy, which means cases such as this may fail::

        with RaisesGroups(ValueError, Matcher(ValueError, match="hello")):
            raise ExceptionGroup("", (ValueError("hello"), ValueError("goodbye")))

    even though it generally does not care about the order of the exceptions in the group.
    To avoid the above you should specify the first ValueError with a Matcher as well.

    Tip: if you install ``hypothesis`` and import it in ``conftest.py`` you will get
    readable ``repr``s of ``check`` callables in the output.
    """

    # allow_unwrapped=True requires: singular exception, exception not being
    # RaisesGroup instance, match is None, check is None
    @overload
    def __init__(
        self,
        exception: type[BaseExcT_co] | Matcher[BaseExcT_co],
        *,
        allow_unwrapped: Literal[True],
        flatten_subgroups: bool = False,
    ) -> None: ...

    # flatten_subgroups = True also requires no nested RaisesGroup
    @overload
    def __init__(
        self,
        exception: type[BaseExcT_co] | Matcher[BaseExcT_co],
        *other_exceptions: type[BaseExcT_co] | Matcher[BaseExcT_co],
        flatten_subgroups: Literal[True],
        match: str | Pattern[str] | None = None,
        check: Callable[[BaseExceptionGroup[BaseExcT_co]], bool] | None = None,
    ) -> None: ...

    # simplify the typevars if possible (the following 3 are equivalent but go simpler->complicated)
    # ... the first handles RaisesGroup[ValueError], the second RaisesGroup[ExceptionGroup[ValueError]],
    #     the third RaisesGroup[ValueError | ExceptionGroup[ValueError]].
    # ... otherwise, we will get results like RaisesGroup[ValueError | ExceptionGroup[Never]] (I think)
    #     (technically correct but misleading)
    @overload
    def __init__(
        self: RaisesGroup[ExcT_1],
        exception: type[ExcT_1] | Matcher[ExcT_1],
        *other_exceptions: type[ExcT_1] | Matcher[ExcT_1],
        match: str | Pattern[str] | None = None,
        check: Callable[[ExceptionGroup[ExcT_1]], bool] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RaisesGroup[ExceptionGroup[ExcT_2]],
        exception: RaisesGroup[ExcT_2],
        *other_exceptions: RaisesGroup[ExcT_2],
        match: str | Pattern[str] | None = None,
        check: Callable[[ExceptionGroup[ExceptionGroup[ExcT_2]]], bool] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RaisesGroup[ExcT_1 | ExceptionGroup[ExcT_2]],
        exception: type[ExcT_1] | Matcher[ExcT_1] | RaisesGroup[ExcT_2],
        *other_exceptions: type[ExcT_1] | Matcher[ExcT_1] | RaisesGroup[ExcT_2],
        match: str | Pattern[str] | None = None,
        check: (
            Callable[[ExceptionGroup[ExcT_1 | ExceptionGroup[ExcT_2]]], bool] | None
        ) = None,
    ) -> None: ...

    # same as the above 3 but handling BaseException
    @overload
    def __init__(
        self: RaisesGroup[BaseExcT_1],
        exception: type[BaseExcT_1] | Matcher[BaseExcT_1],
        *other_exceptions: type[BaseExcT_1] | Matcher[BaseExcT_1],
        match: str | Pattern[str] | None = None,
        check: Callable[[BaseExceptionGroup[BaseExcT_1]], bool] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RaisesGroup[BaseExceptionGroup[BaseExcT_2]],
        exception: RaisesGroup[BaseExcT_2],
        *other_exceptions: RaisesGroup[BaseExcT_2],
        match: str | Pattern[str] | None = None,
        check: (
            Callable[[BaseExceptionGroup[BaseExceptionGroup[BaseExcT_2]]], bool] | None
        ) = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RaisesGroup[BaseExcT_1 | BaseExceptionGroup[BaseExcT_2]],
        exception: type[BaseExcT_1] | Matcher[BaseExcT_1] | RaisesGroup[BaseExcT_2],
        *other_exceptions: type[BaseExcT_1]
        | Matcher[BaseExcT_1]
        | RaisesGroup[BaseExcT_2],
        match: str | Pattern[str] | None = None,
        check: (
            Callable[
                [BaseExceptionGroup[BaseExcT_1 | BaseExceptionGroup[BaseExcT_2]]],
                bool,
            ]
            | None
        ) = None,
    ) -> None: ...

    def __init__(
        self: RaisesGroup[ExcT_1 | BaseExcT_1 | BaseExceptionGroup[BaseExcT_2]],
        exception: type[BaseExcT_1] | Matcher[BaseExcT_1] | RaisesGroup[BaseExcT_2],
        *other_exceptions: type[BaseExcT_1]
        | Matcher[BaseExcT_1]
        | RaisesGroup[BaseExcT_2],
        allow_unwrapped: bool = False,
        flatten_subgroups: bool = False,
        match: str | Pattern[str] | None = None,
        check: (
            Callable[[BaseExceptionGroup[BaseExcT_1]], bool]
            | Callable[[ExceptionGroup[ExcT_1]], bool]
            | None
        ) = None,
    ):
        # The type hint on the `self` and `check` parameters uses different formats
        # that are *very* hard to reconcile while adhering to the overloads, so we cast
        # it to avoid an error when passing it to super().__init__
        check = cast(
            "Callable[["
            "BaseExceptionGroup[ExcT_1|BaseExcT_1|BaseExceptionGroup[BaseExcT_2]]"
            "], bool]",
            check,
        )
        super().__init__(match, check)
        self.expected_exceptions: tuple[
            type[BaseExcT_co] | Matcher[BaseExcT_co] | RaisesGroup[BaseException], ...
        ] = (
            exception,
            *other_exceptions,
        )
        self.allow_unwrapped = allow_unwrapped
        self.flatten_subgroups: bool = flatten_subgroups
        self.is_baseexceptiongroup: bool = False

        if allow_unwrapped and other_exceptions:
            raise ValueError(
                "You cannot specify multiple exceptions with `allow_unwrapped=True.`"
                " If you want to match one of multiple possible exceptions you should"
                " use a `Matcher`."
                " E.g. `Matcher(check=lambda e: isinstance(e, (...)))`",
            )
        if allow_unwrapped and isinstance(exception, RaisesGroup):
            raise ValueError(
                "`allow_unwrapped=True` has no effect when expecting a `RaisesGroup`."
                " You might want it in the expected `RaisesGroup`, or"
                " `flatten_subgroups=True` if you don't care about the structure.",
            )
        if allow_unwrapped and (match is not None or check is not None):
            raise ValueError(
                "`allow_unwrapped=True` bypasses the `match` and `check` parameters"
                " if the exception is unwrapped. If you intended to match/check the"
                " exception you should use a `Matcher` object. If you want to match/check"
                " the exceptiongroup when the exception *is* wrapped you need to"
                " do e.g. `if isinstance(exc.value, ExceptionGroup):"
                " assert RaisesGroup(...).matches(exc.value)` afterwards.",
            )

        # verify `expected_exceptions` and set `self.is_baseexceptiongroup`
        for exc in self.expected_exceptions:
            if isinstance(exc, RaisesGroup):
                if self.flatten_subgroups:
                    raise ValueError(
                        "You cannot specify a nested structure inside a RaisesGroup with"
                        " `flatten_subgroups=True`. The parameter will flatten subgroups"
                        " in the raised exceptiongroup before matching, which would never"
                        " match a nested structure.",
                    )
                self.is_baseexceptiongroup |= exc.is_baseexceptiongroup
                exc._nested = True
            elif isinstance(exc, Matcher):
                if exc.exception_type is not None:
                    # Matcher __init__ assures it's a subclass of BaseException
                    self.is_baseexceptiongroup |= not issubclass(
                        exc.exception_type,
                        Exception,
                    )
                exc._nested = True
            elif isinstance(exc, type) and issubclass(exc, BaseException):
                self.is_baseexceptiongroup |= not issubclass(exc, Exception)
            else:
                raise ValueError(
                    f'Invalid argument "{exc!r}" must be exception type, Matcher, or'
                    " RaisesGroup.",
                )

    @overload
    def __enter__(
        self: RaisesGroup[ExcT_1],
    ) -> ExceptionInfo[ExceptionGroup[ExcT_1]]: ...
    @overload
    def __enter__(
        self: RaisesGroup[BaseExcT_1],
    ) -> ExceptionInfo[BaseExceptionGroup[BaseExcT_1]]: ...

    def __enter__(self) -> ExceptionInfo[BaseExceptionGroup[BaseException]]:
        self.excinfo: ExceptionInfo[BaseExceptionGroup[BaseExcT_co]] = (
            ExceptionInfo.for_later()
        )
        return self.excinfo

    def __repr__(self) -> str:
        parameters = [
            e.__name__ if isinstance(e, type) else repr(e)
            for e in self.expected_exceptions
        ]
        if self.allow_unwrapped:
            parameters.append(f"allow_unwrapped={self.allow_unwrapped}")
        if self.flatten_subgroups:
            parameters.append(f"flatten_subgroups={self.flatten_subgroups}")
        if self.match is not None:
            # If no flags were specified, discard the redundant re.compile() here.
            parameters.append(f"match={_match_pattern(self.match)!r}")
        if self.check is not None:
            parameters.append(f"check={repr_callable(self.check)}")
        return f"RaisesGroup({', '.join(parameters)})"

    def _unroll_exceptions(
        self,
        exceptions: Sequence[BaseException],
    ) -> Sequence[BaseException]:
        """Used if `flatten_subgroups=True`."""
        res: list[BaseException] = []
        for exc in exceptions:
            if isinstance(exc, BaseExceptionGroup):
                res.extend(self._unroll_exceptions(exc.exceptions))

            else:
                res.append(exc)
        return res

    @overload
    def matches(
        self: RaisesGroup[ExcT_1],
        exc_val: BaseException | None,
    ) -> TypeGuard[ExceptionGroup[ExcT_1]]: ...
    @overload
    def matches(
        self: RaisesGroup[BaseExcT_1],
        exc_val: BaseException | None,
    ) -> TypeGuard[BaseExceptionGroup[BaseExcT_1]]: ...

    def matches(
        self,
        exc_val: BaseException | None,
    ) -> TypeGuard[BaseExceptionGroup[BaseExcT_co]]:
        """Check if an exception matches the requirements of this RaisesGroup.
        If it fails, `RaisesGroup.fail_reason` will be set.

        Example::

            with pytest.raises(TypeError) as excinfo:
                ...
            assert RaisesGroups(ValueError).matches(excinfo.value.__cause__)
            # the above line is equivalent to
            myexc = excinfo.value.__cause
            assert isinstance(myexc, BaseExceptionGroup)
            assert len(myexc.exceptions) == 1
            assert isinstance(myexc.exceptions[0], ValueError)
        """
        self._fail_reason = None
        if exc_val is None:
            self._fail_reason = "exception is None"
            return False
        if not isinstance(exc_val, BaseExceptionGroup):
            # we opt to only print type of the exception here, as the repr would
            # likely be quite long
            not_group_msg = f"{type(exc_val).__name__!r} is not an exception group"
            if len(self.expected_exceptions) > 1:
                self._fail_reason = not_group_msg
                return False
            # if we have 1 expected exception, check if it would work even if
            # allow_unwrapped is not set
            res = self._check_expected(self.expected_exceptions[0], exc_val)
            if res is None and self.allow_unwrapped:
                return True

            if res is None:
                self._fail_reason = (
                    f"{not_group_msg}, but would match with `allow_unwrapped=True`"
                )
            elif self.allow_unwrapped:
                self._fail_reason = res
            else:
                self._fail_reason = not_group_msg
            return False

        actual_exceptions: Sequence[BaseException] = exc_val.exceptions
        if self.flatten_subgroups:
            actual_exceptions = self._unroll_exceptions(actual_exceptions)

        if not self._check_match(exc_val):
            old_reason = self._fail_reason
            if (
                len(actual_exceptions) == len(self.expected_exceptions) == 1
                and isinstance(expected := self.expected_exceptions[0], type)
                and isinstance(actual := actual_exceptions[0], expected)
                and self._check_match(actual)
            ):
                assert self.match is not None, "can't be None if _check_match failed"
                assert self._fail_reason is old_reason is not None
                self._fail_reason += f", but matched the expected {self._repr_expected(expected)}. You might want RaisesGroup(Matcher({expected.__name__}, match={_match_pattern(self.match)!r}))"
            else:
                self._fail_reason = old_reason
            return False

        # do the full check on expected exceptions
        if not self._check_exceptions(
            exc_val,
            actual_exceptions,
        ):
            assert self._fail_reason is not None
            old_reason = self._fail_reason
            # if we're not expecting a nested structure, and there is one, do a second
            # pass where we try flattening it
            if (
                not self.flatten_subgroups
                and not any(
                    isinstance(e, RaisesGroup) for e in self.expected_exceptions
                )
                and any(isinstance(e, BaseExceptionGroup) for e in actual_exceptions)
                and self._check_exceptions(
                    exc_val,
                    self._unroll_exceptions(exc_val.exceptions),
                )
            ):
                # only indent if it's a single-line reason. In a multi-line there's already
                # indented lines that this does not belong to.
                indent = "  " if "\n" not in self._fail_reason else ""
                self._fail_reason = (
                    old_reason
                    + f"\n{indent}Did you mean to use `flatten_subgroups=True`?"
                )
            else:
                self._fail_reason = old_reason
            return False

        # Only run `self.check` once we know `exc_val` is of the correct type.
        # TODO: if this fails, we should say the *group* did not match
        return self._check_check(exc_val)

    @staticmethod
    def _check_expected(
        expected_type: (
            type[BaseException] | Matcher[BaseException] | RaisesGroup[BaseException]
        ),
        exception: BaseException,
    ) -> str | None:
        """Helper method for `RaisesGroup.matches` and `RaisesGroup._check_exceptions`
        to check one of potentially several expected exceptions."""
        if isinstance(expected_type, type):
            return _check_raw_type(expected_type, exception)
        res = expected_type.matches(exception)
        if res:
            return None
        assert expected_type.fail_reason is not None
        if expected_type.fail_reason.startswith("\n"):
            return f"\n{expected_type!r}: {indent(expected_type.fail_reason, '  ')}"
        return f"{expected_type!r}: {expected_type.fail_reason}"

    @staticmethod
    def _repr_expected(e: type[BaseException] | AbstractMatcher[BaseException]) -> str:
        """Get the repr of an expected type/Matcher/RaisesGroup, but we only want
        the name if it's a type"""
        if isinstance(e, type):
            return _exception_type_name(e)
        return repr(e)

    @overload
    def _check_exceptions(
        self: RaisesGroup[ExcT_1],
        _exc_val: Exception,
        actual_exceptions: Sequence[Exception],
    ) -> TypeGuard[ExceptionGroup[ExcT_1]]: ...
    @overload
    def _check_exceptions(
        self: RaisesGroup[BaseExcT_1],
        _exc_val: BaseException,
        actual_exceptions: Sequence[BaseException],
    ) -> TypeGuard[BaseExceptionGroup[BaseExcT_1]]: ...

    def _check_exceptions(
        self,
        _exc_val: BaseException,
        actual_exceptions: Sequence[BaseException],
    ) -> TypeGuard[BaseExceptionGroup[BaseExcT_co]]:
        """helper method for RaisesGroup.matches that attempts to pair up expected and actual exceptions"""
        # full table with all results
        results = ResultHolder(self.expected_exceptions, actual_exceptions)

        # (indexes of) raised exceptions that haven't (yet) found an expected
        remaining_actual = list(range(len(actual_exceptions)))
        # (indexes of) expected exceptions that haven't found a matching raised
        failed_expected: list[int] = []
        # successful greedy matches
        matches: dict[int, int] = {}

        # loop over expected exceptions first to get a more predictable result
        for i_exp, expected in enumerate(self.expected_exceptions):
            for i_rem in remaining_actual:
                res = self._check_expected(expected, actual_exceptions[i_rem])
                results.set_result(i_exp, i_rem, res)
                if res is None:
                    remaining_actual.remove(i_rem)
                    matches[i_exp] = i_rem
                    break
            else:
                failed_expected.append(i_exp)

        # All exceptions matched up successfully
        if not remaining_actual and not failed_expected:
            return True

        # in case of a single expected and single raised we simplify the output
        if 1 == len(actual_exceptions) == len(self.expected_exceptions):
            assert not matches
            self._fail_reason = res
            return False

        # The test case is failing, so we can do a slow and exhaustive check to find
        # duplicate matches etc that will be helpful in debugging
        for i_exp, expected in enumerate(self.expected_exceptions):
            for i_actual, actual in enumerate(actual_exceptions):
                if results.has_result(i_exp, i_actual):
                    continue
                results.set_result(
                    i_exp, i_actual, self._check_expected(expected, actual)
                )

        successful_str = (
            f"{len(matches)} matched exception{'s' if len(matches) > 1 else ''}. "
            if matches
            else ""
        )

        # all expected were found
        if not failed_expected and results.no_match_for_actual(remaining_actual):
            self._fail_reason = f"{successful_str}Unexpected exception(s): {[actual_exceptions[i] for i in remaining_actual]!r}"
            return False
        # all raised exceptions were expected
        if not remaining_actual and results.no_match_for_expected(failed_expected):
            self._fail_reason = f"{successful_str}Too few exceptions raised, found no match for: [{', '.join(self._repr_expected(self.expected_exceptions[i]) for i in failed_expected)}]"
            return False

        # if there's only one remaining and one failed, and the unmatched didn't match anything else,
        # we elect to only print why the remaining and the failed didn't match.
        if (
            1 == len(remaining_actual) == len(failed_expected)
            and results.no_match_for_actual(remaining_actual)
            and results.no_match_for_expected(failed_expected)
        ):
            self._fail_reason = f"{successful_str}{results.get_result(failed_expected[0], remaining_actual[0])}"
            return False

        # there's both expected and raised exceptions without matches
        s = ""
        if matches:
            s += f"\n{successful_str}"
        indent_1 = " " * 2
        indent_2 = " " * 4

        if not remaining_actual:
            s += "\nToo few exceptions raised!"
        elif not failed_expected:
            s += "\nUnexpected exception(s)!"

        if failed_expected:
            s += "\nThe following expected exceptions did not find a match:"
            rev_matches = {v: k for k, v in matches.items()}
        for i_failed in failed_expected:
            s += (
                f"\n{indent_1}{self._repr_expected(self.expected_exceptions[i_failed])}"
            )
            for i_actual, actual in enumerate(actual_exceptions):
                if results.get_result(i_exp, i_actual) is None:
                    # we print full repr of match target
                    s += f"\n{indent_2}It matches {actual!r} which was paired with {self._repr_expected(self.expected_exceptions[rev_matches[i_actual]])}"

        if remaining_actual:
            s += "\nThe following raised exceptions did not find a match"
        for i_actual in remaining_actual:
            s += f"\n{indent_1}{actual_exceptions[i_actual]!r}:"
            for i_exp, expected in enumerate(self.expected_exceptions):
                res = results.get_result(i_exp, i_actual)
                if i_exp in failed_expected:
                    assert res is not None
                    if res[0] != "\n":
                        s += "\n"
                    s += indent(res, indent_2)
                if res is None:
                    # we print full repr of match target
                    s += f"\n{indent_2}It matches {self._repr_expected(expected)} which was paired with {actual_exceptions[matches[i_exp]]!r}"

        if len(self.expected_exceptions) == len(actual_exceptions) and possible_match(
            results
        ):
            s += "\nThere exist a possible match when attempting an exhaustive check, but RaisesGroup uses a greedy algorithm. Please make your expected exceptions more stringent with `Matcher` etc so the greedy algorithm can function."
        self._fail_reason = s
        return False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        __tracebackhide__ = True
        assert (
            exc_type is not None
        ), f"DID NOT RAISE any exception, expected {self.expected_type()}"
        assert (
            self.excinfo is not None
        ), "Internal error - should have been constructed in __enter__"

        group_str = (
            "(group)"
            if self.allow_unwrapped and not issubclass(exc_type, BaseExceptionGroup)
            else "group"
        )

        assert self.matches(
            exc_val,
        ), f"Raised exception {group_str} did not match: {self._fail_reason}"

        # Cast to narrow the exception type now that it's verified.
        exc_info = cast(
            "tuple[type[BaseExceptionGroup[BaseExcT_co]], BaseExceptionGroup[BaseExcT_co], types.TracebackType]",
            (exc_type, exc_val, exc_tb),
        )
        self.excinfo.fill_unfilled(exc_info)
        return True

    def expected_type(self) -> str:
        subexcs = []
        for e in self.expected_exceptions:
            if isinstance(e, Matcher):
                subexcs.append(str(e))
            elif isinstance(e, RaisesGroup):
                subexcs.append(e.expected_type())
            elif isinstance(e, type):
                subexcs.append(e.__name__)
            else:  # pragma: no cover
                raise AssertionError("unknown type")
        group_type = "Base" if self.is_baseexceptiongroup else ""
        return f"{group_type}ExceptionGroup({', '.join(subexcs)})"


@final
class NotChecked: ...


class ResultHolder:
    def __init__(
        self,
        expected_exceptions: tuple[
            type[BaseException] | AbstractMatcher[BaseException], ...
        ],
        actual_exceptions: Sequence[BaseException],
    ) -> None:
        self.results: list[list[str | type[NotChecked] | None]] = [
            [NotChecked for _ in expected_exceptions] for _ in actual_exceptions
        ]

    def set_result(self, expected: int, actual: int, result: str | None) -> None:
        self.results[actual][expected] = result

    def get_result(self, expected: int, actual: int) -> str | None:
        res = self.results[actual][expected]
        # mypy doesn't support `assert res is not NotChecked`
        assert not isinstance(res, type)
        return res

    def has_result(self, expected: int, actual: int) -> bool:
        return self.results[actual][expected] is not NotChecked

    def no_match_for_expected(self, expected: list[int]) -> bool:
        for i in expected:
            for actual_results in self.results:
                assert actual_results[i] is not NotChecked
                if actual_results[i] is None:
                    return False
        return True

    def no_match_for_actual(self, actual: list[int]) -> bool:
        for i in actual:
            for res in self.results[i]:
                assert res is not NotChecked
                if res is None:
                    return False
        return True


def possible_match(results: ResultHolder, used: set[int] | None = None) -> bool:
    if used is None:
        used = set()
    curr_row = len(used)
    if curr_row == len(results.results):
        return True

    for i, val in enumerate(results.results[curr_row]):
        if val is None and i not in used and possible_match(results, used | {i}):
            return True
    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\specifiers.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""
.. testsetup::

    from packaging.specifiers import Specifier, SpecifierSet, InvalidSpecifier
    from packaging.version import Version
"""

from __future__ import annotations

import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, TypeVar, Union

from .utils import canonicalize_version
from .version import Version

UnparsedVersion = Union[Version, str]
UnparsedVersionVar = TypeVar("UnparsedVersionVar", bound=UnparsedVersion)
CallableOperator = Callable[[Version, str], bool]


def _coerce_version(version: UnparsedVersion) -> Version:
    if not isinstance(version, Version):
        version = Version(version)
    return version


class InvalidSpecifier(ValueError):
    """
    Raised when attempting to create a :class:`Specifier` with a specifier
    string that is invalid.

    >>> Specifier("lolwat")
    Traceback (most recent call last):
        ...
    packaging.specifiers.InvalidSpecifier: Invalid specifier: 'lolwat'
    """


class BaseSpecifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns the str representation of this Specifier-like object. This
        should be representative of the Specifier itself.
        """

    @abc.abstractmethod
    def __hash__(self) -> int:
        """
        Returns a hash value for this Specifier-like object.
        """

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Returns a boolean representing whether or not the two Specifier-like
        objects are equal.

        :param other: The other object to check against.
        """

    @property
    @abc.abstractmethod
    def prereleases(self) -> bool | None:
        """Whether or not pre-releases as a whole are allowed.

        This can be set to either ``True`` or ``False`` to explicitly enable or disable
        prereleases or it can be set to ``None`` (the default) to use default semantics.
        """

    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        """Setter for :attr:`prereleases`.

        :param value: The value to set.
        """

    @abc.abstractmethod
    def contains(self, item: str, prereleases: bool | None = None) -> bool:
        """
        Determines if the given item is contained within this specifier.
        """

    @abc.abstractmethod
    def filter(
        self, iterable: Iterable[UnparsedVersionVar], prereleases: bool | None = None
    ) -> Iterator[UnparsedVersionVar]:
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """


class Specifier(BaseSpecifier):
    """This class abstracts handling of version specifiers.

    .. tip::

        It is generally not required to instantiate this manually. You should instead
        prefer to work with :class:`SpecifierSet` instead, which can parse
        comma-separated version specifiers (which is what package metadata contains).
    """

    _operator_regex_str = r"""
        (?P<operator>(~=|==|!=|<=|>=|<|>|===))
        """
    _version_regex_str = r"""
        (?P<version>
            (?:
                # The identity operators allow for an escape hatch that will
                # do an exact string match of the version you wish to install.
                # This will not be parsed by PEP 440 and we cannot determine
                # any semantic meaning from it. This operator is discouraged
                # but included entirely as an escape hatch.
                (?<====)  # Only match for the identity operator
                \s*
                [^\s;)]*  # The arbitrary version can be just about anything,
                          # we match everything except for whitespace, a
                          # semi-colon for marker support, and a closing paren
                          # since versions can be enclosed in them.
            )
            |
            (?:
                # The (non)equality operators allow for wild card and local
                # versions to be specified so we have to define these two
                # operators separately to enable that.
                (?<===|!=)            # Only match for equals and not equals

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release

                # You cannot use a wild card and a pre-release, post-release, a dev or
                # local version together so group them with a | and make them optional.
                (?:
                    \.\*  # Wild card syntax of .*
                    |
                    (?:                                  # pre release
                        [-_\.]?
                        (alpha|beta|preview|pre|a|b|c|rc)
                        [-_\.]?
                        [0-9]*
                    )?
                    (?:                                  # post release
                        (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                    )?
                    (?:[-_\.]?dev[-_\.]?[0-9]*)?         # dev release
                    (?:\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*)? # local
                )?
            )
            |
            (?:
                # The compatible operator requires at least two digits in the
                # release segment.
                (?<=~=)               # Only match for the compatible operator

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)+   # release  (We have a + instead of a *)
                (?:                   # pre release
                    [-_\.]?
                    (alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
            |
            (?:
                # All other operators only allow a sub set of what the
                # (non)equality operators do. Specifically they do not allow
                # local versions to be specified nor do they allow the prefix
                # matching wild cards.
                (?<!==|!=|~=)         # We have special cases for these
                                      # operators so we want to make sure they
                                      # don't match here.

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
        )
        """

    _regex = re.compile(
        r"^\s*" + _operator_regex_str + _version_regex_str + r"\s*$",
        re.VERBOSE | re.IGNORECASE,
    )

    _operators = {
        "~=": "compatible",
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
        "===": "arbitrary",
    }

    def __init__(self, spec: str = "", prereleases: bool | None = None) -> None:
        """Initialize a Specifier instance.

        :param spec:
            The string representation of a specifier which will be parsed and
            normalized before use.
        :param prereleases:
            This tells the specifier if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.
        :raises InvalidSpecifier:
            If the given specifier is invalid (i.e. bad syntax).
        """
        match = self._regex.search(spec)
        if not match:
            raise InvalidSpecifier(f"Invalid specifier: {spec!r}")

        self._spec: tuple[str, str] = (
            match.group("operator").strip(),
            match.group("version").strip(),
        )

        # Store whether or not this Specifier should accept prereleases
        self._prereleases = prereleases

    # https://github.com/python/mypy/pull/13475#pullrequestreview-1079784515
    @property  # type: ignore[override]
    def prereleases(self) -> bool:
        # If there is an explicit prereleases set for this, then we'll just
        # blindly use that.
        if self._prereleases is not None:
            return self._prereleases

        # Look at all of our specifiers and determine if they are inclusive
        # operators, and if they are if they are including an explicit
        # prerelease.
        operator, version = self._spec
        if operator in ["==", ">=", "<=", "~=", "===", ">", "<"]:
            # The == specifier can include a trailing .*, if it does we
            # want to remove before parsing.
            if operator == "==" and version.endswith(".*"):
                version = version[:-2]

            # Parse the version, and if it is a pre-release than this
            # specifier allows pre-releases.
            if Version(version).is_prerelease:
                return True

        return False

    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        self._prereleases = value

    @property
    def operator(self) -> str:
        """The operator of this specifier.

        >>> Specifier("==1.2.3").operator
        '=='
        """
        return self._spec[0]

    @property
    def version(self) -> str:
        """The version of this specifier.

        >>> Specifier("==1.2.3").version
        '1.2.3'
        """
        return self._spec[1]

    def __repr__(self) -> str:
        """A representation of the Specifier that shows all internal state.

        >>> Specifier('>=1.0.0')
        <Specifier('>=1.0.0')>
        >>> Specifier('>=1.0.0', prereleases=False)
        <Specifier('>=1.0.0', prereleases=False)>
        >>> Specifier('>=1.0.0', prereleases=True)
        <Specifier('>=1.0.0', prereleases=True)>
        """
        pre = (
            f", prereleases={self.prereleases!r}"
            if self._prereleases is not None
            else ""
        )

        return f"<{self.__class__.__name__}({str(self)!r}{pre})>"

    def __str__(self) -> str:
        """A string representation of the Specifier that can be round-tripped.

        >>> str(Specifier('>=1.0.0'))
        '>=1.0.0'
        >>> str(Specifier('>=1.0.0', prereleases=False))
        '>=1.0.0'
        """
        return "{}{}".format(*self._spec)

    @property
    def _canonical_spec(self) -> tuple[str, str]:
        canonical_version = canonicalize_version(
            self._spec[1],
            strip_trailing_zero=(self._spec[0] != "~="),
        )
        return self._spec[0], canonical_version

    def __hash__(self) -> int:
        return hash(self._canonical_spec)

    def __eq__(self, other: object) -> bool:
        """Whether or not the two Specifier-like objects are equal.

        :param other: The other object to check against.

        The value of :attr:`prereleases` is ignored.

        >>> Specifier("==1.2.3") == Specifier("== 1.2.3.0")
        True
        >>> (Specifier("==1.2.3", prereleases=False) ==
        ...  Specifier("==1.2.3", prereleases=True))
        True
        >>> Specifier("==1.2.3") == "==1.2.3"
        True
        >>> Specifier("==1.2.3") == Specifier("==1.2.4")
        False
        >>> Specifier("==1.2.3") == Specifier("~=1.2.3")
        False
        """
        if isinstance(other, str):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._canonical_spec == other._canonical_spec

    def _get_operator(self, op: str) -> CallableOperator:
        operator_callable: CallableOperator = getattr(
            self, f"_compare_{self._operators[op]}"
        )
        return operator_callable

    def _compare_compatible(self, prospective: Version, spec: str) -> bool:
        # Compatible releases have an equivalent combination of >= and ==. That
        # is that ~=2.2 is equivalent to >=2.2,==2.*. This allows us to
        # implement this in terms of the other specifiers instead of
        # implementing it ourselves. The only thing we need to do is construct
        # the other specifiers.

        # We want everything but the last item in the version, but we want to
        # ignore suffix segments.
        prefix = _version_join(
            list(itertools.takewhile(_is_not_suffix, _version_split(spec)))[:-1]
        )

        # Add the prefix notation to the end of our string
        prefix += ".*"

        return self._get_operator(">=")(prospective, spec) and self._get_operator("==")(
            prospective, prefix
        )

    def _compare_equal(self, prospective: Version, spec: str) -> bool:
        # We need special logic to handle prefix matching
        if spec.endswith(".*"):
            # In the case of prefix matching we want to ignore local segment.
            normalized_prospective = canonicalize_version(
                prospective.public, strip_trailing_zero=False
            )
            # Get the normalized version string ignoring the trailing .*
            normalized_spec = canonicalize_version(spec[:-2], strip_trailing_zero=False)
            # Split the spec out by bangs and dots, and pretend that there is
            # an implicit dot in between a release segment and a pre-release segment.
            split_spec = _version_split(normalized_spec)

            # Split the prospective version out by bangs and dots, and pretend
            # that there is an implicit dot in between a release segment and
            # a pre-release segment.
            split_prospective = _version_split(normalized_prospective)

            # 0-pad the prospective version before shortening it to get the correct
            # shortened version.
            padded_prospective, _ = _pad_version(split_prospective, split_spec)

            # Shorten the prospective version to be the same length as the spec
            # so that we can determine if the specifier is a prefix of the
            # prospective version or not.
            shortened_prospective = padded_prospective[: len(split_spec)]

            return shortened_prospective == split_spec
        else:
            # Convert our spec string into a Version
            spec_version = Version(spec)

            # If the specifier does not have a local segment, then we want to
            # act as if the prospective version also does not have a local
            # segment.
            if not spec_version.local:
                prospective = Version(prospective.public)

            return prospective == spec_version

    def _compare_not_equal(self, prospective: Version, spec: str) -> bool:
        return not self._compare_equal(prospective, spec)

    def _compare_less_than_equal(self, prospective: Version, spec: str) -> bool:
        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return Version(prospective.public) <= Version(spec)

    def _compare_greater_than_equal(self, prospective: Version, spec: str) -> bool:
        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return Version(prospective.public) >= Version(spec)

    def _compare_less_than(self, prospective: Version, spec_str: str) -> bool:
        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec_str)

        # Check to see if the prospective version is less than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective < spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a pre-release version, that we do not accept pre-release
        # versions for the version mentioned in the specifier (e.g. <3.1 should
        # not match 3.1.dev0, but should match 3.0.dev0).
        if not spec.is_prerelease and prospective.is_prerelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # less than the spec version *and* it's not a pre-release of the same
        # version in the spec.
        return True

    def _compare_greater_than(self, prospective: Version, spec_str: str) -> bool:
        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec_str)

        # Check to see if the prospective version is greater than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective > spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a post-release version, that we do not accept
        # post-release versions for the version mentioned in the specifier
        # (e.g. >3.1 should not match 3.0.post0, but should match 3.2.post0).
        if not spec.is_postrelease and prospective.is_postrelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # Ensure that we do not allow a local version of the version mentioned
        # in the specifier, which is technically greater than, to match.
        if prospective.local is not None:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # greater than the spec version *and* it's not a pre-release of the
        # same version in the spec.
        return True

    def _compare_arbitrary(self, prospective: Version, spec: str) -> bool:
        return str(prospective).lower() == str(spec).lower()

    def __contains__(self, item: str | Version) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item: The item to check for.

        This is used for the ``in`` operator and behaves the same as
        :meth:`contains` with no ``prereleases`` argument passed.

        >>> "1.2.3" in Specifier(">=1.2.3")
        True
        >>> Version("1.2.3") in Specifier(">=1.2.3")
        True
        >>> "1.0.0" in Specifier(">=1.2.3")
        False
        >>> "1.3.0a1" in Specifier(">=1.2.3")
        False
        >>> "1.3.0a1" in Specifier(">=1.2.3", prereleases=True)
        True
        """
        return self.contains(item)

    def contains(self, item: UnparsedVersion, prereleases: bool | None = None) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item:
            The item to check for, which can be a version string or a
            :class:`Version` instance.
        :param prereleases:
            Whether or not to match prereleases with this Specifier. If set to
            ``None`` (the default), it uses :attr:`prereleases` to determine
            whether or not prereleases are allowed.

        >>> Specifier(">=1.2.3").contains("1.2.3")
        True
        >>> Specifier(">=1.2.3").contains(Version("1.2.3"))
        True
        >>> Specifier(">=1.2.3").contains("1.0.0")
        False
        >>> Specifier(">=1.2.3").contains("1.3.0a1")
        False
        >>> Specifier(">=1.2.3", prereleases=True).contains("1.3.0a1")
        True
        >>> Specifier(">=1.2.3").contains("1.3.0a1", prereleases=True)
        True
        """

        # Determine if prereleases are to be allowed or not.
        if prereleases is None:
            prereleases = self.prereleases

        # Normalize item to a Version, this allows us to have a shortcut for
        # "2.0" in Specifier(">=2")
        normalized_item = _coerce_version(item)

        # Determine if we should be supporting prereleases in this specifier
        # or not, if we do not support prereleases than we can short circuit
        # logic if this version is a prereleases.
        if normalized_item.is_prerelease and not prereleases:
            return False

        # Actually do the comparison to determine if this item is contained
        # within this Specifier or not.
        operator_callable: CallableOperator = self._get_operator(self.operator)
        return operator_callable(normalized_item, self.version)

    def filter(
        self, iterable: Iterable[UnparsedVersionVar], prereleases: bool | None = None
    ) -> Iterator[UnparsedVersionVar]:
        """Filter items in the given iterable, that match the specifier.

        :param iterable:
            An iterable that can contain version strings and :class:`Version` instances.
            The items in the iterable will be filtered according to the specifier.
        :param prereleases:
            Whether or not to allow prereleases in the returned iterator. If set to
            ``None`` (the default), it will be intelligently decide whether to allow
            prereleases or not (based on the :attr:`prereleases` attribute, and
            whether the only versions matching are prereleases).

        This method is smarter than just ``filter(Specifier().contains, [...])``
        because it implements the rule from :pep:`440` that a prerelease item
        SHOULD be accepted if no other versions match the given specifier.

        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.3", "1.5a1"]))
        ['1.3']
        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.2.3", "1.3", Version("1.4")]))
        ['1.2.3', '1.3', <Version('1.4')>]
        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.5a1"]))
        ['1.5a1']
        >>> list(Specifier(">=1.2.3").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        >>> list(Specifier(">=1.2.3", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        """

        yielded = False
        found_prereleases = []

        kw = {"prereleases": prereleases if prereleases is not None else True}

        # Attempt to iterate over all the values in the iterable and if any of
        # them match, yield them.
        for version in iterable:
            parsed_version = _coerce_version(version)

            if self.contains(parsed_version, **kw):
                # If our version is a prerelease, and we were not set to allow
                # prereleases, then we'll store it for later in case nothing
                # else matches this specifier.
                if parsed_version.is_prerelease and not (
                    prereleases or self.prereleases
                ):
                    found_prereleases.append(version)
                # Either this is not a prerelease, or we should have been
                # accepting prereleases from the beginning.
                else:
                    yielded = True
                    yield version

        # Now that we've iterated over everything, determine if we've yielded
        # any values, and if we have not and we have any prereleases stored up
        # then we will go ahead and yield the prereleases.
        if not yielded and found_prereleases:
            for version in found_prereleases:
                yield version


_prefix_regex = re.compile(r"^([0-9]+)((?:a|b|c|rc)[0-9]+)$")


def _version_split(version: str) -> list[str]:
    """Split version into components.

    The split components are intended for version comparison. The logic does
    not attempt to retain the original version string, so joining the
    components back with :func:`_version_join` may not produce the original
    version string.
    """
    result: list[str] = []

    epoch, _, rest = version.rpartition("!")
    result.append(epoch or "0")

    for item in rest.split("."):
        match = _prefix_regex.search(item)
        if match:
            result.extend(match.groups())
        else:
            result.append(item)
    return result


def _version_join(components: list[str]) -> str:
    """Join split version components into a version string.

    This function assumes the input came from :func:`_version_split`, where the
    first component must be the epoch (either empty or numeric), and all other
    components numeric.
    """
    epoch, *rest = components
    return f"{epoch}!{'.'.join(rest)}"


def _is_not_suffix(segment: str) -> bool:
    return not any(
        segment.startswith(prefix) for prefix in ("dev", "a", "b", "rc", "post")
    )


def _pad_version(left: list[str], right: list[str]) -> tuple[list[str], list[str]]:
    left_split, right_split = [], []

    # Get the release segment of our versions
    left_split.append(list(itertools.takewhile(lambda x: x.isdigit(), left)))
    right_split.append(list(itertools.takewhile(lambda x: x.isdigit(), right)))

    # Get the rest of our versions
    left_split.append(left[len(left_split[0]) :])
    right_split.append(right[len(right_split[0]) :])

    # Insert our padding
    left_split.insert(1, ["0"] * max(0, len(right_split[0]) - len(left_split[0])))
    right_split.insert(1, ["0"] * max(0, len(left_split[0]) - len(right_split[0])))

    return (
        list(itertools.chain.from_iterable(left_split)),
        list(itertools.chain.from_iterable(right_split)),
    )


class SpecifierSet(BaseSpecifier):
    """This class abstracts handling of a set of version specifiers.

    It can be passed a single specifier (``>=3.0``), a comma-separated list of
    specifiers (``>=3.0,!=3.1``), or no specifier at all.
    """

    def __init__(
        self,
        specifiers: str | Iterable[Specifier] = "",
        prereleases: bool | None = None,
    ) -> None:
        """Initialize a SpecifierSet instance.

        :param specifiers:
            The string representation of a specifier or a comma-separated list of
            specifiers which will be parsed and normalized before use.
            May also be an iterable of ``Specifier`` instances, which will be used
            as is.
        :param prereleases:
            This tells the SpecifierSet if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.

        :raises InvalidSpecifier:
            If the given ``specifiers`` are not parseable than this exception will be
            raised.
        """

        if isinstance(specifiers, str):
            # Split on `,` to break each individual specifier into its own item, and
            # strip each item to remove leading/trailing whitespace.
            split_specifiers = [s.strip() for s in specifiers.split(",") if s.strip()]

            # Make each individual specifier a Specifier and save in a frozen set
            # for later.
            self._specs = frozenset(map(Specifier, split_specifiers))
        else:
            # Save the supplied specifiers in a frozen set.
            self._specs = frozenset(specifiers)

        # Store our prereleases value so we can use it later to determine if
        # we accept prereleases or not.
        self._prereleases = prereleases

    @property
    def prereleases(self) -> bool | None:
        # If we have been given an explicit prerelease modifier, then we'll
        # pass that through here.
        if self._prereleases is not None:
            return self._prereleases

        # If we don't have any specifiers, and we don't have a forced value,
        # then we'll just return None since we don't know if this should have
        # pre-releases or not.
        if not self._specs:
            return None

        # Otherwise we'll see if any of the given specifiers accept
        # prereleases, if any of them do we'll return True, otherwise False.
        return any(s.prereleases for s in self._specs)

    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        self._prereleases = value

    def __repr__(self) -> str:
        """A representation of the specifier set that shows all internal state.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> SpecifierSet('>=1.0.0,!=2.0.0')
        <SpecifierSet('!=2.0.0,>=1.0.0')>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=False)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=False)>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=True)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=True)>
        """
        pre = (
            f", prereleases={self.prereleases!r}"
            if self._prereleases is not None
            else ""
        )

        return f"<SpecifierSet({str(self)!r}{pre})>"

    def __str__(self) -> str:
        """A string representation of the specifier set that can be round-tripped.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> str(SpecifierSet(">=1.0.0,!=1.0.1"))
        '!=1.0.1,>=1.0.0'
        >>> str(SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False))
        '!=1.0.1,>=1.0.0'
        """
        return ",".join(sorted(str(s) for s in self._specs))

    def __hash__(self) -> int:
        return hash(self._specs)

    def __and__(self, other: SpecifierSet | str) -> SpecifierSet:
        """Return a SpecifierSet which is a combination of the two sets.

        :param other: The other object to combine with.

        >>> SpecifierSet(">=1.0.0,!=1.0.1") & '<=2.0.0,!=2.0.1'
        <SpecifierSet('!=1.0.1,!=2.0.1,<=2.0.0,>=1.0.0')>
        >>> SpecifierSet(">=1.0.0,!=1.0.1") & SpecifierSet('<=2.0.0,!=2.0.1')
        <SpecifierSet('!=1.0.1,!=2.0.1,<=2.0.0,>=1.0.0')>
        """
        if isinstance(other, str):
            other = SpecifierSet(other)
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        specifier = SpecifierSet()
        specifier._specs = frozenset(self._specs | other._specs)

        if self._prereleases is None and other._prereleases is not None:
            specifier._prereleases = other._prereleases
        elif self._prereleases is not None and other._prereleases is None:
            specifier._prereleases = self._prereleases
        elif self._prereleases == other._prereleases:
            specifier._prereleases = self._prereleases
        else:
            raise ValueError(
                "Cannot combine SpecifierSets with True and False prerelease overrides."
            )

        return specifier

    def __eq__(self, other: object) -> bool:
        """Whether or not the two SpecifierSet-like objects are equal.

        :param other: The other object to check against.

        The value of :attr:`prereleases` is ignored.

        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> (SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False) ==
        ...  SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True))
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == ">=1.0.0,!=1.0.1"
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0,!=1.0.2")
        False
        """
        if isinstance(other, (str, Specifier)):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._specs == other._specs

    def __len__(self) -> int:
        """Returns the number of specifiers in this specifier set."""
        return len(self._specs)

    def __iter__(self) -> Iterator[Specifier]:
        """
        Returns an iterator over all the underlying :class:`Specifier` instances
        in this specifier set.

        >>> sorted(SpecifierSet(">=1.0.0,!=1.0.1"), key=str)
        [<Specifier('!=1.0.1')>, <Specifier('>=1.0.0')>]
        """
        return iter(self._specs)

    def __contains__(self, item: UnparsedVersion) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item: The item to check for.

        This is used for the ``in`` operator and behaves the same as
        :meth:`contains` with no ``prereleases`` argument passed.

        >>> "1.2.3" in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> Version("1.2.3") in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> "1.0.1" in SpecifierSet(">=1.0.0,!=1.0.1")
        False
        >>> "1.3.0a1" in SpecifierSet(">=1.0.0,!=1.0.1")
        False
        >>> "1.3.0a1" in SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True)
        True
        """
        return self.contains(item)

    def contains(
        self,
        item: UnparsedVersion,
        prereleases: bool | None = None,
        installed: bool | None = None,
    ) -> bool:
        """Return whether or not the item is contained in this SpecifierSet.

        :param item:
            The item to check for, which can be a version string or a
            :class:`Version` instance.
        :param prereleases:
            Whether or not to match prereleases with this SpecifierSet. If set to
            ``None`` (the default), it uses :attr:`prereleases` to determine
            whether or not prereleases are allowed.

        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.2.3")
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains(Version("1.2.3"))
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.0.1")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.3.0a1")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True).contains("1.3.0a1")
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.3.0a1", prereleases=True)
        True
        """
        # Ensure that our item is a Version instance.
        if not isinstance(item, Version):
            item = Version(item)

        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # We can determine if we're going to allow pre-releases by looking to
        # see if any of the underlying items supports them. If none of them do
        # and this item is a pre-release then we do not allow it and we can
        # short circuit that here.
        # Note: This means that 1.0.dev1 would not be contained in something
        #       like >=1.0.devabc however it would be in >=1.0.debabc,>0.0.dev0
        if not prereleases and item.is_prerelease:
            return False

        if installed and item.is_prerelease:
            item = Version(item.base_version)

        # We simply dispatch to the underlying specs here to make sure that the
        # given version is contained within all of them.
        # Note: This use of all() here means that an empty set of specifiers
        #       will always return True, this is an explicit design decision.
        return all(s.contains(item, prereleases=prereleases) for s in self._specs)

    def filter(
        self, iterable: Iterable[UnparsedVersionVar], prereleases: bool | None = None
    ) -> Iterator[UnparsedVersionVar]:
        """Filter items in the given iterable, that match the specifiers in this set.

        :param iterable:
            An iterable that can contain version strings and :class:`Version` instances.
            The items in the iterable will be filtered according to the specifier.
        :param prereleases:
            Whether or not to allow prereleases in the returned iterator. If set to
            ``None`` (the default), it will be intelligently decide whether to allow
            prereleases or not (based on the :attr:`prereleases` attribute, and
            whether the only versions matching are prereleases).

        This method is smarter than just ``filter(SpecifierSet(...).contains, [...])``
        because it implements the rule from :pep:`440` that a prerelease item
        SHOULD be accepted if no other versions match the given specifier.

        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.3", "1.5a1"]))
        ['1.3']
        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.3", Version("1.4")]))
        ['1.3', <Version('1.4')>]
        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.5a1"]))
        []
        >>> list(SpecifierSet(">=1.2.3").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet(">=1.2.3", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']

        An "empty" SpecifierSet will filter items based on the presence of prerelease
        versions in the set.

        >>> list(SpecifierSet("").filter(["1.3", "1.5a1"]))
        ['1.3']
        >>> list(SpecifierSet("").filter(["1.5a1"]))
        ['1.5a1']
        >>> list(SpecifierSet("", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet("").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        """
        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # If we have any specifiers, then we want to wrap our iterable in the
        # filter method for each one, this will act as a logical AND amongst
        # each specifier.
        if self._specs:
            for spec in self._specs:
                iterable = spec.filter(iterable, prereleases=bool(prereleases))
            return iter(iterable)
        # If we do not have any specifiers, then we need to have a rough filter
        # which will filter out any pre-releases, unless there are no final
        # releases.
        else:
            filtered: list[UnparsedVersionVar] = []
            found_prereleases: list[UnparsedVersionVar] = []

            for item in iterable:
                parsed_version = _coerce_version(item)

                # Store any item which is a pre-release for later unless we've
                # already found a final version or we are accepting prereleases
                if parsed_version.is_prerelease and not prereleases:
                    if not filtered:
                        found_prereleases.append(item)
                else:
                    filtered.append(item)

            # If we've found no items except for pre-releases, then we'll go
            # ahead and use the pre-releases
            if not filtered and found_prereleases and prereleases is None:
                return iter(found_prereleases)

            return iter(filtered)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\specifiers.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""
.. testsetup::

    from pip._vendor.packaging.specifiers import Specifier, SpecifierSet, InvalidSpecifier
    from pip._vendor.packaging.version import Version
"""

from __future__ import annotations

import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, TypeVar, Union

from .utils import canonicalize_version
from .version import Version

UnparsedVersion = Union[Version, str]
UnparsedVersionVar = TypeVar("UnparsedVersionVar", bound=UnparsedVersion)
CallableOperator = Callable[[Version, str], bool]


def _coerce_version(version: UnparsedVersion) -> Version:
    if not isinstance(version, Version):
        version = Version(version)
    return version


class InvalidSpecifier(ValueError):
    """
    Raised when attempting to create a :class:`Specifier` with a specifier
    string that is invalid.

    >>> Specifier("lolwat")
    Traceback (most recent call last):
        ...
    packaging.specifiers.InvalidSpecifier: Invalid specifier: 'lolwat'
    """


class BaseSpecifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns the str representation of this Specifier-like object. This
        should be representative of the Specifier itself.
        """

    @abc.abstractmethod
    def __hash__(self) -> int:
        """
        Returns a hash value for this Specifier-like object.
        """

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Returns a boolean representing whether or not the two Specifier-like
        objects are equal.

        :param other: The other object to check against.
        """

    @property
    @abc.abstractmethod
    def prereleases(self) -> bool | None:
        """Whether or not pre-releases as a whole are allowed.

        This can be set to either ``True`` or ``False`` to explicitly enable or disable
        prereleases or it can be set to ``None`` (the default) to use default semantics.
        """

    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        """Setter for :attr:`prereleases`.

        :param value: The value to set.
        """

    @abc.abstractmethod
    def contains(self, item: str, prereleases: bool | None = None) -> bool:
        """
        Determines if the given item is contained within this specifier.
        """

    @abc.abstractmethod
    def filter(
        self, iterable: Iterable[UnparsedVersionVar], prereleases: bool | None = None
    ) -> Iterator[UnparsedVersionVar]:
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """


class Specifier(BaseSpecifier):
    """This class abstracts handling of version specifiers.

    .. tip::

        It is generally not required to instantiate this manually. You should instead
        prefer to work with :class:`SpecifierSet` instead, which can parse
        comma-separated version specifiers (which is what package metadata contains).
    """

    _operator_regex_str = r"""
        (?P<operator>(~=|==|!=|<=|>=|<|>|===))
        """
    _version_regex_str = r"""
        (?P<version>
            (?:
                # The identity operators allow for an escape hatch that will
                # do an exact string match of the version you wish to install.
                # This will not be parsed by PEP 440 and we cannot determine
                # any semantic meaning from it. This operator is discouraged
                # but included entirely as an escape hatch.
                (?<====)  # Only match for the identity operator
                \s*
                [^\s;)]*  # The arbitrary version can be just about anything,
                          # we match everything except for whitespace, a
                          # semi-colon for marker support, and a closing paren
                          # since versions can be enclosed in them.
            )
            |
            (?:
                # The (non)equality operators allow for wild card and local
                # versions to be specified so we have to define these two
                # operators separately to enable that.
                (?<===|!=)            # Only match for equals and not equals

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release

                # You cannot use a wild card and a pre-release, post-release, a dev or
                # local version together so group them with a | and make them optional.
                (?:
                    \.\*  # Wild card syntax of .*
                    |
                    (?:                                  # pre release
                        [-_\.]?
                        (alpha|beta|preview|pre|a|b|c|rc)
                        [-_\.]?
                        [0-9]*
                    )?
                    (?:                                  # post release
                        (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                    )?
                    (?:[-_\.]?dev[-_\.]?[0-9]*)?         # dev release
                    (?:\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*)? # local
                )?
            )
            |
            (?:
                # The compatible operator requires at least two digits in the
                # release segment.
                (?<=~=)               # Only match for the compatible operator

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)+   # release  (We have a + instead of a *)
                (?:                   # pre release
                    [-_\.]?
                    (alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
            |
            (?:
                # All other operators only allow a sub set of what the
                # (non)equality operators do. Specifically they do not allow
                # local versions to be specified nor do they allow the prefix
                # matching wild cards.
                (?<!==|!=|~=)         # We have special cases for these
                                      # operators so we want to make sure they
                                      # don't match here.

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
        )
        """

    _regex = re.compile(
        r"^\s*" + _operator_regex_str + _version_regex_str + r"\s*$",
        re.VERBOSE | re.IGNORECASE,
    )

    _operators = {
        "~=": "compatible",
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
        "===": "arbitrary",
    }

    def __init__(self, spec: str = "", prereleases: bool | None = None) -> None:
        """Initialize a Specifier instance.

        :param spec:
            The string representation of a specifier which will be parsed and
            normalized before use.
        :param prereleases:
            This tells the specifier if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.
        :raises InvalidSpecifier:
            If the given specifier is invalid (i.e. bad syntax).
        """
        match = self._regex.search(spec)
        if not match:
            raise InvalidSpecifier(f"Invalid specifier: {spec!r}")

        self._spec: tuple[str, str] = (
            match.group("operator").strip(),
            match.group("version").strip(),
        )

        # Store whether or not this Specifier should accept prereleases
        self._prereleases = prereleases

    # https://github.com/python/mypy/pull/13475#pullrequestreview-1079784515
    @property  # type: ignore[override]
    def prereleases(self) -> bool:
        # If there is an explicit prereleases set for this, then we'll just
        # blindly use that.
        if self._prereleases is not None:
            return self._prereleases

        # Look at all of our specifiers and determine if they are inclusive
        # operators, and if they are if they are including an explicit
        # prerelease.
        operator, version = self._spec
        if operator in ["==", ">=", "<=", "~=", "===", ">", "<"]:
            # The == specifier can include a trailing .*, if it does we
            # want to remove before parsing.
            if operator == "==" and version.endswith(".*"):
                version = version[:-2]

            # Parse the version, and if it is a pre-release than this
            # specifier allows pre-releases.
            if Version(version).is_prerelease:
                return True

        return False

    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        self._prereleases = value

    @property
    def operator(self) -> str:
        """The operator of this specifier.

        >>> Specifier("==1.2.3").operator
        '=='
        """
        return self._spec[0]

    @property
    def version(self) -> str:
        """The version of this specifier.

        >>> Specifier("==1.2.3").version
        '1.2.3'
        """
        return self._spec[1]

    def __repr__(self) -> str:
        """A representation of the Specifier that shows all internal state.

        >>> Specifier('>=1.0.0')
        <Specifier('>=1.0.0')>
        >>> Specifier('>=1.0.0', prereleases=False)
        <Specifier('>=1.0.0', prereleases=False)>
        >>> Specifier('>=1.0.0', prereleases=True)
        <Specifier('>=1.0.0', prereleases=True)>
        """
        pre = (
            f", prereleases={self.prereleases!r}"
            if self._prereleases is not None
            else ""
        )

        return f"<{self.__class__.__name__}({str(self)!r}{pre})>"

    def __str__(self) -> str:
        """A string representation of the Specifier that can be round-tripped.

        >>> str(Specifier('>=1.0.0'))
        '>=1.0.0'
        >>> str(Specifier('>=1.0.0', prereleases=False))
        '>=1.0.0'
        """
        return "{}{}".format(*self._spec)

    @property
    def _canonical_spec(self) -> tuple[str, str]:
        canonical_version = canonicalize_version(
            self._spec[1],
            strip_trailing_zero=(self._spec[0] != "~="),
        )
        return self._spec[0], canonical_version

    def __hash__(self) -> int:
        return hash(self._canonical_spec)

    def __eq__(self, other: object) -> bool:
        """Whether or not the two Specifier-like objects are equal.

        :param other: The other object to check against.

        The value of :attr:`prereleases` is ignored.

        >>> Specifier("==1.2.3") == Specifier("== 1.2.3.0")
        True
        >>> (Specifier("==1.2.3", prereleases=False) ==
        ...  Specifier("==1.2.3", prereleases=True))
        True
        >>> Specifier("==1.2.3") == "==1.2.3"
        True
        >>> Specifier("==1.2.3") == Specifier("==1.2.4")
        False
        >>> Specifier("==1.2.3") == Specifier("~=1.2.3")
        False
        """
        if isinstance(other, str):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._canonical_spec == other._canonical_spec

    def _get_operator(self, op: str) -> CallableOperator:
        operator_callable: CallableOperator = getattr(
            self, f"_compare_{self._operators[op]}"
        )
        return operator_callable

    def _compare_compatible(self, prospective: Version, spec: str) -> bool:
        # Compatible releases have an equivalent combination of >= and ==. That
        # is that ~=2.2 is equivalent to >=2.2,==2.*. This allows us to
        # implement this in terms of the other specifiers instead of
        # implementing it ourselves. The only thing we need to do is construct
        # the other specifiers.

        # We want everything but the last item in the version, but we want to
        # ignore suffix segments.
        prefix = _version_join(
            list(itertools.takewhile(_is_not_suffix, _version_split(spec)))[:-1]
        )

        # Add the prefix notation to the end of our string
        prefix += ".*"

        return self._get_operator(">=")(prospective, spec) and self._get_operator("==")(
            prospective, prefix
        )

    def _compare_equal(self, prospective: Version, spec: str) -> bool:
        # We need special logic to handle prefix matching
        if spec.endswith(".*"):
            # In the case of prefix matching we want to ignore local segment.
            normalized_prospective = canonicalize_version(
                prospective.public, strip_trailing_zero=False
            )
            # Get the normalized version string ignoring the trailing .*
            normalized_spec = canonicalize_version(spec[:-2], strip_trailing_zero=False)
            # Split the spec out by bangs and dots, and pretend that there is
            # an implicit dot in between a release segment and a pre-release segment.
            split_spec = _version_split(normalized_spec)

            # Split the prospective version out by bangs and dots, and pretend
            # that there is an implicit dot in between a release segment and
            # a pre-release segment.
            split_prospective = _version_split(normalized_prospective)

            # 0-pad the prospective version before shortening it to get the correct
            # shortened version.
            padded_prospective, _ = _pad_version(split_prospective, split_spec)

            # Shorten the prospective version to be the same length as the spec
            # so that we can determine if the specifier is a prefix of the
            # prospective version or not.
            shortened_prospective = padded_prospective[: len(split_spec)]

            return shortened_prospective == split_spec
        else:
            # Convert our spec string into a Version
            spec_version = Version(spec)

            # If the specifier does not have a local segment, then we want to
            # act as if the prospective version also does not have a local
            # segment.
            if not spec_version.local:
                prospective = Version(prospective.public)

            return prospective == spec_version

    def _compare_not_equal(self, prospective: Version, spec: str) -> bool:
        return not self._compare_equal(prospective, spec)

    def _compare_less_than_equal(self, prospective: Version, spec: str) -> bool:
        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return Version(prospective.public) <= Version(spec)

    def _compare_greater_than_equal(self, prospective: Version, spec: str) -> bool:
        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return Version(prospective.public) >= Version(spec)

    def _compare_less_than(self, prospective: Version, spec_str: str) -> bool:
        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec_str)

        # Check to see if the prospective version is less than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective < spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a pre-release version, that we do not accept pre-release
        # versions for the version mentioned in the specifier (e.g. <3.1 should
        # not match 3.1.dev0, but should match 3.0.dev0).
        if not spec.is_prerelease and prospective.is_prerelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # less than the spec version *and* it's not a pre-release of the same
        # version in the spec.
        return True

    def _compare_greater_than(self, prospective: Version, spec_str: str) -> bool:
        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec_str)

        # Check to see if the prospective version is greater than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective > spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a post-release version, that we do not accept
        # post-release versions for the version mentioned in the specifier
        # (e.g. >3.1 should not match 3.0.post0, but should match 3.2.post0).
        if not spec.is_postrelease and prospective.is_postrelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # Ensure that we do not allow a local version of the version mentioned
        # in the specifier, which is technically greater than, to match.
        if prospective.local is not None:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # greater than the spec version *and* it's not a pre-release of the
        # same version in the spec.
        return True

    def _compare_arbitrary(self, prospective: Version, spec: str) -> bool:
        return str(prospective).lower() == str(spec).lower()

    def __contains__(self, item: str | Version) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item: The item to check for.

        This is used for the ``in`` operator and behaves the same as
        :meth:`contains` with no ``prereleases`` argument passed.

        >>> "1.2.3" in Specifier(">=1.2.3")
        True
        >>> Version("1.2.3") in Specifier(">=1.2.3")
        True
        >>> "1.0.0" in Specifier(">=1.2.3")
        False
        >>> "1.3.0a1" in Specifier(">=1.2.3")
        False
        >>> "1.3.0a1" in Specifier(">=1.2.3", prereleases=True)
        True
        """
        return self.contains(item)

    def contains(self, item: UnparsedVersion, prereleases: bool | None = None) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item:
            The item to check for, which can be a version string or a
            :class:`Version` instance.
        :param prereleases:
            Whether or not to match prereleases with this Specifier. If set to
            ``None`` (the default), it uses :attr:`prereleases` to determine
            whether or not prereleases are allowed.

        >>> Specifier(">=1.2.3").contains("1.2.3")
        True
        >>> Specifier(">=1.2.3").contains(Version("1.2.3"))
        True
        >>> Specifier(">=1.2.3").contains("1.0.0")
        False
        >>> Specifier(">=1.2.3").contains("1.3.0a1")
        False
        >>> Specifier(">=1.2.3", prereleases=True).contains("1.3.0a1")
        True
        >>> Specifier(">=1.2.3").contains("1.3.0a1", prereleases=True)
        True
        """

        # Determine if prereleases are to be allowed or not.
        if prereleases is None:
            prereleases = self.prereleases

        # Normalize item to a Version, this allows us to have a shortcut for
        # "2.0" in Specifier(">=2")
        normalized_item = _coerce_version(item)

        # Determine if we should be supporting prereleases in this specifier
        # or not, if we do not support prereleases than we can short circuit
        # logic if this version is a prereleases.
        if normalized_item.is_prerelease and not prereleases:
            return False

        # Actually do the comparison to determine if this item is contained
        # within this Specifier or not.
        operator_callable: CallableOperator = self._get_operator(self.operator)
        return operator_callable(normalized_item, self.version)

    def filter(
        self, iterable: Iterable[UnparsedVersionVar], prereleases: bool | None = None
    ) -> Iterator[UnparsedVersionVar]:
        """Filter items in the given iterable, that match the specifier.

        :param iterable:
            An iterable that can contain version strings and :class:`Version` instances.
            The items in the iterable will be filtered according to the specifier.
        :param prereleases:
            Whether or not to allow prereleases in the returned iterator. If set to
            ``None`` (the default), it will be intelligently decide whether to allow
            prereleases or not (based on the :attr:`prereleases` attribute, and
            whether the only versions matching are prereleases).

        This method is smarter than just ``filter(Specifier().contains, [...])``
        because it implements the rule from :pep:`440` that a prerelease item
        SHOULD be accepted if no other versions match the given specifier.

        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.3", "1.5a1"]))
        ['1.3']
        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.2.3", "1.3", Version("1.4")]))
        ['1.2.3', '1.3', <Version('1.4')>]
        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.5a1"]))
        ['1.5a1']
        >>> list(Specifier(">=1.2.3").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        >>> list(Specifier(">=1.2.3", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        """

        yielded = False
        found_prereleases = []

        kw = {"prereleases": prereleases if prereleases is not None else True}

        # Attempt to iterate over all the values in the iterable and if any of
        # them match, yield them.
        for version in iterable:
            parsed_version = _coerce_version(version)

            if self.contains(parsed_version, **kw):
                # If our version is a prerelease, and we were not set to allow
                # prereleases, then we'll store it for later in case nothing
                # else matches this specifier.
                if parsed_version.is_prerelease and not (
                    prereleases or self.prereleases
                ):
                    found_prereleases.append(version)
                # Either this is not a prerelease, or we should have been
                # accepting prereleases from the beginning.
                else:
                    yielded = True
                    yield version

        # Now that we've iterated over everything, determine if we've yielded
        # any values, and if we have not and we have any prereleases stored up
        # then we will go ahead and yield the prereleases.
        if not yielded and found_prereleases:
            for version in found_prereleases:
                yield version


_prefix_regex = re.compile(r"^([0-9]+)((?:a|b|c|rc)[0-9]+)$")


def _version_split(version: str) -> list[str]:
    """Split version into components.

    The split components are intended for version comparison. The logic does
    not attempt to retain the original version string, so joining the
    components back with :func:`_version_join` may not produce the original
    version string.
    """
    result: list[str] = []

    epoch, _, rest = version.rpartition("!")
    result.append(epoch or "0")

    for item in rest.split("."):
        match = _prefix_regex.search(item)
        if match:
            result.extend(match.groups())
        else:
            result.append(item)
    return result


def _version_join(components: list[str]) -> str:
    """Join split version components into a version string.

    This function assumes the input came from :func:`_version_split`, where the
    first component must be the epoch (either empty or numeric), and all other
    components numeric.
    """
    epoch, *rest = components
    return f"{epoch}!{'.'.join(rest)}"


def _is_not_suffix(segment: str) -> bool:
    return not any(
        segment.startswith(prefix) for prefix in ("dev", "a", "b", "rc", "post")
    )


def _pad_version(left: list[str], right: list[str]) -> tuple[list[str], list[str]]:
    left_split, right_split = [], []

    # Get the release segment of our versions
    left_split.append(list(itertools.takewhile(lambda x: x.isdigit(), left)))
    right_split.append(list(itertools.takewhile(lambda x: x.isdigit(), right)))

    # Get the rest of our versions
    left_split.append(left[len(left_split[0]) :])
    right_split.append(right[len(right_split[0]) :])

    # Insert our padding
    left_split.insert(1, ["0"] * max(0, len(right_split[0]) - len(left_split[0])))
    right_split.insert(1, ["0"] * max(0, len(left_split[0]) - len(right_split[0])))

    return (
        list(itertools.chain.from_iterable(left_split)),
        list(itertools.chain.from_iterable(right_split)),
    )


class SpecifierSet(BaseSpecifier):
    """This class abstracts handling of a set of version specifiers.

    It can be passed a single specifier (``>=3.0``), a comma-separated list of
    specifiers (``>=3.0,!=3.1``), or no specifier at all.
    """

    def __init__(
        self,
        specifiers: str | Iterable[Specifier] = "",
        prereleases: bool | None = None,
    ) -> None:
        """Initialize a SpecifierSet instance.

        :param specifiers:
            The string representation of a specifier or a comma-separated list of
            specifiers which will be parsed and normalized before use.
            May also be an iterable of ``Specifier`` instances, which will be used
            as is.
        :param prereleases:
            This tells the SpecifierSet if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.

        :raises InvalidSpecifier:
            If the given ``specifiers`` are not parseable than this exception will be
            raised.
        """

        if isinstance(specifiers, str):
            # Split on `,` to break each individual specifier into its own item, and
            # strip each item to remove leading/trailing whitespace.
            split_specifiers = [s.strip() for s in specifiers.split(",") if s.strip()]

            # Make each individual specifier a Specifier and save in a frozen set
            # for later.
            self._specs = frozenset(map(Specifier, split_specifiers))
        else:
            # Save the supplied specifiers in a frozen set.
            self._specs = frozenset(specifiers)

        # Store our prereleases value so we can use it later to determine if
        # we accept prereleases or not.
        self._prereleases = prereleases

    @property
    def prereleases(self) -> bool | None:
        # If we have been given an explicit prerelease modifier, then we'll
        # pass that through here.
        if self._prereleases is not None:
            return self._prereleases

        # If we don't have any specifiers, and we don't have a forced value,
        # then we'll just return None since we don't know if this should have
        # pre-releases or not.
        if not self._specs:
            return None

        # Otherwise we'll see if any of the given specifiers accept
        # prereleases, if any of them do we'll return True, otherwise False.
        return any(s.prereleases for s in self._specs)

    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        self._prereleases = value

    def __repr__(self) -> str:
        """A representation of the specifier set that shows all internal state.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> SpecifierSet('>=1.0.0,!=2.0.0')
        <SpecifierSet('!=2.0.0,>=1.0.0')>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=False)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=False)>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=True)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=True)>
        """
        pre = (
            f", prereleases={self.prereleases!r}"
            if self._prereleases is not None
            else ""
        )

        return f"<SpecifierSet({str(self)!r}{pre})>"

    def __str__(self) -> str:
        """A string representation of the specifier set that can be round-tripped.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> str(SpecifierSet(">=1.0.0,!=1.0.1"))
        '!=1.0.1,>=1.0.0'
        >>> str(SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False))
        '!=1.0.1,>=1.0.0'
        """
        return ",".join(sorted(str(s) for s in self._specs))

    def __hash__(self) -> int:
        return hash(self._specs)

    def __and__(self, other: SpecifierSet | str) -> SpecifierSet:
        """Return a SpecifierSet which is a combination of the two sets.

        :param other: The other object to combine with.

        >>> SpecifierSet(">=1.0.0,!=1.0.1") & '<=2.0.0,!=2.0.1'
        <SpecifierSet('!=1.0.1,!=2.0.1,<=2.0.0,>=1.0.0')>
        >>> SpecifierSet(">=1.0.0,!=1.0.1") & SpecifierSet('<=2.0.0,!=2.0.1')
        <SpecifierSet('!=1.0.1,!=2.0.1,<=2.0.0,>=1.0.0')>
        """
        if isinstance(other, str):
            other = SpecifierSet(other)
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        specifier = SpecifierSet()
        specifier._specs = frozenset(self._specs | other._specs)

        if self._prereleases is None and other._prereleases is not None:
            specifier._prereleases = other._prereleases
        elif self._prereleases is not None and other._prereleases is None:
            specifier._prereleases = self._prereleases
        elif self._prereleases == other._prereleases:
            specifier._prereleases = self._prereleases
        else:
            raise ValueError(
                "Cannot combine SpecifierSets with True and False prerelease overrides."
            )

        return specifier

    def __eq__(self, other: object) -> bool:
        """Whether or not the two SpecifierSet-like objects are equal.

        :param other: The other object to check against.

        The value of :attr:`prereleases` is ignored.

        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> (SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False) ==
        ...  SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True))
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == ">=1.0.0,!=1.0.1"
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0,!=1.0.2")
        False
        """
        if isinstance(other, (str, Specifier)):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._specs == other._specs

    def __len__(self) -> int:
        """Returns the number of specifiers in this specifier set."""
        return len(self._specs)

    def __iter__(self) -> Iterator[Specifier]:
        """
        Returns an iterator over all the underlying :class:`Specifier` instances
        in this specifier set.

        >>> sorted(SpecifierSet(">=1.0.0,!=1.0.1"), key=str)
        [<Specifier('!=1.0.1')>, <Specifier('>=1.0.0')>]
        """
        return iter(self._specs)

    def __contains__(self, item: UnparsedVersion) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item: The item to check for.

        This is used for the ``in`` operator and behaves the same as
        :meth:`contains` with no ``prereleases`` argument passed.

        >>> "1.2.3" in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> Version("1.2.3") in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> "1.0.1" in SpecifierSet(">=1.0.0,!=1.0.1")
        False
        >>> "1.3.0a1" in SpecifierSet(">=1.0.0,!=1.0.1")
        False
        >>> "1.3.0a1" in SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True)
        True
        """
        return self.contains(item)

    def contains(
        self,
        item: UnparsedVersion,
        prereleases: bool | None = None,
        installed: bool | None = None,
    ) -> bool:
        """Return whether or not the item is contained in this SpecifierSet.

        :param item:
            The item to check for, which can be a version string or a
            :class:`Version` instance.
        :param prereleases:
            Whether or not to match prereleases with this SpecifierSet. If set to
            ``None`` (the default), it uses :attr:`prereleases` to determine
            whether or not prereleases are allowed.

        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.2.3")
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains(Version("1.2.3"))
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.0.1")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.3.0a1")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True).contains("1.3.0a1")
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.3.0a1", prereleases=True)
        True
        """
        # Ensure that our item is a Version instance.
        if not isinstance(item, Version):
            item = Version(item)

        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # We can determine if we're going to allow pre-releases by looking to
        # see if any of the underlying items supports them. If none of them do
        # and this item is a pre-release then we do not allow it and we can
        # short circuit that here.
        # Note: This means that 1.0.dev1 would not be contained in something
        #       like >=1.0.devabc however it would be in >=1.0.debabc,>0.0.dev0
        if not prereleases and item.is_prerelease:
            return False

        if installed and item.is_prerelease:
            item = Version(item.base_version)

        # We simply dispatch to the underlying specs here to make sure that the
        # given version is contained within all of them.
        # Note: This use of all() here means that an empty set of specifiers
        #       will always return True, this is an explicit design decision.
        return all(s.contains(item, prereleases=prereleases) for s in self._specs)

    def filter(
        self, iterable: Iterable[UnparsedVersionVar], prereleases: bool | None = None
    ) -> Iterator[UnparsedVersionVar]:
        """Filter items in the given iterable, that match the specifiers in this set.

        :param iterable:
            An iterable that can contain version strings and :class:`Version` instances.
            The items in the iterable will be filtered according to the specifier.
        :param prereleases:
            Whether or not to allow prereleases in the returned iterator. If set to
            ``None`` (the default), it will be intelligently decide whether to allow
            prereleases or not (based on the :attr:`prereleases` attribute, and
            whether the only versions matching are prereleases).

        This method is smarter than just ``filter(SpecifierSet(...).contains, [...])``
        because it implements the rule from :pep:`440` that a prerelease item
        SHOULD be accepted if no other versions match the given specifier.

        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.3", "1.5a1"]))
        ['1.3']
        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.3", Version("1.4")]))
        ['1.3', <Version('1.4')>]
        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.5a1"]))
        []
        >>> list(SpecifierSet(">=1.2.3").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet(">=1.2.3", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']

        An "empty" SpecifierSet will filter items based on the presence of prerelease
        versions in the set.

        >>> list(SpecifierSet("").filter(["1.3", "1.5a1"]))
        ['1.3']
        >>> list(SpecifierSet("").filter(["1.5a1"]))
        ['1.5a1']
        >>> list(SpecifierSet("", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet("").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        """
        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # If we have any specifiers, then we want to wrap our iterable in the
        # filter method for each one, this will act as a logical AND amongst
        # each specifier.
        if self._specs:
            for spec in self._specs:
                iterable = spec.filter(iterable, prereleases=bool(prereleases))
            return iter(iterable)
        # If we do not have any specifiers, then we need to have a rough filter
        # which will filter out any pre-releases, unless there are no final
        # releases.
        else:
            filtered: list[UnparsedVersionVar] = []
            found_prereleases: list[UnparsedVersionVar] = []

            for item in iterable:
                parsed_version = _coerce_version(item)

                # Store any item which is a pre-release for later unless we've
                # already found a final version or we are accepting prereleases
                if parsed_version.is_prerelease and not prereleases:
                    if not filtered:
                        found_prereleases.append(item)
                else:
                    filtered.append(item)

            # If we've found no items except for pre-releases, then we'll go
            # ahead and use the pre-releases
            if not filtered and found_prereleases and prereleases is None:
                return iter(found_prereleases)

            return iter(filtered)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\polyhedron.py ===
from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import (minlex, unflatten, flatten)
from sympy.utilities.misc import as_int

rmul = Perm.rmul


class Polyhedron(Basic):
    """
    Represents the polyhedral symmetry group (PSG).

    Explanation
    ===========

    The PSG is one of the symmetry groups of the Platonic solids.
    There are three polyhedral groups: the tetrahedral group
    of order 12, the octahedral group of order 24, and the
    icosahedral group of order 60.

    All doctests have been given in the docstring of the
    constructor of the object.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PolyhedralGroup.html

    """
    _edges = None

    def __new__(cls, corners, faces=(), pgroup=()):
        """
        The constructor of the Polyhedron group object.

        Explanation
        ===========

        It takes up to three parameters: the corners, faces, and
        allowed transformations.

        The corners/vertices are entered as a list of arbitrary
        expressions that are used to identify each vertex.

        The faces are entered as a list of tuples of indices; a tuple
        of indices identifies the vertices which define the face. They
        should be entered in a cw or ccw order; they will be standardized
        by reversal and rotation to be give the lowest lexical ordering.
        If no faces are given then no edges will be computed.

            >>> from sympy.combinatorics.polyhedron import Polyhedron
            >>> Polyhedron(list('abc'), [(1, 2, 0)]).faces
            {(0, 1, 2)}
            >>> Polyhedron(list('abc'), [(1, 0, 2)]).faces
            {(0, 1, 2)}

        The allowed transformations are entered as allowable permutations
        of the vertices for the polyhedron. Instance of Permutations
        (as with faces) should refer to the supplied vertices by index.
        These permutation are stored as a PermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy import init_printing
        >>> from sympy.abc import w, x, y, z
        >>> init_printing(pretty_print=False, perm_cyclic=False)

        Here we construct the Polyhedron object for a tetrahedron.

        >>> corners = [w, x, y, z]
        >>> faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3)]

        Next, allowed transformations of the polyhedron must be given. This
        is given as permutations of vertices.

        Although the vertices of a tetrahedron can be numbered in 24 (4!)
        different ways, there are only 12 different orientations for a
        physical tetrahedron. The following permutations, applied once or
        twice, will generate all 12 of the orientations. (The identity
        permutation, Permutation(range(4)), is not included since it does
        not change the orientation of the vertices.)

        >>> pgroup = [Permutation([[0, 1, 2], [3]]), \
                      Permutation([[0, 1, 3], [2]]), \
                      Permutation([[0, 2, 3], [1]]), \
                      Permutation([[1, 2, 3], [0]]), \
                      Permutation([[0, 1], [2, 3]]), \
                      Permutation([[0, 2], [1, 3]]), \
                      Permutation([[0, 3], [1, 2]])]

        The Polyhedron is now constructed and demonstrated:

        >>> tetra = Polyhedron(corners, faces, pgroup)
        >>> tetra.size
        4
        >>> tetra.edges
        {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        >>> tetra.corners
        (w, x, y, z)

        It can be rotated with an arbitrary permutation of vertices, e.g.
        the following permutation is not in the pgroup:

        >>> tetra.rotate(Permutation([0, 1, 3, 2]))
        >>> tetra.corners
        (w, x, z, y)

        An allowed permutation of the vertices can be constructed by
        repeatedly applying permutations from the pgroup to the vertices.
        Here is a demonstration that applying p and p**2 for every p in
        pgroup generates all the orientations of a tetrahedron and no others:

        >>> all = ( (w, x, y, z), \
                    (x, y, w, z), \
                    (y, w, x, z), \
                    (w, z, x, y), \
                    (z, w, y, x), \
                    (w, y, z, x), \
                    (y, z, w, x), \
                    (x, z, y, w), \
                    (z, y, x, w), \
                    (y, x, z, w), \
                    (x, w, z, y), \
                    (z, x, w, y) )

        >>> got = []
        >>> for p in (pgroup + [p**2 for p in pgroup]):
        ...     h = Polyhedron(corners)
        ...     h.rotate(p)
        ...     got.append(h.corners)
        ...
        >>> set(got) == set(all)
        True

        The make_perm method of a PermutationGroup will randomly pick
        permutations, multiply them together, and return the permutation that
        can be applied to the polyhedron to give the orientation produced
        by those individual permutations.

        Here, 3 permutations are used:

        >>> tetra.pgroup.make_perm(3) # doctest: +SKIP
        Permutation([0, 3, 1, 2])

        To select the permutations that should be used, supply a list
        of indices to the permutations in pgroup in the order they should
        be applied:

        >>> use = [0, 0, 2]
        >>> p002 = tetra.pgroup.make_perm(3, use)
        >>> p002
        Permutation([1, 0, 3, 2])


        Apply them one at a time:

        >>> tetra.reset()
        >>> for i in use:
        ...     tetra.rotate(pgroup[i])
        ...
        >>> tetra.vertices
        (x, w, z, y)
        >>> sequentially = tetra.vertices

        Apply the composite permutation:

        >>> tetra.reset()
        >>> tetra.rotate(p002)
        >>> tetra.corners
        (x, w, z, y)
        >>> tetra.corners in all and tetra.corners == sequentially
        True

        Notes
        =====

        Defining permutation groups
        ---------------------------

        It is not necessary to enter any permutations, nor is necessary to
        enter a complete set of transformations. In fact, for a polyhedron,
        all configurations can be constructed from just two permutations.
        For example, the orientations of a tetrahedron can be generated from
        an axis passing through a vertex and face and another axis passing
        through a different vertex or from an axis passing through the
        midpoints of two edges opposite of each other.

        For simplicity of presentation, consider a square --
        not a cube -- with vertices 1, 2, 3, and 4:

        1-----2  We could think of axes of rotation being:
        |     |  1) through the face
        |     |  2) from midpoint 1-2 to 3-4 or 1-3 to 2-4
        3-----4  3) lines 1-4 or 2-3


        To determine how to write the permutations, imagine 4 cameras,
        one at each corner, labeled A-D:

        A       B          A       B
         1-----2            1-----3             vertex index:
         |     |            |     |                 1   0
         |     |            |     |                 2   1
         3-----4            2-----4                 3   2
        C       D          C       D                4   3

        original           after rotation
                           along 1-4

        A diagonal and a face axis will be chosen for the "permutation group"
        from which any orientation can be constructed.

        >>> pgroup = []

        Imagine a clockwise rotation when viewing 1-4 from camera A. The new
        orientation is (in camera-order): 1, 3, 2, 4 so the permutation is
        given using the *indices* of the vertices as:

        >>> pgroup.append(Permutation((0, 2, 1, 3)))

        Now imagine rotating clockwise when looking down an axis entering the
        center of the square as viewed. The new camera-order would be
        3, 1, 4, 2 so the permutation is (using indices):

        >>> pgroup.append(Permutation((2, 0, 3, 1)))

        The square can now be constructed:
            ** use real-world labels for the vertices, entering them in
               camera order
            ** for the faces we use zero-based indices of the vertices
               in *edge-order* as the face is traversed; neither the
               direction nor the starting point matter -- the faces are
               only used to define edges (if so desired).

        >>> square = Polyhedron((1, 2, 3, 4), [(0, 1, 3, 2)], pgroup)

        To rotate the square with a single permutation we can do:

        >>> square.rotate(square.pgroup[0])
        >>> square.corners
        (1, 3, 2, 4)

        To use more than one permutation (or to use one permutation more
        than once) it is more convenient to use the make_perm method:

        >>> p011 = square.pgroup.make_perm([0, 1, 1]) # diag flip + 2 rotations
        >>> square.reset() # return to initial orientation
        >>> square.rotate(p011)
        >>> square.corners
        (4, 2, 3, 1)

        Thinking outside the box
        ------------------------

        Although the Polyhedron object has a direct physical meaning, it
        actually has broader application. In the most general sense it is
        just a decorated PermutationGroup, allowing one to connect the
        permutations to something physical. For example, a Rubik's cube is
        not a proper polyhedron, but the Polyhedron class can be used to
        represent it in a way that helps to visualize the Rubik's cube.

        >>> from sympy import flatten, unflatten, symbols
        >>> from sympy.combinatorics import RubikGroup
        >>> facelets = flatten([symbols(s+'1:5') for s in 'UFRBLD'])
        >>> def show():
        ...     pairs = unflatten(r2.corners, 2)
        ...     print(pairs[::2])
        ...     print(pairs[1::2])
        ...
        >>> r2 = Polyhedron(facelets, pgroup=RubikGroup(2))
        >>> show()
        [(U1, U2), (F1, F2), (R1, R2), (B1, B2), (L1, L2), (D1, D2)]
        [(U3, U4), (F3, F4), (R3, R4), (B3, B4), (L3, L4), (D3, D4)]
        >>> r2.rotate(0) # cw rotation of F
        >>> show()
        [(U1, U2), (F3, F1), (U3, R2), (B1, B2), (L1, D1), (R3, R1)]
        [(L4, L2), (F4, F2), (U4, R4), (B3, B4), (L3, D2), (D3, D4)]

        Predefined Polyhedra
        ====================

        For convenience, the vertices and faces are defined for the following
        standard solids along with a permutation group for transformations.
        When the polyhedron is oriented as indicated below, the vertices in
        a given horizontal plane are numbered in ccw direction, starting from
        the vertex that will give the lowest indices in a given face. (In the
        net of the vertices, indices preceded by "-" indicate replication of
        the lhs index in the net.)

        tetrahedron, tetrahedron_faces
        ------------------------------

            4 vertices (vertex up) net:

                 0 0-0
                1 2 3-1

            4 faces:

            (0, 1, 2) (0, 2, 3) (0, 3, 1) (1, 2, 3)

        cube, cube_faces
        ----------------

            8 vertices (face up) net:

                0 1 2 3-0
                4 5 6 7-4

            6 faces:

            (0, 1, 2, 3)
            (0, 1, 5, 4) (1, 2, 6, 5) (2, 3, 7, 6) (0, 3, 7, 4)
            (4, 5, 6, 7)

        octahedron, octahedron_faces
        ----------------------------

            6 vertices (vertex up) net:

                 0 0 0-0
                1 2 3 4-1
                 5 5 5-5

            8 faces:

            (0, 1, 2) (0, 2, 3) (0, 3, 4) (0, 1, 4)
            (1, 2, 5) (2, 3, 5) (3, 4, 5) (1, 4, 5)

        dodecahedron, dodecahedron_faces
        --------------------------------

            20 vertices (vertex up) net:

                  0  1  2  3  4 -0
                  5  6  7  8  9 -5
                14 10 11 12 13-14
                15 16 17 18 19-15

            12 faces:

            (0, 1, 2, 3, 4) (0, 1, 6, 10, 5) (1, 2, 7, 11, 6)
            (2, 3, 8, 12, 7) (3, 4, 9, 13, 8) (0, 4, 9, 14, 5)
            (5, 10, 16, 15, 14) (6, 10, 16, 17, 11) (7, 11, 17, 18, 12)
            (8, 12, 18, 19, 13) (9, 13, 19, 15, 14)(15, 16, 17, 18, 19)

        icosahedron, icosahedron_faces
        ------------------------------

            12 vertices (face up) net:

                 0  0  0  0 -0
                1  2  3  4  5 -1
                 6  7  8  9  10 -6
                  11 11 11 11 -11

            20 faces:

            (0, 1, 2) (0, 2, 3) (0, 3, 4)
            (0, 4, 5) (0, 1, 5) (1, 2, 6)
            (2, 3, 7) (3, 4, 8) (4, 5, 9)
            (1, 5, 10) (2, 6, 7) (3, 7, 8)
            (4, 8, 9) (5, 9, 10) (1, 6, 10)
            (6, 7, 11) (7, 8, 11) (8, 9, 11)
            (9, 10, 11) (6, 10, 11)

        >>> from sympy.combinatorics.polyhedron import cube
        >>> cube.edges
        {(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)}

        If you want to use letters or other names for the corners you
        can still use the pre-calculated faces:

        >>> corners = list('abcdefgh')
        >>> Polyhedron(corners, cube.faces).corners
        (a, b, c, d, e, f, g, h)

        References
        ==========

        .. [1] www.ocf.berkeley.edu/~wwu/articles/platonicsolids.pdf

        """
        faces = [minlex(f, directed=False, key=default_sort_key) for f in faces]
        corners, faces, pgroup = args = \
            [Tuple(*a) for a in (corners, faces, pgroup)]
        obj = Basic.__new__(cls, *args)
        obj._corners = tuple(corners)  # in order given
        obj._faces = FiniteSet(*faces)
        if pgroup and pgroup[0].size != len(corners):
            raise ValueError("Permutation size unequal to number of corners.")
        # use the identity permutation if none are given
        obj._pgroup = PermutationGroup(
            pgroup or [Perm(range(len(corners)))] )
        return obj

    @property
    def corners(self):
        """
        Get the corners of the Polyhedron.

        The method ``vertices`` is an alias for ``corners``.

        Examples
        ========

        >>> from sympy.combinatorics import Polyhedron
        >>> from sympy.abc import a, b, c, d
        >>> p = Polyhedron(list('abcd'))
        >>> p.corners == p.vertices == (a, b, c, d)
        True

        See Also
        ========

        array_form, cyclic_form
        """
        return self._corners
    vertices = corners

    @property
    def array_form(self):
        """Return the indices of the corners.

        The indices are given relative to the original position of corners.

        Examples
        ========

        >>> from sympy.combinatorics.polyhedron import tetrahedron
        >>> tetrahedron = tetrahedron.copy()
        >>> tetrahedron.array_form
        [0, 1, 2, 3]

        >>> tetrahedron.rotate(0)
        >>> tetrahedron.array_form
        [0, 2, 3, 1]
        >>> tetrahedron.pgroup[0].array_form
        [0, 2, 3, 1]

        See Also
        ========

        corners, cyclic_form
        """
        corners = list(self.args[0])
        return [corners.index(c) for c in self.corners]

    @property
    def cyclic_form(self):
        """Return the indices of the corners in cyclic notation.

        The indices are given relative to the original position of corners.

        See Also
        ========

        corners, array_form
        """
        return Perm._af_new(self.array_form).cyclic_form

    @property
    def size(self):
        """
        Get the number of corners of the Polyhedron.
        """
        return len(self._corners)

    @property
    def faces(self):
        """
        Get the faces of the Polyhedron.
        """
        return self._faces

    @property
    def pgroup(self):
        """
        Get the permutations of the Polyhedron.
        """
        return self._pgroup

    @property
    def edges(self):
        """
        Given the faces of the polyhedra we can get the edges.

        Examples
        ========

        >>> from sympy.combinatorics import Polyhedron
        >>> from sympy.abc import a, b, c
        >>> corners = (a, b, c)
        >>> faces = [(0, 1, 2)]
        >>> Polyhedron(corners, faces).edges
        {(0, 1), (0, 2), (1, 2)}

        """
        if self._edges is None:
            output = set()
            for face in self.faces:
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[i - 1]]))
                    output.add(edge)
            self._edges = FiniteSet(*output)
        return self._edges

    def rotate(self, perm):
        """
        Apply a permutation to the polyhedron *in place*. The permutation
        may be given as a Permutation instance or an integer indicating
        which permutation from pgroup of the Polyhedron should be
        applied.

        This is an operation that is analogous to rotation about
        an axis by a fixed increment.

        Notes
        =====

        When a Permutation is applied, no check is done to see if that
        is a valid permutation for the Polyhedron. For example, a cube
        could be given a permutation which effectively swaps only 2
        vertices. A valid permutation (that rotates the object in a
        physical way) will be obtained if one only uses
        permutations from the ``pgroup`` of the Polyhedron. On the other
        hand, allowing arbitrary rotations (applications of permutations)
        gives a way to follow named elements rather than indices since
        Polyhedron allows vertices to be named while Permutation works
        only with indices.

        Examples
        ========

        >>> from sympy.combinatorics import Polyhedron, Permutation
        >>> from sympy.combinatorics.polyhedron import cube
        >>> cube = cube.copy()
        >>> cube.corners
        (0, 1, 2, 3, 4, 5, 6, 7)
        >>> cube.rotate(0)
        >>> cube.corners
        (1, 2, 3, 0, 5, 6, 7, 4)

        A non-physical "rotation" that is not prohibited by this method:

        >>> cube.reset()
        >>> cube.rotate(Permutation([[1, 2]], size=8))
        >>> cube.corners
        (0, 2, 1, 3, 4, 5, 6, 7)

        Polyhedron can be used to follow elements of set that are
        identified by letters instead of integers:

        >>> shadow = h5 = Polyhedron(list('abcde'))
        >>> p = Permutation([3, 0, 1, 2, 4])
        >>> h5.rotate(p)
        >>> h5.corners
        (d, a, b, c, e)
        >>> _ == shadow.corners
        True
        >>> copy = h5.copy()
        >>> h5.rotate(p)
        >>> h5.corners == copy.corners
        False
        """
        if not isinstance(perm, Perm):
            perm = self.pgroup[perm]
            # and we know it's valid
        else:
            if perm.size != self.size:
                raise ValueError('Polyhedron and Permutation sizes differ.')
        a = perm.array_form
        corners = [self.corners[a[i]] for i in range(len(self.corners))]
        self._corners = tuple(corners)

    def reset(self):
        """Return corners to their original positions.

        Examples
        ========

        >>> from sympy.combinatorics.polyhedron import tetrahedron as T
        >>> T = T.copy()
        >>> T.corners
        (0, 1, 2, 3)
        >>> T.rotate(0)
        >>> T.corners
        (0, 2, 3, 1)
        >>> T.reset()
        >>> T.corners
        (0, 1, 2, 3)
        """
        self._corners = self.args[0]


def _pgroup_calcs():
    """Return the permutation groups for each of the polyhedra and the face
    definitions: tetrahedron, cube, octahedron, dodecahedron, icosahedron,
    tetrahedron_faces, cube_faces, octahedron_faces, dodecahedron_faces,
    icosahedron_faces

    Explanation
    ===========

    (This author did not find and did not know of a better way to do it though
    there likely is such a way.)

    Although only 2 permutations are needed for a polyhedron in order to
    generate all the possible orientations, a group of permutations is
    provided instead. A set of permutations is called a "group" if::

    a*b = c (for any pair of permutations in the group, a and b, their
    product, c, is in the group)

    a*(b*c) = (a*b)*c (for any 3 permutations in the group associativity holds)

    there is an identity permutation, I, such that I*a = a*I for all elements
    in the group

    a*b = I (the inverse of each permutation is also in the group)

    None of the polyhedron groups defined follow these definitions of a group.
    Instead, they are selected to contain those permutations whose powers
    alone will construct all orientations of the polyhedron, i.e. for
    permutations ``a``, ``b``, etc... in the group, ``a, a**2, ..., a**o_a``,
    ``b, b**2, ..., b**o_b``, etc... (where ``o_i`` is the order of
    permutation ``i``) generate all permutations of the polyhedron instead of
    mixed products like ``a*b``, ``a*b**2``, etc....

    Note that for a polyhedron with n vertices, the valid permutations of the
    vertices exclude those that do not maintain its faces. e.g. the
    permutation BCDE of a square's four corners, ABCD, is a valid
    permutation while CBDE is not (because this would twist the square).

    Examples
    ========

    The is_group checks for: closure, the presence of the Identity permutation,
    and the presence of the inverse for each of the elements in the group. This
    confirms that none of the polyhedra are true groups:

    >>> from sympy.combinatorics.polyhedron import (
    ... tetrahedron, cube, octahedron, dodecahedron, icosahedron)
    ...
    >>> polyhedra = (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
    >>> [h.pgroup.is_group for h in polyhedra]
    ...
    [True, True, True, True, True]

    Although tests in polyhedron's test suite check that powers of the
    permutations in the groups generate all permutations of the vertices
    of the polyhedron, here we also demonstrate the powers of the given
    permutations create a complete group for the tetrahedron:

    >>> from sympy.combinatorics import Permutation, PermutationGroup
    >>> for h in polyhedra[:1]:
    ...     G = h.pgroup
    ...     perms = set()
    ...     for g in G:
    ...         for e in range(g.order()):
    ...             p = tuple((g**e).array_form)
    ...             perms.add(p)
    ...
    ...     perms = [Permutation(p) for p in perms]
    ...     assert PermutationGroup(perms).is_group

    In addition to doing the above, the tests in the suite confirm that the
    faces are all present after the application of each permutation.

    References
    ==========

    .. [1] https://dogschool.tripod.com/trianglegroup.html

    """
    def _pgroup_of_double(polyh, ordered_faces, pgroup):
        n = len(ordered_faces[0])
        # the vertices of the double which sits inside a give polyhedron
        # can be found by tracking the faces of the outer polyhedron.
        # A map between face and the vertex of the double is made so that
        # after rotation the position of the vertices can be located
        fmap = dict(zip(ordered_faces,
                        range(len(ordered_faces))))
        flat_faces = flatten(ordered_faces)
        new_pgroup = []
        for p in pgroup:
            h = polyh.copy()
            h.rotate(p)
            c = h.corners
            # reorder corners in the order they should appear when
            # enumerating the faces
            reorder = unflatten([c[j] for j in flat_faces], n)
            # make them canonical
            reorder = [tuple(map(as_int,
                       minlex(f, directed=False)))
                       for f in reorder]
            # map face to vertex: the resulting list of vertices are the
            # permutation that we seek for the double
            new_pgroup.append(Perm([fmap[f] for f in reorder]))
        return new_pgroup

    tetrahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 1),  # upper 3
        (1, 2, 3),  # bottom
    ]

    # cw from top
    #
    _t_pgroup = [
        Perm([[1, 2, 3], [0]]),  # cw from top
        Perm([[0, 1, 2], [3]]),  # cw from front face
        Perm([[0, 3, 2], [1]]),  # cw from back right face
        Perm([[0, 3, 1], [2]]),  # cw from back left face
        Perm([[0, 1], [2, 3]]),  # through front left edge
        Perm([[0, 2], [1, 3]]),  # through front right edge
        Perm([[0, 3], [1, 2]]),  # through back edge
    ]

    tetrahedron = Polyhedron(
        range(4),
        tetrahedron_faces,
        _t_pgroup)

    cube_faces = [
        (0, 1, 2, 3),  # upper
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 3, 7, 4),  # middle 4
        (4, 5, 6, 7),  # lower
    ]

    # U, D, F, B, L, R = up, down, front, back, left, right
    _c_pgroup = [Perm(p) for p in
        [
        [1, 2, 3, 0, 5, 6, 7, 4],  # cw from top, U
        [4, 0, 3, 7, 5, 1, 2, 6],  # cw from F face
        [4, 5, 1, 0, 7, 6, 2, 3],  # cw from R face

        [1, 0, 4, 5, 2, 3, 7, 6],  # cw through UF edge
        [6, 2, 1, 5, 7, 3, 0, 4],  # cw through UR edge
        [6, 7, 3, 2, 5, 4, 0, 1],  # cw through UB edge
        [3, 7, 4, 0, 2, 6, 5, 1],  # cw through UL edge
        [4, 7, 6, 5, 0, 3, 2, 1],  # cw through FL edge
        [6, 5, 4, 7, 2, 1, 0, 3],  # cw through FR edge

        [0, 3, 7, 4, 1, 2, 6, 5],  # cw through UFL vertex
        [5, 1, 0, 4, 6, 2, 3, 7],  # cw through UFR vertex
        [5, 6, 2, 1, 4, 7, 3, 0],  # cw through UBR vertex
        [7, 4, 0, 3, 6, 5, 1, 2],  # cw through UBL
        ]]

    cube = Polyhedron(
        range(8),
        cube_faces,
        _c_pgroup)

    octahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),  # top 4
        (1, 2, 5), (2, 3, 5), (3, 4, 5), (1, 4, 5),  # bottom 4
    ]

    octahedron = Polyhedron(
        range(6),
        octahedron_faces,
        _pgroup_of_double(cube, cube_faces, _c_pgroup))

    dodecahedron_faces = [
        (0, 1, 2, 3, 4),  # top
        (0, 1, 6, 10, 5), (1, 2, 7, 11, 6), (2, 3, 8, 12, 7),  # upper 5
        (3, 4, 9, 13, 8), (0, 4, 9, 14, 5),
        (5, 10, 16, 15, 14), (6, 10, 16, 17, 11), (7, 11, 17, 18,
          12),  # lower 5
        (8, 12, 18, 19, 13), (9, 13, 19, 15, 14),
        (15, 16, 17, 18, 19)  # bottom
    ]

    def _string_to_perm(s):
        rv = [Perm(range(20))]
        p = None
        for si in s:
            if si not in '01':
                count = int(si) - 1
            else:
                count = 1
                if si == '0':
                    p = _f0
                elif si == '1':
                    p = _f1
            rv.extend([p]*count)
        return Perm.rmul(*rv)

    # top face cw
    _f0 = Perm([
        1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11,
        12, 13, 14, 10, 16, 17, 18, 19, 15])
    # front face cw
    _f1 = Perm([
        5, 0, 4, 9, 14, 10, 1, 3, 13, 15,
        6, 2, 8, 19, 16, 17, 11, 7, 12, 18])
    # the strings below, like 0104 are shorthand for F0*F1*F0**4 and are
    # the remaining 4 face rotations, 15 edge permutations, and the
    # 10 vertex rotations.
    _dodeca_pgroup = [_f0, _f1] + [_string_to_perm(s) for s in '''
    0104 140 014 0410
    010 1403 03104 04103 102
    120 1304 01303 021302 03130
    0412041 041204103 04120410 041204104 041204102
    10 01 1402 0140 04102 0412 1204 1302 0130 03120'''.strip().split()]

    dodecahedron = Polyhedron(
        range(20),
        dodecahedron_faces,
        _dodeca_pgroup)

    icosahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
        (1, 6, 7), (1, 2, 7), (2, 7, 8), (2, 3, 8), (3, 8, 9),
        (3, 4, 9), (4, 9, 10), (4, 5, 10), (5, 6, 10), (1, 5, 6),
        (6, 7, 11), (7, 8, 11), (8, 9, 11), (9, 10, 11), (6, 10, 11)]

    icosahedron = Polyhedron(
        range(12),
        icosahedron_faces,
        _pgroup_of_double(
            dodecahedron, dodecahedron_faces, _dodeca_pgroup))

    return (tetrahedron, cube, octahedron, dodecahedron, icosahedron,
        tetrahedron_faces, cube_faces, octahedron_faces,
        dodecahedron_faces, icosahedron_faces)

# -----------------------------------------------------------------------
#   Standard Polyhedron groups
#
#   These are generated using _pgroup_calcs() above. However to save
#   import time we encode them explicitly here.
# -----------------------------------------------------------------------

tetrahedron = Polyhedron(
    Tuple(0, 1, 2, 3),
    Tuple(
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 1, 3),
        Tuple(1, 2, 3)),
    Tuple(
        Perm(1, 2, 3),
        Perm(3)(0, 1, 2),
        Perm(0, 3, 2),
        Perm(0, 3, 1),
        Perm(0, 1)(2, 3),
        Perm(0, 2)(1, 3),
        Perm(0, 3)(1, 2)
    ))

cube = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7),
    Tuple(
        Tuple(0, 1, 2, 3),
        Tuple(0, 1, 5, 4),
        Tuple(1, 2, 6, 5),
        Tuple(2, 3, 7, 6),
        Tuple(0, 3, 7, 4),
        Tuple(4, 5, 6, 7)),
    Tuple(
        Perm(0, 1, 2, 3)(4, 5, 6, 7),
        Perm(0, 4, 5, 1)(2, 3, 7, 6),
        Perm(0, 4, 7, 3)(1, 5, 6, 2),
        Perm(0, 1)(2, 4)(3, 5)(6, 7),
        Perm(0, 6)(1, 2)(3, 5)(4, 7),
        Perm(0, 6)(1, 7)(2, 3)(4, 5),
        Perm(0, 3)(1, 7)(2, 4)(5, 6),
        Perm(0, 4)(1, 7)(2, 6)(3, 5),
        Perm(0, 6)(1, 5)(2, 4)(3, 7),
        Perm(1, 3, 4)(2, 7, 5),
        Perm(7)(0, 5, 2)(3, 4, 6),
        Perm(0, 5, 7)(1, 6, 3),
        Perm(0, 7, 2)(1, 4, 6)))

octahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5),
    Tuple(
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 3, 4),
        Tuple(0, 1, 4),
        Tuple(1, 2, 5),
        Tuple(2, 3, 5),
        Tuple(3, 4, 5),
        Tuple(1, 4, 5)),
    Tuple(
        Perm(5)(1, 2, 3, 4),
        Perm(0, 4, 5, 2),
        Perm(0, 1, 5, 3),
        Perm(0, 1)(2, 4)(3, 5),
        Perm(0, 2)(1, 3)(4, 5),
        Perm(0, 3)(1, 5)(2, 4),
        Perm(0, 4)(1, 3)(2, 5),
        Perm(0, 5)(1, 4)(2, 3),
        Perm(0, 5)(1, 2)(3, 4),
        Perm(0, 4, 1)(2, 3, 5),
        Perm(0, 1, 2)(3, 4, 5),
        Perm(0, 2, 3)(1, 5, 4),
        Perm(0, 4, 3)(1, 5, 2)))

dodecahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    Tuple(
        Tuple(0, 1, 2, 3, 4),
        Tuple(0, 1, 6, 10, 5),
        Tuple(1, 2, 7, 11, 6),
        Tuple(2, 3, 8, 12, 7),
        Tuple(3, 4, 9, 13, 8),
        Tuple(0, 4, 9, 14, 5),
        Tuple(5, 10, 16, 15, 14),
        Tuple(6, 10, 16, 17, 11),
        Tuple(7, 11, 17, 18, 12),
        Tuple(8, 12, 18, 19, 13),
        Tuple(9, 13, 19, 15, 14),
        Tuple(15, 16, 17, 18, 19)),
    Tuple(
        Perm(0, 1, 2, 3, 4)(5, 6, 7, 8, 9)(10, 11, 12, 13, 14)(15, 16, 17, 18, 19),
        Perm(0, 5, 10, 6, 1)(2, 4, 14, 16, 11)(3, 9, 15, 17, 7)(8, 13, 19, 18, 12),
        Perm(0, 10, 17, 12, 3)(1, 6, 11, 7, 2)(4, 5, 16, 18, 8)(9, 14, 15, 19, 13),
        Perm(0, 6, 17, 19, 9)(1, 11, 18, 13, 4)(2, 7, 12, 8, 3)(5, 10, 16, 15, 14),
        Perm(0, 2, 12, 19, 14)(1, 7, 18, 15, 5)(3, 8, 13, 9, 4)(6, 11, 17, 16, 10),
        Perm(0, 4, 9, 14, 5)(1, 3, 13, 15, 10)(2, 8, 19, 16, 6)(7, 12, 18, 17, 11),
        Perm(0, 1)(2, 5)(3, 10)(4, 6)(7, 14)(8, 16)(9, 11)(12, 15)(13, 17)(18, 19),
        Perm(0, 7)(1, 2)(3, 6)(4, 11)(5, 12)(8, 10)(9, 17)(13, 16)(14, 18)(15, 19),
        Perm(0, 12)(1, 8)(2, 3)(4, 7)(5, 18)(6, 13)(9, 11)(10, 19)(14, 17)(15, 16),
        Perm(0, 8)(1, 13)(2, 9)(3, 4)(5, 12)(6, 19)(7, 14)(10, 18)(11, 15)(16, 17),
        Perm(0, 4)(1, 9)(2, 14)(3, 5)(6, 13)(7, 15)(8, 10)(11, 19)(12, 16)(17, 18),
        Perm(0, 5)(1, 14)(2, 15)(3, 16)(4, 10)(6, 9)(7, 19)(8, 17)(11, 13)(12, 18),
        Perm(0, 11)(1, 6)(2, 10)(3, 16)(4, 17)(5, 7)(8, 15)(9, 18)(12, 14)(13, 19),
        Perm(0, 18)(1, 12)(2, 7)(3, 11)(4, 17)(5, 19)(6, 8)(9, 16)(10, 13)(14, 15),
        Perm(0, 18)(1, 19)(2, 13)(3, 8)(4, 12)(5, 17)(6, 15)(7, 9)(10, 16)(11, 14),
        Perm(0, 13)(1, 19)(2, 15)(3, 14)(4, 9)(5, 8)(6, 18)(7, 16)(10, 12)(11, 17),
        Perm(0, 16)(1, 15)(2, 19)(3, 18)(4, 17)(5, 10)(6, 14)(7, 13)(8, 12)(9, 11),
        Perm(0, 18)(1, 17)(2, 16)(3, 15)(4, 19)(5, 12)(6, 11)(7, 10)(8, 14)(9, 13),
        Perm(0, 15)(1, 19)(2, 18)(3, 17)(4, 16)(5, 14)(6, 13)(7, 12)(8, 11)(9, 10),
        Perm(0, 17)(1, 16)(2, 15)(3, 19)(4, 18)(5, 11)(6, 10)(7, 14)(8, 13)(9, 12),
        Perm(0, 19)(1, 18)(2, 17)(3, 16)(4, 15)(5, 13)(6, 12)(7, 11)(8, 10)(9, 14),
        Perm(1, 4, 5)(2, 9, 10)(3, 14, 6)(7, 13, 16)(8, 15, 11)(12, 19, 17),
        Perm(19)(0, 6, 2)(3, 5, 11)(4, 10, 7)(8, 14, 17)(9, 16, 12)(13, 15, 18),
        Perm(0, 11, 8)(1, 7, 3)(4, 6, 12)(5, 17, 13)(9, 10, 18)(14, 16, 19),
        Perm(0, 7, 13)(1, 12, 9)(2, 8, 4)(5, 11, 19)(6, 18, 14)(10, 17, 15),
        Perm(0, 3, 9)(1, 8, 14)(2, 13, 5)(6, 12, 15)(7, 19, 10)(11, 18, 16),
        Perm(0, 14, 10)(1, 9, 16)(2, 13, 17)(3, 19, 11)(4, 15, 6)(7, 8, 18),
        Perm(0, 16, 7)(1, 10, 11)(2, 5, 17)(3, 14, 18)(4, 15, 12)(8, 9, 19),
        Perm(0, 16, 13)(1, 17, 8)(2, 11, 12)(3, 6, 18)(4, 10, 19)(5, 15, 9),
        Perm(0, 11, 15)(1, 17, 14)(2, 18, 9)(3, 12, 13)(4, 7, 19)(5, 6, 16),
        Perm(0, 8, 15)(1, 12, 16)(2, 18, 10)(3, 19, 5)(4, 13, 14)(6, 7, 17)))

icosahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    Tuple(
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 3, 4),
        Tuple(0, 4, 5),
        Tuple(0, 1, 5),
        Tuple(1, 6, 7),
        Tuple(1, 2, 7),
        Tuple(2, 7, 8),
        Tuple(2, 3, 8),
        Tuple(3, 8, 9),
        Tuple(3, 4, 9),
        Tuple(4, 9, 10),
        Tuple(4, 5, 10),
        Tuple(5, 6, 10),
        Tuple(1, 5, 6),
        Tuple(6, 7, 11),
        Tuple(7, 8, 11),
        Tuple(8, 9, 11),
        Tuple(9, 10, 11),
        Tuple(6, 10, 11)),
    Tuple(
        Perm(11)(1, 2, 3, 4, 5)(6, 7, 8, 9, 10),
        Perm(0, 5, 6, 7, 2)(3, 4, 10, 11, 8),
        Perm(0, 1, 7, 8, 3)(4, 5, 6, 11, 9),
        Perm(0, 2, 8, 9, 4)(1, 7, 11, 10, 5),
        Perm(0, 3, 9, 10, 5)(1, 2, 8, 11, 6),
        Perm(0, 4, 10, 6, 1)(2, 3, 9, 11, 7),
        Perm(0, 1)(2, 5)(3, 6)(4, 7)(8, 10)(9, 11),
        Perm(0, 2)(1, 3)(4, 7)(5, 8)(6, 9)(10, 11),
        Perm(0, 3)(1, 9)(2, 4)(5, 8)(6, 11)(7, 10),
        Perm(0, 4)(1, 9)(2, 10)(3, 5)(6, 8)(7, 11),
        Perm(0, 5)(1, 4)(2, 10)(3, 6)(7, 9)(8, 11),
        Perm(0, 6)(1, 5)(2, 10)(3, 11)(4, 7)(8, 9),
        Perm(0, 7)(1, 2)(3, 6)(4, 11)(5, 8)(9, 10),
        Perm(0, 8)(1, 9)(2, 3)(4, 7)(5, 11)(6, 10),
        Perm(0, 9)(1, 11)(2, 10)(3, 4)(5, 8)(6, 7),
        Perm(0, 10)(1, 9)(2, 11)(3, 6)(4, 5)(7, 8),
        Perm(0, 11)(1, 6)(2, 10)(3, 9)(4, 8)(5, 7),
        Perm(0, 11)(1, 8)(2, 7)(3, 6)(4, 10)(5, 9),
        Perm(0, 11)(1, 10)(2, 9)(3, 8)(4, 7)(5, 6),
        Perm(0, 11)(1, 7)(2, 6)(3, 10)(4, 9)(5, 8),
        Perm(0, 11)(1, 9)(2, 8)(3, 7)(4, 6)(5, 10),
        Perm(0, 5, 1)(2, 4, 6)(3, 10, 7)(8, 9, 11),
        Perm(0, 1, 2)(3, 5, 7)(4, 6, 8)(9, 10, 11),
        Perm(0, 2, 3)(1, 8, 4)(5, 7, 9)(6, 11, 10),
        Perm(0, 3, 4)(1, 8, 10)(2, 9, 5)(6, 7, 11),
        Perm(0, 4, 5)(1, 3, 10)(2, 9, 6)(7, 8, 11),
        Perm(0, 10, 7)(1, 5, 6)(2, 4, 11)(3, 9, 8),
        Perm(0, 6, 8)(1, 7, 2)(3, 5, 11)(4, 10, 9),
        Perm(0, 7, 9)(1, 11, 4)(2, 8, 3)(5, 6, 10),
        Perm(0, 8, 10)(1, 7, 6)(2, 11, 5)(3, 9, 4),
        Perm(0, 9, 6)(1, 3, 11)(2, 8, 7)(4, 10, 5)))

tetrahedron_faces = [tuple(arg) for arg in tetrahedron.faces]

cube_faces = [tuple(arg) for arg in cube.faces]

octahedron_faces = [tuple(arg) for arg in octahedron.faces]

dodecahedron_faces = [tuple(arg) for arg in dodecahedron.faces]

icosahedron_faces = [tuple(arg) for arg in icosahedron.faces]