
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\joint.py ===
# coding=utf-8

from abc import ABC, abstractmethod

from sympy import pi, Derivative, Matrix
from sympy.core.function import AppliedUndef
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
                                  ReferenceFrame)
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Joint', 'PinJoint', 'PrismaticJoint', 'CylindricalJoint',
           'PlanarJoint', 'SphericalJoint', 'WeldJoint']


class Joint(ABC):
    """Abstract base class for all specific joints.

    Explanation
    ===========

    A joint subtracts degrees of freedom from a body. This is the base class
    for all specific joints and holds all common methods acting as an interface
    for all joints. Custom joint can be created by inheriting Joint class and
    defining all abstract functions.

    The abstract methods are:

    - ``_generate_coordinates``
    - ``_generate_speeds``
    - ``_orient_frames``
    - ``_set_angular_velocity``
    - ``_set_linear_velocity``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    coordinates : iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Notes
    =====

    When providing a vector as the intermediate frame, a new intermediate frame
    is created which aligns its X axis with the provided vector. This is done
    with a single fixed rotation about a rotation axis. This rotation axis is
    determined by taking the cross product of the ``body.x`` axis with the
    provided vector. In the case where the provided vector is in the ``-body.x``
    direction, the rotation is done about the ``body.y`` axis.

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 parent_joint_pos=None, child_joint_pos=None):

        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name

        if not isinstance(parent, BodyBase):
            raise TypeError('Parent must be a body.')
        self._parent = parent

        if not isinstance(child, BodyBase):
            raise TypeError('Child must be a body.')
        self._child = child

        if parent_axis is not None or child_axis is not None:
            sympy_deprecation_warning(
                """
                The parent_axis and child_axis arguments for the Joint classes
                are deprecated. Instead use parent_interframe, child_interframe.
                """,
                deprecated_since_version="1.12",
                active_deprecations_target="deprecated-mechanics-joint-axis",
                stacklevel=4
            )
            if parent_interframe is None:
                parent_interframe = parent_axis
            if child_interframe is None:
                child_interframe = child_axis

        # Set parent and child frame attributes
        if hasattr(self._parent, 'frame'):
            self._parent_frame = self._parent.frame
        else:
            if isinstance(parent_interframe, ReferenceFrame):
                self._parent_frame = parent_interframe
            else:
                self._parent_frame = ReferenceFrame(
                    f'{self.name}_{self._parent.name}_frame')
        if hasattr(self._child, 'frame'):
            self._child_frame = self._child.frame
        else:
            if isinstance(child_interframe, ReferenceFrame):
                self._child_frame = child_interframe
            else:
                self._child_frame = ReferenceFrame(
                    f'{self.name}_{self._child.name}_frame')

        self._parent_interframe = self._locate_joint_frame(
            self._parent, parent_interframe, self._parent_frame)
        self._child_interframe = self._locate_joint_frame(
            self._child, child_interframe, self._child_frame)
        self._parent_axis = self._axis(parent_axis, self._parent_frame)
        self._child_axis = self._axis(child_axis, self._child_frame)

        if parent_joint_pos is not None or child_joint_pos is not None:
            sympy_deprecation_warning(
                """
                The parent_joint_pos and child_joint_pos arguments for the Joint
                classes are deprecated. Instead use parent_point and child_point.
                """,
                deprecated_since_version="1.12",
                active_deprecations_target="deprecated-mechanics-joint-pos",
                stacklevel=4
            )
            if parent_point is None:
                parent_point = parent_joint_pos
            if child_point is None:
                child_point = child_joint_pos
        self._parent_point = self._locate_joint_pos(
            self._parent, parent_point, self._parent_frame)
        self._child_point = self._locate_joint_pos(
            self._child, child_point, self._child_frame)

        self._coordinates = self._generate_coordinates(coordinates)
        self._speeds = self._generate_speeds(speeds)
        _validate_coordinates(self.coordinates, self.speeds)
        self._kdes = self._generate_kdes()

        self._orient_frames()
        self._set_angular_velocity()
        self._set_linear_velocity()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        """Name of the joint."""
        return self._name

    @property
    def parent(self):
        """Parent body of Joint."""
        return self._parent

    @property
    def child(self):
        """Child body of Joint."""
        return self._child

    @property
    def coordinates(self):
        """Matrix of the joint's generalized coordinates."""
        return self._coordinates

    @property
    def speeds(self):
        """Matrix of the joint's generalized speeds."""
        return self._speeds

    @property
    def kdes(self):
        """Kinematical differential equations of the joint."""
        return self._kdes

    @property
    def parent_axis(self):
        """The axis of parent frame."""
        # Will be removed with `deprecated-mechanics-joint-axis`
        return self._parent_axis

    @property
    def child_axis(self):
        """The axis of child frame."""
        # Will be removed with `deprecated-mechanics-joint-axis`
        return self._child_axis

    @property
    def parent_point(self):
        """Attachment point where the joint is fixed to the parent body."""
        return self._parent_point

    @property
    def child_point(self):
        """Attachment point where the joint is fixed to the child body."""
        return self._child_point

    @property
    def parent_interframe(self):
        return self._parent_interframe

    @property
    def child_interframe(self):
        return self._child_interframe

    @abstractmethod
    def _generate_coordinates(self, coordinates):
        """Generate Matrix of the joint's generalized coordinates."""
        pass

    @abstractmethod
    def _generate_speeds(self, speeds):
        """Generate Matrix of the joint's generalized speeds."""
        pass

    @abstractmethod
    def _orient_frames(self):
        """Orient frames as per the joint."""
        pass

    @abstractmethod
    def _set_angular_velocity(self):
        """Set angular velocity of the joint related frames."""
        pass

    @abstractmethod
    def _set_linear_velocity(self):
        """Set velocity of related points to the joint."""
        pass

    @staticmethod
    def _to_vector(matrix, frame):
        """Converts a matrix to a vector in the given frame."""
        return Vector([(matrix, frame)])

    @staticmethod
    def _axis(ax, *frames):
        """Check whether an axis is fixed in one of the frames."""
        if ax is None:
            ax = frames[0].x
            return ax
        if not isinstance(ax, Vector):
            raise TypeError("Axis must be a Vector.")
        ref_frame = None  # Find a body in which the axis can be expressed
        for frame in frames:
            try:
                ax.to_matrix(frame)
            except ValueError:
                pass
            else:
                ref_frame = frame
                break
        if ref_frame is None:
            raise ValueError("Axis cannot be expressed in one of the body's "
                             "frames.")
        if not ax.dt(ref_frame) == 0:
            raise ValueError('Axis cannot be time-varying when viewed from the '
                             'associated body.')
        return ax

    @staticmethod
    def _choose_rotation_axis(frame, axis):
        components = axis.to_matrix(frame)
        x, y, z = components[0], components[1], components[2]

        if x != 0:
            if y != 0:
                if z != 0:
                    return cross(axis, frame.x)
            if z != 0:
                return frame.y
            return frame.z
        else:
            if y != 0:
                return frame.x
            return frame.y

    @staticmethod
    def _create_aligned_interframe(frame, align_axis, frame_axis=None,
                                   frame_name=None):
        """
        Returns an intermediate frame, where the ``frame_axis`` defined in
        ``frame`` is aligned with ``axis``. By default this means that the X
        axis will be aligned with ``axis``.

        Parameters
        ==========

        frame : BodyBase or ReferenceFrame
            The body or reference frame with respect to which the intermediate
            frame is oriented.
        align_axis : Vector
            The vector with respect to which the intermediate frame will be
            aligned.
        frame_axis : Vector
            The vector of the frame which should get aligned with ``axis``. The
            default is the X axis of the frame.
        frame_name : string
            Name of the to be created intermediate frame. The default adds
            "_int_frame" to the name of ``frame``.

        Example
        =======

        An intermediate frame, where the X axis of the parent becomes aligned
        with ``parent.y + parent.z`` can be created as follows:

        >>> from sympy.physics.mechanics.joint import Joint
        >>> from sympy.physics.mechanics import RigidBody
        >>> parent = RigidBody('parent')
        >>> parent_interframe = Joint._create_aligned_interframe(
        ...     parent, parent.y + parent.z)
        >>> parent_interframe
        parent_int_frame
        >>> parent.frame.dcm(parent_interframe)
        Matrix([
        [        0, -sqrt(2)/2, -sqrt(2)/2],
        [sqrt(2)/2,        1/2,       -1/2],
        [sqrt(2)/2,       -1/2,        1/2]])
        >>> (parent.y + parent.z).express(parent_interframe)
        sqrt(2)*parent_int_frame.x

        Notes
        =====

        The direction cosine matrix between the given frame and intermediate
        frame is formed using a simple rotation about an axis that is normal to
        both ``align_axis`` and ``frame_axis``. In general, the normal axis is
        formed by crossing the ``frame_axis`` with the ``align_axis``. The
        exception is if the axes are parallel with opposite directions, in which
        case the rotation vector is chosen using the rules in the following
        table with the vectors expressed in the given frame:

        .. list-table::
           :header-rows: 1

           * - ``align_axis``
             - ``frame_axis``
             - ``rotation_axis``
           * - ``-x``
             - ``x``
             - ``z``
           * - ``-y``
             - ``y``
             - ``x``
           * - ``-z``
             - ``z``
             - ``y``
           * - ``-x-y``
             - ``x+y``
             - ``z``
           * - ``-y-z``
             - ``y+z``
             - ``x``
           * - ``-x-z``
             - ``x+z``
             - ``y``
           * - ``-x-y-z``
             - ``x+y+z``
             - ``(x+y+z) × x``

        """
        if isinstance(frame, BodyBase):
            frame = frame.frame
        if frame_axis is None:
            frame_axis = frame.x
        if frame_name is None:
            if frame.name[-6:] == '_frame':
                frame_name = f'{frame.name[:-6]}_int_frame'
            else:
                frame_name = f'{frame.name}_int_frame'
        angle = frame_axis.angle_between(align_axis)
        rotation_axis = cross(frame_axis, align_axis)
        if rotation_axis == Vector(0) and angle == 0:
            return frame
        if angle == pi:
            rotation_axis = Joint._choose_rotation_axis(frame, align_axis)

        int_frame = ReferenceFrame(frame_name)
        int_frame.orient_axis(frame, rotation_axis, angle)
        int_frame.set_ang_vel(frame, 0 * rotation_axis)
        return int_frame

    def _generate_kdes(self):
        """Generate kinematical differential equations."""
        kdes = []
        t = dynamicsymbols._t
        for i in range(len(self.coordinates)):
            kdes.append(-self.coordinates[i].diff(t) + self.speeds[i])
        return Matrix(kdes)

    def _locate_joint_pos(self, body, joint_pos, body_frame=None):
        """Returns the attachment point of a body."""
        if body_frame is None:
            body_frame = body.frame
        if joint_pos is None:
            return body.masscenter
        if not isinstance(joint_pos, (Point, Vector)):
            raise TypeError('Attachment point must be a Point or Vector.')
        if isinstance(joint_pos, Vector):
            point_name = f'{self.name}_{body.name}_joint'
            joint_pos = body.masscenter.locatenew(point_name, joint_pos)
        if not joint_pos.pos_from(body.masscenter).dt(body_frame) == 0:
            raise ValueError('Attachment point must be fixed to the associated '
                             'body.')
        return joint_pos

    def _locate_joint_frame(self, body, interframe, body_frame=None):
        """Returns the attachment frame of a body."""
        if body_frame is None:
            body_frame = body.frame
        if interframe is None:
            return body_frame
        if isinstance(interframe, Vector):
            interframe = Joint._create_aligned_interframe(
                body_frame, interframe,
                frame_name=f'{self.name}_{body.name}_int_frame')
        elif not isinstance(interframe, ReferenceFrame):
            raise TypeError('Interframe must be a ReferenceFrame.')
        if not interframe.ang_vel_in(body_frame) == 0:
            raise ValueError(f'Interframe {interframe} is not fixed to body '
                             f'{body}.')
        body.masscenter.set_vel(interframe, 0)  # Fixate interframe to body
        return interframe

    def _fill_coordinate_list(self, coordinates, n_coords, label='q', offset=0,
                              number_single=False):
        """Helper method for _generate_coordinates and _generate_speeds.

        Parameters
        ==========

        coordinates : iterable
            Iterable of coordinates or speeds that have been provided.
        n_coords : Integer
            Number of coordinates that should be returned.
        label : String, optional
            Coordinate type either 'q' (coordinates) or 'u' (speeds). The
            Default is 'q'.
        offset : Integer
            Count offset when creating new dynamicsymbols. The default is 0.
        number_single : Boolean
            Boolean whether if n_coords == 1, number should still be used. The
            default is False.

        """

        def create_symbol(number):
            if n_coords == 1 and not number_single:
                return dynamicsymbols(f'{label}_{self.name}')
            return dynamicsymbols(f'{label}{number}_{self.name}')

        name = 'generalized coordinate' if label == 'q' else 'generalized speed'
        generated_coordinates = []
        if coordinates is None:
            coordinates = []
        elif not iterable(coordinates):
            coordinates = [coordinates]
        if not (len(coordinates) == 0 or len(coordinates) == n_coords):
            raise ValueError(f'Expected {n_coords} {name}s, instead got '
                             f'{len(coordinates)} {name}s.')
        # Supports more iterables, also Matrix
        for i, coord in enumerate(coordinates):
            if coord is None:
                generated_coordinates.append(create_symbol(i + offset))
            elif isinstance(coord, (AppliedUndef, Derivative)):
                generated_coordinates.append(coord)
            else:
                raise TypeError(f'The {name} {coord} should have been a '
                                f'dynamicsymbol.')
        for i in range(len(coordinates) + offset, n_coords + offset):
            generated_coordinates.append(create_symbol(i))
        return Matrix(generated_coordinates)


class PinJoint(Joint):
    """Pin (Revolute) Joint.

    .. raw:: html
        :file: ../../../doc/src/explanation/modules/physics/mechanics/PinJoint.svg

    Explanation
    ===========

    A pin joint is defined such that the joint rotation axis is fixed in both
    the child and parent and the location of the joint is relative to the mass
    center of each body. The child rotates an angle, θ, from the parent about
    the rotation axis and has a simple angular speed, ω, relative to the
    parent. The direction cosine matrix between the child interframe and
    parent interframe is formed using a simple rotation about the joint axis.
    The page on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    coordinates : dynamicsymbol, optional
        Generalized coordinates of the joint.
    speeds : dynamicsymbol, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector
        The axis about which the rotation occurs. Note that the components
        of this axis are the same in the parent_interframe and child_interframe.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : Matrix
        Matrix of the joint's generalized speeds. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    joint_axis : Vector
        The axis about which the rotation occurs. Note that the components of
        this axis are the same in the parent_interframe and child_interframe.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single pin joint is created from two bodies and has the following basic
    attributes:

    >>> from sympy.physics.mechanics import RigidBody, PinJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PinJoint('PC', parent, child)
    >>> joint
    PinJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([[q_PC(t)]])
    >>> joint.speeds
    Matrix([[u_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,             0,            0],
    [0,  cos(q_PC(t)), sin(q_PC(t))],
    [0, -sin(q_PC(t)), cos(q_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the pin joint, the kinematics of simple
    double pendulum that rotates about the Z axis of each connected body can be
    created as follows.

    >>> from sympy import symbols, trigsimp
    >>> from sympy.physics.mechanics import RigidBody, PinJoint
    >>> l1, l2 = symbols('l1 l2')

    First create bodies to represent the fixed ceiling and one to represent
    each pendulum bob.

    >>> ceiling = RigidBody('C')
    >>> upper_bob = RigidBody('U')
    >>> lower_bob = RigidBody('L')

    The first joint will connect the upper bob to the ceiling by a distance of
    ``l1`` and the joint axis will be about the Z axis for each body.

    >>> ceiling_joint = PinJoint('P1', ceiling, upper_bob,
    ... child_point=-l1*upper_bob.frame.x,
    ... joint_axis=ceiling.frame.z)

    The second joint will connect the lower bob to the upper bob by a distance
    of ``l2`` and the joint axis will also be about the Z axis for each body.

    >>> pendulum_joint = PinJoint('P2', upper_bob, lower_bob,
    ... child_point=-l2*lower_bob.frame.x,
    ... joint_axis=upper_bob.frame.z)

    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of pendulum link relative
    to the ceiling are found:

    >>> upper_bob.frame.dcm(ceiling.frame)
    Matrix([
    [ cos(q_P1(t)), sin(q_P1(t)), 0],
    [-sin(q_P1(t)), cos(q_P1(t)), 0],
    [            0,            0, 1]])
    >>> trigsimp(lower_bob.frame.dcm(ceiling.frame))
    Matrix([
    [ cos(q_P1(t) + q_P2(t)), sin(q_P1(t) + q_P2(t)), 0],
    [-sin(q_P1(t) + q_P2(t)), cos(q_P1(t) + q_P2(t)), 0],
    [                      0,                      0, 1]])

    The position of the lower bob's masscenter is found with:

    >>> lower_bob.masscenter.pos_from(ceiling.masscenter)
    l1*U_frame.x + l2*L_frame.x

    The angular velocities of the two pendulum links can be computed with
    respect to the ceiling.

    >>> upper_bob.frame.ang_vel_in(ceiling.frame)
    u_P1(t)*C_frame.z
    >>> lower_bob.frame.ang_vel_in(ceiling.frame)
    u_P1(t)*C_frame.z + u_P2(t)*U_frame.z

    And finally, the linear velocities of the two pendulum bobs can be computed
    with respect to the ceiling.

    >>> upper_bob.masscenter.vel(ceiling.frame)
    l1*u_P1(t)*U_frame.y
    >>> lower_bob.masscenter.vel(ceiling.frame)
    l1*u_P1(t)*U_frame.y + l2*(u_P1(t) + u_P2(t))*L_frame.y

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 joint_axis=None, parent_joint_pos=None, child_joint_pos=None):

        self._joint_axis = joint_axis
        super().__init__(name, parent, child, coordinates, speeds, parent_point,
                         child_point, parent_interframe, child_interframe,
                         parent_axis, child_axis, parent_joint_pos,
                         child_joint_pos)

    def __str__(self):
        return (f'PinJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis about which the child rotates with respect to the parent."""
        return self._joint_axis

    def _generate_coordinates(self, coordinate):
        return self._fill_coordinate_list(coordinate, 1, 'q')

    def _generate_speeds(self, speed):
        return self._fill_coordinate_list(speed, 1, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, self.coordinates[0])

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, self.speeds[
            0] * self.joint_axis.normalize())

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.v2pt_theory(self.parent_point,
                                          self._parent_frame, self._child_frame)


class PrismaticJoint(Joint):
    """Prismatic (Sliding) Joint.

    .. image:: PrismaticJoint.svg

    Explanation
    ===========

    It is defined such that the child body translates with respect to the parent
    body along the body-fixed joint axis. The location of the joint is defined
    by two points, one in each body, which coincide when the generalized
    coordinate is zero. The direction cosine matrix between the
    parent_interframe and child_interframe is the identity matrix. Therefore,
    the direction cosine matrix between the parent and child frames is fully
    defined by the definition of the intermediate frames. The page on the joints
    framework gives a more detailed explanation of the intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    coordinates : dynamicsymbol, optional
        Generalized coordinates of the joint. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : dynamicsymbol, optional
        Generalized speeds of joint. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector
        The axis along which the translation occurs. Note that the components
        of this axis are the same in the parent_interframe and child_interframe.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single prismatic joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, PrismaticJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PrismaticJoint('PC', parent, child)
    >>> joint
    PrismaticJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([[q_PC(t)]])
    >>> joint.speeds
    Matrix([[u_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    0
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q_PC(t)*P_frame.x

    To further demonstrate the use of the prismatic joint, the kinematics of two
    masses sliding, one moving relative to a fixed body and the other relative
    to the moving body. about the X axis of each connected body can be created
    as follows.

    >>> from sympy.physics.mechanics import PrismaticJoint, RigidBody

    First create bodies to represent the fixed ceiling and one to represent
    a particle.

    >>> wall = RigidBody('W')
    >>> Part1 = RigidBody('P1')
    >>> Part2 = RigidBody('P2')

    The first joint will connect the particle to the ceiling and the
    joint axis will be about the X axis for each body.

    >>> J1 = PrismaticJoint('J1', wall, Part1)

    The second joint will connect the second particle to the first particle
    and the joint axis will also be about the X axis for each body.

    >>> J2 = PrismaticJoint('J2', Part1, Part2)

    Once the joint is established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of Part relative
    to the ceiling are found:

    >>> Part1.frame.dcm(wall.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    >>> Part2.frame.dcm(wall.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    The position of the particles' masscenter is found with:

    >>> Part1.masscenter.pos_from(wall.masscenter)
    q_J1(t)*W_frame.x

    >>> Part2.masscenter.pos_from(wall.masscenter)
    q_J1(t)*W_frame.x + q_J2(t)*P1_frame.x

    The angular velocities of the two particle links can be computed with
    respect to the ceiling.

    >>> Part1.frame.ang_vel_in(wall.frame)
    0

    >>> Part2.frame.ang_vel_in(wall.frame)
    0

    And finally, the linear velocities of the two particles can be computed
    with respect to the ceiling.

    >>> Part1.masscenter.vel(wall.frame)
    u_J1(t)*W_frame.x

    >>> Part2.masscenter.vel(wall.frame)
    u_J1(t)*W_frame.x + Derivative(q_J2(t), t)*P1_frame.x

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 joint_axis=None, parent_joint_pos=None, child_joint_pos=None):

        self._joint_axis = joint_axis
        super().__init__(name, parent, child, coordinates, speeds, parent_point,
                         child_point, parent_interframe, child_interframe,
                         parent_axis, child_axis, parent_joint_pos,
                         child_joint_pos)

    def __str__(self):
        return (f'PrismaticJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis along which the child translates with respect to the parent."""
        return self._joint_axis

    def _generate_coordinates(self, coordinate):
        return self._fill_coordinate_list(coordinate, 1, 'q')

    def _generate_speeds(self, speed):
        return self._fill_coordinate_list(speed, 1, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, 0)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, 0)

    def _set_linear_velocity(self):
        axis = self.joint_axis.normalize()
        self.child_point.set_pos(self.parent_point, self.coordinates[0] * axis)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child_point.set_vel(self._parent_frame, self.speeds[0] * axis)
        self.child.masscenter.set_vel(self._parent_frame, self.speeds[0] * axis)


class CylindricalJoint(Joint):
    """Cylindrical Joint.

    .. image:: CylindricalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A cylindrical joint is defined such that the child body both rotates about
    and translates along the body-fixed joint axis with respect to the parent
    body. The joint axis is both the rotation axis and translation axis. The
    location of the joint is defined by two points, one in each body, which
    coincide when the generalized coordinate corresponding to the translation is
    zero. The direction cosine matrix between the child interframe and parent
    interframe is formed using a simple rotation about the joint axis. The page
    on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    translation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the translation distance. The
        default value is ``dynamicsymbols(f'q1_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    translation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the translation velocity. The default
        value is ``dynamicsymbols(f'u1_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector, optional
        The rotation as well as translation axis. Note that the components of
        this axis are the same in the parent_interframe and child_interframe.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    rotation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the rotation angle.
    translation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the translation distance.
    rotation_speed : dynamicsymbol
        Generalized speed corresponding to the angular velocity.
    translation_speed : dynamicsymbol
        Generalized speed corresponding to the translation velocity.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.
    joint_axis : Vector
        The axis of rotation and translation.

    Examples
    =========

    A single cylindrical joint is created between two bodies and has the
    following basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, CylindricalJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = CylindricalJoint('PC', parent, child)
    >>> joint
    CylindricalJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u0_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.x
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.x

    To further demonstrate the use of the cylindrical joint, the kinematics of
    two cylindrical joints perpendicular to each other can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import RigidBody, CylindricalJoint
    >>> r, l, w = symbols('r l w')

    First create bodies to represent the fixed floor with a fixed pole on it.
    The second body represents a freely moving tube around that pole. The third
    body represents a solid flag freely translating along and rotating around
    the Y axis of the tube.

    >>> floor = RigidBody('floor')
    >>> tube = RigidBody('tube')
    >>> flag = RigidBody('flag')

    The first joint will connect the first tube to the floor with it translating
    along and rotating around the Z axis of both bodies.

    >>> floor_joint = CylindricalJoint('C1', floor, tube, joint_axis=floor.z)

    The second joint will connect the tube perpendicular to the flag along the Y
    axis of both the tube and the flag, with the joint located at a distance
    ``r`` from the tube's center of mass and a combination of the distances
    ``l`` and ``w`` from the flag's center of mass.

    >>> flag_joint = CylindricalJoint('C2', tube, flag,
    ...                               parent_point=r * tube.y,
    ...                               child_point=-w * flag.y + l * flag.z,
    ...                               joint_axis=tube.y)

    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of both the body and the
    flag relative to the floor are found:

    >>> tube.frame.dcm(floor.frame)
    Matrix([
    [ cos(q0_C1(t)), sin(q0_C1(t)), 0],
    [-sin(q0_C1(t)), cos(q0_C1(t)), 0],
    [             0,             0, 1]])
    >>> flag.frame.dcm(floor.frame)
    Matrix([
    [cos(q0_C1(t))*cos(q0_C2(t)), sin(q0_C1(t))*cos(q0_C2(t)), -sin(q0_C2(t))],
    [             -sin(q0_C1(t)),               cos(q0_C1(t)),              0],
    [sin(q0_C2(t))*cos(q0_C1(t)), sin(q0_C1(t))*sin(q0_C2(t)),  cos(q0_C2(t))]])

    The position of the flag's center of mass is found with:

    >>> flag.masscenter.pos_from(floor.masscenter)
    q1_C1(t)*floor_frame.z + (r + q1_C2(t))*tube_frame.y + w*flag_frame.y - l*flag_frame.z

    The angular velocities of the two tubes can be computed with respect to the
    floor.

    >>> tube.frame.ang_vel_in(floor.frame)
    u0_C1(t)*floor_frame.z
    >>> flag.frame.ang_vel_in(floor.frame)
    u0_C1(t)*floor_frame.z + u0_C2(t)*tube_frame.y

    Finally, the linear velocities of the two tube centers of mass can be
    computed with respect to the floor, while expressed in the tube's frame.

    >>> tube.masscenter.vel(floor.frame).to_matrix(tube.frame)
    Matrix([
    [       0],
    [       0],
    [u1_C1(t)]])
    >>> flag.masscenter.vel(floor.frame).to_matrix(tube.frame).simplify()
    Matrix([
    [-l*u0_C2(t)*cos(q0_C2(t)) - r*u0_C1(t) - w*u0_C1(t) - q1_C2(t)*u0_C1(t)],
    [                    -l*u0_C1(t)*sin(q0_C2(t)) + Derivative(q1_C2(t), t)],
    [                                    l*u0_C2(t)*sin(q0_C2(t)) + u1_C1(t)]])

    """

    def __init__(self, name, parent, child, rotation_coordinate=None,
                 translation_coordinate=None, rotation_speed=None,
                 translation_speed=None, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None,
                 joint_axis=None):
        self._joint_axis = joint_axis
        coordinates = (rotation_coordinate, translation_coordinate)
        speeds = (rotation_speed, translation_speed)
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'CylindricalJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis about and along which the rotation and translation occurs."""
        return self._joint_axis

    @property
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    def translation_coordinate(self):
        """Generalized coordinate corresponding to the translation distance."""
        return self.coordinates[1]

    @property
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    def translation_speed(self):
        """Generalized speed corresponding to the translation velocity."""
        return self.speeds[1]

    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 2, 'q')

    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, 2, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, self.rotation_coordinate)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(
            self.parent_interframe,
            self.rotation_speed * self.joint_axis.normalize())

    def _set_linear_velocity(self):
        self.child_point.set_pos(
            self.parent_point,
            self.translation_coordinate * self.joint_axis.normalize())
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child_point.set_vel(
            self._parent_frame,
            self.translation_speed * self.joint_axis.normalize())
        self.child.masscenter.v2pt_theory(self.child_point, self._parent_frame,
                                          self.child_interframe)


class PlanarJoint(Joint):
    """Planar Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/PlanarJoint.svg

    Explanation
    ===========

    A planar joint is defined such that the child body translates over a fixed
    plane of the parent body as well as rotate about the rotation axis, which
    is perpendicular to that plane. The origin of this plane is the
    ``parent_point`` and the plane is spanned by two nonparallel planar vectors.
    The location of the ``child_point`` is based on the planar vectors
    ($\\vec{v}_1$, $\\vec{v}_2$) and generalized coordinates ($q_1$, $q_2$),
    i.e. $\\vec{r} = q_1 \\hat{v}_1 + q_2 \\hat{v}_2$. The direction cosine
    matrix between the ``child_interframe`` and ``parent_interframe`` is formed
    using a simple rotation ($q_0$) about the rotation axis.

    In order to simplify the definition of the ``PlanarJoint``, the
    ``rotation_axis`` and ``planar_vectors`` are set to be the unit vectors of
    the ``parent_interframe`` according to the table below. This ensures that
    you can only define these vectors by creating a separate frame and supplying
    that as the interframe. If you however would only like to supply the normals
    of the plane with respect to the parent and child bodies, then you can also
    supply those to the ``parent_interframe`` and ``child_interframe``
    arguments. An example of both of these cases is in the examples section
    below and the page on the joints framework provides a more detailed
    explanation of the intermediate frames.

    .. list-table::

        * - ``rotation_axis``
          - ``parent_interframe.x``
        * - ``planar_vectors[0]``
          - ``parent_interframe.y``
        * - ``planar_vectors[1]``
          - ``parent_interframe.z``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    planar_coordinates : iterable of dynamicsymbols, optional
        Two generalized coordinates used for the planar translation. The default
        value is ``dynamicsymbols(f'q1_{joint.name} q2_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    planar_speeds : dynamicsymbols, optional
        Two generalized speeds used for the planar translation velocity. The
        default value is ``dynamicsymbols(f'u1_{joint.name} u2_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    rotation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the rotation angle.
    planar_coordinates : Matrix
        Two generalized coordinates used for the planar translation.
    rotation_speed : dynamicsymbol
        Generalized speed corresponding to the angular velocity.
    planar_speeds : Matrix
        Two generalized speeds used for the planar translation velocity.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.
    rotation_axis : Vector
        The axis about which the rotation occurs.
    planar_vectors : list
        The vectors that describe the planar translation directions.

    Examples
    =========

    A single planar joint is created between two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, PlanarJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PlanarJoint('PC', parent, child)
    >>> joint
    PlanarJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.rotation_axis
    P_frame.x
    >>> joint.planar_vectors
    [P_frame.y, P_frame.z]
    >>> joint.rotation_coordinate
    q0_PC(t)
    >>> joint.planar_coordinates
    Matrix([
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.rotation_speed
    u0_PC(t)
    >>> joint.planar_speeds
    Matrix([
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u0_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.y + q2_PC(t)*P_frame.z
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.y + u2_PC(t)*P_frame.z

    To further demonstrate the use of the planar joint, the kinematics of a
    block sliding on a slope, can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import PlanarJoint, RigidBody, ReferenceFrame
    >>> a, d, h = symbols('a d h')

    First create bodies to represent the slope and the block.

    >>> ground = RigidBody('G')
    >>> block = RigidBody('B')

    To define the slope you can either define the plane by specifying the
    ``planar_vectors`` or/and the ``rotation_axis``. However it is advisable to
    create a rotated intermediate frame, so that the ``parent_vectors`` and
    ``rotation_axis`` will be the unit vectors of this intermediate frame.

    >>> slope = ReferenceFrame('A')
    >>> slope.orient_axis(ground.frame, ground.y, a)

    The planar joint can be created using these bodies and intermediate frame.
    We can specify the origin of the slope to be ``d`` above the slope's center
    of mass and the block's center of mass to be a distance ``h`` above the
    slope's surface. Note that we can specify the normal of the plane using the
    rotation axis argument.

    >>> joint = PlanarJoint('PC', ground, block, parent_point=d * ground.x,
    ...                     child_point=-h * block.x, parent_interframe=slope)

    Once the joint is established the kinematics of the bodies can be accessed.
    First the ``rotation_axis``, which is normal to the plane and the
    ``plane_vectors``, can be found.

    >>> joint.rotation_axis
    A.x
    >>> joint.planar_vectors
    [A.y, A.z]

    The direction cosine matrix of the block with respect to the ground can be
    found with:

    >>> block.frame.dcm(ground.frame)
    Matrix([
    [              cos(a),              0,              -sin(a)],
    [sin(a)*sin(q0_PC(t)),  cos(q0_PC(t)), sin(q0_PC(t))*cos(a)],
    [sin(a)*cos(q0_PC(t)), -sin(q0_PC(t)), cos(a)*cos(q0_PC(t))]])

    The angular velocity of the block can be computed with respect to the
    ground.

    >>> block.frame.ang_vel_in(ground.frame)
    u0_PC(t)*A.x

    The position of the block's center of mass can be found with:

    >>> block.masscenter.pos_from(ground.masscenter)
    d*G_frame.x + h*B_frame.x + q1_PC(t)*A.y + q2_PC(t)*A.z

    Finally, the linear velocity of the block's center of mass can be
    computed with respect to the ground.

    >>> block.masscenter.vel(ground.frame)
    u1_PC(t)*A.y + u2_PC(t)*A.z

    In some cases it could be your preference to only define the normals of the
    plane with respect to both bodies. This can most easily be done by supplying
    vectors to the ``interframe`` arguments. What will happen in this case is
    that an interframe will be created with its ``x`` axis aligned with the
    provided vector. For a further explanation of how this is done see the notes
    of the ``Joint`` class. In the code below, the above example (with the block
    on the slope) is recreated by supplying vectors to the interframe arguments.
    Note that the previously described option is however more computationally
    efficient, because the algorithm now has to compute the rotation angle
    between the provided vector and the 'x' axis.

    >>> from sympy import symbols, cos, sin
    >>> from sympy.physics.mechanics import PlanarJoint, RigidBody
    >>> a, d, h = symbols('a d h')
    >>> ground = RigidBody('G')
    >>> block = RigidBody('B')
    >>> joint = PlanarJoint(
    ...     'PC', ground, block, parent_point=d * ground.x,
    ...     child_point=-h * block.x, child_interframe=block.x,
    ...     parent_interframe=cos(a) * ground.x + sin(a) * ground.z)
    >>> block.frame.dcm(ground.frame).simplify()
    Matrix([
    [               cos(a),              0,               sin(a)],
    [-sin(a)*sin(q0_PC(t)),  cos(q0_PC(t)), sin(q0_PC(t))*cos(a)],
    [-sin(a)*cos(q0_PC(t)), -sin(q0_PC(t)), cos(a)*cos(q0_PC(t))]])

    """

    def __init__(self, name, parent, child, rotation_coordinate=None,
                 planar_coordinates=None, rotation_speed=None,
                 planar_speeds=None, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None):
        # A ready to merge implementation of setting the planar_vectors and
        # rotation_axis was added and removed in PR #24046
        coordinates = (rotation_coordinate, planar_coordinates)
        speeds = (rotation_speed, planar_speeds)
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'PlanarJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    def planar_coordinates(self):
        """Two generalized coordinates used for the planar translation."""
        return self.coordinates[1:, 0]

    @property
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    def planar_speeds(self):
        """Two generalized speeds used for the planar translation velocity."""
        return self.speeds[1:, 0]

    @property
    def rotation_axis(self):
        """The axis about which the rotation occurs."""
        return self.parent_interframe.x

    @property
    def planar_vectors(self):
        """The vectors that describe the planar translation directions."""
        return [self.parent_interframe.y, self.parent_interframe.z]

    def _generate_coordinates(self, coordinates):
        rotation_speed = self._fill_coordinate_list(coordinates[0], 1, 'q',
                                                    number_single=True)
        planar_speeds = self._fill_coordinate_list(coordinates[1], 2, 'q', 1)
        return rotation_speed.col_join(planar_speeds)

    def _generate_speeds(self, speeds):
        rotation_speed = self._fill_coordinate_list(speeds[0], 1, 'u',
                                                    number_single=True)
        planar_speeds = self._fill_coordinate_list(speeds[1], 2, 'u', 1)
        return rotation_speed.col_join(planar_speeds)

    def _orient_frames(self):
        self.child_interframe.orient_axis(
            self.parent_interframe, self.rotation_axis,
            self.rotation_coordinate)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(
            self.parent_interframe,
            self.rotation_speed * self.rotation_axis)

    def _set_linear_velocity(self):
        self.child_point.set_pos(
            self.parent_point,
            self.planar_coordinates[0] * self.planar_vectors[0] +
            self.planar_coordinates[1] * self.planar_vectors[1])
        self.parent_point.set_vel(self.parent_interframe, 0)
        self.child_point.set_vel(self.child_interframe, 0)
        self.child_point.set_vel(
            self._parent_frame, self.planar_speeds[0] * self.planar_vectors[0] +
            self.planar_speeds[1] * self.planar_vectors[1])
        self.child.masscenter.v2pt_theory(self.child_point, self._parent_frame,
                                          self._child_frame)


class SphericalJoint(Joint):
    """Spherical (Ball-and-Socket) Joint.

    .. image:: SphericalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A spherical joint is defined such that the child body is free to rotate in
    any direction, without allowing a translation of the ``child_point``. As can
    also be seen in the image, the ``parent_point`` and ``child_point`` are
    fixed on top of each other, i.e. the ``joint_point``. This rotation is
    defined using the :func:`parent_interframe.orient(child_interframe,
    rot_type, amounts, rot_order)
    <sympy.physics.vector.frame.ReferenceFrame.orient>` method. The default
    rotation consists of three relative rotations, i.e. body-fixed rotations.
    Based on the direction cosine matrix following from these rotations, the
    angular velocity is computed based on the generalized coordinates and
    generalized speeds.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    coordinates: iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    rot_type : str, optional
        The method used to generate the direction cosine matrix. Supported
        methods are:

        - ``'Body'``: three successive rotations about new intermediate axes,
          also called "Euler and Tait-Bryan angles"
        - ``'Space'``: three successive rotations about the parent frames' unit
          vectors

        The default method is ``'Body'``.
    amounts :
        Expressions defining the rotation angles or direction cosine matrix.
        These must match the ``rot_type``. See examples below for details. The
        input types are:

        - ``'Body'``: 3-tuple of expressions, symbols, or functions
        - ``'Space'``: 3-tuple of expressions, symbols, or functions

        The default amounts are the given ``coordinates``.
    rot_order : str or int, optional
        If applicable, the order of the successive of rotations. The string
        ``'123'`` and integer ``123`` are equivalent, for example. Required for
        ``'Body'`` and ``'Space'``. The default value is ``123``.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single spherical joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, SphericalJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = SphericalJoint('PC', parent, child)
    >>> joint
    SphericalJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_interframe
    P_frame
    >>> joint.child_interframe
    C_frame
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame).to_matrix(child.frame)
    Matrix([
    [ u0_PC(t)*cos(q1_PC(t))*cos(q2_PC(t)) + u1_PC(t)*sin(q2_PC(t))],
    [-u0_PC(t)*sin(q2_PC(t))*cos(q1_PC(t)) + u1_PC(t)*cos(q2_PC(t))],
    [                             u0_PC(t)*sin(q1_PC(t)) + u2_PC(t)]])
    >>> child.frame.x.to_matrix(parent.frame)
    Matrix([
    [                                            cos(q1_PC(t))*cos(q2_PC(t))],
    [sin(q0_PC(t))*sin(q1_PC(t))*cos(q2_PC(t)) + sin(q2_PC(t))*cos(q0_PC(t))],
    [sin(q0_PC(t))*sin(q2_PC(t)) - sin(q1_PC(t))*cos(q0_PC(t))*cos(q2_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the spherical joint, the kinematics of a
    spherical joint with a ZXZ rotation can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import RigidBody, SphericalJoint
    >>> l1 = symbols('l1')

    First create bodies to represent the fixed floor and a pendulum bob.

    >>> floor = RigidBody('F')
    >>> bob = RigidBody('B')

    The joint will connect the bob to the floor, with the joint located at a
    distance of ``l1`` from the child's center of mass and the rotation set to a
    body-fixed ZXZ rotation.

    >>> joint = SphericalJoint('S', floor, bob, child_point=l1 * bob.y,
    ...                        rot_type='body', rot_order='ZXZ')

    Now that the joint is established, the kinematics of the connected body can
    be accessed.

    The position of the bob's masscenter is found with:

    >>> bob.masscenter.pos_from(floor.masscenter)
    - l1*B_frame.y

    The angular velocities of the pendulum link can be computed with respect to
    the floor.

    >>> bob.frame.ang_vel_in(floor.frame).to_matrix(
    ...     floor.frame).simplify()
    Matrix([
    [u1_S(t)*cos(q0_S(t)) + u2_S(t)*sin(q0_S(t))*sin(q1_S(t))],
    [u1_S(t)*sin(q0_S(t)) - u2_S(t)*sin(q1_S(t))*cos(q0_S(t))],
    [                          u0_S(t) + u2_S(t)*cos(q1_S(t))]])

    Finally, the linear velocity of the bob's center of mass can be computed.

    >>> bob.masscenter.vel(floor.frame).to_matrix(bob.frame)
    Matrix([
    [                           l1*(u0_S(t)*cos(q1_S(t)) + u2_S(t))],
    [                                                             0],
    [-l1*(u0_S(t)*sin(q1_S(t))*sin(q2_S(t)) + u1_S(t)*cos(q2_S(t)))]])

    """
    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, rot_type='BODY', amounts=None,
                 rot_order=123):
        self._rot_type = rot_type
        self._amounts = amounts
        self._rot_order = rot_order
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'SphericalJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 3, 'q')

    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, len(self.coordinates), 'u')

    def _orient_frames(self):
        supported_rot_types = ('BODY', 'SPACE')
        if self._rot_type.upper() not in supported_rot_types:
            raise NotImplementedError(
                f'Rotation type "{self._rot_type}" is not implemented. '
                f'Implemented rotation types are: {supported_rot_types}')
        amounts = self.coordinates if self._amounts is None else self._amounts
        self.child_interframe.orient(self.parent_interframe, self._rot_type,
                                     amounts, self._rot_order)

    def _set_angular_velocity(self):
        t = dynamicsymbols._t
        vel = self.child_interframe.ang_vel_in(self.parent_interframe).xreplace(
            {q.diff(t): u for q, u in zip(self.coordinates, self.speeds)}
        )
        self.child_interframe.set_ang_vel(self.parent_interframe, vel)

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.v2pt_theory(self.parent_point, self._parent_frame,
                                          self._child_frame)


class WeldJoint(Joint):
    """Weld Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/WeldJoint.svg

    Explanation
    ===========

    A weld joint is defined such that there is no relative motion between the
    child and parent bodies. The direction cosine matrix between the attachment
    frame (``parent_interframe`` and ``child_interframe``) is the identity
    matrix and the attachment points (``parent_point`` and ``child_point``) are
    coincident. The page on the joints framework gives a more detailed
    explanation of the intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : Matrix
        Matrix of the joint's generalized speeds. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single weld joint is created from two bodies and has the following basic
    attributes:

    >>> from sympy.physics.mechanics import RigidBody, WeldJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = WeldJoint('PC', parent, child)
    >>> joint
    WeldJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.coordinates
    Matrix(0, 0, [])
    >>> joint.speeds
    Matrix(0, 0, [])
    >>> child.frame.ang_vel_in(parent.frame)
    0
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the weld joint, two relatively-fixed
    bodies rotated by a quarter turn about the Y axis can be created as follows:

    >>> from sympy import symbols, pi
    >>> from sympy.physics.mechanics import ReferenceFrame, RigidBody, WeldJoint
    >>> l1, l2 = symbols('l1 l2')

    First create the bodies to represent the parent and rotated child body.

    >>> parent = RigidBody('P')
    >>> child = RigidBody('C')

    Next the intermediate frame specifying the fixed rotation with respect to
    the parent can be created.

    >>> rotated_frame = ReferenceFrame('Pr')
    >>> rotated_frame.orient_axis(parent.frame, parent.y, pi / 2)

    The weld between the parent body and child body is located at a distance
    ``l1`` from the parent's center of mass in the X direction and ``l2`` from
    the child's center of mass in the child's negative X direction.

    >>> weld = WeldJoint('weld', parent, child, parent_point=l1 * parent.x,
    ...                  child_point=-l2 * child.x,
    ...                  parent_interframe=rotated_frame)

    Now that the joint has been established, the kinematics of the bodies can be
    accessed. The direction cosine matrix of the child body with respect to the
    parent can be found:

    >>> child.frame.dcm(parent.frame)
    Matrix([
    [0, 0, -1],
    [0, 1,  0],
    [1, 0,  0]])

    As can also been seen from the direction cosine matrix, the parent X axis is
    aligned with the child's Z axis:
    >>> parent.x == child.z
    True

    The position of the child's center of mass with respect to the parent's
    center of mass can be found with:

    >>> child.masscenter.pos_from(parent.masscenter)
    l1*P_frame.x + l2*C_frame.x

    The angular velocity of the child with respect to the parent is 0 as one
    would expect.

    >>> child.frame.ang_vel_in(parent.frame)
    0

    """

    def __init__(self, name, parent, child, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None):
        super().__init__(name, parent, child, [], [], parent_point,
                         child_point, parent_interframe=parent_interframe,
                         child_interframe=child_interframe)
        self._kdes = Matrix(1, 0, []).T  # Removes stackability problems #10770

    def __str__(self):
        return (f'WeldJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    def _generate_coordinates(self, coordinate):
        return Matrix()

    def _generate_speeds(self, speed):
        return Matrix()

    def _orient_frames(self):
        self.child_interframe.orient_axis(self.parent_interframe,
                                          self.parent_interframe.x, 0)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, 0)

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.set_vel(self._parent_frame, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\decl_base.py ===
# orm/decl_base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Internal implementation for declarative."""

from __future__ import annotations

import collections
import dataclasses
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import attributes
from . import clsregistry
from . import exc as orm_exc
from . import instrumentation
from . import mapperlib
from ._typing import _O
from ._typing import attr_is_internal_proxy
from .attributes import InstrumentedAttribute
from .attributes import QueryableAttribute
from .base import _is_mapped_class
from .base import InspectionAttr
from .descriptor_props import CompositeProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DCAttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MappedAttribute
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .mapper import Mapper
from .properties import ColumnProperty
from .properties import MappedColumn
from .util import _extract_mapped_subtype
from .util import _is_mapped_annotation
from .util import class_mapper
from .util import de_stringify_annotation
from .. import event
from .. import exc
from .. import util
from ..sql import expression
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import Table
from ..util import topological
from ..util.typing import _AnnotationScanType
from ..util.typing import get_args
from ..util.typing import is_fwd_ref
from ..util.typing import is_literal
from ..util.typing import Protocol
from ..util.typing import TypedDict

if TYPE_CHECKING:
    from ._typing import _ClassDict
    from ._typing import _RegistryType
    from .base import Mapped
    from .decl_api import declared_attr
    from .instrumentation import ClassManager
    from ..sql.elements import NamedColumn
    from ..sql.schema import MetaData
    from ..sql.selectable import FromClause

_T = TypeVar("_T", bound=Any)

_MapperKwArgs = Mapping[str, Any]
_TableArgsType = Union[Tuple[Any, ...], Dict[str, Any]]


class MappedClassProtocol(Protocol[_O]):
    """A protocol representing a SQLAlchemy mapped class.

    The protocol is generic on the type of class, use
    ``MappedClassProtocol[Any]`` to allow any mapped class.
    """

    __name__: str
    __mapper__: Mapper[_O]
    __table__: FromClause

    def __call__(self, **kw: Any) -> _O: ...


class _DeclMappedClassProtocol(MappedClassProtocol[_O], Protocol):
    "Internal more detailed version of ``MappedClassProtocol``."
    metadata: MetaData
    __tablename__: str
    __mapper_args__: _MapperKwArgs
    __table_args__: Optional[_TableArgsType]

    _sa_apply_dc_transforms: Optional[_DataclassArguments]

    def __declare_first__(self) -> None: ...

    def __declare_last__(self) -> None: ...


class _DataclassArguments(TypedDict):
    init: Union[_NoArg, bool]
    repr: Union[_NoArg, bool]
    eq: Union[_NoArg, bool]
    order: Union[_NoArg, bool]
    unsafe_hash: Union[_NoArg, bool]
    match_args: Union[_NoArg, bool]
    kw_only: Union[_NoArg, bool]
    dataclass_callable: Union[_NoArg, Callable[..., Type[Any]]]


def _declared_mapping_info(
    cls: Type[Any],
) -> Optional[Union[_DeferredMapperConfig, Mapper[Any]]]:
    # deferred mapping
    if _DeferredMapperConfig.has_cls(cls):
        return _DeferredMapperConfig.config_for_cls(cls)
    # regular mapping
    elif _is_mapped_class(cls):
        return class_mapper(cls, configure=False)
    else:
        return None


def _is_supercls_for_inherits(cls: Type[Any]) -> bool:
    """return True if this class will be used as a superclass to set in
    'inherits'.

    This includes deferred mapper configs that aren't mapped yet, however does
    not include classes with _sa_decl_prepare_nocascade (e.g.
    ``AbstractConcreteBase``); these concrete-only classes are not set up as
    "inherits" until after mappers are configured using
    mapper._set_concrete_base()

    """
    if _DeferredMapperConfig.has_cls(cls):
        return not _get_immediate_cls_attr(
            cls, "_sa_decl_prepare_nocascade", strict=True
        )
    # regular mapping
    elif _is_mapped_class(cls):
        return True
    else:
        return False


def _resolve_for_abstract_or_classical(cls: Type[Any]) -> Optional[Type[Any]]:
    if cls is object:
        return None

    sup: Optional[Type[Any]]

    if cls.__dict__.get("__abstract__", False):
        for base_ in cls.__bases__:
            sup = _resolve_for_abstract_or_classical(base_)
            if sup is not None:
                return sup
        else:
            return None
    else:
        clsmanager = _dive_for_cls_manager(cls)

        if clsmanager:
            return clsmanager.class_
        else:
            return cls


def _get_immediate_cls_attr(
    cls: Type[Any], attrname: str, strict: bool = False
) -> Optional[Any]:
    """return an attribute of the class that is either present directly
    on the class, e.g. not on a superclass, or is from a superclass but
    this superclass is a non-mapped mixin, that is, not a descendant of
    the declarative base and is also not classically mapped.

    This is used to detect attributes that indicate something about
    a mapped class independently from any mapped classes that it may
    inherit from.

    """

    # the rules are different for this name than others,
    # make sure we've moved it out.  transitional
    assert attrname != "__abstract__"

    if not issubclass(cls, object):
        return None

    if attrname in cls.__dict__:
        return getattr(cls, attrname)

    for base in cls.__mro__[1:]:
        _is_classical_inherits = _dive_for_cls_manager(base) is not None

        if attrname in base.__dict__ and (
            base is cls
            or (
                (base in cls.__bases__ if strict else True)
                and not _is_classical_inherits
            )
        ):
            return getattr(base, attrname)
    else:
        return None


def _dive_for_cls_manager(cls: Type[_O]) -> Optional[ClassManager[_O]]:
    # because the class manager registration is pluggable,
    # we need to do the search for every class in the hierarchy,
    # rather than just a simple "cls._sa_class_manager"

    for base in cls.__mro__:
        manager: Optional[ClassManager[_O]] = attributes.opt_manager_of_class(
            base
        )
        if manager:
            return manager
    return None


def _as_declarative(
    registry: _RegistryType, cls: Type[Any], dict_: _ClassDict
) -> Optional[_MapperConfig]:
    # declarative scans the class for attributes.  no table or mapper
    # args passed separately.
    return _MapperConfig.setup_mapping(registry, cls, dict_, None, {})


def _mapper(
    registry: _RegistryType,
    cls: Type[_O],
    table: Optional[FromClause],
    mapper_kw: _MapperKwArgs,
) -> Mapper[_O]:
    _ImperativeMapperConfig(registry, cls, table, mapper_kw)
    return cast("MappedClassProtocol[_O]", cls).__mapper__


@util.preload_module("sqlalchemy.orm.decl_api")
def _is_declarative_props(obj: Any) -> bool:
    _declared_attr_common = util.preloaded.orm_decl_api._declared_attr_common

    return isinstance(obj, (_declared_attr_common, util.classproperty))


def _check_declared_props_nocascade(
    obj: Any, name: str, cls: Type[_O]
) -> bool:
    if _is_declarative_props(obj):
        if getattr(obj, "_cascading", False):
            util.warn(
                "@declared_attr.cascading is not supported on the %s "
                "attribute on class %s.  This attribute invokes for "
                "subclasses in any case." % (name, cls)
            )
        return True
    else:
        return False


class _MapperConfig:
    __slots__ = (
        "cls",
        "classname",
        "properties",
        "declared_attr_reg",
        "__weakref__",
    )

    cls: Type[Any]
    classname: str
    properties: util.OrderedDict[
        str,
        Union[
            Sequence[NamedColumn[Any]], NamedColumn[Any], MapperProperty[Any]
        ],
    ]
    declared_attr_reg: Dict[declared_attr[Any], Any]

    @classmethod
    def setup_mapping(
        cls,
        registry: _RegistryType,
        cls_: Type[_O],
        dict_: _ClassDict,
        table: Optional[FromClause],
        mapper_kw: _MapperKwArgs,
    ) -> Optional[_MapperConfig]:
        manager = attributes.opt_manager_of_class(cls)
        if manager and manager.class_ is cls_:
            raise exc.InvalidRequestError(
                f"Class {cls!r} already has been instrumented declaratively"
            )

        if cls_.__dict__.get("__abstract__", False):
            return None

        defer_map = _get_immediate_cls_attr(
            cls_, "_sa_decl_prepare_nocascade", strict=True
        ) or hasattr(cls_, "_sa_decl_prepare")

        if defer_map:
            return _DeferredMapperConfig(
                registry, cls_, dict_, table, mapper_kw
            )
        else:
            return _ClassScanMapperConfig(
                registry, cls_, dict_, table, mapper_kw
            )

    def __init__(
        self,
        registry: _RegistryType,
        cls_: Type[Any],
        mapper_kw: _MapperKwArgs,
    ):
        self.cls = util.assert_arg_type(cls_, type, "cls_")
        self.classname = cls_.__name__
        self.properties = util.OrderedDict()
        self.declared_attr_reg = {}

        if not mapper_kw.get("non_primary", False):
            instrumentation.register_class(
                self.cls,
                finalize=False,
                registry=registry,
                declarative_scan=self,
                init_method=registry.constructor,
            )
        else:
            manager = attributes.opt_manager_of_class(self.cls)
            if not manager or not manager.is_mapped:
                raise exc.InvalidRequestError(
                    "Class %s has no primary mapper configured.  Configure "
                    "a primary mapper first before setting up a non primary "
                    "Mapper." % self.cls
                )

    def set_cls_attribute(self, attrname: str, value: _T) -> _T:
        manager = instrumentation.manager_of_class(self.cls)
        manager.install_member(attrname, value)
        return value

    def map(self, mapper_kw: _MapperKwArgs = ...) -> Mapper[Any]:
        raise NotImplementedError()

    def _early_mapping(self, mapper_kw: _MapperKwArgs) -> None:
        self.map(mapper_kw)


class _ImperativeMapperConfig(_MapperConfig):
    __slots__ = ("local_table", "inherits")

    def __init__(
        self,
        registry: _RegistryType,
        cls_: Type[_O],
        table: Optional[FromClause],
        mapper_kw: _MapperKwArgs,
    ):
        super().__init__(registry, cls_, mapper_kw)

        self.local_table = self.set_cls_attribute("__table__", table)

        with mapperlib._CONFIGURE_MUTEX:
            if not mapper_kw.get("non_primary", False):
                clsregistry.add_class(
                    self.classname, self.cls, registry._class_registry
                )

            self._setup_inheritance(mapper_kw)

            self._early_mapping(mapper_kw)

    def map(self, mapper_kw: _MapperKwArgs = util.EMPTY_DICT) -> Mapper[Any]:
        mapper_cls = Mapper

        return self.set_cls_attribute(
            "__mapper__",
            mapper_cls(self.cls, self.local_table, **mapper_kw),
        )

    def _setup_inheritance(self, mapper_kw: _MapperKwArgs) -> None:
        cls = self.cls

        inherits = mapper_kw.get("inherits", None)

        if inherits is None:
            # since we search for classical mappings now, search for
            # multiple mapped bases as well and raise an error.
            inherits_search = []
            for base_ in cls.__bases__:
                c = _resolve_for_abstract_or_classical(base_)
                if c is None:
                    continue

                if _is_supercls_for_inherits(c) and c not in inherits_search:
                    inherits_search.append(c)

            if inherits_search:
                if len(inherits_search) > 1:
                    raise exc.InvalidRequestError(
                        "Class %s has multiple mapped bases: %r"
                        % (cls, inherits_search)
                    )
                inherits = inherits_search[0]
        elif isinstance(inherits, Mapper):
            inherits = inherits.class_

        self.inherits = inherits


class _CollectedAnnotation(NamedTuple):
    raw_annotation: _AnnotationScanType
    mapped_container: Optional[Type[Mapped[Any]]]
    extracted_mapped_annotation: Union[_AnnotationScanType, str]
    is_dataclass: bool
    attr_value: Any
    originating_module: str
    originating_class: Type[Any]


class _ClassScanMapperConfig(_MapperConfig):
    __slots__ = (
        "registry",
        "clsdict_view",
        "collected_attributes",
        "collected_annotations",
        "local_table",
        "persist_selectable",
        "declared_columns",
        "column_ordering",
        "column_copies",
        "table_args",
        "tablename",
        "mapper_args",
        "mapper_args_fn",
        "table_fn",
        "inherits",
        "single",
        "allow_dataclass_fields",
        "dataclass_setup_arguments",
        "is_dataclass_prior_to_mapping",
        "allow_unmapped_annotations",
    )

    is_deferred = False
    registry: _RegistryType
    clsdict_view: _ClassDict
    collected_annotations: Dict[str, _CollectedAnnotation]
    collected_attributes: Dict[str, Any]
    local_table: Optional[FromClause]
    persist_selectable: Optional[FromClause]
    declared_columns: util.OrderedSet[Column[Any]]
    column_ordering: Dict[Column[Any], int]
    column_copies: Dict[
        Union[MappedColumn[Any], Column[Any]],
        Union[MappedColumn[Any], Column[Any]],
    ]
    tablename: Optional[str]
    mapper_args: Mapping[str, Any]
    table_args: Optional[_TableArgsType]
    mapper_args_fn: Optional[Callable[[], Dict[str, Any]]]
    inherits: Optional[Type[Any]]
    single: bool

    is_dataclass_prior_to_mapping: bool
    allow_unmapped_annotations: bool

    dataclass_setup_arguments: Optional[_DataclassArguments]
    """if the class has SQLAlchemy native dataclass parameters, where
    we will turn the class into a dataclass within the declarative mapping
    process.

    """

    allow_dataclass_fields: bool
    """if true, look for dataclass-processed Field objects on the target
    class as well as superclasses and extract ORM mapping directives from
    the "metadata" attribute of each Field.

    if False, dataclass fields can still be used, however they won't be
    mapped.

    """

    def __init__(
        self,
        registry: _RegistryType,
        cls_: Type[_O],
        dict_: _ClassDict,
        table: Optional[FromClause],
        mapper_kw: _MapperKwArgs,
    ):
        # grab class dict before the instrumentation manager has been added.
        # reduces cycles
        self.clsdict_view = (
            util.immutabledict(dict_) if dict_ else util.EMPTY_DICT
        )
        super().__init__(registry, cls_, mapper_kw)
        self.registry = registry
        self.persist_selectable = None

        self.collected_attributes = {}
        self.collected_annotations = {}
        self.declared_columns = util.OrderedSet()
        self.column_ordering = {}
        self.column_copies = {}
        self.single = False
        self.dataclass_setup_arguments = dca = getattr(
            self.cls, "_sa_apply_dc_transforms", None
        )

        self.allow_unmapped_annotations = getattr(
            self.cls, "__allow_unmapped__", False
        ) or bool(self.dataclass_setup_arguments)

        self.is_dataclass_prior_to_mapping = cld = dataclasses.is_dataclass(
            cls_
        )

        sdk = _get_immediate_cls_attr(cls_, "__sa_dataclass_metadata_key__")

        # we don't want to consume Field objects from a not-already-dataclass.
        # the Field objects won't have their "name" or "type" populated,
        # and while it seems like we could just set these on Field as we
        # read them, Field is documented as "user read only" and we need to
        # stay far away from any off-label use of dataclasses APIs.
        if (not cld or dca) and sdk:
            raise exc.InvalidRequestError(
                "SQLAlchemy mapped dataclasses can't consume mapping "
                "information from dataclass.Field() objects if the immediate "
                "class is not already a dataclass."
            )

        # if already a dataclass, and __sa_dataclass_metadata_key__ present,
        # then also look inside of dataclass.Field() objects yielded by
        # dataclasses.get_fields(cls) when scanning for attributes
        self.allow_dataclass_fields = bool(sdk and cld)

        self._setup_declared_events()

        self._scan_attributes()

        self._setup_dataclasses_transforms()

        with mapperlib._CONFIGURE_MUTEX:
            clsregistry.add_class(
                self.classname, self.cls, registry._class_registry
            )

            self._setup_inheriting_mapper(mapper_kw)

            self._extract_mappable_attributes()

            self._extract_declared_columns()

            self._setup_table(table)

            self._setup_inheriting_columns(mapper_kw)

            self._early_mapping(mapper_kw)

    def _setup_declared_events(self) -> None:
        if _get_immediate_cls_attr(self.cls, "__declare_last__"):

            @event.listens_for(Mapper, "after_configured")
            def after_configured() -> None:
                cast(
                    "_DeclMappedClassProtocol[Any]", self.cls
                ).__declare_last__()

        if _get_immediate_cls_attr(self.cls, "__declare_first__"):

            @event.listens_for(Mapper, "before_configured")
            def before_configured() -> None:
                cast(
                    "_DeclMappedClassProtocol[Any]", self.cls
                ).__declare_first__()

    def _cls_attr_override_checker(
        self, cls: Type[_O]
    ) -> Callable[[str, Any], bool]:
        """Produce a function that checks if a class has overridden an
        attribute, taking SQLAlchemy-enabled dataclass fields into account.

        """

        if self.allow_dataclass_fields:
            sa_dataclass_metadata_key = _get_immediate_cls_attr(
                cls, "__sa_dataclass_metadata_key__"
            )
        else:
            sa_dataclass_metadata_key = None

        if not sa_dataclass_metadata_key:

            def attribute_is_overridden(key: str, obj: Any) -> bool:
                return getattr(cls, key, obj) is not obj

        else:
            all_datacls_fields = {
                f.name: f.metadata[sa_dataclass_metadata_key]
                for f in util.dataclass_fields(cls)
                if sa_dataclass_metadata_key in f.metadata
            }
            local_datacls_fields = {
                f.name: f.metadata[sa_dataclass_metadata_key]
                for f in util.local_dataclass_fields(cls)
                if sa_dataclass_metadata_key in f.metadata
            }

            absent = object()

            def attribute_is_overridden(key: str, obj: Any) -> bool:
                if _is_declarative_props(obj):
                    obj = obj.fget

                # this function likely has some failure modes still if
                # someone is doing a deep mixing of the same attribute
                # name as plain Python attribute vs. dataclass field.

                ret = local_datacls_fields.get(key, absent)
                if _is_declarative_props(ret):
                    ret = ret.fget

                if ret is obj:
                    return False
                elif ret is not absent:
                    return True

                all_field = all_datacls_fields.get(key, absent)

                ret = getattr(cls, key, obj)

                if ret is obj:
                    return False

                # for dataclasses, this could be the
                # 'default' of the field.  so filter more specifically
                # for an already-mapped InstrumentedAttribute
                if ret is not absent and isinstance(
                    ret, InstrumentedAttribute
                ):
                    return True

                if all_field is obj:
                    return False
                elif all_field is not absent:
                    return True

                # can't find another attribute
                return False

        return attribute_is_overridden

    _include_dunders = {
        "__table__",
        "__mapper_args__",
        "__tablename__",
        "__table_args__",
    }

    _match_exclude_dunders = re.compile(r"^(?:_sa_|__)")

    def _cls_attr_resolver(
        self, cls: Type[Any]
    ) -> Callable[[], Iterable[Tuple[str, Any, Any, bool]]]:
        """produce a function to iterate the "attributes" of a class
        which we want to consider for mapping, adjusting for SQLAlchemy fields
        embedded in dataclass fields.

        """
        cls_annotations = util.get_annotations(cls)

        cls_vars = vars(cls)

        _include_dunders = self._include_dunders
        _match_exclude_dunders = self._match_exclude_dunders

        names = [
            n
            for n in util.merge_lists_w_ordering(
                list(cls_vars), list(cls_annotations)
            )
            if not _match_exclude_dunders.match(n) or n in _include_dunders
        ]

        if self.allow_dataclass_fields:
            sa_dataclass_metadata_key: Optional[str] = _get_immediate_cls_attr(
                cls, "__sa_dataclass_metadata_key__"
            )
        else:
            sa_dataclass_metadata_key = None

        if not sa_dataclass_metadata_key:

            def local_attributes_for_class() -> (
                Iterable[Tuple[str, Any, Any, bool]]
            ):
                return (
                    (
                        name,
                        cls_vars.get(name),
                        cls_annotations.get(name),
                        False,
                    )
                    for name in names
                )

        else:
            dataclass_fields = {
                field.name: field for field in util.local_dataclass_fields(cls)
            }

            fixed_sa_dataclass_metadata_key = sa_dataclass_metadata_key

            def local_attributes_for_class() -> (
                Iterable[Tuple[str, Any, Any, bool]]
            ):
                for name in names:
                    field = dataclass_fields.get(name, None)
                    if field and sa_dataclass_metadata_key in field.metadata:
                        yield field.name, _as_dc_declaredattr(
                            field.metadata, fixed_sa_dataclass_metadata_key
                        ), cls_annotations.get(field.name), True
                    else:
                        yield name, cls_vars.get(name), cls_annotations.get(
                            name
                        ), False

        return local_attributes_for_class

    def _scan_attributes(self) -> None:
        cls = self.cls

        cls_as_Decl = cast("_DeclMappedClassProtocol[Any]", cls)

        clsdict_view = self.clsdict_view
        collected_attributes = self.collected_attributes
        column_copies = self.column_copies
        _include_dunders = self._include_dunders
        mapper_args_fn = None
        table_args = inherited_table_args = None
        table_fn = None
        tablename = None
        fixed_table = "__table__" in clsdict_view

        attribute_is_overridden = self._cls_attr_override_checker(self.cls)

        bases = []

        for base in cls.__mro__:
            # collect bases and make sure standalone columns are copied
            # to be the column they will ultimately be on the class,
            # so that declared_attr functions use the right columns.
            # need to do this all the way up the hierarchy first
            # (see #8190)

            class_mapped = base is not cls and _is_supercls_for_inherits(base)

            local_attributes_for_class = self._cls_attr_resolver(base)

            if not class_mapped and base is not cls:
                locally_collected_columns = self._produce_column_copies(
                    local_attributes_for_class,
                    attribute_is_overridden,
                    fixed_table,
                    base,
                )
            else:
                locally_collected_columns = {}

            bases.append(
                (
                    base,
                    class_mapped,
                    local_attributes_for_class,
                    locally_collected_columns,
                )
            )

        for (
            base,
            class_mapped,
            local_attributes_for_class,
            locally_collected_columns,
        ) in bases:
            # this transfer can also take place as we scan each name
            # for finer-grained control of how collected_attributes is
            # populated, as this is what impacts column ordering.
            # however it's simpler to get it out of the way here.
            collected_attributes.update(locally_collected_columns)

            for (
                name,
                obj,
                annotation,
                is_dataclass_field,
            ) in local_attributes_for_class():
                if name in _include_dunders:
                    if name == "__mapper_args__":
                        check_decl = _check_declared_props_nocascade(
                            obj, name, cls
                        )
                        if not mapper_args_fn and (
                            not class_mapped or check_decl
                        ):
                            # don't even invoke __mapper_args__ until
                            # after we've determined everything about the
                            # mapped table.
                            # make a copy of it so a class-level dictionary
                            # is not overwritten when we update column-based
                            # arguments.
                            def _mapper_args_fn() -> Dict[str, Any]:
                                return dict(cls_as_Decl.__mapper_args__)

                            mapper_args_fn = _mapper_args_fn

                    elif name == "__tablename__":
                        check_decl = _check_declared_props_nocascade(
                            obj, name, cls
                        )
                        if not tablename and (not class_mapped or check_decl):
                            tablename = cls_as_Decl.__tablename__
                    elif name == "__table__":
                        check_decl = _check_declared_props_nocascade(
                            obj, name, cls
                        )
                        # if a @declared_attr using "__table__" is detected,
                        # wrap up a callable to look for "__table__" from
                        # the final concrete class when we set up a table.
                        # this was fixed by
                        # #11509, regression in 2.0 from version 1.4.
                        if check_decl and not table_fn:
                            # don't even invoke __table__ until we're ready
                            def _table_fn() -> FromClause:
                                return cls_as_Decl.__table__

                            table_fn = _table_fn

                    elif name == "__table_args__":
                        check_decl = _check_declared_props_nocascade(
                            obj, name, cls
                        )
                        if not table_args and (not class_mapped or check_decl):
                            table_args = cls_as_Decl.__table_args__
                            if not isinstance(
                                table_args, (tuple, dict, type(None))
                            ):
                                raise exc.ArgumentError(
                                    "__table_args__ value must be a tuple, "
                                    "dict, or None"
                                )
                            if base is not cls:
                                inherited_table_args = True
                    else:
                        # any other dunder names; should not be here
                        # as we have tested for all four names in
                        # _include_dunders
                        assert False
                elif class_mapped:
                    if _is_declarative_props(obj) and not obj._quiet:
                        util.warn(
                            "Regular (i.e. not __special__) "
                            "attribute '%s.%s' uses @declared_attr, "
                            "but owning class %s is mapped - "
                            "not applying to subclass %s."
                            % (base.__name__, name, base, cls)
                        )

                    continue
                elif base is not cls:
                    # we're a mixin, abstract base, or something that is
                    # acting like that for now.

                    if isinstance(obj, (Column, MappedColumn)):
                        # already copied columns to the mapped class.
                        continue
                    elif isinstance(obj, MapperProperty):
                        raise exc.InvalidRequestError(
                            "Mapper properties (i.e. deferred,"
                            "column_property(), relationship(), etc.) must "
                            "be declared as @declared_attr callables "
                            "on declarative mixin classes.  For dataclass "
                            "field() objects, use a lambda:"
                        )
                    elif _is_declarative_props(obj):
                        # tried to get overloads to tell this to
                        # pylance, no luck
                        assert obj is not None

                        if obj._cascading:
                            if name in clsdict_view:
                                # unfortunately, while we can use the user-
                                # defined attribute here to allow a clean
                                # override, if there's another
                                # subclass below then it still tries to use
                                # this.  not sure if there is enough
                                # information here to add this as a feature
                                # later on.
                                util.warn(
                                    "Attribute '%s' on class %s cannot be "
                                    "processed due to "
                                    "@declared_attr.cascading; "
                                    "skipping" % (name, cls)
                                )
                            collected_attributes[name] = column_copies[obj] = (
                                ret
                            ) = obj.__get__(obj, cls)
                            setattr(cls, name, ret)
                        else:
                            if is_dataclass_field:
                                # access attribute using normal class access
                                # first, to see if it's been mapped on a
                                # superclass.   note if the dataclasses.field()
                                # has "default", this value can be anything.
                                ret = getattr(cls, name, None)

                                # so, if it's anything that's not ORM
                                # mapped, assume we should invoke the
                                # declared_attr
                                if not isinstance(ret, InspectionAttr):
                                    ret = obj.fget()
                            else:
                                # access attribute using normal class access.
                                # if the declared attr already took place
                                # on a superclass that is mapped, then
                                # this is no longer a declared_attr, it will
                                # be the InstrumentedAttribute
                                ret = getattr(cls, name)

                            # correct for proxies created from hybrid_property
                            # or similar.  note there is no known case that
                            # produces nested proxies, so we are only
                            # looking one level deep right now.

                            if (
                                isinstance(ret, InspectionAttr)
                                and attr_is_internal_proxy(ret)
                                and not isinstance(
                                    ret.original_property, MapperProperty
                                )
                            ):
                                ret = ret.descriptor

                            collected_attributes[name] = column_copies[obj] = (
                                ret
                            )

                        if (
                            isinstance(ret, (Column, MapperProperty))
                            and ret.doc is None
                        ):
                            ret.doc = obj.__doc__

                        self._collect_annotation(
                            name,
                            obj._collect_return_annotation(),
                            base,
                            True,
                            obj,
                        )
                    elif _is_mapped_annotation(annotation, cls, base):
                        # Mapped annotation without any object.
                        # product_column_copies should have handled this.
                        # if future support for other MapperProperty,
                        # then test if this name is already handled and
                        # otherwise proceed to generate.
                        if not fixed_table:
                            assert (
                                name in collected_attributes
                                or attribute_is_overridden(name, None)
                            )
                        continue
                    else:
                        # here, the attribute is some other kind of
                        # property that we assume is not part of the
                        # declarative mapping.  however, check for some
                        # more common mistakes
                        self._warn_for_decl_attributes(base, name, obj)
                elif is_dataclass_field and (
                    name not in clsdict_view or clsdict_view[name] is not obj
                ):
                    # here, we are definitely looking at the target class
                    # and not a superclass.   this is currently a
                    # dataclass-only path.  if the name is only
                    # a dataclass field and isn't in local cls.__dict__,
                    # put the object there.
                    # assert that the dataclass-enabled resolver agrees
                    # with what we are seeing

                    assert not attribute_is_overridden(name, obj)

                    if _is_declarative_props(obj):
                        obj = obj.fget()

                    collected_attributes[name] = obj
                    self._collect_annotation(
                        name, annotation, base, False, obj
                    )
                else:
                    collected_annotation = self._collect_annotation(
                        name, annotation, base, None, obj
                    )
                    is_mapped = (
                        collected_annotation is not None
                        and collected_annotation.mapped_container is not None
                    )
                    generated_obj = (
                        collected_annotation.attr_value
                        if collected_annotation is not None
                        else obj
                    )
                    if obj is None and not fixed_table and is_mapped:
                        collected_attributes[name] = (
                            generated_obj
                            if generated_obj is not None
                            else MappedColumn()
                        )
                    elif name in clsdict_view:
                        collected_attributes[name] = obj
                    # else if the name is not in the cls.__dict__,
                    # don't collect it as an attribute.
                    # we will see the annotation only, which is meaningful
                    # both for mapping and dataclasses setup

        if inherited_table_args and not tablename:
            table_args = None

        self.table_args = table_args
        self.tablename = tablename
        self.mapper_args_fn = mapper_args_fn
        self.table_fn = table_fn

    def _setup_dataclasses_transforms(self) -> None:
        dataclass_setup_arguments = self.dataclass_setup_arguments
        if not dataclass_setup_arguments:
            return

        # can't use is_dataclass since it uses hasattr
        if "__dataclass_fields__" in self.cls.__dict__:
            raise exc.InvalidRequestError(
                f"Class {self.cls} is already a dataclass; ensure that "
                "base classes / decorator styles of establishing dataclasses "
                "are not being mixed. "
                "This can happen if a class that inherits from "
                "'MappedAsDataclass', even indirectly, is been mapped with "
                "'@registry.mapped_as_dataclass'"
            )

        # can't create a dataclass if __table__ is already there. This would
        # fail an assertion when calling _get_arguments_for_make_dataclass:
        # assert False, "Mapped[] received without a mapping declaration"
        if "__table__" in self.cls.__dict__:
            raise exc.InvalidRequestError(
                f"Class {self.cls} already defines a '__table__'. "
                "ORM Annotated Dataclasses do not support a pre-existing "
                "'__table__' element"
            )

        warn_for_non_dc_attrs = collections.defaultdict(list)

        def _allow_dataclass_field(
            key: str, originating_class: Type[Any]
        ) -> bool:
            if (
                originating_class is not self.cls
                and "__dataclass_fields__" not in originating_class.__dict__
            ):
                warn_for_non_dc_attrs[originating_class].append(key)

            return True

        manager = instrumentation.manager_of_class(self.cls)
        assert manager is not None

        field_list = [
            _AttributeOptions._get_arguments_for_make_dataclass(
                key,
                anno,
                mapped_container,
                self.collected_attributes.get(key, _NoArg.NO_ARG),
            )
            for key, anno, mapped_container in (
                (
                    key,
                    mapped_anno if mapped_anno else raw_anno,
                    mapped_container,
                )
                for key, (
                    raw_anno,
                    mapped_container,
                    mapped_anno,
                    is_dc,
                    attr_value,
                    originating_module,
                    originating_class,
                ) in self.collected_annotations.items()
                if _allow_dataclass_field(key, originating_class)
                and (
                    key not in self.collected_attributes
                    # issue #9226; check for attributes that we've collected
                    # which are already instrumented, which we would assume
                    # mean we are in an ORM inheritance mapping and this
                    # attribute is already mapped on the superclass.   Under
                    # no circumstance should any QueryableAttribute be sent to
                    # the dataclass() function; anything that's mapped should
                    # be Field and that's it
                    or not isinstance(
                        self.collected_attributes[key], QueryableAttribute
                    )
                )
            )
        ]

        if warn_for_non_dc_attrs:
            for (
                originating_class,
                non_dc_attrs,
            ) in warn_for_non_dc_attrs.items():
                util.warn_deprecated(
                    f"When transforming {self.cls} to a dataclass, "
                    f"attribute(s) "
                    f"{', '.join(repr(key) for key in non_dc_attrs)} "
                    f"originates from superclass "
                    f"{originating_class}, which is not a dataclass.  This "
                    f"usage is deprecated and will raise an error in "
                    f"SQLAlchemy 2.1.  When declaring SQLAlchemy Declarative "
                    f"Dataclasses, ensure that all mixin classes and other "
                    f"superclasses which include attributes are also a "
                    f"subclass of MappedAsDataclass.",
                    "2.0",
                    code="dcmx",
                )

        annotations = {}
        defaults = {}
        for item in field_list:
            if len(item) == 2:
                name, tp = item
            elif len(item) == 3:
                name, tp, spec = item
                defaults[name] = spec
            else:
                assert False
            annotations[name] = tp

        for k, v in defaults.items():
            setattr(self.cls, k, v)

        self._apply_dataclasses_to_any_class(
            dataclass_setup_arguments, self.cls, annotations
        )

    @classmethod
    def _update_annotations_for_non_mapped_class(
        cls, klass: Type[_O]
    ) -> Mapping[str, _AnnotationScanType]:
        cls_annotations = util.get_annotations(klass)

        new_anno = {}
        for name, annotation in cls_annotations.items():
            if _is_mapped_annotation(annotation, klass, klass):
                extracted = _extract_mapped_subtype(
                    annotation,
                    klass,
                    klass.__module__,
                    name,
                    type(None),
                    required=False,
                    is_dataclass_field=False,
                    expect_mapped=False,
                )
                if extracted:
                    inner, _ = extracted
                    new_anno[name] = inner
            else:
                new_anno[name] = annotation
        return new_anno

    @classmethod
    def _apply_dataclasses_to_any_class(
        cls,
        dataclass_setup_arguments: _DataclassArguments,
        klass: Type[_O],
        use_annotations: Mapping[str, _AnnotationScanType],
    ) -> None:
        cls._assert_dc_arguments(dataclass_setup_arguments)

        dataclass_callable = dataclass_setup_arguments["dataclass_callable"]
        if dataclass_callable is _NoArg.NO_ARG:
            dataclass_callable = dataclasses.dataclass

        restored: Optional[Any]

        if use_annotations:
            # apply constructed annotations that should look "normal" to a
            # dataclasses callable, based on the fields present.  This
            # means remove the Mapped[] container and ensure all Field
            # entries have an annotation
            restored = getattr(klass, "__annotations__", None)
            klass.__annotations__ = cast("Dict[str, Any]", use_annotations)
        else:
            restored = None

        try:
            dataclass_callable(
                klass,
                **{
                    k: v
                    for k, v in dataclass_setup_arguments.items()
                    if v is not _NoArg.NO_ARG and k != "dataclass_callable"
                },
            )
        except (TypeError, ValueError) as ex:
            raise exc.InvalidRequestError(
                f"Python dataclasses error encountered when creating "
                f"dataclass for {klass.__name__!r}: "
                f"{ex!r}. Please refer to Python dataclasses "
                "documentation for additional information.",
                code="dcte",
            ) from ex
        finally:
            # restore original annotations outside of the dataclasses
            # process; for mixins and __abstract__ superclasses, SQLAlchemy
            # Declarative will need to see the Mapped[] container inside the
            # annotations in order to map subclasses
            if use_annotations:
                if restored is None:
                    del klass.__annotations__
                else:
                    klass.__annotations__ = restored

    @classmethod
    def _assert_dc_arguments(cls, arguments: _DataclassArguments) -> None:
        allowed = {
            "init",
            "repr",
            "order",
            "eq",
            "unsafe_hash",
            "kw_only",
            "match_args",
            "dataclass_callable",
        }
        disallowed_args = set(arguments).difference(allowed)
        if disallowed_args:
            msg = ", ".join(f"{arg!r}" for arg in sorted(disallowed_args))
            raise exc.ArgumentError(
                f"Dataclass argument(s) {msg} are not accepted"
            )

    def _collect_annotation(
        self,
        name: str,
        raw_annotation: _AnnotationScanType,
        originating_class: Type[Any],
        expect_mapped: Optional[bool],
        attr_value: Any,
    ) -> Optional[_CollectedAnnotation]:
        if name in self.collected_annotations:
            return self.collected_annotations[name]

        if raw_annotation is None:
            return None

        is_dataclass = self.is_dataclass_prior_to_mapping
        allow_unmapped = self.allow_unmapped_annotations

        if expect_mapped is None:
            is_dataclass_field = isinstance(attr_value, dataclasses.Field)
            expect_mapped = (
                not is_dataclass_field
                and not allow_unmapped
                and (
                    attr_value is None
                    or isinstance(attr_value, _MappedAttribute)
                )
            )

        is_dataclass_field = False
        extracted = _extract_mapped_subtype(
            raw_annotation,
            self.cls,
            originating_class.__module__,
            name,
            type(attr_value),
            required=False,
            is_dataclass_field=is_dataclass_field,
            expect_mapped=expect_mapped and not is_dataclass,
        )
        if extracted is None:
            # ClassVar can come out here
            return None

        extracted_mapped_annotation, mapped_container = extracted

        if attr_value is None and not is_literal(extracted_mapped_annotation):
            for elem in get_args(extracted_mapped_annotation):
                if is_fwd_ref(
                    elem, check_generic=True, check_for_plain_string=True
                ):
                    elem = de_stringify_annotation(
                        self.cls,
                        elem,
                        originating_class.__module__,
                        include_generic=True,
                    )
                # look in Annotated[...] for an ORM construct,
                # such as Annotated[int, mapped_column(primary_key=True)]
                if isinstance(elem, _IntrospectsAnnotations):
                    attr_value = elem.found_in_pep593_annotated()

        self.collected_annotations[name] = ca = _CollectedAnnotation(
            raw_annotation,
            mapped_container,
            extracted_mapped_annotation,
            is_dataclass,
            attr_value,
            originating_class.__module__,
            originating_class,
        )
        return ca

    def _warn_for_decl_attributes(
        self, cls: Type[Any], key: str, c: Any
    ) -> None:
        if isinstance(c, expression.ColumnElement):
            util.warn(
                f"Attribute '{key}' on class {cls} appears to "
                "be a non-schema SQLAlchemy expression "
                "object; this won't be part of the declarative mapping. "
                "To map arbitrary expressions, use ``column_property()`` "
                "or a similar function such as ``deferred()``, "
                "``query_expression()`` etc. "
            )

    def _produce_column_copies(
        self,
        attributes_for_class: Callable[
            [], Iterable[Tuple[str, Any, Any, bool]]
        ],
        attribute_is_overridden: Callable[[str, Any], bool],
        fixed_table: bool,
        originating_class: Type[Any],
    ) -> Dict[str, Union[Column[Any], MappedColumn[Any]]]:
        cls = self.cls
        dict_ = self.clsdict_view
        locally_collected_attributes = {}
        column_copies = self.column_copies
        # copy mixin columns to the mapped class

        for name, obj, annotation, is_dataclass in attributes_for_class():
            if (
                not fixed_table
                and obj is None
                and _is_mapped_annotation(annotation, cls, originating_class)
            ):
                # obj is None means this is the annotation only path

                if attribute_is_overridden(name, obj):
                    # perform same "overridden" check as we do for
                    # Column/MappedColumn, this is how a mixin col is not
                    # applied to an inherited subclass that does not have
                    # the mixin.   the anno-only path added here for
                    # #9564
                    continue

                collected_annotation = self._collect_annotation(
                    name, annotation, originating_class, True, obj
                )
                obj = (
                    collected_annotation.attr_value
                    if collected_annotation is not None
                    else obj
                )
                if obj is None:
                    obj = MappedColumn()

                locally_collected_attributes[name] = obj
                setattr(cls, name, obj)

            elif isinstance(obj, (Column, MappedColumn)):
                if attribute_is_overridden(name, obj):
                    # if column has been overridden
                    # (like by the InstrumentedAttribute of the
                    # superclass), skip.  don't collect the annotation
                    # either (issue #8718)
                    continue

                collected_annotation = self._collect_annotation(
                    name, annotation, originating_class, True, obj
                )
                obj = (
                    collected_annotation.attr_value
                    if collected_annotation is not None
                    else obj
                )

                if name not in dict_ and not (
                    "__table__" in dict_
                    and (getattr(obj, "name", None) or name)
                    in dict_["__table__"].c
                ):
                    if obj.foreign_keys:
                        for fk in obj.foreign_keys:
                            if (
                                fk._table_column is not None
                                and fk._table_column.table is None
                            ):
                                raise exc.InvalidRequestError(
                                    "Columns with foreign keys to "
                                    "non-table-bound "
                                    "columns must be declared as "
                                    "@declared_attr callables "
                                    "on declarative mixin classes.  "
                                    "For dataclass "
                                    "field() objects, use a lambda:."
                                )

                    column_copies[obj] = copy_ = obj._copy()

                    locally_collected_attributes[name] = copy_
                    setattr(cls, name, copy_)

        return locally_collected_attributes

    def _extract_mappable_attributes(self) -> None:
        cls = self.cls
        collected_attributes = self.collected_attributes

        our_stuff = self.properties

        _include_dunders = self._include_dunders

        late_mapped = _get_immediate_cls_attr(
            cls, "_sa_decl_prepare_nocascade", strict=True
        )

        allow_unmapped_annotations = self.allow_unmapped_annotations
        expect_annotations_wo_mapped = (
            allow_unmapped_annotations or self.is_dataclass_prior_to_mapping
        )

        look_for_dataclass_things = bool(self.dataclass_setup_arguments)

        for k in list(collected_attributes):
            if k in _include_dunders:
                continue

            value = collected_attributes[k]

            if _is_declarative_props(value):
                # @declared_attr in collected_attributes only occurs here for a
                # @declared_attr that's directly on the mapped class;
                # for a mixin, these have already been evaluated
                if value._cascading:
                    util.warn(
                        "Use of @declared_attr.cascading only applies to "
                        "Declarative 'mixin' and 'abstract' classes.  "
                        "Currently, this flag is ignored on mapped class "
                        "%s" % self.cls
                    )

                value = getattr(cls, k)

            elif (
                isinstance(value, QueryableAttribute)
                and value.class_ is not cls
                and value.key != k
            ):
                # detect a QueryableAttribute that's already mapped being
                # assigned elsewhere in userland, turn into a synonym()
                value = SynonymProperty(value.key)
                setattr(cls, k, value)

            if (
                isinstance(value, tuple)
                and len(value) == 1
                and isinstance(value[0], (Column, _MappedAttribute))
            ):
                util.warn(
                    "Ignoring declarative-like tuple value of attribute "
                    "'%s': possibly a copy-and-paste error with a comma "
                    "accidentally placed at the end of the line?" % k
                )
                continue
            elif look_for_dataclass_things and isinstance(
                value, dataclasses.Field
            ):
                # we collected a dataclass Field; dataclasses would have
                # set up the correct state on the class
                continue
            elif not isinstance(value, (Column, _DCAttributeOptions)):
                # using @declared_attr for some object that
                # isn't Column/MapperProperty/_DCAttributeOptions; remove
                # from the clsdict_view
                # and place the evaluated value onto the class.
                collected_attributes.pop(k)
                self._warn_for_decl_attributes(cls, k, value)
                if not late_mapped:
                    setattr(cls, k, value)
                continue
            # we expect to see the name 'metadata' in some valid cases;
            # however at this point we see it's assigned to something trying
            # to be mapped, so raise for that.
            # TODO: should "registry" here be also?   might be too late
            # to change that now (2.0 betas)
            elif k in ("metadata",):
                raise exc.InvalidRequestError(
                    f"Attribute name '{k}' is reserved when using the "
                    "Declarative API."
                )
            elif isinstance(value, Column):
                _undefer_column_name(
                    k, self.column_copies.get(value, value)  # type: ignore
                )
            else:
                if isinstance(value, _IntrospectsAnnotations):
                    (
                        annotation,
                        mapped_container,
                        extracted_mapped_annotation,
                        is_dataclass,
                        attr_value,
                        originating_module,
                        originating_class,
                    ) = self.collected_annotations.get(
                        k, (None, None, None, False, None, None, None)
                    )

                    # issue #8692 - don't do any annotation interpretation if
                    # an annotation were present and a container such as
                    # Mapped[] etc. were not used.  If annotation is None,
                    # do declarative_scan so that the property can raise
                    # for required
                    if (
                        mapped_container is not None
                        or annotation is None
                        # issue #10516: need to do declarative_scan even with
                        # a non-Mapped annotation if we are doing
                        # __allow_unmapped__, for things like col.name
                        # assignment
                        or allow_unmapped_annotations
                    ):
                        try:
                            value.declarative_scan(
                                self,
                                self.registry,
                                cls,
                                originating_module,
                                k,
                                mapped_container,
                                annotation,
                                extracted_mapped_annotation,
                                is_dataclass,
                            )
                        except NameError as ne:
                            raise orm_exc.MappedAnnotationError(
                                f"Could not resolve all types within mapped "
                                f'annotation: "{annotation}".  Ensure all '
                                f"types are written correctly and are "
                                f"imported within the module in use."
                            ) from ne
                    else:
                        # assert that we were expecting annotations
                        # without Mapped[] were going to be passed.
                        # otherwise an error should have been raised
                        # by util._extract_mapped_subtype before we got here.
                        assert expect_annotations_wo_mapped

                if isinstance(value, _DCAttributeOptions):
                    if (
                        value._has_dataclass_arguments
                        and not look_for_dataclass_things
                    ):
                        if isinstance(value, MapperProperty):
                            argnames = [
                                "init",
                                "default_factory",
                                "repr",
                                "default",
                            ]
                        else:
                            argnames = ["init", "default_factory", "repr"]

                        args = {
                            a
                            for a in argnames
                            if getattr(
                                value._attribute_options, f"dataclasses_{a}"
                            )
                            is not _NoArg.NO_ARG
                        }

                        raise exc.ArgumentError(
                            f"Attribute '{k}' on class {cls} includes "
                            f"dataclasses argument(s): "
                            f"{', '.join(sorted(repr(a) for a in args))} but "
                            f"class does not specify "
                            "SQLAlchemy native dataclass configuration."
                        )

                    if not isinstance(value, (MapperProperty, _MapsColumns)):
                        # filter for _DCAttributeOptions objects that aren't
                        # MapperProperty / mapped_column().  Currently this
                        # includes AssociationProxy.   pop it from the things
                        # we're going to map and set it up as a descriptor
                        # on the class.
                        collected_attributes.pop(k)

                        # Assoc Prox (or other descriptor object that may
                        # use _DCAttributeOptions) is usually here, except if
                        # 1. we're a
                        # dataclass, dataclasses would have removed the
                        # attr here or 2. assoc proxy is coming from a
                        # superclass, we want it to be direct here so it
                        # tracks state or 3. assoc prox comes from
                        # declared_attr, uncommon case
                        setattr(cls, k, value)
                        continue

            our_stuff[k] = value

    def _extract_declared_columns(self) -> None:
        our_stuff = self.properties

        # extract columns from the class dict
        declared_columns = self.declared_columns
        column_ordering = self.column_ordering
        name_to_prop_key = collections.defaultdict(set)

        for key, c in list(our_stuff.items()):
            if isinstance(c, _MapsColumns):
                mp_to_assign = c.mapper_property_to_assign
                if mp_to_assign:
                    our_stuff[key] = mp_to_assign
                else:
                    # if no mapper property to assign, this currently means
                    # this is a MappedColumn that will produce a Column for us
                    del our_stuff[key]

                for col, sort_order in c.columns_to_assign:
                    if not isinstance(c, CompositeProperty):
                        name_to_prop_key[col.name].add(key)
                    declared_columns.add(col)

                    # we would assert this, however we want the below
                    # warning to take effect instead.  See #9630
                    # assert col not in column_ordering

                    column_ordering[col] = sort_order

                    # if this is a MappedColumn and the attribute key we
                    # have is not what the column has for its key, map the
                    # Column explicitly under the attribute key name.
                    # otherwise, Mapper will map it under the column key.
                    if mp_to_assign is None and key != col.key:
                        our_stuff[key] = col
            elif isinstance(c, Column):
                # undefer previously occurred here, and now occurs earlier.
                # ensure every column we get here has been named
                assert c.name is not None
                name_to_prop_key[c.name].add(key)
                declared_columns.add(c)
                # if the column is the same name as the key,
                # remove it from the explicit properties dict.
                # the normal rules for assigning column-based properties
                # will take over, including precedence of columns
                # in multi-column ColumnProperties.
                if key == c.key:
                    del our_stuff[key]

        for name, keys in name_to_prop_key.items():
            if len(keys) > 1:
                util.warn(
                    "On class %r, Column object %r named "
                    "directly multiple times, "
                    "only one will be used: %s. "
                    "Consider using orm.synonym instead"
                    % (self.classname, name, (", ".join(sorted(keys))))
                )

    def _setup_table(self, table: Optional[FromClause] = None) -> None:
        cls = self.cls
        cls_as_Decl = cast("MappedClassProtocol[Any]", cls)

        tablename = self.tablename
        table_args = self.table_args
        clsdict_view = self.clsdict_view
        declared_columns = self.declared_columns
        column_ordering = self.column_ordering

        manager = attributes.manager_of_class(cls)

        if (
            self.table_fn is None
            and "__table__" not in clsdict_view
            and table is None
        ):
            if hasattr(cls, "__table_cls__"):
                table_cls = cast(
                    Type[Table],
                    util.unbound_method_to_callable(cls.__table_cls__),  # type: ignore  # noqa: E501
                )
            else:
                table_cls = Table

            if tablename is not None:
                args: Tuple[Any, ...] = ()
                table_kw: Dict[str, Any] = {}

                if table_args:
                    if isinstance(table_args, dict):
                        table_kw = table_args
                    elif isinstance(table_args, tuple):
                        if isinstance(table_args[-1], dict):
                            args, table_kw = table_args[0:-1], table_args[-1]
                        else:
                            args = table_args

                autoload_with = clsdict_view.get("__autoload_with__")
                if autoload_with:
                    table_kw["autoload_with"] = autoload_with

                autoload = clsdict_view.get("__autoload__")
                if autoload:
                    table_kw["autoload"] = True

                sorted_columns = sorted(
                    declared_columns,
                    key=lambda c: column_ordering.get(c, 0),
                )
                table = self.set_cls_attribute(
                    "__table__",
                    table_cls(
                        tablename,
                        self._metadata_for_cls(manager),
                        *sorted_columns,
                        *args,
                        **table_kw,
                    ),
                )
        else:
            if table is None:
                if self.table_fn:
                    table = self.set_cls_attribute(
                        "__table__", self.table_fn()
                    )
                else:
                    table = cls_as_Decl.__table__
            if declared_columns:
                for c in declared_columns:
                    if not table.c.contains_column(c):
                        raise exc.ArgumentError(
                            "Can't add additional column %r when "
                            "specifying __table__" % c.key
                        )

        self.local_table = table

    def _metadata_for_cls(self, manager: ClassManager[Any]) -> MetaData:
        meta: Optional[MetaData] = getattr(self.cls, "metadata", None)
        if meta is not None:
            return meta
        else:
            return manager.registry.metadata

    def _setup_inheriting_mapper(self, mapper_kw: _MapperKwArgs) -> None:
        cls = self.cls

        inherits = mapper_kw.get("inherits", None)

        if inherits is None:
            # since we search for classical mappings now, search for
            # multiple mapped bases as well and raise an error.
            inherits_search = []
            for base_ in cls.__bases__:
                c = _resolve_for_abstract_or_classical(base_)
                if c is None:
                    continue

                if _is_supercls_for_inherits(c) and c not in inherits_search:
                    inherits_search.append(c)

            if inherits_search:
                if len(inherits_search) > 1:
                    raise exc.InvalidRequestError(
                        "Class %s has multiple mapped bases: %r"
                        % (cls, inherits_search)
                    )
                inherits = inherits_search[0]
        elif isinstance(inherits, Mapper):
            inherits = inherits.class_

        self.inherits = inherits

        clsdict_view = self.clsdict_view
        if "__table__" not in clsdict_view and self.tablename is None:
            self.single = True

    def _setup_inheriting_columns(self, mapper_kw: _MapperKwArgs) -> None:
        table = self.local_table
        cls = self.cls
        table_args = self.table_args
        declared_columns = self.declared_columns

        if (
            table is None
            and self.inherits is None
            and not _get_immediate_cls_attr(cls, "__no_table__")
        ):
            raise exc.InvalidRequestError(
                "Class %r does not have a __table__ or __tablename__ "
                "specified and does not inherit from an existing "
                "table-mapped class." % cls
            )
        elif self.inherits:
            inherited_mapper_or_config = _declared_mapping_info(self.inherits)
            assert inherited_mapper_or_config is not None
            inherited_table = inherited_mapper_or_config.local_table
            inherited_persist_selectable = (
                inherited_mapper_or_config.persist_selectable
            )

            if table is None:
                # single table inheritance.
                # ensure no table args
                if table_args:
                    raise exc.ArgumentError(
                        "Can't place __table_args__ on an inherited class "
                        "with no table."
                    )

                # add any columns declared here to the inherited table.
                if declared_columns and not isinstance(inherited_table, Table):
                    raise exc.ArgumentError(
                        f"Can't declare columns on single-table-inherited "
                        f"subclass {self.cls}; superclass {self.inherits} "
                        "is not mapped to a Table"
                    )

                for col in declared_columns:
                    assert inherited_table is not None
                    if col.name in inherited_table.c:
                        if inherited_table.c[col.name] is col:
                            continue
                        raise exc.ArgumentError(
                            f"Column '{col}' on class {cls.__name__} "
                            f"conflicts with existing column "
                            f"'{inherited_table.c[col.name]}'.  If using "
                            f"Declarative, consider using the "
                            "use_existing_column parameter of mapped_column() "
                            "to resolve conflicts."
                        )
                    if col.primary_key:
                        raise exc.ArgumentError(
                            "Can't place primary key columns on an inherited "
                            "class with no table."
                        )

                    if TYPE_CHECKING:
                        assert isinstance(inherited_table, Table)

                    inherited_table.append_column(col)
                    if (
                        inherited_persist_selectable is not None
                        and inherited_persist_selectable is not inherited_table
                    ):
                        inherited_persist_selectable._refresh_for_new_column(
                            col
                        )

    def _prepare_mapper_arguments(self, mapper_kw: _MapperKwArgs) -> None:
        properties = self.properties

        if self.mapper_args_fn:
            mapper_args = self.mapper_args_fn()
        else:
            mapper_args = {}

        if mapper_kw:
            mapper_args.update(mapper_kw)

        if "properties" in mapper_args:
            properties = dict(properties)
            properties.update(mapper_args["properties"])

        # make sure that column copies are used rather
        # than the original columns from any mixins
        for k in ("version_id_col", "polymorphic_on"):
            if k in mapper_args:
                v = mapper_args[k]
                mapper_args[k] = self.column_copies.get(v, v)

        if "primary_key" in mapper_args:
            mapper_args["primary_key"] = [
                self.column_copies.get(v, v)
                for v in util.to_list(mapper_args["primary_key"])
            ]

        if "inherits" in mapper_args:
            inherits_arg = mapper_args["inherits"]
            if isinstance(inherits_arg, Mapper):
                inherits_arg = inherits_arg.class_

            if inherits_arg is not self.inherits:
                raise exc.InvalidRequestError(
                    "mapper inherits argument given for non-inheriting "
                    "class %s" % (mapper_args["inherits"])
                )

        if self.inherits:
            mapper_args["inherits"] = self.inherits

        if self.inherits and not mapper_args.get("concrete", False):
            # note the superclass is expected to have a Mapper assigned and
            # not be a deferred config, as this is called within map()
            inherited_mapper = class_mapper(self.inherits, False)
            inherited_table = inherited_mapper.local_table

            # single or joined inheritance
            # exclude any cols on the inherited table which are
            # not mapped on the parent class, to avoid
            # mapping columns specific to sibling/nephew classes
            if "exclude_properties" not in mapper_args:
                mapper_args["exclude_properties"] = exclude_properties = {
                    c.key
                    for c in inherited_table.c
                    if c not in inherited_mapper._columntoproperty
                }.union(inherited_mapper.exclude_properties or ())
                exclude_properties.difference_update(
                    [c.key for c in self.declared_columns]
                )

            # look through columns in the current mapper that
            # are keyed to a propname different than the colname
            # (if names were the same, we'd have popped it out above,
            # in which case the mapper makes this combination).
            # See if the superclass has a similar column property.
            # If so, join them together.
            for k, col in list(properties.items()):
                if not isinstance(col, expression.ColumnElement):
                    continue
                if k in inherited_mapper._props:
                    p = inherited_mapper._props[k]
                    if isinstance(p, ColumnProperty):
                        # note here we place the subclass column
                        # first.  See [ticket:1892] for background.
                        properties[k] = [col] + p.columns
        result_mapper_args = mapper_args.copy()
        result_mapper_args["properties"] = properties
        self.mapper_args = result_mapper_args

    def map(self, mapper_kw: _MapperKwArgs = util.EMPTY_DICT) -> Mapper[Any]:
        self._prepare_mapper_arguments(mapper_kw)
        if hasattr(self.cls, "__mapper_cls__"):
            mapper_cls = cast(
                "Type[Mapper[Any]]",
                util.unbound_method_to_callable(
                    self.cls.__mapper_cls__  # type: ignore
                ),
            )
        else:
            mapper_cls = Mapper

        return self.set_cls_attribute(
            "__mapper__",
            mapper_cls(self.cls, self.local_table, **self.mapper_args),
        )


@util.preload_module("sqlalchemy.orm.decl_api")
def _as_dc_declaredattr(
    field_metadata: Mapping[str, Any], sa_dataclass_metadata_key: str
) -> Any:
    # wrap lambdas inside dataclass fields inside an ad-hoc declared_attr.
    # we can't write it because field.metadata is immutable :( so we have
    # to go through extra trouble to compare these
    decl_api = util.preloaded.orm_decl_api
    obj = field_metadata[sa_dataclass_metadata_key]
    if callable(obj) and not isinstance(obj, decl_api.declared_attr):
        return decl_api.declared_attr(obj)
    else:
        return obj


class _DeferredMapperConfig(_ClassScanMapperConfig):
    _cls: weakref.ref[Type[Any]]

    is_deferred = True

    _configs: util.OrderedDict[
        weakref.ref[Type[Any]], _DeferredMapperConfig
    ] = util.OrderedDict()

    def _early_mapping(self, mapper_kw: _MapperKwArgs) -> None:
        pass

    # mypy disallows plain property override of variable
    @property  # type: ignore
    def cls(self) -> Type[Any]:
        return self._cls()  # type: ignore

    @cls.setter
    def cls(self, class_: Type[Any]) -> None:
        self._cls = weakref.ref(class_, self._remove_config_cls)
        self._configs[self._cls] = self

    @classmethod
    def _remove_config_cls(cls, ref: weakref.ref[Type[Any]]) -> None:
        cls._configs.pop(ref, None)

    @classmethod
    def has_cls(cls, class_: Type[Any]) -> bool:
        # 2.6 fails on weakref if class_ is an old style class
        return isinstance(class_, type) and weakref.ref(class_) in cls._configs

    @classmethod
    def raise_unmapped_for_cls(cls, class_: Type[Any]) -> NoReturn:
        if hasattr(class_, "_sa_raise_deferred_config"):
            class_._sa_raise_deferred_config()

        raise orm_exc.UnmappedClassError(
            class_,
            msg=(
                f"Class {orm_exc._safe_cls_name(class_)} has a deferred "
                "mapping on it.  It is not yet usable as a mapped class."
            ),
        )

    @classmethod
    def config_for_cls(cls, class_: Type[Any]) -> _DeferredMapperConfig:
        return cls._configs[weakref.ref(class_)]

    @classmethod
    def classes_for_base(
        cls, base_cls: Type[Any], sort: bool = True
    ) -> List[_DeferredMapperConfig]:
        classes_for_base = [
            m
            for m, cls_ in [(m, m.cls) for m in cls._configs.values()]
            if cls_ is not None and issubclass(cls_, base_cls)
        ]

        if not sort:
            return classes_for_base

        all_m_by_cls = {m.cls: m for m in classes_for_base}

        tuples: List[Tuple[_DeferredMapperConfig, _DeferredMapperConfig]] = []
        for m_cls in all_m_by_cls:
            tuples.extend(
                (all_m_by_cls[base_cls], all_m_by_cls[m_cls])
                for base_cls in m_cls.__bases__
                if base_cls in all_m_by_cls
            )
        return list(topological.sort(tuples, classes_for_base))

    def map(self, mapper_kw: _MapperKwArgs = util.EMPTY_DICT) -> Mapper[Any]:
        self._configs.pop(self._cls, None)
        return super().map(mapper_kw)


def _add_attribute(
    cls: Type[Any], key: str, value: MapperProperty[Any]
) -> None:
    """add an attribute to an existing declarative class.

    This runs through the logic to determine MapperProperty,
    adds it to the Mapper, adds a column to the mapped Table, etc.

    """

    if "__mapper__" in cls.__dict__:
        mapped_cls = cast("MappedClassProtocol[Any]", cls)

        def _table_or_raise(mc: MappedClassProtocol[Any]) -> Table:
            if isinstance(mc.__table__, Table):
                return mc.__table__
            raise exc.InvalidRequestError(
                f"Cannot add a new attribute to mapped class {mc.__name__!r} "
                "because it's not mapped against a table."
            )

        if isinstance(value, Column):
            _undefer_column_name(key, value)
            _table_or_raise(mapped_cls).append_column(
                value, replace_existing=True
            )
            mapped_cls.__mapper__.add_property(key, value)
        elif isinstance(value, _MapsColumns):
            mp = value.mapper_property_to_assign
            for col, _ in value.columns_to_assign:
                _undefer_column_name(key, col)
                _table_or_raise(mapped_cls).append_column(
                    col, replace_existing=True
                )
                if not mp:
                    mapped_cls.__mapper__.add_property(key, col)
            if mp:
                mapped_cls.__mapper__.add_property(key, mp)
        elif isinstance(value, MapperProperty):
            mapped_cls.__mapper__.add_property(key, value)
        elif isinstance(value, QueryableAttribute) and value.key != key:
            # detect a QueryableAttribute that's already mapped being
            # assigned elsewhere in userland, turn into a synonym()
            value = SynonymProperty(value.key)
            mapped_cls.__mapper__.add_property(key, value)
        else:
            type.__setattr__(cls, key, value)
            mapped_cls.__mapper__._expire_memoizations()
    else:
        type.__setattr__(cls, key, value)


def _del_attribute(cls: Type[Any], key: str) -> None:
    if (
        "__mapper__" in cls.__dict__
        and key in cls.__dict__
        and not cast(
            "MappedClassProtocol[Any]", cls
        ).__mapper__._dispose_called
    ):
        value = cls.__dict__[key]
        if isinstance(
            value, (Column, _MapsColumns, MapperProperty, QueryableAttribute)
        ):
            raise NotImplementedError(
                "Can't un-map individual mapped attributes on a mapped class."
            )
        else:
            type.__delattr__(cls, key)
            cast(
                "MappedClassProtocol[Any]", cls
            ).__mapper__._expire_memoizations()
    else:
        type.__delattr__(cls, key)


def _declarative_constructor(self: Any, **kwargs: Any) -> None:
    """A simple constructor that allows initialization from kwargs.

    Sets attributes on the constructed instance using the names and
    values in ``kwargs``.

    Only keys that are present as
    attributes of the instance's class are allowed. These could be,
    for example, any mapped columns or relationships.
    """
    cls_ = type(self)
    for k in kwargs:
        if not hasattr(cls_, k):
            raise TypeError(
                "%r is an invalid keyword argument for %s" % (k, cls_.__name__)
            )
        setattr(self, k, kwargs[k])


_declarative_constructor.__name__ = "__init__"


def _undefer_column_name(key: str, column: Column[Any]) -> None:
    if column.key is None:
        column.key = key
    if column.name is None:
        column.name = key

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\cursor.py ===
# engine/cursor.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Define cursor-specific result set constructs including
:class:`.CursorResult`."""


from __future__ import annotations

import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self


if typing.TYPE_CHECKING:
    from .base import Connection
    from .default import DefaultExecutionContext
    from .interfaces import _DBAPICursorDescription
    from .interfaces import DBAPICursor
    from .interfaces import Dialect
    from .interfaces import ExecutionContext
    from .result import _KeyIndexType
    from .result import _KeyMapRecType
    from .result import _KeyMapType
    from .result import _KeyType
    from .result import _ProcessorsType
    from .result import _TupleGetterType
    from ..sql.type_api import _ResultProcessorType


_T = TypeVar("_T", bound=Any)


# metadata entry tuple indexes.
# using raw tuple is faster than namedtuple.
# these match up to the positions in
# _CursorKeyMapRecType
MD_INDEX: Literal[0] = 0
"""integer index in cursor.description

"""

MD_RESULT_MAP_INDEX: Literal[1] = 1
"""integer index in compiled._result_columns"""

MD_OBJECTS: Literal[2] = 2
"""other string keys and ColumnElement obj that can match.

This comes from compiler.RM_OBJECTS / compiler.ResultColumnsEntry.objects

"""

MD_LOOKUP_KEY: Literal[3] = 3
"""string key we usually expect for key-based lookup

this comes from compiler.RM_NAME / compiler.ResultColumnsEntry.name
"""


MD_RENDERED_NAME: Literal[4] = 4
"""name that is usually in cursor.description

this comes from compiler.RENDERED_NAME / compiler.ResultColumnsEntry.keyname
"""


MD_PROCESSOR: Literal[5] = 5
"""callable to process a result value into a row"""

MD_UNTRANSLATED: Literal[6] = 6
"""raw name from cursor.description"""


_CursorKeyMapRecType = Tuple[
    Optional[int],  # MD_INDEX, None means the record is ambiguously named
    int,  # MD_RESULT_MAP_INDEX
    List[Any],  # MD_OBJECTS
    str,  # MD_LOOKUP_KEY
    str,  # MD_RENDERED_NAME
    Optional["_ResultProcessorType[Any]"],  # MD_PROCESSOR
    Optional[str],  # MD_UNTRANSLATED
]

_CursorKeyMapType = Mapping["_KeyType", _CursorKeyMapRecType]

# same as _CursorKeyMapRecType except the MD_INDEX value is definitely
# not None
_NonAmbigCursorKeyMapRecType = Tuple[
    int,
    int,
    List[Any],
    str,
    str,
    Optional["_ResultProcessorType[Any]"],
    str,
]


class CursorResultMetaData(ResultMetaData):
    """Result metadata for DBAPI cursors."""

    __slots__ = (
        "_keymap",
        "_processors",
        "_keys",
        "_keymap_by_result_column_idx",
        "_tuplefilter",
        "_translated_indexes",
        "_safe_for_cache",
        "_unpickled",
        "_key_to_index",
        # don't need _unique_filters support here for now.  Can be added
        # if a need arises.
    )

    _keymap: _CursorKeyMapType
    _processors: _ProcessorsType
    _keymap_by_result_column_idx: Optional[Dict[int, _KeyMapRecType]]
    _unpickled: bool
    _safe_for_cache: bool
    _translated_indexes: Optional[List[int]]

    returns_rows: ClassVar[bool] = True

    def _has_key(self, key: Any) -> bool:
        return key in self._keymap

    def _for_freeze(self) -> ResultMetaData:
        return SimpleResultMetaData(
            self._keys,
            extra=[self._keymap[key][MD_OBJECTS] for key in self._keys],
        )

    def _make_new_metadata(
        self,
        *,
        unpickled: bool,
        processors: _ProcessorsType,
        keys: Sequence[str],
        keymap: _KeyMapType,
        tuplefilter: Optional[_TupleGetterType],
        translated_indexes: Optional[List[int]],
        safe_for_cache: bool,
        keymap_by_result_column_idx: Any,
    ) -> CursorResultMetaData:
        new_obj = self.__class__.__new__(self.__class__)
        new_obj._unpickled = unpickled
        new_obj._processors = processors
        new_obj._keys = keys
        new_obj._keymap = keymap
        new_obj._tuplefilter = tuplefilter
        new_obj._translated_indexes = translated_indexes
        new_obj._safe_for_cache = safe_for_cache
        new_obj._keymap_by_result_column_idx = keymap_by_result_column_idx
        new_obj._key_to_index = self._make_key_to_index(keymap, MD_INDEX)
        return new_obj

    def _remove_processors(self) -> CursorResultMetaData:
        assert not self._tuplefilter
        return self._make_new_metadata(
            unpickled=self._unpickled,
            processors=[None] * len(self._processors),
            tuplefilter=None,
            translated_indexes=None,
            keymap={
                key: value[0:5] + (None,) + value[6:]
                for key, value in self._keymap.items()
            },
            keys=self._keys,
            safe_for_cache=self._safe_for_cache,
            keymap_by_result_column_idx=self._keymap_by_result_column_idx,
        )

    def _splice_horizontally(
        self, other: CursorResultMetaData
    ) -> CursorResultMetaData:
        assert not self._tuplefilter

        keymap = dict(self._keymap)
        offset = len(self._keys)
        keymap.update(
            {
                key: (
                    # int index should be None for ambiguous key
                    (
                        value[0] + offset
                        if value[0] is not None and key not in keymap
                        else None
                    ),
                    value[1] + offset,
                    *value[2:],
                )
                for key, value in other._keymap.items()
            }
        )
        return self._make_new_metadata(
            unpickled=self._unpickled,
            processors=self._processors + other._processors,  # type: ignore
            tuplefilter=None,
            translated_indexes=None,
            keys=self._keys + other._keys,  # type: ignore
            keymap=keymap,
            safe_for_cache=self._safe_for_cache,
            keymap_by_result_column_idx={
                metadata_entry[MD_RESULT_MAP_INDEX]: metadata_entry
                for metadata_entry in keymap.values()
            },
        )

    def _reduce(self, keys: Sequence[_KeyIndexType]) -> ResultMetaData:
        recs = list(self._metadata_for_keys(keys))

        indexes = [rec[MD_INDEX] for rec in recs]
        new_keys: List[str] = [rec[MD_LOOKUP_KEY] for rec in recs]

        if self._translated_indexes:
            indexes = [self._translated_indexes[idx] for idx in indexes]
        tup = tuplegetter(*indexes)
        new_recs = [(index,) + rec[1:] for index, rec in enumerate(recs)]

        keymap = {rec[MD_LOOKUP_KEY]: rec for rec in new_recs}
        # TODO: need unit test for:
        # result = connection.execute("raw sql, no columns").scalars()
        # without the "or ()" it's failing because MD_OBJECTS is None
        keymap.update(
            (e, new_rec)
            for new_rec in new_recs
            for e in new_rec[MD_OBJECTS] or ()
        )

        return self._make_new_metadata(
            unpickled=self._unpickled,
            processors=self._processors,
            keys=new_keys,
            tuplefilter=tup,
            translated_indexes=indexes,
            keymap=keymap,  # type: ignore[arg-type]
            safe_for_cache=self._safe_for_cache,
            keymap_by_result_column_idx=self._keymap_by_result_column_idx,
        )

    def _adapt_to_context(self, context: ExecutionContext) -> ResultMetaData:
        """When using a cached Compiled construct that has a _result_map,
        for a new statement that used the cached Compiled, we need to ensure
        the keymap has the Column objects from our new statement as keys.
        So here we rewrite keymap with new entries for the new columns
        as matched to those of the cached statement.

        """

        if not context.compiled or not context.compiled._result_columns:
            return self

        compiled_statement = context.compiled.statement
        invoked_statement = context.invoked_statement

        if TYPE_CHECKING:
            assert isinstance(invoked_statement, elements.ClauseElement)

        if compiled_statement is invoked_statement:
            return self

        assert invoked_statement is not None

        # this is the most common path for Core statements when
        # caching is used.  In ORM use, this codepath is not really used
        # as the _result_disable_adapt_to_context execution option is
        # set by the ORM.

        # make a copy and add the columns from the invoked statement
        # to the result map.

        keymap_by_position = self._keymap_by_result_column_idx

        if keymap_by_position is None:
            # first retrival from cache, this map will not be set up yet,
            # initialize lazily
            keymap_by_position = self._keymap_by_result_column_idx = {
                metadata_entry[MD_RESULT_MAP_INDEX]: metadata_entry
                for metadata_entry in self._keymap.values()
            }

        assert not self._tuplefilter
        return self._make_new_metadata(
            keymap=compat.dict_union(
                self._keymap,
                {
                    new: keymap_by_position[idx]
                    for idx, new in enumerate(
                        invoked_statement._all_selected_columns
                    )
                    if idx in keymap_by_position
                },
            ),
            unpickled=self._unpickled,
            processors=self._processors,
            tuplefilter=None,
            translated_indexes=None,
            keys=self._keys,
            safe_for_cache=self._safe_for_cache,
            keymap_by_result_column_idx=self._keymap_by_result_column_idx,
        )

    def __init__(
        self,
        parent: CursorResult[Any],
        cursor_description: _DBAPICursorDescription,
    ):
        context = parent.context
        self._tuplefilter = None
        self._translated_indexes = None
        self._safe_for_cache = self._unpickled = False

        if context.result_column_struct:
            (
                result_columns,
                cols_are_ordered,
                textual_ordered,
                ad_hoc_textual,
                loose_column_name_matching,
            ) = context.result_column_struct
            num_ctx_cols = len(result_columns)
        else:
            result_columns = cols_are_ordered = (  # type: ignore
                num_ctx_cols
            ) = ad_hoc_textual = loose_column_name_matching = (
                textual_ordered
            ) = False

        # merge cursor.description with the column info
        # present in the compiled structure, if any
        raw = self._merge_cursor_description(
            context,
            cursor_description,
            result_columns,
            num_ctx_cols,
            cols_are_ordered,
            textual_ordered,
            ad_hoc_textual,
            loose_column_name_matching,
        )

        # processors in key order which are used when building up
        # a row
        self._processors = [
            metadata_entry[MD_PROCESSOR] for metadata_entry in raw
        ]

        # this is used when using this ResultMetaData in a Core-only cache
        # retrieval context.  it's initialized on first cache retrieval
        # when the _result_disable_adapt_to_context execution option
        # (which the ORM generally sets) is not set.
        self._keymap_by_result_column_idx = None

        # for compiled SQL constructs, copy additional lookup keys into
        # the key lookup map, such as Column objects, labels,
        # column keys and other names
        if num_ctx_cols:
            # keymap by primary string...
            by_key = {
                metadata_entry[MD_LOOKUP_KEY]: metadata_entry
                for metadata_entry in raw
            }

            if len(by_key) != num_ctx_cols:
                # if by-primary-string dictionary smaller than
                # number of columns, assume we have dupes; (this check
                # is also in place if string dictionary is bigger, as
                # can occur when '*' was used as one of the compiled columns,
                # which may or may not be suggestive of dupes), rewrite
                # dupe records with "None" for index which results in
                # ambiguous column exception when accessed.
                #
                # this is considered to be the less common case as it is not
                # common to have dupe column keys in a SELECT statement.
                #
                # new in 1.4: get the complete set of all possible keys,
                # strings, objects, whatever, that are dupes across two
                # different records, first.
                index_by_key: Dict[Any, Any] = {}
                dupes = set()
                for metadata_entry in raw:
                    for key in (metadata_entry[MD_RENDERED_NAME],) + (
                        metadata_entry[MD_OBJECTS] or ()
                    ):
                        idx = metadata_entry[MD_INDEX]
                        # if this key has been associated with more than one
                        # positional index, it's a dupe
                        if index_by_key.setdefault(key, idx) != idx:
                            dupes.add(key)

                # then put everything we have into the keymap excluding only
                # those keys that are dupes.
                self._keymap = {
                    obj_elem: metadata_entry
                    for metadata_entry in raw
                    if metadata_entry[MD_OBJECTS]
                    for obj_elem in metadata_entry[MD_OBJECTS]
                    if obj_elem not in dupes
                }

                # then for the dupe keys, put the "ambiguous column"
                # record into by_key.
                by_key.update(
                    {
                        key: (None, None, [], key, key, None, None)
                        for key in dupes
                    }
                )

            else:
                # no dupes - copy secondary elements from compiled
                # columns into self._keymap.  this is the most common
                # codepath for Core / ORM statement executions before the
                # result metadata is cached
                self._keymap = {
                    obj_elem: metadata_entry
                    for metadata_entry in raw
                    if metadata_entry[MD_OBJECTS]
                    for obj_elem in metadata_entry[MD_OBJECTS]
                }
            # update keymap with primary string names taking
            # precedence
            self._keymap.update(by_key)
        else:
            # no compiled objects to map, just create keymap by primary string
            self._keymap = {
                metadata_entry[MD_LOOKUP_KEY]: metadata_entry
                for metadata_entry in raw
            }

        # update keymap with "translated" names.  In SQLAlchemy this is a
        # sqlite only thing, and in fact impacting only extremely old SQLite
        # versions unlikely to be present in modern Python versions.
        # however, the pyhive third party dialect is
        # also using this hook, which means others still might use it as well.
        # I dislike having this awkward hook here but as long as we need
        # to use names in cursor.description in some cases we need to have
        # some hook to accomplish this.
        if not num_ctx_cols and context._translate_colname:
            self._keymap.update(
                {
                    metadata_entry[MD_UNTRANSLATED]: self._keymap[
                        metadata_entry[MD_LOOKUP_KEY]
                    ]
                    for metadata_entry in raw
                    if metadata_entry[MD_UNTRANSLATED]
                }
            )

        self._key_to_index = self._make_key_to_index(self._keymap, MD_INDEX)

    def _merge_cursor_description(
        self,
        context,
        cursor_description,
        result_columns,
        num_ctx_cols,
        cols_are_ordered,
        textual_ordered,
        ad_hoc_textual,
        loose_column_name_matching,
    ):
        """Merge a cursor.description with compiled result column information.

        There are at least four separate strategies used here, selected
        depending on the type of SQL construct used to start with.

        The most common case is that of the compiled SQL expression construct,
        which generated the column names present in the raw SQL string and
        which has the identical number of columns as were reported by
        cursor.description.  In this case, we assume a 1-1 positional mapping
        between the entries in cursor.description and the compiled object.
        This is also the most performant case as we disregard extracting /
        decoding the column names present in cursor.description since we
        already have the desired name we generated in the compiled SQL
        construct.

        The next common case is that of the completely raw string SQL,
        such as passed to connection.execute().  In this case we have no
        compiled construct to work with, so we extract and decode the
        names from cursor.description and index those as the primary
        result row target keys.

        The remaining fairly common case is that of the textual SQL
        that includes at least partial column information; this is when
        we use a :class:`_expression.TextualSelect` construct.
        This construct may have
        unordered or ordered column information.  In the ordered case, we
        merge the cursor.description and the compiled construct's information
        positionally, and warn if there are additional description names
        present, however we still decode the names in cursor.description
        as we don't have a guarantee that the names in the columns match
        on these.   In the unordered case, we match names in cursor.description
        to that of the compiled construct based on name matching.
        In both of these cases, the cursor.description names and the column
        expression objects and names are indexed as result row target keys.

        The final case is much less common, where we have a compiled
        non-textual SQL expression construct, but the number of columns
        in cursor.description doesn't match what's in the compiled
        construct.  We make the guess here that there might be textual
        column expressions in the compiled construct that themselves include
        a comma in them causing them to split.  We do the same name-matching
        as with textual non-ordered columns.

        The name-matched system of merging is the same as that used by
        SQLAlchemy for all cases up through the 0.9 series.   Positional
        matching for compiled SQL expressions was introduced in 1.0 as a
        major performance feature, and positional matching for textual
        :class:`_expression.TextualSelect` objects in 1.1.
        As name matching is no longer
        a common case, it was acceptable to factor it into smaller generator-
        oriented methods that are easier to understand, but incur slightly
        more performance overhead.

        """

        if (
            num_ctx_cols
            and cols_are_ordered
            and not textual_ordered
            and num_ctx_cols == len(cursor_description)
        ):
            self._keys = [elem[0] for elem in result_columns]
            # pure positional 1-1 case; doesn't need to read
            # the names from cursor.description

            # most common case for Core and ORM

            # this metadata is safe to cache because we are guaranteed
            # to have the columns in the same order for new executions
            self._safe_for_cache = True
            return [
                (
                    idx,
                    idx,
                    rmap_entry[RM_OBJECTS],
                    rmap_entry[RM_NAME],
                    rmap_entry[RM_RENDERED_NAME],
                    context.get_result_processor(
                        rmap_entry[RM_TYPE],
                        rmap_entry[RM_RENDERED_NAME],
                        cursor_description[idx][1],
                    ),
                    None,
                )
                for idx, rmap_entry in enumerate(result_columns)
            ]
        else:
            # name-based or text-positional cases, where we need
            # to read cursor.description names

            if textual_ordered or (
                ad_hoc_textual and len(cursor_description) == num_ctx_cols
            ):
                self._safe_for_cache = True
                # textual positional case
                raw_iterator = self._merge_textual_cols_by_position(
                    context, cursor_description, result_columns
                )
            elif num_ctx_cols:
                # compiled SQL with a mismatch of description cols
                # vs. compiled cols, or textual w/ unordered columns
                # the order of columns can change if the query is
                # against a "select *", so not safe to cache
                self._safe_for_cache = False
                raw_iterator = self._merge_cols_by_name(
                    context,
                    cursor_description,
                    result_columns,
                    loose_column_name_matching,
                )
            else:
                # no compiled SQL, just a raw string, order of columns
                # can change for "select *"
                self._safe_for_cache = False
                raw_iterator = self._merge_cols_by_none(
                    context, cursor_description
                )

            return [
                (
                    idx,
                    ridx,
                    obj,
                    cursor_colname,
                    cursor_colname,
                    context.get_result_processor(
                        mapped_type, cursor_colname, coltype
                    ),
                    untranslated,
                )
                for (
                    idx,
                    ridx,
                    cursor_colname,
                    mapped_type,
                    coltype,
                    obj,
                    untranslated,
                ) in raw_iterator
            ]

    def _colnames_from_description(self, context, cursor_description):
        """Extract column names and data types from a cursor.description.

        Applies unicode decoding, column translation, "normalization",
        and case sensitivity rules to the names based on the dialect.

        """

        dialect = context.dialect
        translate_colname = context._translate_colname
        normalize_name = (
            dialect.normalize_name if dialect.requires_name_normalize else None
        )
        untranslated = None

        self._keys = []

        for idx, rec in enumerate(cursor_description):
            colname = rec[0]
            coltype = rec[1]

            if translate_colname:
                colname, untranslated = translate_colname(colname)

            if normalize_name:
                colname = normalize_name(colname)

            self._keys.append(colname)

            yield idx, colname, untranslated, coltype

    def _merge_textual_cols_by_position(
        self, context, cursor_description, result_columns
    ):
        num_ctx_cols = len(result_columns)

        if num_ctx_cols > len(cursor_description):
            util.warn(
                "Number of columns in textual SQL (%d) is "
                "smaller than number of columns requested (%d)"
                % (num_ctx_cols, len(cursor_description))
            )
        seen = set()

        for (
            idx,
            colname,
            untranslated,
            coltype,
        ) in self._colnames_from_description(context, cursor_description):
            if idx < num_ctx_cols:
                ctx_rec = result_columns[idx]
                obj = ctx_rec[RM_OBJECTS]
                ridx = idx
                mapped_type = ctx_rec[RM_TYPE]
                if obj[0] in seen:
                    raise exc.InvalidRequestError(
                        "Duplicate column expression requested "
                        "in textual SQL: %r" % obj[0]
                    )
                seen.add(obj[0])
            else:
                mapped_type = sqltypes.NULLTYPE
                obj = None
                ridx = None
            yield idx, ridx, colname, mapped_type, coltype, obj, untranslated

    def _merge_cols_by_name(
        self,
        context,
        cursor_description,
        result_columns,
        loose_column_name_matching,
    ):
        match_map = self._create_description_match_map(
            result_columns, loose_column_name_matching
        )
        mapped_type: TypeEngine[Any]

        for (
            idx,
            colname,
            untranslated,
            coltype,
        ) in self._colnames_from_description(context, cursor_description):
            try:
                ctx_rec = match_map[colname]
            except KeyError:
                mapped_type = sqltypes.NULLTYPE
                obj = None
                result_columns_idx = None
            else:
                obj = ctx_rec[1]
                mapped_type = ctx_rec[2]
                result_columns_idx = ctx_rec[3]
            yield (
                idx,
                result_columns_idx,
                colname,
                mapped_type,
                coltype,
                obj,
                untranslated,
            )

    @classmethod
    def _create_description_match_map(
        cls,
        result_columns: List[ResultColumnsEntry],
        loose_column_name_matching: bool = False,
    ) -> Dict[
        Union[str, object], Tuple[str, Tuple[Any, ...], TypeEngine[Any], int]
    ]:
        """when matching cursor.description to a set of names that are present
        in a Compiled object, as is the case with TextualSelect, get all the
        names we expect might match those in cursor.description.
        """

        d: Dict[
            Union[str, object],
            Tuple[str, Tuple[Any, ...], TypeEngine[Any], int],
        ] = {}
        for ridx, elem in enumerate(result_columns):
            key = elem[RM_RENDERED_NAME]
            if key in d:
                # conflicting keyname - just add the column-linked objects
                # to the existing record.  if there is a duplicate column
                # name in the cursor description, this will allow all of those
                # objects to raise an ambiguous column error
                e_name, e_obj, e_type, e_ridx = d[key]
                d[key] = e_name, e_obj + elem[RM_OBJECTS], e_type, ridx
            else:
                d[key] = (elem[RM_NAME], elem[RM_OBJECTS], elem[RM_TYPE], ridx)

            if loose_column_name_matching:
                # when using a textual statement with an unordered set
                # of columns that line up, we are expecting the user
                # to be using label names in the SQL that match to the column
                # expressions.  Enable more liberal matching for this case;
                # duplicate keys that are ambiguous will be fixed later.
                for r_key in elem[RM_OBJECTS]:
                    d.setdefault(
                        r_key,
                        (elem[RM_NAME], elem[RM_OBJECTS], elem[RM_TYPE], ridx),
                    )
        return d

    def _merge_cols_by_none(self, context, cursor_description):
        for (
            idx,
            colname,
            untranslated,
            coltype,
        ) in self._colnames_from_description(context, cursor_description):
            yield (
                idx,
                None,
                colname,
                sqltypes.NULLTYPE,
                coltype,
                None,
                untranslated,
            )

    if not TYPE_CHECKING:

        def _key_fallback(
            self, key: Any, err: Optional[Exception], raiseerr: bool = True
        ) -> Optional[NoReturn]:
            if raiseerr:
                if self._unpickled and isinstance(key, elements.ColumnElement):
                    raise exc.NoSuchColumnError(
                        "Row was unpickled; lookup by ColumnElement "
                        "is unsupported"
                    ) from err
                else:
                    raise exc.NoSuchColumnError(
                        "Could not locate column in row for column '%s'"
                        % util.string_or_unprintable(key)
                    ) from err
            else:
                return None

    def _raise_for_ambiguous_column_name(self, rec):
        raise exc.InvalidRequestError(
            "Ambiguous column name '%s' in "
            "result set column descriptions" % rec[MD_LOOKUP_KEY]
        )

    def _index_for_key(self, key: Any, raiseerr: bool = True) -> Optional[int]:
        # TODO: can consider pre-loading ints and negative ints
        # into _keymap - also no coverage here
        if isinstance(key, int):
            key = self._keys[key]

        try:
            rec = self._keymap[key]
        except KeyError as ke:
            x = self._key_fallback(key, ke, raiseerr)
            assert x is None
            return None

        index = rec[0]

        if index is None:
            self._raise_for_ambiguous_column_name(rec)
        return index

    def _indexes_for_keys(self, keys):
        try:
            return [self._keymap[key][0] for key in keys]
        except KeyError as ke:
            # ensure it raises
            CursorResultMetaData._key_fallback(self, ke.args[0], ke)

    def _metadata_for_keys(
        self, keys: Sequence[Any]
    ) -> Iterator[_NonAmbigCursorKeyMapRecType]:
        for key in keys:
            if int in key.__class__.__mro__:
                key = self._keys[key]

            try:
                rec = self._keymap[key]
            except KeyError as ke:
                # ensure it raises
                CursorResultMetaData._key_fallback(self, ke.args[0], ke)

            index = rec[MD_INDEX]

            if index is None:
                self._raise_for_ambiguous_column_name(rec)

            yield cast(_NonAmbigCursorKeyMapRecType, rec)

    def __getstate__(self):
        # TODO: consider serializing this as SimpleResultMetaData
        return {
            "_keymap": {
                key: (
                    rec[MD_INDEX],
                    rec[MD_RESULT_MAP_INDEX],
                    [],
                    key,
                    rec[MD_RENDERED_NAME],
                    None,
                    None,
                )
                for key, rec in self._keymap.items()
                if isinstance(key, (str, int))
            },
            "_keys": self._keys,
            "_translated_indexes": self._translated_indexes,
        }

    def __setstate__(self, state):
        self._processors = [None for _ in range(len(state["_keys"]))]
        self._keymap = state["_keymap"]
        self._keymap_by_result_column_idx = None
        self._key_to_index = self._make_key_to_index(self._keymap, MD_INDEX)
        self._keys = state["_keys"]
        self._unpickled = True
        if state["_translated_indexes"]:
            self._translated_indexes = cast(
                "List[int]", state["_translated_indexes"]
            )
            self._tuplefilter = tuplegetter(*self._translated_indexes)
        else:
            self._translated_indexes = self._tuplefilter = None


class ResultFetchStrategy:
    """Define a fetching strategy for a result object.


    .. versionadded:: 1.4

    """

    __slots__ = ()

    alternate_cursor_description: Optional[_DBAPICursorDescription] = None

    def soft_close(
        self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor]
    ) -> None:
        raise NotImplementedError()

    def hard_close(
        self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor]
    ) -> None:
        raise NotImplementedError()

    def yield_per(
        self,
        result: CursorResult[Any],
        dbapi_cursor: Optional[DBAPICursor],
        num: int,
    ) -> None:
        return

    def fetchone(
        self,
        result: CursorResult[Any],
        dbapi_cursor: DBAPICursor,
        hard_close: bool = False,
    ) -> Any:
        raise NotImplementedError()

    def fetchmany(
        self,
        result: CursorResult[Any],
        dbapi_cursor: DBAPICursor,
        size: Optional[int] = None,
    ) -> Any:
        raise NotImplementedError()

    def fetchall(
        self,
        result: CursorResult[Any],
        dbapi_cursor: DBAPICursor,
    ) -> Any:
        raise NotImplementedError()

    def handle_exception(
        self,
        result: CursorResult[Any],
        dbapi_cursor: Optional[DBAPICursor],
        err: BaseException,
    ) -> NoReturn:
        raise err


class NoCursorFetchStrategy(ResultFetchStrategy):
    """Cursor strategy for a result that has no open cursor.

    There are two varieties of this strategy, one for DQL and one for
    DML (and also DDL), each of which represent a result that had a cursor
    but no longer has one.

    """

    __slots__ = ()

    def soft_close(self, result, dbapi_cursor):
        pass

    def hard_close(self, result, dbapi_cursor):
        pass

    def fetchone(self, result, dbapi_cursor, hard_close=False):
        return self._non_result(result, None)

    def fetchmany(self, result, dbapi_cursor, size=None):
        return self._non_result(result, [])

    def fetchall(self, result, dbapi_cursor):
        return self._non_result(result, [])

    def _non_result(self, result, default, err=None):
        raise NotImplementedError()


class NoCursorDQLFetchStrategy(NoCursorFetchStrategy):
    """Cursor strategy for a DQL result that has no open cursor.

    This is a result set that can return rows, i.e. for a SELECT, or for an
    INSERT, UPDATE, DELETE that includes RETURNING. However it is in the state
    where the cursor is closed and no rows remain available.  The owning result
    object may or may not be "hard closed", which determines if the fetch
    methods send empty results or raise for closed result.

    """

    __slots__ = ()

    def _non_result(self, result, default, err=None):
        if result.closed:
            raise exc.ResourceClosedError(
                "This result object is closed."
            ) from err
        else:
            return default


_NO_CURSOR_DQL = NoCursorDQLFetchStrategy()


class NoCursorDMLFetchStrategy(NoCursorFetchStrategy):
    """Cursor strategy for a DML result that has no open cursor.

    This is a result set that does not return rows, i.e. for an INSERT,
    UPDATE, DELETE that does not include RETURNING.

    """

    __slots__ = ()

    def _non_result(self, result, default, err=None):
        # we only expect to have a _NoResultMetaData() here right now.
        assert not result._metadata.returns_rows
        result._metadata._we_dont_return_rows(err)


_NO_CURSOR_DML = NoCursorDMLFetchStrategy()


class CursorFetchStrategy(ResultFetchStrategy):
    """Call fetch methods from a DBAPI cursor.

    Alternate versions of this class may instead buffer the rows from
    cursors or not use cursors at all.

    """

    __slots__ = ()

    def soft_close(
        self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor]
    ) -> None:
        result.cursor_strategy = _NO_CURSOR_DQL

    def hard_close(
        self, result: CursorResult[Any], dbapi_cursor: Optional[DBAPICursor]
    ) -> None:
        result.cursor_strategy = _NO_CURSOR_DQL

    def handle_exception(
        self,
        result: CursorResult[Any],
        dbapi_cursor: Optional[DBAPICursor],
        err: BaseException,
    ) -> NoReturn:
        result.connection._handle_dbapi_exception(
            err, None, None, dbapi_cursor, result.context
        )

    def yield_per(
        self,
        result: CursorResult[Any],
        dbapi_cursor: Optional[DBAPICursor],
        num: int,
    ) -> None:
        result.cursor_strategy = BufferedRowCursorFetchStrategy(
            dbapi_cursor,
            {"max_row_buffer": num},
            initial_buffer=collections.deque(),
            growth_factor=0,
        )

    def fetchone(
        self,
        result: CursorResult[Any],
        dbapi_cursor: DBAPICursor,
        hard_close: bool = False,
    ) -> Any:
        try:
            row = dbapi_cursor.fetchone()
            if row is None:
                result._soft_close(hard=hard_close)
            return row
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)

    def fetchmany(
        self,
        result: CursorResult[Any],
        dbapi_cursor: DBAPICursor,
        size: Optional[int] = None,
    ) -> Any:
        try:
            if size is None:
                l = dbapi_cursor.fetchmany()
            else:
                l = dbapi_cursor.fetchmany(size)

            if not l:
                result._soft_close()
            return l
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)

    def fetchall(
        self,
        result: CursorResult[Any],
        dbapi_cursor: DBAPICursor,
    ) -> Any:
        try:
            rows = dbapi_cursor.fetchall()
            result._soft_close()
            return rows
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)


_DEFAULT_FETCH = CursorFetchStrategy()


class BufferedRowCursorFetchStrategy(CursorFetchStrategy):
    """A cursor fetch strategy with row buffering behavior.

    This strategy buffers the contents of a selection of rows
    before ``fetchone()`` is called.  This is to allow the results of
    ``cursor.description`` to be available immediately, when
    interfacing with a DB-API that requires rows to be consumed before
    this information is available (currently psycopg2, when used with
    server-side cursors).

    The pre-fetching behavior fetches only one row initially, and then
    grows its buffer size by a fixed amount with each successive need
    for additional rows up the ``max_row_buffer`` size, which defaults
    to 1000::

        with psycopg2_engine.connect() as conn:

            result = conn.execution_options(
                stream_results=True, max_row_buffer=50
            ).execute(text("select * from table"))

    .. versionadded:: 1.4 ``max_row_buffer`` may now exceed 1000 rows.

    .. seealso::

        :ref:`psycopg2_execution_options`
    """

    __slots__ = ("_max_row_buffer", "_rowbuffer", "_bufsize", "_growth_factor")

    def __init__(
        self,
        dbapi_cursor,
        execution_options,
        growth_factor=5,
        initial_buffer=None,
    ):
        self._max_row_buffer = execution_options.get("max_row_buffer", 1000)

        if initial_buffer is not None:
            self._rowbuffer = initial_buffer
        else:
            self._rowbuffer = collections.deque(dbapi_cursor.fetchmany(1))
        self._growth_factor = growth_factor

        if growth_factor:
            self._bufsize = min(self._max_row_buffer, self._growth_factor)
        else:
            self._bufsize = self._max_row_buffer

    @classmethod
    def create(cls, result):
        return BufferedRowCursorFetchStrategy(
            result.cursor,
            result.context.execution_options,
        )

    def _buffer_rows(self, result, dbapi_cursor):
        """this is currently used only by fetchone()."""

        size = self._bufsize
        try:
            if size < 1:
                new_rows = dbapi_cursor.fetchall()
            else:
                new_rows = dbapi_cursor.fetchmany(size)
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)

        if not new_rows:
            return
        self._rowbuffer = collections.deque(new_rows)
        if self._growth_factor and size < self._max_row_buffer:
            self._bufsize = min(
                self._max_row_buffer, size * self._growth_factor
            )

    def yield_per(self, result, dbapi_cursor, num):
        self._growth_factor = 0
        self._max_row_buffer = self._bufsize = num

    def soft_close(self, result, dbapi_cursor):
        self._rowbuffer.clear()
        super().soft_close(result, dbapi_cursor)

    def hard_close(self, result, dbapi_cursor):
        self._rowbuffer.clear()
        super().hard_close(result, dbapi_cursor)

    def fetchone(self, result, dbapi_cursor, hard_close=False):
        if not self._rowbuffer:
            self._buffer_rows(result, dbapi_cursor)
            if not self._rowbuffer:
                try:
                    result._soft_close(hard=hard_close)
                except BaseException as e:
                    self.handle_exception(result, dbapi_cursor, e)
                return None
        return self._rowbuffer.popleft()

    def fetchmany(self, result, dbapi_cursor, size=None):
        if size is None:
            return self.fetchall(result, dbapi_cursor)

        rb = self._rowbuffer
        lb = len(rb)
        close = False
        if size > lb:
            try:
                new = dbapi_cursor.fetchmany(size - lb)
            except BaseException as e:
                self.handle_exception(result, dbapi_cursor, e)
            else:
                if not new:
                    # defer closing since it may clear the row buffer
                    close = True
                else:
                    rb.extend(new)

        res = [rb.popleft() for _ in range(min(size, len(rb)))]
        if close:
            result._soft_close()
        return res

    def fetchall(self, result, dbapi_cursor):
        try:
            ret = list(self._rowbuffer) + list(dbapi_cursor.fetchall())
            self._rowbuffer.clear()
            result._soft_close()
            return ret
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)


class FullyBufferedCursorFetchStrategy(CursorFetchStrategy):
    """A cursor strategy that buffers rows fully upon creation.

    Used for operations where a result is to be delivered
    after the database conversation can not be continued,
    such as MSSQL INSERT...OUTPUT after an autocommit.

    """

    __slots__ = ("_rowbuffer", "alternate_cursor_description")

    def __init__(
        self,
        dbapi_cursor: Optional[DBAPICursor],
        alternate_description: Optional[_DBAPICursorDescription] = None,
        initial_buffer: Optional[Iterable[Any]] = None,
    ):
        self.alternate_cursor_description = alternate_description
        if initial_buffer is not None:
            self._rowbuffer = collections.deque(initial_buffer)
        else:
            assert dbapi_cursor is not None
            self._rowbuffer = collections.deque(dbapi_cursor.fetchall())

    def yield_per(self, result, dbapi_cursor, num):
        pass

    def soft_close(self, result, dbapi_cursor):
        self._rowbuffer.clear()
        super().soft_close(result, dbapi_cursor)

    def hard_close(self, result, dbapi_cursor):
        self._rowbuffer.clear()
        super().hard_close(result, dbapi_cursor)

    def fetchone(self, result, dbapi_cursor, hard_close=False):
        if self._rowbuffer:
            return self._rowbuffer.popleft()
        else:
            result._soft_close(hard=hard_close)
            return None

    def fetchmany(self, result, dbapi_cursor, size=None):
        if size is None:
            return self.fetchall(result, dbapi_cursor)

        rb = self._rowbuffer
        rows = [rb.popleft() for _ in range(min(size, len(rb)))]
        if not rows:
            result._soft_close()
        return rows

    def fetchall(self, result, dbapi_cursor):
        ret = self._rowbuffer
        self._rowbuffer = collections.deque()
        result._soft_close()
        return ret


class _NoResultMetaData(ResultMetaData):
    __slots__ = ()

    returns_rows = False

    def _we_dont_return_rows(self, err=None):
        raise exc.ResourceClosedError(
            "This result object does not return rows. "
            "It has been closed automatically."
        ) from err

    def _index_for_key(self, keys, raiseerr):
        self._we_dont_return_rows()

    def _metadata_for_keys(self, key):
        self._we_dont_return_rows()

    def _reduce(self, keys):
        self._we_dont_return_rows()

    @property
    def _keymap(self):
        self._we_dont_return_rows()

    @property
    def _key_to_index(self):
        self._we_dont_return_rows()

    @property
    def _processors(self):
        self._we_dont_return_rows()

    @property
    def keys(self):
        self._we_dont_return_rows()


_NO_RESULT_METADATA = _NoResultMetaData()


def null_dml_result() -> IteratorResult[Any]:
    it: IteratorResult[Any] = IteratorResult(_NoResultMetaData(), iter([]))
    it._soft_close()
    return it


class CursorResult(Result[_T]):
    """A Result that is representing state from a DBAPI cursor.

    .. versionchanged:: 1.4  The :class:`.CursorResult``
       class replaces the previous :class:`.ResultProxy` interface.
       This classes are based on the :class:`.Result` calling API
       which provides an updated usage model and calling facade for
       SQLAlchemy Core and SQLAlchemy ORM.

    Returns database rows via the :class:`.Row` class, which provides
    additional API features and behaviors on top of the raw data returned by
    the DBAPI.   Through the use of filters such as the :meth:`.Result.scalars`
    method, other kinds of objects may also be returned.

    .. seealso::

        :ref:`tutorial_selecting_data` - introductory material for accessing
        :class:`_engine.CursorResult` and :class:`.Row` objects.

    """

    __slots__ = (
        "context",
        "dialect",
        "cursor",
        "cursor_strategy",
        "_echo",
        "connection",
    )

    _metadata: Union[CursorResultMetaData, _NoResultMetaData]
    _no_result_metadata = _NO_RESULT_METADATA
    _soft_closed: bool = False
    closed: bool = False
    _is_cursor = True

    context: DefaultExecutionContext
    dialect: Dialect
    cursor_strategy: ResultFetchStrategy
    connection: Connection

    def __init__(
        self,
        context: DefaultExecutionContext,
        cursor_strategy: ResultFetchStrategy,
        cursor_description: Optional[_DBAPICursorDescription],
    ):
        self.context = context
        self.dialect = context.dialect
        self.cursor = context.cursor
        self.cursor_strategy = cursor_strategy
        self.connection = context.root_connection
        self._echo = echo = (
            self.connection._echo and context.engine._should_log_debug()
        )

        if cursor_description is not None:
            # inline of Result._row_getter(), set up an initial row
            # getter assuming no transformations will be called as this
            # is the most common case

            metadata = self._init_metadata(context, cursor_description)

            _make_row: Any
            _make_row = functools.partial(
                Row,
                metadata,
                metadata._effective_processors,
                metadata._key_to_index,
            )

            if context._num_sentinel_cols:
                sentinel_filter = operator.itemgetter(
                    slice(-context._num_sentinel_cols)
                )

                def _sliced_row(raw_data):
                    return _make_row(sentinel_filter(raw_data))

                sliced_row = _sliced_row
            else:
                sliced_row = _make_row

            if echo:
                log = self.context.connection._log_debug

                def _log_row(row):
                    log("Row %r", sql_util._repr_row(row))
                    return row

                self._row_logging_fn = _log_row

                def _make_row_2(row):
                    return _log_row(sliced_row(row))

                make_row = _make_row_2
            else:
                make_row = sliced_row
            self._set_memoized_attribute("_row_getter", make_row)

        else:
            assert context._num_sentinel_cols == 0
            self._metadata = self._no_result_metadata

    def _init_metadata(self, context, cursor_description):
        if context.compiled:
            compiled = context.compiled

            if compiled._cached_metadata:
                metadata = compiled._cached_metadata
            else:
                metadata = CursorResultMetaData(self, cursor_description)
                if metadata._safe_for_cache:
                    compiled._cached_metadata = metadata

            # result rewrite/ adapt step.  this is to suit the case
            # when we are invoked against a cached Compiled object, we want
            # to rewrite the ResultMetaData to reflect the Column objects
            # that are in our current SQL statement object, not the one
            # that is associated with the cached Compiled object.
            # the Compiled object may also tell us to not
            # actually do this step; this is to support the ORM where
            # it is to produce a new Result object in any case, and will
            # be using the cached Column objects against this database result
            # so we don't want to rewrite them.
            #
            # Basically this step suits the use case where the end user
            # is using Core SQL expressions and is accessing columns in the
            # result row using row._mapping[table.c.column].
            if (
                not context.execution_options.get(
                    "_result_disable_adapt_to_context", False
                )
                and compiled._result_columns
                and context.cache_hit is context.dialect.CACHE_HIT
                and compiled.statement is not context.invoked_statement
            ):
                metadata = metadata._adapt_to_context(context)

            self._metadata = metadata

        else:
            self._metadata = metadata = CursorResultMetaData(
                self, cursor_description
            )
        if self._echo:
            context.connection._log_debug(
                "Col %r", tuple(x[0] for x in cursor_description)
            )
        return metadata

    def _soft_close(self, hard=False):
        """Soft close this :class:`_engine.CursorResult`.

        This releases all DBAPI cursor resources, but leaves the
        CursorResult "open" from a semantic perspective, meaning the
        fetchXXX() methods will continue to return empty results.

        This method is called automatically when:

        * all result rows are exhausted using the fetchXXX() methods.
        * cursor.description is None.

        This method is **not public**, but is documented in order to clarify
        the "autoclose" process used.

        .. seealso::

            :meth:`_engine.CursorResult.close`


        """

        if (not hard and self._soft_closed) or (hard and self.closed):
            return

        if hard:
            self.closed = True
            self.cursor_strategy.hard_close(self, self.cursor)
        else:
            self.cursor_strategy.soft_close(self, self.cursor)

        if not self._soft_closed:
            cursor = self.cursor
            self.cursor = None  # type: ignore
            self.connection._safe_close_cursor(cursor)
            self._soft_closed = True

    @property
    def inserted_primary_key_rows(self):
        """Return the value of
        :attr:`_engine.CursorResult.inserted_primary_key`
        as a row contained within a list; some dialects may support a
        multiple row form as well.

        .. note:: As indicated below, in current SQLAlchemy versions this
           accessor is only useful beyond what's already supplied by
           :attr:`_engine.CursorResult.inserted_primary_key` when using the
           :ref:`postgresql_psycopg2` dialect.   Future versions hope to
           generalize this feature to more dialects.

        This accessor is added to support dialects that offer the feature
        that is currently implemented by the :ref:`psycopg2_executemany_mode`
        feature, currently **only the psycopg2 dialect**, which provides
        for many rows to be INSERTed at once while still retaining the
        behavior of being able to return server-generated primary key values.

        * **When using the psycopg2 dialect, or other dialects that may support
          "fast executemany" style inserts in upcoming releases** : When
          invoking an INSERT statement while passing a list of rows as the
          second argument to :meth:`_engine.Connection.execute`, this accessor
          will then provide a list of rows, where each row contains the primary
          key value for each row that was INSERTed.

        * **When using all other dialects / backends that don't yet support
          this feature**: This accessor is only useful for **single row INSERT
          statements**, and returns the same information as that of the
          :attr:`_engine.CursorResult.inserted_primary_key` within a
          single-element list. When an INSERT statement is executed in
          conjunction with a list of rows to be INSERTed, the list will contain
          one row per row inserted in the statement, however it will contain
          ``None`` for any server-generated values.

        Future releases of SQLAlchemy will further generalize the
        "fast execution helper" feature of psycopg2 to suit other dialects,
        thus allowing this accessor to be of more general use.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.CursorResult.inserted_primary_key`

        """
        if not self.context.compiled:
            raise exc.InvalidRequestError(
                "Statement is not a compiled expression construct."
            )
        elif not self.context.isinsert:
            raise exc.InvalidRequestError(
                "Statement is not an insert() expression construct."
            )
        elif self.context._is_explicit_returning:
            raise exc.InvalidRequestError(
                "Can't call inserted_primary_key "
                "when returning() "
                "is used."
            )
        return self.context.inserted_primary_key_rows

    @property
    def inserted_primary_key(self):
        """Return the primary key for the row just inserted.

        The return value is a :class:`_result.Row` object representing
        a named tuple of primary key values in the order in which the
        primary key columns are configured in the source
        :class:`_schema.Table`.

        .. versionchanged:: 1.4.8 - the
           :attr:`_engine.CursorResult.inserted_primary_key`
           value is now a named tuple via the :class:`_result.Row` class,
           rather than a plain tuple.

        This accessor only applies to single row :func:`_expression.insert`
        constructs which did not explicitly specify
        :meth:`_expression.Insert.returning`.    Support for multirow inserts,
        while not yet available for most backends, would be accessed using
        the :attr:`_engine.CursorResult.inserted_primary_key_rows` accessor.

        Note that primary key columns which specify a server_default clause, or
        otherwise do not qualify as "autoincrement" columns (see the notes at
        :class:`_schema.Column`), and were generated using the database-side
        default, will appear in this list as ``None`` unless the backend
        supports "returning" and the insert statement executed with the
        "implicit returning" enabled.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() construct.

        """

        if self.context.executemany:
            raise exc.InvalidRequestError(
                "This statement was an executemany call; if primary key "
                "returning is supported, please "
                "use .inserted_primary_key_rows."
            )

        ikp = self.inserted_primary_key_rows
        if ikp:
            return ikp[0]
        else:
            return None

    def last_updated_params(self):
        """Return the collection of updated parameters from this
        execution.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an update() construct.

        """
        if not self.context.compiled:
            raise exc.InvalidRequestError(
                "Statement is not a compiled expression construct."
            )
        elif not self.context.isupdate:
            raise exc.InvalidRequestError(
                "Statement is not an update() expression construct."
            )
        elif self.context.executemany:
            return self.context.compiled_parameters
        else:
            return self.context.compiled_parameters[0]

    def last_inserted_params(self):
        """Return the collection of inserted parameters from this
        execution.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() construct.

        """
        if not self.context.compiled:
            raise exc.InvalidRequestError(
                "Statement is not a compiled expression construct."
            )
        elif not self.context.isinsert:
            raise exc.InvalidRequestError(
                "Statement is not an insert() expression construct."
            )
        elif self.context.executemany:
            return self.context.compiled_parameters
        else:
            return self.context.compiled_parameters[0]

    @property
    def returned_defaults_rows(self):
        """Return a list of rows each containing the values of default
        columns that were fetched using
        the :meth:`.ValuesBase.return_defaults` feature.

        The return value is a list of :class:`.Row` objects.

        .. versionadded:: 1.4

        """
        return self.context.returned_default_rows

    def splice_horizontally(self, other):
        """Return a new :class:`.CursorResult` that "horizontally splices"
        together the rows of this :class:`.CursorResult` with that of another
        :class:`.CursorResult`.

        .. tip::  This method is for the benefit of the SQLAlchemy ORM and is
           not intended for general use.

        "horizontally splices" means that for each row in the first and second
        result sets, a new row that concatenates the two rows together is
        produced, which then becomes the new row.  The incoming
        :class:`.CursorResult` must have the identical number of rows.  It is
        typically expected that the two result sets come from the same sort
        order as well, as the result rows are spliced together based on their
        position in the result.

        The expected use case here is so that multiple INSERT..RETURNING
        statements (which definitely need to be sorted) against different
        tables can produce a single result that looks like a JOIN of those two
        tables.

        E.g.::

            r1 = connection.execute(
                users.insert().returning(
                    users.c.user_name, users.c.user_id, sort_by_parameter_order=True
                ),
                user_values,
            )

            r2 = connection.execute(
                addresses.insert().returning(
                    addresses.c.address_id,
                    addresses.c.address,
                    addresses.c.user_id,
                    sort_by_parameter_order=True,
                ),
                address_values,
            )

            rows = r1.splice_horizontally(r2).all()
            assert rows == [
                ("john", 1, 1, "foo@bar.com", 1),
                ("jack", 2, 2, "bar@bat.com", 2),
            ]

        .. versionadded:: 2.0

        .. seealso::

            :meth:`.CursorResult.splice_vertically`


        """  # noqa: E501

        clone = self._generate()
        total_rows = [
            tuple(r1) + tuple(r2)
            for r1, r2 in zip(
                list(self._raw_row_iterator()),
                list(other._raw_row_iterator()),
            )
        ]

        clone._metadata = clone._metadata._splice_horizontally(other._metadata)

        clone.cursor_strategy = FullyBufferedCursorFetchStrategy(
            None,
            initial_buffer=total_rows,
        )
        clone._reset_memoizations()
        return clone

    def splice_vertically(self, other):
        """Return a new :class:`.CursorResult` that "vertically splices",
        i.e. "extends", the rows of this :class:`.CursorResult` with that of
        another :class:`.CursorResult`.

        .. tip::  This method is for the benefit of the SQLAlchemy ORM and is
           not intended for general use.

        "vertically splices" means the rows of the given result are appended to
        the rows of this cursor result. The incoming :class:`.CursorResult`
        must have rows that represent the identical list of columns in the
        identical order as they are in this :class:`.CursorResult`.

        .. versionadded:: 2.0

        .. seealso::

            :meth:`.CursorResult.splice_horizontally`

        """
        clone = self._generate()
        total_rows = list(self._raw_row_iterator()) + list(
            other._raw_row_iterator()
        )

        clone.cursor_strategy = FullyBufferedCursorFetchStrategy(
            None,
            initial_buffer=total_rows,
        )
        clone._reset_memoizations()
        return clone

    def _rewind(self, rows):
        """rewind this result back to the given rowset.

        this is used internally for the case where an :class:`.Insert`
        construct combines the use of
        :meth:`.Insert.return_defaults` along with the
        "supplemental columns" feature.

        """

        if self._echo:
            self.context.connection._log_debug(
                "CursorResult rewound %d row(s)", len(rows)
            )

        # the rows given are expected to be Row objects, so we
        # have to clear out processors which have already run on these
        # rows
        self._metadata = cast(
            CursorResultMetaData, self._metadata
        )._remove_processors()

        self.cursor_strategy = FullyBufferedCursorFetchStrategy(
            None,
            # TODO: if these are Row objects, can we save on not having to
            # re-make new Row objects out of them a second time?  is that
            # what's actually happening right now?  maybe look into this
            initial_buffer=rows,
        )
        self._reset_memoizations()
        return self

    @property
    def returned_defaults(self):
        """Return the values of default columns that were fetched using
        the :meth:`.ValuesBase.return_defaults` feature.

        The value is an instance of :class:`.Row`, or ``None``
        if :meth:`.ValuesBase.return_defaults` was not used or if the
        backend does not support RETURNING.

        .. seealso::

            :meth:`.ValuesBase.return_defaults`

        """

        if self.context.executemany:
            raise exc.InvalidRequestError(
                "This statement was an executemany call; if return defaults "
                "is supported, please use .returned_defaults_rows."
            )

        rows = self.context.returned_default_rows
        if rows:
            return rows[0]
        else:
            return None

    def lastrow_has_defaults(self):
        """Return ``lastrow_has_defaults()`` from the underlying
        :class:`.ExecutionContext`.

        See :class:`.ExecutionContext` for details.

        """

        return self.context.lastrow_has_defaults()

    def postfetch_cols(self):
        """Return ``postfetch_cols()`` from the underlying
        :class:`.ExecutionContext`.

        See :class:`.ExecutionContext` for details.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() or update() construct.

        """

        if not self.context.compiled:
            raise exc.InvalidRequestError(
                "Statement is not a compiled expression construct."
            )
        elif not self.context.isinsert and not self.context.isupdate:
            raise exc.InvalidRequestError(
                "Statement is not an insert() or update() "
                "expression construct."
            )
        return self.context.postfetch_cols

    def prefetch_cols(self):
        """Return ``prefetch_cols()`` from the underlying
        :class:`.ExecutionContext`.

        See :class:`.ExecutionContext` for details.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() or update() construct.

        """

        if not self.context.compiled:
            raise exc.InvalidRequestError(
                "Statement is not a compiled expression construct."
            )
        elif not self.context.isinsert and not self.context.isupdate:
            raise exc.InvalidRequestError(
                "Statement is not an insert() or update() "
                "expression construct."
            )
        return self.context.prefetch_cols

    def supports_sane_rowcount(self):
        """Return ``supports_sane_rowcount`` from the dialect.

        See :attr:`_engine.CursorResult.rowcount` for background.

        """

        return self.dialect.supports_sane_rowcount

    def supports_sane_multi_rowcount(self):
        """Return ``supports_sane_multi_rowcount`` from the dialect.

        See :attr:`_engine.CursorResult.rowcount` for background.

        """

        return self.dialect.supports_sane_multi_rowcount

    @util.memoized_property
    def rowcount(self) -> int:
        """Return the 'rowcount' for this result.

        The primary purpose of 'rowcount' is to report the number of rows
        matched by the WHERE criterion of an UPDATE or DELETE statement
        executed once (i.e. for a single parameter set), which may then be
        compared to the number of rows expected to be updated or deleted as a
        means of asserting data integrity.

        This attribute is transferred from the ``cursor.rowcount`` attribute
        of the DBAPI before the cursor is closed, to support DBAPIs that
        don't make this value available after cursor close.   Some DBAPIs may
        offer meaningful values for other kinds of statements, such as INSERT
        and SELECT statements as well.  In order to retrieve ``cursor.rowcount``
        for these statements, set the
        :paramref:`.Connection.execution_options.preserve_rowcount`
        execution option to True, which will cause the ``cursor.rowcount``
        value to be unconditionally memoized before any results are returned
        or the cursor is closed, regardless of statement type.

        For cases where the DBAPI does not support rowcount for a particular
        kind of statement and/or execution, the returned value will be ``-1``,
        which is delivered directly from the DBAPI and is part of :pep:`249`.
        All DBAPIs should support rowcount for single-parameter-set
        UPDATE and DELETE statements, however.

        .. note::

           Notes regarding :attr:`_engine.CursorResult.rowcount`:


           * This attribute returns the number of rows *matched*,
             which is not necessarily the same as the number of rows
             that were actually *modified*. For example, an UPDATE statement
             may have no net change on a given row if the SET values
             given are the same as those present in the row already.
             Such a row would be matched but not modified.
             On backends that feature both styles, such as MySQL,
             rowcount is configured to return the match
             count in all cases.

           * :attr:`_engine.CursorResult.rowcount` in the default case is
             *only* useful in conjunction with an UPDATE or DELETE statement,
             and only with a single set of parameters. For other kinds of
             statements, SQLAlchemy will not attempt to pre-memoize the value
             unless the
             :paramref:`.Connection.execution_options.preserve_rowcount`
             execution option is used.  Note that contrary to :pep:`249`, many
             DBAPIs do not support rowcount values for statements that are not
             UPDATE or DELETE, particularly when rows are being returned which
             are not fully pre-buffered.   DBAPIs that dont support rowcount
             for a particular kind of statement should return the value ``-1``
             for such statements.

           * :attr:`_engine.CursorResult.rowcount` may not be meaningful
             when executing a single statement with multiple parameter sets
             (i.e. an :term:`executemany`). Most DBAPIs do not sum "rowcount"
             values across multiple parameter sets and will return ``-1``
             when accessed.

           * SQLAlchemy's :ref:`engine_insertmanyvalues` feature does support
             a correct population of :attr:`_engine.CursorResult.rowcount`
             when the :paramref:`.Connection.execution_options.preserve_rowcount`
             execution option is set to True.

           * Statements that use RETURNING may not support rowcount, returning
             a ``-1`` value instead.

        .. seealso::

            :ref:`tutorial_update_delete_rowcount` - in the :ref:`unified_tutorial`

            :paramref:`.Connection.execution_options.preserve_rowcount`

        """  # noqa: E501
        try:
            return self.context.rowcount
        except BaseException as e:
            self.cursor_strategy.handle_exception(self, self.cursor, e)
            raise  # not called

    @property
    def lastrowid(self):
        """Return the 'lastrowid' accessor on the DBAPI cursor.

        This is a DBAPI specific method and is only functional
        for those backends which support it, for statements
        where it is appropriate.  It's behavior is not
        consistent across backends.

        Usage of this method is normally unnecessary when
        using insert() expression constructs; the
        :attr:`~CursorResult.inserted_primary_key` attribute provides a
        tuple of primary key values for a newly inserted row,
        regardless of database backend.

        """
        try:
            return self.context.get_lastrowid()
        except BaseException as e:
            self.cursor_strategy.handle_exception(self, self.cursor, e)

    @property
    def returns_rows(self):
        """True if this :class:`_engine.CursorResult` returns zero or more
        rows.

        I.e. if it is legal to call the methods
        :meth:`_engine.CursorResult.fetchone`,
        :meth:`_engine.CursorResult.fetchmany`
        :meth:`_engine.CursorResult.fetchall`.

        Overall, the value of :attr:`_engine.CursorResult.returns_rows` should
        always be synonymous with whether or not the DBAPI cursor had a
        ``.description`` attribute, indicating the presence of result columns,
        noting that a cursor that returns zero rows still has a
        ``.description`` if a row-returning statement was emitted.

        This attribute should be True for all results that are against
        SELECT statements, as well as for DML statements INSERT/UPDATE/DELETE
        that use RETURNING.   For INSERT/UPDATE/DELETE statements that were
        not using RETURNING, the value will usually be False, however
        there are some dialect-specific exceptions to this, such as when
        using the MSSQL / pyodbc dialect a SELECT is emitted inline in
        order to retrieve an inserted primary key value.


        """
        return self._metadata.returns_rows

    @property
    def is_insert(self):
        """True if this :class:`_engine.CursorResult` is the result
        of a executing an expression language compiled
        :func:`_expression.insert` construct.

        When True, this implies that the
        :attr:`inserted_primary_key` attribute is accessible,
        assuming the statement did not include
        a user defined "returning" construct.

        """
        return self.context.isinsert

    def _fetchiter_impl(self):
        fetchone = self.cursor_strategy.fetchone

        while True:
            row = fetchone(self, self.cursor)
            if row is None:
                break
            yield row

    def _fetchone_impl(self, hard_close=False):
        return self.cursor_strategy.fetchone(self, self.cursor, hard_close)

    def _fetchall_impl(self):
        return self.cursor_strategy.fetchall(self, self.cursor)

    def _fetchmany_impl(self, size=None):
        return self.cursor_strategy.fetchmany(self, self.cursor, size)

    def _raw_row_iterator(self):
        return self._fetchiter_impl()

    def merge(self, *others: Result[Any]) -> MergedResult[Any]:
        merged_result = super().merge(*others)
        if self.context._has_rowcount:
            merged_result.rowcount = sum(
                cast("CursorResult[Any]", result).rowcount
                for result in (self,) + others
            )
        return merged_result

    def close(self) -> Any:
        """Close this :class:`_engine.CursorResult`.

        This closes out the underlying DBAPI cursor corresponding to the
        statement execution, if one is still present.  Note that the DBAPI
        cursor is automatically released when the :class:`_engine.CursorResult`
        exhausts all available rows.  :meth:`_engine.CursorResult.close` is
        generally an optional method except in the case when discarding a
        :class:`_engine.CursorResult` that still has additional rows pending
        for fetch.

        After this method is called, it is no longer valid to call upon
        the fetch methods, which will raise a :class:`.ResourceClosedError`
        on subsequent use.

        .. seealso::

            :ref:`connections_toplevel`

        """
        self._soft_close(hard=True)

    @_generative
    def yield_per(self, num: int) -> Self:
        self._yield_per = num
        self.cursor_strategy.yield_per(self, self.cursor, num)
        return self


ResultProxy = CursorResult

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\manualintegrate.py ===
"""Integration method that emulates by-hand techniques.

This module also provides functionality to get the steps used to evaluate a
particular integral, in the ``integral_steps`` function. This will return
nested ``Rule`` s representing the integration rules used.

Each ``Rule`` class represents a (maybe parametrized) integration rule, e.g.
``SinRule`` for integrating ``sin(x)`` and ``ReciprocalSqrtQuadraticRule``
for integrating ``1/sqrt(a+b*x+c*x**2)``. The ``eval`` method returns the
integration result.

The ``manualintegrate`` function computes the integral by calling ``eval``
on the rule returned by ``integral_steps``.

The integrator can be extended with new heuristics and evaluation
techniques. To do so, extend the ``Rule`` class, implement ``eval`` method,
then write a function that accepts an ``IntegralInfo`` object and returns
either a ``Rule`` instance or ``None``. If the new technique requires a new
match, add the key and call to the antiderivative function to integral_steps.
To enable simple substitutions, add the match to find_substitutions.

"""

from __future__ import annotations
from typing import NamedTuple, Type, Callable, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Number, E
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne, Boolean
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction, csch,
    cosh, coth, sech, sinh, tanh, asinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (TrigonometricFunction,
    cos, sin, tan, cot, csc, sec, acos, asin, atan, acot, acsc, asec)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.error_functions import (erf, erfi, fresnelc,
    fresnels, Ci, Chi, Si, Shi, Ei, li)
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.elliptic_integrals import elliptic_e, elliptic_f
from sympy.functions.special.polynomials import (chebyshevt, chebyshevu,
    legendre, hermite, laguerre, assoc_laguerre, gegenbauer, jacobi,
    OrthogonalPolynomial)
from sympy.functions.special.zeta_functions import polylog
from .integrals import Integral
from sympy.logic.boolalg import And
from sympy.ntheory.factor_ import primefactors
from sympy.polys.polytools import degree, lcm_list, gcd_list, Poly
from sympy.simplify.radsimp import fraction
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.strategies.core import switch, do_one, null_safe, condition
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug


@dataclass
class Rule(ABC):
    integrand: Expr
    variable: Symbol

    @abstractmethod
    def eval(self) -> Expr:
        pass

    @abstractmethod
    def contains_dont_know(self) -> bool:
        pass


@dataclass
class AtomicRule(Rule, ABC):
    """A simple rule that does not depend on other rules"""
    def contains_dont_know(self) -> bool:
        return False


@dataclass
class ConstantRule(AtomicRule):
    """integrate(a, x)  ->  a*x"""
    def eval(self) -> Expr:
        return self.integrand * self.variable


@dataclass
class ConstantTimesRule(Rule):
    """integrate(a*f(x), x)  ->  a*integrate(f(x), x)"""
    constant: Expr
    other: Expr
    substep: Rule

    def eval(self) -> Expr:
        return self.constant * self.substep.eval()

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class PowerRule(AtomicRule):
    """integrate(x**a, x)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        return Piecewise(
            ((self.base**(self.exp + 1))/(self.exp + 1), Ne(self.exp, -1)),
            (log(self.base), True),
        )


@dataclass
class NestedPowRule(AtomicRule):
    """integrate((x**a)**b, x)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        m = self.base * self.integrand
        return Piecewise((m / (self.exp + 1), Ne(self.exp, -1)),
                         (m * log(self.base), True))


@dataclass
class AddRule(Rule):
    """integrate(f(x) + g(x), x) -> integrate(f(x), x) + integrate(g(x), x)"""
    substeps: list[Rule]

    def eval(self) -> Expr:
        return Add(*(substep.eval() for substep in self.substeps))

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.substeps)


@dataclass
class URule(Rule):
    """integrate(f(g(x))*g'(x), x) -> integrate(f(u), u), u = g(x)"""
    u_var: Symbol
    u_func: Expr
    substep: Rule

    def eval(self) -> Expr:
        result = self.substep.eval()
        if self.u_func.is_Pow:
            base, exp_ = self.u_func.as_base_exp()
            if exp_ == -1:
                # avoid needless -log(1/x) from substitution
                result = result.subs(log(self.u_var), -log(base))
        return result.subs(self.u_var, self.u_func)

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class PartsRule(Rule):
    """integrate(u(x)*v'(x), x) -> u(x)*v(x) - integrate(u'(x)*v(x), x)"""
    u: Symbol
    dv: Expr
    v_step: Rule
    second_step: Rule | None  # None when is a substep of CyclicPartsRule

    def eval(self) -> Expr:
        assert self.second_step is not None
        v = self.v_step.eval()
        return self.u * v - self.second_step.eval()

    def contains_dont_know(self) -> bool:
        return self.v_step.contains_dont_know() or (
            self.second_step is not None and self.second_step.contains_dont_know())


@dataclass
class CyclicPartsRule(Rule):
    """Apply PartsRule multiple times to integrate exp(x)*sin(x)"""
    parts_rules: list[PartsRule]
    coefficient: Expr

    def eval(self) -> Expr:
        result = []
        sign = 1
        for rule in self.parts_rules:
            result.append(sign * rule.u * rule.v_step.eval())
            sign *= -1
        return Add(*result) / (1 - self.coefficient)

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.parts_rules)


@dataclass
class TrigRule(AtomicRule, ABC):
    pass


@dataclass
class SinRule(TrigRule):
    """integrate(sin(x), x) -> -cos(x)"""
    def eval(self) -> Expr:
        return -cos(self.variable)


@dataclass
class CosRule(TrigRule):
    """integrate(cos(x), x) -> sin(x)"""
    def eval(self) -> Expr:
        return sin(self.variable)


@dataclass
class SecTanRule(TrigRule):
    """integrate(sec(x)*tan(x), x) -> sec(x)"""
    def eval(self) -> Expr:
        return sec(self.variable)


@dataclass
class CscCotRule(TrigRule):
    """integrate(csc(x)*cot(x), x) -> -csc(x)"""
    def eval(self) -> Expr:
        return -csc(self.variable)


@dataclass
class Sec2Rule(TrigRule):
    """integrate(sec(x)**2, x) -> tan(x)"""
    def eval(self) -> Expr:
        return tan(self.variable)


@dataclass
class Csc2Rule(TrigRule):
    """integrate(csc(x)**2, x) -> -cot(x)"""
    def eval(self) -> Expr:
        return -cot(self.variable)


@dataclass
class HyperbolicRule(AtomicRule, ABC):
    pass


@dataclass
class SinhRule(HyperbolicRule):
    """integrate(sinh(x), x) -> cosh(x)"""
    def eval(self) -> Expr:
        return cosh(self.variable)


@dataclass
class CoshRule(HyperbolicRule):
    """integrate(cosh(x), x) -> sinh(x)"""
    def eval(self):
        return sinh(self.variable)


@dataclass
class ExpRule(AtomicRule):
    """integrate(a**x, x) -> a**x/ln(a)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        return self.integrand / log(self.base)


@dataclass
class ReciprocalRule(AtomicRule):
    """integrate(1/x, x) -> ln(x)"""
    base: Expr

    def eval(self) -> Expr:
        return log(self.base)


@dataclass
class ArcsinRule(AtomicRule):
    """integrate(1/sqrt(1-x**2), x) -> asin(x)"""
    def eval(self) -> Expr:
        return asin(self.variable)


@dataclass
class ArcsinhRule(AtomicRule):
    """integrate(1/sqrt(1+x**2), x) -> asin(x)"""
    def eval(self) -> Expr:
        return asinh(self.variable)


@dataclass
class ReciprocalSqrtQuadraticRule(AtomicRule):
    """integrate(1/sqrt(a+b*x+c*x**2), x) -> log(2*sqrt(c)*sqrt(a+b*x+c*x**2)+b+2*c*x)/sqrt(c)"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return log(2*sqrt(c)*sqrt(a+b*x+c*x**2)+b+2*c*x)/sqrt(c)


@dataclass
class SqrtQuadraticDenomRule(AtomicRule):
    """integrate(poly(x)/sqrt(a+b*x+c*x**2), x)"""
    a: Expr
    b: Expr
    c: Expr
    coeffs: list[Expr]

    def eval(self) -> Expr:
        a, b, c, coeffs, x = self.a, self.b, self.c, self.coeffs.copy(), self.variable
        # Integrate poly/sqrt(a+b*x+c*x**2) using recursion.
        # coeffs are coefficients of the polynomial.
        # Let I_n = x**n/sqrt(a+b*x+c*x**2), then
        # I_n = A * x**(n-1)*sqrt(a+b*x+c*x**2) - B * I_{n-1} - C * I_{n-2}
        # where A = 1/(n*c), B = (2*n-1)*b/(2*n*c), C = (n-1)*a/(n*c)
        # See https://github.com/sympy/sympy/pull/23608 for proof.
        result_coeffs = []
        coeffs = coeffs.copy()
        for i in range(len(coeffs)-2):
            n = len(coeffs)-1-i
            coeff = coeffs[i]/(c*n)
            result_coeffs.append(coeff)
            coeffs[i+1] -= (2*n-1)*b/2*coeff
            coeffs[i+2] -= (n-1)*a*coeff
        d, e = coeffs[-1], coeffs[-2]
        s = sqrt(a+b*x+c*x**2)
        constant = d-b*e/(2*c)
        if constant == 0:
            I0 = 0
        else:
            step = inverse_trig_rule(IntegralInfo(1/s, x), degenerate=False)
            I0 = constant*step.eval()
        return Add(*(result_coeffs[i]*x**(len(coeffs)-2-i)
                     for i in range(len(result_coeffs))), e/c)*s + I0


@dataclass
class SqrtQuadraticRule(AtomicRule):
    """integrate(sqrt(a+b*x+c*x**2), x)"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        step = sqrt_quadratic_rule(IntegralInfo(self.integrand, self.variable), degenerate=False)
        return step.eval()


@dataclass
class AlternativeRule(Rule):
    """Multiple ways to do integration."""
    alternatives: list[Rule]

    def eval(self) -> Expr:
        return self.alternatives[0].eval()

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.alternatives)


@dataclass
class DontKnowRule(Rule):
    """Leave the integral as is."""
    def eval(self) -> Expr:
        return Integral(self.integrand, self.variable)

    def contains_dont_know(self) -> bool:
        return True


@dataclass
class DerivativeRule(AtomicRule):
    """integrate(f'(x), x) -> f(x)"""
    def eval(self) -> Expr:
        assert isinstance(self.integrand, Derivative)
        variable_count = list(self.integrand.variable_count)
        for i, (var, count) in enumerate(variable_count):
            if var == self.variable:
                variable_count[i] = (var, count - 1)
                break
        return Derivative(self.integrand.expr, *variable_count)


@dataclass
class RewriteRule(Rule):
    """Rewrite integrand to another form that is easier to handle."""
    rewritten: Expr
    substep: Rule

    def eval(self) -> Expr:
        return self.substep.eval()

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class CompleteSquareRule(RewriteRule):
    """Rewrite a+b*x+c*x**2 to a-b**2/(4*c) + c*(x+b/(2*c))**2"""
    pass


@dataclass
class PiecewiseRule(Rule):
    subfunctions: Sequence[tuple[Rule, bool | Boolean]]

    def eval(self) -> Expr:
        return Piecewise(*[(substep.eval(), cond)
                           for substep, cond in self.subfunctions])

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep, _ in self.subfunctions)


@dataclass
class HeavisideRule(Rule):
    harg: Expr
    ibnd: Expr
    substep: Rule

    def eval(self) -> Expr:
        # If we are integrating over x and the integrand has the form
        #       Heaviside(m*x+b)*g(x) == Heaviside(harg)*g(symbol)
        # then there needs to be continuity at -b/m == ibnd,
        # so we subtract the appropriate term.
        result = self.substep.eval()
        return Heaviside(self.harg) * (result - result.subs(self.variable, self.ibnd))

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class DiracDeltaRule(AtomicRule):
    n: Expr
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        n, a, b, x = self.n, self.a, self.b, self.variable
        if n == 0:
            return Heaviside(a+b*x)/b
        return DiracDelta(a+b*x, n-1)/b


@dataclass
class TrigSubstitutionRule(Rule):
    theta: Expr
    func: Expr
    rewritten: Expr
    substep: Rule
    restriction: bool | Boolean

    def eval(self) -> Expr:
        theta, func, x = self.theta, self.func, self.variable
        func = func.subs(sec(theta), 1/cos(theta))
        func = func.subs(csc(theta), 1/sin(theta))
        func = func.subs(cot(theta), 1/tan(theta))

        trig_function = list(func.find(TrigonometricFunction))
        assert len(trig_function) == 1
        trig_function = trig_function[0]
        relation = solve(x - func, trig_function)
        assert len(relation) == 1
        numer, denom = fraction(relation[0])

        if isinstance(trig_function, sin):
            opposite = numer
            hypotenuse = denom
            adjacent = sqrt(denom**2 - numer**2)
            inverse = asin(relation[0])
        elif isinstance(trig_function, cos):
            adjacent = numer
            hypotenuse = denom
            opposite = sqrt(denom**2 - numer**2)
            inverse = acos(relation[0])
        else:  # tan
            opposite = numer
            adjacent = denom
            hypotenuse = sqrt(denom**2 + numer**2)
            inverse = atan(relation[0])

        substitution = [
            (sin(theta), opposite/hypotenuse),
            (cos(theta), adjacent/hypotenuse),
            (tan(theta), opposite/adjacent),
            (theta, inverse)
        ]
        return Piecewise(
                (self.substep.eval().subs(substitution).trigsimp(), self.restriction) # type: ignore
        )

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class ArctanRule(AtomicRule):
    """integrate(a/(b*x**2+c), x) -> a/b / sqrt(c/b) * atan(x/sqrt(c/b))"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return a/b / sqrt(c/b) * atan(x/sqrt(c/b))


@dataclass
class OrthogonalPolyRule(AtomicRule, ABC):
    n: Expr


@dataclass
class JacobiRule(OrthogonalPolyRule):
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        n, a, b, x = self.n, self.a, self.b, self.variable
        return Piecewise(
            (2*jacobi(n + 1, a - 1, b - 1, x)/(n + a + b), Ne(n + a + b, 0)),
            (x, Eq(n, 0)),
            ((a + b + 2)*x**2/4 + (a - b)*x/2, Eq(n, 1)))


@dataclass
class GegenbauerRule(OrthogonalPolyRule):
    a: Expr

    def eval(self) -> Expr:
        n, a, x = self.n, self.a, self.variable
        return Piecewise(
            (gegenbauer(n + 1, a - 1, x)/(2*(a - 1)), Ne(a, 1)),
            (chebyshevt(n + 1, x)/(n + 1), Ne(n, -1)),
            (S.Zero, True))


@dataclass
class ChebyshevTRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return Piecewise(
            ((chebyshevt(n + 1, x)/(n + 1) -
              chebyshevt(n - 1, x)/(n - 1))/2, Ne(Abs(n), 1)),
            (x**2/2, True))


@dataclass
class ChebyshevURule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return Piecewise(
            (chebyshevt(n + 1, x)/(n + 1), Ne(n, -1)),
            (S.Zero, True))


@dataclass
class LegendreRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return(legendre(n + 1, x) - legendre(n - 1, x))/(2*n + 1)


@dataclass
class HermiteRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return hermite(n + 1, x)/(2*(n + 1))


@dataclass
class LaguerreRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return laguerre(n, x) - laguerre(n + 1, x)


@dataclass
class AssocLaguerreRule(OrthogonalPolyRule):
    a: Expr

    def eval(self) -> Expr:
        return -assoc_laguerre(self.n + 1, self.a - 1, self.variable)


@dataclass
class IRule(AtomicRule, ABC):
    a: Expr
    b: Expr


@dataclass
class CiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return cos(b)*Ci(a*x) - sin(b)*Si(a*x)


@dataclass
class ChiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return cosh(b)*Chi(a*x) + sinh(b)*Shi(a*x)


@dataclass
class EiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return exp(b)*Ei(a*x)


@dataclass
class SiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return sin(b)*Ci(a*x) + cos(b)*Si(a*x)


@dataclass
class ShiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return sinh(b)*Chi(a*x) + cosh(b)*Shi(a*x)


@dataclass
class LiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return li(a*x + b)/a


@dataclass
class ErfRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        if a.is_extended_real:
            return Piecewise(
                (sqrt(S.Pi)/sqrt(-a)/2 * exp(c - b**2/(4*a)) *
                    erf((-2*a*x - b)/(2*sqrt(-a))), a < 0),
                (sqrt(S.Pi)/sqrt(a)/2 * exp(c - b**2/(4*a)) *
                    erfi((2*a*x + b)/(2*sqrt(a))), True))
        return sqrt(S.Pi)/sqrt(a)/2 * exp(c - b**2/(4*a)) * \
                erfi((2*a*x + b)/(2*sqrt(a)))


@dataclass
class FresnelCRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return sqrt(S.Pi)/sqrt(2*a) * (
            cos(b**2/(4*a) - c)*fresnelc((2*a*x + b)/sqrt(2*a*S.Pi)) +
            sin(b**2/(4*a) - c)*fresnels((2*a*x + b)/sqrt(2*a*S.Pi)))


@dataclass
class FresnelSRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return sqrt(S.Pi)/sqrt(2*a) * (
            cos(b**2/(4*a) - c)*fresnels((2*a*x + b)/sqrt(2*a*S.Pi)) -
            sin(b**2/(4*a) - c)*fresnelc((2*a*x + b)/sqrt(2*a*S.Pi)))


@dataclass
class PolylogRule(AtomicRule):
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        return polylog(self.b + 1, self.a * self.variable)


@dataclass
class UpperGammaRule(AtomicRule):
    a: Expr
    e: Expr

    def eval(self) -> Expr:
        a, e, x = self.a, self.e, self.variable
        return x**e * (-a*x)**(-e) * uppergamma(e + 1, -a*x)/a


@dataclass
class EllipticFRule(AtomicRule):
    a: Expr
    d: Expr

    def eval(self) -> Expr:
        return elliptic_f(self.variable, self.d/self.a)/sqrt(self.a)


@dataclass
class EllipticERule(AtomicRule):
    a: Expr
    d: Expr

    def eval(self) -> Expr:
        return elliptic_e(self.variable, self.d/self.a)*sqrt(self.a)


class IntegralInfo(NamedTuple):
    integrand: Expr
    symbol: Symbol


def manual_diff(f, symbol):
    """Derivative of f in form expected by find_substitutions

    SymPy's derivatives for some trig functions (like cot) are not in a form
    that works well with finding substitutions; this replaces the
    derivatives for those particular forms with something that works better.

    """
    if f.args:
        arg = f.args[0]
        if isinstance(f, tan):
            return arg.diff(symbol) * sec(arg)**2
        elif isinstance(f, cot):
            return -arg.diff(symbol) * csc(arg)**2
        elif isinstance(f, sec):
            return arg.diff(symbol) * sec(arg) * tan(arg)
        elif isinstance(f, csc):
            return -arg.diff(symbol) * csc(arg) * cot(arg)
        elif isinstance(f, Add):
            return sum(manual_diff(arg, symbol) for arg in f.args)
        elif isinstance(f, Mul):
            if len(f.args) == 2 and isinstance(f.args[0], Number):
                return f.args[0] * manual_diff(f.args[1], symbol)
    return f.diff(symbol)

def manual_subs(expr, *args):
    """
    A wrapper for `expr.subs(*args)` with additional logic for substitution
    of invertible functions.
    """
    if len(args) == 1:
        sequence = args[0]
        if isinstance(sequence, (Dict, Mapping)):
            sequence = sequence.items()
        elif not iterable(sequence):
            raise ValueError("Expected an iterable of (old, new) pairs")
    elif len(args) == 2:
        sequence = [args]
    else:
        raise ValueError("subs accepts either 1 or 2 arguments")

    new_subs = []
    for old, new in sequence:
        if isinstance(old, log):
            # If log(x) = y, then exp(a*log(x)) = exp(a*y)
            # that is, x**a = exp(a*y). Replace nontrivial powers of x
            # before subs turns them into `exp(y)**a`, but
            # do not replace x itself yet, to avoid `log(exp(y))`.
            x0 = old.args[0]
            expr = expr.replace(lambda x: x.is_Pow and x.base == x0,
                lambda x: exp(x.exp*new))
            new_subs.append((x0, exp(new)))

    return expr.subs(list(sequence) + new_subs)

# Method based on that on SIN, described in "Symbolic Integration: The
# Stormy Decade"

inverse_trig_functions = (atan, asin, acos, acot, acsc, asec)


def find_substitutions(integrand, symbol, u_var):
    results = []

    def test_subterm(u, u_diff):
        if u_diff == 0:
            return False
        substituted = integrand / u_diff
        debug("substituted: {}, u: {}, u_var: {}".format(substituted, u, u_var))
        substituted = manual_subs(substituted, u, u_var).cancel()

        if substituted.has_free(symbol):
            return False
        # avoid increasing the degree of a rational function
        if integrand.is_rational_function(symbol) and substituted.is_rational_function(u_var):
            deg_before = max(degree(t, symbol) for t in integrand.as_numer_denom())
            deg_after = max(degree(t, u_var) for t in substituted.as_numer_denom())
            if deg_after > deg_before:
                return False
        return substituted.as_independent(u_var, as_Add=False)

    def exp_subterms(term: Expr):
        linear_coeffs = []
        terms = []
        n = Wild('n', properties=[lambda n: n.is_Integer])
        for exp_ in term.find(exp):
            arg = exp_.args[0]
            if symbol not in arg.free_symbols:
                continue
            match = arg.match(n*symbol)
            if match:
                linear_coeffs.append(match[n])
            else:
                terms.append(exp_)
        if linear_coeffs:
            terms.append(exp(gcd_list(linear_coeffs)*symbol))
        return terms

    def possible_subterms(term):
        if isinstance(term, (TrigonometricFunction, HyperbolicFunction,
                             *inverse_trig_functions,
                             exp, log, Heaviside)):
            return [term.args[0]]
        elif isinstance(term, (chebyshevt, chebyshevu,
                        legendre, hermite, laguerre)):
            return [term.args[1]]
        elif isinstance(term, (gegenbauer, assoc_laguerre)):
            return [term.args[2]]
        elif isinstance(term, jacobi):
            return [term.args[3]]
        elif isinstance(term, Mul):
            r = []
            for u in term.args:
                r.append(u)
                r.extend(possible_subterms(u))
            return r
        elif isinstance(term, Pow):
            r = [arg for arg in term.args if arg.has(symbol)]
            if term.exp.is_Integer:
                r.extend([term.base**d for d in primefactors(term.exp)
                    if 1 < d < abs(term.args[1])])
                if term.base.is_Add:
                    r.extend([t for t in possible_subterms(term.base)
                        if t.is_Pow])
            return r
        elif isinstance(term, Add):
            r = []
            for arg in term.args:
                r.append(arg)
                r.extend(possible_subterms(arg))
            return r
        return []

    for u in list(dict.fromkeys(possible_subterms(integrand) + exp_subterms(integrand))):
        if u == symbol:
            continue
        u_diff = manual_diff(u, symbol)
        new_integrand = test_subterm(u, u_diff)
        if new_integrand is not False:
            constant, new_integrand = new_integrand
            if new_integrand == integrand.subs(symbol, u_var):
                continue
            substitution = (u, constant, new_integrand)
            if substitution not in results:
                results.append(substitution)

    return results

def rewriter(condition, rewrite):
    """Strategy that rewrites an integrand."""
    def _rewriter(integral):
        integrand, symbol = integral
        debug("Integral: {} is rewritten with {} on symbol: {}".format(integrand, rewrite, symbol))
        if condition(*integral):
            rewritten = rewrite(*integral)
            if rewritten != integrand:
                substep = integral_steps(rewritten, symbol)
                if not isinstance(substep, DontKnowRule) and substep:
                    return RewriteRule(integrand, symbol, rewritten, substep)
    return _rewriter

def proxy_rewriter(condition, rewrite):
    """Strategy that rewrites an integrand based on some other criteria."""
    def _proxy_rewriter(criteria):
        criteria, integral = criteria
        integrand, symbol = integral
        debug("Integral: {} is rewritten with {} on symbol: {} and criteria: {}".format(integrand, rewrite, symbol, criteria))
        args = criteria + list(integral)
        if condition(*args):
            rewritten = rewrite(*args)
            if rewritten != integrand:
                return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))
    return _proxy_rewriter

def multiplexer(conditions):
    """Apply the rule that matches the condition, else None"""
    def multiplexer_rl(expr):
        for key, rule in conditions.items():
            if key(expr):
                return rule(expr)
    return multiplexer_rl

def alternatives(*rules):
    """Strategy that makes an AlternativeRule out of multiple possible results."""
    def _alternatives(integral):
        alts = []
        count = 0
        debug("List of Alternative Rules")
        for rule in rules:
            count = count + 1
            debug("Rule {}: {}".format(count, rule))

            result = rule(integral)
            if (result and not isinstance(result, DontKnowRule) and
                result != integral and result not in alts):
                alts.append(result)
        if len(alts) == 1:
            return alts[0]
        elif alts:
            doable = [rule for rule in alts if not rule.contains_dont_know()]
            if doable:
                return AlternativeRule(*integral, doable)
            else:
                return AlternativeRule(*integral, alts)
    return _alternatives

def constant_rule(integral):
    return ConstantRule(*integral)

def power_rule(integral):
    integrand, symbol = integral
    base, expt = integrand.as_base_exp()

    if symbol not in expt.free_symbols and isinstance(base, Symbol):
        if simplify(expt + 1) == 0:
            return ReciprocalRule(integrand, symbol, base)
        return PowerRule(integrand, symbol, base, expt)
    elif symbol not in base.free_symbols and isinstance(expt, Symbol):
        rule = ExpRule(integrand, symbol, base, expt)

        if fuzzy_not(log(base).is_zero):
            return rule
        elif log(base).is_zero:
            return ConstantRule(1, symbol)

        return PiecewiseRule(integrand, symbol, [
            (rule, Ne(log(base), 0)),
            (ConstantRule(1, symbol), True)
        ])

def exp_rule(integral):
    integrand, symbol = integral
    if isinstance(integrand.args[0], Symbol):
        return ExpRule(integrand, symbol, E, integrand.args[0])


def orthogonal_poly_rule(integral):
    orthogonal_poly_classes = {
        jacobi: JacobiRule,
        gegenbauer: GegenbauerRule,
        chebyshevt: ChebyshevTRule,
        chebyshevu: ChebyshevURule,
        legendre: LegendreRule,
        hermite: HermiteRule,
        laguerre: LaguerreRule,
        assoc_laguerre: AssocLaguerreRule
        }
    orthogonal_poly_var_index = {
        jacobi: 3,
        gegenbauer: 2,
        assoc_laguerre: 2
        }
    integrand, symbol = integral
    for klass in orthogonal_poly_classes:
        if isinstance(integrand, klass):
            var_index = orthogonal_poly_var_index.get(klass, 1)
            if (integrand.args[var_index] is symbol and not
                any(v.has(symbol) for v in integrand.args[:var_index])):
                    return orthogonal_poly_classes[klass](integrand, symbol, *integrand.args[:var_index])


_special_function_patterns: list[tuple[Type, Expr, Callable | None, tuple]] = []
_wilds = []
_symbol = Dummy('x')


def special_function_rule(integral):
    integrand, symbol = integral
    if not _special_function_patterns:
        a = Wild('a', exclude=[_symbol], properties=[lambda x: not x.is_zero])
        b = Wild('b', exclude=[_symbol])
        c = Wild('c', exclude=[_symbol])
        d = Wild('d', exclude=[_symbol], properties=[lambda x: not x.is_zero])
        e = Wild('e', exclude=[_symbol], properties=[
            lambda x: not (x.is_nonnegative and x.is_integer)])
        _wilds.extend((a, b, c, d, e))
        # patterns consist of a SymPy class, a wildcard expr, an optional
        # condition coded as a lambda (when Wild properties are not enough),
        # followed by an applicable rule
        linear_pattern = a*_symbol + b
        quadratic_pattern = a*_symbol**2 + b*_symbol + c
        _special_function_patterns.extend((
            (Mul, exp(linear_pattern, evaluate=False)/_symbol, None, EiRule),
            (Mul, cos(linear_pattern, evaluate=False)/_symbol, None, CiRule),
            (Mul, cosh(linear_pattern, evaluate=False)/_symbol, None, ChiRule),
            (Mul, sin(linear_pattern, evaluate=False)/_symbol, None, SiRule),
            (Mul, sinh(linear_pattern, evaluate=False)/_symbol, None, ShiRule),
            (Pow, 1/log(linear_pattern, evaluate=False), None, LiRule),
            (exp, exp(quadratic_pattern, evaluate=False), None, ErfRule),
            (sin, sin(quadratic_pattern, evaluate=False), None, FresnelSRule),
            (cos, cos(quadratic_pattern, evaluate=False), None, FresnelCRule),
            (Mul, _symbol**e*exp(a*_symbol, evaluate=False), None, UpperGammaRule),
            (Mul, polylog(b, a*_symbol, evaluate=False)/_symbol, None, PolylogRule),
            (Pow, 1/sqrt(a - d*sin(_symbol, evaluate=False)**2),
                lambda a, d: a != d, EllipticFRule),
            (Pow, sqrt(a - d*sin(_symbol, evaluate=False)**2),
                lambda a, d: a != d, EllipticERule),
        ))
    _integrand = integrand.subs(symbol, _symbol)
    for type_, pattern, constraint, rule in _special_function_patterns:
        if isinstance(_integrand, type_):
            match = _integrand.match(pattern)
            if match:
                wild_vals = tuple(match.get(w) for w in _wilds
                                  if match.get(w) is not None)
                if constraint is None or constraint(*wild_vals):
                    return rule(integrand, symbol, *wild_vals)


def _add_degenerate_step(generic_cond, generic_step: Rule, degenerate_step: Rule | None) -> Rule:
    if degenerate_step is None:
        return generic_step
    if isinstance(generic_step, PiecewiseRule):
        subfunctions = [(substep, (cond & generic_cond).simplify())
                        for substep, cond in generic_step.subfunctions]
    else:
        subfunctions = [(generic_step, generic_cond)]
    if isinstance(degenerate_step, PiecewiseRule):
        subfunctions += degenerate_step.subfunctions
    else:
        subfunctions.append((degenerate_step, S.true))
    return PiecewiseRule(generic_step.integrand, generic_step.variable, subfunctions)


def nested_pow_rule(integral: IntegralInfo):
    # nested (c*(a+b*x)**d)**e
    integrand, x = integral

    a_ = Wild('a', exclude=[x])
    b_ = Wild('b', exclude=[x, 0])
    pattern = a_+b_*x
    generic_cond = S.true

    class NoMatch(Exception):
        pass

    def _get_base_exp(expr: Expr) -> tuple[Expr, Expr]:
        if not expr.has_free(x):
            return S.One, S.Zero
        if expr.is_Mul:
            _, terms = expr.as_coeff_mul()
            if not terms:
                return S.One, S.Zero
            results = [_get_base_exp(term) for term in terms]
            bases = {b for b, _ in results}
            bases.discard(S.One)
            if len(bases) == 1:
                return bases.pop(), Add(*(e for _, e in results))
            raise NoMatch
        if expr.is_Pow:
            b, e = expr.base, expr.exp  # type: ignore
            if e.has_free(x):
                raise NoMatch
            base_, sub_exp = _get_base_exp(b)
            return base_, sub_exp * e
        match = expr.match(pattern)
        if match:
            a, b = match[a_], match[b_]
            base_ = x + a/b
            nonlocal generic_cond
            generic_cond = Ne(b, 0)
            return base_, S.One
        raise NoMatch

    try:
        base, exp_ = _get_base_exp(integrand)
    except NoMatch:
        return
    if generic_cond is S.true:
        degenerate_step = None
    else:
        # equivalent with subs(b, 0) but no need to find b
        degenerate_step = ConstantRule(integrand.subs(x, 0), x)
    generic_step = NestedPowRule(integrand, x, base, exp_)
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


def inverse_trig_rule(integral: IntegralInfo, degenerate=True):
    """
    Set degenerate=False on recursive call where coefficient of quadratic term
    is assumed non-zero.
    """
    integrand, symbol = integral
    base, exp = integrand.as_base_exp()
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    c = Wild('c', exclude=[symbol, 0])
    match = base.match(a + b*symbol + c*symbol**2)

    if not match:
        return

    def make_inverse_trig(RuleClass, a, sign_a, c, sign_c, h) -> Rule:
        u_var = Dummy("u")
        rewritten = 1/sqrt(sign_a*a + sign_c*c*(symbol-h)**2)  # a>0, c>0
        quadratic_base = sqrt(c/a)*(symbol-h)
        constant = 1/sqrt(c)
        u_func = None
        if quadratic_base is not symbol:
            u_func = quadratic_base
            quadratic_base = u_var
        standard_form = 1/sqrt(sign_a + sign_c*quadratic_base**2)
        substep = RuleClass(standard_form, quadratic_base)
        if constant != 1:
            substep = ConstantTimesRule(constant*standard_form, symbol, constant, standard_form, substep)
        if u_func is not None:
            substep = URule(rewritten, symbol, u_var, u_func, substep)
        if h != 0:
            substep = CompleteSquareRule(integrand, symbol, rewritten, substep)
        return substep

    a, b, c = [match.get(i, S.Zero) for i in (a, b, c)]
    generic_cond = Ne(c, 0)
    if not degenerate or generic_cond is S.true:
        degenerate_step = None
    elif b.is_zero:
        degenerate_step = ConstantRule(a ** exp, symbol)
    else:
        degenerate_step = sqrt_linear_rule(IntegralInfo((a + b * symbol) ** exp, symbol))

    if simplify(2*exp + 1) == 0:
        h, k = -b/(2*c), a - b**2/(4*c)  # rewrite base to k + c*(symbol-h)**2
        non_square_cond = Ne(k, 0)
        square_step = None
        if non_square_cond is not S.true:
            square_step = NestedPowRule(1/sqrt(c*(symbol-h)**2), symbol, symbol-h, S.NegativeOne)
        if non_square_cond is S.false:
            return square_step
        generic_step = ReciprocalSqrtQuadraticRule(integrand, symbol, a, b, c)
        step = _add_degenerate_step(non_square_cond, generic_step, square_step)
        if k.is_real and c.is_real:
            # list of ((rule, base_exp, a, sign_a, b, sign_b), condition)
            rules = []
            for args, cond in (  # don't apply ArccoshRule to x**2-1
                ((ArcsinRule, k, 1, -c, -1, h), And(k > 0, c < 0)),  # 1-x**2
                ((ArcsinhRule, k, 1, c, 1, h), And(k > 0, c > 0)),  # 1+x**2
            ):
                if cond is S.true:
                    return make_inverse_trig(*args)
                if cond is not S.false:
                    rules.append((make_inverse_trig(*args), cond))
            if rules:
                if not k.is_positive:  # conditions are not thorough, need fall back rule
                    rules.append((generic_step, S.true))
                step = PiecewiseRule(integrand, symbol, rules)
            else:
                step = generic_step
        return _add_degenerate_step(generic_cond, step, degenerate_step)
    if exp == S.Half:
        step = SqrtQuadraticRule(integrand, symbol, a, b, c)
        return _add_degenerate_step(generic_cond, step, degenerate_step)


def add_rule(integral):
    integrand, symbol = integral
    results = [integral_steps(g, symbol)
              for g in integrand.as_ordered_terms()]
    return None if None in results else AddRule(integrand, symbol, results)


def mul_rule(integral: IntegralInfo):
    integrand, symbol = integral

    # Constant times function case
    coeff, f = integrand.as_independent(symbol)
    if coeff != 1:
        next_step = integral_steps(f, symbol)
        if next_step is not None:
            return ConstantTimesRule(integrand, symbol, coeff, f, next_step)


def _parts_rule(integrand, symbol) -> tuple[Expr, Expr, Expr, Expr, Rule] | None:
    # LIATE rule:
    # log, inverse trig, algebraic, trigonometric, exponential
    def pull_out_algebraic(integrand):
        integrand = integrand.cancel().together()
        # iterating over Piecewise args would not work here
        algebraic = ([] if isinstance(integrand, Piecewise) or not integrand.is_Mul
            else [arg for arg in integrand.args if arg.is_algebraic_expr(symbol)])
        if algebraic:
            u = Mul(*algebraic)
            dv = (integrand / u).cancel()
            return u, dv

    def pull_out_u(*functions) -> Callable[[Expr], tuple[Expr, Expr] | None]:
        def pull_out_u_rl(integrand: Expr) -> tuple[Expr, Expr] | None:
            if any(integrand.has(f) for f in functions):
                args = [arg for arg in integrand.args
                        if any(isinstance(arg, cls) for cls in functions)]
                if args:
                    u = Mul(*args) # type: ignore
                    dv = integrand / u
                    return u, dv
            return None

        return pull_out_u_rl

    liate_rules = [pull_out_u(log), pull_out_u(*inverse_trig_functions),
                   pull_out_algebraic, pull_out_u(sin, cos),
                   pull_out_u(exp)]


    dummy = Dummy("temporary")
    # we can integrate log(x) and atan(x) by setting dv = 1
    if isinstance(integrand, (log, *inverse_trig_functions)):
        integrand = dummy * integrand

    for index, rule in enumerate(liate_rules):
        result = rule(integrand)

        if result:
            u, dv = result

            # Don't pick u to be a constant if possible
            if symbol not in u.free_symbols and not u.has(dummy):
                return None

            u = u.subs(dummy, 1)
            dv = dv.subs(dummy, 1)

            # Don't pick a non-polynomial algebraic to be differentiated
            if rule == pull_out_algebraic and not u.is_polynomial(symbol):
                return None
            # Don't trade one logarithm for another
            if isinstance(u, log):
                rec_dv = 1/dv
                if (rec_dv.is_polynomial(symbol) and
                    degree(rec_dv, symbol) == 1):
                    return None

            # Can integrate a polynomial times OrthogonalPolynomial
            if rule == pull_out_algebraic:
                if dv.is_Derivative or dv.has(TrigonometricFunction) or \
                        isinstance(dv, OrthogonalPolynomial):
                    v_step = integral_steps(dv, symbol)
                    if v_step.contains_dont_know():
                        return None
                    else:
                        du = u.diff(symbol)
                        v = v_step.eval()
                        return u, dv, v, du, v_step

            # make sure dv is amenable to integration
            accept = False
            if index < 2:  # log and inverse trig are usually worth trying
                accept = True
            elif (rule == pull_out_algebraic and dv.args and
                all(isinstance(a, (sin, cos, exp))
                for a in dv.args)):
                    accept = True
            else:
                for lrule in liate_rules[index + 1:]:
                    r = lrule(integrand)
                    if r and r[0].subs(dummy, 1).equals(dv):
                        accept = True
                        break

            if accept:
                du = u.diff(symbol)
                v_step = integral_steps(simplify(dv), symbol)
                if not v_step.contains_dont_know():
                    v = v_step.eval()
                    return u, dv, v, du, v_step
    return None


def parts_rule(integral):
    integrand, symbol = integral
    constant, integrand = integrand.as_coeff_Mul()

    result = _parts_rule(integrand, symbol)

    steps = []
    if result:
        u, dv, v, du, v_step = result
        debug("u : {}, dv : {}, v : {}, du : {}, v_step: {}".format(u, dv, v, du, v_step))
        steps.append(result)

        if isinstance(v, Integral):
            return

        # Set a limit on the number of times u can be used
        if isinstance(u, (sin, cos, exp, sinh, cosh)):
            cachekey = u.xreplace({symbol: _cache_dummy})
            if _parts_u_cache[cachekey] > 2:
                return
            _parts_u_cache[cachekey] += 1

        # Try cyclic integration by parts a few times
        for _ in range(4):
            debug("Cyclic integration {} with v: {}, du: {}, integrand: {}".format(_, v, du, integrand))
            coefficient = ((v * du) / integrand).cancel()
            if coefficient == 1:
                break
            if symbol not in coefficient.free_symbols:
                rule = CyclicPartsRule(integrand, symbol,
                    [PartsRule(None, None, u, dv, v_step, None)
                     for (u, dv, v, du, v_step) in steps],
                    (-1) ** len(steps) * coefficient)
                if (constant != 1) and rule:
                    rule = ConstantTimesRule(constant * integrand, symbol, constant, integrand, rule)
                return rule

            # _parts_rule is sensitive to constants, factor it out
            next_constant, next_integrand = (v * du).as_coeff_Mul()
            result = _parts_rule(next_integrand, symbol)

            if result:
                u, dv, v, du, v_step = result
                u *= next_constant
                du *= next_constant
                steps.append((u, dv, v, du, v_step))
            else:
                break

    def make_second_step(steps, integrand):
        if steps:
            u, dv, v, du, v_step = steps[0]
            return PartsRule(integrand, symbol, u, dv, v_step, make_second_step(steps[1:], v * du))
        return integral_steps(integrand, symbol)

    if steps:
        u, dv, v, du, v_step = steps[0]
        rule = PartsRule(integrand, symbol, u, dv, v_step, make_second_step(steps[1:], v * du))
        if (constant != 1) and rule:
            rule = ConstantTimesRule(constant * integrand, symbol, constant, integrand, rule)
        return rule


def trig_rule(integral):
    integrand, symbol = integral
    if integrand == sin(symbol):
        return SinRule(integrand, symbol)
    if integrand == cos(symbol):
        return CosRule(integrand, symbol)
    if integrand == sec(symbol)**2:
        return Sec2Rule(integrand, symbol)
    if integrand == csc(symbol)**2:
        return Csc2Rule(integrand, symbol)

    if isinstance(integrand, tan):
        rewritten = sin(*integrand.args) / cos(*integrand.args)
    elif isinstance(integrand, cot):
        rewritten = cos(*integrand.args) / sin(*integrand.args)
    elif isinstance(integrand, sec):
        arg = integrand.args[0]
        rewritten = ((sec(arg)**2 + tan(arg) * sec(arg)) /
                     (sec(arg) + tan(arg)))
    elif isinstance(integrand, csc):
        arg = integrand.args[0]
        rewritten = ((csc(arg)**2 + cot(arg) * csc(arg)) /
                     (csc(arg) + cot(arg)))
    else:
        return

    return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))

def trig_product_rule(integral: IntegralInfo):
    integrand, symbol = integral
    if integrand == sec(symbol) * tan(symbol):
        return SecTanRule(integrand, symbol)
    if integrand == csc(symbol) * cot(symbol):
        return CscCotRule(integrand, symbol)


def quadratic_denom_rule(integral):
    integrand, symbol = integral
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    c = Wild('c', exclude=[symbol])

    match = integrand.match(a / (b * symbol ** 2 + c))

    if match:
        a, b, c = match[a], match[b], match[c]
        general_rule = ArctanRule(integrand, symbol, a, b, c)
        if b.is_extended_real and c.is_extended_real:
            positive_cond = c/b > 0
            if positive_cond is S.true:
                return general_rule
            coeff = a/(2*sqrt(-c)*sqrt(b))
            constant = sqrt(-c/b)
            r1 = 1/(symbol-constant)
            r2 = 1/(symbol+constant)
            log_steps = [ReciprocalRule(r1, symbol, symbol-constant),
                         ConstantTimesRule(-r2, symbol, -1, r2, ReciprocalRule(r2, symbol, symbol+constant))]
            rewritten = sub = r1 - r2
            negative_step = AddRule(sub, symbol, log_steps)
            if coeff != 1:
                rewritten = Mul(coeff, sub, evaluate=False)
                negative_step = ConstantTimesRule(rewritten, symbol, coeff, sub, negative_step)
            negative_step = RewriteRule(integrand, symbol, rewritten, negative_step)
            if positive_cond is S.false:
                return negative_step
            return PiecewiseRule(integrand, symbol, [(general_rule, positive_cond), (negative_step, S.true)])

        power = PowerRule(integrand, symbol, symbol, -2)
        if b != 1:
            power = ConstantTimesRule(integrand, symbol, 1/b, symbol**-2, power)

        return PiecewiseRule(integrand, symbol, [(general_rule, Ne(c, 0)), (power, True)])

    d = Wild('d', exclude=[symbol])
    match2 = integrand.match(a / (b * symbol ** 2 + c * symbol + d))
    if match2:
        b, c =  match2[b], match2[c]
        if b.is_zero:
            return
        u = Dummy('u')
        u_func = symbol + c/(2*b)
        integrand2 = integrand.subs(symbol, u - c / (2*b))
        next_step = integral_steps(integrand2, u)
        if next_step:
            return URule(integrand2, symbol, u, u_func, next_step)
        else:
            return
    e = Wild('e', exclude=[symbol])
    match3 = integrand.match((a* symbol + b) / (c * symbol ** 2 + d * symbol + e))
    if match3:
        a, b, c, d, e = match3[a], match3[b], match3[c], match3[d], match3[e]
        if c.is_zero:
            return
        denominator = c * symbol**2 + d * symbol + e
        const =  a/(2*c)
        numer1 =  (2*c*symbol+d)
        numer2 = - const*d + b
        u = Dummy('u')
        step1 = URule(integrand, symbol,
                      u, denominator, integral_steps(u**(-1), u))
        if const != 1:
            step1 = ConstantTimesRule(const*numer1/denominator, symbol,
                                      const, numer1/denominator, step1)
        if numer2.is_zero:
            return step1
        step2 = integral_steps(numer2/denominator, symbol)
        substeps = AddRule(integrand, symbol, [step1, step2])
        rewriten = const*numer1/denominator+numer2/denominator
        return RewriteRule(integrand, symbol, rewriten, substeps)

    return


def sqrt_linear_rule(integral: IntegralInfo):
    """
    Substitute common (a+b*x)**(1/n)
    """
    integrand, x = integral
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x, 0])
    a0 = b0 = 0
    bases, qs, bs = [], [], []
    for pow_ in integrand.find(Pow):  # collect all (a+b*x)**(p/q)
        base, exp_ = pow_.base, pow_.exp
        if exp_.is_Integer or x not in base.free_symbols:  # skip 1/x and sqrt(2)
            continue
        if not exp_.is_Rational:  # exclude x**pi
            return
        match = base.match(a+b*x)
        if not match:  # skip non-linear
            continue  # for sqrt(x+sqrt(x)), although base is non-linear, we can still substitute sqrt(x)
        a1, b1 = match[a], match[b]
        if a0*b1 != a1*b0 or not (b0/b1).is_nonnegative:  # cannot transform sqrt(x) to sqrt(x+1) or sqrt(-x)
            return
        if b0 == 0 or (b0/b1 > 1) is S.true:  # choose the latter of sqrt(2*x) and sqrt(x) as representative
            a0, b0 = a1, b1
        bases.append(base)
        bs.append(b1)
        qs.append(exp_.q)
    if b0 == 0:  # no such pattern found
        return
    q0: Integer = lcm_list(qs)
    u_x = (a0 + b0*x)**(1/q0)
    u = Dummy("u")
    substituted = integrand.subs({base**(S.One/q): (b/b0)**(S.One/q)*u**(q0/q)
                                  for base, b, q in zip(bases, bs, qs)}).subs(x, (u**q0-a0)/b0)
    substep = integral_steps(substituted*u**(q0-1)*q0/b0, u)
    if not substep.contains_dont_know():
        step: Rule = URule(integrand, x, u, u_x, substep)
        generic_cond = Ne(b0, 0)
        if generic_cond is not S.true:  # possible degenerate case
            simplified = integrand.subs(dict.fromkeys(bs, 0))
            degenerate_step = integral_steps(simplified, x)
            step = PiecewiseRule(integrand, x, [(step, generic_cond), (degenerate_step, S.true)])
        return step


def sqrt_quadratic_rule(integral: IntegralInfo, degenerate=True):
    integrand, x = integral
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    c = Wild('c', exclude=[x, 0])
    f = Wild('f')
    n = Wild('n', properties=[lambda n: n.is_Integer and n.is_odd])
    match = integrand.match(f*sqrt(a+b*x+c*x**2)**n)
    if not match:
        return
    a, b, c, f, n = match[a], match[b], match[c], match[f], match[n]
    f_poly = f.as_poly(x)
    if f_poly is None:
        return

    generic_cond = Ne(c, 0)
    if not degenerate or generic_cond is S.true:
        degenerate_step = None
    elif b.is_zero:
        degenerate_step = integral_steps(f*sqrt(a)**n, x)
    else:
        degenerate_step = sqrt_linear_rule(IntegralInfo(f*sqrt(a+b*x)**n, x))

    def sqrt_quadratic_denom_rule(numer_poly: Poly, integrand: Expr):
        denom = sqrt(a+b*x+c*x**2)
        deg = numer_poly.degree()
        if deg <= 1:
            # integrand == (d+e*x)/sqrt(a+b*x+c*x**2)
            e, d = numer_poly.all_coeffs() if deg == 1 else (S.Zero, numer_poly.as_expr())
            # rewrite numerator to A*(2*c*x+b) + B
            A = e/(2*c)
            B = d-A*b
            pre_substitute = (2*c*x+b)/denom
            constant_step: Rule | None = None
            linear_step: Rule | None = None
            if A != 0:
                u = Dummy("u")
                pow_rule = PowerRule(1/sqrt(u), u, u, -S.Half)
                linear_step = URule(pre_substitute, x, u, a+b*x+c*x**2, pow_rule)
                if A != 1:
                    linear_step = ConstantTimesRule(A*pre_substitute, x, A, pre_substitute, linear_step)
            if B != 0:
                constant_step = inverse_trig_rule(IntegralInfo(1/denom, x), degenerate=False)
                if B != 1:
                    constant_step = ConstantTimesRule(B/denom, x, B, 1/denom, constant_step)  # type: ignore
            if linear_step and constant_step:
                add = Add(A*pre_substitute, B/denom, evaluate=False)
                step: Rule | None = RewriteRule(integrand, x, add, AddRule(add, x, [linear_step, constant_step]))
            else:
                step = linear_step or constant_step
        else:
            coeffs = numer_poly.all_coeffs()
            step = SqrtQuadraticDenomRule(integrand, x, a, b, c, coeffs)
        return step

    if n > 0:  # rewrite poly * sqrt(s)**(2*k-1) to poly*s**k / sqrt(s)
        numer_poly = f_poly * (a+b*x+c*x**2)**((n+1)/2)
        rewritten = numer_poly.as_expr()/sqrt(a+b*x+c*x**2)
        substep = sqrt_quadratic_denom_rule(numer_poly, rewritten)
        generic_step = RewriteRule(integrand, x, rewritten, substep)
    elif n == -1:
        generic_step = sqrt_quadratic_denom_rule(f_poly, integrand)
    else:
        return  # todo: handle n < -1 case
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


def hyperbolic_rule(integral: tuple[Expr, Symbol]):
    integrand, symbol = integral
    if isinstance(integrand, HyperbolicFunction) and integrand.args[0] == symbol:
        if integrand.func == sinh:
            return SinhRule(integrand, symbol)
        if integrand.func == cosh:
            return CoshRule(integrand, symbol)
        u = Dummy('u')
        if integrand.func == tanh:
            rewritten = sinh(symbol)/cosh(symbol)
            return RewriteRule(integrand, symbol, rewritten,
                   URule(rewritten, symbol, u, cosh(symbol), ReciprocalRule(1/u, u, u)))
        if integrand.func == coth:
            rewritten = cosh(symbol)/sinh(symbol)
            return RewriteRule(integrand, symbol, rewritten,
                   URule(rewritten, symbol, u, sinh(symbol), ReciprocalRule(1/u, u, u)))
        else:
            rewritten = integrand.rewrite(tanh)
            if integrand.func == sech:
                return RewriteRule(integrand, symbol, rewritten,
                       URule(rewritten, symbol, u, tanh(symbol/2),
                       ArctanRule(2/(u**2 + 1), u, S(2), S.One, S.One)))
            if integrand.func == csch:
                return RewriteRule(integrand, symbol, rewritten,
                       URule(rewritten, symbol, u, tanh(symbol/2),
                       ReciprocalRule(1/u, u, u)))

@cacheit
def make_wilds(symbol):
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    m = Wild('m', exclude=[symbol], properties=[lambda n: isinstance(n, Integer)])
    n = Wild('n', exclude=[symbol], properties=[lambda n: isinstance(n, Integer)])

    return a, b, m, n

@cacheit
def sincos_pattern(symbol):
    a, b, m, n = make_wilds(symbol)
    pattern = sin(a*symbol)**m * cos(b*symbol)**n

    return pattern, a, b, m, n

@cacheit
def tansec_pattern(symbol):
    a, b, m, n = make_wilds(symbol)
    pattern = tan(a*symbol)**m * sec(b*symbol)**n

    return pattern, a, b, m, n

@cacheit
def cotcsc_pattern(symbol):
    a, b, m, n = make_wilds(symbol)
    pattern = cot(a*symbol)**m * csc(b*symbol)**n

    return pattern, a, b, m, n

@cacheit
def heaviside_pattern(symbol):
    m = Wild('m', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    g = Wild('g')
    pattern = Heaviside(m*symbol + b) * g

    return pattern, m, b, g

def uncurry(func):
    def uncurry_rl(args):
        return func(*args)
    return uncurry_rl

def trig_rewriter(rewrite):
    def trig_rewriter_rl(args):
        a, b, m, n, integrand, symbol = args
        rewritten = rewrite(a, b, m, n, integrand, symbol)
        if rewritten != integrand:
            return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))
    return trig_rewriter_rl

sincos_botheven_condition = uncurry(
    lambda a, b, m, n, i, s: m.is_even and n.is_even and
    m.is_nonnegative and n.is_nonnegative)

sincos_botheven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (((1 - cos(2*a*symbol)) / 2) ** (m / 2)) *
                                    (((1 + cos(2*b*symbol)) / 2) ** (n / 2)) ))

sincos_sinodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd and m >= 3)

sincos_sinodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 - cos(a*symbol)**2)**((m - 1) / 2) *
                                    sin(a*symbol) *
                                    cos(b*symbol) ** n))

sincos_cosodd_condition = uncurry(lambda a, b, m, n, i, s: n.is_odd and n >= 3)

sincos_cosodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 - sin(b*symbol)**2)**((n - 1) / 2) *
                                    cos(b*symbol) *
                                    sin(a*symbol) ** m))

tansec_seceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)
tansec_seceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + tan(b*symbol)**2) ** (n/2 - 1) *
                                    sec(b*symbol)**2 *
                                    tan(a*symbol) ** m ))

tansec_tanodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)
tansec_tanodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (sec(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                     tan(a*symbol) *
                                     sec(b*symbol) ** n ))

tan_tansquared_condition = uncurry(lambda a, b, m, n, i, s: m == 2 and n == 0)
tan_tansquared = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( sec(a*symbol)**2 - 1))

cotcsc_csceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)
cotcsc_csceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + cot(b*symbol)**2) ** (n/2 - 1) *
                                    csc(b*symbol)**2 *
                                    cot(a*symbol) ** m ))

cotcsc_cotodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)
cotcsc_cotodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (csc(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                    cot(a*symbol) *
                                    csc(b*symbol) ** n ))

def trig_sincos_rule(integral):
    integrand, symbol = integral

    if any(integrand.has(f) for f in (sin, cos)):
        pattern, a, b, m, n = sincos_pattern(symbol)
        match = integrand.match(pattern)
        if not match:
            return

        return multiplexer({
            sincos_botheven_condition: sincos_botheven,
            sincos_sinodd_condition: sincos_sinodd,
            sincos_cosodd_condition: sincos_cosodd
        })(tuple(
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            [integrand, symbol]))

def trig_tansec_rule(integral):
    integrand, symbol = integral

    integrand = integrand.subs({
        1 / cos(symbol): sec(symbol)
    })

    if any(integrand.has(f) for f in (tan, sec)):
        pattern, a, b, m, n = tansec_pattern(symbol)
        match = integrand.match(pattern)
        if not match:
            return

        return multiplexer({
            tansec_tanodd_condition: tansec_tanodd,
            tansec_seceven_condition: tansec_seceven,
            tan_tansquared_condition: tan_tansquared
        })(tuple(
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            [integrand, symbol]))

def trig_cotcsc_rule(integral):
    integrand, symbol = integral
    integrand = integrand.subs({
        1 / sin(symbol): csc(symbol),
        1 / tan(symbol): cot(symbol),
        cos(symbol) / tan(symbol): cot(symbol)
    })

    if any(integrand.has(f) for f in (cot, csc)):
        pattern, a, b, m, n = cotcsc_pattern(symbol)
        match = integrand.match(pattern)
        if not match:
            return

        return multiplexer({
            cotcsc_cotodd_condition: cotcsc_cotodd,
            cotcsc_csceven_condition: cotcsc_csceven
        })(tuple(
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            [integrand, symbol]))

def trig_sindouble_rule(integral):
    integrand, symbol = integral
    a = Wild('a', exclude=[sin(2*symbol)])
    match = integrand.match(sin(2*symbol)*a)
    if match:
        sin_double = 2*sin(symbol)*cos(symbol)/sin(2*symbol)
        return integral_steps(integrand * sin_double, symbol)

def trig_powers_products_rule(integral):
    return do_one(null_safe(trig_sincos_rule),
                  null_safe(trig_tansec_rule),
                  null_safe(trig_cotcsc_rule),
                  null_safe(trig_sindouble_rule))(integral)

def trig_substitution_rule(integral):
    integrand, symbol = integral
    A = Wild('a', exclude=[0, symbol])
    B = Wild('b', exclude=[0, symbol])
    theta = Dummy("theta")
    target_pattern = A + B*symbol**2

    matches = integrand.find(target_pattern)
    for expr in matches:
        match = expr.match(target_pattern)
        a = match.get(A, S.Zero)
        b = match.get(B, S.Zero)

        a_positive = ((a.is_number and a > 0) or a.is_positive)
        b_positive = ((b.is_number and b > 0) or b.is_positive)
        a_negative = ((a.is_number and a < 0) or a.is_negative)
        b_negative = ((b.is_number and b < 0) or b.is_negative)
        x_func = None
        if a_positive and b_positive:
            # a**2 + b*x**2. Assume sec(theta) > 0, -pi/2 < theta < pi/2
            x_func = (sqrt(a)/sqrt(b)) * tan(theta)
            # Do not restrict the domain: tan(theta) takes on any real
            # value on the interval -pi/2 < theta < pi/2 so x takes on
            # any value
            restriction = True
        elif a_positive and b_negative:
            # a**2 - b*x**2. Assume cos(theta) > 0, -pi/2 < theta < pi/2
            constant = sqrt(a)/sqrt(-b)
            x_func = constant * sin(theta)
            restriction = And(symbol > -constant, symbol < constant)
        elif a_negative and b_positive:
            # b*x**2 - a**2. Assume sin(theta) > 0, 0 < theta < pi
            constant = sqrt(-a)/sqrt(b)
            x_func = constant * sec(theta)
            restriction = And(symbol > -constant, symbol < constant)
        if x_func:
            # Manually simplify sqrt(trig(theta)**2) to trig(theta)
            # Valid due to assumed domain restriction
            substitutions = {}
            for f in [sin, cos, tan,
                      sec, csc, cot]:
                substitutions[sqrt(f(theta)**2)] = f(theta)
                substitutions[sqrt(f(theta)**(-2))] = 1/f(theta)

            replaced = integrand.subs(symbol, x_func).trigsimp()
            replaced = manual_subs(replaced, substitutions)
            if not replaced.has(symbol):
                replaced *= manual_diff(x_func, theta)
                replaced = replaced.trigsimp()
                secants = replaced.find(1/cos(theta))
                if secants:
                    replaced = replaced.xreplace({
                        1/cos(theta): sec(theta)
                    })

                substep = integral_steps(replaced, theta)
                if not substep.contains_dont_know():
                    return TrigSubstitutionRule(integrand, symbol,
                        theta, x_func, replaced, substep, restriction)

def heaviside_rule(integral):
    integrand, symbol = integral
    pattern, m, b, g = heaviside_pattern(symbol)
    match = integrand.match(pattern)
    if match and 0 != match[g]:
        # f = Heaviside(m*x + b)*g
        substep = integral_steps(match[g], symbol)
        m, b = match[m], match[b]
        return HeavisideRule(integrand, symbol, m*symbol + b, -b/m, substep)


def dirac_delta_rule(integral: IntegralInfo):
    integrand, x = integral
    if len(integrand.args) == 1:
        n = S.Zero
    else:
        n = integrand.args[1] # type: ignore
    if not n.is_Integer or n < 0:
        return
    a, b = Wild('a', exclude=[x]), Wild('b', exclude=[x, 0])
    match = integrand.args[0].match(a+b*x)
    if not match:
        return
    a, b = match[a], match[b]
    generic_cond = Ne(b, 0)
    if generic_cond is S.true:
        degenerate_step = None
    else:
        degenerate_step = ConstantRule(DiracDelta(a, n), x)
    generic_step = DiracDeltaRule(integrand, x, n, a, b)
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


def substitution_rule(integral):
    integrand, symbol = integral

    u_var = Dummy("u")
    substitutions = find_substitutions(integrand, symbol, u_var)
    count = 0
    if substitutions:
        debug("List of Substitution Rules")
        ways = []
        for u_func, c, substituted in substitutions:
            subrule = integral_steps(substituted, u_var)
            count = count + 1
            debug("Rule {}: {}".format(count, subrule))

            if subrule.contains_dont_know():
                continue

            if simplify(c - 1) != 0:
                _, denom = c.as_numer_denom()
                if subrule:
                    subrule = ConstantTimesRule(c * substituted, u_var, c, substituted, subrule)

                if denom.free_symbols:
                    piecewise = []
                    could_be_zero = []

                    if isinstance(denom, Mul):
                        could_be_zero = denom.args
                    else:
                        could_be_zero.append(denom)

                    for expr in could_be_zero:
                        if not fuzzy_not(expr.is_zero):
                            substep = integral_steps(manual_subs(integrand, expr, 0), symbol)

                            if substep:
                                piecewise.append((
                                    substep,
                                    Eq(expr, 0)
                                ))
                    piecewise.append((subrule, True))
                    subrule = PiecewiseRule(substituted, symbol, piecewise)

            ways.append(URule(integrand, symbol, u_var, u_func, subrule))

        if len(ways) > 1:
            return AlternativeRule(integrand, symbol, ways)
        elif ways:
            return ways[0]


partial_fractions_rule = rewriter(
    lambda integrand, symbol: integrand.is_rational_function(),
    lambda integrand, symbol: integrand.apart(symbol))

cancel_rule = rewriter(
    # lambda integrand, symbol: integrand.is_algebraic_expr(),
    # lambda integrand, symbol: isinstance(integrand, Mul),
    lambda integrand, symbol: True,
    lambda integrand, symbol: integrand.cancel())

distribute_expand_rule = rewriter(
    lambda integrand, symbol: (
        isinstance(integrand, (Pow, Mul)) or all(arg.is_Pow or arg.is_polynomial(symbol) for arg in integrand.args)),
    lambda integrand, symbol: integrand.expand())

trig_expand_rule = rewriter(
    # If there are trig functions with different arguments, expand them
    lambda integrand, symbol: (
        len({a.args[0] for a in integrand.atoms(TrigonometricFunction)}) > 1),
    lambda integrand, symbol: integrand.expand(trig=True))

def derivative_rule(integral):
    integrand = integral[0]
    diff_variables = integrand.variables
    undifferentiated_function = integrand.expr
    integrand_variables = undifferentiated_function.free_symbols

    if integral.symbol in integrand_variables:
        if integral.symbol in diff_variables:
            return DerivativeRule(*integral)
        else:
            return DontKnowRule(integrand, integral.symbol)
    else:
        return ConstantRule(*integral)

def rewrites_rule(integral):
    integrand, symbol = integral

    if integrand.match(1/cos(symbol)):
        rewritten = integrand.subs(1/cos(symbol), sec(symbol))
        return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))

def fallback_rule(integral):
    return DontKnowRule(*integral)

# Cache is used to break cyclic integrals.
# Need to use the same dummy variable in cached expressions for them to match.
# Also record "u" of integration by parts, to avoid infinite repetition.
_integral_cache: dict[Expr, Expr | None] = {}
_parts_u_cache: dict[Expr, int] = defaultdict(int)
_cache_dummy = Dummy("z")

def integral_steps(integrand, symbol, **options):
    """Returns the steps needed to compute an integral.

    Explanation
    ===========

    This function attempts to mirror what a student would do by hand as
    closely as possible.

    SymPy Gamma uses this to provide a step-by-step explanation of an
    integral. The code it uses to format the results of this function can be
    found at
    https://github.com/sympy/sympy_gamma/blob/master/app/logic/intsteps.py.

    Examples
    ========

    >>> from sympy import exp, sin
    >>> from sympy.integrals.manualintegrate import integral_steps
    >>> from sympy.abc import x
    >>> print(repr(integral_steps(exp(x) / (1 + exp(2 * x)), x))) \
    # doctest: +NORMALIZE_WHITESPACE
    URule(integrand=exp(x)/(exp(2*x) + 1), variable=x, u_var=_u, u_func=exp(x),
    substep=ArctanRule(integrand=1/(_u**2 + 1), variable=_u, a=1, b=1, c=1))
    >>> print(repr(integral_steps(sin(x), x))) \
    # doctest: +NORMALIZE_WHITESPACE
    SinRule(integrand=sin(x), variable=x)
    >>> print(repr(integral_steps((x**2 + 3)**2, x))) \
    # doctest: +NORMALIZE_WHITESPACE
    RewriteRule(integrand=(x**2 + 3)**2, variable=x, rewritten=x**4 + 6*x**2 + 9,
    substep=AddRule(integrand=x**4 + 6*x**2 + 9, variable=x,
    substeps=[PowerRule(integrand=x**4, variable=x, base=x, exp=4),
    ConstantTimesRule(integrand=6*x**2, variable=x, constant=6, other=x**2,
    substep=PowerRule(integrand=x**2, variable=x, base=x, exp=2)),
    ConstantRule(integrand=9, variable=x)]))

    Returns
    =======

    rule : Rule
        The first step; most rules have substeps that must also be
        considered. These substeps can be evaluated using ``manualintegrate``
        to obtain a result.

    """
    cachekey = integrand.xreplace({symbol: _cache_dummy})
    if cachekey in _integral_cache:
        if _integral_cache[cachekey] is None:
            # Stop this attempt, because it leads around in a loop
            return DontKnowRule(integrand, symbol)
        else:
            # TODO: This is for future development, as currently
            # _integral_cache gets no values other than None
            return (_integral_cache[cachekey].xreplace(_cache_dummy, symbol),
                symbol)
    else:
        _integral_cache[cachekey] = None

    integral = IntegralInfo(integrand, symbol)

    def key(integral):
        integrand = integral.integrand

        if symbol not in integrand.free_symbols:
            return Number
        for cls in (Symbol, TrigonometricFunction, OrthogonalPolynomial):
            if isinstance(integrand, cls):
                return cls
        return type(integrand)

    def integral_is_subclass(*klasses):
        def _integral_is_subclass(integral):
            k = key(integral)
            return k and issubclass(k, klasses)
        return _integral_is_subclass

    result = do_one(
        null_safe(special_function_rule),
        null_safe(switch(key, {
            Pow: do_one(null_safe(power_rule), null_safe(inverse_trig_rule),
                        null_safe(sqrt_linear_rule),
                        null_safe(quadratic_denom_rule)),
            Symbol: power_rule,
            exp: exp_rule,
            Add: add_rule,
            Mul: do_one(null_safe(mul_rule), null_safe(trig_product_rule),
                        null_safe(heaviside_rule), null_safe(quadratic_denom_rule),
                        null_safe(sqrt_linear_rule),
                        null_safe(sqrt_quadratic_rule)),
            Derivative: derivative_rule,
            TrigonometricFunction: trig_rule,
            Heaviside: heaviside_rule,
            DiracDelta: dirac_delta_rule,
            OrthogonalPolynomial: orthogonal_poly_rule,
            Number: constant_rule
        })),
        do_one(
            null_safe(trig_rule),
            null_safe(hyperbolic_rule),
            null_safe(alternatives(
                rewrites_rule,
                substitution_rule,
                condition(
                    integral_is_subclass(Mul, Pow),
                    partial_fractions_rule),
                condition(
                    integral_is_subclass(Mul, Pow),
                    cancel_rule),
                condition(
                    integral_is_subclass(Mul, log,
                    *inverse_trig_functions),
                    parts_rule),
                condition(
                    integral_is_subclass(Mul, Pow),
                    distribute_expand_rule),
                trig_powers_products_rule,
                trig_expand_rule
            )),
            null_safe(condition(integral_is_subclass(Mul, Pow), nested_pow_rule)),
            null_safe(trig_substitution_rule)
        ),
        fallback_rule)(integral)
    del _integral_cache[cachekey]
    return result


def manualintegrate(f, var):
    """manualintegrate(f, var)

    Explanation
    ===========

    Compute indefinite integral of a single variable using an algorithm that
    resembles what a student would do by hand.

    Unlike :func:`~.integrate`, var can only be a single symbol.

    Examples
    ========

    >>> from sympy import sin, cos, tan, exp, log, integrate
    >>> from sympy.integrals.manualintegrate import manualintegrate
    >>> from sympy.abc import x
    >>> manualintegrate(1 / x, x)
    log(x)
    >>> integrate(1/x)
    log(x)
    >>> manualintegrate(log(x), x)
    x*log(x) - x
    >>> integrate(log(x))
    x*log(x) - x
    >>> manualintegrate(exp(x) / (1 + exp(2 * x)), x)
    atan(exp(x))
    >>> integrate(exp(x) / (1 + exp(2 * x)))
    RootSum(4*_z**2 + 1, Lambda(_i, _i*log(2*_i + exp(x))))
    >>> manualintegrate(cos(x)**4 * sin(x), x)
    -cos(x)**5/5
    >>> integrate(cos(x)**4 * sin(x), x)
    -cos(x)**5/5
    >>> manualintegrate(cos(x)**4 * sin(x)**3, x)
    cos(x)**7/7 - cos(x)**5/5
    >>> integrate(cos(x)**4 * sin(x)**3, x)
    cos(x)**7/7 - cos(x)**5/5
    >>> manualintegrate(tan(x), x)
    -log(cos(x))
    >>> integrate(tan(x), x)
    -log(cos(x))

    See Also
    ========

    sympy.integrals.integrals.integrate
    sympy.integrals.integrals.Integral.doit
    sympy.integrals.integrals.Integral
    """
    result = integral_steps(f, var).eval()
    # Clear the cache of u-parts
    _parts_u_cache.clear()
    # If we got Piecewise with two parts, put generic first
    if isinstance(result, Piecewise) and len(result.args) == 2:
        cond = result.args[0][1]
        if isinstance(cond, Eq) and result.args[1][1] == True:
            result = result.func(
                (result.args[1][0], Ne(*cond.args)),
                (result.args[0][0], True))
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\libmp\gammazeta.py ===
"""
-----------------------------------------------------------------------
This module implements gamma- and zeta-related functions:

* Bernoulli numbers
* Factorials
* The gamma function
* Polygamma functions
* Harmonic numbers
* The Riemann zeta function
* Constants related to these functions

-----------------------------------------------------------------------
"""

import math
import sys

from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy

from .libintmath import list_primes, ifac, ifac2, moebius

from .libmpf import (\
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast,
    lshift, sqrt_fixed, isqrt_fast,
    fzero, fone, fnone, fhalf, ftwo, finf, fninf, fnan,
    from_int, to_int, to_fixed, from_man_exp, from_rational,
    mpf_pos, mpf_neg, mpf_abs, mpf_add, mpf_sub,
    mpf_mul, mpf_mul_int, mpf_div, mpf_sqrt, mpf_pow_int,
    mpf_rdiv_int,
    mpf_perturb, mpf_le, mpf_lt, mpf_gt, mpf_shift,
    negative_rnd, reciprocal_rnd,
    bitcount, to_float, mpf_floor, mpf_sign, ComplexResult
)

from .libelefun import (\
    constant_memo,
    def_mpf_constant,
    mpf_pi, pi_fixed, ln2_fixed, log_int_fixed, mpf_ln2,
    mpf_exp, mpf_log, mpf_pow, mpf_cosh,
    mpf_cos_sin, mpf_cosh_sinh, mpf_cos_sin_pi, mpf_cos_pi, mpf_sin_pi,
    ln_sqrt2pi_fixed, mpf_ln_sqrt2pi, sqrtpi_fixed, mpf_sqrtpi,
    cos_sin_fixed, exp_fixed
)

from .libmpc import (\
    mpc_zero, mpc_one, mpc_half, mpc_two,
    mpc_abs, mpc_shift, mpc_pos, mpc_neg,
    mpc_add, mpc_sub, mpc_mul, mpc_div,
    mpc_add_mpf, mpc_mul_mpf, mpc_div_mpf, mpc_mpf_div,
    mpc_mul_int, mpc_pow_int,
    mpc_log, mpc_exp, mpc_pow,
    mpc_cos_pi, mpc_sin_pi,
    mpc_reciprocal, mpc_square,
    mpc_sub_mpf
)



# Catalan's constant is computed using Lupas's rapidly convergent series
# (listed on http://mathworld.wolfram.com/CatalansConstant.html)
#            oo
#            ___       n-1  8n     2                   3    2
#        1  \      (-1)    2   (40n  - 24n + 3) [(2n)!] (n!)
#  K =  ---  )     -----------------------------------------
#       64  /___               3               2
#                             n  (2n-1) [(4n)!]
#           n = 1

@constant_memo
def catalan_fixed(prec):
    prec = prec + 20
    a = one = MPZ_ONE << prec
    s, t, n = 0, 1, 1
    while t:
        a *= 32 * n**3 * (2*n-1)
        a //= (3-16*n+16*n**2)**2
        t = a * (-1)**(n-1) * (40*n**2-24*n+3) // (n**3 * (2*n-1))
        s += t
        n += 1
    return s >> (20 + 6)

# Khinchin's constant is relatively difficult to compute. Here
# we use the rational zeta series

#                    oo                2*n-1
#                   ___                ___
#                   \   ` zeta(2*n)-1  \   ` (-1)^(k+1)
#  log(K)*log(2) =   )    ------------  )    ----------
#                   /___.      n       /___.      k
#                   n = 1              k = 1

# which adds half a digit per term. The essential trick for achieving
# reasonable efficiency is to recycle both the values of the zeta
# function (essentially Bernoulli numbers) and the partial terms of
# the inner sum.

# An alternative might be to use K = 2*exp[1/log(2) X] where

#      / 1     1       [ pi*x*(1-x^2) ]
#  X = |    ------ log [ ------------ ].
#      / 0  x(1+x)     [  sin(pi*x)   ]

# and integrate numerically. In practice, this seems to be slightly
# slower than the zeta series at high precision.

@constant_memo
def khinchin_fixed(prec):
    wp = int(prec + prec**0.5 + 15)
    s = MPZ_ZERO
    fac = from_int(4)
    t = ONE = MPZ_ONE << wp
    pi = mpf_pi(wp)
    pipow = twopi2 = mpf_shift(mpf_mul(pi, pi, wp), 2)
    n = 1
    while 1:
        zeta2n = mpf_abs(mpf_bernoulli(2*n, wp))
        zeta2n = mpf_mul(zeta2n, pipow, wp)
        zeta2n = mpf_div(zeta2n, fac, wp)
        zeta2n = to_fixed(zeta2n, wp)
        term = (((zeta2n - ONE) * t) // n) >> wp
        if term < 100:
            break
        #if not n % 10:
        #    print n, math.log(int(abs(term)))
        s += term
        t += ONE//(2*n+1) - ONE//(2*n)
        n += 1
        fac = mpf_mul_int(fac, (2*n)*(2*n-1), wp)
        pipow = mpf_mul(pipow, twopi2, wp)
    s = (s << wp) // ln2_fixed(wp)
    K = mpf_exp(from_man_exp(s, -wp), wp)
    K = to_fixed(K, prec)
    return K


# Glaisher's constant is defined as A = exp(1/2 - zeta'(-1)).
# One way to compute it would be to perform direct numerical
# differentiation, but computing arbitrary Riemann zeta function
# values at high precision is expensive. We instead use the formula

#     A = exp((6 (-zeta'(2))/pi^2 + log 2 pi + gamma)/12)

# and compute zeta'(2) from the series representation

#              oo
#              ___
#             \     log k
#  -zeta'(2) = )    -----
#             /___     2
#                    k
#            k = 2

# This series converges exceptionally slowly, but can be accelerated
# using Euler-Maclaurin formula. The important insight is that the
# E-M integral can be done in closed form and that the high order
# are given by

#    n  /       \
#   d   | log x |   a + b log x
#   --- | ----- | = -----------
#     n |   2   |      2 + n
#   dx  \  x    /     x

# where a and b are integers given by a simple recurrence. Note
# that just one logarithm is needed. However, lots of integer
# logarithms are required for the initial summation.

# This algorithm could possibly be turned into a faster algorithm
# for general evaluation of zeta(s) or zeta'(s); this should be
# looked into.

@constant_memo
def glaisher_fixed(prec):
    wp = prec + 30
    # Number of direct terms to sum before applying the Euler-Maclaurin
    # formula to the tail. TODO: choose more intelligently
    N = int(0.33*prec + 5)
    ONE = MPZ_ONE << wp
    # Euler-Maclaurin, step 1: sum log(k)/k**2 for k from 2 to N-1
    s = MPZ_ZERO
    for k in range(2, N):
        #print k, N
        s += log_int_fixed(k, wp) // k**2
    logN = log_int_fixed(N, wp)
    #logN = to_fixed(mpf_log(from_int(N), wp+20), wp)
    # E-M step 2: integral of log(x)/x**2 from N to inf
    s += (ONE + logN) // N
    # E-M step 3: endpoint correction term f(N)/2
    s += logN // (N**2 * 2)
    # E-M step 4: the series of derivatives
    pN = N**3
    a = 1
    b = -2
    j = 3
    fac = from_int(2)
    k = 1
    while 1:
        # D(2*k-1) * B(2*k) / fac(2*k) [D(n) = nth derivative]
        D = ((a << wp) + b*logN) // pN
        D = from_man_exp(D, -wp)
        B = mpf_bernoulli(2*k, wp)
        term = mpf_mul(B, D, wp)
        term = mpf_div(term, fac, wp)
        term = to_fixed(term, wp)
        if abs(term) < 100:
            break
        #if not k % 10:
        #    print k, math.log(int(abs(term)), 10)
        s -= term
        # Advance derivative twice
        a, b, pN, j = b-a*j, -j*b, pN*N, j+1
        a, b, pN, j = b-a*j, -j*b, pN*N, j+1
        k += 1
        fac = mpf_mul_int(fac, (2*k)*(2*k-1), wp)
    # A = exp((6*s/pi**2 + log(2*pi) + euler)/12)
    pi = pi_fixed(wp)
    s *= 6
    s = (s << wp) // (pi**2 >> wp)
    s += euler_fixed(wp)
    s += to_fixed(mpf_log(from_man_exp(2*pi, -wp), wp), wp)
    s //= 12
    A = mpf_exp(from_man_exp(s, -wp), wp)
    return to_fixed(A, prec)

# Apery's constant can be computed using the very rapidly convergent
# series
#              oo
#              ___              2                      10
#             \         n  205 n  + 250 n + 77     (n!)
#  zeta(3) =   )    (-1)   -------------------  ----------
#             /___               64                      5
#             n = 0                             ((2n+1)!)

@constant_memo
def apery_fixed(prec):
    prec += 20
    d = MPZ_ONE << prec
    term = MPZ(77) << prec
    n = 1
    s = MPZ_ZERO
    while term:
        s += term
        d *= (n**10)
        d //= (((2*n+1)**5) * (2*n)**5)
        term = (-1)**n * (205*(n**2) + 250*n + 77) * d
        n += 1
    return s >> (20 + 6)

"""
Euler's constant (gamma) is computed using the Brent-McMillan formula,
gamma ~= I(n)/J(n) - log(n), where

   I(n) = sum_{k=0,1,2,...} (n**k / k!)**2 * H(k)
   J(n) = sum_{k=0,1,2,...} (n**k / k!)**2
   H(k) = 1 + 1/2 + 1/3 + ... + 1/k

The error is bounded by O(exp(-4n)). Choosing n to be a power
of two, 2**p, the logarithm becomes particularly easy to calculate.[1]

We use the formulation of Algorithm 3.9 in [2] to make the summation
more efficient.

Reference:
[1] Xavier Gourdon & Pascal Sebah, The Euler constant: gamma
http://numbers.computation.free.fr/Constants/Gamma/gamma.pdf

[2] [BorweinBailey]_
"""

@constant_memo
def euler_fixed(prec):
    extra = 30
    prec += extra
    # choose p such that exp(-4*(2**p)) < 2**-n
    p = int(math.log((prec/4) * math.log(2), 2)) + 1
    n = 2**p
    A = U = -p*ln2_fixed(prec)
    B = V = MPZ_ONE << prec
    k = 1
    while 1:
        B = B*n**2//k**2
        A = (A*n**2//k + B)//k
        U += A
        V += B
        if max(abs(A), abs(B)) < 100:
            break
        k += 1
    return (U<<(prec-extra))//V

# Use zeta accelerated formulas for the Mertens and twin
# prime constants; see
# http://mathworld.wolfram.com/MertensConstant.html
# http://mathworld.wolfram.com/TwinPrimesConstant.html

@constant_memo
def mertens_fixed(prec):
    wp = prec + 20
    m = 2
    s = mpf_euler(wp)
    while 1:
        t = mpf_zeta_int(m, wp)
        if t == fone:
            break
        t = mpf_log(t, wp)
        t = mpf_mul_int(t, moebius(m), wp)
        t = mpf_div(t, from_int(m), wp)
        s = mpf_add(s, t)
        m += 1
    return to_fixed(s, prec)

@constant_memo
def twinprime_fixed(prec):
    def I(n):
        return sum(moebius(d)<<(n//d) for d in xrange(1,n+1) if not n%d)//n
    wp = 2*prec + 30
    res = fone
    primes = [from_rational(1,p,wp) for p in [2,3,5,7]]
    ppowers = [mpf_mul(p,p,wp) for p in primes]
    n = 2
    while 1:
        a = mpf_zeta_int(n, wp)
        for i in range(4):
            a = mpf_mul(a, mpf_sub(fone, ppowers[i]), wp)
            ppowers[i] = mpf_mul(ppowers[i], primes[i], wp)
        a = mpf_pow_int(a, -I(n), wp)
        if mpf_pos(a, prec+10, 'n') == fone:
            break
        #from libmpf import to_str
        #print n, to_str(mpf_sub(fone, a), 6)
        res = mpf_mul(res, a, wp)
        n += 1
    res = mpf_mul(res, from_int(3*15*35), wp)
    res = mpf_div(res, from_int(4*16*36), wp)
    return to_fixed(res, prec)


mpf_euler = def_mpf_constant(euler_fixed)
mpf_apery = def_mpf_constant(apery_fixed)
mpf_khinchin = def_mpf_constant(khinchin_fixed)
mpf_glaisher = def_mpf_constant(glaisher_fixed)
mpf_catalan = def_mpf_constant(catalan_fixed)
mpf_mertens = def_mpf_constant(mertens_fixed)
mpf_twinprime = def_mpf_constant(twinprime_fixed)


#-----------------------------------------------------------------------#
#                                                                       #
#                          Bernoulli numbers                            #
#                                                                       #
#-----------------------------------------------------------------------#

MAX_BERNOULLI_CACHE = 3000


r"""
Small Bernoulli numbers and factorials are used in numerous summations,
so it is critical for speed that sequential computation is fast and that
values are cached up to a fairly high threshold.

On the other hand, we also want to support fast computation of isolated
large numbers. Currently, no such acceleration is provided for integer
factorials (though it is for large floating-point factorials, which are
computed via gamma if the precision is low enough).

For sequential computation of Bernoulli numbers, we use Ramanujan's formula

                           / n + 3 \
  B   =  (A(n) - S(n))  /  |       |
   n                       \   n   /

where A(n) = (n+3)/3 when n = 0 or 2 (mod 6), A(n) = -(n+3)/6
when n = 4 (mod 6), and

         [n/6]
          ___
         \      /  n + 3  \
  S(n) =  )     |         | * B
         /___   \ n - 6*k /    n-6*k
         k = 1

For isolated large Bernoulli numbers, we use the Riemann zeta function
to calculate a numerical value for B_n. The von Staudt-Clausen theorem
can then be used to optionally find the exact value of the
numerator and denominator.
"""

bernoulli_cache = {}
f3 = from_int(3)
f6 = from_int(6)

def bernoulli_size(n):
    """Accurately estimate the size of B_n (even n > 2 only)"""
    lgn = math.log(n,2)
    return int(2.326 + 0.5*lgn + n*(lgn - 4.094))

BERNOULLI_PREC_CUTOFF = bernoulli_size(MAX_BERNOULLI_CACHE)

def mpf_bernoulli(n, prec, rnd=None):
    """Computation of Bernoulli numbers (numerically)"""
    if n < 2:
        if n < 0:
            raise ValueError("Bernoulli numbers only defined for n >= 0")
        if n == 0:
            return fone
        if n == 1:
            return mpf_neg(fhalf)
    # For odd n > 1, the Bernoulli numbers are zero
    if n & 1:
        return fzero
    # If precision is extremely high, we can save time by computing
    # the Bernoulli number at a lower precision that is sufficient to
    # obtain the exact fraction, round to the exact fraction, and
    # convert the fraction back to an mpf value at the original precision
    if prec > BERNOULLI_PREC_CUTOFF and prec > bernoulli_size(n)*1.1 + 1000:
        p, q = bernfrac(n)
        return from_rational(p, q, prec, rnd or round_floor)
    if n > MAX_BERNOULLI_CACHE:
        return mpf_bernoulli_huge(n, prec, rnd)
    wp = prec + 30
    # Reuse nearby precisions
    wp += 32 - (prec & 31)
    cached = bernoulli_cache.get(wp)
    if cached:
        numbers, state = cached
        if n in numbers:
            if not rnd:
                return numbers[n]
            return mpf_pos(numbers[n], prec, rnd)
        m, bin, bin1 = state
        if n - m > 10:
            return mpf_bernoulli_huge(n, prec, rnd)
    else:
        if n > 10:
            return mpf_bernoulli_huge(n, prec, rnd)
        numbers = {0:fone}
        m, bin, bin1 = state = [2, MPZ(10), MPZ_ONE]
        bernoulli_cache[wp] = (numbers, state)
    while m <= n:
        #print m
        case = m % 6
        # Accurately estimate size of B_m so we can use
        # fixed point math without using too much precision
        szbm = bernoulli_size(m)
        s = 0
        sexp = max(0, szbm)  - wp
        if m < 6:
            a = MPZ_ZERO
        else:
            a = bin1
        for j in xrange(1, m//6+1):
            usign, uman, uexp, ubc = u = numbers[m-6*j]
            if usign:
                uman = -uman
            s += lshift(a*uman, uexp-sexp)
            # Update inner binomial coefficient
            j6 = 6*j
            a *= ((m-5-j6)*(m-4-j6)*(m-3-j6)*(m-2-j6)*(m-1-j6)*(m-j6))
            a //= ((4+j6)*(5+j6)*(6+j6)*(7+j6)*(8+j6)*(9+j6))
        if case == 0: b = mpf_rdiv_int(m+3, f3, wp)
        if case == 2: b = mpf_rdiv_int(m+3, f3, wp)
        if case == 4: b = mpf_rdiv_int(-m-3, f6, wp)
        s = from_man_exp(s, sexp, wp)
        b = mpf_div(mpf_sub(b, s, wp), from_int(bin), wp)
        numbers[m] = b
        m += 2
        # Update outer binomial coefficient
        bin = bin * ((m+2)*(m+3)) // (m*(m-1))
        if m > 6:
            bin1 = bin1 * ((2+m)*(3+m)) // ((m-7)*(m-6))
        state[:] = [m, bin, bin1]
    return numbers[n]

def mpf_bernoulli_huge(n, prec, rnd=None):
    wp = prec + 10
    piprec = wp + int(math.log(n,2))
    v = mpf_gamma_int(n+1, wp)
    v = mpf_mul(v, mpf_zeta_int(n, wp), wp)
    v = mpf_mul(v, mpf_pow_int(mpf_pi(piprec), -n, wp))
    v = mpf_shift(v, 1-n)
    if not n & 3:
        v = mpf_neg(v)
    return mpf_pos(v, prec, rnd or round_fast)

def bernfrac(n):
    r"""
    Returns a tuple of integers `(p, q)` such that `p/q = B_n` exactly,
    where `B_n` denotes the `n`-th Bernoulli number. The fraction is
    always reduced to lowest terms. Note that for `n > 1` and `n` odd,
    `B_n = 0`, and `(0, 1)` is returned.

    **Examples**

    The first few Bernoulli numbers are exactly::

        >>> from mpmath import *
        >>> for n in range(15):
        ...     p, q = bernfrac(n)
        ...     print("%s %s/%s" % (n, p, q))
        ...
        0 1/1
        1 -1/2
        2 1/6
        3 0/1
        4 -1/30
        5 0/1
        6 1/42
        7 0/1
        8 -1/30
        9 0/1
        10 5/66
        11 0/1
        12 -691/2730
        13 0/1
        14 7/6

    This function works for arbitrarily large `n`::

        >>> p, q = bernfrac(10**4)
        >>> print(q)
        2338224387510
        >>> print(len(str(p)))
        27692
        >>> mp.dps = 15
        >>> print(mpf(p) / q)
        -9.04942396360948e+27677
        >>> print(bernoulli(10**4))
        -9.04942396360948e+27677

    .. note ::

        :func:`~mpmath.bernoulli` computes a floating-point approximation
        directly, without computing the exact fraction first.
        This is much faster for large `n`.

    **Algorithm**

    :func:`~mpmath.bernfrac` works by computing the value of `B_n` numerically
    and then using the von Staudt-Clausen theorem [1] to reconstruct
    the exact fraction. For large `n`, this is significantly faster than
    computing `B_1, B_2, \ldots, B_2` recursively with exact arithmetic.
    The implementation has been tested for `n = 10^m` up to `m = 6`.

    In practice, :func:`~mpmath.bernfrac` appears to be about three times
    slower than the specialized program calcbn.exe [2]

    **References**

    1. MathWorld, von Staudt-Clausen Theorem:
       http://mathworld.wolfram.com/vonStaudt-ClausenTheorem.html

    2. The Bernoulli Number Page:
       http://www.bernoulli.org/

    """
    n = int(n)
    if n < 3:
        return [(1, 1), (-1, 2), (1, 6)][n]
    if n & 1:
        return (0, 1)
    q = 1
    for k in list_primes(n+1):
        if not (n % (k-1)):
            q *= k
    prec = bernoulli_size(n) + int(math.log(q,2)) + 20
    b = mpf_bernoulli(n, prec)
    p = mpf_mul(b, from_int(q))
    pint = to_int(p, round_nearest)
    return (pint, q)


#-----------------------------------------------------------------------#
#                                                                       #
#                         Polygamma functions                           #
#                                                                       #
#-----------------------------------------------------------------------#

r"""
For all polygamma (psi) functions, we use the Euler-Maclaurin summation
formula. It looks slightly different in the m = 0 and m > 0 cases.

For m = 0, we have
                                 oo
                                ___   B
       (0)                1    \       2 k    -2 k
    psi   (z)  ~ log z + --- -  )    ------  z
                         2 z   /___  (2 k)!
                               k = 1

Experiment shows that the minimum term of the asymptotic series
reaches 2^(-p) when Re(z) > 0.11*p. So we simply use the recurrence
for psi (equivalent, in fact, to summing to the first few terms
directly before applying E-M) to obtain z large enough.

Since, very crudely, log z ~= 1 for Re(z) > 1, we can use
fixed-point arithmetic  (if z is extremely large, log(z) itself
is a sufficient approximation, so we can stop there already).

For Re(z) << 0, we could use recurrence, but this is of course
inefficient for large negative z, so there we use the
reflection formula instead.

For m > 0, we have

                  N - 1
                   ___
  ~~~(m)       [  \          1    ]         1            1
  psi   (z)  ~ [   )     -------- ] +  ---------- +  -------- +
               [  /___        m+1 ]           m+1           m
                  k = 1  (z+k)    ]    2 (z+N)       m (z+N)

      oo
     ___    B
    \        2 k   (m+1) (m+2) ... (m+2k-1)
  +  )     ------  ------------------------
    /___   (2 k)!            m + 2 k
    k = 1               (z+N)

where ~~~ denotes the function rescaled by 1/((-1)^(m+1) m!).

Here again N is chosen to make z+N large enough for the minimum
term in the last series to become smaller than eps.

TODO: the current estimation of N for m > 0 is *very suboptimal*.

TODO: implement the reflection formula for m > 0, Re(z) << 0.
It is generally a combination of multiple cotangents. Need to
figure out a reasonably simple way to generate these formulas
on the fly.

TODO: maybe use exact algorithms to compute psi for integral
and certain rational arguments, as this can be much more
efficient. (On the other hand, the availability of these
special values provides a convenient way to test the general
algorithm.)
"""

# Harmonic numbers are just shifted digamma functions
# We should calculate these exactly when x is an integer
# and when doing so is faster.

def mpf_harmonic(x, prec, rnd):
    if x in (fzero, fnan, finf):
        return x
    a = mpf_psi0(mpf_add(fone, x, prec+5), prec)
    return mpf_add(a, mpf_euler(prec+5, rnd), prec, rnd)

def mpc_harmonic(z, prec, rnd):
    if z[1] == fzero:
        return (mpf_harmonic(z[0], prec, rnd), fzero)
    a = mpc_psi0(mpc_add_mpf(z, fone, prec+5), prec)
    return mpc_add_mpf(a, mpf_euler(prec+5, rnd), prec, rnd)

def mpf_psi0(x, prec, rnd=round_fast):
    """
    Computation of the digamma function (psi function of order 0)
    of a real argument.
    """
    sign, man, exp, bc = x
    wp = prec + 10
    if not man:
        if x == finf: return x
        if x == fninf or x == fnan: return fnan
    if x == fzero or (exp >= 0 and sign):
        raise ValueError("polygamma pole")
    # Near 0 -- fixed-point arithmetic becomes bad
    if exp+bc < -5:
        v = mpf_psi0(mpf_add(x, fone, prec, rnd), prec, rnd)
        return mpf_sub(v, mpf_div(fone, x, wp, rnd), prec, rnd)
    # Reflection formula
    if sign and exp+bc > 3:
        c, s = mpf_cos_sin_pi(x, wp)
        q = mpf_mul(mpf_div(c, s, wp), mpf_pi(wp), wp)
        p = mpf_psi0(mpf_sub(fone, x, wp), wp)
        return mpf_sub(p, q, prec, rnd)
    # The logarithmic term is accurate enough
    if (not sign) and bc + exp > wp:
        return mpf_log(mpf_sub(x, fone, wp), prec, rnd)
    # Initial recurrence to obtain a large enough x
    m = to_int(x)
    n = int(0.11*wp) + 2
    s = MPZ_ZERO
    x = to_fixed(x, wp)
    one = MPZ_ONE << wp
    if m < n:
        for k in xrange(m, n):
            s -= (one << wp) // x
            x += one
    x -= one
    # Logarithmic term
    s += to_fixed(mpf_log(from_man_exp(x, -wp, wp), wp), wp)
    # Endpoint term in Euler-Maclaurin expansion
    s += (one << wp) // (2*x)
    # Euler-Maclaurin remainder sum
    x2 = (x*x) >> wp
    t = one
    prev = 0
    k = 1
    while 1:
        t = (t*x2) >> wp
        bsign, bman, bexp, bbc = mpf_bernoulli(2*k, wp)
        offset = (bexp + 2*wp)
        if offset >= 0: term = (bman << offset) // (t*(2*k))
        else:           term = (bman >> (-offset)) // (t*(2*k))
        if k & 1: s -= term
        else:     s += term
        if k > 2 and term >= prev:
            break
        prev = term
        k += 1
    return from_man_exp(s, -wp, wp, rnd)

def mpc_psi0(z, prec, rnd=round_fast):
    """
    Computation of the digamma function (psi function of order 0)
    of a complex argument.
    """
    re, im = z
    # Fall back to the real case
    if im == fzero:
        return (mpf_psi0(re, prec, rnd), fzero)
    wp = prec + 20
    sign, man, exp, bc = re
    # Reflection formula
    if sign and exp+bc > 3:
        c = mpc_cos_pi(z, wp)
        s = mpc_sin_pi(z, wp)
        q = mpc_mul_mpf(mpc_div(c, s, wp), mpf_pi(wp), wp)
        p = mpc_psi0(mpc_sub(mpc_one, z, wp), wp)
        return mpc_sub(p, q, prec, rnd)
    # Just the logarithmic term
    if (not sign) and bc + exp > wp:
        return mpc_log(mpc_sub(z, mpc_one, wp), prec, rnd)
    # Initial recurrence to obtain a large enough z
    w = to_int(re)
    n = int(0.11*wp) + 2
    s = mpc_zero
    if w < n:
        for k in xrange(w, n):
            s = mpc_sub(s, mpc_reciprocal(z, wp), wp)
            z = mpc_add_mpf(z, fone, wp)
    z = mpc_sub(z, mpc_one, wp)
    # Logarithmic and endpoint term
    s = mpc_add(s, mpc_log(z, wp), wp)
    s = mpc_add(s, mpc_div(mpc_half, z, wp), wp)
    # Euler-Maclaurin remainder sum
    z2 = mpc_square(z, wp)
    t = mpc_one
    prev = mpc_zero
    szprev = fzero
    k = 1
    eps = mpf_shift(fone, -wp+2)
    while 1:
        t = mpc_mul(t, z2, wp)
        bern = mpf_bernoulli(2*k, wp)
        term = mpc_mpf_div(bern, mpc_mul_int(t, 2*k, wp), wp)
        s = mpc_sub(s, term, wp)
        szterm = mpc_abs(term, 10)
        if k > 2 and (mpf_le(szterm, eps) or mpf_le(szprev, szterm)):
            break
        prev = term
        szprev = szterm
        k += 1
    return s

# Currently unoptimized
def mpf_psi(m, x, prec, rnd=round_fast):
    """
    Computation of the polygamma function of arbitrary integer order
    m >= 0, for a real argument x.
    """
    if m == 0:
        return mpf_psi0(x, prec, rnd=round_fast)
    return mpc_psi(m, (x, fzero), prec, rnd)[0]

def mpc_psi(m, z, prec, rnd=round_fast):
    """
    Computation of the polygamma function of arbitrary integer order
    m >= 0, for a complex argument z.
    """
    if m == 0:
        return mpc_psi0(z, prec, rnd)
    re, im = z
    wp = prec + 20
    sign, man, exp, bc = re
    if not im[1]:
        if im in (finf, fninf, fnan):
            return (fnan, fnan)
    if not man:
        if re == finf and im == fzero:
            return (fzero, fzero)
        if re == fnan:
            return (fnan, fnan)
    # Recurrence
    w = to_int(re)
    n = int(0.4*wp + 4*m)
    s = mpc_zero
    if w < n:
        for k in xrange(w, n):
            t = mpc_pow_int(z, -m-1, wp)
            s = mpc_add(s, t, wp)
            z = mpc_add_mpf(z, fone, wp)
    zm = mpc_pow_int(z, -m, wp)
    z2 = mpc_pow_int(z, -2, wp)
    # 1/m*(z+N)^m
    integral_term = mpc_div_mpf(zm, from_int(m), wp)
    s = mpc_add(s, integral_term, wp)
    # 1/2*(z+N)^(-(m+1))
    s = mpc_add(s, mpc_mul_mpf(mpc_div(zm, z, wp), fhalf, wp), wp)
    a = m + 1
    b = 2
    k = 1
    # Important: we want to sum up to the *relative* error,
    # not the absolute error, because psi^(m)(z) might be tiny
    magn = mpc_abs(s, 10)
    magn = magn[2]+magn[3]
    eps = mpf_shift(fone, magn-wp+2)
    while 1:
        zm = mpc_mul(zm, z2, wp)
        bern = mpf_bernoulli(2*k, wp)
        scal = mpf_mul_int(bern, a, wp)
        scal = mpf_div(scal, from_int(b), wp)
        term = mpc_mul_mpf(zm, scal, wp)
        s = mpc_add(s, term, wp)
        szterm = mpc_abs(term, 10)
        if k > 2 and mpf_le(szterm, eps):
            break
        #print k, to_str(szterm, 10), to_str(eps, 10)
        a *= (m+2*k)*(m+2*k+1)
        b *= (2*k+1)*(2*k+2)
        k += 1
    # Scale and sign factor
    v = mpc_mul_mpf(s, mpf_gamma(from_int(m+1), wp), prec, rnd)
    if not (m & 1):
        v = mpf_neg(v[0]), mpf_neg(v[1])
    return v


#-----------------------------------------------------------------------#
#                                                                       #
#                         Riemann zeta function                         #
#                                                                       #
#-----------------------------------------------------------------------#

r"""
We use zeta(s) = eta(s) / (1 - 2**(1-s)) and Borwein's approximation

                  n-1
                  ___       k
             -1  \      (-1)  (d_k - d_n)
  eta(s) ~= ----  )     ------------------
             d_n /___              s
                 k = 0      (k + 1)
where
             k
             ___                i
            \     (n + i - 1)! 4
  d_k  =  n  )    ---------------.
            /___   (n - i)! (2i)!
            i = 0

If s = a + b*I, the absolute error for eta(s) is bounded by

    3 (1 + 2|b|)
    ------------ * exp(|b| pi/2)
               n
    (3+sqrt(8))

Disregarding the linear term, we have approximately,

  log(err) ~= log(exp(1.58*|b|)) - log(5.8**n)
  log(err) ~= 1.58*|b| - log(5.8)*n
  log(err) ~= 1.58*|b| - 1.76*n
  log2(err) ~= 2.28*|b| - 2.54*n

So for p bits, we should choose n > (p + 2.28*|b|) / 2.54.

References:
-----------

Peter Borwein, "An Efficient Algorithm for the Riemann Zeta Function"
http://www.cecm.sfu.ca/personal/pborwein/PAPERS/P117.ps

http://en.wikipedia.org/wiki/Dirichlet_eta_function
"""

borwein_cache = {}

def borwein_coefficients(n):
    if n in borwein_cache:
        return borwein_cache[n]
    ds = [MPZ_ZERO] * (n+1)
    d = MPZ_ONE
    s = ds[0] = MPZ_ONE
    for i in range(1, n+1):
        d = d * 4 * (n+i-1) * (n-i+1)
        d //= ((2*i) * ((2*i)-1))
        s += d
        ds[i] = s
    borwein_cache[n] = ds
    return ds

ZETA_INT_CACHE_MAX_PREC = 1000
zeta_int_cache = {}

def mpf_zeta_int(s, prec, rnd=round_fast):
    """
    Optimized computation of zeta(s) for an integer s.
    """
    wp = prec + 20
    s = int(s)
    if s in zeta_int_cache and zeta_int_cache[s][0] >= wp:
        return mpf_pos(zeta_int_cache[s][1], prec, rnd)
    if s < 2:
        if s == 1:
            raise ValueError("zeta(1) pole")
        if not s:
            return mpf_neg(fhalf)
        return mpf_div(mpf_bernoulli(-s+1, wp), from_int(s-1), prec, rnd)
    # 2^-s term vanishes?
    if s >= wp:
        return mpf_perturb(fone, 0, prec, rnd)
    # 5^-s term vanishes?
    elif s >= wp*0.431:
        t = one = 1 << wp
        t += 1 << (wp - s)
        t += one // (MPZ_THREE ** s)
        t += 1 << max(0, wp - s*2)
        return from_man_exp(t, -wp, prec, rnd)
    else:
        # Fast enough to sum directly?
        # Even better, we use the Euler product (idea stolen from pari)
        m = (float(wp)/(s-1) + 1)
        if m < 30:
            needed_terms = int(2.0**m + 1)
            if needed_terms < int(wp/2.54 + 5) / 10:
                t = fone
                for k in list_primes(needed_terms):
                    #print k, needed_terms
                    powprec = int(wp - s*math.log(k,2))
                    if powprec < 2:
                        break
                    a = mpf_sub(fone, mpf_pow_int(from_int(k), -s, powprec), wp)
                    t = mpf_mul(t, a, wp)
                return mpf_div(fone, t, wp)
    # Use Borwein's algorithm
    n = int(wp/2.54 + 5)
    d = borwein_coefficients(n)
    t = MPZ_ZERO
    s = MPZ(s)
    for k in xrange(n):
        t += (((-1)**k * (d[k] - d[n])) << wp) // (k+1)**s
    t = (t << wp) // (-d[n])
    t = (t << wp) // ((1 << wp) - (1 << (wp+1-s)))
    if (s in zeta_int_cache and zeta_int_cache[s][0] < wp) or (s not in zeta_int_cache):
        zeta_int_cache[s] = (wp, from_man_exp(t, -wp-wp))
    return from_man_exp(t, -wp-wp, prec, rnd)

def mpf_zeta(s, prec, rnd=round_fast, alt=0):
    sign, man, exp, bc = s
    if not man:
        if s == fzero:
            if alt:
                return fhalf
            else:
                return mpf_neg(fhalf)
        if s == finf:
            return fone
        return fnan
    wp = prec + 20
    # First term vanishes?
    if (not sign) and (exp + bc > (math.log(wp,2) + 2)):
        return mpf_perturb(fone, alt, prec, rnd)
    # Optimize for integer arguments
    elif exp >= 0:
        if alt:
            if s == fone:
                return mpf_ln2(prec, rnd)
            z = mpf_zeta_int(to_int(s), wp, negative_rnd[rnd])
            q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
            return mpf_mul(z, q, prec, rnd)
        else:
            return mpf_zeta_int(to_int(s), prec, rnd)
    # Negative: use the reflection formula
    # Borwein only proves the accuracy bound for x >= 1/2. However, based on
    # tests, the accuracy without reflection is quite good even some distance
    # to the left of 1/2. XXX: verify this.
    if sign:
        # XXX: could use the separate refl. formula for Dirichlet eta
        if alt:
            q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
            return mpf_mul(mpf_zeta(s, wp), q, prec, rnd)
        # XXX: -1 should be done exactly
        y = mpf_sub(fone, s, 10*wp)
        a = mpf_gamma(y, wp)
        b = mpf_zeta(y, wp)
        c = mpf_sin_pi(mpf_shift(s, -1), wp)
        wp2 = wp + max(0,exp+bc)
        pi = mpf_pi(wp+wp2)
        d = mpf_div(mpf_pow(mpf_shift(pi, 1), s, wp2), pi, wp2)
        return mpf_mul(a,mpf_mul(b,mpf_mul(c,d,wp),wp),prec,rnd)

    # Near pole
    r = mpf_sub(fone, s, wp)
    asign, aman, aexp, abc = mpf_abs(r)
    pole_dist = -2*(aexp+abc)
    if pole_dist > wp:
        if alt:
            return mpf_ln2(prec, rnd)
        else:
            q = mpf_neg(mpf_div(fone, r, wp))
            return mpf_add(q, mpf_euler(wp), prec, rnd)
    else:
        wp += max(0, pole_dist)

    t = MPZ_ZERO
    #wp += 16 - (prec & 15)
    # Use Borwein's algorithm
    n = int(wp/2.54 + 5)
    d = borwein_coefficients(n)
    t = MPZ_ZERO
    sf = to_fixed(s, wp)
    ln2 = ln2_fixed(wp)
    for k in xrange(n):
        u = (-sf*log_int_fixed(k+1, wp, ln2)) >> wp
        #esign, eman, eexp, ebc = mpf_exp(u, wp)
        #offset = eexp + wp
        #if offset >= 0:
        #    w = ((d[k] - d[n]) * eman) << offset
        #else:
        #    w = ((d[k] - d[n]) * eman) >> (-offset)
        eman = exp_fixed(u, wp, ln2)
        w = (d[k] - d[n]) * eman
        if k & 1:
            t -= w
        else:
            t += w
    t = t // (-d[n])
    t = from_man_exp(t, -wp, wp)
    if alt:
        return mpf_pos(t, prec, rnd)
    else:
        q = mpf_sub(fone, mpf_pow(ftwo, mpf_sub(fone, s, wp), wp), wp)
        return mpf_div(t, q, prec, rnd)

def mpc_zeta(s, prec, rnd=round_fast, alt=0, force=False):
    re, im = s
    if im == fzero:
        return mpf_zeta(re, prec, rnd, alt), fzero

    # slow for large s
    if (not force) and mpf_gt(mpc_abs(s, 10), from_int(prec)):
        raise NotImplementedError

    wp = prec + 20

    # Near pole
    r = mpc_sub(mpc_one, s, wp)
    asign, aman, aexp, abc = mpc_abs(r, 10)
    pole_dist = -2*(aexp+abc)
    if pole_dist > wp:
        if alt:
            q = mpf_ln2(wp)
            y = mpf_mul(q, mpf_euler(wp), wp)
            g = mpf_shift(mpf_mul(q, q, wp), -1)
            g = mpf_sub(y, g)
            z = mpc_mul_mpf(r, mpf_neg(g), wp)
            z = mpc_add_mpf(z, q, wp)
            return mpc_pos(z, prec, rnd)
        else:
            q = mpc_neg(mpc_div(mpc_one, r, wp))
            q = mpc_add_mpf(q, mpf_euler(wp), wp)
            return mpc_pos(q, prec, rnd)
    else:
        wp += max(0, pole_dist)

    # Reflection formula. To be rigorous, we should reflect to the left of
    # re = 1/2 (see comments for mpf_zeta), but this leads to unnecessary
    # slowdown for interesting values of s
    if mpf_lt(re, fzero):
        # XXX: could use the separate refl. formula for Dirichlet eta
        if alt:
            q = mpc_sub(mpc_one, mpc_pow(mpc_two, mpc_sub(mpc_one, s, wp),
                wp), wp)
            return mpc_mul(mpc_zeta(s, wp), q, prec, rnd)
        # XXX: -1 should be done exactly
        y = mpc_sub(mpc_one, s, 10*wp)
        a = mpc_gamma(y, wp)
        b = mpc_zeta(y, wp)
        c = mpc_sin_pi(mpc_shift(s, -1), wp)
        rsign, rman, rexp, rbc = re
        isign, iman, iexp, ibc = im
        mag = max(rexp+rbc, iexp+ibc)
        wp2 = wp + max(0, mag)
        pi = mpf_pi(wp+wp2)
        pi2 = (mpf_shift(pi, 1), fzero)
        d = mpc_div_mpf(mpc_pow(pi2, s, wp2), pi, wp2)
        return mpc_mul(a,mpc_mul(b,mpc_mul(c,d,wp),wp),prec,rnd)
    n = int(wp/2.54 + 5)
    n += int(0.9*abs(to_int(im)))
    d = borwein_coefficients(n)
    ref = to_fixed(re, wp)
    imf = to_fixed(im, wp)
    tre = MPZ_ZERO
    tim = MPZ_ZERO
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)
    critical_line = re == fhalf
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    wp2 = wp+wp
    for k in xrange(n):
        log = log_int_fixed(k+1, wp, ln2)
        # A square root is much cheaper than an exp
        if critical_line:
            w = one_2wp // isqrt_fast((k+1) << wp2)
        else:
            w = exp_fixed((-ref*log) >> wp, wp)
        if k & 1:
            w *= (d[n] - d[k])
        else:
            w *= (d[k] - d[n])
        wre, wim = cos_sin_fixed((-imf*log)>>wp, wp, pi2)
        tre += (w * wre) >> wp
        tim += (w * wim) >> wp
    tre //= (-d[n])
    tim //= (-d[n])
    tre = from_man_exp(tre, -wp, wp)
    tim = from_man_exp(tim, -wp, wp)
    if alt:
        return mpc_pos((tre, tim), prec, rnd)
    else:
        q = mpc_sub(mpc_one, mpc_pow(mpc_two, r, wp), wp)
        return mpc_div((tre, tim), q, prec, rnd)

def mpf_altzeta(s, prec, rnd=round_fast):
    return mpf_zeta(s, prec, rnd, 1)

def mpc_altzeta(s, prec, rnd=round_fast):
    return mpc_zeta(s, prec, rnd, 1)

# Not optimized currently
mpf_zetasum = None


def pow_fixed(x, n, wp):
    if n == 1:
        return x
    y = MPZ_ONE << wp
    while n:
        if n & 1:
            y = (y*x) >> wp
            n -= 1
        x = (x*x) >> wp
        n //= 2
    return y

# TODO: optimize / cleanup interface / unify with list_primes
sieve_cache = []
primes_cache = []
mult_cache = []

def primesieve(n):
    global sieve_cache, primes_cache, mult_cache
    if n < len(sieve_cache):
        sieve = sieve_cache#[:n+1]
        primes = primes_cache[:primes_cache.index(max(sieve))+1]
        mult = mult_cache#[:n+1]
        return sieve, primes, mult
    sieve = [0] * (n+1)
    mult = [0] * (n+1)
    primes = list_primes(n)
    for p in primes:
        #sieve[p::p] = p
        for k in xrange(p,n+1,p):
            sieve[k] = p
    for i, p in enumerate(sieve):
        if i >= 2:
            m = 1
            n = i // p
            while not n % p:
                n //= p
                m += 1
            mult[i] = m
    sieve_cache = sieve
    primes_cache = primes
    mult_cache = mult
    return sieve, primes, mult

def zetasum_sieved(critical_line, sre, sim, a, n, wp):
    if a < 1:
        raise ValueError("a cannot be less than 1")
    sieve, primes, mult = primesieve(a+n)
    basic_powers = {}
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)
    wp2 = wp+wp
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    for p in primes:
        if p*2 > a+n:
            break
        log = log_int_fixed(p, wp, ln2)
        cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
        if critical_line:
            u = one_2wp // isqrt_fast(p<<wp2)
        else:
            u = exp_fixed((-sre*log)>>wp, wp)
        pre = (u*cos) >> wp
        pim = (u*sin) >> wp
        basic_powers[p] = [(pre, pim)]
        tre, tim = pre, pim
        for m in range(1,int(math.log(a+n,p)+0.01)+1):
            tre, tim = ((pre*tre-pim*tim)>>wp), ((pim*tre+pre*tim)>>wp)
            basic_powers[p].append((tre,tim))
    xre = MPZ_ZERO
    xim = MPZ_ZERO
    if a == 1:
        xre += one
    aa = max(a,2)
    for k in xrange(aa, a+n+1):
        p = sieve[k]
        if p in basic_powers:
            m = mult[k]
            tre, tim = basic_powers[p][m-1]
            while 1:
                k //= p**m
                if k == 1:
                    break
                p = sieve[k]
                m = mult[k]
                pre, pim = basic_powers[p][m-1]
                tre, tim = ((pre*tre-pim*tim)>>wp), ((pim*tre+pre*tim)>>wp)
        else:
            log = log_int_fixed(k, wp, ln2)
            cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
            if critical_line:
                u = one_2wp // isqrt_fast(k<<wp2)
            else:
                u = exp_fixed((-sre*log)>>wp, wp)
            tre = (u*cos) >> wp
            tim = (u*sin) >> wp
        xre += tre
        xim += tim
    return xre, xim

# Set to something large to disable
ZETASUM_SIEVE_CUTOFF = 10

def mpc_zetasum(s, a, n, derivatives, reflect, prec):
    """
    Fast version of mp._zetasum, assuming s = complex, a = integer.
    """

    wp = prec + 10
    derivatives = list(derivatives)
    have_derivatives = derivatives != [0]
    have_one_derivative = len(derivatives) == 1

    # parse s
    sre, sim = s
    critical_line = (sre == fhalf)
    sre = to_fixed(sre, wp)
    sim = to_fixed(sim, wp)

    if a > 0 and n > ZETASUM_SIEVE_CUTOFF and not have_derivatives \
            and not reflect and (n < 4e7 or sys.maxsize > 2**32):
        re, im = zetasum_sieved(critical_line, sre, sim, a, n, wp)
        xs = [(from_man_exp(re, -wp, prec, 'n'), from_man_exp(im, -wp, prec, 'n'))]
        return xs, []

    maxd = max(derivatives)
    if not have_one_derivative:
        derivatives = range(maxd+1)

    # x_d = 0, y_d = 0
    xre = [MPZ_ZERO for d in derivatives]
    xim = [MPZ_ZERO for d in derivatives]
    if reflect:
        yre = [MPZ_ZERO for d in derivatives]
        yim = [MPZ_ZERO for d in derivatives]
    else:
        yre = yim = []

    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << (2*wp)

    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp-1)
    wp2 = wp+wp

    for w in xrange(a, a+n+1):
        log = log_int_fixed(w, wp, ln2)
        cos, sin = cos_sin_fixed((-sim*log)>>wp, wp, pi2)
        if critical_line:
            u = one_2wp // isqrt_fast(w<<wp2)
        else:
            u = exp_fixed((-sre*log)>>wp, wp)
        xterm_re = (u * cos) >> wp
        xterm_im = (u * sin) >> wp
        if reflect:
            reciprocal = (one_2wp // (u*w))
            yterm_re = (reciprocal * cos) >> wp
            yterm_im = (reciprocal * sin) >> wp

        if have_derivatives:
            if have_one_derivative:
                log = pow_fixed(log, maxd, wp)
                xre[0] += (xterm_re * log) >> wp
                xim[0] += (xterm_im * log) >> wp
                if reflect:
                    yre[0] += (yterm_re * log) >> wp
                    yim[0] += (yterm_im * log) >> wp
            else:
                t = MPZ_ONE << wp
                for d in derivatives:
                    xre[d] += (xterm_re * t) >> wp
                    xim[d] += (xterm_im * t) >> wp
                    if reflect:
                        yre[d] += (yterm_re * t) >> wp
                        yim[d] += (yterm_im * t) >> wp
                    t = (t * log) >> wp
        else:
            xre[0] += xterm_re
            xim[0] += xterm_im
            if reflect:
                yre[0] += yterm_re
                yim[0] += yterm_im
    if have_derivatives:
        if have_one_derivative:
            if maxd % 2:
                xre[0] = -xre[0]
                xim[0] = -xim[0]
                if reflect:
                    yre[0] = -yre[0]
                    yim[0] = -yim[0]
        else:
            xre = [(-1)**d * xre[d] for d in derivatives]
            xim = [(-1)**d * xim[d] for d in derivatives]
            if reflect:
                yre = [(-1)**d * yre[d] for d in derivatives]
                yim = [(-1)**d * yim[d] for d in derivatives]
    xs = [(from_man_exp(xa, -wp, prec, 'n'), from_man_exp(xb, -wp, prec, 'n'))
        for (xa, xb) in zip(xre, xim)]
    ys = [(from_man_exp(ya, -wp, prec, 'n'), from_man_exp(yb, -wp, prec, 'n'))
        for (ya, yb) in zip(yre, yim)]
    return xs, ys


#-----------------------------------------------------------------------#
#                                                                       #
#              The gamma function  (NEW IMPLEMENTATION)                 #
#                                                                       #
#-----------------------------------------------------------------------#

# Higher means faster, but more precomputation time
MAX_GAMMA_TAYLOR_PREC = 5000
# Need to derive higher bounds for Taylor series to go higher
assert MAX_GAMMA_TAYLOR_PREC < 15000

# Use Stirling's series if abs(x) > beta*prec
# Important: must be large enough for convergence!
GAMMA_STIRLING_BETA = 0.2

SMALL_FACTORIAL_CACHE_SIZE = 150

gamma_taylor_cache = {}
gamma_stirling_cache = {}

small_factorial_cache = [from_int(ifac(n)) for \
    n in range(SMALL_FACTORIAL_CACHE_SIZE+1)]

def zeta_array(N, prec):
    """
    zeta(n) = A * pi**n / n! + B

    where A is a rational number (A = Bernoulli number
    for n even) and B is an infinite sum over powers of exp(2*pi).
    (B = 0 for n even).

    TODO: this is currently only used for gamma, but could
    be very useful elsewhere.
    """
    extra = 30
    wp = prec+extra
    zeta_values = [MPZ_ZERO] * (N+2)
    pi = pi_fixed(wp)
    # STEP 1:
    one = MPZ_ONE << wp
    zeta_values[0] = -one//2
    f_2pi = mpf_shift(mpf_pi(wp),1)
    exp_2pi_k = exp_2pi = mpf_exp(f_2pi, wp)
    # Compute exponential series
    # Store values of 1/(exp(2*pi*k)-1),
    # exp(2*pi*k)/(exp(2*pi*k)-1)**2, 1/(exp(2*pi*k)-1)**2
    # pi*k*exp(2*pi*k)/(exp(2*pi*k)-1)**2
    exps3 = []
    k = 1
    while 1:
        tp = wp - 9*k
        if tp < 1:
            break
        # 1/(exp(2*pi*k-1)
        q1 = mpf_div(fone, mpf_sub(exp_2pi_k, fone, tp), tp)
        # pi*k*exp(2*pi*k)/(exp(2*pi*k)-1)**2
        q2 = mpf_mul(exp_2pi_k, mpf_mul(q1,q1,tp), tp)
        q1 = to_fixed(q1, wp)
        q2 = to_fixed(q2, wp)
        q2 = (k * q2 * pi) >> wp
        exps3.append((q1, q2))
        # Multiply for next round
        exp_2pi_k = mpf_mul(exp_2pi_k, exp_2pi, wp)
        k += 1
    # Exponential sum
    for n in xrange(3, N+1, 2):
        s = MPZ_ZERO
        k = 1
        for e1, e2 in exps3:
            if n%4 == 3:
                t = e1 // k**n
            else:
                U = (n-1)//4
                t = (e1 + e2//U) // k**n
            if not t:
                break
            s += t
            k += 1
        zeta_values[n] = -2*s
    # Even zeta values
    B = [mpf_abs(mpf_bernoulli(k,wp)) for k in xrange(N+2)]
    pi_pow = fpi = mpf_pow_int(mpf_shift(mpf_pi(wp), 1), 2, wp)
    pi_pow = mpf_div(pi_pow, from_int(4), wp)
    for n in xrange(2,N+2,2):
        z = mpf_mul(B[n], pi_pow, wp)
        zeta_values[n] = to_fixed(z, wp)
        pi_pow = mpf_mul(pi_pow, fpi, wp)
        pi_pow = mpf_div(pi_pow, from_int((n+1)*(n+2)), wp)
    # Zeta sum
    reciprocal_pi = (one << wp) // pi
    for n in xrange(3, N+1, 4):
        U = (n-3)//4
        s = zeta_values[4*U+4]*(4*U+7)//4
        for k in xrange(1, U+1):
            s -= (zeta_values[4*k] * zeta_values[4*U+4-4*k]) >> wp
        zeta_values[n] += (2*s*reciprocal_pi) >> wp
    for n in xrange(5, N+1, 4):
        U = (n-1)//4
        s = zeta_values[4*U+2]*(2*U+1)
        for k in xrange(1, 2*U+1):
            s += ((-1)**k*2*k* zeta_values[2*k] * zeta_values[4*U+2-2*k])>>wp
        zeta_values[n] += ((s*reciprocal_pi)>>wp)//(2*U)
    return [x>>extra for x in zeta_values]

def gamma_taylor_coefficients(inprec):
    """
    Gives the Taylor coefficients of 1/gamma(1+x) as
    a list of fixed-point numbers. Enough coefficients are returned
    to ensure that the series converges to the given precision
    when x is in [0.5, 1.5].
    """
    # Reuse nearby cache values (small case)
    if inprec < 400:
        prec = inprec + (10-(inprec%10))
    elif inprec < 1000:
        prec = inprec + (30-(inprec%30))
    else:
        prec = inprec
    if prec in gamma_taylor_cache:
        return gamma_taylor_cache[prec], prec

    # Experimentally determined bounds
    if prec < 1000:
        N = int(prec**0.76 + 2)
    else:
        # Valid to at least 15000 bits
        N = int(prec**0.787 + 2)

    # Reuse higher precision values
    for cprec in gamma_taylor_cache:
        if cprec > prec:
            coeffs = [x>>(cprec-prec) for x in gamma_taylor_cache[cprec][-N:]]
            if inprec < 1000:
                gamma_taylor_cache[prec] = coeffs
            return coeffs, prec

    # Cache at a higher precision (large case)
    if prec > 1000:
        prec = int(prec * 1.2)

    wp = prec + 20
    A = [0] * N
    A[0] = MPZ_ZERO
    A[1] = MPZ_ONE << wp
    A[2] = euler_fixed(wp)
    # SLOW, reference implementation
    #zeta_values = [0,0]+[to_fixed(mpf_zeta_int(k,wp),wp) for k in xrange(2,N)]
    zeta_values = zeta_array(N, wp)
    for k in xrange(3, N):
        a = (-A[2]*A[k-1])>>wp
        for j in xrange(2,k):
            a += ((-1)**j * zeta_values[j] * A[k-j]) >> wp
        a //= (1-k)
        A[k] = a
    A = [a>>20 for a in A]
    A = A[::-1]
    A = A[:-1]
    gamma_taylor_cache[prec] = A
    #return A, prec
    return gamma_taylor_coefficients(inprec)

def gamma_fixed_taylor(xmpf, x, wp, prec, rnd, type):
    # Determine nearest multiple of N/2
    #n = int(x >> (wp-1))
    #steps = (n-1)>>1
    nearest_int = ((x >> (wp-1)) + MPZ_ONE) >> 1
    one = MPZ_ONE << wp
    coeffs, cwp = gamma_taylor_coefficients(wp)
    if nearest_int > 0:
        r = one
        for i in xrange(nearest_int-1):
            x -= one
            r = (r*x) >> wp
        x -= one
        p = MPZ_ZERO
        for c in coeffs:
            p = c + ((x*p)>>wp)
        p >>= (cwp-wp)
        if type == 0:
            return from_man_exp((r<<wp)//p, -wp, prec, rnd)
        if type == 2:
            return mpf_shift(from_rational(p, (r<<wp), prec, rnd), wp)
        if type == 3:
            return mpf_log(mpf_abs(from_man_exp((r<<wp)//p, -wp)), prec, rnd)
    else:
        r = one
        for i in xrange(-nearest_int):
            r = (r*x) >> wp
            x += one
        p = MPZ_ZERO
        for c in coeffs:
            p = c + ((x*p)>>wp)
        p >>= (cwp-wp)
        if wp - bitcount(abs(x)) > 10:
            # pass very close to 0, so do floating-point multiply
            g = mpf_add(xmpf, from_int(-nearest_int))  # exact
            r = from_man_exp(p*r,-wp-wp)
            r = mpf_mul(r, g, wp)
            if type == 0:
                return mpf_div(fone, r, prec, rnd)
            if type == 2:
                return mpf_pos(r, prec, rnd)
            if type == 3:
                return mpf_log(mpf_abs(mpf_div(fone, r, wp)), prec, rnd)
        else:
            r = from_man_exp(x*p*r,-3*wp)
            if type == 0: return mpf_div(fone, r, prec, rnd)
            if type == 2: return mpf_pos(r, prec, rnd)
            if type == 3: return mpf_neg(mpf_log(mpf_abs(r), prec, rnd))

def stirling_coefficient(n):
    if n in gamma_stirling_cache:
        return gamma_stirling_cache[n]
    p, q = bernfrac(n)
    q *= MPZ(n*(n-1))
    gamma_stirling_cache[n] = p, q, bitcount(abs(p)), bitcount(q)
    return gamma_stirling_cache[n]

def real_stirling_series(x, prec):
    """
    Sums the rational part of Stirling's expansion,

    log(sqrt(2*pi)) - z + 1/(12*z) - 1/(360*z^3) + ...

    """
    t = (MPZ_ONE<<(prec+prec)) // x   # t = 1/x
    u = (t*t)>>prec                  # u = 1/x**2
    s = ln_sqrt2pi_fixed(prec) - x
    # Add initial terms of Stirling's series
    s += t//12;            t = (t*u)>>prec
    s -= t//360;           t = (t*u)>>prec
    s += t//1260;          t = (t*u)>>prec
    s -= t//1680;          t = (t*u)>>prec
    if not t: return s
    s += t//1188;          t = (t*u)>>prec
    s -= 691*t//360360;    t = (t*u)>>prec
    s += t//156;           t = (t*u)>>prec
    if not t: return s
    s -= 3617*t//122400;   t = (t*u)>>prec
    s += 43867*t//244188;  t = (t*u)>>prec
    s -= 174611*t//125400;  t = (t*u)>>prec
    if not t: return s
    k = 22
    # From here on, the coefficients are growing, so we
    # have to keep t at a roughly constant size
    usize = bitcount(abs(u))
    tsize = bitcount(abs(t))
    texp = 0
    while 1:
        p, q, pb, qb = stirling_coefficient(k)
        term_mag = tsize + pb + texp
        shift = -texp
        m = pb - term_mag
        if m > 0 and shift < m:
            p >>= m
            shift -= m
        m = tsize - term_mag
        if m > 0 and shift < m:
            w = t >> m
            shift -= m
        else:
            w = t
        term = (t*p//q) >> shift
        if not term:
            break
        s += term
        t = (t*u) >> usize
        texp -= (prec - usize)
        k += 2
    return s

def complex_stirling_series(x, y, prec):
    # t = 1/z
    _m = (x*x + y*y) >> prec
    tre = (x << prec) // _m
    tim = (-y << prec) // _m
    # u = 1/z**2
    ure = (tre*tre - tim*tim) >> prec
    uim = tim*tre >> (prec-1)
    # s = log(sqrt(2*pi)) - z
    sre = ln_sqrt2pi_fixed(prec) - x
    sim = -y

    # Add initial terms of Stirling's series
    sre += tre//12; sim += tim//12;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= tre//360; sim -= tim//360;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += tre//1260; sim += tim//1260;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= tre//1680; sim -= tim//1680;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim
    sre += tre//1188; sim += tim//1188;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= 691*tre//360360; sim -= 691*tim//360360;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += tre//156; sim += tim//156;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim
    sre -= 3617*tre//122400; sim -= 3617*tim//122400;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre += 43867*tre//244188; sim += 43867*tim//244188;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    sre -= 174611*tre//125400; sim -= 174611*tim//125400;
    tre, tim = ((tre*ure-tim*uim)>>prec), ((tre*uim+tim*ure)>>prec)
    if abs(tre) + abs(tim) < 5: return sre, sim

    k = 22
    # From here on, the coefficients are growing, so we
    # have to keep t at a roughly constant size
    usize = bitcount(max(abs(ure), abs(uim)))
    tsize = bitcount(max(abs(tre), abs(tim)))
    texp = 0
    while 1:
        p, q, pb, qb = stirling_coefficient(k)
        term_mag = tsize + pb + texp
        shift = -texp
        m = pb - term_mag
        if m > 0 and shift < m:
            p >>= m
            shift -= m
        m = tsize - term_mag
        if m > 0 and shift < m:
            wre = tre >> m
            wim = tim >> m
            shift -= m
        else:
            wre = tre
            wim = tim
        termre = (tre*p//q) >> shift
        termim = (tim*p//q) >> shift
        if abs(termre) + abs(termim) < 5:
            break
        sre += termre
        sim += termim
        tre, tim = ((tre*ure - tim*uim)>>usize), \
            ((tre*uim + tim*ure)>>usize)
        texp -= (prec - usize)
        k += 2
    return sre, sim


def mpf_gamma(x, prec, rnd='d', type=0):
    """
    This function implements multipurpose evaluation of the gamma
    function, G(x), as well as the following versions of the same:

    type = 0 -- G(x)                    [standard gamma function]
    type = 1 -- G(x+1) = x*G(x+1) = x!  [factorial]
    type = 2 -- 1/G(x)                  [reciprocal gamma function]
    type = 3 -- log(|G(x)|)             [log-gamma function, real part]
    """

    # Specal values
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            if type == 1: return fone
            if type == 2: return fzero
            raise ValueError("gamma function pole")
        if x == finf:
            if type == 2: return fzero
            return finf
        return fnan

    # First of all, for log gamma, numbers can be well beyond the fixed-point
    # range, so we must take care of huge numbers before e.g. trying
    # to convert x to the nearest integer
    if type == 3:
        wp = prec+20
        if exp+bc > wp and not sign:
            return mpf_sub(mpf_mul(x, mpf_log(x, wp), wp), x, prec, rnd)

    # We strongly want to special-case small integers
    is_integer = exp >= 0
    if is_integer:
        # Poles
        if sign:
            if type == 2:
                return fzero
            raise ValueError("gamma function pole")
        # n = x
        n = man << exp
        if n < SMALL_FACTORIAL_CACHE_SIZE:
            if type == 0:
                return mpf_pos(small_factorial_cache[n-1], prec, rnd)
            if type == 1:
                return mpf_pos(small_factorial_cache[n], prec, rnd)
            if type == 2:
                return mpf_div(fone, small_factorial_cache[n-1], prec, rnd)
            if type == 3:
                return mpf_log(small_factorial_cache[n-1], prec, rnd)
    else:
        # floor(abs(x))
        n = int(man >> (-exp))

    # Estimate size and precision
    # Estimate log(gamma(|x|),2) as x*log(x,2)
    mag = exp + bc
    gamma_size = n*mag

    if type == 3:
        wp = prec + 20
    else:
        wp = prec + bitcount(gamma_size) + 20

    # Very close to 0, pole
    if mag < -wp:
        if type == 0:
            return mpf_sub(mpf_div(fone,x, wp),mpf_shift(fone,-wp),prec,rnd)
        if type == 1: return mpf_sub(fone, x, prec, rnd)
        if type == 2: return mpf_add(x, mpf_shift(fone,mag-wp), prec, rnd)
        if type == 3: return mpf_neg(mpf_log(mpf_abs(x), prec, rnd))

    # From now on, we assume having a gamma function
    if type == 1:
        return mpf_gamma(mpf_add(x, fone), prec, rnd, 0)

    # Special case integers (those not small enough to be caught above,
    # but still small enough for an exact factorial to be faster
    # than an approximate algorithm), and half-integers
    if exp >= -1:
        if is_integer:
            if gamma_size < 10*wp:
                if type == 0:
                    return from_int(ifac(n-1), prec, rnd)
                if type == 2:
                    return from_rational(MPZ_ONE, ifac(n-1), prec, rnd)
                if type == 3:
                    return mpf_log(from_int(ifac(n-1)), prec, rnd)
        # half-integer
        if n < 100 or gamma_size < 10*wp:
            if sign:
                w = sqrtpi_fixed(wp)
                if n % 2: f = ifac2(2*n+1)
                else:     f = -ifac2(2*n+1)
                if type == 0:
                    return mpf_shift(from_rational(w, f, prec, rnd), -wp+n+1)
                if type == 2:
                    return mpf_shift(from_rational(f, w, prec, rnd), wp-n-1)
                if type == 3:
                    return mpf_log(mpf_shift(from_rational(w, abs(f),
                        prec, rnd), -wp+n+1), prec, rnd)
            elif n == 0:
                if type == 0: return mpf_sqrtpi(prec, rnd)
                if type == 2: return mpf_div(fone, mpf_sqrtpi(wp), prec, rnd)
                if type == 3: return mpf_log(mpf_sqrtpi(wp), prec, rnd)
            else:
                w = sqrtpi_fixed(wp)
                w = from_man_exp(w * ifac2(2*n-1), -wp-n)
                if type == 0: return mpf_pos(w, prec, rnd)
                if type == 2: return mpf_div(fone, w, prec, rnd)
                if type == 3: return mpf_log(mpf_abs(w), prec, rnd)

    # Convert to fixed point
    offset = exp + wp
    if offset >= 0: absxman = man << offset
    else:           absxman = man >> (-offset)

    # For log gamma, provide accurate evaluation for x = 1+eps and 2+eps
    if type == 3 and not sign:
        one = MPZ_ONE << wp
        one_dist = abs(absxman-one)
        two_dist = abs(absxman-2*one)
        cancellation = (wp - bitcount(min(one_dist, two_dist)))
        if cancellation > 10:
            xsub1 = mpf_sub(fone, x)
            xsub2 = mpf_sub(ftwo, x)
            xsub1mag = xsub1[2]+xsub1[3]
            xsub2mag = xsub2[2]+xsub2[3]
            if xsub1mag < -wp:
                return mpf_mul(mpf_euler(wp), mpf_sub(fone, x), prec, rnd)
            if xsub2mag < -wp:
                return mpf_mul(mpf_sub(fone, mpf_euler(wp)),
                    mpf_sub(x, ftwo), prec, rnd)
            # Proceed but increase precision
            wp += max(-xsub1mag, -xsub2mag)
            offset = exp + wp
            if offset >= 0: absxman = man << offset
            else:           absxman = man >> (-offset)

    # Use Taylor series if appropriate
    n_for_stirling = int(GAMMA_STIRLING_BETA*wp)
    if n < max(100, n_for_stirling) and wp < MAX_GAMMA_TAYLOR_PREC:
        if sign:
            absxman = -absxman
        return gamma_fixed_taylor(x, absxman, wp, prec, rnd, type)

    # Use Stirling's series
    # First ensure that |x| is large enough for rapid convergence
    xorig = x

    # Argument reduction
    r = 0
    if n < n_for_stirling:
        r = one = MPZ_ONE << wp
        d = n_for_stirling - n
        for k in xrange(d):
            r = (r * absxman) >> wp
            absxman += one
        x = xabs = from_man_exp(absxman, -wp)
        if sign:
            x = mpf_neg(x)
    else:
        xabs = mpf_abs(x)

    # Asymptotic series
    y = real_stirling_series(absxman, wp)
    u = to_fixed(mpf_log(xabs, wp), wp)
    u = ((absxman - (MPZ_ONE<<(wp-1))) * u) >> wp
    y += u
    w = from_man_exp(y, -wp)

    # Compute final value
    if sign:
        # Reflection formula
        A = mpf_mul(mpf_sin_pi(xorig, wp), xorig, wp)
        B = mpf_neg(mpf_pi(wp))
        if type == 0 or type == 2:
            A = mpf_mul(A, mpf_exp(w, wp))
            if r:
                B = mpf_mul(B, from_man_exp(r, -wp), wp)
            if type == 0:
                return mpf_div(B, A, prec, rnd)
            if type == 2:
                return mpf_div(A, B, prec, rnd)
        if type == 3:
            if r:
                B = mpf_mul(B, from_man_exp(r, -wp), wp)
            A = mpf_add(mpf_log(mpf_abs(A), wp), w, wp)
            return mpf_sub(mpf_log(mpf_abs(B), wp), A, prec, rnd)
    else:
        if type == 0:
            if r:
                return mpf_div(mpf_exp(w, wp),
                    from_man_exp(r, -wp), prec, rnd)
            return mpf_exp(w, prec, rnd)
        if type == 2:
            if r:
                return mpf_div(from_man_exp(r, -wp),
                    mpf_exp(w, wp), prec, rnd)
            return mpf_exp(mpf_neg(w), prec, rnd)
        if type == 3:
            if r:
                return mpf_sub(w, mpf_log(from_man_exp(r,-wp), wp), prec, rnd)
            return mpf_pos(w, prec, rnd)


def mpc_gamma(z, prec, rnd='d', type=0):
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b

    if b == fzero:
        # Imaginary part on negative half-axis for log-gamma function
        if type == 3 and asign:
            re = mpf_gamma(a, prec, rnd, 3)
            n = (-aman) >> (-aexp)
            im = mpf_mul_int(mpf_pi(prec+10), n, prec, rnd)
            return re, im
        return mpf_gamma(a, prec, rnd, type), fzero

    # Some kind of complex inf/nan
    if (not aman and aexp) or (not bman and bexp):
        return (fnan, fnan)

    # Initial working precision
    wp = prec + 20

    amag = aexp+abc
    bmag = bexp+bbc
    if aman:
        mag = max(amag, bmag)
    else:
        mag = bmag

    # Close to 0
    if mag < -8:
        if mag < -wp:
            # 1/gamma(z) = z + euler*z^2 + O(z^3)
            v = mpc_add(z, mpc_mul_mpf(mpc_mul(z,z,wp),mpf_euler(wp),wp), wp)
            if type == 0: return mpc_reciprocal(v, prec, rnd)
            if type == 1: return mpc_div(z, v, prec, rnd)
            if type == 2: return mpc_pos(v, prec, rnd)
            if type == 3: return mpc_log(mpc_reciprocal(v, prec), prec, rnd)
        elif type != 1:
            wp += (-mag)

    # Handle huge log-gamma values; must do this before converting to
    # a fixed-point value. TODO: determine a precise cutoff of validity
    # depending on amag and bmag
    if type == 3 and mag > wp and ((not asign) or (bmag >= amag)):
        return mpc_sub(mpc_mul(z, mpc_log(z, wp), wp), z, prec, rnd)

    # From now on, we assume having a gamma function
    if type == 1:
        return mpc_gamma((mpf_add(a, fone), b), prec, rnd, 0)

    an = abs(to_int(a))
    bn = abs(to_int(b))
    absn = max(an, bn)
    gamma_size = absn*mag
    if type == 3:
        pass
    else:
        wp += bitcount(gamma_size)

    # Reflect to the right half-plane. Note that Stirling's expansion
    # is valid in the left half-plane too, as long as we're not too close
    # to the real axis, but in order to use this argument reduction
    # in the negative direction must be implemented.
    #need_reflection = asign and ((bmag < 0) or (amag-bmag > 4))
    need_reflection = asign
    zorig = z
    if need_reflection:
        z = mpc_neg(z)
        asign, aman, aexp, abc = a = z[0]
        bsign, bman, bexp, bbc = b = z[1]

    # Imaginary part very small compared to real one?
    yfinal = 0
    balance_prec = 0
    if bmag < -10:
        # Check z ~= 1 and z ~= 2 for loggamma
        if type == 3:
            zsub1 = mpc_sub_mpf(z, fone)
            if zsub1[0] == fzero:
                cancel1 = -bmag
            else:
                cancel1 = -max(zsub1[0][2]+zsub1[0][3], bmag)
            if cancel1 > wp:
                pi = mpf_pi(wp)
                x = mpc_mul_mpf(zsub1, pi, wp)
                x = mpc_mul(x, x, wp)
                x = mpc_div_mpf(x, from_int(12), wp)
                y = mpc_mul_mpf(zsub1, mpf_neg(mpf_euler(wp)), wp)
                yfinal = mpc_add(x, y, wp)
                if not need_reflection:
                    return mpc_pos(yfinal, prec, rnd)
            elif cancel1 > 0:
                wp += cancel1
            zsub2 = mpc_sub_mpf(z, ftwo)
            if zsub2[0] == fzero:
                cancel2 = -bmag
            else:
                cancel2 = -max(zsub2[0][2]+zsub2[0][3], bmag)
            if cancel2 > wp:
                pi = mpf_pi(wp)
                t = mpf_sub(mpf_mul(pi, pi), from_int(6))
                x = mpc_mul_mpf(mpc_mul(zsub2, zsub2, wp), t, wp)
                x = mpc_div_mpf(x, from_int(12), wp)
                y = mpc_mul_mpf(zsub2, mpf_sub(fone, mpf_euler(wp)), wp)
                yfinal = mpc_add(x, y, wp)
                if not need_reflection:
                    return mpc_pos(yfinal, prec, rnd)
            elif cancel2 > 0:
                wp += cancel2
        if bmag < -wp:
            # Compute directly from the real gamma function.
            pp = 2*(wp+10)
            aabs = mpf_abs(a)
            eps = mpf_shift(fone, amag-wp)
            x1 = mpf_gamma(aabs, pp, type=type)
            x2 = mpf_gamma(mpf_add(aabs, eps), pp, type=type)
            xprime = mpf_div(mpf_sub(x2, x1, pp), eps, pp)
            y = mpf_mul(b, xprime, prec, rnd)
            yfinal = (x1, y)
            # Note: we still need to use the reflection formula for
            # near-poles, and the correct branch of the log-gamma function
            if not need_reflection:
                return mpc_pos(yfinal, prec, rnd)
        else:
            balance_prec += (-bmag)

    wp += balance_prec
    n_for_stirling = int(GAMMA_STIRLING_BETA*wp)
    need_reduction = absn < n_for_stirling

    afix = to_fixed(a, wp)
    bfix = to_fixed(b, wp)

    r = 0
    if not yfinal:
        zprered = z
        # Argument reduction
        if absn < n_for_stirling:
            absn = complex(an, bn)
            d = int((1 + n_for_stirling**2 - bn**2)**0.5 - an)
            rre = one = MPZ_ONE << wp
            rim = MPZ_ZERO
            for k in xrange(d):
                rre, rim = ((afix*rre-bfix*rim)>>wp), ((afix*rim + bfix*rre)>>wp)
                afix += one
            r = from_man_exp(rre, -wp), from_man_exp(rim, -wp)
            a = from_man_exp(afix, -wp)
            z = a, b

        yre, yim = complex_stirling_series(afix, bfix, wp)
        # (z-1/2)*log(z) + S
        lre, lim = mpc_log(z, wp)
        lre = to_fixed(lre, wp)
        lim = to_fixed(lim, wp)
        yre = ((lre*afix - lim*bfix)>>wp) - (lre>>1) + yre
        yim = ((lre*bfix + lim*afix)>>wp) - (lim>>1) + yim
        y = from_man_exp(yre, -wp), from_man_exp(yim, -wp)

        if r and type == 3:
            # If re(z) > 0 and abs(z) <= 4, the branches of loggamma(z)
            # and log(gamma(z)) coincide. Otherwise, use the zeroth order
            # Stirling expansion to compute the correct imaginary part.
            y = mpc_sub(y, mpc_log(r, wp), wp)
            zfa = to_float(zprered[0])
            zfb = to_float(zprered[1])
            zfabs = math.hypot(zfa,zfb)
            #if not (zfa > 0.0 and zfabs <= 4):
            yfb = to_float(y[1])
            u = math.atan2(zfb, zfa)
            if zfabs <= 0.5:
                gi = 0.577216*zfb - u
            else:
                gi = -zfb - 0.5*u + zfa*u + zfb*math.log(zfabs)
            n = int(math.floor((gi-yfb)/(2*math.pi)+0.5))
            y = (y[0], mpf_add(y[1], mpf_mul_int(mpf_pi(wp), 2*n, wp), wp))

    if need_reflection:
        if type == 0 or type == 2:
            A = mpc_mul(mpc_sin_pi(zorig, wp), zorig, wp)
            B = (mpf_neg(mpf_pi(wp)), fzero)
            if yfinal:
                if type == 2:
                    A = mpc_div(A, yfinal, wp)
                else:
                    A = mpc_mul(A, yfinal, wp)
            else:
                A = mpc_mul(A, mpc_exp(y, wp), wp)
            if r:
                B = mpc_mul(B, r, wp)
            if type == 0: return mpc_div(B, A, prec, rnd)
            if type == 2: return mpc_div(A, B, prec, rnd)

        # Reflection formula for the log-gamma function with correct branch
        # http://functions.wolfram.com/GammaBetaErf/LogGamma/16/01/01/0006/
        # LogGamma[z] == -LogGamma[-z] - Log[-z] +
        # Sign[Im[z]] Floor[Re[z]] Pi I + Log[Pi] -
        #      Log[Sin[Pi (z - Floor[Re[z]])]] -
        # Pi I (1 - Abs[Sign[Im[z]]]) Abs[Floor[Re[z]]]
        if type == 3:
            if yfinal:
                s1 = mpc_neg(yfinal)
            else:
                s1 = mpc_neg(y)
            # s -= log(-z)
            s1 = mpc_sub(s1, mpc_log(mpc_neg(zorig), wp), wp)
            # floor(re(z))
            rezfloor = mpf_floor(zorig[0])
            imzsign = mpf_sign(zorig[1])
            pi = mpf_pi(wp)
            t = mpf_mul(pi, rezfloor)
            t = mpf_mul_int(t, imzsign, wp)
            s1 = (s1[0], mpf_add(s1[1], t, wp))
            s1 = mpc_add_mpf(s1, mpf_log(pi, wp), wp)
            t = mpc_sin_pi(mpc_sub_mpf(zorig, rezfloor), wp)
            t = mpc_log(t, wp)
            s1 = mpc_sub(s1, t, wp)
            # Note: may actually be unused, because we fall back
            # to the mpf_ function for real arguments
            if not imzsign:
                t = mpf_mul(pi, mpf_floor(rezfloor), wp)
                s1 = (s1[0], mpf_sub(s1[1], t, wp))
            return mpc_pos(s1, prec, rnd)
    else:
        if type == 0:
            if r:
                return mpc_div(mpc_exp(y, wp), r, prec, rnd)
            return mpc_exp(y, prec, rnd)
        if type == 2:
            if r:
                return mpc_div(r, mpc_exp(y, wp), prec, rnd)
            return mpc_exp(mpc_neg(y), prec, rnd)
        if type == 3:
            return mpc_pos(y, prec, rnd)

def mpf_factorial(x, prec, rnd='d'):
    return mpf_gamma(x, prec, rnd, 1)

def mpc_factorial(x, prec, rnd='d'):
    return mpc_gamma(x, prec, rnd, 1)

def mpf_rgamma(x, prec, rnd='d'):
    return mpf_gamma(x, prec, rnd, 2)

def mpc_rgamma(x, prec, rnd='d'):
    return mpc_gamma(x, prec, rnd, 2)

def mpf_loggamma(x, prec, rnd='d'):
    sign, man, exp, bc = x
    if sign:
        raise ComplexResult
    return mpf_gamma(x, prec, rnd, 3)

def mpc_loggamma(z, prec, rnd='d'):
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero and asign:
        re = mpf_gamma(a, prec, rnd, 3)
        n = (-aman) >> (-aexp)
        im = mpf_mul_int(mpf_pi(prec+10), n, prec, rnd)
        return re, im
    return mpc_gamma(z, prec, rnd, 3)

def mpf_gamma_int(n, prec, rnd=round_fast):
    if n < SMALL_FACTORIAL_CACHE_SIZE:
        return mpf_pos(small_factorial_cache[n-1], prec, rnd)
    return mpf_gamma(from_int(n), prec, rnd)