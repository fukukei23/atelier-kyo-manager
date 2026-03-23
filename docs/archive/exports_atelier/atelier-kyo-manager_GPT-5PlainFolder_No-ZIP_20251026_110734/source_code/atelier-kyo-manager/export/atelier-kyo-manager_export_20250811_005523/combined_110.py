
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\body.py ===
from sympy import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, Inertia
from sympy.physics.mechanics.body_base import BodyBase
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Body']


# XXX: We use type:ignore because the classes RigidBody and Particle have
# inconsistent parallel axis methods that take different numbers of arguments.
class Body(RigidBody, Particle):  # type: ignore
    """
    Body is a common representation of either a RigidBody or a Particle SymPy
    object depending on what is passed in during initialization. If a mass is
    passed in and central_inertia is left as None, the Particle object is
    created. Otherwise a RigidBody object will be created.

    .. deprecated:: 1.13
        The Body class is deprecated. Its functionality is captured by
        :class:`~.RigidBody` and :class:`~.Particle`.

    Explanation
    ===========

    The attributes that Body possesses will be the same as a Particle instance
    or a Rigid Body instance depending on which was created. Additional
    attributes are listed below.

    Attributes
    ==========

    name : string
        The body's name
    masscenter : Point
        The point which represents the center of mass of the rigid body
    frame : ReferenceFrame
        The reference frame which the body is fixed in
    mass : Sympifyable
        The body's mass
    inertia : (Dyadic, Point)
        The body's inertia around its center of mass. This attribute is specific
        to the rigid body form of Body and is left undefined for the Particle
        form
    loads : iterable
        This list contains information on the different loads acting on the
        Body. Forces are listed as a (point, vector) tuple and torques are
        listed as (reference frame, vector) tuples.

    Parameters
    ==========

    name : String
        Defines the name of the body. It is used as the base for defining
        body specific properties.
    masscenter : Point, optional
        A point that represents the center of mass of the body or particle.
        If no point is given, a point is generated.
    mass : Sympifyable, optional
        A Sympifyable object which represents the mass of the body. If no
        mass is passed, one is generated.
    frame : ReferenceFrame, optional
        The ReferenceFrame that represents the reference frame of the body.
        If no frame is given, a frame is generated.
    central_inertia : Dyadic, optional
        Central inertia dyadic of the body. If none is passed while creating
        RigidBody, a default inertia is generated.

    Examples
    ========

    As Body has been deprecated, the following examples are for illustrative
    purposes only. The functionality of Body is fully captured by
    :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
    warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings

    Default behaviour. This results in the creation of a RigidBody object for
    which the mass, mass center, frame and inertia attributes are given default
    values. ::

        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     body = Body('name_of_body')

    This next example demonstrates the code required to specify all of the
    values of the Body object. Note this will also create a RigidBody version of
    the Body object. ::

        >>> from sympy import Symbol
        >>> from sympy.physics.mechanics import ReferenceFrame, Point, inertia
        >>> from sympy.physics.mechanics import Body
        >>> mass = Symbol('mass')
        >>> masscenter = Point('masscenter')
        >>> frame = ReferenceFrame('frame')
        >>> ixx = Symbol('ixx')
        >>> body_inertia = inertia(frame, ixx, 0, 0)
        >>> with ignore_warnings(DeprecationWarning):
        ...     body = Body('name_of_body', masscenter, mass, frame, body_inertia)

    The minimal code required to create a Particle version of the Body object
    involves simply passing in a name and a mass. ::

        >>> from sympy import Symbol
        >>> from sympy.physics.mechanics import Body
        >>> mass = Symbol('mass')
        >>> with ignore_warnings(DeprecationWarning):
        ...     body = Body('name_of_body', mass=mass)

    The Particle version of the Body object can also receive a masscenter point
    and a reference frame, just not an inertia.
    """

    def __init__(self, name, masscenter=None, mass=None, frame=None,
                 central_inertia=None):
        sympy_deprecation_warning(
            """
            Support for the Body class has been removed, as its functionality is
            fully captured by RigidBody and Particle.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-mechanics-body-class"
        )

        self._loads = []

        if frame is None:
            frame = ReferenceFrame(name + '_frame')

        if masscenter is None:
            masscenter = Point(name + '_masscenter')

        if central_inertia is None and mass is None:
            ixx = Symbol(name + '_ixx')
            iyy = Symbol(name + '_iyy')
            izz = Symbol(name + '_izz')
            izx = Symbol(name + '_izx')
            ixy = Symbol(name + '_ixy')
            iyz = Symbol(name + '_iyz')
            _inertia = Inertia.from_inertia_scalars(masscenter, frame, ixx, iyy,
                                                    izz, ixy, iyz, izx)
        else:
            _inertia = (central_inertia, masscenter)

        if mass is None:
            _mass = Symbol(name + '_mass')
        else:
            _mass = mass

        masscenter.set_vel(frame, 0)

        # If user passes masscenter and mass then a particle is created
        # otherwise a rigidbody. As a result a body may or may not have inertia.
        # Note: BodyBase.__init__ is used to prevent problems with super() calls in
        # Particle and RigidBody arising due to multiple inheritance.
        if central_inertia is None and mass is not None:
            BodyBase.__init__(self, name, masscenter, _mass)
            self.frame = frame
            self._central_inertia = Dyadic(0)
        else:
            BodyBase.__init__(self, name, masscenter, _mass)
            self.frame = frame
            self.inertia = _inertia

    def __repr__(self):
        if self.is_rigidbody:
            return RigidBody.__repr__(self)
        return Particle.__repr__(self)

    @property
    def loads(self):
        return self._loads

    @property
    def x(self):
        """The basis Vector for the Body, in the x direction."""
        return self.frame.x

    @property
    def y(self):
        """The basis Vector for the Body, in the y direction."""
        return self.frame.y

    @property
    def z(self):
        """The basis Vector for the Body, in the z direction."""
        return self.frame.z

    @property
    def inertia(self):
        """The body's inertia about a point; stored as (Dyadic, Point)."""
        if self.is_rigidbody:
            return RigidBody.inertia.fget(self)
        return (self.central_inertia, self.masscenter)

    @inertia.setter
    def inertia(self, I):
        RigidBody.inertia.fset(self, I)

    @property
    def is_rigidbody(self):
        if hasattr(self, '_inertia'):
            return True
        return False

    def kinetic_energy(self, frame):
        """Kinetic energy of the body.

        Parameters
        ==========

        frame : ReferenceFrame or Body
            The Body's angular velocity and the velocity of it's mass
            center are typically defined with respect to an inertial frame but
            any relevant frame in which the velocities are known can be supplied.

        Examples
        ========

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, ReferenceFrame, Point
        >>> from sympy import symbols
        >>> m, v, r, omega = symbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> with ignore_warnings(DeprecationWarning):
        ...     P = Body('P', masscenter=O, mass=m)
        >>> P.masscenter.set_vel(N, v * N.y)
        >>> P.kinetic_energy(N)
        m*v**2/2

        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B', masscenter=P, frame=b)
        >>> B.kinetic_energy(N)
        B_ixx*omega**2/2 + B_mass*v**2/2

        See Also
        ========

        sympy.physics.mechanics : Particle, RigidBody

        """
        if isinstance(frame, Body):
            frame = Body.frame
        if self.is_rigidbody:
            return RigidBody(self.name, self.masscenter, self.frame, self.mass,
                            (self.central_inertia, self.masscenter)).kinetic_energy(frame)
        return Particle(self.name, self.masscenter, self.mass).kinetic_energy(frame)

    def apply_force(self, force, point=None, reaction_body=None, reaction_point=None):
        """Add force to the body(s).

        Explanation
        ===========

        Applies the force on self or equal and opposite forces on
        self and other body if both are given on the desired point on the bodies.
        The force applied on other body is taken opposite of self, i.e, -force.

        Parameters
        ==========

        force: Vector
            The force to be applied.
        point: Point, optional
            The point on self on which force is applied.
            By default self's masscenter.
        reaction_body: Body, optional
            Second body on which equal and opposite force
            is to be applied.
        reaction_point : Point, optional
            The point on other body on which equal and opposite
            force is applied. By default masscenter of other body.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import Body, Point, dynamicsymbols
        >>> m, g = symbols('m g')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> force1 = m*g*B.z
        >>> B.apply_force(force1) #Applying force on B's masscenter
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z)]

        We can also remove some part of force from any point on the body by
        adding the opposite force to the body on that point.

        >>> f1, f2 = dynamicsymbols('f1 f2')
        >>> P = Point('P') #Considering point P on body B
        >>> B.apply_force(f1*B.x + f2*B.y, P)
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z), (P, f1(t)*B_frame.x + f2(t)*B_frame.y)]

        Let's remove f1 from point P on body B.

        >>> B.apply_force(-f1*B.x, P)
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z), (P, f2(t)*B_frame.y)]

        To further demonstrate the use of ``apply_force`` attribute,
        consider two bodies connected through a spring.

        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> with ignore_warnings(DeprecationWarning):
        ...     N = Body('N') #Newtonion Frame
        >>> x = dynamicsymbols('x')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B1 = Body('B1')
        ...     B2 = Body('B2')
        >>> spring_force = x*N.x

        Now let's apply equal and opposite spring force to the bodies.

        >>> P1 = Point('P1')
        >>> P2 = Point('P2')
        >>> B1.apply_force(spring_force, point=P1, reaction_body=B2, reaction_point=P2)

        We can check the loads(forces) applied to bodies now.

        >>> B1.loads
        [(P1, x(t)*N_frame.x)]
        >>> B2.loads
        [(P2, - x(t)*N_frame.x)]

        Notes
        =====

        If a new force is applied to a body on a point which already has some
        force applied on it, then the new force is added to the already applied
        force on that point.

        """

        if not isinstance(point, Point):
            if point is None:
                point = self.masscenter  # masscenter
            else:
                raise TypeError("Force must be applied to a point on the body.")
        if not isinstance(force, Vector):
            raise TypeError("Force must be a vector.")

        if reaction_body is not None:
            reaction_body.apply_force(-force, point=reaction_point)

        for load in self._loads:
            if point in load:
                force += load[1]
                self._loads.remove(load)
                break

        self._loads.append((point, force))

    def apply_torque(self, torque, reaction_body=None):
        """Add torque to the body(s).

        Explanation
        ===========

        Applies the torque on self or equal and opposite torques on
        self and other body if both are given.
        The torque applied on other body is taken opposite of self,
        i.e, -torque.

        Parameters
        ==========

        torque: Vector
            The torque to be applied.
        reaction_body: Body, optional
            Second body on which equal and opposite torque
            is to be applied.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> t = symbols('t')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> torque1 = t*B.z
        >>> B.apply_torque(torque1)
        >>> B.loads
        [(B_frame, t*B_frame.z)]

        We can also remove some part of torque from the body by
        adding the opposite torque to the body.

        >>> t1, t2 = dynamicsymbols('t1 t2')
        >>> B.apply_torque(t1*B.x + t2*B.y)
        >>> B.loads
        [(B_frame, t1(t)*B_frame.x + t2(t)*B_frame.y + t*B_frame.z)]

        Let's remove t1 from Body B.

        >>> B.apply_torque(-t1*B.x)
        >>> B.loads
        [(B_frame, t2(t)*B_frame.y + t*B_frame.z)]

        To further demonstrate the use, let us consider two bodies such that
        a torque `T` is acting on one body, and `-T` on the other.

        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> with ignore_warnings(DeprecationWarning):
        ...     N = Body('N') #Newtonion frame
        ...     B1 = Body('B1')
        ...     B2 = Body('B2')
        >>> v = dynamicsymbols('v')
        >>> T = v*N.y #Torque

        Now let's apply equal and opposite torque to the bodies.

        >>> B1.apply_torque(T, B2)

        We can check the loads (torques) applied to bodies now.

        >>> B1.loads
        [(B1_frame, v(t)*N_frame.y)]
        >>> B2.loads
        [(B2_frame, - v(t)*N_frame.y)]

        Notes
        =====

        If a new torque is applied on body which already has some torque applied on it,
        then the new torque is added to the previous torque about the body's frame.

        """

        if not isinstance(torque, Vector):
            raise TypeError("A Vector must be supplied to add torque.")

        if reaction_body is not None:
            reaction_body.apply_torque(-torque)

        for load in self._loads:
            if self.frame in load:
                torque += load[1]
                self._loads.remove(load)
                break
        self._loads.append((self.frame, torque))

    def clear_loads(self):
        """
        Clears the Body's loads list.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> force = B.x + B.y
        >>> B.apply_force(force)
        >>> B.loads
        [(B_masscenter, B_frame.x + B_frame.y)]
        >>> B.clear_loads()
        >>> B.loads
        []

        """

        self._loads = []

    def remove_load(self, about=None):
        """
        Remove load about a point or frame.

        Parameters
        ==========

        about : Point or ReferenceFrame, optional
            The point about which force is applied,
            and is to be removed.
            If about is None, then the torque about
            self's frame is removed.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, Point
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> P = Point('P')
        >>> f1 = B.x
        >>> f2 = B.y
        >>> B.apply_force(f1)
        >>> B.apply_force(f2, P)
        >>> B.loads
        [(B_masscenter, B_frame.x), (P, B_frame.y)]

        >>> B.remove_load(P)
        >>> B.loads
        [(B_masscenter, B_frame.x)]

        """

        if about is not None:
            if not isinstance(about, Point):
                raise TypeError('Load is applied about Point or ReferenceFrame.')
        else:
            about = self.frame

        for load in self._loads:
            if about in load:
                self._loads.remove(load)
                break

    def masscenter_vel(self, body):
        """
        Returns the velocity of the mass center with respect to the provided
        rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the velocity in.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        ...     B = Body('B')
        >>> A.masscenter.set_vel(B.frame, 5*B.frame.x)
        >>> A.masscenter_vel(B)
        5*B_frame.x
        >>> A.masscenter_vel(B.frame)
        5*B_frame.x

        """

        if isinstance(body,  ReferenceFrame):
            frame=body
        elif isinstance(body, Body):
            frame = body.frame
        return self.masscenter.vel(frame)

    def ang_vel_in(self, body):
        """
        Returns this body's angular velocity with respect to the provided
        rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the angular velocity in.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, ReferenceFrame
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        >>> N = ReferenceFrame('N')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B', frame=N)
        >>> A.frame.set_ang_vel(N, 5*N.x)
        >>> A.ang_vel_in(B)
        5*N.x
        >>> A.ang_vel_in(N)
        5*N.x

        """

        if isinstance(body,  ReferenceFrame):
            frame=body
        elif isinstance(body, Body):
            frame = body.frame
        return self.frame.ang_vel_in(frame)

    def dcm(self, body):
        """
        Returns the direction cosine matrix of this body relative to the
        provided rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the dcm.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        ...     B = Body('B')
        >>> A.frame.orient_axis(B.frame, B.frame.x, 5)
        >>> A.dcm(B)
        Matrix([
        [1,       0,      0],
        [0,  cos(5), sin(5)],
        [0, -sin(5), cos(5)]])
        >>> A.dcm(B.frame)
        Matrix([
        [1,       0,      0],
        [0,  cos(5), sin(5)],
        [0, -sin(5), cos(5)]])

        """

        if isinstance(body,  ReferenceFrame):
            frame=body
        elif isinstance(body, Body):
            frame = body.frame
        return self.frame.dcm(frame)

    def parallel_axis(self, point, frame=None):
        """Returns the inertia dyadic of the body with respect to another
        point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        >>> P = A.masscenter.locatenew('point', 3 * A.x + 5 * A.y)
        >>> A.parallel_axis(P).to_matrix(A.frame)
        Matrix([
        [A_ixx + 25*A_mass, A_ixy - 15*A_mass,             A_izx],
        [A_ixy - 15*A_mass,  A_iyy + 9*A_mass,             A_iyz],
        [            A_izx,             A_iyz, A_izz + 34*A_mass]])

        """
        if self.is_rigidbody:
            return RigidBody.parallel_axis(self, point, frame)
        return Particle.parallel_axis(self, point, frame)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\contrib\emscripten\fetch.py ===
"""
Support for streaming http requests in emscripten.

A few caveats -

If your browser (or Node.js) has WebAssembly JavaScript Promise Integration enabled
https://github.com/WebAssembly/js-promise-integration/blob/main/proposals/js-promise-integration/Overview.md
*and* you launch pyodide using `pyodide.runPythonAsync`, this will fetch data using the
JavaScript asynchronous fetch api (wrapped via `pyodide.ffi.call_sync`). In this case
timeouts and streaming should just work.

Otherwise, it uses a combination of XMLHttpRequest and a web-worker for streaming.

This approach has several caveats:

Firstly, you can't do streaming http in the main UI thread, because atomics.wait isn't allowed.
Streaming only works if you're running pyodide in a web worker.

Secondly, this uses an extra web worker and SharedArrayBuffer to do the asynchronous fetch
operation, so it requires that you have crossOriginIsolation enabled, by serving over https
(or from localhost) with the two headers below set:

    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp

You can tell if cross origin isolation is successfully enabled by looking at the global crossOriginIsolated variable in
JavaScript console. If it isn't, streaming requests will fallback to XMLHttpRequest, i.e. getting the whole
request into a buffer and then returning it. it shows a warning in the JavaScript console in this case.

Finally, the webworker which does the streaming fetch is created on initial import, but will only be started once
control is returned to javascript. Call `await wait_for_streaming_ready()` to wait for streaming fetch.

NB: in this code, there are a lot of JavaScript objects. They are named js_*
to make it clear what type of object they are.
"""

from __future__ import annotations

import io
import json
from email.parser import Parser
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import js  # type: ignore[import-not-found]
from pyodide.ffi import (  # type: ignore[import-not-found]
    JsArray,
    JsException,
    JsProxy,
    to_js,
)

if TYPE_CHECKING:
    from typing_extensions import Buffer

from .request import EmscriptenRequest
from .response import EmscriptenResponse

"""
There are some headers that trigger unintended CORS preflight requests.
See also https://github.com/koenvo/pyodide-http/issues/22
"""
HEADERS_TO_IGNORE = ("user-agent",)

SUCCESS_HEADER = -1
SUCCESS_EOF = -2
ERROR_TIMEOUT = -3
ERROR_EXCEPTION = -4

_STREAMING_WORKER_CODE = (
    files(__package__)
    .joinpath("emscripten_fetch_worker.js")
    .read_text(encoding="utf-8")
)


class _RequestError(Exception):
    def __init__(
        self,
        message: str | None = None,
        *,
        request: EmscriptenRequest | None = None,
        response: EmscriptenResponse | None = None,
    ):
        self.request = request
        self.response = response
        self.message = message
        super().__init__(self.message)


class _StreamingError(_RequestError):
    pass


class _TimeoutError(_RequestError):
    pass


def _obj_from_dict(dict_val: dict[str, Any]) -> JsProxy:
    return to_js(dict_val, dict_converter=js.Object.fromEntries)


class _ReadStream(io.RawIOBase):
    def __init__(
        self,
        int_buffer: JsArray,
        byte_buffer: JsArray,
        timeout: float,
        worker: JsProxy,
        connection_id: int,
        request: EmscriptenRequest,
    ):
        self.int_buffer = int_buffer
        self.byte_buffer = byte_buffer
        self.read_pos = 0
        self.read_len = 0
        self.connection_id = connection_id
        self.worker = worker
        self.timeout = int(1000 * timeout) if timeout > 0 else None
        self.is_live = True
        self._is_closed = False
        self.request: EmscriptenRequest | None = request

    def __del__(self) -> None:
        self.close()

    # this is compatible with _base_connection
    def is_closed(self) -> bool:
        return self._is_closed

    # for compatibility with RawIOBase
    @property
    def closed(self) -> bool:
        return self.is_closed()

    def close(self) -> None:
        if self.is_closed():
            return
        self.read_len = 0
        self.read_pos = 0
        self.int_buffer = None
        self.byte_buffer = None
        self._is_closed = True
        self.request = None
        if self.is_live:
            self.worker.postMessage(_obj_from_dict({"close": self.connection_id}))
            self.is_live = False
        super().close()

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def readinto(self, byte_obj: Buffer) -> int:
        if not self.int_buffer:
            raise _StreamingError(
                "No buffer for stream in _ReadStream.readinto",
                request=self.request,
                response=None,
            )
        if self.read_len == 0:
            # wait for the worker to send something
            js.Atomics.store(self.int_buffer, 0, ERROR_TIMEOUT)
            self.worker.postMessage(_obj_from_dict({"getMore": self.connection_id}))
            if (
                js.Atomics.wait(self.int_buffer, 0, ERROR_TIMEOUT, self.timeout)
                == "timed-out"
            ):
                raise _TimeoutError
            data_len = self.int_buffer[0]
            if data_len > 0:
                self.read_len = data_len
                self.read_pos = 0
            elif data_len == ERROR_EXCEPTION:
                string_len = self.int_buffer[1]
                # decode the error string
                js_decoder = js.TextDecoder.new()
                json_str = js_decoder.decode(self.byte_buffer.slice(0, string_len))
                raise _StreamingError(
                    f"Exception thrown in fetch: {json_str}",
                    request=self.request,
                    response=None,
                )
            else:
                # EOF, free the buffers and return zero
                # and free the request
                self.is_live = False
                self.close()
                return 0
        # copy from int32array to python bytes
        ret_length = min(self.read_len, len(memoryview(byte_obj)))
        subarray = self.byte_buffer.subarray(
            self.read_pos, self.read_pos + ret_length
        ).to_py()
        memoryview(byte_obj)[0:ret_length] = subarray
        self.read_len -= ret_length
        self.read_pos += ret_length
        return ret_length


class _StreamingFetcher:
    def __init__(self) -> None:
        # make web-worker and data buffer on startup
        self.streaming_ready = False

        js_data_blob = js.Blob.new(
            to_js([_STREAMING_WORKER_CODE], create_pyproxies=False),
            _obj_from_dict({"type": "application/javascript"}),
        )

        def promise_resolver(js_resolve_fn: JsProxy, js_reject_fn: JsProxy) -> None:
            def onMsg(e: JsProxy) -> None:
                self.streaming_ready = True
                js_resolve_fn(e)

            def onErr(e: JsProxy) -> None:
                js_reject_fn(e)  # Defensive: never happens in ci

            self.js_worker.onmessage = onMsg
            self.js_worker.onerror = onErr

        js_data_url = js.URL.createObjectURL(js_data_blob)
        self.js_worker = js.globalThis.Worker.new(js_data_url)
        self.js_worker_ready_promise = js.globalThis.Promise.new(promise_resolver)

    def send(self, request: EmscriptenRequest) -> EmscriptenResponse:
        headers = {
            k: v for k, v in request.headers.items() if k not in HEADERS_TO_IGNORE
        }

        body = request.body
        fetch_data = {"headers": headers, "body": to_js(body), "method": request.method}
        # start the request off in the worker
        timeout = int(1000 * request.timeout) if request.timeout > 0 else None
        js_shared_buffer = js.SharedArrayBuffer.new(1048576)
        js_int_buffer = js.Int32Array.new(js_shared_buffer)
        js_byte_buffer = js.Uint8Array.new(js_shared_buffer, 8)

        js.Atomics.store(js_int_buffer, 0, ERROR_TIMEOUT)
        js.Atomics.notify(js_int_buffer, 0)
        js_absolute_url = js.URL.new(request.url, js.location).href
        self.js_worker.postMessage(
            _obj_from_dict(
                {
                    "buffer": js_shared_buffer,
                    "url": js_absolute_url,
                    "fetchParams": fetch_data,
                }
            )
        )
        # wait for the worker to send something
        js.Atomics.wait(js_int_buffer, 0, ERROR_TIMEOUT, timeout)
        if js_int_buffer[0] == ERROR_TIMEOUT:
            raise _TimeoutError(
                "Timeout connecting to streaming request",
                request=request,
                response=None,
            )
        elif js_int_buffer[0] == SUCCESS_HEADER:
            # got response
            # header length is in second int of intBuffer
            string_len = js_int_buffer[1]
            # decode the rest to a JSON string
            js_decoder = js.TextDecoder.new()
            # this does a copy (the slice) because decode can't work on shared array
            # for some silly reason
            json_str = js_decoder.decode(js_byte_buffer.slice(0, string_len))
            # get it as an object
            response_obj = json.loads(json_str)
            return EmscriptenResponse(
                request=request,
                status_code=response_obj["status"],
                headers=response_obj["headers"],
                body=_ReadStream(
                    js_int_buffer,
                    js_byte_buffer,
                    request.timeout,
                    self.js_worker,
                    response_obj["connectionID"],
                    request,
                ),
            )
        elif js_int_buffer[0] == ERROR_EXCEPTION:
            string_len = js_int_buffer[1]
            # decode the error string
            js_decoder = js.TextDecoder.new()
            json_str = js_decoder.decode(js_byte_buffer.slice(0, string_len))
            raise _StreamingError(
                f"Exception thrown in fetch: {json_str}", request=request, response=None
            )
        else:
            raise _StreamingError(
                f"Unknown status from worker in fetch: {js_int_buffer[0]}",
                request=request,
                response=None,
            )


class _JSPIReadStream(io.RawIOBase):
    """
    A read stream that uses pyodide.ffi.run_sync to read from a JavaScript fetch
    response. This requires support for WebAssembly JavaScript Promise Integration
    in the containing browser, and for pyodide to be launched via runPythonAsync.

    :param js_read_stream:
        The JavaScript stream reader

    :param timeout:
        Timeout in seconds

    :param request:
        The request we're handling

    :param response:
        The response this stream relates to

    :param js_abort_controller:
        A JavaScript AbortController object, used for timeouts
    """

    def __init__(
        self,
        js_read_stream: Any,
        timeout: float,
        request: EmscriptenRequest,
        response: EmscriptenResponse,
        js_abort_controller: Any,  # JavaScript AbortController for timeouts
    ):
        self.js_read_stream = js_read_stream
        self.timeout = timeout
        self._is_closed = False
        self._is_done = False
        self.request: EmscriptenRequest | None = request
        self.response: EmscriptenResponse | None = response
        self.current_buffer = None
        self.current_buffer_pos = 0
        self.js_abort_controller = js_abort_controller

    def __del__(self) -> None:
        self.close()

    # this is compatible with _base_connection
    def is_closed(self) -> bool:
        return self._is_closed

    # for compatibility with RawIOBase
    @property
    def closed(self) -> bool:
        return self.is_closed()

    def close(self) -> None:
        if self.is_closed():
            return
        self.read_len = 0
        self.read_pos = 0
        self.js_read_stream.cancel()
        self.js_read_stream = None
        self._is_closed = True
        self._is_done = True
        self.request = None
        self.response = None
        super().close()

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def _get_next_buffer(self) -> bool:
        result_js = _run_sync_with_timeout(
            self.js_read_stream.read(),
            self.timeout,
            self.js_abort_controller,
            request=self.request,
            response=self.response,
        )
        if result_js.done:
            self._is_done = True
            return False
        else:
            self.current_buffer = result_js.value.to_py()
            self.current_buffer_pos = 0
            return True

    def readinto(self, byte_obj: Buffer) -> int:
        if self.current_buffer is None:
            if not self._get_next_buffer() or self.current_buffer is None:
                self.close()
                return 0
        ret_length = min(
            len(byte_obj), len(self.current_buffer) - self.current_buffer_pos
        )
        byte_obj[0:ret_length] = self.current_buffer[
            self.current_buffer_pos : self.current_buffer_pos + ret_length
        ]
        self.current_buffer_pos += ret_length
        if self.current_buffer_pos == len(self.current_buffer):
            self.current_buffer = None
        return ret_length


# check if we are in a worker or not
def is_in_browser_main_thread() -> bool:
    return hasattr(js, "window") and hasattr(js, "self") and js.self == js.window


def is_cross_origin_isolated() -> bool:
    return hasattr(js, "crossOriginIsolated") and js.crossOriginIsolated


def is_in_node() -> bool:
    return (
        hasattr(js, "process")
        and hasattr(js.process, "release")
        and hasattr(js.process.release, "name")
        and js.process.release.name == "node"
    )


def is_worker_available() -> bool:
    return hasattr(js, "Worker") and hasattr(js, "Blob")


_fetcher: _StreamingFetcher | None = None

if is_worker_available() and (
    (is_cross_origin_isolated() and not is_in_browser_main_thread())
    and (not is_in_node())
):
    _fetcher = _StreamingFetcher()
else:
    _fetcher = None


NODE_JSPI_ERROR = (
    "urllib3 only works in Node.js with pyodide.runPythonAsync"
    " and requires the flag --experimental-wasm-stack-switching in "
    " versions of node <24."
)


def send_streaming_request(request: EmscriptenRequest) -> EmscriptenResponse | None:
    if has_jspi():
        return send_jspi_request(request, True)
    elif is_in_node():
        raise _RequestError(
            message=NODE_JSPI_ERROR,
            request=request,
            response=None,
        )

    if _fetcher and streaming_ready():
        return _fetcher.send(request)
    else:
        _show_streaming_warning()
        return None


_SHOWN_TIMEOUT_WARNING = False


def _show_timeout_warning() -> None:
    global _SHOWN_TIMEOUT_WARNING
    if not _SHOWN_TIMEOUT_WARNING:
        _SHOWN_TIMEOUT_WARNING = True
        message = "Warning: Timeout is not available on main browser thread"
        js.console.warn(message)


_SHOWN_STREAMING_WARNING = False


def _show_streaming_warning() -> None:
    global _SHOWN_STREAMING_WARNING
    if not _SHOWN_STREAMING_WARNING:
        _SHOWN_STREAMING_WARNING = True
        message = "Can't stream HTTP requests because: \n"
        if not is_cross_origin_isolated():
            message += "  Page is not cross-origin isolated\n"
        if is_in_browser_main_thread():
            message += "  Python is running in main browser thread\n"
        if not is_worker_available():
            message += " Worker or Blob classes are not available in this environment."  # Defensive: this is always False in browsers that we test in
        if streaming_ready() is False:
            message += """ Streaming fetch worker isn't ready. If you want to be sure that streaming fetch
is working, you need to call: 'await urllib3.contrib.emscripten.fetch.wait_for_streaming_ready()`"""
        from js import console

        console.warn(message)


def send_request(request: EmscriptenRequest) -> EmscriptenResponse:
    if has_jspi():
        return send_jspi_request(request, False)
    elif is_in_node():
        raise _RequestError(
            message=NODE_JSPI_ERROR,
            request=request,
            response=None,
        )
    try:
        js_xhr = js.XMLHttpRequest.new()

        if not is_in_browser_main_thread():
            js_xhr.responseType = "arraybuffer"
            if request.timeout:
                js_xhr.timeout = int(request.timeout * 1000)
        else:
            js_xhr.overrideMimeType("text/plain; charset=ISO-8859-15")
            if request.timeout:
                # timeout isn't available on the main thread - show a warning in console
                # if it is set
                _show_timeout_warning()

        js_xhr.open(request.method, request.url, False)
        for name, value in request.headers.items():
            if name.lower() not in HEADERS_TO_IGNORE:
                js_xhr.setRequestHeader(name, value)

        js_xhr.send(to_js(request.body))

        headers = dict(Parser().parsestr(js_xhr.getAllResponseHeaders()))

        if not is_in_browser_main_thread():
            body = js_xhr.response.to_py().tobytes()
        else:
            body = js_xhr.response.encode("ISO-8859-15")
        return EmscriptenResponse(
            status_code=js_xhr.status, headers=headers, body=body, request=request
        )
    except JsException as err:
        if err.name == "TimeoutError":
            raise _TimeoutError(err.message, request=request)
        elif err.name == "NetworkError":
            raise _RequestError(err.message, request=request)
        else:
            # general http error
            raise _RequestError(err.message, request=request)


def send_jspi_request(
    request: EmscriptenRequest, streaming: bool
) -> EmscriptenResponse:
    """
    Send a request using WebAssembly JavaScript Promise Integration
    to wrap the asynchronous JavaScript fetch api (experimental).

    :param request:
        Request to send

    :param streaming:
        Whether to stream the response

    :return: The response object
    :rtype: EmscriptenResponse
    """
    timeout = request.timeout
    js_abort_controller = js.AbortController.new()
    headers = {k: v for k, v in request.headers.items() if k not in HEADERS_TO_IGNORE}
    req_body = request.body
    fetch_data = {
        "headers": headers,
        "body": to_js(req_body),
        "method": request.method,
        "signal": js_abort_controller.signal,
    }
    # Call JavaScript fetch (async api, returns a promise)
    fetcher_promise_js = js.fetch(request.url, _obj_from_dict(fetch_data))
    # Now suspend WebAssembly until we resolve that promise
    # or time out.
    response_js = _run_sync_with_timeout(
        fetcher_promise_js,
        timeout,
        js_abort_controller,
        request=request,
        response=None,
    )
    headers = {}
    header_iter = response_js.headers.entries()
    while True:
        iter_value_js = header_iter.next()
        if getattr(iter_value_js, "done", False):
            break
        else:
            headers[str(iter_value_js.value[0])] = str(iter_value_js.value[1])
    status_code = response_js.status
    body: bytes | io.RawIOBase = b""

    response = EmscriptenResponse(
        status_code=status_code, headers=headers, body=b"", request=request
    )
    if streaming:
        # get via inputstream
        if response_js.body is not None:
            # get a reader from the fetch response
            body_stream_js = response_js.body.getReader()
            body = _JSPIReadStream(
                body_stream_js, timeout, request, response, js_abort_controller
            )
    else:
        # get directly via arraybuffer
        # n.b. this is another async JavaScript call.
        body = _run_sync_with_timeout(
            response_js.arrayBuffer(),
            timeout,
            js_abort_controller,
            request=request,
            response=response,
        ).to_py()
    response.body = body
    return response


def _run_sync_with_timeout(
    promise: Any,
    timeout: float,
    js_abort_controller: Any,
    request: EmscriptenRequest | None,
    response: EmscriptenResponse | None,
) -> Any:
    """
    Await a JavaScript promise synchronously with a timeout which is implemented
    via the AbortController

    :param promise:
        Javascript promise to await

    :param timeout:
        Timeout in seconds

    :param js_abort_controller:
        A JavaScript AbortController object, used on timeout

    :param request:
        The request being handled

    :param response:
        The response being handled (if it exists yet)

    :raises _TimeoutError: If the request times out
    :raises _RequestError: If the request raises a JavaScript exception

    :return: The result of awaiting the promise.
    """
    timer_id = None
    if timeout > 0:
        timer_id = js.setTimeout(
            js_abort_controller.abort.bind(js_abort_controller), int(timeout * 1000)
        )
    try:
        from pyodide.ffi import run_sync

        # run_sync here uses WebAssembly JavaScript Promise Integration to
        # suspend python until the JavaScript promise resolves.
        return run_sync(promise)
    except JsException as err:
        if err.name == "AbortError":
            raise _TimeoutError(
                message="Request timed out", request=request, response=response
            )
        else:
            raise _RequestError(message=err.message, request=request, response=response)
    finally:
        if timer_id is not None:
            js.clearTimeout(timer_id)


def has_jspi() -> bool:
    """
    Return true if jspi can be used.

    This requires both browser support and also WebAssembly
    to be in the correct state - i.e. that the javascript
    call into python was async not sync.

    :return: True if jspi can be used.
    :rtype: bool
    """
    try:
        from pyodide.ffi import can_run_sync, run_sync  # noqa: F401

        return bool(can_run_sync())
    except ImportError:
        return False


def streaming_ready() -> bool | None:
    if _fetcher:
        return _fetcher.streaming_ready
    else:
        return None  # no fetcher, return None to signify that


async def wait_for_streaming_ready() -> bool:
    if _fetcher:
        await _fetcher.js_worker_ready_promise
        return True
    else:
        return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\gaussiandomains.py ===
"""Domains of Gaussian type."""

from __future__ import annotations
from sympy.core.numbers import I
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring


class GaussianElement(DomainElement):
    """Base class for elements of Gaussian type domains."""
    base: Domain
    _parent: Domain

    __slots__ = ('x', 'y')

    def __new__(cls, x, y=0):
        conv = cls.base.convert
        return cls.new(conv(x), conv(y))

    @classmethod
    def new(cls, x, y):
        """Create a new GaussianElement of the same domain."""
        obj = super().__new__(cls)
        obj.x = x
        obj.y = y
        return obj

    def parent(self):
        """The domain that this is an element of (ZZ_I or QQ_I)"""
        return self._parent

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        else:
            return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, GaussianElement):
            return NotImplemented
        return [self.y, self.x] < [other.y, other.x]

    def __pos__(self):
        return self

    def __neg__(self):
        return self.new(-self.x, -self.y)

    def __repr__(self):
        return "%s(%s, %s)" % (self._parent.rep, self.x, self.y)

    def __str__(self):
        return str(self._parent.to_sympy(self))

    @classmethod
    def _get_xy(cls, other):
        if not isinstance(other, cls):
            try:
                other = cls._parent.convert(other)
            except CoercionFailed:
                return None, None
        return other.x, other.y

    def __add__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x + x, self.y + y)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x - x, self.y - y)
        else:
            return NotImplemented

    def __rsub__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(x - self.x, y - self.y)
        else:
            return NotImplemented

    def __mul__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x*x - self.y*y, self.x*y + self.y*x)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __pow__(self, exp):
        if exp == 0:
            return self.new(1, 0)
        if exp < 0:
            self, exp = 1/self, -exp
        if exp == 1:
            return self
        pow2 = self
        prod = self if exp % 2 else self._parent.one
        exp //= 2
        while exp:
            pow2 *= pow2
            if exp % 2:
                prod *= pow2
            exp //= 2
        return prod

    def __bool__(self):
        return bool(self.x) or bool(self.y)

    def quadrant(self):
        """Return quadrant index 0-3.

        0 is included in quadrant 0.
        """
        if self.y > 0:
            return 0 if self.x > 0 else 1
        elif self.y < 0:
            return 2 if self.x < 0 else 3
        else:
            return 0 if self.x >= 0 else 2

    def __rdivmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented
        else:
            return other.__divmod__(self)

    def __rtruediv__(self, other):
        try:
            other = QQ_I.convert(other)
        except CoercionFailed:
            return NotImplemented
        else:
            return other.__truediv__(self)

    def __floordiv__(self, other):
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[0]

    def __rfloordiv__(self, other):
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[0]

    def __mod__(self, other):
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[1]

    def __rmod__(self, other):
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[1]


class GaussianInteger(GaussianElement):
    """Gaussian integer: domain element for :ref:`ZZ_I`

        >>> from sympy import ZZ_I
        >>> z = ZZ_I(2, 3)
        >>> z
        (2 + 3*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianInteger'>
    """
    base = ZZ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        return QQ_I.convert(self)/other

    def __divmod__(self, other):
        if not other:
            raise ZeroDivisionError('divmod({}, 0)'.format(self))
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented

        # multiply self and other by x - I*y
        # self/other == (a + I*b)/c
        a, b = self.x*x + self.y*y, -self.x*y + self.y*x
        c = x*x + y*y

        # find integers qx and qy such that
        # |a - qx*c| <= c/2 and |b - qy*c| <= c/2
        qx = (2*a + c) // (2*c)  # -c <= 2*a - qx*2*c < c
        qy = (2*b + c) // (2*c)

        q = GaussianInteger(qx, qy)
        # |self/other - q| < 1 since
        # |a/c - qx|**2 + |b/c - qy|**2 <= 1/4 + 1/4 < 1

        return q, self - q*other  # |r| < |other|


class GaussianRational(GaussianElement):
    """Gaussian rational: domain element for :ref:`QQ_I`

        >>> from sympy import QQ_I, QQ
        >>> z = QQ_I(QQ(2, 3), QQ(4, 5))
        >>> z
        (2/3 + 4/5*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianRational'>
    """
    base = QQ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        if not other:
            raise ZeroDivisionError('{} / 0'.format(self))
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented
        c = x*x + y*y

        return GaussianRational((self.x*x + self.y*y)/c,
                                (-self.x*y + self.y*x)/c)

    def __divmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented
        if not other:
            raise ZeroDivisionError('{} % 0'.format(self))
        else:
            return self/other, QQ_I.zero


class GaussianDomain():
    """Base class for Gaussian domains."""
    dom: Domain

    is_Numerical = True
    is_Exact = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        conv = self.dom.to_sympy
        return conv(a.x) + I*conv(a.y)

    def from_sympy(self, a):
        """Convert a SymPy object to ``self.dtype``."""
        r, b = a.as_coeff_Add()
        x = self.dom.from_sympy(r)  # may raise CoercionFailed
        if not b:
            return self.new(x, 0)
        r, b = b.as_coeff_Mul()
        y = self.dom.from_sympy(r)
        if b is I:
            return self.new(x, y)
        else:
            raise CoercionFailed("{} is not Gaussian".format(a))

    def inject(self, *gens):
        """Inject generators into this domain. """
        return self.poly_ring(*gens)

    def canonical_unit(self, d):
        unit = self.units[-d.quadrant()]  # - for inverse power
        return unit

    def is_negative(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_positive(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_nonnegative(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_nonpositive(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY mpz to ``self.dtype``."""
        return K1(a)

    def from_ZZ(K1, a, K0):
        """Convert a ZZ_python element to ``self.dtype``."""
        return K1(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a ZZ_python element to ``self.dtype``."""
        return K1(a)

    def from_QQ(K1, a, K0):
        """Convert a GMPY mpq to ``self.dtype``."""
        return K1(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY mpq to ``self.dtype``."""
        return K1(a)

    def from_QQ_python(K1, a, K0):
        """Convert a QQ_python element to ``self.dtype``."""
        return K1(a)

    def from_AlgebraicField(K1, a, K0):
        """Convert an element from ZZ<I> or QQ<I> to ``self.dtype``."""
        if K0.ext.args[0] == I:
            return K1.from_sympy(K0.to_sympy(a))


class GaussianIntegerRing(GaussianDomain, Ring):
    r"""Ring of Gaussian integers ``ZZ_I``

    The :ref:`ZZ_I` domain represents the `Gaussian integers`_ `\mathbb{Z}[i]`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of integers and ``I`` (`\sqrt{-1}`)
    will have the domain :ref:`ZZ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I)
    >>> p
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> p.domain
    ZZ_I

    The :ref:`ZZ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian integers.

    >>> from sympy import factor
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, domain='ZZ_I')
    (x - I)*(x + I)

    The corresponding `field of fractions`_ is the domain of the Gaussian
    rationals :ref:`QQ_I`. Conversely :ref:`ZZ_I` is the `ring of integers`_
    of :ref:`QQ_I`.

    >>> from sympy import ZZ_I, QQ_I
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`ZZ_I` can be used as a constructor.

    >>> ZZ_I(3, 4)
    (3 + 4*I)
    >>> ZZ_I(5)
    (5 + 0*I)

    The domain elements of :ref:`ZZ_I` are instances of
    :py:class:`~.GaussianInteger` which support the rings operations
    ``+,-,*,**``.

    >>> z1 = ZZ_I(5, 1)
    >>> z2 = ZZ_I(2, 3)
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 3*I)
    >>> z1 + z2
    (7 + 4*I)
    >>> z1 * z2
    (7 + 17*I)
    >>> z1 ** 2
    (24 + 10*I)

    Both floor (``//``) and modulo (``%``) division work with
    :py:class:`~.GaussianInteger` (see the :py:meth:`~.Domain.div` method).

    >>> z3, z4 = ZZ_I(5), ZZ_I(1, 3)
    >>> z3 // z4  # floor division
    (1 + -1*I)
    >>> z3 % z4   # modulo division (remainder)
    (1 + -2*I)
    >>> (z3//z4)*z4 + z3%z4 == z3
    True

    True division (``/``) in :ref:`ZZ_I` gives an element of :ref:`QQ_I`. The
    :py:meth:`~.Domain.exquo` method can be used to divide in :ref:`ZZ_I` when
    exact division is possible.

    >>> z1 / z2
    (1 + -1*I)
    >>> ZZ_I.exquo(z1, z2)
    (1 + -1*I)
    >>> z3 / z4
    (1/2 + -3/2*I)
    >>> ZZ_I.exquo(z3, z4)
    Traceback (most recent call last):
        ...
    ExactQuotientFailed: (1 + 3*I) does not divide (5 + 0*I) in ZZ_I

    The :py:meth:`~.Domain.gcd` method can be used to compute the `gcd`_ of any
    two elements.

    >>> ZZ_I.gcd(ZZ_I(10), ZZ_I(2))
    (2 + 0*I)
    >>> ZZ_I.gcd(ZZ_I(5), ZZ_I(2, 1))
    (2 + 1*I)

    .. _Gaussian integers: https://en.wikipedia.org/wiki/Gaussian_integer
    .. _gcd: https://en.wikipedia.org/wiki/Greatest_common_divisor

    """
    dom = ZZ
    mod = DMP([ZZ.one, ZZ.zero, ZZ.one], ZZ)
    dtype = GaussianInteger
    zero = dtype(ZZ(0), ZZ(0))
    one = dtype(ZZ(1), ZZ(0))
    imag_unit = dtype(ZZ(0), ZZ(1))
    units = (one, imag_unit, -one, -imag_unit)  # powers of i

    rep = 'ZZ_I'

    is_GaussianRing = True
    is_ZZ_I = True
    is_PID = True

    def __init__(self):  # override Domain.__init__
        """For constructing ZZ_I."""

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, GaussianIntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute hash code of ``self``. """
        return hash('ZZ_I')

    @property
    def has_CharacteristicZero(self):
        return True

    def characteristic(self):
        return 0

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def get_field(self):
        """Returns a field associated with ``self``. """
        return QQ_I

    def normalize(self, d, *args):
        """Return first quadrant element associated with ``d``.

        Also multiply the other arguments by the same power of i.
        """
        unit = self.canonical_unit(d)
        d *= unit
        args = tuple(a*unit for a in args)
        return (d,) + args if args else d

    def gcd(self, a, b):
        """Greatest common divisor of a and b over ZZ_I."""
        while b:
            a, b = b, a % b
        return self.normalize(a)

    def gcdex(self, a, b):
        """Return x, y, g such that x * a + y * b = g = gcd(a, b)"""
        x_a = self.one
        x_b = self.zero
        y_a = self.zero
        y_b = self.one
        while b:
            q = a // b
            a, b = b, a - q * b
            x_a, x_b = x_b, x_a - q * x_b
            y_a, y_b = y_b, y_a - q * y_b

        a, x_a, y_a = self.normalize(a, x_a, y_a)
        return x_a, y_a, a

    def lcm(self, a, b):
        """Least common multiple of a and b over ZZ_I."""
        return (a * b) // self.gcd(a, b)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to ZZ_I."""
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to ZZ_I."""
        return K1.new(ZZ.convert(a.x), ZZ.convert(a.y))

ZZ_I = GaussianInteger._parent = GaussianIntegerRing()


class GaussianRationalField(GaussianDomain, Field):
    r"""Field of Gaussian rationals ``QQ_I``

    The :ref:`QQ_I` domain represents the `Gaussian rationals`_ `\mathbb{Q}(i)`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of rationals and ``I`` (`\sqrt{-1}`)
    will have the domain :ref:`QQ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I/2)
    >>> p
    Poly(x**2 + I/2, x, domain='QQ_I')
    >>> p.domain
    QQ_I

    The polys option ``gaussian=True`` can be used to specify that the domain
    should be :ref:`QQ_I` even if the coefficients do not contain ``I`` or are
    all integers.

    >>> Poly(x**2)
    Poly(x**2, x, domain='ZZ')
    >>> Poly(x**2 + I)
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> Poly(x**2/2)
    Poly(1/2*x**2, x, domain='QQ')
    >>> Poly(x**2, gaussian=True)
    Poly(x**2, x, domain='QQ_I')
    >>> Poly(x**2 + I, gaussian=True)
    Poly(x**2 + I, x, domain='QQ_I')
    >>> Poly(x**2/2, gaussian=True)
    Poly(1/2*x**2, x, domain='QQ_I')

    The :ref:`QQ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian rationals.

    >>> from sympy import factor, QQ_I
    >>> factor(x**2/4 + 1)
    (x**2 + 4)/4
    >>> factor(x**2/4 + 1, domain='QQ_I')
    (x - 2*I)*(x + 2*I)/4
    >>> factor(x**2/4 + 1, domain=QQ_I)
    (x - 2*I)*(x + 2*I)/4

    It is also possible to specify the :ref:`QQ_I` domain explicitly with
    polys functions like :py:func:`~.apart`.

    >>> from sympy import apart
    >>> apart(1/(1 + x**2))
    1/(x**2 + 1)
    >>> apart(1/(1 + x**2), domain=QQ_I)
    I/(2*(x + I)) - I/(2*(x - I))

    The corresponding `ring of integers`_ is the domain of the Gaussian
    integers :ref:`ZZ_I`. Conversely :ref:`QQ_I` is the `field of fractions`_
    of :ref:`ZZ_I`.

    >>> from sympy import ZZ_I, QQ_I, QQ
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`QQ_I` can be used as a constructor.

    >>> QQ_I(3, 4)
    (3 + 4*I)
    >>> QQ_I(5)
    (5 + 0*I)
    >>> QQ_I(QQ(2, 3), QQ(4, 5))
    (2/3 + 4/5*I)

    The domain elements of :ref:`QQ_I` are instances of
    :py:class:`~.GaussianRational` which support the field operations
    ``+,-,*,**,/``.

    >>> z1 = QQ_I(5, 1)
    >>> z2 = QQ_I(2, QQ(1, 2))
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 1/2*I)
    >>> z1 + z2
    (7 + 3/2*I)
    >>> z1 * z2
    (19/2 + 9/2*I)
    >>> z2 ** 2
    (15/4 + 2*I)

    True division (``/``) in :ref:`QQ_I` gives an element of :ref:`QQ_I` and
    is always exact.

    >>> z1 / z2
    (42/17 + -2/17*I)
    >>> QQ_I.exquo(z1, z2)
    (42/17 + -2/17*I)
    >>> z1 == (z1/z2)*z2
    True

    Both floor (``//``) and modulo (``%``) division can be used with
    :py:class:`~.GaussianRational` (see :py:meth:`~.Domain.div`)
    but division is always exact so there is no remainder.

    >>> z1 // z2
    (42/17 + -2/17*I)
    >>> z1 % z2
    (0 + 0*I)
    >>> QQ_I.div(z1, z2)
    ((42/17 + -2/17*I), (0 + 0*I))
    >>> (z1//z2)*z2 + z1%z2 == z1
    True

    .. _Gaussian rationals: https://en.wikipedia.org/wiki/Gaussian_rational
    """
    dom = QQ
    mod = DMP([QQ.one, QQ.zero, QQ.one], QQ)
    dtype = GaussianRational
    zero = dtype(QQ(0), QQ(0))
    one = dtype(QQ(1), QQ(0))
    imag_unit = dtype(QQ(0), QQ(1))
    units = (one, imag_unit, -one, -imag_unit)  # powers of i

    rep = 'QQ_I'

    is_GaussianField = True
    is_QQ_I = True

    def __init__(self):  # override Domain.__init__
        """For constructing QQ_I."""

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, GaussianRationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute hash code of ``self``. """
        return hash('QQ_I')

    @property
    def has_CharacteristicZero(self):
        return True

    def characteristic(self):
        return 0

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return ZZ_I

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def as_AlgebraicField(self):
        """Get equivalent domain as an ``AlgebraicField``. """
        return AlgebraicField(self.dom, I)

    def numer(self, a):
        """Get the numerator of ``a``."""
        ZZ_I = self.get_ring()
        return ZZ_I.convert(a * self.denom(a))

    def denom(self, a):
        """Get the denominator of ``a``."""
        ZZ = self.dom.get_ring()
        QQ = self.dom
        ZZ_I = self.get_ring()
        denom_ZZ = ZZ.lcm(QQ.denom(a.x), QQ.denom(a.y))
        return ZZ_I(denom_ZZ, ZZ.zero)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to QQ_I."""
        return K1.new(a.x, a.y)

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to QQ_I."""
        return a

    def from_ComplexField(K1, a, K0):
        """Convert a ComplexField element to QQ_I."""
        return K1.new(QQ.convert(a.real), QQ.convert(a.imag))


QQ_I = GaussianRational._parent = GaussianRationalField()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\sqlite\pysqlite.py ===
# dialects/sqlite/pysqlite.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


r"""
.. dialect:: sqlite+pysqlite
    :name: pysqlite
    :dbapi: sqlite3
    :connectstring: sqlite+pysqlite:///file_path
    :url: https://docs.python.org/library/sqlite3.html

    Note that ``pysqlite`` is the same driver as the ``sqlite3``
    module included with the Python distribution.

Driver
------

The ``sqlite3`` Python DBAPI is standard on all modern Python versions;
for cPython and Pypy, no additional installation is necessary.


Connect Strings
---------------

The file specification for the SQLite database is taken as the "database"
portion of the URL.  Note that the format of a SQLAlchemy url is:

.. sourcecode:: text

    driver://user:pass@host/database

This means that the actual filename to be used starts with the characters to
the **right** of the third slash.   So connecting to a relative filepath
looks like::

    # relative path
    e = create_engine("sqlite:///path/to/database.db")

An absolute path, which is denoted by starting with a slash, means you
need **four** slashes::

    # absolute path
    e = create_engine("sqlite:////path/to/database.db")

To use a Windows path, regular drive specifications and backslashes can be
used. Double backslashes are probably needed::

    # absolute path on Windows
    e = create_engine("sqlite:///C:\\path\\to\\database.db")

To use sqlite ``:memory:`` database specify it as the filename using
``sqlite:///:memory:``. It's also the default if no filepath is
present, specifying only ``sqlite://`` and nothing else::

    # in-memory database (note three slashes)
    e = create_engine("sqlite:///:memory:")
    # also in-memory database
    e2 = create_engine("sqlite://")

.. _pysqlite_uri_connections:

URI Connections
^^^^^^^^^^^^^^^

Modern versions of SQLite support an alternative system of connecting using a
`driver level URI <https://www.sqlite.org/uri.html>`_, which has the  advantage
that additional driver-level arguments can be passed including options such as
"read only".   The Python sqlite3 driver supports this mode under modern Python
3 versions.   The SQLAlchemy pysqlite driver supports this mode of use by
specifying "uri=true" in the URL query string.  The SQLite-level "URI" is kept
as the "database" portion of the SQLAlchemy url (that is, following a slash)::

    e = create_engine("sqlite:///file:path/to/database?mode=ro&uri=true")

.. note::  The "uri=true" parameter must appear in the **query string**
   of the URL.  It will not currently work as expected if it is only
   present in the :paramref:`_sa.create_engine.connect_args`
   parameter dictionary.

The logic reconciles the simultaneous presence of SQLAlchemy's query string and
SQLite's query string by separating out the parameters that belong to the
Python sqlite3 driver vs. those that belong to the SQLite URI.  This is
achieved through the use of a fixed list of parameters known to be accepted by
the Python side of the driver.  For example, to include a URL that indicates
the Python sqlite3 "timeout" and "check_same_thread" parameters, along with the
SQLite "mode" and "nolock" parameters, they can all be passed together on the
query string::

    e = create_engine(
        "sqlite:///file:path/to/database?"
        "check_same_thread=true&timeout=10&mode=ro&nolock=1&uri=true"
    )

Above, the pysqlite / sqlite3 DBAPI would be passed arguments as::

    sqlite3.connect(
        "file:path/to/database?mode=ro&nolock=1",
        check_same_thread=True,
        timeout=10,
        uri=True,
    )

Regarding future parameters added to either the Python or native drivers. new
parameter names added to the SQLite URI scheme should be automatically
accommodated by this scheme.  New parameter names added to the Python driver
side can be accommodated by specifying them in the
:paramref:`_sa.create_engine.connect_args` dictionary,
until dialect support is
added by SQLAlchemy.   For the less likely case that the native SQLite driver
adds a new parameter name that overlaps with one of the existing, known Python
driver parameters (such as "timeout" perhaps), SQLAlchemy's dialect would
require adjustment for the URL scheme to continue to support this.

As is always the case for all SQLAlchemy dialects, the entire "URL" process
can be bypassed in :func:`_sa.create_engine` through the use of the
:paramref:`_sa.create_engine.creator`
parameter which allows for a custom callable
that creates a Python sqlite3 driver level connection directly.

.. versionadded:: 1.3.9

.. seealso::

    `Uniform Resource Identifiers <https://www.sqlite.org/uri.html>`_ - in
    the SQLite documentation

.. _pysqlite_regexp:

Regular Expression Support
---------------------------

.. versionadded:: 1.4

Support for the :meth:`_sql.ColumnOperators.regexp_match` operator is provided
using Python's re.search_ function.  SQLite itself does not include a working
regular expression operator; instead, it includes a non-implemented placeholder
operator ``REGEXP`` that calls a user-defined function that must be provided.

SQLAlchemy's implementation makes use of the pysqlite create_function_ hook
as follows::


    def regexp(a, b):
        return re.search(a, b) is not None


    sqlite_connection.create_function(
        "regexp",
        2,
        regexp,
    )

There is currently no support for regular expression flags as a separate
argument, as these are not supported by SQLite's REGEXP operator, however these
may be included inline within the regular expression string.  See `Python regular expressions`_ for
details.

.. seealso::

    `Python regular expressions`_: Documentation for Python's regular expression syntax.

.. _create_function: https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.create_function

.. _re.search: https://docs.python.org/3/library/re.html#re.search

.. _Python regular expressions: https://docs.python.org/3/library/re.html#re.search



Compatibility with sqlite3 "native" date and datetime types
-----------------------------------------------------------

The pysqlite driver includes the sqlite3.PARSE_DECLTYPES and
sqlite3.PARSE_COLNAMES options, which have the effect of any column
or expression explicitly cast as "date" or "timestamp" will be converted
to a Python date or datetime object.  The date and datetime types provided
with the pysqlite dialect are not currently compatible with these options,
since they render the ISO date/datetime including microseconds, which
pysqlite's driver does not.   Additionally, SQLAlchemy does not at
this time automatically render the "cast" syntax required for the
freestanding functions "current_timestamp" and "current_date" to return
datetime/date types natively.   Unfortunately, pysqlite
does not provide the standard DBAPI types in ``cursor.description``,
leaving SQLAlchemy with no way to detect these types on the fly
without expensive per-row type checks.

Keeping in mind that pysqlite's parsing option is not recommended,
nor should be necessary, for use with SQLAlchemy, usage of PARSE_DECLTYPES
can be forced if one configures "native_datetime=True" on create_engine()::

    engine = create_engine(
        "sqlite://",
        connect_args={
            "detect_types": sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        },
        native_datetime=True,
    )

With this flag enabled, the DATE and TIMESTAMP types (but note - not the
DATETIME or TIME types...confused yet ?) will not perform any bind parameter
or result processing. Execution of "func.current_date()" will return a string.
"func.current_timestamp()" is registered as returning a DATETIME type in
SQLAlchemy, so this function still receives SQLAlchemy-level result
processing.

.. _pysqlite_threading_pooling:

Threading/Pooling Behavior
---------------------------

The ``sqlite3`` DBAPI by default prohibits the use of a particular connection
in a thread which is not the one in which it was created.  As SQLite has
matured, it's behavior under multiple threads has improved, and even includes
options for memory only databases to be used in multiple threads.

The thread prohibition is known as "check same thread" and may be controlled
using the ``sqlite3`` parameter ``check_same_thread``, which will disable or
enable this check. SQLAlchemy's default behavior here is to set
``check_same_thread`` to ``False`` automatically whenever a file-based database
is in use, to establish compatibility with the default pool class
:class:`.QueuePool`.

The SQLAlchemy ``pysqlite`` DBAPI establishes the connection pool differently
based on the kind of SQLite database that's requested:

* When a ``:memory:`` SQLite database is specified, the dialect by default
  will use :class:`.SingletonThreadPool`. This pool maintains a single
  connection per thread, so that all access to the engine within the current
  thread use the same ``:memory:`` database - other threads would access a
  different ``:memory:`` database.  The ``check_same_thread`` parameter
  defaults to ``True``.
* When a file-based database is specified, the dialect will use
  :class:`.QueuePool` as the source of connections.   at the same time,
  the ``check_same_thread`` flag is set to False by default unless overridden.

  .. versionchanged:: 2.0

    SQLite file database engines now use :class:`.QueuePool` by default.
    Previously, :class:`.NullPool` were used.  The :class:`.NullPool` class
    may be used by specifying it via the
    :paramref:`_sa.create_engine.poolclass` parameter.

Disabling Connection Pooling for File Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pooling may be disabled for a file based database by specifying the
:class:`.NullPool` implementation for the :func:`_sa.create_engine.poolclass`
parameter::

    from sqlalchemy import NullPool

    engine = create_engine("sqlite:///myfile.db", poolclass=NullPool)

It's been observed that the :class:`.NullPool` implementation incurs an
extremely small performance overhead for repeated checkouts due to the lack of
connection re-use implemented by :class:`.QueuePool`.  However, it still
may be beneficial to use this class if the application is experiencing
issues with files being locked.

Using a Memory Database in Multiple Threads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use a ``:memory:`` database in a multithreaded scenario, the same
connection object must be shared among threads, since the database exists
only within the scope of that connection.   The
:class:`.StaticPool` implementation will maintain a single connection
globally, and the ``check_same_thread`` flag can be passed to Pysqlite
as ``False``::

    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

Note that using a ``:memory:`` database in multiple threads requires a recent
version of SQLite.

Using Temporary Tables with SQLite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Due to the way SQLite deals with temporary tables, if you wish to use a
temporary table in a file-based SQLite database across multiple checkouts
from the connection pool, such as when using an ORM :class:`.Session` where
the temporary table should continue to remain after :meth:`.Session.commit` or
:meth:`.Session.rollback` is called, a pool which maintains a single
connection must be used.   Use :class:`.SingletonThreadPool` if the scope is
only needed within the current thread, or :class:`.StaticPool` is scope is
needed within multiple threads for this case::

    # maintain the same connection per thread
    from sqlalchemy.pool import SingletonThreadPool

    engine = create_engine("sqlite:///mydb.db", poolclass=SingletonThreadPool)


    # maintain the same connection across all threads
    from sqlalchemy.pool import StaticPool

    engine = create_engine("sqlite:///mydb.db", poolclass=StaticPool)

Note that :class:`.SingletonThreadPool` should be configured for the number
of threads that are to be used; beyond that number, connections will be
closed out in a non deterministic way.


Dealing with Mixed String / Binary Columns
------------------------------------------------------

The SQLite database is weakly typed, and as such it is possible when using
binary values, which in Python are represented as ``b'some string'``, that a
particular SQLite database can have data values within different rows where
some of them will be returned as a ``b''`` value by the Pysqlite driver, and
others will be returned as Python strings, e.g. ``''`` values.   This situation
is not known to occur if the SQLAlchemy :class:`.LargeBinary` datatype is used
consistently, however if a particular SQLite database has data that was
inserted using the Pysqlite driver directly, or when using the SQLAlchemy
:class:`.String` type which was later changed to :class:`.LargeBinary`, the
table will not be consistently readable because SQLAlchemy's
:class:`.LargeBinary` datatype does not handle strings so it has no way of
"encoding" a value that is in string format.

To deal with a SQLite table that has mixed string / binary data in the
same column, use a custom type that will check each row individually::

    from sqlalchemy import String
    from sqlalchemy import TypeDecorator


    class MixedBinary(TypeDecorator):
        impl = String
        cache_ok = True

        def process_result_value(self, value, dialect):
            if isinstance(value, str):
                value = bytes(value, "utf-8")
            elif value is not None:
                value = bytes(value)

            return value

Then use the above ``MixedBinary`` datatype in the place where
:class:`.LargeBinary` would normally be used.

.. _pysqlite_serializable:

Serializable isolation / Savepoints / Transactional DDL
-------------------------------------------------------

A newly revised version of this important section is now available
at the top level of the SQLAlchemy SQLite documentation, in the section
:ref:`sqlite_transactions`.


.. _pysqlite_udfs:

User-Defined Functions
----------------------

pysqlite supports a `create_function() <https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.create_function>`_
method that allows us to create our own user-defined functions (UDFs) in Python and use them directly in SQLite queries.
These functions are registered with a specific DBAPI Connection.

SQLAlchemy uses connection pooling with file-based SQLite databases, so we need to ensure that the UDF is attached to the
connection when it is created. That is accomplished with an event listener::

    from sqlalchemy import create_engine
    from sqlalchemy import event
    from sqlalchemy import text


    def udf():
        return "udf-ok"


    engine = create_engine("sqlite:///./db_file")


    @event.listens_for(engine, "connect")
    def connect(conn, rec):
        conn.create_function("udf", 0, udf)


    for i in range(5):
        with engine.connect() as conn:
            print(conn.scalar(text("SELECT UDF()")))

"""  # noqa

import math
import os
import re

from .base import DATE
from .base import DATETIME
from .base import SQLiteDialect
from ... import exc
from ... import pool
from ... import types as sqltypes
from ... import util


class _SQLite_pysqliteTimeStamp(DATETIME):
    def bind_processor(self, dialect):
        if dialect.native_datetime:
            return None
        else:
            return DATETIME.bind_processor(self, dialect)

    def result_processor(self, dialect, coltype):
        if dialect.native_datetime:
            return None
        else:
            return DATETIME.result_processor(self, dialect, coltype)


class _SQLite_pysqliteDate(DATE):
    def bind_processor(self, dialect):
        if dialect.native_datetime:
            return None
        else:
            return DATE.bind_processor(self, dialect)

    def result_processor(self, dialect, coltype):
        if dialect.native_datetime:
            return None
        else:
            return DATE.result_processor(self, dialect, coltype)


class SQLiteDialect_pysqlite(SQLiteDialect):
    default_paramstyle = "qmark"
    supports_statement_cache = True
    returns_native_bytes = True

    colspecs = util.update_copy(
        SQLiteDialect.colspecs,
        {
            sqltypes.Date: _SQLite_pysqliteDate,
            sqltypes.TIMESTAMP: _SQLite_pysqliteTimeStamp,
        },
    )

    description_encoding = None

    driver = "pysqlite"

    @classmethod
    def import_dbapi(cls):
        from sqlite3 import dbapi2 as sqlite

        return sqlite

    @classmethod
    def _is_url_file_db(cls, url):
        if (url.database and url.database != ":memory:") and (
            url.query.get("mode", None) != "memory"
        ):
            return True
        else:
            return False

    @classmethod
    def get_pool_class(cls, url):
        if cls._is_url_file_db(url):
            return pool.QueuePool
        else:
            return pool.SingletonThreadPool

    def _get_server_version_info(self, connection):
        return self.dbapi.sqlite_version_info

    _isolation_lookup = SQLiteDialect._isolation_lookup.union(
        {
            "AUTOCOMMIT": None,
        }
    )

    def set_isolation_level(self, dbapi_connection, level):
        if level == "AUTOCOMMIT":
            dbapi_connection.isolation_level = None
        else:
            dbapi_connection.isolation_level = ""
            return super().set_isolation_level(dbapi_connection, level)

    def on_connect(self):
        def regexp(a, b):
            if b is None:
                return None
            return re.search(a, b) is not None

        if util.py38 and self._get_server_version_info(None) >= (3, 9):
            # sqlite must be greater than 3.8.3 for deterministic=True
            # https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.create_function
            # the check is more conservative since there were still issues
            # with following 3.8 sqlite versions
            create_func_kw = {"deterministic": True}
        else:
            create_func_kw = {}

        def set_regexp(dbapi_connection):
            dbapi_connection.create_function(
                "regexp", 2, regexp, **create_func_kw
            )

        def floor_func(dbapi_connection):
            # NOTE: floor is optionally present in sqlite 3.35+ , however
            # as it is normally non-present we deliver floor() unconditionally
            # for now.
            # https://www.sqlite.org/lang_mathfunc.html
            dbapi_connection.create_function(
                "floor", 1, math.floor, **create_func_kw
            )

        fns = [set_regexp, floor_func]

        def connect(conn):
            for fn in fns:
                fn(conn)

        return connect

    def create_connect_args(self, url):
        if url.username or url.password or url.host or url.port:
            raise exc.ArgumentError(
                "Invalid SQLite URL: %s\n"
                "Valid SQLite URL forms are:\n"
                " sqlite:///:memory: (or, sqlite://)\n"
                " sqlite:///relative/path/to/file.db\n"
                " sqlite:////absolute/path/to/file.db" % (url,)
            )

        # theoretically, this list can be augmented, at least as far as
        # parameter names accepted by sqlite3/pysqlite, using
        # inspect.getfullargspec().  for the moment this seems like overkill
        # as these parameters don't change very often, and as always,
        # parameters passed to connect_args will always go to the
        # sqlite3/pysqlite driver.
        pysqlite_args = [
            ("uri", bool),
            ("timeout", float),
            ("isolation_level", str),
            ("detect_types", int),
            ("check_same_thread", bool),
            ("cached_statements", int),
        ]
        opts = url.query
        pysqlite_opts = {}
        for key, type_ in pysqlite_args:
            util.coerce_kw_type(opts, key, type_, dest=pysqlite_opts)

        if pysqlite_opts.get("uri", False):
            uri_opts = dict(opts)
            # here, we are actually separating the parameters that go to
            # sqlite3/pysqlite vs. those that go the SQLite URI.  What if
            # two names conflict?  again, this seems to be not the case right
            # now, and in the case that new names are added to
            # either side which overlap, again the sqlite3/pysqlite parameters
            # can be passed through connect_args instead of in the URL.
            # If SQLite native URIs add a parameter like "timeout" that
            # we already have listed here for the python driver, then we need
            # to adjust for that here.
            for key, type_ in pysqlite_args:
                uri_opts.pop(key, None)
            filename = url.database
            if uri_opts:
                # sorting of keys is for unit test support
                filename += "?" + (
                    "&".join(
                        "%s=%s" % (key, uri_opts[key])
                        for key in sorted(uri_opts)
                    )
                )
        else:
            filename = url.database or ":memory:"
            if filename != ":memory:":
                filename = os.path.abspath(filename)

        pysqlite_opts.setdefault(
            "check_same_thread", not self._is_url_file_db(url)
        )

        return ([filename], pysqlite_opts)

    def is_disconnect(self, e, connection, cursor):
        return isinstance(
            e, self.dbapi.ProgrammingError
        ) and "Cannot operate on a closed database." in str(e)


dialect = SQLiteDialect_pysqlite


class _SQLiteDialect_pysqlite_numeric(SQLiteDialect_pysqlite):
    """numeric dialect for testing only

    internal use only.  This dialect is **NOT** supported by SQLAlchemy
    and may change at any time.

    """

    supports_statement_cache = True
    default_paramstyle = "numeric"
    driver = "pysqlite_numeric"

    _first_bind = ":1"
    _not_in_statement_regexp = None

    def __init__(self, *arg, **kw):
        kw.setdefault("paramstyle", "numeric")
        super().__init__(*arg, **kw)

    def create_connect_args(self, url):
        arg, opts = super().create_connect_args(url)
        opts["factory"] = self._fix_sqlite_issue_99953()
        return arg, opts

    def _fix_sqlite_issue_99953(self):
        import sqlite3

        first_bind = self._first_bind
        if self._not_in_statement_regexp:
            nis = self._not_in_statement_regexp

            def _test_sql(sql):
                m = nis.search(sql)
                assert not m, f"Found {nis.pattern!r} in {sql!r}"

        else:

            def _test_sql(sql):
                pass

        def _numeric_param_as_dict(parameters):
            if parameters:
                assert isinstance(parameters, tuple)
                return {
                    str(idx): value for idx, value in enumerate(parameters, 1)
                }
            else:
                return ()

        class SQLiteFix99953Cursor(sqlite3.Cursor):
            def execute(self, sql, parameters=()):
                _test_sql(sql)
                if first_bind in sql:
                    parameters = _numeric_param_as_dict(parameters)
                return super().execute(sql, parameters)

            def executemany(self, sql, parameters):
                _test_sql(sql)
                if first_bind in sql:
                    parameters = [
                        _numeric_param_as_dict(p) for p in parameters
                    ]
                return super().executemany(sql, parameters)

        class SQLiteFix99953Connection(sqlite3.Connection):
            def cursor(self, factory=None):
                if factory is None:
                    factory = SQLiteFix99953Cursor
                return super().cursor(factory=factory)

            def execute(self, sql, parameters=()):
                _test_sql(sql)
                if first_bind in sql:
                    parameters = _numeric_param_as_dict(parameters)
                return super().execute(sql, parameters)

            def executemany(self, sql, parameters):
                _test_sql(sql)
                if first_bind in sql:
                    parameters = [
                        _numeric_param_as_dict(p) for p in parameters
                    ]
                return super().executemany(sql, parameters)

        return SQLiteFix99953Connection


class _SQLiteDialect_pysqlite_dollar(_SQLiteDialect_pysqlite_numeric):
    """numeric dialect that uses $ for testing only

    internal use only.  This dialect is **NOT** supported by SQLAlchemy
    and may change at any time.

    """

    supports_statement_cache = True
    default_paramstyle = "numeric_dollar"
    driver = "pysqlite_dollar"

    _first_bind = "$1"
    _not_in_statement_regexp = re.compile(r"[^\d]:\d+")

    def __init__(self, *arg, **kw):
        kw.setdefault("paramstyle", "numeric_dollar")
        super().__init__(*arg, **kw)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_ki.py ===
from __future__ import annotations

import contextlib
import inspect
import signal
import sys
import threading
import weakref
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Callable, TypeVar

import outcome
import pytest

from trio.testing import RaisesGroup

from .tutil import gc_collect_harder

try:
    from async_generator import async_generator, yield_
except ImportError:  # pragma: no cover
    async_generator = yield_ = None

from ... import _core
from ..._abc import Instrument
from ..._core import _ki
from ..._timeouts import sleep
from ...testing import wait_all_tasks_blocked

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Generator,
        Iterator,
    )

    from ..._core import Abort, RaiseCancelT


def ki_self() -> None:
    signal.raise_signal(signal.SIGINT)


def test_ki_self() -> None:
    with pytest.raises(KeyboardInterrupt):
        ki_self()


async def test_ki_enabled() -> None:
    # Regular tasks aren't KI-protected
    assert not _core.currently_ki_protected()

    # Low-level call-soon callbacks are KI-protected
    token = _core.current_trio_token()
    record = []

    def check() -> None:
        record.append(_core.currently_ki_protected())

    token.run_sync_soon(check)
    await wait_all_tasks_blocked()
    assert record == [True]

    @_core.enable_ki_protection
    def protected() -> None:
        assert _core.currently_ki_protected()
        unprotected()

    @_core.disable_ki_protection
    def unprotected() -> None:
        assert not _core.currently_ki_protected()

    protected()

    @_core.enable_ki_protection
    async def aprotected() -> None:
        assert _core.currently_ki_protected()
        await aunprotected()

    @_core.disable_ki_protection
    async def aunprotected() -> None:
        assert not _core.currently_ki_protected()

    await aprotected()

    # make sure that the decorator here overrides the automatic manipulation
    # that start_soon() does:
    async with _core.open_nursery() as nursery:
        nursery.start_soon(aprotected)
        nursery.start_soon(aunprotected)

    @_core.enable_ki_protection
    def gen_protected() -> Iterator[None]:
        assert _core.currently_ki_protected()
        yield

    for _ in gen_protected():
        pass

    @_core.disable_ki_protection
    def gen_unprotected() -> Iterator[None]:
        assert not _core.currently_ki_protected()
        yield

    for _ in gen_unprotected():
        pass


# This used to be broken due to
#
#   https://bugs.python.org/issue29590
#
# Specifically, after a coroutine is resumed with .throw(), then the stack
# makes it look like the immediate caller is the function that called
# .throw(), not the actual caller. So child() here would have a caller deep in
# the guts of the run loop, and always be protected, even when it shouldn't
# have been. (Solution: we don't use .throw() anymore.)
async def test_ki_enabled_after_yield_briefly() -> None:
    @_core.enable_ki_protection
    async def protected() -> None:
        await child(True)

    @_core.disable_ki_protection
    async def unprotected() -> None:
        await child(False)

    async def child(expected: bool) -> None:
        import traceback

        traceback.print_stack()
        assert _core.currently_ki_protected() == expected
        await _core.checkpoint()
        traceback.print_stack()
        assert _core.currently_ki_protected() == expected

    await protected()
    await unprotected()


# This also used to be broken due to
#   https://bugs.python.org/issue29590
async def test_generator_based_context_manager_throw() -> None:
    @contextlib.contextmanager
    @_core.enable_ki_protection
    def protected_manager() -> Iterator[None]:
        assert _core.currently_ki_protected()
        try:
            yield
        finally:
            assert _core.currently_ki_protected()

    with protected_manager():
        assert not _core.currently_ki_protected()

    with pytest.raises(KeyError):
        # This is the one that used to fail
        with protected_manager():
            raise KeyError


# the async_generator package isn't typed, hence all the type: ignores
@pytest.mark.skipif(async_generator is None, reason="async_generator not installed")
async def test_async_generator_agen_protection() -> None:
    @_core.enable_ki_protection
    @async_generator  # type: ignore[misc] # untyped generator
    async def agen_protected1() -> None:  # type: ignore[misc] # untyped generator
        assert _core.currently_ki_protected()
        try:
            await yield_()
        finally:
            assert _core.currently_ki_protected()

    @_core.disable_ki_protection
    @async_generator  # type: ignore[misc] # untyped generator
    async def agen_unprotected1() -> None:  # type: ignore[misc] # untyped generator
        assert not _core.currently_ki_protected()
        try:
            await yield_()
        finally:
            assert not _core.currently_ki_protected()

    # Swap the order of the decorators:
    @async_generator  # type: ignore[misc] # untyped generator
    @_core.enable_ki_protection
    async def agen_protected2() -> None:  # type: ignore[misc] # untyped generator
        assert _core.currently_ki_protected()
        try:
            await yield_()
        finally:
            assert _core.currently_ki_protected()

    @async_generator  # type: ignore[misc] # untyped generator
    @_core.disable_ki_protection
    async def agen_unprotected2() -> None:  # type: ignore[misc] # untyped generator
        assert not _core.currently_ki_protected()
        try:
            await yield_()
        finally:
            assert not _core.currently_ki_protected()

    await _check_agen(agen_protected1)
    await _check_agen(agen_protected2)
    await _check_agen(agen_unprotected1)
    await _check_agen(agen_unprotected2)


async def test_native_agen_protection() -> None:
    # Native async generators
    @_core.enable_ki_protection
    async def agen_protected() -> AsyncIterator[None]:
        assert _core.currently_ki_protected()
        try:
            yield
        finally:
            assert _core.currently_ki_protected()

    @_core.disable_ki_protection
    async def agen_unprotected() -> AsyncIterator[None]:
        assert not _core.currently_ki_protected()
        try:
            yield
        finally:
            assert not _core.currently_ki_protected()

    await _check_agen(agen_protected)
    await _check_agen(agen_unprotected)


async def _check_agen(agen_fn: Callable[[], AsyncIterator[None]]) -> None:
    async for _ in agen_fn():
        assert not _core.currently_ki_protected()

    # asynccontextmanager insists that the function passed must itself be an
    # async gen function, not a wrapper around one
    if inspect.isasyncgenfunction(agen_fn):
        async with contextlib.asynccontextmanager(agen_fn)():
            assert not _core.currently_ki_protected()

        # Another case that's tricky due to:
        #   https://bugs.python.org/issue29590
        with pytest.raises(KeyError):
            async with contextlib.asynccontextmanager(agen_fn)():
                raise KeyError


# Test the case where there's no magic local anywhere in the call stack
def test_ki_disabled_out_of_context() -> None:
    assert _core.currently_ki_protected()


def test_ki_disabled_in_del() -> None:
    def nestedfunction() -> bool:
        return _core.currently_ki_protected()

    def __del__() -> None:
        assert _core.currently_ki_protected()
        assert nestedfunction()

    @_core.disable_ki_protection
    def outerfunction() -> None:
        assert not _core.currently_ki_protected()
        assert not nestedfunction()
        __del__()

    __del__()
    outerfunction()
    assert nestedfunction()


def test_ki_protection_works() -> None:
    async def sleeper(name: str, record: set[str]) -> None:
        try:
            while True:
                await _core.checkpoint()
        except _core.Cancelled:
            record.add(name + " ok")

    async def raiser(name: str, record: set[str]) -> None:
        try:
            # os.kill runs signal handlers before returning, so we don't need
            # to worry that the handler will be delayed
            print("killing, protection =", _core.currently_ki_protected())
            ki_self()
        except KeyboardInterrupt:
            print("raised!")
            # Make sure we aren't getting cancelled as well as siginted
            await _core.checkpoint()
            record.add(name + " raise ok")
            raise
        else:
            print("didn't raise!")
            # If we didn't raise (b/c protected), then we *should* get
            # cancelled at the next opportunity
            try:
                await _core.wait_task_rescheduled(lambda _: _core.Abort.SUCCEEDED)
            except _core.Cancelled:
                record.add(name + " cancel ok")

    # simulated control-C during raiser, which is *unprotected*
    print("check 1")
    record_set: set[str] = set()

    async def check_unprotected_kill() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sleeper, "s1", record_set)
            nursery.start_soon(sleeper, "s2", record_set)
            nursery.start_soon(raiser, "r1", record_set)

    # raises inside a nursery, so the KeyboardInterrupt is wrapped in an ExceptionGroup
    with RaisesGroup(KeyboardInterrupt):
        _core.run(check_unprotected_kill)
    assert record_set == {"s1 ok", "s2 ok", "r1 raise ok"}

    # simulated control-C during raiser, which is *protected*, so the KI gets
    # delivered to the main task instead
    print("check 2")
    record_set = set()

    async def check_protected_kill() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sleeper, "s1", record_set)
            nursery.start_soon(sleeper, "s2", record_set)
            nursery.start_soon(_core.enable_ki_protection(raiser), "r1", record_set)
            # __aexit__ blocks, and then receives the KI

    # raises inside a nursery, so the KeyboardInterrupt is wrapped in an ExceptionGroup
    with RaisesGroup(KeyboardInterrupt):
        _core.run(check_protected_kill)
    assert record_set == {"s1 ok", "s2 ok", "r1 cancel ok"}

    # kill at last moment still raises (run_sync_soon until it raises an
    # error, then kill)
    print("check 3")

    async def check_kill_during_shutdown() -> None:
        token = _core.current_trio_token()

        def kill_during_shutdown() -> None:
            assert _core.currently_ki_protected()
            try:
                token.run_sync_soon(kill_during_shutdown)
            except _core.RunFinishedError:
                # it's too late for regular handling! handle this!
                print("kill! kill!")
                ki_self()

        token.run_sync_soon(kill_during_shutdown)

    # no nurseries involved, so the KeyboardInterrupt isn't wrapped
    with pytest.raises(KeyboardInterrupt):
        _core.run(check_kill_during_shutdown)

    # KI arrives very early, before main is even spawned
    print("check 4")

    class InstrumentOfDeath(Instrument):
        def before_run(self) -> None:
            ki_self()

    async def main_1() -> None:
        await _core.checkpoint()

    # no nurseries involved, so the KeyboardInterrupt isn't wrapped
    with pytest.raises(KeyboardInterrupt):
        _core.run(main_1, instruments=[InstrumentOfDeath()])

    # checkpoint_if_cancelled notices pending KI
    print("check 5")

    @_core.enable_ki_protection
    async def main_2() -> None:
        assert _core.currently_ki_protected()
        ki_self()
        with pytest.raises(KeyboardInterrupt):
            await _core.checkpoint_if_cancelled()

    _core.run(main_2)

    # KI arrives while main task is not abortable, b/c already scheduled
    print("check 6")

    @_core.enable_ki_protection
    async def main_3() -> None:
        assert _core.currently_ki_protected()
        ki_self()
        await _core.cancel_shielded_checkpoint()
        await _core.cancel_shielded_checkpoint()
        await _core.cancel_shielded_checkpoint()
        with pytest.raises(KeyboardInterrupt):
            await _core.checkpoint()

    _core.run(main_3)

    # KI arrives while main task is not abortable, b/c refuses to be aborted
    print("check 7")

    @_core.enable_ki_protection
    async def main_4() -> None:
        assert _core.currently_ki_protected()
        ki_self()
        task = _core.current_task()

        def abort(_: RaiseCancelT) -> Abort:
            _core.reschedule(task, outcome.Value(1))
            return _core.Abort.FAILED

        assert await _core.wait_task_rescheduled(abort) == 1
        with pytest.raises(KeyboardInterrupt):
            await _core.checkpoint()

    _core.run(main_4)

    # KI delivered via slow abort
    print("check 8")

    @_core.enable_ki_protection
    async def main_5() -> None:
        assert _core.currently_ki_protected()
        ki_self()
        task = _core.current_task()

        def abort(raise_cancel: RaiseCancelT) -> Abort:
            result = outcome.capture(raise_cancel)
            _core.reschedule(task, result)
            return _core.Abort.FAILED

        with pytest.raises(KeyboardInterrupt):
            assert await _core.wait_task_rescheduled(abort)
        await _core.checkpoint()

    _core.run(main_5)

    # KI arrives just before main task exits, so the run_sync_soon machinery
    # is still functioning and will accept the callback to deliver the KI, but
    # by the time the callback is actually run, main has exited and can't be
    # aborted.
    print("check 9")

    @_core.enable_ki_protection
    async def main_6() -> None:
        ki_self()

    with pytest.raises(KeyboardInterrupt):
        _core.run(main_6)

    print("check 10")
    # KI in unprotected code, with
    # restrict_keyboard_interrupt_to_checkpoints=True
    record_list = []

    async def main_7() -> None:
        # We're not KI protected...
        assert not _core.currently_ki_protected()
        ki_self()
        # ...but even after the KI, we keep running uninterrupted...
        record_list.append("ok")
        # ...until we hit a checkpoint:
        with pytest.raises(KeyboardInterrupt):
            await sleep(10)

    _core.run(main_7, restrict_keyboard_interrupt_to_checkpoints=True)
    assert record_list == ["ok"]
    record_list = []
    # Exact same code raises KI early if we leave off the argument, doesn't
    # even reach the record.append call:
    with pytest.raises(KeyboardInterrupt):
        _core.run(main_7)
    assert record_list == []

    # KI arrives while main task is inside a cancelled cancellation scope
    # the KeyboardInterrupt should take priority
    print("check 11")

    @_core.enable_ki_protection
    async def main_8() -> None:
        assert _core.currently_ki_protected()
        with _core.CancelScope() as cancel_scope:
            cancel_scope.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
            ki_self()
            with pytest.raises(KeyboardInterrupt):
                await _core.checkpoint()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()

    _core.run(main_8)


def test_ki_is_good_neighbor() -> None:
    # in the unlikely event someone overwrites our signal handler, we leave
    # the overwritten one be
    try:
        orig = signal.getsignal(signal.SIGINT)

        def my_handler(signum: object, frame: object) -> None:  # pragma: no cover
            pass

        async def main() -> None:
            signal.signal(signal.SIGINT, my_handler)

        _core.run(main)

        assert signal.getsignal(signal.SIGINT) is my_handler
    finally:
        signal.signal(signal.SIGINT, orig)


# Regression test for #461
# don't know if _active not being visible is a problem
def test_ki_with_broken_threads() -> None:
    thread = threading.main_thread()

    # scary!
    original = threading._active[thread.ident]  # type: ignore[attr-defined]

    # put this in a try finally so we don't have a chance of cascading a
    # breakage down to everything else
    try:
        del threading._active[thread.ident]  # type: ignore[attr-defined]

        @_core.enable_ki_protection
        async def inner() -> None:
            assert signal.getsignal(signal.SIGINT) != signal.default_int_handler

        _core.run(inner)
    finally:
        threading._active[thread.ident] = original  # type: ignore[attr-defined]


_T = TypeVar("_T")


def _identity(v: _T) -> _T:
    return v


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "it was decided not to protect against this case, see discussion in: "
        "https://github.com/python-trio/trio/pull/3110#discussion_r1802123644"
    ),
)
async def test_ki_does_not_leak_across_different_calls_to_inner_functions() -> None:
    assert not _core.currently_ki_protected()

    def factory(enabled: bool) -> Callable[[], bool]:
        @_core.enable_ki_protection if enabled else _identity
        def decorated() -> bool:
            return _core.currently_ki_protected()

        return decorated

    decorated_enabled = factory(True)
    decorated_disabled = factory(False)
    assert decorated_enabled()
    assert not decorated_disabled()


async def test_ki_protection_check_does_not_freeze_locals() -> None:
    class A:
        pass

    a = A()
    wr_a = weakref.ref(a)
    assert not _core.currently_ki_protected()
    del a
    if sys.implementation.name == "pypy":
        gc_collect_harder()
    assert wr_a() is None


def test_identity_weakref_internals() -> None:
    """To cover the parts WeakKeyIdentityDictionary won't ever reach."""

    class A:
        def __eq__(self, other: object) -> bool:
            return False

    a = A()
    assert a != a
    wr = _ki._IdRef(a)
    wr_other_is_self = wr

    # dict always checks identity before equality so we need to do it here
    # to cover `if self is other`
    assert wr == wr_other_is_self

    # we want to cover __ne__ and `return NotImplemented`
    assert wr != object()


def test_weak_key_identity_dict_remove_callback_keyerror() -> None:
    """We need to cover the KeyError in self._remove."""

    class A:
        def __eq__(self, other: object) -> bool:
            return False

    a = A()
    assert a != a
    d: _ki.WeakKeyIdentityDictionary[A, bool] = _ki.WeakKeyIdentityDictionary()

    d[a] = True

    data_copy = d._data.copy()
    d._data.clear()
    del a

    gc_collect_harder()  # would call sys.unraisablehook if there's a problem
    assert data_copy


def test_weak_key_identity_dict_remove_callback_selfref_expired() -> None:
    """We need to cover the KeyError in self._remove."""

    class A:
        def __eq__(self, other: object) -> bool:
            return False

    a = A()
    assert a != a
    d: _ki.WeakKeyIdentityDictionary[A, bool] = _ki.WeakKeyIdentityDictionary()

    d[a] = True

    data_copy = d._data.copy()
    wr_d = weakref.ref(d)
    del d
    gc_collect_harder()  # would call sys.unraisablehook if there's a problem
    assert wr_d() is None
    del a
    gc_collect_harder()
    assert data_copy


@_core.enable_ki_protection
async def _protected_async_gen_fn() -> AsyncGenerator[None, None]:
    yield


@_core.enable_ki_protection
async def _protected_async_fn() -> None:
    pass


@_core.enable_ki_protection
def _protected_gen_fn() -> Generator[None, None, None]:
    yield


@_core.disable_ki_protection
async def _unprotected_async_gen_fn() -> AsyncGenerator[None, None]:
    yield


@_core.disable_ki_protection
async def _unprotected_async_fn() -> None:
    pass


@_core.disable_ki_protection
def _unprotected_gen_fn() -> Generator[None, None, None]:
    yield


async def _consume_async_generator(agen: AsyncGenerator[None, None]) -> None:
    try:
        with pytest.raises(StopAsyncIteration):
            while True:
                await agen.asend(None)
    finally:
        await agen.aclose()


def _consume_function_for_coverage(
    fn: Callable[[], object],
) -> None:
    result = fn()
    if inspect.isasyncgen(result):
        result = _consume_async_generator(result)

    assert inspect.isgenerator(result) or inspect.iscoroutine(result)
    with pytest.raises(StopIteration):
        while True:
            result.send(None)


def test_enable_disable_ki_protection_passes_on_inspect_flags() -> None:
    assert inspect.isasyncgenfunction(_protected_async_gen_fn)
    _consume_function_for_coverage(_protected_async_gen_fn)
    assert inspect.iscoroutinefunction(_protected_async_fn)
    _consume_function_for_coverage(_protected_async_fn)
    assert inspect.isgeneratorfunction(_protected_gen_fn)
    _consume_function_for_coverage(_protected_gen_fn)
    assert inspect.isasyncgenfunction(_unprotected_async_gen_fn)
    _consume_function_for_coverage(_unprotected_async_gen_fn)
    assert inspect.iscoroutinefunction(_unprotected_async_fn)
    _consume_function_for_coverage(_unprotected_async_fn)
    assert inspect.isgeneratorfunction(_unprotected_gen_fn)
    _consume_function_for_coverage(_unprotected_gen_fn)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\llama\benchmark.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import argparse
import datetime
import gc
import itertools
import logging
import os
import sys
import time

import numpy as np
import onnx
import psutil
import torch
from benchmark_helper import measure_memory, setup_logger
from dist_settings import get_rank, get_size
from llama_inputs import (
    add_io_bindings_as_ortvalues,
    get_merged_sample_with_past_kv_inputs,
    get_msft_sample_inputs,
    get_sample_inputs,
    get_sample_with_past_kv_inputs,
    verify_ort_inputs,
)
from optimum.onnxruntime import ORTModelForCausalLM
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import onnxruntime as ort

logger = logging.getLogger(__name__)


# For determining whether the ONNX model can do both prompt generation and token generation or only one of the two
def get_ort_model_inputs_len(args, model):
    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
        return 0
    if args.benchmark_type == "hf-ort":
        try:
            # New Optimum export (https://github.com/huggingface/optimum/blob/888332364c2e0091da1fc974737c7e277af168bf/optimum/onnxruntime/modeling_ort.py#L268)
            return len(model.inputs_names)
        except Exception:
            # Old Optimum export (https://github.com/huggingface/optimum/blob/c5ad7f971cb0a494eac03dc0909f146725f999c5/optimum/onnxruntime/base.py#L54)
            return len(model.decoder.input_names)
    return len(model.get_inputs())


def get_inputs(args: argparse.Namespace, ort_model_inputs_len: int):
    init_inputs, iter_inputs = None, None

    # For past_present_share_buffer:
    # Set max_seq_len to 2048 for Microsoft LLaMA-2 model since that is the max value currently supported
    # Set max_seq_len to config value for other models
    max_seq_len = 2048 if args.benchmark_type == "ort-msft" else args.config.max_position_embeddings

    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
        init_inputs = get_sample_inputs(
            args.config,
            args.target_device,
            args.batch_size,
            args.sequence_length,
            return_dict=True,
        )
        iter_inputs = get_sample_with_past_kv_inputs(
            args.config,
            args.target_device,
            args.batch_size,
            args.sequence_length,
            use_fp16=args.use_fp16,
            return_dict=True,
        )

    elif args.benchmark_type in {"hf-ort"}:
        if ort_model_inputs_len == 3:  # [input_ids, attention_mask, position_ids]
            # Using split models in Optimum (e.g. created by Optimum export)
            init_inputs = get_sample_inputs(
                args.config,
                args.target_device,
                args.batch_size,
                args.sequence_length,
                return_dict=True,
            )
            iter_inputs = get_sample_with_past_kv_inputs(
                args.config,
                args.target_device,
                args.batch_size,
                args.sequence_length,
                use_fp16=args.use_fp16,
                return_dict=True,
            )
        else:
            # Using merged model in Optimum (e.g. created by convert_to_onnx export)
            init_inputs = get_merged_sample_with_past_kv_inputs(
                args.config,
                args.target_device,
                args.batch_size,
                seq_len=args.sequence_length,
                past_seq_len=0,
                max_seq_len=max_seq_len,
                use_fp16=args.use_fp16,
                use_buffer_share=args.use_buffer_share,
                engine="pt",
                return_dict=True,
            )
            iter_inputs = get_merged_sample_with_past_kv_inputs(
                args.config,
                args.target_device,
                args.batch_size,
                seq_len=1,
                past_seq_len=args.sequence_length,
                max_seq_len=max_seq_len,
                use_fp16=args.use_fp16,
                use_buffer_share=args.use_buffer_share,
                engine="pt",
                return_dict=True,
            )

    elif args.benchmark_type == "ort-convert-to-onnx":
        # Microsoft export from convert_to_onnx
        init_inputs = get_merged_sample_with_past_kv_inputs(
            args.config,
            args.target_device,
            args.batch_size,
            seq_len=args.sequence_length,
            past_seq_len=0,
            max_seq_len=max_seq_len,
            use_fp16=args.use_fp16,
            use_buffer_share=args.use_buffer_share,
            engine="ort",
            return_dict=True,
            world_size=args.world_size,
        )
        iter_inputs = get_merged_sample_with_past_kv_inputs(
            args.config,
            args.target_device,
            args.batch_size,
            seq_len=1,
            past_seq_len=args.sequence_length,
            max_seq_len=max_seq_len,
            use_fp16=args.use_fp16,
            use_buffer_share=args.use_buffer_share,
            engine="ort",
            return_dict=True,
            world_size=args.world_size,
        )

    elif args.benchmark_type == "ort-msft":
        # Microsoft export from https://github.com/microsoft/Llama-2-Onnx
        split_kv = ort_model_inputs_len > 5  # original inputs: [x, attn_mask, k_cache, v_cache, pos]

        init_inputs = get_msft_sample_inputs(
            args.config,
            args.batch_size,
            past_seq_len=0,
            seq_len=args.sequence_length,
            max_seq_len=max_seq_len,
            use_fp16=args.use_fp16,
            use_buffer_share=args.use_buffer_share,
            split_kv=split_kv,
        )
        iter_inputs = get_msft_sample_inputs(
            args.config,
            args.batch_size,
            past_seq_len=args.sequence_length,
            seq_len=1,
            max_seq_len=max_seq_len,
            use_fp16=args.use_fp16,
            use_buffer_share=args.use_buffer_share,
            split_kv=split_kv,
        )

    else:
        raise Exception("Unable to auto-detect inputs for provided model")

    return init_inputs, iter_inputs


def get_model(args: argparse.Namespace):
    model, sess_options = None, None
    start_time, end_time = None, None

    # There are multiple sources that the model could come from:
    # 1) Benchmark LLaMA-2 from unofficial source on Hugging Face
    # 2) Benchmark LLaMA-2 from official source on Hugging Face, which requires an authentication token
    # 3) Benchmark LLaMA-2 from local download of model
    # 4) Benchmark LLaMA-2 from Microsoft (already optimized, available at https://github.com/microsoft/Llama-2-Onnx)
    # 5) Benchmark LLaMA-2 from convert_to_onnx

    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
        source = args.hf_pt_dir_path if args.hf_pt_dir_path else args.model_name
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
            use_auth_token=args.auth,
            trust_remote_code=args.auth,
            use_cache=True,
            cache_dir=args.cache_dir,
        ).to(args.target_device)
        end_time = time.time()

        if args.benchmark_type == "hf-pt-compile":
            model = torch.compile(model)

    elif args.benchmark_type in {"hf-ort", "ort-msft", "ort-convert-to-onnx"}:
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = args.profile
        if args.verbose:
            sess_options.log_verbosity_level = 1
            sess_options.log_severity_level = 1

    else:
        raise Exception(f"Cannot recognize {args.benchmark_type}")

    if args.benchmark_type == "hf-ort":
        # Optimum export or convert_to_onnx.py export
        provider = args.execution_provider[0] if type(args.execution_provider) is tuple else args.execution_provider
        provider_options = args.execution_provider[1] if type(args.execution_provider) is tuple else None

        decoder_file_name = None
        decoder_with_past_file_name = None
        for filename in os.listdir(args.hf_ort_dir_path):
            if ".onnx" not in filename or ".onnx_data" in filename or ".onnx.data" in filename:
                continue
            if "decoder_model" in filename or filename == "model.onnx":
                decoder_file_name = filename
            if "decoder_with_past_model" in filename:
                decoder_with_past_file_name = filename
            if "decoder_merged_model" in filename:
                decoder_file_name = filename
                decoder_with_past_file_name = filename

        start_time = time.time()
        model = ORTModelForCausalLM.from_pretrained(
            args.hf_ort_dir_path,
            decoder_file_name=decoder_file_name,
            decoder_with_past_file_name=decoder_with_past_file_name,
            use_auth_token=args.auth,
            trust_remote_code=args.auth,
            use_io_binding=True,  # Large perf gain even for cpu due to avoiding output copy.
            use_merged=(True if decoder_file_name == "model.onnx" else None),
            provider=provider,
            provider_options=provider_options,
            session_options=sess_options,
        )
        end_time = time.time()

    if args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}:
        # Ex: Microsoft export from https://github.com/microsoft/Llama-2-Onnx
        logger.info(f"Loading model from {args.ort_model_path.format(args.rank)}")
        start_time = time.time()
        model = ort.InferenceSession(
            args.ort_model_path.format(args.rank),
            sess_options,
            providers=[args.execution_provider],
        )
        end_time = time.time()

    logger.info(f"Loaded model in {end_time - start_time} s")
    return model


def time_fn(args, fn, inputs):
    # Warm up
    warmup_range = (
        range(args.warmup_runs)
        if args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}
        else trange(args.warmup_runs, file=sys.stdout, desc="Warm up")
    )

    if args.verbose:
        outputs = fn(inputs)
        logger.info(outputs)

    input_sync = lambda *kwargs: (  # noqa: E731
        args.io_binding.synchronize_inputs()
        if args.device != "cpu" and args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}  # ORT synchronize
        else lambda *kwargs: (
            torch.cuda.synchronize()
            if args.device != "cpu" and torch.cuda.is_available()  # PyTorch synchronize
            else lambda *kwargs: None
        )
    )  # no-op function

    output_sync = lambda *kwargs: (  # noqa: E731
        args.io_binding.synchronize_outputs()
        if args.device != "cpu" and args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}  # ORT synchronize
        else lambda *kwargs: (
            torch.cuda.synchronize()
            if args.device != "cpu" and torch.cuda.is_available()  # PyTorch synchronize
            else lambda *kwargs: None
        )
    )  # no-op function

    for _ in warmup_range:
        input_sync()
        fn(inputs)
        output_sync()

    # Benchmark
    total_time = 0
    bench_range = (
        range(args.num_runs)
        if args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}
        else trange(args.num_runs, file=sys.stdout, desc="Benchmark")
    )
    for _ in bench_range:
        input_sync()
        start_time = time.time()

        fn(inputs)

        output_sync()
        end_time = time.time()

        total_time += end_time - start_time

    # Newline print after trange in order to print metrics on new lines without progress bar on same line
    if args.benchmark_type not in {"ort-msft", "ort-convert-to-onnx"}:
        logger.info("")

    latency = total_time / args.num_runs
    throughput = args.batch_size / latency

    if args.rank == 0:
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Sequence Length: {args.sequence_length}")
        logger.info(f"Latency: {latency} s")
        logger.info(f"Throughput: {throughput} tps")
    return


def profile_fn(args, fn, inputs, inputs_type):
    # Filename prefix format:
    # "b<batch-size>_s<sequence-length>_<benchmark-type>-<precision>-<device>_<inference-step>_<inputs-type>_<current-time>"
    prefix = f"b{args.batch_size}_s{args.sequence_length}_{args.benchmark_type.lower()}-{args.precision}-{args.device}_{fn.__name__.replace('_', '-')}_{inputs_type}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}"
    filename = None

    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
        # Profile PyTorch kernels
        with profile(  # noqa: SIM117
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        ) as prof:
            with record_function("model_inference"):
                fn(inputs)
        prof_data = prof.key_averages(group_by_stack_n=5).table(sort_by=args.pt_filter_by, row_limit=args.pt_num_rows)

        filename = os.path.join(args.log_folder, f"{prefix}.log")
        with open(filename, "w") as f:
            f.write(prof_data)

    else:
        # Profile ORT kernels
        fn(inputs)

        # Set new log name for ORT profile log generated
        filename = f"{prefix}.json"

    return filename


def measure_fn(args, fn, inputs):
    # Measure CPU usage
    pid = os.getpid()
    process = psutil.Process(pid)
    process.cpu_percent(interval=0.1)

    fn(inputs)
    if args.rank == 0:
        logger.info(f"CPU usage: {process.cpu_percent(interval=None) / psutil.cpu_count(logical=False)}%")

    # Measure memory usage
    gc.collect()
    torch.cuda.empty_cache()
    measure_memory(is_gpu=(args.device != "cpu"), func=lambda: fn(inputs))

    # Flush output so memory usage is printed
    sys.stdout.flush()


def run_hf_inference(args, init_inputs, iter_inputs, model):
    # Inference steps to measure
    def get_logits(inputs):
        # Inference pass without decoding
        outputs = model(**inputs)
        return outputs

    # Examples of other inference steps that can be measured:
    # To use, uncomment the function and assign it to `generate_fn`

    # def get_pred_ids(inputs):
    #     # Inference pass with predicted token ids generation
    #     predicted_ids = model.generate(**inputs)
    #     return predicted_ids

    # def gen_and_dec(inputs):
    #     # Inference pass with generation and decoding
    #     predicted_ids = get_pred_ids(inputs)
    #     transcription = []
    #     for bs in range(args.batch_size):
    #         for rs in range(args.num_return_sequences):
    #             transcription.append(
    #                 args.tokenizer.batch_decode(
    #                     predicted_ids[bs * args.num_return_sequences + rs], skip_special_tokens=True
    #                 )[0]
    #             )
    #     return transcription

    generate_fn = get_logits

    if args.benchmark_type == "hf-pt-compile":
        # Run forward pass once with each set of inputs to process through Dynamo
        generate_fn(init_inputs)
        generate_fn(iter_inputs)

    if args.profile:
        new_logname = profile_fn(args, generate_fn, init_inputs, "prompt")
        if args.benchmark_type == "hf-ort":
            # Turn profiling off to stop appending to log
            old_logname = model.decoder.session.end_profiling()
            logger.warning(f"Renaming {old_logname} to {new_logname}")
            os.rename(old_logname, os.path.join(args.log_folder, new_logname))

        new_logname = profile_fn(args, generate_fn, iter_inputs, "token")
        if args.benchmark_type == "hf-ort":
            # Turn profiling off to stop appending to log
            old_logname = model.decoder_with_past.session.end_profiling()
            logger.warning(f"Renaming {old_logname} to {new_logname}")
            os.rename(old_logname, os.path.join(args.log_folder, new_logname))

        return

    # PyTorch evaluations
    logger.info("\nEvaluating `model(inputs)` step to get past_key_values")
    time_fn(args, generate_fn, init_inputs)
    measure_fn(args, generate_fn, init_inputs)

    logger.info("\nEvaluating `model(inputs)` step with past_key_values")
    time_fn(args, generate_fn, iter_inputs)
    measure_fn(args, generate_fn, iter_inputs)


def run_ort_inference(args, init_inputs, iter_inputs, model):
    def prepare_ort_inputs(inputs, kv_cache_ortvalues):
        # Verify model inputs
        inputs = verify_ort_inputs(model, inputs)

        # Add IO bindings for non-CPU execution providers
        if args.device != "cpu":
            io_binding, kv_cache_ortvalues = add_io_bindings_as_ortvalues(
                model, inputs, args.device, int(args.rank), args.use_buffer_share, kv_cache_ortvalues
            )
            setattr(args, "io_binding", io_binding)  # noqa: B010
            return io_binding, kv_cache_ortvalues

        return inputs, kv_cache_ortvalues

    def with_io_binding(io_binding):
        # Inference pass with IO binding
        model.run_with_iobinding(io_binding)

    def without_io_binding(inputs):
        # Inference pass without IO binding
        outputs = model.run(None, inputs)
        return outputs

    generate_fn = with_io_binding if args.device != "cpu" else without_io_binding
    kv_cache_ortvalues = {}

    if args.profile:
        ort_init_inputs, kv_cache_ortvalues = prepare_ort_inputs(init_inputs, kv_cache_ortvalues)
        new_logname = profile_fn(args, generate_fn, ort_init_inputs, "prompt")

        # Turn profiling off to stop appending to log file
        old_logname = model.end_profiling()
        logger.warning(f"Renaming {old_logname} to {new_logname}")
        os.rename(old_logname, os.path.join(args.log_folder, new_logname))

        # Re-initialize model for new log file instead of appending to old log file
        model = get_model(args)
        ort_iter_inputs, kv_cache_ortvalues = prepare_ort_inputs(iter_inputs, kv_cache_ortvalues)
        new_logname = profile_fn(args, generate_fn, ort_iter_inputs, "token")

        # Turn profiling off to stop appending to log
        old_logname = model.end_profiling()
        logger.warning(f"Renaming {old_logname} to {new_logname}")
        os.rename(old_logname, os.path.join(args.log_folder, new_logname))
        return

    # ORT evaluations
    logger.info("\nEvaluating `model(inputs)` step to get past_key_values")
    ort_init_inputs, kv_cache_ortvalues = prepare_ort_inputs(init_inputs, kv_cache_ortvalues)
    time_fn(args, generate_fn, ort_init_inputs)
    measure_fn(args, generate_fn, ort_init_inputs)

    logger.info("\nEvaluating `model(inputs)` step with past_key_values")
    ort_iter_inputs, kv_cache_ortvalues = prepare_ort_inputs(iter_inputs, kv_cache_ortvalues)
    time_fn(args, generate_fn, ort_iter_inputs)
    measure_fn(args, generate_fn, ort_iter_inputs)


def run_inference(args, init_inputs, iter_inputs, model):
    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile", "hf-ort"}:
        run_hf_inference(args, init_inputs, iter_inputs, model)
    elif args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}:
        run_ort_inference(args, init_inputs, iter_inputs, model)
    else:
        raise Exception(f"Cannot recognize {args.benchmark_type}")


def get_args(rank=0):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bt",
        "--benchmark-type",
        type=str,
        required=True,
        choices=[
            "hf-pt-eager",
            "hf-pt-compile",
            "hf-ort",
            "ort-msft",
            "ort-convert-to-onnx",
        ],
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face name of model (e.g. 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "-a", "--auth", default=False, action="store_true", help="Use Hugging Face authentication token to access model"
    )

    # Args for choosing the model
    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        type=str,
        default="fp32",
        choices=["int4", "int8", "fp16", "fp32"],
        help="Precision for model. For ONNX models, the model's precision should be set before running this script.",
    )
    parser.add_argument(
        "--hf-pt-dir-path",
        type=str,
        default="",
        help="Path to directory containing all PyTorch files (e.g. tokenizer, PyTorch model)",
    )
    parser.add_argument(
        "--hf-ort-dir-path",
        type=str,
        default="",
        help="Path to directory containing all ONNX files (e.g. tokenizer, decoder_merged, decoder, decoder_with_past)",
    )
    parser.add_argument(
        "--ort-model-path",
        type=str,
        default="",
        help="Path to ONNX model",
    )

    # Args for running and evaluating the model
    parser.add_argument(
        "-b",
        "--batch-sizes",
        default="1 2",
    )
    parser.add_argument(
        "-s",
        "--sequence-lengths",
        default="32 64 128 256 512",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "rocm"],
    )
    parser.add_argument("-id", "--device-id", type=int, default=0)
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-n", "--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)

    # Args for decoding logic
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--num-return-sequences", type=int, default=1)

    # Args for accessing detailed info
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument(
        "--pt-filter-by", type=str, default="self_cpu_time_total", help="What to filter PyTorch profiler by"
    )
    parser.add_argument("--pt-num-rows", type=int, default=1000, help="Number of rows for PyTorch profiler to display")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--log-folder", type=str, default=os.path.join("."), help="Folder to cache log files")
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        default="./model_cache",
        help="Cache dir where Hugging Face files are stored",
    )

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set runtime properties
    if "ort" in args.benchmark_type:
        setattr(args, "execution_provider", f"{args.device.upper()}ExecutionProvider")  # noqa: B010
        if args.execution_provider == "CUDAExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": rank})
        elif args.execution_provider == "ROCMExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": rank})
            args.device = "cuda"

    # Check that paths have been specified for any benchmarking with ORT
    if args.benchmark_type == "hf-ort":
        assert args.hf_ort_dir_path, "Please specify a path to `--hf-ort-dir-path`"
    if args.benchmark_type in {"ort-msft", "ort-convert-to-onnx"}:
        assert args.ort_model_path, "Please specify a path to `--ort-model-path`"

    args.batch_sizes = args.batch_sizes.split(" ")
    args.sequence_lengths = args.sequence_lengths.split(" ")

    # Use FP32 precision for FP32, INT8, INT4 CPU models, use FP16 precision for FP16 and INT4 GPU models
    args.precision = (
        "fp32" if args.precision in {"int8", "fp32"} or (args.precision == "int4" and args.device == "cpu") else "fp16"
    )

    # Check that only one (batch_size, sequence_length) combination is set for profiling
    if args.profile:
        assert len(args.batch_sizes) == 1 and len(args.sequence_lengths) == 1, (
            "Please provide only one (batch_size, sequence_length) combination for profiling"
        )

    return args


def main():
    rank = get_rank()
    world_size = get_size()

    args = get_args(rank)
    setup_logger(args.verbose)
    logger.info(args.__dict__)
    torch.backends.cudnn.benchmark = True

    args.rank = rank
    args.world_size = world_size
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, use_auth_token=args.auth, trust_remote_code=args.auth
    )
    config = AutoConfig.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, use_auth_token=args.auth, trust_remote_code=args.auth
    )
    target_device = f"cuda:{args.rank}" if args.device != "cpu" else args.device
    use_fp16 = args.precision == "fp16"

    setattr(args, "tokenizer", tokenizer)  # noqa: B010
    setattr(args, "config", config)  # noqa: B010
    setattr(args, "target_device", target_device)  # noqa: B010
    setattr(args, "use_fp16", use_fp16)  # noqa: B010

    # Get model and model info
    model = get_model(args)
    ort_model_inputs_len = get_ort_model_inputs_len(args, model)

    # Check if past_present_share_buffer can be enabled (only for FP16 models with GQA)
    if args.benchmark_type in {"ort-convert-to-onnx", "ort-msft"}:
        onnx_model = onnx.load_model(args.ort_model_path.format(args.rank), load_external_data=False)
        gqa_nodes = list(filter(lambda node: node.op_type == "GroupQueryAttention", onnx_model.graph.node))

        use_buffer_share = use_fp16 and len(gqa_nodes) > 0 and args.device != "cpu"
        setattr(args, "use_buffer_share", use_buffer_share)  # noqa: B010
    else:
        setattr(args, "use_buffer_share", False)  # noqa: B010

    # Measure prompt cost (init_inputs) and generated token cost (iter_inputs)
    for batch_size, sequence_length in itertools.product(args.batch_sizes, args.sequence_lengths):
        if args.rank == 0:
            logger.info(f"\nBatch size = {batch_size} and sequence length = {sequence_length}...")
        setattr(args, "batch_size", int(batch_size))  # noqa: B010
        setattr(args, "sequence_length", int(sequence_length))  # noqa: B010

        init_inputs, iter_inputs = get_inputs(args, ort_model_inputs_len)
        run_inference(args, init_inputs, iter_inputs, model)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\gruntz.py ===
"""
Limits
======

Implemented according to the PhD thesis
https://www.cybertester.com/data/gruntz.pdf, which contains very thorough
descriptions of the algorithm including many examples.  We summarize here
the gist of it.

All functions are sorted according to how rapidly varying they are at
infinity using the following rules. Any two functions f and g can be
compared using the properties of L:

L=lim  log|f(x)| / log|g(x)|           (for x -> oo)

We define >, < ~ according to::

    1. f > g .... L=+-oo

        we say that:
        - f is greater than any power of g
        - f is more rapidly varying than g
        - f goes to infinity/zero faster than g

    2. f < g .... L=0

        we say that:
        - f is lower than any power of g

    3. f ~ g .... L!=0, +-oo

        we say that:
        - both f and g are bounded from above and below by suitable integral
          powers of the other

Examples
========
::
    2 < x < exp(x) < exp(x**2) < exp(exp(x))
    2 ~ 3 ~ -5
    x ~ x**2 ~ x**3 ~ 1/x ~ x**m ~ -x
    exp(x) ~ exp(-x) ~ exp(2x) ~ exp(x)**2 ~ exp(x+exp(-x))
    f ~ 1/f

So we can divide all the functions into comparability classes (x and x^2
belong to one class, exp(x) and exp(-x) belong to some other class). In
principle, we could compare any two functions, but in our algorithm, we
do not compare anything below the class 2~3~-5 (for example log(x) is
below this), so we set 2~3~-5 as the lowest comparability class.

Given the function f, we find the list of most rapidly varying (mrv set)
subexpressions of it. This list belongs to the same comparability class.
Let's say it is {exp(x), exp(2x)}. Using the rule f ~ 1/f we find an
element "w" (either from the list or a new one) from the same
comparability class which goes to zero at infinity. In our example we
set w=exp(-x) (but we could also set w=exp(-2x) or w=exp(-3x) ...). We
rewrite the mrv set using w, in our case {1/w, 1/w^2}, and substitute it
into f. Then we expand f into a series in w::

    f = c0*w^e0 + c1*w^e1 + ... + O(w^en),       where e0<e1<...<en, c0!=0

but for x->oo, lim f = lim c0*w^e0, because all the other terms go to zero,
because w goes to zero faster than the ci and ei. So::

    for e0>0, lim f = 0
    for e0<0, lim f = +-oo   (the sign depends on the sign of c0)
    for e0=0, lim f = lim c0

We need to recursively compute limits at several places of the algorithm, but
as is shown in the PhD thesis, it always finishes.

Important functions from the implementation:

compare(a, b, x) compares "a" and "b" by computing the limit L.
mrv(e, x) returns list of most rapidly varying (mrv) subexpressions of "e"
rewrite(e, Omega, x, wsym) rewrites "e" in terms of w
leadterm(f, x) returns the lowest power term in the series of f
mrv_leadterm(e, x) returns the lead term (c0, e0) for e
limitinf(e, x) computes lim e  (for x->oo)
limit(e, z, z0) computes any limit by converting it to the case x->oo

All the functions are really simple and straightforward except
rewrite(), which is the most difficult/complex part of the algorithm.
When the algorithm fails, the bugs are usually in the series expansion
(i.e. in SymPy) or in rewrite.

This code is almost exact rewrite of the Maple code inside the Gruntz
thesis.

Debugging
---------

Because the gruntz algorithm is highly recursive, it's difficult to
figure out what went wrong inside a debugger. Instead, turn on nice
debug prints by defining the environment variable SYMPY_DEBUG. For
example:

[user@localhost]: SYMPY_DEBUG=True ./bin/isympy

In [1]: limit(sin(x)/x, x, 0)
limitinf(_x*sin(1/_x), _x) = 1
+-mrv_leadterm(_x*sin(1/_x), _x) = (1, 0)
| +-mrv(_x*sin(1/_x), _x) = set([_x])
| | +-mrv(_x, _x) = set([_x])
| | +-mrv(sin(1/_x), _x) = set([_x])
| |   +-mrv(1/_x, _x) = set([_x])
| |     +-mrv(_x, _x) = set([_x])
| +-mrv_leadterm(exp(_x)*sin(exp(-_x)), _x, set([exp(_x)])) = (1, 0)
|   +-rewrite(exp(_x)*sin(exp(-_x)), set([exp(_x)]), _x, _w) = (1/_w*sin(_w), -_x)
|     +-sign(_x, _x) = 1
|     +-mrv_leadterm(1, _x) = (1, 0)
+-sign(0, _x) = 0
+-limitinf(1, _x) = 1

And check manually which line is wrong. Then go to the source code and
debug this function to figure out the exact problem.

"""
from functools import reduce

from sympy.core import Basic, S, Mul, PoleError
from sympy.core.cache import cacheit
from sympy.core.function import AppliedUndef
from sympy.core.intfunc import ilcm
from sympy.core.numbers import I, oo
from sympy.core.symbol import Dummy, Wild
from sympy.core.traversal import bottom_up

from sympy.functions import log, exp, sign as _sign
from sympy.series.order import Order
from sympy.utilities.misc import debug_decorator as debug
from sympy.utilities.timeutils import timethis

timeit = timethis('gruntz')


def compare(a, b, x):
    """Returns "<" if a<b, "=" for a == b, ">" for a>b"""
    # log(exp(...)) must always be simplified here for termination
    la, lb = log(a), log(b)
    if isinstance(a, Basic) and (isinstance(a, exp) or (a.is_Pow and a.base == S.Exp1)):
        la = a.exp
    if isinstance(b, Basic) and (isinstance(b, exp) or (b.is_Pow and b.base == S.Exp1)):
        lb = b.exp

    c = limitinf(la/lb, x)
    if c == 0:
        return "<"
    elif c.is_infinite:
        return ">"
    else:
        return "="


class SubsSet(dict):
    """
    Stores (expr, dummy) pairs, and how to rewrite expr-s.

    Explanation
    ===========

    The gruntz algorithm needs to rewrite certain expressions in term of a new
    variable w. We cannot use subs, because it is just too smart for us. For
    example::

        > Omega=[exp(exp(_p - exp(-_p))/(1 - 1/_p)), exp(exp(_p))]
        > O2=[exp(-exp(_p) + exp(-exp(-_p))*exp(_p)/(1 - 1/_p))/_w, 1/_w]
        > e = exp(exp(_p - exp(-_p))/(1 - 1/_p)) - exp(exp(_p))
        > e.subs(Omega[0],O2[0]).subs(Omega[1],O2[1])
        -1/w + exp(exp(p)*exp(-exp(-p))/(1 - 1/p))

    is really not what we want!

    So we do it the hard way and keep track of all the things we potentially
    want to substitute by dummy variables. Consider the expression::

        exp(x - exp(-x)) + exp(x) + x.

    The mrv set is {exp(x), exp(-x), exp(x - exp(-x))}.
    We introduce corresponding dummy variables d1, d2, d3 and rewrite::

        d3 + d1 + x.

    This class first of all keeps track of the mapping expr->variable, i.e.
    will at this stage be a dictionary::

        {exp(x): d1, exp(-x): d2, exp(x - exp(-x)): d3}.

    [It turns out to be more convenient this way round.]
    But sometimes expressions in the mrv set have other expressions from the
    mrv set as subexpressions, and we need to keep track of that as well. In
    this case, d3 is really exp(x - d2), so rewrites at this stage is::

        {d3: exp(x-d2)}.

    The function rewrite uses all this information to correctly rewrite our
    expression in terms of w. In this case w can be chosen to be exp(-x),
    i.e. d2. The correct rewriting then is::

        exp(-w)/w + 1/w + x.
    """
    def __init__(self):
        self.rewrites = {}

    def __repr__(self):
        return super().__repr__() + ', ' + self.rewrites.__repr__()

    def __getitem__(self, key):
        if key not in self:
            self[key] = Dummy()
        return dict.__getitem__(self, key)

    def do_subs(self, e):
        """Substitute the variables with expressions"""
        for expr, var in self.items():
            e = e.xreplace({var: expr})
        return e

    def meets(self, s2):
        """Tell whether or not self and s2 have non-empty intersection"""
        return set(self.keys()).intersection(list(s2.keys())) != set()

    def union(self, s2, exps=None):
        """Compute the union of self and s2, adjusting exps"""
        res = self.copy()
        tr = {}
        for expr, var in s2.items():
            if expr in self:
                if exps:
                    exps = exps.xreplace({var: res[expr]})
                tr[var] = res[expr]
            else:
                res[expr] = var
        for var, rewr in s2.rewrites.items():
            res.rewrites[var] = rewr.xreplace(tr)
        return res, exps

    def copy(self):
        """Create a shallow copy of SubsSet"""
        r = SubsSet()
        r.rewrites = self.rewrites.copy()
        for expr, var in self.items():
            r[expr] = var
        return r


@debug
def mrv(e, x):
    """Returns a SubsSet of most rapidly varying (mrv) subexpressions of 'e',
       and e rewritten in terms of these"""
    from sympy.simplify.powsimp import powsimp
    e = powsimp(e, deep=True, combine='exp')
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")
    if not e.has(x):
        return SubsSet(), e
    elif e == x:
        s = SubsSet()
        return s, s[x]
    elif e.is_Mul or e.is_Add:
        i, d = e.as_independent(x)  # throw away x-independent terms
        if d.func != e.func:
            s, expr = mrv(d, x)
            return s, e.func(i, expr)
        a, b = d.as_two_terms()
        s1, e1 = mrv(a, x)
        s2, e2 = mrv(b, x)
        return mrv_max1(s1, s2, e.func(i, e1, e2), x)
    elif e.is_Pow and e.base != S.Exp1:
        e1 = S.One
        while e.is_Pow:
            b1 = e.base
            e1 *= e.exp
            e = b1
        if b1 == 1:
            return SubsSet(), b1
        if e1.has(x):
            return mrv(exp(e1*log(b1)), x)
        else:
            s, expr = mrv(b1, x)
            return s, expr**e1
    elif isinstance(e, log):
        s, expr = mrv(e.args[0], x)
        return s, log(expr)
    elif isinstance(e, exp) or (e.is_Pow and e.base == S.Exp1):
        # We know from the theory of this algorithm that exp(log(...)) may always
        # be simplified here, and doing so is vital for termination.
        if isinstance(e.exp, log):
            return mrv(e.exp.args[0], x)
        # if a product has an infinite factor the result will be
        # infinite if there is no zero, otherwise NaN; here, we
        # consider the result infinite if any factor is infinite
        li = limitinf(e.exp, x)
        if any(_.is_infinite for _ in Mul.make_args(li)):
            s1 = SubsSet()
            e1 = s1[e]
            s2, e2 = mrv(e.exp, x)
            su = s1.union(s2)[0]
            su.rewrites[e1] = exp(e2)
            return mrv_max3(s1, e1, s2, exp(e2), su, e1, x)
        else:
            s, expr = mrv(e.exp, x)
            return s, exp(expr)
    elif isinstance(e, AppliedUndef):
        raise ValueError("MRV set computation for UndefinedFunction is not allowed")
    elif e.is_Function:
        l = [mrv(a, x) for a in e.args]
        l2 = [s for (s, _) in l if s != SubsSet()]
        if len(l2) != 1:
            # e.g. something like BesselJ(x, x)
            raise NotImplementedError("MRV set computation for functions in"
                                      " several variables not implemented.")
        s, ss = l2[0], SubsSet()
        args = [ss.do_subs(x[1]) for x in l]
        return s, e.func(*args)
    elif e.is_Derivative:
        raise NotImplementedError("MRV set computation for derivatives"
                                  " not implemented yet.")
    raise NotImplementedError(
        "Don't know how to calculate the mrv of '%s'" % e)


def mrv_max3(f, expsf, g, expsg, union, expsboth, x):
    """
    Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. max() compares (two elements of)
    f and g and returns either (f, expsf) [if f is larger], (g, expsg)
    [if g is larger] or (union, expsboth) [if f, g are of the same class].
    """
    if not isinstance(f, SubsSet):
        raise TypeError("f should be an instance of SubsSet")
    if not isinstance(g, SubsSet):
        raise TypeError("g should be an instance of SubsSet")
    if f == SubsSet():
        return g, expsg
    elif g == SubsSet():
        return f, expsf
    elif f.meets(g):
        return union, expsboth

    c = compare(list(f.keys())[0], list(g.keys())[0], x)
    if c == ">":
        return f, expsf
    elif c == "<":
        return g, expsg
    else:
        if c != "=":
            raise ValueError("c should be =")
        return union, expsboth


def mrv_max1(f, g, exps, x):
    """Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. mrv_max1() compares (two elements of)
    f and g and returns the set, which is in the higher comparability class
    of the union of both, if they have the same order of variation.
    Also returns exps, with the appropriate substitutions made.
    """
    u, b = f.union(g, exps)
    return mrv_max3(f, g.do_subs(exps), g, f.do_subs(exps),
                    u, b, x)


@debug
@cacheit
@timeit
def sign(e, x):
    """
    Returns a sign of an expression e(x) for x->oo.

    ::

        e >  0 for x sufficiently large ...  1
        e == 0 for x sufficiently large ...  0
        e <  0 for x sufficiently large ... -1

    The result of this function is currently undefined if e changes sign
    arbitrarily often for arbitrarily large x (e.g. sin(x)).

    Note that this returns zero only if e is *constantly* zero
    for x sufficiently large. [If e is constant, of course, this is just
    the same thing as the sign of e.]
    """
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")

    if e.is_positive:
        return 1
    elif e.is_negative:
        return -1
    elif e.is_zero:
        return 0

    elif not e.has(x):
        from sympy.simplify import logcombine
        e = logcombine(e)
        return _sign(e)
    elif e == x:
        return 1
    elif e.is_Mul:
        a, b = e.as_two_terms()
        sa = sign(a, x)
        if not sa:
            return 0
        return sa * sign(b, x)
    elif isinstance(e, exp):
        return 1
    elif e.is_Pow:
        if e.base == S.Exp1:
            return 1
        s = sign(e.base, x)
        if s == 1:
            return 1
        if e.exp.is_Integer:
            return s**e.exp
    elif isinstance(e, log) and e.args[0].is_positive:
        return sign(e.args[0] - 1, x)

    # if all else fails, do it the hard way
    c0, e0 = mrv_leadterm(e, x)
    return sign(c0, x)


@debug
@timeit
@cacheit
def limitinf(e, x):
    """Limit e(x) for x-> oo."""
    # rewrite e in terms of tractable functions only

    old = e
    if not e.has(x):
        return e  # e is a constant
    from sympy.simplify.powsimp import powdenest
    from sympy.calculus.util import AccumBounds
    if e.has(Order):
        e = e.expand().removeO()
    if not x.is_positive or x.is_integer:
        # We make sure that x.is_positive is True and x.is_integer is None
        # so we get all the correct mathematical behavior from the expression.
        # We need a fresh variable.
        p = Dummy('p', positive=True)
        e = e.subs(x, p)
        x = p
    e = e.rewrite('tractable', deep=True, limitvar=x)
    e = powdenest(e)
    if isinstance(e, AccumBounds):
        if mrv_leadterm(e.min, x) != mrv_leadterm(e.max, x):
            raise NotImplementedError
        c0, e0 = mrv_leadterm(e.min, x)
    else:
        c0, e0 = mrv_leadterm(e, x)
    sig = sign(e0, x)
    if sig == 1:
        return S.Zero  # e0>0: lim f = 0
    elif sig == -1:  # e0<0: lim f = +-oo (the sign depends on the sign of c0)
        if c0.match(I*Wild("a", exclude=[I])):
            return c0*oo
        s = sign(c0, x)
        # the leading term shouldn't be 0:
        if s == 0:
            raise ValueError("Leading term should not be 0")
        return s*oo
    elif sig == 0:
        if c0 == old:
            c0 = c0.cancel()
        return limitinf(c0, x)  # e0=0: lim f = lim c0
    else:
        raise ValueError("{} could not be evaluated".format(sig))


def moveup2(s, x):
    r = SubsSet()
    for expr, var in s.items():
        r[expr.xreplace({x: exp(x)})] = var
    for var, expr in s.rewrites.items():
        r.rewrites[var] = s.rewrites[var].xreplace({x: exp(x)})
    return r


def moveup(l, x):
    return [e.xreplace({x: exp(x)}) for e in l]


@debug
@timeit
@cacheit
def mrv_leadterm(e, x):
    """Returns (c0, e0) for e."""
    Omega = SubsSet()
    if not e.has(x):
        return (e, S.Zero)
    if Omega == SubsSet():
        Omega, exps = mrv(e, x)
    if not Omega:
        # e really does not depend on x after simplification
        return exps, S.Zero
    if x in Omega:
        # move the whole omega up (exponentiate each term):
        Omega_up = moveup2(Omega, x)
        exps_up = moveup([exps], x)[0]
        # NOTE: there is no need to move this down!
        Omega = Omega_up
        exps = exps_up
    #
    # The positive dummy, w, is used here so log(w*2) etc. will expand;
    # a unique dummy is needed in this algorithm
    #
    # For limits of complex functions, the algorithm would have to be
    # improved, or just find limits of Re and Im components separately.
    #
    w = Dummy("w", positive=True)
    f, logw = rewrite(exps, Omega, x, w)

    # Ensure expressions of the form exp(log(...)) don't get simplified automatically in the previous steps.
    # see: https://github.com/sympy/sympy/issues/15323#issuecomment-478639399
    f = f.replace(lambda f: f.is_Pow and f.has(x), lambda f: exp(log(f.base)*f.exp))

    try:
        lt = f.leadterm(w, logx=logw)
    except (NotImplementedError, PoleError, ValueError):
        n0 = 1
        _series = Order(1)
        incr = S.One
        while _series.is_Order:
            _series = f._eval_nseries(w, n=n0+incr, logx=logw)
            incr *= 2
        series = _series.expand().removeO()
        try:
            lt = series.leadterm(w, logx=logw)
        except (NotImplementedError, PoleError, ValueError):
            lt = f.as_coeff_exponent(w)
            if lt[0].has(w):
                base = f.as_base_exp()[0].as_coeff_exponent(w)
                ex = f.as_base_exp()[1]
                lt = (base[0]**ex, base[1]*ex)
    return (lt[0].subs(log(w), logw), lt[1])


def build_expression_tree(Omega, rewrites):
    r""" Helper function for rewrite.

    We need to sort Omega (mrv set) so that we replace an expression before
    we replace any expression in terms of which it has to be rewritten::

        e1 ---> e2 ---> e3
                 \
                  -> e4

    Here we can do e1, e2, e3, e4 or e1, e2, e4, e3.
    To do this we assemble the nodes into a tree, and sort them by height.

    This function builds the tree, rewrites then sorts the nodes.
    """
    class Node:
        def __init__(self):
            self.before = []
            self.expr = None
            self.var = None
        def ht(self):
            return reduce(lambda x, y: x + y,
                          [x.ht() for x in self.before], 1)
    nodes = {}
    for expr, v in Omega:
        n = Node()
        n.var = v
        n.expr = expr
        nodes[v] = n
    for _, v in Omega:
        if v in rewrites:
            n = nodes[v]
            r = rewrites[v]
            for _, v2 in Omega:
                if r.has(v2):
                    n.before.append(nodes[v2])

    return nodes


@debug
@timeit
def rewrite(e, Omega, x, wsym):
    """e(x) ... the function
    Omega ... the mrv set
    wsym ... the symbol which is going to be used for w

    Returns the rewritten e in terms of w and log(w). See test_rewrite1()
    for examples and correct results.
    """

    from sympy import AccumBounds
    if not isinstance(Omega, SubsSet):
        raise TypeError("Omega should be an instance of SubsSet")
    if len(Omega) == 0:
        raise ValueError("Length cannot be 0")
    # all items in Omega must be exponentials
    for t in Omega.keys():
        if not isinstance(t, exp):
            raise ValueError("Value should be exp")
    rewrites = Omega.rewrites
    Omega = list(Omega.items())

    nodes = build_expression_tree(Omega, rewrites)
    Omega.sort(key=lambda x: nodes[x[1]].ht(), reverse=True)

    # make sure we know the sign of each exp() term; after the loop,
    # g is going to be the "w" - the simplest one in the mrv set
    for g, _ in Omega:
        sig = sign(g.exp, x)
        if sig != 1 and sig != -1 and not sig.has(AccumBounds):
            raise NotImplementedError('Result depends on the sign of %s' % sig)
    if sig == 1:
        wsym = 1/wsym  # if g goes to oo, substitute 1/w
    # O2 is a list, which results by rewriting each item in Omega using "w"
    O2 = []
    denominators = []
    for f, var in Omega:
        c = limitinf(f.exp/g.exp, x)
        if c.is_Rational:
            denominators.append(c.q)
        arg = f.exp
        if var in rewrites:
            if not isinstance(rewrites[var], exp):
                raise ValueError("Value should be exp")
            arg = rewrites[var].args[0]
        O2.append((var, exp((arg - c*g.exp))*wsym**c))

    # Remember that Omega contains subexpressions of "e". So now we find
    # them in "e" and substitute them for our rewriting, stored in O2

    # the following powsimp is necessary to automatically combine exponentials,
    # so that the .xreplace() below succeeds:
    # TODO this should not be necessary
    from sympy.simplify.powsimp import powsimp
    f = powsimp(e, deep=True, combine='exp')
    for a, b in O2:
        f = f.xreplace({a: b})

    for _, var in Omega:
        assert not f.has(var)

    # finally compute the logarithm of w (logw).
    logw = g.exp
    if sig == 1:
        logw = -logw  # log(w)->log(1/w)=-log(w)

    # Some parts of SymPy have difficulty computing series expansions with
    # non-integral exponents. The following heuristic improves the situation:
    exponent = reduce(ilcm, denominators, 1)
    f = f.subs({wsym: wsym**exponent})
    logw /= exponent

    # bottom_up function is required for a specific case - when f is
    # -exp(p/(p + 1)) + exp(-p**2/(p + 1) + p). No current simplification
    # methods reduce this to 0 while not expanding polynomials.
    f = bottom_up(f, lambda w: getattr(w, 'normal', lambda: w)())

    return f, logw


def gruntz(e, z, z0, dir="+"):
    """
    Compute the limit of e(z) at the point z0 using the Gruntz algorithm.

    Explanation
    ===========

    ``z0`` can be any expression, including oo and -oo.

    For ``dir="+"`` (default) it calculates the limit from the right
    (z->z0+) and for ``dir="-"`` the limit from the left (z->z0-). For infinite z0
    (oo or -oo), the dir argument does not matter.

    This algorithm is fully described in the module docstring in the gruntz.py
    file. It relies heavily on the series expansion. Most frequently, gruntz()
    is only used if the faster limit() function (which uses heuristics) fails.
    """
    if not z.is_symbol:
        raise NotImplementedError("Second argument must be a Symbol")

    # convert all limits to the limit z->oo; sign of z is handled in limitinf
    r = None
    if z0 in (oo, I*oo):
        e0 = e
    elif z0 in (-oo, -I*oo):
        e0 = e.subs(z, -z)
    else:
        if str(dir) == "-":
            e0 = e.subs(z, z0 - 1/z)
        elif str(dir) == "+":
            e0 = e.subs(z, z0 + 1/z)
        else:
            raise NotImplementedError("dir must be '+' or '-'")

    r = limitinf(e0, z)

    # This is a bit of a heuristic for nice results... we always rewrite
    # tractable functions in terms of familiar intractable ones.
    # It might be nicer to rewrite the exactly to what they were initially,
    # but that would take some work to implement.
    return r.rewrite('intractable', deep=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\input_.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Input
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class TouchPoint:
    #: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    x: float

    #: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to
    #: the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    y: float

    #: X radius of the touch area (default: 1.0).
    radius_x: typing.Optional[float] = None

    #: Y radius of the touch area (default: 1.0).
    radius_y: typing.Optional[float] = None

    #: Rotation angle (default: 0.0).
    rotation_angle: typing.Optional[float] = None

    #: Force (default: 1.0).
    force: typing.Optional[float] = None

    #: The normalized tangential pressure, which has a range of [-1,1] (default: 0).
    tangential_pressure: typing.Optional[float] = None

    #: The plane angle between the Y-Z plane and the plane containing both the stylus axis and the Y axis, in degrees of the range [-90,90], a positive tiltX is to the right (default: 0)
    tilt_x: typing.Optional[float] = None

    #: The plane angle between the X-Z plane and the plane containing both the stylus axis and the X axis, in degrees of the range [-90,90], a positive tiltY is towards the user (default: 0).
    tilt_y: typing.Optional[float] = None

    #: The clockwise rotation of a pen stylus around its own major axis, in degrees in the range [0,359] (default: 0).
    twist: typing.Optional[int] = None

    #: Identifier used to track touch sources between events, must be unique within an event.
    id_: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['x'] = self.x
        json['y'] = self.y
        if self.radius_x is not None:
            json['radiusX'] = self.radius_x
        if self.radius_y is not None:
            json['radiusY'] = self.radius_y
        if self.rotation_angle is not None:
            json['rotationAngle'] = self.rotation_angle
        if self.force is not None:
            json['force'] = self.force
        if self.tangential_pressure is not None:
            json['tangentialPressure'] = self.tangential_pressure
        if self.tilt_x is not None:
            json['tiltX'] = self.tilt_x
        if self.tilt_y is not None:
            json['tiltY'] = self.tilt_y
        if self.twist is not None:
            json['twist'] = self.twist
        if self.id_ is not None:
            json['id'] = self.id_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            x=float(json['x']),
            y=float(json['y']),
            radius_x=float(json['radiusX']) if 'radiusX' in json else None,
            radius_y=float(json['radiusY']) if 'radiusY' in json else None,
            rotation_angle=float(json['rotationAngle']) if 'rotationAngle' in json else None,
            force=float(json['force']) if 'force' in json else None,
            tangential_pressure=float(json['tangentialPressure']) if 'tangentialPressure' in json else None,
            tilt_x=float(json['tiltX']) if 'tiltX' in json else None,
            tilt_y=float(json['tiltY']) if 'tiltY' in json else None,
            twist=int(json['twist']) if 'twist' in json else None,
            id_=float(json['id']) if 'id' in json else None,
        )


class GestureSourceType(enum.Enum):
    DEFAULT = "default"
    TOUCH = "touch"
    MOUSE = "mouse"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class MouseButton(enum.Enum):
    NONE = "none"
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"
    BACK = "back"
    FORWARD = "forward"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class TimeSinceEpoch(float):
    '''
    UTC time in seconds, counted from January 1, 1970.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> TimeSinceEpoch:
        return cls(json)

    def __repr__(self):
        return 'TimeSinceEpoch({})'.format(super().__repr__())


@dataclass
class DragDataItem:
    #: Mime type of the dragged data.
    mime_type: str

    #: Depending of the value of ``mimeType``, it contains the dragged link,
    #: text, HTML markup or any other data.
    data: str

    #: Title associated with a link. Only valid when ``mimeType`` == "text/uri-list".
    title: typing.Optional[str] = None

    #: Stores the base URL for the contained markup. Only valid when ``mimeType``
    #: == "text/html".
    base_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['mimeType'] = self.mime_type
        json['data'] = self.data
        if self.title is not None:
            json['title'] = self.title
        if self.base_url is not None:
            json['baseURL'] = self.base_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            mime_type=str(json['mimeType']),
            data=str(json['data']),
            title=str(json['title']) if 'title' in json else None,
            base_url=str(json['baseURL']) if 'baseURL' in json else None,
        )


@dataclass
class DragData:
    items: typing.List[DragDataItem]

    #: Bit field representing allowed drag operations. Copy = 1, Link = 2, Move = 16
    drag_operations_mask: int

    #: List of filenames that should be included when dropping
    files: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['items'] = [i.to_json() for i in self.items]
        json['dragOperationsMask'] = self.drag_operations_mask
        if self.files is not None:
            json['files'] = [i for i in self.files]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            items=[DragDataItem.from_json(i) for i in json['items']],
            drag_operations_mask=int(json['dragOperationsMask']),
            files=[str(i) for i in json['files']] if 'files' in json else None,
        )


def dispatch_drag_event(
        type_: str,
        x: float,
        y: float,
        data: DragData,
        modifiers: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a drag event into the page.

    **EXPERIMENTAL**

    :param type_: Type of the drag event.
    :param x: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    :param y: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    :param data:
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    params['data'] = data.to_json()
    if modifiers is not None:
        params['modifiers'] = modifiers
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchDragEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_key_event(
        type_: str,
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        text: typing.Optional[str] = None,
        unmodified_text: typing.Optional[str] = None,
        key_identifier: typing.Optional[str] = None,
        code: typing.Optional[str] = None,
        key: typing.Optional[str] = None,
        windows_virtual_key_code: typing.Optional[int] = None,
        native_virtual_key_code: typing.Optional[int] = None,
        auto_repeat: typing.Optional[bool] = None,
        is_keypad: typing.Optional[bool] = None,
        is_system_key: typing.Optional[bool] = None,
        location: typing.Optional[int] = None,
        commands: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a key event to the page.

    :param type_: Type of the key event.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    :param text: *(Optional)* Text as generated by processing a virtual key code with a keyboard layout. Not needed for for ```keyUp```` and ````rawKeyDown```` events (default: "")
    :param unmodified_text: *(Optional)* Text that would have been generated by the keyboard if no modifiers were pressed (except for shift). Useful for shortcut (accelerator) key handling (default: "").
    :param key_identifier: *(Optional)* Unique key identifier (e.g., 'U+0041') (default: "").
    :param code: *(Optional)* Unique DOM defined string value for each physical key (e.g., 'KeyA') (default: "").
    :param key: *(Optional)* Unique DOM defined string value describing the meaning of the key in the context of active modifiers, keyboard layout, etc (e.g., 'AltGr') (default: "").
    :param windows_virtual_key_code: *(Optional)* Windows virtual key code (default: 0).
    :param native_virtual_key_code: *(Optional)* Native virtual key code (default: 0).
    :param auto_repeat: *(Optional)* Whether the event was generated from auto repeat (default: false).
    :param is_keypad: *(Optional)* Whether the event was generated from the keypad (default: false).
    :param is_system_key: *(Optional)* Whether the event was a system key event (default: false).
    :param location: *(Optional)* Whether the event was from the left or right side of the keyboard. 1=Left, 2=Right (default: 0).
    :param commands: **(EXPERIMENTAL)** *(Optional)* Editing commands to send with the key event (e.g., 'selectAll') (default: []). These are related to but not equal the command names used in ````document.execCommand``` and NSStandardKeyBindingResponding. See https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/core/editing/commands/editor_command_names.h for valid command names.
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if text is not None:
        params['text'] = text
    if unmodified_text is not None:
        params['unmodifiedText'] = unmodified_text
    if key_identifier is not None:
        params['keyIdentifier'] = key_identifier
    if code is not None:
        params['code'] = code
    if key is not None:
        params['key'] = key
    if windows_virtual_key_code is not None:
        params['windowsVirtualKeyCode'] = windows_virtual_key_code
    if native_virtual_key_code is not None:
        params['nativeVirtualKeyCode'] = native_virtual_key_code
    if auto_repeat is not None:
        params['autoRepeat'] = auto_repeat
    if is_keypad is not None:
        params['isKeypad'] = is_keypad
    if is_system_key is not None:
        params['isSystemKey'] = is_system_key
    if location is not None:
        params['location'] = location
    if commands is not None:
        params['commands'] = [i for i in commands]
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchKeyEvent',
        'params': params,
    }
    json = yield cmd_dict


def insert_text(
        text: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method emulates inserting text that doesn't come from a key press,
    for example an emoji keyboard or an IME.

    **EXPERIMENTAL**

    :param text: The text to insert.
    '''
    params: T_JSON_DICT = dict()
    params['text'] = text
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.insertText',
        'params': params,
    }
    json = yield cmd_dict


def ime_set_composition(
        text: str,
        selection_start: int,
        selection_end: int,
        replacement_start: typing.Optional[int] = None,
        replacement_end: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method sets the current candidate text for IME.
    Use imeCommitComposition to commit the final text.
    Use imeSetComposition with empty string as text to cancel composition.

    **EXPERIMENTAL**

    :param text: The text to insert
    :param selection_start: selection start
    :param selection_end: selection end
    :param replacement_start: *(Optional)* replacement start
    :param replacement_end: *(Optional)* replacement end
    '''
    params: T_JSON_DICT = dict()
    params['text'] = text
    params['selectionStart'] = selection_start
    params['selectionEnd'] = selection_end
    if replacement_start is not None:
        params['replacementStart'] = replacement_start
    if replacement_end is not None:
        params['replacementEnd'] = replacement_end
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.imeSetComposition',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_mouse_event(
        type_: str,
        x: float,
        y: float,
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        button: typing.Optional[MouseButton] = None,
        buttons: typing.Optional[int] = None,
        click_count: typing.Optional[int] = None,
        force: typing.Optional[float] = None,
        tangential_pressure: typing.Optional[float] = None,
        tilt_x: typing.Optional[float] = None,
        tilt_y: typing.Optional[float] = None,
        twist: typing.Optional[int] = None,
        delta_x: typing.Optional[float] = None,
        delta_y: typing.Optional[float] = None,
        pointer_type: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a mouse event to the page.

    :param type_: Type of the mouse event.
    :param x: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    :param y: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    :param button: *(Optional)* Mouse button (default: "none").
    :param buttons: *(Optional)* A number indicating which buttons are pressed on the mouse when a mouse event is triggered. Left=1, Right=2, Middle=4, Back=8, Forward=16, None=0.
    :param click_count: *(Optional)* Number of times the mouse button was clicked (default: 0).
    :param force: **(EXPERIMENTAL)** *(Optional)* The normalized pressure, which has a range of [0,1] (default: 0).
    :param tangential_pressure: **(EXPERIMENTAL)** *(Optional)* The normalized tangential pressure, which has a range of [-1,1] (default: 0).
    :param tilt_x: *(Optional)* The plane angle between the Y-Z plane and the plane containing both the stylus axis and the Y axis, in degrees of the range [-90,90], a positive tiltX is to the right (default: 0).
    :param tilt_y: *(Optional)* The plane angle between the X-Z plane and the plane containing both the stylus axis and the X axis, in degrees of the range [-90,90], a positive tiltY is towards the user (default: 0).
    :param twist: **(EXPERIMENTAL)** *(Optional)* The clockwise rotation of a pen stylus around its own major axis, in degrees in the range [0,359] (default: 0).
    :param delta_x: *(Optional)* X delta in CSS pixels for mouse wheel event (default: 0).
    :param delta_y: *(Optional)* Y delta in CSS pixels for mouse wheel event (default: 0).
    :param pointer_type: *(Optional)* Pointer type (default: "mouse").
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if button is not None:
        params['button'] = button.to_json()
    if buttons is not None:
        params['buttons'] = buttons
    if click_count is not None:
        params['clickCount'] = click_count
    if force is not None:
        params['force'] = force
    if tangential_pressure is not None:
        params['tangentialPressure'] = tangential_pressure
    if tilt_x is not None:
        params['tiltX'] = tilt_x
    if tilt_y is not None:
        params['tiltY'] = tilt_y
    if twist is not None:
        params['twist'] = twist
    if delta_x is not None:
        params['deltaX'] = delta_x
    if delta_y is not None:
        params['deltaY'] = delta_y
    if pointer_type is not None:
        params['pointerType'] = pointer_type
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchMouseEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_touch_event(
        type_: str,
        touch_points: typing.List[TouchPoint],
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a touch event to the page.

    :param type_: Type of the touch event. TouchEnd and TouchCancel must not contain any touch points, while TouchStart and TouchMove must contains at least one.
    :param touch_points: Active touch points on the touch device. One event per any changed point (compared to previous touch event in a sequence) is generated, emulating pressing/moving/releasing points one by one.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['touchPoints'] = [i.to_json() for i in touch_points]
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchTouchEvent',
        'params': params,
    }
    json = yield cmd_dict


def cancel_dragging() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Cancels any active dragging in the page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.cancelDragging',
    }
    json = yield cmd_dict


def emulate_touch_from_mouse_event(
        type_: str,
        x: int,
        y: int,
        button: MouseButton,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        delta_x: typing.Optional[float] = None,
        delta_y: typing.Optional[float] = None,
        modifiers: typing.Optional[int] = None,
        click_count: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Emulates touch event from the mouse event parameters.

    **EXPERIMENTAL**

    :param type_: Type of the mouse event.
    :param x: X coordinate of the mouse pointer in DIP.
    :param y: Y coordinate of the mouse pointer in DIP.
    :param button: Mouse button. Only "none", "left", "right" are supported.
    :param timestamp: *(Optional)* Time at which the event occurred (default: current time).
    :param delta_x: *(Optional)* X delta in DIP for mouse wheel event (default: 0).
    :param delta_y: *(Optional)* Y delta in DIP for mouse wheel event (default: 0).
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param click_count: *(Optional)* Number of times the mouse button was clicked (default: 0).
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    params['button'] = button.to_json()
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if delta_x is not None:
        params['deltaX'] = delta_x
    if delta_y is not None:
        params['deltaY'] = delta_y
    if modifiers is not None:
        params['modifiers'] = modifiers
    if click_count is not None:
        params['clickCount'] = click_count
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.emulateTouchFromMouseEvent',
        'params': params,
    }
    json = yield cmd_dict


def set_ignore_input_events(
        ignore: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Ignores input events (useful while auditing page).

    :param ignore: Ignores input events processing when set to true.
    '''
    params: T_JSON_DICT = dict()
    params['ignore'] = ignore
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.setIgnoreInputEvents',
        'params': params,
    }
    json = yield cmd_dict


def set_intercept_drags(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Prevents default drag and drop behavior and instead emits ``Input.dragIntercepted`` events.
    Drag and drop behavior can be directly controlled via ``Input.dispatchDragEvent``.

    **EXPERIMENTAL**

    :param enabled:
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.setInterceptDrags',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_pinch_gesture(
        x: float,
        y: float,
        scale_factor: float,
        relative_speed: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a pinch gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param scale_factor: Relative scale factor after zooming (>1.0 zooms in, <1.0 zooms out).
    :param relative_speed: *(Optional)* Relative pointer speed in pixels per second (default: 800).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    params['scaleFactor'] = scale_factor
    if relative_speed is not None:
        params['relativeSpeed'] = relative_speed
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizePinchGesture',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_scroll_gesture(
        x: float,
        y: float,
        x_distance: typing.Optional[float] = None,
        y_distance: typing.Optional[float] = None,
        x_overscroll: typing.Optional[float] = None,
        y_overscroll: typing.Optional[float] = None,
        prevent_fling: typing.Optional[bool] = None,
        speed: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None,
        repeat_count: typing.Optional[int] = None,
        repeat_delay_ms: typing.Optional[int] = None,
        interaction_marker_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a scroll gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param x_distance: *(Optional)* The distance to scroll along the X axis (positive to scroll left).
    :param y_distance: *(Optional)* The distance to scroll along the Y axis (positive to scroll up).
    :param x_overscroll: *(Optional)* The number of additional pixels to scroll back along the X axis, in addition to the given distance.
    :param y_overscroll: *(Optional)* The number of additional pixels to scroll back along the Y axis, in addition to the given distance.
    :param prevent_fling: *(Optional)* Prevent fling (default: true).
    :param speed: *(Optional)* Swipe speed in pixels per second (default: 800).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    :param repeat_count: *(Optional)* The number of times to repeat the gesture (default: 0).
    :param repeat_delay_ms: *(Optional)* The number of milliseconds delay between each repeat. (default: 250).
    :param interaction_marker_name: *(Optional)* The name of the interaction markers to generate, if not empty (default: "").
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if x_distance is not None:
        params['xDistance'] = x_distance
    if y_distance is not None:
        params['yDistance'] = y_distance
    if x_overscroll is not None:
        params['xOverscroll'] = x_overscroll
    if y_overscroll is not None:
        params['yOverscroll'] = y_overscroll
    if prevent_fling is not None:
        params['preventFling'] = prevent_fling
    if speed is not None:
        params['speed'] = speed
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    if repeat_count is not None:
        params['repeatCount'] = repeat_count
    if repeat_delay_ms is not None:
        params['repeatDelayMs'] = repeat_delay_ms
    if interaction_marker_name is not None:
        params['interactionMarkerName'] = interaction_marker_name
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizeScrollGesture',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_tap_gesture(
        x: float,
        y: float,
        duration: typing.Optional[int] = None,
        tap_count: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a tap gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param duration: *(Optional)* Duration between touchdown and touchup events in ms (default: 50).
    :param tap_count: *(Optional)* Number of times to perform the tap (e.g. 2 for double tap, default: 1).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if duration is not None:
        params['duration'] = duration
    if tap_count is not None:
        params['tapCount'] = tap_count
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizeTapGesture',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Input.dragIntercepted')
@dataclass
class DragIntercepted:
    '''
    **EXPERIMENTAL**

    Emitted only when ``Input.setInterceptDrags`` is enabled. Use this data with ``Input.dispatchDragEvent`` to
    restore normal drag and drop behavior.
    '''
    data: DragData

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DragIntercepted:
        return cls(
            data=DragData.from_json(json['data'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\input_.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Input
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class TouchPoint:
    #: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    x: float

    #: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to
    #: the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    y: float

    #: X radius of the touch area (default: 1.0).
    radius_x: typing.Optional[float] = None

    #: Y radius of the touch area (default: 1.0).
    radius_y: typing.Optional[float] = None

    #: Rotation angle (default: 0.0).
    rotation_angle: typing.Optional[float] = None

    #: Force (default: 1.0).
    force: typing.Optional[float] = None

    #: The normalized tangential pressure, which has a range of [-1,1] (default: 0).
    tangential_pressure: typing.Optional[float] = None

    #: The plane angle between the Y-Z plane and the plane containing both the stylus axis and the Y axis, in degrees of the range [-90,90], a positive tiltX is to the right (default: 0)
    tilt_x: typing.Optional[float] = None

    #: The plane angle between the X-Z plane and the plane containing both the stylus axis and the X axis, in degrees of the range [-90,90], a positive tiltY is towards the user (default: 0).
    tilt_y: typing.Optional[float] = None

    #: The clockwise rotation of a pen stylus around its own major axis, in degrees in the range [0,359] (default: 0).
    twist: typing.Optional[int] = None

    #: Identifier used to track touch sources between events, must be unique within an event.
    id_: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['x'] = self.x
        json['y'] = self.y
        if self.radius_x is not None:
            json['radiusX'] = self.radius_x
        if self.radius_y is not None:
            json['radiusY'] = self.radius_y
        if self.rotation_angle is not None:
            json['rotationAngle'] = self.rotation_angle
        if self.force is not None:
            json['force'] = self.force
        if self.tangential_pressure is not None:
            json['tangentialPressure'] = self.tangential_pressure
        if self.tilt_x is not None:
            json['tiltX'] = self.tilt_x
        if self.tilt_y is not None:
            json['tiltY'] = self.tilt_y
        if self.twist is not None:
            json['twist'] = self.twist
        if self.id_ is not None:
            json['id'] = self.id_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            x=float(json['x']),
            y=float(json['y']),
            radius_x=float(json['radiusX']) if 'radiusX' in json else None,
            radius_y=float(json['radiusY']) if 'radiusY' in json else None,
            rotation_angle=float(json['rotationAngle']) if 'rotationAngle' in json else None,
            force=float(json['force']) if 'force' in json else None,
            tangential_pressure=float(json['tangentialPressure']) if 'tangentialPressure' in json else None,
            tilt_x=float(json['tiltX']) if 'tiltX' in json else None,
            tilt_y=float(json['tiltY']) if 'tiltY' in json else None,
            twist=int(json['twist']) if 'twist' in json else None,
            id_=float(json['id']) if 'id' in json else None,
        )


class GestureSourceType(enum.Enum):
    DEFAULT = "default"
    TOUCH = "touch"
    MOUSE = "mouse"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class MouseButton(enum.Enum):
    NONE = "none"
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"
    BACK = "back"
    FORWARD = "forward"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class TimeSinceEpoch(float):
    '''
    UTC time in seconds, counted from January 1, 1970.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> TimeSinceEpoch:
        return cls(json)

    def __repr__(self):
        return 'TimeSinceEpoch({})'.format(super().__repr__())


@dataclass
class DragDataItem:
    #: Mime type of the dragged data.
    mime_type: str

    #: Depending of the value of ``mimeType``, it contains the dragged link,
    #: text, HTML markup or any other data.
    data: str

    #: Title associated with a link. Only valid when ``mimeType`` == "text/uri-list".
    title: typing.Optional[str] = None

    #: Stores the base URL for the contained markup. Only valid when ``mimeType``
    #: == "text/html".
    base_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['mimeType'] = self.mime_type
        json['data'] = self.data
        if self.title is not None:
            json['title'] = self.title
        if self.base_url is not None:
            json['baseURL'] = self.base_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            mime_type=str(json['mimeType']),
            data=str(json['data']),
            title=str(json['title']) if 'title' in json else None,
            base_url=str(json['baseURL']) if 'baseURL' in json else None,
        )


@dataclass
class DragData:
    items: typing.List[DragDataItem]

    #: Bit field representing allowed drag operations. Copy = 1, Link = 2, Move = 16
    drag_operations_mask: int

    #: List of filenames that should be included when dropping
    files: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['items'] = [i.to_json() for i in self.items]
        json['dragOperationsMask'] = self.drag_operations_mask
        if self.files is not None:
            json['files'] = [i for i in self.files]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            items=[DragDataItem.from_json(i) for i in json['items']],
            drag_operations_mask=int(json['dragOperationsMask']),
            files=[str(i) for i in json['files']] if 'files' in json else None,
        )


def dispatch_drag_event(
        type_: str,
        x: float,
        y: float,
        data: DragData,
        modifiers: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a drag event into the page.

    **EXPERIMENTAL**

    :param type_: Type of the drag event.
    :param x: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    :param y: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    :param data:
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    params['data'] = data.to_json()
    if modifiers is not None:
        params['modifiers'] = modifiers
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchDragEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_key_event(
        type_: str,
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        text: typing.Optional[str] = None,
        unmodified_text: typing.Optional[str] = None,
        key_identifier: typing.Optional[str] = None,
        code: typing.Optional[str] = None,
        key: typing.Optional[str] = None,
        windows_virtual_key_code: typing.Optional[int] = None,
        native_virtual_key_code: typing.Optional[int] = None,
        auto_repeat: typing.Optional[bool] = None,
        is_keypad: typing.Optional[bool] = None,
        is_system_key: typing.Optional[bool] = None,
        location: typing.Optional[int] = None,
        commands: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a key event to the page.

    :param type_: Type of the key event.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    :param text: *(Optional)* Text as generated by processing a virtual key code with a keyboard layout. Not needed for for ```keyUp```` and ````rawKeyDown```` events (default: "")
    :param unmodified_text: *(Optional)* Text that would have been generated by the keyboard if no modifiers were pressed (except for shift). Useful for shortcut (accelerator) key handling (default: "").
    :param key_identifier: *(Optional)* Unique key identifier (e.g., 'U+0041') (default: "").
    :param code: *(Optional)* Unique DOM defined string value for each physical key (e.g., 'KeyA') (default: "").
    :param key: *(Optional)* Unique DOM defined string value describing the meaning of the key in the context of active modifiers, keyboard layout, etc (e.g., 'AltGr') (default: "").
    :param windows_virtual_key_code: *(Optional)* Windows virtual key code (default: 0).
    :param native_virtual_key_code: *(Optional)* Native virtual key code (default: 0).
    :param auto_repeat: *(Optional)* Whether the event was generated from auto repeat (default: false).
    :param is_keypad: *(Optional)* Whether the event was generated from the keypad (default: false).
    :param is_system_key: *(Optional)* Whether the event was a system key event (default: false).
    :param location: *(Optional)* Whether the event was from the left or right side of the keyboard. 1=Left, 2=Right (default: 0).
    :param commands: **(EXPERIMENTAL)** *(Optional)* Editing commands to send with the key event (e.g., 'selectAll') (default: []). These are related to but not equal the command names used in ````document.execCommand``` and NSStandardKeyBindingResponding. See https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/core/editing/commands/editor_command_names.h for valid command names.
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if text is not None:
        params['text'] = text
    if unmodified_text is not None:
        params['unmodifiedText'] = unmodified_text
    if key_identifier is not None:
        params['keyIdentifier'] = key_identifier
    if code is not None:
        params['code'] = code
    if key is not None:
        params['key'] = key
    if windows_virtual_key_code is not None:
        params['windowsVirtualKeyCode'] = windows_virtual_key_code
    if native_virtual_key_code is not None:
        params['nativeVirtualKeyCode'] = native_virtual_key_code
    if auto_repeat is not None:
        params['autoRepeat'] = auto_repeat
    if is_keypad is not None:
        params['isKeypad'] = is_keypad
    if is_system_key is not None:
        params['isSystemKey'] = is_system_key
    if location is not None:
        params['location'] = location
    if commands is not None:
        params['commands'] = [i for i in commands]
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchKeyEvent',
        'params': params,
    }
    json = yield cmd_dict


def insert_text(
        text: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method emulates inserting text that doesn't come from a key press,
    for example an emoji keyboard or an IME.

    **EXPERIMENTAL**

    :param text: The text to insert.
    '''
    params: T_JSON_DICT = dict()
    params['text'] = text
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.insertText',
        'params': params,
    }
    json = yield cmd_dict


def ime_set_composition(
        text: str,
        selection_start: int,
        selection_end: int,
        replacement_start: typing.Optional[int] = None,
        replacement_end: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method sets the current candidate text for IME.
    Use imeCommitComposition to commit the final text.
    Use imeSetComposition with empty string as text to cancel composition.

    **EXPERIMENTAL**

    :param text: The text to insert
    :param selection_start: selection start
    :param selection_end: selection end
    :param replacement_start: *(Optional)* replacement start
    :param replacement_end: *(Optional)* replacement end
    '''
    params: T_JSON_DICT = dict()
    params['text'] = text
    params['selectionStart'] = selection_start
    params['selectionEnd'] = selection_end
    if replacement_start is not None:
        params['replacementStart'] = replacement_start
    if replacement_end is not None:
        params['replacementEnd'] = replacement_end
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.imeSetComposition',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_mouse_event(
        type_: str,
        x: float,
        y: float,
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        button: typing.Optional[MouseButton] = None,
        buttons: typing.Optional[int] = None,
        click_count: typing.Optional[int] = None,
        force: typing.Optional[float] = None,
        tangential_pressure: typing.Optional[float] = None,
        tilt_x: typing.Optional[float] = None,
        tilt_y: typing.Optional[float] = None,
        twist: typing.Optional[int] = None,
        delta_x: typing.Optional[float] = None,
        delta_y: typing.Optional[float] = None,
        pointer_type: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a mouse event to the page.

    :param type_: Type of the mouse event.
    :param x: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    :param y: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    :param button: *(Optional)* Mouse button (default: "none").
    :param buttons: *(Optional)* A number indicating which buttons are pressed on the mouse when a mouse event is triggered. Left=1, Right=2, Middle=4, Back=8, Forward=16, None=0.
    :param click_count: *(Optional)* Number of times the mouse button was clicked (default: 0).
    :param force: **(EXPERIMENTAL)** *(Optional)* The normalized pressure, which has a range of [0,1] (default: 0).
    :param tangential_pressure: **(EXPERIMENTAL)** *(Optional)* The normalized tangential pressure, which has a range of [-1,1] (default: 0).
    :param tilt_x: *(Optional)* The plane angle between the Y-Z plane and the plane containing both the stylus axis and the Y axis, in degrees of the range [-90,90], a positive tiltX is to the right (default: 0).
    :param tilt_y: *(Optional)* The plane angle between the X-Z plane and the plane containing both the stylus axis and the X axis, in degrees of the range [-90,90], a positive tiltY is towards the user (default: 0).
    :param twist: **(EXPERIMENTAL)** *(Optional)* The clockwise rotation of a pen stylus around its own major axis, in degrees in the range [0,359] (default: 0).
    :param delta_x: *(Optional)* X delta in CSS pixels for mouse wheel event (default: 0).
    :param delta_y: *(Optional)* Y delta in CSS pixels for mouse wheel event (default: 0).
    :param pointer_type: *(Optional)* Pointer type (default: "mouse").
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if button is not None:
        params['button'] = button.to_json()
    if buttons is not None:
        params['buttons'] = buttons
    if click_count is not None:
        params['clickCount'] = click_count
    if force is not None:
        params['force'] = force
    if tangential_pressure is not None:
        params['tangentialPressure'] = tangential_pressure
    if tilt_x is not None:
        params['tiltX'] = tilt_x
    if tilt_y is not None:
        params['tiltY'] = tilt_y
    if twist is not None:
        params['twist'] = twist
    if delta_x is not None:
        params['deltaX'] = delta_x
    if delta_y is not None:
        params['deltaY'] = delta_y
    if pointer_type is not None:
        params['pointerType'] = pointer_type
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchMouseEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_touch_event(
        type_: str,
        touch_points: typing.List[TouchPoint],
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a touch event to the page.

    :param type_: Type of the touch event. TouchEnd and TouchCancel must not contain any touch points, while TouchStart and TouchMove must contains at least one.
    :param touch_points: Active touch points on the touch device. One event per any changed point (compared to previous touch event in a sequence) is generated, emulating pressing/moving/releasing points one by one.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['touchPoints'] = [i.to_json() for i in touch_points]
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchTouchEvent',
        'params': params,
    }
    json = yield cmd_dict


def cancel_dragging() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Cancels any active dragging in the page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.cancelDragging',
    }
    json = yield cmd_dict


def emulate_touch_from_mouse_event(
        type_: str,
        x: int,
        y: int,
        button: MouseButton,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        delta_x: typing.Optional[float] = None,
        delta_y: typing.Optional[float] = None,
        modifiers: typing.Optional[int] = None,
        click_count: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Emulates touch event from the mouse event parameters.

    **EXPERIMENTAL**

    :param type_: Type of the mouse event.
    :param x: X coordinate of the mouse pointer in DIP.
    :param y: Y coordinate of the mouse pointer in DIP.
    :param button: Mouse button. Only "none", "left", "right" are supported.
    :param timestamp: *(Optional)* Time at which the event occurred (default: current time).
    :param delta_x: *(Optional)* X delta in DIP for mouse wheel event (default: 0).
    :param delta_y: *(Optional)* Y delta in DIP for mouse wheel event (default: 0).
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param click_count: *(Optional)* Number of times the mouse button was clicked (default: 0).
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    params['button'] = button.to_json()
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if delta_x is not None:
        params['deltaX'] = delta_x
    if delta_y is not None:
        params['deltaY'] = delta_y
    if modifiers is not None:
        params['modifiers'] = modifiers
    if click_count is not None:
        params['clickCount'] = click_count
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.emulateTouchFromMouseEvent',
        'params': params,
    }
    json = yield cmd_dict


def set_ignore_input_events(
        ignore: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Ignores input events (useful while auditing page).

    :param ignore: Ignores input events processing when set to true.
    '''
    params: T_JSON_DICT = dict()
    params['ignore'] = ignore
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.setIgnoreInputEvents',
        'params': params,
    }
    json = yield cmd_dict


def set_intercept_drags(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Prevents default drag and drop behavior and instead emits ``Input.dragIntercepted`` events.
    Drag and drop behavior can be directly controlled via ``Input.dispatchDragEvent``.

    **EXPERIMENTAL**

    :param enabled:
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.setInterceptDrags',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_pinch_gesture(
        x: float,
        y: float,
        scale_factor: float,
        relative_speed: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a pinch gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param scale_factor: Relative scale factor after zooming (>1.0 zooms in, <1.0 zooms out).
    :param relative_speed: *(Optional)* Relative pointer speed in pixels per second (default: 800).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    params['scaleFactor'] = scale_factor
    if relative_speed is not None:
        params['relativeSpeed'] = relative_speed
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizePinchGesture',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_scroll_gesture(
        x: float,
        y: float,
        x_distance: typing.Optional[float] = None,
        y_distance: typing.Optional[float] = None,
        x_overscroll: typing.Optional[float] = None,
        y_overscroll: typing.Optional[float] = None,
        prevent_fling: typing.Optional[bool] = None,
        speed: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None,
        repeat_count: typing.Optional[int] = None,
        repeat_delay_ms: typing.Optional[int] = None,
        interaction_marker_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a scroll gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param x_distance: *(Optional)* The distance to scroll along the X axis (positive to scroll left).
    :param y_distance: *(Optional)* The distance to scroll along the Y axis (positive to scroll up).
    :param x_overscroll: *(Optional)* The number of additional pixels to scroll back along the X axis, in addition to the given distance.
    :param y_overscroll: *(Optional)* The number of additional pixels to scroll back along the Y axis, in addition to the given distance.
    :param prevent_fling: *(Optional)* Prevent fling (default: true).
    :param speed: *(Optional)* Swipe speed in pixels per second (default: 800).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    :param repeat_count: *(Optional)* The number of times to repeat the gesture (default: 0).
    :param repeat_delay_ms: *(Optional)* The number of milliseconds delay between each repeat. (default: 250).
    :param interaction_marker_name: *(Optional)* The name of the interaction markers to generate, if not empty (default: "").
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if x_distance is not None:
        params['xDistance'] = x_distance
    if y_distance is not None:
        params['yDistance'] = y_distance
    if x_overscroll is not None:
        params['xOverscroll'] = x_overscroll
    if y_overscroll is not None:
        params['yOverscroll'] = y_overscroll
    if prevent_fling is not None:
        params['preventFling'] = prevent_fling
    if speed is not None:
        params['speed'] = speed
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    if repeat_count is not None:
        params['repeatCount'] = repeat_count
    if repeat_delay_ms is not None:
        params['repeatDelayMs'] = repeat_delay_ms
    if interaction_marker_name is not None:
        params['interactionMarkerName'] = interaction_marker_name
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizeScrollGesture',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_tap_gesture(
        x: float,
        y: float,
        duration: typing.Optional[int] = None,
        tap_count: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a tap gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param duration: *(Optional)* Duration between touchdown and touchup events in ms (default: 50).
    :param tap_count: *(Optional)* Number of times to perform the tap (e.g. 2 for double tap, default: 1).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if duration is not None:
        params['duration'] = duration
    if tap_count is not None:
        params['tapCount'] = tap_count
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizeTapGesture',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Input.dragIntercepted')
@dataclass
class DragIntercepted:
    '''
    **EXPERIMENTAL**

    Emitted only when ``Input.setInterceptDrags`` is enabled. Use this data with ``Input.dispatchDragEvent`` to
    restore normal drag and drop behavior.
    '''
    data: DragData

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DragIntercepted:
        return cls(
            data=DragData.from_json(json['data'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\input_.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Input
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class TouchPoint:
    #: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    x: float

    #: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to
    #: the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    y: float

    #: X radius of the touch area (default: 1.0).
    radius_x: typing.Optional[float] = None

    #: Y radius of the touch area (default: 1.0).
    radius_y: typing.Optional[float] = None

    #: Rotation angle (default: 0.0).
    rotation_angle: typing.Optional[float] = None

    #: Force (default: 1.0).
    force: typing.Optional[float] = None

    #: The normalized tangential pressure, which has a range of [-1,1] (default: 0).
    tangential_pressure: typing.Optional[float] = None

    #: The plane angle between the Y-Z plane and the plane containing both the stylus axis and the Y axis, in degrees of the range [-90,90], a positive tiltX is to the right (default: 0)
    tilt_x: typing.Optional[float] = None

    #: The plane angle between the X-Z plane and the plane containing both the stylus axis and the X axis, in degrees of the range [-90,90], a positive tiltY is towards the user (default: 0).
    tilt_y: typing.Optional[float] = None

    #: The clockwise rotation of a pen stylus around its own major axis, in degrees in the range [0,359] (default: 0).
    twist: typing.Optional[int] = None

    #: Identifier used to track touch sources between events, must be unique within an event.
    id_: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['x'] = self.x
        json['y'] = self.y
        if self.radius_x is not None:
            json['radiusX'] = self.radius_x
        if self.radius_y is not None:
            json['radiusY'] = self.radius_y
        if self.rotation_angle is not None:
            json['rotationAngle'] = self.rotation_angle
        if self.force is not None:
            json['force'] = self.force
        if self.tangential_pressure is not None:
            json['tangentialPressure'] = self.tangential_pressure
        if self.tilt_x is not None:
            json['tiltX'] = self.tilt_x
        if self.tilt_y is not None:
            json['tiltY'] = self.tilt_y
        if self.twist is not None:
            json['twist'] = self.twist
        if self.id_ is not None:
            json['id'] = self.id_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            x=float(json['x']),
            y=float(json['y']),
            radius_x=float(json['radiusX']) if 'radiusX' in json else None,
            radius_y=float(json['radiusY']) if 'radiusY' in json else None,
            rotation_angle=float(json['rotationAngle']) if 'rotationAngle' in json else None,
            force=float(json['force']) if 'force' in json else None,
            tangential_pressure=float(json['tangentialPressure']) if 'tangentialPressure' in json else None,
            tilt_x=float(json['tiltX']) if 'tiltX' in json else None,
            tilt_y=float(json['tiltY']) if 'tiltY' in json else None,
            twist=int(json['twist']) if 'twist' in json else None,
            id_=float(json['id']) if 'id' in json else None,
        )


class GestureSourceType(enum.Enum):
    DEFAULT = "default"
    TOUCH = "touch"
    MOUSE = "mouse"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class MouseButton(enum.Enum):
    NONE = "none"
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"
    BACK = "back"
    FORWARD = "forward"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class TimeSinceEpoch(float):
    '''
    UTC time in seconds, counted from January 1, 1970.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> TimeSinceEpoch:
        return cls(json)

    def __repr__(self):
        return 'TimeSinceEpoch({})'.format(super().__repr__())


@dataclass
class DragDataItem:
    #: Mime type of the dragged data.
    mime_type: str

    #: Depending of the value of ``mimeType``, it contains the dragged link,
    #: text, HTML markup or any other data.
    data: str

    #: Title associated with a link. Only valid when ``mimeType`` == "text/uri-list".
    title: typing.Optional[str] = None

    #: Stores the base URL for the contained markup. Only valid when ``mimeType``
    #: == "text/html".
    base_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['mimeType'] = self.mime_type
        json['data'] = self.data
        if self.title is not None:
            json['title'] = self.title
        if self.base_url is not None:
            json['baseURL'] = self.base_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            mime_type=str(json['mimeType']),
            data=str(json['data']),
            title=str(json['title']) if 'title' in json else None,
            base_url=str(json['baseURL']) if 'baseURL' in json else None,
        )


@dataclass
class DragData:
    items: typing.List[DragDataItem]

    #: Bit field representing allowed drag operations. Copy = 1, Link = 2, Move = 16
    drag_operations_mask: int

    #: List of filenames that should be included when dropping
    files: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['items'] = [i.to_json() for i in self.items]
        json['dragOperationsMask'] = self.drag_operations_mask
        if self.files is not None:
            json['files'] = [i for i in self.files]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            items=[DragDataItem.from_json(i) for i in json['items']],
            drag_operations_mask=int(json['dragOperationsMask']),
            files=[str(i) for i in json['files']] if 'files' in json else None,
        )


def dispatch_drag_event(
        type_: str,
        x: float,
        y: float,
        data: DragData,
        modifiers: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a drag event into the page.

    **EXPERIMENTAL**

    :param type_: Type of the drag event.
    :param x: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    :param y: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    :param data:
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    params['data'] = data.to_json()
    if modifiers is not None:
        params['modifiers'] = modifiers
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchDragEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_key_event(
        type_: str,
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        text: typing.Optional[str] = None,
        unmodified_text: typing.Optional[str] = None,
        key_identifier: typing.Optional[str] = None,
        code: typing.Optional[str] = None,
        key: typing.Optional[str] = None,
        windows_virtual_key_code: typing.Optional[int] = None,
        native_virtual_key_code: typing.Optional[int] = None,
        auto_repeat: typing.Optional[bool] = None,
        is_keypad: typing.Optional[bool] = None,
        is_system_key: typing.Optional[bool] = None,
        location: typing.Optional[int] = None,
        commands: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a key event to the page.

    :param type_: Type of the key event.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    :param text: *(Optional)* Text as generated by processing a virtual key code with a keyboard layout. Not needed for for ```keyUp```` and ````rawKeyDown```` events (default: "")
    :param unmodified_text: *(Optional)* Text that would have been generated by the keyboard if no modifiers were pressed (except for shift). Useful for shortcut (accelerator) key handling (default: "").
    :param key_identifier: *(Optional)* Unique key identifier (e.g., 'U+0041') (default: "").
    :param code: *(Optional)* Unique DOM defined string value for each physical key (e.g., 'KeyA') (default: "").
    :param key: *(Optional)* Unique DOM defined string value describing the meaning of the key in the context of active modifiers, keyboard layout, etc (e.g., 'AltGr') (default: "").
    :param windows_virtual_key_code: *(Optional)* Windows virtual key code (default: 0).
    :param native_virtual_key_code: *(Optional)* Native virtual key code (default: 0).
    :param auto_repeat: *(Optional)* Whether the event was generated from auto repeat (default: false).
    :param is_keypad: *(Optional)* Whether the event was generated from the keypad (default: false).
    :param is_system_key: *(Optional)* Whether the event was a system key event (default: false).
    :param location: *(Optional)* Whether the event was from the left or right side of the keyboard. 1=Left, 2=Right (default: 0).
    :param commands: **(EXPERIMENTAL)** *(Optional)* Editing commands to send with the key event (e.g., 'selectAll') (default: []). These are related to but not equal the command names used in ````document.execCommand``` and NSStandardKeyBindingResponding. See https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/core/editing/commands/editor_command_names.h for valid command names.
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if text is not None:
        params['text'] = text
    if unmodified_text is not None:
        params['unmodifiedText'] = unmodified_text
    if key_identifier is not None:
        params['keyIdentifier'] = key_identifier
    if code is not None:
        params['code'] = code
    if key is not None:
        params['key'] = key
    if windows_virtual_key_code is not None:
        params['windowsVirtualKeyCode'] = windows_virtual_key_code
    if native_virtual_key_code is not None:
        params['nativeVirtualKeyCode'] = native_virtual_key_code
    if auto_repeat is not None:
        params['autoRepeat'] = auto_repeat
    if is_keypad is not None:
        params['isKeypad'] = is_keypad
    if is_system_key is not None:
        params['isSystemKey'] = is_system_key
    if location is not None:
        params['location'] = location
    if commands is not None:
        params['commands'] = [i for i in commands]
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchKeyEvent',
        'params': params,
    }
    json = yield cmd_dict


def insert_text(
        text: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method emulates inserting text that doesn't come from a key press,
    for example an emoji keyboard or an IME.

    **EXPERIMENTAL**

    :param text: The text to insert.
    '''
    params: T_JSON_DICT = dict()
    params['text'] = text
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.insertText',
        'params': params,
    }
    json = yield cmd_dict


def ime_set_composition(
        text: str,
        selection_start: int,
        selection_end: int,
        replacement_start: typing.Optional[int] = None,
        replacement_end: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method sets the current candidate text for IME.
    Use imeCommitComposition to commit the final text.
    Use imeSetComposition with empty string as text to cancel composition.

    **EXPERIMENTAL**

    :param text: The text to insert
    :param selection_start: selection start
    :param selection_end: selection end
    :param replacement_start: *(Optional)* replacement start
    :param replacement_end: *(Optional)* replacement end
    '''
    params: T_JSON_DICT = dict()
    params['text'] = text
    params['selectionStart'] = selection_start
    params['selectionEnd'] = selection_end
    if replacement_start is not None:
        params['replacementStart'] = replacement_start
    if replacement_end is not None:
        params['replacementEnd'] = replacement_end
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.imeSetComposition',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_mouse_event(
        type_: str,
        x: float,
        y: float,
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        button: typing.Optional[MouseButton] = None,
        buttons: typing.Optional[int] = None,
        click_count: typing.Optional[int] = None,
        force: typing.Optional[float] = None,
        tangential_pressure: typing.Optional[float] = None,
        tilt_x: typing.Optional[float] = None,
        tilt_y: typing.Optional[float] = None,
        twist: typing.Optional[int] = None,
        delta_x: typing.Optional[float] = None,
        delta_y: typing.Optional[float] = None,
        pointer_type: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a mouse event to the page.

    :param type_: Type of the mouse event.
    :param x: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    :param y: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    :param button: *(Optional)* Mouse button (default: "none").
    :param buttons: *(Optional)* A number indicating which buttons are pressed on the mouse when a mouse event is triggered. Left=1, Right=2, Middle=4, Back=8, Forward=16, None=0.
    :param click_count: *(Optional)* Number of times the mouse button was clicked (default: 0).
    :param force: **(EXPERIMENTAL)** *(Optional)* The normalized pressure, which has a range of [0,1] (default: 0).
    :param tangential_pressure: **(EXPERIMENTAL)** *(Optional)* The normalized tangential pressure, which has a range of [-1,1] (default: 0).
    :param tilt_x: *(Optional)* The plane angle between the Y-Z plane and the plane containing both the stylus axis and the Y axis, in degrees of the range [-90,90], a positive tiltX is to the right (default: 0).
    :param tilt_y: *(Optional)* The plane angle between the X-Z plane and the plane containing both the stylus axis and the X axis, in degrees of the range [-90,90], a positive tiltY is towards the user (default: 0).
    :param twist: **(EXPERIMENTAL)** *(Optional)* The clockwise rotation of a pen stylus around its own major axis, in degrees in the range [0,359] (default: 0).
    :param delta_x: *(Optional)* X delta in CSS pixels for mouse wheel event (default: 0).
    :param delta_y: *(Optional)* Y delta in CSS pixels for mouse wheel event (default: 0).
    :param pointer_type: *(Optional)* Pointer type (default: "mouse").
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if button is not None:
        params['button'] = button.to_json()
    if buttons is not None:
        params['buttons'] = buttons
    if click_count is not None:
        params['clickCount'] = click_count
    if force is not None:
        params['force'] = force
    if tangential_pressure is not None:
        params['tangentialPressure'] = tangential_pressure
    if tilt_x is not None:
        params['tiltX'] = tilt_x
    if tilt_y is not None:
        params['tiltY'] = tilt_y
    if twist is not None:
        params['twist'] = twist
    if delta_x is not None:
        params['deltaX'] = delta_x
    if delta_y is not None:
        params['deltaY'] = delta_y
    if pointer_type is not None:
        params['pointerType'] = pointer_type
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchMouseEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_touch_event(
        type_: str,
        touch_points: typing.List[TouchPoint],
        modifiers: typing.Optional[int] = None,
        timestamp: typing.Optional[TimeSinceEpoch] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Dispatches a touch event to the page.

    :param type_: Type of the touch event. TouchEnd and TouchCancel must not contain any touch points, while TouchStart and TouchMove must contains at least one.
    :param touch_points: Active touch points on the touch device. One event per any changed point (compared to previous touch event in a sequence) is generated, emulating pressing/moving/releasing points one by one.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['touchPoints'] = [i.to_json() for i in touch_points]
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.dispatchTouchEvent',
        'params': params,
    }
    json = yield cmd_dict


def cancel_dragging() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Cancels any active dragging in the page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.cancelDragging',
    }
    json = yield cmd_dict


def emulate_touch_from_mouse_event(
        type_: str,
        x: int,
        y: int,
        button: MouseButton,
        timestamp: typing.Optional[TimeSinceEpoch] = None,
        delta_x: typing.Optional[float] = None,
        delta_y: typing.Optional[float] = None,
        modifiers: typing.Optional[int] = None,
        click_count: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Emulates touch event from the mouse event parameters.

    **EXPERIMENTAL**

    :param type_: Type of the mouse event.
    :param x: X coordinate of the mouse pointer in DIP.
    :param y: Y coordinate of the mouse pointer in DIP.
    :param button: Mouse button. Only "none", "left", "right" are supported.
    :param timestamp: *(Optional)* Time at which the event occurred (default: current time).
    :param delta_x: *(Optional)* X delta in DIP for mouse wheel event (default: 0).
    :param delta_y: *(Optional)* Y delta in DIP for mouse wheel event (default: 0).
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param click_count: *(Optional)* Number of times the mouse button was clicked (default: 0).
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    params['button'] = button.to_json()
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if delta_x is not None:
        params['deltaX'] = delta_x
    if delta_y is not None:
        params['deltaY'] = delta_y
    if modifiers is not None:
        params['modifiers'] = modifiers
    if click_count is not None:
        params['clickCount'] = click_count
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.emulateTouchFromMouseEvent',
        'params': params,
    }
    json = yield cmd_dict


def set_ignore_input_events(
        ignore: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Ignores input events (useful while auditing page).

    :param ignore: Ignores input events processing when set to true.
    '''
    params: T_JSON_DICT = dict()
    params['ignore'] = ignore
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.setIgnoreInputEvents',
        'params': params,
    }
    json = yield cmd_dict


def set_intercept_drags(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Prevents default drag and drop behavior and instead emits ``Input.dragIntercepted`` events.
    Drag and drop behavior can be directly controlled via ``Input.dispatchDragEvent``.

    **EXPERIMENTAL**

    :param enabled:
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.setInterceptDrags',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_pinch_gesture(
        x: float,
        y: float,
        scale_factor: float,
        relative_speed: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a pinch gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param scale_factor: Relative scale factor after zooming (>1.0 zooms in, <1.0 zooms out).
    :param relative_speed: *(Optional)* Relative pointer speed in pixels per second (default: 800).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    params['scaleFactor'] = scale_factor
    if relative_speed is not None:
        params['relativeSpeed'] = relative_speed
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizePinchGesture',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_scroll_gesture(
        x: float,
        y: float,
        x_distance: typing.Optional[float] = None,
        y_distance: typing.Optional[float] = None,
        x_overscroll: typing.Optional[float] = None,
        y_overscroll: typing.Optional[float] = None,
        prevent_fling: typing.Optional[bool] = None,
        speed: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None,
        repeat_count: typing.Optional[int] = None,
        repeat_delay_ms: typing.Optional[int] = None,
        interaction_marker_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a scroll gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param x_distance: *(Optional)* The distance to scroll along the X axis (positive to scroll left).
    :param y_distance: *(Optional)* The distance to scroll along the Y axis (positive to scroll up).
    :param x_overscroll: *(Optional)* The number of additional pixels to scroll back along the X axis, in addition to the given distance.
    :param y_overscroll: *(Optional)* The number of additional pixels to scroll back along the Y axis, in addition to the given distance.
    :param prevent_fling: *(Optional)* Prevent fling (default: true).
    :param speed: *(Optional)* Swipe speed in pixels per second (default: 800).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    :param repeat_count: *(Optional)* The number of times to repeat the gesture (default: 0).
    :param repeat_delay_ms: *(Optional)* The number of milliseconds delay between each repeat. (default: 250).
    :param interaction_marker_name: *(Optional)* The name of the interaction markers to generate, if not empty (default: "").
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if x_distance is not None:
        params['xDistance'] = x_distance
    if y_distance is not None:
        params['yDistance'] = y_distance
    if x_overscroll is not None:
        params['xOverscroll'] = x_overscroll
    if y_overscroll is not None:
        params['yOverscroll'] = y_overscroll
    if prevent_fling is not None:
        params['preventFling'] = prevent_fling
    if speed is not None:
        params['speed'] = speed
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    if repeat_count is not None:
        params['repeatCount'] = repeat_count
    if repeat_delay_ms is not None:
        params['repeatDelayMs'] = repeat_delay_ms
    if interaction_marker_name is not None:
        params['interactionMarkerName'] = interaction_marker_name
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizeScrollGesture',
        'params': params,
    }
    json = yield cmd_dict


def synthesize_tap_gesture(
        x: float,
        y: float,
        duration: typing.Optional[int] = None,
        tap_count: typing.Optional[int] = None,
        gesture_source_type: typing.Optional[GestureSourceType] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Synthesizes a tap gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param duration: *(Optional)* Duration between touchdown and touchup events in ms (default: 50).
    :param tap_count: *(Optional)* Number of times to perform the tap (e.g. 2 for double tap, default: 1).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if duration is not None:
        params['duration'] = duration
    if tap_count is not None:
        params['tapCount'] = tap_count
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Input.synthesizeTapGesture',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Input.dragIntercepted')
@dataclass
class DragIntercepted:
    '''
    **EXPERIMENTAL**

    Emitted only when ``Input.setInterceptDrags`` is enabled. Use this data with ``Input.dispatchDragEvent`` to
    restore normal drag and drop behavior.
    '''
    data: DragData

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DragIntercepted:
        return cls(
            data=DragData.from_json(json['data'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\utils.py ===
"""
**Contains**

* refraction_angle
* fresnel_coefficients
* deviation
* brewster_angle
* critical_angle
* lens_makers_formula
* mirror_formula
* lens_formula
* hyperfocal_distance
* transverse_magnification
"""

__all__ = ['refraction_angle',
           'deviation',
           'fresnel_coefficients',
           'brewster_angle',
           'critical_angle',
           'lens_makers_formula',
           'mirror_formula',
           'lens_formula',
           'hyperfocal_distance',
           'transverse_magnification'
           ]

from sympy.core.numbers import (Float, I, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2, cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import cancel
from sympy.series.limits import Limit
from sympy.geometry.line import Ray3D
from sympy.geometry.util import intersection
from sympy.geometry.plane import Plane
from sympy.utilities.iterables import is_sequence
from .medium import Medium


def refractive_index_of_medium(medium):
    """
    Helper function that returns refractive index, given a medium
    """
    if isinstance(medium, Medium):
        n = medium.refractive_index
    else:
        n = sympify(medium)
    return n


def refraction_angle(incident, medium1, medium2, normal=None, plane=None):
    """
    This function calculates transmitted vector after refraction at planar
    surface. ``medium1`` and ``medium2`` can be ``Medium`` or any sympifiable object.
    If ``incident`` is a number then treated as angle of incidence (in radians)
    in which case refraction angle is returned.

    If ``incident`` is an object of `Ray3D`, `normal` also has to be an instance
    of `Ray3D` in order to get the output as a `Ray3D`. Please note that if
    plane of separation is not provided and normal is an instance of `Ray3D`,
    ``normal`` will be assumed to be intersecting incident ray at the plane of
    separation. This will not be the case when `normal` is a `Matrix` or
    any other sequence.
    If ``incident`` is an instance of `Ray3D` and `plane` has not been provided
    and ``normal`` is not `Ray3D`, output will be a `Matrix`.

    Parameters
    ==========

    incident : Matrix, Ray3D, sequence or a number
        Incident vector or angle of incidence
    medium1 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 1 or its refractive index
    medium2 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 2 or its refractive index
    normal : Matrix, Ray3D, or sequence
        Normal vector
    plane : Plane
        Plane of separation of the two media.

    Returns
    =======

    Returns an angle of refraction or a refracted ray depending on inputs.

    Examples
    ========

    >>> from sympy.physics.optics import refraction_angle
    >>> from sympy.geometry import Point3D, Ray3D, Plane
    >>> from sympy.matrices import Matrix
    >>> from sympy import symbols, pi
    >>> n = Matrix([0, 0, 1])
    >>> P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    >>> r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    >>> refraction_angle(r1, 1, 1, n)
    Matrix([
    [ 1],
    [ 1],
    [-1]])
    >>> refraction_angle(r1, 1, 1, plane=P)
    Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))

    With different index of refraction of the two media

    >>> n1, n2 = symbols('n1, n2')
    >>> refraction_angle(r1, n1, n2, n)
    Matrix([
    [                                n1/n2],
    [                                n1/n2],
    [-sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)]])
    >>> refraction_angle(r1, n1, n2, plane=P)
    Ray3D(Point3D(0, 0, 0), Point3D(n1/n2, n1/n2, -sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)))
    >>> round(refraction_angle(pi/6, 1.2, 1.5), 5)
    0.41152
    """

    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    # check if an incidence angle was supplied instead of a ray
    try:
        angle_of_incidence = float(incident)
    except TypeError:
        angle_of_incidence = None

    try:
        critical_angle_ = critical_angle(medium1, medium2)
    except (ValueError, TypeError):
        critical_angle_ = None

    if angle_of_incidence is not None:
        if normal is not None or plane is not None:
            raise ValueError('Normal/plane not allowed if incident is an angle')

        if not 0.0 <= angle_of_incidence < pi*0.5:
            raise ValueError('Angle of incidence not in range [0:pi/2)')

        if critical_angle_ and angle_of_incidence > critical_angle_:
            raise ValueError('Ray undergoes total internal reflection')
        return asin(n1*sin(angle_of_incidence)/n2)

    # Treat the incident as ray below
    # A flag to check whether to return Ray3D or not
    return_ray = False

    if plane is not None and normal is not None:
        raise ValueError("Either plane or normal is acceptable.")

    if not isinstance(incident, Matrix):
        if is_sequence(incident):
            _incident = Matrix(incident)
        elif isinstance(incident, Ray3D):
            _incident = Matrix(incident.direction_ratio)
        else:
            raise TypeError(
                "incident should be a Matrix, Ray3D, or sequence")
    else:
        _incident = incident

    # If plane is provided, get direction ratios of the normal
    # to the plane from the plane else go with `normal` param.
    if plane is not None:
        if not isinstance(plane, Plane):
            raise TypeError("plane should be an instance of geometry.plane.Plane")
        # If we have the plane, we can get the intersection
        # point of incident ray and the plane and thus return
        # an instance of Ray3D.
        if isinstance(incident, Ray3D):
            return_ray = True
            intersection_pt = plane.intersection(incident)[0]
        _normal = Matrix(plane.normal_vector)
    else:
        if not isinstance(normal, Matrix):
            if is_sequence(normal):
                _normal = Matrix(normal)
            elif isinstance(normal, Ray3D):
                _normal = Matrix(normal.direction_ratio)
                if isinstance(incident, Ray3D):
                    intersection_pt = intersection(incident, normal)
                    if len(intersection_pt) == 0:
                        raise ValueError(
                            "Normal isn't concurrent with the incident ray.")
                    else:
                        return_ray = True
                        intersection_pt = intersection_pt[0]
            else:
                raise TypeError(
                    "Normal should be a Matrix, Ray3D, or sequence")
        else:
            _normal = normal

    eta = n1/n2  # Relative index of refraction
    # Calculating magnitude of the vectors
    mag_incident = sqrt(sum(i**2 for i in _incident))
    mag_normal = sqrt(sum(i**2 for i in _normal))
    # Converting vectors to unit vectors by dividing
    # them with their magnitudes
    _incident /= mag_incident
    _normal /= mag_normal
    c1 = -_incident.dot(_normal)  # cos(angle_of_incidence)
    cs2 = 1 - eta**2*(1 - c1**2)  # cos(angle_of_refraction)**2
    if cs2.is_negative:  # This is the case of total internal reflection(TIR).
        return S.Zero
    drs = eta*_incident + (eta*c1 - sqrt(cs2))*_normal
    # Multiplying unit vector by its magnitude
    drs = drs*mag_incident
    if not return_ray:
        return drs
    else:
        return Ray3D(intersection_pt, direction_ratio=drs)


def fresnel_coefficients(angle_of_incidence, medium1, medium2):
    """
    This function uses Fresnel equations to calculate reflection and
    transmission coefficients. Those are obtained for both polarisations
    when the electric field vector is in the plane of incidence (labelled 'p')
    and when the electric field vector is perpendicular to the plane of
    incidence (labelled 's'). There are four real coefficients unless the
    incident ray reflects in total internal in which case there are two complex
    ones. Angle of incidence is the angle between the incident ray and the
    surface normal. ``medium1`` and ``medium2`` can be ``Medium`` or any
    sympifiable object.

    Parameters
    ==========

    angle_of_incidence : sympifiable

    medium1 : Medium or sympifiable
        Medium 1 or its refractive index

    medium2 : Medium or sympifiable
        Medium 2 or its refractive index

    Returns
    =======

    Returns a list with four real Fresnel coefficients:
    [reflection p (TM), reflection s (TE),
    transmission p (TM), transmission s (TE)]
    If the ray is undergoes total internal reflection then returns a
    list of two complex Fresnel coefficients:
    [reflection p (TM), reflection s (TE)]

    Examples
    ========

    >>> from sympy.physics.optics import fresnel_coefficients
    >>> fresnel_coefficients(0.3, 1, 2)
    [0.317843553417859, -0.348645229818821,
            0.658921776708929, 0.651354770181179]
    >>> fresnel_coefficients(0.6, 2, 1)
    [-0.235625382192159 - 0.971843958291041*I,
             0.816477005968898 - 0.577377951366403*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_equations
    """
    if not 0 <= 2*angle_of_incidence < pi:
        raise ValueError('Angle of incidence not in range [0:pi/2)')

    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    angle_of_refraction = asin(n1*sin(angle_of_incidence)/n2)
    try:
        angle_of_total_internal_reflection_onset = critical_angle(n1, n2)
    except ValueError:
        angle_of_total_internal_reflection_onset = None

    if angle_of_total_internal_reflection_onset is None or\
    angle_of_total_internal_reflection_onset > angle_of_incidence:
        R_s = -sin(angle_of_incidence - angle_of_refraction)\
                /sin(angle_of_incidence + angle_of_refraction)
        R_p = tan(angle_of_incidence - angle_of_refraction)\
                /tan(angle_of_incidence + angle_of_refraction)
        T_s = 2*sin(angle_of_refraction)*cos(angle_of_incidence)\
                /sin(angle_of_incidence + angle_of_refraction)
        T_p = 2*sin(angle_of_refraction)*cos(angle_of_incidence)\
                /(sin(angle_of_incidence + angle_of_refraction)\
                *cos(angle_of_incidence - angle_of_refraction))
        return [R_p, R_s, T_p, T_s]
    else:
        n = n2/n1
        R_s = cancel((cos(angle_of_incidence)-\
                I*sqrt(sin(angle_of_incidence)**2 - n**2))\
                /(cos(angle_of_incidence)+\
                I*sqrt(sin(angle_of_incidence)**2 - n**2)))
        R_p = cancel((n**2*cos(angle_of_incidence)-\
                I*sqrt(sin(angle_of_incidence)**2 - n**2))\
                /(n**2*cos(angle_of_incidence)+\
                I*sqrt(sin(angle_of_incidence)**2 - n**2)))
        return [R_p, R_s]


def deviation(incident, medium1, medium2, normal=None, plane=None):
    """
    This function calculates the angle of deviation of a ray
    due to refraction at planar surface.

    Parameters
    ==========

    incident : Matrix, Ray3D, sequence or float
        Incident vector or angle of incidence
    medium1 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 1 or its refractive index
    medium2 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 2 or its refractive index
    normal : Matrix, Ray3D, or sequence
        Normal vector
    plane : Plane
        Plane of separation of the two media.

    Returns angular deviation between incident and refracted rays

    Examples
    ========

    >>> from sympy.physics.optics import deviation
    >>> from sympy.geometry import Point3D, Ray3D, Plane
    >>> from sympy.matrices import Matrix
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1, n2')
    >>> n = Matrix([0, 0, 1])
    >>> P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    >>> r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    >>> deviation(r1, 1, 1, n)
    0
    >>> deviation(r1, n1, n2, plane=P)
    -acos(-sqrt(-2*n1**2/(3*n2**2) + 1)) + acos(-sqrt(3)/3)
    >>> round(deviation(0.1, 1.2, 1.5), 5)
    -0.02005
    """
    refracted = refraction_angle(incident,
                                 medium1,
                                 medium2,
                                 normal=normal,
                                 plane=plane)
    try:
        angle_of_incidence = Float(incident)
    except TypeError:
        angle_of_incidence = None

    if angle_of_incidence is not None:
        return float(refracted) - angle_of_incidence

    if refracted != 0:
        if isinstance(refracted, Ray3D):
            refracted = Matrix(refracted.direction_ratio)

        if not isinstance(incident, Matrix):
            if is_sequence(incident):
                _incident = Matrix(incident)
            elif isinstance(incident, Ray3D):
                _incident = Matrix(incident.direction_ratio)
            else:
                raise TypeError(
                    "incident should be a Matrix, Ray3D, or sequence")
        else:
            _incident = incident

        if plane is None:
            if not isinstance(normal, Matrix):
                if is_sequence(normal):
                    _normal = Matrix(normal)
                elif isinstance(normal, Ray3D):
                    _normal = Matrix(normal.direction_ratio)
                else:
                    raise TypeError(
                        "normal should be a Matrix, Ray3D, or sequence")
            else:
                _normal = normal
        else:
            _normal = Matrix(plane.normal_vector)

        mag_incident = sqrt(sum(i**2 for i in _incident))
        mag_normal = sqrt(sum(i**2 for i in _normal))
        mag_refracted = sqrt(sum(i**2 for i in refracted))
        _incident /= mag_incident
        _normal /= mag_normal
        refracted /= mag_refracted
        i = acos(_incident.dot(_normal))
        r = acos(refracted.dot(_normal))
        return i - r


def brewster_angle(medium1, medium2):
    """
    This function calculates the Brewster's angle of incidence to Medium 2 from
    Medium 1 in radians.

    Parameters
    ==========

    medium 1 : Medium or sympifiable
        Refractive index of Medium 1
    medium 2 : Medium or sympifiable
        Refractive index of Medium 1

    Examples
    ========

    >>> from sympy.physics.optics import brewster_angle
    >>> brewster_angle(1, 1.33)
    0.926093295503462

    """

    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    return atan2(n2, n1)

def critical_angle(medium1, medium2):
    """
    This function calculates the critical angle of incidence (marking the onset
    of total internal) to Medium 2 from Medium 1 in radians.

    Parameters
    ==========

    medium 1 : Medium or sympifiable
        Refractive index of Medium 1.
    medium 2 : Medium or sympifiable
        Refractive index of Medium 1.

    Examples
    ========

    >>> from sympy.physics.optics import critical_angle
    >>> critical_angle(1.33, 1)
    0.850908514477849

    """

    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    if n2 > n1:
        raise ValueError('Total internal reflection impossible for n1 < n2')
    else:
        return asin(n2/n1)



def lens_makers_formula(n_lens, n_surr, r1, r2, d=0):
    """
    This function calculates focal length of a lens.
    It follows cartesian sign convention.

    Parameters
    ==========

    n_lens : Medium or sympifiable
        Index of refraction of lens.
    n_surr : Medium or sympifiable
        Index of reflection of surrounding.
    r1 : sympifiable
        Radius of curvature of first surface.
    r2 : sympifiable
        Radius of curvature of second surface.
    d : sympifiable, optional
        Thickness of lens, default value is 0.

    Examples
    ========

    >>> from sympy.physics.optics import lens_makers_formula
    >>> from sympy import S
    >>> lens_makers_formula(1.33, 1, 10, -10)
    15.1515151515151
    >>> lens_makers_formula(1.2, 1, 10, S.Infinity)
    50.0000000000000
    >>> lens_makers_formula(1.33, 1, 10, -10, d=1)
    15.3418463277618

    """

    if isinstance(n_lens, Medium):
        n_lens = n_lens.refractive_index
    else:
        n_lens = sympify(n_lens)
    if isinstance(n_surr, Medium):
        n_surr = n_surr.refractive_index
    else:
        n_surr = sympify(n_surr)
    d = sympify(d)

    focal_length = 1/((n_lens - n_surr) / n_surr*(1/r1 - 1/r2 + (((n_lens - n_surr) * d) / (n_lens * r1 * r2))))

    if focal_length == zoo:
        return S.Infinity
    return focal_length


def mirror_formula(focal_length=None, u=None, v=None):
    """
    This function provides one of the three parameters
    when two of them are supplied.
    This is valid only for paraxial rays.

    Parameters
    ==========

    focal_length : sympifiable
        Focal length of the mirror.
    u : sympifiable
        Distance of object from the pole on
        the principal axis.
    v : sympifiable
        Distance of the image from the pole
        on the principal axis.

    Examples
    ========

    >>> from sympy.physics.optics import mirror_formula
    >>> from sympy.abc import f, u, v
    >>> mirror_formula(focal_length=f, u=u)
    f*u/(-f + u)
    >>> mirror_formula(focal_length=f, v=v)
    f*v/(-f + v)
    >>> mirror_formula(u=u, v=v)
    u*v/(u + v)

    """
    if focal_length and u and v:
        raise ValueError("Please provide only two parameters")

    focal_length = sympify(focal_length)
    u = sympify(u)
    v = sympify(v)
    if u is oo:
        _u = Symbol('u')
    if v is oo:
        _v = Symbol('v')
    if focal_length is oo:
        _f = Symbol('f')
    if focal_length is None:
        if u is oo and v is oo:
            return Limit(Limit(_v*_u/(_v + _u), _u, oo), _v, oo).doit()
        if u is oo:
            return Limit(v*_u/(v + _u), _u, oo).doit()
        if v is oo:
            return Limit(_v*u/(_v + u), _v, oo).doit()
        return v*u/(v + u)
    if u is None:
        if v is oo and focal_length is oo:
            return Limit(Limit(_v*_f/(_v - _f), _v, oo), _f, oo).doit()
        if v is oo:
            return Limit(_v*focal_length/(_v - focal_length), _v, oo).doit()
        if focal_length is oo:
            return Limit(v*_f/(v - _f), _f, oo).doit()
        return v*focal_length/(v - focal_length)
    if v is None:
        if u is oo and focal_length is oo:
            return Limit(Limit(_u*_f/(_u - _f), _u, oo), _f, oo).doit()
        if u is oo:
            return Limit(_u*focal_length/(_u - focal_length), _u, oo).doit()
        if focal_length is oo:
            return Limit(u*_f/(u - _f), _f, oo).doit()
        return u*focal_length/(u - focal_length)


def lens_formula(focal_length=None, u=None, v=None):
    """
    This function provides one of the three parameters
    when two of them are supplied.
    This is valid only for paraxial rays.

    Parameters
    ==========

    focal_length : sympifiable
        Focal length of the mirror.
    u : sympifiable
        Distance of object from the optical center on
        the principal axis.
    v : sympifiable
        Distance of the image from the optical center
        on the principal axis.

    Examples
    ========

    >>> from sympy.physics.optics import lens_formula
    >>> from sympy.abc import f, u, v
    >>> lens_formula(focal_length=f, u=u)
    f*u/(f + u)
    >>> lens_formula(focal_length=f, v=v)
    f*v/(f - v)
    >>> lens_formula(u=u, v=v)
    u*v/(u - v)

    """
    if focal_length and u and v:
        raise ValueError("Please provide only two parameters")

    focal_length = sympify(focal_length)
    u = sympify(u)
    v = sympify(v)
    if u is oo:
        _u = Symbol('u')
    if v is oo:
        _v = Symbol('v')
    if focal_length is oo:
        _f = Symbol('f')
    if focal_length is None:
        if u is oo and v is oo:
            return Limit(Limit(_v*_u/(_u - _v), _u, oo), _v, oo).doit()
        if u is oo:
            return Limit(v*_u/(_u - v), _u, oo).doit()
        if v is oo:
            return Limit(_v*u/(u - _v), _v, oo).doit()
        return v*u/(u - v)
    if u is None:
        if v is oo and focal_length is oo:
            return Limit(Limit(_v*_f/(_f - _v), _v, oo), _f, oo).doit()
        if v is oo:
            return Limit(_v*focal_length/(focal_length - _v), _v, oo).doit()
        if focal_length is oo:
            return Limit(v*_f/(_f - v), _f, oo).doit()
        return v*focal_length/(focal_length - v)
    if v is None:
        if u is oo and focal_length is oo:
            return Limit(Limit(_u*_f/(_u + _f), _u, oo), _f, oo).doit()
        if u is oo:
            return Limit(_u*focal_length/(_u + focal_length), _u, oo).doit()
        if focal_length is oo:
            return Limit(u*_f/(u + _f), _f, oo).doit()
        return u*focal_length/(u + focal_length)

def hyperfocal_distance(f, N, c):
    """

    Parameters
    ==========

    f: sympifiable
        Focal length of a given lens.

    N: sympifiable
        F-number of a given lens.

    c: sympifiable
        Circle of Confusion (CoC) of a given image format.

    Example
    =======

    >>> from sympy.physics.optics import hyperfocal_distance
    >>> round(hyperfocal_distance(f = 0.5, N = 8, c = 0.0033), 2)
    9.47
    """

    f = sympify(f)
    N = sympify(N)
    c = sympify(c)

    return (1/(N * c))*(f**2)

def transverse_magnification(si, so):
    """

    Calculates the transverse magnification upon reflection in a mirror,
    which is the ratio of the image size to the object size.

    Parameters
    ==========

    so: sympifiable
        Lens-object distance.

    si: sympifiable
        Lens-image distance.

    Example
    =======

    >>> from sympy.physics.optics import transverse_magnification
    >>> transverse_magnification(30, 15)
    -2

    """

    si = sympify(si)
    so = sympify(so)

    return (-(si/so))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\symbolic_probability.py ===
import itertools
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Not
from sympy.core.parameters import global_parameters
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.stats import variance, covariance
from sympy.stats.rv import (RandomSymbol, pspace, dependent,
                            given, sampling_E, RandomIndexedSymbol, is_random,
                            PSpace, sampling_P, random_symbols)

__all__ = ['Probability', 'Expectation', 'Variance', 'Covariance']


@is_random.register(Expr)
def _(x):
    atoms = x.free_symbols
    if len(atoms) == 1 and next(iter(atoms)) == x:
        return False
    return any(is_random(i) for i in atoms)

@is_random.register(RandomSymbol)  # type: ignore
def _(x):
    return True


class Probability(Expr):
    """
    Symbolic expression for the probability.

    Examples
    ========

    >>> from sympy.stats import Probability, Normal
    >>> from sympy import Integral
    >>> X = Normal("X", 0, 1)
    >>> prob = Probability(X > 1)
    >>> prob
    Probability(X > 1)

    Integral representation:

    >>> prob.rewrite(Integral)
    Integral(sqrt(2)*exp(-_z**2/2)/(2*sqrt(pi)), (_z, 1, oo))

    Evaluation of the integral:

    >>> prob.evaluate_integral()
    sqrt(2)*(-sqrt(2)*sqrt(pi)*erf(sqrt(2)/2) + sqrt(2)*sqrt(pi))/(4*sqrt(pi))
    """

    is_commutative = True

    def __new__(cls, prob, condition=None, **kwargs):
        prob = _sympify(prob)
        if condition is None:
            obj = Expr.__new__(cls, prob)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, prob, condition)
        obj._condition = condition
        return obj

    def doit(self, **hints):
        condition = self.args[0]
        given_condition = self._condition
        numsamples = hints.get('numsamples', False)
        evaluate = hints.get('evaluate', True)

        if isinstance(condition, Not):
            return S.One - self.func(condition.args[0], given_condition,
                                    evaluate=evaluate).doit(**hints)

        if condition.has(RandomIndexedSymbol):
            return pspace(condition).probability(condition, given_condition,
                                             evaluate=evaluate)

        if isinstance(given_condition, RandomSymbol):
            condrv = random_symbols(condition)
            if len(condrv) == 1 and condrv[0] == given_condition:
                from sympy.stats.frv_types import BernoulliDistribution
                return BernoulliDistribution(self.func(condition).doit(**hints), 0, 1)
            if any(dependent(rv, given_condition) for rv in condrv):
                return Probability(condition, given_condition)
            else:
                return Probability(condition).doit()

        if given_condition is not None and \
                not isinstance(given_condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (given_condition))

        if given_condition == False or condition is S.false:
            return S.Zero
        if not isinstance(condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (condition))
        if condition is S.true:
            return S.One

        if numsamples:
            return sampling_P(condition, given_condition, numsamples=numsamples)
        if given_condition is not None:  # If there is a condition
            # Recompute on new conditional expr
            return Probability(given(condition, given_condition)).doit()

        # Otherwise pass work off to the ProbabilitySpace
        if pspace(condition) == PSpace():
            return Probability(condition, given_condition)

        result = pspace(condition).probability(condition)
        if hasattr(result, 'doit') and evaluate:
            return result.doit()
        else:
            return result

    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return self.func(arg, condition=condition).doit(evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Expectation(Expr):
    """
    Symbolic expression for the expectation.

    Examples
    ========

    >>> from sympy.stats import Expectation, Normal, Probability, Poisson
    >>> from sympy import symbols, Integral, Sum
    >>> mu = symbols("mu")
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Expectation(X)
    Expectation(X)
    >>> Expectation(X).evaluate_integral().simplify()
    mu

    To get the integral expression of the expectation:

    >>> Expectation(X).rewrite(Integral)
    Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    The same integral expression, in more abstract terms:

    >>> Expectation(X).rewrite(Probability)
    Integral(x*Probability(Eq(X, x)), (x, -oo, oo))

    To get the Summation expression of the expectation for discrete random variables:

    >>> lamda = symbols('lamda', positive=True)
    >>> Z = Poisson('Z', lamda)
    >>> Expectation(Z).rewrite(Sum)
    Sum(Z*lamda**Z*exp(-lamda)/factorial(Z), (Z, 0, oo))

    This class is aware of some properties of the expectation:

    >>> from sympy.abc import a
    >>> Expectation(a*X)
    Expectation(a*X)
    >>> Y = Normal("Y", 1, 2)
    >>> Expectation(X + Y)
    Expectation(X + Y)

    To expand the ``Expectation`` into its expression, use ``expand()``:

    >>> Expectation(X + Y).expand()
    Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y).expand()
    a*Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y)
    Expectation(a*X + Y)
    >>> Expectation((X + Y)*(X - Y)).expand()
    Expectation(X**2) - Expectation(Y**2)

    To evaluate the ``Expectation``, use ``doit()``:

    >>> Expectation(X + Y).doit()
    mu + 1
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit()
    3*mu + 1

    To prevent evaluating nested ``Expectation``, use ``doit(deep=False)``

    >>> Expectation(X + Expectation(Y)).doit(deep=False)
    mu + Expectation(Expectation(Y))
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit(deep=False)
    mu + Expectation(Expectation(Expectation(2*X) + Y))

    """

    def __new__(cls, expr, condition=None, **kwargs):
        expr = _sympify(expr)
        if expr.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import ExpectationMatrix
            return ExpectationMatrix(expr, condition)
        if condition is None:
            if not is_random(expr):
                return expr
            obj = Expr.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, expr, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return(self.args[0].is_commutative)

    def expand(self, **hints):
        expr = self.args[0]
        condition = self._condition

        if not is_random(expr):
            return expr

        if isinstance(expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expr.args)

        expand_expr = _expand(expr)
        if isinstance(expand_expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expand_expr.args)

        elif isinstance(expr, Mul):
            rv = []
            nonrv = []
            for a in expr.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a)
            return Mul.fromiter(nonrv)*Expectation(Mul.fromiter(rv), condition=condition)

        return self

    def doit(self, **hints):
        deep = hints.get('deep', True)
        condition = self._condition
        expr = self.args[0]
        numsamples = hints.get('numsamples', False)
        evaluate = hints.get('evaluate', True)

        if deep:
            expr = expr.doit(**hints)

        if not is_random(expr) or isinstance(expr, Expectation):  # expr isn't random?
            return expr
        if numsamples:  # Computing by monte carlo sampling?
            evalf = hints.get('evalf', True)
            return sampling_E(expr, condition, numsamples=numsamples, evalf=evalf)

        if expr.has(RandomIndexedSymbol):
            return pspace(expr).compute_expectation(expr, condition)

        # Create new expr and recompute E
        if condition is not None:  # If there is a condition
            return self.func(given(expr, condition)).doit(**hints)

        # A few known statements for efficiency

        if expr.is_Add:  # We know that E is Linear
            return Add(*[self.func(arg, condition).doit(**hints)
                    if not isinstance(arg, Expectation) else self.func(arg, condition)
                         for arg in expr.args])
        if expr.is_Mul:
            if expr.atoms(Expectation):
                return expr

        if pspace(expr) == PSpace():
            return self.func(expr)
        # Otherwise case is simple, pass work off to the ProbabilitySpace
        result = pspace(expr).compute_expectation(expr, evaluate=evaluate)
        if hasattr(result, 'doit') and evaluate:
            return result.doit(**hints)
        else:
            return result


    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        rvs = arg.atoms(RandomSymbol)
        if len(rvs) > 1:
            raise NotImplementedError()
        if len(rvs) == 0:
            return arg

        rv = rvs.pop()
        if rv.pspace is None:
            raise ValueError("Probability space not known")

        symbol = rv.symbol
        if symbol.name[0].isupper():
            symbol = Symbol(symbol.name.lower())
        else :
            symbol = Symbol(symbol.name + "_1")

        if rv.pspace.is_Continuous:
            return Integral(arg.replace(rv, symbol)*Probability(Eq(rv, symbol), condition), (symbol, rv.pspace.domain.set.inf, rv.pspace.domain.set.sup))
        else:
            if rv.pspace.is_Finite:
                raise NotImplementedError
            else:
                return Sum(arg.replace(rv, symbol)*Probability(Eq(rv, symbol), condition), (symbol, rv.pspace.domain.set.inf, rv.pspace.set.sup))

    def _eval_rewrite_as_Integral(self, arg, condition=None, evaluate=False, **kwargs):
        return self.func(arg, condition=condition).doit(deep=False, evaluate=evaluate)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral # For discrete this will be Sum

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()

    evaluate_sum = evaluate_integral

class Variance(Expr):
    """
    Symbolic expression for the variance.

    Examples
    ========

    >>> from sympy import symbols, Integral
    >>> from sympy.stats import Normal, Expectation, Variance, Probability
    >>> mu = symbols("mu", positive=True)
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Variance(X)
    Variance(X)
    >>> Variance(X).evaluate_integral()
    sigma**2

    Integral representation of the underlying calculations:

    >>> Variance(X).rewrite(Integral)
    Integral(sqrt(2)*(X - Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo)))**2*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    Integral representation, without expanding the PDF:

    >>> Variance(X).rewrite(Probability)
    -Integral(x*Probability(Eq(X, x)), (x, -oo, oo))**2 + Integral(x**2*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the variance in terms of the expectation

    >>> Variance(X).rewrite(Expectation)
    -Expectation(X)**2 + Expectation(X**2)

    Some transformations based on the properties of the variance may happen:

    >>> from sympy.abc import a
    >>> Y = Normal("Y", 0, 1)
    >>> Variance(a*X)
    Variance(a*X)

    To expand the variance in its expression, use ``expand()``:

    >>> Variance(a*X).expand()
    a**2*Variance(X)
    >>> Variance(X + Y)
    Variance(X + Y)
    >>> Variance(X + Y).expand()
    2*Covariance(X, Y) + Variance(X) + Variance(Y)

    """
    def __new__(cls, arg, condition=None, **kwargs):
        arg = _sympify(arg)

        if arg.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import VarianceMatrix
            return VarianceMatrix(arg, condition)
        if condition is None:
            obj = Expr.__new__(cls, arg)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, arg, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return self.args[0].is_commutative

    def expand(self, **hints):
        arg = self.args[0]
        condition = self._condition

        if not is_random(arg):
            return S.Zero

        if isinstance(arg, RandomSymbol):
            return self
        elif isinstance(arg, Add):
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
            variances = Add(*(Variance(xv, condition).expand() for xv in rv))
            map_to_covar = lambda x: 2*Covariance(*x, condition=condition).expand()
            covariances = Add(*map(map_to_covar, itertools.combinations(rv, 2)))
            return variances + covariances
        elif isinstance(arg, Mul):
            nonrv = []
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a**2)
            if len(rv) == 0:
                return S.Zero
            return Mul.fromiter(nonrv)*Variance(Mul.fromiter(rv), condition)

        # this expression contains a RandomSymbol somehow:
        return self

    def _eval_rewrite_as_Expectation(self, arg, condition=None, **kwargs):
            e1 = Expectation(arg**2, condition)
            e2 = Expectation(arg, condition)**2
            return e1 - e2

    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return variance(self.args[0], self._condition, evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Covariance(Expr):
    """
    Symbolic expression for the covariance.

    Examples
    ========

    >>> from sympy.stats import Covariance
    >>> from sympy.stats import Normal
    >>> X = Normal("X", 3, 2)
    >>> Y = Normal("Y", 0, 1)
    >>> Z = Normal("Z", 0, 1)
    >>> W = Normal("W", 0, 1)
    >>> cexpr = Covariance(X, Y)
    >>> cexpr
    Covariance(X, Y)

    Evaluate the covariance, `X` and `Y` are independent,
    therefore zero is the result:

    >>> cexpr.evaluate_integral()
    0

    Rewrite the covariance expression in terms of expectations:

    >>> from sympy.stats import Expectation
    >>> cexpr.rewrite(Expectation)
    Expectation(X*Y) - Expectation(X)*Expectation(Y)

    In order to expand the argument, use ``expand()``:

    >>> from sympy.abc import a, b, c, d
    >>> Covariance(a*X + b*Y, c*Z + d*W)
    Covariance(a*X + b*Y, c*Z + d*W)
    >>> Covariance(a*X + b*Y, c*Z + d*W).expand()
    a*c*Covariance(X, Z) + a*d*Covariance(W, X) + b*c*Covariance(Y, Z) + b*d*Covariance(W, Y)

    This class is aware of some properties of the covariance:

    >>> Covariance(X, X).expand()
    Variance(X)
    >>> Covariance(a*X, b*Y).expand()
    a*b*Covariance(X, Y)
    """

    def __new__(cls, arg1, arg2, condition=None, **kwargs):
        arg1 = _sympify(arg1)
        arg2 = _sympify(arg2)

        if arg1.is_Matrix or arg2.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import CrossCovarianceMatrix
            return CrossCovarianceMatrix(arg1, arg2, condition)

        if kwargs.pop('evaluate', global_parameters.evaluate):
            arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)

        if condition is None:
            obj = Expr.__new__(cls, arg1, arg2)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, arg1, arg2, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return self.args[0].is_commutative

    def expand(self, **hints):
        arg1 = self.args[0]
        arg2 = self.args[1]
        condition = self._condition

        if arg1 == arg2:
            return Variance(arg1, condition).expand()

        if not is_random(arg1):
            return S.Zero
        if not is_random(arg2):
            return S.Zero

        arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)

        if isinstance(arg1, RandomSymbol) and isinstance(arg2, RandomSymbol):
            return Covariance(arg1, arg2, condition)

        coeff_rv_list1 = self._expand_single_argument(arg1.expand())
        coeff_rv_list2 = self._expand_single_argument(arg2.expand())

        addends = [a*b*Covariance(*sorted([r1, r2], key=default_sort_key), condition=condition)
                   for (a, r1) in coeff_rv_list1 for (b, r2) in coeff_rv_list2]
        return Add.fromiter(addends)

    @classmethod
    def _expand_single_argument(cls, expr):
        # return (coefficient, random_symbol) pairs:
        if isinstance(expr, RandomSymbol):
            return [(S.One, expr)]
        elif isinstance(expr, Add):
            outval = []
            for a in expr.args:
                if isinstance(a, Mul):
                    outval.append(cls._get_mul_nonrv_rv_tuple(a))
                elif is_random(a):
                    outval.append((S.One, a))

            return outval
        elif isinstance(expr, Mul):
            return [cls._get_mul_nonrv_rv_tuple(expr)]
        elif is_random(expr):
            return [(S.One, expr)]

    @classmethod
    def _get_mul_nonrv_rv_tuple(cls, m):
        rv = []
        nonrv = []
        for a in m.args:
            if is_random(a):
                rv.append(a)
            else:
                nonrv.append(a)
        return (Mul.fromiter(nonrv), Mul.fromiter(rv))

    def _eval_rewrite_as_Expectation(self, arg1, arg2, condition=None, **kwargs):
        e1 = Expectation(arg1*arg2, condition)
        e2 = Expectation(arg1, condition)*Expectation(arg2, condition)
        return e1 - e2

    def _eval_rewrite_as_Probability(self, arg1, arg2, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, arg1, arg2, condition=None, **kwargs):
        return covariance(self.args[0], self.args[1], self._condition, evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Moment(Expr):
    """
    Symbolic class for Moment

    Examples
    ========

    >>> from sympy import Symbol, Integral
    >>> from sympy.stats import Normal, Expectation, Probability, Moment
    >>> mu = Symbol('mu', real=True)
    >>> sigma = Symbol('sigma', positive=True)
    >>> X = Normal('X', mu, sigma)
    >>> M = Moment(X, 3, 1)

    To evaluate the result of Moment use `doit`:

    >>> M.doit()
    mu**3 - 3*mu**2 + 3*mu*sigma**2 + 3*mu - 3*sigma**2 - 1

    Rewrite the Moment expression in terms of Expectation:

    >>> M.rewrite(Expectation)
    Expectation((X - 1)**3)

    Rewrite the Moment expression in terms of Probability:

    >>> M.rewrite(Probability)
    Integral((x - 1)**3*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the Moment expression in terms of Integral:

    >>> M.rewrite(Integral)
    Integral(sqrt(2)*(X - 1)**3*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    """
    def __new__(cls, X, n, c=0, condition=None, **kwargs):
        X = _sympify(X)
        n = _sympify(n)
        c = _sympify(c)
        if condition is not None:
            condition = _sympify(condition)
            return super().__new__(cls, X, n, c, condition)
        else:
            return super().__new__(cls, X, n, c)

    def doit(self, **hints):
        return self.rewrite(Expectation).doit(**hints)

    def _eval_rewrite_as_Expectation(self, X, n, c=0, condition=None, **kwargs):
        return Expectation((X - c)**n, condition)

    def _eval_rewrite_as_Probability(self, X, n, c=0, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, X, n, c=0, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Integral)


class CentralMoment(Expr):
    """
    Symbolic class Central Moment

    Examples
    ========

    >>> from sympy import Symbol, Integral
    >>> from sympy.stats import Normal, Expectation, Probability, CentralMoment
    >>> mu = Symbol('mu', real=True)
    >>> sigma = Symbol('sigma', positive=True)
    >>> X = Normal('X', mu, sigma)
    >>> CM = CentralMoment(X, 4)

    To evaluate the result of CentralMoment use `doit`:

    >>> CM.doit().simplify()
    3*sigma**4

    Rewrite the CentralMoment expression in terms of Expectation:

    >>> CM.rewrite(Expectation)
    Expectation((-Expectation(X) + X)**4)

    Rewrite the CentralMoment expression in terms of Probability:

    >>> CM.rewrite(Probability)
    Integral((x - Integral(x*Probability(True), (x, -oo, oo)))**4*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the CentralMoment expression in terms of Integral:

    >>> CM.rewrite(Integral)
    Integral(sqrt(2)*(X - Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo)))**4*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    """
    def __new__(cls, X, n, condition=None, **kwargs):
        X = _sympify(X)
        n = _sympify(n)
        if condition is not None:
            condition = _sympify(condition)
            return super().__new__(cls, X, n, condition)
        else:
            return super().__new__(cls, X, n)

    def doit(self, **hints):
        return self.rewrite(Expectation).doit(**hints)

    def _eval_rewrite_as_Expectation(self, X, n, condition=None, **kwargs):
        mu = Expectation(X, condition, **kwargs)
        return Moment(X, n, mu, condition, **kwargs).rewrite(Expectation)

    def _eval_rewrite_as_Probability(self, X, n, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, X, n, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Integral)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\internal\well_known_types.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Contains well known classes.

This files defines well known classes which need extra maintenance including:
  - Any
  - Duration
  - FieldMask
  - Struct
  - Timestamp
"""

__author__ = 'jieluo@google.com (Jie Luo)'

import calendar
import collections.abc
import datetime
from typing import Union
import warnings
from google.protobuf.internal import field_mask

FieldMask = field_mask.FieldMask

_TIMESTAMPFORMAT = '%Y-%m-%dT%H:%M:%S'
_NANOS_PER_SECOND = 1000000000
_NANOS_PER_MILLISECOND = 1000000
_NANOS_PER_MICROSECOND = 1000
_MILLIS_PER_SECOND = 1000
_MICROS_PER_SECOND = 1000000
_SECONDS_PER_DAY = 24 * 3600
_DURATION_SECONDS_MAX = 315576000000
_TIMESTAMP_SECONDS_MIN = -62135596800
_TIMESTAMP_SECONDS_MAX = 253402300799

_EPOCH_DATETIME_NAIVE = datetime.datetime(1970, 1, 1, tzinfo=None)
_EPOCH_DATETIME_AWARE = _EPOCH_DATETIME_NAIVE.replace(
    tzinfo=datetime.timezone.utc
)


class Any(object):
  """Class for Any Message type."""

  __slots__ = ()

  def Pack(
      self, msg, type_url_prefix='type.googleapis.com/', deterministic=None
  ):
    """Packs the specified message into current Any message."""
    if len(type_url_prefix) < 1 or type_url_prefix[-1] != '/':
      self.type_url = '%s/%s' % (type_url_prefix, msg.DESCRIPTOR.full_name)
    else:
      self.type_url = '%s%s' % (type_url_prefix, msg.DESCRIPTOR.full_name)
    self.value = msg.SerializeToString(deterministic=deterministic)

  def Unpack(self, msg):
    """Unpacks the current Any message into specified message."""
    descriptor = msg.DESCRIPTOR
    if not self.Is(descriptor):
      return False
    msg.ParseFromString(self.value)
    return True

  def TypeName(self):
    """Returns the protobuf type name of the inner message."""
    # Only last part is to be used: b/25630112
    return self.type_url.rpartition('/')[2]

  def Is(self, descriptor):
    """Checks if this Any represents the given protobuf type."""
    return '/' in self.type_url and self.TypeName() == descriptor.full_name


class Timestamp(object):
  """Class for Timestamp message type."""

  __slots__ = ()

  def ToJsonString(self):
    """Converts Timestamp to RFC 3339 date string format.

    Returns:
      A string converted from timestamp. The string is always Z-normalized
      and uses 3, 6 or 9 fractional digits as required to represent the
      exact time. Example of the return format: '1972-01-01T10:00:20.021Z'
    """
    _CheckTimestampValid(self.seconds, self.nanos)
    nanos = self.nanos
    seconds = self.seconds % _SECONDS_PER_DAY
    days = (self.seconds - seconds) // _SECONDS_PER_DAY
    dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(days, seconds)

    result = dt.isoformat()
    if (nanos % 1e9) == 0:
      # If there are 0 fractional digits, the fractional
      # point '.' should be omitted when serializing.
      return result + 'Z'
    if (nanos % 1e6) == 0:
      # Serialize 3 fractional digits.
      return result + '.%03dZ' % (nanos / 1e6)
    if (nanos % 1e3) == 0:
      # Serialize 6 fractional digits.
      return result + '.%06dZ' % (nanos / 1e3)
    # Serialize 9 fractional digits.
    return result + '.%09dZ' % nanos

  def FromJsonString(self, value):
    """Parse a RFC 3339 date string format to Timestamp.

    Args:
      value: A date string. Any fractional digits (or none) and any offset are
        accepted as long as they fit into nano-seconds precision. Example of
        accepted format: '1972-01-01T10:00:20.021-05:00'

    Raises:
      ValueError: On parsing problems.
    """
    if not isinstance(value, str):
      raise ValueError('Timestamp JSON value not a string: {!r}'.format(value))
    timezone_offset = value.find('Z')
    if timezone_offset == -1:
      timezone_offset = value.find('+')
    if timezone_offset == -1:
      timezone_offset = value.rfind('-')
    if timezone_offset == -1:
      raise ValueError(
          'Failed to parse timestamp: missing valid timezone offset.'
      )
    time_value = value[0:timezone_offset]
    # Parse datetime and nanos.
    point_position = time_value.find('.')
    if point_position == -1:
      second_value = time_value
      nano_value = ''
    else:
      second_value = time_value[:point_position]
      nano_value = time_value[point_position + 1 :]
    if 't' in second_value:
      raise ValueError(
          "time data '{0}' does not match format '%Y-%m-%dT%H:%M:%S', "
          "lowercase 't' is not accepted".format(second_value)
      )
    date_object = datetime.datetime.strptime(second_value, _TIMESTAMPFORMAT)
    td = date_object - datetime.datetime(1970, 1, 1)
    seconds = td.seconds + td.days * _SECONDS_PER_DAY
    if len(nano_value) > 9:
      raise ValueError(
          'Failed to parse Timestamp: nanos {0} more than '
          '9 fractional digits.'.format(nano_value)
      )
    if nano_value:
      nanos = round(float('0.' + nano_value) * 1e9)
    else:
      nanos = 0
    # Parse timezone offsets.
    if value[timezone_offset] == 'Z':
      if len(value) != timezone_offset + 1:
        raise ValueError(
            'Failed to parse timestamp: invalid trailing data {0}.'.format(
                value
            )
        )
    else:
      timezone = value[timezone_offset:]
      pos = timezone.find(':')
      if pos == -1:
        raise ValueError('Invalid timezone offset value: {0}.'.format(timezone))
      if timezone[0] == '+':
        seconds -= (int(timezone[1:pos]) * 60 + int(timezone[pos + 1 :])) * 60
      else:
        seconds += (int(timezone[1:pos]) * 60 + int(timezone[pos + 1 :])) * 60
    # Set seconds and nanos
    _CheckTimestampValid(seconds, nanos)
    self.seconds = int(seconds)
    self.nanos = int(nanos)

  def GetCurrentTime(self):
    """Get the current UTC into Timestamp."""
    self.FromDatetime(datetime.datetime.now(tz=datetime.timezone.utc))

  def ToNanoseconds(self):
    """Converts Timestamp to nanoseconds since epoch."""
    _CheckTimestampValid(self.seconds, self.nanos)
    return self.seconds * _NANOS_PER_SECOND + self.nanos

  def ToMicroseconds(self):
    """Converts Timestamp to microseconds since epoch."""
    _CheckTimestampValid(self.seconds, self.nanos)
    return (
        self.seconds * _MICROS_PER_SECOND + self.nanos // _NANOS_PER_MICROSECOND
    )

  def ToMilliseconds(self):
    """Converts Timestamp to milliseconds since epoch."""
    _CheckTimestampValid(self.seconds, self.nanos)
    return (
        self.seconds * _MILLIS_PER_SECOND + self.nanos // _NANOS_PER_MILLISECOND
    )

  def ToSeconds(self):
    """Converts Timestamp to seconds since epoch."""
    _CheckTimestampValid(self.seconds, self.nanos)
    return self.seconds

  def FromNanoseconds(self, nanos):
    """Converts nanoseconds since epoch to Timestamp."""
    seconds = nanos // _NANOS_PER_SECOND
    nanos = nanos % _NANOS_PER_SECOND
    _CheckTimestampValid(seconds, nanos)
    self.seconds = seconds
    self.nanos = nanos

  def FromMicroseconds(self, micros):
    """Converts microseconds since epoch to Timestamp."""
    seconds = micros // _MICROS_PER_SECOND
    nanos = (micros % _MICROS_PER_SECOND) * _NANOS_PER_MICROSECOND
    _CheckTimestampValid(seconds, nanos)
    self.seconds = seconds
    self.nanos = nanos

  def FromMilliseconds(self, millis):
    """Converts milliseconds since epoch to Timestamp."""
    seconds = millis // _MILLIS_PER_SECOND
    nanos = (millis % _MILLIS_PER_SECOND) * _NANOS_PER_MILLISECOND
    _CheckTimestampValid(seconds, nanos)
    self.seconds = seconds
    self.nanos = nanos

  def FromSeconds(self, seconds):
    """Converts seconds since epoch to Timestamp."""
    _CheckTimestampValid(seconds, 0)
    self.seconds = seconds
    self.nanos = 0

  def ToDatetime(self, tzinfo=None):
    """Converts Timestamp to a datetime.

    Args:
      tzinfo: A datetime.tzinfo subclass; defaults to None.

    Returns:
      If tzinfo is None, returns a timezone-naive UTC datetime (with no timezone
      information, i.e. not aware that it's UTC).

      Otherwise, returns a timezone-aware datetime in the input timezone.
    """
    # Using datetime.fromtimestamp for this would avoid constructing an extra
    # timedelta object and possibly an extra datetime. Unfortunately, that has
    # the disadvantage of not handling the full precision (on all platforms, see
    # https://github.com/python/cpython/issues/109849) or full range (on some
    # platforms, see https://github.com/python/cpython/issues/110042) of
    # datetime.
    _CheckTimestampValid(self.seconds, self.nanos)
    delta = datetime.timedelta(
        seconds=self.seconds,
        microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND),
    )
    if tzinfo is None:
      return _EPOCH_DATETIME_NAIVE + delta
    else:
      # Note the tz conversion has to come after the timedelta arithmetic.
      return (_EPOCH_DATETIME_AWARE + delta).astimezone(tzinfo)

  def FromDatetime(self, dt):
    """Converts datetime to Timestamp.

    Args:
      dt: A datetime. If it's timezone-naive, it's assumed to be in UTC.
    """
    # Using this guide: http://wiki.python.org/moin/WorkingWithTime
    # And this conversion guide: http://docs.python.org/library/time.html

    # Turn the date parameter into a tuple (struct_time) that can then be
    # manipulated into a long value of seconds.  During the conversion from
    # struct_time to long, the source date in UTC, and so it follows that the
    # correct transformation is calendar.timegm()
    try:
      seconds = calendar.timegm(dt.utctimetuple())
      nanos = dt.microsecond * _NANOS_PER_MICROSECOND
    except AttributeError as e:
      raise AttributeError(
          'Fail to convert to Timestamp. Expected a datetime like '
          'object got {0} : {1}'.format(type(dt).__name__, e)
      ) from e
    _CheckTimestampValid(seconds, nanos)
    self.seconds = seconds
    self.nanos = nanos

  def _internal_assign(self, dt):
    self.FromDatetime(dt)

  def __add__(self, value) -> datetime.datetime:
    if isinstance(value, Duration):
      return self.ToDatetime() + value.ToTimedelta()
    return self.ToDatetime() + value

  __radd__ = __add__

  def __sub__(self, value) -> Union[datetime.datetime, datetime.timedelta]:
    if isinstance(value, Timestamp):
      return self.ToDatetime() - value.ToDatetime()
    elif isinstance(value, Duration):
      return self.ToDatetime() - value.ToTimedelta()
    return self.ToDatetime() - value

  def __rsub__(self, dt) -> datetime.timedelta:
    return dt - self.ToDatetime()


def _CheckTimestampValid(seconds, nanos):
  if seconds < _TIMESTAMP_SECONDS_MIN or seconds > _TIMESTAMP_SECONDS_MAX:
    raise ValueError(
        'Timestamp is not valid: Seconds {0} must be in range '
        '[-62135596800, 253402300799].'.format(seconds))
  if nanos < 0 or nanos >= _NANOS_PER_SECOND:
    raise ValueError(
        'Timestamp is not valid: Nanos {} must be in a range '
        '[0, 999999].'.format(nanos)
    )


class Duration(object):
  """Class for Duration message type."""

  __slots__ = ()

  def ToJsonString(self):
    """Converts Duration to string format.

    Returns:
      A string converted from self. The string format will contains
      3, 6, or 9 fractional digits depending on the precision required to
      represent the exact Duration value. For example: "1s", "1.010s",
      "1.000000100s", "-3.100s"
    """
    _CheckDurationValid(self.seconds, self.nanos)
    if self.seconds < 0 or self.nanos < 0:
      result = '-'
      seconds = -self.seconds + int((0 - self.nanos) // 1e9)
      nanos = (0 - self.nanos) % 1e9
    else:
      result = ''
      seconds = self.seconds + int(self.nanos // 1e9)
      nanos = self.nanos % 1e9
    result += '%d' % seconds
    if (nanos % 1e9) == 0:
      # If there are 0 fractional digits, the fractional
      # point '.' should be omitted when serializing.
      return result + 's'
    if (nanos % 1e6) == 0:
      # Serialize 3 fractional digits.
      return result + '.%03ds' % (nanos / 1e6)
    if (nanos % 1e3) == 0:
      # Serialize 6 fractional digits.
      return result + '.%06ds' % (nanos / 1e3)
    # Serialize 9 fractional digits.
    return result + '.%09ds' % nanos

  def FromJsonString(self, value):
    """Converts a string to Duration.

    Args:
      value: A string to be converted. The string must end with 's'. Any
        fractional digits (or none) are accepted as long as they fit into
        precision. For example: "1s", "1.01s", "1.0000001s", "-3.100s

    Raises:
      ValueError: On parsing problems.
    """
    if not isinstance(value, str):
      raise ValueError('Duration JSON value not a string: {!r}'.format(value))
    if len(value) < 1 or value[-1] != 's':
      raise ValueError('Duration must end with letter "s": {0}.'.format(value))
    try:
      pos = value.find('.')
      if pos == -1:
        seconds = int(value[:-1])
        nanos = 0
      else:
        seconds = int(value[:pos])
        if value[0] == '-':
          nanos = int(round(float('-0{0}'.format(value[pos:-1])) * 1e9))
        else:
          nanos = int(round(float('0{0}'.format(value[pos:-1])) * 1e9))
      _CheckDurationValid(seconds, nanos)
      self.seconds = seconds
      self.nanos = nanos
    except ValueError as e:
      raise ValueError("Couldn't parse duration: {0} : {1}.".format(value, e))

  def ToNanoseconds(self):
    """Converts a Duration to nanoseconds."""
    return self.seconds * _NANOS_PER_SECOND + self.nanos

  def ToMicroseconds(self):
    """Converts a Duration to microseconds."""
    micros = _RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND)
    return self.seconds * _MICROS_PER_SECOND + micros

  def ToMilliseconds(self):
    """Converts a Duration to milliseconds."""
    millis = _RoundTowardZero(self.nanos, _NANOS_PER_MILLISECOND)
    return self.seconds * _MILLIS_PER_SECOND + millis

  def ToSeconds(self):
    """Converts a Duration to seconds."""
    return self.seconds

  def FromNanoseconds(self, nanos):
    """Converts nanoseconds to Duration."""
    self._NormalizeDuration(
        nanos // _NANOS_PER_SECOND, nanos % _NANOS_PER_SECOND
    )

  def FromMicroseconds(self, micros):
    """Converts microseconds to Duration."""
    self._NormalizeDuration(
        micros // _MICROS_PER_SECOND,
        (micros % _MICROS_PER_SECOND) * _NANOS_PER_MICROSECOND,
    )

  def FromMilliseconds(self, millis):
    """Converts milliseconds to Duration."""
    self._NormalizeDuration(
        millis // _MILLIS_PER_SECOND,
        (millis % _MILLIS_PER_SECOND) * _NANOS_PER_MILLISECOND,
    )

  def FromSeconds(self, seconds):
    """Converts seconds to Duration."""
    self.seconds = seconds
    self.nanos = 0

  def ToTimedelta(self) -> datetime.timedelta:
    """Converts Duration to timedelta."""
    return datetime.timedelta(
        seconds=self.seconds,
        microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND),
    )

  def FromTimedelta(self, td):
    """Converts timedelta to Duration."""
    try:
      self._NormalizeDuration(
          td.seconds + td.days * _SECONDS_PER_DAY,
          td.microseconds * _NANOS_PER_MICROSECOND,
      )
    except AttributeError as e:
      raise AttributeError(
          'Fail to convert to Duration. Expected a timedelta like '
          'object got {0}: {1}'.format(type(td).__name__, e)
      ) from e

  def _internal_assign(self, td):
    self.FromTimedelta(td)

  def _NormalizeDuration(self, seconds, nanos):
    """Set Duration by seconds and nanos."""
    # Force nanos to be negative if the duration is negative.
    if seconds < 0 and nanos > 0:
      seconds += 1
      nanos -= _NANOS_PER_SECOND
    self.seconds = seconds
    self.nanos = nanos

  def __add__(self, value) -> Union[datetime.datetime, datetime.timedelta]:
    if isinstance(value, Timestamp):
      return self.ToTimedelta() + value.ToDatetime()
    return self.ToTimedelta() + value

  __radd__ = __add__

  def __sub__(self, value) -> datetime.timedelta:
    return self.ToTimedelta() - value

  def __rsub__(self, value) -> Union[datetime.datetime, datetime.timedelta]:
    return value - self.ToTimedelta()


def _CheckDurationValid(seconds, nanos):
  if seconds < -_DURATION_SECONDS_MAX or seconds > _DURATION_SECONDS_MAX:
    raise ValueError(
        'Duration is not valid: Seconds {0} must be in range '
        '[-315576000000, 315576000000].'.format(seconds)
    )
  if nanos <= -_NANOS_PER_SECOND or nanos >= _NANOS_PER_SECOND:
    raise ValueError(
        'Duration is not valid: Nanos {0} must be in range '
        '[-999999999, 999999999].'.format(nanos)
    )
  if (nanos < 0 and seconds > 0) or (nanos > 0 and seconds < 0):
    raise ValueError('Duration is not valid: Sign mismatch.')


def _RoundTowardZero(value, divider):
  """Truncates the remainder part after division."""
  # For some languages, the sign of the remainder is implementation
  # dependent if any of the operands is negative. Here we enforce
  # "rounded toward zero" semantics. For example, for (-5) / 2 an
  # implementation may give -3 as the result with the remainder being
  # 1. This function ensures we always return -2 (closer to zero).
  result = value // divider
  remainder = value % divider
  if result < 0 and remainder > 0:
    return result + 1
  else:
    return result


def _SetStructValue(struct_value, value):
  if value is None:
    struct_value.null_value = 0
  elif isinstance(value, bool):
    # Note: this check must come before the number check because in Python
    # True and False are also considered numbers.
    struct_value.bool_value = value
  elif isinstance(value, str):
    struct_value.string_value = value
  elif isinstance(value, (int, float)):
    struct_value.number_value = value
  elif isinstance(value, (dict, Struct)):
    struct_value.struct_value.Clear()
    struct_value.struct_value.update(value)
  elif isinstance(value, (list, tuple, ListValue)):
    struct_value.list_value.Clear()
    struct_value.list_value.extend(value)
  else:
    raise ValueError('Unexpected type')


def _GetStructValue(struct_value):
  which = struct_value.WhichOneof('kind')
  if which == 'struct_value':
    return struct_value.struct_value
  elif which == 'null_value':
    return None
  elif which == 'number_value':
    return struct_value.number_value
  elif which == 'string_value':
    return struct_value.string_value
  elif which == 'bool_value':
    return struct_value.bool_value
  elif which == 'list_value':
    return struct_value.list_value
  elif which is None:
    raise ValueError('Value not set')


class Struct(object):
  """Class for Struct message type."""

  __slots__ = ()

  def __getitem__(self, key):
    return _GetStructValue(self.fields[key])

  def __setitem__(self, key, value):
    _SetStructValue(self.fields[key], value)

  def __delitem__(self, key):
    del self.fields[key]

  def __len__(self):
    return len(self.fields)

  def __iter__(self):
    return iter(self.fields)

  def _internal_assign(self, dictionary):
    self.Clear()
    self.update(dictionary)

  def _internal_compare(self, other):
    size = len(self)
    if size != len(other):
      return False
    for key, value in self.items():
      if key not in other:
        return False
      if isinstance(other[key], (dict, list)):
        if not value._internal_compare(other[key]):
          return False
      elif value != other[key]:
        return False
    return True

  def keys(self):  # pylint: disable=invalid-name
    return self.fields.keys()

  def values(self):  # pylint: disable=invalid-name
    return [self[key] for key in self]

  def items(self):  # pylint: disable=invalid-name
    return [(key, self[key]) for key in self]

  def get_or_create_list(self, key):
    """Returns a list for this key, creating if it didn't exist already."""
    if not self.fields[key].HasField('list_value'):
      # Clear will mark list_value modified which will indeed create a list.
      self.fields[key].list_value.Clear()
    return self.fields[key].list_value

  def get_or_create_struct(self, key):
    """Returns a struct for this key, creating if it didn't exist already."""
    if not self.fields[key].HasField('struct_value'):
      # Clear will mark struct_value modified which will indeed create a struct.
      self.fields[key].struct_value.Clear()
    return self.fields[key].struct_value

  def update(self, dictionary):  # pylint: disable=invalid-name
    for key, value in dictionary.items():
      _SetStructValue(self.fields[key], value)


collections.abc.MutableMapping.register(Struct)


class ListValue(object):
  """Class for ListValue message type."""

  __slots__ = ()

  def __len__(self):
    return len(self.values)

  def append(self, value):
    _SetStructValue(self.values.add(), value)

  def extend(self, elem_seq):
    for value in elem_seq:
      self.append(value)

  def __getitem__(self, index):
    """Retrieves item by the specified index."""
    return _GetStructValue(self.values.__getitem__(index))

  def __setitem__(self, index, value):
    _SetStructValue(self.values.__getitem__(index), value)

  def __delitem__(self, key):
    del self.values[key]

  def _internal_assign(self, elem_seq):
    self.Clear()
    self.extend(elem_seq)

  def _internal_compare(self, other):
    size = len(self)
    if size != len(other):
      return False
    for i in range(size):
      if isinstance(other[i], (dict, list)):
        if not self[i]._internal_compare(other[i]):
          return False
      elif self[i] != other[i]:
        return False
    return True

  def items(self):
    for i in range(len(self)):
      yield self[i]

  def add_struct(self):
    """Appends and returns a struct value as the next value in the list."""
    struct_value = self.values.add().struct_value
    # Clear will mark struct_value modified which will indeed create a struct.
    struct_value.Clear()
    return struct_value

  def add_list(self):
    """Appends and returns a list value as the next value in the list."""
    list_value = self.values.add().list_value
    # Clear will mark list_value modified which will indeed create a list.
    list_value.Clear()
    return list_value


collections.abc.MutableSequence.register(ListValue)


# LINT.IfChange(wktbases)
WKTBASES = {
    'google.protobuf.Any': Any,
    'google.protobuf.Duration': Duration,
    'google.protobuf.FieldMask': FieldMask,
    'google.protobuf.ListValue': ListValue,
    'google.protobuf.Struct': Struct,
    'google.protobuf.Timestamp': Timestamp,
}
# LINT.ThenChange(//depot/google.protobuf/compiler/python/pyi_generator.cc:wktbases)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_highlevel_open_tcp_stream.py ===
from __future__ import annotations

import socket
import sys
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING

import attrs
import pytest

import trio
from trio._highlevel_open_tcp_stream import (
    close_all,
    format_host_port,
    open_tcp_stream,
    reorder_for_rfc_6555_section_5_4,
)
from trio.socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, SocketType
from trio.testing import Matcher, RaisesGroup

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trio.testing import MockClock

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


def test_close_all() -> None:
    class CloseMe(SocketType):
        closed = False

        def close(self) -> None:
            self.closed = True

    class CloseKiller(SocketType):
        def close(self) -> None:
            raise OSError("os error text")

    c: CloseMe = CloseMe()
    with close_all() as to_close:
        to_close.add(c)
    assert c.closed

    c = CloseMe()
    with pytest.raises(RuntimeError):
        with close_all() as to_close:
            to_close.add(c)
            raise RuntimeError
    assert c.closed

    c = CloseMe()
    with pytest.raises(OSError, match="os error text"):
        with close_all() as to_close:
            to_close.add(CloseKiller())
            to_close.add(c)
    assert c.closed


def test_reorder_for_rfc_6555_section_5_4() -> None:
    def fake4(
        i: int,
    ) -> tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]:
        return (
            AF_INET,
            SOCK_STREAM,
            IPPROTO_TCP,
            "",
            (f"10.0.0.{i}", 80),
        )

    def fake6(
        i: int,
    ) -> tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]:
        return (AF_INET6, SOCK_STREAM, IPPROTO_TCP, "", (f"::{i}", 80))

    for fake in fake4, fake6:
        # No effect on homogeneous lists
        targets = [fake(0), fake(1), fake(2)]
        reorder_for_rfc_6555_section_5_4(targets)
        assert targets == [fake(0), fake(1), fake(2)]

        # Single item lists also OK
        targets = [fake(0)]
        reorder_for_rfc_6555_section_5_4(targets)
        assert targets == [fake(0)]

    # If the list starts out with different families in positions 0 and 1,
    # then it's left alone
    orig = [fake4(0), fake6(0), fake4(1), fake6(1)]
    targets = list(orig)
    reorder_for_rfc_6555_section_5_4(targets)
    assert targets == orig

    # If not, it's reordered
    targets = [fake4(0), fake4(1), fake4(2), fake6(0), fake6(1)]
    reorder_for_rfc_6555_section_5_4(targets)
    assert targets == [fake4(0), fake6(0), fake4(1), fake4(2), fake6(1)]


def test_format_host_port() -> None:
    assert format_host_port("127.0.0.1", 80) == "127.0.0.1:80"
    assert format_host_port(b"127.0.0.1", 80) == "127.0.0.1:80"
    assert format_host_port("example.com", 443) == "example.com:443"
    assert format_host_port(b"example.com", 443) == "example.com:443"
    assert format_host_port("::1", "http") == "[::1]:http"
    assert format_host_port(b"::1", "http") == "[::1]:http"


# Make sure we can connect to localhost using real kernel sockets
async def test_open_tcp_stream_real_socket_smoketest() -> None:
    listen_sock = trio.socket.socket()
    await listen_sock.bind(("127.0.0.1", 0))
    _, listen_port = listen_sock.getsockname()
    listen_sock.listen(1)
    client_stream = await open_tcp_stream("127.0.0.1", listen_port)
    server_sock, _ = await listen_sock.accept()
    await client_stream.send_all(b"x")
    assert await server_sock.recv(1) == b"x"
    await client_stream.aclose()
    server_sock.close()

    listen_sock.close()


async def test_open_tcp_stream_input_validation() -> None:
    with pytest.raises(ValueError, match=r"^host must be str or bytes, not None$"):
        await open_tcp_stream(None, 80)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        await open_tcp_stream("127.0.0.1", b"80")  # type: ignore[arg-type]


def can_bind_127_0_0_2() -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.2", 0))
        except OSError:
            return False
        # s.getsockname() is typed as returning Any
        return s.getsockname()[0] == "127.0.0.2"  # type: ignore[no-any-return]


async def test_local_address_real() -> None:
    with trio.socket.socket() as listener:
        await listener.bind(("127.0.0.1", 0))
        listener.listen()

        # It's hard to test local_address properly, because you need multiple
        # local addresses that you can bind to. Fortunately, on most Linux
        # systems, you can bind to any 127.*.*.* address, and they all go
        # through the loopback interface. So we can use a non-standard
        # loopback address. On other systems, the only address we know for
        # certain we have is 127.0.0.1, so we can't really test local_address=
        # properly -- passing local_address=127.0.0.1 is indistinguishable
        # from not passing local_address= at all. But, we can still do a smoke
        # test to make sure the local_address= code doesn't crash.
        local_address = "127.0.0.2" if can_bind_127_0_0_2() else "127.0.0.1"

        async with await open_tcp_stream(
            *listener.getsockname(),
            local_address=local_address,
        ) as client_stream:
            assert client_stream.socket.getsockname()[0] == local_address
            if hasattr(trio.socket, "IP_BIND_ADDRESS_NO_PORT"):
                assert client_stream.socket.getsockopt(
                    trio.socket.IPPROTO_IP,
                    trio.socket.IP_BIND_ADDRESS_NO_PORT,
                )
            server_sock, remote_addr = await listener.accept()
            await client_stream.aclose()
            server_sock.close()
            # accept returns tuple[SocketType, object], due to typeshed returning `Any`
            assert remote_addr[0] == local_address

        # Trying to connect to an ipv4 address with the ipv6 wildcard
        # local_address should fail
        with pytest.raises(
            OSError,
            match=r"^all attempts to connect* to *127\.0\.0\.\d:\d+ failed$",
        ):
            await open_tcp_stream(*listener.getsockname(), local_address="::")

        # But the ipv4 wildcard address should work
        async with await open_tcp_stream(
            *listener.getsockname(),
            local_address="0.0.0.0",
        ) as client_stream:
            server_sock, remote_addr = await listener.accept()
            server_sock.close()
            assert remote_addr == client_stream.socket.getsockname()


# Now, thorough tests using fake sockets


@attrs.define(eq=False, slots=False)
class FakeSocket(trio.socket.SocketType):
    scenario: Scenario
    _family: AddressFamily
    _type: SocketKind
    _proto: int

    ip: str | int | None = None
    port: str | int | None = None
    succeeded: bool = False
    closed: bool = False
    failing: bool = False

    @property
    def type(self) -> SocketKind:
        return self._type

    @property
    def family(self) -> AddressFamily:  # pragma: no cover
        return self._family

    @property
    def proto(self) -> int:  # pragma: no cover
        return self._proto

    async def connect(self, sockaddr: tuple[str | int, str | int | None]) -> None:
        self.ip = sockaddr[0]
        self.port = sockaddr[1]
        assert self.ip not in self.scenario.sockets
        self.scenario.sockets[self.ip] = self
        self.scenario.connect_times[self.ip] = trio.current_time()
        delay, result = self.scenario.ip_dict[self.ip]
        await trio.sleep(delay)
        if result == "error":
            raise OSError("sorry")
        if result == "postconnect_fail":
            self.failing = True
        self.succeeded = True

    def close(self) -> None:
        self.closed = True

    # called when SocketStream is constructed
    def setsockopt(self, *args: object, **kwargs: object) -> None:
        if self.failing:
            # raise something that isn't OSError as SocketStream
            # ignores those
            raise KeyboardInterrupt


class Scenario(trio.abc.SocketFactory, trio.abc.HostnameResolver):
    def __init__(
        self,
        port: int,
        ip_list: Sequence[tuple[str, float, str]],
        supported_families: set[AddressFamily],
    ) -> None:
        # ip_list have to be unique
        ip_order = [ip for (ip, _, _) in ip_list]
        assert len(set(ip_order)) == len(ip_list)
        ip_dict: dict[str | int, tuple[float, str]] = {}
        for ip, delay, result in ip_list:
            assert delay >= 0
            assert result in ["error", "success", "postconnect_fail"]
            ip_dict[ip] = (delay, result)

        self.port = port
        self.ip_order = ip_order
        self.ip_dict = ip_dict
        self.supported_families = supported_families
        self.socket_count = 0
        self.sockets: dict[str | int, FakeSocket] = {}
        self.connect_times: dict[str | int, float] = {}

    def socket(
        self,
        family: AddressFamily | int | None = None,
        type_: SocketKind | int | None = None,
        proto: int | None = None,
    ) -> SocketType:
        assert isinstance(family, AddressFamily)
        assert isinstance(type_, SocketKind)
        assert proto is not None
        if family not in self.supported_families:
            raise OSError("pretending not to support this family")
        self.socket_count += 1
        return FakeSocket(self, family, type_, proto)

    def _ip_to_gai_entry(self, ip: str) -> tuple[
        AddressFamily,
        SocketKind,
        int,
        str,
        tuple[str, int, int, int] | tuple[str, int] | tuple[int, bytes],
    ]:
        sockaddr: tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes]
        if ":" in ip:
            family = trio.socket.AF_INET6
            sockaddr = (ip, self.port, 0, 0)
        else:
            family = trio.socket.AF_INET
            sockaddr = (ip, self.port)
        return (family, SOCK_STREAM, IPPROTO_TCP, "", sockaddr)

    async def getaddrinfo(
        self,
        host: bytes | None,
        port: bytes | str | int | None,
        family: int = -1,
        type: int = -1,
        proto: int = -1,
        flags: int = -1,
    ) -> list[
        tuple[
            AddressFamily,
            SocketKind,
            int,
            str,
            tuple[str, int, int, int] | tuple[str, int] | tuple[int, bytes],
        ]
    ]:
        assert host == b"test.example.com"
        assert port == self.port
        assert family == trio.socket.AF_UNSPEC
        assert type == trio.socket.SOCK_STREAM
        assert proto == 0
        assert flags == 0
        return [self._ip_to_gai_entry(ip) for ip in self.ip_order]

    async def getnameinfo(
        self,
        sockaddr: tuple[str, int] | tuple[str, int, int, int],
        flags: int,
    ) -> tuple[str, str]:
        raise NotImplementedError

    def check(self, succeeded: SocketType | None) -> None:
        # sockets only go into self.sockets when connect is called; make sure
        # all the sockets that were created did in fact go in there.
        assert self.socket_count == len(self.sockets)

        for ip, socket_ in self.sockets.items():
            assert ip in self.ip_dict
            if socket_ is not succeeded:
                assert socket_.closed
            assert socket_.port == self.port


async def run_scenario(
    # The port to connect to
    port: int,
    # A list of
    #  (ip, delay, result)
    # tuples, where delay is in seconds and result is "success" or "error"
    # The ip's will be returned from getaddrinfo in this order, and then
    # connect() calls to them will have the given result.
    ip_list: Sequence[tuple[str, float, str]],
    *,
    # If False, AF_INET4/6 sockets error out on creation, before connect is
    # even called.
    ipv4_supported: bool = True,
    ipv6_supported: bool = True,
    # Normally, we return (winning_sock, scenario object)
    # If this is True, we require there to be an exception, and return
    #   (exception, scenario object)
    expect_error: tuple[type[BaseException], ...] | type[BaseException] = (),
    happy_eyeballs_delay: float | None = 0.25,
    local_address: str | None = None,
) -> tuple[SocketType, Scenario] | tuple[BaseException, Scenario]:
    supported_families = set()
    if ipv4_supported:
        supported_families.add(trio.socket.AF_INET)
    if ipv6_supported:
        supported_families.add(trio.socket.AF_INET6)
    scenario = Scenario(port, ip_list, supported_families)
    trio.socket.set_custom_hostname_resolver(scenario)
    trio.socket.set_custom_socket_factory(scenario)

    try:
        stream = await open_tcp_stream(
            "test.example.com",
            port,
            happy_eyeballs_delay=happy_eyeballs_delay,
            local_address=local_address,
        )
        assert expect_error == ()
        scenario.check(stream.socket)
        return (stream.socket, scenario)
    except AssertionError:  # pragma: no cover
        raise
    except expect_error as exc:
        scenario.check(None)
        return (exc, scenario)


async def test_one_host_quick_success(autojump_clock: MockClock) -> None:
    sock, _scenario = await run_scenario(80, [("1.2.3.4", 0.123, "success")])
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "1.2.3.4"
    assert trio.current_time() == 0.123


async def test_one_host_slow_success(autojump_clock: MockClock) -> None:
    sock, _scenario = await run_scenario(81, [("1.2.3.4", 100, "success")])
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "1.2.3.4"
    assert trio.current_time() == 100


async def test_one_host_quick_fail(autojump_clock: MockClock) -> None:
    exc, _scenario = await run_scenario(
        82,
        [("1.2.3.4", 0.123, "error")],
        expect_error=OSError,
    )
    assert isinstance(exc, OSError)
    assert trio.current_time() == 0.123


async def test_one_host_slow_fail(autojump_clock: MockClock) -> None:
    exc, _scenario = await run_scenario(
        83,
        [("1.2.3.4", 100, "error")],
        expect_error=OSError,
    )
    assert isinstance(exc, OSError)
    assert trio.current_time() == 100


async def test_one_host_failed_after_connect(autojump_clock: MockClock) -> None:
    exc, _scenario = await run_scenario(
        83,
        [("1.2.3.4", 1, "postconnect_fail")],
        expect_error=KeyboardInterrupt,
    )
    assert isinstance(exc, KeyboardInterrupt)


# With the default 0.250 second delay, the third attempt will win
async def test_basic_fallthrough(autojump_clock: MockClock) -> None:
    sock, scenario = await run_scenario(
        80,
        [
            ("1.1.1.1", 1, "success"),
            ("2.2.2.2", 1, "success"),
            ("3.3.3.3", 0.2, "success"),
        ],
    )
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "3.3.3.3"
    # current time is default time + default time + connection time
    assert trio.current_time() == (0.250 + 0.250 + 0.2)
    assert scenario.connect_times == {
        "1.1.1.1": 0,
        "2.2.2.2": 0.250,
        "3.3.3.3": 0.500,
    }


async def test_early_success(autojump_clock: MockClock) -> None:
    sock, scenario = await run_scenario(
        80,
        [
            ("1.1.1.1", 1, "success"),
            ("2.2.2.2", 0.1, "success"),
            ("3.3.3.3", 0.2, "success"),
        ],
    )
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "2.2.2.2"
    assert trio.current_time() == (0.250 + 0.1)
    assert scenario.connect_times == {
        "1.1.1.1": 0,
        "2.2.2.2": 0.250,
        # 3.3.3.3 was never even started
    }


# With a 0.450 second delay, the first attempt will win
async def test_custom_delay(autojump_clock: MockClock) -> None:
    sock, scenario = await run_scenario(
        80,
        [
            ("1.1.1.1", 1, "success"),
            ("2.2.2.2", 1, "success"),
            ("3.3.3.3", 0.2, "success"),
        ],
        happy_eyeballs_delay=0.450,
    )
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "1.1.1.1"
    assert trio.current_time() == 1
    assert scenario.connect_times == {
        "1.1.1.1": 0,
        "2.2.2.2": 0.450,
        "3.3.3.3": 0.900,
    }


async def test_none_default(autojump_clock: MockClock) -> None:
    """Copy of test_basic_fallthrough, but specifying the delay =None"""
    sock, scenario = await run_scenario(
        80,
        [
            ("1.1.1.1", 1, "success"),
            ("2.2.2.2", 1, "success"),
            ("3.3.3.3", 0.2, "success"),
        ],
        happy_eyeballs_delay=None,
    )
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "3.3.3.3"
    # current time is default time + default time + connection time
    assert trio.current_time() == (0.250 + 0.250 + 0.2)
    assert scenario.connect_times == {
        "1.1.1.1": 0,
        "2.2.2.2": 0.250,
        "3.3.3.3": 0.500,
    }


async def test_custom_errors_expedite(autojump_clock: MockClock) -> None:
    sock, scenario = await run_scenario(
        80,
        [
            ("1.1.1.1", 0.1, "error"),
            ("2.2.2.2", 0.2, "error"),
            ("3.3.3.3", 10, "success"),
            # .25 is the default timeout
            ("4.4.4.4", 0.25, "success"),
        ],
    )
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "4.4.4.4"
    assert trio.current_time() == (0.1 + 0.2 + 0.25 + 0.25)
    assert scenario.connect_times == {
        "1.1.1.1": 0,
        "2.2.2.2": 0.1,
        "3.3.3.3": 0.1 + 0.2,
        "4.4.4.4": 0.1 + 0.2 + 0.25,
    }


async def test_all_fail(autojump_clock: MockClock) -> None:
    exc, scenario = await run_scenario(
        80,
        [
            ("1.1.1.1", 0.1, "error"),
            ("2.2.2.2", 0.2, "error"),
            ("3.3.3.3", 10, "error"),
            ("4.4.4.4", 0.250, "error"),
        ],
        expect_error=OSError,
    )
    assert isinstance(exc, OSError)

    subexceptions = (Matcher(OSError, match="^sorry$"),) * 4
    assert RaisesGroup(
        *subexceptions,
        match="all attempts to connect to test.example.com:80 failed",
    ).matches(exc.__cause__)

    assert trio.current_time() == (0.1 + 0.2 + 10)
    assert scenario.connect_times == {
        "1.1.1.1": 0,
        "2.2.2.2": 0.1,
        "3.3.3.3": 0.1 + 0.2,
        "4.4.4.4": 0.1 + 0.2 + 0.25,
    }


async def test_multi_success(autojump_clock: MockClock) -> None:
    sock, scenario = await run_scenario(
        80,
        [
            ("1.1.1.1", 0.5, "error"),
            ("2.2.2.2", 10, "success"),
            ("3.3.3.3", 10 - 1, "success"),
            ("4.4.4.4", 10 - 2, "success"),
            ("5.5.5.5", 0.5, "error"),
        ],
        happy_eyeballs_delay=1,
    )
    assert not scenario.sockets["1.1.1.1"].succeeded
    assert (
        scenario.sockets["2.2.2.2"].succeeded
        or scenario.sockets["3.3.3.3"].succeeded
        or scenario.sockets["4.4.4.4"].succeeded
    )
    assert not scenario.sockets["5.5.5.5"].succeeded
    assert isinstance(sock, FakeSocket)
    assert sock.ip in ["2.2.2.2", "3.3.3.3", "4.4.4.4"]
    assert trio.current_time() == (0.5 + 10)
    assert scenario.connect_times == {
        "1.1.1.1": 0,
        "2.2.2.2": 0.5,
        "3.3.3.3": 1.5,
        "4.4.4.4": 2.5,
        "5.5.5.5": 3.5,
    }


async def test_does_reorder(autojump_clock: MockClock) -> None:
    sock, scenario = await run_scenario(
        80,
        [
            ("1.1.1.1", 10, "error"),
            # This would win if we tried it first...
            ("2.2.2.2", 1, "success"),
            # But in fact we try this first, because of section 5.4
            ("::3", 0.5, "success"),
        ],
        happy_eyeballs_delay=1,
    )
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "::3"
    assert trio.current_time() == 1 + 0.5
    assert scenario.connect_times == {
        "1.1.1.1": 0,
        "::3": 1,
    }


async def test_handles_no_ipv4(autojump_clock: MockClock) -> None:
    sock, scenario = await run_scenario(
        80,
        # Here the ipv6 addresses fail at socket creation time, so the connect
        # configuration doesn't matter
        [
            ("::1", 10, "success"),
            ("2.2.2.2", 0, "success"),
            ("::3", 0.1, "success"),
            ("4.4.4.4", 0, "success"),
        ],
        happy_eyeballs_delay=1,
        ipv4_supported=False,
    )
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "::3"
    assert trio.current_time() == 1 + 0.1
    assert scenario.connect_times == {
        "::1": 0,
        "::3": 1.0,
    }


async def test_handles_no_ipv6(autojump_clock: MockClock) -> None:
    sock, scenario = await run_scenario(
        80,
        # Here the ipv6 addresses fail at socket creation time, so the connect
        # configuration doesn't matter
        [
            ("::1", 0, "success"),
            ("2.2.2.2", 10, "success"),
            ("::3", 0, "success"),
            ("4.4.4.4", 0.1, "success"),
        ],
        happy_eyeballs_delay=1,
        ipv6_supported=False,
    )
    assert isinstance(sock, FakeSocket)
    assert sock.ip == "4.4.4.4"
    assert trio.current_time() == 1 + 0.1
    assert scenario.connect_times == {
        "2.2.2.2": 0,
        "4.4.4.4": 1.0,
    }


async def test_no_hosts(autojump_clock: MockClock) -> None:
    exc, _scenario = await run_scenario(80, [], expect_error=OSError)
    assert "no results found" in str(exc)


async def test_cancel(autojump_clock: MockClock) -> None:
    with trio.move_on_after(5) as cancel_scope:
        exc, scenario = await run_scenario(
            80,
            [
                ("1.1.1.1", 10, "success"),
                ("2.2.2.2", 10, "success"),
                ("3.3.3.3", 10, "success"),
                ("4.4.4.4", 10, "success"),
            ],
            expect_error=BaseExceptionGroup,
        )
        assert isinstance(exc, BaseException)
        # What comes out should be 1 or more Cancelled errors that all belong
        # to this cancel_scope; this is the easiest way to check that
        raise exc
    assert cancel_scope.cancelled_caught

    assert trio.current_time() == 5

    # This should have been called already, but just to make sure, since the
    # exception-handling logic in run_scenario is a bit complicated and the
    # main thing we care about here is that all the sockets were cleaned up.
    scenario.check(succeeded=None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\loaders.py ===
"""API and implementations for loading templates from different data
sources.
"""

import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType

from .exceptions import TemplateNotFound
from .utils import internalcode

if t.TYPE_CHECKING:
    from .environment import Environment
    from .environment import Template


def split_template_path(template: str) -> t.List[str]:
    """Split a path into segments and perform a sanity check.  If it detects
    '..' in the path it will raise a `TemplateNotFound` error.
    """
    pieces = []
    for piece in template.split("/"):
        if (
            os.path.sep in piece
            or (os.path.altsep and os.path.altsep in piece)
            or piece == os.path.pardir
        ):
            raise TemplateNotFound(template)
        elif piece and piece != ".":
            pieces.append(piece)
    return pieces


class BaseLoader:
    """Baseclass for all loaders.  Subclass this and override `get_source` to
    implement a custom loading mechanism.  The environment provides a
    `get_template` method that calls the loader's `load` method to get the
    :class:`Template` object.

    A very basic example for a loader that looks up templates on the file
    system could look like this::

        from jinja2 import BaseLoader, TemplateNotFound
        from os.path import join, exists, getmtime

        class MyLoader(BaseLoader):

            def __init__(self, path):
                self.path = path

            def get_source(self, environment, template):
                path = join(self.path, template)
                if not exists(path):
                    raise TemplateNotFound(template)
                mtime = getmtime(path)
                with open(path) as f:
                    source = f.read()
                return source, path, lambda: mtime == getmtime(path)
    """

    #: if set to `False` it indicates that the loader cannot provide access
    #: to the source of templates.
    #:
    #: .. versionadded:: 2.4
    has_source_access = True

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        """Get the template source, filename and reload helper for a template.
        It's passed the environment and template name and has to return a
        tuple in the form ``(source, filename, uptodate)`` or raise a
        `TemplateNotFound` error if it can't locate the template.

        The source part of the returned tuple must be the source of the
        template as a string. The filename should be the name of the
        file on the filesystem if it was loaded from there, otherwise
        ``None``. The filename is used by Python for the tracebacks
        if no loader extension is used.

        The last item in the tuple is the `uptodate` function.  If auto
        reloading is enabled it's always called to check if the template
        changed.  No arguments are passed so the function must store the
        old state somewhere (for example in a closure).  If it returns `False`
        the template will be reloaded.
        """
        if not self.has_source_access:
            raise RuntimeError(
                f"{type(self).__name__} cannot provide access to the source"
            )
        raise TemplateNotFound(template)

    def list_templates(self) -> t.List[str]:
        """Iterates over all templates.  If the loader does not support that
        it should raise a :exc:`TypeError` which is the default behavior.
        """
        raise TypeError("this loader cannot iterate over all templates")

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Loads a template.  This method looks up the template in the cache
        or loads one by calling :meth:`get_source`.  Subclasses should not
        override this method as loaders working on collections of other
        loaders (such as :class:`PrefixLoader` or :class:`ChoiceLoader`)
        will not call this method but `get_source` directly.
        """
        code = None
        if globals is None:
            globals = {}

        # first we try to get the source for this template together
        # with the filename and the uptodate function.
        source, filename, uptodate = self.get_source(environment, name)

        # try to load the code from the bytecode cache if there is a
        # bytecode cache configured.
        bcc = environment.bytecode_cache
        if bcc is not None:
            bucket = bcc.get_bucket(environment, name, filename, source)
            code = bucket.code

        # if we don't have code so far (not cached, no longer up to
        # date) etc. we compile the template
        if code is None:
            code = environment.compile(source, name, filename)

        # if the bytecode cache is available and the bucket doesn't
        # have a code so far, we give the bucket the new code and put
        # it back to the bytecode cache.
        if bcc is not None and bucket.code is None:
            bucket.code = code
            bcc.set_bucket(bucket)

        return environment.template_class.from_code(
            environment, code, globals, uptodate
        )


class FileSystemLoader(BaseLoader):
    """Load templates from a directory in the file system.

    The path can be relative or absolute. Relative paths are relative to
    the current working directory.

    .. code-block:: python

        loader = FileSystemLoader("templates")

    A list of paths can be given. The directories will be searched in
    order, stopping at the first matching template.

    .. code-block:: python

        loader = FileSystemLoader(["/override/templates", "/default/templates"])

    :param searchpath: A path, or list of paths, to the directory that
        contains the templates.
    :param encoding: Use this encoding to read the text from template
        files.
    :param followlinks: Follow symbolic links in the path.

    .. versionchanged:: 2.8
        Added the ``followlinks`` parameter.
    """

    def __init__(
        self,
        searchpath: t.Union[
            str, "os.PathLike[str]", t.Sequence[t.Union[str, "os.PathLike[str]"]]
        ],
        encoding: str = "utf-8",
        followlinks: bool = False,
    ) -> None:
        if not isinstance(searchpath, abc.Iterable) or isinstance(searchpath, str):
            searchpath = [searchpath]

        self.searchpath = [os.fspath(p) for p in searchpath]
        self.encoding = encoding
        self.followlinks = followlinks

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, str, t.Callable[[], bool]]:
        pieces = split_template_path(template)

        for searchpath in self.searchpath:
            # Use posixpath even on Windows to avoid "drive:" or UNC
            # segments breaking out of the search directory.
            filename = posixpath.join(searchpath, *pieces)

            if os.path.isfile(filename):
                break
        else:
            plural = "path" if len(self.searchpath) == 1 else "paths"
            paths_str = ", ".join(repr(p) for p in self.searchpath)
            raise TemplateNotFound(
                template,
                f"{template!r} not found in search {plural}: {paths_str}",
            )

        with open(filename, encoding=self.encoding) as f:
            contents = f.read()

        mtime = os.path.getmtime(filename)

        def uptodate() -> bool:
            try:
                return os.path.getmtime(filename) == mtime
            except OSError:
                return False

        # Use normpath to convert Windows altsep to sep.
        return contents, os.path.normpath(filename), uptodate

    def list_templates(self) -> t.List[str]:
        found = set()
        for searchpath in self.searchpath:
            walk_dir = os.walk(searchpath, followlinks=self.followlinks)
            for dirpath, _, filenames in walk_dir:
                for filename in filenames:
                    template = (
                        os.path.join(dirpath, filename)[len(searchpath) :]
                        .strip(os.path.sep)
                        .replace(os.path.sep, "/")
                    )
                    if template[:2] == "./":
                        template = template[2:]
                    if template not in found:
                        found.add(template)
        return sorted(found)


if sys.version_info >= (3, 13):

    def _get_zipimporter_files(z: t.Any) -> t.Dict[str, object]:
        try:
            get_files = z._get_files
        except AttributeError as e:
            raise TypeError(
                "This zip import does not have the required"
                " metadata to list templates."
            ) from e
        return get_files()
else:

    def _get_zipimporter_files(z: t.Any) -> t.Dict[str, object]:
        try:
            files = z._files
        except AttributeError as e:
            raise TypeError(
                "This zip import does not have the required"
                " metadata to list templates."
            ) from e
        return files  # type: ignore[no-any-return]


class PackageLoader(BaseLoader):
    """Load templates from a directory in a Python package.

    :param package_name: Import name of the package that contains the
        template directory.
    :param package_path: Directory within the imported package that
        contains the templates.
    :param encoding: Encoding of template files.

    The following example looks up templates in the ``pages`` directory
    within the ``project.ui`` package.

    .. code-block:: python

        loader = PackageLoader("project.ui", "pages")

    Only packages installed as directories (standard pip behavior) or
    zip/egg files (less common) are supported. The Python API for
    introspecting data in packages is too limited to support other
    installation methods the way this loader requires.

    There is limited support for :pep:`420` namespace packages. The
    template directory is assumed to only be in one namespace
    contributor. Zip files contributing to a namespace are not
    supported.

    .. versionchanged:: 3.0
        No longer uses ``setuptools`` as a dependency.

    .. versionchanged:: 3.0
        Limited PEP 420 namespace package support.
    """

    def __init__(
        self,
        package_name: str,
        package_path: "str" = "templates",
        encoding: str = "utf-8",
    ) -> None:
        package_path = os.path.normpath(package_path).rstrip(os.path.sep)

        # normpath preserves ".", which isn't valid in zip paths.
        if package_path == os.path.curdir:
            package_path = ""
        elif package_path[:2] == os.path.curdir + os.path.sep:
            package_path = package_path[2:]

        self.package_path = package_path
        self.package_name = package_name
        self.encoding = encoding

        # Make sure the package exists. This also makes namespace
        # packages work, otherwise get_loader returns None.
        import_module(package_name)
        spec = importlib.util.find_spec(package_name)
        assert spec is not None, "An import spec was not found for the package."
        loader = spec.loader
        assert loader is not None, "A loader was not found for the package."
        self._loader = loader
        self._archive = None

        if isinstance(loader, zipimport.zipimporter):
            self._archive = loader.archive
            pkgdir = next(iter(spec.submodule_search_locations))  # type: ignore
            template_root = os.path.join(pkgdir, package_path).rstrip(os.path.sep)
        else:
            roots: t.List[str] = []

            # One element for regular packages, multiple for namespace
            # packages, or None for single module file.
            if spec.submodule_search_locations:
                roots.extend(spec.submodule_search_locations)
            # A single module file, use the parent directory instead.
            elif spec.origin is not None:
                roots.append(os.path.dirname(spec.origin))

            if not roots:
                raise ValueError(
                    f"The {package_name!r} package was not installed in a"
                    " way that PackageLoader understands."
                )

            for root in roots:
                root = os.path.join(root, package_path)

                if os.path.isdir(root):
                    template_root = root
                    break
            else:
                raise ValueError(
                    f"PackageLoader could not find a {package_path!r} directory"
                    f" in the {package_name!r} package."
                )

        self._template_root = template_root

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, str, t.Optional[t.Callable[[], bool]]]:
        # Use posixpath even on Windows to avoid "drive:" or UNC
        # segments breaking out of the search directory. Use normpath to
        # convert Windows altsep to sep.
        p = os.path.normpath(
            posixpath.join(self._template_root, *split_template_path(template))
        )
        up_to_date: t.Optional[t.Callable[[], bool]]

        if self._archive is None:
            # Package is a directory.
            if not os.path.isfile(p):
                raise TemplateNotFound(template)

            with open(p, "rb") as f:
                source = f.read()

            mtime = os.path.getmtime(p)

            def up_to_date() -> bool:
                return os.path.isfile(p) and os.path.getmtime(p) == mtime

        else:
            # Package is a zip file.
            try:
                source = self._loader.get_data(p)  # type: ignore
            except OSError as e:
                raise TemplateNotFound(template) from e

            # Could use the zip's mtime for all template mtimes, but
            # would need to safely reload the module if it's out of
            # date, so just report it as always current.
            up_to_date = None

        return source.decode(self.encoding), p, up_to_date

    def list_templates(self) -> t.List[str]:
        results: t.List[str] = []

        if self._archive is None:
            # Package is a directory.
            offset = len(self._template_root)

            for dirpath, _, filenames in os.walk(self._template_root):
                dirpath = dirpath[offset:].lstrip(os.path.sep)
                results.extend(
                    os.path.join(dirpath, name).replace(os.path.sep, "/")
                    for name in filenames
                )
        else:
            files = _get_zipimporter_files(self._loader)

            # Package is a zip file.
            prefix = (
                self._template_root[len(self._archive) :].lstrip(os.path.sep)
                + os.path.sep
            )
            offset = len(prefix)

            for name in files:
                # Find names under the templates directory that aren't directories.
                if name.startswith(prefix) and name[-1] != os.path.sep:
                    results.append(name[offset:].replace(os.path.sep, "/"))

        results.sort()
        return results


class DictLoader(BaseLoader):
    """Loads a template from a Python dict mapping template names to
    template source.  This loader is useful for unittesting:

    >>> loader = DictLoader({'index.html': 'source here'})

    Because auto reloading is rarely useful this is disabled by default.
    """

    def __init__(self, mapping: t.Mapping[str, str]) -> None:
        self.mapping = mapping

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, None, t.Callable[[], bool]]:
        if template in self.mapping:
            source = self.mapping[template]
            return source, None, lambda: source == self.mapping.get(template)
        raise TemplateNotFound(template)

    def list_templates(self) -> t.List[str]:
        return sorted(self.mapping)


class FunctionLoader(BaseLoader):
    """A loader that is passed a function which does the loading.  The
    function receives the name of the template and has to return either
    a string with the template source, a tuple in the form ``(source,
    filename, uptodatefunc)`` or `None` if the template does not exist.

    >>> def load_template(name):
    ...     if name == 'index.html':
    ...         return '...'
    ...
    >>> loader = FunctionLoader(load_template)

    The `uptodatefunc` is a function that is called if autoreload is enabled
    and has to return `True` if the template is still up to date.  For more
    details have a look at :meth:`BaseLoader.get_source` which has the same
    return value.
    """

    def __init__(
        self,
        load_func: t.Callable[
            [str],
            t.Optional[
                t.Union[
                    str, t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]
                ]
            ],
        ],
    ) -> None:
        self.load_func = load_func

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        rv = self.load_func(template)

        if rv is None:
            raise TemplateNotFound(template)

        if isinstance(rv, str):
            return rv, None, None

        return rv


class PrefixLoader(BaseLoader):
    """A loader that is passed a dict of loaders where each loader is bound
    to a prefix.  The prefix is delimited from the template by a slash per
    default, which can be changed by setting the `delimiter` argument to
    something else::

        loader = PrefixLoader({
            'app1':     PackageLoader('mypackage.app1'),
            'app2':     PackageLoader('mypackage.app2')
        })

    By loading ``'app1/index.html'`` the file from the app1 package is loaded,
    by loading ``'app2/index.html'`` the file from the second.
    """

    def __init__(
        self, mapping: t.Mapping[str, BaseLoader], delimiter: str = "/"
    ) -> None:
        self.mapping = mapping
        self.delimiter = delimiter

    def get_loader(self, template: str) -> t.Tuple[BaseLoader, str]:
        try:
            prefix, name = template.split(self.delimiter, 1)
            loader = self.mapping[prefix]
        except (ValueError, KeyError) as e:
            raise TemplateNotFound(template) from e
        return loader, name

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        loader, name = self.get_loader(template)
        try:
            return loader.get_source(environment, name)
        except TemplateNotFound as e:
            # re-raise the exception with the correct filename here.
            # (the one that includes the prefix)
            raise TemplateNotFound(template) from e

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        loader, local_name = self.get_loader(name)
        try:
            return loader.load(environment, local_name, globals)
        except TemplateNotFound as e:
            # re-raise the exception with the correct filename here.
            # (the one that includes the prefix)
            raise TemplateNotFound(name) from e

    def list_templates(self) -> t.List[str]:
        result = []
        for prefix, loader in self.mapping.items():
            for template in loader.list_templates():
                result.append(prefix + self.delimiter + template)
        return result


class ChoiceLoader(BaseLoader):
    """This loader works like the `PrefixLoader` just that no prefix is
    specified.  If a template could not be found by one loader the next one
    is tried.

    >>> loader = ChoiceLoader([
    ...     FileSystemLoader('/path/to/user/templates'),
    ...     FileSystemLoader('/path/to/system/templates')
    ... ])

    This is useful if you want to allow users to override builtin templates
    from a different location.
    """

    def __init__(self, loaders: t.Sequence[BaseLoader]) -> None:
        self.loaders = loaders

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        for loader in self.loaders:
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(template)

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        for loader in self.loaders:
            try:
                return loader.load(environment, name, globals)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(name)

    def list_templates(self) -> t.List[str]:
        found = set()
        for loader in self.loaders:
            found.update(loader.list_templates())
        return sorted(found)


class _TemplateModule(ModuleType):
    """Like a normal module but with support for weak references"""


class ModuleLoader(BaseLoader):
    """This loader loads templates from precompiled templates.

    Example usage:

    >>> loader = ModuleLoader('/path/to/compiled/templates')

    Templates can be precompiled with :meth:`Environment.compile_templates`.
    """

    has_source_access = False

    def __init__(
        self,
        path: t.Union[
            str, "os.PathLike[str]", t.Sequence[t.Union[str, "os.PathLike[str]"]]
        ],
    ) -> None:
        package_name = f"_jinja2_module_templates_{id(self):x}"

        # create a fake module that looks for the templates in the
        # path given.
        mod = _TemplateModule(package_name)

        if not isinstance(path, abc.Iterable) or isinstance(path, str):
            path = [path]

        mod.__path__ = [os.fspath(p) for p in path]

        sys.modules[package_name] = weakref.proxy(
            mod, lambda x: sys.modules.pop(package_name, None)
        )

        # the only strong reference, the sys.modules entry is weak
        # so that the garbage collector can remove it once the
        # loader that created it goes out of business.
        self.module = mod
        self.package_name = package_name

    @staticmethod
    def get_template_key(name: str) -> str:
        return "tmpl_" + sha1(name.encode("utf-8")).hexdigest()

    @staticmethod
    def get_module_filename(name: str) -> str:
        return ModuleLoader.get_template_key(name) + ".py"

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        key = self.get_template_key(name)
        module = f"{self.package_name}.{key}"
        mod = getattr(self.module, module, None)

        if mod is None:
            try:
                mod = __import__(module, None, None, ["root"])
            except ImportError as e:
                raise TemplateNotFound(name) from e

            # remove the entry from sys.modules, we only want the attribute
            # on the module object we have stored on the loader.
            sys.modules.pop(module, None)

        if globals is None:
            globals = {}

        return environment.template_class.from_module_dict(
            environment, mod.__dict__, globals
        )