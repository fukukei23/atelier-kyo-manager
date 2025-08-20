
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_kane2.py ===
from sympy import cos, Matrix, sin, zeros, tan, pi, symbols
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import (cross, dot, dynamicsymbols,
                                     find_dynamicsymbols, KanesMethod, inertia,
                                     inertia_of_point_mass, Point,
                                     ReferenceFrame, RigidBody)


def test_aux_dep():
    # This test is about rolling disc dynamics, comparing the results found
    # with KanesMethod to those found when deriving the equations "manually"
    # with SymPy.
    # The terms Fr, Fr*, and Fr*_steady are all compared between the two
    # methods. Here, Fr*_steady refers to the generalized inertia forces for an
    # equilibrium configuration.
    # Note: comparing to the test of test_rolling_disc() in test_kane.py, this
    # test also tests auxiliary speeds and configuration and motion constraints
    #, seen in  the generalized dependent coordinates q[3], and depend speeds
    # u[3], u[4] and u[5].


    # First, manual derivation of Fr, Fr_star, Fr_star_steady.

    # Symbols for time and constant parameters.
    # Symbols for contact forces: Fx, Fy, Fz.
    t, r, m, g, I, J = symbols('t r m g I J')
    Fx, Fy, Fz = symbols('Fx Fy Fz')

    # Configuration variables and their time derivatives:
    # q[0] -- yaw
    # q[1] -- lean
    # q[2] -- spin
    # q[3] -- dot(-r*B.z, A.z) -- distance from ground plane to disc center in
    #         A.z direction
    # Generalized speeds and their time derivatives:
    # u[0] -- disc angular velocity component, disc fixed x direction
    # u[1] -- disc angular velocity component, disc fixed y direction
    # u[2] -- disc angular velocity component, disc fixed z direction
    # u[3] -- disc velocity component, A.x direction
    # u[4] -- disc velocity component, A.y direction
    # u[5] -- disc velocity component, A.z direction
    # Auxiliary generalized speeds:
    # ua[0] -- contact point auxiliary generalized speed, A.x direction
    # ua[1] -- contact point auxiliary generalized speed, A.y direction
    # ua[2] -- contact point auxiliary generalized speed, A.z direction
    q = dynamicsymbols('q:4')
    qd = [qi.diff(t) for qi in q]
    u = dynamicsymbols('u:6')
    ud = [ui.diff(t) for ui in u]
    ud_zero = dict(zip(ud, [0.]*len(ud)))
    ua = dynamicsymbols('ua:3')
    ua_zero = dict(zip(ua, [0.]*len(ua))) # noqa:F841

    # Reference frames:
    # Yaw intermediate frame: A.
    # Lean intermediate frame: B.
    # Disc fixed frame: C.
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q[0], N.z])
    B = A.orientnew('B', 'Axis', [q[1], A.x])
    C = B.orientnew('C', 'Axis', [q[2], B.y])

    # Angular velocity and angular acceleration of disc fixed frame
    # u[0], u[1] and u[2] are generalized independent speeds.
    C.set_ang_vel(N, u[0]*B.x + u[1]*B.y + u[2]*B.z)
    C.set_ang_acc(N, C.ang_vel_in(N).diff(t, B)
                   + cross(B.ang_vel_in(N), C.ang_vel_in(N)))

    # Velocity and acceleration of points:
    # Disc-ground contact point: P.
    # Center of disc: O, defined from point P with depend coordinate: q[3]
    # u[3], u[4] and u[5] are generalized dependent speeds.
    P = Point('P')
    P.set_vel(N, ua[0]*A.x + ua[1]*A.y + ua[2]*A.z)
    O = P.locatenew('O', q[3]*A.z + r*sin(q[1])*A.y)
    O.set_vel(N, u[3]*A.x + u[4]*A.y + u[5]*A.z)
    O.set_acc(N, O.vel(N).diff(t, A) + cross(A.ang_vel_in(N), O.vel(N)))

    # Kinematic differential equations:
    # Two equalities: one is w_c_n_qd = C.ang_vel_in(N) in three coordinates
    #                 directions of B, for qd0, qd1 and qd2.
    #                 the other is v_o_n_qd = O.vel(N) in A.z direction for qd3.
    # Then, solve for dq/dt's in terms of u's: qd_kd.
    w_c_n_qd = qd[0]*A.z + qd[1]*B.x + qd[2]*B.y
    v_o_n_qd = O.pos_from(P).diff(t, A) + cross(A.ang_vel_in(N), O.pos_from(P))
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] +
                      [dot(v_o_n_qd - O.vel(N), A.z)])
    qd_kd = solve(kindiffs, qd) # noqa:F841

    # Values of generalized speeds during a steady turn for later substitution
    # into the Fr_star_steady.
    steady_conditions = solve(kindiffs.subs({qd[1] : 0, qd[3] : 0}), u)
    steady_conditions.update({qd[1] : 0, qd[3] : 0})

    # Partial angular velocities and velocities.
    partial_w_C = [C.ang_vel_in(N).diff(ui, N) for ui in u + ua]
    partial_v_O = [O.vel(N).diff(ui, N) for ui in u + ua]
    partial_v_P = [P.vel(N).diff(ui, N) for ui in u + ua]

    # Configuration constraint: f_c, the projection of radius r in A.z direction
    #                                is q[3].
    # Velocity constraints: f_v, for u3, u4 and u5.
    # Acceleration constraints: f_a.
    f_c = Matrix([dot(-r*B.z, A.z) - q[3]])
    f_v = Matrix([dot(O.vel(N) - (P.vel(N) + cross(C.ang_vel_in(N),
        O.pos_from(P))), ai).expand() for ai in A])
    v_o_n = cross(C.ang_vel_in(N), O.pos_from(P))
    a_o_n = v_o_n.diff(t, A) + cross(A.ang_vel_in(N), v_o_n)
    f_a = Matrix([dot(O.acc(N) - a_o_n, ai) for ai in A]) # noqa:F841

    # Solve for constraint equations in the form of
    #                           u_dependent = A_rs * [u_i; u_aux].
    # First, obtain constraint coefficient matrix:  M_v * [u; ua] = 0;
    # Second, taking u[0], u[1], u[2] as independent,
    #         taking u[3], u[4], u[5] as dependent,
    #         rearranging the matrix of M_v to be A_rs for u_dependent.
    # Third, u_aux ==0 for u_dep, and resulting dictionary of u_dep_dict.
    M_v = zeros(3, 9)
    for i in range(3):
        for j, ui in enumerate(u + ua):
            M_v[i, j] = f_v[i].diff(ui)

    M_v_i = M_v[:, :3]
    M_v_d = M_v[:, 3:6]
    M_v_aux = M_v[:, 6:]
    M_v_i_aux = M_v_i.row_join(M_v_aux)
    A_rs = - M_v_d.inv() * M_v_i_aux

    u_dep = A_rs[:, :3] * Matrix(u[:3])
    u_dep_dict = dict(zip(u[3:], u_dep))

    # Active forces: F_O acting on point O; F_P acting on point P.
    # Generalized active forces (unconstrained): Fr_u = F_point * pv_point.
    F_O = m*g*A.z
    F_P = Fx * A.x + Fy * A.y + Fz * A.z
    Fr_u = Matrix([dot(F_O, pv_o) + dot(F_P, pv_p) for pv_o, pv_p in
            zip(partial_v_O, partial_v_P)])

    # Inertia force: R_star_O.
    # Inertia of disc: I_C_O, where J is a inertia component about principal axis.
    # Inertia torque: T_star_C.
    # Generalized inertia forces (unconstrained): Fr_star_u.
    R_star_O = -m*O.acc(N)
    I_C_O = inertia(B, I, J, I)
    T_star_C = -(dot(I_C_O, C.ang_acc_in(N)) \
                 + cross(C.ang_vel_in(N), dot(I_C_O, C.ang_vel_in(N))))
    Fr_star_u = Matrix([dot(R_star_O, pv) + dot(T_star_C, pav) for pv, pav in
                        zip(partial_v_O, partial_w_C)])

    # Form nonholonomic Fr: Fr_c, and nonholonomic Fr_star: Fr_star_c.
    # Also, nonholonomic Fr_star in steady turning condition: Fr_star_steady.
    Fr_c = Fr_u[:3, :].col_join(Fr_u[6:, :]) + A_rs.T * Fr_u[3:6, :]
    Fr_star_c = Fr_star_u[:3, :].col_join(Fr_star_u[6:, :])\
                + A_rs.T * Fr_star_u[3:6, :]
    Fr_star_steady = Fr_star_c.subs(ud_zero).subs(u_dep_dict)\
            .subs(steady_conditions).subs({q[3]: -r*cos(q[1])}).expand()


    # Second, using KaneMethod in mechanics for fr, frstar and frstar_steady.

    # Rigid Bodies: disc, with inertia I_C_O.
    iner_tuple = (I_C_O, O)
    disc = RigidBody('disc', O, C, m, iner_tuple)
    bodyList = [disc]

    # Generalized forces: Gravity: F_o; Auxiliary forces: F_p.
    F_o = (O, F_O)
    F_p = (P, F_P)
    forceList = [F_o,  F_p]

    # KanesMethod.
    kane = KanesMethod(
        N, q_ind= q[:3], u_ind= u[:3], kd_eqs=kindiffs,
        q_dependent=q[3:], configuration_constraints = f_c,
        u_dependent=u[3:], velocity_constraints= f_v,
        u_auxiliary=ua
        )

    # fr, frstar, frstar_steady and kdd(kinematic differential equations).
    (fr, frstar)= kane.kanes_equations(bodyList, forceList)
    frstar_steady = frstar.subs(ud_zero).subs(u_dep_dict).subs(steady_conditions)\
                    .subs({q[3]: -r*cos(q[1])}).expand()
    kdd = kane.kindiffdict()

    assert Matrix(Fr_c).expand() == fr.expand()
    assert Matrix(Fr_star_c.subs(kdd)).expand() == frstar.expand()
    # These Matrices have some Integer(0) and some Float(0). Running under
    # SymEngine gives different types of zero.
    assert (simplify(Matrix(Fr_star_steady).expand()).xreplace({0:0.0}) ==
            simplify(frstar_steady.expand()).xreplace({0:0.0}))

    syms_in_forcing = find_dynamicsymbols(kane.forcing)
    for qdi in qd:
        assert qdi not in syms_in_forcing


def test_non_central_inertia():
    # This tests that the calculation of Fr* does not depend the point
    # about which the inertia of a rigid body is defined. This test solves
    # exercises 8.12, 8.17 from Kane 1985.

    # Declare symbols
    q1, q2, q3 = dynamicsymbols('q1:4')
    q1d, q2d, q3d = dynamicsymbols('q1:4', level=1)
    u1, u2, u3, u4, u5 = dynamicsymbols('u1:6')
    u_prime, R, M, g, e, f, theta = symbols('u\' R, M, g, e, f, theta')
    a, b, mA, mB, IA, J, K, t = symbols('a b mA mB IA J K t')
    Q1, Q2, Q3 = symbols('Q1, Q2 Q3')
    IA22, IA23, IA33 = symbols('IA22 IA23 IA33')

    # Reference Frames
    F = ReferenceFrame('F')
    P = F.orientnew('P', 'axis', [-theta, F.y])
    A = P.orientnew('A', 'axis', [q1, P.x])
    A.set_ang_vel(F, u1*A.x + u3*A.z)
    # define frames for wheels
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])
    B.set_ang_vel(A, u4 * A.z)
    C.set_ang_vel(A, u5 * A.z)

    # define points D, S*, Q on frame A and their velocities
    pD = Point('D')
    pD.set_vel(A, 0)
    # u3 will not change v_D_F since wheels are still assumed to roll without slip.
    pD.set_vel(F, u2 * A.y)

    pS_star = pD.locatenew('S*', e*A.y)
    pQ = pD.locatenew('Q', f*A.y - R*A.x)
    for p in [pS_star, pQ]:
        p.v2pt_theory(pD, F, A)

    # masscenters of bodies A, B, C
    pA_star = pD.locatenew('A*', a*A.y)
    pB_star = pD.locatenew('B*', b*A.z)
    pC_star = pD.locatenew('C*', -b*A.z)
    for p in [pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)

    # points of B, C touching the plane P
    pB_hat = pB_star.locatenew('B^', -R*A.x)
    pC_hat = pC_star.locatenew('C^', -R*A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)

    # the velocities of B^, C^ are zero since B, C are assumed to roll without slip
    kde = [q1d - u1, q2d - u4, q3d - u5]
    vc = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]

    # inertias of bodies A, B, C
    # IA22, IA23, IA33 are not specified in the problem statement, but are
    # necessary to define an inertia object. Although the values of
    # IA22, IA23, IA33 are not known in terms of the variables given in the
    # problem statement, they do not appear in the general inertia terms.
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)

    # define the rigid bodies A, B, C
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))

    km = KanesMethod(F, q_ind=[q1, q2, q3], u_ind=[u1, u2], kd_eqs=kde,
                     u_dependent=[u4, u5], velocity_constraints=vc,
                     u_auxiliary=[u3])

    forces = [(pS_star, -M*g*F.x), (pQ, Q1*A.x + Q2*A.y + Q3*A.z)]
    bodies = [rbA, rbB, rbC]
    fr, fr_star = km.kanes_equations(bodies, forces)
    vc_map = solve(vc, [u4, u5])

    # KanesMethod returns the negative of Fr, Fr* as defined in Kane1985.
    fr_star_expected = Matrix([
            -(IA + 2*J*b**2/R**2 + 2*K +
              mA*a**2 + 2*mB*b**2) * u1.diff(t) - mA*a*u1*u2,
            -(mA + 2*mB +2*J/R**2) * u2.diff(t) + mA*a*u1**2,
            0])
    t = trigsimp(fr_star.subs(vc_map).subs({u3: 0})).doit().expand()
    assert ((fr_star_expected - t).expand() == zeros(3, 1))

    # define inertias of rigid bodies A, B, C about point D
    # I_S/O = I_S/S* + I_S*/O
    bodies2 = []
    for rb, I_star in zip([rbA, rbB, rbC], [inertia_A, inertia_B, inertia_C]):
        I = I_star + inertia_of_point_mass(rb.mass,
                                           rb.masscenter.pos_from(pD),
                                           rb.frame)
        bodies2.append(RigidBody('', rb.masscenter, rb.frame, rb.mass,
                                 (I, pD)))
    fr2, fr_star2 = km.kanes_equations(bodies2, forces)

    t = trigsimp(fr_star2.subs(vc_map).subs({u3: 0})).doit()
    assert (fr_star_expected - t).expand() == zeros(3, 1)

def test_sub_qdot():
    # This test solves exercises 8.12, 8.17 from Kane 1985 and defines
    # some velocities in terms of q, qdot.

    ## --- Declare symbols ---
    q1, q2, q3 = dynamicsymbols('q1:4')
    q1d, q2d, q3d = dynamicsymbols('q1:4', level=1)
    u1, u2, u3 = dynamicsymbols('u1:4')
    u_prime, R, M, g, e, f, theta = symbols('u\' R, M, g, e, f, theta')
    a, b, mA, mB, IA, J, K, t = symbols('a b mA mB IA J K t')
    IA22, IA23, IA33 = symbols('IA22 IA23 IA33')
    Q1, Q2, Q3 = symbols('Q1 Q2 Q3')

    # --- Reference Frames ---
    F = ReferenceFrame('F')
    P = F.orientnew('P', 'axis', [-theta, F.y])
    A = P.orientnew('A', 'axis', [q1, P.x])
    A.set_ang_vel(F, u1*A.x + u3*A.z)
    # define frames for wheels
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])

    ## --- define points D, S*, Q on frame A and their velocities ---
    pD = Point('D')
    pD.set_vel(A, 0)
    # u3 will not change v_D_F since wheels are still assumed to roll w/o slip
    pD.set_vel(F, u2 * A.y)

    pS_star = pD.locatenew('S*', e*A.y)
    pQ = pD.locatenew('Q', f*A.y - R*A.x)
    # masscenters of bodies A, B, C
    pA_star = pD.locatenew('A*', a*A.y)
    pB_star = pD.locatenew('B*', b*A.z)
    pC_star = pD.locatenew('C*', -b*A.z)
    for p in [pS_star, pQ, pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)

    # points of B, C touching the plane P
    pB_hat = pB_star.locatenew('B^', -R*A.x)
    pC_hat = pC_star.locatenew('C^', -R*A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)

    # --- relate qdot, u ---
    # the velocities of B^, C^ are zero since B, C are assumed to roll w/o slip
    kde = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]
    kde += [u1 - q1d]
    kde_map = solve(kde, [q1d, q2d, q3d])
    for k, v in list(kde_map.items()):
        kde_map[k.diff(t)] = v.diff(t)

    # inertias of bodies A, B, C
    # IA22, IA23, IA33 are not specified in the problem statement, but are
    # necessary to define an inertia object. Although the values of
    # IA22, IA23, IA33 are not known in terms of the variables given in the
    # problem statement, they do not appear in the general inertia terms.
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)

    # define the rigid bodies A, B, C
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))

    ## --- use kanes method ---
    km = KanesMethod(F, [q1, q2, q3], [u1, u2], kd_eqs=kde, u_auxiliary=[u3])

    forces = [(pS_star, -M*g*F.x), (pQ, Q1*A.x + Q2*A.y + Q3*A.z)]
    bodies = [rbA, rbB, rbC]

    # Q2 = -u_prime * u2 * Q1 / sqrt(u2**2 + f**2 * u1**2)
    # -u_prime * R * u2 / sqrt(u2**2 + f**2 * u1**2) = R / Q1 * Q2
    fr_expected = Matrix([
            f*Q3 + M*g*e*sin(theta)*cos(q1),
            Q2 + M*g*sin(theta)*sin(q1),
            e*M*g*cos(theta) - Q1*f - Q2*R])
             #Q1 * (f - u_prime * R * u2 / sqrt(u2**2 + f**2 * u1**2)))])
    fr_star_expected = Matrix([
            -(IA + 2*J*b**2/R**2 + 2*K +
              mA*a**2 + 2*mB*b**2) * u1.diff(t) - mA*a*u1*u2,
            -(mA + 2*mB +2*J/R**2) * u2.diff(t) + mA*a*u1**2,
            0])

    fr, fr_star = km.kanes_equations(bodies, forces)
    assert (fr.expand() == fr_expected.expand())
    assert ((fr_star_expected - trigsimp(fr_star)).expand() == zeros(3, 1))

def test_sub_qdot2():
    # This test solves exercises 8.3 from Kane 1985 and defines
    # all velocities in terms of q, qdot. We check that the generalized active
    # forces are correctly computed if u terms are only defined in the
    # kinematic differential equations.
    #
    # This functionality was added in PR 8948. Without qdot/u substitution, the
    # KanesMethod constructor will fail during the constraint initialization as
    # the B matrix will be poorly formed and inversion of the dependent part
    # will fail.

    g, m, Px, Py, Pz, R, t = symbols('g m Px Py Pz R t')
    q = dynamicsymbols('q:5')
    qd = dynamicsymbols('q:5', level=1)
    u = dynamicsymbols('u:5')

    ## Define inertial, intermediate, and rigid body reference frames
    A = ReferenceFrame('A')
    B_prime = A.orientnew('B_prime', 'Axis', [q[0], A.z])
    B = B_prime.orientnew('B', 'Axis', [pi/2 - q[1], B_prime.x])
    C = B.orientnew('C', 'Axis', [q[2], B.z])

    ## Define points of interest and their velocities
    pO = Point('O')
    pO.set_vel(A, 0)

    # R is the point in plane H that comes into contact with disk C.
    pR = pO.locatenew('R', q[3]*A.x + q[4]*A.y)
    pR.set_vel(A, pR.pos_from(pO).diff(t, A))
    pR.set_vel(B, 0)

    # C^ is the point in disk C that comes into contact with plane H.
    pC_hat = pR.locatenew('C^', 0)
    pC_hat.set_vel(C, 0)

    # C* is the point at the center of disk C.
    pCs = pC_hat.locatenew('C*', R*B.y)
    pCs.set_vel(C, 0)
    pCs.set_vel(B, 0)

    # calculate velocites of points C* and C^ in frame A
    pCs.v2pt_theory(pR, A, B) # points C* and R are fixed in frame B
    pC_hat.v2pt_theory(pCs, A, C) # points C* and C^ are fixed in frame C

    ## Define forces on each point of the system
    R_C_hat = Px*A.x + Py*A.y + Pz*A.z
    R_Cs = -m*g*A.z
    forces = [(pC_hat, R_C_hat), (pCs, R_Cs)]

    ## Define kinematic differential equations
    # let ui = omega_C_A & bi (i = 1, 2, 3)
    # u4 = qd4, u5 = qd5
    u_expr = [C.ang_vel_in(A) & uv for uv in B]
    u_expr += qd[3:]
    kde = [ui - e for ui, e in zip(u, u_expr)]
    km1 = KanesMethod(A, q, u, kde)
    fr1, _ = km1.kanes_equations([], forces)

    ## Calculate generalized active forces if we impose the condition that the
    # disk C is rolling without slipping
    u_indep = u[:3]
    u_dep = list(set(u) - set(u_indep))
    vc = [pC_hat.vel(A) & uv for uv in [A.x, A.y]]
    km2 = KanesMethod(A, q, u_indep, kde,
                      u_dependent=u_dep, velocity_constraints=vc)
    fr2, _ = km2.kanes_equations([], forces)

    fr1_expected = Matrix([
        -R*g*m*sin(q[1]),
        -R*(Px*cos(q[0]) + Py*sin(q[0]))*tan(q[1]),
        R*(Px*cos(q[0]) + Py*sin(q[0])),
        Px,
        Py])
    fr2_expected = Matrix([
        -R*g*m*sin(q[1]),
        0,
        0])
    assert (trigsimp(fr1.expand()) == trigsimp(fr1_expected.expand()))
    assert (trigsimp(fr2.expand()) == trigsimp(fr2_expected.expand()))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\tests\test_tensor_operators.py ===
from sympy import sin, cos
from sympy.testing.pytest import raises

from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
                                 tensor_indices,
                                 TensorHead, tensor_heads)
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array

from sympy.core.random import randint


L = TensorIndexType("L")
i, j, k, m, m1, m2, m3, m4 = tensor_indices("i j k m m1 m2 m3 m4", L)
i0 = tensor_indices("i0", L)
L_0, L_1 = tensor_indices("L_0 L_1", L)

A, B, C, D = tensor_heads("A B C D", [L])

H = TensorHead("H", [L, L])


def test_invalid_partial_derivative_valence():
    raises(ValueError, lambda: PartialDerivative(C(j), D(-j)))
    raises(ValueError, lambda: PartialDerivative(C(-j), D(j)))


def test_tensor_partial_deriv():
    # Test flatten:
    expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    assert expr == PartialDerivative(A(i), A(j), A(k))
    assert expr.expr == A(i)
    assert expr.variables == (A(j), A(k))
    assert expr.get_indices() == [i, -j, -k]
    assert expr.get_free_indices() == [i, -j, -k]

    expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(i))
    assert expr.expr == A(L_0)
    assert expr.variables == (A(j), A(L_0))

    expr1 = PartialDerivative(A(i), A(j))
    assert expr1.expr == A(i)
    assert expr1.variables == (A(j),)

    expr2 = A(i)*PartialDerivative(H(k, -i), A(j))
    assert expr2.get_indices() == [L_0, k, -L_0, -j]

    expr2b = A(i)*PartialDerivative(H(k, -i), A(-j))
    assert expr2b.get_indices() == [L_0, k, -L_0, j]

    expr3 = A(i)*PartialDerivative(B(k)*C(-i) + 3*H(k, -i), A(j))
    assert expr3.get_indices() == [L_0, k, -L_0, -j]

    expr4 = (A(i) + B(i))*PartialDerivative(C(j), D(j))
    assert expr4.get_indices() == [i, L_0, -L_0]

    expr4b = (A(i) + B(i))*PartialDerivative(C(-j), D(-j))
    assert expr4b.get_indices() == [i, -L_0, L_0]

    expr5 = (A(i) + B(i))*PartialDerivative(C(-i), D(j))
    assert expr5.get_indices() == [L_0, -L_0, -j]


def test_replace_arrays_partial_derivative():

    x, y, z, t = symbols("x y z t")

    expr = PartialDerivative(A(i), B(j))
    repl = expr.replace_with_arrays({A(i): [sin(x)*cos(y), x**3*y**2], B(i): [x, y]})
    assert repl == Array([[cos(x)*cos(y), -sin(x)*sin(y)], [3*x**2*y**2, 2*x**3*y]])
    repl = expr.replace_with_arrays({A(i): [sin(x)*cos(y), x**3*y**2], B(i): [x, y]}, [-j, i])
    assert repl == Array([[cos(x)*cos(y), 3*x**2*y**2], [-sin(x)*sin(y), 2*x**3*y]])

    # d(A^i)/d(A_j) = d(g^ik A_k)/d(A_j) = g^ik delta_jk
    expr = PartialDerivative(A(i), A(-j))
    assert expr.get_free_indices() == [i, j]
    assert expr.get_indices() == [i, j]
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [i, j]) == Array([[1, 0], [0, -1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [i, j]) == Array([[1, 0], [0, -1]])

    expr = PartialDerivative(A(i), A(j))
    assert expr.get_free_indices() == [i, -j]
    assert expr.get_indices() == [i, -j]
    assert expr.replace_with_arrays({A(i): [x, y]}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [i, -j]) == Array([[1, 0], [0, 1]])

    expr = PartialDerivative(A(-i), A(-j))
    assert expr.get_free_indices() == [-i, j]
    assert expr.get_indices() == [-i, j]
    assert expr.replace_with_arrays({A(-i): [x, y]}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [-i, j]) == Array([[1, 0], [0, 1]])

    expr = PartialDerivative(A(i), A(i))
    assert expr.get_free_indices() == []
    assert expr.get_indices() == [L_0, -L_0]
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, []) == 2
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, []) == 2

    expr = PartialDerivative(A(-i), A(-i))
    assert expr.get_free_indices() == []
    assert expr.get_indices() == [-L_0, L_0]
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, []) == 2
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, []) == 2

    expr = PartialDerivative(H(i, j) + H(j, i), A(i))
    assert expr.get_indices() == [L_0, j, -L_0]
    assert expr.get_free_indices() == [j]

    expr = PartialDerivative(H(i, j) + H(j, i), A(k))*B(-i)
    assert expr.get_indices() == [L_0, j, -k, -L_0]
    assert expr.get_free_indices() == [j, -k]

    expr = PartialDerivative(A(i)*(H(-i, j) + H(j, -i)), A(j))
    assert expr.get_indices() == [L_0, -L_0, L_1, -L_1]
    assert expr.get_free_indices() == []

    expr = A(j)*A(-j) + expr
    assert expr.get_indices() == [L_0, -L_0, L_1, -L_1]
    assert expr.get_free_indices() == []

    expr = A(i)*(B(j)*PartialDerivative(C(-j), D(i)) + C(j)*PartialDerivative(D(-j), B(i)))
    assert expr.get_indices() == [L_0, L_1, -L_1, -L_0]
    assert expr.get_free_indices() == []

    expr = A(i)*PartialDerivative(C(-j), D(i))
    assert expr.get_indices() == [L_0, -j, -L_0]
    assert expr.get_free_indices() == [-j]


def test_expand_partial_derivative_sum_rule():
    tau = symbols("tau")

    # check sum rule for D(tensor, symbol)
    expr1aa = PartialDerivative(A(i), tau)

    assert expr1aa._expand_partial_derivative() == PartialDerivative(A(i), tau)

    expr1ab = PartialDerivative(A(i) + B(i), tau)

    assert (expr1ab._expand_partial_derivative() ==
            PartialDerivative(A(i), tau) +
            PartialDerivative(B(i), tau))

    expr1ac = PartialDerivative(A(i) + B(i) + C(i), tau)

    assert (expr1ac._expand_partial_derivative() ==
            PartialDerivative(A(i), tau) +
            PartialDerivative(B(i), tau) +
            PartialDerivative(C(i), tau))

    # check sum rule for D(tensor, D(j))
    expr1ba = PartialDerivative(A(i), D(j))

    assert expr1ba._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(j))
    expr1bb = PartialDerivative(A(i) + B(i), D(j))

    assert (expr1bb._expand_partial_derivative() ==
            PartialDerivative(A(i), D(j)) +
            PartialDerivative(B(i), D(j)))

    expr1bc = PartialDerivative(A(i) + B(i) + C(i), D(j))
    assert expr1bc._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(j))\
        + PartialDerivative(B(i), D(j))\
        + PartialDerivative(C(i), D(j))

    # check sum rule for D(tensor, H(j, k))
    expr1ca = PartialDerivative(A(i), H(j, k))
    assert expr1ca._expand_partial_derivative() ==\
        PartialDerivative(A(i), H(j, k))
    expr1cb = PartialDerivative(A(i) + B(i), H(j, k))
    assert (expr1cb._expand_partial_derivative() ==
            PartialDerivative(A(i), H(j, k))
            + PartialDerivative(B(i), H(j, k)))
    expr1cc = PartialDerivative(A(i) + B(i) + C(i), H(j, k))
    assert (expr1cc._expand_partial_derivative() ==
            PartialDerivative(A(i), H(j, k))
            + PartialDerivative(B(i), H(j, k))
            + PartialDerivative(C(i), H(j, k)))

    # check sum rule for D(D(tensor, D(j)), H(k, m))
    expr1da = PartialDerivative(A(i), (D(j), H(k, m)))
    assert expr1da._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))
    expr1db = PartialDerivative(A(i) + B(i), (D(j), H(k, m)))
    assert expr1db._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))\
        + PartialDerivative(B(i), (D(j), H(k, m)))
    expr1dc = PartialDerivative(A(i) + B(i) + C(i), (D(j), H(k, m)))
    assert expr1dc._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))\
        + PartialDerivative(B(i), (D(j), H(k, m)))\
        + PartialDerivative(C(i), (D(j), H(k, m)))


def test_expand_partial_derivative_constant_factor_rule():
    nneg = randint(0, 1000)
    pos = randint(1, 1000)
    neg = -randint(1, 1000)

    c1 = Rational(nneg, pos)
    c2 = Rational(neg, pos)
    c3 = Rational(nneg, neg)

    expr2a = PartialDerivative(nneg*A(i), D(j))
    assert expr2a._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))

    expr2b = PartialDerivative(neg*A(i), D(j))
    assert expr2b._expand_partial_derivative() ==\
        neg*PartialDerivative(A(i), D(j))

    expr2ca = PartialDerivative(c1*A(i), D(j))
    assert expr2ca._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))

    expr2cb = PartialDerivative(c2*A(i), D(j))
    assert expr2cb._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))

    expr2cc = PartialDerivative(c3*A(i), D(j))
    assert expr2cc._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))


def test_expand_partial_derivative_full_linearity():
    nneg = randint(0, 1000)
    pos = randint(1, 1000)
    neg = -randint(1, 1000)

    c1 = Rational(nneg, pos)
    c2 = Rational(neg, pos)
    c3 = Rational(nneg, neg)

    # check full linearity
    p = PartialDerivative(42, D(j))
    assert p and not p._expand_partial_derivative()

    expr3a = PartialDerivative(nneg*A(i) + pos*B(i), D(j))
    assert expr3a._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))\
        + pos*PartialDerivative(B(i), D(j))

    expr3b = PartialDerivative(nneg*A(i) + neg*B(i), D(j))
    assert expr3b._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))\
        + neg*PartialDerivative(B(i), D(j))

    expr3c = PartialDerivative(neg*A(i) + pos*B(i), D(j))
    assert expr3c._expand_partial_derivative() ==\
        neg*PartialDerivative(A(i), D(j))\
        + pos*PartialDerivative(B(i), D(j))

    expr3d = PartialDerivative(c1*A(i) + c2*B(i), D(j))
    assert expr3d._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))\
        + c2*PartialDerivative(B(i), D(j))

    expr3e = PartialDerivative(c2*A(i) + c1*B(i), D(j))
    assert expr3e._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))\
        + c1*PartialDerivative(B(i), D(j))

    expr3f = PartialDerivative(c2*A(i) + c3*B(i), D(j))
    assert expr3f._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))\
        + c3*PartialDerivative(B(i), D(j))

    expr3g = PartialDerivative(c3*A(i) + c2*B(i), D(j))
    assert expr3g._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))\
        + c2*PartialDerivative(B(i), D(j))

    expr3h = PartialDerivative(c3*A(i) + c1*B(i), D(j))
    assert expr3h._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))\
        + c1*PartialDerivative(B(i), D(j))

    expr3i = PartialDerivative(c1*A(i) + c3*B(i), D(j))
    assert expr3i._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))\
        + c3*PartialDerivative(B(i), D(j))


def test_expand_partial_derivative_product_rule():
    # check product rule
    expr4a = PartialDerivative(A(i)*B(j), D(k))

    assert expr4a._expand_partial_derivative() == \
        PartialDerivative(A(i), D(k))*B(j)\
        + A(i)*PartialDerivative(B(j), D(k))

    expr4b = PartialDerivative(A(i)*B(j)*C(k), D(m))
    assert expr4b._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(m))*B(j)*C(k)\
        + A(i)*PartialDerivative(B(j), D(m))*C(k)\
        + A(i)*B(j)*PartialDerivative(C(k), D(m))

    expr4c = PartialDerivative(A(i)*B(j), C(k), D(m))
    assert expr4c._expand_partial_derivative() ==\
        PartialDerivative(A(i), C(k), D(m))*B(j) \
        + PartialDerivative(A(i), C(k))*PartialDerivative(B(j), D(m))\
        + PartialDerivative(A(i), D(m))*PartialDerivative(B(j), C(k))\
        + A(i)*PartialDerivative(B(j), C(k), D(m))


def test_eval_partial_derivative_expr_by_symbol():

    tau, alpha = symbols("tau alpha")

    expr1 = PartialDerivative(tau**alpha, tau)
    assert expr1._perform_derivative() == alpha * 1 / tau * tau ** alpha

    expr2 = PartialDerivative(2*tau + 3*tau**4, tau)
    assert expr2._perform_derivative() == 2 + 12 * tau ** 3

    expr3 = PartialDerivative(2*tau + 3*tau**4, alpha)
    assert expr3._perform_derivative() == 0


def test_eval_partial_derivative_single_tensors_by_scalar():

    tau, mu = symbols("tau mu")

    expr = PartialDerivative(tau**mu, tau)
    assert expr._perform_derivative() == mu*tau**mu/tau

    expr1a = PartialDerivative(A(i), tau)
    assert expr1a._perform_derivative() == 0

    expr1b = PartialDerivative(A(-i), tau)
    assert expr1b._perform_derivative() == 0

    expr2a = PartialDerivative(H(i, j), tau)
    assert expr2a._perform_derivative() == 0

    expr2b = PartialDerivative(H(i, -j), tau)
    assert expr2b._perform_derivative() == 0

    expr2c = PartialDerivative(H(-i, j), tau)
    assert expr2c._perform_derivative() == 0

    expr2d = PartialDerivative(H(-i, -j), tau)
    assert expr2d._perform_derivative() == 0


def test_eval_partial_derivative_single_1st_rank_tensors_by_tensor():

    expr1 = PartialDerivative(A(i), A(j))
    assert expr1._perform_derivative() - L.delta(i, -j) == 0

    expr2 = PartialDerivative(A(i), A(-j))
    assert expr2._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, j) == 0

    expr3 = PartialDerivative(A(-i), A(-j))
    assert expr3._perform_derivative() - L.delta(-i, j) == 0

    expr4 = PartialDerivative(A(-i), A(j))
    assert expr4._perform_derivative() - L.metric(-i, -L_0) * L.delta(L_0, -j) == 0

    expr5 = PartialDerivative(A(i), B(j))
    expr6 = PartialDerivative(A(i), C(j))
    expr7 = PartialDerivative(A(i), D(j))
    expr8 = PartialDerivative(A(i), H(j, k))
    assert expr5._perform_derivative() == 0
    assert expr6._perform_derivative() == 0
    assert expr7._perform_derivative() == 0
    assert expr8._perform_derivative() == 0

    expr9 = PartialDerivative(A(i), A(i))
    assert expr9._perform_derivative() - L.delta(L_0, -L_0) == 0

    expr10 = PartialDerivative(A(-i), A(-i))
    assert expr10._perform_derivative() - L.delta(-L_0, L_0) == 0


def test_eval_partial_derivative_single_2nd_rank_tensors_by_tensor():

    expr1 = PartialDerivative(H(i, j), H(m, m1))
    assert expr1._perform_derivative() - L.delta(i, -m) * L.delta(j, -m1) == 0

    expr2 = PartialDerivative(H(i, j), H(-m, m1))
    assert expr2._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, m) * L.delta(j, -m1) == 0

    expr3 = PartialDerivative(H(i, j), H(m, -m1))
    assert expr3._perform_derivative() - L.delta(i, -m) * L.metric(j, L_0) * L.delta(-L_0, m1) == 0

    expr4 = PartialDerivative(H(i, j), H(-m, -m1))
    assert expr4._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, m) * L.metric(j, L_1) * L.delta(-L_1, m1) == 0

def test_eval_partial_derivative_divergence_type():
    expr1a = PartialDerivative(A(i), A(i))
    expr1b = PartialDerivative(A(i), A(k))
    expr1c = PartialDerivative(L.delta(-i, k) * A(i), A(k))

    assert (expr1a._perform_derivative()
            - (L.delta(-i, k) * expr1b._perform_derivative())).contract_delta(L.delta) == 0

    assert (expr1a._perform_derivative()
            - expr1c._perform_derivative()).contract_delta(L.delta) == 0

    expr2a = PartialDerivative(H(i, j), H(i, j))
    expr2b = PartialDerivative(H(i, j), H(k, m))
    expr2c = PartialDerivative(L.delta(-i, k) * L.delta(-j, m) * H(i, j), H(k, m))

    assert (expr2a._perform_derivative()
            - (L.delta(-i, k) * L.delta(-j, m) * expr2b._perform_derivative())).contract_delta(L.delta) == 0

    assert (expr2a._perform_derivative()
            - expr2c._perform_derivative()).contract_delta(L.delta) == 0


def test_eval_partial_derivative_expr1():

    tau, alpha = symbols("tau alpha")

    # this is only some special expression
    # tested: vector derivative
    # tested: scalar derivative
    # tested: tensor derivative
    base_expr1 = A(i)*H(-i, j) + A(i)*A(-i)*A(j) + tau**alpha*A(j)

    tensor_derivative = PartialDerivative(base_expr1, H(k, m))._perform_derivative()
    vector_derivative = PartialDerivative(base_expr1, A(k))._perform_derivative()
    scalar_derivative = PartialDerivative(base_expr1, tau)._perform_derivative()

    assert (tensor_derivative - A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k)*L.delta(j, -m)) == 0

    assert (vector_derivative - (tau**alpha*L.delta(j, -k) +
        L.delta(L_0, -k)*A(-L_0)*A(j) +
        A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k)*A(j) +
        A(L_0)*A(-L_0)*L.delta(j, -k) +
        L.delta(L_0, -k)*H(-L_0, j))).expand().doit() == 0

    assert (vector_derivative.contract_metric(L.metric).contract_delta(L.delta) -
        (tau**alpha*L.delta(j, -k) + A(L_0)*A(-L_0)*L.delta(j, -k) + H(-k, j) + 2*A(j)*A(-k))).expand().doit() == 0

    assert scalar_derivative - alpha*1/tau*tau**alpha*A(j) == 0


def test_eval_partial_derivative_mixed_scalar_tensor_expr2():

    tau, alpha = symbols("tau alpha")

    base_expr2 = A(i)*A(-i) + tau**2

    vector_expression = PartialDerivative(base_expr2, A(k))._perform_derivative()
    assert  (vector_expression -
        (L.delta(L_0, -k)*A(-L_0) + A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k))).expand().doit() == 0

    scalar_expression = PartialDerivative(base_expr2, tau)._perform_derivative()
    assert scalar_expression == 2*tau

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_coordsysrect.py ===
from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.scalar import BaseScalar
from sympy.core.function import expand
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.vector.functions import express
from sympy.vector.point import Point
from sympy.vector.vector import Vector
from sympy.vector.orienters import (AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)


x, y, z = symbols('x y z')
a, b, c, q = symbols('a b c q')
q1, q2, q3, q4 = symbols('q1 q2 q3 q4')


def test_func_args():
    A = CoordSys3D('A')
    assert A.x.func(*A.x.args) == A.x
    expr = 3*A.x + 4*A.y
    assert expr.func(*expr.args) == expr
    assert A.i.func(*A.i.args) == A.i
    v = A.x*A.i + A.y*A.j + A.z*A.k
    assert v.func(*v.args) == v
    assert A.origin.func(*A.origin.args) == A.origin


def test_coordsys3d_equivalence():
    A = CoordSys3D('A')
    A1 = CoordSys3D('A')
    assert A1 == A
    B = CoordSys3D('B')
    assert A != B


def test_orienters():
    A = CoordSys3D('A')
    axis_orienter = AxisOrienter(a, A.k)
    body_orienter = BodyOrienter(a, b, c, '123')
    space_orienter = SpaceOrienter(a, b, c, '123')
    q_orienter = QuaternionOrienter(q1, q2, q3, q4)
    assert axis_orienter.rotation_matrix(A) == Matrix([
        [ cos(a), sin(a), 0],
        [-sin(a), cos(a), 0],
        [      0,      0, 1]])
    assert body_orienter.rotation_matrix() == Matrix([
        [ cos(b)*cos(c),  sin(a)*sin(b)*cos(c) + sin(c)*cos(a),
          sin(a)*sin(c) - sin(b)*cos(a)*cos(c)],
        [-sin(c)*cos(b), -sin(a)*sin(b)*sin(c) + cos(a)*cos(c),
         sin(a)*cos(c) + sin(b)*sin(c)*cos(a)],
        [        sin(b),                        -sin(a)*cos(b),
                 cos(a)*cos(b)]])
    assert space_orienter.rotation_matrix() == Matrix([
        [cos(b)*cos(c), sin(c)*cos(b),       -sin(b)],
        [sin(a)*sin(b)*cos(c) - sin(c)*cos(a),
         sin(a)*sin(b)*sin(c) + cos(a)*cos(c), sin(a)*cos(b)],
        [sin(a)*sin(c) + sin(b)*cos(a)*cos(c), -sin(a)*cos(c) +
         sin(b)*sin(c)*cos(a), cos(a)*cos(b)]])
    assert q_orienter.rotation_matrix() == Matrix([
        [q1**2 + q2**2 - q3**2 - q4**2, 2*q1*q4 + 2*q2*q3,
         -2*q1*q3 + 2*q2*q4],
        [-2*q1*q4 + 2*q2*q3, q1**2 - q2**2 + q3**2 - q4**2,
         2*q1*q2 + 2*q3*q4],
        [2*q1*q3 + 2*q2*q4,
         -2*q1*q2 + 2*q3*q4, q1**2 - q2**2 - q3**2 + q4**2]])


def test_coordinate_vars():
    """
    Tests the coordinate variables functionality with respect to
    reorientation of coordinate systems.
    """
    A = CoordSys3D('A')
    # Note that the name given on the lhs is different from A.x._name
    assert BaseScalar(0, A, 'A_x', r'\mathbf{{x}_{A}}') == A.x
    assert BaseScalar(1, A, 'A_y', r'\mathbf{{y}_{A}}') == A.y
    assert BaseScalar(2, A, 'A_z', r'\mathbf{{z}_{A}}') == A.z
    assert BaseScalar(0, A, 'A_x', r'\mathbf{{x}_{A}}').__hash__() == A.x.__hash__()
    assert isinstance(A.x, BaseScalar) and \
           isinstance(A.y, BaseScalar) and \
           isinstance(A.z, BaseScalar)
    assert A.x*A.y == A.y*A.x
    assert A.scalar_map(A) == {A.x: A.x, A.y: A.y, A.z: A.z}
    assert A.x.system == A
    assert A.x.diff(A.x) == 1
    B = A.orient_new_axis('B', q, A.k)
    assert B.scalar_map(A) == {B.z: A.z, B.y: -A.x*sin(q) + A.y*cos(q),
                                 B.x: A.x*cos(q) + A.y*sin(q)}
    assert A.scalar_map(B) == {A.x: B.x*cos(q) - B.y*sin(q),
                                 A.y: B.x*sin(q) + B.y*cos(q), A.z: B.z}
    assert express(B.x, A, variables=True) == A.x*cos(q) + A.y*sin(q)
    assert express(B.y, A, variables=True) == -A.x*sin(q) + A.y*cos(q)
    assert express(B.z, A, variables=True) == A.z
    assert expand(express(B.x*B.y*B.z, A, variables=True)) == \
           expand(A.z*(-A.x*sin(q) + A.y*cos(q))*(A.x*cos(q) + A.y*sin(q)))
    assert express(B.x*B.i + B.y*B.j + B.z*B.k, A) == \
           (B.x*cos(q) - B.y*sin(q))*A.i + (B.x*sin(q) + \
           B.y*cos(q))*A.j + B.z*A.k
    assert simplify(express(B.x*B.i + B.y*B.j + B.z*B.k, A, \
                            variables=True)) == \
           A.x*A.i + A.y*A.j + A.z*A.k
    assert express(A.x*A.i + A.y*A.j + A.z*A.k, B) == \
           (A.x*cos(q) + A.y*sin(q))*B.i + \
           (-A.x*sin(q) + A.y*cos(q))*B.j + A.z*B.k
    assert simplify(express(A.x*A.i + A.y*A.j + A.z*A.k, B, \
                            variables=True)) == \
           B.x*B.i + B.y*B.j + B.z*B.k
    N = B.orient_new_axis('N', -q, B.k)
    assert N.scalar_map(A) == \
           {N.x: A.x, N.z: A.z, N.y: A.y}
    C = A.orient_new_axis('C', q, A.i + A.j + A.k)
    mapping = A.scalar_map(C)
    assert mapping[A.x].equals(C.x*(2*cos(q) + 1)/3 +
                            C.y*(-2*sin(q + pi/6) + 1)/3 +
                            C.z*(-2*cos(q + pi/3) + 1)/3)
    assert mapping[A.y].equals(C.x*(-2*cos(q + pi/3) + 1)/3 +
                            C.y*(2*cos(q) + 1)/3 +
                            C.z*(-2*sin(q + pi/6) + 1)/3)
    assert mapping[A.z].equals(C.x*(-2*sin(q + pi/6) + 1)/3 +
                            C.y*(-2*cos(q + pi/3) + 1)/3 +
                            C.z*(2*cos(q) + 1)/3)
    D = A.locate_new('D', a*A.i + b*A.j + c*A.k)
    assert D.scalar_map(A) == {D.z: A.z - c, D.x: A.x - a, D.y: A.y - b}
    E = A.orient_new_axis('E', a, A.k, a*A.i + b*A.j + c*A.k)
    assert A.scalar_map(E) == {A.z: E.z + c,
                               A.x: E.x*cos(a) - E.y*sin(a) + a,
                               A.y: E.x*sin(a) + E.y*cos(a) + b}
    assert E.scalar_map(A) == {E.x: (A.x - a)*cos(a) + (A.y - b)*sin(a),
                               E.y: (-A.x + a)*sin(a) + (A.y - b)*cos(a),
                               E.z: A.z - c}
    F = A.locate_new('F', Vector.zero)
    assert A.scalar_map(F) == {A.z: F.z, A.x: F.x, A.y: F.y}


def test_rotation_matrix():
    N = CoordSys3D('N')
    A = N.orient_new_axis('A', q1, N.k)
    B = A.orient_new_axis('B', q2, A.i)
    C = B.orient_new_axis('C', q3, B.j)
    D = N.orient_new_axis('D', q4, N.j)
    E = N.orient_new_space('E', q1, q2, q3, '123')
    F = N.orient_new_quaternion('F', q1, q2, q3, q4)
    G = N.orient_new_body('G', q1, q2, q3, '123')
    assert N.rotation_matrix(C) == Matrix([
        [- sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), - sin(q1) *
        cos(q2), sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)], \
        [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), \
         cos(q1) * cos(q2), sin(q1) * sin(q3) - sin(q2) * cos(q1) * \
         cos(q3)], [- sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    test_mat = D.rotation_matrix(C) - Matrix(
        [[cos(q1) * cos(q3) * cos(q4) - sin(q3) * (- sin(q4) * cos(q2) +
        sin(q1) * sin(q2) * cos(q4)), - sin(q2) * sin(q4) - sin(q1) *
            cos(q2) * cos(q4), sin(q3) * cos(q1) * cos(q4) + cos(q3) * \
          (- sin(q4) * cos(q2) + sin(q1) * sin(q2) * cos(q4))], \
         [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * \
          cos(q2), sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)], \
         [sin(q4) * cos(q1) * cos(q3) - sin(q3) * (cos(q2) * cos(q4) + \
                                                   sin(q1) * sin(q2) * \
                                                   sin(q4)), sin(q2) *
                cos(q4) - sin(q1) * sin(q4) * cos(q2), sin(q3) * \
          sin(q4) * cos(q1) + cos(q3) * (cos(q2) * cos(q4) + \
                                         sin(q1) * sin(q2) * sin(q4))]])
    assert test_mat.expand() == zeros(3, 3)
    assert E.rotation_matrix(N) == Matrix(
        [[cos(q2)*cos(q3), sin(q3)*cos(q2), -sin(q2)],
        [sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1), \
         sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q2)], \
         [sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3), - \
          sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1), cos(q1)*cos(q2)]])
    assert F.rotation_matrix(N) == Matrix([[
        q1**2 + q2**2 - q3**2 - q4**2,
        2*q1*q4 + 2*q2*q3, -2*q1*q3 + 2*q2*q4],[ -2*q1*q4 + 2*q2*q3,
            q1**2 - q2**2 + q3**2 - q4**2, 2*q1*q2 + 2*q3*q4],
                                           [2*q1*q3 + 2*q2*q4,
                                            -2*q1*q2 + 2*q3*q4,
                                q1**2 - q2**2 - q3**2 + q4**2]])
    assert G.rotation_matrix(N) == Matrix([[
        cos(q2)*cos(q3),  sin(q1)*sin(q2)*cos(q3) + sin(q3)*cos(q1),
        sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3)], [
            -sin(q3)*cos(q2), -sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3),
            sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],[
                sin(q2), -sin(q1)*cos(q2), cos(q1)*cos(q2)]])


def test_vector_with_orientation():
    """
    Tests the effects of orientation of coordinate systems on
    basic vector operations.
    """
    N = CoordSys3D('N')
    A = N.orient_new_axis('A', q1, N.k)
    B = A.orient_new_axis('B', q2, A.i)
    C = B.orient_new_axis('C', q3, B.j)

    # Test to_matrix
    v1 = a*N.i + b*N.j + c*N.k
    assert v1.to_matrix(A) == Matrix([[ a*cos(q1) + b*sin(q1)],
                                      [-a*sin(q1) + b*cos(q1)],
                                      [                     c]])

    # Test dot
    assert N.i.dot(A.i) == cos(q1)
    assert N.i.dot(A.j) == -sin(q1)
    assert N.i.dot(A.k) == 0
    assert N.j.dot(A.i) == sin(q1)
    assert N.j.dot(A.j) == cos(q1)
    assert N.j.dot(A.k) == 0
    assert N.k.dot(A.i) == 0
    assert N.k.dot(A.j) == 0
    assert N.k.dot(A.k) == 1

    assert N.i.dot(A.i + A.j) == -sin(q1) + cos(q1) == \
           (A.i + A.j).dot(N.i)

    assert A.i.dot(C.i) == cos(q3)
    assert A.i.dot(C.j) == 0
    assert A.i.dot(C.k) == sin(q3)
    assert A.j.dot(C.i) == sin(q2)*sin(q3)
    assert A.j.dot(C.j) == cos(q2)
    assert A.j.dot(C.k) == -sin(q2)*cos(q3)
    assert A.k.dot(C.i) == -cos(q2)*sin(q3)
    assert A.k.dot(C.j) == sin(q2)
    assert A.k.dot(C.k) == cos(q2)*cos(q3)

    # Test cross
    assert N.i.cross(A.i) == sin(q1)*A.k
    assert N.i.cross(A.j) == cos(q1)*A.k
    assert N.i.cross(A.k) == -sin(q1)*A.i - cos(q1)*A.j
    assert N.j.cross(A.i) == -cos(q1)*A.k
    assert N.j.cross(A.j) == sin(q1)*A.k
    assert N.j.cross(A.k) == cos(q1)*A.i - sin(q1)*A.j
    assert N.k.cross(A.i) == A.j
    assert N.k.cross(A.j) == -A.i
    assert N.k.cross(A.k) == Vector.zero

    assert N.i.cross(A.i) == sin(q1)*A.k
    assert N.i.cross(A.j) == cos(q1)*A.k
    assert N.i.cross(A.i + A.j) == sin(q1)*A.k + cos(q1)*A.k
    assert (A.i + A.j).cross(N.i) == (-sin(q1) - cos(q1))*N.k

    assert A.i.cross(C.i) == sin(q3)*C.j
    assert A.i.cross(C.j) == -sin(q3)*C.i + cos(q3)*C.k
    assert A.i.cross(C.k) == -cos(q3)*C.j
    assert C.i.cross(A.i) == (-sin(q3)*cos(q2))*A.j + \
           (-sin(q2)*sin(q3))*A.k
    assert C.j.cross(A.i) == (sin(q2))*A.j + (-cos(q2))*A.k
    assert express(C.k.cross(A.i), C).trigsimp() == cos(q3)*C.j


def test_orient_new_methods():
    N = CoordSys3D('N')
    orienter1 = AxisOrienter(q4, N.j)
    orienter2 = SpaceOrienter(q1, q2, q3, '123')
    orienter3 = QuaternionOrienter(q1, q2, q3, q4)
    orienter4 = BodyOrienter(q1, q2, q3, '123')
    D = N.orient_new('D', (orienter1, ))
    E = N.orient_new('E', (orienter2, ))
    F = N.orient_new('F', (orienter3, ))
    G = N.orient_new('G', (orienter4, ))
    assert D == N.orient_new_axis('D', q4, N.j)
    assert E == N.orient_new_space('E', q1, q2, q3, '123')
    assert F == N.orient_new_quaternion('F', q1, q2, q3, q4)
    assert G == N.orient_new_body('G', q1, q2, q3, '123')


def test_locatenew_point():
    """
    Tests Point class, and locate_new method in CoordSys3D.
    """
    A = CoordSys3D('A')
    assert isinstance(A.origin, Point)
    v = a*A.i + b*A.j + c*A.k
    C = A.locate_new('C', v)
    assert C.origin.position_wrt(A) == \
           C.position_wrt(A) == \
           C.origin.position_wrt(A.origin) == v
    assert A.origin.position_wrt(C) == \
           A.position_wrt(C) == \
           A.origin.position_wrt(C.origin) == -v
    assert A.origin.express_coordinates(C) == (-a, -b, -c)
    p = A.origin.locate_new('p', -v)
    assert p.express_coordinates(A) == (-a, -b, -c)
    assert p.position_wrt(C.origin) == p.position_wrt(C) == \
           -2 * v
    p1 = p.locate_new('p1', 2*v)
    assert p1.position_wrt(C.origin) == Vector.zero
    assert p1.express_coordinates(C) == (0, 0, 0)
    p2 = p.locate_new('p2', A.i)
    assert p1.position_wrt(p2) == 2*v - A.i
    assert p2.express_coordinates(C) == (-2*a + 1, -2*b, -2*c)


def test_create_new():
    a = CoordSys3D('a')
    c = a.create_new('c', transformation='spherical')
    assert c._parent == a
    assert c.transformation_to_parent() == \
           (c.r*sin(c.theta)*cos(c.phi), c.r*sin(c.theta)*sin(c.phi), c.r*cos(c.theta))
    assert c.transformation_from_parent() == \
           (sqrt(a.x**2 + a.y**2 + a.z**2), acos(a.z/sqrt(a.x**2 + a.y**2 + a.z**2)), atan2(a.y, a.x))


def test_evalf():
    A = CoordSys3D('A')
    v = 3*A.i + 4*A.j + a*A.k
    assert v.n() == v.evalf()
    assert v.evalf(subs={a:1}) == v.subs(a, 1).evalf()


def test_lame_coefficients():
    a = CoordSys3D('a', 'spherical')
    assert a.lame_coefficients() == (1, a.r, sin(a.theta)*a.r)
    a = CoordSys3D('a')
    assert a.lame_coefficients() == (1, 1, 1)
    a = CoordSys3D('a', 'cartesian')
    assert a.lame_coefficients() == (1, 1, 1)
    a = CoordSys3D('a', 'cylindrical')
    assert a.lame_coefficients() == (1, a.r, 1)


def test_transformation_equations():

    x, y, z = symbols('x y z')
    # Str
    a = CoordSys3D('a', transformation='spherical',
                   variable_names=["r", "theta", "phi"])
    r, theta, phi = a.base_scalars()

    assert r == a.r
    assert theta == a.theta
    assert phi == a.phi

    raises(AttributeError, lambda: a.x)
    raises(AttributeError, lambda: a.y)
    raises(AttributeError, lambda: a.z)

    assert a.transformation_to_parent() == (
        r*sin(theta)*cos(phi),
        r*sin(theta)*sin(phi),
        r*cos(theta)
    )
    assert a.lame_coefficients() == (1, r, r*sin(theta))
    assert a.transformation_from_parent_function()(x, y, z) == (
        sqrt(x ** 2 + y ** 2 + z ** 2),
        acos((z) / sqrt(x**2 + y**2 + z**2)),
        atan2(y, x)
    )
    a = CoordSys3D('a', transformation='cylindrical',
                   variable_names=["r", "theta", "z"])
    r, theta, z = a.base_scalars()
    assert a.transformation_to_parent() == (
        r*cos(theta),
        r*sin(theta),
        z
    )
    assert a.lame_coefficients() == (1, a.r, 1)
    assert a.transformation_from_parent_function()(x, y, z) == (sqrt(x**2 + y**2),
                            atan2(y, x), z)

    a = CoordSys3D('a', 'cartesian')
    assert a.transformation_to_parent() == (a.x, a.y, a.z)
    assert a.lame_coefficients() == (1, 1, 1)
    assert a.transformation_from_parent_function()(x, y, z) == (x, y, z)

    # Variables and expressions

    # Cartesian with equation tuple:
    x, y, z = symbols('x y z')
    a = CoordSys3D('a', ((x, y, z), (x, y, z)))
    a._calculate_inv_trans_equations()
    assert a.transformation_to_parent() == (a.x1, a.x2, a.x3)
    assert a.lame_coefficients() == (1, 1, 1)
    assert a.transformation_from_parent_function()(x, y, z) == (x, y, z)
    r, theta, z = symbols("r theta z")

    # Cylindrical with equation tuple:
    a = CoordSys3D('a', [(r, theta, z), (r*cos(theta), r*sin(theta), z)],
                   variable_names=["r", "theta", "z"])
    r, theta, z = a.base_scalars()
    assert a.transformation_to_parent() == (
        r*cos(theta), r*sin(theta), z
    )
    assert a.lame_coefficients() == (
        sqrt(sin(theta)**2 + cos(theta)**2),
        sqrt(r**2*sin(theta)**2 + r**2*cos(theta)**2),
        1
    )  # ==> this should simplify to (1, r, 1), tests are too slow with `simplify`.

    # Definitions with `lambda`:

    # Cartesian with `lambda`
    a = CoordSys3D('a', lambda x, y, z: (x, y, z))
    assert a.transformation_to_parent() == (a.x1, a.x2, a.x3)
    assert a.lame_coefficients() == (1, 1, 1)
    a._calculate_inv_trans_equations()
    assert a.transformation_from_parent_function()(x, y, z) == (x, y, z)

    # Spherical with `lambda`
    a = CoordSys3D('a', lambda r, theta, phi: (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)),
                   variable_names=["r", "theta", "phi"])
    r, theta, phi = a.base_scalars()
    assert a.transformation_to_parent() == (
        r*sin(theta)*cos(phi), r*sin(phi)*sin(theta), r*cos(theta)
    )
    assert a.lame_coefficients() == (
        sqrt(sin(phi)**2*sin(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(theta)**2),
        sqrt(r**2*sin(phi)**2*cos(theta)**2 + r**2*sin(theta)**2 + r**2*cos(phi)**2*cos(theta)**2),
        sqrt(r**2*sin(phi)**2*sin(theta)**2 + r**2*sin(theta)**2*cos(phi)**2)
    )  # ==> this should simplify to (1, r, sin(theta)*r), `simplify` is too slow.

    # Cylindrical with `lambda`
    a = CoordSys3D('a', lambda r, theta, z:
        (r*cos(theta), r*sin(theta), z),
        variable_names=["r", "theta", "z"]
    )
    r, theta, z = a.base_scalars()
    assert a.transformation_to_parent() == (r*cos(theta), r*sin(theta), z)
    assert a.lame_coefficients() == (
        sqrt(sin(theta)**2 + cos(theta)**2),
        sqrt(r**2*sin(theta)**2 + r**2*cos(theta)**2),
        1
    )  # ==> this should simplify to (1, a.x, 1)

    raises(TypeError, lambda: CoordSys3D('a', transformation={
        x: x*sin(y)*cos(z), y:x*sin(y)*sin(z), z:  x*cos(y)}))


def test_check_orthogonality():
    x, y, z = symbols('x y z')
    u,v = symbols('u, v')
    a = CoordSys3D('a', transformation=((x, y, z), (x*sin(y)*cos(z), x*sin(y)*sin(z), x*cos(y))))
    assert a._check_orthogonality(a._transformation) is True
    a = CoordSys3D('a', transformation=((x, y, z), (x * cos(y), x * sin(y), z)))
    assert a._check_orthogonality(a._transformation) is True
    a = CoordSys3D('a', transformation=((u, v, z), (cosh(u) * cos(v), sinh(u) * sin(v), z)))
    assert a._check_orthogonality(a._transformation) is True

    raises(ValueError, lambda: CoordSys3D('a', transformation=((x, y, z), (x, x, z))))
    raises(ValueError, lambda: CoordSys3D('a', transformation=(
        (x, y, z), (x*sin(y/2)*cos(z), x*sin(y)*sin(z), x*cos(y)))))


def test_rotation_trans_equations():
    a = CoordSys3D('a')
    from sympy.core.symbol import symbols
    q0 = symbols('q0')
    assert a._rotation_trans_equations(a._parent_rotation_matrix, a.base_scalars()) == (a.x, a.y, a.z)
    assert a._rotation_trans_equations(a._inverse_rotation_matrix(), a.base_scalars()) == (a.x, a.y, a.z)
    b = a.orient_new_axis('b', 0, -a.k)
    assert b._rotation_trans_equations(b._parent_rotation_matrix, b.base_scalars()) == (b.x, b.y, b.z)
    assert b._rotation_trans_equations(b._inverse_rotation_matrix(), b.base_scalars()) == (b.x, b.y, b.z)
    c = a.orient_new_axis('c', q0, -a.k)
    assert c._rotation_trans_equations(c._parent_rotation_matrix, c.base_scalars()) == \
           (-sin(q0) * c.y + cos(q0) * c.x, sin(q0) * c.x + cos(q0) * c.y, c.z)
    assert c._rotation_trans_equations(c._inverse_rotation_matrix(), c.base_scalars()) == \
           (sin(q0) * c.y + cos(q0) * c.x, -sin(q0) * c.x + cos(q0) * c.y, c.z)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\tests\test_url.py ===
# -*- coding: utf-8 -*-
#
import os
import unittest

from websocket._url import (
    _is_address_in_network,
    _is_no_proxy_host,
    get_proxy_info,
    parse_url,
)
from websocket._exceptions import WebSocketProxyException

"""
test_url.py
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


class UrlTest(unittest.TestCase):
    def test_address_in_network(self):
        self.assertTrue(_is_address_in_network("127.0.0.1", "127.0.0.0/8"))
        self.assertTrue(_is_address_in_network("127.1.0.1", "127.0.0.0/8"))
        self.assertFalse(_is_address_in_network("127.1.0.1", "127.0.0.0/24"))

    def test_parse_url(self):
        p = parse_url("ws://www.example.com/r")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com/r/")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/r/")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com/")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com:8080/r")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com:8080/")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com:8080")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/")
        self.assertEqual(p[3], False)

        p = parse_url("wss://www.example.com:8080/r")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], True)

        p = parse_url("wss://www.example.com:8080/r?key=value")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r?key=value")
        self.assertEqual(p[3], True)

        self.assertRaises(ValueError, parse_url, "http://www.example.com/r")

        p = parse_url("ws://[2a03:4000:123:83::3]/r")
        self.assertEqual(p[0], "2a03:4000:123:83::3")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], False)

        p = parse_url("ws://[2a03:4000:123:83::3]:8080/r")
        self.assertEqual(p[0], "2a03:4000:123:83::3")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], False)

        p = parse_url("wss://[2a03:4000:123:83::3]/r")
        self.assertEqual(p[0], "2a03:4000:123:83::3")
        self.assertEqual(p[1], 443)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], True)

        p = parse_url("wss://[2a03:4000:123:83::3]:8080/r")
        self.assertEqual(p[0], "2a03:4000:123:83::3")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], True)


class IsNoProxyHostTest(unittest.TestCase):
    def setUp(self):
        self.no_proxy = os.environ.get("no_proxy", None)
        if "no_proxy" in os.environ:
            del os.environ["no_proxy"]

    def tearDown(self):
        if self.no_proxy:
            os.environ["no_proxy"] = self.no_proxy
        elif "no_proxy" in os.environ:
            del os.environ["no_proxy"]

    def test_match_all(self):
        self.assertTrue(_is_no_proxy_host("any.websocket.org", ["*"]))
        self.assertTrue(_is_no_proxy_host("192.168.0.1", ["*"]))
        self.assertFalse(_is_no_proxy_host("192.168.0.1", ["192.168.1.1"]))
        self.assertFalse(
            _is_no_proxy_host("any.websocket.org", ["other.websocket.org"])
        )
        self.assertTrue(
            _is_no_proxy_host("any.websocket.org", ["other.websocket.org", "*"])
        )
        os.environ["no_proxy"] = "*"
        self.assertTrue(_is_no_proxy_host("any.websocket.org", None))
        self.assertTrue(_is_no_proxy_host("192.168.0.1", None))
        os.environ["no_proxy"] = "other.websocket.org, *"
        self.assertTrue(_is_no_proxy_host("any.websocket.org", None))

    def test_ip_address(self):
        self.assertTrue(_is_no_proxy_host("127.0.0.1", ["127.0.0.1"]))
        self.assertFalse(_is_no_proxy_host("127.0.0.2", ["127.0.0.1"]))
        self.assertTrue(
            _is_no_proxy_host("127.0.0.1", ["other.websocket.org", "127.0.0.1"])
        )
        self.assertFalse(
            _is_no_proxy_host("127.0.0.2", ["other.websocket.org", "127.0.0.1"])
        )
        os.environ["no_proxy"] = "127.0.0.1"
        self.assertTrue(_is_no_proxy_host("127.0.0.1", None))
        self.assertFalse(_is_no_proxy_host("127.0.0.2", None))
        os.environ["no_proxy"] = "other.websocket.org, 127.0.0.1"
        self.assertTrue(_is_no_proxy_host("127.0.0.1", None))
        self.assertFalse(_is_no_proxy_host("127.0.0.2", None))

    def test_ip_address_in_range(self):
        self.assertTrue(_is_no_proxy_host("127.0.0.1", ["127.0.0.0/8"]))
        self.assertTrue(_is_no_proxy_host("127.0.0.2", ["127.0.0.0/8"]))
        self.assertFalse(_is_no_proxy_host("127.1.0.1", ["127.0.0.0/24"]))
        os.environ["no_proxy"] = "127.0.0.0/8"
        self.assertTrue(_is_no_proxy_host("127.0.0.1", None))
        self.assertTrue(_is_no_proxy_host("127.0.0.2", None))
        os.environ["no_proxy"] = "127.0.0.0/24"
        self.assertFalse(_is_no_proxy_host("127.1.0.1", None))

    def test_hostname_match(self):
        self.assertTrue(_is_no_proxy_host("my.websocket.org", ["my.websocket.org"]))
        self.assertTrue(
            _is_no_proxy_host(
                "my.websocket.org", ["other.websocket.org", "my.websocket.org"]
            )
        )
        self.assertFalse(_is_no_proxy_host("my.websocket.org", ["other.websocket.org"]))
        os.environ["no_proxy"] = "my.websocket.org"
        self.assertTrue(_is_no_proxy_host("my.websocket.org", None))
        self.assertFalse(_is_no_proxy_host("other.websocket.org", None))
        os.environ["no_proxy"] = "other.websocket.org, my.websocket.org"
        self.assertTrue(_is_no_proxy_host("my.websocket.org", None))

    def test_hostname_match_domain(self):
        self.assertTrue(_is_no_proxy_host("any.websocket.org", [".websocket.org"]))
        self.assertTrue(_is_no_proxy_host("my.other.websocket.org", [".websocket.org"]))
        self.assertTrue(
            _is_no_proxy_host(
                "any.websocket.org", ["my.websocket.org", ".websocket.org"]
            )
        )
        self.assertFalse(_is_no_proxy_host("any.websocket.com", [".websocket.org"]))
        os.environ["no_proxy"] = ".websocket.org"
        self.assertTrue(_is_no_proxy_host("any.websocket.org", None))
        self.assertTrue(_is_no_proxy_host("my.other.websocket.org", None))
        self.assertFalse(_is_no_proxy_host("any.websocket.com", None))
        os.environ["no_proxy"] = "my.websocket.org, .websocket.org"
        self.assertTrue(_is_no_proxy_host("any.websocket.org", None))


class ProxyInfoTest(unittest.TestCase):
    def setUp(self):
        self.http_proxy = os.environ.get("http_proxy", None)
        self.https_proxy = os.environ.get("https_proxy", None)
        self.no_proxy = os.environ.get("no_proxy", None)
        if "http_proxy" in os.environ:
            del os.environ["http_proxy"]
        if "https_proxy" in os.environ:
            del os.environ["https_proxy"]
        if "no_proxy" in os.environ:
            del os.environ["no_proxy"]

    def tearDown(self):
        if self.http_proxy:
            os.environ["http_proxy"] = self.http_proxy
        elif "http_proxy" in os.environ:
            del os.environ["http_proxy"]

        if self.https_proxy:
            os.environ["https_proxy"] = self.https_proxy
        elif "https_proxy" in os.environ:
            del os.environ["https_proxy"]

        if self.no_proxy:
            os.environ["no_proxy"] = self.no_proxy
        elif "no_proxy" in os.environ:
            del os.environ["no_proxy"]

    def test_proxy_from_args(self):
        self.assertRaises(
            WebSocketProxyException,
            get_proxy_info,
            "echo.websocket.events",
            False,
            proxy_host="localhost",
        )
        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events", False, proxy_host="localhost", proxy_port=3128
            ),
            ("localhost", 3128, None),
        )
        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events", True, proxy_host="localhost", proxy_port=3128
            ),
            ("localhost", 3128, None),
        )

        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events",
                False,
                proxy_host="localhost",
                proxy_port=9001,
                proxy_auth=("a", "b"),
            ),
            ("localhost", 9001, ("a", "b")),
        )
        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events",
                False,
                proxy_host="localhost",
                proxy_port=3128,
                proxy_auth=("a", "b"),
            ),
            ("localhost", 3128, ("a", "b")),
        )
        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events",
                True,
                proxy_host="localhost",
                proxy_port=8765,
                proxy_auth=("a", "b"),
            ),
            ("localhost", 8765, ("a", "b")),
        )
        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events",
                True,
                proxy_host="localhost",
                proxy_port=3128,
                proxy_auth=("a", "b"),
            ),
            ("localhost", 3128, ("a", "b")),
        )

        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events",
                True,
                proxy_host="localhost",
                proxy_port=3128,
                no_proxy=["example.com"],
                proxy_auth=("a", "b"),
            ),
            ("localhost", 3128, ("a", "b")),
        )
        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events",
                True,
                proxy_host="localhost",
                proxy_port=3128,
                no_proxy=["echo.websocket.events"],
                proxy_auth=("a", "b"),
            ),
            (None, 0, None),
        )

        self.assertEqual(
            get_proxy_info(
                "echo.websocket.events",
                True,
                proxy_host="localhost",
                proxy_port=3128,
                no_proxy=[".websocket.events"],
            ),
            (None, 0, None),
        )

    def test_proxy_from_env(self):
        os.environ["http_proxy"] = "http://localhost/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False), ("localhost", None, None)
        )
        os.environ["http_proxy"] = "http://localhost:3128/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False), ("localhost", 3128, None)
        )

        os.environ["http_proxy"] = "http://localhost/"
        os.environ["https_proxy"] = "http://localhost2/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False), ("localhost", None, None)
        )
        os.environ["http_proxy"] = "http://localhost:3128/"
        os.environ["https_proxy"] = "http://localhost2:3128/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False), ("localhost", 3128, None)
        )

        os.environ["http_proxy"] = "http://localhost/"
        os.environ["https_proxy"] = "http://localhost2/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", True), ("localhost2", None, None)
        )
        os.environ["http_proxy"] = "http://localhost:3128/"
        os.environ["https_proxy"] = "http://localhost2:3128/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", True), ("localhost2", 3128, None)
        )

        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = "http://localhost2/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", True), ("localhost2", None, None)
        )
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False), (None, 0, None)
        )
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = "http://localhost2:3128/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", True), ("localhost2", 3128, None)
        )
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False), (None, 0, None)
        )

        os.environ["http_proxy"] = "http://localhost/"
        os.environ["https_proxy"] = ""
        self.assertEqual(get_proxy_info("echo.websocket.events", True), (None, 0, None))
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False), ("localhost", None, None)
        )
        os.environ["http_proxy"] = "http://localhost:3128/"
        os.environ["https_proxy"] = ""
        self.assertEqual(get_proxy_info("echo.websocket.events", True), (None, 0, None))
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False), ("localhost", 3128, None)
        )

        os.environ["http_proxy"] = "http://a:b@localhost/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False),
            ("localhost", None, ("a", "b")),
        )
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False),
            ("localhost", 3128, ("a", "b")),
        )

        os.environ["http_proxy"] = "http://a:b@localhost/"
        os.environ["https_proxy"] = "http://a:b@localhost2/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False),
            ("localhost", None, ("a", "b")),
        )
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", False),
            ("localhost", 3128, ("a", "b")),
        )

        os.environ["http_proxy"] = "http://a:b@localhost/"
        os.environ["https_proxy"] = "http://a:b@localhost2/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", True),
            ("localhost2", None, ("a", "b")),
        )
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", True),
            ("localhost2", 3128, ("a", "b")),
        )

        os.environ[
            "http_proxy"
        ] = "http://john%40example.com:P%40SSWORD@localhost:3128/"
        os.environ[
            "https_proxy"
        ] = "http://john%40example.com:P%40SSWORD@localhost2:3128/"
        self.assertEqual(
            get_proxy_info("echo.websocket.events", True),
            ("localhost2", 3128, ("john@example.com", "P@SSWORD")),
        )

        os.environ["http_proxy"] = "http://a:b@localhost/"
        os.environ["https_proxy"] = "http://a:b@localhost2/"
        os.environ["no_proxy"] = "example1.com,example2.com"
        self.assertEqual(
            get_proxy_info("example.1.com", True), ("localhost2", None, ("a", "b"))
        )
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        os.environ["no_proxy"] = "example1.com,example2.com, echo.websocket.events"
        self.assertEqual(get_proxy_info("echo.websocket.events", True), (None, 0, None))
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        os.environ["no_proxy"] = "example1.com,example2.com, .websocket.events"
        self.assertEqual(get_proxy_info("echo.websocket.events", True), (None, 0, None))

        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        os.environ["no_proxy"] = "127.0.0.0/8, 192.168.0.0/16"
        self.assertEqual(get_proxy_info("127.0.0.1", False), (None, 0, None))
        self.assertEqual(get_proxy_info("192.168.1.1", False), (None, 0, None))


if __name__ == "__main__":
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_ufunc.py ===
from collections import deque
import re
import string

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray


@pytest.fixture(params=[np.add, np.logaddexp])
def ufunc(request):
    # dunder op
    return request.param


@pytest.fixture(
    params=[pytest.param(True, marks=pytest.mark.fails_arm_wheels), False],
    ids=["sparse", "dense"],
)
def sparse(request):
    return request.param


@pytest.fixture
def arrays_for_binary_ufunc():
    """
    A pair of random, length-100 integer-dtype arrays, that are mostly 0.
    """
    a1 = np.random.default_rng(2).integers(0, 10, 100, dtype="int64")
    a2 = np.random.default_rng(2).integers(0, 10, 100, dtype="int64")
    a1[::3] = 0
    a2[::4] = 0
    return a1, a2


@pytest.mark.parametrize("ufunc", [np.positive, np.floor, np.exp])
def test_unary_ufunc(ufunc, sparse):
    # Test that ufunc(pd.Series) == pd.Series(ufunc)
    arr = np.random.default_rng(2).integers(0, 10, 10, dtype="int64")
    arr[::2] = 0
    if sparse:
        arr = SparseArray(arr, dtype=pd.SparseDtype("int64", 0))

    index = list(string.ascii_letters[:10])
    name = "name"
    series = pd.Series(arr, index=index, name=name)

    result = ufunc(series)
    expected = pd.Series(ufunc(arr), index=index, name=name)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_array(flip, sparse, ufunc, arrays_for_binary_ufunc):
    # Test that ufunc(pd.Series(a), array) == pd.Series(ufunc(a, b))
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    name = "name"  # op(pd.Series, array) preserves the name.
    series = pd.Series(a1, name=name)
    other = a2

    array_args = (a1, a2)
    series_args = (series, other)  # ufunc(series, array)

    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)  # ufunc(array, series)

    expected = pd.Series(ufunc(*array_args), name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_index(flip, sparse, ufunc, arrays_for_binary_ufunc):
    # Test that
    #   * func(pd.Series(a), pd.Series(b)) == pd.Series(ufunc(a, b))
    #   * ufunc(Index, pd.Series) dispatches to pd.Series (returns a pd.Series)
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    name = "name"  # op(pd.Series, array) preserves the name.
    series = pd.Series(a1, name=name)

    other = pd.Index(a2, name=name).astype("int64")

    array_args = (a1, a2)
    series_args = (series, other)  # ufunc(series, array)

    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)  # ufunc(array, series)

    expected = pd.Series(ufunc(*array_args), name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("shuffle", [True, False], ids=["unaligned", "aligned"])
@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_series(
    flip, shuffle, sparse, ufunc, arrays_for_binary_ufunc
):
    # Test that
    #   * func(pd.Series(a), pd.Series(b)) == pd.Series(ufunc(a, b))
    #   with alignment between the indices
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    name = "name"  # op(pd.Series, array) preserves the name.
    series = pd.Series(a1, name=name)
    other = pd.Series(a2, name=name)

    idx = np.random.default_rng(2).permutation(len(a1))

    if shuffle:
        other = other.take(idx)
        if flip:
            index = other.align(series)[0].index
        else:
            index = series.align(other)[0].index
    else:
        index = series.index

    array_args = (a1, a2)
    series_args = (series, other)  # ufunc(series, array)

    if flip:
        array_args = tuple(reversed(array_args))
        series_args = tuple(reversed(series_args))  # ufunc(array, series)

    expected = pd.Series(ufunc(*array_args), index=index, name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flip", [True, False])
def test_binary_ufunc_scalar(ufunc, sparse, flip, arrays_for_binary_ufunc):
    # Test that
    #   * ufunc(pd.Series, scalar) == pd.Series(ufunc(array, scalar))
    #   * ufunc(pd.Series, scalar) == ufunc(scalar, pd.Series)
    arr, _ = arrays_for_binary_ufunc
    if sparse:
        arr = SparseArray(arr)
    other = 2
    series = pd.Series(arr, name="name")

    series_args = (series, other)
    array_args = (arr, other)

    if flip:
        series_args = tuple(reversed(series_args))
        array_args = tuple(reversed(array_args))

    expected = pd.Series(ufunc(*array_args), name="name")
    result = ufunc(*series_args)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.divmod])  # TODO: np.modf, np.frexp
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_multiple_output_binary_ufuncs(ufunc, sparse, shuffle, arrays_for_binary_ufunc):
    # Test that
    #  the same conditions from binary_ufunc_scalar apply to
    #  ufuncs with multiple outputs.

    a1, a2 = arrays_for_binary_ufunc
    # work around https://github.com/pandas-dev/pandas/issues/26987
    a1[a1 == 0] = 1
    a2[a2 == 0] = 1

    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    s1 = pd.Series(a1)
    s2 = pd.Series(a2)

    if shuffle:
        # ensure we align before applying the ufunc
        s2 = s2.sample(frac=1)

    expected = ufunc(a1, a2)
    assert isinstance(expected, tuple)

    result = ufunc(s1, s2)
    assert isinstance(result, tuple)
    tm.assert_series_equal(result[0], pd.Series(expected[0]))
    tm.assert_series_equal(result[1], pd.Series(expected[1]))


def test_multiple_output_ufunc(sparse, arrays_for_binary_ufunc):
    # Test that the same conditions from unary input apply to multi-output
    # ufuncs
    arr, _ = arrays_for_binary_ufunc

    if sparse:
        arr = SparseArray(arr)

    series = pd.Series(arr, name="name")
    result = np.modf(series)
    expected = np.modf(arr)

    assert isinstance(result, tuple)
    assert isinstance(expected, tuple)

    tm.assert_series_equal(result[0], pd.Series(expected[0], name="name"))
    tm.assert_series_equal(result[1], pd.Series(expected[1], name="name"))


def test_binary_ufunc_drops_series_name(ufunc, sparse, arrays_for_binary_ufunc):
    # Drop the names when they differ.
    a1, a2 = arrays_for_binary_ufunc
    s1 = pd.Series(a1, name="a")
    s2 = pd.Series(a2, name="b")

    result = ufunc(s1, s2)
    assert result.name is None


def test_object_series_ok():
    class Dummy:
        def __init__(self, value) -> None:
            self.value = value

        def __add__(self, other):
            return self.value + other.value

    arr = np.array([Dummy(0), Dummy(1)])
    ser = pd.Series(arr)
    tm.assert_series_equal(np.add(ser, ser), pd.Series(np.add(ser, arr)))
    tm.assert_series_equal(np.add(ser, Dummy(1)), pd.Series(np.add(ser, Dummy(1))))


@pytest.fixture(
    params=[
        pd.array([1, 3, 2], dtype=np.int64),
        pd.array([1, 3, 2], dtype="Int64"),
        pd.array([1, 3, 2], dtype="Float32"),
        pd.array([1, 10, 2], dtype="Sparse[int]"),
        pd.to_datetime(["2000", "2010", "2001"]),
        pd.to_datetime(["2000", "2010", "2001"]).tz_localize("CET"),
        pd.to_datetime(["2000", "2010", "2001"]).to_period(freq="D"),
        pd.to_timedelta(["1 Day", "3 Days", "2 Days"]),
        pd.IntervalIndex([pd.Interval(0, 1), pd.Interval(2, 3), pd.Interval(1, 2)]),
    ],
    ids=lambda x: str(x.dtype),
)
def values_for_np_reduce(request):
    # min/max tests assume that these are monotonic increasing
    return request.param


class TestNumpyReductions:
    # TODO: cases with NAs, axis kwarg for DataFrame

    def test_multiply(self, values_for_np_reduce, box_with_array, request):
        box = box_with_array
        values = values_for_np_reduce

        with tm.assert_produces_warning(None):
            obj = box(values)

        if isinstance(values, pd.core.arrays.SparseArray):
            mark = pytest.mark.xfail(reason="SparseArray has no 'prod'")
            request.applymarker(mark)

        if values.dtype.kind in "iuf":
            result = np.multiply.reduce(obj)
            if box is pd.DataFrame:
                expected = obj.prod(numeric_only=False)
                tm.assert_series_equal(result, expected)
            elif box is pd.Index:
                # Index has no 'prod'
                expected = obj._values.prod()
                assert result == expected
            else:
                expected = obj.prod()
                assert result == expected
        else:
            msg = "|".join(
                [
                    "does not support reduction",
                    "unsupported operand type",
                    "ufunc 'multiply' cannot use operands",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                np.multiply.reduce(obj)

    def test_add(self, values_for_np_reduce, box_with_array):
        box = box_with_array
        values = values_for_np_reduce

        with tm.assert_produces_warning(None):
            obj = box(values)

        if values.dtype.kind in "miuf":
            result = np.add.reduce(obj)
            if box is pd.DataFrame:
                expected = obj.sum(numeric_only=False)
                tm.assert_series_equal(result, expected)
            elif box is pd.Index:
                # Index has no 'sum'
                expected = obj._values.sum()
                assert result == expected
            else:
                expected = obj.sum()
                assert result == expected
        else:
            msg = "|".join(
                [
                    "does not support reduction",
                    "unsupported operand type",
                    "ufunc 'add' cannot use operands",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                np.add.reduce(obj)

    def test_max(self, values_for_np_reduce, box_with_array):
        box = box_with_array
        values = values_for_np_reduce

        same_type = True
        if box is pd.Index and values.dtype.kind in ["i", "f"]:
            # ATM Index casts to object, so we get python ints/floats
            same_type = False

        with tm.assert_produces_warning(None):
            obj = box(values)

        result = np.maximum.reduce(obj)
        if box is pd.DataFrame:
            # TODO: cases with axis kwarg
            expected = obj.max(numeric_only=False)
            tm.assert_series_equal(result, expected)
        else:
            expected = values[1]
            assert result == expected
            if same_type:
                # check we have e.g. Timestamp instead of dt64
                assert type(result) == type(expected)

    def test_min(self, values_for_np_reduce, box_with_array):
        box = box_with_array
        values = values_for_np_reduce

        same_type = True
        if box is pd.Index and values.dtype.kind in ["i", "f"]:
            # ATM Index casts to object, so we get python ints/floats
            same_type = False

        with tm.assert_produces_warning(None):
            obj = box(values)

        result = np.minimum.reduce(obj)
        if box is pd.DataFrame:
            expected = obj.min(numeric_only=False)
            tm.assert_series_equal(result, expected)
        else:
            expected = values[0]
            assert result == expected
            if same_type:
                # check we have e.g. Timestamp instead of dt64
                assert type(result) == type(expected)


@pytest.mark.parametrize("type_", [list, deque, tuple])
def test_binary_ufunc_other_types(type_):
    a = pd.Series([1, 2, 3], name="name")
    b = type_([3, 4, 5])

    result = np.add(a, b)
    expected = pd.Series(np.add(a.to_numpy(), b), name="name")
    tm.assert_series_equal(result, expected)


def test_object_dtype_ok():
    class Thing:
        def __init__(self, value) -> None:
            self.value = value

        def __add__(self, other):
            other = getattr(other, "value", other)
            return type(self)(self.value + other)

        def __eq__(self, other) -> bool:
            return type(other) is Thing and self.value == other.value

        def __repr__(self) -> str:
            return f"Thing({self.value})"

    s = pd.Series([Thing(1), Thing(2)])
    result = np.add(s, Thing(1))
    expected = pd.Series([Thing(2), Thing(3)])
    tm.assert_series_equal(result, expected)


def test_outer():
    # https://github.com/pandas-dev/pandas/issues/27186
    ser = pd.Series([1, 2, 3])
    obj = np.array([1, 2, 3])

    with pytest.raises(NotImplementedError, match=""):
        np.subtract.outer(ser, obj)


def test_np_matmul():
    # GH26650
    df1 = pd.DataFrame(data=[[-1, 1, 10]])
    df2 = pd.DataFrame(data=[-1, 1, 10])
    expected = pd.DataFrame(data=[102])

    result = np.matmul(df1, df2)
    tm.assert_frame_equal(expected, result)


def test_array_ufuncs_for_many_arguments():
    # GH39853
    def add3(x, y, z):
        return x + y + z

    ufunc = np.frompyfunc(add3, 3, 1)
    ser = pd.Series([1, 2])

    result = ufunc(ser, ser, 1)
    expected = pd.Series([3, 5], dtype=object)
    tm.assert_series_equal(result, expected)

    df = pd.DataFrame([[1, 2]])

    msg = (
        "Cannot apply ufunc <ufunc 'add3 (vectorized)'> "
        "to mixed DataFrame and Series inputs."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        ufunc(ser, ser, df)


# TODO(CoW) see https://github.com/pandas-dev/pandas/pull/51082
@td.skip_copy_on_write_not_yet_implemented
def test_np_fix():
    # np.fix is not a ufunc but is composed of several ufunc calls under the hood
    # with `out` and `where` keywords
    ser = pd.Series([-1.5, -0.5, 0.5, 1.5])
    result = np.fix(ser)
    expected = pd.Series([-1.0, -0.0, 0.0, 1.0])
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_polysys.py ===
"""Tests for solvers of systems of polynomial equations. """
from sympy.polys.domains import  ZZ, QQ_I
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polyerrors import UnsolvableFactorError
from sympy.polys.polyoptions import Options
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import flatten
from sympy.abc import a, b, c, x, y, z
from sympy.polys import PolynomialError
from sympy.solvers.polysys import (solve_poly_system,
                                   solve_triangulated,
                                   solve_biquadratic, SolveFailed,
                                   solve_generic, factor_system_bool,
                                   factor_system_cond, factor_system_poly,
                                   factor_system, _factor_sets, _factor_sets_slow)
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.testing.pytest import raises
from sympy.core.relational import Eq
from sympy.functions.elementary.trigonometric import sin, cos

from sympy.functions.elementary.exponential import exp


def test_solve_poly_system():
    assert solve_poly_system([x - 1], x) == [(S.One,)]

    assert solve_poly_system([y - x, y - x - 1], x, y) is None

    assert solve_poly_system([y - x**2, y + x**2], x, y) == [(S.Zero, S.Zero)]

    assert solve_poly_system([2*x - 3, y*Rational(3, 2) - 2*x, z - 5*y], x, y, z) == \
        [(Rational(3, 2), Integer(2), Integer(10))]

    assert solve_poly_system([x*y - 2*y, 2*y**2 - x**2], x, y) == \
        [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]

    assert solve_poly_system([y - x**2, y + x**2 + 1], x, y) == \
        [(-I*sqrt(S.Half), Rational(-1, 2)), (I*sqrt(S.Half), Rational(-1, 2))]

    f_1 = x**2 + y + z - 1
    f_2 = x + y**2 + z - 1
    f_3 = x + y + z**2 - 1

    a, b = sqrt(2) - 1, -sqrt(2) - 1

    assert solve_poly_system([f_1, f_2, f_3], x, y, z) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]

    solution = [(1, -1), (1, 1)]

    assert solve_poly_system([Poly(x**2 - y**2), Poly(x - 1)]) == solution
    assert solve_poly_system([x**2 - y**2, x - 1], x, y) == solution
    assert solve_poly_system([x**2 - y**2, x - 1]) == solution

    assert solve_poly_system(
        [x + x*y - 3, y + x*y - 4], x, y) == [(-3, -2), (1, 2)]

    raises(NotImplementedError, lambda: solve_poly_system([x**3 - y**3], x, y))
    raises(NotImplementedError, lambda: solve_poly_system(
        [z, -2*x*y**2 + x + y**2*z, y**2*(-z - 4) + 2]))
    raises(PolynomialError, lambda: solve_poly_system([1/x], x))

    raises(NotImplementedError, lambda: solve_poly_system(
          [x-1,], (x, y)))
    raises(NotImplementedError, lambda: solve_poly_system(
          [y-1,], (x, y)))

    # solve_poly_system should ideally construct solutions using
    # CRootOf for the following four tests
    assert solve_poly_system([x**5 - x + 1], [x], strict=False) == []
    raises(UnsolvableFactorError, lambda: solve_poly_system(
        [x**5 - x + 1], [x], strict=True))

    assert solve_poly_system([(x - 1)*(x**5 - x + 1), y**2 - 1], [x, y],
                             strict=False) == [(1, -1), (1, 1)]
    raises(UnsolvableFactorError,
           lambda: solve_poly_system([(x - 1)*(x**5 - x + 1), y**2-1],
                                     [x, y], strict=True))


def test_solve_generic():
    NewOption = Options((x, y), {'domain': 'ZZ'})
    assert solve_generic([x**2 - 2*y**2, y**2 - y + 1], NewOption) == \
           [(-sqrt(-1 - sqrt(3)*I), Rational(1, 2) - sqrt(3)*I/2),
            (sqrt(-1 - sqrt(3)*I), Rational(1, 2) - sqrt(3)*I/2),
            (-sqrt(-1 + sqrt(3)*I), Rational(1, 2) + sqrt(3)*I/2),
            (sqrt(-1 + sqrt(3)*I), Rational(1, 2) + sqrt(3)*I/2)]

    # solve_generic should ideally construct solutions using
    # CRootOf for the following two tests
    assert solve_generic(
        [2*x - y, (y - 1)*(y**5 - y + 1)], NewOption, strict=False) == \
        [(Rational(1, 2), 1)]
    raises(UnsolvableFactorError, lambda: solve_generic(
        [2*x - y, (y - 1)*(y**5 - y + 1)], NewOption, strict=True))


def test_solve_biquadratic():
    x0, y0, x1, y1, r = symbols('x0 y0 x1 y1 r')

    f_1 = (x - 1)**2 + (y - 1)**2 - r**2
    f_2 = (x - 2)**2 + (y - 2)**2 - r**2
    s = sqrt(2*r**2 - 1)
    a = (3 - s)/2
    b = (3 + s)/2
    assert solve_poly_system([f_1, f_2], x, y) == [(a, b), (b, a)]

    f_1 = (x - 1)**2 + (y - 2)**2 - r**2
    f_2 = (x - 1)**2 + (y - 1)**2 - r**2

    assert solve_poly_system([f_1, f_2], x, y) == \
        [(1 - sqrt((2*r - 1)*(2*r + 1))/2, Rational(3, 2)),
         (1 + sqrt((2*r - 1)*(2*r + 1))/2, Rational(3, 2))]

    query = lambda expr: expr.is_Pow and expr.exp is S.Half

    f_1 = (x - 1 )**2 + (y - 2)**2 - r**2
    f_2 = (x - x1)**2 + (y - 1)**2 - r**2

    result = solve_poly_system([f_1, f_2], x, y)

    assert len(result) == 2 and all(len(r) == 2 for r in result)
    assert all(r.count(query) == 1 for r in flatten(result))

    f_1 = (x - x0)**2 + (y - y0)**2 - r**2
    f_2 = (x - x1)**2 + (y - y1)**2 - r**2

    result = solve_poly_system([f_1, f_2], x, y)

    assert len(result) == 2 and all(len(r) == 2 for r in result)
    assert all(len(r.find(query)) == 1 for r in flatten(result))

    s1 = (x*y - y, x**2 - x)
    assert solve(s1) == [{x: 1}, {x: 0, y: 0}]
    s2 = (x*y - x, y**2 - y)
    assert solve(s2) == [{y: 1}, {x: 0, y: 0}]
    gens = (x, y)
    for seq in (s1, s2):
        (f, g), opt = parallel_poly_from_expr(seq, *gens)
        raises(SolveFailed, lambda: solve_biquadratic(f, g, opt))
    seq = (x**2 + y**2 - 2, y**2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == [
        (-1, -1), (-1, 1), (1, -1), (1, 1)]
    ans = [(0, -1), (0, 1)]
    seq = (x**2 + y**2 - 1, y**2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == ans
    seq = (x**2 + y**2 - 1, x**2 - x + y**2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == ans


def test_solve_triangulated():
    f_1 = x**2 + y + z - 1
    f_2 = x + y**2 + z - 1
    f_3 = x + y + z**2 - 1

    a, b = sqrt(2) - 1, -sqrt(2) - 1

    assert solve_triangulated([f_1, f_2, f_3], x, y, z) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    dom = QQ.algebraic_field(sqrt(2))

    assert solve_triangulated([f_1, f_2, f_3], x, y, z, domain=dom) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]

    a, b = CRootOf(z**2 + 2*z - 1, 0), CRootOf(z**2 + 2*z - 1, 1)
    assert solve_triangulated([f_1, f_2, f_3], x, y, z, extension=True) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]


def test_solve_issue_3686():
    roots = solve_poly_system([((x - 5)**2/250000 + (y - Rational(5, 10))**2/250000) - 1, x], x, y)
    assert roots == [(0, S.Half - 15*sqrt(1111)), (0, S.Half + 15*sqrt(1111))]

    roots = solve_poly_system([((x - 5)**2/250000 + (y - 5.0/10)**2/250000) - 1, x], x, y)
    # TODO: does this really have to be so complicated?!
    assert len(roots) == 2
    assert roots[0][0] == 0
    assert roots[0][1].epsilon_eq(-499.474999374969, 1e12)
    assert roots[1][0] == 0
    assert roots[1][1].epsilon_eq(500.474999374969, 1e12)


def test_factor_system():

    assert factor_system([x**2 + 2*x + 1]) ==  [[x + 1]]
    assert factor_system([x**2 + 2*x + 1, y**2 + 2*y + 1]) ==  [[x + 1, y + 1]]
    assert factor_system([x**2 + 1]) ==  [[x**2 + 1]]
    assert factor_system([]) == [[]]

    assert factor_system([x**2 + y**2 + 2*x*y, x**2 - 2], extension=sqrt(2)) == [
        [x + y, x + sqrt(2)],
        [x + y, x - sqrt(2)],
    ]

    assert factor_system([x**2 + 1, y**2 + 1], gaussian=True) == [
        [x + I, y + I],
        [x + I, y - I],
        [x - I, y + I],
        [x - I, y - I],
    ]

    assert factor_system([x**2 + 1, y**2 + 1], domain=QQ_I) == [
        [x + I, y + I],
        [x + I, y - I],
        [x - I, y + I],
        [x - I, y - I],
    ]

    assert factor_system([0]) == [[]]
    assert factor_system([1]) == []
    assert factor_system([0 , x]) == [[x]]
    assert factor_system([1, 0, x]) == []

    assert factor_system([x**4 - 1, y**6 - 1]) == [
        [x**2 + 1, y**2 + y + 1],
        [x**2 + 1, y**2 - y + 1],
        [x**2 + 1, y + 1],
        [x**2 + 1, y - 1],
        [x + 1, y**2 + y + 1],
        [x + 1, y**2 - y + 1],
        [x - 1, y**2 + y + 1],
        [x - 1, y**2 - y + 1],
        [x + 1, y + 1],
        [x + 1, y - 1],
        [x - 1, y + 1],
        [x - 1, y - 1],
    ]

    assert factor_system([(x - 1)*(y - 2), (y - 2)*(z - 3)]) == [
        [x - 1, z - 3],
        [y - 2]
    ]

    assert factor_system([sin(x)**2 + cos(x)**2 - 1, x]) == [
        [x, sin(x)**2 + cos(x)**2 - 1],
    ]

    assert factor_system([sin(x)**2 + cos(x)**2 - 1]) == [
        [sin(x)**2 + cos(x)**2 - 1]
    ]

    assert factor_system([sin(x)**2 + cos(x)**2]) == [
        [sin(x)**2 + cos(x)**2]
    ]

    assert factor_system([a*x, y, a]) == [[y, a]]

    assert factor_system([a*x, y, a], [x, y]) == []

    assert factor_system([a ** 2 * x, y], [x, y]) == [[x, y]]

    assert factor_system([a*x*(x - 1), b*y, c], [x, y]) == []

    assert factor_system([a*x*(x - 1), b*y, c], [x, y, c]) == [
        [x - 1, y, c],
        [x, y, c],
    ]

    assert factor_system([a*x*(x - 1), b*y, c]) == [
        [x - 1, y, c],
        [x, y, c],
        [x - 1, b, c],
        [x, b, c],
        [y, a, c],
        [a, b, c],
    ]

    assert factor_system([x**2 - 2], [y]) == []

    assert factor_system([x**2 - 2], [x]) == [[x**2 - 2]]

    assert factor_system([cos(x)**2 - sin(x)**2, cos(x)**2 + sin(x)**2 - 1]) == [
        [sin(x)**2 + cos(x)**2 - 1, sin(x) + cos(x)],
        [sin(x)**2 + cos(x)**2 - 1, -sin(x) + cos(x)],
    ]

    assert factor_system([(cos(x) + sin(x))**2 - 1, cos(x)**2 - sin(x)**2 - cos(2*x)]) == [
        [sin(x)**2 - cos(x)**2 + cos(2*x), sin(x) + cos(x) + 1],
        [sin(x)**2 - cos(x)**2 + cos(2*x), sin(x) + cos(x) - 1],
    ]

    assert factor_system([(cos(x) + sin(x))*exp(y) - 1, (cos(x) - sin(x))*exp(y) - 1]) == [
        [exp(y)*sin(x) + exp(y)*cos(x) - 1, -exp(y)*sin(x) + exp(y)*cos(x) - 1]
    ]


def test_factor_system_poly():

    px = lambda e: Poly(e, x)
    pxab = lambda e: Poly(e, x, domain=ZZ[a, b])
    pxI = lambda e: Poly(e, x, domain=QQ_I)
    pxyz = lambda e: Poly(e, (x, y, z))

    assert factor_system_poly([px(x**2 - 1), px(x**2 - 4)]) == [
        [px(x + 2), px(x + 1)],
        [px(x + 2), px(x - 1)],
        [px(x + 1), px(x - 2)],
        [px(x - 1), px(x - 2)],
    ]

    assert factor_system_poly([px(x**2 - 1)]) == [[px(x + 1)], [px(x - 1)]]

    assert factor_system_poly([pxyz(x**2*y - y), pxyz(x**2*z - z)]) == [
        [pxyz(x + 1)],
        [pxyz(x - 1)],
        [pxyz(y), pxyz(z)],
    ]

    assert factor_system_poly([px(x**2*(x - 1)**2), px(x*(x - 1))]) == [
        [px(x)],
        [px(x - 1)],
    ]

    assert factor_system_poly([pxyz(x**2 + y*x), pxyz(x**2 + z*x)]) == [
        [pxyz(x + y), pxyz(x + z)],
        [pxyz(x)],
    ]

    assert factor_system_poly([pxab((a - 1)*(x - 2)), pxab((b - 3)*(x - 2))]) == [
        [pxab(x - 2)],
        [pxab(a - 1), pxab(b - 3)],
    ]

    assert factor_system_poly([pxI(x**2 + 1)]) == [[pxI(x + I)], [pxI(x - I)]]

    assert factor_system_poly([]) == [[]]

    assert factor_system_poly([px(1)]) == []
    assert factor_system_poly([px(0), px(x)]) == [[px(x)]]


def test_factor_system_cond():

    assert factor_system_cond([x ** 2 - 1, x ** 2 - 4]) == [
        [x + 2, x + 1],
        [x + 2, x - 1],
        [x + 1, x - 2],
        [x - 1, x - 2],
    ]

    assert factor_system_cond([1]) == []
    assert factor_system_cond([0]) == [[]]
    assert factor_system_cond([1, x]) == []
    assert factor_system_cond([0, x]) == [[x]]
    assert factor_system_cond([]) == [[]]

    assert factor_system_cond([x**2 + y*x]) == [[x + y], [x]]

    assert factor_system_cond([(a - 1)*(x - 2), (b - 3)*(x - 2)], [x]) == [
        [x - 2],
        [a - 1, b - 3],
    ]

    assert factor_system_cond([a * (x - 1), b], [x]) == [[x - 1, b], [a, b]]

    assert factor_system_cond([a*x*(x-1), b*y, c], [x, y]) == [
        [x - 1, y, c],
        [x, y, c],
        [x - 1, b, c],
        [x, b, c],
        [y, a, c],
        [a, b, c],
    ]

    assert factor_system_cond([x*(x-1), y], [x, y]) == [[x - 1, y], [x, y]]

    assert factor_system_cond([a*x, y, a], [x, y]) == [[y, a]]

    assert factor_system_cond([a*x, b*x], [x, y]) == [[x], [a, b]]

    assert factor_system_cond([a*b*x, y], [x, y]) == [[x, y], [y, a*b]]

    assert factor_system_cond([a*b*x, y]) == [[x, y], [y, a], [y, b]]

    assert factor_system_cond([a**2*x, y], [x, y]) == [[x, y], [y, a]]

def test_factor_system_bool():

    eqs = [a*(x - 1)*(y - 1), b*(x - 2)*(y - 1)*(y - 2)]
    assert factor_system_bool(eqs, [x, y]) == (
        Eq(y - 1, 0)
        | (Eq(a, 0) & Eq(b, 0))
        | (Eq(a, 0) & Eq(x - 2, 0))
        | (Eq(a, 0) & Eq(y - 2, 0))
        | (Eq(b, 0) & Eq(x - 1, 0))
        | (Eq(x - 2, 0) & Eq(x - 1, 0))
        | (Eq(x - 1, 0) & Eq(y - 2, 0))
    )

    assert factor_system_bool([x - 1], [x]) == Eq(x - 1, 0)

    assert factor_system_bool([(x - 1)*(x - 2)], [x]) == Eq(x - 2, 0) | Eq(x - 1, 0)

    assert factor_system_bool([], [x]) == True
    assert factor_system_bool([0], [x]) == True
    assert factor_system_bool([1], [x]) == False
    assert factor_system_bool([a], [x]) == Eq(a, 0)

    assert factor_system_bool([a * x, y, a], [x, y]) == Eq(a, 0) & Eq(y, 0)

    assert (factor_system_bool([a*x, b*y*x, a], [x, y]) == (
        Eq(a, 0) & Eq(b, 0))
        | (Eq(a, 0) & Eq(x, 0))
        | (Eq(a, 0) & Eq(y, 0)))

    assert (factor_system_bool([a*x, b*x], [x, y]) == Eq(x, 0) |
            (Eq(a, 0) & Eq(b, 0)))

    assert (factor_system_bool([a*b*x, y], [x, y]) == (
        Eq(x, 0) & Eq(y, 0)) |
        (Eq(y, 0) & Eq(a*b, 0)))

    assert (factor_system_bool([a**2*x, y], [x, y]) == (
        Eq(a, 0) & Eq(y, 0)) |
        (Eq(x, 0) & Eq(y, 0)))

    assert factor_system_bool([a*x*y, b*y*z], [x, y, z]) == (
        Eq(y, 0)
        | (Eq(a, 0) & Eq(b, 0))
        | (Eq(a, 0) & Eq(z, 0))
        | (Eq(b, 0) & Eq(x, 0))
        | (Eq(x, 0) & Eq(z, 0))
    )

    assert factor_system_bool([a*(x - 1), b], [x]) == (
        (Eq(a, 0) & Eq(b, 0))
        | (Eq(x - 1, 0) & Eq(b, 0))
    )


def test_factor_sets():
    #
    from random import randint

    def generate_random_system(n_eqs=3, n_factors=2, max_val=10):
        return [
            [randint(0, max_val) for _ in range(randint(1, n_factors))]
            for _ in range(n_eqs)
        ]

    test_cases = [
        [[1, 2], [1, 3]],
        [[1, 2], [3, 4]],
        [[1], [1, 2], [2]],
    ]

    for case in test_cases:
        assert _factor_sets(case) == _factor_sets_slow(case)

    for _ in range(100):
        system = generate_random_system()
        assert _factor_sets(system) == _factor_sets_slow(system)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\resample\test_base.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas.core.dtypes.common import is_extension_array_dtype

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.groupby.groupby import DataError
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.core.resample import _asfreq_compat

# a fixture value can be overridden by the test parameter value. Note that the
# value of the fixture can be overridden this way even if the test doesn't use
# it directly (doesn't mention it in the function prototype).
# see https://docs.pytest.org/en/latest/fixture.html#override-a-fixture-with-direct-test-parametrization  # noqa: E501
# in this module we override the fixture values defined in conftest.py
# tuples of '_index_factory,_series_name,_index_start,_index_end'
DATE_RANGE = (date_range, "dti", datetime(2005, 1, 1), datetime(2005, 1, 10))
PERIOD_RANGE = (period_range, "pi", datetime(2005, 1, 1), datetime(2005, 1, 10))
TIMEDELTA_RANGE = (timedelta_range, "tdi", "1 day", "10 day")

all_ts = pytest.mark.parametrize(
    "_index_factory,_series_name,_index_start,_index_end",
    [DATE_RANGE, PERIOD_RANGE, TIMEDELTA_RANGE],
)


@pytest.fixture
def create_index(_index_factory):
    def _create_index(*args, **kwargs):
        """return the _index_factory created using the args, kwargs"""
        return _index_factory(*args, **kwargs)

    return _create_index


@pytest.mark.parametrize("freq", ["2D", "1h"])
@pytest.mark.parametrize(
    "_index_factory,_series_name,_index_start,_index_end", [DATE_RANGE, TIMEDELTA_RANGE]
)
def test_asfreq(series_and_frame, freq, create_index):
    obj = series_and_frame

    result = obj.resample(freq).asfreq()
    new_index = create_index(obj.index[0], obj.index[-1], freq=freq)
    expected = obj.reindex(new_index)
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "_index_factory,_series_name,_index_start,_index_end", [DATE_RANGE, TIMEDELTA_RANGE]
)
def test_asfreq_fill_value(series, create_index):
    # test for fill value during resampling, issue 3715

    ser = series

    result = ser.resample("1h").asfreq()
    new_index = create_index(ser.index[0], ser.index[-1], freq="1h")
    expected = ser.reindex(new_index)
    tm.assert_series_equal(result, expected)

    # Explicit cast to float to avoid implicit cast when setting None
    frame = ser.astype("float").to_frame("value")
    frame.iloc[1] = None
    result = frame.resample("1h").asfreq(fill_value=4.0)
    new_index = create_index(frame.index[0], frame.index[-1], freq="1h")
    expected = frame.reindex(new_index, fill_value=4.0)
    tm.assert_frame_equal(result, expected)


@all_ts
def test_resample_interpolate(frame):
    # GH#12925
    df = frame
    warn = None
    if isinstance(df.index, PeriodIndex):
        warn = FutureWarning
    msg = "Resampling with a PeriodIndex is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        result = df.resample("1min").asfreq().interpolate()
        expected = df.resample("1min").interpolate()
    tm.assert_frame_equal(result, expected)


def test_raises_on_non_datetimelike_index():
    # this is a non datetimelike index
    xp = DataFrame()
    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, "
        "but got an instance of 'RangeIndex'"
    )
    with pytest.raises(TypeError, match=msg):
        xp.resample("YE")


@all_ts
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_empty_series(freq, empty_series_dti, resample_method):
    # GH12771 & GH12868

    ser = empty_series_dti
    if freq == "ME" and isinstance(ser.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq)
        return
    elif freq == "ME" and isinstance(ser.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"

    warn = None
    if isinstance(ser.index, PeriodIndex):
        warn = FutureWarning
    msg = "Resampling with a PeriodIndex is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        rs = ser.resample(freq)
    result = getattr(rs, resample_method)()

    if resample_method == "ohlc":
        expected = DataFrame(
            [], index=ser.index[:0].copy(), columns=["open", "high", "low", "close"]
        )
        expected.index = _asfreq_compat(ser.index, freq)
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        expected = ser.copy()
        expected.index = _asfreq_compat(ser.index, freq)
        tm.assert_series_equal(result, expected, check_dtype=False)

    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq


@all_ts
@pytest.mark.parametrize(
    "freq",
    [
        pytest.param("ME", marks=pytest.mark.xfail(reason="Don't know why this fails")),
        "D",
        "h",
    ],
)
def test_resample_nat_index_series(freq, series, resample_method):
    # GH39227

    ser = series.copy()
    ser.index = PeriodIndex([NaT] * len(ser), freq=freq)

    msg = "Resampling with a PeriodIndex is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.resample(freq)
    result = getattr(rs, resample_method)()

    if resample_method == "ohlc":
        expected = DataFrame(
            [], index=ser.index[:0].copy(), columns=["open", "high", "low", "close"]
        )
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        expected = ser[:0].copy()
        tm.assert_series_equal(result, expected, check_dtype=False)
    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq


@all_ts
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
@pytest.mark.parametrize("resample_method", ["count", "size"])
def test_resample_count_empty_series(freq, empty_series_dti, resample_method):
    # GH28427
    ser = empty_series_dti
    if freq == "ME" and isinstance(ser.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq)
        return
    elif freq == "ME" and isinstance(ser.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"

    warn = None
    if isinstance(ser.index, PeriodIndex):
        warn = FutureWarning
    msg = "Resampling with a PeriodIndex is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        rs = ser.resample(freq)

    result = getattr(rs, resample_method)()

    index = _asfreq_compat(ser.index, freq)

    expected = Series([], dtype="int64", index=index, name=ser.name)

    tm.assert_series_equal(result, expected)


@all_ts
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_empty_dataframe(empty_frame_dti, freq, resample_method):
    # GH13212
    df = empty_frame_dti
    # count retains dimensions too
    if freq == "ME" and isinstance(df.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            df.resample(freq, group_keys=False)
        return
    elif freq == "ME" and isinstance(df.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"

    warn = None
    if isinstance(df.index, PeriodIndex):
        warn = FutureWarning
    msg = "Resampling with a PeriodIndex is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        rs = df.resample(freq, group_keys=False)
    result = getattr(rs, resample_method)()
    if resample_method == "ohlc":
        # TODO: no tests with len(df.columns) > 0
        mi = MultiIndex.from_product([df.columns, ["open", "high", "low", "close"]])
        expected = DataFrame(
            [], index=df.index[:0].copy(), columns=mi, dtype=np.float64
        )
        expected.index = _asfreq_compat(df.index, freq)

    elif resample_method != "size":
        expected = df.copy()
    else:
        # GH14962
        expected = Series([], dtype=np.int64)

    expected.index = _asfreq_compat(df.index, freq)

    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq
    tm.assert_almost_equal(result, expected)

    # test size for GH13212 (currently stays as df)


@all_ts
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_count_empty_dataframe(freq, empty_frame_dti):
    # GH28427

    empty_frame_dti["a"] = []

    if freq == "ME" and isinstance(empty_frame_dti.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            empty_frame_dti.resample(freq)
        return
    elif freq == "ME" and isinstance(empty_frame_dti.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"

    warn = None
    if isinstance(empty_frame_dti.index, PeriodIndex):
        warn = FutureWarning
    msg = "Resampling with a PeriodIndex is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        rs = empty_frame_dti.resample(freq)
    result = rs.count()

    index = _asfreq_compat(empty_frame_dti.index, freq)

    expected = DataFrame(dtype="int64", index=index, columns=Index(["a"], dtype=object))

    tm.assert_frame_equal(result, expected)


@all_ts
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_size_empty_dataframe(freq, empty_frame_dti):
    # GH28427

    empty_frame_dti["a"] = []

    if freq == "ME" and isinstance(empty_frame_dti.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            empty_frame_dti.resample(freq)
        return
    elif freq == "ME" and isinstance(empty_frame_dti.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"

    msg = "Resampling with a PeriodIndex"
    warn = None
    if isinstance(empty_frame_dti.index, PeriodIndex):
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        rs = empty_frame_dti.resample(freq)
    result = rs.size()

    index = _asfreq_compat(empty_frame_dti.index, freq)

    expected = Series([], dtype="int64", index=index)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        PeriodIndex([], freq="M", name="a"),
        DatetimeIndex([], name="a"),
        TimedeltaIndex([], name="a"),
    ],
)
@pytest.mark.parametrize("dtype", [float, int, object, "datetime64[ns]"])
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_resample_empty_dtypes(index, dtype, resample_method):
    # Empty series were sometimes causing a segfault (for the functions
    # with Cython bounds-checking disabled) or an IndexError.  We just run
    # them to ensure they no longer do.  (GH #10228)
    warn = None
    if isinstance(index, PeriodIndex):
        # GH#53511
        index = PeriodIndex([], freq="B", name=index.name)
        warn = FutureWarning
    msg = "Resampling with a PeriodIndex is deprecated"

    empty_series_dti = Series([], index, dtype)
    with tm.assert_produces_warning(warn, match=msg):
        rs = empty_series_dti.resample("d", group_keys=False)
    try:
        getattr(rs, resample_method)()
    except DataError:
        # Ignore these since some combinations are invalid
        # (ex: doing mean with dtype of np.object_)
        pass


@all_ts
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_apply_to_empty_series(empty_series_dti, freq):
    # GH 14313
    ser = empty_series_dti

    if freq == "ME" and isinstance(empty_series_dti.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            empty_series_dti.resample(freq)
        return
    elif freq == "ME" and isinstance(empty_series_dti.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"

    msg = "Resampling with a PeriodIndex"
    warn = None
    if isinstance(empty_series_dti.index, PeriodIndex):
        warn = FutureWarning

    with tm.assert_produces_warning(warn, match=msg):
        rs = ser.resample(freq, group_keys=False)

    result = rs.apply(lambda x: 1)
    with tm.assert_produces_warning(warn, match=msg):
        expected = ser.resample(freq).apply("sum")

    tm.assert_series_equal(result, expected, check_dtype=False)


@all_ts
def test_resampler_is_iterable(series):
    # GH 15314
    freq = "h"
    tg = Grouper(freq=freq, convention="start")
    msg = "Resampling with a PeriodIndex"
    warn = None
    if isinstance(series.index, PeriodIndex):
        warn = FutureWarning

    with tm.assert_produces_warning(warn, match=msg):
        grouped = series.groupby(tg)

    with tm.assert_produces_warning(warn, match=msg):
        resampled = series.resample(freq)
    for (rk, rv), (gk, gv) in zip(resampled, grouped):
        assert rk == gk
        tm.assert_series_equal(rv, gv)


@all_ts
def test_resample_quantile(series):
    # GH 15023
    ser = series
    q = 0.75
    freq = "h"

    msg = "Resampling with a PeriodIndex"
    warn = None
    if isinstance(series.index, PeriodIndex):
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = ser.resample(freq).quantile(q)
        expected = ser.resample(freq).agg(lambda x: x.quantile(q)).rename(ser.name)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("how", ["first", "last"])
def test_first_last_skipna(any_real_nullable_dtype, skipna, how):
    # GH#57019
    if is_extension_array_dtype(any_real_nullable_dtype):
        na_value = Series(dtype=any_real_nullable_dtype).dtype.na_value
    else:
        na_value = np.nan
    df = DataFrame(
        {
            "a": [2, 1, 1, 2],
            "b": [na_value, 3.0, na_value, 4.0],
            "c": [na_value, 3.0, na_value, 4.0],
        },
        index=date_range("2020-01-01", periods=4, freq="D"),
        dtype=any_real_nullable_dtype,
    )
    rs = df.resample("ME")
    method = getattr(rs, how)
    result = method(skipna=skipna)

    gb = df.groupby(df.shape[0] * [pd.to_datetime("2020-01-31")])
    expected = getattr(gb, how)(skipna=skipna)
    expected.index.freq = "ME"
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\tests\test_defmatrix.py ===
import collections.abc

import numpy as np
from numpy import asmatrix, bmat, matrix
from numpy.linalg import matrix_power
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)


class TestCtor:
    def test_basic(self):
        A = np.array([[1, 2], [3, 4]])
        mA = matrix(A)
        assert_(np.all(mA.A == A))

        B = bmat("A,A;A,A")
        C = bmat([[A, A], [A, A]])
        D = np.array([[1, 2, 1, 2],
                      [3, 4, 3, 4],
                      [1, 2, 1, 2],
                      [3, 4, 3, 4]])
        assert_(np.all(B.A == D))
        assert_(np.all(C.A == D))

        E = np.array([[5, 6], [7, 8]])
        AEresult = matrix([[1, 2, 5, 6], [3, 4, 7, 8]])
        assert_(np.all(bmat([A, E]) == AEresult))

        vec = np.arange(5)
        mvec = matrix(vec)
        assert_(mvec.shape == (1, 5))

    def test_exceptions(self):
        # Check for ValueError when called with invalid string data.
        assert_raises(ValueError, matrix, "invalid")

    def test_bmat_nondefault_str(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        Aresult = np.array([[1, 2, 1, 2],
                            [3, 4, 3, 4],
                            [1, 2, 1, 2],
                            [3, 4, 3, 4]])
        mixresult = np.array([[1, 2, 5, 6],
                              [3, 4, 7, 8],
                              [5, 6, 1, 2],
                              [7, 8, 3, 4]])
        assert_(np.all(bmat("A,A;A,A") == Aresult))
        assert_(np.all(bmat("A,A;A,A", ldict={'A': B}) == Aresult))
        assert_raises(TypeError, bmat, "A,A;A,A", gdict={'A': B})
        assert_(
            np.all(bmat("A,A;A,A", ldict={'A': A}, gdict={'A': B}) == Aresult))
        b2 = bmat("A,B;C,D", ldict={'A': A, 'B': B}, gdict={'C': B, 'D': A})
        assert_(np.all(b2 == mixresult))


class TestProperties:
    def test_sum(self):
        """Test whether matrix.sum(axis=1) preserves orientation.
        Fails in NumPy <= 0.9.6.2127.
        """
        M = matrix([[1, 2, 0, 0],
                   [3, 4, 0, 0],
                   [1, 2, 1, 2],
                   [3, 4, 3, 4]])
        sum0 = matrix([8, 12, 4, 6])
        sum1 = matrix([3, 7, 6, 14]).T
        sumall = 30
        assert_array_equal(sum0, M.sum(axis=0))
        assert_array_equal(sum1, M.sum(axis=1))
        assert_equal(sumall, M.sum())

        assert_array_equal(sum0, np.sum(M, axis=0))
        assert_array_equal(sum1, np.sum(M, axis=1))
        assert_equal(sumall, np.sum(M))

    def test_prod(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.prod(), 720)
        assert_equal(x.prod(0), matrix([[4, 10, 18]]))
        assert_equal(x.prod(1), matrix([[6], [120]]))

        assert_equal(np.prod(x), 720)
        assert_equal(np.prod(x, axis=0), matrix([[4, 10, 18]]))
        assert_equal(np.prod(x, axis=1), matrix([[6], [120]]))

        y = matrix([0, 1, 3])
        assert_(y.prod() == 0)

    def test_max(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.max(), 6)
        assert_equal(x.max(0), matrix([[4, 5, 6]]))
        assert_equal(x.max(1), matrix([[3], [6]]))

        assert_equal(np.max(x), 6)
        assert_equal(np.max(x, axis=0), matrix([[4, 5, 6]]))
        assert_equal(np.max(x, axis=1), matrix([[3], [6]]))

    def test_min(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.min(), 1)
        assert_equal(x.min(0), matrix([[1, 2, 3]]))
        assert_equal(x.min(1), matrix([[1], [4]]))

        assert_equal(np.min(x), 1)
        assert_equal(np.min(x, axis=0), matrix([[1, 2, 3]]))
        assert_equal(np.min(x, axis=1), matrix([[1], [4]]))

    def test_ptp(self):
        x = np.arange(4).reshape((2, 2))
        mx = x.view(np.matrix)
        assert_(mx.ptp() == 3)
        assert_(np.all(mx.ptp(0) == np.array([2, 2])))
        assert_(np.all(mx.ptp(1) == np.array([1, 1])))

    def test_var(self):
        x = np.arange(9).reshape((3, 3))
        mx = x.view(np.matrix)
        assert_equal(x.var(ddof=0), mx.var(ddof=0))
        assert_equal(x.var(ddof=1), mx.var(ddof=1))

    def test_basic(self):
        import numpy.linalg as linalg

        A = np.array([[1., 2.],
                      [3., 4.]])
        mA = matrix(A)
        assert_(np.allclose(linalg.inv(A), mA.I))
        assert_(np.all(np.array(np.transpose(A) == mA.T)))
        assert_(np.all(np.array(np.transpose(A) == mA.H)))
        assert_(np.all(A == mA.A))

        B = A + 2j * A
        mB = matrix(B)
        assert_(np.allclose(linalg.inv(B), mB.I))
        assert_(np.all(np.array(np.transpose(B) == mB.T)))
        assert_(np.all(np.array(np.transpose(B).conj() == mB.H)))

    def test_pinv(self):
        x = matrix(np.arange(6).reshape(2, 3))
        xpinv = matrix([[-0.77777778,  0.27777778],
                        [-0.11111111,  0.11111111],
                        [ 0.55555556, -0.05555556]])
        assert_almost_equal(x.I, xpinv)

    def test_comparisons(self):
        A = np.arange(100).reshape(10, 10)
        mA = matrix(A)
        mB = matrix(A) + 0.1
        assert_(np.all(mB == A + 0.1))
        assert_(np.all(mB == matrix(A + 0.1)))
        assert_(not np.any(mB == matrix(A - 0.1)))
        assert_(np.all(mA < mB))
        assert_(np.all(mA <= mB))
        assert_(np.all(mA <= mA))
        assert_(not np.any(mA < mA))

        assert_(not np.any(mB < mA))
        assert_(np.all(mB >= mA))
        assert_(np.all(mB >= mB))
        assert_(not np.any(mB > mB))

        assert_(np.all(mA == mA))
        assert_(not np.any(mA == mB))
        assert_(np.all(mB != mA))

        assert_(not np.all(abs(mA) > 0))
        assert_(np.all(abs(mB > 0)))

    def test_asmatrix(self):
        A = np.arange(100).reshape(10, 10)
        mA = asmatrix(A)
        A[0, 0] = -10
        assert_(A[0, 0] == mA[0, 0])

    def test_noaxis(self):
        A = matrix([[1, 0], [0, 1]])
        assert_(A.sum() == matrix(2))
        assert_(A.mean() == matrix(0.5))

    def test_repr(self):
        A = matrix([[1, 0], [0, 1]])
        assert_(repr(A) == "matrix([[1, 0],\n        [0, 1]])")

    def test_make_bool_matrix_from_str(self):
        A = matrix('True; True; False')
        B = matrix([[True], [True], [False]])
        assert_array_equal(A, B)

class TestCasting:
    def test_basic(self):
        A = np.arange(100).reshape(10, 10)
        mA = matrix(A)

        mB = mA.copy()
        O = np.ones((10, 10), np.float64) * 0.1
        mB = mB + O
        assert_(mB.dtype.type == np.float64)
        assert_(np.all(mA != mB))
        assert_(np.all(mB == mA + 0.1))

        mC = mA.copy()
        O = np.ones((10, 10), np.complex128)
        mC = mC * O
        assert_(mC.dtype.type == np.complex128)
        assert_(np.all(mA != mB))


class TestAlgebra:
    def test_basic(self):
        import numpy.linalg as linalg

        A = np.array([[1., 2.], [3., 4.]])
        mA = matrix(A)

        B = np.identity(2)
        for i in range(6):
            assert_(np.allclose((mA ** i).A, B))
            B = np.dot(B, A)

        Ainv = linalg.inv(A)
        B = np.identity(2)
        for i in range(6):
            assert_(np.allclose((mA ** -i).A, B))
            B = np.dot(B, Ainv)

        assert_(np.allclose((mA * mA).A, np.dot(A, A)))
        assert_(np.allclose((mA + mA).A, (A + A)))
        assert_(np.allclose((3 * mA).A, (3 * A)))

        mA2 = matrix(A)
        mA2 *= 3
        assert_(np.allclose(mA2.A, 3 * A))

    def test_pow(self):
        """Test raising a matrix to an integer power works as expected."""
        m = matrix("1. 2.; 3. 4.")
        m2 = m.copy()
        m2 **= 2
        mi = m.copy()
        mi **= -1
        m4 = m2.copy()
        m4 **= 2
        assert_array_almost_equal(m2, m**2)
        assert_array_almost_equal(m4, np.dot(m2, m2))
        assert_array_almost_equal(np.dot(mi, m), np.eye(2))

    def test_scalar_type_pow(self):
        m = matrix([[1, 2], [3, 4]])
        for scalar_t in [np.int8, np.uint8]:
            two = scalar_t(2)
            assert_array_almost_equal(m ** 2, m ** two)

    def test_notimplemented(self):
        '''Check that 'not implemented' operations produce a failure.'''
        A = matrix([[1., 2.],
                    [3., 4.]])

        # __rpow__
        with assert_raises(TypeError):
            1.0**A

        # __mul__ with something not a list, ndarray, tuple, or scalar
        with assert_raises(TypeError):
            A * object()


class TestMatrixReturn:
    def test_instance_methods(self):
        a = matrix([1.0], dtype='f8')
        methodargs = {
            'astype': ('intc',),
            'clip': (0.0, 1.0),
            'compress': ([1],),
            'repeat': (1,),
            'reshape': (1,),
            'swapaxes': (0, 0),
            'dot': np.array([1.0]),
            }
        excluded_methods = [
            'argmin', 'choose', 'dump', 'dumps', 'fill', 'getfield',
            'getA', 'getA1', 'item', 'nonzero', 'put', 'putmask', 'resize',
            'searchsorted', 'setflags', 'setfield', 'sort',
            'partition', 'argpartition', 'newbyteorder', 'to_device',
            'take', 'tofile', 'tolist', 'tobytes', 'all', 'any',
            'sum', 'argmax', 'argmin', 'min', 'max', 'mean', 'var', 'ptp',
            'prod', 'std', 'ctypes', 'itemset', 'bitwise_count',
            ]
        for attrib in dir(a):
            if attrib.startswith('_') or attrib in excluded_methods:
                continue
            f = getattr(a, attrib)
            if isinstance(f, collections.abc.Callable):
                # reset contents of a
                a.astype('f8')
                a.fill(1.0)
                args = methodargs.get(attrib, ())
                b = f(*args)
                assert_(type(b) is matrix, f"{attrib}")
        assert_(type(a.real) is matrix)
        assert_(type(a.imag) is matrix)
        c, d = matrix([0.0]).nonzero()
        assert_(type(c) is np.ndarray)
        assert_(type(d) is np.ndarray)


class TestIndexing:
    def test_basic(self):
        x = asmatrix(np.zeros((3, 2), float))
        y = np.zeros((3, 1), float)
        y[:, 0] = [0.8, 0.2, 0.3]
        x[:, 1] = y > 0.5
        assert_equal(x, [[0, 1], [0, 0], [0, 0]])


class TestNewScalarIndexing:
    a = matrix([[1, 2], [3, 4]])

    def test_dimesions(self):
        a = self.a
        x = a[0]
        assert_equal(x.ndim, 2)

    def test_array_from_matrix_list(self):
        a = self.a
        x = np.array([a, a])
        assert_equal(x.shape, [2, 2, 2])

    def test_array_to_list(self):
        a = self.a
        assert_equal(a.tolist(), [[1, 2], [3, 4]])

    def test_fancy_indexing(self):
        a = self.a
        x = a[1, [0, 1, 0]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[3,  4,  3]]))
        x = a[[1, 0]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[3, 4], [1, 2]]))
        x = a[[[1], [0]], [[1, 0], [0, 1]]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[4, 3], [1, 2]]))

    def test_matrix_element(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x[0][0], matrix([[1, 2, 3]]))
        assert_equal(x[0][0].shape, (1, 3))
        assert_equal(x[0].shape, (1, 3))
        assert_equal(x[:, 0].shape, (2, 1))

        x = matrix(0)
        assert_equal(x[0, 0], 0)
        assert_equal(x[0], 0)
        assert_equal(x[:, 0].shape, x.shape)

    def test_scalar_indexing(self):
        x = asmatrix(np.zeros((3, 2), float))
        assert_equal(x[0, 0], x[0][0])

    def test_row_column_indexing(self):
        x = asmatrix(np.eye(2))
        assert_array_equal(x[0, :], [[1, 0]])
        assert_array_equal(x[1, :], [[0, 1]])
        assert_array_equal(x[:, 0], [[1], [0]])
        assert_array_equal(x[:, 1], [[0], [1]])

    def test_boolean_indexing(self):
        A = np.arange(6)
        A.shape = (3, 2)
        x = asmatrix(A)
        assert_array_equal(x[:, np.array([True, False])], x[:, 0])
        assert_array_equal(x[np.array([True, False, False]), :], x[0, :])

    def test_list_indexing(self):
        A = np.arange(6)
        A.shape = (3, 2)
        x = asmatrix(A)
        assert_array_equal(x[:, [1, 0]], x[:, ::-1])
        assert_array_equal(x[[2, 1, 0], :], x[::-1, :])


class TestPower:
    def test_returntype(self):
        a = np.array([[0, 1], [0, 0]])
        assert_(type(matrix_power(a, 2)) is np.ndarray)
        a = asmatrix(a)
        assert_(type(matrix_power(a, 2)) is matrix)

    def test_list(self):
        assert_array_equal(matrix_power([[0, 1], [0, 0]], 2), [[0, 0], [0, 0]])


class TestShape:

    a = np.array([[1], [2]])
    m = matrix([[1], [2]])

    def test_shape(self):
        assert_equal(self.a.shape, (2, 1))
        assert_equal(self.m.shape, (2, 1))

    def test_numpy_ravel(self):
        assert_equal(np.ravel(self.a).shape, (2,))
        assert_equal(np.ravel(self.m).shape, (2,))

    def test_member_ravel(self):
        assert_equal(self.a.ravel().shape, (2,))
        assert_equal(self.m.ravel().shape, (1, 2))

    def test_member_flatten(self):
        assert_equal(self.a.flatten().shape, (2,))
        assert_equal(self.m.flatten().shape, (1, 2))

    def test_numpy_ravel_order(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        assert_equal(np.ravel(x), [1, 2, 3, 4, 5, 6])
        assert_equal(np.ravel(x, order='F'), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T, order='A'), [1, 2, 3, 4, 5, 6])
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(np.ravel(x), [1, 2, 3, 4, 5, 6])
        assert_equal(np.ravel(x, order='F'), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T, order='A'), [1, 2, 3, 4, 5, 6])

    def test_matrix_ravel_order(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.ravel(), [[1, 2, 3, 4, 5, 6]])
        assert_equal(x.ravel(order='F'), [[1, 4, 2, 5, 3, 6]])
        assert_equal(x.T.ravel(), [[1, 4, 2, 5, 3, 6]])
        assert_equal(x.T.ravel(order='A'), [[1, 2, 3, 4, 5, 6]])

    def test_array_memory_sharing(self):
        assert_(np.may_share_memory(self.a, self.a.ravel()))
        assert_(not np.may_share_memory(self.a, self.a.flatten()))

    def test_matrix_memory_sharing(self):
        assert_(np.may_share_memory(self.m, self.m.ravel()))
        assert_(not np.may_share_memory(self.m, self.m.flatten()))

    def test_expand_dims_matrix(self):
        # matrices are always 2d - so expand_dims only makes sense when the
        # type is changed away from matrix.
        a = np.arange(10).reshape((2, 5)).view(np.matrix)
        expanded = np.expand_dims(a, axis=1)
        assert_equal(expanded.ndim, 3)
        assert_(not isinstance(expanded, np.matrix))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_deprecations.py ===
"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
import contextlib
import warnings

import numpy._core._struct_ufunc_tests as struct_ufunc
import pytest
from numpy._core._multiarray_tests import fromstring_null_term_c_api  # noqa: F401

import numpy as np
from numpy.testing import assert_raises, temppath

try:
    import pytz  # noqa: F401
    _has_pytz = True
except ImportError:
    _has_pytz = False


class _DeprecationTestCase:
    # Just as warning: warnings uses re.match, so the start of this message
    # must match.
    message = ''
    warning_cls = DeprecationWarning

    def setup_method(self):
        self.warn_ctx = warnings.catch_warnings(record=True)
        self.log = self.warn_ctx.__enter__()

        # Do *not* ignore other DeprecationWarnings. Ignoring warnings
        # can give very confusing results because of
        # https://bugs.python.org/issue4180 and it is probably simplest to
        # try to keep the tests cleanly giving only the right warning type.
        # (While checking them set to "error" those are ignored anyway)
        # We still have them show up, because otherwise they would be raised
        warnings.filterwarnings("always", category=self.warning_cls)
        warnings.filterwarnings("always", message=self.message,
                                category=self.warning_cls)

    def teardown_method(self):
        self.warn_ctx.__exit__()

    def assert_deprecated(self, function, num=1, ignore_others=False,
                          function_fails=False,
                          exceptions=np._NoValue,
                          args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.

        This first checks if the function when called gives `num`
        DeprecationWarnings, after that it tries to raise these
        DeprecationWarnings and compares them with `exceptions`.
        The exceptions can be different for cases where this code path
        is simply not anticipated and the exception is replaced.

        Parameters
        ----------
        function : callable
            The function to test
        num : int
            Number of DeprecationWarnings to expect. This should normally be 1.
        ignore_others : bool
            Whether warnings of the wrong type should be ignored (note that
            the message is not checked)
        function_fails : bool
            If the function would normally fail, setting this will check for
            warnings inside a try/except block.
        exceptions : Exception or tuple of Exceptions
            Exception to expect when turning the warnings into an error.
            The default checks for DeprecationWarnings. If exceptions is
            empty the function is expected to run successfully.
        args : tuple
            Arguments for `function`
        kwargs : dict
            Keyword arguments for `function`
        """
        __tracebackhide__ = True  # Hide traceback for py.test

        # reset the log
        self.log[:] = []

        if exceptions is np._NoValue:
            exceptions = (self.warning_cls,)

        if function_fails:
            context_manager = contextlib.suppress(Exception)
        else:
            context_manager = contextlib.nullcontext()
        with context_manager:
            function(*args, **kwargs)

        # just in case, clear the registry
        num_found = 0
        for warning in self.log:
            if warning.category is self.warning_cls:
                num_found += 1
            elif not ignore_others:
                raise AssertionError(
                        "expected %s but got: %s" %
                        (self.warning_cls.__name__, warning.category))
        if num is not None and num_found != num:
            msg = f"{len(self.log)} warnings found but {num} expected."
            lst = [str(w) for w in self.log]
            raise AssertionError("\n".join([msg] + lst))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=self.message,
                                    category=self.warning_cls)
            try:
                function(*args, **kwargs)
                if exceptions != ():
                    raise AssertionError(
                            "No error raised during function call")
            except exceptions:
                if exceptions == ():
                    raise AssertionError(
                            "Error raised during function call")

    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test that warnings are not raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=(), args=args, kwargs=kwargs)


class _VisibleDeprecationTestCase(_DeprecationTestCase):
    warning_cls = np.exceptions.VisibleDeprecationWarning


class TestTestDeprecated:
    def test_assert_deprecated(self):
        test_case_instance = _DeprecationTestCase()
        test_case_instance.setup_method()
        assert_raises(AssertionError,
                      test_case_instance.assert_deprecated,
                      lambda: None)

        def foo():
            warnings.warn("foo", category=DeprecationWarning, stacklevel=2)

        test_case_instance.assert_deprecated(foo)
        test_case_instance.teardown_method()


class TestBincount(_DeprecationTestCase):
    # 2024-07-29, 2.1.0
    @pytest.mark.parametrize('badlist', [[0.5, 1.2, 1.5],
                                         ['0', '1', '1']])
    def test_bincount_bad_list(self, badlist):
        self.assert_deprecated(lambda: np.bincount(badlist))


class TestGeneratorSum(_DeprecationTestCase):
    # 2018-02-25, 1.15.0
    def test_generator_sum(self):
        self.assert_deprecated(np.sum, args=((i for i in range(5)),))


class BuiltInRoundComplexDType(_DeprecationTestCase):
    # 2020-03-31 1.19.0
    deprecated_types = [np.csingle, np.cdouble, np.clongdouble]
    not_deprecated_types = [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    ]

    def test_deprecated(self):
        for scalar_type in self.deprecated_types:
            scalar = scalar_type(0)
            self.assert_deprecated(round, args=(scalar,))
            self.assert_deprecated(round, args=(scalar, 0))
            self.assert_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})

    def test_not_deprecated(self):
        for scalar_type in self.not_deprecated_types:
            scalar = scalar_type(0)
            self.assert_not_deprecated(round, args=(scalar,))
            self.assert_not_deprecated(round, args=(scalar, 0))
            self.assert_not_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})


class FlatteningConcatenateUnsafeCast(_DeprecationTestCase):
    # NumPy 1.20, 2020-09-03
    message = "concatenate with `axis=None` will use same-kind casting"

    def test_deprecated(self):
        self.assert_deprecated(np.concatenate,
                args=(([0.], [1.]),),
                kwargs={'axis': None, 'out': np.empty(2, dtype=np.int64)})

    def test_not_deprecated(self):
        self.assert_not_deprecated(np.concatenate,
                args=(([0.], [1.]),),
                kwargs={'axis': None, 'out': np.empty(2, dtype=np.int64),
                        'casting': "unsafe"})

        with assert_raises(TypeError):
            # Tests should notice if the deprecation warning is given first...
            np.concatenate(([0.], [1.]), out=np.empty(2, dtype=np.int64),
                           casting="same_kind")


class TestCtypesGetter(_DeprecationTestCase):
    # Deprecated 2021-05-18, Numpy 1.21.0
    warning_cls = DeprecationWarning
    ctypes = np.array([1]).ctypes

    @pytest.mark.parametrize(
        "name", ["get_data", "get_shape", "get_strides", "get_as_parameter"]
    )
    def test_deprecated(self, name: str) -> None:
        func = getattr(self.ctypes, name)
        self.assert_deprecated(func)

    @pytest.mark.parametrize(
        "name", ["data", "shape", "strides", "_as_parameter_"]
    )
    def test_not_deprecated(self, name: str) -> None:
        self.assert_not_deprecated(lambda: getattr(self.ctypes, name))


class TestMachAr(_DeprecationTestCase):
    # Deprecated 2022-11-22, NumPy 1.25
    warning_cls = DeprecationWarning

    def test_deprecated_module(self):
        self.assert_deprecated(lambda: np._core.MachAr)


class TestQuantileInterpolationDeprecation(_DeprecationTestCase):
    # Deprecated 2021-11-08, NumPy 1.22
    @pytest.mark.parametrize("func",
        [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_deprecated(self, func):
        self.assert_deprecated(
            lambda: func([0., 1.], 0., interpolation="linear"))
        self.assert_deprecated(
            lambda: func([0., 1.], 0., interpolation="nearest"))

    @pytest.mark.parametrize("func",
            [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_both_passed(self, func):
        with warnings.catch_warnings():
            # catch the DeprecationWarning so that it does not raise:
            warnings.simplefilter("always", DeprecationWarning)
            with pytest.raises(TypeError):
                func([0., 1.], 0., interpolation="nearest", method="nearest")


class TestScalarConversion(_DeprecationTestCase):
    # 2023-01-02, 1.25.0
    def test_float_conversion(self):
        self.assert_deprecated(float, args=(np.array([3.14]),))

    def test_behaviour(self):
        b = np.array([[3.14]])
        c = np.zeros(5)
        with pytest.warns(DeprecationWarning):
            c[0] = b


class TestPyIntConversion(_DeprecationTestCase):
    message = r".*stop allowing conversion of out-of-bound.*"

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_deprecated_scalar(self, dtype):
        dtype = np.dtype(dtype)
        info = np.iinfo(dtype)

        # Cover the most common creation paths (all end up in the
        # same place):
        def scalar(value, dtype):
            dtype.type(value)

        def assign(value, dtype):
            arr = np.array([0, 0, 0], dtype=dtype)
            arr[2] = value

        def create(value, dtype):
            np.array([value], dtype=dtype)

        for creation_func in [scalar, assign, create]:
            try:
                self.assert_deprecated(
                        lambda: creation_func(info.min - 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.

            try:
                self.assert_deprecated(
                        lambda: creation_func(info.max + 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.


@pytest.mark.parametrize("name", ["str", "bytes", "object"])
def test_future_scalar_attributes(name):
    # FutureWarning added 2022-11-17, NumPy 1.24,
    assert name not in dir(np)  # we may want to not add them
    with pytest.warns(FutureWarning,
            match=f"In the future .*{name}"):
        assert not hasattr(np, name)

    # Unfortunately, they are currently still valid via `np.dtype()`
    np.dtype(name)
    name in np._core.sctypeDict


# Ignore the above future attribute warning for this test.
@pytest.mark.filterwarnings("ignore:In the future:FutureWarning")
class TestRemovedGlobals:
    # Removed 2023-01-12, NumPy 1.24.0
    # Not a deprecation, but the large error was added to aid those who missed
    # the previous deprecation, and should be removed similarly to one
    # (or faster).
    @pytest.mark.parametrize("name",
            ["object", "float", "complex", "str", "int"])
    def test_attributeerror_includes_info(self, name):
        msg = f".*\n`np.{name}` was a deprecated alias for the builtin"
        with pytest.raises(AttributeError, match=msg):
            getattr(np, name)


class TestDeprecatedFinfo(_DeprecationTestCase):
    # Deprecated in NumPy 1.25, 2023-01-16
    def test_deprecated_none(self):
        self.assert_deprecated(np.finfo, args=(None,))


class TestMathAlias(_DeprecationTestCase):
    def test_deprecated_np_lib_math(self):
        self.assert_deprecated(lambda: np.lib.math)


class TestLibImports(_DeprecationTestCase):
    # Deprecated in Numpy 1.26.0, 2023-09
    def test_lib_functions_deprecation_call(self):
        from numpy import in1d, row_stack, trapz
        from numpy._core.numerictypes import maximum_sctype
        from numpy.lib._function_base_impl import disp
        from numpy.lib._npyio_impl import recfromcsv, recfromtxt
        from numpy.lib._shape_base_impl import get_array_wrap
        from numpy.lib._utils_impl import safe_eval
        from numpy.lib.tests.test_io import TextIO

        self.assert_deprecated(lambda: safe_eval("None"))

        data_gen = lambda: TextIO('A,B\n0,1\n2,3')
        kwargs = {'delimiter': ",", 'missing_values': "N/A", 'names': True}
        self.assert_deprecated(lambda: recfromcsv(data_gen()))
        self.assert_deprecated(lambda: recfromtxt(data_gen(), **kwargs))

        self.assert_deprecated(lambda: disp("test"))
        self.assert_deprecated(get_array_wrap)
        self.assert_deprecated(lambda: maximum_sctype(int))

        self.assert_deprecated(lambda: in1d([1], [1]))
        self.assert_deprecated(lambda: row_stack([[]]))
        self.assert_deprecated(lambda: trapz([1], [1]))
        self.assert_deprecated(lambda: np.chararray)


class TestDeprecatedDTypeAliases(_DeprecationTestCase):

    def _check_for_warning(self, func):
        with warnings.catch_warnings(record=True) as caught_warnings:
            func()
        assert len(caught_warnings) == 1
        w = caught_warnings[0]
        assert w.category is DeprecationWarning
        assert "alias 'a' was deprecated in NumPy 2.0" in str(w.message)

    def test_a_dtype_alias(self):
        for dtype in ["a", "a10"]:
            f = lambda: np.dtype(dtype)
            self._check_for_warning(f)
            self.assert_deprecated(f)
            f = lambda: np.array(["hello", "world"]).astype("a10")
            self._check_for_warning(f)
            self.assert_deprecated(f)


class TestDeprecatedArrayWrap(_DeprecationTestCase):
    message = "__array_wrap__.*"

    def test_deprecated(self):
        class Test1:
            def __array__(self, dtype=None, copy=None):
                return np.arange(4)

            def __array_wrap__(self, arr, context=None):
                self.called = True
                return 'pass context'

        class Test2(Test1):
            def __array_wrap__(self, arr):
                self.called = True
                return 'pass'

        test1 = Test1()
        test2 = Test2()
        self.assert_deprecated(lambda: np.negative(test1))
        assert test1.called
        self.assert_deprecated(lambda: np.negative(test2))
        assert test2.called


class TestDeprecatedDTypeParenthesizedRepeatCount(_DeprecationTestCase):
    message = "Passing in a parenthesized single number"

    @pytest.mark.parametrize("string", ["(2)i,", "(3)3S,", "f,(2)f"])
    def test_parenthesized_repeat_count(self, string):
        self.assert_deprecated(np.dtype, args=(string,))


class TestDeprecatedSaveFixImports(_DeprecationTestCase):
    # Deprecated in Numpy 2.1, 2024-05
    message = "The 'fix_imports' flag is deprecated and has no effect."

    def test_deprecated(self):
        with temppath(suffix='.npy') as path:
            sample_args = (path, np.array(np.zeros((1024, 10))))
            self.assert_not_deprecated(np.save, args=sample_args)
            self.assert_deprecated(np.save, args=sample_args,
                                kwargs={'fix_imports': True})
            self.assert_deprecated(np.save, args=sample_args,
                                kwargs={'fix_imports': False})
            for allow_pickle in [True, False]:
                self.assert_not_deprecated(np.save, args=sample_args,
                                        kwargs={'allow_pickle': allow_pickle})
                self.assert_deprecated(np.save, args=sample_args,
                                    kwargs={'allow_pickle': allow_pickle,
                                            'fix_imports': True})
                self.assert_deprecated(np.save, args=sample_args,
                                    kwargs={'allow_pickle': allow_pickle,
                                            'fix_imports': False})


class TestAddNewdocUFunc(_DeprecationTestCase):
    # Deprecated in Numpy 2.2, 2024-11
    def test_deprecated(self):
        self.assert_deprecated(
            lambda: np._core.umath._add_newdoc_ufunc(
                struct_ufunc.add_triplet, "new docs"
            )
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_interval.py ===
from mpmath import *

def test_interval_identity():
    iv.dps = 15
    assert mpi(2) == mpi(2, 2)
    assert mpi(2) != mpi(-2, 2)
    assert not (mpi(2) != mpi(2, 2))
    assert mpi(-1, 1) == mpi(-1, 1)
    assert str(mpi('0.1')) == "[0.099999999999999991673, 0.10000000000000000555]"
    assert repr(mpi('0.1')) == "mpi('0.099999999999999992', '0.10000000000000001')"
    u = mpi(-1, 3)
    assert -1 in u
    assert 2 in u
    assert 3 in u
    assert -1.1 not in u
    assert 3.1 not in u
    assert mpi(-1, 3) in u
    assert mpi(0, 1) in u
    assert mpi(-1.1, 2) not in u
    assert mpi(2.5, 3.1) not in u
    w = mpi(-inf, inf)
    assert mpi(-5, 5) in w
    assert mpi(2, inf) in w
    assert mpi(0, 2) in mpi(0, 10)
    assert not (3 in mpi(-inf, 0))

def test_interval_hash():
    assert hash(mpi(3)) == hash(3)
    assert hash(mpi(3.25)) == hash(3.25)
    assert hash(mpi(3,4)) == hash(mpi(3,4))
    assert hash(iv.mpc(3)) == hash(3)
    assert hash(iv.mpc(3,4)) == hash(3+4j)
    assert hash(iv.mpc((1,3),(2,4))) == hash(iv.mpc((1,3),(2,4)))

def test_interval_arithmetic():
    iv.dps = 15
    assert mpi(2) + mpi(3,4) == mpi(5,6)
    assert mpi(1, 2)**2 == mpi(1, 4)
    assert mpi(1) + mpi(0, 1e-50) == mpi(1, mpf('1.0000000000000002'))
    x = 1 / (1 / mpi(3))
    assert x.a < 3 < x.b
    x = mpi(2) ** mpi(0.5)
    iv.dps += 5
    sq = iv.sqrt(2)
    iv.dps -= 5
    assert x.a < sq < x.b
    assert mpi(1) / mpi(1, inf)
    assert mpi(2, 3) / inf == mpi(0, 0)
    assert mpi(0) / inf == 0
    assert mpi(0) / 0 == mpi(-inf, inf)
    assert mpi(inf) / 0 == mpi(-inf, inf)
    assert mpi(0) * inf == mpi(-inf, inf)
    assert 1 / mpi(2, inf) == mpi(0, 0.5)
    assert str((mpi(50, 50) * mpi(-10, -10)) / 3) == \
        '[-166.66666666666668561, -166.66666666666665719]'
    assert mpi(0, 4) ** 3 == mpi(0, 64)
    assert mpi(2,4).mid == 3
    iv.dps = 30
    a = mpi(iv.pi)
    iv.dps = 15
    b = +a
    assert b.a < a.a
    assert b.b > a.b
    a = mpi(iv.pi)
    assert a == +a
    assert abs(mpi(-1,2)) == mpi(0,2)
    assert abs(mpi(0.5,2)) == mpi(0.5,2)
    assert abs(mpi(-3,2)) == mpi(0,3)
    assert abs(mpi(-3,-0.5)) == mpi(0.5,3)
    assert mpi(0) * mpi(2,3) == mpi(0)
    assert mpi(2,3) * mpi(0) == mpi(0)
    assert mpi(1,3).delta == 2
    assert mpi(1,2) - mpi(3,4) == mpi(-3,-1)
    assert mpi(-inf,0) - mpi(0,inf) == mpi(-inf,0)
    assert mpi(-inf,0) - mpi(-inf,inf) == mpi(-inf,inf)
    assert mpi(0,inf) - mpi(-inf,1) == mpi(-1,inf)

def test_interval_mul():
    assert mpi(-1, 0) * inf == mpi(-inf, 0)
    assert mpi(-1, 0) * -inf == mpi(0, inf)
    assert mpi(0, 1) * inf == mpi(0, inf)
    assert mpi(0, 1) * mpi(0, inf) == mpi(0, inf)
    assert mpi(-1, 1) * inf == mpi(-inf, inf)
    assert mpi(-1, 1) * mpi(0, inf) == mpi(-inf, inf)
    assert mpi(-1, 1) * mpi(-inf, inf) == mpi(-inf, inf)
    assert mpi(-inf, 0) * mpi(0, 1) == mpi(-inf, 0)
    assert mpi(-inf, 0) * mpi(0, 0) * mpi(-inf, 0)
    assert mpi(-inf, 0) * mpi(-inf, inf) == mpi(-inf, inf)
    assert mpi(-5,0)*mpi(-32,28) == mpi(-140,160)
    assert mpi(2,3) * mpi(-1,2) == mpi(-3,6)
    # Should be undefined?
    assert mpi(inf, inf) * 0 == mpi(-inf, inf)
    assert mpi(-inf, -inf) * 0 == mpi(-inf, inf)
    assert mpi(0) * mpi(-inf,2) == mpi(-inf,inf)
    assert mpi(0) * mpi(-2,inf) == mpi(-inf,inf)
    assert mpi(-2,inf) * mpi(0) == mpi(-inf,inf)
    assert mpi(-inf,2) * mpi(0) == mpi(-inf,inf)

def test_interval_pow():
    assert mpi(3)**2 == mpi(9, 9)
    assert mpi(-3)**2 == mpi(9, 9)
    assert mpi(-3, 1)**2 == mpi(0, 9)
    assert mpi(-3, -1)**2 == mpi(1, 9)
    assert mpi(-3, -1)**3 == mpi(-27, -1)
    assert mpi(-3, 1)**3 == mpi(-27, 1)
    assert mpi(-2, 3)**2 == mpi(0, 9)
    assert mpi(-3, 2)**2 == mpi(0, 9)
    assert mpi(4) ** -1 == mpi(0.25, 0.25)
    assert mpi(-4) ** -1 == mpi(-0.25, -0.25)
    assert mpi(4) ** -2 == mpi(0.0625, 0.0625)
    assert mpi(-4) ** -2 == mpi(0.0625, 0.0625)
    assert mpi(0, 1) ** inf == mpi(0, 1)
    assert mpi(0, 1) ** -inf == mpi(1, inf)
    assert mpi(0, inf) ** inf == mpi(0, inf)
    assert mpi(0, inf) ** -inf == mpi(0, inf)
    assert mpi(1, inf) ** inf == mpi(1, inf)
    assert mpi(1, inf) ** -inf == mpi(0, 1)
    assert mpi(2, 3) ** 1 == mpi(2, 3)
    assert mpi(2, 3) ** 0 == 1
    assert mpi(1,3) ** mpi(2) == mpi(1,9)

def test_interval_sqrt():
    assert mpi(4) ** 0.5 == mpi(2)

def test_interval_div():
    assert mpi(0.5, 1) / mpi(-1, 0) == mpi(-inf, -0.5)
    assert mpi(0, 1) / mpi(0, 1) == mpi(0, inf)
    assert mpi(inf, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(inf, inf) / mpi(2, inf) == mpi(0, inf)
    assert mpi(inf, inf) / mpi(2, 2) == mpi(inf, inf)
    assert mpi(0, inf) / mpi(2, inf) == mpi(0, inf)
    assert mpi(0, inf) / mpi(2, 2) == mpi(0, inf)
    assert mpi(2, inf) / mpi(2, 2) == mpi(1, inf)
    assert mpi(2, inf) / mpi(2, inf) == mpi(0, inf)
    assert mpi(-4, 8) / mpi(1, inf) == mpi(-4, 8)
    assert mpi(-4, 8) / mpi(0.5, inf) == mpi(-8, 16)
    assert mpi(-inf, 8) / mpi(0.5, inf) == mpi(-inf, 16)
    assert mpi(-inf, inf) / mpi(0.5, inf) == mpi(-inf, inf)
    assert mpi(8, inf) / mpi(0.5, inf) == mpi(0, inf)
    assert mpi(-8, inf) / mpi(0.5, inf) == mpi(-16, inf)
    assert mpi(-4, 8) / mpi(inf, inf) == mpi(0, 0)
    assert mpi(0, 8) / mpi(inf, inf) == mpi(0, 0)
    assert mpi(0, 0) / mpi(inf, inf) == mpi(0, 0)
    assert mpi(-inf, 0) / mpi(inf, inf) == mpi(-inf, 0)
    assert mpi(-inf, 8) / mpi(inf, inf) == mpi(-inf, 0)
    assert mpi(-inf, inf) / mpi(inf, inf) == mpi(-inf, inf)
    assert mpi(-8, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(0, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(8, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(inf, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(-1, 2) / mpi(0, 1) == mpi(-inf, +inf)
    assert mpi(0, 1) / mpi(0, 1) == mpi(0.0, +inf)
    assert mpi(-1, 0) / mpi(0, 1) == mpi(-inf, 0.0)
    assert mpi(-0.5, -0.25) / mpi(0, 1) == mpi(-inf, -0.25)
    assert mpi(0.5, 1) / mpi(0, 1) == mpi(0.5, +inf)
    assert mpi(0.5, 4) / mpi(0, 1) == mpi(0.5, +inf)
    assert mpi(-1, -0.5) / mpi(0, 1) == mpi(-inf, -0.5)
    assert mpi(-4, -0.5) / mpi(0, 1) == mpi(-inf, -0.5)
    assert mpi(-1, 2) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(0, 1) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-1, 0) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-0.5, -0.25) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(0.5, 1) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(0.5, 4) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-1, -0.5) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-4, -0.5) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-1, 2) / mpi(-1, 0) == mpi(-inf, +inf)
    assert mpi(0, 1) / mpi(-1, 0) == mpi(-inf, 0.0)
    assert mpi(-1, 0) / mpi(-1, 0) == mpi(0.0, +inf)
    assert mpi(-0.5, -0.25) / mpi(-1, 0) == mpi(0.25, +inf)
    assert mpi(0.5, 1) / mpi(-1, 0) == mpi(-inf, -0.5)
    assert mpi(0.5, 4) / mpi(-1, 0) == mpi(-inf, -0.5)
    assert mpi(-1, -0.5) / mpi(-1, 0) == mpi(0.5, +inf)
    assert mpi(-4, -0.5) / mpi(-1, 0) == mpi(0.5, +inf)
    assert mpi(-1, 2) / mpi(0.5, 1) == mpi(-2.0, 4.0)
    assert mpi(0, 1) / mpi(0.5, 1) == mpi(0.0, 2.0)
    assert mpi(-1, 0) / mpi(0.5, 1) == mpi(-2.0, 0.0)
    assert mpi(-0.5, -0.25) / mpi(0.5, 1) == mpi(-1.0, -0.25)
    assert mpi(0.5, 1) / mpi(0.5, 1) == mpi(0.5, 2.0)
    assert mpi(0.5, 4) / mpi(0.5, 1) == mpi(0.5, 8.0)
    assert mpi(-1, -0.5) / mpi(0.5, 1) == mpi(-2.0, -0.5)
    assert mpi(-4, -0.5) / mpi(0.5, 1) == mpi(-8.0, -0.5)
    assert mpi(-1, 2) / mpi(-2, -0.5) == mpi(-4.0, 2.0)
    assert mpi(0, 1) / mpi(-2, -0.5) == mpi(-2.0, 0.0)
    assert mpi(-1, 0) / mpi(-2, -0.5) == mpi(0.0, 2.0)
    assert mpi(-0.5, -0.25) / mpi(-2, -0.5) == mpi(0.125, 1.0)
    assert mpi(0.5, 1) / mpi(-2, -0.5) == mpi(-2.0, -0.25)
    assert mpi(0.5, 4) / mpi(-2, -0.5) == mpi(-8.0, -0.25)
    assert mpi(-1, -0.5) / mpi(-2, -0.5) == mpi(0.25, 2.0)
    assert mpi(-4, -0.5) / mpi(-2, -0.5) == mpi(0.25, 8.0)
    # Should be undefined?
    assert mpi(0, 0) / mpi(0, 0) == mpi(-inf, inf)
    assert mpi(0, 0) / mpi(0, 1) == mpi(-inf, inf)

def test_interval_cos_sin():
    iv.dps = 15
    cos = iv.cos
    sin = iv.sin
    tan = iv.tan
    pi = iv.pi
    # Around 0
    assert cos(mpi(0)) == 1
    assert sin(mpi(0)) == 0
    assert cos(mpi(0,1)) == mpi(0.54030230586813965399, 1.0)
    assert sin(mpi(0,1)) == mpi(0, 0.8414709848078966159)
    assert cos(mpi(1,2)) == mpi(-0.4161468365471424069, 0.54030230586813976501)
    assert sin(mpi(1,2)) == mpi(0.84147098480789650488, 1.0)
    assert sin(mpi(1,2.5)) == mpi(0.59847214410395643824, 1.0)
    assert cos(mpi(-1, 1)) == mpi(0.54030230586813965399, 1.0)
    assert cos(mpi(-1, 0.5)) == mpi(0.54030230586813965399, 1.0)
    assert cos(mpi(-1, 1.5)) == mpi(0.070737201667702906405, 1.0)
    assert sin(mpi(-1,1)) == mpi(-0.8414709848078966159, 0.8414709848078966159)
    assert sin(mpi(-1,0.5)) == mpi(-0.8414709848078966159, 0.47942553860420300538)
    assert mpi(-0.8414709848078966159, 1.00000000000000002e-100) in sin(mpi(-1,1e-100))
    assert mpi(-2.00000000000000004e-100, 1.00000000000000002e-100) in sin(mpi(-2e-100,1e-100))
    # Same interval
    assert cos(mpi(2, 2.5))
    assert cos(mpi(3.5, 4)) == mpi(-0.93645668729079634129, -0.65364362086361182946)
    assert cos(mpi(5, 5.5)) == mpi(0.28366218546322624627, 0.70866977429126010168)
    assert mpi(0.59847214410395654927, 0.90929742682568170942) in sin(mpi(2, 2.5))
    assert sin(mpi(3.5, 4)) == mpi(-0.75680249530792831347, -0.35078322768961983646)
    assert sin(mpi(5, 5.5)) == mpi(-0.95892427466313856499, -0.70554032557039181306)
    # Higher roots
    iv.dps = 55
    w = 4*10**50 + mpi(0.5)
    for p in [15, 40, 80]:
        iv.dps = p
        assert 0 in sin(4*mpi(pi))
        assert 0 in sin(4*10**50*mpi(pi))
        assert 0 in cos((4+0.5)*mpi(pi))
        assert 0 in cos(w*mpi(pi))
        assert 1 in cos(4*mpi(pi))
        assert 1 in cos(4*10**50*mpi(pi))
    iv.dps = 15
    assert cos(mpi(2,inf)) == mpi(-1,1)
    assert sin(mpi(2,inf)) == mpi(-1,1)
    assert cos(mpi(-inf,2)) == mpi(-1,1)
    assert sin(mpi(-inf,2)) == mpi(-1,1)
    u = tan(mpi(0.5,1))
    assert mpf(u.a).ae(mp.tan(0.5))
    assert mpf(u.b).ae(mp.tan(1))
    v = iv.cot(mpi(0.5,1))
    assert mpf(v.a).ae(mp.cot(1))
    assert mpf(v.b).ae(mp.cot(0.5))
    # Sanity check of evaluation at n*pi and (n+1/2)*pi
    for n in range(-5,7,2):
        x = iv.cos(n*iv.pi)
        assert -1 in x
        assert x >= -1
        assert x != -1
        x = iv.sin((n+0.5)*iv.pi)
        assert -1 in x
        assert x >= -1
        assert x != -1
    for n in range(-6,8,2):
        x = iv.cos(n*iv.pi)
        assert 1 in x
        assert x <= 1
        if n:
            assert x != 1
        x = iv.sin((n+0.5)*iv.pi)
        assert 1 in x
        assert x <= 1
        assert x != 1
    for n in range(-6,7):
        x = iv.cos((n+0.5)*iv.pi)
        assert x.a < 0 < x.b
        x = iv.sin(n*iv.pi)
        if n:
            assert x.a < 0 < x.b

def test_interval_complex():
    # TODO: many more tests
    iv.dps = 15
    mp.dps = 15
    assert iv.mpc(2,3) == 2+3j
    assert iv.mpc(2,3) != 2+4j
    assert iv.mpc(2,3) != 1+3j
    assert 1+3j in iv.mpc([1,2],[3,4])
    assert 2+5j not in iv.mpc([1,2],[3,4])
    assert iv.mpc(1,2) + 1j == 1+3j
    assert iv.mpc([1,2],[2,3]) + 2+3j == iv.mpc([3,4],[5,6])
    assert iv.mpc([2,4],[4,8]) / 2 == iv.mpc([1,2],[2,4])
    assert iv.mpc([1,2],[2,4]) * 2j == iv.mpc([-8,-4],[2,4])
    assert iv.mpc([2,4],[4,8]) / 2j == iv.mpc([2,4],[-2,-1])
    assert iv.exp(2+3j).ae(mp.exp(2+3j))
    assert iv.log(2+3j).ae(mp.log(2+3j))
    assert (iv.mpc(2,3) ** iv.mpc(0.5,2)).ae(mp.mpc(2,3) ** mp.mpc(0.5,2))
    assert 1j in (iv.mpf(-1) ** 0.5)
    assert 1j in (iv.mpc(-1) ** 0.5)
    assert abs(iv.mpc(0)) == 0
    assert abs(iv.mpc(inf)) == inf
    assert abs(iv.mpc(3,4)) == 5
    assert abs(iv.mpc(4)) == 4
    assert abs(iv.mpc(0,4)) == 4
    assert abs(iv.mpc(0,[2,3])) == iv.mpf([2,3])
    assert abs(iv.mpc(0,[-3,2])) == iv.mpf([0,3])
    assert abs(iv.mpc([3,5],[4,12])) == iv.mpf([5,13])
    assert abs(iv.mpc([3,5],[-4,12])) == iv.mpf([3,13])
    assert iv.mpc(2,3) ** 0 == 1
    assert iv.mpc(2,3) ** 1 == (2+3j)
    assert iv.mpc(2,3) ** 2 == (2+3j)**2
    assert iv.mpc(2,3) ** 3 == (2+3j)**3
    assert iv.mpc(2,3) ** 4 == (2+3j)**4
    assert iv.mpc(2,3) ** 5 == (2+3j)**5
    assert iv.mpc(2,2) ** (-1) == (2+2j) ** (-1)
    assert iv.mpc(2,2) ** (-2) == (2+2j) ** (-2)
    assert iv.cos(2).ae(mp.cos(2))
    assert iv.sin(2).ae(mp.sin(2))
    assert iv.cos(2+3j).ae(mp.cos(2+3j))
    assert iv.sin(2+3j).ae(mp.sin(2+3j))

def test_interval_complex_arg():
    mp.dps = 15
    iv.dps = 15
    assert iv.arg(3) == 0
    assert iv.arg(0) == 0
    assert iv.arg([0,3]) == 0
    assert iv.arg(-3).ae(pi)
    assert iv.arg(2+3j).ae(iv.arg(2+3j))
    z = iv.mpc([-2,-1],[3,4])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(-1+4j))
    assert t.b.ae(mp.arg(-2+3j))
    z = iv.mpc([-2,1],[3,4])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(1+3j))
    assert t.b.ae(mp.arg(-2+3j))
    z = iv.mpc([1,2],[3,4])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(2+3j))
    assert t.b.ae(mp.arg(1+4j))
    z = iv.mpc([1,2],[-2,3])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(1-2j))
    assert t.b.ae(mp.arg(1+3j))
    z = iv.mpc([1,2],[-4,-3])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(1-4j))
    assert t.b.ae(mp.arg(2-3j))
    z = iv.mpc([-1,2],[-4,-3])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(-1-3j))
    assert t.b.ae(mp.arg(2-3j))
    z = iv.mpc([-2,-1],[-4,-3])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(-2-3j))
    assert t.b.ae(mp.arg(-1-4j))
    z = iv.mpc([-2,-1],[-3,3])
    t = iv.arg(z)
    assert t.a.ae(-mp.pi)
    assert t.b.ae(mp.pi)
    z = iv.mpc([-2,2],[-3,3])
    t = iv.arg(z)
    assert t.a.ae(-mp.pi)
    assert t.b.ae(mp.pi)

def test_interval_ae():
    iv.dps = 15
    x = iv.mpf([1,2])
    assert x.ae(1) is None
    assert x.ae(1.5) is None
    assert x.ae(2) is None
    assert x.ae(2.01) is False
    assert x.ae(0.99) is False
    x = iv.mpf(3.5)
    assert x.ae(3.5) is True
    assert x.ae(3.5+1e-15) is True
    assert x.ae(3.5-1e-15) is True
    assert x.ae(3.501) is False
    assert x.ae(3.499) is False
    assert x.ae(iv.mpf([3.5,3.501])) is None
    assert x.ae(iv.mpf([3.5,4.5+1e-15])) is None

def test_interval_nstr():
    iv.dps = n = 30
    x = mpi(1, 2)
    # FIXME: error_dps should not be necessary
    assert iv.nstr(x, n, mode='plusminus', error_dps=6) == '1.5 +- 0.5'
    assert iv.nstr(x, n, mode='plusminus', use_spaces=False, error_dps=6) == '1.5+-0.5'
    assert iv.nstr(x, n, mode='percent') == '1.5 (33.33%)'
    assert iv.nstr(x, n, mode='brackets', use_spaces=False) == '[1.0,2.0]'
    assert iv.nstr(x, n, mode='brackets' , brackets=('<', '>')) == '<1.0, 2.0>'
    x = mpi('5.2582327113062393041', '5.2582327113062749951')
    assert iv.nstr(x, n, mode='diff') == '5.2582327113062[393041, 749951]'
    assert iv.nstr(iv.cos(mpi(1)), n, mode='diff', use_spaces=False) == '0.54030230586813971740093660744[2955,3053]'
    assert iv.nstr(mpi('1e123', '1e129'), n, mode='diff') == '[1.0e+123, 1.0e+129]'
    exp = iv.exp
    assert iv.nstr(iv.exp(mpi('5000.1')), n, mode='diff') == '3.2797365856787867069110487[0926, 1191]e+2171'
    iv.dps = 15

def test_mpi_from_str():
    iv.dps = 15
    assert iv.convert('1.5 +- 0.5') == mpi(mpf('1.0'), mpf('2.0'))
    assert mpi(1, 2) in iv.convert('1.5 (33.33333333333333333333333333333%)')
    assert iv.convert('[1, 2]') == mpi(1, 2)
    assert iv.convert('1[2, 3]') == mpi(12, 13)
    assert iv.convert('1.[23,46]e-8') == mpi('1.23e-8', '1.46e-8')
    assert iv.convert('12[3.4,5.9]e4') == mpi('123.4e+4', '125.9e4')

def test_interval_gamma():
    mp.dps = 15
    iv.dps = 15
    # TODO: need many more tests
    assert iv.rgamma(0) == 0
    assert iv.fac(0) == 1
    assert iv.fac(1) == 1
    assert iv.fac(2) == 2
    assert iv.fac(3) == 6
    assert iv.gamma(0) == [-inf,inf]
    assert iv.gamma(1) == 1
    assert iv.gamma(2) == 1
    assert iv.gamma(3) == 2
    assert -3.5449077018110320546 in iv.gamma(-0.5)
    assert iv.loggamma(1) == 0
    assert iv.loggamma(2) == 0
    assert 0.69314718055994530942 in iv.loggamma(3)
    # Test tight log-gamma endpoints based on monotonicity
    xs = [iv.mpc([2,3],[1,4]),
          iv.mpc([2,3],[-4,-1]),
          iv.mpc([2,3],[-1,4]),
          iv.mpc([2,3],[-4,1]),
          iv.mpc([2,3],[-4,4]),
          iv.mpc([-3,-2],[2,4]),
          iv.mpc([-3,-2],[-4,-2])]
    for x in xs:
        ys = [mp.loggamma(mp.mpc(x.a,x.c)),
              mp.loggamma(mp.mpc(x.b,x.c)),
              mp.loggamma(mp.mpc(x.a,x.d)),
              mp.loggamma(mp.mpc(x.b,x.d))]
        if 0 in x.imag:
            ys += [mp.loggamma(x.a), mp.loggamma(x.b)]
        min_real = min([y.real for y in ys])
        max_real = max([y.real for y in ys])
        min_imag = min([y.imag for y in ys])
        max_imag = max([y.imag for y in ys])
        z = iv.loggamma(x)
        assert z.a.ae(min_real)
        assert z.b.ae(max_real)
        assert z.c.ae(min_imag)
        assert z.d.ae(max_imag)

def test_interval_conversions():
    mp.dps = 15
    iv.dps = 15
    for a, b in ((-0.0, 0), (0.0, 0.5), (1.0, 1), \
                 ('-inf', 20.5), ('-inf', float(sqrt(2)))):
        r = mpi(a, b)
        assert int(r.b) == int(b)
        assert float(r.a) == float(a)
        assert float(r.b) == float(b)
        assert complex(r.a) == complex(a)
        assert complex(r.b) == complex(b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_mem_policy.py ===
import asyncio
import gc
import os
import sys
import sysconfig
import threading

import pytest

import numpy as np
from numpy._core.multiarray import get_handler_name
from numpy.testing import IS_EDITABLE, IS_WASM, assert_warns, extbuild


@pytest.fixture
def get_module(tmp_path):
    """ Add a memory policy that returns a false pointer 64 bytes into the
    actual allocation, and fill the prefix with some text. Then check at each
    memory manipulation that the prefix exists, to make sure all alloc/realloc/
    free/calloc go via the functions here.
    """
    if sys.platform.startswith('cygwin'):
        pytest.skip('link fails on cygwin')
    if IS_WASM:
        pytest.skip("Can't build module inside Wasm")
    if IS_EDITABLE:
        pytest.skip("Can't build module for editable install")

    functions = [
        ("get_default_policy", "METH_NOARGS", """
             Py_INCREF(PyDataMem_DefaultHandler);
             return PyDataMem_DefaultHandler;
         """),
        ("set_secret_data_policy", "METH_NOARGS", """
             PyObject *secret_data =
                 PyCapsule_New(&secret_data_handler, "mem_handler", NULL);
             if (secret_data == NULL) {
                 return NULL;
             }
             PyObject *old = PyDataMem_SetHandler(secret_data);
             Py_DECREF(secret_data);
             return old;
         """),
        ("set_wrong_capsule_name_data_policy", "METH_NOARGS", """
             PyObject *wrong_name_capsule =
                 PyCapsule_New(&secret_data_handler, "not_mem_handler", NULL);
             if (wrong_name_capsule == NULL) {
                 return NULL;
             }
             PyObject *old = PyDataMem_SetHandler(wrong_name_capsule);
             Py_DECREF(wrong_name_capsule);
             return old;
         """),
        ("set_old_policy", "METH_O", """
             PyObject *old;
             if (args != NULL && PyCapsule_CheckExact(args)) {
                 old = PyDataMem_SetHandler(args);
             }
             else {
                 old = PyDataMem_SetHandler(NULL);
             }
             return old;
         """),
        ("get_array", "METH_NOARGS", """
            char *buf = (char *)malloc(20);
            npy_intp dims[1];
            dims[0] = 20;
            PyArray_Descr *descr =  PyArray_DescrNewFromType(NPY_UINT8);
            return PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims, NULL,
                                        buf, NPY_ARRAY_WRITEABLE, NULL);
         """),
        ("set_own", "METH_O", """
            if (!PyArray_Check(args)) {
                PyErr_SetString(PyExc_ValueError,
                             "need an ndarray");
                return NULL;
            }
            PyArray_ENABLEFLAGS((PyArrayObject*)args, NPY_ARRAY_OWNDATA);
            // Maybe try this too?
            // PyArray_BASE(PyArrayObject *)args) = NULL;
            Py_RETURN_NONE;
         """),
        ("get_array_with_base", "METH_NOARGS", """
            char *buf = (char *)malloc(20);
            npy_intp dims[1];
            dims[0] = 20;
            PyArray_Descr *descr =  PyArray_DescrNewFromType(NPY_UINT8);
            PyObject *arr = PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims,
                                                 NULL, buf,
                                                 NPY_ARRAY_WRITEABLE, NULL);
            if (arr == NULL) return NULL;
            PyObject *obj = PyCapsule_New(buf, "buf capsule",
                                          (PyCapsule_Destructor)&warn_on_free);
            if (obj == NULL) {
                Py_DECREF(arr);
                return NULL;
            }
            if (PyArray_SetBaseObject((PyArrayObject *)arr, obj) < 0) {
                Py_DECREF(arr);
                Py_DECREF(obj);
                return NULL;
            }
            return arr;

         """),
    ]
    prologue = '''
        #define NPY_TARGET_VERSION NPY_1_22_API_VERSION
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <numpy/arrayobject.h>
        /*
         * This struct allows the dynamic configuration of the allocator funcs
         * of the `secret_data_allocator`. It is provided here for
         * demonstration purposes, as a valid `ctx` use-case scenario.
         */
        typedef struct {
            void *(*malloc)(size_t);
            void *(*calloc)(size_t, size_t);
            void *(*realloc)(void *, size_t);
            void (*free)(void *);
        } SecretDataAllocatorFuncs;

        NPY_NO_EXPORT void *
        shift_alloc(void *ctx, size_t sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
            char *real = (char *)funcs->malloc(sz + 64);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated %ld", (unsigned long)sz);
            return (void *)(real + 64);
        }
        NPY_NO_EXPORT void *
        shift_zero(void *ctx, size_t sz, size_t cnt) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
            char *real = (char *)funcs->calloc(sz + 64, cnt);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated %ld via zero",
                     (unsigned long)sz);
            return (void *)(real + 64);
        }
        NPY_NO_EXPORT void
        shift_free(void *ctx, void * p, npy_uintp sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
            if (p == NULL) {
                return ;
            }
            char *real = (char *)p - 64;
            if (strncmp(real, "originally allocated", 20) != 0) {
                fprintf(stdout, "uh-oh, unmatched shift_free, "
                        "no appropriate prefix\\n");
                /* Make C runtime crash by calling free on the wrong address */
                funcs->free((char *)p + 10);
                /* funcs->free(real); */
            }
            else {
                npy_uintp i = (npy_uintp)atoi(real +20);
                if (i != sz) {
                    fprintf(stderr, "uh-oh, unmatched shift_free"
                            "(ptr, %ld) but allocated %ld\\n", sz, i);
                    /* This happens in some places, only print */
                    funcs->free(real);
                }
                else {
                    funcs->free(real);
                }
            }
        }
        NPY_NO_EXPORT void *
        shift_realloc(void *ctx, void * p, npy_uintp sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
            if (p != NULL) {
                char *real = (char *)p - 64;
                if (strncmp(real, "originally allocated", 20) != 0) {
                    fprintf(stdout, "uh-oh, unmatched shift_realloc\\n");
                    return realloc(p, sz);
                }
                return (void *)((char *)funcs->realloc(real, sz + 64) + 64);
            }
            else {
                char *real = (char *)funcs->realloc(p, sz + 64);
                if (real == NULL) {
                    return NULL;
                }
                snprintf(real, 64, "originally allocated "
                         "%ld  via realloc", (unsigned long)sz);
                return (void *)(real + 64);
            }
        }
        /* As an example, we use the standard {m|c|re}alloc/free funcs. */
        static SecretDataAllocatorFuncs secret_data_handler_ctx = {
            malloc,
            calloc,
            realloc,
            free
        };
        static PyDataMem_Handler secret_data_handler = {
            "secret_data_allocator",
            1,
            {
                &secret_data_handler_ctx, /* ctx */
                shift_alloc,              /* malloc */
                shift_zero,               /* calloc */
                shift_realloc,            /* realloc */
                shift_free                /* free */
            }
        };
        void warn_on_free(void *capsule) {
            PyErr_WarnEx(PyExc_UserWarning, "in warn_on_free", 1);
            void * obj = PyCapsule_GetPointer(capsule,
                                              PyCapsule_GetName(capsule));
            free(obj);
        };
        '''
    more_init = "import_array();"
    try:
        import mem_policy
        return mem_policy
    except ImportError:
        pass
    # if it does not exist, build and load it
    if sysconfig.get_platform() == "win-arm64":
        pytest.skip("Meson unable to find MSVC linker on win-arm64")
    return extbuild.build_and_import_extension('mem_policy',
                                               functions,
                                               prologue=prologue,
                                               include_dirs=[np.get_include()],
                                               build_dir=tmp_path,
                                               more_init=more_init)


def test_set_policy(get_module):

    get_handler_name = np._core.multiarray.get_handler_name
    get_handler_version = np._core.multiarray.get_handler_version
    orig_policy_name = get_handler_name()

    a = np.arange(10).reshape((2, 5))  # a doesn't own its own data
    assert get_handler_name(a) is None
    assert get_handler_version(a) is None
    assert get_handler_name(a.base) == orig_policy_name
    assert get_handler_version(a.base) == 1

    orig_policy = get_module.set_secret_data_policy()

    b = np.arange(10).reshape((2, 5))  # b doesn't own its own data
    assert get_handler_name(b) is None
    assert get_handler_version(b) is None
    assert get_handler_name(b.base) == 'secret_data_allocator'
    assert get_handler_version(b.base) == 1

    if orig_policy_name == 'default_allocator':
        get_module.set_old_policy(None)  # tests PyDataMem_SetHandler(NULL)
        assert get_handler_name() == 'default_allocator'
    else:
        get_module.set_old_policy(orig_policy)
        assert get_handler_name() == orig_policy_name

    with pytest.raises(ValueError,
                       match="Capsule must be named 'mem_handler'"):
        get_module.set_wrong_capsule_name_data_policy()


def test_default_policy_singleton(get_module):
    get_handler_name = np._core.multiarray.get_handler_name

    # set the policy to default
    orig_policy = get_module.set_old_policy(None)

    assert get_handler_name() == 'default_allocator'

    # re-set the policy to default
    def_policy_1 = get_module.set_old_policy(None)

    assert get_handler_name() == 'default_allocator'

    # set the policy to original
    def_policy_2 = get_module.set_old_policy(orig_policy)

    # since default policy is a singleton,
    # these should be the same object
    assert def_policy_1 is def_policy_2 is get_module.get_default_policy()


def test_policy_propagation(get_module):
    # The memory policy goes hand-in-hand with flags.owndata

    class MyArr(np.ndarray):
        pass

    get_handler_name = np._core.multiarray.get_handler_name
    orig_policy_name = get_handler_name()
    a = np.arange(10).view(MyArr).reshape((2, 5))
    assert get_handler_name(a) is None
    assert a.flags.owndata is False

    assert get_handler_name(a.base) is None
    assert a.base.flags.owndata is False

    assert get_handler_name(a.base.base) == orig_policy_name
    assert a.base.base.flags.owndata is True


async def concurrent_context1(get_module, orig_policy_name, event):
    if orig_policy_name == 'default_allocator':
        get_module.set_secret_data_policy()
        assert get_handler_name() == 'secret_data_allocator'
    else:
        get_module.set_old_policy(None)
        assert get_handler_name() == 'default_allocator'
    event.set()


async def concurrent_context2(get_module, orig_policy_name, event):
    await event.wait()
    # the policy is not affected by changes in parallel contexts
    assert get_handler_name() == orig_policy_name
    # change policy in the child context
    if orig_policy_name == 'default_allocator':
        get_module.set_secret_data_policy()
        assert get_handler_name() == 'secret_data_allocator'
    else:
        get_module.set_old_policy(None)
        assert get_handler_name() == 'default_allocator'


async def async_test_context_locality(get_module):
    orig_policy_name = np._core.multiarray.get_handler_name()

    event = asyncio.Event()
    # the child contexts inherit the parent policy
    concurrent_task1 = asyncio.create_task(
        concurrent_context1(get_module, orig_policy_name, event))
    concurrent_task2 = asyncio.create_task(
        concurrent_context2(get_module, orig_policy_name, event))
    await concurrent_task1
    await concurrent_task2

    # the parent context is not affected by child policy changes
    assert np._core.multiarray.get_handler_name() == orig_policy_name


def test_context_locality(get_module):
    if (sys.implementation.name == 'pypy'
            and sys.pypy_version_info[:3] < (7, 3, 6)):
        pytest.skip('no context-locality support in PyPy < 7.3.6')
    asyncio.run(async_test_context_locality(get_module))


def concurrent_thread1(get_module, event):
    get_module.set_secret_data_policy()
    assert np._core.multiarray.get_handler_name() == 'secret_data_allocator'
    event.set()


def concurrent_thread2(get_module, event):
    event.wait()
    # the policy is not affected by changes in parallel threads
    assert np._core.multiarray.get_handler_name() == 'default_allocator'
    # change policy in the child thread
    get_module.set_secret_data_policy()


def test_thread_locality(get_module):
    orig_policy_name = np._core.multiarray.get_handler_name()

    event = threading.Event()
    # the child threads do not inherit the parent policy
    concurrent_task1 = threading.Thread(target=concurrent_thread1,
                                        args=(get_module, event))
    concurrent_task2 = threading.Thread(target=concurrent_thread2,
                                        args=(get_module, event))
    concurrent_task1.start()
    concurrent_task2.start()
    concurrent_task1.join()
    concurrent_task2.join()

    # the parent thread is not affected by child policy changes
    assert np._core.multiarray.get_handler_name() == orig_policy_name


@pytest.mark.skip(reason="too slow, see gh-23975")
def test_new_policy(get_module):
    a = np.arange(10)
    orig_policy_name = np._core.multiarray.get_handler_name(a)

    orig_policy = get_module.set_secret_data_policy()

    b = np.arange(10)
    assert np._core.multiarray.get_handler_name(b) == 'secret_data_allocator'

    # test array manipulation. This is slow
    if orig_policy_name == 'default_allocator':
        # when the np._core.test tests recurse into this test, the
        # policy will be set so this "if" will be false, preventing
        # infinite recursion
        #
        # if needed, debug this by
        # - running tests with -- -s (to not capture stdout/stderr
        # - setting verbose=2
        # - setting extra_argv=['-vv'] here
        assert np._core.test('full', verbose=1, extra_argv=[])
        # also try the ma tests, the pickling test is quite tricky
        assert np.ma.test('full', verbose=1, extra_argv=[])

    get_module.set_old_policy(orig_policy)

    c = np.arange(10)
    assert np._core.multiarray.get_handler_name(c) == orig_policy_name


@pytest.mark.xfail(sys.implementation.name == "pypy",
                   reason=("bad interaction between getenv and "
                           "os.environ inside pytest"))
@pytest.mark.parametrize("policy", ["0", "1", None])
def test_switch_owner(get_module, policy):
    a = get_module.get_array()
    assert np._core.multiarray.get_handler_name(a) is None
    get_module.set_own(a)

    if policy is None:
        # See what we expect to be set based on the env variable
        policy = os.getenv("NUMPY_WARN_IF_NO_MEM_POLICY", "0") == "1"
        oldval = None
    else:
        policy = policy == "1"
        oldval = np._core._multiarray_umath._set_numpy_warn_if_no_mem_policy(
            policy)
    try:
        # The policy should be NULL, so we have to assume we can call
        # "free".  A warning is given if the policy == "1"
        if policy:
            with assert_warns(RuntimeWarning) as w:
                del a
                gc.collect()
        else:
            del a
            gc.collect()

    finally:
        if oldval is not None:
            np._core._multiarray_umath._set_numpy_warn_if_no_mem_policy(oldval)


def test_owner_is_base(get_module):
    a = get_module.get_array_with_base()
    with pytest.warns(UserWarning, match='warn_on_free'):
        del a
        gc.collect()
        gc.collect()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\tests\test_immutable_ndim_array.py ===
from copy import copy

from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.core.containers import Dict
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices import SparseMatrix
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import ImmutableSparseNDimArray
from sympy.testing.pytest import raises


def test_ndim_array_initiation():
    arr_with_no_elements = ImmutableDenseNDimArray([], shape=(0,))
    assert len(arr_with_no_elements) == 0
    assert arr_with_no_elements.rank() == 1

    raises(ValueError, lambda: ImmutableDenseNDimArray([0], shape=(0,)))
    raises(ValueError, lambda: ImmutableDenseNDimArray([1, 2, 3], shape=(0,)))
    raises(ValueError, lambda: ImmutableDenseNDimArray([], shape=()))

    raises(ValueError, lambda: ImmutableSparseNDimArray([0], shape=(0,)))
    raises(ValueError, lambda: ImmutableSparseNDimArray([1, 2, 3], shape=(0,)))
    raises(ValueError, lambda: ImmutableSparseNDimArray([], shape=()))

    arr_with_one_element = ImmutableDenseNDimArray([23])
    assert len(arr_with_one_element) == 1
    assert arr_with_one_element[0] == 23
    assert arr_with_one_element[:] == ImmutableDenseNDimArray([23])
    assert arr_with_one_element.rank() == 1

    arr_with_symbol_element = ImmutableDenseNDimArray([Symbol('x')])
    assert len(arr_with_symbol_element) == 1
    assert arr_with_symbol_element[0] == Symbol('x')
    assert arr_with_symbol_element[:] == ImmutableDenseNDimArray([Symbol('x')])
    assert arr_with_symbol_element.rank() == 1

    number5 = 5
    vector = ImmutableDenseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector.rank() == 1

    vector = ImmutableSparseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector._sparse_array == Dict()
    assert vector.rank() == 1

    n_dim_array = ImmutableDenseNDimArray(range(3**4), (3, 3, 3, 3,))
    assert len(n_dim_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == (3, 3, 3, 3)
    assert n_dim_array.rank() == 4

    array_shape = (3, 3, 3, 3)
    sparse_array = ImmutableSparseNDimArray.zeros(*array_shape)
    assert len(sparse_array._sparse_array) == 0
    assert len(sparse_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == array_shape
    assert n_dim_array.rank() == 4

    one_dim_array = ImmutableDenseNDimArray([2, 3, 1])
    assert len(one_dim_array) == 3
    assert one_dim_array.shape == (3,)
    assert one_dim_array.rank() == 1
    assert one_dim_array.tolist() == [2, 3, 1]

    shape = (3, 3)
    array_with_many_args = ImmutableSparseNDimArray.zeros(*shape)
    assert len(array_with_many_args) == 3 * 3
    assert array_with_many_args.shape == shape
    assert array_with_many_args[0, 0] == 0
    assert array_with_many_args.rank() == 2

    shape = (int(3), int(3))
    array_with_long_shape = ImmutableSparseNDimArray.zeros(*shape)
    assert len(array_with_long_shape) == 3 * 3
    assert array_with_long_shape.shape == shape
    assert array_with_long_shape[int(0), int(0)] == 0
    assert array_with_long_shape.rank() == 2

    vector_with_long_shape = ImmutableDenseNDimArray(range(5), int(5))
    assert len(vector_with_long_shape) == 5
    assert vector_with_long_shape.shape == (int(5),)
    assert vector_with_long_shape.rank() == 1
    raises(ValueError, lambda: vector_with_long_shape[int(5)])

    from sympy.abc import x
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        rank_zero_array = ArrayType(x)
        assert len(rank_zero_array) == 1
        assert rank_zero_array.shape == ()
        assert rank_zero_array.rank() == 0
        assert rank_zero_array[()] == x
        raises(ValueError, lambda: rank_zero_array[0])


def test_reshape():
    array = ImmutableDenseNDimArray(range(50), 50)
    assert array.shape == (50,)
    assert array.rank() == 1

    array = array.reshape(5, 5, 2)
    assert array.shape == (5, 5, 2)
    assert array.rank() == 3
    assert len(array) == 50


def test_getitem():
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        array = ArrayType(range(24)).reshape(2, 3, 4)
        assert array.tolist() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        assert array[0] == ArrayType([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        assert array[0, 0] == ArrayType([0, 1, 2, 3])
        value = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    assert array[i, j, k] == value
                    value += 1

    raises(ValueError, lambda: array[3, 4, 5])
    raises(ValueError, lambda: array[3, 4, 5, 6])
    raises(ValueError, lambda: array[3, 4, 5, 3:4])


def test_iterator():
    array = ImmutableDenseNDimArray(range(4), (2, 2))
    assert array[0] == ImmutableDenseNDimArray([0, 1])
    assert array[1] == ImmutableDenseNDimArray([2, 3])

    array = array.reshape(4)
    j = 0
    for i in array:
        assert i == j
        j += 1


def test_sparse():
    sparse_array = ImmutableSparseNDimArray([0, 0, 0, 1], (2, 2))
    assert len(sparse_array) == 2 * 2
    # dictionary where all data is, only non-zero entries are actually stored:
    assert len(sparse_array._sparse_array) == 1

    assert sparse_array.tolist() == [[0, 0], [0, 1]]

    for i, j in zip(sparse_array, [[0, 0], [0, 1]]):
        assert i == ImmutableSparseNDimArray(j)

    def sparse_assignment():
        sparse_array[0, 0] = 123

    assert len(sparse_array._sparse_array) == 1
    raises(TypeError, sparse_assignment)
    assert len(sparse_array._sparse_array) == 1
    assert sparse_array[0, 0] == 0
    assert sparse_array/0 == ImmutableSparseNDimArray([[S.NaN, S.NaN], [S.NaN, S.ComplexInfinity]], (2, 2))

    # test for large scale sparse array
    # equality test
    assert ImmutableSparseNDimArray.zeros(100000, 200000) == ImmutableSparseNDimArray.zeros(100000, 200000)

    # __mul__ and __rmul__
    a = ImmutableSparseNDimArray({200001: 1}, (100000, 200000))
    assert a * 3 == ImmutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert 3 * a == ImmutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert a * 0 == ImmutableSparseNDimArray({}, (100000, 200000))
    assert 0 * a == ImmutableSparseNDimArray({}, (100000, 200000))

    # __truediv__
    assert a/3 == ImmutableSparseNDimArray({200001: Rational(1, 3)}, (100000, 200000))

    # __neg__
    assert -a == ImmutableSparseNDimArray({200001: -1}, (100000, 200000))


def test_calculation():

    a = ImmutableDenseNDimArray([1]*9, (3, 3))
    b = ImmutableDenseNDimArray([9]*9, (3, 3))

    c = a + b
    for i in c:
        assert i == ImmutableDenseNDimArray([10, 10, 10])

    assert c == ImmutableDenseNDimArray([10]*9, (3, 3))
    assert c == ImmutableSparseNDimArray([10]*9, (3, 3))

    c = b - a
    for i in c:
        assert i == ImmutableDenseNDimArray([8, 8, 8])

    assert c == ImmutableDenseNDimArray([8]*9, (3, 3))
    assert c == ImmutableSparseNDimArray([8]*9, (3, 3))


def test_ndim_array_converting():
    dense_array = ImmutableDenseNDimArray([1, 2, 3, 4], (2, 2))
    alist = dense_array.tolist()

    assert alist == [[1, 2], [3, 4]]

    matrix = dense_array.tomatrix()
    assert (isinstance(matrix, Matrix))

    for i in range(len(dense_array)):
        assert dense_array[dense_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == dense_array.shape

    assert ImmutableDenseNDimArray(matrix) == dense_array
    assert ImmutableDenseNDimArray(matrix.as_immutable()) == dense_array
    assert ImmutableDenseNDimArray(matrix.as_mutable()) == dense_array

    sparse_array = ImmutableSparseNDimArray([1, 2, 3, 4], (2, 2))
    alist = sparse_array.tolist()

    assert alist == [[1, 2], [3, 4]]

    matrix = sparse_array.tomatrix()
    assert(isinstance(matrix, SparseMatrix))

    for i in range(len(sparse_array)):
        assert sparse_array[sparse_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == sparse_array.shape

    assert ImmutableSparseNDimArray(matrix) == sparse_array
    assert ImmutableSparseNDimArray(matrix.as_immutable()) == sparse_array
    assert ImmutableSparseNDimArray(matrix.as_mutable()) == sparse_array


def test_converting_functions():
    arr_list = [1, 2, 3, 4]
    arr_matrix = Matrix(((1, 2), (3, 4)))

    # list
    arr_ndim_array = ImmutableDenseNDimArray(arr_list, (2, 2))
    assert (isinstance(arr_ndim_array, ImmutableDenseNDimArray))
    assert arr_matrix.tolist() == arr_ndim_array.tolist()

    # Matrix
    arr_ndim_array = ImmutableDenseNDimArray(arr_matrix)
    assert (isinstance(arr_ndim_array, ImmutableDenseNDimArray))
    assert arr_matrix.tolist() == arr_ndim_array.tolist()
    assert arr_matrix.shape == arr_ndim_array.shape


def test_equality():
    first_list = [1, 2, 3, 4]
    second_list = [1, 2, 3, 4]
    third_list = [4, 3, 2, 1]
    assert first_list == second_list
    assert first_list != third_list

    first_ndim_array = ImmutableDenseNDimArray(first_list, (2, 2))
    second_ndim_array = ImmutableDenseNDimArray(second_list, (2, 2))
    fourth_ndim_array = ImmutableDenseNDimArray(first_list, (2, 2))

    assert first_ndim_array == second_ndim_array

    def assignment_attempt(a):
        a[0, 0] = 0

    raises(TypeError, lambda: assignment_attempt(second_ndim_array))
    assert first_ndim_array == second_ndim_array
    assert first_ndim_array == fourth_ndim_array


def test_arithmetic():
    a = ImmutableDenseNDimArray([3 for i in range(9)], (3, 3))
    b = ImmutableDenseNDimArray([7 for i in range(9)], (3, 3))

    c1 = a + b
    c2 = b + a
    assert c1 == c2

    d1 = a - b
    d2 = b - a
    assert d1 == d2 * (-1)

    e1 = a * 5
    e2 = 5 * a
    e3 = copy(a)
    e3 *= 5
    assert e1 == e2 == e3

    f1 = a / 5
    f2 = copy(a)
    f2 /= 5
    assert f1 == f2
    assert f1[0, 0] == f1[0, 1] == f1[0, 2] == f1[1, 0] == f1[1, 1] == \
    f1[1, 2] == f1[2, 0] == f1[2, 1] == f1[2, 2] == Rational(3, 5)

    assert type(a) == type(b) == type(c1) == type(c2) == type(d1) == type(d2) \
        == type(e1) == type(e2) == type(e3) == type(f1)

    z0 = -a
    assert z0 == ImmutableDenseNDimArray([-3 for i in range(9)], (3, 3))


def test_higher_dimenions():
    m3 = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert m3.tolist() == [[[10, 11, 12, 13],
            [14, 15, 16, 17],
            [18, 19, 20, 21]],

           [[22, 23, 24, 25],
            [26, 27, 28, 29],
            [30, 31, 32, 33]]]

    assert m3._get_tuple_index(0) == (0, 0, 0)
    assert m3._get_tuple_index(1) == (0, 0, 1)
    assert m3._get_tuple_index(4) == (0, 1, 0)
    assert m3._get_tuple_index(12) == (1, 0, 0)

    assert str(m3) == '[[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]]'

    m3_rebuilt = ImmutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]])
    assert m3 == m3_rebuilt

    m3_other = ImmutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]], (2, 3, 4))

    assert m3 == m3_other


def test_rebuild_immutable_arrays():
    sparr = ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    densarr = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert sparr == sparr.func(*sparr.args)
    assert densarr == densarr.func(*densarr.args)


def test_slices():
    md = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert md[:] == ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))
    assert md[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    assert md[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    assert md[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    assert md[:, :, :] == md

    sd = ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd == ImmutableSparseNDimArray(md)

    assert sd[:] == ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    assert sd[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    assert sd[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    assert sd[:, :, :] == sd


def test_diff_and_applyfunc():
    from sympy.abc import x, y, z
    md = ImmutableDenseNDimArray([[x, y], [x*z, x*y*z]])
    assert md.diff(x) == ImmutableDenseNDimArray([[1, 0], [z, y*z]])
    assert diff(md, x) == ImmutableDenseNDimArray([[1, 0], [z, y*z]])

    sd = ImmutableSparseNDimArray(md)
    assert sd == ImmutableSparseNDimArray([x, y, x*z, x*y*z], (2, 2))
    assert sd.diff(x) == ImmutableSparseNDimArray([[1, 0], [z, y*z]])
    assert diff(sd, x) == ImmutableSparseNDimArray([[1, 0], [z, y*z]])

    mdn = md.applyfunc(lambda x: x*3)
    assert mdn == ImmutableDenseNDimArray([[3*x, 3*y], [3*x*z, 3*x*y*z]])
    assert md != mdn

    sdn = sd.applyfunc(lambda x: x/2)
    assert sdn == ImmutableSparseNDimArray([[x/2, y/2], [x*z/2, x*y*z/2]])
    assert sd != sdn

    sdp = sd.applyfunc(lambda x: x+1)
    assert sdp == ImmutableSparseNDimArray([[x + 1, y + 1], [x*z + 1, x*y*z + 1]])
    assert sd != sdp


def test_op_priority():
    from sympy.abc import x
    md = ImmutableDenseNDimArray([1, 2, 3])
    e1 = (1+x)*md
    e2 = md*(1+x)
    assert e1 == ImmutableDenseNDimArray([1+x, 2+2*x, 3+3*x])
    assert e1 == e2

    sd = ImmutableSparseNDimArray([1, 2, 3])
    e3 = (1+x)*sd
    e4 = sd*(1+x)
    assert e3 == ImmutableDenseNDimArray([1+x, 2+2*x, 3+3*x])
    assert e3 == e4


def test_symbolic_indexing():
    x, y, z, w = symbols("x y z w")
    M = ImmutableDenseNDimArray([[x, y], [z, w]])
    i, j = symbols("i, j")
    Mij = M[i, j]
    assert isinstance(Mij, Indexed)
    Ms = ImmutableSparseNDimArray([[2, 3*x], [4, 5]])
    msij = Ms[i, j]
    assert isinstance(msij, Indexed)
    for oi, oj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert Mij.subs({i: oi, j: oj}) == M[oi, oj]
        assert msij.subs({i: oi, j: oj}) == Ms[oi, oj]
    A = IndexedBase("A", (0, 2))
    assert A[0, 0].subs(A, M) == x
    assert A[i, j].subs(A, M) == M[i, j]
    assert M[i, j].subs(M, A) == A[i, j]

    assert isinstance(M[3 * i - 2, j], Indexed)
    assert M[3 * i - 2, j].subs({i: 1, j: 0}) == M[1, 0]
    assert isinstance(M[i, 0], Indexed)
    assert M[i, 0].subs(i, 0) == M[0, 0]
    assert M[0, i].subs(i, 1) == M[0, 1]

    assert M[i, j].diff(x) == ImmutableDenseNDimArray([[1, 0], [0, 0]])[i, j]
    assert Ms[i, j].diff(x) == ImmutableSparseNDimArray([[0, 3], [0, 0]])[i, j]

    Mo = ImmutableDenseNDimArray([1, 2, 3])
    assert Mo[i].subs(i, 1) == 2
    Mos = ImmutableSparseNDimArray([1, 2, 3])
    assert Mos[i].subs(i, 1) == 2

    raises(ValueError, lambda: M[i, 2])
    raises(ValueError, lambda: M[i, -1])
    raises(ValueError, lambda: M[2, i])
    raises(ValueError, lambda: M[-1, i])

    raises(ValueError, lambda: Ms[i, 2])
    raises(ValueError, lambda: Ms[i, -1])
    raises(ValueError, lambda: Ms[2, i])
    raises(ValueError, lambda: Ms[-1, i])


def test_issue_12665():
    # Testing Python 3 hash of immutable arrays:
    arr = ImmutableDenseNDimArray([1, 2, 3])
    # This should NOT raise an exception:
    hash(arr)


def test_zeros_without_shape():
    arr = ImmutableDenseNDimArray.zeros()
    assert arr == ImmutableDenseNDimArray(0)

def test_issue_21870():
    a0 = ImmutableDenseNDimArray(0)
    assert a0.rank() == 0
    a1 = ImmutableDenseNDimArray(a0)
    assert a1.rank() == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_basic_ops.py ===
import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys

try:
    long = long
except NameError:
    long = int

def test_type_compare():
    assert mpf(2) == mpc(2,0)
    assert mpf(0) == mpc(0)
    assert mpf(2) != mpc(2, 0.00001)
    assert mpf(2) == 2.0
    assert mpf(2) != 3.0
    assert mpf(2) == 2
    assert mpf(2) != '2.0'
    assert mpc(2) != '2.0'

def test_add():
    assert mpf(2.5) + mpf(3) == 5.5
    assert mpf(2.5) + 3 == 5.5
    assert mpf(2.5) + 3.0 == 5.5
    assert 3 + mpf(2.5) == 5.5
    assert 3.0 + mpf(2.5) == 5.5
    assert (3+0j) + mpf(2.5) == 5.5
    assert mpc(2.5) + mpf(3) == 5.5
    assert mpc(2.5) + 3 == 5.5
    assert mpc(2.5) + 3.0 == 5.5
    assert mpc(2.5) + (3+0j) == 5.5
    assert 3 + mpc(2.5) == 5.5
    assert 3.0 + mpc(2.5) == 5.5
    assert (3+0j) + mpc(2.5) == 5.5

def test_sub():
    assert mpf(2.5) - mpf(3) == -0.5
    assert mpf(2.5) - 3 == -0.5
    assert mpf(2.5) - 3.0 == -0.5
    assert 3 - mpf(2.5) == 0.5
    assert 3.0 - mpf(2.5) == 0.5
    assert (3+0j) - mpf(2.5) == 0.5
    assert mpc(2.5) - mpf(3) == -0.5
    assert mpc(2.5) - 3 == -0.5
    assert mpc(2.5) - 3.0 == -0.5
    assert mpc(2.5) - (3+0j) == -0.5
    assert 3 - mpc(2.5) == 0.5
    assert 3.0 - mpc(2.5) == 0.5
    assert (3+0j) - mpc(2.5) == 0.5

def test_mul():
    assert mpf(2.5) * mpf(3) == 7.5
    assert mpf(2.5) * 3 == 7.5
    assert mpf(2.5) * 3.0 == 7.5
    assert 3 * mpf(2.5) == 7.5
    assert 3.0 * mpf(2.5) == 7.5
    assert (3+0j) * mpf(2.5) == 7.5
    assert mpc(2.5) * mpf(3) == 7.5
    assert mpc(2.5) * 3 == 7.5
    assert mpc(2.5) * 3.0 == 7.5
    assert mpc(2.5) * (3+0j) == 7.5
    assert 3 * mpc(2.5) == 7.5
    assert 3.0 * mpc(2.5) == 7.5
    assert (3+0j) * mpc(2.5) == 7.5

def test_div():
    assert mpf(6) / mpf(3) == 2.0
    assert mpf(6) / 3 == 2.0
    assert mpf(6) / 3.0 == 2.0
    assert 6 / mpf(3) == 2.0
    assert 6.0 / mpf(3) == 2.0
    assert (6+0j) / mpf(3.0) == 2.0
    assert mpc(6) / mpf(3) == 2.0
    assert mpc(6) / 3 == 2.0
    assert mpc(6) / 3.0 == 2.0
    assert mpc(6) / (3+0j) == 2.0
    assert 6 / mpc(3) == 2.0
    assert 6.0 / mpc(3) == 2.0
    assert (6+0j) / mpc(3) == 2.0

def test_pow():
    assert mpf(6) ** mpf(3) == 216.0
    assert mpf(6) ** 3 == 216.0
    assert mpf(6) ** 3.0 == 216.0
    assert 6 ** mpf(3) == 216.0
    assert 6.0 ** mpf(3) == 216.0
    assert (6+0j) ** mpf(3.0) == 216.0
    assert mpc(6) ** mpf(3) == 216.0
    assert mpc(6) ** 3 == 216.0
    assert mpc(6) ** 3.0 == 216.0
    assert mpc(6) ** (3+0j) == 216.0
    assert 6 ** mpc(3) == 216.0
    assert 6.0 ** mpc(3) == 216.0
    assert (6+0j) ** mpc(3) == 216.0

def test_mixed_misc():
    assert 1 + mpf(3) == mpf(3) + 1 == 4
    assert 1 - mpf(3) == -(mpf(3) - 1) == -2
    assert 3 * mpf(2) == mpf(2) * 3 == 6
    assert 6 / mpf(2) == mpf(6) / 2 == 3
    assert 1.0 + mpf(3) == mpf(3) + 1.0 == 4
    assert 1.0 - mpf(3) == -(mpf(3) - 1.0) == -2
    assert 3.0 * mpf(2) == mpf(2) * 3.0 == 6
    assert 6.0 / mpf(2) == mpf(6) / 2.0 == 3

def test_add_misc():
    mp.dps = 15
    assert mpf(4) + mpf(-70) == -66
    assert mpf(1) + mpf(1.1)/80 == 1 + 1.1/80
    assert mpf((1, 10000000000)) + mpf(3) == mpf((1, 10000000000))
    assert mpf(3) + mpf((1, 10000000000)) == mpf((1, 10000000000))
    assert mpf((1, -10000000000)) + mpf(3) == mpf(3)
    assert mpf(3) + mpf((1, -10000000000)) == mpf(3)
    assert mpf(1) + 1e-15 != 1
    assert mpf(1) + 1e-20 == 1
    assert mpf(1.07e-22) + 0 == mpf(1.07e-22)
    assert mpf(0) + mpf(1.07e-22) == mpf(1.07e-22)

def test_complex_misc():
    # many more tests needed
    assert 1 + mpc(2) == 3
    assert not mpc(2).ae(2 + 1e-13)
    assert mpc(2+1e-15j).ae(2)

def test_complex_zeros():
    for a in [0,2]:
        for b in [0,3]:
            for c in [0,4]:
                for d in [0,5]:
                    assert mpc(a,b)*mpc(c,d) == complex(a,b)*complex(c,d)

def test_hash():
    for i in range(-256, 256):
        assert hash(mpf(i)) == hash(i)
    assert hash(mpf(0.5)) == hash(0.5)
    assert hash(mpc(2,3)) == hash(2+3j)
    # Check that this doesn't fail
    assert hash(inf)
    # Check that overflow doesn't assign equal hashes to large numbers
    assert hash(mpf('1e1000')) != hash('1e10000')
    assert hash(mpc(100,'1e1000')) != hash(mpc(200,'1e1000'))
    from mpmath.rational import mpq
    assert hash(mp.mpq(1,3))
    assert hash(mp.mpq(0,1)) == 0
    assert hash(mp.mpq(-1,1)) == hash(-1)
    assert hash(mp.mpq(1,1)) == hash(1)
    assert hash(mp.mpq(5,1)) == hash(5)
    assert hash(mp.mpq(1,2)) == hash(0.5)
    if sys.version_info >= (3, 2):
        assert hash(mpf(1)*2**2000) == hash(2**2000)
        assert hash(mpf(1)/2**2000) == hash(mpq(1,2**2000))

# Advanced rounding test
def test_add_rounding():
    mp.dps = 15
    a = from_float(1e-50)
    assert mpf_sub(mpf_add(fone, a, 53, round_up), fone, 53, round_up) == from_float(2.2204460492503131e-16)
    assert mpf_sub(fone, a, 53, round_up) == fone
    assert mpf_sub(fone, mpf_sub(fone, a, 53, round_down), 53, round_down) == from_float(1.1102230246251565e-16)
    assert mpf_add(fone, a, 53, round_down) == fone

def test_almost_equal():
    assert mpf(1.2).ae(mpf(1.20000001), 1e-7)
    assert not mpf(1.2).ae(mpf(1.20000001), 1e-9)
    assert not mpf(-0.7818314824680298).ae(mpf(-0.774695868667929))

def test_arithmetic_functions():
    import operator
    ops = [(operator.add, fadd), (operator.sub, fsub), (operator.mul, fmul),
        (operator.truediv, fdiv)]
    a = mpf(0.27)
    b = mpf(1.13)
    c = mpc(0.51+2.16j)
    d = mpc(1.08-0.99j)
    for x in [a,b,c,d]:
        for y in [a,b,c,d]:
            for op, fop in ops:
                if fop is not fdiv:
                    mp.prec = 200
                    z0 = op(x,y)
                mp.prec = 60
                z1 = op(x,y)
                mp.prec = 53
                z2 = op(x,y)
                assert fop(x, y, prec=60) == z1
                assert fop(x, y) == z2
                if fop is not fdiv:
                    assert fop(x, y, prec=inf) == z0
                    assert fop(x, y, dps=inf) == z0
                    assert fop(x, y, exact=True) == z0
                assert fneg(fneg(z1, exact=True), prec=inf) == z1
                assert fneg(z1) == -(+z1)
    mp.dps = 15

def test_exact_integer_arithmetic():
    # XXX: re-fix this so that all operations are tested with all rounding modes
    random.seed(0)
    for prec in [6, 10, 25, 40, 100, 250, 725]:
        for rounding in ['d', 'u', 'f', 'c', 'n']:
            mp.dps = prec
            M = 10**(prec-2)
            M2 = 10**(prec//2-2)
            for i in range(10):
                a = random.randint(-M, M)
                b = random.randint(-M, M)
                assert mpf(a, rounding=rounding) == a
                assert int(mpf(a, rounding=rounding)) == a
                assert int(mpf(str(a), rounding=rounding)) == a
                assert mpf(a) + mpf(b) == a + b
                assert mpf(a) - mpf(b) == a - b
                assert -mpf(a) == -a
                a = random.randint(-M2, M2)
                b = random.randint(-M2, M2)
                assert mpf(a) * mpf(b) == a*b
                assert mpf_mul(from_int(a), from_int(b), mp.prec, rounding) == from_int(a*b)
    mp.dps = 15

def test_odd_int_bug():
    assert to_int(from_int(3), round_nearest) == 3

def test_str_1000_digits():
    mp.dps = 1001
    # last digit may be wrong
    assert str(mpf(2)**0.5)[-10:-1] == '9518488472'[:9]
    assert str(pi)[-10:-1] == '2164201989'[:9]
    mp.dps = 15

def test_str_10000_digits():
    mp.dps = 10001
    # last digit may be wrong
    assert str(mpf(2)**0.5)[-10:-1] == '5873258351'[:9]
    assert str(pi)[-10:-1] == '5256375678'[:9]
    mp.dps = 15

def test_monitor():
    f = lambda x: x**2
    a = []
    b = []
    g = monitor(f, a.append, b.append)
    assert g(3) == 9
    assert g(4) == 16
    assert a[0] == ((3,), {})
    assert b[0] == 9

def test_nint_distance():
    assert nint_distance(mpf(-3)) == (-3, -inf)
    assert nint_distance(mpc(-3)) == (-3, -inf)
    assert nint_distance(mpf(-3.1)) == (-3, -3)
    assert nint_distance(mpf(-3.01)) == (-3, -6)
    assert nint_distance(mpf(-3.001)) == (-3, -9)
    assert nint_distance(mpf(-3.0001)) == (-3, -13)
    assert nint_distance(mpf(-2.9)) == (-3, -3)
    assert nint_distance(mpf(-2.99)) == (-3, -6)
    assert nint_distance(mpf(-2.999)) == (-3, -9)
    assert nint_distance(mpf(-2.9999)) == (-3, -13)
    assert nint_distance(mpc(-3+0.1j)) == (-3, -3)
    assert nint_distance(mpc(-3+0.01j)) == (-3, -6)
    assert nint_distance(mpc(-3.1+0.1j)) == (-3, -3)
    assert nint_distance(mpc(-3.01+0.01j)) == (-3, -6)
    assert nint_distance(mpc(-3.001+0.001j)) == (-3, -9)
    assert nint_distance(mpf(0)) == (0, -inf)
    assert nint_distance(mpf(0.01)) == (0, -6)
    assert nint_distance(mpf('1e-100')) == (0, -332)

def test_floor_ceil_nint_frac():
    mp.dps = 15
    for n in range(-10,10):
        assert floor(n) == n
        assert floor(n+0.5) == n
        assert ceil(n) == n
        assert ceil(n+0.5) == n+1
        assert nint(n) == n
        # nint rounds to even
        if n % 2 == 1:
            assert nint(n+0.5) == n+1
        else:
            assert nint(n+0.5) == n
    assert floor(inf) == inf
    assert floor(ninf) == ninf
    assert isnan(floor(nan))
    assert ceil(inf) == inf
    assert ceil(ninf) == ninf
    assert isnan(ceil(nan))
    assert nint(inf) == inf
    assert nint(ninf) == ninf
    assert isnan(nint(nan))
    assert floor(0.1) == 0
    assert floor(0.9) == 0
    assert floor(-0.1) == -1
    assert floor(-0.9) == -1
    assert floor(10000000000.1) == 10000000000
    assert floor(10000000000.9) == 10000000000
    assert floor(-10000000000.1) == -10000000000-1
    assert floor(-10000000000.9) == -10000000000-1
    assert floor(1e-100) == 0
    assert floor(-1e-100) == -1
    assert floor(1e100) == 1e100
    assert floor(-1e100) == -1e100
    assert ceil(0.1) == 1
    assert ceil(0.9) == 1
    assert ceil(-0.1) == 0
    assert ceil(-0.9) == 0
    assert ceil(10000000000.1) == 10000000000+1
    assert ceil(10000000000.9) == 10000000000+1
    assert ceil(-10000000000.1) == -10000000000
    assert ceil(-10000000000.9) == -10000000000
    assert ceil(1e-100) == 1
    assert ceil(-1e-100) == 0
    assert ceil(1e100) == 1e100
    assert ceil(-1e100) == -1e100
    assert nint(0.1) == 0
    assert nint(0.9) == 1
    assert nint(-0.1) == 0
    assert nint(-0.9) == -1
    assert nint(10000000000.1) == 10000000000
    assert nint(10000000000.9) == 10000000000+1
    assert nint(-10000000000.1) == -10000000000
    assert nint(-10000000000.9) == -10000000000-1
    assert nint(1e-100) == 0
    assert nint(-1e-100) == 0
    assert nint(1e100) == 1e100
    assert nint(-1e100) == -1e100
    assert floor(3.2+4.6j) == 3+4j
    assert ceil(3.2+4.6j) == 4+5j
    assert nint(3.2+4.6j) == 3+5j
    for n in range(-10,10):
        assert frac(n) == 0
    assert frac(0.25) == 0.25
    assert frac(1.25) == 0.25
    assert frac(2.25) == 0.25
    assert frac(-0.25) == 0.75
    assert frac(-1.25) == 0.75
    assert frac(-2.25) == 0.75
    assert frac('1e100000000000000') == 0
    u = mpf('1e-100000000000000')
    assert frac(u) == u
    assert frac(-u) == 1  # rounding!
    u = mpf('1e-400')
    assert frac(-u, prec=0) == fsub(1, u, exact=True)
    assert frac(3.25+4.75j) == 0.25+0.75j

def test_isnan_etc():
    from mpmath.rational import mpq
    assert isnan(nan) == True
    assert isnan(3) == False
    assert isnan(mpf(3)) == False
    assert isnan(inf) == False
    assert isnan(mpc(2,nan)) == True
    assert isnan(mpc(2,nan)) == True
    assert isnan(mpc(nan,nan)) == True
    assert isnan(mpc(2,2)) == False
    assert isnan(mpc(nan,inf)) == True
    assert isnan(mpc(inf,inf)) == False
    assert isnan(mpq((3,2))) == False
    assert isnan(mpq((0,1))) == False
    assert isinf(inf) == True
    assert isinf(-inf) == True
    assert isinf(3) == False
    assert isinf(nan) == False
    assert isinf(3+4j) == False
    assert isinf(mpc(inf)) == True
    assert isinf(mpc(3,inf)) == True
    assert isinf(mpc(inf,3)) == True
    assert isinf(mpc(inf,inf)) == True
    assert isinf(mpc(nan,inf)) == True
    assert isinf(mpc(inf,nan)) == True
    assert isinf(mpc(nan,nan)) == False
    assert isinf(mpq((3,2))) == False
    assert isinf(mpq((0,1))) == False
    assert isnormal(3) == True
    assert isnormal(3.5) == True
    assert isnormal(mpf(3.5)) == True
    assert isnormal(0) == False
    assert isnormal(mpf(0)) == False
    assert isnormal(0.0) == False
    assert isnormal(inf) == False
    assert isnormal(-inf) == False
    assert isnormal(nan) == False
    assert isnormal(float(inf)) == False
    assert isnormal(mpc(0,0)) == False
    assert isnormal(mpc(3,0)) == True
    assert isnormal(mpc(0,3)) == True
    assert isnormal(mpc(3,3)) == True
    assert isnormal(mpc(0,nan)) == False
    assert isnormal(mpc(0,inf)) == False
    assert isnormal(mpc(3,nan)) == False
    assert isnormal(mpc(3,inf)) == False
    assert isnormal(mpc(3,-inf)) == False
    assert isnormal(mpc(nan,0)) == False
    assert isnormal(mpc(inf,0)) == False
    assert isnormal(mpc(nan,3)) == False
    assert isnormal(mpc(inf,3)) == False
    assert isnormal(mpc(inf,nan)) == False
    assert isnormal(mpc(nan,inf)) == False
    assert isnormal(mpc(nan,nan)) == False
    assert isnormal(mpc(inf,inf)) == False
    assert isnormal(mpq((3,2))) == True
    assert isnormal(mpq((0,1))) == False
    assert isint(3) == True
    assert isint(0) == True
    assert isint(long(3)) == True
    assert isint(long(0)) == True
    assert isint(mpf(3)) == True
    assert isint(mpf(0)) == True
    assert isint(mpf(-3)) == True
    assert isint(mpf(3.2)) == False
    assert isint(3.2) == False
    assert isint(nan) == False
    assert isint(inf) == False
    assert isint(-inf) == False
    assert isint(mpc(0)) == True
    assert isint(mpc(3)) == True
    assert isint(mpc(3.2)) == False
    assert isint(mpc(3,inf)) == False
    assert isint(mpc(inf)) == False
    assert isint(mpc(3,2)) == False
    assert isint(mpc(0,2)) == False
    assert isint(mpc(3,2),gaussian=True) == True
    assert isint(mpc(3,0),gaussian=True) == True
    assert isint(mpc(0,3),gaussian=True) == True
    assert isint(3+4j) == False
    assert isint(3+4j, gaussian=True) == True
    assert isint(3+0j) == True
    assert isint(mpq((3,2))) == False
    assert isint(mpq((3,9))) == False
    assert isint(mpq((9,3))) == True
    assert isint(mpq((0,4))) == True
    assert isint(mpq((1,1))) == True
    assert isint(mpq((-1,1))) == True
    assert mp.isnpint(0) == True
    assert mp.isnpint(1) == False
    assert mp.isnpint(-1) == True
    assert mp.isnpint(-1.1) == False
    assert mp.isnpint(-1.0) == True
    assert mp.isnpint(mp.mpq(1,2)) == False
    assert mp.isnpint(mp.mpq(-1,2)) == False
    assert mp.isnpint(mp.mpq(-3,1)) == True
    assert mp.isnpint(mp.mpq(0,1)) == True
    assert mp.isnpint(mp.mpq(1,1)) == False
    assert mp.isnpint(0+0j) == True
    assert mp.isnpint(-1+0j) == True
    assert mp.isnpint(-1.1+0j) == False
    assert mp.isnpint(-1+0.1j) == False
    assert mp.isnpint(0+0.1j) == False


def test_issue_438():
    assert mpf(finf) == mpf('inf')
    assert mpf(fninf) == mpf('-inf')
    assert mpf(fnan)._mpf_ == mpf('nan')._mpf_

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\setitem.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class BaseSetitemTests:
    @pytest.fixture(
        params=[
            lambda x: x.index,
            lambda x: list(x.index),
            lambda x: slice(None),
            lambda x: slice(0, len(x)),
            lambda x: range(len(x)),
            lambda x: list(range(len(x))),
            lambda x: np.ones(len(x), dtype=bool),
        ],
        ids=[
            "index",
            "list[index]",
            "null_slice",
            "full_slice",
            "range",
            "list(range)",
            "mask",
        ],
    )
    def full_indexer(self, request):
        """
        Fixture for an indexer to pass to obj.loc to get/set the full length of the
        object.

        In some cases, assumes that obj.index is the default RangeIndex.
        """
        return request.param

    @pytest.fixture(autouse=True)
    def skip_if_immutable(self, dtype, request):
        if dtype._is_immutable:
            node = request.node
            if node.name.split("[")[0] == "test_is_immutable":
                # This fixture is auto-used, but we want to not-skip
                # test_is_immutable.
                return

            # When BaseSetitemTests is mixed into ExtensionTests, we only
            #  want this fixture to operate on the tests defined in this
            #  class/file.
            defined_in = node.function.__qualname__.split(".")[0]
            if defined_in == "BaseSetitemTests":
                pytest.skip("__setitem__ test not applicable with immutable dtype")

    def test_is_immutable(self, data):
        if data.dtype._is_immutable:
            with pytest.raises(TypeError):
                data[0] = data[0]
        else:
            data[0] = data[1]
            assert data[0] == data[1]

    def test_setitem_scalar_series(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        data[0] = data[1]
        assert data[0] == data[1]

    def test_setitem_sequence(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()

        data[[0, 1]] = [data[1], data[0]]
        assert data[0] == original[1]
        assert data[1] == original[0]

    def test_setitem_sequence_mismatched_length_raises(self, data, as_array):
        ser = pd.Series(data)
        original = ser.copy()
        value = [data[0]]
        if as_array:
            value = data._from_sequence(value, dtype=data.dtype)

        xpr = "cannot set using a {} indexer with a different length"
        with pytest.raises(ValueError, match=xpr.format("list-like")):
            ser[[0, 1]] = value
        # Ensure no modifications made before the exception
        tm.assert_series_equal(ser, original)

        with pytest.raises(ValueError, match=xpr.format("slice")):
            ser[slice(3)] = value
        tm.assert_series_equal(ser, original)

    def test_setitem_empty_indexer(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()
        data[np.array([], dtype=int)] = []
        tm.assert_equal(data, original)

    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        data[[0, 1]] = data[2]
        assert data[0] == data[2]
        assert data[1] == data[2]

    @pytest.mark.parametrize("setter", ["loc", "iloc"])
    def test_setitem_scalar(self, data, setter):
        arr = pd.Series(data)
        setter = getattr(arr, setter)
        setter[0] = data[1]
        assert arr[0] == data[1]

    def test_setitem_loc_scalar_mixed(self, data):
        df = pd.DataFrame({"A": np.arange(len(data)), "B": data})
        df.loc[0, "B"] = data[1]
        assert df.loc[0, "B"] == data[1]

    def test_setitem_loc_scalar_single(self, data):
        df = pd.DataFrame({"B": data})
        df.loc[10, "B"] = data[1]
        assert df.loc[10, "B"] == data[1]

    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        df = pd.DataFrame({"A": data, "B": data})
        df.loc[10, "B"] = data[1]
        assert df.loc[10, "B"] == data[1]

    def test_setitem_iloc_scalar_mixed(self, data):
        df = pd.DataFrame({"A": np.arange(len(data)), "B": data})
        df.iloc[0, 1] = data[1]
        assert df.loc[0, "B"] == data[1]

    def test_setitem_iloc_scalar_single(self, data):
        df = pd.DataFrame({"B": data})
        df.iloc[10, 0] = data[1]
        assert df.loc[10, "B"] == data[1]

    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        df = pd.DataFrame({"A": data, "B": data})
        df.iloc[10, 1] = data[1]
        assert df.loc[10, "B"] == data[1]

    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        arr = data[:5].copy()
        expected = arr.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)
        arr[mask] = data[0]
        tm.assert_equal(expected, arr)

    def test_setitem_mask_raises(self, data, box_in_series):
        # wrong length
        mask = np.array([True, False])

        if box_in_series:
            data = pd.Series(data)

        with pytest.raises(IndexError, match="wrong length"):
            data[mask] = data[0]

        mask = pd.array(mask, dtype="boolean")
        with pytest.raises(IndexError, match="wrong length"):
            data[mask] = data[0]

    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        mask = pd.array(np.zeros(data.shape, dtype="bool"), dtype="boolean")
        mask[:3] = True
        mask[3:5] = pd.NA

        if box_in_series:
            data = pd.Series(data)

        data[mask] = data[0]

        assert (data[:3] == data[0]).all()

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series):
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])

        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        arr[idx] = arr[0]
        tm.assert_equal(arr, expected)

    @pytest.mark.parametrize(
        "idx, box_in_series",
        [
            ([0, 1, 2, pd.NA], False),
            pytest.param(
                [0, 1, 2, pd.NA], True, marks=pytest.mark.xfail(reason="GH-31948")
            ),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
        ],
        ids=["list-False", "list-True", "integer-array-False", "integer-array-True"],
    )
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        arr = data.copy()

        # TODO(xfail) this raises KeyError about labels not found (it tries label-based)
        # for list of labels with Series
        if box_in_series:
            arr = pd.Series(data, index=[chr(100 + i) for i in range(len(data))])

        msg = "Cannot index with an integer indexer containing NA values"
        with pytest.raises(ValueError, match=msg):
            arr[idx] = arr[0]

    @pytest.mark.parametrize("as_callable", [True, False])
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_aligned(self, data, as_callable, setter):
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        if as_callable:
            mask2 = lambda x: mask
        else:
            mask2 = mask

        if setter:
            # loc
            target = getattr(ser, setter)
        else:
            # Series.__setitem__
            target = ser

        target[mask2] = data[5:7]

        ser[mask2] = data[5:7]
        assert ser[0] == data[5]
        assert ser[1] == data[6]

    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        if setter:  # loc
            target = getattr(ser, setter)
        else:  # __setitem__
            target = ser

        target[mask] = data[10]
        assert ser[0] == data[10]
        assert ser[1] == data[10]

    def test_setitem_expand_columns(self, data):
        df = pd.DataFrame({"A": data})
        result = df.copy()
        result["B"] = 1
        expected = pd.DataFrame({"A": data, "B": [1] * len(data)})
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.loc[:, "B"] = 1
        tm.assert_frame_equal(result, expected)

        # overwrite with new type
        result["B"] = data
        expected = pd.DataFrame({"A": data, "B": data})
        tm.assert_frame_equal(result, expected)

    def test_setitem_expand_with_extension(self, data):
        df = pd.DataFrame({"A": [1] * len(data)})
        result = df.copy()
        result["B"] = data
        expected = pd.DataFrame({"A": [1] * len(data), "B": data})
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.loc[:, "B"] = data
        tm.assert_frame_equal(result, expected)

    def test_setitem_frame_invalid_length(self, data):
        df = pd.DataFrame({"A": [1] * len(data)})
        xpr = (
            rf"Length of values \({len(data[:5])}\) "
            rf"does not match length of index \({len(df)}\)"
        )
        with pytest.raises(ValueError, match=xpr):
            df["B"] = data[:5]

    def test_setitem_tuple_index(self, data):
        ser = pd.Series(data[:2], index=[(0, 0), (0, 1)])
        expected = pd.Series(data.take([1, 1]), index=ser.index)
        ser[(0, 0)] = data[1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_slice(self, data, box_in_series):
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        arr[:3] = data[0]
        tm.assert_equal(arr, expected)

    def test_setitem_loc_iloc_slice(self, data):
        arr = data[:5].copy()
        s = pd.Series(arr, index=["a", "b", "c", "d", "e"])
        expected = pd.Series(data.take([0, 0, 0, 3, 4]), index=s.index)

        result = s.copy()
        result.iloc[:3] = data[0]
        tm.assert_equal(result, expected)

        result = s.copy()
        result.loc[:"c"] = data[0]
        tm.assert_equal(result, expected)

    def test_setitem_slice_mismatch_length_raises(self, data):
        arr = data[:5]
        with pytest.raises(ValueError):
            arr[:1] = arr[:2]

    def test_setitem_slice_array(self, data):
        arr = data[:5].copy()
        arr[:5] = data[-5:]
        tm.assert_extension_array_equal(arr, data[-5:])

    def test_setitem_scalar_key_sequence_raise(self, data):
        arr = data[:5].copy()
        with pytest.raises(ValueError):
            arr[0] = arr[[0, 1]]

    def test_setitem_preserves_views(self, data):
        # GH#28150 setitem shouldn't swap the underlying data
        view1 = data.view()
        view2 = data[:]

        data[0] = data[1]
        assert view1[0] == data[1]
        assert view2[0] == data[1]

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer):
        # https://github.com/pandas-dev/pandas/issues/32395
        df = expected = pd.DataFrame({0: pd.Series(data)})
        result = pd.DataFrame(index=df.index)

        key = full_indexer(df)
        result.loc[key, 0] = df[0]

        tm.assert_frame_equal(result, expected)

    def test_setitem_with_expansion_row(self, data, na_value):
        df = pd.DataFrame({"data": data[:1]})

        df.loc[1, "data"] = data[1]
        expected = pd.DataFrame({"data": data[:2]})
        tm.assert_frame_equal(df, expected)

        # https://github.com/pandas-dev/pandas/issues/47284
        df.loc[2, "data"] = na_value
        expected = pd.DataFrame(
            {"data": pd.Series([data[0], data[1], na_value], dtype=data.dtype)}
        )
        tm.assert_frame_equal(df, expected)

    def test_setitem_series(self, data, full_indexer):
        # https://github.com/pandas-dev/pandas/issues/32395
        ser = pd.Series(data, name="data")
        result = pd.Series(index=ser.index, dtype=object, name="data")

        # because result has object dtype, the attempt to do setting inplace
        #  is successful, and object dtype is retained
        key = full_indexer(ser)
        result.loc[key] = ser

        expected = pd.Series(
            data.astype(object), index=ser.index, name="data", dtype=object
        )
        tm.assert_series_equal(result, expected)

    def test_setitem_frame_2d_values(self, data):
        # GH#44514
        df = pd.DataFrame({"A": data})

        # Avoiding using_array_manager fixture
        #  https://github.com/pandas-dev/pandas/pull/44514#discussion_r754002410
        using_array_manager = isinstance(df._mgr, pd.core.internals.ArrayManager)
        using_copy_on_write = pd.options.mode.copy_on_write

        blk_data = df._mgr.arrays[0]

        orig = df.copy()

        df.iloc[:] = df.copy()
        tm.assert_frame_equal(df, orig)

        df.iloc[:-1] = df.iloc[:-1].copy()
        tm.assert_frame_equal(df, orig)

        df.iloc[:] = df.values
        tm.assert_frame_equal(df, orig)
        if not using_array_manager and not using_copy_on_write:
            # GH#33457 Check that this setting occurred in-place
            # FIXME(ArrayManager): this should work there too
            assert df._mgr.arrays[0] is blk_data

        df.iloc[:-1] = df.values[:-1]
        tm.assert_frame_equal(df, orig)

    def test_delitem_series(self, data):
        # GH#40763
        ser = pd.Series(data, name="data")

        taker = np.arange(len(ser))
        taker = np.delete(taker, 1)

        expected = ser[taker]
        del ser[1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_invalid(self, data, invalid_scalar):
        msg = ""  # messages vary by subclass, so we do not test it
        with pytest.raises((ValueError, TypeError), match=msg):
            data[0] = invalid_scalar

        with pytest.raises((ValueError, TypeError), match=msg):
            data[:] = invalid_scalar

    def test_setitem_2d_values(self, data):
        # GH50085
        original = data.copy()
        df = pd.DataFrame({"a": data, "b": data})
        df.loc[[0, 1], :] = df.loc[[1, 0], :].values
        assert (df.loc[0, :] == original[1]).all()
        assert (df.loc[1, :] == original[0]).all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_block_internals.py ===
from datetime import (
    datetime,
    timedelta,
)
import itertools

import numpy as np
import pytest

from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
    Timestamp,
    date_range,
    option_context,
)
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock

# Segregated collection of methods that require the BlockManager internal data
# structure


# TODO(ArrayManager) check which of those tests need to be rewritten to test the
# equivalent for ArrayManager
pytestmark = td.skip_array_manager_invalid_test


class TestDataFrameBlockInternals:
    def test_setitem_invalidates_datetime_index_freq(self):
        # GH#24096 altering a datetime64tz column inplace invalidates the
        #  `freq` attribute on the underlying DatetimeIndex

        dti = date_range("20130101", periods=3, tz="US/Eastern")
        ts = dti[1]

        df = DataFrame({"B": dti})
        assert df["B"]._values.freq is None

        df.iloc[1, 0] = pd.NaT
        assert df["B"]._values.freq is None

        # check that the DatetimeIndex was not altered in place
        assert dti.freq == "D"
        assert dti[1] == ts

    def test_cast_internals(self, float_frame):
        msg = "Passing a BlockManager to DataFrame"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            casted = DataFrame(float_frame._mgr, dtype=int)
        expected = DataFrame(float_frame._series, dtype=int)
        tm.assert_frame_equal(casted, expected)

        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            casted = DataFrame(float_frame._mgr, dtype=np.int32)
        expected = DataFrame(float_frame._series, dtype=np.int32)
        tm.assert_frame_equal(casted, expected)

    def test_consolidate(self, float_frame):
        float_frame["E"] = 7.0
        consolidated = float_frame._consolidate()
        assert len(consolidated._mgr.blocks) == 1

        # Ensure copy, do I want this?
        recons = consolidated._consolidate()
        assert recons is not consolidated
        tm.assert_frame_equal(recons, consolidated)

        float_frame["F"] = 8.0
        assert len(float_frame._mgr.blocks) == 3

        return_value = float_frame._consolidate_inplace()
        assert return_value is None
        assert len(float_frame._mgr.blocks) == 1

    def test_consolidate_inplace(self, float_frame):
        # triggers in-place consolidation
        for letter in range(ord("A"), ord("Z")):
            float_frame[chr(letter)] = chr(letter)

    def test_modify_values(self, float_frame, using_copy_on_write):
        if using_copy_on_write:
            with pytest.raises(ValueError, match="read-only"):
                float_frame.values[5] = 5
            assert (float_frame.values[5] != 5).all()
            return

        float_frame.values[5] = 5
        assert (float_frame.values[5] == 5).all()

        # unconsolidated
        float_frame["E"] = 7.0
        col = float_frame["E"]
        float_frame.values[6] = 6
        # as of 2.0 .values does not consolidate, so subsequent calls to .values
        #  does not share data
        assert not (float_frame.values[6] == 6).all()

        assert (col == 7).all()

    def test_boolean_set_uncons(self, float_frame):
        float_frame["E"] = 7.0

        expected = float_frame.values.copy()
        expected[expected > 1] = 2

        float_frame[float_frame > 1] = 2
        tm.assert_almost_equal(expected, float_frame.values)

    def test_constructor_with_convert(self):
        # this is actually mostly a test of lib.maybe_convert_objects
        # #2845
        df = DataFrame({"A": [2**63 - 1]})
        result = df["A"]
        expected = Series(np.asarray([2**63 - 1], np.int64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2**63]})
        result = df["A"]
        expected = Series(np.asarray([2**63], np.uint64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [datetime(2005, 1, 1), True]})
        result = df["A"]
        expected = Series(
            np.asarray([datetime(2005, 1, 1), True], np.object_), name="A"
        )
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [None, 1]})
        result = df["A"]
        expected = Series(np.asarray([np.nan, 1], np.float64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0, 2]})
        result = df["A"]
        expected = Series(np.asarray([1.0, 2], np.float64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, 3]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, 3], np.complex128), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, 3.0]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, 3.0], np.complex128), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, True]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, True], np.object_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0, None]})
        result = df["A"]
        expected = Series(np.asarray([1.0, np.nan], np.float64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, None]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, np.nan], np.complex128), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2.0, 1, True, None]})
        result = df["A"]
        expected = Series(np.asarray([2.0, 1, True, None], np.object_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2.0, 1, datetime(2006, 1, 1), None]})
        result = df["A"]
        expected = Series(
            np.asarray([2.0, 1, datetime(2006, 1, 1), None], np.object_), name="A"
        )
        tm.assert_series_equal(result, expected)

    def test_construction_with_mixed(self, float_string_frame, using_infer_string):
        # mixed-type frames
        float_string_frame["datetime"] = datetime.now()
        float_string_frame["timedelta"] = timedelta(days=1, seconds=1)
        assert float_string_frame["datetime"].dtype == "M8[us]"
        assert float_string_frame["timedelta"].dtype == "m8[us]"
        result = float_string_frame.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan),
                np.dtype("datetime64[us]"),
                np.dtype("timedelta64[us]"),
            ],
            index=list("ABCD") + ["foo", "datetime", "timedelta"],
        )
        tm.assert_series_equal(result, expected)

    def test_construction_with_conversions(self):
        # convert from a numpy array of non-ns timedelta64; as of 2.0 this does
        #  *not* convert
        arr = np.array([1, 2, 3], dtype="timedelta64[s]")
        df = DataFrame({"A": arr})
        expected = DataFrame(
            {"A": pd.timedelta_range("00:00:01", periods=3, freq="s")}, index=range(3)
        )
        tm.assert_numpy_array_equal(df["A"].to_numpy(), arr)

        expected = DataFrame(
            {
                "dt1": Timestamp("20130101"),
                "dt2": date_range("20130101", periods=3).astype("M8[s]"),
                # 'dt3' : date_range('20130101 00:00:01',periods=3,freq='s'),
                # FIXME: don't leave commented-out
            },
            index=range(3),
        )
        assert expected.dtypes["dt1"] == "M8[s]"
        assert expected.dtypes["dt2"] == "M8[s]"

        dt1 = np.datetime64("2013-01-01")
        dt2 = np.array(
            ["2013-01-01", "2013-01-02", "2013-01-03"], dtype="datetime64[D]"
        )
        df = DataFrame({"dt1": dt1, "dt2": dt2})

        # df['dt3'] = np.array(['2013-01-01 00:00:01','2013-01-01
        # 00:00:02','2013-01-01 00:00:03'],dtype='datetime64[s]')
        # FIXME: don't leave commented-out

        tm.assert_frame_equal(df, expected)

    def test_constructor_compound_dtypes(self):
        # GH 5191
        # compound dtypes should raise not-implementederror

        def f(dtype):
            data = list(itertools.repeat((datetime(2001, 1, 1), "aa", 20), 9))
            return DataFrame(data=data, columns=["A", "B", "C"], dtype=dtype)

        msg = "compound dtypes are not implemented in the DataFrame constructor"
        with pytest.raises(NotImplementedError, match=msg):
            f([("A", "datetime64[h]"), ("B", "str"), ("C", "int32")])

        # pre-2.0 these used to work (though results may be unexpected)
        with pytest.raises(TypeError, match="argument must be"):
            f("int64")
        with pytest.raises(TypeError, match="argument must be"):
            f("float64")

        # 10822
        msg = "^Unknown datetime string format, unable to parse: aa, at position 0$"
        with pytest.raises(ValueError, match=msg):
            f("M8[ns]")

    def test_pickle(self, float_string_frame, timezone_frame):
        empty_frame = DataFrame()

        unpickled = tm.round_trip_pickle(float_string_frame)
        tm.assert_frame_equal(float_string_frame, unpickled)

        # buglet
        float_string_frame._mgr.ndim

        # empty
        unpickled = tm.round_trip_pickle(empty_frame)
        repr(unpickled)

        # tz frame
        unpickled = tm.round_trip_pickle(timezone_frame)
        tm.assert_frame_equal(timezone_frame, unpickled)

    def test_consolidate_datetime64(self):
        # numpy vstack bug

        df = DataFrame(
            {
                "starting": pd.to_datetime(
                    [
                        "2012-06-21 00:00",
                        "2012-06-23 07:00",
                        "2012-06-23 16:30",
                        "2012-06-25 08:00",
                        "2012-06-26 12:00",
                    ]
                ),
                "ending": pd.to_datetime(
                    [
                        "2012-06-23 07:00",
                        "2012-06-23 16:30",
                        "2012-06-25 08:00",
                        "2012-06-26 12:00",
                        "2012-06-27 08:00",
                    ]
                ),
                "measure": [77, 65, 77, 0, 77],
            }
        )

        ser_starting = df.starting
        ser_starting.index = ser_starting.values
        ser_starting = ser_starting.tz_localize("US/Eastern")
        ser_starting = ser_starting.tz_convert("UTC")
        ser_starting.index.name = "starting"

        ser_ending = df.ending
        ser_ending.index = ser_ending.values
        ser_ending = ser_ending.tz_localize("US/Eastern")
        ser_ending = ser_ending.tz_convert("UTC")
        ser_ending.index.name = "ending"

        df.starting = ser_starting.index
        df.ending = ser_ending.index

        tm.assert_index_equal(pd.DatetimeIndex(df.starting), ser_starting.index)
        tm.assert_index_equal(pd.DatetimeIndex(df.ending), ser_ending.index)

    def test_is_mixed_type(self, float_frame, float_string_frame):
        assert not float_frame._is_mixed_type
        assert float_string_frame._is_mixed_type

    def test_stale_cached_series_bug_473(self, using_copy_on_write, warn_copy_on_write):
        # this is chained, but ok
        with option_context("chained_assignment", None):
            Y = DataFrame(
                np.random.default_rng(2).random((4, 4)),
                index=("a", "b", "c", "d"),
                columns=("e", "f", "g", "h"),
            )
            repr(Y)
            Y["e"] = Y["e"].astype("object")
            with tm.raises_chained_assignment_error():
                Y["g"]["c"] = np.nan
            repr(Y)
            Y.sum()
            Y["g"].sum()
            if using_copy_on_write:
                assert not pd.isna(Y["g"]["c"])
            else:
                assert pd.isna(Y["g"]["c"])

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_strange_column_corruption_issue(self, using_copy_on_write):
        # TODO(wesm): Unclear how exactly this is related to internal matters
        df = DataFrame(index=[0, 1])
        df[0] = np.nan
        wasCol = {}

        with tm.assert_produces_warning(
            PerformanceWarning, raise_on_extra_warnings=False
        ):
            for i, dt in enumerate(df.index):
                for col in range(100, 200):
                    if col not in wasCol:
                        wasCol[col] = 1
                        df[col] = np.nan
                    if using_copy_on_write:
                        df.loc[dt, col] = i
                    else:
                        df[col][dt] = i

        myid = 100

        first = len(df.loc[pd.isna(df[myid]), [myid]])
        second = len(df.loc[pd.isna(df[myid]), [myid]])
        assert first == second == 0

    def test_constructor_no_pandas_array(self):
        # Ensure that NumpyExtensionArray isn't allowed inside Series
        # See https://github.com/pandas-dev/pandas/issues/23995 for more.
        arr = Series([1, 2, 3]).array
        result = DataFrame({"A": arr})
        expected = DataFrame({"A": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)
        assert isinstance(result._mgr.blocks[0], NumpyBlock)
        assert result._mgr.blocks[0].is_numeric

    def test_add_column_with_pandas_array(self):
        # GH 26390
        df = DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
        df["c"] = pd.arrays.NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))
        df2 = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": ["a", "b", "c", "d"],
                "c": pd.arrays.NumpyExtensionArray(
                    np.array([1, 2, None, 3], dtype=object)
                ),
            }
        )
        assert type(df["c"]._mgr.blocks[0]) == NumpyBlock
        assert df["c"]._mgr.blocks[0].is_object
        assert type(df2["c"]._mgr.blocks[0]) == NumpyBlock
        assert df2["c"]._mgr.blocks[0].is_object
        tm.assert_frame_equal(df, df2)


def test_update_inplace_sets_valid_block_values(using_copy_on_write):
    # https://github.com/pandas-dev/pandas/issues/33457
    df = DataFrame({"a": Series([1, 2, None], dtype="category")})

    # inplace update of a single column
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["a"].fillna(1, inplace=True)
    else:
        with tm.assert_produces_warning(FutureWarning, match="inplace method"):
            df["a"].fillna(1, inplace=True)

    # check we haven't put a Series into any block.values
    assert isinstance(df._mgr.blocks[0].values, Categorical)

    if not using_copy_on_write:
        # smoketest for OP bug from GH#35731
        assert df.isnull().sum().sum() == 0


def test_nonconsolidated_item_cache_take():
    # https://github.com/pandas-dev/pandas/issues/35521

    # create non-consolidated dataframe with object dtype columns
    df = DataFrame(
        {
            "col1": Series(["a"], dtype=object),
        }
    )
    df["col2"] = Series([0], dtype=object)
    assert not df._mgr.is_consolidated()

    # access column (item cache)
    df["col1"] == "A"
    # take operation
    # (regression was that this consolidated but didn't reset item cache,
    # resulting in an invalid cache and the .at operation not working properly)
    df[df["col2"] == 0]

    # now setting value should update actual dataframe
    df.at[0, "col1"] = "A"

    expected = DataFrame({"col1": ["A"], "col2": [0]}, dtype=object)
    tm.assert_frame_equal(df, expected)
    assert df.at[0, "col1"] == "A"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_from_dummies.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    from_dummies,
    get_dummies,
)
import pandas._testing as tm


@pytest.fixture
def dummies_basic():
    return DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )


@pytest.fixture
def dummies_with_unassigned():
    return DataFrame(
        {
            "col1_a": [1, 0, 0],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [0, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )


def test_error_wrong_data_type():
    dummies = [0, 1, 0]
    with pytest.raises(
        TypeError,
        match=r"Expected 'data' to be a 'DataFrame'; Received 'data' of type: list",
    ):
        from_dummies(dummies)


def test_error_no_prefix_contains_unassigned():
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0]})
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains unassigned value\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies)


def test_error_no_prefix_wrong_default_category_type():
    dummies = DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; "
            r"Received 'default_category' of type: list"
        ),
    ):
        from_dummies(dummies, default_category=["c", "d"])


def test_error_no_prefix_multi_assignment():
    dummies = DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains multi-assignment\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies)


def test_error_no_prefix_contains_nan():
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, np.nan]})
    with pytest.raises(
        ValueError, match=r"Dummy DataFrame contains NA value in column: 'b'"
    ):
        from_dummies(dummies)


def test_error_contains_non_dummies():
    dummies = DataFrame(
        {"a": [1, 6, 3, 1], "b": [0, 1, 0, 2], "c": ["c1", "c2", "c3", "c4"]}
    )
    with pytest.raises(
        TypeError,
        match=r"Passed DataFrame contains non-dummy data",
    ):
        from_dummies(dummies)


def test_error_with_prefix_multiple_seperators():
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2-a": [0, 1, 0],
            "col2-b": [1, 0, 1],
        },
    )
    with pytest.raises(
        ValueError,
        match=(r"Separator not specified for column: col2-a"),
    ):
        from_dummies(dummies, sep="_")


def test_error_with_prefix_sep_wrong_type(dummies_basic):
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'sep' to be of type 'str' or 'None'; "
            r"Received 'sep' of type: list"
        ),
    ):
        from_dummies(dummies_basic, sep=["_"])


def test_error_with_prefix_contains_unassigned(dummies_with_unassigned):
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains unassigned value\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_")


def test_error_with_prefix_default_category_wrong_type(dummies_with_unassigned):
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; "
            r"Received 'default_category' of type: list"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_", default_category=["x", "y"])


def test_error_with_prefix_default_category_dict_not_complete(
    dummies_with_unassigned,
):
    with pytest.raises(
        ValueError,
        match=(
            r"Length of 'default_category' \(1\) did not match "
            r"the length of the columns being encoded \(2\)"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_", default_category={"col1": "x"})


def test_error_with_prefix_contains_nan(dummies_basic):
    # Set float64 dtype to avoid upcast when setting np.nan
    dummies_basic["col2_c"] = dummies_basic["col2_c"].astype("float64")
    dummies_basic.loc[2, "col2_c"] = np.nan
    with pytest.raises(
        ValueError, match=r"Dummy DataFrame contains NA value in column: 'col2_c'"
    ):
        from_dummies(dummies_basic, sep="_")


def test_error_with_prefix_contains_non_dummies(dummies_basic):
    # Set object dtype to avoid upcast when setting "str"
    dummies_basic["col2_c"] = dummies_basic["col2_c"].astype(object)
    dummies_basic.loc[2, "col2_c"] = "str"
    with pytest.raises(TypeError, match=r"Passed DataFrame contains non-dummy data"):
        from_dummies(dummies_basic, sep="_")


def test_error_with_prefix_double_assignment():
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [1, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains multi-assignment\(s\); "
            r"First instance in row: 0"
        ),
    ):
        from_dummies(dummies, sep="_")


def test_roundtrip_series_to_dataframe():
    categories = Series(["a", "b", "c", "a"])
    dummies = get_dummies(categories)
    result = from_dummies(dummies)
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    tm.assert_frame_equal(result, expected)


def test_roundtrip_single_column_dataframe():
    categories = DataFrame({"": ["a", "b", "c", "a"]})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep="_")
    expected = categories
    tm.assert_frame_equal(result, expected)


def test_roundtrip_with_prefixes():
    categories = DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "a", "c"]})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep="_")
    expected = categories
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic():
    dummies = DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic_bool_values():
    dummies = DataFrame(
        {
            "a": [True, False, False, True],
            "b": [False, True, False, False],
            "c": [False, False, True, False],
        }
    )
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic_mixed_bool_values():
    dummies = DataFrame(
        {"a": [1, 0, 0, 1], "b": [False, True, False, False], "c": [0, 0, 1, 0]}
    )
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_int_cats_basic():
    dummies = DataFrame(
        {1: [1, 0, 0, 0], 25: [0, 1, 0, 0], 2: [0, 0, 1, 0], 5: [0, 0, 0, 1]}
    )
    expected = DataFrame({"": [1, 25, 2, 5]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_float_cats_basic():
    dummies = DataFrame(
        {1.0: [1, 0, 0, 0], 25.0: [0, 1, 0, 0], 2.5: [0, 0, 1, 0], 5.84: [0, 0, 0, 1]}
    )
    expected = DataFrame({"": [1.0, 25.0, 2.5, 5.84]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_mixed_cats_basic():
    dummies = DataFrame(
        {
            1.23: [1, 0, 0, 0, 0],
            "c": [0, 1, 0, 0, 0],
            2: [0, 0, 1, 0, 0],
            False: [0, 0, 0, 1, 0],
            None: [0, 0, 0, 0, 1],
        }
    )
    expected = DataFrame({"": [1.23, "c", 2, False, None]}, dtype="object")
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_contains_get_dummies_NaN_column():
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0], "NaN": [0, 0, 1]})
    expected = DataFrame({"": ["a", "b", "NaN"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "default_category, expected",
    [
        pytest.param(
            "c",
            DataFrame({"": ["a", "b", "c"]}),
            id="default_category is a str",
        ),
        pytest.param(
            1,
            DataFrame({"": ["a", "b", 1]}),
            id="default_category is a int",
        ),
        pytest.param(
            1.25,
            DataFrame({"": ["a", "b", 1.25]}),
            id="default_category is a float",
        ),
        pytest.param(
            0,
            DataFrame({"": ["a", "b", 0]}),
            id="default_category is a 0",
        ),
        pytest.param(
            False,
            DataFrame({"": ["a", "b", False]}),
            id="default_category is a bool",
        ),
        pytest.param(
            (1, 2),
            DataFrame({"": ["a", "b", (1, 2)]}),
            id="default_category is a tuple",
        ),
    ],
)
def test_no_prefix_string_cats_default_category(
    default_category, expected, using_infer_string
):
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0]})
    result = from_dummies(dummies, default_category=default_category)
    if using_infer_string:
        expected[""] = expected[""].astype("str")
    tm.assert_frame_equal(result, expected)


def test_with_prefix_basic(dummies_basic):
    expected = DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "a", "c"]})
    result = from_dummies(dummies_basic, sep="_")
    tm.assert_frame_equal(result, expected)


def test_with_prefix_contains_get_dummies_NaN_column():
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 0],
            "col1_b": [0, 1, 0],
            "col1_NaN": [0, 0, 1],
            "col2_a": [0, 1, 0],
            "col2_b": [0, 0, 0],
            "col2_c": [0, 0, 1],
            "col2_NaN": [1, 0, 0],
        },
    )
    expected = DataFrame({"col1": ["a", "b", "NaN"], "col2": ["NaN", "a", "c"]})
    result = from_dummies(dummies, sep="_")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "default_category, expected",
    [
        pytest.param(
            "x",
            DataFrame({"col1": ["a", "b", "x"], "col2": ["x", "a", "c"]}),
            id="default_category is a str",
        ),
        pytest.param(
            0,
            DataFrame({"col1": ["a", "b", 0], "col2": [0, "a", "c"]}),
            id="default_category is a 0",
        ),
        pytest.param(
            False,
            DataFrame({"col1": ["a", "b", False], "col2": [False, "a", "c"]}),
            id="default_category is a False",
        ),
        pytest.param(
            {"col2": 1, "col1": 2.5},
            DataFrame({"col1": ["a", "b", 2.5], "col2": [1, "a", "c"]}),
            id="default_category is a dict with int and float values",
        ),
        pytest.param(
            {"col2": None, "col1": False},
            DataFrame({"col1": ["a", "b", False], "col2": [None, "a", "c"]}),
            id="default_category is a dict with bool and None values",
        ),
        pytest.param(
            {"col2": (1, 2), "col1": [1.25, False]},
            DataFrame({"col1": ["a", "b", [1.25, False]], "col2": [(1, 2), "a", "c"]}),
            id="default_category is a dict with list and tuple values",
        ),
    ],
)
def test_with_prefix_default_category(
    dummies_with_unassigned, default_category, expected, using_infer_string
):
    result = from_dummies(
        dummies_with_unassigned, sep="_", default_category=default_category
    )
    if using_infer_string:
        expected = expected.astype("str")
    tm.assert_frame_equal(result, expected)


def test_ea_categories():
    # GH 54300
    df = DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})
    df.columns = df.columns.astype("string[python]")
    result = from_dummies(df)
    expected = DataFrame({"": Series(list("abca"), dtype="string[python]")})
    tm.assert_frame_equal(result, expected)


def test_ea_categories_with_sep():
    # GH 54300
    df = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        }
    )
    df.columns = df.columns.astype("string[python]")
    result = from_dummies(df, sep="_")
    expected = DataFrame(
        {
            "col1": Series(list("aba"), dtype="string[python]"),
            "col2": Series(list("bac"), dtype="string[python]"),
        }
    )
    expected.columns = expected.columns.astype("string[python]")
    tm.assert_frame_equal(result, expected)


def test_maintain_original_index():
    # GH 54300
    df = DataFrame(
        {"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]}, index=list("abcd")
    )
    result = from_dummies(df)
    expected = DataFrame({"": list("abca")}, index=list("abcd"))
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_leaks.py ===
# -*- coding: utf-8 -*-
"""
Testing scenarios that may have leaked.
"""
from __future__ import print_function, absolute_import, division

import sys
import gc

import time
import weakref
import threading


import greenlet
from . import TestCase
from . import PY314
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX

# pylint:disable=protected-access

assert greenlet.GREENLET_USE_GC # Option to disable this was removed in 1.0

class HasFinalizerTracksInstances(object):
    EXTANT_INSTANCES = set()
    def __init__(self, msg):
        self.msg = sys.intern(msg)
        self.EXTANT_INSTANCES.add(id(self))
    def __del__(self):
        self.EXTANT_INSTANCES.remove(id(self))
    def __repr__(self):
        return "<HasFinalizerTracksInstances at 0x%x %r>" % (
            id(self), self.msg
        )
    @classmethod
    def reset(cls):
        cls.EXTANT_INSTANCES.clear()


class TestLeaks(TestCase):

    def test_arg_refs(self):
        args = ('a', 'b', 'c')
        refcount_before = sys.getrefcount(args)
        # pylint:disable=unnecessary-lambda
        g = greenlet.greenlet(
            lambda *args: greenlet.getcurrent().parent.switch(*args))
        for _ in range(100):
            g.switch(*args)
        self.assertEqual(sys.getrefcount(args), refcount_before)

    def test_kwarg_refs(self):
        kwargs = {}
        self.assertEqual(sys.getrefcount(kwargs), 2 if not PY314 else 1)
        # pylint:disable=unnecessary-lambda
        g = greenlet.greenlet(
            lambda **gkwargs: greenlet.getcurrent().parent.switch(**gkwargs))
        for _ in range(100):
            g.switch(**kwargs)
        # Python 3.14 elides reference counting operations
        # in some cases. See https://github.com/python/cpython/pull/130708
        self.assertEqual(sys.getrefcount(kwargs), 2 if not PY314 else 1)


    @staticmethod
    def __recycle_threads():
        # By introducing a thread that does sleep we allow other threads,
        # that have triggered their __block condition, but did not have a
        # chance to deallocate their thread state yet, to finally do so.
        # The way it works is by requiring a GIL switch (different thread),
        # which does a GIL release (sleep), which might do a GIL switch
        # to finished threads and allow them to clean up.
        def worker():
            time.sleep(0.001)
        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.001)
        t.join(10)

    def test_threaded_leak(self):
        gg = []
        def worker():
            # only main greenlet present
            gg.append(weakref.ref(greenlet.getcurrent()))
        for _ in range(2):
            t = threading.Thread(target=worker)
            t.start()
            t.join(10)
            del t
        greenlet.getcurrent() # update ts_current
        self.__recycle_threads()
        greenlet.getcurrent() # update ts_current
        gc.collect()
        greenlet.getcurrent() # update ts_current
        for g in gg:
            self.assertIsNone(g())

    def test_threaded_adv_leak(self):
        gg = []
        def worker():
            # main and additional *finished* greenlets
            ll = greenlet.getcurrent().ll = []
            def additional():
                ll.append(greenlet.getcurrent())
            for _ in range(2):
                greenlet.greenlet(additional).switch()
            gg.append(weakref.ref(greenlet.getcurrent()))
        for _ in range(2):
            t = threading.Thread(target=worker)
            t.start()
            t.join(10)
            del t
        greenlet.getcurrent() # update ts_current
        self.__recycle_threads()
        greenlet.getcurrent() # update ts_current
        gc.collect()
        greenlet.getcurrent() # update ts_current
        for g in gg:
            self.assertIsNone(g())

    def assertClocksUsed(self):
        used = greenlet._greenlet.get_clocks_used_doing_optional_cleanup()
        self.assertGreaterEqual(used, 0)
        # we don't lose the value
        greenlet._greenlet.enable_optional_cleanup(True)
        used2 = greenlet._greenlet.get_clocks_used_doing_optional_cleanup()
        self.assertEqual(used, used2)
        self.assertGreater(greenlet._greenlet.CLOCKS_PER_SEC, 1)

    def _check_issue251(self,
                        manually_collect_background=True,
                        explicit_reference_to_switch=False):
        # See https://github.com/python-greenlet/greenlet/issues/251
        # Killing a greenlet (probably not the main one)
        # in one thread from another thread would
        # result in leaking a list (the ts_delkey list).
        # We no longer use lists to hold that stuff, though.

        # For the test to be valid, even empty lists have to be tracked by the
        # GC

        assert gc.is_tracked([])
        HasFinalizerTracksInstances.reset()
        greenlet.getcurrent()
        greenlets_before = self.count_objects(greenlet.greenlet, exact_kind=False)

        background_glet_running = threading.Event()
        background_glet_killed = threading.Event()
        background_greenlets = []

        # XXX: Switching this to a greenlet subclass that overrides
        # run results in all callers failing the leaktest; that
        # greenlet instance is leaked. There's a bound method for
        # run() living on the stack of the greenlet in g_initialstub,
        # and since we don't manually switch back to the background
        # greenlet to let it "fall off the end" and exit the
        # g_initialstub function, it never gets cleaned up. Making the
        # garbage collector aware of this bound method (making it an
        # attribute of the greenlet structure and traversing into it)
        # doesn't help, for some reason.
        def background_greenlet():
            # Throw control back to the main greenlet.
            jd = HasFinalizerTracksInstances("DELETING STACK OBJECT")
            greenlet._greenlet.set_thread_local(
                'test_leaks_key',
                HasFinalizerTracksInstances("DELETING THREAD STATE"))
            # Explicitly keeping 'switch' in a local variable
            # breaks this test in all versions
            if explicit_reference_to_switch:
                s = greenlet.getcurrent().parent.switch
                s([jd])
            else:
                greenlet.getcurrent().parent.switch([jd])

        bg_main_wrefs = []

        def background_thread():
            glet = greenlet.greenlet(background_greenlet)
            bg_main_wrefs.append(weakref.ref(glet.parent))

            background_greenlets.append(glet)
            glet.switch() # Be sure it's active.
            # Control is ours again.
            del glet # Delete one reference from the thread it runs in.
            background_glet_running.set()
            background_glet_killed.wait(10)

            # To trigger the background collection of the dead
            # greenlet, thus clearing out the contents of the list, we
            # need to run some APIs. See issue 252.
            if manually_collect_background:
                greenlet.getcurrent()


        t = threading.Thread(target=background_thread)
        t.start()
        background_glet_running.wait(10)
        greenlet.getcurrent()
        lists_before = self.count_objects(list, exact_kind=True)

        assert len(background_greenlets) == 1
        self.assertFalse(background_greenlets[0].dead)
        # Delete the last reference to the background greenlet
        # from a different thread. This puts it in the background thread's
        # ts_delkey list.
        del background_greenlets[:]
        background_glet_killed.set()

        # Now wait for the background thread to die.
        t.join(10)
        del t
        # As part of the fix for 252, we need to cycle the ceval.c
        # interpreter loop to be sure it has had a chance to process
        # the pending call.
        self.wait_for_pending_cleanups()

        lists_after = self.count_objects(list, exact_kind=True)
        greenlets_after = self.count_objects(greenlet.greenlet, exact_kind=False)

        # On 2.7, we observe that lists_after is smaller than
        # lists_before. No idea what lists got cleaned up. All the
        # Python 3 versions match exactly.
        self.assertLessEqual(lists_after, lists_before)
        # On versions after 3.6, we've successfully cleaned up the
        # greenlet references thanks to the internal "vectorcall"
        # protocol; prior to that, there is a reference path through
        # the ``greenlet.switch`` method still on the stack that we
        # can't reach to clean up. The C code goes through terrific
        # lengths to clean that up.
        if not explicit_reference_to_switch \
           and greenlet._greenlet.get_clocks_used_doing_optional_cleanup() is not None:
            # If cleanup was disabled, though, we may not find it.
            self.assertEqual(greenlets_after, greenlets_before)
            if manually_collect_background:
                # TODO: Figure out how to make this work!
                # The one on the stack is still leaking somehow
                # in the non-manually-collect state.
                self.assertEqual(HasFinalizerTracksInstances.EXTANT_INSTANCES, set())
        else:
            # The explicit reference prevents us from collecting it
            # and it isn't always found by the GC either for some
            # reason. The entire frame is leaked somehow, on some
            # platforms (e.g., MacPorts builds of Python (all
            # versions!)), but not on other platforms (the linux and
            # windows builds on GitHub actions and Appveyor). So we'd
            # like to write a test that proves that the main greenlet
            # sticks around, and we can on my machine (macOS 11.6,
            # MacPorts builds of everything) but we can't write that
            # same test on other platforms. However, hopefully iteration
            # done by leakcheck will find it.
            pass

        if greenlet._greenlet.get_clocks_used_doing_optional_cleanup() is not None:
            self.assertClocksUsed()

    def test_issue251_killing_cross_thread_leaks_list(self):
        self._check_issue251()

    def test_issue251_with_cleanup_disabled(self):
        greenlet._greenlet.enable_optional_cleanup(False)
        try:
            self._check_issue251()
        finally:
            greenlet._greenlet.enable_optional_cleanup(True)

    @fails_leakcheck
    def test_issue251_issue252_need_to_collect_in_background(self):
        # Between greenlet 1.1.2 and the next version, this was still
        # failing because the leak of the list still exists when we
        # don't call a greenlet API before exiting the thread. The
        # proximate cause is that neither of the two greenlets from
        # the background thread are actually being destroyed, even
        # though the GC is in fact visiting both objects. It's not
        # clear where that leak is? For some reason the thread-local
        # dict holding it isn't being cleaned up.
        #
        # The leak, I think, is in the CPYthon internal function that
        # calls into green_switch(). The argument tuple is still on
        # the C stack somewhere and can't be reached? That doesn't
        # make sense, because the tuple should be collectable when
        # this object goes away.
        #
        # Note that this test sometimes spuriously passes on Linux,
        # for some reason, but I've never seen it pass on macOS.
        self._check_issue251(manually_collect_background=False)

    @fails_leakcheck
    def test_issue251_issue252_need_to_collect_in_background_cleanup_disabled(self):
        self.expect_greenlet_leak = True
        greenlet._greenlet.enable_optional_cleanup(False)
        try:
            self._check_issue251(manually_collect_background=False)
        finally:
            greenlet._greenlet.enable_optional_cleanup(True)

    @fails_leakcheck
    def test_issue251_issue252_explicit_reference_not_collectable(self):
        self._check_issue251(
            manually_collect_background=False,
            explicit_reference_to_switch=True)

    UNTRACK_ATTEMPTS = 100

    def _only_test_some_versions(self):
        # We're only looking for this problem specifically on 3.11,
        # and this set of tests is relatively fragile, depending on
        # OS and memory management details. So we want to run it on 3.11+
        # (obviously) but not every older 3.x version in order to reduce
        # false negatives. At the moment, those false results seem to have
        # resolved, so we are actually running this on 3.8+
        assert sys.version_info[0] >= 3
        if sys.version_info[:2] < (3, 8):
            self.skipTest('Only observed on 3.11')
        if RUNNING_ON_MANYLINUX:
            self.skipTest("Slow and not worth repeating here")

    @ignores_leakcheck
    # Because we're just trying to track raw memory, not objects, and running
    # the leakcheck makes an already slow test slower.
    def test_untracked_memory_doesnt_increase(self):
        # See https://github.com/gevent/gevent/issues/1924
        # and https://github.com/python-greenlet/greenlet/issues/328
        self._only_test_some_versions()
        def f():
            return 1

        ITER = 10000
        def run_it():
            for _ in range(ITER):
                greenlet.greenlet(f).switch()

        # Establish baseline
        for _ in range(3):
            run_it()

        # uss: (Linux, macOS, Windows): aka "Unique Set Size", this is
        # the memory which is unique to a process and which would be
        # freed if the process was terminated right now.
        uss_before = self.get_process_uss()

        for count in range(self.UNTRACK_ATTEMPTS):
            uss_before = max(uss_before, self.get_process_uss())
            run_it()

            uss_after = self.get_process_uss()
            if uss_after <= uss_before and count > 1:
                break

        self.assertLessEqual(uss_after, uss_before)

    def _check_untracked_memory_thread(self, deallocate_in_thread=True):
        self._only_test_some_versions()
        # Like the above test, but what if there are a bunch of
        # unfinished greenlets in a thread that dies?
        # Does it matter if we deallocate in the thread or not?
        EXIT_COUNT = [0]

        def f():
            try:
                greenlet.getcurrent().parent.switch()
            except greenlet.GreenletExit:
                EXIT_COUNT[0] += 1
                raise
            return 1

        ITER = 10000
        def run_it():
            glets = []
            for _ in range(ITER):
                # Greenlet starts, switches back to us.
                # We keep a strong reference to the greenlet though so it doesn't
                # get a GreenletExit exception.
                g = greenlet.greenlet(f)
                glets.append(g)
                g.switch()

            return glets

        test = self

        class ThreadFunc:
            uss_before = uss_after = 0
            glets = ()
            ITER = 2
            def __call__(self):
                self.uss_before = test.get_process_uss()

                for _ in range(self.ITER):
                    self.glets += tuple(run_it())

                for g in self.glets:
                    test.assertIn('suspended active', str(g))
                # Drop them.
                if deallocate_in_thread:
                    self.glets = ()
                self.uss_after = test.get_process_uss()

        # Establish baseline
        uss_before = uss_after = None
        for count in range(self.UNTRACK_ATTEMPTS):
            EXIT_COUNT[0] = 0
            thread_func = ThreadFunc()
            t = threading.Thread(target=thread_func)
            t.start()
            t.join(30)
            self.assertFalse(t.is_alive())

            if uss_before is None:
                uss_before = thread_func.uss_before

            uss_before = max(uss_before, thread_func.uss_before)
            if deallocate_in_thread:
                self.assertEqual(thread_func.glets, ())
                self.assertEqual(EXIT_COUNT[0], ITER * thread_func.ITER)

            del thread_func # Deallocate the greenlets; but this won't raise into them
            del t
            if not deallocate_in_thread:
                self.assertEqual(EXIT_COUNT[0], 0)
            if deallocate_in_thread:
                self.wait_for_pending_cleanups()

            uss_after = self.get_process_uss()
            # See if we achieve a non-growth state at some point. Break when we do.
            if uss_after <= uss_before and count > 1:
                break

        self.wait_for_pending_cleanups()
        uss_after = self.get_process_uss()
        self.assertLessEqual(uss_after, uss_before, "after attempts %d" % (count,))

    @ignores_leakcheck
    # Because we're just trying to track raw memory, not objects, and running
    # the leakcheck makes an already slow test slower.
    def test_untracked_memory_doesnt_increase_unfinished_thread_dealloc_in_thread(self):
        self._check_untracked_memory_thread(deallocate_in_thread=True)

    @ignores_leakcheck
    # Because the main greenlets from the background threads do not exit in a timely fashion,
    # we fail the object-based leakchecks.
    def test_untracked_memory_doesnt_increase_unfinished_thread_dealloc_in_main(self):
        self._check_untracked_memory_thread(deallocate_in_thread=False)

if __name__ == '__main__':
    __import__('unittest').main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ctypeslib\__init__.py ===
from ._ctypeslib import (
    __all__,
    __doc__,
    _concrete_ndptr,
    _ndptr,
    as_array,
    as_ctypes,
    as_ctypes_type,
    c_intp,
    ctypes,
    load_library,
    ndpointer,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\cversions.py ===
"""Simple script to compute the api hash of the current API.

The API has is defined by numpy_api_order and ufunc_api_order.

"""
from os.path import dirname

from code_generators.genapi import fullapi_hash
from code_generators.numpy_api import full_api

if __name__ == '__main__':
    curdir = dirname(__file__)
    print(fullapi_hash(full_api))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\_constants.py ===
# Copyright (c) 2010-2024 openpyxl

"""
Package metadata
"""

__author__ = "See AUTHORS"
__author_email__ = "charlie.clark@clark-consulting.eu"
__license__ = "MIT"
__maintainer_email__ = "openpyxl-users@googlegroups.com"
__url__ = "https://openpyxl.readthedocs.io"
__version__ = "3.1.5"
__python__ = "3.8"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\__init__.py ===
# ruff: noqa: TCH004
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import modules that have public classes/functions
    from pandas.io import (
        formats,
        json,
        stata,
    )

    # mark only those modules as public
    __all__ = ["formats", "json", "stata"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\__init__.py ===
from typing import List, Optional

__version__ = "25.1.1"


def main(args: Optional[List[str]] = None) -> int:
    """This is an internal API only meant for use by pip's own console scripts.

    For additional details, see https://github.com/pypa/pip/issues/7498.
    """
    from pip._internal.utils.entrypoints import _wrapper

    return _wrapper(args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\dependency_groups\__init__.py ===
from ._implementation import (
    CyclicDependencyError,
    DependencyGroupInclude,
    DependencyGroupResolver,
    resolve,
)

__all__ = (
    "CyclicDependencyError",
    "DependencyGroupInclude",
    "DependencyGroupResolver",
    "resolve",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\handlers\__init__.py ===
"""
Multipledispatch handlers for ``Predicate`` are implemented here.
Handlers in this module are not directly imported to other modules in
order to avoid circular import problem.
"""

from .common import (AskHandler, CommonHandler,
    test_closed_group)

__all__ = [
    'AskHandler', 'CommonHandler',
    'test_closed_group'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\relation\__init__.py ===
"""
A module to implement finitary relations [1] as predicate.

References
==========

.. [1] https://en.wikipedia.org/wiki/Finitary_relation

"""

__all__ = ['BinaryRelation', 'AppliedBinaryRelation']

from .binrel import BinaryRelation, AppliedBinaryRelation

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\benchmarks\bench_trigintegrate.py ===
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.trigonometry import trigintegrate

x = Symbol('x')


def timeit_trigintegrate_sin3x():
    trigintegrate(sin(x)**3, x)


def timeit_trigintegrate_x2():
    trigintegrate(x**2, x)  # -> None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\mutable_ndim_array.py ===
from sympy.tensor.array.ndim_array import NDimArray


class MutableNDimArray(NDimArray):

    def as_immutable(self):
        raise NotImplementedError("abstract method")

    def as_mutable(self):
        return self

    def _sympy_(self):
        return self.as_immutable()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\runtests.py ===
"""
.. deprecated:: 1.6

   sympy.utilities.runtests has been renamed to sympy.testing.runtests.
"""

from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("The sympy.utilities.runtests submodule is deprecated. Use sympy.testing.runtests instead.",
    deprecated_since_version="1.6",
    active_deprecations_target="deprecated-sympy-utilities-submodules")

from sympy.testing.runtests import * # noqa: F401,F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\from_thread.py ===
"""
This namespace represents special functions that can call back into Trio from
an external thread by means of a Trio Token present in Thread Local Storage
"""

from ._threads import (
    from_thread_check_cancelled as check_cancelled,
    from_thread_run as run,
    from_thread_run_sync as run_sync,
)

# need to use __all__ for pyright --verifytypes to see re-exports when renaming them
__all__ = ["check_cancelled", "run", "run_sync"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_tutil.py ===
import pytest

from .tutil import check_sequence_matches


def test_check_sequence_matches() -> None:
    check_sequence_matches([1, 2, 3], [1, 2, 3])
    with pytest.raises(AssertionError):
        check_sequence_matches([1, 3, 2], [1, 2, 3])
    check_sequence_matches([1, 2, 3, 4], [1, {2, 3}, 4])
    check_sequence_matches([1, 3, 2, 4], [1, {2, 3}, 4])
    with pytest.raises(AssertionError):
        check_sequence_matches([1, 2, 4, 3], [1, {2, 3}, 4])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_pairwise.py ===
import numpy as np
import pytest

from pandas.compat import IS64

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core.algorithms import safe_sort


@pytest.fixture(
    params=[
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1, 0]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1, 1]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=["C", "C"]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1.0, 0]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[0.0, 1]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=["C", 1]),
        DataFrame([[2.0, 4.0], [1.0, 2.0], [5.0, 2.0], [8.0, 1.0]], columns=[1, 0.0]),
        DataFrame([[2, 4.0], [1, 2.0], [5, 2.0], [8, 1.0]], columns=[0, 1.0]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1.0]], columns=[1.0, "X"]),
    ]
)
def pairwise_frames(request):
    """Pairwise frames test_pairwise"""
    return request.param


@pytest.fixture
def pairwise_target_frame():
    """Pairwise target frame for test_pairwise"""
    return DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[0, 1])


@pytest.fixture
def pairwise_other_frame():
    """Pairwise other frame for test_pairwise"""
    return DataFrame(
        [[None, 1, 1], [None, 1, 2], [None, 3, 2], [None, 8, 1]],
        columns=["Y", "Z", "X"],
    )


def test_rolling_cov(series):
    A = series
    B = A + np.random.default_rng(2).standard_normal(len(A))

    result = A.rolling(window=50, min_periods=25).cov(B)
    tm.assert_almost_equal(result.iloc[-1], np.cov(A[-50:], B[-50:])[0, 1])


def test_rolling_corr(series):
    A = series
    B = A + np.random.default_rng(2).standard_normal(len(A))

    result = A.rolling(window=50, min_periods=25).corr(B)
    tm.assert_almost_equal(result.iloc[-1], np.corrcoef(A[-50:], B[-50:])[0, 1])


def test_rolling_corr_bias_correction():
    # test for correct bias correction
    a = Series(
        np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
    )
    b = a.copy()
    a[:5] = np.nan
    b[:10] = np.nan

    result = a.rolling(window=len(a), min_periods=1).corr(b)
    tm.assert_almost_equal(result.iloc[-1], a.corr(b))


@pytest.mark.parametrize("func", ["cov", "corr"])
def test_rolling_pairwise_cov_corr(func, frame):
    result = getattr(frame.rolling(window=10, min_periods=5), func)()
    result = result.loc[(slice(None), 1), 5]
    result.index = result.index.droplevel(1)
    expected = getattr(frame[1].rolling(window=10, min_periods=5), func)(frame[5])
    tm.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("method", ["corr", "cov"])
def test_flex_binary_frame(method, frame):
    series = frame[1]

    res = getattr(series.rolling(window=10), method)(frame)
    res2 = getattr(frame.rolling(window=10), method)(series)
    exp = frame.apply(lambda x: getattr(series.rolling(window=10), method)(x))

    tm.assert_frame_equal(res, exp)
    tm.assert_frame_equal(res2, exp)

    frame2 = frame.copy()
    frame2 = DataFrame(
        np.random.default_rng(2).standard_normal(frame2.shape),
        index=frame2.index,
        columns=frame2.columns,
    )

    res3 = getattr(frame.rolling(window=10), method)(frame2)
    exp = DataFrame(
        {k: getattr(frame[k].rolling(window=10), method)(frame2[k]) for k in frame}
    )
    tm.assert_frame_equal(res3, exp)


@pytest.mark.parametrize("window", range(7))
def test_rolling_corr_with_zero_variance(window):
    # GH 18430
    s = Series(np.zeros(20))
    other = Series(np.arange(20))

    assert s.rolling(window=window).corr(other=other).isna().all()


def test_corr_sanity():
    # GH 3155
    df = DataFrame(
        np.array(
            [
                [0.87024726, 0.18505595],
                [0.64355431, 0.3091617],
                [0.92372966, 0.50552513],
                [0.00203756, 0.04520709],
                [0.84780328, 0.33394331],
                [0.78369152, 0.63919667],
            ]
        )
    )

    res = df[0].rolling(5, center=True).corr(df[1])
    assert all(np.abs(np.nan_to_num(x)) <= 1 for x in res)

    df = DataFrame(np.random.default_rng(2).random((30, 2)))
    res = df[0].rolling(5, center=True).corr(df[1])
    assert all(np.abs(np.nan_to_num(x)) <= 1 for x in res)


def test_rolling_cov_diff_length():
    # GH 7512
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.rolling(window=3, min_periods=2).cov(s2)
    expected = Series([None, None, 2.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.rolling(window=3, min_periods=2).cov(s2a)
    tm.assert_series_equal(result, expected)


def test_rolling_corr_diff_length():
    # GH 7512
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.rolling(window=3, min_periods=2).corr(s2)
    expected = Series([None, None, 1.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.rolling(window=3, min_periods=2).corr(s2a)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.rolling(window=10, min_periods=5).cov(x, pairwise=True)),
        lambda x: (x.rolling(window=10, min_periods=5).corr(x, pairwise=True)),
    ],
)
def test_rolling_functions_window_non_shrinkage_binary(f):
    # corr/cov return a MI DataFrame
    df = DataFrame(
        [[1, 5], [3, 2], [3, 9], [-1, 0]],
        columns=Index(["A", "B"], name="foo"),
        index=Index(range(4), name="bar"),
    )
    df_expected = DataFrame(
        columns=Index(["A", "B"], name="foo"),
        index=MultiIndex.from_product([df.index, df.columns], names=["bar", "foo"]),
        dtype="float64",
    )
    df_result = f(df)
    tm.assert_frame_equal(df_result, df_expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.rolling(window=10, min_periods=5).cov(x, pairwise=True)),
        lambda x: (x.rolling(window=10, min_periods=5).corr(x, pairwise=True)),
    ],
)
def test_moment_functions_zero_length_pairwise(f):
    df1 = DataFrame()
    df2 = DataFrame(columns=Index(["a"], name="foo"), index=Index([], name="bar"))
    df2["a"] = df2["a"].astype("float64")

    df1_expected = DataFrame(index=MultiIndex.from_product([df1.index, df1.columns]))
    df2_expected = DataFrame(
        index=MultiIndex.from_product([df2.index, df2.columns], names=["bar", "foo"]),
        columns=Index(["a"], name="foo"),
        dtype="float64",
    )

    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)


class TestPairwise:
    # GH 7738
    @pytest.mark.parametrize("f", [lambda x: x.cov(), lambda x: x.corr()])
    def test_no_flex(self, pairwise_frames, pairwise_target_frame, f):
        # DataFrame methods (which do not call flex_binary_moment())

        result = f(pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.columns)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values

        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x: x.expanding().cov(pairwise=True),
            lambda x: x.expanding().corr(pairwise=True),
            lambda x: x.rolling(window=3).cov(pairwise=True),
            lambda x: x.rolling(window=3).corr(pairwise=True),
            lambda x: x.ewm(com=3).cov(pairwise=True),
            lambda x: x.ewm(com=3).corr(pairwise=True),
        ],
    )
    def test_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):
        # DataFrame with itself, pairwise=True
        # note that we may construct the 1st level of the MI
        # in a non-monotonic way, so compare accordingly
        result = f(pairwise_frames)
        tm.assert_index_equal(
            result.index.levels[0], pairwise_frames.index, check_names=False
        )
        tm.assert_index_equal(
            safe_sort(result.index.levels[1]),
            safe_sort(pairwise_frames.columns.unique()),
        )
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values

        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x: x.expanding().cov(pairwise=False),
            lambda x: x.expanding().corr(pairwise=False),
            lambda x: x.rolling(window=3).cov(pairwise=False),
            lambda x: x.rolling(window=3).corr(pairwise=False),
            lambda x: x.ewm(com=3).cov(pairwise=False),
            lambda x: x.ewm(com=3).corr(pairwise=False),
        ],
    )
    def test_no_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):
        # DataFrame with itself, pairwise=False
        result = f(pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values

        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y, pairwise=True),
            lambda x, y: x.expanding().corr(y, pairwise=True),
            lambda x, y: x.rolling(window=3).cov(y, pairwise=True),
            # TODO: We're missing a flag somewhere in meson
            pytest.param(
                lambda x, y: x.rolling(window=3).corr(y, pairwise=True),
                marks=pytest.mark.xfail(
                    not IS64, reason="Precision issues on 32 bit", strict=False
                ),
            ),
            lambda x, y: x.ewm(com=3).cov(y, pairwise=True),
            lambda x, y: x.ewm(com=3).corr(y, pairwise=True),
        ],
    )
    def test_pairwise_with_other(
        self, pairwise_frames, pairwise_target_frame, pairwise_other_frame, f
    ):
        # DataFrame with another DataFrame, pairwise=True
        result = f(pairwise_frames, pairwise_other_frame)
        tm.assert_index_equal(
            result.index.levels[0], pairwise_frames.index, check_names=False
        )
        tm.assert_index_equal(
            safe_sort(result.index.levels[1]),
            safe_sort(pairwise_other_frame.columns.unique()),
        )
        expected = f(pairwise_target_frame, pairwise_other_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values

        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.filterwarnings("ignore:RuntimeWarning")
    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y, pairwise=False),
            lambda x, y: x.expanding().corr(y, pairwise=False),
            lambda x, y: x.rolling(window=3).cov(y, pairwise=False),
            lambda x, y: x.rolling(window=3).corr(y, pairwise=False),
            lambda x, y: x.ewm(com=3).cov(y, pairwise=False),
            lambda x, y: x.ewm(com=3).corr(y, pairwise=False),
        ],
    )
    def test_no_pairwise_with_other(self, pairwise_frames, pairwise_other_frame, f):
        # DataFrame with another DataFrame, pairwise=False
        result = (
            f(pairwise_frames, pairwise_other_frame)
            if pairwise_frames.columns.is_unique
            else None
        )
        if result is not None:
            # we can have int and str columns
            expected_index = pairwise_frames.index.union(pairwise_other_frame.index)
            expected_columns = pairwise_frames.columns.union(
                pairwise_other_frame.columns
            )
            tm.assert_index_equal(result.index, expected_index)
            tm.assert_index_equal(result.columns, expected_columns)
        else:
            with pytest.raises(ValueError, match="'arg1' columns are not unique"):
                f(pairwise_frames, pairwise_other_frame)
            with pytest.raises(ValueError, match="'arg2' columns are not unique"):
                f(pairwise_other_frame, pairwise_frames)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y),
            lambda x, y: x.expanding().corr(y),
            lambda x, y: x.rolling(window=3).cov(y),
            lambda x, y: x.rolling(window=3).corr(y),
            lambda x, y: x.ewm(com=3).cov(y),
            lambda x, y: x.ewm(com=3).corr(y),
        ],
    )
    def test_pairwise_with_series(self, pairwise_frames, pairwise_target_frame, f):
        # DataFrame with a Series
        result = f(pairwise_frames, Series([1, 1, 3, 8]))
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame, Series([1, 1, 3, 8]))
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

        result = f(Series([1, 1, 3, 8]), pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(Series([1, 1, 3, 8]), pairwise_target_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_corr_freq_memory_error(self):
        # GH 31789
        s = Series(range(5), index=date_range("2020", periods=5))
        result = s.rolling("12h").corr(s)
        expected = Series([np.nan] * 5, index=date_range("2020", periods=5))
        tm.assert_series_equal(result, expected)

    def test_cov_mulittindex(self):
        # GH 34440

        columns = MultiIndex.from_product([list("ab"), list("xy"), list("AB")])
        index = range(3)
        df = DataFrame(np.arange(24).reshape(3, 8), index=index, columns=columns)

        result = df.ewm(alpha=0.1).cov()

        index = MultiIndex.from_product([range(3), list("ab"), list("xy"), list("AB")])
        columns = MultiIndex.from_product([list("ab"), list("xy"), list("AB")])
        expected = DataFrame(
            np.vstack(
                (
                    np.full((8, 8), np.nan),
                    np.full((8, 8), 32.000000),
                    np.full((8, 8), 63.881919),
                )
            ),
            index=index,
            columns=columns,
        )

        tm.assert_frame_equal(result, expected)

    def test_multindex_columns_pairwise_func(self):
        # GH 21157
        columns = MultiIndex.from_arrays([["M", "N"], ["P", "Q"]], names=["a", "b"])
        df = DataFrame(np.ones((5, 2)), columns=columns)
        result = df.rolling(3).corr()
        expected = DataFrame(
            np.nan,
            index=MultiIndex.from_arrays(
                [
                    np.repeat(np.arange(5, dtype=np.int64), 2),
                    ["M", "N"] * 5,
                    ["P", "Q"] * 5,
                ],
                names=[None, "a", "b"],
            ),
            columns=columns,
        )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_xs.py ===
import re

import numpy as np
import pytest

from pandas.errors import SettingWithCopyError

from pandas import (
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm

from pandas.tseries.offsets import BDay


@pytest.fixture
def four_level_index_dataframe():
    arr = np.array(
        [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.364],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
            [-0.6662, -0.5243, -0.358, 0.89145, 2.5838],
        ]
    )
    index = MultiIndex(
        levels=[["a", "x"], ["b", "q"], [10.0032, 20.0, 30.0], [3, 4, 5]],
        codes=[[0, 0, 1], [0, 1, 1], [0, 1, 2], [2, 1, 0]],
        names=["one", "two", "three", "four"],
    )
    return DataFrame(arr, index=index, columns=list("ABCDE"))


class TestXS:
    def test_xs(
        self, float_frame, datetime_frame, using_copy_on_write, warn_copy_on_write
    ):
        float_frame_orig = float_frame.copy()
        idx = float_frame.index[5]
        xs = float_frame.xs(idx)
        for item, value in xs.items():
            if np.isnan(value):
                assert np.isnan(float_frame[item][idx])
            else:
                assert value == float_frame[item][idx]

        # mixed-type xs
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}
        frame = DataFrame(test_data)
        xs = frame.xs("1")
        assert xs.dtype == np.object_
        assert xs["A"] == 1
        assert xs["B"] == "1"

        with pytest.raises(
            KeyError, match=re.escape("Timestamp('1999-12-31 00:00:00')")
        ):
            datetime_frame.xs(datetime_frame.index[0] - BDay())

        # xs get column
        series = float_frame.xs("A", axis=1)
        expected = float_frame["A"]
        tm.assert_series_equal(series, expected)

        # view is returned if possible
        series = float_frame.xs("A", axis=1)
        with tm.assert_cow_warning(warn_copy_on_write):
            series[:] = 5
        if using_copy_on_write:
            # but with CoW the view shouldn't propagate mutations
            tm.assert_series_equal(float_frame["A"], float_frame_orig["A"])
            assert not (expected == 5).all()
        else:
            assert (expected == 5).all()

    def test_xs_corner(self):
        # pathological mixed-type reordering case
        df = DataFrame(index=[0], columns=Index([], dtype="str"))
        df["A"] = 1.0
        df["B"] = "foo"
        df["C"] = 2.0
        df["D"] = "bar"
        df["E"] = 3.0

        xs = df.xs(0)
        exp = Series([1.0, "foo", 2.0, "bar", 3.0], index=list("ABCDE"), name=0)
        tm.assert_series_equal(xs, exp)

        # no columns but Index(dtype=object)
        df = DataFrame(index=["a", "b", "c"])
        result = df.xs("a")
        expected = Series([], name="a", dtype=np.float64)
        tm.assert_series_equal(result, expected)

    def test_xs_duplicates(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            index=["b", "b", "c", "b", "a"],
        )

        cross = df.xs("c")
        exp = df.iloc[2]
        tm.assert_series_equal(cross, exp)

    def test_xs_keep_level(self):
        df = DataFrame(
            {
                "day": {0: "sat", 1: "sun"},
                "flavour": {0: "strawberry", 1: "strawberry"},
                "sales": {0: 10, 1: 12},
                "year": {0: 2008, 1: 2008},
            }
        ).set_index(["year", "flavour", "day"])
        result = df.xs("sat", level="day", drop_level=False)
        expected = df[:1]
        tm.assert_frame_equal(result, expected)

        result = df.xs((2008, "sat"), level=["year", "day"], drop_level=False)
        tm.assert_frame_equal(result, expected)

    def test_xs_view(
        self, using_array_manager, using_copy_on_write, warn_copy_on_write
    ):
        # in 0.14 this will return a view if possible a copy otherwise, but
        # this is numpy dependent

        dm = DataFrame(np.arange(20.0).reshape(4, 5), index=range(4), columns=range(5))
        df_orig = dm.copy()

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                dm.xs(2)[:] = 20
            tm.assert_frame_equal(dm, df_orig)
        elif using_array_manager:
            # INFO(ArrayManager) with ArrayManager getting a row as a view is
            # not possible
            msg = r"\nA value is trying to be set on a copy of a slice from a DataFrame"
            with pytest.raises(SettingWithCopyError, match=msg):
                dm.xs(2)[:] = 20
            assert not (dm.xs(2) == 20).any()
        else:
            with tm.raises_chained_assignment_error():
                dm.xs(2)[:] = 20
            assert (dm.xs(2) == 20).all()


class TestXSWithMultiIndex:
    def test_xs_doc_example(self):
        # TODO: more descriptive name
        # based on example in advanced.rst
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = list(zip(*arrays))

        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 8)),
            index=["A", "B", "C"],
            columns=index,
        )

        result = df.xs(("one", "bar"), level=("second", "first"), axis=1)

        expected = df.iloc[:, [0]]
        tm.assert_frame_equal(result, expected)

    def test_xs_integer_key(self):
        # see GH#2107
        dates = range(20111201, 20111205)
        ids = list("abcde")
        index = MultiIndex.from_product([dates, ids], names=["date", "secid"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 3)),
            index,
            ["X", "Y", "Z"],
        )

        result = df.xs(20111201, level="date")
        expected = df.loc[20111201, :]
        tm.assert_frame_equal(result, expected)

    def test_xs_level(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data
        result = df.xs("two", level="second")
        expected = df[df.index.get_level_values(1) == "two"]
        expected.index = Index(["foo", "bar", "baz", "qux"], name="first")
        tm.assert_frame_equal(result, expected)

    def test_xs_level_eq_2(self):
        arr = np.random.default_rng(2).standard_normal((3, 5))
        index = MultiIndex(
            levels=[["a", "p", "x"], ["b", "q", "y"], ["c", "r", "z"]],
            codes=[[2, 0, 1], [2, 0, 1], [2, 0, 1]],
        )
        df = DataFrame(arr, index=index)
        expected = DataFrame(arr[1:2], index=[["a"], ["b"]])
        result = df.xs("c", level=2)
        tm.assert_frame_equal(result, expected)

    def test_xs_setting_with_copy_error(
        self,
        multiindex_dataframe_random_data,
        using_copy_on_write,
        warn_copy_on_write,
    ):
        # this is a copy in 0.14
        df = multiindex_dataframe_random_data
        df_orig = df.copy()
        result = df.xs("two", level="second")

        if using_copy_on_write or warn_copy_on_write:
            result[:] = 10
        else:
            # setting this will give a SettingWithCopyError
            # as we are trying to write a view
            msg = "A value is trying to be set on a copy of a slice from a DataFrame"
            with pytest.raises(SettingWithCopyError, match=msg):
                result[:] = 10
        tm.assert_frame_equal(df, df_orig)

    def test_xs_setting_with_copy_error_multiple(
        self, four_level_index_dataframe, using_copy_on_write, warn_copy_on_write
    ):
        # this is a copy in 0.14
        df = four_level_index_dataframe
        df_orig = df.copy()
        result = df.xs(("a", 4), level=["one", "four"])

        if using_copy_on_write or warn_copy_on_write:
            result[:] = 10
        else:
            # setting this will give a SettingWithCopyError
            # as we are trying to write a view
            msg = "A value is trying to be set on a copy of a slice from a DataFrame"
            with pytest.raises(SettingWithCopyError, match=msg):
                result[:] = 10
        tm.assert_frame_equal(df, df_orig)

    @pytest.mark.parametrize("key, level", [("one", "second"), (["one"], ["second"])])
    def test_xs_with_duplicates(self, key, level, multiindex_dataframe_random_data):
        # see GH#13719
        frame = multiindex_dataframe_random_data
        df = concat([frame] * 2)
        assert df.index.is_unique is False
        expected = concat([frame.xs("one", level="second")] * 2)

        if isinstance(key, list):
            result = df.xs(tuple(key), level=level)
        else:
            result = df.xs(key, level=level)
        tm.assert_frame_equal(result, expected)

    def test_xs_missing_values_in_index(self):
        # see GH#6574
        # missing values in returned index should be preserved
        acc = [
            ("a", "abcde", 1),
            ("b", "bbcde", 2),
            ("y", "yzcde", 25),
            ("z", "xbcde", 24),
            ("z", None, 26),
            ("z", "zbcde", 25),
            ("z", "ybcde", 26),
        ]
        df = DataFrame(acc, columns=["a1", "a2", "cnt"]).set_index(["a1", "a2"])
        expected = DataFrame(
            {"cnt": [24, 26, 25, 26]},
            index=Index(["xbcde", np.nan, "zbcde", "ybcde"], name="a2"),
        )

        result = df.xs("z", level="a1")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "key, level, exp_arr, exp_index",
        [
            ("a", "lvl0", lambda x: x[:, 0:2], Index(["bar", "foo"], name="lvl1")),
            ("foo", "lvl1", lambda x: x[:, 1:2], Index(["a"], name="lvl0")),
        ],
    )
    def test_xs_named_levels_axis_eq_1(self, key, level, exp_arr, exp_index):
        # see GH#2903
        arr = np.random.default_rng(2).standard_normal((4, 4))
        index = MultiIndex(
            levels=[["a", "b"], ["bar", "foo", "hello", "world"]],
            codes=[[0, 0, 1, 1], [0, 1, 2, 3]],
            names=["lvl0", "lvl1"],
        )
        df = DataFrame(arr, columns=index)
        result = df.xs(key, level=level, axis=1)
        expected = DataFrame(exp_arr(arr), columns=exp_index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer",
        [
            lambda df: df.xs(("a", 4), level=["one", "four"]),
            lambda df: df.xs("a").xs(4, level="four"),
        ],
    )
    def test_xs_level_multiple(self, indexer, four_level_index_dataframe):
        df = four_level_index_dataframe
        expected_values = [[0.4473, 1.4152, 0.2834, 1.00661, 0.1744]]
        expected_index = MultiIndex(
            levels=[["q"], [20.0]], codes=[[0], [0]], names=["two", "three"]
        )
        expected = DataFrame(
            expected_values, index=expected_index, columns=list("ABCDE")
        )
        result = indexer(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer", [lambda df: df.xs("a", level=0), lambda df: df.xs("a")]
    )
    def test_xs_level0(self, indexer, four_level_index_dataframe):
        df = four_level_index_dataframe
        expected_values = [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.364],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
        ]
        expected_index = MultiIndex(
            levels=[["b", "q"], [10.0032, 20.0], [4, 5]],
            codes=[[0, 1], [0, 1], [1, 0]],
            names=["two", "three", "four"],
        )
        expected = DataFrame(
            expected_values, index=expected_index, columns=list("ABCDE")
        )

        result = indexer(df)
        tm.assert_frame_equal(result, expected)

    def test_xs_values(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data
        result = df.xs(("bar", "two")).values
        expected = df.values[4]
        tm.assert_almost_equal(result, expected)

    def test_xs_loc_equality(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data
        result = df.xs(("bar", "two"))
        expected = df.loc[("bar", "two")]
        tm.assert_series_equal(result, expected)

    def test_xs_IndexSlice_argument_not_implemented(self, frame_or_series):
        # GH#35301

        index = MultiIndex(
            levels=[[("foo", "bar", 0), ("foo", "baz", 0), ("foo", "qux", 0)], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        obj = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)
        if frame_or_series is Series:
            obj = obj[0]

        expected = obj.iloc[-2:].droplevel(0)

        result = obj.xs(IndexSlice[("foo", "qux", 0), :])
        tm.assert_equal(result, expected)

        result = obj.loc[IndexSlice[("foo", "qux", 0), :]]
        tm.assert_equal(result, expected)

    def test_xs_levels_raises(self, frame_or_series):
        obj = DataFrame({"A": [1, 2, 3]})
        if frame_or_series is Series:
            obj = obj["A"]

        msg = "Index must be a MultiIndex"
        with pytest.raises(TypeError, match=msg):
            obj.xs(0, level="as")

    def test_xs_multiindex_droplevel_false(self):
        # GH#19056
        mi = MultiIndex.from_tuples(
            [("a", "x"), ("a", "y"), ("b", "x")], names=["level1", "level2"]
        )
        df = DataFrame([[1, 2, 3]], columns=mi)
        result = df.xs("a", axis=1, drop_level=False)
        expected = DataFrame(
            [[1, 2]],
            columns=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y")], names=["level1", "level2"]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_xs_droplevel_false(self):
        # GH#19056
        df = DataFrame([[1, 2, 3]], columns=Index(["a", "b", "c"]))
        result = df.xs("a", axis=1, drop_level=False)
        expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(result, expected)

    def test_xs_droplevel_false_view(
        self, using_array_manager, using_copy_on_write, warn_copy_on_write
    ):
        # GH#37832
        df = DataFrame([[1, 2, 3]], columns=Index(["a", "b", "c"]))
        result = df.xs("a", axis=1, drop_level=False)
        # check that result still views the same data as df
        assert np.shares_memory(result.iloc[:, 0]._values, df.iloc[:, 0]._values)

        with tm.assert_cow_warning(warn_copy_on_write):
            df.iloc[0, 0] = 2
        if using_copy_on_write:
            # with copy on write the subset is never modified
            expected = DataFrame({"a": [1]})
        else:
            # modifying original df also modifies result when having a single block
            expected = DataFrame({"a": [2]})
        tm.assert_frame_equal(result, expected)

        # with mixed dataframe, modifying the parent doesn't modify result
        # TODO the "split" path behaves differently here as with single block
        df = DataFrame([[1, 2.5, "a"]], columns=Index(["a", "b", "c"]))
        result = df.xs("a", axis=1, drop_level=False)
        df.iloc[0, 0] = 2
        if using_copy_on_write:
            # with copy on write the subset is never modified
            expected = DataFrame({"a": [1]})
        elif using_array_manager:
            # Here the behavior is consistent
            expected = DataFrame({"a": [2]})
        else:
            # FIXME: iloc does not update the array inplace using
            # "split" path
            expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(result, expected)

    def test_xs_list_indexer_droplevel_false(self):
        # GH#41760
        mi = MultiIndex.from_tuples([("x", "m", "a"), ("x", "n", "b"), ("y", "o", "c")])
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
        with pytest.raises(KeyError, match="y"):
            df.xs(("x", "y"), drop_level=False, axis=1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_orc.py ===
""" test orc compat """
import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib

import numpy as np
import pytest

import pandas as pd
from pandas import read_orc
import pandas._testing as tm
from pandas.core.arrays import StringArray

pytest.importorskip("pyarrow.orc")

import pyarrow as pa

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def dirpath(datapath):
    return datapath("io", "data", "orc")


@pytest.fixture(
    params=[
        np.array([1, 20], dtype="uint64"),
        pd.Series(["a", "b", "a"], dtype="category"),
        [pd.Interval(left=0, right=2), pd.Interval(left=0, right=5)],
        [pd.Period("2022-01-03", freq="D"), pd.Period("2022-01-04", freq="D")],
    ]
)
def orc_writer_dtypes_not_supported(request):
    # Examples of dataframes with dtypes for which conversion to ORC
    # hasn't been implemented yet, that is, Category, unsigned integers,
    # interval, period and sparse.
    return pd.DataFrame({"unimpl": request.param})


def test_orc_reader_empty(dirpath, using_infer_string):
    columns = [
        "boolean1",
        "byte1",
        "short1",
        "int1",
        "long1",
        "float1",
        "double1",
        "bytes1",
        "string1",
    ]
    dtypes = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "object",
        "str" if using_infer_string else "object",
    ]
    expected = pd.DataFrame(index=pd.RangeIndex(0))
    for colname, dtype in zip(columns, dtypes):
        expected[colname] = pd.Series(dtype=dtype)
    expected.columns = expected.columns.astype("str")

    inputfile = os.path.join(dirpath, "TestOrcFile.emptyFile.orc")
    got = read_orc(inputfile, columns=columns)

    tm.assert_equal(expected, got)


def test_orc_reader_basic(dirpath):
    data = {
        "boolean1": np.array([False, True], dtype="bool"),
        "byte1": np.array([1, 100], dtype="int8"),
        "short1": np.array([1024, 2048], dtype="int16"),
        "int1": np.array([65536, 65536], dtype="int32"),
        "long1": np.array([9223372036854775807, 9223372036854775807], dtype="int64"),
        "float1": np.array([1.0, 2.0], dtype="float32"),
        "double1": np.array([-15.0, -5.0], dtype="float64"),
        "bytes1": np.array([b"\x00\x01\x02\x03\x04", b""], dtype="object"),
        "string1": np.array(["hi", "bye"], dtype="object"),
    }
    expected = pd.DataFrame.from_dict(data)

    inputfile = os.path.join(dirpath, "TestOrcFile.test1.orc")
    got = read_orc(inputfile, columns=data.keys())

    tm.assert_equal(expected, got)


def test_orc_reader_decimal(dirpath):
    # Only testing the first 10 rows of data
    data = {
        "_col0": np.array(
            [
                Decimal("-1000.50000"),
                Decimal("-999.60000"),
                Decimal("-998.70000"),
                Decimal("-997.80000"),
                Decimal("-996.90000"),
                Decimal("-995.10000"),
                Decimal("-994.11000"),
                Decimal("-993.12000"),
                Decimal("-992.13000"),
                Decimal("-991.14000"),
            ],
            dtype="object",
        )
    }
    expected = pd.DataFrame.from_dict(data)

    inputfile = os.path.join(dirpath, "TestOrcFile.decimal.orc")
    got = read_orc(inputfile).iloc[:10]

    tm.assert_equal(expected, got)


def test_orc_reader_date_low(dirpath):
    data = {
        "time": np.array(
            [
                "1900-05-05 12:34:56.100000",
                "1900-05-05 12:34:56.100100",
                "1900-05-05 12:34:56.100200",
                "1900-05-05 12:34:56.100300",
                "1900-05-05 12:34:56.100400",
                "1900-05-05 12:34:56.100500",
                "1900-05-05 12:34:56.100600",
                "1900-05-05 12:34:56.100700",
                "1900-05-05 12:34:56.100800",
                "1900-05-05 12:34:56.100900",
            ],
            dtype="datetime64[ns]",
        ),
        "date": np.array(
            [
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
                datetime.date(1900, 12, 25),
            ],
            dtype="object",
        ),
    }
    expected = pd.DataFrame.from_dict(data)

    inputfile = os.path.join(dirpath, "TestOrcFile.testDate1900.orc")
    got = read_orc(inputfile).iloc[:10]

    tm.assert_equal(expected, got)


def test_orc_reader_date_high(dirpath):
    data = {
        "time": np.array(
            [
                "2038-05-05 12:34:56.100000",
                "2038-05-05 12:34:56.100100",
                "2038-05-05 12:34:56.100200",
                "2038-05-05 12:34:56.100300",
                "2038-05-05 12:34:56.100400",
                "2038-05-05 12:34:56.100500",
                "2038-05-05 12:34:56.100600",
                "2038-05-05 12:34:56.100700",
                "2038-05-05 12:34:56.100800",
                "2038-05-05 12:34:56.100900",
            ],
            dtype="datetime64[ns]",
        ),
        "date": np.array(
            [
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
                datetime.date(2038, 12, 25),
            ],
            dtype="object",
        ),
    }
    expected = pd.DataFrame.from_dict(data)

    inputfile = os.path.join(dirpath, "TestOrcFile.testDate2038.orc")
    got = read_orc(inputfile).iloc[:10]

    tm.assert_equal(expected, got)


def test_orc_reader_snappy_compressed(dirpath):
    data = {
        "int1": np.array(
            [
                -1160101563,
                1181413113,
                2065821249,
                -267157795,
                172111193,
                1752363137,
                1406072123,
                1911809390,
                -1308542224,
                -467100286,
            ],
            dtype="int32",
        ),
        "string1": np.array(
            [
                "f50dcb8",
                "382fdaaa",
                "90758c6",
                "9e8caf3f",
                "ee97332b",
                "d634da1",
                "2bea4396",
                "d67d89e8",
                "ad71007e",
                "e8c82066",
            ],
            dtype="object",
        ),
    }
    expected = pd.DataFrame.from_dict(data)

    inputfile = os.path.join(dirpath, "TestOrcFile.testSnappy.orc")
    got = read_orc(inputfile).iloc[:10]

    tm.assert_equal(expected, got)


def test_orc_roundtrip_file(dirpath):
    # GH44554
    # PyArrow gained ORC write support with the current argument order
    pytest.importorskip("pyarrow")

    data = {
        "boolean1": np.array([False, True], dtype="bool"),
        "byte1": np.array([1, 100], dtype="int8"),
        "short1": np.array([1024, 2048], dtype="int16"),
        "int1": np.array([65536, 65536], dtype="int32"),
        "long1": np.array([9223372036854775807, 9223372036854775807], dtype="int64"),
        "float1": np.array([1.0, 2.0], dtype="float32"),
        "double1": np.array([-15.0, -5.0], dtype="float64"),
        "bytes1": np.array([b"\x00\x01\x02\x03\x04", b""], dtype="object"),
        "string1": np.array(["hi", "bye"], dtype="object"),
    }
    expected = pd.DataFrame.from_dict(data)

    with tm.ensure_clean() as path:
        expected.to_orc(path)
        got = read_orc(path)

        tm.assert_equal(expected, got)


def test_orc_roundtrip_bytesio():
    # GH44554
    # PyArrow gained ORC write support with the current argument order
    pytest.importorskip("pyarrow")

    data = {
        "boolean1": np.array([False, True], dtype="bool"),
        "byte1": np.array([1, 100], dtype="int8"),
        "short1": np.array([1024, 2048], dtype="int16"),
        "int1": np.array([65536, 65536], dtype="int32"),
        "long1": np.array([9223372036854775807, 9223372036854775807], dtype="int64"),
        "float1": np.array([1.0, 2.0], dtype="float32"),
        "double1": np.array([-15.0, -5.0], dtype="float64"),
        "bytes1": np.array([b"\x00\x01\x02\x03\x04", b""], dtype="object"),
        "string1": np.array(["hi", "bye"], dtype="object"),
    }
    expected = pd.DataFrame.from_dict(data)

    bytes = expected.to_orc()
    got = read_orc(BytesIO(bytes))

    tm.assert_equal(expected, got)


def test_orc_writer_dtypes_not_supported(orc_writer_dtypes_not_supported):
    # GH44554
    # PyArrow gained ORC write support with the current argument order
    pytest.importorskip("pyarrow")

    msg = "The dtype of one or more columns is not supported yet."
    with pytest.raises(NotImplementedError, match=msg):
        orc_writer_dtypes_not_supported.to_orc()


def test_orc_dtype_backend_pyarrow(using_infer_string):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(
        {
            "string": list("abc"),
            "string_with_nan": ["a", np.nan, "c"],
            "string_with_none": ["a", None, "c"],
            "bytes": [b"foo", b"bar", None],
            "int": list(range(1, 4)),
            "float": np.arange(4.0, 7.0, dtype="float64"),
            "float_with_nan": [2.0, np.nan, 3.0],
            "bool": [True, False, True],
            "bool_with_na": [True, False, None],
            "datetime": pd.date_range("20130101", periods=3),
            "datetime_with_nat": [
                pd.Timestamp("20130101"),
                pd.NaT,
                pd.Timestamp("20130103"),
            ],
        }
    )

    bytes_data = df.copy().to_orc()
    result = read_orc(BytesIO(bytes_data), dtype_backend="pyarrow")

    expected = pd.DataFrame(
        {
            col: pd.arrays.ArrowExtensionArray(pa.array(df[col], from_pandas=True))
            for col in df.columns
        }
    )
    if using_infer_string:
        # ORC does not preserve distinction between string and large string
        # -> the default large string comes back as string
        string_dtype = pd.ArrowDtype(pa.string())
        expected["string"] = expected["string"].astype(string_dtype)
        expected["string_with_nan"] = expected["string_with_nan"].astype(string_dtype)
        expected["string_with_none"] = expected["string_with_none"].astype(string_dtype)

    tm.assert_frame_equal(result, expected)


def test_orc_dtype_backend_numpy_nullable():
    # GH#50503
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(
        {
            "string": list("abc"),
            "string_with_nan": ["a", np.nan, "c"],
            "string_with_none": ["a", None, "c"],
            "int": list(range(1, 4)),
            "int_with_nan": pd.Series([1, pd.NA, 3], dtype="Int64"),
            "na_only": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "float": np.arange(4.0, 7.0, dtype="float64"),
            "float_with_nan": [2.0, np.nan, 3.0],
            "bool": [True, False, True],
            "bool_with_na": [True, False, None],
        }
    )

    bytes_data = df.copy().to_orc()
    result = read_orc(BytesIO(bytes_data), dtype_backend="numpy_nullable")

    expected = pd.DataFrame(
        {
            "string": StringArray(np.array(["a", "b", "c"], dtype=np.object_)),
            "string_with_nan": StringArray(
                np.array(["a", pd.NA, "c"], dtype=np.object_)
            ),
            "string_with_none": StringArray(
                np.array(["a", pd.NA, "c"], dtype=np.object_)
            ),
            "int": pd.Series([1, 2, 3], dtype="Int64"),
            "int_with_nan": pd.Series([1, pd.NA, 3], dtype="Int64"),
            "na_only": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "float": pd.Series([4.0, 5.0, 6.0], dtype="Float64"),
            "float_with_nan": pd.Series([2.0, pd.NA, 3.0], dtype="Float64"),
            "bool": pd.Series([True, False, True], dtype="boolean"),
            "bool_with_na": pd.Series([True, False, pd.NA], dtype="boolean"),
        }
    )

    tm.assert_frame_equal(result, expected)


def test_orc_uri_path():
    expected = pd.DataFrame({"int": list(range(1, 4))})
    with tm.ensure_clean("tmp.orc") as path:
        expected.to_orc(path)
        uri = pathlib.Path(path).as_uri()
        result = read_orc(uri)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(start=2, stop=5, step=1),
        pd.RangeIndex(start=0, stop=3, step=1, name="non-default"),
        pd.Index([1, 2, 3]),
    ],
)
def test_to_orc_non_default_index(index):
    df = pd.DataFrame({"a": [1, 2, 3]}, index=index)
    msg = (
        "orc does not support serializing a non-default index|"
        "orc does not serialize index meta-data"
    )
    with pytest.raises(ValueError, match=msg):
        df.to_orc()


def test_invalid_dtype_backend():
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    df = pd.DataFrame({"int": list(range(1, 4))})
    with tm.ensure_clean("tmp.orc") as path:
        df.to_orc(path)
        with pytest.raises(ValueError, match=msg):
            read_orc(path, dtype_backend="numpy")


def test_string_inference(tmp_path):
    # GH#54431
    path = tmp_path / "test_string_inference.p"
    df = pd.DataFrame(data={"a": ["x", "y"]})
    df.to_orc(path)
    with pd.option_context("future.infer_string", True):
        result = read_orc(path)
    expected = pd.DataFrame(
        data={"a": ["x", "y"]},
        dtype=pd.StringDtype(na_value=np.nan),
        columns=pd.Index(["a"], dtype=pd.StringDtype(na_value=np.nan)),
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_reindex.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    NA,
    Categorical,
    Float64Dtype,
    Index,
    MultiIndex,
    NaT,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm


def test_reindex(datetime_series, string_series):
    identity = string_series.reindex(string_series.index)

    assert tm.shares_memory(string_series.index, identity.index)

    assert identity.index.is_(string_series.index)
    assert identity.index.identical(string_series.index)

    subIndex = string_series.index[10:20]
    subSeries = string_series.reindex(subIndex)

    for idx, val in subSeries.items():
        assert val == string_series[idx]

    subIndex2 = datetime_series.index[10:20]
    subTS = datetime_series.reindex(subIndex2)

    for idx, val in subTS.items():
        assert val == datetime_series[idx]
    stuffSeries = datetime_series.reindex(subIndex)

    assert np.isnan(stuffSeries).all()

    # This is extremely important for the Cython code to not screw up
    nonContigIndex = datetime_series.index[::2]
    subNonContig = datetime_series.reindex(nonContigIndex)
    for idx, val in subNonContig.items():
        assert val == datetime_series[idx]

    # return a copy the same index here
    result = datetime_series.reindex()
    assert result is not datetime_series


def test_reindex_nan():
    ts = Series([2, 3, 5, 7], index=[1, 4, np.nan, 8])

    i, j = [np.nan, 1, np.nan, 8, 4, np.nan], [2, 0, 2, 3, 1, 2]
    tm.assert_series_equal(ts.reindex(i), ts.iloc[j])

    ts.index = ts.index.astype("object")

    # reindex coerces index.dtype to float, loc/iloc doesn't
    tm.assert_series_equal(ts.reindex(i), ts.iloc[j], check_index_type=False)


def test_reindex_series_add_nat():
    rng = date_range("1/1/2000 00:00:00", periods=10, freq="10s")
    series = Series(rng)

    result = series.reindex(range(15))
    assert np.issubdtype(result.dtype, np.dtype("M8[ns]"))

    mask = result.isna()
    assert mask[-5:].all()
    assert not mask[:-5].any()


def test_reindex_with_datetimes():
    rng = date_range("1/1/2000", periods=20)
    ts = Series(np.random.default_rng(2).standard_normal(20), index=rng)

    result = ts.reindex(list(ts.index[5:10]))
    expected = ts[5:10]
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)

    result = ts[list(ts.index[5:10])]
    tm.assert_series_equal(result, expected)


def test_reindex_corner(datetime_series):
    # (don't forget to fix this) I think it's fixed
    empty = Series(index=[])
    empty.reindex(datetime_series.index, method="pad")  # it works

    # corner case: pad empty series
    reindexed = empty.reindex(datetime_series.index, method="pad")

    # pass non-Index
    reindexed = datetime_series.reindex(list(datetime_series.index))
    datetime_series.index = datetime_series.index._with_freq(None)
    tm.assert_series_equal(datetime_series, reindexed)

    # bad fill method
    ts = datetime_series[::2]
    msg = (
        r"Invalid fill method\. Expecting pad \(ffill\), backfill "
        r"\(bfill\) or nearest\. Got foo"
    )
    with pytest.raises(ValueError, match=msg):
        ts.reindex(datetime_series.index, method="foo")


def test_reindex_pad():
    s = Series(np.arange(10), dtype="int64")
    s2 = s[::2]

    reindexed = s2.reindex(s.index, method="pad")
    reindexed2 = s2.reindex(s.index, method="ffill")
    tm.assert_series_equal(reindexed, reindexed2)

    expected = Series([0, 0, 2, 2, 4, 4, 6, 6, 8, 8])
    tm.assert_series_equal(reindexed, expected)


def test_reindex_pad2():
    # GH4604
    s = Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    new_index = ["a", "g", "c", "f"]
    expected = Series([1, 1, 3, 3], index=new_index)

    # this changes dtype because the ffill happens after
    result = s.reindex(new_index).ffill()
    tm.assert_series_equal(result, expected.astype("float64"))

    msg = "The 'downcast' keyword in ffill is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.reindex(new_index).ffill(downcast="infer")
    tm.assert_series_equal(result, expected)

    expected = Series([1, 5, 3, 5], index=new_index)
    result = s.reindex(new_index, method="ffill")
    tm.assert_series_equal(result, expected)


def test_reindex_inference():
    # inference of new dtype
    s = Series([True, False, False, True], index=list("abcd"))
    new_index = "agc"
    msg = "Downcasting object dtype arrays on"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.reindex(list(new_index)).ffill()
    expected = Series([True, True, False], index=list(new_index))
    tm.assert_series_equal(result, expected)


def test_reindex_downcasting():
    # GH4618 shifted series downcasting
    s = Series(False, index=range(5))
    msg = "Downcasting object dtype arrays on"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.shift(1).bfill()
    expected = Series(False, index=range(5))
    tm.assert_series_equal(result, expected)


def test_reindex_nearest():
    s = Series(np.arange(10, dtype="int64"))
    target = [0.1, 0.9, 1.5, 2.0]
    result = s.reindex(target, method="nearest")
    expected = Series(np.around(target).astype("int64"), target)
    tm.assert_series_equal(expected, result)

    result = s.reindex(target, method="nearest", tolerance=0.2)
    expected = Series([0, 1, np.nan, 2], target)
    tm.assert_series_equal(expected, result)

    result = s.reindex(target, method="nearest", tolerance=[0.3, 0.01, 0.4, 3])
    expected = Series([0, np.nan, np.nan, 2], target)
    tm.assert_series_equal(expected, result)


def test_reindex_int(datetime_series):
    ts = datetime_series[::2]
    int_ts = Series(np.zeros(len(ts), dtype=int), index=ts.index)

    # this should work fine
    reindexed_int = int_ts.reindex(datetime_series.index)

    # if NaNs introduced
    assert reindexed_int.dtype == np.float64

    # NO NaNs introduced
    reindexed_int = int_ts.reindex(int_ts.index[::2])
    assert reindexed_int.dtype == np.dtype(int)


def test_reindex_bool(datetime_series):
    # A series other than float, int, string, or object
    ts = datetime_series[::2]
    bool_ts = Series(np.zeros(len(ts), dtype=bool), index=ts.index)

    # this should work fine
    reindexed_bool = bool_ts.reindex(datetime_series.index)

    # if NaNs introduced
    assert reindexed_bool.dtype == np.object_

    # NO NaNs introduced
    reindexed_bool = bool_ts.reindex(bool_ts.index[::2])
    assert reindexed_bool.dtype == np.bool_


def test_reindex_bool_pad(datetime_series):
    # fail
    ts = datetime_series[5:]
    bool_ts = Series(np.zeros(len(ts), dtype=bool), index=ts.index)
    filled_bool = bool_ts.reindex(datetime_series.index, method="pad")
    assert isna(filled_bool[:5]).all()


def test_reindex_categorical():
    index = date_range("20000101", periods=3)

    # reindexing to an invalid Categorical
    s = Series(["a", "b", "c"], dtype="category")
    result = s.reindex(index)
    expected = Series(
        Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"])
    )
    expected.index = index
    tm.assert_series_equal(result, expected)

    # partial reindexing
    expected = Series(Categorical(values=["b", "c"], categories=["a", "b", "c"]))
    expected.index = [1, 2]
    result = s.reindex([1, 2])
    tm.assert_series_equal(result, expected)

    expected = Series(Categorical(values=["c", np.nan], categories=["a", "b", "c"]))
    expected.index = [2, 3]
    result = s.reindex([2, 3])
    tm.assert_series_equal(result, expected)


def test_reindex_astype_order_consistency():
    # GH#17444
    ser = Series([1, 2, 3], index=[2, 0, 1])
    new_index = [0, 1, 2]
    temp_dtype = "category"
    new_dtype = str
    result = ser.reindex(new_index).astype(temp_dtype).astype(new_dtype)
    expected = ser.astype(temp_dtype).reindex(new_index).astype(new_dtype)
    tm.assert_series_equal(result, expected)


def test_reindex_fill_value():
    # -----------------------------------------------------------
    # floats
    floats = Series([1.0, 2.0, 3.0])
    result = floats.reindex([1, 2, 3])
    expected = Series([2.0, 3.0, np.nan], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)

    result = floats.reindex([1, 2, 3], fill_value=0)
    expected = Series([2.0, 3.0, 0], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)

    # -----------------------------------------------------------
    # ints
    ints = Series([1, 2, 3])

    result = ints.reindex([1, 2, 3])
    expected = Series([2.0, 3.0, np.nan], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)

    # don't upcast
    result = ints.reindex([1, 2, 3], fill_value=0)
    expected = Series([2, 3, 0], index=[1, 2, 3])
    assert issubclass(result.dtype.type, np.integer)
    tm.assert_series_equal(result, expected)

    # -----------------------------------------------------------
    # objects
    objects = Series([1, 2, 3], dtype=object)

    result = objects.reindex([1, 2, 3])
    expected = Series([2, 3, np.nan], index=[1, 2, 3], dtype=object)
    tm.assert_series_equal(result, expected)

    result = objects.reindex([1, 2, 3], fill_value="foo")
    expected = Series([2, 3, "foo"], index=[1, 2, 3], dtype=object)
    tm.assert_series_equal(result, expected)

    # ------------------------------------------------------------
    # bools
    bools = Series([True, False, True])

    result = bools.reindex([1, 2, 3])
    expected = Series([False, True, np.nan], index=[1, 2, 3], dtype=object)
    tm.assert_series_equal(result, expected)

    result = bools.reindex([1, 2, 3], fill_value=False)
    expected = Series([False, True, False], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)


@td.skip_array_manager_not_yet_implemented
@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize("fill_value", ["string", 0, Timedelta(0)])
def test_reindex_fill_value_datetimelike_upcast(dtype, fill_value, using_array_manager):
    # https://github.com/pandas-dev/pandas/issues/42921
    if dtype == "timedelta64[ns]" and fill_value == Timedelta(0):
        # use the scalar that is not compatible with the dtype for this test
        fill_value = Timestamp(0)

    ser = Series([NaT], dtype=dtype)

    result = ser.reindex([0, 1], fill_value=fill_value)
    expected = Series([NaT, fill_value], index=[0, 1], dtype=object)
    tm.assert_series_equal(result, expected)


def test_reindex_datetimeindexes_tz_naive_and_aware():
    # GH 8306
    idx = date_range("20131101", tz="America/Chicago", periods=7)
    newidx = date_range("20131103", periods=10, freq="h")
    s = Series(range(7), index=idx)
    msg = (
        r"Cannot compare dtypes datetime64\[ns, America/Chicago\] "
        r"and datetime64\[ns\]"
    )
    with pytest.raises(TypeError, match=msg):
        s.reindex(newidx, method="ffill")


def test_reindex_empty_series_tz_dtype():
    # GH 20869
    result = Series(dtype="datetime64[ns, UTC]").reindex([0, 1])
    expected = Series([NaT] * 2, dtype="datetime64[ns, UTC]")
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "p_values, o_values, values, expected_values",
    [
        (
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC"), "All"],
            [1.0, 1.0],
            [1.0, 1.0, np.nan],
        ),
        (
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [1.0, 1.0],
            [1.0, 1.0],
        ),
    ],
)
def test_reindex_periodindex_with_object(p_values, o_values, values, expected_values):
    # GH#28337
    period_index = PeriodIndex(p_values)
    object_index = Index(o_values)

    ser = Series(values, index=period_index)
    result = ser.reindex(object_index)
    expected = Series(expected_values, index=object_index)
    tm.assert_series_equal(result, expected)


def test_reindex_too_many_args():
    # GH 40980
    ser = Series([1, 2])
    msg = r"reindex\(\) takes from 1 to 2 positional arguments but 3 were given"
    with pytest.raises(TypeError, match=msg):
        ser.reindex([2, 3], False)


def test_reindex_double_index():
    # GH 40980
    ser = Series([1, 2])
    msg = r"reindex\(\) got multiple values for argument 'index'"
    with pytest.raises(TypeError, match=msg):
        ser.reindex([2, 3], index=[3, 4])


def test_reindex_no_posargs():
    # GH 40980
    ser = Series([1, 2])
    result = ser.reindex(index=[1, 0])
    expected = Series([2, 1], index=[1, 0])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("values", [[["a"], ["x"]], [[], []]])
def test_reindex_empty_with_level(values):
    # GH41170
    ser = Series(
        range(len(values[0])), index=MultiIndex.from_arrays(values), dtype="object"
    )
    result = ser.reindex(np.array(["b"]), level=0)
    expected = Series(
        index=MultiIndex(levels=[["b"], values[1]], codes=[[], []]), dtype="object"
    )
    tm.assert_series_equal(result, expected)


def test_reindex_missing_category():
    # GH#18185
    ser = Series([1, 2, 3, 1], dtype="category")
    msg = r"Cannot setitem on a Categorical with a new category \(-1\)"
    with pytest.raises(TypeError, match=msg):
        ser.reindex([1, 2, 3, 4, 5], fill_value=-1)


def test_reindexing_with_float64_NA_log():
    # GH 47055
    s = Series([1.0, NA], dtype=Float64Dtype())
    s_reindex = s.reindex(range(3))
    result = s_reindex.values._data
    expected = np.array([1, np.nan, np.nan])
    tm.assert_numpy_array_equal(result, expected)
    with tm.assert_produces_warning(None):
        result_log = np.log(s_reindex)
        expected_log = Series([0, np.nan, np.nan], dtype=Float64Dtype())
        tm.assert_series_equal(result_log, expected_log)


@pytest.mark.parametrize("dtype", ["timedelta64", "datetime64"])
def test_reindex_expand_nonnano_nat(dtype):
    # GH 53497
    ser = Series(np.array([1], dtype=f"{dtype}[s]"))
    result = ser.reindex(RangeIndex(2))
    expected = Series(
        np.array([1, getattr(np, dtype)("nat", "s")], dtype=f"{dtype}[s]")
    )
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\util.py ===
"""
Utility functions for

- building and importing modules on test time, using a temporary location
- detecting if compilers are present
- determining paths to tests

"""
import atexit
import concurrent.futures
import contextlib
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import pytest

import numpy
from numpy._utils import asunicode
from numpy.f2py._backends._meson import MesonBackend
from numpy.testing import IS_WASM, temppath

#
# Check if compilers are available at all...
#

def check_language(lang, code_snippet=None):
    if sys.platform == "win32":
        pytest.skip("No Fortran tests on Windows (Issue #25134)", allow_module_level=True)
    tmpdir = tempfile.mkdtemp()
    try:
        meson_file = os.path.join(tmpdir, "meson.build")
        with open(meson_file, "w") as f:
            f.write("project('check_compilers')\n")
            f.write(f"add_languages('{lang}')\n")
            if code_snippet:
                f.write(f"{lang}_compiler = meson.get_compiler('{lang}')\n")
                f.write(f"{lang}_code = '''{code_snippet}'''\n")
                f.write(
                    f"_have_{lang}_feature ="
                    f"{lang}_compiler.compiles({lang}_code,"
                    f" name: '{lang} feature check')\n"
                )
        try:
            runmeson = subprocess.run(
                ["meson", "setup", "btmp"],
                check=False,
                cwd=tmpdir,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            pytest.skip("meson not present, skipping compiler dependent test", allow_module_level=True)
        return runmeson.returncode == 0
    finally:
        shutil.rmtree(tmpdir)


fortran77_code = '''
C Example Fortran 77 code
      PROGRAM HELLO
      PRINT *, 'Hello, Fortran 77!'
      END
'''

fortran90_code = '''
! Example Fortran 90 code
program hello90
  type :: greeting
    character(len=20) :: text
  end type greeting

  type(greeting) :: greet
  greet%text = 'hello, fortran 90!'
  print *, greet%text
end program hello90
'''

# Dummy class for caching relevant checks
class CompilerChecker:
    def __init__(self):
        self.compilers_checked = False
        self.has_c = False
        self.has_f77 = False
        self.has_f90 = False

    def check_compilers(self):
        if (not self.compilers_checked) and (not sys.platform == "cygwin"):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(check_language, "c"),
                    executor.submit(check_language, "fortran", fortran77_code),
                    executor.submit(check_language, "fortran", fortran90_code)
                ]

                self.has_c = futures[0].result()
                self.has_f77 = futures[1].result()
                self.has_f90 = futures[2].result()

            self.compilers_checked = True


if not IS_WASM:
    checker = CompilerChecker()
    checker.check_compilers()

def has_c_compiler():
    return checker.has_c

def has_f77_compiler():
    return checker.has_f77

def has_f90_compiler():
    return checker.has_f90

def has_fortran_compiler():
    return (checker.has_f90 and checker.has_f77)


#
# Maintaining a temporary module directory
#

_module_dir = None
_module_num = 5403

if sys.platform == "cygwin":
    NUMPY_INSTALL_ROOT = Path(__file__).parent.parent.parent
    _module_list = list(NUMPY_INSTALL_ROOT.glob("**/*.dll"))


def _cleanup():
    global _module_dir
    if _module_dir is not None:
        try:
            sys.path.remove(_module_dir)
        except ValueError:
            pass
        try:
            shutil.rmtree(_module_dir)
        except OSError:
            pass
        _module_dir = None


def get_module_dir():
    global _module_dir
    if _module_dir is None:
        _module_dir = tempfile.mkdtemp()
        atexit.register(_cleanup)
        if _module_dir not in sys.path:
            sys.path.insert(0, _module_dir)
    return _module_dir


def get_temp_module_name():
    # Assume single-threaded, and the module dir usable only by this thread
    global _module_num
    get_module_dir()
    name = "_test_ext_module_%d" % _module_num
    _module_num += 1
    if name in sys.modules:
        # this should not be possible, but check anyway
        raise RuntimeError("Temporary module name already in use.")
    return name


def _memoize(func):
    memo = {}

    def wrapper(*a, **kw):
        key = repr((a, kw))
        if key not in memo:
            try:
                memo[key] = func(*a, **kw)
            except Exception as e:
                memo[key] = e
                raise
        ret = memo[key]
        if isinstance(ret, Exception):
            raise ret
        return ret

    wrapper.__name__ = func.__name__
    return wrapper


#
# Building modules
#


@_memoize
def build_module(source_files, options=[], skip=[], only=[], module_name=None):
    """
    Compile and import a f2py module, built from the given files.

    """

    code = f"import sys; sys.path = {sys.path!r}; import numpy.f2py; numpy.f2py.main()"

    d = get_module_dir()
    # gh-27045 : Skip if no compilers are found
    if not has_fortran_compiler():
        pytest.skip("No Fortran compiler available")

    # Copy files
    dst_sources = []
    f2py_sources = []
    for fn in source_files:
        if not os.path.isfile(fn):
            raise RuntimeError(f"{fn} is not a file")
        dst = os.path.join(d, os.path.basename(fn))
        shutil.copyfile(fn, dst)
        dst_sources.append(dst)

        base, ext = os.path.splitext(dst)
        if ext in (".f90", ".f95", ".f", ".c", ".pyf"):
            f2py_sources.append(dst)

    assert f2py_sources

    # Prepare options
    if module_name is None:
        module_name = get_temp_module_name()
    gil_options = []
    if '--freethreading-compatible' not in options and '--no-freethreading-compatible' not in options:
        # default to disabling the GIL if unset in options
        gil_options = ['--freethreading-compatible']
    f2py_opts = ["-c", "-m", module_name] + options + gil_options + f2py_sources
    f2py_opts += ["--backend", "meson"]
    if skip:
        f2py_opts += ["skip:"] + skip
    if only:
        f2py_opts += ["only:"] + only

    # Build
    cwd = os.getcwd()
    try:
        os.chdir(d)
        cmd = [sys.executable, "-c", code] + f2py_opts
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"Running f2py failed: {cmd[4:]}\n{asunicode(out)}")
    finally:
        os.chdir(cwd)

        # Partial cleanup
        for fn in dst_sources:
            os.unlink(fn)

    # Rebase (Cygwin-only)
    if sys.platform == "cygwin":
        # If someone starts deleting modules after import, this will
        # need to change to record how big each module is, rather than
        # relying on rebase being able to find that from the files.
        _module_list.extend(
            glob.glob(os.path.join(d, f"{module_name:s}*"))
        )
        subprocess.check_call(
            ["/usr/bin/rebase", "--database", "--oblivious", "--verbose"]
            + _module_list
        )

    # Import
    return import_module(module_name)


@_memoize
def build_code(source_code,
               options=[],
               skip=[],
               only=[],
               suffix=None,
               module_name=None):
    """
    Compile and import Fortran code using f2py.

    """
    if suffix is None:
        suffix = ".f"
    with temppath(suffix=suffix) as path:
        with open(path, "w") as f:
            f.write(source_code)
        return build_module([path],
                            options=options,
                            skip=skip,
                            only=only,
                            module_name=module_name)


#
# Building with meson
#


class SimplifiedMesonBackend(MesonBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compile(self):
        self.write_meson_build(self.build_dir)
        self.run_meson(self.build_dir)


def build_meson(source_files, module_name=None, **kwargs):
    """
    Build a module via Meson and import it.
    """

    # gh-27045 : Skip if no compilers are found
    if not has_fortran_compiler():
        pytest.skip("No Fortran compiler available")

    build_dir = get_module_dir()
    if module_name is None:
        module_name = get_temp_module_name()

    # Initialize the MesonBackend
    backend = SimplifiedMesonBackend(
        modulename=module_name,
        sources=source_files,
        extra_objects=kwargs.get("extra_objects", []),
        build_dir=build_dir,
        include_dirs=kwargs.get("include_dirs", []),
        library_dirs=kwargs.get("library_dirs", []),
        libraries=kwargs.get("libraries", []),
        define_macros=kwargs.get("define_macros", []),
        undef_macros=kwargs.get("undef_macros", []),
        f2py_flags=kwargs.get("f2py_flags", []),
        sysinfo_flags=kwargs.get("sysinfo_flags", []),
        fc_flags=kwargs.get("fc_flags", []),
        flib_flags=kwargs.get("flib_flags", []),
        setup_flags=kwargs.get("setup_flags", []),
        remove_build_dir=kwargs.get("remove_build_dir", False),
        extra_dat=kwargs.get("extra_dat", {}),
    )

    backend.compile()

    # Import the compiled module
    sys.path.insert(0, f"{build_dir}/{backend.meson_build_dir}")
    return import_module(module_name)


#
# Unittest convenience
#


class F2PyTest:
    code = None
    sources = None
    options = []
    skip = []
    only = []
    suffix = ".f"
    module = None
    _has_c_compiler = None
    _has_f77_compiler = None
    _has_f90_compiler = None

    @property
    def module_name(self):
        cls = type(self)
        return f'_{cls.__module__.rsplit(".", 1)[-1]}_{cls.__name__}_ext_module'

    @classmethod
    def setup_class(cls):
        if sys.platform == "win32":
            pytest.skip("Fails with MinGW64 Gfortran (Issue #9673)")
        F2PyTest._has_c_compiler = has_c_compiler()
        F2PyTest._has_f77_compiler = has_f77_compiler()
        F2PyTest._has_f90_compiler = has_f90_compiler()
        F2PyTest._has_fortran_compiler = has_fortran_compiler()

    def setup_method(self):
        if self.module is not None:
            return

        codes = self.sources or []
        if self.code:
            codes.append(self.suffix)

        needs_f77 = any(str(fn).endswith(".f") for fn in codes)
        needs_f90 = any(str(fn).endswith(".f90") for fn in codes)
        needs_pyf = any(str(fn).endswith(".pyf") for fn in codes)

        if needs_f77 and not self._has_f77_compiler:
            pytest.skip("No Fortran 77 compiler available")
        if needs_f90 and not self._has_f90_compiler:
            pytest.skip("No Fortran 90 compiler available")
        if needs_pyf and not self._has_fortran_compiler:
            pytest.skip("No Fortran compiler available")

        # Build the module
        if self.code is not None:
            self.module = build_code(
                self.code,
                options=self.options,
                skip=self.skip,
                only=self.only,
                suffix=self.suffix,
                module_name=self.module_name,
            )

        if self.sources is not None:
            self.module = build_module(
                self.sources,
                options=self.options,
                skip=self.skip,
                only=self.only,
                module_name=self.module_name,
            )


#
# Helper functions
#


def getpath(*a):
    # Package root
    d = Path(numpy.f2py.__file__).parent.resolve()
    return d.joinpath(*a)


@contextlib.contextmanager
def switchdir(path):
    curpath = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(curpath)