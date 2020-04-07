import modern_robotics as mr

'''
M: Configuration of end effector.
Slist: A list of screw axes.
thetas: A list of theta vales.
'''
import numpy as np

def forward_kinematics_space(M, Slist, thetas):
    T = np.array(M).copy().astype(np.float)
    for i in range(len(thetas)-1, -1, -1):
        T = mr.MatrixExp6(mr.VecTose3(Slist[:, i] * thetas[i])) @ T
    return T

def forward_kinematics_body(M, Blist, thetas):
    T = np.array(M).copy().astype(np.float)
    for i in range(len(thetas)):
        T = T @ mr.MatrixExp6(mr.VecTose3(Blist[:, i] * thetas[i]))
    return T

def jacobian_space(Slist, thetas):
    jac = np.array(Slist).copy().astype(np.float)
    T = np.identity(4)
    for i in range(1, len(thetas), 1):
        # Rigid body transformation from root to current joint i.
        T = T @ mr.MatrixExp6(mr.VecTose3(Slist[:, i - 1] * thetas[i - 1]))
        # Adjoint matrix of this transfomation.
        adj = mr.Adjoint(T)
        jac[:, i] = adj @ Slist[:, i]
    return jac

def jacobian_body(Blist, thetas):
    jac = np.array(Blist).copy().astype(np.float)
    T = np.identity(4)
    for i in range(len(thetas)-2, -1, -1):
        T = T @ mr.MatrixExp6(mr.VecTose3(-Blist[:, i + 1] * thetas[i + 1]))
        adj = mr.Adjoint(T)
        jac[:, i] = adj @ Blist[:, i]
    return jac

'''
Inverse kinematics
Slist: List of screws at zero configuration.
M: SE3 configuration of end effector in zero configuration.
T: Desired end effector configuration
thetas_guess: Initial theta guess.
eomg: Angular error to satisfy.
ev: Linear error to satisfy.
max_iteration: Maximum number of iteration.
return (thetas, success): thetas is joint configuration. success is
true when result is found.
'''
def IK_space(Slist, M, T, thetas_guess, eomg, ev, max_iteration=20):
    thetas = np.array(thetas_guess).copy().astype(np.float)
    Tsd = np.array(T).copy().astype(np.float)
    success = False

    for i in range(max_iteration):
        # Calculate space frame transform from end effector to desired position.
        Tsb = forward_kinematics_space(M, Slist, thetas)
        Tbd = Tsd @ np.linalg.inv(Tsb)
        # Twist in matrix form.
        Vs_se3 = mr.MatrixLog6(Tbd)
        Vs = mr.se3ToVec(Vs_se3)

        # Check if result is a success.
        if np.linalg.norm(Vs[0:3]) < eomg and np.linalg.norm(Vs[3:6]) < ev:
            success = True
            break
        Jac = jacobian_space(Slist, thetas)
        thetas = (thetas + np.linalg.pinv(Jac) @ Vs) % (2 * np.pi)

    return (thetas, success)


def IK_body(Blist, M, T, thetas_guess, eomg, ev, max_iteration=20):
    thetas = np.array(thetas_guess).copy().astype(np.float)
    Tsd = np.array(T).copy().astype(np.float)
    success = False

    for i in range(max_iteration):
        # Calculate space frame transform from end effector to desired position.
        Tsb = forward_kinematics_body(M, Blist, thetas)
        Tbd = np.linalg.inv(Tsb) @ Tsd
        # Twist in matrix form.
        Vb_se3 = mr.MatrixLog6(Tbd)
        Vb = mr.se3ToVec(Vb_se3)

        # Check if result is a success.
        if np.linalg.norm(Vb[0:3]) < eomg and np.linalg.norm(Vb[3:6]) < ev:
            success = True
            break
        Jac = jacobian_body(Blist, thetas)
        thetas = (thetas + np.linalg.pinv(Jac) @ Vb) % (2 * np.pi)

    return (thetas, success)


"""Computes inverse dynamics in the space frame for an open chain robot

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The n-vector of required joint forces/torques
    """
def InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
    V_i_list = []
    dV_i_list = []
    T_i_prev_list = []
    A_i_list = []
    torques = np.zeros(thetalist.shape[0])

    ''' Begin forward iteration. '''

    n = len(thetalist)
    # Previous twist
    V_prev = np.zeros(6)
    dV_prev = np.zeros(6)
    dV_prev[3:6] = -np.array(g)
    M_i = np.identity(4)

    for i in range(n):
        # Home position configuration of current i frame relative to space fixed frame.
        M_i = M_i @ Mlist[i]
        # Screw axis Si of the joint i expressed in i frame.
        A_i = mr.Adjoint(mr.TransInv(M_i)) @ Slist[:, i]
        # Configuration of i-1 frame relative to i frame with theta_i at joint i.
        T_i_prev = mr.MatrixExp6(-mr.VecTose3(A_i) * thetalist[i]) @ mr.TransInv(Mlist[i])
        # Twist of link i in frame i.
        V_i = A_i * dthetalist[i] + mr.Adjoint(T_i_prev) @ V_prev
        dV_i = A_i * ddthetalist[i] + mr.Adjoint(T_i_prev) @ dV_prev + mr.ad(V_i) @ A_i * dthetalist[i]

        # Store these for backward iteration.
        A_i_list.append(A_i)
        V_i_list.append(V_i)
        dV_i_list.append(dV_i)
        T_i_prev_list.append(T_i_prev)

        # Store V and dV for next iteration.
        V_prev = V_i
        dV_prev = dV_i

    # Add the transformation from robotic arm tip to last link.
    T_i_prev_list.append(mr.TransInv(Mlist[n]))

    ''' Begin backward iteration. '''
    # Wrench in frame i + 1
    F_next = Ftip
    for i in range(n-1, -1, -1):
        F_i = mr.Adjoint(T_i_prev_list[i + 1]).T @ F_next \
            + Glist[i] @ dV_i_list[i] - mr.ad(V_i_list[i]).T @ Glist[i] @ V_i_list[i]
        torques[i] = F_i.T @ A_i_list[i]

        F_next = F_i

    return torques


"""Computes the mass matrix of an open chain robot based on the given
    configuration

    :param thetalist: A list of joint variables
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The numerical inertia matrix M(thetalist) of an n-joint serial
             chain at the given configuration thetalist
"""
def MassMatrix(thetalist, Mlist, Glist, Slist):
    n = len(thetalist)
    M = np.zeros((n, n))
    dthetalist = np.zeros(n)
    Ftip = np.zeros(6)
    g = np.zeros(3)

    for i in range(n):
        ddthetalist = np.zeros(n)
        ddthetalist[i] = 1.0
        tau = InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)
        M[i, :] = tau

    return M



"""
Computes the Coriolis and centripetal terms in the inverse dynamics of
an open chain robot

:param thetalist: A list of joint variables,
:param dthetalist: A list of joint rates,
:param Mlist: List of link frames i relative to i-1 at the home position,
:param Glist: Spatial inertia matrices Gi of the links,
:param Slist: Screw axes Si of the joints in a space frame, in the format
                of a matrix with axes as the columns.
:return: The vector c(thetalist,dthetalist) of Coriolis and centripetal
         terms for a given thetalist and dthetalist.
"""
def VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist):
    n = len(thetalist)
    ddthetalist = np.zeros(n)
    Ftip = np.zeros(6)
    g = np.zeros(3)
    return InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)


"""Computes the joint forces/torques an open chain robot requires to
    overcome gravity at its configuration

    :param thetalist: A list of joint variables
    :param g: 3-vector for gravitational acceleration
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return grav: The joint forces/torques required to overcome gravity at
                  thetalist
"""
def GravityForces(thetalist, g, Mlist, Glist, Slist):
    n = len(thetalist)
    dthetalist = np.zeros(n)
    ddthetalist = np.zeros(n)
    Ftip = np.zeros(6)

    return InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)



"""Computes the joint forces/torques an open chain robot requires only to
    create the end-effector force Ftip

    :param thetalist: A list of joint variables
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The joint forces and torques required only to create the
             end-effector force Ftip
"""
def EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist):
    n = len(thetalist)
    dthetalist = np.zeros(n)
    ddthetalist = np.zeros(n)
    g = np.zeros(3)

    return InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)


"""Computes forward dynamics in the space frame for an open chain robot

    :param thetalist: A list of joint variables
    :param dthetalist: A list of joint rates
    :param taulist: An n-vector of joint forces/torques
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The resulting joint accelerations
"""
def ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist):
    M = MassMatrix(thetalist, Mlist, Glist, Slist)
    VQF = VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
    G = GravityForces(thetalist, g, Mlist, Glist, Slist)
    EFF = EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)

    return np.linalg.inv(M) @ (taulist - VQF - G - EFF)


"""Compute the joint angles and velocities at the next timestep using            from here
    first order Euler integration

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param dt: The timestep delta t
    :return thetalistNext: Vector of joint variables after dt from first
                           order Euler integration
    :return dthetalistNext: Vector of joint rates after dt from first order
                            Euler integration
"""
def EulerStep(thetalist, dthetalist, ddthetalist, dt):
    thetalist_next = thetalist + dthetalist * dt
    dthetalist_next = dthetalist + ddthetalist * dt
    return thetalist_next, dthetalist_next


"""Calculates the joint forces/torques required to move the serial chain
    along the given trajectory using inverse dynamics

    :param thetamat: An N x n matrix of robot joint variables
    :param dthetamat: An N x n matrix of robot joint velocities
    :param ddthetamat: An N x n matrix of robot joint accelerations
    :param g: Gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector (If there are no tip forces the user should
                    input a zero and a zero matrix will be used)
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The N x n matrix of joint forces/torques for the specified
             trajectory, where each of the N rows is the vector of joint
             forces/torques at each time step
    """
def InverseDynamicsTrajectory(thetamat, dthetamat, ddthetamat, g, Ftipmat, Mlist, Glist, Slist):
    N, n = thetamat.shape
    taulist = np.zeros((N, n))
    for i in range(N):
        taulist[i, :] = InverseDynamics(thetamat[i, :], dthetamat[i, :], ddthetamat[i, :], g, Ftipmat[i, :], \
                                        Mlist, Glist, Slist)
    return taulist


"""Simulates the motion of a serial chain given an open-loop history of
    joint forces/torques

    :param thetalist: n-vector of initial joint variables
    :param dthetalist: n-vector of initial joint rates
    :param taumat: An N x n matrix of joint forces/torques, where each row is
                   the joint effort at any time step
    :param g: Gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector (If there are no tip forces the user should
                    input a zero and a zero matrix will be used)
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :param dt: The timestep between consecutive joint forces/torques
    :param intRes: Integration resolution is the number of times integration
                   (Euler) takes places between each time step. Must be an
                   integer value greater than or equal to 1
    :return thetamat: The N x n matrix of robot joint angles resulting from
                      the specified joint forces/torques
    :return dthetamat: The N x n matrix of robot joint velocities
"""
def ForwardDynamicsTrajectory(thetalist, dthetalist, taumat, g, Ftipmat, Mlist, Glist, Slist, dt, intRes):
    euler_dt = dt / intRes
    N, n = taumat.shape
    thetamat = np.zeros((N, n))
    dthetamat = np.zeros((N, n))
    curr_thetalist = thetalist
    curr_dthetalist = dthetalist
    for i in range(N):
        for j in range(intRes):
            ddthetalist = ForwardDynamics(curr_thetalist, curr_dthetalist, taumat[i, :], g, Ftipmat[i, :], Mlist, Glist, Slist)
            curr_thetalist, curr_dthetalist = EulerStep(curr_thetalist, curr_dthetalist, ddthetalist, euler_dt)
        thetamat[i, :] = curr_thetalist.copy()
        dthetamat[i, :] = curr_dthetalist.copy()
    return thetamat, dthetamat


"""Computes s(t) for a cubic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a third-order
             polynomial motion that begins and ends at zero velocity
"""
def CubicTimeScaling(Tf, t):
    a2 = 3 / (Tf ** 2)
    a3 = -2 / (Tf ** 3)
    return a2 * t ** 2 + a3 * t ** 3

"""Computes s(t) for a quintic time scaling
    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a fifth-order
             polynomial motion that begins and ends at zero velocity and zero
             acceleration
"""
def QuinticTimeScaling(Tf, t):
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
           + 6 * (1.0 * t / Tf) ** 5


"""Computes a straight-line trajectory in joint space

    :param thetastart: The initial joint variables
    :param thetaend: The final joint variables
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: A trajectory as an N x n matrix, where each row is an n-vector
             of joint variables at an instant in time. The first row is
             thetastart and the Nth row is thetaend . The elapsed time
             between each row is Tf / (N - 1)
"""
def JointTrajectory(thetastart, thetaend, Tf, N, method=3):
    dt = Tf / (N - 1.0)
    trajectory = np.zeros((N, len(thetastart)))
    for i in range(N):
        t = i * dt
        s = 0
        if method == 3:
            s = CubicTimeScaling(Tf, t)
        elif method == 5:
            s = QuinticTimeScaling(Tf, t)
        trajectory[i] = (1 - s) * thetastart + s * thetaend

    return trajectory


"""Computes a trajectory as a list of N SE(3) matrices corresponding to
      the screw motion about a space screw axis

    :param Xstart: The initial end-effector configuration
    :param Xend: The final end-effector configuration
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: The discretized trajectory as a list of N matrices in SE(3)
             separated in time by Tf/(N-1). The first in the list is Xstart
             and the Nth is Xend
"""
def ScrewTrajectory(Xstart, Xend, Tf, N, method=3):
    dt = Tf / (N - 1.0)
    trajectory = []
    Xstart_end = mr.TransInv(Xstart) @ Xend
    for i in range(N):
        t = i * dt
        s = 0
        if method == 3:
            s = CubicTimeScaling(Tf, t)
        elif method == 5:
            s = QuinticTimeScaling(Tf, t)
        trajectory.append(Xstart @ mr.MatrixExp6(mr.MatrixLog6(Xstart_end) * s))
    return trajectory


"""Computes a trajectory as a list of N SE(3) matrices corresponding to
    the origin of the end-effector frame following a straight line

    :param Xstart: The initial end-effector configuration
    :param Xend: The final end-effector configuration
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: The discretized trajectory as a list of N matrices in SE(3)
             separated in time by Tf/(N-1). The first in the list is Xstart
             and the Nth is Xend
    This function is similar to ScrewTrajectory, except the origin of the
    end-effector frame follows a straight line, decoupled from the rotational
    motion.
"""
def CartesianTrajectory(Xstart, Xend, Tf, N, method):
    dt = Tf / (N - 1.0)
    trajectory = []

    Rstart = Xstart[0:3, 0:3]
    Rend = Xend[0:3, 0:3]
    Rstart_end = mr.RotInv(Rstart) @ Rend
    tstart = Xstart[0:3, 3]
    tend = Xend[0:3, 3]

    for i in range(N):
        t = i * dt
        s = 0
        if method == 3:
            s = CubicTimeScaling(Tf, t)
        elif method == 5:
            s = QuinticTimeScaling(Tf, t)
        Rs = Rstart @ mr.MatrixExp3(mr.MatrixLog3(Rstart_end) * s)
        ts = tstart * (1 - s) + tend * s
        T = np.identity(4)
        T[0:3, 0:3] = Rs
        T[0:3, 3] = ts
        trajectory.append(T)

    return trajectory
