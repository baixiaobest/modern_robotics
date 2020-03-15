from __future__ import print_function

import unittest
import numpy as np
import Utility as util
import modern_robotics as mr

def array_equal(A, B, tolerance=0.01):
    if A.shape != B.shape:
        return False

    for i in range(len(A)):
        if np.abs(A[i] - B[i]) > tolerance:
            return False

    return True


def matrix_equal(A, B, tolerance=0.001):
    if A.shape != B.shape:
        return False

    for i in range(len(A)):
        if not array_equal(A[i], B[i], tolerance):
            return False
    return True

class FKTest(unittest.TestCase):
    def test_fk_space(self):
        M = np.array([[-1, 0, 0, 0],
                      [0, 1, 0, 6],
                      [0, 0, -1, 2],
                      [0, 0, 0, 1]])
        Slist = np.array([[0, 0, 1, 4, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
        correct_output = np.array([[0, 1, 0, -5],
                                   [1, 0, 0, 4],
                                   [0, 0, -1, 1.68584073],
                                   [0, 0, 0, 1]])

        self.assertEqual(True, matrix_equal(correct_output, util.forward_kinematics_space(M, Slist, thetalist)))

    def test_fk_body(self):
        M = np.array([[-1, 0, 0, 0],
                      [0, 1, 0, 6],
                      [0, 0, -1, 2],
                      [0, 0, 0, 1]])
        Blist = np.array([[0, 0, -1, 2, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])

        correct_output = np.array([[0, 1,  0,         -5],
                                   [1, 0,  0,          4],
                                   [0, 0, -1, 1.68584073],
                                   [0, 0,  0,          1]])
        self.assertEqual(True, matrix_equal(correct_output, util.forward_kinematics_body(M, Blist, thetalist)))

    def test_jacobian_space(self):
        Slist = np.array([[0, 0, 1, 0, 0.2, 0.2],
                          [1, 0, 0, 2, 0, 3],
                          [0, 1, 0, 0, 2, 1],
                          [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])

        correct_output = np.array(
                  [[  0, 0.98006658, -0.09011564,  0.95749426],
                  [  0, 0.19866933,   0.4445544,  0.28487557],
                  [  1,          0,  0.89120736, -0.04528405],
                  [  0, 1.95218638, -2.21635216, -0.51161537],
                  [0.2, 0.43654132, -2.43712573,  2.77535713],
                  [0.2, 2.96026613,  3.23573065,  2.22512443]])

        self.assertEqual(True, matrix_equal(correct_output, util.jacobian_space(Slist, thetalist)))

    def test_jacobian_body(self):
        Blist = np.array([[0, 0, 1, 0, 0.2, 0.2],
                          [1, 0, 0, 2, 0, 3],
                          [0, 1, 0, 0, 2, 1],
                          [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])

        correct_output = np.array(
                  [[-0.04528405, 0.99500417,           0,   1],
                  [ 0.74359313, 0.09304865,  0.36235775,   0],
                  [-0.66709716, 0.03617541, -0.93203909,   0],
                  [ 2.32586047,    1.66809,  0.56410831, 0.2],
                  [-1.44321167, 2.94561275,  1.43306521, 0.3],
                  [-2.06639565, 1.82881722, -1.58868628, 0.4]])

        self.assertEqual(True, matrix_equal(correct_output, util.jacobian_body(Blist, thetalist)))

    def test_IK_space(self):
        Slist = np.array([[0, 0, 1, 4, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        M = np.array([[-1, 0, 0, 0],
                      [0, 1, 0, 6],
                      [0, 0, -1, 2],
                      [0, 0, 0, 1]])
        T = np.array([[0, 1, 0, -5],
                      [1, 0, 0, 4],
                      [0, 0, -1, 1.6858],
                      [0, 0, 0, 1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001

        correct_theta = np.array([ 1.57073783,  2.99966384,  3.1415342 ])
        success = True

        output_thetas, output_success = util.IK_space(Slist, M, T, thetalist0, eomg, ev)

        self.assertEqual(True, array_equal(correct_theta, output_thetas))
        self.assertEqual(success, output_success)

    def test_IK_body(self):
        Blist = np.array([[0, 0, -1, 2, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0, 0, 0],
                      [0, 1, 0, 6],
                      [0, 0, -1, 2],
                      [0, 0, 0, 1]])
        T = np.array([[0, 1, 0, -5],
                      [1, 0, 0, 4],
                      [0, 0, -1, 1.6858],
                      [0, 0, 0, 1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001

        correct_theta = np.array([1.57073819, 2.999667, 3.14153913])
        success = True

        output_thetas, output_success = util.IK_body(Blist, M, T, thetalist0, eomg, ev)

        self.assertEqual(True, array_equal(correct_theta, output_thetas))
        self.assertEqual(success, output_success)

    def test_inverse_dynamics(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        g = np.array([0, 0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0, 1]])
        M12 = np.array([[0, 0, 1, 0.28],
                        [0, 1, 0, 0.13585],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        M23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1, 0.395],
                        [0, 0, 0, 1]])
        M34 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0, 1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1, 0, 1, 0],
                          [0, 1, 0, -0.089, 0, 0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        correct_output = mr.InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)

        output = util.InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)

        self.assertEqual(True, array_equal(correct_output, output))


    def test_mass_matrix(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        M01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0, 1]])
        M12 = np.array([[0, 0, 1, 0.28],
                        [0, 1, 0, 0.13585],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        M23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1, 0.395],
                        [0, 0, 0, 1]])
        M34 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0, 1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1, 0, 1, 0],
                          [0, 1, 0, -0.089, 0, 0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T

        correct_output = mr.MassMatrix(thetalist, Mlist, Glist, Slist)

        output = util.MassMatrix(thetalist, Mlist, Glist, Slist)

        self.assertEqual(True, matrix_equal(correct_output, output))


    def test_vel_quadratic_forces(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        M01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0, 1]])
        M12 = np.array([[0, 0, 1, 0.28],
                        [0, 1, 0, 0.13585],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        M23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1, 0.395],
                        [0, 0, 0, 1]])
        M34 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0, 1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1, 0, 1, 0],
                          [0, 1, 0, -0.089, 0, 0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T

        correct_output = mr.VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
        output = mr.VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)

        self.assertEqual(True, array_equal(correct_output, output))

    def test_gravity_forces(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        g = np.array([0, 0, -9.8])
        M01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0, 1]])
        M12 = np.array([[0, 0, 1, 0.28],
                        [0, 1, 0, 0.13585],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        M23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1, 0.395],
                        [0, 0, 0, 1]])
        M34 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0, 1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1, 0, 1, 0],
                          [0, 1, 0, -0.089, 0, 0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        correct_output = mr.GravityForces(thetalist, g, Mlist, Glist, Slist)
        output = util.GravityForces(thetalist, g, Mlist, Glist, Slist)

        self.assertEqual(True, array_equal(correct_output, output))

    def test_end_effector_forces(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0, 1]])
        M12 = np.array([[0, 0, 1, 0.28],
                        [0, 1, 0, 0.13585],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        M23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1, 0.395],
                        [0, 0, 0, 1]])
        M34 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0, 1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1, 0, 1, 0],
                          [0, 1, 0, -0.089, 0, 0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        correct_output = mr.EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)
        output = util.EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)

        self.assertEqual(True, array_equal(correct_output, output))

    def test_forward_dynamics(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        taulist = np.array([0.5, 0.6, 0.7])
        g = np.array([0, 0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0, 1]])
        M12 = np.array([[0, 0, 1, 0.28],
                        [0, 1, 0, 0.13585],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        M23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1, 0.395],
                        [0, 0, 0, 1]])
        M34 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0, 1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1, 0, 1, 0],
                          [0, 1, 0, -0.089, 0, 0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        correct_output = mr.ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist)

        output = util.ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist)

        self.assertEqual(True, array_equal(correct_output, output))

    def test_inverse_dynamics_trajectory(self):
        # Create a trajectory to follow using functions from Chapter 9
        thetastart = np.array([0, 0, 0])
        thetaend = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        Tf = 3
        N = 1000
        method = 5
        traj = mr.JointTrajectory(thetastart, thetaend, Tf, N, method)
        thetamat = np.array(traj).copy()
        dthetamat = np.zeros((1000, 3))
        ddthetamat = np.zeros((1000, 3))
        dt = Tf / (N - 1.0)
        for i in range(np.array(traj).shape[0] - 1):
            dthetamat[i + 1, :] = (thetamat[i + 1, :] - thetamat[i, :]) / dt
            ddthetamat[i + 1, :] \
                = (dthetamat[i + 1, :] - dthetamat[i, :]) / dt
        # Initialize robot description (Example with 3 links)
        g = np.array([0, 0, -9.8])
        Ftipmat = np.ones((N, 6))
        M01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0, 1]])
        M12 = np.array([[0, 0, 1, 0.28],
                        [0, 1, 0, 0.13585],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        M23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1, 0.395],
                        [0, 0, 0, 1]])
        M34 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0, 1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1, 0, 1, 0],
                          [0, 1, 0, -0.089, 0, 0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        correct_taumat \
            = mr.InverseDynamicsTrajectory(thetamat, dthetamat, ddthetamat, g, Ftipmat, Mlist, Glist, Slist)
        taumat = util.InverseDynamicsTrajectory(thetamat, dthetamat, ddthetamat, g, Ftipmat, Mlist, Glist, Slist)

        self.assertTrue(True, matrix_equal(correct_taumat, taumat))


    def test_forward_dynamics_trajectory(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        taumat = np.array([[3.63, -6.58, -5.57], [3.74, -5.55, -5.5],
                           [4.31, -0.68, -5.19], [5.18, 5.63, -4.31],
                           [5.85, 8.17, -2.59], [5.78, 2.79, -1.7],
                           [4.99, -5.3, -1.19], [4.08, -9.41, 0.07],
                           [3.56, -10.1, 0.97], [3.49, -9.41, 1.23]])
        # Initialize robot description (Example with 3 links)
        g = np.array([0, 0, -9.8])
        Ftipmat = np.ones((np.array(taumat).shape[0], 6))
        M01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0, 1]])
        M12 = np.array([[0, 0, 1, 0.28],
                        [0, 1, 0, 0.13585],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        M23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1, 0.395],
                        [0, 0, 0, 1]])
        M34 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0, 1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1, 0, 1, 0],
                          [0, 1, 0, -0.089, 0, 0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        dt = 0.1
        intRes = 8
        correct_thetamat, correct_dthetamat \
            = mr.ForwardDynamicsTrajectory(thetalist, dthetalist, taumat, g, \
                                           Ftipmat, Mlist, Glist, Slist, dt, \
                                           intRes)
        thetamat, dthetamat = util.ForwardDynamicsTrajectory(thetalist, dthetalist, taumat, g, \
                                           Ftipmat, Mlist, Glist, Slist, dt, \
                                           intRes)

        self.assertTrue(True, matrix_equal(correct_thetamat, thetamat))
        self.assertTrue(True, matrix_equal(correct_dthetamat, dthetamat))

if __name__ == '__main__':
    unittest.main()
