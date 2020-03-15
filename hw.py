import modern_robotics as mr
import numpy as np
import csv


def IKinBodyIterates(Blist, M, T, thetas_guess, eomg, ev, max_iteration=20):
    thetas = np.array(thetas_guess).copy().astype(np.float)
    Tsd = np.array(T).copy().astype(np.float)
    success = False
    guesses = [thetas_guess]

    print("Initial Guess: {0}".format(thetas))

    for i in range(max_iteration):
        print("Iteration {0}\n".format(i))
        # Calculate space frame transform from end effector to desired position.
        Tsb = mr.FKinBody(M, Blist, thetas)
        print("joint vector: {0}".format(thetas))
        print("SE3 end effector config: \n{0}".format(Tsb))
        Tbd = np.linalg.inv(Tsb) @ Tsd
        # Twist in matrix form.
        Vb_se3 = mr.MatrixLog6(Tbd)
        Vb = mr.se3ToVec(Vb_se3)
        print("error twist: {0}".format(Vb))
        print("angular error magitude || omega_b ||: {0}".format(np.linalg.norm(Vb[0:3])))
        print("linear error magnitude || v_b ||: {0}".format(np.linalg.norm(Vb[3:6])))
        print("-------------------\n")
        # Check if result is a success.
        if np.linalg.norm(Vb[0:3]) < eomg and np.linalg.norm(Vb[3:6]) < ev:
            success = True
            break
        Jac = mr.JacobianBody(Blist, thetas)
        thetas = (thetas + np.linalg.pinv(Jac) @ Vb) % (2 * np.pi)
        guesses.append(thetas)

    return (thetas, success, np.array(guesses))

if __name__ == "__main__":
    W1 = 0.109
    W2 = 0.082
    L1 = 0.425
    L2 = 0.392
    H1 = 0.089
    H2 = 0.095
    Blist = np.array([[0, 1,  0, W1 + W2, 0,        L1 + L2],
                      [0, 0,  1, H2,      -L1 - L2, 0],
                      [0, 0,  1, H2,      -L2,      0],
                      [0, 0,  1, H2,      0,        0],
                      [0, -1, 0, -W2,     0,        0],
                      [0, 0,  1, 0,       0,        0]]).T
    T = np.array([[0,  1, 0,  -0.5],
                  [0,  0, -1, 0.1],
                  [-1, 0, 0,  0.1],
                  [0,  0, 0,  1]])
    M = np.array([[-1, 0, 0, L1 + L2],
                  [0,  0, 1, W1 + W2],
                  [0,  1, 0, H1 - H2],
                  [0,  0, 0, 1]])
    eomg = 0.001
    ev = 0.0001
    thetas = [5, 3, 4, 1, 3, 2]

    thetas_res, success, guesses = IKinBodyIterates(Blist,M, T, thetas, eomg, ev)

    with open('iterates.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        h, w = guesses.shape
        for i in range(h):
            writer.writerow(guesses[i])

    if success:
        print("success")
    else:
        print("fail")

