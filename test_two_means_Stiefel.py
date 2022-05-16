import numpy as np
import matplotlib.pyplot as plt
import Stiefel_fedmanifold

n = 1
K = 1
d_list = [50, 100, 200, 500]

for i in range(4):
    d = d_list[i]
    kPCA = Stiefel_fedmanifold.problem_PCA(n, K, d=d, p=1)
    print("Test on (d, r) = (%d, %d)" % (d, 5))

    # generate xt
    xt = kPCA.manifold.rand()

    # generate xi, i=0,...,99
    x_list = np.zeros((100, d, 5))
    for i in range(100):
        x_list[i] = kPCA.manifold.rand()

    # run Karcher mean
    xplus1, time1 = kPCA.karcher_mean(xt, x_list)

    # run tangent space mean
    xplus2, time2 = kPCA.tangent_space_mean(xt, x_list)

    # calculate the distances
    dist0 = 0
    for i in range(100):
        dist0 += np.linalg.norm(xt - x_list[i], "fro")
    print("distance from xt to all the generated points: %f" % (dist0 / 100))

    print("Result for Karcher mean method: time is %f" % (time1))
    dist1_1 = dist1_2 = 0
    dist1_1 = np.linalg.norm(xt - xplus1, "fro")
    for i in range(100):
        dist1_2 += np.linalg.norm(xplus1 - x_list[i], "fro")
    print("d(xt, x_{t+1})): %f" % (dist1_1))
    print("sum_i d(x_i, x_{t+1})): %f" % (dist1_2 / 100))

    print("Result for tangent space mean method: time is %f" % (time2))
    dist2_1 = dist2_2 = 0
    dist2_1 = np.linalg.norm(xt - xplus2, "fro")
    for i in range(100):
        dist2_2 += np.linalg.norm(xplus2 - x_list[i], "fro")
    print("d(xt, x_{t+1})): %f" % (dist2_1))
    print("sum_i d(x_i, x_{t+1})): %f" % (dist2_2 / 100))
