import numpy as np
import matplotlib.pyplot as plt
import sphere_fedmanifold

n = 1
K = 1
d_list = [100, 200, 500]
# TODO: repeat 10 times
for i in range(3):
    d = d_list[i]
    PCA = sphere_fedmanifold.problem_PCA(n, K, d=d, p=1)
    print("Test on sphere d = %d" % (d))

    dist0 = 0
    dist1_1 = dist1_2 = 0
    dist2_1 = dist2_2 = 0
    for rep in range(10):
        # generate xt
        xt = PCA.manifold.rand()

        # generate xi, i=0,...,99
        x_list = np.zeros((100, d))
        for i in range(100):
            x_list[i] = PCA.manifold.rand()

        # run Karcher mean
        xplus1, time1 = PCA.karcher_mean(xt, x_list)

        # run tangent space mean
        xplus2, time2 = PCA.tangent_space_mean(xt, x_list)

        # calculate the distances
        for i in range(100):
            dist0 += PCA.manifold.dist(xt, x_list[i])**2

        dist1_1 += PCA.manifold.dist(xt, xplus1) ** 2
        for i in range(100):
            dist1_2 += PCA.manifold.dist(xplus1, x_list[i]) ** 2

        dist2_1 += PCA.manifold.dist(xt, xplus2) ** 2
        for i in range(100):
            dist2_2 += PCA.manifold.dist(xplus2, x_list[i]) ** 2

    print("distance from xt to all the generated points: %f" % (dist0 / 1000))

    print("Result for Karcher mean method: time is %f" % (time1))
    print("d(xt, x_{t+1})): %f" % (dist1_1 / 10))
    print("sum_i d(x_i, x_{t+1})): %f" % (dist1_2 / 1000))

    print("Result for tangent space mean method: time is %f" % (time2))
    print("d(xt, x_{t+1})): %f" % (dist2_1 / 10))
    print("sum_i d(x_i, x_{t+1})): %f\n" % (dist2_2 / 1000))
