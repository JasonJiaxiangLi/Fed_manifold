import manifolds
import random
import numpy as np
import time
from scipy.linalg import subspace_angles
import matplotlib.pyplot as plt

class problem_PCA(object):
    """
    This is the class for testing the manifold fedprox and fedlin algorithm
    """
    def __init__(self, n, K, d, p, r=5, A=None):
        self.n = n  # number of clients
        self.K = K  # number of clients selected each iter
        self.d = d  # dim
        self.p = p  # dim
        self.r = r  # dim
        self.manifold = manifolds.stiefel.Stiefel(d, r)
        if A is None:
            # np.random.seed(3432)
            A = np.zeros((n, d, d))
            for i in range(n):
                temp = np.random.normal(size=(d, p))
                A[i] = temp.dot(temp.T)
        self.A = A
        print("Create problem class: kpca with d=%d, r=%d" % (d, r))

        # below we calculate the true PCA
        fullA = np.zeros((d, d))
        for i in range(n):
            fullA += A[i]
        self.fullA = fullA / self.n
        s, v = np.linalg.eig(self.fullA)
        idx = s.argsort()[::-1]
        s = s[idx]
        v = v[:, idx]
        self.x_star = v[:, 0:r]
        self.f_star = - sum(s[0:r]) / 2
        print("true function value %f" % (self.f_star))

    def fi(self, x, i):
        return - np.trace(x.T.dot(self.A[i]).dot(x)) / 2

    def f(self, x):
        return sum([self.fi(x, i) for i in range(self.n)]) / self.n

    def grad_fi(self, x, i):
        return - self.A[i].dot(x)

    def grad_f(self, x):
        grad = np.zeros((self.d, self.r))
        for i in range(self.n):
            grad += self.grad_fi(x, i)
        return grad / self.n

    def grad_dist(self, x, x0):
        """
        calculate the gradient of the distance square d^2(x, x0)
        where x is the variable
        i.e. calculate Exp^{-1}_{x}(x_0)
        this is hard for the Stiefel manifold
        we use inverse of the retraction mapping instead
        the distance is calculated by manifold.dist(x, x0)
        """
        return self.manifold.inv_retr(x, x0)

    def karcher_mean(self, x, x_list, eta=1e-8, max_iter=100, eps=1e-8):
        """
        compute the mean of given x points, using only the Riemannian gradient descent
        :param x_list: d by K, each column is one vector
        """
        K, d, r = np.shape(x_list)
        time0 = time.time()
        for i in range(max_iter):
            grad = np.zeros((d, r))
            # loss = 0
            for j in range(K):
                xj = x_list[j]
                # loss += self.manifold.dist(x, xj)
                grad += self.grad_dist(x, xj)
            grad /= K
            grad = self.manifold.proj(x, grad)
            norm_grad = np.linalg.norm(grad, 'fro')
            # print("iter %d, grad for karcher mean = %f" % (i, norm_grad))
            if norm_grad <= eps:
                break
            # loss /= K
            # print("iter %d, loss value for karcher mean = %f" % (i, loss))
            x = self.manifold.retr(x, -eta * grad)
        return x, time.time() - time0

    def tangent_space_mean(self, x, x_list):
        """
        compute the mean of given x points on the tangent space
        :param x: reference point
        :param g_list: d by K, each column is one vector
        """
        K, d, r = np.shape(x_list)
        g = np.zeros((d, r))
        time0 = time.time()
        for j in range(K):
            # loss += self.manifold.dist(x, xj)
            g += self.manifold.inv_retr(x, x_list[j])
        return self.manifold.retr(x, g / K), time.time() - time0

    def fedprox(self, outer_iter, inner_iter, mu, eta=1e-2, epsilon=1e-8, flag=0, x0=None):
        fval_list = np.zeros(outer_iter + 1)
        norm_list = np.zeros(outer_iter + 1)
        angle_list = np.zeros(outer_iter + 1)  # the principal angles
        time_list = np.zeros(outer_iter + 1)

        if x0 is None:
            x = self.manifold.rand()
        else:
            x = x0
        fval_list[0] = self.f(x)
        angle_list[0] = sum(subspace_angles(x, self.x_star))
        norm_list[0] = np.linalg.norm(self.manifold.proj(x, self.grad_f(x)))
        x_list = np.tile(x, (self.K, 1, 1))
        # g_list = np.zeros((self.K, self.d, self.r))
        for iter in range(1, outer_iter + 1):

            time0 = time.time()
            # optimization for each client
            batch = random.sample(range(self.n), self.K)
            count = 0
            aveg_inner = 0
            aveg_norm = 0.0
            for i in batch:  # do gradient descent for each client
                xi = x_list[count]
                t = 0
                for t in range(inner_iter):
                    grad = self.grad_fi(xi, i) + mu * self.grad_dist(xi, x)
                    grad = self.manifold.proj(xi, grad)
                    if np.linalg.norm(grad) <= epsilon:
                        break
                    xi = self.manifold.retr(xi, -eta * grad)
                aveg_inner += t + 1
                aveg_norm += np.linalg.norm(grad, 'fro')
                # g_list[count] = -eta * grad
                x_list[count] = xi
                count += 1

            # consensus step
            # x, _ = self.karcher_mean(x, x_list)
            x, _ = self.tangent_space_mean(x, x_list)
            # x_true = (x_list[:, 0]+x_list[:, 1])
            # x_true = x_true / np.linalg.norm(x_true)
            # print(np.linalg.norm(x - x_true))
            x_list = np.tile(x, (self.K, 1, 1))

            # print values
            fval_list[iter] = self.f(x)
            angle_list[iter] = sum(subspace_angles(x, self.x_star))
            norm_list[iter] = np.linalg.norm(self.manifold.proj(x, self.grad_f(x)), 'fro')
            time_list[iter] = time.time() - time0 + time_list[iter - 1]
            if flag and iter % 10 == 0:
                print("iter %d, loss value %f, norm of grad %f, average inner loop %d, average norm %f" %
                      (iter, fval_list[iter], norm_list[iter], aveg_inner / self.K, aveg_norm / self.K))

        return x, fval_list, norm_list, time_list, angle_list


    def fedlin(self, outer_iter, inner_iter, mu, eta=1e-4, epsilon=1e-8, flag=0, x0=None):
        fval_list = np.zeros(outer_iter + 1)
        norm_list = np.zeros(outer_iter + 1)
        angle_list = np.zeros(outer_iter + 1)
        time_list = np.zeros(outer_iter + 1)

        if x0 is None:
            x = self.manifold.rand()
        else:
            x = x0
        fval_list[0] = self.f(x)
        angle_list[0] = sum(subspace_angles(x, self.x_star))
        norm_list[0] = np.linalg.norm(self.manifold.proj(x, self.grad_f(x)))
        x_list = np.tile(x, (self.K, 1, 1))
        # g_list = np.zeros((self.K, self.d, self.r))
        for iter in range(1, outer_iter + 1):
            time0 = time.time()
            gt = self.manifold.proj(x, self.grad_f(x))
            # optimization for each client
            batch = random.sample(range(self.n), self.K)
            count = 0
            aveg_inner = 0
            aveg_norm = 0.0
            for i in batch:  # do gradient descent for each client
                xi = x_list[count]
                t = 0
                gti = self.manifold.proj(x, self.grad_fi(x, i))
                for t in range(inner_iter):
                    grad = self.grad_fi(xi, i) + mu * self.grad_dist(xi, x)
                    grad = self.manifold.proj(xi, grad)
                    if np.linalg.norm(grad) <= epsilon:
                        break
                    xi = self.manifold.retr(xi, -eta * (grad + self.manifold.transp(x, xi, -gti + gt)))
                aveg_inner += t + 1
                aveg_norm += np.linalg.norm(grad)
                # g_list[count] = -eta * grad
                x_list[count] = xi
                count += 1

            # consensus step
            # x, _ = self.karcher_mean(x, x_list)
            x, _ = self.tangent_space_mean(x, x_list)
            # x_true = (x_list[:, 0]+x_list[:, 1])
            # x_true = x_true / np.linalg.norm(x_true)
            # print(np.linalg.norm(x - x_true))
            x_list = np.tile(x, (self.K, 1, 1))

            # print values
            fval_list[iter] = self.f(x)
            angle_list[iter] = sum(subspace_angles(x, self.x_star))
            norm_list[iter] = np.linalg.norm(self.manifold.proj(x, self.grad_f(x)))
            time_list[iter] = time.time() - time0 + time_list[iter - 1]
            if flag and iter % 10 == 0:
                print("iter %d, loss value %f, norm of grad %f, average inner loop %d, average norm %f" %
                      (iter, fval_list[iter], norm_list[iter], aveg_inner / self.K, aveg_norm / self.K))

        return x, fval_list, norm_list, time_list, angle_list


if __name__ == "__main__":
    n = 10  # number of clients
    K = 5  # number of clients selected each iter
    d = 20  # dim
    p = 15  # dim

    PCA = problem_PCA(n, K, d, p, r=5)
    # x_list = np.zeros((d, 2))
    # for i in range(2):
    #     x_list[:, i] = PCA.manifold.rand()
    # x_k = PCA.karcher_mean(x_list)
    # x_true = (x_list[:, 0]+x_list[:, 1])
    # x_true = x_true / np.linalg.norm(x_true)
    # print(np.linalg.norm(x_k - x_true))

    print("Start testing on Steifel manifold")
    print("Test on fedavg")
    x0, fval_list0, norm_list0, time_list0, _ = PCA.fedprox(outer_iter=800, inner_iter=1, mu=0.0, eta=1e-3, flag=1)
    print("Test on fedprox")
    x1, fval_list1, norm_list1, time_list1, _ = PCA.fedprox(outer_iter=800, inner_iter=1, mu=20.0, eta=1e-3, flag=1)
    print("Test on fedsvrg")
    x2, fval_list2, norm_list2, time_list2, _ = PCA.fedlin(outer_iter=600, inner_iter=1, mu=0.0, eta=1e-3, flag=1)
    print("Test on fedrlin")
    x3, fval_list3, norm_list3, time_list3, _ = PCA.fedlin(outer_iter=600, inner_iter=1, mu=10.0, eta=1e-3, flag=1)

    plt.plot(time_list0, fval_list0)
    plt.plot(time_list1, fval_list1)
    plt.plot(time_list2, fval_list2, linestyle='--')
    plt.plot(time_list3, fval_list3, linestyle='-.')
    plt.axhline(y=PCA.f_star, linestyle=':')
    plt.xlabel("CPU time")
    plt.ylabel("Function value $f(x_k)$")
    plt.legend(["Fedavg", "Fedprox", "FedSVRG", "FedRegSVRG", "Optimum"])
    # plt.legend(["Fedprox", "FedSVRG", "FedRegSVRG", "Optimum"])
    plt.show()
    # plt.savefig('pca_sphere_time_function_val_' + str(n) + '_' + str(K) + '_' + str(d) + '_' + str(p) + '.pdf')
    plt.clf()

    plt.plot(time_list0, norm_list0)
    plt.plot(time_list1, norm_list1)
    plt.plot(time_list2, norm_list2, linestyle='--')
    plt.plot(time_list3, norm_list3, linestyle='-.')
    plt.yscale("log")
    plt.xlabel("CPU time")
    plt.ylabel("Norm of $\operatorname{grad} f(x_k)$")
    plt.legend(["Fedavg", "Fedprox", "FedSVRG", "FedRegSVRG"])
    # plt.legend(["Fedprox", "FedSVRG", "FedRegSVRG"])
    plt.show()
    # plt.savefig('pca_sphere_time_grad_norm_' + str(n) + '_' + str(K) + '_' + str(d) + '_' + str(p) + '.pdf')