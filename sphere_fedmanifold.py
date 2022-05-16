import manifolds
import random
import numpy as np
import time
import matplotlib.pyplot as plt


class problem_PCA(object):
    """
    This is the class for testing the manifold fedprox and fedlin algorithm
    """

    def __init__(self, n, K, d, p=10, A=None):
        self.n = n  # number of clients
        self.K = K  # number of clients selected each iter
        self.d = d  # dim
        self.p = p  # dim
        self.manifold = manifolds.sphere.Sphere(d)
        if A is None:
            # np.random.seed(3432)
            A = np.zeros((n, d, d))
            for i in range(n):
                temp = np.random.normal(size=(d, p))
                A[i] = temp.dot(temp.T)
        self.A = A
        print("Create problem class: pca with d = %d" % (d))

        # below we calculate the true PCA
        fullA = np.zeros((d, d))
        for i in range(n):
            fullA += A[i]
        self.fullA = fullA / self.n
        s, v = np.linalg.eig(self.fullA)
        self.x_star = v[:, 0]
        self.f_star = -s[0] / 2
        print("true function value %f" % (self.f_star))

    def fi(self, x, i):
        return -x.T.dot(self.A[i]).dot(x) / 2

    def f(self, x):
        return sum([self.fi(x, i) for i in range(self.n)]) / self.n

    def grad_fi(self, x, i):
        return -self.A[i].dot(x)

    def grad_f(self, x):
        grad = np.zeros(self.d)
        for i in range(self.n):
            grad += self.grad_fi(x, i)
        return grad / self.n

    def grad_dist(self, x, x0):
        """
        calculate the gradient of the distance square d^2(x, x0)
        where x is the variable
        the distance is calculated by manifold.dist(x, x0)
        """
        # inner_prod = x.T.dot(x0)
        # if inner_prod >= 0.99:  # approximation by the limit
        #     return - 2 * x0
        # return -2 * np.arccos(inner_prod) / np.sqrt(1 - inner_prod**2) * x0
        return self.manifold.log(x, x0)

    def karcher_mean(self, x, x_list, eta=5e-1, max_iter=300, eps=1e-6):
        """
        compute the mean of given x points, using only the Riemannian gradient descent
        :param x_list: d by K, each column is one vector
        """
        num, d = np.shape(x_list)
        time0 = time.time()
        for i in range(max_iter):
            grad = np.zeros(d)
            # loss = 0
            for j in range(num):
                xj = x_list[j]
                # loss += self.manifold.dist(x, xj)
                grad += self.grad_dist(x, xj)
            grad = grad / num
            norm_grad = np.linalg.norm(grad)
            # print("iter %d, grad for karcher mean = %f" % (i, norm_grad))
            if norm_grad <= eps:
                break
            # loss = loss / num
            # print("iter %d, loss value for karcher mean = %f" % (i, loss))
            x = self.manifold.retr(x, -eta * grad)
        return x, time.time() - time0

    def tangent_space_mean(self, x, x_list):
        """
        compute the mean of given x points on the tangent space
        :param x: reference point
        :param x_list: d by K, each column is one vector
        """
        num, d = np.shape(x_list)
        g = np.zeros(d)
        time0 = time.time()
        for j in range(num):
            xj = x_list[j]
            # loss += self.manifold.dist(x, xj)
            g += self.manifold.log(x, xj)
        return self.manifold.retr(x, g / num), time.time() - time0

    def fedprox(self, outer_iter, inner_iter, mu, eta=1e-2, epsilon=1e-8, flag=0):
        fval_list = np.zeros(outer_iter + 1)
        norm_list = np.zeros(outer_iter + 1)
        time_list = np.zeros(outer_iter + 1)
        angle_list = np.zeros(outer_iter + 1)

        x = self.manifold.rand()
        fval_list[0] = self.f(x)
        norm_list[0] = np.linalg.norm(self.manifold.proj(x, self.grad_f(x)))
        angle_list[0] = min(self.manifold.dist(x, self.x_star), self.manifold.dist(x, -self.x_star))
        x_list = np.tile(x, (self.K, 1))
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
                aveg_norm += np.linalg.norm(grad)
                x_list[count] = xi
                count += 1

            # consensus step
            # x, _ = self.karcher_mean(x, x_list)
            x, _ = self.tangent_space_mean(x, x_list)
            # x_true = (x_list[:, 0]+x_list[:, 1])
            # x_true = x_true / np.linalg.norm(x_true)
            # print(np.linalg.norm(x - x_true))
            x_list = np.tile(x, (self.K, 1))

            # print values
            fval_list[iter] = self.f(x)
            norm_list[iter] = np.linalg.norm(self.manifold.proj(x, self.grad_f(x)))
            angle_list[iter] = min(self.manifold.dist(x, self.x_star), self.manifold.dist(x, -self.x_star))
            time_list[iter] = time.time() - time0 + time_list[iter - 1]
            if flag and iter % 10 == 0:
                print("iter %d, loss value %f, norm of grad %f, average inner loop %d, average norm %f" %
                      (iter, fval_list[iter], norm_list[iter], aveg_inner / self.K, aveg_norm / self.K))

        return x, fval_list, norm_list, time_list, angle_list

    def fedlin(self, outer_iter, inner_iter, mu, eta=1e-4, epsilon=1e-8, flag=0):
        fval_list = np.zeros(outer_iter + 1)
        norm_list = np.zeros(outer_iter + 1)
        time_list = np.zeros(outer_iter + 1)
        angle_list = np.zeros(outer_iter + 1)

        x = self.manifold.rand()
        fval_list[0] = self.f(x)
        norm_list[0] = np.linalg.norm(self.manifold.proj(x, self.grad_f(x)))
        angle_list[0] = min(self.manifold.dist(x, self.x_star), self.manifold.dist(x, -self.x_star))
        x_list = np.tile(x, (self.K, 1))
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
                x_list[count] = xi
                count += 1

            # consensus step
            # x, _ = self.karcher_mean(x, x_list)
            x, _ = self.tangent_space_mean(x, x_list)
            # x_true = (x_list[:, 0]+x_list[:, 1])
            # x_true = x_true / np.linalg.norm(x_true)
            # print(np.linalg.norm(x - x_true))
            x_list = np.tile(x, (self.K, 1))

            # print values
            fval_list[iter] = self.f(x)
            norm_list[iter] = np.linalg.norm(self.manifold.proj(x, self.grad_f(x)))
            angle_list[iter] = min(self.manifold.dist(x, self.x_star), self.manifold.dist(x, -self.x_star))
            time_list[iter] = time.time() - time0 + time_list[iter - 1]
            if flag and iter % 10 == 0:
                print("iter %d, loss value %f, norm of grad %f, average inner loop %d, average norm %f" %
                      (iter, fval_list[iter], norm_list[iter], aveg_inner / self.K, aveg_norm / self.K))

        return x, fval_list, norm_list, time_list, angle_list


if __name__ == "__main__":
    n = 50  # number of clients
    K = 10  # number of clients selected each iter
    d = 100  # dim
    p = 20  # dim

    PCA = problem_PCA(n, K, d, p)
    # x_list = np.zeros((d, 2))
    # for i in range(2):
    #     x_list[:, i] = PCA.manifold.rand()
    # x_k = PCA.karcher_mean(x_list)
    # x_true = (x_list[:, 0]+x_list[:, 1])
    # x_true = x_true / np.linalg.norm(x_true)
    # print(np.linalg.norm(x_k - x_true))

    print("Start testing on Sphere")
    print("Test on fedavg")
    x0, fval_list0, norm_list0, time_list0 = PCA.fedprox(outer_iter=800, inner_iter=1, mu=0.0, eta=1e-4, flag=1)
    print("Test on fedprox")
    x1, fval_list1, norm_list1, time_list1 = PCA.fedprox(outer_iter=800, inner_iter=1, mu=20.0, eta=1e-4, flag=1)
    print("Test on fedsvrg")
    x2, fval_list2, norm_list2, time_list2 = PCA.fedlin(outer_iter=600, inner_iter=1, mu=0.0, eta=1e-4, flag=1)
    print("Test on fedrlin")
    x3, fval_list3, norm_list3, time_list3 = PCA.fedlin(outer_iter=600, inner_iter=1, mu=10.0, eta=1e-4, flag=1)

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
