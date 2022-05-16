import numpy as np
import matplotlib.pyplot as plt
import Stiefel_fedmanifold

rep = 10
d = 200
outer_iter_list = [500, 300, 200, 200]
total_data = 10000
n = 100  # number of clients
p = int(total_data / n)
K = int(n * 0.1)
tau_list = [1, 10, 50, 100]
eta_list0 = [1e-3, 2e-4, 5e-5, 1e-5]
eta_list1 = [1e-3, 2e-4, 5e-5, 1e-5]
eta_list2 = [1e-2, 1e-3, 2e-4, 7e-5]

for i in range(4):
    outer_iter = outer_iter_list[i]
    inner_iter = tau_list[i]

    fval_list0 = np.zeros(outer_iter + 1)
    fval_list1 = np.zeros(outer_iter + 1)
    fval_list2 = np.zeros(outer_iter + 1)

    norm_list0 = np.zeros(outer_iter + 1)
    norm_list1 = np.zeros(outer_iter + 1)
    norm_list2 = np.zeros(outer_iter + 1)

    angle_list0 = np.zeros(outer_iter + 1)
    angle_list1 = np.zeros(outer_iter + 1)
    angle_list2 = np.zeros(outer_iter + 1)

    optimum = 0

    print("Test on n = %d, k = %d, tau = %d" % (n, K, inner_iter))
    for r in range(rep):
        print("Repitition  %d" % (r))
        PCA = Stiefel_fedmanifold.problem_PCA(n, K, d, p=p)
        optimum += PCA.f_star

        print("Test on fedavg")
        x0, fval0, norm0, time0, angle0 = PCA.fedprox(outer_iter=outer_iter, inner_iter=inner_iter,
                                                             mu=0.0, eta=eta_list0[i], flag=0)
        print("Test on fedprox")
        x1, fval1, norm1, time1, angle1 = PCA.fedprox(outer_iter=outer_iter, inner_iter=inner_iter,
                                                             mu=50, eta=eta_list1[i], flag=0)
        print("Test on fedsvrg")
        x2, fval2, norm2, time2, angle2 = PCA.fedlin(outer_iter=outer_iter, inner_iter=inner_iter,
                                                            mu=0.0, eta=eta_list2[i], flag=0)

        fval_list0 += fval0
        fval_list1 += fval1
        fval_list2 += fval2

        norm_list0 += norm0
        norm_list1 += norm1
        norm_list2 += norm2

        angle_list0 += angle0
        angle_list1 += angle1
        angle_list2 += angle2

    fval_list0 /= rep
    fval_list1 /= rep
    fval_list2 /= rep

    norm_list0 /= rep
    norm_list1 /= rep
    norm_list2 /= rep

    angle_list0 /= rep
    angle_list1 /= rep
    angle_list2 /= rep

    optimum /= rep

    plt.plot(fval_list0, linestyle='--')
    plt.plot(fval_list1, linestyle='--')
    plt.plot(fval_list2)
    plt.axhline(y=optimum, linestyle=':')
    plt.xlabel("Round of communication")
    plt.ylabel("Function value $f(x_k)$")
    plt.legend(["RFedavg", "RFedprox", "RFedSVRG", "Optimum"])
    # plt.show()
    plt.savefig('changing_tau/kpca_function_val_tau_' + str(inner_iter) + '.pdf')
    plt.clf()

    plt.plot(norm_list0, linestyle='--')
    plt.plot(norm_list1, linestyle='--')
    plt.plot(norm_list2)
    plt.yscale("log")
    plt.xlabel("Round of communication")
    plt.ylabel("Norm of $\operatorname{grad} f(x_k)$")
    plt.legend(["RFedavg", "RFedprox", "RFedSVRG"])
    # plt.show()
    plt.savefig('changing_tau/kpca_grad_norm_tau_' + str(inner_iter) + '.pdf')
    plt.clf()

    plt.plot(angle_list0, linestyle='--')
    plt.plot(angle_list1, linestyle='--')
    plt.plot(angle_list2)
    plt.yscale("log")
    plt.xlabel("Round of communication")
    plt.ylabel("Angle between $x_t$ and $x^*$")
    plt.legend(["RFedavg", "RFedprox", "RFedSVRG"])
    # plt.show()
    plt.savefig('changing_tau/kpca_angle_tau_' + str(inner_iter) + '.pdf')
    plt.clf()