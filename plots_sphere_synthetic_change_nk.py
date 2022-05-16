import numpy as np
import matplotlib.pyplot as plt
import sphere_fedmanifold

rep = 10
d = 100
outer_iter = 600
total_data = 10000
n_list = [500, 1000, 2500]  # number of clients
eta_list0 = [3e-3, 3e-3, 3e-3]
eta_list1 = [3e-3, 3e-3, 3e-3]
eta_list2 = [5e-2, 5e-2, 5e-2]
for i in range(3):
    n = n_list[i]
    p = int(total_data / n)
    K = int(n / 10)

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

    print("Test on n = %d, k = %d" % (n, K))
    for r in range(rep):
        print("Repitition  %d" % (r))
        PCA = sphere_fedmanifold.problem_PCA(n, K, d, p=p)
        optimum += PCA.f_star

        print("Test on fedavg")
        x0, fval0, norm0, time0, angle0 = PCA.fedprox(outer_iter=outer_iter, inner_iter=5,
                                                             mu=0.0, eta=eta_list0[i], flag=0)
        print("Test on fedprox")
        x1, fval1, norm1, time1, angle1 = PCA.fedprox(outer_iter=outer_iter, inner_iter=5,
                                                             mu=float(n/10), eta=eta_list1[i], flag=0)
        print("Test on fedsvrg")
        x2, fval2, norm2, time2, angle2 = PCA.fedlin(outer_iter=outer_iter, inner_iter=1,
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
    plt.savefig('changing_nk/pca_function_val_' + str(n) + '_' + str(K) + '.pdf')
    plt.clf()

    plt.plot(norm_list0, linestyle='--')
    plt.plot(norm_list1, linestyle='--')
    plt.plot(norm_list2)
    plt.yscale("log")
    plt.xlabel("Round of communication")
    plt.ylabel("Norm of $\operatorname{grad} f(x_k)$")
    plt.legend(["RFedavg", "RFedprox", "RFedSVRG"])
    # plt.show()
    plt.savefig('changing_nk/pca_grad_norm_' + str(n) + '_' + str(K) + '.pdf')
    plt.clf()

    plt.plot(angle_list0, linestyle='--')
    plt.plot(angle_list1, linestyle='--')
    plt.plot(angle_list2)
    plt.yscale("log")
    plt.xlabel("Round of communication")
    plt.ylabel("Angle between $x_t$ and $x^*$")
    plt.legend(["RFedavg", "RFedprox", "RFedSVRG"])
    # plt.show()
    plt.savefig('changing_nk/pca_angle_' + str(n) + '_' + str(K) + '.pdf')
    plt.clf()