#!/usr/bin/env python
import physics_models as pm
import numpy as np
import ec
import matplotlib.pyplot as plt

if __name__=="__main__":
    gauss = pm.gaussian.Gaussian(N=32)
    gauss_ec = ec.EC(gauss)

    Ls_test = np.linspace(5, 20, 50)
    Ls_train = np.arange(5, 13)
    Ls_train = np.linspace(5, 13, 20)
    Es_test = gauss.get_eigenvalues(Ls_test)
    Es_train = gauss.get_eigenvalues(Ls_train)
    Es_predict, _ = gauss_ec.solve(Ls_train, Ls_test, k_num_sample=6)

    fig, ax = plt.subplots()
    ax.plot(Ls_train, Es_train, 'o', label='training values')
    ax.plot(Ls_test, Es_test, '-', label='test energies')
    ax.plot(Ls_test, Es_predict, '--', label='prediction')
    ax.set_xlabel('L')
    ax.set_ylabel('E')
    ax.set_title('E vs L')
    ax.legend()
    plt.show()


     
