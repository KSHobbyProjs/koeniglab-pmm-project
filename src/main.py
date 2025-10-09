#!/usr/bin/env python
import physics_models as pm
import numpy as np
import ec
import matplotlib.pyplot as plt
import pickle

if __name__=="__main__":
    gauss = pm.gaussian.Gaussian(N=32)
    gauss_ec = ec.EC(gauss)

    with open("../data/gauss_energies.pkl", "rb") as f:
        data = pickle.load(f)

    Ls_actual = data["Ls"]
    Es_actual = data["energies"][:,0]
    
    Ls_final = np.arange(7, 15)
    ncols = int(np.ceil(np.sqrt(len(Ls_final))))
    nrows = int(np.ceil(len(Ls_final) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * nrows, 3 * ncols))
    plt.tight_layout(pad=2.0)
    ax = ax.flatten()
    for i, L_final in enumerate(Ls_final):
        cutoff = len(Ls_final) - 1 - i

        Ls_train = 5 + np.linspace(0, 1, 8)**2 * (L_final - 5) 
        Es_train = gauss.get_eigenvalues(Ls_train)
        Es_train = gauss.get_eigenvalues(Ls_train)
        Es_predict, _ = gauss_ec.solve(Ls_train, Ls_actual, k_num_sample=6, dilate=True)

        ax[i].plot(Ls_train, Es_train, 'o', label='training values')
        ax[i].plot(Ls_actual, Es_actual, '-', label='test energies')
        ax[i].plot(Ls_actual, Es_predict, '--', label='prediction')
        ax[i].set_xlabel('System Size L')
        ax[i].set_ylabel('Ground State Energy E')
        ax[i].set_title('Ground State Energy vs System Size')
        ax[i].legend()
    plt.savefig('../results/gaussian')
    plt.show() 
