#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    # load eigenpairs from gauss1d model (N=128, V0=-4, R=2)
    N, V0, R = 128, -4, 2
    Ls, energies, eigenstates = utils.load_model_eigenpairs("gaussian.Gaussian1d", "exact_eigenpairs", N=N, V0=V0, R=R)

    # phys_coods : shape (len(Ls), N)
    # potentials : shape (len(Ls), N)
    # map the indices to the physical coordinates, and grab the potential plot
    lattice_spacings = Ls / N
    phys_coods = np.arange(N) - N // 2
    phys_coods = phys_coods * lattice_spacings[:, None]

    potentials = V0 * np.exp(-phys_coods**2 / R**2)

    # pick a volume size and a select range of eigenvectors
    idx = 49
    kmin, kmax = 0, energies.shape[1]

    # plot eigenstates
    fig, ax = plt.subplots()
    #ax.plot(phys_coods[idx], potentials[idx])
    for i, state in enumerate(eigenstates[idx][kmin:kmax]):
        # make sure state is normalized wrt physical coordinates
        norm = np.sqrt(np.sum(np.abs(state)**2) * lattice_spacings[idx])
        state /= norm
        # plot k_min:k_max states
        ax.plot(phys_coods[idx], state.T, label=f'E{i+kmin}: {round(energies[idx][i+kmin], 2)}')
    ax.set_xlabel("Distance")
    ax.set_ylabel("Potential and Prob")
    ax.set_title(f"States at L: {round(Ls[idx], 2)}")
    ax.legend()
    plt.show()
