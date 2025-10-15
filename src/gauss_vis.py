#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__=="__main__":
    # load data for gauss_1d model (N = 128)
    with open("../data/gauss1d_energies.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Es : shape (len(Ls), k_num)
    # states : shape (len(Ls), k_num, n)
    N, V0, R = 128, -4.0, 2.0
    Ls = data["Ls"]
    Es = data["energies"]
    states = data["eigenstates"]

    # phys_coods : shape (len(Ls), N)
    # potentials : shape (len(Ls), N)
    lattice_spacings = Ls / N
    phys_coods = np.arange(N) - N // 2
    phys_coods = phys_coods * lattice_spacings[:, None]

    potentials = V0 * np.exp(-phys_coods**2 / R**2)

    idx = 49
    fig, ax = plt.subplots()
    #ax.plot(phys_coods[idx], potentials[idx])
    k_min = 3
    k_max = 6#Es.shape[1]
    for i, state in enumerate(states[idx][k_min:k_max]):
        # make sure state is normalized wrt physical coordinates
        norm = np.sqrt(np.sum(np.abs(state)**2) * lattice_spacings[idx])
        state /= norm
        # plot k_min:k_max states
        ax.plot(phys_coods[idx], state.T, label=f'E{i+k_min}: {round(Es[idx][i+k_min], 2)}')
    ax.set_xlabel("Distance")
    ax.set_ylabel("Potential and Prob")
    ax.set_title(f"States at L: {Ls[idx]}")
    ax.legend()
    plt.show()
