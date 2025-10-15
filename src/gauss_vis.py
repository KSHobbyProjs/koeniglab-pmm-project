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
    ax.plot(phys_coods[idx], states[idx].T)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Potential and Prob")
    ax.set_title(f"States at L: {Ls[idx]}")
    plt.show()
