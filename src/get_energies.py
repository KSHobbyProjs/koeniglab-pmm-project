#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import physics_models as pm

if __name__=="__main__":
    # get directory where this file lives
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # get path for data/plots
    DATA_DIR = os.path.join(MODULE_DIR, "../data")
    # os.makedirs(PLOT_DIR, exist_ok=True)

    gauss = pm.gaussian.Gaussian(N=32)
    ising = pm.ising.Ising(N=4)
    spins = pm.noninteracting_spins.NoninteractingSpins(N=4)

    # get lowest two eigenvalues for each model over a range of values
    Ls_gauss = np.linspace(5, 20, 50)
    gauss_energies, gauss_states = gauss.get_eigenvectors(Ls_gauss, k_num=2)

    gs_ising = np.linspace(0, 1.0, 50)
    ising_energies, ising_states = ising.get_eigenvectors(gs_ising, k_num=2)

    cs_spins = np.linspace(-2.0, 2.0, 50)
    spins_energies, spins_states = spins.get_eigenvectors(cs_spins, k_num=2)

    # store eigenvalues in a pickled data file
    path = os.path.join(DATA_DIR, "gauss_energies.pkl")
    with open(path, "wb") as f:
        gauss_dict = {"Ls" : Ls_gauss, 
                      "energies" : gauss_energies,
                      "eigenstates": gauss_states
                      }
        pickle.dump(gauss_dict, f)
    
    path = os.path.join(DATA_DIR, "ising_energies.pkl")
    with open(path, "wb") as f:
        ising_dict = {"Ls" : gs_ising, 
                      "energies" : ising_energies,
                      "eigenstates": ising_states
                      }
        pickle.dump(ising_dict, f)

    path = os.path.join(DATA_DIR, "spins_energies.pkl")
    with open(path, "wb") as f:
        spins_dict = {"Ls" : cs_spins,
                      "energies" : spins_energies,
                      "eigenstates": spins_states
                      }
        pickle.dump(spins_dict, f)

    # plot lowest eigenvalue for each model and store in data folder
    path = os.path.join(DATA_DIR, "plots/gaussian_energies.png")
    fig, ax = plt.subplots()
    ax.plot(Ls_gauss, gauss_energies[:,0], '-')
    ax.set_xlabel(r"System Length")
    ax.set_ylabel("Ground State Energy")
    ax.set_title("(Gaussian) Ground State Energy vs System Volume")
    plt.savefig(path)
    plt.show()

    path = os.path.join(DATA_DIR, "plots/ising_energies.png")
    fig, ax = plt.subplots()
    ax.plot(gs_ising, ising_energies[:,0], '-')
    ax.set_xlabel(r"Transverse Field Strength")
    ax.set_ylabel("Ground State Energy")
    ax.set_title("(Ising) Ground State Energy vs Transverse Field Strength")
    plt.savefig(path)
    plt.show()

    path = os.path.join(DATA_DIR, "plots/spins_energies.png")
    fig, ax = plt.subplots()
    ax.plot(cs_spins, spins_energies[:,0], '-')
    ax.set_xlabel(r"x-Spin Strength")
    ax.set_ylabel("Ground State Energy")
    ax.set_title("(Non-interacting Spins) Ground State Energy vs x-Spin Strength")
    plt.savefig(path)
    plt.show()
