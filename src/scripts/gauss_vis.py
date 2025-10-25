#!/usr/bin/env python
"""
This is a demonstration module that plots the first few states
of a Hamiltonian with a 1D Gaussian potential.

Parameters
----------
N : int, optional
    Number of lattice points. Default is 128.
V0 : float, optional
    Depth of Gaussian potential. Default is -4.
R : float, optional
    Width of Gaussian potential. Default is 2.
plot_potential : bool, optional
    If True, plots the potential well along with the states. Default is False.
plot_probability : bool, optional
    If True, plots the probability amplitude of the state (|psi|^2). 
    If False, plots the raw state. Default is False.
kmin : ints, optional
    Plot the kmin-th energy level. Default is 0 (ground state).
kmax : int, optional
    Plot all energy states in between kmin and kmax. Default is 8.
L : float, optional
    The system volume at which to analyze the system. Default is 10.

Enter all parameters in the shell as [parameter]=[value]; e.g., N=128, V0=-4, krange=(0,8).

Returns
-------
Plots all k energy eigenstates in `krange` (either the raw eigenstates or the probability amplitudes)
along with, optionally, the potential.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from ..processing import process_exact as pe

def main(args):
    # load eigenpairs from gauss1d model (N=128, V0=-4, R=2)
    N, V0, R = 128, -4, 2
    plot_potential = False
    plot_probability = False

    kmin, kmax = 0, 8
    L = 10
    
    for arg in args:
        if arg.startswith('N='):
            N = int(arg[2:])
        elif arg.startswith('V0='):
            V0 = float(arg[3:])
        elif arg.startswith('R='):
            R = float(arg[2:])
        elif arg == "-potential":
            plot_potential = True
        elif arg == "-probability":
            plot_probability = True
        elif arg.startswith('krange='):
            kmin, kmax = arg[7:].split(',')
            kmin = int(kmin)
            kmax = int(kmax)
        elif arg.startswith('kmin='):
            kmin = int(arg[5:])
        elif arg.startswith('kmax='):
            kmax = int(arg[5:])
        elif arg.startswith('L='):
            L = float(arg[2:])
        else:
            sys.stdout.write(f"unknown input: {arg}\n")
            sys.exit()

    energies, eigenstates = pe.compute_exact_eigenpairs("gaussian.Gaussian1d", L, k_num=kmax, N=N, V0=V0, R=R)

    # phys_coods : shape (len(Ls), N)
    # potentials : shape (len(Ls), N)
    # map the indices to the physical coordinates, and grab the potential plot
    lattice_spacing = L / N
    phys_coods = np.arange(N) - N // 2
    phys_coods = phys_coods * lattice_spacing

    potential = V0 * np.exp(-phys_coods**2 / R**2)
   
    # plot eigenstates
    fig, ax = plt.subplots()
    if plot_potential: ax.plot(phys_coods, potential)
    for i, state in enumerate(eigenstates[kmin:kmax]):
        # make sure state is normalized wrt physical coordinates
        norm = np.sqrt(np.sum(np.abs(state)**2) * lattice_spacing)
        state /= norm
        # plot k_min:kmax states
        if plot_probability:
            wavefunc = np.abs(state.T)**2
        else:
            wavefunc = state.T
        ax.plot(phys_coods, wavefunc, label=f'E{i+kmin}: {round(energies[i+kmin], 2)}')

    ax.set_xlabel("Distance")
    ax.set_ylabel("Potential and Prob")
    ax.set_title(f"States at L: {round(L, 2)}")
    ax.legend()
    plt.show()

if __name__=="__main__":
    if sys.argv[-1].endswith("help"):
        print(__doc__)
    else:
        main(sys.argv[1:])
