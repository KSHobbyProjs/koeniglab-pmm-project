#!/usr/bin/env python
import numpy as np
import processing

if __name__=="__main__":
        
    # process Gaussian eigenpairs
    Ls = np.linspace(5, 20, 100)
    processing.process_exact_eigenpairs("gaussian.Gaussian3d", Ls=Ls, k_num=9, plot_kwargs={"xlabel" : "System Length",
                                                                     "title" : "Attractive 3D Gaussian (N=32, V0=-4, R=2)"}, N=32, V0=-4, R=2)
    processing.process_exact_eigenpairs("gaussian.Gaussian3d", Ls=Ls, k_num=10, plot_kwargs={"xlabel" : "System Length",
                                                                     "title" : "Repulsive 3D Gaussian (N=32, V0=4, R=2)"}, N=32, V0=4, R=2)
    processing.process_exact_eigenpairs("gaussian.Gaussian1d", Ls=Ls, k_num=10, plot_kwargs={"xlabel" : "System Length",
                                                                     "title" : "Attractive 1D Gaussian (N=128, V0=-4, R=2)"}, N=128, V0=-4, R=2)
    processing.process_exact_eigenpairs("gaussian.Gaussian1d", Ls=Ls, k_num=10, plot_kwargs={"xlabel" : "System Length",
                                                                     "title" : "Repulsive 1D Gaussian (N=128, V0=4, R=2)"}, N=128, V0=4, R=2)

    # process Ising eigenpairs
    Ls = np.linspace(0, 1.0, 50)
    processing.process_exact_eigenpairs("ising.Ising", Ls=Ls, k_num=6, plot_kwargs={"xlabel" : "Transverse Coupling Strength",
                                                             "title" : "Ising Model (N=4, J=1)"}, N=4, J=1)
    
    # process non-interacting spins eigenpairs
    Ls = np.linspace(-2.0, 2.0, 50)
    processing.process_exact_eigenpairs("noninteracting_spins.NoninteractingSpins", Ls=Ls, k_num=2, plot_kwargs={"xlabel" : "x-spin contribution",
                                                                                                "title" : "Non-interacting Spins (N=4)"}, N=4)
