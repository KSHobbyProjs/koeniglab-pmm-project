#!/usr/bin/env python
import numpy as np
from src.processing import process_ec as pec

if __name__ == "__main__":
    # process Gaussian ec predictions

    pec.process_ec_predicted_eigenpairs("gaussian.Gaussian1d", np.arange(5.0, 13.0), predict_Ls=None, k_num_sample=1, k_num_predict=1, dilate=False,
                                        plot_kwargs={"xlabel" : "System Length", "title" : "Attractive Gaussian1D (N=128, V0=-4, R=2)"}, N=128, V0=-4.0, R=2.0)
    pec.process_ec_predicted_eigenpairs("gaussian.Gaussian1d", np.arange(5.0, 13.0), predict_Ls=None, k_num_sample=1, k_num_predict=1, dilate=False,
                                        plot_kwargs={"xlabel" : "System Length", "title" : "Repulsive Gaussian1D (N=128, V0=4, R=2)"}, N=128, V0=4.0, R=2.0)
    pec.process_ec_predicted_eigenpairs("gaussian.Gaussian3d", np.arange(5.0, 13.0), predict_Ls=None, k_num_sample=4, k_num_predict=1, dilate=False,
                                        plot_kwargs={"xlabel" : "System Length", "title" : "Attractive Gaussian3D (N=32, V0=-4, R=2)"}, N=32, V0=-4.0, R=2.0) 
    pec.process_ec_predicted_eigenpairs("gaussian.Gaussian3d", np.arange(5.0, 13.0), predict_Ls=None, k_num_sample=4, k_num_predict=1, dilate=False,
                                        plot_kwargs={"xlabel" : "System Length", "title" : "Repulsive Gaussian3D (N=32, V0=4, R=2)"}, N=32, V0=4.0, R=2.0)
    pec.process_ec_predicted_eigenpairs("ising.Ising", np.linspace(0.0, 0.5, 4), predict_Ls=None, k_num_sample=1, k_num_predict=1, dilate=False,
                                        plot_kwargs={"xlabel" : "Transverse Coupling Strength", "title" : "Ising Model (N=4, J=1)"}, N=4, J=1.0)
    

    # test    
    pec.process_ec_predicted_eigenpairs("gaussian.Gaussian1d", np.array([6.0, 7.0, 8.0, 9.0]), predict_Ls=None, k_num_sample=4, k_num_predict=3, dilate=False,
                                        plot_kwargs={"xlabel" : "System Length", "title" : "Attractive Gaussian1D (N=128, V0=-4, R=2)"}, N=128, V0=-4.0, R=2.0)
    pec.process_ec_predicted_eigenpairs("gaussian.Gaussian3d", np.array([6.0, 7.0, 8.0, 9.0]), predict_Ls=None, k_num_sample=4, k_num_predict=3, dilate=False,
                                        plot_kwargs={"xlabel" : "System Length", "title" : "Attractive Gaussian3D (N=32, V0=-4, R=2)"}, N=32, V0=-4.0, R=2.0)

    # non-interacting spins' parent Hamiltonian is 2x2. Projecting onto a subspace is often dangerous
    # pec.process_ec_predicted_eigenpairs("noninteracting_spins.NoninteractingSpins", np.linspace(-2.0, 0, 3), predict_Ls=None, k_num_sample=1, k_num_predict=1, dilate=False,
    #                                    plot_kwargs={"xlabel" : "x-spin Contribution", "title" : "Non-interacting Spins (N=4)"}, N=4)
