#!/usr/bin/env python
"""
A module to compare the 'similarity' of two Hermitian matrices.
Two Hermitian matrices are similar iff they share the same eigenvalues.
This module checks to see how close their eigenvalues align.

Parameters
----------
Takes two paths


Returns
-------
The root sum square of the difference between eigenvalues between two matrices.
"""

import os, sys
import numpy as np
import pickle
from src import utils
import matplotlib.pyplot as plt
import numpy as np

def main():
    exact_path = os.path.join(utils.paths.DATA_DIR, "exact_eigenpairs__Gaussian1d__N_128__V0_-4.000__R_2.000.pkl") 
    ec_path = os.path.join(utils.paths.EC_DIR, "ec_predicted_eigenpairs__Gaussian1d__N_128__V0_-4.000__R_2.000__sample_Ls_min-6.000--max-9.000--len-4--hash-9c8e60__k_num_sample_4.pkl")
    pmm_path = os.path.join(utils.paths.RESULTS_DIR, "Gaussian1d__N_128__V0_-4.000__R_2.000/pmm_name_PMM__dim_16__num_primary_2__k_num_sample_4__sample_Ls_min-6.000--max-9.000--len-4--hash-9c8e60__num_secondary_0__eta_0.010__beta1_0.900__beta2_0.999__eps_1.000e-08__absmaxgrad_1.000e+03__l2_0.000__mag_0.100__seed_135/pmm_predicted_eigenpairs.pkl")
    pmm_path = os.path.join(utils.paths.RESULTS_DIR, "Gaussian1d__N_128__V0_-4.000__R_2.000/pmm_name_PMM__dim_16__num_primary_2__k_num_sample_1__sample_Ls_min-6.000--max-9.000--len-4--hash-9c8e60__num_secondary_0__eta_0.010__beta1_0.900__beta2_0.999__eps_1.000e-08__absmaxgrad_1.000e+03__l2_0.000__mag_0.100__seed_135/pmm_predicted_eigenpairs.pkl")
    pmm_path = os.path.join(utils.paths.RESULTS_DIR, "Gaussian1d__N_128__V0_-4.000__R_2.000/pmm_name_PMM__dim_16__num_primary_3__k_num_sample_4__sample_Ls_min-6.000--max-9.000--len-4--hash-9c8e60__num_secondary_0__eta_0.010__beta1_0.900__beta2_0.999__eps_1.000e-08__absmaxgrad_1.000e+03__l2_0.000__mag_0.100__seed_135/pmm_predicted_eigenpairs.pkl")
    pmm_path = os.path.join(utils.paths.RESULTS_DIR, "Gaussian1d__N_128__V0_-4.000__R_2.000/pmm_name_PMM__dim_16__num_primary_2__k_num_sample_1__sample_Ls_min-5.500--max-9.000--len-20--hash-55227c__num_secondary_0__eta_5.000e-03__beta1_0.900__beta2_0.999__eps_1.000e-08__absmaxgrad_1.000e+03__l2_0.000__mag_0.100__seed_135/pmm_predicted_eigenpairs.pkl")
    
    _, exact_eigenvalues = utils.io.load_eigenvalues(exact_path)
    _, ec_eigenvalues = utils.io.load_eigenvalues(ec_path)
    Ls, pmm_eigenvalues = utils.io.load_eigenvalues(pmm_path)
    
    if pmm_eigenvalues.shape != ec_eigenvalues.shape: 
        raise TypeError("The eigenvalues from the PMM algorithm and the EC algorithm need to have the same shape.")

    # should already be sorted, but enforce sorting just in case
    ec_eigenvalues, pmm_eigenvalues = np.sort(ec_eigenvalues, axis=1), np.sort(pmm_eigenvalues, axis=1)
    # measure non-similarity at each L by taking the root mean square of the differences between eigenvalues
    scale = np.mean(np.abs(ec_eigenvalues), axis=1)
    diffs = np.sqrt(np.mean((pmm_eigenvalues - ec_eigenvalues)**2, axis=1))
    rel_diffs = diffs / scale

    fig, ax = plt.subplots()
    ax.plot(Ls, rel_diffs)
    ax.set_xlabel("Ls")
    ax.set_ylabel("Non-similarity (root mean square difference of eigenvalues)")
    ax.set_title("Non similarity of PMM and EC matrix per parameter value")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(Ls, ec_eigenvalues[:, 0], '--', label='ec')
    ax.plot(Ls, pmm_eigenvalues[:, 0], '--', label='pmm')
    ax.plot(Ls, exact_eigenvalues[:, 0], label='exact')
    ax.set_xlabel("Parameter value L")
    ax.set_ylabel("Eigenvalues for EC and PMM")
    ax.set_title("EC vs PMM Ground Energies")
    ax.legend()
    plt.show()

if __name__=="__main__":
    main()
