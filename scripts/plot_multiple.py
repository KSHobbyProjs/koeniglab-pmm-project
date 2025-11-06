#!/usr/bin/env python3

import os
from src import utils
import numpy as np
import matplotlib.pyplot as plt

def main():
    # --------------------------------------------------------------------------
    #                  CONFIG 
    # --------------------------------------------------------------------------
    model_name = "gaussian.Gaussian1d"
    model_kwargs = {"N" : 128, "V0" : -4.0, "R" : 2.0}
   

    sample_Ls_ec = np.array([6.0, 7.0, 8.0, 9.0])
    k_num_sample_ec = 4
    sample_Ls_pmm = np.linspace(5.0, 9.0, 20)
    k_num_sample_pmm = 1


    pmm_kwargs = {"pmm_name"    : "PMM", 
                  "dim"         : 8,
                  "num_primary" : 2,
                  "num_secondary" : 0,
                  "k_num_sample" : k_num_sample_pmm,
                  "sample_Ls" : utils.misc.create_sample_Ls_string(sample_Ls_pmm),
                  "eta" : 1.0e-2,
                  "beta1" : 0.9,
                  "beta2" : 0.999,
                  "eps" : 1.0e-8,
                  "absmaxgrad" : 1.0e3,
                  "l2" : 0.0,
                  "mag" : 1.0e-1,
                  "seed" : 135
                  }
    
    k_num_predict = 1

    # ----------------------------------------------------------------------------
    model_string = utils.misc.make_model_string(model_name, **model_kwargs)
    exact_str = "exact_eigenpairs__" + model_string + ".pkl"
    ec_str = "ec_predicted_eigenpairs__" + utils.misc.make_ec_data_string(model_name, sample_Ls_ec, k_num_sample_ec, **model_kwargs) + ".pkl"
    pmm_str = utils.misc.make_pmm_string(pmm_kwargs)

    Ls, exact_eigenvalues, _ = utils.io.load_eigenpairs(os.path.join(utils.paths.DATA_DIR, exact_str))
    _, ec_eigenvalues, _ = utils.io.load_eigenpairs(os.path.join(utils.paths.EC_DIR, ec_str))
    _, pmm_eigenvalues, _ = utils.io.load_eigenpairs(os.path.join(utils.paths.RESULTS_DIR, model_string, pmm_str, "pmm_predicted_eigenpairs.pkl"))

    fig, ax = plt.subplots()
    ax.plot(Ls, exact_eigenvalues[:, :k_num_predict], label="exact")
    ax.plot(Ls, ec_eigenvalues[:, :k_num_predict], '--', label="ec")
    ax.plot(Ls, pmm_eigenvalues[:, :k_num_predict], '--', label="pmm")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(Ls, 100 * np.abs(ec_eigenvalues[:, :k_num_predict] - exact_eigenvalues[:, :k_num_predict]) / exact_eigenvalues[:, :k_num_predict], label="ec")
    ax.plot(Ls, 100 * np.abs(pmm_eigenvalues[:, :k_num_predict] - exact_eigenvalues[:, :k_num_predict]) / exact_eigenvalues[:, :k_num_predict], label="pmm")
    plt.show()

if __name__=="__main__":
    main()
