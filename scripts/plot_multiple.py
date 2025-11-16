#!/usr/bin/env python3

import os
from src import utils
import numpy as np
import matplotlib.pyplot as plt

def main():
    # --------------------------------------------------------------------------
    #                 EC vs PMM Comparison       
    # --------------------------------------------------------------------------
   
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
