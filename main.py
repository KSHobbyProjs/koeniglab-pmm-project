#!/usr/bin/env python
import os
import numpy as np
from src import utils
from src import processing

def main(model_name, pmm_name, model_kwargs, pmm_kwargs, k_num_sample, k_num_predict, epochs, store_loss, plot_kwargs, sample_Ls, predict_Ls, try_load, save):
    # create directory to store experiment in
    EXPERIMENT_DIR = utils.paths.experiment_subdir(model_name, pmm_name, model_kwargs, pmm_kwargs, k_num_sample, sample_Ls)
    PLOT_DIR = os.path.join(EXPERIMENT_DIR, "plots")

    # grab exact eigenpair data if it exists, otherwise load it. if predict_Ls is None, assume user wants to take predictions at exact Ls.
    print("Grabbing exact eigenpair data.")
    exact_Ls, exact_energies, _ = processing.process_exact.load_exact_eigenpairs(model_name, predict_Ls, k_num_predict, **model_kwargs)
    if predict_Ls is None: predict_Ls = exact_Ls

    # initialize pmm instance
    print("Exact eigenpair data grabbed. Loading / sampling PMM.")
    pmm_instance = processing.process_pmm.initialize_pmm(pmm_name, **pmm_kwargs)

    # load or sample pmm
    if try_load and os.path.isdir(EXPERIMENT_DIR):
        print("[INFO] Found PMM state to load. Set `try_load=False` if you don't want to load a PMM state.")
        energy_norm_bounds, sample_energies = processing.process_pmm.load_pmm(pmm_instance, EXPERIMENT_DIR)
    else:
        print("[INFO] No PMM loaded. Sampling new PMM now.")
        energy_norm_bounds, sample_energies = processing.process_pmm.sample_pmm(pmm_instance, sample_Ls, model_name, k_num_sample, **model_kwargs)

    # train loaded / sampled PMM
    if epochs > 0:
        print("PMM loaded / sampled. Training PMM.")
        losses = processing.process_pmm.train_pmm(pmm_instance, epochs, store_loss)
        print(f"Finished training PMM. Final loss: {losses[-1]}.")

    # predict energies from trained PMM.
    print("Predicting energies now.")
    predict_energies = processing.process_pmm.predict_pmm(pmm_instance, predict_Ls, k_num_predict, energy_norm_bounds)

    # save / don't save pmm
    if save:
        print("Saving PMM state.")
        # if save, create experiment directory and save pmm state
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        os.makedirs(PLOT_DIR, exist_ok=True)
        processing.process_pmm.save_pmm(EXPERIMENT_DIR, pmm_instance, energy_norm_bounds, sample_Ls, predict_Ls, predict_energies)
        print("Finished saving PMM state.")

    # plot predictions
    print("Plotting eigenvalues, loss, and percent error if possible.")
    processing.process_pmm.make_all_plots(PLOT_DIR, sample_Ls, exact_Ls, predict_Ls, sample_energies, exact_energies, predict_energies, losses, store_loss, 
                                          plot_kwargs, save=save, show=True)
    print("Finished plotting.\nExperiment complete.")

if __name__=="__main__":
    model_name = "gaussian.Gaussian1d"
    pmm_name = "PMM"
    model_kwargs = {"N" : 128, "V0" : -4, "R" : 2}
    pmm_kwargs = {
            "dim" : 2,
            "num_primary" : 2,
            "num_secondary" : 0,
            "eta" : 1e-2,
            "beta1" : 0.9,
            "beta2" : 0.999,
            "eps" : 1e-8,
            "absmaxgrad" : 1e3,
            "l2" : 0.0,
            "mag" : 1e-1,
            "seed" : 135 #153
            }
    k_num_sample = 1
    k_num_predict = 1
    epochs = 5000
    store_loss = 100
    plot_kwargs = {"xlabel" : "System Length", 
                   "title" : "Gaussian1d (V0=-4, R=2)"}
    
    Lmin, Lmax = 5, 20
    Llen = 20
    sample_Ls = Lmin + np.linspace(0, 1, Llen)**1.5 * (Lmax - Lmin)
    # sample_Ls = np.linspace(Lmin, Lmax, Llen)
    predict_Ls = None
    try_load = True
    save = True

    main(model_name, pmm_name, model_kwargs, pmm_kwargs, k_num_sample, k_num_predict, epochs, store_loss, plot_kwargs, sample_Ls, predict_Ls, try_load, save)
