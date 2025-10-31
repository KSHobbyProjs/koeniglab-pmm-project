#!/usr/bin/env python
import os
import numpy as np
from src import utils
from src import processing

def main(model_name, pmm_name, model_kwargs, pmm_kwargs, k_num_sample, k_num_predict, epochs, store_loss, plot_kwargs, sample_Ls, predict_Ls, try_load, save, show):
    # create directory to store experiment in
    EXPERIMENT_DIR = utils.paths.experiment_subdir(model_name, pmm_name, model_kwargs, pmm_kwargs, k_num_sample, sample_Ls)
    PLOT_DIR = os.path.join(EXPERIMENT_DIR, "plots")

    # grab exact eigenpair data if it exists, otherwise load it. if predict_Ls is None, assume user wants to take predictions at exact Ls.
    print("Grabbing exact eigenpair data.") 
    exact_Ls, exact_energies = processing.process_exact.load_exact_eigenvalues(model_name, predict_Ls, k_num_predict, **model_kwargs)
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
    print("PMM loaded / sampled. Training PMM.")
    losses = processing.process_pmm.train_pmm(pmm_instance, epochs, store_loss)
    if len(losses) == 0: raise RuntimeError("epochs can't be 0 if no PMM is loaded. No PMM trained.")
    print(f"Finished training PMM. Final loss: {losses[-1]}.")

    # predict energies from trained PMM.
    print("Predicting energies now.")
    predict_energies = processing.process_pmm.predict_pmm(pmm_instance, predict_Ls, k_num_predict, energy_norm_bounds)

    # save / don't save pmm
    if save:
        print("Saving PMM state.")
        # if save, create experiment directory and save pmm state
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        processing.process_pmm.save_pmm(EXPERIMENT_DIR, pmm_instance, energy_norm_bounds, sample_Ls, predict_Ls, predict_energies)
        print("Finished saving PMM state.")

    # plot predictions
    print("Plotting eigenvalues, loss, and percent error if possible.")
    processing.process_pmm.make_all_plots(PLOT_DIR, sample_Ls, exact_Ls, predict_Ls, sample_energies, exact_energies, predict_energies, losses, store_loss, 
                                          save=save, show=show, **plot_kwargs)
    print("Finished plotting.\nExperiment complete.")

if __name__=="__main__":
    cfg = utils.io.load_config(utils.paths.CONFIG_PATH)
    #main(**cfg)
     
    for Lmax in [8, 10, 12, 14, 16, 18]:
        for N in [8, 16, 32]:
            print(f"Training PMM for Lmax={Lmax}, N={N}")
            cfg["model_kwargs"]["N"] = N
            cfg["sample_Ls"] = np.linspace(5, Lmax, 50)
            main(**cfg)
