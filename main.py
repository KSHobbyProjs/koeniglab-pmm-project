#!/usr/bin/env python
import os
import numpy as np
import pickle
from src.algorithms import pmm
from src import utils
from src import processing

def main():
    model_name = "gaussian.Gaussian1d"
    model_kwargs = {"V0" : -4, "R" : 2}
    pmm_kwargs = {
            "dim" : 2,
            "num_primary" : 2,
            "num_secondary" : 0,
            "eta" : .2e-2,
            "beta1" : 0.9,
            "beta2" : 0.999,
            "eps" : 1e-8,
            "absmaxgrad" : 1e3,
            "l2" : 0.0,
            "mag" : 0.5e-1,
            "seed" : 0
            }
    k_num_sample = 1
    k_num_predict = 1
    epochs = 10000
    store_loss = 100
    plot_kwargs = {}

    sample_Ls = 5 + np.linspace(0, 1, 20)**1.5 + (10 - 5)
    predict_Ls = None
    try_load = True
    save = False

    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(MODULE_DIR, "data")
    RESULTS_DIR = os.path.join(MODULE_DIR, "results")
    MODEL_SUBDIR = os.path.join(RESULTS_DIR, utils.misc.make_model_string(model_name, **model_kwargs))
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_SUBDIR, exist_ok=True)
 
    # create file_name_kwargs list that includes pmm_kwargs and sample information and create dir to store experiment in
    pmm_string = utils.misc.make_pmm_string(pmm_kwargs, k_num_sample, sample_Ls)
    EXPERIMENT_DIR = os.path.join(MODEL_SUBDIR, pmm_string)

    # grab exact values if they exist
    model_string = utils.misc.make_model_string(model_name, **model_kwargs)
    file_path = os.join(DATA_DIR, "exact_eigenpairs__" + model_string + ".pkl")
    try:
        exact_Ls, exact_energies, _ = utils.io.load(file_path)
        if predict_Ls is None: predict_Ls = exact_Ls
    except FileNotFoundError:
        if predict_Ls is None: raise FileNotFoundError("No exact eigenpair data was found. predict_Ls can't be None if no exact eigenpair data is preloaded")
        print("[INFO] Exact eigenpair data not found. Computing exact eigenpair data now.")
        exact_energies, _ = processing.process_exact.compute_exact_eigenpairs(model_name, predict_Ls, k_num_predict, **model_kwargs)
        exact_Ls = predict_Ls

    # normalize sample_Ls, predict_Ls, sample_energies
    lmin, lmax, sample_Ls = utils.math.normalize(sample_Ls)
    emin, emax, sample_energies = utils.math.normalize(sample_energies)
    plmin, plmax, predict_Ls = utils.math.normalize(predict_Ls)

    # define pmm instance
    pmm_instance = pmm.PMM(**pmm_kwargs)

    # load, run, predict pmm_state
    if try_load and os.path.isdir(EXPERIMENT_DIR):
        print("[INFO] Found PMM state to load. sample_Ls could be different from what's loaded.\n Set `try_load=False` if don't want to load a pmm state.")
        path = os.path.join(EXPERIMENT_DIR, "pmm_state.pkl")
        state = utils.io.load_state(path)
        pmm_instance.set_state(state) 
        _, losses = pmm_instance.train_pmm(epochs, store_loss)
        predict_energies = pmm_instance.predict_energies(predict_Ls, k_num_predict)
    else:
        os.makedirs(EXPERIMENT_DIRS, exist_ok=True)
        # compute sample energies
        sample_energies, _ = processing.process_exact.compute_exact_eigenpairs(model_name, sample_Ls, k_num_sample, **model_kwargs)
        losses, predict_energies = pmm_instance.run_pmm(sample_Ls, sample_energies, epochs, predict_Ls, k_num_predict, store_loss)

    # save / don't save pmm
    if save:
        state = pmm_instance.get_state()
        metadata = pmm_instance.get_metadata()

        state_path = os.path.join(EXPERIMENT_DIR, "pmm_state.pkl")
        metadata_path = os.path.join(EXPERIMENT_DIR, "metadata.json")
        energies_path = os.path.join(EXPERIMENT_DIR, "pmm_predicted_eigenpairs")
        plots_path = os.path.join(EXPERIMENT_DIR, "plots")
        os.makedirs(plots_path, exist_ok=True)  

        utils.io.save_eigenpairs(energies_path, predict_Ls, predict_energies, None)
        utils.io.save_metadata(metadata_path, metadata)
        utils.io.save_state(state_path, state)

    # denormalize data
    sample_Ls = utils.denormalize(lmin, lmax, sample_Ls)
    sample_energies = utils.denormalize(emin, emax, sample_energies)
    predict_energies = utils.denormalize(emin, emax, predict_energies)
    predict_Ls = utils.denormalize(plmin, plmax, predict_Ls)

    # plot predictions
    Ls = {"sample" : sample_Ls, "exact" : exact_Ls, "prediction" : predict_Ls}
    energies = {"sample" : sample_energies, "exact" : exact_energies, "prediction" : predict_energies}
    linestyle = {"sample" : 'o', "exact" : '--', "predict" : '--'}
    utils.plot.plot_eigenvalues_separately(plot_dir, Ls, energies, k_indices=list(range(k_num_predict)), show=True, save=save, **(plot_kwargs or {}))



if __name__=="__main__":
    
    # load ising model
    """
    N = 4
    ising = pm.ising.Ising(N)
    
    with open("../data/ising_energies.pkl", "rb") as f:
        data = pickle.load(f)

    gs_actual = data["Ls"]
    Es_actual = data["energies"]

    k_num = 3
    gs_sample = np.linspace(0, .5, 10)
    Es_sample = ising.get_eigenvalues(gs_sample, k_num=k_num)
    """    
    
    # ----------------------------------------------- Load Gaussian1d Model ---------------------------
    N = 128
    V0, R = 4.0, 2.0
    gauss = pm.gaussian_1d.Gaussian1d(N, V0=V0, R=R)

    # import ground state energies and eigenvectors
    with open("../data/gauss1d_V0_4_R_2.pkl", "rb") as f:
        data = pickle.load(f)

    gs_actual = data["Ls"]
    Es_actual = data["energies"]
    
    # get sampling data
    k_num = 3
    gs_sample = 5 + np.linspace(0, 1, 20)**1.5 * (10 - 5)
    Es_sample = gauss.get_eigenvalues(gs_sample, k_num=k_num)
    # --------------------------------------------------------------------------------------------------

    # normalize gs_sample, Es_sample to be b/w -1 and 1 (TEST)
    gmin, gmax = np.min(gs_sample), np.max(gs_sample)
    Emin, Emax = np.min(Es_sample), np.max(Es_sample)
    gs_sample = 2 * (gs_sample - gmin) / (gmax - gmin) - 1
    Es_sample = 2 * (Es_sample - Emin) / (Emax - Emin) - 1

    # define pmm
    epochs = 25000
    store_loss = 100
    dim = 8
    pmm_gauss = pmm.PMM(dim=dim,num_primary=3, eta=1e-2, l2=0.0, mag=.5e-1, seed=153)
    # input sampling data
    pmm_gauss.sample(gs=gs_sample, Es=Es_sample)
    # train pmm and grab loss
    params, loss = pmm_gauss.train(epochs=epochs, store_loss=store_loss)
    pmm_gauss.store(path="../data/gauss_state.pkl")
    # predict using trained pmm
    gs_actual = 2 * (gs_actual - gmin) / (gmax - gmin) - 1 # normalize prediction gs to pass to model
    Es_predict = pmm_gauss.predict(gs_actual, k_num=k_num)
    # de-normalize sample and prediction data
    gs_sample = (gs_sample + 1) * (gmax - gmin) / 2 + gmin
    Es_sample = (Es_sample + 1) * (Emax - Emin) / 2 + Emin
    gs_actual = (gs_actual + 1) * (gmax - gmin) / 2 + gmin
    Es_predict = (Es_predict + 1) * (Emax - Emin) / 2 + Emin
    
    # predict function flattens if k_num=1, unflatten for plotting purposes
    if Es_predict.ndim == 1:
        Es_predict = Es_predict[:, None]

    # plot and print loss 
    print(loss[-1])
    fig, ax = plt.subplots()
    ax.plot(store_loss * np.arange(len(loss)), np.log10(loss), '-')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("log(loss)")
    ax.set_title("Loss vs Epochs")
    plt.show()
   
    # plot actual data, sample points, and predictions
    for k in range(k_num):
        fig, ax = plt.subplots()
        ax.plot(gs_actual, Es_actual[:,k], '-', label="Actual")
        ax.plot(gs_sample, Es_sample[:,k], 'o', label="Sample")
        ax.plot(gs_actual, Es_predict[:,k], '--', label="Predictions")
        ax.set_xlabel("System Length")
        ax.set_ylabel("Energy")
        ax.set_title(f"Energy vs System Size: k:{k}")
        ax.legend()
    plt.show()

    # plot percent error
    fig, ax = plt.subplots()
    for k in range(k_num):
        pe = np.abs((Es_actual[:,k] - Es_predict[:,k]) / Es_actual[:,k])
        ax.plot(gs_actual, pe, label=f"{k} state")
    ax.set_xlabel("System Length")
    ax.set_ylabel("% Error")
    ax.set_title("% Error vs System Length")
    ax.legend()
    plt.show()

    """
    # plot ground state eigenvector
    L_index = 10
    L = Ls_actual[L_index]
    a_phys = L / N
    psi = states[L_index]
    psi2 = np.abs(psi)**2 / a_phys**3
    
    ys2d, xs2d = np.indices((N, N))
    xs2d = (xs2d.flatten() - N / 2) * a_phys
    ys2d = (ys2d.flatten() - N / 2) * a_phys
    plt.scatter(xs2d, ys2d, c=psi2[:N**2]*a_phys, cmap='viridis')
    plt.colorbar()
    plt.show()

    # dilate ground state eigenvector and plot
    L_target = 10
    a_phys = L_target / N
    psi_dilated = ec.EC.dilate(L, L_target, psi)
    psi_dilated2 = np.abs(psi_dilated)**2 / a_phys**3

    ys2d, xs2d = np.indices((N, N))
    xs2d = (xs2d.flatten() - N / 2) * a_phys
    ys2d = (ys2d.flatten() - N / 2) * a_phys 
    plt.scatter(xs2d, ys2d, c=psi_dilated2[:N**2]*a_phys, cmap='viridis')
    plt.colorbar()
    plt.show()
    """
    
    """
    # use EC to find predictions for ground state energies
    Ls_train = np.arange(5, 13) 
    Es_train = gauss.get_eigenvalues(Ls_train)
    Es_predict, _ = gauss_ec.solve(Ls_train, Ls_actual, k_num_sample=6, k_num_predict=1, dilate=False)
    
    # find index
    L_star = 20
    temp = np.ones_like(Ls_actual) * L_star
    error = abs(Ls_actual - temp)
    i = np.argmin(error)
    print(Ls_actual[i])
    print(Es_predict[i])

    fig, ax = plt.subplots()
    ax.plot(Ls_train, Es_train, 'o', label='training values')
    ax.plot(Ls_actual, Es_actual, '-', label='test energies')
    ax.plot(Ls_actual, Es_predict, '--', label='prediction')
    ax.set_xlabel('System Size L')
    ax.set_ylabel('Ground State Energy E')
    ax.set_title('Ground State Energy vs System Size')
    ax.legend()
    plt.show() 
    """

