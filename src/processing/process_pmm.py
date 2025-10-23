from .. import utils
from ..algorithms import pmm
import os
from . import process_exact

def normalize_data(sample_Ls, sample_energies, predict_Ls):
    lmin, lmax, sample_Ls = utils.math.normalize(sample_Ls)
    emin, emax, sample_energies = utils.math.normalize(sample_energies)
    plmin, plmax, predict_Ls = utils.math.normalize(predict_Ls)
    return (lmin, lmax, emin, emax, plmin, plmax), sample_Ls, sample_energies, predict_Ls

def denormalize_data(bounds, sample_Ls, sample_energies, predict_Ls, predict_energies):
    lmin, lmax, emin, emax, plmin, plmax = bounds
    sample_Ls = utils.math.denormalize(lmin, lmax, sample_Ls)
    sample_energies = utils.math.denormalize(emin, emax, sample_energies)
    predict_Ls = utils.math.denormalize(plmin, plmax, predict_Ls)
    predict_energies = utils.math.denormalize(emin, emax, predict_energies)
    return sample_Ls, sample_energies, predict_Ls, predict_energies

def initialize_pmm(pmm_name, **pmm_kwargs):
    PMMClass = getattr(pmm, pmm_name)
    pmm_instance = PMMClass(**pmm_kwargs)
    return pmm_instance

def sample_pmm(pmm_instance, sample_Ls, sample_energies):
    pmm_instance.sample_energies(sample_Ls, sample_energies)

def train_pmm(pmm_instance, epochs, store_loss):
    pmm_instance.train_pmm(epochs, store_loss)

def predict_pmm(pmm_instance, predict_Ls, k_num_predict):
    predict_energies = pmm_instance.predict_energies(predict_Ls, k_num_predict)
    return predict_energies

def run_or_load_pmm(experiment_dir, pmm_instance, model_name, model_kwargs, sample_Ls, predict_Ls, k_num_sample, k_num_predict, epochs, store_loss, try_load):
    if try_load and os.path.isdir(experiment_dir):
        print("[INFO] Found PMM state to load. `sample_Ls` might be different from what's loaded. \n set `try_load=False` if you don't want to load a pmm state.")
        bounds, losses, sample_Ls, sample_energies, predict_energies = _load_and_run_pmm(
                pmm_instance, experiment_dir, epochs, store_loss, predict_Ls, k_num_predict)
    else:
        print("[INFO] No PMM loaded. Training new PMM now.")
        bounds, losses, sample_energies, predict_energies = _train_new_pmm(
                model_name, model_kwargs, pmm_instance, sample_Ls, k_num_sample, epochs, predict_Ls, k_num_predict, store_loss)
    return bounds, losses, sample_Ls, sample_energies, predict_energies

def _load_and_run_pmm(pmm_instance, experiment_dir, epochs, store_loss, predict_Ls, k_num_predict):
    # load state data and normalization data
    state_path = os.path.join(experiment_dir, "pmm_state.pkl")
    bounds_path = os.path.join(experiment_dir, "normalization_metadata.json")

    # load state and normalization data
    state = utils.io.load_state(state_path)
    bounds = utils.io.load_normalization_metadata(bounds_path)

    # grab sample_Ls and sample_energies
    data = state["data"]
    sample_Ls, sample_energies = data["Ls"], data["energies"]
    
    # set pmm_instance state to loaded state and run pmm
    pmm_instance.set_state(state)
    if epochs > 0: 
        pmm_instance.train_pmm(epochs, store_loss)

    # with pmm trained, predict energies (normalize predict_Ls first)
    # we change the bounds in case predict_Ls is different from what it was when loading the state
    plmin, plmax, predict_Ls = utils.math.normalize(predict_Ls)
    bounds = (bounds[0], bounds[1], bounds[2], bounds[3], plmin, plmax)
    predict_energies = pmm_instance.predict_energies(predict_Ls, k_num_predict)
    losses = pmm_instance.get_state()["losses"]

    # denormalize data
    sample_Ls, sample_energies, predict_Ls, predict_energies = denormalize_data(
            bounds, sample_Ls, sample_energies, predict_Ls, predict_energies)
    return bounds, losses, sample_Ls, sample_energies, predict_energies

def _train_new_pmm(model_name, model_kwargs, pmm_instance, sample_Ls, k_num_sample, epochs, predict_Ls, k_num_predict, store_loss):
    # compute sample energies from sample_Ls
    sample_energies, _ = process_exact.compute_exact_eigenpairs(model_name, sample_Ls, k_num_sample, **model_kwargs)

    # normalize data before training
    bounds, sample_Ls, sample_energies, predict_Ls = normalize_data(sample_Ls, sample_energies, predict_Ls)

    # run pmm
    losses, predict_energies = pmm_instance.run_pmm(sample_Ls, sample_energies, epochs, predict_Ls, k_num_predict, store_loss)
    # denormalize data
    sample_Ls, sample_energies, predict_Ls, predict_energies = denormalize_data(
            bounds, sample_Ls, sample_energies, predict_Ls, predict_energies)
    return bounds, losses, sample_energies, predict_energies

def save_pmm(experiment_dir, pmm_instance, bounds, sample_Ls, predict_Ls, predict_energies):
    state = pmm_instance.get_state()
    metadata = pmm_instance.get_metadata()
    metadata["sample_Ls"] = f"min-{min(sample_Ls)}--max-{max(sample_Ls)}--len-{len(sample_Ls)}",

    # define paths
    state_path = os.path.join(experiment_dir, "pmm_state.pkl")
    metadata_path = os.path.join(experiment_dir, "metadata.json")
    norm_metadata_path = os.path.join(experiment_dir, "normalization_metadata.json")
    energies_path = os.path.join(experiment_dir, "pmm_predicted_eigenpairs.pkl")
    
    utils.io.save_eigenpairs(energies_path, predict_Ls, predict_energies, None)
    utils.io.save_experiment_metadata(metadata_path, metadata)
    utils.io.save_normalization_metadata(norm_metadata_path, *bounds)
    utils.io.save_state(state_path, state)

def make_all_plots(plot_dir, sample_Ls, exact_Ls, predict_Ls, sample_energies, exact_energies, predict_energies, loss, store_loss, plot_kwargs=None, save=False, show=False):
    k_num_predict = predict_energies.shape[1]
    Ls = {"sample" : sample_Ls, "exact" : exact_Ls, "prediction" : predict_Ls}
    energies = {"sample" : sample_energies, "exact" : exact_energies, "prediction" : predict_energies}
    linestyle = {"sample" : 'None', "exact" : '-', "prediction" : '--'}
    markerstyle= {"sample" : 'o', "exact" : 'None', "prediction" : 'None'}
    plot_kwargs = plot_kwargs.copy() if plot_kwargs else {}
    plot_kwargs["linestyle"] = linestyle
    plot_kwargs["markerstyle"] = markerstyle
   
    if k_num_predict == 1: 
        path = os.path.join(plot_dir, "state_0.png")
        utils.plot.plot_eigenvalues(path, Ls, energies, k_num_predict - 1, show=show, save=save, **plot_kwargs)
    else:
        utils.plot.plot_eigenvalues_separately(plot_dir, Ls, energies, k_indices=list(range(k_num_predict)), show=show, save=save, **plot_kwargs)
    utils.plot.plot_loss(plot_dir, loss, store_loss, show=show, save=save)
    if exact_energies.shape == predict_energies.shape:
        utils.plot.plot_percent_error(plot_dir, exact_Ls, exact_energies, predict_energies, show=show, save=save)
    else:
        print("[INFO] Not plotting percent error since predict_Ls does not match exact_Ls")
