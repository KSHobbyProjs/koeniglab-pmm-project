from .. import utils
from ..algorithms import pmm
import os
from . import process_exact

def initialize_pmm(pmm_name, **pmm_kwargs):
    PMMClass = getattr(pmm, pmm_name)
    pmm_instance = PMMClass(**pmm_kwargs)
    return pmm_instance

def sample_pmm(pmm_instance, sample_Ls, model_name, k_num_sample, **model_kwargs):
    # compute sample energies from sample_Ls
    sample_energies, _ = process_exact.compute_exact_eigenpairs(model_name, sample_Ls, k_num_sample, **model_kwargs)

    # normalize data before training
    _, _, normed_sample_Ls = utils.math.normalize(sample_Ls)
    emin, emax, normed_sample_energies = utils.math.normalize(sample_energies)
    energy_norm_bounds = (emin, emax)
    
    # sample pmm
    pmm_instance.sample_energies(normed_sample_Ls, normed_sample_energies)
    return energy_norm_bounds, sample_energies

def load_pmm(pmm_instance, experiment_dir):
    # load state data and normalization data
    state_path = os.path.join(experiment_dir, "pmm_state.pkl")
    bounds_path = os.path.join(experiment_dir, "normalization_metadata.json")

    # load state and normalization data
    state = utils.io.load_state(state_path)
    energy_norm_bounds = utils.io.load_normalization_metadata(bounds_path)

    # grab sample_Ls and sample_energies
    data = state["data"]
    _, sample_energies = data["Ls"], data["energies"]
    sample_energies = utils.math.denormalize(*energy_norm_bounds, sample_energies)

    # set pmm state
    pmm_instance.set_state(state)
    return energy_norm_bounds, sample_energies

def train_pmm(pmm_instance, epochs, store_loss):
    if epochs > 0:
        pmm_instance.train_pmm(epochs, store_loss)
    losses = pmm_instance.get_state()["losses"]
    return losses

def predict_pmm(pmm_instance, predict_Ls, k_num_predict, energy_norm_bounds):
    # normalize predict_Ls for prediction in PMM
    _, _, predict_Ls = utils.math.normalize(predict_Ls)

    # grab predictions from PMM
    predict_energies = pmm_instance.predict_energies(predict_Ls, k_num_predict)

    # denormalize predictions
    predict_energies = utils.math.denormalize(*energy_norm_bounds, predict_energies)
    return predict_energies

def save_pmm(experiment_dir, pmm_instance, bounds, sample_Ls, predict_Ls, predict_energies):
    sampleLs_hash = utils.misc.create_hash_from_sampleLs(sample_Ls)
    state = pmm_instance.get_state()
    metadata = pmm_instance.get_metadata()
    metadata["sample_Ls"] = f"min-{min(sample_Ls)}--max-{max(sample_Ls)}--len-{len(sample_Ls)}--hash-{sampleLs_hash}",

    # define paths
    state_path = os.path.join(experiment_dir, "pmm_state.pkl")
    metadata_path = os.path.join(experiment_dir, "metadata.json")
    norm_metadata_path = os.path.join(experiment_dir, "normalization_metadata.json")
    energies_path = os.path.join(experiment_dir, "pmm_predicted_eigenpairs.pkl")
    
    utils.io.save_eigenpairs(energies_path, predict_Ls, predict_energies, None)
    utils.io.save_experiment_metadata(metadata_path, metadata)
    utils.io.save_normalization_metadata(norm_metadata_path, *bounds)
    utils.io.save_state(state_path, state)

def make_all_plots(plot_dir, sample_Ls, exact_Ls, predict_Ls, sample_energies, exact_energies, predict_energies, loss, store_loss, save=False, show=False, **plot_kwargs):
    # plot energy comparison
    utils.plot.plot_compare_energies(plot_dir, sample_Ls, exact_Ls, predict_Ls, sample_energies, exact_energies, predict_energies, save=save, show=show, **plot_kwargs)
    # plot loss
    utils.plot.plot_loss(plot_dir, loss, store_loss, show=show, save=save)
    # plot percent error if `predict_energies==exact_energies`
    if exact_energies.shape == predict_energies.shape:
        utils.plot.plot_percent_error(plot_dir, exact_Ls, exact_energies, predict_energies, show=show, save=save)
    else:
        print("[INFO] Not plotting percent error since predict_Ls does not match exact_Ls")
