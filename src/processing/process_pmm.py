from .. import utils
from ..algorithms import pmm
import os
from . import process_exact

def initialize_pmm(pmm_name, **pmm_kwargs):
    PMMClass = getattr(pmm, pmm_name)
    pmm_instance = PMMClass(**pmm_kwargs)
    return pmm_instance

def sample_pmm(pmm_instance, sample_Ls, model_name, k_num_sample, **model_kwargs):
    sample_path = os.path.join(utils.paths.SAMPLE_DATA_DIR, "sample_eigenvalues__" + utils.misc.make_sample_data_string(model_name, sample_Ls, **model_kwargs) + ".pkl")
    if os.path.exists(sample_path):
        print("[INFO] Found data from sample_data dir. Loading sample data.")
        _, sample_energies = utils.io.load_eigenvalues(sample_path)
        sample_energies = sample_energies[:,:k_num_sample] # truncate to only the sample energies
    else:
        print("[INFO] No data found at sample_data dir. Computing sample data now.")
        # compute sample energies from sample_Ls
        sample_energies, _ = process_exact.compute_exact_eigenpairs(model_name, sample_Ls, k_num_sample, **model_kwargs)

    # normalize data before training
    lmin, lmax, normed_sample_Ls = utils.math.normalize(sample_Ls)
    emin, emax, normed_sample_energies = utils.math.normalize(sample_energies)
    norm_bounds = ((lmin, lmax), (emin, emax))
    
    # sample pmm
    pmm_instance.sample_energies(normed_sample_Ls, normed_sample_energies)
    return norm_bounds, sample_energies

def load_pmm(pmm_instance, experiment_dir):
    # load state data and normalization data
    state_path = os.path.join(experiment_dir, "pmm_state.pkl")
    bounds_path = os.path.join(experiment_dir, "normalization_metadata.json")

    # load state and normalization data
    state = utils.io.load_state(state_path)
    norm_bounds = utils.io.load_normalization_metadata(bounds_path)

    # grab sample_Ls and sample_energies
    data = state["data"]
    sample_energies = data["energies"]
    sample_energies = utils.math.denormalize(*norm_bounds[1], sample_energies)

    # set pmm state
    pmm_instance.set_state(state)
    return norm_bounds, sample_energies

def train_pmm(pmm_instance, epochs, store_loss):
    if epochs > 0:
        pmm_instance.train_pmm(epochs, store_loss)
    losses = pmm_instance.get_state()["losses"]
    return losses

def predict_pmm(pmm_instance, predict_Ls, k_num_predict, norm_bounds):
    # normalize predict_Ls for prediction in PMM
    lmin, lmax = norm_bounds[0]
    _, _, predict_Ls = utils.math.normalize(predict_Ls, lmin, lmax)

    # grab predictions from PMM
    predict_energies = pmm_instance.predict_energies(predict_Ls, k_num_predict)

    # denormalize predictions
    predict_energies = utils.math.denormalize(*norm_bounds[1], predict_energies)
    return predict_energies

def save_pmm(experiment_dir, pmm_instance, norm_bounds, sample_Ls, predict_Ls, predict_energies):
    state = pmm_instance.get_state()
    metadata = pmm_instance.get_metadata()
    metadata["sample_Ls"] = utils.misc.create_sample_Ls_string(sample_Ls)

    # define paths
    state_path = os.path.join(experiment_dir, "pmm_state.pkl")
    metadata_path = os.path.join(experiment_dir, "metadata.json")
    norm_metadata_path = os.path.join(experiment_dir, "normalization_metadata.json")
    energies_path = os.path.join(experiment_dir, "pmm_predicted_eigenpairs.pkl")
    
    utils.io.save_eigenpairs(energies_path, predict_Ls, predict_energies, None)
    utils.io.save_experiment_metadata(metadata_path, metadata)
    utils.io.save_normalization_metadata(norm_metadata_path, *norm_bounds)
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
