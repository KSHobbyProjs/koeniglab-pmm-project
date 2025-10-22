from .. import utils
from ..algorithms import pmm

def process_pmm_predicted_eigenpairs(
        model_name,
        sample_Ls,
        target_Ls,
        dim,
        epochs,
        k_num_sample=1,
        k_num_predict=1,
        store_loss=100,
        plot_kwargs=None,
        model_kwargs=None,
        **pmm_kwargs
        ):
    raise NotImplementedError

def normalize_data(sample_Ls, sample_energies, predict_Ls):
    lmin, lmax, sample_Ls = utils.math.normalize(sample_Ls)
    emin, emax, sample_energies = utils.math.normalize(sample_energies)
    plmin, plmax, predict_Ls = utils.math.normalize(predict_Ls)
    return (lmin, lmax, emin, emax, plmin, plmax), (sample_Ls, sample_energies, predict_Ls)

def denormalize_data(bounds, sample_Ls, sample_energies, predict_Ls, predict_energies):
    lmin, lmax, emin, emax, plmin, plmax = bounds
    sample_Ls = utils.math.denormalize(lmin, lmax, sample_Ls)
    sample_energies = utils.math.denormalize(emin, emax, sample_energies)
    predict_Ls = utils.math.denormalize(plmin, plmax, predict_Ls)
    predict_energies = utils.math.denormalize(emin, emax, predict_energies)
    return sample_Ls, sample_energies, predict_Ls, predict_energies

def initialize_pmm(**pmm_kwargs):
    pmm_instance = pmm.PMM(**pmm_kwargs)
    return pmm_instance

def run_or_load_pmm(pmm_instance, experiment_dir, predict_Ls, k_num_predict):
    raise NotImplementedError
