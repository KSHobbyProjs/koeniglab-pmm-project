import os
from .. import physics_models as pm
from ..algorithms import ec
from .. import utils
from . import process_exact

def process_ec_predicted_eigenpairs(
        model_name,
        sample_Ls,
        predict_Ls,
        k_num_sample=1,
        k_num_predict=1,
        dilate=False,
        plot_kwargs=None,
        **model_kwargs
        ):

    # try loading existing exact data
    model_string =  utils.misc.make_model_string(model_name, **model_kwargs)
    file_path = os.path.join(utils.paths.EC_DIR, "ec_predicted_eigenpairs" + model_string + ".pkl")
    plot_dir = os.path.join(utils.paths.EC_PLOTS_DIR, model_string)
    os.makedirs(plot_dir, exist_ok=True)

    # load exact energy data or compute energy data at predict_Ls
    exact_Ls, exact_energies, _ = process_exact.load_exact_eigenpairs(model_name, predict_Ls, k_num_predict, **model_kwargs)
    if predict_Ls is None: predict_Ls = exact_Ls

    # create ec instance
    ec_instance = initialize_ec(model_name, **model_kwargs)
    # sample energies
    sample_energies = sample_ec(ec_instance, sample_Ls, k_num_sample)
    # after sampling, predict energies with EC algorithm
    predict_energies, predict_eigenvectors = predict_ec(ec_instance, predict_Ls, k_num_predict, dilate=dilate)

    # save state
    utils.io.save_eigenpairs(file_path, predict_Ls, predict_energies, predict_eigenvectors)

    # plot the comparison between the exact energies and predicted energies
    utils.plot.plot_compare_energies(plot_dir, sample_Ls, exact_Ls, predict_Ls, sample_energies, exact_energies, predict_energies, save=True, show=False, **(plot_kwargs or {}))

def initialize_ec(model_name, **model_kwargs):
    submodule_name, class_name = model_name.split(".", 1)
    submodule = getattr(pm, submodule_name)
    ModelClass = getattr(submodule, class_name)
    model_instance = ModelClass(**model_kwargs)

    ec_instance = ec.EC(model_instance)
    return ec_instance

def sample_ec(ec_instance, sample_Ls, k_num):
    ec_instance.sample(sample_Ls, k_num)
    sample_energies = ec_instance.get_state()["sample_energies"]
    return sample_energies

def predict_ec(ec_instance, predict_Ls, k_num, dilate=False):
    eigenvalues, eigenvectors = ec_instance.ec_predict(predict_Ls, k_num, dilate)
    return eigenvalues, eigenvectors
