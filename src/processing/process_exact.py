import os
from .. import physics_models as pm
from .. import utils

def load_exact_eigenpairs(model_name, Ls, k_num, **model_kwargs):
    """
    a wrapper to either load eigenpairs from a file or compute them if no such file exists.

    grabs Ls and eigenpair from a pkl file if it exists; if not, computes eigenpair data at Ls. 
    """
    model_string = utils.misc.make_model_string(model_name, **model_kwargs)
    file_path = os.path.join(utils.paths.DATA_DIR, "exact_eigenpairs__" + model_string + ".pkl") 
    try:
        exact_Ls, exact_energies, exact_states = utils.io.load_eigenpairs(file_path)
        exact_energies = exact_energies[:, :k_num]
        exact_states = exact_states[:, :k_num, :]
    except FileNotFoundError:
        if Ls is None: raise FileNotFoundError("No exact eigenpair data was found. `Ls` can't be None if no exact eigenpair data is preloaded")
        exact_energies, exact_states = compute_exact_eigenpairs(model_name, Ls, k_num, **model_kwargs)
        exact_Ls = Ls
    return exact_Ls, exact_energies, exact_states

# get eigenvectors from model
def compute_exact_eigenpairs(model_name, Ls, k_num, **model_kwargs):
    # get instance of class given by model_name
    submodule_name, class_name = model_name.split(".", 1)
    submodule = getattr(pm, submodule_name)
    ModelClass = getattr(submodule, class_name)
    model_instance = ModelClass(**model_kwargs)

    return model_instance.get_eigenvectors(Ls, k_num)

# process exact eigenpairs from physical models
def process_exact_eigenpairs(model_name, Ls, k_num, plot_kwargs=None, **model_kwargs):
    """
    convenience wrapper for computing, storing, and plotting
    exact eigenpairs from a model
    """
    # make string to name data and plots
    model_string = "exact_eigenpairs__" + utils.misc.make_model_string(model_name, **model_kwargs)
    file_path = os.path.join(utils.paths.DATA_DIR, model_string + ".pkl")
    plot_dir = os.path.join(utils.paths.DATA_PLOTS_DIR, model_string)

    # grab values
    energies, eigenstates = compute_exact_eigenpairs(model_name, Ls, k_num, **model_kwargs)
    utils.io.save_eigenpairs(file_path, Ls, energies, eigenstates)
    utils.plot.plot_eigenvalues_separately(plot_dir, Ls, energies, k_indices=list(range(k_num)), show=False, save=True, **(plot_kwargs or {}))
