import os
from .. import physics_models as pm
from .. import utils

# get directory where this file lives
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# get path for data/plots
DATA_DIR = os.path.join(MODULE_DIR, "../../data")
PLOT_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def load_exact_eigenpairs(model_name, Ls, k_num, **model_kwargs):
    model_string = utils.make_model_string(model_name, **model_kwargs)
    try:
        file_path = os.path.join(DATA_DIR, "exact_eigenpairs__" + model_string + ".pkl")
        exact_Ls, exact_energies, exact_states = utils.io.load_eigenpairs(file_path)

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
    # make string to name data and plots
    model_string = "exact_eigenpairs__" + utils.misc.make_model_string(model_name, **model_kwargs)
    file_path = os.path.join(DATA_DIR, model_string + ".pkl")
    plot_dir = os.path.join(PLOT_DIR, model_string)

    # grab values
    energies, eigenstates = compute_exact_eigenpairs(model_name, Ls, k_num, **model_kwargs)
    utils.io.save_eigenpairs(file_path, Ls, energies, eigenstates)
    utils.plot.plot_eigenvalues_separately(plot_dir, Ls, energies, k_indices=list(range(k_num)), show=False, **(plot_kwargs or {}))
