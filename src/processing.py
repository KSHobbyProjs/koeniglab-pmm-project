import os
import physics_models as pm
import utils
import ec
import pmm

# get directory where this file lives
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# get path for data/plots
DATA_DIR = os.path.join(MODULE_DIR, "../data")
RESULTS_DIR = os.path.join(MODULE_DIR, "../results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    model_string = "exact_eigenpairs__" + utils.make_model_string(model_name, **model_kwargs)
    file_path = os.path.join(DATA_DIR, model_string + ".pkl")
    plot_dir = os.path.join(DATA_DIR, "results", model_string)

    # grab values
    energies, eigenstates = compute_exact_eigenpairs(model_name, Ls, k_num, **model_kwargs)
    utils.save_eigenpairs(file_path, Ls, energies, eigenstates)
    utils.plot_eigenvalues_separately(plot_dir, Ls, energies, k_indices=list(range(k_num)), show=False, **(plot_kwargs or {}))

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

def process_ec_predicted_eigenpairs(
        model_name,
        sample_Ls,
        target_Ls,
        k_num_sample=1,
        k_num_predict=1,
        dilate=False,
        plot_kwargs=None,
        **model_kwargs
        ):

    # try loading existing exact data
    model_string =  utils.make_model_string(model_name, **model_kwargs)
    try:
        file_path = os.path.join(DATA_DIR, "exact_eigenpairs__" + model_string + ".pkl")
        exact_Ls, exact_energies, _ = load_eigenpairs(file_path)
    except FileNotFoundError:
        # if missing, compute
        print(f"[INFO] Exact eigenpairs not found for {model_name}, computing now... Run `get_exact_data.py` to preload data")
        exact_Ls = target_Ls
        exact_energies, _ = compute_exact_eigenpairs(model_name, exact_Ls, k_num_predict, **model_kwargs)

    # grab model
    submodule_name, class_name = model_name.split(".", 1)
    submodule = getattr(pm, submodule_name)
    ModelClass = getattr(submodule, class_name)
    model_instance = ModelClass(**model_kwargs)
    
    # define ec instance
    ec_instance = ec.EC(model_instance)
    predicted_energies, predicted_states = ec_instance.run_ec(sample_Ls, target_Ls, k_num_sample=k_num_sample, k_num_predict=k_num_predict, dilate=dilate)

    # save predicted energies and states
    ec_dir = os.path.join(RESULTS_DIR, "ec", model_string)
    os.makedirs(ec_dir, exist_ok=True)
    file_path = os.path.join(ec_dir, "ec_predicted_eigenpairs.pkl")
    metadata_path = os.path.join(ec_dir, "ec_metadata.json")
    utils.save(file_path, sample_Ls, predicted_energies, predicted_states)

    # MAKE EC FUNCTION THAT SAVES METADATA STATE AND POSSIBLY REFACTOR THIS FUNCTION
    


    




