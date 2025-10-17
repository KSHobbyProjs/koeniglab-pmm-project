import os
import pickle
import matplotlib.pyplot as plt
import physics_models as pm

# get directory where this file lives
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# get path for data/plots
DATA_DIR = os.path.join(MODULE_DIR, "../data")
PLOT_DIR = os.path.join(MODULE_DIR, "../data/plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# get eigenvectors from model
def compute_exact_eigenpairs(model_name, Ls, k_num, **kwargs):
    # get instance of class given by model_name
    submodule_name, class_name = model_name.split(".", 1)
    submodule = getattr(pm, submodule_name)
    ModelClass = getattr(submodule, class_name)
    instance = ModelClass(**kwargs)

    return instance.get_eigenvectors(Ls, k_num)

# store eigenvalues in a pickled data file
def save_eigenpairs(path, Ls, energies, eigenstates):
    with open(path, "wb") as f:
        state_dict = {"Ls" : Ls,
                      "energies" : energies,
                      "eigenstates" : eigenstates
                      }
        pickle.dump(state_dict, f)

# load data from pickle file
def load_eigenpairs(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    Ls = data["Ls"]
    energies = data["energies"]
    eigenstates = data["eigenstates"]
    return Ls, energies, eigenstates

def plot_eigenvalues(path, Ls, energies, k_num=1, show=False, **kwargs):
    fig, ax = plt.subplots()
    
    # detect if Ls and Es are dictionaries and plot each energy level
    if isinstance(Ls, dict) and isinstance(energies, dict):
        user_ls = kwargs.get("linestyle", {key : '-' for key in Ls})
        linestyle = user_ls if isinstance(user_ls, dict) else {key : user_ls for key in Ls}
        for key in Ls:
            for k in range(k_num):
                ax.plot(Ls[key], energies[key][:, k], linestyle[key], label=f"{key} : {k}th excited state")
    else:
        linestyle = kwargs.get("linestyle", '-')
        for k in range(k_num):
            ax.plot(Ls, energies[:, k], linestyle=linestyle, label=f'{k}th excited state')

    xlabel = kwargs.get("xlabel", "Parameter") 
    ylabel = kwargs.get("ylabel", "Energy")
    title = kwargs.get("title", "Energy vs Parameter")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)
    if show: plt.show()
    plt.close(fig)

def process_exact_eigenpairs(model_name, Ls, k_num, plot_kwargs=None, **kwargs):
    # make string to name data and plots
    file_name = "exact_eigenpairs__" + make_file_string(model_name, **kwargs)
    file_path = os.path.join(DATA_DIR, file_name + ".pkl")
    plot_path = os.path.join(PLOT_DIR, file_name + ".png")

    # grab values
    energies, eigenstates = compute_exact_eigenpairs(model_name, Ls, k_num, **kwargs)
    save_eigenpairs(file_path, Ls, energies, eigenstates)
    plot_eigenvalues(plot_path, Ls, energies, k_num, show=False, **(plot_kwargs or {}))

def make_file_string(model_name, **kwargs):
    class_name = model_name.split(".", 1)[1]
    file_name = class_name + "__" + "__".join(f"{k}_{v}" for k, v in kwargs.items())
    return file_name

def load_model_eigenpairs(model_name, prelim, **kwargs):
    """
    A wrapper for load_eigenpairs that handles path construction from the 
    name of the model and a prelim tag like exact_eigenpairs, sample_eigenpairs,
    or predicted_eigenpairs
    """
    
    file_name = prelim + "__" + make_file_string(model_name, **kwargs) + ".pkl"
    path = os.path.join(DATA_DIR, file_name)
    return load_eigenpairs(path)

