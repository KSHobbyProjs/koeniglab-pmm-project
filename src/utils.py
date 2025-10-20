import os
import pickle
import matplotlib.pyplot as plt
import warnings
import json
import datetime as dt

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

def save_metadata(path, metdata):
    metadata["data_created"] = dt.datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_eigenvalues(path, Ls, energies, k_num=1, show=False, **kwargs):
    """
    Plot energies vs Ls for one or more sets of data. 

    Parameters
    ----------
    path : str
        File path to save figure.
    Ls : array_like or dict of array_like
        Parameter values for each set of data.
    energies : ndarray or dict of ndarray
        Eigenenergies at given parameter values. Shape (len(Ls), num_eigenvalues)
    k_num : int or iterable of int, optional
        If int, plot the k_num-th energy level on one plot.
        If iterable, plot all kth energy levels in the iterable on one plot.
        Default is 1.
    show : bool, optional
        If True, plot is displayed interactively. Default is False.
    **kwargs : dict, optional
        Optional keyword arguments controlling plot appearance: 
        - **linestyle** : str or dict of str, default '-'
            Linestyle for the plotted curves. If a dict, keys must match those of 
            `Ls`/`energies`.
        - **xlabel** : str, default 'Parameter'
            Label for the x-axis.
        - **ylabel** : str, default 'Energies'
            Label for the y-axis.
        - **title** : str, default 'Energy vs Parameter'
            Title of the plot.
    """

    fig, ax = plt.subplots()
   
    # check if k_num is an integer or a list
    k_indices = _get_k_indices(k_num)
    # check if linestyle is a string or a dictionary of strings
    linestyle = _get_linestyle(Ls, kwargs.get("linestyle", '-'))
    
    # detect if Ls and Es are dictionaries and plot each energy level
    if isinstance(Ls, dict) and isinstance(energies, dict):
        for key in Ls:
            for k in k_indices:
                ax.plot(Ls[key], energies[key][:, k], linestyle=linestyle[key], label=f"{key} : state {k}")
    else:
        for k in k_indices:
            ax.plot(Ls, energies[:, k], linestyle=linestyle, label=f'state {k}')

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

def plot_eigenvalues_separately(directory_name, Ls, energies, k_indices=[0, 1, 2], show=False, **kwargs):
    if isinstance(k_indices, int):
        warnings.warn("`k_indices` is an int: combined.png will duplicate the single-state figure. Did you mean to use `plot_eigenvalues`?")
    
    # k_indices should be a list, but allow int for backwards compatibility
    k_indices = _get_k_indices(k_indices)

    os.makedirs(directory_name, exist_ok=True)
    for k in k_indices:
        fig_path = os.path.join(directory_name, f"state_{k}.png")
        plot_eigenvalues(fig_path, Ls, energies, k_num=k, show=show, **kwargs)
    
    # plot a combined figure of all k_num states
    fig_path = os.path.join(directory_name, "combined.png")
    plot_eigenvalues(fig_path, Ls, energies, k_num=k_indices, show=show, **kwargs)

# ---------------------------------------- Plotting Helpers -----------------------------------------
def _get_k_indices(k_num):
    if isinstance(k_num, int):
        return [k_num]
    try:
        return list(k_num)
    except TypeError:
        raise TypeError("k_num must be an int or iterable of ints.")

def _get_linestyle(Ls, user_style):
    if isinstance(Ls, dict):
        if isinstance(user_style, dict):
            return user_style
        return {key : user_style for key in Ls}
    return user_style
# ---------------------------------------------------------------------------------------------------

def make_model_string(model_name, **kwargs):
    class_name = model_name.split(".", 1)[1]
    file_name = class_name + "__" + "__".join(f"{k}_{v}" for k, v in kwargs.items())
    return file_name
