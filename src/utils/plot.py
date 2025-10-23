import matplotlib.pyplot as plt
import os
import warnings
import numpy as np

def plot_eigenvalues(path, Ls, energies, k_num=0, show=False, save=False, **kwargs):
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
    markerstyle = _get_markerstyle(Ls, kwargs.get("markerstyle", None))
    
    # detect if Ls and Es are dictionaries and plot each energy level
    if isinstance(Ls, dict) and isinstance(energies, dict):
        for key in Ls:
            # because of numpy subtleties, we have to re-arrify this or face consequences
            energy_array = np.array(energies[key], dtype=np.float64)
            num_states = energy_array.shape[1]
            for k in k_indices:
                if k < num_states:
                    ax.plot(Ls[key], energy_array[:, k], linestyle=linestyle[key], marker=markerstyle[key], label=f"{key} : state {k}")
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
    if save: plt.savefig(path)
    if show: plt.show()
    plt.close(fig)

def plot_eigenvalues_separately(directory_name, Ls, energies, k_indices=[0, 1, 2], show=False, save=False, **kwargs):
    if isinstance(k_indices, int):
        warnings.warn("`k_indices` is an int: combined.png will duplicate the single-state figure. Did you mean to use `plot_eigenvalues`?")
    
    # k_indices should be a list, but allow int for backwards compatibility
    k_indices = _get_k_indices(k_indices)

    for k in k_indices:
        fig_path = os.path.join(directory_name, f"state_{k}.png")
        plot_eigenvalues(fig_path, Ls, energies, k_num=k, show=show, save=save, **kwargs)
    
    # plot a combined figure of all k_num states
    fig_path = os.path.join(directory_name, "combined.png")
    plot_eigenvalues(fig_path, Ls, energies, k_num=k_indices, show=show, save=save, **kwargs)

def plot_loss(directory_name, loss, store_loss, show=False, save=False):
    path = os.path.join(directory_name, "loss.png")
    fig, ax = plt.subplots()
    ax.plot(store_loss * np.arange(len(loss)), np.log10(loss), '-')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("log(loss)")
    ax.set_title("Loss vs Epochs")
    plt.tight_layout()
    if save: plt.savefig(path)
    if show: plt.show()
    plt.close(fig)

def plot_percent_error(directory_name, exact_Ls, exact_energies, predict_energies, show=False, save=False):
    path = os.path.join(directory_name, "percent_error.png")
    fig, ax = plt.subplots()
    k_num = predict_energies.shape[1]
    for k in range(k_num):
        percent_error = np.abs((exact_energies[:,k] - predict_energies[:,k]) / exact_energies[:,k]) * 100
        ax.plot(exact_Ls, percent_error, label=f"{k} state")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("% Error")
    ax.set_title("% Error vs Parameter")
    ax.legend()

    plt.tight_layout()
    if save: plt.savefig(path)
    if show: plt.show()
    plt.close(fig)

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

def _get_markerstyle(Ls, user_style):
    if isinstance(Ls, dict):
        if isinstance(user_style, dict):
            return user_style
        return {key : user_style for key in Ls}
    return user_style
# ---------------------------------------------------------------------------------------------------
