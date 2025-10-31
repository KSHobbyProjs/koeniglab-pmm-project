import pickle
import json
import datetime as dt
import yaml

import numpy as np

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

def load_eigenvalues(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    Ls = data["Ls"]
    energies = data["energies"]
    return Ls, energies

def save_experiment_metadata(path, metadata):
    metadata["data_created"] = dt.datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def load_experiment_metadata(path):
    with open(path, "r") as f:
        return json.load(f)

def save_normalization_metadata(path, Ls_norm_bounds, energy_norm_bounds):
    lmin, lmax = Ls_norm_bounds
    emin, emax = energy_norm_bounds
    norm_metadata = {
            "lmin" : lmin,
            "lmax" : lmax,
            "emin" : emin,
            "emax" : emax,
            }
    with open(path, "w") as f:
        json.dump(norm_metadata, f, indent=4)

def load_normalization_metadata(path):
    with open(path, "r") as f:
        norm_metadata = json.load(f)
    emin, emax = norm_metadata["emin"], norm_metadata["emax"]
    lmin, lmax = norm_metadata["lmin"], norm_metadata["lmax"]
    return ((lmin, lmax), (emin, emax))

def save_state(path, state):
    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_state(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
        return state

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    lmin, lmax = cfg["sample_Ls"]["Lmin"], cfg["sample_Ls"]["Lmax"]
    llen, lexp = cfg["sample_Ls"]["Llen"], cfg["sample_Ls"]["Lexp"]
    if lexp == 1.0: 
        cfg["sample_Ls"] = np.linspace(lmin, lmax, llen)
    else: 
        cfg["sample_Ls"] = lmin + np.linspace(0, 1, llen) ** lexp * (lmax - lmin)
    return cfg

