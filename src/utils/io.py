import pickle
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

def save_experiment_metadata(path, metadata):
    metadata["data_created"] = dt.datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def load_experiment_metadata(path):
    with open(path, "r") as f:
        return json.load(f)

def save_normalization_metadata(path, lmin, lmax, emin, emax, plmin, plmax):
    norm_metadata = {
            "lmin" : lmin,
            "lmax" : lmax,
            "emin" : emin,
            "emax" : emax,
            "plmin" : plmin,
            "plmax" : plmax
            }
    with open(path, "w") as f:
        json.dump(norm_metadata, f, indent=4)

def load_normalization_metadata(path):
    with open(path, "r") as f:
        norm_metadata = json.load(f)
    lmin, lmax = norm_metadata["lmin"], norm_metadata["lmax"]
    emin, emax = norm_metadata["emin"], norm_metadata["emax"]
    plmin, plmax = norm_metadata["plmin"], norm_metadata["plmax"]
    return (lmin, lmax, emin, emax, plmin, plmax)

def save_state(path, state):
    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_state(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
        return state
