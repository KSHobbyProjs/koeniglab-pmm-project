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

def save_metadata(path, metdata):
    metadata["data_created"] = dt.datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)
