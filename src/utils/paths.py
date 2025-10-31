import os
from . import misc

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(MODULE_DIR, "../../config.yaml")

DATA_DIR = os.path.join(MODULE_DIR, "../../data/exact_data")
DATA_PLOTS_DIR = os.path.join(DATA_DIR, "plots")

EC_DIR = os.path.join(MODULE_DIR, "../../data/ec_data")
EC_PLOTS_DIR = os.path.join(EC_DIR, "plots")

SAMPLE_DATA_DIR = os.path.join(MODULE_DIR, "../../data/sample_data")

RESULTS_DIR = os.path.join(MODULE_DIR, "../../results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_PLOTS_DIR, exist_ok=True) 
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EC_DIR, exist_ok=True)
os.makedirs(EC_PLOTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

def experiment_subdir(model_name, pmm_name, model_kwargs, pmm_kwargs, k_num_sample, sample_Ls):
    """
    Return the directory in results for the specific model and experiment
    """
    model_string = misc.make_model_string(model_name, **model_kwargs)
    model_subdir = os.path.join(RESULTS_DIR, model_string)
    
    file_name_kwargs = pmm_kwargs.copy()
    file_name_kwargs["k_num_sample"] = k_num_sample
    file_name_kwargs["sample_Ls"] = misc.create_sample_Ls_string(sample_Ls)
    file_name_kwargs["pmm_name"] = pmm_name
    pmm_string = misc.make_pmm_string(file_name_kwargs)

    experiment_subdir = os.path.join(model_subdir, pmm_string)
    return experiment_subdir


