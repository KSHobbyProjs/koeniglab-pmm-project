import hashlib
import numpy as np

def make_model_string(model_name, **model_kwargs):
    class_name = model_name.split(".", 1)[1]
    model_string = class_name + "__" + "__".join(f"{k}_{format_param(v)}" for k, v in model_kwargs.items())
    return model_string

def make_pmm_string(file_name_kwargs): 
    key_order = ["pmm_name", "dim", "num_primary", "k_num_sample", "sample_Ls",
                 "num_secondary", "eta", "beta1", "beta2", 
                 "eps", "absmaxgrad", "l2", "mag", "seed"]
    pmm_string = "__".join(f"{k}_{format_param(file_name_kwargs[k])}" for k in key_order)
    return pmm_string

def make_sample_data_string(model_name, sample_Ls, **model_kwargs):
    model_string = make_model_string(model_name, **model_kwargs)
    sample_data_string = model_string + "__sample_Ls_" + create_sample_Ls_string(sample_Ls)
    return sample_data_string

def create_sample_Ls_string(sample_Ls):
    lmin, lmax, llen = np.min(sample_Ls), np.max(sample_Ls), len(sample_Ls)
    hashstring = create_hash_from_sampleLs(sample_Ls)
    sample_Ls_string = f"min-{format_param(lmin)}--max-{format_param(lmax)}--len-{format_param(llen)}--hash-{hashstring}"
    return sample_Ls_string

def create_hash_from_sampleLs(data : np.ndarray, n=6) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()[:n]

def format_param(param):
    if isinstance(param, (np.floating, float)):    
        param = float(param)
        if param != 0 and (abs(param) < 0.01 or abs(param) >= 100):
            return f"{param:.3e}"
        else:
            return f"{param:.3f}"
    elif isinstance(param, (np.integer, int)):
        param = int(param)
        return str(param)
    elif isinstance(param, str):
        return param
    else:
        raise TypeError(f"Unsupported type: {type(value)}")
