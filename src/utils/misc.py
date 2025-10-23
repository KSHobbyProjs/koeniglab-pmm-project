import hashlib

def make_model_string(model_name, **model_kwargs):
    class_name = model_name.split(".", 1)[1]
    model_string = class_name + "__" + "__".join(f"{k}_{v}" for k, v in model_kwargs.items())
    return model_string

def make_pmm_string(file_name_kwargs): 
    key_order = ["pmm_name", "dim", "num_primary", "k_num_sample", "sample_Ls",
                 "num_secondary", "eta", "beta1", "beta2", 
                 "eps", "absmaxgrad", "l2", "mag", "seed"]
    pmm_string = "__".join(f"{k}_{file_name_kwargs[k]}" for k in key_order)
    return pmm_string

def create_hash_from_sampleLs(data : np.ndarray, n=6) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()[:n]
