def make_model_string(model_name, **kwargs):
    class_name = model_name.split(".", 1)[1]
    model_string = class_name + "__" + "__".join(f"{k}_{v}" for k, v in kwargs.items())
    return model_string

def make_pmm_string(pmm_kwargs, k_num_sample, sample_Ls):
    file_name_kwargs = pmm_kwargs
    file_name_kwargs["k_num_sample"] = k_num_sample
    file_name_kwargs["sample_Ls"] = f"min-{min(sample_Ls)}--max-{max(sample_Ls)}--len-{len(sample_Ls)}"
    
    key_order = ["dim", "num_primary", "k_num_sample", "sample_Ls",
                 "num_secondary", "eta", "beta1", "beta2", 
                 "eps", "absmaxgrad", "l2", "mag", "seed"]
    pmm_string = "__".join(f"{k}_{file_name_kwargs[k]}" for k in key_order)
    return pmm_string


