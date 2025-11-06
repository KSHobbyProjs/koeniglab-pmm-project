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

def make_ec_data_string(model_name, sample_Ls, k_num_sample, **model_kwargs):
    ec_data_string = make_sample_data_string(model_name, sample_Ls, **model_kwargs) + f"__k_num_sample_{k_num_sample}"
    return ec_data_string

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

def parse_kwargs(s):
    """
    Parses a string like N=32,V0=-4.0,R=2.0 into a dict.
    """
    kwargs = {}
    for kv in s.split(","):
        if not kv.strip():
            continue
        if "=" not in kv:
            raise RuntimeError(f"Invalid argument input: '{kv}'. Kwarg arguments need to be input in the form `key1=val1,key2=val2`")
        k, v = kv.split("=", 1)

        # attempt numeric conversion
        try: 
            if "." in v:
                v = float(v)
            else:
                v = int(v)
        except ValueError:
            pass # leave as string if not numeric
        kwargs[k.strip()] = v
    return kwargs

def parse_Ls(s):
    """
    Parses a flexible CLI argument for sample/predict_Ls.

    Examples:
        '1.5'        -> np.array([1.5])
        '1.0,2.0,3.0'    -> np.array([1.0, 2.0])
        '5,20:50'    -> np.linspace(5, 20, 50)
        '5,20:50;1.5 -> 5 + np.linspace(0, 1, 50)**1.5 * (20 - 5)
        'none'       -> None
    """
    s = s.strip().lower()
    if s == "none":
        return None

    # If colon syntax (linspace)
    if ":" in s:
        try:
            lmin_lmax, llen_lexp = s.split(":")
            lmin, lmax = lmin_lmax.split(",")
            if "," in llen_lexp:
                llen, lexp = llen_lexp.split(",")
                lexp = float(lexp)
            else:
                llen = llen_lexp
                lexp = 1.0
            lmin, lmax, llen = float(lmin), float(lmax), int(llen)

            if lexp == 1.0:
                return np.linspace(lmin, lmax, llen)
            else:
                return lmin + np.linspace(0.0, 1.0, llen)**lexp * (lmax - lmin)
        except Exception as e:
            raise ValueError(f"Invalid linspace format: {s}. Use 'lmin,lmax:llen' or 'lmin,lmax:llen,lexp'") from e
    
    # otherwise, assume comma-separated list of numbers
    if "," in s:
        return np.array([float(x) for x in s.split(",")])

    # otherwise, assume a single float
    return np.array([float(s)])
