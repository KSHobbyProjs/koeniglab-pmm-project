import numpy as np

def normalize(Ls):
    lmin, lmax = np.min(Ls), np.max(Ls)
    Ls = 2 * (Ls - lmin) / (lmax - lmin) - 1
    return lmin, lmax, Ls

def denormalize(lmin, lmax, Ls):
    Ls = (Ls + 1) * (lmax - lmin) / 2 + lmin
    return Ls
    
