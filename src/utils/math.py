import numpy as np

def normalize(Ls):
    lmin, lmax = np.min(Ls), np.max(Ls)
    ls = 2 * (ls - lmin) / (lmax - lmin) - 1
    return lmin, lmax, ls

def denormalize(Ls, lmin, lmax):
    Ls = (Ls + 1 ) * (lmax - lmin) / 2 + lmin
    
