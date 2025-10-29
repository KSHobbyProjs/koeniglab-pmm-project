import numpy as np

def normalize(Ls, lmin=None, lmax=None):
    if lmin is None:
        lmin = np.min(Ls)
    if lmax is None:
        lmax = np.max(Ls)
    Ls = 2 * (Ls - lmin) / (lmax - lmin) - 1
    return lmin, lmax, Ls

def denormalize(lmin, lmax, Ls):
    Ls = (Ls + 1) * (lmax - lmin) / 2 + lmin
    return Ls
