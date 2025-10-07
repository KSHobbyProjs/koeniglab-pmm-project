#!/usr/bin/env python
import physics_models as pm
import numpy as np

if __name__=="__main__":
    gauss = pm.noninteracting_spins.NoninteractingSpins(N=32)
    Ls = np.linspace(5, 20, 20)
    print(gauss.get_eigenvectors(Ls, k_num=2))
