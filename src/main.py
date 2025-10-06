#!/usr/bin/env python
import physics_models as pm
import numpy as np

if __name__=="__main__":
    gauss = pm.gaussian.Gaussian(N=32)
    Ls = np.linspace(5, 20, 20)
    print(gauss.get_eigenvalues(Ls))
