#!/usr/bin/env python
"""
A module to compare the 'similarity' of two Hermitian matrices.
Two Hermitian matrices are similar iff they share the same eigenvalues.
This module checks to see how close their eigenvalues align.

Parameters
----------
Takes two paths


Returns
-------
The root sum square of the difference between eigenvalues between two matrices.
"""

import os, sys
import numpy as np
import pickle
from src import utils

def main():
    ec_path = os.path.join(utils.paths.EC_DIR, "ec_predicted_eigenpairs__Gaussian1d__N_128__V0_-4.000__R_2.000__sample_Ls_min-6.000--max-9.000--len-4--hash-9c8e60__k_num_sample_4.pkl")
    pmm_path = "."
    
    _, ec_eigenvalues = utils.io.load_eigenvalues(ec_path)
    _, pmm_eigenvalues = utils.io.load_eigenvalues(pmm_path)

if __name__=="__main__":
    main()
