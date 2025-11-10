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
    ec_path = "."
    pmm_path = "."

    with open(pmm_path, "rb") as f:
        pmm_state = 

if __name__=="__main__":
    main()
