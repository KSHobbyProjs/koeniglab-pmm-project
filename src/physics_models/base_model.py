"""
A module to handle physics models in which the Hamiltonian has N lattice sites
and depends on a varying parameter L.

Prints eigenvalues in the form [[E(L_1)_1, E(L_1)_2, ...], [E(L_2)_1, E(L_2)_2, ...], ...]
if L is given as an array
"""

import abc
import numpy as np
import scipy.sparse as ss

class BaseModel(abc.ABC):
    def __init__(self, N):
        self._N = N

    @abc.abstractmethod
    def _construct_H(self, L):
        pass

    def get_eigenvalues(self, L, k_num=1):
        if np.isscalar(L):
            H = self._construct_H(L)
            eigvals, _ = ss.linalg.eigsh(H, k=k_num, which='SA')
            eigenvalues = np.sort(eigvals)
        elif isinstance(L, np.ndarray):
            eigenvalues = np.zeros((len(L), k_num), dtype=np.float64)
            for i, Li in enumerate(L):
                H = self._construct_H(Li)
                eigvals, _ = ss.linalg.eigsh(H, k=k_num, which='SA')
                eigenvalues[i] = np.sort(eigvals)
                print(f"Finished calculating eigenvalues for {Li}")
        else:
            raise TypeError("L needs to be either a scalar or a numpy array")
        return eigenvalues
