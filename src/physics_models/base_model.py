"""
physics_models/base_model.py

Defines the BaseModel abstract class for parameter-dependent lattice Hamiltonians.

This module provides a reusable interface for physics models in which the Hamiltonian
depends on an external parameter L (for example, a coupling constant, magnetic field,
or system length). The BaseModel class handles eigenvalue and eigenvector computation
using sparse linear algebra methods from SciPy.

Classes
-------
BaseModel
    Abstract base class defining a consistent API for Hamiltonian models.
"""

import abc
import numpy as np
import scipy.sparse as ss

class BaseModel(abc.ABC):
    """
    Abstract base class for lattice Hamiltonian models depending on a parameter L.

    This class defines common methods to compute eigenvalues and eigenvectors
    for Hamiltonians H(L) acting on an N-site lattice. Subclasses must implement
    the method `_construct_H(L)` to return the Hamiltonian matrix for a given L.

    Attributes
    ----------
    _N : int
        Number of lattice sites (system size).

    Methods
    -------
    _construct_H(L)
        Abstract method to build the Hamiltonian for a given L.
    get_eigenvalues(L, k_num=1)
        Compute the lowest k_num eigenvalues of H(L).
    get_eigenvectors(L, k_num=1)
        Compute the lowest k_num eigenvalues and corresponding eigenvectors.
    """
    
    def __init__(self, N=None):
        """ 
        Initialize the base model with a specified number of lattice sites.

        Parameters
        ----------
        N : int, optional
            A model-specific size parameter (e.g., number of lattice sites in 1D).
            Subclasses may use this to help define the Hamiltonian dimension, but
            it is not required.
        """
        self._N = N

    @abc.abstractmethod
    def _construct_H(self, L):
        """
        Construct and return the Hamiltonian matrix for a given parameter L.

        Parameters
        ----------
        L : float
            The value of the external or model parameter (e.g., volume, coupling constant).

        Returns
        -------
        H : scipy.sparse.spmatrix
            The Hamiltonian matrix corresponding to the given parameter L.

        Notes
        -----
        This method must be implemented in every subclass of `BaseModel`.
        """
        pass

    def get_eigenvectors(self, L, k_num=1):
        """
        Compute the lowest k_num eigenvalues and eigenvectors of H(L).

        Parameters
        ----------
        L : float or array_like
            Parameter (or list of parameters) controlling the Hamiltonian
        k_num : int, optional
            Number of lowest eigenpairs to compute. Default is 1.

        Returns
        -------
        eigenvalues : ndarray
            If L is a scalar, returns shape (k_num,)
            If L is an array, returns shape (len(L), k_num),
                where each row corresponds to one parameter value
        eigenvectors : ndarray
            If L is a scalar, returns shape (k_num, n), where each row is an eigenvector
                and n is the dimension of the Hamiltonian
            if L is an array, returns shape (len(L), k_num, n)
 
        Notes
        -----
        Uses `scipy.sparse.linalg.eigsh` to compute the lowest eigenvalues.
        Falls back internally to a dense solver for very small matrices
        Eigenvalues are sorted in ascending order

        """
        Ls = np.atleast_1d(L)
        eigenvalues = np.zeros((len(Ls), k_num), dtype=np.float64)
        eigenvectors = np.zeros((len(Ls), k_num, self._construct_H(Ls[0]).shape[0]), dtype=np.complex128)
        for i, Li in enumerate(Ls):
            H = self._construct_H(Li)
            eigvals, eigvecs = ss.linalg.eigsh(H, k=k_num, which='SA')
            sort_indices = np.argsort(eigvals)
            eigenvalues[i] = eigvals[sort_indices][:k_num]
            eigenvectors[i] = eigvecs[:, sort_indices][:, :k_num].T
            print(f"Finished calculating eigenvalues and eigenvectors for {Li}")
        if np.isscalar(L):
            return eigenvalues[0], eigenvectors[0]
        return eigenvalues, eigenvectors


    def get_eigenvalues(self, L, k_num=1):
        """
        Returns the lowest `k_num` eigenvalues of the Hamiltonian(s) for the given
        parameter(s) L. 

        This is a convenience wrapper around `get_eigenvectors` that discards the eigenvectors
        """
        eigenvalues, _ = self.get_eigenvectors(L, k_num)
        return eigenvalues
