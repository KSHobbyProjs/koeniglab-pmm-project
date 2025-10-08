"""
ec.py

Defines a class that handles eigenvector continuation predictions given a BaseModel subclass

This module provides a 
"""

Classes
-------
EC
    Class that handles eigenvector continuation computations for BaseModel subclasses

import numpy as np
import scipy.sparse as ss

class EC:
    def __init__(self, model):
        self._model = model

        self._sample_vectors = None
        self._S = None

    def sample(self, sample_Ls, k_num=1):
        """compute and store eigenvectors at sample points."""
        _, eigvecs = self._model.get_eigenvectors(sample_Ls, k_num)
        eigvecs = eigvecs.reshape(-1, eigvecs.shape[2])
        self._sample_vectors = eigvecs
        self._S = eigvecs.conj() @ eigvecs.T
        return eigvecs

    def ec_predict(self, target_Ls, k_num=1):
        """predict eigenvalues at target points"""
        if self._sample_vectors is None or self._S is None:
            raise RuntimeError("No sampled vectors found. Run `sample()` first.")
        
        Ls = np.atleast_1d(target_Ls)

        eigenvalues = np.zeros((len(Ls), k_num), dtype=np.float64)
        eigenvectors = np.zeros((len(Ls), k_num, self._model.construct_H(Ls[0]).shape[0]), dtype=np.complex128)
        for i, L in enumerate(Ls):
            H = self._model.construct_H(L)
            H_proj = self._sample_vectors.conj() @ H @ self._sample_vectors.T
            eigval, eigvec = ss.linalg.eigsh(H_proj, M=self._S, k=k_num, which='SA')
            sort_indices = np.argsort(eigval)
            eigenvalues[i] = eigval[sort_indices][:k_num]
            eigenvectors[i] = eigvec[:, sort_indices][:, :k_num].T @ self._sample_vectors # eigenvectors have to be dotted with sample vectors since they're coordinate vectors
        
        if np.isscalar(target_Ls):
            return eigenvalues[0], eigenvectors[0]
        return eigenvalues, eigenvectors


    def solve(self, sample_Ls, target_Ls, k_num_sample=1, k_num_predict=1):
        """ wrapper for sampling and predicting"""
        temp_vecs = self._sample_vectors
        temp_S = self._S

        self.sample(sample_Ls, k_num_sample)
        eigenvalues, eigenvectors = self.ec_predict(target_Ls, k_num_predict)

        self._sample_vectors = temp_vecs
        self._S = temp_S
        return eigenvalues, eigenvectors
