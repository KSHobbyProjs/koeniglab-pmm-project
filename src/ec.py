"""
ec.py
Defines a class that handles eigenvector continuation predictions given a BaseModel subclass

This module provides a 

Classes
-------
EC
    Class that handles eigenvector continuation computations for BaseModel subclasses
"""

import numpy as np
import scipy.sparse as ss
import scipy.ndimage as sn

class EC: 
    _base_coods = {}
    def __init__(self, model):
        self._model = model

        self._sample_vectors = None
        self._sample_Ls = None
        self._S = None


    def sample(self, sample_Ls, k_num=1):
        """compute and store eigenvectors at sample points."""
        _, eigvecs = self._model.get_eigenvectors(sample_Ls, k_num)
        eigvecs = eigvecs.reshape(-1, eigvecs.shape[2])
        self._sample_vectors = eigvecs
        self._S = eigvecs.conj() @ eigvecs.T
        self._sample_Ls = sample_Ls
        return eigvecs

    def ec_predict(self, target_Ls, k_num=1, dilate=False):
        """predict eigenvalues at target points"""
        if self._sample_vectors is None or self._S is None or self._sample_Ls is None:
            raise RuntimeError("No sampled vectors found. Run `sample()` first.")
        
        Ls = np.atleast_1d(target_Ls)

        eigenvalues = np.zeros((len(Ls), k_num), dtype=np.float64)
        eigenvectors = np.zeros((len(Ls), k_num, self._model.construct_H(Ls[0]).shape[0]), dtype=np.complex128)
        for i, L in enumerate(Ls):
            if dilate:
                reshaped_sample_vectors = self._sample_vectors.reshape(len(self._sample_Ls), -1, self._sample_vectors.shape[1])
                sample_vectors, S = EC.get_dilated_basis(self._sample_Ls, reshaped_sample_vectors, L)
            else:
                sample_vectors, S = self._sample_vectors, self._S
            H = self._model.construct_H(L)
            H_proj = sample_vectors.conj() @ H @ sample_vectors.T
            eigval, eigvec = ss.linalg.eigsh(H_proj, M=S, k=k_num, which='SA')
            sort_indices = np.argsort(eigval)
            eigenvalues[i] = eigval[sort_indices][:k_num]
            eigenvectors[i] = eigvec[:, sort_indices][:, :k_num].T @ sample_vectors # eigenvectors have to be dotted with sample vectors since they're coordinate vectors
        
        if np.isscalar(target_Ls):
            return eigenvalues[0], eigenvectors[0]
        return eigenvalues, eigenvectors


    def solve(self, sample_Ls, target_Ls, k_num_sample=1, k_num_predict=1, dilate=False):
        """ wrapper for sampling and predicting"""
        temp_vecs = self._sample_vectors
        temp_S = self._S

        self.sample(sample_Ls, k_num_sample)
        eigenvalues, eigenvectors = self.ec_predict(target_Ls, k_num_predict, dilate)

        self._sample_vectors = temp_vecs
        self._S = temp_S
        return eigenvalues, eigenvectors

    @classmethod
    def _get_base_coods(cls, N):
        if N in cls._base_coods:
            return cls._base_coods[N]

        xs = (np.arange(N) - N / 2).astype(np.float64)
        xs, ys, zs = np.meshgrid(xs, xs, xs, indexing='ij')
        base_coods = np.stack([xs, ys, zs], axis=0)
        cls._base_coods[N] = base_coods
        return base_coods
   
    @staticmethod
    def get_dilated_basis(sample_Ls, sample_vectors, target_L):
        """predict eigenvalues at target points"""
        Ls_length, k_num, N3 = sample_vectors.shape
        N = round(N3**(1/3))

        if Ls_length != len(sample_Ls):
            raise RuntimeError("the first axis of sample_vectors needs to match the length of sample_Ls in `get_dilated_basis(sample_Ls, sample_vectors, target_L)`")

        dilated_basis = np.empty_like(sample_vectors)
        for i, sample_L in enumerate(sample_Ls):
            dilated_basis[i] = EC.dilate(sample_L, target_L, sample_vectors[i])

        # reflatten
        dilated_basis = dilated_basis.reshape(-1, dilated_basis.shape[2])
        S = dilated_basis.conj() @ dilated_basis.T
        return dilated_basis, S
    
    # dilate a state ket psi into a new volume & renormalize
    # given target volume Lprime, old volume L, and wavefunction psi
    @staticmethod
    def dilate(L, L_target, psi):
        if psi.ndim == 1:
            psi = psi[None, :] # make it (1, N^3)

        k_num, N3 = psi.shape
        N = round(N3**(1/3))
        psi_3d = psi.reshape(k_num, N, N, N)

        # define dilation factor
        s = L / L_target

        # physical coordinates are dilated, not the indices, so create physical cood arrays
        base_coods = EC._get_base_coods(N) 

        # dilate physical coods by L / L'
        dilated_coods = s * base_coods

        # map the physical coods from [-N/2,N/2) back into [0, N) indices
        dilated_coods += N / 2 

        # interpolate psi to psi*
        psi_dilated = np.empty_like(psi)
        for k in range(k_num):
            psi_dilated_3d = s**3/2 * sn.map_coordinates(psi_3d[k], dilated_coods, mode='wrap', order=3)
            psi_dilated[k] = psi_dilated_3d.reshape(-1)

        # reflatten and normalize
        psi_dilated = psi_dilated.reshape(k_num, -1)
        norms = np.linalg.norm(psi_dilated, axis=1, keepdims=True)
        psi_dilated /= norms

        if k_num == 1:
            return psi_dilated[0]
        return psi_dilated
