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

    def solve_with_dilation(self, sample_Ls, target_Ls, k_num_sample=1, k_num_predict=1):
        """ wrapper for sampling and predicting"""
        temp_vecs = self._sample_vectors
        temp_S = self._S

        self.sample(sample_Ls, k_num_sample)

        Ls = np.atleast_1d(target_Ls)
        eigenvalues = np.zeros((len(Ls), k_num_predict), dtype=np.float64)
        eigenvectors = np.zeros((len(Ls), k_num_predict, self._model.construct_H(Ls[0]).shape[0]), dtype=np.complex128)
        for i, L in enumerate(Ls):
            sample_vs = []
            sample_vs_temp = self._sample_vectors.reshape(len(sample_Ls), k_num_sample, len(self._sample_vectors[0,:]))
            for i, L_sample in enumerate(sample_Ls):
                for k in range(k_num_sample):
                    sample_vs.append(EC.dilate(L_sample, L, sample_vs_temp[i, k]))
            sample_vs = np.array(sample_vs)
            S = sample_vs.conj() @ sample_vs.T
            H = self._model.construct_H(L)
            H_proj = sample_vs.conj() @ H @ sample_vs.T
            eigval, eigvec = ss.linalg.eigsh(H_proj, M=S, k=k_num_predict, which='SA')
            sort_indices = np.argsort(eigval)
            eigenvalues[i] = eigval[sort_indices][:k_num_predict]
            eigenvectors[i] = eigvec[:, sort_indices][:, :k_num_predict].T @ sample_vs # eigenvectors have to be dotted with sample vectors since they're coordinate vectors
        
        eigenvalues, eigenvectors = self.ec_predict(target_Ls, k_num_predict)

        self._sample_vectors = temp_vecs
        self._S = temp_S
        return eigenvalues, eigenvectors

    # dilate a state ket psi into a new volume & renormalize
    # given target volume Lprime, old volume L, and wavefunction psi
    @staticmethod
    def dilate(L, L_target, psi):
        # grab length of psi and shape into 3D
        if psi.ndim == 1:
            N = round(psi.size**(1 / 3))
            psi_3d = psi.reshape((N, N, N))
        else:
            raise RuntimeError("psi in dilatation(L, L_target, psi) wasn't flattened")

        # define dilation factor
        s = L / L_target

        # physical coordinates are dilated, not the indices, so create physical cood arrays
        xs = (np.arange(N) - N / 2).astype(np.float64)
        ys = np.copy(xs)
        zs = np.copy(xs)

        # dilate physical coods by L / L'
        xs *= s
        ys *= s
        zs *= s

        # remap physical coods to indices (no need because mode='wrap' below)
        #xs = xs % N
        #ys = ys % N
        #zs = zs % N

        xs, ys, zs = np.meshgrid(xs, ys, zs, indexing='ij')
        dilated_coods = np.stack([xs, ys, zs], axis=0)

        # interpolate psi to psi*
        psi_dilated = s**3/2 * sn.map_coordinates(psi_3d, dilated_coods, mode='wrap', order=3)

        # reflatten
        psi_dilated = psi_dilated.flatten()

        # normalize
        psi_dilated /= np.sqrt(np.vdot(psi_dilated, psi_dilated))

        return psi_dilated
