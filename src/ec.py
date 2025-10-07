import numpy as np
import scipy.sparse as ss

# given sample energies and eigenvectors, project and predict
def ec(eigvecs, target_H, k_num):
    num_eigs = len(vs)
    H_proj = np.zeros((n, n), dtype=np.complex128)
    S = np.zeros((n, n), dtype=np.complex128)
    for i, eigi in enumerate(eigvecs):
        for j, eigj in enumerate(eigvecs):
            H_proj[i, j] = np.vdot(eigi, target_H @ eigj)
            S[i, j] = np.vdot(eigi, eigj)

    eigval_target, eigvec_target = ss.linalg.eigsh(H_proj, M=S, k=k_num, which='SA')
    sort_indices = np.argsort(eigval_target)
    eigenvalues = eigval_target[sort_indices][:,:k_num]
    eigenvectors = eige

