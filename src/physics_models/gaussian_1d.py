import numpy as np
import scipy.sparse as ss
from . import base_model

class Gaussian1d(base_model.BaseModel):
    def __init__(self, N, V0=-4.0, R=2.0):
        super().__init__(N)
        self._R = R
        self._V0 = V0

    def construct_H(self, L):
        a = L / self._N
        indices = np.arange(self._N)

        # construct T
        # make diagonal portion
        diags = np.ones(self._N)
        off_diags = diags[1:]
        T = ss.diags([off_diags, -2 * diags, off_diags], [-1, 0, 1], format='lil')
        # fill corners because of BCs
        T[0, self._N - 1] = 1
        T[self._N - 1, 0] = 1
        # multiply by lattice spacing factor and format as csr
        T = (-1 / a**2 * T).tocsr().astype(np.complex128)

        # construct V
        # sum over nearest images if want V(x+L) = V(x) (V(L/2 + 1) = V(-L/2)), 
        # but this condition is nearly met if R << L since V(x + L) approx V(x) 
        # near the edges. In general, V(x) = sum_n V0 exp(-(x_i - nL)**2 / R**2)
        # is needed to ensure wrapped V(x), but you can see that, if R << L,
        # terms at moderate n are already suppressed
        distances = np.abs(indices - self._N // 2)
        vs = self._V0 * np.exp(-(distances * a)**2 / self._R**2)
        V = ss.diags(vs, format='csr', dtype=np.complex128)

        H = T + V
        return H
