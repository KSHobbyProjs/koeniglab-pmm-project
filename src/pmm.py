import jax.numpy as jnp
import jax
from jax import config
config.update("jax_enable_x64", True)

class PMM:
    def __init__(self, model, dim, num_primary=2, num_secondary=0,
                 eta=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, absmaxgrad=1e3,
                 mag=0.5e-1, seed=0):
        
        self._model = model
        self._dim = dim
        self._num_primary = num_primary
        self._num_secondary = num_secondary
    
        self._eta = eta
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._absmaxgrad = absmaxgrad

        # Initialize learnable Hermitian parameters
        key = jax.random.PRNGKey(seed)
        self._params = self._init_params(key, mag)

    def _init_params(self, key, mag):
        num_matrices = self._num_primary + self._num_secondary
        n = self._dim
        k1, k2, k3 = jax.random.split(key, 3)

        # create a batch of diagonal and upper triangular parameters
        diags = mag * jax.random.normal(k1, shape=(num_matrices, n), dtype=jnp.float64)
        upper_real = mag * jax.random.normal(k2, shape=(num_matrices, n * (n - 1) // 2), dtype=jnp.float64)
        upper_imag = mag * jax.random.normal(k3, shape=(num_matrices, n * (n - 1) // 2), dtype=jnp.float64)
        uppers = upper_real + 1j * upper_imag

        # split parameters into primary matrix and secondary matrix parameters 
        split_idx = self._num_primary
        primary_diags, secondary_diags = diags[:split_idx], diags[split_idx:]
        primary_uppers, secondary_uppers = uppers[:split_idx], uppers[split_idx:]
        return {"primary_diags" : primary_diags, "primary_uppers" : primary_uppers,
                "secondary_diags" : secondary_diags, "secondary_uppers" : secondary_uppers}
    
    @staticmethod
    def construct_hermitian(diags, uppers):
        n = diags.shape[1]
        i_off, j_off = jnp.triu_indices(n, k=1)
        # construct diagonal matrices across batch (same as diags[:, :, None] * jnp.eye(n)[None, :, :])
        diag_matrices = jnp.einsum('bi,ij->bij', diags, jnp.eye(n)).astype(jnp.complex128) 
        # construct upper triangular matrices across batch
        upper_matrices = diag_matrices.at[:, i_off, j_off].set(uppers)
        # add them together and force hermiticity
        H = upper_matrices + upper_matrices.conj().swapaxes(1, 2) - diag_matrices
        return H

    # get the k_num-lowest eigenvalues of M (or Ms if M is given as a batch of matrices)
    @staticmethod
    def get_eigenvalues(M, k_num):
        # If M is a single matrix, make it (1, M)
        if M.ndim == 2:
            M = M[None, :, :]
        
        # compute eigenpairs
        eigvals, eigvecs = jnp.linalg.eigh(M)

        # sort eigenpairs
        idx = jnp.argsort(eigvals, axis=1)
        eigvals = jnp.take_along_axis(eigvals, idx, axis=1)
        eigvecs = jnp.take_along_axis(eigvecs, idx[:, None, :], axis=2)

        # take the lowest k_num eigenpairs and transpose eigvecs to (len(M), k_num, :)
        eigvals = eigvals[:, :k_num]
        eigvecs = eigvecs[:, :, :k_num].swapaxes(1, 2)

        if eigvals.shape[0] == 1:
            return eigvals[0], eigvecs[0]
        return eigvals, eigvecs

    def _M(self, gs):
        gs = jnp.atleast_1d(gs)

        # grab primary matrix parameters and construct H for each set
        diags, uppers = self._params["primary_diags"], self._params["primary_uppers"]
        Hs = PMM.construct_hermitian(diags, uppers)
        
        # construct M via power series (H_0 + g*H_1 + g^2*H_2 + ...) for total number of primary matrices
        powers = jnp.arange(len(Hs))
        weights = gs[None, :] ** powers[:, None]
        M = jnp.einsum('bm,bij->mij', weights, Hs)
        return M
   
    def _loss(self):
        pass

    # define general Adam-update for complex parameters and real-loss functions
    @staticmethod
    def adam_update(parameter, vt, mt, t, grad, eta=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, absmaxgrad=1e3):
        # conjugate the gradient and cap it with absmaxgrad
        gt = jnp.clip(grad.real, -absmaxgrad, absmaxgrad) - 1j * jnp.clip(grad.imag, -absmaxgrad, absmaxgrad)
        # compute the moments (momentum and normalizing) step parameters
        vt = beta1 * vt + (1 - beta1) * gt
        mt = beta2 * mt + (1 - beta2) * jnp.abs(gt)**2

        # bias correction
        vt_hat = vt / (1 - beta1 ** (t + 1))
        mt_hat = mt / (1 - beta2 ** (t + 1))

        # step parameter
        parameter = parameter - eta * vt_hat / (jnp.sqrt(mt_hat) + eps)
        return parameter, vt, mt
