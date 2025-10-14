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

        diags = mag * jax.random.normal(k1, shape=(num_matrices, n), dtype=jnp.float64)
        upper_real = mag * jax.random.normal(k2, shape=(num_matrices, n * (n - 1) // 2), dtype=jnp.float64)
        upper_imag = mag * jax.random.normal(k3, shape=(num_matrices, n * (n - 1) // 2), dtype=jnp.float64)
        upper = upper_real + 1j * upper_imag
        return {"diags" : diags, "uppers" : upper}
                                        
    def _construct_hermitian(self, diags, uppers):
        n = self._dim
        i_off, j_off = jnp.triu_indices(n, k=1)
        # construct diagonal matrices across batch (same as diags[:, :, None] * jnp.eye(n)[None, :, :])
        diag_matrices = jnp.einsum('bi,ij->bij', diags, jnp.eye(n)).astype(jnp.complex128) 
        # construct upper triangular matrices across batch
        upper_matrices = diag_matrices.at[:, i_off, j_off].set(uppers)
        # add them together and force hermiticity
        H = upper_matrices + upper_matrices.conj().swapaxes(1, 2) - diag_matrices
        return H
    
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

    def _M(As, Bs, cs):
        pass

