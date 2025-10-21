import numpy as np
import jax.numpy as jnp
import jax
from jax import config
config.update("jax_enable_x64", True)

class PMM:
    # -------------------------- Initialization ---------------------------------------------
    def __init__(self, dim, num_primary=2, num_secondary=0,
                 eta=.2e-2, beta1=0.9, beta2=0.999, eps=1e-8, absmaxgrad=1e3,
                 l2=0.0, mag=0.5e-1, seed=0):

        # raise error if user attempts to train less than two primary matrices
        if num_primary < 2: 
            raise ValueError(f"Parametric matrix models require at least two primary matrices, got {num_primary}")
       
        # PMM state
        self._dim = dim
        self._num_primary = num_primary
        self._num_secondary = num_secondary
        
        self._sample_data = {}
        self._losses = []

        # ADAM state
        self._eta = eta
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._absmaxgrad = absmaxgrad
        self._l2 = l2
        
        self._mag = mag   # these two are only recorded for metadata, never used internally past `_init_params()`
        self._seed = seed

        # Initialize learnable Hermitian parameters
        key = jax.random.PRNGKey(seed)
        self._params = self._init_params(key, mag) 
        self._vt = jax.tree.map(jnp.zeros_like, self._params)
        self._mt = jax.tree.map(jnp.zeros_like, self._params)


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
   
    # ------------------------------------------ Sampling ------------------------------------------------------
    def sample_energies(self, Ls, energies):
        Ls = jnp.atleast_1d(Ls)
        energies = jnp.atleast_1d(energies)
        if Ls.shape[0] != energies.shape[0]:
            raise RuntimeError("Sample parameters (`Ls`) and sample eigenvalues (`energies`) need to have the same length in `sample(Ls, energies)`") 
        if energies.ndim == 1:
            energies = energies[:, None]
       
        self._sample_data["Ls"], self._sample_data["energies"] = Ls, energies
        return Ls, energies

    # -------------------------------------------- Training ----------------------------------------------------
    def train_pmm(self, epochs, store_loss=100):
        if not self._sample_data:
            raise RuntimeError("No data loaded. Run `sample_energies()` or `load()` before `train_pmm()`.")

        # construct vt and mt moments (tree.map allows us to move over the whole dictionary at once)
        params = self._params
        vt, mt = self._vt, self._mt
        Ls, energies = self._sample_data["Ls"], self._sample_data["energies"]

        # create array to store loss at epoch t
        losses = np.zeros(epochs // store_loss)

        # jit the loss function so that it's significantly quicker to call
        jit_loss = jax.jit(PMM.loss)
        grad_loss = jax.jit(jax.grad(jit_loss))

        for t in range(epochs):
            # calculate the gradient (automatically applies through leafs (dictionary keys))
            # update the parameters with jax.tree.map (automatically aligns and moves through
            # dictionary keys so the whole dictionary can be moved through at once)
            gt = grad_loss(params, Ls, energies, self._l2)
            update = jax.tree.map(lambda p, v, m, g: PMM._adam_update(p, v, m, t, g, 
                                                                             self._eta, self._beta1, self._beta2,
                                                                             self._eps, self._absmaxgrad),
                                          params, vt, mt, gt
                                          )

            # jax.tree.map returns updates like update["primary_diags"] = (params, vt, mt), so re-split them
            # PyTrees are recursive, so the tuples inside the values will be looped over if we do another tree.map;
            # the is_leaf call prevents jax from applying the function recursively past the tuples. it stops at the values
            # of the dictionary
            params = jax.tree.map(lambda x: x[0], update, is_leaf=lambda x: isinstance(x, tuple))
            vt = jax.tree.map(lambda x: x[1], update, is_leaf=lambda x: isinstance(x, tuple))
            mt = jax.tree.map(lambda x: x[2], update, is_leaf=lambda x: isinstance(x, tuple))

            # store loss
            if t % store_loss == 0:
                losses_at_t = jit_loss(params, Ls, energies, self._l2)
                losses[t // store_loss] = losses_at_t
        
        self._losses.append(losses)
        self._params = params
        self._vt, self._mt = vt, mt
        return params, losses 

    # -------------------------------------------- Prediction -------------------------------------------------
    def predict_energies(self, Ls_predict, k_num=1):
        Ls_predict = jnp.atleast_1d(Ls_predict)
        Ms = PMM._M(self._params, Ls_predict)
        eigvals, _ = PMM._get_eigenvalues(Ms, k_num)
        return eigvals

    # add function here that wraps all pmm mechanics: sampling, training, predicting, saving, and loading
    # keep saving and loading separate in a pipeline code (like if load: PMM.load, etc.)
    def run_pmm(self, sample_Ls, energies, epochs, target_Ls, k_num=1, store_loss=100):
        self.sample_energies(sample_Ls, energies)
        _, losses = self.train_pmm(epochs, store_loss=store_loss)
        eigvals = self.predict_energies(Ls_predict, k_num=k_num)
        return losses, eigvals

    # ------------------------------------------- Saving / Loading State ---------------------------------------
    def get_metadata(self):
        metadata = {
                "dim" : self._dim,
                "num_primary" : self._num_primary,
                "k_num_sample" : self._sample_data["energies"].shape[1],
                "sample_Ls" : f"min-{min(self._sample_data['Ls'])}--max-{max(self._sample_data['Ls'])}--len-{len(self._sample_data['Ls'])}"
                "epochs" : len(self._losses),
                "final_loss" : self._losses[-1],
                "num_secondary" : self._num_secondary
                "eta" : self._eta,
                "beta1" : self._beta1,
                "beta2" : self._beta2,
                "eps" : self._eps,
                "absmaxgrad" : self._absmaxgrad,
                "l2" : self._l2,
                "mag" : self._mag,
                "seed" : self._seed
                }
        return metadata

    def get_state(self):
        state = {
                # training info
                "data" : self._sample_data,
                "losses" : self._losses,
                "params" : self._params,
                "vt" : self._vt,
                "mt" : self._mt,
                # adam info
                "eta" : self._eta,
                "beta1" : self._beta1,
                "beta2" : self._beta2,
                "eps" : self._eps,
                "absmaxgrad" : self._absmaxgrad,
                "l2" : self._l2,
                # model info
                "dim" : self._dim,
                "num_primary" : self._num_primary,
                "num_secondary" : self._num_secondary,
                "mag" = self._mag,
                "seed" = self._seed
                }
        return state

    def set_state(self, state):
        # training info
        self._sample_data = state["data"]
        self._losses = state["losses"]
        self._params = state["params"]
        self._vt = state["vt"]
        self._mt = state["mt"]
        # adam info
        self._eta = state["eta"]
        self._beta1 = state["beta1"]
        self._beta2 = state["beta2"]
        self._eps = state["eps"]
        self._absmaxgrad = state["absmaxgrad"]
        self._l2 = state["l2"]
        # model info
        self._dim = state["dim"]
        self._num_primary = state["num_primary"]
        self._num_secondary = state["num_secondary"]

        self._mag = state["mag"]
        self._seed = state["seed"]

    # ------------------------------------------- Loss and Basis for M ---------------------------------------
    # loss function
    # mean squared error of the predicted eigenvalues to the true eigenvalues
    @staticmethod
    def loss(params, Ls, energies, l2):
        """
        Ls : jndarray of shape (len(Ls),)
        energies : jndarray of shape (len(energies),k_num)
        """
        k_num = energies.shape[1]
        Ms = PMM._M(params, Ls)
        eigvals, _ = PMM._get_eigenvalues(Ms, k_num)
        loss = jnp.mean(jnp.abs(eigvals - energies)**2)

        # use params['secondary_diags'], etc. to add secondary-matrix behavior to loss

        # l2 penalty
        loss += l2 * (jnp.mean(jnp.abs(params["primary_diags"])**2) + 
                      jnp.mean(jnp.abs(params["primary_uppers"])**2))
        return loss

    @staticmethod
    def get_basis(Ls, num_primary):
        powers = jnp.arange(num_primary)
        basis = Ls[None, :] ** powers[:, None]
        return basis
    
    # -------------------------------------------- Utility Methods --------------------------------------------
    @staticmethod
    def _construct_hermitian(diags, uppers):
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
    def _get_eigenvalues(M, k_num):
        """
        Parameters
        ----------
        M : jnparray
            Array of PMM matrices shape (num_primary, n, n).
        k_num : int
            Take the `k_num`th lowest number of eigenvalues of M.
        Returns
        -------
        eigvals : jnparray
            shape (len(M), k_num,)
        eigvecs : jnparray
            shape (len(M), k_num, n)

        """
        # compute eigenpairs
        eigvals, eigvecs = jnp.linalg.eigh(M)

        # sort eigenpairs
        idx = jnp.argsort(eigvals, axis=1)
        eigvals = jnp.take_along_axis(eigvals, idx, axis=1)
        eigvecs = jnp.take_along_axis(eigvecs, idx[:, None, :], axis=2)

        # take the lowest k_num eigenpairs and transpose eigvecs to (len(M), k_num, :)
        eigvals = eigvals[:, :k_num]
        eigvecs = eigvecs[:, :, :k_num].swapaxes(1, 2)

        return eigvals, eigvecs

    @staticmethod
    def _M(params, Ls):
        """
        Parameters
        ----------
        params : dict of jnparray
            Parameters for PMM matrices.
        Ls : jnparray
            List of parameters. Shape (len(Ls),).
        
        Returns
        -------
        M : jnparray
            List of PMM matrices. Shape (num_primary, n, n).
        """
        # grab primary matrix parameters and construct H for each set
        diags, uppers = params["primary_diags"], params["primary_uppers"]
        Hs = PMM._construct_hermitian(diags, uppers)
        
        # construct M via power series (H_0 + g*H_1 + g^2*H_2 + ...) for total number of primary matrices
        basis = PMM.get_basis(Ls, len(Hs))
        M = jnp.einsum('bm,bij->mij', basis, Hs)
        return M
   
       
    # define general Adam-update for complex parameters and real-loss functions
    @staticmethod
    def _adam_update(parameter, vt, mt, t, grad, eta=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, absmaxgrad=1e3):
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
