class pmm:
    def __init__():
        pass

    def _loss():
        pass

    # define general Adam-update for complex parameters and real-loss functions
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

    def _M(As, Bs, cs):
        pass

