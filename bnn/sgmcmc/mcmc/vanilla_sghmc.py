import numpy as np


def vanilla_sghmc(noisy_grad_fn, eta, L, x, lambd, seed=None):
    dim = x.shape[0]
    data = np.zeros((L, dim))
    sigma = np.sqrt(2 * eta)

    if seed is not None:
        np.random.seed(seed)

    m = np.zeros(dim)
    for i in range(L):
        noisy_grad = noisy_grad_fn(x)
        dm = noisy_grad * eta - lambd * m * eta + np.random.randn(dim) * sigma
        m = m + dm
        dx = m * eta
        x = x + dx
        data[i] = x
    return data
