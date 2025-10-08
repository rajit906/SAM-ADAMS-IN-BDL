import numpy as np


def vanilla_sgld(
    noisy_grad_fn,
    eta,
    L,
    x,
    seed=None,
):
    dim = x.shape[0]
    data = np.zeros((L, dim))
    sigma = np.sqrt(2 * eta)

    if seed is not None:
        np.random.seed(seed)

    for i in range(L):
        noisy_grad = noisy_grad_fn(x)

        dx = noisy_grad * eta + np.random.randn(dim, 1) * sigma
        dx = np.squeeze(dx)

        x = x + dx
        data[i] = x
    return data
