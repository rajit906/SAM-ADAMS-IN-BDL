import numpy as np


def rmsprop_sgld(
    noisy_grad_fn,
    eta,
    L,
    x,
    seed=None,
    lambd=0.9,
    epsilon=1e-5,
):
    dim = x.shape[0]
    data = np.zeros((L, dim))
    sigma = np.sqrt(2 * eta)

    if seed is not None:
        np.random.seed(seed)

    grad_square = np.zeros((dim, 1))

    for i in range(L):
        noisy_grad = noisy_grad_fn(x)

        grad_square = lambd * grad_square + (1 - lambd) * noisy_grad ** 2
        G = np.sqrt(grad_square) + epsilon
        G_sqrt = np.sqrt(G)

        dx = noisy_grad / G * eta + np.random.randn(dim, 1) * sigma / G_sqrt
        dx = np.squeeze(dx)
        x = x + dx
        data[i] = x
    return data
