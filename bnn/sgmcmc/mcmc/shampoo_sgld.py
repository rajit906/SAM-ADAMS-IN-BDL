import numpy as np


def get_metrics(H):
    w, v = np.linalg.eig(H)
    G_r = v @ np.diag(w ** (-1/2)) @ v.T
    G_rsqrt = v @ np.diag(w ** (-1/4)) @ v.T
    return G_r, G_rsqrt

def shampoo_sgld(
    noisy_grad_fn,
    eta,
    L,
    x,
    seed=None,
    lambd=0.9,
    epsilon=1e-3,
    update_interval=1,
):
    # We recover roughly full matrix RMSprop when dimension of gradients is 1.
    dim = x.shape[0]
    data = np.zeros((L, dim))
    sigma = np.sqrt(2 * eta)

    if seed is not None:
        np.random.seed(seed)

    H = np.eye(dim) * epsilon

    for i in range(L):
        noisy_grad = noisy_grad_fn(x)

        H = lambd * H + (1 - lambd) * noisy_grad * noisy_grad.T

        if i % update_interval == 0:
            G_r, G_rsqrt = get_metrics(H)

        dx = G_r @ noisy_grad * eta + G_rsqrt @ np.random.randn(dim, 1) * sigma
        dx = np.squeeze(dx)
        x = x + dx
        data[i] = x
    return data
