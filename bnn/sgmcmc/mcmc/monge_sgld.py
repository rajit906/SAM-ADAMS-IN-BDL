import numpy as np


def get_metrics(dim, grad, alpha_2):
    grad_norm_2 = np.sum(grad**2)
    G_r = np.eye(dim) - alpha_2 / (1 + alpha_2 * grad_norm_2) * grad @ grad.T
    G_rsqrt = (
        np.eye(dim)
        + (1 / np.sqrt(1 + alpha_2 * grad_norm_2) - 1) / grad_norm_2 * grad @ grad.T
    )
    return G_r, G_rsqrt


def monge_sgld(
    noisy_grad_fn,
    eta,
    L,
    x,
    alpha_2=1e-3,
    seed=None,
    lambd=0.9,
):
    threshold = 1e3
    dim = x.shape[0]
    data = np.zeros((L, dim))
    sigma = np.sqrt(2 * eta)

    if seed is not None:
        np.random.seed(seed)

    p_grad = np.zeros((dim, 1))

    for i in range(L):
        noisy_grad = noisy_grad_fn(x)
        p_grad = lambd * p_grad + (1 - lambd) * noisy_grad

        G_r, G_rsqrt = get_metrics(dim, p_grad, alpha_2)
        precond_grad = G_r @ noisy_grad
        precond_grad_norm = np.linalg.norm(precond_grad)

        # Avoid numerical issues
        if precond_grad_norm > threshold:
            factor = np.linalg.norm(noisy_grad) / threshold
            dx = (
                noisy_grad / factor * eta
                + np.random.randn(dim, 1) / np.sqrt(factor) * sigma
            )
        else:
            dx = precond_grad * eta + G_rsqrt @ np.random.randn(dim, 1) * sigma
        dx = np.squeeze(dx)

        x = x + dx

        # if x[1] < -1e2 or x[1] > 1e2 or x[0] < -1e2 or x[0] > 1e2:
        #     print("Breaking")
        #     return data[:i]

        data[i] = x
    return data
