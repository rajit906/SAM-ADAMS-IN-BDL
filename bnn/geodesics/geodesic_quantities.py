import numpy as np
from scipy.integrate import solve_ivp


def dot(x1, x2):
    # Inspired by bnn_priors
    return np.sum(x1 * x2)


def get_monge_fun(dim, M, alpha_2):
    def func(t, y):
        x = y[:dim]
        v = y[dim:]
        x_grad = M.dlogp(x)
        norm_x_grad_2 = dot(x_grad, x_grad)

        W_2 = 1 + alpha_2 * norm_x_grad_2
        mho = alpha_2 * dot(v, M.hvp_logp(x, v)) / W_2

        return np.concatenate([v, -mho * x_grad])

    return func


def geodesic_monge(M, x, v, alpha_2=1.0, t_span=(0.0, 1.0)):
    # Geodesic equation with Monge metric
    fun = get_monge_fun(x.shape[0], M=M, alpha_2=alpha_2)
    return solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=np.concatenate([x, v]),
        t_eval=np.linspace(0.0, 1.0, 20)
    )
