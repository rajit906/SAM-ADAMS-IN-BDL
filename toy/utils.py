import numpy as np
from numba import njit

@njit
def run_sampler(stepper, nsteps, h, gamma, alpha, beta,
                grad_U, m, M, r, s,
                burnin, record_trace=False):
    """
    Runs a sampler for given potential and returns samples & traces.
    """
    x = np.array([1.0, 1.0])
    p = np.array([0.0, 0.0])
    z = 0.0
    samples = np.zeros((nsteps, 2))
    traces = np.zeros((nsteps, 6))

    for t in range(nsteps + burnin):
        x, p, z, dt = stepper(x, p, z, h, gamma, alpha, beta,
                                  grad_U, m, M, r, s)

        if t >= burnin:
            idx = t - burnin
            samples[idx, :] = x
            if record_trace:
                grad = grad_U(x)
                T_conf = np.dot(grad, x) / len(x) # This should be 
                traces[idx] = np.array([x[0], x[1], p[0], p[1], dt, T_conf])

    return samples, traces


def autocorr_func_1d(x, max_lag=2000):
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode="full")
    acf = result[result.size // 2:] / result[result.size // 2]
    return acf[:max_lag]


def ess(x, max_lag=2000):
    acf = autocorr_func_1d(x, max_lag)
    positive_acf = acf[acf > 0]
    tau = 1 + 2 * np.sum(positive_acf[1:])
    return len(x) / tau