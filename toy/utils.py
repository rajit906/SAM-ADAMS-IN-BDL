
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from ipywidgets import interact, FloatLogSlider
import warnings
warnings.filterwarnings("ignore")

# --- Run sampler with optional trace recording ---
@njit
def run_sampler(stepper, nsteps, h, gamma, alpha, beta, burnin=1000, record_trace=False):
    x = np.array([5.0, 0.0])
    p = np.array([0.0, 0.0])
    z = 0.0
    samples = np.zeros((nsteps, 2))
    traces = np.zeros((nsteps, 6))  # [y, x, p_y, p_x, dt, T_conf]

    for t in range(nsteps + burnin):
        x, p, z, dt = stepper(x, p, z, h, gamma, alpha, beta)
        if t >= burnin:
            idx = t - burnin
            samples[idx, 0] = x[0]   # y
            samples[idx, 1] = x[1]   # x
            if record_trace:
                grad = grad_U(x)
                lapl = laplacian_U(x)
                T_conf = np.dot(grad, grad) / lapl

                traces[idx, 0] = x[0]    # y
                traces[idx, 1] = x[1]    # x
                traces[idx, 2] = p[0]    # p_y
                traces[idx, 3] = p[1]    # p_x
                traces[idx, 4] = dt      # dt
                traces[idx, 5] = T_conf  # configurational T
    return samples, traces


# --- Effective Sample Size ---
def autocorr_func_1d(x, max_lag=2000):
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    acf = result[result.size//2:] / result[result.size//2]
    return acf[:max_lag]

def ess(x, max_lag=2000):
    acf = autocorr_func_1d(x, max_lag)
    positive_acf = acf[acf > 0]
    tau = 1 + 2 * np.sum(positive_acf[1:])
    return len(x) / tau
