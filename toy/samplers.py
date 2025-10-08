import numpy as np
from numba import njit

# ============================================================
# Generic adaptive samplers for any potential grad_U(x)
# ============================================================

@njit
def psi_fn(z, m, M, r):
    """Bounded scaling function ψ(z) in [m, M]."""
    return m * (z**r + M/m) / (z**r + 1.0)

# ============================================================
# BAOAB (standard SGHMC / Langevin splitting)
# ============================================================
@njit
def step_BAOAB(x, p, z, h, gamma, alpha, beta, grad_U, m, M, r):
    """
    One BAOAB step for given potential grad_U(x).
    Parameters
    ----------
    x, p : ndarray(2,)
        Position and momentum.
    z : float (unused)
    h : float
        Step size (Δt).
    gamma : float
        Friction coefficient.
    alpha : float (unused)
    beta : float
        Inverse temperature.
    grad_U : callable
        Gradient of potential energy U(x).
    """
    dt = h
    p -= 0.5 * dt * grad_U(x)
    x += 0.5 * dt * p

    c = np.exp(-gamma * dt)
    p = c * p + np.sqrt((1.0 - c**2) / beta) * np.random.randn(2)

    x += 0.5 * dt * p
    p -= 0.5 * dt * grad_U(x)
    return x, p, z, dt


# ============================================================
# ZBAOABZ (SAM-ADAMS adaptive step)
# ============================================================
@njit
def step_ZBAOABZ(x, p, z, dtau, gamma, alpha, beta,
                 grad_U, m, M, r):
    """
    Adaptive BAOAB with auxiliary variable z.
    """
    # Monitor g = ||grad_U(x)|| (can replace with other definitions)
    g_val = np.linalg.norm(grad_U(x))

    # Half update of z in τ-time
    rho = np.exp(-alpha * 0.5 * dtau)
    z = rho * z + (1.0 - rho) * g_val / alpha

    # Effective step in t-time
    psi = psi_fn(z, m, M, r)
    dt = psi * dtau

    # BAOAB main updates
    p -= 0.5 * dt * grad_U(x)
    x += 0.5 * dt * p
    c = np.exp(-gamma * dt)
    p = c * p + np.sqrt((1.0 - c**2) / beta) * np.random.randn(2)
    x += 0.5 * dt * p
    p -= 0.5 * dt * grad_U(x)

    # Second half update of z
    g_val = np.linalg.norm(grad_U(x))
    z = rho * z + (1.0 - rho) * g_val / alpha

    return x, p, z, dt
