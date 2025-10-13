import numpy as np
from numba import njit

# ============================================================
# Generic adaptive samplers for any potential grad_U(x)
# ============================================================
@njit
def psi_fn(z, m, M, r):
    """Bounded scaling function ψ(z) in [m, M]."""
    return m * (z**r + M) / (z**r + m)

# ============================================================
# BAOAB (standard SGHMC / Langevin splitting)
# ============================================================
@njit
def step_BAOAB_SGHMC(x, p, z, h, gamma, alpha, beta, grad_U, m, M, r, s):
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


@njit
def step_EM_SGHMC(x, p, z, h, gamma, alpha, beta, grad_U, m, M, r, s):
    """
    One Euler–Maruyama step for underdamped Langevin.
    Same signature as step_BAOAB for easy swapping.
    """
    dt = h
    noise = np.sqrt(2.0 * gamma * dt / beta) * np.random.randn(2)
    p = p - grad_U(x) * dt - gamma * p * dt + noise
    x = x + p * dt
    return x, p, z, dt

# ============================================================
# ZBAOABZ (SAM-ADAMS adaptive step)
# ============================================================
@njit
def step_ZBAOABZ_SGHMC(x, p, z, dtau, gamma, alpha, beta,
                 grad_U, m, M, r, s):
    """
    Adaptive BAOAB with auxiliary variable z.
    """

    # Z-step
    g_val = np.linalg.norm(grad_U(x))**s
    rho = np.exp(-alpha * 0.5 * dtau)
    z = rho * z + (1.0 - rho) * g_val / alpha

    # Computing dt for this step
    psi = psi_fn(z, m, M, r)
    dt = psi * dtau

    # BAOAB main updates
    p -= 0.5 * dt * grad_U(x) # B-step
    x += 0.5 * dt * p # A-step
    c = np.exp(-gamma * dt)
    p = c * p + np.sqrt((1.0 - c**2) / beta) * np.random.randn(2) # O-step
    x += 0.5 * dt * p # A-step
    p -= 0.5 * dt * grad_U(x) # B-step

    # Z-step
    g_val = np.linalg.norm(grad_U(x))
    z = rho * z + (1.0 - rho) * g_val / alpha

    return x, p, z, dt

@njit
def step_EM_ZSGHMC(x, p, z, dtau, gamma, alpha, beta,
             grad_U, m, M, r, s):
    """
    Adaptive Euler–Maruyama with Z-dynamics (analogous to ZBAOABZ).
    """
    # Z-step (same as ZBAOABZ)
    g_val = np.linalg.norm(grad_U(x))**s
    rho = np.exp(-alpha * 0.5 * dtau)
    z = rho * z + (1.0 - rho) * g_val / alpha

    # Scaled timestep
    psi = psi_fn(z, m, M, r)
    dt = psi * dtau

    # Euler–Maruyama update
    noise = np.sqrt(2.0 * gamma * dt / beta) * np.random.randn(2)
    p = p - grad_U(x) * dt - gamma * p * dt + noise
    x = x + p * dt

    # Second Z-step
    g_val = np.linalg.norm(grad_U(x))
    z = rho * z + (1.0 - rho) * g_val / alpha

    return x, p, z, dt


# ============================================================
# Overdamped Langevin (fixed step, baseline)
# ============================================================
@njit
def step_OLD(x, p, z, h, gamma, alpha, beta, grad_U, m, M, r, s):
    """
    Standard overdamped Langevin.
    Ignores p, gamma (passed only for interface consistency).
    """
    dt = h
    noise = np.sqrt(2.0 * dt / beta) * np.random.randn(2)
    x = x - dt * grad_U(x) + noise
    return x, p, z, dt


# ============================================================
# Adaptive Overdamped Langevin (Z dynamics)
# ============================================================
@njit
def step_ZOLD(x, p, z, dtau, gamma, alpha, beta, grad_U, m, M, r, s):
    """
    Adaptive overdamped Langevin with auxiliary Z variable.
    No momentum p and no friction. Still tracks z to adapt step size.
    """
    # --- Z half step ---
    g_val = np.linalg.norm(grad_U(x))**s
    rho = np.exp(-alpha * 0.5 * dtau)
    z = rho * z + (1.0 - rho) * g_val / alpha

    # --- Adaptive step size ---
    psi = psi_fn(z, m, M, r)
    dt = psi * dtau

    # --- Overdamped Langevin step ---
    noise = np.sqrt(2.0 * dt / beta) * np.random.randn(2)
    x = x - dt * grad_U(x) + noise

    # --- Z final half step ---
    g_val = np.linalg.norm(grad_U(x))**s
    z = rho * z + (1.0 - rho) * g_val / alpha

    return x, p, z, dt

# ------------------------------------------------------------
# pSGLD (preconditioned SGLD à la RMSProp)
# ------------------------------------------------------------
@njit
def step_pSGLD(x, h, beta,
               grad_U, eps=1e-8, lambd=0.99):
    """
    Preconditioned SGLD using RMSProp-like adaptive scaling.
    Designed to match run_sampler() interface.

    Parameters
    ----------
    x : position vector
    h : base stepsize
    beta : inverse temperature
    eps : numerical stability
    lambda : RMSProp decay factor
    """

    dt = h
    g = grad_U(x)                 # gradient
    V = lambd * V + (1.0 - lambd) * g * g    # accumulate grad^2, this is element-wise.
    G = 1.0 / (np.sqrt(V) + eps)   # diagonal scaling
    G_sqrt = np.sqrt(G)

    noise = np.sqrt(2.0 * dt / beta) * np.random.randn(x.shape[0])
    x = x - dt * G * g + G_sqrt * noise # G should be a diagonal matrix but we are just doing elementwise vector multiplication to save memory.
    # Note that this is also missing a divergence term that will indeed cause bias.
    return x