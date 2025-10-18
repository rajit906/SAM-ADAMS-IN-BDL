import numpy as np

# ============================================================
# Generic adaptive samplers for any potential grad_U(x)
# ============================================================
def psi_fn(z, m, M, r):
    """Bounded scaling function psi(z) in (m, M]."""
    return m * (z**r + M) / (z**r + m)
    return m * (z ** r + M/m) / (z ** r + 1)

# ============================================================
# BAOAB (standard SGHMC / Langevin splitting)
# ============================================================
def step_BAOAB_SGHMC(x, p, z, h, gamma, alpha, beta, grad_U, m, M, r, s):
    """
    One BAOAB step for a given potential grad_U (x).
    Parameters
    ----------
    x, p : ndarray(2, )
        Position and momentum.
    z : float (unused)
    h : float
        Step size (delta t).
    gamma : float
        Friction coefficient.
    alpha : float (unused)
    beta : float
        Inverse temperature.
    grad_U : callable
        Gradient of potential energy U(x).
    """
    dt = h

    # half B
    p -= 0.5 * dt * grad_U(x)
    # half A
    x += 0.5 * dt * p
    # O
    c = np.exp(- gamma * dt)
    p = c * p + np.sqrt((1.0 - c**2) / beta) * np.random.randn(2)
    # half A
    x += 0.5 * dt * p
    # half B
    p -= 0.5 * dt * grad_U(x)

    return x, p, z, dt

# ============================================================
# ZBAOABZ (SAM-ADAMS adaptive step)
# ============================================================
def step_ZBAOABZ_SGHMC(x, p, z, dtau, gamma, alpha, beta, grad_U, m, M, r, s):
    """
    Adaptive BAOAB with auxiliary variable z.
    """
    omega = 100
    # Z
    g_val = omega ** (-1) * np.linalg.norm(grad_U(x)) ** s
    rho = np.exp(- alpha * 0.5 * dtau)
    z = rho * z + (1.0 - rho) * g_val / alpha
    # Computing dt for this step
    dt = psi_fn(z, m, M, r) * dtau # bounded in (m * dtau, M * dtau]

    # BAOAB main updates
    # half B
    p -= 0.5 * dt * grad_U(x)
    # half A
    x += 0.5 * dt * p
    # O
    c = np.exp(-gamma * dt)
    p = c * p + np.sqrt((1.0 - c**2) / beta) * np.random.randn(2)
    # half A
    x += 0.5 * dt * p
    # half B
    p -= 0.5 * dt * grad_U(x)

    # Z
    g_val = omega ** (-1) * np.linalg.norm(grad_U(x)) ** s
    z = rho * z + (1.0 - rho) * g_val / alpha

    return x, p, z, dt

# ============================================================
# Overdamped Langevin (fixed step, baseline)
# ============================================================
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
def step_ZOLD(x, p, z, dtau, gamma, alpha, beta, grad_U, m, M, r, s):
    """
    Adaptive overdamped Langevin with an auxiliary Z variable.
    No momentum p and no friction. Still tracks z to adapt step size.
    """
    omega = 100
    # --- Z half step ---
    g_val = omega ** (-1) * np.linalg.norm(grad_U(x)) ** s
    rho = np.exp(- alpha * 0.5 * dtau)
    z = rho * z + (1.0 - rho) * g_val / alpha

    # --- Adaptive step size ---
    dt = psi_fn(z, m, M, r) * dtau

    # --- Overdamped Langevin step ---
    noise = np.sqrt(2.0 * dt / beta) * np.random.randn(2)
    x = x - dt * grad_U(x) + noise

    # --- Z final half step ---
    g_val = omega ** (-1) * np.linalg.norm(grad_U(x)) ** s
    z = rho * z + (1.0 - rho) * g_val / alpha

    return x, p, z, dt
