import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from ipywidgets import interact, FloatLogSlider
import warnings
warnings.filterwarnings("ignore")

# --- BAOAB integrator ---
@njit
def step_BAOAB(x, p, z, h, gamma, alpha, beta):
    dt = h
    p -= 0.5*dt*grad_U(x)
    x += 0.5*dt*p
    c = np.exp(-gamma*dt)
    p = c*p + np.sqrt((1-c**2) / beta)*np.random.randn(2)
    x += 0.5*dt*p
    p -= 0.5*dt*grad_U(x)
    return x, p, z, dt

@njit
def g(x):
    return np.linalg.norm(grad_U(x))

@njit
def psi(z, m, M, r):
    return m * (z**r + M) / (z**r + m)

# --- ZBAOABZ integrator ---
@njit
def step_ZBAOABZ(x, p, z, dtau, gamma, alpha, beta):
    rho = np.exp(-alpha*0.5*dtau)
    z = rho*z + (1-rho) * g(x) / alpha
    dt = psi(z, m, M, r) * dtau

    p -= 0.5*dt*grad_U(x)
    x += 0.5*dt*p
    c = np.exp(-gamma*dt)
    p = c*p + np.sqrt((1-c**2) / beta)*np.random.randn(2)
    x += 0.5*dt*p
    p -= 0.5*dt*grad_U(x)

    rho = np.exp(-alpha*0.5*dtau)
    z = rho*z + (1-rho) * g(x) / alpha
    return x, p, z, dt