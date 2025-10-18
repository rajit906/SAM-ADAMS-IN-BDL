import numpy as np
from viz import plot_samplers, plot_gmm_color, contour

# Parameters
a = 1.

def g(y):
    return np.exp(a * y)

def dg(y):
    return a * np.exp(a * y)

def grad_U(z):
    x, y = z
    g_y = g(y)
    dg_y = dg(y)

    grad_x = x / g_y
    grad_y = - 0.5 * x ** 2 * dg_y / g_y ** 2 + 0.5 * dg_y / g_y + y

    return np.array([grad_x, grad_y])

def log_p(XY):
    x, y = XY[..., 0], XY[..., 1]
    g_y = g(y)
    U = 0.5 * x ** 2 / g_y + 0.5 * np.log(g_y) + 0.5 * y ** 2 + np.log(2 * np.pi)
    return - U

# contour(log_p, xlim=[-10., 10.], ylim=[-10., 10.])
# exit()

# ====================================================
# Plot setup
# ====================================================
xs = np.linspace(-20, 20, 400)
ys = np.linspace(-20, 20, 400)
X, Y = np.meshgrid(xs, ys)
LOGZ = log_p(np.stack([X, Y], axis=-1))
vmax, vmin = LOGZ.max(), LOGZ.max() - 50
levels = np.linspace(vmin, vmax, 50)

# sampler parameters
m, M, r, s = 0.1, 10.0, 1., 2
b = 6.5  # BAOAB stepsize multiplier
burnin = int(1e3)
nsteps = int(2e6)

# ====================================================
# Run and visualize
# ====================================================
plot_samplers(
    alpha=0.8,
    h=0.002,
    gamma=0.5,
    beta=1.0,
    grad_U=grad_U,
    X=X,
    Y=Y,
    LOGZ=LOGZ,
    log_p_xy=log_p,
    levels=levels,
    m=m,
    M=M,
    r=r,
    s=s,
    b=b,
    burnin=burnin,
    nsteps=nsteps,
    plot_stride=1,
    order=1,
    saveto="neal.png"
)
