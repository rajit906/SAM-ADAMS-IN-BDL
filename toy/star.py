import numpy as np
from viz import plot_samplers, plot_gmm_color, contour


def grad_U(z):
    x, y = z
    # print(x,y)
    # exit()
    grad_x = (1 + 1000 * y ** 2) * 2 * x
    grad_y = (1 + 1000 * x ** 2) * 2 * y
    return np.array([grad_x, grad_y])

def log_p(XY):
    x, y = XY[..., 0], XY[..., 1]
    return - (x ** 2 + 1000 * x ** 2 * y ** 2 + y ** 2)

# contour(log_p, xlim=[-10., 10.], ylim=[-10., 10.])
# exit()

# ====================================================
# Plot setup
# ====================================================
xs = np.linspace(-10, 10, 400)
ys = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(xs, ys)
LOGZ = log_p(np.stack([X, Y], axis=-1))
vmax, vmin = LOGZ.max(), LOGZ.max() - 500
levels = np.linspace(vmin, vmax, 50)

# sampler parameters
m, M, r, s = 0.1, 10, 0.5, 2
b = 0.4  # BAOAB stepsize multiplier
burnin = int(1e4)
nsteps = int(1e6)

# ====================================================
# Run and visualize
# ====================================================
plot_samplers(
    alpha=0.5,
    h=0.001,
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
    plot_stride=5,
    order=1,
    saveto="star.png"
)
