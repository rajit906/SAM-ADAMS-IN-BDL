import numpy as np
from viz import plot_samplers, plot_gmm_color, contour
from sklearn.mixture import GaussianMixture

# ====================================================
# 2D Gaussian Mixture Model (GMM)
# ====================================================

# Define mixture components
# mildly unbalanced (one dominant mode, one rare spike)
weights = np.array([0.10, 0.05, 0.25, 0.20, 0.40])

means = np.array([
    [-3.0, -3.0],   # 1: elongated and correlated
    [ 7.0,  7.0],   # 2: very tight spike (hard to hit)
    [-2.0,  7.5],   # 3: medium, tilted
    [ 7.5, -4.5],   # 4: tight-ish, correlated
    [ 2.0,  2.0],   # 5: broad basin (dominant mass)
])

covs = np.array([
    [[0.15,  0.12],
     [0.12,  0.50]],   # det = 0.0606  -> PD, long narrow diagonal

    [[0.03,  0.00],
     [0.00,  0.05]],   # very tight spike

    [[0.10, -0.06],
     [-0.06, 0.20]],   # tilted ellipse (det = 0.0164)

    [[0.40,  0.18],
     [0.18,  0.15]],   # tight & correlated (det = 0.0276)

    [[1.50,  0.00],
     [0.00,  2.50]],   # broad, easy basin
])
inv_covs = np.linalg.inv(covs)
det_covs = np.array([np.linalg.det(c) for c in covs])

# ====================================================
# Gradient of potential energy U = - log p
# ====================================================
def grad_U(z):
    x, y = z
    p_total = 0.0
    grad_total = np.zeros(2)
    for i in range(len(weights)):
        diff = np.array([x - means[i][0], y - means[i][1]])
        coef = weights[i] * np.exp(- 0.5 * diff @ inv_covs[i] @ diff) / np.sqrt((2 * np.pi) ** 2 * det_covs[i])
        p_total += coef
        grad_total += coef * (inv_covs[i] @ diff)

    return grad_total / max(p_total, 1e-16)

# ====================================================
# Log target density (for contours)
# ====================================================
def log_p(XY):
    log_probs = np.zeros(XY.shape[:-1])
    for k in range(len(weights)):
        diff = XY - means[k]
        exponent = - 0.5 * np.sum(diff @ inv_covs[k] * diff, axis=-1)
        log_probs += weights[k] * np.exp(exponent) / np.sqrt((2 * np.pi) ** 2 * det_covs[k])
    return np.log(log_probs)

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
m, M, r, s = 0.5, 50, 0.25, 2
b = 7.6  # BAOAB stepsize multiplier
burnin = int(1e3)
nsteps = int(1e6)

# ====================================================
# Run and visualize
# ====================================================
print("sampling from GMM to prepare MMD")
gmm = GaussianMixture(n_components=len(weights), covariance_type='full')
gmm.weights_ = weights
gmm.means_ = means
gmm.covariances_ = covs
gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
true_samples, _ = gmm.sample(nsteps)
print('finish')
plot_gmm_color(
    alpha=0.05,
    h=0.01,
    gamma=0.5,
    beta=0.7,
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
    true_samples=true_samples,
    pis=weights,
    mus=means,
    Sigmas=covs,
    nsteps=nsteps,
    plot_stride=10,
    order=1,
    saveto="gmm_color.png"
)