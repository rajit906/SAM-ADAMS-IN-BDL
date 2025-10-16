import numpy as np
from scipy.special import psi, gammaln
from sklearn.neighbors import NearestNeighbors



def run_sampler(stepper, nsteps, h, gamma, alpha, beta, grad_U, m, M, r, s, burnin):
    """
    Runs a sampler for a given potential and returns samples & traces.
    """
    x = np.array([0.05, 0.05])
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
            grad = grad_U(x)
            T_conf = np.dot(grad, x) / len(x)
            traces[idx] = np.array([x[0], x[1], p[0], p[1], dt, T_conf])

    return samples, traces

def autocorr_func_1d(x, max_lag=1000):
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode="full")
    acf = result[result.size // 2:] / result[result.size // 2]
    return acf[:max_lag]


def ess(x, max_lag=1000):
    acf = autocorr_func_1d(x, max_lag)
    positive_acf = acf[acf > 0]
    tau = 1 + 2 * np.sum(positive_acf[1:])
    return len(x) / tau

############################## KL
# ---------- 1) kNN entropy (Kozachenko–Leonenko) ----------
def knn_entropy(samples, k=5):
    """
    Kozachenko–Leonenko entropy estimator for q given samples ~ q.
    samples: (N, d) array
    k: 3..10 is typical
    Returns scalar H_hat in nats.
    """
    X = np.asarray(samples, dtype=float)
    N, d = X.shape

    # k+1 because the nearest neighbor of each point is itself
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(X)
    dists, _ = nn.kneighbors(X)     # shape (N, k+1); dists[:,0] = 0 (self)
    eps = dists[:, -1]              # distance to k-th neighbor

    # unit-ball volume in R^d
    c_d = (np.pi**(d/2)) / np.exp(gammaln(1 + d/2))

    # KL estimator (natural log => nats)
    H = psi(N) - psi(k) + np.log(c_d) + d * np.mean(np.log(eps + 1e-12))
    return H

# ---------- 2) Cross-entropy ----------
def cross_entropy_qp(samples, logp_fn):
    """
    Monte Carlo cross-entropy H(q,p) = -E_q[log p(X)].
    samples: (N, d)
    logp_fn: function that takes samples and returns log p for each row
             Either vectorized (N,)->(N,).
    """
    X = np.asarray(samples, dtype=float)
    logp_vals = logp_fn(X)  # vectorized over rows
    return -np.mean(logp_vals)

# ---------- 3) KL(q||p) estimator ----------
def kl_qp(samples, logp_fn, k=5):
    """
    Estimate KL(q||p) given samples ~ q and a log-density function for p.
    Returns scalar in nats.
    """
    H_qp = cross_entropy_qp(samples, logp_fn)
    H_q  = knn_entropy(samples, k=k)
    return H_qp - H_q

############################## MMD
def mmd2_unbiased(samples1, samples2, *, sigma2=None, bandwidth_pairs=int(1e5), rng=None):
    """
    Compute MMD^2 between two sample sets using an RBF kernel.

    - Uses the linear-time unbiased estimator (O(N))
      MMD^2 = mean_i [ k(x_{2i-1},x_{2i}) + k(y_{2i-1},y_{2i})
                        - k(x_{2i-1},y_{2i}) - k(x_{2i},y_{2i-1}) ]
    - Bandwidth (sigma^2) via median heuristic estimated from a small number
      of random pairs (default 5k) unless provided.

    Parameters
    ----------
    samples1 : (N1, d) array-like
    samples2 : (N2, d) array-like
    sigma2   : float or None
        Kernel variance. If None, it’s estimated from random pairs.
    bandwidth_pairs : int
        #random pairs to estimate median squared distance (heuristic).
        Keep this modest (e.g., 2k–20k) for speed.
    rng : np.random.Generator or None
        RNG for bandwidth estimation and pairing.

    Returns
    -------
    mmd2 : float
        Estimated MMD^2 (nonnegative; 0 means identical in RKHS).
    sigma2 : float
        The kernel variance used.
    """
    X = np.asarray(samples1, dtype=float)
    Y = np.asarray(samples2, dtype=float)
    rng = np.random.default_rng() if rng is None else rng

    # Match lengths for pairing (drop the last if odd)
    n = min(len(X), len(Y))
    if n < 2:
        raise ValueError("Need at least 2 samples in each set.")
    n = n - (n % 2)
    X = X[:n]
    Y = Y[:n]

    # --- Bandwidth via median heuristic on random pairs (no O(N^2) memory) ---
    if sigma2 is None:
        def sample_sq_dists(A, B, pairs):
            ia = rng.integers(0, len(A), size=pairs)
            ib = rng.integers(0, len(B), size=pairs)
            diff = A[ia] - B[ib]
            return np.sum(diff * diff, axis=1)

        # Mix of within and cross distances gives a robust scale
        p_each = bandwidth_pairs // 3
        d_xx = sample_sq_dists(X, X, p_each)
        d_yy = sample_sq_dists(Y, Y, p_each)
        d_xy = sample_sq_dists(X, Y, bandwidth_pairs - 2 * p_each)
        med = np.median(np.concatenate([d_xx, d_yy, d_xy]))
        # Guard against degenerate cases
        sigma2 = float(med if med > 1e-12 else 1.0)

    inv2sigma2 = 1.0 / (2.0 * sigma2)

    # Pair up for the linear-time unbiased estimator
    a = X[0::2]; b = X[1::2]
    c = Y[0::2]; d = Y[1::2]

    # RBF kernel on paired rows
    def kpair(U, V):
        diff = U - V
        return np.exp(-np.sum(diff * diff, axis=1) * inv2sigma2)

    mmd2 = np.mean(kpair(a, b) + kpair(c, d) - kpair(a, d) - kpair(b, c))
    # Numerical guard: small negatives can occur from FP error
    return float(max(mmd2, 0.0))
