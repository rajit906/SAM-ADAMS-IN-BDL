import numpy as np
import itertools
import matplotlib.pyplot as plt
from toy.samplers import step_OLD, step_ZOLD
from utils import run_sampler, kl_qp, mmd2_unbiased, ess
from samplers import step_BAOAB_SGHMC, step_ZBAOABZ_SGHMC


def plot_samplers(alpha, h, gamma, beta, grad_U, X, Y, LOGZ, log_p_xy, levels, m, M, r, s, b, nsteps, burnin, plot_stride, order, saveto):
    if order == 1:
        title, title_z = "SGLD", "ZSGLD"
        samples, traces = run_sampler(step_OLD, nsteps, h * b, gamma, alpha, beta, grad_U, m, M, r, s, burnin=burnin)
        print('---- Finished running SGLD ----')
        kl = kl_qp(samples, log_p_xy)
        print(f'KL(q||p) for SGLD is {kl:.5f}')
        samples_z, traces_z = run_sampler(step_ZOLD, nsteps, h, gamma, alpha, beta, grad_U, m, M, r, s, burnin=burnin)
        print('---- Finished running ZSGLD ----')
        kl_z = kl_qp(samples_z, log_p_xy)
        print(f'KL(q||p) for ZSGLD is {kl_z:.5f}')
    else:
        title, title_z = "BAOAB", "ZBAOABZ"
        samples, traces = run_sampler(step_BAOAB_SGHMC, nsteps, h * b, gamma, alpha, beta, grad_U, m, M, r, s, burnin=burnin)
        print('---- Finished running BAOAB ----')
        print('---- Finished running SGLD ----')
        kl = kl_qp(samples, log_p_xy)
        samples_z, traces_z = run_sampler(step_ZBAOABZ_SGHMC, nsteps, h, gamma, alpha, beta, grad_U, m, M, r, s, burnin=burnin)
        print('---- Finished running ZBAOABZ ----')
        kl_z = kl_qp(samples_z, log_p_xy)
        print(f'KL(q||p) for ZSGLD is {kl_z:.5f}')

    print('computing ESS')
    ESS = ess(traces[:, 0])
    ESS_z = ess(traces_z[:, 0])
    print('finished computing ESS')

    # --- Plotting setup ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.5)

    # --- Determine shared equal scaling ---
    all_y = np.concatenate([samples[:, 0], samples_z[:, 0]])
    all_x = np.concatenate([samples[:, 1], samples_z[:, 1]])

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    max_range = max(x_range, y_range) / 2.0

    mid_x = (all_x.max() + all_x.min()) / 2.0
    mid_y = (all_y.max() + all_y.min()) / 2.0

    xlim = (mid_x - max_range, mid_x + max_range)
    ylim = (mid_y - max_range, mid_y + max_range)

    # --- Row 0: Contours + scatter (no lines, transparency) ---
    ax0 = fig.add_subplot(gs[0, 0])
    idx = slice(None, None, plot_stride)
    ax0.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
    ax0.scatter(samples[::plot_stride, 0], samples[::plot_stride, 1], s=1, color='red', alpha=0.5)
    ax0.set_title(f'{title} (h={h}, γ={gamma}, α={alpha})')
    ax0.set_xlabel('x'); ax0.set_ylabel('y')
    ax0.set_xlim(xlim); ax0.set_ylim(ylim); ax0.set_aspect('equal', 'box')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
    ax1.scatter(samples_z[::plot_stride, 0], samples_z[::plot_stride, 1], s=1, color='red', alpha=0.5)
    ax1.set_title(f'{title_z} (h={h}, γ={gamma}, α={alpha})')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_aspect('equal', 'box')

    # --- Row 1: Step size traces ---
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.plot(traces[::plot_stride, 4], lw=0.6)
    ax_left.set_title(f"{title} trace: dt (step size)")
    ax_left.set_xlabel("Step")

    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.plot(traces_z[::plot_stride, 4], lw=0.6)
    ax_right.set_title(f"{title_z} trace: dt (step size)")
    ax_right.set_xlabel("Step")

    # --- Row 2: configurational temperature ---
    T_conf_mean = np.mean(traces[:, 5])
    T_conf_mean_z = np.mean(traces_z[:, 5])
    ax_left = fig.add_subplot(gs[2, 0])
    ax_left.plot(traces[idx, 5], lw=0.7, label="T_conf")
    ax_left.hlines(T_conf_mean, 0, len(traces[idx, 5]), color="red", lw=1.5, linestyle=":",
                   label=f"T_conf_mean={T_conf_mean:.3f}")
    ax_left.set_title(f"{title} trace: Configurational T")
    ax_left.set_xlabel("Step")
    ax_left.legend()

    ax_right = fig.add_subplot(gs[2, 1])
    ax_right.plot(traces_z[idx, 5], lw=0.7, label="T_conf")
    ax_right.hlines(T_conf_mean_z, 0, len(traces_z[idx, 5]), color="red", lw=1.5, linestyle=":",
                    label=f"T_conf_mean={T_conf_mean_z:.3f}")
    ax_right.set_title(f"{title_z} trace: Configurational T")
    ax_right.set_xlabel("Step")
    ax_right.legend()

    # --- Row 3: metrics comparison ---
    ax_m = fig.add_subplot(gs[3, :])

    x = np.arange(4)
    labels = ["ESS", "KL", "ESS(Z)", "KL(Z)"]

    # left axis: ESS bars
    left_mask = np.array([1, 0, 1, 0], dtype=bool)
    ess_vals = np.array([ESS, 0, ESS_z, 0], dtype=float)
    b1 = ax_m.bar(x[left_mask], ess_vals[left_mask], color="tab:blue", alpha=0.7, label="ESS")

    # right axis: KL/MMD bars
    ax2 = ax_m.twinx()
    right_mask = ~left_mask
    km_vals = np.array([0, kl, 0, kl_z], dtype=float)
    b2 = ax2.bar(x[right_mask], km_vals[right_mask], color="tab:green", alpha=0.7, label="KL")
    ax2.set_yscale("log")  # KL/MMD often vary over orders of magnitude

    ax_m.set_xticks(x)
    ax_m.set_xticklabels(labels)
    ax_m.set_title("Metrics")
    ax_m.set_ylabel("ESS")
    ax2.set_ylabel("KL")

    # one legend for both
    bars = [b1, b2]
    labs = ["ESS", "KL"]
    ax_m.legend(bars, labs, loc="upper right")

    plt.savefig(f'./toy/{saveto}')
    plt.show()

def gmm_labels(samples, pis, mus, Sigmas):
    """
    samples: (N, d)
    pis:     (K,) mixture weights (sum to 1)
    mus:     (K, d)
    Sigmas:  (K, d, d) full covariances  (use np.diag for diagonal)
    returns:
      labels: (N,) argmax responsibilities in 0..K-1
      resp:   (N, K) responsibilities
    """
    N, d = samples.shape
    K = len(pis)
    logp = np.empty((N, K))
    # precompute inverses and log|Σ|
    invs = []
    logdets = []
    for k in range(K):
        sign, logdet = np.linalg.slogdet(Sigmas[k])
        if sign <= 0:
            raise ValueError("Covariance not PD")
        logdets.append(logdet)
        invs.append(np.linalg.inv(Sigmas[k]))
    invs = np.stack(invs)
    logdets = np.array(logdets)

    # log N(x|μ,Σ) up to constant
    # full: -0.5 [ d*log(2π) + log|Σ| + (x-μ)^T Σ^{-1} (x-μ) ]
    # we include full constant so responsibilities are correct
    const = -0.5 * d * np.log(2*np.pi)

    X = samples  # (N,d)
    for k in range(K):
        diff = X - mus[k]          # (N,d)
        maha = np.einsum('ni,ij,nj->n', diff, invs[k], diff)  # (N,)
        logN = const - 0.5*logdets[k] - 0.5*maha              # (N,)
        logp[:, k] = np.log(pis[k]) + logN

    # log-sum-exp for normalization
    m = logp.max(axis=1, keepdims=True)
    w = np.exp(logp - m)
    resp = w / w.sum(axis=1, keepdims=True)   # (N,K)
    labels = resp.argmax(axis=1)
    return labels, resp

def labels_to_colors(labels, K, cmap_name='tab10'):
    cmap = plt.get_cmap(cmap_name, K)
    return cmap(labels % K)

def plot_gmm_color(alpha, h, gamma, beta, grad_U, X, Y, LOGZ, log_p_xy, levels, m, M, r, s, b, nsteps, burnin, true_samples, pis, mus, Sigmas, plot_stride, order, saveto):
    if order == 1:
        title, title_z = "SGLD", "ZSGLD"
        samples, traces = run_sampler(step_OLD, nsteps, h * b, gamma, alpha, beta, grad_U, m, M, r, s, burnin=burnin)
        print('---- Finished running SGLD ----')
        kl = kl_qp(samples, log_p_xy)
        mmd2 = mmd2_unbiased(samples, true_samples)
        print(f'KL(q||p) for SGLD is {kl:.5f}')
        print(f'MMD^2 for SGLD is {mmd2:.5f}')
        samples_z, traces_z = run_sampler(step_ZOLD, nsteps, h, gamma, alpha, beta, grad_U, m, M, r, s, burnin=burnin)
        print('---- Finished running ZSGLD ----')
        kl_z = kl_qp(samples_z, log_p_xy)
        mmd2_z = mmd2_unbiased(samples_z, true_samples)
        print(f'KL(q||p) for ZSGLD is {kl_z:.5f}')
        print(f'MMD^2 for ZSGLD is {mmd2_z:.5f}')
    else:
        title, title_z = "BAOAB", "ZBAOABZ"
        samples, traces = run_sampler(step_BAOAB_SGHMC, nsteps, h * b, gamma, alpha, beta, grad_U, m, M, r, s, burnin=burnin)
        print('---- Finished running BAOAB ----')
        samples_z, traces_z = run_sampler(step_ZBAOABZ_SGHMC, nsteps, h, gamma, alpha, beta, grad_U, m, M, r, s, burnin=burnin)
        print('---- Finished running ZBAOABZ ----')

    ESS = ess(traces[:, 0])
    ESS_z = ess(traces_z[:, 0])

    # ----- labels for coloring -----
    lab, _ = gmm_labels(samples[:], pis, mus, Sigmas)   # note your samples are (y,x)
    lab_z, _= gmm_labels(samples_z[:], pis, mus, Sigmas)

    col  = labels_to_colors(lab, len(pis))
    col_z = labels_to_colors(lab_z, len(pis))

    # --- Plotting setup ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.5)

    all_y = np.concatenate([samples[:, 0], samples_z[:, 0]])
    all_x = np.concatenate([samples[:, 1], samples_z[:, 1]])
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    max_range = max(x_range, y_range) / 2.0
    mid_x = (all_x.max() + all_x.min()) / 2.0
    mid_y = (all_y.max() + all_y.min()) / 2.0
    # xlim = (mid_x - max_range, mid_x + max_range)
    # ylim = (mid_y - max_range, mid_y + max_range)
    xlim = (-10, 10)
    ylim = (-10, 10)

    # --- Row 0: Contours + colored scatter ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
    idx = slice(None, None, plot_stride)
    ax0.scatter(samples[idx, 0], samples[idx, 1], s=1, c=col[idx], alpha=0.7)
    ax0.set_title(f'{title} (h={h})')
    ax0.set_xlabel('x'); ax0.set_ylabel('y')
    ax0.set_xlim(xlim); ax0.set_ylim(ylim); ax0.set_aspect('equal', 'box')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
    ax1.scatter(samples_z[idx, 0], samples_z[idx, 1], s=1, c=col_z[idx], alpha=0.7)
    ax1.set_title(f'{title_z} (h={h}, gamma={gamma}, alpha={alpha})')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_aspect('equal', 'box')

    # --- Row 1: Step size traces, coloured by mode (scatter vs step index) ---
    ax_left = fig.add_subplot(gs[1, 0])
    steps = np.arange(traces.shape[0])[idx]
    ax_left.scatter(steps, traces[idx, 4], s=2, c=col[idx], alpha=0.7)
    ax_left.set_title(f"{title} trace: dt (colored by inferred mode)")
    ax_left.set_xlabel("Step"); ax_left.set_ylabel("dt")

    ax_right = fig.add_subplot(gs[1, 1])
    steps_z = np.arange(traces_z.shape[0])[idx]
    ax_right.scatter(steps_z, traces_z[idx, 4], s=2, c=col_z[idx], alpha=0.7)
    ax_right.set_title(f"{title_z} trace: dt (colored by inferred mode)")
    ax_right.set_xlabel("Step"); ax_right.set_ylabel("dt")

    # --- Row 2: configurational temperature ---
    T_conf_mean = np.mean(traces[:, 5])
    T_conf_mean_z = np.mean(traces_z[:,5])
    ax_left = fig.add_subplot(gs[2, 0])
    ax_left.plot(traces[idx, 5], lw=0.7, label="T_conf")
    ax_left.hlines(T_conf_mean, 0, len(traces[idx, 5]), color="red", lw=1.5, linestyle=":", label=f"T_conf_mean={T_conf_mean:.3f}")
    ax_left.set_title(f"{title} trace: Configurational T")
    ax_left.set_xlabel("Step")
    ax_left.legend()

    ax_right = fig.add_subplot(gs[2, 1])
    ax_right.plot(traces_z[idx, 5], lw=0.7, label="T_conf")
    ax_right.hlines(T_conf_mean_z, 0, len(traces_z[idx, 5]), color="red", lw=1.5, linestyle=":", label=f"T_conf_mean={T_conf_mean_z:.3f}")
    ax_right.set_title(f"{title_z} trace: Configurational T")
    ax_right.set_xlabel("Step")
    ax_right.legend()

    # --- Row 3: metrics comparison ---
    ax_m = fig.add_subplot(gs[3, :])

    x = np.arange(6)
    labels = ["ESS", "KL", "MMD", "ESS(Z)", "KL(Z)", "MMD(Z)"]

    # left axis: ESS bars
    left_mask = np.array([1, 0, 0, 1, 0, 0], dtype=bool)
    ess_vals = np.array([ESS, 0, 0, ESS_z, 0, 0], dtype=float)
    b1 = ax_m.bar(x[left_mask], ess_vals[left_mask], color="tab:blue", alpha=0.7, label="ESS")

    # right axis: KL/MMD bars
    ax2 = ax_m.twinx()
    right_mask = ~left_mask
    km_vals = np.array([0, kl, mmd2, 0, kl_z, mmd2_z], dtype=float)
    b2 = ax2.bar(x[right_mask], km_vals[right_mask], color="tab:green", alpha=0.7, label="KL/MMD")
    ax2.set_yscale("log")  # KL/MMD often vary over orders of magnitude

    ax_m.set_xticks(x);
    ax_m.set_xticklabels(labels)
    ax_m.set_title("Metrics")
    ax_m.set_ylabel("ESS")
    ax2.set_ylabel("KL / MMD")

    # one legend for both
    bars = [b1, b2]
    labs = ["ESS", "KL/MMD"]
    ax_m.legend(bars, labs, loc="upper right")

    plt.savefig(f'./toy/{saveto}')
    plt.show()

def contour(log_p, xlim=None, ylim=None):
    # Plot settings
    if ylim is None:
        ylim = [-10, 10]
    if xlim is None:
        xlim = [-10, 10]
    x_min, x_max = xlim
    y_min, y_max = ylim
    n = 500  # grid resolution
    n_levels = 100  # number of contour levels

    # Build grid
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    X, Y = np.meshgrid(xs, ys)

    # If log_p is not vectorized (i.e., only takes scalars), uncomment this line:
    # log_p_vec = np.vectorize(log_p, otypes=[float])
    # Z = log_p_vec(X, Y)

    # If it already works with numpy arrays:
    Z = log_p(np.stack([X, Y], axis=-1))

    # Optional: clip very low values for clearer contours (helpful for log-densities)
    # vmin = np.percentile(Z, 5)
    # Z = np.clip(Z, vmin, None)

    # 3D contour
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.contour3D(X, Y, Z, levels=n_levels)  # no explicit colors per your defaults
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("log_p(x, y)")
    ax.set_title("3D Contour of log_p(x, y)")

    plt.tight_layout()
    plt.show()

def plot_grid_samples(
    X, Y, LOGZ, levels,
    grad_U,
    pis, mus, Sigmas,                # for coloring by GMM mode
    param_grid,                      # dict: {"alpha":[...], "h":[...], ...}
    order=1,                         # 1 -> ZSGLD (step_ZOLD), else -> ZBAOABZ (step_ZBAOABZ_SGHMC)
    nsteps=4000, burnin=500,         # keep light for grids; tune as you like
    plot_stride=3,                   # thin points for clarity
    xlim=(-10,10), ylim=(-10,10),
    ncols=4, figsize=(14, 3.5),      # auto rows; tweak figsize per row
    saveto=None
):
    """
    Runs a grid over params and plots ONLY samples_z (colored by inferred mode) on contours.
    """
    # build list of param combos (keep order stable for titles)
    keys = ["alpha", "h", "beta", "m", "M", "r", "s"]
    for k in keys:
        if k not in param_grid: raise ValueError(f"Missing param '{k}' in param_grid")
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    nplots = len(combos)

    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    # choose Z-step
    if order == 1:
        step_fn = step_ZOLD
    else:
        step_fn = step_ZBAOABZ_SGHMC

    # helper: label colors
    def _labels_to_colors(labels, K):
        cmap = plt.get_cmap('tab10', K)
        return cmap(labels % K)

    # iterate combos and plot
    for idx, combo in enumerate(combos):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        alpha, h, beta, m, M, r, s = combo

        # run only Z sampler
        samples_z, _traces_z = run_sampler(
            step_fn,
            nsteps=nsteps,
            h=h, gamma=0.0, alpha=alpha, beta=beta,  # gamma unused in first-order, harmless otherwise
            grad_U=grad_U, m=m, M=M, r=r, s=s,
            burnin=burnin
        )

        # color by GMM component
        lab_z, _ = gmm_labels(samples_z, pis, mus, Sigmas)
        col_z = _labels_to_colors(lab_z, len(pis))

        # draw contours + samples
        ax.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
        sl = slice(None, None, plot_stride)
        ax.scatter(samples_z[sl, 0], samples_z[sl, 1], s=4, c=col_z[sl], alpha=0.75)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal', 'box')

        # title shows params
        ax.set_title(f"α:{alpha},h:{h},β:{beta},m:{m},M:{M},r:{r},s:{s}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # clean empty axes if grid not full
    for k in range(nplots, nrows*ncols):
        fig.delaxes(axes.flatten()[k])

    plt.tight_layout()
    if saveto:
        plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.show()
