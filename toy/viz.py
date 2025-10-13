import numpy as np
import matplotlib.pyplot as plt

from toy.samplers import step_OLD, step_ZOLD
from utils import ess, run_sampler
from samplers import step_BAOAB_SGHMC, step_ZBAOABZ_SGHMC

def plot_samplers(alpha, h, gamma, beta, grad_U,
                  X, Y, LOGZ, levels,
                  m, M, r, s, b, nsteps, burnin,
                  record_trace=True, plot_stride=10):

    samples_baoab, traces_baoab = run_sampler(
        step_BAOAB_SGHMC, nsteps, h * b, gamma, alpha, beta,
        grad_U, m, M, r, s, burnin=burnin, record_trace=record_trace)

    print('---- Finished running BAOAB ----')

    samples_zbaoabz, traces_zbaoabz = run_sampler(
        step_ZBAOABZ_SGHMC, nsteps, h, gamma, alpha, beta,
        grad_U, m, M, r, s, burnin=burnin, record_trace=record_trace)

    print('---- Finished running ZBAOABZ ----')

    # --- Plotting setup ---
    fig = plt.figure(figsize=(12, 24))
    gs = fig.add_gridspec(8, 2, height_ratios=[2, 1, 1, 1, 1, 1, 1, 1], hspace=0.5)

    # --- Determine shared equal scaling ---
    all_y = np.concatenate([samples_baoab[:, 0], samples_zbaoabz[:, 0]])
    all_x = np.concatenate([samples_baoab[:, 1], samples_zbaoabz[:, 1]])

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    max_range = max(x_range, y_range) / 2.0

    mid_x = (all_x.max() + all_x.min()) / 2.0
    mid_y = (all_y.max() + all_y.min()) / 2.0

    xlim = (mid_x - max_range, mid_x + max_range)
    ylim = (mid_y - max_range, mid_y + max_range)

    # --- Row 0: Contours + scatter (no lines, transparency) ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
    ax0.scatter(samples_baoab[::plot_stride, 1], samples_baoab[::plot_stride, 0],
                s=5, color='red', alpha=0.5)
    ax0.set_title(f'BAOAB (h={h}, γ={gamma}, α={alpha})')
    ax0.set_xlabel('x'); ax0.set_ylabel('y')
    ax0.set_xlim(xlim); ax0.set_ylim(ylim); ax0.set_aspect('equal', 'box')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
    ax1.scatter(samples_zbaoabz[::plot_stride, 1], samples_zbaoabz[::plot_stride, 0],
                s=5, color='red', alpha=0.5)
    ax1.set_title(f'ZBAOABZ (h={h}, γ={gamma}, α={alpha})')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_aspect('equal', 'box')

    # === If record_trace is False, stop here ===
    if not record_trace:
        plt.tight_layout()
        plt.show()
        return

    # --- Compute metrics (only if traces exist) ---
    # ess_baoab = ess(traces_baoab[:,0])
    # ess_zbaoabz = ess(traces_zbaoabz[:,0])

    # T_kin_baoab = np.mean(np.sum(traces_baoab[:,2:4]**2, axis=1)) # This should be by degrees of freedom.
    # T_kin_zbaoabz = np.mean(np.sum(traces_zbaoabz[:,2:4]**2, axis=1))

    # T_conf_mean_baoab = np.mean(traces_baoab[:,5])
    # T_conf_mean_zbaoabz = np.mean(traces_zbaoabz[:,5])

    # y_avg_baoab = np.cumsum(traces_baoab[:,0]) / (np.arange(len(traces_baoab)) + 1)
    # y_avg_zbaoabz = np.cumsum(traces_zbaoabz[:,0]) / (np.arange(len(traces_zbaoabz)) + 1)

    # --- Remaining plots (only if record_trace=True) ---
    # ax_left = fig.add_subplot(gs[1, 0])
    # ax_left.plot(traces_baoab[:,0], lw=0.7, label="y")
    # ax_left.plot(traces_baoab[:,1], lw=0.7, label="x")
    # ax_left.set_title("BAOAB trace: positions"); ax_left.set_xlabel("Step"); ax_left.legend()

    # ax_right = fig.add_subplot(gs[1, 1])
    # ax_right.plot(traces_zbaoabz[:,0], lw=0.7, label="y")
    # ax_right.plot(traces_zbaoabz[:,1], lw=0.7, label="x")
    # ax_right.set_title("ZBAOABZ trace: positions"); ax_right.set_xlabel("Step"); ax_right.legend()

    # # --- Row 2: Momentum traces ---
    # ax_left = fig.add_subplot(gs[2, 0])
    # ax_left.plot(traces_baoab[::plot_stride, 2], lw=0.7, label="p_y")
    # ax_left.plot(traces_baoab[::plot_stride, 3], lw=0.7, label="p_x")
    # ax_left.set_title("BAOAB trace: momenta")
    # ax_left.set_xlabel("Step")
    # ax_left.legend()

    # ax_right = fig.add_subplot(gs[2, 1])
    # ax_right.plot(traces_zbaoabz[::plot_stride, 2], lw=0.7, label="p_y")
    # ax_right.plot(traces_zbaoabz[::plot_stride, 3], lw=0.7, label="p_x")
    # ax_right.set_title("ZBAOABZ trace: momenta")
    # ax_right.set_xlabel("Step")
    # ax_right.legend()

    # --- Row 3: Step size traces ---
    ax_left = fig.add_subplot(gs[3, 0])
    ax_left.plot(traces_baoab[::plot_stride, 4], lw=0.7)
    ax_left.set_title("BAOAB trace: dt (step size)")
    ax_left.set_xlabel("Step")

    ax_right = fig.add_subplot(gs[3, 1])
    ax_right.plot(traces_zbaoabz[::plot_stride, 4], lw=0.7)
    ax_right.set_title("ZBAOABZ trace: dt (step size)")
    ax_right.set_xlabel("Step")

    # # --- Row 4: Configurational vs kinetic temperature ---
    # ax_left = fig.add_subplot(gs[4, 0])
    # ax_left.plot(traces_baoab[::plot_stride, 5], lw=0.7, label="T_conf")
    # ax_left.hlines(T_kin_baoab, 0, len(traces_baoab), color="orange", lw=1.5, linestyle="--", label=f"T_kin={T_kin_baoab:.3f}")
    # ax_left.hlines(T_conf_mean_baoab, 0, len(traces_baoab), color="red", lw=1.5, linestyle=":", label=f"T_conf_mean={T_conf_mean_baoab:.3f}")
    # ax_left.set_title("BAOAB trace: Configurational vs Kinetic T")
    # ax_left.set_xlabel("Step")
    # ax_left.legend()

    # ax_right = fig.add_subplot(gs[4, 1])
    # ax_right.plot(traces_zbaoabz[::plot_stride, 5], lw=0.7, label="T_conf")
    # ax_right.hlines(T_kin_zbaoabz, 0, len(traces_zbaoabz), color="orange", lw=1.5, linestyle="--", label=f"T_kin={T_kin_zbaoabz:.3f}")
    # ax_right.hlines(T_conf_mean_zbaoabz, 0, len(traces_zbaoabz), color="red", lw=1.5, linestyle=":", label=f"T_conf_mean={T_conf_mean_zbaoabz:.3f}")
    # ax_right.set_title("ZBAOABZ trace: Configurational vs Kinetic T")
    # ax_right.set_xlabel("Step")
    # ax_right.legend()

    # # --- Row 5: ESS comparison ---
    # ax_ess = fig.add_subplot(gs[5, :])
    # ax_ess.bar(["BAOAB", "ZBAOABZ"], [ess_baoab, ess_zbaoabz], color=["blue", "green"], alpha=0.7)
    # ax_ess.set_title("Effective Sample Size (ESS) for y")
    # ax_ess.set_ylabel("ESS")

    # # --- Row 6: Histogram of T_conf vs T_kin ---
    # ax_hist = fig.add_subplot(gs[6, :])
    # ax_hist.hist(traces_baoab[:, 5], bins=50, alpha=0.5, label="BAOAB T_conf")
    # ax_hist.hist(traces_zbaoabz[:, 5], bins=50, alpha=0.5, label="ZBAOABZ T_conf")
    # ax_hist.axvline(T_kin_baoab, color="blue", linestyle="--", label="BAOAB T_kin")
    # ax_hist.axvline(T_kin_zbaoabz, color="green", linestyle="--", label="ZBAOABZ T_kin")
    # ax_hist.set_title("Histogram of Configurational vs Kinetic Temperature")
    # ax_hist.set_xlabel("Temperature")
    # ax_hist.set_ylabel("Frequency")
    # ax_hist.legend()

    # # --- Row 7: Running averages of y and x ---
    # y_avg_baoab = np.cumsum(traces_baoab[:, 0]) / (np.arange(len(traces_baoab)) + 1)
    # x_avg_baoab = np.cumsum(traces_baoab[:, 1]) / (np.arange(len(traces_baoab)) + 1)
    # y_avg_zbaoabz = np.cumsum(traces_zbaoabz[:, 0]) / (np.arange(len(traces_zbaoabz)) + 1)
    # x_avg_zbaoabz = np.cumsum(traces_zbaoabz[:, 1]) / (np.arange(len(traces_zbaoabz)) + 1)

    # ax_left = fig.add_subplot(gs[7, 0])
    # ax_left.plot(y_avg_baoab[::plot_stride], lw=0.7, color="purple", label="y_avg")
    # ax_left.plot(x_avg_baoab[::plot_stride], lw=0.7, color="green", label="x_avg")
    # ax_left.set_title("BAOAB: Running averages of y and x")
    # ax_left.set_xlabel("Step")
    # ax_left.set_ylabel("Average")
    # ax_left.legend()

    # ax_right = fig.add_subplot(gs[7, 1])
    # ax_right.plot(y_avg_zbaoabz[::plot_stride], lw=0.7, color="purple", label="y_avg")
    # ax_right.plot(x_avg_zbaoabz[::plot_stride], lw=0.7, color="green", label="x_avg")
    # ax_right.set_title("ZBAOABZ: Running averages of y and x")
    # ax_right.set_xlabel("Step")
    # ax_right.set_ylabel("Average")
    # ax_right.legend()

    plt.show()


def plot_samplers_first_order(alpha, h, gamma, beta, grad_U,
                              X, Y, LOGZ, levels,
                              m, M, r, s, b, nsteps, burnin,
                              record_trace=True, plot_stride=10):
    samples_em, traces_em = run_sampler(
        step_OLD, nsteps, h * b, gamma, alpha, beta,
        grad_U, m, M, r, s, burnin=burnin, record_trace=record_trace)

    print('---- Finished running BAOAB ----')

    samples_zem, traces_zem = run_sampler(
        step_ZOLD, nsteps, h, gamma, alpha, beta,
        grad_U, m, M, r, s, burnin=burnin, record_trace=record_trace)

    print('---- Finished running ZBAOABZ ----')

    # --- Plotting setup ---
    fig = plt.figure(figsize=(12, 24))
    gs = fig.add_gridspec(8, 2, height_ratios=[2, 1, 1, 1, 1, 1, 1, 1], hspace=0.5)

    # --- Determine shared equal scaling ---
    all_y = np.concatenate([samples_em[:, 0], samples_zem[:, 0]])
    all_x = np.concatenate([samples_em[:, 1], samples_zem[:, 1]])

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    max_range = max(x_range, y_range) / 2.0

    mid_x = (all_x.max() + all_x.min()) / 2.0
    mid_y = (all_y.max() + all_y.min()) / 2.0

    xlim = (mid_x - max_range, mid_x + max_range)
    ylim = (mid_y - max_range, mid_y + max_range)

    # --- Row 0: Contours + scatter (no lines, transparency) ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
    ax0.scatter(samples_em[::plot_stride, 1], samples_em[::plot_stride, 0],
                s=5, color='red', alpha=0.5)
    ax0.set_title(f'BAOAB (h={h}, γ={gamma}, α={alpha})')
    ax0.set_xlabel('x'); ax0.set_ylabel('y')
    ax0.set_xlim(xlim); ax0.set_ylim(ylim); ax0.set_aspect('equal', 'box')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.contourf(X, Y, LOGZ, levels=levels, cmap='viridis')
    ax1.scatter(samples_zem[::plot_stride, 1], samples_zem[::plot_stride, 0],
                s=5, color='red', alpha=0.5)
    ax1.set_title(f'ZBAOABZ (h={h}, γ={gamma}, α={alpha})')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_aspect('equal', 'box')

    # === If record_trace is False, stop here ===
    if not record_trace:
        plt.tight_layout()
        plt.show()
        return

    # --- Row 3: Step size traces ---
    ax_left = fig.add_subplot(gs[3, 0])
    ax_left.plot(traces_em[::plot_stride, 4], lw=0.7)
    ax_left.set_title("BAOAB trace: dt (step size)")
    ax_left.set_xlabel("Step")

    ax_right = fig.add_subplot(gs[3, 1])
    ax_right.plot(traces_zem[::plot_stride, 4], lw=0.7)
    ax_right.set_title("ZBAOABZ trace: dt (step size)")
    ax_right.set_xlabel("Step")

    plt.show()
