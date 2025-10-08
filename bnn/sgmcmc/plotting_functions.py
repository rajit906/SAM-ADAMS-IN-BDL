import numpy as np
import matplotlib.pyplot as plt


def sub_sample(samples, max_samples=1000):
    n_rows, n_cols = samples.shape
    if n_rows <= max_samples:
        return samples
    if n_rows > max_samples:
        id_samples = np.random.choice(n_rows, size=max_samples, replace=False)
        return samples[id_samples]


def plot_samples(
    samples,
    M,
    xlim,
    ylim,
    densities=True,
    file_name="figs/samples.png",
    title="samples",
    # Madapt=1000,
    # delta=0.2,
    xaxes=[r"$\boldsymbol{\theta}_{1}$", r"$\boldsymbol{\theta}_{2}$"],
    custom=False,
    file_name_custom="figs/samples.png",
):
    plt.rcParams['text.usetex'] = True
    # https://stackoverflow.com/a/14324826, in order to use \boldsymbol
    plt.rcParams["text.latex.preamble"] = r'\usepackage{amsmath}'
    
    f = lambda x: (M.logp(x), M.dlogp(x))
    if densities:
        density1, density2 = M.densities()

    sub_samples = sub_sample(samples, max_samples=1000)
    # Plot everything together
    x0 = np.arange(xlim[0], xlim[1], 0.1)
    x1 = np.arange(ylim[0], ylim[1], 0.1)
    X, Y = np.meshgrid(x0, x1)
    g = lambda x, y: f([x, y])[0]
    try:
        Z = g(X, Y)
    except:
        g = np.vectorize(g)
        Z = g(X, Y)

    if not custom:
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(2, 3, 1)
        # thresholds found by hand
        ax1.contour(X, Y, Z, [-10.0, -7.5, -5.0, -2.5])
        ax1.set_xlabel(xaxes[0])
        ax1.set_ylabel(xaxes[1])
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.scatter(
            sub_samples[:, 0], sub_samples[:, 1], alpha=0.15, s=20, marker="o", zorder=2
        )

        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(samples[:, 0], bins=30, density=True, label="Hist")
        if density1 is not None:
            ax2.plot(x0, density1(x0), label="Density")
        ax2.set_xlabel(xaxes[0])
        ax2.legend()

        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(samples[:, 1], bins=30, density=True, color="orange", label="Hist")
        if density2 is not None:
            ax3.plot(x1, density2(x1), label="Density")
        ax3.set_xlabel(xaxes[1])
        ax3.legend()

        ax4 = plt.subplot(2, 1, 2)
        ax4.plot(range(samples.shape[0]), samples[:, 0], label=xaxes[0], alpha=0.6)
        ax4.plot(range(samples.shape[0]), samples[:, 1], label=xaxes[1], alpha=0.6)
        ax4.legend()
        plt.suptitle(title, fontsize=30)
        plt.savefig(file_name, dpi=200)

    elif custom == "1":
        plt.figure(figsize=(12, 6))
        plt.rcParams["font.size"] = 22
        # plt.subplots_adjust(wspace=0.35)
        ax1 = plt.subplot(1, 2, 1)
        # thresholds found by hand
        ax1.contour(X, Y, Z, [-10.0, -7.5, -5.0, -2.5])
        ax1.set_xlabel(xaxes[0])
        ax1.set_ylabel(xaxes[1])
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.scatter(
            sub_samples[:, 0], sub_samples[:, 1], alpha=0.15, s=20, marker="o", zorder=2
        )

        ax2 = plt.subplot(1, 2, 2)
        ax2.hist(
            samples[:, 1],
            bins=30,
            density=True,
            color="orange",
            label="Sample estimate",
            orientation="horizontal",
        )
        if density2 is not None:
            ax2.plot(density2(x1), x1, label="True density")
        ax2.set_yticklabels([])
        ax2.legend(loc="lower right", prop={"size": 15})
        plt.subplots_adjust(wspace=0)
        plt.savefig(file_name_custom, dpi=200, bbox_inches="tight")

    if custom == "2":
        plt.rcParams["font.size"] = 22
        fig, axs = plt.subplots(
            1, 2, figsize=(9, 6), gridspec_kw={"width_ratios": [2, 1]}
        )
        axs[0].contour(X, Y, Z, [-10.0, -7.5, -5.0, -2.5])
        axs[0].set_xlabel(xaxes[0], fontsize=30)
        axs[0].set_ylabel(xaxes[1], fontsize=30)
        axs[0].set_xlim(xlim[0], xlim[1])
        axs[0].set_ylim(ylim[0], ylim[1])
        axs[0].scatter(
            sub_samples[:, 0], sub_samples[:, 1], alpha=0.15, s=20, marker="o", zorder=2
        )
        axs[1].hist(
            samples[:, 1],
            bins=30,
            density=True,
            color="orange",
            label="Estimate",
            orientation="horizontal",
        )
        if density2 is not None:
            axs[1].plot(density2(x1), x1, label="True")
        axs[1].set_yticklabels([])
        axs[1].legend(loc="lower right", prop={"size": 15})
        axs[1].set_ylim(ylim[0], ylim[1])
        fig.subplots_adjust(wspace=0.1)
        plt.savefig(file_name_custom, dpi=200, bbox_inches="tight")
    print("Plot saved")
