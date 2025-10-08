import numpy as np
import matplotlib.pyplot as plt


def sub_sample(samples, max_samples=1000):
    n_rows, n_cols = samples.shape
    if n_rows <= max_samples:
        return samples
    if n_rows > max_samples:
        id_samples = np.random.choice(n_rows, size=max_samples, replace=False)
        return samples[id_samples]


def plot_contour(
    hat_x,
    contour,
    M,
    xlim,
    ylim,
    geodesics,
    file_name="figs/contours.png",
    xaxes=[r"$\boldsymbol{\theta}_{1}$", r"$\boldsymbol{\theta}_{2}$"],
):

    plt.rcParams["font.size"] = 22
    plt.rcParams["text.usetex"] = True
    # https://stackoverflow.com/a/14324826, in order to use \boldsymbol
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    x0 = np.arange(xlim[0], xlim[1], 0.05)
    x1 = np.arange(ylim[0], ylim[1], 0.05)
    X, Y = np.meshgrid(x0, x1)
    g = lambda x, y: M.logp([x, y])
    g = np.vectorize(g)
    Z = g(X, Y)

    plt.figure(figsize=(6, 6))

    plt.plot(hat_x[0], hat_x[1], color="darkred", marker="o", markersize=10, zorder=5)
    plt.contour(
        X,
        Y,
        Z,
        [-10.0, -7.5, -5.0, -2.5],
    )

    plt.xlabel(xaxes[0], fontsize=30)
    plt.ylabel(xaxes[1], fontsize=30)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    plt.plot(
        contour[:, 0],
        contour[:, 1],
        color="maroon",
        linewidth=3.0,
    )

    for geodesic in geodesics:
        plt.plot(geodesic[0, :], geodesic[1, :], color="red")
        plt.arrow(
            geodesic[0, -2],
            geodesic[1, -2],
            geodesic[0, -1] - geodesic[0, -2],
            geodesic[1, -1] - geodesic[1, -2],
            length_includes_head=True,
            head_width=0.5,
            head_length=1.0,
            zorder=5,
            color="red",
        )

    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    print("Plot saved")
