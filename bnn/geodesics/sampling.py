from geodesic_quantities import geodesic_monge
import models
import numpy as np
from plotting_function import plot_contour


def get_points(xs):
    return np.stack([np.cos(np.pi * 2 * xs), np.sin(np.pi * 2 * xs)]).T


def get_contour_plots(
    M,
    name,
    hat_x,
    xlim,
    ylim,
    alpha_2=1.0,
    figname=None,
):
    num_points = 500
    xs = np.linspace(0.0, 1.0, num=num_points)
    points = get_points(xs)

    g_num_points = 9
    g_xs = np.linspace(0.0, 1.0, g_num_points)
    g_points = get_points(g_xs)

    geodesics = []

    # can be adjusted
    multiplier = 10.0

    # get the contour
    new_xs = []
    for num_point in range(num_points):
        new_x = geodesic_monge(
            M=M,
            x=hat_x,
            v=points[num_point] * multiplier,
            alpha_2=alpha_2,
        )["y"][:2, -1]
        new_xs.append(new_x)
    contour = np.stack(new_xs)

    # get the individual geodesic lines
    for g_num_point in range(g_num_points):
        geodesics.append(
            geodesic_monge(
                M=M, x=hat_x, v=g_points[g_num_point] * multiplier, alpha_2=alpha_2
            )["y"][:2, :]
        )

    plot_contour(
        hat_x,
        contour,
        M,
        xlim=xlim,
        ylim=ylim,
        geodesics=geodesics,
        file_name=f"figs/g_contours_{name}_{figname}.png",
    )


if __name__ == "__main__":
    M = models.funnel(sig2v=80)
    name = "funnel"
    xlim = [-5.0, 9.0]
    ylim = [-25.0, 25.0]

    get_contour_plots(
        M=M,
        name=name,
        hat_x=np.array([2.0, -4.0]),
        xlim=xlim,
        ylim=ylim,
        figname="1",
    )
    get_contour_plots(
        M=M,
        name=name,
        hat_x=np.array([5.0, 10.0]),
        xlim=xlim,
        ylim=ylim,
        figname="2",
    )
