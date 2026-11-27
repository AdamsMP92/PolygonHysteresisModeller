"""Plot point-symmetric hysteresis loops."""

import numpy as np
import matplotlib.pyplot as plt

def plot_hysteresis(x, y, *, output=None):
    """Return a Matplotlib figure for a point-symmetric hysteresis loop.

    Pass the ascending branch as ``x`` and ``y``. The markers show the nominal
    coercive field ``xc`` and remanence ``yr`` on both branches. Pass an output
    path to save the figure at 150 dpi.
    """
    from matplotlib.figure import Figure

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    figure = Figure(figsize=(6.4, 4.8), layout="tight")
    axes = figure.subplots()
    axes.plot(x, y, "o-")
    axes.axhline(0, color="gray", linewidth=0.5)
    axes.axvline(0, color="gray", linewidth=0.5)
    axes.set(xlabel="x", ylabel="y")
    axes.grid()

    if output is not None:
        figure.savefig(output, dpi=150)

    return figure


def plot_curve(x, y, saturation_bounds, x_p=None, y_p=None):
    if (x_p is None) != (y_p is None):
        raise ValueError("x_p and y_p must be provided together.")

    if len(saturation_bounds) != 2:
        raise ValueError("saturation_bounds must contain two values.")

    fig, ax = plt.subplots()

    ax.plot(x, y, marker="o", color="blue", zorder=3)

    if x_p is not None:
        ax.plot(x_p, y_p, marker="o", color="red",
            linestyle="none", zorder=4)
    
    ax.axvspan(
        -saturation_bounds[0],
        saturation_bounds[1],
        color="gray",
        alpha=0.2,
        zorder=0,
    )
    
    ax.axhline(0, color="black", linewidth=1, zorder=2)
    ax.axvline(0, color="black", linewidth=1, zorder=2)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_xlabel(r"$\mu_0 M$")
    ax.set_ylabel(r"$\mu_0 H$")

    plt.show()
