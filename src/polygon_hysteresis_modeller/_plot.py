"""Plot point-symmetric hysteresis loops."""

import numpy as np


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
    axes.legend()
    axes.grid()

    if output is not None:
        figure.savefig(output, dpi=150)

    return figure
