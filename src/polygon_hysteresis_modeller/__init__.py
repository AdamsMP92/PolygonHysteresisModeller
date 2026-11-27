"""Public API for generating and plotting polygonal hysteresis curves."""

from ._generator import generate_loop, generate_polygon, mirror_polygon
from ._plot import plot_hysteresis, plot_curve
from ._analysis import is_regular

__all__ = ["generate_loop", "generate_polygon", "mirror_polygon", "plot_hysteresis", "plot_curve", "is_regular"]
