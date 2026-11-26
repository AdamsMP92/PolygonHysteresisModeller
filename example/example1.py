"""Save an example plot with ``pixi run demo``."""

from polygon_hysteresis_modeller import generate_polygon, plot_hysteresis, mirror_polygon

import mammos_analysis as ma
import mammos_entity as me
import numpy as np

Xc = [-0.5, 0.0, 10000]  # coercivity anchor
Yr = [0.0, 0.8, 10000]   # remanence anchor
x, y = generate_polygon(
    N=51,
    slope_penalty=1.0,
    curvature_penalty=0.01,
    left_saturation_width=0.7,
    right_saturation_width=0.4,
    anchorpoints=[Xc, Yr],
    sine_transform=True, 
    mirror=True,
)

plot_hysteresis(x, y, output="hysteresis.png")

H = me.Entity("MagneticFieldStrength", x)
M = me.Entity("SaturationMagnetisation", y)
print(ma.hysteresis.extrinsic_properties(H, M, demagnetization_coefficient=1/3))
