"""Save an example plot with ``pixi run demo``."""

from polygon_hysteresis_modeller import generate_polygon, plot_hysteresis, is_regular

import mammos_analysis as ma
import mammos_entity as me
import numpy as np

Xc = [-0.3, 0.0, 10000]  # coercivity anchor
Yr = [0.0, 0.8, 10000]   # remanence anchor
x, y = generate_polygon(
    points_per_branch=51,
    smoothness_weights=[1.0, 1e-3],
    saturation_bounds=[0.5, 0.5],
    anchorpoints=[Xc, Yr],
    apply_sine=True, 
    apply_mirror=False,
)

plot_hysteresis(x, y, output="hysteresis1.png")

analysis = ma.hysteresis.extrinsic_properties(
            me.Entity("MagneticFieldStrength", x), 
            me.Entity("SaturationMagnetisation", y), 
            demagnetization_coefficient=1/3
           )

print(analysis.Mr)
print(analysis.Hc)
print(analysis.BHmax)
