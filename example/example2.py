"""Plot a hysteresis loop with three negative-field zero crossings and its sectors."""
 
from polygon_hysteresis_modeller import generate_polygon, plot_hysteresis, is_regular

import mammos_analysis as ma
import mammos_entity as me
import numpy as np

Xc = [-0.3, 0.0, 10000]  # coercivity anchor
Yr = [0.0, 0.7, 10000]   # remanence anchor
x, y = generate_polygon(
    points_per_branch=101,
    smoothness_weights=[1.0, 1e-3],
    saturation_bounds=[0.8, 0.8],
    anchorpoints=[
        [-0.60, 0.50, 10000],
        Xc,
        [-0.16, -0.40, 10000],
        Yr,
    ],
    apply_sine=False, 
    apply_mirror=False,
)



plot_hysteresis(x, y, output="hysteresis2.png")

analysis = ma.hysteresis.extrinsic_properties(
            me.Entity("MagneticFieldStrength", x), 
            me.Entity("SaturationMagnetisation", y), 
            demagnetization_coefficient=1/3
           )

print(analysis.Mr)
print(analysis.Hc)
print(analysis.BHmax)
