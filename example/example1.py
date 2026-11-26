"""Save an example plot with ``pixi run demo``."""

from polygon_hysteresis_modeller import generate_polygon, plot_hysteresis

import mammos_analysis as ma
import mammos_entity as me

xc = -0.25  # coercivity control parameter
yr = 0.5    # remanence control parameter
sat_r = 0.4 # saturation width right
sat_l = 0.7 # saturation width left
N_points = 51 # number of datapoints on the upper branch
x, y = generate_polygon(
    N=N_points,
    xc=xc,
    yr=yr,
    A=1.0,
    B=0.01,
    Kc=10000,
    Kr=10000,
    Km=1000,
    left_saturation_width=sat_l,
    right_saturation_width=sat_r,
)

plot_hysteresis(x, y, xc, yr, output="hysteresis.png")


