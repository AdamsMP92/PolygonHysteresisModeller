"""Plot a loop with three zero crossings at negative fields."""

from polygon_hysteresis_modeller import generate_polygon, plot_hysteresis

Xc = [-0.3, 0.0, 10000]  # coercivity anchor
Yr = [0.0, 0.7, 10000]   # remanence anchor
x, y = generate_polygon(
    N=101,
    slope_penalty=1.0,
    curvature_penalty=1e-3,
    left_saturation_width=0.8,
    right_saturation_width=0.8,
    anchorpoints=[
        [-0.60, 0.50, 10000],
        Xc,
        [-0.16, -0.40, 10000],
        Yr,
    ],
    sine_transform=True, 
    mirror=True,
)

plot_hysteresis(x, y, output="hysteresis_example2.png")
