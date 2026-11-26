"""Generate normalized polygonal hysteresis curves."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, diags, vstack
from scipy.sparse.linalg import spsolve


def generate_polygon(
    N=21,
    xc=-0.4,
    yr=0.5,
    A=1.0,
    B=0.1,
    Kc=1000.0,
    Kr=1000.0,
    Km=1000.0,
    flat_edges=True,
    left_saturation_width=1.0,
    right_saturation_width=1.0,
):
    """Generate a synthetic hysteresis branch as a polygonal chain.

    The branch consists of N support points on [-1, 1].
    The magnetization values are determined by minimizing
    a discrete energy functional consisting of slope,
    curvature, coercivity, remanence, and monotonicity terms.

    The continuous counterpart of the energy is

        E = A/2 * integral(f'**2, dx)
          + B/2 * integral(f''**2, dx)
          + Kc/2 * f(xc)**2
          + Kr/2 * (f(0) - yr)**2
          + Km/2 * integral(min(0, f')**2, dx).

    The slope and curvature contributions are discretized
    using differences between adjacent support points.

    The endpoint and saturation conditions are imposed
    exactly:

        f(x) = -1 for -1 <= x <= -left_saturation_width
        f(x) =  1 for right_saturation_width <= x <= 1

    Only the magnetization values outside the prescribed
    saturation regions are optimized.

    Parameters
    ----------
    N : int, optional
        Total number of support points, including endpoints.
    xc : float, optional
        Nominal signed coercive field.
    yr : float, optional
        Nominal normalized remanent magnetization.
    A : float, optional
        Nonnegative slope penalty coefficient.
    B : float, optional
        Positive discrete curvature penalty coefficient.
    Kc : float, optional
        Coercivity alignment coefficient.
    Kr : float, optional
        Remanence alignment coefficient.
    Km : float, optional
        Monotonicity penalty coefficient.
    flat_edges : bool, optional
        If True, additionally enforce horizontal first
        and last polygon segments.
    left_saturation_width : float, optional
        Absolute field coordinate at which left saturation
        begins. For example, 0.4 enforces f(x) = -1
        throughout [-1, -0.4].
    right_saturation_width : float, optional
        Field coordinate at which right saturation begins.
        For example, 0.4 enforces f(x) = 1
        throughout [0.4, 1].

    Returns
    -------
    x : numpy.ndarray
        Field coordinates, including the saturation boundaries.
    y : numpy.ndarray
        Optimized magnetization values.

    Notes
    -----
    The saturation conditions are hard constraints.

    Coercivity, remanence, and monotonicity are soft
    constraints controlled by penalty coefficients.

    The grid is generally nonuniform because the saturation
    boundaries are explicitly included as support points.
    The discrete energy accounts for nonuniform spacing.

    Boundedness outside the fixed saturation regions is
    not explicitly enforced.

    A point-symmetric descending branch is obtained by

        x_desc = -x[::-1]
        y_desc = -y[::-1].
    """

    if N < 5:
        raise ValueError("Require N >= 5.")

    if not -1 < xc < 0:
        raise ValueError("Require -1 < xc < 0.")

    if not 0 < left_saturation_width <= 1:
        raise ValueError("Invalid left saturation width.")

    if not 0 < right_saturation_width <= 1:
        raise ValueError("Invalid right saturation width.")

    if B <= 0 or min(A, Kc, Kr, Km) < 0:
        raise ValueError("Invalid energy coefficients.")

    # Saturation boundaries
    xl = -left_saturation_width
    xr = right_saturation_width

    if xc <= xl:
        raise ValueError(
            "Coercive point lies inside the left saturation region."
        )

    # Construct a grid containing both saturation boundaries
    x = np.linspace(-1, 1, N)

    used = {0, N - 1}

    for boundary in (xl, xr):

        if boundary in (-1.0, 1.0):
            continue

        candidates = [
            i for i in range(1, N - 1)
            if i not in used
        ]

        j = min(candidates, key=lambda i: abs(x[i] - boundary))

        x[j] = boundary
        used.add(j)

    x = np.sort(x)

    if np.any(np.diff(x) <= 0):
        raise ValueError("Degenerate grid.")

    # Local segment lengths
    h = np.diff(x)

    # First derivative operator
    D1 = diags(
        [-1/h, 1/h],
        [0, 1],
        shape=(N - 1, N),
        format="csr",
    )

    # Second derivative operator on a nonuniform grid
    hc = (h[:-1] + h[1:]) / 2

    D2 = diags(
        [
            1 / (hc * h[:-1]),
            -(1/h[:-1] + 1/h[1:]) / hc,
            1 / (hc * h[1:]),
        ],
        [0, 1, 2],
        shape=(N - 2, N),
        format="csr",
    )

    def interpolation_row(xp):

        j = np.searchsorted(x, xp) - 1
        j = np.clip(j, 0, N - 2)

        t = (xp - x[j]) / h[j]

        return csr_matrix(
            ([1 - t, t], ([0, 0], [j, j + 1])),
            shape=(1, N),
        )

    # Control point operators
    Pc = interpolation_row(xc)
    Pr = interpolation_row(0.0)

    # Quadratic energy residual operator
    L = vstack([
        diags(np.sqrt(A*h)) @ D1,
        diags(np.sqrt(B*hc)) @ D2,
        np.sqrt(Kc) * Pc,
        np.sqrt(Kr) * Pr,
    ]).tocsr()

    target = np.zeros(L.shape[0])
    target[-1] = np.sqrt(Kr) * yr

    # Hard saturation constraints
    left = x <= xl
    right = x >= xr

    if flat_edges:
        left[:2] = True
        right[-2:] = True

    if np.any(left & right):
        raise ValueError("Overlapping saturation constraints.")

    y = np.zeros(N)

    y[left] = -1
    y[right] = 1

    # Free optimization variables
    free = np.flatnonzero(~(left | right))

    if len(free) == 0:
        raise ValueError("No free support points remain.")

    L_free = L[:, free]
    D_free = D1[:, free]

    offset = L @ y - target
    d_offset = D1 @ y

    # Initial solution without monotonicity penalty
    z0 = spsolve(
        L_free.T @ L_free,
        -L_free.T @ offset,
    )

    weight = np.sqrt(Km * h)

    def residual(z):

        dy = D_free @ z + d_offset

        return np.concatenate([
            L_free @ z + offset,
            weight * np.minimum(dy, 0),
        ])

    def jacobian(z):

        dy = D_free @ z + d_offset

        return vstack([
            L_free,
            diags(weight * (dy < 0)) @ D_free,
        ]).tocsr()

    result = least_squares(
        residual,
        z0,
        jac=jacobian,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    if not result.success:
        raise RuntimeError(result.message)

    y[free] = result.x

    return x, y


def generate_loop(*args, **kwargs):
    """Return ``((x_up, y_up), (x_down, y_down))`` for a point-symmetric loop.

    Pass the same arguments as to :func:`generate_polygon`. Field coordinates
    increase along the ascending branch and decrease along the descending branch.
    """
    x, y = generate_polygon(*args, **kwargs)
    return (x, y), (-x, -y)

def mirror_polygon(x: np.ndarray, y: np.ndarray):
    return np.concatenate((x, x[::-1])), np.concatenate((y, -y))