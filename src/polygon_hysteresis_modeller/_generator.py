"""Generate normalized polygonal hysteresis curves."""

import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from scipy.sparse.linalg import spsolve


def generate_polygon(
    N=21,
    slope_penalty=1.0,
    curvature_penalty=0.1,
    flat_edges=True,
    left_saturation_width=1.0,
    right_saturation_width=1.0,
    anchorpoints=None,
    *,
    sine_transform=False,
    mirror=False,
):
    """Generate a synthetic hysteresis branch as a polygonal chain.

    The branch consists of N equally spaced support points on [-1, 1].
    The magnetization values are determined by minimizing
    a discrete energy functional with slope, curvature, and optional anchor terms.

    The continuous counterpart of the energy is

        E = slope_penalty/2 * integral(f'**2, dx)
          + curvature_penalty/2 * integral(f''**2, dx)
          + 1/2 * sum(K_j * (f(x_anchor_j) - y_anchor_j)**2).

    The slope and curvature contributions are discretized
    using differences between adjacent support points.

    The saturation thresholds fix the magnetization at existing support points.

        y_i = -1 if x_i <= -left_saturation_width
        y_i =  1 if x_i >=  right_saturation_width

    The optimization varies the free magnetization values.

    Parameters
    ----------
    N : int, optional
        Total number of support points, including endpoints.
    slope_penalty : float, optional
        Nonnegative slope penalty coefficient.
    curvature_penalty : float, optional
        Nonnegative discrete curvature penalty coefficient. At least one
        of slope_penalty and curvature_penalty must be positive.
    flat_edges : bool, optional
        If True, additionally enforce horizontal first
        and last polygon segments.
    left_saturation_width : float, optional
        Magnitude of the left saturation threshold.
    right_saturation_width : float, optional
        Field coordinate of the right saturation threshold.
    anchorpoints : array_like of shape (n, 3), optional
        Rows [x, y, K] give a field coordinate, a target magnetization,
        and a gain. The targets use the piecewise-linear interpolant.
        Require -left_saturation_width < x < right_saturation_width and K >= 0.
        Default is None.
    sine_transform : bool, optional
        If True, apply sin(pi/2 * y) to the optimized values.
    mirror : bool, optional
        If True, append the mirrored branch after any sine transformation.

    Returns
    -------
    x : numpy.ndarray
        Equidistant field coordinates along each branch.
    y : numpy.ndarray
        Magnetization values after the requested transformations.

    Notes
    -----
    The saturation conditions are hard constraints.

    Anchors are soft constraints controlled by their gains.

    Boundedness outside the fixed saturation regions is
    not explicitly enforced.

    A point-symmetric descending branch is obtained by

        x_desc = -x[::-1]
        y_desc = -y[::-1].
    """

    if N < 5:
        raise ValueError("Require N >= 5.")

    if not 0 < left_saturation_width <= 1:
        raise ValueError("Invalid left saturation width.")

    if not 0 < right_saturation_width <= 1:
        raise ValueError("Invalid right saturation width.")

    if curvature_penalty < 0 or slope_penalty < 0:
        raise ValueError("Invalid energy coefficients.")
    if slope_penalty == curvature_penalty == 0:
        raise ValueError("At least one penalty weight must be positive.")

    # Center the indices before division to keep opposite nodes symmetric.
    x = (2.0 * np.arange(N) - (N - 1)) / (N - 1)

    # Local segment lengths
    h = np.diff(x)

    # First derivative operator
    D1 = diags(
        [-1/h, 1/h],
        [0, 1],
        shape=(N - 1, N),
        format="csr",
    )

    # Second derivative operator
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

    # Quadratic energy residual operator
    L = vstack([
        diags(np.sqrt(slope_penalty * h)) @ D1,
        diags(np.sqrt(curvature_penalty * hc)) @ D2,
    ]).tocsr()

    # Hard saturation constraints
    left = x <= -left_saturation_width
    right = x >= right_saturation_width

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
    offset = L @ y

    if anchorpoints is not None:
        anchors = np.asarray(anchorpoints, dtype=float)
        if anchors.shape == (0,):
            anchors = anchors.reshape(0, 3)
        if anchors.ndim != 2 or anchors.shape[1] != 3:
            raise ValueError("anchorpoints must have shape (n, 3).")
        if not np.all(np.isfinite(anchors)) or np.any(anchors[:, 2] < 0):
            raise ValueError("Anchor entries must be finite and gains must be nonnegative.")
        if (
            np.any(anchors[:, 0] <= -left_saturation_width)
            or np.any(anchors[:, 0] >= right_saturation_width)
        ):
            raise ValueError(
                "Anchor field coordinates must satisfy "
                f"{-left_saturation_width} < x < {right_saturation_width}."
            )

        anchor_rows = []
        anchor_offsets = []
        for anchor_x, anchor_y, gain in anchors:
            if gain == 0:
                continue
            row = interpolation_row(anchor_x)
            free_row = row[:, free]
            if free_row.count_nonzero() == 0:
                continue  # The anchor residual is constant on the free variables.
            anchor_weight = np.sqrt(gain)
            anchor_rows.append(anchor_weight * free_row)
            anchor_offsets.append(anchor_weight * ((row @ y)[0] - anchor_y))

        if anchor_rows:
            L_free = vstack([L_free, *anchor_rows], format="csr")
            offset = np.concatenate((offset, anchor_offsets))

    # Solve the normal equations for the free support points.
    solution = spsolve(
        L_free.T @ L_free,
        -L_free.T @ offset,
    )
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Quadratic solve returned non-finite magnetization values.")
    y[free] = solution

    if sine_transform:
        y = np.sin(np.pi / 2 * y)
    if mirror:
        return mirror_polygon(x, y)
    return x, y


def generate_loop(*args, **kwargs):
    """Return ``((x_up, y_up), (x_down, y_down))`` for a point-symmetric loop.

    Pass the same generation arguments as to :func:`generate_polygon`.
    This function constructs the two branches separately. Field coordinates
    increase along the ascending branch and decrease along the descending branch.
    """
    kwargs["mirror"] = False
    x, y = generate_polygon(*args, **kwargs)
    return (x, y), (-x, -y)

def mirror_polygon(x: np.ndarray, y: np.ndarray):
    return np.concatenate((x, x[::-1])), np.concatenate((y, -y))