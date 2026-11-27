"""Generate normalized polygonal hysteresis curves."""

import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from scipy.sparse.linalg import spsolve


def generate_polygon(
    points_per_branch=21,
    smoothness_weights=[1.0, 0.1],
    apply_flat_edges=True,
    saturation_bounds=[1.0, 1.0],
    anchorpoints=None,
    *,
    apply_sine=False,
    apply_mirror=False,
):
    """Generate a synthetic hysteresis branch as a polygonal chain.

    The branch consists of equally spaced support points on [-1, 1].
    The magnetization values are determined by minimizing
    a discrete energy functional with slope, curvature, and optional anchor terms.

    The continuous counterpart of the energy is

        E = slope_penalty/2 * integral(f'**2, dx)
          + curvature_penalty/2 * integral(f''**2, dx)
          + 1/2 * sum(K_j * (f(x_anchor_j) - y_anchor_j)**2).

    The slope and curvature contributions are discretized
    using differences between adjacent support points.

    The saturation thresholds fix the magnetization at existing support points.

        y_i = -1 if x_i <= -sat_left
        y_i =  1 if x_i >=  sat_right

    The optimization varies the free magnetization values.

    Parameters
    ----------
    points_per_branch : int, optional
        Number of support points in one branch, including endpoints.
    smoothness_weights : array_like of length 2, optional
        Nonnegative weights [slope_penalty, curvature_penalty].
        At least one weight must be positive. Default is [1.0, 0.1].
    apply_flat_edges : bool, optional
        If True, additionally enforce horizontal first
        and last polygon segments.
    saturation_bounds : array_like of length 2, optional
        Threshold magnitudes [sat_left, sat_right], each in (0, 1].
        Default is [1.0, 1.0].
    anchorpoints : array_like of shape (n, 3), optional
        Rows [x, y, K] give a field coordinate, a target magnetization,
        and a gain. The targets use the piecewise-linear interpolant.
        Require -sat_left < x < sat_right and K >= 0.
        Default is None.
    apply_sine : bool, optional
        If True, apply sin(pi/2 * y) to the optimized values.
    apply_mirror : bool, optional
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

    if points_per_branch < 5:
        raise ValueError("Require points_per_branch >= 5.")

    sat_left, sat_right = saturation_bounds
    if not (0 < sat_left <= 1 and 0 < sat_right <= 1):
        raise ValueError("saturation_bounds must contain two values in (0, 1].")

    slope_penalty, curvature_penalty = smoothness_weights
    if curvature_penalty < 0 or slope_penalty < 0:
        raise ValueError("Invalid energy coefficients.")
    if slope_penalty == curvature_penalty == 0:
        raise ValueError("At least one penalty weight must be positive.")

    # Center the indices before division to keep opposite nodes symmetric.
    segment_count = points_per_branch - 1
    x = (2.0 * np.arange(points_per_branch) - segment_count) / segment_count

    # Local segment lengths
    h = np.diff(x)

    # First derivative operator
    D1 = diags(
        [-1/h, 1/h],
        [0, 1],
        shape=(segment_count, points_per_branch),
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
        shape=(points_per_branch - 2, points_per_branch),
        format="csr",
    )

    def interpolation_row(xp):

        j = np.searchsorted(x, xp) - 1
        j = np.clip(j, 0, points_per_branch - 2)

        t = (xp - x[j]) / h[j]

        return csr_matrix(
            ([1 - t, t], ([0, 0], [j, j + 1])),
            shape=(1, points_per_branch),
        )

    # Quadratic energy residual operator
    L = vstack([
        diags(np.sqrt(slope_penalty * h)) @ D1,
        diags(np.sqrt(curvature_penalty * hc)) @ D2,
    ]).tocsr()

    # Hard saturation constraints
    left = x <= -sat_left
    right = x >= sat_right

    if apply_flat_edges:
        left[:2] = True
        right[-2:] = True

    if np.any(left & right):
        raise ValueError("Overlapping saturation constraints.")

    y = np.zeros(points_per_branch)

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
            np.any(anchors[:, 0] <= -sat_left)
            or np.any(anchors[:, 0] >= sat_right)
        ):
            raise ValueError(
                "Anchor field coordinates must satisfy "
                f"{-sat_left} < x < {sat_right}."
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

    if apply_sine:
        y = np.sin(np.pi / 2 * y)
    if apply_mirror:
        return mirror_polygon(x, y)
    return x, y


def generate_loop(*args, **kwargs):
    """Return ``((x_up, y_up), (x_down, y_down))`` for a point-symmetric loop.

    Pass the same generation arguments as to :func:`generate_polygon`.
    This function constructs the two branches separately. Field coordinates
    increase along the ascending branch and decrease along the descending branch.
    """
    kwargs["apply_mirror"] = False
    x, y = generate_polygon(*args, **kwargs)
    return (x, y), (-x, -y)

def mirror_polygon(x: np.ndarray, y: np.ndarray):
    return np.concatenate((x, x[::-1])), np.concatenate((y, -y))