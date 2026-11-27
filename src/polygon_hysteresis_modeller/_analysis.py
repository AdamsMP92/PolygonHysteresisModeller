import numpy as np


def monotone_sectors(values):
    direction = np.sign(np.diff(values))
    moving = np.flatnonzero(direction)
    turns = np.zeros(direction.size, dtype=int)
    turns[moving[1:]] = direction[moving[1:]] != direction[moving[:-1]]
    return np.cumsum(turns)


def contains_sign_changes(value):
    return bool(np.any(value < 0) and np.any(value > 0))


def is_non_constant(x, y):
    return bool(np.diff(x).any() and np.diff(y).any())


def is_min_min(x, y):
    return bool(np.any((x == x.min()) & (y == y.min())))


def is_max_max(x, y):
    return bool(np.any((x == x.max()) & (y == y.max())))


def is_rise_fall_symmetric(x, y):

    x_sectors = monotone_sectors(x)
    y_sectors = monotone_sectors(y)

    joint_turns = (np.diff(x_sectors) != 0) | (np.diff(y_sectors) != 0)
    joint_edges = np.r_[0, np.flatnonzero(joint_turns) + 1, len(y) - 1]
    x_directions = np.sign(np.diff(x[joint_edges])).astype(int)
    y_directions = np.sign(np.diff(y[joint_edges])).astype(int)
    direction_products = x_directions * y_directions
    return bool(np.all(direction_products >= 0))


def is_sign_change_regular(x, y):
    nonzero = y != 0
    sign_steps = np.diff(np.sign(y[nonzero]))
    dx = np.diff(x[nonzero])
    crossings = sign_steps != 0
    return bool(np.all(dx[crossings] * sign_steps[crossings] > 0))


def is_regular(x, y):

    if not is_non_constant(x, y):
        print("data is constant")
        return False
    
    if not contains_sign_changes(y):
        print("no sign changes in y")
        return False

    if not contains_sign_changes(x):
        print("no sign changes in x")
        return False

    if not is_min_min(x, y):
        return False

    if not is_max_max(x, y):
        return False

    if not is_rise_fall_symmetric(x, y):
        return False

    if not is_sign_change_regular(x, y):
        return False

    return True

