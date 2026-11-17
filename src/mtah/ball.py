import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from mtah.data import FPS


def runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def clean(ball_xy, max_jump=4.0, fill_gap=12, smooth=7):
    xy = ball_xy.astype(np.float64).copy()
    N = len(xy)
    present = ~np.isnan(xy[:, 0])
    d_prev = np.r_[np.inf, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    d_next = np.r_[d_prev[1:], np.inf]
    spike = present & (np.nan_to_num(d_prev, nan=0) > max_jump) & (np.nan_to_num(d_next, nan=0) > max_jump)
    xy[spike] = np.nan
    present = ~np.isnan(xy[:, 0])
    s, e = runs(~present)
    for a, b in zip(s, e):
        if a == 0 or b >= N or b - a > fill_gap:
            continue
        w = np.linspace(0, 1, b - a + 2)[1:-1, None]
        xy[a:b] = xy[a - 1] * (1 - w) + xy[b] * w
    present = ~np.isnan(xy[:, 0])
    vel = np.full((N, 2), np.nan)
    s, e = runs(present)
    for a, b in zip(s, e):
        seg = xy[a:b]
        if b - a >= smooth + 2:
            seg = savgol_filter(seg, smooth, 2, axis=0)
            v = savgol_filter(xy[a:b], smooth, 2, deriv=1, axis=0) * FPS
        elif b - a >= 2:
            v = np.gradient(seg, axis=0) * FPS
        else:
            v = np.zeros_like(seg)
        xy[a:b] = seg
        vel[a:b] = v
    speed = np.linalg.norm(vel, axis=1)
    return xy, vel, speed, present
