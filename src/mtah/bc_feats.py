import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BC_DIR = os.path.join(ROOT, "data_external", "bc")
FPS = 25
WIN_C = (1, 3, 9, 27)
WIN_PF = (2, 6, 20)
CAP = 60.0

NAMES = (["bc_away", "bc_nvis", "bc_ballvis"]
         + ["bc_away_c%d" % w for w in WIN_C] + ["bc_nvis_c%d" % w for w in WIN_C]
         + ["bc_ball_c%d" % w for w in WIN_C]
         + ["bc_away_p%d" % w for w in WIN_PF] + ["bc_away_f%d" % w for w in WIN_PF]
         + ["bc_since_away", "bc_until_away", "bc_since_live", "bc_until_live", "bc_run_len"])


def has_cues(name):
    return os.path.exists(os.path.join(BC_DIR, name + ".npz"))


def load_cues(name):
    if name == "provided":
        from mtah.bc_provided import build
        return build()
    z = np.load(os.path.join(BC_DIR, name + ".npz"))
    return {k: z[k] for k in z.files}


def _box(x, w):
    n = len(x)
    c = np.r_[0.0, np.cumsum(x)]
    h = w // 2
    i = np.arange(n)
    lo, hi = np.maximum(i - h, 0), np.minimum(i + h + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def _past(x, w):
    n = len(x)
    c = np.r_[0.0, np.cumsum(x)]
    i = np.arange(n)
    lo = np.maximum(i - w, 0)
    return np.where(i > 0, (c[i] - c[lo]) / np.maximum(i - lo, 1), np.nan)


def _future(x, w):
    return _past(x[::-1], w)[::-1]


def _since(m):
    n = len(m)
    idx = np.where(m, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum((np.arange(n) - last) / FPS, CAP)


def _seg(a, b):
    away = (np.nan_to_num(a["has"], nan=0.0) < 0.5) | (np.nan_to_num(a["n_vis"], nan=0.0) < 4)
    nv = np.clip(np.nan_to_num(a["n_vis"], nan=0.0), 0, 22) / 22.0
    nv = np.where(away & (np.nan_to_num(a["has"], nan=0.0) < 0.5), 0.0, nv)
    bv = np.nan_to_num(a["ball_vis"], nan=0.0)
    cols = [away.astype(float), nv, bv]
    for x in (away.astype(float), nv, bv):
        cols += [_box(x, w * FPS + 1) for w in WIN_C]
    cols += [_past(away.astype(float), w * FPS) for w in WIN_PF]
    cols += [_future(away.astype(float), w * FPS) for w in WIN_PF]
    cols += [_since(away), _since(away[::-1])[::-1], _since(~away), _since((~away)[::-1])[::-1]]
    ch = np.r_[0, np.flatnonzero(away[1:] != away[:-1]) + 1]
    ln = np.diff(np.r_[ch, len(away)])
    cols.append(np.minimum(np.repeat(ln, ln) / FPS, 2 * CAP))
    return np.stack(cols, 1)


def cue_features(c, half):
    n = len(half)
    for k in ("n_vis", "ball_vis", "has"):
        if len(c[k]) != n:
            raise ValueError("cue array %s has %d frames, match has %d" % (k, len(c[k]), n))
    X = np.full((n, len(NAMES)), np.nan, np.float32)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        seg = {k: c[k][idx] for k in ("n_vis", "ball_vis", "has")}
        X[idx] = _seg(seg, None)
    return X
