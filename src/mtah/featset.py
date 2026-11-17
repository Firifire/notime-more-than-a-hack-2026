import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d

from mtah.ball import clean, runs

FPS = 25
LAGS = (-50, -25, -12, -5, 0, 5, 12, 25, 50)


def attack_sign(players, team, half, pitch):
    L = pitch[0]
    sign = np.zeros(len(half), np.int8)
    x = players[:, :, 0]
    for h in np.unique(half):
        if h == 0:
            continue
        m = half == h
        with np.errstate(all="ignore"):
            min0 = np.nanmedian(np.nanmin(np.where(team[None] == 0, x[m], np.nan), 1))
            min1 = np.nanmedian(np.nanmin(np.where(team[None] == 1, x[m], np.nan), 1))
        sign[m] = 1 if min0 < min1 else -1
    return sign


def normalise(match):
    L, W = match["pitch"]
    s = attack_sign(match["players"], match["team"], match["half"], match["pitch"])
    flip = s < 0
    B = match["ball"][:, :2].astype(np.float64).copy()
    P = match["players"][:, :, :2].astype(np.float64).copy()
    B[:, 0] -= L / 2
    B[:, 1] -= W / 2
    P[:, :, 0] -= L / 2
    P[:, :, 1] -= W / 2
    B[flip] *= -1
    P[flip] *= -1
    return B, P


def swap_teams(match):
    m = dict(match)
    m["team"] = 1 - match["team"]
    if "label" in match:
        lab = match["label"].copy()
        lab[match["label"] == 0] = 1
        lab[match["label"] == 1] = 0
        m["label"] = lab
    return m


def _roll(x, w, fn):
    return fn(x, size=w, mode="nearest")


def _since(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(n) - last, 3000).astype(np.float32)


def _until(mask):
    return _since(mask[::-1])[::-1]


def build(match):
    L, W = match["pitch"]
    team = match["team"]
    Bn, Pn = normalise(match)
    bxy, bvel, bspd, bpres = clean(Bn)
    N = len(bxy)
    F = {}
    F["b_pres"] = bpres.astype(np.float32)
    F["b_x"] = bxy[:, 0]
    F["b_y"] = bxy[:, 1]
    F["b_spd"] = bspd
    F["b_out_x"] = np.abs(bxy[:, 0]) - L / 2
    F["b_out_y"] = np.abs(bxy[:, 1]) - W / 2
    F["t_since_ball"] = _since(bpres)
    F["t_until_ball"] = _until(bpres)
    acc = np.r_[0, np.linalg.norm(np.diff(bvel, axis=0), axis=1)] * FPS
    F["b_acc"] = acc
    F["b_acc_max12"] = _roll(np.nan_to_num(acc), 13, maximum_filter1d)
    F["b_spd_max25"] = _roll(np.nan_to_num(bspd), 25, maximum_filter1d)
    F["b_still"] = uniform_filter1d((np.nan_to_num(bspd, nan=99) < 0.5).astype(float), 50)

    k = 5
    v = np.full(Pn.shape[:2], np.nan)
    v[k:-k] = np.linalg.norm(Pn[2 * k:] - Pn[:-2 * k], axis=-1) / (2 * k) * FPS
    v[v > 12] = np.nan
    pres = ~np.isnan(Pn[:, :, 0])
    d = np.sqrt(((Pn - bxy[:, None, :]) ** 2).sum(-1))
    with np.errstate(all="ignore"):
        F["v_all"] = np.nanmean(v, 1)
        F["v_p80"] = np.nanpercentile(v, 80, 1)
    for t, nm in ((0, "h"), (1, "a")):
        m = team == t
        dt = np.where(pres[:, m], d[:, m], np.inf)
        ds = np.sort(dt, 1)
        F["d1_" + nm] = np.where(np.isfinite(ds[:, 0]), ds[:, 0], np.nan)
        F["d2_" + nm] = np.where(np.isfinite(ds[:, 1]), ds[:, 1], np.nan)
        F["n3_" + nm] = (dt < 3).sum(1).astype(np.float32)
        F["n10_" + nm] = (dt < 10).sum(1).astype(np.float32)
        F["np_" + nm] = pres[:, m].sum(1).astype(np.float32)
        with np.errstate(all="ignore"):
            x = np.where(pres[:, m], Pn[:, m, 0], np.nan)
            y = np.where(pres[:, m], Pn[:, m, 1], np.nan)
            F["cx_" + nm] = np.nanmean(x, 1)
            F["sx_" + nm] = np.nanstd(x, 1)
            F["sy_" + nm] = np.nanstd(y, 1)
            F["v_" + nm] = np.nanmean(np.where(pres[:, m], v[:, m], np.nan), 1)
            vx = np.full(x.shape, np.nan)
            vx[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2 * k) * FPS
            F["vx_" + nm] = np.nanmean(vx, 1)
    F["d_diff"] = F["d1_h"] - F["d1_a"]
    F["cx_diff"] = F["cx_h"] - F["cx_a"]
    F["spread_diff"] = (F["sx_h"] * F["sy_h"]) - (F["sx_a"] * F["sy_a"])

    for key in ("d_diff", "d1_h", "d1_a", "b_spd", "v_all"):
        x = F[key]
        for lag in LAGS:
            if lag == 0:
                continue
            F["%s@%+d" % (key, lag)] = np.roll(x, -lag)
    for key, w in (("v_all", 25), ("v_all", 100), ("v_all", 250), ("v_p80", 100),
                   ("cx_diff", 100), ("vx_h", 50), ("vx_a", 50), ("b_pres", 250)):
        x = np.nan_to_num(F[key], nan=np.nanmean(F[key]))
        F["%s_m%d" % (key, w)] = uniform_filter1d(x, w)
    F["v_all_drop"] = F["v_all_m100"] - np.roll(F["v_all_m100"], 100)
    F["v_all_rise"] = np.roll(F["v_all_m100"], -100) - F["v_all_m100"]
    px, py = Pn[:, :, 0], Pn[:, :, 1]
    with np.errstate(invalid="ignore"):
        F["n_beyond_touch"] = (np.abs(py) > W / 2 + 0.3).sum(1).astype(np.float32)
        F["n_beyond_goal"] = (np.abs(px) > L / 2 + 0.3).sum(1).astype(np.float32)
        dc = np.sqrt((np.abs(px) - L / 2) ** 2 + (np.abs(py) - W / 2) ** 2)
        F["min_corner_d"] = np.nanmin(np.where(pres, dc, np.inf), 1)
        F["min_centre_d"] = np.nanmin(np.where(pres, np.sqrt(px ** 2 + py ** 2), np.inf), 1)
        own_h = np.where(pres[:, team == 0], px[:, team == 0] < 0.5, np.nan)
        own_a = np.where(pres[:, team == 1], px[:, team == 1] > -0.5, np.nan)
        F["own_half_h"] = np.nanmean(own_h, 1)
        F["own_half_a"] = np.nanmean(own_a, 1)
        in_ga_left = (px < -L / 2 + 5.5) & (np.abs(py) < 9.2) & pres
        in_ga_right = (px > L / 2 - 5.5) & (np.abs(py) < 9.2) & pres
        F["n_ga_left"] = in_ga_left.sum(1).astype(np.float32)
        F["n_ga_right"] = in_ga_right.sum(1).astype(np.float32)
        in_box_left = (px < -L / 2 + 16.5) & (np.abs(py) < 20.2) & pres
        in_box_right = (px > L / 2 - 16.5) & (np.abs(py) < 20.2) & pres
        F["n_box_left"] = in_box_left.sum(1).astype(np.float32)
        F["n_box_right"] = in_box_right.sum(1).astype(np.float32)
    for key in ("n_beyond_touch", "min_corner_d", "min_centre_d", "own_half_h", "own_half_a"):
        x = np.nan_to_num(F[key], nan=0.0, posinf=60.0)
        F[key + "_m50"] = uniform_filter1d(x, 50)
    for key in ("v_all", "v_h", "v_a", "cx_h", "cx_a", "sx_h", "sx_a", "b_pres"):
        x = np.nan_to_num(F[key], nan=np.nanmean(F[key]))
        c = np.r_[0, np.cumsum(x)]
        n = len(x)
        i = np.arange(n)
        for w in (10, 50, 100, 500, 2000):
            lo = np.maximum(0, i - w)
            hi = np.minimum(n, i + w + 1)
            F["%s_past%d" % (key, w)] = (c[i + 1] - c[lo]) / (i + 1 - lo)
            F["%s_fut%d" % (key, w)] = (c[hi] - c[i]) / (hi - i)
    c_h = (F["d1_h"] < 1.5) & (F["d1_h"] + 0.5 < np.nan_to_num(F["d1_a"], nan=99))
    c_a = (F["d1_a"] < 1.5) & (F["d1_a"] + 0.5 < np.nan_to_num(F["d1_h"], nan=99))
    F["since_ctrl_h"] = _since(c_h)
    F["since_ctrl_a"] = _since(c_a)
    F["until_ctrl_h"] = _until(c_h)
    F["until_ctrl_a"] = _until(c_a)
    F["ctrl_frac_h_250"] = uniform_filter1d(c_h.astype(float), 250)
    F["ctrl_frac_a_250"] = uniform_filter1d(c_a.astype(float), 250)
    F["ctrl_frac_h_50"] = uniform_filter1d(c_h.astype(float), 50)
    F["ctrl_frac_a_50"] = uniform_filter1d(c_a.astype(float), 50)
    acc_s = np.nan_to_num(uniform_filter1d(np.nan_to_num(acc), 3))
    vprev = np.roll(bvel, 4, axis=0)
    vnext = np.roll(bvel, -4, axis=0)
    with np.errstate(invalid="ignore"):
        cosang = (vprev * vnext).sum(1) / (np.linalg.norm(vprev, axis=1) * np.linalg.norm(vnext, axis=1) + 1e-6)
        dspeed = np.abs(np.linalg.norm(vnext, axis=1) - np.linalg.norm(vprev, axis=1))
    peak = (acc_s == maximum_filter1d(acc_s, 7)) & (acc_s > 12)
    turn = (np.nan_to_num(cosang, nan=1) < 0.85) | (np.nan_to_num(dspeed) > 3)
    cand = bpres & (peak | (turn & (np.r_[0, np.diff(turn.astype(int))] == 1)))
    d1h = np.nan_to_num(F["d1_h"], nan=99)
    d1a = np.nan_to_num(F["d1_a"], nan=99)
    t_h = cand & (d1h < 2.5) & (d1h <= d1a)
    t_a = cand & (d1a < 2.5) & (d1a < d1h)
    F["touch_h"] = t_h.astype(np.float32)
    F["touch_a"] = t_a.astype(np.float32)
    F["since_touch_h"] = _since(t_h)
    F["since_touch_a"] = _since(t_a)
    F["until_touch_h"] = _until(t_h)
    F["until_touch_a"] = _until(t_a)
    F["last_touch_diff"] = F["since_touch_a"] - F["since_touch_h"]
    F["next_touch_diff"] = F["until_touch_a"] - F["until_touch_h"]
    ev = np.flatnonzero(t_h | t_a)
    chain = np.zeros(N, np.float32)
    run, last = 0, 0
    prev_i = 0
    for i in ev:
        chain[prev_i:i] = run * last
        tm = 1 if t_h[i] else -1
        run = run + 1 if tm == last else 1
        last = tm
        prev_i = i
    chain[prev_i:] = run * last
    F["touch_chain"] = chain
    F["touch_rate_h_250"] = uniform_filter1d(t_h.astype(float), 250) * 250
    F["touch_rate_a_250"] = uniform_filter1d(t_a.astype(float), 250) * 250
    names = sorted(F)
    X = np.stack([np.asarray(F[n], np.float32) for n in names], 1)
    X[~np.isfinite(X)] = np.nan
    return X, names
