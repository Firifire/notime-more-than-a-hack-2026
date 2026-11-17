import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.common import (CACHE, N_FRAMES, FPS, PITCH, W, H, HALVES, MARGIN,  # noqa: E402
                           load_homography, metres_to_px, px_to_metres, load_atd_ball)

MARGIN_NEAR_M = 1.5
MARGIN_FAR_M = 8.0
MARGIN_GOAL_M = 4.0
HOT_CELL_M = 0.5
STATIC_WIN = 60 * FPS
STATIC_FRAC = 0.5
HOT_PENALTY = 0.5
CONF_MIN = 0.05
GATE_M = 2.0
GATE_GROW_M = 1.2
MAX_MISS = 8
CONF0 = 0.30
LINK_GAP = 75
LINK_M = 3.0
VMAX = 35.0
LINK_COST = 0.02
RESTART = 1.0
SEG_LEN = 50
STATIC_EXTENT_M = 1.0
ISO_M = 5.0
INTERP_MAX = LINK_GAP
ATD_BONUS = 1.0
ATD_ANCHOR_M = 2.0
WASB_THR = 0.3
AGREE_PX = 12.0
WASB_BONUS = 0.30


def load_wasb_peaks(wdir, thr=WASB_THR):
    fs = sorted(glob.glob(os.path.join(wdir, "wasb_*.npz")))
    if not fs:
        return None
    parts = [dict(np.load(f)) for f in fs]
    w = {k: np.concatenate([p[k] for p in parts]) for k in ("frame", "u", "v", "score")}
    m = w["score"] >= thr
    w = {k: v[m] for k, v in w.items()}
    o = np.lexsort((-w["score"], w["frame"]))
    return {k: v[o] for k, v in w.items()}


def agreement(frame, u, v, w, tol=AGREE_PX):
    out = np.zeros(len(frame), bool)
    if w is None or len(w["frame"]) == 0:
        return out
    a = np.searchsorted(w["frame"], frame, "left")
    b = np.searchsorted(w["frame"], frame, "right")
    for k in range(int((b - a).max()) if len(a) else 0):
        j = a + k
        ok = j < b
        jj = np.where(ok, j, 0)
        d = np.hypot(w["u"][jj] - u, w["v"][jj] - v)
        out |= ok & (d <= tol)
    return out


def load_cands(cdir):
    fs = sorted(glob.glob(os.path.join(cdir, "cands_*.npz")))
    if not fs:
        raise SystemExit("no candidate files in " + cdir)
    parts = [dict(np.load(f)) for f in fs]
    keys = ["frame", "u", "v", "w", "h", "conf"]
    c = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    covered = np.zeros(N_FRAMES, bool)
    for p in parts:
        covered[int(p["start"]):int(p["stop"])] = True
    o = np.lexsort((-c["conf"], c["frame"]))
    return {k: v[o] for k, v in c.items()}, covered


def to_metres(c, Hs):
    Hinv = np.linalg.inv(Hs[c["frame"]])
    x, y = px_to_metres(Hinv, c["u"], c["v"] + c["h"] / 2)
    return x, y


def expected_px_per_m(Hs, frames, x, y):
    u0, v0 = metres_to_px(Hs[frames], x, y)
    u1, v1 = metres_to_px(Hs[frames], x + 1.0, y)
    return np.hypot(u1 - u0, v1 - v0)


def hot_mask(frame, x, y):
    gx = np.floor(x / HOT_CELL_M).astype(np.int64)
    gy = np.floor(y / HOT_CELL_M).astype(np.int64)
    key = (gx + 1000) * 100000 + (gy + 1000)
    cells = {}
    o = np.lexsort((frame, key))
    ks, fs = key[o], frame[o]
    bnd = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, b in zip(bnd[:-1], bnd[1:]):
        cells[int(ks[a])] = np.unique(fs[a:b])
    cnt = np.zeros(len(frame), np.int64)
    half = STATIC_WIN // 2
    for a, b in zip(bnd[:-1], bnd[1:]):
        k0 = int(ks[a])
        ii = o[a:b]
        f = frame[ii]
        tot = np.zeros(len(ii), np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                arr = cells.get(k0 + dx * 100000 + dy)
                if arr is None:
                    continue
                tot += np.searchsorted(arr, f + half, "right") - np.searchsorted(arr, f - half)
        cnt[ii] = tot
    return cnt > STATIC_FRAC * STATIC_WIN


def build_tracklets(frame, x, y, conf):
    n = len(frame)
    tid = np.full(n, -1, np.int64)
    starts = np.searchsorted(frame, np.unique(frame))
    ends = np.append(starts[1:], n)
    A_id = np.zeros(0, np.int64); A_f = np.zeros(0, np.int64)
    A_x = np.zeros(0); A_y = np.zeros(0); A_vx = np.zeros(0); A_vy = np.zeros(0)
    A_n = np.zeros(0, np.int64)
    next_id = 0
    for s, e in zip(starts, ends):
        f = frame[s]
        alive = f - A_f <= MAX_MISS
        if not alive.all():
            A_id, A_f, A_x, A_y, A_vx, A_vy, A_n = (v[alive] for v in
                                                    (A_id, A_f, A_x, A_y, A_vx, A_vy, A_n))
        k = e - s
        used = np.zeros(k, bool)
        if len(A_id):
            dt = (f - A_f).astype(np.float64)
            px_, py_ = A_x + A_vx * dt, A_y + A_vy * dt
            gate = GATE_M + GATE_GROW_M * (dt - 1)
            D = np.hypot(x[s:e][None, :] - px_[:, None], y[s:e][None, :] - py_[:, None])
            ok = D < gate[:, None]
            if ok.any():
                ti, cj = np.nonzero(ok)
                cost = D[ti, cj] - 0.5 * conf[s + cj]
                tused = np.zeros(len(A_id), bool)
                for q in np.argsort(cost):
                    t, j = ti[q], cj[q]
                    if tused[t] or used[j]:
                        continue
                    tused[t] = used[j] = True
                    jj = s + j
                    nvx, nvy = (x[jj] - A_x[t]) / dt[t], (y[jj] - A_y[t]) / dt[t]
                    a = 0.6 if A_n[t] > 1 else 1.0
                    A_vx[t] = a * nvx + (1 - a) * A_vx[t]
                    A_vy[t] = a * nvy + (1 - a) * A_vy[t]
                    A_x[t], A_y[t], A_f[t] = x[jj], y[jj], f
                    A_n[t] += 1
                    tid[jj] = A_id[t]
        new = np.flatnonzero(~used & (conf[s:e] >= CONF0))
        if len(new):
            m = len(new)
            ids = np.arange(next_id, next_id + m)
            next_id += m
            tid[s + new] = ids
            A_id = np.concatenate([A_id, ids]); A_f = np.concatenate([A_f, np.full(m, f)])
            A_x = np.concatenate([A_x, x[s + new]]); A_y = np.concatenate([A_y, y[s + new]])
            A_vx = np.concatenate([A_vx, np.zeros(m)]); A_vy = np.concatenate([A_vy, np.zeros(m)])
            A_n = np.concatenate([A_n, np.ones(m, np.int64)])
    return tid


def select(frame, x, y, conf, tid, hot, anchor, isolated=None, bonus=None):
    if isolated is None:
        isolated = np.zeros(len(frame), bool)
    sel = np.flatnonzero(tid >= 0)
    if len(sel) == 0:
        return np.zeros(len(frame), bool), []
    o = sel[np.lexsort((frame[sel], tid[sel]))]
    t_of = tid[o]
    tb = np.flatnonzero(np.r_[True, t_of[1:] != t_of[:-1], True])
    seg_members, seg_score = [], []
    for a, b in zip(tb[:-1], tb[1:]):
        m = o[a:b]
        mx, my = np.median(x[m]), np.median(y[m])
        extent = np.hypot(x[m] - mx, y[m] - my).max()
        c = conf[m] - CONF0
        if extent <= STATIC_EXTENT_M:
            c = c - HOT_PENALTY * (hot[m] | isolated[m])
        c = c + ATD_BONUS * anchor[m]
        if bonus is not None:
            c = c + bonus[m]
        f = frame[m]
        cut = np.flatnonzero(np.diff(f // SEG_LEN) != 0) + 1
        for mm, cc in zip(np.split(m, cut), np.split(c, cut)):
            seg_members.append(mm)
            seg_score.append(cc.sum())
    T = len(seg_members)
    score = np.array(seg_score)
    tf0 = np.array([frame[m[0]] for m in seg_members]); tf1 = np.array([frame[m[-1]] for m in seg_members])
    x0 = np.array([x[m[0]] for m in seg_members]); y0 = np.array([y[m[0]] for m in seg_members])
    x1 = np.array([x[m[-1]] for m in seg_members]); y1 = np.array([y[m[-1]] for m in seg_members])
    order = np.argsort(tf0, kind="stable")
    end_order = np.argsort(tf1, kind="stable")
    tf1_sorted = tf1[end_order]
    best = np.full(T, -np.inf)
    prev = np.full(T, -1, np.int64)
    ptr = 0
    run_best, run_arg = -np.inf, -1
    for j in order:
        hi = np.searchsorted(tf1_sorted, tf0[j], "left")
        while ptr < hi:
            i = end_order[ptr]
            if best[i] > run_best:
                run_best, run_arg = best[i], i
            ptr += 1
        b, p = 0.0, -1
        if run_arg >= 0 and run_best - RESTART > b:
            b, p = run_best - RESTART, run_arg
        lo = np.searchsorted(tf1_sorted, tf0[j] - LINK_GAP, "left")
        cand = end_order[lo:hi]
        if len(cand):
            gap = (tf0[j] - tf1[cand]).astype(np.float64)
            dist = np.hypot(x0[j] - x1[cand], y0[j] - y1[cand])
            ok = dist <= LINK_M + VMAX * gap / FPS
            if ok.any():
                v = best[cand[ok]] - LINK_COST * dist[ok]
                k = np.argmax(v)
                if v[k] > b:
                    b, p = v[k], cand[ok][k]
        best[j] = b + score[j]
        prev[j] = p
    k = int(np.argmax(best))
    chosen = np.zeros(len(frame), bool)
    linked = []
    while k >= 0:
        if score[k] > 0 or best[k] > 0:
            chosen[seg_members[k]] = True
        p = prev[k]
        if p >= 0:
            linked.append((tf1[p], tf0[k]))
        k = p
    return chosen, linked


def isolated_mask(frame, x, y):
    from mtah.data import load_tracking
    P = load_tracking()["players"][:, :, :2]
    out = np.zeros(len(frame), bool)
    for s in range(0, len(frame), 200000):
        f = frame[s:s + 200000]
        d = np.hypot(P[f, :, 0] - x[s:s + 200000, None], P[f, :, 1] - y[s:s + 200000, None])
        with np.errstate(all="ignore"):
            dm = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
        out[s:s + 200000] = np.isfinite(dm) & (dm > ISO_M)
    return out


def run(cdir, out, seed_atd=False, wasb_dir=None):
    Hs = load_homography()
    c, covered = load_cands(cdir)
    keep = c["conf"] >= CONF_MIN
    c = {k: v[keep] for k, v in c.items()}
    x, y = to_metres(c, Hs)
    inb = ((x > -MARGIN_GOAL_M) & (x < PITCH[0] + MARGIN_GOAL_M) &
           (y > -MARGIN_FAR_M) & (y < PITCH[1] + MARGIN_NEAR_M) & np.isfinite(x))
    c = {k: v[inb] for k, v in c.items()}
    x, y = x[inb], y[inb]
    print("candidates in pitch area:", len(x))
    wp = load_wasb_peaks(wasb_dir) if wasb_dir else None
    agree_all = agreement(c["frame"], c["u"], c["v"], wp)
    if wp is not None:
        print("WASB peaks %d, YOLO candidates confirmed by WASB: %.3f" % (len(wp["frame"]),
                                                                       agree_all.mean()))
    atd = load_atd_ball()

    res = {k: np.full(N_FRAMES, np.nan, np.float32) for k in ("x", "y", "u", "v", "conf",
                                                              "size_ratio")}
    state = np.zeros(N_FRAMES, np.int8)
    for half, a, b in HALVES:
        m = (c["frame"] >= a - MARGIN) & (c["frame"] <= b + MARGIN)
        if not m.any():
            continue
        fr, xx, yy, cf = c["frame"][m], x[m], y[m], c["conf"][m]
        uu, vv, ww = c["u"][m], c["v"][m], c["w"][m]
        hot = hot_mask(fr, xx, yy)
        anchor = np.zeros(len(fr), bool)
        if seed_atd:
            ax, ay = atd[fr, 0], atd[fr, 1]
            anchor = np.hypot(ax - xx, ay - yy) < ATD_ANCHOR_M
        tid = build_tracklets(fr, xx, yy, cf)
        iso = isolated_mask(fr, xx, yy)
        bonus = WASB_BONUS * agree_all[m] if wp is not None else None
        chosen, linked = select(fr, xx, yy, cf, tid, hot, anchor, iso, bonus)
        print("half %d: cands %d hot %.1f%% tracklets %d chosen %d" % (
            half, len(fr), 100 * hot.mean(), tid.max() + 1, chosen.sum()))
        ci = np.flatnonzero(chosen)
        f = fr[ci]
        res["x"][f], res["y"][f] = xx[ci], yy[ci]
        res["u"][f], res["v"][f], res["conf"][f] = uu[ci], vv[ci], cf[ci]
        exp = expected_px_per_m(Hs, f, xx[ci], yy[ci]) * 0.22
        res["size_ratio"][f] = ww[ci] / np.maximum(exp, 1e-3)
        state[f] = 1
        det_f = np.flatnonzero(state[a - MARGIN:b + MARGIN + 1] == 1) + a - MARGIN
        if len(det_f) > 1:
            gaps = np.diff(det_f)
            same_t = np.ones(len(gaps), bool)
            for g0, g1 in zip(det_f[:-1][gaps > 1], det_f[1:][gaps > 1]):
                if g1 - g0 > INTERP_MAX:
                    continue
                t = (np.arange(g0 + 1, g1) - g0) / (g1 - g0)
                for k in ("x", "y", "u", "v"):
                    res[k][g0 + 1:g1] = res[k][g0] + t * (res[k][g1] - res[k][g0])
                res["conf"][g0 + 1:g1] = 0.0
                state[g0 + 1:g1] = 2
    interp = state == 2
    fi = np.flatnonzero(interp)
    uu, vv = metres_to_px(Hs[fi], res["x"][fi].astype(np.float64), res["y"][fi].astype(np.float64))
    res["u"][fi], res["v"][fi] = uu, vv
    sr = res["size_ratio"].copy()
    airborne = np.zeros(N_FRAMES, bool)
    good = np.isfinite(sr)
    if good.any():
        from scipy.ndimage import median_filter
        s2 = np.where(good, sr, 1.0)
        airborne = (median_filter(s2, 9) > 1.6) & (state > 0)
    np.savez_compressed(
        out, xy=np.stack([res["x"], res["y"]], 1), conf=res["conf"],
        uv=np.stack([res["u"], res["v"]], 1), state=state, size_ratio=sr, airborne=airborne,
        processed=covered, seed_atd=seed_atd,
        doc="xy metres (pitch 105.3x68) NaN where none; uv pixels; state 0 none/1 detected/"
            "2 interpolated; conf detector score (0 when interpolated); airborne = weak flag "
            "(box larger than a ground ball at the projected point)")
    g = np.zeros(N_FRAMES, bool)
    for _, a, b in HALVES:
        g[a:b + 1] = True
    print("wrote %s  graded coverage: detected %.3f  +interp %.3f" % (
        out, (state[g] == 1).mean(), (state[g] > 0).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=os.path.join(CACHE, "cands"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed-atd", action="store_true")
    ap.add_argument("--wasb", default=None, help="dir of wasb_*.npz peaks (second detector)")
    a = ap.parse_args()
    out = a.out or os.path.join(CACHE, "ball_video_seeded.npz" if a.seed_atd
                                else "ball_video_pure.npz")
    run(a.cands, out, a.seed_atd, a.wasb)


if __name__ == "__main__":
    main()
