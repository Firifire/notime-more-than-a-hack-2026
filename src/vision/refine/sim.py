import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.common import CACHE, ROOT  # noqa: E402

COMMON = os.path.join(ROOT, "data_external", "common")
OUT = os.path.join(CACHE, "refine")
CLEAN = os.path.join(OUT, "clean")
FPS = 25.0
K = 8
S = 24
L, W = 105.0, 68.0
SOURCES = ("idsse", "pff", "metrica")


def player_feats(P, fps=FPS):
    P = P.astype(np.float32)
    V = np.full_like(P, np.nan)
    V[1:-1] = (P[2:] - P[:-2]) * (fps / 2)
    V[0], V[-1] = (P[1] - P[0]) * fps, (P[-1] - P[-2]) * fps
    V = np.where(np.isnan(V) & ~np.isnan(P), 0.0, V)
    F = np.concatenate([P, V], -1)
    absent = np.isnan(P[..., 0])
    o = np.argsort(absent, axis=1, kind="stable")[:, :S]
    out = np.take_along_axis(F, o[..., None], 1)
    if out.shape[1] < S:
        out = np.concatenate([out, np.full((len(out), S - out.shape[1], 4), np.nan, np.float32)], 1)
    return out.astype(np.float16)


def cmd_cache():
    os.makedirs(CLEAN, exist_ok=True)
    for src in SOURCES:
        for f in sorted(glob.glob(os.path.join(COMMON, "%s_*.npz" % src))):
            name = os.path.basename(f)[:-4]
            d = os.path.join(CLEAN, name)
            if os.path.exists(os.path.join(d, "players.npy")):
                continue
            z = np.load(f, allow_pickle=False)
            if float(np.mean(np.isnan(z["ball"][:, 0]))) > 0.6:
                continue
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "ball.npy"), z["ball"][:, :2].astype(np.float32))
            np.save(os.path.join(d, "label.npy"), z["label"].astype(np.int8))
            np.save(os.path.join(d, "half.npy"), z["half"].astype(np.int8))
            P = z["players"][:, :, :2]
            out = np.full((len(P), S, 4), np.nan, np.float16)
            for h in np.unique(z["half"]):
                if h == 0:
                    continue
                s = np.flatnonzero(z["half"] == h)
                out[s[0]:s[-1] + 1] = player_feats(P[s[0]:s[-1] + 1])
            np.save(os.path.join(d, "players.npy"), out)
            print(name, len(P), flush=True)


def clean_matches():
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(CLEAN, "*")) if
                  os.path.exists(os.path.join(d, "players.npy")))


def _lu(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def _beta(rng, mean, conc, size):
    mean = np.clip(mean, 0.02, 0.98)
    return rng.beta(mean * conc, (1 - mean) * conc, size)


def _markov_runs(rng, T, p_on, mean_off):
    mean_off = max(mean_off, 1.0)
    mean_on = max(mean_off * p_on / max(1 - p_on, 1e-3), 1.0)
    out = np.zeros(T, bool)
    t = 0
    on = rng.random() < p_on
    while t < T:
        n = int(rng.geometric(1.0 / (mean_on if on else mean_off)))
        out[t:t + n] = on
        t += n
        on = not on
    return out


MARKINGS = np.array([[52.5, 34.0], [11.0, 34.0], [94.0, 34.0], [0.0, 0.0], [105.0, 0.0], [0.0, 68.0], [105.0, 68.0],
                     [16.5, 34.0], [88.5, 34.0], [52.5, 0.0], [52.5, 68.0]])


def simulate(ball, players, label, rng):
    T = len(ball)
    known = ~np.isnan(ball[:, 0])
    bx = np.where(known[:, None], ball, 0.0)
    cands = []
    cam = np.array([52.5 + rng.uniform(-15, 15), -rng.uniform(15, 60)])
    cam_h = rng.uniform(12, 35)
    p_on = rng.uniform(0.05, 0.5) if rng.random() < 1 / 3 else rng.uniform(0.6, 0.99)
    vis = _markov_runs(rng, T, p_on, _lu(rng, 2, 300))
    far = rng.uniform(0, 0.9) if rng.random() < 0.5 else 0.0
    q_drop = rng.uniform(0, 0.6) if rng.random() < 0.4 else 0.0
    yfrac = np.clip(bx[:, 1] / W, 0, 1)
    vis &= rng.random(T) > (q_drop + far * yfrac ** 2)
    if label is not None and rng.random() < 0.5:
        dead = label == 2
        if dead.any():
            vis &= ~(dead & (_markov_runs(rng, T, rng.uniform(0.1, 0.9), _lu(rng, 10, 300))))
    vis &= known
    sig = rng.uniform(0.05, 0.5)
    bias_s = rng.uniform(0, 0.6)
    bias = np.zeros((T, 2))
    e = rng.normal(0, 1, (T, 2)) * bias_s * np.sqrt(1 - 0.995 ** 2)
    b = rng.normal(0, bias_s, 2)
    for t in range(T):
        b = 0.995 * b + e[t]
        bias[t] = b
    pos = bx + rng.normal(0, sig, (T, 2)) * (1 + 0.3 * yfrac[:, None]) + bias
    if rng.random() < 0.8:
        sp = np.r_[0, np.hypot(*np.diff(bx, axis=0).T)] * FPS
        sp[~known] = 0
        fast = sp > rng.uniform(11, 18)
        d = np.diff(np.r_[0, fast.astype(np.int8), 0])
        for a, z in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if z - a < 8 or rng.random() < 0.3:
                continue
            length = float(np.sum(sp[a:z]) / FPS)
            H = rng.uniform(0.3, max(0.6, 0.2 * length))
            tt = np.linspace(0, 1, z - a)
            h = 4 * H * tt * (1 - tt)
            v = bx[a:z] - cam
            dist = np.hypot(*v.T)[:, None]
            pos[a:z] += v / np.maximum(dist, 1) * (h[:, None] * dist / np.maximum(cam_h - h[:, None], 1.0))
    mu_tp = rng.uniform(0.25, 0.85)
    conf_tp = _beta(rng, mu_tp, rng.uniform(2, 8), T)
    f = np.flatnonzero(vis)
    cands.append((f, pos[f, 0], pos[f, 1], conf_tp[f]))
    dbl = vis & (rng.random(T) < rng.uniform(0, 0.1))
    f = np.flatnonzero(dbl)
    cands.append((f, pos[f, 0] + rng.normal(0, 0.8, len(f)), pos[f, 1] + rng.normal(0, 0.8, len(f)),
                  conf_tp[f] * rng.uniform(0.2, 0.8, len(f))))
    present = ~np.isnan(players[:, :, 0].astype(np.float32))
    rate = _lu(rng, 0.003, 0.15)
    nb = rng.poisson(rate * T)
    mu_fp = rng.uniform(0.07, 0.35)
    for _ in range(nb):
        t0 = int(rng.integers(0, T))
        dur = int(rng.geometric(1.0 / _lu(rng, 3, 150)))
        t1 = min(T, t0 + dur)
        pres = np.flatnonzero(present[t0])
        if not len(pres):
            continue
        j = int(rng.choice(pres))
        tt = np.arange(t0, t1)
        tt = tt[present[tt, j]]
        if not len(tt):
            continue
        off = rng.normal(0, 0.6, 2)
        px = players[tt, j, :2].astype(np.float64) + off + rng.normal(0, 0.2, (len(tt), 2))
        hi = rng.random() < 0.12
        cf = _beta(rng, rng.uniform(0.3, 0.8) if hi else mu_fp, 4, len(tt))
        keep = rng.random(len(tt)) < rng.uniform(0.3, 1.0)
        cands.append((tt[keep], px[keep, 0], px[keep, 1], cf[keep]))
    for _ in range(rng.poisson(rng.uniform(0, 3))):
        if rng.random() < 0.5:
            p0 = MARKINGS[rng.integers(len(MARKINGS))] + rng.normal(0, 0.3, 2)
        else:
            side = rng.integers(4)
            p0 = np.array([rng.uniform(-4, 109), rng.uniform(-4, 72)])
            if side == 0:
                p0[0] = rng.uniform(-4, 0)
            elif side == 1:
                p0[0] = rng.uniform(105, 109)
            elif side == 2:
                p0[1] = rng.uniform(-4, 0)
            else:
                p0[1] = rng.uniform(68, 72)
        m = rng.random(T) < rng.uniform(0.1, 0.9)
        f = np.flatnonzero(m)
        cf = _beta(rng, rng.uniform(0.08, 0.6), 6, len(f))
        cands.append((f, p0[0] + rng.normal(0, 0.15, len(f)), p0[1] + rng.normal(0, 0.15, len(f)), cf))
    lam = _lu(rng, 0.02, 2.0)
    nc = rng.poisson(lam, T)
    f = np.repeat(np.arange(T), nc)
    near = rng.random(len(f)) < rng.uniform(0.1, 0.7)
    cx = rng.uniform(-3, 108, len(f))
    cy = rng.uniform(-3, 71, len(f))
    if near.any():
        for i in np.flatnonzero(near):
            pr = np.flatnonzero(present[f[i]])
            if len(pr):
                j = pr[rng.integers(len(pr))]
                cx[i], cy[i] = players[f[i], j, :2].astype(np.float64) + rng.normal(0, 1.5, 2)
    cands.append((f, cx, cy, _beta(rng, rng.uniform(0.06, 0.2), 4, len(f))))
    fr = np.concatenate([c[0] for c in cands]).astype(np.int64)
    xx = np.concatenate([c[1] for c in cands])
    yy = np.concatenate([c[2] for c in cands])
    cc = np.clip(np.concatenate([c[3] for c in cands]), 0.05, 1.0)
    keep = (xx > -4) & (xx < 109) & (yy > -8) & (yy < 76)
    return pack(fr[keep], xx[keep], yy[keep], cc[keep], T)


def pack(fr, x, y, c, T, k=K):
    o = np.lexsort((-c, fr))
    fr, x, y, c = fr[o], x[o], y[o], c[o]
    first = np.searchsorted(fr, fr)
    r = np.arange(len(fr)) - first
    s = r < k
    out = np.full((T, k, 3), np.nan, np.float32)
    out[fr[s], r[s]] = np.stack([x[s], y[s], c[s]], -1)
    return out


def perturb_players(players, rng):
    P = players.astype(np.float32).copy()
    sig = rng.uniform(0, 0.3)
    P[..., :2] += rng.normal(0, sig, P[..., :2].shape)
    drop = rng.uniform(0, 0.12)
    if drop > 0:
        T, Sn = P.shape[:2]
        for j in range(Sn):
            if rng.random() < drop * 3:
                m = _markov_runs(rng, T, 1 - drop, _lu(rng, 10, 500))
                P[~m, j] = np.nan
    return P


def flip(ball, players, cands, fx, fy):
    for a in (ball, players, cands):
        if a is None:
            continue
        if fx:
            a[..., 0] = L - a[..., 0]
        if fy:
            a[..., 1] = W - a[..., 1]
    if players is not None:
        if fx:
            players[..., 2] *= -1
        if fy:
            players[..., 3] *= -1


def cmd_measure(halves):
    res = {}
    for hn in halves:
        z = np.load(os.path.join(OUT, "stv2_%s.npz" % hn))
        n = int(z["n"])
        C = pack(z["c_frame"].astype(np.int64), z["c_x"], z["c_y"], z["c_conf"], n, k=16)
        A = z["anchor"].astype(np.float64)
        ref = z["ref"].astype(np.float64)
        P = z["players"].astype(np.float32)
        fa = np.flatnonzero(~np.isnan(A[:, 0]))
        d = np.hypot(C[fa, :, 0] - A[fa, None, 0], C[fa, :, 1] - A[fa, None, 1])
        valid = ~np.isnan(C[fa, :, 2])
        tp = valid & (d < 1.5)
        fp = valid & (d > 3)
        has_tp = tp.any(1)
        best = np.where(tp, C[fa, :, 2], -1).max(1)
        k_tp = np.argmax(np.where(tp, C[fa, :, 2], -1), 1)
        off = C[fa, k_tp, :2] - A[fa]
        cf_fp = C[fa, :, 2][fp]
        pl = P[fa]
        dpl = np.hypot(C[fa, :, None, 0] - pl[:, None, :, 0], C[fa, :, None, 1] - pl[:, None, :, 1])
        dpl = np.nanmin(np.where(np.isnan(dpl), np.inf, dpl), 2)
        fr = np.flatnonzero(~np.isnan(ref[:, 0]))
        dr = np.hypot(C[fr, :, 0] - ref[fr, None, 0], C[fr, :, 1] - ref[fr, None, 1])
        hit = np.zeros(n, bool)
        hit[fr] = np.nanmin(np.where(np.isnan(dr), np.inf, dr), 1) < 1.5
        cont = np.zeros(n, bool)
        cont[fr] = True
        runs_hit, runs_miss = [], []
        dd = np.diff(np.r_[0, cont.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
            if b - a < 25:
                continue
            h = hit[a:b]
            e = np.flatnonzero(np.diff(h.astype(np.int8))) + 1
            for s0, s1 in zip(np.r_[0, e], np.r_[e, len(h)]):
                if s0 == 0 or s1 == len(h):
                    continue
                (runs_hit if h[s0] else runs_miss).append(s1 - s0)
        q = lambda v, p=(10, 50, 90): [round(float(x), 3) for x in np.percentile(v, p)] if len(v) else None
        res[hn] = {"anchors": int(len(fa)), "tp_rate_1.5m": round(float(has_tp.mean()), 4),
                   "tp_conf_p10_50_90": q(best[has_tp]), "tp_offset_m_p50_90": q(np.hypot(*off[has_tp].T), (50, 90)),
                   "fp_per_frame_mean": round(float(fp.sum(1).mean()), 3),
                   "fp_conf_p10_50_90": q(cf_fp), "fp_conf_ge0.3_share": round(float(np.mean(cf_fp >= 0.3)), 4),
                   "fp_nearest_player_m_p25_50_75": q(dpl[fp], (25, 50, 75)),
                   "fp_within_2m_of_player_share": round(float(np.mean(dpl[fp] < 2)), 4),
                   "hit_run_frames_p10_50_90": q(runs_hit), "miss_run_frames_p10_50_90": q(runs_miss),
                   "dense_ref_frames": int(len(fr)), "dense_ref_hit_rate": round(float(hit[fr].mean()), 4)}
        print(hn, json.dumps(res[hn]), flush=True)
    json.dump(res, open(os.path.join(OUT, "stv2_evidence_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cmd_cache()
    elif sys.argv[1] == "measure":
        cmd_measure(sys.argv[2:])
