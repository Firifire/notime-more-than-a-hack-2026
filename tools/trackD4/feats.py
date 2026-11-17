import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools", "trackK2"))
from vision.players_common import hist_dist  # noqa: E402
import k2track as K2  # noqa: E402

K = 12
NEW, JUNK = K, K + 1
V = K + 2
MAXGAP = 250
DT = 0.04
FT = 16 + 48
FP = 14
BASE_M = 1.0


def tracklet_table(pieces, xy, cf, ft):
    n = len(xy)
    frames = np.arange(n)
    fpos = {t: t for t in frames}
    amb, clean = K2.frame_flags(frames, xy)
    rows = []
    for k, obs in pieces.items():
        s = K2._summ(obs, xy, ft, cf, fpos, clean)
        ts = np.array([t for t, _ in obs])
        ps = np.array([xy[t][d] for t, d in obs])
        cs = np.array([cf[t][d] for t, d in obs])
        fa = np.array([ft[t][d] for t, d in obs])
        fa = fa[fa.sum(1) > 0]
        s["kit_all"] = fa.mean(0) if len(fa) else None
        s["id"] = k
        s["obs"] = obs
        s["pm"] = ps.mean(0)
        s["psd"] = ps.std(0).mean() if len(ps) > 1 else 0.0
        s["conf"] = float(cs.mean())
        s["amb"] = float(np.mean([amb[t][d] for t, d in obs]))
        s["dur"] = int(ts[-1] - ts[0] + 1)
        rows.append(s)
    rows.sort(key=lambda r: (r["t0"], r["t1"], r["id"]))
    return rows


def kit_of(r):
    return r["kit"] if r["kit"] is not None else r["kit_all"]


def pair_cost(a, b):
    gap = b["t0"] - a["t1"]
    if gap < 1 or gap > MAXGAP:
        return None
    g = gap * DT
    d = float(np.hypot(*(b["p0"] - a["p1"])))
    if d > BASE_M + K2.P.v_max * g:
        return None
    h = min(g, K2.P.vel_horizon)
    ea = np.hypot(*(a["p1"] + a["v1"] * h - b["p0"]))
    eb = np.hypot(*(b["p0"] - b["v0"] * h - a["p1"]))
    e = float(min(0.5 * (ea + eb), d))
    ka, kb = kit_of(a), kit_of(b)
    kd = float(hist_dist(ka, kb)) if ka is not None and kb is not None else -1.0
    cost = e / (K2.P.sig0 + K2.P.sig_v * g) + K2.P.w_gap * g + K2.P.w_kit * (kd if kd >= 0 else 0.15)
    return dict(cost=cost, g=g, d=d, e=e, kd=kd)


def canvas_inputs(rows, n, ours_of=None, k2_of=None, k2_prev=None):
    Pn = len(rows)
    tf = np.zeros((Pn, FT), np.float32)
    cand = np.full((Pn, K), -1, np.int64)
    pf = np.zeros((Pn, K, FP), np.float32)
    kit = np.zeros((Pn, 48), np.float32)
    kv = np.zeros(Pn, np.float32)
    for i, r in enumerate(rows):
        kk = kit_of(r)
        if kk is not None:
            kit[i], kv[i] = kk, 1.0
        tf[i, :16] = [r["t0"] / n, r["t1"] / n, np.log1p(r["n"]) / 6, np.log1p(r["dur"]) / 6, r["n"] / r["dur"],
                      r["conf"], r["amb"], r["p0"][0] / 105, r["p0"][1] / 68, r["p1"][0] / 105, r["p1"][1] / 68,
                      r["v0"][0] / 8, r["v0"][1] / 8, r["v1"][0] / 8, r["v1"][1] / 8, (r["kit"] is not None) * 1.0]
        tf[i, 16:] = kit[i]
    t0 = np.array([r["t0"] for r in rows])
    t1 = np.array([r["t1"] for r in rows])
    for i, b in enumerate(rows):
        cs = []
        for j in np.flatnonzero((t1[:i] < t0[i]) & (t0[i] - t1[:i] <= MAXGAP)):
            c = pair_cost(rows[j], b)
            if c is not None:
                cs.append((c["cost"], j, c))
        cs.sort(key=lambda z: z[0])
        for k, (cost, j, c) in enumerate(cs[:K]):
            a = rows[j]
            cand[i, k] = j
            same_o = float(ours_of is not None and ours_of[a["id"]] == ours_of[b["id"]])
            same_k = float(k2_of is not None and k2_of[a["id"]] == k2_of[b["id"]])
            k2p = float(k2_prev is not None and k2_prev.get(b["id"]) == a["id"])
            pf[i, k] = [c["g"] / 4, c["d"] / 20, c["d"] / (BASE_M + K2.P.v_max * c["g"]),
                        c["e"] / (K2.P.sig0 + K2.P.sig_v * c["g"]) / 3, max(c["kd"], 0), float(c["kd"] >= 0),
                        cost / 5, k / K, np.log1p(a["n"]) / 6, a["conf"], same_o, same_k, k2p,
                        len(cs) / 30]
    return dict(tf=tf, cand=cand, pf=pf, t0=t0, t1=t1, pm=np.array([r["pm"] for r in rows], np.float32),
                kit=kit, kv=kv)


def targets(rows, cand, maj):
    Pn = len(rows)
    y = np.full(Pn, NEW, np.int64)
    claimed = set()
    last = {}
    for i, r in enumerate(rows):
        g = maj[r["id"]]
        if g < 0:
            y[i] = JUNK
            continue
        prev = [(t1, j) for t1, j in last.get(g, []) if t1 < r["t0"] and j not in claimed]
        if prev:
            j = max(prev)[1]
            w = np.flatnonzero(cand[i] == j)
            if len(w):
                y[i] = int(w[0])
                claimed.add(j)
        last.setdefault(g, []).append((r["t1"], i))
    return y


def team_inputs(rows, det_team, cand):
    Pn = len(rows)
    tv = np.zeros((Pn, 3), np.float32)
    conf_team = np.full(Pn, -1)
    for i, r in enumerate(rows):
        v = np.zeros(2)
        for t, d in r["obs"]:
            tm = det_team[t][d]
            if tm >= 0:
                v[tm] += 1
        s = v.sum()
        if s:
            tv[i, :2] = v / s
        tv[i, 2] = np.log1p(s) / 6
        if s >= 5 and v.max() >= 0.85 * s:
            conf_team[i] = int(np.argmax(v))
    pt = np.zeros((Pn, K, 2), np.float32)
    for i in range(Pn):
        for k in range(K):
            j = cand[i, k]
            if j >= 0 and conf_team[i] >= 0 and conf_team[j] >= 0:
                pt[i, k, 0] = float(conf_team[i] != conf_team[j])
                pt[i, k, 1] = float(conf_team[i] == conf_team[j])
    return tv, pt


EMBD = "/mnt/x/Data/AI/more-than-a-hack/trackK2/reid"


def load_emb(name, xy, keep=None):
    z = np.load(os.path.join(EMBD, "emb_h3_%s.npz" % name))
    full = {}
    for f, d, e in zip(z["frame"], z["det"], z["emb"].astype(np.float32)):
        full[(int(f), int(d))] = e
    out = []
    for t in range(len(xy)):
        idx = np.flatnonzero(keep[t]) if keep is not None else np.arange(len(xy[t]))
        a = np.full((len(idx), 128), np.nan, np.float32)
        for j, d in enumerate(idx):
            e = full.get((t, int(d)))
            if e is not None:
                a[j] = e
        out.append(a)
    return out


def emb_inputs(rows, embl, cand):
    Pn = len(rows)
    m = np.full((Pn, 128), np.nan, np.float32)
    for i, r in enumerate(rows):
        es = [embl[t][d] for t, d in r["obs"] if np.isfinite(embl[t][d][0])]
        if es:
            v = np.mean(es, 0)
            m[i] = v / max(np.linalg.norm(v), 1e-9)
    pe = np.zeros((Pn, K, 2), np.float32)
    ok = np.isfinite(m[:, 0])
    for i in range(Pn):
        for k in range(K):
            j = cand[i, k]
            if j >= 0 and ok[i] and ok[j]:
                pe[i, k] = [1.0 - float(m[i] @ m[j]), 1.0]
    return pe


def clip_canvas(res):
    pieces = res["pieces"]
    rows = tracklet_table(pieces, res["xy"], res["cf"], res["ft"])
    ours_of, k2_of, k2_prev = {}, {}, {}
    o_map = {o: k for k, obs in res["ours"].items() for o in obs}
    for r in rows:
        ours_of[r["id"]] = o_map.get(r["obs"][0], -1)
    km = {o: k for k, obs in res["k2"].items() for o in obs}
    for r in rows:
        k2_of[r["id"]] = km.get(r["obs"][0], -1)
    by = {}
    for r in rows:
        by.setdefault(k2_of[r["id"]], []).append(r)
    for k, rs in by.items():
        rs.sort(key=lambda r: r["t0"])
        for a, b in zip(rs[:-1], rs[1:]):
            k2_prev[b["id"]] = a["id"]
    X = canvas_inputs(rows, res["n"], ours_of, k2_of, k2_prev)
    X["ours_id"] = np.array([ours_of[r["id"]] for r in rows])
    X["k2_id"] = np.array([k2_of[r["id"]] for r in rows])
    if "maj" in res:
        X["y"] = targets(rows, X["cand"], res["maj"])
    if res.get("det_team") is not None:
        X["team"], X["pteam"] = team_inputs(rows, res["det_team"], X["cand"])
    if res.get("embl") is not None:
        X["pemb"] = emb_inputs(rows, res["embl"], X["cand"])
    return rows, X


def decode(rows, cand, tok, drop_junk=False):
    Pn = len(rows)
    root = list(range(Pn))
    for i in range(Pn):
        t = int(tok[i])
        if t < K and cand[i, t] >= 0:
            root[i] = root[int(cand[i, t])]
    out = {}
    for i in range(Pn):
        if drop_junk and int(tok[i]) == JUNK:
            continue
        out.setdefault(root[i], []).extend(rows[i]["obs"])
    return {k: sorted(v) for k, v in out.items()}
