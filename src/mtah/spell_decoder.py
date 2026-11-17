import json
import os
import sys

import numpy as np

from mtah import pipe_ens2
from mtah.decode import viterbi_tv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def load_dev(oof_dir, split_file="notes/avo/split_v3.json", folds=4):
    from mtah import eval_ext
    cfg = eval_ext.resolve({"split_file": split_file, "folds": folds,
                            "data": {"dead": ["idsse-ctrl", "stv2"],
                                     "team": ["idsse-ctrl", "metrica", "pff", "skillcorner", "stv2"]}})
    P = eval_ext.plan(cfg, "dev")
    S = P["split"]
    out = []
    for k, fold in enumerate(P["folds"]):
        for n in fold:
            z = np.load(os.path.join(oof_dir, "oof_%s.npz" % n))
            p = {h: z[h].astype(np.float64) for h in z.files if h not in ("label", "half")}
            y, half = eval_ext.truth(n, eval_ext.label_key(n, S))
            out.append((n, eval_ext.src_of(n), k, p, half, y))
    return out, S


def score_match(name, pred, S):
    from mtah import eval_ext
    if os.path.join(ROOT, "evaluation") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    return eval_ext.score(name, pred, S)


def combined_heads(p, half, cfg):
    if "lgb_team" not in p:
        return {k: p[k] for k in pipe_ens2.HEADS}
    w = dict(cfg.get("w_nn") or {})
    g = cfg.get("dead_gate")
    if g:
        ok = half > 0
        c = float(np.corrcoef(p["lgb_dead"][ok], p["nn_dead"][ok])[0, 1])
        lo, hi = float(g.get("lo", 0.7)), float(g.get("hi", 0.85))
        gate = float(np.clip((c - lo) / (hi - lo), 0.0, 1.0)) if np.isfinite(c) else 0.0
        for k in g.get("heads", ["dead"]):
            w[k] = float(g.get("w", 0.5)) * gate
    q = pipe_ens2._combine(p, w, cfg.get("clip"), bool(cfg.get("logit_avg", False)))
    if cfg.get("w_nnb") and "nnb_dead" in p:
        for k, wk in cfg["w_nnb"].items():
            q[k] = (1 - float(wk)) * q[k] + float(wk) * p["nnb_" + k]
    if "nn_raw_team" in p:
        wr = dict(cfg.get("w_raw") or {})
        g = cfg.get("raw_gate")
        if g:
            ok = half > 0
            c = float(np.corrcoef(p["lgb_dead"][ok], p["nn_raw_dead"][ok])[0, 1])
            lo, hi = float(g.get("lo", 0.7)), float(g.get("hi", 0.85))
            gate = float(np.clip((c - lo) / (hi - lo), 0.0, 1.0)) if np.isfinite(c) else 0.0
            for k in g.get("heads", ["dead", "on", "off"]):
                wr[k] = float(g.get("w", 1 / 3)) * gate
        for k in pipe_ens2.HEADS:
            wk = float(wr.get(k, 0.0))
            if wk:
                q[k] = (1 - wk) * q[k] + wk * p["nn_raw_" + k]
    return q


def _lg(x, eps=1e-6):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def decode_q(q, half, switch, dead, beta, team_temp=1.0, dead_temp=1.0, dead_off=None, prior_d=0.0):
    team, dd = q["team"], q["dead"]
    if team_temp != 1.0:
        team = 1 / (1 + np.exp(-team_temp * _lg(team)))
    if dead_temp != 1.0 or prior_d:
        dd = 1 / (1 + np.exp(-(dead_temp * _lg(dd) + prior_d)))
    prob = np.stack([(1 - dd) * team, (1 - dd) * (1 - team), dd], 1)
    lp = np.log(np.clip(prob, 1e-4, 1))
    d_off = dead if dead_off is None else dead_off
    out = np.full(len(prob), 3, np.int8)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        n = len(idx)
        c = np.zeros((n, 3, 3))
        s = pipe_ens2._term(n, switch, q["sw"][idx], beta)
        on = pipe_ens2._term(n, dead, q["on"][idx], beta)
        off = pipe_ens2._term(n, d_off, q["off"][idx], beta)
        c[:, 0, 1] = c[:, 1, 0] = s
        c[:, 0, 2] = c[:, 1, 2] = on
        c[:, 2, 0] = c[:, 2, 1] = off
        out[idx] = viterbi_tv(lp[idx], c)
    return out


def spells_of(codes, half):
    out = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        a0, b0 = int(idx[0]), int(idx[-1]) + 1
        w = codes[a0:b0]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bd = np.concatenate(([0], br, [len(w)]))
        for i in range(len(bd) - 1):
            k = int(w[bd[i]])
            if k in (0, 1):
                out.append((a0 + int(bd[i]), a0 + int(bd[i + 1]), k))
    return out


FAMILY = [dict(switch=2, dead=30, beta=3.0)]
for _sw in (0.5, 1.0, 2.0, 4.0, 8.0):
    for _dd in (10.0, 30.0, 80.0):
        for _b in (2.0, 3.0, 4.0):
            f = dict(switch=_sw, dead=_dd, beta=_b)
            if f != FAMILY[0]:
                FAMILY.append(f)
for _tt in (0.6, 1.6):
    FAMILY.append(dict(switch=2, dead=30, beta=3.0, team_temp=_tt))
for _pd in (-1.0, 1.0):
    FAMILY.append(dict(switch=2, dead=30, beta=3.0, prior_d=_pd))


def candidates(q, half, family=FAMILY):
    seen = {}
    base = None
    for i, f in enumerate(family):
        codes = decode_q(q, half, **f)
        if i == 0:
            base = codes
        for s in spells_of(codes, half):
            seen[s] = seen.get(s, 0) + 1
    keys = sorted(seen)
    C = np.array(keys, dtype=np.int64).reshape(-1, 3)
    votes = np.array([seen[k] for k in keys], dtype=np.float64)
    return C, votes, base


def max_iou(C, gt_spells):
    out = np.zeros(len(C))
    if not len(gt_spells):
        return out
    G = np.array(gt_spells, dtype=np.int64)
    order = np.argsort(G[:, 0])
    G = G[order]
    for team in (0, 1):
        g = G[G[:, 2] == team]
        if not len(g):
            continue
        m = np.flatnonzero(C[:, 2] == team)
        if not len(m):
            continue
        a, b = C[m, 0], C[m, 1]
        lo = np.searchsorted(g[:, 1], a, side="right")
        hi = np.searchsorted(g[:, 0], b, side="left")
        best = np.zeros(len(m))
        for j in range(len(m)):
            for t in range(lo[j], hi[j]):
                inter = min(b[j], g[t, 1]) - max(a[j], g[t, 0])
                if inter > 0:
                    uni = max(b[j], g[t, 1]) - min(a[j], g[t, 0])
                    v = inter / uni
                    if v > best[j]:
                        best[j] = v
        out[m] = best
    return out


def gt_spells_of(y, half):
    yy = y.astype(np.int64).copy()
    yy[half == 0] = 3
    return spells_of(yy, half)


def select(C, w):
    import bisect
    keep = np.flatnonzero(w > 0)
    if not len(keep):
        return keep
    order = keep[np.argsort(C[keep, 1], kind="stable")]
    stops = {0: [], 1: []}
    best = {0: [], 1: []}
    arg = {0: [], 1: []}
    val = np.zeros(len(order))
    prev = np.full(len(order), -1)
    for r, i in enumerate(order):
        a, b, t = int(C[i, 0]), int(C[i, 1]), int(C[i, 2])
        cand_v, cand_p = 0.0, -1
        for tt, lim in ((1 - t, a), (t, a - 1)):
            j = bisect.bisect_right(stops[tt], lim) - 1
            if j >= 0 and best[tt][j] > cand_v:
                cand_v, cand_p = best[tt][j], arg[tt][j]
        val[r] = cand_v + w[i]
        prev[r] = cand_p
        st, bs, ag = stops[t], best[t], arg[t]
        st.append(b)
        if bs and bs[-1] >= val[r]:
            bs.append(bs[-1])
            ag.append(ag[-1])
        else:
            bs.append(val[r])
            ag.append(r)
    r = int(np.argmax(val))
    if val[r] <= 0:
        return np.array([], dtype=np.int64)
    chosen = []
    while r >= 0:
        chosen.append(order[r])
        r = prev[r]
    return np.array(sorted(chosen, key=lambda i: C[i, 0]), dtype=np.int64)


def render(C, sel, half, n):
    out = np.full(n, 3, np.int8)
    out[half > 0] = 2
    for i in sel:
        out[C[i, 0]:C[i, 1]] = C[i, 2]
    return out


def _cs(x):
    return np.concatenate(([0.0], np.cumsum(x)))


def _wmean(cs, a, b):
    a = np.clip(a, 0, len(cs) - 1)
    b = np.clip(b, 0, len(cs) - 1)
    return (cs[b] - cs[a]) / np.maximum(b - a, 1)


def _wmax(x, a, b):
    out = np.zeros(len(a))
    n = len(x)
    for i in range(len(a)):
        lo, hi = max(int(a[i]), 0), min(int(b[i]), n)
        out[i] = x[lo:hi].max() if hi > lo else 0.0
    return out


FEAT_NAMES = None


def features(C, votes, base, q, p, half, n_family):
    global FEAT_NAMES
    n = len(half)
    a, b, t = C[:, 0], C[:, 1], C[:, 2]
    L = (b - a).astype(np.float64)
    team = q["team"]
    own_sign = np.where(t == 0, 1.0, -1.0)
    F = {}
    F["log_len"] = np.log(L)
    F["votes"] = votes / n_family
    srcs = {"q": (q["team"], q["dead"])}
    if "lgb_team" in p:
        srcs["lgb"] = (p["lgb_team"], p["lgb_dead"])
        srcs["nn"] = (p["nn_team"], p["nn_dead"])
    if "nn_raw_team" in p:
        srcs["raw"] = (p["nn_raw_team"], p["nn_raw_dead"])
    for nm, (tm, dd) in srcs.items():
        lt = _lg(tm)
        cs_lt, cs_d = _cs(lt), _cs(dd)
        cs_pos = _cs((tm > 0.5).astype(np.float64))
        F[nm + "_own_lo"] = own_sign * _wmean(cs_lt, a, b)
        frac_h = _wmean(cs_pos, a, b)
        F[nm + "_own_frac"] = np.where(t == 0, frac_h, 1 - frac_h)
        F[nm + "_dead_in"] = _wmean(cs_d, a, b)
        if nm == "q":
            for W in (25, 125):
                F["q_own_lo_pre%d" % W] = own_sign * _wmean(cs_lt, a - W, a)
                F["q_own_lo_post%d" % W] = own_sign * _wmean(cs_lt, b, b + W)
                F["q_dead_pre%d" % W] = _wmean(cs_d, a - W, a)
                F["q_dead_post%d" % W] = _wmean(cs_d, b, b + W)
            th = np.maximum((b - a) // 3, 1)
            F["q_own_lo_head"] = own_sign * _wmean(cs_lt, a, a + th)
            F["q_own_lo_tail"] = own_sign * _wmean(cs_lt, b - th, b)
            mn = np.zeros(len(C))
            for i in range(len(C)):
                seg = lt[a[i]:b[i]] * own_sign[i]
                mn[i] = np.percentile(seg, 10) if len(seg) else 0.0
            F["q_own_lo_p10"] = mn
    for hd in ("sw", "on", "off"):
        x = q[hd]
        F[hd + "_start"] = _wmax(x, a - 12, a + 13)
        F[hd + "_end"] = _wmax(x, b - 12, b + 13)
        F[hd + "_inside"] = _wmax(x, a + 25, b - 25)
        F[hd + "_mean_in"] = _wmean(_cs(x), a, b)
    bs = spells_of(base, half)
    bset = set(bs)
    F["in_base"] = np.array([tuple(c) in bset for c in C.tolist()], dtype=np.float64)
    Bm = np.array(bs, dtype=np.int64).reshape(-1, 3)
    F["base_iou"] = max_iou(C, [tuple(x) for x in Bm]) if len(Bm) else np.zeros(len(C))
    cs_bd = _cs((base == 2).astype(np.float64))
    F["base_dead_in"] = _wmean(cs_bd, a, b)
    cs_bown = _cs((base == 0).astype(np.float64))
    bh = _wmean(cs_bown, a, b)
    cs_ba = _cs((base == 1).astype(np.float64))
    ba = _wmean(cs_ba, a, b)
    F["base_own_in"] = np.where(t == 0, bh, ba)
    F["base_opp_in"] = np.where(t == 0, ba, bh)
    ok = half > 0
    if "lgb_dead" in p:
        c = float(np.corrcoef(p["lgb_dead"][ok], p["nn_dead"][ok])[0, 1])
        F["m_dead_corr"] = np.full(len(C), c if np.isfinite(c) else 0.0)
        F["m_team_corr"] = np.full(len(C), float(np.corrcoef(p["lgb_team"][ok], p["nn_team"][ok])[0, 1]))
    F["m_dead_mean"] = np.full(len(C), float(q["dead"][ok].mean()))
    F["m_team_conf"] = np.full(len(C), float(np.abs(q["team"][ok] - 0.5).mean()))
    F["m_base_rate"] = np.full(len(C), len(bs) / max(ok.sum(), 1) * 25 * 60)
    FEAT_NAMES = list(F)
    return np.stack([F[k] for k in FEAT_NAMES], 1)


def target(iou):
    return np.mean([iou >= t for t in IOUS], 0)


def context_features(C, votes, base, half):
    a, b, t = C[:, 0], C[:, 1], C[:, 2]
    M = len(C)
    order = np.argsort(a, kind="stable")
    F = {k: np.zeros(M) for k in ("ov_same_n", "ov_other_n", "ov_same_maxv", "ov_other_maxv", "ov_other_frac")}
    sa = a[order]
    for i in range(M):
        hi = np.searchsorted(sa, b[i], side="left")
        js = order[:hi]
        js = js[(b[js] > a[i]) & (js != i)]
        if not len(js):
            continue
        same = js[t[js] == t[i]]
        oth = js[t[js] != t[i]]
        F["ov_same_n"][i] = len(same)
        F["ov_other_n"][i] = len(oth)
        if len(same):
            F["ov_same_maxv"][i] = votes[same].max()
        if len(oth):
            F["ov_other_maxv"][i] = votes[oth].max()
            inter = np.minimum(b[oth], b[i]) - np.maximum(a[oth], a[i])
            F["ov_other_frac"][i] = float((inter * votes[oth]).sum()) / (b[i] - a[i]) / max(votes.max(), 1)
    bnd = np.flatnonzero((base[1:] != base[:-1]) & (half[1:] > 0) & (half[:-1] > 0)) + 1
    if len(bnd):
        for nm, x in (("d_start_bnd", a), ("d_end_bnd", b)):
            j = np.clip(np.searchsorted(bnd, x), 1, len(bnd) - 1)
            F[nm] = np.log1p(np.minimum(np.abs(bnd[j] - x), np.abs(bnd[j - 1] - x)))
    bs = np.array(spells_of(base, half), dtype=np.int64).reshape(-1, 3)
    for k in ("prev_gap", "prev_same", "prev_len", "next_gap", "next_same", "next_len"):
        F[k] = np.zeros(M)
    if len(bs):
        stops = bs[:, 1]
        j = np.searchsorted(stops, a, side="right") - 1
        ok = j >= 0
        jj = np.clip(j, 0, len(bs) - 1)
        F["prev_gap"] = np.where(ok, np.log1p(np.maximum(a - bs[jj, 1], 0)), 8.0)
        F["prev_same"] = np.where(ok, (bs[jj, 2] == t).astype(float), 0.5)
        F["prev_len"] = np.where(ok, np.log(np.maximum(bs[jj, 1] - bs[jj, 0], 1)), 0.0)
        j = np.searchsorted(bs[:, 0], b, side="left")
        ok = j < len(bs)
        jj = np.clip(j, 0, len(bs) - 1)
        F["next_gap"] = np.where(ok, np.log1p(np.maximum(bs[jj, 0] - b, 0)), 8.0)
        F["next_same"] = np.where(ok, (bs[jj, 2] == t).astype(float), 0.5)
        F["next_len"] = np.where(ok, np.log(np.maximum(bs[jj, 1] - bs[jj, 0], 1)), 0.0)
    names = list(F)
    return np.stack([F[k] for k in names], 1), names


def frame_extra_features(C, extra):
    a, b = C[:, 0], C[:, 1]
    cols, names = [], []
    for k in sorted(extra):
        cs = _cs(np.asarray(extra[k], np.float64))
        cols += [_wmean(cs, a, b), _wmean(cs, a - 25, a + 25), _wmean(cs, b - 25, b + 25)]
        names += [k + "_in", k + "_start", k + "_end"]
    return np.stack(cols, 1), names


EXTRA_COLS = (("ballmiss", "b_x"), ("accmiss", "b_acc"), ("d1miss", "d1_h"))


def extras_from_view(X, names):
    out = {}
    for k, col in EXTRA_COLS:
        if col in names:
            out[k] = np.isnan(X[:, names.index(col)]).astype(np.float32)
    return out


def family_v2():
    fam = [dict(f) for f in FAMILY]
    for sw in (0.25, 16.0):
        for dd in (10.0, 30.0):
            fam.append(dict(switch=sw, dead=dd, beta=3.0))
    for dd in (3.0, 200.0):
        fam.append(dict(switch=2.0, dead=dd, beta=3.0))
    for tt in (0.6, 1.6):
        for sw in (0.5, 8.0):
            fam.append(dict(switch=sw, dead=30.0, beta=3.0, team_temp=tt))
    for pd in (-2.0, 2.0):
        fam.append(dict(switch=1.0, dead=10.0, beta=3.0, prior_d=pd))
    return fam


DROP_DEFAULT = ("m_dead_corr", "m_team_corr", "m_dead_mean", "m_team_conf", "m_base_rate")
LGB = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6, seed=0,
           deterministic=True, force_col_wise=True)
LAMS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45)


def half_key(half):
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(np.asarray(half, np.int8)).tobytes()).hexdigest()


MEMBER_FAMILY = [dict(switch=sw, dead=30.0, beta=3.0) for sw in (1.0, 2.0, 4.0)]


MEMBER_FAMILY2 = [dict(switch=sw, dead=dd, beta=3.0) for sw in (0.5, 1.0, 2.0, 4.0, 8.0) for dd in (10.0, 30.0)]


def member_candidates(C, votes, p, half, dec, level=1):
    fam = MEMBER_FAMILY if level == 1 else MEMBER_FAMILY2
    mem = [({"team": 0.0, "sw": 0.0}, {"team": 0.0, "sw": 0.0}), ({"team": 1.0, "sw": 1.0}, {"team": 0.0, "sw": 0.0})]
    if "nn_raw_team" in p:
        mem.append(({"team": 0.0, "sw": 0.0}, {"team": 1.0, "sw": 1.0}))
    nm = len(mem)
    seen = {tuple(c): [v] + [0.0] * nm for c, v in zip(C.tolist(), votes.tolist())}
    for j, (wn, wr) in enumerate(mem):
        d2 = dict(dec)
        d2["w_nn"] = dict(wn)
        if "nn_raw_team" in p:
            d2["w_raw"] = dict(wr)
        qm = combined_heads(p, half, d2)
        for f in fam:
            for sp in spells_of(decode_q(qm, half, **f), half):
                seen.setdefault(sp, [0.0] * (1 + nm))[1 + j] += 1.0 / len(fam)
    keys = sorted(seen)
    Cn = np.array(keys, dtype=np.int64).reshape(-1, 3)
    V = np.array([seen[k] for k in keys])
    return Cn, V[:, 0], V[:, 1:]


def candidate_table(p, half, dec, family=None, extra=None, ctx=True, shift=False, members=False):
    family = family_names(dec.get("family", "v1"), dec) if family is None else family
    q = combined_heads(p, half, dec)
    C, votes, base = candidates(q, half, family)
    mv = None
    if members and "lgb_team" in p:
        C, votes, mv = member_candidates(C, votes, p, half, dec, level=int(members))
    if shift:
        C, votes = shift_candidates(C, votes, base, q, half)
    X1 = features(C, votes, base, q, p, half, len(family))
    cols, names = [X1], list(FEAT_NAMES)
    if mv is not None:
        cols.append(mv)
        names += ["votes_lgb", "votes_nn", "votes_raw"][:mv.shape[1]]
    if ctx:
        X2, n2 = context_features(C, votes / len(family), base, half)
        cols.append(X2)
        names += n2
    if extra:
        X3, n3 = frame_extra_features(C, extra)
        cols.append(X3)
        names += n3
    return C, np.hstack(cols).astype(np.float32), names, base


def family_names(name, dec=None):
    fam = FAMILY if name == "v1" else family_v2()
    if dec is None:
        return fam
    b = dict(switch=dec["switch"], dead=dec["dead"], beta=float(dec["beta"]))
    return [b] + [f for f in fam if f != b]


class _Avg:
    def __init__(self, ms):
        self.ms = ms

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], 0) if len(self.ms) > 1 else self.ms[0].predict(X)

    def save(self, stem):
        for s, m in enumerate(self.ms):
            m.save_model(stem + ("" if s == 0 else "_s%d" % s) + ".txt")


def _fit(Xs, ys, rounds, seeds=1):
    import lightgbm as lgb
    X, y = np.concatenate(Xs), np.concatenate(ys)
    return _Avg([lgb.train(dict(LGB, seed=s), lgb.Dataset(X, y), rounds) for s in range(seeds)])


def decode_with(C, qv, lam, half):
    sel = select(C, qv - lam)
    return render(C, sel, half, len(half))


def target_rule(C, y, half):
    alive = (half > 0) & (y != 2)
    gts = gt_spells_of(y, half)
    pieces, owner = [], []
    for i, (a, b, t) in enumerate(C.tolist()):
        w = alive[a:b]
        if not w.any():
            continue
        e = np.diff(np.concatenate(([0], w.astype(np.int8), [0])))
        for u, v in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            pieces.append((a + int(u), a + int(v), t))
            owner.append(i)
    best = np.zeros(len(C))
    if pieces:
        iou = max_iou(np.array(pieces, dtype=np.int64), gts)
        np.maximum.at(best, np.array(owner), iou)
    return target(best)


def build_tables(oof_dir, dec, split_file, ctx=False, shift=False, members=False, log=print):
    data, S = load_dev(oof_dir, split_file)
    drop = set(DROP_DEFAULT)
    tab, names = {}, None
    for (n, src, k, p, half, y) in data:
        C, X, names, base = candidate_table(p, half, dec, ctx=ctx, shift=shift, members=members)
        keep = [i for i, f in enumerate(names) if f not in drop]
        tab[n] = dict(src=src, fold=k, C=C, X=X[:, keep], t_full=target(max_iou(C, gt_spells_of(y, half))),
                      t_rule=target_rule(C, y, half), half=half, y=y, key=half_key(half))
        log("  table %s %d candidates" % (n, len(C)))
    return tab, [f for f in names if f not in drop], S


def train_bundle(oof_dir, dec, out, split_file="notes/avo/split_v3.json", rounds=300, lams=LAMS, ctx=False,
                 shift=False, members=False, seeds=1, target_mode="full", train_sources=None, tables=None, log=print):
    import collections
    from mtah import eval_ext
    if tables is not None and os.path.exists(tables):
        import pickle
        tab, feat, S = pickle.load(open(tables, "rb"))
        log("  tables from %s" % tables)
    else:
        tab, feat, S = build_tables(oof_dir, dec, split_file, ctx, shift, members, log)
        if tables is not None:
            import pickle
            pickle.dump((tab, feat, S), open(tables, "wb"))
    folds = sorted({v["fold"] for v in tab.values()})

    def tgt(n):
        if target_mode == "rule" and eval_ext.score_rule_of(n, S):
            return tab[n]["t_rule"]
        return tab[n]["t_full"]

    def fit(ns):
        ns = [n for n in ns if train_sources is None or tab[n]["src"] in train_sources]
        return _fit([tab[n]["X"] for n in ns], [tgt(n) for n in ns], rounds, seeds)

    def f1(n, qv, lam):
        t = tab[n]
        pred = decode_with(t["C"], qv, lam, t["half"])
        rule = eval_ext.score_rule_of(n, S)
        if rule:
            return eval_ext.rule_spell_f1(n, pred, t["y"], t["half"], rule)
        return eval_ext.spell_f1(pred, t["y"], t["half"])

    def best_lam(qd):
        res = {}
        for lam in lams:
            by = collections.defaultdict(list)
            for n, qv in qd.items():
                by[tab[n]["src"]].append(f1(n, qv, lam))
            res[lam] = float(np.mean([np.mean(v) for v in by.values()]))
        return max(res, key=res.get), res

    def crossfit(fs):
        qd = {}
        for j in fs:
            m = fit([n for n in tab if tab[n]["fold"] in fs and tab[n]["fold"] != j])
            qd.update({n: m.predict(tab[n]["X"]) for n in tab if tab[n]["fold"] == j})
        return qd

    os.makedirs(out, exist_ok=True)
    meta = {"oof_dir": oof_dir, "decoder": dec, "split_file": split_file, "rounds": rounds, "lams": list(lams),
            "features": feat, "ctx": ctx, "shift": shift, "members": members, "seeds": seeds, "lgb": LGB,
            "target_mode": target_mode, "train_sources": train_sources, "folds": {}, "index": {}}
    for k in folds:
        others = [j for j in folds if j != k]
        lam_k, res = best_lam(crossfit(others))
        fit([n for n in tab if tab[n]["fold"] != k]).save(os.path.join(out, "model_%d" % k))
        meta["folds"][str(k)] = {"lam": lam_k, "curve": res}
        log("  fold %d lam %.2f %s" % (k, lam_k, {l: round(v, 4) for l, v in res.items()}))
    lam_all, res = best_lam(crossfit(folds))
    fit(list(tab)).save(os.path.join(out, "model_all"))
    meta["all"] = {"lam": lam_all, "curve": res}
    meta["index"] = {tab[n]["key"]: {"name": n, "fold": tab[n]["fold"]} for n in tab}
    json.dump(meta, open(os.path.join(out, "bundle.json"), "w"), indent=1)
    log("  all lam %.2f %s" % (lam_all, {l: round(v, 4) for l, v in res.items()}))
    return meta


_BUNDLES = {}


def _bundle(path):
    if path not in _BUNDLES:
        import lightgbm as lgb
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        meta = json.load(open(os.path.join(full, "bundle.json")))
        sfx = [""] + ["_s%d" % s for s in range(1, int(meta.get("seeds", 1)))]
        models = {k: _Avg([lgb.Booster(model_file=os.path.join(full, "model_%s%s.txt" % (k, x))) for x in sfx])
                  for k in list(meta["folds"]) + ["all"]}
        _BUNDLES[path] = (meta, models)
    return _BUNDLES[path]


def decode(p, half, cfg):
    rs = cfg["rescore"]
    meta, models = _bundle(rs["bundle"])
    dec = {k: v for k, v in cfg.items() if k != "rescore"}
    if dec != meta["decoder"]:
        raise ValueError("rescore bundle was trained on decoder %s, config has %s" % (meta["decoder"], dec))
    hit = meta["index"].get(half_key(half))
    which = str(hit["fold"]) if hit else "all"
    lam = meta["folds"][which]["lam"] if hit else meta["all"]["lam"]
    if rs.get("lam") is not None:
        lam = rs["lam"]
    lam = float(lam)
    C, X, names, base = candidate_table(p, half, dec, ctx=meta["ctx"], shift=meta.get("shift", False),
                                        members=meta.get("members", False))
    X = X[:, [names.index(f) for f in meta["features"]]]
    qv = models[which].predict(X)
    return decode_with(C, qv, lam, half)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--oof", required=True)
    ap.add_argument("--config", required=True, help="config whose 'decoder' block (minus rescore) builds candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--ctx", action="store_true")
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--members", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--split-file", default="notes/avo/split_v3.json")
    ap.add_argument("--target", default="full", choices=["full", "rule"])
    ap.add_argument("--train-sources", default=None, help="comma list, e.g. idsse-ctrl,stv2")
    ap.add_argument("--tables", default=None, help="pickle cache of the candidate tables")
    a = ap.parse_args()
    if os.path.join(ROOT, "evaluation") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    dec = {k: v for k, v in json.load(open(a.config))["decoder"].items() if k != "rescore"}
    train_bundle(a.oof, dec, a.out, split_file=a.split_file, rounds=a.rounds, ctx=a.ctx, shift=a.shift,
                 members=a.members, seeds=a.seeds, target_mode=a.target,
                 train_sources=a.train_sources.split(",") if a.train_sources else None, tables=a.tables)


if __name__ == "__main__":
    main()


def shift_candidates(C, votes, base, q, half, W=25, shifts=(-12, 12)):
    bs = spells_of(base, half)
    bnd = q["sw"] + q["on"] + q["off"]
    hs = {}
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        hs[h] = (int(idx[0]), int(idx[-1]) + 1)
    seen = {tuple(c): v for c, v in zip(C.tolist(), votes.tolist())}
    for a, b, t in bs:
        lo, hi = hs[int(half[a])]

        def opts(x):
            o = {x}
            for s in shifts:
                o.add(x + s)
            l, r = max(lo, x - W), min(hi, x + W + 1)
            o.add(l + int(np.argmax(bnd[l:r])))
            return {min(max(v, lo), hi) for v in o}
        for a2 in opts(a):
            for b2 in opts(b):
                if b2 - a2 >= 5:
                    seen.setdefault((a2, b2, t), 0.0)
    keys = sorted(seen)
    return np.array(keys, dtype=np.int64).reshape(-1, 3), np.array([seen[k] for k in keys], dtype=np.float64)
