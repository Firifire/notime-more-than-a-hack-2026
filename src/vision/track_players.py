import numpy as np
from scipy.optimize import linear_sum_assignment

from vision.players_common import hist_dist

DT = 0.04


class P:
    t_hi = 0.45
    t_lo = 0.15
    sigma_a = 6.0
    sigma_m = 0.35
    gate_d2 = 13.8
    gate_max_m = 3.0
    col_w = 2.0
    col_max = 0.75
    n_init = 3
    max_age_in = 50
    max_age_out = 125
    v_decay = 0.96
    v_max = 10.0
    dup_m = 0.6
    st_gap = 75
    st_v = 7.0
    st_base = 1.0
    st_col = 0.5


class _Track:
    __slots__ = ("id", "x", "Pc", "hits", "age_lost", "confirmed", "obs", "feat", "nfeat",
                 "last_obs", "last_t")

    def __init__(self, tid, xy, t, di, feat):
        self.id = tid
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.Pc = np.diag([P.sigma_m ** 2, P.sigma_m ** 2, 4.0, 4.0])
        self.hits = 1
        self.age_lost = 0
        self.confirmed = False
        self.obs = [(t, di)]
        self.feat = feat.copy() if feat is not None else None
        self.nfeat = 1 if feat is not None else 0
        self.last_obs = np.array(xy, float)
        self.last_t = t


def _kf_mats():
    F = np.eye(4)
    F[0, 2] = F[1, 3] = DT
    q = P.sigma_a ** 2
    Q1 = np.array([[DT ** 4 / 4, DT ** 3 / 2], [DT ** 3 / 2, DT ** 2]]) * q
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    Hm = np.zeros((2, 4))
    Hm[0, 0] = Hm[1, 1] = 1
    R = np.eye(2) * P.sigma_m ** 2
    return F, Q, Hm, R


def run_tracker(frames, xy, conf, feat=None, in_view=None):
    F, Q, Hm, R = _kf_mats()
    tracks, done = [], []
    next_id = 0
    for fi, t in enumerate(frames):
        z, c = xy[fi], conf[fi]
        fe = feat[fi] if feat is not None else None
        for tr in tracks:
            if tr.age_lost > 0:
                tr.x[2:] *= P.v_decay
            tr.x = F @ tr.x
            tr.Pc = F @ tr.Pc @ F.T + Q
            sp = np.hypot(tr.x[2], tr.x[3])
            if sp > P.v_max:
                tr.x[2:] *= P.v_max / sp
        hi = np.flatnonzero(c >= P.t_hi)
        lo = np.flatnonzero((c >= P.t_lo) & (c < P.t_hi))
        unmatched_tr = list(range(len(tracks)))

        def associate(tr_idx, det_idx, strict):
            if not tr_idx or not len(det_idx):
                return [], tr_idx, list(det_idx)
            det_idx = np.asarray(det_idx)
            X = np.array([tracks[ti].x[:2] for ti in tr_idx])
            Pp = np.array([tracks[ti].Pc[:2, :2] for ti in tr_idx]) + R[None]
            det = Pp[:, 0, 0] * Pp[:, 1, 1] - Pp[:, 0, 1] * Pp[:, 1, 0]
            d = z[det_idx][None, :, :] - X[:, None, :]
            d2 = (Pp[:, None, 1, 1] * d[..., 0] ** 2 - 2 * Pp[:, None, 0, 1] * d[..., 0] * d[..., 1]
                  + Pp[:, None, 0, 0] * d[..., 1] ** 2) / det[:, None]
            eu = np.hypot(d[..., 0], d[..., 1])
            ok = (d2 < P.gate_d2) & (eu < (P.gate_max_m if not strict else 1.5))
            cost = d2 / P.gate_d2
            if fe is not None:
                has = np.array([tracks[ti].feat is not None for ti in tr_idx])
                if has.any():
                    TF = np.array([tracks[ti].feat if tracks[ti].feat is not None
                                   else np.zeros(fe.shape[1]) for ti in tr_idx])
                    cd = hist_dist(TF[:, None, :], fe[det_idx][None, :, :])
                    cd[~has] = 0.0
                    ok &= cd < P.col_max
                    cost = cost + P.col_w * cd
            C = np.where(ok, cost, 1e6)
            r, k = linear_sum_assignment(C)
            m = [(tr_idx[i], det_idx[j]) for i, j in zip(r, k) if C[i, j] < 1e5]
            mt = {a for a, _ in m}
            md = {b for _, b in m}
            return m, [ti for ti in tr_idx if ti not in mt], [d for d in det_idx if d not in md]

        conf_tr = [i for i in unmatched_tr if tracks[i].confirmed]
        tent_tr = [i for i in unmatched_tr if not tracks[i].confirmed]
        m1, rem_tr, rem_hi = associate(conf_tr, hi, False)
        m1b, rem_tent, rem_hi = associate(tent_tr, rem_hi, True)
        st2 = [i for i in rem_tr if tracks[i].age_lost == 0]
        m2, rem2, _ = associate(st2, lo, True)
        matches = m1 + m1b + m2
        matched_set = {a for a, _ in matches}
        for ti, di in matches:
            tr = tracks[ti]
            zz = z[di]
            if tr.age_lost > 0:
                gap = (t - tr.last_t) * DT
                v = (zz - tr.last_obs) / max(gap, DT)
                sp = np.hypot(*v)
                if sp > P.v_max:
                    v *= P.v_max / sp
                tr.x[2:] = v
            S = Hm @ tr.Pc @ Hm.T + R
            K = tr.Pc @ Hm.T @ np.linalg.inv(S)
            tr.x = tr.x + K @ (zz - Hm @ tr.x)
            tr.Pc = (np.eye(4) - K @ Hm) @ tr.Pc
            tr.hits += 1
            tr.age_lost = 0
            tr.obs.append((t, di))
            tr.last_obs = zz.copy()
            tr.last_t = t
            if not tr.confirmed and tr.hits >= P.n_init:
                tr.confirmed = True
            if fe is not None and c[di] >= P.t_hi and fe[di].sum() > 0:
                if tr.feat is None:
                    tr.feat = fe[di].copy()
                else:
                    a = 1.0 / min(tr.nfeat + 1, 20)
                    tr.feat = (1 - a) * tr.feat + a * fe[di]
                tr.nfeat += 1
        keep = []
        for i, tr in enumerate(tracks):
            if i in matched_set:
                keep.append(tr)
                continue
            if not tr.confirmed:
                continue
            tr.age_lost += 1
            lim = P.max_age_in
            if in_view is not None and not in_view(t, tr.x[None, :2])[0]:
                lim = P.max_age_out
            if tr.age_lost > lim:
                done.append(tr)
            else:
                keep.append(tr)
        tracks = keep
        for di in rem_hi:
            if tracks:
                pos = np.array([tr.x[:2] for tr in tracks])
                if np.min(np.hypot(*(pos - z[di]).T)) < P.dup_m:
                    continue
            tracks.append(_Track(next_id, z[di], t, di,
                                 fe[di] if fe is not None and fe[di].sum() > 0 else None))
            next_id += 1
    done += tracks
    return {tr.id: tr.obs for tr in done if tr.confirmed}


def tracklets_table(tracks, frames, xy, feat=None):
    fpos = {t: i for i, t in enumerate(frames)}
    out = []
    for tid, obs in tracks.items():
        ts = np.array([t for t, _ in obs])
        ps = np.array([xy[fpos[t]][d] for t, d in obs])
        f = None
        if feat is not None:
            fs = np.array([feat[fpos[t]][d] for t, d in obs])
            fs = fs[fs.sum(1) > 0]
            f = fs.mean(0) if len(fs) else None
        out.append(dict(id=tid, t0=ts[0], t1=ts[-1], p0=ps[:3].mean(0), p1=ps[-3:].mean(0),
                        feat=f, n=len(ts)))
    return out


def stitch(tracks, frames, xy, feat=None, extra_link=None):
    tab = sorted(tracklets_table(tracks, frames, xy, feat), key=lambda r: r["t0"])
    t0s = np.array([r["t0"] for r in tab])
    maxgap = 250 if extra_link is not None else P.st_gap
    cands = []
    for a in tab:
        lo = np.searchsorted(t0s, a["t1"], side="right")
        hi = np.searchsorted(t0s, a["t1"] + maxgap, side="right")
        for b in tab[lo:hi]:
            gap = b["t0"] - a["t1"]
            forced = extra_link is not None and extra_link(a["id"], b["id"])
            if gap > (P.st_gap if not forced else 250):
                continue
            d = np.hypot(*(b["p0"] - a["p1"]))
            allow = P.st_base + P.st_v * gap * DT
            if d > allow:
                continue
            cd = 0.0
            if a["feat"] is not None and b["feat"] is not None:
                cd = float(hist_dist(a["feat"], b["feat"]))
                if cd > P.st_col:
                    continue
            cost = d / allow + cd + gap / P.st_gap - (1.0 if forced else 0.0)
            cands.append((cost, a["id"], b["id"]))
    cands.sort()
    nxt, prv = {}, {}
    for cost, ia, ib in cands:
        if ia in nxt or ib in prv:
            continue
        nxt[ia] = ib
        prv[ib] = ia
    out = {}
    for tid in tracks:
        if tid in prv:
            continue
        obs, cur = list(tracks[tid]), tid
        while cur in nxt:
            cur = nxt[cur]
            obs += tracks[cur]
        out[tid] = obs
    return out


def densify(tracks, frames, xy, boxes=None, max_gap=125):
    fpos = {t: i for i, t in enumerate(frames)}
    res = {}
    nid = 0
    for tid, obs in tracks.items():
        obs = sorted(obs)
        segs = [[obs[0]]]
        for o in obs[1:]:
            if o[0] - segs[-1][-1][0] > max_gap:
                segs.append([o])
            else:
                segs[-1].append(o)
        for seg in segs:
            ts = np.array([t for t, _ in seg])
            ds = np.array([d for _, d in seg])
            ps = np.array([xy[fpos[t]][d] for t, d in seg])
            tt = np.arange(ts[0], ts[-1] + 1)
            px = np.interp(tt, ts, ps[:, 0])
            py = np.interp(tt, ts, ps[:, 1])
            st = np.full(len(tt), 2, np.int8)
            di = np.full(len(tt), -1)
            st[ts - ts[0]] = 1
            di[ts - ts[0]] = ds
            r = dict(t=tt, xy=np.c_[px, py], state=st, det=di, src=tid)
            if boxes is not None:
                bs = np.array([boxes[fpos[t]][d] for t, d in seg])
                r["box"] = np.stack([np.interp(tt, ts, bs[:, k]) for k in range(4)], 1)
            res[nid] = r
            nid += 1
    return res


def assign_teams(dense, frames, det_team, feat=None, det_gk=None, min_votes=3, win=1500,
                 min_match_frac=0.15, min_det_other=125):
    fpos = {t: i for i, t in enumerate(frames)}
    team, src, feats, gkf = {}, {}, {}, {}
    for k, tr in dense.items():
        votes = np.zeros(2)
        fs = []
        ngk = nm = 0
        for t, d in zip(tr["t"], tr["det"]):
            if d < 0:
                continue
            tm = det_team[fpos[t]][d]
            if tm >= 0:
                votes[tm] += 1
                nm += 1
                if det_gk is not None and det_gk[fpos[t]][d]:
                    ngk += 1
            if feat is not None:
                f = feat[fpos[t]][d]
                if f.sum() > 0:
                    fs.append(f)
        feats[k] = np.mean(fs, 0) if fs else None
        gkf[k] = ngk / max(nm, 1)
        ndet = sum(1 for d in tr["det"] if d >= 0)
        if ndet >= min_det_other and nm < min_match_frac * ndet:
            team[k], src[k] = -1, 3
        elif votes.sum() >= min_votes and votes.max() >= 0.6 * votes.sum():
            team[k], src[k] = int(np.argmax(votes)), 1
        else:
            team[k], src[k] = -1, 0
    if feat is None:
        return team, src
    anch = [(k, int(team[k]), 0.5 * (dense[k]["t"][0] + dense[k]["t"][-1]), feats[k])
            for k in dense if src[k] == 1 and feats[k] is not None and gkf[k] < 0.5]
    if not anch:
        return team, src
    am = np.array([a[2] for a in anch])
    at = np.array([a[1] for a in anch])
    af = np.array([a[3] for a in anch])
    cache = {}

    def protos(mid):
        key = int(mid // 250)
        if key in cache:
            return cache[key]
        sel = np.abs(am - mid) <= win
        if (at[sel] == 0).sum() < 3 or (at[sel] == 1).sum() < 3:
            sel = np.ones(len(am), bool)
        pr, th = [], []
        for c in (0, 1):
            fc = af[sel & (at == c)]
            p = np.median(fc, 0)
            p = p / max(p.sum(), 1e-9)
            pr.append(p)
            th.append(float(np.percentile(hist_dist(fc, p[None]), 95)))
        cache[key] = (pr, th)
        return cache[key]

    for k in dense:
        if feats[k] is None or src[k] == 3:
            continue
        mid = 0.5 * (dense[k]["t"][0] + dense[k]["t"][-1])
        pr, th = protos(mid)
        d_t = [float(hist_dist(feats[k], pr[c])) for c in (0, 1)]
        if src[k] == 1:
            if d_t[0] > th[0] and d_t[1] > th[1] and gkf[k] < 0.5:
                team[k], src[k] = -1, 3
            continue
        c = int(np.argmin(d_t))
        if d_t[c] <= th[c] and d_t[c] < 0.8 * d_t[1 - c]:
            team[k], src[k] = c, 2
        elif d_t[0] > th[0] and d_t[1] > th[1]:
            src[k] = 3
    return team, src


def _hota_alignment(gt_frames, pr_frames, sim_fn):
    alphas = np.arange(0.05, 0.99, 0.05)
    gids = np.unique(np.concatenate([g[0] for g in gt_frames] + [np.zeros(0, int)]))
    pids = np.unique(np.concatenate([p[0] for p in pr_frames] + [np.zeros(0, int)]))
    gmap = {g: i for i, g in enumerate(gids)}
    pmap = {p: i for i, p in enumerate(pids)}
    G, Pn = len(gids), len(pids)
    pot = np.zeros((G, Pn))
    gcount = np.zeros((G, 1))
    pcount = np.zeros((1, Pn))
    sims = []
    for (gi, gd), (pi, pd) in zip(gt_frames, pr_frames):
        gi = np.array([gmap[x] for x in gi], int)
        pi = np.array([pmap[x] for x in pi], int)
        S = sim_fn(gd, pd) if len(gi) and len(pi) else np.zeros((len(gi), len(pi)))
        sims.append((gi, pi, S))
        if len(gi) and len(pi):
            den = S.sum(0)[None, :] + S.sum(1)[:, None] - S
            sim_iou = np.zeros_like(S)
            ok = den > np.finfo(float).eps
            sim_iou[ok] = S[ok] / den[ok]
            pot[np.ix_(gi, pi)] += sim_iou
        gcount[gi] += 1
        pcount[0, pi] += 1
    gas = pot / np.maximum(gcount + pcount - pot, 1e-9)
    TP = np.zeros(len(alphas))
    FN = np.zeros(len(alphas))
    FP = np.zeros(len(alphas))
    LocS = np.zeros(len(alphas))
    mc = [np.zeros((G, Pn)) for _ in alphas]
    for gi, pi, S in sims:
        if len(gi) == 0:
            FP += len(pi)
            continue
        if len(pi) == 0:
            FN += len(gi)
            continue
        score = gas[np.ix_(gi, pi)] * S
        r, c = linear_sum_assignment(-score)
        for a, al in enumerate(alphas):
            ok = S[r, c] >= al - np.finfo(float).eps
            rr, cc = r[ok], c[ok]
            n = len(rr)
            TP[a] += n
            FN[a] += len(gi) - n
            FP[a] += len(pi) - n
            LocS[a] += S[rr, cc].sum()
            mc[a][gi[rr], pi[cc]] += 1
    AssA = np.zeros(len(alphas))
    for a in range(len(alphas)):
        m = mc[a]
        ass = m / np.maximum(gcount + pcount - m, 1)
        AssA[a] = (m * ass).sum() / max(TP[a], 1)
    DetA = TP / np.maximum(TP + FN + FP, 1)
    HOTA = np.sqrt(DetA * AssA)
    return dict(HOTA=float(HOTA.mean()), DetA=float(DetA.mean()), AssA=float(AssA.mean()),
                LocA=float((LocS / np.maximum(TP, 1)).mean()))


def clear_idf1(gt_frames, pr_frames, sim_fn, thr=0.5):
    gids = np.unique(np.concatenate([g[0] for g in gt_frames]))
    pids = np.unique(np.concatenate([p[0] for p in pr_frames] + [np.zeros(0, int)]))
    gmap = {g: i for i, g in enumerate(gids)}
    pmap = {p: i for i, p in enumerate(pids)}
    idtp = np.zeros((len(gids), len(pids)))
    ngt = npr = 0
    tp = fn = fp = idsw = 0
    last = {}
    for (gi, gd), (pi, pd) in zip(gt_frames, pr_frames):
        ngt += len(gi)
        npr += len(pi)
        if len(gi) == 0 or len(pi) == 0:
            fn += len(gi)
            fp += len(pi)
            continue
        S = sim_fn(gd, pd)
        okm = S >= thr
        g_ix = np.array([gmap[x] for x in gi])
        p_ix = np.array([pmap[x] for x in pi])
        rr, cc = np.nonzero(okm)
        np.add.at(idtp, (g_ix[rr], p_ix[cc]), 1)
        C = np.where(okm, 1 - S, 1e6)
        for a, g in enumerate(gi):
            if g in last:
                for b, p in enumerate(pi):
                    if p == last[g] and okm[a, b]:
                        C[a, b] -= 10
        r, c = linear_sum_assignment(C)
        ok = C[r, c] < 1e5
        r, c = r[ok], c[ok]
        tp += len(r)
        fn += len(gi) - len(r)
        fp += len(pi) - len(r)
        for a, b in zip(r, c):
            g, p = gi[a], pi[b]
            if g in last and last[g] != p:
                idsw += 1
            last[g] = p
    r, c = linear_sum_assignment(-idtp)
    IDTP = idtp[r, c].sum()
    return dict(MOTA=1 - (fn + fp + idsw) / max(ngt, 1), IDF1=2 * IDTP / max(ngt + npr, 1),
                IDP=IDTP / max(npr, 1), IDR=IDTP / max(ngt, 1), IDsw=int(idsw),
                Recall=tp / max(ngt, 1), Precision=tp / max(tp + fp, 1),
                n_gt_ids=len(gids), n_pred_ids=len(pids))
