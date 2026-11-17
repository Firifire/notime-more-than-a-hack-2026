import numpy as np
from scipy.optimize import linear_sum_assignment

from vision.players_common import hist_dist  # noqa: E402
from vision import track_players as TP  # noqa: E402

DT = 0.04


class P:
    split_gap = 5
    amb_m = 0.0
    jump_m = 0.0
    amb_min = 2
    kit_win = 15
    split_mode = "notsame"
    kit_split = 0.35
    rounds = (12, 50, 125, 250)
    v_max = 9.0
    base_m = 1.0
    sig0 = 0.6
    sig_v = 1.5
    vel_horizon = 1.0
    w_kit = 4.0
    kit_veto = 0.45
    w_gap = 0.5
    c_nolink = 5.0
    min_clean = 3
    kit_team = True
    kit_other = True
    other_fac = 1.5
    kit_ratio = 1.25
    kit_call_min = 5
    kit_overrides = True
    kit_alone = True
    w_emb = 12.0
    emb_mid = 0.3
    emb_split = 0.0
    emb_win = 8
    emb_min = 2
    w_role = 0.0
    sig_role = 1.0
    rejoin = True
    rejoin_kit = False
    gk_rule = True
    gk_min_seq = 7500
    gk_zone_m = 25.0
    gk_side_min = 2.0
    pitch_len = 105.0
    chain_team = True
    kit_final = True
    team_clean = False
    team_min = 5
    team_frac = 0.85
    conf_kit = 0.45
    clean_m = 1.5
    fill_max = 50


def _segments(obs, frames_xy, fpos, amb):
    if len(obs) < 2:
        return [obs]
    ts = np.array([t for t, _ in obs])
    cut = np.zeros(len(obs), bool)
    cut[1:] |= np.diff(ts) > P.split_gap
    if P.jump_m > 0 and len(obs) > 4:
        ps = np.array([frames_xy[fpos[t]][d] for t, d in obs])
        for i in range(4, len(obs)):
            v = (ps[i - 1] - ps[i - 4]) / max(ts[i - 1] - ts[i - 4], 1)
            pred = ps[i - 1] + v * (ts[i] - ts[i - 1])
            if np.hypot(*(ps[i] - pred)) > P.jump_m + 0.4 * (ts[i] - ts[i - 1] - 1) * DT * 10:
                cut[i] = True
    a = np.array([amb[fpos[t]][d] for t, d in obs])
    i = 0
    while i < len(a):
        if a[i]:
            j = i
            while j + 1 < len(a) and a[j + 1]:
                j += 1
            if j - i + 1 >= P.amb_min:
                if i > 0:
                    cut[i] = True
                if j + 1 < len(a):
                    cut[j + 1] = True
            i = j + 1
        else:
            i += 1
    pieces, cur = [], [obs[0]]
    for k in range(1, len(obs)):
        if cut[k]:
            pieces.append(cur)
            cur = []
        cur.append(obs[k])
    pieces.append(cur)
    return pieces


def _kit_split(obs, feat, conf, fpos, clean):
    fs, idx = [], []
    for i, (t, d) in enumerate(obs):
        if clean[fpos[t]][d] and conf[fpos[t]][d] >= P.conf_kit and feat[fpos[t]][d].sum() > 0:
            fs.append(feat[fpos[t]][d])
            idx.append(i)
    w = P.kit_win
    if len(fs) < 2 * w:
        return [obs]
    fs = np.array(fs)
    cs = np.cumsum(np.r_[np.zeros((1, fs.shape[1])), fs], 0)
    best, bi = 0.0, -1
    for k in range(w, len(fs) - w + 1):
        left = (cs[k] - cs[k - w]) / w
        right = (cs[k + w] - cs[k]) / w
        dd = float(hist_dist(left, right))
        if dd > best:
            best, bi = dd, k
    if best < P.kit_split:
        return [obs]
    if P.split_mode == "notsame":
        call = _CALLER[0]
        if call is not None:
            L, R = fs[:bi], fs[bi:]
            tl = call(L.mean(0), obs[idx[bi - 1]][0], len(L))
            tr = call(R.mean(0), obs[idx[bi]][0], len(R))
            if tl >= 0 and tl == tr:
                return [obs]
    if P.split_mode == "team":
        call = _CALLER[0]
        if call is None:
            return [obs]
        L, R = fs[:bi], fs[bi:]
        tl = call(L.mean(0), obs[idx[bi - 1]][0], len(L))
        tr = call(R.mean(0), obs[idx[bi]][0], len(R))
        if tl < 0 or tr < 0 or tl == tr:
            return [obs]
    s = idx[bi]
    return _kit_split(obs[:s], feat, conf, fpos, clean) + _kit_split(obs[s:], feat, conf, fpos, clean)


_CALLER = [None]


def _emb_split(obs, fpos):
    E = _CTX["emb"]
    es, idx = [], []
    for i, (t, d) in enumerate(obs):
        e = E[fpos[t]][d]
        if np.isfinite(e[0]):
            es.append(e)
            idx.append(i)
    w = P.emb_win
    if len(es) < 2 * w:
        return [obs]
    es = np.array(es)
    cs = np.cumsum(np.r_[np.zeros((1, es.shape[1])), es], 0)
    best, bi = 0.0, -1
    for k in range(w, len(es) - w + 1):
        l_ = cs[k] - cs[k - w]
        r_ = cs[k + w] - cs[k]
        dd = 1 - float(l_ @ r_) / max(np.linalg.norm(l_) * np.linalg.norm(r_), 1e-9)
        if dd > best:
            best, bi = dd, k
    if best < P.emb_split:
        return [obs]
    s_ = idx[bi]
    return _emb_split(obs[:s_], fpos) + _emb_split(obs[s_:], fpos)


def _summ(obs, xy, feat, conf, fpos, clean, det_team=None):
    obs = sorted(obs)
    ts = np.array([t for t, _ in obs], float)
    ps = np.array([xy[fpos[t]][d] for t, d in obs])
    n = len(ts)

    def vel(sel_t, sel_p):
        if len(sel_t) < 3 or sel_t[-1] - sel_t[0] < 3:
            return np.zeros(2)
        A = np.c_[sel_t - sel_t.mean(), np.ones(len(sel_t))]
        c = np.linalg.lstsq(A, sel_p, rcond=None)[0]
        v = c[0] / DT
        sp = np.hypot(*v)
        return v * min(1.0, 10.0 / max(sp, 1e-9))
    k = min(n, 10)
    v1 = vel(ts[-k:], ps[-k:])
    v0 = vel(ts[:k], ps[:k])
    fs = [feat[fpos[t]][d] for t, d in obs if clean[fpos[t]][d] and conf[fpos[t]][d] >= P.conf_kit
          and feat[fpos[t]][d].sum() > 0]
    kit = np.mean(fs, 0) if len(fs) >= P.min_clean else None
    votes = np.zeros(2)
    if det_team is not None:
        for t, d in obs:
            if clean[fpos[t]][d]:
                tm = det_team[fpos[t]][d]
                if tm >= 0:
                    votes[tm] += 1
    emb = r0 = r1 = None
    if _CTX["emb"] is not None:
        es = [_CTX["emb"][fpos[t]][d] for t, d in obs]
        es = [x for x in es if np.isfinite(x[0])]
        if len(es) >= P.emb_min:
            emb = np.mean(es, 0)
            emb = emb / max(np.linalg.norm(emb), 1e-9)
    if _CTX["cent"] is not None:
        C = _CTX["cent"]
        r0 = ps[:3].mean(0)[None] - C[fpos[int(ts[0])]]
        r1 = ps[-3:].mean(0)[None] - C[fpos[int(ts[-1])]]
    return dict(emb=emb, r0=r0, r1=r1,
                t0=int(ts[0]), t1=int(ts[-1]), p0=ps[:3].mean(0), p1=ps[-3:].mean(0), v0=v0, v1=v1,
                kit=kit, nkit=len(fs), n=n, votes=votes)


def _team_conf(v):
    s = v.sum()
    if s < P.team_min:
        return -1
    c = int(np.argmax(v))
    return c if v[c] >= P.team_frac * s else -1


_CTX = {"emb": None, "cent": None, "fpos": None}


def team_centroids(frames, xy, det_team, half_win=12, min_n=4):
    n = len(frames)
    S = np.zeros((n, 2, 2))
    N = np.zeros((n, 2))
    for i, (p, tm) in enumerate(zip(xy, det_team)):
        for c in (0, 1):
            m = tm == c
            if m.any():
                S[i, c] = p[m].sum(0)
                N[i, c] = m.sum()
    k = np.ones(2 * half_win + 1)
    out = np.full((n, 2, 2), np.nan)
    for c in (0, 1):
        nn_ = np.convolve(N[:, c], k, "same")
        for j in (0, 1):
            ss = np.convolve(S[:, c, j], k, "same")
            out[:, c, j] = np.where(nn_ >= min_n * (2 * half_win + 1) / 2, ss / np.maximum(nn_, 1e-9), np.nan)
    return out


def link_cost(a, b):
    gap = b["t0"] - a["t1"]
    if gap < 1:
        return None
    g = gap * DT
    d = np.hypot(*(b["p0"] - a["p1"]))
    if d > P.base_m + P.v_max * g:
        return None
    h = min(g, P.vel_horizon)
    ea = np.hypot(*(a["p1"] + a["v1"] * h - b["p0"]))
    eb = np.hypot(*(b["p0"] - b["v0"] * h - a["p1"]))
    e = min(0.5 * (ea + eb), d)
    cost = e / (P.sig0 + P.sig_v * g) + P.w_gap * g
    if a["kit"] is not None and b["kit"] is not None:
        kd = float(hist_dist(a["kit"], b["kit"]))
        if kd > P.kit_veto:
            return None
        cost += P.w_kit * kd
    else:
        cost += P.w_kit * 0.15
    ta, tb = piece_team(a), piece_team(b)
    if ta >= 0 and tb >= 0 and ta != tb:
        return None
    if P.w_emb > 0:
        if a.get("emb") is not None and b.get("emb") is not None:
            cost += P.w_emb * (float(1 - a["emb"] @ b["emb"]) - P.emb_mid)
    if P.w_role > 0:
        tt = ta if ta in (0, 1) else (tb if tb in (0, 1) else -1)
        if tt >= 0 and a.get("r1") is not None and b.get("r0") is not None:
            ra, rb = a["r1"][tt], b["r0"][tt]
            if np.isfinite(ra).all() and np.isfinite(rb).all():
                er = np.hypot(*(ra - rb))
                cost += P.w_role * (er / (P.sig0 + P.sig_role * g) - e / (P.sig0 + P.sig_v * g))
    return cost


def make_kit_caller(S, win=1500):
    anc = [(0.5 * (s["t0"] + s["t1"]), _team_conf(s["votes"]), s["kit"]) for s in S.values()
           if s["kit"] is not None and s["votes"].sum() >= 20]
    anc = [a for a in anc if a[1] >= 0]
    if len(anc) < 6:
        return lambda kit, mid, nkit: -1
    am = np.array([a[0] for a in anc])
    at = np.array([a[1] for a in anc])
    af = np.array([a[2] for a in anc])
    cache = {}

    def call(kit, mid, nkit):
        if kit is None or nkit < P.kit_call_min:
            return -1
        key = int(mid // 250)
        if key not in cache:
            sel = np.abs(am - mid) <= win
            if (at[sel] == 0).sum() < 3 or (at[sel] == 1).sum() < 3:
                sel = np.ones(len(am), bool)
            pr, th = [], []
            for c in (0, 1):
                fc = af[sel & (at == c)]
                if len(fc) == 0:
                    pr = None
                    break
                pp = np.median(fc, 0)
                pp = pp / max(pp.sum(), 1e-9)
                pr.append(pp)
                th.append(float(np.percentile(hist_dist(fc, pp[None]), 95)))
            cache[key] = (pr, th)
        pr, th = cache[key]
        if pr is None:
            return -1
        d = [float(hist_dist(kit, pr[c])) for c in (0, 1)]
        c = int(np.argmin(d))
        if d[c] <= th[c] and d[c] * P.kit_ratio < d[1 - c]:
            return c
        if P.kit_other and d[0] > P.other_fac * th[0] and d[1] > P.other_fac * th[1]:
            return 2
        return -1
    return call


def kit_team_calls(S, win=1500):
    call = make_kit_caller(S, win)
    return {k: call(s["kit"], 0.5 * (s["t0"] + s["t1"]), s["nkit"]) for k, s in S.items()}


def _kt(s):
    return s.get("kit_team", -1)


def piece_team(s):
    tv, tk = _team_conf(s["votes"]), s.get("kit_team", -1)
    if tk == 2:
        return 2 if tv < 0 else tv
    if tk >= 0 and tv >= 0 and tk != tv:
        return tk if P.kit_overrides else -1
    if tv >= 0:
        return tv
    return tk if P.kit_alone else -1


def global_link(pieces, xy, feat, conf, fpos, clean, det_team=None):
    ids = dict(pieces)
    for maxgap in P.rounds:
        keys = list(ids)
        S = {k: _summ(ids[k], xy, feat, conf, fpos, clean, det_team) for k in keys}
        if P.kit_team:
            kt = kit_team_calls(S)
            for k in S:
                S[k]["kit_team"] = kt[k]
        order = sorted(keys, key=lambda k: S[k]["t0"])
        t0s = np.array([S[k]["t0"] for k in order])
        ends = keys
        eidx = {k: i for i, k in enumerate(ends)}
        sidx = {k: i for i, k in enumerate(order)}
        rows, cols, vals = [], [], []
        for a in ends:
            lo = np.searchsorted(t0s, S[a]["t1"], side="right")
            hi = np.searchsorted(t0s, S[a]["t1"] + maxgap, side="right")
            for b in order[lo:hi]:
                c = link_cost(S[a], S[b])
                if c is not None and c < P.c_nolink:
                    rows.append(eidx[a])
                    cols.append(sidx[b])
                    vals.append(c - P.c_nolink)
        if not rows:
            continue
        ur, ri = np.unique(rows, return_inverse=True)
        uc, ci = np.unique(cols, return_inverse=True)
        M = np.zeros((len(ur), len(uc)))
        M[ri, ci] = vals
        r, c = linear_sum_assignment(M)
        nxt, prv = {}, {}
        for i, j in zip(r, c):
            if M[i, j] < 0:
                a, b = ends[ur[i]], order[uc[j]]
                nxt[a], prv[b] = b, a
        new = {}
        for k in keys:
            if k in prv:
                continue
            chain, cur = [k], k
            while cur in nxt:
                cur = nxt[cur]
                chain.append(cur)
            groups, cur_t = [[chain[0]]], (piece_team(S[chain[0]]) if P.chain_team else -1)
            for u in chain[1:]:
                tu = piece_team(S[u]) if P.chain_team else -1
                if tu >= 0 and cur_t >= 0 and tu != cur_t:
                    groups.append([u])
                    cur_t = tu
                else:
                    groups[-1].append(u)
                    if cur_t < 0:
                        cur_t = tu
            for gr in groups:
                obs = []
                for u in gr:
                    obs += ids[u]
                new[gr[0]] = sorted(obs)
        ids = new
    return ids


def frame_flags(frames, xy):
    amb, clean = [], []
    for p in xy:
        if len(p) < 2:
            amb.append(np.zeros(len(p), bool))
            clean.append(np.ones(len(p), bool))
            continue
        D = np.hypot(*(p[:, None] - p[None]).transpose(2, 0, 1))
        np.fill_diagonal(D, 1e9)
        m = D.min(1)
        amb.append(m < P.amb_m)
        clean.append(m >= P.clean_m)
    return amb, clean


def rejoin(ids, pieces, adj):
    first = {}
    last = {}
    for k, obs in ids.items():
        first[obs[0]] = k
        last[obs[-1]] = k
    for a, b, by_kit in adj:
        if by_kit and not P.rejoin_kit:
            continue
        ka = last.get(pieces[a][-1])
        kb = first.get(pieces[b][0])
        if ka is None or kb is None or ka == kb:
            continue
        obs = sorted(ids[ka] + ids[kb])
        del last[ids[ka][-1]], first[ids[kb][0]]
        del ids[kb]
        ids[ka] = obs
        last[obs[-1]] = ka
        first[obs[0]] = ka
    return ids


def track_v2(frames, xy, conf, feat, in_view=None, det_team=None, tracklets=None, return_pieces=False, emb=None):
    fpos = {t: i for i, t in enumerate(frames)}
    _CTX["emb"] = emb if (P.w_emb > 0 or PX.w_emb > 0) else None
    _CTX["cent"] = team_centroids(frames, xy, det_team) if (P.w_role > 0 and det_team is not None) else None
    if tracklets is None:
        tracklets = TP.run_tracker(frames, xy, conf, feat, in_view=in_view)
    amb, clean = frame_flags(frames, xy)
    if P.split_mode in ("team", "notsame"):
        S0 = {k: _summ(o, xy, feat, conf, fpos, clean, det_team) for k, o in tracklets.items()}
        _CALLER[0] = make_kit_caller(S0)
    pieces = {}
    nid = 0
    adj = []
    for k, obs in tracklets.items():
        prev = None
        for seg in _segments(sorted(obs), xy, fpos, amb):
            ks = _kit_split(seg, feat, conf, fpos, clean) if P.split_mode != "off" else [seg]
            if P.emb_split > 0 and _CTX["emb"] is not None:
                ks = [u for k_ in ks for u in _emb_split(k_, fpos)]
            for s2 in ks:
                pieces[nid] = s2
                if prev is not None:
                    adj.append((prev, nid, s2 is not seg))
                prev = nid
                nid += 1
    ids = global_link(pieces, xy, feat, conf, fpos, clean, det_team)
    if P.rejoin:
        ids = rejoin(ids, pieces, adj)
    if PX.on and _CTX["emb"] is not None:
        ids, _ = resolve_crossings(ids, frames, xy, feat, conf, det_team)
    if return_pieces:
        return ids, pieces
    return ids


def densify_v2(tracks, frames, xy, boxes=None, fill_max=None):
    fill_max = P.fill_max if fill_max is None else fill_max
    fpos = {t: i for i, t in enumerate(frames)}
    res = {}
    for nid, (tid, obs) in enumerate(tracks.items()):
        obs = sorted(obs)
        ts = np.array([t for t, _ in obs])
        ds = np.array([d for _, d in obs])
        ps = np.array([xy[fpos[t]][d] for t, d in obs])
        T, X, S, Dd, B = [], [], [], [], []
        bs = np.array([boxes[fpos[t]][d] for t, d in obs]) if boxes is not None else None
        for i in range(len(ts)):
            T.append([ts[i]])
            X.append(ps[i][None])
            S.append([1])
            Dd.append([ds[i]])
            if bs is not None:
                B.append(bs[i][None])
            if i + 1 < len(ts):
                g = ts[i + 1] - ts[i]
                if 1 < g <= fill_max:
                    tt = np.arange(ts[i] + 1, ts[i + 1])
                    w = ((tt - ts[i]) / g)[:, None]
                    T.append(tt)
                    X.append(ps[i] * (1 - w) + ps[i + 1] * w)
                    S.append(np.full(len(tt), 2))
                    Dd.append(np.full(len(tt), -1))
                    if bs is not None:
                        B.append(bs[i] * (1 - w) + bs[i + 1] * w)
        r = dict(t=np.concatenate(T), xy=np.concatenate(X), state=np.concatenate(S).astype(np.int8),
                 det=np.concatenate(Dd).astype(int), src=tid)
        if bs is not None:
            r["box"] = np.concatenate(B)
        res[nid] = r
    return res


class PX:
    on = True
    close_m = 1.0
    max_len = 50
    win = 10
    horizon = 1.0
    w_kit = 0.0
    margin = 0.5
    sig = 0.5
    w_emb = 4.0
    same_team = True


def _fit(ts, ps):
    if len(ts) < 3:
        return ps[-1] if len(ps) else None, np.zeros(2)
    A = np.c_[ts - ts.mean(), np.ones(len(ts))]
    c = np.linalg.lstsq(A, ps, rcond=None)[0]
    return c[1] + c[0] * (ts[-1] - ts.mean()), c[0] / DT


def resolve_crossings(ids, frames, xy, feat, conf, det_team=None):
    import bisect
    fpos = {t: i for i, t in enumerate(frames)}
    ids = {k: sorted(v) for k, v in ids.items()}
    at = {}
    for k, obs in ids.items():
        for t, d in obs:
            at.setdefault(t, []).append((k, d))
    episodes = {}
    for t in sorted(at):
        L = at[t]
        if len(L) < 2:
            continue
        p = np.array([xy[fpos[t]][d] for _, d in L])
        D = np.hypot(*(p[:, None] - p[None]).transpose(2, 0, 1))
        ii, jj = np.nonzero(np.triu(D < PX.close_m, 1))
        for i, j in zip(ii, jj):
            a, b = sorted((L[i][0], L[j][0]))
            ep = episodes.setdefault((a, b), [])
            if ep and t - ep[-1][1] <= 5:
                ep[-1][1] = t
            else:
                ep.append([t, t])
    evs = sorted((s, e, a, b) for (a, b), L in episodes.items() for s, e in L)
    nsw = 0
    owner = {k: k for k in ids}

    def side(obs, t, before):
        ts = [o[0] for o in obs]
        i = bisect.bisect_left(ts, t)
        sel = obs[max(0, i - PX.win):i] if before else obs[i:i + PX.win]
        if len(sel) < 3:
            return None
        T = np.array([o[0] for o in sel], float)
        X = np.array([xy[fpos[o[0]]][o[1]] for o in sel])
        fs = [feat[fpos[o[0]]][o[1]] for o in sel if conf[fpos[o[0]]][o[1]] >= P.conf_kit and feat[fpos[o[0]]][o[1]].sum() > 0]
        kit = np.mean(fs, 0) if fs else None
        em = None
        if _CTX["emb"] is not None:
            es = [_CTX["emb"][fpos[o[0]]][o[1]] for o in (obs[max(0, i - 3 * PX.win):i] if before else obs[i:i + 3 * PX.win])]
            es = [x for x in es if np.isfinite(x[0])]
            if len(es) >= 2:
                em = np.mean(es, 0)
                em = em / max(np.linalg.norm(em), 1e-9)
        v = np.zeros(2)
        if det_team is not None:
            v = np.bincount([det_team[fpos[o[0]]][o[1]] for o in sel if det_team[fpos[o[0]]][o[1]] >= 0], minlength=2)[:2]
        if before:
            p_end, vel = _fit(T, X)
            return dict(t=T[-1], p=p_end, v=vel, kit=kit, votes=v, emb=em)
        Tr, Xr = T[::-1], X[::-1]
        p0, vel = _fit(Tr, Xr)
        return dict(t=T[0], p=p0, v=vel, kit=kit, votes=v, emb=em)

    def pair_cost(pre, post):
        g = (post["t"] - pre["t"]) * DT
        h = min(g, PX.horizon)
        ea = np.hypot(*(pre["p"] + pre["v"] * h - post["p"]))
        eb = np.hypot(*(post["p"] + post["v"] * h - pre["p"]))
        c = 0.5 * (ea + eb) / (PX.sig + 1.0 * g)
        if pre["kit"] is not None and post["kit"] is not None:
            c += PX.w_kit * float(hist_dist(pre["kit"], post["kit"]))
        if PX.w_emb > 0 and pre["emb"] is not None and post["emb"] is not None:
            c += PX.w_emb * float(1 - pre["emb"] @ post["emb"])
        if pre["votes"].sum() >= 3 and post["votes"].sum() >= 3 and \
                np.argmax(pre["votes"]) != np.argmax(post["votes"]):
            c += 2.0
        return c

    for s, e, a, b in evs:
        if e - s + 1 > PX.max_len or a not in ids or b not in ids:
            continue
        A, B = ids[a], ids[b]
        if PX.same_team and det_team is not None:
            va = np.bincount([det_team[fpos[t]][d] for t, d in A if det_team[fpos[t]][d] >= 0], minlength=2)[:2]
            vb = np.bincount([det_team[fpos[t]][d] for t, d in B if det_team[fpos[t]][d] >= 0], minlength=2)[:2]
            ta, tb = _team_conf(va), _team_conf(vb)
            if ta < 0 or tb < 0 or ta != tb:
                continue
        pa, pb = side(A, s, True), side(B, s, True)
        qa, qb = side(A, e + 1, False), side(B, e + 1, False)
        if pa is None or pb is None or qa is None or qb is None:
            continue
        keep = pair_cost(pa, qa) + pair_cost(pb, qb)
        swap = pair_cost(pa, qb) + pair_cost(pb, qa)
        if swap + PX.margin < keep:
            tm = (s + e) // 2 + 1
            ia = bisect.bisect_left([o[0] for o in A], tm)
            ib = bisect.bisect_left([o[0] for o in B], tm)
            ids[a], ids[b] = A[:ia] + B[ib:], B[:ib] + A[ia:]
            nsw += 1
    return ids, nsw


def identity_teams(dense, frames, det_team, feat, det_gk, conf=None):
    team, src = TP.assign_teams(dense, frames, det_team, feat, det_gk)
    if not (P.kit_final or P.gk_rule):
        return team, src
    fpos = {t: i for i, t in enumerate(frames)}
    S = {}
    for k, r in dense.items():
        fs = [feat[fpos[t]][d] for t, d in zip(r["t"], r["det"]) if d >= 0 and feat[fpos[t]][d].sum() > 0
              and (conf is None or conf[fpos[t]][d] >= P.conf_kit)]
        votes = np.zeros(2)
        ngk = 0
        for t, d in zip(r["t"], r["det"]):
            if d >= 0 and det_team[fpos[t]][d] >= 0:
                votes[det_team[fpos[t]][d]] += 1
                ngk += bool(det_gk[fpos[t]][d]) if det_gk is not None else 0
        S[k] = dict(t0=int(r["t"][0]), t1=int(r["t"][-1]), kit=np.mean(fs, 0) if len(fs) >= P.min_clean else None,
                    nkit=len(fs), votes=votes, gk=ngk / max(votes.sum(), 1))
    if P.kit_final:
        kt = kit_team_calls(S)
        for k in dense:
            if src[k] == 1 and kt[k] in (0, 1) and kt[k] != team[k] and S[k]["gk"] < 0.5:
                team[k], src[k] = kt[k], 2
    if P.gk_rule:
        mx = {k: float(np.median(r["xy"][:, 0])) for k, r in dense.items()}
        wts = {k: len(r["t"]) for k, r in dense.items()}
        med = []
        for c in (0, 1):
            ks = [k for k in dense if team[k] == c and S[k]["gk"] < 0.5 and src[k] in (1, 2)]
            if not ks:
                return team, src
            xs = np.repeat([mx[k] for k in ks], [wts[k] for k in ks])
            med.append(np.median(xs))
        left = int(np.argmin(med))
        span = max(r["t"][-1] for r in dense.values()) - min(r["t"][0] for r in dense.values())
        if abs(med[0] - med[1]) >= P.gk_side_min and span >= P.gk_min_seq:
            for k in dense:
                keeper = S[k]["gk"] >= 0.5 and S[k]["votes"].sum() >= 10 and src[k] != 3
                if keeper:
                    x = mx[k]
                    if x < P.gk_zone_m:
                        team[k], src[k] = left, 2
                    elif x > P.pitch_len - P.gk_zone_m:
                        team[k], src[k] = 1 - left, 2
    return team, src
