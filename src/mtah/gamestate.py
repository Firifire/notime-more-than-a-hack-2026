import os

import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d

FPS = 25
R_CTRL = 1.5
K_VEL = 5
SIG_B = 1.6
A_LOOSE = 900.0
EPS_GHOST = 0.03
LR_ABS = 0.3
NOMINAL = 11.0


def _attack_sign(P, team, half):
    sign = np.zeros(len(half), np.int8)
    x = P[:, :, 0]
    for h in np.unique(half):
        if h == 0:
            continue
        m = half == h
        with np.errstate(all="ignore"):
            min0 = np.nanmedian(np.nanmin(np.where(team[None] == 0, x[m], np.nan), 1))
            min1 = np.nanmedian(np.nanmin(np.where(team[None] == 1, x[m], np.nan), 1))
        sign[m] = 1 if min0 < min1 else -1
    return sign


def _centre(m):
    L, W = m["pitch"]
    P = m["players"][:, :, :2].astype(np.float64).copy()
    P[:, :, 0] -= L / 2
    P[:, :, 1] -= W / 2
    B = m["ball"][:, :2].astype(np.float64).copy()
    B[:, 0] -= L / 2
    B[:, 1] -= W / 2
    return P, B


FEATS = ("x_own", "y_abs", "spd", "acc", "spd_max25", "spd_fut_past", "d_opp1", "d_opp2", "d_mate1", "n_opp5",
         "n_opp10", "n_all10", "press_opp", "press_mate", "wclose_opp", "d_cent", "dens", "dx_mate", "dx_opp",
         "is_gk", "rk_press", "rk_dens", "rk_cent", "rk_dopp", "press_opp_m13", "press_opp_m51", "dens_m13",
         "n_opp5_m25", "rk_press_m13", "rk_dens_m13", "n_pres", "d_cent_m13", "press_mate_m13",
         "spd_m13", "acc_m13")


def _rank(v, pres, desc):
    x = np.where(pres, v, np.inf if not desc else -np.inf)
    x = -x if desc else x
    r = np.argsort(np.argsort(x, 1), 1).astype(np.float32)
    r[~pres] = np.nan
    return r


def player_features(m, chunk=6000, gk_mode="half"):
    P, _ = _centre(m)
    team = np.asarray(m["team"]).astype(np.int64)
    half = m["half"]
    N, S, _ = P.shape
    sign = _attack_sign(m["players"][:, :, :2], team, half).astype(np.float64)
    own = np.where(team[None, :] == 0, sign[:, None], -sign[:, None])
    pres = ~np.isnan(P[:, :, 0])
    V = np.full_like(P, np.nan)
    V[K_VEL:-K_VEL] = (P[2 * K_VEL:] - P[:-2 * K_VEL]) / (2 * K_VEL) * FPS
    spd = np.linalg.norm(V, axis=-1)
    bad = spd > 12
    V[bad] = np.nan
    spd[bad] = np.nan
    acc = np.full((N, S), np.nan)
    acc[K_VEL:-K_VEL] = (spd[2 * K_VEL:] - spd[:-2 * K_VEL]) / (2 * K_VEL) * FPS
    out = np.full((N, S, len(FEATS)), np.nan, np.float32)
    ix = {k: i for i, k in enumerate(FEATS)}
    x_own = P[:, :, 0] * own
    out[:, :, ix["x_own"]] = x_own
    out[:, :, ix["y_abs"]] = np.abs(P[:, :, 1])
    out[:, :, ix["spd"]] = spd
    out[:, :, ix["acc"]] = acc
    sp0 = np.nan_to_num(spd)
    out[:, :, ix["spd_max25"]] = maximum_filter1d(sp0, 25, axis=0)
    c = np.cumsum(np.r_[np.zeros((1, S)), sp0], 0)
    i = np.arange(N)
    lo, hi = np.maximum(i - 25, 0), np.minimum(i + 26, N)
    past = (c[i + 1] - c[lo]) / (i + 1 - lo)[:, None]
    fut = (c[hi] - c[i]) / (hi - i)[:, None]
    out[:, :, ix["spd_fut_past"]] = fut - past
    gk = np.zeros((N, S), np.float32)
    if gk_mode == "frame":
        for t in (0, 1):
            sl = np.flatnonzero(team == t)
            xo = np.where(pres[:, sl], x_own[:, sl], np.inf)
            j = np.argmin(xo, 1)
            ok = np.isfinite(xo[np.arange(N), j])
            gk[np.flatnonzero(ok), sl[j[ok]]] = 1
    for h in (np.unique(half[half > 0]) if gk_mode == "half" else []):
        mh = half == h
        for t in (0, 1):
            sl = np.flatnonzero(team == t)
            pr = pres[mh][:, sl].mean(0)
            with np.errstate(all="ignore"):
                med = np.nanmedian(x_own[mh][:, sl], 0)
            med = np.where(pr > 0.3, med, np.inf)
            if np.isfinite(med).any():
                gk[np.ix_(mh, [sl[int(np.nanargmin(med))]])] = 1
    out[:, :, ix["is_gk"]] = gk
    same = team[:, None] == team[None, :]
    eye = np.eye(S, dtype=bool)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        p = P[a:b]
        v = np.nan_to_num(V[a:b])
        pr = pres[a:b]
        diff = p[:, :, None, :] - p[:, None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        ok = pr[:, :, None] & pr[:, None, :] & ~eye[None]
        d = np.where(ok, d, np.inf)
        with np.errstate(all="ignore"):
            clo = np.einsum("cjk,cijk->cij", v, diff) / np.where(np.isfinite(d), d, 1.0)
        clo = np.where(ok, clo, 0.0)
        opp, mate = ~same[None] & ok, same[None] & ok
        dopp = np.where(opp, d, np.inf)
        dmate = np.where(mate, d, np.inf)
        so = np.sort(dopp, 2)
        f = lambda x: np.where(np.isfinite(x), x, np.nan)
        out[a:b, :, ix["d_opp1"]] = f(so[:, :, 0])
        out[a:b, :, ix["d_opp2"]] = f(so[:, :, 1])
        out[a:b, :, ix["d_mate1"]] = f(dmate.min(2))
        out[a:b, :, ix["n_opp5"]] = (dopp < 5).sum(2)
        out[a:b, :, ix["n_opp10"]] = (dopp < 10).sum(2)
        out[a:b, :, ix["n_all10"]] = (d < 10).sum(2)
        w8 = np.exp(-np.where(np.isfinite(d), d, 1e3) / 8)
        out[a:b, :, ix["press_opp"]] = (np.maximum(clo, 0) * w8 * opp).sum(2)
        out[a:b, :, ix["press_mate"]] = (np.maximum(clo, 0) * w8 * mate).sum(2)
        w4 = np.exp(-np.where(np.isfinite(dopp), dopp, 1e3) / 4)
        out[a:b, :, ix["wclose_opp"]] = (clo * w4).sum(2) / np.maximum(w4.sum(2), 1e-9)
        out[a:b, :, ix["dens"]] = (np.exp(-np.where(np.isfinite(d), d, 1e3) ** 2 / 50.0)).sum(2)
        with np.errstate(all="ignore"):
            cen = np.nanmean(np.where(pr[:, :, None], p, np.nan), 1)
            out[a:b, :, ix["d_cent"]] = np.sqrt(((p - cen[:, None]) ** 2).sum(-1))
            xo = x_own[a:b]
            xs = np.where(pr, P[a:b, :, 0], np.nan)
            mh = np.nanmean(np.where(team[None] == 0, xs, np.nan), 1)
            ma = np.nanmean(np.where(team[None] == 1, xs, np.nan), 1)
            mt = np.where(team[None] == 0, mh[:, None], ma[:, None]) * own[a:b]
            mo = np.where(team[None] == 0, ma[:, None], mh[:, None]) * own[a:b]
            out[a:b, :, ix["dx_mate"]] = xo - mt
            out[a:b, :, ix["dx_opp"]] = xo - mo
        out[a:b, :, ix["n_pres"]] = pr.sum(1)[:, None]
    po = out[:, :, ix["press_opp"]]
    out[:, :, ix["rk_press"]] = _rank(po, pres, True)
    out[:, :, ix["rk_dens"]] = _rank(out[:, :, ix["dens"]], pres, True)
    out[:, :, ix["rk_cent"]] = _rank(out[:, :, ix["d_cent"]], pres, False)
    out[:, :, ix["rk_dopp"]] = _rank(out[:, :, ix["d_opp1"]], pres, False)
    for src, dst, w in (("press_opp", "press_opp_m13", 13), ("press_opp", "press_opp_m51", 51),
                        ("dens", "dens_m13", 13), ("n_opp5", "n_opp5_m25", 25), ("rk_press", "rk_press_m13", 13),
                        ("rk_dens", "rk_dens_m13", 13), ("d_cent", "d_cent_m13", 13),
                        ("press_mate", "press_mate_m13", 13), ("spd", "spd_m13", 13), ("acc", "acc_m13", 13)):
        x = out[:, :, ix[src]]
        fill = np.nanmean(x) if np.isfinite(x).any() else 0.0
        sm = uniform_filter1d(np.where(np.isnan(x), fill, x), w, axis=0)
        out[:, :, ix[dst]] = np.where(pres, sm, np.nan)
    out[~pres] = np.nan
    return out, pres


def carrier_target(clean, r=R_CTRL):
    P = clean["players"][:, :, :2].astype(np.float64)
    B = clean["ball"][:, :2].astype(np.float64)
    lab, half = clean["label"], clean["half"]
    d = np.sqrt(((P - B[:, None]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, 1)
    dm = d[np.arange(len(d)), j]
    out = np.where(dm < r, j, -1).astype(np.int16)
    unk = np.isnan(B[:, 0]) | (lab >= 2) | (half <= 0)
    out[unk] = -2
    return out


def team_state(tgt, team):
    st = np.full(len(tgt), -1, np.int8)
    last = -1
    for t in range(len(tgt)):
        c = tgt[t]
        if c == -2:
            last = -1
            continue
        if c >= 0:
            last = int(team[c])
            st[t] = last
        elif last >= 0:
            st[t] = 2 + last
    return st


def fit_dynamics(states):
    C = np.zeros((4, 4))
    for s in states:
        a, b = s[:-1], s[1:]
        ok = (a >= 0) & (b >= 0)
        np.add.at(C, (a[ok], b[ok]), 1)
    perm = [1, 0, 3, 2]
    C = C + C[np.ix_(perm, perm)]
    T = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    return T


def _numba():
    from numba import njit

    @njit(cache=True)
    def fb(E, T, seg_start):
        N, K = E.shape
        alpha = np.empty((N, K))
        beta = np.empty((N, K))
        sc = np.empty(N)
        for t in range(N):
            if seg_start[t]:
                for k in range(K):
                    alpha[t, k] = E[t, k] / K
            else:
                for k in range(K):
                    s = 0.0
                    for j in range(K):
                        s += alpha[t - 1, j] * T[j, k]
                    alpha[t, k] = s * E[t, k]
            z = 0.0
            for k in range(K):
                z += alpha[t, k]
            if z <= 0:
                for k in range(K):
                    alpha[t, k] = 1.0 / K
                z = 1.0
            else:
                for k in range(K):
                    alpha[t, k] /= z
            sc[t] = z
        for t in range(N - 1, -1, -1):
            if t == N - 1 or seg_start[t + 1]:
                for k in range(K):
                    beta[t, k] = 1.0
            else:
                for k in range(K):
                    s = 0.0
                    for j in range(K):
                        s += T[k, j] * E[t + 1, j] * beta[t + 1, j]
                    beta[t, k] = s
                z = 0.0
                for k in range(K):
                    z += beta[t, k]
                if z <= 0:
                    z = 1.0
                for k in range(K):
                    beta[t, k] /= z
        post = alpha * beta
        for t in range(N):
            z = 0.0
            for k in range(K):
                z += post[t, k]
            if z <= 0:
                z = 1.0
            for k in range(K):
                post[t, k] /= z
        return post
    return fb


_FB = None


def forward_backward(E, T, seg_start):
    global _FB
    if _FB is None:
        _FB = _numba()
    return _FB(np.ascontiguousarray(E, np.float64), np.ascontiguousarray(T, np.float64),
               np.ascontiguousarray(seg_start, np.bool_))


def carrier_probs(model, F, pres):
    N, S, K = F.shape
    idx = np.flatnonzero(pres.ravel())
    p = np.full(N * S, np.nan)
    if len(idx):
        p[idx] = model.predict_proba(F.reshape(-1, K)[idx])[:, 1]
    return p.reshape(N, S)


def infer(m, pc, T, prior, sig_b=SIG_B, a_loose=A_LOOSE, eps=EPS_GHOST, lr_abs=LR_ABS, tau=1.0, use_ball=True,
          norm="v1"):
    P, B = _centre(m)
    team = np.asarray(m["team"]).astype(np.int64)
    half = m["half"]
    N, S = pc.shape
    pres = ~np.isnan(pc)
    q = np.clip(np.nan_to_num(pc, nan=prior), 1e-5, 1 - 1e-5)
    lr = (q / (1 - q)) / (prior / (1 - prior))
    lr = lr ** tau
    lr[~pres] = lr_abs
    bpres = ~np.isnan(B[:, 0]) if use_ball else np.zeros(N, bool)
    d2 = ((P - B[:, None]) ** 2).sum(-1)
    g = np.exp(-d2 / (2 * sig_b ** 2)) / (2 * np.pi * sig_b ** 2)
    blr = ((1 - eps) * np.nan_to_num(g) + eps / a_loose) / (1.0 / a_loose)
    blr = np.where(pres & bpres[:, None], blr, np.where(bpres[:, None], eps, 1.0))
    w = lr * blr
    E = np.ones((N, 4))
    for t in (0, 1):
        sl = team == t
        ns = sl.sum()
        if norm == "v2":
            npres = pres[:, sl].sum(1)
            s = np.where(pres[:, sl], w[:, sl], 0.0).sum(1) + \
                np.maximum(NOMINAL - npres, 0) * lr_abs * np.where(bpres, eps, 1.0)
            E[:, t] = s / np.maximum(NOMINAL, npres)
            continue
        s = w[:, sl].sum(1) + max(NOMINAL - ns, 0) * lr_abs * np.where(bpres, eps, 1.0)
        E[:, t] = s / max(NOMINAL, ns)
    seg = np.r_[True, half[1:] != half[:-1]]
    post = forward_backward(E, T, seg)
    wp = np.where(pres, w, 0.0)
    out = {"p_ch": post[:, 0], "p_ca": post[:, 1], "p_lh": post[:, 2], "p_la": post[:, 3]}
    wh = wp * (team[None] == 0) * (post[:, 0] / np.maximum(wp[:, team == 0].sum(1), 1e-12))[:, None]
    wa = wp * (team[None] == 1) * (post[:, 1] / np.maximum(wp[:, team == 1].sum(1), 1e-12))[:, None]
    wc = wh + wa
    tot = wc.sum(1)
    Pn = np.nan_to_num(P)
    with np.errstate(all="ignore"):
        ex = (wc * Pn[:, :, 0]).sum(1) / tot
        ey = (wc * Pn[:, :, 1]).sum(1) / tot
        ev = (wc * ((Pn[:, :, 0] - ex[:, None]) ** 2 + (Pn[:, :, 1] - ey[:, None]) ** 2)).sum(1) / tot
    out.update(bx=ex, by=ey, bsd=np.sqrt(ev), c_mass=tot, E=E)
    out["pmax_h"] = np.nanmax(np.where(team[None] == 0, np.nan_to_num(pc), 0), 1)
    out["pmax_a"] = np.nanmax(np.where(team[None] == 1, np.nan_to_num(pc), 0), 1)
    return out
