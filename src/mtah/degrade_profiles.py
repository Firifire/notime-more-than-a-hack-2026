import argparse
import hashlib
import json
import os

import numpy as np

ENV, ENV_EVAL = "MTAH_DEG_PROFILE", "MTAH_DEG_PROFILE_EVAL"
PROFILES = ("default", "atd_like", "fused_like")
PROFILES = PROFILES + ("video_ext",)


def _stripped_source():
    import inspect
    import sys
    out, skip = [], False
    tags = ("video_ext", "ext_rand")
    for line in inspect.getsource(sys.modules[__name__]).splitlines(keepends=True):
        t = line.strip()
        if t in ["# >>> " + x for x in tags]:
            skip = True
        elif t in ["# <<< " + x for x in tags]:
            skip = False
        elif not skip:
            out.append(line)
    return "".join(out)


class _StrippedInspect:
    @staticmethod
    def getsource(_module):
        return _stripped_source()
PROFILES = PROFILES + ("ext_rand",)
COMPLIANT = ("ext_rand",)
FUSED_FILE = "ball_video_v3.npz"
EVAL_SEED_MIN = 1000
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SEQ_PATH = os.path.join(_ROOT, "cache", "degprof", "target_sequences.npz")
STATS_PATH = os.path.join(_ROOT, "notes", "avo", "deg_profiles_stats.json")
FPS = 25
MODES = {0: "missing", 1: "atd", 2: "video_detected", 3: "triangulated", 4: "broadcast",
         5: "crop_redetect", 6: "video_interpolated"}


def active_profile(seed=None):
    p = os.environ.get(ENV, "default") or "default"
    if seed is not None and seed >= EVAL_SEED_MIN and os.environ.get(ENV_EVAL):
        p = os.environ[ENV_EVAL]
    if p == "mix" and seed is not None:
        if seed >= EVAL_SEED_MIN:
            raise ValueError("MTAH_DEG_PROFILE=mix needs MTAH_DEG_PROFILE_EVAL for evaluation views")
        p = ("atd_like", "fused_like")[(seed // 7) % 2]
    if p not in PROFILES + ("mix",):
        raise ValueError("unknown degradation profile %r (choose from %s)" % (p, PROFILES + ("mix",)))
    return p


def profile_hash(profile):
    import inspect
    import sys
    if profile == "ext_rand":
        from mtah.degrade_ext_rand import profile_hash as profile_hash_ext_rand
        return profile_hash_ext_rand()
    if profile == "video_ext":
        from mtah.degrade_video_ext import profile_hash_ext
        return profile_hash_ext(_stripped_source())
    inspect = _StrippedInspect()
    h = hashlib.sha1(inspect.getsource(sys.modules[__name__]).encode())
    h.update(profile.encode())
    with open(SEQ_PATH, "rb") as f:
        h.update(hashlib.sha1(f.read()).digest())
    return h.hexdigest()[:6]


def speed_rank(players, smooth_s=2.0):
    from scipy.ndimage import uniform_filter1d
    P = players[:, :, :2].astype(np.float64)
    v = np.linalg.norm(np.diff(P, axis=0), axis=2) * FPS
    v[(v > 12) | np.isnan(v)] = np.nan
    v = np.vstack([v, v[-1:]])
    s = np.nansum(np.nan_to_num(v), axis=1)
    c = np.sum(~np.isnan(v), axis=1).astype(np.float64)
    w = int(smooth_s * FPS)
    s, c = uniform_filter1d(s, w), uniform_filter1d(c, w)
    sp = np.where(c > 0.5, s / np.maximum(c, 1e-9), np.nan)
    ok = ~np.isnan(sp)
    r = np.full(len(sp), 0.5)
    if ok.sum() > 1:
        order = np.argsort(sp[ok], kind="stable")
        rk = np.empty(ok.sum())
        rk[order] = np.arange(ok.sum()) / (ok.sum() - 1)
        r[ok] = rk
    return r.astype(np.float32)


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


POOL = {1: (1,), 2: (2, 3), 3: (2, 3), 4: (2, 3), 5: (5,), 6: (6,)}


def _splice_copy(pool_runs, L, rng, tau=12.0):
    out = np.empty((L, 2))
    k, prev = 0, None
    w = np.array([len(r) for r in pool_runs], np.float64)
    w /= w.sum()
    while k < L:
        r = pool_runs[rng.choice(len(pool_runs), p=w)]
        st = int(rng.integers(len(r)))
        piece = r[st:st + L - k].astype(np.float64)
        if prev is not None:
            piece = piece + (prev - piece[0]) * np.exp(-np.arange(len(piece)) / tau)[:, None]
        out[k:k + len(piece)] = piece
        k += len(piece)
        prev = out[k - 1]
    return out


def _fill_offsets(off, mode, rng):
    off = off.copy()
    present = mode > 0
    for md in np.unique(mode[present]):
        sel = mode == md
        meas = np.isin(mode, POOL[int(md)]) & ~np.isnan(off[:, 0])
        if meas.sum() < 50:
            meas = np.isin(mode, POOL[2]) & ~np.isnan(off[:, 0])
        rs, re_ = _runs(meas)
        pool_runs = [off[x:y].copy() for x, y in zip(rs, re_)]
        s, e = _runs(sel & np.isnan(off[:, 0]))
        for a, b in zip(s, e):
            piece = _splice_copy(pool_runs, b - a, rng)
            if a > 0 and not np.isnan(off[a - 1, 0]):
                piece += (off[a - 1] - piece[0]) * np.exp(-np.arange(b - a) / 12.0)[:, None]
            if b < len(off) and not np.isnan(off[b, 0]):
                piece += (off[b] - piece[-1]) * np.exp(-np.arange(b - a)[::-1] / 12.0)[:, None]
            off[a:b] = piece
    return off


def _lowpass_runs(x, present, sigma=4.0):
    from scipy.ndimage import gaussian_filter1d
    x = x.copy()
    s, e = _runs(present)
    for a, b in zip(s, e):
        if b - a > 3:
            x[a:b] = gaussian_filter1d(x[a:b], sigma, axis=0, mode="nearest")
    return x


def _medfilt_runs(x, k=5):
    from scipy.ndimage import median_filter
    x = x.copy()
    s, e = _runs(~np.isnan(x[:, 0]))
    for a, b in zip(s, e):
        if b - a >= k:
            x[a:b] = median_filter(x[a:b], size=(k, 1), mode="nearest")
    return x


def _q(x, qs=(50, 75, 90, 95, 99)):
    x = np.asarray(x, np.float64)
    x = x[~np.isnan(x)]
    if not len(x):
        return None
    return {"p%d" % q: round(float(np.percentile(x, q)), 3) for q in qs}


def _acf(off, mode_ok, lags=(1, 5, 25, 125)):
    out = {}
    x = off.copy()
    x[~mode_ok] = np.nan
    mu = np.nanmean(x, axis=0)
    xc = x - mu
    var = np.nanmean(xc ** 2, axis=0)
    for L in lags:
        a, b = xc[:-L], xc[L:]
        ok = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
        out["lag%d" % L] = round(float(np.mean(np.sum(a[ok] * b[ok], 0) / var / ok.sum())), 3) if ok.sum() else None
    return out


def _gap_stats(present, halves):
    lens = []
    for a, b in halves:
        s, e = _runs(~present[a:b])
        lens += list(e - s)
    lens = np.array(lens) / FPS
    miss = 1 - present.mean()
    return {"coverage": round(float(present.mean()), 4), "n_gaps": int(len(lens)),
            "gap_s": _q(lens, (10, 25, 50, 75, 90, 99)),
            "gap_s_max": round(float(lens.max()), 1) if len(lens) else 0,
            "missing_in_gaps_gt10s": round(float(lens[lens > 10].sum() * FPS / max(1, (~present).sum())), 3),
            "missing_in_gaps_gt60s": round(float(lens[lens > 60].sum() * FPS / max(1, (~present).sum())), 3),
            "missing_frac": round(float(miss), 4)}


def build(seed=0):
    from mtah.data import load_tracking, HALVES, CACHE
    t = load_tracking()
    atd = t["ball"][:, :2].astype(np.float64)
    z = np.load(os.path.join(CACHE, "vision", FUSED_FILE))
    v1 = np.load(os.path.join(CACHE, "vision", "ball_video.npz"))
    idx = np.concatenate([np.arange(a, b + 1) for _, a, b in HALVES])
    hid = np.concatenate([np.full(b + 1 - a, h, np.int8) for h, a, b in HALVES])
    halves, o = [], 0
    for _, a, b in HALVES:
        halves.append((o, o + b + 1 - a))
        o += b + 1 - a
    A = atd[idx]
    F = z["xy_fused"][idx].astype(np.float64)
    fsrc = z["fused_source"][idx]
    vstate = z["state"][idx]
    tri = z["xyz_tri"][idx].astype(np.float64)
    bc = z["bc_xy"][idx].astype(np.float64)
    bcs = z["bc_state"][idx]
    air = ~np.isnan(tri[:, 2]) & (tri[:, 2] > 1.0)
    ref = np.full_like(A, np.nan)
    use_bc = (bcs == 1) & ~air
    ref[use_bc] = bc[use_bc]
    ref[air] = tri[air, :2]
    rs, re_ = _runs(np.isnan(ref[:, 0]))
    for a0, b0 in zip(rs, re_):
        if a0 > 0 and b0 < len(ref) and b0 - a0 <= 6:
            w = np.linspace(0, 1, b0 - a0 + 2)[1:-1, None]
            ref[a0:b0] = ref[a0 - 1] * (1 - w) + ref[b0] * w
    ref = _medfilt_runs(ref)
    a_pres = ~np.isnan(A[:, 0])
    a_mode = a_pres.astype(np.int8)
    a_off = A - ref
    a_off[~a_pres] = np.nan
    f_pres = ~np.isnan(F[:, 0])
    f_mode = np.zeros(len(F), np.int8)
    f_mode[fsrc == 1] = 1
    f_mode[(fsrc == 2) & (vstate == 1)] = 2
    f_mode[(fsrc == 2) & (vstate != 1)] = 6
    f_mode[fsrc == 3] = 3
    f_mode[fsrc == 4] = 4
    f_mode[fsrc == 5] = 5
    f_mode[~f_pres] = 0
    f_off = F - ref
    f_off[~f_pres] = np.nan
    f_off[(f_mode == 3) | (f_mode == 4)] = np.nan
    V = z["xy"][idx].astype(np.float64)
    m3 = (f_mode == 3) & (vstate == 1) & (bcs >= 1) & ~air
    f_off[m3] = V[m3] - ref[m3]
    a_off, f_off = _medfilt_runs(a_off, 9), _medfilt_runs(f_off, 9)
    rank = speed_rank(t["players"][idx])
    rng = np.random.default_rng(seed)
    a_off_f = _lowpass_runs(_fill_offsets(a_off, a_mode, rng), a_pres)
    f_off_f = _lowpass_runs(_fill_offsets(f_off, f_mode, rng), f_pres)
    os.makedirs(os.path.dirname(SEQ_PATH), exist_ok=True)
    np.savez_compressed(SEQ_PATH, rank=rank, half=hid,
                        atd_present=a_pres, atd_mode=a_mode, atd_off=a_off_f.astype(np.float32),
                        fused_present=f_pres, fused_mode=f_mode, fused_off=f_off_f.astype(np.float32),
                        doc=np.array("degrade_profiles.build: per graded frame of the provided game; "
                                     "label-free corruption sequences of the ATD and fused3 ball inputs"))

    st = {"what": "unlabelled statistics of the two target ball inputs (graded frames, no GT/labels)",
          "reference": "broadcast-camera track (bc_state==1) when not airborne; triangulated ground "
                       "point when tri z > 1 m; measured on %.1f%% of graded frames" % (100 * (~np.isnan(ref[:, 0])).mean()),
          "fused_file": FUSED_FILE}
    for name, pres, off_m, mode in (("atd", a_pres, a_off, a_mode), ("fused", f_pres, f_off, f_mode)):
        d = _gap_stats(pres, halves)
        e = np.linalg.norm(off_m, axis=1)
        d["offset_measured_frac_of_present"] = round(float((~np.isnan(e[pres])).mean()), 3)
        d["offset_m"] = _q(e)
        d["offset_gt5m"] = round(float(np.nanmean(e[pres] > 5)), 4)
        d["offset_gt10m"] = round(float(np.nanmean(e[pres] > 10)), 4)
        d["acf_offset"] = _acf(off_m, pres & ~np.isnan(off_m[:, 0]))
        d["by_mode"] = {}
        for md in np.unique(mode):
            sel = mode == md
            d["by_mode"][MODES[int(md)]] = {"frac": round(float(sel.mean()), 4),
                                            "offset_m": _q(e[sel]) if md > 0 else None}
        d["by_speed_quintile"] = []
        X = A if name == "atd" else F
        step = np.r_[np.nan, np.linalg.norm(np.diff(X, axis=0), axis=1)]
        for qi in range(5):
            sel = (rank >= qi / 5) & (rank < (qi + 1) / 5 + (qi == 4))
            d["by_speed_quintile"].append({
                "quintile": qi, "present": round(float(pres[sel].mean()), 3),
                "offset_p50": round(float(np.nanpercentile(e[sel], 50)), 2) if np.any(~np.isnan(e[sel])) else None,
                "offset_p90": round(float(np.nanpercentile(e[sel], 90)), 2) if np.any(~np.isnan(e[sel])) else None,
                "step_gt1m_per_frame": round(float(np.nanmean(step[sel] > 1.0)), 4),
            })
        st[name] = d
    e_f = np.linalg.norm(f_off, axis=1)
    st["fused_interpolated_frac"] = round(float((f_mode == 6).mean()), 4)
    st["fused_interpolated_offset_m"] = _q(e_f[f_mode == 6])
    st["fused_detected_offset_m"] = _q(e_f[f_mode == 2])
    v1xy = v1["xy"][idx].astype(np.float64)
    airerr = np.linalg.norm(v1xy - tri[:, :2], axis=1)
    st["airborne"] = {"frac_tri_z_gt1m_of_graded": round(float(air.mean()), 4),
                      "tri_height_m": _q(tri[air, 2]),
                      "video_ground_projection_err_m": _q(airerr[air]),
                      "atd_err_vs_tri_m": _q(np.linalg.norm(A[air] - tri[air, :2], axis=1))}
    ghost = z["atd_ghost"][idx]
    st["atd_ghost_frac_of_atd_frames"] = round(float(ghost[a_pres].mean()), 4)
    gs, ge = _runs(ghost)
    st["atd_ghost_run_s"] = _q((ge - gs) / FPS, (50, 90, 99))
    st["atd_offset_on_ghost_m"] = _q(np.linalg.norm(a_off[ghost], axis=1))
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    json.dump(st, open(STATS_PATH, "w"), indent=1)
    return st


_SEQ = None


def _seq():
    global _SEQ
    if _SEQ is None:
        z = np.load(SEQ_PATH)
        _SEQ = {k: z[k] for k in z.files if k != "doc"}
        h = _SEQ["half"]
        _SEQ["half_end"] = np.zeros(len(h), np.int64)
        for v in np.unique(h):
            ii = np.flatnonzero(h == v)
            _SEQ["half_end"][ii] = ii[-1] + 1
    return _SEQ


TRAJ = np.array([-50, -25, 0, 25, 50, 100, 150, 200, 250])


def _index(which):
    S = _seq()
    key = "_idx_" + which
    if key not in S:
        pres = S[which + "_present"]
        n = len(pres)
        start = np.r_[True, (pres[1:] != pres[:-1]) | (S["half"][1:] != S["half"][:-1])]
        pos = np.arange(n)
        elapsed = pos - np.maximum.accumulate(np.where(start, pos, 0))
        hs = pos - np.maximum.accumulate(np.where(np.r_[True, S["half"][1:] != S["half"][:-1]], pos, 0))
        c = np.flatnonzero((S["half_end"] - pos > TRAJ[-1] + 1) & (hs >= -TRAJ[0]))[::5]
        feats = S["rank"][c[:, None] + TRAJ[None, :]].astype(np.float32)
        S[key] = (elapsed, c, feats, pres[c], elapsed[c])
    return S[key]


def _traj(rank, t):
    i = np.clip(t + TRAJ, 0, len(rank) - 1)
    return rank[i]


def overlay(rank_ext, which, rng, step=25, win=50, diverge=0.1, k_nn=20, splice=12.0, keep_state=1.0,
            match_elapsed=False):
    S = _seq()
    pres_t, off_t, mode_t = S[which + "_present"], S[which + "_off"], S[which + "_mode"]
    rank_t, hend = S["rank"], S["half_end"]
    elapsed_t, cand, feats, cstate, celap = _index(which)
    N = len(rank_ext)
    pres = np.zeros(N, bool)
    off = np.zeros((N, 2), np.float32)
    mode = np.zeros(N, np.int8)
    decay = np.exp(-np.arange(4 * int(splice) + 1) / splice)[:, None]

    def jump(t, state, el):
        q = _traj(rank_ext, t)
        d = np.sum((feats - q) ** 2, axis=1)
        nn = np.argpartition(d, k_nn)[:k_nn]
        if state is not None and (keep_state >= 1 or rng.random() < keep_state):
            same = nn[cstate[nn] == state]
            if len(same):
                good = same[(celap[same] >= 0.5 * el) & (celap[same] <= 2 * el + step)] \
                    if match_elapsed else same
                nn = good if len(good) else same
        return int(cand[nn[rng.integers(len(nn))]])

    s = jump(0, None, None)
    t, pending, since = 0, None, 0
    stats = {"jumps": 0, "absdiff": 0.0}
    while t < N:
        L = int(min(step, N - t, hend[s] - s))
        o = off_t[s:s + L].astype(np.float64)
        if pending is not None:
            k = min(L, len(decay) - pending[1])
            o[:k] += pending[0] * decay[pending[1]:pending[1] + k]
            pending = (pending[0], pending[1] + k) if pending[1] + k < len(decay) else None
        pres[t:t + L], off[t:t + L], mode[t:t + L] = pres_t[s:s + L], o, mode_t[s:s + L]
        stats["absdiff"] += float(np.sum(np.abs(rank_ext[t:t + L] - rank_t[s:s + L])))
        t, s, since = t + L, s + L, since + L
        if t >= N:
            break
        w = min(win, since)
        d = np.mean(np.abs(rank_ext[t - w:t] - rank_t[s - w:s]))
        if s >= hend[s - 1] or d > diverge:
            last = off[t - 1].astype(np.float64)
            st = bool(pres_t[s - 1]) if s >= hend[s - 1] else bool(pres_t[s])
            el = int(elapsed_t[s - 1]) + 1 if s >= hend[s - 1] else int(elapsed_t[s])
            s = jump(t, st, el)
            since = 0
            stats["jumps"] += 1
            pending = (last - off_t[s].astype(np.float64), 0) if pres[t - 1] and pres_t[s] else None
    overlay.last_stats = {"jumps": stats["jumps"], "mean_abs_rank_diff": stats["absdiff"] / max(N, 1)}
    return pres, off, mode


def _fill_missing_ball(B, P, rng):
    from mtah.train_ext import unfreeze_ball
    xy = B[:, :2].copy()
    ok = ~np.isnan(xy[:, 0])
    if not ok.any():
        return B
    i = np.where(ok, np.arange(len(xy)), -1)
    i = np.maximum.accumulate(i)
    first = int(np.argmax(ok))
    i[:first] = first
    B = B.copy()
    B[:, :2] = xy[i]
    return unfreeze_ball(B, P, rng)


def degrade_profile(m, rng, gap_lengths, target_missing, player_drop=0.05, noise=0.15, profile=None,
                    default=None):
    profile = profile or active_profile()
    if profile == "default":
        if default is None:
            from mtah.train_ext import _degrade_default as default
        return default(m, rng, gap_lengths, target_missing, player_drop, noise)
    if profile == "ext_rand":
        from mtah.degrade_ext_rand import degrade_ext_rand
        if gap_lengths is not None or target_missing is not None:
            raise ValueError("ext_rand must not receive provided-match statistics (gap_lengths / target_missing)")
        return degrade_ext_rand(m, rng, player_drop, noise)
    if profile == "video_ext":
        from mtah.degrade_video_ext import degrade_video_ext
        return degrade_video_ext(m, rng, player_drop, noise)
    from mtah.train_ext import unfreeze_ball
    which = {"atd_like": "atd", "fused_like": "fused"}[profile]
    m = dict(m)
    P = m["players"].copy()
    B = unfreeze_ball(m["ball"], m["players"], rng)
    B = _fill_missing_ball(B, m["players"], rng)
    pres, off, _ = overlay(speed_rank(m["players"]), which, rng)
    B[:, :2] = B[:, :2] + off
    B[~pres] = np.nan
    N, S = len(B), P.shape[1]
    for s in range(S):
        n_gaps = rng.poisson(player_drop * N / 75)
        for _ in range(n_gaps):
            L = int(rng.exponential(75)) + 1
            a = int(rng.integers(0, max(1, N - L)))
            P[a:a + L, s] = np.nan
    if noise > 0:
        from scipy.ndimage import gaussian_filter1d
        n = gaussian_filter1d(rng.normal(0, noise * 3, P[:, :, :2].shape), 5, axis=0)
        P[:, :, :2] += n
    m["ball"], m["players"] = B, P
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", default=None, help="external match name: print overlay summary")
    args = ap.parse_args()
    if args.build:
        st = build()
        print(json.dumps(st, indent=1)[:6000])
    if args.check:
        from mtah.train_ext import load_ext, our_gap_lengths
        m = [x for x in load_ext([args.check.split("_")[0]]) if x["name"] == args.check][0]
        for prof in ("atd_like", "fused_like"):
            d = degrade_profile(m, np.random.default_rng(0), None, None, profile=prof)
            b = d["ball"][:, 0]
            g = m["half"] > 0
            e = np.linalg.norm(d["ball"][:, :2] - m["ball"][:, :2], axis=1)
            print(prof, "coverage %.3f" % np.mean(~np.isnan(b[g])), "err p50/p90 %.2f/%.2f" % tuple(
                np.nanpercentile(e[g], [50, 90])), "D-frame coverage %.3f / alive %.3f" % (
                np.mean(~np.isnan(b[g & (m["label"] == 2)])), np.mean(~np.isnan(b[g & (m["label"] < 2)]))))


if __name__ == "__main__":
    main()
