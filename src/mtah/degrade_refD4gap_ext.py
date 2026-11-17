import argparse
import copy
import hashlib
import inspect
import json
import os
import sys

import numpy as np

PROFILE = "refD4gap_ext"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
RUNS = os.path.join(_ROOT, "cache", "degprof", "refD4gap_ext_runs.npz")
STATS = os.path.join(_ROOT, "notes", "avo", "refD4gap_ext_stats.json")
REFINE = os.path.join(_ROOT, "cache", "vision", "refine")
STV2 = os.path.join(_ROOT, "cache", "vision", "stv2")
TAGS = ("fA_kin10_argmax", "fB_kin10_argmax")
CONF_DIR = os.path.join(REFINE, "eval", "calib_conf")
CONF_THR = 0.4454349481027427
BALL_FILE = "ball_video_refD4.npz"
from mtah.degrade_refD3gap_ext import PRES_PREFIX  # noqa: E402  (flag columns, featset wrapper of refD3gap_ext)


def _q(x, qs=(50, 75, 90, 95)):
    x = np.asarray(x, np.float64)
    x = x[~np.isnan(x)]
    return {"p%d" % q: round(float(np.percentile(x, q)), 3) for q in qs} if len(x) else None


def _rate(e, t):
    e = e[~np.isnan(e)]
    return round(float(np.mean(e > t)), 4) if len(e) else None


def build(seed=0):
    from mtah import degrade_profiles as dp
    rng = np.random.default_rng(seed)
    runs_off, runs_fill, st_h = [], [], {}
    ea, er, eda, conf_gap, fill_g, fill_g12 = [], [], [], [], [], []
    for tag in TAGS:
        rep = json.load(open(os.path.join(REFINE, "eval", tag, "report.json")))
        for hn in rep["halves"]:
            r = np.load(os.path.join(REFINE, "eval", tag, "%s.npz" % hn))
            z = np.load(os.path.join(REFINE, "stv2_%s.npz" % hn))
            b = np.load(os.path.join(STV2, hn, "ball.npz"))
            conf = np.load(os.path.join(CONF_DIR, hn + ".npy"))
            assert np.array_equal(z["npz_index"], b["npz_index"]) and len(conf) == len(b["state"])
            keep = z["npz_index"] >= 0
            V = r["xy"][keep].astype(np.float64)
            mode = np.where(r["is_cand"][keep], 2, 6).astype(np.int8)
            st = b["state"][keep]
            G = st == 0
            fill = G & (conf[keep] >= CONF_THR)
            off_r = V - z["ref"][keep].astype(np.float64)
            off_a = V - z["anchor"][keep].astype(np.float64)
            ea.append(np.linalg.norm(off_a[fill], axis=1)), er.append(np.linalg.norm(off_r[fill], axis=1))
            eda.append(np.linalg.norm(off_a[G & ~fill], axis=1))
            c = conf[keep]
            conf_gap.append(c[G]), fill_g.append(fill[G]), fill_g12.append((c >= CONF_THR)[st != 1])
            og = off_r.copy()
            og[~fill] = np.nan
            mg = np.where(fill, mode, 0).astype(np.int8)
            filled = dp._lowpass_runs(dp._fill_offsets(og, mg, rng), fill)
            s, e = dp._runs(G)
            for a, cc in zip(s, e):
                runs_off.append(np.where(fill[a:cc, None], filled[a:cc], np.nan).astype(np.float32))
                runs_fill.append(fill[a:cc])
            st_h[hn] = {"tag": tag, "frames": int(keep.sum()), "gap_frac": round(float(G.mean()), 4),
                        "fill_rate_in_gap": round(float(fill[G].mean()), 4) if G.any() else None,
                        "measured_filled_frames_ref": int(np.sum(fill & ~np.isnan(off_r[:, 0]))),
                        "gap_run_s": _q((e - s) / 25.0, (10, 50, 90, 99))}
    lens = np.array([len(x) for x in runs_fill])
    os.makedirs(os.path.dirname(RUNS), exist_ok=True)
    tmp = RUNS + ".tmp.npz"
    np.savez_compressed(tmp, off=np.concatenate(runs_off), fill=np.concatenate(runs_fill), lens=lens.astype(np.int64),
                        conf_thr=np.float64(CONF_THR),
                        doc=np.array("degrade_refD4gap_ext.build: refD4 (refD3 decoding + cross-fitted conf >= %.6f) "
                                     "fill pattern and offsets on SoccerTrack v2 tracker-gap frames (state 0); held-out "
                                     "fold models %s" % (CONF_THR, ",".join(TAGS))))
    os.replace(tmp, RUNS)
    ea, er, eda = np.concatenate(ea), np.concatenate(er), np.concatenate(eda)
    fg = np.concatenate(fill_g)
    cg = np.concatenate(conf_gap)
    pool_off = np.concatenate(runs_off)
    out = {"what": "refD4 fill pattern + errors on tracker-gap frames (state 0) of SoccerTrack v2 (external)",
           "runs_file": os.path.relpath(RUNS, _ROOT), "tags": TAGS, "conf_thr": CONF_THR,
           "n_runs": int(len(lens)), "pool_frames": int(lens.sum()),
           "fill_rate_in_gap": round(float(fg.mean()), 4),
           "fill_rate_in_no_detection_frames(state!=1)": round(float(np.concatenate(fill_g12).mean()), 4),
           "filled_offset_anchor_m": _q(ea), "filled_offset_ref_short_m": _q(er),
           "filled_measured_frames_anchor": int(np.sum(~np.isnan(ea))),
           "filled_measured_frames_ref_short": int(np.sum(~np.isnan(er))),
           "filled_wrong_gt3m_anchor": _rate(ea, 3), "filled_wrong_gt5m_anchor": _rate(ea, 5),
           "filled_wrong_gt3m_ref_short": _rate(er, 3), "filled_wrong_gt5m_ref_short": _rate(er, 5),
           "abstained_offset_anchor_m (diagnostic)": _q(eda),
           "pool_offset_on_filled_m (after fill + low-pass)": _q(np.linalg.norm(pool_off, axis=1)),
           "conf_hist_gap": np.histogram(cg, bins=20, range=(0, 1))[0].tolist(),
           "per_half": st_h}
    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    json.dump(out, open(STATS, "w"), indent=1)
    return out


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        with np.load(RUNS) as z:
            off, fill, lens = z["off"].astype(np.float64), z["fill"], z["lens"]
        ends = np.cumsum(lens)
        _POOL = ([off[e - n:e] for n, e in zip(lens, ends)], [fill[e - n:e] for n, e in zip(lens, ends)],
                 lens / lens.sum())
    return _POOL


def profile_hash():
    from mtah import degrade_refD3gap_ext as G3
    from mtah.degrade_ext_rand import profile_hash as ph_ext_rand
    h = hashlib.sha1(inspect.getsource(sys.modules[__name__]).encode())
    h.update(inspect.getsource(G3).encode())
    h.update(PROFILE.encode())
    with np.load(RUNS) as z:
        for k in ("off", "fill", "lens", "conf_thr"):
            h.update(np.ascontiguousarray(z[k]).tobytes())
    h.update(ph_ext_rand().encode())
    return h.hexdigest()[:6]


def draw_pattern(L, rng, tau=12.0):
    offs, fills, w = _pool()
    off = np.full((L, 2), np.nan)
    fill = np.zeros(L, bool)
    k, prev = 0, None
    while k < L:
        i = int(rng.choice(len(fills), p=w))
        st = int(rng.integers(len(fills[i])))
        po, pf = offs[i][st:st + L - k].copy(), fills[i][st:st + L - k]
        if prev is not None and pf[0]:
            t = np.arange(len(po))
            po += (prev - po[0]) * np.exp(-t / tau)[:, None]
        off[k:k + len(po)], fill[k:k + len(po)] = po, pf
        k += len(po)
        prev = off[k - 1] if fill[k - 1] else None
    off[~fill] = np.nan
    return fill, off


def degrade(m, rng, player_drop=0.05, noise=0.15):
    from mtah import degrade_profiles as dp
    from mtah.degrade_ext_rand import degrade_ext_rand, view_params
    from mtah.train_ext import unfreeze_ball
    rng0 = copy.deepcopy(rng)
    d = degrade_ext_rand(m, rng, player_drop, noise)
    r2, _ = view_params(rng0, m["name"])
    B0 = unfreeze_ball(m["ball"], m["players"], r2)
    r3 = np.random.default_rng([int(r2.integers(2 ** 32)), 0x5EF4])
    Bc = dp._fill_missing_ball(B0, m["players"], r3)
    inper = m["half"] > 0
    pre = d["ball"]
    gap = np.isnan(pre[:, 0]) & inper & ~np.isnan(Bc[:, 0])
    out = pre.copy()
    filled = np.zeros(len(pre), bool)
    s, e = dp._runs(gap)
    for a, c in zip(s, e):
        f, off = draw_pattern(c - a, r3)
        ii = np.arange(a, c)[f]
        out[ii, :2] = Bc[ii, :2] + off[f]
        out[ii, 2:] = np.nan
        filled[ii] = True
    d = dict(d)
    d["ball"], d["ball_prefill"] = out, pre
    err = np.linalg.norm(out[filled, :2] - Bc[filled, :2], axis=1)
    degrade.last = {"name": m["name"], "prefill_coverage": float(np.mean(~np.isnan(pre[inper, 0]))) if inper.any() else None,
                    "coverage": float(np.mean(~np.isnan(out[inper, 0]))) if inper.any() else None,
                    "fill_rate_in_gaps": float(filled[gap].mean()) if gap.any() else None,
                    "fill_err_p50_p90": [round(float(x), 2) for x in np.percentile(err, [50, 90])] if len(err) else None}
    return d


def tokens_with_flag(orig_tokens, d):
    segs = orig_tokens(d)
    pre = d.get("ball_prefill")
    if pre is None:
        return segs
    from mtah import raw_seq
    obs = (~np.isnan(pre[:, 0])).astype(np.float64)
    out = []
    for h, a, n, tok, msk in segs:
        nb = len(tok)
        po = raw_seq._blocks(obs[a:a + n], nb).mean(1)
        tok = tok.copy()
        pres_all = tok[:, -1, 5].astype(np.float64)
        tok[:, -1, 5] = po
        tok[:, -1, 6] = np.clip(pres_all - po, 0, 1)
        out.append((h, a, n, tok.astype(np.float16), msk))
    return out


def register():
    from mtah import degrade_profiles as dp
    from mtah import degrade_refD3gap_ext as G3
    G3.register()
    if PROFILE in dp.PROFILES:
        return
    orig_hash, orig_degrade = dp.profile_hash, dp.degrade_profile

    def profile_hash_wrapped(profile):
        return profile_hash() if profile == PROFILE else orig_hash(profile)

    def degrade_profile_wrapped(m, rng, gap_lengths, target_missing, player_drop=0.05, noise=0.15, profile=None,
                                default=None):
        profile = profile or dp.active_profile()
        if profile == PROFILE:
            if gap_lengths is not None or target_missing is not None:
                raise ValueError("refD4gap_ext must not receive provided-match statistics")
            return degrade(m, rng, player_drop, noise)
        return orig_degrade(m, rng, gap_lengths, target_missing, player_drop, noise, profile=profile, default=default)

    dp.PROFILES = dp.PROFILES + (PROFILE,)
    dp.COMPLIANT = dp.COMPLIANT + (PROFILE,)
    dp.profile_hash = profile_hash_wrapped
    dp.degrade_profile = degrade_profile_wrapped

    from mtah import raw_seq
    orig_tok = raw_seq.tokens_from_match

    def tokens_wrapped(d):
        return tokens_with_flag(orig_tok, d)
    tokens_wrapped.__wrapped__ = orig_tok
    raw_seq.tokens_from_match = tokens_wrapped

    from mtah import pipe_ens
    orig_ctx = pipe_ens._ctx

    def ctx_wrapped():
        if not pipe_ens._CTX and dp.active_profile(0) == PROFILE:
            from mtah.eval_ext import canon_index, seed_fns, all_names, EXT
            from mtah.featcache import code_hash
            from mtah.match import BALL
            if dp.active_profile(1000) != PROFILE:
                raise RuntimeError("refD4gap_ext must degrade training and evaluation views alike")
            ci, ti = canon_index()
            tr, ev = seed_fns(0)
            pipe_ens._CTX.update(h=code_hash(), ball=BALL, ci=ci, ti=ti, ext=EXT, lengths={}, train_seed=tr,
                                 eval_seed=ev, names=all_names(), prof=PROFILE)
        return orig_ctx()
    pipe_ens._ctx = ctx_wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", nargs="*", default=None)
    a = ap.parse_args()
    if a.build:
        st = build()
        print(json.dumps({k: v for k, v in st.items() if k != "per_half"}, indent=1))
    register()
    from mtah.featcache import code_hash
    from mtah import raw_seq
    print("hashes: default %s ext_rand %s %s %s | raw view_hash(ext_rand) %s" % (
        code_hash("default"), code_hash("ext_rand"), PROFILE, code_hash(PROFILE), "n/a"))
    for name in a.check or []:
        from mtah.train_ext import load_ext
        from mtah.degrade_ext_rand import degrade_ext_rand
        m = [x for x in load_ext([name.split("_")[0]]) if x["name"] == name][0]
        for seed in (0, 1000):
            d = degrade(m, np.random.default_rng(seed))
            a_ = degrade_ext_rand(m, np.random.default_rng(seed))
            same = np.array_equal(d["ball_prefill"], a_["ball"], equal_nan=True)
            segs = raw_seq.tokens_from_match(d)
            t = np.concatenate([s[3][:, -1].astype(np.float32) for s in segs])
            print(name, seed, degrade.last, "pre-fill == ext_rand:", same,
                  "| raw ball token obs %.3f filled %.3f" % (t[:, 5].mean(), t[:, 6].mean()))


if __name__ == "__main__":
    main()
