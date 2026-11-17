import argparse
import copy
import hashlib
import inspect
import json
import os
import sys

import numpy as np

PROFILE = "refD3gap_ext"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
RUNS = os.path.join(_ROOT, "cache", "degprof", "refD3gap_ext_runs.npz")
STATS = os.path.join(_ROOT, "notes", "avo", "refD3gap_ext_stats.json")
REFINE = os.path.join(_ROOT, "cache", "vision", "refine")
STV2 = os.path.join(_ROOT, "cache", "vision", "stv2")
TAGS = ("fA_kin10_argmax", "fB_kin10_argmax")
BALL_FILE = "ball_video_refD3.npz"
PRES_PREFIX = ("b_pres", "t_since_ball", "t_until_ball")


def _q(x, qs=(50, 75, 90, 95)):
    x = np.asarray(x, np.float64)
    x = x[~np.isnan(x)]
    return {"p%d" % q: round(float(np.percentile(x, q)), 3) for q in qs} if len(x) else None


def build(seed=0):
    from mtah import degrade_profiles as dp
    rng = np.random.default_rng(seed)
    runs, st_h = [], {}
    e_gap_a, e_gap_r, e_det_a, e_det_r = [], [], [], []
    for tag in TAGS:
        rep = json.load(open(os.path.join(REFINE, "eval", tag, "report.json")))
        for hn in rep["halves"]:
            r = np.load(os.path.join(REFINE, "eval", tag, "%s.npz" % hn))
            z = np.load(os.path.join(REFINE, "stv2_%s.npz" % hn))
            b = np.load(os.path.join(STV2, hn, "ball.npz"))
            assert np.array_equal(z["npz_index"], b["npz_index"])
            keep = z["npz_index"] >= 0
            V = r["xy"][keep].astype(np.float64)
            mode = np.where(r["is_cand"][keep], 2, 6).astype(np.int8)
            G = b["state"][keep] != 1
            off_r = V - z["ref"][keep].astype(np.float64)
            off_a = V - z["anchor"][keep].astype(np.float64)
            e_gap_a.append(np.linalg.norm(off_a[G], axis=1)), e_gap_r.append(np.linalg.norm(off_r[G], axis=1))
            e_det_a.append(np.linalg.norm(off_a[~G], axis=1)), e_det_r.append(np.linalg.norm(off_r[~G], axis=1))
            og = off_r.copy()
            og[~G] = np.nan
            mg = np.where(G, mode, 0).astype(np.int8)
            meas = int(np.sum(G & ~np.isnan(og[:, 0])))
            filled = dp._lowpass_runs(dp._fill_offsets(og, mg, rng), G)
            s, e = dp._runs(G)
            for a, c in zip(s, e):
                runs.append(filled[a:c].astype(np.float32))
            st_h[hn] = {"tag": tag, "frames": int(keep.sum()), "gap_frac": round(float(G.mean()), 4),
                        "gap_measured_frames": meas, "gap_runs": int(len(s)),
                        "gap_run_s": _q((e - s) / 25.0, (10, 50, 90, 99))}
    lens = np.array([len(x) for x in runs])
    cat = np.concatenate(runs)
    os.makedirs(os.path.dirname(RUNS), exist_ok=True)
    tmp = RUNS + ".tmp.npz"
    np.savez_compressed(tmp, off=cat, lens=lens.astype(np.int64),
                        doc=np.array("degrade_refD3gap_ext.build: refD3 refiner offsets on SoccerTrack v2 frames where "
                                     "the current tracker had no detection (contiguous runs; held-out fold models %s)"
                                     % ",".join(TAGS)))
    os.replace(tmp, RUNS)
    ea, er = np.concatenate(e_gap_a), np.concatenate(e_gap_r)
    da, dr = np.concatenate(e_det_a), np.concatenate(e_det_r)
    st = {"what": "refD3 refiner error on tracker-gap frames of SoccerTrack v2 (external); pool for gap filling",
          "runs_file": os.path.relpath(RUNS, _ROOT), "tags": TAGS, "n_runs": int(len(lens)), "pool_frames": int(lens.sum()),
          "gap_offset_measured_anchor_m": _q(ea), "gap_offset_measured_ref_short_m": _q(er),
          "gap_measured_frames_anchor": int(np.sum(~np.isnan(ea))), "gap_measured_frames_ref_short": int(np.sum(~np.isnan(er))),
          "gap_offset_gt5m_ref_short": round(float(np.mean(er[~np.isnan(er)] > 5)), 4),
          "gap_offset_gt10m_ref_short": round(float(np.nanmean(er[~np.isnan(er)] > 10)), 4),
          "detected_offset_measured_anchor_m": _q(da), "detected_offset_measured_ref_short_m": _q(dr),
          "pool_offset_m (after fill + low-pass)": _q(np.linalg.norm(cat, axis=1)),
          "per_half": st_h}
    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    json.dump(st, open(STATS, "w"), indent=1)
    return st


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        with np.load(RUNS) as z:
            off, lens = z["off"].astype(np.float64), z["lens"]
        ends = np.cumsum(lens)
        _POOL = [off[e - n:e] for n, e in zip(lens, ends)]
    return _POOL


def profile_hash():
    h = hashlib.sha1(inspect.getsource(sys.modules[__name__]).encode())
    h.update(PROFILE.encode())
    with np.load(RUNS) as z:
        for k in ("off", "lens"):
            h.update(np.ascontiguousarray(z[k]).tobytes())
    from mtah.degrade_ext_rand import profile_hash as ph_ext_rand
    h.update(ph_ext_rand().encode())
    return h.hexdigest()[:6]


def fill_gaps(B, Bclean, inper, rng):
    from mtah import degrade_profiles as dp
    out = B.copy()
    gap = np.isnan(B[:, 0]) & inper & ~np.isnan(Bclean[:, 0])
    s, e = dp._runs(gap)
    pool = _pool()
    for a, c in zip(s, e):
        out[a:c, :2] = Bclean[a:c, :2] + dp._splice_copy(pool, c - a, rng)
        out[a:c, 2:] = np.nan
    return out, gap


def degrade(m, rng, player_drop=0.05, noise=0.15):
    from mtah import degrade_profiles as dp
    from mtah.degrade_ext_rand import degrade_ext_rand, view_params
    from mtah.train_ext import unfreeze_ball
    rng0 = copy.deepcopy(rng)
    d = degrade_ext_rand(m, rng, player_drop, noise)
    r2, _ = view_params(rng0, m["name"])
    B0 = unfreeze_ball(m["ball"], m["players"], r2)
    r3 = np.random.default_rng([int(r2.integers(2 ** 32)), 0x5EF3])
    Bc = dp._fill_missing_ball(B0, m["players"], r3)
    inper = m["half"] > 0
    pre = d["ball"]
    filled, gap = fill_gaps(pre, Bc, inper, r3)
    d = dict(d)
    d["ball"], d["ball_prefill"] = filled, pre
    e = np.linalg.norm(filled[gap, :2] - Bc[gap, :2], axis=1)
    degrade.last = {"name": m["name"], "prefill_coverage": float(np.mean(~np.isnan(pre[inper, 0]))) if inper.any() else None,
                    "coverage": float(np.mean(~np.isnan(filled[inper, 0]))) if inper.any() else None,
                    "filled_frac": float(gap[inper].mean()) if inper.any() else None,
                    "fill_err_p50_p90": [round(float(x), 2) for x in np.percentile(e, [50, 90])] if len(e) else None}
    return d


def build_features(orig_build, match):
    X, names = orig_build(match)
    pre = match.get("ball_prefill")
    if pre is None or np.array_equal(pre, match["ball"], equal_nan=True):
        return X, names
    m2 = dict(match)
    m2["ball"] = pre
    X2, n2 = orig_build(m2)
    assert n2 == names
    cols = [i for i, n in enumerate(names) if n.startswith(PRES_PREFIX)]
    X[:, cols] = X2[:, cols]
    return X, names


def register():
    from mtah import degrade_profiles as dp
    from mtah import degrade_refD3_ext
    degrade_refD3_ext.register()
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
                raise ValueError("refD3gap_ext must not receive provided-match statistics")
            return degrade(m, rng, player_drop, noise)
        return orig_degrade(m, rng, gap_lengths, target_missing, player_drop, noise, profile=profile, default=default)

    dp.PROFILES = dp.PROFILES + (PROFILE,)
    dp.COMPLIANT = dp.COMPLIANT + (PROFILE,)
    dp.profile_hash = profile_hash_wrapped
    dp.degrade_profile = degrade_profile_wrapped

    import mtah.featset as fs
    orig_build = fs.build

    def build_wrapped(match):
        return build_features(orig_build, match)
    build_wrapped.__wrapped__ = orig_build
    fs.build = build_wrapped

    from mtah import pipe_ens
    orig_ctx = pipe_ens._ctx

    def ctx_wrapped():
        if not pipe_ens._CTX and dp.active_profile(0) == PROFILE:
            from mtah.eval_ext import canon_index, seed_fns, all_names, EXT
            from mtah.featcache import code_hash
            from mtah.match import BALL
            if dp.active_profile(1000) != PROFILE:
                raise RuntimeError("refD3gap_ext must degrade training and evaluation views alike")
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
    print("hashes: default %s ext_rand %s refD_ext %s refD3_ext %s %s %s" % (
        code_hash("default"), code_hash("ext_rand"), code_hash("refD_ext"), code_hash("refD3_ext"), PROFILE,
        code_hash(PROFILE)))
    for name in a.check or []:
        from mtah.train_ext import load_ext
        from mtah.degrade_ext_rand import degrade_ext_rand
        m = [x for x in load_ext([name.split("_")[0]]) if x["name"] == name][0]
        for seed in (0, 1000):
            d = degrade(m, np.random.default_rng(seed))
            a_ = degrade_ext_rand(m, np.random.default_rng(seed))
            same = np.array_equal(d["ball_prefill"], a_["ball"], equal_nan=True) and \
                np.array_equal(d["players"], a_["players"], equal_nan=True)
            print(name, seed, degrade.last, "pre-fill view == ext_rand view:", same)


if __name__ == "__main__":
    main()
