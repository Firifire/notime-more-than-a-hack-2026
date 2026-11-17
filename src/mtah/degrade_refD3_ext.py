import argparse
import hashlib
import inspect
import os
import sys

import numpy as np

PROFILE = "refD3_ext"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
SEQ = os.path.join(_ROOT, "cache", "degprof", "refD3_ext_sequences.npz")
BALL_FILE = "ball_video_refD3.npz"

_SEQ = None


def _seq():
    global _SEQ
    if _SEQ is None:
        with np.load(SEQ) as z:
            S = {k: z[k] for k in z.files if k not in ("doc", "halves")}
        h = S["half"]
        S["half_end"] = np.zeros(len(h), np.int64)
        for v in np.unique(h):
            ii = np.flatnonzero(h == v)
            S["half_end"][ii] = ii[-1] + 1
        _SEQ = S
    return _SEQ


def profile_hash():
    h = hashlib.sha1(inspect.getsource(sys.modules[__name__]).encode())
    h.update(PROFILE.encode())
    with np.load(SEQ) as z:
        for k in sorted(z.files):
            if k != "doc":
                h.update(k.encode())
                h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()[:6]


def overlay(rank_ext, rng):
    from mtah import degrade_profiles as dp
    saved = dp._SEQ
    dp._SEQ = _seq()
    try:
        return dp.overlay(rank_ext, "video", rng)
    finally:
        dp._SEQ = saved


def degrade(m, rng, player_drop=0.05, noise=0.15):
    from mtah import degrade_profiles as dp
    from mtah.degrade_ext_rand import name_key
    from mtah.train_ext import unfreeze_ball
    r = np.random.default_rng([int(rng.integers(2 ** 32)), name_key(m["name"])])
    m = dict(m)
    B = unfreeze_ball(m["ball"], m["players"], r)
    B = dp._fill_missing_ball(B, m["players"], r)
    pres, off, mode = overlay(dp.speed_rank(m["players"]), r)
    B[:, :2] = B[:, :2] + off
    B[~pres] = np.nan
    P = m["players"].copy()
    N, S = len(B), P.shape[1]
    for s in range(S):
        n_gaps = r.poisson(player_drop * N / 75)
        for _ in range(n_gaps):
            L = int(r.exponential(75)) + 1
            a = int(r.integers(0, max(1, N - L)))
            P[a:a + L, s] = np.nan
    if noise > 0:
        from scipy.ndimage import gaussian_filter1d
        n = gaussian_filter1d(r.normal(0, noise * 3, P[:, :, :2].shape), 5, axis=0)
        P[:, :, :2] += n
    m["ball"], m["players"] = B, P
    inper = m["half"] > 0
    degrade.last = {"name": m["name"], "coverage": float(np.mean(~np.isnan(B[inper, 0]))) if inper.any() else None,
                    "detected_share": float(np.mean(mode[inper] == 2)) if inper.any() else None}
    return m


def register():
    from mtah import degrade_profiles as dp
    from mtah import degrade_refD_ext as R
    R.register()
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
                raise ValueError("refD3_ext must not receive provided-match statistics (gap_lengths / target_missing)")
            return degrade(m, rng, player_drop, noise)
        return orig_degrade(m, rng, gap_lengths, target_missing, player_drop, noise, profile=profile, default=default)

    dp.PROFILES = dp.PROFILES + (PROFILE,)
    dp.COMPLIANT = dp.COMPLIANT + (PROFILE,)
    dp.profile_hash = profile_hash_wrapped
    dp.degrade_profile = degrade_profile_wrapped

    from mtah import pipe_ens
    orig_ctx = pipe_ens._ctx

    def ctx_wrapped():
        if not pipe_ens._CTX and dp.active_profile(0) == PROFILE:
            from mtah.eval_ext import canon_index, seed_fns, all_names, EXT
            from mtah.featcache import code_hash
            from mtah.match import BALL
            if dp.active_profile(1000) != PROFILE:
                raise RuntimeError("refD3_ext must degrade training and evaluation views alike")
            ci, ti = canon_index()
            tr, ev = seed_fns(0)
            pipe_ens._CTX.update(h=code_hash(), ball=BALL, ci=ci, ti=ti, ext=EXT, lengths={}, train_seed=tr,
                                 eval_seed=ev, names=all_names(), prof=PROFILE)
        return orig_ctx()
    pipe_ens._ctx = ctx_wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="*", default=None)
    a = ap.parse_args()
    register()
    from mtah.featcache import code_hash
    print("hashes: default %s ext_rand %s refD_ext %s refD3_ext %s (profile %s)" % (
        code_hash("default"), code_hash("ext_rand"), code_hash("refD_ext"), code_hash(PROFILE), profile_hash()))
    for name in a.check or []:
        from mtah.train_ext import load_ext
        m = [x for x in load_ext([name.split("_")[0]]) if x["name"] == name][0]
        pres, off, _ = overlay(__import__("mtah.degrade_profiles", fromlist=["x"]).speed_rank(m["players"]),
                               np.random.default_rng(0))
        e = np.linalg.norm(off, axis=1)[m["half"] > 0]
        d = degrade(m, np.random.default_rng(1000))
        print(name, degrade.last, "overlay |off| p50/p90 %.2f/%.2f m" % tuple(np.percentile(e, [50, 90])))


if __name__ == "__main__":
    main()
