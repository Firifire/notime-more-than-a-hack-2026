import argparse
import glob
import hashlib
import inspect
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
COMMON = os.path.join(_ROOT, "data_external", "common")
VIDEO_EXT_SEQ = os.path.join(_ROOT, "cache", "degprof", "video_ext_sequences.npz")
GAPS_PATH = os.path.join(_ROOT, "cache", "degprof", "ext_rand_gaps.npz")
STATS_PATH = os.path.join(_ROOT, "notes", "avo", "ext_rand_stats.json")
FPS = 25
NATURAL_SOURCES = ("pff", "skillcorner", "skillcorner2020", "metrica")
POOL_SOURCES = NATURAL_SOURCES + ("video_ext",)
F_MIN, F_MAX = 0.05, 0.65
W_NAT = 0.5
L_MIN, L_MAX = 5, 3000


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def _q(x, qs=(10, 25, 50, 75, 90, 99)):
    x = np.asarray(x, np.float64)
    return {"p%d" % q: round(float(np.percentile(x, q)), 3) for q in qs} if len(x) else None


def build(path=GAPS_PATH, stats_path=STATS_PATH):
    pool, miss, files = {}, {}, {}
    for src in NATURAL_SOURCES + ("idsse-ctrl",):
        gaps, mf, fl = [], [], []
        for p in sorted(glob.glob(os.path.join(COMMON, src + "_*.npz"))):
            with np.load(p) as z:
                b, h = z["ball"][:, 0], z["half"]
            for hv in np.unique(h[h > 0]):
                s, e = _runs(np.isnan(b[h == hv]))
                gaps += list(e - s)
            mf.append(float(np.isnan(b[h > 0]).mean()))
            fl.append(os.path.basename(p))
        pool[src], miss[src], files[src] = np.asarray(gaps, np.int32), np.asarray(mf), fl
    with np.load(VIDEO_EXT_SEQ) as z:
        pres, half, halves = z["video_present"], z["half"], list(z["halves"])
    gaps, mf = [], []
    for hv in np.unique(half):
        s, e = _runs(~pres[half == hv])
        gaps += list(e - s)
        mf.append(float(1 - pres[half == hv].mean()))
    pool["video_ext"], miss["video_ext"], files["video_ext"] = np.asarray(gaps, np.int32), np.asarray(mf), \
        ["stv2 video halves " + " ".join(str(x) for x in halves)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{"gaps_" + s: pool[s] for s in POOL_SOURCES}, **{"miss_" + s: miss[s] for s in miss},
             doc=np.array("degrade_ext_rand.build: natural ball-gap lengths (frames, inside periods) of EXTERNAL "
                          "tracking sources (data_external/common) and of our video ball on SoccerTrack v2 video"))
    os.replace(tmp, path)
    st = {"what": "natural ball missingness of EXTERNAL data (no provided-match data); per-match missing fraction "
                  "inside periods and gap lengths in seconds", "pool_file": os.path.relpath(path, _ROOT),
          "profile": {"F_MIN": F_MIN, "F_MAX": F_MAX, "W_NAT": W_NAT, "L_MIN_s": L_MIN / FPS, "L_MAX_s": L_MAX / FPS,
                      "pool_sources": list(POOL_SOURCES)},
          "sources": {s: {"n_matches_or_halves": int(len(miss[s])),
                          "missing_frac": _q(miss[s], (0, 25, 50, 75, 100)),
                          "n_gaps": int(len(pool.get(s, []))),
                          "gap_s": _q(pool[s] / FPS) if len(pool.get(s, [])) else None,
                          "files": files[s] if s == "video_ext" else len(files[s])} for s in miss}}
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    json.dump(st, open(stats_path, "w"), indent=1)
    return st


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        with np.load(GAPS_PATH) as z:
            _POOL = [z["gaps_" + s].astype(np.int64) for s in POOL_SOURCES]
    return _POOL


def profile_hash():
    h = hashlib.sha1(inspect.getsource(sys.modules[__name__]).encode())
    h.update(b"ext_rand")
    for s, g in zip(POOL_SOURCES, _pool()):
        h.update(s.encode())
        h.update(np.ascontiguousarray(g, np.int64).tobytes())
    return h.hexdigest()[:6]


def name_key(name):
    return int(hashlib.sha1(str(name).encode()).hexdigest()[:8], 16)


def draw_gap(r, pool):
    if r.random() < W_NAT:
        g = pool[int(r.integers(len(pool)))]
        return int(g[int(r.integers(len(g)))])
    return int(np.exp(r.uniform(np.log(L_MIN), np.log(L_MAX))))


def view_params(rng, name):
    r = np.random.default_rng([int(rng.integers(2 ** 32)), name_key(name)])
    return r, float(r.uniform(F_MIN, F_MAX))


def degrade_ext_rand(m, rng, player_drop=0.05, noise=0.15):
    from mtah.train_ext import unfreeze_ball
    r, f = view_params(rng, m["name"])
    m = dict(m)
    B = unfreeze_ball(m["ball"], m["players"], r)
    P = m["players"].copy()
    N = len(B)
    inper = m["half"] > 0
    miss = np.isnan(B[:, 0])
    budget = int(f * inper.sum()) - int((miss & inper).sum())
    pool = _pool()
    while budget > 0:
        L = min(draw_gap(r, pool), N // 2)
        a = int(r.integers(0, N - L))
        newly = int((~miss[a:a + L] & inper[a:a + L]).sum())
        miss[a:a + L] = True
        budget -= newly
    B[miss] = np.nan
    S = P.shape[1]
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
        nb = gaussian_filter1d(r.normal(0, noise * 3, B[:, :2].shape), 3, axis=0)
        B[:, :2] += nb
    m["ball"], m["players"] = B, P
    degrade_ext_rand.last = {"name": m["name"], "target_missing": f,
                             "missing": float(np.isnan(B[inper, 0]).mean()) if inper.any() else None}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", nargs="*", default=None, help="external match names: print realised missingness")
    a = ap.parse_args()
    if a.build:
        st = build()
        print(json.dumps({s: {k: v for k, v in d.items() if k != "files"} for s, d in st["sources"].items()},
                         indent=1))
        print("wrote", GAPS_PATH, STATS_PATH, "| profile hash", profile_hash())
    if a.check:
        from mtah.train_ext import load_ext
        for name in a.check:
            m = [x for x in load_ext([name.split("_")[0]]) if x["name"] == name][0]
            for seed in (0, 1000, 1001):
                degrade_ext_rand(m, np.random.default_rng(seed))
                print(name, seed, degrade_ext_rand.last)


if __name__ == "__main__":
    main()
