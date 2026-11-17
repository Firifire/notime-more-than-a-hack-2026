import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from mtah.data import ROOT, CACHE, write_submission
from mtah.decode import make_trans, viterbi
from mtah.featset import build, swap_teams
from mtah.match import provided_match

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

EXT = os.path.join(ROOT, "data_external", "common")
BALL_MASKED_SOURCES = ("stv2",)


def mask_ball(m):
    if m["name"].split("_")[0] in BALL_MASKED_SOURCES:
        m = dict(m)
        m["ball"] = np.full_like(m["ball"], np.nan)
        if "ball_anchor_only" in m:
            m["ball_anchor_only"] = np.full_like(m["ball_anchor_only"], np.nan)
    return m


def load_ext(prefixes):
    out = []
    for p in sorted(glob.glob(os.path.join(EXT, "*.npz"))):
        name = os.path.basename(p)[:-4]
        if prefixes and name.split("_")[0] not in prefixes:
            continue
        z = np.load(p)
        m = {k: z[k] for k in z.files}
        m["name"] = name
        out.append(mask_ball(m))
    return out


def our_gap_lengths():
    from mtah.data import graded_mask
    from mtah.ball import runs
    from mtah.match import ball_array
    b = ball_array()[:, 0]
    s, e = runs(np.isnan(b) & graded_mask())
    return (e - s), float(np.mean(np.isnan(b[graded_mask()])))


def unfreeze_ball(B, P, rng, min_run=5):
    B = B.copy()
    N = len(B)
    xy = B[:, :2]
    d = np.r_[np.inf, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    still = np.nan_to_num(d, nan=np.inf) < 1e-6
    e = np.diff(np.r_[0, still.astype(np.int8), 0])
    starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    Pxy = P[:, :, :2]
    for a, b in zip(starts, ends):
        a = max(a - 1, 0)
        if b - a < min_run:
            continue
        target_end = xy[b - 1].copy()
        pos = xy[a].copy()
        t = a
        def nearest(t, pos, exclude=None, far=False):
            pp = Pxy[t]
            ok = ~np.isnan(pp[:, 0])
            if exclude is not None:
                ok[exclude] = False
            if not ok.any():
                return None
            dd = np.linalg.norm(pp - pos, axis=1)
            dd[~ok] = np.inf
            if far:
                cand = np.flatnonzero((dd > 5) & (dd < 30))
                return int(rng.choice(cand)) if len(cand) else None
            return int(np.argmin(dd))
        holder = nearest(t, pos)
        end_blend = max(a, b - 25)
        while t < end_blend:
            L = int(rng.uniform(25, 100))
            off = rng.normal(0, 0.5, 2)
            for k in range(t, min(t + L, end_blend)):
                if holder is not None and not np.isnan(Pxy[k, holder, 0]):
                    pos = Pxy[k, holder] + off
                xy[k] = pos + rng.normal(0, 0.15, 2)
            t = min(t + L, end_blend)
            if t >= end_blend:
                break
            tgt = nearest(t, pos, exclude=holder, far=True)
            if tgt is None:
                continue
            dest = Pxy[t, tgt]
            dist = np.linalg.norm(dest - pos)
            T = max(2, int(dist / rng.uniform(5, 14) * 25))
            for i, k in enumerate(range(t, min(t + T, end_blend))):
                xy[k] = pos + (dest - pos) * (i + 1) / T
            pos = xy[min(t + T, end_blend) - 1].copy()
            holder = tgt
            t = min(t + T, end_blend)
        n = b - end_blend
        for i, k in enumerate(range(end_blend, b)):
            xy[k] = pos + (target_end - pos) * (i + 1) / n
    B[:, :2] = xy
    return B


def degrade(m, rng, gap_lengths, target_missing, player_drop=0.05, noise=0.15):
    m = dict(m)
    B = unfreeze_ball(m["ball"], m["players"], rng)
    P = m["players"].copy()
    N = len(B)
    miss = np.isnan(B[:, 0])
    budget = int(target_missing * N) - int(miss.sum())
    while budget > 0:
        L = int(rng.choice(gap_lengths))
        a = int(rng.integers(0, N - L))
        newly = (~miss[a:a + L]).sum()
        miss[a:a + L] = True
        budget -= newly
    B[miss] = np.nan
    S = P.shape[1]
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
        nb = gaussian_filter1d(rng.normal(0, noise * 3, B[:, :2].shape), 3, axis=0)
        B[:, :2] += nb
    m["ball"], m["players"] = B, P
    return m


_degrade_default = degrade


def degrade(m, rng, gap_lengths, target_missing, player_drop=0.05, noise=0.15, profile=None):  # noqa: F811
    from mtah.degrade_profiles import degrade_profile
    return degrade_profile(m, rng, gap_lengths, target_missing, player_drop, noise, profile=profile,
                           default=_degrade_default)


def match_xy(m, rng, gaps, miss, step):
    rows_X, rows_y, rows_g = [], [], []
    for variant in (m, swap_teams(m)):
        d = degrade(variant, rng, gaps, miss)
        X, names = build(d)
        y = variant["label"]
        keep = np.flatnonzero((y < 3) & (variant["half"] > 0))[::step]
        rows_X.append(X[keep])
        rows_y.append(y[keep])
        rows_g.append(keep)
    return rows_X, rows_y, names


def spell_f1(labels, gt, half):
    runs = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        runs.append((int(idx[0]), int(idx[-1]) + 1))
    gt = gt.astype(np.int64).copy()
    gt[half == 0] = 3
    return pm.spell_scores(gt, labels.astype(np.int64), runs)["mean_f1"]


def decode_match(prob, half, trans):
    out = np.full(len(prob), 3, np.int8)
    lp = np.log(np.clip(prob, 1e-4, 1))
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        out[idx] = viterbi(lp[idx], trans)
    return out


def symmetric_predict(model, m):
    X, _ = build(m)
    p = model.predict_proba(X)
    Xs, _ = build(swap_teams(m))
    ps = model.predict_proba(Xs)[:, [1, 0, 2]]
    return (p + ps) / 2


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-degrade", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    gaps, miss = our_gap_lengths()
    if args.no_degrade:
        miss = 0.0
    matches = load_ext(args.data)
    print("external matches:", [m["name"] for m in matches], "target ball missing %.2f" % miss)
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1,
                  random_state=args.seed)

    data = []
    for m in matches:
        t = time.time()
        Xs, ys, names = match_xy(m, rng, gaps, miss, args.step)
        data.append((np.concatenate(Xs), np.concatenate(ys)))
        print("  %s %d rows %.0fs" % (m["name"], len(data[-1][1]), time.time() - t))

    oof = []
    for i, m in enumerate(matches):
        Xtr = np.concatenate([d[0] for j, d in enumerate(data) if j != i])
        ytr = np.concatenate([d[1] for j, d in enumerate(data) if j != i])
        clf = lgb.LGBMClassifier(**params).fit(Xtr, ytr)
        dm = degrade(m, np.random.default_rng(100 + i), gaps, miss)
        p = symmetric_predict(clf, dm)
        oof.append((m, p))
        y = m["label"]
        ok = (y < 3) & (m["half"] > 0)
        print("  OOF %s frame acc %.3f" % (m["name"], np.mean(np.argmax(p[ok], 1) == y[ok])))

    best = None
    grid = [(s, d) for s in (4, 6, 8, 12, 16, 24) for d in (4, 8, 12, 20, 30)]
    for s, d in grid:
        tr = make_trans(s, d, d)
        f1 = np.mean([spell_f1(decode_match(p, m["half"], tr), m["label"], m["half"])
                      for m, p in oof])
        if best is None or f1 > best[0]:
            best = (f1, s, d)
    print("best decoder on external OOF: spellF1 %.4f switch=%s dead=%s" % best)
    for m, p in oof:
        lab = decode_match(p, m["half"], make_trans(best[1], best[2], best[2]))
        print("   %s spellF1 %.4f" % (m["name"], spell_f1(lab, m["label"], m["half"])))

    clf = lgb.LGBMClassifier(**params).fit(np.concatenate([d[0] for d in data]),
                                           np.concatenate([d[1] for d in data]))
    ours = provided_match()
    prob = symmetric_predict(clf, ours)
    from mtah.decode import decode
    lab = decode(prob, make_trans(best[1], best[2], best[2]))
    name = os.path.basename(args.out.rstrip("/"))
    path = write_submission(os.path.join(ROOT, "evaluation", "results", name + ".csv"), lab)
    np.savez(os.path.join(ROOT, "runs", name + ".signals.npz"),
             p_home=prob[:, 0], p_away=prob[:, 1], p_dead=prob[:, 2])
    json.dump({"decoder": {"switch": best[1], "dead": best[2], "ext_oof_spellf1": best[0]},
               "data": [m["name"] for m in matches], "params": params, "features": names},
              open(os.path.join(args.out, "config.json"), "w"), indent=1)
    imp = sorted(zip(clf.feature_importances_, names), reverse=True)[:20]
    print("top features:", ", ".join(n for _, n in imp))
    print("wrote", path)


if __name__ == "__main__":
    main()
