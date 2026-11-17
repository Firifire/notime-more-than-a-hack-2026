import argparse
import json
import os
import time

import numpy as np

from mtah.data import ROOT, write_submission, HALVES
from mtah.decode import viterbi_tv, boundary_costs
from mtah.eval_oof import gate
from mtah.featset import build, swap_teams
from mtah.match import provided_match
from mtah.train_ext import load_ext, our_gap_lengths, degrade, spell_f1

PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=12)
if os.environ.get("MTAH_LGB_DEVICE", "cpu") == "cuda":
    PARAMS.update(device_type="cuda", n_jobs=4)
else:
    PARAMS.update(deterministic=True, force_row_wise=True)
W = 6
HEADS = ("dead", "team", "sw", "on", "off")


SIGMA = 3.0


def near(mask, w=W, sigma=SIGMA):
    from scipy.ndimage import maximum_filter1d
    out = np.zeros(len(mask), np.float32)
    ev = np.flatnonzero(mask)
    for d in range(-w, w + 1):
        idx = ev + d
        idx = idx[(idx >= 0) & (idx < len(mask))]
        out[idx] = np.maximum(out[idx], np.exp(-d * d / (2 * sigma * sigma)))
    return out


def boundary_labels(y, half):
    yy = y.astype(np.int64)
    prev = np.r_[yy[0], yy[:-1]]
    same_half = np.r_[False, half[1:] == half[:-1]] & (half > 0)
    sw = same_half & (yy < 2) & (prev < 2) & (yy != prev)
    on = same_half & (yy == 2) & (prev < 2)
    off = same_half & (yy < 2) & (prev == 2)
    return near(sw), near(on), near(off)


def rows_for(m, rng, gaps, miss, step):
    out = []
    for v in (m, swap_teams(m)):
        d = degrade(v, rng, gaps, miss)
        X, names = build(d)
        y = v["label"]
        sw, on, off = boundary_labels(y, v["half"])
        keep = np.flatnonzero((y < 3) & (v["half"] > 0))[::step]
        out.append((X[keep], y[keep], sw[keep], on[keep], off[keep]))
    return out, names


def fit(rows, dead_set, team_set):
    import lightgbm as lgb
    def cat(sel, f):
        return np.concatenate([f(r) for r in rows if sel(r[0])])
    D = lambda n: n in dead_set
    T = lambda n: n in team_set
    models = {}
    Xd = cat(D, lambda r: r[1])
    models["dead"] = lgb.LGBMClassifier(**PARAMS).fit(Xd, cat(D, lambda r: (r[2] == 2).astype(int)))
    soft = dict(PARAMS, objective="cross_entropy")
    models["on"] = lgb.LGBMRegressor(**soft).fit(Xd, cat(D, lambda r: r[4]))
    models["off"] = lgb.LGBMRegressor(**soft).fit(Xd, cat(D, lambda r: r[5]))
    Xt = cat(T, lambda r: r[1][r[2] < 2])
    models["team"] = lgb.LGBMClassifier(**PARAMS).fit(Xt, cat(T, lambda r: (r[2][r[2] < 2] == 0).astype(int)))
    models["sw"] = lgb.LGBMRegressor(**soft).fit(Xt, cat(T, lambda r: r[3][r[2] < 2]))
    return models


def predict(models, m=None, X=None, Xs=None):
    if X is None:
        X, _ = build(m)
        Xs, _ = build(swap_teams(m))
    p = {}
    p["dead"] = (models["dead"].predict_proba(X)[:, 1] + models["dead"].predict_proba(Xs)[:, 1]) / 2
    for k in ("sw", "on", "off"):
        p[k] = np.clip((models[k].predict(X) + models[k].predict(Xs)) / 2, 0, 1)
    p["team"] = (models["team"].predict_proba(X)[:, 1] + 1 - models["team"].predict_proba(Xs)[:, 1]) / 2
    return p


def to_prob(p):
    return np.stack([(1 - p["dead"]) * p["team"], (1 - p["dead"]) * (1 - p["team"]), p["dead"]], 1)


def decode_heads(p, half, cfg):
    prob = to_prob(p)
    lp = np.log(np.clip(prob, 1e-4, 1))
    out = np.full(len(prob), 3, np.int8)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        c = boundary_costs(len(idx), cfg["switch"], cfg["dead"], cfg["dead"],
                           p["sw"][idx], p["on"][idx], p["off"][idx], beta=cfg["beta"])
        out[idx] = viterbi_tv(lp[idx], c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead-data", nargs="+", default=["idsse-ctrl"])
    ap.add_argument("--team-data", nargs="+", default=["idsse-ctrl", "skillcorner", "metrica"])
    ap.add_argument("--tune-data", nargs="+", default=["idsse-ctrl"])
    ap.add_argument("--out", default="runs/v5")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    gaps, miss = our_gap_lengths()
    matches = load_ext(sorted(set(args.dead_data + args.team_data + args.tune_data)))
    pick = lambda prefs: {m["name"] for m in matches if m["name"].split("_")[0] in prefs}
    dead_set, team_set, tune_set = pick(args.dead_data), pick(args.team_data), pick(args.tune_data)
    from mtah.featcache import feature_paths, load_view
    tune = [m for m in matches if m["name"] in tune_set]
    folds = [tune[i::args.folds] for i in range(args.folds)]
    keys = [(m["name"], sw_, 7 * i + args.seed) for i, m in enumerate(matches) for sw_ in (False, True)]
    keys += [(m["name"], sw_, 1000 + k) for k, fold in enumerate(folds) for m in fold for sw_ in (False, True)]
    t = time.time()
    feats = feature_paths(keys, workers=args.workers)
    print("features for %d views in %.0fs" % (len(keys), time.time() - t), flush=True)
    if args.prepare_only:
        return
    rows = []
    for i, m in enumerate(matches):
        step = 3 if m["name"] in dead_set else 6
        for sw_ in (False, True):
            X, y, half, names = load_view(feats[(m["name"], sw_, 7 * i + args.seed)])
            bsw, bon, boff = boundary_labels(y, half)
            keep = np.flatnonzero((y < 3) & (half > 0))[::step]
            rows.append((m["name"], X[keep].copy(), y[keep], bsw[keep], bon[keep], boff[keep]))
            del X

    oof = []
    for k, fold in enumerate(folds):
        held = {m["name"] for m in fold}
        done = [os.path.join(args.out, "oof_%s.npz" % m["name"]) for m in fold]
        if all(os.path.exists(f) for f in done):
            for m, f in zip(fold, done):
                z = np.load(f)
                oof.append((m, {h: z[h].astype(np.float64) for h in HEADS}))
            print("  fold %d resumed from saved OOF" % k, flush=True)
            continue
        models = fit([r for r in rows if r[0] not in held], dead_set - held, team_set - held)
        for m in fold:
            p = predict(models, X=load_view(feats[(m["name"], False, 1000 + k)])[0],
                        Xs=load_view(feats[(m["name"], True, 1000 + k)])[0])
            oof.append((m, p))
            np.savez_compressed(os.path.join(args.out, "oof_%s.npz" % m["name"]),
                                label=m["label"], half=m["half"],
                                **{h: p[h].astype(np.float32) for h in HEADS})
            print("  OOF %s done" % m["name"], flush=True)

    def mean_f1(cfg):
        return {m["name"]: spell_f1(decode_heads(p, m["half"], cfg), m["label"], m["half"]) for m, p in oof}

    results = []
    for s in (2, 4, 8, 12):
        for d in (8, 20, 40):
            for b in (0.0, 0.5, 1.0, 2.0):
                cfg = {"switch": s, "dead": d, "beta": b}
                r = mean_f1(cfg)
                results.append((float(np.mean(list(r.values()))), cfg, r))
                print("   %s spellF1 %.4f" % (cfg, results[-1][0]), flush=True)
    results.sort(key=lambda x: -x[0])
    best = results[0]
    base = max((x for x in results if x[1]["beta"] == 0.0), key=lambda x: x[0])
    g = gate(base[2], best[2])
    print("best", best[1], "%.4f" % best[0], "| best beta=0", base[1], "%.4f" % base[0], "| gate", g)
    chosen = best if g["accept"] else base

    models = fit(rows, dead_set, team_set)
    import joblib
    joblib.dump(models, os.path.join(args.out, "models.joblib"))
    ours = provided_match()
    p = predict(models, ours)
    lab = np.full(len(p["dead"]), 3, np.int8)
    for _, a, b in HALVES:
        idx = np.arange(a, b + 1)
        sub = {h: p[h][idx] for h in HEADS}
        lab[idx] = decode_heads(sub, np.ones(len(idx), np.int8), chosen[1])
    name = os.path.basename(args.out.rstrip("/"))
    path = write_submission(os.path.join(ROOT, "evaluation", "results", name + ".csv"), lab)
    prob = to_prob(p)
    np.savez(os.path.join(ROOT, "runs", name + ".signals.npz"), p_home=prob[:, 0], p_away=prob[:, 1],
             p_dead=prob[:, 2], p_switch=p["sw"], p_dead_on=p["on"], p_dead_off=p["off"])
    json.dump({"decoder": chosen[1], "oof_spellf1": chosen[0], "gate_vs_beta0": g,
               "dead_data": sorted(dead_set), "team_data": sorted(team_set), "tune_data": sorted(tune_set),
               "params": PARAMS, "features": names}, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
