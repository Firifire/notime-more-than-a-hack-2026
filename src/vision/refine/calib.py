import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.refine.sample import OUT, issia_inputs, track_baseline  # noqa: E402
from vision.refine.sim import CACHE, pack, player_feats  # noqa: E402
from vision.refine.confidence import FOLDS, REF, OTHERS, curve, at_wrong  # noqa: E402

FEATS = ("ent1", "pmax", "is_det", "is_det1", "chosen_conf", "top_conf", "n_cand", "d_player",
         "since_det", "until_det", "run_det", "speed5", "agree1_d", "d_topconf")
MODEL = os.path.join(OUT, "calib_refD4.txt")
CONF_DIR = os.path.join(OUT, "eval", "calib_conf")


def features(fold, name, cand, players):
    r = np.load(os.path.join(OUT, "eval", REF % fold, name + ".npz"))
    o1 = np.load(os.path.join(OUT, "eval", "f%sv3tied_oneshot" % fold, name + ".npz"))
    return r["xy"].astype(np.float64), features_from(r, o1, cand, players)


def features_from(r, o1, cand, players):
    xy = r["xy"].astype(np.float64)
    n = len(xy)
    isd = r["is_cand"].astype(bool)
    tok = r["tok"]
    cc = np.where(isd, cand[np.arange(n), np.clip(tok, 0, 7), 2], 0.0)
    top = np.nan_to_num(cand[:, 0, 2])
    nc = (~np.isnan(cand[..., 2])).sum(1)
    P = players[..., :2].astype(np.float64)
    dp = np.hypot(P[..., 0] - xy[:, None, 0], P[..., 1] - xy[:, None, 1])
    dp = np.nanmin(np.where(np.isnan(dp), 1e3, dp), 1)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(isd, idx, -10 ** 6))
    nxt = np.minimum.accumulate(np.where(isd, idx, 10 ** 7)[::-1])[::-1]
    since = np.clip(idx - last, 0, 500)
    until = np.clip(nxt - idx, 0, 500)
    d = np.diff(np.r_[0, isd.astype(np.int8), 0])
    run = np.zeros(n)
    s_ = np.flatnonzero(d != 0)
    for a, b in zip(s_[:-1], s_[1:]):
        run[a:b] = b - a
    sp = np.zeros(n)
    sp[5:-5] = np.hypot(*(xy[10:] - xy[:-10]).T) / 10 * 25
    a1 = np.hypot(*(o1["xy"].astype(np.float64) - xy).T)
    dt = np.hypot(*(np.nan_to_num(cand[:, 0, :2], nan=-100) - xy).T)
    F = np.c_[o1["ent"], r["pmax"], isd, o1["is_cand"], cc, top, nc, np.minimum(dp, 100), since, until, run, sp,
              np.minimum(a1, 100), np.minimum(dt, 200)].astype(np.float32)
    return F


def stv2_rows(fold):
    out = []
    for hn in FOLDS[fold]:
        z = np.load(os.path.join(OUT, "stv2_%s.npz" % hn))
        n = int(z["n"])
        cand = pack(z["c_frame"].astype(np.int64), z["c_x"], z["c_y"], z["c_conf"], n)
        P = player_feats(z["players"].astype(np.float32))
        xy, F = features(fold, hn, cand, P)
        A = z["anchor"].astype(np.float64)
        pl = (~np.isnan(z["players"][:, :, 0].astype(np.float32))).any(1)
        b = np.load(os.path.join(CACHE, "stv2", hn, "ball.npz"))
        out.append(dict(hn=hn, F=F, err=np.hypot(*(xy - A).T), tm=~np.isnan(A[:, 0]), pl=pl, gap=b["state"] == 0,
                        berr=np.hypot(*(b["xy"].astype(np.float64) - A).T), bcov=b["state"] > 0))
    return out


def fit(rows):
    import lightgbm as lgb
    X = np.concatenate([r["F"][r["tm"] & r["pl"]] for r in rows])
    y = np.concatenate([(r["err"][r["tm"] & r["pl"]] <= 3).astype(int) for r in rows])
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=6,
                           deterministic=True, force_row_wise=True, verbose=-1)
    m.fit(X, y)
    return m


def main():
    from sklearn.metrics import roc_auc_score
    rows = {f: stv2_rows(f) for f in FOLDS}
    models = {f: fit(rows[f]) for f in FOLDS}
    out = {"features": FEATS}
    pooled = {}
    for f, other in (("A", "B"), ("B", "A")):
        for r in rows[f]:
            r["conf"] = models[other].predict_proba(r["F"])[:, 1]
            for k in ("conf", "err", "tm", "pl", "gap", "berr", "bcov"):
                pooled.setdefault(k, []).append(r[k] if k != "conf" else
                                                np.where(r["pl"], r["conf"], -1.0))
    os.makedirs(CONF_DIR, exist_ok=True)
    for f in FOLDS:
        for r in rows[f]:
            np.save(os.path.join(CONF_DIR, r["hn"] + ".npy"), r["conf"].astype(np.float32))
    final = fit(rows["A"] + rows["B"])
    final.booster_.save_model(MODEL)
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    m = P["tm"] & P["pl"]
    auc = roc_auc_score(P["err"][m] <= 3, P["conf"][m])
    gm = m & P["gap"]
    auc_gap = roc_auc_score(P["err"][gm] <= 3, P["conf"][gm])
    bc = P["bcov"] & P["pl"]
    base = {"coverage": float(bc[P["pl"]].mean()), "wrong3": float(np.mean(P["berr"][P["tm"] & bc] > 3)),
            "wrong5": float(np.mean(P["berr"][P["tm"] & bc] > 5))}
    rows_all = curve(P["conf"], P["err"], P["pl"], P["tm"])
    rows_gap = curve(P["conf"], P["err"], P["gap"] & P["pl"], P["tm"])
    out["stv2"] = {"auc": auc, "auc_gap": auc_gap, "tracker": base, "curve": rows_all, "curve_gap": rows_gap,
                   "op": at_wrong(rows_all, base["wrong3"]), "op_gap": at_wrong(rows_gap, base["wrong3"])}
    print("stv2 cross-fitted calibrator: AUC %.3f (gap frames %.3f)" % (auc, auc_gap))
    print("stv2 tracker coverage %.3f wrong3 %.3f; calibrated refiner at that wrong rate: %s; gap frames: %s" % (
        base["coverage"], base["wrong3"], out["stv2"]["op"], out["stv2"]["op_gap"]))
    print("  all curve", [(round(r[0], 2), round(r[1], 3), round(r[2], 3)) for r in rows_all[::5]])
    print("  gap curve", [(round(r[0], 2), round(r[1], 3), round(r[2], 3)) for r in rows_gap[::5]])
    imp = {f: dict(zip(FEATS, models[f].feature_importances_.tolist())) for f in FOLDS}
    out["importance"] = imp
    z, n, cand, Pl, (pfr, px, py) = issia_inputs()
    truth = z["truth"].astype(np.float64)
    tm = z["truth_src"] > 0
    bxy, bst = track_baseline(z["c_frame"].astype(np.int64), z["c_x"].astype(np.float64), z["c_y"].astype(np.float64),
                              z["c_conf"], n, pfr, px, py)
    berr = np.hypot(*(bxy - truth).T)
    bc = bst > 0
    ib = {"coverage": float(bc.mean()), "wrong3": float(np.mean(berr[tm & bc] > 3))}
    out["issia"] = {"tracker": ib}
    for f, other in (("A", "B"), ("B", "A")):
        xy, F = features(f, "issia", cand, Pl)
        conf = models[other].predict_proba(F)[:, 1]
        err = np.hypot(*(xy - truth).T)
        ra = curve(conf, err, np.ones(n, bool), tm)
        rg = curve(conf, err, ~bc, tm)
        out["issia"][f] = {"auc": roc_auc_score(err[tm] <= 3, conf[tm]), "curve": ra, "curve_gap": rg,
                           "op": at_wrong(ra, ib["wrong3"]), "op_gap": at_wrong(rg, ib["wrong3"])}
        print("issia[%s] AUC %.3f tracker cov %.3f wrong3 %.3f -> refiner %s | gap %s" % (
            f, out["issia"][f]["auc"], ib["coverage"], ib["wrong3"], out["issia"][f]["op"], out["issia"][f]["op_gap"]))
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"))


if __name__ == "__main__":
    main()
