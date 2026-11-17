import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.refine.sample import OUT, issia_inputs, track_baseline  # noqa: E402
from vision.refine.sim import CACHE  # noqa: E402

FOLDS = {"A": ("117093_1st", "118576_1st", "118578_1st"), "B": ("118575_1st", "118577_1st", "128058_1st")}
REF = "f%s_kin10_argmax"
OTHERS = ("f%s_kin0_argmax", "f%s_kin3_argmax", "f%s_kin30_argmax", "f%s_kin10_sample", "f%sv3tied_oneshot")
GRID = np.linspace(0.02, 1.0, 50)


def scores(fold, name):
    r = np.load(os.path.join(OUT, "eval", REF % fold, name + ".npz"))
    xy = r["xy"].astype(np.float64)
    o1 = np.load(os.path.join(OUT, "eval", "f%sv3tied_oneshot" % fold, name + ".npz"))
    agree = []
    for t in OTHERS:
        q = np.load(os.path.join(OUT, "eval", t % fold, name + ".npz"))["xy"].astype(np.float64)
        agree.append(np.hypot(*(q - xy).T) < 2.0)
    com = np.sum(agree, 0).astype(np.float64)
    pm = r["pmax"].astype(np.float64)
    e1 = -o1["ent"].astype(np.float64)
    isdet = r["is_cand"].astype(np.float64)
    tie = lambda a, b: a + 1e-3 * (b - np.nanmin(b)) / max(np.nanmax(b) - np.nanmin(b), 1e-9)
    return xy, {"pmax": pm, "neg_ent1": e1, "agree1": tie(agree[-1].astype(np.float64), e1),
                "committee": tie(com, pm), "det": tie(isdet, pm),
                "committee_x_det": tie(com + 0.5 * isdet, pm)}


def curve(score, err, cov_mask, truth_mask):
    s = score[cov_mask]
    rows = []
    for c in GRID:
        thr = np.quantile(s, 1 - c) if c < 1 else -np.inf
        cov = cov_mask & (score >= thr)
        tm = truth_mask & cov
        e = err[tm]
        rows.append((float(cov[cov_mask].mean()), float(np.mean(e > 3)) if len(e) else np.nan,
                     float(np.mean(e > 5)) if len(e) else np.nan, int(tm.sum()), float(thr)))
    return rows


def at_wrong(rows, target):
    ok = [r for r in rows if np.isfinite(r[1]) and r[1] <= target]
    return max(ok, key=lambda r: r[0]) if ok else None


def main():
    out = {"stv2": {}, "issia": {}}
    pool = {}
    base_all, base_gap = [], []
    for fold, halves in FOLDS.items():
        for hn in halves:
            z = np.load(os.path.join(OUT, "stv2_%s.npz" % hn))
            A = z["anchor"].astype(np.float64)
            tm = ~np.isnan(A[:, 0])
            pl = (~np.isnan(z["players"][:, :, 0].astype(np.float32))).any(1)
            b = np.load(os.path.join(CACHE, "stv2", hn, "ball.npz"))
            bst = b["state"]
            xy, sc = scores(fold, hn)
            err = np.hypot(*(xy - A).T)
            berr = np.hypot(*(b["xy"].astype(np.float64) - A).T)
            gap = bst == 0
            pool.setdefault("err", []).append(err)
            pool.setdefault("tm", []).append(tm)
            pool.setdefault("pl", []).append(pl)
            pool.setdefault("gap", []).append(gap)
            pool.setdefault("berr", []).append(berr)
            pool.setdefault("bcov", []).append(bst > 0)
            for k, v in sc.items():
                r = np.full(len(v), np.nan)
                r[pl] = (np.argsort(np.argsort(v[pl], kind="stable"), kind="stable") + 0.5) / pl.sum()
                pool.setdefault("s_" + k, []).append(np.nan_to_num(r, nan=-1))
    P = {k: np.concatenate(v) for k, v in pool.items()}
    bc = P["bcov"] & P["pl"]
    btm = P["tm"] & bc
    base = {"coverage": float(bc[P["pl"]].mean()), "wrong3": float(np.mean(P["berr"][btm] > 3)),
            "wrong5": float(np.mean(P["berr"][btm] > 5)), "truth_covered": int(btm.sum())}
    out["stv2"]["tracker"] = base
    print("stv2 tracker: coverage %.3f wrong>3m %.3f wrong>5m %.3f" % (base["coverage"], base["wrong3"], base["wrong5"]))
    gapm = P["gap"] & P["pl"]
    for k in [k for k in P if k.startswith("s_")]:
        rows = curve(P[k], P["err"], P["pl"], P["tm"])
        rows_gap = curve(P[k], P["err"], gapm, P["tm"])
        op = at_wrong(rows, base["wrong3"])
        opg = at_wrong(rows_gap, base["wrong3"])
        out["stv2"][k[2:]] = {"curve": rows, "curve_gap": rows_gap, "op_at_tracker_wrong3": op, "op_gap": opg}
        print("stv2 %-16s at tracker wrong3 %.3f: coverage %s (wrong5 %s) | gap frames: coverage %s (wrong3 %s)" % (
            k[2:], base["wrong3"], None if op is None else round(op[0], 3), None if op is None else round(op[2], 3),
            None if opg is None else round(opg[0], 3), None if opg is None else round(opg[1], 3)))
    z, n, cand, Pl, (pfr, px, py) = issia_inputs()
    truth = z["truth"].astype(np.float64)
    tm = z["truth_src"] > 0
    bxy, bst = track_baseline(z["c_frame"].astype(np.int64), z["c_x"].astype(np.float64), z["c_y"].astype(np.float64),
                              z["c_conf"], n, pfr, px, py)
    berr = np.hypot(*(bxy - truth).T)
    bc = bst > 0
    ib = {"coverage": float(bc.mean()), "wrong3": float(np.mean(berr[tm & bc] > 3)),
          "wrong5": float(np.mean(berr[tm & bc] > 5))}
    out["issia"]["tracker"] = ib
    print("issia tracker: coverage %.3f wrong>3m %.3f wrong>5m %.3f" % (ib["coverage"], ib["wrong3"], ib["wrong5"]))
    allf = np.ones(n, bool)
    for fold in FOLDS:
        xy, sc = scores(fold, "issia")
        err = np.hypot(*(xy - truth).T)
        for k, v in sc.items():
            rows = curve(v, err, allf, tm)
            rows_gap = curve(v, err, ~bc, tm)
            op, opg = at_wrong(rows, ib["wrong3"]), at_wrong(rows_gap, ib["wrong3"])
            out["issia"]["%s_%s" % (k, fold)] = {"curve": rows, "curve_gap": rows_gap, "op": op, "op_gap": opg}
            print("issia[%s] %-16s coverage %s (wrong3 %s) | gap: coverage %s (wrong3 %s)" % (
                fold, k, None if op is None else round(op[0], 3), None if op is None else round(op[1], 3),
                None if opg is None else round(opg[0], 3), None if opg is None else round(opg[1], 3)))
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"))


if __name__ == "__main__":
    main()
