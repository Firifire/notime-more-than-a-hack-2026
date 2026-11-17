import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.refine.sample import metrics, coverage_threshold, issia_inputs, track_baseline, OUT  # noqa: E402
from vision.refine.sim import CACHE  # noqa: E402

KS = ("recall_1.5m", "recall_3m", "recall_5m", "coverage_at_truth", "median_err_covered_m",
      "wrong_object_gt10m_of_covered", "jumps_gt3m_per_min_covered", "false_jumps_per_1k_truth_pairs")


def fill_all(xy):
    xy = xy.astype(np.float64).copy()
    ok = np.isfinite(xy[:, 0])
    if not ok.any():
        return xy
    f = np.flatnonzero(ok)
    t = np.arange(len(xy))
    for k in range(2):
        xy[:, k] = np.interp(t, f, xy[f, k])
    return xy


def pool(rows):
    tot = sum(r["truth_frames"] for r in rows)
    out = {"truth_frames": tot}
    for m in ("recall_1.5m", "recall_3m", "recall_5m", "coverage_at_truth"):
        out[m] = round(sum(r[m] * r["truth_frames"] for r in rows) / tot, 4)
    for m in ("median_err_covered_m", "wrong_object_gt10m_of_covered", "jumps_gt3m_per_min_covered"):
        out[m + "_mean"] = round(float(np.mean([r[m] for r in rows if r[m] is not None])), 3)
    if all("truth_pairs" in r for r in rows):
        tp = sum(r["truth_pairs"] for r in rows)
        out["false_jumps_per_1k_truth_pairs"] = round(
            sum(r["false_jumps_per_1k_truth_pairs"] * r["truth_pairs"] for r in rows) / max(tp, 1), 3)
    return out


def main():
    a = sys.argv[1:]
    out_path = a[a.index("--out") + 1] if "--out" in a else None
    one = a[a.index("--oneshot") + 1:a.index("--oneshot") + 3] if "--oneshot" in a else []
    main_tags = [t for t in a[:a.index("--oneshot")]] if "--oneshot" in a else a
    res = {"main": main_tags, "oneshot": one, "stv2": {}, "issia": {}}
    rows = {k: [] for k in ("baseline", "baseline_fill", "refiner_full", "refiner_cov1shot")}
    for ti, t in enumerate(main_tags):
        rep = json.load(open(os.path.join(OUT, "eval", t, "report.json")))
        for hn in rep["halves"]:
            z = np.load(os.path.join(OUT, "stv2_%s.npz" % hn))
            A = z["anchor"].astype(np.float64)
            tm = ~np.isnan(A[:, 0])
            pl = (~np.isnan(z["players"][:, :, 0].astype(np.float32))).any(1)
            b = np.load(os.path.join(CACHE, "stv2", hn, "ball.npz"))
            r = np.load(os.path.join(OUT, "eval", t, "%s.npz" % hn))
            rows["baseline"].append(metrics(b["xy"].astype(np.float64), b["state"] > 0, A, tm, pl))
            rows["baseline_fill"].append(metrics(fill_all(b["xy"]), pl, A, tm, pl))
            rows["refiner_full"].append(metrics(r["xy"].astype(np.float64), pl, A, tm, pl))
            if one:
                r1 = np.load(os.path.join(OUT, "eval", one[ti], "%s.npz" % hn))
                thr = coverage_threshold(r1["ent"], pl, float((b["state"][pl] > 0).mean()))
                rows["refiner_cov1shot"].append(metrics(r["xy"].astype(np.float64), pl & (r1["ent"] <= thr), A, tm, pl))
    res["stv2"] = {k: pool(v) for k, v in rows.items() if v}
    z, n, cand, P, (pfr, px, py) = issia_inputs()
    truth = z["truth"].astype(np.float64)
    tm = z["truth_src"] > 0
    bxy, bst = track_baseline(z["c_frame"].astype(np.int64), z["c_x"].astype(np.float64), z["c_y"].astype(np.float64),
                              z["c_conf"], n, pfr, px, py)
    allf = np.ones(n, bool)
    res["issia"]["baseline"] = metrics(bxy, bst > 0, truth, tm)
    res["issia"]["baseline_fill"] = metrics(fill_all(bxy), allf, truth, tm)
    for ti, t in enumerate(main_tags):
        r = np.load(os.path.join(OUT, "eval", t, "issia.npz"))
        res["issia"]["refiner_full[%s]" % t] = metrics(r["xy"].astype(np.float64), allf, truth, tm)
        if one:
            r1 = np.load(os.path.join(OUT, "eval", one[ti], "issia.npz"))
            thr = coverage_threshold(r1["ent"], allf, float((bst > 0).mean()))
            res["issia"]["refiner_cov1shot[%s]" % t] = metrics(r["xy"].astype(np.float64), r1["ent"] <= thr, truth, tm)
    for k, v in res["stv2"].items():
        print("stv2  %-22s" % k, " ".join("%s %s" % (m.replace("_covered", "").replace("_of", ""), v[m]) for m in v))
    for k, v in res["issia"].items():
        print("issia %-30s" % k, " ".join("%s %s" % (m, v[m]) for m in KS))
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
