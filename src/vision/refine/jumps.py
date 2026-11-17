import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.refine.sample import metrics, viterbi, issia_inputs, OUT, VIT_DFREE  # noqa: E402
from vision.refine.train import load_real  # noqa: E402
from vision.refine.model import cell_centres  # noqa: E402
from vision.refine.analyse import pool  # noqa: E402

LAMS = (3.0, 30.0)
CI_LENS = (12, 25, 75, 250)
DFREE = (VIT_DFREE,)
SPIKE_MAX = 5
SPIKE_M = 3.0


def frame_off(r, centres):
    tok = r["tok"]
    off = r["xy"].astype(np.float64) - centres[np.clip(tok - 8, 0, None)]
    off[tok < 8] = 0.0
    return np.nan_to_num(np.clip(off, -1, 1))


def despike(xy, max_len=SPIKE_MAX, thr=SPIKE_M):
    xy = xy.astype(np.float64).copy()
    n = len(xy)
    d = np.r_[0, np.hypot(*np.diff(xy, axis=0).T)]
    big = np.flatnonzero(d > thr)
    for i in big:
        for L in range(1, max_len + 1):
            j = i + L
            if j >= n:
                break
            if d[j] > thr and np.hypot(*(xy[j] - xy[i - 1])) <= thr * (L + 1):
                t = np.arange(1, L + 1) / (L + 1)
                xy[i:j] = xy[i - 1] + t[:, None] * (xy[j] - xy[i - 1])
                d[i:j + 1] = 0
                break
    return xy


def cell_interp(xy, tok, max_len):
    xy = xy.astype(np.float64).copy()
    isc = tok < 8
    n = len(tok)
    d = np.diff(np.r_[0, (~isc).astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if a == 0 or b >= n or b - a > max_len:
            continue
        t = np.arange(1, b - a + 1) / (b - a + 1)
        xy[a:b] = xy[a - 1] + t[:, None] * (xy[b] - xy[a - 1])
    return xy


def score(p):
    return p["recall_3m"] - 0.002 * p["jumps_gt3m_per_min_covered_mean"]


def main():
    a = sys.argv[1:]
    out_path = a[a.index("--out") + 1] if "--out" in a else None
    tags = [t for t in a if not t.startswith("--") and t != out_path]
    centres = cell_centres().numpy()
    res = {"stv2": {}, "issia": {}}
    per_fold = {}
    for t in tags:
        rep = json.load(open(os.path.join(OUT, "eval", t, "report.json")))
        rows = {}
        for hn in rep["halves"]:
            z = np.load(os.path.join(OUT, "stv2_%s.npz" % hn))
            A = z["anchor"].astype(np.float64)
            tm = ~np.isnan(A[:, 0])
            pl = (~np.isnan(z["players"][:, :, 0].astype(np.float32))).any(1)
            cand = load_real(hn)["cand"]
            r = np.load(os.path.join(OUT, "eval", t, "%s.npz" % hn))
            rows.setdefault("raw", []).append(metrics(r["xy"].astype(np.float64), pl, A, tm, pl))
            rows.setdefault("raw_despike", []).append(metrics(despike(r["xy"]), pl, A, tm, pl))
            off = frame_off(r, centres)
            for df in DFREE:
                for lam in LAMS:
                    xv, _ = viterbi(cand, r["top_l"], r["top_i"], off, centres, lam=lam, dfree=df)
                    rows.setdefault("vit_l%g_d%g" % (lam, df), []).append(metrics(xv.astype(np.float64), pl, A, tm, pl))
                    rows.setdefault("vitds_l%g_d%g" % (lam, df), []).append(metrics(despike(xv), pl, A, tm, pl))
                    if lam == 30.0:
                        _, vt = viterbi(cand, r["top_l"], r["top_i"], off, centres, lam=lam, dfree=df)
                        for L in CI_LENS:
                            rows.setdefault("vitdsci%d_l%g" % (L, lam), []).append(
                                metrics(despike(cell_interp(xv, vt, L)), pl, A, tm, pl))
        per_fold[t] = {k: pool(v) for k, v in rows.items()}
    for i, t in enumerate(tags):
        other = tags[1 - i] if len(tags) == 2 else t
        cands = {k: v for k, v in per_fold[other].items() if k != "raw"}
        best = max(cands, key=lambda k: score(cands[k]))
        res["stv2"][t] = {"raw": per_fold[t]["raw"], "chosen_on_other_fold": best, "vit_chosen": per_fold[t][best],
                          "grid": per_fold[t]}
        print("%s raw rec1.5 %.3f rec3 %.3f jumps %.1f | %s -> rec1.5 %.3f rec3 %.3f jumps %.1f" % (
            t, per_fold[t]["raw"]["recall_1.5m"], per_fold[t]["raw"]["recall_3m"],
            per_fold[t]["raw"]["jumps_gt3m_per_min_covered_mean"], best, per_fold[t][best]["recall_1.5m"],
            per_fold[t][best]["recall_3m"], per_fold[t][best]["jumps_gt3m_per_min_covered_mean"]))
    for k in sorted(per_fold[tags[0]]):
        print("  grid %-16s" % k, " | ".join("%s rec1.5 %.3f rec3 %.3f j %.1f" % (
            t, per_fold[t][k]["recall_1.5m"], per_fold[t][k]["recall_3m"], per_fold[t][k]["jumps_gt3m_per_min_covered_mean"])
            for t in tags))
    z, n, cand, P, _ = issia_inputs()
    truth = z["truth"].astype(np.float64)
    tm = z["truth_src"] > 0
    allf = np.ones(n, bool)
    for t in tags:
        r = np.load(os.path.join(OUT, "eval", t, "issia.npz"))
        best = res["stv2"][t]["chosen_on_other_fold"]
        if best == "raw_despike":
            xv = despike(r["xy"])
        elif best.startswith("vitdsci"):
            L = int(best[7:].split("_")[0])
            lam = float(best.split("_l")[1])
            xv, vt = viterbi(cand, r["top_l"], r["top_i"], frame_off(r, centres), centres, lam=lam)
            xv = despike(cell_interp(xv, vt, L))
        else:
            lam, df = float(best.split("_l")[1].split("_d")[0]), float(best.split("_d")[1])
            xv, _ = viterbi(cand, r["top_l"], r["top_i"], frame_off(r, centres), centres, lam=lam, dfree=df)
            if best.startswith("vitds"):
                xv = despike(xv)
        res["issia"][t] = {"raw": metrics(r["xy"].astype(np.float64), allf, truth, tm),
                           "raw_despike": metrics(despike(r["xy"]), allf, truth, tm),
                           best: metrics(xv.astype(np.float64), allf, truth, tm)}
        for k, v in res["issia"][t].items():
            print("ISSIA %s %-14s rec1.5 %.3f rec3 %.3f jumps %.1f wrong %.3f" % (
                t, k, v["recall_1.5m"], v["recall_3m"], v["jumps_gt3m_per_min_covered"], v["wrong_object_gt10m_of_covered"]))
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
