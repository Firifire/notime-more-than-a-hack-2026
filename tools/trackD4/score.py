import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools", "trackK2"))
from vision import track_players as TP  # noqa: E402
from vision.players_common import iou_matrix  # noqa: E402
import k2track as K2  # noqa: E402

METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDsw", "pred_ids_per_gt_id", "n_pred_ids")


def score(tracks, res, mode="v1"):
    n = res["n"]
    frames = np.arange(n)
    if not tracks:
        tracks = {}
    if mode == "v1":
        dense = TP.densify(tracks, frames, res["xy"], res["bx"])
    else:
        dense = K2.densify_v2(tracks, frames, res["xy"], res["bx"])
    pf = [([], []) for _ in range(n)]
    for tid, r in dense.items():
        key = r["src"] if mode == "v2" else tid
        for t, b in zip(r["t"], r["box"]):
            pf[t][0].append(key)
            pf[t][1].append(b)
    pr = [(np.array(i, int), np.array(b).reshape(-1, 4)) for i, b in pf]
    gt = res["gt"]
    r = TP._hota_alignment(gt, pr, iou_matrix)
    r.update(TP.clear_idf1(gt, pr, iou_matrix))
    frag = {}
    for (gi, gb), (pi, pb) in zip(gt, pr):
        if len(gi) == 0 or len(pi) == 0:
            continue
        S = iou_matrix(gb, pb)
        a_, b_ = linear_sum_assignment(-S)
        for i, j in zip(a_, b_):
            if S[i, j] >= 0.5:
                frag.setdefault(gi[i], set()).add(pi[j])
    r["pred_ids_per_gt_id"] = float(np.mean([len(v) for v in frag.values()])) if frag else 0
    r["n_pred_ids"] = len({k for i, _ in pr for k in i})
    out = {}
    for k in METRICS:
        v = r.get(k, r.get(k.lower(), np.nan))
        out[k] = float(v) if np.isscalar(v) else np.nan
    return out


def boot(diff, B=2000, seed=0):
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
