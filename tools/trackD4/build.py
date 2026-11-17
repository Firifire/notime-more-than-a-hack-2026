import argparse
import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools", "trackK2"))
from vision.players_common import h_apply, iou_matrix  # noqa: E402
from vision import track_players as TP  # noqa: E402
import k2track as K2  # noqa: E402

PACKS = "/mnt/x/Data/AI/more-than-a-hack/trackK2/gsr_packs"
OUT = "/mnt/x/Data/AI/more-than-a-hack/trackD4/tracklets"


def load(path):
    z = np.load(path)
    n = int(z["n"])
    f = z["det_f"]
    idx = [np.flatnonzero(f == t) for t in range(n)]
    xy = [z["det_xy"][i] for i in idx]
    cf = [z["det_conf"][i] for i in idx]
    ft = [z["det_feat"][i] for i in idx]
    bx = [z["det_box"][i] for i in idx]
    gf = z["gt_f"]
    gidx = [np.flatnonzero(gf == t) for t in range(n)]
    gt = [(z["gt_tid"][i], z["gt_box"][i]) for i in gidx]
    gmeta = dict(role={}, team={})
    for tid, r, tm in zip(z["gt_tid"], z["gt_role"], z["gt_team"]):
        gmeta["role"][int(tid)] = int(r)
        gmeta["team"][int(tid)] = int(tm)
    Hinv = [np.linalg.inv(h) for h in z["H"]]

    def in_view(t, pts):
        u, v = h_apply(Hinv[t], pts[:, 0], pts[:, 1])
        return (u > 0) & (u < 1920) & (v > 0) & (v < 1080)
    return dict(n=n, xy=xy, cf=cf, ft=ft, bx=bx, gt=gt, gmeta=gmeta, in_view=in_view)


def det_gt(P):
    out = []
    for t in range(P["n"]):
        g = np.full(len(P["xy"][t]), -1, int)
        gid, gb = P["gt"][t]
        if len(g) and len(gid):
            S = iou_matrix(P["bx"][t], gb)
            a, b = linear_sum_assignment(-S)
            for i, j in zip(a, b):
                if S[i, j] >= 0.5:
                    g[i] = gid[j]
        out.append(g)
    return out


def build(path):
    name = os.path.basename(path)[:-4]
    out = os.path.join(OUT, name + ".pkl")
    if os.path.exists(out):
        return out
    P = load(path)
    n = P["n"]
    frames = np.arange(n)
    trk = TP.run_tracker(frames, P["xy"], P["cf"], P["ft"], in_view=P["in_view"])
    ours = TP.stitch(trk, frames, P["xy"], P["ft"])
    k2ids, pieces = K2.track_v2(frames, P["xy"], P["cf"], P["ft"], tracklets=trk, return_pieces=True)
    dg = det_gt(P)
    maj, pur = {}, {}
    for k, obs in pieces.items():
        ids = np.array([dg[t][d] for t, d in obs])
        u, c = np.unique(ids, return_counts=True)
        j = int(np.argmax(c))
        maj[k] = int(u[j])
        pur[k] = float(c[j] / len(ids))
    res = dict(name=name, n=n, xy=P["xy"], cf=P["cf"], ft=P["ft"], bx=P["bx"], gt=P["gt"], gmeta=P["gmeta"],
               det_gt=dg, ours={int(k): sorted(v) for k, v in ours.items()},
               k2={int(k): sorted(v) for k, v in k2ids.items()},
               pieces={int(k): sorted(v) for k, v in pieces.items()}, maj=maj, pur=pur)
    os.makedirs(OUT, exist_ok=True)
    with open(out + ".tmp", "wb") as fh:
        pickle.dump(res, fh)
    os.replace(out + ".tmp", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(PACKS, "*.npz")))
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(build, paths):
            print(r, flush=True)


if __name__ == "__main__":
    main()
