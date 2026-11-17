import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from vision.players_common import EXT, load_sn_seq, fit_ground_h, h_apply  # noqa: E402

OUT = "/mnt/x/Data/AI/more-than-a-hack/trackK2/gsr_packs"
PEVAL = os.path.join(ROOT, "cache", "vision", "players_eval")


def build(args):
    split, sd = args
    seq = os.path.basename(sd)
    out = os.path.join(OUT, "%s_%s.npz" % (split, seq))
    if os.path.exists(out):
        return out
    det_f = os.path.join(PEVAL, "trk_ours_s15" + ("" if split == "valid" else "_train"), seq + ".npz")
    if not os.path.exists(det_f):
        return None
    data = dict(np.load(det_f, allow_pickle=True))
    per, files, _ = load_sn_seq(sd)
    n = len(files)
    Hs, last = [], None
    for k in range(n):
        Hm = fit_ground_h(per[k])
        last = Hm if Hm is not None else last
        Hs.append(last)
    if Hs[0] is None:
        first = next(h for h in Hs if h is not None)
        Hs = [h if h is not None else first for h in Hs]
    dets, feats = list(data["dets"]), list(data["feats"])
    F, XY, C, FT, BX = [], [], [], [], []
    for k in range(n):
        d = dets[k]
        x, y = h_apply(Hs[k], (d[:, 0] + d[:, 2]) / 2, d[:, 3])
        ok = (x > -3) & (x < 108) & (y > -3) & (y < 71) & np.isfinite(x)
        F.append(np.full(ok.sum(), k, np.int32))
        XY.append(np.c_[x[ok], y[ok]])
        C.append(d[ok, 4])
        FT.append(feats[k][ok].astype(np.float32))
        BX.append(d[ok, :4])
    g = [(np.full(len(per[k]["tid"]), k), per[k]) for k in range(n)]
    np.savez_compressed(
        out, n=n, det_f=np.concatenate(F), det_xy=np.concatenate(XY).astype(np.float64),
        det_conf=np.concatenate(C).astype(np.float32), det_feat=np.concatenate(FT).astype(np.float32),
        det_box=np.concatenate(BX).astype(np.float64), H=np.array(Hs), s=float(data["s"]),
        gt_f=np.concatenate([a for a, _ in g]).astype(np.int32),
        gt_tid=np.concatenate([p["tid"] for _, p in g]), gt_box=np.concatenate([p["boxes"] for _, p in g]),
        gt_role=np.concatenate([p["role"] for _, p in g]), gt_team=np.concatenate([p["team"] for _, p in g]),
        gt_xy=np.concatenate([p["pitch"] for _, p in g]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [("valid", sd) for sd in sorted(glob.glob(os.path.join(EXT, "sn_gsr_valid", "SNGS-*")))]
    jobs += [("train", sd) for sd in sorted(glob.glob(os.path.join(EXT, "sn_gsr", "SNGS-*")))]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(build, jobs):
            print(r, flush=True)


if __name__ == "__main__":
    main()
