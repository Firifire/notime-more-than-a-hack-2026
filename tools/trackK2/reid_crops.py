import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from vision.players_common import EXT, iou_matrix  # noqa: E402
from gsr_eval import load_pack, PACKS  # noqa: E402

OUT = "/mnt/x/Data/AI/more-than-a-hack/trackK2/reid_crops"
CH, CW = 64, 32


def crop_one(img, b):
    x0, y0, x1, y1 = b
    h = max(y1 - y0, 4.0)
    w = max(x1 - x0, 2.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hh, ww = h * 1.1, max(w * 1.1, h * 1.1 / 2)
    a0, a1 = int(round(cx - ww / 2)), int(round(cx + ww / 2))
    b0, b1 = int(round(cy - hh / 2)), int(round(cy + hh / 2))
    H, W = img.shape[:2]
    pad = np.zeros((b1 - b0, a1 - a0, 3), np.uint8)
    sa0, sa1, sb0, sb1 = max(a0, 0), min(a1, W), max(b0, 0), min(b1, H)
    if sa1 > sa0 and sb1 > sb0:
        pad[sb0 - b0:sb1 - b0, sa0 - a0:sa1 - a0] = img[sb0:sb1, sa0:sa1]
    return cv2.resize(pad, (CW, CH), interpolation=cv2.INTER_AREA if hh > CH else cv2.INTER_LINEAR)


def build(args):
    f, every = args
    name = os.path.basename(f)
    out = os.path.join(OUT, name)
    if os.path.exists(out):
        return out
    split, seq = name.replace(".npz", "").split("_", 1)
    sd = os.path.join(EXT, "sn_gsr_valid" if split == "valid" else "sn_gsr", seq)
    z = np.load(f)
    s = float(z["s"])
    n, xy, cf, ft, bx, per, Hs = load_pack(f)
    files = sorted(os.listdir(os.path.join(sd, "img1")))
    F, D, C, G, T, R = [], [], [], [], [], []
    for k in range(0, n, every):
        if len(bx[k]) == 0:
            continue
        img = cv2.imread(os.path.join(sd, "img1", files[k]))
        if s != 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g = np.full(len(bx[k]), -1)
        tm = np.full(len(bx[k]), -1)
        rl = np.full(len(bx[k]), -1)
        if len(per[k]["tid"]):
            S = iou_matrix(bx[k], per[k]["boxes"])
            a, b = linear_sum_assignment(-S)
            ok = S[a, b] >= 0.5
            g[a[ok]] = per[k]["tid"][b[ok]]
            tm[a[ok]] = per[k]["team"][b[ok]]
            rl[a[ok]] = per[k]["role"][b[ok]]
        for d in range(len(bx[k])):
            C.append(crop_one(img, bx[k][d] * s))
            F.append(k)
            D.append(d)
        G.append(g)
        T.append(tm)
        R.append(rl)
    os.makedirs(OUT, exist_ok=True)
    np.savez(out, frame=np.array(F, np.int32), det=np.array(D, np.int16), crop=np.array(C, np.uint8),
             gt=np.concatenate(G).astype(np.int32), team=np.concatenate(T).astype(np.int8),
             role=np.concatenate(R).astype(np.int8), every=every)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(PACKS, "*_SNGS-*.npz")))
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(build, [(f, a.every) for f in files]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
