import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.common import ROOT, CACHE  # noqa: E402

EXT = os.path.join(ROOT, "data_external")
TILE = 960
SN_VAL_SEQS = 6


def scale_for(ball_w, rng):
    if ball_w is None:
        return rng.uniform(0.45, 1.0)
    t = rng.uniform(6.0, 15.0)
    return float(np.clip(t / max(ball_w, 1.0), 0.35, 1.3))


def write_sample(img, boxes, s, rng, out_img, out_lbl, ball_xy=None):
    if abs(s - 1.0) > 1e-3:
        img = cv2.resize(img, None, fx=s, fy=s,
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    hh, ww = img.shape[:2]
    tw, th = min(TILE, ww), min(TILE, hh)
    if ball_xy is not None:
        bx, by = ball_xy[0] * s, ball_xy[1] * s
        x0 = int(np.clip(bx - rng.uniform(0.1, 0.9) * tw, 0, ww - tw))
        y0 = int(np.clip(by - rng.uniform(0.1, 0.9) * th, 0, hh - th))
    else:
        x0 = rng.randint(0, ww - tw)
        y0 = rng.randint(0, hh - th)
    crop = img[y0:y0 + th, x0:x0 + tw]
    lines = []
    for c, a, b, cc, d in boxes:
        a, b, cc, d = a * s - x0, b * s - y0, cc * s - x0, d * s - y0
        ia, ib, ic, idd = max(a, 0), max(b, 0), min(cc, tw), min(d, th)
        if ic - ia < 2 or idd - ib < 2:
            continue
        if c == 1 and (ic - ia) * (idd - ib) < 0.4 * (cc - a) * (d - b):
            continue
        if c == 0 and (ic - ia) * (idd - ib) < 0.6 * (cc - a) * (d - b):
            continue
        lines.append("%d %.6f %.6f %.6f %.6f" % (c, (ia + ic) / 2 / tw, (ib + idd) / 2 / th,
                                                 (ic - ia) / tw, (idd - ib) / th))
    cv2.imwrite(out_img, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out_lbl, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def sn_gsr(out, rng, step):
    seqs = sorted(glob.glob(os.path.join(EXT, "sn_gsr", "SNGS-*")))
    n = {"train": 0, "val": 0}
    for k, sd in enumerate(seqs):
        split = "val" if k >= len(seqs) - SN_VAL_SEQS else "train"
        d = json.load(open(os.path.join(sd, "Labels-GameState.json")))
        by_img = {}
        for a in d["annotations"]:
            if a["category_id"] not in (1, 2, 3, 4) or not a.get("bbox_image"):
                continue
            b = a["bbox_image"]
            c = 0 if a["category_id"] == 4 else 1
            by_img.setdefault(a["image_id"], []).append((c, b["x"], b["y"], b["x"] + b["w"],
                                                         b["y"] + b["h"]))
        imgs = d["images"][::step if split == "train" else step * 2]
        for im in imgs:
            boxes = by_img.get(im["image_id"], [])
            balls = [b for b in boxes if b[0] == 0]
            img = cv2.imread(os.path.join(sd, "img1", im["file_name"]))
            if img is None:
                continue
            name = "sn_%s_%s" % (os.path.basename(sd), im["file_name"][:-4])
            if balls:
                b = balls[0]
                s = scale_for(b[3] - b[1], rng)
                xy = ((b[1] + b[3]) / 2, (b[2] + b[4]) / 2)
            else:
                s, xy = scale_for(None, rng), None
            write_sample(img, boxes, s, rng, os.path.join(out, "images", split, name + ".jpg"),
                         os.path.join(out, "labels", split, name + ".txt"), xy)
            n[split] += 1
        print(sd, n, flush=True)
    return n


def roboflow(out, rng):
    n = {"train": 0, "val": 0}
    for sub, split in (("train", "train"), ("valid", "train"), ("test", "val")):
        root = os.path.join(EXT, "roboflow_kucab", sub)
        d = json.load(open(os.path.join(root, "_annotations.coco.json")))
        by_img = {}
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            c = 0 if a["category_id"] == 0 else 1
            by_img.setdefault(a["image_id"], []).append((c, x, y, x + w, y + h))
        for im in d["images"]:
            boxes = by_img.get(im["id"], [])
            balls = [b for b in boxes if b[0] == 0]
            img = cv2.imread(os.path.join(root, im["file_name"]))
            if img is None:
                continue
            reps = 2 if split == "train" else 1
            for r in range(reps):
                name = "rf_%s_%d_%d" % (sub, im["id"], r)
                if balls:
                    b = balls[0]
                    s = scale_for(b[3] - b[1], rng) * 1.0
                    s = float(np.clip(s, 0.6, 2.0))
                    xy = ((b[1] + b[3]) / 2, (b[2] + b[4]) / 2)
                else:
                    s, xy = rng.uniform(0.8, 1.5), None
                write_sample(img, boxes, s, rng, os.path.join(out, "images", split, name + ".jpg"),
                             os.path.join(out, "labels", split, name + ".txt"), xy)
                n[split] += 1
    print("roboflow", n)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CACHE, "ds"))
    ap.add_argument("--sn-step", type=int, default=3, help="use every k-th SN-GSR frame")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for s in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", s), exist_ok=True)
    roboflow(a.out, rng)
    sn_gsr(a.out, rng, a.sn_step)
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n  1: person\n"
                % a.out)
    print("done", a.out)


if __name__ == "__main__":
    main()
