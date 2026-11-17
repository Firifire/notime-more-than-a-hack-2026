import argparse
import glob
import json
import os
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.common import ROOT, CACHE  # noqa: E402

EXT = os.path.join(ROOT, "data_external")
TILE = 960
SN_VAL_SEQS = 4


def write_tile(img, boxes, s, rng, out_img, out_lbl):
    if abs(s - 1.0) > 1e-3:
        img = cv2.resize(img, None, fx=s, fy=s,
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    hh, ww = img.shape[:2]
    tw, th = min(TILE, ww), min(TILE, hh)
    x0 = rng.randint(0, ww - tw)
    y0 = rng.randint(0, hh - th)
    crop = img[y0:y0 + th, x0:x0 + tw]
    lines = []
    for a, b, c, d in boxes:
        a, b, c, d = a * s - x0, b * s - y0, c * s - x0, d * s - y0
        ia, ib, ic, idd = max(a, 0), max(b, 0), min(c, tw), min(d, th)
        if ic - ia < 2 or idd - ib < 2:
            continue
        if (ic - ia) * (idd - ib) < 0.4 * (c - a) * (d - b):
            continue
        lines.append("0 %.6f %.6f %.6f %.6f" % ((ia + ic) / 2 / tw, (ib + idd) / 2 / th,
                                                (ic - ia) / tw, (idd - ib) / th))
    cv2.imwrite(out_img, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out_lbl, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def sn_boxes(sd):
    d = json.load(open(os.path.join(sd, "Labels-GameState.json")))
    by_img = {}
    for a in d["annotations"]:
        if a["category_id"] not in (1, 2, 3) or not a.get("bbox_image"):
            continue
        b = a["bbox_image"]
        by_img.setdefault(a["image_id"], []).append((b["x"], b["y"], b["x"] + b["w"],
                                                     b["y"] + b["h"]))
    return d["images"], by_img


def do_sn_seq(args):
    sd, split, step, out, seed = args
    rng = random.Random(seed)
    images, by_img = sn_boxes(sd)
    n = 0
    for im in images[::step]:
        if not im.get("has_labeled_person", True):
            continue
        boxes = by_img.get(im["image_id"], [])
        img = cv2.imread(os.path.join(sd, "img1", im["file_name"]))
        if img is None:
            continue
        hmed = np.median([b[3] - b[1] for b in boxes]) if boxes else 100.0
        s = float(np.clip(rng.uniform(45, 100) / max(hmed, 1.0), 0.5, 1.0))
        name = "sn_%s_%s" % (os.path.basename(sd), im["file_name"][:-4])
        write_tile(img, boxes, s, rng, os.path.join(out, "images", split, name + ".jpg"),
                   os.path.join(out, "labels", split, name + ".txt"))
        n += 1
    return sd, n


def do_rf(args):
    sub, out, seed, reps = args
    rng = random.Random(seed)
    root = os.path.join(EXT, "roboflow_kucab", sub)
    d = json.load(open(os.path.join(root, "_annotations.coco.json")))
    by_img = {}
    for a in d["annotations"]:
        if a["category_id"] == 0:
            continue
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
    n = 0
    for im in d["images"]:
        img = cv2.imread(os.path.join(root, im["file_name"]))
        if img is None:
            continue
        for r in range(reps):
            s = rng.uniform(1.0, 2.0)
            name = "rf_%s_%d_%d" % (sub, im["id"], r)
            write_tile(img, by_img.get(im["id"], []), s, rng,
                       os.path.join(out, "images", "train", name + ".jpg"),
                       os.path.join(out, "labels", "train", name + ".txt"))
            n += 1
    return sub, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CACHE, "ds_players"))
    ap.add_argument("--sn-step", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    for s in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", s), exist_ok=True)
    seqs = sorted(glob.glob(os.path.join(EXT, "sn_gsr", "SNGS-*")))
    jobs = [(sd, "val" if k >= len(seqs) - SN_VAL_SEQS else "train",
             a.sn_step if k < len(seqs) - SN_VAL_SEQS else a.sn_step * 3, a.out, 1000 + k)
            for k, sd in enumerate(seqs)]
    with Pool(a.workers) as p:
        for sd, n in p.imap_unordered(do_sn_seq, jobs):
            print(sd, n, flush=True)
        for sub, n in p.imap_unordered(do_rf, [("train", a.out, 1, 3), ("valid", a.out, 2, 3)]):
            print("roboflow", sub, n, flush=True)
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: person\n" % a.out)
    print("done", a.out)


if __name__ == "__main__":
    main()
