import argparse
import glob
import json
import os
import sys
import threading
import queue

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.common import CACHE  # noqa: E402
from vision.players_common import (EXT, load_model, infer, pad32, load_sn_seq, fit_ground_h,  # noqa: E402
                                   h_apply, iou_matrix)

ODIR = os.path.join(CACHE, "players_eval")
TACT_MED_H = 48.0
W = os.path.join(CACHE, "weights")
CONFIGS = {
    "ours_s15": (os.path.join(CACHE, "runs", "players_s", "weights", "best.pt"), 0, 1.5),
    "ours_s10": (os.path.join(CACHE, "runs", "players_s", "weights", "best.pt"), 0, 1.0),
    "ours_m15": (os.path.join(CACHE, "runs", "players_m", "weights", "best.pt"), 0, 1.5),
    "ballmodel15": (os.path.join(CACHE, "runs", "ball", "weights", "best.pt"), 1, 1.5),
    "coco_s15": (os.path.join(W, "yolo11s.pt"), 0, 1.5),
    "coco_s10": (os.path.join(W, "yolo11s.pt"), 0, 1.0),
    "coco_x15": (os.path.join(W, "yolo11x.pt"), 0, 1.5),
    "coco_x20": (os.path.join(W, "yolo11x.pt"), 0, 2.0),
}
HBINS = [0, 40, 55, 70, 1e9]


def eval_items():
    items = []
    for sd in sorted(glob.glob(os.path.join(EXT, "sn_gsr_valid", "SNGS-*"))):
        per, files, ims = load_sn_seq(sd)
        for k in range(0, len(files), 25):
            g = per[k]
            if len(g["boxes"]) < 3 or not ims[k].get("has_labeled_person", True):
                continue
            Hm = fit_ground_h(g)
            if Hm is None:
                continue
            hmed = np.median(g["boxes"][:, 3] - g["boxes"][:, 1])
            items.append(dict(set="sn_valid", path=os.path.join(sd, "img1", files[k]),
                              s=min(1.0, TACT_MED_H / hmed), gt=g["boxes"], H=Hm,
                              gxy=g["pitch"], key="%s_%d" % (os.path.basename(sd), k)))
    root = os.path.join(EXT, "roboflow_kucab", "test")
    d = json.load(open(os.path.join(root, "_annotations.coco.json")))
    by = {}
    for a in d["annotations"]:
        if a["category_id"] != 0:
            x, y, w, h = a["bbox"]
            by.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
    for im in d["images"]:
        items.append(dict(set="rf_test", path=os.path.join(root, im["file_name"]), s=1.0,
                          gt=np.array(by.get(im["id"], []), np.float64).reshape(-1, 4), H=None,
                          gxy=None, key="rf_%d" % im["id"]))
    return items


def run_config(name, items, device="cuda"):
    out = os.path.join(ODIR, "det_%s.npz" % name)
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True)
        return list(z["dets"])
    weights, cls, zoom = CONFIGS[name]
    model = load_model(weights, device)
    q = queue.Queue(maxsize=16)

    def prep(it):
        img = cv2.imread(it["path"])
        if it["s"] != 1.0:
            img = cv2.resize(img, None, fx=it["s"], fy=it["s"], interpolation=cv2.INTER_AREA)
        if zoom != 1.0:
            img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
        return img[:, :, ::-1].copy()

    def producer():
        from vision.players_common import bounded_map
        for img in bounded_map(prep, items):
            q.put(img)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    dets = []
    with torch.inference_mode():
        while True:
            img = q.get()
            if img is None:
                break
            x = torch.from_numpy(img).to(device).permute(2, 0, 1)[None].float() / 255.0
            d = infer(model, pad32(x), cls=cls, conf=0.01, topk=150)[0]
            dets.append(d)
    dets = [np.c_[d[:, :4] / (zoom * it["s"]), d[:, 4]].astype(np.float32)
            for d, it in zip(dets, items)]
    os.makedirs(ODIR, exist_ok=True)
    np.savez_compressed(out, dets=np.array(dets, dtype=object))
    return dets


def pitch_filter(d, it):
    if it["H"] is None or len(d) == 0:
        return d
    x, y = h_apply(it["H"], (d[:, 0] + d[:, 2]) / 2, d[:, 3])
    ok = (x > -3) & (x < 108) & (y > -3) & (y < 71)
    return d[ok]


def match(d, gt, thr=0.5):
    order = np.argsort(-d[:, 4])
    d = d[order]
    iou = iou_matrix(d[:, :4], gt)
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(d), bool)
    mi = np.full(len(d), -1)
    for i in range(len(d)):
        if len(gt) == 0:
            break
        cand = np.where(~used, iou[i], -1)
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            used[j] = True
            tp[i] = True
            mi[i] = j
    return d, tp, mi


def ap_from(conf, tp, ngt):
    o = np.argsort(-conf)
    tp = tp[o].astype(float)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / max(ngt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.r_[0, rec, rec[-1] if len(rec) else 0]
    mpre = np.r_[1, prec, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(np.argmax(f1)) if len(f1) else 0
    return ap, (float(conf[o][k]) if len(f1) else 1.0, float(prec[k]) if len(f1) else 0,
                float(rec[k]) if len(f1) else 0, float(f1[k]) if len(f1) else 0)


def evaluate(dets, items, setname):
    C, TP, NGT = [], [], 0
    per = []
    for d, it in zip(dets, items):
        if it["set"] != setname:
            continue
        d = pitch_filter(d, it)
        d, tp, mi = match(d, it["gt"])
        C.append(d[:, 4])
        TP.append(tp)
        NGT += len(it["gt"])
        per.append((d, tp, mi, it))
    C, TP = np.concatenate(C), np.concatenate(TP)
    ap, (cbest, p, r, f1) = ap_from(C, TP, NGT)
    res = dict(n_img=len(per), n_gt=NGT, AP50=round(ap, 4), best_f1=round(f1, 4),
               conf_at_best_f1=round(cbest, 3), P_at_best=round(p, 4), R_at_best=round(r, 4))
    for tag, thr in (("best", cbest), ("c010", 0.10), ("c025", 0.25)):
        hits = []
        for d, tp, mi, it in per:
            g = it["gt"]
            hit = np.zeros(len(g), bool)
            sel = tp & (d[:, 4] >= thr)
            hit[mi[sel]] = True
            hgt = (g[:, 3] - g[:, 1]) * it["s"]
            iou = iou_matrix(g, g)
            np.fill_diagonal(iou, 0)
            crowd = iou.max(1) >= 0.1 if len(g) > 1 else np.zeros(len(g), bool)
            near = np.zeros(len(g), bool)
            if it["gxy"] is not None and len(g) > 1:
                dd = np.hypot(*(it["gxy"][:, None, :] - it["gxy"][None, :, :]).transpose(2, 0, 1))
                np.fill_diagonal(dd, 1e9)
                near = np.nanmin(dd, 1) < 1.5
            err = np.full(len(g), np.nan)
            if it["H"] is not None and sel.any():
                dm = d[sel]
                x, y = h_apply(it["H"], (dm[:, 0] + dm[:, 2]) / 2, dm[:, 3])
                err[mi[sel]] = np.hypot(x - it["gxy"][mi[sel], 0], y - it["gxy"][mi[sel], 1])
            hits.append(np.c_[hit, hgt, crowd, near, err])
        hh = np.concatenate(hits)
        rr = {"recall": round(float(hh[:, 0].mean()), 4),
              "n_det_per_img": None}
        for a, b in zip(HBINS[:-1], HBINS[1:]):
            m = (hh[:, 1] >= a) & (hh[:, 1] < b)
            rr["recall_h%d_%s" % (a, "inf" if b > 1e8 else int(b))] = (
                round(float(hh[m, 0].mean()), 4) if m.any() else None, int(m.sum()))
        m = hh[:, 2] > 0
        rr["recall_crowd_iou"] = (round(float(hh[m, 0].mean()), 4), int(m.sum()))
        rr["recall_isolated"] = (round(float(hh[~m, 0].mean()), 4), int((~m).sum()))
        m = hh[:, 3] > 0
        if m.any():
            rr["recall_near1.5m"] = (round(float(hh[m, 0].mean()), 4), int(m.sum()))
        e = hh[:, 4][np.isfinite(hh[:, 4])]
        if len(e):
            rr["foot_err_m_p50_p90"] = [round(float(np.percentile(e, 50)), 3),
                                        round(float(np.percentile(e, 90)), 3)]
        rr["n_det_per_img"] = round(float(np.mean([(d[:, 4] >= thr).sum() for d, *_ in per])), 2)
        res["at_" + tag + ("(%.2f)" % thr)] = rr
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="ours_s15,ours_s10,ballmodel15,coco_s15,coco_s10,coco_x15")
    a = ap.parse_args()
    items = eval_items()
    print("items", len(items), flush=True)
    allres = {}
    jpath = os.path.join(ODIR, "det_eval.json")
    if os.path.exists(jpath):
        allres = json.load(open(jpath))
    for name in a.configs.split(","):
        if not os.path.exists(CONFIGS[name][0]) and "coco" not in name:
            print("missing weights", name)
            continue
        dets = run_config(name, items)
        allres[name] = {s: evaluate(dets, items, s) for s in ("sn_valid", "rf_test")}
        print(name, json.dumps(allres[name], indent=1), flush=True)
        os.makedirs(ODIR, exist_ok=True)
        json.dump(allres, open(jpath, "w"), indent=1)
    print("\n%-12s %-9s %6s %6s %6s %6s %6s %6s %6s" % ("config", "set", "AP50", "F1", "R@F1",
                                                      "R<40", "Rcrowd", "R.1", "ft_err"))
    for name, r in allres.items():
        for s, v in r.items():
            b = [k for k in v if k.startswith("at_best")][0]
            c = [k for k in v if k.startswith("at_c010")][0]
            print("%-12s %-9s %6.3f %6.3f %6.3f %6s %6.3f %6.3f %s" % (
                name, s, v["AP50"], v["best_f1"], v["R_at_best"], v[b]["recall_h0_40"][0],
                v[b]["recall_crowd_iou"][0], v[c]["recall"], v[b].get("foot_err_m_p50_p90")))


if __name__ == "__main__":
    main()
