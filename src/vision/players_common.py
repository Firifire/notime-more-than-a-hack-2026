import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from vision.common import CACHE, ROOT

EXT = os.path.join(ROOT, "data_external")
PLAYER_WEIGHTS = os.path.join(CACHE, "runs", "players_s", "weights", "best.pt")
NMS_IOU = 0.6


def load_model(weights, device="cuda"):
    from ultralytics import YOLO
    m = YOLO(weights).model.to(device).eval().half()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def infer(model, x, cls=0, conf=0.05, iou=NMS_IOU, topk=100):
    import torchvision
    pred = model(x.half())
    pred = pred[0] if isinstance(pred, (list, tuple)) else pred
    out = []
    boxes = pred[:, :4, :]
    scores = pred[:, 4 + cls, :]
    for b in range(pred.shape[0]):
        s = scores[b]
        keep = s > conf
        if not keep.any():
            out.append(np.zeros((0, 5), np.float32))
            continue
        bx = boxes[b][:, keep].T.float()
        sc = s[keep].float()
        xyxy = torch.cat([bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2], 1)
        k = torchvision.ops.nms(xyxy, sc, iou)[:topk]
        out.append(torch.cat([xyxy[k], sc[k, None]], 1).cpu().numpy().astype(np.float32))
    return out


def pad32(x, value=114 / 255.0):
    h, w = x.shape[-2:]
    ph, pw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    return F.pad(x, (0, pw - w, 0, ph - h), value=value)


YB, UB, VB = 3, 4, 4
NFEAT = YB * UB * VB


def yuv_planes_from_i420(buf, h, w):
    y = buf[:h]
    uv = buf[h:].reshape(2, h // 2, w // 2)
    return y, uv[0], uv[1]


def kit_features(Y, U, V, boxes):
    out = np.zeros((len(boxes), NFEAT), np.float32)
    h, w = Y.shape
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        bw, bh = x1 - x0, y1 - y0
        cx0, cx1 = int(x0 + 0.28 * bw), int(np.ceil(x0 + 0.72 * bw))
        ry0, ry1 = int(y0 + 0.15 * bh), int(np.ceil(y0 + 0.62 * bh))
        cx0, cx1 = max(cx0, 0), min(cx1, w)
        ry0, ry1 = max(ry0, 0), min(ry1, h)
        if cx1 - cx0 < 2 or ry1 - ry0 < 2:
            continue
        yy = Y[ry0:ry1, cx0:cx1]
        uu = U[ry0 // 2:(ry1 + 1) // 2, cx0 // 2:(cx1 + 1) // 2]
        vv = V[ry0 // 2:(ry1 + 1) // 2, cx0 // 2:(cx1 + 1) // 2]
        uu = np.repeat(np.repeat(uu, 2, 0), 2, 1)[:yy.shape[0], :yy.shape[1]]
        vv = np.repeat(np.repeat(vv, 2, 0), 2, 1)[:yy.shape[0], :yy.shape[1]]
        yi = np.clip((yy.astype(np.int32) - 16) * YB // 220, 0, YB - 1)
        ui = np.clip((uu.astype(np.int32) - 64) * UB // 128, 0, UB - 1)
        vi = np.clip((vv.astype(np.int32) - 64) * VB // 128, 0, VB - 1)
        idx = (yi * UB + ui) * VB + vi
        hist = np.bincount(idx.ravel(), minlength=NFEAT).astype(np.float32)
        out[i] = hist / max(hist.sum(), 1)
    return out


def bgr_to_planes(img):
    import cv2
    h, w = img.shape[:2]
    h2, w2 = h // 2 * 2, w // 2 * 2
    buf = cv2.cvtColor(img[:h2, :w2], cv2.COLOR_BGR2YUV_I420)
    return yuv_planes_from_i420(buf, h2, w2)


def hist_dist(a, b):
    return np.sqrt(np.clip(1 - (np.sqrt(a * b)).sum(-1), 0, 1))


def load_sn_seq(sd):
    d = json.load(open(os.path.join(sd, "Labels-GameState.json")))
    fid = {im["image_id"]: k for k, im in enumerate(d["images"])}
    per = {}
    for a in d["annotations"]:
        if a["category_id"] not in (1, 2, 3) or not a.get("bbox_image"):
            continue
        b, p = a["bbox_image"], a.get("bbox_pitch") or {}
        at = a.get("attributes") or {}
        role = {1: 0, 2: 1, 3: 2}[a["category_id"]]
        team = {"left": 0, "right": 1}.get(at.get("team"), -1)
        px = p.get("x_bottom_middle")
        py = p.get("y_bottom_middle")
        per.setdefault(fid[a["image_id"]], []).append(
            (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"],
             np.nan if px is None else px + 52.5, np.nan if py is None else py + 34.0,
             a["track_id"], role, team))
    out = {}
    for k in range(len(d["images"])):
        r = np.array(per.get(k, []), np.float64).reshape(-1, 9)
        out[k] = dict(boxes=r[:, :4], pitch=r[:, 4:6], tid=r[:, 6].astype(int),
                      role=r[:, 7].astype(int), team=r[:, 8].astype(int))
    return out, [im["file_name"] for im in d["images"]], d["images"]


def fit_ground_h(gt):
    import cv2
    ok = np.isfinite(gt["pitch"][:, 0])
    if ok.sum() < 5:
        return None
    b = gt["boxes"][ok]
    src = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]].astype(np.float64)
    dst = gt["pitch"][ok].astype(np.float64)
    Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.75)
    return Hm


def h_apply(Hm, u, v):
    d = Hm[2, 0] * u + Hm[2, 1] * v + Hm[2, 2]
    return (Hm[0, 0] * u + Hm[0, 1] * v + Hm[0, 2]) / d, (Hm[1, 0] * u + Hm[1, 1] * v + Hm[1, 2]) / d


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def bounded_map(fn, items, workers=4, inflight=12):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        dq = deque()
        it = iter(items)
        for x in it:
            dq.append(ex.submit(fn, x))
            if len(dq) >= inflight:
                yield dq.popleft().result()
        while dq:
            yield dq.popleft().result()
