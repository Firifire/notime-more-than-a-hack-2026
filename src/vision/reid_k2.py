import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.common import CACHE, FrameReader, N_FRAMES

WEIGHTS = os.path.join(CACHE, "track", "k2_reid_h3.pt")
CH, CW = 64, 32


class Net(nn.Module):
    def __init__(self, dim=128, ncls=0):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(blk(3, 32, 1), blk(32, 64, 2), blk(64, 128, 2), blk(128, 256, 2))
        self.head = nn.Sequential(nn.Linear(256 * 2, dim), nn.BatchNorm1d(dim))
        self.cls = nn.Linear(dim, ncls) if ncls else None

    def forward(self, x):
        h = self.f(x)
        h = torch.cat([F.adaptive_avg_pool2d(h, 1).flatten(1), F.adaptive_max_pool2d(h, 1).flatten(1)], 1)
        e = self.head(h)
        return e, (self.cls(e) if self.cls is not None else None)


MEAN = torch.tensor([0.4, 0.45, 0.4]).view(1, 3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)


def prep(u8, dev):
    x = torch.from_numpy(u8).to(dev).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(dev)) / STD.to(dev)


def load_net(dev, path=WEIGHTS):
    net = Net().to(dev)
    sd = torch.load(path, map_location=dev, weights_only=True)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


@torch.no_grad()
def embed(net, u8, dev, bs=2048):
    out = []
    for i in range(0, len(u8), bs):
        e, _ = net(prep(u8[i:i + bs], dev))
        out.append(F.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 128), np.float32)


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


def load_cands(cdir):
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(cdir, "pc_*.npz")))]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("frame", "box", "conf", "feat")}
    o = np.argsort(cat["frame"], kind="stable")
    return {k: v[o] for k, v in cat.items()}, np.unique(np.concatenate(
        [np.arange(int(p["start"]), int(p["stop"])) for p in parts]))


def segments(frames):
    br = np.flatnonzero(np.diff(frames) > 1)
    st = np.r_[frames[0], frames[br + 1]]
    en = np.r_[frames[br], frames[-1]] + 1
    return list(zip(st, en))


def match_embeddings(cdir, out, every=2, dev="cuda", weights=WEIGHTS):
    cands, processed = load_cands(cdir)
    fr = cands["frame"]
    box = cands["box"]
    bounds = np.searchsorted(fr, np.arange(N_FRAMES + 1))
    net = load_net(dev, weights)
    IDX, EMB = [], []
    buf_c, buf_i = [], []
    t0 = time.time()

    def flush():
        if buf_c:
            EMB.append(embed(net, np.array(buf_c, np.uint8), dev).astype(np.float16))
            IDX.append(np.array(buf_i, np.int64))
            buf_c.clear()
            buf_i.clear()
    for s, e in segments(processed):
        for t, yuv in FrameReader(s, e):
            if t % every:
                continue
            i0, i1 = bounds[t], bounds[t + 1]
            if i1 == i0:
                continue
            img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            for i in range(i0, i1):
                buf_c.append(crop_one(img, box[i]))
                buf_i.append(i)
            if len(buf_c) >= 8192:
                flush()
    flush()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, idx=np.concatenate(IDX), emb=np.concatenate(EMB))
    print("[reid_k2] wrote %s: %d embeddings (%.0f s)" % (out, sum(len(x) for x in IDX), time.time() - t0), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=os.path.join(CACHE, "players_cands"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    match_embeddings(a.cands, a.out)


if __name__ == "__main__":
    main()
