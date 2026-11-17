import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CROPS = "/mnt/x/Data/AI/more-than-a-hack/trackK2/reid_crops"
PACKS = "/mnt/x/Data/AI/more-than-a-hack/trackK2/gsr_packs"
OUTD = "/mnt/x/Data/AI/more-than-a-hack/trackK2/reid"


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


def augment(x):
    B = x.shape[0]
    dev = x.device
    out = torch.empty_like(x)
    sc = torch.empty(B).uniform_(0.55, 1.0)
    for s in torch.unique((sc * 10).round() / 10):
        m = ((sc * 10).round() / 10) == s
        xi = x[m.to(dev)]
        h, w = max(8, int(64 * s)), max(4, int(32 * s))
        xi = F.interpolate(F.interpolate(xi, size=(h, w), mode="area"), size=(64, 32), mode="bilinear",
                           align_corners=False)
        out[m.to(dev)] = xi
    k = torch.tensor([1, 2, 1], dtype=x.dtype, device=dev)
    k = (k[:, None] * k[None]) / 16
    blur = F.conv2d(F.pad(out, (1, 1, 1, 1), mode="replicate"), k.expand(3, 1, 3, 3), groups=3)
    bm = (torch.rand(B, 1, 1, 1, device=dev) < 0.3).float()
    out = bm * blur + (1 - bm) * out
    a = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.3, 0.3)
    out = out * a + b + torch.randn_like(out) * 0.05
    fl = torch.rand(B, device=dev) < 0.5
    out[fl] = out[fl].flip(3)
    return out


def load_split(split, need_gt=True):
    data = []
    for f in sorted(glob.glob(os.path.join(CROPS, split + "_SNGS-*.npz"))):
        z = np.load(f)
        d = {k: z[k] for k in ("frame", "det", "crop", "gt", "team", "role")}
        if need_gt:
            m = d["gt"] >= 0
            d = {k: v[m] for k, v in d.items()}
        d["seq"] = os.path.basename(f).replace(".npz", "")
        data.append(d)
    return data


def batch_hard(e, y, margin=0.3):
    e = F.normalize(e, dim=1)
    D = torch.cdist(e, e)
    same = y[:, None] == y[None]
    pos = (D * same.float()).max(1).values
    neg = (D + same.float() * 1e3).min(1).values
    return F.relu(pos - neg + margin).mean(), float((pos < neg).float().mean())


def cmd_train(a):
    dev = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = load_split("train")
    if a.hold:
        data = [d for i, d in enumerate(data) if i % a.hold != 0]
    off, labels = 0, []
    for d in data:
        u, inv = np.unique(d["gt"], return_inverse=True)
        d["lab"] = inv + off
        d["ids"] = [np.flatnonzero(inv == i) for i in range(len(u))]
        off += len(u)
    print("train clips %d, identities %d, crops %d" % (len(data), off, sum(len(d["gt"]) for d in data)), flush=True)
    net = Net(ncls=off).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.iters, pct_start=0.1)
    t0 = time.time()
    for it in range(a.iters):
        idx_u8, labs = [], []
        for d in [data[i] for i in rng.choice(len(data), 2, replace=False)]:
            ok = [i for i, ix in enumerate(d["ids"]) if len(ix) >= 4]
            pick = rng.choice(ok, min(16, len(ok)), replace=False)
            for p in pick:
                sel = rng.choice(d["ids"][p], 4, replace=False)
                idx_u8.append(d["crop"][sel])
                labs.append(d["lab"][sel])
        x = augment(prep(np.concatenate(idx_u8), dev))
        y = torch.from_numpy(np.concatenate(labs)).to(dev).long()
        e, logit = net(x)
        lt, acc = batch_hard(e, y)
        lc = F.cross_entropy(logit, y, label_smoothing=0.1)
        loss = lt + 0.5 * lc
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == a.iters - 1:
            print("it %d  triplet %.3f  ce %.3f  pos<neg %.3f  %.0fs" % (it, lt.item(), lc.item(), acc, time.time() - t0),
                  flush=True)
    os.makedirs(OUTD, exist_ok=True)
    torch.save({k: v for k, v in net.state_dict().items() if not k.startswith("cls.")}, os.path.join(OUTD, "model%s.pt" % a.tag))


def load_net(dev, tag=""):
    net = Net().to(dev)
    sd = torch.load(os.path.join(OUTD, "model%s.pt" % tag), map_location=dev, weights_only=True)
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


def hell(a, b):
    return np.sqrt(np.clip(1 - (np.sqrt(np.clip(a[:, None], 0, None) * np.clip(b[None], 0, None))).sum(-1), 0, 1))


def cmd_eval(a):
    dev = "cuda"
    net = load_net(dev, a.tag)
    data = load_split("valid")
    res = {}
    for gap_s in (1, 3, 10):
        gap = 2 * int(round(25 * gap_s / 2))
        st = {m: [0, 0] for m in ("emb1", "kit1", "emb5", "kit5", "both5")}
        chance = []
        for d in data:
            pk = np.load(os.path.join(PACKS, d["seq"] + ".npz"))
            det_f, det_feat = pk["det_f"], pk["det_feat"]
            b = np.searchsorted(det_f, np.arange(int(pk["n"]) + 1))
            kit = det_feat[b[d["frame"]] + d["det"]].astype(np.float64)
            kit = kit / np.maximum(kit.sum(1, keepdims=True), 1e-9)
            emb = embed(net, d["crop"], dev)
            ok = (d["role"] < 2) & (d["team"] >= 0)
            fr = d["frame"]
            key = {}
            for i in np.flatnonzero(ok):
                key[(int(fr[i]), int(d["gt"][i]))] = i
            frames = np.unique(fr)
            for t in frames[::4]:
                t2 = t + gap
                qs = [i for i in np.flatnonzero((fr == t) & ok)]
                gal = np.flatnonzero((fr == t2) & ok)
                for q in qs:
                    g, c = d["gt"][q], d["team"][q]
                    G = gal[d["team"][gal] == c]
                    if len(G) < 2 or g not in d["gt"][G]:
                        continue
                    chance.append(1.0 / len(G))
                    tgt = int(np.flatnonzero(d["gt"][G] == g)[0])
                    de = 1 - emb[G] @ emb[q]
                    dk = hell(kit[q][None], kit[G])[0]
                    st["emb1"][0] += int(np.argmin(de) == tgt)
                    st["kit1"][0] += int(np.argmin(dk) == tgt)
                    st["emb1"][1] += 1
                    st["kit1"][1] += 1
                    qi = [key.get((t - j, int(g))) for j in range(0, 10, 2)]
                    qi = [x for x in qi if x is not None]
                    gm_e, gm_k, okg = [], [], True
                    for gi in G:
                        ii = [key.get((t2 + j, int(d["gt"][gi]))) for j in range(0, 10, 2)]
                        ii = [x for x in ii if x is not None]
                        gm_e.append(emb[ii].mean(0))
                        gm_k.append(kit[ii].mean(0))
                    qe, qk = emb[qi].mean(0), kit[qi].mean(0)
                    gm_e, gm_k = np.array(gm_e), np.array(gm_k)
                    gm_e /= np.linalg.norm(gm_e, axis=1, keepdims=True)
                    de5 = 1 - gm_e @ (qe / np.linalg.norm(qe))
                    dk5 = hell(qk[None], gm_k)[0]
                    st["emb5"][0] += int(np.argmin(de5) == tgt)
                    st["kit5"][0] += int(np.argmin(dk5) == tgt)
                    st["both5"][0] += int(np.argmin(de5 / max(de5.std(), 1e-6) + dk5 / max(dk5.std(), 1e-6)) == tgt)
                    for m in ("emb5", "kit5", "both5"):
                        st[m][1] += 1
        res["gap_%ds" % gap_s] = {m: round(v[0] / max(v[1], 1), 4) for m, v in st.items()}
        res["gap_%ds" % gap_s]["chance"] = round(float(np.mean(chance)), 4)
        res["gap_%ds" % gap_s]["n"] = st["emb1"][1]
        print(gap_s, res["gap_%ds" % gap_s], flush=True)
    os.makedirs(os.path.join(ROOT, "runs", "trackK2"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "runs", "trackK2", "reid_eval%s.json" % a.tag), "w"), indent=1)


def cmd_embed(a):
    dev = "cuda"
    net = load_net(dev, a.tag)
    for f in sorted(glob.glob(os.path.join(CROPS, "*_SNGS-*.npz"))):
        z = np.load(f)
        e = embed(net, z["crop"], dev)
        np.savez(os.path.join(OUTD, "emb%s_" % a.tag + os.path.basename(f)), frame=z["frame"], det=z["det"],
                 emb=e.astype(np.float16))
    print("embedded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "embed"])
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--hold", type=int, default=0, help="exclude train clips with index %% hold == 0")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "embed": cmd_embed}[a.cmd](a)


if __name__ == "__main__":
    main()
