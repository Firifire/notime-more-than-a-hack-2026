import argparse
import glob
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feats as Fe  # noqa: E402
import score as S  # noqa: E402

BASE = "/mnt/x/Data/AI/more-than-a-hack/trackD4"
TRK = os.path.join(BASE, "tracklets")
CAN = os.path.join(BASE, "canvas")
RUNS = os.path.join(BASE, "runs")
K, NEW, JUNK, V = Fe.K, Fe.NEW, Fe.JUNK, Fe.V
L = 256
FA = 10
CLASSIC = (10, 11, 12)
EMB = {"on": False}


def prep_one(path):
    name = os.path.basename(path)[:-4]
    out = os.path.join(CAN, name + ".pkl")
    if os.path.exists(out):
        return out
    res = pickle.load(open(path, "rb"))
    rows, X = Fe.clip_canvas(res)
    X["rows"] = [dict(id=r["id"], obs=r["obs"], t0=r["t0"], t1=r["t1"]) for r in rows]
    X["n"] = res["n"]
    os.makedirs(CAN, exist_ok=True)
    pickle.dump(X, open(out + ".tmp", "wb"))
    os.replace(out + ".tmp", out)
    return out


def cmd_prep(a):
    from concurrent.futures import ProcessPoolExecutor
    paths = sorted(glob.glob(os.path.join(TRK, "*.pkl")))
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(prep_one, paths):
            pass
    print("prepared", len(paths))


def load_split(split, suffix=""):
    out = {}
    for c in CAN.split(","):
        c = c if os.path.isabs(c) else os.path.join(BASE, c)
        for p in sorted(glob.glob(os.path.join(c, split + "_*%s.pkl" % suffix))):
            out[os.path.basename(c) + "/" + os.path.basename(p)[:-4]] = pickle.load(open(p, "rb"))
    return out


def decode_fb(rows, cand, tok, fb):
    pred = {i: int(cand[i, t]) for i, t in enumerate(tok) if t < K and cand[i, t] >= 0}
    claimed = set(pred.values())
    for i in range(len(rows)):
        j = fb.get(i)
        if i not in pred and j is not None and j not in claimed:
            pred[i] = j
            claimed.add(j)
    root = list(range(len(rows)))
    for i in range(len(rows)):
        if i in pred:
            root[i] = root[pred[i]]
    out = {}
    for i in range(len(rows)):
        out.setdefault(root[i], []).extend(rows[i]["obs"])
    return {k: sorted(v) for k, v in out.items()}


def static_pairs(X, idx):
    t0, t1 = X["t0"][idx].astype(np.float32), X["t1"][idx].astype(np.float32)
    pm = X["pm"][idx]
    kit, kv = X["kit"][idx], X["kv"][idx]
    W = len(idx)
    ov = np.minimum(t1[:, None], t1[None]) - np.maximum(t0[:, None], t0[None])
    gap = t0[:, None] - t1[None]
    dpos = np.hypot(*(pm[:, None] - pm[None]).transpose(2, 0, 1))
    kd = np.sqrt(np.clip(1 - np.sqrt(kit[:, None] * kit[None]).sum(-1), 0, 1))
    kvv = kv[:, None] * kv[None]
    cand = X["cand"][idx]
    loc = -np.ones(len(X["t0"]), np.int64)
    loc[idx] = np.arange(W)
    cl = np.where(cand >= 0, loc[np.maximum(cand, 0)], -1)
    isc = np.zeros((W, W), np.float32)
    r, c = np.nonzero(cl >= 0)
    isc[r, cl[r, c]] = 1
    same_o = (X["ours_id"][idx][:, None] == X["ours_id"][idx][None]).astype(np.float32)
    f = np.stack([np.clip(ov, -500, 500) / 250, (ov >= 0).astype(np.float32), np.clip(gap, -750, 750) / 250,
                  dpos / 30, kd * kvv, kvv, isc, isc.T, same_o, (t0[:, None] - t0[None]) / 250], -1)
    return f.astype(np.float32), cl


def window(X, idx, flip=(False, False), classic=True, team=False):
    W = len(idx)
    tf = X["tf"][idx].copy()
    tw = X["t0"][idx].min()
    tf[:, 0] = (X["t0"][idx] - tw) / 750.0
    tf[:, 1] = (X["t1"][idx] - tw) / 750.0
    pf = X["pf"][idx].copy()
    if not classic:
        pf[..., list(CLASSIC)] = 0
    if team:
        tf = np.concatenate([tf, X["team"][idx]], 1)
        pf = np.concatenate([pf, X["pteam"][idx]], 2)
    if EMB["on"]:
        pe = X["pemb"][idx] if "pemb" in X else np.zeros((len(idx), K, 2), np.float32)
        pf = np.concatenate([pf, pe], 2)
    if flip[0]:
        tf[:, [7, 9]] = 1 - tf[:, [7, 9]]
        tf[:, [11, 13]] *= -1
    if flip[1]:
        tf[:, [8, 10]] = 1 - tf[:, [8, 10]]
        tf[:, [12, 14]] *= -1
    fa, cl = static_pairs(X, idx)
    if not classic:
        fa[..., 8] = 0
    y = X["y"][idx].copy() if "y" in X else np.full(W, NEW)
    bad = (y < K) & (cl[np.arange(W), np.minimum(y, K - 1)] < 0)
    y[bad] = NEW
    valid = np.concatenate([cl >= 0, np.ones((W, 2), bool)], 1)
    return dict(tf=tf, pf=pf, fa=fa, cl=cl, y=y, valid=valid)


def collate(ws, dev):
    B = len(ws)
    W = max(len(w["y"]) for w in ws)
    ft_, fp_ = ws[0]["tf"].shape[1], ws[0]["pf"].shape[2]
    out = dict(tf=np.zeros((B, W, ft_), np.float32), pf=np.zeros((B, W, K, fp_), np.float32),
               fa=np.zeros((B, W, W, FA), np.float32), cl=-np.ones((B, W, K), np.int64),
               y=np.full((B, W), -100, np.int64), valid=np.zeros((B, W, V), bool), mask=np.zeros((B, W), bool))
    for b, w in enumerate(ws):
        n = len(w["y"])
        out["tf"][b, :n], out["pf"][b, :n], out["fa"][b, :n, :n] = w["tf"], w["pf"], w["fa"]
        out["cl"][b, :n], out["y"][b, :n], out["valid"][b, :n], out["mask"][b, :n] = w["cl"], w["y"], w["valid"], True
    out["valid"][~out["mask"]] = False
    out["valid"][~out["mask"], NEW] = True
    return {k: torch.from_numpy(v).to(dev) for k, v in out.items()}


class RMSNorm(nn.Module):
    def __init__(self, d, scale=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d)) if scale else None

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w if self.w is not None else x


class GeGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.g, self.u, self.o = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.o(F.gelu(self.g(x), approximate="tanh") * self.u(x))


class SelfCond(nn.Module):

    def __init__(self, d, ff):
        super().__init__()
        self.pre, self.post, self.f = RMSNorm(d), RMSNorm(d, scale=False), GeGLU(d, ff)

    def forward(self, emb, soft):
        return self.post(emb + self.f(self.pre(soft)))


class Block(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ff = GeGLU(d, ff)
        self.drop = drop

    def forward(self, x, bias):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.o(a.transpose(1, 2).reshape(B, T, D)), self.drop, self.training)
        return x + F.dropout(self.ff(self.n2(x)), self.drop, self.training)


def mlp(i, d, o):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))


class Linker(nn.Module):
    def __init__(self, d=128, layers=6, heads=4, drop=0.1, team=False):
        super().__init__()
        self.d, self.heads = d, heads
        ft_, fp_ = Fe.FT + 3 * team, Fe.FP + 2 * team + 2 * EMB["on"]
        self.tf = mlp(ft_, 2 * d, d)
        self.pf = mlp(fp_, d, d)
        self.cand_proj = nn.Linear(d, d, bias=False)
        self.special = nn.Embedding(2, d)
        self.claimed = nn.Embedding(4, d)
        self.sc = SelfCond(d, 4 * d)
        self.bias = mlp(FA + 4, 32, heads)
        self.blocks = nn.ModuleList([Block(d, heads, 4 * d, drop) for _ in range(layers)])
        self.nf = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.out_special = nn.Linear(d, 2)
        self.pair_logit = mlp(fp_, d // 2, 1)

    def vocab(self, I):
        h0 = self.tf(I["tf"])
        B, W, _ = h0.shape
        cl = I["cl"].clamp(min=0)
        hc = torch.gather(h0[:, None].expand(B, W, W, self.d), 2, cl[..., None].expand(B, W, K, self.d))
        E = self.pf(I["pf"]) + self.cand_proj(hc)
        E = torch.cat([E, self.special.weight[None, None].expand(B, W, 2, self.d)], 2)
        return h0, E, hc

    def forward(self, I, tok, sc_logits=None, sc_on=None):
        h0, E, hc = self.vocab(I)
        B, W = tok.shape
        emb = torch.gather(E, 2, tok[..., None, None].expand(B, W, 1, self.d))[:, :, 0]
        if sc_logits is None:
            soft = torch.zeros_like(emb)
        else:
            p = sc_logits.float().masked_fill(~I["valid"], -1e9).softmax(-1)
            soft = (p[..., None] * E).sum(2)
            if sc_on is not None:
                soft = soft * sc_on[:, None, None].float()
        pt = torch.where(tok < K, torch.gather(I["cl"], 2, tok.clamp(max=K - 1)[..., None])[..., 0],
                         torch.full_like(tok, -1))
        Ph = torch.zeros(B, W, W + 1, device=tok.device)
        Ph.scatter_(2, torch.where(pt >= 0, pt, torch.full_like(pt, W))[..., None], 1.0)
        Ph = Ph[..., :W]
        if sc_logits is not None:
            Ps = torch.zeros(B, W, W + 1, device=tok.device)
            idx = torch.where(I["cl"] >= 0, I["cl"], torch.full_like(I["cl"], W))
            Ps.scatter_add_(2, idx, p[..., :K])
            Ps = Ps[..., :W]
            if sc_on is not None:
                Ps = Ps * sc_on[:, None, None].float()
        else:
            Ps = torch.zeros_like(Ph)
        mask = I["mask"]
        claimed = (Ph * mask[..., None]).sum(1).clamp(max=3).long()
        x = self.sc(emb, soft) + h0 + self.claimed(claimed)
        fa = torch.cat([I["fa"], Ph[..., None], Ph.transpose(1, 2)[..., None], Ps[..., None],
                        Ps.transpose(1, 2)[..., None]], -1)
        bias = self.bias(fa).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
        for blk in self.blocks:
            x = blk(x, bias)
        h = self.nf(x)
        hk = torch.gather(h[:, None].expand(B, W, W, self.d), 2,
                          I["cl"].clamp(min=0)[..., None].expand(B, W, K, self.d))
        lc = (self.q(h)[:, :, None] * self.k(hk)).sum(-1) / math.sqrt(self.d) + self.pair_logit(I["pf"])[..., 0]
        logits = torch.cat([lc, self.out_special(h)], -1)
        return logits.masked_fill(~I["valid"], -1e4)


def noise_like(I, gen=None):
    u = torch.rand(I["valid"].shape, device=I["valid"].device, generator=gen)
    return torch.where(I["valid"], u, torch.full_like(u, -1.0)).argmax(-1)


def corrupt(I, y, gen=None):
    B, W = y.shape
    t = torch.rand(B, 1, device=y.device, generator=gen)
    m = torch.rand(B, W, device=y.device, generator=gen) < t
    y0 = torch.where(y >= 0, y, torch.full_like(y, NEW))
    return torch.where(m, noise_like(I, gen), y0)


def sample_windows(data, rng, bs, classic=True, team=False):
    keys = list(data)
    ws = []
    for _ in range(bs):
        X = data[keys[rng.integers(len(keys))]]
        P = len(X["y"])
        Wn = min(P, L)
        s = rng.integers(0, P - Wn + 1)
        idx = np.arange(s, s + Wn)
        ws.append(window(X, idx, flip=(rng.random() < 0.5, rng.random() < 0.5), classic=classic, team=team))
    return ws


def cmd_train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    data = load_split("train")
    import re
    clip = {k: re.search(r"(train|valid)_SNGS-\d+", v.get("clip", k)).group(0) for k, v in data.items()}
    dev_clips = set(sorted(set(clip.values()))[::8]) if a.holdout else set()
    tr = {k: v for k, v in data.items() if clip[k] not in dev_clips}
    dv = {k: v for k, v in data.items() if clip[k] in dev_clips and k.endswith(("_L0", "_L1")) or
          (clip[k] in dev_clips and "_L" not in k)}
    print("train clips", len(tr), "tracklets", sum(len(v["y"]) for v in tr.values()),
          "dev clips", len(dv), flush=True)
    EMB["on"] = a.emb
    model = Linker(a.d, a.layers, a.heads, a.drop, a.team).to(dev)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    init = None
    if a.init:
        st = torch.load(os.path.join(RUNS, a.init, "model.pt"), map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        I = collate(sample_windows(tr, rng, a.batch, not a.no_classic, a.team), dev)
        y = I["y"]
        if a.rollout:
            xt, sc, sc_on = rollout_inputs(model, I, rng)
        else:
            xt = corrupt(I, y)
            sc_on = torch.rand(y.shape[0], device=dev) < 0.5
            with torch.no_grad():
                sc = model(I, xt).detach()
        logits = model(I, xt, sc, sc_on)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps:
            msg = "step %d loss %.4f %.0fs" % (step, loss.item(), time.time() - t0)
            if dv:
                msg += " dev " + json.dumps(dev_eval(model, dv, dev, classic=not a.no_classic, team=a.team))
            print(msg, flush=True)
    torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(out, "model.pt"))


@torch.no_grad()
def rollout_inputs(model, I, rng):
    model.eval()
    N = 16
    stop = int(rng.integers(0, N))
    x, sc = sampler(model, I, N=N, stop_after=stop, return_state=True)
    model.train()
    sc_on = torch.ones(x.shape[0], dtype=torch.bool, device=x.device) if sc is not None else None
    return x, sc, sc_on


@torch.no_grad()
def dev_eval(model, dv, dev, classic=True, team=False):
    model.eval()
    acc, nll, n = 0, 0.0, 0
    for X in dv.values():
        for idx in windows_of(len(X["y"]))[0]:
            I = collate([window(X, idx, classic=classic, team=team)], dev)
            x = noise_like(I)
            lg = model(I, x)
            lg = model(I, lg.argmax(-1), lg)
            y = I["y"][0]
            p = lg[0].float().log_softmax(-1)
            acc += (p.argmax(-1) == y).sum().item()
            nll += -p[torch.arange(len(y)), y].sum().item()
            n += len(y)
    return {"acc": round(acc / n, 4), "nll": round(nll / n, 4)}


def windows_of(P, Wn=L, stride=L // 2):
    if P <= Wn:
        return [np.arange(P)], np.zeros(P, int)
    starts = list(range(0, P - Wn, stride)) + [P - Wn]
    wins = [np.arange(s, s + Wn) for s in starts]
    centre = np.array([s + Wn / 2 for s in starts])
    owner = np.abs(np.arange(P)[:, None] - centre[None]).argmin(1)
    return wins, owner


def exclusive(I, xd, p, acc, committed_tok):
    B, W = xd.shape
    tgt = torch.where(xd < K, torch.gather(I["cl"], 2, xd.clamp(max=K - 1)[..., None])[..., 0],
                      torch.full_like(xd, -1))
    pr = torch.gather(p, 2, xd[..., None])[..., 0]
    acc = acc.clone()
    tgt_np, pr_np, acc_np = tgt.cpu().numpy(), pr.cpu().numpy(), acc.cpu().numpy()
    ct = committed_tok
    for b in range(B):
        owner = {}
        if ct is not None:
            ctb, cmb = ct[0][b], ct[1][b]
            for i in np.flatnonzero(cmb & ~acc_np[b]):
                if ctb[i] >= 0:
                    owner[int(ctb[i])] = (2.0, i)
        order = np.argsort(-pr_np[b])
        for i in order:
            if not acc_np[b, i] or tgt_np[b, i] < 0:
                continue
            j = int(tgt_np[b, i])
            if j in owner:
                acc_np[b, i] = False
            else:
                owner[j] = (pr_np[b, i], i)
    return torch.from_numpy(acc_np).to(acc.device), tgt


@torch.no_grad()
def sampler(model, I, N=24, b=0.1, tmax=0.8, tmin=0.4, e_stop=0.005, schedule=True, gen=None,
            stop_after=None, return_state=False, constrain=True):
    B, W = I["mask"].shape
    dev = I["mask"].device
    canvas = I["mask"]
    nC = canvas.sum(-1, keepdim=True)
    x = noise_like(I, gen)
    sc = None
    committed = torch.zeros_like(canvas)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    prev = torch.full((B, W), -1, dtype=torch.long, device=dev)
    out = x.clone()
    p1 = None
    steps_used = torch.zeros(B, device=dev)
    ent_hist = []
    for n in range(N, 0, -1):
        if stop_after is not None and N - n == stop_after:
            break
        logits = model(I, x, sc)
        tau = tmin + (tmax - tmin) * n / N
        pl = logits.float() / tau
        p = pl.masked_fill(~I["valid"], -1e9).softmax(-1)
        new_am = pl.argmax(-1)
        xd = torch.multinomial(p.flatten(0, 1), 1, generator=gen).view(B, W)
        e = -(p * p.clamp(min=1e-12).log()).sum(-1)
        p1 = logits.float().masked_fill(~I["valid"], -1e9).softmax(-1)
        e1 = -(p1 * p1.clamp(min=1e-12).log()).sum(-1)
        ent_hist.append(((e1 * canvas).sum() / canvas.sum()).item())
        e_c = torch.where(canvas, e, torch.full_like(e, float("inf")))
        se, si = torch.sort(e_c, dim=-1, stable=True)
        acc_sorted = ((se.cumsum(-1) - se) <= b) & torch.isfinite(se)
        if schedule:
            frac = 1.0 - math.cos(math.pi / 2 * (N - n + 1) / N)
            rank = torch.arange(W, device=dev)[None]
            acc_sorted |= (rank < torch.ceil(frac * nC)) & torch.isfinite(se)
        acc = torch.zeros_like(acc_sorted).scatter(1, si, acc_sorted)
        if constrain:
            acc, _ = exclusive(I, xd, p, acc, None)
        committed = torch.where(finished[:, None], committed, acc & canvas)
        xa = torch.where(acc, xd, x)
        xr = torch.where(canvas & ~acc, noise_like(I, gen), xa)
        f = finished[:, None]
        out = torch.where(f, out, torch.where(acc, xd, new_am))
        x = torch.where(f, x, xr)
        sc = pl if sc is None else torch.where(f[..., None], sc, pl)
        steps_used += (~finished).float()
        mean_e = (e * canvas).sum(-1) / canvas.sum(-1).clamp(min=1)
        stable = ((new_am == prev) | ~canvas).all(-1)
        stop = stable & (mean_e < e_stop) & ~finished
        out = torch.where(stop[:, None], new_am, out)
        finished |= stop
        prev = new_am
        if finished.all():
            break
    if return_state:
        return x, sc
    return out, p1, steps_used, ent_hist


def hungarian_clean(tok, logp, cl):
    W = len(tok)
    tgt = np.where(tok < K, cl[np.arange(W), np.minimum(tok, K - 1)], -1)
    u, c = np.unique(tgt[tgt >= 0], return_counts=True)
    bad = set(u[c > 1].tolist())
    if not bad:
        return tok
    rows = np.flatnonzero([t in bad for t in tgt])
    cols = sorted(bad)
    M = np.full((len(rows), len(cols) + len(rows)), 1e6)
    for a, i in enumerate(rows):
        for k in range(K):
            if cl[i, k] in bad:
                M[a, cols.index(cl[i, k])] = -logp[i, k]
        M[a, len(cols) + a] = -max(logp[i, NEW], logp[i, JUNK])
    r, c = linear_sum_assignment(M)
    tok = tok.copy()
    for a, cc in zip(r, c):
        i = rows[a]
        if cc < len(cols):
            tok[i] = int(np.flatnonzero(cl[i] == cols[cc])[0])
        else:
            tok[i] = NEW if logp[i, NEW] >= logp[i, JUNK] else JUNK
    return tok


@torch.no_grad()
def link_clip(model, X, dev, mode="sampler", N=24, seed=0, classic=True, b=0.1, team=False):
    P = len(X["t0"])
    wins, owner = windows_of(P)
    tok = np.full(P, NEW)
    logp = np.zeros((P, V))
    stats = []
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    for wi, idx in enumerate(wins):
        w = window(X, idx, classic=classic, team=team)
        I = collate([w], dev)
        if mode == "oneshot":
            lg = model(I, noise_like(I, gen))
            lp = lg[0].float().log_softmax(-1)
            t = lp.argmax(-1)
            st, eh = 1, [float((-(lp.exp() * lp).sum(-1)).mean())]
        elif mode == "oneshot_sc":
            lg = model(I, noise_like(I, gen))
            lg = model(I, lg.argmax(-1), lg)
            lp = lg[0].float().log_softmax(-1)
            t = lp.argmax(-1)
            st, eh = 2, [float((-(lp.exp() * lp).sum(-1)).mean())]
        else:
            t, p1, steps, eh = sampler(model, I, N=N, gen=gen, b=b)
            t = t[0]
            lp = p1[0].clamp(min=1e-12).log()
            st = steps.item()
        stats.append((st, eh[-1]))
        own = owner[idx] == wi
        loc_cl = w["cl"]
        tt = hungarian_clean(t.cpu().numpy(), lp.cpu().numpy(), loc_cl)
        tok[idx[own]] = tt[own]
        logp[idx[own]] = lp.cpu().numpy()[own]
    tok = hungarian_clean(tok, logp, X["cand"])
    return tok, logp, stats


def _score_job(args):
    name, tracks_by_var, path = args
    res = pickle.load(open(path, "rb"))
    return name, {v: {m: S.score(tr, res, m) for m in ("v1", "v2")} for v, tr in tracks_by_var.items()}


def cmd_eval(a):
    from concurrent.futures import ProcessPoolExecutor
    dev = "cuda"
    st = torch.load(os.path.join(RUNS, a.name, "model.pt"), map_location=dev, weights_only=False)
    ar = st["args"]
    classic = not ar.get("no_classic", False)
    team = ar.get("team", False)
    EMB["on"] = ar.get("emb", False)
    model = Linker(ar["d"], ar["layers"], ar["heads"], ar["drop"], team).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    data = load_split("valid", a.level)
    if a.seqs:
        data = dict(list(data.items())[:a.seqs])
    jobs, diag = [], {}
    for ci, (name, X) in enumerate(data.items()):
        rows = X["rows"]
        name = X.get("clip", name)
        res = pickle.load(open(X.get("trk_path", os.path.join(TRK, name + ".pkl")), "rb"))
        tv = {"ours": res["ours"], "k2": res["k2"], "pieces": res["pieces"],
              "oracle": Fe.decode(rows, X["cand"], X["y"]), "canvas_nolink": {r["id"]: r["obs"] for r in rows}}
        y = X["y"]
        d = {}
        for mode in a.modes.split(","):
            tok, logp, stats = link_clip(model, X, dev, mode=mode, N=a.N, seed=ci, classic=classic, b=a.b, team=team)
            tv["d4_" + mode] = Fe.decode(rows, X["cand"], tok)
            tv["d4_" + mode + "_dropjunk"] = Fe.decode(rows, X["cand"], tok, drop_junk=True)
            if "fb" in X:
                tv["d4_" + mode + "_fbk2"] = decode_fb(rows, X["cand"], tok, X["fb"])
            d[mode] = dict(acc=float((tok == y).mean()), link_acc=float((tok[y < K] == y[y < K]).mean()),
                           steps=float(np.mean([s for s, _ in stats])), ent=float(np.mean([e for _, e in stats])),
                           nll=float(-logp[np.arange(len(y)), y].mean()))
        diag[name] = d
        jobs.append((name, tv, X.get("trk_path", os.path.join(TRK, name + ".pkl"))))
        print(name, json.dumps(d), flush=True)
    per = {}
    with ProcessPoolExecutor(8) as ex:
        for name, r in ex.map(_score_job, jobs):
            per[name] = r
    names = sorted(per)
    variants = list(per[names[0]])
    summ = {}
    for m in ("v1", "v2"):
        for v in variants:
            summ["%s/%s" % (m, v)] = {k: round(float(np.mean([per[n][v][m][k] for n in names])), 4) for k in S.METRICS}
    ci = {}
    for m in ("v1", "v2"):
        for v in variants:
            if not v.startswith("d4"):
                continue
            for base in ("ours", "k2"):
                for k in ("HOTA", "AssA", "IDF1"):
                    ci["%s/%s-%s/%s" % (m, v, base, k)] = [round(x, 4) for x in S.boot(
                        [per[n][v][m][k] - per[n][base][m][k] for n in names])]
    dsum = {mode: {k: round(float(np.mean([diag[n][mode][k] for n in names])), 4)
                   for k in ("acc", "link_acc", "steps", "ent", "nll")} for mode in a.modes.split(",")}
    print("SUMMARY", json.dumps(summ, indent=0))
    print("CI", json.dumps(ci, indent=0))
    print("DIAG", json.dumps(dsum))
    json.dump({"summary": summ, "ci": ci, "diag": dsum, "per_clip": per, "diag_clip": diag, "args": vars(a)},
              open(os.path.join(RUNS, a.name, "eval_%s.json" % a.tag), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--name", default="d4a")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--no-classic", action="store_true")
    ap.add_argument("--rollout", action="store_true")
    ap.add_argument("--team", action="store_true")
    ap.add_argument("--emb", action="store_true")
    ap.add_argument("--init", default="")
    ap.add_argument("--seqs", type=int, default=0)
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--b", type=float, default=0.1)
    ap.add_argument("--modes", default="oneshot,oneshot_sc,sampler")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--canvas", default="canvas")
    ap.add_argument("--level", default="")
    a = ap.parse_args()
    global CAN
    CAN = a.canvas
    {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
