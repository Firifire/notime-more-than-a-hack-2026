"""Track D4 x K2 v7 composed player tracker for the provided match: the code path that wrote Track D4's
players_video_d4h_k2v7_fbveto.npz (tools/trackD4/compose.py --name d4h --fallback --veto --k2emb _h3, i.e. Track K2's
tools/trackK2/run_match.py --emb _h3 with pass 2 replaced by the D4 learned linker), moved into src/ so that run.py can
rebuild it bit-exactly from the cached detections. Functions below are copied verbatim from tools/trackD4/feats.py,
linker.py, match.py, gsr_compose.py (commit b70ce3d / 0192077); only module plumbing differs.

Pipeline per half (as run_match): candidates -> pitch filter -> ATD association (team votes, keeper slots) -> v1 online
tracker (track_players.run_tracker) -> K2 v7 pass 1 pieces + K2's classical linking WITH its re-ID embedding
(vision.k2track7.track_v2) -> D4 learned linker over K2's pieces in 30 s windows (DiffusionGemma-style joint canvas:
predecessor-pointer token per tracklet, entropy-bounded sampler, seeded torch.Generator per window, in-sampler
exclusion) -> where D4 starts a new identity, K2's link if free -> identity-level confident-team veto -> densify_v2
(fill 125) -> K2 identity_teams (team decided once per identity) -> per-frame slot writer + priority cut to 32 slots.

Learned parts (shipped weights, notes/trackD4/SHIP.md): cache/vision/track/k2_reid_h3.pt (K2's re-ID CNN, via
vision.reid_k2) and cache/vision/track/d4h_linker.pt (the D4 linker). No match label or match statistic is used.

    python -m vision.d4track --cands cache/vision/players_cands --out FILE [--emb WORK/match_emb_h3.npz]
    track_match(cands_dir, work_dir) -> path of players_video_d4h.npz in work_dir
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from vision import k2track7 as K2
from vision import track_players as TP
from vision.players_common import hist_dist
from vision.common import CACHE

Fe = Lk = sys.modules[__name__]

LINKER_WEIGHTS = os.path.join(CACHE, "track", "d4h_linker.pt")

K = 12
NEW, JUNK = K, K + 1
V = K + 2
MAXGAP = 250
DT = 0.04
FT = 16 + 48
FP = 14
BASE_M = 3.0
L = 256
FA = 10
CLASSIC = (10, 11, 12)
EMB = {"on": True}


def tracklet_table(pieces, xy, cf, ft):
    n = len(xy)
    frames = np.arange(n)
    fpos = {t: t for t in frames}
    amb, clean = K2.frame_flags(frames, xy)
    rows = []
    for k, obs in pieces.items():
        s = K2._summ(obs, xy, ft, cf, fpos, clean)
        ts = np.array([t for t, _ in obs])
        ps = np.array([xy[t][d] for t, d in obs])
        cs = np.array([cf[t][d] for t, d in obs])
        fa = np.array([ft[t][d] for t, d in obs])
        fa = fa[fa.sum(1) > 0]
        s["kit_all"] = fa.mean(0) if len(fa) else None
        s["id"] = k
        s["obs"] = obs
        s["pm"] = ps.mean(0)
        s["psd"] = ps.std(0).mean() if len(ps) > 1 else 0.0
        s["conf"] = float(cs.mean())
        s["amb"] = float(np.mean([amb[t][d] for t, d in obs]))
        s["dur"] = int(ts[-1] - ts[0] + 1)
        rows.append(s)
    rows.sort(key=lambda r: (r["t0"], r["t1"], r["id"]))
    return rows


def kit_of(r):
    return r["kit"] if r["kit"] is not None else r["kit_all"]


def pair_cost(a, b):
    gap = b["t0"] - a["t1"]
    if gap < 1 or gap > MAXGAP:
        return None
    g = gap * DT
    d = float(np.hypot(*(b["p0"] - a["p1"])))
    if d > BASE_M + K2.P.v_max * g:
        return None
    h = min(g, K2.P.vel_horizon)
    ea = np.hypot(*(a["p1"] + a["v1"] * h - b["p0"]))
    eb = np.hypot(*(b["p0"] - b["v0"] * h - a["p1"]))
    e = float(min(0.5 * (ea + eb), d))
    ka, kb = kit_of(a), kit_of(b)
    kd = float(hist_dist(ka, kb)) if ka is not None and kb is not None else -1.0
    cost = e / (K2.P.sig0 + K2.P.sig_v * g) + K2.P.w_gap * g + K2.P.w_kit * (kd if kd >= 0 else 0.15)
    return dict(cost=cost, g=g, d=d, e=e, kd=kd)


def canvas_inputs(rows, n, ours_of=None, k2_of=None, k2_prev=None):
    Pn = len(rows)
    tf = np.zeros((Pn, FT), np.float32)
    cand = np.full((Pn, K), -1, np.int64)
    pf = np.zeros((Pn, K, FP), np.float32)
    kit = np.zeros((Pn, 48), np.float32)
    kv = np.zeros(Pn, np.float32)
    for i, r in enumerate(rows):
        kk = kit_of(r)
        if kk is not None:
            kit[i], kv[i] = kk, 1.0
        tf[i, :16] = [r["t0"] / n, r["t1"] / n, np.log1p(r["n"]) / 6, np.log1p(r["dur"]) / 6, r["n"] / r["dur"],
                      r["conf"], r["amb"], r["p0"][0] / 105, r["p0"][1] / 68, r["p1"][0] / 105, r["p1"][1] / 68,
                      r["v0"][0] / 8, r["v0"][1] / 8, r["v1"][0] / 8, r["v1"][1] / 8, (r["kit"] is not None) * 1.0]
        tf[i, 16:] = kit[i]
    t0 = np.array([r["t0"] for r in rows])
    t1 = np.array([r["t1"] for r in rows])
    for i, b in enumerate(rows):
        cs = []
        for j in np.flatnonzero((t1[:i] < t0[i]) & (t0[i] - t1[:i] <= MAXGAP)):
            c = pair_cost(rows[j], b)
            if c is not None:
                cs.append((c["cost"], j, c))
        cs.sort(key=lambda z: z[0])
        for k, (cost, j, c) in enumerate(cs[:K]):
            a = rows[j]
            cand[i, k] = j
            same_o = float(ours_of is not None and ours_of[a["id"]] == ours_of[b["id"]])
            same_k = float(k2_of is not None and k2_of[a["id"]] == k2_of[b["id"]])
            k2p = float(k2_prev is not None and k2_prev.get(b["id"]) == a["id"])
            pf[i, k] = [c["g"] / 4, c["d"] / 20, c["d"] / (BASE_M + K2.P.v_max * c["g"]),
                        c["e"] / (K2.P.sig0 + K2.P.sig_v * c["g"]) / 3, max(c["kd"], 0), float(c["kd"] >= 0),
                        cost / 5, k / K, np.log1p(a["n"]) / 6, a["conf"], same_o, same_k, k2p,
                        len(cs) / 30]
    return dict(tf=tf, cand=cand, pf=pf, t0=t0, t1=t1, pm=np.array([r["pm"] for r in rows], np.float32),
                kit=kit, kv=kv)


def targets(rows, cand, maj):
    Pn = len(rows)
    y = np.full(Pn, NEW, np.int64)
    claimed = set()
    last = {}
    for i, r in enumerate(rows):
        g = maj[r["id"]]
        if g < 0:
            y[i] = JUNK
            continue
        prev = [(t1, j) for t1, j in last.get(g, []) if t1 < r["t0"] and j not in claimed]
        if prev:
            j = max(prev)[1]
            w = np.flatnonzero(cand[i] == j)
            if len(w):
                y[i] = int(w[0])
                claimed.add(j)
        last.setdefault(g, []).append((r["t1"], i))
    return y


def team_inputs(rows, det_team, cand):
    Pn = len(rows)
    tv = np.zeros((Pn, 3), np.float32)
    conf_team = np.full(Pn, -1)
    for i, r in enumerate(rows):
        v = np.zeros(2)
        for t, d in r["obs"]:
            tm = det_team[t][d]
            if tm >= 0:
                v[tm] += 1
        s = v.sum()
        if s:
            tv[i, :2] = v / s
        tv[i, 2] = np.log1p(s) / 6
        if s >= 5 and v.max() >= 0.85 * s:
            conf_team[i] = int(np.argmax(v))
    pt = np.zeros((Pn, K, 2), np.float32)
    for i in range(Pn):
        for k in range(K):
            j = cand[i, k]
            if j >= 0 and conf_team[i] >= 0 and conf_team[j] >= 0:
                pt[i, k, 0] = float(conf_team[i] != conf_team[j])
                pt[i, k, 1] = float(conf_team[i] == conf_team[j])
    return tv, pt


def emb_inputs(rows, embl, cand):
    Pn = len(rows)
    m = np.full((Pn, 128), np.nan, np.float32)
    for i, r in enumerate(rows):
        es = [embl[t][d] for t, d in r["obs"] if np.isfinite(embl[t][d][0])]
        if es:
            v = np.mean(es, 0)
            m[i] = v / max(np.linalg.norm(v), 1e-9)
    pe = np.zeros((Pn, K, 2), np.float32)
    ok = np.isfinite(m[:, 0])
    for i in range(Pn):
        for k in range(K):
            j = cand[i, k]
            if j >= 0 and ok[i] and ok[j]:
                pe[i, k] = [1.0 - float(m[i] @ m[j]), 1.0]
    return pe


def clip_canvas(res):
    pieces = res["pieces"]
    rows = tracklet_table(pieces, res["xy"], res["cf"], res["ft"])
    ours_of, k2_of, k2_prev = {}, {}, {}
    o_map = {o: k for k, obs in res["ours"].items() for o in obs}
    for r in rows:
        ours_of[r["id"]] = o_map.get(r["obs"][0], -1)
    km = {o: k for k, obs in res["k2"].items() for o in obs}
    for r in rows:
        k2_of[r["id"]] = km.get(r["obs"][0], -1)
    by = {}
    for r in rows:
        by.setdefault(k2_of[r["id"]], []).append(r)
    for k, rs in by.items():
        rs.sort(key=lambda r: r["t0"])
        for a, b in zip(rs[:-1], rs[1:]):
            k2_prev[b["id"]] = a["id"]
    X = canvas_inputs(rows, res["n"], ours_of, k2_of, k2_prev)
    X["ours_id"] = np.array([ours_of[r["id"]] for r in rows])
    X["k2_id"] = np.array([k2_of[r["id"]] for r in rows])
    if "maj" in res:
        X["y"] = targets(rows, X["cand"], res["maj"])
    if res.get("det_team") is not None:
        X["team"], X["pteam"] = team_inputs(rows, res["det_team"], X["cand"])
    if res.get("embl") is not None:
        X["pemb"] = emb_inputs(rows, res["embl"], X["cand"])
    return rows, X


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


def _conf_team(v):
    return K2._team_conf(v)


def link_windows(model, dev, mode, N, classic, team, pieces, ours, k2src, xyL, cfL, ftL, n, dt=None, fallback=None,
                 embl=None, veto=False):
    W, S = 750, 375
    start = {k: obs[0][0] for k, obs in pieces.items()}
    pred, stats = {}, []
    cnt = dict(links=0, new=0, junk=0)
    w0 = 0
    while True:
        w1 = min(w0 + W, n)

        def clip(d):
            o = {}
            for k, obs in d.items():
                v = [(t - w0, dd) for t, dd in obs if w0 <= t < w1]
                if v:
                    o[k] = v
            return o
        res = dict(n=w1 - w0, xy=xyL[w0:w1], cf=cfL[w0:w1], ft=ftL[w0:w1], pieces=clip(pieces), ours=clip(ours),
                   k2=clip(k2src))
        if dt is not None:
            res["det_team"] = dt[w0:w1]
        if embl is not None:
            res["embl"] = embl[w0:w1]
        lo = w0 + S // 2 if w0 > 0 else -1
        hi = w0 + S // 2 + S if w1 < n else n + 1
        if res["pieces"]:
            rows, X = Fe.clip_canvas(res)
            tok, logp, st_ = Lk.link_clip(model, X, dev, mode=mode, N=N, classic=classic,
                                          team=team)
            stats += st_
            for i, r in enumerate(rows):
                if not (lo <= start[r["id"]] < hi) or start[r["id"]] < w0:
                    continue
                t = int(tok[i])
                if t < Fe.K and X["cand"][i, t] >= 0:
                    j = rows[int(X["cand"][i, t])]["id"]
                    pred[r["id"]] = (j, float(np.exp(logp[i, t])))
                    cnt["links"] += 1
                else:
                    cnt["new" if t == Fe.NEW else "junk"] += 1
        if w1 >= n:
            break
        w0 += S
    best = {}
    for i, (j, pr) in pred.items():
        if j not in best or pr > best[j][1]:
            best[j] = (i, pr)
    keep = {i: j for j, (i, _) in best.items()}
    root = {}
    votes = {}
    nveto = nfb = 0
    if veto and dt is not None:
        for k, obs in pieces.items():
            v = np.zeros(2)
            for t, d in obs:
                tm = dt[t][d]
                if tm >= 0:
                    v[tm] += 1
            votes[k] = v
    fb = fallback or {}
    claimed = set()
    d4_targets = set(keep.values())
    for k in sorted(pieces, key=lambda k: start[k]):
        root[k] = k
        for src_, j in (("d4", keep.get(k)), ("fb", fb.get(k))):
            if j is None or j in claimed or j not in root or (src_ == "fb" and j in d4_targets):
                continue
            if votes:
                r = root[j]
                a_, b_ = _conf_team(votes[r]), _conf_team(votes[k])
                if a_ >= 0 and b_ >= 0 and a_ != b_:
                    nveto += 1
                    continue
            root[k] = root[j]
            claimed.add(j)
            if votes:
                votes[root[k]] = votes[root[k]] + votes[k]
            nfb += src_ == "fb"
            break
    cnt["fallback_links"] = nfb
    cnt["team_veto"] = nveto
    ids = {}
    for k, obs in pieces.items():
        ids.setdefault(root[k], []).extend(obs)
    return ids, dict(pieces=len(pieces), d4=len(ids), conflicts_dropped=len(pred) - len(keep), windows=len(stats),
                     steps=float(np.mean([s for s, _ in stats])), ent=float(np.mean([e for _, e in stats])), **cnt)


def chain_fallback(k2ids, pieces):
    p_of = {o: k for k, obs in pieces.items() for o in obs}
    fb = {}
    for _, obs in k2ids.items():
        ps = []
        for o in sorted(obs):
            p = p_of.get(o)
            if p is not None and (not ps or ps[-1] != p):
                ps.append(p)
        for a, b in zip(ps[:-1], ps[1:]):
            fb[b] = a
    return fb


FILL_MAX_MATCH = 125


def load_linker(dev="cuda", path=LINKER_WEIGHTS):
    st = torch.load(path, map_location=dev, weights_only=True)
    ar = st["args"]
    EMB["on"] = ar.get("emb", False)
    model = Linker(ar["d"], ar["layers"], ar["heads"], ar["drop"], ar.get("team", False)).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    return model, ar


def d4_identities(model, ar, dev, frames, xy, conf, feat, in_view=None, det_team=None, tracklets=None, emb=None):
    classic, team = not ar.get("no_classic", False), ar.get("team", False)
    k2ids, pieces = K2.track_v2(frames, xy, conf, feat, in_view, det_team, tracklets=tracklets, return_pieces=True,
                                emb=emb)
    f0 = int(frames[0])
    n = int(frames[-1]) - f0 + 1
    e2, e1, ef = np.zeros((0, 2)), np.zeros(0), np.zeros((0, 48), np.float32)
    xyL, cfL, ftL, dtL = [e2] * n, [e1] * n, [ef] * n, [np.zeros(0, int)] * n
    emL = [np.zeros((0, 128), np.float32)] * n
    if emb is not None:
        for i, t in enumerate(frames):
            emL[t - f0] = np.asarray(emb[i], np.float32)
    for i, t in enumerate(frames):
        xyL[t - f0], cfL[t - f0], ftL[t - f0] = xy[i], conf[i], feat[i]
        if det_team is not None:
            dtL[t - f0] = np.asarray(det_team[i])
    loc = lambda d: {k: sorted((t - f0, dd) for t, dd in obs) for k, obs in d.items()}  # noqa: E731
    pl = loc(pieces)
    trk = loc(tracklets)
    ours = TP.stitch(trk, np.arange(n), xyL, ftL)
    k2src = ours
    fb = chain_fallback(loc(k2ids), pl)
    ctx_saved = dict(K2._CTX)
    K2._CTX["emb"] = None
    ids, stt = link_windows(model, dev, "sampler", 24, classic, team, pl, ours, k2src, xyL, cfL, ftL, n,
                            dtL if team else None, fallback=fb, veto=True,
                            embl=emL if (EMB["on"] and emb is not None) else None)
    K2._CTX.update(ctx_saved)
    stt.update(k2_identities=len(k2ids))
    print("[d4track] D4 link %s" % json.dumps(stt), flush=True)
    out = {k: sorted((t + f0, dd) for t, dd in obs) for k, obs in ids.items()}
    return out, pieces


def build(cdir, emb_path, dev="cuda"):
    from vision.common import W, H, N_FRAMES, FPS, HALVES, load_homography, metres_to_px, px_to_metres, PITCH
    from vision.run_players import MARGIN_M, MATCH_M, K as KOUT, KBUF
    from vision.reid_k2 import load_cands
    from mtah.data import load_tracking
    model, ar = load_linker(dev)
    cands, processed = load_cands(cdir)
    Hs = load_homography()
    atd = load_tracking()
    P_atd, team_atd = atd["players"][..., :2].astype(np.float64), atd["team"].astype(int)
    fr_all = cands["frame"]
    bounds = np.searchsorted(fr_all, np.arange(N_FRAMES + 1))
    ez = np.load(emb_path)
    emb_all = np.full((len(fr_all), ez["emb"].shape[1]), np.nan, np.float16)
    emb_all[ez["idx"]] = ez["emb"]
    out = dict(
        xy=np.full((N_FRAMES, KBUF, 2), np.nan, np.float32), team=np.full((N_FRAMES, KBUF), -1, np.int8),
        track_id=np.full((N_FRAMES, KBUF), -1, np.int32), conf=np.zeros((N_FRAMES, KBUF), np.float32),
        atd_slot=np.full((N_FRAMES, KBUF), -1, np.int16), state=np.zeros((N_FRAMES, KBUF), np.int8),
        team_src=np.zeros((N_FRAMES, KBUF), np.int8), uv=np.full((N_FRAMES, KBUF, 2), np.nan, np.float32),
        processed=np.zeros(N_FRAMES, bool))
    out["processed"][processed] = True
    next_tid = 0
    for half, ga, gb in HALVES:
        t0 = time.time()
        frames = np.arange(max(ga - 50, processed.min()), gb + 51)
        frames = frames[out["processed"][frames]]
        px = P_atd[ga:gb + 1, :, 0]
        gl = np.nanmedian(np.minimum(px, PITCH[0] - px), axis=0)
        gk_slot = (gl < 16.5) & (np.isfinite(px).sum(0) > FPS * 60)
        xy, cf, ft, bx, slot, tm, gkd, em = [], [], [], [], [], [], [], []
        for t in frames:
            s, e = bounds[t], bounds[t + 1]
            b = cands["box"][s:e].astype(np.float64)
            x, y = px_to_metres(np.linalg.inv(Hs[t]), (b[:, 0] + b[:, 2]) / 2, b[:, 3])
            ok = (np.isfinite(x) & (x > -MARGIN_M) & (x < PITCH[0] + MARGIN_M) & (y > -MARGIN_M)
                  & (y < PITCH[1] + MARGIN_M))
            p = np.c_[x[ok], y[ok]]
            xy.append(p)
            cf.append(cands["conf"][s:e][ok])
            ft.append(cands["feat"][s:e][ok].astype(np.float32))
            bx.append(b[ok])
            em.append(emb_all[s:e][ok].astype(np.float32))
            ap_ = P_atd[t]
            av = np.flatnonzero(~np.isnan(ap_[:, 0]))
            sl = np.full(len(p), -1)
            if len(p) and len(av):
                D = np.hypot(*(p[:, None] - ap_[av][None]).transpose(2, 0, 1))
                r, c = linear_sum_assignment(np.where(D < MATCH_M, D, 1e6))
                k = D[r, c] < MATCH_M
                sl[r[k]] = av[c[k]]
            slot.append(sl)
            tm.append(np.where(sl >= 0, team_atd[np.maximum(sl, 0)], -1))
            gkd.append((sl >= 0) & gk_slot[np.maximum(sl, 0)])
        fpos = {t: i for i, t in enumerate(frames)}

        def in_view(t, pts):
            u, v = metres_to_px(Hs[t], pts[:, 0], pts[:, 1])
            return (u > 0) & (u < W) & (v > 0) & (v < H)
        _, clean = K2.frame_flags(frames, xy)
        tm_clean = [np.where(c_, x_, -1) for x_, c_ in zip(tm, clean)] if K2.P.team_clean else tm
        tr = TP.run_tracker(frames, xy, cf, ft, in_view=in_view)
        ids, pieces = d4_identities(model, ar, dev, frames, xy, cf, ft, in_view, tm_clean, tracklets=tr, emb=em)
        dense = K2.densify_v2(ids, frames, xy, bx, fill_max=FILL_MAX_MATCH)
        team, src = K2.identity_teams(dense, frames, tm_clean, ft, gkd, cf)
        print("[d4track] half %d: online %d, pieces %d, identities %d (%.0fs)" % (half, len(tr), len(pieces), len(ids),
                                                                                time.time() - t0), flush=True)
        fill = np.zeros(N_FRAMES, np.int32)
        for k, r in dense.items():
            tid = next_tid
            next_tid += 1
            for t, p, st, d, b in zip(r["t"], r["xy"], r["state"], r["det"], r["box"]):
                j = fill[t]
                if j >= KBUF:
                    continue
                fill[t] += 1
                out["xy"][t, j] = p
                out["track_id"][t, j] = tid
                out["state"][t, j] = st
                out["uv"][t, j] = ((b[0] + b[2]) / 2, b[3])
                if d >= 0:
                    i = fpos[t]
                    out["conf"][t, j] = cf[i][d]
                    out["atd_slot"][t, j] = slot[i][d]
                if src[k] == 3:
                    out["team_src"][t, j] = 4
                elif team[k] >= 0:
                    out["team"][t, j] = team[k]
                    out["team_src"][t, j] = 2 if src[k] == 1 else 3
    pri = ((out["state"] > 0) * 8.0 + (out["team"] >= 0) * 4 + (out["state"] == 1) * 2 + out["conf"])
    order = np.argsort(-pri, axis=1, kind="stable")[:, :KOUT]
    for key in ("xy", "uv"):
        out[key] = np.take_along_axis(out[key], order[..., None], 1)
    for key in ("team", "track_id", "conf", "atd_slot", "state", "team_src"):
        out[key] = np.take_along_axis(out[key], order, 1)
    return out


def track_match(cands_dir, work_dir, emb_path=None, out_name="players_video_d4h.npz"):
    os.makedirs(work_dir, exist_ok=True)
    if emb_path is None:
        from vision.reid_k2 import match_embeddings
        emb_path = os.path.join(work_dir, "match_emb_h3.npz")
        if not os.path.exists(emb_path):
            match_embeddings(cands_dir, emb_path)
    out = build(cands_dir, emb_path)
    out["doc"] = np.array(__doc__)
    path = os.path.join(work_dir, out_name)
    np.savez_compressed(path, **out)
    print("[d4track] wrote %s (peak CUDA memory allocated %.2f GB)" % (path, torch.cuda.max_memory_allocated() / 1e9),
          flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=os.path.join(CACHE, "players_cands"))
    ap.add_argument("--work", required=True, help="work directory (embeddings + output)")
    ap.add_argument("--emb", default=None, help="precomputed re-ID embeddings (vision.reid_k2 output)")
    ap.add_argument("--out-name", default="players_video_d4h.npz")
    a = ap.parse_args()
    track_match(a.cands, a.work, a.emb, a.out_name)


if __name__ == "__main__":
    main()
