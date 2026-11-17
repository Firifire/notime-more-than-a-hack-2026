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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.refine.model import (RMSNorm, SelfConditioning, mlp, fourier, NF as NFOUR, cell_of, cell_centres, G, nxy,  # noqa: E402
                                 shift_t)
from vision.refine import ud_data as U  # noqa: E402

ROOT = U.ROOT
RUNS = os.path.join(ROOT, "cache", "ud", "runs")
POST = os.path.join(ROOT, "cache", "ud", "post")
NPL = U.NP
NONE = NPL
MP, MC, MB = 3, NPL + 1, G
T_WIN, C0, C1 = 512, 128, 384
FPS5 = 5.0
KIN_B = (7.5, 1.0)


class Unified(nn.Module):
    def __init__(self, d=192, layers=6, heads=6, ff=768, nf=U.NF, dropout=0.1):
        super().__init__()
        self.d = d
        self.p_enc = mlp(nf + NFOUR, d, d)
        self.team_pool = nn.Linear(4 * d, d)
        self.obs_enc = mlp(NFOUR + 1, d, d)
        self.ev_norm = nn.LayerNorm(d)
        self.poss_emb = nn.Embedding(4, d)
        self.car_tok = nn.Linear(d, d)
        self.none_emb = nn.Parameter(torch.zeros(d))
        self.cmask_emb = nn.Parameter(torch.zeros(d))
        self.cell_emb = nn.Embedding(G, d)
        self.cell_pos = mlp(NFOUR, d, d)
        self.bmask_emb = nn.Parameter(torch.zeros(d))
        self.geo = mlp(6 + NFOUR, d, d)
        self.sc = SelfConditioning(d, ff)
        self.flag = nn.Embedding(3, d)
        self.pos = nn.Embedding(T_WIN, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.poss_head = nn.Linear(d, 3)
        self.cq, self.ck, self.none_head = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 1)
        self.bq, self.bk = nn.Linear(d, d), nn.Linear(d, d)
        self.register_buffer("centres", cell_centres(), persistent=False)

    def evidence(self, pf, obs):
        pv = pf[..., U.NF - 6] != 0
        pxy = pf[..., U.NF - 5:U.NF - 3]
        x = torch.cat([pf, fourier(pxy) * pv[..., None]], -1)
        h = self.p_enc(x) * pv[..., None]
        sg = pf[..., U.NF - 6]
        pools = []
        for side in (1.0, -1.0):
            m = (sg == side)
            cnt = m.sum(2, keepdim=True).clamp(min=1)
            pools += [(h * m[..., None]).sum(2) / cnt, h.masked_fill(~m[..., None], -1e4).max(2).values.clamp(min=-50)]
        e = self.team_pool(torch.cat(pools, -1))
        ob = ~torch.isnan(obs[..., 0])
        o = torch.nan_to_num(obs)
        e = e + self.obs_enc(torch.cat([fourier(o) * ob[..., None], ob[..., None].float()], -1))
        return {"e": self.ev_norm(e), "h": h, "pv": pv, "pxy": pxy, "sg": sg}

    def _tables(self, ev):
        ct = torch.where(ev["pv"][..., None], self.car_tok(ev["h"]), torch.zeros_like(ev["h"]))
        ce = self.cell_emb.weight + self.cell_pos(fourier(self.centres))
        return ct, ce

    def forward(self, ev, xp, xc, xb, flag, sc=None, sc_on=None):
        B, T = xp.shape
        ct, ce = self._tables(ev)
        ep = self.poss_emb(xp)
        cidx = xc.clamp(max=NPL - 1)
        ec = torch.gather(ct, 2, cidx[..., None, None].expand(-1, -1, 1, self.d))[:, :, 0]
        ec = torch.where((xc == NONE)[..., None], self.none_emb.expand_as(ec), ec)
        ec = torch.where((xc == MC)[..., None], self.cmask_emb.expand_as(ec), ec)
        eb = ce[xb.clamp(max=G - 1)]
        eb = torch.where((xb == MB)[..., None], self.bmask_emb.expand_as(eb), eb)
        bxy = self.centres[xb.clamp(max=G - 1)]
        cxy = torch.gather(ev["pxy"], 2, cidx[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
        csg = torch.gather(ev["sg"], 2, cidx[..., None])[..., 0]
        hasc = (xc < NPL).float()
        hasb = (xb < MB).float()
        dbc = (bxy - cxy).norm(dim=-1).clamp(max=30) / 10 * hasc * hasb
        dpl = (ev["pxy"] - bxy[:, :, None]).norm(dim=-1).masked_fill(~ev["pv"], 1e3)
        dh = dpl.masked_fill(ev["sg"] != 1, 1e3).min(-1).values.clamp(max=30) / 10
        da = dpl.masked_fill(ev["sg"] != -1, 1e3).min(-1).values.clamp(max=30) / 10
        geo = torch.cat([torch.stack([dbc, csg * hasc, hasc, hasb, dh * hasb, da * hasb], -1),
                         fourier(bxy) * hasb[..., None]], -1)
        emb = ep + ec + eb + self.geo(geo)
        if sc is None:
            soft = torch.zeros_like(emb)
        else:
            lp, lc, lb = sc
            pp = lp.float().softmax(-1)
            soft = pp @ self.poss_emb.weight[:3]
            pc = lc.float().softmax(-1)
            soft = soft + (pc[..., :NPL, None] * ct).sum(2) + pc[..., NPL:] * self.none_emb
            soft = soft + lb.float().softmax(-1) @ ce
            if sc_on is not None:
                soft = soft * sc_on[:, None, None].float()
        x = self.sc(emb, soft) + ev["e"] + self.flag(flag) + self.pos.weight[:T]
        h = self.out_norm(self.tf(x))
        lp = self.poss_head(h)
        lcc = torch.einsum("btd,btpd->btp", self.cq(h), self.ck(ev["h"])) / math.sqrt(self.d)
        lcc = lcc.masked_fill(~ev["pv"], -1e4)
        lc = torch.cat([lcc, self.none_head(h)], -1)
        lb = self.bq(h) @ self.bk(ce).T / math.sqrt(self.d)
        return lp, lc, lb


def load_view(n, seed):
    z = np.load(U.path(n, seed))
    return {k: z[k] for k in z.files}


def fold_train_views(fold):
    from mtah import gs_build as B
    Gp = B.groups()
    out = []
    for n, seed in U.views():
        if seed != B.train_seed(n):
            continue
        g = Gp[n]
        if g == "S" or (fold != "all" and g == str(fold)):
            continue
        out.append((n, seed))
    return out


def tokens_of(v):
    yp = v["y_poss"].astype(np.int64)
    kp = yp >= 0
    yc = v["y_car"].astype(np.int64)
    kc = yc >= 0
    yb = np.full(len(yp), 0, np.int64)
    kb = ~np.isnan(v["y_ball"][:, 0])
    yb[kb] = cell_of(torch.from_numpy(v["y_ball"][kb])).numpy()
    return np.where(kp, yp, 0), kp, np.where(kc, yc, 0), kc, yb, kb


def window(v, s0, rng=None):
    return _window(v, s0, rng) + (np.array(bool(v.get("dead_ok", True))),)


def _window(v, s0, rng=None):
    n = len(v["half"])
    lo, hi = max(s0, 0), min(s0 + T_WIN, n)
    pf = np.zeros((T_WIN, NPL, U.NF), np.float32)
    obs = np.full((T_WIN, 2), np.nan, np.float32)
    valid = np.zeros(T_WIN, bool)
    sl = slice(lo - s0, hi - s0)
    pf[sl] = v["pf"][lo:hi].astype(np.float32)
    obs[sl] = v["obs"][lo:hi]
    valid[sl] = True
    t = tokens_of(v)
    out = []
    for a in t:
        w = np.zeros(T_WIN, a.dtype)
        w[sl] = a[lo:hi]
        out.append(w)
    yp, kp, yc, kc, yb, kb = out
    yball = np.full((T_WIN, 2), np.nan, np.float32)
    yball[sl] = v["y_ball"][lo:hi]
    if rng is not None:
        if rng.random() < 0.5:
            pf[..., U.NF - 6] *= -1
            yp = np.where(kp & (yp < 2), 1 - yp, yp)
        fx, fy = rng.random() < 0.5, rng.random() < 0.5
        X, Y = U.NF - 5, U.NF - 4
        if fx:
            pf[..., X] = np.where(pf[..., U.NF - 6] != 0, 105 - pf[..., X], 0)
            pf[..., U.NF - 3] *= -1
            obs[:, 0] = 105 - obs[:, 0]
            yball[:, 0] = 105 - yball[:, 0]
        if fy:
            pf[..., Y] = np.where(pf[..., U.NF - 6] != 0, 68 - pf[..., Y], 0)
            pf[..., U.NF - 2] *= -1
            obs[:, 1] = 68 - obs[:, 1]
            yball[:, 1] = 68 - yball[:, 1]
        if fx or fy:
            kb = ~np.isnan(yball[:, 0])
            yb = np.zeros(T_WIN, np.int64)
            yb[kb] = cell_of(torch.from_numpy(yball[kb])).numpy()
        if rng.random() < 0.2:
            obs[:] = np.nan
            pf[..., U.NF - 1] = 99.0 * (pf[..., U.NF - 6] != 0)
    return pf, obs, valid, yp, kp, yc, kc, yb, kb


class Windows(torch.utils.data.IterableDataset):
    def __init__(self, names, seed):
        self.names, self.seed = names, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        k = wi.id if wi else 0
        rng = np.random.default_rng([self.seed, k])
        mine = self.names[k::(wi.num_workers if wi else 1)] or self.names
        cache = {}
        while True:
            n, seed = mine[int(rng.integers(len(mine)))]
            if (n, seed) not in cache:
                if len(cache) > 6:
                    cache.pop(next(iter(cache)))
                cache[(n, seed)] = load_view(n, seed)
            v = cache[(n, seed)]
            N = len(v["half"])
            s0 = int(rng.integers(-C0, max(-C0 + 1, N - C1)))
            w = window(v, s0, rng)
            if w[4][C0:C1].sum() + w[6][C0:C1].sum() < 20:
                continue
            yield tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in w)


def noise(pv, gen, B, T, dev):
    xp = torch.randint(0, 3, (B, T), device=dev, generator=gen)
    npres = pv.sum(-1)
    kc = (torch.rand(B, T, device=dev, generator=gen) * npres).floor().long().clamp(max=NPL - 1)
    xc = torch.where((torch.rand(B, T, device=dev, generator=gen) < 0.3) | (npres == 0),
                     torch.full_like(kc, NONE), kc)
    xb = torch.randint(0, G, (B, T), device=dev, generator=gen)
    return xp, xc, xb


def corrupt_all(clean, known, pv, gen, B, T, dev):
    nz = noise(pv, gen, B, T, dev)
    flag = torch.zeros(B, T, dtype=torch.long, device=dev)
    flag[:, :C0] = 1
    flag[:, C1:] = 1
    la = torch.rand(B, device=dev, generator=gen) < 0.25
    ra = torch.rand(B, device=dev, generator=gen) < 0.5
    absent = torch.zeros(B, T, dtype=torch.bool, device=dev)
    absent[:, :C0] |= la[:, None]
    absent[:, C1:] |= ra[:, None]
    out = []
    for c, k, z_, mk in zip(clean, known, nz, (MP, MC, MB)):
        t = torch.rand(B, 1, device=dev, generator=gen)
        tc = torch.rand(B, 1, device=dev, generator=gen) * 0.25
        p = torch.where(flag == 0, t.expand(B, T), tc.expand(B, T))
        u = torch.rand(B, T, device=dev, generator=gen)
        x = torch.where((u < p) | ~k, z_, c)
        out.append(torch.where(absent, torch.full_like(x, mk), x))
    flag = torch.where(absent, torch.full_like(flag, 2), flag)
    return out, flag


def cmd_train(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(a.seed + (0 if a.fold == "all" else int(a.fold)))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    names = fold_train_views(a.fold)
    miss = [x for x in names if not os.path.exists(U.path(*x))]
    if miss:
        print("WARNING %d training views missing (skipped), e.g. %s" % (len(miss), miss[:3]), flush=True)
    names = [x for x in names if os.path.exists(U.path(*x))]
    odir = os.path.join(RUNS, a.name, "fold_%s" % a.fold)
    os.makedirs(odir, exist_ok=True)
    model = Unified(d=a.d, layers=a.layers, heads=a.heads, ff=4 * a.d).to(dev)
    print("fold %s: %d training views, params %.2f M" % (a.fold, len(names), sum(p.numel() for p in model.parameters()) / 1e6),
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * (0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)) * 0.95 + 0.05))
    dl = torch.utils.data.DataLoader(Windows(names, a.seed), batch_size=a.batch, num_workers=a.workers,
                                     persistent_workers=True, prefetch_factor=2)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    step, hist, t0 = 0, [], time.time()
    for pf, obs, valid, yp, kp, yc, kc, yb, kb, dok in dl:
        if step >= a.steps:
            break
        pf, obs, yp, kp, yc, kc, yb, kb, dok = (x.to(dev) for x in (pf, obs, yp, kp, yc, kc, yb, kb, dok))
        B, T = yp.shape
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ev = model.evidence(pf, obs)
            (xp, xc, xb), flag = corrupt_all((yp, yc, yb), (kp, kc, kb), ev["pv"], gen, B, T, dev)
            sc_on = torch.rand(B, device=dev, generator=gen) < 0.5
            with torch.no_grad():
                sc = model(ev, xp, xc, xb, flag)
            lp, lc, lb = model(ev, xp, xc, xb, flag, tuple(s.detach() for s in sc), sc_on)
        cm = flag == 0
        lpf = lp.float().log_softmax(-1)
        ce3 = -lpf.gather(-1, yp[..., None])[..., 0]
        ha = torch.logsumexp(lpf[..., :2], -1)
        ce2 = -(lpf.gather(-1, yp.clamp(max=1)[..., None])[..., 0] - ha)
        full = dok.reshape(B)[:, None].expand(B, T)
        wp = (kp & cm & (full | (yp < 2))).float()
        cep = torch.where(full, ce3, ce2)
        loss = (cep * wp).sum() / wp.sum().clamp(min=1)
        parts = [((lp.argmax(-1) == yp).float() * wp).sum().item() / max(wp.sum().item(), 1)]
        for lg, y, k, wgt in ((lc, yc, kc, a.w_car), (lb, yb, kb, a.w_ball)):
            w = (k & cm).float()
            ce = F.cross_entropy(lg.float().flatten(0, 1), y.flatten(), reduction="none").view(B, T)
            l_ = (ce * w).sum() / w.sum().clamp(min=1)
            loss = loss + wgt * l_
            parts.append(((lg.argmax(-1) == y).float() * w).sum().item() / max(w.sum().item(), 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        hist.append([loss.item()] + parts)
        if step % 500 == 0:
            h = np.mean(hist, 0)
            hist = []
            print("fold %s step %d loss %.3f acc poss %.3f car %.3f ball %.3f  %.1f s/100" % (
                a.fold, step, *h, (time.time() - t0) / 5), flush=True)
            t0 = time.time()
    torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(odir, "model.pt"))


def kin_ball(model, xb, committed, lam=10.0):
    B, T = xb.shape
    dev = xb.device
    vmax, slack = KIN_B
    pos = model.centres[xb.clamp(max=G - 1)]
    ar = torch.arange(T, device=dev)[None].expand(B, T)
    li = torch.where(committed, ar, torch.full_like(ar, -1))
    li = torch.cummax(torch.cat([torch.full((B, 1), -1, device=dev, dtype=li.dtype), li[:, :-1]], 1), 1).values
    ri = torch.where(committed, ar, torch.full_like(ar, T))
    ri = torch.flip(torch.cummin(torch.flip(torch.cat([ri[:, 1:], torch.full((B, 1), T, device=dev, dtype=ri.dtype)],
                                                      1), [1]), 1).values, [1])
    pen = torch.zeros(B, T, G, device=dev)
    for idx, val in ((li, li >= 0), (ri, ri < T)):
        dt = (ar - idx).abs().float()
        val = val & (dt <= 10)
        a = torch.gather(pos, 1, idx.clamp(0, T - 1)[..., None].expand(-1, -1, 2))
        d = torch.cdist(a.reshape(B * T, 1, 2), model.centres[None].expand(B * T, -1, -1)).view(B, T, -1)
        pen += torch.relu(d - (vmax * dt + slack)[..., None]) * val[..., None]
    return -lam * pen


SPELL_K = 5


def spell_prior(xp, committed, lam_sw, lam_d, k=SPELL_K):
    C = torch.tensor([[0.0, lam_sw, lam_d], [lam_sw, 0.0, lam_d], [lam_d, lam_d, 0.0]], device=xp.device)
    oh = F.one_hot(xp.clamp(max=2), 3).float() * (committed & (xp < 3))[..., None].float()
    pen = torch.zeros_like(oh)
    for s_ in range(1, k + 1):
        pen += shift_t(oh, s_, 0.0) + shift_t(oh, -s_, 0.0)
    return -(pen @ C) / k


@torch.no_grad()
def denoise_all(model, ev, x, flag, gen, N=8, tmax=0.8, tmin=0.4, kin_lam=10.0, lam_sw=0.0, lam_d=0.0):
    xp, xc, xb = x
    B, T = xp.shape
    dev = xp.device
    canvas = flag == 0
    nC = canvas.sum(-1, keepdim=True)
    nz = noise(ev["pv"], gen, B, T, dev)
    xs = [torch.where(canvas, z_, x_) for z_, x_ in zip(nz, (xp, xc, xb))]
    committed = [flag == 1, flag == 1, flag == 1]
    sc, first = None, None
    rank = torch.arange(T, device=dev)[None]
    for n in range(N, 0, -1):
        lp, lc, lb = model(ev, *xs, flag, sc)
        lb = lb.float()
        if kin_lam > 0:
            lb = lb + kin_ball(model, xs[2], committed[2], kin_lam)
        lp = lp.float()
        if lam_sw > 0 or lam_d > 0:
            lp = lp + spell_prior(xs[0], committed[0], lam_sw, lam_d)
        logits = [lp, lc.float(), lb]
        probs1 = [l.softmax(-1) for l in logits]
        if first is None:
            first = probs1
        tau = tmin + (tmax - tmin) * n / N
        frac = 1.0 - math.cos(math.pi / 2 * (N - n + 1) / N)
        new = []
        for i, l in enumerate(logits):
            pl = l / tau
            p = pl.softmax(-1)
            am = pl.argmax(-1)
            e = -(p * p.clamp(min=1e-12).log()).sum(-1)
            e = torch.where(canvas, e, torch.full_like(e, float("inf")))
            se, si = torch.sort(e, dim=-1, stable=True)
            acc_s = ((se.cumsum(-1) - se) <= 0.1) | (rank < torch.ceil(frac * nC))
            acc_s &= torch.isfinite(se)
            acc = torch.zeros_like(acc_s).scatter(1, si, acc_s)
            xa = torch.where(acc, am, xs[i])
            xr = torch.where(canvas & ~acc, nz[i] if n > 1 else xa, xa)
            new.append(xr)
            committed[i] = (acc & canvas) | (flag == 1)
        xs = new
        sc = tuple(l / (tmin + (tmax - tmin) * n / N) for l in logits)
        nz = noise(ev["pv"], gen, B, T, dev)
    return xs, probs1, first


def infer_view(model, v, dev, N=8, sweeps=2, batch=48, seed=0, lam_sw=0.0, lam_d=0.0):
    Tn = len(v["half"])
    can = C1 - C0
    toks = [np.full(Tn, MP, np.int64), np.full(Tn, MC, np.int64), np.full(Tn, MB, np.int64)]
    pp = np.full((Tn, 3), np.nan, np.float32)
    pp1 = np.full((Tn, 3), np.nan, np.float32)
    pnone = np.full(Tn, np.nan, np.float32)
    gen = torch.Generator(device=dev)
    for sw in range(sweeps):
        gen.manual_seed(seed * 1000 + sw)
        shift = can // 2 if sw % 2 else 0
        starts = np.arange(-shift, Tn, can)
        new = [t.copy() for t in toks]
        for i in range(0, len(starts), batch):
            g = starts[i:i + batch]
            ws = [window(v, s - C0) for s in g]
            pf = torch.from_numpy(np.stack([w[0] for w in ws])).to(dev)
            obs = torch.from_numpy(np.stack([w[1] for w in ws])).to(dev)
            valid = np.stack([w[2] for w in ws])
            xs, fl = [], np.full((len(g), T_WIN), 1, np.int64)
            for k, mk in enumerate((MP, MC, MB)):
                tw = np.full((len(g), T_WIN), mk, np.int64)
                for b, s in enumerate(g):
                    a0 = s - C0
                    lo, hi = max(a0, 0), min(a0 + T_WIN, Tn)
                    tw[b, lo - a0:hi - a0] = toks[k][lo:hi]
                xs.append(tw)
            fl[xs[0] == MP] = 2
            fl[:, C0:C1] = 0
            fl[:, C0:C1][~valid[:, C0:C1]] = 2
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ev = model.evidence(pf, obs)
                out, probs, first = denoise_all(model, ev, [torch.from_numpy(x).to(dev) for x in xs],
                                                torch.from_numpy(fl).to(dev), gen, N=N, lam_sw=lam_sw, lam_d=lam_d)
            out = [o.cpu().numpy() for o in out]
            pr = probs[0].float().cpu().numpy()
            p1 = first[0].float().cpu().numpy()
            pn = probs[1][..., NONE].float().cpu().numpy()
            for b, s in enumerate(g):
                fr = np.arange(s, s + can)
                ok = valid[b, C0:C1]
                for k in range(3):
                    new[k][fr[ok]] = out[k][b, C0:C1][ok]
                if sw == sweeps - 1:
                    pp[fr[ok]] = pr[b, C0:C1][ok]
                    pnone[fr[ok]] = pn[b, C0:C1][ok]
                if sw == 0:
                    pp1[fr[ok]] = p1[b, C0:C1][ok]
        toks = new
    return dict(p_poss=pp, p_poss1=pp1, p_none=pnone, tok_poss=toks[0], tok_car=toks[1], tok_ball=toks[2])


def post_path(name, seed, fold):
    return os.path.join(POST, "%s_s%d_F%s.npz" % (name, seed, fold))


def load_model(name, fold, dev):
    st = torch.load(os.path.join(RUNS, name, "fold_%s" % fold, "model.pt"), map_location=dev, weights_only=False)
    ar = st["args"]
    m = Unified(d=ar["d"], layers=ar["layers"], heads=ar["heads"], ff=4 * ar["d"]).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    return m


def dev_views_of_fold(fold):
    from mtah import gs_build as B, eval_ext as E
    s, _ = E.load_split(U.SPLIT)
    Gp = B.groups()
    return [(n, B.eval_seed(n)) for n in s["dev"] if Gp[n] == str(fold)]


def cmd_infer(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    model = load_model(a.name, a.fold, dev)
    os.makedirs(POST, exist_ok=True)
    for i, (n, seed) in enumerate(dev_views_of_fold(a.fold)):
        if not os.path.exists(U.path(n, seed)):
            print("missing view", n, flush=True)
            continue
        out = post_path(n, seed, a.fold).replace(".npz", "_%s%s.npz" % (a.name, a.tag))
        if os.path.exists(out):
            continue
        v = load_view(n, seed)
        r = infer_view(model, v, dev, N=a.steps, seed=i, lam_sw=a.lam_sw, lam_d=a.lam_d)
        N = int(v["n_frames"])
        up = lambda x: np.repeat(x, U.STEP, axis=0)[:N]
        np.savez_compressed(out, p_poss=up(r["p_poss"]), p_poss1=up(r["p_poss1"]), p_none=up(r["p_none"]),
                            tok_poss=r["tok_poss"], tok_car=r["tok_car"], tok_ball=r["tok_ball"])
        print(n, "done", flush=True)


def cmd_heldout(a):
    from mtah import gs_build as B, eval_ext as E
    s, _ = E.load_split(U.SPLIT)
    Gp = B.groups()
    rows = {}
    for n in s["dev"]:
        seed = B.eval_seed(n)
        p = post_path(n, seed, Gp[n]).replace(".npz", "_%s%s.npz" % (a.name, a.tag))
        if not os.path.exists(p):
            continue
        z = np.load(p)
        v = load_view(n, seed)
        src = E.src_of(n)
        r = rows.setdefault(src, {"poss_n": 0, "poss_ok": 0, "car_n": 0, "car_ok": 0, "ball_err": []})
        yp, kp, yc, kc, yb, kb = tokens_of(v)
        tp = z["tok_poss"]
        r["poss_n"] += int(kp.sum())
        r["poss_ok"] += int((tp[kp] == yp[kp]).sum())
        tc = z["tok_car"]
        r["car_n"] += int(kc.sum())
        r["car_ok"] += int((tc[kc] == yc[kc]).sum())
        if kb.any():
            bxy = cell_centres().numpy()[np.clip(z["tok_ball"], 0, G - 1)]
            r["ball_err"] += np.hypot(*(bxy[kb] - v["y_ball"][kb]).T).tolist()
    out = {}
    for src, r in rows.items():
        be = np.array(r["ball_err"])
        out[src] = {"poss_acc": r["poss_ok"] / max(r["poss_n"], 1), "carrier_acc": r["car_ok"] / max(r["car_n"], 1),
                    "carrier_frames": r["car_n"],
                    "ball_rec3": float(np.mean(be < 3)) if len(be) else None,
                    "ball_med": float(np.median(be)) if len(be) else None}
    print("HELDOUT", json.dumps(out))
    json.dump(out, open(os.path.join(RUNS, a.name, "heldout%s.json" % a.tag), "w"), indent=1)


def cmd_score(a):
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    from mtah import gs_build as B, eval_ext as E
    s, _ = E.load_split(U.SPLIT)
    Gp = B.groups()
    per = {}
    for n in s["dev"]:
        p = post_path(n, B.eval_seed(n), Gp[n]).replace(".npz", "_%s%s.npz" % (a.name, a.tag))
        z = np.load(p)
        N = len(z["p_poss"])
        pred = np.repeat(z["tok_poss"], U.STEP)[:N].astype(np.int8)
        y, half = E.truth(n, E.label_key(n, s))
        pred = np.where(half > 0, pred, 3).astype(np.int8)
        per[n] = E.score(n, pred, s)
    by = {}
    for n in per:
        by.setdefault(E.src_of(n), []).append(n)
    bs = {k: {"n": len(v), "spell_f1": float(np.mean([per[n]["spell_f1"] for n in v])),
              "macro_f1": float(np.mean([per[n]["macro_f1"] for n in v]))} for k, v in by.items()}
    out = {"balanced": float(np.mean([v["spell_f1"] for v in bs.values()])), "by_source": bs,
           "fc": float(np.mean([bs[k]["spell_f1"] for k in ("idsse-ctrl", "stv2") if k in bs])),
           "per_match": {n: {"spell_f1": per[n]["spell_f1"], "macro_f1": per[n]["macro_f1"]} for n in per}}
    print("STANDALONE %s%s balanced %.4f fc-mean %.4f %s" % (a.name, a.tag, out["balanced"], out["fc"],
                                                            {k: round(v["spell_f1"], 4) for k, v in bs.items()}))
    json.dump(out, open(os.path.join(RUNS, a.name, "standalone%s.json" % a.tag), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--name", default="ud1")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--w-car", type=float, default=0.3)
    ap.add_argument("--w-ball", type=float, default=0.2)
    ap.add_argument("--lam-sw", type=float, default=0.0)
    ap.add_argument("--lam-d", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "infer":
        if a.steps == 6000:
            a.steps = 8
        cmd_infer(a)
    elif a.cmd == "heldout":
        cmd_heldout(a)
    elif a.cmd == "score":
        cmd_score(a)


if __name__ == "__main__":
    main()
