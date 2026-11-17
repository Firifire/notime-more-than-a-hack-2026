import hashlib
import inspect
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

POOL = 5
L = 900
NH = 12
NT = 2 * NH + 1
FT = 14
SWAP_NEG = (0, 1, 2, 3, 6, 7)
MIRR_NEG = (1, 3, 7)
HEADS = ("dead", "team", "sw", "on", "off")


def _ext_dir():
    from mtah.eval_ext import EXT
    return EXT


def load_match(name):
    from mtah.train_ext import mask_ball
    z = np.load(os.path.join(_ext_dir(), name + ".npz"))
    m = {k: z[k] for k in z.files}
    m["name"] = name
    return mask_ball(m)


def _vel(p):
    v = np.full_like(p, np.nan)
    v[2:-2] = (p[4:] - p[:-4]) * (25.0 / 4.0)
    return v


def _blocks(a, nb):
    n = len(a)
    if nb * POOL > n:
        a = np.concatenate([a, np.repeat(a[-1:], nb * POOL - n, 0)])
    return a.reshape((nb, POOL) + a.shape[1:])


def tokens_from_match(d):
    from mtah.featset import normalise
    import warnings
    Lp, Wp = float(d["pitch"][0]), float(d["pitch"][1])
    B, P = normalise(d)
    team = np.asarray(d["team"])
    half = d["half"]
    segs = []
    scale = np.array([Lp / 2, Wp / 2])
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        a, n = int(idx[0]), len(idx)
        assert idx[-1] - a + 1 == n, "half not contiguous"
        nb = -(-n // POOL)
        b, p = B[a:a + n], P[a:a + n]
        vb, vp = _vel(b), _vel(p)
        acc = np.full(n, np.nan)
        acc[1:-1] = np.linalg.norm(vb[2:] - vb[:-2], axis=-1) * 12.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pb, pp = _blocks(b, nb), _blocks(p, nb)
            Bm, Pm = np.nanmean(pb, 1), np.nanmean(pp, 1)
            Vb, Vp = np.nanmean(_blocks(vb, nb), 1), np.nanmean(_blocks(vp, nb), 1)
            presb = (~np.isnan(pb[..., 0])).mean(1)
            presp = (~np.isnan(pp[..., 0])).mean(1)
            accb = np.nanmax(_blocks(acc, nb), 1)
        tok = np.zeros((nb, NT, FT), np.float32)
        msk = np.zeros((nb, NT), bool)
        bok = presb > 0
        for g in (0, 1):
            sl = np.flatnonzero(team == g)
            pr = presp[:, sl]
            order = np.argsort(-pr, axis=1, kind="stable")[:, :NH]
            k = order.shape[1]
            rows = np.arange(nb)[:, None]
            sel = sl[order]
            pos, vel, pres = Pm[rows, sel], Vp[rows, sel], presp[rows, sel]
            ok = pres > 0
            off = g * NH
            t = tok[:, off:off + k]
            t[..., 0:2] = pos / scale
            t[..., 2:4] = vel / 8.0
            t[..., 4] = np.linalg.norm(vel, axis=-1) / 8.0
            t[..., 5] = pres
            rel = pos - Bm[:, None, :]
            t[..., 6:8] = rel / 20.0
            t[..., 8] = np.linalg.norm(rel, axis=-1) / 20.0
            t[..., 9] = bok[:, None]
            t[..., 10 + g] = 1.0
            msk[:, off:off + k] = ok
        tb = tok[:, -1]
        tb[:, 0:2] = Bm / scale
        tb[:, 2:4] = Vb / 8.0
        tb[:, 4] = np.linalg.norm(Vb, axis=-1) / 8.0
        tb[:, 5] = presb
        tb[:, 12] = 1.0
        tb[:, 13] = accb / 50.0
        msk[:, -1] = True
        tok[~msk] = 0.0
        rel_ok = msk[:, :-1] & bok[:, None]
        tok[:, :-1, 6:10] *= rel_ok[..., None]
        tok = np.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(tok, -5, 5, out=tok)
        segs.append((int(h), a, n, tok.astype(np.float16), msk))
    return segs


def view_hash():
    from mtah.featcache import code_hash
    src = inspect.getsource(tokens_from_match) + inspect.getsource(_vel) + inspect.getsource(_blocks) + \
        inspect.getsource(load_match) + "NT%d FT%d NH%d POOL%d" % (NT, FT, NH, POOL)
    return hashlib.sha1((src + code_hash()).encode()).hexdigest()[:10]


def _profile_for(seed):
    from mtah.degrade_profiles import active_profile, COMPLIANT
    prof = active_profile(seed)
    if prof not in COMPLIANT:
        raise RuntimeError("raw_seq builds views only for compliant degradation profiles (got %s)" % prof)
    return prof


def view_path(name, seed):
    from mtah.data import CACHE
    prof = _profile_for(seed)
    return os.path.join(CACHE, "raw5", "%s_p-%s" % (view_hash(), prof), "%s_s%d.npz" % (name, seed))


def _compute_view(args):
    name, seed, path = args
    if os.path.exists(path):
        return path
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from mtah.train_ext import degrade
    m = load_match(name)
    d = degrade(m, np.random.default_rng(seed), None, None, profile=_profile_for(seed))
    segs = tokens_from_match(d)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, half=m["half"], seg=np.array([s[:3] for s in segs]),
                        **{"T%d" % i: s[3] for i, s in enumerate(segs)}, **{"M%d" % i: s[4] for i, s in enumerate(segs)})
    os.replace(tmp, path)
    return path


def ensure_views(keys, workers=4):
    from concurrent.futures import ProcessPoolExecutor
    todo = [(n, s, view_path(n, s)) for n, s in keys]
    miss = [t for t in todo if not os.path.exists(t[2])]
    if miss:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_compute_view, miss))
        else:
            for t in miss:
                _compute_view(t)
    return {(n, s): p for n, s, p in todo}


def load_view(path):
    z = np.load(path)
    return [(int(h), int(a), int(n), z["T%d" % i], z["M%d" % i]) for i, (h, a, n) in enumerate(z["seg"])], z["half"]


def spell_weight_full(lab, half, o):
    lab, half = lab.astype(np.int64), half.astype(np.int64)
    key = lab + 10 * half
    ch = np.flatnonzero(key[1:] != key[:-1]) + 1
    st = np.r_[0, ch]
    ln = np.diff(np.r_[st, len(key)])
    Ln = np.repeat(ln, ln).astype(np.float64)
    w = np.clip((float(o.get("ref", 250)) / Ln) ** float(o.get("alpha", 0.5)), float(o.get("min", 0.5)),
                float(o.get("max", 3.0)))
    w[lab >= 2] = 1.0
    return w


def targets(y, half, spell_w=None):
    from mtah.train_v5 import boundary_labels
    bsw, bon, boff = boundary_labels(y, half)
    w = spell_weight_full(y, half, spell_w) if spell_w else np.ones(len(y))
    return np.stack([y.astype(np.float32), bsw, bon, boff, w.astype(np.float32)], 0)


def build_train(items):
    import torch
    out = []
    for it in items:
        segs, tgt, dsrc, tsrc = it[:4]
        bsrc = bool(it[4]) if len(it) > 4 else False
        for h, a, n, T, M in segs:
            out.append(dict(x=torch.from_numpy(np.ascontiguousarray(T)), m=torch.from_numpy(np.ascontiguousarray(M)),
                            t=torch.from_numpy(np.ascontiguousarray(tgt[:, a:a + n]).astype(np.float16)),
                            n=n, nb=len(T), dead=dsrc, team=tsrc, blind=bsrc and not dsrc))
    return out


def make_model(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class MHA(nn.Module):
        def __init__(self, d, h, drop, rope=False):
            super().__init__()
            self.h, self.dk, self.rope = h, d // h, rope
            self.qkv = nn.Linear(d, 3 * d)
            self.o = nn.Linear(d, d)
            self.drop = nn.Dropout(drop)

        def forward(self, x, key_mask=None, cs=None):
            Bn, T, D = x.shape
            q, k, v = self.qkv(x).view(Bn, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
            if self.rope:
                cos, sin = cs
                q = q * cos + _rot(q) * sin
                k = k * cos + _rot(k) * sin
            s = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)
            if key_mask is not None:
                s = s.masked_fill(~key_mask[:, None, None, :], -1e4)
            att = self.drop(torch.softmax(s, -1))
            return self.o((att @ v).transpose(1, 2).reshape(Bn, T, D))

    def _rot(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], -1)

    class Block(nn.Module):
        def __init__(self, d, h, ff, drop, rope=False):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.att = MHA(d, h, drop, rope)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, key_mask=None, cs=None):
            x = x + self.drop(self.att(self.n1(x), key_mask, cs))
            return x + self.drop(self.ff(self.n2(x)))

    class RawNet(nn.Module):
        def __init__(self):
            super().__init__()
            dt, d = a.tok_dim, a.dim
            self.tin = nn.Sequential(nn.Linear(FT, dt), nn.GELU(), nn.Linear(dt, dt))
            self.tblocks = nn.ModuleList([Block(dt, a.tok_heads, 2 * dt, a.drop) for _ in range(a.tok_layers)])
            self.tnorm = nn.LayerNorm(dt)
            self.fproj = nn.Sequential(nn.Linear(5 * dt + 2, d), nn.GELU(), nn.LayerNorm(d))
            self.in_drop = nn.Dropout(a.in_drop)
            self.conv = nn.Conv1d(d, d, 5, padding=2)
            self.blocks = nn.ModuleList([Block(d, a.heads, 2 * d, a.drop, rope=True) for _ in range(a.layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, len(HEADS))
            dk = d // a.heads
            inv = 1.0 / (10000 ** (torch.arange(0, dk // 2).float() / (dk // 2)))
            self.register_buffer("inv", inv, persistent=False)

        def encode(self, x, m):
            Bn, T = x.shape[:2]
            t = x.reshape(Bn * T, NT, FT)
            mm = m.reshape(Bn * T, NT)
            h = self.tin(t)
            for blk in self.tblocks:
                h = blk(h, mm)
            h = self.tnorm(h)
            mf = mm.unsqueeze(-1).to(h.dtype)
            outs = []
            for g in (10, 11):
                sel = mf * t[..., g:g + 1].to(h.dtype)
                cnt = sel.sum(1)
                mean = (h * sel).sum(1) / cnt.clamp(min=1)
                mx = h.masked_fill(sel == 0, -1e4).amax(1)
                mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
                outs += [mean, mx]
            cnts = torch.stack([(mf[..., 0] * t[..., 10]).sum(1), (mf[..., 0] * t[..., 11]).sum(1)], -1) / 11.0
            f = torch.cat(outs + [h[:, -1], cnts.to(h.dtype)], -1)
            return self.fproj(f).view(Bn, T, -1)

        def forward(self, x, m):
            f = self.in_drop(self.encode(x, m))
            f = f + Fn.gelu(self.conv(f.transpose(1, 2))).transpose(1, 2)
            T = f.shape[1]
            pos = torch.arange(T, device=f.device, dtype=torch.float32)
            ang = pos[:, None] * self.inv[None]
            cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None].to(f.dtype)
            sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None].to(f.dtype)
            for blk in self.blocks:
                f = blk(f, None, (cos, sin))
            return self.head(self.norm(f))
    return RawNet()


class Args:
    tok_dim = 64
    tok_heads = 4
    tok_layers = 2
    dim = 128
    heads = 4
    layers = 4
    drop = 0.1
    in_drop = 0.1
    iters = 4000
    batch = 16
    lr = 6e-4
    wd = 0.05
    p_dead = 0.35
    tmse = 0.1
    swap_p = 0.5
    mirror_p = 0.5
    bf16 = True
    window = L
    crf = 0.0
    blind_share = 0.0
    blind_sources = ("pff",)
    spell_w = {"alpha": 0.5, "ref": 250, "min": 0.5, "max": 3.0}

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(Args, k):
                raise KeyError("unknown raw_seq arg %s" % k)
            setattr(self, k, v)

    def dict(self):
        return {k: getattr(self, k) for k in dir(Args) if not k.startswith("_") and not callable(getattr(Args, k))}


def set_determinism(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _augment(x, t, swap, mirror, blind=None):
    import torch
    if blind is not None and bool(blind.any()):
        keep = (~blind).to(x.dtype)[:, None, None]
        x = x.clone()
        x[:, :, -1, :12] = x[:, :, -1, :12] * keep
        x[:, :, -1, 13] = x[:, :, -1, 13] * keep[..., 0]
        x[:, :, :-1, 6:10] = x[:, :, :-1, 6:10] * keep[..., None]
    sgn = torch.ones(x.shape[0], FT, device=x.device, dtype=x.dtype)
    sgn[:, list(SWAP_NEG)] = torch.where(swap[:, None], -1.0, 1.0).to(x.dtype)
    ms = torch.where(mirror[:, None], -1.0, 1.0).to(x.dtype)
    sgn[:, list(MIRR_NEG)] = sgn[:, list(MIRR_NEG)] * ms
    x = x * sgn[:, None, None, :]
    hcol, acol = x[..., 10].clone(), x[..., 11].clone()
    sw = swap[:, None, None]
    x[..., 10] = torch.where(sw, acol, hcol)
    x[..., 11] = torch.where(sw, hcol, acol)
    y = t[:, 0]
    ys = torch.where(y == 0, torch.ones_like(y), torch.where(y == 1, torch.zeros_like(y), y))
    t = t.clone()
    t[:, 0] = torch.where(swap[:, None], ys, y)
    return x, t


def sample_batch(segs, rng, B, p_dead, swap_p, mirror_p, blind_share=0.0, L=L):
    import torch
    dead_ix = [i for i, s in enumerate(segs) if s["dead"]]
    blind_ix = [i for i, s in enumerate(segs) if s.get("blind")]
    w = np.array([s["nb"] for s in segs], float) * np.array([s["team"] for s in segs])
    w /= w.sum()
    xs, ms, ts, fl = [], [], [], []
    for b in range(B):
        blind = False
        if rng.random() < p_dead:
            if blind_share > 0 and blind_ix and rng.random() < blind_share:
                i, blind = blind_ix[rng.integers(len(blind_ix))], True
            else:
                i = dead_ix[rng.integers(len(dead_ix))]
        else:
            i = int(rng.choice(len(segs), p=w))
        s = segs[i]
        st = int(rng.integers(0, max(s["nb"] - L, 0) + 1))
        x, m = s["x"][st:st + L], s["m"][st:st + L]
        t = s["t"][:, st * POOL:(st + L) * POOL]
        if x.shape[0] < L:
            k = L - x.shape[0]
            x = torch.cat([x, torch.zeros((k, NT, FT), dtype=x.dtype)], 0)
            mp = torch.zeros((k, NT), dtype=torch.bool)
            mp[:, -1] = True
            m = torch.cat([m, mp], 0)
        if t.shape[1] < L * POOL:
            pad = torch.zeros(5, L * POOL - t.shape[1], dtype=t.dtype)
            pad[0] = 3
            t = torch.cat([t, pad], 1)
        xs.append(x)
        ms.append(m)
        ts.append(t)
        sw_, mi_ = rng.random() < swap_p, rng.random() < mirror_p
        fl.append((s["dead"] or blind, s["team"] and not blind, sw_, mi_, blind))
    f = torch.tensor(fl, dtype=torch.bool)
    return torch.stack(xs), torch.stack(ms), torch.stack(ts), f


def loss_fn(z5, t, f, U, lam):
    import torch
    import torch.nn.functional as Fn
    z = torch.einsum("blc,lt->btc", z5.float(), U)
    y = t[:, 0]
    valid = y < 2.5
    alive = y < 1.5
    md = (valid & f[:, 0:1]).float()
    mt = (alive & f[:, 1:2]).float()
    wt = mt * t[:, 4]

    def bce(zz, tt, mask):
        l = Fn.binary_cross_entropy_with_logits(zz, tt, reduction="none")
        return (l * mask).sum() / mask.sum().clamp(min=1)
    parts = {"dead": bce(z[..., 0], (y > 1.5).float(), md),
             "team": bce(z[..., 1], (y < 0.5).float(), wt),
             "sw": bce(z[..., 2], t[:, 1], wt),
             "on": bce(z[..., 3], t[:, 2], md),
             "off": bce(z[..., 4], t[:, 3], md)}
    zz = z5[..., :2].float()
    lp = torch.stack([Fn.logsigmoid(zz), Fn.logsigmoid(-zz)], -1)
    d = (lp[:, 1:] - lp[:, :-1].detach()) ** 2
    parts["tmse"] = d.clamp(max=16).mean()
    total = sum(v for k, v in parts.items() if k != "tmse") + lam * parts["tmse"]
    return total, parts


def crf_nll(z5, t, f, prm):
    import torch
    import torch.nn.functional as Fn
    z = z5.float()
    zd, zt, zs, zon, zoff = z.unbind(-1)
    e = torch.stack([Fn.logsigmoid(-zd) + Fn.logsigmoid(zt), Fn.logsigmoid(-zd) + Fn.logsigmoid(-zt),
                     Fn.logsigmoid(zd)], -1)
    Bn, T = zd.shape
    ls = prm[0] + prm[3] * zs
    lon = prm[1] + prm[3] * zon
    loff = prm[2] + prm[3] * zoff
    zero = torch.zeros_like(ls)
    tr = torch.stack([torch.stack([zero, ls, lon], -1), torch.stack([ls, zero, lon], -1),
                      torch.stack([loff, loff, zero], -1)], -2)
    y5 = t[:, 0, POOL // 2::POOL][:, :T]
    oh = Fn.one_hot(y5.clamp(max=2).long(), 3).bool()
    free = (y5 > 2.5)[..., None]
    dead_w = f[:, 0][:, None, None]
    alive = (y5 < 1.5)[..., None]
    team_ok = oh | (alive & torch.tensor([False, False, True], device=z.device)) | (~alive)
    allow = free | torch.where(dead_w, oh, team_ok)
    neg = torch.full_like(e, -1e4)
    ec = torch.where(allow, e, neg)
    def logz(em):
        M = tr[:, 1:] + em[:, 1:, None, :]
        eye = torch.full((3, 3), -1e4, device=z.device)
        eye.fill_diagonal_(0.0)
        while M.shape[1] > 1:
            if M.shape[1] % 2:
                M = torch.cat([M, eye.expand(Bn, 1, 3, 3)], 1)
            M = torch.logsumexp(M[:, 0::2, :, :, None] + M[:, 1::2, None, :, :], 3)
        return torch.logsumexp(em[:, 0, :, None] + M[:, 0], (1, 2))
    nll = logz(e) - logz(ec)
    return nll.mean() / T


def train_model(segs, a, seed, log=print, val=None):
    import torch
    from mtah.nn_seq import upsample_matrix
    set_determinism(seed)
    dev = torch.device("cuda")
    model = make_model(a).to(dev)
    if a.crf:
        model.crf_prm = torch.nn.Parameter(torch.tensor([-2.0, -6.0, -6.0, 1.0], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
    U = torch.tensor(upsample_matrix(a.window, a.window * POOL), device=dev)
    rng = np.random.default_rng(seed)
    t0, run = time.time(), {}
    every = max(1, a.iters // 8)
    for it in range(a.iters):
        model.train()
        x, m, t, f = sample_batch(segs, rng, a.batch, a.p_dead, a.swap_p, a.mirror_p, a.blind_share, a.window)
        x, m, t, f = x.to(dev).float(), m.to(dev), t.to(dev).float(), f.to(dev)
        x, t = _augment(x, t, f[:, 2], f[:, 3], f[:, 4] if a.blind_share > 0 else None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.bf16)):
            z5 = model(x, m)
        loss, parts = loss_fn(z5, t, f, U, a.tmse)
        if a.crf:
            parts["crf"] = crf_nll(z5, t, f, model.crf_prm)
            loss = loss + a.crf * parts["crf"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run[k] = run.get(k, 0) + v.detach()
        if (it + 1) % every == 0:
            log("    raw it %d %.0fs %s" % (it + 1, time.time() - t0,
                                           " ".join("%s %.4f" % (k, float(v) / every) for k, v in run.items())))
            run = {}
            if val is not None and it + 1 < a.iters:
                val(model, it + 1)
    model.eval()
    return model


def _variant(T, swap, mirror):
    T = T.astype(np.float32).copy()
    if swap:
        T[..., list(SWAP_NEG)] *= -1
        T[..., [10, 11]] = T[..., [11, 10]]
    if mirror:
        T[..., list(MIRR_NEG)] *= -1
    return T


def predict_view(model, segs, N, bf16=True, tta=((False, False), (True, False), (False, True), (True, True)),
                 bs=8, L=L):
    import torch
    from mtah.nn_seq import upsample
    model.eval()
    dev = next(model.parameters()).device
    out = {h: np.full(N, 0.5 if h in ("dead", "team") else 0.0, np.float64) for h in HEADS}
    for h, a, n, T, M in segs:
        nb = len(T)
        starts = list(range(0, max(nb - L, 0) + 1, L // 2))
        if starts[-1] + L < nb:
            starts.append(nb - L)
        acc_p = np.zeros((n, len(HEADS)))
        for swap, mirror in tta:
            Tv = _variant(T, swap, mirror)
            acc = np.zeros((nb, len(HEADS)))
            wsum = np.zeros(nb)
            for i in range(0, len(starts), bs):
                chunk = starts[i:i + bs]
                xs, ms, ks = [], [], []
                for st in chunk:
                    x, m = Tv[st:st + L], M[st:st + L]
                    k = len(x)
                    if k < L:
                        x = np.concatenate([x, np.zeros((L - k, NT, FT), np.float32)])
                        mp = np.zeros((L - k, NT), bool)
                        mp[:, -1] = True
                        m = np.concatenate([m, mp])
                    xs.append(x)
                    ms.append(m)
                    ks.append(k)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(bf16)):
                    z = model(torch.tensor(np.stack(xs), device=dev), torch.tensor(np.stack(ms), device=dev))
                z = z.float().cpu().numpy().astype(np.float64)
                for st, k, zz in zip(chunk, ks, z):
                    w = np.minimum(np.arange(1, k + 1), np.arange(k, 0, -1)).astype(float)
                    acc[st:st + k] += zz[:k] * w[:, None]
                    wsum[st:st + k] += w
            z25 = upsample(acc / wsum[:, None], n)
            p = 1 / (1 + np.exp(-z25))
            if swap:
                p[:, 1] = 1 - p[:, 1]
            acc_p += p
        acc_p /= len(tta)
        for j, hd in enumerate(HEADS):
            out[hd][a:a + n] = acc_p[:, j]
    return out
