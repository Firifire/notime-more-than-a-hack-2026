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
from vision.refine.model import SelfConditioning, mlp, fourier, NF as NFOUR  # noqa: E402
from vision.refine import ud_data as U  # noqa: E402
from vision.refine.unified import T_WIN, C0, C1, MP, spell_prior, POST  # noqa: E402

ROOT = U.ROOT
OOF = os.environ.get("MTAH_PDC_OOF") or os.path.join(ROOT, "runs", "eval", "gsraw_r3_v4_oof")
BASE = os.path.join(ROOT, "runs", "eval", "gsraw_r3_d383rg_v4.json")
PACK = os.environ.get("MTAH_PDC_PACK") or os.path.join(ROOT, "cache", "ud", "corr")
PACK_KEYS = ("heads", "head_keys", "dec", "y", "half", "frame", "dead_ok", "n_frames")
RUNS = os.path.join(ROOT, "cache", "ud", "runs_corr")
TAC = "/mnt/x/Data/AI/more-than-a-hack/cache/tacvid"
HEAD_KEYS = None
NV = 7
NVD = 128
_RP = (np.random.default_rng(20261113).normal(size=(768, NVD)) / np.sqrt(768)).astype(np.float32)


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def dev_names():
    from mtah import eval_ext as E
    s, _ = E.load_split(U.SPLIT)
    return sorted(s["dev"])


def video_tokens(n, frames):
    mid = n.split("_")[1]
    T = len(frames)
    V = np.zeros((T, NV, NVD), np.float16)
    ok = np.zeros(T, bool)
    for hh in ("1st", "2nd"):
        f = os.path.join(TAC, "stv2_%s_%s.npz" % (mid, hh))
        if not os.path.exists(f):
            continue
        z = np.load(f)
        idx = z["npz_index"]
        good = idx >= 0
        if not good.any():
            continue
        o = np.argsort(idx[good])
        si = idx[good][o]
        tok = (np.concatenate([z["glob"], z["crop"]], 1)[good][o].astype(np.float32) @ _RP).astype(np.float16)
        j = np.clip(np.searchsorted(si, frames), 1, len(si) - 1)
        jj = np.where(np.abs(si[j - 1] - frames) < np.abs(si[j] - frames), j - 1, j)
        near = np.abs(si[jj] - frames) <= 3
        V[near] = tok[jj[near]]
        ok |= near
    return V, ok


def cmd_pack(a):
    pack_all(getattr(a, "config", None), video=bool(getattr(a, "video", 0)))


def pack_all(config=None, video=False):
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    from mtah import eval_ext as E, gs_build as B
    import importlib
    if config:
        cfg = json.load(open(os.path.join(ROOT, config)))
        pipe = importlib.import_module("mtah.pipe_gsraw")
        dec = cfg["lgb"]["pipe_pdc"]["decoder"]
    else:
        base = json.load(open(BASE))
        pipe = importlib.import_module(base["config"]["pipeline"])
        dec = base["config"]["decoder"]
    s, _ = E.load_split(U.SPLIT)
    os.makedirs(PACK, exist_ok=True)
    for n in dev_names():
        out = os.path.join(PACK, n + ".npz")
        if os.path.exists(out):
            continue
        z = np.load(os.path.join(OOF, "oof_%s.npz" % n))
        keys = sorted(k for k in z.files if k not in ("label", "half"))
        p = {k: z[k].astype(np.float64) for k in keys}
        y, half = E.truth(n, E.label_key(n, s))
        decd = pipe.decode_heads(dict(p), half, dec)
        v = np.load(U.path(n, B.eval_seed(n)))
        fr = v["frame"]
        N = len(y)
        blk = lambda x: np.stack([x[np.clip(fr + k, 0, N - 1)] for k in range(U.STEP)]).mean(0)
        H = np.stack([blk(_logit(p[k])) for k in keys], 1).astype(np.float16)
        extra = {}
        if video and E.src_of(n) == "stv2":
            V, ok = video_tokens(n, fr)
            extra = dict(vid=V, vid_ok=ok)
        np.savez_compressed(out, heads=H, head_keys=np.array(keys), dec=decd[fr].astype(np.int8),
                            y=np.where(y[fr] <= 2, y[fr], -1).astype(np.int8), half=half[fr].astype(np.int8),
                            frame=fr, dead_ok=E.src_of(n) in U.DEAD_SOURCES, n_frames=N, **extra)
        print(n, H.shape, "video" if extra else "", flush=True)


class Corrector(nn.Module):
    def __init__(self, nh, d=192, layers=6, heads=6, ff=768, dropout=0.1, video=False):
        super().__init__()
        self.d, self.video = d, video
        self.h_enc = mlp(nh, d, d)
        self.p_enc = mlp(U.NF + NFOUR, d, d)
        self.team_pool = nn.Linear(4 * d, d)
        self.obs_enc = mlp(NFOUR + 1, d, d)
        if video:
            self.v_proj = nn.Linear(NVD, d)
            self.v_mix = mlp(NV * d, d, d)
            self.v_null = nn.Parameter(torch.zeros(d))
        self.ev_norm = nn.LayerNorm(d)
        self.tok = nn.Embedding(4, d)
        self.sc = SelfConditioning(d, ff)
        self.flag = nn.Embedding(3, d)
        self.pos = nn.Embedding(T_WIN, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))

    def evidence(self, H, pf, obs, vid=None, vok=None):
        pv = pf[..., U.NF - 6] != 0
        x = torch.cat([pf, fourier(pf[..., U.NF - 5:U.NF - 3]) * pv[..., None]], -1)
        h = self.p_enc(x) * pv[..., None]
        sg = pf[..., U.NF - 6]
        pools = []
        for side in (1.0, -1.0):
            m = sg == side
            cnt = m.sum(2, keepdim=True).clamp(min=1)
            pools += [(h * m[..., None]).sum(2) / cnt, h.masked_fill(~m[..., None], -1e4).max(2).values.clamp(min=-50)]
        e = self.team_pool(torch.cat(pools, -1)) + self.h_enc(H)
        ob = ~torch.isnan(obs[..., 0])
        e = e + self.obs_enc(torch.cat([fourier(torch.nan_to_num(obs)) * ob[..., None], ob[..., None].float()], -1))
        if self.video:
            ve = self.v_mix(self.v_proj(vid.float()).flatten(2))
            e = e + torch.where(vok[..., None], ve, self.v_null.expand_as(ve))
        return self.ev_norm(e)

    def forward(self, e, x, flag, sc=None, sc_on=None):
        emb = self.tok(x)
        if sc is None:
            soft = torch.zeros_like(emb)
        else:
            soft = sc.float().softmax(-1) @ self.tok.weight[:3]
            if sc_on is not None:
                soft = soft * sc_on[:, None, None].float()
        h = self.tf(self.sc(emb, soft) + e + self.flag(flag) + self.pos.weight[:x.shape[1]])
        return self.out(h)


MM = os.environ.get("MTAH_PDC_MM") or os.path.join(ROOT, "cache", "ud", "corr_mm")


def load_pack(n):
    from mtah import gs_build as B
    d_ = os.path.join(MM, n)
    done = os.path.join(d_, "_ok")
    if not os.path.exists(done):
        os.makedirs(d_, exist_ok=True)
        z = np.load(os.path.join(PACK, n + ".npz"))
        arrs = {k: z[k] for k in z.files}
        v = np.load(U.path(n, B.eval_seed(n)))
        arrs["pf"], arrs["obs"] = v["pf"], v["obs"]
        for k, x in arrs.items():
            np.save(os.path.join(d_, k + ".npy"), np.asarray(x))
        open(done, "w").close()
    out = {}
    for f in os.listdir(d_):
        if f.endswith(".npy"):
            k = f[:-4]
            x = np.load(os.path.join(d_, f), mmap_mode="r" if k in ("pf", "heads", "vid") else None)
            out[k] = x if x.ndim else x.item() if k != "head_keys" else x
    return out


def margin(H, keys):
    kt, kd = list(keys).index("team"), list(keys).index("dead")
    pt = 1 / (1 + np.exp(-H[:, kt].astype(np.float32)))
    pd = 1 / (1 + np.exp(-H[:, kd].astype(np.float32)))
    return np.minimum(np.abs(pt - 0.5), np.abs(pd - 0.5))


def remask_mask(m, dec, frac, near=5):
    T = len(m)
    thr = np.quantile(m, frac) if frac > 0 else -1
    mk = m <= thr
    ch = np.flatnonzero(np.diff(dec.astype(np.int16)) != 0)
    for c in ch:
        mk[max(0, c - near + 1):c + near + 1] = True
    return mk


def window(d, s0, rng=None, video=False, frac=0.2):
    T = len(d["y"])
    lo, hi = max(s0, 0), min(s0 + T_WIN, T)
    sl = slice(lo - s0, hi - s0)
    nh = d["heads"].shape[1]
    H = np.zeros((T_WIN, nh), np.float32)
    pf = np.zeros((T_WIN, U.NP, U.NF), np.float32)
    obs = np.full((T_WIN, 2), np.nan, np.float32)
    dec = np.full(T_WIN, MP, np.int64)
    y = np.zeros(T_WIN, np.int64)
    ky = np.zeros(T_WIN, bool)
    valid = np.zeros(T_WIN, bool)
    H[sl] = d["heads"][lo:hi]
    pf[sl] = d["pf"][lo:hi]
    obs[sl] = d["obs"][lo:hi]
    dec[sl] = np.where(d["dec"][lo:hi] <= 2, d["dec"][lo:hi], MP)
    yy = d["y"][lo:hi].astype(np.int64)
    y[sl] = np.where(yy >= 0, yy, 0)
    ky[sl] = yy >= 0
    valid[sl] = d["half"][lo:hi] > 0
    vid = np.zeros((T_WIN, NV, NVD), np.float16)
    vok = np.zeros(T_WIN, bool)
    if video and "vid" in d:
        vid[sl] = d["vid"][lo:hi]
        vok[sl] = d["vid_ok"][lo:hi]
    if rng is not None:
        frac = rng.uniform(0.05, 0.5)
        if rng.random() < 0.5:
            keys = list(d["head_keys"])
            for i, k in enumerate(keys):
                if k.endswith("team"):
                    H[:, i] = -H[:, i]
            pf[..., U.NF - 6] *= -1
            y = np.where(ky & (y < 2), 1 - y, y)
            dec = np.where(dec < 2, 1 - dec, dec)
        if video and rng.random() < 0.3:
            vok[:] = False
    mk = remask_mask(margin(H, d["head_keys"]), dec, frac)
    return H, pf, obs, dec, mk, y, ky, valid, vid, vok, np.array(bool(d["dead_ok"]))


class Windows(torch.utils.data.IterableDataset):
    def __init__(self, names, seed, video):
        self.names, self.seed, self.video = names, seed, video

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0])
        packs = {}
        names = list(self.names)
        if self.video:
            w = np.array([4.0 if n.startswith("stv2") else 1.0 for n in names])
        else:
            w = np.ones(len(names))
        w /= w.sum()
        while True:
            n = names[int(rng.choice(len(names), p=w))]
            if n not in packs:
                packs[n] = load_pack(n)
            d = packs[n]
            T = len(d["y"])
            s0 = int(rng.integers(-C0, max(-C0 + 1, T - C1)))
            out = window(d, s0, rng, self.video)
            if out[6][C0:C1].sum() < 20:
                continue
            yield tuple(torch.from_numpy(np.ascontiguousarray(x)) for x in out)


def fold_of():
    from mtah import gs_build as B
    G = B.groups()
    return {n: G[n] for n in dev_names()}


def cmd_train(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(a.seed + int(a.fold))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    fo = fold_of()
    names = [n for n in dev_names() if fo[n] != str(a.fold)]
    nh = len(np.load(os.path.join(PACK, names[0] + ".npz"))["head_keys"])
    model = Corrector(nh, d=a.d, layers=a.layers, heads=a.heads, ff=4 * a.d, video=bool(a.video)).to(dev)
    odir = os.path.join(RUNS, a.name, "fold_%s" % a.fold)
    os.makedirs(odir, exist_ok=True)
    print("fold %s: %d training matches (%d stv2), params %.2f M, video %d" % (
        a.fold, len(names), sum(n.startswith("stv2") for n in names), sum(p.numel() for p in model.parameters()) / 1e6,
        a.video), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 300) * (0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)) * 0.95 + 0.05))
    dl = torch.utils.data.DataLoader(Windows(names, a.seed, bool(a.video)), batch_size=a.batch,
                                     num_workers=a.workers, persistent_workers=True, prefetch_factor=2)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    step, hist, t0 = 0, [], time.time()
    for H, pf, obs, dec, mk, y, ky, valid, vid, vok, dok in dl:
        if step >= a.steps:
            break
        H, pf, obs, dec, mk, y, ky, vid, vok, dok = (t.to(dev) for t in (H, pf, obs, dec, mk, y, ky, vid, vok, dok))
        B, T = y.shape
        flag = torch.ones(B, T, dtype=torch.long, device=dev)
        flag[:, C0:C1] = 0
        flag = torch.where(dec == MP, torch.full_like(flag, 2), flag)
        canvas = flag == 0
        rnd = torch.randint(0, 3, (B, T), device=dev, generator=gen)
        extra = torch.rand(B, T, device=dev, generator=gen) < 0.1
        x = torch.where(canvas & (mk | extra), rnd, dec)
        x = torch.where(flag == 2, torch.full_like(x, MP), x)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.evidence(H, pf, obs, vid, vok)
            sc_on = torch.rand(B, device=dev, generator=gen) < 0.5
            with torch.no_grad():
                sc = model(e, x, flag)
            lg = model(e, x, flag, sc.detach(), sc_on)
        lpf = lg.float().log_softmax(-1)
        ce3 = -lpf.gather(-1, y[..., None])[..., 0]
        ce2 = -(lpf.gather(-1, y.clamp(max=1)[..., None])[..., 0] - torch.logsumexp(lpf[..., :2], -1))
        full = dok.reshape(B)[:, None].expand(B, T)
        w = (ky & canvas & (full | (y < 2))).float()
        w = w * (1.0 + 2.0 * mk.float())
        loss = (torch.where(full, ce3, ce2) * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        with torch.no_grad():
            am = lg.argmax(-1)
            m = (ky & canvas & mk & (full | (y < 2)))
            acc_fix = ((am == y) & m).sum().item() / max(m.sum().item(), 1)
            acc_dec = ((dec == y) & m).sum().item() / max(m.sum().item(), 1)
        hist.append([loss.item(), acc_fix, acc_dec])
        if step % 500 == 0:
            h = np.mean(hist, 0)
            hist = []
            print("fold %s step %d loss %.3f re-masked acc corrector %.3f vs #9 decode %.3f  %.1f s/100" % (
                a.fold, step, h[0], h[1], h[2], (time.time() - t0) / 5), flush=True)
            t0 = time.time()
    torch.save({"model": model.state_dict(), "args": vars(a), "nh": model.h_enc[0].in_features},
               os.path.join(odir, "model.pt"))


@torch.no_grad()
def correct(model, d, dev, N=8, frac=0.2, video=False, lam_sw=0.0, lam_d=0.0, seed=0, batch=48):
    T = len(d["y"])
    can = C1 - C0
    starts = np.arange(0, T, can)
    pp = np.full((T, 3), np.nan, np.float32)
    rm = np.zeros(T, bool)
    tok = d["dec"].astype(np.int64).copy()
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    for i in range(0, len(starts), batch):
        g = starts[i:i + batch]
        ws = [window(d, s - C0, None, video, frac) for s in g]
        H, pf, obs, dec, mk, y, ky, valid, vid, vok, dok = (np.stack([w[k] for w in ws]) for k in range(11))
        Tt = lambda x: torch.from_numpy(x).to(dev)
        H, pf, obs, dec_t, mk_t, vid, vok = Tt(H), Tt(pf), Tt(obs), Tt(dec), Tt(mk), Tt(vid), Tt(vok)
        B = len(g)
        flag = torch.ones(B, T_WIN, dtype=torch.long, device=dev)
        flag[:, C0:C1] = 0
        flag = torch.where(dec_t == MP, torch.full_like(flag, 2), flag)
        canvas = (flag == 0) & mk_t
        x = torch.where(canvas, torch.randint(0, 3, (B, T_WIN), device=dev, generator=gen), dec_t)
        x = torch.where(flag == 2, torch.full_like(x, MP), x)
        committed = ~canvas & (flag != 2)
        nC = canvas.sum(-1, keepdim=True)
        rank = torch.arange(T_WIN, device=dev)[None]
        sc = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.evidence(H, pf, obs, vid, vok)
            for n in range(N, 0, -1):
                lg = model(e, x, flag, sc).float()
                if lam_sw > 0 or lam_d > 0:
                    lg = lg + spell_prior(x, committed, lam_sw, lam_d)
                tau = 0.4 + 0.4 * n / N
                pl = lg / tau
                p = pl.softmax(-1)
                am = pl.argmax(-1)
                ent = -(p * p.clamp(min=1e-12).log()).sum(-1)
                ent = torch.where(canvas, ent, torch.full_like(ent, float("inf")))
                se, si = torch.sort(ent, dim=-1, stable=True)
                frac_c = 1.0 - math.cos(math.pi / 2 * (N - n + 1) / N)
                acc_s = (((se.cumsum(-1) - se) <= 0.1) | (rank < torch.ceil(frac_c * nC))) & torch.isfinite(se)
                acc = torch.zeros_like(acc_s).scatter(1, si, acc_s)
                x = torch.where(canvas & acc, am, x)
                x = torch.where(canvas & ~acc, torch.randint(0, 3, (B, T_WIN), device=dev, generator=gen), x)
                committed = (~canvas | acc) & (flag != 2)
                sc = pl
            pr = lg.softmax(-1).cpu().numpy()
        xs = x.cpu().numpy()
        for b, s in enumerate(g):
            fr = np.arange(s, min(s + can, T))
            k = fr - s + C0
            pp[fr] = pr[b, k]
            rm[fr] = mk[b, k]
            tok[fr] = np.where(mk[b, k], xs[b, k], tok[fr])
    return pp, rm, tok


def cmd_infer(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    from mtah import gs_build as B
    dev = "cuda"
    st = torch.load(os.path.join(RUNS, a.name, "fold_%s" % a.fold, "model.pt"), map_location=dev, weights_only=False)
    ar = st["args"]
    model = Corrector(st["nh"], d=ar["d"], layers=ar["layers"], heads=ar["heads"], ff=4 * ar["d"],
                      video=bool(ar["video"])).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    fo = fold_of()
    os.makedirs(POST, exist_ok=True)
    for i, n in enumerate([n for n in dev_names() if fo[n] == str(a.fold)]):
        d = load_pack(n)
        use_v = bool(ar["video"]) and not a.no_video
        pp, rm, tok = correct(model, d, dev, N=a.steps, frac=a.frac, video=use_v, lam_sw=a.lam_sw, lam_d=a.lam_d,
                              seed=i)
        N = int(d["n_frames"])
        up = lambda x: np.repeat(x, U.STEP, axis=0)[:N]
        out = os.path.join(POST, "%s_s%d_F%s_%s.npz" % (n, B.eval_seed(n), a.fold, a.name + a.tag))
        np.savez_compressed(out, p_poss=up(pp), remask=up(rm), tok_poss=tok)
        print(n, "remasked %.3f" % rm.mean(), flush=True)


DEFAULT_OPTS = {"d": 192, "layers": 6, "heads": 6, "lr": 2e-4, "steps": 4000, "batch": 16, "seed": 20261113,
                "video": False, "frac": 0.35, "N": 8}


def train_model(names, opts, device="cuda", log=print):
    o = dict(DEFAULT_OPTS, **(opts or {}))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(o["seed"])
    nh = len(np.load(os.path.join(PACK, names[0] + ".npz"))["head_keys"])
    model = Corrector(nh, d=o["d"], layers=o["layers"], heads=o["heads"], ff=4 * o["d"], video=bool(o["video"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=0.01, betas=(0.9, 0.98))
    steps = o["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s_: min(1.0, s_ / 300) * (0.5 * (1 + np.cos(np.pi * min(s_, steps) / steps)) * 0.95 + 0.05))
    it = iter(Windows(sorted(names), o["seed"], bool(o["video"])))
    gen = torch.Generator(device=device)
    gen.manual_seed(o["seed"])
    t0 = time.time()
    for step in range(steps):
        batch = [next(it) for _ in range(o["batch"])]
        H, pf, obs, dec, mk, y, ky, valid, vid, vok, dok = (torch.stack([b[k] for b in batch]).to(device)
                                                            for k in range(11))
        B, T = y.shape
        flag = torch.ones(B, T, dtype=torch.long, device=device)
        flag[:, C0:C1] = 0
        flag = torch.where(dec == MP, torch.full_like(flag, 2), flag)
        canvas = flag == 0
        rnd = torch.randint(0, 3, (B, T), device=device, generator=gen)
        extra = torch.rand(B, T, device=device, generator=gen) < 0.1
        x = torch.where(canvas & (mk | extra), rnd, dec)
        x = torch.where(flag == 2, torch.full_like(x, MP), x)
        model.train()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            e = model.evidence(H, pf, obs, vid, vok)
            sc_on = torch.rand(B, device=device, generator=gen) < 0.5
            with torch.no_grad():
                sc = model(e, x, flag)
            lg = model(e, x, flag, sc.detach(), sc_on)
        lpf = lg.float().log_softmax(-1)
        ce3 = -lpf.gather(-1, y[..., None])[..., 0]
        ce2 = -(lpf.gather(-1, y.clamp(max=1)[..., None])[..., 0] - torch.logsumexp(lpf[..., :2], -1))
        full = dok.reshape(B)[:, None].expand(B, T)
        w = (ky & canvas & (full | (y < 2))).float() * (1.0 + 2.0 * mk.float())
        loss = (torch.where(full, ce3, ce2) * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 1000 == 0:
            log("  [corrector] step %d loss %.4f %.0fs" % (step + 1, loss.item(), time.time() - t0))
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"state": sd, "nh": nh, "opts": o, "names": sorted(names)}


def build_model(art, device="cuda"):
    o = art["opts"]
    m = Corrector(art["nh"], d=o["d"], layers=o["layers"], heads=o["heads"], ff=4 * o["d"], video=bool(o["video"]))
    m.load_state_dict(art["state"])
    return m.to(device).eval()


def pack_from_heads(p, dec25, view, head_keys):
    fr = view["frame"]
    N = len(dec25)
    blk = lambda x: np.stack([x[np.clip(fr + k, 0, N - 1)] for k in range(U.STEP)]).mean(0)
    H = np.stack([blk(_logit(np.asarray(p[k], np.float64))) for k in head_keys], 1).astype(np.float16)
    T = len(fr)
    return {"heads": H, "head_keys": np.array(head_keys), "dec": np.asarray(dec25)[fr].astype(np.int8),
            "y": np.full(T, -1, np.int8), "half": np.asarray(view["half"]).astype(np.int8), "frame": fr,
            "dead_ok": True, "n_frames": N, "pf": view["pf"], "obs": view["obs"]}


def apply(art, p, dec25, view, device="cuda", seed=0):
    o = art["opts"]
    torch.use_deterministic_algorithms(True, warn_only=False)
    model = build_model(art, device)
    keys = [str(k) for k in np.load(os.path.join(PACK, art["names"][0] + ".npz"))["head_keys"]]
    missing = [k for k in keys if k not in p]
    if missing:
        raise KeyError("base heads lack corrector inputs %s" % missing[:5])
    d = pack_from_heads(p, dec25, view, keys)
    pp, rm, tok = correct(model, d, device, N=o["N"], frac=o["frac"], video=False, seed=seed)
    return np.repeat(pp, U.STEP, axis=0)[:len(dec25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--name", default="c1")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--video", type=int, default=0)
    ap.add_argument("--no-video", action="store_true", help="infer with the video stream absent (null token)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--lam-sw", type=float, default=0.0)
    ap.add_argument("--lam-d", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=None, help="pack: pipe_gsraw_pdc config (decoder source)")
    a = ap.parse_args()
    if a.cmd == "pack":
        cmd_pack(a)
    elif a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "infer":
        if a.steps == 4000:
            a.steps = 8
        cmd_infer(a)


if __name__ == "__main__":
    main()
