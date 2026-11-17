import argparse
import glob
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

from mtah.data import ROOT, CACHE

EXT = os.path.join(ROOT, "data_external", "common")
HEADS = ("dead", "team", "sw", "on", "off")
POOL = 5
MISS_COLS = ("b_x", "b_acc", "b_out_x", "d1_h", "d1_a")
L = 900
CLIP = 5.0


def layout(dead, team, tune, n_folds, seed=0):
    srcs = sorted(set(dead + team + tune))
    names = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(EXT, "*.npz")))]
    names = [n for n in names if n.split("_")[0] in srcs]
    tune_names = [n for n in names if n.split("_")[0] in tune]
    folds = [tune_names[i::n_folds] for i in range(n_folds)]
    train_seed = {n: 7 * i + seed for i, n in enumerate(names)}
    return names, folds, train_seed


def view_path(name, swap, seed, h, ball):
    from mtah.featcache import DIR
    return os.path.join(DIR, h + "_" + ball, "%s_%s_s%d.npz" % (name, "swap" if swap else "orig", seed))


def wait_for(path, idle_limit=3600):
    d = os.path.dirname(path)
    last, t_last = -1, time.time()
    while not os.path.exists(path):
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        if n != last:
            last, t_last = n, time.time()
        if time.time() - t_last > idle_limit:
            return False
        time.sleep(20)
    return True


def pool_view(X, half, miss_idx):
    segs = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        a, n = int(idx[0]), len(idx)
        assert idx[-1] - a + 1 == n, "half not contiguous"
        nb = -(-n // POOL)
        Xs = X[a:a + n]
        if nb * POOL > n:
            Xs = np.concatenate([Xs, np.repeat(Xs[-1:], nb * POOL - n, 0)])
        M = np.isnan(Xs)
        S = np.where(M, 0, Xs).reshape(nb, POOL, -1).sum(1, dtype=np.float64)
        C = (~M).reshape(nb, POOL, -1).sum(1)
        P = (S / np.maximum(C, 1)).astype(np.float32)
        P[C == 0] = np.nan
        ind = [M[:, j].reshape(nb, POOL).mean(1) for j in miss_idx]
        rest = np.ones(M.shape[1], bool)
        rest[list(miss_idx)] = False
        ind.append(M[:, rest].any(1).reshape(nb, POOL).mean(1))
        segs.append((int(h), a, n, np.concatenate([P, np.stack(ind, 1).astype(np.float32)], 1)))
    return segs


def pooled(name, swap, seed, h, ball):
    out = os.path.join(CACHE, "nn5", h + "_" + ball, "%s_%s_s%d.npz" % (name, "swap" if swap else "orig", seed))
    if not os.path.exists(out):
        src = view_path(name, swap, seed, h, ball)
        if not wait_for(src):
            from mtah.featcache import _compute
            _compute((name, swap, seed, h))
        z = np.load(src)
        names = list(z["names"])
        miss = [names.index(c) for c in MISS_COLS]
        segs = pool_view(z["X"], z["half"], miss)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez(tmp, y=z["y"], half=z["half"], seg=np.array([s[:3] for s in segs]),
                 **{"P%d" % i: s[3] for i, s in enumerate(segs)})
        os.replace(tmp, out)
    z = np.load(out)
    segs = [(int(h), int(a), int(n), z["P%d" % i]) for i, (h, a, n) in enumerate(z["seg"])]
    return segs, z["y"], z["half"]


def _atd_gaps():
    from mtah.data import graded_mask
    from mtah.ball import runs
    from mtah.match import ball_array
    b = ball_array("atd")[:, 0]
    s, e = runs(np.isnan(b) & graded_mask())
    return e - s


def pooled_heavy(name, swap, seed, h, ball, frac):
    out = os.path.join(CACHE, "nn5", "%s_%s_heavy%02d" % (h, ball, round(frac * 100)),
                       "%s_%s_s%d.npz" % (name, "swap" if swap else "orig", seed))
    if not os.path.exists(out):
        from mtah.featcache import code_hash
        assert code_hash() == h, "heavy views need the current feature code"
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        from mtah.featset import build, swap_teams
        from mtah.train_ext import load_ext, degrade
        m = [x for x in load_ext([name.split("_")[0]]) if x["name"] == name][0]
        if swap:
            m = swap_teams(m)
        d = degrade(m, np.random.default_rng(seed), _atd_gaps(), frac)
        X, names = build(d)
        segs = pool_view(X, m["half"], [names.index(c) for c in MISS_COLS])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez(tmp, y=m["label"], half=m["half"], seg=np.array([x[:3] for x in segs]),
                 **{"P%d" % i: x[3] for i, x in enumerate(segs)})
        os.replace(tmp, out)
    z = np.load(out)
    segs = [(int(hh), int(a), int(n), z["P%d" % i]) for i, (hh, a, n) in enumerate(z["seg"])]
    return segs, z["y"], z["half"]


def _heavy_job(k):
    return pooled_heavy(*k)[0] is not None


def make_model(F, H=128, layers=2, drop=0.2, in_drop=0.1, arch="gru"):
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.c = nn.Conv1d(H, H, 3, dilation=d, padding=d)
            self.p = nn.Conv1d(H, H, 1)
            self.dr = nn.Dropout(drop)

        def forward(self, x):
            return x + self.dr(self.p(nn.functional.gelu(self.c(x))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Dropout(in_drop), nn.Linear(F, H), nn.GELU(), nn.LayerNorm(H))
            if arch == "gru":
                self.rnn = nn.GRU(H, H, num_layers=layers, batch_first=True, bidirectional=True, dropout=drop)
                out = 2 * H
            elif arch == "tcn":
                self.tcn = nn.Sequential(*[Res(2 ** i) for i in range(10)])
                out = H
            else:
                self.mlp = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H), nn.GELU())
                out = H
            self.drop = nn.Dropout(drop)
            self.head = nn.Linear(out, len(HEADS))

        def forward(self, x):
            x = self.inp(x)
            if arch == "gru":
                x, _ = self.rnn(x)
            elif arch == "tcn":
                x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.mlp(x)
            return self.head(self.drop(x))
    return Net()


def upsample_matrix(n5, n25):
    U = np.zeros((n5, n25), np.float32)
    pos = (np.arange(n25) - (POOL - 1) / 2) / POOL
    i0 = np.clip(np.floor(pos).astype(int), 0, n5 - 1)
    i1 = np.minimum(i0 + 1, n5 - 1)
    w = np.clip(pos - i0, 0, 1).astype(np.float32)
    np.add.at(U, (i0, np.arange(n25)), 1 - w)
    np.add.at(U, (i1, np.arange(n25)), w)
    return U


def upsample(logits5, n25):
    pos = (np.arange(n25) - (POOL - 1) / 2) / POOL
    return np.stack([np.interp(pos, np.arange(len(logits5)), c) for c in logits5.T], 1)


def norm_stats(views):
    s = s2 = c = 0
    for segs, _, _ in views:
        for _, _, _, P in segs:
            m = ~np.isnan(P)
            V = np.where(m, P, 0).astype(np.float64)
            s = s + V.sum(0)
            s2 = s2 + (V * V).sum(0)
            c = c + m.sum(0)
    mu = s / np.maximum(c, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(c, 1) - mu * mu, 0))
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def standardise(P, mu, sd):
    Z = (P - mu) / sd
    Z[np.isnan(Z)] = 0
    return np.clip(Z, -CLIP, CLIP).astype(np.float32)


def build_train(views, mu, sd, device):
    import torch
    from mtah.train_v5 import boundary_labels
    out = []
    for segs, y, half, dsrc, tsrc in views:
        bsw, bon, boff = boundary_labels(y, half)
        for h, a, n, P in segs:
            tgt = np.stack([y[a:a + n].astype(np.float32), bsw[a:a + n], bon[a:a + n], boff[a:a + n]], 0)
            out.append(dict(x=torch.tensor(standardise(P, mu, sd), dtype=torch.float16, device=device),
                            t=torch.tensor(tgt, dtype=torch.float16, device=device),
                            n=n, nb=len(P), dead=dsrc, team=tsrc))
    return out


def sample_batch(segs, rng, B, p_dead, U, device):
    import torch
    dead_ix = [i for i, s in enumerate(segs) if s["dead"]]
    w = np.array([s["nb"] for s in segs], float) * np.array([s["team"] for s in segs])
    w /= w.sum()
    xs, ts, ms = [], [], []
    for b in range(B):
        i = dead_ix[rng.integers(len(dead_ix))] if rng.random() < p_dead else int(rng.choice(len(segs), p=w))
        s = segs[i]
        st = int(rng.integers(0, max(s["nb"] - L, 0) + 1))
        x = s["x"][st:st + L]
        t = s["t"][:, st * POOL:(st + L) * POOL]
        if x.shape[0] < L:
            x = torch.nn.functional.pad(x, (0, 0, 0, L - x.shape[0]))
        if t.shape[1] < L * POOL:
            pad = torch.zeros(4, L * POOL - t.shape[1], dtype=t.dtype, device=device)
            pad[0] = 3
            t = torch.cat([t, pad], 1)
        xs.append(x)
        ts.append(t)
        ms.append((s["dead"], s["team"]))
    m = torch.tensor(ms, dtype=torch.bool, device=device)
    return torch.stack(xs).float(), torch.stack(ts).float(), m


def loss_fn(z5, t, m, U, lam):
    import torch
    import torch.nn.functional as Fn
    z = torch.einsum("blc,lt->btc", z5, U)
    y = t[:, 0]
    valid = y < 2.5
    alive = y < 1.5
    md = valid & m[:, 0:1]
    mt = alive & m[:, 1:2]

    def bce(zz, tt, mask):
        l = Fn.binary_cross_entropy_with_logits(zz, tt, reduction="none")
        return (l * mask).sum() / mask.sum().clamp(min=1)
    parts = {"dead": bce(z[..., 0], (y > 1.5).float(), md),
             "team": bce(z[..., 1], (y < 0.5).float(), mt),
             "sw": bce(z[..., 2], t[:, 1], mt),
             "on": bce(z[..., 3], t[:, 2], md),
             "off": bce(z[..., 4], t[:, 3], md)}
    zz = z5[..., :2]
    lp = torch.stack([Fn.logsigmoid(zz), Fn.logsigmoid(-zz)], -1)
    d = (lp[:, 1:] - lp[:, :-1].detach()) ** 2
    parts["tmse"] = d.clamp(max=16).mean()
    total = sum(v for k, v in parts.items() if k != "tmse") + lam * parts["tmse"]
    return total, parts


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


def train_model(segs, F, args, seed, log=print):
    import torch
    set_determinism(seed)
    dev = torch.device("cuda")
    model = make_model(F, H=args.hidden, layers=args.layers, in_drop=args.in_drop,
                       arch=args.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.1)
    U = torch.tensor(upsample_matrix(L, L * POOL), device=dev)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    run = {}
    for it in range(args.iters):
        model.train()
        x, t, m = sample_batch(segs, rng, args.batch, args.p_dead, U, dev)
        loss, parts = loss_fn(model(x), t, m, U, args.tmse)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run[k] = run.get(k, 0) + v.detach()
        if (it + 1) % 500 == 0:
            log("    it %d %.0fs %s" % (it + 1, time.time() - t0,
                                       " ".join("%s %.4f" % (k, float(v) / 500) for k, v in run.items())))
            run = {}
    return model


def predict_segs(model, segs, mu, sd, T):
    import torch
    model.eval()
    out = {h: np.full(T, 0.5 if h in ("dead", "team") else 0.0, np.float32) for h in HEADS}
    dev = next(model.parameters()).device
    for h, a, n, P in segs:
        Z = standardise(P, mu, sd)
        nb = len(Z)
        acc = np.zeros((nb, len(HEADS)), np.float64)
        wsum = np.zeros(nb)
        starts = list(range(0, max(nb - L, 0) + 1, L // 2))
        if starts[-1] + L < nb:
            starts.append(nb - L)
        with torch.no_grad():
            for st in starts:
                x = torch.tensor(Z[st:st + L], device=dev)[None]
                z = model(x)[0].double().cpu().numpy()
                k = len(z)
                w = np.minimum(np.arange(1, k + 1), np.arange(k, 0, -1)).astype(float)
                acc[st:st + k] += z * w[:, None]
                wsum[st:st + k] += w
        z25 = upsample(acc / wsum[:, None], n)
        p = 1 / (1 + np.exp(-z25))
        for j, hd in enumerate(HEADS):
            out[hd][a:a + n] = p[:, j]
    return out


def combine(po, ps):
    p = {k: (po[k] + ps[k]) / 2 for k in ("dead", "sw", "on", "off")}
    p["team"] = (po["team"] + 1 - ps["team"]) / 2
    return p


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead-data", nargs="+", default=["idsse-ctrl"])
    ap.add_argument("--team-data", nargs="+", default=["idsse-ctrl", "skillcorner", "metrica", "pff"])
    ap.add_argument("--tune-data", nargs="+", default=["idsse-ctrl", "pff"])
    ap.add_argument("--out", default="runs/nn1")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--only-folds", type=int, nargs="*", default=None)
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hash", default=None, help="feature cache hash (default: current code_hash)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--p-dead", type=float, default=0.35)
    ap.add_argument("--tmse", type=float, default=0.1)
    ap.add_argument("--in-drop", type=float, default=0.1)
    ap.add_argument("--arch", default="gru", choices=["gru", "tcn", "mlp"])
    ap.add_argument("--drop-prefix", nargs="*", default=[],
                    help="zero out input features whose name starts with any of these prefixes")
    ap.add_argument("--heavy", type=float, default=0.0,
                    help="add heavy-ball-gap views of the dead-data matches at this missing fraction")
    ap.add_argument("--provided", action="store_true", help="only: predict the provided game")
    ap.add_argument("--compare", default=None, help="only: decoder grid + gate vs this LightGBM run dir")
    ap.add_argument("--wait", action="store_true", help="with --compare: wait until the LightGBM run has "
                    "all OOF files and its signals file (it is another job's output)")
    ap.add_argument("--ens-signals", default=None, help="only: write runs/<out>_ens.signals.npz (all heads "
                    "averaged) and runs/<out>_ens_t.signals.npz (team/sw averaged, dead/on/off from LightGBM) "
                    "from runs/<out>.signals.npz and runs/<this>.signals.npz")
    return ap.parse_args(argv)


def _pool_job(k):
    return pooled(*k)[0] is not None


def main():
    args = get_args()
    from mtah.match import BALL
    import torch
    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    if args.compare:
        if args.wait:
            n_need = len(glob.glob(os.path.join(args.out, "oof_*.npz")))
            sig = os.path.join(ROOT, "runs", os.path.basename(args.compare.rstrip("/")) + ".signals.npz")
            while len(glob.glob(os.path.join(args.compare, "oof_*.npz"))) < n_need or not os.path.exists(sig):
                time.sleep(60)
            time.sleep(30)
        return compare(args)
    if args.ens_signals:
        return ens_signals(args)
    from mtah.featcache import code_hash
    h = args.hash or code_hash()
    names, folds, tseed = layout(args.dead_data, args.team_data, args.tune_data, args.folds, args.seed)
    src = lambda n: n.split("_")[0]
    if args.provided:
        return provided(args, h, BALL)
    keys = [(n, sw, tseed[n], h, BALL) for n in names for sw in (False, True)]
    keys += [(n, sw, 1000 + k, h, BALL) for k, f in enumerate(folds) for n in f for sw in (False, True)]
    t = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_pool_job, keys))
    print("pooled %d views (hash %s, ball %s) in %.0fs" % (len(keys), h, BALL, time.time() - t), flush=True)
    hv = [(n, sw, 5000 + tseed[n], h, BALL, args.heavy) for n in names if src(n) in args.dead_data
          for sw in (False, True)] if args.heavy > 0 else []
    if hv:
        t = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_heavy_job, hv))
        print("heavy-gap views: %d in %.0fs" % (len(hv), time.time() - t), flush=True)
    json.dump(dict(vars(args), hash=h, ball=BALL, L=L, pool=POOL, miss_cols=MISS_COLS),
              open(os.path.join(args.out, "config.json"), "w"), indent=1)

    plan = [(k, set(f)) for k, f in enumerate(folds) if args.only_folds is None or k in args.only_folds]
    if not args.no_final:
        plan.append((-1, set()))
    for k, held in plan:
        tag = "all" if k < 0 else "f%d" % k
        if k >= 0 and all(os.path.exists(os.path.join(args.out, "oof_%s.npz" % n)) for n in held):
            print("fold %d resumed from saved OOF" % k, flush=True)
            continue
        if k < 0 and os.path.exists(os.path.join(args.out, "model_all.pt")):
            continue
        t = time.time()
        tr = [(n, sw) for n in names if n not in held for sw in (False, True)]
        mu, sd = norm_stats(pooled(n, sw, tseed[n], h, BALL) for n, sw in tr)
        if args.drop_prefix:
            fn = list(np.load(view_path(names[0], False, tseed[names[0]], h, BALL))["names"])
            fn += ["m_" + c for c in MISS_COLS] + ["m_other"]
            dropped = [i for i, f in enumerate(fn) if f.startswith(tuple(args.drop_prefix))]
            mu[dropped], sd[dropped] = 0, 1e30
            print("dropped %d features" % len(dropped), flush=True)
        trh = [k_ for k_ in hv if k_[0] not in held]

        def gen():
            for n, sw in tr:
                yield (*pooled(n, sw, tseed[n], h, BALL), src(n) in args.dead_data, src(n) in args.team_data)
            for k_ in trh:
                yield (*pooled_heavy(*k_), True, src(k_[0]) in args.team_data)
        segs = build_train(gen(), mu, sd, torch.device("cuda"))
        F = segs[0]["x"].shape[1]
        print("fold %s: %d+%d heavy train views, %d segments, F=%d, prep %.0fs" % (tag, len(tr), len(trh), len(segs), F,
                                                                         time.time() - t), flush=True)
        model = train_model(segs, F, args, seed=args.seed * 100 + k + 1)
        del segs
        torch.cuda.empty_cache()
        torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "F": F, "args": vars(args)},
                   os.path.join(args.out, "model_%s.pt" % tag))
        print("fold %s trained in %.0fs" % (tag, time.time() - t), flush=True)
        for n in sorted(held):
            so, y, half = pooled(n, False, 1000 + k, h, BALL)
            ss, _, _ = pooled(n, True, 1000 + k, h, BALL)
            p = combine(predict_segs(model, so, mu, sd, len(y)), predict_segs(model, ss, mu, sd, len(y)))
            np.savez_compressed(os.path.join(args.out, "oof_%s.npz" % n), label=y, half=half,
                                **{hd: p[hd].astype(np.float32) for hd in HEADS})
        print("fold %s OOF saved (%d matches)" % (tag, len(held)), flush=True)


def load_model(path):
    import torch
    ck = torch.load(path, weights_only=False)
    a = ck["args"]
    model = make_model(ck["F"], H=a["hidden"], layers=a["layers"], in_drop=a.get("in_drop", 0.1),
                       arch=a.get("arch", "gru")).cuda()
    model.load_state_dict(ck["state"])
    return model, ck["mu"], ck["sd"]


def provided(args, h, ball):
    from mtah.featset import build, swap_teams
    from mtah.match import provided_match
    import torch
    set_determinism(0)
    model, mu, sd = load_model(os.path.join(args.out, "model_all.pt"))
    m = provided_match()
    res = []
    for v in (m, swap_teams(m)):
        X, names = build(v)
        miss = [names.index(c) for c in MISS_COLS]
        segs = pool_view(X, m["half"], miss)
        del X
        res.append(predict_segs(model, segs, mu, sd, len(m["half"])))
    p = combine(*res)
    from mtah.train_v5 import to_prob
    prob = to_prob(p)
    name = os.path.basename(args.out.rstrip("/"))
    path = os.path.join(ROOT, "runs", name + ".signals.npz")
    np.savez(path, p_home=prob[:, 0], p_away=prob[:, 1], p_dead=prob[:, 2], p_switch=p["sw"],
             p_dead_on=p["on"], p_dead_off=p["off"], **{"head_" + k: p[k] for k in HEADS})
    print("wrote", path)


def ens_signals(args):
    from mtah.train_v5 import to_prob
    name = os.path.basename(args.out.rstrip("/"))
    a = np.load(os.path.join(ROOT, "runs", name + ".signals.npz"))
    b = np.load(os.path.join(ROOT, "runs", os.path.basename(args.ens_signals.rstrip("/")) + ".signals.npz"))
    nn = {k: a["head_" + k].astype(np.float64) for k in HEADS}
    lg = {"dead": b["p_dead"], "team": b["p_home"] / np.maximum(b["p_home"] + b["p_away"], 1e-12),
          "sw": b["p_switch"], "on": b["p_dead_on"], "off": b["p_dead_off"]}
    for tag, keys in (("ens", HEADS), ("ens_t", ("team", "sw"))):
        p = {k: (nn[k] + lg[k]) / 2 if k in keys else lg[k] for k in HEADS}
        prob = to_prob(p)
        path = os.path.join(ROOT, "runs", "%s_%s.signals.npz" % (name, tag))
        np.savez(path, p_home=prob[:, 0], p_away=prob[:, 1], p_dead=prob[:, 2], p_switch=p["sw"],
                 p_dead_on=p["on"], p_dead_off=p["off"])
        print("wrote", path)


GRID = [{"switch": s, "dead": d, "beta": b} for s in (2, 4, 8, 12) for d in (8, 20, 40)
        for b in (0.0, 0.5, 1.0, 2.0)]
_OOF = None


def load_heads(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "oof_*.npz"))):
        z = np.load(p)
        out[os.path.basename(p)[4:-4]] = ({k: z[k].astype(np.float64) for k in HEADS}, z["label"], z["half"])
    return out


def _score_cfg(args):
    from mtah.train_v5 import decode_heads
    from mtah.train_ext import spell_f1
    variant, cfg = args
    return {n: spell_f1(decode_heads(p, half, cfg), y, half) for n, (p, y, half) in _OOF[variant].items()}


def compare(args):
    global _OOF
    from multiprocessing import Pool
    from mtah.eval_oof import gate
    nn, lgb = load_heads(args.out), load_heads(args.compare)
    common = sorted(set(nn) & set(lgb))
    assert common, "no common OOF matches"
    for n in common:
        assert np.array_equal(nn[n][1], lgb[n][1]), n
    _OOF = {"nn": {n: nn[n] for n in common}, "lgb": {n: lgb[n] for n in common},
            "ens": {n: ({k: (nn[n][0][k] + lgb[n][0][k]) / 2 for k in HEADS}, nn[n][1], nn[n][2]) for n in common},
            "ens_t": {n: ({k: (nn[n][0][k] + lgb[n][0][k]) / 2 if k in ("team", "sw") else lgb[n][0][k]
                           for k in HEADS}, nn[n][1], nn[n][2]) for n in common}}
    V = ("nn", "lgb", "ens", "ens_t")
    jobs = [(v, c) for v in V for c in GRID]
    with Pool(4) as pool:
        res = pool.map(_score_cfg, jobs)
    best = {}
    for (v, c), r in zip(jobs, res):
        s = float(np.mean(list(r.values())))
        if v not in best or s > best[v][0]:
            best[v] = (s, c, r)
    try:
        split = json.load(open(os.path.join(ROOT, "notes", "avo", "split.json")))
        dev = set(split["dev"])
    except Exception:
        dev = set()
    rep = {"n_matches": len(common), "lgb_dir": args.compare}
    for v in V:
        s, c, r = best[v]
        by = {}
        for n, x in r.items():
            by.setdefault(n.split("_")[0], []).append(x)
        rep[v] = {"spellF1": s, "decoder": c, "by_source": {k: float(np.mean(x)) for k, x in by.items()},
                  "dev53": float(np.mean([x for n, x in r.items() if n in dev])) if dev else None,
                  "per_match": r}
    rep["gate_nn_vs_lgb"] = gate(best["lgb"][2], best["nn"][2])
    rep["gate_ens_vs_lgb"] = gate(best["lgb"][2], best["ens"][2])
    rep["gate_ens_vs_nn"] = gate(best["nn"][2], best["ens"][2])
    rep["gate_ens_t_vs_lgb"] = gate(best["lgb"][2], best["ens_t"][2])
    tag = os.path.basename(args.compare.rstrip("/"))
    json.dump(rep, open(os.path.join(args.out, "compare_%s.json" % tag), "w"), indent=1)
    for v in V:
        print("%-4s spellF1 %.4f dev53 %s %s %s" % (v, rep[v]["spellF1"], rep[v]["dev53"], rep[v]["by_source"],
                                                    rep[v]["decoder"]))
    for g in ("gate_nn_vs_lgb", "gate_ens_vs_lgb", "gate_ens_vs_nn", "gate_ens_t_vs_lgb"):
        print(g, {k: rep[g][k] for k in ("mean_delta", "ci90", "p_signflip", "wins", "losses", "by_source", "accept")})


if __name__ == "__main__":
    main()
