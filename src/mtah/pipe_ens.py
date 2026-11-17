import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

from mtah import nn_seq
from mtah.train_v5 import PARAMS, boundary_labels, decode_heads  # noqa: F401  (PARAMS: same dict object)
from mtah import train_v5

ENS_HEADS = ("team", "sw")
NN_ARGS = nn_seq.get_args([])
_CTX = {}


def _ctx():
    if not _CTX:
        from mtah.eval_ext import canon_index, seed_fns, all_names, EXT
        from mtah.featcache import code_hash
        from mtah.degrade_profiles import active_profile
        from mtah.match import BALL
        prof = active_profile(0)
        if active_profile(1000) != prof or prof not in ("default", "ext_rand"):
            raise RuntimeError("pipe_ens supports the default or the ext_rand degradation profile, the same for "
                               "training and evaluation views")
        ci, ti = canon_index()
        tr, ev = seed_fns(0)
        _CTX.update(h=code_hash(), ball=BALL, ci=ci, ti=ti, ext=EXT, lengths={}, train_seed=tr, eval_seed=ev,
                    names=all_names(), prof=prof)
    return _CTX


def _train_seed(n):
    return _ctx()["train_seed"](n)


def _pooled(n, sw, seed):
    c = _ctx()
    if c["prof"] == "default":
        return nn_seq.pooled(n, sw, seed, c["h"], c["ball"])
    from mtah.data import CACHE
    from mtah.featcache import _path, _compute
    out = os.path.join(CACHE, "nn5", "%s_%s_p-%s" % (c["h"], c["ball"], c["prof"]),
                       "%s_%s_s%d.npz" % (n, "swap" if sw else "orig", seed))
    if not os.path.exists(out):
        src = _path(n, sw, seed, c["h"], c["prof"])
        if not os.path.exists(src):
            _compute((n, sw, seed, c["h"], c["prof"]))
        z = np.load(src)
        fn = list(z["names"])
        segs = nn_seq.pool_view(z["X"], z["half"], [fn.index(k) for k in nn_seq.MISS_COLS])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez(tmp, y=z["y"], half=z["half"], seg=np.array([x[:3] for x in segs]),
                 **{"P%d" % i: x[3] for i, x in enumerate(segs)})
        os.replace(tmp, out)
    z = np.load(out)
    return [(int(h), int(a), int(k), z["P%d" % i]) for i, (h, a, k) in enumerate(z["seg"])], z["y"], z["half"]


def _check_rows(rows):
    from mtah.featcache import _path
    c = _ctx()
    n, Xk = rows[0][0], rows[0][1]
    z = np.load(_path(n, False, _train_seed(n), c["h"], c["prof"]))
    X, y, half = z["X"], z["y"], z["half"]
    if X.shape[1] != Xk.shape[1]:
        raise RuntimeError("pipe_ens needs all features (drop_features unsupported)")
    first = np.flatnonzero((y < 3) & (half > 0))[0]
    if not np.array_equal(X[first], Xk[0], equal_nan=True):
        raise RuntimeError("training view of %s is not the deg_seed=0 cache view" % n)


def _train_nn(segs, F, seed, log=print):
    import time
    import torch
    a = NN_ARGS
    nn_seq.set_determinism(seed)
    dev = torch.device("cuda")
    cpu = torch.device("cpu")
    model = nn_seq.make_model(F, H=a.hidden, layers=a.layers, in_drop=a.in_drop, arch=a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
    U = torch.tensor(nn_seq.upsample_matrix(nn_seq.L, nn_seq.L * nn_seq.POOL), device=dev)
    rng = np.random.default_rng(seed)
    t0, run = time.time(), {}
    for it in range(a.iters):
        model.train()
        x, t, m = nn_seq.sample_batch(segs, rng, a.batch, a.p_dead, U, cpu)
        x, t, m = x.to(dev), t.to(dev), m.to(dev)
        loss, parts = nn_seq.loss_fn(model(x), t, m, U, a.tmse)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run[k] = run.get(k, 0) + v.detach()
        if (it + 1) % 1000 == 0:
            log("    nn it %d %.0fs %s" % (it + 1, time.time() - t0,
                                          " ".join("%s %.4f" % (k, float(v) / 1000) for k, v in run.items())))
            run = {}
    model.eval()
    return model


_LAB_KEYS = {}


def _swap(y):
    out = y.copy()
    out[y == 0] = 1
    out[y == 1] = 0
    return out


def _row_labels(rows):
    out = {}
    first = {}
    for r in rows:
        first.setdefault(r[0], r[2])
    for n, yr in first.items():
        if n in _LAB_KEYS:
            k, lab = _LAB_KEYS[n]
        else:
            z = np.load(os.path.join(_ctx()["ext"], n + ".npz"))
            half = z["half"]
            k = lab = None
            for key in ["label"] + sorted(f for f in z.files if f.startswith("label_")):
                cand = z[key]
                idx = np.flatnonzero((cand < 3) & (half > 0))
                if any(len(idx[::st]) == len(yr) and np.array_equal(cand[idx[::st]], yr) for st in range(1, 13)):
                    k, lab = key, cand
                    break
            if k is None:
                raise RuntimeError("pipe_ens: no label array of %s reproduces its training rows" % n)
            _LAB_KEYS[n] = (k, lab)
        if k != "label":
            out[n] = lab
    return out


def fit(rows, dead_set, team_set):
    import time
    import torch
    torch.set_num_threads(4)
    c = _ctx()
    _check_rows(rows)
    models = {"lgb": train_v5.fit(rows, dead_set, team_set)}
    t = time.time()
    names = sorted({r[0] for r in rows})
    tr = [(n, sw) for n in names for sw in (False, True)]
    view = lambda n, sw: _pooled(n, sw, _train_seed(n))
    mu, sd = nn_seq.norm_stats(view(n, sw) for n, sw in tr)
    lab = _row_labels(rows)

    def train_view(n, sw):
        segs_, y, half = view(n, sw)
        if n in lab:
            y = _swap(lab[n]) if sw else lab[n]
        return segs_, y, half, n in dead_set, n in team_set
    segs = nn_seq.build_train((train_view(n, sw) for n, sw in tr), mu, sd, torch.device("cpu"))
    if lab:
        print("  [pipe_ens] nn labels from rows: %s" % {n: k for n, (k, _) in sorted(_LAB_KEYS.items()) if n in lab},
              flush=True)
    F = segs[0]["x"].shape[1]
    print("  [pipe_ens] nn: %d train views (%d dead, %d team matches), %d segments, F=%d, prep %.0fs"
          % (len(tr), len(set(names) & set(dead_set)), len(set(names) & set(team_set)), len(segs), F,
             time.time() - t), flush=True)
    torch.cuda.empty_cache()
    seed = 100 * int(PARAMS.get("random_state", 0) or 0) + 1
    model = _train_nn(segs, F, seed)
    del segs
    print("  [pipe_ens] nn trained %.0fs (seed %d, peak VRAM %.2f GB)"
          % (time.time() - t, seed, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    models.update(nn=model, mu=mu, sd=sd)
    return models


def _identify(X):
    from mtah.featcache import _path
    c = _ctx()
    if not c["lengths"]:
        for n in c["names"]:
            with np.load(os.path.join(c["ext"], n + ".npz")) as z:
                c["lengths"][n] = len(z["half"])
    for n in sorted(k for k, T in c["lengths"].items() if T == len(X)):
        p = _path(n, False, c["eval_seed"](n), c["h"], c["prof"])
        if not os.path.exists(p):
            continue
        z = np.load(p)
        if z["X"].shape == X.shape and np.array_equal(z["X"], X, equal_nan=True):
            return n, z["half"], list(z["names"])
    raise RuntimeError("eval view (T=%d) not found among cached eval views" % len(X))


def predict(models, m=None, X=None, Xs=None, half=None, names=None):
    p = train_v5.predict(models["lgb"], m=m, X=X, Xs=Xs)
    if X is None:
        raise ValueError("pipe_ens.predict needs the feature views X / Xs")
    if half is None:
        _, half, fnames = _identify(X)
    else:
        fnames = list(names)
    miss = [fnames.index(col) for col in nn_seq.MISS_COLS]
    nn, mu, sd = models["nn"], models["mu"], models["sd"]
    q = nn_seq.combine(nn_seq.predict_segs(nn, nn_seq.pool_view(X, half, miss), mu, sd, len(X)),
                       nn_seq.predict_segs(nn, nn_seq.pool_view(Xs, half, miss), mu, sd, len(X)))
    for k in ENS_HEADS:
        p[k] = (p[k] + q[k].astype(np.float64)) / 2
    return p
