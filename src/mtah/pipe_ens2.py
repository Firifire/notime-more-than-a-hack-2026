import os

import numpy as np

from mtah import nn_seq, pipe_ens, train_v5
from mtah.decode import viterbi_tv
from mtah.train_v5 import PARAMS, boundary_labels, HEADS  # noqa: F401  (PARAMS: same dict object as train_v5)

DEFAULT_W = {"team": 0.5, "sw": 0.5, "dead": 0.0, "on": 0.0, "off": 0.0}


def _opts():
    return {k: v for k, v in PARAMS.items() if k.startswith("pipe_")}


def _nn_args(opts):
    a = nn_seq.get_args([])
    for k in ("iters", "hidden", "layers", "lr", "tmse", "in_drop"):
        if "pipe_nn_" + k in opts:
            setattr(a, k, opts["pipe_nn_" + k])
    return a


_KEEP = {}


def _ext():
    from mtah.eval_ext import EXT
    return EXT


def _keep_of(n, y):
    import hashlib
    key = (n, hashlib.sha1(np.ascontiguousarray(y).tobytes()).hexdigest())
    if key in _KEEP:
        return _KEEP[key]
    z = np.load(os.path.join(_ext(), n + ".npz"))
    half = z["half"]
    for lk in ["label"] + sorted(f for f in z.files if f.startswith("label_")):
        cand = z[lk]
        idx = np.flatnonzero((cand < 3) & (half > 0))
        for st in range(1, 13):
            keep = idx[::st]
            if len(keep) != len(y):
                continue
            if np.array_equal(cand[keep], y):
                _KEEP[key] = (keep, lk, False, st)
                return _KEEP[key]
            if np.array_equal(pipe_ens._swap(cand)[keep], y):
                _KEEP[key] = (keep, lk, True, st)
                return _KEEP[key]
    raise RuntimeError("pipe_ens2: no label array of %s reproduces its training rows" % n)


def _transition_weight(n, y, k):
    keep, lk, _, _ = _keep_of(n, y)
    z = np.load(os.path.join(_ext(), n + ".npz"))
    lab, half = z[lk].astype(np.int64), z["half"]
    ch = np.r_[False, (lab[1:] != lab[:-1]) & (half[1:] == half[:-1]) & (half[1:] > 0)]
    from scipy.ndimage import maximum_filter1d
    near = maximum_filter1d(ch.astype(np.int8), 2 * k + 1) > 0
    return np.where(near[keep], 0.0, 1.0)


def _spell_weight(n, y, o):
    keep, lk, _, _ = _keep_of(n, y)
    return _spell_weight_full(n, lk, o)[keep]


def _spell_weight_full(n, lk, o):
    z = np.load(os.path.join(_ext(), n + ".npz"))
    lab, half = z[lk].astype(np.int64), z["half"].astype(np.int64)
    key = lab + 10 * half
    ch = np.flatnonzero(key[1:] != key[:-1]) + 1
    st = np.r_[0, ch]
    ln = np.diff(np.r_[st, len(key)])
    L = np.repeat(ln, ln).astype(np.float64)
    w = np.clip((float(o.get("ref", 250)) / L) ** float(o.get("alpha", 0.5)), float(o.get("min", 0.5)),
                float(o.get("max", 3.0)))
    if not o.get("dead"):
        w[lab >= 2] = 1.0
    return w


def _sb_targets(n, y, half_rows=None):
    if not n.startswith("pff_"):
        return None
    keep, lk, swapped, _ = _keep_of(n, y)
    p = os.path.join(_ext(), "pffsb_" + n.split("_", 1)[1] + ".npz")
    if not os.path.exists(p):
        raise FileNotFoundError("pipe_soft_pffsb needs the StatsBomb labels %s" % p)
    z = np.load(p)
    lab = z["label"]
    if swapped:
        lab = pipe_ens._swap(lab)
    bsw, _, _ = boundary_labels(lab, z["half"])
    return lab[keep], bsw[keep]


def fit_lgb(rows, dead_set, team_set, opts):
    import lightgbm as lgb
    P = {k: v for k, v in PARAMS.items() if not k.startswith("pipe_")}
    soft = dict(P, objective="cross_entropy")
    wsrc = opts.get("pipe_src_weight") or {}
    dc = int(opts.get("pipe_dontcare", 0) or 0)
    wsb = float(opts.get("pipe_soft_pffsb", 0.0) or 0.0)
    spw = opts.get("pipe_spell_w") or None
    use_w = bool(wsrc) or dc > 0
    use_wt = use_w or bool(spw)

    def weight(r):
        w = np.full(len(r[2]), float(wsrc.get(r[0].split("_")[0], 1.0)))
        if dc > 0:
            w = w * _transition_weight(r[0], r[2], dc)
        return w

    def cat(sel, f):
        return np.concatenate([f(r) for r in rows if sel(r[0])])
    D = lambda n: n in dead_set
    T = lambda n: n in team_set
    models = {}
    Xd = cat(D, lambda r: r[1])
    spw_d = spw if (spw and spw.get("dead")) else None
    bl = opts.get("pipe_dead_blind") or None
    if bl:
        return _fit_lgb_blind(rows, dead_set, team_set, opts, bl, P, soft, models, Xd)

    def weight_d(r):
        w = weight(r)
        return w * _spell_weight(r[0], r[2], spw_d) if spw_d else w
    wd = cat(D, weight_d) if (use_w or spw_d) else None
    models["dead"] = lgb.LGBMClassifier(**P).fit(Xd, cat(D, lambda r: (r[2] == 2).astype(int)), sample_weight=wd)
    models["on"] = lgb.LGBMRegressor(**soft).fit(Xd, cat(D, lambda r: r[4]), sample_weight=wd)
    models["off"] = lgb.LGBMRegressor(**soft).fit(Xd, cat(D, lambda r: r[5]), sample_weight=wd)
    del Xd
    Xt = cat(T, lambda r: r[1][r[2] < 2])
    def weight_t(r):
        w = weight(r)
        if spw:
            w = w * _spell_weight(r[0], r[2], spw)
        return w[r[2] < 2]
    wt = cat(T, weight_t) if use_wt else None

    def team_t(r):
        a = r[2] < 2
        t = (r[2][a] == 0).astype(np.float64)
        if wsb > 0:
            sb = _sb_targets(r[0], r[2])
            if sb is not None:
                lab = sb[0][a]
                ok = lab < 2
                t = np.where(ok, (1 - wsb) * t + wsb * (lab == 0), t)
        return t

    def sw_t(r):
        a = r[2] < 2
        t = r[3][a].astype(np.float64)
        if wsb > 0:
            sb = _sb_targets(r[0], r[2])
            if sb is not None:
                t = (1 - wsb) * t + wsb * sb[1][a]
        return t
    if wsb > 0:
        models["team"] = lgb.LGBMRegressor(**soft).fit(Xt, cat(T, team_t), sample_weight=wt)
        models["team_soft"] = True
    else:
        models["team"] = lgb.LGBMClassifier(**P).fit(Xt, cat(T, lambda r: (r[2][r[2] < 2] == 0).astype(int)),
                                                     sample_weight=wt)
    models["sw"] = lgb.LGBMRegressor(**soft).fit(Xt, cat(T, sw_t), sample_weight=wt)
    del Xt
    nb = int(opts.get("pipe_lgb_bags", 1) or 1)
    if nb > 1:
        if wsb > 0 or use_w:
            raise RuntimeError("pipe_lgb_bags supports only pipe_spell_w among the row options")
        models["team_bags"], models["sw_bags"] = [], []
        for b in range(1, nb):
            Xb, yb, sb, wb = _bag_rows([r for r in rows if T(r[0])], b, nb, spw)
            Pb = dict(P, random_state=int(P.get("random_state", 0) or 0) + b)
            models["team_bags"].append(lgb.LGBMClassifier(**Pb).fit(Xb, (yb == 0).astype(int), sample_weight=wb))
            models["sw_bags"].append(lgb.LGBMRegressor(**dict(soft, random_state=Pb["random_state"]))
                                     .fit(Xb, sb, sample_weight=wb))
            del Xb
    return models


def _feature_names():
    c = pipe_ens._ctx()
    if "fnames" not in c:
        from mtah.featcache import _path
        n = sorted(c["names"])[0]
        for cand in sorted(c["names"]):
            p = _path(cand, False, pipe_ens._train_seed(cand), c["h"], c["prof"])
            if os.path.exists(p):
                with np.load(p) as z:
                    c["fnames"] = list(z["names"])
                break
    return c["fnames"]


def _fit_lgb_blind(rows, dead_set, team_set, opts, bl, P, soft, models, Xd):
    import lightgbm as lgb
    from mtah import blind
    if opts.get("pipe_src_weight") or opts.get("pipe_dontcare") or opts.get("pipe_soft_pffsb") or \
            (opts.get("pipe_spell_w") or {}).get("dead") or int(opts.get("pipe_lgb_bags", 1) or 1) > 1:
        raise RuntimeError("pipe_dead_blind supports only pipe_spell_w (team rows) among the row options")
    srcs = set(bl.get("sources", ["pff"]))
    wb = float(bl.get("w", 0.5))
    names = _feature_names()
    if Xd.shape[1] != len(names):
        raise RuntimeError("pipe_dead_blind needs all features")
    B = [r for r in rows if r[0].split("_")[0] in srcs and r[0] not in dead_set]
    if not B:
        raise RuntimeError("pipe_dead_blind: no rows of %s" % sorted(srcs))
    masked = [r for r in rows if r[0] in dead_set and r[0].startswith("stv2_")]
    if masked:
        blind.check_against(masked[0][1], names)
    D = lambda n: n in dead_set
    nd = len(Xd)
    Xd = np.concatenate([Xd] + [blind.blind_rows(r[1], names) for r in B])
    yd = np.concatenate([r[2] for r in rows if D(r[0])] + [r[2] for r in B])
    on = np.concatenate([r[4] for r in rows if D(r[0])] + [r[4] for r in B])
    off = np.concatenate([r[5] for r in rows if D(r[0])] + [r[5] for r in B])
    wd = np.r_[np.ones(nd), np.full(len(Xd) - nd, wb)]
    print("  [pipe_ens2] dead_blind: %d native + %d blind rows (%s, w %.2f)" % (nd, len(Xd) - nd, sorted(srcs), wb),
          flush=True)
    wp = float(bl.get("present_w", 0.0) or 0.0)
    if wp > 0:
        jp = names.index("b_pres")
        Xp, yp, onp, offp, wpp = [], [], [], [], []
        clean = bl.get("present_mode") == "clean"
        j50 = [names.index("b_pres_past50"), names.index("b_pres_fut50")]
        jl = [names.index(c) for c in names if c.startswith("b_pres_past") or c.startswith("b_pres_fut")
              or c == "b_pres_m250"]
        for r in B:
            m = r[1][:, jp] == 1
            if clean:
                m &= (r[1][:, j50] > 0.999).all(1)
            d = r[2] == 2
            pa, pp = d.mean(), d[m].mean()
            if m.sum() < 100 or not (0 < pp < 1):
                continue
            xm = r[1][m]
            if clean:
                xm[:, jl] = 1.0
            Xp.append(xm); yp.append(r[2][m]); onp.append(r[4][m]); offp.append(r[5][m])
            wpp.append(wp * np.where(d[m], pa / pp, (1 - pa) / (1 - pp)))
        n0 = len(Xd)
        Xd = np.concatenate([Xd] + Xp)
        yd, on, off = np.concatenate([yd] + yp), np.concatenate([on] + onp), np.concatenate([off] + offp)
        wd = np.concatenate([wd] + wpp)
        print("  [pipe_ens2] dead_blind: + %d ball-present native rows (w %.2f, prior-corrected)" % (len(Xd) - n0, wp),
              flush=True)
    models["dead"] = lgb.LGBMClassifier(**P).fit(Xd, (yd == 2).astype(int), sample_weight=wd)
    models["on"] = lgb.LGBMRegressor(**soft).fit(Xd, on, sample_weight=wd)
    models["off"] = lgb.LGBMRegressor(**soft).fit(Xd, off, sample_weight=wd)
    del Xd
    T = lambda n: n in team_set
    spw = opts.get("pipe_spell_w") or None
    Xt = np.concatenate([r[1][r[2] < 2] for r in rows if T(r[0])])
    wt = np.concatenate([_spell_weight(r[0], r[2], spw)[r[2] < 2] for r in rows if T(r[0])]) if spw else None
    models["team"] = lgb.LGBMClassifier(**P).fit(
        Xt, np.concatenate([(r[2][r[2] < 2] == 0).astype(int) for r in rows if T(r[0])]), sample_weight=wt)
    models["sw"] = lgb.LGBMRegressor(**soft).fit(
        Xt, np.concatenate([r[3][r[2] < 2].astype(np.float64) for r in rows if T(r[0])]), sample_weight=wt)
    return models


def _bag_rows(rows, b, nb, spw):
    from mtah import eval_ext as E
    from mtah.featcache import _path, code_hash
    from mtah.degrade_profiles import active_profile
    prof = active_profile(0)
    h = code_hash() if prof == "default" else code_hash(prof)
    tr, _ = E.seed_fns(0)
    Xs_, ys_, ss_, ws_ = [], [], [], []
    for r in rows:
        n, y = r[0], r[2]
        keep, lk, swapped, st = _keep_of(n, y)
        z = np.load(os.path.join(_ext(), n + ".npz"))
        lab, half = z[lk], z["half"]
        if swapped:
            lab = pipe_ens._swap(lab)
        bsw = boundary_labels(lab, half)[0]
        if not np.array_equal(bsw[keep], r[3]):
            raise RuntimeError("pipe_lgb_bags: boundary targets of %s not reproduced" % n)
        idx = np.flatnonzero((lab < 3) & (half > 0))
        k2 = idx[b * max(st // nb, 1)::st]
        k2 = k2[lab[k2] < 2]
        v = np.load(_path(n, swapped, tr(n), h, prof))
        X = v["X"]
        if X.shape[1] != r[1].shape[1] or not np.array_equal(X[keep[0]], r[1][0], equal_nan=True):
            raise RuntimeError("pipe_lgb_bags: cached view of %s does not match its rows" % n)
        Xs_.append(np.ascontiguousarray(X[k2]))
        del X, v
        ys_.append(lab[k2])
        ss_.append(bsw[k2])
        if spw:
            full = _spell_weight_full(n, lk, spw)
            ws_.append(full[k2])
    return (np.concatenate(Xs_), np.concatenate(ys_), np.concatenate(ss_),
            np.concatenate(ws_) if spw else None)


def predict_lgb(models, X, Xs):
    if not models.get("team_soft"):
        p = train_v5.predict(models, X=X, Xs=Xs)
        if models.get("team_bags"):
            ts = [p["team"]] + [(m.predict_proba(X)[:, 1] + 1 - m.predict_proba(Xs)[:, 1]) / 2
                                for m in models["team_bags"]]
            ss = [p["sw"]] + [np.clip((m.predict(X) + m.predict(Xs)) / 2, 0, 1) for m in models["sw_bags"]]
            p["team"], p["sw"] = np.mean(ts, 0), np.mean(ss, 0)
        return p
    p = {}
    p["dead"] = (models["dead"].predict_proba(X)[:, 1] + models["dead"].predict_proba(Xs)[:, 1]) / 2
    for k in ("sw", "on", "off"):
        p[k] = np.clip((models[k].predict(X) + models[k].predict(Xs)) / 2, 0, 1)
    p["team"] = np.clip((models["team"].predict(X) + 1 - models["team"].predict(Xs)) / 2, 0, 1)
    return p


def _loss_spell_w(z5, t, m, U, lam, o, cw=None):
    import torch
    import torch.nn.functional as Fn
    z = torch.einsum("blc,lt->btc", z5, U)
    y = t[:, 0]
    valid = y < 2.5
    alive = y < 1.5
    md = valid & m[:, 0:1]
    if cw is not None:
        md = md * torch.where(y > 1.5, cw[:, 0:1], cw[:, 1:2])
    mt = alive & m[:, 1:2]
    w = torch.ones_like(y)
    for b in range(y.shape[0]):
        _, inv, cnt = torch.unique_consecutive(y[b], return_inverse=True, return_counts=True)
        w[b] = cnt[inv].float()
    w = ((float(o.get("ref", 250)) / w) ** float(o.get("alpha", 0.5))).clamp(float(o.get("min", 0.5)),
                                                                                 float(o.get("max", 3.0)))
    wt = w * mt

    def bce(zz, tt, mask):
        l = Fn.binary_cross_entropy_with_logits(zz, tt, reduction="none")
        return (l * mask).sum() / mask.sum().clamp(min=1)
    parts = {"dead": bce(z[..., 0], (y > 1.5).float(), md),
             "team": bce(z[..., 1], (y < 0.5).float(), wt),
             "sw": bce(z[..., 2], t[:, 1], wt),
             "on": bce(z[..., 3], t[:, 2], md),
             "off": bce(z[..., 4], t[:, 3], md)}
    zz = z5[..., :2]
    lp = torch.stack([Fn.logsigmoid(zz), Fn.logsigmoid(-zz)], -1)
    d = (lp[:, 1:] - lp[:, :-1].detach()) ** 2
    parts["tmse"] = d.clamp(max=16).mean()
    total = sum(v for k, v in parts.items() if k != "tmse") + lam * parts["tmse"]
    return total, parts


def _sample_batch_blind(segs, rng, B, p_dead, share, device, want_cw=False):
    import torch
    L, POOL = nn_seq.L, nn_seq.POOL
    nat = [i for i, s in enumerate(segs) if s["dead"] and not s.get("blind")]
    bli = [i for i, s in enumerate(segs) if s.get("blind")]
    w = np.array([s["nb"] for s in segs], float) * np.array([s["team"] for s in segs])
    w /= w.sum()
    xs, ts, ms, cws = [], [], [], []
    for b in range(B):
        if rng.random() < p_dead:
            pool = bli if rng.random() < share else nat
            i = pool[rng.integers(len(pool))]
        else:
            i = int(rng.choice(len(segs), p=w))
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
        cws.append(s.get("cw", (1.0, 1.0)))
    m = torch.tensor(ms, dtype=torch.bool, device=device)
    if want_cw:
        return torch.stack(xs).float(), torch.stack(ts).float(), m, torch.tensor(cws, dtype=torch.float32)
    return torch.stack(xs).float(), torch.stack(ts).float(), m


def _train_nn(segs, F, seed, a, log=print, spell_w=None, blind_share=None):
    import time
    import torch
    nn_seq.set_determinism(seed)
    dev = torch.device("cuda")
    cpu = torch.device("cpu")
    model = nn_seq.make_model(F, H=a.hidden, layers=a.layers, in_drop=a.in_drop, arch=a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
    U = torch.tensor(nn_seq.upsample_matrix(nn_seq.L, nn_seq.L * nn_seq.POOL), device=dev)
    rng = np.random.default_rng(seed)
    t0, run = time.time(), {}
    use_cw = blind_share is not None and any("cw" in s_ for s_ in segs)
    if use_cw and not spell_w:
        raise RuntimeError("nn_present needs pipe_nn_spell_w (weighted loss path)")
    for it in range(a.iters):
        model.train()
        if use_cw:
            pass
        elif blind_share is None:
            x, t, m = nn_seq.sample_batch(segs, rng, a.batch, a.p_dead, U, cpu)
        else:
            x, t, m = _sample_batch_blind(segs, rng, a.batch, a.p_dead, blind_share, cpu)
        cw = None
        if use_cw:
            x, t, m, cw = _sample_batch_blind(segs, rng, a.batch, a.p_dead, blind_share, cpu, want_cw=True)
            cw = cw.to(dev)
        x, t, m = x.to(dev), t.to(dev), m.to(dev)
        if spell_w:
            loss, parts = _loss_spell_w(model(x), t, m, U, a.tmse, spell_w, cw=cw)
        else:
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


def fit(rows, dead_set, team_set):
    import time
    import torch
    torch.set_num_threads(4)
    opts = _opts()
    c = pipe_ens._ctx()
    pipe_ens._check_rows(rows)
    models = {"lgb": fit_lgb(rows, dead_set, team_set, opts), "opts": opts}
    t = time.time()
    names = sorted({r[0] for r in rows})
    tr = [(n, sw) for n in names for sw in (False, True)]
    view = lambda n, sw, k=0: pipe_ens._pooled(n, sw, pipe_ens._train_seed(n) + 100000 * k)
    mu, sd = nn_seq.norm_stats(view(n, sw) for n, sw in tr)
    lab = pipe_ens._row_labels(rows)
    n_aug = int(opts.get("pipe_nn_aug", 0) or 0)
    if n_aug and c["prof"] == "default":
        raise RuntimeError("pipe_nn_aug needs a compliant degradation profile (split v3)")

    def train_view(n, sw, k=0):
        segs_, y, half = view(n, sw, k)
        if n in lab:
            y = pipe_ens._swap(lab[n]) if sw else lab[n]
        return segs_, y, half, n in dead_set, n in team_set
    segs = nn_seq.build_train((train_view(n, sw, k) for k in range(n_aug + 1) for n, sw in tr), mu, sd,
                              torch.device("cpu"))
    bl = opts.get("pipe_dead_blind") or None
    blind_share = None
    if bl and float(bl.get("nn_share", 0.5)) > 0:
        from mtah import blind
        fn = _feature_names()
        srcs = set(bl.get("sources", ["pff"]))

        def blind_view(n, sw):
            segs_, y, half, _, _ = train_view(n, sw)
            return [(h, a_, k, blind.blind_pooled(P, fn)) for h, a_, k, P in segs_], y, half, True, False
        nb0 = len(segs)
        segs += nn_seq.build_train((blind_view(n, sw) for n, sw in tr
                                    if n.split("_")[0] in srcs and n not in dead_set), mu, sd, torch.device("cpu"))
        for s_ in segs[nb0:]:
            s_["blind"] = True
        blind_share = float(bl.get("nn_share", 0.5))
        print("  [pipe_ens2] dead_blind nn: %d blind segments (%s), share %.2f" % (len(segs) - nb0, sorted(srcs),
                                                                                  blind_share), flush=True)
    if bl and float(bl.get("nn_present", 0) or 0) > 0:
        if blind_share is not None:
            raise RuntimeError("nn_present and nn_share are exclusive")
        srcs = set(bl.get("sources", ["pff"]))
        Fn_ = len(_feature_names())
        nb0 = len(segs)
        for n, sw in tr:
            if n.split("_")[0] not in srcs or n in dead_set:
                continue
            segs_, y, half, _, _ = train_view(n, sw)
            ym = np.full(len(y), 3, y.dtype)
            for h, a_, k, P in segs_:
                pres = np.repeat(P[:, Fn_] == 0, nn_seq.POOL)[:k]
                ym[a_:a_ + k] = np.where(pres, y[a_:a_ + k], 3)
            ok = (y < 3) & (half > 0)
            pa, pp = (y[ok] == 2).mean(), (ym[ym < 3] == 2).mean() if (ym < 3).any() else 0.0
            if not (0 < pp < 1 and 0 < pa < 1):
                continue
            new = nn_seq.build_train([(segs_, ym, half, True, False)], mu, sd, torch.device("cpu"))
            for s_ in new:
                s_["blind"] = True
                s_["cw"] = (float(pa / pp), float((1 - pa) / (1 - pp)))
            segs += new
        blind_share = float(bl["nn_present"])
        print("  [pipe_ens2] dead nn_present: %d ball-present segments (%s), share %.2f" % (len(segs) - nb0,
                                                                                       sorted(srcs), blind_share), flush=True)
    F = segs[0]["x"].shape[1]
    a = _nn_args(opts)
    n_seeds = int(opts.get("pipe_nn_seeds", 1) or 1)
    print("  [pipe_ens2] nn: %d train views, %d segments, F=%d, prep %.0fs, seeds %d, opts %s"
          % (len(tr), len(segs), F, time.time() - t, n_seeds, opts), flush=True)
    torch.cuda.empty_cache()
    base = 100 * int(PARAMS.get("random_state", 0) or 0)
    nets = []
    archs = opts.get("pipe_nn_archs") or [a.arch]
    for ai, arch in enumerate(archs):
        a.arch = arch
        for k in range(n_seeds):
            seed_k = base + 1 + k + 10 * ai
            nets.append(_train_nn(segs, F, seed_k, a, spell_w=opts.get("pipe_nn_spell_w") or None,
                                  blind_share=blind_share))
            print("  [pipe_ens2] nn %s seed %d trained %.0fs (peak VRAM %.2f GB)"
                  % (arch, seed_k, time.time() - t, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    del segs
    models.update(nn=nets, mu=mu, sd=sd)
    nbl = opts.get("pipe_nn_blind") or None
    if nbl:
        from mtah import blind
        fn = _feature_names()
        srcs = set(nbl.get("sources", ["pff"]))

        def bview(n, sw):
            segs_, y, half, _, _ = train_view(n, sw)
            return [(h, a_, k, blind.blind_pooled(P, fn)) for h, a_, k, P in segs_], y, half, True, False
        nat = [(n, sw) for n, sw in tr if n in dead_set]
        ext = [(n, sw) for n, sw in tr if n.split("_")[0] in srcs and n not in dead_set]
        bsegs = nn_seq.build_train((bview(n, sw) for n, sw in nat), mu, sd, torch.device("cpu"))
        n_nat = len(bsegs)
        bsegs += nn_seq.build_train((bview(n, sw) for n, sw in ext), mu, sd, torch.device("cpu"))
        for s_ in bsegs[n_nat:]:
            s_["blind"] = True
        share = float(nbl.get("share", 0.5)) if len(bsegs) > n_nat else 0.0
        ab = _nn_args(opts)
        ab.p_dead = 1.0
        bn = []
        for k in range(int(nbl.get("seeds", 2))):
            bn.append(_train_nn(bsegs, F, base + 51 + k, ab, blind_share=share))
        print("  [pipe_ens2] blind dead net: %d native + %d extra blind segments (%s), share %.2f, %d seeds, %.0fs"
              % (n_nat, len(bsegs) - n_nat, sorted(srcs), share, len(bn), time.time() - t), flush=True)
        del bsegs
        models["nnb"] = bn
    return models


def predict(models, m=None, X=None, Xs=None, half=None, names=None):
    if X is None:
        raise ValueError("pipe_ens2.predict needs the feature views X / Xs")
    pl = predict_lgb(models["lgb"], X, Xs)
    if half is None:
        _, half, fnames = pipe_ens._identify(X)
    else:
        fnames = list(names)
    miss = [fnames.index(col) for col in nn_seq.MISS_COLS]
    mu, sd = models["mu"], models["sd"]
    so, ss = nn_seq.pool_view(X, half, miss), nn_seq.pool_view(Xs, half, miss)
    nets = models["nn"]
    qs = [nn_seq.combine(nn_seq.predict_segs(nn, so, mu, sd, len(X)), nn_seq.predict_segs(nn, ss, mu, sd, len(X)))
          for nn in nets]
    q = qs[0] if len(qs) == 1 else {k: np.mean([x[k] for x in qs], 0) for k in HEADS}
    p = {}
    for k in HEADS:
        p["lgb_" + k] = pl[k]
        p["nn_" + k] = q[k].astype(np.float64)
    p.update(_combine(p, DEFAULT_W, None, False))
    if models.get("nnb"):
        from mtah import blind
        bo = [(h, a_, k, blind.blind_pooled(P, fnames)) for h, a_, k, P in so]
        bs = [(h, a_, k, blind.blind_pooled(P, fnames)) for h, a_, k, P in ss]
        qb = [nn_seq.combine(nn_seq.predict_segs(nn, bo, mu, sd, len(X)), nn_seq.predict_segs(nn, bs, mu, sd, len(X)))
              for nn in models["nnb"]]
        for k in ("dead", "on", "off"):
            p["nnb_" + k] = np.mean([x[k] for x in qb], 0).astype(np.float64)
    return p


def _logit(x, eps=1e-6):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def _combine(p, w, clip, logit_avg):
    out = {}
    for k in HEADS:
        a, b = p["lgb_" + k], p["nn_" + k]
        wk = float(w.get(k, DEFAULT_W[k]))
        if clip is not None or logit_avg:
            la, lb = _logit(a), _logit(b)
            if clip is not None:
                la, lb = np.clip(la, -clip, clip), np.clip(lb, -clip, clip)
            if logit_avg:
                out[k] = 1 / (1 + np.exp(-((1 - wk) * la + wk * lb)))
                continue
            a, b = 1 / (1 + np.exp(-la)), 1 / (1 + np.exp(-lb))
        if wk == 0.0:
            out[k] = a
        elif wk == 1.0:
            out[k] = b
        elif wk == 0.5:
            out[k] = (a + b) / 2
        else:
            out[k] = (1 - wk) * a + wk * b
    return out


def _term(n, base, pb, beta, eps=1e-3):
    if pb is None or beta == 0:
        return np.full(n, base)
    pb = np.clip(pb, eps, 1 - eps)
    return np.maximum(base - beta * np.log(pb / (1 - pb)), 0.0)


def decode_heads(p, half, cfg):
    if "lgb_team" in p:
        w = dict(cfg.get("w_nn") or {})
        g = cfg.get("dead_gate")
        if g:
            ok = half > 0
            c = float(np.corrcoef(p["lgb_dead"][ok], p["nn_dead"][ok])[0, 1])
            lo, hi = float(g.get("lo", 0.7)), float(g.get("hi", 0.85))
            gate = float(np.clip((c - lo) / (hi - lo), 0.0, 1.0)) if np.isfinite(c) else 0.0
            for k in g.get("heads", ["dead"]):
                w[k] = float(g.get("w", 0.5)) * gate
        q = _combine(p, w, cfg.get("clip"), bool(cfg.get("logit_avg", False)))
        for k, wk in (cfg.get("w_nnb") or {}).items():
            q[k] = (1 - float(wk)) * q[k] + float(wk) * p["nnb_" + k]
    else:
        q = {k: p[k] for k in HEADS}
    team, dead = q["team"], q["dead"]
    tt, dt = float(cfg.get("team_temp", 1.0)), float(cfg.get("dead_temp", 1.0))
    if tt != 1.0:
        team = 1 / (1 + np.exp(-tt * _logit(team)))
    if dt != 1.0:
        dead = 1 / (1 + np.exp(-dt * _logit(dead)))
    prob = np.stack([(1 - dead) * team, (1 - dead) * (1 - team), dead], 1)
    lp = np.log(np.clip(prob, 1e-4, 1))
    beta = cfg["beta"]
    b_sw, b_on, b_off = (float(cfg.get(k, beta)) for k in ("beta_sw", "beta_on", "beta_off"))
    d_on, d_off = cfg["dead"], float(cfg.get("dead_off", cfg["dead"]))
    out = np.full(len(prob), 3, np.int8)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        n = len(idx)
        c = np.zeros((n, 3, 3))
        s = _term(n, cfg["switch"], q["sw"][idx], b_sw)
        on = _term(n, d_on, q["on"][idx], b_on)
        off = _term(n, d_off, q["off"][idx], b_off)
        c[:, 0, 1] = c[:, 1, 0] = s
        c[:, 0, 2] = c[:, 1, 2] = on
        c[:, 2, 0] = c[:, 2, 1] = off
        out[idx] = viterbi_tv(lp[idx], c)
    return out
