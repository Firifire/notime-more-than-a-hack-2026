import numpy as np

from mtah import bc_feats, pipe_ens, pipe_ens2
from mtah.decode import viterbi_tv
from mtah.pipe_ens2 import PARAMS, boundary_labels, HEADS, _combine, _term, _logit  # noqa: F401

BC_LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, deterministic=True,
              force_row_wise=True, n_jobs=6, random_state=0)
BC_HEADS = ("dead", "on", "off")


def _opts():
    return {k: v for k, v in PARAMS.items() if k.startswith("pipe_bc_")}


def bc_rows(names, step=5):
    from mtah.eval_ext import EXT
    import os
    Xs, ys = [], {k: [] for k in BC_HEADS}
    for n in names:
        z = np.load(os.path.join(EXT, n + ".npz"))
        lab, half = z["label"], z["half"]
        keep = np.flatnonzero((lab < 3) & (half > 0))[::step]
        if len(keep) == 0:
            continue
        X = bc_feats.cue_features(bc_feats.load_cues(n), half)
        _, on, off = boundary_labels(lab, half)
        Xs.append(X[keep])
        ys["dead"].append((lab[keep] == 2).astype(np.float64))
        ys["on"].append(on[keep].astype(np.float64))
        ys["off"].append(off[keep].astype(np.float64))
    return np.concatenate(Xs), {k: np.concatenate(v) for k, v in ys.items()}


def fit_bc(names, step=5, seed=0):
    import lightgbm as lgb
    X, ys = bc_rows(names, step)
    m = {"names": sorted(names), "n_rows": len(X)}
    for k in BC_HEADS:
        y = ys[k]
        m["prior_" + k] = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        P = dict(BC_LGB, random_state=seed)
        if k == "dead":
            m[k] = lgb.LGBMClassifier(**P).fit(X, y.astype(int))
        else:
            m[k] = lgb.LGBMRegressor(**dict(P, objective="cross_entropy")).fit(X, y)
    return m


def predict_bc(m, cues, half):
    X = bc_feats.cue_features(cues, half)
    ok = half > 0
    out = {}
    for k in BC_HEADS:
        p = np.full(len(half), np.nan)
        if ok.any():
            p[ok] = m[k].predict_proba(X[ok])[:, 1] if k == "dead" else m[k].predict(X[ok])
        out["bc_" + k] = p
        out["bc_llr_" + k] = _logit(p) - _logit(m["prior_" + k])
    return out


def _base():
    import importlib
    return importlib.import_module(str(_opts().get("pipe_bc_base", "mtah.pipe_ens2")))


def fit(rows, dead_set, team_set):
    models = _base().fit(rows, dead_set, team_set)
    o = _opts()
    srcs = set(o.get("pipe_bc_sources", ["pff", "skillcorner", "skillcorner2020"]))
    names = sorted({r[0] for r in rows if r[0].split("_")[0] in srcs and bc_feats.has_cues(r[0])})
    if names:
        models["bc"] = fit_bc(names, int(o.get("pipe_bc_step", 5)), int(PARAMS.get("random_state", 0) or 0))
        print("  [pipe_bc] cue model: %d matches (%s), %d rows" % (
            len(names), sorted({n.split("_")[0] for n in names}), models["bc"]["n_rows"]), flush=True)
    return models


def predict(models, m=None, X=None, Xs=None, half=None, names=None):
    p = _base().predict(models, m=m, X=X, Xs=Xs, half=half, names=names)
    if half is None:
        name, half_, _ = pipe_ens._identify(X)
        cues = bc_feats.load_cues(name) if bc_feats.has_cues(name) else None
    else:
        half_ = np.asarray(half)
        cues = bc_feats.load_cues("provided")
    if "bc" in models and cues is not None:
        p.update(predict_bc(models["bc"], cues, half_))
    else:
        for k in BC_HEADS:
            p["bc_" + k] = np.full(len(X), np.nan)
            p["bc_llr_" + k] = np.full(len(X), np.nan)
    return p


def _shift(prob, llr, w, clip):
    if w == 0 or llr is None:
        return prob
    ok = np.isfinite(llr)
    if not ok.any():
        return prob
    z = _logit(prob)
    z = np.where(ok, z + w * np.clip(np.nan_to_num(llr), -clip, clip), z)
    return 1 / (1 + np.exp(-z))


def decode_heads(p, half, cfg):
    if "nn_raw_team" in p and "lgb_team" in p:
        if "nnb_dead" in p:
            from mtah import pipe_raw2 as pipe_raw
        else:
            from mtah import pipe_raw
        bc = cfg.get("bc") or {}
        rest = {k: v for k, v in cfg.items() if k != "bc"}
        orig = pipe_ens2.decode_heads

        def hooked(q, half_, c_):
            if "lgb_team" in q:
                return _decode_ens2(q, half_, dict(c_, bc=bc))
            q = dict(q)
            for k in ("bc_llr_dead", "bc_llr_on", "bc_llr_off"):
                q[k] = p.get(k)
            return _decode_ens2(q, half_, dict(c_, bc=bc))
        pipe_ens2.decode_heads = hooked
        try:
            return pipe_raw.decode_heads(p, half, rest)
        finally:
            pipe_ens2.decode_heads = orig
    return _decode_ens2(p, half, cfg)


def _decode_ens2(p, half, cfg):
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
    else:
        q = {k: p[k] for k in HEADS}
    bc = cfg.get("bc") or {}
    if bc:
        cl = float(bc.get("clip", 6.0))
        q = dict(q)
        q["dead"] = _shift(q["dead"], p.get("bc_llr_dead"), float(bc.get("w", 1.0)), cl)
        q["on"] = _shift(q["on"], p.get("bc_llr_on"), float(bc.get("w_on", 0.0)), cl)
        q["off"] = _shift(q["off"], p.get("bc_llr_off"), float(bc.get("w_off", 0.0)), cl)
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
