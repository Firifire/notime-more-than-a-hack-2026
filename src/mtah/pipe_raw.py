import os

import numpy as np

from mtah import pipe_ens, pipe_ens2, raw_seq
from mtah.train_v5 import PARAMS, boundary_labels, HEADS  # noqa: F401  (PARAMS: same dict object as train_v5)


def _opts():
    return {k: v for k, v in PARAMS.items() if k.startswith("pipe_raw_")}


def _labels_for(rows):
    lab = pipe_ens._row_labels(rows)
    out = {}
    for n in sorted({r[0] for r in rows}):
        if n in lab:
            out[n] = lab[n]
        else:
            with np.load(os.path.join(pipe_ens._ctx()["ext"], n + ".npz")) as z:
                out[n] = z["label"]
    return out


def _half(n):
    with np.load(os.path.join(pipe_ens._ctx()["ext"], n + ".npz")) as z:
        return z["half"]


def fit_raw(names, labels, dead_set, team_set, o):
    import time
    import torch
    c = pipe_ens._ctx()
    a = raw_seq.Args(**(o.get("pipe_raw_args") or {}))
    K = int(o.get("pipe_raw_aug", 0) or 0)
    ns = int(o.get("pipe_raw_seeds", 1) or 1)
    t = time.time()
    keys = [(n, c["train_seed"](n) + 100000 * k) for n in names for k in range(K + 1)]
    paths = raw_seq.ensure_views(keys, workers=4)

    def items():
        for n in names:
            tgt = raw_seq.targets(labels[n], _half(n), a.spell_w)
            for k in range(K + 1):
                yield (raw_seq.load_view(paths[(n, c["train_seed"](n) + 100000 * k)])[0], tgt, n in dead_set, n in team_set,
                       n.split("_")[0] in a.blind_sources)
    segs = raw_seq.build_train(items())
    print("  [pipe_raw] raw: %d matches x %d degradation seeds, %d segments, prep %.0fs, args %s"
          % (len(names), K + 1, len(segs), time.time() - t, a.dict()), flush=True)
    nets = []
    base = 100 * int(PARAMS.get("random_state", 0) or 0)
    for s in range(ns):
        nets.append(raw_seq.train_model(segs, a, seed=base + 1 + s))
        print("  [pipe_raw] raw seed %d trained %.0fs (peak VRAM %.2f GB)"
              % (base + 1 + s, time.time() - t, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    del segs
    torch.cuda.empty_cache()
    return {"nets": nets, "args": a}


def fit(rows, dead_set, team_set):
    o = _opts()
    if o.get("pipe_raw_only"):
        pipe_ens._check_rows(rows)
        models = {}
    else:
        models = pipe_ens2.fit(rows, dead_set, team_set)
    names = sorted({r[0] for r in rows})
    models["raw"] = fit_raw(names, _labels_for(rows), dead_set, team_set, o)
    return models


def _raw_segs(X, half):
    if half is None:
        n, h, _ = pipe_ens._identify(X)
        seed = pipe_ens._ctx()["eval_seed"](n)
        p = raw_seq.ensure_views([(n, seed)], workers=1)[(n, seed)]
        return raw_seq.load_view(p)[0], h
    from mtah.match import provided_match
    m = provided_match()
    if len(m["half"]) != len(half) or not np.array_equal(m["half"], half):
        raise RuntimeError("pipe_raw: half= given but it is not the provided match's")
    return raw_seq.tokens_from_match(m), half


def predict(models, m=None, X=None, Xs=None, half=None, names=None):
    if X is None:
        raise ValueError("pipe_raw.predict needs the feature views X / Xs")
    p = {} if "lgb" not in models else pipe_ens2.predict(models, m=m, X=X, Xs=Xs, half=half, names=names)
    segs, h = _raw_segs(X, half)
    R = models["raw"]
    qs = [raw_seq.predict_view(net, segs, len(X), R["args"].bf16, L=R["args"].window) for net in R["nets"]]
    for k in HEADS:
        p["nn_raw_" + k] = np.mean([q[k] for q in qs], 0).astype(np.float64)
    if "lgb_team" not in p:
        for k in HEADS:
            p[k] = p["nn_raw_" + k]
    return p


def decode_heads(p, half, cfg):
    if cfg.get("rescore"):
        from mtah import spell_decoder
        return spell_decoder.decode(p, half, cfg)
    w = dict(cfg.get("w_raw") or {})
    if "lgb_team" not in p:
        q = {k: p["nn_raw_" + k] for k in HEADS}
        return pipe_ens2.decode_heads(q, half, {k: v for k, v in cfg.items() if k not in ("w_raw", "dead_gate", "raw_gate")})
    g = cfg.get("raw_gate")
    if g:
        ok = half > 0
        c = float(np.corrcoef(p["lgb_dead"][ok], p["nn_raw_dead"][ok])[0, 1])
        lo, hi = float(g.get("lo", 0.7)), float(g.get("hi", 0.85))
        gate = float(np.clip((c - lo) / (hi - lo), 0.0, 1.0)) if np.isfinite(c) else 0.0
        for k in g.get("heads", ["dead", "on", "off"]):
            w[k] = float(g.get("w", 1 / 3)) * gate
    if not any(float(v) for v in w.values()):
        return pipe_ens2.decode_heads(p, half, cfg)
    q = _base_heads(p, half, cfg)
    for k in HEADS:
        wk = float(w.get(k, 0.0))
        if wk:
            q[k] = (1 - wk) * q[k] + wk * p["nn_raw_" + k]
    rest = {k: v for k, v in cfg.items() if k not in ("w_raw", "raw_gate", "w_nn", "dead_gate", "clip", "logit_avg")}
    return pipe_ens2.decode_heads(q, half, rest)


def _base_heads(p, half, cfg):
    w = dict(cfg.get("w_nn") or {})
    g = cfg.get("dead_gate")
    if g:
        ok = half > 0
        c = float(np.corrcoef(p["lgb_dead"][ok], p["nn_dead"][ok])[0, 1])
        lo, hi = float(g.get("lo", 0.7)), float(g.get("hi", 0.85))
        gate = float(np.clip((c - lo) / (hi - lo), 0.0, 1.0)) if np.isfinite(c) else 0.0
        for k in g.get("heads", ["dead"]):
            w[k] = float(g.get("w", 0.5)) * gate
    return pipe_ens2._combine(p, w, cfg.get("clip"), bool(cfg.get("logit_avg", False)))
