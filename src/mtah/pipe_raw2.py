import numpy as np

from mtah import pipe_ens2, pipe_raw
from mtah.pipe_raw import PARAMS, boundary_labels, HEADS, fit, predict  # noqa: F401


def decode_heads(p, half, cfg):
    wb = cfg.get("w_nnb") or {}
    if not wb or "lgb_team" not in p:
        return pipe_raw.decode_heads(p, half, {k: v for k, v in cfg.items() if k != "w_nnb"})
    q = pipe_raw._base_heads(p, half, cfg)
    for k, wk in wb.items():
        q[k] = (1 - float(wk)) * q[k] + float(wk) * p["nnb_" + k]
    w = dict(cfg.get("w_raw") or {})
    g = cfg.get("raw_gate")
    if g:
        ok = half > 0
        c = float(np.corrcoef(p["lgb_dead"][ok], p["nn_raw_dead"][ok])[0, 1])
        lo, hi = float(g.get("lo", 0.7)), float(g.get("hi", 0.85))
        gate = float(np.clip((c - lo) / (hi - lo), 0.0, 1.0)) if np.isfinite(c) else 0.0
        for k in g.get("heads", ["dead", "on", "off"]):
            w[k] = float(g.get("w", 1 / 3)) * gate
    for k in HEADS:
        wk = float(w.get(k, 0.0))
        if wk:
            q[k] = (1 - wk) * q[k] + wk * p["nn_raw_" + k]
    rest = {k: v for k, v in cfg.items()
            if k not in ("w_raw", "raw_gate", "w_nn", "dead_gate", "clip", "logit_avg", "w_nnb")}
    return pipe_ens2.decode_heads(q, half, rest)
