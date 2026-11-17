import argparse
import importlib
import json
import os
import sys

import numpy as np

HEADS = ("dead", "team", "sw", "on", "off")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_SIT = ("b_pres", "b_spd", "b_spd_max25", "b_acc_max12", "b_still", "t_since_ball", "t_until_ball", "v_all",
           "v_all_m100", "v_all_rise", "v_all_drop", "v_p80", "min_corner_d", "min_centre_d", "n_beyond_touch",
           "n_beyond_goal", "touch_chain")


def expert_configs(cfg):
    return {k: json.load(open(os.path.join(ROOT, v["config"]))) for k, v in cfg["experts"].items()}


def capture(ecfg, p, half, dec=None):
    from mtah import pipe_ens2, pipe_bc
    dec = dec or ecfg["decoder"]
    pipe = importlib.import_module(ecfg["pipeline"])
    got = {}
    o_ens, o_bc = pipe_ens2.decode_heads, pipe_bc._decode_ens2

    def comb(q, c):
        w = dict(c.get("w_nn") or {})
        g = c.get("dead_gate")
        if g:
            ok = half > 0
            cc = float(np.corrcoef(q["lgb_dead"][ok], q["nn_dead"][ok])[0, 1])
            lo, hi = float(g.get("lo", 0.7)), float(g.get("hi", 0.85))
            gate = float(np.clip((cc - lo) / (hi - lo), 0.0, 1.0)) if np.isfinite(cc) else 0.0
            for k in g.get("heads", ["dead"]):
                w[k] = float(g.get("w", 0.5)) * gate
        qq = pipe_ens2._combine(q, w, c.get("clip"), bool(c.get("logit_avg", False)))
        for k, wk in (c.get("w_nnb") or {}).items():
            qq[k] = (1 - float(wk)) * qq[k] + float(wk) * q["nnb_" + k]
        for k in ("bc_llr_dead", "bc_llr_on", "bc_llr_off"):
            if k in q:
                qq[k] = q[k]
        return qq, {k: v for k, v in c.items() if k not in ("w_nn", "dead_gate", "clip", "logit_avg", "w_nnb")}

    def h_ens(q, h, c):
        if "lgb_team" in q:
            q, c = comb(q, c)
        got["q"] = {k: np.asarray(q[k], np.float64) for k in HEADS}
        got["dec"] = dict(c)
        return o_ens(got["q"], h, c)

    def h_bc(q, h, c):
        if "lgb_team" in q:
            q, c = comb(q, c)
        q = dict(q)
        bc = c.get("bc") or {}
        if bc:
            cl = float(bc.get("clip", 6.0))
            q["dead"] = pipe_bc._shift(q["dead"], q.get("bc_llr_dead"), float(bc.get("w", 1.0)), cl)
            q["on"] = pipe_bc._shift(q["on"], q.get("bc_llr_on"), float(bc.get("w_on", 0.0)), cl)
            q["off"] = pipe_bc._shift(q["off"], q.get("bc_llr_off"), float(bc.get("w_off", 0.0)), cl)
        got["q"] = {k: np.asarray(q[k], np.float64) for k in HEADS}
        got["dec"] = {k: v for k, v in c.items() if k != "bc"}
        return o_ens(got["q"], h, got["dec"])

    pipe_ens2.decode_heads, pipe_bc._decode_ens2 = h_ens, h_bc
    try:
        pipe.decode_heads(dict(p), half, dec)
    finally:
        pipe_ens2.decode_heads, pipe_bc._decode_ens2 = o_ens, o_bc
    return got["q"], got["dec"]


def heads_all(cfg, E_cfg, get_p, half):
    Q, D = {}, {}
    for e in cfg["order"]:
        p = get_p(e)
        plain = cfg["experts"][e].get("plain")
        if plain:
            Q[e], D[e] = {k: p[k] for k in HEADS}, dict(plain)
        else:
            Q[e], D[e] = capture(E_cfg[e], p, half)
    return Q, D


def load_oof(path, n):
    z = np.load(os.path.join(path, "oof_%s.npz" % n))
    return {k: z[k].astype(np.float64) for k in z.files if k not in ("label", "half", "bt_on", "bt_off")}


def signals_p(path):
    z = np.load(path)
    p = {k: z["head_" + k].astype(np.float64) for k in HEADS}
    p.update({k: z[k].astype(np.float64) for k in z.files if k.startswith(("lgb_", "nn_", "gs_", "nnb_"))})
    return p


def situation(X, nm):
    ix = {k: i for i, k in enumerate(nm)}
    g = lambda k: X[:, ix[k]].astype(np.float32)   # noqa: E731
    cols = [np.abs(g("b_x")), np.abs(g("b_y")), g("n_ga_left") + g("n_ga_right"), g("n_box_left") + g("n_box_right")]
    cols += [g(k) for k in RAW_SIT]
    return np.stack(cols, 1)


def logit(x):
    x = np.clip(np.asarray(x, np.float64), 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def design(order, Q, sit):
    return np.column_stack([logit(Q[e][h]) for e in order for h in HEADS] + [sit]).astype(np.float32)


def team_cols(order):
    return [i for i, (e, h) in enumerate((e, h) for e in order for h in HEADS) if h == "team"]


def swapped(F, tc):
    G = F.copy()
    G[:, tc] = -G[:, tc]
    return G


def dev_rows(cfg, E_cfg):
    from mtah import eval_ext as E
    from mtah.featcache import feature_paths, load_view
    S, _ = E.load_split(cfg["split_file"])
    _, ev = E.seed_fns(0)
    out = []
    for n in sorted(S["dev"]):
        y, half = E.truth(n, E.label_key(n, S))
        Q, _ = heads_all(cfg, E_cfg, lambda e: load_oof(os.path.join(ROOT, cfg["experts"][e]["oof"]), n), half)
        k = (n, False, ev(n))
        X, _, _, nm = load_view(feature_paths([k], workers=1)[k])
        out.append((n, y, half, design(cfg["order"], Q, situation(X, list(nm)))))
    return out


def fit(cfg, rows):
    import lightgbm as lgb
    from mtah import eval_ext as E
    tc = team_cols(cfg["order"])
    step = int(cfg["step"])
    Fd, yd, Ft, yt = [], [], [], []
    for n, y, half, F in rows:
        ok = (half > 0) & (y < 3)
        idx = np.flatnonzero(ok)[::step]
        if E.src_of(n) in set(cfg["dead_sources"]):
            Fd.append(F[idx])
            yd.append((y[idx] == 2).astype(np.int8))
        ia = idx[y[idx] < 2]
        Ft += [F[ia], swapped(F[ia], tc)]
        yt += [(y[ia] == 0).astype(np.int8), (y[ia] == 1).astype(np.int8)]
    md = lgb.LGBMClassifier(**cfg["lgb"]).fit(np.concatenate(Fd), np.concatenate(yd))
    mt = lgb.LGBMClassifier(**cfg["lgb"]).fit(np.concatenate(Ft), np.concatenate(yt))
    return md, mt


def predict(cfg, models, F):
    md, mt = models
    tc = team_cols(cfg["order"])
    pd = md.predict_proba(F)[:, 1]
    pt = 0.5 * (mt.predict_proba(F)[:, 1] + 1 - mt.predict_proba(swapped(F, tc))[:, 1])
    return pd, pt


def heads_of(variant, pd, pt, Q):
    return {"dead": pd, "team": pt if variant == "S" else Q["n11"]["team"], "sw": Q["n11"]["sw"],
            "on": Q["n4"]["on"], "off": Q["n4"]["off"]}


def decode(cfg, sig_of, half):
    from mtah import eval_ext as E, pipe_ens2
    from mtah.featset import build
    E_cfg = expert_configs(cfg)
    S, _ = E.load_split(cfg["split_file"])
    E.apply_profile(S)
    from mtah import ckpt
    fit_ck = ckpt.wrap("stack_final.fit", lambda c, r: fit(c, r), roots=["mtah.stack_final"])
    models = fit_ck(cfg, dev_rows(cfg, E_cfg))
    from mtah.match import provided_match
    m = provided_match()
    X, fn = build(m)
    Q, D = heads_all(cfg, E_cfg, lambda e: signals_p(sig_of(E_cfg[e]["name"])), half)
    pd, pt = predict(cfg, models, design(cfg["order"], Q, situation(X, list(fn))))
    from mtah import tag_layer
    with tag_layer.patched(tag_layer.spec_of(cfg)):
        return np.asarray(pipe_ens2.decode_heads(heads_of(cfg["variant"], pd, pt, Q), half, D["n4"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--signals", default="runs/final/{name}.signals.npz", help="expert signals path template")
    ap.add_argument("--decoder", default=None, help="ignored (the router uses #4's captured decoder); run.py passes it")
    ap.add_argument("--tag-layer", default=None, help='JSON "tag_layer" spec (mtah.tag_layer); overrides the config key')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.tag_layer:
        cfg["tag_layer"] = json.loads(a.tag_layer)
    os.environ["MTAH_BALL"] = "atd"
    from mtah.data import HALVES, write_submission
    sig_of = lambda name: a.signals.format(name=name)   # noqa: E731
    n = len(np.load(sig_of(expert_configs(cfg)[cfg["order"][0]]["name"]))["p_dead"])
    half = np.zeros(n, np.int8)
    for h, s, e in HALVES:
        half[s:e + 1] = h
    lab = decode(cfg, sig_of, half)
    path = write_submission(a.out, lab)
    hm = lab[half > 0]
    print("wrote", path, "| router %s | label share H/A/D/X %s | label runs in halves %d" % (
        cfg["variant"], np.round(np.bincount(hm, minlength=4) / len(hm), 3).tolist(), int(np.sum(np.diff(hm) != 0)) + 1))


if __name__ == "__main__":
    sys.exit(main())
