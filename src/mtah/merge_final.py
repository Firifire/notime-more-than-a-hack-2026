import argparse
import importlib
import json
import os

import numpy as np

from mtah.data import HALVES, write_submission

HEADS = ("dead", "team", "sw", "on", "off")


def member_configs(cfg):
    return {k: json.load(open(v)) for k, v in cfg["members"].items()}


def member_heads(path):
    z = np.load(path)
    p = {k: z["head_" + k].astype(np.float64) for k in HEADS}
    p.update({k: z[k].astype(np.float64) for k in z.files if k.startswith(("lgb_", "nn_", "gs_", "nnb_"))})
    return p


def capture(pipeline, p, half, dec):
    from mtah import pipe_ens2
    orig, got = pipe_ens2.decode_heads, {}

    def hook(q, h, c):
        if "lgb_team" in q:
            raise RuntimeError("merge_final: the base pipeline passes member heads to pipe_ens2.decode_heads; "
                               "only pipelines that combine their members first are supported")
        got["q"] = {k: np.asarray(q[k], np.float64) for k in HEADS}
        got["dec"] = dict(c)
        return orig(got["q"], h, c)

    pipe_ens2.decode_heads = hook
    try:
        importlib.import_module(pipeline).decode_heads(dict(p), half, dec)
    finally:
        pipe_ens2.decode_heads = orig
    return got["q"], got["dec"]


def source_path(path, src):
    if not src:
        return path
    if not path.endswith(".signals.npz"):
        raise SystemExit("merge_final sources: %s does not end in .signals.npz" % path)
    return path[:-len(".signals.npz")] + ".%s.signals.npz" % src


def mix_sources(sp, cap, path, is_base):
    Q = {s: cap(source_path(path, s)) for s in dict.fromkeys(s for s, _ in sp["terms"])}
    rest = Q[sp["terms"][0][0]][1]
    if any(Q[s][1] != rest for s in Q):
        raise RuntimeError("merge_final sources: residual Viterbi options differ between sources")
    q = {}
    for k in HEADS:
        acc = None
        for s, c in sp["terms"]:
            t = float(c) * Q[s][0][k]
            acc = t if acc is None else acc + t
        q[k] = acc
    if is_base and sp.get("dead_from") is not None:
        q["dead"] = Q[sp["dead_from"]][0]["dead"]
    return q, rest


def decode(cfg, sig_of, half, dec=None):
    from mtah import pipe_ens2
    M = member_configs(cfg)
    mg = cfg["merge"]
    b, x, w = mg["base"], mg["X"], float(mg["w"])
    sp = cfg.get("sources")
    if sp:
        if cfg.get("mix") or not mg.get("X_capture"):
            raise SystemExit("merge_final sources: only for X_capture merges without mix")
        q, rest = mix_sources(sp, lambda f: capture(M[b]["pipeline"], member_heads(f), half, dec or cfg["decoder"]),
                              sig_of(M[b]["name"]), True)
        qx, _ = mix_sources(sp, lambda f: capture(M[x]["pipeline"], member_heads(f), half, M[x]["decoder"]),
                            sig_of(M[x]["name"]), False)
    else:
        q, rest = capture(M[b]["pipeline"], member_heads(sig_of(M[b]["name"])), half, dec or cfg["decoder"])
        px = member_heads(sig_of(M[x]["name"]))
        if mg.get("X_capture"):
            qx, _ = capture(M[x]["pipeline"], px, half, M[x]["decoder"])
        else:
            qx = {k: v for k, v in px.items() if k in HEADS}
    mx = cfg.get("mix")
    if mx:
        Q = {b: {k: np.asarray(v, np.float64) for k, v in q.items()}, x: qx}
    q = dict(q)
    for k in mg["heads"]:
        q[k] = (1 - w) * q[k] + w * qx[k]
    if mx:
        from mtah import stack_final
        for m, mc in M.items():
            if m not in Q:
                Q[m], _ = stack_final.capture(mc, stack_final.signals_p(sig_of(mc["name"])), half)
        for k in mx["heads"]:
            acc = None
            for m, c in mx["terms"]:
                t = float(c) * Q[m][k]
                acc = t if acc is None else acc + t
            q[k] = acc
    from mtah import tag_layer
    with tag_layer.patched(tag_layer.spec_of(cfg)):
        return np.asarray(pipe_ens2.decode_heads(q, half, rest))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--signals", default="runs/final/{name}.signals.npz", help="member signals path template")
    ap.add_argument("--decoder", default=None, help="base decoder JSON (default: the merge config's)")
    ap.add_argument("--tag-layer", default=None, help='JSON "tag_layer" spec (mtah.tag_layer); overrides the config key')
    ap.add_argument("--sources", default=None, help='JSON "sources" spec (player-source mixture); overrides the config key')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.tag_layer:
        cfg["tag_layer"] = json.loads(a.tag_layer)
    if a.sources:
        cfg["sources"] = json.loads(a.sources)
    sig_of = lambda name: a.signals.format(name=name)   # noqa: E731
    first = sig_of(next(iter(member_configs(cfg).values()))["name"])
    n = len(np.load(first)["p_dead"])
    half = np.zeros(n, np.int8)
    for h, s, e in HALVES:
        half[s:e + 1] = h
    lab = decode(cfg, sig_of, half, json.loads(a.decoder) if a.decoder else None)
    path = write_submission(a.out, lab)
    hm = lab[half > 0]
    print("wrote", path, "| merge %s%s%s | label share H/A/D/X %s | label runs in halves %d" % (
        json.dumps(cfg["merge"]), (" | mix " + json.dumps(cfg["mix"])) if cfg.get("mix") else "",
        (" | sources " + json.dumps(cfg["sources"])) if cfg.get("sources") else "", np.round(np.bincount(hm, minlength=4) / len(hm), 3).tolist(),
        int(np.sum(np.diff(hm) != 0)) + 1))


if __name__ == "__main__":
    main()
