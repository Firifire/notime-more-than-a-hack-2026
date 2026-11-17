import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.environ.get("MTAH_UD_VIEWS") or os.path.join(ROOT, "cache", "ud", "views")
STEP = 5
NP = 24
NF = 41
SOURCES = ("idsse-ctrl", "pff", "metrica", "skillcorner", "stv2")
DEAD_SOURCES = ("idsse-ctrl", "stv2")
SPLIT = "notes/avo/split_v4.json"


def views(sealed=False):
    from mtah import eval_ext as E
    from mtah import gs_build as B
    s, _ = E.load_split(SPLIT)
    excl = set(s.get("exclude") or [])
    out = set()
    for n in E.all_names():
        if n in excl or E.src_of(n) not in SOURCES:
            continue
        out.add((n, B.train_seed(n)))
    for n in s["dev"] + (s["sealed"] if sealed else []):
        out.add((n, B.eval_seed(n)))
    return sorted(out)


def path(n, seed):
    return os.path.join(OUT, "%s_s%d.npz" % (n, seed))


def build_view(args):
    n, seed = args
    p = path(n, seed)
    if os.path.exists(p):
        return p
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from mtah import gs_build as B, gamestate as gs, eval_ext as E
    s, _ = E.load_split(SPLIT)
    m = B._load(n)
    d = B._degrade(m, seed)
    v = view_from(d)
    idx, pslot = v["frame"], v["pslot"]
    T, N = len(idx), v["n_frames"]
    pf, obs = v["pf"], v["obs"]
    _write_targets(p, n, s, m, v)
    return p


def view_from(d):
    from mtah import gamestate as gs
    F, pres = gs.player_features(d, gk_mode="half")
    N, S = pres.shape
    P = d["players"][:, :, :2].astype(np.float32)
    V = np.zeros_like(P)
    V[5:-5] = (P[10:] - P[:-10]) * (25.0 / 10)
    V = np.nan_to_num(V)
    team = np.asarray(d["team"])
    sign = np.where(team == 0, 1.0, -1.0).astype(np.float32)
    obs = d["ball"][:, :2].astype(np.float32)
    idx = np.arange(0, N, STEP)
    T = len(idx)
    pf = np.zeros((T, NP, NF), np.float16)
    pslot = np.full((T, NP), -1, np.int16)
    for i, t in enumerate(idx):
        sl = np.flatnonzero(pres[t])[:NP]
        k = len(sl)
        if not k:
            continue
        pslot[i, :k] = sl
        db = np.hypot(P[t, sl, 0] - obs[t, 0], P[t, sl, 1] - obs[t, 1])
        db = np.where(np.isfinite(db), db, 99.0)
        pf[i, :k] = np.c_[np.nan_to_num(F[t, sl]), sign[sl], P[t, sl], V[t, sl], db]
    return {"pf": pf, "pslot": pslot, "obs": obs[idx], "half": np.asarray(d["half"])[idx].astype(np.int8),
            "frame": idx.astype(np.int32), "n_frames": N}


def _write_targets(p, n, s, m, v):
    from mtah import gamestate as gs, eval_ext as E
    idx, pslot, T = v["frame"], v["pslot"], len(v["frame"])
    lk = E.label_key(n, s)
    lab = np.asarray(m[lk] if lk in m else m["label"]).astype(np.int16)[idx]
    y_poss = np.where(lab <= 2, lab, -1).astype(np.int8)
    tgt = gs.carrier_target(m)[idx]
    y_car = np.full(T, -2, np.int16)
    y_car[tgt == -1] = NP
    for i in np.flatnonzero(tgt >= 0):
        j = np.flatnonzero(pslot[i] == tgt[i])
        if len(j):
            y_car[i] = j[0]
    y_ball = m["ball"][idx, :2].astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, pf=v["pf"], pslot=pslot, obs=v["obs"], half=v["half"], y_poss=y_poss, y_car=y_car,
                        y_ball=y_ball, frame=v["frame"], n_frames=v["n_frames"], dead_ok=E.src_of(n) in DEAD_SOURCES)
    os.replace(tmp, p)


def provided_view():
    from mtah.match import provided_match, BALL
    p = os.path.join(OUT, "provided_%s.npz" % BALL)
    if not os.path.exists(p):
        v = view_from(provided_match())
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(p + ".tmp.npz", **v)
        os.replace(p + ".tmp.npz", p)
    z = np.load(p)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sealed", action="store_true")
    a = ap.parse_args()
    vs = views(a.sealed)
    print("views", len(vs), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, p in enumerate(ex.map(build_view, vs)):
            if i % 20 == 0:
                print(i, os.path.basename(p), flush=True)
    print("done")


if __name__ == "__main__":
    main()
