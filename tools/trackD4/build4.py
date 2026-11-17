import glob
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = list(sys.argv)
sys.argv = [sys.argv[0], "--team"]
import build as Bd  # noqa: E402
import build2 as B2  # noqa: E402
sys.argv = _argv
import feats as Fe  # noqa: E402
import k2track as K2  # noqa: E402
from vision import track_players as TP  # noqa: E402

Fe.BASE_M = 3.0
BASE = os.path.dirname(Bd.OUT)
EMBON = os.environ.get("D4_EMB") == "1"
OUT_T = os.path.join(BASE, "tracklets_deg" + ("7" if EMBON else ""))
OUT_C = os.path.join(BASE, "canvas4" + ("v7" if EMBON else ""))
NV = {"train": 3, "valid": 1}


def draw(rng):
    return dict(h_lo=20.0, h_hi=60.0, conf=rng.uniform(0.75, 1.0), miss=rng.uniform(0.0, 0.25),
                run=rng.uniform(2, 25), sxy=rng.uniform(0.05, 0.5))


def degrade(P, dg, prm, rng):
    n = P["n"]
    p_end = 1.0 / prm["run"]
    p_start = p_end * prm["miss"] / max(1 - prm["miss"], 1e-6)
    state = {}
    keep_all, out = [], dict(xy=[], cf=[], ft=[], bx=[])
    for t in range(n):
        m = len(P["xy"][t])
        keep = np.ones(m, bool)
        for d in range(m):
            key = ("g", int(dg[t][d])) if dg[t][d] >= 0 else None
            miss = state.get(key, False) if key else False
            miss = (rng.random() > p_end) if miss else (rng.random() < p_start)
            if key:
                state[key] = miss
            keep[d] = not miss
        keep_all.append(keep)
        k = np.flatnonzero(keep)
        h = rng.uniform(prm["h_lo"], prm["h_hi"], len(k))
        npx = np.maximum((0.072 * h * h).astype(int), 4)
        ft = P["ft"][t][k].astype(np.float64)
        f2 = np.zeros_like(ft)
        for i in range(len(k)):
            s = ft[i].sum()
            if s > 0:
                f2[i] = rng.multinomial(npx[i], ft[i] / s) / npx[i]
        out["xy"].append(P["xy"][t][k] + rng.normal(0, prm["sxy"], (len(k), 2)))
        out["cf"].append(P["cf"][t][k] * prm["conf"])
        out["ft"].append(f2.astype(np.float32))
        out["bx"].append(P["bx"][t][k])
    return out, keep_all


def shorten(pieces, rng):
    out, nid = {}, 0
    every = rng.uniform(50, 400)
    for k, obs in pieces.items():
        cur = [obs[0]]
        for o in obs[1:]:
            if rng.random() < 1.0 / every:
                out[nid] = cur
                nid += 1
                cur = []
            cur.append(o)
        out[nid] = cur
        nid += 1
    return out


def fb_rows(rows, k2ids):
    p_of = {}
    for i, r in enumerate(rows):
        for o in r["obs"]:
            p_of[o] = i
    fb = {}
    for _, obs in k2ids.items():
        seq = []
        for o in sorted(obs):
            i = p_of.get(o)
            if i is not None and (not seq or seq[-1] != i):
                seq.append(i)
        for a, b in zip(seq[:-1], seq[1:]):
            if rows[a]["t1"] < rows[b]["t0"]:
                fb[b] = a
    return fb


def job(args):
    path, v = args
    res0 = pickle.load(open(path, "rb"))
    name = res0["name"]
    split = name.split("_")[0]
    tag = "%s_v%d" % (name, v)
    sets = ["P", "L0"] + (["PS", "R"] if split == "train" else [])
    if all(os.path.exists(os.path.join(OUT_C, "%s_%s.pkl" % (tag, s))) for s in sets):
        return tag
    rng = np.random.default_rng(zlib.crc32(tag.encode()))
    prm = draw(rng)
    P = Bd.load(os.path.join(Bd.PACKS, name + ".npz"))
    D, keep = degrade(P, res0["det_gt"], prm, rng)
    n = P["n"]
    frames = np.arange(n)
    dg = [g[k] for g, k in zip(res0["det_gt"], keep)]
    dt_full = B2.sim_team(name, n)
    dt = [x[k] for x, k in zip(dt_full, keep)]
    tr = TP.run_tracker(frames, D["xy"], D["cf"], D["ft"], in_view=P["in_view"])
    tr = {int(k): sorted(v_) for k, v_ in tr.items()}
    ours = TP.stitch(tr, frames, D["xy"], D["ft"])
    embl = Fe.load_emb(name, P["xy"], keep) if EMBON else None
    k2ids, pieces = K2.track_v2(frames, D["xy"], D["cf"], D["ft"], P["in_view"], dt, tracklets=tr, return_pieces=True,
                                emb=embl)
    pieces = {int(k): sorted(v_) for k, v_ in pieces.items()}
    res = dict(name=tag, n=n, xy=D["xy"], cf=D["cf"], ft=D["ft"], bx=D["bx"], gt=P["gt"], det_gt=dg,
               ours={int(k): sorted(v_) for k, v_ in ours.items()},
               k2={int(k): sorted(v_) for k, v_ in k2ids.items()}, pieces=pieces, prm=prm)
    os.makedirs(OUT_T, exist_ok=True)
    os.makedirs(OUT_C, exist_ok=True)
    pickle.dump(res, open(os.path.join(OUT_T, tag + ".pkl"), "wb"))
    for s in sets:
        if s == "P":
            trs = pieces
        elif s == "L0":
            trs = tr
        elif s == "PS":
            trs = shorten(pieces, rng)
        else:
            trs = B2.level_set(tr, pieces, rng.random(), rng)
        r2 = dict(res, pieces=trs, k2=res["ours"], det_team=dt, embl=embl)
        r2["maj"] = B2.maj_pur(trs, dg)
        rows, X = Fe.clip_canvas(r2)
        X["rows"] = [dict(id=r["id"], obs=r["obs"], t0=r["t0"], t1=r["t1"]) for r in rows]
        X["n"] = n
        X["set"] = s
        X["clip"] = tag
        X["trk_path"] = os.path.join(OUT_T, tag + ".pkl")
        if s == "P":
            X["fb"] = fb_rows(rows, k2ids)
        out = os.path.join(OUT_C, "%s_%s.pkl" % (tag, s))
        pickle.dump(X, open(out + ".tmp", "wb"))
        os.replace(out + ".tmp", out)
    return tag


def main():
    jobs = []
    for p in sorted(glob.glob(os.path.join(Bd.OUT, "*.pkl"))):
        split = os.path.basename(p).split("_")[0]
        if os.environ.get("D4_SPLIT") and split != os.environ["D4_SPLIT"]:
            continue
        jobs += [(p, v) for v in range(NV[split])]
    jobs.sort(key=lambda j: 0 if "valid_" in j[0] else 1)
    with ProcessPoolExecutor(int(os.environ.get("D4_WORKERS", 4))) as ex:
        for r in ex.map(job, jobs):
            print(r, flush=True)


if __name__ == "__main__":
    main()
