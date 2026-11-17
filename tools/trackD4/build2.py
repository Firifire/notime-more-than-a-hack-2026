import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feats as Fe  # noqa: E402
import build as Bd  # noqa: E402
from vision import track_players as TP  # noqa: E402

Fe.BASE_M = 3.0
OUT = os.path.join(Bd.OUT, "..", "canvas2")
TEAM = "--team" in sys.argv
if TEAM:
    OUT = os.path.join(Bd.OUT, "..", "canvas3")


def sim_team(name, n):
    from vision.eval_tracking import sim_atd, match_atd
    z = np.load(os.path.join(Bd.PACKS, name + ".npz"))
    per = []
    for k in range(n):
        m = z["gt_f"] == k
        per.append(dict(role=z["gt_role"][m], team=z["gt_team"][m], tid=z["gt_tid"][m], pitch=z["gt_xy"][m]))
    rng = np.random.default_rng(int(name[-3:]))
    atd = sim_atd(per, rng, n)
    f = z["det_f"]
    out = []
    for k in range(n):
        xy = z["det_xy"][f == k]
        mi = match_atd(xy, atd[k][0])
        out.append(np.where(mi >= 0, atd[k][1][np.maximum(mi, 0)], -1) if len(atd[k][1]) else np.full(len(xy), -1))
    return out
NLEV = 10


def maj_pur(tr, dg):
    m = {}
    for k, obs in tr.items():
        ids = np.array([dg[t][d] for t, d in obs])
        u, c = np.unique(ids, return_counts=True)
        m[k] = int(u[np.argmax(c)])
    return m


def level_set(trk, pieces, s, rng):
    p_of = {}
    for k, obs in pieces.items():
        for o in obs:
            p_of[o] = k
    out = {}
    nid = 0
    for k, obs in trk.items():
        cur = [obs[0]]
        for a, b in zip(obs[:-1], obs[1:]):
            if p_of.get(a) != p_of.get(b) and rng.random() < s:
                out[nid] = cur
                nid += 1
                cur = []
            cur.append(b)
        out[nid] = cur
        nid += 1
    return out


def job(path):
    res = pickle.load(open(path, "rb"))
    name = res["name"]
    split = name.split("_")[0]
    levels = [0.0, 1.0] + ([] if split == "valid" else list(np.random.default_rng(zlib.crc32(name.encode())).random(NLEV)))
    todo = [li for li in range(len(levels)) if not os.path.exists(os.path.join(OUT, "%s_L%d.pkl" % (name, li)))]
    if not todo:
        return name
    P = Bd.load(os.path.join(Bd.PACKS, name + ".npz"))
    trk = TP.run_tracker(np.arange(res["n"]), P["xy"], P["cf"], P["ft"], in_view=P["in_view"])
    trk = {int(k): sorted(v) for k, v in trk.items()}
    rng = np.random.default_rng(zlib.crc32(name.encode()) + 1)
    det_team = sim_team(name, res["n"]) if TEAM else None
    os.makedirs(OUT, exist_ok=True)
    for li in todo:
        s = levels[li]
        tr = level_set(trk, res["pieces"], s, rng)
        r2 = dict(res, pieces=tr)
        r2["maj"] = maj_pur(tr, res["det_gt"])
        r2["det_team"] = det_team
        rows, X = Fe.clip_canvas(r2)
        X["rows"] = [dict(id=r["id"], obs=r["obs"], t0=r["t0"], t1=r["t1"]) for r in rows]
        X["n"] = res["n"]
        X["level"] = s
        X["clip"] = name
        out = os.path.join(OUT, "%s_L%d.pkl" % (name, li))
        pickle.dump(X, open(out + ".tmp", "wb"))
        os.replace(out + ".tmp", out)
    return name


def main():
    paths = sorted(glob.glob(os.path.join(Bd.OUT, "*.pkl")))
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(job, paths):
            print(r, flush=True)


if __name__ == "__main__":
    main()
