import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv_saved = list(sys.argv)
sys.argv = [sys.argv[0], "--team"]
import build as Bd  # noqa: E402
import build2 as B2  # noqa: E402
sys.argv = sys.argv_saved
import feats as Fe  # noqa: E402
import linker as Lk  # noqa: E402
import match as M  # noqa: E402
import score as S  # noqa: E402
import k2track as K2  # noqa: E402
from vision import track_players as TP  # noqa: E402


def chain_fallback(k2ids, pieces):
    p_of = {o: k for k, obs in pieces.items() for o in obs}
    fb = {}
    for _, obs in k2ids.items():
        ps = []
        for o in sorted(obs):
            p = p_of.get(o)
            if p is not None and (not ps or ps[-1] != p):
                ps.append(p)
        for a, b in zip(ps[:-1], ps[1:]):
            fb[b] = a
    return fb


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="d4e")
    ap.add_argument("--seqs", type=int, default=0)
    ap.add_argument("--k2emb", action="store_true", help="K2 baseline with its re-ID embedding (v7)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(os.path.join(Lk.RUNS, a.name, "model.pt"), map_location=dev, weights_only=False)
    ar = st["args"]
    classic, team = not ar.get("no_classic", False), ar.get("team", False)
    Lk.EMB["on"] = ar.get("emb", False)
    k2emb = a.k2emb or Lk.EMB["on"]
    model = Lk.Linker(ar["d"], ar["layers"], ar["heads"], ar["drop"], team).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    Fe.BASE_M = 3.0
    paths = sorted(glob.glob(os.path.join(Bd.OUT, "valid_*.pkl")))
    if a.seqs:
        paths = paths[:a.seqs]
    per = {}
    for p in paths:
        res = pickle.load(open(p, "rb"))
        name, n = res["name"], res["n"]
        P = Bd.load(os.path.join(Bd.PACKS, name + ".npz"))
        frames = np.arange(n)
        tr = TP.run_tracker(frames, P["xy"], P["cf"], P["ft"], in_view=P["in_view"])
        dt = B2.sim_team(name, n)
        embl = Fe.load_emb(name, P["xy"]) if k2emb else None
        k2ids, pieces = K2.track_v2(frames, P["xy"], P["cf"], P["ft"], P["in_view"], dt, tracklets=tr,
                                    return_pieces=True, emb=embl)
        pieces = {k: sorted(v) for k, v in pieces.items()}
        ours = TP.stitch({k: sorted(v) for k, v in tr.items()}, frames, P["xy"], P["ft"])
        fb = chain_fallback(k2ids, pieces)
        tv = {"k2": k2ids, "ours": res["ours"]}
        for tag, f, vt in (("d4", None, False), ("d4_fbk2", fb, False), ("d4_fbk2_veto", fb, True)):
            ids, stt = M.link_windows(model, dev, "sampler", 24, classic, team, pieces, ours, ours, P["xy"], P["cf"],
                                      P["ft"], n, dt if team else None, fallback=f,
                                      embl=embl if Lk.EMB["on"] else None, veto=vt)
            tv[tag] = ids
        per[name] = {k: S.score(v, res, "v2") for k, v in tv.items()}
        per[name]["n_pieces"] = len(pieces)
        print(name, {k: round(v["HOTA"], 4) for k, v in per[name].items() if isinstance(v, dict)}, flush=True)
    names = sorted(per)
    out = {"summary": {}, "ci": {}}
    for v in ("ours", "k2", "d4", "d4_fbk2", "d4_fbk2_veto"):
        out["summary"][v] = {m: round(float(np.mean([per[c][v][m] for c in names])), 4)
                             for m in ("HOTA", "AssA", "IDF1", "IDsw", "pred_ids_per_gt_id")}
    for v in ("d4", "d4_fbk2", "d4_fbk2_veto"):
        for m in ("HOTA", "AssA", "IDF1"):
            out["ci"]["%s-k2/%s" % (v, m)] = [round(x, 4) for x in S.boot([per[c][v][m] - per[c]["k2"][m] for c in names])]
    out["pieces_per_clip"] = float(np.mean([per[c]["n_pieces"] for c in names]))
    print(json.dumps(out, indent=1))
    json.dump(dict(out, per_clip=per), open(os.path.join(Lk.RUNS, a.name, "gsr_compose%s.json" % a.tag), "w"), indent=1)


if __name__ == "__main__":
    main()
