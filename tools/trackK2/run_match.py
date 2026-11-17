"""Track K2: tracker v2 on the match -> players_video_v2.npz (same format as cache/vision/players_video.npz).

    PYTHONPATH=src .venv-vision/bin/python tools/trackK2/run_match.py [--tag v2]
-> /mnt/x/Data/AI/more-than-a-hack/trackK2/players_video_<tag>.npz (+ _report.json)
   /mnt/x/Data/AI/more-than-a-hack/trackK2/tracklets/match_half<h>.npz (pass-1 pure tracklets, for Track D4)

Inputs as src/vision/run_players.py track (unchanged detections cache/vision/players_cands, provided homography,
provided ATD for the per-detection team vote and the slot column). Differences from v1:
  - pass 1: v1 online tracker, then pure tracklets (split at detection gaps > 5 frames and at kit-colour changes);
  - pass 2: global linking (rounds of Hungarian end->start matching; k2track.py) instead of greedy stitching and
    instead of the ATD-slot continuity cue (the ATD carries its own identity swaps, Track T2);
  - team is a property of the linked identity (vote of the ATD matches of all its detections + kit colour, as
    TP.assign_teams), written on every frame: team_src 2 (vote) / 3
    (colour) / 4 (other); team_src 1 (per-frame copy of the matched ATD slot's team) is no longer produced;
  - interpolation inside gaps <= 125 frames (as v1), longer gaps keep the identity without positions.
No match label is read.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from vision.common import W, H, N_FRAMES, HALVES, load_homography, metres_to_px, px_to_metres, graded_mask, PITCH  # noqa
from vision.run_players import load_cands, MARGIN_M, MATCH_M, K, KBUF  # noqa: E402
from vision import track_players as TP  # noqa: E402
import k2track as K2  # noqa: E402

OUTD = "/mnt/x/Data/AI/more-than-a-hack/trackK2"


def save_tracklets(path, pieces, frames, xy, cf, ft, bx, emb=None):
    fpos = {t: i for i, t in enumerate(frames)}
    pid, T, D, X, C, B, kits, kid, embs = [], [], [], [], [], [], [], [], []
    for k, obs in pieces.items():
        obs = sorted(obs)
        for t, d in obs:
            i = fpos[t]
            pid.append(k)
            T.append(t)
            D.append(d)
            X.append(xy[i][d])
            C.append(cf[i][d])
            B.append(bx[i][d])
        fs = [ft[fpos[t]][d] for t, d in obs if ft[fpos[t]][d].sum() > 0]
        kits.append(np.mean(fs, 0) if fs else np.zeros(ft[0].shape[1] if len(ft[0]) else 48))
        kid.append(k)
        if emb is not None:
            es = [emb[fpos[t]][d] for t, d in obs if np.isfinite(emb[fpos[t]][d][0])]
            m = np.mean(es, 0) if es else np.full(emb[fpos[obs[0][0]]].shape[1], np.nan)
            embs.append(m / max(np.linalg.norm(m), 1e-9) if es else m)
    np.savez_compressed(path, piece=np.array(pid, np.int32), t=np.array(T, np.int32), det=np.array(D, np.int16),
                        xy=np.array(X, np.float32), conf=np.array(C, np.float32), box=np.array(B, np.float32),
                        kit_piece=np.array(kid, np.int32), kit=np.array(kits, np.float32),
                        **({"emb": np.array(embs, np.float32)} if emb is not None else {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--set", default="{}")
    ap.add_argument("--fill", type=int, default=125)
    ap.add_argument("--emb", default="", help="re-ID embedding tag (reid/match_emb<tag>.npz); empty = none")
    a = ap.parse_args()
    for k, v in json.loads(a.set).items():
        if k.startswith("x_"):
            setattr(K2.PX, k[2:], v)
        else:
            setattr(K2.P, k, tuple(v) if isinstance(v, list) else v)
    from mtah.data import load_tracking
    cands, processed = load_cands()
    Hs = load_homography()
    atd = load_tracking()
    P_atd, team_atd = atd["players"][..., :2].astype(np.float64), atd["team"].astype(int)
    fr_all = cands["frame"]
    bounds = np.searchsorted(fr_all, np.arange(N_FRAMES + 1))
    emb_all = None
    if a.emb:
        ez = np.load(os.path.join(OUTD, "reid", "match_emb%s.npz" % a.emb))
        emb_all = np.full((len(fr_all), ez["emb"].shape[1]), np.nan, np.float16)
        emb_all[ez["idx"]] = ez["emb"]
    out = dict(
        xy=np.full((N_FRAMES, KBUF, 2), np.nan, np.float32), team=np.full((N_FRAMES, KBUF), -1, np.int8),
        track_id=np.full((N_FRAMES, KBUF), -1, np.int32), conf=np.zeros((N_FRAMES, KBUF), np.float32),
        atd_slot=np.full((N_FRAMES, KBUF), -1, np.int16), state=np.zeros((N_FRAMES, KBUF), np.int8),
        team_src=np.zeros((N_FRAMES, KBUF), np.int8), uv=np.full((N_FRAMES, KBUF, 2), np.nan, np.float32),
        processed=np.zeros(N_FRAMES, bool))
    out["processed"][processed] = True
    report = {"params": {k: getattr(K2.P, k) for k in dir(K2.P) if not k.startswith("_")}, "fill": a.fill,
              "params_x": {k: getattr(K2.PX, k) for k in dir(K2.PX) if not k.startswith("_")}, "emb": a.emb}
    next_tid = 0
    os.makedirs(os.path.join(OUTD, "tracklets"), exist_ok=True)
    for half, ga, gb in HALVES:
        t0 = time.time()
        frames = np.arange(max(ga - 50, processed.min()), gb + 51)
        frames = frames[out["processed"][frames]]
        px = P_atd[ga:gb + 1, :, 0]
        gl = np.nanmedian(np.minimum(px, PITCH[0] - px), axis=0)
        gk_slot = (gl < 16.5) & (np.isfinite(px).sum(0) > 25 * 60)
        xy, cf, ft, bx, slot, tm, gkd, em = [], [], [], [], [], [], [], []
        for t in frames:
            s, e = bounds[t], bounds[t + 1]
            b = cands["box"][s:e].astype(np.float64)
            x, y = px_to_metres(np.linalg.inv(Hs[t]), (b[:, 0] + b[:, 2]) / 2, b[:, 3])
            ok = (np.isfinite(x) & (x > -MARGIN_M) & (x < PITCH[0] + MARGIN_M) & (y > -MARGIN_M)
                  & (y < PITCH[1] + MARGIN_M))
            p = np.c_[x[ok], y[ok]]
            xy.append(p)
            cf.append(cands["conf"][s:e][ok])
            ft.append(cands["feat"][s:e][ok].astype(np.float32))
            bx.append(b[ok])
            if emb_all is not None:
                em.append(emb_all[s:e][ok].astype(np.float32))
            ap_ = P_atd[t]
            av = np.flatnonzero(~np.isnan(ap_[:, 0]))
            sl = np.full(len(p), -1)
            if len(p) and len(av):
                D = np.hypot(*(p[:, None] - ap_[av][None]).transpose(2, 0, 1))
                r, c = linear_sum_assignment(np.where(D < MATCH_M, D, 1e6))
                k = D[r, c] < MATCH_M
                sl[r[k]] = av[c[k]]
            slot.append(sl)
            tm.append(np.where(sl >= 0, team_atd[np.maximum(sl, 0)], -1))
            gkd.append((sl >= 0) & gk_slot[np.maximum(sl, 0)])
        fpos = {t: i for i, t in enumerate(frames)}

        def in_view(t, pts):
            u, v = metres_to_px(Hs[t], pts[:, 0], pts[:, 1])
            return (u > 0) & (u < W) & (v > 0) & (v < H)
        _, clean = K2.frame_flags(frames, xy)
        tm_clean = [np.where(c_, x_, -1) for x_, c_ in zip(tm, clean)] if K2.P.team_clean else tm
        tr = TP.run_tracker(frames, xy, cf, ft, in_view=in_view)
        ids, pieces = K2.track_v2(frames, xy, cf, ft, in_view, tm_clean, tracklets=tr, return_pieces=True,
                                  emb=em if emb_all is not None else None)
        save_tracklets(os.path.join(OUTD, "tracklets", "match_half%d.npz" % half), pieces, frames, xy, cf, ft, bx,
                       emb=em if emb_all is not None else None)
        dense = K2.densify_v2(ids, frames, xy, bx, fill_max=a.fill)
        team, src = K2.identity_teams(dense, frames, tm_clean, ft, gkd, cf)
        print("half %d: online %d, pieces %d, identities %d (%.0fs)" % (half, len(tr), len(pieces), len(ids),
                                                                       time.time() - t0), flush=True)
        report["half%d" % half] = dict(online=len(tr), pieces=len(pieces), identities=len(ids))
        fill = np.zeros(N_FRAMES, np.int32)
        for k, r in dense.items():
            tid = next_tid
            next_tid += 1
            for t, p, st, d, b in zip(r["t"], r["xy"], r["state"], r["det"], r["box"]):
                j = fill[t]
                if j >= KBUF:
                    continue
                fill[t] += 1
                out["xy"][t, j] = p
                out["track_id"][t, j] = tid
                out["state"][t, j] = st
                out["uv"][t, j] = ((b[0] + b[2]) / 2, b[3])
                if d >= 0:
                    i = fpos[t]
                    out["conf"][t, j] = cf[i][d]
                    out["atd_slot"][t, j] = slot[i][d]
                if src[k] == 3:
                    out["team_src"][t, j] = 4
                elif team[k] >= 0:
                    out["team"][t, j] = team[k]
                    out["team_src"][t, j] = 2 if src[k] == 1 else 3
    pri = ((out["state"] > 0) * 8.0 + (out["team"] >= 0) * 4 + (out["state"] == 1) * 2 + out["conf"])
    order = np.argsort(-pri, axis=1, kind="stable")[:, :K]
    for key in ("xy", "uv"):
        out[key] = np.take_along_axis(out[key], order[..., None], 1)
    for key in ("team", "track_id", "conf", "atd_slot", "state", "team_src"):
        out[key] = np.take_along_axis(out[key], order, 1)
    out["doc"] = np.array(__doc__)
    path = os.path.join(OUTD, "players_video_%s.npz" % a.tag)
    np.savez_compressed(path, **out)
    json.dump(report, open(path.replace(".npz", "_report.json"), "w"), indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
