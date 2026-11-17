"""Track D4 step 4: the learned tracklet linker on the provided match (inference only, no labels read).

Runs src/vision/run_players.cmd_track UNCHANGED (same detections cache/vision/players_cands, per-frame homography,
pitch filter, v1 online tracker, ATD association for team votes, assign_teams, per-frame slot writer), with two
hooks (monkeypatched in this process only; src/ is not edited):
  * the final identity linking (the stitch call that carries the ATD-continuity cue) is replaced by the D4 linker:
    K2 pass-1 pure tracklets -> learned predecessor-pointer canvas in overlapping windows of 256 tracklets
    (each tracklet decided in the window where it is most central) -> exclusion clean-up -> identities;
  * densify keeps the identity across gaps > 125 frames (no boxes there) instead of splitting it into a new track id
    (k2track.densify_v2 with fill 125 = v1's interpolation limit).
Output: /mnt/x/Data/AI/more-than-a-hack/trackD4/players_video_<tag>.npz in the format of cache/vision/players_video.npz.

    PYTHONPATH=src .venv-vision/bin/python tools/trackD4/match.py run --name d4a [--mode sampler]
    PYTHONPATH=src .venv-vision/bin/python tools/trackD4/match.py quality FILE [FILE...]
    PYTHONPATH=src .venv-vision/bin/python tools/trackD4/match.py sheet FILE_A FILE_B
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import feats as Fe  # noqa: E402
import linker as Lk  # noqa: E402
import k2track as K2  # noqa: E402
from vision import track_players as TP  # noqa: E402

BASE = Lk.BASE


def atd_team(frames, xy, f0, n):
    from scipy.optimize import linear_sum_assignment
    from mtah.data import load_tracking
    atd = load_tracking()
    P_atd, team_atd = atd["players"][..., :2].astype(np.float64), atd["team"].astype(int)
    out = [np.zeros(0, int)] * n
    for i, t in enumerate(frames):
        p = xy[i]
        tm = np.full(len(p), -1)
        ap = P_atd[t]
        av = np.flatnonzero(~np.isnan(ap[:, 0]))
        if len(p) and len(av):
            D = np.hypot(*(p[:, None] - ap[av][None]).transpose(2, 0, 1))
            r, c = linear_sum_assignment(np.where(D < 2.0, D, 1e6))
            k = D[r, c] < 2.0
            tm[r[k]] = team_atd[av[c[k]]]
        out[t - f0] = tm
    return out


def _conf_team(v):
    s_ = v.sum()
    if s_ < 5:
        return -1
    c = int(np.argmax(v))
    return c if v[c] >= 0.85 * s_ else -1


def link_windows(model, dev, mode, N, classic, team, pieces, ours, k2src, xyL, cfL, ftL, n, dt=None, fallback=None,
                 embl=None, veto=False):
    W, S = 750, 375
    start = {k: obs[0][0] for k, obs in pieces.items()}
    pred, stats = {}, []
    cnt = dict(links=0, new=0, junk=0)
    w0 = 0
    while True:
        w1 = min(w0 + W, n)

        def clip(d):
            o = {}
            for k, obs in d.items():
                v = [(t - w0, dd) for t, dd in obs if w0 <= t < w1]
                if v:
                    o[k] = v
            return o
        res = dict(n=w1 - w0, xy=xyL[w0:w1], cf=cfL[w0:w1], ft=ftL[w0:w1], pieces=clip(pieces), ours=clip(ours),
                   k2=clip(k2src))
        if dt is not None:
            res["det_team"] = dt[w0:w1]
        if embl is not None:
            res["embl"] = embl[w0:w1]
        lo = w0 + S // 2 if w0 > 0 else -1
        hi = w0 + S // 2 + S if w1 < n else n + 1
        if res["pieces"]:
            rows, X = Fe.clip_canvas(res)
            tok, logp, st_ = Lk.link_clip(model, X, dev, mode=mode, N=N, classic=classic,
                                          team=team)
            stats += st_
            for i, r in enumerate(rows):
                if not (lo <= start[r["id"]] < hi) or start[r["id"]] < w0:
                    continue
                t = int(tok[i])
                if t < Fe.K and X["cand"][i, t] >= 0:
                    j = rows[int(X["cand"][i, t])]["id"]
                    pred[r["id"]] = (j, float(np.exp(logp[i, t])))
                    cnt["links"] += 1
                else:
                    cnt["new" if t == Fe.NEW else "junk"] += 1
        if w1 >= n:
            break
        w0 += S
    best = {}
    for i, (j, pr) in pred.items():
        if j not in best or pr > best[j][1]:
            best[j] = (i, pr)
    keep = {i: j for j, (i, _) in best.items()}
    root = {}
    votes = {}
    nveto = nfb = 0
    if veto and dt is not None:
        for k, obs in pieces.items():
            v = np.zeros(2)
            for t, d in obs:
                tm = dt[t][d]
                if tm >= 0:
                    v[tm] += 1
            votes[k] = v
    fb = fallback or {}
    claimed = set()
    d4_targets = set(keep.values())
    for k in sorted(pieces, key=lambda k: start[k]):
        root[k] = k
        for src_, j in (("d4", keep.get(k)), ("fb", fb.get(k))):
            if j is None or j in claimed or j not in root or (src_ == "fb" and j in d4_targets):
                continue
            if votes:
                r = root[j]
                a_, b_ = _conf_team(votes[r]), _conf_team(votes[k])
                if a_ >= 0 and b_ >= 0 and a_ != b_:
                    nveto += 1
                    continue
            root[k] = root[j]
            claimed.add(j)
            if votes:
                votes[root[k]] = votes[root[k]] + votes[k]
            nfb += src_ == "fb"
            break
    cnt["fallback_links"] = nfb
    cnt["team_veto"] = nveto
    ids = {}
    for k, obs in pieces.items():
        ids.setdefault(root[k], []).extend(obs)
    return ids, dict(pieces=len(pieces), d4=len(ids), conflicts_dropped=len(pred) - len(keep), windows=len(stats),
                     steps=float(np.mean([s for s, _ in stats])), ent=float(np.mean([e for _, e in stats])), **cnt)


def make_link(model, dev, mode, N, classic, stash, orig_stitch):
    def d4_link(tracks, frames, xy, feat):
        cf = stash["conf"]
        f0 = int(frames[0])
        n = int(frames[-1]) - f0 + 1
        e2, e1, ef = np.zeros((0, 2)), np.zeros(0), np.zeros((0, 48), np.float32)
        xyL, cfL, ftL = [e2] * n, [e1] * n, [ef] * n
        for i, t in enumerate(frames):
            xyL[t - f0], cfL[t - f0], ftL[t - f0] = xy[i], cf[i], feat[i]
        trk = {k: sorted((t - f0, d) for t, d in obs) for k, obs in tracks.items()}
        fr = np.arange(n)
        ours = orig_stitch(trk, fr, xyL, ftL)
        k2ids, pieces = K2.track_v2(fr, xyL, cfL, ftL, tracklets=trk, return_pieces=True)
        if stash.get("tracklets") == "v1":
            pieces = {int(k): sorted(v) for k, v in trk.items()}
        k2src = ours if stash.get("k2_as_ours") else k2ids
        dt = atd_team(frames, xy, f0, n) if stash.get("team") else None
        ids, st = link_windows(model, dev, mode, N, classic, stash.get("team", False), pieces, ours, k2src, xyL, cfL,
                               ftL, n, dt)
        out = {k: sorted((t + f0, d) for t, d in obs) for k, obs in ids.items()}
        st.update(ours=len(ours), k2=len(k2ids))
        stash.setdefault("halves", []).append(st)
        print("D4 link", json.dumps(st), flush=True)
        return out
    return d4_link


def cmd_run(a):
    import torch
    from vision import run_players as RP
    dev = "cuda"
    st = torch.load(os.path.join(Lk.RUNS, a.name, "model.pt"), map_location=dev, weights_only=False)
    ar = st["args"]
    classic = not ar.get("no_classic", False)
    model = Lk.Linker(ar["d"], ar["layers"], ar["heads"], ar["drop"], ar.get("team", False)).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    stash = {"tracklets": a.tracklets, "team": ar.get("team", False), "k2_as_ours": ar.get("canvas") == "canvas3"}
    if ar.get("canvas") in ("canvas2", "canvas3"):
        Fe.BASE_M = 3.0
    orig_run, orig_stitch = TP.run_tracker, TP.stitch

    def run_tracker(frames, xy, conf, feat=None, in_view=None):
        stash["conf"] = conf
        return orig_run(frames, xy, conf, feat, in_view=in_view)
    link = make_link(model, dev, a.mode, a.N, classic, stash, orig_stitch)

    def stitch(tracks, frames, xy, feat=None, extra_link=None):
        if extra_link is None:
            return orig_stitch(tracks, frames, xy, feat)
        return link(tracks, frames, xy, feat)

    def densify(tracks, frames, xy, boxes=None, max_gap=125):
        return K2.densify_v2(tracks, frames, xy, boxes, fill_max=max_gap)
    TP.run_tracker, TP.stitch, TP.densify = run_tracker, stitch, densify
    tag = a.tag or "%s_%s" % (a.name, a.mode)
    RP.OUT = os.path.join(BASE, "players_video_%s.npz" % tag)
    RP.__doc__ = (RP.__doc__ or "") + "\n[Track D4] identities linked by the learned tracklet linker %s (%s); " \
        "see tools/trackD4/match.py\n" % (a.name, a.mode)
    RP.cmd_track(argparse.Namespace())
    json.dump(stash.get("halves"), open(RP.OUT.replace(".npz", "_d4link.json"), "w"), indent=1)
    print("WROTE", RP.OUT)


def quality(path):
    from vision.common import graded_mask
    o = np.load(path)
    g = np.flatnonzero(graded_mask())
    st, tm, tid = o["state"][g], o["team"][g], o["track_id"][g]
    pres = st > 0
    r = {}
    fi, ki = np.nonzero(pres & (tm >= 0))
    ids, teams, frs = tid[fi, ki], tm[fi, ki], fi
    o_ = np.lexsort((frs, ids))
    ids, teams = ids[o_], teams[o_]
    same = ids[1:] == ids[:-1]
    flips = int(((teams[1:] != teams[:-1]) & same).sum())
    ntr = len(np.unique(ids))
    mixed = 0
    for u in np.unique(ids):
        t = teams[ids == u]
        if len(t) and min((t == 0).mean(), (t == 1).mean()) > 0.05:
            mixed += 1
    r["team_assigned_tracks_on_graded"] = ntr
    r["tracks_per_player (team-assigned tracks / 22)"] = round(ntr / 22, 1)
    r["team_flips_inside_identity"] = flips
    r["tracks_with_both_teams_gt5pct"] = mixed
    for c in (0, 1):
        r["frames_gt11_team%d_share" % c] = float(((pres & (tm == c)).sum(1) > 11).mean())
    _, cnt = np.unique(tid[pres & (tm >= 0)], return_counts=True)
    r["team_track_len_s p50/p90"] = [float(np.percentile(cnt, 50)) / 25, float(np.percentile(cnt, 90)) / 25]
    r["players_per_frame (team-assigned)"] = float((pres & (tm >= 0)).sum(1).mean())
    return r


def cmd_quality(a):
    out = {}
    for p in a.files:
        out[p] = quality(p)
        print(p, json.dumps(out[p]), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def cmd_sheet(a):
    import cv2
    import torch
    from vision.common import FrameReader, yuv420_to_rgb
    files = a.files
    los = [np.load(f) for f in files]
    frames = []
    for s in a.frames.split(","):
        if "-" in s:
            lo, hi = map(int, s.split("-"))
            frames += list(range(lo, hi + 1, a.step))
        else:
            frames.append(int(s))
    frames = sorted(set(frames))
    want = set(frames)
    tiles = {}
    groups = []
    for t in frames:
        if groups and t - groups[-1][-1] <= a.step:
            groups[-1].append(t)
        else:
            groups.append([t])
    for gr in groups:
        for i, fr in FrameReader(gr[0], gr[-1] + 1):
            if i not in want:
                continue
            x = yuv420_to_rgb(torch.from_numpy(fr[None].copy()).cuda())[0]
            img = (x.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()[:, :, ::-1].copy()
            row = []
            for name, o in zip(a.names.split(","), los):
                im = img.copy()
                for j in range(o["state"].shape[1]):
                    if o["state"][i, j] == 0:
                        continue
                    u, v = o["uv"][i, j]
                    if not np.isfinite(u):
                        continue
                    tmv = int(o["team"][i, j])
                    col = {0: (255, 255, 255), 1: (0, 0, 255), -1: (0, 255, 255)}[tmv]
                    cv2.circle(im, (int(u), int(v)), 6, col, -1 if o["state"][i, j] == 1 else 2)
                    cv2.putText(im, str(int(o["track_id"][i, j])), (int(u) + 6, int(v) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                cv2.putText(im, "%s  frame %d" % (name, i), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
                row.append(cv2.resize(im, (960, 540)))
            tiles[i] = np.concatenate(row, 1)
    sheet = np.concatenate([tiles[t] for t in frames if t in tiles], 0)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("WROTE", a.out, sheet.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--name", default="d4a")
    ap.add_argument("--mode", default="sampler")
    ap.add_argument("--tracklets", default="v1", help="v1 (online tracklets) or pieces (K2 pass 1)")
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--frames", default=None, help="sheet: frame range A-B (required)")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--names", default="v1,d4")
    a = ap.parse_args()
    if a.cmd == "sheet" and not a.frames:
        raise SystemExit("sheet needs --frames A-B")
    {"run": cmd_run, "quality": cmd_quality, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
