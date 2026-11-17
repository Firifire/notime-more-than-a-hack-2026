"""Players on the provided tactical video: INFERENCE ONLY with the external-data detector.

    .venv-vision/bin/python src/vision/run_players.py detect     # NVDEC -> x1.5 -> YOLO, resumable chunks
    .venv-vision/bin/python src/vision/run_players.py track      # -> cache/vision/players_video.npz + report

detect: graded frames (+-50), chunks of 10,000 frames -> cache/vision/players_cands/pc_<a>_<b>.npz
  flat arrays frame, box (M,4) original px, conf, feat (M,48) float16 kit histogram (yuv planes).
track (per half):
  1. foot point (box bottom-centre) -> pitch metres with the provided per-frame homography; boxes
     whose foot is outside the pitch +-1.5 m are dropped (spectators, staff, bench, photographers).
  2. tracker in metres (track_players.run_tracker; constants physical / checked on SN-GSR), then
     offline stitching; ATD slot continuity across a gap is an extra stitching cue (ATD is an
     automatic input of the challenge, not a label).
  3. ATD association per frame: Hungarian detections <-> ATD players, gate 2 m -> slot + team.
     Team per detection: the matched ATD player's team; else the track's ATD majority vote; else
     kit colour against the ATD-anchored players of the same +-60 s window (inference step, no
     training); else -1. Tracks whose kit is an outlier for both teams and that are not matched to
     an ATD goalkeeper slot are "other" (referees): team -1 even where they match an ATD player.
No match label (GT) is read anywhere.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.common import (CACHE, FrameReader, W, H, N_FRAMES, HALVES, process_ranges,  # noqa: E402
                           yuv420_to_rgb, load_homography, metres_to_px, px_to_metres,
                           graded_mask, PITCH)
from vision.players_common import (PLAYER_WEIGHTS, load_model, infer, pad32, kit_features,  # noqa: E402
                                   yuv_planes_from_i420)
from vision import track_players as TP  # noqa: E402

CDIR = os.path.join(CACHE, "players_cands")
OUT = os.path.join(CACHE, "players_video.npz")
_ZOOM = None


def zoom():
    global _ZOOM
    if _ZOOM is None:
        from vision.player_scale import inference_zoom
        _ZOOM = inference_zoom()[0]
    return _ZOOM


CONF_MIN = 0.10
MATCH_M = 2.0
K = 32
KBUF = 64
MARGIN_M = 1.5


def detect_chunk(model, start, stop, out, batch=8):
    import threading
    import queue
    ZOOM = zoom()
    sw, sh = int(round(W * ZOOM)), int(round(H * ZOOM))
    q = queue.Queue(maxsize=6)

    def producer():
        buf, idx = [], []
        for i, fr in FrameReader(start, stop):
            buf.append(fr)
            idx.append(i)
            if len(buf) == batch:
                q.put((idx, np.stack(buf)))
                buf, idx = [], []
        if buf:
            q.put((idx, np.stack(buf)))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    Fr, B, C, Ft = [], [], [], []
    n, t0 = 0, time.time()
    with torch.inference_mode():
        while True:
            item = q.get()
            if item is None:
                break
            idx, arr = item
            x = yuv420_to_rgb(torch.from_numpy(arr).to("cuda", non_blocking=True))
            x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
            dets = infer(model, pad32(x), cls=0, conf=CONF_MIN, topk=60)
            for i, fr, d in zip(idx, arr, dets):
                d[:, :4] /= ZOOM
                Y, U, V = yuv_planes_from_i420(fr, H, W)
                Fr.append(np.full(len(d), i, np.int32))
                B.append(d[:, :4])
                C.append(d[:, 4])
                Ft.append(kit_features(Y, U, V, d[:, :4]).astype(np.float16))
            n += len(idx)
            if n % 2000 < batch:
                print("  frame %d  %.1f fps" % (idx[-1], n / (time.time() - t0)), flush=True)
    tmp = os.path.join(os.path.dirname(out), "tmp_" + os.path.basename(out))
    np.savez_compressed(tmp, frame=np.concatenate(Fr), box=np.concatenate(B).astype(np.float32),
                        conf=np.concatenate(C).astype(np.float32), feat=np.concatenate(Ft),
                        start=start, stop=stop, zoom=ZOOM, weights=PLAYER_WEIGHTS)
    os.replace(tmp, out)
    print("wrote %s  %d frames  %.1f fps" % (out, n, n / (time.time() - t0)), flush=True)


def cmd_detect(a):
    os.makedirs(CDIR, exist_ok=True)
    model = None
    for _, s, e in process_ranges():
        for c in range(s, e, a.chunk):
            out = os.path.join(CDIR, "pc_%d_%d.npz" % (c, min(e, c + a.chunk)))
            if os.path.exists(out):
                continue
            model = model or load_model(PLAYER_WEIGHTS)
            detect_chunk(model, c, min(e, c + a.chunk), out)


def load_cands():
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(CDIR, "pc_*.npz")))]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("frame", "box", "conf", "feat")}
    o = np.argsort(cat["frame"], kind="stable")
    return {k: v[o] for k, v in cat.items()}, np.unique(np.concatenate(
        [np.arange(int(p["start"]), int(p["stop"])) for p in parts]))


def atd_runs(P_atd, frames):
    ok = ~np.isnan(P_atd[frames, :, 0])
    lens = []
    for s in range(ok.shape[1]):
        v = np.r_[False, ok[:, s], False].astype(np.int8)
        d = np.diff(v)
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        lens += list(en - st)
    return np.array(lens)


def cmd_track(a):
    from mtah.data import load_tracking
    cands, processed = load_cands()
    Hs = load_homography()
    atd = load_tracking()
    P_atd, team_atd = atd["players"][..., :2].astype(np.float64), atd["team"].astype(int)
    graded = graded_mask()
    fr_all = cands["frame"]
    bounds = np.searchsorted(fr_all, np.arange(N_FRAMES + 1))

    out = dict(
        xy=np.full((N_FRAMES, KBUF, 2), np.nan, np.float32), team=np.full((N_FRAMES, KBUF), -1, np.int8),
        track_id=np.full((N_FRAMES, KBUF), -1, np.int32), conf=np.zeros((N_FRAMES, KBUF), np.float32),
        atd_slot=np.full((N_FRAMES, KBUF), -1, np.int16), state=np.zeros((N_FRAMES, KBUF), np.int8),
        team_src=np.zeros((N_FRAMES, KBUF), np.int8), uv=np.full((N_FRAMES, KBUF, 2), np.nan, np.float32),
        processed=np.zeros(N_FRAMES, bool))
    out["processed"][processed] = True
    report = {}
    next_tid = 0
    overflow = 0
    for half, ga, gb in HALVES:
        frames = np.arange(max(ga - 50, processed.min()), gb + 51)
        frames = frames[out["processed"][frames]]
        px = P_atd[ga:gb + 1, :, 0]
        gl = np.nanmedian(np.minimum(px, PITCH[0] - px), axis=0)
        n_pres = np.isfinite(px).sum(0)
        gk_slot = (gl < 16.5) & (n_pres > 25 * 60)
        report["half%d_gk_slots" % half] = np.flatnonzero(gk_slot).tolist()
        xy, cf, ft, bx, slot, tm, gkd = [], [], [], [], [], [], []
        for t in frames:
            s, e = bounds[t], bounds[t + 1]
            b = cands["box"][s:e].astype(np.float64)
            Hinv = np.linalg.inv(Hs[t])
            x, y = px_to_metres(Hinv, (b[:, 0] + b[:, 2]) / 2, b[:, 3])
            ok = (np.isfinite(x) & (x > -MARGIN_M) & (x < PITCH[0] + MARGIN_M) & (y > -MARGIN_M)
                  & (y < PITCH[1] + MARGIN_M))
            p = np.c_[x[ok], y[ok]]
            xy.append(p)
            cf.append(cands["conf"][s:e][ok])
            ft.append(cands["feat"][s:e][ok].astype(np.float32))
            bx.append(b[ok])
            ap = P_atd[t]
            av = np.flatnonzero(~np.isnan(ap[:, 0]))
            sl = np.full(len(p), -1)
            if len(p) and len(av):
                D = np.hypot(*(p[:, None] - ap[av][None]).transpose(2, 0, 1))
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

        t0 = time.time()
        tr = TP.run_tracker(frames, xy, cf, ft, in_view=in_view)
        n_online = len(tr)
        tr_plain = TP.stitch(tr, frames, xy, ft)

        def slot_at(obs, first):
            seq = obs[:10] if first else obs[-10:]
            v = [slot[fpos[t]][d] for t, d in seq]
            v = [x for x in v if x >= 0]
            if len(v) < 3:
                return -1
            u, c = np.unique(v, return_counts=True)
            return int(u[np.argmax(c)]) if c.max() >= 0.7 * len(v) else -1

        s_end = {k: slot_at(sorted(o), False) for k, o in tr.items()}
        s_beg = {k: slot_at(sorted(o), True) for k, o in tr.items()}
        t_end = {k: max(t for t, _ in o) for k, o in tr.items()}
        t_beg = {k: min(t for t, _ in o) for k, o in tr.items()}

        def extra(ia, ib):
            s = s_end[ia]
            if s < 0 or s != s_beg[ib]:
                return False
            seg = P_atd[t_end[ia]:t_beg[ib] + 1, s, 0]
            return bool(np.all(np.isfinite(seg)))

        tr_s = TP.stitch(tr, frames, xy, ft, extra_link=extra)
        dense = TP.densify(tr_s, frames, xy, bx)
        dense_plain = TP.densify(tr_plain, frames, xy, bx)
        team, src = TP.assign_teams(dense, frames, tm, ft, gkd, win=1500)
        print("half %d: online %d tracklets, stitched %d, with ATD-continuity %d, dense %d (%.0fs)"
              % (half, n_online, len(tr_plain), len(tr_s), len(dense), time.time() - t0), flush=True)
        fill = np.zeros(N_FRAMES, np.int32)
        lens = []
        for k, r in dense.items():
            tid = next_tid
            next_tid += 1
            gm = graded[r["t"]]
            if gm.any():
                lens.append(int(gm.sum()))
            for t, p, st, d, b in zip(r["t"], r["xy"], r["state"], r["det"], r["box"]):
                j = fill[t]
                if j >= KBUF:
                    overflow += 1
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
                    if tm[i][d] >= 0 and src[k] != 3:
                        out["team"][t, j], out["team_src"][t, j] = tm[i][d], 1
                        continue
                if src[k] == 3:
                    out["team_src"][t, j] = 4
                elif team[k] >= 0:
                    out["team"][t, j] = team[k]
                    out["team_src"][t, j] = 2 if src[k] == 1 else 3
        lp = [int(graded[r["t"]].sum()) for r in dense_plain.values() if graded[r["t"]].any()]
        report["half%d" % half] = dict(
            online_tracklets=n_online, stitched=len(tr_plain), stitched_atd=len(tr_s),
            tracks_graded=len(lens), track_len_median_s=float(np.median(lens)) / 25,
            tracks_graded_no_atd_link=len(lp), track_len_median_s_no_atd_link=float(np.median(lp)) / 25,
            atd_fragments_graded=int((atd_runs(P_atd, np.arange(ga, gb + 1)) > 0).sum()),
            atd_fragment_len_median_s=float(np.median(atd_runs(P_atd, np.arange(ga, gb + 1)))) / 25)
    n_pres = (out["state"] > 0).sum(1)
    report["frames_with_more_than_K_tracks"] = int((n_pres > K).sum())
    report["frames_with_more_than_K_tracks_graded"] = int((n_pres[graded] > K).sum())
    pri = ((out["state"] > 0) * 8.0 + (out["team"] >= 0) * 4 + (out["state"] == 1) * 2
           + out["conf"])
    order = np.argsort(-pri, axis=1, kind="stable")[:, :K]
    for key in ("xy", "uv"):
        out[key] = np.take_along_axis(out[key], order[..., None], 1)
    for key in ("team", "track_id", "conf", "atd_slot", "state", "team_src"):
        out[key] = np.take_along_axis(out[key], order, 1)
    report["overflow_slots"] = overflow
    out["doc"] = np.array(__doc__)
    np.savez_compressed(OUT, **out)
    report.update(diagnostics(out, P_atd, team_atd, graded))
    json.dump(report, open(OUT.replace(".npz", "_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


def diagnostics(o, P_atd, team_atd, graded):
    g = np.flatnonzero(graded)
    st = o["state"][g]
    tm = o["team"][g]
    pres = st > 0
    det = st == 1
    atd_ok = ~np.isnan(P_atd[g, :, 0])
    r = {}
    r["graded_frames"] = int(len(g))
    r["frames_processed"] = float(o["processed"][g].mean())
    r["persons_per_frame_mean (all tracked, incl. referees/unknown)"] = float(pres.sum(1).mean())
    r["players_per_frame_mean (team assigned)"] = float((pres & (tm >= 0)).sum(1).mean())
    r["detected_players_per_frame_mean (state 1, team assigned)"] = float((det & (tm >= 0)).sum(1).mean())
    r["atd_players_per_frame_mean"] = float(atd_ok.sum(1).mean())
    r["team_source_frac (of team-assigned player slots) direct/vote/colour"] = [
        float(((o["team_src"][g] == k) & pres).sum() / max((pres & (tm >= 0)).sum(), 1)) for k in (1, 2, 3)]
    r["unknown_team_per_frame (referees/unknown)"] = float((pres & (tm < 0)).sum(1).mean())
    r["other_kit_outlier_per_frame (referees)"] = float((pres & (o["team_src"][g] == 4)).sum(1).mean())
    added = pres & (tm >= 0) & (o["atd_slot"][g] < 0)
    r["ours_unmatched_team_players_per_frame"] = float(added.sum(1).mean())
    miss = []
    for c in (0, 1):
        n_atd = (atd_ok & (team_atd[None, :] == c)).sum(1)
        n_add = (added & (tm == c)).sum(1)
        miss.append((np.maximum(0, 11 - n_atd), np.minimum(np.maximum(0, 11 - n_atd), n_add)))
    tot_miss = miss[0][0] + miss[1][0]
    tot_fill = miss[0][1] + miss[1][1]
    r["atd_missing_players_per_frame (22 - ATD, per team capped at 11)"] = float(tot_miss.mean())
    r["atd_missing_filled_by_video_per_frame"] = float(tot_fill.mean())
    r["atd_missing_filled_frac"] = float(tot_fill.sum() / max(tot_miss.sum(), 1))
    r["union_players_per_frame (ATD + added, capped 11/team)"] = float(
        sum(np.minimum(11, (atd_ok & (team_atd[None, :] == c)).sum(1) + (added & (tm == c)).sum(1))
            for c in (0, 1)).mean())
    bz = np.load(os.path.join(CACHE, "ball_video_v2.npz"))
    ball = bz["xy_fused"][g].astype(np.float64)
    bok = np.isfinite(ball[:, 0])
    d_our = np.hypot(o["xy"][g][..., 0] - ball[:, None, 0], o["xy"][g][..., 1] - ball[:, None, 1])
    d_atd = np.hypot(P_atd[g, :, 0] - ball[:, None, 0], P_atd[g, :, 1] - ball[:, None, 1])
    near_our = (pres & (tm >= 0) & (d_our < 10)).sum(1)[bok]
    near_all = (pres & (d_our < 10)).sum(1)[bok]
    near_atd = (atd_ok & (d_atd < 10)).sum(1)[bok]
    near_add = (added & (d_our < 10)).sum(1)[bok]
    r["ball_frames"] = int(bok.sum())
    r["within_10m_of_ball: ATD / ours team-assigned / ours all persons / ours added (unmatched)"] = [
        float(near_atd.mean()), float(near_our.mean()), float(near_all.mean()), float(near_add.mean())]
    r["within_10m: union ATD + added"] = float((near_atd + near_add).mean())
    sl = o["atd_slot"][g]
    m = det & (sl >= 0)
    fi, ki = np.nonzero(m)
    pa = P_atd[g[fi], sl[fi, ki]]
    dd = np.hypot(o["xy"][g][fi, ki, 0] - pa[:, 0], o["xy"][g][fi, ki, 1] - pa[:, 1])
    r["matched_det_vs_atd_m p50/p90"] = [float(np.percentile(dd, 50)), float(np.percentile(dd, 90))]
    r["matched_frac_of_detected_team_players"] = float(m.sum() / max((det & (tm >= 0)).sum(), 1))
    conf_atd = np.zeros_like(atd_ok)
    conf_atd[fi, sl[fi, ki]] = True
    r["atd_players_with_a_matching_detection_frac"] = float(conf_atd[atd_ok].mean())
    tid = o["track_id"][g][pres]
    _, cnt = np.unique(tid, return_counts=True)
    r["tracks_on_graded"] = int(len(cnt))
    r["track_len_s p25/p50/p75"] = [float(np.percentile(cnt, q)) / 25 for q in (25, 50, 75)]
    r["track_len_s team-assigned p50"] = float(np.median(np.unique(
        o["track_id"][g][pres & (tm >= 0)], return_counts=True)[1])) / 25
    lens = atd_runs(P_atd, g)
    r["atd_fragments_graded"] = int(len(lens))
    r["atd_fragment_len_s p25/p50/p75"] = [float(np.percentile(lens, q)) / 25 for q in (25, 50, 75)]
    r["player_seconds_covered: ours tracks / ATD"] = [float(pres.sum() / 25), float(atd_ok.sum() / 25)]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "track"])
    ap.add_argument("--chunk", type=int, default=10000)
    a = ap.parse_args()
    {"detect": cmd_detect, "track": cmd_track}[a.cmd](a)


if __name__ == "__main__":
    main()
