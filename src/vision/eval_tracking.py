import argparse
import glob
import json
import os
import sys
import threading
import queue

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.common import CACHE  # noqa: E402
from vision.players_common import (EXT, load_model, infer, pad32, load_sn_seq, fit_ground_h,  # noqa: E402
                                   h_apply, iou_matrix, kit_features, bgr_to_planes)
from vision import track_players as TP  # noqa: E402
from vision.eval_players import CONFIGS, TACT_MED_H  # noqa: E402

ODIR = os.path.join(CACHE, "players_eval")
MATCH_M = 2.0


def detect_seq(sd, cfg, model, out):
    if os.path.exists(out):
        return dict(np.load(out, allow_pickle=True))
    weights, cls, zoom = CONFIGS[cfg]
    per, files, _ = load_sn_seq(sd)
    hm = np.median([np.median(per[k]["boxes"][:, 3] - per[k]["boxes"][:, 1])
                    for k in per if len(per[k]["boxes"])])
    s = min(1.0, TACT_MED_H / hm)
    q = queue.Queue(maxsize=16)

    def prep(f):
        img = cv2.imread(os.path.join(sd, "img1", f))
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else img
        big = cv2.resize(small, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
        return small, big[:, :, ::-1].copy()

    def producer():
        from vision.players_common import bounded_map
        for r in bounded_map(prep, files):
            q.put(r)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    dets, feats = [], []
    with torch.inference_mode():
        while True:
            r = q.get()
            if r is None:
                break
            small, big = r
            x = torch.from_numpy(big).to("cuda").permute(2, 0, 1)[None].float() / 255.0
            d = infer(model, pad32(x), cls=cls, conf=0.1, topk=100)[0]
            d[:, :4] /= zoom
            Y, U, V = bgr_to_planes(small)
            feats.append(kit_features(Y, U, V, d[:, :4]).astype(np.float16))
            d[:, :4] /= s
            dets.append(d)
    res = dict(dets=np.array(dets, dtype=object), feats=np.array(feats, dtype=object), s=s)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **res)
    return res


class _Res:
    def __init__(self, d):
        self.conf = d[:, 4]
        self.cls = np.zeros(len(d))
        self.xywh = np.c_[(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2, d[:, 2] - d[:, 0],
                          d[:, 3] - d[:, 1]]
        self.d = d

    def __getitem__(self, m):
        return _Res(self.d[m])

    def __len__(self):
        return len(self.d)


def bytetrack_image(dets, s):
    from types import SimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker
    args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                           track_buffer=30, match_thresh=0.8, fuse_score=True)
    bt = BYTETracker(args)
    out = []
    for d in dets:
        dd = d.copy()
        dd[:, :4] *= s
        o = bt.update(_Res(dd))
        o = np.asarray(o).reshape(-1, 8)
        out.append((o[:, 4].astype(int), o[:, :4] / s))
    return out


def sim_atd(per, rng, nframes):
    st = {}
    out = []
    for k in range(nframes):
        g = per[k]
        keep = (g["role"] < 2) & (g["team"] >= 0) & np.isfinite(g["pitch"][:, 0])
        pos, team, tid = g["pitch"][keep], g["team"][keep], g["tid"][keep]
        gk = g["role"][keep] == 1
        crowd = np.zeros(len(pos), bool)
        if len(pos) > 1:
            dd = np.hypot(*(pos[:, None] - pos[None]).transpose(2, 0, 1))
            np.fill_diagonal(dd, 1e9)
            crowd = dd.min(1) < 2.0
        vis = np.ones(len(pos), bool)
        for i, t in enumerate(tid):
            miss = st.get(t, False)
            p_end = 1 / 37.5
            pm = 0.30 if crowd[i] else 0.05
            p_start = p_end * pm / (1 - pm)
            miss = (rng.random() > p_end) if miss else (rng.random() < p_start)
            st[t] = miss
            vis[i] = not miss
        out.append((pos[vis] + rng.normal(0, 0.3, (vis.sum(), 2)), team[vis], gk[vis]))
    return out


def match_atd(xy, atd_pos):
    m = np.full(len(xy), -1)
    if len(xy) == 0 or len(atd_pos) == 0:
        return m
    D = np.hypot(*(xy[:, None] - atd_pos[None]).transpose(2, 0, 1))
    r, c = linear_sum_assignment(np.where(D < MATCH_M, D, 1e6))
    ok = D[r, c] < MATCH_M
    m[r[ok]] = c[ok]
    return m


def eval_seq(sd, data, rng):
    per, files, _ = load_sn_seq(sd)
    n = len(files)
    Hs, last = [], None
    for k in range(n):
        Hm = fit_ground_h(per[k])
        last = Hm if Hm is not None else last
        Hs.append(last)
    if Hs[0] is None:
        first = next(h for h in Hs if h is not None)
        Hs = [h if h is not None else first for h in Hs]
    dets, feats, s = list(data["dets"]), list(data["feats"]), float(data["s"])
    xy, cf, ft, bx = [], [], [], []
    for k in range(n):
        d = dets[k]
        x, y = h_apply(Hs[k], (d[:, 0] + d[:, 2]) / 2, d[:, 3])
        ok = (x > -3) & (x < 108) & (y > -3) & (y < 71) & np.isfinite(x)
        dets[k] = d[ok]
        xy.append(np.c_[x[ok], y[ok]])
        cf.append(d[ok, 4])
        ft.append(feats[k][ok].astype(np.float32))
        bx.append(d[ok, :4])
    Hinv = [np.linalg.inv(h) for h in Hs]

    def in_view(t, pts):
        Hi = Hinv[t]
        u, v = h_apply(Hi, pts[:, 0], pts[:, 1])
        return (u > 0) & (u < 1920) & (v > 0) & (v < 1080)

    frames = np.arange(n)
    gt_frames = [(per[k]["tid"], per[k]["boxes"]) for k in range(n)]
    res = {}
    out_tracks = {}
    bt = bytetrack_image(dets, s)
    out_tracks["bytetrack_img"] = bt
    variants = {
        "ours_nostitch": dict(stitch=False, colour=True),
        "ours": dict(stitch=True, colour=True),
        "ours_nocolour": dict(stitch=True, colour=False),
    }
    dense_main = None
    for name, v in variants.items():
        tr = TP.run_tracker(frames, xy, cf, ft if v["colour"] else None, in_view=in_view)
        if v["stitch"]:
            tr = TP.stitch(tr, frames, xy, ft if v["colour"] else None)
        dense = TP.densify(tr, frames, xy, bx)
        if name == "ours":
            dense_main = dense
        pf = [([], []) for _ in range(n)]
        for tid, r in dense.items():
            for t, b in zip(r["t"], r["box"]):
                pf[t][0].append(tid)
                pf[t][1].append(b)
        out_tracks[name] = [(np.array(i, int), np.array(b).reshape(-1, 4)) for i, b in pf]
    for name, pr in out_tracks.items():
        r = TP._hota_alignment(gt_frames, pr, iou_matrix)
        r.update(TP.clear_idf1(gt_frames, pr, iou_matrix))
        ids = np.concatenate([p[0] for p in pr]) if pr else np.zeros(0)
        _, cnt = np.unique(ids, return_counts=True)
        r["pred_len_median"] = float(np.median(cnt)) if len(cnt) else 0
        frag = {}
        for (gi, gb), (pi, pb) in zip(gt_frames, pr):
            if len(gi) == 0 or len(pi) == 0:
                continue
            S = iou_matrix(gb, pb)
            a, b = linear_sum_assignment(-S)
            for i, j in zip(a, b):
                if S[i, j] >= 0.5:
                    frag.setdefault(gi[i], set()).add(pi[j])
        r["pred_ids_per_gt_id"] = float(np.mean([len(v) for v in frag.values()])) if frag else 0
        res[name] = r
    _, gcnt = np.unique(np.concatenate([g[0] for g in gt_frames]), return_counts=True)
    res["gt"] = dict(n_ids=len(gcnt), len_median=float(np.median(gcnt)))

    atd = sim_atd(per, rng, n)
    det_team, det_gk = [], []
    for k in range(n):
        mi = match_atd(xy[k], atd[k][0])
        tm = np.full(len(mi), -1)
        tm[mi >= 0] = atd[k][1][mi[mi >= 0]]
        det_team.append(tm)
        gk = np.zeros(len(mi), bool)
        gk[mi >= 0] = atd[k][2][mi[mi >= 0]]
        det_gk.append(gk)
    team, src = TP.assign_teams(dense_main, frames, det_team, ft, det_gk)
    rows = []
    for tid, r in dense_main.items():
        for t, d, b in zip(r["t"], r["det"], r["box"]):
            g = per[t]
            if len(g["boxes"]) == 0:
                continue
            iou = iou_matrix(b[None], g["boxes"])[0]
            j = int(np.argmax(iou))
            if iou[j] < 0.5:
                continue
            direct = d >= 0 and det_team[t][d] >= 0 and src[tid] != 3
            pt = det_team[t][d] if direct else team[tid]
            rows.append((g["role"][j], g["team"][j], pt, src[tid], int(direct)))
    rows = np.array(rows).reshape(-1, 5)
    res["team_rows"] = rows.tolist()
    res["sim_atd_present_frac"] = float(np.mean([len(a[1]) for a in atd]) /
                                        max(np.mean([((per[k]["role"] < 2) & (per[k]["team"] >= 0)).sum()
                                                     for k in range(n)]), 1))
    return res


def summarise(allres):
    keys = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDsw", "Recall", "Precision",
            "pred_len_median", "pred_ids_per_gt_id", "n_pred_ids"]
    summ = {}
    names = [k for k in next(iter(allres.values())) if k not in ("gt", "team_rows")
             and not k.startswith("sim_")]
    for nme in names:
        summ[nme] = {k: round(float(np.mean([r[nme][k] for r in allres.values()])), 4)
                     for k in keys}
    summ["gt"] = dict(n_ids=float(np.mean([r["gt"]["n_ids"] for r in allres.values()])),
                      len_median=float(np.median([r["gt"]["len_median"] for r in allres.values()])))
    rows = np.concatenate([np.array(r["team_rows"]).reshape(-1, 5) for r in allres.values()])
    role, gteam, pt, src, direct = rows.T
    pl = (role < 2) & (gteam >= 0)
    ref = role == 2
    t = {}
    t["player_frames"] = int(pl.sum())
    t["player_acc_all"] = round(float((pt[pl] == gteam[pl]).mean()), 4)
    t["player_undecided"] = round(float((pt[pl] < 0).mean()), 4)
    t["player_wrong"] = round(float(((pt[pl] >= 0) & (pt[pl] != gteam[pl])).mean()), 4)
    for nm, m in (("direct_atd", pl & (direct == 1)), ("track_vote", pl & (direct == 0) & (src == 1)),
                  ("colour", pl & (direct == 0) & (src == 2)), ("none", pl & (direct == 0) & (src == 0)),
                  ("other", pl & (src == 3))):
        dec = m & (pt >= 0)
        t[nm] = dict(frac=round(float(m.sum() / max(pl.sum(), 1)), 4),
                     acc_decided=round(float((pt[dec] == gteam[dec]).mean()), 4) if dec.any() else None)
    t["referee_frames"] = int(ref.sum())
    t["referee_given_a_team"] = round(float((pt[ref] >= 0).mean()), 4) if ref.any() else None
    gkm = role == 1
    t["goalkeeper_frames"] = int(gkm.sum())
    t["goalkeeper_acc"] = round(float((pt[gkm] == gteam[gkm]).mean()), 4) if gkm.any() else None
    t["outfield_acc"] = round(float((pt[pl & (role == 0)] == gteam[pl & (role == 0)]).mean()), 4)
    t["sim_atd_present_frac"] = round(float(np.mean([r["sim_atd_present_frac"] for r in allres.values()])), 4)
    summ["team"] = t
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="ours_s15")
    ap.add_argument("--seqs", type=int, default=58)
    a = ap.parse_args()
    seqs = sorted(glob.glob(os.path.join(EXT, "sn_gsr_valid", "SNGS-*")))[:a.seqs]
    model = None
    allres = {}
    for sd in seqs:
        out = os.path.join(ODIR, "trk_" + a.cfg, os.path.basename(sd) + ".npz")
        if not os.path.exists(out) and model is None:
            model = load_model(CONFIGS[a.cfg][0])
        data = detect_seq(sd, a.cfg, model, out)
        rng = np.random.default_rng(int(os.path.basename(sd)[-3:]))
        allres[os.path.basename(sd)] = eval_seq(sd, data, rng)
        r = allres[os.path.basename(sd)]
        print(os.path.basename(sd), {k: (round(r[k]["HOTA"], 3), round(r[k]["IDF1"], 3))
                                     for k in r if isinstance(r[k], dict) and "HOTA" in r[k]},
              flush=True)
    summ = summarise(allres)
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, per_seq={k: {kk: vv for kk, vv in v.items() if kk != "team_rows"}
                                          for k, v in allres.items()}),
              open(os.path.join(ODIR, "tracking_eval_%s.json" % a.cfg), "w"), indent=1)


if __name__ == "__main__":
    main()
