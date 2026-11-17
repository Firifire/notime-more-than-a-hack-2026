import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from vision.players_common import iou_matrix, h_apply  # noqa: E402
from vision import track_players as TP  # noqa: E402
from vision.eval_tracking import sim_atd, match_atd  # noqa: E402
import k2track as K  # noqa: E402

PACKS = "/mnt/x/Data/AI/more-than-a-hack/trackK2/gsr_packs"
OUTD = os.path.join(ROOT, "runs", "trackK2")
EXPORT = os.environ.get("K2_EXPORT")
EMB_EXPORT = bool(os.environ.get("K2_EMB_EXPORT"))
EMBD = "/mnt/x/Data/AI/more-than-a-hack/trackK2/reid"


def load_pack(f):
    zz = np.load(f)
    z = {k: zz[k] for k in zz.files}
    n = int(z["n"])
    b = np.searchsorted(z["det_f"], np.arange(n + 1))
    xy = [z["det_xy"][b[k]:b[k + 1]] for k in range(n)]
    cf = [z["det_conf"][b[k]:b[k + 1]] for k in range(n)]
    ft = [z["det_feat"][b[k]:b[k + 1]] for k in range(n)]
    bx = [z["det_box"][b[k]:b[k + 1]] for k in range(n)]
    g = np.searchsorted(z["gt_f"], np.arange(n + 1))
    per = {k: dict(boxes=z["gt_box"][g[k]:g[k + 1]], tid=z["gt_tid"][g[k]:g[k + 1]],
                   role=z["gt_role"][g[k]:g[k + 1]], team=z["gt_team"][g[k]:g[k + 1]],
                   pitch=z["gt_xy"][g[k]:g[k + 1]]) for k in range(n)}
    return n, xy, cf, ft, bx, per, z["H"]


def to_frames(dense, n, team=None, src=None, det_team=None, override=False):
    pf = [([], [], []) for _ in range(n)]
    for tid, r in dense.items():
        for t, b, d in zip(r["t"], r["box"], r["det"]):
            if team is None:
                tm = -1
            elif src[tid] == 3:
                tm = -1
            elif override and d >= 0 and det_team[t][d] >= 0:
                tm = det_team[t][d]
            else:
                tm = team[tid]
            pf[t][0].append(tid)
            pf[t][1].append(b)
            pf[t][2].append(tm)
    return [(np.array(i, int), np.array(b).reshape(-1, 4), np.array(m, int)) for i, b, m in pf]


def gt_match(gt_frames, pr):
    out = []
    for (gi, gb), (pi, pb, _) in zip(gt_frames, pr):
        if len(gi) == 0 or len(pi) == 0:
            out.append([])
            continue
        S = iou_matrix(gb, pb)
        a, b = linear_sum_assignment(-S)
        out.append([(int(gi[i]), int(pi[j]), int(j)) for i, j in zip(a, b) if S[i, j] >= 0.5])
    return out


def diagnose(per, gm, xy_det_frames, n):
    present = {}
    for k in range(n):
        for i, g in enumerate(per[k]["tid"]):
            present.setdefault(int(g), []).append(k)
    seq = {}
    owner = {}
    for k, ms in enumerate(gm):
        for g, p, _ in ms:
            seq.setdefault(g, []).append((k, p))
            owner.setdefault(p, []).append((k, g))
    ev = []
    for g, s in seq.items():
        for (k0, p0), (k1, p1) in zip(s[:-1], s[1:]):
            if p0 == p1:
                continue
            sw = any(gg != g and k1 - 50 <= kk < k1 for kk, gg in owner.get(p1, [])) or \
                any(gg != g and k0 < kk <= k0 + 50 for kk, gg in owner.get(p0, []))
            pres = set(present[g])
            absent = sum(1 for k in range(k0 + 1, k1) if k not in pres)
            cross = False
            for k in range(max(0, k0 - 5), min(n, k1 + 6)):
                G = per[k]
                j = np.flatnonzero(G["tid"] == g)
                if not len(j):
                    continue
                j = j[0]
                oth = np.flatnonzero(G["tid"] != g)
                if not len(oth):
                    continue
                dpos = np.hypot(G["pitch"][oth, 0] - G["pitch"][j, 0], G["pitch"][oth, 1] - G["pitch"][j, 1])
                iou = iou_matrix(G["boxes"][j][None], G["boxes"][oth])[0]
                if np.nanmin(np.r_[dpos, 99]) < 1.5 or iou.max() > 0.2:
                    cross = True
                    break
            gap = k1 - k0 - 1
            if absent >= 5:
                cause = "reentry"
            elif cross:
                cause = "crossing"
            elif gap >= 5:
                cause = "det_gap"
            else:
                cause = "other"
            ev.append(dict(gt=g, k0=k0, k1=k1, gap=gap, kind="switch" if sw else "frag", cause=cause))
    return ev


def eval_clip(args):
    f, overrides = args
    for k, v in overrides.items():
        if k.startswith("tp_"):
            setattr(TP.P, k[3:], v)
        elif k.startswith("x_"):
            setattr(K.PX, k[2:], v)
        else:
            setattr(K.P, k, tuple(v) if isinstance(v, list) else v)
    seq = os.path.basename(f).split("_", 1)[1].replace(".npz", "")
    n, xy, cf, ft, bx, per, Hs = load_pack(f)
    Hinv = [np.linalg.inv(h) for h in Hs]

    def in_view(t, pts):
        u, v = h_apply(Hinv[t], pts[:, 0], pts[:, 1])
        return (u > 0) & (u < 1920) & (v > 0) & (v < 1080)
    frames = np.arange(n)
    rng = np.random.default_rng(int(seq[-3:]))
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
    gt_frames = [(per[k]["tid"], per[k]["boxes"]) for k in range(n)]
    tr = TP.run_tracker(frames, xy, cf, ft, in_view=in_view)
    out = {}
    d1 = TP.densify(TP.stitch(tr, frames, xy, ft), frames, xy, bx)
    t1, s1 = TP.assign_teams(d1, frames, det_team, ft, det_gk)
    _, clean = K.frame_flags(frames, xy)
    det_team_c = [np.where(c_, x_, -1) for x_, c_ in zip(det_team, clean)] if K.P.team_clean else det_team
    embl = None
    if K.P.w_emb > 0 or K.PX.w_emb > 0 or EMB_EXPORT:
        ef = os.path.join(EMBD, "emb%s_" % os.environ.get("K2_EMB_TAG", "") + os.path.basename(f))
        ez = np.load(ef)
        embl = [np.full((len(x_), ez["emb"].shape[1]), np.nan, np.float32) for x_ in xy]
        for fr_, d_, e_ in zip(ez["frame"], ez["det"], ez["emb"].astype(np.float32)):
            embl[fr_][d_] = e_
    ids, pieces = K.track_v2(frames, xy, cf, ft, in_view, det_team_c, tracklets=tr, return_pieces=True, emb=embl)
    d2 = K.densify_v2(ids, frames, xy, bx)
    t2, s2 = K.identity_teams(d2, frames, det_team_c, ft, det_gk, cf)
    if EXPORT:
        from run_match import save_tracklets
        os.makedirs(EXPORT, exist_ok=True)
        save_tracklets(os.path.join(EXPORT, os.path.basename(f)), pieces, frames, xy, cf, ft, bx, emb=embl)
    det_gt = []
    for k in range(n):
        g = np.full(len(bx[k]), -1)
        if len(bx[k]) and len(per[k]["tid"]):
            S_ = iou_matrix(bx[k], per[k]["boxes"])
            a_, b_ = linear_sum_assignment(-S_)
            ok_ = S_[a_, b_] >= 0.5
            g[a_[ok_]] = per[k]["tid"][b_[ok_]]
        det_gt.append(g)
    pur_num = pur_den = 0
    plen = []
    orc = {}
    for pk, obs in pieces.items():
        gg = np.array([det_gt[t][d] for t, d in obs])
        m = gg[gg >= 0]
        plen.append(len(obs))
        if len(m):
            u, c = np.unique(m, return_counts=True)
            pur_num += c.max()
            pur_den += len(m)
            maj = int(u[np.argmax(c)])
            orc.setdefault(("g", maj), []).extend(obs)
        else:
            orc[("p", pk)] = list(obs)
    oid = {}
    for key, obs in orc.items():
        by_t = {}
        for t, d in sorted(obs):
            if t not in by_t or (key[0] == "g" and det_gt[t][d] == key[1]):
                by_t[t] = d
        oid[len(oid)] = sorted(by_t.items())
    d_or = K.densify_v2(oid, frames, xy, bx)
    out["purity"] = dict(num=int(pur_num), den=int(pur_den), n_pieces=len(pieces), len_median=float(np.median(plen)))
    variants = {"v1": to_frames(d1, n, t1, s1, det_team, override=True),
                "oracle": to_frames(d_or, n),
                "v1_trackteam": to_frames(d1, n, t1, s1, det_team, override=False),
                "v2": to_frames(d2, n, t2, s2, det_team, override=False)}
    for name, pr in variants.items():
        if name == "v1_trackteam":
            pass
        r = TP._hota_alignment(gt_frames, [(p[0], p[1]) for p in pr], iou_matrix)
        r.update(TP.clear_idf1(gt_frames, [(p[0], p[1]) for p in pr], iou_matrix))
        gm = gt_match(gt_frames, pr)
        frag = {}
        rows = []
        trk_gt = {}
        for k, ms in enumerate(gm):
            for g, p, j in ms:
                frag.setdefault(g, set()).add(p)
                gi = int(np.flatnonzero(per[k]["tid"] == g)[0])
                role, gteam = int(per[k]["role"][gi]), int(per[k]["team"][gi])
                rows.append((role, gteam, int(pr[k][2][j])))
                if role < 2 and gteam >= 0:
                    trk_gt.setdefault(p, []).append((gteam, int(pr[k][2][j])))
        r["pred_ids_per_gt_id"] = float(np.mean([len(v) for v in frag.values()])) if frag else 0.0
        rows = np.array(rows).reshape(-1, 3)
        pl = (rows[:, 0] < 2) & (rows[:, 1] >= 0)
        r["team_frames"] = int(pl.sum())
        rf = rows[:, 0] == 2
        r["ref_frames"], r["ref_team"] = int(rf.sum()), int((rows[rf, 2] >= 0).sum())
        gkm = (rows[:, 0] == 1) & (rows[:, 1] >= 0)
        r["gk_frames"], r["gk_correct"] = int(gkm.sum()), int((rows[gkm, 2] == rows[gkm, 1]).sum())
        r["team_correct"] = int((rows[pl, 2] == rows[pl, 1]).sum())
        r["team_wrong"] = int(((rows[pl, 2] >= 0) & (rows[pl, 2] != rows[pl, 1])).sum())
        nflip = nwrong = ntr = 0
        for p, v in trk_gt.items():
            v = np.array(v)
            ntr += 1
            dec = v[v[:, 1] >= 0, 1]
            if len(np.unique(dec)) > 1:
                nflip += 1
            gmaj = np.bincount(v[:, 0]).argmax()
            if len(dec) and np.bincount(dec).argmax() != gmaj:
                nwrong += 1
        r["tracks_scored"], r["tracks_team_flip"], r["tracks_team_wrong"] = ntr, nflip, nwrong
        if name not in ("v1_trackteam", "oracle"):
            ev = diagnose(per, gm, xy, n)
            r["events"] = ev
        out[name] = r
    out["n_frames"] = n
    out["n_tracklets"] = len(tr)
    out["n_pieces"] = len(pieces)
    out["n_gt_ids"] = len(np.unique(np.concatenate([per[k]["tid"] for k in range(n)])))
    return seq, out


KEYS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDsw", "pred_ids_per_gt_id", "n_pred_ids"]


def summarise(res, variants=("v1", "v1_trackteam", "v2", "oracle"), nboot=2000):
    seqs = sorted(res)
    summ = {}
    for v in variants:
        s = {k: float(np.mean([res[q][v][k] for q in seqs])) for k in KEYS}
        tf = sum(res[q][v]["team_frames"] for q in seqs)
        s["team_acc"] = sum(res[q][v]["team_correct"] for q in seqs) / tf
        s["team_wrong"] = sum(res[q][v]["team_wrong"] for q in seqs) / tf
        s["ref_given_team"] = sum(res[q][v].get("ref_team", 0) for q in seqs) / max(sum(res[q][v].get("ref_frames", 0) for q in seqs), 1)
        gf = sum(res[q][v].get("gk_frames", 0) for q in seqs)
        s["gk_acc"] = sum(res[q][v].get("gk_correct", 0) for q in seqs) / max(gf, 1)
        nt = sum(res[q][v]["tracks_scored"] for q in seqs)
        s["tracks_team_flip_frac"] = sum(res[q][v]["tracks_team_flip"] for q in seqs) / nt
        s["tracks_team_wrong_frac"] = sum(res[q][v]["tracks_team_wrong"] for q in seqs) / nt
        if "events" in res[seqs[0]][v]:
            mins = sum(res[q]["n_frames"] for q in seqs) / 25 / 60
            ev = [e for q in seqs for e in res[q][v]["events"]]
            s["id_changes_per_min"] = len(ev) / mins
            s["events"] = {}
            for kind in ("switch", "frag"):
                for cause in ("crossing", "det_gap", "reentry", "other"):
                    c = sum(1 for e in ev if e["kind"] == kind and e["cause"] == cause)
                    s["events"]["%s_%s_per_min" % (kind, cause)] = round(c / mins, 3)
        summ[v] = s
    rng = np.random.default_rng(0)
    ci = {}
    for k in ["HOTA", "DetA", "AssA", "IDF1", "IDsw", "pred_ids_per_gt_id"]:
        d = np.array([res[q]["v2"][k] - res[q]["v1"][k] for q in seqs])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)]
        ci[k] = dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                     wins=int((d > 0).sum()), losses=int((d < 0).sum()))
    d = np.array([(res[q]["v2"]["team_correct"] - res[q]["v1"]["team_correct"]) / max(res[q]["v1"]["team_frames"], 1)
                  for q in seqs])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)]
    ci["team_acc"] = dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    summ["ci_v2_minus_v1"] = ci
    summ["purity"] = sum(res[q]["purity"]["num"] for q in seqs) / max(sum(res[q]["purity"]["den"] for q in seqs), 1)
    summ["pieces_per_clip"] = float(np.mean([res[q]["purity"]["n_pieces"] for q in seqs]))
    summ["piece_len_median"] = float(np.median([res[q]["purity"]["len_median"] for q in seqs]))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--set", default="{}")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--hold", type=int, default=0)
    a = ap.parse_args()
    ov = json.loads(a.set)
    files = sorted(glob.glob(os.path.join(PACKS, a.split + "_SNGS-*.npz")))
    if a.hold:
        files = [f for i, f in enumerate(files) if i % a.hold == 0]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(eval_clip, [(f, ov) for f in files]))
    summ = summarise(res)
    os.makedirs(OUTD, exist_ok=True)
    slim = {q: {v: ({k: x for k, x in r.items() if k != "events"} if isinstance(r, dict) else r)
                for v, r in res[q].items()} for q in res}
    json.dump(dict(split=a.split, overrides=ov, summary=summ, per_clip=slim),
              open(os.path.join(OUTD, "gsr_%s_%s.json" % (a.split, a.tag)), "w"), indent=1)
    if a.quiet:
        print(a.tag, ov, "purity %.4f pieces/clip %.1f len_med %.0f" % (summ["purity"], summ["pieces_per_clip"],
              summ["piece_len_median"]), {v: {k: round(summ[v][k], 4) for k in ("HOTA", "AssA", "IDF1", "team_acc", "gk_acc", "ref_given_team")}
                                          for v in ("v1", "v2", "oracle")}, flush=True)
    else:
        print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
