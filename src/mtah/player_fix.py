"""Optional corrected PLAYER tracking of the provided match for MATCH INFERENCE (FINAL key "player_fix"; E205 Track
S3ship; the method is Track T2's E182 / E196 "T2team" input, moved here from tools/trackT2 unchanged in substance).

Mode "T2team" = the provided tactical tracks with
  (a) the team of a track corrected where the kit colour of OUR player detections on the tactical video contradicts it
      for a sustained stretch (windowed evidence) or for most of a track fragment,
  (b) goalkeeper-kit calls assigned to the team defending the goal the keeper stands in (keeper tracks found by rule:
      median distance to the nearest goal line < penalty-area depth, present > 60 s; defending side from where the home
      keeper track stands),
  (c) referee-kit stretches removed (not a player).
Mode "T2team_refill" (Track T2's PTR2) = T2team + players the provided tracks miss, refilled near the ball from OUR
  tracker (vision.k2track, version = recipe parameter, default v6): per frame and team up to 11 - provided players
  present, weak tracks refused at a measured confidence floor, within 10 m of the ball (+-2 s), inside the field lines.
No slot number, frame number or match-specific constant: kit clusters / group names / thresholds are measured on the
match's own label-free statistics; the remaining constants are Laws of the Game / geometry / evidence-strength design
constants fixed in E182 before any score (notes/trackT2/README.md "Hard-coding audit"). No ground truth is read.

Inputs: cache/atd_tactical.npz (provided tracking), cache/homography_tactical.npz (run.ensure_homography), the
player detections cache/vision/players_cands/pc_*.npz (src/vision/run_players.py detect: shipped external-trained YOLO
weights, frame / box / conf / 48-bin YUV torso kit histogram per detection) and 40 frames of the tactical video (grass
colour, CPU ffmpeg). Output: one npz with ball (unchanged), players (N, 2S, 3) and team (2S,): slot s keeps its
provided team, slot S + s (team 1 - team[s]) carries slot s's positions on the flipped frames (the members' feature
code takes any number of slots, team per slot).

    PYTHONPATH=src .venv/bin/python -m mtah.player_fix build --mode T2team --out cache/player_fix/atd_tactical_T2team.npz
    PYTHONPATH=src .venv/bin/python -m mtah.player_fix build --mode T2team_refill --tracker v6 \
        --out cache/player_fix/atd_tactical_PTR2_v6.npz     # + cache/vision/ball_video_refD3.npz (run.py step vision)

Inference hook (mtah.final_signals --tracking FILE, after the fit): use_tracking(FILE) makes mtah.match.load_tracking
return FILE's players / team for the tactical source; training, checkpoint keys and every other input are unchanged.

Determinism: pure numpy / scipy; the two k-means fits run single-threaded (threadpoolctl) because sklearn's
multi-threaded Lloyd reduction order is not fixed; the kit-colour k-means is seeded (random_state 0, sample
default_rng(0)). Intermediate stages are cached in the work dir (resumable).
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODES = ("T2team", "T2team_refill", "OWN")

GATE_M = 2.0
PEN_DEPTH = 16.5
MIN_PRESENT_S = 60
K_REF = 3
H_CLEAN = 40
MARGIN_CONF = 0.15
GRASS_BIN = 0.005
GRASS_MAX = 0.7
WIN, NMIN, SHARE = 12, 5, 0.7
FRAG_MIN, FRAG_SHARE = 10, 0.5
TEAM_SIZE = 11
REFILL_SEP_M = 1.0
NEAR_M, NEAR_DILATE_F = 10.0, 50
MIN_TRACK_F = 25
MAX_INTERP_F = 25


def log(msg):
    print("[player_fix] %s" % msg, flush=True)


def content_hash(path, keys=None):
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(keys or z.files):
            a = np.ascontiguousarray(z[k])
            h.update(("%s|%s|%s|" % (k, a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def cands_hash(cdir):
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(cdir, "pc_*.npz"))):
        with np.load(f) as z:
            for k in ("frame", "box", "conf", "feat"):
                a = np.ascontiguousarray(z[k])
                h.update(("%s|%s|%s|%s|" % (os.path.basename(f), k, a.dtype.str, a.shape)).encode())
                h.update(a.tobytes())
    return h.hexdigest()


def use_tracking(path):
    import mtah.match as mm
    orig = mm.load_tracking
    with np.load(path) as z:
        P, T = z["players"].astype(np.float32), z["team"].astype(np.int8)

    def load_tracking_fixed(source="tactical", refresh=False):
        t = dict(orig(source, refresh))
        if source == "tactical":
            t["players"], t["team"] = P, T
        return t
    mm.load_tracking = load_tracking_fixed
    log("match inference reads players / team from %s (%d slots)" % (path, len(T)))


def half_of(frames):
    from vision.common import HALVES
    h = np.zeros(len(frames), int)
    for k, a, b in HALVES:
        h[(frames >= a) & (frames <= b)] = k
    return h


def keeper_slots(P):
    import warnings
    from vision.common import HALVES, PITCH, FPS
    per = {}
    for k, a, b in HALVES:
        px = P[a:b + 1, :, 0]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gl = np.nanmedian(np.minimum(px, PITCH[0] - px), 0)
        n = np.isfinite(px).sum(0)
        per[k] = np.flatnonzero((gl < PEN_DEPTH) & (n > FPS * MIN_PRESENT_S))
    allk = np.unique(np.concatenate(list(per.values()))) if per else np.zeros(0, int)
    return per, allk


def embed(feat, grass):
    f = feat.astype(np.float32)
    gb = grass > GRASS_BIN
    gshare = f[:, gb].sum(1)
    f[:, gb] = 0
    s = f.sum(1, keepdims=True)
    return np.sqrt(f / np.maximum(s, 1e-6)), gshare


def kmeans(k, X):
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return KMeans(k, n_init=4, random_state=0).fit(X)


def stage_assoc(atd, cdir):
    from scipy.optimize import linear_sum_assignment
    from vision.common import W, H, HALVES, load_homography, metres_to_px, px_to_metres
    P = atd["players"][..., :2].astype(np.float64)
    N, S = P.shape[:2]
    Hs = load_homography()
    files = sorted(glob.glob(os.path.join(cdir, "pc_*.npz")))
    if not files:
        raise SystemExit("no player detections in %s (run src/vision/run_players.py detect)" % cdir)
    parts = [np.load(f) for f in files]
    fr = np.concatenate([p["frame"] for p in parts]).astype(np.int64)
    box = np.concatenate([p["box"] for p in parts]).astype(np.float32)
    conf = np.concatenate([p["conf"] for p in parts]).astype(np.float32)
    feat = np.concatenate([p["feat"] for p in parts])
    o = np.argsort(fr, kind="stable")
    fr, box, conf, feat = fr[o], box[o], conf[o], feat[o]
    bounds = np.searchsorted(fr, np.arange(N + 1))
    occ_all = np.zeros(len(fr), np.float32)
    PF, PS, PD, PDist = [], [], [], []
    for half, ga, gb in HALVES:
        for t in range(ga, gb + 1):
            ap = P[t]
            av = np.flatnonzero(np.isfinite(ap[:, 0]))
            s, e = bounds[t], bounds[t + 1]
            b = box[s:e].astype(np.float64)
            if e - s > 1:
                ix0 = np.maximum(b[:, None, 0], b[None, :, 0]); ix1 = np.minimum(b[:, None, 2], b[None, :, 2])
                iy0 = np.maximum(b[:, None, 1], b[None, :, 1]); iy1 = np.minimum(b[:, None, 3], b[None, :, 3])
                inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
                np.fill_diagonal(inter, 0)
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                occ_all[s:e] = (inter.max(1) / np.maximum(area, 1)).astype(np.float32)
            if not len(av) or e <= s:
                continue
            Hinv = np.linalg.inv(Hs[t])
            x, y = px_to_metres(Hinv, (b[:, 0] + b[:, 2]) / 2, b[:, 3])
            D = np.hypot(x[:, None] - ap[av][None, :, 0], y[:, None] - ap[av][None, :, 1])
            D = np.where(np.isfinite(D), D, 1e6)
            r, c = linear_sum_assignment(np.where(D < GATE_M, D, 1e6))
            k = D[r, c] < GATE_M
            r, c = r[k], c[k]
            PF.append(np.full(len(r), t)); PS.append(av[c]); PD.append(s + r); PDist.append(D[r, c])
        log("assoc: half %d done" % half)
    pd_ = np.concatenate(PD)
    A = {"frame": np.concatenate(PF).astype(np.int32), "slot": np.concatenate(PS).astype(np.int8),
         "det": pd_.astype(np.int64), "dist": np.concatenate(PDist).astype(np.float32), "box": box[pd_],
         "conf": conf[pd_], "occ": occ_all[pd_], "feat": feat[pd_], "team": atd["team"]}
    Dt = {"frame": fr.astype(np.int32), "box": box, "conf": conf, "feat": feat, "occ": occ_all}
    return A, Dt


def stage_grass(Dt):
    from vision.common import W, H, FPS, HALVES, VIDEO, load_homography, metres_to_px, check_video_size
    from vision.players_common import YB, UB, VB, NFEAT
    check_video_size(VIDEO)
    Hs = load_homography()
    fr = Dt["frame"]
    rng = np.random.default_rng(0)
    frames = np.r_[np.linspace(HALVES[0][1] + 500, HALVES[0][2] - 500, 20),
                   np.linspace(HALVES[1][1] + 500, HALVES[1][2] - 500, 20)].astype(int)
    tot = np.zeros(NFEAT)
    for t in frames:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", "%.4f" % (t / float(FPS)), "-i", VIDEO, "-frames:v", "1",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
        a = np.frombuffer(subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout, np.uint8)
        Y = a[:W * H].reshape(H, W)
        U = a[W * H:W * H + W * H // 4].reshape(H // 2, W // 2)
        V = a[W * H + W * H // 4:].reshape(H // 2, W // 2)
        xm = rng.uniform(1, 104, 4000); ym = rng.uniform(1, 67, 4000)
        u, v = metres_to_px(Hs[t], xm, ym)
        ok = np.isfinite(u) & (u > 0) & (u < W - 1) & (v > 0) & (v < H - 1)
        u, v = u[ok].astype(int), v[ok].astype(int)
        far = np.ones(len(u), bool)
        for x0, y0, x1, y1 in Dt["box"][fr == t]:
            far &= ~((u > x0 - 20) & (u < x1 + 20) & (v > y0 - 20) & (v < y1 + 20))
        u, v = u[far], v[far]
        y_, u_, v_ = Y[v, u], U[v // 2, u // 2], V[v // 2, u // 2]
        yi = np.clip((y_.astype(np.int32) - 16) * YB // 220, 0, YB - 1)
        ui = np.clip((u_.astype(np.int32) - 64) * UB // 128, 0, UB - 1)
        vi = np.clip((v_.astype(np.int32) - 64) * VB // 128, 0, VB - 1)
        tot += np.bincount((yi * UB + ui) * VB + vi, minlength=NFEAT)
    return tot / tot.sum()


def fragments(P):
    from vision.common import HALVES
    fr = []
    for _, ga, gb in HALVES:
        ok = np.isfinite(P[ga:gb + 1, :, 0])
        for s in range(P.shape[1]):
            v = np.r_[False, ok[:, s], False].astype(np.int8)
            d = np.diff(v)
            fr += [(s, ga + a, ga + b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return np.array(fr, np.int64)


def kit_groups():
    sys.path.insert(0, ROOT)
    from games import GAME, GAMES
    g = GAMES[GAME]
    try:
        home, away = g["home_kit"], g["away_kit"]
    except KeyError as e:
        raise SystemExit("player_fix: the match description (games.py) has no %s: add the main colours of the two "
                         "outfield kits as home_kit / away_kit" % e)
    n = len(home) + len(away) + 2
    log("kit colour groups: %d = home kit %s + away kit %s + 2 goalkeeper kits" % (n, home, away))
    return n


def stage_colour(A, grass, players):
    team = A["team"].astype(int)
    slot = A["slot"].astype(int)
    h = A["box"][:, 3] - A["box"][:, 1]
    E, gshare = embed(A["feat"], grass)
    clean = (h >= H_CLEAN) & (A["occ"] == 0) & (A["dist"] < 0.5) & (A["conf"] > 0.5) & (gshare < GRASS_MAX)
    idx = np.flatnonzero(clean)
    samp = np.random.default_rng(0).choice(idx, min(300000, len(idx)), replace=False)
    k_kit = kit_groups()
    C = kmeans(k_kit, E[samp]).cluster_centers_
    lab = np.empty(len(E), np.int8); d1 = np.empty(len(E), np.float32)
    for s in range(0, len(E), 500000):
        D = np.sqrt(((E[s:s + 500000, None, :] - C[None]) ** 2).sum(-1))
        lab[s:s + 500000] = D.argmin(1); d1[s:s + 500000] = D.min(1)
    _, KS = keeper_slots(players[..., :2])
    gk = np.isin(slot, KS)
    cgrp = []
    for c in range(k_kit):
        m = clean & (lab == c)
        t0, gks = float((team[slot[m]] == 0).mean()), float(gk[m].mean())
        cgrp.append(2 if gks > 0.5 else 0 if t0 > 0.8 else 1 if t0 < 0.2 else -1)
        log("kit cluster %d: clean pairs %d, home share %.3f, keeper-track share %.3f -> group %d"
            % (c, int(m.sum()), t0, gks, cgrp[-1]))
    return {"E": E, "gshare": gshare, "C": C, "d1": d1, "clean": clean, "cgrp": cgrp}


def stage_groups(A, Dt, Cl, grass):
    from vision.common import load_homography, PITCH
    matched = np.zeros(len(Dt["frame"]), bool); matched[A["det"]] = True
    Hs = load_homography()
    fr = Dt["frame"].astype(int); b = Dt["box"].astype(np.float64)
    h = b[:, 3] - b[:, 1]
    ci_ = np.flatnonzero((~matched) & (h >= H_CLEAN) & (Dt["occ"] == 0) & (Dt["conf"] > 0.5))
    Hi = np.linalg.inv(Hs[fr[ci_]])
    u = (b[ci_, 0] + b[ci_, 2]) / 2; v = b[ci_, 3]
    p = np.einsum("nij,nj->ni", Hi, np.c_[u, v, np.ones_like(u)])
    x = p[:, 0] / p[:, 2] * PITCH[0]; y = p[:, 1] / p[:, 2] * PITCH[1]
    onp = (x > 0.5) & (x < PITCH[0] - 0.5) & (y > 0.5) & (y < PITCH[1] - 0.5)
    Er, gsr = embed(Dt["feat"][ci_[onp]], grass)
    Er = Er[gsr < GRASS_MAX]
    Dm = np.sqrt(((Er[:, None] - Cl["C"][None]) ** 2).sum(-1)).min(1)
    thr = np.percentile(Cl["d1"][Cl["clean"]], 99)
    pool = Er[Dm > thr]
    log("referee pool: %d of %d unmatched clean on-pitch detections beyond the 99th pct kit distance %.4f"
        % (len(pool), len(Er), thr))
    km = kmeans(K_REF, pool[np.random.default_rng(0).choice(len(pool), min(100000, len(pool)), replace=False)])
    Call = np.r_[Cl["C"], km.cluster_centers_]
    G = np.r_[np.array(Cl["cgrp"]), [3] * K_REF]
    E = Cl["E"]
    grp = np.empty(len(E), np.int8); margin = np.empty(len(E), np.float32)
    for s in range(0, len(E), 400000):
        D = np.sqrt(((E[s:s + 400000, None, :] - Call[None]) ** 2).sum(-1))
        g = G[D.argmin(1)]
        own = np.full(len(D), np.inf); oth = np.full(len(D), np.inf)
        for gg in np.unique(G):
            dg = D[:, G == gg].min(1)
            own = np.where(g == gg, dg, own)
            oth = np.where(g != gg, np.minimum(oth, dg), oth)
        grp[s:s + 400000] = g; margin[s:s + 400000] = oth - own
    return grp, margin


def stage_calls(A, grp, margin, gshare, P, team_atd, F):
    from vision.common import HALVES, PITCH
    team = A["team"].astype(int)
    fr, sl = A["frame"].astype(int), A["slot"].astype(int)
    grp = grp.astype(int)
    per, KS = keeper_slots(P)
    sides = {}
    for half, ga, gb in HALVES:
        k0 = [s for s in per[half] if team_atd[s] == 0]
        sides[half] = 0 if np.nanmedian(P[ga:gb + 1, k0, 0]) < PITCH[0] / 2 else 1
    half = half_of(fr)
    side = (P[fr, sl, 0] >= PITCH[0] / 2).astype(int)
    gk_team = np.where(side == np.where(half == 1, sides[1], sides[2]), 0, 1)
    imp = np.where(grp == 2, gk_team, grp)
    h = A["box"][:, 3] - A["box"][:, 1]
    conf = (h >= H_CLEAN) & (A["occ"] == 0) & (A["dist"] < 1.0) & (margin >= MARGIN_CONF) & (gshare < GRASS_MAX)
    frag = np.full(len(fr), -1, np.int32)
    for s_ in range(P.shape[1]):
        ids = np.flatnonzero(F[:, 0] == s_)
        m = sl == s_
        k = np.searchsorted(F[ids, 1], fr[m], side="right") - 1
        ok = (k >= 0) & (fr[m] < F[ids[np.maximum(k, 0)], 2])
        frag[np.flatnonzero(m)[ok]] = ids[k[ok]]
    nf = len(F)
    cnt = np.zeros((nf, 4)); np.add.at(cnt, (frag[conf & (frag >= 0)], imp[conf & (frag >= 0)]), 1)
    ncf = cnt.sum(1)
    agree = cnt[np.arange(nf), team[F[:, 0]]]
    fdis = np.where(ncf > 0, 1 - agree / np.maximum(ncf, 1), np.nan)
    return imp.astype(np.int8), conf, cnt, ncf, fdis


def stage_evidence(A, imp, conf, N, S):
    team = A["team"].astype(int)
    fr, sl = A["frame"].astype(int), A["slot"].astype(int)
    imp = imp.astype(int)
    cnt = np.zeros((N + 1, S, 3), np.int32)
    cls = np.where(imp == team[sl], 0, np.where(imp == 3, 2, 1))
    np.add.at(cnt, (fr[conf] + 1, sl[conf], cls[conf]), 1)
    cs = np.cumsum(cnt, 0)
    lo = np.clip(np.arange(N) - WIN, 0, N); hi = np.clip(np.arange(N) + WIN + 1, 0, N)
    w = cs[hi] - cs[lo]
    n = w.sum(-1)
    contra = (n >= NMIN) & ((w[..., 1] + w[..., 2]) >= SHARE * n)
    confirm = (n >= NMIN) & (w[..., 0] >= SHARE * n)
    kind = np.where(w[..., 2] > w[..., 1], 3, 1 - team[None, :]).astype(np.int8)
    return contra, confirm, kind


def correct_T2team(atd, F, ncf, fdis, cnt, contra, confirm, kind):
    P3, team, ball = atd["players"], atd["team"].astype(int), atd["ball"]
    N, S = P3.shape[:2]
    pres = np.isfinite(P3[..., 0].astype(np.float64))
    kind = kind.astype(int)
    newteam = np.where(pres, team[None, :], -1).astype(np.int8)
    ref = np.zeros((N, S), bool)
    for k in np.flatnonzero((ncf >= FRAG_MIN) & (fdis > FRAG_SHARE)):
        s, a_, b_ = F[k]
        maj = int(cnt[k].argmax())
        keep = confirm[a_:b_, s]
        if maj == 3:
            ref[a_:b_, s] |= ~keep
        else:
            newteam[a_:b_, s] = np.where(keep, team[s], maj)
    c_other = contra & (kind != 3) & pres
    newteam[c_other] = kind[c_other]
    ref |= contra & (kind == 3) & pres
    newteam[ref] = -1
    flipped = pres & (newteam >= 0) & (newteam != team[None, :])
    Pn = np.full((N, 2 * S, 3), np.nan, np.float32)
    keep_orig = pres & ~flipped & ~ref
    Pn[:, :S][keep_orig] = P3[keep_orig]
    Pn[:, S:][flipped] = P3[flipped]
    team2 = np.r_[team, 1 - team].astype(np.int8)
    log("T2team: %d player-frames flipped, %d referee player-frames removed, %d slots"
        % (int(flipped.sum()), int(ref.sum()), 2 * S))
    return {"ball": ball, "players": Pn, "team": team2}, {"flipped": flipped, "referee": ref}


def weak_conf(conf, state, atd_slot):
    return float(np.percentile(conf[(state == 1) & (atd_slot >= 0)], 1))


def refill(atd, v, ball_xy):
    from scipy.ndimage import binary_dilation
    P, team, ball = atd["players"].astype(np.float64), atd["team"].astype(np.int64), atd["ball"]
    N = P.shape[0]
    vxy, vt, vtid, vas, vst, vsrc = v["xy"], v["team"], v["track_id"], v["atd_slot"], v["state"], v["team_src"]
    cand = (vt >= 0) & (vas == -1) & np.isin(vst, (1, 2)) & np.isin(vsrc, (1, 2, 3)) & np.isfinite(vxy[..., 0])
    ids = vtid[cand]
    conf_det = np.where(vst == 1, v["conf"], np.nan)
    u, inv = np.unique(vtid[vtid >= 0], return_inverse=True)
    n_t = np.bincount(inv)
    med = {}
    cd = conf_det[vtid >= 0]
    order = np.argsort(inv, kind="stable")
    for k, seg in enumerate(np.split(cd[order], np.cumsum(n_t)[:-1])):
        med[int(u[k])] = np.nanmedian(seg) if np.isfinite(seg).any() else 0.0
    cmin = weak_conf(v["conf"], vst, vas)
    log("refill: weak-track confidence floor (1st pct of provider-confirmed detections) %.4f" % cmin)
    okt = np.array([n_t[np.searchsorted(u, i)] >= MIN_TRACK_F and med[int(i)] >= cmin for i in ids])
    last_det = np.full(vtid.shape, -10**9, np.int64)
    seen = {}
    for t in range(len(vtid)):
        for j in np.flatnonzero(vtid[t] >= 0):
            i = int(vtid[t, j])
            if vst[t, j] == 1:
                seen[i] = t
            last_det[t, j] = t - seen.get(i, -10**9)
    okg = last_det[cand] <= MAX_INTERP_F
    cc = cand.copy(); cc[cand] = okt & okg
    cand = cc
    sel = np.zeros_like(cand)
    for t in np.flatnonzero(cand.any(1)):
        c = np.flatnonzero(cand[t])
        for g in (0, 1):
            cg = c[vt[t, c] == g]
            if not len(cg):
                continue
            pa = P[t, team == g, :2]
            pa = pa[np.isfinite(pa[:, 0])]
            k = max(0, TEAM_SIZE - len(pa))
            if k == 0:
                continue
            d = np.sqrt(((vxy[t, cg][:, None, :] - pa[None]) ** 2).sum(-1)).min(1) if len(pa) \
                else np.full(len(cg), np.inf)
            ok = d > REFILL_SEP_M
            cg, d = cg[ok], d[ok]
            sel[t, cg[np.argsort(-d)[:k]]] = True
    extra, eteam = [], []
    for g in (0, 1):
        K = int((sel & (vt == g)).sum(1).max())
        X = np.full((N, K, 2), np.nan)
        slot_of = {}
        for t in range(N):
            tids = [(int(vtid[t, j]), j) for j in np.flatnonzero(sel[t] & (vt[t] == g))]
            used, keep = set(), {}
            for tid, j in tids:
                s = slot_of.get(tid)
                if s is not None and s not in used:
                    used.add(s); keep[tid] = (s, j)
            for tid, j in tids:
                if tid not in keep:
                    s = next(i for i in range(K) if i not in used)
                    used.add(s); keep[tid] = (s, j)
            slot_of = {tid: s for tid, (s, j) in keep.items()}
            for tid, (s, j) in keep.items():
                X[t, s] = vxy[t, j]
        extra.append(X); eteam += [g] * K
    UA = np.concatenate(extra, 1).astype(np.float32)
    UT = np.array(eteam, np.int8)
    bxy = ball[:, :2].astype(np.float64)
    bxy = np.where(np.isfinite(bxy), bxy, ball_xy.astype(np.float64))
    upres = np.isfinite(UA[..., 0])
    ud = np.hypot(UA[..., 0] - bxy[:, None, 0], UA[..., 1] - bxy[:, None, 1])
    unear = binary_dilation(upres & (ud <= NEAR_M), structure=np.ones((2 * NEAR_DILATE_F + 1, 1), bool)) & upres
    return UA, UT, unear


def add_refill(T, UA, UT, unear, inside_field=True):
    from vision.common import PITCH
    x, y = UA[..., 0], UA[..., 1]
    keep = unear & ((x >= 0) & (x <= PITCH[0]) & (y >= 0) & (y <= PITCH[1]) if inside_field else True)
    A2 = np.full(UA.shape[:2] + (3,), np.nan, np.float32)
    A2[..., :2] = np.where(keep[..., None], UA, np.nan)
    log("refill: %d near-ball player-frames kept of %d (inside the field: %s)" % (int(keep.sum()), int(unear.sum()),
                                                                                  inside_field))
    return {"ball": T["ball"], "players": np.concatenate([T["players"], A2], 1),
            "team": np.r_[T["team"], UT].astype(np.int8)}


OWN_REFEREE_SRC = 4


def _track_conf(v):
    tid, st, conf = v["track_id"], v["state"], v["conf"]
    m = (tid >= 0) & (st == 1) & np.isfinite(conf)
    ids, c = tid[m], conf[m].astype(np.float64)
    o = np.argsort(ids, kind="stable")
    ids, c = ids[o], c[o]
    u, start = np.unique(ids, return_index=True)
    return {int(i): float(np.median(seg)) for i, seg in zip(u, np.split(c, start[1:]))}


class _Slots:

    def __init__(self, team_vec):
        self.block = {g: list(np.flatnonzero(team_vec == g)) for g in (0, 1)}
        self.last_used = {s: -10 ** 9 for s in range(len(team_vec))}
        self.owner_last = {s: None for s in range(len(team_vec))}
        self.slot_of, self.active = {}, {}
        self.handover, self.moved = [], 0

    def assign(self, t, keys_by_team):
        new_active = {}
        for g, keys in keys_by_team.items():
            used = set()
            for k in keys:
                s = self.active.get(k)
                if s is not None:
                    new_active[k] = s; used.add(s)
            for k in keys:
                if k in new_active:
                    continue
                s = self.slot_of.get(k)
                if s is not None and s not in used and self.owner_last[s] == k:
                    new_active[k] = s; used.add(s)
            for k in keys:
                if k in new_active:
                    continue
                free = [s for s in self.block[g] if s not in used and s not in self.active.values()]
                if not free:
                    free = [s for s in self.block[g] if s not in used]
                s = min(free, key=lambda s_: (self.last_used[s_], s_))
                if k in self.slot_of:
                    self.moved += 1
                if self.owner_last[s] is not None and self.owner_last[s] != k:
                    self.handover.append(t - self.last_used[s])
                new_active[k] = s; used.add(s)
        for k, s in new_active.items():
            self.slot_of[k] = s
            self.last_used[s] = t
            self.owner_last[s] = k
        self.active = new_active
        return new_active


def build_own(atd, v):
    PA = atd["players"]
    TA = atd["team"].astype(np.int8)
    N, S = PA.shape[:2]
    xy, vt, tid, st, src, conf = (v["xy"].astype(np.float64), v["team"], v["track_id"], v["state"], v["team_src"],
                                  v["conf"])
    med = _track_conf(v)
    ok = (vt >= 0) & (src != OWN_REFEREE_SRC) & np.isin(st, (1, 2)) & np.isfinite(xy[..., 0]) & (tid >= 0)
    P = np.full((N, S, 3), np.nan, np.float32)
    SL = _Slots(TA)
    sub_pos = {}
    capped = dup = 0
    for t in range(N):
        jj = np.flatnonzero(ok[t])
        keys_by_team = {0: [], 1: []}
        pos = {}
        for g in (0, 1):
            jg = jj[vt[t, jj] == g]
            if len(jg) > TEAM_SIZE:
                tc = np.array([med.get(int(i), 0.0) for i in tid[t, jg]])
                ec = np.nan_to_num(conf[t, jg].astype(np.float64))
                o = np.lexsort((jg, -ec, -tc))
                capped += len(jg) - TEAM_SIZE
                jg = jg[o[:TEAM_SIZE]]
            ids, inv = np.unique(tid[t, jg], return_inverse=True)
            for u_i, i in enumerate(ids):
                js = jg[inv == u_i]
                if len(js) == 1:
                    keys = [(int(i), 0)]
                else:
                    dup += len(js) - 1
                    prev = [(k, p) for k, p in sub_pos.items() if k[0] == int(i)]
                    keys, taken = [None] * len(js), set()
                    pairs = sorted((float(np.hypot(*(xy[t, j] - p))), a_, k) for a_, j in enumerate(js) for k, p in prev)
                    for _, a_, k in pairs:
                        if keys[a_] is None and k not in taken:
                            keys[a_] = k; taken.add(k)
                    nxt = max([k[1] for k, _ in prev] + [0]) + 1
                    for a_ in range(len(js)):
                        if keys[a_] is None:
                            keys[a_] = (int(i), nxt); nxt += 1
                for k, j in zip(keys, js):
                    sub_pos[k] = xy[t, j]
                    keys_by_team[g].append(("o",) + k)
                    pos[("o",) + k] = xy[t, j]
        for k, s in SL.assign(t, keys_by_team).items():
            P[t, s, :2] = pos[k]
    ho = np.array(SL.handover)
    stats = {"capped_entries": int(capped), "dup_track_entries": int(dup), "identities": len(SL.slot_of),
             "slot_handovers": int(len(ho)), "handovers_within_10_frames": int((ho <= 10).sum()) if len(ho) else 0,
             "identities_moved_slot": int(SL.moved)}
    return {"ball": atd["ball"], "players": P, "team": TA}, stats


def build_own_file(tracker_file, out, work):
    t0 = time.time()
    os.makedirs(work, exist_ok=True)
    with np.load(os.path.join(ROOT, "cache", "atd_tactical.npz")) as z:
        atd = {k: z[k] for k in z.files}
    with np.load(tracker_file) as z:
        v = {k: z[k] for k in ("xy", "team", "track_id", "conf", "state", "team_src")}
    T, stats = build_own(atd, v)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out + ".tmp.npz", **T)
    os.replace(out + ".tmp.npz", out)
    rep = {"mode": "OWN", "tracker_file": tracker_file, "tracker_content_sha256": content_hash(tracker_file),
           "out": out, "content_sha256": content_hash(out),
           "file_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(), "runtime_s": round(time.time() - t0, 1),
           **stats}
    json.dump(rep, open(os.path.join(work, "build_OWN.json"), "w"), indent=1)
    log("OWN: wrote %s | content %s | file sha256 %s | tracker %s | %s | %.0f s" % (
        out, rep["content_sha256"][:16], rep["file_sha256"][:16], rep["tracker_content_sha256"][:16],
        json.dumps(stats), rep["runtime_s"]))
    return rep


def build(mode, out, cdir, work, tracker="v6", tracker_file=None):
    if mode not in MODES:
        raise SystemExit("player_fix mode %r not implemented (have %s)" % (mode, MODES))
    if mode == "OWN":
        if not tracker_file:
            raise SystemExit("mode OWN needs --tracker-file (a players_video npz of our tracker)")
        return build_own_file(tracker_file, out, work)
    t0 = time.time()
    os.makedirs(work, exist_ok=True)
    atd = dict(np.load(os.path.join(ROOT, "cache", "atd_tactical.npz")))
    P = atd["players"][..., :2].astype(np.float64)
    N, S = P.shape[:2]
    ch = cands_hash(cdir)
    log("detections %s: content %s" % (cdir, ch[:16]))
    fa = os.path.join(work, "assoc_%s.npz" % ch[:16])
    fd = os.path.join(work, "dets_%s.npz" % ch[:16])
    if os.path.exists(fa) and os.path.exists(fd):
        A, Dt = dict(np.load(fa)), dict(np.load(fd))
        log("assoc reused (%s)" % os.path.basename(fa))
    else:
        A, Dt = stage_assoc(atd, cdir)
        np.savez(fa + ".tmp.npz", **A); os.replace(fa + ".tmp.npz", fa)
        np.savez(fd + ".tmp.npz", **Dt); os.replace(fd + ".tmp.npz", fd)
        log("assoc: %d pairs, %d detections, %.0f s" % (len(A["frame"]), len(Dt["frame"]), time.time() - t0))
    fg = os.path.join(work, "grass_%s.npy" % ch[:16])
    if os.path.exists(fg):
        grass = np.load(fg)
    else:
        grass = stage_grass(Dt)
        np.save(fg + ".tmp.npy", grass); os.replace(fg + ".tmp.npy", fg)
    log("grass top bins %s, %.0f s" % ([int(i) for i in np.argsort(-grass)[:4]], time.time() - t0))
    F = fragments(P)
    Cl = stage_colour(A, grass, atd["players"])
    grp, margin = stage_groups(A, Dt, Cl, grass)
    log("kit groups done, %.0f s" % (time.time() - t0))
    imp, conf, cnt, ncf, fdis = stage_calls(A, grp, margin, Cl["gshare"], P, atd["team"].astype(int), F)
    log("confident calls %.3f of %d pairs; fragments %d, mostly another kit %d"
        % (float(conf.mean()), len(conf), len(F), int(((ncf >= FRAG_MIN) & (fdis > FRAG_SHARE)).sum())))
    contra, confirm, kind = stage_evidence(A, imp, conf, N, S)
    T, flags = correct_T2team(atd, F, ncf, fdis, cnt, contra, confirm, kind)
    if mode == "T2team_refill":
        from vision import k2track
        ft = os.path.join(work, "players_video_%s_%s.npz" % (tracker, ch[:16]))
        if os.path.exists(ft):
            v = dict(np.load(ft))
            log("tracker %s reused (%s)" % (tracker, os.path.basename(ft)))
        else:
            v = k2track.track_match(tracker, cdir)
            np.savez_compressed(ft[:-4] + ".tmp.npz", **v); os.replace(ft[:-4] + ".tmp.npz", ft)
            log("tracker %s: %s, %.0f s" % (tracker, os.path.basename(ft), time.time() - t0))
        with np.load(os.path.join(ROOT, "cache", "vision", "ball_video_refD3.npz")) as zb:
            vb = zb["xy"]
        UA, UT, unear = refill(atd, v, vb)
        T = add_refill(T, UA, UT, unear)
        flags.update(unear=unear, UA=UA)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out + ".tmp.npz", **T)
    os.replace(out + ".tmp.npz", out)
    np.savez_compressed(os.path.join(work, "flags_%s.npz" % mode), **flags)
    rep = {"mode": mode, "tracker": tracker if mode == "T2team_refill" else None, "out": out, "cands_content_sha256": ch, "content_sha256": content_hash(out),
           "content_sha256_players_team": content_hash(out, ["players", "team"]),
           "file_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(), "runtime_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(work, "build_%s.json" % mode), "w"), indent=1)
    log("wrote %s | content %s | file sha256 %s | %.0f s" % (out, rep["content_sha256"][:16], rep["file_sha256"][:16],
                                                            rep["runtime_s"]))
    return rep


def main():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--mode", default="T2team", choices=MODES)
    b.add_argument("--out", required=True)
    b.add_argument("--cands", default=os.path.join(ROOT, "cache", "vision", "players_cands"))
    b.add_argument("--work", default=os.path.join(ROOT, "cache", "player_fix"))
    b.add_argument("--tracker", default="v6", help="our player tracker version (mode T2team_refill; vision.k2track)")
    b.add_argument("--tracker-file", default=None, help="mode OWN: our tracker's players_video npz")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.mode, a.out, a.cands, a.work, a.tracker, a.tracker_file)


if __name__ == "__main__":
    main()
