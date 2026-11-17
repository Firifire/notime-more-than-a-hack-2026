from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import common as C

CAP = 10.0
RANGE_S = 120.0
PATH_RANGE_S = 15.0
MISS = 8.0
STEP_COST = 1.0
JUMP_COST = 40.0
JUMP_REPORT_S = 0.2
PRIOR_PER_S = 0.02
CONFIRM_S = 5.0
ORDINARY_GAP_S = 1.0
SETPIECE_PASS = {"Throw-in", "Goal Kick", "Corner", "Free Kick", "Kick Off"}
SETPIECE_SHOT = {"Free Kick", "Penalty", "Corner", "Kick Off"}
SYNC_TYPES = {"Pass", "Ball Receipt*", "Carry", "Shot", "Ball Recovery"}
GK_CONTROL = {"Success", "Saved Twice", "Collected Twice", "Claim", "Won"}
WON = {"Won", "Success In Play", "Success To Team", "Success"}


def ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _outcome(e):
    for k in ("pass", "shot", "duel", "interception", "goalkeeper", "50_50", "dribble", "ball_receipt",
              "clearance"):
        o = (e.get(k) or {}).get("outcome")
        if o:
            return o["name"]
    return None


def classify(e):
    typ = e["type"]["name"]
    t = e["t"]
    dur = float(e.get("duration") or 0.0)
    out = _outcome(e)
    kind, stop, restart = "neutral", None, None
    if typ == "Pass":
        kind = "control"
        ptype = ((e["pass"].get("type") or {}).get("name"))
        if ptype in SETPIECE_PASS:
            restart = "setpiece"
        if out in ("Out", "Pass Offside"):
            stop = t + dur
    elif typ == "Shot":
        kind = "control"
        if (e["shot"].get("type") or {}).get("name") in SETPIECE_SHOT:
            restart = "setpiece"
        if out in ("Goal", "Off T", "Wayward", "Saved Off T"):
            stop = t + dur
    elif typ == "Carry":
        kind = "control"
    elif typ == "Ball Receipt*":
        kind = "neutral" if out == "Incomplete" else "control"
    elif typ == "Ball Recovery":
        kind = "unctrl" if (e.get("ball_recovery") or {}).get("recovery_failure") else "control"
    elif typ == "Dribble":
        kind = "control" if out == "Complete" else "unctrl"
    elif typ == "Goal Keeper":
        gt = e["goalkeeper"]["type"]["name"]
        if gt in ("Shot Faced", "Goal Conceded", "Penalty Conceded") and out in (None, "No Touch"):
            kind = "neutral"
        elif gt == "Collected" or out in GK_CONTROL:
            kind = "control"
        else:
            kind = "unctrl"
    elif typ in ("Interception", "50/50"):
        kind = "cond" if out in WON else "unctrl"
    elif typ == "Duel":
        dt = (e["duel"].get("type") or {}).get("name")
        kind = ("cond" if out in WON else "unctrl") if dt == "Tackle" else "neutral"
    elif typ in ("Clearance", "Block", "Miscontrol"):
        kind = "unctrl"
    elif typ in ("Foul Committed", "Foul Won"):
        k = "foul_committed" if typ == "Foul Committed" else "foul_won"
        if not (e.get(k) or {}).get("advantage"):
            stop = t
    elif typ in ("Offside", "Injury Stoppage", "Half End", "Own Goal Against"):
        stop = t
    elif typ == "Referee Ball-Drop":
        restart = "setpiece"
    if e.get("out") or (out or "").endswith("Out"):
        stop = t + (dur if typ == "Pass" else 0.0) if stop is None else stop
    return kind, stop, restart


def load_events(path: Path):
    ev = json.loads(Path(path).read_text())
    ev.sort(key=lambda e: e["index"])
    for e in ev:
        e["t"] = ts(e["timestamp"])
        e["kind"], e["stop"], e["restart"] = classify(e)
    return ev


def sb_xy(loc, L, W):
    return np.array([loc[0] * L / 120.0, (80.0 - loc[1]) * W / 80.0])


def _cost(ball, t, xy, deltas, fps):
    n = len(ball)
    j = np.rint((deltas[:, None] + t[None, :]) * fps).astype(int)
    ok = (j >= 0) & (j < n)
    bj = ball[np.clip(j, 0, n - 1)]
    d = np.linalg.norm(bj - xy[None], axis=-1)
    good = ok & ~np.isnan(d)
    d = np.where(good, np.minimum(d, CAP), 0.0)
    cnt = good.sum(1)
    return d.sum(1) / np.maximum(cnt, 1) + (cnt < 20) * 99.0, cnt


def _event_costs(ball, t, xy, offs, fps):
    n = len(ball)
    out = np.empty((len(t), len(offs)), np.float32)
    for s0 in range(0, len(t), 256):
        tt = t[s0:s0 + 256]
        j = np.rint((tt[:, None] + offs[None, :]) * fps).astype(int)
        ok = (j >= 0) & (j < n)
        bj = ball[np.clip(j, 0, n - 1)]
        d = np.linalg.norm(bj - xy[s0:s0 + 256, None], axis=-1)
        out[s0:s0 + 256] = np.where(ok & ~np.isnan(d), np.minimum(d, CAP), MISS)
    return out


def _viterbi(C):
    M, K = C.shape
    D = C[0].astype(np.float64).copy()
    back = np.zeros((M, K), np.int32)
    ar = np.arange(K)
    for m in range(1, M):
        best = ar.copy()
        val = D.copy()
        left = np.r_[np.inf, D[:-1]] + STEP_COST
        right = np.r_[D[1:], np.inf] + STEP_COST
        u = left < val
        val[u], best[u] = left[u], ar[u] - 1
        u = right < val
        val[u], best[u] = right[u], ar[u] + 1
        g = int(np.argmin(D))
        u = D[g] + JUMP_COST < val
        val[u], best[u] = D[g] + JUMP_COST, g
        back[m] = best
        D = val + C[m]
    path = np.empty(M, int)
    path[-1] = int(np.argmin(D))
    for m in range(M - 1, 0, -1):
        path[m - 1] = back[m, path[m]]
    return path


def offset_fn(sy):
    kt, ko = np.array(sy["knots_t"]), np.array(sy["knots_off"])
    return lambda t: np.interp(t, kt, ko)


def sync_period(ball, events, home_id, L, W, fps=C.FPS, rng=RANGE_S):
    sel = [e for e in events if e["type"]["name"] in SYNC_TYPES and e.get("location") and not e.get("off_camera")]
    sel.sort(key=lambda e: e["t"])
    t = np.array([e["t"] for e in sel])
    xy0 = np.array([sb_xy(e["location"], L, W) for e in sel])
    is_home = np.array([e["team"]["id"] == home_id for e in sel])
    rot = np.c_[L - xy0[:, 0], W - xy0[:, 1]]
    best = None
    coarse = np.arange(-rng, rng + 1e-9, 0.2)
    for home_plus in (True, False):
        xy = np.where((is_home == home_plus)[:, None], xy0, rot)
        c, _ = _cost(ball, t, xy, coarse, fps)
        k = int(np.argmin(c))
        fine = coarse[k] + np.arange(-1.0, 1.0 + 1e-9, 1.0 / fps)
        cf, cnt = _cost(ball, t, xy, fine, fps)
        kf = int(np.argmin(cf))
        cand = dict(home_attacks_plus_x=home_plus, delta=float(fine[kf]), cost=float(cf[kf]), n_used=int(cnt[kf]),
                    xy=xy)
        if best is None or cand["cost"] < best["cost"]:
            other = best
            best = cand
        else:
            other = cand
    xy = best.pop("xy")
    best["cost_other_side"] = float(other["cost"])
    offs = best["delta"] + np.arange(-PATH_RANGE_S, PATH_RANGE_S + 1e-9, 1.0 / fps)
    Cm = _event_costs(ball, t, xy, offs, fps)
    Cm += PRIOR_PER_S * np.abs(offs - best["delta"])[None, :].astype(np.float32)
    path = _viterbi(Cm)
    off_e = offs[path]
    ch = np.flatnonzero(np.abs(np.diff(off_e)) > 1e-6)
    keep = np.unique(np.r_[0, ch, ch + 1, len(t) - 1])
    jumps = [(round(float(t[i + 1]), 1), round(float(off_e[i + 1] - off_e[i]), 2)) for i in ch
             if abs(off_e[i + 1] - off_e[i]) > JUMP_REPORT_S]
    n = len(ball)
    j = np.rint((t + off_e) * fps).astype(int)
    okj = (j >= 0) & (j < n)
    d = np.full(len(t), np.nan)
    d[okj] = np.linalg.norm(ball[j[okj]] - xy[okj], axis=-1)
    dd = d[~np.isnan(d)]
    j0 = np.rint((t + best["delta"]) * fps).astype(int)
    ok0 = (j0 >= 0) & (j0 < n)
    d0 = np.linalg.norm(ball[j0[ok0]] - xy[ok0], axis=-1)
    d0 = d0[~np.isnan(d0)]
    best.update(knots_t=[float(x) for x in t[keep]], knots_off=[float(x) for x in off_e[keep]],
                off_min=float(off_e.min()), off_max=float(off_e.max()),
                off_first=float(off_e[0]), off_last=float(off_e[-1]), jumps=jumps, n_jumps=len(jumps),
                path_at_range_edge=bool((np.abs(off_e - best["delta"]) > PATH_RANGE_S - 0.1).any()),
                n_events_located=int(len(t)), n_events_with_ball=int(len(dd)),
                dist_median=float(np.median(dd)) if len(dd) else float("nan"),
                dist_p75=float(np.percentile(dd, 75)) if len(dd) else float("nan"),
                dist_p90=float(np.percentile(dd, 90)) if len(dd) else float("nan"),
                frac_within_2m=float((dd < 2).mean()) if len(dd) else float("nan"),
                frac_within_5m=float((dd < 5).mean()) if len(dd) else float("nan"),
                dist_median_const_offset=float(np.median(d0)) if len(d0) else float("nan"),
                frac_within_5m_const_offset=float((d0 < 5).mean()) if len(d0) else float("nan"))
    return best


def build_label(events, half, syncs, home_id, fps=C.FPS):
    n = len(half)
    lab = np.full(n, C.UNK, np.int8)
    diag = dict(n_stoppages=0, n_restart_setpiece=0, n_restart_ordinary=0, n_restart_none=0,
                n_switch=0, n_cond_confirmed=0, n_cond=0)
    for p, sy in syncs.items():
        idx = np.flatnonzero(half == p)
        if not len(idx):
            continue
        npf = len(idx)
        evp = [e for e in events if e["period"] == p]
        off = offset_fn(sy)

        def fr(t):
            return int(np.rint((t + off(t)) * fps))
        act = [e for e in evp if e["kind"] in ("control", "cond", "unctrl") and e.get("team")]
        sw = []
        for i, e in enumerate(act):
            team = C.H if e["team"]["id"] == home_id else C.A
            if e["kind"] == "control":
                sw.append((fr(e["t"]), team))
            elif e["kind"] == "cond":
                diag["n_cond"] += 1
                nx = act[i + 1] if i + 1 < len(act) else None
                if nx is not None and nx["kind"] == "control" and nx["team"]["id"] == e["team"]["id"] \
                        and nx["t"] - e["t"] <= CONFIRM_S:
                    sw.append((fr(e["t"]), team))
                    diag["n_cond_confirmed"] += 1
        diag["n_switch"] += len(sw)
        mark = np.full(npf, -1, np.int8)
        for f, team in sorted(sw, key=lambda x: x[0]):
            if 0 <= f < npf:
                mark[f] = team
            elif f < 0:
                mark[0] = team
        cur = -1
        lp = np.full(npf, C.UNK, np.int8)
        for j in range(npf):
            if mark[j] >= 0:
                cur = mark[j]
            if cur >= 0:
                lp[j] = cur
        seen_restart = set()
        for i, e in enumerate(evp):
            if e["stop"] is None:
                continue
            t0 = e["stop"]
            t1 = None
            how = "none"
            for nx in evp[i + 1:]:
                if nx["restart"] == "setpiece" and nx["t"] >= e["t"]:
                    t1, how = nx["t"], "setpiece"
                    break
                if nx["kind"] in ("control", "cond", "unctrl") and nx["t"] > t0 + ORDINARY_GAP_S:
                    t1, how = nx["t"], "ordinary"
                    break
            if (t1, how) not in seen_restart:
                seen_restart.add((t1, how))
                diag["n_stoppages"] += 1
                diag[f"n_restart_{how}"] += 1
            f0 = max(fr(t0), 0)
            f1 = npf if t1 is None else min(fr(t1), npf)
            if f1 > f0:
                lp[f0:f1] = C.D
        lab[idx] = lp
    return lab, diag


def transitions(lab, half, fps=C.FPS):
    out = {"D_start": [], "D_end": [], "switch": []}
    a, bb = lab[:-1], lab[1:]
    same = half[:-1] == half[1:]
    for i in np.flatnonzero((a != bb) & same):
        x, y = a[i], bb[i]
        if x in (C.H, C.A) and y == C.D:
            out["D_start"].append((i + 1, y))
        elif x == C.D and y in (C.H, C.A):
            out["D_end"].append((i + 1, y))
        elif x in (C.H, C.A) and y in (C.H, C.A):
            out["switch"].append((i + 1, y))
    return out


def compare(ref, new, half, fps=C.FPS, window_s=10.0):
    both = (ref != C.UNK) & (new != C.UNK)
    alive = both & np.isin(ref, [C.H, C.A]) & np.isin(new, [C.H, C.A])
    rD, nD = (ref == C.D) & both, (new == C.D) & both
    k = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            k[i, j] = ((ref == i) & (new == j) & both).sum()
    po = np.trace(k) / k.sum()
    pe = (k.sum(0) * k.sum(1)).sum() / k.sum() ** 2
    tr_r, tr_n = transitions(ref, half), transitions(new, half)
    bnd = {}
    for typ in tr_r:
        nf = np.array([f for f, _ in tr_n[typ]])
        nl = np.array([l for _, l in tr_n[typ]])
        offs = []
        for f, l in tr_r[typ]:
            m = nl == l if typ == "switch" else np.ones(len(nf), bool)
            if not m.any():
                continue
            c = nf[m]
            j = int(np.argmin(np.abs(c - f)))
            if abs(c[j] - f) <= window_s * fps:
                offs.append((c[j] - f) / fps)
        offs = np.array(offs)
        bnd[typ] = dict(n_ref=len(tr_r[typ]), n_new=len(tr_n[typ]), n_matched=int(len(offs)),
                        median_s=float(np.median(offs)) if len(offs) else float("nan"),
                        q25_s=float(np.percentile(offs, 25)) if len(offs) else float("nan"),
                        q75_s=float(np.percentile(offs, 75)) if len(offs) else float("nan"),
                        p90_abs_s=float(np.percentile(np.abs(offs), 90)) if len(offs) else float("nan"),
                        frac_within_1s=float((np.abs(offs) <= 1).mean()) if len(offs) else float("nan"),
                        offsets=[round(float(x), 2) for x in offs])
    sr, sn = C.spells(ref), C.spells(new)
    dr = np.array([(e - s) / fps for s, e, _ in sr]) if sr else np.zeros(1)
    dn = np.array([(e - s) / fps for s, e, _ in sn]) if sn else np.zeros(1)
    return dict(
        frames_both_labelled=int(both.sum()),
        frac_ref_labelled_new_unknown=float(((ref != C.UNK) & (new == C.UNK)).sum() / max((ref != C.UNK).sum(), 1)),
        agree_3class=float(po), kappa_3class=float((po - pe) / (1 - pe)),
        agree_alive_both=float((ref[alive] == new[alive]).mean()) if alive.any() else float("nan"),
        frac_alive_both=float(alive.sum() / max(both.sum(), 1)),
        D_frac_ref=float(rD.sum() / max(both.sum(), 1)), D_frac_new=float(nD.sum() / max(both.sum(), 1)),
        D_recall=float((rD & nD).sum() / max(rD.sum(), 1)), D_precision=float((rD & nD).sum() / max(nD.sum(), 1)),
        D_iou=float((rD & nD).sum() / max((rD | nD).sum(), 1)),
        confusion_ref_rows_new_cols=k.astype(int).tolist(),
        spells_ref=len(sr), spells_new=len(sn), mean_spell_ref_s=float(dr.mean()), mean_spell_new_s=float(dn.mean()),
        frac_spells_lt1s_new=float((dn < 1).mean()), boundaries=bnd)
