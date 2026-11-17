from __future__ import annotations

import argparse
import json
import warnings
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C
import sb_labels as S

RAW = C.EXT_DIR / "raw" / "soccertrack_v2" / "raw"
PERIODS = {"FIRST_HALF": 1, "SECOND_HALF": 2}
FLIP_BALL_Y = {"132831", "132877"}
L, W = 105.0, 68.0
MS_PER_FRAME = 40.0
CONFIRM_S = 5.0
BALL_OUT_SPEED = 15.0
TRAVEL_CAP_S = 1.5
ANCHOR_HALF_W = 2

ADMIN = {"changeIn", "changeOut", "yellowCard", "redCard", "secondYellowCard"}
SETPIECE = {"throwIn", "cornerKick", "goalKickSucceeded", "goalKickFailed", "freeKick", "penaltyKick", "setPieceShot"}
CONTROL = SETPIECE | {"passSucceeded", "passFailed", "crossSucceeded", "crossFailed", "shot", "passReceived",
                      "crossReceived", "ballReceived", "possession", "possessionWon", "dribbleSucceeded",
                      "dribbleToSpace", "controlUnderPressure", "saveByCatching", "aerialClearanceSucceededByCatch"}
UNCTRL = {"clearance", "block", "blockCross", "blockShot", "deflection", "ballMissed", "saveByPunching",
          "aerialClearanceSucceeded", "aerialClearanceSucceededByPunch", "aerialClearanceFailed", "hit",
          "dribbleFailed"}
COND = {"intercept", "cutoff", "tackleSucceeded", "turnoverWon", "duelSucceeded", "groundDuelSucceeded",
        "aerialDuelSucceeded", "looseBallDuelSucceeded", "defensiveLineSupportSucceeded"}
WHISTLE = {"foulCommitted", "foulConceded", "handballFoul", "offside", "pause"}
STATUS = {"BALLOUT": 0, "HOME": 1, "AWAY": 2, "NEUTRAL": 3}

FR_RE = re.compile(rb'<frame matchTime="(-?[\d.]+)" frameNumber="(\d+)" eventPeriod="(\w+)" ballStatus="(\w+)">')
OB_RE = re.compile(rb'<(?:player|ball) playerId="(\w+)" loc="\[([^,\]]+), ([^\]]+)\]" speed="([^"]*)"')


def _f(b: bytes) -> float:
    try:
        return float(b)
    except ValueError:
        return np.nan


def load_meta(mid: str):
    meta = (RAW / mid / f"{mid}_tracker_box_metadata.xml").read_text(encoding="utf-8")
    teams = dict((s, t) for t, s in re.findall(r'<team id="(\d+)"[^>]*side="(\w+)"', meta))
    p_team = dict(re.findall(r'<player id="(\d+)"[^>]*teamId="(\d+)"', meta))
    p_team.update({k: v for k, v in re.findall(r'<player playerId="(\d+)" teamId="(\d+)"', meta) if k not in p_team})
    m = re.search(r'<pitch width="([\d.]+)" height="([\d.]+)"', meta)
    assert m and float(m.group(1)) == L and float(m.group(2)) == W, "pitch is not 105 x 68"
    return teams["home"], teams["away"], p_team


def load_tracking(mid: str):
    txt = (RAW / mid / f"{mid}_tracker_box_data.xml").read_bytes()
    frs = list(FR_RE.finditer(txt))
    per = np.array([PERIODS.get(m.group(3).decode(), 0) for m in frs], np.int8)
    keep = per > 0
    mt = np.array([float(m.group(1)) for m in frs])
    status = np.array([STATUS.get(m.group(4).decode(), -1) for m in frs], np.int8)
    starts = np.array([m.start() for m in frs])
    obs = list(OB_RE.finditer(txt))
    del txt
    fi = np.searchsorted(starts, np.array([m.start() for m in obs])) - 1
    pid = np.array([m.group(1).decode() for m in obs])
    lx = np.array([_f(m.group(2)) for m in obs])
    ly = np.array([_f(m.group(3)) for m in obs])
    sp = np.array([_f(m.group(4)) for m in obs])
    del obs
    new = np.full(len(frs), -1)
    new[keep] = np.arange(keep.sum())
    ok = keep[fi]
    fi, pid, lx, ly, sp = new[fi[ok]], pid[ok], lx[ok], ly[ok], sp[ok]
    half, mt, status = per[keep], mt[keep], status[keep]
    for h in (1, 2):
        d = np.diff(mt[half == h])
        assert len(d) and np.all(np.abs(d - MS_PER_FRAME) < 1e-6), f"{mid} half {h}: matchTime not 40 ms steps"
    assert np.all(np.diff(half) >= 0)
    n = len(half)
    isb = pid == "ball"
    ball = np.full((n, 2), np.nan)
    ball[fi[isb]] = np.c_[lx[isb] * L, (ly[isb] if mid in FLIP_BALL_Y else 1 - ly[isb]) * W]
    return dict(n=n, half=half, mt=mt, status=status, ball=ball, fi=fi[~isb], pid=pid[~isb], x=lx[~isb] * L,
                y=(1 - ly[~isb]) * W, sp=sp[~isb])


def build_players(tr, home, p_team):
    pid, fi = tr["pid"], tr["fi"]
    ids = sorted(set(pid))
    firstf = {i: int(fi[pid == i].min()) for i in ids}
    team_of = {i: 0 if p_team[i] == home else 1 for i in ids}
    ids.sort(key=lambda i: (team_of[i], firstf[i]))
    slot = {i: k for k, i in enumerate(ids)}
    sl = np.array([slot[i] for i in pid])
    xy = np.full((tr["n"], len(ids), 2), np.nan, np.float32)
    spd = np.full((tr["n"], len(ids)), np.nan, np.float32)
    xy[fi, sl] = np.c_[tr["x"], tr["y"]]
    spd[fi, sl] = tr["sp"]
    team = np.array([team_of[i] for i in ids], np.int8)
    return ids, slot, xy, spd, team


def load_events(mid: str, home: str):
    d = pd.read_csv(RAW / mid / f"{mid}_player_nodes.csv", dtype={"team_id": str})
    d = d[d.event_period.isin(PERIODS)].copy()
    d["half"] = d.event_period.map(PERIODS)
    d["order"] = np.arange(len(d))
    d = d.sort_values(["half", "event_time", "order"], kind="stable").reset_index(drop=True)
    toks = (d.event_types.fillna("") + " " + d.filtered_event_types.fillna("")).str.split()
    d["tok"] = [set(t) for t in toks]
    d["team"] = np.where(d.team_id == home, C.H, C.A)
    d["pid"] = [str(int(p)) if p == p else None for p in d.player_id]

    def kind(r):
        t = r.tok
        if r.pid is None:
            return "endnode"
        if t and t <= ADMIN | {"footLeft", "footRight"}:
            return "admin"
        if t & CONTROL:
            return "control"
        if t & UNCTRL:
            return "unctrl"
        if t & COND:
            return "cond"
        return "neutral"
    d["kind"] = [kind(r) for r in d.itertuples()]
    return d


def frame_of(t_ms, mt0):
    return int(np.rint((t_ms - mt0) / MS_PER_FRAME))


def travel_frames(ball_xy):
    if np.isnan(ball_xy[0]):
        return 0
    dist = max(0.0, min(ball_xy[0], L - ball_xy[0], ball_xy[1], W - ball_xy[1]))
    return int(np.rint(min(dist / BALL_OUT_SPEED, TRAVEL_CAP_S) * C.FPS))


def build_label(ev, half, mt, ball):
    n = len(half)
    lab = np.full(n, C.UNK, np.int8)
    anchors = []
    diag = dict(n_switch=0, n_cond=0, n_cond_confirmed=0, n_restarts=0, restart_by={}, stop_by={},
                dead_len_s=[], n_dead_overlap=0)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        i0, npf = idx[0], len(idx)
        mt0 = mt[i0]
        e = ev[ev.half == h].reset_index(drop=True)
        e["f"] = [frame_of(t, mt0) for t in e.event_time]
        anchors += [i0 + f for f in e.f[e.kind != "admin"] if 0 <= f < npf]
        act = e[e.kind.isin(["control", "cond", "unctrl"])].reset_index(drop=True)
        mark = np.full(npf, -1, np.int8)
        for i, r in act.iterrows():
            ok = r.kind == "control"
            if r.kind == "cond":
                diag["n_cond"] += 1
                nx = act.iloc[i + 1] if i + 1 < len(act) else None
                ok = (nx is not None and nx.kind == "control" and nx.team == r.team
                      and nx.event_time - r.event_time <= CONFIRM_S * 1000)
                diag["n_cond_confirmed"] += int(ok)
            if ok:
                diag["n_switch"] += 1
                mark[min(max(r.f, 0), npf - 1)] = r.team
        cur = -1
        lp = np.full(npf, C.UNK, np.int8)
        for j in range(npf):
            if mark[j] >= 0:
                cur = mark[j]
            if cur >= 0:
                lp[j] = cur
        rest = []
        after_goal = after_pause = False
        for i, r in e.iterrows():
            if r.kind == "admin":
                continue
            if r.tok & SETPIECE:
                rest.append((i, "setpiece"))
                after_goal = after_pause = False
            elif r.kind == "control" and (after_goal or after_pause):
                rest.append((i, "kickoff" if after_goal else "after_pause"))
                after_goal = after_pause = False
            if "goal" in r.tok:
                after_goal = True
            if "pause" in r.tok:
                after_pause = True
        nonadmin = e.index[e.kind != "admin"].to_numpy()

        def stop_frame(k):
            prev = nonadmin[nonadmin < k]
            if not len(prev):
                return None, "none"
            r = e.loc[prev[-1]]
            if "goalAgainst" in r.tok:
                return r.f, "goal"
            if r.tok & WHISTLE:
                return r.f, "whistle"
            if r.kind == "endnode":
                return r.f, "endnode"
            f = r.f
            b = ball[i0 + min(max(f, 0), npf - 1)]
            return f + travel_frames(b), ("goal_travel" if "goal" in r.tok else "travel")

        dead = np.zeros(npf, bool)
        for k, how in rest:
            f0, why = stop_frame(k)
            f1 = int(e.f[k])
            if f0 is None:
                continue
            f0 = min(f0, f1)
            diag["n_restarts"] += 1
            diag["restart_by"][how] = diag["restart_by"].get(how, 0) + 1
            diag["stop_by"][why] = diag["stop_by"].get(why, 0) + 1
            a, b = max(f0, 0), min(f1, npf)
            if b > a:
                diag["n_dead_overlap"] += int(dead[a:b].any())
                dead[a:b] = True
                diag["dead_len_s"].append((b - a) / C.FPS)
        f0, why = stop_frame(len(e))
        if f0 is not None and f0 < npf:
            dead[max(f0, 0):] = True
            diag["stop_by"]["end_of_half_" + why] = diag["stop_by"].get("end_of_half_" + why, 0) + 1
        lp[dead] = C.D
        lab[idx] = lp
    dl = np.array(diag.pop("dead_len_s"))
    diag["dead_len_s_quantiles"] = [float(np.percentile(dl, q)) for q in (5, 25, 50, 75, 95)] if len(dl) else []
    return lab, np.array(sorted(set(anchors))), diag


def provider_label(status, half):
    lab = np.full(len(status), C.UNK, np.int8)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        cur = -1
        for j in idx:
            s = status[j]
            if s in (1, 2):
                cur = C.H if s == 1 else C.A
            if s == 0:
                lab[j] = C.D
            elif s in (1, 2, 3) and cur >= 0:
                lab[j] = cur
    return lab


def convert(mid: str):
    warnings.filterwarnings("ignore", "Mean of empty slice")
    home, away, p_team = load_meta(mid)
    tr = load_tracking(mid)
    half, mt, n = tr["half"], tr["mt"], tr["n"]
    ids, slot, pxy, pspd, team = build_players(tr, home, p_team)
    ball = tr["ball"]
    ev = load_events(mid, home)
    label, anchors, diag = build_label(ev, half, mt, ball)
    pos_spd = C.speed_from_positions(pxy, half)
    spd = np.where(np.isnan(pspd) & ~np.isnan(pxy[..., 0]), pos_spd, pspd)
    players = np.concatenate([pxy, spd[..., None]], -1)
    ballc = np.concatenate([ball, C.speed_from_positions(ball, half, smooth=3)[..., None]], -1)
    near = np.zeros(n, bool)
    for o in range(-ANCHOR_HALF_W, ANCHOR_HALF_W + 1):
        j = anchors + o
        near[j[(j >= 0) & (j < n)]] = True
    ball_anchor = np.where(near[:, None], ballc, np.nan).astype(np.float32)
    has_prov = bool((tr["status"] >= 0).any())
    label_prov = provider_label(tr["status"], half) if has_prov else np.full(n, C.UNK, np.int8)
    name = f"stv2_{mid}"
    s = C.save(name, ball=ballc, players=players, team=team, label=label, half=half, pitch=(L, W))
    p = C.COMMON_DIR / f"{name}.npz"
    z = dict(np.load(p))
    z.update(ball_interpolated=np.bool_(True), ball_anchor_only=ball_anchor,
             provider_status=tr["status"].astype(np.int8), label_provider=label_prov)
    np.savez_compressed(p, **z)

    st = dict(s)
    st["label_diag"] = diag
    alive = np.isin(label, [C.H, C.A])
    vis = (~np.isnan(pxy[..., 0])).sum(1)
    st["players_per_frame_mean"] = float(vis.mean())
    st["players_per_frame_q"] = [float(np.percentile(vis, q)) for q in (1, 50, 99)]
    st["players_per_team_frame_median"] = [float(np.median((~np.isnan(pxy[:, team == t, 0])).sum(1))) for t in (0, 1)]
    inside = (pxy[..., 0] >= -2) & (pxy[..., 0] <= L + 2) & (pxy[..., 1] >= -2) & (pxy[..., 1] <= W + 2)
    st["players_inside_pitch_2m"] = float(inside[~np.isnan(pxy[..., 0])].mean())
    sides = {}
    for h in (1, 2):
        m = half == h
        dx = np.nanmean(pxy[m][:, team == 0, 0], 1) - np.nanmean(pxy[m][:, team == 1, 0], 1)
        sgn = np.sign(np.nanmedian(dx))
        sides[h] = dict(home_minus_away_mean_x_median=float(np.nanmedian(dx)), frac_frames_same_sign=float(np.mean(np.sign(dx) == sgn)))
        gk = []
        for t in (0, 1):
            mx = np.nanmean(pxy[m][:, team == t, 0], 0)
            cnt = (~np.isnan(pxy[m][:, team == t, 0])).sum(0)
            mx[cnt < 0.5 * m.sum()] = np.nan
            gk.append(float(mx[np.nanargmax(np.abs(mx - L / 2))]) if np.isfinite(mx).any() else float("nan"))
        sides[h]["gk_mean_x_home_away"] = gk
    st["sides"] = sides
    st["sides_swap_at_half_time"] = bool(np.sign(sides[1]["home_minus_away_mean_x_median"]) != np.sign(sides[2]["home_minus_away_mean_x_median"]))
    dd = []
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        e = ev[(ev.half == h) & ev.pid.isin(list(slot))]
        f = np.rint((e.event_time.to_numpy() - mt[idx[0]]) / MS_PER_FRAME).astype(int)
        ok = (f >= 0) & (f < len(idx))
        a = pxy[idx[0] + f[ok], [slot[q] for q in e.pid[ok]]]
        dd.append(np.linalg.norm(a - ball[idx[0] + f[ok]], axis=1))
    dd = np.concatenate(dd)
    st["actor_to_anchor_m_median"] = float(np.nanmedian(dd))
    st["actor_to_anchor_within_2m"] = float(np.nanmean(dd < 2))
    dist = np.linalg.norm(pxy - ball[:, None, :].astype(np.float32), axis=-1)
    dist = np.where(np.isnan(dist), np.inf, dist)
    k = np.argmin(dist, 1)
    close = dist[np.arange(n), k] < 1.0
    sel = alive & close
    st["nearest_player_team_agrees_with_label"] = float((team[k[sel]] == label[sel]).mean()) if sel.any() else None
    st["nearest_close_frac_alive"] = float(sel.sum() / max(alive.sum(), 1))
    if has_prov:
        selp = np.isin(label_prov, [C.H, C.A]) & close
        st["nearest_player_team_agrees_with_label_provider"] = float((team[k[selp]] == label_prov[selp]).mean())
    st["ball_anchor_only_present_frac"] = float((~np.isnan(ball_anchor[label != C.UNK, 0])).mean())
    st["ball_anchor_only_present_frac_alive_dead"] = [float((~np.isnan(ball_anchor[alive, 0])).mean()),
                                                      float((~np.isnan(ball_anchor[label == C.D, 0])).mean())]
    if has_prov:
        cmp = S.compare(label_prov, label, half)
        st["vs_provider"] = cmp
        st["provider_stats"] = C.stats(dict(label=label_prov, fps=C.FPS, ball=ballc, players=players))
    st["minutes_total"] = n / C.FPS / 60
    st["n_events"] = int(len(ev))
    msg = C.fmt_stats(name, s) + (f" | players/frame {st['players_per_frame_mean']:.2f}, inside {st['players_inside_pitch_2m']:.4f}, "
                                  f"sides swap {st['sides_swap_at_half_time']}, actor-anchor {st['actor_to_anchor_m_median']:.2f} m, "
                                  f"nearest-team agree {st['nearest_player_team_agrees_with_label']:.3f}")
    if has_prov:
        c = st["vs_provider"]
        msg += (f" | vs provider: kappa {c['kappa_3class']:.3f} alive-agree {c['agree_alive_both']:.3f} D IoU {c['D_iou']:.3f} "
                f"prov D {st['provider_stats']['frac_D']:.3f} prov spell {st['provider_stats']['mean_spell_s']:.1f}s "
                f"D_start med {c['boundaries']['D_start']['median_s']:+.2f}s D_end med {c['boundaries']['D_end']['median_s']:+.2f}s")
    print(msg, flush=True)
    return mid, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    ids = a.ids or sorted(p.name for p in RAW.iterdir() if p.is_dir())
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = dict(ex.map(convert, ids))
    statp = C.COMMON_DIR / "stats_stv2.json"
    old = json.loads(statp.read_text()) if statp.exists() else {}
    old.update(res)
    statp.write_text(json.dumps(old, indent=1, default=float))


if __name__ == "__main__":
    main()
