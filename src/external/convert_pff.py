from __future__ import annotations

import bz2
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import common as C

RAW = C.EXT_DIR / "pff" / "raw"
KICKOFFS = {"FIRSTKICKOFF": 1, "SECONDKICKOFF": 2, "THIRDKICKOFF": 3, "FOURTHKICKOFF": 4}
CONTROL_PE = {"PA", "CR", "SH", "BC", "IT"}
CONFIRM_S = 5.0
MAX_GAP_S = 0.1
DT = 1.0 / C.FPS


def load_events(gid: str):
    rows = json.loads((RAW / "Event Data" / f"{gid}.json").read_text())
    groups: dict[int, list] = {}
    for r in rows:
        groups.setdefault(r["gameEventId"], []).append(r)
    evs = []
    for gid_, rs in groups.items():
        ge = rs[0]["gameEvents"]
        typ = ge["gameEventType"]
        pes = [(r["possessionEvents"] or {}) for r in rs]
        kinds = {p.get("possessionEventType") for p in pes} - {None}
        sp = ge.get("setpieceType")
        team = None if ge.get("homeTeam") is None else (C.H if ge["homeTeam"] else C.A)
        if typ in KICKOFFS:
            cls = "control"
        elif typ == "OTB":
            if team is None:
                cls = "neutral"
            elif sp not in (None, "O"):
                cls = "control"
            elif kinds & CONTROL_PE:
                cls = "control"
            elif "TC" in kinds and any(p.get("possessionEventType") == "TC" and (
                    p.get("touchOutcomeType") == "R" or p.get("touchType") == "T") for p in pes):
                cls = "control"
            elif kinds and kinds <= {"CH"}:
                cls = "cond"
            elif kinds:
                cls = "unctrl"
            else:
                cls = "neutral"
        else:
            cls = typ.lower()
        seq = rs[0].get("sequence")
        evs.append(dict(id=gid_, period=int(ge["period"]), t=float(rs[0]["startTime"]), type=typ,
                        cls=cls, team=team, setpiece=sp, out=ge.get("outType"), kinds=kinds,
                        seq=seq, video_missing=bool(ge.get("videoMissing"))))
    evs.sort(key=lambda e: (e["period"], e["t"]))
    side = {}
    for r in rows:
        ge = r["gameEvents"]
        b = (r.get("ball") or [None])[0]
        if (ge["gameEventType"] != "OTB" or ge.get("homeTeam") is None or not b
                or b.get("visibility") != "VISIBLE" or b.get("x") is None):
            continue
        own, opp = ((r.get("homePlayers"), r.get("awayPlayers")) if ge["homeTeam"]
                    else (r.get("awayPlayers"), r.get("homePlayers")))
        if not own or not opp:
            continue
        dmin = [min(np.hypot(q["x"] - b["x"], q["y"] - b["y"]) for q in L) for L in (own, opp)]
        side.setdefault(int(ge["period"]), []).append(dmin[0] < dmin[1])
    side = {p: (float(np.mean(v)), len(v)) for p, v in side.items()}
    meta = rows[0].get("stadiumMetadata") or {}
    teams = {}
    for r in rows:
        ge = r["gameEvents"]
        if ge.get("teamName") and ge.get("homeTeam") is not None:
            teams[bool(ge["homeTeam"])] = ge["teamName"]
    return evs, meta, teams, side


def period_windows(evs):
    out = {}
    for e in evs:
        if e["type"] in KICKOFFS:
            out.setdefault(KICKOFFS[e["type"]], [e["t"], None])
    for e in evs:
        if e["type"] == "END" and e["period"] in out and out[e["period"]][1] is None:
            out[e["period"]][1] = e["t"]
    return {p: tuple(v) for p, v in out.items() if v[1] is not None}


def switch_events(evs, rule="ctrl"):
    sw = []
    act = [e for e in evs if e["cls"] in ("control", "cond", "unctrl")]
    for i, e in enumerate(act):
        if rule == "touch" or e["cls"] == "control":
            sw.append((e["period"], e["t"], e["team"]))
        elif e["cls"] == "cond":
            n = act[i + 1] if i + 1 < len(act) else None
            if (n is not None and n["period"] == e["period"] and n["cls"] == "control"
                    and n["team"] == e["team"] and n["t"] - e["t"] <= CONFIRM_S):
                sw.append((e["period"], e["t"], e["team"]))
    return sw


def dead_intervals(evs, windows):
    out = []
    for i, e in enumerate(evs):
        if e["type"] != "OUT" or e["period"] not in windows:
            continue
        t1 = windows[e["period"]][1]
        for n in evs[i + 1:]:
            if n["period"] != e["period"]:
                break
            if n["cls"] in ("control", "cond", "unctrl") and n["t"] >= e["t"]:
                t1 = n["t"]
                break
        out.append((e["period"], e["t"], t1))
    return out


def label_from(times, half, sw, dead):
    lab = np.full(len(times), C.UNK, np.int8)
    for p in np.unique(half):
        if not p:
            continue
        m = np.flatnonzero(half == p)
        s = sorted((t, team) for pp, t, team in sw if pp == p)
        st = np.array([t for t, _ in s])
        tm = np.array([team for _, team in s], np.int8)
        k = np.searchsorted(st, times[m] + 1e-6, side="right") - 1
        lp = np.where(k >= 0, tm[np.maximum(k, 0)], C.UNK).astype(np.int8)
        for pp, t0, t1 in dead:
            if pp == p:
                lp[(times[m] >= t0 - 1e-6) & (times[m] < t1 - 1e-6)] = C.D
        lab[m] = lp
    return lab


def seq_switches(evs):
    own: dict = {}
    for e in evs:
        if e["cls"] in ("control", "cond", "unctrl") and e["seq"] is not None:
            own.setdefault((e["period"], e["seq"]), Counter())[e["team"]] += 1
    sw = []
    for e in evs:
        if e["cls"] in ("control", "cond", "unctrl") and e["seq"] is not None:
            sw.append((e["period"], e["t"], own[(e["period"], e["seq"])].most_common(1)[0][0]))
    return sw


def load_tracking(gid: str, swap_periods=()):
    per, ts, ball = [], [], []
    pl_rows = []
    keys: dict = {}
    seen = set()
    with bz2.open(RAW / "Tracking Data" / f"{gid}.jsonl.bz2", "rt") as f:
        for line in f:
            r = json.loads(line)
            fn = r["frameNum"]
            if fn in seen:
                continue
            seen.add(fn)
            per.append(r["period"] or 0)
            ts.append(r["videoTimeMs"] / 1000.0)
            b = r.get("balls") or []
            b = b[0] if b else None
            ball.append((b["x"], b["y"], b.get("z")) if b and b.get("x") is not None else (np.nan,) * 3)
            row = []
            sides = (("homePlayers", 1), ("awayPlayers", 0)) if r["period"] in swap_periods else \
                (("homePlayers", 0), ("awayPlayers", 1))
            for side, k in sides:
                done = set()
                for q in r.get(side) or []:
                    j = str(q.get("jerseyNum"))
                    if j in done or q.get("x") is None:
                        continue
                    done.add(j)
                    s = keys.setdefault((k, j), len(keys))
                    row.append((s, q["x"], q["y"]))
            pl_rows.append(row)
    n = len(ts)
    order = sorted(keys, key=lambda kj: (kj[0], int(kj[1]) if kj[1].isdigit() else 999, kj[1]))
    remap = {keys[kj]: i for i, kj in enumerate(order)}
    xy = np.full((n, len(order), 2), np.nan)
    for i, row in enumerate(pl_rows):
        for s, x, y in row:
            xy[i, remap[s]] = (x, y)
    team = np.array([k for k, _ in order], np.int8)
    return (np.asarray(per, np.int8), np.asarray(ts), np.asarray(ball, float), xy, team,
            [j for _, j in order])


def resample(ts_src, vals, t_out):
    out = np.full((len(t_out),) + vals.shape[1:], np.nan)
    if len(ts_src) < 2:
        return out
    i = np.clip(np.searchsorted(ts_src, t_out, side="right") - 1, 0, len(ts_src) - 2)
    t0, t1 = ts_src[i], ts_src[i + 1]
    ok = (t_out >= t0 - 0.02) & (t_out <= t1 + 0.02) & (t1 - t0 <= MAX_GAP_S) & (t1 > t0)
    w = np.clip((t_out - t0) / np.where(t1 > t0, t1 - t0, 1), 0, 1)
    w = w.reshape((-1,) + (1,) * (vals.ndim - 1))
    v = vals[i] * (1 - w) + vals[i + 1] * w
    out[ok] = v[ok]
    return out


def spell_stats(lab):
    sp = C.spells(lab)
    d = np.array([(e - s) / C.FPS for s, e, _ in sp]) if sp else np.zeros(1)
    return dict(n_spells=len(sp), mean_spell_s=float(d.mean()), median_spell_s=float(np.median(d)),
                frac_spells_lt1s=float((d < 1).mean()), frac_spells_lt2s=float((d < 2).mean()))


def convert(gid: str, verbose=True):
    gid = str(gid)
    evs, smeta, teamnames, side = load_events(gid)
    windows = period_windows(evs)
    swap = sorted(p for p, (f, n) in side.items() if n >= 30 and f < 0.3)
    per_src, ts_src, ball_src, xy_src, team, jerseys = load_tracking(gid, set(swap))
    untracked = sorted(p for p in windows if not (per_src == p).any())
    windows = {p: w for p, w in windows.items() if p not in untracked}
    L = float(smeta.get("pitchLength") or 105.0)
    W = float(smeta.get("pitchWidth") or 68.0)

    times, half = [], []
    ball_l, xy_l = [], []
    for p in sorted(windows):
        t0, t1 = windows[p]
        t = t0 + np.arange(int(np.floor((t1 - t0) / DT)) + 1) * DT
        m = per_src == p
        o = np.argsort(ts_src[m], kind="stable")
        times.append(t)
        half.append(np.full(len(t), p, np.int8))
        ball_l.append(resample(ts_src[m][o], ball_src[m][o], t))
        xy_l.append(resample(ts_src[m][o], xy_src[m][o], t))
    times = np.concatenate(times)
    half = np.concatenate(half)
    ball = np.concatenate(ball_l)
    xy = np.concatenate(xy_l)
    xy[..., 0] += L / 2
    xy[..., 1] += W / 2
    ball[:, 0] += L / 2
    ball[:, 1] += W / 2

    dead = dead_intervals(evs, windows)
    label = label_from(times, half, switch_events(evs, "ctrl"), dead)
    lab_touch = label_from(times, half, switch_events(evs, "touch"), dead)
    lab_seq = label_from(times, half, seq_switches(evs), dead)

    players = np.concatenate([xy, C.speed_from_positions(xy, half)[..., None]], -1)
    ballc = np.concatenate([ball[:, :2], C.speed_from_positions(ball[:, :2], half, smooth=3)[..., None]], -1)

    n = len(label)
    alive = np.isin(label, [C.H, C.A])
    d = np.linalg.norm(xy - ball[:, None, :2], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    near = np.argmin(d, axis=1)
    close = d[np.arange(n), near] < 1.0
    sel = alive & close
    agree = float((team[near[sel]] == label[sel]).mean()) if sel.any() else float("nan")
    sel_t = np.isin(lab_touch, [C.H, C.A]) & close
    agree_touch = float((team[near[sel_t]] == lab_touch[sel_t]).mean()) if sel_t.any() else float("nan")
    ball_ok = ~np.isnan(ball[:, 0])
    track_ok = ~np.isnan(xy[..., 0]).all(1)
    vis = Counter()
    out_d = []
    for e in evs:
        if e["type"] == "OUT" and e["out"] == "T" and e["period"] in windows:
            k = np.searchsorted(times, e["t"])
            if k < n and half[k] == e["period"]:
                w = ball[max(k - 12, 0): k + 13, :2]
                w = w[~np.isnan(w[:, 0])]
                if len(w):
                    dist_in = np.minimum.reduce([w[:, 0], L - w[:, 0], w[:, 1], W - w[:, 1]])
                    out_d.append(float(dist_in.min()))
    out_d = np.array(out_d) if out_d else np.full(1, np.nan)

    title = f"{teamnames.get(True, '?')} vs {teamnames.get(False, '?')}"
    mp = RAW / "Metadata" / f"{gid}.json"
    if mp.exists():
        try:
            md = json.loads(mp.read_text())
            md = md[0] if isinstance(md, list) else md
            title += f" ({str(md.get('date', ''))[:10]})"
        except (ValueError, KeyError, IndexError):
            pass

    name = f"pff_{gid}"
    s = C.save(name, ball=ballc, players=players, team=team, label=label, half=half, pitch=(L, W))
    s.update(
        match=gid, title=title, periods={int(p): [round(a, 2), round(b, 2)] for p, (a, b) in windows.items()},
        n_game_events=len(evs), n_out=sum(e["type"] == "OUT" for e in evs),
        n_dead_stretches=len(dead), event_classes=dict(Counter(e["cls"] for e in evs)),
        src_frames=int(len(ts_src)), src_fps=float(1 / np.median(np.diff(np.sort(ts_src)))),
        frac_frames_with_tracking=float(track_ok.mean()),
        ball_present_alive=float(ball_ok[alive].mean()) if alive.any() else float("nan"),
        ball_present_dead=float(ball_ok[label == C.D].mean()) if (label == C.D).any() else float("nan"),
        nearest_player_team_agrees_with_label=agree, nearest_close_frac_alive=float(sel.sum() / max(alive.sum(), 1)),
        nearest_player_agrees_touch_label=agree_touch,
        agree_with_touch_label_alive=float((lab_touch[alive] == label[alive]).mean()),
        agree_with_pff_sequence_alive=float((lab_seq[alive] == label[alive]).mean()),
        touch=spell_stats(lab_touch), pff_sequence=spell_stats(lab_seq), **spell_stats(label),
        out_T_ball_dist_inside_line_median=float(np.nanmedian(out_d)),
        out_T_ball_within_1m_of_line=float(np.nanmean(out_d < 1.0)), out_T_n=int(len(out_d)),
        jerseys=jerseys, side_check={int(p): [round(f, 3), n] for p, (f, n) in side.items()},
        swapped_periods=swap, dropped_untracked_periods=untracked,
    )
    if verbose:
        print(C.fmt_stats(name, s), f"| <1s {s['frac_spells_lt1s']:.3f} | near-player agree {agree:.3f} "
              f"(touch-label {agree_touch:.3f}) | agree touch {s['agree_with_touch_label_alive']:.3f} "
              f"seq {s['agree_with_pff_sequence_alive']:.3f} | ball alive/dead {s['ball_present_alive']:.2f}/"
              f"{s['ball_present_dead']:.2f} | swapped {swap} | {title}", flush=True)
    return s


def _run(gid):
    try:
        return gid, convert(gid)
    except Exception as e:  # noqa: BLE001 - report and continue with the other matches
        import traceback
        traceback.print_exc()
        return gid, {"error": repr(e)}


if __name__ == "__main__":
    args = sys.argv[1:]
    jobs = 1
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        del args[i: i + 2]
    ids = args or sorted((p.name.split(".")[0] for p in (RAW / "Tracking Data").glob("*.jsonl.bz2")
                          if (RAW / "Event Data" / f"{p.name.split('.')[0]}.json").exists()), key=int)
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            res = dict(pool.imap_unordered(_run, ids))
    else:
        res = dict(_run(g) for g in ids)
    statp = C.COMMON_DIR / "stats_pff.json"
    old = json.loads(statp.read_text()) if statp.exists() else {}
    old.update(res)
    statp.write_text(json.dumps(old, indent=1))
