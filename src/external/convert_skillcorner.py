from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C

RAW = C.EXT_DIR / "skillcorner" / "data" / "matches"
SRC_FPS = 10.0


def load(mid: str):
    d = RAW / mid
    meta = json.loads((d / f"{mid}_match.json").read_text())
    home, away = meta["home_team"]["id"], meta["away_team"]["id"]
    players = [p for p in meta["players"] if p["team_id"] in (home, away)]
    players.sort(key=lambda p: (p["team_id"] != home, p["trackable_object"]))
    pid_slot = {p["id"]: i for i, p in enumerate(players)}
    team = np.array([0 if p["team_id"] == home else 1 for p in players], np.int8)

    periods = [(p["period"], p["start_frame"], p["end_frame"]) for p in meta["match_periods"]]
    fmax = max(e for _, _, e in periods)
    xy = np.full((fmax + 1, len(players), 2), np.nan)
    ball = np.full((fmax + 1, 2), np.nan)
    with open(d / f"{mid}_tracking_extrapolated.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fr = r["frame"]
            if fr > fmax:
                continue
            b = r["ball_data"]
            if b and b.get("x") is not None:
                ball[fr] = (b["x"], b["y"])
            for p in r["player_data"]:
                s = pid_slot.get(p["player_id"])
                if s is not None and p.get("x") is not None:
                    xy[fr, s] = (p["x"], p["y"])

    src_lab = np.full(fmax + 1, C.UNK, np.int8)
    for _, s, e in periods:
        src_lab[s: e + 1] = C.D
    ph = pd.read_csv(d / f"{mid}_phases_of_play.csv")
    for r in ph.itertuples():
        src_lab[r.frame_start: r.frame_end + 1] = C.H if r.team_in_possession_id == home else C.A
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    title = f"{meta['home_team']['short_name']} vs {meta['away_team']['short_name']} ({meta['date_time'][:10]})"
    return periods, xy, ball, team, src_lab, (L, W), title


def resample(periods, xy, ball, src_lab):
    out_xy, out_ball, out_lab, out_half = [], [], [], []
    for pid, s, e in periods:
        t = np.arange(0, (e - s) * C.FPS / SRC_FPS + 1e-9) * SRC_FPS / C.FPS + s
        i0 = np.floor(t).astype(int)
        i1 = np.minimum(i0 + 1, e)
        w = (t - i0)
        out_xy.append(xy[i0] * (1 - w)[:, None, None] + xy[i1] * w[:, None, None])
        out_ball.append(ball[i0] * (1 - w)[:, None] + ball[i1] * w[:, None])
        out_lab.append(src_lab[np.rint(t).astype(int)])
        out_half.append(np.full(len(t), pid, np.int8))
    return (np.concatenate(out_xy), np.concatenate(out_ball), np.concatenate(out_lab),
            np.concatenate(out_half))


def convert(mid: str):
    periods, xy, ball, team, src_lab, (L, W), title = load(mid)
    pxy, bxy, label, half = resample(periods, xy, ball, src_lab)
    pxy = pxy + np.array([L / 2, W / 2])
    bxy = bxy + np.array([L / 2, W / 2])
    players = np.concatenate([pxy, C.speed_from_positions(pxy, half)[..., None]], -1)
    ballc = np.concatenate([bxy, C.speed_from_positions(bxy, half, smooth=3)[..., None]], -1)
    name = f"skillcorner_{mid}"
    s = C.save(name, ball=ballc, players=players, team=team, label=label, half=half, pitch=(L, W))
    s["title"] = title
    print(C.fmt_stats(name, s), "|", title, flush=True)
    return s


if __name__ == "__main__":
    ids = sys.argv[1:] or sorted(p.name for p in RAW.iterdir() if p.is_dir())
    allstats = {mid: convert(mid) for mid in ids}
    statp = C.COMMON_DIR / "stats_skillcorner.json"
    old = json.loads(statp.read_text()) if statp.exists() else {}
    old.update(allstats)
    statp.write_text(json.dumps(old, indent=1))
