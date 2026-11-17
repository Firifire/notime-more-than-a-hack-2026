from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C

RAW = C.EXT_DIR / "metrica" / "repo" / "data"
L, W = 105.0, 68.0
POSS_TYPES = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT"}


def to_m(x, y):
    return np.asarray(x, np.float64) * L, (1.0 - np.asarray(y, np.float64)) * W


def read_csv_tracking(path: Path):
    df = pd.read_csv(path, skiprows=2)
    cols = list(df.columns)
    names = [c for c in cols if c.startswith("Player")]
    xy = []
    for nm in names:
        i = cols.index(nm)
        xy.append(df.iloc[:, [i, i + 1]].to_numpy(np.float64))
    bi = cols.index("Ball")
    ball = df.iloc[:, [bi, bi + 1]].to_numpy(np.float64)
    return df["Frame"].to_numpy(), df["Period"].to_numpy(), np.stack(xy, 1), ball


def load_csv_game(g: int):
    d = RAW / f"Sample_Game_{g}"
    fh, ph, xyh, bh = read_csv_tracking(d / f"Sample_Game_{g}_RawTrackingData_Home_Team.csv")
    fa, pa, xya, ba = read_csv_tracking(d / f"Sample_Game_{g}_RawTrackingData_Away_Team.csv")
    assert (fh == fa).all() and np.allclose(bh, ba, equal_nan=True)
    xy = np.concatenate([xyh, xya], 1)
    team = np.r_[np.zeros(xyh.shape[1]), np.ones(xya.shape[1])].astype(np.int8)
    ev = pd.read_csv(d / f"Sample_Game_{g}_RawEventsData.csv")
    events = pd.DataFrame({
        "team": (ev["Team"] == "Away").astype(int),
        "type": ev["Type"].fillna(""),
        "sub": ev["Subtype"].fillna("").astype(str),
        "start": ev["Start Frame"].astype(int),
        "end": ev["End Frame"].astype(int),
        "period": ev["Period"].astype(int),
    })
    return fh, ph, xy, bh, team, events


def load_epts_game():
    from kloppy import metrica

    d = RAW / "Sample_Game_3"
    ds = metrica.load_tracking_epts(meta_data=str(d / "Sample_Game_3_metadata.xml"),
                                    raw_data=str(d / "Sample_Game_3_tracking.txt"),
                                    coordinates="metrica")
    home, away = ds.metadata.teams
    plist = list(home.players) + list(away.players)
    team = np.array([0] * len(home.players) + [1] * len(away.players), np.int8)
    n = len(ds.frames)
    frame = np.zeros(n, int)
    period = np.zeros(n, int)
    xy = np.full((n, len(plist), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    pidx = {p: i for i, p in enumerate(plist)}
    for k, fr in enumerate(ds.frames):
        frame[k] = fr.frame_id
        period[k] = fr.period.id
        if fr.ball_coordinates is not None:
            ball[k] = (fr.ball_coordinates.x, fr.ball_coordinates.y)
        for p, pd_ in fr.players_data.items():
            if pd_.coordinates is not None:
                xy[k, pidx[p]] = (pd_.coordinates.x, pd_.coordinates.y)
    ev = json.load(open(d / "Sample_Game_3_events.json"))["data"]
    rows = []
    for e in ev:
        st = e["subtypes"]
        if isinstance(st, dict):
            sub = st["name"]
        elif st:
            sub = "-".join(s["name"] for s in st)
        else:
            sub = ""
        rows.append({"team": 0 if e["team"]["id"] == home.team_id else 1, "type": e["type"]["name"],
                     "sub": sub, "start": e["start"]["frame"], "end": e["end"]["frame"] or 0,
                     "period": e["period"]})
    return frame, period, xy, ball, team, pd.DataFrame(rows), (home.team_id, away.team_id)


def is_stoppage(t, s):
    if t == "BALL OUT":
        return "end"
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return "end"
    if t == "FAULT RECEIVED":
        return "start"
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return "start"
    if "OFFSIDE" in s or "HAND BALL" in s:
        return "end"
    if "END HALF" in s:
        return "start"
    return None


def is_restart(t, s):
    return t == "SET PIECE" or "GOAL KICK" in s


def derive_labels(frame, period, events: pd.DataFrame):
    n = len(frame)
    f2i = {f: i for i, f in enumerate(frame)}
    label = np.full(n, C.UNK, np.int8)
    ev = events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)

    for p in np.unique(period):
        if p == 0:
            continue
        idx = np.flatnonzero(period == p)
        f0, f1 = frame[idx[0]], frame[idx[-1]]
        evp = ev[(ev["period"] == p) & (ev["start"] >= f0) & (ev["start"] <= f1)]
        lab = np.full(f1 - f0 + 1, C.D, np.int8)
        poss = evp[evp["type"].isin(POSS_TYPES)]
        cur_team, cur_f = None, f0
        for r in poss.itertuples():
            if cur_team is not None:
                lab[cur_f - f0: r.start - f0] = cur_team
            cur_team, cur_f = r.team, r.start
        if cur_team is not None:
            lab[cur_f - f0:] = cur_team
        rows = list(evp.itertuples())
        for k, r in enumerate(rows):
            when = is_stoppage(r.type, r.sub)
            if when is None:
                continue
            fs = r.end if (when == "end" and r.end > 0) else r.start
            fe = f1 + 1
            for q in rows[k + 1:]:
                if q.start < fs:
                    continue
                if is_restart(q.type, q.sub):
                    fe = q.start
                    break
                if q.type in POSS_TYPES and q.type != "BALL OUT" and q.start > fs:
                    fe = q.start
                    break
            fs, fe = max(fs, f0), min(fe, f1 + 1)
            if fe > fs:
                lab[fs - f0: fe - f0] = C.D
        label[idx] = lab[frame[idx] - f0]
    return label


def convert(g: int):
    if g in (1, 2):
        frame, period, xy, ball, team, events = load_csv_game(g)
    else:
        frame, period, xy, ball, team, events, _ = load_epts_game()
    assert (np.diff(frame) == 1).all(), "tracking frames not contiguous"
    half = period.astype(np.int8)
    px, py = to_m(xy[..., 0], xy[..., 1])
    bx, by = to_m(ball[:, 0], ball[:, 1])
    pxy = np.stack([px, py], -1)
    bxy = np.stack([bx, by], -1)
    players = np.concatenate([pxy, C.speed_from_positions(pxy, half)[..., None]], -1)
    ballc = np.concatenate([bxy, C.speed_from_positions(bxy, half, smooth=3)[..., None]], -1)
    label = derive_labels(frame, period, events)
    name = f"metrica_game{g}"
    s = C.save(name, ball=ballc, players=players, team=team, label=label, half=half, pitch=(L, W))
    print(C.fmt_stats(name, s), flush=True)
    return s


if __name__ == "__main__":
    games = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    allstats = {f"game{g}": convert(g) for g in games}
    statp = C.COMMON_DIR / "stats_metrica.json"
    old = json.loads(statp.read_text()) if statp.exists() else {}
    old.update(allstats)
    statp.write_text(json.dumps(old, indent=1))
