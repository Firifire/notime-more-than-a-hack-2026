import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, half_video_blocks  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]
HALVES = half_video_blocks(GAME)
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def _track_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_atd(raw_path, n_slots):
    with open(raw_path) as f:
        text = f.read()
    text = text.replace(":", ",").replace(";", ",")
    arr = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                      na_values=["NaN"], engine="c").to_numpy()
    frame_id = arr[:, 0].astype(np.int64)
    players = arr[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = arr[:, 1 + 3 * n_slots:1 + 3 * n_slots + 3]
    return frame_id, players, ball


def load_tracking(source="tactical", refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "atd_{}.npz".format(source))
    if os.path.exists(cache) and not refresh:
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    team = _track_teams(meta)
    frame_id, players, ball = _parse_atd(raw, len(team))
    n = int(frame_id.max())
    scale = np.array([PITCH[0], PITCH[1], 1.0])
    P = np.full((n, len(team), 3), np.nan, np.float32)
    B = np.full((n, 3), np.nan, np.float32)
    P[frame_id - 1] = players * scale
    B[frame_id - 1] = ball * scale
    out = {"ball": B, "players": P, "team": team}
    np.savez(cache, **out)
    return out


def load_tagging():
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import load_smart_tagging as lst
    rows = []
    for e in lst.load():
        r = {"code": e["code"], "start": e["start_frame"], "end": e["end_frame"]}
        r.update(e["tags"])
        rows.append(r)
    return pd.DataFrame(rows)


CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
LETTER = np.array(["H", "A", "D", "X"])


def load_labels(path):
    df = pd.read_csv(path, dtype={"label": str})
    lab = df["label"].str.strip().str.upper().map(CODE).to_numpy()
    return lab.astype(np.int8)


def load_gt():
    return load_labels(GT_PUBLIC)


def graded_mask():
    m = np.zeros(N_FRAMES, bool)
    for _, a, b in HALVES:
        m[a:b + 1] = True
    return m


def half_of(frames):
    h = np.zeros(len(frames), np.int8)
    for half, a, b in HALVES:
        h[(frames >= a) & (frames <= b)] = half
    return h


def write_submission(path, codes):
    codes = np.asarray(codes).copy()
    codes[~graded_mask()] = CODE["X"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    letters = LETTER[codes]
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.write("".join("{},{}\n".format(i, l) for i, l in enumerate(letters)))
    return path


if __name__ == "__main__":
    import time
    t = time.time()
    for src in ("tactical", "broadcast"):
        d = load_tracking(src, refresh=True)
        print(src, {k: v.shape for k, v in d.items()}, "%.1fs" % (time.time() - t))
        print("  ball present %.3f" % np.mean(~np.isnan(d["ball"][:, 0])))
