from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C
from convert_idsse import MATCHES

PATHCRF_DIR = C.EXT_DIR / "pathcrf" / "repo" / "data" / "sportec_synced"
ELASTIC_DIR = C.EXT_DIR / "elastic" / "gt"

CONTROL = {"pass", "cross", "shot", "shot_freekick", "shot_penalty", "throw_in", "goalkick",
           "corner_short", "corner_crossed", "freekick_short", "freekick_crossed", "control",
           "ball_recovery", "keeper_pick_up", "keeper_claim"}
CONDITIONAL = {"interception", "tackle"}
UNCONTROLLED = {"clearance", "bad_touch", "shot_block", "keeper_save", "keeper_punch"}
NEUTRAL = {"dispossessed", "foul", "out", "goal"}
CONFIRM_S = 5.0
PRE_S, POST_S = 1.0, 3.0


def _frame_index(npz_half: np.ndarray, period_first_N: dict[int, int]):
    starts = {int(h): int(np.flatnonzero(npz_half == h)[0]) for h in np.unique(npz_half) if h}
    return lambda p, N: starts[p] + (N - period_first_N[p])


def load_events(source: str, mid: str) -> pd.DataFrame:
    if source == "pathcrf":
        ev = pd.read_parquet(PATHCRF_DIR / f"{mid}.parquet")
        ev = ev[ev["frame_id"].notna()]
        rows = ev[["period_id", "frame_id", "player_id", "spadl_type"]].values.tolist()
    else:
        ev = pd.read_parquet(ELASTIC_DIR / f"{mid}.parquet")
        rows = ev[["period_id", "frame_id", "player_id", "spadl_type"]].values.tolist()
        side = ev["player_id"].str.split("_").str[0]
        rside = ev["receiver_id"].fillna("").str.split("_").str[0]
        rec = ev[(side == rside) & ev["receive_frame_id"].notna()
                 & ev["spadl_type"].isin(CONTROL - {"control"})]
        rows += [[p, f, r, "control"] for p, f, r in
                 rec[["period_id", "receive_frame_id", "receiver_id"]].values.tolist()]
    out = []
    for order, (p, f, pid, t) in enumerate(rows):
        s = str(pid).split("_")[0]
        if s not in ("home", "away"):
            continue
        kind = ("control" if t in CONTROL else "cond" if t in CONDITIONAL else
                "unctrl" if t in UNCONTROLLED else "neutral" if t in NEUTRAL else None)
        if kind is None:
            raise ValueError(f"unmapped event type {t!r}")
        if kind == "neutral":
            continue
        out.append((int(p), int(f), order, 0 if s == "home" else 1, kind, t))
    df = pd.DataFrame(out, columns=["period", "N", "order", "team", "kind", "type"])
    return df.sort_values(["period", "N", "order"]).reset_index(drop=True)


def switch_events(ev: pd.DataFrame, fps: float):
    sw = []
    kinds, teams, Ns, periods = ev["kind"].values, ev["team"].values, ev["N"].values, ev["period"].values
    for i in range(len(ev)):
        if kinds[i] == "control":
            sw.append((periods[i], Ns[i], teams[i]))
        elif kinds[i] == "cond":
            j = i + 1
            if (j < len(ev) and periods[j] == periods[i] and kinds[j] == "control"
                    and teams[j] == teams[i] and Ns[j] - Ns[i] <= CONFIRM_S * fps):
                sw.append((periods[i], Ns[i], teams[i]))
    return sw


def build_label(z, period_first_N, sw):
    orig = z["label"]
    half = z["half"]
    fps = float(z["fps"])
    n = len(orig)
    to_idx = _frame_index(half, period_first_N)
    team_at = np.full(n, -1, np.int8)
    sw_idx = []
    for p, N, t in sw:
        i = to_idx(p, N)
        if 0 <= i < n:
            sw_idx.append((i, t))
    sw_idx.sort()
    mark = np.full(n, -1, np.int8)
    for i, t in sw_idx:
        mark[i] = t
    for h in np.unique(half):
        if not h:
            continue
        idx = np.flatnonzero(half == h)
        cur = -1
        for i in idx:
            if mark[i] >= 0:
                cur = mark[i]
            team_at[i] = cur
    sw_arr = np.array([i for i, _ in sw_idx]) if sw_idx else np.zeros(0, int)
    alive = np.isin(orig, [C.H, C.A])
    edges = np.flatnonzero(np.diff(np.r_[0, alive.astype(np.int8), 0]))
    pre, post = int(PRE_S * fps), int(POST_S * fps)
    for s, e in zip(edges[::2], edges[1::2]):
        k = np.searchsorted(sw_arr, s - pre)
        if k < len(sw_arr) and sw_arr[k] < min(s + post, e) and half[sw_arr[k]] == half[s]:
            first = sw_arr[k]
            if first > s:
                team_at[s:first] = mark[first]
    lab = orig.copy()
    ok = alive & (team_at >= 0)
    lab[ok] = team_at[ok]
    n_fallback = int((alive & (team_at < 0)).sum())
    return lab, n_fallback


def spell_stats(lab, fps):
    sp = C.spells(lab)
    d = np.array([(e - s) / fps for s, e, _ in sp])
    return dict(n_spells=len(sp), mean_spell_s=float(d.mean()), median_spell_s=float(np.median(d)),
                frac_spells_lt1s=float((d < 1).mean()), frac_spells_lt2s=float((d < 2).mean()))


def period_first_frames(mid: str) -> dict[int, int]:
    from lxml import etree
    from convert_idsse import RAW, SECTIONS, _parser_kwargs
    pos_p = next(RAW.glob(f"DFL_04_03_positions_raw_observed_*{mid}.xml"))
    out = {}
    cur = None
    for ev, el in etree.iterparse(str(pos_p), events=("start", "end"), tag=("FrameSet", "Frame"),
                                  **_parser_kwargs()):
        if el.tag == "FrameSet" and ev == "start":
            cur = SECTIONS[el.get("GameSection")] if el.get("TeamId") == "BALL" else None
        elif el.tag == "Frame" and ev == "end" and cur is not None and cur not in out:
            out[cur] = int(el.get("N"))
        if ev == "end":
            el.clear()
    return out


def main(ids):
    allstats = {}
    for mid in ids:
        z = dict(np.load(C.COMMON_DIR / f"idsse_{mid}.npz"))
        fps = float(z["fps"])
        pf = period_first_frames(mid)
        ev = load_events("pathcrf", mid)
        lab, nfb = build_label(z, pf, switch_events(ev, fps))
        orig = z["label"]
        alive = np.isin(orig, [C.H, C.A])
        st = dict(match=mid, source="pathcrf sportec_synced", n_events=len(ev),
                  n_switch_events=len(switch_events(ev, fps)), alive_frames_without_event=nfb,
                  agree_with_dfl_alive=float((lab[alive] == orig[alive]).mean()),
                  dfl=spell_stats(orig, fps), **spell_stats(lab, fps))
        if (ELASTIC_DIR / f"{mid}.parquet").exists():
            ev2 = load_events("elastic", mid)
            lab2, _ = build_label(z, pf, switch_events(ev2, fps))
            st["elastic_gt"] = dict(agree_with_pathcrf_alive=float((lab2[alive] == lab[alive]).mean()),
                                    agree_with_dfl_alive=float((lab2[alive] == orig[alive]).mean()),
                                    **spell_stats(lab2, fps))
        z["label"] = lab
        C.save(f"idsse-ctrl_{mid}", ball=z["ball"], players=z["players"], team=z["team"], label=lab,
               half=z["half"], pitch=z["pitch"], fps=z["fps"])
        allstats[mid] = st
        print(json.dumps(st), flush=True)
    p = C.COMMON_DIR / "stats_idsse_ctrl.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(allstats)
    p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or MATCHES)
