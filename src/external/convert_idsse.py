from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from lxml import etree

import common as C

RAW = C.EXT_DIR / "idsse"
SECTIONS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def _parser_kwargs():
    return dict(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=True)


def read_meta(path: Path):
    root = etree.parse(str(path), etree.XMLParser(**_parser_kwargs())).getroot()
    gen = root.find(".//General")
    env = root.find(".//Environment")
    teams = {}
    for t in root.iter("Team"):
        teams[t.get("TeamId")] = t.get("Role")
        for p in t.iter("Player"):
            teams[p.get("PersonId")] = t.get("TeamId")
    return {
        "home": gen.get("HomeTeamId"),
        "away": gen.get("GuestTeamId"),
        "title": gen.get("MatchTitle"),
        "competition": gen.get("CompetitionName"),
        "result": gen.get("Result"),
        "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))),
        "person_team": teams,
    }


def read_positions(path: Path):
    out = {}
    cur = None
    buf = None
    ctx = etree.iterparse(str(path), events=("start", "end"), tag=("FrameSet", "Frame"), **_parser_kwargs())
    for ev, el in ctx:
        if el.tag == "FrameSet":
            if ev == "start":
                cur = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
                buf = defaultdict(list)
            else:
                arr = {k: np.asarray(v) for k, v in buf.items()}
                out[cur] = arr
                el.clear()
                while el.getprevious() is not None:
                    del el.getparent()[0]
            continue
        if ev != "end":
            continue
        a = el.attrib
        buf["N"].append(int(a["N"]))
        buf["X"].append(float(a["X"]))
        buf["Y"].append(float(a["Y"]))
        buf["S"].append(float(a.get("S", "nan")))
        if cur[1] == "BALL":
            buf["Z"].append(float(a.get("Z", "nan")))
            buf["BS"].append(int(a.get("BallStatus", "-1")))
            buf["BP"].append(int(a.get("BallPossession", "0")))
        el.clear()
    return out


def convert(mid: str, verbose=True):
    meta_p = next(RAW.glob(f"DFL_02_01_matchinformation_*{mid}.xml"))
    pos_p = next(RAW.glob(f"DFL_04_03_positions_raw_observed_*{mid}.xml"))
    meta = read_meta(meta_p)
    L, W = meta["pitch"]
    fs = read_positions(pos_p)

    sec_spans = []
    for (sec, tid, pid), arr in fs.items():
        if tid == "BALL":
            sec_spans.append((SECTIONS[sec], sec, int(arr["N"].min()), int(arr["N"].max())))
    sec_spans.sort()
    offsets = {}
    n = 0
    for hid, sec, n0, n1 in sec_spans:
        offsets[sec] = (n - n0, n0, n1, hid)
        n += n1 - n0 + 1

    persons = sorted({(tid, pid) for (sec, tid, pid) in fs if tid not in ("BALL", "referee")},
                     key=lambda t: (t[0] != meta["home"], t[1]))
    slot = {p: i for i, p in enumerate(persons)}
    team = np.array([0 if tid == meta["home"] else 1 for tid, _ in persons], np.int8)
    assert set(tid for tid, _ in persons) <= {meta["home"], meta["away"]}, set(tid for tid, _ in persons)

    ball = np.full((n, 3), np.nan, np.float32)
    players = np.full((n, len(persons), 3), np.nan, np.float32)
    label = np.full(n, C.UNK, np.int8)
    half = np.zeros(n, np.int8)
    prov_speed_ball = np.full(n, np.nan, np.float32)
    for sec, (off, n0, n1, hid) in offsets.items():
        half[off + n0: off + n1 + 1] = hid

    for (sec, tid, pid), arr in fs.items():
        if tid == "referee":
            continue
        off = offsets[sec][0]
        idx = arr["N"] + off
        xy = np.stack([arr["X"] + L / 2, arr["Y"] + W / 2], -1)
        spd = arr["S"] / 3.6
        if tid == "BALL":
            ball[idx, :2] = xy
            prov_speed_ball[idx] = spd
            bs, bp = arr["BS"], arr["BP"]
            lab = np.full(len(idx), C.UNK, np.int8)
            lab[(bs == 1) & (bp == 1)] = C.H
            lab[(bs == 1) & (bp == 2)] = C.A
            lab[bs == 0] = C.D
            label[idx] = lab
        else:
            s = slot[(tid, pid)]
            players[idx, s, :2] = xy
            players[idx, s, 2] = spd

    comp = C.speed_from_positions(ball[:, :2], half)
    ball[:, 2] = np.where(np.isnan(prov_speed_ball), comp, prov_speed_ball)

    pcomp = C.speed_from_positions(players[..., :2], half)
    ok = (~np.isnan(pcomp)) & (~np.isnan(players[..., 2])) & (pcomp > 1)
    ratio = float(np.median(players[..., 2][ok] / pcomp[ok]))
    d = np.linalg.norm(players[..., :2] - ball[:, None, :2], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    near = np.argmin(d, axis=1)
    close = d[np.arange(n), near] < 1.0
    alive = np.isin(label, [C.H, C.A]) & close
    agree = float((team[near[alive]] == label[alive]).mean())

    name = f"idsse_{mid}"
    s = C.save(name, ball=ball, players=players, team=team, label=label, half=half, pitch=(L, W))
    s.update(match=mid, title=meta["title"], competition=meta["competition"], result=meta["result"],
             speed_ratio_provider_over_computed=ratio, nearest_player_team_agrees_with_label=agree)
    if verbose:
        print(C.fmt_stats(name, s), f"| speed ratio {ratio:.3f} | nearest-player agreement {agree:.3f}",
              f"| {meta['title']}", flush=True)
    return s


if __name__ == "__main__":
    import json
    ids = sys.argv[1:] or MATCHES
    allstats = {}
    for mid in ids:
        allstats[mid] = convert(mid)
    statp = C.COMMON_DIR / "stats_idsse.json"
    old = json.loads(statp.read_text()) if statp.exists() else {}
    old.update(allstats)
    statp.write_text(json.dumps(old, indent=1))
