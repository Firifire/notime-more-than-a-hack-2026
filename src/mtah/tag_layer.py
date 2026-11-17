import functools
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]
TIMING = os.path.join(ROOT, "notes", "trackG_metrica_timing.json")

OUT_MARGIN = 0.5
MIN_OUT_S = 0.4
CTRL_R = 1.0
STILL_V = 0.5
CORNER_R = 3.0
GOAL_AREA = (5.5, 18.32)
MIN_SPELL_S = 0.5
CAL_Q = (5, 95)

TYPE_KEY = {"throw-in": "throw_in", "corner kick": "corner", "goal kick": "goal_kick", "free kick": "free_kick",
            "kick off (after goal)": "kick_off"}


@functools.lru_cache(maxsize=None)
def pitch():
    from games import asset_paths
    fs = ET.parse(asset_paths()["atd_meta"]).getroot().find(".//FieldSize")
    return float(fs.findtext("Width")), float(fs.findtext("Height"))


def load_tags(path=None):
    if path is None:
        import sys
        sys.path.insert(0, ROOT)
        from games import asset_paths
        path = asset_paths()["smart_tagging"]
    out = []
    for ins in ET.parse(path).iter("instance"):
        if (ins.findtext("code") or "").strip().lower() != "set pieces":
            continue
        tags = {lb.findtext("group"): lb.findtext("text") for lb in ins.findall("label")}
        k = TYPE_KEY.get((tags.get("Type") or "").strip().lower())
        if k is None:
            continue
        out.append((k, int(round(float(ins.findtext("start")) * FPS)), int(round(float(ins.findtext("end")) * FPS))))
    return sorted(out, key=lambda r: r[1])


def timing():
    return json.load(open(TIMING))


def kinematics(xy, P):
    sp = np.full(len(xy), np.nan)
    sp[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=1) * FPS / 2
    d = np.sqrt(((P[:, :, :2].astype(np.float32) - xy[:, None, :].astype(np.float32)) ** 2).sum(-1))
    dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), 1).astype(np.float64)
    dmin[~np.isfinite(dmin)] = np.nan
    return sp, dmin


def ball_and_players(video_ball=None):
    ap = os.path.join(ROOT, "cache", "atd_tactical.npz")
    if not os.path.exists(ap):
        from mtah.data import load_tracking
        load_tracking("tactical")
    z = np.load(ap)
    xy = z["ball"][:, :2].astype(np.float64)
    vid = os.path.join(ROOT, "cache", "vision", "ball_video_refD3.npz")
    if video_ball and not os.path.exists(vid):
        raise SystemExit("tag_layer: video_ball requested but %s is missing (run.py step 'vision')" % vid)
    if os.path.exists(vid) and video_ball is not False:
        v = np.load(vid)["xy"].astype(np.float64)[: len(xy)]
        miss = np.isnan(xy[:, 0])
        xy[miss] = v[miss]
    sp, dmin = kinematics(xy, z["players"])
    return xy, sp, dmin


def out_of_pitch(xy):
    x, y = xy[:, 0], xy[:, 1]
    PITCH = pitch()
    with np.errstate(invalid="ignore"):
        return (x < -OUT_MARGIN) | (x > PITCH[0] + OUT_MARGIN) | (y < -OUT_MARGIN) | (y > PITCH[1] + OUT_MARGIN)


def at_spot(kind, xy):
    x, y = xy[:, 0], xy[:, 1]
    PITCH = pitch()
    with np.errstate(invalid="ignore"):
        if kind == "corner":
            cx = np.minimum(np.abs(x), np.abs(x - PITCH[0]))
            cy = np.minimum(np.abs(y), np.abs(y - PITCH[1]))
            return np.hypot(cx, cy) <= CORNER_R
        if kind == "goal_kick":
            gx = np.minimum(np.abs(x), np.abs(x - PITCH[0]))
            return (gx <= GOAL_AREA[0] + OUT_MARGIN) & (np.abs(y - PITCH[1] / 2) <= GOAL_AREA[1] / 2 + OUT_MARGIN)
        if kind == "kick_off":
            return np.hypot(x - PITCH[0] / 2, y - PITCH[1] / 2) <= CTRL_R + OUT_MARGIN
        return np.isfinite(x)


def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def calibrate(lab, tags, half, T):
    d = (lab == 2) & (half > 0)
    R = runs(d)
    offs = []
    for k, s, e in tags:
        if k == "kick_off":
            continue
        mn = T[k]["stop"]["p10"] * FPS
        c = [(b - s) / FPS for a, b in R if b - a >= mn and abs(b - s) <= e - s]
        if c:
            offs.append(min(c, key=abs))
    return np.asarray(offs)


def windows(tags, offs):
    lo, hi = np.percentile(offs, CAL_Q) if len(offs) else (0.0, 0.0)
    out = []
    for k, s, e in tags:
        if k == "kick_off":
            out.append((s, e))
        else:
            out.append((s + int(round(lo * FPS)), s + int(round(hi * FPS))))
    return out


def reception(f, kind, sp, dmin, T, n):
    fl = T[kind]["flight"]
    a, b, med = f + int(fl["p10"] * FPS), min(n - 1, f + int(np.ceil(fl["max"] * FPS))), f + int(round(fl["median"] * FPS))
    seg = dmin[f:b + 1]
    if np.isnan(seg).mean() > 0.5:
        return med, "median"
    left = np.flatnonzero(seg > CTRL_R)
    if not len(left):
        return med, "median"
    j0 = f + left[0]
    for g in range(max(a, j0), b + 1):
        if np.isfinite(dmin[g]) and dmin[g] <= CTRL_R:
            return g, "ball"
    return med, "median"


def find_stoppage(kind, win, xy, sp, dmin, T, n):
    lo, hi = win
    a = max(0, lo - int(T[kind]["stop"]["p90"] * FPS))
    b = min(n, hi + 1)
    out = out_of_pitch(xy[a:b])
    R = [(a + s, a + e) for s, e in runs(out) if e - s >= MIN_OUT_S * FPS]
    if kind == "throw_in":
        R = [(s, e) for s, e in R if e <= hi]
        if not R or R[-1][1] < lo:
            return None
        rs = R[-1][1]
        s0 = R[-1][0]
        for s, e in reversed(R[:-1]):
            if s0 - e > T[kind]["stop"]["median"] * FPS:
                break
            s0 = s
        return s0, rs, "ball_out"
    still = (sp[a:b] < STILL_V) & at_spot(kind, xy[a:b])
    S = [(a + s, a + e) for s, e in runs(still) if e - s >= T[kind]["stop"]["p10"] * FPS]
    S = [(s, e) for s, e in S if lo <= e <= hi]
    if not S:
        return None
    s, rs = S[-1]
    prev = [r for r in R if r[1] <= s]
    start = prev[-1][0] if prev else s
    return start, rs, "still" + ("+out" if prev else "")


def prior_shift(tags, W, offs, T, n, pi):
    sh = np.zeros(n)
    if not len(offs):
        return sh
    for (k, s, e), (lo, hi) in zip(tags, W):
        stops = np.sort(np.asarray([T[k]["stop"]["p10"], T[k]["stop"]["median"], T[k]["stop"]["p90"]]))
        o_list = [0.5 * (lo + hi) - s] if k == "kick_off" else list(offs * FPS)
        flight = T[k]["flight"]["median"] * FPS
        span_lo = int(min(o_list) - T[k]["stop"]["max"] * FPS) + s
        span_hi = int(max(o_list) + flight) + s + 1
        t = np.arange(max(0, span_lo), min(n, span_hi))
        P = np.zeros(len(t))
        for o in o_list:
            r = s + o
            u = (r - t) / FPS
            xs = [0.0, T[k]["stop"]["p10"], T[k]["stop"]["median"], T[k]["stop"]["p90"], T[k]["stop"]["max"]]
            ys = [1.0, 0.9, 0.5, 0.1, 0.0]
            surv = np.interp(u, xs, ys, left=1.0, right=0.0)
            P += np.where(u > 0, surv, np.where(-u * FPS < flight, 1.0, 0.0))
        P /= len(o_list)
        eps = 1.0 / (len(o_list) + 1)
        P = np.clip(P, eps, 1 - eps)
        lg = np.log(P / (1 - P)) - np.log(pi / (1 - pi))
        sh[t] = np.maximum(sh[t], np.maximum(lg, 0.0))
        del stops
    return sh


def spec_of(cfg):
    t = cfg.get("tag_layer")
    if not t:
        return None
    if isinstance(t, str):
        return {"variant": t, "video_ball": False}
    return {"variant": t["variant"], "video_ball": bool(t.get("video_ball", False))}


class patched:

    def __init__(self, spec):
        self.spec, self.calls = spec, 0

    def __enter__(self):
        if not self.spec:
            return self
        from mtah import pipe_ens2
        self.orig = pipe_ens2.decode_heads
        spec, orig, me = self.spec, self.orig, self

        def hook(q, half, rest):
            if "lgb_team" in q:
                raise RuntimeError("tag_layer: this pipeline passes member heads to pipe_ens2.decode_heads; the tag "
                                   "layer needs the combined heads")
            me.calls += 1
            pipe_ens2.decode_heads = orig
            try:
                return apply(q, half, rest, spec["variant"], video_ball=spec["video_ball"])
            finally:
                pipe_ens2.decode_heads = hook
        pipe_ens2.decode_heads = hook
        return self

    def __exit__(self, *exc):
        if self.spec:
            from mtah import pipe_ens2
            pipe_ens2.decode_heads = self.orig
            if exc[0] is None and self.calls != 1:
                raise RuntimeError("tag_layer: expected exactly one final Viterbi call, saw %d" % self.calls)
        return False


def apply(q, half, rest, variant, tags=None, return_info=False, video_ball=None):
    from mtah import pipe_ens2
    T = timing()
    tags = load_tags() if tags is None else tags
    n = len(q["dead"])
    base = np.asarray(pipe_ens2.decode_heads(q, half, rest))
    offs = calibrate(base, tags, half, T)
    W = windows(tags, offs)
    info = {"offsets_s": np.round(np.percentile(offs, [5, 10, 50, 90, 95]), 2).tolist() if len(offs) else [],
            "n_offsets": int(len(offs)), "events": []}
    lab = base
    if "A" in variant:
        ok = half > 0
        pi = float(np.clip(np.mean(q["dead"][ok]), 1e-3, 1 - 1e-3))
        sh = prior_shift(tags, W, offs, T, n, pi)
        qq = dict(q)
        x = np.clip(q["dead"], 1e-6, 1 - 1e-6)
        qq["dead"] = 1 / (1 + np.exp(-(np.log(x / (1 - x)) + sh)))
        lab = np.asarray(pipe_ens2.decode_heads(qq, half, rest))
        info["pi"] = pi
    if "B" in variant or "C" in variant:
        xy, sp, dmin = ball_and_players(video_ball)
        lab = lab.copy()
        for (k, s, e), (lo, hi) in zip(tags, W):
            if half[min(max(lo, 0), n - 1)] == 0 and half[min(hi, n - 1)] == 0:
                continue
            R = [(a, b) for a, b in runs((lab == 2) & (half > 0)) if lo <= b <= hi
                 and b - a >= T[k]["stop"]["p10"] * FPS * (0 if k == "kick_off" else 1)]
            ev = {"type": k, "tag": int(s), "win": [int(lo), int(hi)]}
            if R:
                a, b = max(R, key=lambda r: r[1] - r[0])
                ev["base_run"] = [int(a), int(b)]
                if "C" in variant:
                    g, how = reception(b, k, sp, dmin, T, n)
                    if g > b:
                        nxt = np.flatnonzero(lab[b:g] == 2)
                        g = b + int(nxt[0]) if len(nxt) else g
                        lab[b:g] = 2
                        ev["extend"] = [int(b), int(g), how]
            else:
                st = find_stoppage(k, (lo, hi), xy, sp, dmin, T, n)
                if st is not None:
                    a, r, how = st
                    g, how2 = reception(r, k, sp, dmin, T, n)
                    m = (half[a:g] > 0)
                    lab[a:g][m] = 2
                    ev["insert"] = [int(a), int(g), how, how2]
                    if "2" in variant and g < n and lab[g] in (0, 1) and half[g] > 0:
                        j = g
                        while j < n and lab[j] == lab[g] and half[j] == half[g]:
                            j += 1
                        if j < n and j - g < MIN_SPELL_S * FPS and half[j] == half[g] and lab[j] in (0, 1, 2):
                            ev["merge"] = [int(g), int(j), int(lab[g]), int(lab[j])]
                            lab[g:j] = lab[j]
            info["events"].append(ev)
    return (lab, info) if return_info else lab
