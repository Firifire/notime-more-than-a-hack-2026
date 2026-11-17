import argparse
import glob
import hashlib
import json
import os

import numpy as np

from mtah import degrade_profiles as dp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
STV2_DIR = os.path.join(_ROOT, "cache", "vision", "stv2")
COMMON = os.path.join(_ROOT, "data_external", "common")
SEQ_EXT = os.path.join(_ROOT, "cache", "degprof", "video_ext_sequences.npz")
STATS_EXT = os.path.join(_ROOT, "notes", "avo", "deg_profiles_stats_video_ext.json")
SHORT_S = 1.0
FPS = 25


def _halves():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(STV2_DIR, "*", "ball.npz")))


def _half_seq(hn):
    mid = hn.split("_")[0]
    z = np.load(os.path.join(STV2_DIR, hn, "ball.npz"))
    m = np.load(os.path.join(COMMON, "stv2_%s.npz" % mid), allow_pickle=False)
    idx_all = z["npz_index"]
    keep = idx_all >= 0
    idx = idx_all[keep]
    assert np.all(np.diff(idx) == 1), "video->npz index must be contiguous"
    state = z["state"][keep]
    V = z["xy"][keep].astype(np.float64)
    A = m["ball_anchor_only"][idx, :2].astype(np.float64)
    Bi = m["ball"][idx, :2].astype(np.float64)
    has = ~np.isnan(A[:, 0])
    ref = A.copy()
    s, e = dp._runs(~has)
    for a, b in zip(s, e):
        if a > 0 and b < len(has) and (b - a) <= SHORT_S * FPS:
            ref[a:b] = Bi[a:b]
    present = state > 0
    mode = np.where(state == 1, 2, np.where(state == 2, 6, 0)).astype(np.int8)
    off_anchor = V - A
    off_anchor[~present] = np.nan
    off_ref = V - ref
    off_ref[~present] = np.nan
    return dict(hn=hn, present=present, mode=mode, off_anchor=off_anchor, off_ref=off_ref,
                rank=dp.speed_rank(m["players"][idx]), label=m["label"][idx], V=V, A=A)


def build(seed=0, halves=None, write_seq=True, stats_path=STATS_EXT):
    hs = [_half_seq(h) for h in (halves or _halves())]
    if not hs:
        raise SystemExit("no stv2 ball tracks in " + STV2_DIR)
    rng = np.random.default_rng(seed)
    pres = np.concatenate([h["present"] for h in hs])
    mode = np.concatenate([h["mode"] for h in hs])
    off_a = np.concatenate([h["off_anchor"] for h in hs])
    off_r = np.concatenate([h["off_ref"] for h in hs])
    rank = np.concatenate([h["rank"] for h in hs])
    half = np.concatenate([np.full(len(h["present"]), k, np.int16) for k, h in enumerate(hs)])
    halves, o = [], 0
    for h in hs:
        halves.append((o, o + len(h["present"])))
        o += len(h["present"])
    off_f = np.empty_like(off_r)
    for a, b in halves:
        off_f[a:b] = dp._lowpass_runs(dp._fill_offsets(off_r[a:b], mode[a:b], rng), pres[a:b])
    if write_seq:
        os.makedirs(os.path.dirname(SEQ_EXT), exist_ok=True)
        np.savez_compressed(SEQ_EXT, rank=rank, half=half, video_present=pres, video_mode=mode,
                            video_off=off_f.astype(np.float32), halves=np.array([h["hn"] for h in hs]),
                            doc=np.array("degrade_video_ext.build: label-free corruption sequence of OUR video ball "
                                         "measured on SoccerTrack v2 video vs event anchors"))
    st = stats(hs, pres, mode, off_a, off_r, rank, halves)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    json.dump(st, open(stats_path, "w"), indent=1)
    return st


def _acf_pairs(off, lags=(1, 5, 25, 125)):
    out = {}
    for L in lags:
        a, b = off[:-L], off[L:]
        ok = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
        if ok.sum() > 20:
            aa, bb = a[ok].ravel(), b[ok].ravel()
            out["lag%d" % L] = round(float(np.corrcoef(aa, bb)[0, 1]), 3)
        else:
            out["lag%d" % L] = None
        out["lag%d_pairs" % L] = int(ok.sum())
    return out


def stats(hs, pres, mode, off_a, off_r, rank, halves):
    st = {"what": "unlabelled error statistics of OUR video ball on SoccerTrack v2 panoramic video (external)",
          "halves": [h["hn"] for h in hs], "frames": int(len(pres)),
          "reference": "event anchors (ball_anchor_only, +-2 frames around annotated events; ball at the actor's "
                       "feet, alive play only); 'ref_short' adds the provider interpolation between anchors <= %.1f s "
                       "apart" % SHORT_S,
          "schema": "as notes/avo/deg_profiles_stats.json 'fused' block (key 'video_ext')"}
    d = dp._gap_stats(pres, halves)
    e = np.linalg.norm(off_a, axis=1)
    er = np.linalg.norm(off_r, axis=1)
    d["offset_measured_frac_of_present"] = round(float((~np.isnan(e[pres])).mean()), 4)
    d["offset_measured_frac_of_present_ref_short"] = round(float((~np.isnan(er[pres])).mean()), 4)
    d["offset_m"] = dp._q(e)
    d["offset_m_ref_short"] = dp._q(er)
    me = ~np.isnan(e)
    d["offset_gt5m"] = round(float(np.mean(e[me] > 5)), 4)
    d["offset_gt10m"] = round(float(np.mean(e[me] > 10)), 4)
    mr = ~np.isnan(er)
    d["offset_gt5m_ref_short"] = round(float(np.mean(er[mr] > 5)), 4)
    d["offset_gt10m_ref_short"] = round(float(np.mean(er[mr] > 10)), 4)
    sep = np.full((125, 2), np.nan)
    cat_a = np.concatenate(sum([[off_a[a:b], sep] for a, b in halves], []))
    cat_r = np.concatenate(sum([[off_r[a:b], sep] for a, b in halves], []))
    d["acf_offset"] = _acf_pairs(cat_a)
    d["acf_offset_ref_short"] = _acf_pairs(cat_r)
    d["by_mode"] = {}
    for md in np.unique(mode):
        sel = mode == md
        d["by_mode"][dp.MODES[int(md)]] = {"frac": round(float(sel.mean()), 4),
                                           "offset_m": dp._q(e[sel]) if md > 0 else None,
                                           "offset_m_ref_short": dp._q(er[sel]) if md > 0 else None}
    d["by_speed_quintile"] = []
    V = np.concatenate([h["V"] for h in hs])
    step = np.r_[np.nan, np.linalg.norm(np.diff(V, axis=0), axis=1)]
    for a, _ in halves:
        step[a] = np.nan
    for qi in range(5):
        sel = (rank >= qi / 5) & (rank < (qi + 1) / 5 + (qi == 4))
        d["by_speed_quintile"].append({
            "quintile": qi, "present": round(float(pres[sel].mean()), 3),
            "offset_p50": round(float(np.nanpercentile(e[sel], 50)), 2) if np.any(~np.isnan(e[sel])) else None,
            "offset_p90": round(float(np.nanpercentile(e[sel], 90)), 2) if np.any(~np.isnan(e[sel])) else None,
            "offset_p50_ref_short": round(float(np.nanpercentile(er[sel], 50)), 2) if np.any(~np.isnan(er[sel])) else None,
            "offset_p90_ref_short": round(float(np.nanpercentile(er[sel], 90)), 2) if np.any(~np.isnan(er[sel])) else None,
            "measured_frames": int(np.sum(~np.isnan(e[sel]))),
            "step_gt1m_per_frame": round(float(np.nanmean(step[sel] > 1.0)), 4),
        })
    st["video_ext"] = d
    st["video_interpolated_frac"] = round(float((mode == 6).mean()), 4)
    lab = np.concatenate([h["label"] for h in hs])
    st["diag_coverage_by_label"] = {"alive": round(float(pres[(lab >= 0) & (lab < 2)].mean()), 4),
                                    "dead": round(float(pres[lab == 2].mean()), 4)}
    st["per_half"] = {h["hn"]: {"coverage": round(float(h["present"].mean()), 4),
                                "detected": round(float((h["mode"] == 2).mean()), 4),
                                "offset_anchor_p50": round(float(np.nanmedian(np.linalg.norm(h["off_anchor"], axis=1))), 3)}
                      for h in hs}
    return st


_EXT = None


def _seq_ext():
    global _EXT
    if _EXT is None:
        z = np.load(SEQ_EXT)
        S = {k: z[k] for k in z.files if k not in ("doc", "halves")}
        h = S["half"]
        S["half_end"] = np.zeros(len(h), np.int64)
        for v in np.unique(h):
            ii = np.flatnonzero(h == v)
            S["half_end"][ii] = ii[-1] + 1
        _EXT = S
    return _EXT


def overlay_ext(rank_ext, rng):
    saved = dp._SEQ
    dp._SEQ = _seq_ext()
    try:
        return dp.overlay(rank_ext, "video", rng)
    finally:
        dp._SEQ = saved


def degrade_video_ext(m, rng, player_drop=0.05, noise=0.15):
    from mtah.train_ext import unfreeze_ball
    m = dict(m)
    P = m["players"].copy()
    B = unfreeze_ball(m["ball"], m["players"], rng)
    B = dp._fill_missing_ball(B, m["players"], rng)
    pres, off, _ = overlay_ext(dp.speed_rank(m["players"]), rng)
    B[:, :2] = B[:, :2] + off
    B[~pres] = np.nan
    N, S = len(B), P.shape[1]
    for s in range(S):
        n_gaps = rng.poisson(player_drop * N / 75)
        for _ in range(n_gaps):
            L = int(rng.exponential(75)) + 1
            a = int(rng.integers(0, max(1, N - L)))
            P[a:a + L, s] = np.nan
    if noise > 0:
        from scipy.ndimage import gaussian_filter1d
        n = gaussian_filter1d(rng.normal(0, noise * 3, P[:, :, :2].shape), 5, axis=0)
        P[:, :, :2] += n
    m["ball"], m["players"] = B, P
    return m


def profile_hash_ext(stripped_profiles_source):
    import inspect
    import sys
    h = hashlib.sha1(inspect.getsource(sys.modules[__name__]).encode())
    h.update(stripped_profiles_source.encode())
    h.update(b"video_ext")
    with open(SEQ_EXT, "rb") as f:
        h.update(hashlib.sha1(f.read()).digest())
    return h.hexdigest()[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stats-only", default=None, help="write only the stats json to this path (all tracked "
                    "halves; the sequence file used by the profile is left untouched)")
    ap.add_argument("--check", default=None, help="external match name: print overlay summary")
    a = ap.parse_args()
    if a.stats_only:
        build(write_seq=False, stats_path=a.stats_only)
    if a.build:
        st = build()
        print(json.dumps(st, indent=1)[:8000])
    if a.check:
        from mtah.train_ext import load_ext
        m = [x for x in load_ext([a.check.split("_")[0]]) if x["name"] == a.check][0]
        d = degrade_video_ext(m, np.random.default_rng(0))
        g = m["half"] > 0
        b = d["ball"][:, 0]
        e = np.linalg.norm(d["ball"][:, :2] - m["ball"][:, :2], axis=1)
        print("video_ext coverage %.3f" % np.mean(~np.isnan(b[g])), "err p50/p90 %.2f/%.2f" % tuple(
            np.nanpercentile(e[g], [50, 90])), "D-frame coverage %.3f / alive %.3f" % (
            np.mean(~np.isnan(b[g & (m["label"] == 2)])), np.mean(~np.isnan(b[g & (m["label"] < 2)]))),
            getattr(dp.overlay, "last_stats", None))


if __name__ == "__main__":
    main()
