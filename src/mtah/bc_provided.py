import os

import numpy as np

from mtah.data import CACHE, HALVES, N_FRAMES

GK_SLOTS = (25, 50)
OUT = os.path.join(CACHE, "bc_cues_provided.npz")


def _footprint_area(H, valid, W=1920, Hh=1080, L=105.3, Wp=68.0):
    area = np.full(len(H), np.nan)
    corners = np.array([[0, 0, 1], [0, Hh, 1], [W, Hh, 1], [W, 0, 1]], float).T
    idx = np.flatnonzero(valid)
    Hi = np.linalg.inv(H[idx])
    p = Hi @ corners
    w = p[:, 2, :]
    behind = w <= 0
    x = np.clip(p[:, 0, :] / np.where(behind, np.nan, w) * L, -L, 2 * L)
    y = np.clip(p[:, 1, :] / np.where(behind, np.nan, w) * Wp, -Wp, 2 * Wp)
    a = 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, 1))
    area[idx] = a
    return area


def build(refresh=False):
    if os.path.exists(OUT) and not refresh:
        z = np.load(OUT)
        return {k: z[k] for k in z.files}
    a = np.load(os.path.join(CACHE, "broadcast_align.npz"))
    tb = np.load(os.path.join(CACHE, "atd_broadcast.npz"))
    hb = np.load(os.path.join(CACHE, "vision", "homography_broadcast.npz"))
    P, B = tb["players"], tb["ball"]
    nb = len(B)
    vis = ~np.isnan(P[:, :, 0])
    vis[:, list(GK_SLOTS)] = False
    nv_b = vis.sum(1).astype(np.float32)
    bv_b = (~np.isnan(B[:, 0])).astype(np.float32)
    area_b = _footprint_area(hb["H"], hb["valid"])
    cents = sorted(int(c) for c in a["centres"])
    out = {k: np.full(N_FRAMES, np.nan, np.float32) for k in ("n_vis", "ball_vis", "has", "cam_area", "raw_nvis")}
    out["bframe"] = np.full(N_FRAMES, -1, np.int64)
    for (h, s, e), c in zip(HALVES, cents):
        t = np.arange(max(s - 3000, 0), min(e + 3000, N_FRAMES - 1) + 1)
        b = t - c
        okb = (b >= 0) & (b < nb)
        t, b = t[okb], b[okb]
        live = (a["best_cost"][b] < 1.0) & (np.abs(a["best_t"][b] - (b + c)) <= 3)
        out["has"][t] = live
        out["n_vis"][t] = np.where(live, nv_b[b], 0.0)
        out["ball_vis"][t] = np.where(live, bv_b[b], 0.0)
        out["cam_area"][t] = area_b[b]
        out["raw_nvis"][t] = nv_b[b]
        out["bframe"][t] = b
    np.savez_compressed(OUT, **out)
    return out


def describe(c):
    ins = np.zeros(N_FRAMES, bool)
    for _, s, e in HALVES:
        ins[s:e + 1] = True
    r = {}
    for nm, m in (("in_halves", ins),):
        has = c["has"][m]
        away = (has < 0.5) | (c["n_vis"][m] < 4)
        rr = {"frames": int(m.sum()), "live_share": float(np.nanmean(has)), "away_share": float(away.mean()),
              "ball_vis_share": float(np.nanmean(c["ball_vis"][m])),
              "n_vis_live_mean": float(np.nanmean(c["n_vis"][m][has > 0.5])),
              "replay_like_share": float(np.mean((c["raw_nvis"][m] >= 4) & (has < 0.5))),
              "cam_area_valid_share": float(np.mean(np.isfinite(c["cam_area"][m]))),
              "cam_area_live_median": float(np.nanmedian(c["cam_area"][m][has > 0.5])),
              "cam_area_notlive_median": float(np.nanmedian(c["cam_area"][m][has < 0.5]))}
        d = np.diff(np.r_[0, away.astype(np.int8), 0])
        ln = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) / 25.0
        rr["away_runs"] = int(len(ln))
        rr["away_runs_ge2s"] = int((ln >= 2).sum())
        rr["away_run_len_s_quantiles_10_50_90"] = [float(np.percentile(ln, q)) for q in (10, 50, 90)] if len(ln) else []
        rr["away_runs_ge2s_per_90min"] = float((ln >= 2).sum() / (m.sum() / 25 / 60) * 90)
        r[nm] = rr
    return r


if __name__ == "__main__":
    import json
    c = build(refresh=True)
    print(json.dumps(describe(c), indent=1))
