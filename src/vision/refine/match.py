import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision import track as T  # noqa: E402
from vision.common import CACHE, N_FRAMES, FPS, PITCH, W, H, process_ranges, load_homography, px_to_metres  # noqa: E402
from vision.refine import sim  # noqa: E402
from vision.refine.sample import load_model, run_half  # noqa: E402


def camera_near_side(Hs):
    fr = np.concatenate([np.arange(s, e) for _, s, e in process_ranges()])
    _, yb = px_to_metres(np.linalg.inv(Hs[fr]), W / 2.0, H - 1.0)
    return bool(np.nanmedian(yb) > PITCH[1] / 2)


def in_bounds(x, y, near_is_max_y):
    ylo, yhi = ((-T.MARGIN_FAR_M, PITCH[1] + T.MARGIN_NEAR_M) if near_is_max_y else
                (-T.MARGIN_NEAR_M, PITCH[1] + T.MARGIN_FAR_M))
    return ((x > -T.MARGIN_GOAL_M) & (x < PITCH[0] + T.MARGIN_GOAL_M) & (y > ylo) & (y < yhi) & np.isfinite(x))


def match_inputs(cdir=None):
    from mtah.data import load_tracking
    Hs = load_homography()
    c, covered = T.load_cands(cdir or os.path.join(CACHE, "cands"))
    keep = c["conf"] >= T.CONF_MIN
    c = {k: v[keep] for k, v in c.items()}
    x, y = T.to_metres(c, Hs)
    inb = in_bounds(x, y, camera_near_side(Hs))
    cand = sim.pack(c["frame"][inb].astype(np.int64), x[inb], y[inb], c["conf"][inb], N_FRAMES)
    P = load_tracking()["players"][:, :, :2]
    P = np.concatenate([P, np.full((N_FRAMES - len(P), P.shape[1], 2), np.nan, np.float32)]) if len(P) < N_FRAMES \
        else P[:N_FRAMES]
    return cand, P, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cands", default=None, help="detection candidate folder (default cache/vision/cands)")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--b", type=float, default=0.1)
    ap.add_argument("--readout", action="store_true", help="refD2: jump-suppressed read-out (vision.refine.readout)")
    ap.add_argument("--commit", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--kin-lam", type=float, default=0.0, help="refD3: in-sampler kinematic prior (nats per metre)")
    ap.add_argument("--calib", default=None, help="refD4: LightGBM confidence model (vision.refine.calib)")
    ap.add_argument("--conf-thr", type=float, default=None, help="refD4: confidence threshold fixed on external data")
    ap.add_argument("--ent-thr", type=float, default=float("inf"),
                    help="one-shot entropy threshold for state/xy (fixed on EXTERNAL data; default: full coverage)")
    a = ap.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(a.model, "cuda")
    cand, P, covered = match_inputs(a.cands)
    xy = np.full((N_FRAMES, 2), np.nan, np.float32)
    xy_raw = np.full((N_FRAMES, 2), np.nan, np.float32)
    ent = np.full(N_FRAMES, np.inf, np.float32)
    pmax = np.zeros(N_FRAMES, np.float32)
    ent1 = np.full(N_FRAMES, np.inf, np.float32)
    conf = np.full(N_FRAMES, np.nan, np.float32)
    feats = None
    is_c = np.zeros(N_FRAMES, bool)
    for h, s, e in process_ranges():
        Ph = sim.player_feats(P[s:e], fps=FPS)
        r = run_half(model, cand[s:e], Ph, "cuda", N=a.steps, sweeps=a.sweeps, b=a.b, seed=h, commit=a.commit,
                     kin_lam=a.kin_lam)
        xy[s:e], ent[s:e], pmax[s:e], is_c[s:e] = r["xy"], r["ent"], r["pmax"], r["is_cand"]
        xy_raw[s:e] = r["xy"]
        if a.readout:
            from vision.refine.readout import final_xy
            xy[s:e], vt = final_xy(r, cand[s:e])
            is_c[s:e] = vt < 8
        r1 = run_half(model, cand[s:e], Ph, "cuda", N=1, sweeps=1, commit="argmax", seed=h)
        ent1[s:e] = r1["ent"]
        if a.calib:
            from vision.refine.calib import features_from
            Fm = features_from(r, r1, cand[s:e], Ph)
            if feats is None:
                feats = np.full((N_FRAMES, Fm.shape[1]), np.nan, np.float32)
            feats[s:e] = Fm
        print("half", h, "frames", e - s, "on-candidate share %.3f" % r["is_cand"].mean(), flush=True)
    ok = np.isfinite(xy[:, 0]) & (ent1 <= a.ent_thr)

    state = np.where(ok, np.where(is_c, 1, 2), 0).astype(np.int8)
    out_xy = np.where(ok[:, None], xy, np.nan)
    np.savez_compressed(a.out, xy=out_xy, xy_all=xy, xy_raw=xy_raw, readout=a.readout, kin_lam=a.kin_lam, conf=conf,
                        conf_thr=np.nan if a.conf_thr is None else a.conf_thr,
                        calib_F=feats if feats is not None else np.zeros((0, 0), np.float32), is_cand=is_c,
                        commit=a.commit, steps=a.steps, ent=ent, ent1=ent1, pmax=pmax, state=state, processed=covered,
                        xy_fused=out_xy, fused_source=np.where(ok, 7, 0).astype(np.int8),
                        conf_det=np.where(is_c & ok, pmax, 0).astype(np.float32),
                        doc="TrajDiffusion refiner (track D, external-trained) on the match candidates + ATD players; "
                            "xy metres NaN where entropy > ent_thr; state 0 none / 1 on a detection / 2 regressed; "
                            "xy_all = every processed frame; ent1 = one-shot predictive entropy (nats, the uncertainty "
                            "signal); ent = last-pass entropy; conf = refD4 calibrated P(error <= 3 m) (NaN without --calib); "
                            "fused_source 7 = refiner")
    print("wrote", a.out, "coverage %.3f" % ok.mean())


if __name__ == "__main__":
    main()
