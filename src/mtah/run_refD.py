"""Entry points for the refined-ball (Track D) possession experiments: the frozen tools with the `refD_ext`
degradation profile registered (mtah.degrade_refD_ext.register) and, for the provided match, the refined ball as
input. Nothing frozen is edited; existing ball sources and profiles behave exactly as before.

    # external evaluation (eval_ext unchanged; split notes/avo/split_v4_refD.json = split_v4 lists with
    # "deg_profile": "refD_ext" for training AND evaluation views)
    PYTHONPATH=src MTAH_LGB_DEVICE=cpu .venv/bin/python -m mtah.run_refD eval --config C --out R.json [--save-oof D]
    # provided-match signals of a refD_ext config (final_signals unchanged) with the provided ball replaced by the
    # refined ball cache/vision/ball_video_refD.npz (inference only)
    PYTHONPATH=src MTAH_LGB_DEVICE=cpu .venv/bin/python -m mtah.run_refD final --config C --out O.signals.npz

The provided-match ball is chosen from the config split's deg_profile (BALL_OF_PROFILE: refD_ext -> ball_video_refD.npz,
refD3_ext -> ball_video_refD3.npz), so input and training corruption always match.
Ball source "refD" (this module only; mtah.match.ball_array is not changed): xy = the refiner's `xy` (= xy_fused,
100% of graded frames; trained on external data only, phase 2 of Track D), z = NaN. The refiner's uncertainty
(`ent1`) is NOT used: the possession features have no reliability channel and no compliant threshold was chosen.
"""
import os
import sys

import numpy as np

REFD_FILE = "ball_video_refD.npz"
BALL_OF_PROFILE = {"refD_ext": "ball_video_refD.npz", "refD3_ext": "ball_video_refD3.npz",
                   "refD3gap_ext": "ball_video_refD3.npz", "refD4gap_ext": "ball_video_refD4.npz"}
GAP_ONLY = ("refD3gap_ext", "refD4gap_ext")


def ball_array_refD(trk=None, fn=REFD_FILE):
    from mtah.data import CACHE, load_tracking
    t = trk or load_tracking()
    B = t["ball"].copy()
    with np.load(os.path.join(CACHE, "vision", fn)) as z:
        B[:, :2] = z["xy"]
        thr = os.environ.get("MTAH_REFD4_THR")
        if thr and "conf" in z.files:
            B[z["conf"] < float(thr), :2] = np.nan
            print("[run_refD] %s: abstain where conf < %s" % (fn, thr), flush=True)
    B[:, 2] = np.nan
    return B


def _patch_provided_match(fn, gap_only=False):
    import mtah.match as mm
    orig = mm.provided_match

    def provided_match_refD(source="tactical", ball=None):
        m = orig(source, ball)
        if source == "tactical":
            R = ball_array_refD(fn=fn)
            if gap_only:
                pre = m["ball"]
                gap = np.isnan(pre[:, 0]) & (m["half"] > 0) & ~np.isnan(R[:, 0])
                B = pre.copy()
                B[gap, :2] = R[gap, :2]
                B[gap, 2:] = np.nan
                m["ball"], m["ball_prefill"] = B, pre
                print("[run_refD] provided-match ball = ATD + cache/vision/%s in %d gap frames (%.3f of periods)"
                      % (fn, int(gap.sum()), float(gap[m["half"] > 0].mean())), flush=True)
            else:
                m["ball"] = R
                print("[run_refD] provided-match ball = cache/vision/%s" % fn, flush=True)
        return m
    mm.provided_match = provided_match_refD


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("eval", "final"):
        sys.exit(__doc__)
    cmd = sys.argv[1]
    from mtah.degrade_refD4gap_ext import register
    register()
    if cmd == "eval":
        from mtah import eval_ext
        sys.argv = ["eval_ext"] + sys.argv[2:]
        eval_ext.main()
    else:
        import json
        from mtah import eval_ext as E
        cfg = json.load(open(sys.argv[sys.argv.index("--config") + 1]))
        prof = E.load_split(cfg.get("split_file"))[0].get("deg_profile")
        if prof not in BALL_OF_PROFILE:
            raise SystemExit("final: the config's split must fix a refined-ball profile, got %r" % prof)
        _patch_provided_match(BALL_OF_PROFILE[prof], gap_only=prof in GAP_ONLY)
        from mtah import final_signals
        sys.argv = ["final_signals"] + sys.argv[2:]
        final_signals.main()


if __name__ == "__main__":
    main()
