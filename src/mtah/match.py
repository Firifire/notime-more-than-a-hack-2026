import os

import numpy as np

from mtah.data import load_tracking, HALVES, PITCH, CACHE

BALL = os.environ.get("MTAH_BALL", "atd")


def ball_array(source=None, trk=None):
    source = source or BALL
    t = trk or load_tracking()
    B = t["ball"].copy()
    if source in ("fused", "fused2", "fused3"):
        fn = {"fused": "ball_video.npz", "fused2": "ball_video_v2.npz",
              "fused3": "ball_video_v3.npz"}[source]
        z = np.load(os.path.join(CACHE, "vision", fn))
        B[:, :2] = z["xy_fused"]
        B[:, 2] = np.nan
    elif source == "atd_ng":
        z = np.load(os.path.join(CACHE, "vision", "ball_video.npz"))
        B[z["atd_ghost"] > 0, :] = np.nan
    elif source != "atd":
        raise ValueError(source)
    return B


def provided_match(source="tactical", ball=None):
    t = load_tracking(source)
    half = np.zeros(len(t["ball"]), np.int8)
    if source == "tactical":
        for h, a, b in HALVES:
            half[a:b + 1] = h
    B = ball_array(ball, t) if source == "tactical" else t["ball"]
    return {"ball": B, "players": t["players"], "team": t["team"],
            "pitch": np.array(PITCH, np.float32), "half": half, "fps": 25}
