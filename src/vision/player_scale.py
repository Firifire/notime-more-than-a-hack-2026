import numpy as np

TRAIN_PERSON_PX = (45.0 + 100.0) / 2
PERSON_M = 1.8
SAMPLE_EVERY = 250


def camera_from_homography(Hm, W, H):
    cx, cy = W / 2.0, H / 2.0
    h1, h2, h3 = Hm[:, 0], Hm[:, 1], Hm[:, 2]

    def uvw(h):
        return h[0] - cx * h[2], h[1] - cy * h[2], h[2]
    u1, v1, w1 = uvw(h1)
    u2, v2, w2 = uvw(h2)
    if abs(w1 * w2) < 1e-12:
        return None
    f2 = -(u1 * u2 + v1 * v2) / (w1 * w2)
    if not np.isfinite(f2) or f2 <= 0:
        return None
    f = np.sqrt(f2)
    Kinv = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1.0]])
    b1, b2, b3 = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = 2.0 / (np.linalg.norm(b1) + np.linalg.norm(b2))
    r1, r2, t = b1 * lam, b2 * lam, b3 * lam
    r3 = np.cross(r1, r2)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.c_[r1, r2, r3, t]


def projected_height(P, x, y, h=PERSON_M):
    foot = P @ np.array([x, y, 0.0, 1.0])
    out = []
    for s in (1.0, -1.0):
        head = P @ np.array([x, y, s * h, 1.0])
        out.append((head[1] / head[2], foot[1] / foot[2]))
    cand = [fv - hv for hv, fv in out if fv - hv > 0]
    return max(cand) if cand else np.nan


def measure(Hs, players_xy, halves, pitch, W, H, every=SAMPLE_EVERY):
    S = np.diag([1.0 / pitch[0], 1.0 / pitch[1], 1.0])
    hs = []
    for _, ga, gb in halves:
        for t in range(ga, gb + 1, every):
            P = camera_from_homography(Hs[t] @ S, W, H)
            if P is None:
                continue
            for x, y in players_xy[t]:
                if np.isfinite(x) and np.isfinite(y):
                    v = projected_height(P, x, y)
                    if np.isfinite(v):
                        hs.append(v)
    return float(np.median(hs)), len(hs)


def round_half(z):
    return float(np.floor(z * 2 + 0.5) / 2)


def inference_zoom(verbose=True):
    from vision.common import W, H, HALVES, PITCH, load_homography
    from mtah.data import load_tracking
    atd = load_tracking()
    med, n = measure(load_homography(), atd["players"][..., :2].astype(np.float64), HALVES, PITCH, W, H)
    ratio = TRAIN_PERSON_PX / med
    zoom = round_half(ratio)
    if verbose:
        print("[player_scale] median projected %.1f m person: %.2f px over %d positions; %.1f / %.2f = %.4f -> zoom %.1f"
              % (PERSON_M, med, n, TRAIN_PERSON_PX, med, ratio, zoom), flush=True)
    return zoom, med, ratio


if __name__ == "__main__":
    inference_zoom()
