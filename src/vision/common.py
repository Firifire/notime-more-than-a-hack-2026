import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, DATA_DIR, asset_paths, half_video_blocks  # noqa: E402

DATA = DATA_DIR
VIDEO = asset_paths(GAME)["video"]
VIDEO_BROADCAST = asset_paths(GAME, source="broadcast")["video"]
CACHE = os.path.join(ROOT, "cache", "vision")
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]
HALVES = half_video_blocks(GAME)
W, H = GAMES[GAME]["frame_size"]
MARGIN = int(round(2 * FPS))
_CHECKED = set()


def check_video_size(path=VIDEO):
    if path in _CHECKED:
        return
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0:s=x", path], stdout=subprocess.PIPE, check=True, text=True).stdout
    w, h = (int(x) for x in out.strip().split("x")[:2])
    if (w, h) != (W, H):
        raise SystemExit("%s: frame size %dx%d differs from the declared %dx%d (games.py frame_size)" % (path, w, h, W, H))
    _CHECKED.add(path)


def process_ranges(margin=MARGIN):
    return [(h, max(0, a - margin), min(N_FRAMES, b + 1 + margin)) for h, a, b in HALVES]


def graded_mask():
    m = np.zeros(N_FRAMES, bool)
    for _, a, b in HALVES:
        m[a:b + 1] = True
    return m


_HCACHE = os.path.join(ROOT, "cache", "homography_tactical.npz")


def load_homography():
    if os.path.exists(_HCACHE):
        return np.load(_HCACHE)["H"].reshape(-1, 3, 3).astype(np.float64)
    import pandas as pd
    t = pd.read_csv(asset_paths(GAME)["homography"])
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    return t[cols].values.astype(np.float64).reshape(-1, 3, 3)


def apply_h(M, x, y):
    d = M[..., 2, 0] * x + M[..., 2, 1] * y + M[..., 2, 2]
    return ((M[..., 0, 0] * x + M[..., 0, 1] * y + M[..., 0, 2]) / d,
            (M[..., 1, 0] * x + M[..., 1, 1] * y + M[..., 1, 2]) / d)


def metres_to_px(Hs, xm, ym):
    return apply_h(Hs, xm / PITCH[0], ym / PITCH[1])


def px_to_metres(Hinv, u, v):
    x, y = apply_h(Hinv, u, v)
    return x * PITCH[0], y * PITCH[1]


def load_atd_ball():
    from mtah.data import load_tracking
    return load_tracking()["ball"][:, :2].astype(np.float64)


class FrameReader:

    def __init__(self, start, stop, path=VIDEO, gpu=True):
        check_video_size(path)
        self.start, self.stop = int(start), int(stop)
        cmd = ["ffmpeg", "-v", "error", "-nostdin"]
        if gpu:
            cmd += ["-c:v", "h264_cuvid"]
        cmd += ["-ss", "%.6f" % (self.start / FPS), "-i", path,
                "-frames:v", str(self.stop - self.start), "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-"]
        self.cmd = cmd
        self.fsize = W * H * 3 // 2

    def __iter__(self):
        p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, bufsize=self.fsize * 4)
        try:
            i = self.start
            while i < self.stop:
                buf = p.stdout.read(self.fsize)
                if len(buf) < self.fsize:
                    break
                yield i, np.frombuffer(buf, np.uint8).reshape(H * 3 // 2, W)
                i += 1
        finally:
            p.stdout.close()
            p.kill()
            p.wait()


def yuv420_to_rgb(batch_u8):
    import torch
    import torch.nn.functional as F
    B = batch_u8.shape[0]
    y = batch_u8[:, :H, :].float()
    uv = batch_u8[:, H:, :].reshape(B, 2, H // 2, W // 2).float()
    uv = F.interpolate(uv, scale_factor=2, mode="nearest")
    y = (y - 16.0) * (255.0 / 219.0)
    u = (uv[:, 0] - 128.0) * (255.0 / 224.0)
    v = (uv[:, 1] - 128.0) * (255.0 / 224.0)
    r = y + 1.5748 * v
    g = y - 0.1873 * u - 0.4681 * v
    b = y + 1.8556 * u
    return (torch.stack([r, g, b], 1) / 255.0).clamp_(0, 1)


def verify_seek(frames=None, path=VIDEO, back_s=4):
    if frames is None:
        frames = [a for _, a, _ in HALVES] + [(HALVES[0][1] + HALVES[0][2]) // 2]
    out = []
    for f in frames:
        k = f - int(round(back_s * FPS))
        seq = None
        for i, fr in FrameReader(max(0, k), f + 1, path):
            if i == f:
                seq = fr.copy()
        _, direct = next(iter(FrameReader(f, f + 1, path)))
        _, nxt = next(iter(FrameReader(f + 1, f + 2, path)))
        out.append((f, bool(np.array_equal(seq, direct)), bool(np.array_equal(seq, nxt))))
    return out


if __name__ == "__main__":
    print("seek check (frame, equal_to_sequential, equal_to_next[should be False]):", verify_seek())
