import argparse
import os
import sys
import threading
import queue
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.common import CACHE, FrameReader, W, H, process_ranges, yuv420_to_rgb  # noqa: E402

TRAIN_BALL_PX = (6.0 + 15.0) / 2
BALL_M = 0.22


def ball_zoom(verbose=True):
    from vision.common import HALVES, PITCH, load_homography
    from vision.player_scale import SAMPLE_EVERY, camera_from_homography, round_half
    from mtah.data import load_tracking
    Hs = load_homography()
    pl = load_tracking()["players"][..., :2].astype(np.float64)
    S = np.diag([1.0 / PITCH[0], 1.0 / PITCH[1], 1.0])
    ws = []
    for _, ga, gb in HALVES:
        for t in range(ga, gb + 1, SAMPLE_EVERY):
            P = camera_from_homography(Hs[t] @ S, W, H)
            if P is None:
                continue
            f = np.linalg.norm(P[0, :3] - (W / 2.0) * P[2, :3])
            for x, y in pl[t]:
                if np.isfinite(x) and np.isfinite(y):
                    d = abs((P @ np.array([x, y, BALL_M / 2, 1.0]))[2])
                    ws.append(f * BALL_M / d)
    med = float(np.median(ws))
    ratio = TRAIN_BALL_PX / med
    zoom = round_half(ratio)
    if verbose:
        print("[ball zoom] median projected %.2f m ball: %.3f px over %d positions; %.1f / %.3f = %.4f -> zoom %.1f"
              % (BALL_M, med, len(ws), TRAIN_BALL_PX, med, ratio, zoom), flush=True)
    return zoom, med, ratio


def load_model(weights, device):
    from ultralytics import YOLO
    m = YOLO(weights).model.to(device).eval().half()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def decode_raw(pred, conf_thr, topk, ball_cls=0, iou=0.3):
    import torchvision
    out = []
    boxes = pred[:, :4, :]
    scores = pred[:, 4 + ball_cls, :]
    for b in range(pred.shape[0]):
        s = scores[b]
        keep = s > conf_thr
        if not keep.any():
            out.append(np.zeros((0, 5), np.float32))
            continue
        bx = boxes[b][:, keep].T.float()
        sc = s[keep].float()
        xyxy = torch.cat([bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2], 1)
        k = torchvision.ops.nms(xyxy, sc, iou)[:topk]
        out.append(torch.cat([bx[k], sc[k, None]], 1).cpu().numpy())
    return out


def run(weights, start, stop, scale, batch=8, conf=0.05, topk=16, out=None, device="cuda",
        video=None, keep=None):
    model = load_model(weights, device)
    sw, sh = int(round(W * scale)), int(round(H * scale))
    pw, ph = (sw + 31) // 32 * 32, (sh + 31) // 32 * 32

    q = queue.Queue(maxsize=8)

    def producer():
        buf, idx = [], []
        for i, fr in FrameReader(start, stop, *( [video] if video else [])):
            if keep is not None and not keep[i]:
                continue
            buf.append(fr)
            idx.append(i)
            if len(buf) == batch:
                q.put((idx, np.stack(buf)))
                buf, idx = [], []
        if buf:
            q.put((idx, np.stack(buf)))
        q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    F_, U, V, BW, BH, C = [], [], [], [], [], []
    n, t0 = 0, time.time()
    with torch.inference_mode():
        while True:
            item = q.get()
            if item is None:
                break
            idx, arr = item
            x = torch.from_numpy(arr).to(device, non_blocking=True)
            x = yuv420_to_rgb(x)
            if scale != 1.0:
                x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
            x = F.pad(x, (0, pw - sw, 0, ph - sh), value=114 / 255.0).half()
            pred = model(x)
            pred = pred[0] if isinstance(pred, (list, tuple)) else pred
            dets = decode_raw(pred, conf, topk)
            for i, d in zip(idx, dets):
                if len(d):
                    F_.append(np.full(len(d), i, np.int32))
                    U.append(d[:, 0] / scale)
                    V.append(d[:, 1] / scale)
                    BW.append(d[:, 2] / scale)
                    BH.append(d[:, 3] / scale)
                    C.append(d[:, 4])
            n += len(idx)
            if n % (batch * 250) < batch:
                el = time.time() - t0
                print("  frame %d  %d done  %.1f fps" % (idx[-1], n, n / el), flush=True)
    el = time.time() - t0
    cat = lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt)
    res = dict(frame=cat(F_, np.int32), u=cat(U, np.float32), v=cat(V, np.float32),
               w=cat(BW, np.float32), h=cat(BH, np.float32), conf=cat(C, np.float32),
               start=start, stop=stop, scale=scale, fps=n / max(el, 1e-9), weights=weights)
    out = out or os.path.join(CACHE, "cands", "cands_%d_%d.npz" % (start, stop))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **res)
    print("wrote %s  frames %d  cands %d  %.1f fps" % (out, n, len(res["frame"]), n / el))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(CACHE, "runs", "ball", "weights", "best.pt"))
    ap.add_argument("--start", type=int)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--scale", type=float, default=None, help="zoom (default: the label-free ball_zoom() rule)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--out-dir", default=os.path.join(CACHE, "cands"),
                    help="candidate folder (default cache/vision/cands; run.py step 'vision' passes its own)")
    ap.add_argument("--reverse", action="store_true", help="process chunks last-first "
                    "(run a second process in parallel; existing outputs are skipped)")
    ap.add_argument("--broadcast", action="store_true",
                    help="run on the broadcast video, frames aligned to tactical with a valid "
                         "homography only (cache/broadcast_align.npz best_cost < 1 m)")
    a = ap.parse_args()
    if a.scale is None:
        a.scale = ball_zoom()[0]
    if a.broadcast:
        from vision.common import VIDEO_BROADCAST, graded_mask
        z = np.load(os.path.join(os.path.dirname(CACHE), "broadcast_align.npz"))
        hb = np.load(os.path.join(CACHE, "homography_broadcast.npz"))
        bt, bc = z["best_t"], z["best_cost"]
        keep = (bc < 1.0) & hb["valid"] & (bt >= 0)
        keep[keep] = graded_mask()[bt[keep]]
        idx = np.flatnonzero(keep)
        br = np.flatnonzero(np.diff(idx) > 50)
        starts, ends = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]] + 1
        odir = os.path.join(CACHE, "cands_broadcast")
        pairs = list(zip(starts, ends))
        for s0, e0 in (pairs[::-1] if a.reverse else pairs):
            out = os.path.join(odir, "cands_%d_%d.npz" % (s0, e0))
            if os.path.exists(out):
                continue
            run(a.weights, int(s0), int(e0), scale=a.scale, batch=a.batch, conf=a.conf, out=out,
                video=VIDEO_BROADCAST, keep=keep)
        return
    if a.all:
        jobs = []
        for _, s, e in process_ranges():
            for c in range(s, e, a.chunk):
                jobs.append((c, min(e, c + a.chunk)))
    else:
        jobs = [(a.start, a.stop)]
    for s, e in jobs:
        out = os.path.join(a.out_dir, "cands_%d_%d.npz" % (s, e))
        if os.path.exists(out):
            print("skip", out)
            continue
        run(a.weights, s, e, scale=a.scale, batch=a.batch, conf=a.conf, out=out)


if __name__ == "__main__":
    main()
