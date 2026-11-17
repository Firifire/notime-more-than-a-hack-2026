from __future__ import annotations

from pathlib import Path

import numpy as np

H, A, D, UNK = 0, 1, 2, 3
LABEL_NAMES = {H: "H", A: "A", D: "D", UNK: "?"}
FPS = 25.0

ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "data_external"
COMMON_DIR = EXT_DIR / "common"


def speed_from_positions(xy: np.ndarray, half: np.ndarray, fps: float = FPS, smooth: int = 5) -> np.ndarray:
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    for h in np.unique(half):
        if h == 0:
            continue
        idx = np.flatnonzero(half == h)
        seg = xy[idx[0]: idx[-1] + 1].astype(np.float64)
        if len(seg) < 3:
            continue
        if smooth > 1:
            pad = smooth // 2
            sm = np.full_like(seg, np.nan)
            m = len(seg) - 2 * pad
            sm[pad: len(seg) - pad] = sum(seg[j: j + m] for j in range(smooth)) / smooth
            sm[:pad] = seg[:pad]
            sm[len(seg) - pad:] = seg[len(seg) - pad:]
            seg = sm
        v = np.gradient(seg, axis=0) * fps
        out[idx[0]: idx[-1] + 1] = np.sqrt((v ** 2).sum(-1)).astype(np.float32)
    return out


def spells(label: np.ndarray):
    out = []
    n = len(label)
    i = 0
    while i < n:
        if label[i] in (H, A):
            j = i
            while j < n and label[j] == label[i]:
                j += 1
            out.append((i, j, int(label[i])))
            i = j
        else:
            i += 1
    return out


def stats(z: dict) -> dict:
    label = z["label"]
    fps = float(z["fps"])
    played = label != UNK
    n_played = int(played.sum())
    sp = spells(label)
    durs = np.array([(e - s) / fps for s, e, _ in sp]) if sp else np.zeros(0)
    ball_ok = ~np.isnan(z["ball"][:, 0])
    return {
        "frames": int(len(label)),
        "frames_labelled": n_played,
        "minutes_labelled": n_played / fps / 60,
        "frac_H": float((label == H).sum() / max(n_played, 1)),
        "frac_A": float((label == A).sum() / max(n_played, 1)),
        "frac_D": float((label == D).sum() / max(n_played, 1)),
        "n_spells": len(sp),
        "mean_spell_s": float(durs.mean()) if len(durs) else float("nan"),
        "median_spell_s": float(np.median(durs)) if len(durs) else float("nan"),
        "ball_present_frac": float(ball_ok[played].mean()) if n_played else float("nan"),
        "n_slots": int(z["players"].shape[1]),
    }


def save(name: str, *, ball, players, team, label, half, pitch, fps=FPS) -> dict:
    COMMON_DIR.mkdir(parents=True, exist_ok=True)
    z = dict(
        ball=np.asarray(ball, np.float32),
        players=np.asarray(players, np.float32),
        team=np.asarray(team, np.int8),
        label=np.asarray(label, np.int8),
        half=np.asarray(half, np.int8),
        pitch=np.asarray(pitch, np.float32),
        fps=np.float32(fps),
    )
    n = len(z["label"])
    assert z["ball"].shape == (n, 3), z["ball"].shape
    assert z["players"].shape[0] == n and z["players"].shape[2] == 3
    assert z["team"].shape == (z["players"].shape[1],)
    assert z["half"].shape == (n,)
    np.savez_compressed(COMMON_DIR / f"{name}.npz", **z)
    return stats(z)


def fmt_stats(name: str, s: dict) -> str:
    return (
        f"{name}: {s['minutes_labelled']:.1f} min labelled, H {s['frac_H']:.3f} A {s['frac_A']:.3f} "
        f"D {s['frac_D']:.3f}, {s['n_spells']} spells, mean {s['mean_spell_s']:.1f}s "
        f"(median {s['median_spell_s']:.1f}s), ball present {s['ball_present_frac']:.3f}, "
        f"{s['n_slots']} slots"
    )
