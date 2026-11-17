import glob
import os

import numpy as np

from mtah.train_ext import spell_f1


def load_oof(run_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "oof_*.npz"))):
        z = np.load(p)
        name = os.path.basename(p)[4:-4]
        out.append({"name": name, "prob": z["prob"].astype(np.float64), "label": z["label"],
                    "half": z["half"]})
    if not out:
        raise FileNotFoundError("no oof_*.npz in " + run_dir)
    return out


def score_decoder(oof, decoder):
    return {m["name"]: spell_f1(decoder(m["prob"], m["half"]), m["label"], m["half"]) for m in oof}


def gate(base, new, n_boot=5000, seed=0):
    names = sorted(set(base) & set(new))
    d = np.array([new[n] - base[n] for n in names])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [5, 95])
    flips = rng.choice([-1, 1], size=(n_boot, len(d)))
    p = float(np.mean((flips * np.abs(d)).mean(1) >= d.mean()))
    by_src = {}
    for n, x in zip(names, d):
        by_src.setdefault(n.split("_")[0], []).append(x)
    return {"mean_delta": float(d.mean()), "ci90": (float(lo), float(hi)), "p_signflip": p,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": len(d),
            "by_source": {k: float(np.mean(v)) for k, v in by_src.items()},
            "accept": bool(lo > 0 and p < 0.05)}
