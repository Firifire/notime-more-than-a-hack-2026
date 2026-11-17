import glob
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build4 as B4  # noqa: E402
import feats as Fe  # noqa: E402

BASE = B4.BASE
TRAIN = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(B4.Bd.OUT, "train_*.pkl")))
DEV19 = {c for i, c in enumerate(TRAIN) if i % 3 == 0}


def job(path):
    src_dir = os.path.basename(os.path.dirname(path))
    out = os.path.join(BASE, src_dir + "e", os.path.basename(path))
    if os.path.exists(out):
        return out
    X = pickle.load(open(path, "rb"))
    clip = X["clip"]
    base = clip[:14]
    if base in DEV19:
        res0 = pickle.load(open(os.path.join(B4.Bd.OUT, base + ".pkl"), "rb"))
        keep = None
        if clip != base:
            rng = np.random.default_rng(zlib.crc32(clip.encode()))
            prm = B4.draw(rng)
            P = B4.Bd.load(os.path.join(B4.Bd.PACKS, base + ".npz"))
            _, keep = B4.degrade(P, res0["det_gt"], prm, rng)
        embl = Fe.load_emb(base, res0["xy"], keep)
        X["pemb"] = Fe.emb_inputs(X["rows"], embl, X["cand"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pickle.dump(X, open(out + ".tmp", "wb"))
    os.replace(out + ".tmp", out)
    return out


def main():
    paths = sorted(glob.glob(os.path.join(BASE, "canvas3", "train_*.pkl")) +
                   glob.glob(os.path.join(BASE, "canvas4", "train_*.pkl")))
    with ProcessPoolExecutor(int(os.environ.get("D4_WORKERS", 4))) as ex:
        for _ in ex.map(job, paths, chunksize=8):
            pass
    print("done", len(paths), "dev19", sorted(DEV19))


if __name__ == "__main__":
    main()
