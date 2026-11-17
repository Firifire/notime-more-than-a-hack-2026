import hashlib
import inspect
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mtah.data import CACHE

DIR = os.path.join(CACHE, "featcache")


def code_hash(profile=None):
    import mtah.featset as fs
    import mtah.ball as bl
    import mtah.train_ext as te
    from mtah.degrade_profiles import active_profile, profile_hash
    src = inspect.getsource(fs) + inspect.getsource(bl) + \
        inspect.getsource(getattr(te, "_degrade_default", te.degrade)) + inspect.getsource(te.our_gap_lengths)
    profile = profile or active_profile()
    if profile != "default":
        src += "\nprofile:" + profile + ":" + profile_hash(profile)
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def _profile_of(seed):
    from mtah.degrade_profiles import active_profile
    return active_profile(seed)


def _path(name, swap, seed, h, profile="default"):
    from mtah.match import BALL
    sub = h + "_" + BALL + ("" if profile == "default" else "_p-" + profile)
    return os.path.join(DIR, sub, "%s_%s_s%d.npz" % (name, "swap" if swap else "orig", seed))


def _compute(args):
    name, swap, seed, h, prof = args
    p = _path(name, swap, seed, h, prof)
    if os.path.exists(p):
        return p
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from mtah.featset import build, swap_teams
    from mtah.train_ext import load_ext, our_gap_lengths, degrade
    from mtah.degrade_profiles import COMPLIANT
    m = [x for x in load_ext([name.split("_")[0]]) if x["name"] == name][0]
    if swap:
        m = swap_teams(m)
    gaps, miss = (None, None) if prof in COMPLIANT else our_gap_lengths()
    d = degrade(m, np.random.default_rng(seed), gaps, miss, profile=prof)
    X, names = build(d)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    save = np.savez if prof == "default" else np.savez_compressed
    save(tmp, X=X.astype(np.float32), y=m["label"], half=m["half"], names=np.array(names))
    os.replace(tmp, p)
    return p


def feature_paths(keys, workers=6):
    todo = []
    for n, s, sd in keys:
        prof = _profile_of(sd)
        todo.append((n, s, sd, code_hash(prof), prof))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(_compute, todo))
    return dict(zip(keys, paths))


def load_view(path):
    z = np.load(path)
    return z["X"], z["y"], z["half"], list(z["names"])


def features_parallel(keys, workers=6):
    todo = []
    for n, s, sd in keys:
        prof = _profile_of(sd)
        todo.append((n, s, sd, code_hash(prof), prof))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(_compute, todo))
    out = {}
    for (n, s, sd), p in zip(keys, paths):
        z = np.load(p)
        out[(n, s, sd)] = (z["X"], z["y"], z["half"], list(z["names"]))
    return out
