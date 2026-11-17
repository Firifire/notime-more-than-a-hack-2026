import os

import numpy as np

from mtah import pipe_ens, pipe_ens2, nn_seq
from mtah.pipe_ens2 import PARAMS, boundary_labels, decode_heads, HEADS  # noqa: F401  (same dict object)

GS_NAMES = ("gs_lg", "gs_lg_m13", "gs_lg_m51", "gs_lg_d25", "gs_pch", "gs_pca", "gs_ploose", "gs_ctrl_diff_m13",
            "gs_bx", "gs_by", "gs_bsd", "gs_pmax_h", "gs_pmax_a", "gs_pmax_diff_m13")
_SWAP_NEG = {"gs_lg", "gs_lg_m13", "gs_lg_m51", "gs_lg_d25", "gs_ctrl_diff_m13", "gs_bx", "gs_by",
             "gs_pmax_diff_m13"}
_SWAP_PAIR = {"gs_pch": "gs_pca", "gs_pca": "gs_pch", "gs_pmax_h": "gs_pmax_a", "gs_pmax_a": "gs_pmax_h"}


def _logit(x, eps=1e-4):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def _cmean(x, w):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, w, mode="nearest")


def frame_features(z, swapped=False):
    pch, pca, plh, pla = (np.asarray(z[k], np.float64) for k in ("p_ch", "p_ca", "p_lh", "p_la"))
    lg = _logit(pch + plh)
    n = len(lg)
    c = np.r_[0, np.cumsum(lg)]
    i = np.arange(n)
    lo, hi = np.maximum(i - 25, 0), np.minimum(i + 26, n)
    sign = np.asarray(z["sign"], np.float64)
    cm = np.asarray(z["c_mass"], np.float64) > 1e-6
    bx = np.where(cm, np.nan_to_num(np.asarray(z["bx"], np.float64)) * sign, 0.0)
    by = np.where(cm, np.nan_to_num(np.asarray(z["by"], np.float64)) * sign, 0.0)
    bsd = np.where(cm, np.nan_to_num(np.asarray(z["bsd"], np.float64), nan=50.0), 50.0)
    ph, pa = np.asarray(z["pmax_h"], np.float64), np.asarray(z["pmax_a"], np.float64)
    F = {"gs_lg": lg, "gs_lg_m13": _cmean(lg, 13), "gs_lg_m51": _cmean(lg, 51),
         "gs_lg_d25": (c[hi] - c[i + 1]) / np.maximum(hi - i - 1, 1) - (c[i] - c[lo]) / np.maximum(i - lo, 1),
         "gs_pch": pch, "gs_pca": pca, "gs_ploose": plh + pla, "gs_ctrl_diff_m13": _cmean(pch - pca, 13),
         "gs_bx": bx, "gs_by": by, "gs_bsd": bsd, "gs_pmax_h": np.nan_to_num(ph), "gs_pmax_a": np.nan_to_num(pa),
         "gs_pmax_diff_m13": _cmean(np.nan_to_num(ph) - np.nan_to_num(pa), 13)}
    if swapped:
        F = {k: (-F[k] if k in _SWAP_NEG else F[_SWAP_PAIR.get(k, k)]) for k in F}
    return {k: np.nan_to_num(v).astype(np.float32) for k, v in F.items()}


def _opts():
    return {k: v for k, v in PARAMS.items() if k.startswith("pipe_gs")}


def _cols(opts):
    c = opts.get("pipe_gs_cols") or list(GS_NAMES)
    bad = set(c) - set(GS_NAMES)
    if bad:
        raise KeyError("unknown pipe_gs_cols %s" % sorted(bad))
    return [k for k in GS_NAMES if k in c]


_GS = {}


def gs_matrix(n, seed, swapped, cols, ver=None):
    from mtah import gs_build
    key = (n, seed, swapped, tuple(cols), ver)
    if key in _GS:
        return _GS[key]
    G = gs_build.groups()
    _check_ver(ver)
    p = gs_build.post_path(n, seed, G[n])
    if not os.path.exists(p):
        gs_build.infer_view((n, seed, G[n]))
    with np.load(p) as z:
        F = frame_features(z, swapped)
    M = np.stack([F[k] for k in cols], 1)
    if len(_GS) > 8:
        _GS.clear()
    _GS[key] = M
    return M


def _check_ver(ver):
    from mtah import gs_build
    if ver not in (None, gs_build.VERSION):
        raise RuntimeError("pipe_gs_ver=%s needs MTAH_GS_VER=%s (environment has %s)" % (ver, ver, gs_build.VERSION))


def gs_matrix_noball(n, seed, swapped, cols):
    from mtah import gs_build, gamestate as gs
    G = gs_build.groups()
    p = os.path.join(gs_build.CACHE, gs_build.VERSION, "post_noball", "%s_s%d_M%s.npz" % (n, seed, G[n]))
    if not os.path.exists(p):
        with np.load(gs_build.post_path(n, seed, G[n])) as z:
            pc = z["pc"].astype(np.float64)
            sign = z["sign"]
        M = gs_build._model(G[n])
        d = gs_build._degrade(gs_build._load(n), seed)
        o = gs.infer(d, pc, M["T"], M["prior"], use_ball=False, norm=gs_build.SETTINGS["norm"])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp.npz"
        np.savez_compressed(tmp, **{k: np.asarray(o[k], np.float32) for k in gs_build.POST_KEYS}, sign=sign)
        os.replace(tmp, p)
    with np.load(p) as z:
        F = frame_features(z, swapped)
    return np.stack([F[k] for k in cols], 1)


_BLIND_GS = {}


def _augment_rows(rows, cols, ver, blind_srcs=()):
    out = []
    _BLIND_GS.clear()
    for r in rows:
        n, y = r[0], r[2]
        keep, lk, swapped, st = pipe_ens2._keep_of(n, y)
        M = gs_matrix(n, pipe_ens._train_seed(n), swapped, cols, ver)
        Xa = np.ascontiguousarray(np.concatenate([r[1], M[keep]], 1))
        if n.split("_")[0] in blind_srcs:
            _BLIND_GS[id(Xa)] = gs_matrix_noball(n, pipe_ens._train_seed(n), swapped, cols)[keep]
        out.append((n, Xa) + tuple(r[2:]))
    return out


class _Patched:

    def __init__(self, cols, ver, nn):
        self.cols, self.ver, self.nn = cols, ver, nn

    def __enter__(self):
        self.orig_pooled, self.orig_check = pipe_ens._pooled, pipe_ens._check_rows
        cols, ver, orig = self.cols, self.ver, self.orig_pooled
        nn = self.nn

        def pooled(n, sw, seed):
            segs, y, half = orig(n, sw, seed)
            if not nn:
                return segs, y, half
            M = gs_matrix(n, seed, sw, cols, ver)
            F0 = segs[0][3].shape[1] - len(nn_seq.MISS_COLS) - 1
            out = []
            for h, a, k, P in segs:
                nb = P.shape[0]
                Ms = M[a:a + k]
                if nb * nn_seq.POOL > k:
                    Ms = np.concatenate([Ms, np.repeat(Ms[-1:], nb * nn_seq.POOL - k, 0)])
                Mp = Ms.reshape(nb, nn_seq.POOL, -1).astype(np.float64).mean(1).astype(np.float32)
                out.append((h, a, k, np.concatenate([P[:, :F0], Mp, P[:, F0:]], 1)))
            return out, y, half
        pipe_ens._pooled = pooled
        pipe_ens._check_rows = lambda rows: None
        from mtah import blind
        self.orig_fn, self.orig_blind = pipe_ens2._feature_names, blind.blind_rows
        orig_fn, orig_blind = self.orig_fn, self.orig_blind
        pipe_ens2._feature_names = lambda: list(orig_fn()) + list(cols)

        def blind_rows(X, names):
            Y = orig_blind(X, names)
            if id(X) not in _BLIND_GS:
                raise RuntimeError("pipe_gs: blind copy of an unknown row block")
            Y[:, -len(cols):] = _BLIND_GS[id(X)]
            return Y
        blind.blind_rows = blind_rows
        return self

    def __exit__(self, *a):
        from mtah import blind
        pipe_ens._pooled, pipe_ens._check_rows = self.orig_pooled, self.orig_check
        pipe_ens2._feature_names, blind.blind_rows = self.orig_fn, self.orig_blind
        _BLIND_GS.clear()


def _ens2_params():
    return {k: PARAMS.pop(k) for k in [k for k in PARAMS if k.startswith("pipe_gs")]}


def fit(rows, dead_set, team_set):
    opts = _opts()
    cols, ver, nn = _cols(opts), opts.get("pipe_gs_ver"), bool(opts.get("pipe_gs_nn", True))
    pipe_ens._check_rows(rows)
    bl = PARAMS.get("pipe_dead_blind") or None
    if bl and float(bl.get("nn_share", 0.5)) > 0 and nn:
        raise RuntimeError("pipe_gs: pipe_dead_blind with a blind BiGRU share is not supported (use nn_share 0)")
    rows = _augment_rows(rows, cols, ver, tuple(bl.get("sources", ["pff"])) if bl else ())
    print("  [pipe_gs] +%d GS columns (%s), nn %s" % (len(cols), ver or "default", nn), flush=True)
    saved = _ens2_params()
    try:
        with _Patched(cols, ver, nn):
            models = pipe_ens2.fit(rows, dead_set, team_set)
    finally:
        PARAMS.update(saved)
    models["gs"] = {"cols": cols, "ver": ver, "nn": nn}
    return models


def _provided_gs(cols, half=None):
    from mtah import gs_build, gamestate as gs
    from mtah.match import provided_match
    import pickle
    M = pickle.load(open(gs_build.model_path("all"), "rb"))
    m = provided_match()
    if half is not None and (len(m["half"]) != len(half) or not np.array_equal(m["half"], half)):
        raise RuntimeError("pipe_gs: half= given but it is not the provided match's")
    F, pres = gs.player_features(m, gk_mode=gs_build.SETTINGS["gk_mode"])
    pc = gs.carrier_probs(M["model"], F, pres)
    del F
    o = gs.infer(m, pc, M["T"], M["prior"], norm=gs_build.SETTINGS["norm"])
    z = {k: o[k] for k in gs_build.POST_KEYS}
    z["sign"] = gs._attack_sign(m["players"][:, :, :2], np.asarray(m["team"]), m["half"])
    out = []
    for sw in (False, True):
        Fd = frame_features(z, sw)
        out.append(np.stack([Fd[k] for k in cols], 1))
    return out


def predict(models, m=None, X=None, Xs=None, half=None, names=None):
    if X is None:
        raise ValueError("pipe_gs.predict needs the feature views X / Xs")
    g = models["gs"]
    cols, ver, nn = g["cols"], g["ver"], g["nn"]
    _check_ver(ver)
    if half is None:
        n, half, fnames = pipe_ens._identify(X)
        Mo = gs_matrix(n, pipe_ens._ctx()["eval_seed"](n), False, cols, ver)
        Ms = gs_matrix(n, pipe_ens._ctx()["eval_seed"](n), True, cols, ver)
    else:
        fnames = list(names)
        Mo, Ms = _provided_gs(cols, half)
    Xa, Xsa = np.concatenate([X, Mo], 1), np.concatenate([Xs, Ms], 1)
    if nn:
        return pipe_ens2.predict(models, X=Xa, Xs=Xsa, half=half, names=fnames + list(cols))
    pl = pipe_ens2.predict_lgb(models["lgb"], Xa, Xsa)
    q = pipe_ens2.predict({"lgb": _NoLgb(pl), "nn": models["nn"], "mu": models["mu"], "sd": models["sd"]},
                          X=X, Xs=Xs, half=half, names=fnames)
    return q


class _NoLgb(dict):

    def __init__(self, p):
        super().__init__()
        self.p = p


_orig_predict_lgb = pipe_ens2.predict_lgb


def _predict_lgb(models, X, Xs):
    if isinstance(models, _NoLgb):
        return models.p
    return _orig_predict_lgb(models, X, Xs)


pipe_ens2.predict_lgb = _predict_lgb
