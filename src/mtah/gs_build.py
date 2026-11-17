import argparse
import hashlib
import multiprocessing
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mtah import gamestate as gs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get("MTAH_GS_CACHE") or os.path.join(ROOT, "cache", "gs")
VERSION = os.environ.get("MTAH_GS_VER", "v1")
SETTINGS = {"v1": {"gk_mode": "half", "norm": "v1"}, "v2": {"gk_mode": "frame", "norm": "v2"}}[VERSION]
BALL_SOURCES = ("idsse-ctrl", "pff", "metrica", "skillcorner", "skillcorner2020")
ROW_STEP = {"idsse-ctrl": 4, "pff": 20, "metrica": 6, "skillcorner": 10, "skillcorner2020": 10}
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True)


def _pool(workers):
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


def _E():
    from mtah import eval_ext as E
    return E


def split():
    E = _E()
    s, _ = E.load_split("notes/avo/split_v3.json")
    return s


def groups():
    E = _E()
    s = split()
    dev = sorted(s["dev"], key=lambda n: (E.src_of(n), n))
    g = {}
    for r, n in enumerate(dev):
        g[E.group_of(n)] = str(r % 4)
    for n in s["sealed"]:
        g[E.group_of(n)] = "S"
    out = {}
    for n in E.all_names():
        k = E.group_of(n)
        out[n] = g.get(k, str(int(hashlib.sha1(k.encode()).hexdigest()[:8], 16) % 4))
    return out


def _load(name):
    from mtah.train_ext import load_ext
    return [x for x in load_ext([name.split("_")[0]]) if x["name"] == name][0]


def _degrade(m, seed):
    from mtah.degrade_ext_rand import degrade_ext_rand
    return degrade_ext_rand(m, np.random.default_rng(seed))


def train_seed(n):
    return _E().seed_fns(0)[0](n)


def eval_seed(n):
    return _E().seed_fns(0)[1](n)


def rows_path(n):
    return os.path.join(CACHE, VERSION, "rows", n + ".npz")


def _rows(n):
    p = rows_path(n)
    if os.path.exists(p):
        return p
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    m = _load(n)
    d = _degrade(m, train_seed(n))
    F, pres = gs.player_features(d, gk_mode=SETTINGS["gk_mode"])
    tgt = gs.carrier_target(m)
    st = gs.team_state(tgt, m["team"])
    fr = np.flatnonzero(tgt > -2)[::ROW_STEP[n.split("_")[0]]]
    ff, ss = np.nonzero(pres[fr])
    X = F[fr[ff], ss]
    y = (tgt[fr[ff]] == ss).astype(np.int8)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, X=X.astype(np.float32), y=y, frame=fr[ff].astype(np.int32), state=st)
    os.replace(tmp, p)
    return p


def ball_names():
    E = _E()
    return [n for n in E.all_names() if E.src_of(n) in BALL_SOURCES]


def model_path(g):
    return os.path.join(CACHE, VERSION, "models", "M_%s.pkl" % g)


def fit_group(g, n_jobs=8):
    import lightgbm as lgb
    G = groups()
    names = [n for n in ball_names() if (g == "all" or (G[n] != g and G[n] != "S"))]
    if g == "S":
        names = [n for n in ball_names() if G[n] != "S"]
    Xs, ys, sts = [], [], []
    for n in names:
        z = np.load(rows_path(n))
        Xs.append(z["X"])
        ys.append(z["y"])
        sts.append(z["state"])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    t = time.time()
    clf = lgb.LGBMClassifier(**LGB, n_jobs=n_jobs, random_state=0).fit(X, y)
    T = gs.fit_dynamics(sts)
    prior = float(y.mean())
    out = {"model": clf, "T": T, "prior": prior, "names": names, "group": g, "n_rows": len(y),
           "features": list(gs.FEATS)}
    os.makedirs(os.path.dirname(model_path(g)), exist_ok=True)
    pickle.dump(out, open(model_path(g), "wb"))
    print("M_%s: %d matches, %d rows, prior %.4f, fit %.0fs; T diag %s" % (
        g, len(names), len(y), prior, time.time() - t, np.round(np.diag(T), 4)), flush=True)
    return out


POST_KEYS = ("p_ch", "p_ca", "p_lh", "p_la", "bx", "by", "bsd", "pmax_h", "pmax_a", "c_mass")


def post_path(n, seed, g):
    return os.path.join(CACHE, VERSION, "post", "%s_s%d_M%s.npz" % (n, seed, g))


_MODELS = {}


def _model(g):
    if g not in _MODELS:
        _MODELS[g] = pickle.load(open(model_path(g), "rb"))
    return _MODELS[g]


def infer_view(args):
    n, seed, g = args
    p = post_path(n, seed, g)
    if os.path.exists(p):
        return p
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    M = _model(g)
    m = _load(n)
    d = _degrade(m, seed)
    F, pres = gs.player_features(d, gk_mode=SETTINGS["gk_mode"])
    M["model"].set_params(n_jobs=2)
    pc = gs.carrier_probs(M["model"], F, pres)
    del F
    o = gs.infer(d, pc, M["T"], M["prior"], norm=SETTINGS["norm"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, **{k: np.asarray(o[k], np.float32) for k in POST_KEYS},
                        pc=pc.astype(np.float16), ball_obs=~np.isnan(d["ball"][:, 0]),
                        sign=gs._attack_sign(d["players"][:, :, :2], np.asarray(d["team"]), d["half"]))
    os.replace(tmp, p)
    return p


def views_needed():
    E = _E()
    s = split()
    G = groups()
    excl = set(s.get("exclude") or [])
    out = set()
    pool_srcs = ("idsse-ctrl", "metrica", "pff", "skillcorner", "stv2")
    for n in E.all_names():
        if n in excl or E.src_of(n) not in pool_srcs:
            continue
        out.add((n, train_seed(n), G[n]))
    for n in s["dev"] + s["sealed"]:
        out.add((n, eval_seed(n), G[n]))
    return sorted(out)


GS_PIPELINES = ("mtah.pipe_gs", "mtah.pipe_gsraw", "mtah.pipe_gsraw2")
GROUPS = ("0", "1", "2", "3", "S", "all")


def uses_gs(cfg):
    return cfg.get("pipeline") in GS_PIPELINES


def cfg_version(cfg):
    return (cfg.get("lgb") or {}).get("pipe_gs_ver") or "v1"


def final_views(cfg):
    E = _E()
    s, _ = E.load_split(cfg.get("split_file"))
    excl = set(s.get("exclude") or [])
    srcs = set(cfg["data"]["dead"]) | set(cfg["data"]["team"])
    G = groups()
    return [(n, train_seed(n), G[n]) for n in E.all_names() if E.src_of(n) in srcs and n not in excl]


def ensure(cfg_path, workers=6, expect_train=None):
    cfg = json.load(open(cfg_path))
    if not uses_gs(cfg):
        print("[gs ensure] %s does not use the game-state member: nothing to do" % cfg.get("pipeline"), flush=True)
        return
    if cfg_version(cfg) != VERSION:
        raise SystemExit("config needs GS %s but MTAH_GS_VER=%s" % (cfg_version(cfg), VERSION))
    names = ball_names()
    if expect_train is not None and sorted(names) != sorted(expect_train):
        raise SystemExit("carrier-model training matches differ from the build: missing %s, extra %s"
                         % (sorted(set(expect_train) - set(names))[:5], sorted(set(names) - set(expect_train))[:5]))
    t = time.time()
    todo = [n for n in names if not os.path.exists(rows_path(n))]
    fits = [g for g in GROUPS if not os.path.exists(model_path(g))]
    if todo:
        fits = list(GROUPS)
    views = [v for v in final_views(cfg) if not os.path.exists(post_path(*v))]
    print("[gs ensure] %s cache %s: rows missing %d/%d, models missing %s, posteriors missing %d/%d"
          % (VERSION, CACHE, len(todo), len(names), fits, len(views), len(final_views(cfg))), flush=True)
    if todo:
        with _pool(workers) as ex:
            for i, p in enumerate(ex.map(_rows, todo)):
                print("  rows %d/%d %s %.0fs" % (i + 1, len(todo), os.path.basename(p), time.time() - t), flush=True)
    for g in fits:
        fit_group(g)
    if fits:
        views = final_views(cfg) if len(fits) == len(GROUPS) else \
            [v for v in final_views(cfg) if v[2] in fits or not os.path.exists(post_path(*v))]
        for v in views:
            if os.path.exists(post_path(*v)):
                os.remove(post_path(*v))
    if views:
        _MODELS.clear()
        with _pool(workers) as ex:
            for i, p in enumerate(ex.map(infer_view, views)):
                print("  post %d/%d %s %.0fs" % (i + 1, len(views), os.path.basename(p), time.time() - t), flush=True)
    print("[gs ensure] done in %.0f s" % (time.time() - t), flush=True)


def validate(names=None, g_override=None):
    E = _E()
    s = split()
    G = groups()
    names = names or [n for n in s["dev"] if E.src_of(n) != "stv2"]
    res = {}
    oofdir = os.path.join(ROOT, "runs", "eval", "ens2_spellw_s3_teamstv2_v3_oof")
    for n in names:
        g = g_override or G[n]
        p = post_path(n, eval_seed(n), g)
        if not os.path.exists(p):
            continue
        z = np.load(p)
        m = _load(n)
        lab, half = m["label"], m["half"]
        P, B = gs._centre(m)
        tgt = gs.carrier_target(m)
        st = gs.team_state(tgt, m["team"])
        obs = z["ball_obs"]
        hid = (~obs) & (half > 0) & ~np.isnan(B[:, 0]) & (lab < 2)
        d = _degrade(m, eval_seed(n))
        Bo = gs._centre(d)[1]
        idx = np.arange(len(Bo))
        lin = np.full_like(Bo, np.nan)
        for h in np.unique(half[half > 0]):
            mh = np.flatnonzero((half == h) & obs)
            if len(mh) < 2:
                continue
            sel = np.flatnonzero(half == h)
            for k in (0, 1):
                lin[sel, k] = np.interp(sel, mh, Bo[mh, k])
        gsxy = np.c_[z["bx"], z["by"]]
        e_gs = np.linalg.norm(gsxy[hid] - B[hid], axis=1)
        e_lin = np.linalg.norm(lin[hid] - B[hid], axis=1)
        p_home = z["p_ch"] + z["p_lh"]
        okst = hid & (st >= 0)
        stt = np.where(st[okst] % 2 == 0, 0, 1)
        acc_state = float(np.mean((p_home[okst] < 0.5) == (stt == 1))) if okst.any() else np.nan
        acc_lab = float(np.mean((p_home[hid] < 0.5) == (lab[hid] == 1))) if hid.any() else np.nan
        vis = obs & (half > 0) & (lab < 2)
        acc_lab_vis = float(np.mean((p_home[vis] < 0.5) == (lab[vis] == 1))) if vis.any() else np.nan
        r = {"group": g, "hidden_frames": int(hid.sum()), "ball_err_gs_med": float(np.nanmedian(e_gs)),
             "ball_err_lin_med": float(np.nanmedian(e_lin)), "ball_err_gs_mean": float(np.nanmean(e_gs)),
             "ball_err_lin_mean": float(np.nanmean(e_lin)), "acc_ctrlstate_hidden": acc_state,
             "acc_label_hidden": acc_lab, "acc_label_visible": acc_lab_vis}
        of = os.path.join(oofdir, "oof_%s.npz" % n)
        if os.path.exists(of):
            o = np.load(of)
            r["oof_team_acc_hidden"] = float(np.mean((o["team"][hid] < 0.5) == (lab[hid] == 1)))
            r["oof_team_acc_visible"] = float(np.mean((o["team"][vis] < 0.5) == (lab[vis] == 1)))
            lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / np.clip(1 - x, 1e-4, 1))
            mix = lg(o["team"]) + lg(p_home)
            r["mix_acc_hidden"] = float(np.mean((mix[hid] < 0) == (lab[hid] == 1)))
            r["mix_acc_visible"] = float(np.mean((mix[vis] < 0) == (lab[vis] == 1)))
        e = np.diff(np.r_[0, hid.astype(np.int8), 0])
        a_, b_ = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
        ends = [b - 1 for a, b in zip(a_, b_) if b - a >= 50 and st[b - 1] >= 0]
        if ends:
            ends = np.array(ends)
            r["gap_end_n"] = len(ends)
            r["gap_end_acc_gs"] = float(np.mean((p_home[ends] >= 0.5) == (st[ends] % 2 == 0)))
            starts = [a for a, b in zip(a_, b_) if b - a >= 50 and st[b - 1] >= 0]
            bl = []
            for a in starts:
                bl.append(p_home[max(a - 1, 0)] >= 0.5)
            r["gap_end_acc_hold"] = float(np.mean(np.array(bl) == (st[ends] % 2 == 0)))
        res[n] = r
        print(n, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    return res


def summarise(res):
    by = {}
    for n, r in res.items():
        by.setdefault(n.split("_")[0], []).append(r)
    out = {}
    for s_, rs in by.items():
        keys = [k for k in rs[0] if isinstance(rs[0][k], float)]
        out[s_] = {k: round(float(np.nanmean([r.get(k, np.nan) for r in rs])), 4) for k in keys}
        out[s_]["n"] = len(rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rows", "fit", "infer", "validate", "groups", "ensure"])
    ap.add_argument("--config", default=None, help="ensure: eval_ext config of the final recipe")
    ap.add_argument("--expect-train", default=None, help="ensure: JSON file whose 'gs_train' lists the carrier "
                    "models' training matches (docs/external_manifest.json)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--groups", nargs="*", default=["0", "1", "2", "3", "S", "all"])
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--only-eval", action="store_true", help="infer: eval views of dev matches only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "ensure":
        exp = json.load(open(a.expect_train)).get("gs_train") if a.expect_train else None
        ensure(a.config, a.workers, exp)
    elif a.cmd == "groups":
        G = groups()
        print({g: sum(v == g for v in G.values()) for g in sorted(set(G.values()))})
    elif a.cmd == "rows":
        names = a.names or ball_names()
        t = time.time()
        with _pool(a.workers) as ex:
            for i, p in enumerate(ex.map(_rows, names)):
                print("%d/%d %s %.0fs" % (i + 1, len(names), os.path.basename(p), time.time() - t), flush=True)
    elif a.cmd == "fit":
        for g in a.groups:
            fit_group(g)
    elif a.cmd == "infer":
        v = views_needed()
        if a.only_eval:
            s = split()
            v = [x for x in v if x[0] in s["dev"] and x[1] >= 1000]
        if a.names:
            v = [x for x in v if x[0] in a.names]
        t = time.time()
        with _pool(a.workers) as ex:
            for i, p in enumerate(ex.map(infer_view, v)):
                print("%d/%d %s %.0fs" % (i + 1, len(v), os.path.basename(p), time.time() - t), flush=True)
    elif a.cmd == "validate":
        res = validate(a.names)
        sm = summarise(res)
        print(json.dumps(sm, indent=1))
        if a.out:
            json.dump({"per_match": res, "summary": sm}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()


def sweep(names, grid, seed_fn="train"):
    G = groups()
    out = {}
    for n in names:
        seed = train_seed(n) if seed_fn == "train" else eval_seed(n)
        p = post_path(n, seed, G[n])
        if not os.path.exists(p):
            continue
        z = np.load(p)
        m = _load(n)
        d = _degrade(m, seed)
        M = _model(G[n])
        pc = z["pc"].astype(np.float64)
        lab, half = m["label"], m["half"]
        obs = ~np.isnan(d["ball"][:, 0])
        hid = (~obs) & (half > 0) & (lab < 2)
        vis = obs & (half > 0) & (lab < 2)
        for k, kw in grid.items():
            kw2 = dict({"norm": SETTINGS["norm"]}, **{a: b for a, b in kw.items() if a != "T"})
            o = gs.infer(d, pc, kw.get("T", M["T"]), M["prior"], **kw2)
            ph = o["p_ch"] + o["p_lh"]
            r = out.setdefault(k, {"hid": [], "vis": []})
            r["hid"].append(float(np.mean((ph[hid] < 0.5) == (lab[hid] == 1))) if hid.any() else np.nan)
            r["vis"].append(float(np.mean((ph[vis] < 0.5) == (lab[vis] == 1))) if vis.any() else np.nan)
    return {k: {a: round(float(np.nanmean(b)), 4) for a, b in v.items()} for k, v in out.items()}
