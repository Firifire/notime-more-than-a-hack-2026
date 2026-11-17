import argparse
import json
import os
import sys
import time

import numpy as np

EXTRA_MEMBERS = {"mtah.pipe_raw2": ("nnb_",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="eval_ext config JSON (split_file decides label keys)")
    ap.add_argument("--out", required=True, help="signals npz path (runs/<name>_ens_t.signals.npz)")
    ap.add_argument("--workers", type=int, default=None, help="feature workers (default: config)")
    ap.add_argument("--smoke", type=int, default=None, help="TEST ONLY: first N matches per source")
    ap.add_argument("--tracking", action="append", default=[], help="optional corrected provided-match player "
                    "tracking (FINAL key player_fix, mtah.player_fix): read at MATCH INFERENCE only, after the fit; "
                    "FILE = the main prediction, NAME=FILE (repeatable) = an extra prediction written to "
                    "<out>.NAME.signals.npz")
    a = ap.parse_args()
    main_trk = [x for x in a.tracking if not _named(x)]
    if len(main_trk) > 1:
        raise SystemExit("at most one unnamed --tracking FILE")
    extra = [tuple(x.split("=", 1)) for x in a.tracking if _named(x)]
    a.tracking = main_trk[0] if main_trk else None
    t0 = time.time()
    from mtah import eval_ext as E
    cfg = E.resolve(json.load(open(a.config)))
    os.environ["MTAH_BALL"] = cfg["ball"]
    if "mtah.match" in sys.modules and sys.modules["mtah.match"].BALL != cfg["ball"]:
        raise RuntimeError("mtah.match already imported with another ball")
    import importlib
    pipe = importlib.import_module(cfg["pipeline"])
    from mtah import ckpt
    ckpt.guard_duplicate()
    ckpt.install(cfg["pipeline"])
    from mtah.featcache import feature_paths, load_view, code_hash
    from mtah.featset import build, swap_teams
    from mtah.match import provided_match
    from mtah.train_v5 import to_prob, HEADS

    S, split_sha = E.load_split(cfg.get("split_file"))
    deg_profile = E.apply_profile(S)
    exclude = set(S.get("exclude") or [])
    srcs = set(cfg["data"]["dead"]) | set(cfg["data"]["team"])
    pool = [n for n in E.all_names() if E.src_of(n) in srcs and n not in exclude]
    if a.smoke:
        pool = [n for n in pool if sum(E.src_of(x) == E.src_of(n) for x in pool[:pool.index(n)]) < a.smoke]
    dead_set = {n for n in pool if E.src_of(n) in cfg["data"]["dead"]}
    team_set = {n for n in pool if E.src_of(n) in cfg["data"]["team"]}
    train_seed, _ = E.seed_fns(cfg["deg_seed"])
    print("[final] %s: pool %d (%s), split %s, deg_profile %s" % (cfg["name"], len(pool),
          {s: sum(E.src_of(n) == s for n in pool) for s in sorted(srcs)}, split_sha, deg_profile), flush=True)

    keys = [(n, sw, train_seed(n)) for n in pool for sw in (False, True)]
    feats = feature_paths(keys, workers=a.workers or cfg["feature_workers"])
    rows, cols, names = [], None, None
    for n in pool:
        step = cfg["step"]["dead"] if n in dead_set else cfg["step"]["other"]
        lk = E.label_key(n, S)
        for sw in (False, True):
            X, y, half, nm = load_view(feats[(n, sw, train_seed(n))])
            E.check_masked(n, X, nm, S)
            if lk != "label":
                y0, h0 = E.truth(n)
                yl, _ = E.truth(n, lk)
                if sw:
                    y0, yl = E.swap_label(y0), E.swap_label(yl)
                if not (np.array_equal(y, y0) and np.array_equal(half, h0)):
                    raise RuntimeError("cached labels differ from %s.npz" % n)
                y = yl
            if cols is None:
                names = nm
                cols = np.array([i for i, x in enumerate(nm) if x not in set(cfg["drop_features"])])
            elif nm != names:
                raise RuntimeError("feature names differ across cached views")
            bsw, bon, boff = pipe.boundary_labels(y, half)
            keep = np.flatnonzero((y < 3) & (half > 0))[::step]
            Xk = X[keep] if len(cols) == X.shape[1] else X[np.ix_(keep, cols)]
            del X
            rows.append((n, np.ascontiguousarray(Xk), y[keep], bsw[keep], bon[keep], boff[keep]))
    n_rows = sum(len(r[2]) for r in rows)
    print("[final] %d rows x %d feats, %.0fs" % (n_rows, len(cols), time.time() - t0), flush=True)

    saved = dict(pipe.PARAMS)
    pipe.PARAMS.update(n_jobs=cfg["n_jobs"], random_state=cfg["seed"], **cfg["lgb"])
    try:
        t = time.time()
        models = pipe.fit(rows, dead_set, team_set)
        t_fit = time.time() - t
        del rows
        st = ckpt._state() if extra else None
        if a.tracking:
            from mtah import player_fix
            player_fix.use_tracking(a.tracking)

        def predict_match():
            m = provided_match()
            X, fn = build(m)
            Xs, fn2 = build(swap_teams(m))
            assert fn == names and fn2 == names, "provided-match feature names differ from the training views"
            if len(cols) != X.shape[1]:
                X, Xs = X[:, cols], Xs[:, cols]
            import inspect
            takes_half = "half" in inspect.signature(pipe.predict).parameters
            kw = {"half": m["half"], "names": [names[i] for i in cols]} if takes_half else {}
            return m, pipe.predict(models, X=X, Xs=Xs, **kw)
        m, p = predict_match()
        extra_p = []
        for nm_, fp in extra:
            from mtah import player_fix
            ckpt._restore(st)
            player_fix.use_tracking(fp)
            extra_p.append((nm_, fp) + predict_match())
    finally:
        params_used = dict(pipe.PARAMS)
        pipe.PARAMS.clear()
        pipe.PARAMS.update(saved)
    write(a, a.out, a.tracking, cfg, p, m, split_sha, pool, n_rows, params_used, t_fit, t0, deg_profile)
    for nm_, fp, m_, p_ in extra_p:
        write(a, extra_out(a.out, nm_), fp, cfg, p_, m_, split_sha, pool, n_rows, params_used, t_fit, t0, deg_profile)


def _named(x):
    k, eq, _ = x.partition("=")
    return bool(eq) and k.replace("_", "").isalnum()


def extra_out(out, name):
    if out.endswith(".signals.npz"):
        return out[:-len(".signals.npz")] + ".%s.signals.npz" % name
    return out[:-4] + ".%s.npz" % name


def write(a, out, tracking, cfg, p, m, split_sha, pool, n_rows, params_used, t_fit, t0, deg_profile):
    from mtah import eval_ext as E
    from mtah.featcache import code_hash
    from mtah.train_v5 import to_prob, HEADS
    prob = to_prob(p)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    members = {k: np.asarray(v) for k, v in p.items()
               if k.startswith(("lgb_", "nn_", "gs_") + EXTRA_MEMBERS.get(cfg["pipeline"], ()))}
    np.savez(out, p_home=prob[:, 0], p_away=prob[:, 1], p_dead=prob[:, 2], p_switch=p["sw"],
             p_dead_on=p["on"], p_dead_off=p["off"], **{"head_" + k: p[k] for k in HEADS}, **members)
    meta = {"config": cfg, "config_file": a.config, "split_sha": split_sha, "pool": pool, "n_rows": n_rows,
            "feature_hash": code_hash(), "git_head": E.git_head(), "lgb_params": params_used,
            "lgb_device": os.environ.get("MTAH_LGB_DEVICE", "cpu"), "fit_s": round(t_fit, 1),
            "runtime_s": round(time.time() - t0, 1), "peak_rss_gb": E.peak_gb()[0],
            "signals_sha256": E.sha_file(out), "smoke": a.smoke}
    if tracking:
        meta["tracking"] = tracking
    if deg_profile:
        meta["deg_profile"] = deg_profile
    json.dump(meta, open(out[:-4] + ".json", "w"), indent=1)
    print("[final] wrote %s (sha256 %s) | fit %.0fs, total %.0fs, peak RSS %.2f GB | D share %.3f"
          % (out, meta["signals_sha256"][:12], t_fit, meta["runtime_s"], meta["peak_rss_gb"],
             float(np.mean(prob[m["half"] > 0, 2]))), flush=True)


if __name__ == "__main__":
    main()
