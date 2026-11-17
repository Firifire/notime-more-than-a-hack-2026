import argparse
import copy
import glob
import hashlib
import importlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, "data_external", "common")
SPLIT_PATH = os.path.join(ROOT, "notes", "avo", "split.json")
CANON_SOURCES = ("idsse-ctrl", "metrica", "pff", "skillcorner")
TUNE_SOURCES = ("idsse-ctrl", "pff")

DEFAULT = {
    "name": None,
    "pipeline": "mtah.train_v5",
    "ball": "fused",
    "data": {"dead": ["idsse-ctrl"], "team": ["idsse-ctrl", "metrica", "pff", "skillcorner"]},
    "step": {"dead": 3, "other": 6},
    "decoder": {"switch": 12, "dead": 40, "beta": 2.0},
    "lgb": {},
    "drop_features": [],
    "folds": 4,
    "seed": 0,
    "deg_seed": 0,
    "n_jobs": 6,
    "feature_workers": 4,
    "limit": None,
    "reuse_oof": None,
}
OPTIONAL = {
    "split_file": "notes/avo/split.json",
}
NOT_TRAINING = ("decoder", "name", "reuse_oof", "feature_workers", "split_file")
DICT_KEYS = ("data", "step", "decoder")


def resolve(config):
    unknown = set(config) - set(DEFAULT) - set(OPTIONAL)
    if unknown:
        raise KeyError("unknown config keys: %s" % sorted(unknown))
    cfg = copy.deepcopy(DEFAULT)
    for k, v in config.items():
        if k in DICT_KEYS and isinstance(v, dict):
            cfg[k] = dict(cfg[k], **v)
        else:
            cfg[k] = copy.deepcopy(v)
    return cfg


def sha_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def src_of(name):
    return name.split("_")[0]


def group_of(name):
    s, rest = name.split("_", 1)
    if s in ("idsse", "idsse-ctrl"):
        return "dfl_" + rest
    if s in ("pff", "pffsb"):
        return "pff_" + rest
    return name


def all_names():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(EXT, "*.npz")))


def canon_index():
    names = all_names()
    canon = [n for n in names if src_of(n) in CANON_SOURCES]
    canon += [n for n in names if src_of(n) not in CANON_SOURCES]
    tune = [n for n in names if src_of(n) in TUNE_SOURCES]
    return {n: i for i, n in enumerate(canon)}, {n: j for j, n in enumerate(tune)}


def stable_index(name):
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def seed_fns(deg_seed=0):
    ci, ti = canon_index()

    def train_seed(n):
        return 7 * (ci[n] if src_of(n) in CANON_SOURCES else stable_index(n)) + deg_seed

    def eval_seed(n):
        return 1000 + (ti[n] if src_of(n) in TUNE_SOURCES else stable_index(n)) % 4 + 4 * deg_seed
    return train_seed, eval_seed


def split_path(path=None):
    if path is None:
        return SPLIT_PATH
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_split(path=None):
    p = split_path(path)
    s = json.load(open(p))
    return s, sha_file(p)[:16]


def apply_profile(s):
    prof = s.get("deg_profile")
    if not prof:
        return None
    for k in ("MTAH_DEG_PROFILE", "MTAH_DEG_PROFILE_EVAL"):
        if os.environ.get(k) not in (None, "", prof):
            raise RuntimeError("split fixes deg_profile=%s but %s=%s" % (prof, k, os.environ[k]))
        os.environ[k] = prof
    return prof


def label_key(name, s):
    return (s.get("label_key") or {}).get(src_of(name), "label")


def masked_sources(s):
    from mtah.train_ext import BALL_MASKED_SOURCES
    return set(s.get("masked_ball_sources") or []) | set(BALL_MASKED_SOURCES)


def swap_label(y):
    out = y.copy()
    out[y == 0] = 1
    out[y == 1] = 0
    return out


def round_robin(names, n):
    by = {}
    for x in sorted(names):
        by.setdefault(src_of(x), []).append(x)
    out, i = [], 0
    while len(out) < min(n, len(names)):
        for s in sorted(by):
            if i < len(by[s]) and len(out) < n:
                out.append(by[s][i])
        i += 1
    return sorted(out)


def spell_f1(pred, gt, half):
    import possession_metrics as pm
    runs = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        runs.append((int(idx[0]), int(idx[-1]) + 1))
    gt = gt.astype(np.int64).copy()
    gt[half == 0] = 3
    pred = pm.fill_abstentions(pred.astype(np.int64), runs)
    return float(pm.spell_scores(gt, pred, runs)["mean_f1"])


def macro_f1(pred, gt, half):
    import possession_metrics as pm
    gt = gt.astype(np.int64).copy()
    gt[half == 0] = 3
    return float(pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred.astype(np.int64)))))


def truth(name, key="label"):
    z = np.load(os.path.join(EXT, name + ".npz"))
    return z[key], z["half"]


def second_name(name, s):
    other = (s.get("second_labeller") or {}).get(src_of(name))
    return None if other is None else other + "_" + name.split("_", 1)[1]


TRAIN_KEYS = ("dev", "sealed", "exclude", "label_key", "masked_ball_sources", "deg_profile")


def score_rule_of(name, s):
    return (s.get("score_rule") or {}).get(src_of(name))


def rule_spell_f1(name, pred, y, half, rule):
    import possession_metrics as pm
    keep = half > 0
    if rule.get("mask"):
        m = rule["mask"]
        keep = keep & ~np.asarray(importlib.import_module(m["module"]).score_mask(name, half, float(m["sec"])), bool)
    p = pred.astype(np.int64).copy()
    if rule.get("oracle_dead"):
        keep = keep & (y != 2)
        p[p == 2] = 3
    gt = y.astype(np.int64).copy()
    gt[~keep] = 3
    edges = np.diff(np.concatenate(([0], keep.astype(np.int8), [0])))
    runs = list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
    p = pm.fill_abstentions(p, runs)
    return float(pm.spell_scores(gt, p, runs)["mean_f1"])


def train_equivalent_shas(s):
    te = s.get("train_equivalent")
    if not te:
        return set()
    base, sha16 = load_split(te["file"])
    if sha_file(split_path(te["file"]))[:len(te["sha256"])] != te["sha256"]:
        raise RuntimeError("train_equivalent %s changed (sha)" % te["file"])
    diff = [k for k in TRAIN_KEYS if base.get(k) != s.get(k)]
    if diff:
        raise RuntimeError("train_equivalent %s differs in training keys %s" % (te["file"], diff))
    return {sha16}


def score(name, pred, s):
    y, half = truth(name, label_key(name, s))
    rule = score_rule_of(name, s)
    out = {"spell_f1": spell_f1(pred, y, half), "macro_f1": macro_f1(pred, y, half)}
    if rule:
        out = {"spell_f1": rule_spell_f1(name, pred, y, half, rule), "macro_f1": out["macro_f1"],
               "spell_f1_full": out["spell_f1"]}
    n2 = second_name(name, s)
    if n2 is not None:
        y2, half2 = truth(n2)
        if not np.array_equal(half2, half):
            raise RuntimeError("second labeller %s is not frame-aligned with %s" % (n2, name))
        out.update(spell_f1_2nd=spell_f1(pred, y2, half2), macro_f1_2nd=macro_f1(pred, y2, half2))
        if rule:
            out.update(spell_f1_2nd_full=out["spell_f1_2nd"], spell_f1_2nd=rule_spell_f1(name, pred, y2, half2, rule))
    return out


def check_masked(name, X, names, s):
    if src_of(name) in masked_sources(s) and not np.isnan(X[:, names.index("b_x")]).all():
        raise RuntimeError("%s: ball is not masked in its feature view (stale cache?)" % name)


def git_head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def peak_gb():
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    return round(s, 2), round(c, 2)


def plan(cfg, split):
    s, split_sha = load_split(cfg.get("split_file"))
    dev, sealed = list(s["dev"]), list(s["sealed"])
    exclude = set(s.get("exclude") or [])
    lim = cfg["limit"] or {}
    if lim.get("dev"):
        dev = round_robin(dev, lim["dev"])
    if lim.get("sealed"):
        sealed_eval = round_robin(sealed, lim["sealed"])
    else:
        sealed_eval = sealed
    sealed_groups = {group_of(n) for n in s["sealed"]}
    srcs = set(cfg["data"]["dead"]) | set(cfg["data"]["team"])
    names = all_names()
    if split == "dev":
        evalset = sorted(dev, key=lambda n: (src_of(n), n))
        K = max(1, min(int(cfg["folds"]), len(evalset)))
        folds = [[n for r, n in enumerate(evalset) if r % K == k] for k in range(K)]
    elif split == "sealed":
        folds = [sorted(sealed_eval)]
    else:
        raise ValueError(split)
    eval_names = sorted(n for f in folds for n in f)
    pool = [n for n in names if src_of(n) in srcs and group_of(n) not in sealed_groups and n not in exclude]
    if lim.get("train_per_source"):
        keep, cnt = [], {}
        for n in pool:
            if n in eval_names:
                keep.append(n)
            elif cnt.get(src_of(n), 0) < lim["train_per_source"]:
                cnt[src_of(n)] = cnt.get(src_of(n), 0) + 1
                keep.append(n)
        pool = keep
    assert all(group_of(n) not in sealed_groups for n in pool), "sealed match leaked into training"
    assert not exclude & set(eval_names), "excluded match in the eval set"
    return {"folds": folds, "pool": pool, "split_sha": split_sha, "eval": eval_names, "split": s}


def evaluate(config, split="dev", save_oof=None, verbose=True):
    t0 = time.time()
    cfg = resolve(config)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    os.environ["MTAH_BALL"] = cfg["ball"]
    if "mtah.match" in sys.modules and sys.modules["mtah.match"].BALL != cfg["ball"]:
        raise RuntimeError("mtah.match already imported with MTAH_BALL=%s" % sys.modules["mtah.match"].BALL)
    if os.path.join(ROOT, "evaluation") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    pipe = importlib.import_module(cfg["pipeline"])
    from mtah.featcache import feature_paths, load_view, code_hash

    P = plan(cfg, split)
    folds, pool, S = P["folds"], P["pool"], P["split"]
    deg_profile = apply_profile(S)
    train_seed, eval_seed = seed_fns(cfg["deg_seed"])
    dead_set = {n for n in pool if src_of(n) in cfg["data"]["dead"]}
    team_set = {n for n in pool if src_of(n) in cfg["data"]["team"]}
    timing = {}
    device = os.environ.get("MTAH_LGB_DEVICE", "cpu")
    log("[eval_ext] split=%s folds=%s pool=%d eval=%d" % (split, [len(f) for f in folds], len(pool),
                                                          len(P["eval"])))
    if cfg["reuse_oof"] and split != "dev":
        raise ValueError("reuse_oof is dev-only")

    saved = dict(pipe.PARAMS)
    pipe.PARAMS.update(n_jobs=cfg["n_jobs"], random_state=cfg["seed"], **cfg["lgb"])
    params_used = dict(pipe.PARAMS)
    train_sig = sha_json({k: cfg[k] for k in cfg if k not in NOT_TRAINING})
    try:
        per, n_rows, n_feat = {}, 0, None
        if cfg["reuse_oof"]:
            meta = json.load(open(os.path.join(cfg["reuse_oof"], "meta.json")))
            ok_sha = {P["split_sha"]} | train_equivalent_shas(S)
            if meta["train_sig"] != train_sig or meta["split_sha"] not in ok_sha or meta["split"] != split:
                raise ValueError("reuse_oof was produced with a different training config / split")
            device = meta.get("lgb_device", "unknown")
            t = time.time()
            for k, fold in enumerate(folds):
                for n in fold:
                    z = np.load(os.path.join(cfg["reuse_oof"], "oof_%s.npz" % n))
                    p = {h: z[h].astype(np.float64) for h in z.files if h not in ("label", "half")}
                    y, half = truth(n)
                    pred = pipe.decode_heads(p, half, cfg["decoder"])
                    per[n] = dict(score(n, pred, S), fold=k)
            timing["decode"] = time.time() - t
        else:
            keys = [(n, sw, train_seed(n)) for n in pool for sw in (False, True)]
            keys += [(n, sw, eval_seed(n)) for f in folds for n in f for sw in (False, True)]
            t = time.time()
            feats = feature_paths(keys, workers=cfg["feature_workers"])
            timing["features"] = time.time() - t
            log("[eval_ext] features %d views %.0fs" % (len(keys), timing["features"]))

            t = time.time()
            rows, cols, names = [], None, None
            for n in pool:
                step = cfg["step"]["dead"] if n in dead_set else cfg["step"]["other"]
                lk = label_key(n, S)
                for sw in (False, True):
                    X, y, half, nm = load_view(feats[(n, sw, train_seed(n))])
                    check_masked(n, X, nm, S)
                    if lk != "label":
                        y0, h0 = truth(n)
                        yl, _ = truth(n, lk)
                        if sw:
                            y0, yl = swap_label(y0), swap_label(yl)
                        if not (np.array_equal(y, y0) and np.array_equal(half, h0)):
                            raise RuntimeError("cached labels differ from %s.npz" % n)
                        y = yl
                    if cols is None:
                        names = nm
                        bad = set(cfg["drop_features"]) - set(nm)
                        if bad:
                            raise KeyError("drop_features not in feature set: %s" % sorted(bad))
                        cols = np.array([i for i, x in enumerate(nm) if x not in set(cfg["drop_features"])])
                        n_feat = len(cols)
                    elif nm != names:
                        raise RuntimeError("feature names differ across cached views")
                    bsw, bon, boff = pipe.boundary_labels(y, half)
                    keep = np.flatnonzero((y < 3) & (half > 0))[::step]
                    Xk = X[keep] if len(cols) == X.shape[1] else X[np.ix_(keep, cols)]
                    del X
                    rows.append((n, np.ascontiguousarray(Xk), y[keep], bsw[keep], bon[keep], boff[keep]))
                    n_rows += len(keep)
            timing["load"] = time.time() - t
            log("[eval_ext] %d training rows x %d feats loaded %.0fs, peak %s GB" % (
                n_rows, n_feat, timing["load"], peak_gb()))
            timing["fit"] = timing["predict"] = 0.0
            if save_oof:
                os.makedirs(save_oof, exist_ok=True)
            for k, fold in enumerate(folds):
                held = {group_of(n) for n in fold}
                tr = [r for r in rows if group_of(r[0]) not in held]
                ds = {n for n in dead_set if group_of(n) not in held}
                ts = {n for n in team_set if group_of(n) not in held}
                t = time.time()
                models = pipe.fit(tr, ds, ts)
                timing["fit"] += time.time() - t
                t = time.time()
                for n in fold:
                    Xo, yo, _, nmo = load_view(feats[(n, False, eval_seed(n))])
                    Xs, _, _, nms = load_view(feats[(n, True, eval_seed(n))])
                    check_masked(n, Xo, nmo, S)
                    check_masked(n, Xs, nms, S)
                    if len(cols) != Xo.shape[1]:
                        Xo, Xs = Xo[:, cols], Xs[:, cols]
                    y, half = truth(n)
                    if not np.array_equal(yo, y):
                        raise RuntimeError("cached labels differ from %s.npz" % n)
                    p = pipe.predict(models, X=Xo, Xs=Xs)
                    del Xo, Xs
                    pred = pipe.decode_heads(p, half, cfg["decoder"])
                    per[n] = dict(score(n, pred, S), fold=k)
                    if save_oof:
                        np.savez_compressed(os.path.join(save_oof, "oof_%s.npz" % n),
                                            **{h: np.asarray(v, np.float32) for h, v in p.items()})
                    log("  fold %d %-22s spellF1 %.4f macroF1 %.4f" % (k, n, per[n]["spell_f1"],
                                                                     per[n]["macro_f1"]))
                timing["predict"] += time.time() - t
                del models
            if save_oof:
                json.dump({"train_sig": train_sig, "split_sha": P["split_sha"], "split": split,
                           "lgb_device": device, "config": cfg}, open(os.path.join(save_oof, "meta.json"), "w"), indent=1)
    finally:
        pipe.PARAMS.clear()
        pipe.PARAMS.update(saved)

    sf = np.array([per[n]["spell_f1"] for n in sorted(per)])
    mf = np.array([per[n]["macro_f1"] for n in sorted(per)])
    by = {}
    for n in sorted(per):
        by.setdefault(src_of(n), []).append(n)
    sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    by_source = {s: {"n": len(v), "spell_f1_mean": float(np.mean([per[n]["spell_f1"] for n in v])),
                     "macro_f1_mean": float(np.mean([per[n]["macro_f1"] for n in v])),
                     "spell_f1_sem": sem([per[n]["spell_f1"] for n in v])}
                 for s, v in by.items()}
    for s, v in by.items():
        if all("spell_f1_full" in per[n] for n in v):
            by_source[s]["spell_f1_full_mean"] = float(np.mean([per[n]["spell_f1_full"] for n in v]))
    second = {}
    for s, v in by.items():
        v2 = [n for n in v if "spell_f1_2nd" in per[n]]
        if v2:
            second[s] = {"labeller": (S.get("second_labeller") or {})[s], "n": len(v2),
                         "spell_f1_mean": float(np.mean([per[n]["spell_f1_2nd"] for n in v2])),
                         "spell_f1_sem": sem([per[n]["spell_f1_2nd"] for n in v2]),
                         "macro_f1_mean": float(np.mean([per[n]["macro_f1_2nd"] for n in v2])),
                         "spell_f1_mean_primary": float(np.mean([per[n]["spell_f1"] for n in v2]))}
    extra = {}
    if deg_profile:
        extra["deg_profile"] = deg_profile
    if "pff" in second and second["pff"]["labeller"] == "pffsb":
        extra["pff_vs_sb"] = second["pff"]["spell_f1_mean"]
    if S.get("score_rule"):
        extra["score_rule"] = S["score_rule"]
    ps, pc = peak_gb()
    return {
        "name": cfg["name"], "split": split, "split_sha": P["split_sha"],
        "split_file": os.path.relpath(split_path(cfg.get("split_file")), ROOT),
        "label_key": dict(S.get("label_key") or {}),
        "n": len(per), "spell_f1_mean": float(sf.mean()), "spell_f1_median": float(np.median(sf)),
        "spell_f1_sem": float(sf.std(ddof=1) / np.sqrt(len(sf))) if len(sf) > 1 else 0.0,
        "macro_f1_mean": float(mf.mean()),
        "spell_f1_balanced": float(np.mean([v["spell_f1_mean"] for v in by_source.values()])),
        "spell_f1_balanced_sem": float(np.sqrt(sum(v["spell_f1_sem"] ** 2 for v in by_source.values()))
                                       / len(by_source)),
        "by_source": by_source,
        "second_labeller": second, **extra,
        "per_match": per,
        "runtime_s": round(time.time() - t0, 1), "timing_s": {k: round(v, 1) for k, v in timing.items()},
        "peak_rss_gb": ps, "peak_rss_children_gb": pc,
        "n_train_rows": n_rows, "n_features": n_feat, "pool": P["pool"], "folds": folds,
        "config": cfg, "config_sha": sha_json(cfg), "train_sig": train_sig,
        "lgb_params": {k: v for k, v in params_used.items()},
        "lgb_device": device,
        "feature_hash": code_hash(), "git_head": git_head(),
        "smoke": bool(cfg["limit"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config (keys: see eval_ext.DEFAULT)")
    ap.add_argument("--split", default="dev", choices=["dev", "sealed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-oof", default=None, help="dir for per-match head probabilities (dev)")
    ap.add_argument("--allow-sealed", action="store_true", help="only mtah.avo_gate / humans")
    ap.add_argument("--split-file", default=None,
                    help="split JSON (relative to the repo root); overrides the config's split_file. "
                         "Default: v1 notes/avo/split.json. Evaluator v2: notes/avo/split_v2.json, "
                         "v3: notes/avo/split_v3.json")
    args = ap.parse_args()
    if args.split == "sealed" and not args.allow_sealed:
        sys.exit("sealed results are produced only by mtah.avo_gate (see notes/avo/README.md)")
    if args.split == "sealed" and args.save_oof:
        sys.exit("no OOF saving on sealed")
    cfg = json.load(open(args.config))
    if args.split_file:
        cfg["split_file"] = args.split_file
    r = evaluate(cfg, split=args.split, save_oof=args.save_oof)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(r, open(args.out, "w"), indent=1)
    print("[eval_ext] %s n=%d spellF1 %.4f (median %.4f) balanced %.4f macroF1 %.4f | %s | %.0fs, peak RSS %.2f GB"
          % (args.split, r["n"], r["spell_f1_mean"], r["spell_f1_median"], r["spell_f1_balanced"],
             r["macro_f1_mean"], {s: round(v["spell_f1_mean"], 4) for s, v in r["by_source"].items()},
             r["runtime_s"], r["peak_rss_gb"]))
    if "pff_vs_sb" in r:
        print("[eval_ext] report-only: pff matches vs StatsBomb labels (pffsb) spellF1 %.4f (vs PFF %.4f)"
              % (r["pff_vs_sb"], r["second_labeller"]["pff"]["spell_f1_mean_primary"]))


if __name__ == "__main__":
    main()
