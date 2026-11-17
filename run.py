#!/usr/bin/env python3
"""Regenerate the submission file with one command.

    .venv/bin/python run.py                  # everything: download + convert external data, train, predict, decode
    .venv/bin/python run.py --skip-download  # raw external data already under data_external/<source>/
    .venv/bin/python run.py --from-cache     # external npz (data_external/common) present: train + predict + decode,
                                             # reusing cached feature views (cache/featcache, cache/nn5) if present
    .venv/bin/python run.py --decode-only    # signals file already built: decode + validate (seconds)
    .venv/bin/python run.py --dry-run        # list the steps and what they would do; runs nothing
    .venv/bin/python run.py --allow-drift    # another GPU / driver: intermediate vision / player files that differ from
                                             # the build are logged as DRIFT and the run still writes a CSV (its final
                                             # sha256 check reports MATCH / DIFFERS, exit 1 on DIFFERS)

The recipe (training config, decoder, expected hashes) is read from FINAL.json; nothing else names it.
Inputs: the competition bundle unpacked in data/ (or MTAH_DATA_DIR) and the external datasets
(docs/REPRODUCE.md: sources, access, citations). No ground truth is read at any step.

Steps
  1 check       Python packages, CUDA GPU (BiGRU), provided-match tracking files
  2 download    raw external datasets (src/external/download_all.py; PFF and SoccerTrack v2 need access, see docs)
  3 convert     raw -> data_external/common/<source>_<match>.npz (src/external/convert_*.py, idsse_ctrl_labels.py)
  4 inputs      content hash of every external npz the recipe trains on + the provided ATD vs docs/external_manifest.json
  5 assets      compliant degradation: cache/degprof/ext_rand_gaps.npz (shipped, external data only) -> feature hash
  6 gamestate   game-state member (configs with a mtah.pipe_gs* pipeline): mtah.gs_build ensure -> carrier rows,
                cross-fitted carrier models + all-data model, training-view posteriors (cache/gs/<ver>); no-op when the
                cache is complete
  6b corrector  chunk-diffusion possession corrector inputs (configs with mtah.pipe_gsraw_pdc, ledger_v4 #11):
                mtah.pdc_build ensure -> base game-state caches + dev evaluation-view posteriors, the base's dev OOF
                (mtah.eval_ext --save-oof, 4 deterministic fold fits), dev 5-fps views, corrector packs; each stage a
                no-op when present and matching docs/external_manifest.json "pdc" content hashes
  6c vision     only for a member trained under a refined-ball profile (refD3_ext; merge config R1, E148): the provided
                match's refined video ball cache/vision/ball_video_refD3.npz from the competition video with the shipped
                external-trained weights (.venv-vision: src/vision/detect.py -> candidates, vision.refine.match ->
                ball), content-hash checked against FINAL expected.vision; a no-op when present and matching
  6d players    only with the optional FINAL key "player_fix" (E205): the provided match's corrected player tracking
                (mode "T2team": team fixes from kit colour, keeper rule, referees removed; mode "T2team_refill": + missing
                players refilled near the ball from our tracker, version "tracker", inside the field; mtah.player_fix) from our player
                detections on the competition video (.venv-vision: src/vision/run_players.py detect with the shipped
                external-trained weights), content-hash checked against FINAL player_fix.expected; the members read it at
                MATCH INFERENCE only (final_signals --tracking, after the fit); a no-op when present and matching
                + optional FINAL key "player_sources" (Track P1 F3): further player trackings of the match (mode "OWN":
                players purely from our own video tracker, mtah.player_fix --mode OWN on the tracker file the entry
                names / builds), content-hash checked; each member also predicts the match on each of them (one fit,
                final_signals --tracking NAME=FILE) and merge_final mixes the sources (FINAL key "merge_sources")
  7 signals     mtah.final_signals: feature views (cached per file), LightGBM (CPU, deterministic) + BiGRU (CUDA,
                deterministic) on all external matches of the config, prediction on the provided match
                (refined-ball members: mtah.run_refD final; merge configs: one signals file per member)
  8 decode      mtah.redecode with the FINAL.json decoder -> CSV (merge configs: mtah.merge_final)
  9 validate    evaluation/validate_submission.py
 10 sha256      print and compare with FINAL.json expected.csv_sha256
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
COMMON = os.path.join(ROOT, "data_external", "common")
PY = sys.executable

CONVERT = {
    "idsse-ctrl": (["idsse", "pathcrf", "elastic"],
                   [["src/external/convert_idsse.py"], ["src/external/idsse_ctrl_labels.py"]]),
    "metrica": (["metrica"], [["src/external/convert_metrica.py"]]),
    "skillcorner": (["skillcorner"], [["src/external/convert_skillcorner.py"]]),
    "pff": (["pff"], [["src/external/convert_pff.py", "--jobs", "4"]]),
    "stv2": (["stv2"], [["src/external/convert_stv2.py", "--jobs", "3"]]),
    "skillcorner2020": (["skillcorner2020"], [["src/external/convert_skillcorner2020.py"]]),
}
CONVERT["pffsb"] = (["statsbomb"], [["src/external/statsbomb_pff.py", "--jobs", "4"]])
PDC_PIPELINES = ("mtah.pipe_gsraw_pdc",)
GS_SOURCES = ("idsse-ctrl", "pff", "metrica", "skillcorner", "skillcorner2020")
GS_PIPELINES = ("mtah.pipe_gs", "mtah.pipe_gsraw", "mtah.pipe_gsraw2",
                "mtah.pipe_gsraw_pdc")
MERGE_PIPELINES = ("mtah.merge_final", "mtah.stack_final")
STACK_PIPELINES = ("mtah.stack_final",)
REFINED_BALL = {"refD3_ext": "ball_video_refD3.npz"}
VISION_PY = os.environ.get("MTAH_VISION_PY") or os.path.join(ROOT, ".venv-vision", "bin", "python")
GS_VER = None
RAW_MARKER = {
    "idsse": "data_external/idsse/files.txt",
    "pathcrf": "data_external/pathcrf/repo/data/sportec_synced",
    "elastic": "data_external/elastic/gt",
    "metrica": "data_external/metrica/repo/data/Sample_Game_3",
    "skillcorner": "data_external/skillcorner/commit.txt",
    "stv2": "data_external/raw/soccertrack_v2/raw",
    "skillcorner2020": "data_external/statsbomb/data/events/303596.json",
    "statsbomb": "data_external/statsbomb/wc2022_done.txt",
    "pff": "data_external/pff/raw/Tracking Data",
}
MANUAL = {
    "pff": "PFF FC World Cup 2022: request free access with PFF's form (the Google Drive folder link is in "
           "src/external/download_pff.py); if the folder is not reachable, download it by hand into data_external/pff/raw/",
    "stv2": "SoccerTrack v2 (Hugging Face atomscott/soccertrack-v2, tag v1.2): log in with `huggingface-cli login` "
            "if the dataset asks for it",
}


def log(msg):
    print("[run] %s" % msg, flush=True)


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = SRC + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e["MTAH_LGB_DEVICE"] = "cpu"
    e.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for k in ("MTAH_DEG_PROFILE", "MTAH_DEG_PROFILE_EVAL", "MTAH_BALL", "MTAH_GS_VER", "MTAH_REFD4_THR"):
        e.pop(k, None)
    if GS_VER:
        e["MTAH_GS_VER"] = GS_VER
    return e


def sh(cmd, dry, isolated=False):
    full = [PY] + (["-I"] if isolated else []) + cmd
    log("$ " + " ".join(full))
    if dry:
        return
    t = time.time()
    subprocess.run(full, cwd=ROOT, env=env(), check=True)
    log("  done in %.0f s" % (time.time() - t))


DRIFTS = []


def drift(a, msg):
    if not getattr(a, "allow_drift", False):
        raise SystemExit(msg)
    DRIFTS.append(msg)
    log("!" * 100)
    log("DRIFT (--allow-drift, continuing): %s" % msg)
    log("!" * 100)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def content_hash(path, keys=None):
    import numpy as np
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(keys or z.files):
            a = np.ascontiguousarray(z[k])
            h.update(("%s|%s|%s|" % (k, a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def load_final(path):
    F = json.load(open(os.path.join(ROOT, path)))
    cfg = json.load(open(os.path.join(ROOT, F["config"])))
    dec = F.get("decoder") or cfg["decoder"]
    name = cfg["name"]
    F["_cfg"], F["_dec"], F["_name"] = cfg, dec, name
    F["_signals"] = os.path.join(ROOT, F["signals"].format(name=name))
    F["_parts"] = [F]
    if cfg.get("pipeline") in MERGE_PIPELINES:
        F["_parts"] = []
        mem = cfg["members"] if "members" in cfg else {k: v["config"] for k, v in cfg["experts"].items()}
        for key, path in mem.items():
            mc = json.load(open(os.path.join(ROOT, path)))
            F["_parts"].append({"config": path, "_cfg": mc, "_dec": mc["decoder"], "_name": mc["name"], "_key": key,
                                "_signals": os.path.join(ROOT, F["signals"].format(name=mc["name"])),
                                "expected": F["expected"]})
    return F


def profile_of(P):
    sys.path.insert(0, SRC)
    from mtah import eval_ext as E
    return E.load_split(P["_cfg"].get("split_file"))[0].get("deg_profile")


def refined_ball(P):
    return REFINED_BALL.get(profile_of(P))


def pool_names(F):
    sys.path.insert(0, SRC)
    from mtah import eval_ext as E
    cfg = E.resolve(dict(F["_cfg"]))
    S, _ = E.load_split(cfg.get("split_file"))
    exclude = set(S.get("exclude") or [])
    srcs = set(cfg["data"]["dead"]) | set(cfg["data"]["team"])
    return sorted(srcs), [n for n in E.all_names() if E.src_of(n) in srcs and n not in exclude]


def uses_gs(F):
    return any(P["_cfg"].get("pipeline") in GS_PIPELINES for P in F.get("_parts", [F]))


def sources_of(F):
    out = set()
    for P in F.get("_parts", [F]):
        c = P["_cfg"]
        out |= (set(c["data"]["dead"]) | set(c["data"]["team"]) | (set(GS_SOURCES) if uses_gs(P) else set())
                | ({"pffsb"} if c.get("pipeline") in PDC_PIPELINES else set()))
    return sorted(out)


def manifest_path(F):
    return os.path.join(ROOT, F["expected"]["external_manifest"])


def step_check(F, a):
    log("recipe: %s (config %s), decoder %s" % (F["_name"], F["config"], json.dumps(F["_dec"])))
    log("python %s" % sys.version.split()[0])
    if a.dry_run:
        log("would check: numpy/pandas/scipy/lightgbm/torch importable, torch.cuda.is_available(), "
            "data/tactical/BAR-ATH_tactical_atd.{txt,xml}")
        return
    bad = []
    mods = ["numpy", "pandas", "scipy", "lightgbm", "torch"] + ([] if a.from_cache else ["pyarrow", "kloppy"])
    for m in mods:
        try:
            mod = __import__(m)
            log("  %-9s %s" % (m, getattr(mod, "__version__", "?")))
        except Exception as e:   # noqa: BLE001
            bad.append("%s (%s)" % (m, e))
    if bad:
        raise SystemExit("missing packages: %s -> pip install -r requirements-final.lock.txt" % ", ".join(bad))
    import torch
    if a.decode_only:
        pass
    elif not torch.cuda.is_available():
        raise SystemExit("the BiGRU member trains on CUDA (no CPU path): a CUDA GPU with >= 2 GB free is required")
    else:
        log("  cuda      %s, torch cuda %s, cudnn %s" % (torch.cuda.get_device_name(0), torch.version.cuda,
                                                        torch.backends.cudnn.version()))
    sys.path.insert(0, ROOT)
    from games import asset_paths
    p = asset_paths(source="tactical")
    miss = [p[k] for k in ("atd_meta", "atd_raw") if not os.path.isfile(p[k])]
    if miss and not os.path.exists(os.path.join(ROOT, "cache", "atd_tactical.npz")):
        raise SystemExit("provided match tracking missing: %s (unpack the bundle into data/ or set MTAH_DATA_DIR)"
                         % miss)
    try:
        import psutil
        gb = psutil.virtual_memory().available / 2 ** 30
    except ImportError:
        gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    log("  RAM available %.0f GB (the signals step peaks at ~22 GB with the game-state member (#9), ~12 GB without)"
        % gb)


def step_download(F, a):
    if a.skip_download or a.from_cache:
        log("skipped (%s)" % ("--from-cache" if a.from_cache else "--skip-download"))
        return
    todo = []
    for s in sources_of(F):
        for d in CONVERT[s][0]:
            if d not in todo and not os.path.exists(os.path.join(ROOT, RAW_MARKER[d])):
                todo.append(d)
    if a.dry_run:
        for s in sources_of(F):
            for d in CONVERT[s][0]:
                log("  %-11s raw %-48s %s" % (d, RAW_MARKER[d], "MISSING -> download" if d in todo else "present"))
    if not todo:
        log("all raw sources present")
        return
    for d in todo:
        if d in MANUAL:
            log("NOTE %s: %s" % (d, MANUAL[d]))
    sh(["src/external/download_all.py"] + todo, a.dry_run, isolated=True)


def step_convert(F, a):
    if a.from_cache:
        log("skipped (--from-cache): data_external/common/*.npz must exist")
        return
    try:
        M = json.load(open(manifest_path(F)))
        pool = M["pool"] + [n for n in M.get("gs_train", []) if n not in M["pool"]]
        pool += ["pffsb_" + n.split("_", 1)[1] for n in M["pool"] if n.startswith("pff_")]
    except (OSError, KeyError, ValueError):
        _, pool = pool_names(F)
    have = {p[:-4] for p in os.listdir(COMMON) if p.endswith(".npz")} if os.path.isdir(COMMON) else set()
    for s in sources_of(F):
        need = [n for n in pool if n.startswith(s + "_")]
        if need and all(n in have for n in need) and not a.reconvert:
            log("%s: %d npz present%s" % (s, len(need), (" (else: %s)" % " ; ".join(
                "python -I " + " ".join(c) for c in CONVERT[s][1])) if a.dry_run else ""))
            continue
        for cmd in CONVERT[s][1]:
            sh(cmd, a.dry_run, isolated=True)


def step_inputs(F, a):
    mp = manifest_path(F)
    if a.dry_run:
        log("would hash the %s external npz + provided ATD and compare with %s" % ("pool's", mp))
        return
    srcs, pool = pool_names(F["_parts"][0])
    M = json.load(open(mp))
    for P in F["_parts"][1:]:
        if pool_names(P)[1] != pool:
            raise SystemExit("merge members train on different pools (%s vs %s): not covered by one manifest"
                             % (P["_name"], F["_parts"][0]["_name"]))
    sys.path.insert(0, SRC)
    from mtah import eval_ext as E
    canon = [n for n in E.all_names() if E.src_of(n) in E.CANON_SOURCES]
    if canon != M["canon_names"]:
        raise SystemExit("data_external/common holds a different set of %s files than the build (%d vs %d): the "
                         "degradation seeds would move" % (E.CANON_SOURCES, len(canon), len(M["canon_names"])))
    miss = [n for n in M["pool"] if n not in pool]
    if miss or pool != M["pool"]:
        raise SystemExit("training pool differs from the build: missing %s, extra %s"
                         % (miss[:5], [n for n in pool if n not in M["pool"]][:5]))
    gs_extra = []
    if uses_gs(F):
        from mtah import gs_build
        have = gs_build.ball_names()
        if sorted(have) != sorted(M.get("gs_train", [])):
            raise SystemExit("game-state training matches differ from the build: missing %s, extra %s"
                             % (sorted(set(M.get("gs_train", [])) - set(have))[:5],
                                sorted(set(have) - set(M.get("gs_train", [])))[:5]))
        gs_extra = [n for n in have if n not in pool]
    if a.no_verify_inputs:
        log("content hashes NOT checked (--no-verify-inputs)")
        return
    t, bad = time.time(), []
    for n in pool + gs_extra:
        h = content_hash(os.path.join(COMMON, n + ".npz"))
        if h != (M["npz"].get(n) or M.get("gs_npz", {}).get(n)):
            bad.append(n)
    log("%d external npz hashed in %.0f s, %d differ" % (len(pool) + len(gs_extra), time.time() - t, len(bad)))
    from mtah.data import load_tracking
    load_tracking("tactical")
    h = content_hash(os.path.join(ROOT, "cache", "atd_tactical.npz"))
    if h != M["provided_atd_tactical"]:
        bad.append("provided ATD (cache/atd_tactical.npz)")
    if bad:
        raise SystemExit("inputs differ from the build (not bit-reproducible): %s ... (--no-verify-inputs to go on)"
                         % bad[:8])
    log("provided ATD matches the build")


def step_assets(F, a):
    if F["_cfg"].get("pipeline") in MERGE_PIPELINES:
        for P in F["_parts"]:
            if refined_ball(P):
                step_assets_refined(P, F, a)
            else:
                step_assets(P, a)
        return
    gp = os.path.join(ROOT, "cache", "degprof", "ext_rand_gaps.npz")
    if a.dry_run:
        log("would check %s (shipped; external natural ball-gap pool) and featcache.code_hash('ext_rand') == %s"
            % (os.path.relpath(gp, ROOT), expected_hash(F, F)))
        return
    if not os.path.isfile(gp):
        raise SystemExit("%s missing: it ships with the repository (git checkout -- cache/degprof/ext_rand_gaps.npz). "
                         "Rebuilding it needs the vision pipeline on SoccerTrack v2 video (docs/REPRODUCE.md)" % gp)
    sys.path.insert(0, SRC)
    from mtah.featcache import code_hash
    h = code_hash("ext_rand")
    if h != expected_hash(F, F):
        raise SystemExit("feature hash %s != expected %s: feature / degradation code or the gap pool changed"
                         % (h, expected_hash(F, F)))
    d = os.path.join(ROOT, "cache", "featcache", "%s_%s_p-ext_rand" % (h, F["_cfg"].get("ball", "fused")))
    n = len(os.listdir(d)) if os.path.isdir(d) else 0
    log("feature hash %s OK; %d cached feature views in %s (missing ones are computed in the next step)"
        % (h, n, os.path.relpath(d, ROOT)))


def expected_hash(F, P):
    h = F["expected"]["feature_hash"]
    return h[P["_name"]] if isinstance(h, dict) else h


def step_assets_refined(P, F, a):
    prof = profile_of(P)
    sp = os.path.join(ROOT, "cache", "degprof", prof + "_sequences.npz")
    exp = expected_hash(F, P)
    if a.dry_run:
        log("%s: would check %s (shipped; refiner errors on held-out SoccerTrack v2 video) and code_hash('%s') == %s"
            % (P["_name"], os.path.relpath(sp, ROOT), prof, exp))
        return
    if not os.path.isfile(sp):
        raise SystemExit("%s missing: it ships with the repository (git checkout -- %s)" % (sp, os.path.relpath(sp, ROOT)))
    sys.path.insert(0, SRC)
    from mtah.degrade_refD4gap_ext import register
    register()
    from mtah.featcache import code_hash
    h = code_hash(prof)
    if h != exp:
        raise SystemExit("%s: feature hash %s != expected %s (feature / degradation code or %s changed)"
                         % (P["_name"], h, exp, os.path.relpath(sp, ROOT)))
    d = os.path.join(ROOT, "cache", "featcache", "%s_%s_p-%s" % (h, P["_cfg"].get("ball", "fused"), prof))
    n = len(os.listdir(d)) if os.path.isdir(d) else 0
    log("%s: feature hash %s OK; %d cached feature views in %s" % (P["_name"], h, n, os.path.relpath(d, ROOT)))


def step_gamestate(F, a):
    if F["_cfg"].get("pipeline") in MERGE_PIPELINES:
        for P in F["_parts"]:
            log("member %s:" % P["_name"])
            step_gamestate(P, a)
        return
    if not uses_gs(F):
        log("skipped (pipeline %s has no game-state member)" % F["_cfg"].get("pipeline"))
        return
    if a.decode_only:
        log("skipped (--decode-only)")
        return
    if F["_cfg"].get("pipeline") == "mtah.pipe_gsraw_pdc":
        log("covered by step 'corrector' (mtah.pdc_build runs gs_build.ensure on the base config)")
        return
    sh(["-m", "mtah.gs_build", "ensure", "--config", F["config"], "--workers", "6",
        "--expect-train", os.path.relpath(manifest_path(F), ROOT)], a.dry_run)


def step_corrector(F, a):
    if F["_cfg"].get("pipeline") in MERGE_PIPELINES:
        for P in F["_parts"]:
            log("member %s:" % P["_name"])
            step_corrector(P, a)
        return
    if F["_cfg"].get("pipeline") != "mtah.pipe_gsraw_pdc":
        log("skipped (pipeline %s has no corrector)" % F["_cfg"].get("pipeline"))
        return
    if a.decode_only:
        log("skipped (--decode-only)")
        return
    sh(["-m", "mtah.pdc_build", "ensure", "--config", F["config"], "--workers", "6"], a.dry_run)


def ensure_homography():
    import numpy as np
    import pandas as pd
    p = os.path.join(ROOT, "cache", "homography_tactical.npz")
    if os.path.exists(p):
        return
    sys.path.insert(0, ROOT)
    from games import asset_paths
    csv = asset_paths(source="tactical")["homography"]
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    t = pd.read_csv(csv, usecols=["frame", "reprojection_error"] + cols)
    assert np.array_equal(t["frame"].to_numpy(), np.arange(len(t)))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, H=t[cols].to_numpy(np.float64).astype(np.float32), err=t["reprojection_error"].to_numpy(np.float32))
    log("wrote %s from %s" % (os.path.relpath(p, ROOT), csv))


def step_vision(F, a):
    parts = [P for P in F["_parts"] if refined_ball(P)]
    tag_vid = bool((tag_spec(F) or {}).get("video_ball"))
    if not parts and not tag_vid:
        log("skipped (no member predicts on a refined video ball)")
        return
    if a.decode_only:
        log("skipped (--decode-only)")
        return
    V = F["expected"]["vision"]
    for fn in sorted({refined_ball(P) for P in parts} | (set(V) if tag_vid else set())):
        v = V[fn]
        out = os.path.join(ROOT, "cache", "vision", fn)
        if a.dry_run:
            log("%s: if missing or different: homography cache, then (%s)" % (fn, os.path.relpath(VISION_PY, ROOT)))
        elif os.path.isfile(out) and content_hash(out, v.get("content_keys")) == v["content_sha256"]:
            log("%s present and matching (content %s...)" % (os.path.relpath(out, ROOT), v["content_sha256"][:12]))
            continue
        for w in (v["detector"], v["refiner"]):
            if not a.dry_run and sha256(os.path.join(ROOT, w["path"])) != w["sha256"]:
                raise SystemExit("%s missing or different: it ships with the repository" % w["path"])
        if not a.dry_run:
            if not os.path.isfile(VISION_PY):
                raise SystemExit("the vision step needs the vision environment: python3.12 -m venv .venv-vision && "
                                 ".venv-vision/bin/pip install -r requirements-vision.lock.txt (or MTAH_VISION_PY)")
            ensure_homography()
        for cmd in (v["detect_cmd"], v["refine_cmd"]):
            full = [VISION_PY, "-W", "ignore"] + cmd
            log("$ " + " ".join(full))
            if not a.dry_run:
                t = time.time()
                subprocess.run(full, cwd=ROOT, env=env(), check=True)
                log("  done in %.0f s" % (time.time() - t))
        if not a.dry_run:
            h = content_hash(out, v.get("content_keys"))
            if h != v["content_sha256"]:
                drift(a, "%s content %s != expected %s (not bit-reproducible: GPU / driver / library versions?)"
                      % (out, h[:12], v["content_sha256"][:12]))
            else:
                log("%s content hash matches the build" % os.path.relpath(out, ROOT))


def player_fix_spec(F):
    return F.get("player_fix")


def check_player_fix(F, a):
    X = player_fix_spec(F)
    p = os.path.join(ROOT, X["tracking"])
    e = X["expected"]
    if not a.dry_run and not os.path.isfile(p):
        raise SystemExit("player_fix: %s missing (run step 'players')" % p)
    if not a.dry_run and content_hash(p, e.get("content_keys")) != e["content_sha256"]:
        drift(a, "player_fix: %s different from the build" % p)
    return X["tracking"]


def player_sources(F):
    return F.get("player_sources") or []


def check_player_sources(F, a):
    out = []
    for X in player_sources(F):
        p = os.path.join(ROOT, X["tracking"])
        if not a.dry_run and not os.path.isfile(p):
            raise SystemExit("player source %s: %s missing (run step 'players')" % (X["name"], p))
        if not a.dry_run and content_hash(p) != X["expected"]["content_sha256"]:
            drift(a, "player source %s: %s different from the build" % (X["name"], p))
        out.append("%s=%s" % (X["name"], X["tracking"]))
    return out


def step_player_sources(F, a):
    for X in player_sources(F):
        out = os.path.join(ROOT, X["tracking"])
        e, tr = X["expected"], X["tracker"]
        if a.dry_run:
            log("player source %s (mode %s): if %s is missing or different: tracker file %s%s, then mtah.player_fix "
                "build --mode %s" % (X["name"], X["mode"], X["tracking"], tr["file"],
                                     (" from " + " ".join(tr["build_cmd"])) if tr.get("build_cmd") else "", X["mode"]))
            continue
        if os.path.isfile(out) and content_hash(out) == e["content_sha256"]:
            log("player source %s: %s present and matching (content %s...)" % (X["name"], X["tracking"],
                                                                                e["content_sha256"][:12]))
            continue
        tf = os.path.join(ROOT, tr["file"])
        te = tr["expected"]["content_sha256"]
        if not (os.path.isfile(tf) and content_hash(tf, tr["expected"].get("content_keys")) == te):
            if not tr.get("build_cmd"):
                raise SystemExit("player source %s: tracker file %s missing or different and no build_cmd"
                                 % (X["name"], tr["file"]))
            for w in tr.get("weights") or []:
                if sha256(os.path.join(ROOT, w["path"])) != w["sha256"]:
                    raise SystemExit("%s missing or different: it ships with the repository" % w["path"])
            py = VISION_PY if tr.get("vision_env") else PY
            full = [py, "-W", "ignore"] + tr["build_cmd"]
            log("$ " + " ".join(full))
            t = time.time()
            subprocess.run(full, cwd=ROOT, env=env(), check=True)
            log("  done in %.0f s" % (time.time() - t))
            h = content_hash(tf, tr["expected"].get("content_keys"))
            if h != te:
                drift(a, "%s content %s != expected %s (tracker not bit-reproducible)" % (tf, h[:12], te[:12]))
            else:
                log("player source %s: tracker file %s matches the build" % (X["name"], tr["file"]))
        else:
            log("player source %s: tracker file %s matches the build" % (X["name"], tr["file"]))
        sh(["-W", "ignore", "-m", "mtah.player_fix", "build", "--mode", X["mode"], "--tracker-file", tr["file"],
            "--out", X["tracking"]], False)
        h = content_hash(out)
        if h != e["content_sha256"]:
            drift(a, "%s content %s != expected %s (not bit-reproducible)" % (out, h[:12], e["content_sha256"][:12]))
        else:
            log("%s content hash matches the build" % X["tracking"])


def step_players(F, a):
    X = player_fix_spec(F)
    if not X:
        log("skipped (no player_fix in the recipe)")
        return
    if a.decode_only:
        log("skipped (--decode-only)")
        return
    step_player_fix(F, a)
    step_player_sources(F, a)


def step_player_fix(F, a):
    X = player_fix_spec(F)
    out = os.path.join(ROOT, X["tracking"])
    e = X["expected"]
    if a.dry_run:
        log("player_fix %s: if %s is missing or different: homography cache, (%s) %s, then mtah.player_fix build"
            % (X["mode"], X["tracking"], os.path.relpath(VISION_PY, ROOT), " ".join(X["detect_cmd"])))
        return
    if os.path.isfile(out) and content_hash(out, e.get("content_keys")) == e["content_sha256"]:
        log("%s present and matching (content %s...)" % (X["tracking"], e["content_sha256"][:12]))
        return
    w = X["detector"]
    if sha256(os.path.join(ROOT, w["path"])) != w["sha256"]:
        raise SystemExit("%s missing or different: it ships with the repository" % w["path"])
    if not os.path.isfile(VISION_PY):
        raise SystemExit("the players step needs the vision environment: python3.12 -m venv .venv-vision && "
                         ".venv-vision/bin/pip install -r requirements-vision.lock.txt (or MTAH_VISION_PY)")
    ensure_homography()
    sys.path.insert(0, SRC)
    from mtah.data import load_tracking
    load_tracking("tactical")
    full = [VISION_PY, "-W", "ignore"] + X["detect_cmd"]
    log("$ " + " ".join(full))
    t = time.time()
    subprocess.run(full, cwd=ROOT, env=env(), check=True)
    log("  done in %.0f s" % (time.time() - t))
    from mtah.player_fix import cands_hash
    ch = cands_hash(os.path.join(ROOT, X["cands_dir"]))
    log("player detections content %s (build %s): %s" % (ch[:12], e["cands_sha256"][:12],
                                                          "match" if ch == e["cands_sha256"] else "DIFFER"))
    sh(["-W", "ignore", "-m", "mtah.player_fix", "build", "--mode", X["mode"], "--out", X["tracking"],
        "--cands", X["cands_dir"]] + (["--tracker", X["tracker"]] if X.get("tracker") else []), False)
    h = content_hash(out, e.get("content_keys"))
    if h != e["content_sha256"]:
        drift(a, "%s content %s != expected %s (not bit-reproducible)" % (out, h[:12], e["content_sha256"][:12]))
    else:
        log("%s content hash matches the build" % X["tracking"])


def step_signals(F, a):
    trk = check_player_fix(F, a) if (player_fix_spec(F) and not a.decode_only) else None
    extra = check_player_sources(F, a) if (player_sources(F) and not a.decode_only) else []
    for P in F["_parts"]:
        out = P["_signals"]
        if a.decode_only:
            if not os.path.isfile(out):
                raise SystemExit("--decode-only: %s does not exist" % out)
            log("reusing %s" % os.path.relpath(out, ROOT))
            continue
        mod = "mtah.run_refD" if refined_ball(P) else "mtah.final_signals"
        sh(["-m", mod] + (["final"] if refined_ball(P) else []) +
           ["--config", P["config"], "--out", os.path.relpath(out, ROOT)] + (["--tracking", trk] if trk else []) +
           [x for e in extra for x in ("--tracking", e)], a.dry_run)


def oof_hashes(d, names):
    return {n: content_hash(os.path.join(d, "oof_%s.npz" % n)) for n in names}


def step_oof(F, a):
    if F["_cfg"].get("pipeline") not in STACK_PIPELINES:
        log("skipped (no stacked experts)")
        return
    if a.decode_only:
        log("skipped (--decode-only)")
        return
    sys.path.insert(0, SRC)
    cfg = F["_cfg"]
    for e, x in cfg["experts"].items():
        d = os.path.join(ROOT, x["oof"])
        exp = F["expected"]["oof"][e]
        names = sorted(exp)
        if a.dry_run:
            log("%s: if %s is missing or differs from the %d expected hashes: %s" % (e, x["oof"], len(names),
                                                                                       " ".join(x["oof_cmd"])))
            continue
        if all(os.path.isfile(os.path.join(d, "oof_%s.npz" % n)) for n in names) and oof_hashes(d, names) == exp:
            log("%s: %s present and matching (%d files)" % (e, x["oof"], len(names)))
            continue
        cmd = [c.format(oof=x["oof"], oof_json=x["oof"].rstrip("/").replace("_oof", "") + ".json",
                        base_json=x["oof"].rstrip("/") + "_base.json") for c in x["oof_cmd"]]
        if "{base_json}" in " ".join(x["oof_cmd"]):
            from mtah import eval_ext as E
            bc = json.load(open(os.path.join(ROOT, x["base_config"])))
            P = E.plan(E.resolve(dict(bc)), "dev")
            bj = os.path.join(ROOT, x["oof"].rstrip("/") + "_base.json")
            os.makedirs(os.path.dirname(bj), exist_ok=True)
            json.dump({"name": bc["name"], "split_file": bc["split_file"], "folds": P["folds"], "config": bc,
                       "spell_f1_balanced": float("nan"), "runtime_s": 0.0,
                       "note": "fold list of the base config (eval_ext.plan), written by run.py step 'oof'"},
                      open(bj, "w"), indent=1)
        sh(cmd, a.dry_run)
        h = oof_hashes(d, names)
        bad = [n for n in names if h[n] != exp[n]]
        if bad:
            raise SystemExit("%s: rebuilt OOF differs from the build in %d of %d files (%s ...)"
                             % (e, len(bad), len(names), bad[:3]))
        log("%s: rebuilt OOF matches the build (%d files)" % (e, len(names)))


def tag_spec(F):
    t = F.get("tag_layer") or F["_cfg"].get("tag_layer")
    if not t:
        return None
    return {"variant": t, "video_ball": False} if isinstance(t, str) else t


def check_tag_inputs(F, a):
    t = tag_spec(F)
    if not t:
        return
    sys.path.insert(0, ROOT)
    from games import asset_paths
    E = F["expected"]["tag_layer"]
    files = {"smart_tagging_sha256": asset_paths(source="tactical")["smart_tagging"],
             "timing_sha256": os.path.join(ROOT, "notes", "trackG_metrica_timing.json")}
    if a.dry_run:
        log("tag layer %s: would check %s%s" % (json.dumps(t), ", ".join(os.path.basename(p) for p in files.values()),
                                               " + the refined video ball" if t.get("video_ball") else ""))
        return
    for k, p in files.items():
        if not os.path.isfile(p) or sha256(p) != E[k]:
            raise SystemExit("tag layer input %s missing or different from the build (%s)" % (p, k))
    if t.get("video_ball"):
        for fn, v in F["expected"]["vision"].items():
            p = os.path.join(ROOT, "cache", "vision", fn)
            if not os.path.isfile(p):
                raise SystemExit("tag layer: %s missing (run step 'vision')" % p)
            if content_hash(p, v.get("content_keys")) != v["content_sha256"]:
                drift(a, "tag layer: %s different from the build" % p)
    log("tag layer %s: Smart Tagging XML, timing constants%s match the build"
        % (json.dumps(t), " and the refined video ball" if t.get("video_ball") else ""))


def step_decode(F, a):
    check_tag_inputs(F, a)
    if F["_cfg"].get("pipeline") in STACK_PIPELINES:
        sh(["-m", "mtah.stack_final", "--config", F["config"], "--signals", F["signals"]] +
           (["--tag-layer", json.dumps(tag_spec(F))] if tag_spec(F) else []) + ["--out", a.out], a.dry_run)
        return
    if F["_cfg"].get("pipeline") in MERGE_PIPELINES:
        sh(["-m", "mtah.merge_final", "--config", F["config"], "--signals", F["signals"],
            "--decoder", json.dumps(F["_dec"])] +
           (["--tag-layer", json.dumps(tag_spec(F))] if tag_spec(F) else []) +
           (["--sources", json.dumps(F["merge_sources"])] if F.get("merge_sources") else []) + ["--out", a.out], a.dry_run)
        return
    pipe = F["_cfg"].get("pipeline", "mtah.train_v5")
    sh(["-m", "mtah.redecode", "--signals", os.path.relpath(F["_signals"], ROOT), "--pipeline", pipe,
        "--decoder", json.dumps(F["_dec"])] +
       (["--tag-layer", json.dumps(tag_spec(F))] if tag_spec(F) else []) + ["--out", a.out], a.dry_run)


def step_validate(F, a):
    sh(["evaluation/validate_submission.py", a.out], a.dry_run)


def step_sha(F, a):
    exp = F["expected"]["csv_sha256"]
    if a.dry_run:
        log("would print sha256(%s) and compare with %s..." % (a.out, exp[:16]))
        return 0
    h = sha256(os.path.join(ROOT, a.out))
    log("sha256 %s  %s" % (h, a.out))
    if h == exp:
        log("MATCHES FINAL.json expected.csv_sha256: bit-exact reproduction")
        return 0
    log("DIFFERS from FINAL.json expected.csv_sha256 %s" % exp)
    return 1


STEPS = [("check", step_check), ("download", step_download), ("convert", step_convert), ("inputs", step_inputs),
         ("assets", step_assets), ("gamestate", step_gamestate), ("corrector", step_corrector), ("vision", step_vision),
         ("players", step_players), ("oof", step_oof), ("signals", step_signals), ("decode", step_decode), ("validate", step_validate),
         ("sha256", step_sha)]


def gs_names():
    sys.path.insert(0, SRC)
    from mtah import gs_build
    return gs_build.ball_names()


def write_manifest(F):
    sys.path.insert(0, SRC)
    from mtah import eval_ext as E
    srcs, pool = pool_names(F)
    M = {"what": "content hashes (run.content_hash: arrays only) of the inputs the submission was built from; "
                 "checked by run.py step 'inputs'. Hashes only - no data.",
         "config": F["config"], "sources": srcs,
         "canon_names": [n for n in E.all_names() if E.src_of(n) in E.CANON_SOURCES],
         "pool": pool, "npz": {n: content_hash(os.path.join(COMMON, n + ".npz")) for n in pool},
         **({"gs_train": sorted(gs_names()), "gs_npz": {n: content_hash(os.path.join(COMMON, n + ".npz"))
                                                         for n in sorted(gs_names()) if n not in pool}}
            if uses_gs(F) else {}),
         "ext_rand_gaps": content_hash(os.path.join(ROOT, "cache", "degprof", "ext_rand_gaps.npz")),
         "provided_atd_tactical": content_hash(os.path.join(ROOT, "cache", "atd_tactical.npz"))}
    p = manifest_path(F)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p) and "pdc" in json.load(open(p)):
        M["pdc"] = json.load(open(p))["pdc"]
    json.dump(M, open(p, "w"), indent=1)
    log("wrote %s (%d npz)" % (p, len(pool)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", default="FINAL.json", help="recipe file (default FINAL.json)")
    ap.add_argument("--out", default=None, help="CSV path (default: FINAL.json output)")
    ap.add_argument("--dry-run", action="store_true", help="list the steps and commands, run nothing")
    ap.add_argument("--skip-download", action="store_true", help="raw external data already present")
    ap.add_argument("--from-cache", action="store_true", help="skip download + conversion (npz present)")
    ap.add_argument("--decode-only", action="store_true", help="reuse the signals file (implies --from-cache)")
    ap.add_argument("--reconvert", action="store_true", help="re-run converters even if the npz exist")
    ap.add_argument("--no-verify-inputs", action="store_true", help="do not hash the external npz")
    ap.add_argument("--only", nargs="*", default=None, help="run only these steps (%s)" % " ".join(s for s, _ in STEPS))
    ap.add_argument("--allow-drift", action="store_true", help="a content-hash mismatch of an intermediate vision / "
                    "player file (other GPU / driver) is logged as DRIFT and the run goes on to a CSV; the final csv "
                    "sha256 check still reports MATCH / DIFFERS (exit 1 on DIFFERS). Default: stop at the mismatch")
    ap.add_argument("--write-manifest", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    F = load_final(a.final)
    global GS_VER
    gp = [P for P in F["_parts"] if uses_gs(P)]
    GS_VER = ((gp[0]["_cfg"].get("lgb") or {}).get("pipe_gs_ver") or "v1") if gp else None
    a.out = a.out or F["output"]
    if a.decode_only:
        a.from_cache = True
    if a.write_manifest:
        return write_manifest(F)
    t0, rc = time.time(), 0
    for i, (name, fn) in enumerate(STEPS, 1):
        if a.only and name not in a.only:
            continue
        if a.decode_only and name in ("inputs", "assets"):
            continue
        log("=== step %d/%d: %s" % (i, len(STEPS), name))
        rc = fn(F, a) or 0
    log("finished in %.0f s%s" % (time.time() - t0, " (dry run)" if a.dry_run else ""))
    for m in DRIFTS:
        log("DRIFT accepted (--allow-drift): %s" % m)
    return rc


if __name__ == "__main__":
    sys.exit(main())
