"""Checkpoints of every trained unit of a final build + the duplicate-build guard (E148, Track S; user request
2026-11-15: no progress may be lost to crashes or stops, and checkpoints must never change a result).

    from mtah import ckpt
    ckpt.guard_duplicate()      # refuse to start when another python process runs the same command (minus --out)
    ckpt.install(pipeline)      # wrap the trainers below; mtah.final_signals calls both

Trained units (each wrapped at runtime; the pipeline files are not edited):
    mtah.pipe_ens2.fit_lgb      LightGBM heads (also the name mtah.pipe_lgb2 imported)
    mtah.pipe_ens2._train_nn    one BiGRU seed (base members, game-state member, players-only nnb member)
    mtah.pipe_ens._train_nn     one BiGRU seed (pipe_ens v1)
    mtah.raw_seq.train_model    one raw-tracking seed (pipe_raw; raw_seq_bt / raw_seq3 likewise)
    vision.refine.corrector.train_model   the chunk-diffusion possession corrector (pipe_gsraw_pdc)

Key of a unit = sha256 over: the unit name; the CONTENT of every argument (numpy / torch arrays byte by byte, dicts,
lists, argparse namespaces, ...); the hidden inputs it reads (pipe_ens2 PARAMS and feature names, pipe_gs blind GS
columns, the corrector's pack files); the sources the unit can run (ckpt.dep_hash: the static src import closure of
the trainer's module + the files of any runtime patch bound in those namespaces; a change there -> a miss, a change
in an unrelated module -> still a hit); numpy / lightgbm / torch versions. A file runs/ckpt/<unit>_<key[:24]>.pt holds the trained result AND the global
state the unit leaves behind (python / numpy / torch CPU / CUDA RNG states, deterministic-algorithm and TF32 flags),
which is restored on reuse, so the rest of the build runs exactly as if the unit had been trained in this process.
Writes are atomic (tmp + rename): a crash mid-write leaves no checkpoint.

Off with MTAH_CKPT=0; directory MTAH_CKPT_DIR (default runs/ckpt). Nothing here reads the provided match.
"""
import hashlib
import os
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
_INSTALLED = set()


def enabled():
    return os.environ.get("MTAH_CKPT", "1") != "0"


def ckpt_dir():
    return os.environ.get("MTAH_CKPT_DIR") or os.path.join(ROOT, "runs", "ckpt")


def _feed(h, x, memo):
    if x is None or isinstance(x, (bool, int, float, str, bytes)):
        h.update(("%s:%r|" % (type(x).__name__, x)).encode())
        return
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        h.update(("nd|%s|%s|" % (a.dtype.str, a.shape)).encode())
        if a.dtype == object:
            for v in a.ravel():
                _feed(h, v, memo)
        else:
            h.update(memoryview(a.reshape(-1).view(np.uint8)))
        return
    if isinstance(x, np.generic):
        _feed(h, np.asarray(x), memo)
        return
    mod = type(x).__module__ or ""
    if mod.startswith("torch") and hasattr(x, "detach") and hasattr(x, "cpu"):
        import torch
        y = x.detach().cpu().contiguous()
        h.update(("tensor|%s|%s|" % (y.dtype, x.device.type)).encode())
        _feed(h, (y.float() if y.dtype == torch.bfloat16 else y).numpy(), memo)
        return
    if isinstance(x, dict):
        h.update(("dict%d|" % len(x)).encode())
        for k in sorted(x, key=repr):
            _feed(h, k, memo)
            _feed(h, x[k], memo)
        return
    if isinstance(x, (list, tuple)):
        h.update(("%s%d|" % (type(x).__name__, len(x))).encode())
        for v in x:
            _feed(h, v, memo)
        return
    if isinstance(x, (set, frozenset)):
        _feed(h, sorted(x, key=repr), memo)
        return
    if isinstance(x, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)):
        h.update(("fn|%s|" % getattr(x, "__qualname__", repr(x))).encode())
        return
    if hasattr(x, "__dict__"):
        h.update(("obj|%s.%s|" % (mod, type(x).__qualname__)).encode())
        _feed(h, {k: v for k, v in vars(x).items() if not callable(v)}, memo)
        return
    raise TypeError("ckpt: cannot hash an argument of type %s" % type(x))


_MEMO = None


def code_hash():
    h = hashlib.sha256()
    for name in sorted(sys.modules):
        f = getattr(sys.modules[name], "__file__", None) or ""
        if f.endswith(".py") and os.path.abspath(f).startswith(SRC + os.sep) and name != "mtah.ckpt":
            h.update(name.encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
    for lib in ("numpy", "lightgbm", "torch", "scipy"):
        m = sys.modules.get(lib)
        h.update(("%s=%s|" % (lib, getattr(m, "__version__", "-"))).encode())
    return h.hexdigest()


_IMPORT_RE = None
_CLOSURE = {}


def _mod_file(name):
    base = os.path.join(SRC, *name.split("."))
    for f in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(f):
            return f
    return None


def _closure(roots):
    import re
    global _IMPORT_RE
    _IMPORT_RE = _IMPORT_RE or re.compile(r"^\s*(?:from\s+((?:mtah|vision)[\w.]*)\s+import\s+([\w ,()]+)|"
                                          r"import\s+((?:mtah|vision)[\w.]*))", re.M)
    key = tuple(sorted(roots))
    if key in _CLOSURE:
        return _CLOSURE[key]
    seen, todo = set(), [r for r in roots if _mod_file(r)]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        with open(_mod_file(m)) as f:
            src = f.read()
        for frm, names, imp in _IMPORT_RE.findall(src):
            cands = [imp] if imp else [frm] + [frm + "." + x.strip() for x in names.replace("(", " ").replace(
                ")", " ").split(",") if x.strip()]
            for c in cands:
                c = c.split(" as ")[0].strip()
                if c and c != "mtah.ckpt" and _mod_file(c) and c not in seen:
                    todo.append(c)
    _CLOSURE[key] = sorted(seen)
    return _CLOSURE[key]


def dep_hash(roots):
    import inspect
    mods = _closure(roots)
    files = {_mod_file(m) for m in mods}
    for m in mods:
        mod = sys.modules.get(m)
        if mod is None:
            continue
        for v in list(vars(mod).values()):
            f = None
            code = getattr(v, "__code__", None)
            if code is not None:
                f = code.co_filename
            elif isinstance(v, type):
                try:
                    f = inspect.getsourcefile(v)
                except (TypeError, OSError):
                    f = None
            if f and os.path.abspath(f).startswith(SRC + os.sep) and os.path.abspath(f) != os.path.abspath(__file__):
                files.add(os.path.abspath(f))
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(os.path.relpath(f, SRC).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    for lib in ("numpy", "lightgbm", "torch", "scipy"):
        m = sys.modules.get(lib)
        h.update(("%s=%s|" % (lib, getattr(m, "__version__", "-"))).encode())
    return h.hexdigest()


def key_of(unit, args, kwargs, hidden, roots=None):
    h = hashlib.sha256()
    h.update(("unit|%s|" % unit).encode())
    h.update((dep_hash(roots) if roots else code_hash()).encode())
    _feed(h, list(args), _MEMO)
    _feed(h, dict(kwargs), _MEMO)
    _feed(h, hidden, _MEMO)
    return h.hexdigest()


def _state():
    import random
    s = {"py": random.getstate(), "np": np.random.get_state()}
    t = sys.modules.get("torch")
    if t is not None:
        s["torch"] = t.get_rng_state()
        s["flags"] = {"det": t.are_deterministic_algorithms_enabled(),
                      "det_warn": t.is_deterministic_algorithms_warn_only_enabled(),
                      "cudnn_det": t.backends.cudnn.deterministic, "cudnn_bench": t.backends.cudnn.benchmark,
                      "cudnn_tf32": t.backends.cudnn.allow_tf32, "mm_tf32": t.backends.cuda.matmul.allow_tf32}
        if t.cuda.is_available() and t.cuda.is_initialized():
            s["cuda"] = t.cuda.get_rng_state_all()
    s["env"] = {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG",)}
    return s


def _restore(s):
    import random
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    for k, v in s.get("env", {}).items():
        if v is not None:
            os.environ[k] = v
    if "torch" in s:
        import torch
        torch.set_rng_state(s["torch"])
        f = s["flags"]
        torch.use_deterministic_algorithms(f["det"], warn_only=f["det_warn"])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = f["cudnn_det"], f["cudnn_bench"]
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = f["cudnn_tf32"], f["mm_tf32"]
        if "cuda" in s:
            torch.cuda.set_rng_state_all(s["cuda"])


def _save(path, obj):
    import pickle
    tmp = "%s.tmp%d" % (path, os.getpid())
    try:
        if "torch" in sys.modules:
            import torch
            torch.save(obj, tmp, pickle_protocol=4)
        else:
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=4)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _pack(res, build):
    t = sys.modules.get("torch")
    if build is not None and t is not None and isinstance(res, t.nn.Module):
        dev = next(iter(res.parameters())).device
        return {"__module_state__": {k: v.detach().cpu().clone() for k, v in res.state_dict().items()},
                "device": str(dev), "training": res.training}
    return res


def _unpack(res, build, args, kwargs):
    if isinstance(res, dict) and "__module_state__" in res:
        import torch
        st = res["__module_state__"]
        m = build(args, kwargs, st)
        m = m.to(torch.device(res["device"]))
        m.load_state_dict(st, strict=True)
        m.train(res["training"])
        return m
    return res


def _load(path):
    import pickle
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        import torch
        return torch.load(path, map_location=None, weights_only=False)
    with open(path, "rb") as f:
        return pickle.load(f)


def wrap(unit, fn, hidden=lambda: None, drop=("log",), build=None, roots=None):
    if getattr(fn, "_ckpt_unit", None):
        return fn

    def g(*args, **kwargs):
        if not enabled():
            return fn(*args, **kwargs)
        t = time.time()
        try:
            key = key_of(unit, args, {k: v for k, v in kwargs.items() if k not in drop}, hidden(),
                         roots or [getattr(fn, "__module__", None) or "mtah"])
            d = ckpt_dir()
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "%s_%s.pt" % (unit.replace(".", "-"), key[:24]))
        except Exception as e:   # noqa: BLE001  (a checkpoint problem must never stop or change a build)
            print("  [ckpt] %s: no checkpoint (%s: %s)" % (unit, type(e).__name__, e), flush=True)
            return fn(*args, **kwargs)
        if os.path.exists(path):
            try:
                z = _load(path)
            except Exception as e:   # noqa: BLE001  (unreadable file: train again, overwrite)
                print("  [ckpt] %s: unreadable %s (%s), training" % (unit, path, e), flush=True)
                z = {}
            if z.get("key") == key:
                res = _unpack(z["result"], build, args, kwargs)
                _restore(z["state"])
                print("  [ckpt] %s reused %s (key %.0fs)" % (unit, path, time.time() - t), flush=True)
                return res
        res = fn(*args, **kwargs)
        try:
            _save(path, {"key": key, "unit": unit, "result": _pack(res, build), "state": _state(),
                         "time": time.time()})
            print("  [ckpt] %s saved %s (key %.0fs)" % (unit, path, time.time() - t), flush=True)
        except Exception as e:   # noqa: BLE001
            print("  [ckpt] %s: save failed (%s: %s)" % (unit, type(e).__name__, e), flush=True)
        return res
    g._ckpt_unit = unit
    g.__wrapped__ = fn
    g.__name__, g.__doc__ = getattr(fn, "__name__", unit), getattr(fn, "__doc__", None)
    return g


def _ens2_hidden():
    from mtah import pipe_ens2
    h = {"params": dict(pipe_ens2.PARAMS)}
    try:
        h["feature_names"] = list(pipe_ens2._feature_names())
    except Exception as e:   # noqa: BLE001  (no cached view yet: the names cannot matter then either)
        h["feature_names"] = "unavailable: %s" % type(e).__name__
    g = sys.modules.get("mtah.pipe_gs")
    if g is not None and getattr(g, "_BLIND_GS", None):
        h["blind_gs"] = [g._BLIND_GS[k] for k in sorted(g._BLIND_GS)]
    return h


def _arg(args, kwargs, i, name):
    return kwargs[name] if name in kwargs else args[i]


def _build_nn(args, kwargs, st):
    from mtah import nn_seq
    F, a = _arg(args, kwargs, 1, "F"), _arg(args, kwargs, 3, "a")
    return nn_seq.make_model(F, H=a.hidden, layers=a.layers, in_drop=a.in_drop, arch=a.arch)


def _build_nn_v1(args, kwargs, st):
    from mtah import nn_seq, pipe_ens
    a = pipe_ens.NN_ARGS
    return nn_seq.make_model(_arg(args, kwargs, 1, "F"), H=a.hidden, layers=a.layers, in_drop=a.in_drop, arch=a.arch)


def _build_raw(mod):
    def b(args, kwargs, st):
        import torch
        a = _arg(args, kwargs, 1, "a")
        m = sys.modules[mod].make_model(a)
        if "crf_prm" in st:
            m.crf_prm = torch.nn.Parameter(torch.zeros_like(st["crf_prm"]))
        return m
    return b


def install(pipeline=None):
    if not enabled():
        return
    import importlib
    for mn in ("mtah.blind", "mtah.nn_seq", "mtah.raw_seq"):
        try:
            importlib.import_module(mn)
        except Exception:   # noqa: BLE001
            pass
    if pipeline and "pdc" in pipeline:
        importlib.import_module("vision.refine.corrector")
    ens2 = importlib.import_module("mtah.pipe_ens2")
    w = wrap("pipe_ens2.fit_lgb", ens2.fit_lgb, _ens2_hidden)
    ens2.fit_lgb = w
    if "mtah.pipe_lgb2" in sys.modules or pipeline == "mtah.pipe_lgb2":
        importlib.import_module("mtah.pipe_lgb2").fit_lgb = w
    ens2._train_nn = wrap("pipe_ens2._train_nn", ens2._train_nn, _ens2_hidden, build=_build_nn)
    ens = importlib.import_module("mtah.pipe_ens")
    ens._train_nn = wrap("pipe_ens._train_nn", ens._train_nn,
                         lambda: {"params": dict(ens.PARAMS), "nn_args": vars(ens.NN_ARGS)}, build=_build_nn_v1)
    for mn in ("mtah.raw_seq", "mtah.raw_seq_bt", "mtah.raw_seq3"):
        if mn in sys.modules:
            m = sys.modules[mn]
            m.train_model = wrap(mn.split(".")[1] + ".train_model", m.train_model, drop=("log", "val"),
                                 build=_build_raw(mn))
    if pipeline and "pdc" in pipeline:
        from vision.refine import corrector as C

        def hidden_c():
            return {"pack_dir": C.PACK, "packs": _pack_hashes(C)}
        C.train_model = wrap("corrector.train_model", C.train_model, hidden_c)
    _INSTALLED.add(pipeline)


_PACK_MEMO = {}


def _pack_hashes(C):
    out = {}
    for f in sorted(os.listdir(C.PACK)):
        if f.endswith(".npz"):
            st = os.stat(os.path.join(C.PACK, f))
            mk = (C.PACK, f, st.st_size, st.st_mtime_ns)
            if mk in _PACK_MEMO:
                out[f] = _PACK_MEMO[mk]
                continue
            h = hashlib.sha256()
            with open(os.path.join(C.PACK, f), "rb") as fh:
                for b in iter(lambda: fh.read(1 << 22), b""):
                    h.update(b)
            out[f] = _PACK_MEMO[mk] = h.hexdigest()
    return out


def _cmd(pid):
    try:
        with open("/proc/%s/cmdline" % pid, "rb") as f:
            a = [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
        cwd = os.path.realpath("/proc/%s/cwd" % pid)
    except OSError:
        return None, None
    return a, cwd


def _norm(argv):
    if not argv or not os.path.basename(argv[0]).startswith("python"):
        return None
    i = 1
    while i < len(argv) and argv[i].startswith("-") and argv[i] != "-m":
        i += 2 if argv[i] in ("-W", "-X") else 1
    if i >= len(argv):
        return None
    body = argv[i:]
    out, skip = [], False
    for x in body:
        if skip:
            skip = False
            continue
        if x == "--out":
            skip = True
            continue
        if x.startswith("--out="):
            continue
        out.append(x)
    return tuple(out)


def guard_duplicate():
    if os.environ.get("MTAH_ALLOW_DUPLICATE") == "1" or not os.path.isdir("/proc"):
        return
    me, cwd = _cmd("self")
    mine = _norm(me)
    if not mine:
        return
    anc, p = set(), os.getpid()
    while p > 1:
        anc.add(p)
        try:
            with open("/proc/%d/stat" % p) as f:
                p = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in anc:
            continue
        a, c = _cmd(d)
        if a and c == cwd and _norm(a) == mine:
            raise SystemExit("duplicate build refused: pid %s already runs %s in %s (MTAH_ALLOW_DUPLICATE=1 to "
                             "override)" % (d, " ".join(a), c))
