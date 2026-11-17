import gc

from mtah import pipe_gs, pipe_raw2
from mtah.pipe_raw import PARAMS, boundary_labels, HEADS  # noqa: F401  (PARAMS: same dict object as train_v5)

BASE = pipe_raw2

GS_MEMBER_KEYS = ("pipe_spell_w", "pipe_nn_spell_w", "pipe_nn_seeds")
DEFAULT_MEMBERS = ("lgb_team", "nn_team", "lgb_sw", "nn_sw")


def _hide(pred):
    return {k: PARAMS.pop(k) for k in [k for k in PARAMS if pred(k)]}


def fit(rows, dead_set, team_set):
    saved = _hide(lambda k: k.startswith("pipe_gs"))
    try:
        base = BASE.fit(rows, dead_set, team_set)
    finally:
        PARAMS.update(saved)
    gc.collect()
    saved = _hide(lambda k: k.startswith("pipe_") and not k.startswith("pipe_gs") and k not in GS_MEMBER_KEYS)
    try:
        print("  [pipe_gsraw2] GS member opts %s" % {k: v for k, v in PARAMS.items() if k.startswith("pipe_")},
              flush=True)
        gsm = pipe_gs.fit(rows, dead_set, team_set)
    finally:
        PARAMS.update(saved)
    return {"base": base, "gs": gsm}


def predict(models, m=None, X=None, Xs=None, half=None, names=None):
    p = BASE.predict(models["base"], m=m, X=X, Xs=Xs, half=half, names=names)
    q = pipe_gs.predict(models["gs"], X=X, Xs=Xs, half=half, names=names)
    for k in HEADS:
        p["gs_lgb_" + k] = q["lgb_" + k]
        p["gs_nn_" + k] = q["nn_" + k]
    return p


def decode_heads(p, half, cfg):
    members = cfg.get("gs_members", list(DEFAULT_MEMBERS))
    if "gs_lgb_team" in p:
        p = dict(p, **{k: p["gs_" + k] for k in members})
    return BASE.decode_heads(p, half, {k: v for k, v in cfg.items() if k != "gs_members"})
