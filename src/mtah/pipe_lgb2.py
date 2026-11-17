from mtah.pipe_ens2 import PARAMS, boundary_labels, decode_heads, fit_lgb, predict_lgb, _opts  # noqa: F401


def fit(rows, dead_set, team_set):
    opts = _opts()
    print("  [pipe_lgb2] opts %s" % opts, flush=True)
    return fit_lgb(rows, dead_set, team_set, opts)


def predict(models, m=None, X=None, Xs=None):
    if X is None:
        raise ValueError("pipe_lgb2.predict needs the feature views X / Xs")
    return predict_lgb(models, X, Xs)
