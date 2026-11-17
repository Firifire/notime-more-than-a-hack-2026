import functools

from mtah import pipe_gsraw2 as _g, pipe_raw
from mtah.pipe_raw import PARAMS, boundary_labels, HEADS  # noqa: F401

GS_MEMBER_KEYS, DEFAULT_MEMBERS = _g.GS_MEMBER_KEYS, _g.DEFAULT_MEMBERS


def _with_base(f):
    @functools.wraps(f)
    def g(*a, **k):
        old = _g.BASE
        _g.BASE = pipe_raw
        try:
            return f(*a, **k)
        finally:
            _g.BASE = old
    return g


fit = _with_base(_g.fit)
predict = _with_base(_g.predict)
decode_heads = _with_base(_g.decode_heads)
