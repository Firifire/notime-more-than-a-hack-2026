import numpy as np

from mtah.data import HALVES, N_FRAMES

H, A, D, X = 0, 1, 2, 3


def viterbi(logp, trans_cost):
    T, K = logp.shape
    score = logp[0].copy()
    back = np.zeros((T, K), np.int8)
    neg = -trans_cost
    for t in range(1, T):
        cand = score[:, None] + neg
        back[t] = np.argmax(cand, 0)
        score = cand[back[t], np.arange(K)] + logp[t]
    path = np.empty(T, np.int8)
    path[-1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def make_trans(switch_team=8.0, to_dead=8.0, from_dead=8.0):
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = switch_team
    c[H, D] = c[A, D] = to_dead
    c[D, H] = c[D, A] = from_dead
    return c


def decode(prob, trans, eps=1e-4, halves=HALVES, n=N_FRAMES):
    out = np.full(n, X, np.int8)
    lp = np.log(np.clip(prob, eps, 1))
    for _, a, b in halves:
        out[a:b + 1] = viterbi(lp[a:b + 1], trans)
    return out


def viterbi_mindur(logp, min_dur, switch_cost):
    T, K = logp.shape
    cls, first, last = [], [], []
    for c in range(K):
        first.append(len(cls))
        cls += [c] * int(min_dur[c])
        last.append(len(cls) - 1)
    cls = np.array(cls)
    S = len(cls)
    NEG = -1e18
    score = np.full(S, NEG)
    for c in range(K):
        score[first[c]] = logp[0, c]
    back = np.zeros((T, S), np.int32)
    idx = np.arange(S)
    for t in range(1, T):
        new = np.full(S, NEG)
        arg = np.zeros(S, np.int32)
        chain = np.ones(S, bool)
        chain[first] = False
        new[chain] = score[idx[chain] - 1]
        arg[chain] = idx[chain] - 1
        for c in range(K):
            l = last[c]
            if score[l] > new[l]:
                new[l], arg[l] = score[l], l
        for j in range(K):
            f = first[j]
            for i in range(K):
                if i == j:
                    continue
                v = score[last[i]] - switch_cost[i, j]
                if v > new[f]:
                    new[f], arg[f] = v, last[i]
        score = new + logp[t, cls]
        back[t] = arg
    s = int(np.argmax(np.where(np.isin(idx, last), score, NEG)))
    path = np.empty(T, np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = cls[s]
        s = back[t, s]
    return path


def decode_mindur(prob, min_dur, switch_cost, halves_mask, eps=1e-4, prior=None, scale=1.0):
    lp = np.log(np.clip(prob, eps, 1))
    if prior is not None:
        lp = lp - np.log(prior)[None]
    lp = lp * scale
    out = np.full(len(prob), X, np.int8)
    for idx in halves_mask:
        out[idx] = viterbi_mindur(lp[idx], min_dur, switch_cost)
    return out


try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _viterbi_tv_py(logp, cost):
    T, K = logp.shape
    score = logp[0].copy()
    back = np.zeros((T, K), np.int8)
    ar = np.arange(K)
    for t in range(1, T):
        cand = score[:, None] - cost[t]
        b = np.argmax(cand, 0)
        back[t] = b
        score = cand[b, ar] + logp[t]
    path = np.empty(T, np.int8)
    path[-1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


if njit is not None:
    @njit(cache=True)
    def _viterbi_tv_nb(logp, cost):
        T, K = logp.shape
        score = logp[0].copy()
        back = np.zeros((T, K), np.int8)
        new = np.empty(K)
        for t in range(1, T):
            for j in range(K):
                best = -1e300
                bi = 0
                for i in range(K):
                    v = score[i] - cost[t, i, j]
                    if v > best:
                        best = v
                        bi = i
                back[t, j] = bi
                new[j] = best + logp[t, j]
            for j in range(K):
                score[j] = new[j]
        path = np.empty(T, np.int8)
        bj = 0
        for j in range(K):
            if score[j] > score[bj]:
                bj = j
        path[T - 1] = bj
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path


def viterbi_tv(logp, cost):
    if njit is not None:
        return _viterbi_tv_nb(np.ascontiguousarray(logp, np.float64), np.ascontiguousarray(cost, np.float64))
    return _viterbi_tv_py(logp, cost)


def boundary_costs(n, switch, dead_on, dead_off, p_sw=None, p_on=None, p_off=None, beta=0.0,
                   eps=1e-3):
    def term(base, p):
        if p is None or beta == 0:
            return np.full(n, base)
        p = np.clip(p, eps, 1 - eps)
        return np.maximum(base - beta * np.log(p / (1 - p)), 0.0)
    c = np.zeros((n, 3, 3))
    s = term(switch, p_sw)
    on = term(dead_on, p_on)
    off = term(dead_off, p_off)
    c[:, H, A] = c[:, A, H] = s
    c[:, H, D] = c[:, A, D] = on
    c[:, D, H] = c[:, D, A] = off
    return c


def step_energy(x, h, w=None):
    x = np.nan_to_num(np.asarray(x, np.float64))
    w = np.ones(len(x)) if w is None else np.asarray(w, np.float64)
    cw = np.r_[0, np.cumsum(w)]
    cx = np.r_[0, np.cumsum(w * x)]
    n = len(x)
    t = np.arange(n)
    lo = np.maximum(0, t - h)
    hi = np.minimum(n, t + h)
    WL = cw[t] - cw[lo]
    WR = cw[hi] - cw[t]
    muL = (cx[t] - cx[lo]) / np.maximum(WL, 1e-9)
    muR = (cx[hi] - cx[t]) / np.maximum(WR, 1e-9)
    return WL * WR / np.maximum(WL + WR, 1e-9) * (muL - muR) ** 2
