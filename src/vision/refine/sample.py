import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.refine import sim  # noqa: E402
from vision.refine.model import TrajDiffusion, K, V, MASK, noise_tokens  # noqa: E402
import math  # noqa: E402
from vision.refine.train import T_WIN, C0, C1, load_real  # noqa: E402

OUT = sim.OUT
CAN = C1 - C0
M_TOP = 16
VIT_LAM = 1.0
VIT_DFREE = 35.0 / 25.0


def load_model(path, dev):
    st = torch.load(path, map_location=dev, weights_only=False)
    a = st["args"]
    m = TrajDiffusion(d=a["d"], layers=a["layers"], heads=a["heads"], ff=4 * a["d"],
                      tied="cell_head.weight" not in st["model"]).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    return m


def _windows(arr, starts, fill):
    n = len(arr)
    out = np.full((len(starts), T_WIN) + arr.shape[1:], fill, arr.dtype)
    for b, s in enumerate(starts):
        a0 = s - C0
        lo, hi = max(a0, 0), min(a0 + T_WIN, n)
        if hi > lo:
            out[b, lo - a0:hi - a0] = arr[lo:hi]
    return out


KIN_VMAX = 1.5
KIN_SLACK = 0.5
KIN_W = 50


def kin_prior(model, ev, x, committed, lam, vmax=KIN_VMAX, slack=KIN_SLACK, W=KIN_W, kin=None, free_tok=None):
    if kin is not None:
        vmax, slack = kin
    if free_tok is not None:
        committed = committed & (x != free_tok)
    B, T = x.shape
    dev = x.device
    pos = model.token_xy(ev, x.clamp(max=V - 1))
    ar = torch.arange(T, device=dev)[None].expand(B, T)
    li = torch.where(committed, ar, torch.full_like(ar, -1))
    li = torch.cummax(torch.cat([torch.full((B, 1), -1, device=dev, dtype=li.dtype), li[:, :-1]], 1), 1).values
    ri = torch.where(committed, ar, torch.full_like(ar, T))
    ri = torch.flip(torch.cummin(torch.flip(torch.cat([ri[:, 1:], torch.full((B, 1), T, device=dev,
                                                                              dtype=ri.dtype)], 1), [1]), 1).values, [1])
    cells = model.centres
    pen = torch.zeros(B, T, V, device=dev)
    for idx, valid in ((li, li >= 0), (ri, ri < T)):
        dt = (ar - idx).abs().float()
        valid = valid & (dt <= W)
        a = torch.gather(pos, 1, idx.clamp(0, T - 1)[..., None].expand(-1, -1, 2))
        allow = (vmax * dt + slack)[..., None]
        dc = (ev["cxy"] - a[:, :, None]).norm(dim=-1)
        dg = torch.cdist(a.reshape(B * T, 1, 2), cells[None].expand(B * T, -1, -1)).view(B, T, -1)
        d = torch.cat([dc, dg], -1)
        pen += torch.relu(d - allow) * valid[..., None]
    if free_tok is not None:
        pen[..., free_tok] = 0.0
    return -lam * pen


@torch.no_grad()
def denoise(model, ev, x, flag, gen, N=48, b=0.1, tmax=0.8, tmin=0.4, e_stop=0.005, commit="sample",
            schedule=True, kin_lam=0.0, kin=None, free_tok=None):
    B, T = x.shape
    dev = x.device
    canvas = flag == 0
    nC = canvas.sum(-1, keepdim=True)
    x = torch.where(canvas, noise_tokens(ev["cv"], gen), x)
    sc = None
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    prev = torch.full((B, T), -1, dtype=torch.long, device=dev)
    out = off = ent = pmax = None
    committed = flag == 1
    for n in range(N, 0, -1):
        logits, o = model(ev, x, flag, sc)
        if kin_lam > 0:
            logits = logits.float() + kin_prior(model, ev, x, committed, kin_lam, kin=kin, free_tok=free_tok)
        tau = tmin + (tmax - tmin) * n / N
        pl = logits.float() / tau
        p = pl.softmax(-1)
        new_am = pl.argmax(-1)
        xd = new_am if commit == "argmax" else torch.multinomial(p.flatten(0, 1), 1, generator=gen).view(B, T)
        e = -(p * p.clamp(min=1e-12).log()).sum(-1)
        p1 = logits.float().softmax(-1)
        e1 = -(p1 * p1.clamp(min=1e-12).log()).sum(-1)
        e_c = torch.where(canvas, e, torch.full_like(e, float("inf")))
        se, si = torch.sort(e_c, dim=-1, stable=True)
        acc_sorted = ((se.cumsum(-1) - se) <= b) & torch.isfinite(se)
        if schedule:
            frac = 1.0 - math.cos(math.pi / 2 * (N - n + 1) / N)
            rank = torch.arange(T, device=dev)[None]
            acc_sorted |= (rank < torch.ceil(frac * nC)) & torch.isfinite(se)
        acc = torch.zeros_like(acc_sorted).scatter(1, si, acc_sorted)
        committed = torch.where(finished[:, None], committed, (acc & canvas) | (flag == 1))
        xa = torch.where(acc, xd, x)
        xr = torch.where(canvas & ~acc, noise_tokens(ev["cv"], gen), xa)
        f = finished[:, None]
        if out is None:
            out, off, ent, pmax = xa, o.float(), e1, p1.max(-1).values
            tl, ti = p1.clamp(min=1e-12).log().topk(M_TOP, dim=-1)
        else:
            out = torch.where(f, out, xa)
            off = torch.where(f[..., None], off, o.float())
            ent = torch.where(f, ent, e1)
            pmax = torch.where(f, pmax, p1.max(-1).values)
            ntl, nti = p1.clamp(min=1e-12).log().topk(M_TOP, dim=-1)
            tl = torch.where(f[..., None], tl, ntl)
            ti = torch.where(f[..., None], ti, nti)
        x = torch.where(f, x, xr)
        sc = pl if sc is None else torch.where(f[..., None], sc, pl)
        mean_e = (e * canvas).sum(-1) / canvas.sum(-1).clamp(min=1)
        stable = ((new_am == prev) | ~canvas).all(-1)
        stop = stable & (mean_e < e_stop) & ~finished
        out = torch.where(stop[:, None], new_am, out)
        finished |= stop
        prev = new_am
        if finished.all():
            break
    if not schedule:
        out = torch.where(finished[:, None], out, new_am)
    return out, off, ent, pmax, tl, ti


def run_half(model, cand, players, dev, N=48, sweeps=3, b=0.1, batch=16, seed=0, commit="sample", schedule=True,
             kin_lam=0.0):
    n = len(cand)
    tok = np.full(n, MASK, np.int64)
    top_l = np.full((n, M_TOP), -1e9, np.float32)
    top_i = np.zeros((n, M_TOP), np.int64)
    off = np.zeros((n, 2), np.float32)
    ent = np.full(n, np.inf, np.float32)
    pmax = np.zeros(n, np.float32)
    gen = torch.Generator(device=dev)
    for sw in range(sweeps):
        gen.manual_seed(seed * 1000 + sw)
        shift = (CAN // 2) if sw % 2 == 1 else 0
        starts = np.arange(-shift, n, CAN)
        groups = [[s] for s in starts] if sw == 0 else [starts[i:i + batch] for i in range(0, len(starts), batch)]
        new_tok, new_off, new_ent, new_pmax = tok.copy(), off.copy(), ent.copy(), pmax.copy()
        new_tl, new_ti = top_l.copy(), top_i.copy()
        for g in groups:
            g = np.asarray(g)
            src = new_tok if sw == 0 else tok
            cw = torch.from_numpy(_windows(cand, g, np.nan)).to(dev)
            pw = torch.from_numpy(_windows(players.astype(np.float32), g, np.nan)).to(dev)
            tw = _windows(src, g, MASK)
            fw = np.full(tw.shape, 1, np.int64)
            fw[tw == MASK] = 2
            fw[:, C0:C1] = 0
            valid = _windows(np.ones(n, bool), g, False)
            fw[:, C0:C1][~valid[:, C0:C1]] = 2
            if sw == 0:
                fw[:, C1:] = 2
                tw[:, C1:] = MASK
            canvas_ok = valid[:, C0:C1]
            x = torch.from_numpy(tw).to(dev)
            fl = torch.from_numpy(fw).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                ev = model.encode_evidence(cw, pw)
                am, o, e, pm, tl, ti = denoise(model, ev, x, fl, gen, N=N, b=b, commit=commit, schedule=schedule,
                                               kin_lam=kin_lam)
            am, o, e, pm = am.cpu().numpy(), o.cpu().numpy(), e.float().cpu().numpy(), pm.float().cpu().numpy()
            tl, ti = tl.float().cpu().numpy(), ti.cpu().numpy()
            for bi, s in enumerate(g):
                fr = np.arange(s, s + CAN)
                ok = canvas_ok[bi]
                new_tok[fr[ok]] = am[bi, C0:C1][ok]
                new_off[fr[ok]] = o[bi, C0:C1][ok]
                new_ent[fr[ok]] = e[bi, C0:C1][ok]
                new_pmax[fr[ok]] = pm[bi, C0:C1][ok]
                new_tl[fr[ok]] = tl[bi, C0:C1][ok]
                new_ti[fr[ok]] = ti[bi, C0:C1][ok]
        tok, off, ent, pmax, top_l, top_i = new_tok, new_off, new_ent, new_pmax, new_tl, new_ti
    is_c = tok < K
    xy = model.centres.cpu().numpy()[np.clip(tok - K, 0, None)] + off
    cxy = cand[np.arange(n), np.clip(tok, 0, K - 1), :2]
    xy = np.where(is_c[:, None], cxy, xy).astype(np.float32)
    out = dict(xy=xy, ent=ent, pmax=pmax, is_cand=is_c, tok=tok, top_l=top_l.astype(np.float32), top_i=top_i)
    vx, vtok = viterbi(cand, top_l, top_i, off, model.centres.cpu().numpy())
    out["xy_vit"], out["tok_vit"] = vx, vtok
    return out


def viterbi(cand, top_l, top_i, off, centres, lam=VIT_LAM, dfree=VIT_DFREE):
    n, M = top_i.shape
    isc = top_i < K
    cx = cand[np.arange(n)[:, None], np.clip(top_i, 0, K - 1), :2]
    gx = centres[np.clip(top_i - K, 0, None)]
    if off is not None:
        gx = gx + off[:, None, :]
    pos = np.where(isc[..., None], cx, gx).astype(np.float64)
    lp = top_l.astype(np.float64)
    bad = ~np.isfinite(pos[..., 0])
    lp[bad] = -1e9
    pos[bad] = 0.0
    back = np.zeros((n, M), np.int32)
    score = lp[0].copy()
    for t in range(1, n):
        d = np.hypot(pos[t][:, None, 0] - pos[t - 1][None, :, 0], pos[t][:, None, 1] - pos[t - 1][None, :, 1])
        v = score[None, :] - lam * np.maximum(0.0, d - dfree)
        j = np.argmax(v, 1)
        back[t] = j
        score = v[np.arange(M), j] + lp[t]
    path = np.zeros(n, np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    tok = top_i[np.arange(n), path]
    xy = pos[np.arange(n), path]
    return xy.astype(np.float32), tok


def metrics(xy, cov, truth_xy, tmask, frames_all=None):
    f = np.flatnonzero(tmask)
    err = np.hypot(xy[f, 0] - truth_xy[f, 0], xy[f, 1] - truth_xy[f, 1])
    c = cov[f] & np.isfinite(err)
    errc = np.where(c, err, np.inf)
    out = {"truth_frames": int(len(f)), "coverage_at_truth": round(float(c.mean()), 4),
           "recall_1.5m": round(float(np.mean(errc < 1.5)), 4), "recall_3m": round(float(np.mean(errc < 3)), 4),
           "recall_5m": round(float(np.mean(errc < 5)), 4),
           "median_err_covered_m": round(float(np.median(err[c])), 3) if c.any() else None,
           "p90_err_covered_m": round(float(np.percentile(err[c], 90)), 3) if c.any() else None,
           "wrong_object_gt10m_of_covered": round(float(np.mean(err[c] > 10)), 4) if c.any() else None}
    cv = cov & np.isfinite(xy[:, 0])
    d = np.hypot(*np.diff(xy, axis=0).T)
    both = cv[1:] & cv[:-1]
    out["jumps_gt3m_per_min_covered"] = round(float(np.sum(both & (d > 3)) / max(cv.sum() / 1500, 1e-9)), 3)
    out["coverage_all_frames"] = round(float(cv[frames_all].mean() if frames_all is not None else cv.mean()), 4)
    tp = tmask[1:] & tmask[:-1]
    dtru = np.hypot(*np.diff(truth_xy, axis=0).T)
    pairs = tp & (dtru <= 1.5) & both
    out["false_jumps_per_1k_truth_pairs"] = round(float(1000 * np.sum(pairs & (d > 3)) / max(pairs.sum(), 1)), 3)
    out["truth_pairs"] = int(pairs.sum())
    return out


def coverage_threshold(ent, frames, target):
    e = np.sort(ent[frames])
    k = int(np.clip(round(target * len(e)) - 1, 0, len(e) - 1))
    return float(e[k])


def eval_stv2(model, hn, args, dev):
    r = load_real(hn)
    z = np.load(os.path.join(OUT, "stv2_%s.npz" % hn))
    A = z["anchor"].astype(np.float64)
    if args.no_cands:
        r["cand"][:] = np.nan
    res = run_half(model, r["cand"], r["players"], dev, N=args.steps, sweeps=args.sweeps, b=args.b,
                   commit=args.commit, schedule=not args.no_schedule, kin_lam=args.kin_lam)
    base = np.load(os.path.join(sim.CACHE, "stv2", hn, "ball.npz"))
    bxy = base["xy"].astype(np.float64)
    bst = base["state"]
    pl_ok = (~np.isnan(z["players"][:, :, 0].astype(np.float32))).any(1)
    tm = ~np.isnan(A[:, 0])
    out = {"half": hn, "baseline": metrics(bxy, bst > 0, A, tm, pl_ok),
           "baseline_detected_only": metrics(bxy, bst == 1, A, tm, pl_ok)}
    xy = res["xy"].astype(np.float64)
    out["refiner_full"] = metrics(xy, pl_ok, A, tm, pl_ok)
    thr = coverage_threshold(res["ent"], pl_ok, float((bst[pl_ok] > 0).mean()))
    out["refiner_cov_matched"] = metrics(xy, pl_ok & (res["ent"] <= thr), A, tm, pl_ok) | {"entropy_thr": thr}
    xv = res["xy_vit"].astype(np.float64)
    out["vit_full"] = metrics(xv, pl_ok, A, tm, pl_ok)
    out["vit_cov_matched"] = metrics(xv, pl_ok & (res["ent"] <= thr), A, tm, pl_ok)
    out["refiner_on_cand_share"] = round(float(res["is_cand"][pl_ok].mean()), 4)
    return out, res


def track_baseline(fr, x, y, cf, n, pfr, px, py):
    from vision import track as T
    keep = (cf >= T.CONF_MIN) & (x > -T.MARGIN_GOAL_M) & (x < 105 + T.MARGIN_GOAL_M) & (y > -T.MARGIN_GOAL_M) & \
        (y < 68 + T.MARGIN_GOAL_M) & np.isfinite(x)
    fr, x, y, cf = fr[keep], x[keep], y[keep], cf[keep]
    o = np.lexsort((-cf, fr))
    fr, x, y, cf = fr[o], x[o], y[o], cf[o]
    hot = T.hot_mask(fr, x, y)
    tid = T.build_tracklets(fr, x, y, cf)
    iso = np.zeros(len(fr), bool)
    ps = np.searchsorted(pfr, np.arange(n + 1))
    for i in range(len(fr)):
        a, b_ = ps[fr[i]], ps[fr[i] + 1]
        if b_ > a:
            iso[i] = np.min(np.hypot(px[a:b_] - x[i], py[a:b_] - y[i])) > T.ISO_M
    chosen, _ = T.select(fr, x, y, cf, tid, hot, np.zeros(len(fr), bool), iso)
    xy = np.full((n, 2), np.nan)
    st = np.zeros(n, np.int8)
    ci = np.flatnonzero(chosen)
    xy[fr[ci]] = np.c_[x[ci], y[ci]]
    st[fr[ci]] = 1
    det = np.flatnonzero(st == 1)
    for g0, g1 in zip(det[:-1], det[1:]):
        if 1 < g1 - g0 <= T.INTERP_MAX:
            t = (np.arange(g0 + 1, g1) - g0) / (g1 - g0)
            xy[g0 + 1:g1] = xy[g0] + t[:, None] * (xy[g1] - xy[g0])
            st[g0 + 1:g1] = 2
    return xy, st


def issia_inputs():
    z = np.load(os.path.join(OUT, "issia_evidence.npz"))
    n = int(z["n"])
    cand = sim.pack(z["c_frame"].astype(np.int64), z["c_x"].astype(np.float64), z["c_y"].astype(np.float64),
                    z["c_conf"], n)
    pfr, px, py = z["p_frame"], z["p_x"], z["p_y"]
    ok = np.isfinite(px) & (px > -3) & (px < 108) & (py > -3) & (py < 71)
    pfr, px, py = pfr[ok], px[ok], py[ok]
    o = np.argsort(pfr, kind="stable")
    pfr, px, py = pfr[o], px[o], py[o]
    P = np.full((n, sim.S, 4), np.nan, np.float32)
    st = np.searchsorted(pfr, np.arange(n + 1))
    for f in range(n):
        q = np.c_[px[st[f]:st[f + 1]], py[st[f]:st[f + 1]]]
        keep = []
        for p in q:
            if all(np.hypot(*(p - k)) > 1.0 for k in keep):
                keep.append(p)
        keep = np.array(keep[:sim.S]).reshape(-1, 2)
        P[f, :len(keep), :2] = keep
        P[f, :len(keep), 2:] = 0.0
    return z, n, cand, P, (pfr, px, py)


def eval_issia(model, args, dev):
    z, n, cand, P, (pfr, px, py) = issia_inputs()
    truth = z["truth"].astype(np.float64)
    tm = z["truth_src"] > 0
    bxy, bst = track_baseline(z["c_frame"].astype(np.int64), z["c_x"].astype(np.float64),
                              z["c_y"].astype(np.float64), z["c_conf"], n, pfr, px, py)
    if args.no_cands:
        cand[:] = np.nan
    res = run_half(model, cand, P, dev, N=args.steps, sweeps=args.sweeps, b=args.b, commit=args.commit,
                   schedule=not args.no_schedule, kin_lam=args.kin_lam)
    allf = np.ones(n, bool)
    out = {"baseline": metrics(bxy, bst > 0, truth, tm), "baseline_detected_only": metrics(bxy, bst == 1, truth, tm),
           "refiner_full": metrics(res["xy"].astype(np.float64), allf, truth, tm)}
    thr = coverage_threshold(res["ent"], allf, float((bst > 0).mean()))
    out["refiner_cov_matched"] = metrics(res["xy"].astype(np.float64), res["ent"] <= thr, truth, tm) | {
        "entropy_thr": thr}
    xv = res["xy_vit"].astype(np.float64)
    out["vit_full"] = metrics(xv, allf, truth, tm)
    out["vit_cov_matched"] = metrics(xv, res["ent"] <= thr, truth, tm)
    for nm, sel in (("truth3d", z["truth_src"] == 1), ("truth2d_ground", z["truth_src"] == 2)):
        out["refiner_full_" + nm] = metrics(res["xy"].astype(np.float64), allf, truth, sel)
        out["baseline_" + nm] = metrics(bxy, bst > 0, truth, sel)
    return out, res


def pooled(rows, key):
    tot = sum(r[key]["truth_frames"] for r in rows)
    out = {"truth_frames": tot}
    for m in ("coverage_at_truth", "recall_1.5m", "recall_3m", "recall_5m"):
        out[m] = round(sum(r[key][m] * r[key]["truth_frames"] for r in rows) / tot, 4)
    out["median_err_covered_m_mean"] = round(float(np.mean([r[key]["median_err_covered_m"] or np.nan for r in rows])), 3)
    out["wrong_object_gt10m_mean"] = round(float(np.mean([r[key]["wrong_object_gt10m_of_covered"] or 0 for r in rows])), 4)
    out["jumps_per_min_mean"] = round(float(np.mean([r[key]["jumps_gt3m_per_min_covered"] for r in rows])), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--halves", nargs="*", default=[])
    ap.add_argument("--issia", action="store_true")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--b", type=float, default=0.1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-cands", action="store_true", help="ablation: players only, no detections")
    ap.add_argument("--commit", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--no-schedule", action="store_true", help="pure entropy-bound commits (text sampler)")
    ap.add_argument("--kin-lam", type=float, default=0.0, help="in-sampler kinematic prior (nats per metre)")
    a = ap.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    model = load_model(a.model, dev)
    odir = os.path.join(OUT, "eval", a.tag)
    os.makedirs(odir, exist_ok=True)
    rows = []
    rep = {"model": a.model, "steps": a.steps, "sweeps": a.sweeps, "b": a.b, "commit": a.commit,
           "schedule": not a.no_schedule, "no_cands": a.no_cands, "kin_lam": a.kin_lam, "halves": {}}
    for hn in a.halves:
        r, res = eval_stv2(model, hn, a, dev)
        np.savez_compressed(os.path.join(odir, "%s.npz" % hn), **res)
        rows.append(r)
        rep["halves"][hn] = r
        print(hn, json.dumps({k: r[k] for k in ("baseline", "refiner_full", "refiner_cov_matched", "vit_full",
                                                 "vit_cov_matched")}), flush=True)
    if rows:
        rep["stv2_pooled"] = {k: pooled(rows, k) for k in ("baseline", "baseline_detected_only", "refiner_full",
                                                           "refiner_cov_matched", "vit_full", "vit_cov_matched")}
        print("POOLED", json.dumps(rep["stv2_pooled"]), flush=True)
    if a.issia:
        r, res = eval_issia(model, a, dev)
        np.savez_compressed(os.path.join(odir, "issia.npz"), **res)
        rep["issia"] = r
        print("ISSIA", json.dumps(r), flush=True)
    json.dump(rep, open(os.path.join(odir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
