import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.refine import sim  # noqa: E402
from vision.refine.model import TrajDiffusion, K, V, MASK, G, cell_of, noise_tokens  # noqa: E402

OUT = sim.OUT
T_WIN, C0, C1 = 512, 128, 384
R_SNAP = 1.0


def load_real(hn):
    z = np.load(os.path.join(OUT, "stv2_%s.npz" % hn))
    n = int(z["n"])
    cand = sim.pack(z["c_frame"].astype(np.int64), z["c_x"], z["c_y"], z["c_conf"], n)
    P = sim.player_feats(z["players"].astype(np.float32))
    truth = z["ref"].astype(np.float32)
    w = np.where(~np.isnan(z["anchor"][:, 0]), 1.0, np.where(~np.isnan(truth[:, 0]), 0.5, 0.0)).astype(np.float32)
    return dict(cand=cand, players=P, truth=truth, w=w, n=n)


class Windows(torch.utils.data.IterableDataset):
    def __init__(self, real_halves, seed, real_share=0.25):
        self.clean = sim.clean_matches()
        self.real_halves = real_halves
        self.seed = seed
        self.real_share = real_share

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng([self.seed, wid])
        real = [load_real(h) for h in self.real_halves]
        cl = {m: {k: np.load(os.path.join(sim.CLEAN, m, k + ".npy"), mmap_mode="r") for k in
                  ("ball", "players", "label", "half")} for m in self.clean}
        while True:
            if real and rng.random() < self.real_share:
                r = real[rng.integers(len(real))]
                for _ in range(50):
                    s = int(rng.integers(0, r["n"] - T_WIN))
                    if (r["w"][s + C0:s + C1] >= 1).sum() >= 8:
                        break
                cand = r["cand"][s:s + T_WIN].copy()
                P = sim.perturb_players(r["players"][s:s + T_WIN], rng) if rng.random() < 0.5 else \
                    r["players"][s:s + T_WIN].astype(np.float32)
                truth = r["truth"][s:s + T_WIN].copy()
                w = r["w"][s:s + T_WIN].copy()
                src = 1
            else:
                m = cl[self.clean[rng.integers(len(self.clean))]]
                hv = np.asarray(m["half"])
                for _ in range(50):
                    s = int(rng.integers(0, len(hv) - T_WIN))
                    if hv[s] > 0 and hv[s] == hv[s + T_WIN - 1] and \
                            np.mean(~np.isnan(m["ball"][s:s + T_WIN, 0])) > 0.3:
                        break
                truth = np.array(m["ball"][s:s + T_WIN], np.float32)
                P = np.array(m["players"][s:s + T_WIN], np.float32)
                lab = np.array(m["label"][s:s + T_WIN])
                fx, fy = rng.random() < 0.5, rng.random() < 0.5
                sim.flip(truth, P, None, fx, fy)
                cand = sim.simulate(truth.astype(np.float64), P, lab, rng)
                P = sim.perturb_players(P, rng)
                w = (~np.isnan(truth[:, 0])).astype(np.float32)
                src = 0
            if rng.random() < 0.15:
                P[..., 2:] = 0.0
            if src == 1:
                fx, fy = rng.random() < 0.5, rng.random() < 0.5
                sim.flip(truth, P, cand, fx, fy)
            yield (torch.from_numpy(cand), torch.from_numpy(P.astype(np.float32)), torch.from_numpy(truth),
                   torch.from_numpy(w), torch.tensor(src))


def clean_tokens(cand, truth):
    d = (torch.nan_to_num(cand[..., :2], nan=1e4) - torch.nan_to_num(truth, nan=-1e4)[:, :, None]).norm(dim=-1)
    dm, km = d.min(-1)
    cellt = K + cell_of(torch.nan_to_num(truth))
    return torch.where(dm < R_SNAP, km, cellt)


def corrupt(tok, known, cv, gen, B, T, device):
    t = torch.rand(B, 1, device=device, generator=gen)
    rnd = noise_tokens(cv, gen)
    u = torch.rand(B, T, device=device, generator=gen)
    flag = torch.zeros(B, T, dtype=torch.long, device=device)
    flag[:, :C0] = 1
    flag[:, C1:] = 1
    tc = torch.rand(B, 1, device=device, generator=gen) * 0.25
    p = torch.where(flag == 0, t.expand(B, T), tc.expand(B, T))
    x = torch.where((u < p) | ~known, rnd, tok)
    la = torch.rand(B, device=device, generator=gen) < 0.25
    ra = torch.rand(B, device=device, generator=gen) < 0.5
    absent = torch.zeros(B, T, dtype=torch.bool, device=device)
    absent[:, :C0] |= la[:, None]
    absent[:, C1:] |= ra[:, None]
    x = torch.where(absent, torch.full_like(x, MASK), x)
    flag = torch.where(absent, torch.full_like(flag, 2), flag)
    return x, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tied", type=int, default=0, help="1 = cell logits tied to the cell embeddings (v3)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--real-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    odir = os.path.join(OUT, "runs", a.name)
    os.makedirs(odir, exist_ok=True)
    halves = sorted(os.path.basename(f)[5:-4] for f in glob.glob(os.path.join(OUT, "stv2_*.npz")))
    train_h = [h for h in halves if h not in a.test]
    json.dump(vars(a) | {"train_halves": train_h}, open(os.path.join(odir, "args.json"), "w"), indent=1)
    model = TrajDiffusion(d=a.d, layers=a.layers, heads=a.heads, ff=4 * a.d, tied=bool(a.tied)).to(dev)
    print("params %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6), "train real halves", train_h,
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 1000) * (0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)) * 0.95 + 0.05))
    step = 0
    ck = os.path.join(odir, "last.pt")
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = st["step"]
        print("resumed at", step, flush=True)
    dl = torch.utils.data.DataLoader(Windows(train_h, a.seed + step, a.real_share), batch_size=a.batch,
                                     num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed + 7 * step)
    t0 = time.time()
    hist = []
    for cand, P, truth, w, src in dl:
        if step >= a.steps:
            break
        cand, P, truth, w = cand.to(dev), P.to(dev), truth.to(dev), w.to(dev)
        B, T = w.shape
        known = w > 0
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ev = model.encode_evidence(cand, P)
            tok = clean_tokens(cand, truth)
            x, flag = corrupt(tok, known, ev["cv"], gen, B, T, dev)
            sc_on = torch.rand(B, device=dev, generator=gen) < 0.5
            with torch.no_grad():
                sc, _ = model(ev, x, flag)
            logits, off = model(ev, x, flag, sc.detach(), sc_on)
        cm = known & (flag == 0)
        wt = w * cm
        ce = F.cross_entropy(logits.float().flatten(0, 1), tok.flatten(), reduction="none").view(B, T)
        l_ce = (ce * wt).sum() / wt.sum().clamp(min=1)
        isg = tok >= K
        cen = model.centres[(tok - K).clamp(0, G - 1)]
        tgt = (torch.nan_to_num(truth) - cen).clamp(-1, 1)
        wo = wt * isg
        l_off = ((off.float() - tgt).abs().sum(-1) * wo).sum() / wo.sum().clamp(min=1)
        loss = l_ce + l_off
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        with torch.no_grad():
            acc = ((logits.argmax(-1) == tok).float() * wt).sum() / wt.sum().clamp(min=1)
        hist.append([l_ce.item(), l_off.item(), acc.item()])
        if step % 200 == 0:
            h = np.mean(hist, 0)
            hist = []
            print("step %d ce %.3f off %.3f acc %.3f lr %.2e %.1f s/100 mem %.2f GB" % (
                step, h[0], h[1], h[2], sched.get_last_lr()[0], (time.time() - t0) / 2,
                torch.cuda.max_memory_allocated() / 1e9), flush=True)
            t0 = time.time()
        if step % 2000 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "args": vars(a)}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            if step % 10000 == 0 or step == a.steps:
                torch.save({"model": model.state_dict(), "step": step, "args": vars(a)},
                           os.path.join(odir, "model_%d.pt" % step))
    print("done", step)


if __name__ == "__main__":
    main()
