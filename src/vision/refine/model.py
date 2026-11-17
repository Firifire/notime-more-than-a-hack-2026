import math

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 8
X0, Y0, CELL, NX, NY = -6.0, -10.0, 2.0, 59, 44
G = NX * NY
V = K + G
MASK = V
NCP = 4


def cell_of(xy):
    ix = ((xy[..., 0] - X0) / CELL).floor().clamp(0, NX - 1).long()
    iy = ((xy[..., 1] - Y0) / CELL).floor().clamp(0, NY - 1).long()
    return ix * NY + iy


def cell_centres(device=None):
    j = torch.arange(G, device=device)
    return torch.stack([X0 + CELL * (j // NY + 0.5), Y0 + CELL * (j % NY + 0.5)], -1).float()


def noise_tokens(cv, gen, q_cand=0.5):
    B, T = cv.shape[:2]
    dev = cv.device
    nc = cv.sum(-1)
    kc = (torch.rand(B, T, device=dev, generator=gen) * nc).floor().long().clamp(max=K - 1)
    cell = K + torch.randint(0, G, (B, T), device=dev, generator=gen)
    use_c = (torch.rand(B, T, device=dev, generator=gen) < q_cand) & (nc > 0)
    return torch.where(use_c, kc, cell)


def nxy(xy):
    return torch.stack([xy[..., 0] / 52.5 - 1.0, xy[..., 1] / 34.0 - 1.0], -1)


class RMSNorm(nn.Module):
    def __init__(self, d, scale=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d)) if scale else None

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w if self.w is not None else x


class SelfConditioning(nn.Module):

    def __init__(self, d, ff):
        super().__init__()
        self.pre_norm = RMSNorm(d)
        self.post_norm = RMSNorm(d, scale=False)
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, emb, soft):
        n = self.pre_norm(soft)
        return self.post_norm(emb + self.down(F.gelu(self.gate(n), approximate="tanh") * self.up(n)))


WAVES = (128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0)
NF = 4 * len(WAVES)
SHIFTS = (-4, -2, -1, 1, 2, 4)
HSHIFTS = (-3, -1, 1, 3)


def fourier(xy):
    w = torch.tensor([2 * math.pi / l for l in WAVES], device=xy.device, dtype=torch.float32)
    a = xy.float()[..., None] * w
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def shift_t(x, s, fill):
    y = torch.full_like(x, fill)
    T = x.shape[1]
    if s > 0:
        y[:, :T - s] = x[:, s:]
    elif s < 0:
        y[:, -s:] = x[:, :T + s]
    else:
        y = x.clone()
    return y


def mlp(i, d, o):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))


class TrajDiffusion(nn.Module):
    def __init__(self, d=192, layers=6, heads=6, ff=768, max_len=512, dropout=0.1, tied=True, ce_dim=0, fe_dim=0):
        super().__init__()
        self.d = d
        self.ce_dim, self.fe_dim = ce_dim, fe_dim
        self.cand_enc = mlp(8 + NF + 2 * len(SHIFTS) + ce_dim, d, d)
        if fe_dim:
            self.fe_enc = mlp(fe_dim, d, d)
        self.cand_pool = nn.Linear(2 * d, d)
        self.pl_enc = mlp(6 + NF, d, d)
        self.pl_pool = nn.Linear(2 * d, d)
        self.ev_norm = nn.LayerNorm(d)
        self.cand_tok = nn.Linear(d, d)
        self.cell_emb = nn.Embedding(G, d)
        self.cell_pos = mlp(NF, d, d)
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.null_emb = nn.Parameter(torch.zeros(d))
        self.geo = mlp(K + 3 * NCP + 2 + NF + len(HSHIFTS), d, d)
        self.sc = SelfConditioning(d, ff)
        self.flag = nn.Embedding(3, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.tied = tied
        if tied:
            self.cq = nn.Linear(d, d)
            self.ck = nn.Linear(d, d)
        else:
            self.cell_head = nn.Linear(d, G)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.off_head = nn.Linear(d, 2)
        self.register_buffer("centres", cell_centres(), persistent=False)

    def encode_evidence(self, cand, players, cand_extra=None, frame_extra=None):
        cv = ~torch.isnan(cand[..., 2])
        cxy = torch.nan_to_num(cand[..., :2])
        pv = ~torch.isnan(players[..., 0])
        pxy = torch.nan_to_num(players[..., :2])
        pvel = torch.nan_to_num(players[..., 2:4])
        d = torch.cdist(cxy.flatten(0, 1), pxy.flatten(0, 1)).view(*cxy.shape[:3], pxy.shape[2])
        d = d.masked_fill(~pv[:, :, None, :], 1e3)
        dn, jn = d.min(-1)
        spd = torch.gather(pvel.norm(dim=-1), 2, jn.clamp(min=0)).masked_fill(dn > 999, 0)
        sup = []
        conf = torch.nan_to_num(cand[..., 2])
        for s_ in SHIFTS:
            o = shift_t(cxy, s_, 0.0)
            ov = shift_t(cv, s_, False)
            dd = (cxy[:, :, :, None] - o[:, :, None]).norm(dim=-1).masked_fill(~ov[:, :, None, :], 1e3)
            dm, jm = dd.min(-1)
            oc = torch.gather(shift_t(conf, s_, 0.0), 2, jm)
            sup += [(dm.clamp(max=10) / 5)[..., None], (oc * (dm < 999))[..., None]]
        cf = torch.cat([fourier(cxy), *sup, nxy(cxy), torch.nan_to_num(cand[..., 2:3]), cv[..., None].float(),
                        (dn.clamp(max=30) / 10)[..., None], (spd / 10).clamp(max=3)[..., None],
                        (torch.arange(K, device=cand.device).float() / K).expand(cv.shape)[..., None],
                        torch.ones_like(dn)[..., None]], -1)
        if self.ce_dim:
            cf = torch.cat([cf, torch.nan_to_num(cand_extra.float())], -1)
        c = self.cand_enc(cf) * cv[..., None]
        cm = c.masked_fill(~cv[..., None], -1e4).max(2).values.masked_fill(~cv.any(2)[..., None], 0)
        ca = c.sum(2) / cv.sum(2, keepdim=True).clamp(min=1)
        pf = torch.cat([fourier(pxy), nxy(pxy), pvel / 10, pv[..., None].float(),
                        (pvel.abs().sum(-1, keepdim=True) > 0).float()], -1)
        p = self.pl_enc(pf) * pv[..., None]
        pm = p.masked_fill(~pv[..., None], -1e4).max(2).values.masked_fill(~pv.any(2)[..., None], 0)
        pa = p.sum(2) / pv.sum(2, keepdim=True).clamp(min=1)
        e = self.cand_pool(torch.cat([cm, ca], -1)) + self.pl_pool(torch.cat([pm, pa], -1))
        if self.fe_dim:
            e = e + self.fe_enc(torch.nan_to_num(frame_extra.float()))
        e = self.ev_norm(e)
        return {"e": e, "c": c, "cv": cv, "cxy": cxy, "pv": pv, "pxy": pxy}

    def token_xy(self, ev, tok, off=None):
        isc = tok < K
        kc = tok.clamp(max=K - 1)
        cxy = torch.gather(ev["cxy"], 2, kc[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
        g = (tok - K).clamp(0, G - 1)
        gxy = self.centres[g]
        if off is not None:
            gxy = gxy + off
        return torch.where(isc[..., None], cxy, gxy)

    def vocab_emb(self, ev):
        ct = self.cand_tok(ev["c"])
        ct = torch.where(ev["cv"][..., None], ct, self.null_emb)
        return ct, self.cell_emb.weight + self.cell_pos(fourier(self.centres))

    def forward(self, ev, tok, flag, sc_logits=None, sc_on=None):
        B, T = tok.shape
        ct, ce = self.vocab_emb(ev)
        tk = tok.clamp(max=V - 1)
        isc = tk < K
        emb_c = torch.gather(ct, 2, tk.clamp(max=K - 1)[..., None, None].expand(-1, -1, 1, self.d))[:, :, 0]
        emb_g = ce[(tk - K).clamp(0, G - 1)]
        emb = torch.where(isc[..., None], emb_c, emb_g)
        emb = torch.where((tok == MASK)[..., None], self.mask_emb.expand_as(emb), emb)
        xy = self.token_xy(ev, tk)
        dc = (xy[:, :, None] - ev["cxy"]).norm(dim=-1).masked_fill(~ev["cv"], 30.0)
        dp = (xy[:, :, None] - ev["pxy"]).norm(dim=-1).masked_fill(~ev["pv"], 1e3)
        dps, jp = dp.topk(NCP, dim=-1, largest=False)
        rel = (torch.gather(ev["pxy"], 2, jp[..., None].expand(-1, -1, -1, 2)) - xy[:, :, None]).clamp(-30, 30)
        present = tok != MASK
        hs = []
        for s_ in HSHIFTS:
            dn_ = (xy - shift_t(xy, s_, 0.0)).norm(dim=-1)
            ok_ = present & shift_t(present, s_, False)
            hs.append(torch.where(ok_, dn_.clamp(max=20), torch.full_like(dn_, 20.0))[..., None] / 5)
        geo = torch.cat([(dc.clamp(max=30) / 10), (dps.clamp(max=30) / 10), rel.flatten(2) / 10, nxy(xy),
                         fourier(xy), *hs], -1)
        emb = emb + self.geo(geo) * (tok != MASK)[..., None]
        if sc_logits is None:
            soft = torch.zeros_like(emb)
        else:
            p = sc_logits.float().softmax(-1)
            soft = (p[..., :K, None] * ct).sum(2) + p[..., K:] @ ce
            if sc_on is not None:
                soft = soft * sc_on[:, None, None].float()
        x = self.sc(emb, soft) + ev["e"] + self.flag(flag) + self.pos.weight[:T]
        h = self.out_norm(self.tf(x))
        if self.tied:
            lc = self.cq(h) @ self.ck(ce).T / math.sqrt(self.d)
        else:
            lc = self.cell_head(h)
        lk = torch.einsum("btd,btkd->btk", self.q(h), self.k(ev["c"])) / math.sqrt(self.d)
        lk = lk.masked_fill(~ev["cv"], -1e4)
        off = torch.tanh(self.off_head(h)) * (CELL / 2)
        return torch.cat([lk, lc], -1), off
