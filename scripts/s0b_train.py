#!/usr/bin/env python
"""Stage 0b (prereg.md): train one configuration on cached frozen MAE-Face features, evaluate on val10.

  base        (i)   f_t = CLS_t
  framewise   (ii)  f_t = CLS_t + g * mean_p MLP(LN(R_tp))                       MLP 768-32-768, no temporal mixing
  motionpool  (iii) f_t = CLS_t + g * sum_p softmax_p(score(LN(R_tp) - mean_t LN(R_tp))) LN(R_tp)   block A
  mixer       (iv)  f_t = CLS_t + g * U(GELU(mean_p sum_t' M_tt' D LN(R_t'p)))    block C, exploratory
g is a per-dimension gate initialised to zero, so (ii)-(iv) equal (i) at initialisation.
Head = the MMA-DFER host's: Linear(768, 512) -> Temporal_Transformer_Cls(16, 512, 1 layer, 8 heads, mlp 1024) -> Linear(512, 7).
Optimisation = the group's trainer: AdamW lr 1e-4, wd 0.01 (gates 0), batch 8, 25 epochs, cosine over epochs, CE, last epoch.
"""

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--cache', required=True)
ap.add_argument('--train_split', required=True)
ap.add_argument('--eval_split', required=True)
ap.add_argument('--config', required=True, choices=('base', 'framewise', 'motionpool', 'mixer'))
ap.add_argument('--block', default='b12', choices=('b12', 'b9'))
ap.add_argument('--epochs', type=int, default=25)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--wd', type=float, default=0.01)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.src))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from avdfer.hosts.mma_dfer.Temporal_Model import Temporal_Transformer_Cls  # noqa: E402

DEV = torch.device('cuda')
T, D, NCLS = 16, 768, 7
EXPECTED_PARAMS = {'base': 0, 'framewise': 52256, 'motionpool': 51584, 'mixer': 52512}   # prereg.md


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_proj = nn.Linear(D, 512)
        self.temporal_net = Temporal_Transformer_Cls(num_patches=T, input_dim=512, depth=1, heads=8, mlp_dim=1024,
                                                     dim_head=64)
        self.classifier = nn.Linear(512, NCLS)

    def forward(self, f):
        return self.classifier(self.temporal_net(self.vision_proj(f)))


class Framewise(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, D))
        self.gate = nn.Parameter(torch.zeros(D))

    def forward(self, cls, r, suppress=False):
        return cls + self.gate * self.mlp(self.ln(r)).mean(2)


class MotionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.w1 = nn.Linear(D, 64)
        self.w2 = nn.Linear(64, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(D))

    def weights(self, r, suppress=False):
        z = self.ln(r)
        d = torch.zeros_like(z) if suppress else z - z.mean(1, keepdim=True)
        return z, torch.softmax(self.w2(F.gelu(self.w1(d))).squeeze(-1), dim=-1)      # [B, T, P]

    def forward(self, cls, r, suppress=False):
        z, w = self.weights(r, suppress)
        return cls + self.gate * (w.unsqueeze(-1) * z).sum(2)


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.down = nn.Linear(D, 32)
        self.up = nn.Linear(32, D)
        self.mix = nn.Parameter(torch.eye(T))
        self.gate = nn.Parameter(torch.zeros(D))

    def forward(self, cls, r, suppress=False):
        z = self.down(self.ln(r))                                   # [B, T, P, 32]
        m = torch.einsum('ts,bspd->btpd', self.mix, z).mean(2)      # [B, T, 32]
        return cls + self.gate * self.up(F.gelu(m))


ADAPTERS = {'framewise': Framewise, 'motionpool': MotionPool, 'mixer': Mixer}


def split_ids(path):
    rows = []
    for line in open(path):
        p = line.split()
        if p:
            rows.append((os.path.basename(p[0]), int(p[2])))
    return rows


def uar(pred, y):
    return 100.0 * float(np.mean([(pred[y == c] == c).mean() for c in range(NCLS) if (y == c).any()]))


def spread(f):
    """mean over clips of mean_t ||f_t - fbar|| / ||fbar||."""
    m = f.mean(1, keepdim=True)
    return float(((f - m).norm(dim=-1).mean(1) / m.squeeze(1).norm(dim=-1)).mean())


def main():
    ids = open(os.path.join(a.cache, 'clips.txt')).read().split()
    row = {c: i for i, c in enumerate(ids)}
    labels = np.load(os.path.join(a.cache, 'labels.npy'))
    feats = np.load(os.path.join(a.cache, f'{a.block}.npy'), mmap_mode='r')
    sets = {}
    for name, path in (('train', a.train_split), ('eval', a.eval_split)):
        sp = split_ids(path)
        idx = np.array([row[c] for c, _ in sp])
        lab = labels[idx]
        offset = {int(l - lab_) for (_, l), lab_ in zip(sp, lab)}
        assert len(offset) == 1, f'{name}: split labels and cache labels disagree ({offset})'
        x = torch.from_numpy(np.ascontiguousarray(feats[np.sort(idx)]))   # sorted read, then reorder
        order = np.argsort(np.argsort(idx))
        sets[name] = (x[torch.from_numpy(order)].to(DEV), torch.from_numpy(lab).to(DEV), [c for c, _ in sp])
        print(f'{name}: {len(idx)} clips from {path}; label counts {np.bincount(lab, minlength=NCLS).tolist()}',
              flush=True)

    torch.manual_seed(a.seed)
    head = Head().to(DEV)
    torch.manual_seed(a.seed + 1)
    adapter = ADAPTERS[a.config]().to(DEV) if a.config != 'base' else None
    n_adapter = sum(p.numel() for p in adapter.parameters()) if adapter else 0
    assert n_adapter == EXPECTED_PARAMS[a.config], (a.config, n_adapter)
    params = list(head.parameters()) + (list(adapter.parameters()) if adapter else [])
    gates = [p for n, p in (adapter.named_parameters() if adapter else []) if n == 'gate']
    rest = [p for p in params if all(p is not g for g in gates)]
    opt = torch.optim.AdamW([{'params': rest, 'weight_decay': a.wd}, {'params': gates, 'weight_decay': 0.0}], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()
    xtr, ytr, _ = sets['train']
    xev, yev, ev_ids = sets['eval']
    gen = torch.Generator().manual_seed(a.seed)

    def frame_desc(x, suppress=False):
        x = x.float()
        cls, r = x[:, :, 0], x[:, :, 1:]
        return adapter(cls, r, suppress) if adapter else cls

    @torch.no_grad()
    def evaluate(suppress=False):
        head.eval()
        if adapter:
            adapter.eval()
        logits, fs = [], []
        for i in range(0, len(yev), 64):
            f = frame_desc(xev[i:i + 64], suppress)
            fs.append(f)
            logits.append(head(f))
        head.train()
        if adapter:
            adapter.train()
        lg = torch.cat(logits)
        return lg, torch.cat(fs)

    hist = []
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(ytr), generator=gen).to(DEV)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            loss = crit(head(frame_desc(xtr[b])), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        sched.step()
        lg, _ = evaluate()
        pred = lg.argmax(1).cpu().numpy()
        y = yev.cpu().numpy()
        rec = {'epoch': ep + 1, 'loss': tot / nb, 'uar': uar(pred, y), 'war': 100.0 * float((pred == y).mean()),
               'gate_abs_mean': float(torch.cat([g.detach().abs() for g in gates]).mean()) if gates else 0.0}
        hist.append(rec)
        print(f"ep {ep + 1:2d}  loss {rec['loss']:.4f}  val UAR {rec['uar']:.2f}  WAR {rec['war']:.2f}  "
              f"|g| {rec['gate_abs_mean']:.4f}  {time.time() - t0:.0f}s", flush=True)

    lg, f = evaluate()
    y = yev.cpu().numpy()
    out = {'config': a.config, 'block': a.block, 'seed': a.seed, 'adapter_params': n_adapter,
           'head_params': sum(p.numel() for p in head.parameters()), 'history': hist,
           'last': hist[-1], 'best_epoch': max(hist, key=lambda r: r['uar']),
           'spread_f': spread(f), 'spread_cls': spread(xev[:, :, 0].float()), 'n_eval': int(len(y))}
    arrays = {'labels': y, 'prob': torch.softmax(lg, -1).cpu().numpy(), 'clips': np.array(ev_ids)}
    if a.config == 'motionpool':
        lgs, _ = evaluate(suppress=True)
        ps = lgs.argmax(1).cpu().numpy()
        out['suppress_uar'] = uar(ps, y)
        out['suppress_changed_pct'] = 100.0 * float((ps != lg.argmax(1).cpu().numpy()).mean())
        arrays['prob_suppress'] = torch.softmax(lgs, -1).cpu().numpy()
        with torch.no_grad():
            _, w = adapter.weights(xev[:256].float()[:, :, 1:])
        out['motion_weight_entropy'] = float((-(w * w.clamp_min(1e-12).log()).sum(-1)).mean())
        out['motion_weight_entropy_uniform'] = float(np.log(w.shape[-1]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out + '.json', 'w'), indent=1)
    np.savez_compressed(a.out + '.npz', **arrays)
    print(f"-> {a.out}.json  last UAR {out['last']['uar']:.2f}  spread f {out['spread_f']:.4f} (CLS {out['spread_cls']:.4f})")


if __name__ == '__main__':
    main()
