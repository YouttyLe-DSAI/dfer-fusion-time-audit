#!/usr/bin/env python
"""P6 stage 0 (prereg.md, 2026-11-11): positive control for within-clip variation, CREMA-D composite clips.

For each anchor clip i of the test fold, three inputs to the router A0 (audio and timing are i's in all three):
  O  the clip itself
  S  frames 1-8 of i + frames 9-16 of j: same actor, different sentence, SAME emotion
  D  as S with a DIFFERENT emotion
Partners are drawn uniformly among the candidates (seed 0 per fold); CREMA-D folds are actor-disjoint. At every routing
block, on what the cost reads: cost-space spread, raw spread, sigma_v, kappa, beta, and the boundary gap
||mean_{j<=8} v_cs - mean_{j>8} v_cs|| / ||mean v_cs||. Reads the code read-only; writes only --out.json/.npz.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--run', required=True)
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(os.path.expanduser(a.src)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402

DEV = torch.device('cuda')
TINY = 1e-12
CONDS = ('O', 'S', 'D')
KEYS = ('cs_spread', 'raw_spread', 'sigma_v', 'kappa', 'beta', 'gap_cs', 'gap_raw')


def dev_stats(x, w=None):
    if w is None:
        w = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
    ws = w.sum(1).clamp_min(TINY)
    mu = (x * w.unsqueeze(-1)).sum(1) / ws.unsqueeze(-1)
    d = (x - mu.unsqueeze(1)).norm(dim=-1)
    return (d * w).sum(1) / ws / mu.norm(dim=-1).clamp_min(TINY), ((d.pow(2) * w).sum(1) / ws).sqrt()


def gap(x):
    h = x.shape[1] // 2
    return (x[:, :h].mean(1) - x[:, h:].mean(1)).norm(dim=-1) / x.mean(1).norm(dim=-1).clamp_min(TINY)


class Trip(torch.utils.data.Dataset):
    def __init__(self, ds, pS, pD):
        self.ds, self.pS, self.pD = ds, pS, pD

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, k):
        img, label, fbank, index, timing = self.ds[k]
        imS = self.ds[self.pS[k]][0] if self.pS[k] >= 0 else torch.zeros_like(img)
        imD = self.ds[self.pD[k]][0] if self.pD[k] >= 0 else torch.zeros_like(img)
        return img, imS, imD, label, fbank, timing, k


def main():
    fold_dir = os.path.join(a.run, f'fold{a.fold}')
    cfg = json.load(open(os.path.join(fold_dir, 'config.json')))
    assert cfg['data']['dataset'] == 'CREMAD'
    model = build_host(cfg)
    load_checkpoint_state(model, torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu',
                                            weights_only=False)['state_dict'])
    model.to(DEV).eval()
    mods = model.routing_modules
    d = cfg['data']
    ann = os.path.join(os.path.expanduser(a.src), 'annotation', f'CREMAD_set_{a.fold}_test.txt')
    ds = build_dataset(ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                       audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                       frame_order=d.get('frame_order', 'lexicographic'),
                       audio_channels=d.get('audio_channels', 'first'),
                       frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    meta = []
    for c in ds.clips:
        actor, sent, emo = os.path.basename(c.frames_dir).split('_')[:3]
        meta.append((actor, sent, int(c.label)))
    rng = np.random.default_rng(0)
    pS, pD = [], []
    for k, (ac, se, la) in enumerate(meta):
        cs = [j for j, (a2, s2, l2) in enumerate(meta) if a2 == ac and s2 != se and l2 == la]
        cd = [j for j, (a2, s2, l2) in enumerate(meta) if a2 == ac and s2 != se and l2 != la]
        pS.append(int(rng.choice(cs)) if cs else -1)
        pD.append(int(rng.choice(cd)) if cd else -1)
    loader = torch.utils.data.DataLoader(Trip(ds, pS, pD), batch_size=cfg['eval']['batch_size'], shuffle=False,
                                         num_workers=a.workers)
    cur = {'cond': None}
    vals = [{c: defaultdict(list) for c in CONDS} for _ in mods]

    def hook_for(i):
        def hook(mod, inp, out):
            img, aud, n, t, *rest = inp
            timing = rest[0] if rest else None
            with torch.autocast(device_type='cuda', enabled=False):
                if timing is not None:
                    timing = {k: (x.float() if x.is_floating_point() else x) for k, x in timing.items()}
                v, au = mod.video_tokens(img.float(), n, t), mod.audio_tokens(aud.float())
                v_cs = F.normalize(mod.norm_v(v), dim=-1)
                a_cs = F.normalize(mod.norm_a(au), dim=-1)
                valid = timing['slot_valid'] if (timing is not None and mod.mask_padding) else torch.ones(n, au.shape[1], dtype=torch.bool, device=DEV)
                u = (valid | ~valid.any(1, keepdim=True)).float()
                cs, sv = dev_stats(v_cs)
                rs, _ = dev_stats(v)
                _, sa = dev_stats(a_cs, u)
                C = mod.cost(v, au)
                w = u.unsqueeze(1)
                ns = w.sum(2, keepdim=True)
                row = (C * w).sum(2, keepdim=True) / ns
                col = (C * w).mean(1, keepdim=True)
                m = row.mean(1, keepdim=True)
                kap = (((C - row - col + m) * w).pow(2).sum((1, 2)) / (ns.flatten() * C.shape[1])).sqrt() / mod.eps
                rec = {'cs_spread': cs, 'raw_spread': rs, 'sigma_v': sv, 'kappa': kap, 'beta': sv * sa / mod.eps,
                       'gap_cs': gap(v_cs), 'gap_raw': gap(v)}
                for k2, x in rec.items():
                    vals[i][cur['cond']][k2].append(x.cpu())
        return hook

    for i, m in enumerate(mods):
        m.register_forward_hook(hook_for(i))
    keep = []
    with torch.no_grad(), autocast(DEV, cfg['train'].get('precision', 'fp32')):
        for it, (img, imS, imD, label, fbank, timing, k) in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            tm = timing_to(timing, DEV)
            fb = fbank.to(DEV)
            h = img.shape[1] // 2
            for cond, x in (('O', img), ('S', torch.cat([img[:, :h], imS[:, h:]], 1)),
                            ('D', torch.cat([img[:, :h], imD[:, h:]], 1))):
                cur['cond'] = cond
                model(x.to(DEV), fb, tm)
            keep.append(torch.tensor([(pS[int(j)] >= 0) and (pD[int(j)] >= 0) for j in k]))
    keep = torch.cat(keep)
    res = {'run': a.run, 'fold': a.fold, 'n_anchor': int(len(keep)), 'n_paired': int(keep.sum()),
           'n_no_S': int(sum(p < 0 for p in pS)), 'n_no_D': int(sum(p < 0 for p in pD)), 'blocks': []}
    npz = {}
    for i, b in enumerate(model.fusion_blocks):
        row = {'block': b}
        for c in CONDS:
            for k2 in KEYS:
                x = torch.cat(vals[i][c][k2])[keep].double()
                row[f'{c}_{k2}'] = float(x.mean())
                if k2 in ('cs_spread', 'gap_cs', 'kappa'):
                    npz[f'b{b}_{c}_{k2}'] = x.float().numpy()
        for k2 in KEYS:
            dd = torch.cat(vals[i]['D'][k2])[keep].double() - torch.cat(vals[i]['S'][k2])[keep].double()
            row[f'DminusS_{k2}'] = float(dd.mean())
        res['blocks'].append(row)
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    np.savez_compressed(a.out + '.npz', **npz)
    print(f'fold {a.fold}: {res["n_paired"]} paired anchors of {res["n_anchor"]}')
    print('block   O_cs     S_cs     D_cs   D-S_cs   O_gap    S_gap    D_gap   D-S_kappa')
    for r in res['blocks']:
        print(f'{r["block"]:>5} {r["O_cs_spread"]:.4f} {r["S_cs_spread"]:.4f} {r["D_cs_spread"]:.4f} '
              f'{r["DminusS_cs_spread"]:+.4f} {r["O_gap_cs"]:.4f} {r["S_gap_cs"]:.4f} {r["D_gap_cs"]:.4f} '
              f'{r["DminusS_kappa"]:+.4f}')


if __name__ == '__main__':
    main()
