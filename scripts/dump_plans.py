#!/usr/bin/env python
"""Plan dump for the paper's figures 2-3 (prereg.md, 2026-11-11). Inference only, one fold.

At every routing block: pi (the run's own plan), C (the cost), the usable slots. Per clip, the residual magnitude at
block 12, rho = ||D||_F / ||pi||_F with D = pi - r c^T / S. Declared selection: the clips at the 10th, 50th and 90th
percentile of rho over the fold; their pi, C, D and C_int (double-centred over usable slots) are saved for all blocks,
with the per-block distribution of rho over all clips. Reads the code read-only; writes only --out.npz/.json.
"""

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--run', required=True)
ap.add_argument('--fold', type=int, default=1)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--set', nargs='+', action='extend', default=[])
ap.add_argument('--out', required=True)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(os.path.expanduser(a.src)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402
from avdfer.run.config import apply_overrides  # noqa: E402

DEV = torch.device('cuda')
TINY = 1e-12


def usable_slots(mod, timing, n):
    if timing is None or not mod.mask_padding:
        return torch.ones(n, mod.n_audio_t, dtype=torch.bool, device=DEV)
    valid = timing['slot_valid']
    return valid | ~valid.any(1, keepdim=True)


def main():
    fold_dir = os.path.join(a.run, f'fold{a.fold}')
    cfg = apply_overrides(json.load(open(os.path.join(fold_dir, 'config.json'))), a.set)
    model = build_host(cfg)
    load_checkpoint_state(model, torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu',
                                            weights_only=False)['state_dict'])
    model.to(DEV).eval()
    mods = model.routing_modules
    store = [{'pi': [], 'C': [], 'u': []} for _ in mods]

    def hook_for(i):
        def hook(mod, inp, out):
            img, aud, n, t, *rest = inp
            timing = rest[0] if rest else None
            with torch.autocast(device_type='cuda', enabled=False):
                if timing is not None:
                    timing = {k: (x.float() if x.is_floating_point() else x) for k, x in timing.items()}
                v, au = mod.video_tokens(img.float(), n, t), mod.audio_tokens(aud.float())
                pi, _ = mod.transport_plan(v, au, timing)
                store[i]['pi'].append(pi.float().cpu())
                store[i]['C'].append(mod.cost(v, au).float().cpu())
                store[i]['u'].append(usable_slots(mod, timing, n).cpu())
        return hook

    for i, m in enumerate(mods):
        m.register_forward_hook(hook_for(i))
    d = cfg['data']
    suffix = '_faces' if d['dataset'] == 'MAFW' else ''
    ann = os.path.join(os.path.expanduser(a.src), d.get('annotation_dir', 'annotation'),
                       f"{d['dataset']}_set_{a.fold}_test{suffix}.txt")
    ds = build_dataset(ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                       audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                       frame_order=d.get('frame_order', 'lexicographic'),
                       audio_channels=d.get('audio_channels', 'first'),
                       frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg['eval']['batch_size'], shuffle=False, num_workers=a.workers)
    labels, idx = [], []
    with torch.no_grad(), autocast(DEV, cfg['train'].get('precision', 'fp32')):
        for it, (images, label, audio, index, timing) in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            model(images.to(DEV), audio.to(DEV), timing_to(timing, DEV))
            labels.append(label)
            idx.append(index)
    labels, idx = torch.cat(labels).numpy(), torch.cat(idx).numpy()
    PI = torch.stack([torch.cat(s['pi']) for s in store], 1)            # [N, B, T, M]
    C = torch.stack([torch.cat(s['C']) for s in store], 1)
    U = torch.stack([torch.cat(s['u']) for s in store], 1)              # [N, B, M]
    r = PI.sum(3, keepdim=True)
    c = PI.sum(2, keepdim=True)
    S = PI.sum((2, 3), keepdim=True)
    Dm = PI - r * c / S.clamp_min(TINY)
    rho = Dm.flatten(2).norm(dim=2) / PI.flatten(2).norm(dim=2).clamp_min(TINY)      # [N, B]
    w = U.float().unsqueeze(2)                                                         # [N, B, 1, M]
    ns = w.sum(3, keepdim=True)
    row = (C * w).sum(3, keepdim=True) / ns
    col = (C * w).mean(2, keepdim=True)
    m = row.mean(2, keepdim=True)
    Cint = (C - row - col + m) * w
    q = torch.quantile(rho[:, -1], torch.tensor([0.1, 0.5, 0.9]))
    sel = [int((rho[:, -1] - v).abs().argmin()) for v in q]
    names = [os.path.basename(ds.clips[int(idx[k])].frames_dir) for k in sel]
    np.savez_compressed(a.out + '.npz', names=np.array(names), labels=labels[sel], rho_sel=rho[sel].numpy(),
                        pi=PI[sel].numpy(), C=C[sel].numpy(), D=Dm[sel].numpy(), Cint=Cint[sel].numpy(),
                        usable=U[sel].numpy(), blocks=np.array(model.fusion_blocks), rho_all=rho.numpy().astype(np.float32))
    res = {'run': a.run, 'fold': a.fold, 'n_clips': int(len(labels)), 'selection': 'p10/p50/p90 of rho at block 12',
           'selected': [{'clip': n, 'label': int(labels[k]), 'rho_block12': float(rho[k, -1])} for n, k in zip(names, sel)],
           'rho_per_block': [{'block': b, 'p10': float(torch.quantile(rho[:, i], 0.1)),
                              'median': float(rho[:, i].median()), 'p90': float(torch.quantile(rho[:, i], 0.9))}
                             for i, b in enumerate(model.fusion_blocks)]}
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
