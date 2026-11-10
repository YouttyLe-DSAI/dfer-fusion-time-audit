#!/usr/bin/env python
"""Stage 0a: within-clip spread of the visual tokens each routing module reads, by spatial granularity.

For every routed block, on the tokens the routing module receives (image_lowdim, 128-d,
the space in which F1's s_v = 2.2% was measured), at these levels:

  cls     the class token (the router's default frame descriptor; reproduces F1's frame_spread)
  mean    the uniform mean of the patch tokens (fusion.video_token=mean)
  r2, r4  adaptive average pooling of the patch grid to 2x2 / 4x4 regions
  patch   every patch on its own (10x10 for MMA-DFER at 160 px, 14x14 for ABVnet at 224 px)

For a level with R regions, x [n, T, R, D]:

  spread       mean_j ||x_jr - xbar_r|| / ||xbar_r||, averaged over r   (F1's definition, per region)
  within_rms   sqrt(mean_{j,r} ||x_jr - xbar_r||^2)
  between_rms  sqrt(mean_r Var_c[xbar_cr]), over the clips of the fold
  ratio        within_rms / between_rms

slot_spread (audio side) is reported as a check against F1's 24%.

Reads the model code, checkpoints and data read-only. Imports the code from --src with
PYTHONDONTWRITEBYTECODE expected set, and writes only --out.

    python stage0a_spread.py --src ~/AVDFEROT/AV-DFER-OT --run <src>/runs/DFEW/dev/scr-<id> \
        --folds 1 2 3 4 5 --out results/stage0a/MMA_DFEW_scr.json
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

LEVELS = ('cls', 'mean', 'r2', 'r4', 'patch')
TINY = 1e-12


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='AV-DFER-OT source tree (read-only)')
    ap.add_argument('--run', required=True)
    ap.add_argument('--folds', nargs='+', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0, help='clips per fold, 0 = all')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--set', nargs='+', action='extend', default=[], metavar='KEY=VALUE')
    ap.add_argument('--out', required=True)
    return ap.parse_args()


args = parse()
sys.path.insert(0, os.path.abspath(os.path.expanduser(args.src)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402
from avdfer.run.config import annotation_paths, apply_overrides  # noqa: E402


def spread(x):
    """x [n, K, D] -> [n]: mean distance of the rows from their mean, relative to the mean's norm."""
    mu = x.mean(1, keepdim=True)
    return (x - mu).norm(dim=-1).mean(1) / mu.squeeze(1).norm(dim=-1).clamp_min(TINY)


def levels(img, n, t, n_image):
    """image_lowdim [n*t, L, D] -> {level: [n, t, R, D]}."""
    x = img.float().view(n, t, img.shape[1], img.shape[2])
    patches = x[:, :, 1:1 + n_image, :]
    g = int(round(math.sqrt(n_image)))
    assert g * g == n_image, n_image
    d = x.shape[-1]
    grid = patches.reshape(n * t, g, g, d).permute(0, 3, 1, 2)          # row-major patch grid
    out = {'cls': x[:, :, :1, :], 'mean': patches.mean(2, keepdim=True), 'patch': patches}
    for k, name in ((2, 'r2'), (4, 'r4')):
        r = F.adaptive_avg_pool2d(grid, k).flatten(2).transpose(1, 2)   # [n*t, k*k, D]
        out[name] = r.reshape(n, t, k * k, d)
    return out


class Acc:
    """Per-block accumulator of per-clip spreads and of the between-clip moments."""

    def __init__(self):
        self.vals = defaultdict(list)       # key -> list of [n] tensors
        self.s1, self.s2, self.cnt = {}, {}, 0

    def add(self, lv, slot_spread, frame_spread):
        n = next(iter(lv.values())).shape[0]
        self.vals['slot_spread'].append(slot_spread.cpu())
        self.vals['frame_spread'].append(frame_spread.cpu())
        for name, x in lv.items():
            # x [n, t, R, D]
            xbar = x.mean(1)                                             # [n, R, D]
            dev = x - xbar.unsqueeze(1)
            per_region = dev.norm(dim=-1).mean(1) / xbar.norm(dim=-1).clamp_min(TINY)   # [n, R]
            self.vals[f'{name}_spread'].append(per_region.mean(1).cpu())
            self.vals[f'{name}_within_sq'].append(dev.pow(2).sum(-1).mean((1, 2)).cpu())
            xb = xbar.double()
            self.s1[name] = self.s1.get(name, 0) + xb.sum(0).cpu()                       # [R, D]
            self.s2[name] = self.s2.get(name, 0) + xb.pow(2).sum(-1).sum(0).cpu()        # [R]
        self.cnt += n

    def result(self):
        row = {}
        for k, v in self.vals.items():
            if k.endswith('_within_sq'):
                continue
            v = torch.cat(v)
            row[k] = float(v.mean())
            row[k + '_std'] = float(v.std()) if len(v) > 1 else 0.0
        for name in LEVELS:
            within = float(torch.cat(self.vals[f'{name}_within_sq']).double().mean().sqrt())
            m1 = self.s1[name] / self.cnt
            var = (self.s2[name] / self.cnt - m1.pow(2).sum(-1)).clamp_min(0)              # [R]
            between = float(var.mean().sqrt())
            row[f'{name}_within_rms'] = within
            row[f'{name}_between_rms'] = between
            row[f'{name}_ratio'] = within / max(between, TINY)
        return row


def measure_fold(cfg, fold_dir, ann, device, workers, limit):
    model = build_host(cfg)
    state = torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu', weights_only=False)['state_dict']
    load_checkpoint_state(model, state)
    model.to(device).eval()
    mods = model.routing_modules
    accs = [Acc() for _ in mods]

    def hook_for(i):
        def hook(mod, inp, out):
            img, aud, n, t, *rest = inp
            with torch.autocast(device_type=img.device.type, enabled=False):
                lv = levels(img, n, t, mod.n_image)
                v = mod.video_tokens(img.float(), n, t)
                a = mod.audio_tokens(aud.float())
                accs[i].add(lv, spread(a.float()), spread(v.float()))
        return hook

    for i, mod in enumerate(mods):
        mod.register_forward_hook(hook_for(i))

    d = cfg['data']
    ds = build_dataset(ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                       audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                       frame_order=d.get('frame_order', 'lexicographic'),
                       audio_channels=d.get('audio_channels', 'first'),
                       frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg['eval']['batch_size'], shuffle=False,
                                         num_workers=workers)
    seen = 0
    with torch.no_grad(), autocast(device, cfg['train'].get('precision', 'fp32')):
        for images, _, audio, _, timing in loader:
            if limit and seen >= limit:
                break
            model(images.to(device), audio.to(device), timing_to(timing, device))
            seen += images.size(0)
    blocks = [{'block': model.fusion_blocks[i], **acc.result()} for i, acc in enumerate(accs)]
    return {'n_clip': accs[0].cnt, 'n_image': mods[0].n_image, 'blocks': blocks}


def table(title, blocks, keys, fmt='{:>9.4f}'):
    print(f'\n{title}\nblock' + ''.join(f'{k:>9}' for k in keys))
    for r in blocks:
        print(f'{r["block"]:>5}' + ''.join(fmt.format(r[f'{k}']) for k in keys))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_fold, n_clip = {}, {}
    for fold in args.folds:
        fold_dir = os.path.join(args.run, f'fold{fold}')
        with open(os.path.join(fold_dir, 'config.json')) as f:
            cfg = apply_overrides(json.load(f), args.set)
        if cfg['fusion']['rule'] == 'none':
            raise SystemExit('the backbone has no routing module')
        ann = annotation_paths(cfg, fold)[1].format(fold=fold)
        res = measure_fold(cfg, fold_dir, ann, device, args.workers, args.limit)
        per_fold[fold], n_clip[fold] = res['blocks'], res['n_clip']
        print(f'fold {fold}: {res["n_clip"]} clips, n_image {res["n_image"]} ({ann})', flush=True)

    first = per_fold[args.folds[0]]
    keys = [k for k in first[0] if k != 'block']
    blocks = [{'block': r['block'], **{k: sum(per_fold[f][i][k] for f in args.folds) / len(args.folds)
                                       for k in keys}} for i, r in enumerate(first)]
    res = {'run': args.run, 'folds': args.folds, 'host': cfg['host']['name'], 'dataset': cfg['data']['dataset'],
           'rule': cfg['fusion']['rule'], 'limit': args.limit, 'n_clip': n_clip, 'blocks': blocks,
           'per_fold': per_fold}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)

    print(f'\n{cfg["host"]["name"]} {cfg["data"]["dataset"]} {cfg["fusion"]["rule"]}, mean over folds {args.folds}')
    table('check (frame_spread must equal cls_spread; slot_spread vs F1 24%)', blocks,
          ['frame_spread', 'cls_spread', 'slot_spread'])
    table('spread = mean_j ||x_j - xbar|| / ||xbar|| per region', blocks, [f'{lv}_spread' for lv in LEVELS])
    table('within-clip / between-clip RMS', blocks, [f'{lv}_ratio' for lv in LEVELS])
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
