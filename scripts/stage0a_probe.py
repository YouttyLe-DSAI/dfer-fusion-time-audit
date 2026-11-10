#!/usr/bin/env python
"""Stage 0a-bis: linear probes on static vs within-clip (dynamic) features, by spatial level.

For each fold of a trained run, extracts from the routing module's input (image_lowdim, 128-d)
at the chosen blocks, for each level (cls, mean, r4 = 4x4 regions, patch):

  S   clip mean xbar                                  [R*D]      static only
  D   residual x_j - xbar averaged over 4 segments    [4*R*D]    within-clip only (zero-sum)
  SD  S and D together

Each feature type is reduced to --pca dims by PCA fitted on train90 (SD = both reductions
concatenated), z-scored, and probed by L2 multinomial logistic regression (full-batch LBFGS,
class-weighted). lambda is picked on a fixed 10% hold-out of train90, then the probe is refit
on all of train90 and scored on val10 (UAR). Pre-registration: prereg.md, stage 0a-bis.

Reads code, checkpoints and data read-only; writes only --out.
"""

import argparse
import json
import math
import os
import sys
import time

LEVELS = ('cls', 'mean', 'r4', 'patch')
FEATS = ('S', 'D', 'SD')
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SEG = 4


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--folds', nargs='+', type=int, required=True)
    ap.add_argument('--blocks', nargs='+', type=int, default=[6, 9, 12])
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--limit', type=int, default=0, help='clips per split, 0 = all (smoke tests only)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--set', nargs='+', action='extend', default=[], metavar='KEY=VALUE')
    ap.add_argument('--out', required=True)
    return ap.parse_args()


args = parse()
sys.path.insert(0, os.path.abspath(os.path.expanduser(args.src)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402
from avdfer.run.config import annotation_paths, apply_overrides  # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def levels(img, n, t, n_image):
    """image_lowdim [n*t, L, D] -> {level: [n, t, R, D]} (float32)."""
    x = img.float().view(n, t, img.shape[1], img.shape[2])
    patches = x[:, :, 1:1 + n_image, :]
    g = int(round(math.sqrt(n_image)))
    d = x.shape[-1]
    grid = patches.reshape(n * t, g, g, d).permute(0, 3, 1, 2)
    r4 = F.adaptive_avg_pool2d(grid, 4).flatten(2).transpose(1, 2).reshape(n, t, 16, d)
    return {'cls': x[:, :, :1, :], 'mean': patches.mean(2, keepdim=True), 'r4': r4, 'patch': patches}


def clip_features(x):
    """[n, t, R, D] -> S [n, R*D], D [n, SEG*R*D]."""
    n, t, r, d = x.shape
    xbar = x.mean(1)
    seg = (x - xbar.unsqueeze(1)).view(n, SEG, t // SEG, r, d).mean(2)
    return xbar.reshape(n, -1), seg.reshape(n, -1)


def extract(model, cfg, ann, block_idx, limit):
    """{(block, level, 'S'|'D'): fp16 CPU [N, F]}, labels [N]."""
    mods = model.routing_modules
    buf = {}
    handles = []

    def hook_for(block, mod):
        def hook(m, inp, out):
            img, _, n, t, *_ = inp
            with torch.autocast(device_type=img.device.type, enabled=False):
                for lv, x in levels(img, n, t, mod.n_image).items():
                    s, dyn = clip_features(x)
                    buf[(block, lv, 'S')] = s.half().cpu()
                    buf[(block, lv, 'D')] = dyn.half().cpu()
        return hook

    for block, i in block_idx.items():
        handles.append(mods[i].register_forward_hook(hook_for(block, mods[i])))

    d = cfg['data']
    ds = build_dataset(ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                       audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                       frame_order=d.get('frame_order', 'lexicographic'),
                       audio_channels=d.get('audio_channels', 'first'),
                       frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg['eval']['batch_size'], shuffle=False,
                                         num_workers=args.workers)
    chunks, labels, seen = {}, [], 0
    with torch.no_grad(), autocast(DEV, cfg['train'].get('precision', 'fp32')):
        for images, label, audio, _, timing in loader:
            if limit and seen >= limit:
                break
            model(images.to(DEV), audio.to(DEV), timing_to(timing, DEV))
            for k, v in buf.items():
                chunks.setdefault(k, []).append(v)
            labels.extend(int(x) for x in label)      # plain ints: keeping worker tensors leaks fds
            seen += images.size(0)
    for h in handles:
        h.remove()
    return {k: torch.cat(v) for k, v in chunks.items()}, torch.tensor(labels)


def pca(xtr, xva, k):
    """Centre on train, project both on the top-k train principal axes (all axes if F <= k)."""
    xtr = xtr.to(DEV).float()
    mu = xtr.mean(0, keepdim=True)
    xtr = xtr - mu
    xva = xva.to(DEV).float() - mu
    if xtr.shape[1] <= k:
        return xtr, xva
    torch.manual_seed(0)
    _, _, v = torch.pca_lowrank(xtr, q=min(k, xtr.shape[0]), center=False, niter=4)
    return xtr @ v, xva @ v


def standardize(ztr, zva):
    m, s = ztr.mean(0, keepdim=True), ztr.std(0, keepdim=True).clamp_min(1e-6)
    return (ztr - m) / s, (zva - m) / s


def fit(z, y, n_cls, lam, w_cls):
    w = torch.zeros(z.shape[1], n_cls, device=DEV, requires_grad=True)
    b = torch.zeros(n_cls, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=300, history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z @ w + b, y, weight=w_cls) + lam * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def uar(pred, y, n_cls):
    rec = [(pred[y == c] == c).float().mean().item() for c in range(n_cls) if (y == c).any()]
    return 100.0 * sum(rec) / len(rec)


def probe(ztr, ytr, zva, yva, n_cls, hold):
    """Pick lambda on the hold-out of train, refit on all train, score val."""
    w_cls = (len(ytr) / (n_cls * torch.bincount(ytr, minlength=n_cls).clamp_min(1).float())).to(DEV)
    keep = torch.ones(len(ytr), dtype=torch.bool)
    keep[hold] = False
    a, h = standardize(ztr[keep], ztr[~keep])
    scores = {}
    for lam in LAMBDAS:
        w, b = fit(a, ytr[keep], n_cls, lam, w_cls)
        scores[lam] = uar((h @ w + b).argmax(1), ytr[~keep], n_cls)
    best = max(LAMBDAS, key=lambda l: (scores[l], l))      # ties -> stronger regularisation
    a, v = standardize(ztr, zva)
    w, b = fit(a, ytr, n_cls, best, w_cls)
    pred = (v @ w + b).argmax(1)
    return {'uar': uar(pred, yva, n_cls), 'war': 100.0 * (pred == yva).float().mean().item(),
            'lambda': best, 'holdout_uar': scores, 'pred': pred.cpu().tolist()}


def run_fold(fold):
    fold_dir = os.path.join(args.run, f'fold{fold}')
    with open(os.path.join(fold_dir, 'config.json')) as f:
        cfg = apply_overrides(json.load(f), args.set)
    tr_ann, va_ann = (p.format(fold=fold) for p in annotation_paths(cfg, fold))
    model = build_host(cfg)
    state = torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu', weights_only=False)['state_dict']
    load_checkpoint_state(model, state)
    model.to(DEV).eval()
    block_idx = {b: model.fusion_blocks.index(b) for b in args.blocks}

    t0 = time.time()
    ftr, ytr_raw = extract(model, cfg, tr_ann, block_idx, args.limit)
    fva, yva_raw = extract(model, cfg, va_ann, block_idx, args.limit)
    print(f'fold {fold}: extracted train {len(ytr_raw)} / val {len(yva_raw)} clips in {time.time() - t0:.0f}s',
          flush=True)
    del model
    torch.cuda.empty_cache()

    classes = sorted(set(ytr_raw.tolist()) | set(yva_raw.tolist()))
    cmap = {c: i for i, c in enumerate(classes)}
    ytr = torch.tensor([cmap[c] for c in ytr_raw.tolist()], device=DEV)
    yva = torch.tensor([cmap[c] for c in yva_raw.tolist()], device=DEV)
    n_cls = len(classes)
    g = torch.Generator().manual_seed(0)
    hold = torch.randperm(len(ytr), generator=g)[:max(1, len(ytr) // 10)]

    probes = {}
    for block in args.blocks:
        for lv in LEVELS:
            z = {f: pca(ftr[(block, lv, f)], fva[(block, lv, f)], args.pca) for f in ('S', 'D')}
            z['SD'] = (torch.cat([z['S'][0], z['D'][0]], 1), torch.cat([z['S'][1], z['D'][1]], 1))
            for f in FEATS:
                res = probe(z[f][0], ytr, z[f][1], yva, n_cls, hold)
                probes[f'{block}/{lv}/{f}'] = res
                print(f'  block {block:>2} {lv:>5} {f:>2}: UAR {res["uar"]:6.2f}  WAR {res["war"]:6.2f}  '
                      f'lambda {res["lambda"]:g}', flush=True)
            del z
            torch.cuda.empty_cache()
    return {'n_train': len(ytr), 'n_val': len(yva), 'classes': classes, 'val_labels': yva.cpu().tolist(),
            'train_ann': tr_ann, 'val_ann': va_ann, 'probes': probes}


def main():
    out = {'run': args.run, 'folds': args.folds, 'blocks': args.blocks, 'levels': LEVELS, 'feats': FEATS,
           'pca': args.pca, 'lambdas': LAMBDAS, 'segments': SEG, 'limit': args.limit, 'per_fold': {}}
    if os.path.exists(args.out):                       # resume: keep finished folds
        with open(args.out) as f:
            prev = json.load(f)
        if prev.get('limit') == args.limit and prev.get('blocks') == args.blocks:
            out['per_fold'] = prev['per_fold']
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for fold in args.folds:
        if str(fold) in out['per_fold']:
            print(f'fold {fold}: already done, skipped', flush=True)
            continue
        out['per_fold'][str(fold)] = run_fold(fold)
        with open(args.out + '.tmp', 'w') as f:
            json.dump(out, f)
        os.replace(args.out + '.tmp', args.out)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
